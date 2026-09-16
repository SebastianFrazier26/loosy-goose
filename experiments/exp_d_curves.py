"""Phase 2 / Experiment D: the unified rate-distortion runner.

Puts every method — baselines, Experiment A (as a labeller falsification check), Experiment B
selection strategies, and the Experiment C stacked variants — on one rate-distortion curve per
transcript, all competing at the identical token budget. This is the experiment that decides
whether the eigen-machinery stays in the design (see docs/PHASE1.md, "Honest overall read").

Usage: uv run python experiments/exp_d_curves.py [--public-sample 8] [--swe-gym-sample 5]
                                                  [--only NAME_SUBSTR] [--limit N] [--force]
Per transcript writes experiments/output/exp_d/<label>.json (checkpoint, skipped on rerun unless
--force) and <label>.png, plus experiments/output/exp_d/summary.txt aggregated across whatever
has been checkpointed so far. Local sessions are private: only aggregate numbers and the
withheld local_5k..local_200k labels leave the JSON, matching Phase 1's withholding pattern.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import matplotlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from loosy_goose import embed, metrics
from loosy_goose.budget import ProtectPolicy, select_with_budget, token_budget
from loosy_goose.code import TRANSFORMS_VERSION, quantize_segment
from loosy_goose.paths import PATHS_VERSION, PathTable, build_table, substitute
from loosy_goose.segment import ATOMS_VERSION, Segment, segment
from loosy_goose.select import CompressConfig, Strategy, embed_segments, score_segments
from loosy_goose.supersede import apply_supersession
from loosy_goose.tokens import count_tokens
from loosy_goose.topics import compress as topics_compress
from loosy_goose.transcript import Transcript, load_claude_code_jsonl, load_messages_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Run directly as a script (not `python -m`), so the repo root is not on sys.path by default;
# add it before importing the sibling experiments module, same trick tests/ uses.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.baselines import random_drop, recency_only  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "experiments" / "output" / "exp_d"
CACHE = ROOT / "experiments" / "output" / "cache"

KEEP_RATIOS: tuple[float, ...] = (
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)
# The deliverable operating points: this project buys token reduction on purpose, so aggressive
# rates (~2x/3x/4x compression) are the interesting ones, not r=0.8/0.9. Not required to be grid
# points — the achieved-rate curves are interpolated continuously, so any target rate works.
AGGRESSIVE_RATES: tuple[float, ...] = (0.5, 0.33, 0.25)
# Floor for the knee estimate (see _knee()): the achieved-rate threshold below which
# atom_recall_final is judged to have "degraded sharply". A simple floor-crossing choice, stated
# here rather than computed from curvature, because the sweep's rate grid is coarse enough that a
# numerical second-derivative estimate would be noise-dominated.
KNEE_FLOOR = 0.90
CODE_KINDS = ("code", "tool_use")
# Excludes supersede+tfidf: that stack is TF-IDF plus lossless dedupe, not a spectral method, so
# it is not part of the head-to-head this experiment exists to answer.
SPECTRAL_METHODS = (
    "leverage",
    "ridge",
    "cur_residual",
    "topics",
    "supersede+leverage",
    "band+leverage",
    "supersede+band+leverage",
)
PROTECT_CODE_ONLY_METHODS = ("tfidf", "leverage")
# Methods that put segment text through `code.quantize_segment`, and so depend on the shrinker's
# behaviour rather than only on the segmenter's output. Their records carry TRANSFORMS_VERSION,
# so rewriting the shrinker recomputes them and leaves the other nine methods' records valid.
# The list is hand-written, which is a real risk — `test_transforming_methods_lists_every_method_
# that_shrinks` detects the true set by observation and fails if this drifts from it.
TRANSFORMING_METHODS = ("band+leverage", "supersede+band+leverage")
# Version of the runner's own transform: `_band` hands quantize_segment the sweep's keep
# ratio as its fidelity knob, so trimming depth follows the global ratio. That mapping lives
# here and not in code.py, so changing it moves neither segments_digest nor
# TRANSFORMS_VERSION — stored band records would keep loading as valid while describing a
# depth policy that no longer exists. Bump on any change to what `_band` passes as `quality`.
BAND_QUALITY_VERSION = 1
# Methods whose selection runs through CompressConfig, so a scoring knob such as drop_top can
# change their behaviour. The band+ stacks are deliberately excluded from the drop_top variants:
# they are the slowest methods in the sweep and banding is orthogonal to which directions the
# scorer ignores, so paying for that cross now would buy an interaction we have no reason to
# expect. See docs/PHASE1.md on measuring one thing at a time.
SPECTRAL_SCORED = ("leverage", "ridge", "cur_residual", "supersede+leverage")
# Methods the path-substitution arms are swept on. Deliberately a subset: three arms across all
# eleven methods would more than double a grid that already takes ~2h, and the question the arms
# ask ("does naming each file once help?") is answered by the strongest non-spectral control plus
# the spectral methods that currently lead. `random`/`recency` cannot be helped by better scoring,
# and `topics` was cut in Phase 1.
PATH_METHODS = ("tfidf", "leverage", "ridge", "supersede+leverage", "supersede+band+leverage")
# A genuine tradeoff rather than a tuning constant, so it is measured rather than picked: at 2
# the table holds only paths that repeat, which is token-optimal, and at 1 it holds every path,
# which is recall-optimal because a path mentioned once is the one selection is likeliest to
# drop. Arms C and D are both swept at both settings: C prices the extra rows on their own
# (nothing but the table moves), D prices the same table paired with substitution. Arm B
# emits no table, so the setting cannot reach it.
PATH_MIN_MENTIONS = 2
PATH_MIN_MENTIONS_ALT = 1
# Jobs between incremental checkpoint writes. A job in this grid costs ~1-2s (the largest
# transcript spends ~19 min on its 600), while a full checkpoint dump is ~500 KB and tens of
# milliseconds — two orders of magnitude apart. So 10 keeps the write overhead under ~0.5% of run
# time while capping what a kill can destroy at ~20s of work instead of a whole transcript.
FLUSH_EVERY = 10

# Which of path substitution's three separable effects an arm turns on. See docs/PHASE2.md,
# "The path arms": the base variant is arm A.
PathArm = Literal["off", "score_only", "table_only", "full"]


@dataclass(frozen=True)
class Variant:
    """One configuration of the pipeline's knobs, swept alongside the baseline rather than
    replacing it. Every record carries the variant that produced it, so variants accumulate in a
    transcript's checkpoint instead of invalidating it — adding one re-runs only its own
    combinations, not the whole grid."""

    drop_top: int = 0
    paths: PathArm = "off"
    path_min_mentions: int = PATH_MIN_MENTIONS
    # Methods this variant is meaningful for; None means every method in METHODS. A scoring knob
    # applied to `random` would just duplicate the baseline record at twice the cost.
    applies_to: tuple[str, ...] | None = None

    def key(self) -> str:
        # min_mentions only appears in the key when an arm is on and it is off the default, so
        # every record written before this knob existed keeps the key it already had.
        mentions = self.paths != "off" and self.path_min_mentions != PATH_MIN_MENTIONS
        parts = [
            f"drop_top={self.drop_top}" if self.drop_top else "",
            f"paths={self.paths}" if self.paths != "off" else "",
            f"mm={self.path_min_mentions}" if mentions else "",
        ]
        return "+".join(p for p in parts if p) or "base"

    def covers(self, method: str) -> bool:
        return self.applies_to is None or method in self.applies_to

    def compress_config(self, strategy: Strategy) -> CompressConfig:
        return CompressConfig(strategy=strategy, drop_top=self.drop_top)

    @property
    def substitutes(self) -> bool:
        """Whether scoring sees markers instead of paths (arms B and D)."""
        return self.paths in ("score_only", "full")

    @property
    def emits_table(self) -> bool:
        """Whether the output carries the path table (arms C and D) — which both guarantees its
        atoms and costs tokens. Arm B substitutes for scoring only and emits neither."""
        return self.paths in ("table_only", "full")


BASE_VARIANT = Variant()
# Every min_mentions setting some variant asks for, so analyse() builds each table exactly once.
_MIN_MENTIONS_IN_USE = (PATH_MIN_MENTIONS, PATH_MIN_MENTIONS_ALT)
VARIANTS: tuple[Variant, ...] = (
    BASE_VARIANT,
    Variant(drop_top=1, applies_to=SPECTRAL_SCORED),
    Variant(drop_top=2, applies_to=SPECTRAL_SCORED),
    Variant(drop_top=4, applies_to=SPECTRAL_SCORED),
    Variant(paths="score_only", applies_to=PATH_METHODS),
    Variant(paths="table_only", applies_to=PATH_METHODS),
    Variant(paths="table_only", path_min_mentions=PATH_MIN_MENTIONS_ALT, applies_to=PATH_METHODS),
    Variant(paths="full", applies_to=PATH_METHODS),
    Variant(paths="full", path_min_mentions=PATH_MIN_MENTIONS_ALT, applies_to=PATH_METHODS),
)


@dataclass(frozen=True)
class RunContext:
    """Everything a method needs beyond the segments and the rate.

    `table` is built once per transcript from the ORIGINAL segments and shared by every method
    and arm, so the atoms it guarantees and the tokens it costs are identical across the
    comparison. A per-method table (built from whatever that method's pipeline had left) would
    make `guaranteed_share` a property of the method and quietly break the head-to-head.
    """

    cache_dir: Path
    variant: Variant
    table: PathTable


MethodFn = Callable[[list[Segment], float, ProtectPolicy, RunContext], list[Segment]]


def _spread(items: list[Any], n: int) -> list[Any]:
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def iter_local() -> Iterator[tuple[str, bool, Transcript]]:
    # Stem split drops the session uuid: labels are local_5k .. local_200k, sample content never
    # leaves this process. Matches the withholding pattern in exp_c_code.py / PHASE1.md.
    for p in sorted((DATA / "local" / "sessions").glob("*.jsonl")):
        yield f"local_{p.stem.split('_')[0]}", False, load_claude_code_jsonl(p)


def iter_trace_commons(sample: int) -> Iterator[tuple[str, bool, Transcript]]:
    files = sorted(
        (DATA / "public" / "trace-commons" / "sessions" / "claude_code").glob("*.jsonl"),
        # Name breaks size ties: `_spread` picks by position, so an unstable order would change
        # which files are sampled and what they are labelled, orphaning their checkpoints.
        key=lambda p: (p.stat().st_size, p.name),
    )
    for p in _spread(files, sample):
        yield f"trace-commons_{p.stem[:8]}", True, load_claude_code_jsonl(p)


def iter_swe_gym(sample: int) -> Iterator[tuple[str, bool, Transcript]]:
    files = sorted((DATA / "public" / "swe-gym").rglob("*.parquet"))
    if not files:
        return
    import pyarrow.parquet as pq

    table = pq.read_table(files[0])
    col = "messages" if "messages" in table.column_names else table.column_names[0]
    msgs = table.column(col).to_pylist()
    order = sorted(range(len(msgs)), key=lambda i: len(json.dumps(msgs[i], default=str)))
    for i in _spread(order, sample):
        payload = msgs[i]
        if isinstance(payload, str):
            payload = json.loads(payload)
        yield f"swe-gym_row{i}", True, load_messages_json(payload, name=f"swe-gym-{i}")


def _iter_sources(
    public_sample: int, swe_gym_sample: int
) -> Iterator[tuple[str, bool, Transcript]]:
    yield from iter_local()
    yield from iter_trace_commons(public_sample)
    yield from iter_swe_gym(swe_gym_sample)


def _tfidf_scores(segments: list[Segment]) -> np.ndarray:
    # Duplicates experiments/baselines.py's tfidf vectorizer config rather than importing it,
    # because that function returns selected Segments (via select_by_score's own budget), and the
    # stacked supersede+tfidf method needs bare scores to feed select_with_budget instead.
    if not segments:
        return np.zeros(0)
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][\w./-]+|\d+", sublinear_tf=True)
    try:
        matrix = vec.fit_transform([s.text for s in segments])
    except ValueError:
        return np.zeros(len(segments))
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    lengths = np.asarray((matrix > 0).sum(axis=1)).ravel()
    scores: np.ndarray = np.divide(sums, lengths, out=np.zeros_like(sums), where=lengths > 0)
    return scores


def _leverage_scores(segments: list[Segment], ctx: RunContext) -> np.ndarray:
    x = embed_segments(segments, embed.SELECT_MODEL, cache_dir=ctx.cache_dir)
    return score_segments(x, ctx.variant.compress_config("leverage"))


def _band(segments: list[Segment], quality: float) -> list[Segment]:
    # quality == keep_ratio: the quantity available at each sweep point is exactly the fidelity
    # knob quantize_segment expects, and both already live on [0, 1].
    return [quantize_segment(s, quality) if s.kind in CODE_KINDS else s for s in segments]


def _pool(segs: list[Segment], ctx: RunContext) -> tuple[list[Segment], list[Segment] | None]:
    """(what scoring sees, what is emitted when that differs).

    Substitution happens here — after supersession and after banding, immediately before scoring
    — because supersession matches raw `file_path` values across tool calls and banding parses
    real code, and both would be reading markers otherwise.
    """
    if not ctx.variant.substitutes:
        return segs, None
    subbed = substitute(segs, ctx.table)
    # Arm B keeps the emitted text original: it isolates the scoring effect alone, so it must not
    # also collect the marker's token saving.
    return (subbed, segs) if ctx.variant.paths == "score_only" else (subbed, None)


def _run_plain_baseline(fn: Callable[..., list[Segment]]) -> MethodFn:
    def run(
        segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
    ) -> list[Segment]:
        del ctx
        return fn(segs, ratio, protect=protect)

    return run


def _run_tfidf(
    segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
) -> list[Segment]:
    # Reproduces baselines.tfidf exactly (proven in tests) rather than calling it, because the
    # path arms need the scored list and the emitted list to be separable, which the shared
    # `compress` contract does not expose.
    scoring, emit = _pool(segs, ctx)
    return select_with_budget(
        scoring, _tfidf_scores(scoring), token_budget(segs, ratio), protect, emit
    )


def _run_spectral(strategy: Strategy) -> MethodFn:
    def run(
        segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
    ) -> list[Segment]:
        scoring, emit = _pool(segs, ctx)
        x = embed_segments(scoring, embed.SELECT_MODEL, cache_dir=ctx.cache_dir)
        scores = score_segments(x, ctx.variant.compress_config(strategy))
        return select_with_budget(scoring, scores, token_budget(segs, ratio), protect, emit)

    return run


def _run_topics(
    segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
) -> list[Segment]:
    del ctx
    return topics_compress(segs, ratio, protect=protect)


def _run_supersede_tfidf(
    segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
) -> list[Segment]:
    budget = token_budget(segs, ratio)
    scoring, emit = _pool(apply_supersession(segs), ctx)
    return select_with_budget(scoring, _tfidf_scores(scoring), budget, protect, emit)


def _run_supersede_leverage(
    segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
) -> list[Segment]:
    budget = token_budget(segs, ratio)
    scoring, emit = _pool(apply_supersession(segs), ctx)
    return select_with_budget(scoring, _leverage_scores(scoring, ctx), budget, protect, emit)


def _run_band_leverage(
    segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
) -> list[Segment]:
    budget = token_budget(segs, ratio)
    scoring, emit = _pool(_band(segs, ratio), ctx)
    return select_with_budget(scoring, _leverage_scores(scoring, ctx), budget, protect, emit)


def _run_supersede_band_leverage(
    segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
) -> list[Segment]:
    budget = token_budget(segs, ratio)
    scoring, emit = _pool(_band(apply_supersession(segs), ratio), ctx)
    return select_with_budget(scoring, _leverage_scores(scoring, ctx), budget, protect, emit)


METHODS: dict[str, MethodFn] = {
    "random": _run_plain_baseline(random_drop),
    "recency": _run_plain_baseline(recency_only),
    "tfidf": _run_tfidf,
    "leverage": _run_spectral("leverage"),
    "ridge": _run_spectral("ridge"),
    "cur_residual": _run_spectral("cur_residual"),
    "topics": _run_topics,
    "supersede+tfidf": _run_supersede_tfidf,
    "supersede+leverage": _run_supersede_leverage,
    "band+leverage": _run_band_leverage,
    "supersede+band+leverage": _run_supersede_band_leverage,
}

# Bump only when a record's FIELDS change shape or meaning — that is the one thing a checkpoint
# cannot recover from, so it discards the file. Adding a method, a ratio or a variant does not
# bump this: those are new job identities, and analyse() simply runs the ones it is missing.
# 3: records carry their own identity (variant_key/variant) and the file carries a
# segments_digest, replacing the run-level config fingerprint.
SCHEMA_VERSION = 3


def segments_digest(segments: list[Segment]) -> str:
    """Fingerprint of the segmenter's output for a transcript.

    This is the checkpoint's *input* identity, and it is the only thing that can invalidate a
    whole file: if the loader or segmenter changes, every record was scored against text that no
    longer exists. Downstream text transformations (banding, and later path substitution and the
    per-kind shrinkers) are deliberately NOT in here — they belong to the variant that produced a
    record, so a run with substitution on and a run with it off coexist in one checkpoint rather
    than evicting each other.
    """
    h = hashlib.sha256()
    # The atom extractor is part of the input identity even though it changes no segment text:
    # atoms are the recall metric's denominator, so a changed extractor makes old records
    # incomparable to new ones in exactly the way a changed segmenter does.
    h.update(f"atoms={ATOMS_VERSION}".encode())
    h.update(b"\x02")
    for s in segments:
        h.update(s.kind.encode("utf-8"))
        h.update(b"\x00")
        h.update(s.text.encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()


@dataclass(frozen=True)
class Job:
    method: str
    protect: ProtectPolicy
    ratio: float
    variant: Variant

    def identity(self) -> RecordIdentity:
        return RecordIdentity(
            self.method,
            self.protect,
            self.ratio,
            self.variant.key(),
            transforms_stamp(self.method),
            paths_stamp(self.variant),
            band_quality_stamp(self.method),
        )


def plan_jobs() -> list[Job]:
    """Every (method, protect, ratio, variant) combination the current configuration asks for.
    Order puts the base variant first so an interrupted run still leaves a usable baseline."""
    jobs: list[Job] = []
    for variant in VARIANTS:
        for name in METHODS:
            if variant.covers(name):
                jobs += [Job(name, "none", r, variant) for r in KEEP_RATIOS]
        # protect=code_only exists to price the old binary protect flag against protect=none.
        # That is a question about the baseline, not about any knob a variant turns, so it is
        # swept on the base variant alone. Guarding only the path arms left 36 drop_top x
        # code_only records that every table, curve and plot filtered back out again.
        if variant.key() != BASE_VARIANT.key():
            continue
        for name in PROTECT_CODE_ONLY_METHODS:
            if variant.covers(name):
                jobs += [Job(name, "code_only", r, variant) for r in KEEP_RATIOS]
    return jobs


def transforms_stamp(method: str) -> int:
    """The shrinker version a method's results depend on; 0 for methods that never shrink text."""
    return TRANSFORMS_VERSION if method in TRANSFORMING_METHODS else 0


def paths_stamp(variant: Variant) -> int:
    """The `paths.py` version an arm's results depend on; 0 for the arms that never touch it."""
    return PATHS_VERSION if variant.paths != "off" else 0


def band_quality_stamp(method: str) -> int:
    """The banding-depth mapping version a method depends on; 0 for methods that never band.

    Scoped by TRANSFORMING_METHODS because banding is exactly what routes a method's text through
    quantize_segment, so "bands" and "shrinks" are one set here — and that set is detected by
    observation in the tests rather than trusted to stay hand-written.
    """
    return BAND_QUALITY_VERSION if method in TRANSFORMING_METHODS else 0


class RecordIdentity(NamedTuple):
    """What makes a stored result the answer to a specific question.

    A NamedTuple rather than a bare tuple on purpose: this grew from four fields to seven, and
    the positional unpack left behind by the five-field version crashed `_write_summary`
    — the last step of a multi-hour run. Fields are read by name so adding one cannot do that
    again.
    """

    method: str
    protect: str
    ratio: float
    variant_key: str
    # Every stamp is absent on records written before it existed, and defaults to 0. That is what
    # makes the scoping work: a method that never shrinks carries stamp 0 too, so its old records
    # stay valid, while a shrinking method's old record is recomputed. Same for the path arms,
    # and for the banding-depth mapping this file owns itself.
    transforms_version: int
    paths_version: int
    band_quality_version: int


def _record_identity(rec: dict[str, Any]) -> RecordIdentity:
    return RecordIdentity(
        str(rec["method"]),
        str(rec["protect"]),
        float(rec["keep_ratio"]),
        str(rec.get("variant_key", BASE_VARIANT.key())),
        int(rec.get("transforms_version", 0)),
        int(rec.get("paths_version", 0)),
        int(rec.get("band_quality_version", 0)),
    )


_EARNED_TWIN = {
    "atom_recall_earned": "atom_recall",
    "atom_recall_final_earned": "atom_recall_final",
}


def _metric(rec: dict[str, Any], key: str) -> float:
    """Read a metric from a record, backfilling the fields added with the path arms.

    The backfill is exact, not approximate: `*_earned` scores the atoms no side table hands over,
    and every record written before the arms existed emitted no table, so its earned value is its
    unearned value by definition and its guaranteed share is zero. That is why adding these
    fields does not bump SCHEMA_VERSION and re-run 18 transcripts.
    """
    if key in rec:
        return float(rec[key])
    if key == "guaranteed_share":
        return 0.0
    return float(rec[_EARNED_TWIN[key]])


def _record(
    method: str,
    protect: ProtectPolicy,
    ratio: float,
    ctx: RunContext,
    budget: int,
    original: list[Segment],
    kept: list[Segment],
    original_vectors: np.ndarray,
    final_atoms: set[str],
    elapsed: float,
) -> dict[str, Any]:
    variant = ctx.variant
    # `original` is always the unsubstituted transcript, so every arm is scored against the same
    # denominator; what an arm changes is what the output holds, not what it is measured against.
    guaranteed = ctx.table.guaranteed_atoms() if variant.emits_table else None
    # The table is charged HERE and only here: selection still gets the full shared budget, and
    # the table's cost lands in the achieved compression_ratio instead. So an arm that emits one
    # is not literally at the same output size as the other methods at the same keep_ratio — it
    # shows up further right on the rate axis, and every cross-method table in this file
    # interpolates on achieved rate, which is where the cost is paid back. Deducting it from the
    # budget was the alternative; it makes "matched budget" exact but hands the arms a smaller
    # candidate pool at the aggressive rates, which confounds the mechanism with the deduction.
    table_tokens = ctx.table.tokens() if variant.emits_table else 0
    ev = metrics.evaluate(
        original,
        kept,
        original_vectors=original_vectors,
        final_atoms=final_atoms,
        guaranteed=guaranteed,
        extra_tokens=table_tokens,
    )
    kept_tokens = sum(count_tokens(s.text) for s in kept)
    return {
        "method": method,
        "protect": protect,
        "keep_ratio": ratio,
        "budget_tokens": budget,
        "kept_segments": len(kept),
        "kept_tokens": kept_tokens,
        "table_tokens": table_tokens,
        # A stacked method's candidate pool (post-supersession and/or post-banding) can be
        # smaller than the requested budget, so the greedy selector keeps everything it has and
        # stops short — the achieved compression_ratio then plateaus below what the ratio
        # requested. That is a real ceiling, not a defect, but any comparison "at keep_ratio X"
        # is only a matched-budget comparison when this is true; see the head-to-head table.
        "budget_binding": kept_tokens >= 0.99 * budget,
        "variant_key": variant.key(),
        "paths_version": paths_stamp(variant),
        "transforms_version": transforms_stamp(method),
        "band_quality_version": band_quality_stamp(method),
        "variant": {
            "drop_top": variant.drop_top,
            "paths": variant.paths,
            "path_min_mentions": variant.path_min_mentions,
        },
        "compression_ratio": ev["compression_ratio"],
        "atom_recall": ev["atom_recall"],
        "atom_recall_earned": ev["atom_recall_earned"],
        "atom_recall_final": ev["atom_recall_final"],
        "atom_recall_final_earned": ev["atom_recall_final_earned"],
        "guaranteed_share": ev["guaranteed_share"],
        "atom_recall_by_kind": ev["atom_recall_by_kind"],
        "coverage": ev["coverage"],
        "doc_cosine": ev["doc_cosine"],
        "coverage_by_kind": ev["coverage_by_kind"],
        "elapsed_seconds": elapsed,
    }


def _job_desc(job: Job) -> str:
    return f"{job.method} [{job.variant.key()}] protect={job.protect} r={job.ratio}"


def analyse(
    label: str,
    public: bool,
    t: Transcript,
    cache_dir: Path,
    prior: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run every planned job that `prior` does not already hold, and merge the two.

    `prior` is a checkpoint whose segments_digest already matched, so its records were scored
    against the same segmenter output this run produces. Records it holds for jobs still in the
    plan are reused verbatim; records for jobs no longer planned are kept too, since a variant
    that has been commented out of VARIANTS is cheaper to leave on disk than to re-measure if it
    comes back.

    `checkpoint` is called with the partially-filled result every FLUSH_EVERY jobs and after every
    failure, so a kill mid-transcript costs seconds rather than everything this transcript has
    done so far."""
    segs = segment(t)
    total_tokens = sum(count_tokens(s.text) for s in segs)
    digest = segments_digest(segs)
    # One table per min_mentions setting any variant asks for, built from the ORIGINAL segments
    # so a given setting hands every method and arm the identical table. Counts only reach the
    # checkpoint, never the paths themselves: a local session's file paths are exactly the kind
    # of content that must not land in a committed file.
    tables = {mm: build_table(segs, min_mentions=mm) for mm in _MIN_MENTIONS_IN_USE}
    result: dict[str, Any] = {
        "label": label,
        "public": public,
        "n_turns": len(t.turns),
        "n_segments": len(segs),
        "total_tokens": total_tokens,
        "segments_by_kind": dict(Counter(s.kind for s in segs)),
        "schema_version": SCHEMA_VERSION,
        "segments_digest": digest,
        "path_tables": {
            str(mm): {"paths": len(tb.paths), "tokens": tb.tokens()}
            for mm, tb in sorted(tables.items())
        },
        "records": [],
        # Jobs that raised. Deliberately NOT in "records": every table, curve and plot downstream
        # reads that list, so a failure stored there would be averaged in as a measurement.
        "failures": [],
    }
    if len(segs) < 3:
        result["skipped"] = "too few segments"
        return result

    if prior is not None and prior.get("segments_digest") != digest:
        # The segmenter or loader changed, so every prior record was scored against text that no
        # longer exists. This is the one condition that invalidates a whole checkpoint; adding a
        # variant does not.
        if progress is not None:
            progress("segments digest changed, discarding all prior records")
        prior = None
    kept_records = list(prior.get("records", [])) if prior else []
    have = {_record_identity(r) for r in kept_records}
    todo = [j for j in plan_jobs() if j.identity() not in have]
    if progress is not None:
        progress(f"{len(kept_records)} records reused, {len(todo)} to run")
    if not todo:
        result["records"] = kept_records
        result["supersede_ceiling"] = prior.get("supersede_ceiling", 1.0) if prior else 1.0
        return result

    # The supersession pass does not depend on keep_ratio, so its ceiling is one number per
    # transcript: the largest achievable compression_ratio any supersede+* method can reach
    # before the candidate pool itself, not the budget, is the limiting factor.
    superseded = apply_supersession(segs)
    superseded_tokens = sum(count_tokens(s.text) for s in superseded)
    result["supersede_ceiling"] = superseded_tokens / total_tokens if total_tokens else 1.0
    # Computed once from the ORIGINAL segments (not from whatever a given method kept), so every
    # method/ratio is scored against the identical final-state atom set — see metrics.py.
    final_atoms = {a for s in superseded for a in s.atoms}

    # Embedded once with the scoring model and reused for every method/ratio's metrics.evaluate
    # call; selection scoring uses the disjoint SELECT_MODEL internally (embed_segments'
    # cache_dir), so this one-time embed never overlaps with what a method scores on.
    original_vectors = embed.embed_texts([s.text for s in segs], embed.SCORE_MODEL)

    fresh: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def flush() -> None:
        result["records"] = kept_records + fresh
        result["failures"] = failures
        if checkpoint is not None:
            checkpoint(result)

    for n, job in enumerate(todo, start=1):
        fn = METHODS[job.method]
        ctx = RunContext(
            cache_dir=cache_dir,
            variant=job.variant,
            table=tables[job.variant.path_min_mentions],
        )
        budget = token_budget(segs, job.ratio)
        t0 = time.perf_counter()
        try:
            kept = fn(segs, job.ratio, job.protect, ctx)
            fresh.append(
                _record(
                    job.method,
                    job.protect,
                    job.ratio,
                    ctx,
                    budget,
                    segs,
                    kept,
                    original_vectors,
                    final_atoms,
                    time.perf_counter() - t0,
                )
            )
        except Exception as exc:
            # One job must not take a multi-hour grid down with it, and must not be mistaken for
            # a job that ran badly either. The failure is kept out of "records" entirely, and its
            # identity therefore stays absent from the checkpoint — so the next run retries it,
            # which is what recovers the cell once the cause is fixed. KeyboardInterrupt and
            # SystemExit are BaseExceptions and still abort the run, as they must.
            failures.append(
                {
                    "method": job.method,
                    "protect": job.protect,
                    "keep_ratio": job.ratio,
                    "variant_key": job.variant.key(),
                    "job_index": n,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                    "elapsed_seconds": time.perf_counter() - t0,
                }
            )
            if progress is not None:
                progress(f"JOB FAILED {n}/{len(todo)} {_job_desc(job)} -> {type(exc).__name__}")
            flush()
        if n % FLUSH_EVERY == 0:
            if progress is not None:
                done = len(fresh)
                progress(f"job {n}/{len(todo)} ok={done} failed={len(failures)} {_job_desc(job)}")
            flush()
    flush()
    return result


def _plot(label: str, records: list[dict[str, Any]]) -> None:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["protect"] == "none":
            by_method[r["method"]].append(r)
    names = list(METHODS)
    cmap = plt.get_cmap("tab20")
    colors = {name: cmap(i / max(len(names) - 1, 1)) for i, name in enumerate(names)}

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for name in names:
        # x-axis is the ACHIEVED compression_ratio, not the requested keep_ratio: a stacked
        # method's candidate pool can be smaller than the budget (see budget_binding on each
        # record), so plotting against the request would show it "losing" at a ratio it never
        # actually operated at. A plateau here is a real ceiling, drawn as one, not hidden.
        rows = sorted(by_method.get(name, []), key=lambda r: float(r["compression_ratio"]))
        if not rows:
            continue
        x = [r["compression_ratio"] for r in rows]
        style = dict(marker="o", ms=3, lw=1.3, label=name, color=colors[name])
        axes[0].plot(x, [r["atom_recall"] for r in rows], **style)
        axes[1].plot(x, [r["atom_recall_final"] for r in rows], **style)
        axes[2].plot(x, [r["coverage"] for r in rows], **style)
    axes[0].set_title("atom recall (all-history) vs achieved ratio")
    axes[1].set_title("atom recall (final-state) vs achieved ratio")
    axes[2].set_title("semantic coverage vs achieved ratio")
    for ax in axes:
        ax.set_xlabel("achieved compression ratio (kept tokens / total tokens)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, color="#e6e6e3", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[2].legend(fontsize=7, ncols=1, frameon=False, loc="lower right")
    fig.suptitle(f"{label}  (protect=none)")
    fig.tight_layout()
    fig.savefig(OUT / f"{label}.png", dpi=110)
    plt.close(fig)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = ~np.isnan(v)
    if not mask.any() or w[mask].sum() <= 0:
        return float("nan")
    ww = w[mask] / w[mask].sum()
    return float(np.sum(v[mask] * ww))


METRICS = ("recall", "recall_final", "coverage")
_METRIC_LABEL = {
    "recall": "atom recall (all-history)",
    "recall_final": "atom recall (final-state)",
    "coverage": "semantic coverage",
}


def _index_records(
    results: list[dict[str, Any]],
) -> dict[tuple[str, str, float], list[tuple[float, float, float, float]]]:
    idx: dict[tuple[str, str, float], list[tuple[float, float, float, float]]] = defaultdict(list)
    for r in results:
        if "skipped" in r:
            continue
        weight = float(r["total_tokens"])
        for rec in r["records"]:
            key = (rec["method"], rec["protect"], rec["keep_ratio"])
            idx[key].append((rec["atom_recall"], rec["atom_recall_final"], rec["coverage"], weight))
    return idx


def _aggregate_table(
    idx: dict[tuple[str, str, float], list[tuple[float, float, float, float]]],
) -> list[str]:
    def row(metric_pos: int) -> Iterator[list[float]]:
        for method in METHODS:
            yield [
                _weighted_mean(
                    [v[metric_pos] for v in idx.get((method, "none", ratio), [])],
                    [v[3] for v in idx.get((method, "none", ratio), [])],
                )
                for ratio in KEEP_RATIOS
            ]

    header = f"{'method':<26}" + "".join(f"{'r=' + str(q):>9}" for q in KEEP_RATIOS)
    out = [
        "== token-weighted metrics across all transcripts, by REQUESTED keep_ratio "
        "(protect=none) ==",
        "columns are the shared token budget every method was handed, not what each method",
        "achieved; a supersede+/band+ row can fall short of its budget once its candidate pool",
        "is exhausted (see budget_binding per record, and the ceiling table below). r=0.7-0.9 are",
        "included for completeness, not because they are the target operating points — see the",
        "aggressive-rate table below for the rates this project actually cares about.",
    ]
    for pos, metric in enumerate(METRICS):
        out += ["", f"-- {_METRIC_LABEL[metric]} --", header, "-" * len(header)]
        for method, vals in zip(METHODS, row(pos), strict=True):
            out.append(f"{method:<26}" + "".join(f"{v:>9.3f}" for v in vals))
    return out


def _curves_by_method(
    results: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Per (method, protect), one entry per transcript: its achieved-rate curve (deduplicated
    and sorted by compression_ratio) plus its token weight. This is the shape the head-to-head
    and knee estimates need — comparing at a REQUESTED keep_ratio conflates two methods that
    landed at different achieved rates; interpolating each method's own curve onto a common
    achieved-rate grid does not."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if "skipped" in r:
            continue
        by_key: dict[tuple[str, str], list[tuple[float, float, float, float]]] = defaultdict(list)
        for rec in r["records"]:
            key = (rec["method"], rec["protect"])
            by_key[key].append(
                (
                    rec["compression_ratio"],
                    rec["atom_recall"],
                    rec["atom_recall_final"],
                    rec["coverage"],
                )
            )
        for key, pts in by_key.items():
            pts.sort(key=lambda p: p[0])
            rates: list[float] = []
            recall: list[float] = []
            recall_final: list[float] = []
            coverage: list[float] = []
            for rate, rec_, rec_final, cov in pts:
                # Stacked methods repeat the same ceiling point at every keep_ratio past it;
                # np.interp requires a strictly increasing x, and a repeated point carries no
                # extra information (kept segments are identical), so only the first is kept.
                if rates and rate <= rates[-1] + 1e-9:
                    continue
                rates.append(rate)
                recall.append(rec_)
                recall_final.append(rec_final)
                coverage.append(cov)
            out[key].append(
                {
                    "label": r["label"],
                    "weight": float(r["total_tokens"]),
                    "rates": rates,
                    "recall": recall,
                    "recall_final": recall_final,
                    "coverage": coverage,
                }
            )
    return out


def _interp_at(curve: dict[str, Any] | None, target_rate: float, metric: str) -> float | None:
    if curve is None:
        return None
    rates = curve["rates"]
    if not rates or target_rate < rates[0] - 1e-9 or target_rate > rates[-1] + 1e-9:
        return None
    return float(np.interp(target_rate, rates, curve[metric]))


_H2H_COL = 14


def _head_to_head(curves: dict[tuple[str, str], list[dict[str, Any]]]) -> list[str]:
    header = f"{'method vs tfidf':<26}" + "".join(
        f"{'r=' + str(q):>{_H2H_COL}}" for q in KEEP_RATIOS
    )
    out = [
        "",
        "== spectral methods vs tfidf at matched ACHIEVED rate ==",
        "This is a rate-distortion comparison, not a scoreboard: losing some distortion is the",
        "point of lossy compaction, so a negative cell is not automatically a defect — read it",
        "together with the aggressive-rate and knee tables below, which are the operating points",
        "that matter for this project. Each cell interpolates both curves onto the same achieved",
        "compression_ratio and compares only transcripts where the method reached that rate;",
        "'n/a(ceilX)' means none did — X is the best ceiling reached.",
        header,
        "-" * len(header),
    ]

    def _row(method: str, metric: str) -> str:
        cells = []
        spectral_curves = curves.get((method, "none"), [])
        tfidf_curves = {c["label"]: c for c in curves.get(("tfidf", "none"), [])}
        for target in KEEP_RATIOS:
            vals: list[float] = []
            weights: list[float] = []
            ceilings: list[float] = []
            for c in spectral_curves:
                sv = _interp_at(c, target, metric)
                if sv is None:
                    ceilings.append(c["rates"][-1] if c["rates"] else 0.0)
                    continue
                bv = _interp_at(tfidf_curves.get(c["label"]), target, metric)
                if bv is None:
                    continue
                vals.append(sv - bv)
                weights.append(c["weight"])
            if not vals:
                ceiling = max(ceilings) if ceilings else 0.0
                cells.append(f"n/a(ceil{ceiling:.2f})".rjust(_H2H_COL))
                continue
            n = len(vals)
            total = len(spectral_curves)
            mark = "" if n == total else f"*{n}/{total}"
            cells.append((f"{_weighted_mean(vals, weights):+.3f}" + mark).rjust(_H2H_COL))
        return f"{method:<26}" + "".join(cells)

    for metric in METRICS:
        out += [
            "",
            f"-- {_METRIC_LABEL[metric]} delta (spectral - tfidf) --",
            header,
            "-" * len(header),
        ]
        for method in SPECTRAL_METHODS:
            out.append(_row(method, metric))
    out += [
        "",
        "* = fraction of transcripts contributing to that cell, when fewer than all of them",
        "reached the target rate (spectral side); n/a(ceilX) = none did, X is the best ceiling.",
    ]
    return out


_METRIC_SHORT = {"recall": "recall", "recall_final": "recall_f", "coverage": "coverage"}


def _aggressive_rate_table(curves: dict[tuple[str, str], list[dict[str, Any]]]) -> list[str]:
    header = f"{'method':<26}" + "".join(
        f"{_METRIC_SHORT[m] + '@' + str(r):>16}" for r in AGGRESSIVE_RATES for m in METRICS
    )
    out = [
        "",
        "== distortion at the aggressive rates this project is actually built for "
        f"(achieved ~= {', '.join(str(r) for r in AGGRESSIVE_RATES)}, i.e. 2x/3x/4x) ==",
        "recall = atom_recall (all-history), recall_f = atom_recall_final, coverage = semantic",
        "coverage.",
        header,
        "-" * len(header),
    ]
    for method in METHODS:
        method_curves = curves.get((method, "none"), [])
        cells = []
        for target in AGGRESSIVE_RATES:
            for metric in METRICS:
                vals, weights, ceilings = [], [], []
                for c in method_curves:
                    v = _interp_at(c, target, metric)
                    if v is None:
                        ceilings.append(c["rates"][-1] if c["rates"] else 0.0)
                        continue
                    vals.append(v)
                    weights.append(c["weight"])
                if not vals:
                    ceiling = max(ceilings) if ceilings else 0.0
                    cells.append(f"n/a(ceil{ceiling:.2f})".rjust(16))
                else:
                    cells.append(f"{_weighted_mean(vals, weights):.3f}".rjust(16))
        out.append(f"{method:<26}" + "".join(cells))
    return out


def _supersede_ceiling_table(results: list[dict[str, Any]]) -> list[str]:
    out = [
        "",
        "== per-transcript supersede ceiling (post-dedupe pool / original tokens) ==",
        f"{'transcript':<26}{'tokens':>10}{'ceiling':>10}",
        "-" * 46,
    ]
    for r in results:
        if "skipped" in r:
            continue
        out.append(f"{r['label']:<26}{r['total_tokens']:>10}{r['supersede_ceiling']:>10.3f}")
    return out


def _supersede_recall_cost_table(
    results: list[dict[str, Any]], curves: dict[tuple[str, str], list[dict[str, Any]]]
) -> list[str]:
    # The requested comparison: atom recall of a supersede-based method AT ITS OWN CEILING
    # against a non-supersede method constrained to that same achieved rate. Both operate at the
    # identical achieved compression ratio. All-history recall shows a real cost of the dedupe
    # (it removes segments whose atoms atom_recall still counts); final-state recall shows how
    # much of that cost is the counting artefact atom_recall_final exists to strip out.
    tfidf_curves = {c["label"]: c for c in curves.get(("tfidf", "none"), [])}
    leverage_curves = {c["label"]: c for c in curves.get(("leverage", "none"), [])}
    out = [
        "",
        "== atom-recall cost of supersession: supersede+leverage at its own ceiling vs "
        "leverage/tfidf constrained to that same rate ==",
        f"{'transcript':<22}{'ceiling':>8}"
        f"{'sup+lev':>9}{'sup+lev_f':>10}{'leverage':>9}{'lev_f':>7}{'tfidf':>7}{'tfidf_f':>8}",
        "-" * 80,
    ]
    for c in curves.get(("supersede+leverage", "none"), []):
        if not c["rates"]:
            continue
        ceiling = c["rates"][-1]
        sup_recall = c["recall"][-1]
        sup_recall_final = c["recall_final"][-1]
        lev = leverage_curves.get(c["label"])
        tf = tfidf_curves.get(c["label"])

        def _cell(
            curve: dict[str, Any] | None, metric: str, width: int, *, rate: float = ceiling
        ) -> str:
            v = _interp_at(curve, rate, metric)
            return f"{v:.3f}".rjust(width) if v is not None else "-".rjust(width)

        out.append(
            f"{c['label']:<22}{ceiling:>8.3f}"
            f"{sup_recall:>9.3f}{sup_recall_final:>10.3f}"
            f"{_cell(lev, 'recall', 9)}{_cell(lev, 'recall_final', 7)}"
            f"{_cell(tf, 'recall', 7)}{_cell(tf, 'recall_final', 8)}"
        )
    out += [
        "(_f columns are atom_recall_final; sup+lev's own value is by construction its ceiling",
        "point, leverage/tfidf are interpolated to the same achieved rate for comparison.)",
    ]
    return out


def _knee(curve: dict[str, Any], floor: float = KNEE_FLOOR) -> float | None:
    """Knee = the lowest achieved rate, scanning down from the highest tested rate, at which
    atom_recall_final has not yet dropped below `floor`. This is a floor-crossing estimate, not
    a curvature computation: the sweep's rate grid is coarse enough that a numerical second
    derivative would be noise-dominated, and a floor is easy to defend and state plainly. Returns
    None if even the highest tested rate fails the floor."""
    rates = curve["rates"]
    recall_final = curve["recall_final"]
    if not rates:
        return None
    knee: float | None = None
    for rate, v in zip(reversed(rates), reversed(recall_final), strict=True):
        if v >= floor:
            knee = rate
        else:
            break
    return knee


def _knee_table(curves: dict[tuple[str, str], list[dict[str, Any]]]) -> list[str]:
    out = [
        "",
        f"== knee per method: lowest achieved rate (token-weighted mean) at which "
        f"atom_recall_final >= {KNEE_FLOOR} still holds ==",
        "Lower is better here: it means the method holds an acceptable final-state atom recall",
        "further down the compression axis before degrading. 'never' = failed the floor even at",
        "the least-aggressive rate tested; ranked ascending (best first).",
        f"{'method':<26}{'knee':>8}{'n':>5}",
        "-" * 39,
    ]
    ranked: list[tuple[str, float]] = []
    never: list[str] = []
    for method in METHODS:
        knees = [
            (k, c["weight"])
            for c in curves.get((method, "none"), [])
            if (k := _knee(c)) is not None
        ]
        if not knees:
            never.append(method)
            continue
        mean_knee = _weighted_mean([k for k, _ in knees], [w for _, w in knees])
        ranked.append((method, mean_knee))
    ranked.sort(key=lambda p: p[1])
    for method, knee in ranked:
        n = len(curves.get((method, "none"), []))
        out.append(f"{method:<26}{knee:>8.3f}{n:>5}")
    for method in never:
        out.append(f"{method:<26}{'never':>8}{len(curves.get((method, 'none'), [])):>5}")
    return out


def _protect_cost_table(
    idx: dict[tuple[str, str, float], list[tuple[float, float, float, float]]],
) -> list[str]:
    header = f"{'method (code_only - none)':<26}" + "".join(
        f"{'r=' + str(q):>9}" for q in KEEP_RATIOS
    )
    out = ["", "== cost of protect=code_only vs protect=none =="]
    for pos, metric in enumerate(METRICS):
        out += ["", f"-- {_METRIC_LABEL[metric]} delta --", header, "-" * len(header)]
        for method in PROTECT_CODE_ONLY_METHODS:
            deltas = []
            for ratio in KEEP_RATIOS:
                co = _weighted_mean(
                    [v[pos] for v in idx.get((method, "code_only", ratio), [])],
                    [v[3] for v in idx.get((method, "code_only", ratio), [])],
                )
                none_ = _weighted_mean(
                    [v[pos] for v in idx.get((method, "none", ratio), [])],
                    [v[3] for v in idx.get((method, "none", ratio), [])],
                )
                deltas.append(co - none_)
            out.append(f"{method:<26}" + "".join(f"{d:>+9.3f}" for d in deltas))
    return out


def _base_only(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same results with every non-baseline variant's records removed.

    Every table below this line was written when a checkpoint held exactly one configuration.
    Feeding them the variant records too would silently average a drop_top=2 curve into the
    baseline's, so variants are compared in their own table and nowhere else.
    """
    out: list[dict[str, Any]] = []
    for r in results:
        base = [
            rec
            for rec in r.get("records", [])
            if _record_identity(rec).variant_key == BASE_VARIANT.key()
        ]
        out.append({**r, "records": base})
    return out


def _variant_table(results: list[dict[str, Any]]) -> list[str]:
    """Each variant against the baseline, per method, at the aggressive operating points.

    Compared at matched ACHIEVED rate like every other cross-method table here: a scoring knob
    can shift where a method lands, so comparing at a requested keep_ratio would confound the
    knob's effect with a rate difference.
    """
    curves = _curves_by_method_variant(results)
    # Path arms are excluded here and get their own table: this one compares raw final-state
    # recall, which an arm that emits a table inflates for free. Mixing them would read as a win.
    keys = [
        v.key() for v in VARIANTS if v.key() != BASE_VARIANT.key() and not _is_path_key(v.key())
    ]
    if not keys:
        return []
    header = f"{'method / variant':<34}" + "".join(
        f"{'d recall_f@' + f'{1 / r:.0f}x':>16}" for r in AGGRESSIVE_RATES
    )
    out = [
        "",
        "== variants vs the base configuration ==",
        "Difference in final-state atom recall against the same method's baseline curve, at",
        "matched achieved compression. Positive means the variant is better. 'n/a' means one of",
        "the two curves never reached that rate on any transcript.",
        header,
        "-" * len(header),
    ]
    for method in METHODS:
        base = curves.get((method, BASE_VARIANT.key()))
        if not base:
            continue
        for key in keys:
            variant = curves.get((method, key))
            if not variant:
                continue
            cells = []
            for rate in AGGRESSIVE_RATES:
                bv = _weighted_at_rate(base, rate, "recall_final")
                vv = _weighted_at_rate(variant, rate, "recall_final")
                cells.append("n/a" if bv is None or vv is None else f"{vv - bv:+.3f}")
            out.append(f"{method + '  ' + key:<34}" + "".join(f"{c:>16}" for c in cells))
    return out


def _is_path_key(key: str) -> bool:
    return "paths=" in key


# Metrics the variant curves carry, and the record field each reads.
_VARIANT_METRICS = {
    "recall_final": "atom_recall_final",
    "recall_final_earned": "atom_recall_final_earned",
    "coverage": "coverage",
    "guaranteed_share": "guaranteed_share",
}


def _curves_by_method_variant(
    results: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """_curves_by_method's shape, keyed on (method, variant_key) at protect=none instead of
    (method, protect), for the variant comparison."""
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if "skipped" in r:
            continue
        by_key: dict[tuple[str, str], list[tuple[float, ...]]] = defaultdict(list)
        for rec in r["records"]:
            identity = _record_identity(rec)
            method, protect, variant_key = identity.method, identity.protect, identity.variant_key
            if protect != "none":
                continue
            by_key[(method, variant_key)].append(
                (
                    float(rec["compression_ratio"]),
                    *(_metric(rec, field) for field in _VARIANT_METRICS.values()),
                )
            )
        for key, pts in by_key.items():
            pts.sort(key=lambda p: p[0])
            rates: list[float] = []
            series: dict[str, list[float]] = {name: [] for name in _VARIANT_METRICS}
            for point in pts:
                rate = point[0]
                if rates and rate <= rates[-1] + 1e-9:
                    continue
                rates.append(rate)
                for name, value in zip(_VARIANT_METRICS, point[1:], strict=True):
                    series[name].append(value)
            out[key].append(
                {
                    "label": r["label"],
                    "weight": float(r["total_tokens"]),
                    "rates": rates,
                    **series,
                }
            )
    return out


def _weighted_at_rate(curves: list[dict[str, Any]], rate: float, metric: str) -> float | None:
    vals: list[float] = []
    weights: list[float] = []
    for c in curves:
        v = _interp_at(c, rate, metric)
        if v is None:
            continue
        vals.append(v)
        weights.append(c["weight"])
    if not vals:
        return None
    # _weighted_mean rather than np.average: this is end-of-run code, and np.average raises
    # ZeroDivisionError when every weight is zero (a transcript whose total_tokens is 0).
    mean = _weighted_mean(vals, weights)
    return None if np.isnan(mean) else mean


_PATH_ARM_METRICS = ("recall_final_earned", "recall_final", "coverage")
_PATH_ARM_LABEL = {
    "recall_final_earned": "recall_f_earn",
    "recall_final": "recall_f",
    "coverage": "coverage",
}


def _path_arm_table(results: list[dict[str, Any]]) -> list[str]:
    """The path-substitution arms against the base configuration, at matched achieved rate.

    `recall_f_earn` is the column the arms are judged on. It scores only the atoms the table does
    NOT hand over, so arms C and D cannot win by guaranteeing facts — the difference there is
    selection genuinely getting better or worse. `recall_f` is the total the output preserves,
    guaranteed atoms included, which is what a downstream reader would actually see. The two
    disagreeing is the expected result, not a defect, and is exactly why arm C exists.
    """
    curves = _curves_by_method_variant(results)
    keys = [v.key() for v in VARIANTS if _is_path_key(v.key())]
    if not keys:
        return []
    header = f"{'method / arm':<46}" + "".join(
        f"{_PATH_ARM_LABEL[m] + '@' + f'{1 / r:.0f}x':>18}"
        for r in AGGRESSIVE_RATES
        for m in _PATH_ARM_METRICS
    )
    out = [
        "",
        "== path substitution arms vs the base configuration ==",
        "Deltas against the same method's base curve at matched achieved compression. The table",
        "is charged into the reported compression ratio ONLY; it is deliberately NOT deducted",
        "from the selection budget, so a table arm selects from the same token budget as every",
        "other method and then pays for the table by landing further right on the rate axis.",
        "These arm results are therefore NOT budget-neutral: the matched-rate comparison below is",
        "where the table's cost is charged back, so read every cell as 'at the same achieved",
        "compression' and never as 'at the same budget'. recall_f_earn = final-state atom recall",
        "over only the atoms",
        "the table does not guarantee; recall_f = the same including them; coverage = semantic",
        "coverage. Positive is better. 'n/a' = one of the two curves never reached that rate.",
        "Arms: score_only = B (substitute for scoring, emit the original, no table),",
        "table_only = C (emit the table, substitute nothing — pure overhead, not shippable),",
        "full = D (table plus markers). '+mm=1' is the same arm with the table built from every",
        "path rather than only the repeated ones; C and D are each swept at both settings.",
        header,
        "-" * len(header),
    ]
    for method in METHODS:
        base = curves.get((method, BASE_VARIANT.key()))
        if not base:
            continue
        for key in keys:
            arm = curves.get((method, key))
            if not arm:
                continue
            cells = []
            for rate in AGGRESSIVE_RATES:
                for metric in _PATH_ARM_METRICS:
                    bv = _weighted_at_rate(base, rate, metric)
                    av = _weighted_at_rate(arm, rate, metric)
                    cells.append("n/a" if bv is None or av is None else f"{av - bv:+.3f}")
            out.append(f"{method + '  ' + key:<46}" + "".join(f"{c:>18}" for c in cells))
    return out


def _emits_table_at(rec: dict[str, Any], min_mentions: int) -> bool:
    """Whether this record's arm emitted the table built at `min_mentions`.

    Read off the stored variant fields rather than by matching text in the variant key: the key
    omits mm at its default and now spells more than one table-emitting arm, so a suffix test on
    it drops arms silently — a row would read as unmeasured rather than wrong.
    """
    variant = rec.get("variant")
    if not isinstance(variant, dict):
        return False
    if variant.get("paths") not in ("table_only", "full"):
        return False
    return int(variant.get("path_min_mentions", PATH_MIN_MENTIONS)) == min_mentions


def _path_table_cost_table(results: list[dict[str, Any]]) -> list[str]:
    """What the table costs and what it buys, per transcript, before any method runs.

    Both numbers are properties of the transcript and the min_mentions setting, not of any
    method, which is why they are reported once here rather than repeated per row above.
    """
    rows = [r for r in results if "skipped" not in r and "path_tables" in r]
    if not rows:
        return []
    out = [
        "",
        "== path table cost and guarantee, per min_mentions setting ==",
        "This is the tradeoff min_mentions is a knob for, priced directly: 1 tabulates every",
        "path (recall-optimal, a path mentioned once is the one selection is likeliest to drop),",
        "2 tabulates only repeated ones (token-optimal, a row for a single mention costs more",
        "than the mention it replaces). share = fraction of the FINAL-STATE atom set the table",
        "spells out so selection never has to; cost = table tokens over transcript tokens. Path",
        "strings are never printed: local sessions are in this corpus.",
        f"{'transcript':<26}{'mm':>4}{'paths':>8}{'tokens':>9}{'cost':>8}{'share':>8}",
        "-" * 63,
    ]
    for r in rows:
        total = float(r["total_tokens"]) or 1.0
        for mm, pt in sorted(r["path_tables"].items()):
            share = max(
                (
                    _metric(rec, "guaranteed_share")
                    for rec in r["records"]
                    if _emits_table_at(rec, int(mm))
                ),
                default=float("nan"),
            )
            share_cell = "-".rjust(8) if np.isnan(share) else f"{share:>8.3f}"
            out.append(
                f"{r['label']:<26}{mm:>4}{pt['paths']:>8}{pt['tokens']:>9}"
                f"{pt['tokens'] / total:>8.3f}{share_cell}"
            )
    return out


def failure_report(results: list[dict[str, Any]]) -> list[str]:
    """Failed and missing cells, first thing in the summary.

    Counts both, because they are different holes: a failure is a job that raised this run, while
    a missing cell is a planned identity absent from the checkpoint for any reason at all (an
    aborted earlier run, a `--limit`, a stamp bump). Every table below interpolates over whatever
    cells exist, so a silent hole reads as a normal curve measured on fewer points — which is why
    this is loud and at the top rather than a footnote.
    """
    planned = {j.identity() for j in plan_jobs()}
    rows: list[str] = []
    errors: Counter[tuple[str, str, str]] = Counter()
    total_failed = 0
    total_missing = 0
    for r in results:
        if "skipped" in r:
            continue
        failures = list(r.get("failures", []))
        missing = len(planned - {_record_identity(rec) for rec in r.get("records", [])})
        total_failed += len(failures)
        total_missing += missing
        for f in failures:
            errors[(str(f.get("method")), str(f.get("variant_key")), str(f.get("error")))] += 1
        if failures or missing:
            rows.append(f"{r['label']:<26}{len(failures):>9}{missing:>9}")
    if not rows:
        return ["All planned jobs present in every checkpoint; no failures.", ""]
    out = [
        "!! INCOMPLETE RUN — READ BEFORE THE TABLES BELOW !!",
        f"{total_failed} job(s) raised and were skipped; {total_missing} planned cell(s) are not",
        "in the checkpoints. Skipped jobs are NOT records and are not averaged into anything, so",
        "an affected method is summarised over fewer rates or fewer transcripts than its",
        "neighbours rather than being pulled down by a zero. Each failure is stored with its",
        "traceback in its checkpoint's `failures` list, and is retried on the next run.",
        "",
        f"{'transcript':<26}{'failed':>9}{'missing':>9}",
        "-" * 44,
        *rows,
    ]
    if errors:
        out += ["", "-- distinct failures (most frequent first) --"]
        out += [
            f"{n:>5}x  {method} [{key}]  {err}" for (method, key, err), n in errors.most_common(10)
        ]
    out.append("")
    return out


def _write_summary(results: list[dict[str, Any]]) -> None:
    base_results = _base_only(results)
    idx = _index_records(base_results)
    curves = _curves_by_method(base_results)
    lines = [f"{len(results)} transcripts checkpointed.", ""]
    lines += failure_report(results)
    lines += _aggressive_rate_table(curves)
    lines += _knee_table(curves)
    lines += _aggregate_table(idx)
    lines += _head_to_head(curves)
    lines += _protect_cost_table(idx)
    lines += _supersede_ceiling_table(base_results)
    lines += _supersede_recall_cost_table(base_results, curves)
    lines += _variant_table(results)
    lines += _path_arm_table(results)
    lines += _path_table_cost_table(results)
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


def _write_checkpoint(out_path: Path, payload: dict[str, Any]) -> None:
    """Serialise to a sibling temp file, then os.replace onto the real name.

    A direct write_text of ~500 KB is not atomic: a kill mid-write leaves truncated JSON, which
    _load_checkpoint can only treat as unreadable — silently discarding a whole transcript's work
    and re-running it. The temp file must share the directory so the replace is a same-volume
    rename, which is atomic on both POSIX and Windows.
    """
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, out_path)


def _load_checkpoint(out_path: Path, force: bool) -> tuple[dict[str, Any] | None, str | None]:
    """Load a checkpoint to build on. Returns (prior, None) when its records may be reused, or
    (None, reason) when they may not — reason is None only when there was no checkpoint at all.

    Unlike the run-level fingerprint this replaces, a mismatch here is rare by design: only an
    unreadable file, `--force`, or a record-schema bump discards work. Whether the *segments*
    still match is decided in analyse(), which is where they are computed. Split out from main()
    so the decision is unit-testable without a transcript or an embedding model.
    """
    if not out_path.exists():
        return None, None
    if force:
        return None, "--force"
    try:
        candidate = json.loads(out_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"checkpoint unreadable ({type(exc).__name__})"
    if candidate.get("schema_version") != SCHEMA_VERSION:
        return None, (f"record schema {candidate.get('schema_version')!r} != {SCHEMA_VERSION}")
    return candidate, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-sample", type=int, default=8)
    parser.add_argument("--swe-gym-sample", type=int, default=5)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    n_done = 0
    for label, public, t in _iter_sources(args.public_sample, args.swe_gym_sample):
        if args.only and args.only not in label:
            continue
        if args.limit is not None and n_done >= args.limit:
            break
        out_path = OUT / f"{label}.json"
        prior, discard_reason = _load_checkpoint(out_path, args.force)
        if prior is None and discard_reason:
            print(f"[{label}] prior checkpoint discarded ({discard_reason})", flush=True)
        t0 = time.perf_counter()
        r = analyse(
            label,
            public,
            t,
            CACHE,
            prior=prior,
            progress=lambda msg, label=label: print(  # type: ignore[misc]
                f"{time.strftime('%H:%M:%S')} [{label}] {msg}", flush=True
            ),
            checkpoint=lambda payload, out_path=out_path: _write_checkpoint(out_path, payload),  # type: ignore[misc]
        )
        elapsed = time.perf_counter() - t0
        # Checkpoint immediately: a crash on the next transcript must not lose this one.
        _write_checkpoint(out_path, r)
        base_records = [
            rec for rec in r["records"] if _record_identity(rec).variant_key == BASE_VARIANT.key()
        ]
        if base_records:
            _plot(label, base_records)
        failed = len(r.get("failures", []))
        note = f", {failed} JOBS FAILED" if failed else ""
        print(
            f"{time.strftime('%H:%M:%S')} [{label}] done in {elapsed:.1f}s, "
            f"{len(r['records'])} records{note}",
            flush=True,
        )
        results.append(r)
        n_done += 1

    _write_summary(results)
    total_failed = sum(len(r.get("failures", [])) for r in results)
    if total_failed:
        print("\n" + "!" * 78, flush=True)
        print(f"{total_failed} job(s) failed and were skipped. Where:", flush=True)
        for r in results:
            for f in r.get("failures", []):
                print(
                    f"  [{r['label']}] {f['method']} [{f['variant_key']}] "
                    f"protect={f['protect']} r={f['keep_ratio']} -> {f['error']}",
                    flush=True,
                )
        print(
            "Full tracebacks are in each checkpoint's `failures` list. Re-running (without "
            "--force) retries exactly these jobs and keeps everything that succeeded.",
            flush=True,
        )
        print("!" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

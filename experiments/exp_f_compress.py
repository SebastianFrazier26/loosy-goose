"""Phase 2 / Experiment F: compression-based selectors on Experiment D's curves.

Scores a segment by how many compressed bytes it adds given the text before it
(`loosy_goose.compress_select`), with zlib's 32 KiB window against lzma's whole-transcript
dictionary, exact against lazy and stochastic greedy, and each with and without the supersession
pass. No reference method runs here: tfidf, leverage and supersede+leverage are read from the
exp_d checkpoints, and only where those were scored against the same segmenter output.

Usage: uv run python experiments/exp_d_curves.py --grid exp_f [--only NAME_SUBSTR] [--methods A,B]
Writes experiments/output/exp_f/<label>.json and experiments/output/exp_f/summary.txt.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import exp_d_curves as exp_d  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    AGGRESSIVE_RATES,
    BASE_VARIANT,
    EXP_D,
    KNEE_FLOOR,
    METRICS,
    Grid,
    Job,
    MethodFn,
    RunContext,
    Variant,
    _base_only,
    _curves_by_method,
    _interp_at,
    _knee,
    _record_identity,
    _weighted_mean,
    plan_for_transcript,
    plan_jobs,
)
from loosy_goose.budget import ProtectPolicy, select_with_budget, token_budget  # noqa: E402
from loosy_goose.compress_select import (  # noqa: E402
    Codec,
    greedy_select,
    onepass_scores,
    stochastic_select,
)
from loosy_goose.segment import Segment  # noqa: E402
from loosy_goose.supersede import apply_supersession  # noqa: E402

OUT = exp_d.ROOT / "experiments" / "output" / "exp_f"
REFERENCES = ("tfidf", "leverage", "supersede+leverage")
SHIPPED = "supersede+leverage"

# Calibrated 2026-09-22 against exact greedy on the selector's own objective (compressed length of
# the kept text), never on recall or coverage; full tables in docs/PHASE2.md. Rule: the largest
# epsilon of {0.3, 0.1, 0.03, 0.01} whose median over 54 zlib cells (18 transcripts x keep
# 0.5/0.3/0.1) of mean-over-5-seeds f(stochastic)/f(exact) is >= 0.99 with p10 >= 0.97.
# 0.03 is it: median 0.998, p10 0.986 (0.1 failed at p10 0.964). Against exact lzma greedy it
# also passes (median 0.998, p10 0.976), but only on the 8 smallest transcripts (<= 37k tokens)
# where exact lzma finished in 10 minutes. Seed spread (median 0.017) exceeds half the
# stochastic-vs-lazy gap (0.0035), so one seed is not a representative draw. The seed is the
# variant's stoch_seed; the base variant's 0 is the draw every seed-0 record was taken with.
STOCHASTIC_EPSILON = 0.03

ZLIB_STOCHASTIC = ("zlib-stochastic", "supersede+zlib-stochastic")
LZMA_STOCHASTIC = ("lzma-stochastic", "supersede+lzma-stochastic")
ZLIB_SEEDS = 10
LZMA_SEEDS = 5
# lzma seeds 1..4 run only up to here. The corpus jumps from 98,917 to 151,611 tokens, and per
# extra seed the lzma pair costs ~1.8 h on the 13 transcripts below the cap against ~5.9 h on all
# 18 (scaled from 741 s on the 52,853-token smoke transcript). Seed 0 runs everywhere.
LZMA_SEED_MAX_TOKENS = 100_000
SEED_VARIANTS: tuple[Variant, ...] = tuple(
    Variant(stoch_seed=s, applies_to=ZLIB_STOCHASTIC + (LZMA_STOCHASTIC if s < LZMA_SEEDS else ()))
    for s in range(1, ZLIB_SEEDS)
)


def _job_filter(job: Job, total_tokens: int) -> bool:
    return not (
        job.method in LZMA_STOCHASTIC
        and job.variant.stoch_seed != 0
        and total_tokens > LZMA_SEED_MAX_TOKENS
    )


# One-pass scores do not depend on the keep ratio, and lzma's cost ~277 s on the largest
# transcript, so each (codec, pool) is scored once per process and every ratio reuses it. The
# cold time is stored beside the scores so every record of that pool reports the same honest
# cost, whether or not its own job paid it.
_SCORE_CACHE: dict[tuple[Codec, str], tuple[npt.NDArray[np.float64], float]] = {}


def _pool_digest(pool: list[Segment]) -> str:
    h = hashlib.sha256()
    for s in pool:
        b = s.text.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def _cached_scores(pool: list[Segment], codec: Codec) -> tuple[npt.NDArray[np.float64], float]:
    key = (codec, _pool_digest(pool))
    hit = _SCORE_CACHE.get(key)
    if hit is None:
        t0 = time.perf_counter()
        scores = onepass_scores(pool, codec)
        hit = (scores, time.perf_counter() - t0)
        _SCORE_CACHE[key] = hit
    return hit


def _run_greedy(codec: Codec, *, lazy: bool, supersede: bool) -> MethodFn:
    def run(
        segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
    ) -> list[Segment]:
        del ctx
        budget = token_budget(segs, ratio)
        pool = apply_supersession(segs) if supersede else segs
        return greedy_select(pool, budget, codec, lazy=lazy, protect=protect)

    return run


def _run_onepass(codec: Codec, *, supersede: bool) -> MethodFn:
    def run(
        segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
    ) -> list[Segment]:
        budget = token_budget(segs, ratio)
        pool = apply_supersession(segs) if supersede else segs
        scores, cold_seconds = _cached_scores(pool, codec)
        # Only the one-pass arms carry this field: greedy scoring is interleaved with selection
        # and cannot be cached across ratios, so its whole cost is already in elapsed_seconds.
        ctx.extras["score_seconds"] = cold_seconds
        return select_with_budget(pool, scores, budget, protect)

    return run


def _run_stochastic(codec: Codec, *, supersede: bool) -> MethodFn:
    def run(
        segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
    ) -> list[Segment]:
        budget = token_budget(segs, ratio)
        pool = apply_supersession(segs) if supersede else segs
        return stochastic_select(
            pool,
            budget,
            codec,
            epsilon=STOCHASTIC_EPSILON,
            seed=ctx.variant.stoch_seed,
            protect=protect,
        )

    return run


def _arms(supersede: bool) -> dict[str, MethodFn]:
    prefix = "supersede+" if supersede else ""
    return {
        f"{prefix}zlib-greedy": _run_greedy("zlib", lazy=False, supersede=supersede),
        f"{prefix}zlib-lazy": _run_greedy("zlib", lazy=True, supersede=supersede),
        f"{prefix}zlib-onepass": _run_onepass("zlib", supersede=supersede),
        f"{prefix}lzma-lazy": _run_greedy("lzma", lazy=True, supersede=supersede),
        f"{prefix}lzma-onepass": _run_onepass("lzma", supersede=supersede),
        f"{prefix}zlib-stochastic": _run_stochastic("zlib", supersede=supersede),
        f"{prefix}lzma-stochastic": _run_stochastic("lzma", supersede=supersede),
    }


METHODS: dict[str, MethodFn] = {**_arms(False), **_arms(True)}


def _load_references(
    results: list[dict[str, Any]], reference_dir: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    """`results` with each transcript's exp_d reference records merged in, plus one line per
    transcript left without references.

    A reference counts only when its checkpoint's segments_digest equals this one's: otherwise
    the two were measured on different segmenter output and a delta between them means nothing.
    Reference records must also be ones exp_d still plans, so a stale stamp is never compared.
    """
    exp_d_plan = {j.identity() for j in plan_jobs(grid=EXP_D)}
    merged: list[dict[str, Any]] = []
    excluded: list[str] = []
    for r in results:
        if "skipped" in r:
            merged.append(r)
            continue
        path = reference_dir / f"{r['label']}.json"
        reason = None
        refs: list[dict[str, Any]] = []
        if not path.exists():
            reason = "no exp_d checkpoint"
        else:
            ref = json.loads(path.read_text(encoding="utf-8"))
            if ref.get("segments_digest") != r.get("segments_digest"):
                reason = "segments_digest differs from exp_d's"
            else:
                for rec in ref.get("records", []):
                    ident = _record_identity(rec)
                    if (
                        ident.method in REFERENCES
                        and ident.protect == "none"
                        and ident.variant_key == BASE_VARIANT.key()
                        and ident in exp_d_plan
                    ):
                        refs.append(rec)
                if not refs:
                    reason = "exp_d checkpoint holds no reference records"
        if reason is not None:
            excluded.append(f"  {r['label']}: {reason}")
        merged.append({**r, "records": list(r.get("records", [])) + refs})
    return merged, excluded


def _n_transcripts(results: list[dict[str, Any]]) -> int:
    return sum(1 for r in results if "skipped" not in r)


def _header(results: list[dict[str, Any]], excluded: list[str], grid: Grid) -> list[str]:
    n = _n_transcripts(results)
    plans = {
        r["label"]: {j.identity() for j in plan_for_transcript(r.get("total_tokens"), grid=grid)}
        for r in results
        if "skipped" not in r
    }
    rows: list[str] = []
    not_run: list[str] = []
    incomplete: list[str] = []
    for method in grid.methods:
        present = failed = want = 0
        for r in results:
            if "skipped" in r:
                continue
            mine = {i for i in plans[r["label"]] if i.method == method}
            want += len(mine)
            present += len({_record_identity(rec) for rec in r["records"]} & mine)
            failed += sum(1 for f in r.get("failures", []) if f.get("method") == method)
        if present == 0 and failed == 0:
            not_run.append(method)
        elif present < want or failed:
            incomplete.append(method)
        rows.append(f"{method:<26}{want:>9}{present:>9}{failed:>9}{want - present:>9}")
    out = [
        f"exp_f: compression-based selectors. {n} transcripts checkpointed, "
        f"{n - len(excluded)} with exp_d references ({', '.join(REFERENCES)}).",
    ]
    if excluded:
        out += ["Transcripts excluded from every reference comparison:", *excluded]
    out += [
        "",
        f"{'method':<26}{'planned':>9}{'present':>9}{'failed':>9}{'missing':>9}",
        "-" * 62,
        *rows,
    ]
    if not_run:
        out += [
            "",
            f"Not run yet: {', '.join(not_run)}. No records by design (left out of a --methods",
            "run); their cells below read 'missing' and are never averaged as zero.",
        ]
    if incomplete:
        out += [
            "",
            f"!! INCOMPLETE: {', '.join(incomplete)} — some cells failed or are missing; those",
            "methods are summarised over fewer rates or transcripts than the others. Failures",
            "and tracebacks are in each checkpoint's `failures` list and retry on the next run.",
        ]
    out += [
        "",
        "planned/present count every seed; every table below reads seed 0 (the base variant)",
        "only, except the seed section at the end.",
        "",
    ]
    return out


_Curves = dict[tuple[str, str], list[dict[str, Any]]]


def _value_cell(curves: list[dict[str, Any]], rate: float, metric: str, total: int) -> str:
    if not curves:
        return "missing"
    vals: list[float] = []
    weights: list[float] = []
    for c in curves:
        v = _interp_at(c, rate, metric)
        if v is not None:
            vals.append(v)
            weights.append(c["weight"])
    if not vals:
        return "n/a"
    mark = "" if len(vals) == total else f"*{len(vals)}/{total}"
    return f"{_weighted_mean(vals, weights):.3f}{mark}"


def _delta_cell(
    a: list[dict[str, Any]], b: list[dict[str, Any]], rate: float, metric: str, total: int
) -> str:
    """Token-weighted mean of (a - b) over transcripts where both curves reach `rate`."""
    if not a or not b:
        return "missing"
    b_by_label = {c["label"]: c for c in b}
    vals: list[float] = []
    weights: list[float] = []
    for c in a:
        av = _interp_at(c, rate, metric)
        bv = _interp_at(b_by_label.get(c["label"]), rate, metric)
        if av is None or bv is None:
            continue
        vals.append(av - bv)
        weights.append(c["weight"])
    if not vals:
        return "n/a"
    mark = "" if len(vals) == total else f"*{len(vals)}/{total}"
    return f"{_weighted_mean(vals, weights):+.3f}{mark}"


_SHORT = {"recall": "recall", "recall_final": "recall_f", "coverage": "coverage"}
_DELTA_METRICS = ("recall_final", "coverage")
_W = 16


def _rate_label(rate: float) -> str:
    return f"{1 / rate:.0f}x"


def _distortion_table(curves: _Curves, methods: list[str], total: int) -> list[str]:
    header = f"{'method':<26}" + "".join(
        f"{_SHORT[m] + '@' + str(r):>{_W}}" for r in AGGRESSIVE_RATES for m in METRICS
    )
    out = [
        "",
        "== distortion at achieved rate 0.5 / 0.33 / 0.25 ==",
        "*n/N = only n of N transcripts reached that rate; n/a = none did.",
        header,
        "-" * len(header),
    ]
    for method in methods:
        cs = curves.get((method, "none"), [])
        cells = [_value_cell(cs, r, m, total) for r in AGGRESSIVE_RATES for m in METRICS]
        out.append(f"{method:<26}" + "".join(c.rjust(_W) for c in cells))
    return out


def _delta_table(
    title: str, notes: list[str], pairs: list[tuple[str, str]], curves: _Curves, total: int
) -> list[str]:
    header = f"{'pair':<46}" + "".join(
        f"{'d ' + _SHORT[m] + '@' + _rate_label(r):>{_W}}"
        for r in AGGRESSIVE_RATES
        for m in _DELTA_METRICS
    )
    out = ["", title, *notes, header, "-" * len(header)]
    for a, b in pairs:
        ca = curves.get((a, "none"), [])
        cb = curves.get((b, "none"), [])
        cells = [_delta_cell(ca, cb, r, m, total) for r in AGGRESSIVE_RATES for m in _DELTA_METRICS]
        out.append(f"{a + ' - ' + b:<46}" + "".join(c.rjust(_W) for c in cells))
    return out


def _knee_table(curves: _Curves, methods: list[str]) -> list[str]:
    out = [
        "",
        f"== knee: lowest achieved rate still at atom_recall_final >= {KNEE_FLOOR} ==",
        "Token-weighted mean over transcripts that reach the floor; lower is better. n = those",
        "transcripts / transcripts with the method.",
        f"{'method':<26}{'knee':>8}{'n':>8}",
        "-" * 42,
    ]
    for method in methods:
        cs = curves.get((method, "none"), [])
        if not cs:
            out.append(f"{method:<26}{'missing':>8}{'0/0':>8}")
            continue
        knees = [(k, c["weight"]) for c in cs if (k := _knee(c)) is not None]
        cell = (
            f"{_weighted_mean([k for k, _ in knees], [w for _, w in knees]):.3f}"
            if knees
            else "never"
        )
        out.append(f"{method:<26}{cell:>8}{f'{len(knees)}/{len(cs)}':>8}")
    return out


def _cost_table(merged: list[dict[str, Any]], methods: list[str]) -> list[str]:
    out = [
        "",
        "== cost: median seconds per 100k tokens ==",
        "Per transcript: median elapsed_seconds over its ratios, plus score_seconds (the cold",
        "one-pass scoring time, paid once and cached across ratios) for the onepass arms. Then",
        "the median over transcripts. References are exp_d's own timings.",
        f"{'method':<26}{'s/100k':>10}{'n':>5}",
        "-" * 41,
    ]
    for method in methods:
        per: list[float] = []
        for r in merged:
            if "skipped" in r or not r.get("total_tokens"):
                continue
            recs = [rec for rec in r["records"] if rec["method"] == method]
            if not recs:
                continue
            seconds = float(np.median([float(rec["elapsed_seconds"]) for rec in recs]))
            seconds += max(float(rec.get("score_seconds", 0.0)) for rec in recs)
            per.append(seconds / (float(r["total_tokens"]) / 100_000))
        cell = f"{float(np.median(per)):.2f}" if per else "missing"
        out.append(f"{method:<26}{cell:>10}{len(per):>5}")
    return out


_STOCHASTIC_SEEDS: dict[str, tuple[int, ...]] = {
    **{m: tuple(range(ZLIB_SEEDS)) for m in ZLIB_STOCHASTIC},
    **{m: tuple(range(LZMA_SEEDS)) for m in LZMA_STOCHASTIC},
}
# (stochastic arm, deterministic arm it is judged against, verdict name for that arm). lzma has
# no exact arm: exact greedy is infeasible for it (compress_select.greedy_select).
_SEED_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("zlib-stochastic", "zlib-lazy", "lazy"),
    ("zlib-stochastic", "zlib-greedy", "exact"),
    ("supersede+zlib-stochastic", "supersede+zlib-lazy", "lazy"),
    ("supersede+zlib-stochastic", "supersede+zlib-greedy", "exact"),
    ("lzma-stochastic", "lzma-lazy", "lazy"),
    ("supersede+lzma-stochastic", "supersede+lzma-lazy", "lazy"),
)


def _only_variant(results: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [
        {
            **r,
            "records": [
                rec for rec in r.get("records", []) if _record_identity(rec).variant_key == key
            ],
        }
        for r in results
    ]


def _all_seed_labels(
    results: list[dict[str, Any]], arm: str, seeds: tuple[int, ...], grid: Grid
) -> list[str]:
    """Transcripts holding every planned ratio of `arm` for every seed in `seeds`."""
    ratios = set(grid.ratios)
    out: list[str] = []
    for r in results:
        if "skipped" in r:
            continue
        have: dict[str, set[float]] = {}
        for rec in r.get("records", []):
            if rec["method"] == arm and rec["protect"] == "none":
                key = _record_identity(rec).variant_key
                have.setdefault(key, set()).add(float(rec["keep_ratio"]))
        if all(ratios <= have.get(Variant(stoch_seed=s).key(), set()) for s in seeds):
            out.append(r["label"])
    return out


def _per_seed(
    seed_curves: dict[int, _Curves],
    arm: str,
    seeds: tuple[int, ...],
    labels: list[str],
    rate: float,
    metric: str,
    ref: list[dict[str, Any]] | None = None,
) -> tuple[list[float], int] | None:
    """Per seed, the token-weighted mean of `metric` (minus `ref`'s, when given) at `rate`, over
    the transcripts of `labels` where every seed's curve and `ref`'s reach the rate. The set is
    the same for every seed, so the seeds differ only by their draw."""
    by_seed = {s: {c["label"]: c for c in seed_curves[s].get((arm, "none"), [])} for s in seeds}
    ref_by = None if ref is None else {c["label"]: c for c in ref}
    rows: list[tuple[list[float], float, float]] = []
    for lab in labels:
        vals = [_interp_at(by_seed[s].get(lab), rate, metric) for s in seeds]
        present = [v for v in vals if v is not None]
        if len(present) < len(seeds):
            continue
        offset = 0.0
        if ref_by is not None:
            rv = _interp_at(ref_by.get(lab), rate, metric)
            if rv is None:
                continue
            offset = rv
        rows.append((present, offset, float(by_seed[seeds[0]][lab]["weight"])))
    if not rows:
        return None
    weights = [w for _, _, w in rows]
    per = [_weighted_mean([v[i] - off for v, off, _ in rows], weights) for i in range(len(seeds))]
    return per, len(rows)


def _mark(n: int, total: int) -> str:
    return "" if n == total else f"*{n}/{total}"


def _verdict(lo: float, hi: float, ref_name: str) -> str:
    if lo <= 0.0 <= hi:
        return "inside noise"
    return "stochastic better" if lo > 0.0 else f"{ref_name} better"


def _seed_section(merged: list[dict[str, Any]], grid: Grid) -> list[str]:
    all_seeds = sorted({s for seeds in _STOCHASTIC_SEEDS.values() for s in seeds})
    seed_curves = {
        s: _curves_by_method(_only_variant(merged, Variant(stoch_seed=s).key())) for s in all_seeds
    }
    labels = {
        arm: _all_seed_labels(merged, arm, seeds, grid) for arm, seeds in _STOCHASTIC_SEEDS.items()
    }
    out = [
        "",
        "== Stochastic greedy across seeds ==",
        f"zlib arms: seeds 0-{ZLIB_SEEDS - 1} on every transcript. lzma arms: seeds "
        f"0-{LZMA_SEEDS - 1}, the extra seeds only on transcripts <= {LZMA_SEED_MAX_TOKENS:,}",
        "tokens. Each arm is read only over transcripts holding all its seeds (n); each seed is",
        "aggregated token-weighted over the same transcripts, then summarised as mean [min, max]",
        "across seeds. *u/n = only u of those n reach the rate for every seed; missing = no",
        "transcript holds every seed.",
        "",
        "-- per-seed aggregate at matched achieved rate --",
    ]
    head = f"{'arm':<28}{'rate':>5}{'n':>4}{'recall_f':>26}{'coverage':>26}"
    out += [head, "-" * len(head)]
    for arm, seeds in _STOCHASTIC_SEEDS.items():
        labs = labels[arm]
        for rate in AGGRESSIVE_RATES:
            cells: list[str] = []
            for metric in _DELTA_METRICS:
                if not labs:
                    cells.append("missing")
                    continue
                got = _per_seed(seed_curves, arm, seeds, labs, rate, metric)
                if got is None:
                    cells.append("n/a")
                    continue
                per, used = got
                cells.append(
                    f"{float(np.mean(per)):.3f} [{min(per):.3f}, {max(per):.3f}]"
                    f"{_mark(used, len(labs))}"
                )
            out.append(
                f"{arm:<28}{_rate_label(rate):>5}{len(labs):>4}"
                + "".join(c.rjust(26) for c in cells)
            )

    out += [
        "",
        "-- stochastic minus lazy / exact, same transcripts --",
        "Per seed: token-weighted mean of (stochastic - reference) at matched achieved rate;",
        "shown as mean [min, max] across seeds. The verdict is 'inside noise' when that range",
        "contains 0. The reference is the seed-free base record of the same codec and stacking.",
    ]
    head = f"{'pair':<48}{'rate':>5}" + "".join(
        f"{'d ' + _SHORT[m]:>28}{'verdict':>19}" for m in _DELTA_METRICS
    )
    out += [head, "-" * len(head)]
    for arm, ref_arm, ref_name in _SEED_PAIRS:
        seeds = _STOCHASTIC_SEEDS[arm]
        labs = labels[arm]
        ref = seed_curves[0].get((ref_arm, "none"), [])
        for rate in AGGRESSIVE_RATES:
            cells = []
            for metric in _DELTA_METRICS:
                got = (
                    _per_seed(seed_curves, arm, seeds, labs, rate, metric, ref)
                    if labs and ref
                    else None
                )
                if got is None:
                    cells += ["missing" if not labs or not ref else "n/a", ""]
                    continue
                per, used = got
                lo, hi = min(per), max(per)
                cells += [
                    f"{float(np.mean(per)):+.3f} [{lo:+.3f}, {hi:+.3f}]{_mark(used, len(labs))}",
                    _verdict(lo, hi, ref_name),
                ]
            row = "".join(c.rjust(28) if i % 2 == 0 else c.rjust(19) for i, c in enumerate(cells))
            out.append(f"{arm + ' - ' + ref_arm:<48}{_rate_label(rate):>5}{row}")

    out += [
        "",
        "-- one-draw noise: seed spread of recall_final --",
        "Per transcript, max - min across seeds of recall_final at matched achieved rate; the",
        "median over transcripts that reach the rate for every seed. n as above.",
    ]
    head = f"{'arm':<28}{'n':>4}" + "".join(f"{'@' + _rate_label(r):>12}" for r in AGGRESSIVE_RATES)
    out += [head, "-" * len(head)]
    for arm, seeds in _STOCHASTIC_SEEDS.items():
        labs = labels[arm]
        cells = []
        for rate in AGGRESSIVE_RATES:
            spreads: list[float] = []
            for lab in labs:
                vals = [
                    _interp_at(
                        next(
                            (c for c in seed_curves[s].get((arm, "none"), []) if c["label"] == lab),
                            None,
                        ),
                        rate,
                        "recall_final",
                    )
                    for s in seeds
                ]
                present = [v for v in vals if v is not None]
                if len(present) == len(seeds):
                    spreads.append(max(present) - min(present))
            if not labs:
                cells.append("missing")
            elif not spreads:
                cells.append("n/a")
            else:
                cells.append(f"{float(np.median(spreads)):.3f}{_mark(len(spreads), len(labs))}")
        out.append(f"{arm:<28}{len(labs):>4}" + "".join(c.rjust(12) for c in cells))
    return out


def write_summary_with(
    results: list[dict[str, Any]], out: Path, *, reference_dir: Path, grid: Grid
) -> str:
    merged, excluded = _load_references(results, reference_dir)
    base = _base_only(merged)
    curves = _curves_by_method(base)
    total = _n_transcripts(results)
    arms = list(grid.methods)
    everything = arms + list(REFERENCES)
    scorers = [m for m in arms if not m.startswith("supersede+")]
    lines = _header(results, excluded, grid)
    lines += _distortion_table(curves, everything, total)
    lines += _delta_table(
        "== each arm minus leverage and minus supersede+leverage (shipped default), matched "
        "achieved rate ==",
        ["Positive = the arm keeps more. *n/N = transcripts where both curves reach the rate."],
        [(a, ref) for ref in ("leverage", SHIPPED) for a in arms],
        curves,
        total,
    )
    lines += _delta_table(
        "== approximation cost of lazy greedy: lazy minus exact ==",
        [
            "What lazy greedy gives up against exact greedy; lzma-lazy has no exact twin to",
            "measure against and inherits an approximation of this kind.",
        ],
        [("zlib-lazy", "zlib-greedy"), ("supersede+zlib-lazy", "supersede+zlib-greedy")],
        curves,
        total,
    )
    lines += _delta_table(
        "== approximation cost of stochastic greedy: stochastic minus exact ==",
        [
            f"Seed 0 alone (epsilon {STOCHASTIC_EPSILON}); one draw is not representative, so the",
            "seed section below is the one to read for stochastic vs lazy/exact. lzma-stochastic",
            "was checked against exact lzma greedy only on the small transcripts where exact",
            "finished; beyond them it is unvalidated.",
        ],
        [
            ("zlib-stochastic", "zlib-greedy"),
            ("supersede+zlib-stochastic", "supersede+zlib-greedy"),
        ],
        curves,
        total,
    )
    lines += _delta_table(
        "== effect of supersession per scorer: supersede+X minus X ==",
        [],
        [(f"supersede+{m}", m) for m in scorers],
        curves,
        total,
    )
    lines += _knee_table(curves, everything)
    lines += _cost_table(base, everything)
    lines += _seed_section(merged, grid)
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    return text


def _write_summary(results: list[dict[str, Any]], out: Path) -> None:
    print("\n" + write_summary_with(results, out, reference_dir=EXP_D.out, grid=EXP_F))


EXP_F = Grid(
    name="exp_f",
    out=OUT,
    methods=METHODS,
    variants=(BASE_VARIANT, *SEED_VARIANTS),
    protect_code_only=(),
    write_summary=_write_summary,
    plots=False,
    job_filter=_job_filter,
)

"""Downstream probe QA for slot compression (2026-10-22), on a fresh held-out swe-gym set "C".

Set C: swe-gym transcripts disjoint from Experiment I's 20 (Phase A) and the spike's 10 (Phase B),
drawn with seed exi.SEED + 2. Probes are written by a model other than the Sonnet summariser
(Codex gpt-6-sol; Haiku only if the codex CLI is missing, recorded in the report). A probe counts
only if Haiku answers it correctly from the full context. Haiku answers, Sonnet grades.

Set D (2026-10-22, `--set D`): a larger fresh swe-gym set (n=20, seed exi.SEED + 3, disjoint from
Phase A, Phase B and set C) comparing loosy / line-keep with and without the "gentle" slot profile:
gloosy (supersede -> gentle slot english -> the shipped loosy leverage + per-token novelty
selection) and gline (supersede -> gentle slot english -> line-keep code -> select). Own store and
report in output/slot_compress/d/; line/loosy keys are the set C pipeline, so the sets pool.

Conditions per level: loosy (shipped pipeline), line (supersede -> line-keep code -> select),
slotd / slote (supersede -> slot prose -> line-keep code -> select; direct rows with a legend, or
rows re-realised as English), slme (same order, prose rewritten by a local qwen3-1.7b SLM in
english mode; rewrites are precomputed in the main thread before any call so workers never load
torch), summary (Sonnet at loosy's rendered size), plus full once.

  uv run python experiments/slot_compress/downstream.py --plan [--set D]
  uv run python experiments/slot_compress/downstream.py --dry-run [--conditions loosy,line]
  nohup caffeinate -i uv run python experiments/slot_compress/downstream.py \\
      >> experiments/output/slot_compress/run.log 2>&1 &

Falsification arms (set D, 2026-10-23; see the falsify-extractive spec):
hyb70 / hyb50 (a Sonnet summary at 0.7 / 0.5 of the level budget, the rest filled with the shipped
loosy selection) and lsum (the summary task by a local Qwen3-1.7B, precomputed):

  uv run python experiments/slot_compress/downstream.py --set D --precompute-lsum [--limit N]
  uv run python experiments/slot_compress/downstream.py --set D --pending
  uv run python experiments/slot_compress/downstream.py --set D --conditions hyb70,hyb50,lsum

Larger local summarisers (2026-11-15): lsum4b / lsum8b (official Qwen3 MLX 4-bit; one model loaded
at a time). Default --max-calls 260 when both are in --conditions. Include loosy,summary,lsum so
the report carries the paired rows against them (already-done keys cost no calls):

  uv run python experiments/slot_compress/downstream.py --set D --precompute-lsum \
      --lsum-models lsum4b,lsum8b
  uv run python experiments/slot_compress/downstream.py --set D --conditions \
      loosy,summary,lsum,lsum4b,lsum8b

Resumable: calls are checkpointed in output/slot_compress/calls.jsonl; reruns skip finished keys.
A spend/usage-limit error or --max-calls stops the driver, which then writes the report from
whatever exists.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import shutil
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "spike_thin"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chunked  # noqa: E402
import lines  # noqa: E402
import lsum  # noqa: E402

from experiments import exp_i_downstream as exi  # noqa: E402
from loosy_goose.novelty_select import novelty_select  # noqa: E402
from loosy_goose.pipeline import NOVELTY_ALPHA  # noqa: E402
from loosy_goose.rate import LEVELS  # noqa: E402
from loosy_goose.segment import Segment, segment  # noqa: E402
from loosy_goose.select import CompressConfig, embed_segments, score_segments  # noqa: E402
from loosy_goose.supersede import apply_supersession  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Transcript  # noqa: E402

OUT = ROOT / "experiments" / "output" / "slot_compress"
EXP_I = ROOT / "experiments" / "output" / "exp_i"
SPIKE_B = ROOT / "experiments" / "output" / "spike_thin" / "downstream" / "b_selection.json"

HAIKU = exi.MODEL_BY_NAME["haiku-5.5"]
SONNET = exi.HELPER
NEMO = exi.MODEL_BY_NAME["mistral-nemo-12b"]
CODEX_PROBER = exi.MODEL_BY_NAME["gpt-6-sol"]
C_SEED = exi.SEED + 2
C_N = 10
D_SEED = exi.SEED + 3
D_N = 20
FAMILIES = ("loosy", "line", "slotd", "slote", "slme", "summary", "chunk")
HYB_VERSION = 1
HYB_FRAC = {"hyb70": 0.7, "hyb50": 0.5}
HYB_FAMILIES = tuple(HYB_FRAC)
LSUM_FAMILIES = tuple(lsum.LSUM_MODELS)
NEW_FAMILIES = HYB_FAMILIES + LSUM_FAMILIES
D_FAMILIES = ("loosy", "line", "gloosy", "gline", "summary") + NEW_FAMILIES + ("chunk",)


LSUM_BIG_MAX_CALLS = 260


def lsum_dir(family: str) -> Path:
    return OUT / "d" / f"{family}_cache"


HYB_HEADER = (
    "[Earlier part of the session: a summary of it, followed by passages kept verbatim from it]"
)
HYB_SUMMARY_LABEL = "[Summary of the earlier part]"
HYB_VERBATIM_LABEL = "[Passages kept verbatim from the earlier part]"
GENTLE_FAMILIES = ("gloosy", "gline")
ALL_FAMILIES = tuple(dict.fromkeys(FAMILIES + D_FAMILIES))
SLOT_MODE = {"slotd": "direct", "slote": "english", "gloosy": "english", "gline": "english"}
SLM_MODEL = "qwen3-1.7b"
PROSE_FAMILIES = ("slotd", "slote", "slme", "gloosy", "gline")
LEVEL_NAMES = ("light", "medium", "heavy")
STOP = threading.Event()
# 429s are deliberately absent: a plain rate limit is transient and exi backs off through it.
LIMIT = re.compile(r"spend.?limit|usage.?limit|credit|quota|billing", re.IGNORECASE)

LINE_HEADER = (
    "[Earlier part of the session, compressed: some passages were dropped, and inside long file "
    "and command-output dumps only the lines naming identifiers, paths, numbers or errors were "
    "kept (gaps are marked with '… (N lines elided)'; line numbers were removed). Every line that "
    "remains is verbatim"
)
GLOOSY_HEADER = (
    "[Earlier part of the session, compressed extractively: some passages were dropped, and "
    "prose and reasoning sentences were mechanically shortened and re-worded, with identifiers, "
    "paths and numbers kept exactly; what remains is otherwise verbatim]"
)
SLOTE_NOTE = (
    "prose and reasoning sentences were mechanically shortened and re-worded, with identifiers, "
    "paths and numbers kept exactly"
)


@dataclass(frozen=True)
class SetCfg:
    name: str
    families: tuple[str, ...]
    calls: Path
    prep_dir: Path
    report: Path
    selection: Path
    max_calls: int


SETS = {
    "C": SetCfg(
        "C",
        FAMILIES,
        OUT / "calls.jsonl",
        OUT / "c",
        OUT / "report.md",
        OUT / "c_selection.json",
        450,
    ),
    "D": SetCfg(
        "D",
        D_FAMILIES,
        OUT / "d" / "calls.jsonl",
        OUT / "d",
        OUT / "d" / "report.md",
        OUT / "d_selection.json",
        520,
    ),
}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def load_slot() -> Any:
    try:
        return importlib.import_module("slot")
    except ImportError as exc:
        raise SystemExit(f"slot conditions need experiments/slot_compress/slot.py: {exc}") from exc


@dataclass
class Plan:
    families: list[str]
    levels: list[str]
    slot_version: Any = None
    slm_version: Any = None
    gentle_version: Any = None

    def cond(self, family: str, level: str) -> str:
        if family == "chunk":
            return f"chunk.v{chunked.CHUNK_VERSION}-{level}"
        if family in HYB_FAMILIES:
            return f"{family}.v{HYB_VERSION}-{level}"
        if family in LSUM_FAMILIES:
            return f"{family}.v{lsum.LSUM_VERSION}-{level}"
        if family in GENTLE_FAMILIES:
            return f"{family}.v{self.gentle_version}-{level}"
        if family in SLOT_MODE:
            return f"{family}.v{self.slot_version}-{level}"
        if family == "slme":
            return f"slme.v{self.slm_version}-{SLM_MODEL}-{level}"
        return f"{family}-{level}"


def header_for(family: str, slot: Any) -> str:
    if family == "slotd":
        return f"{LINE_HEADER}; {slot.LEGEND}]"
    if family in ("slote", "slme", "gline"):
        return f"{LINE_HEADER}; {SLOTE_NOTE}]"
    if family == "gloosy":
        return GLOOSY_HEADER
    return f"{LINE_HEADER}]"


def load_slm() -> Any:
    try:
        return importlib.import_module("slm_slot")
    except ImportError as exc:
        raise SystemExit(f"slme needs experiments/slot_compress/slm_slot.py: {exc}") from exc


def transform_pool(pool: list[Segment], fam: str, slot: Any, slm: Any) -> list[Segment]:
    if fam in GENTLE_FAMILIES:
        return slot.slot_segments(pool, SLOT_MODE[fam], profile="gentle", include_tool_prose=True)
    if fam in SLOT_MODE:
        return slot.slot_segments(pool, SLOT_MODE[fam])
    if fam == "slme":
        return slm.slm_slot_segments(pool, "english", slm.MODELS[SLM_MODEL])
    return pool


# --------------------------------------------------------------------------- guarded runner


class Guard:
    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.used = 0
        self._lock = threading.Lock()

    def runner(self, model: exi.Model, prompt: str) -> exi.CallResult:
        with self._lock:
            if STOP.is_set():
                raise exi.CallError("halted by budget guard")
            if self.used >= self.max_calls:
                STOP.set()
                log(f"!! --max-calls {self.max_calls} reached, stopping")
                raise exi.CallError("halted by budget guard")
            self.used += 1
        try:
            return exi.cli_runner(model, prompt)
        except exi.CallError as exc:
            if LIMIT.search(str(exc)):
                STOP.set()
                log(f"!! spend/usage limit hit, stopping: {exc}")
                raise exi.CallError("halted by budget guard") from exc
            raise


def guarded_sleep(s: float) -> None:
    if not STOP.is_set():
        time.sleep(min(s, 300.0))


# --------------------------------------------------------------------------- selection


def exclusions() -> set[str]:
    taken = {p.stem for p in (EXP_I / "contexts").glob("*.json")}
    manifest = EXP_I / "manifest.json"
    if manifest.exists():
        taken |= set(
            re.findall(r"(?:swe-gym_row\d+|trace-commons_[0-9a-f]+)", manifest.read_text())
        )
    if SPIKE_B.exists():
        taken |= set(json.loads(SPIKE_B.read_text(encoding="utf-8")))
    return taken


def select_set(name: str) -> list[tuple[exi.Candidate, exi.Cut]]:
    taken = exclusions()
    n, seed = (C_N, C_SEED) if name == "C" else (D_N, D_SEED)
    if name == "D":
        taken |= set(json.loads(SETS["C"].selection.read_text(encoding="utf-8")))
    cands = (c for c in exi.iter_candidates() if c.source == "swe-gym" and c.label not in taken)
    return exi.select_transcripts(cands, n=n, per_source=n, seed=seed)


# --------------------------------------------------------------------------- contexts


def loosy_select(pool: list[Segment], budget: int) -> list[Segment]:
    """The shipped selection (pipeline.compress_transcript: leverage scores, per-token novelty at
    alpha 0.5, no protect) on an arbitrary pool, so a slotted pool is selected exactly as loosy
    selects the plain supersession pool."""
    config = CompressConfig(strategy="leverage")
    scores = score_segments(embed_segments(pool, config.model_name, exi.CACHE), config)
    return novelty_select(pool, scores, budget, NOVELTY_ALPHA, "none", per_token=True)


def kept_segments(original: list[Segment], level: str, pool: list[Segment]) -> list[Segment]:
    """`pool` is the superseded (and, for slot families, prose-transformed) segment list. Supersede
    first, slot second, line-keep and budget selection third, so tokens the transform frees are
    refilled by selection. The budget stays that of the untransformed text."""
    k = LEVELS[level]
    budget = math.ceil(sum(count_tokens(s.text) for s in original) * k)
    pre = [lines.line_keep(s) if s.kind == "code" else s for s in pool]
    rule, value = lines.FORCE[level]
    return lines.force_select(pre, budget, rule, value)


@dataclass
class Item:
    label: str
    prep: exi.Prepared
    original: list[Segment]
    ctx: dict[str, str]
    earlier_tokens: dict[str, int]
    seg_tokens: dict[str, int]
    budget: dict[str, int]
    pools: dict[str, list[Segment]]
    base: list[Segment]
    target: dict[str, int]
    hyb_placeholder: set[str]
    lsum_missing: list[str]


def build_item(
    cand: exi.Candidate, cut: exi.Cut, plan: Plan, slot: Any, slm: Any, prep_dir: Path
) -> Item:
    prep = exi.load_or_prepare(cand, cut, prep_dir)
    comp = Transcript(f"{cand.label}#compressible", list(cand.transcript.turns[: cut.tail_from]))
    original = segment(comp)
    total = sum(count_tokens(s.text) for s in original)
    base = apply_supersession(original)
    it = Item(cand.label, prep, original, {}, {}, {}, {}, {}, base, {}, set(), [])
    for fam in plan.families:
        if fam in PROSE_FAMILIES:
            it.pools[fam] = transform_pool(base, fam, slot, slm)
        elif fam == "line":
            it.pools[fam] = base
    it.ctx["full"] = exi.assemble("full", prep.full_earlier, prep.tail)
    it.earlier_tokens["full"] = count_tokens(prep.full_earlier)
    for level in plan.levels:
        budget = math.ceil(total * LEVELS[level])
        for fam in plan.families:
            if fam == "summary":
                it.budget[plan.cond(fam, level)] = count_tokens(prep.earlier[f"loosy-{level}"])
                continue
            cond = plan.cond(fam, level)
            it.budget[cond] = budget
            if fam in HYB_FAMILIES:
                it.target[cond] = round(HYB_FRAC[fam] * budget)
                continue
            if fam in LSUM_FAMILIES:
                it.budget[cond] = count_tokens(prep.earlier[f"loosy-{level}"])
                it.target[cond] = it.budget[cond]
                text = lsum.cached_text(
                    lsum_dir(fam),
                    cand.label,
                    level,
                    prep.full_earlier,
                    it.target[cond],
                    lsum.LSUM_MODELS[fam],
                )
                if text is None:
                    it.lsum_missing.append(cond)
                else:
                    it.ctx[cond] = exi.assemble("summary", text, prep.tail)
                    it.earlier_tokens[cond] = count_tokens(text)
                continue
            if fam == "loosy":
                it.ctx[cond] = exi.assemble("loosy", prep.earlier[f"loosy-{level}"], prep.tail)
                it.earlier_tokens[cond] = count_tokens(prep.earlier[f"loosy-{level}"])
                it.seg_tokens[cond] = prep.kept_segment_tokens[f"loosy-{level}"]
                continue
            if fam == "chunk":
                kept = chunked.chunked_select(base, budget, loosy_select)
                earlier = exi.render_segments(kept)
                it.ctx[cond] = exi.assemble("loosy", earlier, prep.tail)
                it.earlier_tokens[cond] = count_tokens(earlier)
                it.seg_tokens[cond] = sum(count_tokens(s.text) for s in kept)
                continue
            if fam == "gloosy":
                kept = loosy_select(it.pools[fam], budget)
            else:
                kept = kept_segments(original, level, it.pools[fam])
            earlier = exi.render_segments(kept)
            hdr = header_for(fam, slot)
            it.ctx[cond] = f"{hdr}\n\n{earlier}\n\n{exi.TAIL_HEADER}\n\n{prep.tail}"
            it.earlier_tokens[cond] = count_tokens(earlier)
            it.seg_tokens[cond] = sum(count_tokens(s.text) for s in kept)
    return it


def clip_tokens(text: str, n: int) -> str:
    return exi.truncate_tokens(text, max(n, 0) / exi.SUMMARY_OVERRUN)[0] if n > 0 else ""


def finish_hyb(it: Item, plan: Plan, store: exi.Store, placeholder: bool) -> None:
    """Builds the hybrid contexts once each summary exists: header, summary block, then the shipped
    loosy selection at (budget - actual summary tokens), in transcript order, then the tail.
    `placeholder` substitutes a target-length slice of the transcript for a missing summary so a
    dry run can check the token arithmetic; such contexts are not for answering."""
    for fam in HYB_FAMILIES:
        for lv in plan.levels:
            if fam not in plan.families:
                continue
            cond = plan.cond(fam, lv)
            text = summary_text(store, it, cond)
            it.hyb_placeholder.discard(cond)
            if text is None:
                if not placeholder:
                    it.ctx.pop(cond, None)
                    continue
                text = exi.truncate_tokens(it.prep.full_earlier, it.target[cond])[0]
                it.hyb_placeholder.add(cond)
            budget = it.budget[cond]
            summary = text
            while count_tokens(summary) > budget:
                summary = clip_tokens(summary, budget - 1)
            s_tok = count_tokens(summary)
            kept = loosy_select(it.base, budget - s_tok) if budget - s_tok > 0 else []
            earlier = (
                f"{HYB_SUMMARY_LABEL}\n{summary}\n\n{HYB_VERBATIM_LABEL}\n"
                f"{exi.render_segments(kept)}"
            )
            it.ctx[cond] = f"{HYB_HEADER}\n\n{earlier}\n\n{exi.TAIL_HEADER}\n\n{it.prep.tail}"
            it.earlier_tokens[cond] = count_tokens(earlier)
            it.seg_tokens[cond] = s_tok + sum(count_tokens(x.text) for x in kept)


# --------------------------------------------------------------------------- jobs


def probes_for(store: exi.Store, label: str, gen: exi.Model) -> list[dict[str, str]] | None:
    text = exi._text(store, label, "-", gen.name, "probe_gen")
    return None if text is None else exi.parse_probes(text)


def summary_text(store: exi.Store, it: Item, cond: str) -> str | None:
    text = exi._text(store, it.label, cond, SONNET.name, "summary")
    budget = it.target.get(cond, it.budget[cond])
    if text is not None and exi.summary_ok(text, budget):
        return exi.truncate_tokens(text.strip(), budget)[0]
    return None


def summary_job(it: Item, cond: str) -> exi.Job:
    budget = it.target.get(cond, it.budget[cond])
    return exi.Job(
        it.label,
        cond,
        SONNET,
        "summary",
        exi.summary_prompt(it.prep.full_earlier, budget),
        lambda t: exi.summary_ok(t, budget),
    )


def answer_job(
    label: str,
    cond: str,
    context: str,
    probes: list[dict[str, str]],
    answerer: exi.Model = HAIKU,
) -> exi.Job:
    n = len(probes)
    return exi.Job(
        label,
        cond,
        answerer,
        "probe",
        exi.probe_answer_prompt(context, [x["q"] for x in probes]),
        lambda t, n=n: exi.parse_answers(t, n) is not None,  # type: ignore[misc]
    )


def grade_job(
    store: exi.Store,
    label: str,
    cond: str,
    probes: list[dict[str, str]],
    answerer: exi.Model = HAIKU,
    grader: exi.Model = SONNET,
) -> exi.Job | None:
    text = exi._text(store, label, cond, answerer.name, "probe")
    if text is None:
        return None
    n = len(probes)
    answers = exi.parse_answers(text, n) or [""] * n
    items = [(x["q"], x["a"], a or "(no answer)") for x, a in zip(probes, answers, strict=True)]
    return exi.Job(
        label,
        cond,
        answerer,
        "grade_probe" if grader is SONNET else f"grade_probe@{grader.name}",
        exi.grade_probe_prompt(items),
        lambda t, n=n: exi.parse_probe_grades(t, n) is not None,  # type: ignore[misc]
        caller=grader,
    )


PENDING: dict[str, int] | None = None


def family_of(cond: str) -> str:
    return "full" if cond in ("-", "full") else cond.split(".")[0].split("-")[0]


def run_jobs(rt: exi.Runtime, name: str, jobs: list[exi.Job]) -> None:
    if PENDING is not None:
        for j in jobs:
            if not j.done_by(rt.store.get(j.key)):
                k = f"{j.task}:{family_of(j.condition)}"
                PENDING[k] = PENDING.get(k, 0) + 1
        return
    todo = sum(1 for j in jobs if not j.done_by(rt.store.get(j.key)))
    log(f"{name}: {len(jobs)} jobs, {todo} to run")
    if todo and not STOP.is_set():
        rt.run(jobs)


def valid_indices(store: exi.Store, label: str, n: int) -> list[int]:
    text = exi._text(store, label, "full", HAIKU.name, "grade_probe")
    g = exi.parse_probe_grades(text, n) if text else None
    return [i for i, x in enumerate(g or []) if x == 1.0]


def run_all(rt: exi.Runtime, items: list[Item], plan: Plan, gen: exi.Model) -> None:
    store = rt.store
    run_jobs(
        rt,
        "probe generation",
        [
            exi.Job(
                it.label,
                "-",
                gen,
                "probe_gen",
                exi.probe_gen_prompt(it.prep.full_earlier),
                lambda t: exi.parse_probes(t) is not None,
            )
            for it in items
        ],
    )
    probes = {it.label: probes_for(store, it.label, gen) for it in items}
    live = [it for it in items if probes[it.label]]
    sum_fams = [f for f in plan.families if f == "summary" or f in HYB_FAMILIES]
    if sum_fams:
        sj = [
            summary_job(it, plan.cond(f, lv)) for f in sum_fams for it in live for lv in plan.levels
        ]
        run_jobs(rt, "summaries", sj)
    run_jobs(
        rt,
        "full answers (Haiku)",
        [answer_job(it.label, "full", it.ctx["full"], probes[it.label] or []) for it in live],
    )
    run_jobs(
        rt,
        "full grading",
        [j for it in live if (j := grade_job(store, it.label, "full", probes[it.label] or []))],
    )
    valid = [it for it in live if valid_indices(store, it.label, len(probes[it.label] or []))]
    log(f"{len(valid)}/{len(items)} transcripts have a valid probe")
    for it in valid:
        finish_hyb(it, plan, store, placeholder=False)
    conds: list[tuple[Item, str, str]] = []
    for it in valid:
        for lv in plan.levels:
            for fam in plan.families:
                cond = plan.cond(fam, lv)
                if fam == "summary":
                    s = summary_text(store, it, cond)
                    if s is None:
                        continue
                    ctx = exi.assemble("summary", s, it.prep.tail)
                elif cond in it.ctx:
                    ctx = it.ctx[cond]
                else:
                    continue
                conds.append((it, cond, ctx))
    run_jobs(
        rt,
        "answers (Haiku)",
        [answer_job(it.label, c, ctx, probes[it.label] or []) for it, c, ctx in conds],
    )
    run_jobs(
        rt,
        "grading (Sonnet)",
        [j for it, c, _ in conds if (j := grade_job(store, it.label, c, probes[it.label] or []))],
    )


def precompute_lsum(items: list[Item], plan: Plan, models: list[str]) -> None:
    for fam in models:
        model_id = lsum.LSUM_MODELS[fam]
        cache = lsum_dir(fam)
        jobs = [
            (it.label, lv, it.prep.full_earlier, it.target[plan.cond(fam, lv)])
            for it in items
            for lv in plan.levels
        ]
        todo = [j for j in jobs if lsum.cached_text(cache, *j, model_id) is None]
        log(f"{fam}: {len(jobs)} generations, {len(todo)} to run ({model_id}, mlx-lm, greedy)")
        stats = lsum.precompute(jobs, cache, exi.summary_prompt, log, model_id=model_id)
        if stats.calls:
            log(
                f"{fam} throughput: prefill {stats.prompt_tps:.0f} tok/s "
                f"({stats.prompt_tokens} tok), decode {stats.gen_tps:.1f} tok/s "
                f"({stats.gen_tokens} tok), {stats.calls} model calls, "
                f"peak Metal memory {stats.peak_bytes / 1e9:.1f} GB"
            )
            all_prompt = sum(count_tokens(j[2]) for j in jobs)
            all_target = sum(j[3] for j in jobs)
            ratio = stats.gen_tokens / max(sum(j[3] for j in todo), 1)
            est = all_prompt / max(stats.prompt_tps, 1) + ratio * all_target / max(stats.gen_tps, 1)
            log(
                f"{fam} estimated total for all {len(jobs)} generations: ~{est / 60:.0f} min "
                f"({all_prompt} prompt tok, ~{ratio * all_target:.0f} generated tok at the "
                f"measured {ratio:.2f} of target)"
            )
        free_mlx()


def free_mlx() -> None:
    """Drop the finished model's weights and KV cache before the next one loads (16 GB machine)."""
    import gc

    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


# --------------------------------------------------------------------------- planning


def call_plan(n: int, plan: Plan) -> dict[str, int]:
    fams = [f for f in plan.families if f != "summary"]
    summ = n * len(plan.levels) * sum(f == "summary" or f in HYB_FAMILIES for f in plan.families)
    compressed = n * len(plan.levels) * len(plan.families)
    return {
        "probe_gen": n,
        "summary_gen": summ,
        "full_answers": n,
        "full_grades": n,
        "compressed_answers": compressed,
        "compressed_grades": compressed,
        "_families_no_summary": len(fams),
    }


def print_plan(
    labels: Sequence[str], plan: Plan, max_calls: int, gen: exi.Model, set_name: str = "C"
) -> int:
    cp = call_plan(len(labels), plan)
    total = sum(v for k, v in cp.items() if not k.startswith("_"))
    log(f"set {set_name} ({len(labels)}): {', '.join(labels)}")
    log(f"probe generator: {gen.name}; answers: {HAIKU.name}; grader/summariser: {SONNET.name}")
    log(f"families={plan.families} levels={plan.levels}")
    for k, v in cp.items():
        if not k.startswith("_"):
            log(f"  {k:20s} {v}")
    probe_gen = cp["probe_gen"]
    new = [f for f in plan.families if f in NEW_FAMILIES]
    if new:
        n, lv = len(labels), len(plan.levels)
        gens = n * lv * sum(f in HYB_FAMILIES for f in new)
        log(
            f"  new arms {new}: {gens} hybrid summary gens + {2 * n * lv * len(new)} "
            f"answer/grade = {gens + 2 * n * lv * len(new)} Claude calls (upper bound)"
        )
    log(
        f"  {'TOTAL (upper bound)':20s} {total}   (cap --max-calls {max_calls}; "
        f"{total - probe_gen} Claude + {probe_gen} probe-gen)"
    )
    return total


# --------------------------------------------------------------------------- dry run


def dry_run(items: list[Item], plan: Plan, sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    log(
        "tokens: mean rendered earlier-stretch / mean kept-segment tokens vs budget (n over budget)"
    )
    log(f"{'condition':22s} {'earlier':>9s} {'segments':>9s} {'budget':>9s}  over")
    full = float(np.mean([it.earlier_tokens["full"] for it in items]))
    log(f"{'full':22s} {full:9.0f}")
    for lv in plan.levels:
        for fam in plan.families:
            cond = plan.cond(fam, lv)
            if fam == "summary":
                b = np.mean([it.budget[cond] for it in items])
                log(f"{cond:22s} {'(Sonnet)':>9s} {'':9s} {b:9.0f}  n/a")
                continue
            have = [it for it in items if cond in it.earlier_tokens]
            if fam in LSUM_FAMILIES:
                b = np.mean([it.budget[cond] for it in items])
                e = np.mean([it.earlier_tokens[cond] for it in have]) if have else float("nan")
                log(f"{cond:22s} {e:9.0f} {'':9s} {b:9.0f}  cached {len(have)}/{len(items)}")
                continue
            if fam in HYB_FAMILIES:
                ph = sum(cond in it.hyb_placeholder for it in items)
                note = f" ({ph} with placeholder summary at target length)" if ph else ""
                items_h = have
                e = np.mean([it.earlier_tokens[cond] for it in items_h])
                s = np.mean([it.seg_tokens[cond] for it in items_h])
                b = np.mean([it.budget[cond] for it in items_h])
                over = sum(it.seg_tokens[cond] > it.budget[cond] for it in items_h)
                log(f"{cond:22s} {e:9.0f} {s:9.0f} {b:9.0f}  {over}{note}")
                continue
            e = np.mean([it.earlier_tokens[cond] for it in items])
            s = np.mean([it.seg_tokens[cond] for it in items])
            b = np.mean([it.budget[cond] for it in items])
            over = sum(it.seg_tokens[cond] > it.budget[cond] for it in items)
            log(f"{cond:22s} {e:9.0f} {s:9.0f} {b:9.0f}  {over}")
    for fam in PROSE_FAMILIES:
        if fam in plan.families:
            before = after = 0
            for it in items:
                pool = apply_supersession(it.original)
                tp = [s for s in pool if s.kind in ("prose", "thinking")]
                slotted = {s.id: s for s in it.pools[fam]}
                before += sum(count_tokens(s.text) for s in tp)
                after += sum(count_tokens(slotted[s.id].text) for s in tp if s.id in slotted)
            log(f"{fam} prose/thinking tokens in pool: {before} -> {after} ({after / before:.2f}x)")
    for it in items[:2]:
        for cond, text in it.ctx.items():
            (sample_dir / f"{it.label}__{cond}.txt").write_text(text, encoding="utf-8")
    log(f"sample contexts in {sample_dir}")


# --------------------------------------------------------------------------- report


def sign_p(diffs: list[float]) -> float:
    nz = [d for d in diffs if abs(d) > 1e-12]
    if not nz:
        return 1.0
    k = sum(d > 0 for d in nz)
    n = len(nz)
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def per_transcript(store: exi.Store, gen: exi.Model, label: str, cond: str) -> float | None:
    probes = probes_for(store, label, gen)
    if not probes:
        return None
    n = len(probes)
    keep = valid_indices(store, label, n)
    if not keep:
        return None
    text = exi._text(store, label, cond, HAIKU.name, "grade_probe")
    g = exi.parse_probe_grades(text, n) if text else None
    return None if g is None else float(np.mean([g[i] for i in keep]))


def cell(vals: list[float]) -> str:
    if not vals:
        return "-"
    m, lo, hi = exi.bootstrap_ci(vals)
    return f"{m:.3f} [{lo:.2f},{hi:.2f}] n={len(vals)}"


def paired(a: dict[str, float], b: dict[str, float]) -> str:
    common = sorted(set(a) & set(b))
    if not common:
        return "-"
    d = [a[k] - b[k] for k in common]
    m, lo, hi = exi.bootstrap_ci(d)
    return f"{m:+.3f} [{lo:+.2f},{hi:+.2f}] sign p={sign_p(d):.3f} n={len(common)}"


D_PAIRS = (
    ("line", "loosy"),
    ("gloosy", "loosy"),
    ("gline", "line"),
    ("gline", "loosy"),
    ("summary", "loosy"),
    ("summary", "line"),
    ("summary", "gloosy"),
    ("summary", "gline"),
    ("chunk", "loosy"),
    ("chunk", "summary"),
    ("hyb70", "summary"),
    ("hyb50", "summary"),
    ("hyb70", "loosy"),
    ("hyb50", "loosy"),
    ("loosy", "lsum"),
    ("lsum", "summary"),
    ("loosy", "lsum4b"),
    ("loosy", "lsum8b"),
    ("lsum4b", "summary"),
    ("lsum8b", "summary"),
    ("lsum8b", "lsum"),
)


def pooled_line_loosy(
    plan: Plan, gen: exi.Model, d_store: exi.Store, d_labels: list[str]
) -> list[str]:
    """line - loosy over set C + D. The line-/loosy- condition keys come from the same pipeline in
    both sets; transcripts are disjoint, so the pooled n is the sum."""
    c_cfg = SETS["C"]
    if not (c_cfg.calls.exists() and c_cfg.selection.exists()):
        return []
    c_store = exi.Store(c_cfg.calls)
    c_labels = json.loads(c_cfg.selection.read_text(encoding="utf-8"))
    rows = [
        "",
        "Pooled line - loosy over sets C + D:",
        "",
        "| level | delta accuracy |",
        "|---|---|",
    ]
    for lv in plan.levels:
        got: dict[str, dict[str, float]] = {"line": {}, "loosy": {}}
        for st, labs in ((c_store, c_labels), (d_store, d_labels)):
            for fam in got:
                for lb in labs:
                    v = per_transcript(st, gen, lb, f"{fam}-{lv}")
                    if v is not None:
                        got[fam][lb] = v
        rows.append(f"| {lv} | {paired(got['line'], got['loosy'])} |")
    return rows


def report_plan(cfg: SetCfg, plan: Plan) -> Plan:
    """The report covers every arm of the set whatever --conditions selected, so a partial run
    cannot overwrite it with only those arms. Scores come from the checkpoint; only the condition
    names (and their version stamps) are needed, never the slot or SLM models themselves."""
    full = Plan(list(cfg.families), plan.levels, plan.slot_version, plan.slm_version)
    full.gentle_version = plan.gentle_version
    for attr, mod, name in (
        ("slot_version", "slot", "SLOT_VERSION"),
        ("slm_version", "slm_slot", "SLM_VERSION"),
        ("gentle_version", "slot", "GENTLE_VERSION"),
    ):
        if getattr(full, attr) is None:
            try:
                setattr(full, attr, getattr(importlib.import_module(mod), name, None))
            except ImportError:
                pass
    return full


def report(
    store: exi.Store,
    labels: list[str],
    plan: Plan,
    gen: exi.Model,
    items: list[Item] | None,
    set_name: str = "C",
) -> str:
    out = [
        f"# Slot-compression downstream QA, set {set_name}, written "
        f"{time.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    out.append(
        f"Haiku 5.5 probe QA accuracy, graded by Sonnet 5.5; probes written by {gen.name}, "
        "kept only if Haiku answers them correctly from the full context; mean [95% bootstrap CI "
        "over transcripts]."
    )
    if gen is not CODEX_PROBER:
        out.append(f"NOTE: probe generator fell back to {gen.name} (codex unavailable).")
    if STOP.is_set():
        out.append("NOTE: the driver stopped on a spend limit or --max-calls; tables are partial.")
    scores: dict[str, dict[str, float]] = {}
    conds = ["full"] + [plan.cond(f, lv) for lv in plan.levels for f in plan.families]
    for c in conds:
        scores[c] = {}
        for lb in labels:
            v = per_transcript(store, gen, lb, c)
            if v is not None:
                scores[c][lb] = v

    def tok(c: str) -> str:
        if not items:
            return "-"
        if c == "full":
            vals = [it.earlier_tokens["full"] for it in items]
        elif c.startswith("summary"):
            vals = [it.budget[c] for it in items if c in it.budget]
        else:
            vals = [it.earlier_tokens[c] for it in items if c in it.earlier_tokens]
        return f"{np.mean(vals):.0f}" if vals else "-"

    out += ["", "| condition | accuracy | mean earlier tokens |", "|---|---|---|"]
    out += [f"| {c} | {cell(list(scores[c].values()))} | {tok(c)} |" for c in conds]
    lsum_rows = [
        f"| {c} | {np.mean([it.earlier_tokens[c] for it in have]):.0f} | "
        f"{np.mean([it.target[c] for it in have]):.0f} | "
        f"{np.mean([it.earlier_tokens[c] / it.target[c] for it in have]):.2f} | {len(have)} |"
        for lv in plan.levels
        for f in plan.families
        if f in LSUM_FAMILIES
        for c in [plan.cond(f, lv)]
        if (have := [it for it in items or [] if c in it.earlier_tokens])
    ]
    if lsum_rows:
        out += [
            "",
            "Local summary fidelity (o200k tokens after sentence truncation; fidelity = mean of "
            "summary / target per transcript):",
            "",
            "| condition | mean summary tokens | mean target | fidelity | n |",
            "|---|---|---|---|---|",
            *lsum_rows,
        ]
    out += ["", "| paired difference | delta accuracy |", "|---|---|"]
    for lv in plan.levels:
        if set_name == "D":
            for a, b in D_PAIRS:
                if a in plan.families and b in plan.families:
                    ca, cb = plan.cond(a, lv), plan.cond(b, lv)
                    out.append(f"| {ca} - {cb} | {paired(scores[ca], scores[cb])} |")
            continue
        for fam in plan.families:
            c = plan.cond(fam, lv)
            for ref in ("line", "loosy") + (("summary",) if fam == "chunk" else ()):
                if ref in plan.families and fam != ref:
                    out.append(f"| {c} - {ref}-{lv} | {paired(scores[c], scores[f'{ref}-{lv}'])} |")
    if set_name == "D" and {"line", "loosy"} <= set(plan.families):
        out += pooled_line_loosy(plan, gen, store, labels)
    out.append("")
    out.append(f"Calls recorded in checkpoint: {len(store.records)}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the call plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="build contexts, no model calls")
    ap.add_argument("--set", dest="set_name", choices=sorted(SETS), default="C")
    ap.add_argument("--max-calls", type=int, default=None, help="default 450 (C) / 520 (D)")
    ap.add_argument("--conditions", default=None, help="default: the set's families")
    ap.add_argument("--levels", default=",".join(LEVEL_NAMES))
    ap.add_argument("--limit", type=int, default=None, help="first N transcripts of the set")
    ap.add_argument(
        "--precompute-lsum",
        action="store_true",
        help="generate and cache the local Qwen3 summaries (no Claude calls), then exit",
    )
    ap.add_argument(
        "--lsum-models",
        default=",".join(lsum.LSUM_MODELS),
        help="families to precompute with --precompute-lsum, one model loaded at a time",
    )
    ap.add_argument(
        "--pending", action="store_true", help="count new calls per task and arm; no calls"
    )
    ap.add_argument("--only", default=None, help="comma-separated transcript labels (smoke runs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe-model", default=None, help="override probe generator (model name)")
    args = ap.parse_args()

    cfg = SETS[args.set_name]
    fams = [f for f in (args.conditions or ",".join(cfg.families)).split(",") if f]
    if args.max_calls is None:
        args.max_calls = cfg.max_calls
        if args.set_name == "D" and {"lsum4b", "lsum8b"} <= set(fams):
            args.max_calls = LSUM_BIG_MAX_CALLS
    lsum_models = [f for f in args.lsum_models.split(",") if f]
    if bad_m := [f for f in lsum_models if f not in lsum.LSUM_MODELS]:
        raise SystemExit(f"unknown --lsum-models: {bad_m}")
    if args.precompute_lsum:
        fams = lsum_models
    if set(fams) & set(NEW_FAMILIES) and args.set_name != "D":
        raise SystemExit("hyb70/hyb50/lsum* are set D arms (--set D)")
    levels = [lv for lv in args.levels.split(",") if lv]
    if bad := [f for f in fams if f not in ALL_FAMILIES] + [
        lv for lv in levels if lv not in LEVELS
    ]:
        raise SystemExit(f"unknown condition/level: {bad}")
    slot = load_slot() if any(f in SLOT_MODE for f in fams) else None
    slm = load_slm() if "slme" in fams else None
    if slot is None and any(f in GENTLE_FAMILIES for f in fams):
        slot = load_slot()
    plan = Plan(
        fams,
        levels,
        getattr(slot, "SLOT_VERSION", None),
        getattr(slm, "SLM_VERSION", None),
        getattr(slot, "GENTLE_VERSION", None) if any(f in GENTLE_FAMILIES for f in fams) else None,
    )
    if any(f in GENTLE_FAMILIES for f in fams) and plan.gentle_version is None and not args.plan:
        raise SystemExit("gentle conditions need slot.GENTLE_VERSION (slot.py has no gentle yet)")

    if args.probe_model:
        gen = exi.MODEL_BY_NAME[args.probe_model]
    else:
        gen = CODEX_PROBER if shutil.which("codex") else HAIKU

    cfg.report.parent.mkdir(parents=True, exist_ok=True)
    guard_none = Guard(0).runner
    selected = select_set(args.set_name)
    cfg.selection.write_text(json.dumps([c.label for c, _ in selected]), encoding="utf-8")
    if args.limit:
        selected = selected[: args.limit]
    if args.only:
        keep = set(args.only.split(","))
        selected = [x for x in selected if x[0].label in keep]
    labels = [c.label for c, _ in selected]
    total = print_plan(labels, plan, args.max_calls, gen, args.set_name)
    if args.plan:
        return
    if total > args.max_calls and not (args.dry_run or args.pending or cfg.calls.exists()):
        raise SystemExit(f"planned {total} calls exceeds --max-calls {args.max_calls}")

    items = []
    for cand, cut in selected:
        items.append(build_item(cand, cut, plan, slot, slm, cfg.prep_dir))
        log(f"context {cand.label}")
    if args.precompute_lsum:
        precompute_lsum(items, plan, lsum_models)
        return
    store = exi.Store(cfg.calls)
    for it in items:
        finish_hyb(it, plan, store, placeholder=args.dry_run or args.pending)
    if args.pending:
        for it in items:
            for c in it.lsum_missing:
                stub = exi.truncate_tokens(it.prep.full_earlier, it.target[c])[0]
                it.ctx[c] = exi.assemble("summary", stub, it.prep.tail)
                it.earlier_tokens[c] = count_tokens(stub)
    if set(fams) & set(LSUM_FAMILIES) and not (args.dry_run or args.pending):
        if miss := [(it.label, c) for it in items for c in it.lsum_missing]:
            raise SystemExit(
                f"{len(miss)} lsum contexts not cached (e.g. {miss[0]}); "
                "run --precompute-lsum first"
            )
    if args.dry_run:
        dry_run(items, plan, OUT / "contexts" if args.set_name == "C" else OUT / "d" / "samples")
        return

    if args.pending:
        global PENDING
        PENDING = {}
        rt0 = exi.Runtime(store, guard_none, cfg.report.parent / "failures.jsonl", workers=1)
        run_all(rt0, items, plan, gen)
        for k in sorted(PENDING):
            log(f"pending {k:28s} {PENDING[k]}")
        log(f"pending total (excl. hyb answers/grades awaiting summaries): {sum(PENDING.values())}")
        return
    guard = Guard(args.max_calls)
    rt = exi.Runtime(
        store, guard.runner, cfg.report.parent / "failures.jsonl", workers=args.workers
    )
    rt.sleep = guarded_sleep
    rt.log = lambda m: print(m, flush=True)
    try:
        run_all(rt, items, plan, gen)
    finally:
        text = report(store, labels, report_plan(cfg, plan), gen, items, args.set_name)
        cfg.report.write_text(text, encoding="utf-8")
        print(text, flush=True)
        log(
            f"DONE: {rt.new_calls} new calls, {rt.failed} failed, runner invocations "
            f"{guard.used}/{guard.max_calls}, stop={STOP.is_set()}"
        )


if __name__ == "__main__":
    main()

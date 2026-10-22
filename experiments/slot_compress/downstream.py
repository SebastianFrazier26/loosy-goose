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

import lines  # noqa: E402

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
CODEX_PROBER = exi.MODEL_BY_NAME["gpt-6-sol"]
C_SEED = exi.SEED + 2
C_N = 10
D_SEED = exi.SEED + 3
D_N = 20
FAMILIES = ("loosy", "line", "slotd", "slote", "slme", "summary")
D_FAMILIES = ("loosy", "line", "gloosy", "gline", "summary")
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
        750,
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


def build_item(
    cand: exi.Candidate, cut: exi.Cut, plan: Plan, slot: Any, slm: Any, prep_dir: Path
) -> Item:
    prep = exi.load_or_prepare(cand, cut, prep_dir)
    comp = Transcript(f"{cand.label}#compressible", list(cand.transcript.turns[: cut.tail_from]))
    original = segment(comp)
    total = sum(count_tokens(s.text) for s in original)
    it = Item(cand.label, prep, original, {}, {}, {}, {}, {})
    base = apply_supersession(original)
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
            if fam == "loosy":
                it.ctx[cond] = exi.assemble("loosy", prep.earlier[f"loosy-{level}"], prep.tail)
                it.earlier_tokens[cond] = count_tokens(prep.earlier[f"loosy-{level}"])
                it.seg_tokens[cond] = prep.kept_segment_tokens[f"loosy-{level}"]
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


# --------------------------------------------------------------------------- jobs


def probes_for(store: exi.Store, label: str, gen: exi.Model) -> list[dict[str, str]] | None:
    text = exi._text(store, label, "-", gen.name, "probe_gen")
    return None if text is None else exi.parse_probes(text)


def summary_text(store: exi.Store, it: Item, cond: str) -> str | None:
    text = exi._text(store, it.label, cond, SONNET.name, "summary")
    budget = it.budget[cond]
    if text is not None and exi.summary_ok(text, budget):
        return exi.truncate_tokens(text.strip(), budget)[0]
    return None


def summary_job(it: Item, cond: str) -> exi.Job:
    budget = it.budget[cond]
    return exi.Job(
        it.label,
        cond,
        SONNET,
        "summary",
        exi.summary_prompt(it.prep.full_earlier, budget),
        lambda t: exi.summary_ok(t, budget),
    )


def answer_job(label: str, cond: str, context: str, probes: list[dict[str, str]]) -> exi.Job:
    n = len(probes)
    return exi.Job(
        label,
        cond,
        HAIKU,
        "probe",
        exi.probe_answer_prompt(context, [x["q"] for x in probes]),
        lambda t, n=n: exi.parse_answers(t, n) is not None,  # type: ignore[misc]
    )


def grade_job(
    store: exi.Store, label: str, cond: str, probes: list[dict[str, str]]
) -> exi.Job | None:
    text = exi._text(store, label, cond, HAIKU.name, "probe")
    if text is None:
        return None
    n = len(probes)
    answers = exi.parse_answers(text, n) or [""] * n
    items = [(x["q"], x["a"], a or "(no answer)") for x, a in zip(probes, answers, strict=True)]
    return exi.Job(
        label,
        cond,
        HAIKU,
        "grade_probe",
        exi.grade_probe_prompt(items),
        lambda t, n=n: exi.parse_probe_grades(t, n) is not None,  # type: ignore[misc]
        caller=SONNET,
    )


def run_jobs(rt: exi.Runtime, name: str, jobs: list[exi.Job]) -> None:
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
    if "summary" in plan.families:
        sj = [summary_job(it, plan.cond("summary", lv)) for it in live for lv in plan.levels]
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
                else:
                    ctx = it.ctx[cond]
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


# --------------------------------------------------------------------------- planning


def call_plan(n: int, plan: Plan) -> dict[str, int]:
    fams = [f for f in plan.families if f != "summary"]
    summ = n * len(plan.levels) if "summary" in plan.families else 0
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
            vals = [it.budget[c] for it in items]
        else:
            vals = [it.earlier_tokens[c] for it in items if c in it.earlier_tokens]
        return f"{np.mean(vals):.0f}" if vals else "-"

    out += ["", "| condition | accuracy | mean earlier tokens |", "|---|---|---|"]
    out += [f"| {c} | {cell(list(scores[c].values()))} | {tok(c)} |" for c in conds]
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
            for ref in ("line", "loosy"):
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
    ap.add_argument("--max-calls", type=int, default=None, help="default 450 (C) / 750 (D)")
    ap.add_argument("--conditions", default=None, help="default: the set's families")
    ap.add_argument("--levels", default=",".join(LEVEL_NAMES))
    ap.add_argument("--limit", type=int, default=None, help="first N transcripts of the set")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probe-model", default=None, help="override probe generator (model name)")
    args = ap.parse_args()

    cfg = SETS[args.set_name]
    if args.max_calls is None:
        args.max_calls = cfg.max_calls
    fams = [f for f in (args.conditions or ",".join(cfg.families)).split(",") if f]
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
    selected = select_set(args.set_name)
    cfg.selection.write_text(json.dumps([c.label for c, _ in selected]), encoding="utf-8")
    if args.limit:
        selected = selected[: args.limit]
    labels = [c.label for c, _ in selected]
    total = print_plan(labels, plan, args.max_calls, gen, args.set_name)
    if args.plan:
        return
    if total > args.max_calls and not args.dry_run:
        raise SystemExit(f"planned {total} calls exceeds --max-calls {args.max_calls}")

    items = []
    for cand, cut in selected:
        items.append(build_item(cand, cut, plan, slot, slm, cfg.prep_dir))
        log(f"context {cand.label}")
    if args.dry_run:
        dry_run(items, plan, OUT / "contexts" if args.set_name == "C" else OUT / "d" / "samples")
        return

    store = exi.Store(cfg.calls)
    guard = Guard(args.max_calls)
    rt = exi.Runtime(
        store, guard.runner, cfg.report.parent / "failures.jsonl", workers=args.workers
    )
    rt.sleep = guarded_sleep
    rt.log = lambda m: print(m, flush=True)
    try:
        run_all(rt, items, plan, gen)
    finally:
        text = report(store, labels, plan, gen, items, args.set_name)
        cfg.report.write_text(text, encoding="utf-8")
        print(text, flush=True)
        log(
            f"DONE: {rt.new_calls} new calls, {rt.failed} failed, runner invocations "
            f"{guard.used}/{guard.max_calls}, stop={STOP.is_set()}"
        )


if __name__ == "__main__":
    main()

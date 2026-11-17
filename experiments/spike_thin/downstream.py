"""Downstream rerun (2026-10-22) of the token-thinning spike, through Experiment I's runner.

Phase A: Experiment I's 20 transcripts. Adds thin-light/medium/heavy (Haiku answers, Sonnet
grades) and compares against Experiment I's already-graded Haiku cells, read from
experiments/output/exp_i/calls.jsonl (read-only).
Phase B: ~10 held-out swe-gym transcripts, the same selection rules on a different seed with the
first 20 excluded. Sonnet writes and filters probes as in Experiment I; Haiku answers full,
loosy-* and thin-*; Sonnet grades. No summaries.

  nohup caffeinate -i uv run python experiments/spike_thin/downstream.py \\
      >> experiments/output/spike_thin/downstream/run.log 2>&1 &

Resumable: every call is checkpointed in downstream/calls.jsonl, and reruns skip finished keys.
A spend/usage-limit error stops the driver, which then writes tables from whatever exists.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run  # noqa: E402
import surprisal  # noqa: E402
import thin  # noqa: E402

from experiments import exp_i_downstream as exi  # noqa: E402
from loosy_goose.rate import LEVELS  # noqa: E402
from loosy_goose.segment import Segment, segment  # noqa: E402
from loosy_goose.supersede import apply_supersession  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Transcript  # noqa: E402

OUT = run.OUT / "downstream"
EXP_I = run.EXP_I
HAIKU = exi.MODEL_BY_NAME["haiku-5.5"]
SONNET = exi.HELPER
THIN_CONDS = {"thin-light": "light", "thin-medium": "medium", "thin-heavy": "heavy"}
# The arm each level runs: (force-keep rule, value). code-keep forces every code skeleton;
# code-cap forces skeletons up to that share of the budget.
THIN_ARM = {"light": ("keep", 0.25), "medium": ("cap", 0.5), "heavy": ("cap", 0.3)}
THIN_HEADER = (
    "[Earlier part of the session, compressed: some passages were dropped, and long file and "
    "command-output dumps were reduced to the identifiers, paths, numbers and error lines they "
    "contained, so those dumps are NOT verbatim and are not valid code. Everything else that "
    "remains is verbatim]"
)
B_SEED = exi.SEED + 1
B_N = 10
STOP = threading.Event()
LIMIT = re.compile(r"spend.?limit|usage.?limit|credit|quota|billing", re.IGNORECASE)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def guarded_runner(model: exi.Model, prompt: str) -> exi.CallResult:
    if STOP.is_set():
        raise exi.CallError("halted by budget guard")
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


# --------------------------------------------------------------------------- thin contexts


def thin_segments(original: list[Segment], level: str) -> list[Segment]:
    """The spike's chosen arm at `level`, scored by the zero-parameter unigram scorer (every LM
    tied it on these arms)."""
    k = LEVELS[level]
    total = sum(count_tokens(s.text) for s in original)
    budget = math.ceil(total * k)
    words = thin.score_words(original, surprisal.unigram_token_bits)
    pool = apply_supersession(original)
    rule, value = THIN_ARM[level]
    p = {"pool": [s.id for s in pool], "original": original}
    only = {"code-keep>sel"} if rule == "keep" else {"code-cap>sel"}
    for _arm, sel_keep, kept in run.arm_outputs(
        p, pool, words, surprisal.unigram_token_bits, k, budget, only
    ):
        if math.isclose(sel_keep, value):
            return kept
    raise RuntimeError(f"no arm output for {level}")


def thin_context(original: list[Segment], level: str, tail: str) -> tuple[str, int]:
    kept = thin_segments(original, level)
    earlier = exi.render_segments(kept)
    return f"{THIN_HEADER}\n\n{earlier}\n\n{exi.TAIL_HEADER}\n\n{tail}", count_tokens(earlier)


# --------------------------------------------------------------------------- jobs


def probe_job(label: str, cond: str, model: exi.Model, context: str, probes: list[Any]) -> exi.Job:
    n = len(probes)
    return exi.Job(
        label,
        cond,
        model,
        "probe",
        exi.probe_answer_prompt(context, [x["q"] for x in probes]),
        lambda t, n=n: exi.parse_answers(t, n) is not None,  # type: ignore[misc]
    )


def grade_jobs(
    store: exi.Store, label: str, conds: Sequence[str], model: exi.Model, probes: list[Any]
) -> list[exi.Job]:
    jobs = []
    n = len(probes)
    for cond in conds:
        text = exi._text(store, label, cond, model.name, "probe")
        if text is None:
            continue
        answers = exi.parse_answers(text, n) or [""] * n
        items = [(x["q"], x["a"], a or "(no answer)") for x, a in zip(probes, answers, strict=True)]
        jobs.append(
            exi.Job(
                label,
                cond,
                model,
                "grade_probe",
                exi.grade_probe_prompt(items),
                lambda t, n=n: exi.parse_probe_grades(t, n) is not None,  # type: ignore[misc]
                caller=SONNET,
            )
        )
    return jobs


def run_jobs(rt: exi.Runtime, name: str, jobs: list[exi.Job]) -> None:
    todo = sum(1 for j in jobs if not j.done_by(rt.store.get(j.key)))
    log(f"{name}: {len(jobs)} jobs, {todo} to run")
    if not STOP.is_set():
        rt.run(jobs)


# --------------------------------------------------------------------------- phase A


def phase_a(rt: exi.Runtime, exp_store: exi.Store) -> dict[str, Any]:
    data: dict[str, Any] = pickle.loads(run.PREP.read_bytes())
    ctx: dict[str, dict[str, str]] = {}
    tokens: dict[str, dict[str, int]] = {}
    for label, p in data.items():
        d = json.loads((EXP_I / "contexts" / f"{label}.json").read_text(encoding="utf-8"))
        ctx[label], tokens[label] = {}, {}
        for cond, level in THIN_CONDS.items():
            ctx[label][cond], tokens[label][cond] = thin_context(p["original"], level, d["tail"])
        for cond in ("full", *exi.LOOSY_LEVELS):
            tokens[label][cond] = count_tokens(d["earlier"][cond])
    probes = {label: exi.probes_for(exp_store, label) or [] for label in data}
    jobs = [
        probe_job(label, cond, HAIKU, ctx[label][cond], probes[label])
        for label in data
        for cond in THIN_CONDS
        if probes[label]
    ]
    run_jobs(rt, "A answers", jobs)
    grades = [
        j
        for label in data
        if probes[label]
        for j in grade_jobs(rt.store, label, list(THIN_CONDS), HAIKU, probes[label])
    ]
    run_jobs(rt, "A grading", grades)
    return {"labels": list(data), "tokens": tokens}


# --------------------------------------------------------------------------- phase B


def select_b() -> list[tuple[exi.Candidate, exi.Cut]]:
    taken = {p.stem for p in (EXP_I / "contexts").glob("*.json")}
    cands = (c for c in exi.iter_candidates() if c.source == "swe-gym" and c.label not in taken)
    return exi.select_transcripts(cands, n=B_N, per_source=B_N, seed=B_SEED)


def phase_b(rt: exi.Runtime) -> dict[str, Any]:
    sel_path = OUT / "b_selection.json"
    selected = select_b()
    sel_path.write_text(json.dumps([c.label for c, _ in selected]), encoding="utf-8")
    log(f"B selected: {', '.join(c.label for c, _ in selected)}")
    prepared = [exi.load_or_prepare(c, cut, OUT / "b") for c, cut in selected]
    cands = {c.label: (c, cut) for c, cut in selected}
    ctx: dict[str, dict[str, str]] = {}
    tokens: dict[str, dict[str, int]] = {}
    for p in prepared:
        cand, cut = cands[p.label]
        compressible = Transcript(
            f"{p.label}#compressible", list(cand.transcript.turns[: cut.tail_from])
        )
        original = segment(compressible)
        ctx[p.label] = {"full": exi.assemble("full", p.full_earlier, p.tail)}
        tokens[p.label] = {"full": count_tokens(p.full_earlier)}
        for cond in exi.LOOSY_LEVELS:
            ctx[p.label][cond] = exi.assemble("loosy", p.earlier[cond], p.tail)
            tokens[p.label][cond] = count_tokens(p.earlier[cond])
        for cond, level in THIN_CONDS.items():
            ctx[p.label][cond], tokens[p.label][cond] = thin_context(original, level, p.tail)
    gen = [
        exi.Job(
            p.label,
            "-",
            SONNET,
            "probe_gen",
            exi.probe_gen_prompt(p.full_earlier),
            lambda t: exi.parse_probes(t) is not None,
        )
        for p in prepared
    ]
    run_jobs(rt, "B probe generation", gen)
    probes = {p.label: exi.probes_for(rt.store, p.label) or [] for p in prepared}
    # Sonnet's full-context answers are Experiment I's probe filter.
    filt = [
        probe_job(p.label, "full", SONNET, ctx[p.label]["full"], probes[p.label])
        for p in prepared
        if probes[p.label]
    ]
    run_jobs(rt, "B filter answers (Sonnet, full)", filt)
    run_jobs(
        rt,
        "B filter grading",
        [
            j
            for p in prepared
            if probes[p.label]
            for j in grade_jobs(rt.store, p.label, ["full"], SONNET, probes[p.label])
        ],
    )
    conds = ["full", *exi.LOOSY_LEVELS, *THIN_CONDS]
    answers = [
        probe_job(p.label, cond, HAIKU, ctx[p.label][cond], probes[p.label])
        for p in prepared
        if probes[p.label]
        for cond in conds
    ]
    run_jobs(rt, "B answers (Haiku)", answers)
    grades = [
        j
        for p in prepared
        if probes[p.label]
        for j in grade_jobs(rt.store, p.label, conds, HAIKU, probes[p.label])
    ]
    run_jobs(rt, "B grading", grades)
    return {"labels": [p.label for p in prepared], "tokens": tokens}


# --------------------------------------------------------------------------- tables


def per_transcript(
    stores: Sequence[exi.Store], filter_store: exi.Store, label: str, cond: str, model: str
) -> float | None:
    probes = exi.probes_for(filter_store, label)
    if not probes:
        return None
    n = len(probes)
    ref = exi._text(filter_store, label, "full", SONNET.name, "grade_probe")
    ref_g = exi.parse_probe_grades(ref, n) if ref else None
    keep = [i for i, g in enumerate(ref_g or []) if g == 1.0]
    if not keep:
        return None
    for st in stores:
        text = exi._text(st, label, cond, model, "grade_probe")
        g = exi.parse_probe_grades(text, n) if text else None
        if g is not None:
            return float(np.mean([g[i] for i in keep]))
    return None


def cell(vals: list[float]) -> str:
    if not vals:
        return "—"
    m, lo, hi = exi.bootstrap_ci(vals)
    return f"{m:.3f} [{lo:.2f},{hi:.2f}] n={len(vals)}"


def paired(a: dict[str, float], b: dict[str, float]) -> str:
    common = sorted(set(a) & set(b))
    if not common:
        return "—"
    m, lo, hi = exi.bootstrap_ci([a[k] - b[k] for k in common])
    return f"{m:+.3f} [{lo:+.2f},{hi:+.2f}] n={len(common)}"


def tables(
    store: exi.Store, exp_store: exi.Store, a: dict[str, Any] | None, b: dict[str, Any] | None
) -> str:
    lines = [f"Token-thinning downstream rerun, written {time.strftime('%Y-%m-%d %H:%M')}"]
    lines.append("Haiku 5.5 probe QA accuracy, graded by Sonnet 5.5; probes Sonnet answers")
    lines.append("correctly from full context; mean [95% bootstrap CI over transcripts]")
    if STOP.is_set():
        lines.append("NOTE: the driver stopped on a spend/usage limit; tables are partial.")
    for name, info, filt, conds in (
        (
            "Phase A (Experiment I's 20 transcripts)",
            a,
            exp_store,
            ["full", *exi.LOOSY_LEVELS, *exi.SUMMARY_OF, *THIN_CONDS],
        ),
        (
            "Phase B (held-out swe-gym)",
            b,
            store,
            ["full", *exi.LOOSY_LEVELS, *THIN_CONDS],
        ),
    ):
        if info is None:
            continue
        labels = info["labels"]
        scores: dict[str, dict[str, float]] = {}
        for cond in conds:
            scores[cond] = {}
            for label in labels:
                v = per_transcript([store, exp_store], filt, label, cond, HAIKU.name)
                if v is not None:
                    scores[cond][label] = v
        lines += ["", f"## {name}", "", "| condition | accuracy | earlier-stretch tokens |"]
        lines.append("|---|---|---|")
        for cond in conds:
            toks = [info["tokens"][lb][cond] for lb in labels if cond in info["tokens"][lb]]
            lines.append(
                f"| {cond} | {cell(list(scores[cond].values()))} | {np.mean(toks):.0f} |"
                if toks
                else f"| {cond} | {cell(list(scores[cond].values()))} | (Exp I) |"
            )
        lines += ["", "| paired difference | Δ accuracy |", "|---|---|"]
        for level in ("light", "medium", "heavy"):
            t = scores[f"thin-{level}"]
            lines.append(
                f"| thin-{level} − loosy-{level} | {paired(t, scores[f'loosy-{level}'])} |"
            )
            if f"summary-{level}" in scores:
                lines.append(
                    f"| thin-{level} − summary-{level} | {paired(t, scores[f'summary-{level}'])} |"
                )
            lines.append(f"| thin-{level} − full | {paired(t, scores['full'])} |")
    new = [r for r in store.records.values()]
    lines += ["", f"New Claude calls recorded: {len(new)}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exp_store = exi.Store(EXP_I / "calls.jsonl")
    store = exi.Store(OUT / "calls.jsonl")
    rt = exi.Runtime(store, guarded_runner, OUT / "failures.jsonl", workers=4)
    rt.sleep = guarded_sleep
    rt.log = lambda m: print(m, flush=True)
    a = b = None
    try:
        log(">> phase A")
        a = phase_a(rt, exp_store)
        (OUT / "tables.md").write_text(tables(store, exp_store, a, None), encoding="utf-8")
        if not STOP.is_set():
            log(">> phase B")
            b = phase_b(rt)
    finally:
        text = tables(store, exp_store, a, b)
        (OUT / "tables.md").write_text(text, encoding="utf-8")
        print(text, flush=True)
        log(f"DOWNSTREAM DONE: {rt.new_calls} new calls, {rt.failed} failed, stop={STOP.is_set()}")


if __name__ == "__main__":
    main()

"""Downstream run of the line-keep variant (2026-10-22): Haiku answers, Sonnet grades, on Phase A's
20 and Phase B's 10 transcripts, reusing their probes, filters and graded baselines. Calls go to
the same checkpoint as downstream.py (keys differ by condition), so reruns skip finished work.

  nohup caffeinate -i uv run python experiments/spike_thin/downstream_lines.py \\
      >> experiments/output/spike_thin/downstream/run_lines.log 2>&1 &
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import downstream as ds  # noqa: E402
import lines  # noqa: E402
import run  # noqa: E402

from experiments import exp_i_downstream as exi  # noqa: E402
from loosy_goose.segment import segment  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Transcript  # noqa: E402

LINE_CONDS = {"line-light": "light", "line-medium": "medium", "line-heavy": "heavy"}
LINE_HEADER = (
    "[Earlier part of the session, compressed: some passages were dropped, and inside long file "
    "and command-output dumps only the lines naming identifiers, paths, numbers or errors were "
    "kept (gaps are marked with '… (N lines elided)'; line numbers were removed). Every line that "
    "remains is verbatim]"
)


def line_context(original: list[Any], level: str, tail: str) -> tuple[str, int]:
    earlier = exi.render_segments(lines.line_segments(original, level, "line"))
    return f"{LINE_HEADER}\n\n{earlier}\n\n{exi.TAIL_HEADER}\n\n{tail}", count_tokens(earlier)


def build(rt: exi.Runtime, exp_store: exi.Store) -> tuple[dict[str, Any], dict[str, Any]]:
    data: dict[str, Any] = pickle.loads(run.PREP.read_bytes())
    a_ctx: dict[str, dict[str, str]] = {}
    a_tok: dict[str, dict[str, int]] = {}
    for label, p in data.items():
        d = json.loads((ds.EXP_I / "contexts" / f"{label}.json").read_text(encoding="utf-8"))
        a_ctx[label], a_tok[label] = {}, {}
        for cond, level in LINE_CONDS.items():
            a_ctx[label][cond], a_tok[label][cond] = line_context(p["original"], level, d["tail"])
        for cond in ("full", *exi.LOOSY_LEVELS):
            a_tok[label][cond] = count_tokens(d["earlier"][cond])
        ds.log(f"A context {label}")
    b_ctx: dict[str, dict[str, str]] = {}
    b_tok: dict[str, dict[str, int]] = {}
    selected = ds.select_b()
    for cand, cut in selected:
        prep = exi.load_or_prepare(cand, cut, ds.OUT / "b")
        comp = Transcript(
            f"{cand.label}#compressible", list(cand.transcript.turns[: cut.tail_from])
        )
        original = segment(comp)
        b_ctx[cand.label], b_tok[cand.label] = {}, {"full": count_tokens(prep.full_earlier)}
        for cond in exi.LOOSY_LEVELS:
            b_tok[cand.label][cond] = count_tokens(prep.earlier[cond])
        for cond, level in LINE_CONDS.items():
            b_ctx[cand.label][cond], b_tok[cand.label][cond] = line_context(
                original, level, prep.tail
            )
        ds.log(f"B context {cand.label}")
    a = {"labels": list(data), "tokens": a_tok, "ctx": a_ctx}
    b = {"labels": [c.label for c, _ in selected], "tokens": b_tok, "ctx": b_ctx}
    return a, b


def phase(rt: exi.Runtime, name: str, info: dict[str, Any], probe_store: exi.Store) -> None:
    probes = {lb: exi.probes_for(probe_store, lb) or [] for lb in info["labels"]}
    jobs = [
        ds.probe_job(lb, cond, ds.HAIKU, info["ctx"][lb][cond], probes[lb])
        for lb in info["labels"]
        if probes[lb]
        for cond in LINE_CONDS
    ]
    ds.run_jobs(rt, f"{name} answers", jobs)
    grades = [
        j
        for lb in info["labels"]
        if probes[lb]
        for j in ds.grade_jobs(rt.store, lb, list(LINE_CONDS), ds.HAIKU, probes[lb])
    ]
    ds.run_jobs(rt, f"{name} grading", grades)


def tables(store: exi.Store, exp_store: exi.Store, a: dict[str, Any], b: dict[str, Any]) -> str:
    out = ["Line-keep downstream run: Haiku 5.5 probe QA accuracy, Sonnet 5.5 grading"]
    if ds.STOP.is_set():
        out.append("NOTE: stopped on a spend/usage limit; tables are partial.")
    for name, info, filt, base in (
        ("Phase A (Exp I's 20)", a, exp_store, ["full", *exi.LOOSY_LEVELS, *exi.SUMMARY_OF]),
        ("Phase B (held-out 10)", b, store, ["full", *exi.LOOSY_LEVELS]),
    ):
        conds = [*base, *ds.THIN_CONDS, *LINE_CONDS]
        scores: dict[str, dict[str, float]] = {}
        for cond in conds:
            scores[cond] = {}
            for lb in info["labels"]:
                v = ds.per_transcript([store, exp_store], filt, lb, cond, ds.HAIKU.name)
                if v is not None:
                    scores[cond][lb] = v
        out += ["", f"## {name}", "", "| condition | accuracy | earlier tokens |", "|---|---|---|"]
        for cond in conds:
            toks = [info["tokens"][lb][cond] for lb in info["labels"] if cond in info["tokens"][lb]]
            tk = f"{sum(toks) / len(toks):.0f}" if toks else "—"
            out.append(f"| {cond} | {ds.cell(list(scores[cond].values()))} | {tk} |")
        out += ["", "| paired | Δ accuracy |", "|---|---|"]
        for level in ("light", "medium", "heavy"):
            ln = scores[f"line-{level}"]
            for other in (f"loosy-{level}", f"thin-{level}", f"summary-{level}"):
                if other in scores:
                    out.append(f"| line-{level} − {other} | {ds.paired(ln, scores[other])} |")
    return "\n".join(out) + "\n"


def main() -> None:
    exp_store = exi.Store(ds.EXP_I / "calls.jsonl")
    store = exi.Store(ds.OUT / "calls.jsonl")
    rt = exi.Runtime(store, ds.guarded_runner, ds.OUT / "failures.jsonl", workers=4)
    rt.sleep = ds.guarded_sleep
    rt.log = lambda m: print(m, flush=True)
    a, b = build(rt, exp_store)
    try:
        ds.log(">> line phase A")
        phase(rt, "A", a, exp_store)
        if not ds.STOP.is_set():
            ds.log(">> line phase B")
            phase(rt, "B", b, store)
    finally:
        text = tables(store, exp_store, a, b)
        (ds.OUT / "tables_lines.md").write_text(text, encoding="utf-8")
        print(text, flush=True)
        ds.log(f"LINES DONE: {rt.new_calls} new calls, {rt.failed} failed, stop={ds.STOP.is_set()}")


if __name__ == "__main__":
    main()

"""Continuous-compaction pilot driver

Spec: docs/superpowers/specs/2026-11-16-continuous-compact-design.md.

Set E: 10 swe-gym transcripts (seed exi.SEED + 4), disjoint from Phase A/B and sets C/D. Arms full /
thresh / crec / cfrz are replayed at three checkpoints per transcript; Haiku answers probes written
by gpt-6-sol (no full-context filter), Sonnet grades and writes the thresh summaries.

  uv run python experiments/continuous/run.py --plan
  uv run python experiments/continuous/run.py --dry-run     # no model calls
  nohup caffeinate -i uv run python experiments/continuous/run.py \\
      >> experiments/output/continuous/run.log 2>&1 &
  uv run python experiments/continuous/run.py --report

Resumable: calls are checkpointed in output/continuous/calls.jsonl. A spend/usage-limit error or
--max-calls stops the driver, which then writes the report from whatever exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments import exp_i_downstream as exi  # noqa: E402
from experiments.continuous import sim  # noqa: E402
from experiments.slot_compress import downstream as dl  # noqa: E402

OUT = ROOT / "experiments" / "output" / "continuous"
DOCS_REPORT = ROOT / "docs" / "results" / "exp_continuous_pilot.md"
MAX_CALLS = 360
V = sim.CONT_VERSION
HAIKU, SONNET = dl.HAIKU, dl.SONNET
PAIRS = (
    ("cfrz", "thresh"),
    ("crec", "thresh"),
    ("cfrz", "crec"),
    ("thresh", "full"),
    ("crec", "full"),
    ("cfrz", "full"),
)
log = dl.log


def cond(arm: str) -> str:
    return f"{arm}.v{V}"


def ck_label(p: sim.TranscriptPlan, ck: sim.Checkpoint) -> str:
    return f"{p.label}@{ck.pct}"


def taken_labels() -> set[str]:
    taken = dl.exclusions()
    for name in ("C", "D"):
        path = dl.SETS[name].selection
        if not path.exists():
            raise SystemExit(f"{path} missing: run slot_compress/downstream.py --plan --set {name}")
        taken |= set(json.loads(path.read_text(encoding="utf-8")))
    return taken


def select_plans() -> list[sim.TranscriptPlan]:
    taken = taken_labels()
    selected = sim.select_set_e(exi.iter_candidates(), taken)
    labels = [c.label for c, _ in selected]
    if overlap := set(labels) & taken:
        raise SystemExit(f"set E overlaps earlier sets: {sorted(overlap)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "e_selection.json").write_text(json.dumps(labels), encoding="utf-8")
    return [sim.plan_transcript(c.label, c.transcript.turns) for c, _ in selected]


# --------------------------------------------------------------------------- summaries


def summary_job(label: str, k: int, text: str, budget: int) -> exi.Job:
    return exi.Job(
        label,
        f"thresh.v{V}.e{k}",
        SONNET,
        "summary",
        exi.summary_prompt(text, budget),
        lambda t: exi.summary_ok(t, budget),
    )


class Summaries:
    """Cached Sonnet summaries for one transcript. Strict raises on the first missing event so the
    caller can queue it; non-strict stands in a placeholder and records that it did."""

    def __init__(self, store: exi.Store, label: str, strict: bool) -> None:
        self.store, self.label, self.strict = store, label, strict
        self.used: list[str] = []
        self.placeholder = False

    def __call__(self, k: int, text: str, budget: int) -> str:
        job = summary_job(self.label, k, text, budget)
        rec = self.store.get(job.key)
        if job.done_by(rec):
            out = exi.truncate_tokens(str(rec["text"]).strip(), budget)[0] if rec else ""
        elif self.strict:
            raise sim.MissingSummaryError(k, text, budget)
        else:
            self.placeholder = True
            out = sim.placeholder_summary(k, text, budget)
        self.used.append(out)
        return out

    def sig(self) -> str:
        if self.placeholder:
            return "placeholder"
        return hashlib.sha256("\x00".join(self.used).encode("utf-8")).hexdigest()[:16]


def summary_rounds(rt: exi.Runtime, plans: list[sim.TranscriptPlan]) -> None:
    while not dl.STOP.is_set():
        jobs = []
        for p in plans:
            try:
                sim.thresh_events(p, Summaries(rt.store, p.label, strict=True))
            except sim.MissingSummaryError as m:
                jobs.append(summary_job(p.label, m.k, m.text, m.budget))
        if not jobs:
            return
        log(f"thresh summaries: {len(jobs)} jobs")
        rt.run(jobs)
        if not any(j.done_by(rt.store.get(j.key)) for j in jobs):
            return


# --------------------------------------------------------------------------- cached simulation


def cached_run(
    arm: str, p: sim.TranscriptPlan, sig: str, build: Callable[[], sim.ArmRun]
) -> sim.ArmRun:
    params = {
        "v": V,
        "keep": sim.KEEP,
        "freeze": sim.FREEZE_TOKENS,
        "cap": p.cap,
        "target": p.target,
        "n": len(p.turns),
        "total": p.total,
        "cks": [c.t for c in p.checkpoints],
        "sig": sig,
    }
    path = OUT / "sims" / f"{p.label}.{arm}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("params") == params:
            return sim.ArmRun.from_json(d["run"])
    run = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params, "run": run.to_json()}), encoding="utf-8")
    return run


def arm_run(
    arm: str, p: sim.TranscriptPlan, store: exi.Store | None, strict: bool = False
) -> tuple[sim.ArmRun, bool]:
    """The cached run for one arm; the flag is True when thresh used placeholder summaries."""
    if arm != "thresh":
        return cached_run(arm, p, "", lambda: sim.run_arm(arm, p, exi.loosy_compressor)), False
    sums = Summaries(store, p.label, strict) if store is not None else None
    if sums is None:
        run = cached_run(arm, p, "placeholder", lambda: sim.run_arm(arm, p, exi.loosy_compressor))
        return run, True
    # Run once up front to learn the signature; the cached run is keyed on it.
    probe = sim.run_arm(arm, p, exi.loosy_compressor, sums, keep_at=[c.t for c in p.checkpoints])
    run = cached_run(arm, p, sums.sig(), lambda: probe)
    return run, sums.placeholder


# --------------------------------------------------------------------------- planning


def call_counts(plans: list[sim.TranscriptPlan]) -> dict[str, int]:
    n_ck = sum(len(p.checkpoints) for p in plans)
    return {
        "probe_gen": n_ck,
        "thresh_summaries": sum(len(sim.thresh_events(p)) for p in plans),
        "answers": n_ck * len(sim.ARMS),
        "grades": n_ck * len(sim.ARMS),
    }


def print_plan(plans: list[sim.TranscriptPlan], gen: exi.Model, max_calls: int) -> int:
    log(f"set E ({len(plans)}): {', '.join(p.label for p in plans)}")
    log(f"probe generator: {gen.name}; answers: {HAIKU.name}; grader/summariser: {SONNET.name}")
    log(f"CONT_VERSION={V} KEEP={sim.KEEP} F={sim.FREEZE_TOKENS} cap={sim.CAP_SHARE} of total")
    log(
        f"{'transcript':20s}{'tokens':>8s}{'turns':>6s}{'C':>7s}{'target':>7s}  checkpoints  thresh"
    )
    for p in plans:
        cks = " ".join(f"{c.pct}%@t{c.t}" for c in p.checkpoints)
        ev = sim.thresh_events(p)
        at = ", ".join(f"t{e.t}" for e in ev)
        log(
            f"{p.label:20s}{p.total:8d}{len(p.turns):6d}{p.cap:7d}{p.target:7d}  {cks}"
            f"  {len(ev)} events ({at})"
        )
    counts = call_counts(plans)
    for k, v in counts.items():
        log(f"  {k:20s} {v}")
    total = sum(counts.values())
    log(f"  {'TOTAL (placeholder events)':26s} {total}   (cap --max-calls {max_calls})")
    return total


# --------------------------------------------------------------------------- dry run


def cost_table(plans: list[sim.TranscriptPlan], runs: dict[str, dict[str, sim.ArmRun]]) -> str:
    rows = [
        "Offline cost per transcript (mean context tokens / mean uncached tokens per model turn; "
        "thresh uses placeholder summaries of the target length where no real ones are cached)",
        f"{'transcript':20s}{'turns':>6s}" + "".join(f"{a:>20s}" for a in sim.ARMS),
    ]
    for p in plans:
        r = runs[p.label]
        rows.append(
            f"{p.label:20s}{len(r['full'].boundaries):6d}"
            + "".join(f"{r[a].mean_ctx():>11.0f}/{r[a].mean_uncached():<8.0f}" for a in sim.ARMS)
        )
    rows += [
        "",
        f"{'arm':8s}{'ctx/turn':>10s}{'uncached/turn':>15s}{'ctx sum':>12s}"
        f"{'uncached sum':>14s}{'events':>8s}",
    ]
    for a in sim.ARMS:
        rs = [runs[p.label][a] for p in plans]
        rows.append(
            f"{a:8s}{np.mean([r.mean_ctx() for r in rs]):>10.0f}"
            f"{np.mean([r.mean_uncached() for r in rs]):>15.0f}"
            f"{np.mean([sum(r.ctx_tokens) for r in rs]):>12.0f}"
            f"{np.mean([sum(r.uncached_tokens) for r in rs]):>14.0f}"
            f"{np.mean([len(r.events) for r in rs]):>8.1f}"
        )
    return "\n".join(rows)


def dry_run(plans: list[sim.TranscriptPlan]) -> None:
    runs: dict[str, dict[str, sim.ArmRun]] = {}
    for p in plans:
        runs[p.label] = {a: arm_run(a, p, None)[0] for a in sim.ARMS}
        log(f"simulated {p.label}")
    table = cost_table(plans, runs)
    (OUT / "dry_run").mkdir(parents=True, exist_ok=True)
    (OUT / "dry_run" / "cost.txt").write_text(table + "\n", encoding="utf-8")
    p = plans[0]
    ck = p.checkpoints[1]
    for a in sim.ARMS:
        (OUT / "dry_run" / f"{p.label}@{ck.pct}__{a}.txt").write_text(
            runs[p.label][a].contexts[ck.t], encoding="utf-8"
        )
    print(table, flush=True)
    log(f"sample contexts for {p.label}@{ck.pct} in {OUT / 'dry_run'}")


# --------------------------------------------------------------------------- jobs


def probes_for(store: exi.Store, label: str, gen: exi.Model) -> list[dict[str, str]] | None:
    text = exi._text(store, label, f"probes.v{V}", gen.name, "probe_gen")
    return None if text is None else exi.parse_probes(text)


def run_all(rt: exi.Runtime, plans: list[sim.TranscriptPlan], gen: exi.Model) -> None:
    store = rt.store
    gen_jobs = [
        exi.Job(
            ck_label(p, ck),
            f"probes.v{V}",
            gen,
            "probe_gen",
            exi.probe_gen_prompt(exi.render_turns(p.turns[: ck.tail_from])),
            lambda t: exi.parse_probes(t) is not None,
        )
        for p in plans
        for ck in p.checkpoints
    ]
    log(f"probe generation: {len(gen_jobs)} jobs")
    rt.run(gen_jobs)
    summary_rounds(rt, plans)
    answers: list[exi.Job] = []
    contexts: list[tuple[str, str, list[dict[str, str]]]] = []
    for p in plans:
        runs: dict[str, sim.ArmRun] = {}
        for a in sim.ARMS:
            try:
                runs[a], _ = arm_run(a, p, store, strict=True)
            except sim.MissingSummaryError:
                log(f"{p.label}: thresh summaries incomplete, skipping the arm")
        for ck in p.checkpoints:
            probes = probes_for(store, ck_label(p, ck), gen)
            if not probes:
                continue
            for a, r in runs.items():
                contexts.append((ck_label(p, ck), a, probes))
                answers.append(dl.answer_job(ck_label(p, ck), cond(a), r.contexts[ck.t], probes))
    log(f"answers (Haiku): {len(answers)} jobs")
    if not dl.STOP.is_set():
        rt.run(answers)
    grades = [j for lb, a, pr in contexts if (j := dl.grade_job(store, lb, cond(a), pr))]
    log(f"grading (Sonnet): {len(grades)} jobs")
    if not dl.STOP.is_set():
        rt.run(grades)


# --------------------------------------------------------------------------- report


def checkpoint_score(store: exi.Store, label: str, arm: str, gen: exi.Model) -> float | None:
    probes = probes_for(store, label, gen)
    if not probes:
        return None
    n = len(probes)
    text = exi._text(store, label, cond(arm), HAIKU.name, "grade_probe")
    grades = exi.parse_probe_grades(text, n) if text else None
    return None if grades is None else float(np.mean(grades))


def report(store: exi.Store, plans: list[sim.TranscriptPlan], gen: exi.Model) -> str:
    out = [
        f"# Continuous compaction pilot, set E, written {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"CONT_VERSION {V}. Haiku 5.5 probe QA accuracy, graded by Sonnet 5.5; probes by "
        f"{gen.name}, no full-context filter (every probe counts, for every arm). A transcript "
        "counts for an arm only if all three of its checkpoints are graded. Mean [95% bootstrap "
        "CI over transcripts].",
    ]
    if gen is not dl.CODEX_PROBER:
        out.append(f"NOTE: probe generator fell back to {gen.name} (codex unavailable).")
    if dl.STOP.is_set():
        out.append("NOTE: the driver stopped on a spend limit or --max-calls; tables are partial.")
    by_pct: dict[str, dict[int, dict[str, float]]] = {a: {} for a in sim.ARMS}
    scores: dict[str, dict[str, float]] = {a: {} for a in sim.ARMS}
    for a in sim.ARMS:
        for p in plans:
            got = {c.pct: checkpoint_score(store, ck_label(p, c), a, gen) for c in p.checkpoints}
            for pct, v in got.items():
                if v is not None:
                    by_pct[a].setdefault(pct, {})[p.label] = v
            if all(v is not None for v in got.values()):
                scores[a][p.label] = float(np.mean([v for v in got.values() if v is not None]))
    cost: dict[str, tuple[float, float]] = {}
    placeholder = False
    for a in sim.ARMS:
        rs = []
        for p in plans:
            r, ph = arm_run(a, p, store)
            rs.append(r)
            placeholder |= ph
        cost[a] = (
            float(np.mean([r.mean_ctx() for r in rs])),
            float(np.mean([r.mean_uncached() for r in rs])),
        )
    out += ["", "| arm | accuracy | mean context tokens/turn | mean uncached tokens/turn |"]
    out.append("|---|---|---|---|")
    out += [
        f"| {a} | {dl.cell(list(scores[a].values()))} | {cost[a][0]:.0f} | {cost[a][1]:.0f} |"
        for a in sim.ARMS
    ]
    if placeholder:
        out.append("")
        out.append("NOTE: thresh cost uses placeholder summaries where real ones are not cached.")
    out += ["", "| paired difference | delta accuracy |", "|---|---|"]
    out += [f"| {a} - {b} | {dl.paired(scores[a], scores[b])} |" for a, b in PAIRS]
    out += ["", "| arm | 50% | 75% | 100% |", "|---|---|---|---|"]
    out += [
        f"| {a} | "
        + " | ".join(dl.cell(list(by_pct[a].get(pct, {}).values())) for pct in (50, 75, 100))
        + " |"
        for a in sim.ARMS
    ]
    out += ["", f"Calls recorded in checkpoint: {len(store.records)}"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the call plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="offline cost metrics, no model calls")
    ap.add_argument("--report", action="store_true", help="write the report from the checkpoint")
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    gen = dl.CODEX_PROBER if shutil.which("codex") else HAIKU
    plans = select_plans()
    total = print_plan(plans, gen, args.max_calls)
    if args.plan:
        return
    if args.dry_run:
        dry_run(plans)
        return
    store = exi.Store(OUT / "calls.jsonl")
    if args.report:
        text = report(store, plans, gen)
        DOCS_REPORT.write_text(text, encoding="utf-8")
        (OUT / "report.md").write_text(text, encoding="utf-8")
        print(text, flush=True)
        return
    if total > args.max_calls and not (OUT / "calls.jsonl").exists():
        raise SystemExit(f"planned {total} calls exceeds --max-calls {args.max_calls}")
    guard = dl.Guard(args.max_calls)
    rt = exi.Runtime(store, guard.runner, OUT / "failures.jsonl", workers=args.workers)
    rt.sleep = dl.guarded_sleep
    rt.log = lambda m: print(m, flush=True)
    try:
        run_all(rt, plans, gen)
    finally:
        text = report(store, plans, gen)
        (OUT / "report.md").write_text(text, encoding="utf-8")
        print(text, flush=True)
        log(
            f"DONE: {rt.new_calls} new calls, {rt.failed} failed, runner invocations "
            f"{guard.used}/{guard.max_calls}, stop={dl.STOP.is_set()}"
        )


if __name__ == "__main__":
    main()

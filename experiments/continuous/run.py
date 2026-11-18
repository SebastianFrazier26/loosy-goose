"""Continuous-compaction driver

Spec: docs/superpowers/specs/2026-11-16-continuous-compact-design.md.

Set E (default, the pilot): 10 swe-gym transcripts (seed exi.SEED + 4), disjoint from Phase A/B and
sets C/D; arms full / thresh / crec / cfrz. Set F (--set F): 25 swe-gym transcripts (seed
exi.SEED + 5), disjoint from A-E, C = 65% of the transcript; arms full / thresh / cfrz / cfrzm
(cfrz at a calibrated higher keep) / cdec.
Arms are replayed at three checkpoints per transcript; Haiku answers probes written by gpt-6-sol
(no full-context filter), Sonnet grades and writes the thresh summaries.

  uv run python experiments/continuous/run.py --set F --plan
  uv run python experiments/continuous/run.py --set F --dry-run     # no model calls
  uv run python experiments/continuous/run.py --set F --calibrate   # no model calls
  nohup caffeinate -i uv run python experiments/continuous/run.py --set F \\
      >> experiments/output/continuous/f/run.log 2>&1 &
  uv run python experiments/continuous/run.py --set F --report

Open-weight replay (--answerer qwen3-8b, spec docs/superpowers/specs/2026-11-17-continuous-local-
design.md section A): set F's checkpoints with arms sim.Q_ARMS, all generation local through MLX.
Mistral-Nemo writes probes and grades, Qwen3-8B answers and writes the qthresh summaries; any
closed-model call raises. Phases run one model at a time, workers=1.

  uv run python experiments/continuous/run.py --set F --answerer qwen3-8b --plan
  caffeinate -i uv run python experiments/continuous/run.py --set F --answerer qwen3-8b \\
      >> experiments/output/continuous/f/qwen.log 2>&1
  uv run python experiments/continuous/run.py --set F --answerer qwen3-8b --report

Resumable: calls are checkpointed per set (E: output/continuous/calls.jsonl, F:
output/continuous/f/calls.jsonl). A spend/usage-limit error or --max-calls stops the driver, which
then writes the report from whatever exists.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments import exp_i_downstream as exi  # noqa: E402
from experiments.continuous import local, sim  # noqa: E402
from experiments.slot_compress import downstream as dl  # noqa: E402
from experiments.slot_compress import lsum  # noqa: E402

OUT = ROOT / "experiments" / "output" / "continuous"
V = sim.CONT_VERSION
HAIKU, SONNET, NEMO = dl.HAIKU, dl.SONNET, dl.NEMO
QWEN = exi.MODEL_BY_NAME["qwen3-8b"]
QSUM_DIR = OUT / "f" / "qsum_cache"
QSUM_MODEL_ID = lsum.LSUM_MODELS["lsum8b"]
SUMMARY_ARMS = ("thresh", "qthresh")


@dataclass(frozen=True)
class SetCfg:
    name: str
    arms: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    out: Path
    report: Path
    max_calls: int
    cap_share: float
    n: int
    tag: str  # in every record key; empty for E so its stored records keep their keys
    answerer: exi.Model = HAIKU
    grader: exi.Model = SONNET
    prober: exi.Model | None = None  # None: gpt-6-sol when codex is installed, else Haiku
    probe_cond: str = ""  # empty: probes{tag}.v{V}
    prefix: str = ""  # on the dry-run directory and report.md, so a variant keeps its own

    @property
    def selection(self) -> Path:
        return OUT / f"{self.name.lower()}_selection.json"

    @property
    def calls(self) -> Path:
        return self.out / "calls.jsonl"

    @property
    def probes_key(self) -> str:
        return self.probe_cond or f"probes{self.tag}.v{V}"

    @property
    def grade_task(self) -> str:
        return "grade_probe" if self.grader is SONNET else f"grade_probe@{self.grader.name}"

    @property
    def summary_arm(self) -> str | None:
        return next((a for a in SUMMARY_ARMS if a in self.arms), None)


CFGS = {
    "E": SetCfg(
        "E",
        sim.ARMS,
        (
            ("cfrz", "thresh"),
            ("crec", "thresh"),
            ("cfrz", "crec"),
            ("thresh", "full"),
            ("crec", "full"),
            ("cfrz", "full"),
        ),
        OUT,
        ROOT / "docs" / "results" / "exp_continuous_pilot.md",
        360,
        sim.CAP_SHARE,
        sim.E_N,
        "",
    ),
    "F": SetCfg(
        "F",
        sim.F_ARMS,
        (
            ("cdec", "thresh"),
            ("cfrz", "thresh"),
            ("cfrzm", "thresh"),
            ("cdec", "cfrzm"),
            ("cfrzm", "tloosy"),
            ("cdec", "tloosy"),
            ("thresh", "tloosy"),
            ("tloosy", "full"),
            ("thresh", "full"),
            ("cfrz", "full"),
            ("cfrzm", "full"),
            ("cdec", "full"),
        ),
        OUT / "f",
        ROOT / "docs" / "results" / "exp_continuous_f.md",
        1000,
        sim.F_CAP_SHARE,
        sim.F_N,
        ".F",
    ),
}
# Shares set F's selection, simulations and checkpoint file; its records differ by model name.
# The call cap only guards against a runaway loop (about 875 generations plus retries): every call
# is local, so there is no spend to protect.
QWEN_CFG = dataclasses.replace(
    CFGS["F"],
    arms=sim.Q_ARMS,
    pairs=(
        ("qthresh", "tloosy"),
        ("cfrzm", "qthresh"),
        ("cfrzm", "tloosy"),
        ("cfrzg", "cfrzm"),
        ("qthresh", "full"),
        ("tloosy", "full"),
        ("cfrzm", "full"),
        ("cfrzg", "full"),
    ),
    report=ROOT / "docs" / "results" / "exp_continuous_qwen.md",
    max_calls=2000,
    answerer=QWEN,
    grader=NEMO,
    prober=NEMO,
    probe_cond="probes.nemo1",
    prefix="qwen_",
)
ANSWERER_CFGS = {"qwen3-8b": QWEN_CFG}
# The report flags qthresh when its mean context is further than this share from thresh's real
# (Sonnet-summary) mean on set F: qthresh finds its own events, so its size is not pinned.
QTHRESH_TOL = 0.10
log = dl.log


def cond(cfg: SetCfg, arm: str) -> str:
    if arm == "qthresh":
        return f"qthresh{cfg.tag}.v{V}.q{sim.QTHRESH_VERSION}"
    if arm == "cfrzg":
        return f"cfrzg{cfg.tag}.v{V}.q{sim.CFRZG_VERSION}"
    if arm == "cfrzm":
        return f"cfrzm{cfg.tag}.v{V}.k{sim.FRZM_KEEP}"
    if arm == "cdec":
        return f"cdec{cfg.tag}.v{V}.d{sim.DEC_VERSION}.k{sim.DEC_KEEP}.m{sim.DEC_M}"
    if arm == "tloosy":
        return f"tloosy{cfg.tag}.v{V}.t{sim.TLOOSY_VERSION}.s{sim.TLOOSY_SHARE}"
    return f"{arm}{cfg.tag}.v{V}"


def ck_label(p: sim.TranscriptPlan, ck: sim.Checkpoint) -> str:
    return f"{p.label}@{ck.pct}"


def taken_labels(cfg: SetCfg) -> set[str]:
    taken = dl.exclusions()
    paths = [dl.SETS[name].selection for name in ("C", "D")]
    if cfg.name == "F":
        paths.append(CFGS["E"].selection)
    for path in paths:
        if not path.exists():
            raise SystemExit(f"{path} missing: run the earlier set's --plan first")
        taken |= set(json.loads(path.read_text(encoding="utf-8")))
    return taken


def select_plans(cfg: SetCfg) -> list[sim.TranscriptPlan]:
    taken = taken_labels(cfg)
    pick = sim.select_set_f if cfg.name == "F" else sim.select_set_e
    selected = pick(exi.iter_candidates(), taken)
    labels = [c.label for c, _ in selected]
    if overlap := set(labels) & taken:
        raise SystemExit(f"set {cfg.name} overlaps earlier sets: {sorted(overlap)}")
    cfg.selection.parent.mkdir(parents=True, exist_ok=True)
    cfg.selection.write_text(json.dumps(labels), encoding="utf-8")
    return [sim.plan_transcript(c.label, c.transcript.turns, cfg.cap_share) for c, _ in selected]


# --------------------------------------------------------------------------- summaries


def summary_job(cfg: SetCfg, label: str, k: int, text: str, budget: int) -> exi.Job:
    return exi.Job(
        label,
        f"thresh{cfg.tag}.v{V}.e{k}",
        SONNET,
        "summary",
        exi.summary_prompt(text, budget),
        lambda t: exi.summary_ok(t, budget),
    )


class Summaries:
    """Cached Sonnet summaries for one transcript. Strict raises on the first missing event so the
    caller can queue it; non-strict stands in a placeholder and records that it did."""

    def __init__(self, cfg: SetCfg, store: exi.Store, label: str, strict: bool) -> None:
        self.cfg, self.store, self.label, self.strict = cfg, store, label, strict
        self.used: list[str] = []
        self.placeholder = False

    def lookup(self, k: int, text: str, budget: int) -> str | None:
        job = summary_job(self.cfg, self.label, k, text, budget)
        rec = self.store.get(job.key)
        if rec is None or not job.done_by(rec):
            return None
        return exi.truncate_tokens(str(rec["text"]).strip(), budget)[0]

    def __call__(self, k: int, text: str, budget: int) -> str:
        out = self.lookup(k, text, budget)
        if out is None:
            if self.strict:
                raise sim.MissingSummaryError(k, text, budget)
            self.placeholder = True
            out = sim.placeholder_summary(k, text, budget)
        self.used.append(out)
        return out

    def sig(self) -> str:
        if self.placeholder:
            return "placeholder"
        return hashlib.sha256("\x00".join(self.used).encode("utf-8")).hexdigest()[:16]


class QSummaries(Summaries):
    """qthresh: Qwen3-8B's own summaries, generated by `lsum` and read from its on-disk cache."""

    def __init__(self, label: str, strict: bool) -> None:
        self.label, self.strict = label, strict
        self.used = []
        self.placeholder = False

    def lookup(self, k: int, text: str, budget: int) -> str | None:
        return lsum.cached_text(QSUM_DIR, self.label, f"e{k}", text, budget, QSUM_MODEL_ID)


def record_failure(cfg: SetCfg, key: str, error: str) -> None:
    path = cfg.out / f"{cfg.prefix}failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "error": error, "ts": time.time()}) + "\n")


def qwen_summary_rounds(
    cfg: SetCfg, plans: list[sim.TranscriptPlan], make_gen: Callable[[str], Any] | None = None
) -> set[str]:
    """Each event's input holds the previous summary, so events are generated a round at a time.
    A transcript whose summary fails is recorded and dropped from later rounds; its qthresh arm
    then finds the summary missing and is skipped. Returns the failed labels."""
    make_gen = make_gen or lsum.MlxGenerator
    gen: Any = None
    seen: set[tuple[str, str, str]] = set()
    failed: set[str] = set()
    try:
        while not dl.STOP.is_set():
            jobs = []
            for p in plans:
                if p.label in failed:
                    continue
                try:
                    sim.thresh_events(p, QSummaries(p.label, strict=True))
                except sim.MissingSummaryError as m:
                    jobs.append((p.label, f"e{m.k}", m.text, m.budget))
            if not jobs:
                return failed
            log(f"qthresh summaries ({QSUM_MODEL_ID}, local): {len(jobs)} jobs")
            for job in jobs:
                label, level, text, budget = job
                key = f"qsum|{label}|{level}"
                ident = (label, level, lsum.source_sha(text, budget, QSUM_MODEL_ID))
                # Broad on purpose: an MLX out-of-memory surfaces as a plain RuntimeError, and
                # one bad transcript must not stop the other arms' phases.
                try:
                    if ident in seen:
                        raise RuntimeError("summary not cached after generation")
                    seen.add(ident)
                    gen = gen or make_gen(QSUM_MODEL_ID)
                    lsum.precompute(
                        [job], QSUM_DIR, exi.summary_prompt, log, gen=gen, model_id=QSUM_MODEL_ID
                    )
                except Exception as exc:
                    failed.add(label)
                    record_failure(cfg, key, f"{type(exc).__name__}: {exc}")
                    log(f"FAILED {key}: {exc}; skipping qthresh for {label}")
        return failed
    finally:
        if gen is not None:
            del gen
            dl.free_mlx()


def summary_rounds(cfg: SetCfg, rt: exi.Runtime, plans: list[sim.TranscriptPlan]) -> None:
    if cfg.summary_arm == "qthresh":
        qwen_summary_rounds(cfg, plans)
        return
    if cfg.summary_arm is None:
        return
    while not dl.STOP.is_set():
        jobs = []
        for p in plans:
            try:
                sim.thresh_events(p, Summaries(cfg, rt.store, p.label, strict=True))
            except sim.MissingSummaryError as m:
                jobs.append(summary_job(cfg, p.label, m.k, m.text, m.budget))
        if not jobs:
            return
        log(f"thresh summaries: {len(jobs)} jobs")
        rt.run(jobs)
        if not any(j.done_by(rt.store.get(j.key)) for j in jobs):
            return


# --------------------------------------------------------------------------- cached simulation


def cached_run(
    cfg: SetCfg, arm: str, p: sim.TranscriptPlan, sig: str, build: Callable[[], sim.ArmRun]
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
    if arm == "cdec":
        params |= {"dec_v": sim.DEC_VERSION, "dec_keep": sim.DEC_KEEP, "dec_m": sim.DEC_M}
    if arm == "cfrzm":
        params["frzm_keep"] = sim.FRZM_KEEP
    if arm == "cfrzg":
        params |= {"frzm_keep": sim.FRZM_KEEP, "cfrzg_v": sim.CFRZG_VERSION, "gate": sim.CFRZG_GATE}
    if arm == "qthresh":
        params |= {"qthresh_v": sim.QTHRESH_VERSION, "qsum_model": QSUM_MODEL_ID}
    if arm == "tloosy":
        params |= {"tloosy_v": sim.TLOOSY_VERSION, "tloosy_share": sim.TLOOSY_SHARE}
    path = cfg.out / "sims" / f"{p.label}.{arm}.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("params") == params:
            return sim.ArmRun.from_json(d["run"])
    run = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": params, "run": run.to_json()}), encoding="utf-8")
    return run


def arm_run(
    cfg: SetCfg, arm: str, p: sim.TranscriptPlan, store: exi.Store | None, strict: bool = False
) -> tuple[sim.ArmRun, bool]:
    """The cached run for one arm; the flag is True when a summary arm used placeholders."""
    if arm not in SUMMARY_ARMS:
        build = lambda: sim.run_arm(arm, p, exi.loosy_compressor)  # noqa: E731
        return cached_run(cfg, arm, p, "", build), False
    sums: Summaries | None = None
    if store is not None:
        sums = (
            QSummaries(p.label, strict)
            if arm == "qthresh"
            else Summaries(cfg, store, p.label, strict)
        )
    if sums is None:
        run = cached_run(
            cfg, arm, p, "placeholder", lambda: sim.run_arm(arm, p, exi.loosy_compressor)
        )
        return run, True
    # Run once up front to learn the signature; the cached run is keyed on it.
    probe = sim.run_arm(arm, p, exi.loosy_compressor, sums, keep_at=[c.t for c in p.checkpoints])
    run = cached_run(cfg, arm, p, sums.sig(), lambda: probe)
    return run, sums.placeholder


# --------------------------------------------------------------------------- planning


def call_counts(cfg: SetCfg, plans: list[sim.TranscriptPlan]) -> dict[str, int]:
    n_ck = sum(len(p.checkpoints) for p in plans)
    out = {"probe_gen": n_ck}
    if cfg.summary_arm is not None:
        out[f"{cfg.summary_arm}_summaries"] = sum(len(sim.thresh_events(p)) for p in plans)
    return out | {"answers": n_ck * len(cfg.arms), "grades": n_ck * len(cfg.arms)}


def phase_models(cfg: SetCfg, gen: exi.Model) -> dict[str, exi.Model]:
    out = {"probe_gen": gen}
    if cfg.summary_arm is not None:
        out[f"{cfg.summary_arm}_summaries"] = QWEN if cfg.summary_arm == "qthresh" else SONNET
    return out | {"answers": cfg.answerer, "grades": cfg.grader}


def split_calls(cfg: SetCfg, gen: exi.Model, counts: dict[str, int]) -> tuple[int, int]:
    """(local, API) call counts for a `call_counts` plan."""
    models = phase_models(cfg, gen)
    local_n = sum(v for k, v in counts.items() if models[k].cli == "mlx")
    return local_n, sum(counts.values()) - local_n


def print_plan(cfg: SetCfg, plans: list[sim.TranscriptPlan], gen: exi.Model, max_calls: int) -> int:
    log(f"set {cfg.name} ({len(plans)}): {', '.join(p.label for p in plans)}")
    summariser = QSUM_MODEL_ID if cfg.summary_arm == "qthresh" else SONNET.name
    log(
        f"probe generator: {gen.name}; answers: {cfg.answerer.name}; grader: {cfg.grader.name}; "
        f"summariser: {summariser}"
    )
    log(f"CONT_VERSION={V} KEEP={sim.KEEP} F={sim.FREEZE_TOKENS} cap={cfg.cap_share} of total")
    if "tloosy" in cfg.arms:
        log(f"TLOOSY_VERSION={sim.TLOOSY_VERSION} TLOOSY_SHARE={sim.TLOOSY_SHARE} of C")
    if "cdec" in cfg.arms:
        log(f"DEC_VERSION={sim.DEC_VERSION} DEC_KEEP={sim.DEC_KEEP} DEC_M={sim.DEC_M}")
        log(f"FRZM_KEEP={sim.FRZM_KEEP}")
    if "qthresh" in cfg.arms or "cfrzg" in cfg.arms:
        log(
            f"QTHRESH_VERSION={sim.QTHRESH_VERSION} CFRZG_VERSION={sim.CFRZG_VERSION} "
            f"CFRZG_GATE={sim.CFRZG_GATE} FRZM_KEEP={sim.FRZM_KEEP}"
        )
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
    counts = call_counts(cfg, plans)
    for k, v in counts.items():
        log(f"  {k:20s} {v}")
    total = sum(counts.values())
    log(f"  {'TOTAL (placeholder events)':26s} {total}   (cap --max-calls {max_calls})")
    local_n, api_n = split_calls(cfg, gen, counts)
    log(f"  local (mlx) {local_n}, API {api_n}")
    return total


def pending_counts(
    cfg: SetCfg, plans: list[sim.TranscriptPlan], gen: exi.Model, store: exi.Store
) -> dict[str, int]:
    """Calls a run would still make against the checkpoint, built from the same jobs `run_all`
    builds. Grades whose answer is not stored yet count as pending (they follow the answer)."""
    out = {"probe_gen": 0, "summaries": 0, "answers": 0, "grades": 0}
    for p in plans:
        runs: dict[str, sim.ArmRun] = {}
        for a in cfg.arms:
            try:
                runs[a], _ = arm_run(cfg, a, p, store, strict=True)
            except sim.MissingSummaryError:
                out["summaries"] += 1
        for ck in p.checkpoints:
            label = ck_label(p, ck)
            probes = probes_for(cfg, store, label, gen)
            if not probes:
                out["probe_gen"] += 1
                continue
            for a, r in runs.items():
                ans = answer_job(cfg, label, a, r.contexts[ck.t], probes)
                if cfg.answerer is QWEN and qwen_over(ans.prompt) is not None:
                    continue
                answered = ans.done_by(store.get(ans.key))
                out["answers"] += not answered
                grade = grade_job(cfg, store, label, a, probes) if answered else None
                out["grades"] += grade is None or not grade.done_by(store.get(grade.key))
    return out


# --------------------------------------------------------------------------- dry run


def cost_table(
    cfg: SetCfg, plans: list[sim.TranscriptPlan], runs: dict[str, dict[str, sim.ArmRun]]
) -> str:
    rows = [
        "Offline cost per transcript (mean context tokens / mean uncached tokens per model turn; "
        "thresh uses placeholder summaries of the target length where no real ones are cached)",
        f"{'transcript':20s}{'turns':>6s}" + "".join(f"{a:>20s}" for a in cfg.arms),
    ]
    for p in plans:
        r = runs[p.label]
        rows.append(
            f"{p.label:20s}{len(r['full'].boundaries):6d}"
            + "".join(f"{r[a].mean_ctx():>11.0f}/{r[a].mean_uncached():<8.0f}" for a in cfg.arms)
        )
    rows += [
        "",
        f"{'arm':8s}{'ctx/turn':>10s}{'uncached/turn':>15s}{'ctx sum':>12s}"
        f"{'uncached sum':>14s}{'events':>8s}",
    ]
    for a in cfg.arms:
        rs = [runs[p.label][a] for p in plans]
        rows.append(
            f"{a:8s}{np.mean([r.mean_ctx() for r in rs]):>10.0f}"
            f"{np.mean([r.mean_uncached() for r in rs]):>15.0f}"
            f"{np.mean([sum(r.ctx_tokens) for r in rs]):>12.0f}"
            f"{np.mean([sum(r.uncached_tokens) for r in rs]):>14.0f}"
            f"{np.mean([len(r.events) for r in rs]):>8.1f}"
        )
    return "\n".join(rows)


def dry_run(cfg: SetCfg, plans: list[sim.TranscriptPlan]) -> None:
    runs: dict[str, dict[str, sim.ArmRun]] = {}
    for p in plans:
        runs[p.label] = {a: arm_run(cfg, a, p, None)[0] for a in cfg.arms}
        log(f"simulated {p.label}")
    table = cost_table(cfg, plans, runs)
    out = cfg.out / f"{cfg.prefix}dry_run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cost.txt").write_text(table + "\n", encoding="utf-8")
    p = plans[0]
    ck = p.checkpoints[1]
    for a in cfg.arms:
        (out / f"{p.label}@{ck.pct}__{a}.txt").write_text(
            runs[p.label][a].contexts[ck.t], encoding="utf-8"
        )
    print(table, flush=True)
    log(f"sample contexts for {p.label}@{ck.pct} in {out}")


# --------------------------------------------------------------------------- jobs


def probes_for(
    cfg: SetCfg, store: exi.Store, label: str, gen: exi.Model
) -> list[dict[str, str]] | None:
    text = exi._text(store, label, cfg.probes_key, gen.name, "probe_gen")
    return None if text is None else exi.parse_probes(text)


def probe_jobs(cfg: SetCfg, plans: list[sim.TranscriptPlan], gen: exi.Model) -> list[exi.Job]:
    return [
        exi.Job(
            ck_label(p, ck),
            cfg.probes_key,
            gen,
            "probe_gen",
            exi.probe_gen_prompt(exi.render_turns(p.turns[: ck.tail_from])),
            lambda t: exi.parse_probes(t) is not None,
        )
        for p in plans
        for ck in p.checkpoints
    ]


def answer_job(
    cfg: SetCfg, label: str, arm: str, context: str, probes: list[dict[str, str]]
) -> exi.Job:
    return dl.answer_job(label, cond(cfg, arm), context, probes, answerer=cfg.answerer)


def grade_job(
    cfg: SetCfg, store: exi.Store, label: str, arm: str, probes: list[dict[str, str]]
) -> exi.Job | None:
    return dl.grade_job(
        store, label, cond(cfg, arm), probes, answerer=cfg.answerer, grader=cfg.grader
    )


class ClosedModelError(RuntimeError):
    """Not an exi.CallError on purpose: call_with_retry would retry it and Runtime would log it
    as one failed call and carry on, where the open-weight run must stop."""


def refuse_closed(model: exi.Model, prompt: str) -> exi.CallResult:
    raise ClosedModelError(f"{model.name} is not a local open-weight model; this run is MLX only")


class LocalGuard:
    """dl.Guard's call cap and STOP handling in front of the MLX-only runner."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.used = 0
        self._run = local.dispatch(refuse_closed)

    def runner(self, model: exi.Model, prompt: str) -> exi.CallResult:
        if dl.STOP.is_set():
            raise exi.CallError("halted by budget guard")
        if self.used >= self.max_calls:
            dl.STOP.set()
            log(f"!! --max-calls {self.max_calls} reached, stopping")
            raise exi.CallError("halted by budget guard")
        self.used += 1
        return self._run(model, prompt)


def run_all(cfg: SetCfg, rt: exi.Runtime, plans: list[sim.TranscriptPlan], gen: exi.Model) -> None:
    """Probes, then summaries, then answers, then grades. Each phase uses one model, and local
    weights are dropped between phases so two models are never resident (16 GB machine)."""
    store = rt.store
    gen_jobs = probe_jobs(cfg, plans, gen)
    log(f"probe generation ({gen.name}): {len(gen_jobs)} jobs")
    rt.run(gen_jobs)
    local.unload()
    summary_rounds(cfg, rt, plans)
    answers: list[exi.Job] = []
    over: list[str] = []
    contexts: list[tuple[str, str, list[dict[str, str]]]] = []
    for p in plans:
        runs: dict[str, sim.ArmRun] = {}
        for a in cfg.arms:
            try:
                runs[a], _ = arm_run(cfg, a, p, store, strict=True)
            except sim.MissingSummaryError:
                log(f"{p.label}: {a} summaries incomplete, skipping the arm")
        for ck in p.checkpoints:
            probes = probes_for(cfg, store, ck_label(p, ck), gen)
            if not probes:
                continue
            for a, r in runs.items():
                job = answer_job(cfg, ck_label(p, ck), a, r.contexts[ck.t], probes)
                # Not queued: the runner would refuse it with a CallError, which call_with_retry
                # retries (with sleeps) although the prompt can never fit.
                if cfg.answerer is QWEN and (n := qwen_over(job.prompt)) is not None:
                    over.append(job.key)
                    record_failure(cfg, job.key, f"over Qwen window: {n} tokens, not queued")
                    continue
                contexts.append((ck_label(p, ck), a, probes))
                answers.append(job)
    if over:
        log(f"answers over the Qwen window, not queued: {len(over)}")
    log(f"answers ({cfg.answerer.name}): {len(answers)} jobs")
    if not dl.STOP.is_set():
        rt.run(answers)
    local.unload()
    grades = [j for lb, a, pr in contexts if (j := grade_job(cfg, store, lb, a, pr))]
    log(f"grading ({cfg.grader.name}): {len(grades)} jobs")
    if not dl.STOP.is_set():
        rt.run(grades)
    local.unload()


# --------------------------------------------------------------------------- report


def checkpoint_score(
    cfg: SetCfg, store: exi.Store, label: str, arm: str, gen: exi.Model
) -> float | None:
    probes = probes_for(cfg, store, label, gen)
    if not probes:
        return None
    n = len(probes)
    text = exi._text(store, label, cond(cfg, arm), cfg.answerer.name, cfg.grade_task)
    grades = exi.parse_probe_grades(text, n) if text else None
    return None if grades is None else float(np.mean(grades))


def report(cfg: SetCfg, store: exi.Store, plans: list[sim.TranscriptPlan], gen: exi.Model) -> str:
    dec_note = (
        f", DEC_VERSION {sim.DEC_VERSION} (keep {sim.DEC_KEEP}, m {sim.DEC_M}), "
        f"cfrzm keep {sim.FRZM_KEEP}"
        if "cdec" in cfg.arms
        else ""
    )
    title = "pilot, " if cfg.name == "E" else ""
    out = [
        f"# Continuous compaction {title}set {cfg.name}, written {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"CONT_VERSION {V}{dec_note}. Haiku 5.5 probe QA accuracy, graded by Sonnet 5.5; probes by "
        f"{gen.name}, no full-context filter (every probe counts, for every arm). A transcript "
        "counts for an arm only if all three of its checkpoints are graded. Mean [95% bootstrap "
        "CI over transcripts].",
    ]
    if gen is not dl.CODEX_PROBER:
        out.append(f"NOTE: probe generator fell back to {gen.name} (codex unavailable).")
    if dl.STOP.is_set():
        out.append("NOTE: the driver stopped on a spend limit or --max-calls; tables are partial.")
    scores, by_pct = collect_scores(cfg, store, plans, gen)
    cost, placeholder = arm_costs(cfg, store, plans)
    out += arm_table(cfg, scores, cost)
    if placeholder:
        out.append("")
        out.append("NOTE: thresh cost uses placeholder summaries where real ones are not cached.")
    out += pair_table(cfg, scores) + pct_table(cfg, by_pct)
    out += ["", f"Calls recorded in checkpoint: {len(store.records)}"]
    return "\n".join(out) + "\n"


Scores = dict[str, dict[str, float]]
PctScores = dict[str, dict[int, dict[str, float]]]


def collect_scores(
    cfg: SetCfg, store: exi.Store, plans: list[sim.TranscriptPlan], gen: exi.Model
) -> tuple[Scores, PctScores]:
    """Per-arm transcript means (only transcripts with all checkpoints graded) and per-pct
    checkpoint scores."""
    by_pct: PctScores = {a: {} for a in cfg.arms}
    scores: Scores = {a: {} for a in cfg.arms}
    for a in cfg.arms:
        for p in plans:
            got = {
                c.pct: checkpoint_score(cfg, store, ck_label(p, c), a, gen) for c in p.checkpoints
            }
            for pct, v in got.items():
                if v is not None:
                    by_pct[a].setdefault(pct, {})[p.label] = v
            if all(v is not None for v in got.values()):
                scores[a][p.label] = float(np.mean([v for v in got.values() if v is not None]))
    return scores, by_pct


def arm_costs(
    cfg: SetCfg, store: exi.Store, plans: list[sim.TranscriptPlan]
) -> tuple[dict[str, tuple[float, float]], bool]:
    """(mean context, mean uncached) per arm, and whether any summary arm used placeholders."""
    cost: dict[str, tuple[float, float]] = {}
    placeholder = False
    for a in cfg.arms:
        rs = []
        for p in plans:
            r, ph = arm_run(cfg, a, p, store)
            rs.append(r)
            placeholder |= ph
        cost[a] = (
            float(np.mean([r.mean_ctx() for r in rs])),
            float(np.mean([r.mean_uncached() for r in rs])),
        )
    return cost, placeholder


def arm_table(cfg: SetCfg, scores: Scores, cost: dict[str, tuple[float, float]]) -> list[str]:
    out = ["", "| arm | accuracy | mean context tokens/turn | mean uncached tokens/turn |"]
    out.append("|---|---|---|---|")
    out += [
        f"| {a} | {dl.cell(list(scores[a].values()))} | {cost[a][0]:.0f} | {cost[a][1]:.0f} |"
        for a in cfg.arms
    ]
    return out


def pair_table(cfg: SetCfg, scores: Scores, by_pct: PctScores | None = None) -> list[str]:
    """With `by_pct`, rows against full pair checkpoints rather than transcripts."""
    out = ["", "| paired difference | delta accuracy |", "|---|---|"]
    return out + [
        f"| {a} - {b} | "
        + (
            paired_checkpoints(by_pct, a, b)
            if by_pct and b == "full"
            else dl.paired(scores[a], scores[b])
        )
        + " |"
        for a, b in cfg.pairs
    ]


def paired_checkpoints(by_pct: PctScores, a: str, b: str) -> str:
    """a - b over the checkpoints both arms have graded: averaged within a transcript first, so
    the bootstrap and sign test still resample transcripts. For full, which is refused at the
    checkpoints over Qwen's window and would otherwise lose those whole transcripts."""
    diffs: dict[str, list[float]] = {}
    for pct, row in sorted(by_pct[b].items()):
        for label, vb in sorted(row.items()):
            va = by_pct[a].get(pct, {}).get(label)
            if va is not None:
                diffs.setdefault(label, []).append(va - vb)
    if not diffs:
        return "-"
    d = [float(np.mean(diffs[k])) for k in sorted(diffs)]
    m, lo, hi = exi.bootstrap_ci(d)
    n_ck = sum(len(v) for v in diffs.values())
    return (
        f"{m:+.3f} [{lo:+.2f},{hi:+.2f}] sign p={dl.sign_p(d):.3f} "
        f"n={len(d)} transcripts, {n_ck} checkpoints"
    )


def pct_table(cfg: SetCfg, by_pct: PctScores) -> list[str]:
    out = ["", "| arm | 50% | 75% | 100% |", "|---|---|---|---|"]
    return out + [
        f"| {a} | "
        + " | ".join(dl.cell(list(by_pct[a].get(pct, {}).values())) for pct in (50, 75, 100))
        + " |"
        for a in cfg.arms
    ]


def qwen_over(prompt: str) -> int | None:
    """The prompt's Qwen token count when it leaves less than MAX_TOKENS of Qwen's window."""
    n = local.qwen_tokens(prompt)
    return n if n > local.QWEN_WINDOW - local.MAX_TOKENS else None


def over_window(
    cfg: SetCfg, store: exi.Store | None, plans: list[sim.TranscriptPlan], gen: exi.Model
) -> dict[str, list[str]]:
    """Checkpoints whose answer prompt (the context alone while probes are missing) leaves Qwen
    less than MAX_TOKENS of its window, per arm. The runner refuses those calls; they are listed,
    never cut."""
    out: dict[str, list[str]] = {a: [] for a in cfg.arms}
    for p in plans:
        for a in cfg.arms:
            r, _ = arm_run(cfg, a, p, store)
            for ck in p.checkpoints:
                label = ck_label(p, ck)
                probes = probes_for(cfg, store, label, gen) if store is not None else None
                ctx = r.contexts[ck.t]
                text = exi.probe_answer_prompt(ctx, [x["q"] for x in probes]) if probes else ctx
                if qwen_over(text) is not None:
                    out[a].append(label)
    return out


def qwen_report(
    cfg: SetCfg, store: exi.Store, plans: list[sim.TranscriptPlan], gen: exi.Model
) -> str:
    out = [
        f"# Continuous compaction, open-weight replay on set F, written "
        f"{time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"CONT_VERSION {V}, QTHRESH_VERSION {sim.QTHRESH_VERSION}, CFRZG_VERSION "
        f"{sim.CFRZG_VERSION} (gate {sim.CFRZG_GATE} o200k tokens), cfrzm/cfrzg keep "
        f"{sim.FRZM_KEEP}, tloosy share {sim.TLOOSY_SHARE}. {cfg.answerer.name} (4-bit MLX, "
        f"greedy, thinking off) probe QA accuracy, graded by {cfg.grader.name}; probes by "
        f"{gen.name} ({cfg.probes_key}); qthresh summaries by {QSUM_MODEL_ID} (lsum). Every "
        "call is local and no closed model is used anywhere, so these numbers are a standalone "
        "set, not comparable with the Claude-answered set F report. No full-context filter. In "
        "the arm table and the pairs between compacted arms, a transcript counts for an arm only "
        "if all three of its checkpoints are graded. Rows against full are paired per checkpoint "
        "instead: the difference is taken only at checkpoints where full has a grade (full is "
        "refused at checkpoints over Qwen's window), averaged within each transcript, and its n "
        "gives both transcripts and checkpoints. Mean [95% bootstrap CI over transcripts].",
    ]
    if dl.STOP.is_set():
        out.append("NOTE: the driver stopped on --max-calls; tables are partial.")
    scores, by_pct = collect_scores(cfg, store, plans, gen)
    cost, placeholder = arm_costs(cfg, store, plans)
    out += arm_table(cfg, scores, cost)
    if placeholder:
        out += ["", "NOTE: qthresh cost uses placeholder summaries where Qwen's are not cached."]
    ref = float(np.mean([arm_run(CFGS["F"], "thresh", p, store)[0].mean_ctx() for p in plans]))
    off = cost["qthresh"][0] / ref - 1
    out += [
        "",
        f"qthresh mean context {cost['qthresh'][0]:.0f} vs thresh (Sonnet summaries, set F) "
        f"{ref:.0f}: {off:+.1%}.",
    ]
    if abs(off) > QTHRESH_TOL:
        out.append(
            f"FLAG: qthresh is more than {QTHRESH_TOL:.0%} off thresh's context size, so its "
            "comparisons read as accuracy at the context size each arm reached."
        )
    out += pair_table(cfg, scores, by_pct)
    out += ["", "Per checkpoint position:"] + pct_table(cfg, by_pct)
    over = over_window(cfg, store, plans, gen)
    n_ck = sum(len(p.checkpoints) for p in plans)
    limit = local.QWEN_WINDOW - local.MAX_TOKENS
    out += [
        "",
        f"Checkpoints over the Qwen window (answer prompt over {limit} Qwen tokens, i.e. "
        f"{local.QWEN_WINDOW} less the {local.MAX_TOKENS}-token reply cap), out of {n_ck}. These "
        "answers are never queued and the context is never cut; the checkpoint stays ungraded "
        "for that arm (full's pairs skip it, see above):",
        "",
        "| arm | over window | checkpoints |",
        "|---|---|---|",
    ]
    out += [f"| {a} | {len(over[a])} | {', '.join(over[a]) or '-'} |" for a in cfg.arms]
    mine = sum(
        r.get("caller") in (cfg.answerer.name, cfg.grader.name, gen.name)
        for r in store.records.values()
    )
    out += ["", f"Local calls recorded in checkpoint: {mine}"]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- main


def calibrate(cfg: SetCfg, plans: list[sim.TranscriptPlan]) -> None:
    ref = np.mean([arm_run(cfg, "thresh", p, None)[0].mean_ctx() for p in plans])
    cfrz = [arm_run(cfg, "cfrz", p, None)[0] for p in plans]
    cfrz_ctx = float(np.mean([r.mean_ctx() for r in cfrz]))
    log(f"thresh (placeholder summaries) mean ctx/turn {ref:.0f}")
    log(
        f"cfrz mean ctx/turn {cfrz_ctx:.0f} ({cfrz_ctx / ref - 1:+.1%} vs thresh, not recalibrated)"
    )
    log("cfrzm (plain freeze, raised keep):")
    log(f"{'keep':>6s}{'ctx/turn':>10s}{'vs thresh':>11s}{'uncached/turn':>15s}")
    for keep in CAL_FRZM_KEEPS:
        rs = [
            sim.run_arm("cfrzm", p, exi.loosy_compressor, keep_at=[], frzm_keep=keep) for p in plans
        ]
        ctx = float(np.mean([r.mean_ctx() for r in rs]))
        unc = float(np.mean([r.mean_uncached() for r in rs]))
        ok = "  *" if abs(ctx / ref - 1) <= CAL_TOL else ""
        log(f"{keep:6.2f}{ctx:10.0f}{ctx / ref - 1:+11.1%}{unc:15.0f}{ok}")
    log("cdec:")
    log(f"{'m':>3s}{'keep':>6s}{'ctx/turn':>10s}{'vs thresh':>11s}{'uncached/turn':>15s}")
    for m in CAL_M:
        for keep in CAL_KEEPS:
            rs = [
                sim.run_arm("cdec", p, exi.loosy_compressor, keep_at=[], dec=(keep, m))
                for p in plans
            ]
            ctx = float(np.mean([r.mean_ctx() for r in rs]))
            unc = float(np.mean([r.mean_uncached() for r in rs]))
            ok = "  *" if abs(ctx / ref - 1) <= CAL_TOL else ""
            log(f"{m:3d}{keep:6.2f}{ctx:10.0f}{ctx / ref - 1:+11.1%}{unc:15.0f}{ok}")


def calibrate_tloosy(cfg: SetCfg, plans: list[sim.TranscriptPlan], store: exi.Store) -> None:
    runs = [arm_run(cfg, "thresh", p, store) for p in plans]
    ref = float(np.mean([r.mean_ctx() for r, _ in runs]))
    ph = sum(ph for _, ph in runs)
    log(f"thresh REAL mean ctx/turn {ref:.0f} ({ph} transcripts fell back to placeholders)")
    log("tloosy (target = share x C):")
    log(f"{'share':>6s}{'ctx/turn':>10s}{'vs thresh':>11s}{'uncached/turn':>15s}{'events':>8s}")
    for share in CAL_TLOOSY_SHARES:
        rs = [
            sim.run_arm("tloosy", p, exi.loosy_compressor, keep_at=[], tloosy_share=share)
            for p in plans
        ]
        ctx = float(np.mean([r.mean_ctx() for r in rs]))
        unc = float(np.mean([r.mean_uncached() for r in rs]))
        ev = float(np.mean([len(r.events) for r in rs]))
        ok = "  *" if abs(ctx / ref - 1) <= CAL_TOL else ""
        log(f"{share:6.2f}{ctx:10.0f}{ctx / ref - 1:+11.1%}{unc:15.0f}{ev:8.1f}{ok}")


CAL_TLOOSY_SHARES = (0.20, 0.25, 0.30, 0.33, 0.40, 0.50, 0.60)
CAL_M = (2, 3, 4)
CAL_KEEPS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
CAL_FRZM_KEEPS = (0.40, 0.45, 0.50, 0.55, 0.60)
CAL_TOL = 0.05


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--set", dest="set_name", choices=sorted(CFGS), default="E")
    ap.add_argument("--plan", action="store_true", help="print the call plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="offline cost metrics, no model calls")
    ap.add_argument(
        "--calibrate", action="store_true", help="offline cfrzm / cdec keep grids, set F"
    )
    ap.add_argument(
        "--calibrate-tloosy", action="store_true", help="offline tloosy target grid, set F"
    )
    ap.add_argument("--report", action="store_true", help="write the report from the checkpoint")
    ap.add_argument(
        "--max-calls", type=int, default=None, help="default 360 (E) / 1000 (F) / 2000 (qwen3-8b)"
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--answerer",
        choices=sorted(ANSWERER_CFGS),
        default=None,
        help="open-weight replay of set F, all local (workers forced to 1)",
    )
    args = ap.parse_args()

    if args.answerer is not None:
        if args.set_name != "F":
            raise SystemExit("--answerer runs on set F only")
        cfg = ANSWERER_CFGS[args.answerer]
        workers = 1
    else:
        cfg = CFGS[args.set_name]
        workers = args.workers
    max_calls = args.max_calls or cfg.max_calls
    gen = cfg.prober or (dl.CODEX_PROBER if shutil.which("codex") else HAIKU)
    write_report = qwen_report if cfg.answerer is QWEN else report
    plans = select_plans(cfg)
    total = print_plan(cfg, plans, gen, max_calls)
    if args.plan:
        if cfg.calls.exists():
            pend = pending_counts(cfg, plans, gen, exi.Store(cfg.calls))
            log(f"pending against {cfg.calls.name}: {pend} = {sum(pend.values())} new calls")
        if cfg.answerer is QWEN:
            store = exi.Store(cfg.calls) if cfg.calls.exists() else None
            over = over_window(cfg, store, plans, gen)
            log(f"checkpoints over the Qwen window: { {a: len(v) for a, v in over.items()} }")
        return
    if args.calibrate:
        calibrate(cfg, plans)
        return
    if args.calibrate_tloosy:
        calibrate_tloosy(cfg, plans, exi.Store(cfg.calls))
        return
    if args.dry_run:
        dry_run(cfg, plans)
        return
    store = exi.Store(cfg.calls)
    if args.report:
        text = write_report(cfg, store, plans, gen)
        cfg.report.write_text(text, encoding="utf-8")
        (cfg.out / f"{cfg.prefix}report.md").write_text(text, encoding="utf-8")
        print(text, flush=True)
        return
    if total > max_calls and not cfg.calls.exists():
        raise SystemExit(f"planned {total} calls exceeds --max-calls {max_calls}")
    guard: dl.Guard | LocalGuard = (
        LocalGuard(max_calls) if cfg.answerer.cli == "mlx" else dl.Guard(max_calls)
    )
    failures = cfg.out / f"{cfg.prefix}failures.jsonl"
    rt = exi.Runtime(store, guard.runner, failures, workers=workers)
    rt.sleep = dl.guarded_sleep
    rt.log = lambda m: print(m, flush=True)
    try:
        run_all(cfg, rt, plans, gen)
    finally:
        text = write_report(cfg, store, plans, gen)
        (cfg.out / f"{cfg.prefix}report.md").write_text(text, encoding="utf-8")
        if cfg.answerer is QWEN:
            cfg.report.write_text(text, encoding="utf-8")
        print(text, flush=True)
        log(
            f"DONE: {rt.new_calls} new calls, {rt.failed} failed, runner invocations "
            f"{guard.used}/{guard.max_calls}, stop={dl.STOP.is_set()}"
        )


if __name__ == "__main__":
    main()

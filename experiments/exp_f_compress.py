"""Phase 2 / Experiment F: compression-based selectors on Experiment D's curves.

Scores a segment by how many compressed bytes it adds given the text already kept
(`loosy_goose.compress_select`, lazy greedy), with zlib's 32 KiB window against lzma's
whole-transcript dictionary, each with and without the supersession pass. No reference method
runs here: tfidf, leverage and supersede+leverage are read from the exp_d checkpoints, and only
where those were scored against the same segmenter output.

Usage: uv run python experiments/exp_d_curves.py --grid exp_f [--only NAME_SUBSTR] [--methods A,B]
Writes experiments/output/exp_f/<label>.json and experiments/output/exp_f/summary.txt.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import exp_d_curves as exp_d  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    AGGRESSIVE_RATES,
    BASE_VARIANT,
    EXP_D,
    KNEE_FLOOR,
    METRICS,
    Grid,
    MethodFn,
    RunContext,
    _base_only,
    _curves_by_method,
    _interp_at,
    _knee,
    _record_identity,
    _weighted_mean,
    plan_jobs,
)
from loosy_goose.budget import ProtectPolicy, token_budget  # noqa: E402
from loosy_goose.compress_select import Codec, greedy_select  # noqa: E402
from loosy_goose.segment import Segment  # noqa: E402
from loosy_goose.supersede import apply_supersession  # noqa: E402

OUT = exp_d.ROOT / "experiments" / "output" / "exp_f"
REFERENCES = ("tfidf", "leverage", "supersede+leverage")
SHIPPED = "supersede+leverage"


def _run_lazy(codec: Codec, *, supersede: bool) -> MethodFn:
    def run(
        segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
    ) -> list[Segment]:
        del ctx
        budget = token_budget(segs, ratio)
        pool = apply_supersession(segs) if supersede else segs
        return greedy_select(pool, budget, codec, lazy=True, protect=protect)

    return run


# Cut 2026-09-29 after the full 14-arm grid (docs/results/exp_f_summary.txt): one-pass lost to
# lazy on both recall and coverage at every rate; exact greedy had worse recall_final than lazy
# (-0.029 to -0.042) at 24x the cost; stochastic greedy matched exact but was noisy (one-draw
# recall spread 0.03-0.09) and slower. Exact greedy stays in compress_select as the tests'
# correctness anchor.
def _arms(supersede: bool) -> dict[str, MethodFn]:
    prefix = "supersede+" if supersede else ""
    return {
        f"{prefix}zlib-lazy": _run_lazy("zlib", supersede=supersede),
        f"{prefix}lzma-lazy": _run_lazy("lzma", supersede=supersede),
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
    plan = {j.identity() for j in plan_jobs(grid=grid)}
    rows: list[str] = []
    not_run: list[str] = []
    incomplete: list[str] = []
    for method in grid.methods:
        mine = {i for i in plan if i.method == method}
        present = failed = want = 0
        for r in results:
            if "skipped" in r:
                continue
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
    out.append("")
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
        "Per transcript: median elapsed_seconds over its ratios; then the median over",
        "transcripts. References are exp_d's own timings.",
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
            per.append(seconds / (float(r["total_tokens"]) / 100_000))
        cell = f"{float(np.median(per)):.2f}" if per else "missing"
        out.append(f"{method:<26}{cell:>10}{len(per):>5}")
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
        "== window length: lzma-lazy minus zlib-lazy ==",
        ["Whole-transcript dictionary against zlib's 32 KiB window, same selector otherwise."],
        [("lzma-lazy", "zlib-lazy"), ("supersede+lzma-lazy", "supersede+zlib-lazy")],
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
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    return text


def _write_summary(results: list[dict[str, Any]], out: Path) -> None:
    print("\n" + write_summary_with(results, out, reference_dir=EXP_D.out, grid=EXP_F))


EXP_F = Grid(
    name="exp_f",
    out=OUT,
    methods=METHODS,
    variants=(BASE_VARIANT,),
    protect_code_only=(),
    write_summary=_write_summary,
    plots=False,
)

"""Phase 2 / Experiment E: rate-distortion analysis over the Experiment D checkpoints.

Experiment D produced the curves; this reads them back and asks what they cost. Four questions,
all answered from `experiments/output/exp_d/*.json` alone — no transcript, no embedding model, no
re-sweep, so it is cheap to re-run whenever D is extended:

  E1  elasticity      how many recall points does the next 1x of compression cost?
  E2  per-kind        which channel (prose / code / tool_use / tool_result) bleeds, and where?
  E3  predictors      which transcript properties predict how far a method can compress?
  E4  cost            what does the spectral machinery buy per second of compute?

Everything is keyed on ACHIEVED compression ratio, never on requested keep_ratio, for the reason
stated in exp_d_curves._curves_by_method: a supersede/band method can stop short of its budget,
so two methods at the same keep_ratio are not at the same rate.

Usage: uv run python experiments/exp_e_analysis.py
Writes experiments/output/exp_e/summary.txt and elasticity.png. Local sessions stay withheld:
only the local_5k..local_200k labels and aggregate numbers leave the checkpoints, exactly as in
Experiment D.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "experiments" / "output" / "exp_d"
OUT = ROOT / "experiments" / "output" / "exp_e"

# Compression multipliers, not keep ratios: "3x" is the unit the project is actually steered by,
# and a multiplier grid spaces the aggressive end (where the curves bend) more densely than a
# uniform rate grid would.
MULTIPLIERS: tuple[float, ...] = (1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0)
# Bands for the marginal-cost table. Each is a (low, high) multiplier pair; the reported number is
# the mean recall lost per +1x of compression across that band.
BANDS: tuple[tuple[float, float], ...] = ((1.25, 2.0), (2.0, 4.0), (4.0, 8.0))
KIND_RATES: tuple[float, ...] = (0.5, 0.33, 0.25)
KINDS: tuple[str, ...] = ("prose", "code", "tool_use", "tool_result", "thinking")
KNEE_FLOOR = 0.90
BASELINE = "tfidf"
# A kind carried by fewer transcripts than this is dropped from the per-channel table; see
# _reported_kinds for why a one-transcript channel reads as a perfect score.
MIN_KIND_TRANSCRIPTS = 3
# The stack Experiment D ranked first on the knee; E3 correlates transcript properties against
# how far *this* method can push, since a per-method predictor table would be mostly noise at
# n = 18 transcripts.
FOCUS_METHOD = "supersede+band+leverage"
PLOT_METHODS: tuple[str, ...] = (
    "random",
    "tfidf",
    "leverage",
    "ridge",
    "supersede+tfidf",
    "supersede+leverage",
    "supersede+band+leverage",
)


def load_checkpoints() -> list[dict[str, Any]]:
    files = sorted(IN.glob("*.json"))
    if not files:
        raise SystemExit(f"no Experiment D checkpoints in {IN}; run exp_d_curves.py first")
    results = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    # Same guard as exp_d's own fingerprint check, for the same reason: a checkpoint written
    # under a different metric set or ratio grid must not be silently averaged in with the rest.
    configs = {json.dumps(r.get("config"), sort_keys=True) for r in results}
    if len(configs) > 1:
        raise SystemExit(
            f"checkpoints disagree on config fingerprint ({len(configs)} distinct); "
            "re-run exp_d_curves.py --force before analysing"
        )
    missing = [r["label"] for r in results if not r.get("records")]
    if missing:
        raise SystemExit(f"checkpoints with no records: {missing}")
    return results


def _metric_names() -> list[str]:
    names = ["recall", "recall_final", "coverage", "doc_cosine"]
    names += [f"recall:{k}" for k in KINDS]
    names += [f"cov:{k}" for k in KINDS]
    return names


def _record_metrics(rec: dict[str, Any]) -> dict[str, float]:
    out = {
        "recall": float(rec["atom_recall"]),
        "recall_final": float(rec["atom_recall_final"]),
        "coverage": float(rec["coverage"]),
        "doc_cosine": float(rec["doc_cosine"]),
    }
    by_kind = rec.get("atom_recall_by_kind") or {}
    cov_kind = rec.get("coverage_by_kind") or {}
    for k in KINDS:
        # nan, not 0.0, for a kind this transcript does not contain: averaging a missing channel
        # in as a zero would make prose-only transcripts look like catastrophic code loss.
        out[f"recall:{k}"] = float(by_kind.get(k, np.nan))
        out[f"cov:{k}"] = float(cov_kind.get(k, np.nan))
    return out


def build_curves(results: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Per (method, protect): one curve per transcript, sorted by achieved rate.

    Mirrors exp_d_curves._curves_by_method — including its rule that a repeated achieved rate is
    dropped rather than averaged, since a stacked method that has exhausted its candidate pool
    reports the identical kept set at every higher keep_ratio — but carries every metric, per-kind
    ones included, instead of the four the head-to-head needed.
    """
    names = _metric_names()
    out: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_key: dict[tuple[str, str], list[tuple[float, dict[str, float], float]]] = defaultdict(
            list
        )
        for rec in r["records"]:
            by_key[(rec["method"], rec["protect"])].append(
                (
                    float(rec["compression_ratio"]),
                    _record_metrics(rec),
                    float(rec.get("elapsed_seconds", np.nan)),
                )
            )
        for key, pts in by_key.items():
            pts.sort(key=lambda p: p[0])
            rates: list[float] = []
            series: dict[str, list[float]] = {n: [] for n in names}
            seconds: list[float] = []
            for rate, vals, secs in pts:
                if rates and rate <= rates[-1] + 1e-9:
                    continue
                rates.append(rate)
                seconds.append(secs)
                for n in names:
                    series[n].append(vals[n])
            out[key].append(
                {
                    "label": r["label"],
                    "weight": float(r["total_tokens"]),
                    "rates": rates,
                    "seconds": seconds,
                    **series,
                }
            )
    return out


def interp_at(curve: dict[str, Any], rate: float, metric: str) -> float | None:
    """Value of `metric` at achieved rate `rate`, or None if this transcript's curve never
    reached that rate. Extrapolation is refused on purpose: past the ceiling the method is not
    compressing further, so an extrapolated point would be an invention, not a measurement."""
    rates = curve["rates"]
    if not rates or rate < rates[0] - 1e-9 or rate > rates[-1] + 1e-9:
        return None
    ys = curve[metric]
    finite = [(x, y) for x, y in zip(rates, ys, strict=True) if np.isfinite(y)]
    if len(finite) < 2:
        return None
    xs, vals = zip(*finite, strict=True)
    if rate < xs[0] - 1e-9 or rate > xs[-1] + 1e-9:
        return None
    return float(np.interp(rate, xs, vals))


def weighted_at(
    curves: list[dict[str, Any]], rate: float, metric: str
) -> tuple[float | None, int, float]:
    """Token-weighted mean of `metric` at an achieved rate, over the transcripts that reached it.
    Returns (value, n_transcripts, best_ceiling_among_those_that_missed)."""
    vals: list[float] = []
    weights: list[float] = []
    ceilings: list[float] = []
    for c in curves:
        v = interp_at(c, rate, metric)
        if v is None:
            ceilings.append(c["rates"][-1] if c["rates"] else 1.0)
            continue
        vals.append(v)
        weights.append(c["weight"])
    if not vals:
        return None, 0, min(ceilings) if ceilings else 1.0
    return (
        float(np.average(vals, weights=weights)),
        len(vals),
        min(ceilings) if ceilings else 0.0,
    )


def knee(curve: dict[str, Any], floor: float = KNEE_FLOOR) -> float | None:
    rates, rf = curve["rates"], curve["recall_final"]
    out: float | None = None
    for rate, v in zip(reversed(rates), reversed(rf), strict=True):
        if v >= floor:
            out = rate
        else:
            break
    return out


def _fmt(v: float | None, width: int = 8, prec: int = 3) -> str:
    return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.{prec}f}"


def e1_elasticity(
    curves: dict[tuple[str, str], list[dict[str, Any]]], methods: list[str]
) -> list[str]:
    header = f"{'method':<26}" + "".join(f"{str(m) + 'x':>8}" for m in MULTIPLIERS)
    out = [
        "",
        "== E1a  final-state atom recall by compression multiplier ==",
        "Token-weighted, interpolated onto achieved compression multiplier (1/achieved rate).",
        "'n/a' = no transcript's curve reached that multiplier; for supersede/band stacks that",
        "is a real ceiling, not a gap in the sweep.",
        header,
        "-" * len(header),
    ]
    for method in methods:
        cs = curves.get((method, "none"), [])
        cells = [weighted_at(cs, 1.0 / m, "recall_final")[0] for m in MULTIPLIERS]
        out.append(f"{method:<26}" + "".join(_fmt(v) for v in cells))

    band_header = (
        f"{'method':<26}"
        + "".join(f"{f'{lo}-{hi}x':>12}" for lo, hi in BANDS)
        + f"{'worst band':>14}"
    )
    out += [
        "",
        "== E1b  marginal cost: final-state recall points lost per +1x compression ==",
        "The elasticity the project is steered by. A cell of 0.080 means: inside that band, each",
        "additional 1x of compression costs 8.0 points of final-state atom recall. Lower is",
        "better. Bands where a method cannot reach the upper multiplier are 'n/a'.",
        band_header,
        "-" * len(band_header),
    ]
    for method in methods:
        cells = _band_costs(curves.get((method, "none"), []), log_scale=False)
        finite = [(c, f"{lo}-{hi}x") for c, (lo, hi) in zip(cells, BANDS, strict=True) if c]
        worst = max(finite)[1] if finite else "n/a"
        row = "".join(f"{'n/a':>12}" if c is None else f"{c:>12.3f}" for c in cells)
        out.append(f"{method:<26}{row}{worst:>14}")

    out += [
        "",
        "== E1c  the same cost in relative terms (log-log elasticity) ==",
        "E1b is measured in absolute recall points, and recall is bounded at 0, so its decline",
        "across bands is partly a floor effect rather than a real bargain at high compression.",
        "This table divides it out: a cell of 0.60 means a 1% increase in compression costs 0.60%",
        "of the recall still remaining. A method whose row is FLAT is a power law — compressing",
        "harder is neither a bargain nor a penalty. A row that RISES means the deep end really is",
        "more expensive, and the E1b decline was the floor effect talking.",
        band_header,
        "-" * len(band_header),
    ]
    for method in methods:
        cells = _band_costs(curves.get((method, "none"), []), log_scale=True)
        finite = [(c, f"{lo}-{hi}x") for c, (lo, hi) in zip(cells, BANDS, strict=True) if c]
        worst = max(finite)[1] if finite else "n/a"
        row = "".join(f"{'n/a':>12}" if c is None else f"{c:>12.3f}" for c in cells)
        out.append(f"{method:<26}{row}{worst:>14}")
    return out


def _band_costs(cs: list[dict[str, Any]], *, log_scale: bool) -> list[float | None]:
    """Recall lost per unit of extra compression, per band. `log_scale` switches from absolute
    recall points per +1x to the log-log elasticity (fraction of remaining recall per fraction of
    extra compression), which is what makes bands at different depths comparable."""
    cells: list[float | None] = []
    for lo, hi in BANDS:
        a = weighted_at(cs, 1.0 / lo, "recall_final")[0]
        b = weighted_at(cs, 1.0 / hi, "recall_final")[0]
        if a is None or b is None:
            cells.append(None)
        elif not log_scale:
            cells.append((a - b) / (hi - lo))
        elif a > 0 and b > 0:
            cells.append(float((np.log(a) - np.log(b)) / (np.log(hi) - np.log(lo))))
        else:
            cells.append(None)
    return cells


def _reported_kinds(
    results: list[dict[str, Any]], min_transcripts: int = MIN_KIND_TRANSCRIPTS
) -> tuple[list[str], list[str]]:
    """Split KINDS into (reported, skipped). A kind carried by one transcript is not a channel
    measurement, it is that transcript: `thinking` appears once in this corpus, as a single
    segment, and atom_recall_by_kind returns 1.0 for a kind with no atoms — so an unfiltered
    column reads as a perfect score for every method."""
    present = {
        k: sum(1 for r in results if (r.get("segments_by_kind") or {}).get(k)) for k in KINDS
    }
    reported = [k for k in KINDS if present[k] >= min_transcripts]
    skipped = [f"{k} ({present[k]})" for k in KINDS if present[k] < min_transcripts]
    return reported, skipped


def e2_by_kind(
    curves: dict[tuple[str, str], list[dict[str, Any]]],
    methods: list[str],
    results: list[dict[str, Any]],
) -> list[str]:
    kinds, skipped = _reported_kinds(results)
    out = [
        "",
        "== E2  per-channel breakdown at the aggressive operating points ==",
        "recall: is the atom-recall metric restricted to atoms first seen in a segment of that",
        "kind (all-history, since the checkpoints carry no per-kind final-state variant).",
        "cov: is mean best-match cosine for that kind's segments. '.' = no transcript reached",
        "that rate.",
    ]
    if skipped:
        out.append(
            f"Kinds omitted, carried by fewer than {MIN_KIND_TRANSCRIPTS} transcripts: "
            + ", ".join(skipped)
        )
    for rate in KIND_RATES:
        mult = 1.0 / rate
        header = (
            f"{'method':<26}"
            + "".join(f"{'rec:' + k[:9]:>14}" for k in kinds)
            + "".join(f"{'cov:' + k[:9]:>14}" for k in kinds)
        )
        out += [
            "",
            f"-- achieved rate {rate} ({mult:.2f}x) --",
            header,
            "-" * len(header),
        ]
        for method in methods:
            cs = curves.get((method, "none"), [])
            cells = [weighted_at(cs, rate, f"recall:{k}")[0] for k in kinds]
            cells += [weighted_at(cs, rate, f"cov:{k}")[0] for k in kinds]
            out.append(
                f"{method:<26}"
                + "".join(f"{'.':>14}" if v is None else f"{v:>14.3f}" for v in cells)
            )
    return out


def _features(result: dict[str, Any]) -> dict[str, float]:
    by_kind = result.get("segments_by_kind") or {}
    n_seg = max(int(result["n_segments"]), 1)
    return {
        "log10_tokens": float(np.log10(max(float(result["total_tokens"]), 1.0))),
        "n_segments": float(n_seg),
        # Segment counts, not token shares: the checkpoints record counts only. A tool-heavy
        # transcript by count is usually tool-heavy by tokens too, but this is the weaker proxy
        # and the correlation below should be read as such.
        "tool_seg_share": (
            float(by_kind.get("tool_use", 0) + by_kind.get("tool_result", 0)) / n_seg
        ),
        "code_seg_share": float(by_kind.get("code", 0)) / n_seg,
        "prose_seg_share": float(by_kind.get("prose", 0)) / n_seg,
        "supersede_ceiling": float(result.get("supersede_ceiling", np.nan)),
    }


def e3_predictors(
    results: list[dict[str, Any]], curves: dict[tuple[str, str], list[dict[str, Any]]]
) -> list[str]:
    by_label = {c["label"]: c for c in curves.get((FOCUS_METHOD, "none"), [])}
    feats: dict[str, list[float]] = defaultdict(list)
    outcomes: dict[str, list[float]] = defaultdict(list)
    rows: list[tuple[str, float, float | None, float | None]] = []
    for r in results:
        curve = by_label.get(r["label"])
        if curve is None:
            continue
        k = knee(curve)
        rf25 = interp_at(curve, 0.25, "recall_final")
        f = _features(r)
        rows.append((r["label"], f["supersede_ceiling"], k, rf25))
        # Spearman needs a complete pair, and a method that never held the 0.90 floor has no
        # knee at all: dropping those rows is honest, imputing a sentinel would not be.
        if k is None or rf25 is None:
            continue
        for name, v in f.items():
            feats[name].append(v)
        outcomes["knee"].append(k)
        outcomes["recall_f@4x"].append(rf25)

    out = [
        "",
        f"== E3  what predicts how far '{FOCUS_METHOD}' can compress? ==",
        "Spearman rank correlation over transcripts. n is small (see below), so these are",
        "descriptive: they say which properties are worth designing around, not which are",
        "significant. knee = lowest achieved rate still holding final-state recall >= 0.90, so a",
        "NEGATIVE correlation with knee means 'more of this property lets us compress further'.",
        "",
    ]
    n = len(outcomes["knee"])
    header = f"{'feature':<20}" + f"{'rho vs knee':>14}{'rho vs recall_f@4x':>22}"
    out += [f"transcripts with a usable knee: {n} of {len(results)}", header, "-" * len(header)]
    for name in sorted(feats):
        cells = []
        for outcome in ("knee", "recall_f@4x"):
            xs, ys = feats[name], outcomes[outcome]
            if n < 3 or len(set(xs)) < 2:
                cells.append("n/a")
            else:
                rho = float(spearmanr(xs, ys).statistic)
                cells.append(f"{rho:+.3f}")
        out.append(f"{name:<20}{cells[0]:>14}{cells[1]:>22}")

    out += [
        "",
        "-- per transcript --",
        f"{'label':<26}{'supersede_ceiling':>19}{'knee':>10}{'recall_f@4x':>14}",
        "-" * 69,
    ]
    for label, ceiling, k, rf in sorted(rows, key=lambda x: -x[1]):
        out.append(f"{label:<26}{ceiling:>19.3f}{_fmt(k, 10)}{_fmt(rf, 14)}")
    return out


def e4_cost(curves: dict[tuple[str, str], list[dict[str, Any]]], methods: list[str]) -> list[str]:
    base = curves.get((BASELINE, "none"), [])
    base_at = {rate: weighted_at(base, rate, "recall_final")[0] for rate in (0.5, 0.25)}
    header = (
        f"{'method':<26}{'s/100k tok':>12}{'vs tfidf':>10}"
        f"{'d recall_f@2x':>15}{'d recall_f@4x':>15}{'s per point@4x':>16}"
    )
    out = [
        "",
        "== E4  what the machinery costs ==",
        "s/100k tok is the median over all sweep points of that method's wall-clock seconds",
        "scaled to a 100K-token transcript, measured on a WARM embedding cache — a cold run pays",
        "the embedding cost on top, so treat this as the steady-state figure, not first-run cost.",
        "'s per point@4x' is the extra seconds over tfidf bought per point of final-state recall",
        "gained at 4x; negative means the method is both slower and worse.",
        header,
        "-" * len(header),
    ]

    def _median_seconds(method: str) -> float:
        per100k: list[float] = []
        for c in curves.get((method, "none"), []):
            scale = 100_000.0 / max(c["weight"], 1.0)
            per100k += [s * scale for s in c["seconds"] if np.isfinite(s)]
        return float(np.median(per100k)) if per100k else float("nan")

    # Computed before the loop, not inside it: the baseline is not first in METHODS order, so a
    # running assignment left every row above tfidf without a comparison.
    base_secs = _median_seconds(BASELINE)
    for method in methods:
        cs = curves.get((method, "none"), [])
        secs = _median_seconds(method)
        d2 = weighted_at(cs, 0.5, "recall_final")[0]
        d4 = weighted_at(cs, 0.25, "recall_final")[0]
        d2 = None if d2 is None or base_at[0.5] is None else d2 - base_at[0.5]
        d4 = None if d4 is None or base_at[0.25] is None else d4 - base_at[0.25]
        ratio = "n/a"
        if np.isfinite(base_secs) and np.isfinite(secs) and d4 is not None and abs(d4) > 1e-9:
            ratio = f"{(secs - base_secs) / (d4 * 100.0):+.2f}"
        rel = "n/a" if not np.isfinite(base_secs) or base_secs == 0 else f"{secs / base_secs:.1f}x"
        out.append(f"{method:<26}{secs:>12.2f}{rel:>10}{_fmt(d2, 15)}{_fmt(d4, 15)}{ratio:>16}")
    return out


def plot_elasticity(curves: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    grid = np.linspace(1.1, 10.0, 60)
    for metric, ax, title in (
        ("recall_final", axes[0], "final-state atom recall"),
        ("coverage", axes[1], "semantic coverage"),
    ):
        for method in PLOT_METHODS:
            cs = curves.get((method, "none"), [])
            xs = [m for m in grid if weighted_at(cs, 1.0 / m, metric)[0] is not None]
            ys = [weighted_at(cs, 1.0 / m, metric)[0] for m in xs]
            if xs:
                ax.plot(xs, ys, label=method, linewidth=1.4)
        ax.axhline(KNEE_FLOOR, color="0.6", linestyle=":", linewidth=1)
        ax.set_xlabel("compression multiplier (x)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs compression, token-weighted")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "elasticity.png", dpi=130)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    OUT.mkdir(parents=True, exist_ok=True)
    results = load_checkpoints()
    curves = build_curves(results)
    methods = [m for (m, p) in curves if p == "none"]
    methods = sorted(set(methods), key=methods.index)

    lines = [
        f"Experiment E — analysis over {len(results)} Experiment D checkpoints.",
        "All rates below are ACHIEVED compression ratios, never requested keep_ratio.",
    ]
    lines += e1_elasticity(curves, methods)
    lines += e2_by_kind(curves, methods, results)
    lines += e3_predictors(results, curves)
    lines += e4_cost(curves, methods)
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n", encoding="utf-8")
    plot_elasticity(curves)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

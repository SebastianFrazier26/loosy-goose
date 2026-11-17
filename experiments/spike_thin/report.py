"""Aggregate the spike's results: uv run python experiments/spike_thin/report.py [models...]"""

from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "experiments" / "output" / "spike_thin"
KINDS = ("code", "prose", "tool_result", "tool_use")
LEVELS = ("light", "medium", "heavy")


def boot(vals: list[float], reps: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    a = np.asarray(vals)
    means = a[rng.integers(0, len(a), (reps, len(a)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def groups_select(groups: dict[tuple[str, str, str, float], list[dict[str, Any]]]) -> list[Any]:
    return [r for k, g in groups.items() if k[1] == "select" for r in g]


def main() -> None:
    prep: dict[str, Any] = pickle.loads((OUT / "prep.pkl").read_bytes())
    kinds = {label: p["probe_kind"] for label, p in prep.items()}
    rows: list[dict[str, Any]] = []
    for f in sorted(OUT.glob("results_*.jsonl")):
        rows += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    groups: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["level"], r["arm"], r["model"], r["sel_keep"])].append(r)
    lines = []
    for level in ("full", *LEVELS):
        lines.append(f"\n### {level}\n")
        lines.append(
            "| arm | model | sel keep | tok/budget | over-budget n | atom recall | coverage "
            "| probe survival [95% CI] | " + " | ".join(f"surv {k}" for k in KINDS) + " |"
        )
        lines.append("|" + "---|" * (8 + len(KINDS)))
        keyed = sorted(
            (k for k in groups if k[0] == level),
            key=lambda k: -np.nanmean([r["probe_survival"] for r in groups[k]]),
        )
        for key in keyed:
            g = groups[key]
            surv = [r["probe_survival"] for r in g]
            lo, hi = boot(surv)
            per_kind: dict[str, list[int]] = defaultdict(list)
            for r in g:
                for s, k in zip(r["probe_surv"], kinds[r["label"]], strict=True):
                    if s >= 0:
                        per_kind[k].append(s)
            over = sum(r["tokens"] > r["budget"] for r in g)
            kcells = [
                f"{np.mean(per_kind[k]):.2f} ({len(per_kind[k])})" if per_kind[k] else "-"
                for k in KINDS
            ]
            lines.append(
                f"| {key[1]} | {key[2]} | {key[3]:.2f} "
                f"| {np.mean([r['tokens'] / r['budget'] for r in g]):.3f} | {over} "
                f"| {np.mean([r['atom_recall'] for r in g]):.3f} "
                f"| {np.mean([r['coverage'] for r in g]):.3f} "
                f"| {np.mean(surv):.3f} [{lo:.2f},{hi:.2f}] | " + " | ".join(kcells) + " |"
            )
    lines.append(
        "\n### paired vs select (same transcripts; within-budget arms only; mean diff [95% CI])\n"
    )
    lines.append("| level | arm | model | sel keep | n | d survival | d atom recall | d coverage |")
    lines.append("|---|---|---|---|---|---|---|---|")
    base = {(r["level"], r["label"]): r for r in groups_select(groups)}
    for level in LEVELS:
        cand = []
        for key, g in groups.items():
            if key[0] != level or key[1] in ("select", "full"):
                continue
            if any(r["tokens"] > r["budget"] * 1.01 for r in g):
                continue
            pairs = [(r, base[(level, r["label"])]) for r in g if (level, r["label"]) in base]
            ds = [r["probe_survival"] - b["probe_survival"] for r, b in pairs]
            da = [r["atom_recall"] - b["atom_recall"] for r, b in pairs]
            dc = [r["coverage"] - b["coverage"] for r, b in pairs]
            cand.append((float(np.mean(ds)), key, ds, da, dc))
        for m, key, ds, da, dc in sorted(cand, key=lambda x: -x[0]):
            lo, hi = boot(ds)
            alo, ahi = boot(da)
            lines.append(
                f"| {level} | {key[1]} | {key[2]} | {key[3]:.2f} | {len(ds)} "
                f"| {m:+.3f} [{lo:+.2f},{hi:+.2f}] | {np.mean(da):+.3f} [{alo:+.2f},{ahi:+.2f}] "
                f"| {np.mean(dc):+.3f} |"
            )
    lines.append("\n### model cost\n")
    lines.append(
        "| model | device | load s | score s/transcript (mean) | LM tokens/s "
        "| peak RSS MB | peak MPS MB |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for f in sorted(OUT.glob("meta_*.json")):
        m = json.loads(f.read_text())
        w = m["score_wall_s"]
        tps = (
            sum(m["lm_tokens"].values()) / sum(w[k] for k in m["lm_tokens"])
            if m.get("lm_tokens")
            else float("nan")
        )
        lines.append(
            f"| {m['model']} | {m['device']} | {m['load_s']:.1f} | {np.mean(list(w.values())):.2f} "
            f"| {tps:.0f} | {m['peak_rss_mb']:.0f} | {m['peak_mps_mb']:.0f} |"
        )
    text = "\n".join(lines)
    (OUT / "tables.md").write_text(text)
    print(text if len(sys.argv) < 2 else "")


if __name__ == "__main__":
    main()

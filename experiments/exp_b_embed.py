"""Experiment B: embedding-SVD spectrum, leverage scores and CUR-style subset selection.

Usage: uv run python experiments/exp_b_embed.py [--public-sample 8] [--swe-gym-sample 5]
                                                 [--only NAME_SUBSTRING]
Per transcript writes experiments/output/exp_b/<label>.json and <label>.png, plus a summary
table to experiments/output/exp_b/summary.txt. Local sessions are private: their direction
labels go only into the gitignored JSON, never to stdout or the summary.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from loosy_goose.budget import ProtectPolicy, select_by_score
from loosy_goose.segment import Segment, segment
from loosy_goose.select import (
    Strategy,
    cur_residual_scores,
    embed_segments,
    label_directions,
    leverage_scores,
    rank_for_energy,
    svd_energy,
)
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Transcript, load_claude_code_jsonl, load_messages_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "experiments" / "output" / "exp_b"
CACHE = ROOT / "experiments" / "output" / "cache"
KINDS = ("prose", "code", "tool_result", "tool_use", "thinking")
KEEP_RATIOS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
PROTECTS: tuple[ProtectPolicy, ...] = ("all", "code_only", "none")
STRATEGIES: tuple[Strategy, ...] = ("leverage", "cur_residual")
ENERGIES = (0.90, 0.95, 0.99)
N_LABEL_DIRECTIONS = 8


def _spread(items: list[Any], n: int) -> list[Any]:
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def iter_local() -> Iterator[tuple[str, bool, Transcript]]:
    for p in sorted((DATA / "local" / "sessions").glob("*.jsonl")):
        yield f"local_{p.stem}", False, load_claude_code_jsonl(p)


def iter_trace_commons(sample: int) -> Iterator[tuple[str, bool, Transcript]]:
    files = sorted(
        (DATA / "public" / "trace-commons" / "sessions" / "claude_code").glob("*.jsonl"),
        key=lambda p: p.stat().st_size,
    )
    for p in _spread(files, sample):
        yield f"trace-commons_{p.stem[:8]}", True, load_claude_code_jsonl(p)


def iter_swe_gym(sample: int) -> Iterator[tuple[str, bool, Transcript]]:
    files = sorted((DATA / "public" / "swe-gym").rglob("*.parquet"))
    if not files:
        return
    import pyarrow.parquet as pq

    table = pq.read_table(files[0])
    col = "messages" if "messages" in table.column_names else table.column_names[0]
    msgs = table.column(col).to_pylist()
    order = sorted(range(len(msgs)), key=lambda i: len(json.dumps(msgs[i], default=str)))
    for i in _spread(order, sample):
        payload = msgs[i]
        if isinstance(payload, str):
            payload = json.loads(payload)
        yield f"swe-gym_row{i}", True, load_messages_json(payload, name=f"swe-gym-{i}")


def _quantiles(v: np.ndarray) -> dict[str, float]:
    qs = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
    return {f"q{int(q * 100):02d}": float(x) for q, x in zip(qs, np.quantile(v, qs), strict=True)}


def _gini(v: np.ndarray) -> float:
    s = np.sort(v)
    n = s.size
    if n == 0 or s.sum() == 0:
        return 0.0
    return float((2 * np.sum((np.arange(1, n + 1)) * s) / (n * s.sum())) - (n + 1) / n)


def _plot(
    label: str, s: np.ndarray, energy: np.ndarray, lev: np.ndarray, ranks: dict[str, int]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    idx = np.arange(1, s.size + 1)
    axes[0].semilogy(idx, s, lw=1.2)
    axes[0].set_title("singular values (centred)")
    axes[0].set_xlabel("component")
    axes[1].plot(idx, energy, lw=1.2)
    for e, k in ranks.items():
        axes[1].axvline(k, color="grey", ls="--", lw=0.8)
        axes[1].annotate(f"{e}: k={k}", (k, 0.2 + 0.15 * list(ranks).index(e)), fontsize=8)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("cumulative energy")
    axes[1].set_xlabel("rank")
    axes[2].hist(lev, bins=50, color="tab:blue", alpha=0.8)
    axes[2].set_title("leverage scores @0.95")
    axes[2].set_xlabel("leverage")
    fig.suptitle(f"{label}  n={s.size + 1 if s.size else 0}")
    fig.tight_layout()
    fig.savefig(OUT / f"{label}.png", dpi=110)
    plt.close(fig)


def _kept_summary(segs: list[Segment], kept: list[Segment], total_tokens: int) -> dict[str, Any]:
    by_kind: Counter[str] = Counter(s.kind for s in kept)
    kept_tokens = sum(count_tokens(s.text) for s in kept)
    return {
        "kept_segments": len(kept),
        "kept_token_fraction": kept_tokens / total_tokens if total_tokens else 0.0,
        "kept_by_kind": {k: by_kind.get(k, 0) for k in KINDS},
        "kept_ids": [s.id for s in kept],
    }


def analyse(label: str, public: bool, t: Transcript) -> dict[str, Any]:
    segs = segment(t)
    n = len(segs)
    t0 = time.perf_counter()
    x = embed_segments(segs, cache_dir=CACHE)
    embed_seconds = time.perf_counter() - t0
    total_tokens = sum(count_tokens(s.text) for s in segs)
    result: dict[str, Any] = {
        "label": label,
        "public": public,
        "n_turns": len(t.turns),
        "n_segments": n,
        "total_tokens": total_tokens,
        "segments_by_kind": dict(Counter(s.kind for s in segs)),
        "embed_seconds": embed_seconds,
        "embed_dim": int(x.shape[1]) if x.ndim == 2 and n else 0,
    }
    if n < 3:
        result["skipped"] = "too few segments"
        return result

    u, s, vt, energy = svd_energy(x)
    ranks = {f"{e:.2f}": rank_for_energy(s, e) for e in ENERGIES}
    k95 = ranks["0.95"]
    lev = leverage_scores(u, k95)
    result.update(
        {
            "singular_values": s.tolist(),
            "energy_cumulative": energy.tolist(),
            "rank_at_energy": ranks,
            "rank_fraction_at_0.95": k95 / n,
            "leverage": {
                "k": k95,
                "sum": float(lev.sum()),
                "gini": _gini(lev),
                "quantiles": _quantiles(lev),
                "top_kinds": dict(Counter(segs[i].kind for i in np.argsort(-lev)[:20])),
            },
        }
    )
    scores: dict[str, np.ndarray] = {
        "leverage": lev,
        "cur_residual": cur_residual_scores(u, s, k95),
    }
    top_n = max(1, n // 10)
    top_lev = set(np.argsort(-scores["leverage"])[:top_n].tolist())
    top_cur = set(np.argsort(-scores["cur_residual"])[:top_n].tolist())
    result["strategy_top10pct_overlap"] = len(top_lev & top_cur) / top_n

    grid: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for protect in PROTECTS:
            for ratio in KEEP_RATIOS:
                kept = select_by_score(segs, scores[strategy], ratio, protect)
                row = {"strategy": strategy, "protect": protect, "keep_ratio": ratio}
                row.update(_kept_summary(segs, kept, total_tokens))
                grid.append(row)
    result["grid"] = grid
    overlap: list[dict[str, Any]] = []
    for ratio in KEEP_RATIOS:
        a = {s_.id for s_ in select_by_score(segs, scores["leverage"], ratio, "none")}
        b = {s_.id for s_ in select_by_score(segs, scores["cur_residual"], ratio, "none")}
        overlap.append({"keep_ratio": ratio, "jaccard": len(a & b) / len(a | b) if a | b else 1.0})
    result["strategy_selection_jaccard_protect_none"] = overlap

    k_label = min(N_LABEL_DIRECTIONS, k95)
    result["direction_labels"] = label_directions(segs, vt, u, k_label)
    _plot(label, s, energy, lev, ranks)
    return result


def _summary_row(r: dict[str, Any]) -> str:
    n = r["n_segments"]
    if "skipped" in r:
        return f"{r['label']:<36} {n:>6}  {r['skipped']}"
    ranks = r["rank_at_energy"]
    cells = [
        f"{r['label']:<36}",
        f"{n:>6}",
        f"{ranks['0.90']:>5}",
        f"{ranks['0.95']:>5}",
        f"{ranks['0.99']:>5}",
        f"{r['rank_fraction_at_0.95']:>6.2f}",
        f"{r['strategy_top10pct_overlap']:>6.2f}",
        f"{r['embed_seconds']:>7.1f}",
    ]
    if r["public"]:
        labels = r["direction_labels"][:2]
        cells.append("  ".join("/".join(words[:5]) for words in labels))
    else:
        cells.append("(private)")
    return " ".join(cells)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-sample", type=int, default=8)
    parser.add_argument("--swe-gym-sample", type=int, default=5)
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    header = " ".join(
        [
            f"{'transcript':<36}",
            f"{'segs':>6}",
            f"{'k@.90':>5}",
            f"{'k@.95':>5}",
            f"{'k@.99':>5}",
            f"{'k95/n':>6}",
            f"{'ovl10':>6}",
            f"{'emb_s':>7}",
            "top-2 direction labels (public only)",
        ]
    )
    lines = [header, "-" * len(header)]
    sources = [
        iter_local(),
        iter_trace_commons(args.public_sample),
        iter_swe_gym(args.swe_gym_sample),
    ]
    for src in sources:
        for label, public, t in src:
            if args.only and args.only not in label:
                continue
            r = analyse(label, public, t)
            (OUT / f"{label}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
            row = _summary_row(r)
            print(row, flush=True)
            lines.append(row)
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

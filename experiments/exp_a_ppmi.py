"""Experiment A: does Shin et al. 2018 PPMI-SVD eigenvector analysis survive at
single-conversation scale, and is the eigenbasis stable enough to drive extractive compression?

Usage: uv run python experiments/exp_a_ppmi.py [--only SUBSTR] [--no-background] [--folds N]
Writes experiments/output/exp_a/<label>.json + .png and summary.{txt,json}.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import scipy.sparse as sp

from loosy_goose.cooccur import Vocab, align_vocab, cooccurrence, differential_ppmi, tokenize
from loosy_goose.segment import Segment, segment
from loosy_goose.topics import (
    EigenTopics,
    TopicConfig,
    density_order,
    fit_from_token_lists,
    fit_topics,
    jackknife_stability,
    top_words,
)
from loosy_goose.transcript import Transcript, load_claude_code_jsonl, load_messages_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "experiments" / "output" / "exp_a"

CONFIGS = [
    TopicConfig(window=w, min_count=mc, shift=sh, cds=0.75)
    for w in (2, 5)
    for mc in (2, 5)
    for sh in (1.0, 5.0)
]
DEFAULT = TopicConfig()
BG_MIN_COUNT = 5
SLOW_SECONDS = 180.0
# dataviz reference palette, categorical slots 1 and 2 (light surface).
SERIES = ("#2a78d6", "#eb6834")


def cfg_label(cfg: TopicConfig) -> str:
    return f"w{cfg.window}_mc{cfg.min_count}_s{int(cfg.shift)}"


def iter_local() -> Iterator[tuple[str, Transcript, bool]]:
    for p in sorted((DATA / "local" / "sessions").glob("*.jsonl")):
        yield f"local/{p.stem.split('_')[0]}", load_claude_code_jsonl(p), True


def trace_commons_files() -> list[Path]:
    return sorted((DATA / "public" / "trace-commons").rglob("sessions/claude_code/*.jsonl"))


def spread(items: list[Any], n: int) -> list[Any]:
    if len(items) <= n:
        return items
    idx = np.unique(np.round(np.linspace(0, len(items) - 1, n)).astype(int))
    return [items[i] for i in idx]


def iter_trace_commons(n: int) -> Iterator[tuple[str, Transcript, bool]]:
    files = sorted(trace_commons_files(), key=lambda p: p.stat().st_size)
    for p in spread(files, n):
        yield f"tc/{p.stem[:8]}", load_claude_code_jsonl(p), False


def swe_gym_rows() -> list[tuple[int, Any]]:
    files = sorted((DATA / "public" / "swe-gym").rglob("*.parquet"))
    if not files:
        return []
    import pyarrow.parquet as pq

    table = pq.read_table(files[0])
    col = "messages" if "messages" in table.column_names else table.column_names[0]
    msgs = table.column(col).to_pylist()
    rows = []
    for i, payload in enumerate(msgs):
        if isinstance(payload, str):
            payload = json.loads(payload)
        rows.append((i, payload))
    return rows


def iter_swe_gym(rows: list[tuple[int, Any]], n: int) -> Iterator[tuple[str, Transcript, bool]]:
    ordered = sorted(rows, key=lambda r: len(json.dumps(r[1], default=str)))
    for i, payload in spread(ordered, n):
        yield f"swe/row{i}", load_messages_json(payload, name=f"swe-gym-{i}"), False


def token_lists_by_turn(segs: list[Segment]) -> list[list[list[str]]]:
    turns: dict[int, list[list[str]]] = {}
    for s in segs:
        turns.setdefault(s.turn, []).append(tokenize(s.text))
    return [turns[t] for t in sorted(turns)]


def describe(topics: EigenTopics, n_dense: int = 3, n_sparse: int = 10) -> dict[str, Any]:
    order = density_order(topics)
    part = topics.participation
    return {
        "vocab": len(topics.vocab),
        "k": topics.k,
        "eigenvalues": topics.s.tolist(),
        "ipr": topics.ipr.tolist(),
        "participation": {
            "mean": float(part.mean()),
            "min": float(part.min()),
            "max": float(part.max()),
        },
        "densest": [
            {"index": int(j), "ipr": float(topics.ipr[j]), "words": top_words(topics, int(j))}
            for j in order[:n_dense]
        ],
        "sparsest": [
            {"index": int(j), "ipr": float(topics.ipr[j]), "words": top_words(topics, int(j))}
            for j in order[::-1][:n_sparse]
        ],
    }


def stability_summary(cos: np.ndarray, topics: EigenTopics) -> dict[str, Any]:
    sparse_idx = density_order(topics)[::-1][:20]
    return {
        "per_eigenvector": cos.tolist(),
        "top20": float(cos[:20].mean()),
        "sparsest20": float(cos[sparse_idx].mean()),
        "all": float(cos.mean()),
    }


class Background:
    """Pooled window-2 co-occurrence counts over every public transcript, in a shared vocab.

    Per-transcript matrices are kept so the transcript under test can be subtracted (leave-one-out)
    instead of letting it explain itself away.
    """

    def __init__(self, token_lists: dict[str, list[list[str]]]) -> None:
        self.vocab = Vocab.build(
            (toks for lists in token_lists.values() for toks in lists), min_count=BG_MIN_COUNT
        )
        self.per_transcript = {
            label: cooccurrence(lists, self.vocab, window=DEFAULT.window)
            for label, lists in token_lists.items()
        }
        total = sp.csr_matrix((len(self.vocab), len(self.vocab)), dtype=np.float64)
        for m in self.per_transcript.values():
            total = total + m
        self.total = sp.csr_matrix(total)

    def counts_excluding(self, label: str) -> sp.csr_matrix:
        own = self.per_transcript.get(label)
        return self.total if own is None else sp.csr_matrix(self.total - own)


def plot_ipr(label: str, base: EigenTopics, bg: EigenTopics | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=110)
    ax.scatter(base.s, base.ipr, s=36, c=SERIES[0], alpha=0.85, label="PPMI", edgecolors="none")
    if bg is not None:
        ax.scatter(
            bg.s,
            bg.ipr,
            s=36,
            c=SERIES[1],
            alpha=0.85,
            label="PPMI - background",
            marker="D",
            edgecolors="none",
        )
    ax.set_yscale("log")
    ax.set_xlabel("singular value")
    ax.set_ylabel("IPR (log)")
    ax.set_title(f"{label}: IPR vs eigenvalue, |V|={len(base.vocab)}, k={base.k}", fontsize=10)
    ax.grid(True, color="#e6e6e3", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if bg is not None:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_transcript(
    label: str,
    transcript: Transcript,
    private: bool,
    background: Background | None,
    folds: int,
) -> dict[str, Any]:
    segs = segment(transcript)
    token_lists = [tokenize(s.text) for s in segs]
    by_turn = token_lists_by_turn(segs)
    n_tokens = sum(len(t) for t in token_lists)
    result: dict[str, Any] = {
        "label": label,
        "private": private,
        "segments": len(segs),
        "turns": len(transcript.turns),
        "tokens": n_tokens,
        "configs": {},
    }
    k_cap = 200
    for cfg in CONFIGS:
        t0 = time.perf_counter()
        run_cfg = TopicConfig(**{**asdict(cfg), "k": None if k_cap == 200 else k_cap})
        fitted = fit_from_token_lists(token_lists, run_cfg, n_segments=len(segs))
        elapsed = time.perf_counter() - t0
        entry: dict[str, Any] = {"config": asdict(cfg), "seconds": elapsed}
        if fitted is None:
            entry["skipped"] = "vocab too small"
        else:
            entry.update(describe(fitted[0]))
            if elapsed > SLOW_SECONDS:
                k_cap = max(8, fitted[0].k // 2)
                entry["note"] = f"exceeded {SLOW_SECONDS:.0f}s; later configs capped at k={k_cap}"
        result["configs"][cfg_label(cfg)] = entry
        print(
            f"  {label} {cfg_label(cfg)} |V|={entry.get('vocab')} k={entry.get('k')} {elapsed:.1f}s"
        )

    default_fit = fit_from_token_lists(token_lists, DEFAULT, n_segments=len(segs))
    base_topics: EigenTopics | None = None
    bg_topics: EigenTopics | None = None
    if default_fit is not None:
        base_topics, fg_counts = default_fit
        k = base_topics.k
        t0 = time.perf_counter()
        cos = jackknife_stability(by_turn, base_topics.vocab, k, folds=folds, config=DEFAULT)
        result["stability"] = stability_summary(cos, base_topics)
        result["stability"]["seconds"] = time.perf_counter() - t0
        print(
            f"  {label} jackknife top20={result['stability']['top20']:.3f} "
            f"sparsest20={result['stability']['sparsest20']:.3f} "
            f"{result['stability']['seconds']:.1f}s"
        )

        if background is not None:
            bg_counts = background.counts_excluding(label)
            vmap = align_vocab(base_topics.vocab, background.vocab)

            def fn(c: sp.csr_matrix) -> sp.csr_matrix:
                return differential_ppmi(c, bg_counts, vmap, cds=DEFAULT.cds, shift=DEFAULT.shift)

            t0 = time.perf_counter()
            dppmi = fn(fg_counts)
            bg_entry: dict[str, Any] = {
                "bg_vocab": len(background.vocab),
                "fg_words_in_bg": int((vmap >= 0).sum()),
                "nnz_before": int(fg_counts.nnz),
                "nnz_after": int(dppmi.nnz),
            }
            if dppmi.nnz > 0:
                bg_topics = fit_topics(dppmi, base_topics.vocab, k)
                bg_entry.update(describe(bg_topics))
                bg_cos = jackknife_stability(
                    by_turn, base_topics.vocab, k, folds=folds, config=DEFAULT, ppmi_fn=fn
                )
                bg_entry["stability"] = stability_summary(bg_cos, bg_topics)
                print(
                    f"  {label} background top20={bg_entry['stability']['top20']:.3f} "
                    f"sparsest20={bg_entry['stability']['sparsest20']:.3f}"
                )
            bg_entry["seconds"] = time.perf_counter() - t0
            result["background"] = bg_entry
    if base_topics is not None:
        plot_ipr(label, base_topics, bg_topics, OUT / f"{label.replace('/', '_')}.png")
    return result


def sample_topics(res: dict[str, Any], n_topics: int = 3, n_words: int = 6) -> str:
    if res["private"]:
        return "(local session: withheld)"
    default = res["configs"].get(cfg_label(DEFAULT), {})
    lists = [
        "/".join(w for w, _ in t["words"][:n_words]) for t in default.get("sparsest", [])[:n_topics]
    ]
    return " | ".join(lists)


def summary_table(results: list[dict[str, Any]]) -> str:
    cols = [
        ("transcript", 16),
        ("tokens", 8),
        ("|V|", 6),
        ("k", 4),
        ("part%", 6),
        ("st_top20", 8),
        ("st_sp20", 7),
        ("bg_top20", 8),
        ("bg_sp20", 7),
    ]
    header = " ".join(f"{name:>{w}}" for name, w in cols) + "  sample sparse topics"
    lines = [header, "-" * len(header)]
    for r in results:
        d = r["configs"].get(cfg_label(DEFAULT), {})
        st = r.get("stability", {})
        bg = r.get("background", {}).get("stability", {})
        part = d.get("participation", {}).get("mean")
        cells = [
            f"{r['label']:>16}",
            f"{r['tokens']:>8}",
            f"{d.get('vocab', '-'):>6}",
            f"{d.get('k', '-'):>4}",
            f"{(part * 100 if part is not None else float('nan')):>6.1f}",
            f"{st.get('top20', float('nan')):>8.3f}",
            f"{st.get('sparsest20', float('nan')):>7.3f}",
            f"{bg.get('top20', float('nan')):>8.3f}",
            f"{bg.get('sparsest20', float('nan')):>7.3f}",
        ]
        lines.append(" ".join(cells) + "  " + sample_topics(r))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", default=None, help="run only transcripts whose label contains this"
    )
    parser.add_argument("--no-background", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trace-commons", type=int, default=8)
    parser.add_argument("--swe-gym", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = swe_gym_rows()
    under_test = list(iter_local())
    under_test += list(iter_trace_commons(args.trace_commons))
    under_test += list(iter_swe_gym(rows, args.swe_gym))
    if args.only:
        under_test = [u for u in under_test if args.only in u[0]]

    background: Background | None = None
    if not args.no_background:
        t0 = time.perf_counter()
        pooled: dict[str, list[list[str]]] = {}
        for p in trace_commons_files():
            pooled[f"tc/{p.stem[:8]}"] = [
                tokenize(s.text) for s in segment(load_claude_code_jsonl(p))
            ]
        for i, payload in rows:
            t = load_messages_json(payload, name=f"swe-gym-{i}")
            pooled[f"swe/row{i}"] = [tokenize(s.text) for s in segment(t)]
        background = Background(pooled)
        print(
            f"background: {len(pooled)} transcripts, |V|={len(background.vocab)}, "
            f"nnz={background.total.nnz}, {time.perf_counter() - t0:.1f}s"
        )

    results: list[dict[str, Any]] = []
    for label, transcript, private in under_test:
        print(f"== {label}")
        res = run_transcript(label, transcript, private, background, args.folds)
        results.append(res)
        (OUT / f"{label.replace('/', '_')}.json").write_text(
            json.dumps(res, indent=1), encoding="utf-8"
        )

    table = summary_table(results)
    print()
    print(table)
    (OUT / "summary.txt").write_text(table + "\n", encoding="utf-8")
    (OUT / "summary.json").write_text(
        json.dumps(
            [
                {k: v for k, v in r.items() if k != "configs"}
                | {"default": r["configs"].get(cfg_label(DEFAULT), {})}
                for r in results
            ],
            indent=1,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

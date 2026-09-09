"""Experiment C: the code channel.

C2 supersession dedupe (lossless w.r.t. final file state), then C1 structural quantization of
code + tool_use segments at quality in {1.0, 0.66, 0.33, 0.0} against whole-segment dropping at
the same token budget, then C3 code-specialised embedder feasibility on code-only segments.

Usage: uv run python experiments/exp_c_code.py [--public-sample 8] [--swe-gym-sample 5]
                                               [--only NAME_SUBSTRING] [--skip-c3]
                                               [--code-models MODEL ...]
Writes experiments/output/exp_c/{c1,c2,c3}.json and summary.txt. Local sessions are private:
only aggregate numbers leave the JSON; no segment text is printed for them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from loosy_goose import embed, metrics
from loosy_goose.budget import select_by_score
from loosy_goose.code import quantize_segment
from loosy_goose.segment import Segment, segment
from loosy_goose.select import embed_segments, leverage_scores, rank_for_energy, svd_energy
from loosy_goose.supersede import find_superseded, tool_names_from_transcript
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Transcript, load_claude_code_jsonl, load_messages_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "experiments" / "output" / "exp_c"
CACHE = ROOT / "experiments" / "output" / "cache"
QUALITIES = (1.0, 0.66, 0.33, 0.0)
CODE_KINDS = ("code", "tool_use")
C3_KEEP_RATIO = 0.3
# Both load through plain SentenceTransformer(name) with no trust_remote_code: DistilRoBERTa and
# ModernBERT are native transformers architectures. jina-v2-code, CodeRankEmbed and codet5p all
# need trust_remote_code=True and are deliberately not on this list.
DEFAULT_CODE_MODELS = (
    "flax-sentence-embeddings/st-codesearch-distilroberta-base",
    "Alibaba-NLP/gte-modernbert-base",
)


def _spread(items: list[Any], n: int) -> list[Any]:
    if len(items) <= n:
        return items
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def iter_local() -> Iterator[tuple[str, bool, Transcript]]:
    for p in sorted((DATA / "local" / "sessions").glob("*.jsonl")):
        yield f"local_{p.stem.split('_')[0]}", False, load_claude_code_jsonl(p)


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


def _tokens(segs: list[Segment]) -> int:
    return sum(count_tokens(s.text) for s in segs)


def _lang_hints(t: Transcript, segs: list[Segment]) -> dict[int, str]:
    # Code segments are atomic copies of their block, so (turn, text) recovers the fence label
    # that segmentation dropped.
    by_key: dict[tuple[int, str], str] = {}
    for turn in t.turns:
        for b in turn.blocks:
            if b.kind == "code" and b.meta.get("lang"):
                by_key[(turn.index, b.text)] = b.meta["lang"]
    return {s.id: by_key[(s.turn, s.text)] for s in segs if (s.turn, s.text) in by_key}


def run_c2(label: str, t: Transcript, segs: list[Segment]) -> dict[str, Any]:
    total = _tokens(segs)
    names = tool_names_from_transcript(t, segs)
    gone = find_superseded(segs, tool_names=names)
    by_reason_tokens: Counter[str] = Counter()
    by_reason_count: Counter[str] = Counter()
    by_id = {s.id: s for s in segs}
    for sid, sup in gone.items():
        by_reason_tokens[sup.reason] += count_tokens(by_id[sid].text)
        by_reason_count[sup.reason] += 1
    removed = sum(by_reason_tokens.values())
    return {
        "label": label,
        "total_tokens": total,
        "n_segments": len(segs),
        "removed_tokens": removed,
        "removed_fraction": removed / total if total else 0.0,
        "removed_segments": len(gone),
        "by_reason_tokens": dict(by_reason_tokens),
        "by_reason_fraction": {k: v / total for k, v in by_reason_tokens.items()} if total else {},
        "by_reason_segments": dict(by_reason_count),
    }


def _tfidf_scores(segs: list[Segment]) -> np.ndarray:
    if len(segs) < 2:
        return np.ones(len(segs))
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][A-Za-z0-9_]{1,}", sublinear_tf=True)
    try:
        x = vec.fit_transform([s.text for s in segs])
    except ValueError:
        return np.ones(len(segs))
    return np.asarray(x.sum(axis=1)).ravel()


def _score(original: list[Segment], kept: list[Segment], vectors: np.ndarray) -> dict[str, Any]:
    ev = metrics.evaluate(original, kept, original_vectors=vectors)
    return {
        "kept_token_fraction": ev["compression_ratio"],
        "atom_recall": ev["atom_recall"],
        "coverage": ev["coverage"],
        "doc_cosine": ev["doc_cosine"],
    }


def run_c1(label: str, t: Transcript, segs: list[Segment]) -> dict[str, Any]:
    code = [s for s in segs if s.kind in CODE_KINDS]
    result: dict[str, Any] = {
        "label": label,
        "n_code_segments": len(code),
        "code_tokens": _tokens(code),
        "code_token_share": _tokens(code) / _tokens(segs) if segs else 0.0,
        "by_kind": dict(Counter(s.kind for s in code)),
    }
    if len(code) < 2:
        result["skipped"] = "too few code segments"
        return result
    hints = _lang_hints(t, code)
    vectors = embed_segments(code, embed.SCORE_MODEL, cache_dir=CACHE)
    tfidf = _tfidf_scores(code)
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for q in QUALITIES:
        t0 = time.perf_counter()
        quantized = [quantize_segment(s, q, lang=hints.get(s.id)) for s in code]
        quant_seconds = time.perf_counter() - t0
        row: dict[str, Any] = {"quality": q, "quantize_seconds": quant_seconds}
        row["structural"] = _score(code, quantized, vectors)
        f = row["structural"]["kept_token_fraction"]
        if q < 1.0 and 0.0 < f < 1.0:
            row["drop_tfidf"] = _score(code, select_by_score(code, tfidf, f, "none"), vectors)
            row["drop_random"] = _score(
                code, select_by_score(code, rng.random(len(code)), f, "none"), vectors
            )
        rows.append(row)
    result["grid"] = rows
    return result


def run_c3(label: str, segs: list[Segment], code_models: tuple[str, ...]) -> dict[str, Any]:
    code = [s for s in segs if s.kind == "code"]
    result: dict[str, Any] = {"label": label, "n_code_segments": len(code)}
    if len(code) < 5:
        result["skipped"] = "too few code segments"
        return result
    score_vectors = embed_segments(code, embed.SCORE_MODEL, cache_dir=CACHE)
    per_model: dict[str, Any] = {}
    for model in (embed.SELECT_MODEL, *code_models):
        t0 = time.perf_counter()
        try:
            x = embed_segments(code, model, cache_dir=CACHE)
        except Exception as exc:  # noqa: BLE001 - report load failures instead of aborting
            per_model[model] = {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
            continue
        seconds = time.perf_counter() - t0
        u, s, _, _ = svd_energy(x)
        k95 = rank_for_energy(s, 0.95)
        lev = leverage_scores(u, k95)
        kept = select_by_score(code, lev, C3_KEEP_RATIO, "none")
        per_model[model] = {
            "embed_seconds": seconds,
            "seconds_per_100": 100 * seconds / len(code),
            "dim": int(x.shape[1]),
            "rank_at_0.95": k95,
            "rank_fraction_at_0.95": k95 / len(code),
            "leverage_keep_0.3": _score(code, kept, score_vectors),
        }
    result["models"] = per_model
    return result


def _fmt(x: float | None) -> str:
    return f"{x:6.3f}" if x is not None else "     -"


def _c2_table(rows: list[dict[str, Any]]) -> list[str]:
    head = (
        f"{'transcript':<28} {'tokens':>8} {'removed':>8} {'frac':>6} "
        f"{'edit':>6} {'read':>6} {'dup':>6}"
    )
    out = [head, "-" * len(head)]
    for r in rows:
        fr = r["by_reason_fraction"]
        out.append(
            f"{r['label']:<28} {r['total_tokens']:>8} {r['removed_tokens']:>8} "
            f"{r['removed_fraction']:6.3f} {fr.get('edit_superseded', 0.0):6.3f} "
            f"{fr.get('read_superseded', 0.0):6.3f} {fr.get('duplicate', 0.0):6.3f}"
        )
    total = sum(r["total_tokens"] for r in rows)
    removed = sum(r["removed_tokens"] for r in rows)
    agg: Counter[str] = Counter()
    for r in rows:
        agg.update(r["by_reason_tokens"])
    out.append(
        f"{'ALL':<28} {total:>8} {removed:>8} {removed / total if total else 0:6.3f} "
        f"{agg['edit_superseded'] / total if total else 0:6.3f} "
        f"{agg['read_superseded'] / total if total else 0:6.3f} "
        f"{agg['duplicate'] / total if total else 0:6.3f}"
    )
    return out


def _c1_table(rows: list[dict[str, Any]]) -> list[str]:
    head = (
        f"{'transcript':<28} {'q':>4} {'kept':>6} | {'struct':^20} | {'tfidf-drop':^20} | "
        f"{'random-drop':^20}"
    )
    sub = f"{'':<28} {'':>4} {'':>6} | {'recall':>6} {'cover':>6} {'dcos':>6} | " * 1
    sub += f"{'recall':>6} {'cover':>6} {'dcos':>6} | {'recall':>6} {'cover':>6} {'dcos':>6}"
    out = [head, sub, "-" * len(head)]
    for r in rows:
        if "skipped" in r:
            out.append(f"{r['label']:<28} {r['skipped']}")
            continue
        for g in r["grid"]:
            cells = [f"{r['label']:<28} {g['quality']:>4.2f}"]
            s = g["structural"]
            cells.append(f"{s['kept_token_fraction']:6.3f} |")
            for key in ("structural", "drop_tfidf", "drop_random"):
                m = g.get(key)
                if m is None:
                    cells.append(f"{'-':>6} {'-':>6} {'-':>6} |")
                else:
                    cells.append(
                        f"{m['atom_recall']:6.3f} {m['coverage']:6.3f} {m['doc_cosine']:6.3f} |"
                    )
            out.append(" ".join(cells))
    return out


def _c1_aggregate(rows: list[dict[str, Any]]) -> list[str]:
    out = ["", "C1 token-weighted means over all transcripts (code + tool_use segments only):"]
    head = (
        f"{'q':>4} {'kept':>6} | {'struct recall/cover':>20} | "
        f"{'tfidf recall/cover':>20} | {'random recall/cover':>20}"
    )
    out += [head, "-" * len(head)]
    for qi, q in enumerate(QUALITIES):
        w: list[float] = []
        acc: dict[str, list[tuple[float, float, float]]] = {
            "structural": [],
            "drop_tfidf": [],
            "drop_random": [],
        }
        for r in rows:
            if "skipped" in r:
                continue
            g = r["grid"][qi]
            w.append(r["code_tokens"])
            for key in acc:
                m = g.get(key)
                acc[key].append(
                    (m["kept_token_fraction"], m["atom_recall"], m["coverage"])
                    if m
                    else (np.nan,) * 3
                )
        if not w:
            continue
        wt = np.asarray(w)
        cells = [f"{q:>4.2f}"]
        for key in ("structural", "drop_tfidf", "drop_random"):
            a = np.asarray(acc[key])
            mask = ~np.isnan(a[:, 0])
            if not mask.any():
                cells.append(f"{'-':>20} |")
                continue
            ww = wt[mask] / wt[mask].sum()
            kept, rec, cov = (float(np.sum(a[mask, i] * ww)) for i in range(3))
            if key == "structural":
                cells.insert(1, f"{kept:6.3f} |")
            cells.append(f"{rec:9.3f} / {cov:8.3f} |")
        out.append(" ".join(cells))
    return out


def _c3_table(rows: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    head = (
        f"{'transcript':<28} {'model':<52} {'n':>5} {'dim':>4} {'k95/n':>6} {'s/100':>6} "
        f"{'recall':>6} {'cover':>6} {'dcos':>6}"
    )
    out += [head, "-" * len(head)]
    for r in rows:
        if "skipped" in r:
            out.append(f"{r['label']:<28} {r['skipped']}")
            continue
        for model, m in r["models"].items():
            if "error" in m:
                out.append(f"{r['label']:<28} {model:<52} {m['error']}")
                continue
            lk = m["leverage_keep_0.3"]
            out.append(
                f"{r['label']:<28} {model:<52} {r['n_code_segments']:>5} {m['dim']:>4} "
                f"{m['rank_fraction_at_0.95']:6.3f} {m['seconds_per_100']:6.2f} "
                f"{lk['atom_recall']:6.3f} {lk['coverage']:6.3f} {lk['doc_cosine']:6.3f}"
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-sample", type=int, default=8)
    parser.add_argument("--swe-gym-sample", type=int, default=5)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--skip-c3", action="store_true")
    parser.add_argument("--code-models", nargs="*", default=list(DEFAULT_CODE_MODELS))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    c1: list[dict[str, Any]] = []
    c2: list[dict[str, Any]] = []
    c3: list[dict[str, Any]] = []
    c3_labels = {"local_20k", "local_50k"}
    public_c3_seen = 0
    sources = [
        iter_local(),
        iter_trace_commons(args.public_sample),
        iter_swe_gym(args.swe_gym_sample),
    ]
    for src in sources:
        for label, _public, t in src:
            if args.only and args.only not in label:
                continue
            segs = segment(t)
            print(f"[{label}] {len(segs)} segments", flush=True)
            c2.append(run_c2(label, t, segs))
            c1.append(run_c1(label, t, segs))
            if not args.skip_c3:
                want = label in c3_labels or (
                    label.startswith("trace-commons") and public_c3_seen < 3
                )
                if want:
                    if label.startswith("trace-commons"):
                        public_c3_seen += 1
                    c3.append(run_c3(label, segs, tuple(args.code_models)))
    (OUT / "c2.json").write_text(json.dumps(c2, indent=1), encoding="utf-8")
    (OUT / "c1.json").write_text(json.dumps(c1, indent=1), encoding="utf-8")
    (OUT / "c3.json").write_text(json.dumps(c3, indent=1), encoding="utf-8")

    lines = ["== C2 supersession (lossless w.r.t. final state): tokens removed / total =="]
    lines += _c2_table(c2)
    lines += ["", "== C1 structural quantization vs whole-segment drop at equal token budget =="]
    lines += _c1_table(c1)
    lines += _c1_aggregate(c1)
    if c3:
        lines += [
            "",
            "== C3 code-only segments: embedder comparison (leverage keep 0.3, MiniLM-scored) ==",
        ]
        lines += _c3_table(c3)
    text = "\n".join(lines)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prefill latency replay (2026-11-17): what each continuous arm's context stream costs to prefill
on Qwen3-8B through mlx-lm. One prompt cache per (arm, transcript) is trimmed to the common
token prefix with the previous boundary's context and only the rest is prefilled, which is the
prefix-cache behaviour `sim.uncached_tokens` models in text space. Timed around the prefill only.

Prefill is run by feeding the new tokens through the model in chunks and evaluating the cache
(no sampling), so the cache ends exactly at the context length; `stream_generate` with
max_tokens=1 would leave a sampled-token residue in it.

`--dry-run` builds the contexts and Qwen token counts with the tokenizer alone, no model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments import exp_i_downstream as exi  # noqa: E402
from experiments.continuous import local, run, sim  # noqa: E402

OUT = run.OUT / "f" / "latency.json"
CHUNK = 2048
DEFAULT_ARMS = ("qthresh", "tloosy", "cfrzm")


def reused_prefix(prev_tokens: Sequence[int], cur_tokens: Sequence[int]) -> int:
    n = 0
    for a, b in zip(prev_tokens, cur_tokens, strict=False):
        if a != b:
            break
        n += 1
    return n


def chat_tokens(context: str) -> list[int]:
    tok = local._tokenizer(run.QWEN.model_id)
    ids = tok.apply_chat_template(
        [{"role": "user", "content": context}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return list(ids)


def boundary_contexts(arm: str, p: sim.TranscriptPlan) -> Iterator[tuple[int, str]] | None:
    """(boundary, cost_context) per boundary; None when cached summaries are missing."""
    summarize: Any = sim.placeholder_summary
    if arm == "qthresh":
        summarize = run.QSummaries(p.label, strict=True)
    try:
        pairs = [
            (t, sim.cost_context(view, p.turns, t))
            for t, view, _ in sim._views(arm, p, exi.loosy_compressor, summarize)
        ]
    except sim.MissingSummaryError:
        return None
    return iter(pairs)


def plans(n: int, labels: Sequence[str] = ()) -> list[sim.TranscriptPlan]:
    ordered = sorted(run.select_plans(run.QWEN_CFG), key=lambda p: p.label)
    if labels:
        return [p for p in ordered if p.label in set(labels)]
    return ordered[:n]


def dry_run(arms: Sequence[str], n: int) -> None:
    for p in plans(n):
        for arm in arms:
            ctxs = boundary_contexts(arm, p)
            if ctxs is None:
                print(f"{p.label} {arm}: SKIPPED, Qwen summaries not cached", flush=True)
                continue
            prev: list[int] = []
            sizes, new = [], []
            for _, ctx in ctxs:
                ids = chat_tokens(ctx)
                new.append(len(ids) - min(reused_prefix(prev, ids), len(ids) - 1))
                sizes.append(len(ids))
                prev = ids
            over = sum(s > local.QWEN_WINDOW - local.MAX_TOKENS for s in sizes)
            print(
                f"{p.label} {arm}: {len(sizes)} boundaries, mean ctx {sum(sizes) / len(sizes):.0f}"
                f" tok, max {max(sizes)}, mean new {sum(new) / len(new):.0f},"
                f" total new {sum(new)}, over window {over}",
                flush=True,
            )


def prefill(net: Any, cache: list[Any], ids: Sequence[int], chunk: int = CHUNK) -> None:
    import mlx.core as mx

    for i in range(0, len(ids), chunk):
        net(mx.array(ids[i : i + chunk])[None], cache=cache)
        mx.eval([c.state for c in cache])


def replay(
    arms: Sequence[str], n: int, labels: Sequence[str] = (), chunk: int = CHUNK
) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm.models import cache as kv

    net, _ = local.load(run.QWEN.model_id)
    # First forward pass compiles Metal kernels; keep that out of the first timed prefill. A full
    # CHUNK, because kernels are specialised per shape and every timed chunk but the last is one.
    scratch = kv.make_prompt_cache(net)
    prefill(net, scratch, [1] * chunk, chunk)
    del scratch
    mx.clear_cache()
    out: dict[str, Any] = {"model": run.QWEN.model_id, "chunk": chunk, "arms": {}}
    for p in plans(n, labels):
        for arm in arms:
            ctxs = boundary_contexts(arm, p)
            if ctxs is None:
                print(f"{p.label} {arm}: SKIPPED, Qwen summaries not cached", flush=True)
                out["arms"].setdefault(arm, {})[p.label] = {"skipped": "summaries not cached"}
                continue
            cache = kv.make_prompt_cache(net)
            prev: list[int] = []
            rows = []
            for t, ctx in ctxs:
                ids = chat_tokens(ctx)
                if len(ids) > local.QWEN_WINDOW - local.MAX_TOKENS:
                    rows.append({"t": t, "tokens": len(ids), "skipped": "over window"})
                    continue
                reused = min(reused_prefix(prev, ids), len(ids) - 1)
                drop = len(prev) - reused
                if drop > 0:
                    if not kv.can_trim_prompt_cache(cache):
                        cache = kv.make_prompt_cache(net)
                        reused = 0
                    else:
                        kv.trim_prompt_cache(cache, drop)
                mx.reset_peak_memory()
                start = time.perf_counter()
                prefill(net, cache, ids[reused:], chunk)
                secs = time.perf_counter() - start
                rows.append(
                    {
                        "t": t,
                        "tokens": len(ids),
                        "new": len(ids) - reused,
                        "secs": secs,
                        "peak_gb": mx.get_peak_memory() / 2**30,
                    }
                )
                prev = ids
                mx.clear_cache()
            timed = [r["secs"] for r in rows if "secs" in r]
            out["arms"].setdefault(arm, {})[p.label] = {
                "rows": rows,
                "mean_secs": sum(timed) / len(timed) if timed else None,
                "total_secs": sum(timed),
            }
            print(
                f"{p.label} {arm}: {len(timed)} timed, total {sum(timed):.1f}s"
                f" mean {sum(timed) / max(len(timed), 1):.2f}s",
                flush=True,
            )
    local.unload()
    return out


def common_turns(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Per arm, the timed rows at boundaries that every arm timed in that transcript. An arm
    that skips its over-window turns would otherwise be averaged over easier turns than one that
    fits them all."""
    arms = result["arms"]
    out: dict[str, list[dict[str, Any]]] = {arm: [] for arm in arms}
    labels = set.intersection(*(set(by) for by in arms.values())) if arms else set()
    for label in sorted(labels):
        timed = {
            arm: {r["t"]: r for r in by[label].get("rows", []) if "secs" in r}
            for arm, by in arms.items()
        }
        shared = set.intersection(*(set(rows) for rows in timed.values()))
        for arm, rows in timed.items():
            out[arm].extend(rows[t] for t in sorted(shared))
    return out


def table(result: dict[str, Any]) -> str:
    lines = [
        "| arm | transcripts | common turns | mean prefill s/turn (common) | mean new tokens/turn"
        " (common) | mean prefill s/turn (all timed) | skipped (over window) |",
        "|---|---|---|---|---|---|---|",
    ]
    shared = common_turns(result)
    for arm, by_label in result["arms"].items():
        done = [v for v in by_label.values() if "rows" in v]
        timed = [r["secs"] for v in done for r in v["rows"] if "secs" in r]
        mean = f"{sum(timed) / len(timed):.2f}" if timed else "n/a"
        rows = shared[arm]
        common = f"{sum(r['secs'] for r in rows) / len(rows):.2f}" if rows else "n/a"
        new = f"{sum(r['new'] for r in rows) / len(rows):.0f}" if rows else "n/a"
        skipped = sum(1 for v in done for r in v["rows"] if "skipped" in r)
        lines.append(
            f"| {arm} | {len(done)}/{len(by_label)} | {len(rows)} | {common} | {new} | {mean}"
            f" | {skipped} |"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--set", dest="set_name", choices=["F"], default="F")
    ap.add_argument("--n", type=int, default=3, help="first N transcripts in label order")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--dry-run", action="store_true", help="contexts and token counts, no model")
    ap.add_argument("--labels", default="", help="only these transcripts (comma-separated)")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="prefill chunk size in tokens")
    ap.add_argument("--out", type=Path, default=OUT, help="result JSON path")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    unknown = set(arms) - set(sim.Q_ARMS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}; choose from {sim.Q_ARMS}")
    if args.dry_run:
        dry_run(arms, args.n)
        return
    labels = [x for x in args.labels.split(",") if x]
    result = replay(arms, args.n, labels, args.chunk)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(table(result), flush=True)


if __name__ == "__main__":
    main()

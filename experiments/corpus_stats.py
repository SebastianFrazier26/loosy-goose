"""Segment-level statistics for every transcript in data/local and data/public.

Usage: uv run python experiments/corpus_stats.py [--swe-gym-sample N]
Writes experiments/output/corpus_stats.txt as well as printing it.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from loosy_goose.segment import Segment, segment
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Transcript, load_claude_code_jsonl, load_messages_json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "experiments" / "output" / "corpus_stats.txt"
KINDS = ("prose", "code", "tool_result", "tool_use", "thinking")


def iter_local() -> Iterator[tuple[str, Transcript]]:
    for p in sorted((DATA / "local" / "sessions").glob("*.jsonl")):
        yield f"local/{p.name}", load_claude_code_jsonl(p)


def iter_trace_commons() -> Iterator[tuple[str, Transcript]]:
    for p in sorted((DATA / "public" / "trace-commons").rglob("*.jsonl")):
        yield f"trace-commons/{p.name}", load_claude_code_jsonl(p)


def iter_swe_gym(sample: int) -> Iterator[tuple[str, Transcript]]:
    files = sorted((DATA / "public" / "swe-gym").rglob("*.parquet"))
    if not files:
        return
    import pyarrow.parquet as pq

    table = pq.read_table(files[0])
    col = "messages" if "messages" in table.column_names else table.column_names[0]
    msgs = table.column(col).to_pylist()
    # Spread the sample across the length distribution instead of taking the head.
    order = sorted(range(len(msgs)), key=lambda i: len(json.dumps(msgs[i], default=str)))
    step = max(1, len(order) // sample)
    for i in order[::step][:sample]:
        payload = msgs[i]
        if isinstance(payload, str):
            payload = json.loads(payload)
        yield f"swe-gym/row{i}", load_messages_json(payload, name=f"swe-gym-{i}")


def summarise(label: str, t: Transcript) -> str:
    segs: list[Segment] = segment(t)
    tok_by_kind: Counter[str] = Counter()
    for s in segs:
        tok_by_kind[s.kind] += count_tokens(s.text)
    total = sum(tok_by_kind.values())
    protected_tokens = sum(count_tokens(s.text) for s in segs if s.protected)
    atoms = sum(len(s.atoms) for s in segs)
    skipped = sum(t.meta.values())
    cells = [
        f"{label:<48}",
        f"{len(t.turns):>5}",
        f"{len(segs):>6}",
        f"{total:>8}",
    ]
    cells += [f"{tok_by_kind[k]:>8}" for k in KINDS]
    cells += [
        f"{(protected_tokens / total if total else 0):>6.2f}",
        f"{atoms:>6}",
        f"{skipped:>6}",
    ]
    return " ".join(cells)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swe-gym-sample", type=int, default=10)
    args = parser.parse_args()

    header = " ".join(
        [f"{'transcript':<48}", f"{'turns':>5}", f"{'segs':>6}", f"{'tokens':>8}"]
        + [f"{k[:8]:>8}" for k in KINDS]
        + [f"{'prot':>6}", f"{'atoms':>6}", f"{'skip':>6}"]
    )
    lines = [header, "-" * len(header)]
    sources = [iter_local(), iter_trace_commons(), iter_swe_gym(args.swe_gym_sample)]
    for src in sources:
        for label, t in src:
            lines.append(summarise(label, t))
    text = "\n".join(lines)
    print(text)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

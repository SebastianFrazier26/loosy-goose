"""Pick local Claude Code sessions closest to target token sizes into data/local/sessions/.

Usage: uv run python experiments/pick_local.py [--projects DIR] [--exclude SESSION_ID ...]
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Transcript, load_claude_code_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "local" / "sessions"
TARGETS = (5_000, 20_000, 50_000, 100_000, 200_000)
DEFAULT_PROJECTS = Path.home() / ".claude" / "projects"


def tokens_by_kind(t: Transcript) -> Counter[str]:
    c: Counter[str] = Counter()
    for turn in t.turns:
        for b in turn.blocks:
            c[b.kind] += count_tokens(b.text)
    return c


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", type=Path, default=DEFAULT_PROJECTS)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--min-bytes", type=int, default=100_000)
    args = parser.parse_args()

    rows: list[tuple[Path, Transcript, Counter[str]]] = []
    for path in sorted(args.projects.glob("*/*.jsonl")):
        if path.stat().st_size < args.min_bytes or path.stem in args.exclude:
            continue
        try:
            t = load_claude_code_jsonl(path)
        except OSError as exc:
            print(f"skip {path.name}: {exc}")
            continue
        if not t.turns:
            continue
        rows.append((path, t, tokens_by_kind(t)))

    def total(c: Counter[str]) -> int:
        return c["text"] + c["code"] + c["tool_result"]

    DEST.mkdir(parents=True, exist_ok=True)
    chosen: set[Path] = set()
    print(
        f"{'target':>8} {'file':<44} {'turns':>6} {'text':>8} {'code':>8} {'tool':>9} {'total':>9}"
    )
    for target in TARGETS:
        best = min(
            (r for r in rows if r[0] not in chosen),
            key=lambda r: abs(total(r[2]) - target),
            default=None,
        )
        if best is None:
            break
        path, t, c = best
        chosen.add(path)
        out = DEST / f"{target // 1000}k_{path.stem}.jsonl"
        shutil.copyfile(path, out)
        print(
            f"{target:>8} {out.name:<44} {len(t.turns):>6} {c['text']:>8} {c['code']:>8} "
            f"{c['tool_result']:>9} {total(c):>9}"
        )
    print(f"\nscanned {len(rows)} sessions; copied {len(chosen)} to {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

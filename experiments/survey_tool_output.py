"""What is actually inside tool_result blocks, measured rather than assumed.

`tool_result` is the largest channel by tokens in agentic transcripts and the only large one
with no shrink mechanism at all. Before building format-aware shrinkers, this counts what shapes
are really there and how many tokens each accounts for, so the recognizers get written for what
the corpus holds instead of what seems likely.

Deliberately surveys BLOCKS, not segments. `segment.py` chunks a tool_result every 600
characters and reclassifies code-looking results as `code`, both of which destroy the structure
a format recognizer needs — a diff split across five chunks is not recognizable as a diff. What
a shrinker would have to see is the whole block, which is what this measures.

Usage: uv run python experiments/survey_tool_output.py [--public-sample 8] [--swe-gym-sample 5]
Writes experiments/output/survey/tool_output_shapes.txt. Sample lines are printed for PUBLIC
transcripts only; local sessions contribute counts and never content.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.exp_d_curves import _iter_sources  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "output" / "survey"

# Ordered: the first matching rule wins, so the specific shapes are tested before the generic
# ones. Each rule is (name, predicate over the block's lines and text).
_NUMBERED = re.compile(r"^\s*\d+(→|\t)")
_DIFF_HUNK = re.compile(r"^@@ .* @@|^diff --git |^(---|\+\+\+) ")
_DIFF_ADD = re.compile(r"^\+[^+]")
_DIFF_DEL = re.compile(r"^-[^-]")
_GREP_HIT = re.compile(r"^[\w./\\-]+:\d+:")
_LISTING = re.compile(r"^[\w./\\-]+$")
_TEST_SUMMARY = re.compile(
    r"\b\d+ (passed|failed|error|skipped)\b|={3,}.*(FAILURES|short test summary|ERRORS)"
    r"|^(OK|FAILED)\b|\bTests? run:",
    re.IGNORECASE,
)
_TRACEBACK = re.compile(r"^Traceback \(most recent call last\)|^\s+File \".*\", line \d+")
_ERROR_MARK = re.compile(r"\b(error|exception|fatal|cannot|not found|no such file)\b", re.I)


def _nonblank(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.strip()]


def _fraction(lines: list[str], pattern: re.Pattern[str]) -> float:
    live = _nonblank(lines)
    if not live:
        return 0.0
    return sum(1 for ln in live if pattern.search(ln)) / len(live)


def classify(text: str) -> str:
    stripped = text.strip()
    lines = text.split("\n")
    live = _nonblank(lines)
    if not stripped:
        return "empty"
    if len(stripped) < 200 and len(live) <= 2:
        return "short"
    if _fraction(lines, _NUMBERED) >= 0.6:
        return "numbered_file"
    # A hunk header is decisive. Without one, require BOTH added and removed lines: the earlier
    # "30% of lines start with + or -" rule matched PowerShell directory listings, whose every
    # entry begins with a dash, and reported them as diffs.
    if _DIFF_HUNK.search(text) or (
        _fraction(lines, _DIFF_ADD) >= 0.15 and _fraction(lines, _DIFF_DEL) >= 0.15
    ):
        return "diff"
    if _TRACEBACK.search(text):
        return "traceback"
    if _TEST_SUMMARY.search(text):
        return "test_output"
    if _fraction(lines, _GREP_HIT) >= 0.5:
        return "grep_hits"
    if (stripped.startswith(("{", "[")) and stripped.endswith(("}", "]"))) or _fraction(
        lines, re.compile(r"^\s*[\"{}\[\],]")
    ) >= 0.7:
        return "json"
    if _fraction(lines, _LISTING) >= 0.7:
        return "path_listing"
    if _ERROR_MARK.search(stripped[:400]):
        return "error_text"
    return "other"


def _repetition(lines: list[str]) -> float:
    """Fraction of non-blank lines that repeat an earlier line verbatim. High values mark output
    a line-level shrinker could collapse without any format knowledge at all."""
    live = _nonblank(lines)
    if len(live) < 2:
        return 0.0
    counts = Counter(live)
    return sum(c - 1 for c in counts.values()) / len(live)


def survey(public_sample: int, swe_gym_sample: int) -> dict[str, Any]:
    by_shape: dict[str, dict[str, float]] = defaultdict(
        lambda: {"blocks": 0.0, "tokens": 0.0, "lines": 0.0, "repetition": 0.0}
    )
    sizes: list[int] = []
    samples: dict[str, str] = {}
    total_blocks = 0
    total_tokens = 0
    for label, public, t in _iter_sources(public_sample, swe_gym_sample):
        for turn in t.turns:
            for block in turn.blocks:
                if block.kind != "tool_result":
                    continue
                shape = classify(block.text)
                tokens = count_tokens(block.text)
                lines = block.text.split("\n")
                row = by_shape[shape]
                row["blocks"] += 1
                row["tokens"] += tokens
                row["lines"] += len(lines)
                row["repetition"] += _repetition(lines)
                sizes.append(tokens)
                total_blocks += 1
                total_tokens += tokens
                # Content leaves the process only for public corpora; local sessions contribute
                # counts alone, matching the withholding rule in PHASE1.md.
                if public and shape not in samples and tokens > 40:
                    first = next((ln for ln in lines if ln.strip()), "")
                    samples[shape] = f"[{label}] {first[:110]}"
    return {
        "by_shape": {k: dict(v) for k, v in by_shape.items()},
        "sizes": sizes,
        "samples": samples,
        "total_blocks": total_blocks,
        "total_tokens": total_tokens,
    }


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[idx]


def report(data: dict[str, Any]) -> list[str]:
    total_blocks = int(data["total_blocks"])
    total_tokens = int(data["total_tokens"])
    rows = sorted(data["by_shape"].items(), key=lambda kv: -kv[1]["tokens"])
    header = (
        f"{'shape':<16}{'blocks':>9}{'% blocks':>10}{'tokens':>12}{'% tokens':>10}"
        f"{'med lines':>11}{'repeat':>9}"
    )
    out = [
        f"tool_result blocks surveyed: {total_blocks}, {total_tokens} tokens.",
        "",
        "Shapes are assigned by first matching rule, most specific first. 'repeat' is the mean",
        "fraction of non-blank lines that duplicate an earlier line in the same block — high",
        "values mark output a line-level shrinker could collapse with no format knowledge.",
        "",
        header,
        "-" * len(header),
    ]
    for shape, row in rows:
        blocks = row["blocks"]
        out.append(
            f"{shape:<16}{int(blocks):>9}{blocks / max(total_blocks, 1):>9.1%}"
            f"{int(row['tokens']):>12}{row['tokens'] / max(total_tokens, 1):>9.1%}"
            f"{row['lines'] / max(blocks, 1):>11.0f}{row['repetition'] / max(blocks, 1):>8.1%}"
        )

    sizes = list(data["sizes"])
    out += [
        "",
        "== block size distribution (tokens) ==",
        f"p50 {_percentile(sizes, 0.50)}   p75 {_percentile(sizes, 0.75)}   "
        f"p90 {_percentile(sizes, 0.90)}   p99 {_percentile(sizes, 0.99)}   "
        f"max {max(sizes) if sizes else 0}",
    ]
    big = [s for s in sizes if s >= 500]
    out.append(
        f"blocks >= 500 tokens: {len(big)} ({len(big) / max(len(sizes), 1):.1%} of blocks, "
        f"{sum(big) / max(total_tokens, 1):.1%} of tokens)"
    )

    out += ["", "== first line of one public example per shape =="]
    for shape, line in sorted(data["samples"].items()):
        out.append(f"{shape:<16}{line}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-sample", type=int, default=8)
    parser.add_argument("--swe-gym-sample", type=int, default=5)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    OUT.mkdir(parents=True, exist_ok=True)
    data = survey(args.public_sample, args.swe_gym_sample)
    text = "\n".join(report(data))
    (OUT / "tool_output_shapes.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

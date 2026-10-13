import argparse
import io
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from loosy_goose.transcript import (
    Transcript,
    load_claude_code_jsonl,
    load_messages_json,
    write_blocks_jsonl,
)

KEEP_MIN = 0.1
KEEP_MAX = 0.8
LEVELS: dict[str, float] = {"light": 0.50, "medium": 0.33, "heavy": 0.25}
DEFAULT_LEVEL = "medium"

RateSource = Literal["keep", "level", "target-tokens"]


@dataclass(frozen=True)
class Rate:
    """The compression rate as the pipeline consumes it. `budget` is set only when a token target
    was honoured exactly; otherwise the pipeline derives the budget from `keep`."""

    keep: float
    budget: int | None
    source: RateSource
    level: str | None
    target_tokens: int | None
    notes: tuple[str, ...]


def _keep(value: str) -> float:
    k = float(value)
    if not KEEP_MIN <= k <= KEEP_MAX:
        raise argparse.ArgumentTypeError(f"keep must be in [{KEEP_MIN}, {KEEP_MAX}], got {k}")
    return k


def _target_tokens(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"target-tokens must be a positive integer, got {n}")
    return n


def resolve_rate(
    keep: float | None, level: str | None, target_tokens: int | None, input_tokens: int
) -> Rate:
    """`input_tokens` must be the count `token_budget` sums (original segments), or a target's
    implied ratio would not match the budget the pipeline would derive from it."""
    if keep is not None:
        rounded = round(keep, 2)
        notes: tuple[str, ...] = ()
        if rounded != keep:
            notes = (f"--keep {keep} rounded to 0.01 steps; using {rounded:.2f}",)
        return Rate(rounded, None, "keep", None, None, notes)
    if target_tokens is not None:
        implied = target_tokens / input_tokens if input_tokens else math.inf
        if KEEP_MIN <= implied <= KEEP_MAX:
            return Rate(implied, target_tokens, "target-tokens", None, target_tokens, ())
        clamped = KEEP_MIN if implied < KEEP_MIN else KEEP_MAX
        warning = (
            f"warning: --target-tokens {target_tokens} implies keep {implied:.4f} of "
            f"{input_tokens} input tokens, outside [{KEEP_MIN}, {KEEP_MAX}]; clamped to "
            f"keep {clamped:.2f}"
        )
        return Rate(clamped, None, "target-tokens", None, target_tokens, (warning,))
    name = level if level is not None else DEFAULT_LEVEL
    return Rate(LEVELS[name], None, "level", name, None, ())


def _min_mentions(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"min-mentions must be at least 1, got {n}")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loosy-goose",
        description="Lossy compression of LLM conversation context.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    compress = sub.add_parser("compress", help="compress a conversation transcript")
    compress.add_argument("transcript", help="path to the transcript to compress")
    rate = compress.add_mutually_exclusive_group()
    rate.add_argument(
        "--keep",
        type=_keep,
        help=f"fraction of input tokens to keep, in [{KEEP_MIN}, {KEEP_MAX}], 0.01 steps "
        "(finer values are rounded)",
    )
    rate.add_argument(
        "--level",
        choices=tuple(LEVELS),
        help="preset rate: light = keep 0.50 (~2x), medium = keep 0.33 (~3x), "
        "heavy = keep 0.25 (~4x) (default: medium)",
    )
    rate.add_argument(
        "--target-tokens",
        type=_target_tokens,
        help=f"output token budget; one implying keep outside [{KEEP_MIN}, {KEEP_MAX}] is "
        "clamped to that range, with a warning",
    )
    compress.add_argument(
        "--no-novelty",
        dest="novelty",
        action="store_false",
        help="select by plain leverage fill (supersede+leverage) instead of the default leverage "
        "x word novelty ranked per token (supersede+leverage+novelty-per-token@0.5)",
    )
    compress.add_argument(
        "--band",
        action="store_true",
        help="AST-depth banding of code and tool_use before scoring; selection falls back to "
        "the plain leverage fill (supersede+band+leverage)",
    )
    compress.add_argument(
        "--paths",
        action="store_true",
        help="path substitution: name each repeated path once in a leading table, replace "
        "mentions with markers; selection falls back to the plain leverage fill",
    )
    compress.add_argument(
        "--min-mentions",
        type=_min_mentions,
        default=2,
        help="paths: tabulate paths mentioned at least this often (1 = every path)",
    )
    compress.add_argument("--output", default="-", help="output path, or - for stdout (default)")
    return parser


def _load(path: Path) -> Transcript | None:
    if path.suffix == ".jsonl":
        return load_claude_code_jsonl(path)
    if path.suffix == ".json":
        return load_messages_json(path)
    return None


def _compress(args: argparse.Namespace) -> int:
    # Imported here so `--help` and argument errors never pay for torch/sentence-transformers.
    from loosy_goose.code import TRANSFORMS_VERSION
    from loosy_goose.paths import PATHS_VERSION
    from loosy_goose.pipeline import (
        PipelineOptions,
        compress_transcript,
        selector_label,
        to_turns,
    )
    from loosy_goose.segment import ATOMS_VERSION, segment
    from loosy_goose.tokens import count_tokens

    transcript = _load(Path(args.transcript))
    if transcript is None:
        print(
            f"loosy-goose compress: unsupported transcript suffix {Path(args.transcript).suffix!r}"
            " (expected .jsonl or .json)",
            file=sys.stderr,
        )
        return 2
    input_tokens = (
        sum(count_tokens(s.text) for s in segment(transcript))
        if args.target_tokens is not None
        else 0
    )
    rate = resolve_rate(args.keep, args.level, args.target_tokens, input_tokens)
    for note in rate.notes:
        print(f"loosy-goose compress: {note}", file=sys.stderr)
    opts = PipelineOptions(
        keep=rate.keep,
        budget=rate.budget,
        band=args.band,
        paths=args.paths,
        min_mentions=args.min_mentions,
        novelty=args.novelty,
    )
    result = compress_transcript(transcript, opts)
    header: dict[str, Any] = {
        "keep": opts.keep,
        "keep_source": rate.source,
        "level": rate.level,
        "target_tokens": rate.target_tokens,
        "band": opts.band,
        "paths": opts.paths,
        "min_mentions": opts.min_mentions,
        "strategy": opts.strategy,
        "novelty": opts.novelty,
        "selector": selector_label(opts),
        "budget": result.budget,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "atom_recall_final": result.atom_recall_final,
        "atoms_version": ATOMS_VERSION,
        "transforms_version": TRANSFORMS_VERSION,
        "paths_version": PATHS_VERSION,
        "table": result.table.render() if result.table is not None else None,
    }
    turns = to_turns(result, transcript)
    if args.output == "-":
        # A console stdout follows the locale code page (cp1252 on Windows) and would choke on
        # the transcript's Unicode; the `--output` branch already opens its file as UTF-8.
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        write_blocks_jsonl(turns, header, sys.stdout)
    else:
        with Path(args.output).open("w", encoding="utf-8", newline="\n") as fh:
            write_blocks_jsonl(turns, header, fh)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compress":
        return _compress(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())

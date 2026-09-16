import argparse
import io
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loosy_goose.transcript import (
    Transcript,
    load_claude_code_jsonl,
    load_messages_json,
    write_blocks_jsonl,
)

KEEP_MIN = 0.1
KEEP_MAX = 0.8


def _quality(value: str) -> float:
    q = float(value)
    if not 0.0 <= q <= 1.0:
        raise argparse.ArgumentTypeError(f"quality must be in [0, 1], got {q}")
    return q


def _keep(value: str) -> float:
    k = float(value)
    if not KEEP_MIN <= k <= KEEP_MAX:
        raise argparse.ArgumentTypeError(f"keep must be in [{KEEP_MIN}, {KEEP_MAX}], got {k}")
    return k


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
    compress.add_argument(
        "--quality",
        type=_quality,
        default=0.6,
        help="retained-meaning target in [0, 1]; accepted, not yet wired to the pipeline "
        "(default: 0.6)",
    )
    compress.add_argument(
        "--keep",
        type=_keep,
        required=True,
        help=f"fraction of input tokens to keep, in [{KEEP_MIN}, {KEEP_MAX}]",
    )
    compress.add_argument(
        "--band",
        action="store_true",
        help="AST-depth banding of code and tool_use before scoring",
    )
    compress.add_argument(
        "--paths",
        action="store_true",
        help="path substitution: name each repeated path once in a leading table, replace "
        "mentions with markers",
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
    from loosy_goose.pipeline import PipelineOptions, compress_transcript, to_turns
    from loosy_goose.segment import ATOMS_VERSION

    transcript = _load(Path(args.transcript))
    if transcript is None:
        print(
            f"loosy-goose compress: unsupported transcript suffix {Path(args.transcript).suffix!r}"
            " (expected .jsonl or .json)",
            file=sys.stderr,
        )
        return 2
    opts = PipelineOptions(
        keep=args.keep, band=args.band, paths=args.paths, min_mentions=args.min_mentions
    )
    result = compress_transcript(transcript, opts)
    header: dict[str, Any] = {
        "keep": opts.keep,
        "band": opts.band,
        "paths": opts.paths,
        "min_mentions": opts.min_mentions,
        "strategy": opts.strategy,
        "budget": result.budget,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
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

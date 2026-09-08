import argparse
import sys
from collections.abc import Sequence


def _quality(value: str) -> float:
    q = float(value)
    if not 0.0 <= q <= 1.0:
        raise argparse.ArgumentTypeError(f"quality must be in [0, 1], got {q}")
    return q


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
        help="retained-meaning target in [0, 1]; 1 = lossless (default: 0.6)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compress":
        print(
            f"loosy-goose compress: not implemented yet "
            f"(transcript={args.transcript}, quality={args.quality})",
            file=sys.stderr,
        )
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())

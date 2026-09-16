from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loosy_goose.budget import ProtectPolicy, select_with_budget, token_budget
from loosy_goose.code import quantize_segment
from loosy_goose.paths import PathTable, build_table, substitute
from loosy_goose.segment import Segment, SegmentKind, segment
from loosy_goose.select import CompressConfig, Strategy, embed_segments, score_segments
from loosy_goose.supersede import apply_supersession
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Block, BlockKind, Transcript, Turn

# Mirrors the Experiment D runner's `supersede+band+leverage` stack (`experiments/exp_d_curves.py`),
# which is the measured configuration; every step here has a twin there and should change with it.
_BAND_KINDS: tuple[SegmentKind, ...] = ("code", "tool_use")
_BLOCK_KIND: dict[SegmentKind, BlockKind] = {
    "prose": "text",
    "code": "code",
    "tool_use": "tool_use",
    "tool_result": "tool_result",
    "thinking": "thinking",
}


@dataclass(frozen=True)
class PipelineOptions:
    keep: float
    band: bool = False
    paths: bool = False
    min_mentions: int = 2
    strategy: Strategy = "leverage"
    protect: ProtectPolicy = "none"


@dataclass(frozen=True)
class Compressed:
    segments: list[Segment]
    table: PathTable | None
    budget: int
    input_tokens: int
    output_tokens: int


def compress_transcript(
    transcript: Transcript, opts: PipelineOptions, *, cache_dir: Path | None = None
) -> Compressed:
    original = segment(transcript)
    # Budget from the original segments, before anything shrinks them: the runner's shared-budget
    # contract. Deriving it after supersession or banding would hand the stack a smaller budget
    # than the keep ratio names.
    budget = token_budget(original, opts.keep)
    segs = apply_supersession(original)
    if opts.band:
        # quality == keep ratio, as the runner's `_band` does; only code and tool_use are banded.
        segs = [quantize_segment(s, opts.keep) if s.kind in _BAND_KINDS else s for s in segs]
    table: PathTable | None = None
    scored = segs
    if opts.paths:
        # Table from the original segments, substitution on the transformed ones: that is the
        # configuration the arm-D numbers were measured under (the runner builds its tables
        # before supersession).
        table = build_table(original, min_mentions=opts.min_mentions)
        scored = substitute(segs, table)
    config = CompressConfig(strategy=opts.strategy)
    x = embed_segments(scored, config.model_name, cache_dir)
    scores = score_segments(x, config)
    kept = select_with_budget(scored, scores, budget, opts.protect)
    table_tokens = table.tokens() if table is not None else 0
    return Compressed(
        segments=kept,
        table=table,
        budget=budget,
        input_tokens=sum(count_tokens(s.text) for s in original),
        output_tokens=sum(count_tokens(s.text) for s in kept) + table_tokens,
    )


def to_turns(compressed: Compressed, transcript: Transcript) -> list[Turn]:
    by_index = {t.index: t for t in transcript.turns}
    grouped: dict[int, list[Segment]] = {}
    for s in sorted(compressed.segments, key=lambda s: s.id):
        grouped.setdefault(s.turn, []).append(s)
    return [
        Turn(
            index,
            by_index[index].role,
            [Block(_BLOCK_KIND[s.kind], s.text, {}) for s in segs],
        )
        for index, segs in sorted(grouped.items())
    ]

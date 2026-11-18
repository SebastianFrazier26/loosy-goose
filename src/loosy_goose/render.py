"""Kept segments as prompt text.

Ported from `experiments/exp_i_downstream.py` with identical output; the experiment keeps its own
copy so its prompt hashes never move with this one.
"""

from __future__ import annotations

from collections.abc import Sequence

from .segment import Segment

SEGMENT_BLOCK_KIND = {"prose": "text"}


def render_block(kind: str, text: str, name: str | None) -> str:
    if kind == "tool_use":
        return f'<tool_use name="{name or "?"}">\n{text}\n</tool_use>'
    if kind == "tool_result":
        return f"<tool_result>\n{text}\n</tool_result>"
    if kind == "thinking":
        return f"<thinking>\n{text}\n</thinking>"
    if kind == "code":
        return f"```\n{text}\n```"
    return text


def render_segments(segments: Sequence[Segment]) -> str:
    """Kept segments regrouped by turn and block. A gap between kept segments of the same block is
    marked, so a partly dropped tool result does not read as complete."""
    parts: list[str] = []
    ordered = sorted(segments, key=lambda s: s.id)
    i = 0
    while i < len(ordered):
        turn = ordered[i].turn
        blocks: list[str] = []
        while i < len(ordered) and ordered[i].turn == turn:
            first = ordered[i]
            pieces = [first.text]
            last_id = first.id
            i += 1
            while i < len(ordered) and (ordered[i].turn, ordered[i].block) == (turn, first.block):
                if ordered[i].id != last_id + 1:
                    pieces.append("[…]")
                pieces.append(ordered[i].text)
                last_id = ordered[i].id
                i += 1
            sep = "\n" if first.kind == "tool_result" else "\n\n"
            kind = SEGMENT_BLOCK_KIND.get(first.kind, first.kind)
            blocks.append(render_block(kind, sep.join(pieces), first.tool_name))
        parts.append(f"### {first.role}\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)

"""In-memory `Session` objects for the pure planning tests: one record per spec, in order."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loosy_goose.segment import Segment, segment
from loosy_goose.sessions.model import Session, SessionRecord, Tool, tagged, text_blocks
from loosy_goose.sessions.plan import Region


def words(n: int, word: str = "alpha") -> str:
    return " ".join([word] * n)


def build(
    *specs: tuple[str, str],
    tool: Tool = "claude",
    open_exchange: int | None = None,
    loose: tuple[str, ...] = (),
) -> Session:
    """Shapes: prompt, reply[:reply-id], think[:reply-id], call:<call-id>, handoff:<call-id>,
    result:<call-id>, summary, meta, side, unknown."""
    records: list[SessionRecord] = []
    exchange = 0
    for n, (shape, text) in enumerate(specs):
        head, _, arg = shape.partition(":")
        if head == "prompt":
            exchange += 1
        rec = _record(n, head, arg, text)
        rec.exchange = exchange
        records.append(rec)
    extra = [
        SessionRecord(len(records) + i, {"type": kind}, kind, "unknown")
        for i, kind in enumerate(loose)
    ]
    return Session(
        tool,
        Path("built.jsonl"),
        "source-id",
        "/work",
        None,
        records + extra,
        list(range(len(records))),
        [r.line for r in extra],
        open_exchange,
        [],
    )


def _record(n: int, head: str, arg: str, text: str) -> SessionRecord:
    raw: dict[str, Any] = {"n": n}
    if head == "prompt":
        return SessionRecord(
            n, raw, "user", "message", role="user", blocks=text_blocks(text, 0), prompt=text
        )
    if head == "reply":
        return SessionRecord(
            n,
            raw,
            "assistant",
            "message",
            role="assistant",
            blocks=text_blocks(text, 0),
            reply=arg or None,
        )
    if head == "think":
        return SessionRecord(
            n,
            raw,
            "assistant",
            "message",
            role="assistant",
            blocks=tagged("thinking", text, 0),
            thinking=True,
            reply=arg or None,
        )
    if head in ("call", "handoff"):
        name = "mcp__loosy-goose__handoff_session" if head == "handoff" else "Bash"
        return SessionRecord(
            n,
            raw,
            "assistant",
            "message",
            role="assistant",
            blocks=tagged("tool_use", text, 0, name=name),
            call_ids=(arg,),
            call_side="call",
            tool_name=name,
        )
    if head == "result":
        return SessionRecord(
            n,
            raw,
            "user:tool_result",
            "message",
            role="user",
            blocks=tagged("tool_result", text, 0),
            call_ids=(arg,),
            call_side="result",
        )
    if head == "summary":
        return SessionRecord(
            n,
            raw,
            "user:compact_summary",
            "message",
            role="user",
            blocks=text_blocks(text, 0),
            pinned=True,
        )
    if head == "meta":
        return SessionRecord(n, raw, "attachment", "meta")
    if head == "side":
        return SessionRecord(n, raw, "queue-operation", "side")
    if head == "unknown":
        return SessionRecord(n, raw, "mystery", "unknown")
    raise ValueError(f"unknown shape {head!r}")


def kept_where(region: Region, keep: Callable[[Segment], bool]) -> list[Segment]:
    """The segments a compressor run would return if it kept exactly those matching `keep`."""
    return [s for s in segment(region.transcript) if keep(s)]

"""Pure planning over a `Session`: what the handoff excludes, protects, compresses and keeps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from loosy_goose.sessions.model import Session, Tool
from loosy_goose.transcript import Transcript, Turn

HANDOFF_SENTINEL = "loosy-goose:handoff"
HANDOFF_TOOL = "handoff_session"
PROTECT_TAIL_TOKENS = 4000
DEFAULT_TRIGGERS: dict[Tool, tuple[str, ...]] = {
    # Claude Code records a custom slash command as a user record carrying this tag; the expanded
    # body (where the sentinel sits) is a separate isMeta record, which is not a prompt.
    "claude": ("<command-name>/loosy</command-name>", HANDOFF_SENTINEL),
    "codex": ("$loosy", HANDOFF_SENTINEL),
}

TriggerHow = Literal["text", "open-turn", "tool-call", "none"]


@dataclass(frozen=True)
class Trigger:
    """`position` indexes `Session.live`: the first record excluded from the fork."""

    position: int | None
    how: TriggerHow


@dataclass(frozen=True)
class Region:
    """`live[:end]` is history; `live[tail_from:end]` is the verbatim tail; `transcript` holds
    the compressible message records before it, `turn_record[i]` naming turn i's record."""

    end: int
    tail_from: int
    transcript: Transcript
    turn_record: tuple[int, ...]


def exchange_starts(session: Session) -> dict[int, int]:
    starts: dict[int, int] = {}
    for pos, ri in enumerate(session.live):
        starts.setdefault(session.records[ri].exchange, pos)
    return starts


def find_trigger(session: Session, triggers: Sequence[str]) -> Trigger:
    records, live = session.records, session.live
    starts = exchange_starts(session)
    found: list[tuple[int, TriggerHow]] = []
    text = next(
        (
            ri
            for ri in reversed(live)
            if (p := records[ri].prompt) is not None and any(t in p for t in triggers)
        ),
        None,
    )
    if text is not None:
        found.append((starts[records[text].exchange], "text"))
    if session.open_exchange is not None and session.open_exchange in starts:
        found.append((starts[session.open_exchange], "open-turn"))
    call = next(
        (
            ri
            for ri in reversed(live)
            if records[ri].call_side == "call"
            and (records[ri].tool_name or "").endswith(HANDOFF_TOOL)
        ),
        None,
    )
    if call is not None:
        found.append((starts[records[call].exchange], "tool-call"))
    if not found:
        return Trigger(None, "none")
    # The current invocation is always the latest of these; an older /loosy text must not win
    # over it, or everything since that older handoff would be cut. max() keeps the first of
    # equal positions, so text beats the weaker signals on a tie.
    position, how = max(found, key=lambda f: f[0])
    return Trigger(position, how)


def unit_ids(session: Session) -> dict[int, int]:
    parent: dict[int, int] = {}

    def root(x: int) -> int:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    owner: dict[str, int] = {}
    for ri in session.live:
        for cid in session.records[ri].call_ids:
            if cid in owner:
                a, b = root(owner[cid]), root(ri)
                if a != b:
                    parent[b] = a
            else:
                owner[cid] = ri
    return {ri: root(ri) for ri in session.live}


def split_region(
    session: Session, trigger: Trigger, protect_tail_tokens: int = PROTECT_TAIL_TOKENS
) -> Region:
    records, live = session.records, session.live
    end = trigger.position if trigger.position is not None else len(live)
    starts = exchange_starts(session)
    tail_from = 0
    for pos in range(end - 1, -1, -1):
        rec = records[live[pos]]
        if rec.prompt is not None:
            tail_from = starts[rec.exchange]
            break
    tail = sum(session.tokens(ri) for ri in live[tail_from:end])
    # Extended record by record, not exchange by exchange: one agentic exchange can run to six
    # figures of tokens, and taking it whole would leave nothing to compress.
    while tail < protect_tail_tokens and tail_from > 0:
        tail_from -= 1
        tail += session.tokens(live[tail_from])
    units = unit_ids(session)
    position = {ri: pos for pos, ri in enumerate(live)}
    first_of: dict[int, int] = {}
    for ri, unit in units.items():
        first_of[unit] = min(first_of.get(unit, position[ri]), position[ri])
    moved = True
    while moved:
        moved = False
        for pos in range(tail_from, end):
            first = first_of[units[live[pos]]]
            if first < tail_from:
                tail_from, moved = first, True
                break
    turns: list[Turn] = []
    turn_record: list[int] = []
    for ri in live[:tail_from]:
        rec = records[ri]
        if rec.category != "message" or rec.thinking or rec.pinned or not rec.blocks:
            continue
        turns.append(Turn(len(turns), rec.role or "user", rec.blocks))
        turn_record.append(ri)
    return Region(
        end, tail_from, Transcript(f"{session.path}#compressible", turns), tuple(turn_record)
    )

"""Pure planning over a `Session`: what the handoff excludes, protects, compresses and keeps."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from loosy_goose.segment import Segment, segment
from loosy_goose.sessions.model import PART, Session, SessionRecord, Tool
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Block, Transcript, Turn

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


TRIMMED_RESULT = "[loosy-goose: output trimmed]"
GAP = "[loosy-goose: lines trimmed]"
PLACEHOLDERS = (GAP, TRIMMED_RESULT)
_FENCE = "`" * 3
MARKER_PREFIX = "[loosy-goose: compressed from session "
_MARKER = MARKER_PREFIX + (
    "{source} on {date}; kept {kept} of {total} tokens before the last {tail} tokens, which "
    "are verbatim. Earlier turns are an extractive selection — some were dropped.]"
)
_MARKER_FIELDS = re.compile(
    re.escape(MARKER_PREFIX)
    + r"(?P<source>\S+) on (?P<date>[^;]+); kept (?P<kept>\d+) of \d+ tokens before the last "
    r"(?P<tail>\d+) tokens"
)
# Tool input must stay valid structured data and thinking carries signatures; neither is ever
# rewritten, so neither is stripped.
_NEVER_STRIPPED = frozenset({"tool_use", "thinking"})


class HandoffRefusedError(Exception):
    pass


def strip_placeholders(text: str) -> str:
    if not any(p in text for p in PLACEHOLDERS):
        return text
    out: list[str] = []
    for line in text.split("\n"):
        cut = line
        for p in PLACEHOLDERS:
            cut = cut.replace(p, "")
        if cut.strip() or not line.strip():
            out.append(cut)
    return "\n".join(out)


def strip_part(text: str, part: int, cleaned: dict[int, str]) -> str:
    out = strip_placeholders(text)
    if out != text:
        cleaned[part] = out
    return out


@dataclass(frozen=True)
class PriorFork:
    """An earlier loosy-goose fork whose marker is in the history being compressed again."""

    source: str
    date: str
    added_tokens: int


def prior_fork(marker: str, tokens_after: int) -> PriorFork | None:
    found = _MARKER_FIELDS.match(marker)
    if found is None:
        return None
    # The marker states what that fork carried after it: `kept` selected tokens and a verbatim
    # `tail`. The rest of what follows it now is counted as added since. Records the fork kept
    # outside both figures (a compaction summary, a command's expanded body) count as added too,
    # so this is an estimate, floored at zero.
    carried = int(found["kept"]) + int(found["tail"])
    return PriorFork(found["source"], found["date"], max(0, tokens_after - carried))


def find_prior_fork(session: Session, end: int) -> PriorFork | None:
    live = session.live[:end]
    for pos in range(len(live) - 1, -1, -1):
        marker = session.records[live[pos]].marker
        if marker is not None:
            return prior_fork(marker, sum(session.tokens(ri) for ri in live[pos + 1 :]))
    return None


def recompress_warning(prior: PriorFork) -> str:
    return (
        f"this session was already compressed by loosy-goose on {prior.date}; "
        f"about {prior.added_tokens:,} tokens added since — compressing again compounds the loss"
    )


def clean_transcript(transcript: Transcript) -> tuple[Transcript, PriorFork | None]:
    """`transcript` without loosy-goose's own artifacts (marker blocks dropped, placeholders
    stripped, emptied turns removed, the rest renumbered), plus the earlier fork the latest
    marker names."""
    turns: list[Turn] = []
    marker: str | None = None
    after = 0
    for turn in transcript.turns:
        blocks: list[Block] = []
        for block in turn.blocks:
            if block.text.startswith(MARKER_PREFIX):
                marker, after = block.text, 0
                continue
            text = block.text if block.kind in _NEVER_STRIPPED else strip_placeholders(block.text)
            if text.strip():
                blocks.append(Block(block.kind, text, dict(block.meta)))
                after += count_tokens(text)
        if blocks:
            turns.append(Turn(len(turns), turn.role, blocks))
    prior = prior_fork(marker, after) if marker is not None else None
    return Transcript(transcript.source, turns, dict(transcript.meta)), prior


@dataclass(frozen=True)
class ForkPlan:
    """`keep` is in history order. `rewrites[record][part]` replaces that content part's text;
    an empty string drops the part, except in a tool result, which keeps the part with empty
    text. `passthrough` are loose unknown records written unchanged."""

    keep: tuple[int, ...]
    rewrites: dict[int, dict[int, str]]
    passthrough: tuple[int, ...]
    marker: str
    tail_from: int
    compressible_tokens: int
    kept_tokens: int
    tail_tokens: int
    warnings: tuple[str, ...]


def marker_text(source: str, date: str, kept: int, total: int, tail: int) -> str:
    return _MARKER.format(source=source, date=date, kept=kept, total=total, tail=tail)


Piece = tuple[Segment, bool, Block]


def _rebuild(pieces: list[Piece]) -> str:
    if pieces[0][2].kind == "tool_result":
        lines: list[str] = []
        dropped = False
        for seg, kept, _ in pieces:
            if not kept:
                dropped = True
                continue
            if dropped:
                lines.append(GAP)
            lines.append(seg.text)
            dropped = False
        if dropped and lines:
            lines.append(GAP)
        return "\n".join(lines) if lines else TRIMMED_RESULT
    paras: list[str] = []
    for seg, kept, block in pieces:
        if not kept:
            continue
        if seg.kind == "code":
            paras.append(f"{_FENCE}{block.meta.get('lang', '')}\n{seg.text}\n{_FENCE}")
        else:
            paras.append(seg.text)
    return "\n\n".join(paras)


def region_tokens(session: Session, region: Region) -> tuple[int, int]:
    compressible = sum(session.tokens(ri) for ri in region.turn_record)
    tail = sum(session.tokens(ri) for ri in session.live[region.tail_from : region.end])
    return compressible, tail


def _tokens_after(rec: SessionRecord, rewrite: dict[int, str]) -> int:
    untouched = sum(count_tokens(b.text) for b in rec.blocks if int(b.meta[PART]) not in rewrite)
    return untouched + sum(count_tokens(text) for text in rewrite.values())


def plan_fork(
    session: Session,
    region: Region,
    kept: Sequence[Segment],
    *,
    date: str,
    keep_thinking: bool = False,
) -> ForkPlan:
    records = session.records
    kept_ids = {s.id for s in kept}
    # Same segmenter, same defaults as compress_transcript's `original`, so ids line up.
    pieces: dict[tuple[int, int], list[Piece]] = {}
    for seg in segment(region.transcript):
        block = region.transcript.turns[seg.turn].blocks[seg.block]
        key = (region.turn_record[seg.turn], int(block.meta[PART]))
        pieces.setdefault(key, []).append((seg, seg.id in kept_ids, block))
    units = unit_ids(session)
    # The fork ends at `end`; a unit with a record at or after it (parallel calls straddling the
    # trigger) cannot be kept whole, so its in-region half is dropped.
    split = {units[ri] for ri in session.live[region.end :]}
    # Its reply's other records go too (a fork must not end on a thinking-only assistant turn),
    # and with them their own units (a parallel call's result), to a fixed point: reply ids
    # and call ids each join records the other does not.
    drop = {ri for ri in session.live[: region.end] if units[ri] in split}
    while True:
        replies_out = {r for ri in drop if (r := records[ri].reply) is not None}
        units_out = {units[ri] for ri in drop}
        grown = {
            ri
            for ri in session.live[: region.end]
            if records[ri].reply in replies_out or units[ri] in units_out
        }
        if grown <= drop:
            break
        drop |= grown

    selected = {ri for (ri, _), group in pieces.items() if any(k for _, k, _ in group)}
    kept_units = {units[ri] for ri in selected}
    # The whole stretch, not just the turns: a result with no text (an empty output, or one an
    # earlier handoff trimmed to a placeholder) is no turn, yet its kept call needs it.
    chosen = {
        ri for ri in session.live[: region.tail_from] if units[ri] in kept_units and ri not in drop
    }
    rewrites: dict[int, dict[int, str]] = {}
    for (ri, part), group in pieces.items():
        if ri not in chosen or all(k for _, k, _ in group) or group[0][2].kind == "tool_use":
            continue
        rewrites.setdefault(ri, {})[part] = _rebuild(group)

    exchanges = {records[ri].exchange for ri in chosen}
    replies = {r for ri in chosen if (r := records[ri].reply) is not None}
    keep: list[int] = []
    warnings: list[str] = []
    for pos, ri in enumerate(session.live[: region.end]):
        rec = records[ri]
        if rec.category == "side" or ri in drop:
            continue
        if rec.category == "unknown":
            where = f"{rec.kind!r} record at line {rec.line + 1}"
            if pos < region.tail_from:
                raise HandoffRefusedError(
                    f"unknown {where} lies inside the stretch to be compressed; refusing rather "
                    f"than guess how {session.tool} reads it"
                )
            warnings.append(f"unknown {where} passed through unchanged")
            keep.append(ri)
        elif pos >= region.tail_from or rec.pinned:
            keep.append(ri)
        elif rec.thinking:
            if keep_thinking and rec.reply in replies:
                keep.append(ri)
        elif rec.category == "meta":
            # Tied to a message (Codex's echo of a prompt): it goes where that message goes.
            if ri in chosen if rec.call_ids else rec.exchange in exchanges:
                keep.append(ri)
        elif ri in chosen:
            keep.append(ri)
    passthrough = tuple(ri for ri in session.loose if records[ri].category == "unknown")
    warnings += [
        f"unknown {records[ri].kind!r} record at line {records[ri].line + 1} passed through "
        "unchanged"
        for ri in passthrough
    ]
    compressible, tail = region_tokens(session, region)
    kept_tokens = sum(
        _tokens_after(records[ri], rewrites.get(ri, {})) for ri in keep if ri in chosen
    )
    # After the count: a stripped part's blocks already hold its stripped text. A trim rewrite
    # wins, since it was rebuilt from stripped blocks.
    for ri in keep:
        for part, text in records[ri].cleaned.items():
            rewrites.setdefault(ri, {}).setdefault(part, text)
    return ForkPlan(
        keep=tuple(keep),
        rewrites=rewrites,
        passthrough=passthrough,
        marker=marker_text(session.session_id, date, kept_tokens, compressible, tail),
        tail_from=region.tail_from,
        compressible_tokens=compressible,
        kept_tokens=kept_tokens,
        tail_tokens=tail,
        warnings=tuple(warnings),
    )

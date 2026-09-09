"""Supersession dedupe: drop segments whose information a later segment makes redundant.

Lossless with respect to the final state of the conversation's files: a superseded Edit's
new_string is no longer the file's content once a later Edit/Write to the same path applied, a
Read output is stale once the file was read again or written, and an exact-duplicate text is
already present later in the transcript.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from loosy_goose.code import parse_tool_payload
from loosy_goose.segment import Segment
from loosy_goose.transcript import Transcript

Reason = Literal["edit_superseded", "read_superseded", "duplicate"]
Op = Literal["read", "write"]

_PATH_KEYS = ("file_path", "path", "notebook_path")
_READ_TOOLS = frozenset({"Read", "NotebookRead", "view", "str_replace_editor:view"})
_WRITE_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit", "create", "str_replace"})


@dataclass(frozen=True)
class Supersession:
    reason: Reason
    by: int


def _payload(segment: Segment) -> dict[str, Any] | None:
    return parse_tool_payload(segment.text) if segment.kind == "tool_use" else None


def _file_op(payload: dict[str, Any], tool_name: str | None) -> tuple[Op, str] | None:
    path = next((payload[k] for k in _PATH_KEYS if isinstance(payload.get(k), str)), None)
    if not path:
        return None
    if tool_name in _WRITE_TOOLS:
        return "write", path
    if tool_name in _READ_TOOLS:
        return "read", path
    if tool_name:
        return None
    # No tool name (Segment drops Block.meta): infer the operation from the payload shape.
    # Claude Code's Edit/Write/MultiEdit carry a body key; Read carries only path/offset/limit.
    # OpenHands' str_replace_editor multiplexes on `command`.
    command = payload.get("command")
    if command in ("view",):
        return "read", path
    if command in ("create", "str_replace", "insert", "undo_edit"):
        return "write", path
    if any(k in payload for k in ("new_string", "content", "edits", "new_str", "file_text")):
        return "write", path
    if set(payload) <= {*_PATH_KEYS, "offset", "limit", "view_range", "pages"}:
        return "read", path
    return None


def tool_names_from_transcript(transcript: Transcript, segments: list[Segment]) -> dict[int, str]:
    by_turn: dict[int, list[str]] = defaultdict(list)
    for turn in transcript.turns:
        for block in turn.blocks:
            if block.kind == "tool_use":
                by_turn[turn.index].append(block.meta.get("name", ""))
    out: dict[int, str] = {}
    cursor: dict[int, int] = defaultdict(int)
    for seg in segments:
        if seg.kind != "tool_use":
            continue
        names = by_turn.get(seg.turn, [])
        i = cursor[seg.turn]
        if i < len(names):
            out[seg.id] = names[i]
            cursor[seg.turn] = i + 1
    return out


def attribute_results(segments: list[Segment]) -> dict[int, int]:
    """Map each tool-result segment id to the tool_use segment id it answers.

    Segments carry no tool_use_id, so attribution is by adjacency: results land in the turn right
    after the calls, in call order. With one call the mapping is unambiguous; with several calls
    a result is attributed only when the result count matches, since a chunked result blurs the
    boundary between consecutive calls' outputs.
    """
    by_turn: dict[int, list[Segment]] = defaultdict(list)
    for s in segments:
        by_turn[s.turn].append(s)
    out: dict[int, int] = {}
    for turn, segs in by_turn.items():
        calls = [s for s in segs if s.kind == "tool_use"]
        if not calls:
            continue
        results = [s for s in by_turn.get(turn + 1, []) if s.kind in ("tool_result", "code")]
        if not results:
            continue
        if len(calls) == 1:
            for r in results:
                out[r.id] = calls[0].id
        elif len(calls) == len(results):
            for c, r in zip(calls, results, strict=True):
                out[r.id] = c.id
    return out


def find_superseded(
    segments: list[Segment],
    *,
    tool_names: Mapping[int, str] | None = None,
    result_owner: Mapping[int, int] | None = None,
) -> dict[int, Supersession]:
    names = tool_names or {}
    owner = result_owner if result_owner is not None else attribute_results(segments)
    out: dict[int, Supersession] = {}

    ops: dict[int, tuple[Op, str]] = {}
    for s in segments:
        payload = _payload(s)
        if payload is not None:
            file_op = _file_op(payload, names.get(s.id))
            if file_op is not None:
                ops[s.id] = file_op

    by_path: dict[str, list[int]] = defaultdict(list)
    for sid, (_, path) in ops.items():
        by_path[path].append(sid)
    for sids in by_path.values():
        sids.sort()
        for i, sid in enumerate(sids):
            op, _ = ops[sid]
            later = sids[i + 1 :]
            if op == "write":
                later_write = next((x for x in later if ops[x][0] == "write"), None)
                if later_write is not None:
                    out[sid] = Supersession("edit_superseded", later_write)
            elif later:
                out[sid] = Supersession("read_superseded", later[0])

    # A superseded Read drags its output along; the tool_use itself is a few tokens but the
    # result is the bulk.
    for rid, cid in owner.items():
        hit = out.get(cid)
        if hit is not None and hit.reason == "read_superseded" and rid not in out:
            out[rid] = Supersession("read_superseded", hit.by)

    last_by_text: dict[str, int] = {}
    for s in reversed(segments):
        key = s.text.strip()
        if not key:
            continue
        prev = last_by_text.get(key)
        if prev is None:
            last_by_text[key] = s.id
        elif s.id not in out:
            out[s.id] = Supersession("duplicate", prev)
    return out


def apply_supersession(
    segments: list[Segment], superseded: Mapping[int, Supersession] | None = None
) -> list[Segment]:
    gone = superseded if superseded is not None else find_superseded(segments)
    return [s for s in segments if s.id not in gone]

"""Chat-completions wire format (as spoken by local servers) to Transcript."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .transcript import Block, Role, Transcript, Turn, _text_of, split_fences

_ROLES: frozenset[str] = frozenset({"user", "assistant", "tool"})


_PREFIX_ROLES: frozenset[str] = frozenset({"system", "developer"})


def system_prefix(messages: Sequence[Mapping[str, Any]]) -> int:
    n = 0
    for m in messages:
        if m.get("role") not in _PREFIX_ROLES:
            break
        n += 1
    return n


def _tool_use(call: Any) -> Block | None:
    if not isinstance(call, Mapping):
        return None
    fn = call.get("function")
    if not isinstance(fn, Mapping):
        return None
    raw = fn.get("arguments", "")
    if isinstance(raw, str):
        try:
            text = json.dumps(json.loads(raw), ensure_ascii=False, sort_keys=True)
        except json.JSONDecodeError:
            text = raw
    else:
        text = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    return Block("tool_use", text, {"name": str(fn.get("name", ""))})


def is_barrier(msg: Mapping[str, Any]) -> bool:
    """True when `to_transcript` cannot carry the whole message, so freezing it would drop
    content: an unmapped role, a non-text content part, or a malformed tool call."""
    if msg.get("role") not in _ROLES:
        return True
    content = msg.get("content")
    if isinstance(content, list):
        if any(
            not (isinstance(p, str) or (isinstance(p, Mapping) and p.get("type") == "text"))
            for p in content
        ):
            return True
    elif content is not None and not isinstance(content, str):
        return True
    calls = msg.get("tool_calls")
    if calls is None:
        return False
    return (
        not isinstance(calls, Sequence)
        or isinstance(calls, str)
        or any(_tool_use(c) is None for c in calls)
    )


def to_transcript(messages: Sequence[Mapping[str, Any]]) -> tuple[Transcript, list[int]]:
    """Map each non-system message to a Turn; the list maps turn index to message index.

    Messages that yield no blocks (null content, no tool calls) get no turn, so the map
    stays aligned with Transcript.turns rather than with the raw message list.
    """
    turns: list[Turn] = []
    index: list[int] = []
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if not isinstance(role, str) or role not in _ROLES:
            continue
        blocks = split_fences(_text_of(msg.get("content")))
        if role == "tool":
            blocks = [Block("tool_result", b.text, b.meta) for b in blocks]
        elif role == "assistant":
            calls = msg.get("tool_calls")
            if isinstance(calls, Sequence) and not isinstance(calls, str):
                blocks.extend(b for c in calls if (b := _tool_use(c)) is not None)
        if blocks:
            turns.append(Turn(len(turns), cast(Role, role), blocks))
            index.append(i)
    return Transcript("<chat-messages>", turns), index

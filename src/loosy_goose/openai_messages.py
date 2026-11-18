"""Chat-completions wire format (as spoken by local servers) to Transcript."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .transcript import Block, Role, Transcript, Turn, _text_of, split_fences

_ROLES: frozenset[str] = frozenset({"user", "assistant", "tool"})


def system_prefix(messages: Sequence[Mapping[str, Any]]) -> int:
    n = 0
    for m in messages:
        if m.get("role") != "system":
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

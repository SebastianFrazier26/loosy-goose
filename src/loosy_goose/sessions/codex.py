"""Codex session files: `<home>/sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl`.

Undocumented. Each line is `{timestamp, type, payload}` (newer lines add `ordinal`, the line
index). History is file order; a `compacted` record replaces everything before it. The session id
appears in many fields. Codex indexes a new file into its own sqlite by scanning; that database
is never touched here (probe, 2026-10-12, codex-cli 0.158.0).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from loosy_goose.sessions.model import (
    Session,
    SessionFormatError,
    SessionRecord,
    Tool,
    explicit_session,
    read_jsonl,
    same_dir,
    tagged,
    text_blocks,
)
from loosy_goose.sessions.plan import MARKER_PREFIX, strip_part
from loosy_goose.transcript import Block

__all__ = ["Session", "TESTED_VERSIONS", "TOOL", "default_home", "locate", "read"]

TOOL: Tool = "codex"
TESTED_VERSIONS: tuple[str, ...] = ("0.158.0",)

_CALLS = frozenset({"function_call", "custom_tool_call"})
_OUTPUTS = frozenset({"function_call_output", "custom_tool_call_output"})
_META_ITEMS = frozenset(
    {"web_search_call", "agent_message", "tool_search_call", "tool_search_output"}
)
_META_TOP = frozenset(
    {
        "event_msg",
        "turn_context",
        "world_state",
        "token_usage_record",
        "inter_agent_communication_metadata",
    }
)
_TURN_END = frozenset({"event_msg:task_complete", "event_msg:turn_aborted"})


def default_home() -> Path:
    env = os.environ.get("CODEX_HOME")
    return Path(env) if env else Path.home() / ".codex"


def _head(path: Path) -> dict[str, Any] | None:
    with path.open("rb") as fh:
        try:
            obj = json.loads(fh.readline())
        except ValueError:
            return None
    if isinstance(obj, dict) and obj.get("type") == "session_meta":
        payload = obj.get("payload")
        return payload if isinstance(payload, dict) else None
    return None


def locate(cwd: Path, session: str, home: Path) -> list[Path]:
    root = home / "sessions"
    if session != "current":
        return explicit_session(session, lambda sid: root.glob(f"*/*/*/rollout-*-{sid}.jsonl"))
    files = sorted(
        root.glob("*/*/*/rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    out: list[Path] = []
    for path in files:
        head = _head(path)
        if head is not None and isinstance(head.get("cwd"), str) and same_dir(head["cwd"], cwd):
            out.append(path)
    return out


def _texts(content: Any) -> list[tuple[int, str]]:
    if not isinstance(content, list):
        return []
    return [
        (i, str(c["text"]))
        for i, c in enumerate(content)
        if isinstance(c, dict) and isinstance(c.get("text"), str)
    ]


def _arguments(raw: Any) -> str:
    if not isinstance(raw, str):
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, sort_keys=True)
    except ValueError:
        return raw


def _output(payload: dict[str, Any]) -> str:
    out = payload.get("output")
    if isinstance(out, str):
        return out
    return "\n".join(text for _, text in _texts(out))


def _item(n: int, obj: dict[str, Any], payload: dict[str, Any]) -> SessionRecord:
    ptype = str(payload.get("type"))
    kind = f"response_item:{ptype}"
    cleaned: dict[int, str] = {}
    if ptype == "message":
        role = payload.get("role")
        texts = _texts(payload.get("content"))
        if role == "user" and texts and texts[0][1].startswith(MARKER_PREFIX):
            return SessionRecord(n, obj, "response_item:loosy-marker", "side", marker=texts[0][1])
        blocks: list[Block] = [
            b for i, text in texts for b in text_blocks(strip_part(text, i, cleaned), i)
        ]
        if role == "assistant":
            return SessionRecord(
                n,
                obj,
                f"{kind}:assistant",
                "message",
                role="assistant",
                blocks=blocks,
                cleaned=cleaned,
            )
        if role in ("user", "developer"):
            return SessionRecord(
                n,
                obj,
                f"{kind}:{role}",
                "meta",
                role="user" if role == "user" else "system",
                blocks=blocks,
                pinned=True,
                cleaned=cleaned,
            )
        return SessionRecord(n, obj, f"{kind}:{role}", "unknown")
    if ptype == "reasoning":
        summary = _texts(payload.get("summary"))
        return SessionRecord(
            n,
            obj,
            kind,
            "message",
            role="assistant",
            blocks=[b for i, text in summary for b in tagged("thinking", text, i)],
            thinking=True,
        )
    if ptype in _CALLS:
        name = str(payload.get("name", ""))
        body = _arguments(payload.get("arguments", payload.get("input", "")))
        return SessionRecord(
            n,
            obj,
            kind,
            "message",
            role="assistant",
            blocks=tagged("tool_use", body, 0, name=name),
            call_ids=(str(payload.get("call_id", "")),),
            call_side="call",
            tool_name=name,
        )
    if ptype in _OUTPUTS:
        return SessionRecord(
            n,
            obj,
            kind,
            "message",
            role="tool",
            blocks=tagged("tool_result", strip_part(_output(payload), 0, cleaned), 0),
            call_ids=(str(payload.get("call_id", "")),),
            call_side="result",
            cleaned=cleaned,
        )
    if ptype in _META_ITEMS:
        return SessionRecord(n, obj, kind, "meta")
    return SessionRecord(n, obj, kind, "unknown")


def _event_type(payload: dict[str, Any]) -> str:
    # codex-cli 0.158.0 records the prompt echo as item_completed/UserMessage; older builds
    # wrote a user_message event. Both are the same echo.
    item = payload.get("item")
    if payload.get("type") == "item_completed" and isinstance(item, dict):
        if item.get("type") == "UserMessage":
            return "user_message"
    return str(payload.get("type"))


def _echo_text(payload: dict[str, Any]) -> str | None:
    message = payload.get("message")
    if isinstance(message, str):
        return message
    item = payload.get("item")
    if isinstance(item, dict):
        texts = _texts(item.get("content"))
        if texts:
            return "\n".join(text for _, text in texts)
    return None


def _classify(n: int, obj: dict[str, Any]) -> SessionRecord:
    top = str(obj.get("type"))
    raw_payload = obj.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    if top in ("session_meta", "compacted"):
        return SessionRecord(n, obj, top, "meta", pinned=True)
    if top == "response_item":
        return _item(n, obj, payload)
    if top == "event_msg":
        return SessionRecord(n, obj, f"event_msg:{_event_type(payload)}", "meta")
    if top in _META_TOP:
        return SessionRecord(n, obj, top, "meta")
    return SessionRecord(n, obj, top, "unknown")


def _promote_prompts(records: list[SessionRecord], live: list[int]) -> None:
    pending: int | None = None
    for ri in live:
        rec = records[ri]
        if rec.kind == "response_item:message:user":
            pending = ri
        elif rec.kind == "event_msg:task_started":
            pending = None
        elif rec.kind == "event_msg:user_message" and pending is not None:
            prompt = records[pending]
            message = _echo_text(rec.raw.get("payload", {}))
            prompt.category, prompt.pinned = "message", False
            prompt.prompt = (
                message if message is not None else " ".join(b.text for b in prompt.blocks)
            )
            # One unit, so the echo is kept or dropped with its prompt: a dropped prompt's text
            # would otherwise ship in its echo, and a later read would pin it on a context message.
            prompt.call_ids = rec.call_ids = (f"prompt-{prompt.line}",)
            pending = None


def _mark_exchanges(records: list[SessionRecord], live: list[int]) -> int | None:
    exchange = 0
    open_exchange: int | None = None
    for ri in live:
        rec = records[ri]
        if rec.kind == "event_msg:task_started":
            exchange += 1
            open_exchange = exchange
        rec.exchange = exchange
        if rec.role == "assistant":
            rec.reply = f"exchange-{exchange}"
        if rec.kind in _TURN_END:
            open_exchange = None
    return open_exchange


def read(path: Path) -> Session:
    lines, warnings = read_jsonl(path)
    records = [_classify(n, obj) for n, obj in lines]
    if not records or records[0].kind != "session_meta":
        raise SessionFormatError(f"{path}: the first line is not session_meta")
    meta = records[0].raw.get("payload", {})
    compactions = [i for i, r in enumerate(records) if r.kind == "compacted"]
    if compactions:
        last = compactions[-1]
        live = [i for i in range(last) if records[i].kind == "session_meta"]
        live += list(range(last, len(records)))
    else:
        live = list(range(len(records)))
    _promote_prompts(records, live)
    open_exchange = _mark_exchanges(records, live)
    version = meta.get("cli_version")
    return Session(
        tool=TOOL,
        path=path,
        session_id=str(meta.get("id", "")),
        cwd=str(meta.get("cwd", "")),
        cli_version=version if isinstance(version, str) else None,
        records=records,
        live=live,
        loose=[],
        open_exchange=open_exchange,
        warnings=warnings,
    )

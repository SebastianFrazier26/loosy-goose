"""Codex session files: `<home>/sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl`.

Undocumented. Each line is `{timestamp, type, payload}` (newer lines add `ordinal`, the line
index). History is file order; a `compacted` record replaces everything before it. The session id
appears in many fields. Codex indexes a new file into its own sqlite by scanning; that database
is never touched here (probe, 2026-10-12, codex-cli 0.158.0).
"""

from __future__ import annotations

import copy
import json
import os
import shlex
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loosy_goose.sessions.model import (
    Session,
    SessionFormatError,
    SessionRecord,
    Tool,
    dump_line,
    explicit_session,
    read_jsonl,
    same_dir,
    tagged,
    text_blocks,
)
from loosy_goose.sessions.plan import MARKER_PREFIX, ForkPlan, strip_part
from loosy_goose.transcript import Block

__all__ = [
    "Session",
    "TESTED_VERSIONS",
    "TOOL",
    "default_home",
    "fork_destination",
    "locate",
    "new_session_id",
    "read",
    "resume_command",
    "validate",
    "verify_command",
    "write_fork",
]

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


def new_session_id() -> str:
    # Codex ids are UUIDv7 (time-ordered); Python 3.12 has no uuid.uuid7.
    ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (
        (ms & ((1 << 48) - 1)) << 80
        | 0x7 << 76
        | ((rand >> 68) & 0xFFF) << 64
        | 0b10 << 62
        | (rand & ((1 << 62) - 1))
    )
    return str(uuid.UUID(int=value))


def fork_destination(source: Path, new_id: str, home: Path, now: datetime) -> Path:
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    return home / "sessions" / now.strftime("%Y/%m/%d") / f"rollout-{stamp}-{new_id}.jsonl"


def resume_command(session_id: str, cwd: str) -> str:
    return f"cd {shlex.quote(cwd)} && codex resume {session_id}"


def verify_command(session_id: str) -> list[str]:
    return ["codex", "exec", "resume", session_id, "Reply with exactly: OK"]


def _apply_rewrites(obj: dict[str, Any], rewrite: dict[int, str]) -> None:
    if not rewrite:
        return
    payload = obj["payload"]
    ptype = payload.get("type")
    if ptype == "message":
        content: list[Any] = []
        for i, part in enumerate(payload.get("content") or []):
            if i not in rewrite:
                content.append(part)
            elif rewrite[i]:
                content.append({**part, "text": rewrite[i]})
        payload["content"] = content
    elif ptype in _OUTPUTS:
        text = rewrite.get(0, "")
        out = payload.get("output")
        if isinstance(out, list):
            others = [c for c in out if not (isinstance(c, dict) and c.get("type") == "input_text")]
            payload["output"] = [{"type": "input_text", "text": text}, *others]
        else:
            payload["output"] = text


def _empty(obj: dict[str, Any]) -> bool:
    payload = obj.get("payload")
    if obj.get("type") != "response_item" or not isinstance(payload, dict):
        return False
    if payload.get("type") != "message":
        return False
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        return True
    return any(isinstance(p, dict) and "text" in p and not p["text"] for p in content)


def write_fork(session: Session, plan: ForkPlan, new_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = session.records
    kept: list[tuple[int, dict[str, Any]]] = []
    for ri in sorted(plan.keep):
        obj = copy.deepcopy(records[ri].raw)
        _apply_rewrites(obj, plan.rewrites.get(ri, {}))
        # Codex sends a message with no content to the model as-is, which the API rejects.
        if not _empty(obj):
            kept.append((ri, obj))
    objs = [obj for _, obj in kept]
    # After the last compaction, or the compaction would replace it; inside a kept turn, as
    # every user message Codex writes is.
    last_compaction = max(
        (i for i, (ri, _) in enumerate(kept) if records[ri].kind == "compacted"), default=-1
    )
    at = next(
        (
            i
            for i, (ri, _) in enumerate(kept)
            if i > last_compaction and records[ri].kind.startswith("response_item:")
        ),
        len(kept),
    )
    anchor = objs[at - 1] if at > 0 else objs[0]
    marker: dict[str, Any] = {
        "timestamp": anchor.get("timestamp"),
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": plan.marker}],
        },
    }
    if "ordinal" in anchor:
        marker["ordinal"] = 0
    objs.insert(at, marker)
    old = session.session_id
    dest = out_dir / f"{new_id}.jsonl"
    # "x", not "w": the destination sits beside the user's own sessions, and an id clash must
    # never truncate one of them.
    with dest.open("xb") as fh:
        for index, obj in enumerate(objs):
            if "ordinal" in obj:
                obj["ordinal"] = index
            line = dump_line(obj).decode("utf-8", errors="surrogateescape")
            # The marker names the source session on purpose; every other mention is the id.
            if obj is not marker:
                line = line.replace(old, new_id)
            if index == 0:
                head = json.loads(line)
                head["payload"]["forked_from_id"] = old
                line = dump_line(head).decode("utf-8", errors="surrogateescape")
            fh.write(line.encode("utf-8", errors="surrogateescape"))
    return dest


def validate(fork: Path, session_id: str) -> list[str]:
    lines, problems = read_jsonl(fork)
    failures = ["parse"] if problems else []
    objs = [obj for _, obj in lines]
    head = objs[0].get("payload", {}) if objs and objs[0].get("type") == "session_meta" else {}
    if head.get("id") != session_id:
        failures.append("session-id")
    old = head.get("forked_from_id")
    if not isinstance(old, str) or not old:
        failures.append("forked-from")
    else:
        dumped = [json.dumps(o, ensure_ascii=False) for o in objs[1:]]
        if any(old in d for d in dumped if MARKER_PREFIX not in d):
            failures.append("stale-id")
    calls: set[str] = set()
    outputs: set[str] = set()
    started: set[str] = set()
    any_started = False
    broken_turn = False
    for obj in objs:
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        ptype = payload.get("type")
        if obj.get("type") == "response_item" and ptype in _CALLS:
            calls.add(str(payload.get("call_id")))
        if obj.get("type") == "response_item" and ptype in _OUTPUTS:
            outputs.add(str(payload.get("call_id")))
        if obj.get("type") == "event_msg" and ptype == "task_started":
            any_started = True
            started.add(str(payload.get("turn_id")))
        # An end whose start is missing breaks the turn; an end before any start is the tail
        # of a turn that a compaction cut, which Codex itself writes.
        if (
            obj.get("type") == "event_msg"
            and ptype in ("task_complete", "turn_aborted")
            and any_started
            and str(payload.get("turn_id")) not in started
        ):
            broken_turn = True
    if calls != outputs:
        failures.append("tool-pairs")
    if broken_turn:
        failures.append("turns")
    if any(_empty(o) for o in objs):
        failures.append("empty-content")
    numbered = [(i, o["ordinal"]) for i, o in enumerate(objs) if "ordinal" in o]
    if any(i != ordinal for i, ordinal in numbered):
        failures.append("ordinals")
    return failures

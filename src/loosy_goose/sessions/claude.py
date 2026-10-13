"""Claude Code session files: `<home>/projects/<cwd slug>/<sessionId>.jsonl`.

Undocumented, and the format changes between releases. History is rebuilt by walking
`parentUuid` back from the leaf, not by line order. A gap that is not re-linked loads without
error but silently truncates history (probe, 2026-10-12, Claude Code 2.1.296). Parallel tool
calls are the exception: their records branch, and the walk alone misses some (see `_splice`).
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
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
from loosy_goose.sessions.plan import MARKER_PREFIX, ForkPlan, strip_part, strip_placeholders
from loosy_goose.transcript import Block

__all__ = [
    "Session",
    "TESTED_VERSIONS",
    "TOOL",
    "default_home",
    "fork_destination",
    "locate",
    "new_session_id",
    "project_slug",
    "read",
    "resume_command",
    "validate",
    "verify_command",
    "write_fork",
]

TOOL: Tool = "claude"
TESTED_VERSIONS: tuple[str, ...] = ("2.1.296",)

_SIDE_TYPES = frozenset(
    {
        "queue-operation",
        "last-prompt",
        "cost-state",
        "mode",
        "permission-mode",
        "atis-latch",
        "ai-title",
        "pr-link",
        "agent-name",
        "file-history-snapshot",
        "file-history-delta",
        "continued-in",
    }
)
_SIDE_SYSTEM = frozenset(
    {"turn_duration", "stop_hook_summary", "away_summary", "informational", "local_command"}
)
_THINKING = frozenset({"thinking", "redacted_thinking"})


def default_home() -> Path:
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(env) if env else Path.home() / ".claude"


def project_slug(cwd: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", cwd)


def _recorded_cwd(path: Path) -> str | None:
    with path.open("rb") as fh:
        for n, raw in enumerate(fh):
            if n >= 50:
                break
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("cwd"), str):
                return str(obj["cwd"])
    return None


def locate(cwd: Path, session: str, home: Path) -> list[Path]:
    projects = home / "projects"
    if session != "current":
        return explicit_session(session, lambda sid: projects.glob(f"*/{sid}.jsonl"))
    slugged = projects / project_slug(str(cwd))
    # The slug rule is inferred, not documented, so the recorded cwd decides; the slug only
    # narrows the scan when its directory exists.
    pool = slugged.glob("*.jsonl") if slugged.is_dir() else projects.glob("*/*.jsonl")
    found = [p for p in pool if (rec := _recorded_cwd(p)) is not None and same_dir(rec, cwd)]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _blocks(content: Any, cleaned: dict[int, str]) -> list[Block]:
    if isinstance(content, str):
        return text_blocks(strip_part(content, 0, cleaned), 0)
    if not isinstance(content, list):
        return []
    blocks: list[Block] = []
    for i, part in enumerate(content):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            blocks += text_blocks(strip_part(str(part.get("text", "")), i, cleaned), i)
        elif kind == "tool_use":
            body = json.dumps(part.get("input", {}), ensure_ascii=False, sort_keys=True)
            blocks += tagged("tool_use", body, i, name=str(part.get("name", "")))
        elif kind == "tool_result":
            body = strip_part(_content_text(part.get("content")), i, cleaned)
            blocks += tagged("tool_result", body, i)
        elif kind == "thinking":
            blocks += tagged("thinking", str(part.get("thinking", "")), i)
    return blocks


def _message(obj: dict[str, Any]) -> dict[str, Any]:
    msg = obj.get("message")
    return msg if isinstance(msg, dict) else {}


def _user(n: int, obj: dict[str, Any]) -> SessionRecord:
    content = _message(obj).get("content")
    parts = [p for p in content if isinstance(p, dict)] if isinstance(content, list) else []
    results = [p for p in parts if p.get("type") == "tool_result"]
    cleaned: dict[int, str] = {}
    if results:
        return SessionRecord(
            n,
            obj,
            "user:tool_result",
            "message",
            role="user",
            blocks=_blocks(content, cleaned),
            call_ids=tuple(str(p.get("tool_use_id", "")) for p in results),
            call_side="result",
            cleaned=cleaned,
        )
    if obj.get("isCompactSummary"):
        return SessionRecord(
            n,
            obj,
            "user:compact_summary",
            "message",
            role="user",
            blocks=_blocks(content, cleaned),
            pinned=True,
            cleaned=cleaned,
        )
    if obj.get("isMeta"):
        return SessionRecord(
            n,
            obj,
            "user:meta",
            "meta",
            role="user",
            blocks=_blocks(content, cleaned),
            cleaned=cleaned,
        )
    text = _content_text(content)
    if text.startswith(MARKER_PREFIX):
        return SessionRecord(n, obj, "user:loosy-marker", "side", marker=text)
    return SessionRecord(
        n,
        obj,
        "user",
        "message",
        role="user",
        blocks=_blocks(content, cleaned),
        prompt=strip_placeholders(text),
        cleaned=cleaned,
    )


def _assistant(n: int, obj: dict[str, Any]) -> SessionRecord:
    msg = _message(obj)
    content = msg.get("content")
    parts = [p for p in content if isinstance(p, dict)] if isinstance(content, list) else []
    types = {p.get("type") for p in parts}
    calls = [p for p in parts if p.get("type") == "tool_use"]
    cleaned: dict[int, str] = {}
    return SessionRecord(
        n,
        obj,
        "assistant",
        "message",
        role="assistant",
        blocks=_blocks(content, cleaned),
        thinking=bool(types) and types <= _THINKING,
        reply=str(msg["id"]) if "id" in msg else None,
        call_ids=tuple(str(c.get("id", "")) for c in calls),
        call_side="call" if calls else None,
        tool_name=str(calls[-1].get("name", "")) if calls else None,
        cleaned=cleaned,
    )


def _classify(n: int, obj: dict[str, Any]) -> SessionRecord:
    rtype = str(obj.get("type"))
    if rtype == "user":
        return _user(n, obj)
    if rtype == "assistant":
        return _assistant(n, obj)
    if rtype == "attachment":
        return SessionRecord(n, obj, "attachment", "meta")
    if rtype == "system":
        sub = str(obj.get("subtype"))
        if sub == "compact_boundary":
            return SessionRecord(n, obj, "system:compact_boundary", "meta", pinned=True)
        return SessionRecord(n, obj, f"system:{sub}", "side" if sub in _SIDE_SYSTEM else "unknown")
    return SessionRecord(n, obj, rtype, "side" if rtype in _SIDE_TYPES else "unknown")


def _chain(records: list[SessionRecord], by_uuid: dict[str, int], warnings: list[str]) -> list[int]:
    leaf = next(
        (
            i
            for i in range(len(records) - 1, -1, -1)
            if isinstance(records[i].raw.get("uuid"), str) and not records[i].raw.get("isSidechain")
        ),
        None,
    )
    chain: list[int] = []
    seen: set[int] = set()
    current = leaf
    while current is not None and current not in seen:
        seen.add(current)
        chain.append(current)
        parent = records[current].raw.get("parentUuid")
        if parent is None:
            break
        current = by_uuid.get(str(parent))
        if current is None:
            warnings.append(
                f"line {records[chain[-1]].line + 1}: parent {parent} is not in the file; "
                "history before it is not loaded"
            )
    chain.reverse()
    return _splice(records, chain)


def _splice(records: list[SessionRecord], chain: list[int]) -> list[int]:
    """`chain` with the parallel tool calls and results that hang off it put back.

    Claude Code writes one reply's parallel calls as a parentUuid fork: a sibling call record
    (same message id) and its result, or just a call's result, can sit on a side branch parented
    on a record of that reply; both shapes were recorded live (Claude Code 2.1.296, 2026-10-13).
    The model saw them, so they are history. The writer relinks them into one line, which a
    resumed fork loads with every result (live check, same date)."""
    on_chain = set(chain)
    uuid_of = {i: str(records[i].raw.get("uuid")) for i in range(len(records))}
    reply_uuids: dict[str, set[str]] = {}
    for i in chain:
        if (reply := records[i].reply) is not None:
            reply_uuids.setdefault(reply, set()).add(uuid_of[i])

    def off_chain(j: int) -> bool:
        rec = records[j]
        return j not in on_chain and "uuid" in rec.raw and not rec.raw.get("isSidechain")

    sibling_calls: dict[str, list[int]] = {}
    for j, rec in enumerate(records):
        if not off_chain(j) or rec.call_side != "call" or rec.reply not in reply_uuids:
            continue
        siblings = reply_uuids[str(rec.reply)]
        if str(rec.raw.get("parentUuid")) in siblings:
            sibling_calls.setdefault(str(rec.reply), []).append(j)
            siblings.add(uuid_of[j])
    call_reply: dict[str, str] = {}
    issuer: dict[str, str] = {}
    for i in [*chain, *(j for js in sibling_calls.values() for j in js)]:
        rec = records[i]
        if rec.call_side == "call" and rec.reply is not None:
            for cid in rec.call_ids:
                call_reply[cid], issuer[cid] = rec.reply, uuid_of[i]
    answered = {c for i in chain if records[i].call_side == "result" for c in records[i].call_ids}
    side_results: dict[str, list[int]] = {}
    for j, rec in enumerate(records):
        if not off_chain(j) or rec.call_side != "result":
            continue
        ids = [c for c in rec.call_ids if c in call_reply and c not in answered]
        if ids and all(str(rec.raw.get("parentUuid")) == issuer[c] for c in ids):
            side_results.setdefault(call_reply[ids[0]], []).append(j)
            answered.update(ids)
    if not sibling_calls and not side_results:
        return chain
    last_of = {reply: pos for pos, i in enumerate(chain) if (reply := records[i].reply)}
    out: list[int] = []
    for pos, i in enumerate(chain):
        out.append(i)
        reply = records[i].reply
        # After the reply's last record: the results answer all its calls, so they follow them.
        if reply is not None and last_of[reply] == pos:
            out += sibling_calls.get(reply, []) + side_results.get(reply, [])
    return out


def _mark_exchanges(records: list[SessionRecord], live: list[int]) -> None:
    exchange = 0
    for ri in live:
        rec = records[ri]
        if rec.prompt is not None:
            exchange += 1
        rec.exchange = exchange
        if rec.kind == "attachment" and exchange == 0:
            rec.pinned = True


def _first_str(records: list[SessionRecord], key: str) -> str | None:
    return next((str(r.raw[key]) for r in records if isinstance(r.raw.get(key), str)), None)


def read(path: Path) -> Session:
    lines, warnings = read_jsonl(path)
    records = [_classify(n, obj) for n, obj in lines]
    by_uuid = {
        str(r.raw["uuid"]): i for i, r in enumerate(records) if isinstance(r.raw.get("uuid"), str)
    }
    if not by_uuid:
        raise SessionFormatError(f"{path}: no Claude Code message records")
    live = _chain(records, by_uuid, warnings)
    _mark_exchanges(records, live)
    return Session(
        tool=TOOL,
        path=path,
        session_id=_first_str(records, "sessionId") or "",
        cwd=_first_str(records, "cwd") or "",
        cli_version=_first_str(list(reversed(records)), "version"),
        records=records,
        live=live,
        loose=[i for i, r in enumerate(records) if not isinstance(r.raw.get("uuid"), str)],
        warnings=warnings,
    )


_MARKER_KEYS = ("isSidechain", "userType", "cwd", "version", "gitBranch", "entrypoint", "timestamp")


def new_session_id() -> str:
    return str(uuid.uuid4())


def fork_destination(source: Path, new_id: str, home: Path, now: datetime) -> Path:
    return source.parent / f"{new_id}.jsonl"


def resume_command(session_id: str, cwd: str) -> str:
    return f"cd {shlex.quote(cwd)} && claude --resume {session_id}"


def verify_command(session_id: str) -> list[str]:
    return ["claude", "-p", "Reply with exactly: OK", "--resume", session_id]


def _apply_rewrites(obj: dict[str, Any], rewrite: dict[int, str]) -> None:
    if not rewrite:
        return
    msg = obj["message"]
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = rewrite.get(0, content)
        return
    parts: list[Any] = []
    for i, part in enumerate(content):
        result = part.get("type") == "tool_result"
        if i not in rewrite:
            parts.append(part)
        # An emptied tool_result stays: dropping it would orphan its tool_use.
        elif rewrite[i] or result:
            parts.append({**part, "content" if result else "text": rewrite[i]})
    msg["content"] = parts


def _with_session(obj: dict[str, Any], new_id: str) -> dict[str, Any]:
    for key in ("sessionId", "session_id"):
        if key in obj:
            obj[key] = new_id
    return obj


def _empty(obj: dict[str, Any]) -> bool:
    if obj.get("type") not in ("user", "assistant"):
        return False
    content = _message(obj).get("content")
    return content == "" or content == []


def write_fork(session: Session, plan: ForkPlan, new_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = session.records
    kept: list[tuple[int, dict[str, Any]]] = []
    for ri in plan.keep:
        obj = copy.deepcopy(records[ri].raw)
        _apply_rewrites(obj, plan.rewrites.get(ri, {}))
        # The API rejects a message with empty content; the chain re-links over the gap.
        if not _empty(obj):
            kept.append((ri, obj))
    template = next((records[ri].raw for ri, _ in kept if records[ri].prompt is not None), {})
    marker: dict[str, Any] = {k: template[k] for k in _MARKER_KEYS if k in template}
    marker.update(
        {
            "type": "user",
            "uuid": new_session_id(),
            "message": {"role": "user", "content": plan.marker},
        }
    )
    # Claude Code reads history from the last compact boundary on; a marker above it is cut.
    boundary = max(
        (i for i, (ri, _) in enumerate(kept) if records[ri].kind == "system:compact_boundary"),
        default=-1,
    )
    objs = [obj for _, obj in kept]
    chain = objs[: boundary + 1] + [marker] + objs[boundary + 1 :]
    parent: str | None = None
    for obj in chain:
        obj["parentUuid"] = parent
        obj["sessionId"] = new_id
        _with_session(obj, new_id)
        parent = str(obj["uuid"])
    loose = [_with_session(copy.deepcopy(records[ri].raw), new_id) for ri in plan.passthrough]
    last_prompt = next((p for ri, _ in reversed(kept) if (p := records[ri].prompt) is not None), "")
    leaf = {
        "type": "last-prompt",
        "lastPrompt": last_prompt,
        "leafUuid": parent,
        "sessionId": new_id,
    }
    dest = out_dir / f"{new_id}.jsonl"
    # "x", not "w": the destination sits beside the user's own sessions, and an id clash must
    # never truncate one of them.
    with dest.open("xb") as fh:
        for obj in [*chain, *loose, leaf]:
            fh.write(dump_line(obj))
    return dest


def validate(fork: Path, session_id: str) -> list[str]:
    lines, problems = read_jsonl(fork)
    failures = ["parse"] if problems else []
    objs = [obj for _, obj in lines]
    chain = [o for o in objs if isinstance(o.get("uuid"), str)]
    by_uuid = {str(o["uuid"]): o for o in chain}
    seen: set[str] = set()
    walked: list[dict[str, Any]] = []
    intact = bool(chain)
    current: dict[str, Any] | None = chain[-1] if chain else None
    while current is not None:
        uid = str(current["uuid"])
        if uid in seen:
            intact = False
            break
        seen.add(uid)
        walked.append(current)
        parent = current.get("parentUuid")
        if parent is None:
            break
        current = by_uuid.get(str(parent))
        if current is None:
            intact = False
    if not intact:
        failures.append("chain")
    if seen != set(by_uuid):
        failures.append("coverage")
    path = list(reversed(walked))
    if not _tool_pairs_ordered(path):
        failures.append("tool-pairs")
    if any(_empty_message(obj) for obj in path):
        failures.append("empty-content")
    if {o.get("sessionId") for o in objs if "sessionId" in o} != {session_id}:
        failures.append("session-id")
    return failures


def _parts(obj: dict[str, Any]) -> list[dict[str, Any]]:
    content = _message(obj).get("content")
    return [p for p in content if isinstance(p, dict)] if isinstance(content, list) else []


def _tool_pairs_ordered(path: list[dict[str, Any]]) -> bool:
    """The API's rule: an assistant turn's calls are all answered by the very next user
    message, results first. Consecutive records of one role merge into one message; records
    of other types (attachments, system) are not messages and are skipped."""
    pending: set[str] = set()
    previous: str | None = None
    for obj in path:
        role = obj.get("type")
        if role not in ("user", "assistant"):
            continue
        parts = _parts(obj)
        if role == "assistant":
            if previous != "assistant":
                if pending:
                    return False
            pending |= {str(p.get("id")) for p in parts if p.get("type") == "tool_use"}
        else:
            answered = [str(p.get("tool_use_id")) for p in parts if p.get("type") == "tool_result"]
            if pending and not answered:
                return False
            for cid in answered:
                if cid not in pending:
                    return False
                pending.discard(cid)
        previous = role
    return not pending


def _empty_message(obj: dict[str, Any]) -> bool:
    if obj.get("type") not in ("user", "assistant"):
        return False
    content = _message(obj).get("content")
    if isinstance(content, str):
        return content == ""
    if not isinstance(content, list) or not content:
        return True
    return any(p.get("type") == "text" and not p.get("text") for p in _parts(obj))

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Role = Literal["user", "assistant", "system", "tool"]
BlockKind = Literal["text", "code", "tool_use", "tool_result", "thinking"]

_FENCE = re.compile(r"```([\w+#.-]*)[ \t]*\n(.*?)```", re.DOTALL)
# OpenHands SFT trajectories (SWE-Gym) carry tool calls inline as text rather than as
# structured blocks: assistant turns wrap calls in <function=NAME>…</function>, and the
# following user turn starts with "EXECUTION RESULT of [NAME]:".
_INLINE_CALL = re.compile(r"<function=([\w.-]+)>(.*?)</function>", re.DOTALL)
_INLINE_RESULT = re.compile(r"\AEXECUTION RESULT of \[[\w.-]+\]:\s*", re.DOTALL)


@dataclass
class Block:
    kind: BlockKind
    text: str
    meta: dict[str, str] = field(default_factory=dict)


@dataclass
class Turn:
    index: int
    role: Role
    blocks: list[Block]


@dataclass
class Transcript:
    source: str
    turns: list[Turn]
    meta: dict[str, int] = field(default_factory=dict)

    def text(self, kinds: Iterable[BlockKind] | None = None, sep: str = "\n\n") -> str:
        wanted = set(kinds) if kinds is not None else None
        parts = [
            b.text
            for t in self.turns
            for b in t.blocks
            if (wanted is None or b.kind in wanted) and b.text
        ]
        return sep.join(parts)


def split_fences(text: str) -> list[Block]:
    blocks: list[Block] = []
    pos = 0
    for m in _FENCE.finditer(text):
        before = text[pos : m.start()].strip()
        if before:
            blocks.append(Block("text", before))
        lang = m.group(1)
        code = m.group(2).rstrip("\n")
        if code:
            blocks.append(Block("code", code, {"lang": lang} if lang else {}))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        blocks.append(Block("text", tail))
    return blocks


def _split_inline_tools(text: str, role: str) -> list[Block] | None:
    if role == "user":
        m = _INLINE_RESULT.match(text)
        if m is None:
            return None
        body = text[m.end() :]
        return [Block("tool_result", body)] if body.strip() else []
    if role == "assistant" and _INLINE_CALL.search(text):
        blocks: list[Block] = []
        pos = 0
        for m in _INLINE_CALL.finditer(text):
            before = text[pos : m.start()]
            blocks.extend(split_fences(before))
            blocks.append(Block("tool_use", m.group(2).strip(), {"name": m.group(1)}))
            pos = m.end()
        blocks.extend(split_fences(text[pos:]))
        return blocks
    return None


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
        return "\n".join(pieces)
    return ""


def _blocks_from_content(content: Any, skipped: Counter[str]) -> list[Block]:
    if isinstance(content, str):
        return split_fences(content)
    if not isinstance(content, list):
        skipped["content:" + type(content).__name__] += 1
        return []
    blocks: list[Block] = []
    for item in content:
        if not isinstance(item, dict):
            skipped["block:non-dict"] += 1
            continue
        kind = item.get("type")
        if kind == "text":
            blocks.extend(split_fences(str(item.get("text", ""))))
        elif kind == "tool_use":
            blocks.append(
                Block(
                    "tool_use",
                    json.dumps(item.get("input", {}), ensure_ascii=False, sort_keys=True),
                    {"name": str(item.get("name", ""))},
                )
            )
        elif kind == "tool_result":
            body = _text_of(item.get("content"))
            if body:
                blocks.append(Block("tool_result", body))
        elif kind == "thinking":
            body = str(item.get("thinking", ""))
            if body:
                blocks.append(Block("thinking", body))
        else:
            skipped[f"block:{kind}"] += 1
    return blocks


def load_claude_code_jsonl(path: str | Path) -> Transcript:
    # Claude Code's session JSONL is explicitly undocumented and changes between releases
    # (docs: "scripts that parse these files directly can break on any release"). Everything
    # unrecognised is skipped and tallied into Transcript.meta so a format drift shows up as a
    # jump in skip counts rather than as a crash or silently emptier transcripts.
    path = Path(path)
    skipped: Counter[str] = Counter()
    turns: list[Turn] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped["line:malformed"] += 1
                continue
            if not isinstance(rec, dict):
                skipped["line:non-object"] += 1
                continue
            rtype = rec.get("type")
            if rtype not in ("user", "assistant"):
                skipped[f"record:{rtype}"] += 1
                continue
            msg = rec.get("message")
            if not isinstance(msg, dict):
                skipped["record:no-message"] += 1
                continue
            role = msg.get("role", rtype)
            if role not in ("user", "assistant"):
                skipped[f"role:{role}"] += 1
                continue
            blocks = _blocks_from_content(msg.get("content"), skipped)
            if blocks:
                turns.append(Turn(len(turns), role, blocks))
    return Transcript(str(path), turns, dict(skipped))


def load_messages_json(
    source: str | Path | Sequence[Any], *, name: str | None = None
) -> Transcript:
    if isinstance(source, str | Path):
        path = Path(source)
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        label = name or str(path)
    else:
        payload = list(source)
        label = name or "<messages>"
    if isinstance(payload, dict) and "messages" in payload:
        payload = payload["messages"]
    skipped: Counter[str] = Counter()
    turns: list[Turn] = []
    for item in payload:
        if not isinstance(item, dict):
            skipped["message:non-dict"] += 1
            continue
        role = item.get("role")
        if role not in ("user", "assistant", "system", "tool"):
            skipped[f"role:{role}"] += 1
            continue
        content = item.get("content")
        blocks: list[Block] | None = None
        if isinstance(content, str):
            blocks = _split_inline_tools(content, role)
        if blocks is None:
            blocks = _blocks_from_content(content, skipped)
        if role == "tool":
            blocks = [Block("tool_result", b.text, b.meta) for b in blocks]
        if blocks:
            turns.append(Turn(len(turns), role, blocks))
    return Transcript(label, turns, dict(skipped))

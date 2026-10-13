"""The tool-agnostic view of an agent CLI's session file that the handoff plans over.

Only the adapters (`claude.py`, `codex.py`) know a file format; everything here and in `plan.py`
works on `Session` alone.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Block, BlockKind, Role, split_fences

Tool = Literal["claude", "codex"]
Category = Literal["message", "meta", "side", "unknown"]
CallSide = Literal["call", "result"]

PART = "part"
SESSION_ID = re.compile(r"[0-9A-Za-z-]{8,64}")


class SessionFormatError(ValueError):
    pass


@dataclass
class SessionRecord:
    line: int
    raw: dict[str, Any]
    kind: str
    category: Category
    role: Role | None = None
    blocks: list[Block] = field(default_factory=list)
    prompt: str | None = None
    thinking: bool = False
    pinned: bool = False
    reply: str | None = None
    call_ids: tuple[str, ...] = ()
    call_side: CallSide | None = None
    tool_name: str | None = None
    exchange: int = -1
    marker: str | None = None
    cleaned: dict[int, str] = field(default_factory=dict)


@dataclass
class Session:
    tool: Tool
    path: Path
    session_id: str
    cwd: str
    cli_version: str | None
    records: list[SessionRecord]
    live: list[int]
    loose: list[int] = field(default_factory=list)
    open_exchange: int | None = None
    warnings: list[str] = field(default_factory=list)

    def tokens(self, index: int) -> int:
        return sum(count_tokens(b.text) for b in self.records[index].blocks)

    def prompts(self) -> list[str]:
        return [p for i in self.live if (p := self.records[i].prompt) is not None]


def snapshot(src: Path, dest: Path) -> int:
    """Copy `src` to `dest` up to its last newline; returns the number of bytes left out."""
    # The CLI is appending to this file while the handoff runs: a final line without its newline
    # is a record still being written, not a record.
    data = src.read_bytes()
    cut = data.rfind(b"\n") + 1
    dest.write_bytes(data[:cut])
    return len(data) - cut


def read_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    out: list[tuple[int, dict[str, Any]]] = []
    warnings: list[str] = []
    for n, raw in enumerate(path.read_bytes().split(b"\n")):
        line = raw.rstrip(b"\r")
        if not line.strip():
            continue
        try:
            # surrogateescape so a byte that is not UTF-8 survives into the fork unchanged.
            obj = json.loads(line.decode("utf-8", errors="surrogateescape"))
        except json.JSONDecodeError:
            warnings.append(f"line {n + 1}: not valid JSON, skipped")
            continue
        if not isinstance(obj, dict):
            warnings.append(f"line {n + 1}: not a JSON object, skipped")
            continue
        out.append((n, obj))
    return out, warnings


def dump_line(obj: dict[str, Any]) -> bytes:
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return (text + "\n").encode("utf-8", errors="surrogateescape")


def text_blocks(text: str, part: int) -> list[Block]:
    blocks = split_fences(text)
    for b in blocks:
        b.meta[PART] = str(part)
    return blocks


def tagged(kind: BlockKind, text: str, part: int, **meta: str) -> list[Block]:
    return [Block(kind, text, {PART: str(part), **meta})] if text.strip() else []


def same_dir(recorded: str, cwd: Path) -> bool:
    try:
        return Path(recorded).resolve() == cwd.resolve()
    except OSError:
        return False


def explicit_session(session: str, by_id: Callable[[str], Iterable[Path]]) -> list[Path]:
    given = Path(session)
    if given.is_absolute() and given.is_file():
        return [given]
    # Only an id-shaped string reaches the adapter's glob, so "*" cannot match every session.
    if not SESSION_ID.fullmatch(session):
        return []
    return sorted(by_id(session))


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version)[:3])


def newer_than_tested(version: str | None, tested: tuple[str, ...]) -> bool:
    if version is None or not tested:
        return False
    return version_key(version) > max(version_key(t) for t in tested)

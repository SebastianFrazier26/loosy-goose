"""Paths and install helpers for the committed synthetic session fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from session_recorder import CODE_WORDS, FIXTURE_CWD, FIXTURE_HOME, SENTINEL

__all__ = [
    "CLAUDE_FIXTURE",
    "CODEX_FIXTURE",
    "CODE_WORDS",
    "FIXTURES",
    "FIXTURE_CWD",
    "FIXTURE_HOME",
    "SENTINEL",
    "fixture_lines",
    "write_lines",
]

FIXTURES = Path(__file__).parent / "fixtures" / "sessions"
CLAUDE_FIXTURE = "claude_codewords.jsonl"
CODEX_FIXTURE = "codex_codewords.jsonl"


def fixture_lines(name: str) -> list[dict[str, Any]]:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_lines(path: Path, lines: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n" for obj in lines
    )
    path.write_text(body, encoding="utf-8", newline="\n")
    return path

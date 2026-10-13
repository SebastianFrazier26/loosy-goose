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
    "install_claude",
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


def install_claude(
    home: Path,
    *,
    lines: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    project_dir: str | None = None,
) -> Path:
    from loosy_goose.sessions import claude

    body = [dict(o) for o in (lines if lines is not None else fixture_lines(CLAUDE_FIXTURE))]
    sid = session_id or next(str(o["sessionId"]) for o in body if "sessionId" in o)
    for obj in body:
        for key in ("sessionId", "session_id"):
            if key in obj:
                obj[key] = sid
    folder = project_dir or claude.project_slug(str(Path(FIXTURE_CWD)))
    return write_lines(home / "projects" / folder / f"{sid}.jsonl", body)

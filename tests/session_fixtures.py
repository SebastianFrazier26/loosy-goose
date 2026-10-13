"""Paths and install helpers for the committed synthetic session fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from session_recorder import CODE_WORDS, FIXTURE_CWD, FIXTURE_HOME, SENTINEL

if TYPE_CHECKING:
    from loosy_goose.segment import Segment
    from loosy_goose.sessions.model import Session
    from loosy_goose.sessions.plan import ForkPlan

__all__ = [
    "CLAUDE_FIXTURE",
    "CODEX_FIXTURE",
    "CODE_WORDS",
    "FIXTURES",
    "FIXTURE_CWD",
    "FIXTURE_HOME",
    "SENTINEL",
    "fixture_lines",
    "fork_with",
    "install_claude",
    "read_lines",
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


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def fork_with(
    adapter: ModuleType,
    tmp_path: Path,
    lines: list[dict[str, Any]] | None = None,
    keep: Callable[[Segment], bool] = lambda s: True,
) -> tuple[Session, ForkPlan, str, Path]:
    """Fork `lines` (default: the adapter's fixture) keeping the segments `keep` selects, with
    no protected tail, into `tmp_path / "out"`."""
    from session_builders import kept_where

    from loosy_goose.sessions.plan import DEFAULT_TRIGGERS, find_trigger, plan_fork, split_region

    fixture = CLAUDE_FIXTURE if adapter.TOOL == "claude" else CODEX_FIXTURE
    source = write_lines(tmp_path / "src.jsonl", lines) if lines is not None else FIXTURES / fixture
    s = adapter.read(source)
    trigger = find_trigger(s, DEFAULT_TRIGGERS[adapter.TOOL])
    region = split_region(s, trigger, protect_tail_tokens=0)
    plan = plan_fork(s, region, kept_where(region, keep), date="2026-10-13")
    new_id = adapter.new_session_id()
    return s, plan, new_id, adapter.write_fork(s, plan, new_id, tmp_path / "out")

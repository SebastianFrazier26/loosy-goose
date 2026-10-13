"""Drift and end-to-end handoff tests against the real CLIs.

Opt-in: `uv run pytest -m live`. Needs `claude` and `codex` on PATH and logged in, and spends a
little API use. Run after upgrading either CLI: a failure here means its session format moved.
Each test deletes only what it made: the session file it recorded, the forks it wrote, and its
scratch repo. Codex's sqlite index is never touched, so it may keep a stale row for a deleted
synthetic Codex session.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import pytest
from session_builders import kept_where
from session_recorder import CODE_WORDS, SENTINEL, record_claude, record_codex, scratch_repo

from loosy_goose.segment import Segment
from loosy_goose.sessions import claude, codex
from loosy_goose.sessions.adapter import SessionAdapter
from loosy_goose.sessions.model import Session
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    ForkPlan,
    find_trigger,
    plan_fork,
    split_region,
)

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[1]
ASK = (
    # "Code word you were told" alone made both CLIs' models skip GAMMA, which only appears in
    # a shell command and its output (live run, 2026-10-13).
    "List every code word or all-caps word that appears earlier in this conversation, including "
    "in shell commands and their output, comma-separated, nothing else. "
    # Not "the first message": in Codex that is the injected AGENTS.md, not the marker.
    'Then, on a new line, copy word for word the message that starts with "[loosy-goose:".'
)
MARKER_SEEN = "loosy-goose: compressed from session"
SERVER = [
    "uv",
    "run",
    "--project",
    str(REPO),
    "--extra",
    "embed",
    "--extra",
    "mcp",
    "loosy-goose-mcp",
]
Recorder = Callable[..., Path]


def _need(*clis: str) -> None:
    for cli in clis:
        if shutil.which(cli) is None:
            pytest.skip(f"{cli} is not on PATH")


def _ask(argv: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
        env={**os.environ, **(env or {})},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def _words(text: str) -> set[str]:
    return {w for w in CODE_WORDS if w in text}


def _session_words(s: Session) -> set[str]:
    return _words(" ".join(b.text for ri in s.live for b in s.records[ri].blocks))


def _setup(tool: str) -> tuple[SessionAdapter, Path, Recorder, Callable[[str], list[str]]]:
    if tool == "claude":
        return (
            claude,
            claude.default_home(),
            record_claude,
            (lambda fid: ["claude", "-p", ASK, "--resume", fid]),
        )
    return (
        codex,
        codex.default_home(),
        record_codex,
        (lambda fid: ["codex", "exec", "resume", fid, ASK]),
    )


def _cleanup(created: list[Path], cwd: Path) -> None:
    # Only what the test made (Global Constraints: the only files deleted are our own).
    for path in created:
        path.unlink(missing_ok=True)
    shutil.rmtree(cwd, ignore_errors=True)


def _drift(
    adapter: SessionAdapter,
    source: Path,
    home: Path,
    cwd: Path,
    work: Path,
    keep: Callable[[Segment], bool],
    resume: Callable[[str], list[str]],
    *,
    keep_thinking: bool = False,
) -> tuple[str, Session, ForkPlan]:
    session = adapter.read(source)
    region = split_region(
        session, find_trigger(session, DEFAULT_TRIGGERS[adapter.TOOL]), protect_tail_tokens=0
    )
    plan = plan_fork(
        session,
        region,
        kept_where(region, keep),
        date=date.today().isoformat(),
        keep_thinking=keep_thinking,
    )
    fork_id = adapter.new_session_id()
    staged = adapter.write_fork(session, plan, fork_id, work)
    assert adapter.validate(staged, fork_id) == []
    dest = adapter.fork_destination(source, fork_id, home, datetime.now())
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(staged, dest)
    try:
        return _ask(resume(fork_id), cwd), session, plan
    finally:
        dest.unlink(missing_ok=True)


@pytest.mark.parametrize("tool", ["claude", "codex"])
def test_drift_a_fork_resumes_with_exactly_the_kept_words(tool: str, tmp_path: Path) -> None:
    _need(tool)
    adapter, home, record, resume = _setup(tool)
    cwd = scratch_repo()
    created: list[Path] = []
    try:
        source = record(cwd, home)
        created.append(source)
        answer, _, _ = _drift(
            adapter, source, home, cwd, tmp_path, lambda s: "BETA" not in s.text, resume
        )
    finally:
        _cleanup(created, cwd)
    assert _words(answer) == {"ALPHA", "GAMMA", "DELTA"}
    assert MARKER_SEEN in answer


@pytest.mark.parametrize("tool", ["claude", "codex"])
def test_drift_dropping_older_thinking_survives_resume(tool: str, tmp_path: Path) -> None:
    _need(tool)
    adapter, home, record, resume = _setup(tool)
    cwd = scratch_repo()
    created: list[Path] = []
    try:
        source = record(cwd, home)
        created.append(source)
        answer, session, plan = _drift(adapter, source, home, cwd, tmp_path, lambda s: True, resume)
    finally:
        _cleanup(created, cwd)
    # Only thinking before the tail is dropped; the trigger turn and the tail say nothing here.
    older = [ri for ri in session.live[: plan.tail_from] if session.records[ri].thinking]
    if not older:
        pytest.skip("this recording has no thinking before the protected tail; nothing dropped")
    assert not set(older) & set(plan.keep)
    assert _words(answer) == set(CODE_WORDS)


def _check_e2e(source: Session, fork: Session, answer: str) -> None:
    assert _words(answer) == _session_words(fork)
    assert MARKER_SEEN in answer
    delta = next(p for p in source.prompts() if "DELTA" in p)
    assert delta in fork.prompts()
    assert not any(SENTINEL in p or "/loosy" in p for p in fork.prompts())


def test_e2e_claude_handoff_through_the_mcp_tool(tmp_path: Path) -> None:
    _need("claude", "uv")
    home = claude.default_home()
    cwd = scratch_repo()
    shutil.copy(REPO / "integrations" / "claude" / "loosy.md", cwd / ".claude" / "commands")
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps({"mcpServers": {"loosy-goose": {"command": SERVER[0], "args": SERVER[1:]}}}),
        encoding="utf-8",
    )
    created: list[Path] = []
    try:
        source = record_claude(cwd, home, trigger=False)
        created.append(source)
        existing = set(source.parent.glob("*.jsonl"))
        # Prompt before --allowedTools: the flag is variadic and would swallow it.
        out = _ask(
            [
                "claude",
                "-p",
                "/loosy heavy protect_tail_tokens=0",
                "--resume",
                source.stem,
                "--mcp-config",
                str(config),
                "--strict-mcp-config",
                "--allowedTools",
                "mcp__loosy-goose__handoff_session",
            ],
            cwd,
            env={"MCP_TIMEOUT": "120000", "MCP_TOOL_TIMEOUT": "600000"},
        )
        forks = sorted(set(source.parent.glob("*.jsonl")) - existing)
        created += forks
        assert len(forks) == 1, out
        fork = claude.read(forks[0])
        answer = _ask(["claude", "-p", ASK, "--resume", fork.session_id], cwd)
        _check_e2e(claude.read(source), fork, answer)
    finally:
        _cleanup(created, cwd)


def _forked_from(path: Path) -> str | None:
    with path.open(encoding="utf-8") as fh:
        head = json.loads(fh.readline())
    value = head.get("payload", {}).get("forked_from_id")
    return value if isinstance(value, str) else None


def test_e2e_codex_handoff_through_the_mcp_tool(tmp_path: Path) -> None:
    _need("codex", "uv")
    home = codex.default_home()
    cwd = scratch_repo()
    created: list[Path] = []
    try:
        source = record_codex(cwd, home, trigger=False)
        created.append(source)
        sid = codex.read(source).session_id
        overrides = [
            "-c",
            f"mcp_servers.loosy-goose.command={json.dumps(SERVER[0])}",
            "-c",
            f"mcp_servers.loosy-goose.args={json.dumps(SERVER[1:])}",
            "-c",
            "mcp_servers.loosy-goose.startup_timeout_sec=120",
            "-c",
            "mcp_servers.loosy-goose.tool_timeout_sec=600",
            # Without `required`, exec mode starts the turn before the server has loaded its
            # model, so the tool is missing. Exec runs with approval policy "never", which
            # rejects an MCP call needing approval; approve this one tool, for this run only.
            "-c",
            "mcp_servers.loosy-goose.required=true",
            "-c",
            'mcp_servers.loosy-goose.tools.handoff_session.approval_mode="approve"',
        ]
        prompt = (
            f"{SENTINEL} Call the handoff_session tool of the loosy-goose MCP server exactly "
            f'once with level "heavy", protect_tail_tokens 0, tool "codex" and cwd '
            f"{json.dumps(str(cwd))}, then reply with its resume_command only."
        )
        _ask(["codex", "exec", "resume", *overrides, sid, prompt], cwd)
        forks = [
            p for p in (home / "sessions").glob("*/*/*/rollout-*.jsonl") if _forked_from(p) == sid
        ]
        created += forks
        assert len(forks) == 1
        fork = codex.read(forks[0])
        answer = _ask(["codex", "exec", "resume", fork.session_id, ASK], cwd)
        _check_e2e(codex.read(source), fork, answer)
    finally:
        _cleanup(created, cwd)

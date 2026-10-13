import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp import Client  # noqa: E402
from session_fixtures import FIXTURE_CWD, fake_embeddings, install_claude  # noqa: E402

from loosy_goose import mcp_server  # noqa: E402
from loosy_goose.sessions.plan import GAP, MARKER_PREFIX, marker_text  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402

_CONTENTS = [
    "Please look at the failing test in tests/test_main.py and tell me what you see.",
    "The assertion compares floats exactly; a tolerance is needed around line 42.",
    "Add the tolerance and rerun the suite for me.",
    "Done, the suite is green again with pytest.approx in place.",
    "Now also update src/app/main.py so compute() returns a float.",
    "Updated compute() in src/app/main.py; it now returns float(total) / 3.",
    "Great, thanks. Summarise what changed.",
    "Two edits: tolerance in tests/test_main.py and a float return in src/app/main.py.",
]
MESSAGES = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": c} for i, c in enumerate(_CONTENTS)
]


@pytest.fixture(autouse=True)
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embeddings)


def _call(server: Any, name: str, args: dict[str, Any]) -> Any:
    async def go() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, args)

    return anyio.run(go)


def _text(result: Any) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


def _homes(tmp_path: Path) -> dict[str, Path]:
    return {"claude": tmp_path / "claude", "codex": tmp_path / "codex"}


def test_serves_both_tools() -> None:
    async def go() -> set[str]:
        async with Client(mcp_server.build_server()) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert anyio.run(go) == {"compress", "handoff_session"}


def test_compress_returns_kept_turns_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _call(mcp_server.build_server(), "compress", {"messages": MESSAGES, "level": "light"})
    assert not result.is_error, _text(result)
    data = result.structured_content
    assert (data["header"]["keep"], data["header"]["keep_source"]) == (0.5, "level")
    assert data["turns"] and all(t["blocks"] for t in data["turns"])
    assert list(tmp_path.iterdir()) == []


def test_compress_reads_a_transcript_path(tmp_path: Path) -> None:
    path = tmp_path / "chat.json"
    path.write_text(json.dumps(MESSAGES), encoding="utf-8")
    result = _call(mcp_server.build_server(), "compress", {"path": str(path), "keep": 0.5})
    assert not result.is_error, _text(result)
    assert result.structured_content["header"]["keep_source"] == "keep"


def test_compress_strips_an_earlier_forks_artifacts_and_warns() -> None:
    marker = {"role": "user", "content": marker_text("old-id", "2026-10-12", 10, 40, 5)}
    tagged = [dict(m) for m in MESSAGES]
    tagged[3]["content"] = f"{tagged[3]['content']}\n{GAP}"
    plain = _call(mcp_server.build_server(), "compress", {"messages": MESSAGES, "level": "light"})
    again = _call(
        mcp_server.build_server(), "compress", {"messages": [marker, *tagged], "level": "light"}
    )
    assert not again.is_error, _text(again)
    header = again.structured_content["header"]
    assert header["input_tokens"] == plain.structured_content["header"]["input_tokens"]
    added = sum(count_tokens(c) for c in _CONTENTS) - (10 + 5)
    assert header["warnings"] == [
        f"this session was already compressed by loosy-goose on 2026-10-12; about {added:,} "
        "tokens added since — compressing again compounds the loss"
    ]
    assert plain.structured_content["header"]["warnings"] == []
    body = json.dumps(again.structured_content["turns"])
    assert GAP not in body and MARKER_PREFIX not in body


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"messages": MESSAGES, "keep": 0.5, "level": "light"}, "only one of"),
        ({"messages": MESSAGES, "level": "huge"}, "level must be one of"),
        ({"messages": MESSAGES, "keep": 0.9}, "keep must be in"),
        ({"level": "light"}, "exactly one of messages or path"),
    ],
)
def test_compress_rejects_bad_arguments(args: dict[str, Any], message: str) -> None:
    result = _call(mcp_server.build_server(), "compress", args)
    assert result.is_error and message in _text(result)


def test_handoff_session_forks_the_fixture(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude")
    server = mcp_server.build_server(homes=_homes(tmp_path))
    result = _call(
        server,
        "handoff_session",
        {"tool": "claude", "cwd": FIXTURE_CWD, "level": "heavy", "protect_tail_tokens": 0},
    )
    assert not result.is_error, _text(result)
    data = result.structured_content
    assert data["status"] == "forked" and Path(data["fork_path"]).exists()
    assert data["resume_command"].endswith(data["fork_session"])
    assert data["original_resume_command"].endswith(data["source_session"])


def test_handoff_session_reports_errors_as_tool_errors(tmp_path: Path) -> None:
    server = mcp_server.build_server(homes=_homes(tmp_path))
    result = _call(server, "handoff_session", {"tool": "claude", "cwd": FIXTURE_CWD})
    assert result.is_error and "no Claude Code session found" in _text(result)


def test_preload_exits_with_the_install_command_without_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_server, "find_spec", lambda name: None)
    with pytest.raises(SystemExit, match=r"embed.*uv tool install"):
        mcp_server.preload_model()

import json
from pathlib import Path
from typing import Any

import pytest
from session_fixtures import CLAUDE_FIXTURE, CODEX_FIXTURE, FIXTURES, fixture_lines
from session_recorder import (
    PROMPTS,
    SCRUBBED,
    RecorderError,
    check_clean,
    scrub_claude,
    scrub_codex,
    write_fixture,
)


def test_unlisted_strings_are_blanked_and_exact_ones_kept() -> None:
    rec: dict[str, Any] = {
        "type": "user",
        "isMeta": True,
        "uuid": "u-1",
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": f"injected rule mentioning ALPHA and {PROMPTS[0]}"},
                {"type": "text", "text": PROMPTS[0]},
            ],
        },
        "toolUseResult": {"stdout": "/Users/someone/secret", "note": "free text"},
        "gitBranch": "feature/private-thing",
    }
    out = scrub_claude(rec)
    first, second = out["message"]["content"]
    assert first["text"] == SCRUBBED
    assert second["text"] == PROMPTS[0]
    assert out["message"]["role"] == "user"
    assert out["uuid"] == "u-1"
    assert out["toolUseResult"] == {"stdout": SCRUBBED, "note": SCRUBBED}
    assert out["gitBranch"] == "main"


def test_codex_timezone_and_nested_fields() -> None:
    rec = {
        "type": "turn_context",
        "payload": {"timezone": "America/New_York", "nested": {"secret": "x"}, "turn_id": "t-1"},
    }
    out = scrub_codex(rec)
    assert out["payload"] == {"timezone": "UTC", "nested": {"secret": SCRUBBED}, "turn_id": "t-1"}


@pytest.mark.parametrize("scrub", [scrub_claude, scrub_codex])
def test_unknown_record_type_fails(scrub: Any) -> None:
    with pytest.raises(RecorderError, match="unrecognised record type"):
        scrub({"type": "brand-new-record"})


@pytest.mark.parametrize(
    "value",
    [
        "mail me at a@b.example",
        "/Users/x/y",
        "/home/someone/x",
        "/private/tmp/x",
        "/var/folders/ab",
    ],
)
def test_check_clean_rejects_leaks(value: str) -> None:
    with pytest.raises(RecorderError):
        check_clean("t", [{"type": "user", "x": value}])


def test_check_clean_rejects_base64_but_not_signature() -> None:
    blob = "A" * 40
    with pytest.raises(RecorderError, match="base64"):
        check_clean("t", [{"type": "user", "x": blob}])
    check_clean("t", [{"type": "assistant", "signature": blob}])


def test_check_clean_allows_placeholder_home() -> None:
    check_clean("t", [{"type": "user", "cwd": "/home/user/proj"}])


@pytest.mark.parametrize(
    ("name", "scrub"), [(CLAUDE_FIXTURE, scrub_claude), (CODEX_FIXTURE, scrub_codex)]
)
def test_scrubbing_committed_fixture_is_idempotent(name: str, scrub: Any, tmp_path: Path) -> None:
    out = tmp_path / name
    write_fixture(FIXTURES / name, out, scrub, lambda text: text)
    assert [json.loads(line) for line in out.read_text("utf-8").splitlines()] == fixture_lines(name)


@pytest.mark.parametrize(
    "head", [b"", b"\n", b"{not json\n", b"\xff\n", b"[1, 2]\n", b'{"payload": 3}\n']
)
def test_the_live_suites_fork_scan_skips_an_unreadable_rollout(tmp_path: Path, head: bytes) -> None:
    from test_live_handoff import _forked_from

    path = tmp_path / "rollout.jsonl"
    path.write_bytes(head)
    assert _forked_from(path) is None

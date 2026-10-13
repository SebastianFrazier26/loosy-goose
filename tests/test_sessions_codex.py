from pathlib import Path
from typing import Any

import pytest
from session_fixtures import (
    CODE_WORDS,
    CODEX_FIXTURE,
    FIXTURE_CWD,
    FIXTURES,
    SENTINEL,
    fixture_lines,
    install_codex,
    write_lines,
)

from loosy_goose.sessions import codex
from loosy_goose.sessions.model import SessionFormatError
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    MARKER_PREFIX,
    TRIMMED_RESULT,
    Trigger,
    exchange_starts,
    find_trigger,
    unit_ids,
)


def _read() -> codex.Session:
    return codex.read(FIXTURES / CODEX_FIXTURE)


def test_reads_the_fixture_session() -> None:
    s = _read()
    assert (s.tool, s.cwd) == ("codex", FIXTURE_CWD)
    assert s.session_id == fixture_lines(CODEX_FIXTURE)[0]["payload"]["id"]
    assert s.cli_version is not None
    prompts = s.prompts()
    assert len(prompts) == 5
    assert [w for w in CODE_WORDS for p in prompts if w in p] == list(CODE_WORDS)
    assert SENTINEL in prompts[-1]


def test_calls_pair_with_their_outputs() -> None:
    s = _read()
    calls = [ri for ri in s.live if s.records[ri].call_side == "call"]
    outputs = [ri for ri in s.live if s.records[ri].call_side == "result"]
    assert calls and outputs
    assert {c for ri in calls for c in s.records[ri].call_ids} == {
        c for ri in outputs for c in s.records[ri].call_ids
    }
    assert all(s.records[ri].tool_name for ri in calls)
    units = unit_ids(s)
    assert {units[ri] for ri in calls} == {units[ri] for ri in outputs}


def test_context_messages_are_pinned_and_prompts_are_messages() -> None:
    s = _read()
    developer = [r for r in s.records if r.kind == "response_item:message:developer"]
    prompts = [r for r in s.records if r.prompt is not None]
    assert developer and prompts
    assert all(r.pinned and r.prompt is None for r in developer)
    assert all(r.category == "message" and not r.pinned for r in prompts)


def test_a_prompt_and_its_user_message_echo_form_one_unit() -> None:
    s = _read()
    units = unit_ids(s)
    echoes = [ri for ri in s.live if s.records[ri].kind == "event_msg:user_message"]
    assert len(echoes) == len(s.prompts())
    for ri in echoes:
        mates = [rj for rj in s.live if rj != ri and units[rj] == units[ri]]
        assert [s.records[rj].prompt is not None for rj in mates] == [True]


def test_exchanges_follow_task_started_and_the_trigger_is_the_sentinel_turn() -> None:
    s = _read()
    started = sum(1 for r in s.records if r.kind == "event_msg:task_started")
    assert max(r.exchange for r in s.records) == started
    assert s.open_exchange is None
    assert find_trigger(s, DEFAULT_TRIGGERS["codex"]).how == "text"


def test_an_unclosed_last_turn_is_open(tmp_path: Path) -> None:
    lines = fixture_lines(CODEX_FIXTURE)
    last = max(
        i
        for i, o in enumerate(lines)
        if o["type"] == "event_msg" and o["payload"].get("type") == "task_complete"
    )
    s = codex.read(write_lines(tmp_path / "open.jsonl", lines[:last] + lines[last + 1 :]))
    assert s.open_exchange == max(r.exchange for r in s.records)
    position = exchange_starts(s)[s.open_exchange]
    assert find_trigger(s, ()) == Trigger(position, "open-turn")


def test_unknown_types_and_earlier_markers_are_classified(tmp_path: Path) -> None:
    lines = fixture_lines(CODEX_FIXTURE)
    stamp = lines[-1]["timestamp"]
    extra: list[dict[str, Any]] = [
        {"timestamp": stamp, "type": "brand_new", "payload": {}},
        {"timestamp": stamp, "type": "response_item", "payload": {"type": "hologram"}},
        {
            "timestamp": stamp,
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": f"{MARKER_PREFIX}x on d; ...]"}],
            },
        },
    ]
    s = codex.read(write_lines(tmp_path / "s.jsonl", lines + extra))
    assert [r.category for r in s.records[-3:]] == ["unknown", "unknown", "side"]
    assert s.records[-1].marker == f"{MARKER_PREFIX}x on d; ...]"
    assert s.records[-1].blocks == []


def test_placeholders_are_stripped_on_read(tmp_path: Path) -> None:
    lines = fixture_lines(CODEX_FIXTURE)
    out = next(
        o
        for o in lines
        if o["type"] == "response_item"
        and o["payload"].get("type") in ("function_call_output", "custom_tool_call_output")
    )
    out["payload"]["output"] = TRIMMED_RESULT
    s = codex.read(write_lines(tmp_path / "s.jsonl", lines))
    rec = next(r for r in s.records if r.call_side == "result")
    assert (rec.blocks, rec.cleaned) == ([], {0: ""})
    assert not any(TRIMMED_RESULT in b.text for r in s.records for b in r.blocks)


def test_locate_by_cwd_id_and_path(tmp_path: Path) -> None:
    home = tmp_path / "codex"
    path = install_codex(home)
    sid = fixture_lines(CODEX_FIXTURE)[0]["payload"]["id"]
    assert codex.locate(Path(FIXTURE_CWD), "current", home) == [path]
    assert codex.locate(tmp_path / "elsewhere", "current", home) == []
    assert codex.locate(tmp_path / "elsewhere", sid, home) == [path]
    assert codex.locate(tmp_path / "elsewhere", str(path), home) == [path]
    assert codex.locate(tmp_path / "elsewhere", "*", home) == []


def test_a_file_without_session_meta_is_refused(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "bad.jsonl", [{"type": "event_msg", "payload": {}}])
    with pytest.raises(SessionFormatError, match="session_meta"):
        codex.read(path)

import dataclasses
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from session_builders import kept_where
from session_fixtures import (
    CODE_WORDS,
    CODEX_FIXTURE,
    FIXTURE_CWD,
    FIXTURES,
    SENTINEL,
    fixture_lines,
    fork_with,
    install_codex,
    read_lines,
    write_lines,
)

from loosy_goose.sessions import codex
from loosy_goose.sessions.model import SessionFormatError
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    GAP,
    MARKER_PREFIX,
    TRIMMED_RESULT,
    Trigger,
    exchange_starts,
    find_trigger,
    plan_fork,
    split_region,
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


def _is_marker(o: dict[str, Any]) -> bool:
    return o.get("type") == "response_item" and MARKER_PREFIX in json.dumps(o, ensure_ascii=False)


def test_a_fork_without_beta_validates(tmp_path: Path) -> None:
    s, _, new_id, fork = fork_with(codex, tmp_path, keep=lambda seg: "BETA" not in seg.text)
    assert codex.validate(fork, new_id) == []
    out = read_lines(fork)
    assert out[0]["payload"]["id"] == new_id
    assert out[0]["payload"]["forked_from_id"] == s.session_id
    assert s.session_id not in json.dumps([o for o in out[1:] if not _is_marker(o)])
    assert f"compressed from session {s.session_id} " in json.dumps(out, ensure_ascii=False)
    f = codex.read(fork)
    text = " ".join(f.prompts())
    assert "BETA" not in text and SENTINEL not in text
    assert all(w in text for w in ("ALPHA", "GAMMA", "DELTA"))
    marker = next(i for i, o in enumerate(out) if _is_marker(o))
    assert all(o["type"] != "response_item" for o in out[:marker])
    assert "BETA" not in json.dumps(out)
    assert "BETA" not in fork.read_text(encoding="utf-8")


def _with_turn_context(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A context user message after every `turn_context`, which Codex may write (Facts)."""
    out: list[dict[str, Any]] = []
    for o in lines:
        out.append(o)
        if o["type"] == "turn_context":
            text = "<environment_context>scrubbed</environment_context>"
            out.append(
                {
                    "timestamp": o["timestamp"],
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
    return out


def test_a_dropped_prompt_takes_its_echo_along_even_with_per_turn_context(
    tmp_path: Path,
) -> None:
    lines = _with_turn_context(fixture_lines(CODEX_FIXTURE))
    _, _, new_id, fork = fork_with(codex, tmp_path, lines, keep=lambda seg: "BETA" not in seg.text)
    assert codex.validate(fork, new_id) == []
    assert "BETA" not in fork.read_text(encoding="utf-8")
    prompts = codex.read(fork).prompts()
    assert all(w in " ".join(prompts) for w in ("ALPHA", "GAMMA", "DELTA"))
    assert not any("environment_context" in p for p in prompts)


def test_ordinals_are_renumbered(tmp_path: Path) -> None:
    lines = [{**o, "ordinal": i} for i, o in enumerate(fixture_lines(CODEX_FIXTURE))]
    _, _, new_id, fork = fork_with(codex, tmp_path, lines, keep=lambda seg: "BETA" not in seg.text)
    out = read_lines(fork)
    assert [o.get("ordinal") for o in out if "ordinal" in o] == [
        i for i, o in enumerate(out) if "ordinal" in o
    ]
    assert codex.validate(fork, new_id) == []


def test_a_fork_of_a_fork_has_one_marker_and_no_inherited_placeholders(tmp_path: Path) -> None:
    _, _, first_id, first = fork_with(codex, tmp_path)
    lines = read_lines(first)
    out = next(
        o
        for o in lines
        if o["type"] == "response_item"
        and o["payload"].get("type") in ("function_call_output", "custom_tool_call_output")
    )
    out["payload"]["output"] = TRIMMED_RESULT
    s, _, new_id, second = fork_with(codex, tmp_path / "again", lines)
    assert s.session_id == first_id
    assert codex.validate(second, new_id) == []
    text = second.read_text(encoding="utf-8")
    assert text.count(MARKER_PREFIX) == 1
    assert f"compressed from session {first_id} " in text
    assert GAP not in text and TRIMMED_RESULT not in text


def _corrupt(lines: list[dict[str, Any]], how: str, old: str) -> list[dict[str, Any]]:
    if how == "tool-pairs":
        out = next(
            i
            for i, o in enumerate(lines)
            if o["type"] == "response_item"
            and o["payload"].get("type") in ("function_call_output", "custom_tool_call_output")
        )
        del lines[out]
    elif how == "stale-id":
        lines[-1] = {**lines[-1], "note": old}
    elif how == "session-id":
        lines[0]["payload"]["id"] = "someone-else"
    elif how == "turns":
        # The first turn's end precedes every start, and an end with no start before it is
        # excused by design (a compaction can cut a turn's start); so drop the second start.
        starts = [
            i
            for i, o in enumerate(lines)
            if o["type"] == "event_msg" and o["payload"].get("type") == "task_started"
        ]
        del lines[starts[1]]
    elif how == "ordinals":
        lines = [{**o, "ordinal": i} for i, o in enumerate(lines)]
        del lines[2]
    elif how == "empty-content":
        msg = next(
            o
            for o in lines
            if o["type"] == "response_item"
            and o["payload"].get("type") == "message"
            and o["payload"].get("role") == "assistant"
        )
        msg["payload"]["content"] = []
    return lines


@pytest.mark.parametrize(
    "how", ["tool-pairs", "stale-id", "session-id", "turns", "ordinals", "empty-content"]
)
def test_validate_names_the_failed_check(tmp_path: Path, how: str) -> None:
    s, _, new_id, fork = fork_with(codex, tmp_path)
    bad = write_lines(tmp_path / "bad.jsonl", _corrupt(read_lines(fork), how, s.session_id))
    assert how in codex.validate(bad, new_id)


def _compacted_lines() -> list[dict[str, Any]]:
    """The fixture as if Codex had compacted its context just before the GAMMA turn."""
    lines = fixture_lines(CODEX_FIXTURE)
    echo = codex.read(FIXTURES / CODEX_FIXTURE)
    gamma = next(
        r.line
        for r in echo.records
        if r.kind == "event_msg:user_message"
        and "GAMMA" in (codex._echo_text(r.raw["payload"]) or "")
    )
    start = max(
        i
        for i in range(gamma)
        if lines[i]["type"] == "event_msg" and lines[i]["payload"].get("type") == "task_started"
    )
    compacted = {
        "timestamp": lines[start]["timestamp"],
        "type": "compacted",
        "payload": {
            "message": "",
            "replacement_history": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Earlier: two code words."}],
                }
            ],
        },
    }
    return lines[:start] + [compacted] + lines[start:]


def test_rf4_a_session_spanning_context_windows_forks_from_its_last_compaction(
    tmp_path: Path,
) -> None:
    lines = _compacted_lines()
    s = codex.read(write_lines(tmp_path / "compacted.jsonl", lines))
    first_live = s.records[s.live[1]]
    assert first_live.kind == "compacted" and first_live.pinned
    assert len(s.prompts()) == 3 and not any("ALPHA" in p or "BETA" in p for p in s.prompts())
    _, _, new_id, fork = fork_with(codex, tmp_path, lines)
    assert codex.validate(fork, new_id) == []
    out = read_lines(fork)
    compacted = next(i for i, o in enumerate(out) if o["type"] == "compacted")
    marker = next(i for i, o in enumerate(out) if _is_marker(o))
    assert compacted < marker
    assert all(o["type"] != "response_item" for o in out[compacted + 1 : marker])
    assert "ALPHA" not in json.dumps(out) and "BETA" not in json.dumps(out)


def test_destination_ids_and_commands() -> None:
    when = datetime(2026, 10, 13, 9, 8, 7)
    assert codex.fork_destination(Path("/x.jsonl"), "abc", Path("/h"), when) == Path(
        "/h/sessions/2026/10/13/rollout-2026-10-13T09-08-07-abc.jsonl"
    )
    assert uuid.UUID(codex.new_session_id()).version == 7
    assert codex.resume_command("abc", "/work/my dir") == "cd '/work/my dir' && codex resume abc"
    assert codex.verify_command("abc") == [
        "codex",
        "exec",
        "resume",
        "abc",
        "Reply with exactly: OK",
    ]


def test_writing_onto_an_existing_destination_fails_and_leaves_it_alone(tmp_path: Path) -> None:
    s, plan, new_id, fork = fork_with(codex, tmp_path)
    before = fork.read_bytes()
    with pytest.raises(FileExistsError):
        codex.write_fork(s, plan, new_id, fork.parent)
    assert fork.read_bytes() == before


def test_a_message_emptied_by_a_rewrite_is_not_written(tmp_path: Path) -> None:
    s, plan, _, _ = fork_with(codex, tmp_path)
    target = next(
        ri
        for ri in plan.keep
        if s.records[ri].kind == "response_item:message:assistant"
        and len(s.records[ri].raw["payload"]["content"]) == 1
    )
    ident = s.records[target].raw["payload"]["id"]
    emptied = dataclasses.replace(plan, rewrites={**plan.rewrites, target: {0: ""}})
    new_id = codex.new_session_id()
    fork = codex.write_fork(s, emptied, new_id, tmp_path / "emptied")
    out = read_lines(fork)
    assert not [o for o in out if o["type"] == "response_item" and o["payload"].get("id") == ident]
    assert codex.validate(fork, new_id) == []


def test_an_output_rewritten_to_empty_keeps_its_call_paired(tmp_path: Path) -> None:
    s, plan, _, _ = fork_with(codex, tmp_path)
    target = next(ri for ri in plan.keep if s.records[ri].call_side == "result")
    emptied = dataclasses.replace(plan, rewrites={**plan.rewrites, target: {0: ""}})
    new_id = codex.new_session_id()
    fork = codex.write_fork(s, emptied, new_id, tmp_path / "emptied")
    assert codex.validate(fork, new_id) == []
    assert any(
        o["payload"].get("call_id") in s.records[target].call_ids
        for o in read_lines(fork)
        if o["type"] == "response_item" and o["payload"].get("type", "").endswith("_output")
    )


def test_a_tail_starting_mid_turn_keeps_that_turns_start(tmp_path: Path) -> None:
    s = _read()
    trigger = find_trigger(s, DEFAULT_TRIGGERS["codex"])
    assert trigger.position is not None
    starts = exchange_starts(s)
    last = s.records[s.live[trigger.position - 1]].exchange
    short = sum(s.tokens(ri) for ri in s.live[starts[last] : trigger.position])
    region = split_region(s, trigger, protect_tail_tokens=short + 1)
    cut = s.records[s.live[region.tail_from]].exchange
    assert cut == last - 1 and region.tail_from > starts[cut]
    plan = plan_fork(s, region, kept_where(region, lambda g: "ALPHA" in g.text), date="2026-10-13")
    new_id = codex.new_session_id()
    fork = codex.write_fork(s, plan, new_id, tmp_path / "out")
    assert codex.validate(fork, new_id) == []
    turn = s.records[s.live[starts[cut]]].raw["payload"]["turn_id"]
    started = [
        o["payload"].get("turn_id")
        for o in read_lines(fork)
        if o["type"] == "event_msg" and o["payload"].get("type") == "task_started"
    ]
    assert turn in started


def test_an_interrupted_call_with_no_output_is_dropped_and_the_fork_validates(
    tmp_path: Path,
) -> None:
    lines = fixture_lines(CODEX_FIXTURE)
    call = next(o for o in lines if o["payload"].get("type") == "custom_tool_call")
    orphan = json.loads(json.dumps(call))
    orphan["payload"]["call_id"] = "call_interrupted"
    lines.insert(lines.index(call), orphan)
    for protect in (0, 10_000):
        s = codex.read(write_lines(tmp_path / "src.jsonl", lines))
        region = split_region(s, find_trigger(s, DEFAULT_TRIGGERS["codex"]), protect)
        plan = plan_fork(s, region, kept_where(region, lambda g: True), date="2026-10-13")
        new_id = codex.new_session_id()
        fork = codex.write_fork(s, plan, new_id, tmp_path / f"out{protect}")
        assert codex.validate(fork, new_id) == []
        text = fork.read_text(encoding="utf-8")
        assert "call_interrupted" not in text and call["payload"]["call_id"] in text

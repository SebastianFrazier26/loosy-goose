import dataclasses
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from session_fixtures import (
    CLAUDE_FIXTURE,
    CODE_WORDS,
    FIXTURE_CWD,
    FIXTURES,
    fixture_lines,
    fork_with,
    install_claude,
    read_lines,
    write_lines,
)

from loosy_goose.sessions import claude
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    GAP,
    MARKER_PREFIX,
    TRIMMED_RESULT,
    find_trigger,
    unit_ids,
)


def _read() -> claude.Session:
    return claude.read(FIXTURES / CLAUDE_FIXTURE)


def test_reads_the_fixture_session() -> None:
    s = _read()
    assert (s.tool, s.cwd) == ("claude", FIXTURE_CWD)
    assert len(s.session_id) == 36 and s.cli_version is not None
    prompts = s.prompts()
    assert len(prompts) == 5
    assert [w for w in CODE_WORDS for p in prompts if w in p] == list(CODE_WORDS)
    assert "<command-name>/loosy</command-name>" in prompts[-1]


def test_the_echo_call_and_its_result_form_one_unit() -> None:
    s = _read()
    calls = [ri for ri in s.live if s.records[ri].call_side == "call"]
    results = [ri for ri in s.live if s.records[ri].call_side == "result"]
    assert calls and results
    assert {c for ri in calls for c in s.records[ri].call_ids} == {
        c for ri in results for c in s.records[ri].call_ids
    }
    assert all(s.records[ri].tool_name == "Bash" for ri in calls)
    units = unit_ids(s)
    assert {units[ri] for ri in calls} == {units[ri] for ri in results}


def test_the_trigger_is_the_recorded_slash_command() -> None:
    s = _read()
    t = find_trigger(s, DEFAULT_TRIGGERS["claude"])
    assert t.how == "text" and t.position is not None
    assert "/loosy" in (s.records[s.live[t.position]].prompt or "")


def test_meta_and_startup_records_are_not_prompts() -> None:
    s = _read()
    live = [s.records[ri] for ri in s.live]
    metas = [r for r in live if r.raw.get("isMeta")]
    startup = [r for r in live if r.kind == "attachment" and r.exchange == 0]
    assert metas and startup
    assert all(r.category == "meta" and r.prompt is None for r in metas)
    assert all(r.pinned for r in startup)


def test_side_unknown_and_marker_records_are_classified(tmp_path: Path) -> None:
    lines = fixture_lines(CLAUDE_FIXTURE)
    sid = next(str(o["sessionId"]) for o in lines if "sessionId" in o)
    first = next(o for o in lines if isinstance(o.get("uuid"), str))
    marker = {
        "type": "user",
        "uuid": "11111111-1111-4111-8111-111111111111",
        "parentUuid": None,
        "sessionId": sid,
        "message": {"role": "user", "content": f"{MARKER_PREFIX}x on 2026-10-13; ...]"},
    }
    first["parentUuid"] = marker["uuid"]
    extra = [
        {"type": "queue-operation", "sessionId": sid},
        {"type": "brand-new-thing", "sessionId": sid},
    ]
    s = claude.read(write_lines(tmp_path / "s.jsonl", [marker, *lines, *extra]))
    kinds = {r.kind: r.category for r in s.records}
    assert kinds["queue-operation"] == "side"
    assert kinds["brand-new-thing"] == "unknown"
    assert kinds["user:loosy-marker"] == "side"
    old = next(r for r in s.records if r.kind == "user:loosy-marker")
    assert (old.marker, old.blocks) == (marker["message"]["content"], [])
    assert len(s.records) - 1 in s.loose


def test_placeholders_are_stripped_on_read(tmp_path: Path) -> None:
    lines = fixture_lines(CLAUDE_FIXTURE)
    result = next(
        o
        for o in lines
        if isinstance(o.get("message", {}).get("content"), list)
        and any(p.get("type") == "tool_result" for p in o["message"]["content"])
    )
    part = next(
        i for i, p in enumerate(result["message"]["content"]) if p.get("type") == "tool_result"
    )
    result["message"]["content"][part]["content"] = TRIMMED_RESULT
    reply = next(
        o
        for o in lines
        if o.get("type") == "assistant"
        and any(p.get("type") == "text" for p in o["message"]["content"])
    )
    text_part = next(
        i for i, p in enumerate(reply["message"]["content"]) if p.get("type") == "text"
    )
    reply["message"]["content"][text_part]["text"] += f"\n{GAP}"
    s = claude.read(write_lines(tmp_path / "s.jsonl", lines))
    rec = next(r for r in s.records if r.raw.get("uuid") == result["uuid"])
    assert (rec.call_side, rec.blocks, rec.cleaned) == ("result", [], {part: ""})
    edited = next(r for r in s.records if r.raw.get("uuid") == reply["uuid"])
    assert text_part in edited.cleaned
    texts = [b.text for r in s.records for b in r.blocks]
    assert not any(GAP in t or TRIMMED_RESULT in t for t in texts)


def test_a_missing_parent_truncates_history_with_a_warning(tmp_path: Path) -> None:
    lines = fixture_lines(CLAUDE_FIXTURE)
    chain = [o for o in lines if isinstance(o.get("uuid"), str)]
    victim = chain[len(chain) // 2]
    victim["parentUuid"] = "00000000-0000-4000-8000-000000000000"
    s = claude.read(write_lines(tmp_path / "s.jsonl", lines))
    assert any("not in the file" in w for w in s.warnings)
    assert s.records[s.live[0]].raw["uuid"] == victim["uuid"]


def test_locate_finds_the_cwd_sessions_newest_first(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    older = install_claude(home, session_id="11111111-1111-4111-8111-111111111111")
    newer = install_claude(home, session_id="22222222-2222-4222-8222-222222222222")
    os.utime(older, (1, 1))
    assert claude.locate(Path(FIXTURE_CWD), "current", home) == [newer, older]
    assert claude.locate(tmp_path / "elsewhere", "current", home) == []


def test_locate_falls_back_to_the_recorded_cwd_when_the_slug_differs(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    path = install_claude(home, project_dir="-some-other-slug")
    assert claude.locate(Path(FIXTURE_CWD), "current", home) == [path]


def test_locate_by_explicit_id_or_path_and_rejects_glob_input(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    sid = "33333333-3333-4333-8333-333333333333"
    path = install_claude(home, session_id=sid)
    assert claude.locate(tmp_path / "anywhere", sid, home) == [path]
    assert claude.locate(tmp_path / "anywhere", str(path), home) == [path]
    assert claude.locate(tmp_path / "anywhere", "*", home) == []


def test_a_fork_without_beta_validates_and_relinks(tmp_path: Path) -> None:
    _, _, new_id, fork = fork_with(claude, tmp_path, keep=lambda seg: "BETA" not in seg.text)
    assert fork.name == f"{new_id}.jsonl"
    assert claude.validate(fork, new_id) == []
    f = claude.read(fork)
    assert f.session_id == new_id
    assert f.records[f.live[0]].kind == "user:loosy-marker"
    text = " ".join(f.prompts())
    assert "BETA" not in text
    assert all(w in text for w in ("ALPHA", "GAMMA", "DELTA"))
    assert "/loosy" not in text


def test_kept_records_keep_their_ids_and_the_file_ends_on_the_leaf(tmp_path: Path) -> None:
    s, _, new_id, fork = fork_with(claude, tmp_path)
    original = {r.raw["uuid"] for r in s.records if isinstance(r.raw.get("uuid"), str)}
    out = read_lines(fork)
    chain = [o for o in out if isinstance(o.get("uuid"), str)]
    assert {o["uuid"] for o in chain[1:]} <= original
    assert all(o.get("sessionId") == new_id for o in out if "sessionId" in o)
    assert all(o.get("session_id") == new_id for o in out if "session_id" in o)
    assert out[-1]["type"] == "last-prompt"
    assert out[-1]["leafUuid"] == chain[-1]["uuid"]
    assert "DELTA" in out[-1]["lastPrompt"]


def test_a_rewritten_tool_result_is_written(tmp_path: Path) -> None:
    s, plan, new_id, _ = fork_with(claude, tmp_path)
    ri = next(ri for ri in plan.keep if s.records[ri].call_side == "result")
    part = int(s.records[ri].blocks[0].meta["part"])
    trimmed = dataclasses.replace(plan, rewrites={ri: {part: "trimmed"}})
    fork = claude.write_fork(s, trimmed, new_id, tmp_path / "trimmed")
    assert claude.validate(fork, new_id) == []
    f = claude.read(fork)
    rec = next(r for r in f.records if r.raw.get("uuid") == s.records[ri].raw["uuid"])
    assert [b.text for b in rec.blocks] == ["trimmed"]


def test_an_emptied_tool_result_stays_paired_with_its_call(tmp_path: Path) -> None:
    s, plan, new_id, _ = fork_with(claude, tmp_path)
    ri = next(ri for ri in plan.keep if s.records[ri].call_side == "result")
    part = int(s.records[ri].blocks[0].meta["part"])
    emptied = dataclasses.replace(plan, rewrites={ri: {part: ""}})
    fork = claude.write_fork(s, emptied, new_id, tmp_path / "emptied")
    assert claude.validate(fork, new_id) == []
    out = next(o for o in read_lines(fork) if o.get("uuid") == s.records[ri].raw["uuid"])
    assert out["message"]["content"][part]["content"] == ""


def _corrupt(lines: list[dict[str, Any]], how: str) -> list[dict[str, Any]]:
    chain = [o for o in lines if isinstance(o.get("uuid"), str)]
    if how == "chain":
        chain[len(chain) // 2]["parentUuid"] = "00000000-0000-4000-8000-000000000000"
    elif how == "tool-pairs":
        result = next(
            o
            for o in chain
            if isinstance(o.get("message", {}).get("content"), list)
            and any(p.get("type") == "tool_result" for p in o["message"]["content"])
        )
        result["message"]["content"] = [{"type": "text", "text": "gone"}]
    elif how == "session-id":
        chain[-1]["sessionId"] = "someone-else"
    elif how == "coverage":
        lines.insert(
            0,
            {
                "type": "user",
                "uuid": str(uuid.uuid4()),
                "parentUuid": None,
                "sessionId": chain[0]["sessionId"],
                "message": {"role": "user", "content": "stray"},
            },
        )
    return lines


@pytest.mark.parametrize("how", ["chain", "tool-pairs", "session-id", "coverage"])
def test_validate_names_the_failed_check(tmp_path: Path, how: str) -> None:
    _, _, new_id, fork = fork_with(claude, tmp_path)
    bad = write_lines(tmp_path / "bad.jsonl", _corrupt(read_lines(fork), how))
    assert how in claude.validate(bad, new_id)


def _compacted_lines() -> list[dict[str, Any]]:
    """The fixture as if built-in /compact had run just before the BETA turn."""
    lines = fixture_lines(CLAUDE_FIXTURE)
    sid = next(str(o["sessionId"]) for o in lines if "sessionId" in o)
    beta = next(
        i
        for i, o in enumerate(lines)
        if o.get("type") == "user"
        and not o.get("isMeta")
        and "toolUseResult" not in o
        and "BETA" in json.dumps(o.get("message"))
    )
    boundary_id, summary_id = str(uuid.uuid4()), str(uuid.uuid4())
    boundary = {
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": boundary_id,
        "parentUuid": None,
        "logicalParentUuid": lines[beta]["parentUuid"],
        "sessionId": sid,
        "isSidechain": False,
        "cwd": FIXTURE_CWD,
        "content": "Conversation compacted",
        "compactMetadata": {"trigger": "manual", "preTokens": 1000},
    }
    summary = {
        "type": "user",
        "uuid": summary_id,
        "parentUuid": boundary_id,
        "sessionId": sid,
        "isSidechain": False,
        "isCompactSummary": True,
        "isVisibleInTranscriptOnly": True,
        "cwd": FIXTURE_CWD,
        "message": {"role": "user", "content": "Summary: the user shared one code word earlier."},
    }
    lines[beta]["parentUuid"] = summary_id
    return lines[:beta] + [boundary, summary] + lines[beta:]


def test_rf1_an_already_compacted_session_forks_from_its_boundary(tmp_path: Path) -> None:
    lines = _compacted_lines()
    s = claude.read(write_lines(tmp_path / "compacted.jsonl", lines))
    assert s.records[s.live[0]].kind == "system:compact_boundary"
    assert s.records[s.live[1]].pinned and s.records[s.live[1]].kind == "user:compact_summary"
    assert not any("ALPHA" in p for p in s.prompts())
    _, _, new_id, fork = fork_with(claude, tmp_path, lines)
    assert claude.validate(fork, new_id) == []
    f = claude.read(fork)
    kinds = [f.records[ri].kind for ri in f.live[:3]]
    assert kinds == ["system:compact_boundary", "user:loosy-marker", "user:compact_summary"]
    assert "ALPHA" not in fork.read_text(encoding="utf-8")


def test_destination_and_commands() -> None:
    src = Path("/h/projects/-work/old.jsonl")
    assert claude.fork_destination(src, "new", Path("/h"), datetime(2026, 10, 13)) == (
        Path("/h/projects/-work/new.jsonl")
    )
    assert (
        claude.resume_command("abc", "/work/my dir") == "cd '/work/my dir' && claude --resume abc"
    )
    assert claude.verify_command("abc") == [
        "claude",
        "-p",
        "Reply with exactly: OK",
        "--resume",
        "abc",
    ]
    assert uuid.UUID(claude.new_session_id()).version == 4

import json
import os
import re
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from session_fixtures import (
    CLAUDE_FIXTURE,
    FIXTURE_CWD,
    fake_embeddings,
    fixture_lines,
    install_claude,
    install_codex,
    without_trigger,
    write_lines,
)

from loosy_goose import handoff as handoff_mod
from loosy_goose.handoff import HandoffError, HandoffOptions, handoff
from loosy_goose.pipeline import Compressed, PipelineOptions
from loosy_goose.segment import segment
from loosy_goose.sessions import claude, codex
from loosy_goose.sessions.plan import GAP, MARKER_PREFIX, TRIMMED_RESULT
from loosy_goose.transcript import Transcript

NOW = datetime(2026, 10, 13, 12, 0, 0)
RECOMPRESSED = re.compile(
    r"this session was already compressed by loosy-goose on 2026-10-13; about [\d,]+ tokens "
    r"added since — compressing again compounds the loss"
)


@pytest.fixture(autouse=True)
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embeddings)


def _opts(tmp_path: Path, **overrides: Any) -> HandoffOptions:
    values: dict[str, Any] = {
        "tool": "claude",
        "cwd": Path(FIXTURE_CWD),
        "protect_tail_tokens": 0,
        "level": "heavy",
        "homes": {"claude": tmp_path / "claude", "codex": tmp_path / "codex"},
    }
    values.update(overrides)
    return HandoffOptions(**values)


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def test_forks_the_claude_fixture(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    result = handoff(_opts(tmp_path), now=NOW)
    assert (result.status, result.tool, result.trigger) == ("forked", "claude", "text")
    assert result.fork_session is not None and result.fork_path is not None
    fork = Path(result.fork_path)
    assert fork == source.parent / f"{result.fork_session}.jsonl"
    assert claude.validate(fork, result.fork_session) == []
    assert result.resume_command == f"cd {FIXTURE_CWD} && claude --resume {result.fork_session}"
    assert result.original_resume_command == (
        f"cd {FIXTURE_CWD} && claude --resume {result.source_session}"
    )
    assert "ALPHA" in result.first_prompt and "DELTA" in result.last_prompt
    assert min(result.before_tokens, result.after_tokens, result.tail_tokens) > 0
    assert result.keep == 0.25 and result.verified is None
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert _names(source.parent) == sorted([source.name, fork.name])
    assert json.loads(json.dumps(result.to_dict()))["fork_session"] == result.fork_session


def test_forks_the_codex_fixture(tmp_path: Path) -> None:
    source = install_codex(tmp_path / "codex")
    result = handoff(_opts(tmp_path, tool="codex"), now=NOW)
    assert (result.status, result.tool) == ("forked", "codex")
    assert result.fork_path == str(
        tmp_path
        / "codex"
        / "sessions"
        / "2026"
        / "10"
        / "13"
        / f"rollout-2026-10-13T12-00-00-{result.fork_session}.jsonl"
    )
    assert codex.validate(Path(result.fork_path), str(result.fork_session)) == []
    assert result.resume_command == f"cd {FIXTURE_CWD} && codex resume {result.fork_session}"
    assert source.exists()


def test_auto_picks_the_newest_session_across_tools(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude")
    old = install_codex(tmp_path / "codex")
    os.utime(old, (1, 1))
    assert handoff(_opts(tmp_path, tool="auto"), now=NOW).tool == "claude"


def test_nothing_to_do_when_the_session_is_all_tail(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")
    result = handoff(_opts(tmp_path, protect_tail_tokens=4000), now=NOW)
    assert result.status == "nothing-to-do" and result.fork_path is None
    assert result.reason is not None and "protected tail" in result.reason
    assert _names(source.parent) == [source.name]


def test_nothing_to_do_when_already_under_target(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude")
    result = handoff(_opts(tmp_path, level=None, target_tokens=10**6), now=NOW)
    assert result.status == "nothing-to-do"
    assert result.reason is not None and "under target" in result.reason


def test_a_target_below_what_must_stay_is_an_error(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude")
    with pytest.raises(HandoffError, match="leaves no room"):
        handoff(_opts(tmp_path, level=None, target_tokens=1), now=NOW)


def test_no_session_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(HandoffError, match="no Claude Code session found"):
        handoff(_opts(tmp_path), now=NOW)


def test_a_failed_validation_deletes_the_fork_and_names_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = install_claude(tmp_path / "claude")
    monkeypatch.setattr(claude, "validate", lambda fork, sid: ["chain"])
    with pytest.raises(HandoffError, match=r"validation \(chain\)"):
        handoff(_opts(tmp_path), now=NOW)
    assert _names(source.parent) == [source.name]


def test_an_unknown_record_inside_the_compressed_stretch_refuses(tmp_path: Path) -> None:
    lines = fixture_lines(CLAUDE_FIXTURE)
    alpha = next(
        i
        for i, o in enumerate(lines)
        if o.get("type") == "user" and "ALPHA" in json.dumps(o.get("message"))
    )
    alpha_id = lines[alpha]["uuid"]
    child = next(o for o in lines if o.get("parentUuid") == alpha_id)
    stranger = {
        "type": "brand-new-thing",
        "uuid": str(uuid.uuid4()),
        "parentUuid": alpha_id,
        "sessionId": lines[alpha]["sessionId"],
    }
    child["parentUuid"] = stranger["uuid"]
    source = install_claude(
        tmp_path / "claude", lines=[*lines[: alpha + 1], stranger, *lines[alpha + 1 :]]
    )
    with pytest.raises(HandoffError, match="brand-new-thing"):
        handoff(_opts(tmp_path), now=NOW)
    assert _names(source.parent) == [source.name]


def test_a_newer_cli_than_tested_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_claude(tmp_path / "claude")
    monkeypatch.setattr(claude, "TESTED_VERSIONS", ("0.0.1",))
    result = handoff(_opts(tmp_path), now=NOW)
    assert any("newer than the tested 0.0.1" in w for w in result.warnings)


def test_verify_runs_a_throwaway_copy_and_deletes_it(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")
    calls: list[list[str]] = []

    def runner(argv: list[str], cwd: Path) -> tuple[int, str]:
        calls.append(argv)
        assert (source.parent / f"{argv[-1]}.jsonl").exists()
        return 0, " OK\n"

    result = handoff(_opts(tmp_path, verify=True), now=NOW, runner=runner)
    assert result.verified is True
    assert calls[0][:4] == ["claude", "-p", "Reply with exactly: OK", "--resume"]
    assert calls[0][-1] != result.fork_session
    assert _names(source.parent) == sorted([source.name, f"{result.fork_session}.jsonl"])


def test_a_failed_live_check_deletes_the_fork(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")
    with pytest.raises(HandoffError, match="live check failed"):
        handoff(_opts(tmp_path, verify=True), now=NOW, runner=lambda argv, cwd: (0, "nope"))
    assert _names(source.parent) == [source.name]


@pytest.mark.parametrize(
    ("code", "reply"),
    [(0, "TOKEN"), (0, "BOOK"), (0, "Error: LOOKUP failed"), (0, "ok"), (1, "OK"), (2, "OK\n")],
)
def test_only_exit_zero_and_exactly_ok_pass_the_live_check(
    tmp_path: Path, code: int, reply: str
) -> None:
    source = install_claude(tmp_path / "claude")
    with pytest.raises(HandoffError, match="live check failed"):
        handoff(_opts(tmp_path, verify=True), now=NOW, runner=lambda argv, cwd: (code, reply))
    assert _names(source.parent) == [source.name]


def test_rf2_a_partial_last_line_is_ignored_and_the_original_untouched(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")
    with source.open("ab") as fh:
        fh.write(b'{"type":"user","uuid":"half')
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    result = handoff(_opts(tmp_path), now=NOW)
    assert result.status == "forked"
    assert any("partial last line (27 bytes)" in w for w in result.warnings)
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


def test_rf3_the_session_carrying_the_trigger_beats_a_newer_one(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    with_trigger = install_claude(home)
    install_claude(
        home,
        lines=without_trigger(fixture_lines(CLAUDE_FIXTURE)),
        session_id="44444444-4444-4444-8444-444444444444",
    )
    os.utime(with_trigger, (time.time() - 30, time.time() - 30))
    result = handoff(_opts(tmp_path), now=NOW)
    assert result.source_path == str(with_trigger)
    assert result.trigger == "text"
    assert any("2 sessions in" in w for w in result.warnings)
    assert "ALPHA" in result.first_prompt and "DELTA" in result.last_prompt


def test_rf5_an_unidentifiable_trigger_excludes_nothing_and_says_so(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude", lines=without_trigger(fixture_lines(CLAUDE_FIXTURE)))
    result = handoff(_opts(tmp_path), now=NOW)
    assert (result.status, result.trigger) == ("forked", "none")
    assert any("handoff trigger not found" in w for w in result.warnings)
    fork = claude.read(Path(str(result.fork_path)))
    assert "DELTA" in fork.prompts()[-1]


def _keep_everything(
    transcript: Transcript, opts: PipelineOptions, *, cache_dir: Path | None = None
) -> Compressed:
    return Compressed(segment(transcript), None, 0, 0, 0, 1.0)


def _parts(obj: dict[str, Any]) -> list[dict[str, Any]]:
    message = obj.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [p for p in content if isinstance(p, dict)] if isinstance(content, list) else []


def test_a_second_handoff_on_a_fork_warns_and_inherits_no_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Keeping every segment makes this run trim nothing, so any placeholder in the second fork
    # would be one inherited from the first.
    monkeypatch.setattr("loosy_goose.handoff.compress_transcript", _keep_everything)
    install_claude(tmp_path / "claude")
    first = handoff(_opts(tmp_path), now=NOW)
    earlier = Path(str(first.fork_path))
    lines = [json.loads(line) for line in earlier.read_text(encoding="utf-8").splitlines()]
    for obj in lines:
        for part in _parts(obj):
            if part.get("type") == "tool_result":
                part["content"] = TRIMMED_RESULT
    reply = next(
        p
        for o in reversed(lines)
        if o.get("type") == "assistant"
        for p in _parts(o)
        if p.get("type") == "text"
    )
    reply["text"] += f"\n{GAP}"
    write_lines(earlier, lines)
    assert first.status == "forked" and not any(RECOMPRESSED.fullmatch(w) for w in first.warnings)

    second = handoff(_opts(tmp_path, session=str(first.fork_session)), now=NOW)
    assert (second.status, second.source_session) == ("forked", first.fork_session)
    assert any(RECOMPRESSED.fullmatch(w) for w in second.warnings)
    fork = Path(str(second.fork_path))
    assert claude.validate(fork, str(second.fork_session)) == []
    text = fork.read_text(encoding="utf-8")
    assert text.count(MARKER_PREFIX) == 1
    assert f"compressed from session {first.fork_session} " in text
    assert GAP not in text and TRIMMED_RESULT not in text
    parts = [p for o in map(json.loads, text.splitlines()) for p in _parts(o)]
    results = [p["content"] for p in parts if p.get("type") == "tool_result"]
    assert results and set(results) == {""}


def test_the_move_never_clobbers_an_existing_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = install_claude(tmp_path / "claude")
    taken = source.parent / "55555555-5555-4555-8555-555555555555.jsonl"
    taken.write_bytes(b"precious\n")
    monkeypatch.setattr(claude, "new_session_id", lambda: taken.stem)
    with pytest.raises(HandoffError, match="already exists"):
        handoff(_opts(tmp_path), now=NOW)
    assert taken.read_bytes() == b"precious\n"
    assert _names(source.parent) == sorted([source.name, taken.name])


def test_any_failure_after_staging_leaves_no_fork_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = install_claude(tmp_path / "claude")

    def boom(fork: Path, sid: str) -> list[str]:
        raise RuntimeError("validator crashed")

    monkeypatch.setattr(claude, "validate", boom)
    with pytest.raises(RuntimeError, match="validator crashed"):
        handoff(_opts(tmp_path), now=NOW)
    assert _names(source.parent) == [source.name]


def test_a_runner_crash_during_verify_deletes_the_fork_and_the_copy(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")

    def runner(argv: list[str], cwd: Path) -> tuple[int, str]:
        raise RuntimeError("runner crashed")

    with pytest.raises(RuntimeError, match="runner crashed"):
        handoff(_opts(tmp_path, verify=True), now=NOW, runner=runner)
    assert _names(source.parent) == [source.name]


def test_a_write_fork_failure_partway_leaves_no_fork_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = install_claude(tmp_path / "claude")
    real = claude.write_fork

    def partial(*args: Any, **kwargs: Any) -> Path:
        path = real(*args, **kwargs)
        path.write_bytes(path.read_bytes()[:10])
        raise OSError("disk full")

    monkeypatch.setattr(claude, "write_fork", partial)
    with pytest.raises(OSError, match="disk full"):
        handoff(_opts(tmp_path), now=NOW)
    assert _names(source.parent) == [source.name]


def _codex_runner(reply: str | None, stdout: str, calls: list[list[str]], code: int = 0) -> Any:
    def runner(argv: list[str], cwd: Path) -> tuple[int, str]:
        calls.append(argv)
        if reply is not None:
            Path(argv[argv.index("--output-last-message") + 1]).write_text(reply)
        return code, stdout

    return runner


def test_codex_verify_reads_the_reply_file_and_ignores_stdout_noise(tmp_path: Path) -> None:
    install_codex(tmp_path / "codex")
    calls: list[list[str]] = []
    runner = _codex_runner("OK\n", "OpenAI Codex v0.158.0\nworkdir: /x\nOK\ntokens used: 9", calls)
    result = handoff(_opts(tmp_path, tool="codex", verify=True), now=NOW, runner=runner)
    assert result.verified is True
    argv = calls[0]
    assert argv[:3] == ["codex", "exec", "resume"]
    assert "--skip-git-repo-check" in argv
    reply_file = Path(argv[argv.index("--output-last-message") + 1])
    assert not reply_file.exists()


@pytest.mark.parametrize(
    ("reply", "stdout", "code"),
    [("nope", "OK", 0), (None, "OK", 0), ("OK", "OK", 1), ("", "OK", 0)],
)
def test_codex_verify_fails_on_the_file_not_stdout(
    tmp_path: Path, reply: str | None, stdout: str, code: int
) -> None:
    install_codex(tmp_path / "codex")
    calls: list[list[str]] = []
    runner = _codex_runner(reply, stdout, calls, code)
    with pytest.raises(HandoffError, match="live check failed"):
        handoff(_opts(tmp_path, tool="codex", verify=True), now=NOW, runner=runner)
    assert not Path(calls[0][calls[0].index("--output-last-message") + 1]).exists()


def test_codex_reply_file_is_removed_when_the_runner_crashes(tmp_path: Path) -> None:
    install_codex(tmp_path / "codex")
    seen: list[Path] = []

    def runner(argv: list[str], cwd: Path) -> tuple[int, str]:
        path = Path(argv[argv.index("--output-last-message") + 1])
        path.write_text("OK")
        seen.append(path)
        raise RuntimeError("runner crashed")

    with pytest.raises(RuntimeError, match="runner crashed"):
        handoff(_opts(tmp_path, tool="codex", verify=True), now=NOW, runner=runner)
    assert not seen[0].exists()


def test_claude_verify_still_reads_stdout(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude")
    calls: list[list[str]] = []

    def runner(argv: list[str], cwd: Path) -> tuple[int, str]:
        calls.append(argv)
        return 0, "OK\n"

    assert handoff(_opts(tmp_path, verify=True), now=NOW, runner=runner).verified is True
    assert "--output-last-message" not in calls[0]


def test_run_gives_the_subprocess_no_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, stdout="OK", stderr="")

    monkeypatch.setattr(handoff_mod.subprocess, "run", fake_run)
    assert handoff_mod._run(["x"], tmp_path) == (0, "OK")
    assert seen["stdin"] is subprocess.DEVNULL

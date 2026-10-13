from pathlib import Path

from session_builders import build, words

from loosy_goose.sessions.model import (
    dump_line,
    explicit_session,
    newer_than_tested,
    read_jsonl,
    snapshot,
)
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    Trigger,
    find_trigger,
    split_region,
    unit_ids,
)

TRIGGER = "<command-name>/loosy</command-name>"


def test_snapshot_ignores_a_partial_last_line(tmp_path: Path) -> None:
    src = tmp_path / "s.jsonl"
    src.write_bytes(b'{"a":1}\n{"b":2}\n{"c":')
    dest = tmp_path / "snap.jsonl"
    assert snapshot(src, dest) == 5
    assert dest.read_bytes() == b'{"a":1}\n{"b":2}\n'
    assert src.read_bytes() == b'{"a":1}\n{"b":2}\n{"c":'


def test_read_jsonl_tolerates_crlf_and_round_trips_undecodable_bytes(tmp_path: Path) -> None:
    src = tmp_path / "s.jsonl"
    src.write_bytes(b'{"t":"ok"}\r\n\n{"t":"caf\xe9"}\nnot json\n[1]\n')
    lines, warnings = read_jsonl(src)
    assert [n for n, _ in lines] == [0, 2]
    assert dump_line(lines[1][1]) == b'{"t":"caf\xe9"}\n'
    assert warnings == ["line 4: not valid JSON, skipped", "line 5: not a JSON object, skipped"]


def test_explicit_session_takes_an_absolute_file_or_an_id_only(tmp_path: Path) -> None:
    path = tmp_path / "s.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    found = tmp_path / "found.jsonl"
    asked: list[str] = []

    def by_id(sid: str) -> list[Path]:
        asked.append(sid)
        return [found]

    assert explicit_session(str(path), by_id) == [path]
    assert explicit_session("33333333-3333-4333-8333-333333333333", by_id) == [found]
    assert explicit_session("*", by_id) == []
    assert explicit_session("relative/s.jsonl", by_id) == []
    assert asked == ["33333333-3333-4333-8333-333333333333"]


def test_newer_than_tested_compares_numerically() -> None:
    assert newer_than_tested("2.1.300", ("2.1.296",))
    assert not newer_than_tested("2.1.296", ("2.1.296",))
    assert not newer_than_tested("2.1.99", ("2.1.296",))
    assert not newer_than_tested(None, ("2.1.296",))


def test_trigger_is_the_last_prompt_carrying_a_trigger_string() -> None:
    s = build(
        ("prompt", f"older {TRIGGER}"),
        ("reply", "a"),
        ("prompt", "work"),
        ("reply", "b"),
        ("prompt", TRIGGER),
        ("reply", "c"),
    )
    assert find_trigger(s, DEFAULT_TRIGGERS["claude"]) == Trigger(4, "text")


def test_a_later_handoff_call_beats_an_older_trigger_text() -> None:
    s = build(
        ("prompt", TRIGGER),
        ("reply", "a"),
        ("prompt", "hand this session off please"),
        ("handoff:h1", "{}"),
    )
    assert find_trigger(s, DEFAULT_TRIGGERS["claude"]) == Trigger(2, "tool-call")


def test_an_open_codex_turn_is_the_trigger() -> None:
    s = build(
        ("prompt", "one"),
        ("reply", "a"),
        ("prompt", "two"),
        ("reply", "b"),
        tool="codex",
        open_exchange=2,
    )
    assert find_trigger(s, DEFAULT_TRIGGERS["codex"]) == Trigger(2, "open-turn")


def test_no_trigger_excludes_nothing() -> None:
    s = build(("prompt", "one"), ("reply", "a"))
    assert find_trigger(s, DEFAULT_TRIGGERS["claude"]) == Trigger(None, "none")


def test_units_join_a_call_with_its_result_only() -> None:
    s = build(
        ("prompt", "a"),
        ("call:c1", "{}"),
        ("call:c2", "{}"),
        ("result:c1", "x"),
        ("result:c2", "y"),
    )
    u = unit_ids(s)
    assert u[1] == u[3] and u[2] == u[4]
    assert len({u[0], u[1], u[2]}) == 3


def test_tail_starts_at_the_last_prompt_before_the_trigger() -> None:
    s = build(
        ("prompt", words(50)),
        ("reply", words(50)),
        ("prompt", words(10)),
        ("reply", words(10)),
        ("prompt", TRIGGER),
    )
    region = split_region(s, Trigger(4, "text"), protect_tail_tokens=0)
    assert (region.end, region.tail_from, region.turn_record) == (4, 2, (0, 1))
    assert [t.index for t in region.transcript.turns] == [0, 1]


def test_tail_extends_back_to_protect_tail_tokens() -> None:
    s = build(
        ("prompt", words(50)),
        ("reply", words(50)),
        ("prompt", words(10)),
        ("reply", words(10)),
        ("prompt", TRIGGER),
    )
    region = split_region(s, Trigger(4, "text"), protect_tail_tokens=30)
    assert region.tail_from == 1
    assert region.turn_record == (0,)


def test_tail_never_splits_a_tool_pair_even_for_a_huge_result() -> None:
    s = build(
        ("prompt", words(20)),
        ("call:c1", '{"command": "cat big.log"}'),
        ("result:c1", words(20_000, "line")),
        ("prompt", words(5)),
        ("reply", words(5)),
    )
    region = split_region(s, Trigger(None, "none"), protect_tail_tokens=100)
    assert region.tail_from == 1
    assert region.turn_record == (0,)


def test_compressible_part_leaves_out_thinking_pinned_and_meta() -> None:
    s = build(
        ("summary", "earlier work"),
        ("meta", ""),
        ("prompt", "a"),
        ("think:m1", "hmm"),
        ("reply:m1", "b"),
        ("prompt", "c"),
    )
    region = split_region(s, Trigger(None, "none"), protect_tail_tokens=0)
    assert region.tail_from == 5
    assert region.turn_record == (2, 4)


def test_a_session_with_no_prompt_before_the_trigger_has_nothing_to_compress() -> None:
    s = build(("meta", ""), ("prompt", TRIGGER))
    region = split_region(s, Trigger(1, "text"), protect_tail_tokens=0)
    assert region.turn_record == ()

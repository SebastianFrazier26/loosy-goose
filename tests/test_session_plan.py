from collections.abc import Callable

import pytest
from session_builders import build, kept_where, words

from loosy_goose.segment import Segment, segment
from loosy_goose.sessions.model import Session
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    GAP,
    MARKER_PREFIX,
    TRIMMED_RESULT,
    ForkPlan,
    HandoffRefusedError,
    PriorFork,
    Region,
    clean_transcript,
    find_prior_fork,
    find_trigger,
    marker_text,
    plan_fork,
    prior_fork,
    recompress_warning,
    region_tokens,
    split_region,
    strip_part,
    strip_placeholders,
)
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Block, Transcript, Turn

TRIGGER = "<command-name>/loosy</command-name>"
LOG = [f"log line {i} " + "x" * 190 for i in range(6)]


def _session() -> Session:
    return build(
        ("prompt", "Remember ALPHA."),  # 0
        ("reply", "noted ALPHA"),  # 1
        ("prompt", "Run the log."),  # 2
        ("call:c1", '{"command": "cat run.log"}'),  # 3
        ("result:c1", "\n".join(LOG)),  # 4: three 600-char chunks
        ("reply", "first para BETA.\n\nsecond para GAMMA."),  # 5
        ("think:m9", "private musing"),  # 6
        ("reply:m9", "DELTA reply"),  # 7
        ("prompt", "tail prompt"),  # 8
        ("reply", "tail reply"),  # 9
        ("prompt", TRIGGER),  # 10
        ("reply", "running handoff"),  # 11
    )


def _plan(
    keep: Callable[[Segment], bool],
    *,
    session: Session | None = None,
    keep_thinking: bool = False,
) -> tuple[Session, Region, ForkPlan]:
    s = session if session is not None else _session()
    region = split_region(s, find_trigger(s, DEFAULT_TRIGGERS["claude"]), protect_tail_tokens=0)
    plan = plan_fork(
        s, region, kept_where(region, keep), date="2026-10-13", keep_thinking=keep_thinking
    )
    return s, region, plan


def test_marker_text_is_exact() -> None:
    assert marker_text("abc", "2026-10-13", 10, 40, 7) == (
        "[loosy-goose: compressed from session abc on 2026-10-13; kept 10 of 40 tokens before "
        "the last 7 tokens, which are verbatim. Earlier turns are an extractive selection — "
        "some were dropped.]"
    )
    assert marker_text("x", "d", 1, 2, 3).startswith(MARKER_PREFIX)


def test_everything_kept_rewrites_nothing_and_excludes_the_trigger_turn() -> None:
    _, _, plan = _plan(lambda s: True)
    assert plan.keep == (0, 1, 2, 3, 4, 5, 7, 8, 9)
    assert plan.rewrites == {}
    assert plan.tail_from == 8


def test_a_kept_result_chunk_keeps_its_call_whole() -> None:
    _, _, plan = _plan(lambda s: s.kind == "tool_result" and "log line 0" in s.text)
    assert plan.keep == (3, 4, 8, 9)
    assert plan.rewrites == {4: {0: f"{LOG[0]}\n{LOG[1]}\n{GAP}"}}


def test_a_kept_call_alone_keeps_its_result_trimmed() -> None:
    _, _, plan = _plan(lambda s: s.kind == "tool_use")
    assert plan.keep == (3, 4, 8, 9)
    assert plan.rewrites == {4: {0: TRIMMED_RESULT}}


def test_dropping_a_middle_chunk_marks_the_gap() -> None:
    _, _, plan = _plan(
        lambda s: s.kind == "tool_result" and ("log line 0" in s.text or "log line 4" in s.text)
    )
    assert plan.rewrites == {4: {0: f"{LOG[0]}\n{LOG[1]}\n{GAP}\n{LOG[4]}\n{LOG[5]}"}}


def test_a_partly_kept_reply_keeps_only_its_kept_paragraphs() -> None:
    _, _, plan = _plan(lambda s: s.text == "second para GAMMA.")
    assert plan.keep == (5, 8, 9)
    assert plan.rewrites == {5: {0: "second para GAMMA."}}


def test_older_thinking_is_dropped_unless_asked_and_its_reply_survives() -> None:
    assert 6 not in _plan(lambda s: True)[2].keep
    assert 6 in _plan(lambda s: True, keep_thinking=True)[2].keep
    assert 6 not in _plan(lambda s: "DELTA" not in s.text, keep_thinking=True)[2].keep


def test_meta_records_follow_their_exchange() -> None:
    s = build(
        ("prompt", "a"),
        ("meta", ""),
        ("reply", "b"),
        ("prompt", "c"),
        ("meta", ""),
        ("reply", "d"),
        ("prompt", "e"),
    )
    _, _, plan = _plan(lambda seg: seg.text == "b", session=s)
    assert plan.keep == (1, 2, 6)


def test_a_meta_record_tied_to_a_prompt_follows_that_prompt_not_its_exchange() -> None:
    s = build(
        ("prompt:p1", "Remember ALPHA."),  # 0
        ("echo:p1", ""),  # 1
        ("reply", "noted"),  # 2
        ("meta", ""),  # 3
        ("prompt", "tail"),  # 4
        ("prompt", TRIGGER),  # 5
    )
    assert _plan(lambda seg: seg.text == "noted", session=s)[2].keep == (2, 3, 4)
    assert _plan(lambda seg: True, session=s)[2].keep == (0, 1, 2, 3, 4)


def test_pinned_records_are_kept_even_when_nothing_is_selected() -> None:
    s = build(("summary", "old work"), ("prompt", "a"), ("reply", "b"), ("prompt", "c"))
    _, _, plan = _plan(lambda seg: False, session=s)
    assert plan.keep == (0, 3)


def test_an_unknown_record_inside_the_compressed_stretch_refuses() -> None:
    s = build(("prompt", "a"), ("unknown", ""), ("reply", "b"), ("prompt", "c"))
    with pytest.raises(HandoffRefusedError, match="'mystery' record at line 2"):
        _plan(lambda seg: True, session=s)


def test_an_unknown_record_in_the_tail_passes_through_with_a_warning() -> None:
    s = build(("prompt", "a"), ("reply", "b"), ("prompt", "c"), ("unknown", ""))
    _, _, plan = _plan(lambda seg: True, session=s)
    assert 3 in plan.keep
    assert any("'mystery' record at line 4 passed through unchanged" in w for w in plan.warnings)


def test_side_records_drop_and_loose_unknown_records_pass_through() -> None:
    s = build(
        ("prompt", "a"), ("side", ""), ("reply", "b"), ("prompt", "c"), loose=("mystery-side",)
    )
    _, _, plan = _plan(lambda seg: True, session=s)
    assert plan.keep == (0, 2, 3)
    assert plan.passthrough == (4,)
    assert any("mystery-side" in w for w in plan.warnings)


def test_token_accounting_and_marker() -> None:
    s, region, plan = _plan(lambda seg: True)
    assert plan.compressible_tokens == sum(s.tokens(ri) for ri in region.turn_record)
    assert plan.kept_tokens == plan.compressible_tokens
    assert plan.tail_tokens == s.tokens(8) + s.tokens(9)
    assert region_tokens(s, region) == (plan.compressible_tokens, plan.tail_tokens)
    assert (
        f"kept {plan.kept_tokens} of {plan.compressible_tokens} tokens before the last "
        f"{plan.tail_tokens} tokens" in plan.marker
    )
    assert "from session source-id on 2026-10-13" in plan.marker
    partial = _plan(lambda seg: seg.text == "second para GAMMA.")[2]
    assert 0 < partial.kept_tokens < partial.compressible_tokens


def test_strip_placeholders_removes_placeholder_lines_and_inline_mentions() -> None:
    assert strip_placeholders(f"a\n{GAP}\nb") == "a\nb"
    assert strip_placeholders(TRIMMED_RESULT) == ""
    assert strip_placeholders(f"saw {TRIMMED_RESULT} here\n\nnext") == "saw  here\n\nnext"
    assert strip_placeholders("plain\n\ntext") == "plain\n\ntext"
    cleaned: dict[int, str] = {}
    assert strip_part("plain", 0, cleaned) == "plain" and cleaned == {}
    assert strip_part(GAP, 2, cleaned) == "" and cleaned == {2: ""}


def test_a_prior_fork_reports_its_date_and_the_tokens_added_since() -> None:
    kept_text, tail_text, new_text = words(3), words(2, "gamma"), words(1300, "beta")
    marker = marker_text(
        "old-id", "2026-10-12", count_tokens(kept_text), 50, count_tokens(tail_text)
    )
    s = build(
        ("marker", marker),
        ("prompt", kept_text),
        ("reply", tail_text),
        ("prompt", new_text),
        ("prompt", TRIGGER),
    )
    added = count_tokens(new_text)
    prior = find_prior_fork(s, 4)
    assert prior == PriorFork("old-id", "2026-10-12", added)
    assert added >= 1000
    assert recompress_warning(prior) == (
        f"this session was already compressed by loosy-goose on 2026-10-12; about {added:,} "
        "tokens added since — compressing again compounds the loss"
    )
    assert find_prior_fork(s, 0) is None
    assert find_prior_fork(build(("prompt", "a")), 1) is None
    assert prior_fork(f"{MARKER_PREFIX}x on 2026-10-13; ...]", 10) is None


def test_placeholders_and_the_marker_never_reach_scoring_or_token_counts() -> None:
    s = build(
        ("marker", marker_text("old-id", "2026-10-12", 1, 2, 3)),  # 0
        ("prompt", "Run the log."),  # 1
        ("call:c1", '{"command": "cat run.log"}'),  # 2
        ("result:c1", f"{LOG[0]}\n{GAP}\n{LOG[4]}"),  # 3
        ("reply", f"saw {TRIMMED_RESULT} earlier.\n\nall fine."),  # 4
        ("prompt", f"tail {GAP} prompt"),  # 5
        ("prompt", TRIGGER),  # 6
    )
    region = split_region(s, find_trigger(s, DEFAULT_TRIGGERS["claude"]), protect_tail_tokens=0)
    scored = [b.text for t in region.transcript.turns for b in t.blocks]
    scored += [seg.text for seg in segment(region.transcript)]
    assert not any(a in text for text in scored for a in (GAP, TRIMMED_RESULT, MARKER_PREFIX))
    assert region.turn_record == (1, 2, 3, 4)
    assert s.tokens(0) == 0
    assert s.tokens(3) == count_tokens(f"{LOG[0]}\n{LOG[4]}")
    plan = plan_fork(s, region, kept_where(region, lambda seg: True), date="2026-10-13")
    assert 0 not in plan.keep
    assert plan.rewrites == {
        3: {0: f"{LOG[0]}\n{LOG[4]}"},
        4: {0: "saw  earlier.\n\nall fine."},
        5: {0: "tail  prompt"},
    }
    assert plan.compressible_tokens == sum(s.tokens(ri) for ri in (1, 2, 3, 4))


def test_a_placeholder_only_result_stays_with_its_call_and_empty() -> None:
    s = build(
        ("prompt", "Run it."),  # 0
        ("call:c1", '{"command": "true"}'),  # 1
        ("result:c1", TRIMMED_RESULT),  # 2
        ("prompt", "tail"),  # 3
        ("prompt", TRIGGER),  # 4
    )
    assert (s.records[2].blocks, s.records[2].cleaned) == ([], {0: ""})
    _, region, plan = _plan(lambda seg: seg.kind == "tool_use", session=s)
    assert region.turn_record == (0, 1)
    assert plan.keep == (1, 2, 3)
    assert plan.rewrites == {2: {0: ""}}


def test_a_result_trimmed_again_gets_placeholders_only_where_this_run_trims() -> None:
    earlier = "\n".join([LOG[0], LOG[1], GAP, LOG[4], LOG[5]])
    s = build(
        ("prompt", "Run the log."),
        ("call:c1", '{"command": "cat run.log"}'),
        ("result:c1", earlier),  # 2: two chunks once the inherited gap is gone
        ("prompt", "tail"),
        ("prompt", TRIGGER),
    )
    whole = _plan(lambda seg: seg.kind == "tool_result", session=s)[2]
    assert whole.rewrites == {2: {0: "\n".join([LOG[0], LOG[1], LOG[4], LOG[5]])}}
    later = _plan(lambda seg: seg.kind == "tool_result" and "log line 4" in seg.text, session=s)[2]
    assert later.rewrites == {2: {0: "\n".join([GAP, LOG[4], LOG[5]])}}


def test_clean_transcript_drops_the_marker_and_placeholders_and_finds_the_prior_fork() -> None:
    transcript = Transcript(
        "t",
        [
            Turn(0, "user", [Block("text", marker_text("old-id", "2026-10-12", 0, 9, 0))]),
            Turn(1, "assistant", [Block("text", "noted ALPHA")]),
            Turn(2, "tool", [Block("tool_result", TRIMMED_RESULT)]),
            Turn(3, "tool", [Block("tool_result", f"{LOG[0]}\n{GAP}")]),
        ],
    )
    cleaned, prior = clean_transcript(transcript)
    assert [(t.index, t.role, [b.text for b in t.blocks]) for t in cleaned.turns] == [
        (0, "assistant", ["noted ALPHA"]),
        (1, "tool", [LOG[0]]),
    ]
    added = count_tokens("noted ALPHA") + count_tokens(LOG[0])
    assert prior == PriorFork("old-id", "2026-10-12", added)
    assert clean_transcript(cleaned)[1] is None


def test_a_unit_straddling_the_trigger_is_dropped_whole() -> None:
    in_tail = build(
        ("prompt", "a"),  # 0
        ("call:c1", '{"command": "true"}'),  # 1: its result lands after the trigger
        ("prompt", TRIGGER),  # 2
        ("result:c1", "late output"),  # 3
    )
    plan = _plan(lambda seg: True, session=in_tail)[2]
    assert plan.keep == (0,)

    compressible = build(
        ("prompt", "a"),  # 0
        ("call:c1", '{"command": "true"}'),  # 1
        ("reply", "b"),  # 2
        ("prompt", "tail"),  # 3
        ("prompt", TRIGGER),  # 4
        ("result:c1", "late output"),  # 5
    )
    plan = _plan(lambda seg: True, session=compressible)[2]
    assert 1 not in plan.keep and 5 not in plan.keep
    assert plan.keep == (0, 2, 3)

    result_first = build(
        ("prompt", "a"),  # 0
        ("result:c1", "early output"),  # 1: its call comes after the trigger
        ("prompt", TRIGGER),  # 2
        ("call:c1", '{"command": "true"}'),  # 3
    )
    plan = _plan(lambda seg: True, session=result_first)[2]
    assert plan.keep == (0,)


def test_siblings_of_a_straddling_call_are_dropped_with_it() -> None:
    session = build(
        ("prompt", "a"),  # 0
        ("reply", "earlier answer"),  # 1: an unrelated reply
        ("prompt", "b"),  # 2
        ("think:r1", "pondering"),  # 3
        ("reply:r1", "running it"),  # 4
        ("call:c1", '{"command": "true"}'),  # 5: its result lands after the trigger
        ("prompt", TRIGGER),  # 6
        ("result:c1", "late output"),  # 7
    )
    session.records[5].reply = "r1"  # the builder gives calls no reply id
    for protect in (0, 10_000):
        region = split_region(
            session, find_trigger(session, DEFAULT_TRIGGERS["claude"]), protect_tail_tokens=protect
        )
        plan = plan_fork(session, region, kept_where(region, lambda s: True), date="2026-10-13")
        assert not {3, 4, 5, 7} & set(plan.keep)
        assert {0, 1, 2} <= set(plan.keep)

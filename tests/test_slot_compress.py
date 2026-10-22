import sys
from pathlib import Path

import pytest

pytest.importorskip("spacy")
pytest.importorskip("en_core_web_sm")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.slot_compress import slot  # noqa: E402
from loosy_goose.segment import Segment, SegmentKind, extract_atoms  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402

PROSE = (
    "I think we should probably refactor the budget module because it does not handle "
    "`token_budget` correctly. Do not push the change to main until the tests in "
    "tests/test_budget.py pass. The loader reads the transcript file and returns a list of "
    "turns. We never reuse the cache after the schema changes."
)


def _seg(text: str, kind: SegmentKind = "prose") -> Segment:
    return Segment(0, 0, "assistant", kind, text, False, extract_atoms(text))


@pytest.mark.parametrize("mode", ["direct", "english"])
def test_negation_survives(mode: slot.Mode) -> None:
    out = slot.slot_text("You should not push the change to the main branch until review.", mode)
    if mode == "direct":
        assert "~" in out.split()[0]
    else:
        assert "not" in out.split()


@pytest.mark.parametrize("mode", ["direct", "english"])
def test_never_flags_negation(mode: slot.Mode) -> None:
    text = (
        "The loader of the system really does never read all of the transcript files from the "
        "old cache directory."
    )
    out = slot.slot_text(text, mode)
    assert out != text
    assert ("~" in out) if mode == "direct" else ("not" in out.split())


@pytest.mark.parametrize("mode", ["direct", "english"])
def test_atoms_survive_verbatim(mode: slot.Mode) -> None:
    out = slot.slot_text(PROSE, mode)
    for atom in extract_atoms(PROSE):
        assert atom in out


def test_modality_flag() -> None:
    out = slot.slot_text(
        "The team should really rewrite the whole scheduler module before the next release "
        "happens.",
        "direct",
    )
    assert "!rewrite" in out


def test_code_fence_passthrough() -> None:
    fenced = "```\nThe loader reads the transcript file and returns a list of turns.\n```"
    assert slot.slot_text(fenced, "direct") == fenced
    assert slot.slot_text(fenced, "english") == fenced


def test_heading_table_tag_code_passthrough() -> None:
    text = (
        "# Heading here\n| a | b |\n<tag attr>\n"
        "int check(int major, int minor); // returns 1 if compatible\n"
        "E           ValueError: to_pydatetime cannot be called with date32 type here\n"
    )
    assert slot.slot_text(text, "direct") == text


def test_list_items_are_slotted_behind_their_marker() -> None:
    text = "- The loader reads the transcript file and returns the turns.\n1. We must never push."
    out = slot.slot_text(text, "direct").split("\n")
    assert out[0].startswith("- loader: read")
    assert out[-1].startswith("1. ~!push")


def test_agent_written_once_until_changed() -> None:
    text = (
        "The loader of the system really does read all of the transcript files from the old "
        "cache directory. The loader of the system really does write all of the summary "
        "reports to the new output directory."
    )
    out = slot.slot_text(text, "direct").split("\n")
    assert len(out) == 2
    assert out[0].startswith("loader of system: ")
    assert ":" not in out[1]
    assert slot.slot_text(text, "english").count("loader of system") == 2


def test_consecutive_same_act_rows_merge() -> None:
    text = "The loader fixes the parser bug. The loader fixes the cache bug."
    assert slot.slot_text(text, "direct") == "loader: fix parser bug, cache bug"


def test_copula_and_filler() -> None:
    assert slot.slot_text("The license is a new community license.", "direct") == (
        "license=community license"
    )
    assert slot.slot_text("The license is not a new community license.", "english") == (
        "license is not community license."
    )
    out = slot.slot_text("Perfect! Now let's run the test again to see the output.", "direct")
    assert out == "run test"


def test_soft_wrapped_lines_join_before_parsing() -> None:
    wrapped = "The loader reads the transcript file and returns\na list of turns for each session."
    assert slot.slot_text(wrapped, "direct") == slot.slot_text(wrapped.replace("\n", " "), "direct")
    assert "\n" not in slot.slot_text(wrapped, "english")


def test_shouted_contraction_keeps_negation() -> None:
    out = slot.slot_text("This means you DON'T have to modify the testing logic at all.", "direct")
    assert out.startswith("~")


def test_exact_duplicate_rows_elided() -> None:
    text = "The loader reads the transcript file.\nThe loader reads the transcript file."
    direct = slot.slot_text(text, "direct")
    assert len(direct.split("\n")) == 1
    assert slot.slot_text(text, "english").count("read") == 1


def test_english_has_no_blank_carry() -> None:
    out = slot.slot_text(PROSE, "english")
    assert "|" not in out


def test_segments_never_longer() -> None:
    segs = [_seg(PROSE), _seg("Short one."), _seg("Run it now.", "thinking")]
    for mode in ("direct", "english"):
        for old, new in zip(segs, slot.slot_segments(segs, mode), strict=True):
            assert count_tokens(new.text) <= count_tokens(old.text)


def test_non_prose_kinds_untouched() -> None:
    segs = [_seg(PROSE, k) for k in ("code", "tool_result", "tool_use")]
    out = slot.slot_segments(segs, "direct")
    assert [s.text for s in out] == [s.text for s in segs]


def test_prose_and_thinking_are_transformed_not_mutated() -> None:
    segs = [_seg(PROSE, "prose"), _seg(PROSE, "thinking")]
    out = slot.slot_segments(segs, "english")
    assert all(n.text != PROSE for n in out)
    assert all(s.text == PROSE for s in segs)


def test_deterministic() -> None:
    for mode in ("direct", "english"):
        assert slot.slot_text(PROSE, mode) == slot.slot_text(PROSE, mode)


def test_legend_budget() -> None:
    assert count_tokens(slot.LEGEND) <= 60


def _gentle(text: str) -> str:
    return slot.slot_text(text, "english", profile="gentle")


def test_version_stamps_differ() -> None:
    assert slot.version_for("v2") == slot.SLOT_VERSION == 2
    assert slot.version_for("gentle") == slot.GENTLE_VERSION == 3


def test_v2_default_profile_unchanged() -> None:
    assert slot.slot_text(PROSE, "direct") == slot.slot_text(PROSE, "direct", profile="v2")


def test_gentle_dont_have_to_is_not_a_prohibition() -> None:
    out = _gentle("This means you DON'T have to modify the testing logic or any of the tests.")
    assert "need not modify" in out
    assert "not modify" not in out.replace("need not modify", "")
    assert "don't" not in out.lower()


@pytest.mark.parametrize(
    "text",
    [
        "You don't need to rebuild the index first.",
        "There is no need to restart the server after the change.",
    ],
)
def test_gentle_need_not_rewrites(text: str) -> None:
    assert "need not" in _gentle(text).lower()


def test_gentle_copula_negation_stays_on_is() -> None:
    out = _gentle("That isn't a catalog-recognized license for the package.")
    assert "is not" in out
    assert "catalog-recognized" in out


def test_gentle_must_not_adjacent_to_verb() -> None:
    out = _gentle("You must not push the change to the main branch until the review ends.")
    assert "must not push" in out.lower()
    out = _gentle("We shouldn't ship the build before the checks pass on the server.")
    assert "should not ship" in out.lower()


def test_gentle_not_only_kept() -> None:
    assert "not only" in _gentle("The cache is not only stale but also wrong for the new schema.")


def test_gentle_cause_clause_and_marker_kept() -> None:
    out = _gentle("The test for the parser failed because the mask was not preserved as expected.")
    assert "because" in out and "mask" in out and "not preserved" in out


@pytest.mark.parametrize("marker", ["if", "unless", "until", "although", "when", "since"])
def test_gentle_clause_markers_kept(marker: str) -> None:
    out = _gentle(f"We will deploy the service {marker} the review of the change is finished.")
    assert marker in out.split()


def test_gentle_conjuncts_kept() -> None:
    out = _gentle("The loader reads the transcript file and keeps the schema version around.")
    assert "reads" in out and "keeps" in out and "and" in out


def test_gentle_prepositional_detail_kept() -> None:
    out = _gentle("The worker writes the summary to the shared output directory in the morning.")
    assert "to shared output directory" in out and "in morning" in out


def test_gentle_hedges_collapse_to_one_marker() -> None:
    out = _gentle("The cache is maybe stale, and the index is probably broken as well.")
    assert out.split().count("likely") == 1
    assert "maybe" not in out and "probably" not in out


def test_gentle_tense_marker_kept() -> None:
    assert "fixed" in _gentle("The team fixed the parser bug in the loader module yesterday.")
    assert "will fix" in _gentle("The team will fix the parser bug in the loader module later.")


def test_gentle_drops_filler_and_agents() -> None:
    out = _gentle("Perfect! Now let me check the logs because the build failed.")
    assert out == "Check logs because build failed."
    assert "I " not in _gentle("I updated the loader module and the config file for the service.")


def test_gentle_keeps_you_in_copula() -> None:
    assert _gentle("You are a helpful assistant that can interact with a computer.").startswith(
        "You are"
    )


def test_gentle_atoms_survive_and_never_longer() -> None:
    out = _gentle(PROSE)
    for atom in extract_atoms(PROSE):
        assert atom in out
    for s in slot.slot_segments([_seg(PROSE)], "english", profile="gentle"):
        assert count_tokens(s.text) <= count_tokens(PROSE)


def test_gentle_code_and_tags_pass_through() -> None:
    text = (
        "x = compute(a, b);\n"
        "<response clipped><NOTE>To save on the context only part of this.</NOTE>"
    )
    assert _gentle(text) == text


PROSE_RESULT = "\n".join(
    [
        "The loader reads the transcript file and returns a list of turns.",
        "It does not handle the token budget correctly when the schema changes.",
        "We should refactor the module because the cache is stale.",
        "This means you don't have to rebuild the index by hand after the change.",
        "Please check the other loaders too, since they share the same parser.",
    ]
)
CODE_RESULT = "\n".join(f"src/mod{i}.py:{i}:    x = foo(bar, baz);" for i in range(12))
TRACE_RESULT = (
    "Traceback (most recent call last):\n"
    '  File "/repo/a.py", line 3, in main\n'
    "    raise ValueError('The loader reads the transcript file and returns a list of turns.')\n"
    "ValueError: The loader reads the transcript file and returns a list of turns today."
)


def _tool(text: str, kind: SegmentKind = "tool_result") -> Segment:
    return _seg(text, kind)


def test_is_prose_like_on_and_off() -> None:
    assert slot.is_prose_like(_tool(PROSE_RESULT))
    assert not slot.is_prose_like(_tool(CODE_RESULT))
    assert not slot.is_prose_like(_tool(TRACE_RESULT))
    assert not slot.is_prose_like(_tool("The loader reads the file."))
    assert not slot.is_prose_like(_tool(PROSE_RESULT, "prose"))


def test_tool_prose_only_with_flag_and_gentle() -> None:
    seg = _tool(PROSE_RESULT)
    assert slot.slot_segments([seg], "english", profile="gentle")[0].text == PROSE_RESULT
    assert slot.slot_segments([seg], "english", include_tool_prose=True)[0].text == PROSE_RESULT
    out = slot.slot_segments([seg], "english", profile="gentle", include_tool_prose=True)[0]
    assert count_tokens(out.text) < count_tokens(PROSE_RESULT)
    assert "need not rebuild" in out.text


def test_tool_use_and_non_prose_results_untouched_under_gentle() -> None:
    segs = [_tool(PROSE_RESULT, "tool_use"), _tool(CODE_RESULT), _tool(TRACE_RESULT)]
    out = slot.slot_segments(segs, "english", profile="gentle", include_tool_prose=True)
    assert [s.text for s in out] == [s.text for s in segs]


def test_code_like_lines_inside_prose_result_verbatim() -> None:
    code = "    result = compute(alpha, beta);"
    text = PROSE_RESULT + "\n" + code
    out = slot.slot_segments([_tool(text)], "english", profile="gentle", include_tool_prose=True)
    assert out[0].text.endswith(code)
    assert out[0].text != text


def test_grep_style_lines_verbatim_in_tool_prose() -> None:
    line = "src/loader.py:42:The loader reads the transcript file and returns the turns."
    text = PROSE_RESULT + "\n" + line
    out = slot.slot_segments([_tool(text)], "english", profile="gentle", include_tool_prose=True)
    assert line in out[0].text.split("\n")

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
        assert "~" in out.split("|")[0]
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
    assert ("~" in out.split("|")[0]) if mode == "direct" else ("not" in out.split())


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
    assert out.startswith("!")


def test_code_fence_passthrough() -> None:
    fenced = "```\nThe loader reads the transcript file and returns a list of turns.\n```"
    assert slot.slot_text(fenced, "direct") == fenced
    assert slot.slot_text(fenced, "english") == fenced


def test_list_heading_table_passthrough() -> None:
    text = "# Heading here\n- the loader reads the transcript file\n| a | b |\n"
    assert slot.slot_text(text, "direct") == text


def test_carry_forward_blanks_repeated_slot() -> None:
    text = (
        "The loader of the system really does read all of the transcript files from the old "
        "cache directory. The loader of the system really does write all of the summary "
        "reports to the new output directory."
    )
    out = slot.slot_text(text, "direct").split("\n")
    assert len(out) == 2
    assert out[1].split("|")[1] == ""


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

import json

import pytest

from loosy_goose.paths import (
    PathTable,
    build_table,
    expand,
    find_paths,
    substitute,
    substitute_text,
)
from loosy_goose.segment import Segment, SegmentKind, extract_atoms


def _seg(i: int, text: str, kind: SegmentKind = "prose") -> Segment:
    return Segment(i, i, "assistant", kind, text, False, extract_atoms(text))


def test_find_paths_does_not_double_count_the_filename_inside_a_path() -> None:
    assert find_paths("edit src/main.py now") == ["src/main.py"]
    assert find_paths("edit main.py and src/other.py") == ["main.py", "src/other.py"]


def test_build_table_holds_each_path_once_in_first_mention_order() -> None:
    segs = [
        _seg(0, "read src/beta.py then src/alpha.py"),
        _seg(1, "read src/alpha.py again and src/beta.py again"),
    ]
    table = build_table(segs)
    assert table.paths == ("src/beta.py", "src/alpha.py")
    assert table.render().splitlines() == ["[P0] src/beta.py", "[P1] src/alpha.py"]


def test_min_mentions_trades_tokens_against_recall() -> None:
    segs = [_seg(0, "src/twice.py and src/once.py"), _seg(1, "src/twice.py again")]
    assert build_table(segs).paths == ("src/twice.py",)
    assert set(build_table(segs, min_mentions=1).paths) == {"src/twice.py", "src/once.py"}
    with pytest.raises(ValueError):
        build_table(segs, min_mentions=0)


def test_substitution_round_trips_through_expand() -> None:
    segs = [
        _seg(0, "open src/alpha.py"),
        _seg(1, "open src/alpha.py and src/alpha.py once more"),
    ]
    table = build_table(segs)
    for original, done in zip(segs, substitute(segs, table), strict=True):
        assert "src/alpha.py" not in done.text
        assert expand(done.text, table) == original.text


def test_longer_paths_are_substituted_before_the_bare_filename_they_end_with() -> None:
    mapping = {"src/main.py": "[P0]", "main.py": "[P1]"}
    assert substitute_text("src/main.py and main.py", mapping) == "[P0] and [P1]"


def test_a_tool_call_stays_valid_json_and_keeps_its_escaping() -> None:
    payload = json.dumps(
        {"file_path": "venv\\Scripts\\pip.exe", "new_string": "x = 1\ny = 2"},
        ensure_ascii=False,
        sort_keys=True,
    )
    table = PathTable(("venv\\Scripts\\pip.exe",))
    done = substitute_text(payload, table.mapping())
    decoded = json.loads(done)
    assert decoded["file_path"] == "[P0]"
    # The body's real newline must survive as an escape, not be flattened into the text.
    assert decoded["new_string"] == "x = 1\ny = 2"


def test_a_tool_call_with_no_substitutable_path_is_returned_unchanged() -> None:
    payload = json.dumps({"command": "ls -la", "limit": 20}, ensure_ascii=False, sort_keys=True)
    assert substitute_text(payload, {"src/x.py": "[P0]"}) == payload


def test_paths_inside_a_tool_call_are_found_in_decoded_form() -> None:
    # The raw segment text spells this path with doubled backslashes; the table must hold the
    # single-backslash path the reader would recognise.
    payload = json.dumps({"file_path": "venv\\Scripts\\pip.exe"}, ensure_ascii=False)
    segs = [_seg(0, payload, kind="tool_use"), _seg(1, payload, kind="tool_use")]
    assert build_table(segs).paths == ("venv\\Scripts\\pip.exe",)


def test_marker_escapes_a_collision_with_real_text() -> None:
    segs = [_seg(0, "the log prints [P0] literally, see src/a.py"), _seg(1, "src/a.py again")]
    table = build_table(segs)
    assert table.marker != "[P"
    assert table.placeholder(0) not in segs[0].text


def test_substituted_segments_lose_the_path_and_the_table_guarantees_it() -> None:
    segs = [_seg(0, "touch src/alpha.py"), _seg(1, "touch src/alpha.py twice")]
    table = build_table(segs)
    done = substitute(segs, table)
    assert all("src/alpha.py" not in set(s.atoms) for s in done)
    assert "src/alpha.py" in table.guaranteed_atoms()


def test_empty_table_is_a_no_op_and_costs_nothing() -> None:
    segs = [_seg(0, "no paths mentioned here at all")]
    table = build_table(segs)
    assert table.paths == ()
    assert table.tokens() == 0
    assert table.guaranteed_atoms() == set()
    assert substitute(segs, table) == segs


def test_table_tokens_count_the_markers_not_just_the_paths() -> None:
    table = PathTable(("src/alpha.py", "src/beta.py"))
    assert table.tokens() > 0
    assert "[P1] src/beta.py" in table.render()

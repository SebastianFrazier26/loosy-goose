import json

import pytest

from loosy_goose.code import quantize_segment
from loosy_goose.paths import (
    PATHS_VERSION,
    PathTable,
    build_table,
    expand,
    find_paths,
    substitute,
    substitute_text,
)
from loosy_goose.segment import Segment, SegmentKind, atoms_in_decoded, extract_atoms


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


def test_a_path_the_table_does_not_hold_is_not_mangled_by_one_it_does() -> None:
    # The fixture is a path rather than a bare `main.c` because `.c` is one of the seven
    # attribute-colliding extensions ATOMS_VERSION 4 recognises only inside a path; a bare
    # `main.c` is deliberately not a path here any more. The prefix relation is what matters.
    segs = [
        _seg(0, "open src/main.c now"),
        _seg(1, "src/main.c again"),
        _seg(2, "but src/main.cpp is different"),
    ]
    table = build_table(segs)
    assert table.paths == ("src/main.c",)
    done = substitute(segs, table)
    assert done[2].text == "but src/main.cpp is different"
    assert "src/main.cpp" in set(done[2].atoms)


def test_a_longer_path_ending_in_a_tabulated_filename_is_left_whole() -> None:
    # `main.c` is the table's row, so `src/main.c` may not be emitted as `src/[P0]`: the span is
    # the whole path, and the table promises nothing about it.
    assert substitute_text("see src/main.c here", {"main.c": "[P0]"}) == "see src/main.c here"


def test_a_tabulated_path_still_substitutes_against_adjacent_punctuation() -> None:
    # `src/main.c`, not a bare `main.c`: ATOMS_VERSION 4 recognises `.c` only inside a path, so
    # the bare form is no longer a span the table can name.
    mapping = {"src/main.c": "[P0]", "src/alpha.py": "[P1]"}
    assert substitute_text("(src/main.c), src/main.c:12", mapping) == "([P0]), [P0]:12"
    assert substitute_text("edit src/alpha.py, then run", mapping) == "edit [P1], then run"


def test_a_sentence_final_slash_path_drops_the_full_stop_as_the_extractor_does() -> None:
    # `_PATH_SLASH` accepts `.` inside a component, so the raw match is `src/alpha.py.`; the
    # extractor strips it and the metric counts `src/alpha.py`, so the table must key on that.
    assert find_paths("edit src/alpha.py.") == ["src/alpha.py"]
    assert find_paths("see C:\\repo\\alpha.py.") == ["C:\\repo\\alpha.py"]
    assert extract_atoms("edit src/alpha.py.")[0] == "src/alpha.py"


def test_a_sentence_final_mention_and_a_mid_sentence_one_share_a_row() -> None:
    segs = [_seg(0, "edit src/alpha.py."), _seg(1, "then src/alpha.py again")]
    assert build_table(segs).paths == ("src/alpha.py",)


def test_substituting_a_sentence_final_path_keeps_the_full_stop() -> None:
    mapping = {"src/alpha.py": "[P0]"}
    assert substitute_text("edit src/alpha.py.", mapping) == "edit [P0]."
    assert substitute_text("edit src/alpha.py...", mapping) == "edit [P0]..."


def test_a_last_component_with_inner_dots_is_untouched() -> None:
    assert find_paths("unpack a/b.tar.gz now") == ["a/b.tar.gz"]
    assert substitute_text("unpack a/b.tar.gz.", {"a/b.tar.gz": "[P0]"}) == "unpack [P0]."


def test_paths_version_is_6() -> None:
    assert PATHS_VERSION == 6


@pytest.mark.parametrize(
    "text",
    [
        "edit src/alpha.py.",
        "read C:\\repo\\src\\beta.py, then src/gamma.py; done",
        "unpack a/b.tar.gz. also main.py: and lib/x.c...",
        "(src/main.c), src/main.c:12",
        "cd foo/.. and ./tmp/",
        "no punctuation src/last.py",
    ],
)
def test_every_found_path_is_an_atom_of_the_same_text(text: str) -> None:
    # The table is keyed on `find_paths` and the metric on `extract_atoms`; a path the first names
    # that the second does not count is a guarantee the table cannot keep.
    atoms = set(extract_atoms(text))
    for p in find_paths(text):
        assert p in atoms, (p, text)


def test_substitution_round_trips_when_a_near_miss_path_shares_a_prefix() -> None:
    # Path forms, not bare filenames: under ATOMS_VERSION 4 a bare main.c is
    # deliberately not a path, which would leave the table empty and the
    # round-trip below asserting nothing.
    segs = [
        _seg(0, "open src/main.c and src/main.cpp"),
        _seg(1, "src/main.c again, plus lib/src/main.c and src/utils.c"),
        _seg(2, "src/utils.c twice"),
    ]
    table = build_table(segs)
    done = substitute(segs, table)
    assert table.paths, "fixture no longer builds a table; the round-trip would be vacuous"
    assert any(out.text != original.text for original, out in zip(segs, done, strict=True))
    for original, out in zip(segs, done, strict=True):
        assert expand(out.text, table) == original.text


def test_the_mask_blanks_exactly_the_placeholders_the_table_emits() -> None:
    table = PathTable(("src/a.py",) * 12, marker="[PP")
    for i in (0, 11):
        assert table.mask(f"x{table.placeholder(i)}y") == "x y"
    assert table.mask("[P0] [PP] PP0]") == "[P0] [PP] PP0]"


def test_substitution_never_adds_an_atom() -> None:
    # The substituted path must not come back, and nothing the marker sits next to may appear.
    segs = [_seg(0, "open src/alpha.py then 10"), _seg(1, "src/alpha.py again")]
    table = build_table(segs)
    for original, out in zip(segs, substitute(segs, table), strict=True):
        assert set(out.atoms) <= set(original.atoms)
        assert "src/alpha.py" not in out.atoms


def test_substitution_chains_after_banding_without_restoring_key_names() -> None:
    payload = {"file_path": "src/alpha.py", "new_string": "run_job(1)\n" * 40, "replace_all": True}
    text = json.dumps(payload)
    call = Segment(0, 0, "assistant", "tool_use", text, False, extract_atoms(text), "Edit")
    banded = quantize_segment(call, 0.0)
    assert "new_string: " in banded.text
    segs = [banded, _seg(1, "src/alpha.py again")]
    table = build_table(segs)
    done = substitute(segs, table)
    assert done[0].text != banded.text, "fixture no longer substitutes the call"
    assert set(done[0].atoms) <= set(banded.atoms) <= set(call.atoms)
    assert not {"new_string", "replace_all"} & set(done[0].atoms)
    assert "run_job" in done[0].atoms


def _banded_edit(new_string: str) -> Segment:
    payload = {"file_path": "new_string/a.py", "new_string": new_string, "replace_all": True}
    text = json.dumps(payload)
    call = Segment(0, 0, "assistant", "tool_use", text, False, extract_atoms(text), "Edit")
    return quantize_segment(call, 0.0)


def _substitute_after(banded: Segment) -> Segment:
    segs = [banded, _seg(1, "see new_string/a.py again")]
    done = substitute(segs, build_table(segs))
    assert done[0].text.startswith("[P0]"), "fixture no longer substitutes the target"
    return done[0]


def test_a_key_name_only_inside_a_substituted_path_is_not_credited() -> None:
    # The rendered text still says `new_string: `; only the path held the name as content.
    banded = _banded_edit("x = 1")
    assert "new_string" in banded.atoms
    done = _substitute_after(banded)
    assert "new_string: " in done.text
    assert "new_string" not in done.atoms


def test_a_key_name_also_present_outside_the_path_stays_credited() -> None:
    done = _substitute_after(_banded_edit("rename new_string now"))
    assert "new_string" in done.atoms


def test_every_rewrite_leaves_atoms_readable_from_its_atom_source() -> None:
    # A stale `atom_source` would silently decide the next rewrite's atoms, so each rewrite must
    # leave it describing exactly the atoms it recorded.
    banded = _banded_edit("run_job(1)\n" * 40)
    done = _substitute_after(banded)
    for before, after in ((None, banded), (banded, done)):
        assert after.atom_source is not None
        found = set(atoms_in_decoded(after.atom_source))
        if before is not None:
            found &= set(before.atoms)
        assert set(after.atoms) <= found
    assert set(done.atoms) == set(atoms_in_decoded(done.atom_source or "")) & set(banded.atoms)
    assert _seg(0, "untouched prose").atom_source is None


def test_the_tail_of_a_hyphenated_name_is_not_tabulated_as_a_path() -> None:
    assert find_paths("call run.foo-set.js") == []
    assert find_paths(r"run -C:\x\y now") == [r"C:\x\y"]

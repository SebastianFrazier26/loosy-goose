import json

from loosy_goose.segment import Segment, SegmentKind
from loosy_goose.supersede import (
    SUPERSEDE_VERSION,
    apply_supersession,
    attribute_results,
    find_superseded,
)


class _Builder:
    def __init__(self) -> None:
        self.segments: list[Segment] = []
        self.turn = 0

    def add(
        self, kind: SegmentKind, text: str, role: str = "assistant", tool_name: str | None = None
    ) -> Segment:
        seg = Segment(
            len(self.segments),
            self.turn,
            role,  # type: ignore[arg-type]
            kind,
            text,
            kind != "prose",
            [],
            tool_name=tool_name,
        )
        self.segments.append(seg)
        return seg

    def next_turn(self) -> None:
        self.turn += 1

    def call(self, tool_name: str | None = None, /, **payload: object) -> Segment:
        return self.add("tool_use", json.dumps(payload, sort_keys=True), tool_name=tool_name)

    def result(self, text: str, kind: SegmentKind = "tool_result") -> Segment:
        self.next_turn()
        seg = self.add(kind, text, role="user")
        self.next_turn()
        return seg


def test_two_edits_same_path_first_is_superseded() -> None:
    b = _Builder()
    first = b.call(file_path="a.py", old_string="x", new_string="y")
    b.result("ok")
    second = b.call(file_path="a.py", old_string="y", new_string="z")
    b.result("ok")
    other = b.call(file_path="b.py", old_string="p", new_string="q")
    b.result("ok")
    gone = find_superseded(b.segments)
    assert gone[first.id].reason == "edit_superseded"
    assert gone[first.id].by == second.id
    assert second.id not in gone
    assert other.id not in gone


def test_write_then_edit_same_path_write_is_superseded() -> None:
    b = _Builder()
    write = b.call(file_path="a.py", content="x = 1\n")
    b.result("wrote")
    edit = b.call(file_path="a.py", old_string="1", new_string="2")
    b.result("edited")
    gone = find_superseded(b.segments)
    assert gone[write.id] == gone[write.id].__class__("edit_superseded", edit.id)


def test_read_then_edit_same_path_read_and_its_output_are_superseded() -> None:
    b = _Builder()
    read = b.call(file_path="a.py")
    dump = b.result("     1→x = 1\n     2→y = 2\n     3→z = 3", kind="code")
    edit = b.call(file_path="a.py", old_string="1", new_string="2")
    b.result("edited")
    gone = find_superseded(b.segments)
    assert gone[read.id].reason == "read_superseded" and gone[read.id].by == edit.id
    assert gone[dump.id].reason == "read_superseded" and gone[dump.id].by == edit.id
    assert edit.id not in gone


def test_read_after_last_write_is_kept() -> None:
    b = _Builder()
    edit = b.call(file_path="a.py", old_string="1", new_string="2")
    b.result("edited")
    read = b.call(file_path="a.py")
    dump = b.result("     1→x = 2")
    gone = find_superseded(b.segments)
    assert read.id not in gone and dump.id not in gone and edit.id not in gone


def test_exact_duplicates_keep_last_occurrence() -> None:
    b = _Builder()
    a = b.add("prose", "Running the tests now.")
    b.next_turn()
    b.add("prose", "Something else entirely.")
    b.next_turn()
    c = b.add("prose", "Running the tests now.  ")
    gone = find_superseded(b.segments)
    assert gone[a.id].reason == "duplicate" and gone[a.id].by == c.id
    assert c.id not in gone
    assert len(gone) == 1


def test_unrelated_segments_untouched_and_apply_preserves_order() -> None:
    b = _Builder()
    b.add("prose", "Plan first.")
    b.call(command="ls")
    b.result("a.py\nb.py")
    b.call(file_path="a.py", old_string="x", new_string="y")
    b.result("ok")
    b.add("prose", "Done.")
    gone = find_superseded(b.segments)
    assert gone == {}
    kept = apply_supersession(b.segments)
    assert [s.id for s in kept] == [s.id for s in b.segments]


def test_tool_names_override_payload_shape_inference() -> None:
    b = _Builder()
    # Payload shape alone says "read" (path only); a Delete-like tool name must not become a Read.
    odd = b.call(file_path="a.py", offset=0)
    b.result("deleted")
    read = b.call(file_path="a.py")
    b.result("     1→x")
    gone = find_superseded(b.segments, tool_names={odd.id: "Delete", read.id: "Read"})
    assert odd.id not in gone
    gone_default = find_superseded(b.segments)
    assert gone_default[odd.id].reason == "read_superseded"


def test_attribute_results_single_and_matched_calls() -> None:
    b = _Builder()
    c1 = b.call(file_path="a.py")
    c2 = b.call(file_path="b.py")
    b.next_turn()
    r1 = b.add("tool_result", "a", role="user")
    r2 = b.add("tool_result", "b", role="user")
    b.next_turn()
    c3 = b.call(command="ls")
    b.next_turn()
    r3 = b.add("tool_result", "x", role="user")
    r4 = b.add("tool_result", "y", role="user")
    owner = attribute_results(b.segments)
    assert owner == {r1.id: c1.id, r2.id: c2.id, r3.id: c3.id, r4.id: c3.id}


def test_attribute_results_ambiguous_count_is_left_unattributed() -> None:
    b = _Builder()
    b.call(file_path="a.py")
    b.call(file_path="b.py")
    b.next_turn()
    b.add("tool_result", "a", role="user")
    b.add("tool_result", "b", role="user")
    b.add("tool_result", "c", role="user")
    assert attribute_results(b.segments) == {}


def test_supersede_version_is_two() -> None:
    assert SUPERSEDE_VERSION == 2


def test_stored_tool_name_wins_where_positional_walk_would_have_drifted() -> None:
    # The transcript turn carried [Read, Delete] but the loader kept only the second call, so the
    # v1 positional walk would have paired this lone segment with "Read" and superseded it. The
    # segment's own name says Delete, and a Delete is never a stale read.
    b = _Builder()
    delete = b.call("Delete", file_path="a.py", offset=0)
    b.result("deleted")
    read = b.call("Read", file_path="a.py")
    b.result("     1→x")
    gone = find_superseded(b.segments)
    assert delete.id not in gone
    assert read.id not in gone


def test_stored_tool_name_classifies_a_call_payload_shape_cannot() -> None:
    b = _Builder()
    # `pages` alone is read-shaped, but with a body key the shape rule says write; the stored name
    # is what makes it a read.
    read = b.call("Read", file_path="a.pdf", pages="1-3", content="ignored")
    dump = b.result("page 1")
    write = b.call("Write", file_path="a.pdf", content="new")
    b.result("wrote")
    gone = find_superseded(b.segments)
    assert gone[read.id].reason == "read_superseded" and gone[read.id].by == write.id
    assert gone[dump.id].reason == "read_superseded"
    assert write.id not in gone


def test_explicit_tool_names_override_stored_name() -> None:
    b = _Builder()
    odd = b.call("Read", file_path="a.py", offset=0)
    b.result("deleted")
    b.call("Read", file_path="a.py")
    b.result("     1→x")
    assert find_superseded(b.segments)[odd.id].reason == "read_superseded"
    assert odd.id not in find_superseded(b.segments, tool_names={odd.id: "Delete"})


def test_openhands_xml_payloads_participate_in_supersession() -> None:
    b = _Builder()
    view = b.add(
        "tool_use",
        "<parameter=command>view</parameter>\n<parameter=path>/w/a.py</parameter>",
    )
    dump = b.result("1\tx = 1\n2\ty = 2")
    edit = b.add(
        "tool_use",
        "<parameter=command>str_replace</parameter>\n<parameter=path>/w/a.py</parameter>\n"
        "<parameter=old_str>x = 1</parameter>\n<parameter=new_str>x = 2</parameter>",
    )
    gone = find_superseded(b.segments)
    assert gone[view.id].reason == "read_superseded" and gone[view.id].by == edit.id
    assert gone[dump.id].reason == "read_superseded"
    assert edit.id not in gone

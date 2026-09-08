from loosy_goose.segment import (
    extract_atoms,
    looks_like_code_dump,
    segment,
    split_prose,
    split_sentences,
)
from loosy_goose.transcript import Block, Transcript, Turn


def _transcript(*blocks: Block) -> Transcript:
    return Transcript("t", [Turn(0, "assistant", list(blocks))])


def test_split_sentences_basic() -> None:
    assert split_sentences("First one. Second one! Third? Yes.") == [
        "First one.",
        "Second one!",
        "Third?",
        "Yes.",
    ]


def test_split_sentences_does_not_break_on_dotted_identifiers() -> None:
    assert split_sentences("Call foo.bar() then run np.sum here. Done.") == [
        "Call foo.bar() then run np.sum here.",
        "Done.",
    ]


def test_split_prose_paragraphs_then_merge_sentences() -> None:
    para_a = "Short paragraph."
    long_sentences = " ".join(f"Sentence number {i} is here." for i in range(20))
    chunks = split_prose(f"{para_a}\n\n{long_sentences}", max_chars=80)
    assert chunks[0] == para_a
    assert all(len(c) <= 80 for c in chunks[1:])
    assert " ".join(chunks[1:]) == long_sentences


def test_code_and_tool_use_are_atomic_and_protected() -> None:
    long_code = "\n".join(f"x{i} = {i}" for i in range(200))
    segs = segment(_transcript(Block("code", long_code), Block("tool_use", '{"a": 1}')))
    assert [s.kind for s in segs] == ["code", "tool_use"]
    assert all(s.protected for s in segs)
    assert segs[0].text == long_code


def test_tool_result_prose_is_chunked_and_unprotected() -> None:
    lines = "\n".join(f"Log line {i}: everything is fine so far" for i in range(50))
    segs = segment(_transcript(Block("tool_result", lines)), max_prose_chars=200)
    assert len(segs) > 1
    assert all(s.kind == "tool_result" and not s.protected for s in segs)
    assert "\n".join(s.text for s in segs) == lines


def test_tool_result_code_dump_becomes_protected_code() -> None:
    dump = "\n".join(
        [
            "def main():",
            "    x = load()",
            "    return x",
            "",
            "class Foo:",
            "    pass",
        ]
    )
    assert looks_like_code_dump(dump)
    segs = segment(_transcript(Block("tool_result", dump)))
    assert len(segs) == 1
    assert segs[0].kind == "code" and segs[0].protected


def test_plain_log_is_not_a_code_dump() -> None:
    log = "\n".join(["Starting server", "Listening on port", "Ready to accept connections"])
    assert not looks_like_code_dump(log)


def test_extract_atoms_examples() -> None:
    text = (
        "Edit src/loosy_goose/segment.py and call `extract_atoms`; run_job failed 42 times, "
        "see https://example.com/x?y=1 and setUserName in config.toml after 3.14 seconds."
    )
    atoms = extract_atoms(text)
    for expected in (
        "src/loosy_goose/segment.py",
        "extract_atoms",
        "run_job",
        "42",
        "https://example.com/x?y=1",
        "setUserName",
        "config.toml",
        "3.14",
    ):
        assert expected in atoms
    assert "Edit" not in atoms
    assert "times" not in atoms


def test_segment_ids_and_turns() -> None:
    t = Transcript(
        "t",
        [
            Turn(0, "user", [Block("text", "Hello there.")]),
            Turn(1, "assistant", [Block("text", "Hi.\n\nSecond paragraph.")]),
        ],
    )
    segs = segment(t)
    assert [s.id for s in segs] == [0, 1, 2]
    assert [s.turn for s in segs] == [0, 1, 1]
    assert [s.role for s in segs] == ["user", "assistant", "assistant"]

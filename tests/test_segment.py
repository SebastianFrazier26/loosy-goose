from loosy_goose.segment import (
    _ATOM_PATTERNS,
    ATOMS_VERSION,
    extract_atoms,
    looks_like_code_dump,
    scannable,
    segment,
    split_prose,
    split_sentences,
    strip_line_numbers,
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


def test_numbered_read_routes_on_its_body_and_keeps_prefixes() -> None:
    # Single-digit prefixes: `1→def main():` fails the code-line regex and carries no indent, so
    # the prefixed text was never a code dump — only the stripped body is.
    dump = "\n".join(
        [
            "1→def main():",
            "2→    x = load()",
            "3→    return x",
            "4→",
            "5→class Foo:",
            "6→    pass",
        ]
    )
    assert not looks_like_code_dump(dump)
    segs = segment(_transcript(Block("tool_result", dump)))
    assert len(segs) == 1
    assert segs[0].kind == "code" and segs[0].protected
    assert segs[0].text == dump


def test_numbered_prose_is_not_a_code_dump() -> None:
    # Right-aligned numbers pad every line past the four-space indentation fallback; before the
    # strip that alone routed a numbered prose file to `code`.
    dump = "\n".join(
        [
            "     1→The quick brown fox jumps over the lazy dog.",
            "     2→Nothing here resembles a statement.",
            "     3→It is plain prose with line numbers.",
            "     4→Which is what a numbered README read looks like.",
        ]
    )
    assert looks_like_code_dump(dump)
    segs = segment(_transcript(Block("tool_result", dump)))
    assert all(s.kind == "tool_result" and not s.protected for s in segs)
    assert "\n".join(s.text for s in segs) == dump


def test_strip_line_numbers_requires_a_numbered_majority() -> None:
    assert strip_line_numbers("Log line 1\nLog line 2\nLog line 3") is None
    stripped = strip_line_numbers("1→a\n2→b\n3→c")
    assert stripped == (["1→", "2→", "3→"], ["a", "b", "c"])


def test_atoms_version() -> None:
    assert ATOMS_VERSION == 5


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


def test_split_prose_drops_markdown_horizontal_rules() -> None:
    text = "First part.\n\n---\n\nSecond part.\n\n***\n\nThird part."
    assert split_prose(text, 600) == ["First part.", "Second part.", "Third part."]


def test_split_prose_keeps_paragraphs_that_merely_start_with_punctuation() -> None:
    assert split_prose("- item one\n\n- item two", 600) == ["- item one", "- item two"]


def test_split_prose_bounds_paragraphs_with_no_sentence_punctuation() -> None:
    para = " ".join(f"word{i}" for i in range(400))
    chunks = split_prose(para, 100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == para.split()


def test_split_prose_bounds_a_run_with_no_whitespace_at_all() -> None:
    chunks = split_prose("x" * 500, 100)
    assert chunks == ["x" * 100] * 5


def test_chunk_lines_bounds_a_single_overlong_line() -> None:
    long_line = " ".join(f"tok{i}" for i in range(300))
    block = Block("tool_result", f"short line\n{long_line}\ntail line")
    segs = segment(_transcript(block), max_prose_chars=120)
    assert all(len(s.text) <= 120 for s in segs)
    assert any("tok0" in s.text for s in segs)
    assert any("tail line" in s.text for s in segs)


def test_scannable_decodes_a_json_payload_to_values_only() -> None:
    payload = r'{"file_path": "app/main.py", "new_string": "import os\nimport sys", "limit": 2000}'
    scanned = scannable(payload)
    assert "import os\nimport sys" in scanned
    assert "file_path" not in scanned
    assert "2000" in scanned


def test_scannable_leaves_non_json_text_alone() -> None:
    for text in ("plain prose about app/main.py", "{not json at all", '["a", "b"]'):
        assert scannable(text) == text


def test_escaped_newlines_are_not_file_paths() -> None:
    # The defect this guards: json.dumps renders an edit body's newlines as a literal
    # backslash-n, and the path pattern read that backslash as a separator, inventing path
    # atoms out of consecutive escapes. 8.7% of the corpus's distinct atoms were this.
    payload = r'{"new_string": "return 1;\n\nstatic_thing = 2;\n"}'
    atoms = extract_atoms(payload)
    assert not any("\\" in a for a in atoms)
    # And the identifier the escape used to be glued to survives intact.
    assert "static_thing" in atoms


def test_real_windows_path_inside_a_payload_is_an_atom() -> None:
    # The mirror-image defect: a genuine backslash is doubled by the same serializer, and the
    # pattern could not span the empty component, so these paths were invisible entirely.
    payload = r'{"file_path": "venv\\Scripts\\pip.exe"}'
    assert "venv\\Scripts\\pip.exe" in extract_atoms(payload)


def test_nested_payload_values_are_scanned() -> None:
    payload = '{"edits": [{"old_string": "a/b/old.py", "new_string": "a/b/new.py"}]}'
    atoms = extract_atoms(payload)
    assert "a/b/old.py" in atoms
    assert "a/b/new.py" in atoms


def test_a_windows_path_keeps_its_drive_letter() -> None:
    # ATOMS_VERSION 3. The old pattern needed a word character before its first separator, so it
    # matched from `repo` onward and every drive-qualified path was silently truncated.
    atoms = extract_atoms(r"Open D:\BentoFolio\vercel.json and check it.")
    assert "D:\\BentoFolio\\vercel.json" in atoms
    assert "BentoFolio\\vercel.json" not in atoms
    assert "C:/repo/src/main.py" in extract_atoms("see C:/repo/src/main.py")


def test_constant_case_identifiers_are_atoms() -> None:
    # The snake pattern accepts lowercase only, so env vars and config keys were invisible.
    atoms = extract_atoms("Set TRAIN_RATIO and API_CHANGES before the run.")
    assert "TRAIN_RATIO" in atoms
    assert "API_CHANGES" in atoms


def test_acronym_led_identifiers_are_atoms_but_bare_plurals_are_not() -> None:
    atoms = extract_atoms("DOMContentLoaded fires; CMakeLists lists the URLs and IDs.")
    assert "DOMContentLoaded" in atoms
    assert "CMakeLists" in atoms
    # A pluralised acronym identifies nothing, and the {2,} tail is what keeps it out.
    assert "URLs" not in atoms
    assert "IDs" not in atoms


def test_c_family_and_build_file_extensions_are_recognised() -> None:
    atoms = extract_atoms("Build epilog_ffi.cpp into epilog.exe, per build.gradle.")
    assert {"epilog_ffi.cpp", "epilog.exe", "build.gradle"} <= set(atoms)


def test_an_attribute_colliding_extension_is_a_filename_only_inside_a_path() -> None:
    # ATOMS_VERSION 4. The v3 widening made every one of these register as a filename; on the
    # corpus those same extensions also carry real files (1,510 `.h`, 731 `.c`), and nothing in
    # prose tells the two apart. The path forms below are what survives.
    assert extract_atoms("self.c and obj.h and df.a and x.o and node.so") == []
    assert "src/main.c" in extract_atoms("edit src/main.c now")
    assert "C:\\repo\\main.c" in extract_atoms(r"open C:\repo\main.c please")
    assert "build/solver.o" in extract_atoms("link build/solver.o in")


def test_the_path_requirement_covers_only_the_seven_colliding_extensions() -> None:
    # The criterion is collision with attribute access, not extension length. These are the
    # two-character entries that name files rather than attributes — 1,633 corpus mentions
    # between them — and narrowing by length instead would have silently dropped every one.
    text = "open main.py, README.md, build.sh, main.js, index.ts, App.kt, lib.rs and mod.go"
    atoms = set(extract_atoms(text))
    assert {"main.py", "README.md", "build.sh", "main.js", "index.ts", "lib.rs"} <= atoms


def test_a_second_extension_is_consumed_rather_than_rejected() -> None:
    # The chain rule must not fire on a double extension: rejecting `node.tar.gz` outright left
    # no atom at all, which is worse than the truncated `node.tar` it replaced.
    assert "node.tar.gz" in extract_atoms("grab node.tar.gz now")
    assert "build.gradle.kts" in extract_atoms("build.gradle.kts here")
    assert "device.svelte.ts" in extract_atoms("device.svelte.ts changed")
    # And the exemption is not a hole in the chain rule: the tail has to be an allowlisted
    # extension, which `NODE_ENV`, `Types` and `solver` are not.
    assert "process.env" not in extract_atoms("process.env.NODE_ENV is read")
    assert "java.sql" not in extract_atoms("catch java.sql.Types here")
    assert "self.cfg" not in extract_atoms("read self.cfg.solver now")


def test_a_dotted_chain_is_not_read_as_a_filename() -> None:
    # The rule that kills `process.env.NODE_ENV`, and it applies to every extension, not only the
    # short ones: a match followed by a dot and an identifier character is attribute access.
    assert "process.env" not in extract_atoms("process.env.NODE_ENV is read")
    assert "NODE_ENV" in extract_atoms("process.env.NODE_ENV is read")
    assert "java.sql" not in extract_atoms("catch java.sql.Types here")
    assert "self.cfg" not in extract_atoms("read self.cfg.solver now")
    # `env` is out of the allowlist as well, so the unchained form does not slip through.
    assert extract_atoms("values come from process.env only") == []


def test_paths_module_shares_the_bare_filename_pattern() -> None:
    # PHASE2's invariant: the path table guarantees whatever it names and the metric credits only
    # what the extractor recognises, so a divergence would let the table claim uncounted atoms.
    from loosy_goose.paths import _PATH_BARE, _PATH_SLASH

    assert _PATH_BARE.pattern == _ATOM_PATTERNS[2].pattern
    assert _PATH_SLASH.pattern == _ATOM_PATTERNS[1].pattern


def test_atoms_are_in_first_appearance_order_and_deduplicated() -> None:
    # Determinism: extraction is an insertion-ordered dict, never a set.
    assert extract_atoms("run_job then setUserName then run_job again") == [
        "run_job",
        "setUserName",
    ]


def test_version_3_behaviours_survive_the_version_4_extension_rules() -> None:
    # One guard over the four v3 wins, so a later tightening of the extension rules cannot drop
    # them quietly: they live in separate patterns and none of them should have moved.
    assert "C:\\repo\\src\\budget.py" in extract_atoms(r"see C:\repo\src\budget.py")
    assert "TRAIN_RATIO" in extract_atoms("Set TRAIN_RATIO now.")
    acronyms = extract_atoms("DOMContentLoaded and IOError and XMLHttpRequest, not URLs or IDs")
    assert {"DOMContentLoaded", "IOError", "XMLHttpRequest"} <= set(acronyms)
    assert not {"URLs", "IDs"} & set(acronyms)
    ticked = extract_atoms("`prettier --write conf/site.json`")
    assert "prettier --write conf/site.json" not in ticked
    assert {"prettier", "--write", "conf/site.json"} <= set(ticked)
    assert extract_atoms("console.log(x) runs, e.g. once") == []


def test_attribute_access_is_not_read_as_a_filename() -> None:
    # Why the extension list is an allowlist. `.log` and `.g` were measured on the corpus as
    # dominated by these two shapes, not by files, and are deliberately absent from it.
    atoms = extract_atoms("Then console.log(x) runs, e.g. once, and urls.Count is checked.")
    assert "console.log" not in atoms
    assert "e.g" not in atoms


def test_a_backticked_name_is_kept_whole() -> None:
    atoms = extract_atoms("Call `extract_atoms` on it.")
    assert "extract_atoms" in atoms


def test_a_backticked_command_is_split_rather_than_taken_whole() -> None:
    # Taking the span whole made one atom of an entire command line, so recall demanded exact
    # reproduction of a long string instead of asking whether the identifier survived.
    atoms = extract_atoms("Run `prettier --write conf/site.json` now.")
    assert "prettier --write conf/site.json" not in atoms
    assert "prettier" in atoms  # first token: the command itself
    assert "--write" in atoms  # carries a flag mark
    assert "conf/site.json" in atoms


def test_a_backticked_english_phrase_contributes_only_its_first_word() -> None:
    # The known and accepted cost of the rule: `run` survives, the prose after it does not.
    atoms = extract_atoms("The button says `save the file` in the UI.")
    assert "the" not in atoms
    assert "file" not in atoms

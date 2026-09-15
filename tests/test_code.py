import json

import pytest

from loosy_goose.code import (
    _MAX_ARG_DEPTH,
    ELISION,
    PROSE_UNIT_TOKENS,
    SHELL_COMMAND_TOKENS,
    _percentile_budget,
    _shrink_shell,
    band_lines,
    detect_language,
    normalize_language,
    parse_tool_payload,
    quantize_code,
    quantize_segment,
    quantize_tool_use,
    shell_budget,
    shrink_prose,
    split_shell,
)
from loosy_goose.segment import Segment, extract_atoms
from loosy_goose.tokens import count_tokens

PY = '''import os
from x import y

CONST = 3


@decorator
def f(a, b):
    """Doc."""
    total = 0
    for i in range(a):
        if i % 2:
            total += i
        else:
            total -= 1
    return total


class K(Base):
    def m(self):
        y = self.x
        return y
'''

TS = """import { x } from './y';
export class A extends B {
  private n: number = 1;
  constructor(a: string) {
    super();
    if (a) {
      this.n = 2;
    }
  }
}
export const K = 3;
"""


# Long enough that eliding the body saves tokens; one- or two-line bodies are kept on purpose.
BODY = (
    "def g(a):\n"
    "    first_value = compute_first(a)\n"
    "    second_value = compute_second(first_value)\n"
    "    third_value = combine(first_value, second_value)\n"
    "    return third_value"
)


def _lines(text: str) -> list[str]:
    return text.split("\n")


def _kept_lines(text: str) -> set[str]:
    return {ln for ln in text.split("\n") if "lines elided" not in ln}


def test_band_lines_python_signatures_are_band_zero() -> None:
    bands = dict(zip(_lines(PY), band_lines(PY, "python"), strict=True))
    for sig in ("import os", "from x import y", "CONST = 3", "@decorator", "def f(a, b):"):
        assert bands[sig] == 0, sig
    assert bands["class K(Base):"] == 0
    assert bands["    def m(self):"] == 0
    assert bands['    """Doc."""'] == 0
    assert bands["    return total"] == 0
    assert bands["        return y"] == 0


def test_band_lines_python_bodies_deepen_with_nesting() -> None:
    bands = dict(zip(_lines(PY), band_lines(PY, "python"), strict=True))
    assert bands["    total = 0"] == 1
    assert bands["    for i in range(a):"] == 1
    assert bands["        if i % 2:"] == 2
    assert bands["            total += i"] == 2
    assert bands["        y = self.x"] == 2
    assert bands["    total = 0"] < bands["            total += i"]


def test_band_lines_typescript_parses_and_bands() -> None:
    bands = dict(zip(_lines(TS), band_lines(TS, "typescript"), strict=True))
    assert bands["import { x } from './y';"] == 0
    assert bands["export class A extends B {"] == 0
    assert bands["  private n: number = 1;"] == 0
    assert bands["  constructor(a: string) {"] == 0
    assert bands["    super();"] == 2
    assert bands["      this.n = 2;"] == 3
    # Closers take their opener's band so brace pairs survive or vanish together.
    assert bands["    }"] == bands["    if (a) {"]
    assert bands["  }"] == 0
    assert bands["export const K = 3;"] == 0


def test_band_lines_unknown_language_falls_back_to_indentation() -> None:
    text = "top\n  child\n    grandchild\ntop again"
    assert band_lines(text, None) == [0, 1, 2, 0]
    assert band_lines(text, "no-such-language") == [0, 1, 2, 0]


def test_quantize_code_is_monotone_in_quality() -> None:
    previous: set[str] | None = None
    for q in (0.0, 0.33, 0.66, 1.0):
        kept = _kept_lines(quantize_code(PY, "python", q))
        if previous is not None:
            assert previous <= kept, q
        previous = kept
    assert quantize_code(PY, "python", 1.0) == PY


def test_quantize_code_zero_quality_is_signatures_only() -> None:
    out = quantize_code(PY, "python", 0.0)
    assert "def f(a, b):" in out
    assert "    return total" in out
    assert "total += i" not in out
    assert "for i in range(a):" not in out


def test_elision_marker_counts_lines_and_keeps_indent() -> None:
    text = "def f():\n    a = 1\n    b = 2\n    c = 3\n    return a\n"
    out = quantize_code(text, "python", 0.0)
    assert out.split("\n") == ["def f():", "    " + ELISION.format(n=3), "    return a", ""]


def test_quantize_code_preserves_read_style_line_numbers() -> None:
    numbered = ["     1→import os", "     2→def f():", "     3→    if x:", "     4→        y()"]
    text = "\n".join([*numbered, "     5→    return 1"])
    out = quantize_code(text, "python", 0.0)
    assert out.startswith("     1→import os\n     2→def f():\n")
    assert "     5→    return 1" in out
    assert "y()" not in out


def test_quantize_code_rejects_bad_quality() -> None:
    with pytest.raises(ValueError):
        quantize_code("x = 1", "python", 1.5)


def test_quantize_tool_use_drops_old_string_and_keeps_path() -> None:
    payload = {
        "file_path": "src/a.py",
        "old_string": "x = 1",
        "new_string": BODY,
        "replace_all": False,
    }
    out = quantize_tool_use(json.dumps(payload), 0.0, "Edit")
    # The target heads the call line, and the rendering is text rather than JSON on purpose.
    assert out.startswith("src/a.py")
    assert "old_string" not in out
    assert "replace_all: false" in out
    assert "new_string: def g(a):" in out
    assert "compute_second" not in out
    assert "return third_value" in out
    assert "lines elided" in out


def test_render_call_spends_nothing_on_json_scaffolding() -> None:
    payload = {"file_path": "a.py", "new_string": "one\ntwo"}
    out = quantize_tool_use(json.dumps(payload), 0.5, "Write")
    assert "{" not in out and '"' not in out
    # The expensive half of json.dumps is escaping: a newline cost two characters there.
    assert "\\n" not in out
    assert out.count("\n") >= 2


def test_the_rendered_call_carries_no_tool_name() -> None:
    # The name reaches the shrinker from the transcript record's metadata, never from the
    # segment's text, so emitting it was neither a verbatim span nor a defined pointer.
    text = json.dumps({"file_path": "src/a.py", "content": "x = 1"})
    out = quantize_tool_use(text, 0.5, "MultiEdit")
    assert "MultiEdit" not in out
    assert out.startswith("src/a.py")
    # A payload with no path target was headed by the name alone, so it is the sharper case.
    bare = quantize_tool_use(json.dumps({"command": "ls | wc"}), 0.5, "NotebookEdit")
    assert "NotebookEdit" not in bare


def test_rendered_atoms_are_a_subset_of_the_original_atoms() -> None:
    # The metric-facing half of the same decision, asserted directly. `MultiEdit` matches the
    # identifier pattern, so a prepended name entered the rendered segment's atoms — and recall
    # intersects against the atoms of the ORIGINAL text, which never held it, so those tokens
    # were spent where no recall could return them.
    # The keys here are single words on purpose: `file_path` and `new_string` are atoms too, and
    # they are the separate, argued exception (they disambiguate values). This pins the name.
    text = json.dumps({"content": BODY, "description": "rewrites module_a.py"})
    original = set(extract_atoms(text))
    rendered = set(extract_atoms(quantize_tool_use(text, 0.0, "MultiEdit")))
    assert rendered, "a vacuous subset would pass whatever the renderer emitted"
    assert rendered <= original


def test_quantize_tool_use_quality_one_is_identity() -> None:
    text = json.dumps({"file_path": "a.py", "old_string": "a", "new_string": "b"})
    assert quantize_tool_use(text, 1.0) == text


def test_an_unnamed_tool_is_not_assumed_to_be_a_shell() -> None:
    # Without a tool name that `|` could be an alternation, so the stages are left alone.
    text = json.dumps({"command": "ls -la\n  | grep x", "description": "list"}, sort_keys=True)
    out = quantize_tool_use(text, 0.0)
    assert "ls -la" in out and "grep x" in out


def test_split_shell_finds_stages_and_respects_quotes() -> None:
    assert len(split_shell("git status")) == 1
    assert len(split_shell("cat x | grep y | head -5")) == 3
    assert len(split_shell("a && b || c ; d")) == 4
    # A separator inside an argument is not a boundary, which is why the scanner tracks quotes.
    assert len(split_shell('git commit -m "one | two && three"')) == 1
    assert len(split_shell("grep 'a;b' file")) == 1


def test_shell_budget_is_a_percentile_of_real_command_lengths() -> None:
    # The point of the table: the cut length is measured, not chosen. Quality names how long a
    # command we still keep, as a percentile of commands that needed no pipeline.
    assert shell_budget(0.5) == SHELL_COMMAND_TOKENS[5]
    assert shell_budget(0.0) == SHELL_COMMAND_TOKENS[0]
    assert shell_budget(1.0) == SHELL_COMMAND_TOKENS[-1]
    # Monotone, so a higher quality never keeps less.
    budgets = [shell_budget(q / 20) for q in range(21)]
    assert budgets == sorted(budgets)


def test_shell_command_is_cut_at_the_budget_not_at_a_stage_fraction() -> None:
    command = " | ".join(f"stage{i} --flag value{i} --other setting{i}" for i in range(8))
    out = quantize_tool_use(json.dumps({"command": command}), 0.0, "Bash")
    assert "command: stage0 --flag value0" in out  # stage 0 survives, as band 0 does for code
    assert "stage7" not in out
    assert "stages elided" in out


def test_a_short_pipeline_inside_the_budget_is_left_alone() -> None:
    text = json.dumps({"command": "ls | wc"})
    # A call with no path target is the bare argument line: 5 tokens against the payload's 8.
    # While the tool name was prepended this rendered to exactly 8 and survived only because the
    # size guard admitted ties, which is the tie the guard's `>` sign existed for.
    assert quantize_tool_use(text, 0.5, "Bash") == "command: ls | wc"
    assert count_tokens("command: ls | wc") < count_tokens(text)


def test_a_pipeline_is_kept_whole_when_the_marker_costs_what_it_replaces() -> None:
    # `_elide`'s guard, applied to stages: replacing two short stages with "… (2 stages elided)"
    # is pure loss, so the command survives even below the budget.
    text = json.dumps({"command": "echo one | grep two | sort | uniq -c | head -20"})
    assert quantize_tool_use(text, 0.0, "Bash").endswith("head -20")


def test_a_pipe_in_a_non_shell_argument_is_not_a_stage_boundary() -> None:
    # Gating on the tool, not on the value: `|` is a pipe for Bash and an alternation for Grep.
    text = json.dumps({"pattern": "foo|bar|baz|qux|quux|corge|grault|garply|waldo|fred"})
    assert quantize_tool_use(text, 0.0, "Grep").count("|") == 9


def test_arguments_outside_the_old_body_allowlist_are_now_shrunk() -> None:
    # The hole this closed: `command` was 24.5% of the corpus's tool_use tokens and passed
    # through a bare `else` verbatim, because it is not one of the four named body keys.
    long_command = " | ".join(f"step{i} --flag value{i}" for i in range(12))
    text = json.dumps({"command": long_command})
    assert len(quantize_tool_use(text, 0.0, "Bash")) < len(text)


def test_target_path_is_never_shrunk() -> None:
    text = json.dumps({"file_path": "a/b/c/very/long/path/to/module.py", "content": BODY})
    out = quantize_tool_use(text, 0.0, "Write")
    assert out.startswith("a/b/c/very/long/path/to/module.py")


def test_string_leaves_inside_nested_arguments_are_shrunk() -> None:
    # "Keep the shape, shrink the leaves": the container survives, its strings are routed.
    text = json.dumps({"edits": [{"new_string": BODY}, {"new_string": BODY}]})
    out = quantize_tool_use(text, 0.0, "MultiEdit")
    assert out.count("def g(a):") == 2  # container survived, both leaves shrunk
    assert "compute_second" not in out


def test_nested_arguments_stop_at_the_depth_bound() -> None:
    deep: dict[str, object] = {"new_string": BODY}
    for _ in range(_MAX_ARG_DEPTH + 2):
        deep = {"level": [deep]}
    # Must terminate and stay valid rather than recursing without end.
    assert quantize_tool_use(json.dumps(deep), 0.0, "Edit").strip()


def test_tool_quality_overrides_quality_for_tool_use_only() -> None:
    call = Segment(0, 0, "assistant", "tool_use", json.dumps({"content": BODY}), True, [], "Write")
    code = Segment(1, 0, "assistant", "code", BODY, True, [])
    # Full global quality, aggressive tool depth: the call shrinks, the code segment does not.
    assert len(quantize_segment(call, 1.0, tool_quality=0.0).text) < len(call.text)
    assert quantize_segment(code, 1.0, tool_quality=0.0).text == code.text


def test_quantize_tool_use_non_json_is_treated_as_code() -> None:
    out = quantize_tool_use(BODY, 0.0)
    assert "def g(a):" in out and "compute_second" not in out


def test_elision_keeps_runs_cheaper_than_the_marker() -> None:
    text = "def f():\n    a = 1\n    return a"
    assert quantize_code(text, "python", 0.0) == text


def test_normalize_language_aliases() -> None:
    assert normalize_language("py") == "python"
    assert normalize_language("c#") == "csharp"
    assert normalize_language("TS") == "typescript"
    assert normalize_language("text") is None
    assert normalize_language("klingon") is None


def test_detect_language_hint_then_path_then_heuristic() -> None:
    seg = Segment(0, 0, "assistant", "code", "x = 1", True, [])
    assert detect_language(seg, hint="js") == "javascript"
    with_path = Segment(1, 0, "assistant", "code", "see src/app.cs", True, ["src/app.cs"])
    assert detect_language(with_path) == "csharp"
    py = Segment(2, 0, "assistant", "code", "def f():\n    pass", True, [])
    assert detect_language(py) == "python"
    tool = Segment(3, 0, "assistant", "tool_use", '{"a": 1}', True, [])
    assert detect_language(tool) == "json"
    prose = Segment(4, 0, "assistant", "code", "just some words here", True, [])
    assert detect_language(prose) is None


def test_quantize_segment_returns_new_unprotected_segment_with_atoms() -> None:
    seg = Segment(7, 3, "assistant", "code", BODY, True, extract_atoms(BODY))
    out = quantize_segment(seg, 0.0)
    assert out is not seg
    assert (out.id, out.turn, out.role, out.kind) == (7, 3, "assistant", "code")
    assert out.protected is False
    assert "third_value" in out.atoms
    assert "compute_second" not in out.atoms
    assert seg.text == BODY


def test_parse_tool_payload_reads_json_and_openhands_xml() -> None:
    assert parse_tool_payload('{"file_path": "a.py"}') == {"file_path": "a.py"}
    xml = "<parameter=command>view</parameter>\n<parameter=path>/w/a.py</parameter>"
    assert parse_tool_payload(xml) == {"command": "view", "path": "/w/a.py"}
    assert parse_tool_payload("not a payload") is None
    assert parse_tool_payload("[1, 2]") is None


def test_quantize_tool_use_openhands_payload_keeps_shape_and_drops_old_str() -> None:
    xml = (
        "<parameter=command>str_replace</parameter>\n"
        "<parameter=path>/w/a.py</parameter>\n"
        "<parameter=old_str>x = 1</parameter>\n"
        f"<parameter=new_str>{BODY}</parameter>"
    )
    out = quantize_tool_use(xml, 0.0)
    assert out.startswith("<parameter=command>str_replace</parameter>")
    assert "<parameter=path>/w/a.py</parameter>" in out
    assert "old_str" not in out
    assert "def g(a):" in out and "compute_second" not in out


@pytest.mark.slow
def test_quantized_code_stays_semantically_close_to_original() -> None:
    pytest.importorskip("sentence_transformers")
    from loosy_goose.metrics import semantic_coverage

    seg = Segment(0, 0, "assistant", "code", PY, True, extract_atoms(PY))
    low = quantize_segment(seg, 0.0)
    cov = semantic_coverage([seg], [low])
    assert cov["coverage"] > 0.6


PARAGRAPHS = "\n\n".join(
    f"Passage {i} makes a claim about topic {i} and names module_{i}.py along the way. "
    f"It then runs on long enough that the passage budget has somewhere to bite."
    for i in range(8)
)
NEWLINE_PROSE = "\n".join(
    f"Line {i} states a fact worth keeping and mentions symbol_{i}." for i in range(20)
)
QUALITIES = (0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.99)


def _head(out: str, original: str) -> str:
    """The retained span, with the elision marker (always the last line) removed."""
    return original if out == original else out.rsplit("\n", 1)[0]


@pytest.mark.parametrize("original", [PARAGRAPHS, NEWLINE_PROSE])
@pytest.mark.parametrize("quality", QUALITIES)
def test_shrink_prose_output_is_a_verbatim_substring(original: str, quality: float) -> None:
    # `split_prose` strips chunks, rejoins sentences with one space and drops punctuation-only
    # paragraphs, so a prefix rebuilt from its output is not quotable from the input.
    out = shrink_prose(original, quality)
    assert _head(out, original) in original


def test_shrink_prose_actually_elides_so_the_substring_test_is_not_vacuous() -> None:
    out = shrink_prose(PARAGRAPHS, 0.25)
    assert out != PARAGRAPHS
    assert out.endswith("passages elided)")
    assert count_tokens(out) < count_tokens(PARAGRAPHS)


def test_shrink_prose_keeps_the_blank_line_the_original_had() -> None:
    out = shrink_prose(PARAGRAPHS, 0.75)
    assert "\n\n" in _head(out, PARAGRAPHS)


def test_shrink_prose_locates_a_repeated_passage_left_to_right() -> None:
    passage = "The same paragraph appears twice in this value, word for word, with no variation."
    middle = "A middle passage that differs from both of its neighbours in this value."
    tail = "A fourth passage that exists only to give the budget something to drop here."
    original = f"{passage}\n\n{middle}\n\n{passage}\n\n{tail}"
    for quality in QUALITIES:
        out = shrink_prose(original, quality)
        assert _head(out, original) in original


@pytest.mark.parametrize("original", [PARAGRAPHS, NEWLINE_PROSE])
def test_shrink_prose_is_monotone_in_quality(original: str) -> None:
    sizes = [count_tokens(shrink_prose(original, q / 100)) for q in range(0, 101, 5)]
    assert sizes == sorted(sizes)


def test_shrink_shell_is_monotone_in_quality() -> None:
    command = " | ".join(f"step{i} --flag value{i} --other value{i}" for i in range(14))
    sizes = [count_tokens(_shrink_shell(command, q / 100)) for q in range(0, 101, 5)]
    assert sizes == sorted(sizes)


def test_shell_budget_matches_the_generic_percentile_helper() -> None:
    # The hardcoded `min(low + 1, 10)` was correct only while the table held 11 entries, and
    # the comment above it plans for a wider corpus.
    for q in range(0, 101, 5):
        assert shell_budget(q / 100) == _percentile_budget(SHELL_COMMAND_TOKENS, q / 100)
    assert _percentile_budget(PROSE_UNIT_TOKENS, 1.0) == PROSE_UNIT_TOKENS[-1]


EMPTYABLE = [
    ("{}", None),
    ("{}", ""),
    ("{}", "TodoWrite"),
    ('{"old_str": "x"}', None),
    ('{"old_str": "x"}', ""),
    ('{"old_string": "x", "new_string": "y"}', ""),
]


@pytest.mark.parametrize(("text", "tool_name"), EMPTYABLE)
def test_quantize_tool_use_never_returns_empty(text: str, tool_name: str | None) -> None:
    # `transcript.py` stores a missing name as "", not None, so the falsy-name path is the
    # common case; an empty segment still reaches embed_texts in the coverage metric.
    assert quantize_tool_use(text, 0.5, tool_name).strip()


PAYLOAD_SHAPES = [
    "{}",
    '{"a": 1}',
    json.dumps({"a": {"b": {"c": {"d": "x" * 50}}}}),
    json.dumps({"edits": [{"new_string": BODY}, {"new_string": BODY}]}),
    json.dumps({"file_path": "a/b/c/module.py", "content": BODY}),
    json.dumps({"command": " | ".join(f"step{i} --flag value{i}" for i in range(12))}),
    json.dumps({"todos": [{"content": "one", "status": "pending"} for _ in range(6)]}),
    json.dumps({"description": "short label"}),
    "<parameter=command>view</parameter>\n<parameter=path>/w/a.py</parameter>",
]


@pytest.mark.parametrize("text", PAYLOAD_SHAPES)
@pytest.mark.parametrize("tool_name", [None, "", "Bash", "Edit", "TodoWrite"])
@pytest.mark.parametrize("quality", [0.0, 0.5, 0.9])
def test_quantize_tool_use_never_grows_its_input(
    text: str, tool_name: str | None, quality: float
) -> None:
    # `render_call` re-serialises nested containers with json.dumps, repaying the scaffolding
    # `_shrink_arg` just saved, so shrinking a payload of containers could grow it.
    assert count_tokens(quantize_tool_use(text, quality, tool_name)) <= count_tokens(text)


def test_quantize_segment_never_grows_a_tool_use_segment() -> None:
    for text in PAYLOAD_SHAPES:
        seg = Segment(0, 0, "assistant", "tool_use", text, True, [], "TodoWrite")
        assert count_tokens(quantize_segment(seg, 0.0).text) <= count_tokens(text)

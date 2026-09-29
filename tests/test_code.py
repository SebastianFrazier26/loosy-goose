import json

import pytest

from loosy_goose.code import (
    _MAX_ARG_DEPTH,
    ELISION,
    band_lines,
    detect_language,
    normalize_language,
    parse_tool_payload,
    quantize_code,
    quantize_segment,
    quantize_tool_use,
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
    out = quantize_tool_use(json.dumps(payload), 0.0)
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
    out = quantize_tool_use(json.dumps(payload), 0.5)
    assert "{" not in out and '"' not in out
    # The expensive half of json.dumps is escaping: a newline cost two characters there.
    assert "\\n" not in out
    assert out.count("\n") >= 2


def test_rendered_atoms_are_a_subset_of_the_original_atoms() -> None:
    # Recall intersects against the atoms of the ORIGINAL text, so anything the rendering adds —
    # the tool name it once prepended matched the identifier pattern — is spent where no recall
    # can return it. The keys here are single words on purpose: `file_path` and `new_string` are
    # atoms too, and they are the separate, argued exception (they disambiguate values).
    text = json.dumps({"content": BODY, "description": "rewrites module_a.py"})
    original = set(extract_atoms(text))
    rendered = set(extract_atoms(quantize_tool_use(text, 0.0)))
    assert rendered, "a vacuous subset would pass whatever the renderer emitted"
    assert rendered <= original


def test_quantize_tool_use_quality_one_is_identity() -> None:
    text = json.dumps({"file_path": "a.py", "old_string": "a", "new_string": "b"})
    assert quantize_tool_use(text, 1.0) == text


def test_a_call_with_no_target_renders_as_its_bare_argument_line() -> None:
    text = json.dumps({"command": "ls | wc"})
    # 5 tokens against the payload's 8. While the tool name was prepended this rendered to
    # exactly 8 and survived only because the size guard admitted ties.
    assert quantize_tool_use(text, 0.5) == "command: ls | wc"
    assert count_tokens("command: ls | wc") < count_tokens(text)


def test_target_path_is_never_shrunk() -> None:
    text = json.dumps({"file_path": "a/b/c/very/long/path/to/module.py", "content": BODY})
    out = quantize_tool_use(text, 0.0)
    assert out.startswith("a/b/c/very/long/path/to/module.py")


def test_string_leaves_inside_nested_arguments_are_shrunk() -> None:
    # "Keep the shape, shrink the leaves": the container survives, its strings are routed.
    text = json.dumps({"edits": [{"new_string": BODY}, {"new_string": BODY}]})
    out = quantize_tool_use(text, 0.0)
    assert out.count("def g(a):") == 2  # container survived, both leaves shrunk
    assert "compute_second" not in out


def test_nested_arguments_stop_at_the_depth_bound() -> None:
    deep: dict[str, object] = {"new_string": BODY}
    for _ in range(_MAX_ARG_DEPTH + 2):
        deep = {"level": [deep]}
    # Must terminate and stay valid rather than recursing without end.
    assert quantize_tool_use(json.dumps(deep), 0.0).strip()


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
    f"It then runs on for a second sentence."
    for i in range(8)
)
LONG_COMMAND = " | ".join(f"step{i} --flag value{i} --other setting{i}" for i in range(12))
LONG_PROSE = " ".join(
    f"Sentence {i} of the prompt explains one more requirement for the subagent." for i in range(12)
)


def _tool_segment(payload: dict[str, object], tool_name: str) -> Segment:
    return Segment(0, 0, "assistant", "tool_use", json.dumps(payload), True, [], tool_name)


@pytest.mark.parametrize("quality", [0.0, 0.3, 0.5])
def test_a_shell_command_argument_survives_verbatim(quality: float) -> None:
    # TRANSFORMS_VERSION 4: no leading-stages cut. The rendering may still drop the JSON quoting.
    out = quantize_segment(_tool_segment({"command": LONG_COMMAND}, "Bash"), quality).text
    assert LONG_COMMAND in out
    assert "elided" not in out


@pytest.mark.parametrize("quality", [0.0, 0.3, 0.5])
def test_a_prose_argument_survives_verbatim(quality: float) -> None:
    payload = {"description": "delegate", "prompt": LONG_PROSE}
    out = quantize_segment(_tool_segment(payload, "Task"), quality).text
    assert LONG_PROSE in out
    assert "elided" not in out


def test_a_code_argument_is_still_banded_beside_verbatim_prose() -> None:
    payload = {"description": LONG_PROSE, "content": BODY}
    out = quantize_tool_use(json.dumps(payload), 0.0)
    assert LONG_PROSE in out
    assert "def g(a):" in out and "compute_second" not in out
    assert "lines elided" in out


@pytest.mark.parametrize("kind", ["prose", "thinking", "tool_result"])
def test_quantize_segment_returns_other_kinds_unchanged(kind: str) -> None:
    seg = Segment(3, 1, "assistant", kind, PARAGRAPHS, False, extract_atoms(PARAGRAPHS))
    assert quantize_segment(seg, 0.0) is seg


def test_quantize_segment_bands_a_code_segment() -> None:
    seg = Segment(0, 0, "assistant", "code", BODY, True, extract_atoms(BODY))
    assert quantize_segment(seg, 0.0).text == quantize_code(BODY, "python", 0.0)
    assert quantize_segment(seg, 1.0).text == BODY


EMPTYABLE = ["{}", '{"old_str": "x"}', '{"old_string": "x", "new_string": "y"}']


@pytest.mark.parametrize("text", EMPTYABLE)
def test_quantize_tool_use_never_returns_empty(text: str) -> None:
    # An empty segment still reaches embed_texts in the coverage metric.
    assert quantize_tool_use(text, 0.5).strip()


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
@pytest.mark.parametrize("quality", [0.0, 0.5, 0.9])
def test_quantize_tool_use_never_grows_its_input(text: str, quality: float) -> None:
    # `render_call` re-serialises nested containers with json.dumps, repaying the scaffolding
    # `_band_arg` just saved, so shrinking a payload of containers could grow it.
    assert count_tokens(quantize_tool_use(text, quality)) <= count_tokens(text)


def test_quantize_segment_never_grows_a_tool_use_segment() -> None:
    for text in PAYLOAD_SHAPES:
        seg = Segment(0, 0, "assistant", "tool_use", text, True, [], "TodoWrite")
        assert count_tokens(quantize_segment(seg, 0.0).text) <= count_tokens(text)

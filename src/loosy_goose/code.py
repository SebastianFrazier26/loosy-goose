"""Structural quantization of code: the "chroma table" for the code channel.

A code segment keeps its label (fence language, file path, tool payload keys) but its body is
banded by AST nesting depth and truncated at a per-quality band, the way JPEG keeps low-frequency
coefficients and zeroes high-frequency ones.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from dataclasses import replace
from functools import lru_cache
from typing import Any

import tree_sitter_language_pack as _pack
from tree_sitter import Node, Parser, Tree

from loosy_goose.segment import Segment, extract_atoms
from loosy_goose.tokens import count_tokens

ELISION = "… ({n} lines elided)"

_FENCE_ALIASES: dict[str, str] = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "mjs": "javascript",
    "cjs": "javascript",
    "node": "javascript",
    "ts": "typescript",
    "cs": "csharp",
    "c#": "csharp",
    "c_sharp": "csharp",
    "sh": "bash",
    "shell": "bash",
    "zsh": "bash",
    "console": "bash",
    "ps1": "powershell",
    "pwsh": "powershell",
    "yml": "yaml",
    "jsonc": "json",
    "json5": "json",
    "text": "",
    "plaintext": "",
    "txt": "",
    "output": "",
    "diff": "",
}

# Definitions: the header line(s) of these nodes are pinned to band 0 and the node's body child
# starts the next band. Field/property declarations are here too because they are part of a
# class's interface even though they sit inside its body.
_DEFINITION_TYPES = frozenset(
    {
        "function_definition",
        "function_declaration",
        "function_signature",
        "function_statement",
        "function_item",
        "method_definition",
        "method_declaration",
        "method_signature",
        "abstract_method_signature",
        "constructor_declaration",
        "class_definition",
        "class_declaration",
        "class_specifier",
        "struct_declaration",
        "struct_item",
        "struct_specifier",
        "interface_declaration",
        "enum_declaration",
        "enum_item",
        "record_declaration",
        "namespace_declaration",
        "trait_item",
        "impl_item",
        "mod_item",
        "decorated_definition",
        "type_alias_declaration",
        "field_declaration",
        "property_declaration",
        "public_field_definition",
        "property_signature",
        "event_field_declaration",
    }
)

# Pinned wholesale wherever they appear: imports and decorators carry the interface, returns
# carry the contract of the enclosing definition.
_PINNED_TYPES = frozenset(
    {
        "import_statement",
        "import_from_statement",
        "future_import_statement",
        "import_declaration",
        "using_directive",
        "use_declaration",
        "export_statement",
        "decorator",
        "attribute_list",
        "return_statement",
        "package_clause",
        "preproc_include",
    }
)

_BODY_TYPES = frozenset(
    {
        "block",
        "statement_block",
        "class_body",
        "declaration_list",
        "field_declaration_list",
        "enum_body",
        "interface_body",
        "object_type",
        "compound_statement",
        "script_block",
        "do_group",
    }
)

_COMMENT_TYPES = frozenset({"comment", "line_comment", "block_comment"})

# Claude Code's Read tool prints `cat -n` style: right-aligned number, U+2192 arrow, content.
_NUMBERED_LINE = re.compile(r"^\s*\d+(→|\t)")

# Ordered: ES-module imports and C# usings are checked before Python because a bare `import X`
# prefix is shared by all three; Python's own patterns are anchored to its statement shapes.
_HEURISTICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "typescript",
        re.compile(
            r"^\s*(?:import .+ from ['\"]|export |interface \w+|type \w+ =|(?:const|let) \w+: )",
            re.M,
        ),
    ),
    (
        "csharp",
        re.compile(r"^\s*(?:using [\w.]+;|namespace [\w.]+|public (?:class|void|static) )", re.M),
    ),
    (
        "python",
        re.compile(
            r"^\s*(?:def \w+\(|class \w+[:(]|import \w+(?:\.\w+)*(?:, *\w+)* *$"
            r"|from [\w.]+ import )",
            re.M,
        ),
    ),
    (
        "javascript",
        re.compile(r"^\s*(?:function \w+\(|const \w+ = |module\.exports|require\()", re.M),
    ),
    ("bash", re.compile(r"^\s*(?:#!/bin/(?:ba)?sh|export \w+=|if \[|fi$|echo )", re.M)),
    ("powershell", re.compile(r"^\s*(?:\$\w+ = |function [\w-]+ \{|Write-Host|param\()", re.M)),
)


def normalize_language(name: str | None) -> str | None:
    if not name:
        return None
    key = name.strip().lower()
    key = _FENCE_ALIASES.get(key, key)
    if not key:
        return None
    return key if _pack.has_language(key) else None


def language_from_path(path: str) -> str | None:
    detected = _pack.detect_language_from_path(path)
    return normalize_language(detected)


def guess_language(text: str) -> str | None:
    stripped = text.lstrip()
    if stripped[:1] in "{[":
        try:
            json.loads(text)
        except ValueError:
            pass
        else:
            return "json"
    shebang = _pack.detect_language_from_content(text) if stripped.startswith("#!") else None
    if shebang:
        return normalize_language(shebang)
    for lang, pattern in _HEURISTICS:
        if pattern.search(text):
            return lang
    return None


def detect_language(segment: Segment, *, hint: str | None = None) -> str | None:
    # Segment does not carry the fence language or tool name (Block.meta is dropped at
    # segmentation), so callers that still hold the Transcript pass it as `hint`.
    lang = normalize_language(hint)
    if lang:
        return lang
    if segment.kind == "tool_use":
        return "json"
    for atom in segment.atoms:
        if "/" in atom or "\\" in atom or "." in atom:
            lang = language_from_path(atom)
            if lang:
                return lang
    return guess_language(segment.text)


@lru_cache(maxsize=32)
def _parser(lang: str) -> Parser | None:
    # The 1.16 language pack ships no grammars in the wheel: get_parser downloads a checksummed
    # prebuilt .dll/.so per language into a per-user cache on first use. Offline first use is a
    # failure we degrade from (indentation banding), not one we propagate.
    try:
        return _pack.get_parser(lang)
    except Exception:  # noqa: BLE001 - the pack raises several unrelated error classes
        return None


def parse(text: str, lang: str) -> Tree | None:
    parser = _parser(lang)
    if parser is None:
        return None
    return parser.parse(text.encode("utf-8"))


def _walk(node: Node) -> Iterator[tuple[Node, list[Node]]]:
    stack: list[tuple[Node, list[Node]]] = [(node, [])]
    while stack:
        n, ancestors = stack.pop()
        yield n, ancestors
        child_ancestors = [*ancestors, n]
        for c in reversed(n.children):
            stack.append((c, child_ancestors))


def _is_definition(node: Node) -> bool:
    return node.type in _DEFINITION_TYPES


def _body_start_row(node: Node) -> int | None:
    for c in node.children:
        if c.type in _BODY_TYPES:
            return int(c.start_point.row)
    return None


def _indent_bands(lines: list[str]) -> list[int]:
    widths = sorted({len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()})
    rank = {w: i for i, w in enumerate(widths)}
    bands: list[int] = []
    for ln in lines:
        bands.append(rank[len(ln) - len(ln.lstrip())] if ln.strip() else -1)
    return bands


def _fill_blank(bands: list[int]) -> list[int]:
    # A blank line takes the band of the next non-blank line so it is elided along with the body
    # that follows it rather than surviving as a stray separator inside an elision.
    out = list(bands)
    nxt = 0
    for i in range(len(out) - 1, -1, -1):
        if out[i] < 0:
            out[i] = nxt
        else:
            nxt = out[i]
    return out


def _is_container(node: Node) -> bool:
    # Containers are the constructs that add a nesting level: indentation bodies (Python's
    # `block` has no brace token) and anything opened by a bracket. Only multi-line ones count,
    # so `f(a, b)` on one line adds no depth but a 40-line dict literal does.
    if node.end_point.row <= node.start_point.row:
        return False
    if node.type in _BODY_TYPES:
        return True
    return bool(node.children) and node.children[0].type in ("{", "(", "[")


def band_lines(text: str, lang: str | None) -> list[int]:
    """Structural band per line: 0 = interface (the DC coefficient), higher = deeper detail.

    Depth is the number of multi-line containers enclosing the line's outermost statement, so a
    body line under `if` under `def` is band 2 in Python and in brace languages alike. Definition
    headers, imports, decorators and returns are pinned to band 0 regardless of depth: they are
    the interface a reader needs to know what the code does; bodies are how it does it. Lines
    that only open or close a container (`{`, `}`) take the band of their header line so brace
    pairs are kept or elided together.
    """
    lines = text.split("\n")
    tree = parse(text, lang) if lang else None
    if tree is None:
        return _fill_blank(_indent_bands(lines))

    root = tree.root_node
    owner: dict[int, tuple[Node, list[Node]]] = {}
    opener: dict[int, tuple[Node, list[Node]]] = {}
    covering: dict[int, tuple[Node, list[Node]]] = {}
    for node, ancestors in _walk(root):
        if node is root or not node.is_named:
            continue
        row = int(node.start_point.row)
        if _is_container(node):
            opener.setdefault(row, (node, ancestors))
        else:
            owner.setdefault(row, (node, ancestors))
        # Clamp: py-tree-sitter 0.26.0 handed back nodes with garbage end points on large error-
        # recovered trees (hence the <0.26 pin); an out-of-range span must never drive a loop.
        end = min(max(int(node.end_point.row), row), len(lines) - 1)
        for r in range(row + 1, end + 1):
            covering[r] = (node, ancestors)

    def depth_of(ancestors: list[Node]) -> int:
        return sum(1 for a in ancestors if a is not root and _is_container(a))

    header_rows: set[int] = set()
    for node, _ in owner.values():
        if _is_definition(node):
            body = _body_start_row(node)
            start = int(node.start_point.row)
            stop = body if body is not None and body > start else start + 1
            header_rows.update(range(start, stop))

    bands: list[int] = []
    for row in range(len(lines)):
        if not lines[row].strip():
            bands.append(-1)
            continue
        if row in header_rows:
            bands.append(0)
            continue
        hit = owner.get(row)
        if hit is not None:
            node, ancestors = hit
            if node.type in _PINNED_TYPES or _is_definition(node):
                bands.append(0)
            elif node.type in _COMMENT_TYPES and _attached_to_definition(node):
                bands.append(0)
            elif _is_docstring(node):
                bands.append(0)
            else:
                bands.append(depth_of(ancestors))
            continue
        open_hit = opener.get(row)
        if open_hit is not None:
            _node, ancestors = open_hit
            parent_row = int(ancestors[-1].start_point.row) if ancestors else row
            bands.append(bands[parent_row] if parent_row < row else depth_of(ancestors))
            continue
        cover = covering.get(row)
        if cover is None:
            bands.append(0)
            continue
        node, ancestors = cover
        open_row = int(node.start_point.row)
        bands.append(bands[open_row] if bands[open_row] >= 0 else depth_of(ancestors))
    return _fill_blank(bands)


def _attached_to_definition(node: Node) -> bool:
    sib = node.next_named_sibling
    while sib is not None and sib.type in _COMMENT_TYPES:
        sib = sib.next_named_sibling
    return sib is not None and (_is_definition(sib) or sib.type == "decorator")


def _is_docstring(node: Node) -> bool:
    # tree-sitter-python puts a docstring straight under `block` as a bare `string`; older
    # grammars wrapped it in an expression_statement, so both shapes are accepted.
    inner = node.children[0] if node.type == "expression_statement" and node.children else node
    if inner.type != "string":
        return False
    parent = node.parent
    if parent is None or parent.type not in _BODY_TYPES or parent.parent is None:
        return False
    return node.prev_named_sibling is None and _is_definition(parent.parent)


def _strip_line_numbers(text: str) -> tuple[list[str], list[str]] | None:
    lines = text.split("\n")
    matches = [_NUMBERED_LINE.match(ln) for ln in lines]
    hits = sum(1 for m in matches if m)
    if hits < 3 or hits < 0.8 * sum(1 for ln in lines if ln.strip()):
        return None
    prefixes = [m.group(0) if m else "" for m in matches]
    bodies = [ln[len(p) :] for ln, p in zip(lines, prefixes, strict=True)]
    return prefixes, bodies


def _elide(lines: list[str], keep: list[bool]) -> str:
    out: list[str] = []
    i = 0
    while i < len(lines):
        if keep[i]:
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < len(lines) and not keep[j]:
            j += 1
        first = next((ln for ln in lines[i:j] if ln.strip()), "")
        indent = first[: len(first) - len(first.lstrip())]
        marker = indent + ELISION.format(n=j - i)
        # A marker that costs as many tokens as the run it replaces is pure loss; keeping the run
        # preserves the superset property (a run kept at low quality is a run kept at higher).
        if count_tokens("\n".join(lines[i:j])) <= count_tokens(marker):
            out.extend(lines[i:j])
        else:
            out.append(marker)
        i = j
    return "\n".join(out)


def quantize_code(text: str, lang: str | None, quality: float) -> str:
    """Keep lines with band <= floor(quality * max_band); 1.0 is verbatim, 0.0 signatures only."""
    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality must be in [0, 1]")
    if quality >= 1.0 or not text.strip():
        return text
    numbered = _strip_line_numbers(text)
    prefixes, lines = numbered if numbered else ([], text.split("\n"))
    bands = band_lines("\n".join(lines), lang)
    max_band = max(bands) if bands else 0
    # floor(quality * max_band) would make every quality below 1.0 collapse to band 0 on the
    # shallow two-band snippets that dominate tool_use payloads; spreading over max_band + 1
    # levels keeps 1.0 verbatim and 0.0 signatures-only while giving 0.66 / 0.33 distinct cuts.
    cutoff = min(max_band, math.floor(quality * (max_band + 1)))
    keep = [b <= cutoff for b in bands]
    if numbered:
        lines = [p + ln for p, ln in zip(prefixes, lines, strict=True)]
    return _elide(lines, keep)


_BODY_KEYS = ("new_string", "content", "new_str", "file_text")
_STALE_KEYS = ("old_string", "old_str")
_PATH_KEYS = ("file_path", "path", "notebook_path")


def _payload_language(payload: dict[str, Any]) -> str | None:
    for key in _PATH_KEYS:
        path = payload.get(key)
        if isinstance(path, str) and path:
            return language_from_path(path)
    return None


def _quantize_payload(payload: dict[str, Any], lang: str | None, quality: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _STALE_KEYS:
            # Once an edit has applied, the pre-edit text is redundant: new_string is the state
            # of the file that matters, and the diff can be recovered from a later Read anyway.
            continue
        if key in _BODY_KEYS and isinstance(value, str):
            out[key] = quantize_code(value, lang, quality)
        elif key == "edits" and isinstance(value, list):
            out[key] = [
                _quantize_payload(e, lang, quality) if isinstance(e, dict) else e for e in value
            ]
        else:
            out[key] = value
    return out


# OpenHands / SWE-Gym trajectories serialise tool arguments as <parameter=NAME>value</parameter>
# runs instead of JSON; both shapes are read into the same dict so the code channel and the
# supersession pass need not know which agent produced the transcript.
_XML_PARAM = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.DOTALL)


def parse_tool_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except ValueError:
            return None
        return payload if isinstance(payload, dict) else None
    params = _XML_PARAM.findall(stripped)
    if params and _XML_PARAM.sub("", stripped).strip() == "":
        return {name: value.strip("\n") for name, value in params}
    return None


def _format_xml_payload(payload: dict[str, Any]) -> str:
    return "\n".join(f"<parameter={k}>{v}</parameter>" for k, v in payload.items())


def quantize_tool_use(text: str, quality: float) -> str:
    if quality >= 1.0:
        return text
    payload = parse_tool_payload(text)
    if payload is None:
        return quantize_code(text, guess_language(text), quality)
    # `command` and the other short keys are kept verbatim: they are the label, not the body.
    reduced = _quantize_payload(payload, _payload_language(payload), quality)
    if text.lstrip().startswith("<"):
        return _format_xml_payload(reduced)
    return json.dumps(reduced, ensure_ascii=False, sort_keys=True)


def quantize_segment(segment: Segment, quality: float, *, lang: str | None = None) -> Segment:
    if segment.kind == "tool_use":
        text = quantize_tool_use(segment.text, quality)
    else:
        text = quantize_code(segment.text, detect_language(segment, hint=lang), quality)
    return replace(segment, text=text, protected=False, atoms=extract_atoms(text))

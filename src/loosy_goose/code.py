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

from loosy_goose.segment import Segment, extract_atoms, looks_like_code_dump, split_prose
from loosy_goose.tokens import count_tokens

ELISION = "… ({n} lines elided)"

# Bump whenever this module's OUTPUT changes for the same input and quality — a new routing rule,
# a changed budget table, a different call rendering. Experiment records are keyed on it, so a
# stored result whose text was produced by an older shrinker is recomputed instead of being
# reused. `segments_digest` cannot cover this: it fingerprints the segmenter's output, and
# shrinking happens downstream of it on purpose, so that variants accumulate rather than evict
# each other. That exclusion is what let a rewritten shrinker leave stale records looking healthy.
# 1: shape routing, measured shell/prose budgets, compact call rendering.
# 2: prose elision slices the original instead of rejoining normalised chunks (so a kept span is
#    a substring again, and retains the whitespace it had); the call rendering no longer prepends
#    the tool name, which came from the record's metadata rather than from the segment's text;
#    tool_use output that is empty, or not strictly smaller than its input, falls back to the
#    input. Those are three separate output changes under one stamp on purpose: no grid has run
#    since the stamp was introduced, so there are no records at 1 or 2 for a bump to invalidate.
TRANSFORMS_VERSION = 2

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
    # Segment carries `tool_name` but not the fence language, so callers that still hold the
    # Transcript pass it as `hint`.
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


_STALE_KEYS = ("old_string", "old_str")
_PATH_KEYS = ("file_path", "path", "notebook_path")

# Tools whose `command` argument is a shell command line, so its separators can be read as
# statement boundaries. Gated on the tool rather than on the value, because a `|` is a pipe here
# and an alternation in a Grep `pattern` — the same characters mean different things, and only
# the caller knows which. This is why `Segment.tool_name` is carried.
_SHELL_TOOLS = frozenset({"Bash", "PowerShell", "Shell", "run_command", "execute_bash"})
SHELL_ELISION = "… ({n} stages elided)"
# Bounds the walk over nested arguments. One level covers the `edits` list this replaces; the
# cap stops a pathological payload from recursing without end.
_MAX_ARG_DEPTH = 3


def split_shell(text: str) -> list[int]:
    """Start offsets of each top-level stage in a one-line shell command.

    Shell marks its own statement boundaries, so cutting a long one-liner needs no length
    threshold: `a | b | c` is three stages and `git commit -m "a | b"` is one. Quotes are tracked
    so a separator inside an argument does not split it.
    """
    starts = [0]
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote is not None:
            if ch == "\\" and i + 1 < len(text):
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        width = 2 if text[i : i + 2] in ("&&", "||") else (1 if ch in "|;" else 0)
        if width:
            i += width
            while i < len(text) and text[i] == " ":
                i += 1
            if i < len(text):
                starts.append(i)
            continue
        i += 1
    return starts


# Token lengths of single-stage shell commands across the corpus, by decile — commands that
# needed no pipeline and so are one complete unit of meaning by construction. Measured
# 2026-09-10 over n=136, which is small; widen the corpus before treating the tail as precise.
# This is what makes the cut point non-arbitrary: `quality` names how long a command we are
# still willing to keep, as a percentile of commands people actually wrote.
SHELL_COMMAND_TOKENS: tuple[int, ...] = (6, 11, 15, 17, 18, 20, 21, 26, 37, 62, 161)


def shell_budget(quality: float) -> int:
    """The token length `quality` says a command line is allowed to reach."""
    return _percentile_budget(SHELL_COMMAND_TOKENS, quality)


def _shrink_shell(text: str, quality: float) -> str:
    """Keep whole leading stages of a pipeline while the command line stays within the length a
    real command reaches at this quality, then elide the rest.

    Stages are the unit because shell marks its own boundaries, and the budget is the unit of
    meaning because a command that needed no pipeline is exactly one of those — so neither the
    split point nor the cut length is a number anyone picked. Stage 0 always survives, the same
    way band 0 does in the code channel.

    Known weakness, recorded rather than hidden: in `cat x | grep y` the filter is arguably the
    point, and this keeps `cat x`.
    """
    starts = split_shell(text)
    if len(starts) <= 1:
        return text
    budget = shell_budget(quality)
    keep = 1
    while keep < len(starts) and count_tokens(text[: starts[keep]].rstrip()) <= budget:
        keep += 1
    if keep >= len(starts):
        return text
    marker = SHELL_ELISION.format(n=len(starts) - keep)
    # Same guard as _elide: a marker costing what it replaces is pure loss.
    if count_tokens(text[starts[keep] :]) <= count_tokens(marker):
        return text
    return f"{text[: starts[keep]].rstrip()} {marker}"


def _payload_language(payload: dict[str, Any]) -> str | None:
    for key in _PATH_KEYS:
        path = payload.get(key)
        if isinstance(path, str) and path:
            return language_from_path(path)
    return None


# Token lengths of the corpus's own prose segments, by decile — complete units by construction,
# since the segmenter produced them. Measured 2026-09-10 over n=4,374. Same role the shell table
# plays: `quality` names how much prose we still keep, as a percentile of real prose units.
#
# Rejected on method, not on results: keeping the sentences that contain atoms and dropping the
# rest. That scores well on atom recall by construction — the shrinker would be optimising the
# metric that grades it, the same self-grading the select/score model split exists to prevent.
PROSE_UNIT_TOKENS: tuple[int, ...] = (1, 8, 14, 20, 29, 40, 52, 70, 100, 145, 322)
PROSE_ELISION = "… ({n} passages elided)"
# The segmenter's own prose bound, so the units this splits into are the same shape as the units
# whose lengths define the budget above.
_PROSE_CHUNK_CHARS = 600


def _percentile_budget(table: tuple[int, ...], quality: float) -> int:
    if quality >= 1.0:
        return table[-1]
    position = max(quality, 0.0) * (len(table) - 1)
    low = int(position)
    lower, upper = table[low], table[min(low + 1, len(table) - 1)]
    return int(round(lower + (upper - lower) * (position - low)))


def prose_budget(quality: float) -> int:
    """The token length `quality` says a prose value is allowed to reach."""
    return _percentile_budget(PROSE_UNIT_TOKENS, quality)


def _match_at(text: str, chunk: str, start: int) -> int | None:
    """End offset in `text` if `chunk` matches at `start` up to whitespace runs, else None."""
    i, j, end = start, 0, start
    while j < len(chunk):
        if chunk[j].isspace():
            j += 1
            continue
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != chunk[j]:
            return None
        i, j = i + 1, j + 1
        end = i
    return end


def _chunk_end(text: str, chunks: list[str], keep: int) -> int | None:
    """Offset in `text` where the `keep`-th chunk of `split_prose(text)` ends.

    `split_prose` normalises what it returns — it strips each chunk, rejoins sentences with a
    single space and drops punctuation-only paragraphs — so a kept prefix rebuilt from its output
    is not a substring of the input, which is the one property this project promises. Only
    whitespace ever differs, so the chunks are re-located in the original by a whitespace-tolerant
    walk and the original is sliced instead. The walk is strictly left-to-right: a chunk repeated
    verbatim later in the text cannot be matched against the wrong copy.
    """
    cursor = 0
    for chunk in chunks[:keep]:
        first = next((ch for ch in chunk if not ch.isspace()), "")
        if not first:
            return None
        at = text.find(first, cursor)
        while at != -1:
            end = _match_at(text, chunk, at)
            if end is not None:
                cursor = end
                break
            at = text.find(first, at + 1)
        else:
            return None
    return cursor


def shrink_prose(text: str, quality: float) -> str:
    """Keep whole leading passages while the value stays within the length real prose reaches at
    this quality, then elide the rest.

    Leading rather than best-scoring: picking the best would need the embedding machinery inside
    a per-segment shrinker, which is a different cost profile entirely, and picking by atom
    content would grade itself. First-passage-first is the prose analogue of band 0.
    """
    if quality >= 1.0 or not text.strip():
        return text
    chunks = split_prose(text, _PROSE_CHUNK_CHARS)
    if len(chunks) <= 1:
        return text
    budget = prose_budget(quality)
    keep, used = 0, 0
    while keep < len(chunks):
        used += count_tokens(chunks[keep])
        if keep and used > budget:
            break
        keep += 1
    if keep >= len(chunks):
        return text
    cut = _chunk_end(text, chunks, keep)
    if cut is None:
        return text
    marker = PROSE_ELISION.format(n=len(chunks) - keep)
    # Same guard as _elide: a marker costing what it replaces is pure loss.
    if count_tokens(text[cut:]) <= count_tokens(marker):
        return text
    return f"{text[:cut]}\n{marker}"


def _shrink_string(text: str, lang: str | None, quality: float, *, shell: bool) -> str:
    """Route one string argument to the shrinker its shape calls for.

    Shape, not key name. The previous rule shrank four named keys and passed everything else
    through verbatim, which left 34.1% of the corpus's `tool_use` tokens untouched — `command`
    alone was 24.5% of the channel.
    """
    if shell:
        return _shrink_shell(text, quality)
    if looks_like_code_dump(text):
        return quantize_code(text, lang, quality)
    # Everything else is prose: a prompt, a description, a message. Short labels survive this
    # untouched, because a value below the budget is returned as-is and the elision guard
    # refuses to trade a marker for text no longer than the marker.
    return shrink_prose(text, quality)


def _shrink_arg(value: Any, lang: str | None, quality: float, *, shell: bool, depth: int) -> Any:
    if isinstance(value, str):
        return _shrink_string(value, lang, quality, shell=shell)
    if depth >= _MAX_ARG_DEPTH:
        return value
    if isinstance(value, list):
        return [_shrink_arg(v, lang, quality, shell=shell, depth=depth + 1) for v in value]
    if isinstance(value, dict):
        return _quantize_payload(value, lang, quality, shell=shell, depth=depth + 1)
    return value


def _quantize_payload(
    payload: dict[str, Any],
    lang: str | None,
    quality: float,
    *,
    shell: bool = False,
    depth: int = 0,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _STALE_KEYS:
            # Once an edit has applied, the pre-edit text is redundant: new_string is the state
            # of the file that matters, and the diff can be recovered from a later Read anyway.
            continue
        if key in _PATH_KEYS:
            # The target is the identifying fact and supersession's key: never shrunk.
            out[key] = value
        else:
            out[key] = _shrink_arg(
                value, lang, quality, shell=shell and key == "command", depth=depth
            )
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


def render_call(payload: dict[str, Any]) -> str:
    """A tool call as a compact call line instead of a JSON object.

    `json.dumps` spends tokens on braces, quoting, commas and — the expensive part — escaping:
    every newline in an edit body costs two characters instead of one, and every Windows
    separator doubles. None of that is information about the session. Measured at 12.2% of the
    corpus's `tool_use` tokens.

    The tool name is deliberately not emitted, though the caller knows it (decided 2026-09-10).
    It reaches us from the transcript record's metadata, not from the segment's text, so it is
    neither a verbatim span of the original nor a mechanically-defined pointer — the only two
    things this output is allowed to contain. It is also strictly bad on the metric rather than a
    trade: `MultiEdit`, `TodoWrite` and `NotebookEdit` all match the identifier pattern, so they
    enter the rendered segment's atoms, while recall intersects against the atoms of the ORIGINAL
    text, which never held them. The tokens buy nothing back and count against the ratio. The
    rejected alternatives were redefining it as a pointer, and keeping it silently.

    Key names are kept even though they are the tool's schema rather than session facts, because
    dropping them makes a Bash call's command and description indistinguishable. They cost tokens
    but cannot improve recall either: it intersects against atoms from the ORIGINAL segments,
    where `scannable` strips keys already, so a key name appearing here is never counted. The
    difference from the tool name is that a key is a substring of the payload being rendered, and
    that it disambiguates values that would otherwise be unreadable.
    """
    target = next((payload[k] for k in _PATH_KEYS if isinstance(payload.get(k), str)), None)
    lines: list[str] = [target] if target else []
    for key, value in payload.items():
        if key in _PATH_KEYS and value == target:
            continue
        # Known limitation: a non-string value is re-serialised here, so the braces, quoting and
        # escaping that this rendering exists to remove come back for every nested container —
        # `_shrink_arg` shrank its string leaves, but the scaffolding around them is repaid in
        # full. The `quantize_tool_use` size guard is what stops that from making output larger
        # than input; recovering the saving needs a non-JSON rendering for containers.
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {text}")
    return "\n".join(lines)


def quantize_tool_use(text: str, quality: float, tool_name: str | None = None) -> str:
    if quality >= 1.0:
        return text
    payload = parse_tool_payload(text)
    if payload is None:
        return quantize_code(text, guess_language(text), quality)
    reduced = _quantize_payload(
        payload,
        _payload_language(payload),
        quality,
        shell=tool_name in _SHELL_TOOLS,
    )
    rendered = (
        _format_xml_payload(reduced) if text.lstrip().startswith("<") else render_call(reduced)
    )
    # A shrinker that returns nothing, or that does not strictly shrink, has failed at its one
    # job. Both are reachable: every key can be dropped or empty (`{}` renders to ""), and a
    # payload of nested containers pays back more scaffolding in `render_call` than `_shrink_arg`
    # saved. An empty segment is the worse of the two — it still reaches `embed_texts` in the
    # coverage metric and contributes a degenerate vector to it.
    # A tie falls back too: a rendering that costs exactly what the JSON cost has bought nothing,
    # and the raw payload is the form that needs no defending. This was `>` while the tool name
    # was prepended, because that name cost a short single-argument call exactly what dropping
    # the JSON scaffolding saved; with the name gone those calls shrink strictly and the weaker
    # sign has nothing left to protect.
    if not rendered.strip() or count_tokens(rendered) >= count_tokens(text):
        return text
    return rendered


def quantize_segment(
    segment: Segment,
    quality: float,
    *,
    lang: str | None = None,
    tool_quality: float | None = None,
) -> Segment:
    """`tool_quality` overrides `quality` for `tool_use` only.

    That channel tolerates more loss than any other and the argument for shrinking it does not
    weaken as the rate knob relaxes, so tying its depth to the global rate leaves the cheapest
    tokens in the transcript untouched at exactly the rates where fidelity is cheap elsewhere.
    Defaults to `quality`, so every prior number reproduces until a caller asks otherwise.
    """
    if segment.kind == "tool_use":
        depth = quality if tool_quality is None else tool_quality
        text = quantize_tool_use(segment.text, depth, segment.tool_name)
    else:
        text = quantize_code(segment.text, detect_language(segment, hint=lang), quality)
    return replace(segment, text=text, protected=False, atoms=extract_atoms(text))

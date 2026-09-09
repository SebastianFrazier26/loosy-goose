"""Path substitution: name every file once, and refer to it by a marker after that.

A file path is repeated far more than it is informative. Across the corpus, 17,760 mentions of
an already-mentioned path cost 136,363 tokens, and every one of them says the same thing as the
first. Substitution states each path once in a table and replaces every mention with a short
marker, which does three separate things at once — and they are measured separately, because
only the first is certain:

  * the table guarantees the path survives compression, whatever selection decides to drop;
  * the marker is shorter than the path, so the text shrinks (3.87% of corpus tokens net of the
    table's own cost);
  * scoring no longer sees long, repetitive path strings, which may make the embeddings rank
    segments better or worse.

Runs AFTER supersession. Supersession matches `file_path` values across tool calls to find
superseded reads and edits; substituting first would leave it comparing markers, and while the
mapping is one-to-one, the paths it reports would be meaningless to a reader of the results.

The marker is numeric, not the file's basename: only 52% of basenames are unique within a
transcript, so a basename marker needs disambiguation half the time and still costs more per
mention (1.93% net saving against 3.87%).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from loosy_goose.segment import Segment, extract_atoms, scannable
from loosy_goose.tokens import count_tokens

DEFAULT_MARKER = "[P"

# Mirrors the two path-shaped patterns in segment._ATOM_PATTERNS, so what the table guarantees is
# the same population the recall metric counts. A path the extractor does not recognise as an
# atom would be substituted for no measurable gain.
_PATH_SLASH = re.compile(r"(?<![\w.])(?:[A-Za-z]:)?(?:[\w.-]+[\\/])+[\w.-]+")
_PATH_BARE = re.compile(
    r"(?<![\w./\\])[\w-]+\.(?:py|js|ts|tsx|jsx|json|md|txt|yml|yaml|toml|cs|rs|go|"
    r"java|sql|sh|ps1|csv|ipynb|html|css|xml|cfg|ini|lock|pdf|png|jpg|jsonl)\b"
)


def find_paths(text: str) -> list[str]:
    """Every path-shaped run in the text. The slash pattern is applied first and its spans are
    reserved, so a full path is not also counted as the bare filename it ends with."""
    taken: list[tuple[int, int]] = []
    found: list[tuple[int, str]] = []
    for pat in (_PATH_SLASH, _PATH_BARE):
        for m in pat.finditer(text):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((m.start(), m.group(0)))
    return [s for _, s in sorted(found)]


@dataclass(frozen=True)
class PathTable:
    """Distinct paths, one row each, in first-mention order."""

    paths: tuple[str, ...]
    marker: str = DEFAULT_MARKER

    def placeholder(self, index: int) -> str:
        return f"{self.marker}{index}]"

    def mapping(self) -> dict[str, str]:
        return {p: self.placeholder(i) for i, p in enumerate(self.paths)}

    def render(self) -> str:
        return "\n".join(f"{self.placeholder(i)} {p}" for i, p in enumerate(self.paths))

    def tokens(self) -> int:
        return count_tokens(self.render()) if self.paths else 0

    def guaranteed_atoms(self) -> set[str]:
        """Atoms a reader can recover from the emitted table alone. Derived from the rendered
        text rather than from `paths`, so it cannot claim more than the output actually says."""
        return set(extract_atoms(self.render())) if self.paths else set()


def _choose_marker(segments: list[Segment], preferred: str = DEFAULT_MARKER) -> str:
    """A marker that appears nowhere in the text, so expansion cannot hit a false positive."""
    marker = preferred
    while any(marker in s.text for s in segments):
        marker = f"{marker[:-1]}P]" if marker.endswith("]") else marker + "P"
    return marker


def build_table(
    segments: list[Segment], *, min_mentions: int = 2, marker: str = DEFAULT_MARKER
) -> PathTable:
    """Collect the paths worth tabulating, in first-mention order.

    `min_mentions` is a genuine tradeoff, not a tuning constant, which is why it is a knob. At 2
    the table only holds paths that repeat, which is token-optimal: a row for a path mentioned
    once costs more than the mention it replaces. At 1 it holds everything, which is
    recall-optimal for exactly the opposite reason — a path mentioned once is the one selection
    is most likely to drop.
    """
    if min_mentions < 1:
        raise ValueError("min_mentions must be at least 1")
    counts: dict[str, int] = {}
    for s in segments:
        # Searched in decoded form: inside a tool call a real backslash is doubled and a newline
        # is a literal escape, so the raw text does not contain the path as a reader would write
        # it. `scannable` is the same decode the atom extractor uses, which keeps the table's
        # population aligned with the metric's.
        for p in find_paths(scannable(s.text)):
            counts[p] = counts.get(p, 0) + 1
    ordered = tuple(p for p, n in counts.items() if n >= min_mentions)
    return PathTable(ordered, _choose_marker(segments, marker))


def _sub_in_string(text: str, mapping: dict[str, str]) -> str:
    # Longest first: substituting `main.py` before `src/main.py` would strand the directory.
    for path in sorted(mapping, key=len, reverse=True):
        text = text.replace(path, mapping[path])
    return text


def _sub_in_json(node: Any, mapping: dict[str, str]) -> Any:
    if isinstance(node, str):
        return _sub_in_string(node, mapping)
    if isinstance(node, dict):
        return {k: _sub_in_json(v, mapping) for k, v in node.items()}
    if isinstance(node, list):
        return [_sub_in_json(v, mapping) for v in node]
    return node


def substitute_text(text: str, mapping: dict[str, str]) -> str:
    """Replace path mentions with their markers, preserving a tool call's JSON validity.

    A tool_use segment is re-encoded through json.dumps with transcript.py's exact arguments, so
    a segment containing no substitutable path comes back byte-identical rather than reformatted.
    Substituting the raw text instead would have to match the escaped spelling of every path.
    """
    if not mapping:
        return text
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            return json.dumps(_sub_in_json(payload, mapping), ensure_ascii=False, sort_keys=True)
    return _sub_in_string(text, mapping)


def substitute(segments: list[Segment], table: PathTable) -> list[Segment]:
    """Segments with paths replaced by markers and atoms re-extracted.

    Atoms are recomputed rather than carried over: a substituted segment genuinely no longer
    holds the path, and the table is what preserves it. Passing the table's `guaranteed_atoms()`
    to `metrics.evaluate` is what puts it back on the credit side of the ledger.
    """
    mapping = table.mapping()
    if not mapping:
        return list(segments)
    out: list[Segment] = []
    for s in segments:
        text = substitute_text(s.text, mapping)
        out.append(s if text == s.text else replace(s, text=text, atoms=extract_atoms(text)))
    return out


def expand(text: str, table: PathTable) -> str:
    """Inverse of substitution. Substitution is only acceptable because this exists: the emitted
    text plus the table reconstructs the original exactly, so the output is still verbatim, just
    not as a contiguous substring."""
    for i, path in enumerate(table.paths):
        text = text.replace(table.placeholder(i), path)
    return text

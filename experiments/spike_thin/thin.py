"""Within-segment token thinning: drop the most predictable words until an o200k budget is met.

Spike code (2026-10-21). The unit dropped is a whitespace word, not an LM sub-token: dropping
sub-tokens leaves half-words that read as typos, and the budget currency (o200k) does not share
the LM's token boundaries anyway. A word's information is the sum of the surprisal of the LM
tokens that end inside it; words go in ascending order of bits per o200k token, so what is
dropped is what costs the most budget for the least information.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from loosy_goose.segment import Segment, extract_atoms
from loosy_goose.tokens import count_tokens

THIN_KINDS = frozenset({"prose", "thinking", "tool_result"})
THIN_KINDS_CODE = THIN_KINDS | {"code"}
# A unit is a word with its trailing punctuation, or a bare punctuation run. Whitespace runs
# alone were too coarse: a JSON line or a long `a.b(c=d)` expression is one whitespace "word",
# and one atom inside it protected the whole run (measured 2026-10-21: thin-only stalled far above
# budget on such transcripts).
_WORD = re.compile(r"\w+[^\w\s]*|[^\w\s]+")
# Error lines in tool output are kept whole: an error message is an atom-like fact whose words
# (`not`, `found`, `expected`) are individually predictable and would otherwise be thinned away.
_ERROR_LINE = re.compile(
    r"(?i)\b(error|exception|traceback|failed|failure|fatal|assert\w*|denied|not found)\b"
)

TokenBits = Callable[[str], tuple[np.ndarray, np.ndarray]]


@dataclass
class SegWords:
    spans: list[tuple[int, int]]
    bits: np.ndarray
    cost: np.ndarray
    protected: np.ndarray


def _doc(segments: Sequence[Segment]) -> tuple[str, list[int]]:
    parts: list[str] = []
    starts: list[int] = []
    pos = 0
    prev_turn = None
    for s in segments:
        head = (f"\n\n### {s.role}\n" if s.turn != prev_turn else "\n") + f"[{s.kind}]\n"
        prev_turn = s.turn
        parts.append(head)
        pos += len(head)
        starts.append(pos)
        parts.append(s.text)
        pos += len(s.text)
    return "".join(parts), starts


def _protected_mask(seg: Segment, spans: list[tuple[int, int]]) -> np.ndarray:
    text = seg.text
    cover = np.zeros(len(text) + 1, dtype=bool)
    for atom in seg.atoms:
        start = 0
        while (i := text.find(atom, start)) >= 0:
            cover[i : i + len(atom)] = True
            start = i + 1
    if seg.kind in ("tool_result", "code"):
        off = 0
        for line in text.split("\n"):
            if _ERROR_LINE.search(line):
                cover[off : off + len(line)] = True
            off += len(line) + 1
    out = np.zeros(len(spans), dtype=bool)
    for k, (a, b) in enumerate(spans):
        out[k] = bool(cover[a:b].any()) or bool(extract_atoms(text[a:b]))
    return out


def score_words(
    segments: Sequence[Segment], token_bits: TokenBits, kinds: frozenset[str] = THIN_KINDS_CODE
) -> dict[int, SegWords]:
    """Per thinnable segment id: its words, their bits, o200k costs and protection. Scored once
    over the whole stretch in order, so every word is conditioned on what came before it."""
    doc, starts = _doc(segments)
    bits, ends = token_bits(doc)
    order = np.argsort(ends, kind="stable")
    ends_sorted = ends[order]
    bits_sorted = bits[order]
    csum = np.concatenate([[0.0], np.cumsum(bits_sorted)])
    out: dict[int, SegWords] = {}
    for s, base in zip(segments, starts, strict=True):
        if s.kind not in kinds:
            continue
        spans = [(m.start(), m.end()) for m in _WORD.finditer(s.text)]
        if not spans:
            continue
        # A token belongs to the word holding its last character (end offset - 1).
        lo = np.searchsorted(ends_sorted, np.array([base + a + 1 for a, _ in spans]), "left")
        hi = np.searchsorted(ends_sorted, np.array([base + b + 1 for _, b in spans]), "left")
        wbits = csum[hi] - csum[lo]
        cost = np.array(
            [
                count_tokens((" " if a and s.text[a - 1].isspace() else "") + s.text[a:b])
                for a, b in spans
            ],
            dtype=np.float64,
        )
        out[s.id] = SegWords(spans, wbits, cost, _protected_mask(s, spans))
    return out


def _rebuild(text: str, spans: list[tuple[int, int]], keep: np.ndarray) -> str:
    pieces: list[str] = []
    prev: int | None = None
    for k, (a, b) in enumerate(spans):
        if not keep[k]:
            continue
        if prev is not None:
            gap = text[spans[prev][1] : a]
            if prev == k - 1:
                pieces.append(gap)
            elif "\n" in gap:
                # A line break inside the dropped run survives, with this line's indentation.
                line = text[text.rfind("\n", 0, a) + 1 : a]
                pieces.append("\n" + line[: len(line) - len(line.lstrip())])
            elif any(c.isspace() for c in gap):
                pieces.append(" ")
        pieces.append(text[a:b])
        prev = k
    return "".join(pieces)


@dataclass
class ThinResult:
    segments: list[Segment]
    tokens: int
    dropped_words: int
    candidate_words: int
    over_budget: bool


def thin_to_budget(
    kept: Sequence[Segment],
    words: dict[int, SegWords],
    budget: int,
    *,
    max_drop_frac: float = 1.0,
    kinds: frozenset[str] = THIN_KINDS_CODE,
) -> ThinResult:
    kept = sorted(kept, key=lambda s: s.id)
    seg_tokens = {s.id: count_tokens(s.text) for s in kept}
    total = sum(seg_tokens.values())
    cand: list[tuple[float, int, int]] = []
    for s in kept:
        w = words.get(s.id)
        if w is None or s.kind not in kinds:
            continue
        for k in range(len(w.spans)):
            if not w.protected[k]:
                cand.append((float(w.bits[k] / max(w.cost[k], 1.0)), s.id, k))
    cand.sort()
    keep = {sid: np.ones(len(w.spans), dtype=bool) for sid, w in words.items()}
    cap = {sid: int(max_drop_frac * len(w.spans)) for sid, w in words.items()}
    dropped = {sid: 0 for sid in words}
    by_id = {s.id: s for s in kept}
    i = 0
    cur = total
    touched: set[int] = set()
    while cur > budget and i < len(cand):
        need = cur - budget
        saved = 0.0
        # Estimate with per-word costs, then re-measure exactly; o200k merges across a dropped
        # word's neighbours make the estimate run slightly optimistic.
        while saved < need and i < len(cand):
            _, sid, k = cand[i]
            i += 1
            if dropped[sid] >= cap[sid]:
                continue
            keep[sid][k] = False
            dropped[sid] += 1
            saved += words[sid].cost[k]
            touched.add(sid)
        for sid in touched:
            w = words[sid]
            seg_tokens[sid] = count_tokens(_rebuild(by_id[sid].text, w.spans, keep[sid]))
        touched.clear()
        cur = sum(seg_tokens.values())
    out: list[Segment] = []
    for s in kept:
        w = words.get(s.id)
        if w is None or dropped.get(s.id, 0) == 0:
            out.append(s)
            continue
        text = _rebuild(s.text, w.spans, keep[s.id])
        out.append(replace(s, text=text, atoms=extract_atoms(text)))
    return ThinResult(
        out,
        sum(count_tokens(s.text) for s in out),
        sum(dropped.values()),
        len(cand),
        cur > budget,
    )

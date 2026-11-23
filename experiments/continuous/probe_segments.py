"""Chunked probe generation for the open-weight replay (2026-11-17, final-review ruling): a Nemo
probe prompt over a whole pre-tail history reached 53k tokens, which does not fit 16 GB. The
history is split at turn boundaries into segments of at most `SEGMENT_TOKENS` Nemo tokens, and
N_PROBES is shared across segments by token share (at least one each). A bounded recent window was
rejected: it would bias probes toward recent text and so favour tail-heavy arms."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from experiments import exp_i_downstream as exi
from loosy_goose.transcript import Turn

SEGMENT_TOKENS = 12_000
Counter = Callable[[str], int]
_SEP = "\n\n"


def allocate(sizes: Sequence[int], n: int = exi.N_PROBES) -> list[int]:
    """Largest-remainder split of n in proportion to sizes, with a floor of one per segment."""
    k = len(sizes)
    if k > n:
        raise ValueError(f"{k} segments cannot each get a probe out of {n}")
    total = sum(sizes)
    quota = [n * s / total for s in sizes] if total else [n / k] * k
    got = [max(1, math.floor(q)) for q in quota]
    while sum(got) < n:
        i = max(range(k), key=lambda j: (quota[j] - got[j], -j))
        got[i] += 1
    while sum(got) > n:
        i = min((j for j in range(k) if got[j] > 1), key=lambda j: (quota[j] - got[j], j))
        got[i] -= 1
    return got


def _split_text(text: str, count: Counter, cap: int) -> list[str]:
    """A single turn over the cap, halved near the middle (at a line break, else a space) until
    each piece fits."""
    if len(text) <= 1 or count(text) <= cap:
        return [text]
    mid = len(text) // 2
    cut = text.rfind("\n", 0, mid) + 1
    if cut < mid // 2:
        cut = text.rfind(" ", 0, mid) + 1 or mid
    return _split_text(text[:cut], count, cap) + _split_text(text[cut:], count, cap)


def _fit(units: list[str], count: Counter, cap: int) -> list[str]:
    # Packing sums per-turn counts; a joined segment can still tokenise a little differently.
    text = _SEP.join(units)
    if len(units) == 1 or count(text) <= cap:
        return [text]
    mid = len(units) // 2
    return _fit(units[:mid], count, cap) + _fit(units[mid:], count, cap)


def segment_history(turns: Sequence[Turn], count: Counter, cap: int = SEGMENT_TOKENS) -> list[str]:
    units: list[tuple[str, int]] = []
    for t in turns:
        units += [(u, count(u)) for u in _split_text(exi.render_turns([t]), count, cap)]
    if not units:
        return [exi.render_turns(turns)]
    sep = count(_SEP)
    groups: list[list[str]] = []
    cur: list[str] = []
    cur_n = 0
    for text, n in units:
        if cur and cur_n + sep + n > cap:
            groups.append(cur)
            cur, cur_n = [], 0
        cur_n += n + (sep if cur else 0)
        cur.append(text)
    groups.append(cur)
    return [seg for g in groups for seg in _fit(g, count, cap)]


def segment_jobs(
    label: str, turns: Sequence[Turn], count: Counter, cap: int = SEGMENT_TOKENS
) -> list[tuple[str, str, int]]:
    """(segment label, segment text, probes asked) in history order; the segment label is the
    record's transcript field, so each segment has its own stable key."""
    segs = segment_history(turns, count, cap)
    alloc = allocate([count(s) for s in segs])
    return [(f"{label}#s{i}", s, n) for i, (s, n) in enumerate(zip(segs, alloc, strict=True))]


def parse_segment_probes(text: str, n: int) -> list[dict[str, str]] | None:
    """The first n probes; valid with at least exi's 3-in-10 share of n (and at least one)."""
    obj = exi.parse_json_object(text)
    if obj is None or not isinstance(obj.get("probes"), list):
        return None
    probes = [
        {"q": str(p["q"]).strip(), "a": str(p["a"]).strip()}
        for p in obj["probes"]
        if isinstance(p, dict) and p.get("q") and p.get("a")
    ][:n]
    return probes if len(probes) >= max(1, math.ceil(n * 3 / exi.N_PROBES)) else None


def assemble(
    segments: Sequence[tuple[str, int]], text_of: Callable[[str], str | None]
) -> list[dict[str, str]] | None:
    """A checkpoint's probe set: every segment's probes concatenated in segment order, or None
    while any segment is missing or invalid."""
    out: list[dict[str, str]] = []
    for seg, n in segments:
        text = text_of(seg)
        probes = parse_segment_probes(text, n) if text is not None else None
        if probes is None:
            return None
        out += probes
    return out

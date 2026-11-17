"""Per-chunk one-shot selection (the `chunk` arm of downstream.py, 2026-11-16).

Splits the supersession pool into consecutive chunks of at most F tokens at segment boundaries and
runs a selection function inside each. Chunk i is allowed budget * (tokens up to and including
chunk i) / pool tokens, less what earlier chunks actually kept, so a chunk that cannot use its
share (or a segment-granularity undershoot) spills forward, and the last chunk absorbs the
remainder. A final global pass over the not-yet-kept segments spends whatever is still left. The
total never exceeds the budget the whole-pool selection would get.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from loosy_goose.segment import Segment
from loosy_goose.tokens import count_tokens

CHUNK_VERSION = 1
CHUNK_TOKENS = 4000

Selector = Callable[[list[Segment], int], list[Segment]]


def split_chunks(pool: Sequence[Segment], f_tokens: int = CHUNK_TOKENS) -> list[list[Segment]]:
    """A chunk is closed before the segment that would push it past `f_tokens`; a single segment
    larger than `f_tokens` forms its own chunk."""
    chunks: list[list[Segment]] = []
    cur: list[Segment] = []
    used = 0
    for seg in pool:
        t = count_tokens(seg.text)
        if cur and used + t > f_tokens:
            chunks.append(cur)
            cur, used = [], 0
        cur.append(seg)
        used += t
    if cur:
        chunks.append(cur)
    return chunks


def chunk_shares(chunks: Sequence[Sequence[Segment]], budget: int) -> list[int]:
    """Nominal per-chunk budgets, proportional to chunk tokens; cumulative rounding makes them sum
    to exactly `budget`."""
    sizes = [sum(count_tokens(s.text) for s in c) for c in chunks]
    total = sum(sizes)
    if total == 0:
        return [0] * len(chunks)
    shares: list[int] = []
    cum = given = 0
    for size in sizes:
        cum += size
        target = round(budget * cum / total)
        shares.append(target - given)
        given = target
    return shares


def chunked_select(
    pool: Sequence[Segment],
    budget: int,
    select: Selector,
    f_tokens: int = CHUNK_TOKENS,
) -> list[Segment]:
    chunks = split_chunks(pool, f_tokens)
    if not chunks:
        return []
    shares = chunk_shares(chunks, budget)
    order = {s.id: i for i, s in enumerate(pool)}
    kept: list[Segment] = []
    spent = 0
    given = 0
    for chunk, share in zip(chunks, shares, strict=True):
        given += share
        allowance = given - spent
        if allowance <= 0:
            continue
        got = sorted(select(list(chunk), allowance), key=lambda s: order[s.id])
        kept.extend(got)
        spent += sum(count_tokens(s.text) for s in got)
    # Segment granularity strands budget the spill cannot place; refill it globally so the arm
    # meets the same token budget as whole-pool selection (budget.py contract).
    leftover = budget - spent
    if leftover > 0:
        taken = {s.id for s in kept}
        rest = [s for s in pool if s.id not in taken]
        if rest:
            kept.extend(select(rest, leftover))
    return sorted(kept, key=lambda s: order[s.id])

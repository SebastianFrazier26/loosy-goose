from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "slot_compress"))

import chunked  # noqa: E402

from loosy_goose.segment import Segment  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402


def _seg(i: int, n_words: int) -> Segment:
    return Segment(i, i, "assistant", "prose", " ".join(f"w{i}x{j}" for j in range(n_words)), False)


def _pool(sizes: list[int]) -> list[Segment]:
    return [_seg(i, n) for i, n in enumerate(sizes)]


def _tok(segs: list[Segment]) -> int:
    return sum(count_tokens(s.text) for s in segs)


def _longest_first(pool: list[Segment], budget: int) -> list[Segment]:
    """Deterministic stand-in for loosy_select: greedy fill by token size, never over budget."""
    out: list[Segment] = []
    used = 0
    for s in sorted(pool, key=lambda s: -count_tokens(s.text)):
        t = count_tokens(s.text)
        if used + t <= budget:
            out.append(s)
            used += t
    return sorted(out, key=lambda s: s.id)


def test_chunks_respect_f_and_order() -> None:
    pool = _pool([50, 80, 30, 120, 60, 90, 40, 70, 110, 20] * 3)
    f = 1000
    chunks = chunked.split_chunks(pool, f)
    assert [s.id for c in chunks for s in c] == [s.id for s in pool]
    assert len(chunks) > 1
    assert all(_tok(c) <= f for c in chunks)


def test_oversize_segment_is_its_own_chunk() -> None:
    pool = _pool([10, 400, 10])
    chunks = chunked.split_chunks(pool, 100)
    assert [[s.id for s in c] for c in chunks] == [[0], [1], [2]]


def test_empty_pool() -> None:
    assert chunked.split_chunks([], 100) == []
    assert chunked.chunked_select([], 100, _longest_first, 100) == []


@pytest.mark.parametrize("budget", [0, 1, 137, 1000, 5000])
def test_shares_sum_to_budget(budget: int) -> None:
    chunks = chunked.split_chunks(_pool([50, 80, 30, 120, 60, 90, 40, 70] * 4), 250)
    shares = chunked.chunk_shares(chunks, budget)
    assert sum(shares) == budget
    assert all(s >= 0 for s in shares)


def test_result_within_budget_and_in_order() -> None:
    pool = _pool([50, 80, 30, 120, 60, 90, 40, 70] * 4)
    budget = _tok(pool) // 3
    kept = chunked.chunked_select(pool, budget, _longest_first, 250)
    assert _tok(kept) <= budget
    ids = [s.id for s in kept]
    assert ids == sorted(ids)


def test_unused_share_spills_forward() -> None:
    pool = _pool([30] * 12)
    chunks = chunked.split_chunks(pool, 150)
    first = {s.id for s in chunks[0]}
    budget = _tok(pool) // 2
    seen: list[int] = []

    def stingy_first(p: list[Segment], b: int) -> list[Segment]:
        seen.append(b)
        return [] if {s.id for s in p} == first else _longest_first(p, b)

    kept = chunked.chunked_select(pool, budget, stingy_first, 150)
    shares = chunked.chunk_shares(chunks, budget)
    assert seen[1] > shares[1]
    assert _tok(kept) <= budget
    assert _tok(kept) > sum(shares[1:]) - count_tokens(pool[0].text)


def test_single_chunk_equals_plain_selection() -> None:
    pool = _pool([50, 80, 30, 120, 60])
    budget = _tok(pool) // 2
    got = chunked.chunked_select(pool, budget, _longest_first, f_tokens=10**9)
    assert [s.id for s in got] == [s.id for s in _longest_first(pool, budget)]


def test_topup_spends_leftover_without_exceeding_budget() -> None:
    pool = _pool([200, 20, 20, 200, 20, 20, 200, 20, 20])
    f = 700
    budget = _tok(pool) * 3 // 4
    calls: list[int] = []

    def tracking(p: list[Segment], b: int) -> list[Segment]:
        calls.append(b)
        return _longest_first(p, b)

    kept = chunked.chunked_select(pool, budget, tracking, f)
    n_chunks = len(chunked.split_chunks(pool, f))
    assert len(calls) == n_chunks + 1
    assert calls[-1] > 0
    assert _tok(kept) <= budget
    assert len({s.id for s in kept}) == len(kept)
    assert [s.id for s in kept] == sorted(s.id for s in kept)
    assert _tok(kept) >= budget - min(count_tokens(s.text) for s in pool)

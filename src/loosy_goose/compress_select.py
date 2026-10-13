from __future__ import annotations

import heapq
import lzma
import zlib
from typing import Literal, Protocol

from loosy_goose.budget import ProtectPolicy, forced_mask
from loosy_goose.segment import Segment
from loosy_goose.tokens import count_tokens

Codec = Literal["zlib", "lzma"]

# Bump whenever any score or selection output changes for the same input.
SELECT_VERSION: int = 1

_SEP = b"\n"
# 16 MiB dictionary so every earlier byte of a transcript stays in reach; zlib's 32 KiB window is
# the contrast the two codecs exist to measure.
_LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 1 | lzma.PRESET_EXTREME, "dict_size": 1 << 24}]


def _lzma_len(data: bytes) -> int:
    return len(lzma.compress(data, format=lzma.FORMAT_RAW, filters=_LZMA_FILTERS))


class _Context(Protocol):
    def gain(self, i: int) -> int: ...
    def accept(self, i: int) -> None: ...


class _ZlibContext:
    def __init__(self, data: list[bytes]) -> None:
        self._data = data
        self._c = zlib.compressobj(6, zlib.DEFLATED, -15)
        self._empty = True

    def _framed(self, i: int) -> bytes:
        return self._data[i] if self._empty else _SEP + self._data[i]

    def gain(self, i: int) -> int:
        fork = self._c.copy()
        return len(fork.compress(self._framed(i)) + fork.flush())

    def accept(self, i: int) -> None:
        self._c.compress(self._framed(i))
        # Sync-flush so a later fork's output holds only the candidate's bytes, never pending
        # bytes of already-kept text.
        self._c.flush(zlib.Z_SYNC_FLUSH)
        self._empty = False


class _LzmaContext:
    # LZMACompressor has no copy(), so every gain recompresses the whole kept buffer. `_memo`
    # holds len(C(K + s)) per candidate for the current K, so accepting a just-evaluated
    # candidate costs no extra compression.
    def __init__(self, data: list[bytes]) -> None:
        self._data = data
        self._kept = bytearray()
        self._empty = True
        self._base: int | None = None
        self._memo: dict[int, int] = {}

    def _framed(self, i: int) -> bytes:
        return self._data[i] if self._empty else _SEP + self._data[i]

    def gain(self, i: int) -> int:
        if self._base is None:
            self._base = _lzma_len(bytes(self._kept))
        total = self._memo.get(i)
        if total is None:
            total = _lzma_len(bytes(self._kept) + self._framed(i))
            self._memo[i] = total
        return total - self._base

    def accept(self, i: int) -> None:
        self._kept += self._framed(i)
        self._empty = False
        self._base = self._memo.get(i)
        self._memo = {}


def _context(data: list[bytes], codec: Codec) -> _Context:
    if codec == "zlib":
        return _ZlibContext(data)
    if codec == "lzma":
        return _LzmaContext(data)
    raise ValueError(f"unknown codec: {codec!r}")


def greedy_select(
    segments: list[Segment],
    budget: int,
    codec: Codec,
    *,
    lazy: bool,
    protect: ProtectPolicy = "none",
) -> list[Segment]:
    if codec == "lzma" and not lazy:
        raise ValueError(
            "exact greedy is infeasible for lzma (no forkable compressor state: every evaluation "
            "recompresses the whole kept buffer); use lazy=True"
        )
    if not segments:
        return []
    st = _Seeded(segments, codec, protect)
    if st.used < budget:
        run = _lazy_greedy if lazy else _exact_greedy
        run(st.ctx, st.candidates, st.costs, st.denom, st.keep, budget - st.used)
    return [s for s, k in zip(segments, st.keep, strict=True) if k]


class _Seeded:
    def __init__(self, segments: list[Segment], codec: Codec, protect: ProtectPolicy) -> None:
        self.ctx = _context([s.text.encode("utf-8") for s in segments], codec)
        self.costs = [count_tokens(s.text) for s in segments]
        self.denom = [max(c, 1) for c in self.costs]
        self.keep = forced_mask(segments, protect)
        self.used = 0
        for i, forced in enumerate(self.keep):
            if forced:
                self.used += self.costs[i]
                self.ctx.accept(i)
        self.candidates = [i for i, forced in enumerate(self.keep) if not forced]


def _best_density(ctx: _Context, ascending: list[int], denom: list[int]) -> int:
    # Strict `>` over ascending indices breaks ties toward the earliest segment.
    best = -1
    best_score = 0.0
    for i in ascending:
        score = ctx.gain(i) / denom[i]
        if best < 0 or score > best_score:
            best, best_score = i, score
    return best


def _exact_greedy(
    ctx: _Context,
    candidates: list[int],
    costs: list[int],
    denom: list[int],
    keep: list[bool],
    remaining: int,
) -> None:
    pool = candidates
    while True:
        # The budget only shrinks, so a candidate that no longer fits never will again.
        pool = [i for i in pool if costs[i] <= remaining]
        if not pool:
            break
        best = _best_density(ctx, pool, denom)
        ctx.accept(best)
        keep[best] = True
        remaining -= costs[best]
        pool = [i for i in pool if i != best]


def _lazy_greedy(
    ctx: _Context,
    candidates: list[int],
    costs: list[int],
    denom: list[int],
    keep: list[bool],
    remaining: int,
) -> None:
    # Minoux lazy greedy: a stale bound is trusted as an upper bound on the candidate's current
    # density. That holds only if gains never grow as context grows (submodularity), which
    # compressed length satisfies only approximately; tests compare against `_exact_greedy` on
    # zlib to bound the gap.
    heap: list[tuple[float, int, int]] = []
    for i in candidates:
        if costs[i] <= remaining:
            heap.append((-ctx.gain(i) / denom[i], i, 0))
    heapq.heapify(heap)
    picks = 0
    while heap:
        _, i, stamp = heapq.heappop(heap)
        if costs[i] > remaining:
            continue
        if stamp == picks:
            ctx.accept(i)
            keep[i] = True
            remaining -= costs[i]
            picks += 1
        else:
            heapq.heappush(heap, (-ctx.gain(i) / denom[i], i, picks))

from __future__ import annotations

import random
import string

import pytest

from loosy_goose.compress_select import Codec, _context, greedy_select
from loosy_goose.segment import Segment, SegmentKind, extract_atoms
from loosy_goose.tokens import count_tokens

VARIANTS: list[tuple[Codec, bool]] = [("zlib", False), ("zlib", True), ("lzma", True)]


def _seg(i: int, text: str, kind: SegmentKind = "prose", protected: bool = False) -> Segment:
    return Segment(i, i, "assistant", kind, text, protected, extract_atoms(text))


def _words(seed: int, chars: int) -> str:
    rng = random.Random(seed)
    words: list[str] = []
    size = 0
    while size < chars:
        word = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
        words.append(word)
        size += len(word) + 1
    return " ".join(words)


def _pool(n: int, chars: int, seed: int = 0) -> list[Segment]:
    return [_seg(i, _words(seed * 1000 + i, chars)) for i in range(n)]


def _cost(segs: list[Segment]) -> int:
    return sum(count_tokens(s.text) for s in segs)


@pytest.mark.parametrize(("codec", "lazy"), VARIANTS)
def test_greedy_respects_budget_and_input_order(codec: Codec, lazy: bool) -> None:
    segs = _pool(12, 300)
    budget = _cost(segs) // 3
    kept = greedy_select(segs, budget, codec, lazy=lazy)
    assert kept
    assert _cost(kept) <= budget
    assert [s.id for s in kept] == sorted(s.id for s in kept)


@pytest.mark.parametrize(("codec", "lazy"), VARIANTS)
def test_greedy_keeps_one_copy_of_a_duplicate(codec: Codec, lazy: bool) -> None:
    dup = _words(99, 1500)
    distinct = [_words(i, 300) for i in range(6)]
    texts = [distinct[0], dup, distinct[1], distinct[2], dup, *distinct[3:]]
    segs = [_seg(i, t) for i, t in enumerate(texts)]
    assert sum(len(t) for t in texts) < 32 * 1024
    budget = _cost(segs) - 1
    kept = greedy_select(segs, budget, codec, lazy=lazy)
    assert sum(1 for s in kept if s.text == dup) == 1
    assert {s.text for s in kept} >= set(distinct)


def _vocab_text(seed: int, vocab: int, chars: int) -> str:
    rng = random.Random(seed)
    pool = _words(seed + 10_000, vocab * 7).split()[:vocab]
    out: list[str] = []
    while sum(len(w) + 1 for w in out) < chars:
        out.append(rng.choice(pool))
    return " ".join(out)


def test_zlib_exact_and_lazy_agree_on_small_case() -> None:
    # Sanity check only, not a guarantee: lazy greedy is exact only under submodular gains, which
    # compressed length satisfies approximately. Near-incompressible segments of equal density
    # do break the tie differently (their gains grow slightly with context), so the crafted pool
    # spreads densities apart with vocabularies of very different sizes.
    segs = [_seg(i, _vocab_text(i, 4 << (2 * i), 600)) for i in range(6)]
    segs.append(_seg(6, segs[4].text))
    segs.append(_seg(7, segs[5].text[:300] + " " + _vocab_text(50, 3, 300)))
    budget = _cost(segs) // 2
    exact = greedy_select(segs, budget, "zlib", lazy=False)
    lazy = greedy_select(segs, budget, "zlib", lazy=True)
    assert [s.id for s in exact] == [s.id for s in lazy]


def _repeat_gain_share(texts: list[str], codec: Codec) -> float:
    """Gain of the last text given all the others kept, over its gain given nothing kept."""
    data = [t.encode("utf-8") for t in texts]
    cold = _context(data, codec).gain(len(data) - 1)
    ctx = _context(data, codec)
    for i in range(len(data) - 1):
        ctx.accept(i)
    return ctx.gain(len(data) - 1) / cold


def test_window_difference_between_codecs() -> None:
    # The reason both codecs exist: a repeat beyond zlib's 32 KiB window is invisible to it,
    # while lzma's 16 MiB dictionary still finds it.
    body = _words(42, 2000)
    filler = [_words(500 + k, 1000) for k in range(45)]
    texts = [body, *filler, body]
    assert sum(len(t) for t in filler) > 40 * 1024
    assert _repeat_gain_share(texts, "lzma") < 0.2
    assert _repeat_gain_share(texts, "zlib") > 0.8


def test_lzma_exact_greedy_raises() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        greedy_select(_pool(3, 100), 1000, "lzma", lazy=False)


@pytest.mark.parametrize(("codec", "lazy"), VARIANTS)
def test_forced_segments_always_kept_and_charged(codec: Codec, lazy: bool) -> None:
    segs = _pool(8, 300)
    segs[2] = _seg(2, "def f(x):\n    return x * 2\n" * 10, kind="code", protected=True)
    segs[5] = _seg(5, segs[5].text, protected=True)
    forced_all = [segs[2], segs[5]]

    budget = _cost(forced_all) + count_tokens(segs[0].text)
    kept = greedy_select(segs, budget, codec, lazy=lazy, protect="all")
    assert all(s in kept for s in forced_all)
    assert _cost(kept) <= budget

    kept = greedy_select(segs, budget, codec, lazy=lazy, protect="code_only")
    assert segs[2] in kept
    assert _cost(kept) <= budget

    tiny = _cost(forced_all) - 1
    kept = greedy_select(segs, tiny, codec, lazy=lazy, protect="all")
    assert kept == forced_all


@pytest.mark.parametrize(("codec", "lazy"), VARIANTS)
def test_empty_and_whitespace_segments(codec: Codec, lazy: bool) -> None:
    assert greedy_select([], 100, codec, lazy=lazy) == []
    segs = [_seg(0, ""), _seg(1, "   \n  "), _seg(2, _words(1, 200))]
    kept = greedy_select(segs, 1000, codec, lazy=lazy)
    assert [s.id for s in kept] == [0, 1, 2]


@pytest.mark.parametrize(("codec", "lazy"), VARIANTS)
def test_deterministic(codec: Codec, lazy: bool) -> None:
    segs = _pool(10, 250, seed=7)
    segs.append(_seg(10, segs[4].text))
    budget = _cost(segs) // 2
    first = greedy_select(segs, budget, codec, lazy=lazy)
    second = greedy_select(segs, budget, codec, lazy=lazy)
    assert [s.id for s in first] == [s.id for s in second]

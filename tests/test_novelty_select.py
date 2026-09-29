from __future__ import annotations

import random
import string

import numpy as np
import pytest

from loosy_goose.budget import ProtectPolicy, select_with_budget
from loosy_goose.novelty_select import novelty_select, novelty_share, plain_words, word_sets
from loosy_goose.segment import Segment, SegmentKind, extract_atoms
from loosy_goose.tokens import count_tokens


def _seg(i: int, text: str, kind: SegmentKind = "prose", protected: bool = False) -> Segment:
    return Segment(i, i, "assistant", kind, text, protected, extract_atoms(text))


def _words(rng: random.Random, n: int) -> str:
    return " ".join(
        "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8)))
        for _ in range(n)
    )


def _pool(n: int, seed: int) -> list[Segment]:
    rng = random.Random(seed)
    segs: list[Segment] = []
    for i in range(n):
        kind: SegmentKind = rng.choice(["prose", "code", "tool_result"])
        segs.append(_seg(i, _words(rng, rng.randint(5, 60)), kind, protected=rng.random() < 0.2))
    return segs


def _cost(segs: list[Segment]) -> int:
    return sum(count_tokens(s.text) for s in segs)


def test_plain_words_are_lowercase_alphabetic_runs_of_three_or_more() -> None:
    words = plain_words("Read src/Foo_bar.py: x=42, ok, the HelperFn at line 7 -- déjà vu")
    assert words == {"read", "src", "foo", "bar", "the", "helperfn", "line", "déjà"}


def test_words_in_more_than_half_the_segments_are_dropped_as_boilerplate() -> None:
    texts = [
        "tool output alpha",
        "tool output bravo",
        "tool charlie",
        "delta echo alpha",
    ]
    sets = word_sets([_seg(i, t) for i, t in enumerate(texts)])
    # "tool" is in 3/4 segments (> half): dropped. "output" and "alpha" are in exactly half: kept.
    assert sets == [
        {"output", "alpha"},
        {"output", "bravo"},
        {"charlie"},
        {"delta", "echo", "alpha"},
    ]


def test_novelty_share_only_falls_as_the_seen_set_grows() -> None:
    rng = random.Random(3)
    vocab = [f"w{c}{d}x" for c in string.ascii_lowercase for d in string.ascii_lowercase[:4]]
    for _ in range(200):
        words = frozenset(rng.sample(vocab, rng.randint(1, 20)))
        seen: set[str] = set()
        last = novelty_share(words, seen)
        assert last == 1.0
        for _ in range(5):
            seen |= set(rng.sample(vocab, rng.randint(1, 15)))
            share = novelty_share(words, seen)
            assert share <= last
            last = share
    assert novelty_share(frozenset(), set(vocab)) == 1.0


@pytest.mark.parametrize("protect", ["none", "all", "code_only"])
@pytest.mark.parametrize("seed", range(8))
def test_alpha_zero_reproduces_the_plain_budget_fill(protect: ProtectPolicy, seed: int) -> None:
    segs = _pool(40, seed)
    rng = np.random.default_rng(seed)
    scores = rng.random(len(segs))
    # Ties exercise the earlier-index tiebreak on both sides.
    scores[::7] = scores[0]
    for frac in (0.05, 0.25, 0.5, 0.9):
        budget = int(_cost(segs) * frac)
        want = select_with_budget(segs, scores, budget, protect)
        got = novelty_select(segs, scores, budget, 0.0, protect)
        assert [s.id for s in got] == [s.id for s in want]


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("seed", range(4))
def test_budget_is_respected_and_input_order_kept(alpha: float, seed: int) -> None:
    segs = _pool(40, seed)
    scores = np.random.default_rng(seed).random(len(segs))
    for frac in (0.1, 0.3, 0.6):
        budget = int(_cost(segs) * frac)
        kept = novelty_select(segs, scores, budget, alpha)
        assert _cost(kept) <= budget
        assert [s.id for s in kept] == sorted(s.id for s in kept)


@pytest.mark.parametrize("protect", ["all", "code_only"])
def test_protected_segments_are_always_kept_even_over_budget(protect: ProtectPolicy) -> None:
    segs = _pool(30, 11)
    scores = np.random.default_rng(11).random(len(segs))
    forced = {s.id for s in segs if (s.protected if protect == "all" else s.kind == "code")}
    assert forced
    for budget in (0, _cost(segs) // 4):
        kept = {s.id for s in novelty_select(segs, scores, budget, 1.0, protect)}
        assert forced <= kept


def test_a_repeat_of_a_kept_segment_loses_to_novel_text() -> None:
    rng = random.Random(5)
    base = _words(rng, 30)
    novel = _words(rng, 30)
    filler = [_words(rng, 30) for _ in range(4)]
    segs = [_seg(i, t) for i, t in enumerate([base, base + " again", novel, *filler])]
    scores = np.array([1.0, 0.9, 0.5, 0.0, 0.0, 0.0, 0.0])
    # Room for base plus exactly one of the repeat and the novel segment.
    budget = count_tokens(base) + max(count_tokens(segs[1].text), count_tokens(novel))
    plain = {s.id for s in novelty_select(segs, scores, budget, 0.0)}
    discounted = {s.id for s in novelty_select(segs, scores, budget, 1.0)}
    assert {0, 1} <= plain and 2 not in plain
    assert {0, 2} <= discounted and 1 not in discounted


def test_protected_words_count_as_already_seen() -> None:
    rng = random.Random(6)
    shared = _words(rng, 30)
    other = _words(rng, 30)
    segs = [
        _seg(0, shared, protected=True),
        _seg(1, shared),
        _seg(2, other),
        *[_seg(i, _words(rng, 30)) for i in range(3, 7)],
    ]
    scores = np.array([0.0, 1.0, 0.6, 0.0, 0.0, 0.0, 0.0])
    budget = count_tokens(shared) + max(count_tokens(shared), count_tokens(other)) + 1
    kept = [s.id for s in novelty_select(segs, scores, budget, 1.0, "all")]
    assert kept == [0, 2]


def test_negative_scores_are_rejected_only_when_novelty_applies() -> None:
    segs = _pool(5, 0)
    scores = np.array([-1.0, 0.5, 0.2, 0.1, 0.3])
    novelty_select(segs, scores, 50, 0.0)
    with pytest.raises(ValueError, match="non-negative"):
        novelty_select(segs, scores, 50, 1.0)
    with pytest.raises(ValueError, match="alpha"):
        novelty_select(segs, np.abs(scores), 50, -0.5)
    with pytest.raises(ValueError, match="one entry"):
        novelty_select(segs, scores[:3], 50, 1.0)
    assert novelty_select([], [], 50, 1.0) == []

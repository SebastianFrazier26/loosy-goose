from __future__ import annotations

import random
import string

import numpy as np
import pytest

from loosy_goose.budget import ProtectPolicy, forced_mask, select_with_budget
from loosy_goose.kind_select import (
    MIN_KIND_SEGMENTS,
    kind_budget_select,
    kind_budget_shares,
    kind_norm_scores,
    kind_space_scores,
)
from loosy_goose.segment import Segment, SegmentKind, extract_atoms
from loosy_goose.select import CompressConfig, score_segments
from loosy_goose.tokens import count_tokens


def _seg(i: int, text: str, kind: SegmentKind = "prose", protected: bool = False) -> Segment:
    return Segment(i, i, "assistant", kind, text, protected, extract_atoms(text))


def _words(rng: random.Random, n: int) -> str:
    return " ".join(
        "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8)))
        for _ in range(n)
    )


def _pool(n: int, seed: int, kinds: tuple[SegmentKind, ...] | None = None) -> list[Segment]:
    rng = random.Random(seed)
    choices: list[SegmentKind] = list(kinds or ("prose", "code", "tool_use", "tool_result"))
    return [
        _seg(i, _words(rng, rng.randint(5, 60)), rng.choice(choices), protected=rng.random() < 0.2)
        for i in range(n)
    ]


def _cost(segs: list[Segment]) -> int:
    return sum(count_tokens(s.text) for s in segs)


def _x(n: int, seed: int, d: int = 12) -> np.ndarray:
    x = np.random.default_rng(seed).normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# --- kind-space --------------------------------------------------------------------------------


def test_kind_space_scores_each_big_kind_in_its_own_space_at_mean_one() -> None:
    segs = [_seg(i, f"text {i}", "prose" if i < 8 else "tool_result") for i in range(20)]
    x = _x(20, 0)
    cfg = CompressConfig()
    got = kind_space_scores(segs, x, cfg)
    prose, tool = list(range(8)), list(range(8, 20))
    own = score_segments(x[prose], cfg)
    np.testing.assert_allclose(got[prose], own / own.mean())
    assert got[prose].mean() == pytest.approx(1.0)
    assert got[tool].mean() == pytest.approx(1.0)


def test_kind_space_removes_the_k_over_n_scale_bias() -> None:
    # Raw leverage of a small kind scored alone averages k/n with small n, so it would dwarf a big
    # kind's scores; after dividing by the kind mean both sit at mean 1.
    segs = [_seg(i, f"t{i}", "code" if i < 6 else "tool_result") for i in range(46)]
    x = _x(46, 1)
    cfg = CompressConfig()
    small = score_segments(x[:6], cfg)
    big = score_segments(x[6:], cfg)
    assert small.mean() > 2 * big.mean()
    got = kind_space_scores(segs, x, cfg)
    assert got[:6].mean() == pytest.approx(got[6:].mean())


def test_kind_space_tiny_kind_keeps_its_shared_score_on_the_pool_mean_scale() -> None:
    segs = [_seg(i, f"t{i}", "prose" if i < 3 else "tool_result") for i in range(15)]
    x = _x(15, 2)
    cfg = CompressConfig()
    shared = score_segments(x, cfg)
    got = kind_space_scores(segs, x, cfg)
    np.testing.assert_allclose(got[:3], shared[:3] / shared.mean())
    assert len([s for s in segs if s.kind == "prose"]) < MIN_KIND_SEGMENTS


def test_kind_space_with_every_kind_tiny_is_shared_leverage_order() -> None:
    segs = [_seg(i, f"t{i}", ("prose", "code", "tool_use")[i % 3]) for i in range(9)]
    x = _x(9, 3)
    shared = score_segments(x, CompressConfig())
    got = kind_space_scores(segs, x, CompressConfig())
    np.testing.assert_allclose(got, shared / shared.mean())


def test_kind_space_rejects_mismatched_rows() -> None:
    with pytest.raises(ValueError):
        kind_space_scores([_seg(0, "a")], _x(2, 0), CompressConfig())
    assert kind_space_scores([], np.zeros((0, 0)), CompressConfig()).shape == (0,)


# --- kind-norm ---------------------------------------------------------------------------------


def test_rank_norm_puts_every_big_kind_on_zero_to_one() -> None:
    segs = [_seg(i, "x", "prose" if i < 5 else "tool_result") for i in range(12)]
    scores = np.array([5.0, 1.0, 3.0, 2.0, 4.0] + [0.01 * i for i in range(7)])
    got = kind_norm_scores(segs, scores, "rank")
    np.testing.assert_allclose(got[:5], [1.0, 0.0, 0.5, 0.25, 0.75])
    np.testing.assert_allclose(got[5:], np.linspace(0.0, 1.0, 7))


def test_z_norm_is_zero_mean_unit_std_within_kind() -> None:
    rng = np.random.default_rng(0)
    segs = [_seg(i, "x", "code" if i < 6 else "tool_use") for i in range(16)]
    scores = np.concatenate([rng.normal(10, 3, 6), rng.normal(0.1, 0.01, 10)])
    got = kind_norm_scores(segs, scores, "z")
    for idx in (slice(0, 6), slice(6, 16)):
        assert got[idx].mean() == pytest.approx(0.0, abs=1e-12)
        assert got[idx].std() == pytest.approx(1.0)


def test_zero_variance_kind_is_all_half_rank_and_all_zero_z() -> None:
    segs = [_seg(i, "x", "prose" if i < 5 else "code") for i in range(10)]
    scores = np.array([0.3] * 5 + [0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(kind_norm_scores(segs, scores, "rank")[:5], [0.5] * 5)
    np.testing.assert_allclose(kind_norm_scores(segs, scores, "z")[:5], [0.0] * 5)


def test_rank_ties_share_their_average_rank() -> None:
    segs = [_seg(i, "x") for i in range(5)]
    got = kind_norm_scores(segs, [1.0, 2.0, 2.0, 3.0, 0.0], "rank")
    np.testing.assert_allclose(got, [0.25, 0.625, 0.625, 1.0, 0.0])


def test_single_segment_is_half_rank_and_zero_z() -> None:
    segs = [_seg(0, "x")]
    assert kind_norm_scores(segs, [7.0], "rank").tolist() == [0.5]
    assert kind_norm_scores(segs, [7.0], "z").tolist() == [0.0]
    # min_segments=1 makes the lone segment its own kind group; same answer.
    mixed = [_seg(0, "x", "code"), _seg(1, "y", "prose"), _seg(2, "z", "prose")]
    assert kind_norm_scores(mixed, [9.0, 1.0, 2.0], "rank", min_segments=1)[0] == 0.5
    assert kind_norm_scores(mixed, [9.0, 1.0, 2.0], "z", min_segments=1)[0] == 0.0


@pytest.mark.parametrize("norm", ["rank", "z"])
def test_tiny_kind_is_normalised_against_the_whole_pool(norm: str) -> None:
    segs = [_seg(i, "x", "thinking" if i < 2 else "tool_result") for i in range(10)]
    scores = np.array([9.0, 0.5] + [float(i) for i in range(8)])
    got = kind_norm_scores(segs, scores, norm)  # type: ignore[arg-type]
    pool = kind_norm_scores([_seg(i, "x") for i in range(10)], scores, norm)  # type: ignore[arg-type]
    np.testing.assert_allclose(got[:2], pool[:2])
    assert not np.allclose(got[2:], pool[2:])


def test_rank_and_z_differ_on_skewed_scores() -> None:
    segs = [_seg(i, "x", "prose" if i < 6 else "tool_result") for i in range(14)]
    scores = np.array([100.0, 1, 1.1, 1.2, 1.3, 1.4] + [0.1 * i for i in range(8)])
    rank = kind_norm_scores(segs, scores, "rank")
    z = kind_norm_scores(segs, scores, "z")
    assert list(np.argsort(-rank, kind="stable")) != list(np.argsort(-z, kind="stable"))


def test_kind_norm_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        kind_norm_scores([_seg(0, "a")], [1.0, 2.0], "rank")
    with pytest.raises(ValueError):
        kind_norm_scores([_seg(0, "a")], [1.0], "minmax")  # type: ignore[arg-type]


# --- kind-budget -------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("protect", ["none", "code_only", "all"])
@pytest.mark.parametrize("frac", [0.05, 0.25, 0.5, 0.9])
def test_kind_budget_never_exceeds_the_shared_budget(
    seed: int, protect: ProtectPolicy, frac: float
) -> None:
    segs = _pool(40, seed)
    scores = np.random.default_rng(seed).random(40)
    budget = int(_cost(segs) * frac)
    kept = kind_budget_select(segs, scores, budget, protect)
    forced = [s for s, f in zip(segs, forced_mask(segs, protect), strict=True) if f]
    assert _cost(kept) <= max(budget, _cost(forced))
    assert {id(s) for s in forced} <= {id(s) for s in kept}
    if _cost(forced) >= budget:
        assert _cost(kept) == _cost(forced)
    assert [s.id for s in kept] == sorted(s.id for s in kept)


def test_kind_budget_shares_are_floored_proportional_and_skip_tiny_kinds() -> None:
    segs = (
        [_seg(i, "word " * 30, "prose") for i in range(5)]
        + [_seg(5 + i, "word " * 10, "tool_result") for i in range(10)]
        + [_seg(15 + i, "word " * 50, "code") for i in range(2)]
    )
    tokens = {k: _cost([s for s in segs if s.kind == k]) for k in ("prose", "tool_result", "code")}
    total = sum(tokens.values())
    shares = kind_budget_shares(segs, 1000)
    assert set(shares) == {"prose", "tool_result"}
    assert shares["prose"] == (1000 * tokens["prose"]) // total
    assert shares["tool_result"] == (1000 * tokens["tool_result"]) // total
    assert sum(shares.values()) <= 1000
    assert kind_budget_shares(segs, 0) == {}


def test_kind_budget_spills_what_a_small_kind_cannot_spend() -> None:
    # code has 2 segments, under MIN_KIND_SEGMENTS: no slice of its own, so only the spill pass
    # (fed by the tiny kind's unallocated share and the slices' knapsack leftovers) can keep it.
    segs = (
        [_seg(i, "alpha " * 40, "prose") for i in range(6)]
        + [_seg(6 + i, "beta " * 3, "tool_use") for i in range(5)]
        + [_seg(11 + i, "gamma " * 20, "code") for i in range(2)]
    )
    scores = np.array([1.0] * 6 + [0.5] * 5 + [0.9] * 2)
    budget = _cost(segs) // 2
    kept = kind_budget_select(segs, scores, budget, "none")
    kinds = {s.kind for s in kept}
    assert "code" in kinds  # tiny kind: no slice, only reachable through the spill pass
    assert _cost(kept) <= budget
    # Every slice plus the spill is spent down to what no remaining segment fits in.
    left = budget - _cost(kept)
    assert all(count_tokens(s.text) > left for s in segs if s not in kept)


def test_kind_budget_with_every_kind_tiny_is_the_plain_fill() -> None:
    segs = [_seg(i, _words(random.Random(i), 10 + i), ("prose", "code")[i % 2]) for i in range(8)]
    scores = np.random.default_rng(4).random(8)
    budget = _cost(segs) // 3
    got = kind_budget_select(segs, scores, budget, "none")
    assert [s.id for s in got] == [s.id for s in select_with_budget(segs, scores, budget)]


def test_kind_budget_tracks_pool_shares_where_shared_scores_would_not() -> None:
    # Shared scores favour prose outright; the plain fill spends nearly everything on it, the
    # kind-budget arm holds each kind near its pool share.
    segs = [_seg(i, "word " * 20, "prose" if i < 10 else "tool_result") for i in range(30)]
    scores = np.array([10.0] * 10 + [1.0] * 20)
    budget = _cost(segs) // 3
    plain = select_with_budget(segs, scores, budget)
    kb = kind_budget_select(segs, scores, budget, "none")

    def prose_share(kept: list[Segment]) -> float:
        return _cost([s for s in kept if s.kind == "prose"]) / _cost(kept)

    assert prose_share(plain) == 1.0
    # Ten equal segments fit the budget, so the floored slices leave one segment's worth (0.1) to
    # the spill, which prose's higher score wins.
    assert prose_share(kb) == pytest.approx(1 / 3, abs=0.1)


@pytest.mark.parametrize("seed", range(4))
def test_every_arm_is_deterministic(seed: int) -> None:
    segs = _pool(30, seed)
    x = _x(30, seed)
    cfg = CompressConfig()
    scores = score_segments(x, cfg)
    budget = _cost(segs) // 4
    for _ in range(2):
        np.testing.assert_array_equal(
            kind_space_scores(segs, x, cfg), kind_space_scores(segs, x.copy(), cfg)
        )
        for norm in ("rank", "z"):
            np.testing.assert_array_equal(
                kind_norm_scores(segs, scores, norm),  # type: ignore[arg-type]
                kind_norm_scores(segs, scores.copy(), norm),  # type: ignore[arg-type]
            )
        a = kind_budget_select(segs, scores, budget)
        b = kind_budget_select(list(segs), scores.copy(), budget)
        assert [s.id for s in a] == [s.id for s in b]


def test_kind_budget_rejects_mismatched_scores_and_handles_empty() -> None:
    assert kind_budget_select([], [], 10) == []
    with pytest.raises(ValueError):
        kind_budget_select([_seg(0, "a")], [1.0, 2.0], 10)

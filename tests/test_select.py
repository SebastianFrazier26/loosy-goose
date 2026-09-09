from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loosy_goose.segment import Segment
from loosy_goose.select import (
    CompressConfig,
    center,
    clamp_drop_top,
    compress,
    cur_residual_order,
    cur_residual_scores,
    embed_segments,
    greedy_cur_select,
    label_directions,
    leverage_scores,
    make_compress,
    rank_for_energy,
    ridge_leverage_scores,
    score_segments,
    svd_energy,
)
from loosy_goose.tokens import count_tokens


def _seg(i: int, text: str, *, kind: str = "prose", protected: bool = False) -> Segment:
    return Segment(id=i, turn=i, role="assistant", kind=kind, text=text, protected=protected)  # type: ignore[arg-type]


def _low_rank(n: int, d: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, k)) @ rng.normal(size=(k, d))


def test_svd_energy_recovers_rank_of_low_rank_matrix() -> None:
    k = 4
    x = _low_rank(40, 16, k)
    u, s, vt, energy = svd_energy(x)
    assert u.shape == (40, 16) and vt.shape == (16, 16)
    # Centring removes at most one direction, so the centred rank is k or k-1 and everything
    # past that is numerical noise.
    assert energy[k - 1] == pytest.approx(1.0, abs=1e-9)
    assert rank_for_energy(s, 0.95) <= k
    assert rank_for_energy(s, 1.0) <= k


def test_svd_energy_is_cumulative_and_normalised() -> None:
    rng = np.random.default_rng(3)
    _, _, _, energy = svd_energy(rng.normal(size=(12, 7)))
    assert np.all(np.diff(energy) >= -1e-12)
    assert energy[-1] == pytest.approx(1.0)


def test_center_removes_mean() -> None:
    rng = np.random.default_rng(4)
    xc = center(rng.normal(size=(9, 5)) + 3.0)
    assert xc.mean(axis=0) == pytest.approx(np.zeros(5), abs=1e-12)


def test_rank_for_energy_thresholds() -> None:
    s = np.array([3.0, 2.0, 1.0])  # power 9, 4, 1 -> cumulative 9/14, 13/14, 1
    assert rank_for_energy(s, 0.5) == 1
    assert rank_for_energy(s, 0.9) == 2
    assert rank_for_energy(s, 0.99) == 3
    assert rank_for_energy(s, 1.0) == 3
    with pytest.raises(ValueError):
        rank_for_energy(s, 0.0)


def test_leverage_scores_sum_to_k() -> None:
    x = _low_rank(30, 10, 6, seed=1)
    u, _, _, _ = svd_energy(x)
    for k in (1, 3, 6):
        lev = leverage_scores(u, k)
        assert lev.shape == (30,)
        assert lev.sum() == pytest.approx(k)
        assert np.all(lev >= 0)
    with pytest.raises(ValueError):
        leverage_scores(u, 0)


def test_ridge_leverage_sums_to_effective_dimension_and_shrinks() -> None:
    x = _low_rank(25, 8, 5, seed=2)
    _, s, _, _ = svd_energy(x)
    lam = 2.0
    ridge = ridge_leverage_scores(x, lam)
    assert ridge.sum() == pytest.approx(float(np.sum(s**2 / (s**2 + lam))))
    plain = ridge_leverage_scores(x, 0.0)
    assert plain.sum() == pytest.approx(np.linalg.matrix_rank(center(x)), abs=1e-6)
    assert ridge.sum() < plain.sum()


def test_cur_residual_picks_orthogonal_rows_first() -> None:
    z = np.array(
        [
            [3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.9, 0.1, 0.0],
        ]
    )
    order, norms = cur_residual_order(z)
    assert order[:2] == [0, 1]
    assert norms[0] == pytest.approx(3.0)
    assert norms[1] == pytest.approx(3.0)
    # Every remaining row lies in span(e1, e2), so the pivot order stops after two picks.
    assert len(order) == 2
    assert np.all(np.diff(norms) <= 1e-12)


def test_cur_residual_scores_rank_pivots_above_tail_and_cover_all_rows() -> None:
    x = _low_rank(20, 6, 3, seed=5)
    u, s, _, _ = svd_energy(x)
    k = rank_for_energy(s, 0.999)
    scores = cur_residual_scores(u, s, k)
    assert scores.shape == (20,)
    order, norms = cur_residual_order(u[:, :k] * s[:k])
    assert len(order) == k
    assert scores[order] == pytest.approx(norms)
    tail = np.setdiff1d(np.arange(20), order)
    assert scores[tail].max() < norms.min()
    assert np.all(scores[tail] >= 0)


def test_greedy_cur_select_respects_budget_and_skips_rather_than_stops() -> None:
    z = np.array([[4.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]])
    costs = {0: 10, 1: 5, 2: 1}
    # Pivot order is 0, 1, 2; row 1 does not fit the budget after row 0 but row 2 still does.
    picks = greedy_cur_select(z, 3, lambda idx: sum(costs[i] for i in idx) <= 11)
    assert picks == [0, 2]


def _corpus() -> list[Segment]:
    return [
        _seg(0, "Parsing the transcript loader and jsonl records for the session file."),
        _seg(
            1,
            "def load(path):\n    return json.loads(path.read_text())",
            kind="code",
            protected=True,
        ),
        _seg(2, "Token budgets use tiktoken counts to decide how many segments to keep."),
        _seg(3, "Leverage scores from the singular vectors rank segments by subspace mass."),
        _seg(4, "The loader tolerates unknown block types and tallies them in meta."),
        _seg(5, '{"command": "pytest -q"}', kind="tool_use", protected=True),
        _seg(6, "Budget selection is greedy under the token cap with protected forced in."),
        _seg(7, "Singular value spectrum energy picks the rank for the semantic subspace."),
    ]


def _fake_embeddings(n: int, d: int = 16) -> np.ndarray:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(n, d))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_compress_respects_budget_order_and_forced_protected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segs = _corpus()
    monkeypatch.setattr(
        "loosy_goose.select.embed_texts", lambda texts, model_name: _fake_embeddings(len(texts))
    )
    total = sum(count_tokens(s.text) for s in segs)
    for strategy in ("leverage", "ridge", "cur_residual"):
        cfg = CompressConfig(strategy=strategy)  # type: ignore[arg-type]
        kept = compress(segs, 0.5, protect="all", config=cfg)
        ids = [s.id for s in kept]
        assert ids == sorted(ids)
        assert {1, 5} <= set(ids)
        protected_tokens = sum(count_tokens(s.text) for s in segs if s.protected)
        kept_tokens = sum(count_tokens(s.text) for s in kept)
        assert kept_tokens <= max(protected_tokens, -(-total * 0.5 // 1))
    none_kept = compress(segs, 0.3, protect="none")
    assert sum(count_tokens(s.text) for s in none_kept) <= -(-total * 0.3 // 1)
    assert compress([], 0.5) == []
    fn = make_compress(CompressConfig(strategy="cur_residual"))
    assert [s.id for s in fn(segs, 0.5)] == [
        s.id for s in compress(segs, 0.5, config=CompressConfig(strategy="cur_residual"))
    ]


def test_score_segments_strategies_disagree_but_share_shape() -> None:
    x = _fake_embeddings(12)
    lev = score_segments(x, CompressConfig(strategy="leverage", energy=0.9))
    cur = score_segments(x, CompressConfig(strategy="cur_residual", energy=0.9))
    ridge = score_segments(x, CompressConfig(strategy="ridge"))
    assert lev.shape == cur.shape == ridge.shape == (12,)
    assert score_segments(np.zeros((0, 3)), CompressConfig()).shape == (0,)
    assert score_segments(np.ones((1, 3)), CompressConfig()).tolist() == [1.0]


def test_label_directions_returns_n_words_per_direction() -> None:
    segs = _corpus()
    x = _fake_embeddings(len(segs))
    u, _, vt, _ = svd_energy(x)
    labels = label_directions(segs, vt, u, k=3, n_words=4)
    assert len(labels) == 3
    assert all(len(words) == 4 for words in labels)
    assert all(w == w.lower() and len(w) >= 3 for words in labels for w in words)
    with pytest.raises(ValueError):
        label_directions(segs, vt, u, k=vt.shape[0] + 1)


def test_embed_segments_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake(texts: list[str], model_name: str) -> np.ndarray:
        calls.append(len(texts))
        return _fake_embeddings(len(texts))

    monkeypatch.setattr("loosy_goose.select.embed_texts", fake)
    segs = _corpus()
    first = embed_segments(segs, "fake-model", cache_dir=tmp_path)
    second = embed_segments(segs, "fake-model", cache_dir=tmp_path)
    assert calls == [len(segs)]
    assert first == pytest.approx(second)
    embed_segments(segs, "other-model", cache_dir=tmp_path)
    assert len(calls) == 2


def test_embed_segments_truncates_long_text(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake(texts: list[str], model_name: str) -> np.ndarray:
        seen.extend(texts)
        return _fake_embeddings(len(texts))

    monkeypatch.setattr("loosy_goose.select.embed_texts", fake)
    embed_segments([_seg(0, "x" * 5000)], "fake-model")
    assert len(seen[0]) == 2000


@pytest.mark.slow
def test_embed_segments_real_model_smoke() -> None:
    segs = [
        _seg(0, "Fix the failing unit test in the parser."),
        _seg(1, "The parser test fails on empty input."),
        _seg(2, "Bake the cake for forty minutes."),
        _seg(3, "Preheat the oven before baking."),
    ]
    x = embed_segments(segs)
    assert x.shape[0] == 4
    assert np.linalg.norm(x, axis=1) == pytest.approx(np.ones(4), abs=1e-5)
    sim = x @ x.T
    assert sim[0, 1] > sim[0, 2]
    assert sim[2, 3] > sim[2, 0]


def test_clamp_drop_top_leaves_one_direction_alive() -> None:
    assert clamp_drop_top(0, 5) == 0
    assert clamp_drop_top(3, 5) == 3
    assert clamp_drop_top(9, 5) == 4
    assert clamp_drop_top(2, 1) == 0
    with pytest.raises(ValueError):
        clamp_drop_top(-1, 5)


def test_leverage_scores_drop_top_ignores_leading_directions() -> None:
    u = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    full = leverage_scores(u, 2)
    dropped = leverage_scores(u, 2, drop_top=1)
    assert full == pytest.approx([1.0, 1.0, 1.0])
    assert dropped == pytest.approx([0.0, 1.0, 0.64])


def test_drop_top_changes_which_segments_win() -> None:
    # Rows 0 and 1 are identical along the leading direction and differ only on the second one,
    # which is the situation All-but-the-Top targets: a shared boilerplate direction masking the
    # distinction that actually matters.
    u = np.array([[0.7, 0.1], [0.7, 0.6], [0.1, 0.2]])
    assert int(np.argmax(leverage_scores(u, 2))) == 1
    assert int(np.argmax(leverage_scores(u, 2, drop_top=1))) == 1
    assert leverage_scores(u, 2, drop_top=1)[0] < leverage_scores(u, 2)[0]


def test_ridge_drop_top_zeroes_the_leading_weight() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(8, 4))
    full = ridge_leverage_scores(x, 1.0)
    dropped = ridge_leverage_scores(x, 1.0, drop_top=1)
    assert np.all(dropped <= full + 1e-12)
    assert not np.allclose(full, dropped)


def test_cur_residual_drop_top_runs_and_stays_finite() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(9, 5))
    u, s, _, _ = svd_energy(x)
    k = rank_for_energy(s, 0.95)
    scores = cur_residual_scores(u, s, k, drop_top=1)
    assert scores.shape == (9,)
    assert np.all(np.isfinite(scores)) and np.all(scores >= 0)


def test_score_segments_threads_drop_top_through_every_strategy() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(10, 6))
    for strategy in ("leverage", "ridge", "cur_residual"):
        base = score_segments(x, CompressConfig(strategy=strategy))
        dropped = score_segments(x, CompressConfig(strategy=strategy, drop_top=2))
        assert not np.allclose(base, dropped), strategy


def test_drop_top_defaults_to_zero_so_phase_two_numbers_still_reproduce() -> None:
    assert CompressConfig().drop_top == 0

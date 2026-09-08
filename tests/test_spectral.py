import numpy as np
import pytest

from loosy_goose.spectral import (
    inverse_participation_ratio,
    participation_fraction,
    ppmi,
    truncated_svd,
)


def test_ipr_one_hot_is_one() -> None:
    v = np.zeros((10, 1))
    v[3, 0] = 5.0
    assert inverse_participation_ratio(v)[0] == pytest.approx(1.0)


def test_ipr_uniform_is_one_over_n() -> None:
    n = 64
    v = np.full((n, 1), 0.7)
    assert inverse_participation_ratio(v)[0] == pytest.approx(1.0 / n)


def test_ipr_axis_one_treats_rows_as_vectors() -> None:
    m = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    got = inverse_participation_ratio(m, axis=1)
    assert got == pytest.approx([1.0, 1.0 / 3.0])


def test_participation_fraction_inverts_ipr() -> None:
    n = 50
    uniform = np.full((n, 1), 1.0)
    one_hot = np.zeros((n, 1))
    one_hot[0, 0] = 1.0
    ipr = inverse_participation_ratio(np.hstack([uniform, one_hot]))
    frac = participation_fraction(ipr, n)
    assert frac == pytest.approx([1.0, 1.0 / n])


def test_participation_fraction_rejects_nonpositive_n() -> None:
    with pytest.raises(ValueError):
        participation_fraction(np.array([0.5]), 0)


def test_ppmi_independent_counts_is_zero() -> None:
    rows = np.array([[1.0], [2.0], [3.0]])
    cols = np.array([[4.0, 5.0, 6.0, 7.0]])
    counts = rows @ cols
    assert ppmi(counts) == pytest.approx(np.zeros_like(counts))


def test_ppmi_zero_where_counts_zero_and_finite() -> None:
    counts = np.array([[10.0, 0.0, 1.0], [0.0, 5.0, 0.0], [2.0, 0.0, 8.0]])
    out = ppmi(counts)
    assert np.all(np.isfinite(out))
    assert np.all(out[counts == 0] == 0.0)
    assert np.all(out >= 0.0)
    assert out[0, 0] > 0.0


def test_shifted_ppmi_never_exceeds_unshifted() -> None:
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 20, size=(12, 9)).astype(float)
    plain = ppmi(counts)
    shifted = ppmi(counts, shift=5.0)
    assert np.all(shifted <= plain + 1e-12)
    assert shifted.sum() < plain.sum()


def test_ppmi_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        ppmi(np.ones(4))


def test_truncated_svd_eig_weight_one_reconstructs_rank_k() -> None:
    rng = np.random.default_rng(1)
    k = 3
    m = rng.normal(size=(20, k)) @ rng.normal(size=(k, 15))
    w, s, vt = truncated_svd(m, rank=k, eig_weight=1.0)
    assert w.shape == (20, k)
    assert s.shape == (k,)
    assert vt.shape == (k, 15)
    assert w @ vt == pytest.approx(m, abs=1e-8)


def test_truncated_svd_eig_weight_zero_has_orthonormal_columns() -> None:
    rng = np.random.default_rng(2)
    m = rng.normal(size=(30, 12))
    w, _, _ = truncated_svd(m, rank=5)
    assert w.T @ w == pytest.approx(np.eye(5), abs=1e-10)


def test_truncated_svd_rejects_bad_rank() -> None:
    m = np.eye(4)
    with pytest.raises(ValueError):
        truncated_svd(m, rank=0)
    with pytest.raises(ValueError):
        truncated_svd(m, rank=5)

import numpy as np
import numpy.typing as npt
import scipy.linalg

FloatArray = npt.NDArray[np.floating]


def ppmi(counts: npt.ArrayLike, shift: float = 0.0, cds: float = 1.0) -> FloatArray:
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim != 2:
        raise ValueError("counts must be a 2-D word x context matrix")
    row_sums = c.sum(axis=1, keepdims=True)
    col_sums = c.sum(axis=0, keepdims=True) ** cds
    total = col_sums.sum()
    out = np.zeros_like(c)
    nz = c > 0
    # log() is only evaluated on observed pairs: unobserved cells stay exactly 0 rather than
    # passing -inf through the clamp, which also keeps the result finite under cds < 1.
    ratio = (c[nz] * total) / (
        np.broadcast_to(row_sums, c.shape)[nz] * np.broadcast_to(col_sums, c.shape)[nz]
    )
    pmi = np.log(ratio)
    if shift > 1.0:
        pmi -= np.log(shift)
    out[nz] = np.maximum(pmi, 0.0)
    return out


def inverse_participation_ratio(vectors: npt.ArrayLike, axis: int = 0) -> FloatArray:
    v = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(v, axis=axis, keepdims=True)
    unit = np.divide(v, norms, out=np.zeros_like(v), where=norms > 0)
    result: FloatArray = np.sum(unit**4, axis=axis)
    return result


def participation_fraction(ipr: npt.ArrayLike, n: int) -> FloatArray:
    if n <= 0:
        raise ValueError("n must be positive")
    i = np.asarray(ipr, dtype=np.float64)
    result: FloatArray = 1.0 / (i * n)
    return result


def truncated_svd(
    m: npt.ArrayLike, rank: int, eig_weight: float = 0.0
) -> tuple[FloatArray, FloatArray, FloatArray]:
    a = np.asarray(m, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("m must be 2-D")
    if rank < 1 or rank > min(a.shape):
        raise ValueError(f"rank must be in [1, {min(a.shape)}]")
    u, s, vt = scipy.linalg.svd(a, full_matrices=False, check_finite=False)
    u_k, s_k, vt_k = u[:, :rank], s[:rank], vt[:rank, :]
    # Default eig_weight=0 (W = U_k, no singular-value scaling) follows Levy, Goldberg & Dagan
    # 2015: weighting by S hurts downstream similarity tasks, and Shin et al. 2018 analyse
    # exactly this unweighted form. eig_weight=1 recovers the classical U_k S_k factor.
    w = u_k if eig_weight == 0.0 else u_k * (s_k**eig_weight)
    return w, s_k, vt_k

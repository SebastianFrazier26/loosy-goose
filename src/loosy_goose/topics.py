from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

from loosy_goose.budget import ProtectPolicy, select_by_score
from loosy_goose.cooccur import Vocab, cooccurrence, ppmi_sparse, tokenize
from loosy_goose.segment import Segment
from loosy_goose.spectral import (
    inverse_participation_ratio,
    participation_fraction,
    truncated_svd_sparse,
)

FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.intp]


@dataclass(frozen=True)
class TopicConfig:
    window: int = 2
    min_count: int = 2
    shift: float = 1.0
    cds: float = 0.75
    k: int | None = None
    drop_top: int = 1
    ipr_weight: bool = True


@dataclass
class EigenTopics:
    w: FloatArray
    s: FloatArray
    ipr: FloatArray
    participation: FloatArray
    vocab: Vocab

    @property
    def k(self) -> int:
        return int(self.w.shape[1])


def default_rank(n_vocab: int, n_segments: int, cap: int = 200) -> int:
    return max(1, min(cap, n_vocab // 4, n_segments - 1, n_vocab - 1))


def fit_topics(ppmi_matrix: sp.csr_matrix, vocab: Vocab, k: int, seed: int = 0) -> EigenTopics:
    if ppmi_matrix.shape != (len(vocab), len(vocab)):
        raise ValueError("ppmi_matrix must be |V| x |V| for the given vocab")
    w, s, _ = truncated_svd_sparse(ppmi_matrix, rank=k, seed=seed)
    ipr = inverse_participation_ratio(w, axis=0)
    return EigenTopics(
        w=w, s=s, ipr=ipr, participation=participation_fraction(ipr, len(vocab)), vocab=vocab
    )


def fit_from_token_lists(
    token_lists: Sequence[Sequence[str]],
    config: TopicConfig | None = None,
    *,
    n_segments: int | None = None,
    seed: int = 0,
) -> tuple[EigenTopics, sp.csr_matrix] | None:
    config = config or TopicConfig()
    vocab = Vocab.build(token_lists, min_count=config.min_count)
    if len(vocab) < 3:
        return None
    counts = cooccurrence(token_lists, vocab, window=config.window)
    if counts.nnz == 0:
        return None
    k = (
        config.k
        if config.k is not None
        else default_rank(len(vocab), n_segments or len(token_lists))
    )
    k = max(1, min(k, len(vocab) - 1))
    ppmi = ppmi_sparse(counts, cds=config.cds, shift=config.shift)
    return fit_topics(ppmi, vocab, k, seed=seed), counts


def density_order(topics: EigenTopics) -> IntArray:
    # Ascending IPR: index 0 is the densest eigenvector (Shin's "global bias" direction).
    return np.argsort(topics.ipr, kind="stable")


def top_words(topics: EigenTopics, j: int, n: int = 20) -> list[tuple[str, float]]:
    col = topics.w[:, j]
    order = np.argsort(-np.abs(col), kind="stable")[:n]
    return [(topics.vocab.words[i], float(col[i])) for i in order]


def segment_matrix(topics: EigenTopics, segments: Sequence[Segment]) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for r, seg in enumerate(segments):
        ids = topics.vocab.encode(tokenize(seg.text))
        if ids.size:
            uniq, cnt = np.unique(ids, return_counts=True)
            rows.extend([r] * uniq.size)
            cols.extend(uniq.tolist())
            vals.extend((cnt / ids.size).tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(segments), len(topics.vocab)))


def segment_scores(
    topics: EigenTopics,
    segments: Sequence[Segment],
    *,
    drop_top: int = 1,
    ipr_weight: bool = True,
) -> FloatArray:
    if not segments:
        return np.zeros(0, dtype=np.float64)
    vectors = np.asarray(segment_matrix(topics, segments) @ topics.w)
    weights = np.ones(topics.k, dtype=np.float64)
    if drop_top > 0:
        weights[density_order(topics)[: min(drop_top, topics.k)]] = 0.0
    if ipr_weight and topics.ipr.max() > 0:
        weights *= topics.ipr / topics.ipr.max()
    result: FloatArray = np.linalg.norm(vectors * weights, axis=1)
    return result


def compress(
    segments: list[Segment],
    keep_ratio: float,
    *,
    protect: ProtectPolicy = "all",
    recency: float = 0.0,
    seed: int = 0,
    config: TopicConfig | None = None,
) -> list[Segment]:
    config = config or TopicConfig()
    token_lists = [tokenize(s.text) for s in segments]
    fitted = fit_from_token_lists(token_lists, config, n_segments=len(segments), seed=seed)
    if fitted is None:
        # Too little text to span a basis: fall back to a flat score so the budget rule alone
        # decides (protected segments first, then earliest segments).
        scores: FloatArray = np.zeros(len(segments), dtype=np.float64)
    else:
        scores = segment_scores(
            fitted[0], segments, drop_top=config.drop_top, ipr_weight=config.ipr_weight
        )
    return select_by_score(segments, scores, keep_ratio, protect=protect, recency=recency)


def align_cosines(reference: FloatArray, other: FloatArray) -> FloatArray:
    # Both factors have orthonormal columns, so the k x k product is the cosine matrix; the
    # Hungarian step handles eigenvectors that swapped order between fits, |.| handles sign.
    c = np.abs(reference.T @ other)
    rows, cols = linear_sum_assignment(-c)
    aligned = np.zeros(reference.shape[1], dtype=np.float64)
    aligned[rows] = c[rows, cols]
    return aligned


def jackknife_stability(
    token_lists_by_turn: Sequence[Sequence[Sequence[str]]],
    vocab: Vocab,
    k: int,
    folds: int = 5,
    *,
    config: TopicConfig | None = None,
    seed: int = 0,
    ppmi_fn: Callable[[sp.csr_matrix], sp.csr_matrix] | None = None,
) -> FloatArray:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if k < 1 or k >= len(vocab):
        raise ValueError(f"k must be in [1, {len(vocab) - 1}]")
    cfg = config or TopicConfig()

    def fit(turns: Sequence[Sequence[Sequence[str]]]) -> FloatArray:
        flat = [toks for turn in turns for toks in turn]
        counts = cooccurrence(flat, vocab, window=cfg.window)
        # ppmi_fn lets the caller swap in differential_ppmi so the background-prior variant
        # is jackknifed with exactly the same folds as the plain fit.
        ppmi = ppmi_fn(counts) if ppmi_fn else ppmi_sparse(counts, cds=cfg.cds, shift=cfg.shift)
        return fit_topics(ppmi, vocab, k, seed=seed).w

    full = fit(token_lists_by_turn)
    # Folds interleave turns (i % folds) rather than cutting contiguous blocks: a contiguous
    # cut removes whole topics and would measure topic coverage, not eigenvector stability.
    per_fold = np.stack(
        [
            align_cosines(
                full, fit([t for i, t in enumerate(token_lists_by_turn) if i % folds != f])
            )
            for f in range(folds)
        ]
    )
    result: FloatArray = per_fold.mean(axis=0)
    return result

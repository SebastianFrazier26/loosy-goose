"""Non-spectral reference compressors sharing the project-wide `compress` contract."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from loosy_goose.budget import ProtectPolicy, select_by_score
from loosy_goose.segment import Segment

CompressFn = Callable[..., list[Segment]]


def random_drop(
    segments: list[Segment],
    keep_ratio: float,
    *,
    protect: ProtectPolicy = "all",
    recency: float = 0.0,
    seed: int = 0,
) -> list[Segment]:
    scores = np.random.default_rng(seed).uniform(size=len(segments))
    return select_by_score(segments, scores, keep_ratio, protect, recency)


def recency_only(
    segments: list[Segment],
    keep_ratio: float,
    *,
    protect: ProtectPolicy = "all",
    recency: float = 0.0,
    seed: int = 0,
) -> list[Segment]:
    scores = np.arange(len(segments), dtype=np.float64)
    return select_by_score(segments, scores, keep_ratio, protect, recency)


def tfidf(
    segments: list[Segment],
    keep_ratio: float,
    *,
    protect: ProtectPolicy = "all",
    recency: float = 0.0,
    seed: int = 0,
) -> list[Segment]:
    if not segments:
        return []
    # token_pattern widened so identifiers/paths survive; sublinear_tf so one repeated token
    # cannot dominate a segment's sum. Length-normalised (mean weight) so long segments do not
    # win simply by containing more terms.
    vec = TfidfVectorizer(token_pattern=r"[A-Za-z_][\w./-]+|\d+", sublinear_tf=True)
    try:
        matrix = vec.fit_transform([s.text for s in segments])
    except ValueError:
        # Vocabulary empty (all stop words / non-word text): nothing to rank on.
        return select_by_score(segments, np.zeros(len(segments)), keep_ratio, protect, recency)
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    lengths = np.asarray((matrix > 0).sum(axis=1)).ravel()
    scores = np.divide(sums, lengths, out=np.zeros_like(sums), where=lengths > 0)
    return select_by_score(segments, scores, keep_ratio, protect, recency)


METHODS: dict[str, CompressFn] = {
    "random_drop": random_drop,
    "recency_only": recency_only,
    "tfidf": tfidf,
}

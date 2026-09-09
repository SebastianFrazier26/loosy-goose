"""Experiment B: embedding-SVD subset selection.

The segment x dim embedding matrix is the conversation's own Karhunen-Loeve basis; the
compressed context is a CUR-style row subset that spans its retained subspace.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import scipy.linalg

from loosy_goose.budget import ProtectPolicy, select_by_score
from loosy_goose.embed import SELECT_MODEL, FloatArray, embed_texts
from loosy_goose.segment import Segment

Strategy = Literal["leverage", "ridge", "cur_residual"]

# bge-small's context window is 512 tokens; sentence-transformers silently truncates past it,
# so anything beyond roughly 2000 characters never reaches the model. Truncating up front keeps
# the cache key honest about what was actually embedded.
EMBED_MAX_CHARS = 2000


class CompressFn(Protocol):
    def __call__(
        self,
        segments: list[Segment],
        keep_ratio: float,
        *,
        protect: ProtectPolicy = "all",
        recency: float = 0.0,
        seed: int = 0,
    ) -> list[Segment]: ...


@dataclass(frozen=True)
class CompressConfig:
    model_name: str = SELECT_MODEL
    energy: float = 0.95
    strategy: Strategy = "leverage"
    ridge_lambda: float = 1.0


def _cache_key(model_name: str, texts: list[str]) -> str:
    h = hashlib.sha256(model_name.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def embed_segments(
    segments: list[Segment],
    model_name: str = SELECT_MODEL,
    cache_dir: Path | None = None,
) -> FloatArray:
    texts = [s.text[:EMBED_MAX_CHARS] for s in segments]
    if not texts:
        return np.zeros((0, 0), dtype=np.float64)
    cache_path: Path | None = None
    if cache_dir is not None:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
        cache_path = cache_dir / f"{slug}_{_cache_key(model_name, texts)}.npy"
        if cache_path.exists():
            cached: FloatArray = np.load(cache_path)
            return cached
    vectors = embed_texts(texts, model_name)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, vectors)
    return vectors


def center(x: FloatArray) -> FloatArray:
    # Text embeddings are anisotropic: they share a dominant common direction (Mu & Viswanath
    # 2018 "All-but-the-Top"; Ethayarajh 2019), so the uncentred top singular vector is a
    # "market mode" carrying no topical information and would soak up most of the energy.
    # Subtracting the mean removes it before the spectrum is read.
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("embedding matrix must be 2-D")
    centred: FloatArray = a - a.mean(axis=0, keepdims=True)
    return centred


def svd_energy(x: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    xc = center(x)
    u, s, vt = scipy.linalg.svd(xc, full_matrices=False, check_finite=False)
    power = s**2
    total = power.sum()
    energy: FloatArray = np.cumsum(power) / total if total > 0 else np.ones_like(power)
    return u, s, vt, energy


def rank_for_energy(s: FloatArray, threshold: float) -> int:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    power = np.asarray(s, dtype=np.float64) ** 2
    if power.size == 0:
        return 0
    total = power.sum()
    if total <= 0:
        return 1
    cum = np.cumsum(power) / total
    # Floating error can leave the final cumulative slightly under 1.0; searchsorted would then
    # return len(s) for threshold=1.0, which is clamped back to a valid rank.
    k = int(np.searchsorted(cum, threshold - 1e-12, side="left")) + 1
    return min(k, int(power.size))


def leverage_scores(u: FloatArray, k: int) -> FloatArray:
    if k < 1 or k > u.shape[1]:
        raise ValueError(f"k must be in [1, {u.shape[1]}]")
    scores: FloatArray = np.sum(u[:, :k] ** 2, axis=1)
    return scores


def ridge_leverage_scores(x: FloatArray, lam: float) -> FloatArray:
    if lam < 0:
        raise ValueError("lam must be non-negative")
    u, s, _, _ = svd_energy(x)
    # x_i^T (X^T X + lam I)^-1 x_i = sum_j U_ij^2 s_j^2 / (s_j^2 + lam); the shrinkage keeps the
    # small-n tail directions from being weighted as heavily as the real structure.
    # lam=0 must not count numerically-zero singular directions (SVD returns min(n, d) of them
    # regardless of rank), so the rank cut-off mirrors numpy.linalg.matrix_rank's tolerance.
    rank_tol = float(s.max()) * max(x.shape) * np.finfo(np.float64).eps if s.size else 0.0
    live = s > rank_tol
    shrink = np.where(live, s**2 / np.where(live, s**2 + lam, 1.0), 0.0)
    scores: FloatArray = np.sum((u**2) * shrink, axis=1)
    return scores


def cur_residual_order(z: FloatArray, tol: float = 1e-9) -> tuple[list[int], FloatArray]:
    """Pivoted Gram-Schmidt over rows: each pick is the row with the largest residual norm after
    projecting out everything already picked. Returns (pick order, residual norm at pick time);
    the residual sequence is non-increasing. Stops once the remaining rows are numerically in
    the span of the picks, so the returned order may be shorter than the number of rows."""
    a = np.asarray(z, dtype=np.float64)
    n = a.shape[0]
    if n == 0:
        return [], np.zeros(0, dtype=np.float64)
    residual = a.copy()
    sq = np.sum(residual**2, axis=1)
    scale = math.sqrt(float(sq.max())) if sq.max() > 0 else 1.0
    order: list[int] = []
    picked = np.zeros(n, dtype=bool)
    norms: list[float] = []
    for _ in range(min(n, a.shape[1])):
        sq = np.where(picked, -1.0, np.sum(residual**2, axis=1))
        i = int(np.argmax(sq))
        norm = math.sqrt(max(float(sq[i]), 0.0))
        if norm <= tol * scale:
            break
        q = residual[i] / norm
        residual -= np.outer(residual @ q, q)
        residual[i] = 0.0
        picked[i] = True
        order.append(i)
        norms.append(norm)
    return order, np.asarray(norms, dtype=np.float64)


def cur_residual_scores(u: FloatArray, s: FloatArray, k: int) -> FloatArray:
    if k < 1 or k > u.shape[1]:
        raise ValueError(f"k must be in [1, {u.shape[1]}]")
    n = u.shape[0]
    z = u[:, :k] * s[:k]
    order, norms = cur_residual_order(z)
    # At most k rows can be picked before the rank-k span is exhausted. Anything past that has
    # zero residual by construction, so the tail is ordered by leverage instead of left as an
    # index-ordered tie, scaled to sit strictly below every residual score.
    scores = np.zeros(n, dtype=np.float64)
    scores[order] = norms
    remaining = np.ones(n, dtype=bool)
    remaining[order] = False
    if remaining.any():
        floor = float(norms.min()) if norms.size else 1.0
        lev = leverage_scores(u, k)[remaining]
        top = float(lev.max()) if lev.size else 1.0
        scores[remaining] = 0.5 * floor * (lev / top if top > 0 else 0.0)
    return scores


def greedy_cur_select(z: FloatArray, k: int, budget_fn: Callable[[list[int]], bool]) -> list[int]:
    """Direct CUR selection: walk the residual pivot order and keep a pick while
    `budget_fn(picks_so_far + [candidate])` stays true; skip (do not stop at) candidates that do
    not fit so smaller segments deeper in the order can still be used."""
    if k < 1:
        raise ValueError("k must be positive")
    order, _ = cur_residual_order(np.asarray(z, dtype=np.float64)[:, :k])
    picks: list[int] = []
    for i in order:
        if budget_fn([*picks, i]):
            picks.append(i)
    return picks


def score_segments(x: FloatArray, config: CompressConfig) -> FloatArray:
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    if config.strategy == "ridge":
        return ridge_leverage_scores(x, config.ridge_lambda)
    u, s, _, _ = svd_energy(x)
    k = rank_for_energy(s, config.energy)
    if config.strategy == "leverage":
        return leverage_scores(u, k)
    if config.strategy == "cur_residual":
        return cur_residual_scores(u, s, k)
    raise ValueError(f"unknown strategy: {config.strategy!r}")


def compress(
    segments: list[Segment],
    keep_ratio: float,
    *,
    protect: ProtectPolicy = "all",
    recency: float = 0.0,
    seed: int = 0,
    config: CompressConfig | None = None,
    cache_dir: Path | None = None,
) -> list[Segment]:
    # `seed` is part of the shared compress contract; this method is deterministic and ignores it.
    del seed
    cfg = config or CompressConfig()
    if not segments:
        return []
    x = embed_segments(segments, cfg.model_name, cache_dir)
    scores = score_segments(x, cfg)
    return select_by_score(segments, scores, keep_ratio, protect, recency)


def make_compress(config: CompressConfig, cache_dir: Path | None = None) -> CompressFn:
    def _compress(
        segments: list[Segment],
        keep_ratio: float,
        *,
        protect: ProtectPolicy = "all",
        recency: float = 0.0,
        seed: int = 0,
    ) -> list[Segment]:
        return compress(
            segments,
            keep_ratio,
            protect=protect,
            recency=recency,
            seed=seed,
            config=config,
            cache_dir=cache_dir,
        )

    return _compress


_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_STOPWORDS = frozenset(
    """
    the and for that with this from are was were will would can could should you your not
    have has had but they them their there then than into out about which what when where
    who how all any each also just like more most some such only over under our its it's
    been being does did doing use used using one two new get got let may might must shall
    here now yes true false none null self def return import class type list dict str int
    """.split()
)


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in _STOPWORDS]


def label_directions(
    segments: list[Segment],
    vt: FloatArray,
    u: FloatArray,
    k: int,
    n_words: int = 12,
    n_segments: int = 10,
) -> list[list[str]]:
    if k > vt.shape[0] or k > u.shape[1]:
        raise ValueError("k exceeds the number of retained directions")
    docs = [Counter(_tokens(s.text)) for s in segments]
    df: Counter[str] = Counter()
    for d in docs:
        df.update(d.keys())
    n_docs = max(len(docs), 1)
    idf = {w: math.log((1 + n_docs) / (1 + c)) + 1.0 for w, c in df.items()}
    labels: list[list[str]] = []
    for j in range(k):
        top = np.argsort(-np.abs(u[:, j]))[:n_segments]
        weight: dict[str, float] = defaultdict(float)
        for i in top:
            d = docs[int(i)]
            total = sum(d.values()) or 1
            for w, c in d.items():
                weight[w] += (c / total) * idf[w]
        ranked = sorted(weight.items(), key=lambda kv: (-kv[1], kv[0]))
        labels.append([w for w, _ in ranked[:n_words]])
    return labels

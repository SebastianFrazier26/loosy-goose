"""Per-kind scoring on the supersession pool (PHASE2 queue item 8).

Shared-space leverage puts prose, code, tool_use and tool_result in one SVD, so whichever kind
owns the dominant directions wins the budget. Three ways to take kind out of that competition:
score each kind in its own space (`kind_space_scores`), normalise the shared scores within kind
(`kind_norm_scores`), or give each kind its own slice of the budget (`kind_budget_select`).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import scipy.stats

from loosy_goose.budget import ProtectPolicy, forced_mask, select_with_budget
from loosy_goose.embed import FloatArray
from loosy_goose.segment import Segment, SegmentKind
from loosy_goose.select import CompressConfig, score_segments
from loosy_goose.tokens import count_tokens

# Bump whenever any selection output changes for the same input.
KIND_SELECT_VERSION: int = 1

# Below this many segments a kind's own SVD, rank or z-score is noise, so it is scored in the
# shared space instead (and, in the budget arm, gets no slice of its own).
MIN_KIND_SEGMENTS = 5

KINDS: tuple[SegmentKind, ...] = get_args(SegmentKind)

KindNorm = Literal["rank", "z"]


def kind_groups(segments: list[Segment]) -> dict[SegmentKind, list[int]]:
    groups: dict[SegmentKind, list[int]] = defaultdict(list)
    for i, s in enumerate(segments):
        groups[s.kind].append(i)
    return dict(groups)


def _mean_scaled(scores: FloatArray) -> FloatArray:
    # Leverage over n rows sums to the retained rank k, so its mean is k/n: dividing by the mean
    # removes that size-dependent scale and leaves every group with mean 1. An all-zero group has
    # no scale to remove and stays zero.
    mean = float(scores.mean()) if scores.size else 0.0
    out: FloatArray = scores / mean if mean > 0 else np.zeros_like(scores)
    return out


def kind_space_scores(
    segments: list[Segment],
    x: FloatArray,
    config: CompressConfig,
    min_segments: int = MIN_KIND_SEGMENTS,
) -> FloatArray:
    """`score_segments` run on each kind's own rows of `x`, divided by that kind's mean score.

    A kind with fewer than `min_segments` segments keeps its shared-space score, divided by the
    whole pool's mean shared score so it lands on the same mean-1 scale as the others rather than
    on leverage's raw k/n scale."""
    n = len(segments)
    if x.shape[0] != n:
        raise ValueError("x must have one row per segment")
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    shared = _mean_scaled(np.asarray(score_segments(x, config), dtype=np.float64))
    out = shared.copy()
    for idx in kind_groups(segments).values():
        if len(idx) >= min_segments:
            out[idx] = _mean_scaled(np.asarray(score_segments(x[idx], config), dtype=np.float64))
    return out


def _rank_pct(v: FloatArray) -> FloatArray:
    # Average ranks mapped onto [0, 1]: ties share one value, so a zero-variance group is all 0.5,
    # and a single segment, having no one to rank against, is 0.5 too.
    if v.size <= 1:
        return np.full(v.size, 0.5, dtype=np.float64)
    ranks = np.asarray(scipy.stats.rankdata(v, method="average"), dtype=np.float64)
    out: FloatArray = (ranks - 1.0) / (v.size - 1)
    return out


def _zscore(v: FloatArray) -> FloatArray:
    # Population std. Zero variance (including a single segment) means no segment stands out, so
    # every z is 0 rather than a division by zero.
    if v.size == 0:
        return np.zeros(0, dtype=np.float64)
    std = float(v.std())
    out: FloatArray = (v - v.mean()) / std if std > 0 else np.zeros_like(v)
    return out


def kind_norm_scores(
    segments: list[Segment],
    scores: npt.ArrayLike,
    norm: KindNorm,
    min_segments: int = MIN_KIND_SEGMENTS,
) -> FloatArray:
    """Shared-space `scores` normalised within each kind: rank percentile in [0, 1], or z-score.

    A kind with fewer than `min_segments` segments is normalised against the whole pool instead
    of against itself, so its segments keep their shared-space standing on the same scale."""
    raw = np.asarray(scores, dtype=np.float64).reshape(-1)
    if raw.shape[0] != len(segments):
        raise ValueError("scores must have one entry per segment")
    if norm == "rank":
        fn = _rank_pct
    elif norm == "z":
        fn = _zscore
    else:
        raise ValueError(f"unknown normalisation: {norm!r}")
    out = fn(raw)
    for idx in kind_groups(segments).values():
        if len(idx) >= min_segments:
            out[idx] = fn(raw[idx])
    return out


def _kept_mask(candidates: list[Segment], kept: list[Segment]) -> list[bool]:
    # select_with_budget returns an in-order subsequence of the very objects it was given, so one
    # forward pass matched on identity recovers which candidates it kept.
    mask = [False] * len(candidates)
    j = 0
    for i, s in enumerate(candidates):
        if j < len(kept) and kept[j] is s:
            mask[i] = True
            j += 1
    if j != len(kept):
        raise AssertionError("kept is not an in-order subsequence of candidates")
    return mask


def kind_budget_shares(
    segments: list[Segment],
    budget: int,
    min_segments: int = MIN_KIND_SEGMENTS,
    exclude: list[bool] | None = None,
) -> dict[SegmentKind, int]:
    """Each kind's slice of `budget`, proportional to its share of the tokens in `segments`
    (less `exclude`d ones). Floored, so the slices never sum past `budget`; tiny kinds get no
    slice, and their proportion is left for the spill pass."""
    skip = exclude if exclude is not None else [False] * len(segments)
    tokens: dict[SegmentKind, int] = defaultdict(int)
    counts: dict[SegmentKind, int] = defaultdict(int)
    for s, x in zip(segments, skip, strict=True):
        if not x:
            tokens[s.kind] += count_tokens(s.text)
            counts[s.kind] += 1
    total = sum(tokens.values())
    if total <= 0 or budget <= 0:
        return {}
    return {
        k: math.floor(budget * tokens[k] / total)
        for k in KINDS
        if counts[k] >= min_segments and tokens[k] > 0
    }


def kind_budget_select(
    segments: list[Segment],
    scores: npt.ArrayLike,
    budget: int,
    protect: ProtectPolicy = "none",
    min_segments: int = MIN_KIND_SEGMENTS,
) -> list[Segment]:
    """Per-kind budget slices, each filled by `select_with_budget` on the shared `scores`, then
    one global `select_with_budget` pass over everything not yet kept with whatever is unspent.

    Forced (protected) segments are taken first and their tokens come off `budget` before it is
    split: a forced segment inside a kind's fill would otherwise be charged to that kind's slice
    and could push the total past the shared budget. As with `select_with_budget`, forced tokens
    alone may exceed `budget`; nothing else is then added."""
    n = len(segments)
    if n == 0:
        return []
    raw = np.asarray(scores, dtype=np.float64).reshape(-1)
    if raw.shape[0] != n:
        raise ValueError("scores must have one entry per segment")
    costs = [count_tokens(s.text) for s in segments]
    keep = forced_mask(segments, protect)
    remaining = budget - sum(c for c, k in zip(costs, keep, strict=True) if k)
    if remaining > 0:
        shares = kind_budget_shares(segments, remaining, min_segments, exclude=keep)
        groups = kind_groups(segments)
        for kind, share in shares.items():
            idx = [i for i in groups[kind] if not keep[i]]
            cands = [segments[i] for i in idx]
            picked = _kept_mask(cands, select_with_budget(cands, raw[idx], share))
            for i, p in zip(idx, picked, strict=True):
                if p:
                    keep[i] = True
                    remaining -= costs[i]
        rest = [i for i in range(n) if not keep[i]]
        if rest and remaining > 0:
            cands = [segments[i] for i in rest]
            picked = _kept_mask(cands, select_with_budget(cands, raw[rest], remaining))
            for i, p in zip(rest, picked, strict=True):
                if p:
                    keep[i] = True
    return [s for s, k in zip(segments, keep, strict=True) if k]

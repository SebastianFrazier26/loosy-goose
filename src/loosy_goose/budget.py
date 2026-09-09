from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from loosy_goose.segment import Segment
from loosy_goose.tokens import count_tokens

ProtectPolicy = Literal["all", "code_only", "none"]


def token_budget(segments: list[Segment], keep_ratio: float) -> int:
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    total = sum(count_tokens(s.text) for s in segments)
    # ceil rather than round: a budget of 0 would make keep_ratio > 0 return nothing.
    return math.ceil(total * keep_ratio)


def forced_mask(segments: list[Segment], protect: ProtectPolicy) -> list[bool]:
    if protect == "all":
        return [s.protected for s in segments]
    if protect == "code_only":
        return [s.kind == "code" for s in segments]
    if protect == "none":
        return [False] * len(segments)
    raise ValueError(f"unknown protect policy: {protect!r}")


def select_by_score(
    segments: list[Segment],
    scores: npt.ArrayLike,
    keep_ratio: float,
    protect: ProtectPolicy = "all",
    recency: float = 0.0,
) -> list[Segment]:
    n = len(segments)
    raw = np.asarray(scores, dtype=np.float64).reshape(-1)
    if raw.shape[0] != n:
        raise ValueError("scores must have one entry per segment")
    if not 0.0 <= recency <= 1.0:
        raise ValueError("recency must be in [0, 1]")
    if n == 0:
        return []

    # Blend on a min-max normalised score so `recency` means the same thing regardless of the
    # method's native score scale (tf-idf sums vs. cosines vs. uniform noise).
    span = raw.max() - raw.min()
    norm = (raw - raw.min()) / span if span > 0 else np.zeros(n)
    position = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    blended = (1.0 - recency) * norm + recency * position

    budget = token_budget(segments, keep_ratio)
    costs = [count_tokens(s.text) for s in segments]
    keep = forced_mask(segments, protect)
    used = sum(c for c, k in zip(costs, keep, strict=True) if k)

    if used < budget:
        # Greedy by score; ties broken toward earlier segments so results are deterministic.
        order = sorted((i for i in range(n) if not keep[i]), key=lambda i: (-blended[i], i))
        for i in order:
            if used + costs[i] <= budget:
                keep[i] = True
                used += costs[i]
    return [s for s, k in zip(segments, keep, strict=True) if k]

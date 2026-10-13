from __future__ import annotations

import heapq
import math
import re
from collections import Counter

import numpy as np
import numpy.typing as npt

from loosy_goose.budget import ProtectPolicy, forced_mask
from loosy_goose.segment import Segment
from loosy_goose.tokens import count_tokens

# Bump whenever any selection output changes for the same input.
NOVELTY_VERSION: int = 1

# Plain words, deliberately not `segment.extract_atoms`: atoms are the recall metric's own unit,
# and selecting on them would grade the selector with the quantity it optimised.
_WORD = re.compile(r"[^\W\d_]{3,}")


def plain_words(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def word_sets(segments: list[Segment]) -> list[frozenset[str]]:
    """Each segment's plain words, minus the ones in more than half the pool's segments: those
    are boilerplate every segment shares, and counting them would make almost nothing novel."""
    sets = [plain_words(s.text) for s in segments]
    df = Counter(w for ws in sets for w in ws)
    common = {w for w, c in df.items() if c > len(sets) / 2}
    return [ws - common for ws in sets]


def novelty_share(words: frozenset[str], seen: set[str]) -> float:
    # A segment with no words has nothing to repeat, so it keeps its full leverage.
    if not words:
        return 1.0
    return sum(1 for w in words if w not in seen) / len(words)


def novelty_select(
    segments: list[Segment],
    scores: npt.ArrayLike,
    budget: int,
    alpha: float,
    protect: ProtectPolicy = "none",
    *,
    per_token: bool = False,
) -> list[Segment]:
    """`budget.select_with_budget`'s greedy fill, with each candidate's score discounted by
    novelty_share(its words, the kept set's words) ** alpha. Ranked by raw score by default, to
    match that fill; alpha=0 reproduces it exactly. `per_token` ranks by that score divided by the
    segment's token cost instead, so a large segment no longer wins on size alone."""
    n = len(segments)
    if n == 0:
        return []
    raw = np.asarray(scores, dtype=np.float64).reshape(-1)
    if raw.shape[0] != n:
        raise ValueError("scores must have one entry per segment")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")
    # The lazy bound below needs scores to only fall as the kept set grows; a negative score
    # times a shrinking share would rise instead.
    if alpha > 0 and (raw < 0).any():
        raise ValueError("scores must be non-negative when alpha > 0")
    words = word_sets(segments)
    costs = [count_tokens(s.text) for s in segments]
    keep = forced_mask(segments, protect)
    used = sum(c for c, k in zip(costs, keep, strict=True) if k)
    seen: set[str] = set()
    for i, forced in enumerate(keep):
        if forced:
            seen |= words[i]

    # Only empty text costs 0 tokens; it fits any remaining budget and adds no words to `seen`,
    # so its rank cannot change another pick, and a divisor of 1 keeps the key finite.
    denom = [max(c, 1) for c in costs] if per_token else None

    def score(i: int) -> float:
        gain = float(raw[i]) * math.pow(novelty_share(words[i], seen), alpha)
        return gain if denom is None else gain / denom[i]

    if used < budget:
        remaining = budget - used
        # Minoux lazy greedy, exact here: novelty only shrinks as `seen` grows and a segment's
        # token cost is fixed, so a stale score is a true upper bound. The index breaks ties
        # toward earlier segments, as the fill does.
        heap = [(-score(i), i, 0) for i in range(n) if not keep[i]]
        heapq.heapify(heap)
        picks = 0
        while heap:
            _, i, stamp = heapq.heappop(heap)
            if costs[i] > remaining:
                continue
            if stamp == picks:
                keep[i] = True
                remaining -= costs[i]
                seen |= words[i]
                picks += 1
            else:
                heapq.heappush(heap, (-score(i), i, picks))
    return [s for s, k in zip(segments, keep, strict=True) if k]

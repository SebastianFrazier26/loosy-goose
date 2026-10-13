import math
from dataclasses import dataclass
from typing import Literal

KEEP_MIN = 0.1
KEEP_MAX = 0.8
LEVELS: dict[str, float] = {"light": 0.50, "medium": 0.33, "heavy": 0.25}
DEFAULT_LEVEL = "medium"

RateSource = Literal["keep", "level", "target-tokens"]


@dataclass(frozen=True)
class Rate:
    """The compression rate as the pipeline consumes it. `budget` is set only when a token target
    was honoured exactly; otherwise the pipeline derives the budget from `keep`."""

    keep: float
    budget: int | None
    source: RateSource
    level: str | None
    target_tokens: int | None
    notes: tuple[str, ...]


def check_keep(k: float) -> float:
    if not KEEP_MIN <= k <= KEEP_MAX:
        raise ValueError(f"keep must be in [{KEEP_MIN}, {KEEP_MAX}], got {k}")
    return k


def resolve_rate(
    keep: float | None, level: str | None, target_tokens: int | None, input_tokens: int
) -> Rate:
    """`input_tokens` must be the count `token_budget` sums (original segments), or a target's
    implied ratio would not match the budget the pipeline would derive from it."""
    if keep is not None:
        rounded = round(keep, 2)
        notes: tuple[str, ...] = ()
        if rounded != keep:
            notes = (f"--keep {keep} rounded to 0.01 steps; using {rounded:.2f}",)
        return Rate(rounded, None, "keep", None, None, notes)
    if target_tokens is not None:
        implied = target_tokens / input_tokens if input_tokens else math.inf
        if KEEP_MIN <= implied <= KEEP_MAX:
            return Rate(implied, target_tokens, "target-tokens", None, target_tokens, ())
        clamped = KEEP_MIN if implied < KEEP_MIN else KEEP_MAX
        warning = (
            f"warning: --target-tokens {target_tokens} implies keep {implied:.4f} of "
            f"{input_tokens} input tokens, outside [{KEEP_MIN}, {KEEP_MAX}]; clamped to "
            f"keep {clamped:.2f}"
        )
        return Rate(clamped, None, "target-tokens", None, target_tokens, (warning,))
    name = level if level is not None else DEFAULT_LEVEL
    return Rate(LEVELS[name], None, "level", name, None, ())

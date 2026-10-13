"""Phase 2 / Experiment G: leverage discounted by word novelty, on Experiment D's curves.

supersede+leverage with each candidate's leverage multiplied by (share of its plain words the
kept set does not already hold) ** alpha (`loosy_goose.novelty_select`), at alpha 0.5 / 1 / 2,
ranked by that raw gain; plus one arm ranked by gain per token, at alpha 0.5.
alpha=0 is supersede+leverage exactly, so it is not run: supersede+leverage and leverage are read
from the exp_d checkpoints, and only where those were scored against the same segmenter output.

Usage: uv run python experiments/exp_d_curves.py --grid exp_g [--only NAME_SUBSTR] [--methods A,B]
Writes experiments/output/exp_g/<label>.json and experiments/output/exp_g/summary.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import exp_d_curves as exp_d  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    BASE_VARIANT,
    EXP_D,
    NOVELTY_ALPHAS,
    NOVELTY_METHODS,
    NOVELTY_PER_TOKEN_ALPHAS,
    NOVELTY_PER_TOKEN_METHODS,
    Grid,
    MethodFn,
    RunContext,
    _base_only,
    _curves_by_method,
    _leverage_scores,
)
from experiments.exp_f_compress import (  # noqa: E402
    _cost_table,
    _delta_table,
    _distortion_table,
    _header,
    _knee_table,
    _load_references,
    _n_transcripts,
)
from loosy_goose.budget import ProtectPolicy, token_budget  # noqa: E402
from loosy_goose.novelty_select import novelty_select  # noqa: E402
from loosy_goose.segment import Segment  # noqa: E402
from loosy_goose.supersede import apply_supersession  # noqa: E402

OUT = exp_d.ROOT / "experiments" / "output" / "exp_g"
REFERENCES = ("leverage", "supersede+leverage")
SHIPPED = "supersede+leverage"
PER_TOKEN_PAIRS = [
    (pt, raw)
    for pt, a in zip(NOVELTY_PER_TOKEN_METHODS, NOVELTY_PER_TOKEN_ALPHAS, strict=True)
    for raw, b in zip(NOVELTY_METHODS, NOVELTY_ALPHAS, strict=True)
    if a == b
]
_NAME_W = 42
_PAIR_W = 76


def _run_novelty(alpha: float, *, per_token: bool = False) -> MethodFn:
    def run(
        segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
    ) -> list[Segment]:
        budget = token_budget(segs, ratio)
        pool = apply_supersession(segs)
        return novelty_select(
            pool, _leverage_scores(pool, ctx), budget, alpha, protect, per_token=per_token
        )

    return run


METHODS: dict[str, MethodFn] = {
    **{
        name: _run_novelty(alpha)
        for name, alpha in zip(NOVELTY_METHODS, NOVELTY_ALPHAS, strict=True)
    },
    **{
        name: _run_novelty(alpha, per_token=True)
        for name, alpha in zip(NOVELTY_PER_TOKEN_METHODS, NOVELTY_PER_TOKEN_ALPHAS, strict=True)
    },
}


def write_summary_with(
    results: list[dict[str, Any]], out: Path, *, reference_dir: Path, grid: Grid
) -> str:
    merged, excluded = _load_references(results, reference_dir, REFERENCES)
    base = _base_only(merged)
    curves = _curves_by_method(base)
    total = _n_transcripts(results)
    arms = list(grid.methods)
    everything = [*REFERENCES, *arms]
    lines = _header(
        results,
        excluded,
        grid,
        title="exp_g: leverage x word novelty",
        references=REFERENCES,
        name_w=_NAME_W,
    )
    lines += _distortion_table(curves, everything, total, _NAME_W)
    lines += _delta_table(
        "== each alpha minus supersede+leverage (alpha=0, shipped default) and minus leverage, "
        "matched achieved rate ==",
        ["Positive = the arm keeps more. *n/N = transcripts where both curves reach the rate."],
        [(a, ref) for ref in (SHIPPED, "leverage") for a in arms],
        curves,
        total,
        _PAIR_W,
    )
    lines += _delta_table(
        "== per-token minus raw-gain ranking, same alpha ==",
        ["Positive = ranking by gain per token keeps more than ranking by raw gain."],
        [p for p in PER_TOKEN_PAIRS if p[0] in grid.methods and p[1] in grid.methods],
        curves,
        total,
        _PAIR_W,
    )
    lines += _knee_table(curves, everything, _NAME_W)
    lines += _cost_table(base, everything, _NAME_W)
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    return text


def _write_summary(results: list[dict[str, Any]], out: Path) -> None:
    print("\n" + write_summary_with(results, out, reference_dir=EXP_D.out, grid=EXP_G))


EXP_G = Grid(
    name="exp_g",
    out=OUT,
    methods=METHODS,
    variants=(BASE_VARIANT,),
    protect_code_only=(),
    write_summary=_write_summary,
    plots=False,
)

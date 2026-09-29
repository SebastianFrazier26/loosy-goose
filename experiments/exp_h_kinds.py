"""Phase 2 / Experiment H: per-kind scoring on the supersession pool, on Experiment D's curves.

Four arms over supersede+leverage (`loosy_goose.kind_select`), each taking segment kind out of the
shared-space competition a different way:
- kind-space: leverage computed in each kind's own embedding space, divided by the kind's mean;
- kind-norm-rank / kind-norm-z: one shared SVD, scores rank-percentiled or z-scored within kind;
- kind-budget: the budget split by the kinds' pool-token shares, each slice filled on shared
  leverage, the unspent remainder spilled into one global pass.
supersede+leverage and leverage are read from the exp_d checkpoints, and only where those were
scored against the same segmenter output.

Usage: uv run python experiments/exp_d_curves.py --grid exp_h [--only NAME_SUBSTR] [--methods A,B]
Writes experiments/output/exp_h/<label>.json and experiments/output/exp_h/summary.txt.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import exp_d_curves as exp_d  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    BASE_VARIANT,
    EXP_D,
    KIND_METHODS,
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
from loosy_goose import embed  # noqa: E402
from loosy_goose.budget import ProtectPolicy, select_with_budget, token_budget  # noqa: E402
from loosy_goose.kind_select import (  # noqa: E402
    KINDS,
    KindNorm,
    kind_budget_select,
    kind_norm_scores,
    kind_space_scores,
)
from loosy_goose.segment import Segment  # noqa: E402
from loosy_goose.select import embed_segments  # noqa: E402
from loosy_goose.supersede import apply_supersession  # noqa: E402

OUT = exp_d.ROOT / "experiments" / "output" / "exp_h"
REFERENCES = ("leverage", "supersede+leverage")
SHIPPED = "supersede+leverage"
# Nominal keep ratios for the per-kind share table: grid points, so every transcript has a record
# there, nearest to the 0.5 / 0.33 / 0.25 operating points.
SHARE_RATIOS = (0.5, 0.3, 0.25)
_NAME_W = 38
_PAIR_W = 62
_KIND_W = 13


def _run_scored(score: Callable[[list[Segment], RunContext], np.ndarray]) -> MethodFn:
    def run(
        segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
    ) -> list[Segment]:
        budget = token_budget(segs, ratio)
        pool = apply_supersession(segs)
        return select_with_budget(pool, score(pool, ctx), budget, protect)

    return run


def _kind_space(pool: list[Segment], ctx: RunContext) -> np.ndarray:
    x = embed_segments(pool, embed.SELECT_MODEL, cache_dir=ctx.cache_dir)
    return kind_space_scores(pool, x, ctx.variant.compress_config("leverage"))


def _kind_norm(norm: KindNorm) -> Callable[[list[Segment], RunContext], np.ndarray]:
    def score(pool: list[Segment], ctx: RunContext) -> np.ndarray:
        return kind_norm_scores(pool, _leverage_scores(pool, ctx), norm)

    return score


def _run_kind_budget(
    segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext
) -> list[Segment]:
    budget = token_budget(segs, ratio)
    pool = apply_supersession(segs)
    return kind_budget_select(pool, _leverage_scores(pool, ctx), budget, protect)


METHODS: dict[str, MethodFn] = dict(
    zip(
        KIND_METHODS,
        (
            _run_scored(_kind_space),
            _run_scored(_kind_norm("rank")),
            _run_scored(_kind_norm("z")),
            _run_kind_budget,
        ),
        strict=True,
    )
)


def _share_row(name: str, tokens: dict[str, float], kinds: list[str]) -> str:
    total = sum(tokens.values())
    cells = [f"{tokens.get(k, 0.0) / total:.3f}" if total > 0 else "n/a" for k in kinds]
    return f"{name:<{_NAME_W}}" + "".join(c.rjust(_KIND_W) for c in cells)


def _kind_share_table(merged: list[dict[str, Any]], methods: list[str]) -> list[str]:
    """Kept tokens per kind as a share of all kept tokens, pooled over transcripts, beside the
    post-supersession pool's own shares. Records predating `kept_tokens_by_kind` (exp_d's stored
    references, until re-run) are left out and their row reads 'n/a'."""
    runs = [r for r in merged if "skipped" not in r]
    kinds = [k for k in KINDS if any(k in r.get("pool_tokens_by_kind", {}) for r in runs)]
    header = f"{'method @ nominal keep ratio':<{_NAME_W}}" + "".join(
        k.rjust(_KIND_W) for k in kinds
    )
    out = [
        "",
        "== kept-token share per kind (pooled over transcripts) ==",
        "The pool row is the post-supersession candidate pool, what kind-budget splits by.",
        header,
        "-" * len(header),
    ]
    pool: dict[str, float] = {}
    for r in runs:
        for k, v in r.get("pool_tokens_by_kind", {}).items():
            pool[k] = pool.get(k, 0.0) + float(v)
    out.append(_share_row("pool (post-supersession)", pool, kinds))
    for ratio in SHARE_RATIOS:
        for method in methods:
            kept: dict[str, float] = {}
            for r in runs:
                for rec in r["records"]:
                    if (
                        rec["method"] == method
                        and rec.get("protect") == "none"
                        and rec.get("variant_key", BASE_VARIANT.key()) == BASE_VARIANT.key()
                        and rec.get("keep_ratio") == ratio
                        and "kept_tokens_by_kind" in rec
                    ):
                        for k, v in rec["kept_tokens_by_kind"].items():
                            kept[k] = kept.get(k, 0.0) + float(v)
            out.append(_share_row(f"{method} @ {ratio:g}", kept, kinds))
    return out


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
        title="exp_h: per-kind scoring",
        references=REFERENCES,
        name_w=_NAME_W,
    )
    lines += _distortion_table(curves, everything, total, _NAME_W)
    lines += _delta_table(
        "== each arm minus supersede+leverage (shipped default) and minus leverage, "
        "matched achieved rate ==",
        ["Positive = the arm keeps more. *n/N = transcripts where both curves reach the rate."],
        [(a, ref) for ref in (SHIPPED, "leverage") for a in arms],
        curves,
        total,
        _PAIR_W,
    )
    lines += _knee_table(curves, everything, _NAME_W)
    lines += _cost_table(base, everything, _NAME_W)
    lines += _kind_share_table(base, everything)
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    return text


def _write_summary(results: list[dict[str, Any]], out: Path) -> None:
    print("\n" + write_summary_with(results, out, reference_dir=EXP_D.out, grid=EXP_H))


EXP_H = Grid(
    name="exp_h",
    out=OUT,
    methods=METHODS,
    variants=(BASE_VARIANT,),
    protect_code_only=(),
    write_summary=_write_summary,
    plots=False,
)

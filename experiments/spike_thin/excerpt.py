"""Before/after excerpts of thinned segments, and a clean timing pass.

uv run python experiments/spike_thin/excerpt.py show <model> <label> <level> [chars]
uv run python experiments/spike_thin/excerpt.py timing <model>   # scoring only, run alone
"""

from __future__ import annotations

import json
import math
import os
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run  # noqa: E402
import surprisal  # noqa: E402
import thin  # noqa: E402

from loosy_goose.rate import LEVELS  # noqa: E402


def show(key: str, label: str, level: str, chars: int = 700) -> None:
    p: dict[str, Any] = pickle.loads(run.PREP.read_bytes())[label]
    bits = surprisal.Scorer(surprisal.SPECS[key]).token_bits
    words = thin.score_words(p["original"], bits)
    pool = run.by_ids(p, p["pool"])
    res = thin.thin_to_budget(pool, words, math.ceil(p["total"] * LEVELS[level]))
    before = {s.id: s for s in pool}
    for kind in ("prose", "tool_result", "code"):
        cands = [s for s in res.segments if s.kind == kind and 150 < len(before[s.id].text) < 4000]
        if not cands:
            continue
        s = max(cands, key=lambda s: len(before[s.id].text) - len(s.text))
        print(f"\n===== {label} seg {s.id} [{kind}] {level} {key}")
        print("--- before\n" + before[s.id].text[:chars])
        print("--- after\n" + s.text[: int(chars * len(s.text) / len(before[s.id].text)) + 1])


def show_rate(key: str, label: str, q: float, chars: int = 700) -> None:
    """Thin code, tool_result and prose to share `q` of their tokens (not to a transcript budget),
    which is what the hybrid arms do before selecting; shows what a kept segment then reads like."""
    p: dict[str, Any] = pickle.loads(run.PREP.read_bytes())[label]
    bits = surprisal.Scorer(surprisal.SPECS[key]).token_bits
    words = thin.score_words(p["original"], bits)
    pool = run.by_ids(p, p["pool"])
    kinds = frozenset({"code", "tool_result", "prose"})
    bulk = sum(run.count_tokens(s.text) for s in pool if s.kind in kinds)
    total = sum(run.count_tokens(s.text) for s in pool)
    res = thin.thin_to_budget(pool, words, total - math.ceil((1 - q) * bulk), kinds=kinds)
    before = {s.id: s for s in pool}
    for kind in ("prose", "tool_result", "code"):
        cands = [s for s in res.segments if s.kind == kind and 300 < len(before[s.id].text) < 3000]
        if not cands:
            continue
        s = cands[len(cands) // 2]
        print(f"\n===== {label} seg {s.id} [{kind}] keep~{q} {key}")
        print("--- before\n" + before[s.id].text[:chars])
        print("--- after\n" + s.text[: int(chars * len(s.text) / len(before[s.id].text)) + 1])


def timing(key: str) -> None:
    import torch

    data: dict[str, Any] = pickle.loads(run.PREP.read_bytes())
    scorer = surprisal.Scorer(surprisal.SPECS[key])
    walls, toks = {}, {}
    peak = 0
    limit = int(os.environ.get("SPIKE_TIMING_N", "0")) or len(data)
    for label, p in list(data.items())[:limit]:
        doc = thin._doc(p["original"])[0]
        t0 = time.perf_counter()
        scorer.token_bits(doc)
        if scorer.dev == "mps":
            torch.mps.synchronize()
            peak = max(peak, torch.mps.driver_allocated_memory())
        walls[label] = time.perf_counter() - t0
        toks[label] = len(scorer.tok(doc)["input_ids"])
    out = {
        "model": key,
        "device": scorer.dev,
        "load_s": scorer.load_s,
        "score_wall_s": walls,
        "lm_tokens": toks,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "peak_mps_mb": peak / 1e6,
    }
    tag = "" if scorer.dev == "mps" else f"_{scorer.dev}"
    (run.OUT / f"meta_{key}{tag}.json").write_text(json.dumps(out, indent=1))
    print(
        key,
        f"{sum(walls.values()):.1f}s total, {sum(toks.values()) / sum(walls.values()):.0f} tok/s",
    )


if __name__ == "__main__":
    if sys.argv[1] == "rate":
        show_rate(sys.argv[2], sys.argv[3], float(sys.argv[4]), *(int(x) for x in sys.argv[5:]))
    elif sys.argv[1] == "show":
        show(sys.argv[2], sys.argv[3], sys.argv[4], *(int(x) for x in sys.argv[5:]))
    else:
        timing(sys.argv[2])

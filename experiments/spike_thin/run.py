"""Spike (2026-10-21): token thinning vs whole-segment selection on Experiment I's 20 public
transcripts and cut points, at loosy's light/medium/heavy budgets.

  uv run python experiments/spike_thin/run.py prep
  uv run python experiments/spike_thin/run.py model <key>   # a surprisal.SPECS key, unigram, random
  uv run python experiments/spike_thin/run.py report

Reads experiments/output/exp_i/ (contexts/*.json for the cuts, calls.jsonl for the probes) and
never writes there. Writes experiments/output/spike_thin/. No API calls.
"""

from __future__ import annotations

import json
import math
import pickle
import random
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import thin  # noqa: E402

from experiments import exp_i_downstream as exi  # noqa: E402
from loosy_goose import embed  # noqa: E402
from loosy_goose.code import quantize_segment  # noqa: E402
from loosy_goose.metrics import atom_recall_final, final_state_atoms  # noqa: E402
from loosy_goose.novelty_select import novelty_select  # noqa: E402
from loosy_goose.pipeline import NOVELTY_ALPHA, PipelineOptions, compress_transcript  # noqa: E402
from loosy_goose.rate import LEVELS  # noqa: E402
from loosy_goose.segment import Segment, extract_atoms, segment  # noqa: E402
from loosy_goose.select import CompressConfig, embed_segments, score_segments  # noqa: E402
from loosy_goose.supersede import apply_supersession  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import (  # noqa: E402
    Transcript,
    load_claude_code_jsonl,
    load_messages_json,
)

EXP_I = ROOT / "experiments" / "output" / "exp_i"
OUT = ROOT / "experiments" / "output" / "spike_thin"
PREP = OUT / "prep.pkl"

# Hybrid splits, named by the share of the final reduction left to thinning: selection runs to
# keep / f, thinning takes it the rest of the way. Plus a fixed select-to-0.7 arm.
HYBRID_F = (0.8, 0.65)
HYBRID_ABS = 0.7


def load_transcript(label: str) -> Transcript:
    if label.startswith("trace-commons_"):
        prefix = label.split("_", 1)[1]
        d = exi.DATA / "public" / "trace-commons" / "sessions" / "claude_code"
        (p,) = [p for p in d.glob("*.jsonl") if p.stem.startswith(prefix)]
        return load_claude_code_jsonl(p)
    import pyarrow.parquet as pq

    row = int(label.removeprefix("swe-gym_row"))
    files = sorted((exi.DATA / "public" / "swe-gym").rglob("*.parquet"))
    table = pq.read_table(files[0])
    col = "messages" if "messages" in table.column_names else table.column_names[0]
    payload = table.column(col)[row].as_py()
    if isinstance(payload, str):
        payload = json.loads(payload)
    return load_messages_json(payload, name=f"swe-{row}")


# --------------------------------------------------------------------------- probe survival

_STOP = set(
    "the a an of to in on for and or is are was were be by with from at as it its this that "
    "which what when where who how into via than then not no yes".split()
)


def _norm(s: str) -> str:
    s = s.lower().replace("\\n", " ").replace('\\"', '"')
    s = re.sub(r"[`'\"“”‘’*]", "", s)
    return re.sub(r"\s+", " ", s)


def answer_keys(answer: str) -> list[str]:
    """The checkable content of a reference answer: its atoms if it has any, otherwise its
    content words."""
    atoms = [a for a in extract_atoms(answer) if len(a) > 1]
    if atoms:
        return [_norm(a) for a in atoms]
    words = [w.strip(".,;:()[]") for w in _norm(answer).split()]
    keys = [w for w in words if len(w) > 2 and w not in _STOP]
    return keys or [_norm(answer).strip()]


def survives(keys: list[str], ctx_norm: str) -> bool:
    return all(k in ctx_norm for k in keys)


# --------------------------------------------------------------------------- prep


def source_kind(original: list[Segment], keys: list[str]) -> str:
    """Kind of the segment holding the answer: one holding every key if any, else the one
    holding the most keys."""
    best, best_n = "unlocated", 0
    for s in original:
        t = _norm(s.text)
        n = sum(k in t for k in keys)
        if n == len(keys):
            return s.kind
        if n > best_n:
            best, best_n = s.kind, n
    return best


def render(kept: list[Segment]) -> str:
    return exi.render_segments(kept)


def prep() -> None:
    store = exi.Store(EXP_I / "calls.jsonl")
    out: dict[str, Any] = {}
    for path in sorted((EXP_I / "contexts").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        label = d["label"]
        t = load_transcript(label)
        compressible = Transcript(f"{label}#compressible", list(t.turns[: d["cut"]["tail_from"]]))
        original = segment(compressible)
        total = sum(count_tokens(s.text) for s in original)
        assert total == d["compressible_segment_tokens"], (label, total)
        probes = exi.probes_for(store, label) or []
        ref = exi._text(store, label, "full", exi.HELPER.name, "grade_probe")
        grades = exi.parse_probe_grades(ref, len(probes)) if ref else None
        keep_idx = [i for i, g in enumerate(grades or []) if g == 1.0]
        sel: dict[float, list[int]] = {}
        keeps = set()
        for k in LEVELS.values():
            keeps.add(k)
            keeps.update(round(k / f, 4) for f in HYBRID_F)
            if HYBRID_ABS > k / HYBRID_F[-1]:
                keeps.add(HYBRID_ABS)
        t0 = time.perf_counter()
        for k in sorted(keeps):
            sel[k] = [s.id for s in exi.loosy_compressor(compressible, k)]
        band_sel = {
            k: compress_transcript(
                compressible, PipelineOptions(keep=k, band=True), cache_dir=exi.CACHE
            ).segments
            for k in LEVELS.values()
        }
        pool = apply_supersession(original)
        band_pool = {
            k: [quantize_segment(s, k) if s.kind == "code" else s for s in pool]
            for k in LEVELS.values()
        }
        vecs = embed.embed_texts([s.text for s in original], embed.SCORE_MODEL)
        full_norm = _norm(render(original))
        kept_probes = [probes[i] for i in keep_idx]
        out[label] = {
            "original": original,
            "total": total,
            "sel": sel,
            "pool": [s.id for s in pool],
            "band_sel": band_sel,
            "band_pool": band_pool,
            "probes": kept_probes,
            "probe_kind": [source_kind(original, answer_keys(pr["a"])) for pr in kept_probes],
            "probe_in_full": [survives(answer_keys(pr["a"]), full_norm) for pr in kept_probes],
            "full_render": render(original),
            "vecs": vecs,
            "final_atoms": final_state_atoms(original),
        }
        print(f"{label}: {total} tok, {len(keep_idx)} probes, sel {time.perf_counter() - t0:.1f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    PREP.write_bytes(pickle.dumps(out))


# --------------------------------------------------------------------------- evaluation

_VEC_CACHE: dict[str, np.ndarray] = {}


def coverage(orig_vecs: np.ndarray, kept: list[Segment]) -> float:
    # metrics.semantic_coverage's `coverage` (mean best cosine per original segment, scored by
    # SCORE_MODEL, never a selection or thinning model), with a text-keyed vector cache.
    missing = [s.text for s in kept if s.text not in _VEC_CACHE]
    if missing:
        for txt, v in zip(missing, embed.embed_texts(missing, embed.SCORE_MODEL), strict=True):
            _VEC_CACHE[txt] = v
    kv = np.stack([_VEC_CACHE[s.text] for s in kept])
    return float((orig_vecs @ kv.T).max(axis=1).mean())


def evaluate(p: dict[str, Any], kept: list[Segment]) -> dict[str, Any]:
    ctx = _norm(render(kept))
    surv = [
        int(survives(answer_keys(pr["a"]), ctx)) if ok else -1
        for pr, ok in zip(p["probes"], p["probe_in_full"], strict=True)
    ]
    scored = [x for x in surv if x >= 0]
    return {
        "tokens": sum(count_tokens(s.text) for s in kept),
        "atom_recall": atom_recall_final(p["original"], kept, p["final_atoms"]),
        "coverage": coverage(p["vecs"], kept),
        "probe_n": len(scored),
        "probe_survival": float(np.mean(scored)) if scored else float("nan"),
        "probe_surv": surv,
    }


def by_ids(p: dict[str, Any], ids: list[int]) -> list[Segment]:
    want = set(ids)
    return [s for s in p["original"] if s.id in want]


def random_bits(doc: str) -> tuple[np.ndarray, np.ndarray]:
    words = list(re.finditer(r"\S+", doc))
    rng = random.Random(len(doc))
    return np.array([rng.random() for _ in words]), np.array([m.end() for m in words])


THIN_THEN_SELECT_Q = (0.7, 0.5)
KIND_THIN_Q = (0.5, 0.25)
EXTRA_ARMS = frozenset({"code>sel", "code+tr>sel"})
# Third pass (2026-10-22): thinned code dumps are force-kept rather than left to the selector,
# which otherwise still drops most of them; the rest of the budget is selected as usual.
KEEP_ARMS = frozenset({"code-keep>sel"})
SEL_CACHE = OUT / "cache"


def select_from(segs: list[Segment], budget: int, protect: str = "none") -> list[Segment]:
    """The shipped selector (leverage x per-token novelty@0.5) run on already-thinned segments."""
    config = CompressConfig(strategy="leverage")
    x = embed_segments(segs, config.model_name, SEL_CACHE)
    scores = score_segments(x, config)
    return novelty_select(segs, scores, budget, NOVELTY_ALPHA, protect, per_token=True)  # type: ignore[arg-type]


def arm_outputs(
    p: dict[str, Any],
    pool: list[Segment],
    words: dict[int, thin.SegWords],
    token_bits: thin.TokenBits,
    k: float,
    budget: int,
    only: set[str] | None,
) -> Iterator[tuple[str, float, list[Segment]]]:
    """(arm, selection keep, kept segments) per arm, lazily so `arm_s` times each one."""

    def want(arm: str) -> bool:
        return arm in only if only else arm not in EXTRA_ARMS | KEEP_ARMS

    if want("thin-nocode"):
        yield (
            "thin-nocode",
            1.0,
            thin.thin_to_budget(pool, words, budget, kinds=thin.THIN_KINDS).segments,
        )
    if want("thin"):
        yield "thin", 1.0, thin.thin_to_budget(pool, words, budget).segments
    if want("sel>thin"):
        for f in HYBRID_F:
            ids = p["sel"][round(k / f, 4)]
            yield (
                "sel>thin",
                round(k / f, 2),
                thin.thin_to_budget(by_ids(p, ids), words, budget).segments,
            )
    if want("thin>sel"):
        pool_tokens = sum(count_tokens(s.text) for s in pool)
        for q in THIN_THEN_SELECT_Q:
            pre = thin.thin_to_budget(pool, words, math.ceil(pool_tokens * q)).segments
            yield "thin>sel", q, select_from(pre, budget)
    if want("band>thin"):
        banded = p["band_pool"][k]
        bw = thin.score_words(banded, token_bits)
        yield "band>thin", 1.0, thin.thin_to_budget(banded, bw, budget).segments
    # Added after the first pass (2026-10-22): prose thinning cost prose-sourced probes, and the
    # diagnosed loss sits in code dumps, so these thin only the bulky kinds, then select.
    for arm, kinds in (
        ("code>sel", frozenset({"code"})),
        ("code+tr>sel", frozenset({"code", "tool_result"})),
    ):
        if not want(arm):
            continue
        bulk = sum(count_tokens(s.text) for s in pool if s.kind in kinds)
        pool_tokens = sum(count_tokens(s.text) for s in pool)
        for q in KIND_THIN_Q:
            pre_budget = pool_tokens - math.ceil((1 - q) * bulk)
            pre = thin.thin_to_budget(pool, words, pre_budget, kinds=kinds).segments
            yield arm, q, select_from(pre, budget)
    if want("code-keep>sel"):
        code = frozenset({"code"})
        bulk = sum(count_tokens(s.text) for s in pool if s.kind == "code")
        pool_tokens = sum(count_tokens(s.text) for s in pool)
        for q in KIND_THIN_Q:
            pre_budget = pool_tokens - math.ceil((1 - q) * bulk)
            pre = thin.thin_to_budget(pool, words, pre_budget, kinds=code).segments
            yield "code-keep>sel", q, select_from(pre, budget, "code_only")


def token_bits_for(key: str) -> thin.TokenBits:
    import surprisal

    if key == "unigram":
        return surprisal.unigram_token_bits
    if key == "random":
        return random_bits
    return surprisal.Scorer(surprisal.SPECS[key]).token_bits


def run_model(key: str, only: set[str] | None = None, suffix: str = "") -> None:
    """Appends one line per (transcript, level, arm) to results_<key>.jsonl after each transcript
    finishes; a rerun skips transcripts already there."""
    data: dict[str, Any] = pickle.loads(PREP.read_bytes())
    path = OUT / f"results_{key}{suffix}.jsonl"
    done: set[str] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                done.add(json.loads(line)["label"])
            except (json.JSONDecodeError, KeyError):
                continue
    todo = [label for label in data if label not in done]
    if not todo:
        return
    token_bits = token_bits_for(key)
    for label in todo:
        p = data[label]
        t0 = time.perf_counter()
        words = thin.score_words(p["original"], token_bits)
        score_s = time.perf_counter() - t0
        pool = by_ids(p, p["pool"])
        rows: list[dict[str, Any]] = []
        for level, k in LEVELS.items():
            budget = math.ceil(p["total"] * k)
            tt = time.perf_counter()
            for arm, sel_keep, kept in arm_outputs(p, pool, words, token_bits, k, budget, only):
                row = {
                    "label": label,
                    "model": key,
                    "level": level,
                    "arm": arm,
                    "sel_keep": sel_keep,
                    "budget": budget,
                    "score_s": score_s,
                    "arm_s": time.perf_counter() - tt,
                }
                row.update(evaluate(p, kept))
                rows.append(row)
                tt = time.perf_counter()
        with path.open("a") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{time.strftime('%H:%M:%S')} {key} {label}: score {score_s:.1f}s", flush=True)


def run_baseline() -> None:
    data: dict[str, Any] = pickle.loads(PREP.read_bytes())
    rows = []
    for label, p in data.items():
        rows.append(
            {
                "label": label,
                "model": "-",
                "level": "full",
                "arm": "full",
                "sel_keep": 1.0,
                "budget": p["total"],
                **evaluate(p, p["original"]),
            }
        )
        for level, k in LEVELS.items():
            budget = math.ceil(p["total"] * k)
            for arm, kept in (
                ("select", by_ids(p, p["sel"][k])),
                ("band+select", p["band_sel"][k]),
            ):
                row = {
                    "label": label,
                    "model": "-",
                    "level": level,
                    "arm": arm,
                    "sel_keep": k,
                    "budget": budget,
                }
                row.update(evaluate(p, kept))
                rows.append(row)
    (OUT / "results_select.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
        run_baseline()
    elif cmd == "model":
        run_model(sys.argv[2], set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None)
    elif cmd == "baseline":
        run_baseline()
    else:
        raise SystemExit(f"unknown command {cmd}")

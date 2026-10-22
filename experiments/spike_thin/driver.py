"""Resumable overnight driver for the whole spike (2026-10-21). Every stage skips work already on
disk, so relaunching after a crash continues where it stopped:

  nohup caffeinate -i uv run python experiments/spike_thin/driver.py \
      >> experiments/output/spike_thin/run.log 2>&1 &
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import excerpt  # noqa: E402
import report  # noqa: E402
import run  # noqa: E402

MODELS = (
    "unigram",
    "random",
    "pythia-31m",
    "pythia-70m",
    "smol_llama-81m",
    "smollm2-135m",
    "qwen3-0.6b",
)
EXTRA_MODELS = (
    "unigram",
    "pythia-31m",
    "pythia-70m",
    "smol_llama-81m",
    "smollm2-135m",
    "qwen3-0.6b",
)
# Only a quality ceiling, so it runs after the report already exists.
CEILING = "qwen3-1.7b"
EXCERPTS = (
    ("pythia-70m", "trace-commons_89cf9e13", "medium"),
    ("pythia-70m", "swe-gym_row145", "medium"),
    ("qwen3-0.6b", "trace-commons_89cf9e13", "medium"),
)


def stage(name: str, fn: object, *args: object) -> None:
    print(f"{time.strftime('%H:%M:%S')} >> {name}", flush=True)
    try:
        fn(*args)  # type: ignore[operator]
    except Exception:
        print(f"{time.strftime('%H:%M:%S')} !! {name} failed", flush=True)
        traceback.print_exc()


def lm_stages(key: str) -> None:
    stage(f"run {key}", run.run_model, key)
    if key not in ("unigram", "random") and not (run.OUT / f"meta_{key}.json").exists():
        stage(f"timing {key}", excerpt.timing, key)


def excerpts() -> None:
    path = run.OUT / "excerpts.txt"
    if path.exists():
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for key, label, level in EXCERPTS:
            excerpt.show(key, label, level, 900)
    path.write_text(buf.getvalue())


def main() -> None:
    if not run.PREP.exists():
        stage("prep", run.prep)
    if not (run.OUT / "results_select.jsonl").exists():
        stage("baseline", run.run_baseline)
    for key in MODELS:
        lm_stages(key)
    stage("excerpts", excerpts)
    stage("report", report.main)
    for key in EXTRA_MODELS:
        stage(f"extra arms {key}", run.run_model, key, set(run.EXTRA_ARMS), "__x")
    stage("report (with extra arms)", report.main)
    lm_stages(CEILING)
    stage(f"extra arms {CEILING}", run.run_model, CEILING, set(run.EXTRA_ARMS), "__x")
    stage("report (with ceiling)", report.main)
    print(f"{time.strftime('%H:%M:%S')} ALL DONE", flush=True)


if __name__ == "__main__":
    main()

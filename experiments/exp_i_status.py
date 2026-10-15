"""Progress view for a running Experiment I, for a second terminal beside the run.

Reads manifest.json (the plan the runner wrote), calls.jsonl and failures.jsonl; never the
transcripts, never a CLI. Read-only: it writes nothing.

Usage: uv run python experiments/exp_i_status.py [--out experiments/output/exp_i]
       [--watch [--interval 30]]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "experiments" / "output" / "exp_i"
TASK_ORDER = ("probe_gen", "summary", "probe", "next", "grade_probe", "grade_next")
STALL_S = 45 * 60


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _model_of(key: str) -> str:
    return key.split("|")[3]


def render(out: Path, now: float | None = None) -> str:
    now = time.time() if now is None else now
    manifest_path = out / "manifest.json"
    if not manifest_path.exists():
        return f"no manifest at {manifest_path}: the run has not reached its first call yet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    planned: dict[str, list[str]] = manifest["planned"]
    done_recs = {r["key"]: r for r in _jsonl(out / "calls.jsonl")}
    failed_keys = {r["key"] for r in _jsonl(out / "failures.jsonl")} - set(done_recs)

    models = sorted({_model_of(k) for keys in planned.values() for k in keys})
    started = time.strftime("%Y-%m-%d %H:%M", time.localtime(manifest["started"]))
    lines = [
        f"Experiment I status — {len(manifest['transcripts'])} transcripts, "
        f"version {manifest['version']}, started {started}",
        "",
        f"{'task':<13}" + "".join(f"{m:>14}" for m in models) + f"{'total':>14}",
    ]
    total_planned = total_done = 0
    for task in TASK_ORDER:
        keys = planned.get(task, [])
        if not keys:
            continue
        per_model: Counter[str] = Counter()
        per_model_planned: Counter[str] = Counter()
        for k in keys:
            per_model_planned[_model_of(k)] += 1
            if k in done_recs:
                per_model[_model_of(k)] += 1
        cells = "".join(
            f"{f'{per_model[m]}/{per_model_planned[m]}' if per_model_planned[m] else '':>14}"
            for m in models
        )
        n_done = sum(per_model.values())
        lines.append(f"{task:<13}{cells}{f'{n_done}/{len(keys)}':>14}")
        total_planned += len(keys)
        total_done += n_done
    lines.append(f"{'all':<13}{'':>{14 * len(models)}}{f'{total_done}/{total_planned}':>14}")
    lines.append("")

    stamps = sorted(float(r["ts"]) for r in done_recs.values() if "ts" in r)
    recent = [t for t in stamps if t >= now - 3600]
    in_tok = sum(int(r.get("input_tokens", 0)) for r in done_recs.values())
    out_tok = sum(int(r.get("output_tokens", 0)) for r in done_recs.values())
    lines.append(f"tokens so far: {in_tok:,} in / {out_tok:,} out")
    if stamps:
        lines.append(f"last call finished {(now - stamps[-1]) / 60:.1f} min ago")
    remaining = total_planned - total_done
    if recent and remaining:
        # Stages run in order and grading calls are far cheaper than answer calls, so this
        # overestimates once grading starts; it is a rough guide, not a promise.
        lines.append(
            f"rate {len(recent)}/h over the last hour; ETA ~{remaining / len(recent):.1f} h"
        )
    if failed_keys:
        lines.append(f"FAILED and not yet retried: {len(failed_keys)} (rerun the runner to retry)")
    pid = int(manifest.get("pid", 0))
    if remaining and pid and not _alive(pid):
        lines.append(f"NOT RUNNING: pid {pid} is gone with {remaining} calls left")
    elif remaining and stamps and now - stamps[-1] > STALL_S:
        lines.append(
            f"STALLED? no call finished for {(now - stamps[-1]) / 60:.0f} min "
            "(a usage-limit backoff sleeps up to 30 min)"
        )
    elif not remaining:
        lines.append("COMPLETE")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()
    while True:
        text = render(args.out)
        if args.watch:
            print("\033[2J\033[H", end="")
        print(text, flush=True)
        if not args.watch:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())

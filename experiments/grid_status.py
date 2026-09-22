"""Live completion view of an Experiment D-style grid, for a second terminal beside the run.

Reads the grid's checkpoints (and optionally the runner's log), never the transcripts, so it is
cheap enough to redraw every few seconds. Read-only: it writes nothing.

Usage: uv run python experiments/grid_status.py [--grid exp_f] [--methods A,B] [--variants base]
       [--log experiments/output/run.log] [--watch [--interval 10]]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import exp_d_curves  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    FLUSH_EVERY,
    Grid,
    RecordIdentity,
    _record_identity,
    plan_for_transcript,
)

WIDTH = 120
MIN_STALL_SECONDS = 30 * 60
STALL_FACTOR = 3

# A pytest run of tests/test_exp_d_curves.py also contains "exp_d_curves.py"; the leading
# separator keeps it from reading as a live runner.
_RUNNER_RE = re.compile(r"(?:^|[\s/\\\"'])exp_d_curves\.py")
_GRID_FLAG_RE = re.compile(r"--grid[= ]+(?P<name>[\w.-]+)")
_LOG_RE = re.compile(r"^\d\d:\d\d:\d\d \[(?P<label>[^\]]+)\] (?P<msg>.*)$")
_JOB_RE = re.compile(r"^job (?P<n>\d+)/(?P<m>\d+) ")
_TODO_RE = re.compile(r"^\d+ records reused, (?P<m>\d+) to run")


@dataclass(frozen=True)
class Checkpoint:
    label: str
    total_tokens: int | None
    records: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    skipped: bool = False


@dataclass(frozen=True)
class Cell:
    planned: int
    done: int
    failed: int

    def text(self) -> str:
        if self.planned == 0:
            return "-"
        body = "ok" if self.done >= self.planned else ("." if self.done == 0 else str(self.done))
        return body + ("F" if self.failed else "")


@dataclass(frozen=True)
class Eta:
    seconds: float | None
    guessed_methods: tuple[str, ...]
    unknown_labels: tuple[str, ...]


@dataclass(frozen=True)
class LogState:
    order: tuple[str, ...]
    current: str | None
    job_n: int | None
    job_m: int | None
    finished: bool


@dataclass
class Status:
    grid: str
    labels: list[str]
    methods: list[str]
    cells: dict[tuple[str, str], Cell]
    eta: Eta
    idle_seconds: float | None
    runner_alive: bool | None
    stall_after: float
    log: LogState | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def planned(self) -> int:
        return sum(c.planned for c in self.cells.values())

    @property
    def done(self) -> int:
        return sum(min(c.done, c.planned) for c in self.cells.values())

    @property
    def failed(self) -> int:
        return sum(c.failed for c in self.cells.values())


def planned_by_method(
    grid: Grid,
    methods: Collection[str] | None,
    variants: Collection[str] | None,
    total_tokens: int | None = None,
) -> dict[str, set[RecordIdentity]]:
    """The plan for one transcript of `total_tokens`; None (size unknown) is the unfiltered plan,
    which is also the union over every transcript size."""
    out: dict[str, set[RecordIdentity]] = {}
    for job in plan_for_transcript(total_tokens, methods, variants, grid=grid):
        out.setdefault(job.method, set()).add(job.identity())
    return out


def _failure_key(f: Mapping[str, Any]) -> tuple[str, str, float, str]:
    return (str(f["method"]), str(f["protect"]), float(f["keep_ratio"]), str(f["variant_key"]))


def collect(
    labels: Sequence[str],
    planned: Mapping[str, set[RecordIdentity]],
    checkpoints: Mapping[str, Checkpoint],
    by_label: Mapping[str, Mapping[str, set[RecordIdentity]]] | None = None,
) -> dict[tuple[str, str], Cell]:
    """Done counts only records whose full identity (stamps included) is still planned: a stale
    record is one the runner will redo, so counting it would overstate progress.

    `by_label` holds a transcript's own plan where the grid's job_filter makes it differ from
    `planned`, whose methods still set the columns."""
    cells: dict[tuple[str, str], Cell] = {}
    for label in labels:
        cp = checkpoints.get(label)
        own = planned if by_label is None else by_label.get(label, planned)
        for method in planned:
            ids = own.get(method, set())
            if cp is not None and cp.skipped:
                cells[(label, method)] = Cell(0, 0, 0)
                continue
            have = {_record_identity(r) for r in cp.records} if cp else set()
            done_ids = have & ids
            # Failures carry no stamps; one counts only while no record answers its job.
            done_keys = {(i.method, i.protect, i.ratio, i.variant_key) for i in done_ids}
            plan_keys = {(i.method, i.protect, i.ratio, i.variant_key) for i in ids}
            open_failures = {
                k
                for k in (_failure_key(f) for f in (cp.failures if cp else []))
                if k in plan_keys and k not in done_keys
            }
            cells[(label, method)] = Cell(len(ids), len(done_ids), len(open_failures))
    return cells


def method_rates(
    planned: Mapping[str, set[RecordIdentity]], checkpoints: Mapping[str, Checkpoint]
) -> dict[str, float]:
    """Seconds per transcript token, per method: sum(elapsed) / sum(total_tokens of the record's
    transcript) over done records."""
    elapsed: dict[str, float] = {}
    tokens: dict[str, float] = {}
    for cp in checkpoints.values():
        if not cp.total_tokens:
            continue
        for rec in cp.records:
            ident = _record_identity(rec)
            if ident not in planned.get(ident.method, set()):
                continue
            elapsed[ident.method] = elapsed.get(ident.method, 0.0) + float(rec["elapsed_seconds"])
            tokens[ident.method] = tokens.get(ident.method, 0.0) + cp.total_tokens
    return {m: elapsed[m] / tokens[m] for m in elapsed if tokens[m] > 0}


def estimate_eta(
    cells: Mapping[tuple[str, str], Cell],
    rates: Mapping[str, float],
    tokens: Mapping[str, int | None],
) -> Eta:
    remaining = [
        (lab, m, c.planned - c.done) for (lab, m), c in cells.items() if c.done < c.planned
    ]
    fallback = statistics.fmean(rates.values()) if rates else None
    guessed = sorted({m for _, m, _ in remaining if m not in rates})
    unknown = sorted({lab for lab, _, _ in remaining if not tokens.get(lab)})
    if remaining and fallback is None:
        return Eta(None, tuple(guessed), tuple(unknown))
    total = 0.0
    for lab, m, left in remaining:
        n_tok = tokens.get(lab)
        if not n_tok:
            continue
        total += left * rates.get(m, fallback or 0.0) * n_tok
    return Eta(total, tuple(guessed), tuple(unknown))


def per_job_seconds(
    label: str,
    methods: Iterable[str],
    rates: Mapping[str, float],
    tokens: Mapping[str, int | None],
) -> float | None:
    n_tok = tokens.get(label)
    known = [rates[m] for m in methods if m in rates]
    if not n_tok or not known:
        return None
    return statistics.fmean(known) * n_tok


def stall_threshold(typical_gap: float | None) -> float:
    return max(MIN_STALL_SECONDS, STALL_FACTOR * typical_gap) if typical_gap else MIN_STALL_SECONDS


def typical_gap(
    status_labels: Sequence[str],
    cells: Mapping[tuple[str, str], Cell],
    methods: Sequence[str],
    rates: Mapping[str, float],
    tokens: Mapping[str, int | None],
    current: str | None,
) -> float | None:
    """Expected seconds between checkpoint writes (FLUSH_EVERY jobs). Without a log the in-flight
    transcript is unknown, so the slowest unfinished one is assumed: a late alarm beats a false
    one on the day's biggest transcript."""
    if current is not None:
        candidates = [current]
    else:
        candidates = [
            lab
            for lab in status_labels
            if any(cells[(lab, m)].done < cells[(lab, m)].planned for m in methods)
        ]
    per_job = [p for lab in candidates if (p := per_job_seconds(lab, methods, rates, tokens))]
    return FLUSH_EVERY * max(per_job) if per_job else None


def runner_lines(
    alive: bool | None, remaining: int, idle: float | None, stall_after: float
) -> list[str]:
    if alive is None:
        return ["RUNNER: unknown (process detection failed)"]
    if not alive:
        if remaining > 0:
            return [
                f"!!! NOT RUNNING - {remaining} jobs left. "
                "Resume with the same command using >> instead of >."
            ]
        return ["RUNNER: not running (nothing left)"]
    lines = ["RUNNER: running"]
    if idle is not None and idle > stall_after:
        lines.append(f"?? no progress for {idle / 60:.0f} min - slow job or stuck")
    return lines


def parse_log(text: str) -> LogState:
    order: list[str] = []
    current: str | None = None
    job_n: int | None = None
    job_m: int | None = None
    finished = False
    for raw in re.split(r"[\r\n]+", text):
        m = _LOG_RE.match(raw.strip())
        if m is None:
            continue
        label, msg = m["label"], m["msg"]
        if label not in order:
            order.append(label)
        if label != current:
            current, job_n, job_m, finished = label, None, None, False
        if t := _TODO_RE.match(msg):
            job_n, job_m = 0, int(t["m"])
        elif j := _JOB_RE.match(msg):
            job_n, job_m = int(j["n"]), int(j["m"])
        elif msg.startswith("done in "):
            finished = True
    return LogState(tuple(order), current, job_n, job_m, finished)


def is_runner(cmdlines: Iterable[str], grid: str) -> bool:
    """A runner of another grid does not count: its checkpoints are not the ones being watched."""
    for c in cmdlines:
        if not _RUNNER_RE.search(c) or "grid_status" in c:
            continue
        flag = _GRID_FLAG_RE.search(c)
        if (flag["name"] if flag else exp_d_curves.EXP_D.name) == grid:
            return True
    return False


def detect_runner(grid: str) -> bool | None:
    if os.name == "nt":
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
            "| Select-Object -ExpandProperty CommandLine",
        ]
    else:
        cmd = ["ps", "-eo", "args"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return is_runner(out.stdout.splitlines(), grid)


def _abbrev(method: str) -> str:
    *prefix, last = method.split("+")
    pieces = re.split(r"[-_]", last)
    tail = "".join(p[:1] for p in pieces) if len(pieces) > 1 else last[:4]
    return "".join(p[:1] for p in prefix) + ("+" if prefix else "") + tail


def _letters(n: int) -> list[str]:
    return [chr(ord("A") + i) if i < 26 else f"M{i}" for i in range(n)]


def _duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h{minutes % 60:02d}m"


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=width, subsequent_indent="    ", break_on_hyphens=False) or [
        ""
    ]


def render(status: Status, width: int = WIDTH) -> str:
    lines: list[str] = []
    planned, done = status.planned, status.done
    pct = 100.0 * done / planned if planned else 100.0
    eta = "n/a" if status.eta.seconds is None else _duration(status.eta.seconds)
    idle = "n/a" if status.idle_seconds is None else f"{_duration(status.idle_seconds)} ago"
    lines.append(
        f"GRID {status.grid}  done {done}/{planned} jobs ({pct:.1f}%)  failures {status.failed}"
        f"  ETA (rough) {eta}  last write {idle}"
    )
    log = status.log
    if log is not None and log.current is not None:
        if log.finished:
            where = f"last finished {log.current} (between transcripts)"
        elif log.job_m is not None:
            where = f"{log.current} job {log.job_n}/{log.job_m} (from log)"
        else:
            where = f"{log.current} (starting)"
        lines.append(f"current: {where}")
    lines += runner_lines(
        status.runner_alive, planned - done, status.idle_seconds, status.stall_after
    )
    if status.eta.guessed_methods:
        lines += _wrap(
            "ETA guesses (no done records, mean rate used): "
            + ", ".join(status.eta.guessed_methods),
            width,
        )
    if status.eta.unknown_labels:
        lines += _wrap(
            "ETA excludes (token count unknown): " + ", ".join(status.eta.unknown_labels), width
        )
    for note in status.notes:
        lines += _wrap(note, width)

    label_w = max([len(lab) for lab in status.labels] + [len("of"), 5])

    def sums(cells: Iterable[Cell]) -> tuple[int, int]:
        cs = list(cells)
        return sum(min(c.done, c.planned) for c in cs), sum(c.planned for c in cs)

    rows = [sums(status.cells[(lab, m)] for m in status.methods) for lab in status.labels]
    cols = [sums(status.cells[(lab, m)] for lab in status.labels) for m in status.methods]
    row_tot = [f"{d}/{p}" for d, p in rows]
    col_done = [str(d) for d, _ in cols]
    col_plan = [str(p) for _, p in cols]
    tot_w = max([len(t) for t in row_tot] + [len(f"{done}/{planned}")]) + 1
    footer_w = max([len(s) for s in col_done + col_plan] + [3])
    heads = [_abbrev(m) for m in status.methods]
    cell_w = max([len(h) for h in heads] + [footer_w]) + 1
    too_wide = label_w + cell_w * len(heads) + tot_w > width
    if len(set(heads)) < len(heads) or too_wide:
        heads = _letters(len(status.methods))
        cell_w = max([len(h) for h in heads] + [footer_w]) + 1
    lines += _wrap(
        "legend: " + "  ".join(f"{h}={m}" for h, m in zip(heads, status.methods, strict=True)),
        width,
    )
    lines.append("")
    lines.append("".ljust(label_w) + "".join(h.rjust(cell_w) for h in heads) + "done".rjust(tot_w))
    for lab, tot in zip(status.labels, row_tot, strict=True):
        row = "".join(status.cells[(lab, m)].text().rjust(cell_w) for m in status.methods)
        lines.append(lab.ljust(label_w) + row + tot.rjust(tot_w))
    lines.append(
        "done".ljust(label_w)
        + "".join(s.rjust(cell_w) for s in col_done)
        + f"{done}/{planned}".rjust(tot_w)
    )
    lines.append("of".ljust(label_w) + "".join(s.rjust(cell_w) for s in col_plan))
    return "\n".join(lines)


def load_checkpoints(folder: Path) -> dict[str, Checkpoint]:
    out: dict[str, Checkpoint] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict) or "records" not in data:
            continue
        label = str(data.get("label", path.stem))
        tok = data.get("total_tokens")
        out[label] = Checkpoint(
            label=label,
            total_tokens=int(tok) if isinstance(tok, int | float) else None,
            records=list(data.get("records", [])),
            failures=list(data.get("failures", [])),
            skipped="skipped" in data,
        )
    return out


def newest_write(paths: Iterable[Path]) -> float | None:
    times: list[float] = []
    for p in paths:
        try:
            times.append(p.stat().st_mtime)
        except OSError:
            continue
    return max(times) if times else None


def build_status(
    grid: Grid,
    reference: Path,
    methods: Collection[str] | None,
    variants: Collection[str] | None,
    log_path: Path | None,
    runner_alive: bool | None,
    now: float,
) -> Status:
    own = load_checkpoints(grid.out)
    ref = load_checkpoints(reference) if reference.resolve() != grid.out.resolve() else own
    tokens: dict[str, int | None] = {lab: cp.total_tokens for lab, cp in ref.items()}
    tokens.update({lab: cp.total_tokens for lab, cp in own.items() if cp.total_tokens})
    log: LogState | None = None
    notes: list[str] = []
    stamps = [*grid.out.glob("*.json"), *grid.out.glob("*.json.tmp")]
    if log_path is not None:
        try:
            log = parse_log(log_path.read_text(encoding="utf-8", errors="replace"))
            stamps.append(log_path)
        except OSError as exc:
            notes.append(f"log unreadable: {type(exc).__name__}")
    ordered = [lab for lab in (log.order if log else ()) if lab in tokens or lab in own]
    labels = ordered + sorted((set(tokens) | set(own)) - set(ordered))
    planned = planned_by_method(grid, methods, variants)
    method_list = [m for m in grid.methods if m in planned]
    by_label = {lab: planned_by_method(grid, methods, variants, tokens.get(lab)) for lab in labels}
    cells = collect(labels, planned, own, by_label)
    rates = method_rates(planned, own)
    newest = newest_write(stamps)
    current = log.current if log and not log.finished else None
    gap = typical_gap(labels, cells, method_list, rates, tokens, current)
    return Status(
        grid=grid.name,
        labels=labels,
        methods=method_list,
        cells=cells,
        eta=estimate_eta(cells, rates, tokens),
        idle_seconds=None if newest is None else max(0.0, now - newest),
        runner_alive=runner_alive,
        stall_after=stall_threshold(gap),
        log=log,
        notes=notes,
    )


def _clear() -> None:
    if os.name == "nt":
        os.system("cls")
    else:
        print("\x1b[2J\x1b[H", end="")


def main(
    argv: Sequence[str] | None = None,
    registry: Mapping[str, Grid] | None = None,
    reference: Path | None = None,
    probe: Callable[[str], bool | None] = detect_runner,
) -> int:
    reg = exp_d_curves.grids() if registry is None else registry
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=sorted(reg), default=exp_d_curves.EXP_D.name)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args(argv)
    grid = reg[args.grid]
    methods = exp_d_curves._parse_names(args.methods, grid.methods, "method")
    variants = exp_d_curves._parse_names(args.variants, [v.key() for v in grid.variants], "variant")
    ref = exp_d_curves.EXP_D.out if reference is None else reference
    try:
        while True:
            status = build_status(
                grid, ref, methods, variants, args.log, probe(grid.name), time.time()
            )
            text = render(status)
            if args.watch:
                _clear()
                print(time.strftime("%H:%M:%S"), f"(every {args.interval:g}s, Ctrl+C to stop)")
            print(text, flush=True)
            if not args.watch:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

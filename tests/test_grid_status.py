import contextlib
import dataclasses
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import grid_status as gs  # noqa: E402
from experiments.exp_d_curves import BASE_VARIANT, Grid  # noqa: E402
from experiments.exp_f_compress import EXP_F  # noqa: E402

RATIOS = (0.1, 0.5)


def _grid(out: Path, methods: tuple[str, ...] = ("a", "b")) -> Grid:
    return Grid(
        name="fake",
        out=out,
        methods={m: lambda *a: [] for m in methods},
        variants=(BASE_VARIANT,),
        protect_code_only=(),
        write_summary=lambda results, out: None,
        plots=False,
        ratios=RATIOS,
    )


def _rec(method: str, ratio: float, elapsed: float) -> dict[str, Any]:
    return {
        "method": method,
        "protect": "none",
        "keep_ratio": ratio,
        "variant_key": "base",
        "elapsed_seconds": elapsed,
    }


def _fail(method: str, ratio: float) -> dict[str, Any]:
    return {"method": method, "protect": "none", "keep_ratio": ratio, "variant_key": "base"}


def _checkpoints() -> dict[str, gs.Checkpoint]:
    return {
        "t1": gs.Checkpoint(
            "t1",
            1000,
            [_rec("a", 0.1, 10), _rec("a", 0.5, 10), _rec("b", 0.1, 30), _rec("b", 0.5, 30)],
            # Answered by a record since, so no longer open.
            [_fail("a", 0.1)],
        ),
        "t2": gs.Checkpoint("t2", 500, [_rec("a", 0.1, 5)], [_fail("a", 0.5)]),
        "t3": gs.Checkpoint("t3", 2000, [], []),
    }


def test_collect_counts_full_partial_and_unstarted(tmp_path: Path) -> None:
    planned = gs.planned_by_method(_grid(tmp_path), None, None)
    cells = gs.collect(["t1", "t2", "t3"], planned, _checkpoints())
    assert cells[("t1", "a")] == gs.Cell(2, 2, 0)
    assert cells[("t1", "a")].text() == "ok"
    assert cells[("t2", "a")] == gs.Cell(2, 1, 1)
    assert cells[("t2", "a")].text() == "1F"
    assert cells[("t2", "b")].text() == "."
    assert cells[("t3", "b")] == gs.Cell(2, 0, 0)


def test_label_without_checkpoint_counts_as_unstarted(tmp_path: Path) -> None:
    planned = gs.planned_by_method(_grid(tmp_path), None, None)
    cells = gs.collect(["t9"], planned, {})
    assert cells[("t9", "a")] == gs.Cell(2, 0, 0)


def test_stale_stamp_record_is_not_done(tmp_path: Path) -> None:
    planned = gs.planned_by_method(_grid(tmp_path), None, None)
    stale = {**_rec("a", 0.1, 1), "select_version": 99}
    cells = gs.collect(["t"], planned, {"t": gs.Checkpoint("t", 10, [stale], [])})
    assert cells[("t", "a")].done == 0


def test_methods_filter_narrows_the_plan(tmp_path: Path) -> None:
    planned = gs.planned_by_method(_grid(tmp_path), ["a"], None)
    assert set(planned) == {"a"}
    cells = gs.collect(["t1", "t2"], planned, _checkpoints())
    assert {m for _, m in cells} == {"a"}


def test_eta_arithmetic(tmp_path: Path) -> None:
    planned = gs.planned_by_method(_grid(tmp_path), None, None)
    cps = {k: v for k, v in _checkpoints().items() if k != "t2"}
    cps["t2"] = gs.Checkpoint("t2", 500, [], [])
    rates = gs.method_rates(planned, cps)
    assert rates == pytest.approx({"a": 20 / 2000, "b": 60 / 2000})
    cells = gs.collect(["t1", "t2", "t3"], planned, cps)
    eta = gs.estimate_eta(cells, rates, {"t1": 1000, "t2": 500, "t3": 2000})
    # t2: 2*0.01*500 + 2*0.03*500 = 40; t3: 2*0.01*2000 + 2*0.03*2000 = 160
    assert eta.seconds == pytest.approx(200)
    assert eta.guessed_methods == ()


def test_eta_guesses_unseen_method_at_mean_rate_and_flags_it() -> None:
    cells = {("t", "a"): gs.Cell(1, 0, 0), ("t", "c"): gs.Cell(1, 0, 0)}
    eta = gs.estimate_eta(cells, {"a": 0.01, "b": 0.03}, {"t": 100})
    assert eta.seconds == pytest.approx(0.01 * 100 + 0.02 * 100)
    assert eta.guessed_methods == ("c",)


def test_eta_flags_unknown_tokens_and_none_without_rates() -> None:
    cells = {("t", "a"): gs.Cell(1, 0, 0)}
    assert gs.estimate_eta(cells, {"a": 1.0}, {}).unknown_labels == ("t",)
    assert gs.estimate_eta(cells, {}, {"t": 5}).seconds is None


def test_not_running_with_jobs_left_is_loud() -> None:
    lines = gs.runner_lines(False, 7, 10.0, 1800)
    assert lines[0].startswith("!!! NOT RUNNING - 7 jobs left.")
    assert ">>" in lines[0]
    assert gs.runner_lines(False, 0, 10.0, 1800) == ["RUNNER: not running (nothing left)"]


def test_stall_message_only_past_threshold() -> None:
    assert gs.runner_lines(True, 5, 100.0, 1800) == ["RUNNER: running"]
    lines = gs.runner_lines(True, 5, 2400.0, 1800)
    assert lines[1] == "?? no progress for 40 min - slow job or stuck"
    assert gs.runner_lines(None, 5, 99999.0, 1800) == ["RUNNER: unknown (process detection failed)"]


def test_stall_threshold_floor_and_scaling() -> None:
    assert gs.stall_threshold(None) == 1800
    assert gs.stall_threshold(100) == 1800
    assert gs.stall_threshold(1000) == 3000


def test_typical_gap_uses_current_transcript() -> None:
    cells = {("big", "a"): gs.Cell(1, 0, 0), ("small", "a"): gs.Cell(1, 0, 0)}
    tokens: dict[str, int | None] = {"big": 1000, "small": 10}
    rates = {"a": 0.1}
    assert gs.typical_gap(["big", "small"], cells, ["a"], rates, tokens, "small") == 10 * 1.0
    assert gs.typical_gap(["big", "small"], cells, ["a"], rates, tokens, None) == 10 * 100.0


def test_is_runner_ignores_the_test_file_the_watcher_and_other_grids() -> None:
    line = '"C:\py\python.exe" experiments/exp_d_curves.py --grid exp_f --methods a'
    assert gs.is_runner([line], "exp_f")
    assert not gs.is_runner([line], "exp_d")
    assert gs.is_runner(["python C:\repo\experiments\exp_d_curves.py"], "exp_d")
    assert gs.is_runner(["python experiments/exp_d_curves.py --grid=exp_f"], "exp_f")
    assert not gs.is_runner(["python -m pytest tests/test_exp_d_curves.py"], "exp_d")
    assert not gs.is_runner(["python experiments/grid_status.py exp_d_curves.py"], "exp_d")
    assert not gs.is_runner([], "exp_d")


def test_parse_log_order_and_progress() -> None:
    text = (
        "partial run: methods=all\n"
        "10:00:00 [t2] 0 records reused, 24 to run\n"
        "10:05:00 [t2] job 10/24 ok=10 failed=0 a [base] protect=none r=0.1\n"
        "10:09:00 [t2] done in 540.0s, 24 records\n"
        "Loading weights: 100%\r10:09:01 [t1] 4 records reused, 20 to run\n"
        "10:12:00 [t1] job 10/20 ok=9 failed=1 b [base] protect=none r=0.5\n"
    )
    log = gs.parse_log(text)
    assert log.order == ("t2", "t1")
    assert (log.current, log.job_n, log.job_m, log.finished) == ("t1", 10, 20, False)
    assert gs.parse_log(text.rsplit("10:09:01", 1)[0]).finished


def test_render_is_ascii_and_fits_for_fourteen_methods(tmp_path: Path) -> None:
    methods = tuple(
        f"{p}{c}-{k}"
        for p in ("", "supersede+")
        for c in ("zlib", "lzma")
        for k in ("greedy", "lazy", "onepass", "stochastic")
    )[:14]
    grid = _grid(tmp_path, methods)
    planned = gs.planned_by_method(grid, None, None)
    labels = [f"trace-commons_{i:08x}" for i in range(18)]
    cps = {labels[0]: gs.Checkpoint(labels[0], 1000, [_rec(methods[0], 0.1, 3.0)], [])}
    cells = gs.collect(labels, planned, cps)
    status = gs.Status(
        grid="fake",
        labels=labels,
        methods=list(methods),
        cells=cells,
        eta=gs.estimate_eta(cells, gs.method_rates(planned, cps), {labels[0]: 1000}),
        idle_seconds=7200.0,
        runner_alive=False,
        stall_after=1800,
        log=gs.LogState((labels[0],), labels[0], 3, 28, False),
    )
    text = gs.render(status)
    assert text.isascii()
    assert max(len(line) for line in text.splitlines()) <= 120
    assert "!!! NOT RUNNING" in text
    assert "ETA (rough)" in text


def test_once_mode_plans_the_real_exp_f_grid(tmp_path: Path) -> None:
    out = tmp_path / "grid"
    out.mkdir()
    ref = tmp_path / "ref"
    ref.mkdir()
    for label, tok in (("t1", 50_000), ("t2", 298_173)):
        (ref / f"{label}.json").write_text(
            json.dumps({"label": label, "total_tokens": tok, "records": []}), encoding="utf-8"
        )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gs.main(
            ["--grid", "exp_f"],
            registry={"exp_f": dataclasses.replace(EXP_F, out=out)},
            reference=ref,
            probe=lambda name: None,
        )
    assert "done 0/96 jobs" in buf.getvalue()


def test_once_mode_smoke(tmp_path: Path) -> None:
    out = tmp_path / "grid"
    out.mkdir()
    ref = tmp_path / "ref"
    ref.mkdir()
    payload = {"label": "t1", "total_tokens": 1000, "records": [_rec("a", 0.1, 2)], "failures": []}
    (out / "t1.json").write_text(json.dumps(payload), encoding="utf-8")
    (ref / "t2.json").write_text(
        json.dumps({"label": "t2", "total_tokens": 50, "records": []}), encoding="utf-8"
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gs.main(
            ["--grid", "fake"],
            registry={"fake": _grid(out)},
            reference=ref,
            probe=lambda name: None,
        )
    text = buf.getvalue()
    assert rc == 0
    assert "done 1/8 jobs" in text
    assert "t1" in text and "t2" in text
    assert "RUNNER: unknown" in text

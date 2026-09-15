import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_e_analysis import (  # noqa: E402
    BASE_VARIANT_KEY,
    KNEE_FLOOR,
    MIN_KIND_TRANSCRIPTS,
    _reported_kinds,
    build_curves,
    grid_fingerprint,
    interp_at,
    knee,
    load_checkpoints,
    weighted_at,
)


def _record(method: str, rate: float, **over: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "method": method,
        "protect": "none",
        "keep_ratio": rate,
        "compression_ratio": rate,
        "atom_recall": rate,
        "atom_recall_final": rate,
        "coverage": rate,
        "doc_cosine": 1.0,
        "atom_recall_by_kind": {"prose": rate, "code": rate},
        "coverage_by_kind": {"prose": rate, "code": rate},
        "elapsed_seconds": 1.0,
    }
    rec.update(over)
    return rec


def _result(label: str, rates: list[float], tokens: float = 1000.0, **over: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "label": label,
        "total_tokens": tokens,
        "n_segments": 10,
        "segments_by_kind": {"prose": 6, "code": 4},
        "schema_version": 3,
        "supersede_ceiling": 0.5,
        "records": [_record("m", r) for r in rates],
    }
    out.update(over)
    return out


def test_build_curves_ignores_every_non_baseline_variant() -> None:
    # The defect this guards: this analysis keys on (method, protect) alone, so once variants
    # started sharing a checkpoint it folded drop_top curves and path arms into the baseline's
    # average without saying so. Records predating variants carry no key and are baseline.
    result = _result("t", [0.2, 0.5])
    result["records"] += [
        _record("m", 0.3, variant_key="drop_top=2"),
        _record("m", 0.4, variant_key="paths=full"),
    ]
    curves = build_curves([result])
    assert curves[("m", "none")][0]["rates"] == [0.2, 0.5]


def test_base_variant_key_matches_the_one_exp_d_writes() -> None:
    # Duplicated as a literal to keep this module cheap to import; that makes drift the risk.
    from experiments.exp_d_curves import BASE_VARIANT

    assert BASE_VARIANT_KEY == BASE_VARIANT.key()


def test_build_curves_sorts_and_drops_repeated_rates() -> None:
    curves = build_curves([_result("a", [0.5, 0.2, 0.5, 0.8])])
    curve = curves[("m", "none")][0]
    assert curve["rates"] == [0.2, 0.5, 0.8]
    assert curve["recall_final"] == [0.2, 0.5, 0.8]


def test_build_curves_keeps_per_kind_series() -> None:
    curve = build_curves([_result("a", [0.25, 0.75])])[("m", "none")][0]
    assert curve["recall:prose"] == [0.25, 0.75]
    assert curve["cov:code"] == [0.25, 0.75]


def test_build_curves_marks_absent_kind_as_nan_not_zero() -> None:
    curve = build_curves([_result("a", [0.25, 0.75])])[("m", "none")][0]
    assert all(v != v for v in curve["recall:tool_use"])


def test_interp_at_interpolates_between_measured_points() -> None:
    curve = build_curves([_result("a", [0.2, 0.6])])[("m", "none")][0]
    assert interp_at(curve, 0.4, "recall_final") == pytest.approx(0.4)


def test_interp_at_refuses_to_extrapolate_past_the_ceiling() -> None:
    # A supersede stack that stalls at 0.6 has not been measured at 0.25; inventing a value there
    # is exactly the mistake the achieved-rate keying exists to prevent.
    curve = build_curves([_result("a", [0.6, 0.9])])[("m", "none")][0]
    assert interp_at(curve, 0.25, "recall_final") is None
    assert interp_at(curve, 0.99, "recall_final") is None


def test_interp_at_skips_all_nan_metric() -> None:
    curve = build_curves([_result("a", [0.2, 0.6])])[("m", "none")][0]
    assert interp_at(curve, 0.4, "recall:thinking") is None


def test_weighted_at_weights_by_tokens() -> None:
    a = _result("a", [0.2, 0.6], tokens=1000.0)
    b = _result("b", [0.2, 0.6], tokens=3000.0)
    for rec in b["records"]:
        rec["atom_recall_final"] = 1.0
    curves = build_curves([a, b])[("m", "none")]
    value, n, _ = weighted_at(curves, 0.4, "recall_final")
    assert n == 2
    assert value == pytest.approx((0.4 * 1000 + 1.0 * 3000) / 4000)


def test_weighted_at_excludes_transcripts_that_never_reached_the_rate() -> None:
    reached = _result("a", [0.2, 0.6])
    stalled = _result("b", [0.6, 0.9])
    value, n, ceiling = weighted_at(
        build_curves([reached, stalled])[("m", "none")], 0.3, "coverage"
    )
    assert n == 1
    assert value == pytest.approx(0.3)
    assert ceiling == pytest.approx(0.9)


def test_weighted_at_reports_ceiling_when_nothing_reached_the_rate() -> None:
    value, n, ceiling = weighted_at(
        build_curves([_result("a", [0.6, 0.9])])[("m", "none")], 0.1, "coverage"
    )
    assert value is None and n == 0
    assert ceiling == pytest.approx(0.9)


def test_knee_is_the_lowest_rate_still_above_the_floor() -> None:
    result = _result("a", [0.2, 0.5, 0.95])
    curve = build_curves([result])[("m", "none")][0]
    assert curve["recall_final"][-1] >= KNEE_FLOOR
    assert knee(curve) == pytest.approx(0.95)


def test_knee_is_none_when_even_the_gentlest_rate_fails_the_floor() -> None:
    curve = build_curves([_result("a", [0.2, 0.5])])[("m", "none")][0]
    assert knee(curve) is None


def test_reported_kinds_drops_channels_carried_by_too_few_transcripts() -> None:
    many = [
        _result(f"t{i}", [0.5], segments_by_kind={"prose": 5, "code": 2, "thinking": 1 if i else 0})
        for i in range(MIN_KIND_TRANSCRIPTS)
    ]
    reported, skipped = _reported_kinds(many)
    assert "prose" in reported and "code" in reported
    assert "thinking" not in reported
    assert any(s.startswith("thinking") for s in skipped)


def _write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *results: dict[str, Any]) -> None:
    import experiments.exp_e_analysis as mod

    monkeypatch.setattr(mod, "IN", tmp_path)
    for r in results:
        (tmp_path / f"{r['label']}.json").write_text(json.dumps(r), encoding="utf-8")


def test_grid_fingerprint_reads_fields_that_still_exist_after_schema_3() -> None:
    # The guard used to key on a run-level "config" key that SCHEMA_VERSION 3 removed, so every
    # fingerprint was None, the set was always size 1, and it could never fire.
    result = _result("a", [0.25, 0.5])
    assert "config" not in result
    fingerprint = grid_fingerprint(result)
    assert fingerprint[0] == 3
    assert fingerprint[1] == (0.25, 0.5)
    assert fingerprint[2] == ("none",)


def test_grid_fingerprint_ignores_which_methods_are_present() -> None:
    # A method that failed on every ratio leaves the grid intact and shows up as a missing curve;
    # only a changed ratio grid silently shifts every interpolation, so only that is fingerprinted.
    a = _result("a", [0.25, 0.5])
    b = _result("b", [0.25, 0.5])
    b["records"] = [{**rec, "method": "other"} for rec in b["records"]]
    assert grid_fingerprint(a) == grid_fingerprint(b)


def test_load_checkpoints_rejects_a_different_ratio_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, monkeypatch, _result("a", [0.25, 0.5]), _result("b", [0.3, 0.6]))
    with pytest.raises(SystemExit, match="disagree on the sweep grid"):
        load_checkpoints()


def test_load_checkpoints_rejects_a_different_record_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(
        tmp_path,
        monkeypatch,
        _result("a", [0.25, 0.5]),
        _result("b", [0.25, 0.5], schema_version=2),
    )
    with pytest.raises(SystemExit, match="disagree on the sweep grid"):
        load_checkpoints()


def test_load_checkpoints_rejects_a_checkpoint_missing_a_protect_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    both = _result("a", [0.25, 0.5])
    both["records"] += [_record("m", r, protect="code_only") for r in (0.25, 0.5)]
    _write(tmp_path, monkeypatch, both, _result("b", [0.25, 0.5]))
    with pytest.raises(SystemExit, match="disagree on the sweep grid"):
        load_checkpoints()


def test_load_checkpoints_tolerates_scattered_missing_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # exp_d now skips a job that raises rather than dying, so a checkpoint can be short a few
    # cells. That must not lock the analysis out: the ratio still appears via the other methods,
    # and the guard exists for a changed grid, not for an incomplete one.
    full = _result("a", [0.25, 0.5])
    full["records"] += [_record("other", r) for r in (0.25, 0.5)]
    holed = _result("b", [0.25, 0.5])
    holed["records"] += [_record("other", 0.25)]
    _write(tmp_path, monkeypatch, full, holed)
    assert {r["label"] for r in load_checkpoints()} == {"a", "b"}


def test_load_checkpoints_rejects_empty_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path, monkeypatch, _result("a", [], records=[]))
    with pytest.raises(SystemExit, match="no records"):
        load_checkpoints()

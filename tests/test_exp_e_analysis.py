import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_e_analysis import (  # noqa: E402
    KNEE_FLOOR,
    MIN_KIND_TRANSCRIPTS,
    _reported_kinds,
    build_curves,
    interp_at,
    knee,
    load_checkpoints,
    weighted_at,
)


def _record(method: str, rate: float, **over: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "method": method,
        "protect": "none",
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
        "config": {"schema_version": 2},
        "supersede_ceiling": 0.5,
        "records": [_record("m", r) for r in rates],
    }
    out.update(over)
    return out


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


def test_load_checkpoints_rejects_mixed_config_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.exp_e_analysis as mod

    monkeypatch.setattr(mod, "IN", tmp_path)
    (tmp_path / "a.json").write_text(json.dumps(_result("a", [0.5])), encoding="utf-8")
    stale = _result("b", [0.5], config={"schema_version": 1})
    (tmp_path / "b.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(SystemExit, match="config fingerprint"):
        load_checkpoints()


def test_load_checkpoints_rejects_empty_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.exp_e_analysis as mod

    monkeypatch.setattr(mod, "IN", tmp_path)
    (tmp_path / "a.json").write_text(json.dumps(_result("a", [], records=[])), encoding="utf-8")
    with pytest.raises(SystemExit, match="no records"):
        load_checkpoints()

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp_d_curves import (  # noqa: E402
    KEEP_RATIOS,
    METHODS,
    PROTECT_CODE_ONLY_METHODS,
    SCHEMA_VERSION,
    _band,
    _current_config,
    _interp_at,
    _knee,
    _load_checkpoint,
    _select_with_budget,
    _tfidf_scores,
)
from loosy_goose.budget import ProtectPolicy, token_budget  # noqa: E402
from loosy_goose.segment import Segment, SegmentKind, extract_atoms  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402


def _seg(i: int, text: str, kind: SegmentKind = "prose", protected: bool = False) -> Segment:
    return Segment(i, i, "assistant", kind, text, protected, extract_atoms(text))


def _corpus() -> list[Segment]:
    segs = [_seg(i, f"Paragraph {i} talks about topic {i % 3} in some detail.") for i in range(8)]
    segs.append(_seg(8, "def f():\n    return 1", kind="code", protected=True))
    segs.append(_seg(9, '{"command": "ls -la /tmp"}', kind="tool_use", protected=True))
    segs.append(_seg(10, "Closing remarks about topic 1 and the plan for topic 2."))
    return segs


def test_select_with_budget_never_exceeds_explicit_budget() -> None:
    segs = _corpus()
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=len(segs))
    budget = 20
    kept = _select_with_budget(segs, scores, budget, protect="none")
    assert sum(count_tokens(s.text) for s in kept) <= budget
    assert [s.id for s in kept] == sorted(s.id for s in kept)


def test_select_with_budget_forces_protected_kinds() -> None:
    segs = _corpus()
    scores = np.zeros(len(segs))
    kept = _select_with_budget(segs, scores, budget=1_000_000, protect="code_only")
    assert {8} <= {s.id for s in kept}


def test_select_with_budget_uses_the_budget_passed_in_not_the_list() -> None:
    # The whole reason this helper exists: a stacked method hands it a shrunk segment list, and
    # the budget must still be the one derived from the *original* full transcript, not
    # re-derived from the shrunk list (which would silently give the stack a smaller budget than
    # single-stage methods get at the same keep_ratio).
    full = _corpus()
    shared_budget = token_budget(full, 0.5)
    shrunk = full[:4]  # far fewer tokens than the full corpus
    scores = np.arange(len(shrunk), dtype=np.float64)
    kept = _select_with_budget(shrunk, scores, shared_budget, protect="none")
    assert kept == shrunk  # small enough that the shared budget covers all of it
    assert sum(count_tokens(s.text) for s in kept) < shared_budget


def test_tfidf_scores_prefers_distinctive_segment() -> None:
    filler = [_seg(i, "the same common words again and again") for i in range(5)]
    rare = _seg(5, "zeppelin quartz oxbow")
    scores = _tfidf_scores([*filler, rare])
    assert int(np.argmax(scores)) == 5


def test_tfidf_scores_handles_empty_vocabulary_and_empty_list() -> None:
    segs = [_seg(0, "!!!"), _seg(1, "???")]
    assert _tfidf_scores(segs).tolist() == [0.0, 0.0]
    assert _tfidf_scores([]).tolist() == []


def test_band_only_touches_code_and_tool_use_kinds() -> None:
    segs = _corpus()
    banded = _band(segs, 0.0)
    for orig, b in zip(segs, banded, strict=True):
        if orig.kind in ("code", "tool_use"):
            assert b is not orig
        else:
            assert b is orig


def test_band_at_quality_one_is_verbatim() -> None:
    segs = _corpus()
    banded = _band(segs, 1.0)
    code = next(s for s in segs if s.kind == "code")
    banded_code = next(s for s in banded if s.kind == "code")
    assert banded_code.text == code.text


@pytest.mark.parametrize("name", sorted(METHODS))
def test_methods_respect_the_shared_token_budget(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    segs = _corpus()

    def fake_embed(texts: list[str], model_name: str) -> np.ndarray:
        rng = np.random.default_rng(hash((tuple(texts), model_name)) % (2**32))
        x = rng.normal(size=(len(texts), 8))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms > 0, norms, 1.0)

    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embed)
    budget = token_budget(segs, 0.5)
    kept = METHODS[name](segs, 0.5, "none", tmp_path)
    assert sum(count_tokens(s.text) for s in kept) <= budget
    assert [s.id for s in kept] == sorted(s.id for s in kept)


def test_current_config_fingerprint_covers_what_would_invalidate_a_checkpoint() -> None:
    cfg = _current_config()
    assert cfg["schema_version"] == SCHEMA_VERSION
    assert cfg["keep_ratios"] == list(KEEP_RATIOS)
    assert cfg["methods"] == sorted(METHODS)
    assert set(cfg["protects"]) == {"none", "code_only"}
    assert PROTECT_CODE_ONLY_METHODS  # sanity: the "code_only" branch above is exercised
    # Deterministic and JSON-round-trippable, since it is compared against a loaded checkpoint.
    assert _current_config() == json.loads(json.dumps(_current_config()))


def test_load_checkpoint_reuses_a_matching_fingerprint(tmp_path: Path) -> None:
    cfg = _current_config()
    out_path = tmp_path / "t.json"
    out_path.write_text(json.dumps({"label": "t", "config": cfg, "records": []}), encoding="utf-8")
    cached, reason = _load_checkpoint(out_path, force=False, current_config=cfg)
    assert cached is not None and cached["label"] == "t"
    assert reason is None


def test_load_checkpoint_invalidates_on_fingerprint_mismatch(tmp_path: Path) -> None:
    cfg = _current_config()
    stale_cfg = {**cfg, "keep_ratios": [*cfg["keep_ratios"], 0.07]}
    out_path = tmp_path / "t.json"
    out_path.write_text(
        json.dumps({"label": "t", "config": stale_cfg, "records": []}), encoding="utf-8"
    )
    cached, reason = _load_checkpoint(out_path, force=False, current_config=cfg)
    assert cached is None
    assert reason == "config fingerprint mismatch"


def test_load_checkpoint_treats_a_missing_config_key_as_a_mismatch(tmp_path: Path) -> None:
    # The exact regression this guards: a checkpoint written before "config" existed at all.
    cfg = _current_config()
    out_path = tmp_path / "t.json"
    out_path.write_text(json.dumps({"label": "t", "records": []}), encoding="utf-8")
    cached, reason = _load_checkpoint(out_path, force=False, current_config=cfg)
    assert cached is None
    assert reason == "config fingerprint mismatch"


def test_load_checkpoint_force_always_reruns_even_with_a_matching_fingerprint(
    tmp_path: Path,
) -> None:
    cfg = _current_config()
    out_path = tmp_path / "t.json"
    out_path.write_text(json.dumps({"label": "t", "config": cfg, "records": []}), encoding="utf-8")
    cached, reason = _load_checkpoint(out_path, force=True, current_config=cfg)
    assert cached is None
    assert reason == "--force"


def test_load_checkpoint_no_file_means_no_checkpoint(tmp_path: Path) -> None:
    cached, reason = _load_checkpoint(tmp_path / "missing.json", force=False, current_config={})
    assert cached is None and reason is None


def _curve(rates: list[float], recall_final: list[float]) -> dict[str, object]:
    return {
        "label": "t",
        "weight": 1.0,
        "rates": rates,
        "recall": recall_final,
        "recall_final": recall_final,
        "coverage": recall_final,
    }


def test_interp_at_rejects_targets_outside_the_measured_range() -> None:
    curve = _curve([0.2, 0.5, 0.8], [0.5, 0.8, 0.95])
    assert _interp_at(curve, 0.5, "recall_final") == pytest.approx(0.8)
    assert _interp_at(curve, 0.1, "recall_final") is None  # below the lowest measured rate
    assert _interp_at(curve, 0.9, "recall_final") is None  # above the highest measured rate
    assert _interp_at(None, 0.5, "recall_final") is None


def test_knee_is_the_lowest_rate_still_above_the_floor() -> None:
    # Holds the floor down to 0.3, then fails at 0.2: knee is 0.3, not the lowest tested rate.
    curve = _curve([0.2, 0.3, 0.5, 0.8], [0.5, 0.91, 0.93, 0.99])
    assert _knee(curve, floor=0.9) == pytest.approx(0.3)


def test_knee_holds_all_the_way_down_to_the_lowest_tested_rate() -> None:
    curve = _curve([0.2, 0.5, 0.8], [0.91, 0.95, 0.99])
    assert _knee(curve, floor=0.9) == pytest.approx(0.2)


def test_knee_never_reaches_floor_returns_none() -> None:
    curve = _curve([0.2, 0.5, 0.8], [0.5, 0.6, 0.7])
    assert _knee(curve, floor=0.9) is None
    assert _knee({"rates": [], "recall_final": []}, floor=0.9) is None


@pytest.mark.parametrize("protect", ["none", "code_only"])
def test_plain_baselines_respect_protect_policy(
    protect: ProtectPolicy, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    segs = _corpus()

    def fake_embed(texts: list[str], model_name: str) -> np.ndarray:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(len(texts), 8))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms > 0, norms, 1.0)

    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embed)
    forced = {"none": set(), "code_only": {8}}[protect]
    for name in ("tfidf", "leverage"):
        kept = METHODS[name](segs, 0.5, protect, tmp_path)
        assert forced <= {s.id for s in kept}

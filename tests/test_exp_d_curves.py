import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_d_curves as exp_d  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    BASE_VARIANT,
    KEEP_RATIOS,
    METHODS,
    PROTECT_CODE_ONLY_METHODS,
    SCHEMA_VERSION,
    VARIANTS,
    Job,
    Variant,
    _band,
    _interp_at,
    _knee,
    _load_checkpoint,
    _record_identity,
    _select_with_budget,
    _tfidf_scores,
    plan_jobs,
    segments_digest,
)
from loosy_goose.budget import ProtectPolicy, token_budget  # noqa: E402
from loosy_goose.segment import (  # noqa: E402
    ATOMS_VERSION,
    Segment,
    SegmentKind,
    extract_atoms,
)
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
    kept = METHODS[name](segs, 0.5, "none", tmp_path, BASE_VARIANT)
    assert sum(count_tokens(s.text) for s in kept) <= budget
    assert [s.id for s in kept] == sorted(s.id for s in kept)


def test_variant_key_is_base_only_for_the_default_configuration() -> None:
    assert BASE_VARIANT.key() == "base"
    assert Variant(drop_top=2).key() == "drop_top=2"


def test_variant_applies_only_to_the_methods_it_names() -> None:
    v = Variant(drop_top=1, applies_to=("leverage",))
    assert v.covers("leverage")
    assert not v.covers("random")
    assert BASE_VARIANT.covers("random")


def test_variant_threads_drop_top_into_the_compress_config() -> None:
    assert BASE_VARIANT.compress_config("leverage").drop_top == 0
    assert Variant(drop_top=3).compress_config("ridge").drop_top == 3


def test_plan_jobs_covers_every_method_ratio_and_variant_exactly_once() -> None:
    jobs = plan_jobs()
    assert len({j.identity() for j in jobs}) == len(jobs)
    base = [j for j in jobs if j.variant.key() == "base"]
    expected = len(METHODS) * len(KEEP_RATIOS) + len(PROTECT_CODE_ONLY_METHODS) * len(KEEP_RATIOS)
    assert len(base) == expected
    # A scoring variant must not schedule work for methods it cannot change.
    for job in jobs:
        assert job.variant.covers(job.method)


def test_plan_jobs_puts_the_baseline_first_so_an_interrupted_run_is_still_usable() -> None:
    jobs = plan_jobs()
    first_non_base = next(i for i, j in enumerate(jobs) if j.variant.key() != "base")
    assert all(j.variant.key() == "base" for j in jobs[:first_non_base])


def test_segments_digest_tracks_text_and_kind_but_not_ordering_of_unrelated_runs() -> None:
    a = [_seg(0, "alpha"), _seg(1, "beta")]
    b = [_seg(0, "alpha"), _seg(1, "beta")]
    assert segments_digest(a) == segments_digest(b)
    assert segments_digest(a) != segments_digest([_seg(0, "alpha"), _seg(1, "gamma")])
    assert segments_digest(a) != segments_digest([_seg(0, "alpha", kind="code"), _seg(1, "beta")])


def test_segments_digest_covers_the_atom_extractor_version() -> None:
    # Atoms are the recall metric's denominator, so a changed extractor invalidates stored
    # records exactly as a changed segmenter does, even though no segment text moves.
    segs = [_seg(0, "alpha"), _seg(1, "beta")]
    before = segments_digest(segs)
    with mock.patch.object(exp_d, "ATOMS_VERSION", ATOMS_VERSION + 1):
        assert segments_digest(segs) != before


def test_record_identity_defaults_a_variantless_record_to_the_baseline() -> None:
    # Records written before variants existed carry no variant_key; they are baseline records.
    rec = {"method": "tfidf", "protect": "none", "keep_ratio": 0.5}
    assert _record_identity(rec) == ("tfidf", "none", 0.5, "base")


def test_job_identity_distinguishes_variants_of_the_same_method() -> None:
    a = Job("leverage", "none", 0.5, BASE_VARIANT)
    b = Job("leverage", "none", 0.5, Variant(drop_top=2))
    assert a.identity() != b.identity()


def test_load_checkpoint_reuses_a_matching_schema(tmp_path: Path) -> None:
    out_path = tmp_path / "t.json"
    out_path.write_text(
        json.dumps({"label": "t", "schema_version": SCHEMA_VERSION, "records": []}),
        encoding="utf-8",
    )
    prior, reason = _load_checkpoint(out_path, force=False)
    assert prior is not None and prior["label"] == "t"
    assert reason is None


def test_load_checkpoint_discards_an_older_record_schema(tmp_path: Path) -> None:
    # A record-shape change is the one thing a checkpoint cannot be salvaged from; adding a
    # method or a variant must NOT land here, which is what the plan_jobs tests above cover.
    out_path = tmp_path / "t.json"
    out_path.write_text(
        json.dumps({"label": "t", "schema_version": SCHEMA_VERSION - 1, "records": []}),
        encoding="utf-8",
    )
    prior, reason = _load_checkpoint(out_path, force=False)
    assert prior is None
    assert reason is not None and "record schema" in reason


def test_load_checkpoint_discards_a_checkpoint_with_no_schema_version(tmp_path: Path) -> None:
    out_path = tmp_path / "t.json"
    out_path.write_text(json.dumps({"label": "t", "records": []}), encoding="utf-8")
    prior, reason = _load_checkpoint(out_path, force=False)
    assert prior is None and reason is not None


def test_load_checkpoint_force_always_reruns(tmp_path: Path) -> None:
    out_path = tmp_path / "t.json"
    out_path.write_text(
        json.dumps({"label": "t", "schema_version": SCHEMA_VERSION, "records": []}),
        encoding="utf-8",
    )
    prior, reason = _load_checkpoint(out_path, force=True)
    assert prior is None
    assert reason == "--force"


def test_load_checkpoint_no_file_means_no_checkpoint(tmp_path: Path) -> None:
    prior, reason = _load_checkpoint(tmp_path / "missing.json", force=False)
    assert prior is None and reason is None


def test_variants_declare_at_least_one_non_baseline_configuration() -> None:
    assert VARIANTS[0] == BASE_VARIANT
    assert len({v.key() for v in VARIANTS}) == len(VARIANTS)


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
        kept = METHODS[name](segs, 0.5, protect, tmp_path, BASE_VARIANT)
        assert forced <= {s.id for s in kept}

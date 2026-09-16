import contextlib
import io
import json
import random
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_d_curves as exp_d  # noqa: E402
from experiments.baselines import tfidf as baseline_tfidf  # noqa: E402
from experiments.exp_d_curves import (  # noqa: E402
    BAND_QUALITY_VERSION,
    BASE_VARIANT,
    FLUSH_EVERY,
    KEEP_RATIOS,
    METHODS,
    PATH_METHODS,
    PATH_MIN_MENTIONS,
    PATH_MIN_MENTIONS_ALT,
    PROTECT_CODE_ONLY_METHODS,
    SCHEMA_VERSION,
    SPECTRAL_SCORED,
    SUPERSEDING_METHODS,
    TRANSFORMING_METHODS,
    VARIANTS,
    Job,
    RunContext,
    Variant,
    _band,
    _interp_at,
    _is_path_key,
    _knee,
    _load_checkpoint,
    _metric,
    _path_arm_table,
    _path_table_cost_table,
    _pool,
    _record_identity,
    _tfidf_scores,
    _weighted_at_rate,
    analyse,
    band_quality_stamp,
    failure_report,
    iter_trace_commons,
    paths_stamp,
    plan_jobs,
    segments_digest,
    supersede_stamp,
    transforms_stamp,
)
from loosy_goose.budget import (  # noqa: E402
    ProtectPolicy,
    select_by_score,
    select_with_budget,
    token_budget,
)
from loosy_goose.code import TRANSFORMS_VERSION  # noqa: E402
from loosy_goose.paths import PATHS_VERSION, PathTable, build_table, expand  # noqa: E402
from loosy_goose.segment import (  # noqa: E402
    ATOMS_VERSION,
    Segment,
    SegmentKind,
    extract_atoms,
)
from loosy_goose.supersede import SUPERSEDE_VERSION  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Block, Transcript, Turn  # noqa: E402


def _seg(i: int, text: str, kind: SegmentKind = "prose", protected: bool = False) -> Segment:
    return Segment(i, i, "assistant", kind, text, protected, extract_atoms(text))


def _ctx(cache_dir: Path, variant: Variant, segs: list[Segment]) -> RunContext:
    return RunContext(cache_dir=cache_dir, variant=variant, table=build_table(segs))


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
    kept = select_with_budget(segs, scores, budget, protect="none")
    assert sum(count_tokens(s.text) for s in kept) <= budget
    assert [s.id for s in kept] == sorted(s.id for s in kept)


def test_select_with_budget_forces_protected_kinds() -> None:
    segs = _corpus()
    scores = np.zeros(len(segs))
    kept = select_with_budget(segs, scores, budget=1_000_000, protect="code_only")
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
    kept = select_with_budget(shrunk, scores, shared_budget, protect="none")
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
    kept = METHODS[name](segs, 0.5, "none", _ctx(tmp_path, BASE_VARIANT, segs))
    assert sum(count_tokens(s.text) for s in kept) <= budget
    assert [s.id for s in kept] == sorted(s.id for s in kept)


def test_variant_key_is_base_only_for_the_default_configuration() -> None:
    assert BASE_VARIANT.key() == "base"
    assert Variant(drop_top=2).key() == "drop_top=2"
    # The depth knob at its default must leave every existing record's key alone.
    assert Variant(tool_depth="follow").key() == "base"
    assert Variant(tool_depth="half").key() == "tool_depth=half"


def test_variant_applies_only_to_the_methods_it_names() -> None:
    v = Variant(drop_top=1, applies_to=("leverage",))
    assert v.covers("leverage")
    assert not v.covers("random")
    assert BASE_VARIANT.covers("random")


def test_variant_threads_drop_top_into_the_compress_config() -> None:
    assert BASE_VARIANT.compress_config("leverage").drop_top == 0
    assert Variant(drop_top=3).compress_config("ridge").drop_top == 3


def test_tool_quality_follows_the_depth_setting() -> None:
    assert BASE_VARIANT.tool_quality(0.3) is None
    assert Variant(tool_depth="flat0.5").tool_quality(0.3) == 0.5
    assert Variant(tool_depth="flat0.5").tool_quality(0.9) == 0.5
    assert Variant(tool_depth="flat0.25").tool_quality(0.05) == 0.25
    assert Variant(tool_depth="half").tool_quality(0.3) == pytest.approx(0.15)
    assert Variant(tool_depth="half").tool_quality(0.9) == pytest.approx(0.45)


def test_band_threads_the_tool_depth_override_into_the_shrinker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segs = _corpus()
    seen: list[tuple[str, float, float | None]] = []
    real = exp_d.quantize_segment

    def watched(seg: Segment, quality: float, **kwargs: Any) -> Segment:
        seen.append((seg.kind, quality, kwargs.get("tool_quality")))
        return real(seg, quality, **kwargs)

    monkeypatch.setattr(exp_d, "quantize_segment", watched)
    _band(segs, 0.4)
    assert seen and all(tq is None for _, _, tq in seen)
    seen.clear()
    _band(segs, 0.4, 0.2)
    # The override reaches every shrunk segment; `quantize_segment` itself scopes it to tool_use.
    assert {(q, tq) for _, q, tq in seen} == {(0.4, 0.2)}
    assert {kind for kind, _, _ in seen} == {"code", "tool_use"}


@pytest.mark.parametrize("name", sorted(TRANSFORMING_METHODS))
def test_band_methods_hand_the_variants_depth_to_the_shrinker(
    name: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    segs = _corpus()

    def fake_embed(texts: list[str], model_name: str) -> np.ndarray:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(len(texts), 8))
        return x / np.where((n := np.linalg.norm(x, axis=1, keepdims=True)) > 0, n, 1.0)

    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embed)
    seen: set[float | None] = set()
    real = exp_d.quantize_segment

    def watched(seg: Segment, quality: float, **kwargs: Any) -> Segment:
        seen.add(kwargs.get("tool_quality"))
        return real(seg, quality, **kwargs)

    monkeypatch.setattr(exp_d, "quantize_segment", watched)
    variant = Variant(tool_depth="half", applies_to=TRANSFORMING_METHODS)
    METHODS[name](segs, 0.5, "none", _ctx(tmp_path, variant, segs))
    assert seen == {0.25}
    seen.clear()
    METHODS[name](segs, 0.5, "none", _ctx(tmp_path, BASE_VARIANT, segs))
    assert seen == {None}


def test_depth_variants_are_swept_on_the_band_methods_only() -> None:
    depth = [j for j in plan_jobs() if j.variant.tool_depth != "follow"]
    assert len(depth) == 72
    assert {j.variant.key() for j in depth} == {
        "tool_depth=flat0.5",
        "tool_depth=flat0.25",
        "tool_depth=half",
    }
    assert {j.method for j in depth} == set(TRANSFORMING_METHODS)
    assert {j.protect for j in depth} == {"none"}
    for key in ("tool_depth=flat0.5", "tool_depth=flat0.25", "tool_depth=half"):
        for method in TRANSFORMING_METHODS:
            ratios = [j.ratio for j in depth if j.variant.key() == key and j.method == method]
            assert sorted(ratios) == sorted(KEEP_RATIOS)


def test_plan_jobs_covers_every_method_ratio_and_variant_exactly_once() -> None:
    jobs = plan_jobs()
    assert len({j.identity() for j in jobs}) == len(jobs)
    base = [j for j in jobs if j.variant.key() == "base"]
    expected = len(METHODS) * len(KEEP_RATIOS) + len(PROTECT_CODE_ONLY_METHODS) * len(KEEP_RATIOS)
    assert len(base) == expected
    # A scoring variant must not schedule work for methods it cannot change.
    for job in jobs:
        assert job.variant.covers(job.method)


def test_protect_code_only_is_measured_on_the_base_variant_only() -> None:
    # Crossing protect=code_only with the drop_top variants produced 36 records that every table,
    # curve and plot filtered straight back out again — a dead corner of the grid, not a result.
    code_only = [j for j in plan_jobs() if j.protect == "code_only"]
    assert {j.variant.key() for j in code_only} == {BASE_VARIANT.key()}
    assert len(code_only) == len(PROTECT_CODE_ONLY_METHODS) * len(KEEP_RATIOS)


def test_plan_jobs_totals_the_grid_the_run_is_budgeted_for() -> None:
    # The grid is a multi-hour run, so its size is pinned rather than inferred: 156 base (11
    # methods + 2 protect=code_only rows, x 12 ratios) + 144 drop_top (3 settings x 4 scored
    # methods x 12) + 300 path (5 arms x 5 methods x 12) + 72 depth (3 settings x 2 band
    # methods x 12).
    jobs = plan_jobs()
    base = (len(METHODS) + len(PROTECT_CODE_ONLY_METHODS)) * len(KEEP_RATIOS)
    drop_top = len([v for v in VARIANTS if v.drop_top]) * len(SPECTRAL_SCORED) * len(KEEP_RATIOS)
    path = len([v for v in VARIANTS if v.paths != "off"]) * len(PATH_METHODS) * len(KEEP_RATIOS)
    depth = (
        len([v for v in VARIANTS if v.tool_depth != "follow"])
        * len(TRANSFORMING_METHODS)
        * len(KEEP_RATIOS)
    )
    assert (base, drop_top, path, depth) == (156, 144, 300, 72)
    assert len(jobs) == base + drop_top + path + depth == 672
    assert len({j.identity() for j in jobs}) == len(jobs)


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


def test_segments_digest_covers_the_supersession_version() -> None:
    # apply_supersession builds final_atoms, the denominator of atom_recall_final on every
    # record, so a changed pass invalidates the whole checkpoint, not only supersede+* records.
    segs = [_seg(0, "alpha"), _seg(1, "beta")]
    before = segments_digest(segs)
    with mock.patch.object(exp_d, "SUPERSEDE_VERSION", SUPERSEDE_VERSION + 1):
        assert segments_digest(segs) != before


def test_record_identity_defaults_a_variantless_record_to_the_baseline() -> None:
    # Records written before variants existed carry no variant_key; they are baseline records.
    rec = {"method": "tfidf", "protect": "none", "keep_ratio": 0.5}
    identity = _record_identity(rec)
    assert identity.method == "tfidf"
    assert identity.variant_key == "base"
    # Both stamps are 0 for a method that neither shrinks nor substitutes, so a record predating
    # either version stays valid rather than being needlessly recomputed.
    assert identity.transforms_version == 0
    assert identity.paths_version == 0
    assert identity.supersede_version == 0


def test_record_identity_is_read_by_name_not_position() -> None:
    # It went from four fields to seven, and the leftover positional unpack crashed
    # _write_summary — the last step of a multi-hour run. Names are what stop that recurring.
    identity = _record_identity({"method": "tfidf", "protect": "none", "keep_ratio": 0.5})
    assert identity._fields == (
        "method",
        "protect",
        "ratio",
        "variant_key",
        "transforms_version",
        "paths_version",
        "band_quality_version",
        "supersede_version",
    )


def test_a_paths_rewrite_invalidates_only_the_arm_records() -> None:
    # Same scoping as the shrinker stamp: paths.py drives the arms and nothing else, so a
    # rewrite there recomputes the arms and leaves the base variant's records valid.
    old_base = {"method": "tfidf", "protect": "none", "keep_ratio": 0.5}
    old_arm = {
        "method": "tfidf",
        "protect": "none",
        "keep_ratio": 0.5,
        "variant_key": "paths=full",
    }
    planned = {j.identity() for j in plan_jobs()}
    assert _record_identity(old_base) in planned
    assert _record_identity(old_arm) not in planned
    assert paths_stamp(Variant()) == 0
    assert paths_stamp(Variant(paths="full")) == PATHS_VERSION


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
        kept = METHODS[name](segs, 0.5, protect, _ctx(tmp_path, BASE_VARIANT, segs))
        assert forced <= {s.id for s in kept}


@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5, 0.9])
def test_select_with_budget_agrees_with_the_shared_selector(ratio: float) -> None:
    # Every method now routes through select_with_budget so the path arms can score one list and
    # emit another. That is only safe if it picks exactly what budget.select_by_score picks —
    # otherwise the base records already on disk would no longer describe what the code does.
    segs = _corpus()
    scores = np.random.default_rng(7).uniform(size=len(segs))
    assert select_with_budget(segs, scores, token_budget(segs, ratio), "none") == select_by_score(
        segs, scores, ratio, "none"
    )


@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5, 0.9])
def test_runner_tfidf_reproduces_the_baseline_selector_exactly(
    ratio: float, tmp_path: Path
) -> None:
    segs = _corpus()
    ctx = _ctx(tmp_path, BASE_VARIANT, segs)
    assert METHODS["tfidf"](segs, ratio, "none", ctx) == baseline_tfidf(segs, ratio, protect="none")


def _path_corpus() -> list[Segment]:
    """A transcript whose paths repeat, including one Windows path that only exists in escaped
    form inside a tool call — the case the extractor fix was written for."""
    return [
        _seg(0, "First we read src/loosy_goose/select.py to find the scoring code we want."),
        _seg(1, '{"file_path": "src/loosy_goose/select.py"}', kind="tool_use", protected=True),
        _seg(2, "Then src/loosy_goose/select.py is edited and src/loosy_goose/budget.py read."),
        _seg(3, '{"file_path": "C:\\\\repo\\\\src\\\\budget.py"}', kind="tool_use", protected=True),
        _seg(4, '{"file_path": "C:\\\\repo\\\\src\\\\budget.py"}', kind="tool_use", protected=True),
        _seg(5, "Finally src/loosy_goose/budget.py passes its tests and the run finishes here."),
    ]


def test_the_test_corpus_actually_has_a_table_to_substitute() -> None:
    table = build_table(_path_corpus())
    assert "src/loosy_goose/select.py" in table.paths
    # Only reachable through the decoded form: the raw segment text doubles every backslash. The
    # drive letter has to survive — ATOMS_VERSION 3 fixed a pattern that matched from `repo`
    # onward and silently truncated every Windows path.
    assert "C:\\repo\\src\\budget.py" in table.paths


def test_score_only_arm_emits_the_original_text(tmp_path: Path) -> None:
    segs = _path_corpus()
    ctx = RunContext(tmp_path, Variant(paths="score_only"), build_table(segs))
    kept = METHODS["tfidf"](segs, 1.0, "none", ctx)
    assert [s.text for s in kept] == [s.text for s in segs]


def test_full_arm_emits_markers_that_expand_back_to_the_original(tmp_path: Path) -> None:
    segs = _path_corpus()
    table = build_table(segs)
    ctx = RunContext(tmp_path, Variant(paths="full"), table)
    kept = METHODS["tfidf"](segs, 1.0, "none", ctx)
    assert any(table.marker in s.text for s in kept)
    original = {s.id: s.text for s in segs}
    for s in kept:
        if s.kind == "prose":
            assert expand(s.text, table) == original[s.id]


def test_table_only_arm_changes_no_text_at_all(tmp_path: Path) -> None:
    segs = _path_corpus()
    ctx = RunContext(tmp_path, Variant(paths="table_only"), build_table(segs))
    kept = METHODS["tfidf"](segs, 1.0, "none", ctx)
    assert [s.text for s in kept] == [s.text for s in segs]


@pytest.mark.parametrize("arm", ["off", "score_only", "table_only", "full"])
def test_every_arm_selects_against_the_same_shared_budget(arm: str, tmp_path: Path) -> None:
    # The table is charged into the reported compression ratio, not deducted from the budget, so
    # selection is handed the identical number of tokens in every arm. An arm that emits a table
    # therefore lands further right on the achieved-rate axis, which is where its cost is paid.
    segs = _path_corpus()
    ctx = RunContext(tmp_path, Variant(paths=arm), build_table(segs))  # type: ignore[arg-type]
    kept = METHODS["tfidf"](segs, 0.5, "none", ctx)
    assert sum(count_tokens(s.text) for s in kept) <= token_budget(segs, 0.5)


def test_only_the_table_arms_pay_for_the_table_in_the_reported_ratio(tmp_path: Path) -> None:
    segs = _path_corpus()
    table = build_table(segs)
    assert table.tokens() > 0
    ratios = {}
    for arm in ("off", "table_only"):
        ctx = RunContext(tmp_path, Variant(paths=arm), table)  # type: ignore[arg-type]
        kept = METHODS["tfidf"](segs, 0.5, "none", ctx)
        ratios[arm] = exp_d._record(
            "tfidf",
            "none",
            0.5,
            ctx,
            token_budget(segs, 0.5),
            segs,
            kept,
            np.eye(len(segs)),
            set(),
            0.0,
        )
    assert ratios["table_only"]["table_tokens"] == table.tokens()
    assert ratios["off"]["table_tokens"] == 0
    assert ratios["table_only"]["compression_ratio"] > ratios["off"]["compression_ratio"]
    # The same kept text, so anything the table arm gains in recall it gained from the table.
    assert ratios["table_only"]["kept_tokens"] == ratios["off"]["kept_tokens"]
    assert ratios["table_only"]["guaranteed_share"] > 0.0
    assert ratios["off"]["guaranteed_share"] == 0.0


def test_pool_emits_the_original_only_for_the_score_only_arm(tmp_path: Path) -> None:
    segs = _path_corpus()
    table = build_table(segs)

    def pool(arm: str) -> tuple[list[Segment], list[Segment] | None]:
        return _pool(segs, RunContext(tmp_path, Variant(paths=arm), table))  # type: ignore[arg-type]

    scoring, emit = pool("off")
    assert scoring == segs and emit is None
    scoring, emit = pool("score_only")
    assert scoring != segs and emit == segs
    scoring, emit = pool("table_only")
    assert scoring == segs and emit is None
    scoring, emit = pool("full")
    assert scoring != segs and emit is None


def test_only_the_table_arms_emit_and_only_the_marker_arms_substitute() -> None:
    assert not BASE_VARIANT.substitutes and not BASE_VARIANT.emits_table
    assert Variant(paths="score_only").substitutes
    assert not Variant(paths="score_only").emits_table
    assert not Variant(paths="table_only").substitutes
    assert Variant(paths="table_only").emits_table
    assert Variant(paths="full").substitutes and Variant(paths="full").emits_table


def test_variant_key_names_the_path_arm() -> None:
    assert Variant(paths="full").key() == "paths=full"
    assert Variant(drop_top=2, paths="full").key() == "drop_top=2+paths=full"
    assert _is_path_key("paths=full")
    assert not _is_path_key("drop_top=2")


def test_path_arms_are_scheduled_only_where_they_are_swept() -> None:
    path_jobs = [j for j in plan_jobs() if _is_path_key(j.variant.key())]
    assert {j.method for j in path_jobs} == set(PATH_METHODS)
    # protect=code_only is a baseline-side question; the arms must not double their cost on it.
    assert {j.protect for j in path_jobs} == {"none"}
    assert {j.variant.paths for j in path_jobs} == {"score_only", "table_only", "full"}
    # Five arms, not three: both table-emitting arms are swept at both min_mentions settings,
    # which is how that tradeoff gets measured rather than picked.
    arms = {j.variant.key() for j in path_jobs}
    assert arms == {
        "paths=score_only",
        "paths=table_only",
        "paths=table_only+mm=1",
        "paths=full",
        "paths=full+mm=1",
    }
    assert len(path_jobs) == len(arms) * len(PATH_METHODS) * len(KEEP_RATIOS)


def test_the_cost_only_arm_is_swept_at_both_table_thresholds() -> None:
    # Arm C at mm=1 is what prices the expensive table on its own; without it the extra rows are
    # only ever observed jointly with substitution, which cannot separate cost from effect.
    settings = {v.path_min_mentions for v in VARIANTS if v.paths == "table_only"}
    assert settings == {PATH_MIN_MENTIONS, PATH_MIN_MENTIONS_ALT}
    arm = Variant(paths="table_only", path_min_mentions=PATH_MIN_MENTIONS_ALT)
    assert arm.key() == "paths=table_only+mm=1"
    assert arm.emits_table and not arm.substitutes
    assert len({v.key() for v in VARIANTS}) == len(VARIANTS)


def test_the_cost_only_arm_at_mm_1_reaches_the_table_cost_row(tmp_path: Path) -> None:
    # That row used to find its guaranteed share by an endswith test against "paths=full+mm=1",
    # so a second arm at the same setting would have been read as unmeasured ("-") rather than
    # reported.
    arm = Variant(paths="table_only", path_min_mentions=PATH_MIN_MENTIONS_ALT)
    ctx = RunContext(tmp_path, arm, PathTable(("a/b.py", "c/d.py")))
    rec = exp_d._record("tfidf", "none", 0.5, ctx, 100, [], [], None, set(), 0.0)
    rec["guaranteed_share"] = 0.421
    result = {
        "label": "t0",
        "total_tokens": 10_000,
        "path_tables": {"1": {"paths": 9, "tokens": 90}},
        "records": [rec],
    }
    text = "\n".join(_path_table_cost_table([result]))
    assert "0.421" in text


def _full_grid_results(tmp_path: Path, n_transcripts: int = 3) -> list[dict[str, Any]]:
    """Synthetic checkpoints over the real `plan_jobs()` grid, so every method, protect policy,
    ratio, drop_top variant and path arm is present."""
    rng = random.Random(0)
    table = PathTable(("a/b.py", "c/d.py"))
    results = []
    for t in range(n_transcripts):
        records = []
        for job in plan_jobs():
            ctx = RunContext(tmp_path, job.variant, table)
            rec = exp_d._record(
                job.method, job.protect, job.ratio, ctx, 100, [], [], None, set(), 0.1
            )
            rec.update(
                compression_ratio=max(0.02, job.ratio * rng.uniform(0.8, 1.0)),
                atom_recall=job.ratio * 0.9,
                atom_recall_earned=job.ratio * 0.88,
                atom_recall_final=job.ratio * 0.92,
                atom_recall_final_earned=job.ratio * 0.9,
                coverage=job.ratio * 0.85,
                guaranteed_share=0.28 if job.variant.emits_table else 0.0,
            )
            records.append(rec)
        results.append(
            {
                "label": f"t{t}",
                "public": True,
                "n_turns": 5,
                "n_segments": 50,
                "total_tokens": 10_000,
                "segments_by_kind": {},
                "schema_version": SCHEMA_VERSION,
                "segments_digest": "x",
                "path_tables": {"1": {"paths": 9, "tokens": 90}, "2": {"paths": 2, "tokens": 20}},
                "supersede_ceiling": 0.7,
                "records": records,
                "failures": [],
            }
        )
    return results


def test_write_summary_renders_every_section_for_a_full_grid(tmp_path: Path) -> None:
    """The last step of a run that takes hours, previously untested.

    A crash here costs the whole run's reporting, and the tables are the only thing anyone
    reads.
    """
    results = _full_grid_results(tmp_path)

    with mock.patch.object(exp_d, "OUT", tmp_path), contextlib.redirect_stdout(io.StringIO()):
        exp_d._write_summary(results)
    text = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    for section in (
        "aggressive rates",
        "knee per method",
        "spectral methods vs tfidf",
        "cost of protect=code_only",
        "supersede ceiling",
        "variants vs the base",
        "path substitution arms",
        "path table cost",
    ):
        assert section in text, section


def test_transforming_methods_lists_every_method_that_shrinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The guard on a hand-written list.

    TRANSFORMING_METHODS decides whose stored results a shrinker rewrite invalidates. Getting it
    wrong is silent — stale records load looking healthy — so the true set is detected by running
    every method and watching who calls the shrinker, rather than trusted to stay in sync.
    """
    segs = _corpus()

    def fake_embed(texts: list[str], model_name: str) -> np.ndarray:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(len(texts), 8))
        return x / np.where((n := np.linalg.norm(x, axis=1, keepdims=True)) > 0, n, 1.0)

    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embed)
    called: set[str] = set()
    real = exp_d.quantize_segment
    current: list[str] = []

    def watched(*args: object, **kwargs: object) -> object:
        called.add(current[0])
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(exp_d, "quantize_segment", watched)
    for name, fn in METHODS.items():
        current[:] = [name]
        fn(segs, 0.5, "none", _ctx(tmp_path, BASE_VARIANT, segs))
    assert called == set(TRANSFORMING_METHODS)


def test_transforms_stamp_is_zero_for_methods_that_never_shrink() -> None:
    assert transforms_stamp("tfidf") == 0
    assert transforms_stamp("band+leverage") == TRANSFORMS_VERSION


def test_a_shrinker_rewrite_invalidates_only_the_records_that_depend_on_it() -> None:
    # Records written before the shrinker was versioned carry no stamp. A non-shrinking method's
    # stamp is 0 too, so its record stays valid; a shrinking method's record is recomputed.
    old_tfidf = {"method": "tfidf", "protect": "none", "keep_ratio": 0.5}
    old_band = {"method": "band+leverage", "protect": "none", "keep_ratio": 0.5}
    planned = {j.identity() for j in plan_jobs()}
    assert _record_identity(old_tfidf) in planned
    assert _record_identity(old_band) not in planned


def test_band_quality_stamp_is_zero_for_methods_that_never_band() -> None:
    assert band_quality_stamp("tfidf") == 0
    assert band_quality_stamp("leverage") == 0
    for name in TRANSFORMING_METHODS:
        assert band_quality_stamp(name) == BAND_QUALITY_VERSION


def test_a_banding_depth_change_invalidates_only_the_banding_methods() -> None:
    # The keep-ratio-to-depth mapping lives in the runner, so changing it moves neither
    # segments_digest nor TRANSFORMS_VERSION: without a stamp of its own, every stored band
    # record would keep loading as valid while describing a depth policy that no longer exists.
    banding = Job("band+leverage", "none", 0.5, BASE_VARIANT)
    other = Job("leverage", "none", 0.5, BASE_VARIANT)
    before_banding, before_other = banding.identity(), other.identity()
    with mock.patch.object(exp_d, "BAND_QUALITY_VERSION", BAND_QUALITY_VERSION + 1):
        assert banding.identity() != before_banding
        assert other.identity() == before_other


def test_the_banding_stamp_is_folded_into_the_stored_records_identity(tmp_path: Path) -> None:
    job = next(j for j in plan_jobs() if j.method == "band+leverage")
    ctx = RunContext(tmp_path, job.variant, PathTable(()))
    rec = exp_d._record(job.method, job.protect, job.ratio, ctx, 100, [], [], None, set(), 0.0)
    assert rec["band_quality_version"] == BAND_QUALITY_VERSION
    assert _record_identity(rec) == job.identity()
    # Absent on records written before the stamp existed, so those are recomputed rather than
    # trusted — the same default that makes the other two stamps' scoping work.
    old = {"method": "band+leverage", "protect": "none", "keep_ratio": 0.5}
    assert _record_identity(old).band_quality_version == 0
    assert _record_identity(old) not in {j.identity() for j in plan_jobs()}


def test_superseding_methods_lists_every_method_that_supersedes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Same guard as the shrinker list: SUPERSEDING_METHODS is hand-written and decides whose
    # records a supersession change invalidates, so the true set is observed, not trusted.
    segs = _corpus()

    def fake_embed(texts: list[str], model_name: str) -> np.ndarray:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(len(texts), 8))
        return x / np.where((n := np.linalg.norm(x, axis=1, keepdims=True)) > 0, n, 1.0)

    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embed)
    called: set[str] = set()
    real = exp_d.apply_supersession
    current: list[str] = []

    def watched(*args: object, **kwargs: object) -> object:
        called.add(current[0])
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(exp_d, "apply_supersession", watched)
    for name, fn in METHODS.items():
        current[:] = [name]
        fn(segs, 0.5, "none", _ctx(tmp_path, BASE_VARIANT, segs))
    assert called == set(SUPERSEDING_METHODS)


def test_supersede_stamp_is_zero_for_methods_that_never_supersede() -> None:
    assert supersede_stamp("leverage") == 0
    assert supersede_stamp("band+leverage") == 0
    for name in SUPERSEDING_METHODS:
        assert supersede_stamp(name) == SUPERSEDE_VERSION


def test_a_supersession_change_invalidates_only_the_superseding_methods() -> None:
    # The pass runs downstream of segmentation, so a changed rule moves no other stamp: without
    # one of its own, a stored supersede+* record would keep loading as valid while describing
    # an elision policy that no longer exists (segments_digest also carries it, for final_atoms).
    stacked = Job("supersede+leverage", "none", 0.5, BASE_VARIANT)
    plain = Job("leverage", "none", 0.5, BASE_VARIANT)
    before_stacked, before_plain = stacked.identity(), plain.identity()
    with mock.patch.object(exp_d, "SUPERSEDE_VERSION", SUPERSEDE_VERSION + 1):
        assert stacked.identity() != before_stacked
        assert plain.identity() == before_plain


def test_the_supersession_stamp_is_folded_into_the_stored_records_identity(
    tmp_path: Path,
) -> None:
    job = next(j for j in plan_jobs() if j.method == "supersede+leverage")
    ctx = RunContext(tmp_path, job.variant, PathTable(()))
    rec = exp_d._record(job.method, job.protect, job.ratio, ctx, 100, [], [], None, set(), 0.0)
    assert rec["supersede_version"] == SUPERSEDE_VERSION
    assert _record_identity(rec) == job.identity()
    # A record written under supersession version 1 (or before the stamp existed) is stale for a
    # supersede+* method and still valid for a plain one.
    planned = {j.identity() for j in plan_jobs()}
    old_stacked = {
        "method": "supersede+leverage",
        "protect": "none",
        "keep_ratio": 0.5,
        "transforms_version": 0,
        "supersede_version": 1,
    }
    old_plain = {"method": "leverage", "protect": "none", "keep_ratio": 0.5}
    assert _record_identity(old_stacked) not in planned
    assert _record_identity(old_plain) in planned


def test_metric_backfills_the_earned_fields_of_a_pre_arm_record() -> None:
    # Exact, not approximate: a record written before any table existed guaranteed nothing, so
    # its earned recall IS its recall. This is what lets the arms land without a schema bump.
    old = {"atom_recall": 0.8, "atom_recall_final": 0.9}
    assert _metric(old, "atom_recall_earned") == pytest.approx(0.8)
    assert _metric(old, "atom_recall_final_earned") == pytest.approx(0.9)
    assert _metric(old, "guaranteed_share") == 0.0
    new = {"atom_recall": 0.9, "atom_recall_earned": 0.5}
    assert _metric(new, "atom_recall_earned") == pytest.approx(0.5)


def _transcript(n: int = 8) -> Transcript:
    return Transcript(
        "test",
        [
            Turn(i, "assistant", [Block("text", f"Paragraph {i} discusses topic {i % 3} here.")])
            for i in range(n)
        ],
    )


@pytest.fixture
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed(texts: list[str], model_name: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash((tuple(texts), model_name))) % (2**32))
        x = rng.normal(size=(len(texts), 8))
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(norms > 0, norms, 1.0)

    monkeypatch.setattr("loosy_goose.embed.embed_texts", fake_embed)
    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embed)


def _fake_grid(monkeypatch: pytest.MonkeyPatch, fn: Any, ratios: tuple[float, ...]) -> None:
    """One synthetic method over `ratios`, so analyse()'s job loop is exercised without paying
    for real selection."""
    monkeypatch.setattr(exp_d, "METHODS", {"fake": fn})
    jobs = [Job("fake", "none", r, BASE_VARIANT) for r in ratios]
    monkeypatch.setattr(exp_d, "plan_jobs", lambda: list(jobs))


def test_a_job_that_raises_is_isolated_and_the_grid_keeps_going(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_embed: None
) -> None:
    # Before isolation a single unhandled exception killed a multi-hour grid AND discarded every
    # completed job for the in-flight transcript.
    seen: list[float] = []

    def flaky(segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext) -> Any:
        seen.append(ratio)
        if ratio == 0.25:
            raise RuntimeError("boom in the middle of the grid")
        return segs[:2]

    _fake_grid(monkeypatch, flaky, (0.5, 0.25, 0.1))
    r = analyse("t", True, _transcript(), tmp_path)

    assert seen == [0.5, 0.25, 0.1]
    assert sorted(rec["keep_ratio"] for rec in r["records"]) == [0.1, 0.5]
    assert len(r["failures"]) == 1
    failure = r["failures"][0]
    assert failure["method"] == "fake"
    assert failure["keep_ratio"] == 0.25
    assert failure["variant_key"] == "base"
    assert "RuntimeError" in failure["error"]
    assert "boom in the middle of the grid" in failure["traceback"]


def test_a_failed_job_is_not_a_record_so_nothing_reads_it_as_a_low_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_embed: None
) -> None:
    # The dangerous shape of this bug is the silent one: a failure stored as a record with zeroed
    # metrics would be averaged into every table, curve and plot as a genuine measurement.
    def flaky(segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext) -> Any:
        if ratio == 0.25:
            raise ValueError("no")
        # Keeps a ratio-dependent slice, so the two survivors land at distinct achieved rates and
        # the curve builder does not dedupe one of them away.
        return segs[: max(1, int(len(segs) * ratio))]

    _fake_grid(monkeypatch, flaky, (0.5, 0.25, 0.1))
    r = analyse("t", True, _transcript(), tmp_path)

    assert ("fake", "none", 0.25) not in exp_d._index_records([r])
    curve = exp_d._curves_by_method([r])[("fake", "none")][0]
    assert len(curve["rates"]) == 2
    assert all(np.isfinite(v) for v in curve["recall_final"])
    identities = {_record_identity(rec) for rec in r["records"]}
    failed = _record_identity({"method": "fake", "protect": "none", "keep_ratio": 0.25})
    assert failed not in identities


def test_an_incremental_checkpoint_survives_a_mid_transcript_abort_and_resumes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stub_embed: None
) -> None:
    # Previously the checkpoint was written only once analyse() returned all 600 jobs, so a kill
    # part-way through a transcript threw away everything it had done (up to ~19 min).
    out_path = tmp_path / "t.json"
    ratios = tuple(round(0.04 * i, 2) for i in range(1, 26))
    budget = {"left": 12}
    ran: list[float] = []

    class Killed(BaseException):
        """Stands in for Ctrl-C: a BaseException, so job isolation must not swallow it."""

    def fn(segs: list[Segment], ratio: float, protect: ProtectPolicy, ctx: RunContext) -> Any:
        if budget["left"] == 0:
            raise Killed
        budget["left"] -= 1
        ran.append(ratio)
        return segs[:2]

    def flush(payload: dict[str, Any]) -> None:
        exp_d._write_checkpoint(out_path, payload)

    _fake_grid(monkeypatch, fn, ratios)
    with pytest.raises(Killed):
        analyse("t", True, _transcript(), tmp_path, checkpoint=flush)

    prior, reason = _load_checkpoint(out_path, force=False)
    assert reason is None
    assert prior is not None
    # The last flush boundary before the kill: not zero, and not everything.
    assert len(prior["records"]) == FLUSH_EVERY

    budget["left"] = 1_000
    resumed = analyse("t", True, _transcript(), tmp_path, prior=prior, checkpoint=flush)

    assert sorted(rec["keep_ratio"] for rec in resumed["records"]) == sorted(ratios)
    # Only the jobs the checkpoint did not already hold were paid for a second time.
    assert ran[12:] == list(ratios[FLUSH_EVERY:])


def test_checkpoint_writes_are_atomic_so_a_kill_leaves_no_truncated_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A truncated checkpoint is not merely lost: _load_checkpoint can only read it as unreadable
    # and discard it, degrading to a silent full re-run of that transcript.
    out_path = tmp_path / "t.json"
    payload = {"schema_version": SCHEMA_VERSION, "label": "t", "records": [{"a": 1}]}
    exp_d._write_checkpoint(out_path, payload)
    intact = out_path.read_text(encoding="utf-8")

    real_write = Path.write_text

    def half_write(self: Path, data: str, *args: Any, **kwargs: Any) -> int:
        real_write(self, data[: len(data) // 2], *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(Path, "write_text", half_write)
    with pytest.raises(KeyboardInterrupt):
        exp_d._write_checkpoint(out_path, {**payload, "records": [{"a": i} for i in range(500)]})
    monkeypatch.undo()

    assert out_path.read_text(encoding="utf-8") == intact
    prior, reason = _load_checkpoint(out_path, force=False)
    assert prior == payload
    assert reason is None


def test_weighted_at_rate_returns_none_instead_of_dividing_by_a_zero_weight() -> None:
    # np.average raises ZeroDivisionError when every weight is zero, and this runs at the end of
    # a multi-hour grid where a throw costs the whole run's reporting.
    zero = [{"label": "z", "weight": 0.0, "rates": [0.2, 0.8], "recall_final": [0.5, 0.9]}]
    assert _weighted_at_rate(zero, 0.5, "recall_final") is None
    nonzero = [{"label": "a", "weight": 2.0, "rates": [0.2, 0.8], "recall_final": [0.5, 0.9]}]
    assert _weighted_at_rate(nonzero, 0.5, "recall_final") == pytest.approx(0.7)


def test_path_arm_prose_matches_what_the_code_actually_charges() -> None:
    # The summary used to state the table was "charged into both the budget and the ratio", the
    # opposite of _record's locked behaviour, directly above the arm C/D deltas. The behaviour
    # itself is pinned by test_every_arm_selects_against_the_same_shared_budget.
    text = " ".join(_path_arm_table([]))
    assert "charged into both the budget" not in text
    assert "NOT deducted" in text
    assert "NOT budget-neutral" in text


def test_trace_commons_sampling_breaks_size_ties_on_name_not_glob_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Equal-size files used to tie-break on glob order, so a directory reorder changed which
    # files were sampled and what they were labelled - orphaning their checkpoints.
    d = tmp_path / "public" / "trace-commons" / "sessions" / "claude_code"
    d.mkdir(parents=True)
    for name in ("aaa00000", "bbb00000", "ccc00000"):
        (d / f"{name}.jsonl").write_text("x" * 10, encoding="utf-8")
    shuffled = [d / "ccc00000.jsonl", d / "aaa00000.jsonl", d / "bbb00000.jsonl"]
    monkeypatch.setattr(exp_d, "DATA", tmp_path)
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(shuffled))
    monkeypatch.setattr(exp_d, "load_claude_code_jsonl", lambda p: Transcript(str(p), []))

    labels = [label for label, _, _ in iter_trace_commons(2)]
    assert labels == ["trace-commons_aaa00000", "trace-commons_ccc00000"]


def test_failure_report_names_every_failed_and_missing_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        exp_d, "plan_jobs", lambda: [Job("fake", "none", r, BASE_VARIANT) for r in (0.5, 0.25)]
    )
    result = {
        "label": "t0",
        "total_tokens": 100,
        "records": [{"method": "fake", "protect": "none", "keep_ratio": 0.5}],
        "failures": [
            {
                "method": "fake",
                "protect": "none",
                "keep_ratio": 0.25,
                "variant_key": "base",
                "error": "RuntimeError: boom",
            }
        ],
    }
    text = "\n".join(failure_report([result]))
    assert "INCOMPLETE RUN" in text
    assert "RuntimeError: boom" in text
    assert "t0" in text


def test_failure_report_is_quiet_when_the_grid_is_complete(tmp_path: Path) -> None:
    text = "\n".join(failure_report(_full_grid_results(tmp_path, n_transcripts=1)))
    assert "INCOMPLETE RUN" not in text
    assert "no failures" in text


def test_write_summary_puts_failures_above_the_tables(tmp_path: Path) -> None:
    # A run that silently completes with missing cells is the worst outcome, so this cannot be a
    # footnote: it has to be visible before the first number anyone reads.
    results = _full_grid_results(tmp_path, n_transcripts=1)
    results[0]["records"] = results[0]["records"][:-1]
    results[0]["failures"] = [
        {
            "method": "leverage",
            "protect": "none",
            "keep_ratio": 0.9,
            "variant_key": "base",
            "error": "MemoryError: out of memory",
        }
    ]
    with mock.patch.object(exp_d, "OUT", tmp_path), contextlib.redirect_stdout(io.StringIO()):
        exp_d._write_summary(results)
    text = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "INCOMPLETE RUN" in text
    assert "MemoryError: out of memory" in text
    assert text.index("INCOMPLETE RUN") < text.index("aggressive rates")

import numpy as np
import pytest

from loosy_goose.budget import forced_mask, select_by_score, token_budget
from loosy_goose.metrics import (
    atom_recall,
    atom_recall_by_kind,
    compression_ratio,
    evaluate,
    semantic_coverage,
)
from loosy_goose.segment import Segment, SegmentKind, extract_atoms
from loosy_goose.tokens import count_tokens


def _seg(i: int, text: str, kind: SegmentKind = "prose", protected: bool = False) -> Segment:
    return Segment(i, i, "assistant", kind, text, protected, extract_atoms(text))


def _corpus() -> list[Segment]:
    return [
        _seg(0, "We should refactor the parser before adding new features to it."),
        _seg(1, "def parse(x):\n    return load_config(x)", kind="code", protected=True),
        _seg(2, "The build failed 42 times because of config.toml being stale."),
        _seg(3, '{"tool": "Read", "path": "src/main.py"}', kind="tool_use", protected=True),
        _seg(4, "Finally the tests passed and we shipped it."),
        _seg(5, "One more note about the release process for next week."),
    ]


def _tokens(segs: list[Segment]) -> int:
    return sum(count_tokens(s.text) for s in segs)


def test_token_budget_ceil_and_bounds() -> None:
    segs = _corpus()
    assert token_budget(segs, 1.0) == _tokens(segs)
    assert token_budget(segs, 0.001) == 1
    with pytest.raises(ValueError):
        token_budget(segs, 0.0)
    with pytest.raises(ValueError):
        token_budget(segs, 1.5)


def test_forced_mask_policies() -> None:
    segs = _corpus()
    assert forced_mask(segs, "all") == [False, True, False, True, False, False]
    assert forced_mask(segs, "code_only") == [False, True, False, False, False, False]
    assert forced_mask(segs, "none") == [False] * 6


def test_select_keeps_forced_and_respects_budget_and_order() -> None:
    segs = _corpus()
    scores = np.array([0.9, 0.0, 0.8, 0.0, 0.7, 0.6])
    kept = select_by_score(segs, scores, 0.6, "all")
    assert segs[1] in kept and segs[3] in kept
    assert _tokens(kept) <= token_budget(segs, 0.6)
    assert [s.id for s in kept] == sorted(s.id for s in kept)
    assert all(any(k is s for s in segs) for k in kept)


def test_select_forced_exceeding_budget_returns_only_forced() -> None:
    segs = _corpus()
    kept = select_by_score(segs, np.ones(6), 0.05, "all")
    assert [s.id for s in kept] == [1, 3]
    assert _tokens(kept) > token_budget(segs, 0.05)


def test_select_greedy_prefers_highest_score() -> None:
    segs = _corpus()
    scores = np.array([0.1, 0.0, 0.2, 0.0, 0.9, 0.3])
    kept = select_by_score(segs, scores, 0.25, "none")
    assert kept[0] is segs[4]


def test_recency_shifts_choice_toward_late_segments() -> None:
    segs = _corpus()
    # Pure score favours the first segment; full recency must favour the last one instead.
    scores = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    early = select_by_score(segs, scores, 0.2, "none", recency=0.0)
    late = select_by_score(segs, scores, 0.2, "none", recency=1.0)
    assert segs[0] in early and segs[0] not in late
    assert segs[5] in late


def test_select_rejects_bad_inputs() -> None:
    segs = _corpus()
    with pytest.raises(ValueError):
        select_by_score(segs, np.ones(3), 0.5)
    with pytest.raises(ValueError):
        select_by_score(segs, np.ones(6), 0.5, recency=2.0)
    assert select_by_score([], np.zeros(0), 0.5) == []


def test_compression_ratio() -> None:
    segs = _corpus()
    assert compression_ratio(segs, segs) == pytest.approx(1.0)
    assert compression_ratio(segs, []) == 0.0
    assert compression_ratio([], []) == 1.0
    half = segs[:3]
    assert compression_ratio(segs, half) == pytest.approx(_tokens(half) / _tokens(segs))


def test_atom_recall_hand_built() -> None:
    segs = _corpus()
    assert atom_recall(segs, segs) == 1.0
    assert atom_recall(segs, []) == 0.0
    kept = [segs[2]]
    all_atoms = {a for s in segs for a in s.atoms}
    assert atom_recall(segs, kept) == pytest.approx(len(set(segs[2].atoms)) / len(all_atoms))
    by_kind = atom_recall_by_kind(segs, kept)
    assert by_kind["prose"] > 0.0
    assert by_kind["code"] == 0.0
    assert by_kind["tool_use"] == 0.0
    # A kind with no atoms at all is trivially fully recalled.
    assert atom_recall_by_kind([_seg(0, "plain words only")], []) == {"prose": 1.0}


def test_semantic_coverage_degenerate_cases_need_no_model() -> None:
    segs = _corpus()
    empty_kept = semantic_coverage(segs, [])
    assert empty_kept["coverage"] == 0.0 and empty_kept["doc_cosine"] == 0.0
    assert set(empty_kept["coverage_by_kind"]) == {"prose", "code", "tool_use"}
    assert semantic_coverage([], [])["coverage"] == 1.0


def test_semantic_coverage_uses_precomputed_vectors() -> None:
    segs = [_seg(0, "a"), _seg(1, "b"), _seg(2, "c")]
    vecs = np.eye(3)
    full = semantic_coverage(segs, segs, original_vectors=vecs)
    assert full["coverage"] == pytest.approx(1.0)
    assert full["doc_cosine"] == pytest.approx(1.0)
    partial = semantic_coverage(segs, [segs[0]], original_vectors=vecs)
    assert partial["coverage"] == pytest.approx(1 / 3)
    assert partial["doc_cosine"] == pytest.approx(1 / np.sqrt(3))
    with pytest.raises(ValueError):
        semantic_coverage(segs, segs, original_vectors=np.eye(2))


@pytest.mark.slow
def test_semantic_coverage_with_real_model() -> None:
    pytest.importorskip("sentence_transformers")
    cat = _seg(0, "The cat sat on the mat.")
    kitten = _seg(1, "A kitten rested on the rug.")
    finance = _seg(2, "Quarterly revenue exceeded the forecast by twelve percent.")
    segs = [cat, kitten, finance]

    same = semantic_coverage(segs, segs)
    assert same["coverage"] == pytest.approx(1.0, abs=1e-6)
    assert same["doc_cosine"] == pytest.approx(1.0, abs=1e-6)

    result = evaluate(segs, [cat])
    assert 0.0 < result["coverage"] < 1.0
    assert set(result) >= {"compression_ratio", "atom_recall", "coverage", "doc_cosine"}
    # The paraphrase must be covered far better than the unrelated finance sentence.
    assert (
        semantic_coverage([kitten], [cat])["coverage"]
        > semantic_coverage([finance], [cat])["coverage"]
    )

import numpy as np
import pytest

from loosy_goose.budget import forced_mask, select_by_score, token_budget
from loosy_goose.metrics import (
    atom_recall,
    atom_recall_by_kind,
    atom_recall_earned,
    atom_recall_final,
    compression_ratio,
    evaluate,
    final_state_atoms,
    guaranteed_share,
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


def test_atom_recall_final_matches_atom_recall_when_nothing_superseded() -> None:
    segs = _corpus()  # no repeated read/write paths, nothing for apply_supersession to drop
    assert final_state_atoms(segs) == {a for s in segs for a in s.atoms}
    for kept in ([segs[0]], segs[:3], segs, []):
        assert atom_recall_final(segs, kept) == pytest.approx(atom_recall(segs, kept))


def test_atom_recall_final_only_counts_the_last_read() -> None:
    # Three reads of the same path with a distinguishing offset atom each; only the last read
    # survives supersession, so only its atoms belong in the final-state denominator.
    reads = [
        _seg(i, f'{{"file_path": "config.py", "offset": {off}}}', kind="tool_use", protected=True)
        for i, off in enumerate((10, 20, 30))
    ]
    final = final_state_atoms(reads)
    # `file_path` is a payload key, not a fact about the session, so it is not an atom; see
    # segment.scannable.
    assert final == set(reads[-1].atoms) == {"config.py", "30"}

    kept = [reads[-1]]
    # Historical recall: only the last read's atoms are present, out of every read's atoms.
    assert atom_recall(reads, kept) == pytest.approx(2 / 4)
    # Final-state recall: kept IS the segment whose atoms define the denominator, so it's total.
    assert atom_recall_final(reads, kept) == pytest.approx(1.0)
    assert atom_recall_final(reads, []) == pytest.approx(0.0)
    # A precomputed final_atoms set must be usable directly, without recomputing supersession.
    assert atom_recall_final(reads, kept, final_atoms=final) == pytest.approx(1.0)


def test_evaluate_reports_both_atom_recall_metrics() -> None:
    segs = _corpus()
    result = evaluate(segs, [segs[2]])
    assert "atom_recall" in result and "atom_recall_final" in result


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


def _guarantee_corpus() -> tuple[list[Segment], list[Segment]]:
    segs = [
        _seg(0, "edited src/alpha.py for the parser"),
        _seg(1, "then touched src/beta.py and run_job"),
    ]
    return segs, [segs[0]]


def test_guaranteed_atoms_count_as_recovered() -> None:
    segs, kept = _guarantee_corpus()
    assert "src/beta.py" not in set(kept[0].atoms)
    plain = atom_recall(segs, kept)
    with_table = atom_recall(segs, kept, guaranteed={"src/beta.py"})
    assert with_table > plain


def test_earned_recall_ignores_what_the_table_hands_over() -> None:
    segs, kept = _guarantee_corpus()
    guaranteed = {"src/alpha.py", "src/beta.py"}
    # Both paths leave the denominator, so the score reflects only the non-path atoms, and a
    # table cannot lift it the way it lifts atom_recall.
    assert atom_recall_earned(segs, kept, guaranteed) < atom_recall(segs, kept, guaranteed)
    assert atom_recall_earned(segs, kept, None) == pytest.approx(atom_recall(segs, kept))


def test_earned_recall_gives_no_credit_for_a_guaranteed_atom_in_a_dropped_segment() -> None:
    segs, kept = _guarantee_corpus()
    only_dropped = atom_recall_earned(segs, kept, guaranteed={"src/beta.py"})
    without = atom_recall_earned(segs, kept, guaranteed=set())
    assert only_dropped > without  # the atom left the denominator, not the numerator


def test_guaranteed_share_reports_how_much_was_free() -> None:
    segs, _ = _guarantee_corpus()
    wanted = {a for s in segs for a in s.atoms}
    assert guaranteed_share(segs, None) == 0.0
    assert guaranteed_share(segs, {"src/alpha.py"}) == pytest.approx(1 / len(wanted))
    # An atom the transcript never mentions cannot inflate the share.
    assert guaranteed_share(segs, {"nowhere/at/all.py"}) == 0.0


def test_side_table_tokens_are_charged_against_the_output() -> None:
    segs, kept = _guarantee_corpus()
    bare = compression_ratio(segs, kept)
    charged = compression_ratio(segs, kept, extra_tokens=50)
    assert charged > bare
    total = sum(count_tokens(s.text) for s in segs)
    assert charged == pytest.approx(bare + 50 / total)


def test_evaluate_reports_both_recall_columns() -> None:
    segs, kept = _guarantee_corpus()
    out = evaluate(segs, kept, guaranteed={"src/beta.py"}, extra_tokens=7)
    assert out["atom_recall"] > out["atom_recall_earned"]
    assert out["guaranteed_share"] > 0.0
    assert out["compression_ratio"] > compression_ratio(segs, kept)

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.baselines import METHODS, random_drop, recency_only, tfidf  # noqa: E402
from loosy_goose.budget import ProtectPolicy, token_budget  # noqa: E402
from loosy_goose.segment import Segment, SegmentKind, extract_atoms  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402


def _seg(i: int, text: str, kind: SegmentKind = "prose", protected: bool = False) -> Segment:
    return Segment(i, i, "assistant", kind, text, protected, extract_atoms(text))


def _corpus() -> list[Segment]:
    segs = [_seg(i, f"Paragraph {i} talks about topic {i % 3} in some detail.") for i in range(8)]
    segs.append(_seg(8, "def f():\n    return 1", kind="code", protected=True))
    segs.append(_seg(9, '{"tool": "Bash", "cmd": "ls"}', kind="tool_use", protected=True))
    segs.append(_seg(10, "Closing remarks about topic 1 and the plan for topic 2."))
    return segs


@pytest.mark.parametrize("name", sorted(METHODS))
@pytest.mark.parametrize("protect", ["all", "code_only", "none"])
def test_baselines_respect_contract(name: str, protect: ProtectPolicy) -> None:
    segs = _corpus()
    kept = METHODS[name](segs, 0.5, protect=protect, recency=0.0, seed=1)
    assert sum(count_tokens(s.text) for s in kept) <= token_budget(segs, 0.5)
    assert [s.id for s in kept] == sorted(s.id for s in kept)
    assert all(any(k is s for s in segs) for k in kept)
    forced = {"all": {8, 9}, "code_only": {8}, "none": set()}[protect]
    assert forced <= {s.id for s in kept}


def test_random_drop_is_seeded() -> None:
    segs = _corpus()
    a = random_drop(segs, 0.5, protect="none", seed=3)
    b = random_drop(segs, 0.5, protect="none", seed=3)
    assert [s.id for s in a] == [s.id for s in b]


def test_recency_only_keeps_tail() -> None:
    segs = _corpus()
    ids = [s.id for s in recency_only(segs, 0.3, protect="none")]
    assert ids and ids == list(range(11 - len(ids), 11))


def test_tfidf_prefers_distinctive_segment() -> None:
    filler = [_seg(i, "the same common words again and again") for i in range(5)]
    rare = _seg(5, "zeppelin quartz oxbow")
    kept = tfidf([*filler, rare], 0.15, protect="none")
    assert rare in kept


def test_tfidf_handles_empty_vocabulary() -> None:
    segs = [_seg(0, "!!!"), _seg(1, "???")]
    assert tfidf(segs, 1.0, protect="none") == segs
    assert tfidf([], 0.5) == []

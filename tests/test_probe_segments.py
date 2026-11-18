import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_i_downstream as exi  # noqa: E402
from experiments.continuous import probe_segments as ps  # noqa: E402
from loosy_goose.transcript import Block, Turn  # noqa: E402


def words(text: str) -> int:
    return len(text.split())


def _turns(sizes: list[int]) -> list[Turn]:
    return [
        Turn(
            i,
            "user" if i % 2 else "assistant",
            [Block("text", " ".join(f"w{i}x{j}" for j in range(n)))],
        )
        for i, n in enumerate(sizes)
    ]


@pytest.mark.parametrize(
    "sizes", [[12000], [100, 100], [5000, 3000, 2000, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [7, 0, 3]]
)
def test_allocation_sums_to_n_with_at_least_one(sizes):
    got = ps.allocate(sizes, exi.N_PROBES)
    assert sum(got) == exi.N_PROBES
    assert len(got) == len(sizes) and min(got) >= 1


def test_allocation_follows_token_share():
    assert ps.allocate([9000, 1000], 10) == [9, 1]
    assert ps.allocate([6000, 6000, 6000], 10) == [4, 3, 3]
    assert ps.allocate([12000], 10) == [10]


def test_allocation_refuses_more_segments_than_probes():
    with pytest.raises(ValueError):
        ps.allocate([1] * 11, 10)


def test_segments_never_exceed_cap_and_keep_order():
    turns = _turns([300, 450, 120, 800, 50, 2500, 10, 600])
    segs = ps.segment_history(turns, words, 1000)
    assert len(segs) > 1
    assert all(words(s) <= 1000 for s in segs)
    flat = " ".join(segs).split()
    assert flat == exi.render_turns(turns).split()


def test_segments_split_at_turn_boundaries_when_turns_fit():
    turns = _turns([400, 400, 400, 400])
    segs = ps.segment_history(turns, words, 1000)
    rendered = [exi.render_turns([t]) for t in turns]
    assert segs == ["\n\n".join(rendered[:2]), "\n\n".join(rendered[2:])]


def test_short_history_is_one_segment_identical_to_unchunked_text():
    turns = _turns([100, 200])
    assert ps.segment_history(turns, words, 12000) == [exi.render_turns(turns)]


def test_segment_jobs_deterministic_and_assembly_in_order():
    turns = _turns([400, 400, 400, 400, 900])
    a = ps.segment_jobs("t@50", turns, words, 1000)
    b = ps.segment_jobs("t@50", turns, words, 1000)
    assert a == b
    assert sum(n for _, _, n in a) == exi.N_PROBES
    assert [s for s, _, _ in a] == [f"t@50#s{i}" for i in range(len(a))]
    texts = {
        s: json.dumps({"probes": [{"q": f"{s} q{i}", "a": f"a{i}"} for i in range(n)]})
        for s, _, n in a
    }
    got = ps.assemble([(s, n) for s, _, n in a], texts.get)
    assert got is not None and len(got) == exi.N_PROBES
    assert [p["q"].split()[0] for p in got] == [s for s, _, n in a for _ in range(n)]
    missing = dict(texts)
    missing.pop(a[-1][0])
    assert ps.assemble([(s, n) for s, _, n in a], missing.get) is None


def test_segment_probe_parse_keeps_first_n_and_needs_share():
    def text(k: int) -> str:
        return json.dumps({"probes": [{"q": f"q{i}", "a": f"a{i}"} for i in range(k)]})

    assert ps.parse_segment_probes(text(5), 3) == [{"q": f"q{i}", "a": f"a{i}"} for i in range(3)]
    assert ps.parse_segment_probes(text(1), 1) is not None
    assert ps.parse_segment_probes(text(0), 1) is None
    assert ps.parse_segment_probes(text(2), 10) is None
    assert ps.parse_segment_probes(text(3), 10) is not None
    assert ps.parse_segment_probes("not json", 2) is None

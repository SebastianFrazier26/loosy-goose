import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_i_downstream as exi  # noqa: E402
from experiments.continuous import sim  # noqa: E402
from loosy_goose.segment import Segment, segment  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Block, Transcript, Turn  # noqa: E402


def _words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def _turns(rounds: int, size: int = 300, double: bool = False) -> list[Turn]:
    turns = [Turn(0, "system", [Block("text", "system prompt")])]
    turns.append(Turn(1, "user", [Block("text", _words(size, "ask"))]))
    for r in range(rounds):
        if double:
            turns.append(
                Turn(len(turns), "assistant", [Block("text", _words(size // 3, f"p{r}x"))])
            )
        use = json.dumps({"file_path": f"src/f{r}.py"})
        turns.append(Turn(len(turns), "assistant", [Block("tool_use", use, {"name": "Read"})]))
        turns.append(Turn(len(turns), "user", [Block("tool_result", _words(size, f"r{r}x"))]))
    return turns


def keep_every_third(t: Transcript, keep: float) -> list[Segment]:
    return segment(t)[::3]


def stub_summary(k: int, text: str, budget: int) -> str:
    return sim.placeholder_summary(k, text, budget)


def _plan(rounds: int = 30, double: bool = False) -> sim.TranscriptPlan:
    return sim.plan_transcript("t", _turns(rounds, double=double))


def test_freeze_never_mutates_prefix_and_chunks_at_f() -> None:
    p = _plan()
    f = sim.Freezer(p.turns, p.per_turn, keep_every_third, chunk_tokens=1_500)
    seen: list[list[str]] = []
    for t in p.boundaries:
        f.advance(exi.tail_start(p.per_turn, t))
        seen.append(list(f.frozen))
    assert len(f.frozen) >= 3
    for a, b in zip(seen, seen[1:], strict=False):
        assert b[: len(a)] == a
    for lo, hi in f.chunks:
        assert sum(p.per_turn[lo:hi]) >= 1_500
        assert sum(p.per_turn[lo : hi - 1]) < 1_500
    assert [lo for lo, _ in f.chunks[1:]] == [hi for _, hi in f.chunks[:-1]]


def test_freeze_text_in_cfrz_context_keeps_prefix() -> None:
    p = _plan()
    run_a = sim.run_arm("cfrz", p, keep_every_third, keep_at=[p.checkpoints[0].t])
    run_b = sim.run_arm("cfrz", p, keep_every_third)
    assert run_a.contexts[p.checkpoints[0].t] == run_b.contexts[p.checkpoints[0].t]
    assert run_b.uncached_tokens[-1] < run_b.ctx_tokens[-1]


def test_recompute_equals_loosy_over_pretail() -> None:
    p = _plan()
    ck = p.checkpoints[1]
    run = sim.run_arm("crec", p, keep_every_third)
    pre = Transcript("x", p.turns[: ck.tail_from])
    expected = exi.assemble(
        "loosy",
        exi.render_segments(keep_every_third(pre, sim.KEEP)),
        exi.render_turns(p.turns[ck.tail_from : ck.t]),
    )
    assert run.contexts[ck.t] == expected


def test_thresh_fires_over_cap_and_recompacts_summary() -> None:
    p = _plan(40)
    inputs: list[str] = []

    def summarize(k: int, text: str, budget: int) -> str:
        inputs.append(text)
        return f"SUMMARY{k} " + "word " * (budget - 3)

    run = sim.run_arm("thresh", p, keep_every_third, summarize)
    assert len(run.events) >= 2
    assert "SUMMARY" not in inputs[0]
    assert "SUMMARY0 " in inputs[1] and "ask" not in inputs[1].split("[Session since")[0]
    first = run.events[0]
    i = run.boundaries.index(first.t)
    assert max(run.ctx_tokens[:i]) <= p.cap + 100
    assert run.ctx_tokens[i] < run.ctx_tokens[i - 1]
    last_ck = p.checkpoints[-1]
    assert "SUMMARY" in run.contexts[last_ck.t]


def test_thresh_input_is_summary_plus_new_text_only() -> None:
    p = _plan(40)
    inputs: list[str] = []

    def summarize(k: int, text: str, budget: int) -> str:
        inputs.append(text)
        return f"SUMMARY{k} " + "word " * (budget - 3)

    run = sim.run_arm("thresh", p, keep_every_third, summarize)
    e0, e1 = run.events[0], run.events[1]
    assert inputs[1].endswith(exi.render_turns(p.turns[e0.tail_from : e1.tail_from]))


def test_checkpoints_snap_to_action_end() -> None:
    p = _plan(30, double=True)
    n = len(p.turns)
    assert [c.pct for c in p.checkpoints] == [50, 75, 100]
    assert p.checkpoints[-1].t == n
    for share, c in zip(sim.CHECKPOINT_SHARES, p.checkpoints, strict=True):
        assert p.before[c.t] >= share * p.total
        if c.t < n:
            assert p.turns[c.t - 1].role != "assistant"
        assert c.t in p.boundaries
        assert c.tail_from == exi.tail_start(p.per_turn, c.t)
    # roles: 0 system, 1 user, then (a text, a tool_use, u result) triples from index 2
    assert sim._snap(p.turns, 3) == 5
    assert sim._snap(p.turns, 4) == 5
    assert sim._snap(p.turns, 5) == 5
    assert sim._snap(p.turns, n) == n


def test_uncached_prefix_metric() -> None:
    assert sim.lcp_len("abcdef", "abcxyz") == 3
    assert sim.uncached_tokens("", "hello world") == count_tokens("hello world")
    base = _words(200, "a")
    assert sim.uncached_tokens(base, base + " " + _words(20, "b")) == count_tokens(
        (base + " " + _words(20, "b"))[len(base) :]
    )
    assert sim.uncached_tokens(base, "zzz" + base) >= count_tokens(base)
    p = _plan()
    full = sim.run_arm("full", p, keep_every_third)
    crec = sim.run_arm("crec", p, keep_every_third)
    assert full.uncached_tokens[0] == full.ctx_tokens[0]
    assert sum(full.uncached_tokens[1:]) < sum(crec.uncached_tokens[1:])


def test_set_e_disjoint_and_qualified() -> None:
    cands = [
        exi.Candidate(f"swe-gym_row{i}", "swe-gym", Transcript(f"s{i}", _turns(40 + i)))
        for i in range(25)
    ]
    cands.append(exi.Candidate("trace-commons_ab", "trace-commons", Transcript("tc", _turns(40))))
    cands.append(exi.Candidate("swe-gym_short", "swe-gym", Transcript("short", _turns(3))))
    taken = {f"swe-gym_row{i}" for i in range(0, 25, 2)}
    picked = sim.select_set_e(cands, taken, token_range=(1_000, 10**7), min_compressible=500)
    labels = [c.label for c, _ in picked]
    assert len(labels) == sim.E_N
    assert not set(labels) & taken
    assert "trace-commons_ab" not in labels and "swe-gym_short" not in labels
    assert not sim.qualifies(_turns(3))
    assert sim.qualifies(_turns(40))


def test_placeholder_has_target_length() -> None:
    text = sim.placeholder_summary(2, "", 500)
    assert abs(count_tokens(text) - 500) <= 12

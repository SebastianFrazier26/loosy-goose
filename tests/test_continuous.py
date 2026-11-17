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


def _kept_texts(chunks: list[sim.DecChunk]) -> set[str]:
    return {s.text for ch in chunks for s in ch.segments}


def _decayer(m: int = 2) -> tuple[sim.TranscriptPlan, sim.Decayer]:
    p = _plan(60)
    return p, sim.Decayer(p.turns, p.per_turn, keep_every_third, keep=0.5, m=m, chunk_tokens=900)


def test_dec_merge_triggers_at_m_and_cascades() -> None:
    p, d = _decayer(m=2)
    f = sim.Freezer(p.turns, p.per_turn, keep_every_third, chunk_tokens=900)
    for t in p.boundaries:
        tail_from = exi.tail_start(p.per_turn, t)
        d.advance(tail_from)
        f.advance(tail_from)
        assert all(len(tier) < 2 for tier in d.tiers)
    assert d.merges[0][0] == 0 and max(tier for tier, _, _ in d.merges) >= 1
    assert sum(len(tier) * 2**i for i, tier in enumerate(d.tiers)) == len(f.chunks)
    assert d.upto == f.upto


def test_dec_merged_chunk_is_subset_of_inputs_kept_segments() -> None:
    p, d = _decayer(m=3)
    for t in p.boundaries:
        d.advance(exi.tail_start(p.per_turn, t))
    assert d.merges
    for _tier, inputs, merged in d.merges:
        assert len(inputs) == 3
        assert merged.span == (inputs[0].span[0], inputs[-1].span[1])
        assert merged.segments
        kept_before = {s.text for c in inputs for s in c.segments}
        assert {s.text for s in merged.segments} <= kept_before
        assert sum(count_tokens(s.text) for s in merged.segments) < sum(
            count_tokens(s.text) for c in inputs for s in c.segments
        )


def test_dec_tier0_merge_leaves_older_tiers_unchanged_and_layout_oldest_first() -> None:
    p, d = _decayer(m=2)
    seen_high = False
    for t in p.boundaries:
        old = [list(tier) for tier in d.tiers]
        d.advance(exi.tail_start(p.per_turn, t))
        for i in range(1, len(old)):
            if len(old[i]) <= len(d.tiers[i]):
                assert d.tiers[i][: len(old[i])] == old[i]
        chunks = [c for tier in reversed(d.tiers) for c in tier]
        spans = [c.span for c in chunks]
        assert spans == sorted(spans)
        assert [lo for lo, _ in spans[1:]] == [hi for _, hi in spans[:-1]]
        assert d.text() == "\n\n".join(c.text for c in chunks)
        seen_high |= len(d.tiers) > 2
    assert seen_high


def test_dec_uncached_on_merge_rewrites_only_suffix() -> None:
    p, d = _decayer(m=2)
    prev = ""
    checked = False
    for t in p.boundaries:
        n_merges = len(d.merges)
        old_tiers = len(d.tiers)
        d.advance(exi.tail_start(p.per_turn, t))
        view = sim.View(sim.HEADERS["cdec"], d.text(), d.upto, exi.tail_start(p.per_turn, t))
        ctx = sim.cost_context(view, p.turns, t)
        if len(d.merges) > n_merges and prev and len(d.tiers) == old_tiers and len(d.tiers) > 1:
            deepest_old = d.tiers[-1]
            assert deepest_old
            unc = sim.uncached_tokens(prev, ctx)
            assert unc < count_tokens(ctx)
            assert sim.lcp_len(prev, ctx) >= len(sim.HEADERS["cdec"]) + len(d.tiers[-1][0].text)
            checked = True
        prev = ctx
    assert checked
    run = sim.run_arm("cdec", p, keep_every_third, dec=(0.5, 2))
    assert sum(run.uncached_tokens[1:]) < sum(run.ctx_tokens[1:])


def test_cdec_run_is_cacheable_json_and_distinct_from_cfrz() -> None:
    p = _plan(60)
    run = sim.run_arm("cdec", p, keep_every_third, dec=(0.5, 2))
    again = sim.ArmRun.from_json(run.to_json())
    assert again.contexts == run.contexts and again.ctx_tokens == run.ctx_tokens
    assert run.contexts[p.checkpoints[-1].t].startswith(sim.HEADERS["cdec"])


def test_set_f_disjoint_from_earlier_sets() -> None:
    cands = [
        exi.Candidate(f"swe-gym_row{i}", "swe-gym", Transcript(f"s{i}", _turns(40 + i)))
        for i in range(60)
    ]
    earlier = {f"swe-gym_row{i}" for i in range(0, 30)}
    # Uniform synthetic turns cannot fire thresh twice at C = 65%, so the event minimum drops to 1.
    picked = sim.select_set(
        cands,
        earlier,
        n=sim.F_N,
        seed=sim.F_SEED,
        cap_share=sim.F_CAP_SHARE,
        min_events=1,
        token_range=(1_000, 10**7),
        min_compressible=500,
    )
    labels = [c.label for c, _ in picked]
    assert len(labels) == sim.F_N
    assert not set(labels) & earlier
    assert sim.F_SEED == exi.SEED + 5 and sim.F_CAP_SHARE == 0.65
    assert (
        sim.plan_transcript("x", _turns(40), sim.F_CAP_SHARE).cap
        > sim.plan_transcript("x", _turns(40)).cap
    )


def budget_keeper(t: Transcript, keep: float) -> list[Segment]:
    """Keeps every other segment, then trims from the end to `keep` of the input tokens."""
    segs = segment(t)[::2]
    budget = keep * sum(count_tokens(s.text) for s in segment(t))
    out, used = [], 0
    for s in segs:
        n = count_tokens(s.text)
        if used + n > budget:
            break
        out.append(s)
        used += n
    return out


def test_tloosy_fires_at_cap_and_matches_thresh_schedule() -> None:
    p = _plan(40)
    run = sim.run_arm("tloosy", p, budget_keeper, tloosy_share=0.33)
    assert len(run.events) >= 2
    i = run.boundaries.index(run.events[0].t)
    assert max(run.ctx_tokens[:i]) <= p.cap + 100
    assert run.ctx_tokens[i] < run.ctx_tokens[i - 1]
    assert run.contexts[p.checkpoints[-1].t].startswith(sim.HEADERS["tloosy"])
    full = sim.run_arm("full", p, budget_keeper)
    first = run.boundaries.index(run.events[0].t)
    assert run.ctx_tokens[:first] == full.ctx_tokens[:first]


def test_tloosy_recompresses_compacted_text_within_target() -> None:
    p = _plan(60)
    calls: list[tuple[set[str], float, list[Segment]]] = []

    def spy(t: Transcript, keep: float) -> list[Segment]:
        out = budget_keeper(t, keep)
        calls.append(({s.text for s in segment(t)}, keep, out))
        return out

    share = 0.33
    run = sim.run_arm("tloosy", p, spy, tloosy_share=share)
    target = int(p.cap * share)
    assert len(calls) == len(run.events) >= 2
    for inp, _, out in calls:
        assert {s.text for s in out} <= inp
        assert sum(count_tokens(s.text) for s in out) <= target
    first_in, _, first_out = calls[0]
    second_in = calls[1][0]
    kept = {s.text for s in first_out}
    assert kept <= second_in
    assert not (first_in - kept) & second_in


def test_tloosy_cacheable_json_and_distinct_header() -> None:
    p = _plan(40)
    run = sim.run_arm("tloosy", p, budget_keeper)
    again = sim.ArmRun.from_json(run.to_json())
    assert again.contexts == run.contexts and again.events == run.events
    assert "tloosy" in sim.F_ARMS and "tloosy" not in sim.ARMS

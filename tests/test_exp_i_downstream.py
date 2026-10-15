import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_i_downstream as exp_i  # noqa: E402
from experiments import exp_i_status  # noqa: E402
from loosy_goose.segment import Segment  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Block, Transcript, Turn  # noqa: E402


def _words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i % 50}" for i in range(n))


def _session(n_rounds: int, tokens_per_turn: int = 400, label: str = "t") -> Transcript:
    turns: list[Turn] = [Turn(0, "user", [Block("text", _words(tokens_per_turn, "ask"))])]
    for r in range(n_rounds):
        i = len(turns)
        turns.append(Turn(i, "assistant", [Block("text", f"step {r} " + _words(20, "plan"))]))
        turns.append(
            Turn(
                i + 1,
                "assistant",
                [Block("tool_use", json.dumps({"file_path": f"src/f{r}.py"}), {"name": "Read"})],
            )
        )
        turns.append(Turn(i + 2, "user", [Block("tool_result", _words(tokens_per_turn, "out"))]))
    return Transcript(label, turns)


def _keep_every_other(t: Transcript, keep: float) -> list[Segment]:
    from loosy_goose.segment import segment

    segs = segment(t)
    stride = max(1, round(1 / keep))
    return [s for s in segs if s.id % stride == 0]


# --------------------------------------------------------------------------- selection and cut


def test_find_cut_starts_an_assistant_action_in_the_last_third() -> None:
    t = _session(60)
    cut = exp_i.find_cut(t, random.Random(0), tail_tokens=1000, min_compressible=2000)
    assert cut is not None
    turns = t.turns
    assert turns[cut.cut].role == "assistant" and turns[cut.cut - 1].role != "assistant"
    assert all(x.role == "assistant" for x in turns[cut.cut : cut.end])
    assert cut.end == len(turns) or turns[cut.end].role != "assistant"
    per_turn = [sum(count_tokens(b.text) for b in x.blocks) for x in turns]
    before = sum(per_turn[: cut.cut])
    assert exp_i.CUT_WINDOW[0] <= before / sum(per_turn) <= exp_i.CUT_WINDOW[1]
    assert cut.tail_tokens >= 1000
    assert sum(per_turn[cut.tail_from : cut.cut]) == cut.tail_tokens
    assert sum(per_turn[: cut.tail_from]) == cut.compressible_tokens


def test_find_cut_is_deterministic_per_label_and_refuses_short_sessions() -> None:
    t = _session(60)
    a = exp_i.find_cut(t, exp_i.cut_rng("x"), tail_tokens=1000, min_compressible=2000)
    b = exp_i.find_cut(t, exp_i.cut_rng("x"), tail_tokens=1000, min_compressible=2000)
    assert a == b
    assert exp_i.find_cut(_session(3), random.Random(0)) is None


def test_tail_start_takes_whole_turns_until_the_tail_budget_is_met() -> None:
    assert exp_i.tail_start([100, 100, 100, 100], 4, 150) == 2
    assert exp_i.tail_start([100, 100, 100, 100], 4, 1000) == 0


def test_select_transcripts_filters_range_balances_sources_and_interleaves() -> None:
    cands = [exp_i.Candidate(f"a{i}", "a", _session(40, label=f"a{i}")) for i in range(5)]
    cands += [exp_i.Candidate(f"b{i}", "b", _session(40, label=f"b{i}")) for i in range(2)]
    cands.append(exp_i.Candidate("tiny", "a", _session(2)))
    kw = {"token_range": (5_000, 60_000), "tail_tokens": 1000, "min_compressible": 2000}
    picked = exp_i.select_transcripts(cands, n=5, per_source=3, **kw)  # type: ignore[arg-type]
    labels = [c.label for c, _ in picked]
    assert "tiny" not in labels
    assert len(labels) == 5
    assert sum(lbl.startswith("b") for lbl in labels) == 2
    assert [lbl[0] for lbl in labels[:4]] == ["a", "b", "a", "b"]
    again = exp_i.select_transcripts(cands, n=5, per_source=3, **kw)  # type: ignore[arg-type]
    assert [c.label for c, _ in again] == labels


# --------------------------------------------------------------------------- conditions


def _prepared() -> exp_i.Prepared:
    t = _session(60)
    cut = exp_i.find_cut(t, random.Random(1), tail_tokens=1000, min_compressible=2000)
    assert cut is not None
    return exp_i.prepare(exp_i.Candidate("lbl", "a", t), cut, _keep_every_other)


def test_every_condition_keeps_the_tail_verbatim_and_only_compresses_the_earlier_stretch() -> None:
    p = _prepared()
    summaries = {c: "a short summary" for c in exp_i.SUMMARY_OF}
    for cond in exp_i.CONDITIONS:
        ctx = exp_i.context_for(p, cond, summaries)
        assert ctx is not None
        assert ctx.endswith(p.tail)
        assert exp_i.TAIL_HEADER in ctx
    assert exp_i.context_for(p, "summary-light", {}) is None
    full = count_tokens(p.earlier["full"])
    light, heavy = (count_tokens(p.earlier[c]) for c in ("loosy-light", "loosy-heavy"))
    assert heavy < light < full
    assert p.reference.startswith("### assistant")
    assert "<tool_use" in p.reference


def test_summary_budget_is_the_matching_loosy_conditions_rendered_size() -> None:
    p = _prepared()
    for cond, loosy in exp_i.SUMMARY_OF.items():
        assert p.summary_budget(cond) == count_tokens(p.earlier[loosy])


def test_truncate_tokens_allows_a_small_overrun_then_cuts_to_budget() -> None:
    text = _words(1000, "w")
    n = count_tokens(text)
    assert exp_i.truncate_tokens(text, int(n / 1.05)) == (text, False)
    cut, truncated = exp_i.truncate_tokens(text, 100)
    assert truncated and count_tokens(cut) <= 101


def test_render_segments_marks_gaps_within_a_block() -> None:
    def seg(i: int, block: int, text: str) -> Segment:
        return Segment(i, 1, "user", "tool_result", text, False, block=block)

    out = exp_i.render_segments([seg(0, 0, "a"), seg(2, 0, "c"), seg(3, 1, "d")])
    assert "a\n[…]\nc" in out
    assert out.count("<tool_result>") == 2


def test_prepared_round_trips_through_its_cache(tmp_path: Path) -> None:
    t = _session(60)
    cut = exp_i.find_cut(t, random.Random(1), tail_tokens=1000, min_compressible=2000)
    assert cut is not None
    cand = exp_i.Candidate("lbl", "a", t)
    calls: list[float] = []

    def counting(t: Transcript, keep: float) -> list[Segment]:
        calls.append(keep)
        return _keep_every_other(t, keep)

    first = exp_i.load_or_prepare(cand, cut, tmp_path, counting)
    second = exp_i.load_or_prepare(cand, cut, tmp_path, counting)
    assert first == second
    assert len(calls) == 3


# --------------------------------------------------------------------------- parsing


def test_parsers_tolerate_fences_and_reject_malformed_replies() -> None:
    body = json.dumps({"probes": [{"q": f"{c}?", "a": str(i)} for i, c in enumerate("abc", 1)]})
    probes = f"```json\n{body}\n```"
    assert exp_i.parse_probes(probes) == [
        {"q": "a?", "a": "1"},
        {"q": "b?", "a": "2"},
        {"q": "c?", "a": "3"},
    ]
    assert exp_i.parse_probes('{"probes": [{"q": "a?", "a": "1"}]}') is None
    assert exp_i.parse_answers('{"answers": {"1": "x", "3": "z"}}', 3) == ["x", "", "z"]
    assert exp_i.parse_answers("1. x\n2) y", 2) == ["x", "y"]
    assert exp_i.parse_answers("no idea", 2) is None
    assert exp_i.parse_probe_grades(
        '{"grades": {"1": "correct", "2": "Partial", "3": "wrong"}}', 3
    ) == [1.0, 0.5, 0.0]
    assert exp_i.parse_probe_grades('{"grades": {"1": "correct"}}', 2) is None
    assert exp_i.parse_next_grade('{"grade": "same", "why": "x"}') == 1.0
    assert exp_i.parse_next_grade('{"grade": "maybe"}') is None


def test_claude_and_codex_output_parsing() -> None:
    ok = json.dumps(
        {
            "is_error": False,
            "result": "hi",
            "usage": {
                "input_tokens": 2,
                "cache_creation_input_tokens": 600,
                "cache_read_input_tokens": 10,
                "output_tokens": 5,
            },
        }
    )
    assert exp_i.parse_claude_output(ok, 0) == exp_i.CallResult("hi", 612, 5)
    err = json.dumps({"is_error": True, "api_error_status": 429, "result": "rate limited"})
    with pytest.raises(exp_i.CallError) as exc:
        exp_i.parse_claude_output(err, 1)
    assert exc.value.transient
    events = "\n".join(
        json.dumps(e)
        for e in [
            {"type": "item.completed", "item": {"type": "command_execution"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "x"}},
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "output_tokens": 3, "reasoning_output_tokens": 2},
            },
        ]
    )
    assert exp_i.parse_codex_output(events, "final", 0) == exp_i.CallResult("final", 100, 5, 1)
    with pytest.raises(exp_i.CallError) as exc:
        exp_i.parse_codex_output('{"type": "error", "message": "bad model"}', None, 1)
    assert not exc.value.transient


# --------------------------------------------------------------------------- retry and checkpoint


def test_retry_backs_off_on_transient_errors_and_gives_up_on_persistent_ones() -> None:
    sleeps: list[float] = []
    seq: list[Exception | exp_i.CallResult] = [
        exp_i.CallError("usage limit reached"),
        exp_i.CallError("429 rate limit"),
        exp_i.CallResult("ok", 1, 1),
    ]

    def runner(model: exp_i.Model, prompt: str) -> exp_i.CallResult:
        nxt = seq.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    result, attempts = exp_i.call_with_retry(
        runner, exp_i.HELPER, "p", sleep=sleeps.append, log=lambda m: None
    )
    assert result.text == "ok" and attempts == 3
    assert sleeps == [exp_i.BACKOFF_START_S, exp_i.BACKOFF_START_S * 2]

    def broken(model: exp_i.Model, prompt: str) -> exp_i.CallResult:
        raise exp_i.CallError("unsupported model")

    with pytest.raises(exp_i.CallFailedError):
        exp_i.call_with_retry(broken, exp_i.HELPER, "p", sleep=lambda s: None, log=lambda m: None)

    def invalid(model: exp_i.Model, prompt: str) -> exp_i.CallResult:
        return exp_i.CallResult("not json", 1, 1)

    with pytest.raises(exp_i.CallFailedError):
        exp_i.call_with_retry(
            invalid, exp_i.HELPER, "p", lambda t: False, sleep=lambda s: None, log=lambda m: None
        )


def test_backoff_is_capped() -> None:
    sleeps: list[float] = []

    def always(model: exp_i.Model, prompt: str) -> exp_i.CallResult:
        raise exp_i.CallError("overloaded")

    with pytest.raises(exp_i.CallFailedError):
        exp_i.call_with_retry(always, exp_i.HELPER, "p", sleep=sleeps.append, log=lambda m: None)
    assert len(sleeps) == exp_i.MAX_TRANSIENT_ATTEMPTS - 1
    assert max(sleeps) == exp_i.BACKOFF_MAX_S


def test_store_skips_a_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "calls.jsonl"
    path.write_text('{"key": "a", "text": "x"}\n{"key": "b", "te', encoding="utf-8")
    store = exp_i.Store(path)
    assert set(store.records) == {"a"}


class _FakeCli:
    """Answers every prompt shape the experiment sends with a well-formed reply."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model: exp_i.Model, prompt: str) -> exp_i.CallResult:
        self.calls.append((model.name, prompt[:40]))
        if '"probes"' in prompt:
            probes = [{"q": f"q{i}?", "a": f"a{i}"} for i in range(4)]
            return exp_i.CallResult(json.dumps({"probes": probes}), 10, 10)
        if '"answers"' in prompt:
            return exp_i.CallResult(json.dumps({"answers": {"1": "a0", "2": "a1"}}), 10, 10)
        if '"grades"' in prompt:
            n = prompt.count("Candidate answer:")
            grades = {str(i): ("correct" if i <= 2 else "wrong") for i in range(1, n + 1)}
            return exp_i.CallResult(json.dumps({"grades": grades}), 10, 10)
        if '"grade"' in prompt:
            return exp_i.CallResult('{"grade": "partial", "why": "x"}', 10, 10)
        if "ACTION:" in prompt:
            return exp_i.CallResult(f"ACTION: Read\nTARGET: x\nINTENT: {len(prompt)}", 10, 10)
        return exp_i.CallResult("a summary " * 20000, 10, 10)


def _selected() -> list[tuple[exp_i.Candidate, exp_i.Cut]]:
    out = []
    for label in ("s1", "s2"):
        t = _session(60, label=label)
        cut = exp_i.find_cut(t, exp_i.cut_rng(label), tail_tokens=1000, min_compressible=2000)
        assert cut is not None
        out.append((exp_i.Candidate(label, "a", t), cut))
    return out


def test_full_run_with_a_stub_cli_then_rerun_makes_zero_calls(tmp_path: Path) -> None:
    fake = _FakeCli()
    rt = exp_i.Runtime(
        exp_i.Store(tmp_path / "calls.jsonl"), fake, tmp_path / "failures.jsonl", workers=1
    )
    rt.log = lambda m: None
    prepared, summary = exp_i.run(_selected(), rt, tmp_path, _keep_every_other)
    per_transcript = 1 + 3 + 2 * len(exp_i.CONDITIONS) * len(exp_i.MODELS) * 2
    assert rt.new_calls == len(fake.calls) == 2 * per_transcript
    assert rt.failed == 0
    planned = json.loads((tmp_path / "manifest.json").read_text())["planned"]
    assert sum(len(v) for v in planned.values()) == 2 * per_transcript
    assert set(rt.store.records) == {k for v in planned.values() for k in v}
    grading = [r for r in rt.store.records.values() if r["task"].startswith("grade")]
    assert grading and all(r["caller"] == exp_i.HELPER.name for r in grading)
    # Probes 1-2 pass the full-context filter and every condition scores them correct.
    assert "1.00 [1.00,1.00] n=2" in summary
    assert "0.50 [0.50,0.50] n=2" in summary

    rt2 = exp_i.Runtime(
        exp_i.Store(tmp_path / "calls.jsonl"), fake, tmp_path / "failures.jsonl", workers=2
    )
    rt2.log = lambda m: None
    before = len(fake.calls)
    exp_i.run(_selected(), rt2, tmp_path, _keep_every_other)
    assert rt2.new_calls == 0 and len(fake.calls) == before
    assert "COMPLETE" in exp_i_status.render(tmp_path)


def test_a_failed_call_is_logged_and_retried_on_the_next_run(tmp_path: Path) -> None:
    fake = _FakeCli()
    state = {"fail": True}

    def flaky(model: exp_i.Model, prompt: str) -> exp_i.CallResult:
        if state["fail"] and "ACTION:" in prompt and model.name == "gpt-6-sol":
            raise exp_i.CallError("unsupported model")
        return fake(model, prompt)

    def make() -> exp_i.Runtime:
        rt = exp_i.Runtime(
            exp_i.Store(tmp_path / "calls.jsonl"),
            flaky,
            tmp_path / "failures.jsonl",
            workers=1,
            sleep=lambda s: None,
        )
        rt.log = lambda m: None
        return rt

    sel = _selected()[:1]
    rt = make()
    exp_i.run(sel, rt, tmp_path, _keep_every_other)
    assert rt.failed == len(exp_i.CONDITIONS)
    assert "FAILED and not yet retried" in exp_i_status.render(tmp_path)
    state["fail"] = False
    rt = make()
    exp_i.run(sel, rt, tmp_path, _keep_every_other)
    # The failed next-step answers plus their now-possible grading calls.
    assert rt.new_calls == 2 * len(exp_i.CONDITIONS)
    assert "COMPLETE" in exp_i_status.render(tmp_path)


# --------------------------------------------------------------------------- aggregation


def test_bootstrap_ci_brackets_the_mean_and_collapses_on_constant_data() -> None:
    mean, lo, hi = exp_i.bootstrap_ci([0.0, 0.5, 1.0, 1.0, 0.5])
    assert lo <= mean <= hi and mean == pytest.approx(0.6)
    assert exp_i.bootstrap_ci([0.7] * 5) == pytest.approx((0.7, 0.7, 0.7))
    assert all(x != x for x in exp_i.bootstrap_ci([]))


def test_probe_filter_uses_only_probes_the_helper_gets_right_from_full_context(
    tmp_path: Path,
) -> None:
    store = exp_i.Store(tmp_path / "c.jsonl")

    def put(cond: str, model: str, task: str, text: str) -> None:
        key = exp_i.call_key("L", cond, model, task)
        store.put({"key": key, "transcript": "L", "task": task, "model": model, "text": text})

    probes = [{"q": f"q{i}", "a": f"a{i}"} for i in range(3)]
    put("-", exp_i.HELPER.name, "probe_gen", json.dumps({"probes": probes}))
    put(
        "full",
        exp_i.HELPER.name,
        "grade_probe",
        '{"grades": {"1": "correct", "2": "wrong", "3": "correct"}}',
    )
    put(
        "loosy-heavy",
        "haiku-5.5",
        "grade_probe",
        '{"grades": {"1": "wrong", "2": "correct", "3": "partial"}}',
    )
    probe, _, kept = exp_i.collect_scores(store, ["L"])
    assert kept == {"L": 2}
    assert probe[("loosy-heavy", "haiku-5.5")]["L"] == pytest.approx(0.25)
    assert probe[("full", exp_i.HELPER.name)]["L"] == 1.0


def test_a_refused_summary_is_retried_and_never_used(tmp_path: Path) -> None:
    fake = _FakeCli()
    refusals = {"left": 1}

    def refusing(model: exp_i.Model, prompt: str) -> exp_i.CallResult:
        if "<earlier_session>" in prompt and '"probes"' not in prompt and refusals["left"]:
            refusals["left"] -= 1
            return exp_i.CallResult("I won't write that.", 5, 5)
        return fake(model, prompt)

    rt = exp_i.Runtime(
        exp_i.Store(tmp_path / "calls.jsonl"), refusing, tmp_path / "f.jsonl", workers=1
    )
    rt.log = lambda m: None
    exp_i.run(_selected()[:1], rt, tmp_path, _keep_every_other)
    assert rt.failed == 0 and refusals["left"] == 0
    summaries = [r for r in rt.store.records.values() if r["task"] == "summary"]
    assert all("won't" not in r["text"] for r in summaries)


def test_a_changed_upstream_summary_reruns_its_answers_and_their_grading(tmp_path: Path) -> None:
    fake = _FakeCli()
    sel = _selected()[:1]

    def make() -> exp_i.Runtime:
        rt = exp_i.Runtime(
            exp_i.Store(tmp_path / "calls.jsonl"), fake, tmp_path / "f.jsonl", workers=1
        )
        rt.log = lambda m: None
        return rt

    exp_i.run(sel, make(), tmp_path, _keep_every_other)
    store = exp_i.Store(tmp_path / "calls.jsonl")
    key = exp_i.call_key(sel[0][0].label, "summary-light", exp_i.HELPER.name, "summary")
    store.put({**store.records[key], "text": "a different summary " * 20000})
    rt = make()
    exp_i.run(sel, rt, tmp_path, _keep_every_other)
    # summary-light: 4 models x (probe + next) answers rerun; only the next-step replies differ
    # (the stub's probe answers ignore the context), so only those 4 gradings rerun.
    assert rt.new_calls == 4 * 2 + 4

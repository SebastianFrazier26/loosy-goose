import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_i_downstream as exi  # noqa: E402
from experiments.continuous import local  # noqa: E402

QWEN = exi.MODEL_BY_NAME["qwen3-8b"]
NEMO = exi.MODEL_BY_NAME["mistral-nemo-12b"]
HAIKU = exi.MODEL_BY_NAME["haiku-5.5"]


def test_dispatch_routes_mlx_to_local(monkeypatch):
    seen = []
    monkeypatch.setattr(
        local, "mlx_runner", lambda m, p: seen.append(("local", m.name)) or exi.CallResult("", 0, 0)
    )

    def cli(m, p):
        seen.append(("cli", m.name))
        return exi.CallResult("", 0, 0)

    run = local.dispatch(cli)
    run(QWEN, "x")
    run(NEMO, "x")
    run(HAIKU, "x")
    assert seen == [("local", "qwen3-8b"), ("local", "mistral-nemo-12b"), ("cli", "haiku-5.5")]


def test_existing_model_names_unchanged():
    assert [m.name for m in exi.MODELS][:4] == [
        "haiku-5.5",
        "sonnet-5.5",
        "gpt-6-luna",
        "gpt-6-sol",
    ]
    assert NEMO.cli == "mlx" and QWEN.cli == "mlx"


def test_qwen_tokens_counts(monkeypatch):
    class Tok:
        def encode(self, text):
            return text.split()

    monkeypatch.setattr(local, "_tokenizer", lambda model_id: Tok())
    assert local.qwen_tokens("a b c") == 3


def test_context_over_window_raises(monkeypatch):
    monkeypatch.setattr(
        local, "_count", lambda model, prompt: local.QWEN_WINDOW - local.MAX_TOKENS + 1
    )
    with pytest.raises(exi.CallError, match=r"context over Qwen window: \d+ tokens"):
        local.mlx_runner(QWEN, "x")


def test_nemo_has_larger_window(monkeypatch):
    n = local.QWEN_WINDOW + 5000
    monkeypatch.setattr(local, "_count", lambda model, prompt: n)
    monkeypatch.setattr(local, "_generate", lambda model, prompt: ("ok", n, 2))
    assert local.mlx_runner(NEMO, "x") == exi.CallResult("ok", n, 2)
    monkeypatch.setattr(local, "_count", lambda model, prompt: local.NEMO_WINDOW)
    with pytest.raises(exi.CallError, match="context over mistral-nemo-12b window"):
        local.mlx_runner(NEMO, "x")


def test_load_evicts_previous_model(monkeypatch):
    loaded, freed = [], []
    monkeypatch.setattr(local, "_loaded", None)
    monkeypatch.setattr(local, "_load_model", lambda mid: loaded.append(mid) or (mid, "tok"))
    monkeypatch.setattr(local, "_free", lambda: freed.append(True))
    local.load("a")
    local.load("a")
    assert loaded == ["a"] and freed == []
    local.load("b")
    assert loaded == ["a", "b"] and freed == [True]
    assert local.load("b")[0] == "b"


PROBES = [{"q": "q1", "a": "a1"}, {"q": "q2", "a": "a2"}]


def test_default_answerer_keys_unchanged(monkeypatch):
    from experiments.slot_compress import downstream as dl

    a = dl.answer_job("L", "c", "ctx", PROBES)
    assert a.key == "v1|L|c|haiku-5.5|probe"
    assert a.prompt_sha == "046bc94fed9448ed"
    monkeypatch.setattr(exi, "_text", lambda *args: "1. x\n2. y")
    g = dl.grade_job(None, "L", "c", PROBES)
    assert g.key == "v1|L|c|haiku-5.5|grade_probe"
    assert g.prompt_sha == "697c0acd38952940"
    assert g.runs_as is dl.SONNET


def test_qwen_answer_key_differs():
    from experiments.slot_compress import downstream as dl

    q = dl.answer_job("L", "c", "ctx", PROBES, answerer=QWEN)
    assert q.key != dl.answer_job("L", "c", "ctx", PROBES).key
    assert "qwen3-8b" in q.key


def test_nemo_grades_qwen_answers(monkeypatch):
    from experiments.slot_compress import downstream as dl

    seen = []
    monkeypatch.setattr(exi, "_text", lambda *args: seen.append(args[3]) or "1. x\n2. y")
    g = dl.grade_job(None, "L", "c", PROBES, answerer=QWEN, grader=NEMO)
    assert seen == ["qwen3-8b"]
    assert g.key == "v1|L|c|qwen3-8b|grade_probe@mistral-nemo-12b"
    assert g.runs_as is NEMO


def _turns(rounds: int = 30, size: int = 300) -> list:
    import json

    from loosy_goose.transcript import Block, Turn

    def words(n: int, tag: str) -> str:
        return " ".join(f"{tag}{i}" for i in range(n))

    turns = [Turn(0, "system", [Block("text", "system prompt")])]
    turns.append(Turn(1, "user", [Block("text", words(size, "ask"))]))
    for r in range(rounds):
        use = json.dumps({"file_path": f"src/f{r}.py"})
        turns.append(Turn(len(turns), "assistant", [Block("tool_use", use, {"name": "Read"})]))
        turns.append(Turn(len(turns), "user", [Block("tool_result", words(size, f"r{r}x"))]))
    return turns


def _stub_plans() -> list:
    from experiments.continuous import sim

    return [sim.plan_transcript(f"t{i}", _turns()) for i in range(2)]


def test_qwen_plan_counts():
    from experiments.continuous import run, sim

    cfg = run.QWEN_CFG
    plans = _stub_plans()
    assert [len(p.checkpoints) for p in plans] == [3, 3]
    counts = run.call_counts(cfg, plans)
    assert counts["probe_gen"] == 2 * 3
    assert counts["answers"] == 2 * 3 * 5
    assert counts["grades"] == 2 * 3 * 5
    n_sum = sum(len(sim.thresh_events(p)) for p in plans)
    assert n_sum > 0 and counts["qthresh_summaries"] == n_sum
    models = run.phase_models(cfg, cfg.prober)
    assert models["qthresh_summaries"] is QWEN
    assert QWEN.model_id == run.QSUM_MODEL_ID
    local_n, api_n = run.split_calls(cfg, cfg.prober, counts)
    assert api_n == 0 and local_n == sum(counts.values())
    f_local, f_api = run.split_calls(run.CFGS["F"], HAIKU, run.call_counts(run.CFGS["F"], plans))
    assert f_local == 0 and f_api > 0


def test_qwen_run_uses_no_closed_model(tmp_path):
    from experiments.continuous import run

    cfg = run.QWEN_CFG
    plans = _stub_plans()
    assert all(m.cli == "mlx" for m in run.phase_models(cfg, cfg.prober).values())
    probes = run.probe_jobs(cfg, plans, cfg.prober)
    assert len(probes) == 6
    assert all(j.runs_as.cli == "mlx" and j.condition == "probes.nemo1" for j in probes)
    store = exi.Store(tmp_path / "calls.jsonl")
    label = run.ck_label(plans[0], plans[0].checkpoints[0])
    for arm in cfg.arms:
        ans = run.answer_job(cfg, label, arm, "ctx", PROBES)
        assert ans.runs_as is QWEN
        store.put({"key": ans.key, "prompt_sha": ans.prompt_sha, "text": "1. x\n2. y"})
        grade = run.grade_job(cfg, store, label, arm, PROBES)
        assert grade is not None and grade.runs_as is NEMO
        assert grade.task == "grade_probe@mistral-nemo-12b"


def test_qwen_runner_refuses_closed_model(monkeypatch):
    from experiments.continuous import run

    monkeypatch.setattr(local, "mlx_runner", lambda m, p: exi.CallResult(m.name, 1, 1))
    guard = run.LocalGuard(10)
    assert guard.runner(QWEN, "x").text == "qwen3-8b"
    assert guard.runner(NEMO, "x").text == "mistral-nemo-12b"
    for closed in exi.MODELS:
        with pytest.raises(run.ClosedModelError):
            guard.runner(closed, "x")


def test_qwen_condition_keys():
    from experiments.continuous import run, sim

    q, f = run.QWEN_CFG, run.CFGS["F"]
    assert run.cond(q, "qthresh") == f"qthresh.F.v{sim.CONT_VERSION}.q{sim.QTHRESH_VERSION}"
    assert run.cond(q, "cfrzg") == f"cfrzg.F.v{sim.CONT_VERSION}.q{sim.CFRZG_VERSION}"
    for arm in ("full", "tloosy", "cfrzm"):
        assert run.cond(q, arm) == run.cond(f, arm)


def test_qwen_summaries_read_lsum_cache(tmp_path, monkeypatch):
    import json

    from experiments.continuous import run, sim
    from experiments.slot_compress import lsum

    monkeypatch.setattr(run, "QSUM_DIR", tmp_path)
    strict = run.QSummaries("L", strict=True)
    with pytest.raises(sim.MissingSummaryError):
        strict(0, "earlier text", 50)
    rec = {"sha": lsum.source_sha("earlier text", 50, run.QSUM_MODEL_ID), "raw": "A summary."}
    lsum.cache_path(tmp_path, "L", "e0").write_text(json.dumps(rec), encoding="utf-8")
    sums = run.QSummaries("L", strict=True)
    assert sums(0, "earlier text", 50) == "A summary."
    assert sums.sig() != "placeholder"
    loose = run.QSummaries("L", strict=False)
    assert loose(1, "other", 50).startswith("[placeholder summary 1]") and loose.placeholder


def _probe_text() -> str:
    import json

    return json.dumps({"probes": [{"q": f"q{i}", "a": f"a{i}"} for i in range(3)]})


def _grade_text(g: str) -> str:
    import json

    return json.dumps({"grades": {str(i): g for i in range(1, 4)}})


def test_full_pairs_per_checkpoint(tmp_path):
    from experiments.continuous import run

    cfg = run.QWEN_CFG
    plans = _stub_plans()
    store = exi.Store(tmp_path / "calls.jsonl")
    for p in plans:
        for ck in p.checkpoints:
            label = run.ck_label(p, ck)
            key = exi.call_key(label, cfg.probes_key, NEMO.name, "probe_gen")
            store.put({"key": key, "text": _probe_text()})
            missing_full = p.label == "t0" and ck.pct == 100
            for arm, g in (("full", "wrong"), ("cfrzm", "correct")):
                if arm == "full" and missing_full:
                    continue
                g = "wrong" if missing_full else g
                key = exi.call_key(label, run.cond(cfg, arm), QWEN.name, cfg.grade_task)
                store.put({"key": key, "text": _grade_text(g)})
    scores, by_pct = run.collect_scores(cfg, store, plans, NEMO)
    assert set(scores["full"]) == {"t1"}
    pcfg = dataclasses.replace(cfg, pairs=(("cfrzm", "full"),))
    row = run.pair_table(pcfg, scores, by_pct)[-1]
    assert row.startswith("| cfrzm - full | +1.000 ")
    assert "n=2 transcripts, 5 checkpoints" in row
    assert "n=1" in run.pair_table(pcfg, scores)[-1]


def test_over_window_answers_not_queued(tmp_path, monkeypatch):
    from experiments.continuous import run, sim

    plans = _stub_plans()[:1]
    cfg = dataclasses.replace(run.QWEN_CFG, out=tmp_path, arms=("full", "cfrzm"))
    store = exi.Store(tmp_path / "calls.jsonl")
    for j in run.probe_jobs(cfg, plans, NEMO):
        store.put({"key": j.key, "prompt_sha": j.prompt_sha, "text": _probe_text()})
    p = plans[0]

    def fake_arm_run(cfg_, arm, plan, store_, strict=False):
        ctx = {ck.t: ("BIG " if arm == "full" else "") + f"ctx {ck.t}" for ck in plan.checkpoints}
        return sim.ArmRun(arm, contexts=ctx), False

    monkeypatch.setattr(run, "arm_run", fake_arm_run)
    monkeypatch.setattr(run, "qwen_over", lambda prompt: 40_000 if "BIG" in prompt else None)
    prompts = []

    def runner(model, prompt):
        prompts.append(prompt)
        if model is QWEN:
            return exi.CallResult("1. x\n2. y\n3. z", 1, 1)
        return exi.CallResult(_grade_text("correct"), 1, 1)

    rt = exi.Runtime(store, runner, tmp_path / "f.jsonl", workers=1)
    rt.log = lambda m: None
    run.run_all(cfg, rt, plans, NEMO)
    assert prompts and not any("BIG" in x for x in prompts)
    assert len(prompts) == 2 * len(p.checkpoints)
    failures = (tmp_path / "qwen_failures.jsonl").read_text().splitlines()
    assert len(failures) == len(p.checkpoints)
    assert all("over Qwen window: 40000 tokens" in f for f in failures)


def test_failed_qthresh_summary_skips_transcript(tmp_path, monkeypatch):
    import json

    from experiments.continuous import run, sim
    from experiments.slot_compress import lsum

    plans = _stub_plans()
    cfg = dataclasses.replace(run.QWEN_CFG, out=tmp_path)
    monkeypatch.setattr(run, "QSUM_DIR", tmp_path / "qsum")

    def fake_precompute(jobs, cache_dir, prompt_fn, log, gen=None, model_id=""):
        for label, level, earlier, target in jobs:
            if label == "t0":
                raise RuntimeError("metal out of memory")
            cache_dir.mkdir(parents=True, exist_ok=True)
            rec = {"sha": lsum.source_sha(earlier, target, model_id), "raw": "Summary text."}
            lsum.cache_path(cache_dir, label, level).write_text(json.dumps(rec))

    monkeypatch.setattr(lsum, "precompute", fake_precompute)
    monkeypatch.setattr(run.dl, "free_mlx", lambda: None)
    failed = run.qwen_summary_rounds(cfg, plans, make_gen=lambda model_id: object())
    assert failed == {"t0"}
    lines = [json.loads(x) for x in (tmp_path / "qwen_failures.jsonl").read_text().splitlines()]
    assert [x["key"] for x in lines] == ["qsum|t0|e0"]
    assert "metal out of memory" in lines[0]["error"]
    assert sim.thresh_events(plans[1], run.QSummaries("t1", strict=True))
    with pytest.raises(sim.MissingSummaryError):
        sim.thresh_events(plans[0], run.QSummaries("t0", strict=True))


def test_nemo_tokenizer_gets_regex_fix(monkeypatch):
    import transformers

    seen = []

    def fake(model_id, **kwargs):
        seen.append((model_id, kwargs))
        return object()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake)
    monkeypatch.setattr(local, "_tokenizers", {})
    monkeypatch.setattr(local, "_fix_ok", None)
    local._tokenizer(QWEN.model_id)
    local._tokenizer(NEMO.model_id)
    assert seen == [(QWEN.model_id, {}), (NEMO.model_id, {"fix_mistral_regex": True})]
    assert local._fix_ok is True
    assert local._tokenizer_kwargs(NEMO.model_id) == {"fix_mistral_regex": True}
    assert local._tokenizer_kwargs(QWEN.model_id) == {}


def test_nemo_tokenizer_falls_back_without_regex_fix(monkeypatch):
    import transformers

    seen = []

    def fake(model_id, **kwargs):
        if kwargs:
            raise TypeError("unexpected keyword argument 'fix_mistral_regex'")
        seen.append(model_id)
        return object()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake)
    monkeypatch.setattr(local, "_tokenizers", {})
    monkeypatch.setattr(local, "_fix_ok", None)
    local._tokenizer(NEMO.model_id)
    assert seen == [NEMO.model_id] and local._fix_ok is False
    assert local._tokenizer_kwargs(NEMO.model_id) == {}


def test_reused_prefix_len():
    from experiments.continuous import latency

    assert latency.reused_prefix([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert latency.reused_prefix([1, 2], [1, 2, 3]) == 2
    assert latency.reused_prefix([1, 2, 3], [1, 2]) == 2
    assert latency.reused_prefix([], [1]) == 0
    assert latency.reused_prefix([5], [6]) == 0

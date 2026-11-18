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

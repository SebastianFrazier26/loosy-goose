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

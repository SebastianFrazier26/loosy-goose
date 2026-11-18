"""Local open-weight runner for the continuous replay (2026-11-17): Qwen3-8B answers and
Mistral-Nemo 12B writes probes and grades, both 4-bit through mlx-lm. Only `mlx` models come here;
`dispatch` sends every other model to the CLI runner. Greedy decoding."""

from __future__ import annotations

import re
from typing import Any

from experiments import exp_i_downstream as exi
from experiments.slot_compress import downstream as sd
from experiments.slot_compress import lsum

QWEN_WINDOW = 32_768
NEMO_WINDOW = 131_072
# One cap for every local call: answers and grades need under 1024, probe generation under 1536.
MAX_TOKENS = 1536

_WINDOWS = {"qwen3-8b": QWEN_WINDOW, "mistral-nemo-12b": NEMO_WINDOW}
_loaded: tuple[str, Any, Any] | None = None
_tokenizers: dict[str, Any] = {}


NEMO_ID = exi.MODEL_BY_NAME["mistral-nemo-12b"].model_id
# The Nemo repo ships a pre-tekken pre-tokenizer regex; transformers 5.x warns that it mis-splits
# text and repairs it under this flag. Older transformers may lack it, so support is probed on the
# first Nemo tokenizer load and the mlx_lm load reuses that answer.
_FIX = {"fix_mistral_regex": True}
_fix_ok: bool | None = None


def _tokenizer_kwargs(model_id: str) -> dict[str, Any]:
    return dict(_FIX) if model_id == NEMO_ID and _fix_ok is not False else {}


def _tokenizer(model_id: str) -> Any:
    global _fix_ok
    if model_id not in _tokenizers:
        from transformers import AutoTokenizer

        kwargs = _tokenizer_kwargs(model_id)
        try:
            _tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id, **kwargs)
        except TypeError:
            if not kwargs:
                raise
            _fix_ok = False
            _tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
        else:
            if kwargs:
                _fix_ok = True
    return _tokenizers[model_id]


def qwen_tokens(text: str) -> int:
    return len(_tokenizer(exi.MODEL_BY_NAME["qwen3-8b"].model_id).encode(text))


def _count(model: exi.Model, prompt: str) -> int:
    return len(_tokenizer(model.model_id).encode(prompt))


def _load_model(model_id: str) -> tuple[Any, Any]:
    import mlx.core as mx
    from mlx_lm import load

    # Same allocator cap as lsum.MlxGenerator: freed Metal buffers otherwise count against the
    # working-set limit and a long decode dies out of memory on 16 GB.
    mx.set_cache_limit(lsum.CACHE_LIMIT_BYTES)
    _tokenizer(model_id)
    model, tok = load(model_id, tokenizer_config=_tokenizer_kwargs(model_id) or None)
    return model, tok


def _free() -> None:
    sd.free_mlx()


def load(model_id: str) -> tuple[Any, Any]:
    """The resident (model, tokenizer); loading another model evicts the previous one first."""
    global _loaded
    if _loaded is not None and _loaded[0] == model_id:
        return _loaded[1], _loaded[2]
    if _loaded is not None:
        _loaded = None
        _free()
    model, tok = _load_model(model_id)
    _loaded = (model_id, model, tok)
    return model, tok


def unload() -> None:
    global _loaded
    if _loaded is not None:
        _loaded = None
        _free()


def _generate(model: exi.Model, prompt: str) -> tuple[str, int, int]:
    import mlx.core as mx
    from mlx_lm import stream_generate

    net, tok = load(model.model_id)
    kwargs = {"enable_thinking": False} if model.name == "qwen3-8b" else {}
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    pieces: list[str] = []
    last = None
    for resp in stream_generate(net, tok, prompt=text, max_tokens=MAX_TOKENS):
        pieces.append(resp.text)
        last = resp
    mx.clear_cache()
    if last is None:
        raise exi.CallError(f"{model.name} produced no output")
    out = re.sub(r"<think>.*?</think>", "", "".join(pieces), flags=re.S).strip()
    return out, last.prompt_tokens, last.generation_tokens


def mlx_runner(model: exi.Model, prompt: str) -> exi.CallResult:
    window = _WINDOWS[model.name]
    n = _count(model, prompt)
    if n > window - MAX_TOKENS:
        label = "Qwen" if model.name == "qwen3-8b" else model.name
        raise exi.CallError(f"context over {label} window: {n} tokens")
    text, n_in, n_out = _generate(model, prompt)
    return exi.CallResult(text, n_in, n_out)


def dispatch(cli: exi.Runner) -> exi.Runner:
    def run(model: exi.Model, prompt: str) -> exi.CallResult:
        return mlx_runner(model, prompt) if model.cli == "mlx" else cli(model, prompt)

    return run

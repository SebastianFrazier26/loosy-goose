"""Local open-weight runner for the continuous replay (2026-11-17): Qwen3-8B answers and
Mistral-Nemo 12B writes probes and grades, both 4-bit through mlx-lm. Only `mlx` models come here;
`dispatch` sends every other model to the CLI runner. Greedy decoding."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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


def nemo_tokens(text: str) -> int:
    return len(_tokenizer(NEMO_ID).encode(text))


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


PROBE_PREFILL = '{"probes": [{"q": "'
ANSWER_PREFILL = '{"answers": {"1": "'
GRADE_PREFILL = '{"grades": {"1": "'
# Greedy decoding repeats the same bad reply, so attempts after the first sample. The pilot's one
# remaining failure was a probe answer that copied a long tool output until the token cap.
PROBE_ATTEMPTS = 3
RETRY_TEMP = 0.7


@dataclass(frozen=True)
class ProbeGenOptions:
    """Optional limits on local probe writing (2026-11-22, owner request). A greedy first attempt
    sometimes loops on one question until MAX_TOKENS, about 50 s, before a sampled retry
    succeeds. Both are off by default: the committed set F probes (probes.nemo4) were written
    without them, and either one changes what Nemo writes while the probe key stays the same, so
    set them only for a fresh run."""

    max_tokens: int | None = None
    repetition_penalty: float | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None and not 0 < self.max_tokens <= MAX_TOKENS:
            raise ValueError(f"probe max tokens must be in 1..{MAX_TOKENS}")
        if self.repetition_penalty is not None and self.repetition_penalty < 1.0:
            raise ValueError("probe repetition penalty must be at least 1.0")


PROBE_GEN = ProbeGenOptions()


def generation_limits(prefill: str, opts: ProbeGenOptions) -> tuple[int, float | None]:
    """(token cap, repetition penalty) for one call; the options touch probe writing only."""
    if prefill != PROBE_PREFILL:
        return MAX_TOKENS, None
    return opts.max_tokens or MAX_TOKENS, opts.repetition_penalty


class PrefilledPrompt(str):
    """A prompt whose assistant turn starts with `prefill`; the runner continues from it and
    returns prefill + continuation, so the stored text parses as the whole reply."""

    prefill: str
    max_attempts: int
    runs: int

    def __new__(cls, prompt: str, prefill: str, max_attempts: int = 1) -> PrefilledPrompt:
        obj = super().__new__(cls, prompt)
        obj.prefill = prefill
        obj.max_attempts = max_attempts
        obj.runs = 0
        return obj


def prefilled(prompt: str, prefill: str) -> PrefilledPrompt:
    """The same prompt text (so a stored record's prompt_sha still matches and successful calls
    stay done) with a JSON-opening prefill and PROBE_ATTEMPTS attempts. Only local jobs use it."""
    return PrefilledPrompt(prompt, prefill, PROBE_ATTEMPTS)


def probe_gen_prompt(earlier: str, n: int = exi.N_PROBES) -> PrefilledPrompt:
    """Nemo ignored the JSON-only instruction in the pilot (9 of 12 replies were markdown Q/A
    lists), so the local prompt prefills the opening of the JSON. A concrete example object was
    tried first (nemo2) and Nemo copied its values into probes, so the format line is bare.
    Closed models keep exi.probe_gen_prompt unchanged."""
    return PrefilledPrompt(
        "Below is the earlier part of an AI coding-agent session. Write exactly "
        f"{n} short factual questions about specific details stated in it, each with a short "
        "checkable answer (at most 12 words): file paths, values, error messages, identifiers, "
        "commands, decisions and their reasons. Spread the questions across the whole text. "
        "Each must be answerable from the text alone, with one unambiguous answer.\n"
        "Reply with a single JSON object and nothing else - no introduction, no markdown, no "
        "numbering. Format:\n"
        '{"probes": [{"q": "<question>", "a": "<short answer>"}, ...]}\n\n'
        f"<earlier_session>\n{earlier}\n</earlier_session>",
        PROBE_PREFILL,
        PROBE_ATTEMPTS,
    )


def close_probe_json(text: str) -> str:
    """Nemo ends the probe array with the object's brace misplaced, in two ways: it forgets it
    (`...}]`, 4 of the 6 nemo3 failures) or doubles the item's brace and drops it after the array
    (`...}}]`, the 8 probes.nemo4 failures replayed 2026-11-21). Only those two endings are
    repaired; the strict validator still decides whether the probes are usable."""
    stripped = text.rstrip()
    if stripped.endswith("}}]"):
        return stripped[:-3] + "}]}"
    return stripped + "}" if stripped.endswith("}]") else text


def drop_extra_closing_braces(text: str) -> str:
    """A complete JSON object followed only by stray closing braces (a nemo4 probe reply ended
    `...]}}`). Anything else trailing, or an object that does not parse, is left for the
    validator."""
    try:
        _, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return text
    rest = text[end:].strip()
    return text[:end] if rest and set(rest) == {"}"} else text


_UNKNOWN_LABEL = re.compile(r'("\d+"\s*:\s*")unknown(")', re.IGNORECASE)


def unknown_grade_to_wrong(text: str) -> str:
    """Nemo copies an UNKNOWN candidate into the grade label; the grader prompt already says an
    UNKNOWN candidate is wrong (owner ruling 2026-11-22). Applies only to a label value, and the
    validator still rejects every other label."""
    return _UNKNOWN_LABEL.sub(r"\1wrong\2", text)


def escape_string_controls(text: str) -> str:
    """Raw tabs and newlines inside a JSON string, which json.loads rejects. Qwen copies `cat -n`
    output ("    16<tab>...") into answers verbatim. Only characters inside strings are escaped, so
    the decoded value is unchanged and layout whitespace between tokens is left alone."""
    out: list[str] = []
    in_str = escaped = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            elif ord(ch) < 0x20:
                out.append(json.dumps(ch)[1:-1])
                continue
        elif ch == '"':
            in_str = True
        out.append(ch)
    return "".join(out)


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
    prefill = prompt.prefill if isinstance(prompt, PrefilledPrompt) else ""
    text += prefill
    pieces: list[str] = []
    last = None
    extra: dict[str, Any] = {}
    if isinstance(prompt, PrefilledPrompt):
        prompt.runs += 1
        if prompt.runs > 1:
            from mlx_lm.sample_utils import make_sampler

            extra["sampler"] = make_sampler(temp=RETRY_TEMP)
    max_tokens, penalty = generation_limits(prefill, PROBE_GEN)
    if penalty is not None:
        from mlx_lm.sample_utils import make_logits_processors

        extra["logits_processors"] = make_logits_processors(repetition_penalty=penalty)
    for resp in stream_generate(net, tok, prompt=text, max_tokens=max_tokens, **extra):
        pieces.append(resp.text)
        last = resp
    mx.clear_cache()
    if last is None:
        raise exi.CallError(f"{model.name} produced no output")
    out = re.sub(r"<think>.*?</think>", "", prefill + "".join(pieces), flags=re.S).strip()
    if prefill == GRADE_PREFILL:
        out = unknown_grade_to_wrong(out)
    if prefill:
        out = drop_extra_closing_braces(close_probe_json(escape_string_controls(out)))
    return out, last.prompt_tokens, last.generation_tokens


def mlx_runner(model: exi.Model, prompt: str) -> exi.CallResult:
    window = _WINDOWS[model.name]
    n = _count(model, prompt)
    if n > window - MAX_TOKENS:
        label = "Qwen" if model.name == "qwen3-8b" else model.name
        raise exi.CallError(f"context over {label} window: {n} tokens")
    try:
        text, n_in, n_out = _generate(model, prompt)
    except RuntimeError as exc:
        # MLX raises a plain RuntimeError, out of memory included. Greedy decoding would fail the
        # same way again, so it is non-retryable; the cache is freed so the next call can run.
        _free()
        err = exi.CallError(f"MLX RuntimeError in {model.name}: {exc}")
        err.transient = False
        raise err from exc
    return exi.CallResult(text, n_in, n_out)


def dispatch(cli: exi.Runner) -> exi.Runner:
    def run(model: exi.Model, prompt: str) -> exi.CallResult:
        return mlx_runner(model, prompt) if model.cli == "mlx" else cli(model, prompt)

    return run

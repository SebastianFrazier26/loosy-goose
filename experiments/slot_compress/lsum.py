"""Local summary arms `lsum` / `lsum4b` / `lsum8b` (2026-10-23, larger models 2026-11-15): the
Sonnet summary task given to a local Qwen3 (1.7B bf16, 4B and 8B official MLX 4-bit), greedy.

Generations are precomputed (`downstream.py --set D --precompute-lsum`) and cached on disk; the
call phase only reads the cache, so API workers never load a model. mlx-lm on Apple Silicon.
Thinking mode is off (`enable_thinking=False`). An earlier stretch longer than CHUNK_TOKENS is
split on block boundaries; each chunk is summarised to its share of the target and the pieces are
concatenated in order (the per-chunk targets already sum to the target, so no second model pass).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tiktoken

LSUM_VERSION = 1
MODEL_ID = "Qwen/Qwen3-1.7B"
# Family -> model. The 4B/8B are 4-bit because bf16 8B (16 GB) does not fit this machine, so size
# is confounded with quantisation against the bf16 1.7B; unquantised 4B (8 GB) was rejected for
# the same ladder-consistency reason.
LSUM_MODELS = {
    "lsum": MODEL_ID,
    "lsum4b": "Qwen/Qwen3-4B-MLX-4bit",
    "lsum8b": "Qwen/Qwen3-8B-MLX-4bit",
}
# Qwen's context is 32k; o200k counts differ from Qwen's, and the prompt and generation need room.
CHUNK_TOKENS = 16000
MAX_NEW_FACTOR = 1.4
CACHE_LIMIT_BYTES = 512 * 1024 * 1024
# Without it the 1.7B model continued the agent transcript (emitting tool calls) instead of
# summarising it; the user turn is exactly the Sonnet summary prompt. PREFILL opens the assistant
# turn for the same reason.
PREFILL = "Summary of the session so far:\n\n"
SYSTEM = (
    "You write prose summaries of AI coding-agent sessions. Reply with the summary text only: "
    "never tool calls, function-call markup or a continuation of the session."
)
_ENC = tiktoken.get_encoding("o200k_base")
_SENT_END = re.compile(r"[.!?](?=\s|$)|\n")


def _ntok(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def split_chunks(text: str, limit: int = CHUNK_TOKENS) -> list[str]:
    """Greedy packing of blank-line-separated blocks into chunks of at most `limit` tokens; a
    single block over the limit is cut by tokens."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_n = 0
    for block in text.split("\n\n"):
        n = _ntok(block)
        if n > limit:
            if cur:
                chunks.append("\n\n".join(cur))
                cur, cur_n = [], 0
            ids = _ENC.encode(block, disallowed_special=())
            chunks += [_ENC.decode(ids[i : i + limit]) for i in range(0, len(ids), limit)]
            continue
        if cur and cur_n + n + 2 > limit:
            chunks.append("\n\n".join(cur))
            cur, cur_n = [], 0
        cur.append(block)
        cur_n += n + 2
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def truncate_sentence(text: str, target: int) -> str:
    """Longest prefix within `target` o200k tokens that ends at a sentence or line boundary; a hard
    token cut when no boundary exists in the second half of the allowance."""
    text = text.strip()
    ids = _ENC.encode(text, disallowed_special=())
    if len(ids) <= target:
        return text
    head = _ENC.decode(ids[:target])
    ends = [m.end() for m in _SENT_END.finditer(head)]
    if ends and ends[-1] >= len(head) // 2:
        return head[: ends[-1]].strip()
    return head.strip()


def source_sha(earlier: str, target: int, model_id: str = MODEL_ID) -> str:
    h = hashlib.sha256(f"{model_id}|{LSUM_VERSION}|{SYSTEM}|{PREFILL}|{target}|{earlier}".encode())
    return h.hexdigest()[:16]


def cache_path(cache_dir: Path, label: str, level: str) -> Path:
    return cache_dir / f"{label}__{level}.json"


def cached_text(
    cache_dir: Path, label: str, level: str, earlier: str, target: int, model_id: str = MODEL_ID
) -> str | None:
    """The final (sentence-truncated) summary, or None when absent or stale."""
    path = cache_path(cache_dir, label, level)
    if not path.exists():
        return None
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("sha") != source_sha(earlier, target, model_id):
        return None
    return truncate_sentence(str(rec["raw"]), target)


@dataclass
class Stats:
    calls: int = 0
    prompt_tokens: int = 0
    prompt_seconds: float = 0.0
    gen_tokens: int = 0
    gen_seconds: float = 0.0
    peak_bytes: int = 0

    @property
    def prompt_tps(self) -> float:
        return self.prompt_tokens / self.prompt_seconds if self.prompt_seconds else 0.0

    @property
    def gen_tps(self) -> float:
        return self.gen_tokens / self.gen_seconds if self.gen_seconds else 0.0


class MlxGenerator:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        import mlx.core as mx
        from mlx_lm import load

        # MLX keeps freed buffers (2.6 GB after one 15k-token chunk with the 8B) and counts them
        # against the Metal working-set limit; on a 16 GB machine shared with other processes the
        # next chunk's decode then died with kIOGPUCommandBufferCallbackErrorOutOfMemory.
        # Allocator-only: outputs are unchanged.
        mx.set_cache_limit(CACHE_LIMIT_BYTES)
        self.model, self.tok = load(model_id)
        self.stats = Stats()

    def __call__(self, prompt: str, max_tokens: int) -> str:
        import mlx.core as mx
        from mlx_lm import stream_generate

        text = self.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        text += PREFILL
        pieces: list[str] = []
        last = None
        for resp in stream_generate(self.model, self.tok, prompt=text, max_tokens=max_tokens):
            pieces.append(resp.text)
            last = resp
        if last is not None:
            st = self.stats
            st.calls += 1
            st.prompt_tokens += last.prompt_tokens
            st.prompt_seconds += last.prompt_tokens / last.prompt_tps if last.prompt_tps else 0.0
            st.gen_tokens += last.generation_tokens
            st.gen_seconds += (
                last.generation_tokens / last.generation_tps if last.generation_tps else 0.0
            )
            st.peak_bytes = max(st.peak_bytes, int(last.peak_memory * 1e9))
        mx.clear_cache()
        return PREFILL + "".join(pieces)


def summarise(
    earlier: str,
    target: int,
    prompt_fn: Callable[[str, int], str],
    gen: Callable[[str, int], str],
    limit: int = CHUNK_TOKENS,
) -> str:
    chunks = split_chunks(earlier, limit)
    total = sum(_ntok(c) for c in chunks) or 1
    parts: list[str] = []
    for c in chunks:
        share = max(32, round(target * _ntok(c) / total))
        out = gen(prompt_fn(c, share), int(share * MAX_NEW_FACTOR) + 64)
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).removeprefix(PREFILL)
        parts.append(out.strip())
    return "\n\n".join(parts)


def precompute(
    jobs: list[tuple[str, str, str, int]],
    cache_dir: Path,
    prompt_fn: Callable[[str, int], str],
    log: Callable[[str], None],
    gen: Any = None,
    model_id: str = MODEL_ID,
) -> Stats:
    """jobs: (label, level, earlier, target). Skips fresh cache entries."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    gen = gen or None
    for label, level, earlier, target in jobs:
        if cached_text(cache_dir, label, level, earlier, target, model_id) is not None:
            continue
        gen = gen or MlxGenerator(model_id)
        t0 = time.perf_counter()
        raw = summarise(earlier, target, prompt_fn, gen)
        rec = {
            "sha": source_sha(earlier, target, model_id),
            "target": target,
            "raw": raw,
            "raw_tokens": _ntok(raw),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        cache_path(cache_dir, label, level).write_text(json.dumps(rec), encoding="utf-8")
        raw_n, secs = rec["raw_tokens"], rec["seconds"]
        log(f"{model_id} {label} {level}: target {target} raw {raw_n} in {secs}s")
    return gen.stats if gen is not None else Stats()

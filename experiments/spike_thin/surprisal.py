"""Per-character-span self-information from a small causal LM, scored with a sliding window.

Spike code (2026-10-21). Built directly on transformers/torch: no LLMLingua or similar.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch

LN2 = math.log(2.0)


@dataclass(frozen=True)
class LMSpec:
    key: str
    repo: str
    revision: str
    params_m: float
    license: str
    download_mb: int
    window: int
    dtype: str


SPECS: dict[str, LMSpec] = {
    s.key: s
    for s in (
        LMSpec(
            "pythia-31m",
            "EleutherAI/pythia-31m",
            "e556ace21b489575e94e9d50b6dad2fcc7419679",
            30.5,
            "Apache-2.0",
            61,
            2048,
            "float32",
        ),
        LMSpec(
            "pythia-70m",
            "EleutherAI/pythia-70m",
            "a39f36b100fe8a5377810d56c3f4789b9c53ac42",
            70.4,
            "Apache-2.0",
            166,
            2048,
            "float32",
        ),
        # smol_llama-81M was the first ~80M pick; its tokenizer needs sentencepiece, which this
        # env lacks (2026-10-22), so distilgpt2 stands in at that size.
        LMSpec(
            "distilgpt2-82m",
            "distilbert/distilgpt2",
            "2290a62682d06624634c1f46a6ad5be0f47f38aa",
            81.9,
            "Apache-2.0",
            353,
            1024,
            "float32",
        ),
        LMSpec(
            "smollm2-135m",
            "HuggingFaceTB/SmolLM2-135M",
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            134.5,
            "Apache-2.0",
            269,
            2048,
            "float32",
        ),
        LMSpec(
            "qwen3-0.6b",
            "Qwen/Qwen3-0.6B-Base",
            "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
            596.0,
            "Apache-2.0",
            1192,
            2048,
            "bfloat16",
        ),
        LMSpec(
            "qwen3-1.7b",
            "Qwen/Qwen3-1.7B-Base",
            "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
            1720.6,
            "Apache-2.0",
            3441,
            2048,
            "bfloat16",
        ),
    )
}


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Scorer:
    def __init__(self, spec: LMSpec) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.dev = device()
        t0 = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(spec.repo, revision=spec.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.repo, revision=spec.revision, dtype=getattr(torch, spec.dtype)
        ).to(self.dev)
        self.model.eval()
        self.load_s = time.perf_counter() - t0

    @torch.inference_mode()
    def token_bits(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """(bits per LM token, char end offset per LM token). The document's first token has no
        context to be predicted from, so it is given the mean rather than a fake 0 or infinity."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offs = np.array([e for _, e in enc["offset_mapping"]], dtype=np.int64)
        n = len(ids)
        bits = np.zeros(n, dtype=np.float64)
        if n < 2:
            return bits, offs
        w = self.spec.window
        stride = w // 2
        # Window k covers [start, start+w); it contributes the predictions for positions it has at
        # least `w - stride` tokens of context for (all of them for the first window).
        starts = list(range(0, max(n - w, 0) + 1, stride))
        if starts[-1] + w < n:
            starts.append(n - w)
        filled = np.zeros(n, dtype=bool)
        ids_t = torch.tensor(ids, dtype=torch.long)
        for start in starts:
            end = min(start + w, n)
            chunk = ids_t[start:end].unsqueeze(0).to(self.dev)
            logits = self.model(chunk).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1)
            tgt = chunk[0, 1:]
            nll = -lp.gather(1, tgt.unsqueeze(1)).squeeze(1).cpu().numpy() / LN2
            pos = np.arange(start + 1, end)
            new = ~filled[pos]
            bits[pos[new]] = nll[new]
            filled[pos[new]] = True
        bits[0] = float(bits[1:].mean())
        return bits, offs


def unigram_token_bits(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Zero-parameter floor: each whitespace word's self-information under the stretch's own
    word frequencies, as one pseudo-token per word."""
    words = list(re.finditer(r"\S+", text))
    counts = Counter(m.group(0).lower() for m in words)
    total = sum(counts.values())
    bits = np.array([-math.log2(counts[m.group(0).lower()] / total) for m in words])
    offs = np.array([m.end() for m in words], dtype=np.int64)
    return bits, offs

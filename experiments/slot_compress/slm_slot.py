"""SLM slot compression (spike, 2026-10-22): a small local instruct model rewrites each sentence.

Second engine for the scheme in docs/superpowers/specs/2026-10-22-slot-compress-design.md; the
deterministic engine is slot.py. Greedy decoding, so a (model, mode, SLM_VERSION, sentence) always
yields the same output and is cached on disk. The model only ever sees sentences with atoms
(identifiers, paths, numbers) swapped for short placeholders; a rewrite that loses a placeholder
or a negation falls back to the original sentence. Prose and thinking segments only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from loosy_goose.segment import Segment, extract_atoms
from loosy_goose.tokens import count_tokens

SLM_VERSION = 1

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "experiments" / "output" / "slot_compress" / "slm_cache"

SLM_LEGEND = (
    "Rows: act obj >to (extra). Flags before act: ~ negated, ! must/should, ^ may/can, "
    "+ will, ? question. Other lines verbatim."
)

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}

_NEG_RE = re.compile(
    r"\b(?:not|never|no|cannot|neither|nor|without|nothing|nobody|none)\b|n['’]t\b", re.I
)
_NEG_OUT_RE = re.compile(
    r"\b(?:not|never|no|cannot|neither|nor|without|nothing|nobody|none|non|un\w+|"
    r"don't|doesn't|didn't|won't|can't|isn't|aren't|wasn't|shouldn't)\b|n['’]t\b|^~|\s~|\bno-",
    re.I,
)
_PH = "⟨{}⟩"
_PH_RE = re.compile("⟨(\\d+)⟩")
_GLUED_RE = re.compile("\u27e9\u27e8|\\w\u27e8\\d+\u27e9|\u27e8\\d+\u27e9\\w")
_LIST_RE = re.compile(r"^\s*(?:[-*+•]\s|\d+[.)]\s|>\s?|#|\|)")
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'`(\[⟨])")
MIN_WORDS = 7

SYSTEM = {
    "english": (
        "Rewrite the sentence as ultra-terse caveman notes. Keep every fact, every negation "
        "(not, never, no) and every ⟨n⟩ token exactly as written. Drop articles, filler, "
        "hedges and pronouns. Use as few words as possible. Output only the rewrite."
    ),
    "direct": (
        "Rewrite the sentence as one terse row: `act obj >to (extra)` using as few words as "
        "possible. Prefix act with flags: ~ negated, ! must/should, ^ may/can, + will, ? question. "
        "Keep every ⟨n⟩ token exactly as written. Drop articles, filler, hedges and "
        "pronouns. Output only the row."
    ),
}
SHOTS: dict[str, list[tuple[str, str]]] = {
    "english": [
        (
            "I think we should probably not push directly to ⟨1⟩ because the CI run "
            "hasn't finished yet.",
            "not push ⟨1⟩ direct. CI unfinished.",
        ),
        (
            "Let me take a look at the failing test in ⟨1⟩ and see whether the fixture "
            "is causing the timeout.",
            "check failing test ⟨1⟩. fixture cause timeout?",
        ),
        (
            "The function returns ⟨1⟩ when the list is empty, which is why the caller crashes.",
            "function return ⟨1⟩ on empty list. caller crash.",
        ),
    ],
    "direct": [
        (
            "I think we should probably not push directly to ⟨1⟩ because the CI run "
            "hasn't finished yet.",
            "~!push ⟨1⟩ (CI unfinished)",
        ),
        (
            "Let me take a look at the failing test in ⟨1⟩ and see whether the fixture "
            "is causing the timeout.",
            "check failing test ⟨1⟩ (fixture cause timeout?)",
        ),
        (
            "The function returns ⟨1⟩ when the list is empty, which is why the caller crashes.",
            "return ⟨1⟩ (empty list; caller crash)",
        ),
    ],
}

Generator = Callable[[Sequence[str]], list[str]]


@dataclass
class Stats:
    sentences: int = 0
    cached: int = 0
    generated: int = 0
    accepted: int = 0
    fb_atoms: int = 0
    fb_neg: int = 0
    fb_empty: int = 0
    fb_longer: int = 0
    gen_seconds: float = 0.0

    @property
    def fallbacks(self) -> int:
        return self.fb_atoms + self.fb_neg + self.fb_empty + self.fb_longer


def mask_atoms(text: str, atoms: Sequence[str]) -> tuple[str, list[str]]:
    """Swap each atom present in text for a numbered placeholder; longest first so a path hides
    the identifiers inside it. Returns (masked, atoms by placeholder number - 1)."""
    used: list[str] = []
    for a in sorted(set(atoms), key=len, reverse=True):
        if a in text:
            used.append(a)
    masked = text
    for i, a in enumerate(used, 1):
        masked = masked.replace(a, _PH.format(i))
    return masked, used


def restore_atoms(text: str, used: Sequence[str]) -> str:
    return _PH_RE.sub(lambda m: used[int(m[1]) - 1] if 0 < int(m[1]) <= len(used) else m[0], text)


def check_rewrite(masked_in: str, out: str, n_atoms: int, mode: str) -> str | None:
    """None if the rewrite is acceptable, else the failing reason."""
    if not out.strip():
        return "empty"
    found = [int(m) for m in _PH_RE.findall(out)]
    if sorted(set(found)) != list(range(1, n_atoms + 1)) or _GLUED_RE.search(out):
        return "atoms"
    if _NEG_RE.search(masked_in) and not _NEG_OUT_RE.search(out):
        return "neg"
    return None


def _clean(raw: str) -> str:
    line = next((ln for ln in raw.strip().splitlines() if ln.strip()), "")
    line = re.sub(r"^(?:row|output|rewrite)\s*:\s*", "", line.strip(), flags=re.I)
    return line.strip().strip("`").strip()


def split_sentences(line: str) -> list[str]:
    return [p for p in _SPLIT_RE.split(line) if p]


def _skip_line(line: str) -> bool:
    return not line.strip() or bool(_LIST_RE.match(line))


def _ckey(model: str, mode: str, masked: str) -> str:
    h = hashlib.sha256(f"{model}\0{mode}\0{SLM_VERSION}\0{masked}".encode()).hexdigest()
    return h[:24]


def _cache_get(model: str, mode: str, masked: str) -> str | None:
    p = CACHE_DIR / f"{_ckey(model, mode, masked)}.json"
    return json.loads(p.read_text())["out"] if p.exists() else None


def _cache_put(model: str, mode: str, masked: str, out: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{_ckey(model, mode, masked)}.json").write_text(
        json.dumps({"model": model, "mode": mode, "v": SLM_VERSION, "in": masked, "out": out})
    )


def build_prompt(tokenizer: Any, mode: str, masked: str) -> str:
    msgs: list[dict[str, str]] = [{"role": "system", "content": SYSTEM[mode]}]
    for q, a in SHOTS[mode]:
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": masked})
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def make_hf_generator(model_id: str, mode: str, batch_size: int = 16) -> Generator:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16 if dev == "mps" else torch.float32
    ).to(dev)
    model.eval()

    def gen(masked: Sequence[str]) -> list[str]:
        order = sorted(range(len(masked)), key=lambda i: len(masked[i]))
        res = [""] * len(masked)
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            prompts = [build_prompt(tok, mode, masked[i]) for i in idx]
            enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
            cap = min(128, max(24, int(max(len(masked[i]) for i in idx) / 3)))
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=cap, do_sample=False, pad_token_id=tok.pad_token_id
                )
            for i, o in zip(idx, out, strict=True):
                res[i] = tok.decode(o[enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return res

    return gen


def rewrite_sentences(
    masked: Sequence[str], model_id: str, mode: str, generator: Generator | None, stats: Stats
) -> list[str]:
    """Raw cleaned model output per masked sentence (cache first, then one batched generate)."""
    outs: list[str | None] = [_cache_get(model_id, mode, m) for m in masked]
    missing = [i for i, o in enumerate(outs) if o is None]
    stats.cached += len(masked) - len(missing)
    if missing:
        if generator is None:
            generator = _default_generator(model_id, mode)
        import time

        t0 = time.perf_counter()
        got = generator([masked[i] for i in missing])
        stats.gen_seconds += time.perf_counter() - t0
        stats.generated += len(missing)
        for i, g in zip(missing, got, strict=True):
            outs[i] = _clean(g)
            _cache_put(model_id, mode, masked[i], outs[i] or "")
    return [o or "" for o in outs]


_GENS: dict[tuple[str, str], Generator] = {}


def _default_generator(model_id: str, mode: str) -> Generator:
    if (model_id, mode) not in _GENS:
        _GENS[(model_id, mode)] = make_hf_generator(model_id, mode)
    return _GENS[(model_id, mode)]


@dataclass
class _Sent:
    seg: int
    line: int
    raw: str
    masked: str
    used: list[str]


def slm_segments(
    segs: list[Segment],
    mode: str,
    model_id: str,
    generator: Generator | None = None,
    stats: Stats | None = None,
) -> list[Segment]:
    stats = stats if stats is not None else Stats()
    lines_by_seg: dict[int, list[str]] = {}
    units: dict[tuple[int, int], list[str | _Sent]] = {}
    work: list[_Sent] = []
    for si, s in enumerate(segs):
        if s.kind not in ("prose", "thinking"):
            continue
        lines = s.text.split("\n")
        lines_by_seg[si] = lines
        all_atoms = extract_atoms(s.text)
        fence = False
        for li, ln in enumerate(lines):
            if _FENCE_RE.match(ln):
                fence = not fence
                continue
            if fence or _skip_line(ln):
                continue
            lmasked, lused = mask_atoms(ln, [a for a in all_atoms if a in ln])
            parts: list[str | _Sent] = []
            for sent in split_sentences(lmasked):
                nums = sorted({int(n) for n in _PH_RE.findall(sent)})
                remap = {n: i for i, n in enumerate(nums, 1)}
                local = [lused[n - 1] for n in nums]
                m = _PH_RE.sub(lambda mm, r=remap: _PH.format(r[int(mm[1])]), sent)
                raw = restore_atoms(sent, lused)
                if len(_PH_RE.sub("", m).split()) < MIN_WORDS:
                    parts.append(raw)
                    continue
                st = _Sent(si, li, raw, m, local)
                parts.append(st)
                work.append(st)
            units[(si, li)] = parts
    stats.sentences += len(work)
    outs = rewrite_sentences([w.masked for w in work], model_id, mode, generator, stats)
    result: dict[int, str] = {}
    for w, o in zip(work, outs, strict=True):
        why = check_rewrite(w.masked, o, len(w.used), mode)
        new = restore_atoms(o, w.used) if why is None else w.raw
        if why is None and count_tokens(new) >= count_tokens(w.raw):
            why = "longer"
        if why == "empty":
            stats.fb_empty += 1
        elif why == "atoms":
            stats.fb_atoms += 1
        elif why == "neg":
            stats.fb_neg += 1
        elif why == "longer":
            stats.fb_longer += 1
        else:
            stats.accepted += 1
        result[id(w)] = new if why is None else w.raw
    out: list[Segment] = []
    for si, s in enumerate(segs):
        if si not in lines_by_seg:
            out.append(s)
            continue
        lines = list(lines_by_seg[si])
        for li in range(len(lines)):
            parts = units.get((si, li))
            if parts is None:
                continue
            pieces = [result[id(p)] if isinstance(p, _Sent) else p for p in parts]
            sep = "\n" if mode == "direct" else " "
            lines[li] = sep.join(pieces)
        new_text = "\n".join(lines)
        if new_text != s.text and count_tokens(new_text) < count_tokens(s.text):
            out.append(replace(s, text=new_text))
        else:
            out.append(s)
    return out


def slm_slot_segments(
    segs: list[Segment], mode: str, model_id: str, generator: Generator | None = None
) -> list[Segment]:
    return slm_segments(segs, mode, model_id, generator)

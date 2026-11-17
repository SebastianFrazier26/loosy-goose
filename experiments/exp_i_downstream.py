"""Experiment I: downstream model performance after compression.

Phase 1 and 2 measured proxies (atom recall, semantic coverage). This asks the question they stand
in for: does a model do as well from a loosy-compressed context as from the full one, and versus a
model-written summary at the same token budget (a stand-in for built-in /compact)?

Per transcript: cut late, immediately before a real assistant action; keep the last 4,000 tokens
before the cut verbatim; render the earlier stretch seven ways (full, loosy light/medium/heavy,
Sonnet summary at each loosy level's rendered size). Four answering models then (a) answer factual
probes Sonnet wrote from the earlier stretch and (b) predict the held-out next action. Sonnet
grades every answer blind to condition and model.

Usage: uv run python experiments/exp_i_downstream.py [--limit N] [--only NAME] [--workers 4]
                                                     [--dry-run] [--summarize-only]
Writes experiments/output/exp_i/: calls.jsonl (one line per finished CLI call, the checkpoint;
reruns skip every key already in it), failures.jsonl, manifest.json (the plan, read by
exp_i_status.py), contexts/<label>.json (cut and rendered contexts) and summary.txt. A run without
--limit/--only also writes docs/results/exp_i_downstream.txt. Public corpora only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tiktoken

from loosy_goose.rate import LEVELS
from loosy_goose.segment import Segment, segment
from loosy_goose.sessions.plan import PROTECT_TAIL_TOKENS
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import (
    Block,
    Transcript,
    Turn,
    load_claude_code_jsonl,
    load_messages_json,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "experiments" / "output" / "exp_i"
CACHE = ROOT / "experiments" / "output" / "cache"
RESULTS = ROOT / "docs" / "results" / "exp_i_downstream.txt"

# Part of every call key. Bump when a prompt, a renderer or the cut/selection rule changes, so old
# records stop matching instead of loading beside records that describe different inputs.
VERSION = 1

SEED = 20261014
TOKEN_RANGE = (20_000, 60_000)
N_TRANSCRIPTS = 20
PER_SOURCE = 10
CUT_WINDOW = (2 / 3, 0.95)
TAIL_TOKENS = PROTECT_TAIL_TOKENS
MIN_COMPRESSIBLE_TOKENS = 8_000
N_PROBES = 10
REFERENCE_BLOCK_CHARS = 1_500
# A summary may run a little over its budget before it is cut back to it; models count tokens
# loosely and a hard stop at exactly the budget would truncate nearly every one.
SUMMARY_OVERRUN = 1.10
SUMMARY_MIN_SHARE = 0.30
BOOTSTRAP_REPS = 2_000

CALL_TIMEOUT_S = 900
BACKOFF_START_S = 60.0
BACKOFF_MAX_S = 1_800.0
MAX_TRANSIENT_ATTEMPTS = 12
MAX_OTHER_ATTEMPTS = 3
TRANSIENT = re.compile(
    r"\b(429|529|50[0-4])\b|rate.?limit|usage.?limit|overloaded|quota|timed? ?out|timeout"
    r"|temporar|try again|ECONNRESET|connection|network|capacity",
    re.IGNORECASE,
)

LOOSY_LEVELS: dict[str, str] = {
    "loosy-light": "light",
    "loosy-medium": "medium",
    "loosy-heavy": "heavy",
}
SUMMARY_OF: dict[str, str] = {
    "summary-light": "loosy-light",
    "summary-medium": "loosy-medium",
    "summary-heavy": "loosy-heavy",
}
CONDITIONS: tuple[str, ...] = ("full", *LOOSY_LEVELS, *SUMMARY_OF)
TASKS: tuple[str, ...] = ("probe", "next")
GRADE_PROBE_SCORE: dict[str, float] = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
GRADE_NEXT_SCORE: dict[str, float] = {"same": 1.0, "partial": 0.5, "different": 0.0}

Cli = Literal["claude", "codex"]


@dataclass(frozen=True)
class Model:
    name: str
    cli: Cli
    model_id: str


MODELS: tuple[Model, ...] = (
    Model("haiku-5.5", "claude", "claude-haiku-5-5"),
    Model("sonnet-5.5", "claude", "claude-sonnet-5-5"),
    Model("gpt-6-luna", "codex", "gpt-6-luna"),
    Model("gpt-6-sol", "codex", "gpt-6-sol"),
)
MODEL_BY_NAME = {m.name: m for m in MODELS}
# Writes the probes and summaries, grades every answer, and its full-context probe answers are
# the filter that keeps a probe: one model, so a probe it cannot answer from the full context is
# a bad probe rather than a compression loss.
HELPER = MODEL_BY_NAME["sonnet-5.5"]
EFFORT = "medium"

EVAL_PREAMBLE = (
    "You are taking part in an evaluation. You are given a transcript of an earlier AI "
    "coding-agent session. Do not use any tools and do not run any commands: answer only from "
    "the text provided, in exactly the format requested."
)

CLAUDE_FLAGS: tuple[str, ...] = (
    "--effort",
    EFFORT,
    # `--tools ""` removes every built-in tool; `--safe-mode` drops CLAUDE.md, skills, plugins,
    # hooks and MCP servers while keeping OAuth (`--bare` would also isolate, but it accepts only
    # ANTHROPIC_API_KEY auth); `--no-session-persistence` keeps the call out of ~/.claude/projects.
    "--tools",
    "",
    "--safe-mode",
    "--strict-mcp-config",
    "--no-session-persistence",
    "--output-format",
    "json",
    "--system-prompt",
    EVAL_PREAMBLE,
)
# Codex has no flag that removes its own tools or the global ~/.codex/AGENTS.md (checked against
# `codex exec --help` and `codex features list`, codex-cli 0.158.0); what can be turned off is.
# `--ignore-user-config` skips config.toml (so its MCP servers), `--ephemeral` skips
# ~/.codex/sessions, and a read-only sandbox in an empty temp cwd leaves the exec tool nothing to
# touch. Tool calls that still happen are counted per record (`tool_calls`).
CODEX_FLAGS: tuple[str, ...] = (
    "-c",
    f'model_reasoning_effort="{EFFORT}"',
    "-c",
    'web_search="disabled"',
    "--ephemeral",
    "--ignore-user-config",
    "--ignore-rules",
    "--skip-git-repo-check",
    "-s",
    "read-only",
    *(
        arg
        for feature in (
            "hooks",
            "apps",
            "plugins",
            "multi_agent",
            "shell_tool",
            "unified_exec",
            "browser_use",
            "computer_use",
            "image_generation",
            "view_image",
            "goals",
            "tool_suggest",
            "skill_search",
            "sleep_tool",
        )
        for arg in ("--disable", feature)
    ),
)


# --------------------------------------------------------------------------- corpus and cut


@dataclass(frozen=True)
class Candidate:
    label: str
    source: str
    transcript: Transcript


@dataclass(frozen=True)
class Cut:
    cut: int
    end: int
    tail_from: int
    total_tokens: int
    compressible_tokens: int
    tail_tokens: int


def transcript_tokens(t: Transcript) -> int:
    """The count `token_budget` sums: segment texts, not raw blocks."""
    return sum(count_tokens(s.text) for s in segment(t))


def _turn_tokens(turn: Turn) -> int:
    return sum(count_tokens(b.text) for b in turn.blocks)


def _action_end(turns: Sequence[Turn], start: int) -> int:
    end = start
    while end < len(turns) and turns[end].role == "assistant":
        end += 1
    return end


def _is_action(turns: Sequence[Turn], start: int, end: int) -> bool:
    return any(
        b.kind == "tool_use" or (b.kind in ("text", "code") and b.text.strip())
        for t in turns[start:end]
        for b in t.blocks
    )


def tail_start(turn_tokens: Sequence[int], cut: int, tail_tokens: int = TAIL_TOKENS) -> int:
    tail_from, tail = cut, 0
    while tail < tail_tokens and tail_from > 0:
        tail_from -= 1
        tail += turn_tokens[tail_from]
    return tail_from


def find_cut(
    t: Transcript,
    rng: random.Random,
    *,
    window: tuple[float, float] = CUT_WINDOW,
    tail_tokens: int = TAIL_TOKENS,
    min_compressible: int = MIN_COMPRESSIBLE_TOKENS,
) -> Cut | None:
    """A cut at the start of an assistant action (an assistant turn after a non-assistant one)
    whose position by tokens falls in `window`. Everything before is context; the consecutive
    assistant turns from the cut are the held-out action."""
    turns = t.turns
    per_turn = [_turn_tokens(x) for x in turns]
    total = sum(per_turn)
    if total == 0:
        return None
    before = np.concatenate([[0], np.cumsum(per_turn)]).tolist()
    options: list[Cut] = []
    for i in range(1, len(turns)):
        if turns[i].role != "assistant" or turns[i - 1].role == "assistant":
            continue
        if not window[0] <= before[i] / total <= window[1]:
            continue
        end = _action_end(turns, i)
        if not _is_action(turns, i, end):
            continue
        tail_from = tail_start(per_turn, i, tail_tokens)
        compressible = int(before[tail_from])
        if compressible < min_compressible:
            continue
        options.append(
            Cut(i, end, tail_from, total, compressible, int(before[i] - before[tail_from]))
        )
    return rng.choice(options) if options else None


def cut_rng(label: str) -> random.Random:
    return random.Random(f"{SEED}:{label}")


def iter_candidates(swe_chars: tuple[int, int] = (40_000, 400_000)) -> Iterator[Candidate]:
    for p in sorted(
        (DATA / "public" / "trace-commons" / "sessions" / "claude_code").glob("*.jsonl")
    ):
        yield Candidate(f"trace-commons_{p.stem[:8]}", "trace-commons", load_claude_code_jsonl(p))
    files = sorted((DATA / "public" / "swe-gym").rglob("*.parquet"))
    if not files:
        return
    import pyarrow.parquet as pq

    table = pq.read_table(files[0])
    col = "messages" if "messages" in table.column_names else table.column_names[0]
    for i, payload in enumerate(table.column(col).to_pylist()):
        # Cheap length prefilter before tokenising: 491 rows, most far outside the token range.
        if not swe_chars[0] <= len(json.dumps(payload, default=str)) <= swe_chars[1]:
            continue
        if isinstance(payload, str):
            payload = json.loads(payload)
        yield Candidate(f"swe-gym_row{i}", "swe-gym", load_messages_json(payload, name=f"swe-{i}"))


def select_transcripts(
    candidates: Iterable[Candidate],
    *,
    token_range: tuple[int, int] = TOKEN_RANGE,
    n: int = N_TRANSCRIPTS,
    per_source: int = PER_SOURCE,
    seed: int = SEED,
    tail_tokens: int = TAIL_TOKENS,
    min_compressible: int = MIN_COMPRESSIBLE_TOKENS,
) -> list[tuple[Candidate, Cut]]:
    """Eligible = in the token range with a valid cut. Up to `per_source` drawn per source with a
    fixed seed, the shortfall filled from the other sources' leftovers. Returned interleaved by
    source, so `--limit 2` takes one of each."""
    eligible: dict[str, list[tuple[Candidate, Cut]]] = defaultdict(list)
    for c in candidates:
        tokens = transcript_tokens(c.transcript)
        if not token_range[0] <= tokens <= token_range[1]:
            continue
        cut = find_cut(
            c.transcript,
            cut_rng(c.label),
            tail_tokens=tail_tokens,
            min_compressible=min_compressible,
        )
        if cut is not None:
            eligible[c.source].append((c, cut))
    rng = random.Random(seed)
    picked: dict[str, list[tuple[Candidate, Cut]]] = {}
    leftovers: list[tuple[Candidate, Cut]] = []
    for source in sorted(eligible):
        pool = sorted(eligible[source], key=lambda x: x[0].label)
        rng.shuffle(pool)
        picked[source] = pool[:per_source]
        leftovers.extend(pool[per_source:])
    room = n - sum(len(v) for v in picked.values())
    for c, cut in leftovers[: max(room, 0)]:
        picked[c.source].append((c, cut))
    out: list[tuple[Candidate, Cut]] = []
    queues = [list(picked[s]) for s in sorted(picked)]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out[:n]


# --------------------------------------------------------------------------- rendering


def _render_block(kind: str, text: str, name: str | None) -> str:
    if kind == "tool_use":
        return f'<tool_use name="{name or "?"}">\n{text}\n</tool_use>'
    if kind == "tool_result":
        return f"<tool_result>\n{text}\n</tool_result>"
    if kind == "thinking":
        return f"<thinking>\n{text}\n</thinking>"
    if kind == "code":
        return f"```\n{text}\n```"
    return text


def render_turns(turns: Iterable[Turn]) -> str:
    return "\n\n".join(
        f"### {t.role}\n"
        + "\n\n".join(_render_block(b.kind, b.text, b.meta.get("name")) for b in t.blocks)
        for t in turns
    )


_SEGMENT_BLOCK_KIND = {"prose": "text"}


def render_segments(segments: Sequence[Segment]) -> str:
    """Kept segments regrouped by turn and block. A gap between kept segments of the same block is
    marked, so a partly dropped tool result does not read as complete."""
    parts: list[str] = []
    ordered = sorted(segments, key=lambda s: s.id)
    i = 0
    while i < len(ordered):
        turn = ordered[i].turn
        blocks: list[str] = []
        while i < len(ordered) and ordered[i].turn == turn:
            first = ordered[i]
            pieces = [first.text]
            last_id = first.id
            i += 1
            while i < len(ordered) and (ordered[i].turn, ordered[i].block) == (turn, first.block):
                if ordered[i].id != last_id + 1:
                    pieces.append("[…]")
                pieces.append(ordered[i].text)
                last_id = ordered[i].id
                i += 1
            sep = "\n" if first.kind == "tool_result" else "\n\n"
            kind = _SEGMENT_BLOCK_KIND.get(first.kind, first.kind)
            blocks.append(_render_block(kind, sep.join(pieces), first.tool_name))
        parts.append(f"### {first.role}\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)


EARLIER_HEADER: dict[str, str] = {
    "full": "[Earlier part of the session]",
    "loosy": (
        "[Earlier part of the session, compressed extractively: some passages were dropped, "
        "what remains is verbatim]"
    ),
    "summary": "[Earlier part of the session, replaced by a summary]",
}
TAIL_HEADER = "[Most recent part of the session, verbatim]"


def assemble(kind: str, earlier: str, tail: str) -> str:
    return f"{EARLIER_HEADER[kind]}\n\n{earlier}\n\n{TAIL_HEADER}\n\n{tail}"


def reference_action(turns: Sequence[Turn]) -> str:
    def clip(text: str) -> str:
        return text if len(text) <= REFERENCE_BLOCK_CHARS else text[:REFERENCE_BLOCK_CHARS] + "…"

    kept = [
        Turn(
            t.index,
            t.role,
            [Block(b.kind, clip(b.text), b.meta) for b in t.blocks if b.kind != "thinking"],
        )
        for t in turns
    ]
    return render_turns([t for t in kept if t.blocks])


def truncate_tokens(text: str, budget: int) -> tuple[str, bool]:
    enc = tiktoken.get_encoding("o200k_base")
    ids = enc.encode(text, disallowed_special=())
    if len(ids) <= budget * SUMMARY_OVERRUN:
        return text, False
    return enc.decode(ids[:budget]), True


Compressor = Callable[[Transcript, float], list[Segment]]


def loosy_compressor(t: Transcript, keep: float) -> list[Segment]:
    from loosy_goose.pipeline import PipelineOptions, compress_transcript

    return compress_transcript(t, PipelineOptions(keep=keep), cache_dir=CACHE).segments


@dataclass
class Prepared:
    label: str
    source: str
    cut: Cut
    earlier: dict[str, str]
    tail: str
    full_earlier: str
    reference: str
    compressible_segment_tokens: int
    kept_segment_tokens: dict[str, int] = field(default_factory=dict)

    def summary_budget(self, condition: str) -> int:
        return count_tokens(self.earlier[SUMMARY_OF[condition]])


def prepare(cand: Candidate, cut: Cut, compressor: Compressor = loosy_compressor) -> Prepared:
    turns = cand.transcript.turns
    compressible = Transcript(f"{cand.label}#compressible", list(turns[: cut.tail_from]))
    tail = render_turns(turns[cut.tail_from : cut.cut])
    full_earlier = render_turns(compressible.turns)
    earlier = {"full": full_earlier}
    kept_tokens: dict[str, int] = {}
    for cond, level in LOOSY_LEVELS.items():
        kept = compressor(compressible, LEVELS[level])
        earlier[cond] = render_segments(kept)
        kept_tokens[cond] = sum(count_tokens(s.text) for s in kept)
    return Prepared(
        label=cand.label,
        source=cand.source,
        cut=cut,
        earlier=earlier,
        tail=tail,
        full_earlier=full_earlier,
        reference=reference_action(turns[cut.cut : cut.end]),
        compressible_segment_tokens=transcript_tokens(compressible),
        kept_segment_tokens=kept_tokens,
    )


def context_for(p: Prepared, condition: str, summaries: Mapping[str, str]) -> str | None:
    if condition == "full":
        return assemble("full", p.full_earlier, p.tail)
    if condition in LOOSY_LEVELS:
        return assemble("loosy", p.earlier[condition], p.tail)
    summary = summaries.get(condition)
    return None if summary is None else assemble("summary", summary, p.tail)


def _prepared_to_json(p: Prepared) -> dict[str, Any]:
    d = asdict(p)
    d["version"] = VERSION
    return d


def _prepared_from_json(d: dict[str, Any]) -> Prepared:
    d = dict(d)
    d.pop("version", None)
    d["cut"] = Cut(**d["cut"])
    return Prepared(**d)


def load_or_prepare(
    cand: Candidate, cut: Cut, out: Path, compressor: Compressor = loosy_compressor
) -> Prepared:
    path = out / "contexts" / f"{cand.label}.json"
    if path.exists():
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("version") == VERSION and d.get("cut") == asdict(cut):
                return _prepared_from_json(d)
        except (json.JSONDecodeError, OSError, TypeError, KeyError):
            pass
    p = prepare(cand, cut, compressor)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_prepared_to_json(p)), encoding="utf-8")
    os.replace(tmp, path)
    return p


# --------------------------------------------------------------------------- prompts and parsing


def summary_prompt(earlier: str, budget: int) -> str:
    # The length is a requirement placed after the session, not a cap placed before it: asked for
    # "at most N tokens" up front, Sonnet wrote ~1,000 tokens against 3.5k-13k budgets in the
    # first smoke run (2026-10-14), which would have compared loosy with a summary a fraction of
    # its size. This form reached ~76% of budget on the same input.
    words = int(budget * 0.7)
    return (
        "Below is the earlier part of an AI coding-agent session.\n\n"
        f"<earlier_session>\n{earlier}\n</earlier_session>\n\n"
        "Write a detailed summary that will replace the session above in the agent's context so "
        "it can continue the work (the most recent part is kept separately and is not shown). "
        "Preserve concrete details: file paths, identifiers, values, error messages, commands run "
        "and their results, decisions made, and the current state of the task. "
        f"Required length: about {words} words (between {int(words * 0.8)} and {words} words). "
        f"This is a hard requirement: a summary much shorter than {int(words * 0.8)} words is a "
        "failure, and one longer than the maximum will be cut. Go through the session in order. "
        "Output only the summary."
    )


def summary_ok(text: str, budget: int) -> bool:
    # Catches refusals and stubs: in the smoke run Sonnet once declined outright in 56 tokens,
    # and answers built on that "summary" were scored as if it were one.
    return count_tokens(text) >= SUMMARY_MIN_SHARE * budget


def probe_gen_prompt(earlier: str, n: int = N_PROBES) -> str:
    return (
        "Below is the earlier part of an AI coding-agent session. Write exactly "
        f"{n} short factual questions about specific details stated in it, each with a short "
        "checkable answer (at most 12 words): file paths, values, error messages, identifiers, "
        "commands, decisions and their reasons. Spread the questions across the whole text, not "
        "just the start. Each must be answerable from the text alone, with one unambiguous "
        "answer.\n"
        'Return only JSON: {"probes": [{"q": "...", "a": "..."}, ...]}\n\n'
        f"<earlier_session>\n{earlier}\n</earlier_session>"
    )


def probe_answer_prompt(context: str, questions: Sequence[str]) -> str:
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    return (
        f"{EVAL_PREAMBLE}\n\n<transcript>\n{context}\n</transcript>\n\n"
        "Answer each question from the transcript above. Keep each answer short (at most 12 "
        "words). If the transcript does not say, answer UNKNOWN.\n\n"
        f"{numbered}\n\n"
        'Return only JSON: {"answers": {"1": "...", "2": "...", ...}}'
    )


def next_step_prompt(context: str) -> str:
    return (
        f"{EVAL_PREAMBLE}\n\n<transcript>\n{context}\n</transcript>\n\n"
        "You are the assistant in the session above, and the transcript ends where you act next. "
        "Do not take the action; describe the single next action you would take now, briefly, in "
        "exactly this format:\n"
        "ACTION: <tool name, or 'reply' if you would answer in text>\n"
        "TARGET: <file path, command or arguments, briefly>\n"
        "INTENT: <one sentence>"
    )


def grade_probe_prompt(items: Sequence[tuple[str, str, str]]) -> str:
    body = "\n\n".join(
        f"{i}.\nQuestion: {q}\nReference answer: {a}\nCandidate answer: {c}"
        for i, (q, a, c) in enumerate(items, 1)
    )
    return (
        "Grade each candidate answer against the reference answer. correct: it states the same "
        "fact as the reference (wording may differ). partial: it is partly right or less "
        "specific than the reference. wrong: it is wrong, missing, or UNKNOWN.\n\n"
        f"{body}\n\n"
        'Return only JSON: {"grades": {"1": "correct|partial|wrong", ...}}'
    )


def grade_next_prompt(reference: str, predicted: str) -> str:
    return (
        "An AI coding agent was at a point in a session. Below is the action it really took "
        "next, and a predicted next action. Grade whether the prediction agrees with the real "
        "action. same: same kind of action (same tool) on the same target with the same intent. "
        "partial: same kind of action or same target, but they differ materially. different: "
        "otherwise.\n\n"
        f"<real_action>\n{reference}\n</real_action>\n\n"
        f"<predicted_action>\n{predicted}\n</predicted_action>\n\n"
        'Return only JSON: {"grade": "same|partial|different", "why": "<at most 12 words>"}'
    )


def parse_json_object(text: str) -> dict[str, Any] | None:
    """The outermost {...} in a reply, tolerating code fences and chatter around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_probes(text: str) -> list[dict[str, str]] | None:
    obj = parse_json_object(text)
    if obj is None or not isinstance(obj.get("probes"), list):
        return None
    probes = [
        {"q": str(p["q"]).strip(), "a": str(p["a"]).strip()}
        for p in obj["probes"]
        if isinstance(p, dict) and p.get("q") and p.get("a")
    ]
    return probes if len(probes) >= 3 else None


_NUMBERED = re.compile(r"^\s*(\d+)[.):]\s*(.+?)\s*$", re.MULTILINE)


def parse_answers(text: str, n: int) -> list[str] | None:
    obj = parse_json_object(text)
    found: dict[int, str] = {}
    if obj is not None and isinstance(obj.get("answers"), dict):
        for k, v in obj["answers"].items():
            if str(k).strip().isdigit():
                found[int(k)] = str(v)
    else:
        found = {int(m.group(1)): m.group(2) for m in _NUMBERED.finditer(text)}
    if not found:
        return None
    return [found.get(i, "") for i in range(1, n + 1)]


def parse_probe_grades(text: str, n: int) -> list[float] | None:
    obj = parse_json_object(text)
    if obj is None or not isinstance(obj.get("grades"), dict):
        return None
    grades = {str(k).strip(): str(v).strip().lower() for k, v in obj["grades"].items()}
    out: list[float] = []
    for i in range(1, n + 1):
        g = grades.get(str(i))
        if g not in GRADE_PROBE_SCORE:
            return None
        out.append(GRADE_PROBE_SCORE[g])
    return out


def parse_next_grade(text: str) -> float | None:
    obj = parse_json_object(text)
    if obj is not None:
        g = str(obj.get("grade", "")).strip().lower()
        if g in GRADE_NEXT_SCORE:
            return GRADE_NEXT_SCORE[g]
    return None


# --------------------------------------------------------------------------- CLI calls


@dataclass(frozen=True)
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    tool_calls: int = 0


class CallError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.transient = bool(TRANSIENT.search(message))


class CallFailedError(Exception):
    pass


Runner = Callable[[Model, str], CallResult]


def claude_argv(model: Model) -> list[str]:
    return ["claude", "-p", "--model", model.model_id, *CLAUDE_FLAGS]


def codex_argv(model: Model, cwd: Path, last_message: Path) -> list[str]:
    return [
        "codex",
        "exec",
        "-m",
        model.model_id,
        *CODEX_FLAGS,
        "-C",
        str(cwd),
        "-o",
        str(last_message),
        "--json",
        "-",
    ]


def parse_claude_output(stdout: str, returncode: int) -> CallResult:
    lines = [ln for ln in stdout.splitlines() if ln.lstrip().startswith("{")]
    try:
        d = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        d = None
    if not isinstance(d, dict):
        raise CallError(f"claude rc={returncode}, unparseable output: {stdout[-300:]!r}")
    if d.get("is_error") or returncode != 0:
        raise CallError(
            f"claude rc={returncode} status={d.get('api_error_status')}: {d.get('result')!r}"
        )
    u = d.get("usage") or {}
    inp = sum(
        int(u.get(k) or 0)
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    return CallResult(str(d.get("result") or ""), inp, int(u.get("output_tokens") or 0))


def parse_codex_output(events: str, last_message: str | None, returncode: int) -> CallResult:
    inp = out = tools = 0
    errors: list[str] = []
    agent_text = ""
    for line in events.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        kind = ev.get("type")
        if kind == "turn.completed":
            u = ev.get("usage") or {}
            inp += int(u.get("input_tokens") or 0)
            out += int(u.get("output_tokens") or 0) + int(u.get("reasoning_output_tokens") or 0)
        elif kind in ("error", "turn.failed"):
            err = ev.get("error")
            errors.append(str(err.get("message") if isinstance(err, dict) else ev.get("message")))
        elif kind == "item.completed":
            item = ev.get("item") or {}
            if item.get("type") == "agent_message":
                agent_text = str(item.get("text") or "")
            elif item.get("type") not in ("reasoning",):
                tools += 1
    text = last_message if last_message is not None else agent_text
    if errors or returncode != 0 or not text.strip():
        raise CallError(f"codex rc={returncode}: {' | '.join(errors)[-500:] or 'empty reply'}")
    return CallResult(text, inp, out, tools)


def cli_runner(model: Model, prompt: str) -> CallResult:
    """One fresh empty temp dir per call, so neither CLI sees a project or a prior call's files."""
    with tempfile.TemporaryDirectory(prefix="loosy-exp-i-") as tmp:
        cwd = Path(tmp) / "cwd"
        cwd.mkdir()
        last = Path(tmp) / "last-message.txt"
        argv = claude_argv(model) if model.cli == "claude" else codex_argv(model, cwd, last)
        try:
            done = subprocess.run(
                argv,
                cwd=cwd,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CALL_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CallError(f"timeout after {CALL_TIMEOUT_S}s") from exc
        except OSError as exc:
            raise CallError(f"could not start {argv[0]}: {exc}") from exc
        if model.cli == "claude":
            return parse_claude_output(done.stdout + "\n" + done.stderr, done.returncode)
        reply = last.read_text(encoding="utf-8") if last.exists() else None
        try:
            return parse_codex_output(done.stdout, reply, done.returncode)
        except CallError as exc:
            raise CallError(f"{exc} stderr={done.stderr[-300:]!r}") from exc


def call_with_retry(
    runner: Runner,
    model: Model,
    prompt: str,
    validate: Callable[[str], bool] | None = None,
    *,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> tuple[CallResult, int]:
    """Usage-limit and rate errors back off exponentially up to ~30 min and keep trying; other
    errors and invalid replies get a few quick retries. Raises CallFailedError when out of
    attempts — the caller logs it and moves on, and a rerun picks the key up again."""
    transient = other = 0
    delay = BACKOFF_START_S
    while True:
        attempts = transient + other + 1
        try:
            result = runner(model, prompt)
        except CallError as exc:
            if exc.transient and transient + 1 < MAX_TRANSIENT_ATTEMPTS:
                transient += 1
                log(f"  transient error ({exc}); sleeping {delay:.0f}s")
                sleep(delay)
                delay = min(delay * 2, BACKOFF_MAX_S)
                continue
            if not exc.transient and other + 1 < MAX_OTHER_ATTEMPTS:
                other += 1
                sleep(10.0)
                continue
            raise CallFailedError(str(exc)) from exc
        if validate is not None and not validate(result.text):
            if other + 1 < MAX_OTHER_ATTEMPTS:
                other += 1
                continue
            raise CallFailedError(f"invalid reply after {attempts} attempts: {result.text[:200]!r}")
        return result, attempts


# --------------------------------------------------------------------------- checkpoint


def call_key(transcript: str, condition: str, model: str, task: str) -> str:
    return f"v{VERSION}|{transcript}|{condition}|{model}|{task}"


class Store:
    """Append-only JSONL checkpoint keyed by call_key. A torn last line (killed mid-write) is
    skipped on load, which only costs that one call on the rerun."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict) and "key" in rec:
                        self.records[rec["key"]] = rec

    def get(self, key: str) -> dict[str, Any] | None:
        return self.records.get(key)

    def put(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.records[rec["key"]] = rec


@dataclass
class Job:
    transcript: str
    condition: str
    model: Model
    task: str
    prompt: str
    validate: Callable[[str], bool] | None = None
    # Who answers the call. Grading jobs are keyed by the graded answer's model but run as HELPER.
    caller: Model | None = None

    @property
    def key(self) -> str:
        return call_key(self.transcript, self.condition, self.model.name, self.task)

    @property
    def runs_as(self) -> Model:
        return self.caller or self.model

    @property
    def prompt_sha(self) -> str:
        return hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()[:16]

    def done_by(self, rec: dict[str, Any] | None) -> bool:
        """A stored record counts only if it answered this exact prompt and still validates.
        The key alone cannot tell: an answer built on a summary that has since been redone has the
        same key but a different prompt, and must be asked again (as must its grading)."""
        if rec is None or rec.get("prompt_sha") != self.prompt_sha:
            return False
        return self.validate is None or self.validate(str(rec["text"]))


@dataclass
class Runtime:
    store: Store
    runner: Runner
    failures: Path
    workers: int = 4
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = print
    new_calls: int = 0
    failed: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, jobs: Sequence[Job]) -> None:
        todo = [j for j in jobs if not j.done_by(self.store.get(j.key))]
        if not todo:
            return
        if self.workers <= 1:
            for j in todo:
                self._one(j)
            return
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self._one, todo))

    def _one(self, job: Job) -> None:
        t0 = time.time()
        try:
            result, attempts = call_with_retry(
                self.runner, job.runs_as, job.prompt, job.validate, sleep=self.sleep, log=self.log
            )
        except CallFailedError as exc:
            with self._lock:
                self.failed += 1
                self.failures.parent.mkdir(parents=True, exist_ok=True)
                with self.failures.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps({"key": job.key, "error": str(exc), "ts": time.time()}) + "\n"
                    )
            self.log(f"{time.strftime('%H:%M:%S')} FAILED {job.key}: {exc}")
            return
        wall = time.time() - t0
        self.store.put(
            {
                "key": job.key,
                "prompt_sha": job.prompt_sha,
                "version": VERSION,
                "transcript": job.transcript,
                "condition": job.condition,
                "model": job.model.name,
                "task": job.task,
                "caller": job.runs_as.name,
                "text": result.text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "tool_calls": result.tool_calls,
                "attempts": attempts,
                "wall_s": round(wall, 2),
                "ts": time.time(),
            }
        )
        with self._lock:
            self.new_calls += 1
        self.log(
            f"{time.strftime('%H:%M:%S')} ok {job.key} {wall:.1f}s "
            f"in={result.input_tokens} out={result.output_tokens}"
        )


# --------------------------------------------------------------------------- stages


def _text(store: Store, transcript: str, condition: str, model: str, task: str) -> str | None:
    rec = store.get(call_key(transcript, condition, model, task))
    return None if rec is None else str(rec["text"])


def probes_for(store: Store, label: str) -> list[dict[str, str]] | None:
    text = _text(store, label, "-", HELPER.name, "probe_gen")
    return None if text is None else parse_probes(text)


def summaries_for(store: Store, p: Prepared) -> dict[str, str]:
    out: dict[str, str] = {}
    for cond in SUMMARY_OF:
        text = _text(store, p.label, cond, HELPER.name, "summary")
        budget = p.summary_budget(cond)
        if text is not None and summary_ok(text, budget):
            out[cond] = truncate_tokens(text.strip(), budget)[0]
    return out


def stage1_jobs(prepared: Sequence[Prepared]) -> list[Job]:
    jobs: list[Job] = []
    for p in prepared:
        jobs.append(
            Job(
                p.label,
                "-",
                HELPER,
                "probe_gen",
                probe_gen_prompt(p.full_earlier),
                lambda t: parse_probes(t) is not None,
            )
        )
        for cond in SUMMARY_OF:
            budget = p.summary_budget(cond)
            jobs.append(
                Job(
                    p.label,
                    cond,
                    HELPER,
                    "summary",
                    summary_prompt(p.full_earlier, budget),
                    lambda t, b=budget: summary_ok(t, b),  # type: ignore[misc]
                )
            )
    return jobs


def stage2_jobs(store: Store, prepared: Sequence[Prepared]) -> list[Job]:
    jobs: list[Job] = []
    for p in prepared:
        probes = probes_for(store, p.label)
        summaries = summaries_for(store, p)
        for cond in CONDITIONS:
            context = context_for(p, cond, summaries)
            if context is None:
                continue
            for model in MODELS:
                if probes is not None:
                    n = len(probes)
                    jobs.append(
                        Job(
                            p.label,
                            cond,
                            model,
                            "probe",
                            probe_answer_prompt(context, [x["q"] for x in probes]),
                            lambda t, n=n: parse_answers(t, n) is not None,  # type: ignore[misc]
                        )
                    )
                jobs.append(Job(p.label, cond, model, "next", next_step_prompt(context)))
    # Interleave models so a pool of workers spreads load across both CLIs and all four models.
    return sorted(jobs, key=lambda j: (j.transcript, j.condition, j.task, j.model.name))


def stage3_jobs(store: Store, prepared: Sequence[Prepared]) -> list[Job]:
    jobs: list[Job] = []
    for p in prepared:
        probes = probes_for(store, p.label)
        for cond in CONDITIONS:
            for model in MODELS:
                if probes is not None:
                    text = _text(store, p.label, cond, model.name, "probe")
                    if text is not None:
                        n = len(probes)
                        answers = parse_answers(text, n) or [""] * n
                        items = [
                            (x["q"], x["a"], a or "(no answer)")
                            for x, a in zip(probes, answers, strict=True)
                        ]
                        jobs.append(
                            Job(
                                p.label,
                                cond,
                                model,
                                "grade_probe",
                                grade_probe_prompt(items),
                                lambda t, n=n: parse_probe_grades(t, n) is not None,  # type: ignore[misc]
                                caller=HELPER,
                            )
                        )
                text = _text(store, p.label, cond, model.name, "next")
                if text is not None:
                    jobs.append(
                        Job(
                            p.label,
                            cond,
                            model,
                            "grade_next",
                            grade_next_prompt(p.reference, text.strip()),
                            lambda t: parse_next_grade(t) is not None,
                            caller=HELPER,
                        )
                    )
    return jobs


def planned_keys(labels: Sequence[str]) -> dict[str, list[str]]:
    plan: dict[str, list[str]] = defaultdict(list)
    for label in labels:
        plan["probe_gen"].append(call_key(label, "-", HELPER.name, "probe_gen"))
        for cond in SUMMARY_OF:
            plan["summary"].append(call_key(label, cond, HELPER.name, "summary"))
        for cond in CONDITIONS:
            for m in MODELS:
                for task in ("probe", "next", "grade_probe", "grade_next"):
                    plan[task].append(call_key(label, cond, m.name, task))
    return dict(plan)


# --------------------------------------------------------------------------- aggregation


def bootstrap_ci(
    values: Sequence[float], reps: int = BOOTSTRAP_REPS, seed: int = SEED
) -> tuple[float, float, float]:
    """Mean and 95% percentile CI, resampling transcripts (the unit that varies)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, arr.size, size=(reps, arr.size))].mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


Scores = dict[tuple[str, str], dict[str, float]]


def collect_scores(store: Store, labels: Sequence[str]) -> tuple[Scores, Scores, dict[str, int]]:
    """Per (condition, model): transcript -> score, for each task. A probe counts only if HELPER
    answered it correctly from the full context; a transcript with no such probe drops out of
    the probe task."""
    probe: Scores = defaultdict(dict)
    nxt: Scores = defaultdict(dict)
    kept_probes: dict[str, int] = {}
    for label in labels:
        probes = probes_for(store, label)
        if probes is not None:
            n = len(probes)
            ref_text = _text(store, label, "full", HELPER.name, "grade_probe")
            ref = parse_probe_grades(ref_text, n) if ref_text is not None else None
            keep = [i for i, g in enumerate(ref or []) if g == 1.0]
            kept_probes[label] = len(keep)
            if keep:
                for cond in CONDITIONS:
                    for m in MODELS:
                        text = _text(store, label, cond, m.name, "grade_probe")
                        grades = parse_probe_grades(text, n) if text is not None else None
                        if grades is not None:
                            probe[(cond, m.name)][label] = float(np.mean([grades[i] for i in keep]))
        for cond in CONDITIONS:
            for m in MODELS:
                text = _text(store, label, cond, m.name, "grade_next")
                g = parse_next_grade(text) if text is not None else None
                if g is not None:
                    nxt[(cond, m.name)][label] = g
    return probe, nxt, kept_probes


def _cell(scores: Mapping[str, float]) -> str:
    if not scores:
        return "—"
    mean, lo, hi = bootstrap_ci(list(scores.values()))
    return f"{mean:.2f} [{lo:.2f},{hi:.2f}] n={len(scores)}"


def summarize(store: Store, prepared: Sequence[Prepared]) -> str:
    labels = [p.label for p in prepared]
    probe, nxt, kept_probes = collect_scores(store, labels)
    lines = [
        "Experiment I — downstream performance after compression",
        f"version {VERSION}; {len(labels)} transcripts "
        f"({sum(p.source == 'trace-commons' for p in prepared)} trace-commons, "
        f"{sum(p.source == 'swe-gym' for p in prepared)} swe-gym); "
        f"cells: mean [95% bootstrap CI over transcripts] n=transcripts",
        "",
    ]
    colw = 30
    header = f"{'condition':<16}" + "".join(f"{m.name:<{colw}}" for m in MODELS)
    for title, scores in (
        ("Probe QA accuracy (probes Sonnet answers correctly from full context)", probe),
        ("Next-step agreement with the real action (same=1, partial=0.5)", nxt),
    ):
        lines += [title, header]
        for cond in CONDITIONS:
            lines.append(
                f"{cond:<16}"
                + "".join(f"{_cell(scores.get((cond, m.name), {})):<{colw}}" for m in MODELS)
            )
        lines.append("")
    lines.append(
        "Probes kept after the full-context filter: "
        + ", ".join(f"{k}={v}" for k, v in sorted(kept_probes.items()))
        if kept_probes
        else "Probes kept: none yet"
    )
    lines += ["", "Tokens of the earlier stretch as rendered (tail always verbatim, not counted)"]
    lines.append(f"{'condition':<16}{'mean tokens':>12}{'share of full':>15}")
    for cond in CONDITIONS:
        toks: list[float] = []
        shares: list[float] = []
        for p in prepared:
            full = count_tokens(p.full_earlier)
            if cond in SUMMARY_OF:
                text = summaries_for(store, p).get(cond)
                if text is None:
                    continue
            else:
                text = p.earlier[cond]
            n = count_tokens(text)
            toks.append(n)
            shares.append(n / full if full else float("nan"))
        if toks:
            lines.append(f"{cond:<16}{np.mean(toks):>12.0f}{np.mean(shares):>15.3f}")
    tails = [p.cut.tail_tokens for p in prepared]
    if tails:
        lines.append(f"verbatim tail: mean {np.mean(tails):.0f} tokens")
    lines += ["", _usage_block(store, labels)]
    return "\n".join(lines) + "\n"


def _usage_block(store: Store, labels: Sequence[str]) -> str:
    wanted = set(labels)
    by: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0, 0, 0, 0.0])
    tools = 0
    for rec in store.records.values():
        if rec.get("version") != VERSION or rec.get("transcript") not in wanted:
            continue
        row = by[(str(rec.get("caller", rec["model"])), str(rec["task"]))]
        row[0] += 1
        row[1] += rec.get("input_tokens", 0)
        row[2] += rec.get("output_tokens", 0)
        row[3] += rec.get("wall_s", 0.0)
        tools += int(rec.get("tool_calls", 0))
    out = ["Calls and tokens, CLI-reported (codex input includes its own ~10-16k prompt per call)"]
    out.append(
        f"{'caller':<12}{'task':<13}{'calls':>7}{'input tok':>14}{'output tok':>12}{'call-h':>8}"
    )
    for (caller, task), (c, i, o, w) in sorted(by.items()):
        out.append(f"{caller:<12}{task:<13}{int(c):>7}{int(i):>14,}{int(o):>12,}{w / 3600:>8.2f}")
    out.append(f"codex tool calls made despite the no-tools instruction: {tools}")
    return "\n".join(out)


# --------------------------------------------------------------------------- main


def _log(msg: str) -> None:
    print(msg, flush=True)


def run(
    selected: Sequence[tuple[Candidate, Cut]],
    rt: Runtime,
    out: Path,
    compressor: Compressor = loosy_compressor,
    *,
    dry_run: bool = False,
    summarize_only: bool = False,
) -> tuple[list[Prepared], str]:
    prepared: list[Prepared] = []
    for cand, cut in selected:
        t0 = time.time()
        prepared.append(load_or_prepare(cand, cut, out, compressor))
        rt.log(
            f"{time.strftime('%H:%M:%S')} [{cand.label}] prepared in {time.time() - t0:.1f}s: "
            f"{cut.total_tokens} tokens, cut at turn {cut.cut}/{len(cand.transcript.turns)}, "
            f"compressible {cut.compressible_tokens}, tail {cut.tail_tokens}"
        )
    labels = [p.label for p in prepared]
    manifest = {
        "version": VERSION,
        "started": time.time(),
        "pid": os.getpid(),
        "workers": rt.workers,
        "transcripts": [{"label": p.label, "source": p.source, **asdict(p.cut)} for p in prepared],
        "planned": planned_keys(labels),
    }
    if not dry_run and not summarize_only:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        for name, build in (
            ("stage 1: probes + summaries", lambda: stage1_jobs(prepared)),
            ("stage 2: answers", lambda: stage2_jobs(rt.store, prepared)),
            ("stage 3: grading", lambda: stage3_jobs(rt.store, prepared)),
        ):
            jobs = build()
            todo = sum(1 for j in jobs if not j.done_by(rt.store.get(j.key)))
            rt.log(f"{time.strftime('%H:%M:%S')} {name}: {len(jobs)} jobs, {todo} to run")
            rt.run(jobs)
    elif dry_run:
        rt.log(json.dumps({k: len(v) for k, v in manifest["planned"].items()}))
    return prepared, summarize(rt.store, prepared)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="first N selected transcripts")
    parser.add_argument("--only", type=str, default=None, help="labels containing this")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="select and prepare, no calls")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    selected = select_transcripts(iter_candidates())
    _log(
        f"{time.strftime('%H:%M:%S')} selected {len(selected)} transcripts in "
        f"{time.time() - t0:.0f}s: {', '.join(c.label for c, _ in selected)}"
    )
    if args.only:
        selected = [x for x in selected if args.only in x[0].label]
    if args.limit is not None:
        selected = selected[: args.limit]
    rt = Runtime(Store(OUT / "calls.jsonl"), cli_runner, OUT / "failures.jsonl", args.workers)
    rt.log = _log
    _, text = run(selected, rt, OUT, dry_run=args.dry_run, summarize_only=args.summarize_only)
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    partial = args.limit is not None or args.only is not None
    if not partial and not args.dry_run:
        RESULTS.write_text(text, encoding="utf-8")
    print(text)
    _log(
        f"{time.strftime('%H:%M:%S')} done in {(time.time() - t0) / 60:.1f} min: "
        f"{rt.new_calls} new calls, {rt.failed} failed"
        + (" (see failures.jsonl; rerun retries them)" if rt.failed else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

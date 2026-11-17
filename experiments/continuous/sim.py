"""Arm simulation for the continuous-compaction pilot (spec 2026-11-16-continuous-compact-design).

Four arms replay a transcript turn by turn: `full`, `thresh` (threshold summarisation), `crec`
(loosy recomputed over the whole pre-tail history at every turn) and `cfrz` (loosy applied once per
F-token chunk, appended to a frozen prefix). A "boundary" is a history length `t` at which the
model is called; each arm yields a `View` per boundary, from which both the context a model would
be shown at a checkpoint and the offline cost metrics are derived.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from experiments import exp_i_downstream as exi
from loosy_goose.rate import LEVELS
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Transcript, Turn

# Covers the freeze, recompute and threshold simulations; goes into every record key.
CONT_VERSION = 1

KEEP = LEVELS["medium"]
FREEZE_TOKENS = 4_000
CAP_SHARE = 0.40
SUMMARY_SHARE = 0.33
CHECKPOINT_SHARES = (0.5, 0.75, 1.0)
ARMS = ("full", "thresh", "crec", "cfrz")
E_SEED = exi.SEED + 4
E_N = 10
MIN_THRESH_EVENTS = 2

HEADERS = {
    "full": exi.EARLIER_HEADER["full"],
    "crec": exi.EARLIER_HEADER["loosy"],
    "thresh": "[Earlier part of the session: a summary of its older part, then the rest verbatim]",
    "cfrz": (
        "[Earlier part of the session: its older part compressed extractively (some passages "
        "were dropped, what remains is verbatim), then the rest verbatim]"
    ),
}

Summarizer = Callable[[int, str, int], str]
Compressor = exi.Compressor


class MissingSummaryError(Exception):
    def __init__(self, k: int, text: str, budget: int) -> None:
        super().__init__(f"no summary for compaction event {k}")
        self.k, self.text, self.budget = k, text, budget


@dataclass(frozen=True)
class Checkpoint:
    pct: int
    t: int
    tail_from: int


@dataclass
class TranscriptPlan:
    label: str
    turns: list[Turn]
    per_turn: list[int]
    before: list[int]
    boundaries: list[int]
    total: int
    cap: int
    target: int
    checkpoints: list[Checkpoint]


@dataclass(frozen=True)
class View:
    """`compressed` is the arm's non-verbatim text; turns[verbatim_from:t] follow it verbatim, of
    which turns[tail_from:t] is the protected tail."""

    header: str
    compressed: str
    verbatim_from: int
    tail_from: int


@dataclass(frozen=True)
class Event:
    t: int
    tail_from: int


@dataclass
class ArmRun:
    arm: str
    boundaries: list[int] = field(default_factory=list)
    ctx_tokens: list[int] = field(default_factory=list)
    uncached_tokens: list[int] = field(default_factory=list)
    contexts: dict[int, str] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "boundaries": self.boundaries,
            "ctx_tokens": self.ctx_tokens,
            "uncached_tokens": self.uncached_tokens,
            "contexts": {str(t): c for t, c in self.contexts.items()},
            "events": [[e.t, e.tail_from] for e in self.events],
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> ArmRun:
        return ArmRun(
            d["arm"],
            d["boundaries"],
            d["ctx_tokens"],
            d["uncached_tokens"],
            {int(t): c for t, c in d["contexts"].items()},
            [Event(*e) for e in d["events"]],
        )

    def mean_ctx(self) -> float:
        return sum(self.ctx_tokens) / len(self.ctx_tokens) if self.ctx_tokens else 0.0

    def mean_uncached(self) -> float:
        return (
            sum(self.uncached_tokens) / len(self.uncached_tokens) if self.uncached_tokens else 0.0
        )


# --------------------------------------------------------------------------- planning


def action_boundaries(turns: Sequence[Turn]) -> list[int]:
    n = len(turns)
    return [
        t
        for t in range(1, n + 1)
        if t == n or (turns[t].role == "assistant" and turns[t - 1].role != "assistant")
    ]


def _snap(turns: Sequence[Turn], t: int) -> int:
    """A checkpoint inside an action moves to the end of that action (`exi._action_end`) plus the
    result turn that answers it, so the history ends where the model acts next."""
    n = len(turns)
    if t >= n or turns[t - 1].role != "assistant":
        return min(t, n)
    t = exi._action_end(turns, t)
    return min(t + 1, n)


def checkpoint_turns(
    turns: Sequence[Turn], per_turn: Sequence[int], shares: Sequence[float] = CHECKPOINT_SHARES
) -> list[int]:
    total = sum(per_turn)
    before = [0]
    for x in per_turn:
        before.append(before[-1] + x)
    out: list[int] = []
    for share in shares:
        t = next((i for i in range(1, len(turns) + 1) if before[i] >= share * total), len(turns))
        out.append(_snap(turns, t))
    return out


def plan_transcript(label: str, turns: list[Turn]) -> TranscriptPlan:
    per_turn = [exi._turn_tokens(t) for t in turns]
    before = [0]
    for x in per_turn:
        before.append(before[-1] + x)
    total = before[-1]
    cks = [
        Checkpoint(round(s * 100), t, exi.tail_start(per_turn, t))
        for s, t in zip(CHECKPOINT_SHARES, checkpoint_turns(turns, per_turn), strict=True)
    ]
    boundaries = sorted(set(action_boundaries(turns)) | {c.t for c in cks})
    cap = int(CAP_SHARE * total)
    return TranscriptPlan(
        label, turns, per_turn, before, boundaries, total, cap, int(cap * SUMMARY_SHARE), cks
    )


# --------------------------------------------------------------------------- rendering


def compress_text(turns: Sequence[Turn], compressor: Compressor, keep: float = KEEP) -> str:
    if not turns:
        return ""
    t = Transcript("chunk", [Turn(i, x.role, x.blocks) for i, x in enumerate(turns)])
    return exi.render_segments(compressor(t, keep))


def _join(*parts: str) -> str:
    return "\n\n".join(p for p in parts if p)


def eval_context(view: View, turns: Sequence[Turn], t: int) -> str:
    earlier = _join(view.compressed, exi.render_turns(turns[view.verbatim_from : view.tail_from]))
    tail = exi.render_turns(turns[view.tail_from : t])
    return f"{view.header}\n\n{earlier}\n\n{exi.TAIL_HEADER}\n\n{tail}"


def cost_context(view: View, turns: Sequence[Turn], t: int) -> str:
    """The prompt as a cache would see it: no moving tail marker, the verbatim stretch contiguous
    after the compressed text."""
    return _join(view.header, view.compressed, exi.render_turns(turns[view.verbatim_from : t]))


def lcp_len(a: str, b: str) -> int:
    return len(os.path.commonprefix([a, b]))


def uncached_tokens(prev: str, cur: str) -> int:
    return count_tokens(cur[lcp_len(prev, cur) :])


# --------------------------------------------------------------------------- arms


class Freezer:
    """Compresses each F-token chunk of pre-tail text once and appends it; `frozen` only grows."""

    def __init__(
        self,
        turns: Sequence[Turn],
        per_turn: Sequence[int],
        compressor: Compressor,
        chunk_tokens: int = FREEZE_TOKENS,
    ) -> None:
        self.turns, self.per_turn, self.compressor = turns, per_turn, compressor
        self.chunk_tokens = chunk_tokens
        self.frozen: list[str] = []
        self.chunks: list[tuple[int, int]] = []
        self.upto = 0

    def advance(self, tail_from: int) -> None:
        while True:
            end, acc = self.upto, 0
            while end < tail_from and acc < self.chunk_tokens:
                acc += self.per_turn[end]
                end += 1
            if acc < self.chunk_tokens:
                return
            self.frozen.append(compress_text(self.turns[self.upto : end], self.compressor))
            self.chunks.append((self.upto, end))
            self.upto = end

    def text(self) -> str:
        return "\n\n".join(self.frozen)


def summary_input(prev: str | None, turns: Sequence[Turn]) -> str:
    body = exi.render_turns(turns)
    if prev is None:
        return body
    return f"[Summary of the session so far]\n\n{prev}\n\n[Session since that summary]\n\n{body}"


def placeholder_summary(k: int, text: str, budget: int) -> str:
    """Offline stand-in at the target length; opens differently per event so a placeholder never
    shares a long prefix with the one it replaces."""
    return f"[placeholder summary {k}] " + "alpha " * max(budget - 6, 0)


def _thresh_views(
    p: TranscriptPlan, summarize: Summarizer
) -> Iterator[tuple[int, View, Event | None]]:
    summary: str | None = None
    sum_tokens = cstart = k = 0
    for t in p.boundaries:
        tail_from = exi.tail_start(p.per_turn, t)
        event = None
        if sum_tokens + p.before[t] - p.before[cstart] > p.cap and tail_from > cstart:
            summary = summarize(k, summary_input(summary, p.turns[cstart:tail_from]), p.target)
            sum_tokens = count_tokens(summary)
            cstart, k = tail_from, k + 1
            event = Event(t, tail_from)
        header = HEADERS["thresh"] if summary is not None else HEADERS["full"]
        yield t, View(header, summary or "", cstart, tail_from), event


def _views(
    arm: str, p: TranscriptPlan, compressor: Compressor, summarize: Summarizer
) -> Iterator[tuple[int, View, Event | None]]:
    if arm == "thresh":
        yield from _thresh_views(p, summarize)
        return
    freezer = Freezer(p.turns, p.per_turn, compressor)
    memo: dict[int, str] = {}
    for t in p.boundaries:
        tail_from = exi.tail_start(p.per_turn, t)
        if arm == "full":
            view = View(HEADERS["full"], "", 0, tail_from)
        elif arm == "crec":
            if tail_from not in memo:
                memo[tail_from] = compress_text(p.turns[:tail_from], compressor)
            view = View(HEADERS["crec"], memo[tail_from], tail_from, tail_from)
        else:
            freezer.advance(tail_from)
            view = View(HEADERS["cfrz"], freezer.text(), freezer.upto, tail_from)
        yield t, view, None


def thresh_events(p: TranscriptPlan, summarize: Summarizer = placeholder_summary) -> list[Event]:
    return [e for _, _, e in _thresh_views(p, summarize) if e is not None]


def run_arm(
    arm: str,
    p: TranscriptPlan,
    compressor: Compressor,
    summarize: Summarizer = placeholder_summary,
    keep_at: Collection[int] | None = None,
) -> ArmRun:
    keep = {c.t for c in p.checkpoints} if keep_at is None else set(keep_at)
    run = ArmRun(arm)
    prev = ""
    for t, view, event in _views(arm, p, compressor, summarize):
        ctx = cost_context(view, p.turns, t)
        run.boundaries.append(t)
        run.ctx_tokens.append(count_tokens(ctx))
        run.uncached_tokens.append(uncached_tokens(prev, ctx))
        prev = ctx
        if event is not None:
            run.events.append(event)
        if t in keep:
            run.contexts[t] = eval_context(view, p.turns, t)
    return run


# --------------------------------------------------------------------------- set E


def qualifies(turns: list[Turn]) -> bool:
    p = plan_transcript("qualify", turns)
    last = p.checkpoints[-1].t
    return sum(e.t <= last for e in thresh_events(p)) >= MIN_THRESH_EVENTS


def select_set_e(
    candidates: Iterable[exi.Candidate],
    taken: Collection[str],
    *,
    token_range: tuple[int, int] = exi.TOKEN_RANGE,
    min_compressible: int = exi.MIN_COMPRESSIBLE_TOKENS,
) -> list[tuple[exi.Candidate, exi.Cut]]:
    pool = (
        c
        for c in candidates
        if c.source == "swe-gym" and c.label not in taken and qualifies(c.transcript.turns)
    )
    return exi.select_transcripts(
        pool,
        token_range=token_range,
        n=E_N,
        per_source=E_N,
        seed=E_SEED,
        min_compressible=min_compressible,
    )

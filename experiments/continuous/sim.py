"""Arm simulation for the continuous-compaction experiments (2026-11-16 spec).

Six arms replay a transcript turn by turn: `full`, `thresh` (threshold summarisation), `tloosy`
(the same threshold, loosy in place of the summary), `crec`
(loosy recomputed over the whole pre-tail history at every turn), `cfrz` (loosy applied once per
F-token chunk, appended to a frozen prefix) and `cdec` (the same chunks, tiered: when a tier holds
m chunks they are merged and recompressed into one chunk of the next tier). A "boundary" is a
history length `t` at which the model is called; each arm yields a `View` per boundary, from which
both the context a model would be shown at a checkpoint and the offline cost metrics are derived.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from experiments import exp_i_downstream as exi
from loosy_goose.rate import LEVELS
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Block, Transcript, Turn

if TYPE_CHECKING:
    from loosy_goose.segment import Segment

# Covers the freeze, recompute and threshold simulations; goes into every record key.
CONT_VERSION = 1
# Covers the tiered decaying-freeze simulation; goes into every cdec record key.
DEC_VERSION = 1
# Covers the threshold-compaction-with-loosy simulation; goes into every tloosy record key.
TLOOSY_VERSION = 1
# Covers the Qwen-summarizer threshold simulation; goes into every qthresh record key.
QTHRESH_VERSION = 1
# Covers the gated-freeze simulation; goes into every cfrzg record key.
CFRZG_VERSION = 1

KEEP = LEVELS["medium"]
FREEZE_TOKENS = 4_000
CAP_SHARE = 0.40
SUMMARY_SHARE = 0.33
CHECKPOINT_SHARES = (0.5, 0.75, 1.0)
ARMS = ("full", "thresh", "crec", "cfrz")
F_ARMS = ("full", "thresh", "cfrz", "cfrzm", "cdec", "tloosy")
Q_ARMS = ("full", "qthresh", "tloosy", "cfrzm", "cfrzg")
E_SEED = exi.SEED + 4
E_N = 10
MIN_THRESH_EVENTS = 2
F_SEED = exi.SEED + 5
F_N = 25
F_CAP_SHARE = 0.65
# Chosen by experiments/continuous/run.py --calibrate; see the spec's 2b section.
DEC_KEEP = 0.7
DEC_M = 4
# cfrz with its keep raised to thresh's context size; chosen by run.py --calibrate.
FRZM_KEEP = 0.6
# tloosy's compaction target as a share of the cap C (thresh's is SUMMARY_SHARE); chosen by
# run.py --calibrate against thresh's real mean context.
TLOOSY_SHARE = 0.33
# cfrzg stays equal to full while the whole context is at most this many tokens.
CFRZG_GATE = 16_000

HEADERS = {
    "full": exi.EARLIER_HEADER["full"],
    "crec": exi.EARLIER_HEADER["loosy"],
    "thresh": "[Earlier part of the session: a summary of its older part, then the rest verbatim]",
    "cfrz": (
        "[Earlier part of the session: its older part compressed extractively (some passages "
        "were dropped, what remains is verbatim), then the rest verbatim]"
    ),
}
HEADERS["cdec"] = (
    "[Earlier part of the session: its older part compressed extractively (some passages were "
    "dropped, what remains is verbatim; the oldest parts were compressed more than once and "
    "kept less), then the rest verbatim]"
)

HEADERS["tloosy"] = HEADERS["cfrz"]
HEADERS["qthresh"] = HEADERS["thresh"]
HEADERS["cfrzg"] = HEADERS["cfrz"]

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


def plan_transcript(label: str, turns: list[Turn], cap_share: float = CAP_SHARE) -> TranscriptPlan:
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
    cap = int(cap_share * total)
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


def next_chunk_end(
    per_turn: Sequence[int], upto: int, tail_from: int, chunk_tokens: int
) -> int | None:
    """End of the next chunk of at least `chunk_tokens` that starts at `upto` and stays before
    the tail, or None while the pending text is still short of a chunk."""
    end, acc = upto, 0
    while end < tail_from and acc < chunk_tokens:
        acc += per_turn[end]
        end += 1
    return end if acc >= chunk_tokens else None


class Freezer:
    """Compresses each F-token chunk of pre-tail text once and appends it; `frozen` only grows."""

    def __init__(
        self,
        turns: Sequence[Turn],
        per_turn: Sequence[int],
        compressor: Compressor,
        chunk_tokens: int = FREEZE_TOKENS,
        keep: float = KEEP,
    ) -> None:
        self.turns, self.per_turn, self.compressor = turns, per_turn, compressor
        self.chunk_tokens, self.keep = chunk_tokens, keep
        self.frozen: list[str] = []
        self.chunks: list[tuple[int, int]] = []
        self.upto = 0

    def advance(self, tail_from: int) -> None:
        while end := next_chunk_end(self.per_turn, self.upto, tail_from, self.chunk_tokens):
            self.frozen.append(
                compress_text(self.turns[self.upto : end], self.compressor, self.keep)
            )
            self.chunks.append((self.upto, end))
            self.upto = end

    def text(self) -> str:
        return "\n\n".join(self.frozen)


@dataclass(frozen=True)
class DecChunk:
    segments: tuple[Segment, ...]
    text: str
    span: tuple[int, int]


def _segments_to_turns(chunks: Sequence[DecChunk]) -> list[Turn]:
    """Kept segments regrouped into turns and blocks, chunk after chunk, so the shipped compressor
    can run over them again. A gap between kept segments of one block closes up: the segmenter
    re-reads the joined text."""
    turns: list[Turn] = []
    for ch in chunks:
        ordered = sorted(ch.segments, key=lambda s: s.id)
        i = 0
        while i < len(ordered):
            first, blocks = ordered[i], []
            j = i
            while j < len(ordered) and ordered[j].turn == first.turn:
                b0 = ordered[j]
                k = j
                while k < len(ordered) and (ordered[k].turn, ordered[k].block) == (
                    b0.turn,
                    b0.block,
                ):
                    k += 1
                sep = "\n" if b0.kind == "tool_result" else "\n\n"
                meta = {"name": b0.tool_name} if b0.tool_name else {}
                blocks.append(
                    Block(
                        exi._SEGMENT_BLOCK_KIND.get(b0.kind, b0.kind),
                        sep.join(s.text for s in ordered[j:k]),
                        meta,
                    )
                )
                j = k
            turns.append(Turn(len(turns), first.role, blocks))
            i = j
    return turns


class Decayer:
    """LSM-style decaying freeze. A chunk that ages out of the tail is compressed at `keep` into
    tier 0; when a tier holds `m` chunks they are merged, recompressed at `keep` over their kept
    segments, and appended to the next tier as one chunk. Tiers are laid out oldest-first (highest
    tier first), so a merge rewrites the context only from the merged tier's slot onward."""

    def __init__(
        self,
        turns: Sequence[Turn],
        per_turn: Sequence[int],
        compressor: Compressor,
        keep: float = DEC_KEEP,
        m: int = DEC_M,
        chunk_tokens: int = FREEZE_TOKENS,
    ) -> None:
        self.turns, self.per_turn, self.compressor = turns, per_turn, compressor
        self.keep, self.m, self.chunk_tokens = keep, m, chunk_tokens
        self.tiers: list[list[DecChunk]] = []
        self.merges: list[tuple[int, tuple[DecChunk, ...], DecChunk]] = []
        self.upto = 0

    def _compress(self, turns: list[Turn], span: tuple[int, int]) -> DecChunk:
        t = Transcript("chunk", [Turn(i, x.role, x.blocks) for i, x in enumerate(turns)])
        segs = self.compressor(t, self.keep)
        return DecChunk(tuple(segs), exi.render_segments(segs), span)

    def _push(self, tier: int, chunk: DecChunk) -> None:
        while len(self.tiers) <= tier:
            self.tiers.append([])
        self.tiers[tier].append(chunk)
        if len(self.tiers[tier]) >= self.m:
            group, self.tiers[tier] = self.tiers[tier], []
            span = (group[0].span[0], group[-1].span[1])
            merged = self._compress(_segments_to_turns(group), span)
            self.merges.append((tier, tuple(group), merged))
            self._push(tier + 1, merged)

    def advance(self, tail_from: int) -> None:
        while end := next_chunk_end(self.per_turn, self.upto, tail_from, self.chunk_tokens):
            span = (self.upto, end)
            self._push(0, self._compress(list(self.turns[self.upto : end]), span))
            self.upto = end

    def text(self) -> str:
        return "\n\n".join(c.text for tier in reversed(self.tiers) for c in tier)


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


def _tloosy_views(
    p: TranscriptPlan, compressor: Compressor, share: float
) -> Iterator[tuple[int, View, Event | None]]:
    """`thresh` with shipped loosy for the summary. The input to a compaction is the previous
    compacted segments (regrouped into turns) plus the verbatim turns since, compressed to
    `share` of the cap by handing the compressor that target as its keep ratio."""
    segs: tuple[Segment, ...] = ()
    compacted = ""
    sum_tokens = cstart = 0
    target = int(p.cap * share)
    for t in p.boundaries:
        tail_from = exi.tail_start(p.per_turn, t)
        event = None
        if sum_tokens + p.before[t] - p.before[cstart] > p.cap and tail_from > cstart:
            turns = _segments_to_turns([DecChunk(segs, "", (0, 0))]) if segs else []
            turns += p.turns[cstart:tail_from]
            tokens = sum(exi._turn_tokens(x) for x in turns)
            chunk = Transcript("chunk", [Turn(i, x.role, x.blocks) for i, x in enumerate(turns)])
            segs = tuple(compressor(chunk, min(1.0, target / tokens)))
            compacted = exi.render_segments(segs)
            sum_tokens = count_tokens(compacted)
            cstart = tail_from
            event = Event(t, tail_from)
        header = HEADERS["tloosy"] if segs else HEADERS["full"]
        yield t, View(header, compacted, cstart, tail_from), event


def _views(
    arm: str,
    p: TranscriptPlan,
    compressor: Compressor,
    summarize: Summarizer,
    dec: tuple[float, int] = (DEC_KEEP, DEC_M),
    frzm_keep: float = FRZM_KEEP,
    tloosy_share: float = TLOOSY_SHARE,
) -> Iterator[tuple[int, View, Event | None]]:
    if arm in ("thresh", "qthresh"):
        yield from _thresh_views(p, summarize)
        return
    if arm == "tloosy":
        yield from _tloosy_views(p, compressor, tloosy_share)
        return
    freezer = Freezer(
        p.turns, p.per_turn, compressor, keep=frzm_keep if arm in ("cfrzm", "cfrzg") else KEEP
    )
    gate_open = False
    decayer = Decayer(p.turns, p.per_turn, compressor, *dec)
    memo: dict[int, str] = {}
    for t in p.boundaries:
        tail_from = exi.tail_start(p.per_turn, t)
        if arm == "full":
            view = View(HEADERS["full"], "", 0, tail_from)
        elif arm == "crec":
            if tail_from not in memo:
                memo[tail_from] = compress_text(p.turns[:tail_from], compressor)
            view = View(HEADERS["crec"], memo[tail_from], tail_from, tail_from)
        elif arm == "cdec":
            decayer.advance(tail_from)
            view = View(HEADERS["cdec"], decayer.text(), decayer.upto, tail_from)
        elif arm == "cfrzg":
            gate_open = gate_open or p.before[t] > CFRZG_GATE
            if gate_open:
                freezer.advance(tail_from)
                view = View(HEADERS["cfrzg"], freezer.text(), freezer.upto, tail_from)
            else:
                view = View(HEADERS["full"], "", 0, tail_from)
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
    dec: tuple[float, int] = (DEC_KEEP, DEC_M),
    frzm_keep: float = FRZM_KEEP,
    tloosy_share: float = TLOOSY_SHARE,
) -> ArmRun:
    keep = {c.t for c in p.checkpoints} if keep_at is None else set(keep_at)
    run = ArmRun(arm)
    prev = ""
    for t, view, event in _views(arm, p, compressor, summarize, dec, frzm_keep, tloosy_share):
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


def qualifies(
    turns: list[Turn], cap_share: float = CAP_SHARE, min_events: int = MIN_THRESH_EVENTS
) -> bool:
    p = plan_transcript("qualify", turns, cap_share)
    last = p.checkpoints[-1].t
    return sum(e.t <= last for e in thresh_events(p)) >= min_events


def select_set(
    candidates: Iterable[exi.Candidate],
    taken: Collection[str],
    *,
    n: int,
    seed: int,
    cap_share: float,
    min_events: int,
    token_range: tuple[int, int] = exi.TOKEN_RANGE,
    min_compressible: int = exi.MIN_COMPRESSIBLE_TOKENS,
) -> list[tuple[exi.Candidate, exi.Cut]]:
    pool = (
        c
        for c in candidates
        if c.source == "swe-gym"
        and c.label not in taken
        and qualifies(c.transcript.turns, cap_share, min_events)
    )
    return exi.select_transcripts(
        pool,
        token_range=token_range,
        n=n,
        per_source=n,
        seed=seed,
        min_compressible=min_compressible,
    )


def select_set_e(
    candidates: Iterable[exi.Candidate],
    taken: Collection[str],
    *,
    token_range: tuple[int, int] = exi.TOKEN_RANGE,
    min_compressible: int = exi.MIN_COMPRESSIBLE_TOKENS,
) -> list[tuple[exi.Candidate, exi.Cut]]:
    return select_set(
        candidates,
        taken,
        n=E_N,
        seed=E_SEED,
        cap_share=CAP_SHARE,
        min_events=MIN_THRESH_EVENTS,
        token_range=token_range,
        min_compressible=min_compressible,
    )


def select_set_f(
    candidates: Iterable[exi.Candidate],
    taken: Collection[str],
    *,
    token_range: tuple[int, int] = exi.TOKEN_RANGE,
    min_compressible: int = exi.MIN_COMPRESSIBLE_TOKENS,
) -> list[tuple[exi.Candidate, exi.Cut]]:
    return select_set(
        candidates,
        taken,
        n=F_N,
        seed=F_SEED,
        cap_share=F_CAP_SHARE,
        min_events=MIN_THRESH_EVENTS,
        token_range=token_range,
        min_compressible=min_compressible,
    )

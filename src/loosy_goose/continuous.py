"""Continuous compaction for chat-completions requests: freeze, tool-safe tail, decay.

The policy mirrors `experiments/continuous/sim.py` (`Freezer`, `Decayer`, `next_chunk_end`,
`_segments_to_turns`), working in message indices rather than turn indices.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .openai_messages import is_barrier, system_prefix, to_transcript
from .render import SEGMENT_BLOCK_KIND, render_segments
from .segment import Segment
from .tokens import count_tokens
from .transcript import Block, Transcript, Turn, _text_of

CONTINUOUS_VERSION = 1

HEADER = (
    "[Earlier part of the conversation, compressed extractively: some passages were dropped, "
    "what remains is verbatim]"
)

Compress = Callable[[Transcript, float], list[Segment]]


@dataclass(frozen=True)
class ContinuousConfig:
    tail_tokens: int = 4000
    chunk_tokens: int = 4000
    keep: float = 0.6
    window: int | None = None
    start_at: float = 0.0
    decay_m: int = 4
    decay_keep: float = 0.7
    decay_at: float = 0.9

    def __post_init__(self) -> None:
        if self.start_at > 0 and self.window is None:
            raise ValueError("start_at > 0 needs a window: the gate is a share of it")
        # decay_m < 2 would merge a lone chunk into itself and never shrink the header.
        if self.decay_m < 2:
            raise ValueError("decay_m must be at least 2")
        if self.tail_tokens <= 0 or self.chunk_tokens <= 0:
            raise ValueError("tail_tokens and chunk_tokens must be positive")
        if not (0 < self.keep <= 1 and 0 < self.decay_keep <= 1):
            raise ValueError("keep and decay_keep must be in (0, 1]")
        if self.window is not None and self.window <= 0:
            raise ValueError("window must be positive")
        if not (0 <= self.start_at <= 1 and 0 < self.decay_at <= 1):
            raise ValueError("start_at must be in [0, 1] and decay_at in (0, 1]")


@dataclass
class Chunk:
    span: tuple[int, int]
    source_sha: str
    tier: int
    segments: list[Segment]
    text: str


def _pipeline_compress(t: Transcript, keep: float) -> list[Segment]:
    from .pipeline import PipelineOptions, compress_transcript

    return compress_transcript(t, PipelineOptions(keep=keep)).segments


def source_sha(messages: Sequence[Mapping[str, Any]]) -> str:
    canon = json.dumps(list(messages), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def conversation_id(messages: Sequence[Mapping[str, Any]]) -> str:
    """Identity of a conversation across requests: its leading system messages and first user
    message, which later turns only ever append after."""
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    canon = json.dumps(
        [list(messages[: system_prefix(messages)]), first_user],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:32]


_CID = re.compile(r"[0-9a-f]{32}")


class StateStore:
    """One JSON file per conversation. Holds frozen chunk text, i.e. conversation content, so
    nothing here logs it."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, cid: str) -> Path:
        if not _CID.fullmatch(cid):
            raise ValueError("conversation id must be 32 lowercase hex characters")
        return self.root / f"{cid}.json"

    def load(self, cid: str, config: ContinuousConfig) -> list[Chunk]:
        path = self._path(cid)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data["version"] != CONTINUOUS_VERSION or data["config"] != asdict(config):
                return []
            return [
                Chunk(
                    (c["span"][0], c["span"][1]),
                    c["source_sha"],
                    c["tier"],
                    [Segment(**s) for s in c["segments"]],
                    c["text"],
                )
                for c in data["chunks"]
            ]
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            # A missing, torn or foreign-shaped file means no usable state, not a failed request.
            return []

    def save(self, cid: str, config: ContinuousConfig, chunks: list[Chunk]) -> None:
        path = self._path(cid)
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CONTINUOUS_VERSION,
            "config": asdict(config),
            "chunks": [asdict(c) for c in chunks],
        }
        fd, tmp = tempfile.mkstemp(dir=self.root, prefix=f".{cid}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def _tool_owners(messages: Sequence[Mapping[str, Any]]) -> list[int]:
    """For each message, the index of the assistant message whose `tool_calls` it answers, or its
    own index when it answers none. A `tool` message whose id matches no call falls back to the
    nearest earlier assistant message with tool calls, so it still never lands alone."""
    owner_of_id: dict[str, int] = {}
    last_caller: int | None = None
    owners: list[int] = []
    for i, m in enumerate(messages):
        owners.append(i)
        calls = m.get("tool_calls")
        if m.get("role") == "assistant" and isinstance(calls, Sequence) and calls:
            last_caller = i
            for c in calls:
                if isinstance(c, Mapping) and isinstance(c.get("id"), str):
                    owner_of_id[c["id"]] = i
        elif m.get("role") == "tool":
            call_id = m.get("tool_call_id")
            if isinstance(call_id, str) and call_id in owner_of_id:
                owners[i] = owner_of_id[call_id]
            elif last_caller is not None:
                owners[i] = last_caller
    return owners


def _segments_to_turns(chunks: Sequence[Chunk]) -> list[Turn]:
    """Kept segments regrouped into turns and blocks, chunk after chunk, so the compressor can
    run over them again. A gap between kept segments of one block closes up: the segmenter
    re-reads the joined text."""
    turns: list[Turn] = []
    for ch in chunks:
        ordered = sorted(ch.segments, key=lambda s: s.id)
        i = 0
        while i < len(ordered):
            first = ordered[i]
            blocks: list[Block] = []
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
                        SEGMENT_BLOCK_KIND.get(b0.kind, b0.kind),  # type: ignore[arg-type]
                        sep.join(s.text for s in ordered[j:k]),
                        meta,
                    )
                )
                j = k
            turns.append(Turn(len(turns), first.role, blocks))
            i = j
    return turns


class FreezeCompactor:
    def __init__(
        self,
        config: ContinuousConfig = ContinuousConfig(),  # noqa: B008 - frozen, never mutated
        compress: Compress | None = None,
        chunks: list[Chunk] | None = None,
    ) -> None:
        self.config = config
        self._compress = compress or _pipeline_compress
        self._chunks: list[Chunk] = list(chunks or [])

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cfg = self.config
        prefix = system_prefix(messages)
        transcript, index = to_transcript(messages)
        turn_of = dict(zip(index, transcript.turns, strict=True))
        per_msg = [
            sum(count_tokens(b.text) for b in turn_of[i].blocks)
            if i in turn_of
            else count_tokens(_text_of(m.get("content")))
            for i, m in enumerate(messages)
        ]
        barriers = [i for i, m in enumerate(messages) if i >= prefix and is_barrier(m)]
        self._drop_stale(messages, prefix, barriers)
        if (
            not self._chunks
            and cfg.window is not None
            and cfg.start_at > 0
            and sum(per_msg) <= cfg.start_at * cfg.window
        ):
            return list(messages)

        # min_owner_from[b] < b exactly when a cut at b would separate a tool reply from its call.
        owners = _tool_owners(messages)
        min_owner_from = list(range(len(messages) + 1))
        for i in range(len(messages) - 1, -1, -1):
            min_owner_from[i] = min(owners[i], min_owner_from[i + 1])

        tail_from, acc = len(messages), 0
        while acc < cfg.tail_tokens and tail_from > prefix:
            tail_from -= 1
            acc += per_msg[tail_from]
        # Nothing at or after the first message `to_transcript` cannot carry is frozen: it and
        # everything behind it stay verbatim, in order. Freezing past it while keeping it would
        # reorder the output.
        if barriers:
            tail_from = min(tail_from, barriers[0])
        while min_owner_from[tail_from] < tail_from:
            tail_from = max(min_owner_from[tail_from], prefix)

        # A client that pruned messages after the frozen span leaves chunks a fresh run would
        # keep verbatim.
        for k, ch in enumerate(self._chunks):
            if ch.span[1] > tail_from:
                del self._chunks[k:]
                break
        upto = self._chunks[-1].span[1] if self._chunks else prefix
        while (end := self._next_chunk_end(per_msg, min_owner_from, upto, tail_from)) is not None:
            self._chunks.append(self._freeze(messages, transcript, index, (upto, end)))
            upto = end

        if not self._chunks:
            return list(messages)

        if cfg.window is not None:
            limit = cfg.decay_at * cfg.window
            rest = sum(per_msg[:prefix]) + sum(per_msg[upto:])
            # Merging only shrinks the header, never below the bare HEADER line: when even that
            # leaves the context over the limit, merging would cost detail for nothing.
            if rest + count_tokens(HEADER) <= limit:
                while rest + count_tokens(self._header_text()) > limit:
                    if not self._merge_lowest_full_tier(messages):
                        break

        head = self._header_text()
        rest_msgs = messages[upto:]
        # Folded into a following user message rather than sent beside it: strict-alternation chat
        # templates reject two user messages in a row.
        if rest_msgs and rest_msgs[0].get("role") == "user":
            first = dict(rest_msgs[0])
            content = first.get("content")
            if isinstance(content, list):
                first["content"] = [{"type": "text", "text": head + "\n\n"}, *content]
            else:
                first["content"] = head + "\n\n" + (content if isinstance(content, str) else "")
            return [*messages[:prefix], first, *rest_msgs[1:]]
        return [*messages[:prefix], {"role": "user", "content": head}, *rest_msgs]

    def _drop_stale(
        self, messages: Sequence[Mapping[str, Any]], prefix: int, barriers: Sequence[int]
    ) -> None:
        """A chunk whose source messages are no longer the request's (history edited, a shorter
        conversation) is discarded with every chunk after it, so the frozen text never describes
        messages the client did not send."""
        for k, ch in enumerate(self._chunks):
            a, b = ch.span
            if (
                a < prefix
                or b > len(messages)
                or any(a <= i < b for i in barriers)
                or source_sha(messages[a:b]) != ch.source_sha
            ):
                del self._chunks[k:]
                return

    def _next_chunk_end(
        self, per_msg: Sequence[int], min_owner_from: Sequence[int], upto: int, tail_from: int
    ) -> int | None:
        end, acc = upto, 0
        while end < tail_from and acc < self.config.chunk_tokens:
            acc += per_msg[end]
            end += 1
        if acc < self.config.chunk_tokens:
            return None
        # tail_from is itself a safe cut, so this stops at or before it.
        while min_owner_from[end] < end:
            end += 1
        return end

    def _freeze(
        self,
        messages: Sequence[Mapping[str, Any]],
        transcript: Transcript,
        index: Sequence[int],
        span: tuple[int, int],
    ) -> Chunk:
        a, b = span
        turns = [t for t, i in zip(transcript.turns, index, strict=True) if a <= i < b]
        segs = self._run(turns, self.config.keep)
        return Chunk(span, source_sha(messages[a:b]), 0, segs, render_segments(segs))

    def _run(self, turns: Sequence[Turn], keep: float) -> list[Segment]:
        t = Transcript("chunk", [Turn(i, x.role, x.blocks) for i, x in enumerate(turns)])
        return self._compress(t, keep)

    def _merge_lowest_full_tier(self, messages: Sequence[Mapping[str, Any]]) -> bool:
        m = self.config.decay_m
        tiers = sorted({ch.tier for ch in self._chunks})
        full = next((t for t in tiers if sum(ch.tier == t for ch in self._chunks) >= m), None)
        if full is None:
            return False
        # Chunks of one tier are contiguous and oldest-first, so the oldest m are adjacent.
        start = next(k for k, ch in enumerate(self._chunks) if ch.tier == full)
        group = self._chunks[start : start + m]
        span = (group[0].span[0], group[-1].span[1])
        segs = self._run(_segments_to_turns(group), self.config.decay_keep)
        merged = Chunk(
            span, source_sha(messages[span[0] : span[1]]), full + 1, segs, render_segments(segs)
        )
        self._chunks[start : start + m] = [merged]
        return True

    def _header_text(self) -> str:
        return HEADER + "\n\n" + "\n\n".join(ch.text for ch in self._chunks)

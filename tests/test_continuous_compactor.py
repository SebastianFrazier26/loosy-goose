import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_i_downstream as exi  # noqa: E402
from loosy_goose.continuous import (  # noqa: E402
    HEADER,
    ContinuousConfig,
    FreezeCompactor,
    StateStore,
    conversation_id,
)
from loosy_goose.render import render_segments  # noqa: E402
from loosy_goose.segment import Segment, segment  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Transcript  # noqa: E402


def keep_every_third(t: Transcript, keep: float) -> list[Segment]:
    return segment(t)[::3]


def _words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def _messages(rounds: int, size: int = 120) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": _words(size, "ask")},
    ]
    for r in range(rounds):
        call = {
            "id": f"call{r}",
            "type": "function",
            "function": {"name": "read", "arguments": json.dumps({"path": f"src/f{r}.py"})},
        }
        msgs.append(
            {"role": "assistant", "content": _words(size // 4, f"a{r}x"), "tool_calls": [call]}
        )
        msgs.append({"role": "tool", "tool_call_id": f"call{r}", "content": _words(size, f"r{r}x")})
    return msgs


SMALL = ContinuousConfig(tail_tokens=600, chunk_tokens=500)


def _compactor(config: ContinuousConfig = SMALL) -> FreezeCompactor:
    return FreezeCompactor(config, compress=keep_every_third)


def _context_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        total += count_tokens(m.get("content") or "")
        for c in m.get("tool_calls") or []:
            total += count_tokens(c["function"]["arguments"])
    return total


def test_config_rejects_gate_without_window() -> None:
    with pytest.raises(ValueError):
        ContinuousConfig(start_at=0.5)
    ContinuousConfig(start_at=0.5, window=8000)


def test_short_conversation_unchanged() -> None:
    msgs = _messages(2, size=30)
    assert _compactor().compact(msgs) == msgs
    assert _compactor().chunks == []


def test_frozen_chunks_never_change_as_conversation_grows() -> None:
    full = _messages(40)
    c = _compactor()
    n = 30
    c.compact(full[:n])
    first = [(ch.span, ch.text) for ch in c.chunks]
    assert first
    for extra in (5, 10):
        c.compact(full[: n + extra])
        assert [(ch.span, ch.text) for ch in c.chunks[: len(first)]] == first
    assert len(c.chunks) > len(first)


def test_output_deterministic() -> None:
    msgs = _messages(30)
    a = json.dumps(_compactor().compact(msgs), sort_keys=True)
    b = json.dumps(_compactor().compact(msgs), sort_keys=True)
    assert a == b
    c = _compactor()
    c.compact(msgs[:20])
    assert json.dumps(c.compact(msgs), sort_keys=True) == a


def test_tail_verbatim_and_tool_pair_not_split() -> None:
    msgs = _messages(30)
    for tail in range(200, 1200, 37):
        cfg = ContinuousConfig(tail_tokens=tail, chunk_tokens=400)
        out = _compactor(cfg).compact(msgs)
        rest = out[2:]
        assert rest == msgs[len(msgs) - len(rest) :]
        assert rest[0]["role"] != "tool"
        assert _context_tokens(rest) >= tail


def test_chunk_end_does_not_strand_tool_result() -> None:
    msgs = _messages(30)
    for chunk in range(150, 900, 41):
        c = _compactor(ContinuousConfig(tail_tokens=600, chunk_tokens=chunk))
        c.compact(msgs)
        for ch in c.chunks:
            assert msgs[ch.span[1]]["role"] != "tool"


def test_header_and_system_first() -> None:
    msgs = [{"role": "system", "content": "second system"}, *_messages(30)]
    out = _compactor().compact(msgs)
    assert out[:2] == msgs[:2]
    assert out[2]["role"] == "user"
    head = (
        "[Earlier part of the conversation, compressed extractively: some passages were "
        "dropped, what remains is verbatim]"
    )
    assert HEADER == head
    assert out[2]["content"].startswith(head + "\n\n")
    assert len(out) < len(msgs)


def test_gate_passes_through_until_start_at() -> None:
    msgs = _messages(60)
    window = 8000
    cfg = ContinuousConfig(tail_tokens=600, chunk_tokens=500, window=window, start_at=0.5)
    c = _compactor(cfg)
    n = next(k for k in range(3, len(msgs)) if _context_tokens(msgs[:k]) > 0.5 * window)
    for k in range(3, n):
        assert c.compact(msgs[:k]) == msgs[:k]
        assert c.chunks == []
    assert c.compact(msgs[:n]) != msgs[:n]
    assert c.chunks
    restored = FreezeCompactor(cfg, compress=keep_every_third, chunks=c.chunks[:1])
    short = msgs[: c.chunks[0].span[1] + 2]
    assert _context_tokens(short) <= 0.5 * window
    assert restored.compact(short)[1]["content"].startswith(HEADER)


def test_decay_bounds_context_and_keeps_older_tiers() -> None:
    msgs = _messages(200)
    window = 6000
    cfg = ContinuousConfig(tail_tokens=600, chunk_tokens=500, window=window)
    c = _compactor(cfg)
    seen_high: dict[tuple[int, int], str] = {}
    for k in range(10, len(msgs) + 1, 6):
        out = c.compact(msgs[:k])
        tiers = [ch.tier for ch in c.chunks]
        assert tiers == sorted(tiers, reverse=True)
        starts = [ch.span[0] for ch in c.chunks]
        assert starts == sorted(starts)
        tier0 = sum(t == 0 for t in tiers)
        if _context_tokens(out) > 0.9 * window:
            assert tier0 < cfg.decay_m
        for ch in c.chunks:
            if ch.tier > 0 and ch.span in seen_high:
                assert seen_high[ch.span] == ch.text
                assert ch.text in out[1]["content"]
            if ch.tier > 0:
                seen_high[ch.span] = ch.text
    assert max(ch.tier for ch in c.chunks) >= 1
    assert _context_tokens(out) <= 0.9 * window
    assert _context_tokens(out) < _context_tokens(msgs) / 3


def test_no_decay_without_window() -> None:
    c = _compactor()
    c.compact(_messages(200))
    assert {ch.tier for ch in c.chunks} == {0}


def test_render_matches_experiment_render() -> None:
    segs = [
        Segment(0, 0, "user", "prose", "first paragraph", False),
        Segment(1, 0, "user", "prose", "second paragraph", False),
        Segment(
            3, 1, "assistant", "tool_use", '{"path": "a.py"}', False, tool_name="read", block=0
        ),
        Segment(4, 1, "assistant", "code", "x = 1", False, block=1),
        Segment(5, 2, "user", "tool_result", "line one", False),
        Segment(7, 2, "user", "tool_result", "line three", False),
        Segment(8, 2, "user", "tool_result", "line four", False),
        Segment(9, 3, "assistant", "thinking", "hmm", False),
        Segment(10, 3, "assistant", "tool_use", "{}", False, block=1),
    ]
    assert render_segments(segs) == exi.render_segments(segs)
    assert render_segments(list(reversed(segs))) == exi.render_segments(segs)


def _alternating(rounds: int, size: int = 120) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "you are an agent"}]
    for r in range(rounds):
        msgs.append({"role": "user", "content": _words(size, f"u{r}x")})
        msgs.append({"role": "assistant", "content": _words(size, f"a{r}x")})
    return msgs


def test_header_folds_into_following_user_message() -> None:
    msgs = _alternating(20)
    c = _compactor()
    out = c.compact(msgs)
    upto = c.chunks[-1].span[1]
    assert msgs[upto]["role"] == "user"
    roles = [m["role"] for m in out]
    assert all(not (a == b == "user") for a, b in zip(roles, roles[1:], strict=False))
    assert sum(m["content"].count(HEADER) for m in out) == 1
    assert out[1]["content"] == (
        HEADER + "\n\n" + "\n\n".join(ch.text for ch in c.chunks) + "\n\n" + msgs[upto]["content"]
    )
    assert out[2:] == msgs[upto + 1 :]
    assert msgs[upto]["content"].startswith("u")


def test_header_folds_into_list_content() -> None:
    msgs = _alternating(20)
    for m in msgs:
        if m["role"] == "user":
            m["content"] = [{"type": "text", "text": m["content"]}]
    c = _compactor()
    out = c.compact(msgs)
    upto = c.chunks[-1].span[1]
    part, *rest = out[1]["content"]
    assert part == {
        "type": "text",
        "text": HEADER + "\n\n" + "\n\n".join(ch.text for ch in c.chunks) + "\n\n",
    }
    assert rest == msgs[upto]["content"]
    assert [m["role"] for m in out[:3]] == ["system", "user", "assistant"]


def test_state_round_trip_resumes_identically(tmp_path: Path) -> None:
    full = _messages(40)
    cid = conversation_id(full)
    store = StateStore(tmp_path)
    first = _compactor()
    first.compact(full[:30])
    assert first.chunks
    store.save(cid, SMALL, first.chunks)

    restored = FreezeCompactor(SMALL, compress=keep_every_third, chunks=store.load(cid, SMALL))
    assert restored.chunks == first.chunks
    assert restored.compact(full) == first.compact(full)
    assert restored.chunks == first.chunks


def test_edited_early_message_equals_fresh_run() -> None:
    full = _messages(40)
    c = _compactor()
    c.compact(full)
    assert len(c.chunks) > 2
    edited = [dict(m) for m in full]
    edited[8] = {**edited[8], "content": _words(120, "edited")}
    assert c.compact(edited) == _compactor().compact(edited)
    fresh = _compactor()
    fresh.compact(edited)
    assert c.chunks == fresh.chunks


def test_config_change_discards_state(tmp_path: Path) -> None:
    full = _messages(40)
    cid = conversation_id(full)
    store = StateStore(tmp_path)
    c = _compactor()
    c.compact(full)
    store.save(cid, SMALL, c.chunks)
    assert store.load(cid, SMALL)
    assert store.load(cid, ContinuousConfig(tail_tokens=700, chunk_tokens=500)) == []
    assert store.load("0" * 32, SMALL) == []


def test_version_mismatch_and_corrupt_file_discard_state(tmp_path: Path) -> None:
    full = _messages(40)
    cid = conversation_id(full)
    store = StateStore(tmp_path)
    c = _compactor()
    c.compact(full)
    store.save(cid, SMALL, c.chunks)
    path = tmp_path / f"{cid}.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps({**data, "version": data["version"] + 1}))
    assert store.load(cid, SMALL) == []
    path.write_text("{not json")
    assert store.load(cid, SMALL) == []
    with pytest.raises(ValueError):
        store.load("../x", SMALL)


def test_save_leaves_no_temp_files(tmp_path: Path) -> None:
    full = _messages(40)
    cid = conversation_id(full)
    c = _compactor()
    c.compact(full)
    StateStore(tmp_path / "nested").save(cid, SMALL, c.chunks)
    assert [p.name for p in (tmp_path / "nested").iterdir()] == [f"{cid}.json"]


def test_conversation_id_stable_across_appends() -> None:
    full = _messages(40)
    assert conversation_id(full[:10]) == conversation_id(full)
    assert len(conversation_id(full)) == 32
    other = [dict(m) for m in full]
    other[1] = {**other[1], "content": "different task"}
    assert conversation_id(other) != conversation_id(full)
    other = [dict(m) for m in full]
    other[0] = {**other[0], "content": "different system"}
    assert conversation_id(other) != conversation_id(full)

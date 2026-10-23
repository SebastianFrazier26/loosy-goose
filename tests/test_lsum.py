from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "slot_compress"))

import lsum  # noqa: E402


def test_truncate_sentence_stops_at_boundary() -> None:
    text = "First sentence here. Second sentence follows. Third one is the long tail of it all."
    out = lsum.truncate_sentence(text, 9)
    assert out.endswith(".")
    assert text.startswith(out)
    assert len(lsum._ENC.encode(out)) <= 9


def test_truncate_sentence_noop_when_short() -> None:
    assert lsum.truncate_sentence("  short text.  ", 50) == "short text."


def test_split_chunks_respects_limit_and_order() -> None:
    blocks = [f"block {i} " + "word " * 20 for i in range(10)]
    chunks = lsum.split_chunks("\n\n".join(blocks), limit=60)
    assert len(chunks) > 1
    assert "\n\n".join(chunks) == "\n\n".join(blocks)
    assert all(lsum._ntok(c) <= 60 for c in chunks)


def test_split_chunks_cuts_oversized_block() -> None:
    chunks = lsum.split_chunks("word " * 500, limit=100)
    assert all(lsum._ntok(c) <= 100 for c in chunks)
    assert len(chunks) >= 5


def test_summarise_chunked_targets_sum_to_target() -> None:
    seen: list[int] = []

    def prompt_fn(text: str, share: int) -> str:
        seen.append(share)
        return text

    out = lsum.summarise(
        "\n\n".join(["alpha " * 40] * 6),
        300,
        prompt_fn,
        lambda p, n: "<think>x</think> ok",
        limit=100,
    )
    assert len(seen) > 1
    assert abs(sum(seen) - 300) <= len(seen)
    assert out.count("ok") == len(seen)

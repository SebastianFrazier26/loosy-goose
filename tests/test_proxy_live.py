"""End-to-end smoke of `loosy-goose proxy` in front of a real `mlx_lm.server`.

Opt-in, with `-s` to see the per-turn table:
`uv run --extra embed --extra proxy --extra slot pytest -m live -s tests/test_proxy_live.py`.
Needs Apple silicon, the slot and proxy extras, and `Qwen/Qwen3-4B-MLX-4bit` in the Hugging Face
cache. Do not run it while another job holds the GPU.
"""

from __future__ import annotations

import json
import platform
import random
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from importlib.util import find_spec
from pathlib import Path

import pytest

from loosy_goose.tokens import count_tokens

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason="mlx_lm.server needs Apple silicon",
    ),
    pytest.mark.skipif(
        any(find_spec(m) is None for m in ("mlx_lm", "fastapi", "uvicorn", "httpx")),
        reason="needs the slot and proxy extras",
    ),
]

MODEL = "Qwen/Qwen3-4B-MLX-4bit"
TURNS = 12
FROZEN_BY = 8
RESULT_TOKENS = 1500
STARTUP_S = 300
METAL_CACHE_BYTES = 512 << 20
# The server's default prompt cache is unbounded: nine 8k-token sequences reached 8.8 GB on a
# 16 GB machine and the Metal command buffer died with out-of-memory on turn 10.
PROMPT_CACHE_BYTES = 1 << 30

_VOCAB = (
    "alpha bridge cache driver engine filter gradient handler index journal kernel ledger "
    "module node output parser queue router socket thread update vector worker yield buffer "
    "client domain event frame graph header input loader mapper object packet record schema "
    "token unit value window export import commit branch deploy config build lint trace"
).split()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _tool_result(turn: int) -> str:
    rng = random.Random(turn)
    lines: list[str] = []
    text = f"$ grep -rn handler src/ (run {turn})\n"
    while count_tokens(text) < RESULT_TOKENS:
        words = " ".join(rng.choice(_VOCAB) for _ in range(9))
        lines.append(
            f"src/{rng.choice(_VOCAB)}/{rng.choice(_VOCAB)}.py:{rng.randrange(1, 900)}: {words}"
        )
        text = f"$ grep -rn handler src/ (run {turn})\n" + "\n".join(lines)
    return text


def _wait_for(url: str, proc: subprocess.Popen[bytes], timeout: float) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"process behind {url} exited with {proc.returncode}")
        try:
            httpx.get(url, timeout=2.0)
            return
        except httpx.HTTPError:
            time.sleep(1.0)
    pytest.fail(f"{url} did not come up within {timeout:.0f}s")


def _stop(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _launch(argv: list[str], log: Path) -> subprocess.Popen[bytes]:
    with log.open("wb") as f:
        return subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT)


# mlx_lm.server has no flag for the Metal buffer cache; the shared GPU gets the same 512 MB
# cap the experiments ran under, set in-process before the server module runs.
_SERVER = (
    "import runpy, sys, mlx.core as mx; mx.set_cache_limit({cap}); "
    "sys.argv = ['mlx_lm.server', *sys.argv[1:]]; "
    "runpy.run_module('mlx_lm.server', run_name='__main__')"
)


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    state = tmp_path / "state"
    upstream_port, proxy_port = _free_port(), _free_port()
    procs: list[subprocess.Popen[bytes]] = []
    try:
        upstream = _launch(
            [
                sys.executable, "-c", _SERVER.format(cap=METAL_CACHE_BYTES),
                "--model", MODEL,
                "--host", "127.0.0.1", "--port", str(upstream_port),
                "--chat-template-args", '{"enable_thinking": false}',
                "--prompt-cache-bytes", str(PROMPT_CACHE_BYTES),
            ],
            tmp_path / "upstream.log",
        )  # fmt: skip
        procs.append(upstream)
        _wait_for(f"http://127.0.0.1:{upstream_port}/v1/models", upstream, STARTUP_S)
        proxy = _launch(
            [
                sys.executable, "-m", "loosy_goose.cli", "proxy",
                "--upstream", f"http://127.0.0.1:{upstream_port}",
                "--port", str(proxy_port), "--state-dir", str(state),
            ],
            tmp_path / "proxy.log",
        )  # fmt: skip
        procs.append(proxy)
        _wait_for(f"http://127.0.0.1:{proxy_port}/v1/models", proxy, 120)
        yield f"http://127.0.0.1:{proxy_port}", state
    finally:
        for p in reversed(procs):
            _stop(p)


def _chunks(state: Path) -> list[dict[str, object]]:
    files = list(state.glob("*.json"))
    assert len(files) == 1, "one conversation, one state file"
    chunks = json.loads(files[0].read_text(encoding="utf-8"))["chunks"]
    assert isinstance(chunks, list)
    return chunks


def test_twelve_turn_loop_freezes_and_keeps_chunk_text(stack: tuple[str, Path]) -> None:
    import httpx

    base, state = stack
    messages: list[dict[str, str]] = [{"role": "system", "content": "You are a terse assistant."}]
    frozen_after_8: str | None = None
    started = time.monotonic()
    for turn in range(1, TURNS + 1):
        sent = _tool_result(turn)
        messages.append(
            {
                "role": "user",
                "content": f"{sent}\n\nReply with one short sentence acknowledging result {turn}.",
            }
        )
        t0 = time.monotonic()
        resp = httpx.post(
            f"{base}/v1/chat/completions",
            json={"model": MODEL, "messages": messages, "max_tokens": 48, "temperature": 0},
            timeout=600.0,
        )
        assert resp.status_code == 200, f"turn {turn}: {resp.status_code}"
        body = resp.json()
        reply = body["choices"][0]["message"]["content"]
        assert reply and reply.strip(), f"turn {turn}: empty content"
        messages.append({"role": "assistant", "content": reply})

        full = sum(count_tokens(m["content"]) for m in messages[:-1])
        n_chunks = len(_chunks(state)) if list(state.glob("*.json")) else 0
        print(
            f"turn {turn:2d}: uncompacted~{full} prompt_tokens={body['usage']['prompt_tokens']} "
            f"chunks={n_chunks} {time.monotonic() - t0:.1f}s"
        )

        if turn == FROZEN_BY:
            chunks = _chunks(state)
            assert len(chunks) >= 1
            frozen_after_8 = str(chunks[0]["text"])
        elif turn > FROZEN_BY:
            assert frozen_after_8 is not None
            assert str(_chunks(state)[0]["text"]) == frozen_after_8, f"turn {turn}: chunk moved"
    print(f"total {time.monotonic() - started:.1f}s")

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from loosy_goose.cli import main  # noqa: E402
from loosy_goose.continuous import HEADER, ContinuousConfig, StateStore  # noqa: E402
from loosy_goose.proxy import create_app, prepare_state_dir  # noqa: E402
from loosy_goose.segment import Segment, segment  # noqa: E402
from loosy_goose.transcript import Transcript  # noqa: E402

UPSTREAM = "http://upstream.test"
SMALL = ContinuousConfig(tail_tokens=600, chunk_tokens=500)


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
        msgs.append({"role": "assistant", "content": _words(size, f"a{r}x")})
        msgs.append({"role": "user", "content": _words(size, f"u{r}x")})
    return msgs


def _chunks(parts: list[bytes]) -> AsyncIterator[bytes]:
    async def gen() -> AsyncIterator[bytes]:
        for p in parts:
            yield p

    return gen()


def _json(status: int, payload: Any, headers: dict[str, str] | None = None) -> httpx.Response:
    # Streamed like a real transport's response; a bytes body would arrive already read.
    h = {"content-type": "application/json", **(headers or {})}
    return httpx.Response(status, content=_chunks([json.dumps(payload).encode()]), headers=h)


class Upstream:
    def __init__(self, response: httpx.Response | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.response = response

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.response is not None:
            return self.response
        return _json(200, {"ok": True}, {"x-upstream": "yes"})


def _client(tmp_path: Path, upstream: Upstream) -> TestClient:
    app = create_app(
        UPSTREAM,
        SMALL,
        StateStore(tmp_path / "state"),
        transport=httpx.MockTransport(upstream),
        compress=keep_every_third,
    )
    return TestClient(app)


def test_forwards_compacted_messages(tmp_path: Path) -> None:
    up = Upstream()
    msgs = _messages(12)
    with _client(tmp_path, up) as c:
        r = c.post("/v1/chat/completions", json={"model": "m", "messages": msgs})
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert r.headers["x-upstream"] == "yes"
    sent = json.loads(up.requests[0].content)
    assert str(up.requests[0].url) == f"{UPSTREAM}/v1/chat/completions"
    assert sent["model"] == "m"
    assert len(sent["messages"]) < len(msgs)
    assert any(HEADER in str(m["content"]) for m in sent["messages"])
    assert list((tmp_path / "state").glob("*.json"))


def test_short_conversation_forwarded_unchanged(tmp_path: Path) -> None:
    up = Upstream()
    msgs = _messages(0, size=10)
    with _client(tmp_path, up) as c:
        c.post("/v1/chat/completions", json={"model": "m", "messages": msgs})
    assert json.loads(up.requests[0].content)["messages"] == msgs


def test_streaming_bytes_relayed_in_order(tmp_path: Path) -> None:
    parts = [f'data: {{"n": {i}}}\n\n'.encode() for i in range(20)] + [b"data: [DONE]\n\n"]
    sse = {"content-type": "text/event-stream"}
    up = Upstream(httpx.Response(200, content=_chunks(parts), headers=sse))
    with _client(tmp_path, up) as c:
        body = {"model": "m", "stream": True, "messages": _messages(1, size=10)}
        with c.stream("POST", "/v1/chat/completions", json=body) as r:
            got = b"".join(r.iter_raw())
            assert r.headers["content-type"].startswith("text/event-stream")
    assert got == b"".join(parts)


def test_authorization_passed_through_not_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    up = Upstream()
    msgs = _messages(12)
    with _client(tmp_path, up) as c:
        c.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": msgs},
            headers={"Authorization": "Bearer sekrit-token-123"},
        )
    assert up.requests[0].headers["authorization"] == "Bearer sekrit-token-123"
    assert "sekrit-token-123" not in caplog.text
    assert "ask5" not in caplog.text and "u3x7" not in caplog.text


def test_non_loopback_host_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["proxy", "--upstream", UPSTREAM, "--host", "0.0.0.0"])
    assert exc.value.code == 2
    assert "loopback" in capsys.readouterr().err


def test_other_v1_paths_passthrough(tmp_path: Path) -> None:
    up = Upstream(_json(200, {"data": [{"id": "m"}]}))
    with _client(tmp_path, up) as c:
        r = c.get("/v1/models?x=1", headers={"Authorization": "Bearer k"})
    assert r.json() == {"data": [{"id": "m"}]}
    assert str(up.requests[0].url) == f"{UPSTREAM}/v1/models?x=1"
    assert up.requests[0].headers["authorization"] == "Bearer k"


def test_state_dir_private_and_pruned(tmp_path: Path) -> None:
    root = tmp_path / "state"
    prepare_state_dir(root)
    assert (root.stat().st_mode & 0o777) == 0o700
    old, fresh, other = root / ("a" * 32 + ".json"), root / ("b" * 32 + ".json"), root / "keep.json"
    for p in (old, fresh, other):
        p.write_text("{}")
    week_ago = time.time() - 8 * 86400
    os.utime(old, (week_ago, week_ago))
    os.utime(other, (week_ago, week_ago))
    prepare_state_dir(root)
    assert not old.exists() and fresh.exists() and other.exists()

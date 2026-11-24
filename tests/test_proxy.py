import asyncio
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
from loosy_goose.proxy import SESSION_HEADER, create_app, prepare_state_dir  # noqa: E402
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
        self.raises = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.raises:
            raise httpx.ConnectError("refused", request=request)
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
    return TestClient(app, base_url="http://127.0.0.1:8765")


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
    raw = json.dumps({"model": "m", "messages": _messages(0, size=10)}, indent=3).encode()
    with _client(tmp_path, up) as c:
        c.post("/v1/chat/completions", content=raw, headers={"content-type": "application/json"})
    assert up.requests[0].content == raw


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
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    prepare_state_dir(root)
    # Windows has no POSIX permission bits: stat reports 0o777 for every directory.
    if os.name == "posix":
        assert (root.stat().st_mode & 0o777) == 0o700
    old, fresh, other = root / ("a" * 32 + ".json"), root / ("b" * 32 + ".json"), root / "keep.json"
    for p in (old, fresh, other):
        p.write_text("{}")
    week_ago = time.time() - 8 * 86400
    os.utime(old, (week_ago, week_ago))
    os.utime(other, (week_ago, week_ago))
    prepare_state_dir(root)
    assert not old.exists() and fresh.exists() and other.exists()


@pytest.mark.parametrize("raw", [b"not json {", b'{"model": "m", "prompt": "hi"}'])
def test_uncompactable_body_forwarded_byte_identical(tmp_path: Path, raw: bytes) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        r = c.post(
            "/v1/chat/completions", content=raw, headers={"content-type": "application/json"}
        )
    assert r.status_code == 200
    assert up.requests[0].content == raw


def test_upstream_error_status_and_body_relayed(tmp_path: Path) -> None:
    up = Upstream(_json(422, {"error": "bad model"}, {"x-why": "nope"}))
    with _client(tmp_path, up) as c:
        r = c.post("/v1/chat/completions", json={"model": "m", "messages": _messages(12)})
    assert r.status_code == 422
    assert r.json() == {"error": "bad model"}
    assert r.headers["x-why"] == "nope"


def test_upstream_connect_error_is_502(tmp_path: Path) -> None:
    up = Upstream()
    up.raises = True
    with _client(tmp_path, up) as c:
        r = c.get("/v1/models")
    assert r.status_code == 502


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "10.0.0.5:8765"])
def test_foreign_host_header_refused(tmp_path: Path, host: str) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        r1 = c.get("/v1/models", headers={"host": host})
        r2 = c.post("/v1/chat/completions", json={"messages": []}, headers={"host": host})
    assert r1.status_code == r2.status_code == 403
    assert r1.json()["error"]["message"] == "loosy-goose: host not allowed"
    assert not up.requests


@pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765", "[::1]:8765", "localhost"])
def test_loopback_host_header_allowed(tmp_path: Path, host: str) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        r = c.get("/v1/models", headers={"host": host})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "path",
    [
        "/v1/%2e%2e/admin",
        "/v1/..%2fadmin",
        "/v1/x%3Fq=1",
        "/v1/x%23frag",
        "/v1/%2E%2E/admin",
        "/v1/..%5cadmin",
    ],
)
def test_passthrough_cannot_leave_v1(tmp_path: Path, path: str) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        r = c.get(path)
    assert r.status_code == 404
    assert r.json()["error"]["message"] == "loosy-goose: path refused"
    assert not up.requests


def test_cli_binds_loopback_without_access_log(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    seen: dict[str, Any] = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    assert main(["proxy", "--upstream", UPSTREAM, "--host", "localhost"]) == 0
    assert seen["host"] == "127.0.0.1"
    assert seen["access_log"] is False


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "https://evil.example"},
        {"origin": "null"},
        {"origin": "http://localhost.evil.example"},
        {"origin": "http://127.0.0.1@evil.example"},
        {"origin": "ftp://localhost"},
        {"origin": "http://[::1"},
        {"sec-fetch-site": "cross-site"},
    ],
)
def test_cross_origin_refused(tmp_path: Path, headers: dict[str, str]) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        r1 = c.get("/v1/models", headers=headers)
        r2 = c.post("/v1/chat/completions", json={"messages": []}, headers=headers)
    assert r1.status_code == r2.status_code == 403
    assert r1.json()["error"]["message"] == "loosy-goose: cross-origin refused"
    assert not up.requests


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "http://localhost:3000"},
        {"origin": "https://127.0.0.1"},
        {"origin": "http://[::1]:8080"},
        {"sec-fetch-site": "same-origin"},
        {"sec-fetch-site": "none"},
    ],
)
def test_loopback_origin_allowed(tmp_path: Path, headers: dict[str, str]) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        assert c.get("/v1/models", headers=headers).status_code == 200


@pytest.mark.parametrize("ctype", [None, "text/plain", "application/x-www-form-urlencoded"])
def test_chat_requires_json_content_type(tmp_path: Path, ctype: str | None) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        r = c.post(
            "/v1/chat/completions",
            content=b'{"messages": []}',
            headers={"content-type": ctype} if ctype else {},
        )
    assert r.status_code == 415
    assert not up.requests


def test_json_content_type_with_params_accepted(tmp_path: Path) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        r = c.post(
            "/v1/chat/completions",
            content=b"{}",
            headers={"content-type": "Application/JSON; charset=utf-8"},
        )
    assert r.status_code == 200


def test_lone_surrogate_fails_closed_without_logging(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    up = Upstream()
    raw = b'{"messages": [{"role": "user", "content": "secret-ish \\ud800 text"}]}'
    with _client(tmp_path, up) as c:
        r = c.post(
            "/v1/chat/completions", content=raw, headers={"content-type": "application/json"}
        )
    assert r.status_code == 400
    assert not up.requests
    assert "secret-ish" not in caplog.text and "ud800" not in caplog.text


def test_non_latin1_response_header_relayed(tmp_path: Path) -> None:
    # Driven through ASGITransport: TestClient re-encodes response headers itself and would
    # fail on this value before the assertion.
    value = "\u20ac".encode()
    resp = httpx.Response(
        200,
        content=_chunks([b"{}"]),
        headers=[(b"content-type", b"application/json"), (b"x-note", value)],
    )
    app = create_app(
        UPSTREAM,
        SMALL,
        StateStore(tmp_path / "state"),
        transport=httpx.MockTransport(Upstream(resp)),
        compress=keep_every_third,
    )

    async def run() -> list[tuple[bytes, bytes]]:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as c:
                r = await c.get("/v1/models")
        assert r.status_code == 200
        return r.headers.raw

    assert (b"x-note", value) in asyncio.run(run())


def test_cli_rejects_upstream_ending_in_v1(capsys: pytest.CaptureFixture[str]) -> None:
    for url in ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080/v1/"):
        with pytest.raises(SystemExit) as exc:
            main(["proxy", "--upstream", url])
        assert exc.value.code == 2
    assert "/v1" in capsys.readouterr().err


def test_warms_embedding_model_only_without_injected_compressor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import loosy_goose.embed as embed

    calls: list[int] = []
    monkeypatch.setattr(embed, "warm", lambda *a: calls.append(1))
    app = create_app(
        UPSTREAM, SMALL, StateStore(tmp_path / "s"), transport=httpx.MockTransport(Upstream())
    )
    with TestClient(app, base_url="http://127.0.0.1:8765"):
        pass
    assert calls == [1]
    with _client(tmp_path, Upstream()):
        pass
    assert calls == [1]


def _counting_client(tmp_path: Path, upstream: Upstream, calls: list[int]) -> TestClient:
    def counted(t: Transcript, keep: float) -> list[Segment]:
        calls.append(1)
        return keep_every_third(t, keep)

    app = create_app(
        UPSTREAM,
        SMALL,
        StateStore(tmp_path / "state"),
        transport=httpx.MockTransport(upstream),
        compress=counted,
    )
    return TestClient(app, base_url="http://127.0.0.1:8765")


def _same_opening_pair() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    a = _messages(14)
    b = a[:2] + [{"role": m["role"], "content": str(m["content"]).replace("x", "y")} for m in a[2:]]
    return a, b


def _interleave(
    tmp_path: Path, sessions: tuple[str, str] | None
) -> tuple[int, int, list[dict[str, Any]]]:
    """Compress calls for A's second turn when B, same opening, ran in between, and for the
    same turn without B; plus what A's second turn sent upstream with B in between."""
    a, b = _same_opening_pair()
    ha, hb = ({SESSION_HEADER: s} for s in sessions) if sessions else ({}, {})
    up = Upstream()
    calls: list[int] = []
    with _counting_client(tmp_path / "mixed", up, calls) as c:
        c.post("/v1/chat/completions", json={"messages": a[:-2]}, headers=ha)
        c.post("/v1/chat/completions", json={"messages": b[:-2]}, headers=hb)
        before = len(calls)
        c.post("/v1/chat/completions", json={"messages": a}, headers=ha)
        mixed = len(calls) - before
    sent = json.loads(up.requests[-1].content)["messages"]
    alone_calls: list[int] = []
    with _counting_client(tmp_path / "alone", Upstream(), alone_calls) as c:
        c.post("/v1/chat/completions", json={"messages": a[:-2]}, headers=ha)
        before = len(alone_calls)
        c.post("/v1/chat/completions", json={"messages": a}, headers=ha)
        alone = len(alone_calls) - before
    return mixed, alone, sent


def test_same_opening_conversations_isolated_by_session_header(tmp_path: Path) -> None:
    mixed, alone, _ = _interleave(tmp_path, ("conv-a", "conv-b"))
    assert mixed == alone
    assert len(list((tmp_path / "mixed" / "state").glob("*.json"))) == 2


def test_same_opening_conversations_without_header_still_share_state(tmp_path: Path) -> None:
    mixed, alone, _ = _interleave(tmp_path, None)
    assert mixed > alone
    assert len(list((tmp_path / "mixed" / "state").glob("*.json"))) == 1


def test_session_isolation_does_not_change_output(tmp_path: Path) -> None:
    _, _, with_sessions = _interleave(tmp_path / "s", ("conv-a", "conv-b"))
    _, _, without = _interleave(tmp_path / "n", None)
    assert with_sessions == without


def test_session_header_not_forwarded(tmp_path: Path) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        c.post(
            "/v1/chat/completions", json={"messages": _messages(12)}, headers={SESSION_HEADER: "s1"}
        )
        c.get("/v1/models", headers={SESSION_HEADER: "s1"})
    assert all(SESSION_HEADER not in r.headers for r in up.requests)


@pytest.mark.parametrize("value", ["", "has space", "x" * 129, "café"])
def test_malformed_session_header_refused(tmp_path: Path, value: str) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"messages": _messages(12)},
            headers=[(SESSION_HEADER.encode(), value.encode("latin-1"))],  # type: ignore[arg-type]
        )
    assert r.status_code == 400
    assert not up.requests


def test_repeated_session_header_refused(tmp_path: Path) -> None:
    up = Upstream()
    with _client(tmp_path, up) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"messages": _messages(12)},
            headers=[(SESSION_HEADER, "a"), (SESSION_HEADER, "b")],
        )
    assert r.status_code == 400
    assert not up.requests


def test_cross_origin_refused_before_session_checked(tmp_path: Path) -> None:
    with _client(tmp_path, Upstream()) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"messages": []},
            headers={"origin": "https://evil.example", SESSION_HEADER: "bad value"},
        )
    assert r.status_code == 403

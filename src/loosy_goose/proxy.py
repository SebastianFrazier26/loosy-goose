"""Localhost proxy in front of a local model server: continuous compaction of each chat request.

Message content and request headers are never logged. Only conversation ids (hashes) and
message counts are.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .continuous import Compress, ContinuousConfig, FreezeCompactor, StateStore, conversation_id

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = Path.home() / ".cache" / "loosy-goose" / "continuous"
STATE_MAX_AGE_S = 7 * 86400

_STATE_FILE = re.compile(r"\.?[0-9a-f]{32}(\.json|\..*\.tmp)")
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "proxy-authorization",
        "proxy-connection",
    }
)
# Host-header allowlist: a DNS-rebinding page reaches a loopback port with its own Host name.
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})
_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]


def prepare_state_dir(root: Path, max_age_s: float = STATE_MAX_AGE_S) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    cutoff = time.time() - max_age_s
    # Only names the state store writes are pruned, so a --state-dir pointed at a shared
    # directory never loses anything else.
    for p in root.iterdir():
        if _STATE_FILE.fullmatch(p.name) and p.is_file() and p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)


def _host_allowed(host: str | None) -> bool:
    if not host:
        return False
    name = host.lower()
    if name.startswith("["):
        name = name[: name.find("]") + 1]
    else:
        name = name.rsplit(":", 1)[0]
    return name in _ALLOWED_HOSTS


def _drop_set(headers: list[tuple[str, str]]) -> frozenset[str]:
    named = {
        t.strip().lower()
        for k, v in headers
        if k.lower() == "connection"
        for t in v.split(",")
        if t.strip()
    }
    return _HOP_BY_HOP | named


def _safe_subpath(raw_path: bytes) -> str | None:
    """The still-encoded request path, or None when it could leave `/v1/` once decoded or
    normalised upstream."""
    path = raw_path.decode("latin-1")
    decoded = unquote(path)
    if not path.startswith("/v1/") or "?" in decoded or "#" in decoded or "\\" in decoded:
        return None
    if any(seg in (".", "..") for seg in decoded.split("/")):
        return None
    return path


def _compact(
    store: StateStore, config: ContinuousConfig, compress: Compress | None, cid: str, msgs: Any
) -> list[dict[str, Any]]:
    loaded = store.load(cid, config)
    compactor = FreezeCompactor(config, compress=compress, chunks=loaded)
    out = compactor.compact(msgs)
    if compactor.chunks != loaded:
        store.save(cid, config, compactor.chunks)
    return out


def create_app(
    upstream: str,
    config: ContinuousConfig,
    store: StateStore,
    transport: httpx.AsyncBaseTransport | None = None,
    compress: Compress | None = None,
) -> FastAPI:
    base = upstream.rstrip("/")
    locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
    holder: dict[str, httpx.AsyncClient] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        prepare_state_dir(store.root)
        # No read timeout: a local model can take minutes before its first token.
        async with httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(None, connect=10.0)
        ) as client:
            holder["client"] = client
            yield

    app = FastAPI(lifespan=lifespan, openapi_url=None, docs_url=None, redoc_url=None)

    base_path = urlsplit(base).path
    pending: set[asyncio.Task[list[dict[str, Any]]]] = set()

    def _settle(task: asyncio.Task[list[dict[str, Any]]]) -> None:
        pending.discard(task)
        # Retrieved here so an orphaned task's exception is never printed by asyncio with a
        # traceback that may quote conversation content.
        if not task.cancelled():
            task.exception()

    async def forward(request: Request, path: str, content: bytes) -> Response:
        in_headers = request.headers.items()
        drop = _drop_set(in_headers) | {"host", "content-length"}
        url = httpx.URL(base + path)
        if query := request.scope.get("query_string"):
            url = url.copy_with(query=query)
        req = holder["client"].build_request(
            request.method,
            url,
            headers=[(k, v) for k, v in in_headers if k.lower() not in drop],
            content=content,
        )
        if not req.url.path.startswith(base_path + "/v1/"):
            return JSONResponse({"error": {"message": "loosy-goose: path refused"}}, 404)
        try:
            resp = await holder["client"].send(req, stream=True)
        except httpx.HTTPError as exc:
            logger.warning("upstream request failed: %s", type(exc).__name__)
            return JSONResponse({"error": {"message": "loosy-goose: upstream unreachable"}}, 502)

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            except httpx.HTTPError as exc:
                logger.warning("upstream stream failed: %s", type(exc).__name__)
            finally:
                await resp.aclose()

        out_headers = resp.headers.multi_items()
        out_drop = _drop_set(out_headers)
        out = StreamingResponse(
            body(), status_code=resp.status_code, background=BackgroundTask(resp.aclose)
        )
        # Set raw so repeated headers (set-cookie and the like) are not merged into one.
        out.raw_headers = [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in out_headers
            if k.lower() not in out_drop
        ]
        return out

    def refused(request: Request) -> Response | None:
        if not _host_allowed(request.headers.get("host")):
            return JSONResponse({"error": {"message": "loosy-goose: host not allowed"}}, 403)
        return None

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        if (no := refused(request)) is not None:
            return no
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        msgs = body.get("messages") if isinstance(body, dict) else None
        if not (isinstance(msgs, list) and msgs and all(isinstance(m, dict) for m in msgs)):
            return await forward(request, "/v1/chat/completions", raw)
        cid = conversation_id(msgs)
        lock = locks.get(cid)
        if lock is None:
            lock = locks[cid] = asyncio.Lock()

        async def locked() -> list[dict[str, Any]]:
            async with lock:
                return await asyncio.to_thread(_compact, store, config, compress, cid, msgs)

        # The lock is held inside a shielded task, so a client that disconnects mid-compaction
        # cannot release it while the worker thread is still writing this conversation's state.
        task = asyncio.ensure_future(locked())
        pending.add(task)
        task.add_done_callback(_settle)
        try:
            compacted = await asyncio.shield(task)
        except Exception as exc:
            # The type only: an exception's message can quote conversation content.
            logger.error("compaction failed for %s: %s", cid, type(exc).__name__)
            return JSONResponse({"error": {"message": "loosy-goose: compaction failed"}}, 500)
        logger.info("conversation %s: %d -> %d messages", cid, len(msgs), len(compacted))
        if compacted == msgs:
            return await forward(request, "/v1/chat/completions", raw)
        content = json.dumps({**body, "messages": compacted}, ensure_ascii=False).encode("utf-8")
        return await forward(request, "/v1/chat/completions", content)

    @app.api_route("/v1/{path:path}", methods=_METHODS)
    async def passthrough(request: Request, path: str) -> Response:
        if (no := refused(request)) is not None:
            return no
        sub = _safe_subpath(request.scope.get("raw_path") or request.url.path.encode("latin-1"))
        if sub is None:
            return JSONResponse({"error": {"message": "loosy-goose: path refused"}}, 404)
        return await forward(request, sub, await request.body())

    return app

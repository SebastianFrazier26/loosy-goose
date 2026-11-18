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

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from .continuous import Compress, ContinuousConfig, FreezeCompactor, StateStore, conversation_id

logger = logging.getLogger(__name__)

DEFAULT_STATE_DIR = Path.home() / ".cache" / "loosy-goose" / "continuous"
STATE_MAX_AGE_S = 7 * 86400

_STATE_FILE = re.compile(r"\.?[0-9a-f]{32}(\.json|\..*\.tmp)")
_DROP_REQUEST = frozenset({"host", "content-length", "transfer-encoding", "connection"})
_DROP_RESPONSE = frozenset({"transfer-encoding", "connection", "keep-alive"})
_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]


def prepare_state_dir(root: Path, max_age_s: float = STATE_MAX_AGE_S) -> None:
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    cutoff = time.time() - max_age_s
    # Only names the state store writes are pruned, so a --state-dir pointed at a shared
    # directory never loses anything else.
    for p in root.iterdir():
        if _STATE_FILE.fullmatch(p.name) and p.is_file() and p.stat().st_mtime < cutoff:
            p.unlink(missing_ok=True)


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

    async def forward(request: Request, path: str, content: bytes, stream: bool) -> Response:
        headers = [(k, v) for k, v in request.headers.items() if k.lower() not in _DROP_REQUEST]
        req = holder["client"].build_request(
            request.method,
            base + path,
            params=str(request.query_params),
            headers=headers,
            content=content,
        )
        try:
            resp = await holder["client"].send(req, stream=True)
        except httpx.HTTPError as exc:
            logger.warning("upstream request failed: %s", type(exc).__name__)
            return JSONResponse({"error": {"message": "loosy-goose: upstream unreachable"}}, 502)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _DROP_RESPONSE}
        if stream:
            return StreamingResponse(
                resp.aiter_raw(),
                status_code=resp.status_code,
                headers=out_headers,
                background=BackgroundTask(resp.aclose),
            )
        try:
            body = b"".join([c async for c in resp.aiter_raw()])
        finally:
            await resp.aclose()
        return Response(body, status_code=resp.status_code, headers=out_headers)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        msgs = body.get("messages") if isinstance(body, dict) else None
        if not (isinstance(msgs, list) and msgs and all(isinstance(m, dict) for m in msgs)):
            return await forward(request, "/v1/chat/completions", raw, stream=False)
        cid = conversation_id(msgs)
        lock = locks.get(cid)
        if lock is None:
            lock = locks[cid] = asyncio.Lock()
        try:
            async with lock:
                compacted = await asyncio.to_thread(_compact, store, config, compress, cid, msgs)
        except Exception as exc:
            # The type only: an exception's message can quote conversation content.
            logger.error("compaction failed for %s: %s", cid, type(exc).__name__)
            return JSONResponse({"error": {"message": "loosy-goose: compaction failed"}}, 500)
        logger.info("conversation %s: %d -> %d messages", cid, len(msgs), len(compacted))
        content = json.dumps({**body, "messages": compacted}, ensure_ascii=False).encode("utf-8")
        return await forward(
            request, "/v1/chat/completions", content, stream=body.get("stream") is True
        )

    @app.api_route("/v1/{path:path}", methods=_METHODS)
    async def passthrough(request: Request, path: str) -> Response:
        return await forward(request, f"/v1/{path}", await request.body(), stream=False)

    return app

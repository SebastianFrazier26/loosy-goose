"""MCP server: `handoff_session` forks the current Claude Code or Codex session into a compressed
copy and says how to resume it; `compress` returns compressed turns and writes nothing."""

from __future__ import annotations

from collections.abc import Mapping
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from loosy_goose.embed import warm
from loosy_goose.handoff import HandoffError, HandoffOptions, handoff
from loosy_goose.pipeline import PipelineOptions, compress_transcript, selector_label, to_turns
from loosy_goose.rate import LEVELS, check_keep, resolve_rate
from loosy_goose.segment import segment
from loosy_goose.sessions.model import Tool
from loosy_goose.sessions.plan import PROTECT_TAIL_TOKENS, clean_transcript, recompress_warning
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Transcript, load_claude_code_jsonl, load_messages_json

INSTALL_HINT = (
    'reinstall with it: `uv tool install --editable "<your clone>[embed,mcp]"`, or `uv tool '
    'install --torch-backend cpu "loosy-goose[embed,mcp] @ '
    'git+https://github.com/SebastianFrazier26/loosy-goose"`'
)
_INSTRUCTIONS = (
    "loosy-goose compresses agent conversations extractively. handoff_session forks the current "
    "Claude Code or Codex session into a compressed copy (kept turns, a verbatim recent tail, a "
    "marker) and returns the command to resume it; the original session is never modified."
)


def _check_rate(level: str | None, keep: float | None, target_tokens: int | None) -> None:
    given = [
        name
        for name, value in (("level", level), ("keep", keep), ("target_tokens", target_tokens))
        if value is not None
    ]
    if len(given) > 1:
        raise ToolError(f"pass only one of level, keep, target_tokens (got {', '.join(given)})")
    if level is not None and level not in LEVELS:
        raise ToolError(f"level must be one of {', '.join(LEVELS)}, got {level!r}")
    if keep is not None:
        try:
            check_keep(keep)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
    if target_tokens is not None and target_tokens < 1:
        raise ToolError(f"target_tokens must be a positive integer, got {target_tokens}")


def _transcript(messages: list[dict[str, Any]] | None, path: str | None) -> Transcript:
    if (messages is None) == (path is None):
        raise ToolError("pass exactly one of messages or path")
    if messages is not None:
        return load_messages_json(messages)
    source = Path(str(path))
    if not source.is_file():
        raise ToolError(f"no such file: {source}")
    if source.suffix == ".jsonl":
        return load_claude_code_jsonl(source)
    if source.suffix == ".json":
        return load_messages_json(source)
    raise ToolError(f"unsupported transcript suffix {source.suffix!r} (expected .jsonl or .json)")


def build_server(*, homes: Mapping[Tool, Path] | None = None) -> MCPServer:
    server = MCPServer("loosy-goose", instructions=_INSTRUCTIONS)

    @server.tool()
    def compress(
        messages: list[dict[str, Any]] | None = None,
        path: str | None = None,
        level: str | None = None,
        keep: float | None = None,
        target_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Compress a conversation, given as [{role, content}] messages or a .jsonl/.json path,
        and return the kept turns. Writes nothing. Rate: one of level (light|medium|heavy,
        default medium), keep (0.10-0.80) or target_tokens."""
        _check_rate(level, keep, target_tokens)
        transcript, prior = clean_transcript(_transcript(messages, path))
        input_tokens = (
            sum(count_tokens(s.text) for s in segment(transcript))
            if target_tokens is not None
            else 0
        )
        rate = resolve_rate(keep, level, target_tokens, input_tokens)
        opts = PipelineOptions(keep=rate.keep, budget=rate.budget)
        result = compress_transcript(transcript, opts)
        header = {
            "keep": opts.keep,
            "keep_source": rate.source,
            "level": rate.level,
            "target_tokens": rate.target_tokens,
            "selector": selector_label(opts),
            "budget": result.budget,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "atom_recall_final": result.atom_recall_final,
            "notes": list(rate.notes),
            "warnings": [recompress_warning(prior)] if prior is not None else [],
        }
        turns = [
            {
                "index": t.index,
                "role": t.role,
                "blocks": [{"kind": b.kind, "text": b.text, "meta": b.meta} for b in t.blocks],
            }
            for t in to_turns(result, transcript)
        ]
        return {"header": header, "turns": turns}

    @server.tool()
    def handoff_session(
        level: str | None = None,
        keep: float | None = None,
        target_tokens: int | None = None,
        session: str = "current",
        tool: Literal["auto", "claude", "codex"] = "auto",
        cwd: str | None = None,
        verify: bool = False,
        protect_tail_tokens: int = PROTECT_TAIL_TOKENS,
    ) -> dict[str, Any]:
        """Fork the current Claude Code or Codex session into a compressed copy and return how
        to resume it, plus the command back to the original, which is never modified. Pass
        cwd (the session's working directory) and tool. verify=true resumes a throwaway copy
        headless first, which sends the compressed history to the model."""
        _check_rate(level, keep, target_tokens)
        if protect_tail_tokens < 0:
            raise ToolError(f"protect_tail_tokens must be >= 0, got {protect_tail_tokens}")
        opts = HandoffOptions(
            level=level,
            keep=keep,
            target_tokens=target_tokens,
            session=session,
            tool=tool,
            cwd=Path(cwd) if cwd else Path.cwd(),
            verify=verify,
            protect_tail_tokens=protect_tail_tokens,
            homes=dict(homes) if homes else None,
        )
        try:
            return handoff(opts).to_dict()
        except HandoffError as exc:
            raise ToolError(str(exc)) from exc

    return server


def preload_model() -> None:
    if find_spec("sentence_transformers") is None:
        raise SystemExit(f"loosy-goose-mcp needs the `embed` extra; {INSTALL_HINT}")
    warm()


def main() -> None:
    # Loaded before serving: a first /loosy that has to load the model can outrun the client's
    # tool-call timeout.
    preload_model()
    build_server().run(transport="stdio")

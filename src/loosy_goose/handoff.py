"""Fork the live Claude Code or Codex session into a compressed copy the user resumes.

The original session file is only ever read, through a snapshot. The only files this module
deletes are forks of its own: one that failed validation, the live-check copy, and a fork whose
live check failed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from loosy_goose.pipeline import PipelineOptions, compress_transcript
from loosy_goose.rate import resolve_rate
from loosy_goose.segment import segment
from loosy_goose.sessions import claude, codex
from loosy_goose.sessions.adapter import SessionAdapter
from loosy_goose.sessions.model import (
    Session,
    SessionFormatError,
    Tool,
    newer_than_tested,
    snapshot,
)
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    PROTECT_TAIL_TOKENS,
    ForkPlan,
    HandoffRefusedError,
    Region,
    Trigger,
    find_prior_fork,
    find_trigger,
    plan_fork,
    recompress_warning,
    region_tokens,
    split_region,
)
from loosy_goose.tokens import count_tokens

ADAPTERS: dict[Tool, SessionAdapter] = {"claude": claude, "codex": codex}
_NAMES: dict[str, str] = {"auto": "Claude Code or Codex", "claude": "Claude Code", "codex": "Codex"}
# Flip to True if the drift test shows a resume failing on dropped thinking (spec open item).
DEFAULT_KEEP_THINKING = False
CANDIDATES_CHECKED = 3
CONCURRENT_WINDOW_S = 600
VERIFY_TIMEOUT_S = 300

# (exit code, stdout).
Runner = Callable[[list[str], Path], tuple[int, str]]


class HandoffError(Exception):
    """The handoff stopped; the user's session is exactly as it was."""


@dataclass(frozen=True)
class HandoffOptions:
    level: str | None = None
    keep: float | None = None
    target_tokens: int | None = None
    session: str = "current"
    tool: Literal["auto", "claude", "codex"] = "auto"
    cwd: Path = field(default_factory=Path.cwd)
    verify: bool = False
    protect_tail_tokens: int = PROTECT_TAIL_TOKENS
    keep_thinking: bool = DEFAULT_KEEP_THINKING
    homes: dict[Tool, Path] | None = None


@dataclass(frozen=True)
class HandoffResult:
    status: Literal["forked", "nothing-to-do"]
    reason: str | None
    tool: Tool
    source_session: str
    source_path: str
    original_resume_command: str
    fork_session: str | None
    fork_path: str | None
    resume_command: str | None
    before_tokens: int
    after_tokens: int
    tail_tokens: int
    keep: float | None
    first_prompt: str
    last_prompt: str
    trigger: str
    verified: bool | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Picked:
    adapter: SessionAdapter
    source: Path
    session: Session
    trigger: Trigger
    warnings: list[str]


def _homes(opts: HandoffOptions) -> dict[Tool, Path]:
    given = opts.homes or {}
    return {tool: given.get(tool) or adapter.default_home() for tool, adapter in ADAPTERS.items()}


def _candidates(opts: HandoffOptions, homes: dict[Tool, Path]) -> list[tuple[SessionAdapter, Path]]:
    tools: tuple[Tool, ...] = ("claude", "codex") if opts.tool == "auto" else (opts.tool,)
    found = [
        (ADAPTERS[tool], path)
        for tool in tools
        for path in ADAPTERS[tool].locate(opts.cwd, opts.session, homes[tool])
    ]
    return sorted(found, key=lambda c: c[1].stat().st_mtime, reverse=True)


def _pick(
    candidates: list[tuple[SessionAdapter, Path]], opts: HandoffOptions, work: Path
) -> _Picked:
    notes: list[str] = []
    fresh = [p for _, p in candidates if time.time() - p.stat().st_mtime < CONCURRENT_WINDOW_S]
    if opts.session == "current" and len(fresh) > 1:
        notes.append(
            f"{len(fresh)} sessions in {opts.cwd} changed in the last "
            f"{CONCURRENT_WINDOW_S // 60} minutes; check the first and last prompt below are "
            "the session you meant, or pass session=<id>"
        )
    read: list[_Picked] = []
    for i, (adapter, path) in enumerate(candidates[:CANDIDATES_CHECKED]):
        snap = work / f"snapshot-{i}.jsonl"
        dropped = snapshot(path, snap)
        try:
            session = adapter.read(snap)
        except SessionFormatError as exc:
            if opts.session != "current":
                raise HandoffError(f"{exc}; nothing was written") from exc
            notes.append(f"skipped {path}: {exc}")
            continue
        own = (
            [f"ignored a partial last line ({dropped} bytes) the CLI is still writing"]
            if dropped
            else []
        )
        trigger = find_trigger(session, DEFAULT_TRIGGERS[adapter.TOOL])
        read.append(_Picked(adapter, path, session, trigger, own))
        if trigger.how != "none" or opts.session != "current":
            break
    if not read:
        raise HandoffError(f"no readable session found for {opts.cwd}; nothing was written")
    chosen = next((p for p in read if p.trigger.how != "none"), read[0])
    if chosen.trigger.how == "none":
        notes.append(
            "handoff trigger not found (no /loosy turn, open turn or handoff_session call); "
            "nothing was excluded from the end of the session"
        )
    chosen.warnings = notes + chosen.warnings
    return chosen


def _clip(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _total(session: Session) -> int:
    return sum(session.tokens(ri) for ri in session.live)


def _run(argv: list[str], cwd: Path) -> tuple[int, str]:
    try:
        done = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=VERIFY_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HandoffError(f"the live check could not run `{argv[0]}`: {exc}") from exc
    return done.returncode, done.stdout


def _place(staged: Path, dest: Path) -> None:
    """Move `staged` to `dest` without ever replacing a file already there.

    A rename would silently overwrite a session; link-then-unlink and exclusive create both fail
    instead. The staged file lives in a temp dir, often on another filesystem, so the exclusive
    copy is the common path and the hard link the fast one.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        try:
            os.link(staged, dest)
        except FileExistsError:
            raise
        except OSError:
            with staged.open("rb") as src, dest.open("xb") as out:
                try:
                    shutil.copyfileobj(src, out)
                except BaseException:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise
    except FileExistsError as exc:
        raise HandoffError(
            f"{dest} already exists; it was left untouched and nothing was written"
        ) from exc
    staged.unlink()


def _verify(
    p: _Picked, plan: ForkPlan, home: Path, stamp: datetime, cwd: Path, work: Path, runner: Runner
) -> bool:
    check_id = p.adapter.new_session_id()
    staged = p.adapter.write_fork(p.session, plan, check_id, work / "verify")
    copy = p.adapter.fork_destination(p.source, check_id, home, stamp)
    _place(staged, copy)
    try:
        code, reply = runner(p.adapter.verify_command(check_id), cwd)
        # Exact, not a substring: an error message containing "TOKEN" or "BOOK" is no answer.
        return code == 0 and reply.strip() == "OK"
    finally:
        copy.unlink(missing_ok=True)


def handoff(
    opts: HandoffOptions,
    *,
    now: datetime | None = None,
    runner: Runner | None = None,
    cache_dir: Path | None = None,
) -> HandoffResult:
    stamp = now or datetime.now()
    homes = _homes(opts)
    candidates = _candidates(opts, homes)
    if not candidates:
        which = "" if opts.session == "current" else f" with id {opts.session}"
        raise HandoffError(
            f"no {_NAMES[opts.tool]} session found for {opts.cwd}{which}; nothing was written"
        )
    with tempfile.TemporaryDirectory(prefix="loosy-goose-") as tmp:
        work = Path(tmp)
        picked = _pick(candidates, opts, work)
        return _fork(picked, opts, homes, work, stamp, runner or _run, cache_dir)


def _fork(
    p: _Picked,
    opts: HandoffOptions,
    homes: dict[Tool, Path],
    work: Path,
    stamp: datetime,
    runner: Runner,
    cache_dir: Path | None,
) -> HandoffResult:
    adapter, session = p.adapter, p.session
    warnings = p.warnings + session.warnings
    if newer_than_tested(session.cli_version, adapter.TESTED_VERSIONS):
        warnings.append(
            f"{adapter.TOOL} {session.cli_version} is newer than the tested "
            f"{', '.join(adapter.TESTED_VERSIONS)}; its session format may have changed"
        )
    region: Region = split_region(session, p.trigger, opts.protect_tail_tokens)
    prior = find_prior_fork(session, region.end)
    if prior is not None:
        warnings.append(recompress_warning(prior))
    history = session.live[: region.end]
    prompts = [r.prompt for ri in history if (r := session.records[ri]).prompt is not None]
    before = sum(session.tokens(ri) for ri in history)
    compressible, tail = region_tokens(session, region)

    def result(
        status: Literal["forked", "nothing-to-do"],
        *,
        reason: str | None = None,
        fork_id: str | None = None,
        fork_path: Path | None = None,
        after: int | None = None,
        keep: float | None = None,
        verified: bool | None = None,
    ) -> HandoffResult:
        return HandoffResult(
            status=status,
            reason=reason,
            tool=adapter.TOOL,
            source_session=session.session_id,
            source_path=str(p.source),
            original_resume_command=adapter.resume_command(session.session_id, session.cwd),
            fork_session=fork_id,
            fork_path=str(fork_path) if fork_path is not None else None,
            resume_command=adapter.resume_command(fork_id, session.cwd) if fork_id else None,
            before_tokens=before,
            after_tokens=after if after is not None else before,
            tail_tokens=tail,
            keep=keep,
            first_prompt=_clip(prompts[0]) if prompts else "",
            last_prompt=_clip(prompts[-1]) if prompts else "",
            trigger=p.trigger.how,
            verified=verified,
            warnings=warnings,
        )

    if not region.turn_record:
        return result(
            "nothing-to-do",
            reason=f"the session ({before} tokens) is no larger than the protected tail; "
            "nothing was written",
        )
    if opts.target_tokens is not None and before <= opts.target_tokens:
        return result(
            "nothing-to-do",
            reason=f"the session is already under target ({before} <= {opts.target_tokens} "
            "tokens); nothing was written",
        )
    target: int | None = None
    if opts.target_tokens is not None:
        fixed = before - compressible
        target = opts.target_tokens - fixed
        if target < 1:
            raise HandoffError(
                f"target_tokens {opts.target_tokens} leaves no room: the protected tail and the "
                f"records always kept already take {fixed} tokens; nothing was written"
            )
    input_tokens = sum(count_tokens(s.text) for s in segment(region.transcript))
    rate = resolve_rate(opts.keep, opts.level, target, input_tokens)
    warnings += rate.notes
    compressed = compress_transcript(
        region.transcript, PipelineOptions(keep=rate.keep, budget=rate.budget), cache_dir=cache_dir
    )
    try:
        plan = plan_fork(
            session,
            region,
            compressed.segments,
            date=stamp.date().isoformat(),
            keep_thinking=opts.keep_thinking,
        )
    except HandoffRefusedError as exc:
        raise HandoffError(f"{exc}; nothing was written") from exc
    warnings += plan.warnings
    fork_id = adapter.new_session_id()
    dest = adapter.fork_destination(p.source, fork_id, homes[adapter.TOOL], stamp)
    placed = False
    try:
        staged = adapter.write_fork(session, plan, fork_id, work / "fork")
        failures = adapter.validate(staged, fork_id)
        if failures:
            raise HandoffError(
                f"the fork failed validation ({', '.join(failures)}) and was deleted; the "
                "original session is unchanged"
            )
        after = _total(adapter.read(staged))
        _place(staged, dest)
        placed = True
        verified: bool | None = None
        if opts.verify:
            try:
                verified = _verify(p, plan, homes[adapter.TOOL], stamp, opts.cwd, work, runner)
            except HandoffError as exc:
                raise HandoffError(
                    f"{exc}; the fork was deleted and the original is unchanged"
                ) from exc
            if not verified:
                raise HandoffError(
                    "live check failed: the resumed throwaway copy did not answer OK; the fork "
                    "was deleted and the original session is unchanged"
                )
    except BaseException:
        # Only a fork we placed ourselves is removed; a pre-existing file at dest is never ours.
        if placed:
            dest.unlink(missing_ok=True)
        raise
    return result(
        "forked", fork_id=fork_id, fork_path=dest, after=after, keep=rate.keep, verified=verified
    )

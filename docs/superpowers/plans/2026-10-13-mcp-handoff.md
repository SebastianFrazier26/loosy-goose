# MCP Session Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Commits and pushes.** The user authorized commits and pushes of `feat/mcp-handoff` on 2026-10-13
> ("commitandpush whenever"). Run each task's commit step without asking, and push the branch at
> natural checkpoints after scanning the diff for secrets. Never push to `main`, never force-push, and
> never open or merge a PR; those stay with the user. Commit messages end with
> `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**Goal:** `/loosy [light|medium|heavy]` in Claude Code (and `$loosy` in Codex) calls an MCP tool that
forks the live session into a new session file. The fork's history is the compressed transcript, made
of real kept turns plus a verbatim recent tail, and it opens with a marker. The tool then prints the
command that resumes the fork. The original session is never written.

**Architecture:** Each CLI has a session adapter (`sessions/claude.py`, `sessions/codex.py`), and it is
the only code that knows that CLI's file format. An adapter reads the session into a tool-agnostic
`Session` of `SessionRecord`s, writes a fork from a `ForkPlan`, and validates the fork the way the CLI
loads it. Pure planning functions (`sessions/plan.py`) work out which records to keep:

- find the `/loosy` turn;
- split off the protected tail;
- build the compressible `Transcript`;
- map the compressor's kept `Segment`s back to records through the new `Segment.block` and the
  `part` index in `Block.meta`;
- keep tool calls and their results together as atomic units.

`handoff.py` orchestrates: locate, snapshot, read, plan, compress (unchanged `compress_transcript`),
write to staging, validate, move into place, optionally verify. `mcp_server.py` is a thin layer over
`handoff` and `compress` on the official `mcp` 2.x SDK.

**Tech Stack:** Python 3.12, uv, ruff, mypy (strict on `src/`), pytest, the official `mcp` SDK 2.x
(`mcp.server.mcpserver.MCPServer`), anyio (an `mcp` dependency) in the MCP tests, and the existing
`loosy_goose` pipeline (tiktoken, sentence-transformers bge-small).

**Spec:** docs/superpowers/specs/2026-10-13-mcp-handoff-design.md

## Global Constraints

- Python 3.12 pinned (`.python-version`); uv; `ruff` line-length 100; `mypy` strict on `src/`; one-command suite `uv run pytest`.
- Fork only: new session id, original never written, locked or ended; Codex's `~/.codex/state_5.sqlite` is never touched.
- The only files ever deleted are forks of ours: one that failed validation, the `verify=true` throwaway copy, and a fork whose live check failed.
- Snapshot the session file before reading; a partial last line (no trailing newline) is ignored.
- Rate presets unchanged: `LEVELS = {"light": 0.50, "medium": 0.33, "heavy": 0.25}`, default `medium`, keep bounds `[0.10, 0.80]`; CLI behaviour unchanged.
- Protected tail: last user turn before the `/loosy` turn onward, extended back to at least `protect_tail_tokens=4000` (configurable); the budget applies to the compressible part only.
- The `/loosy` turn and everything after it are excluded from the fork.
- Marker, a user-role message at the start of the fork, exactly: `[loosy-goose: compressed from session <id> on <date>; kept <k> of <n> tokens before the last <t> tokens, which are verbatim. Earlier turns are an extractive selection — some were dropped.]`
- Atomic units kept or dropped whole: a tool call with its result; tool input never trimmed; tool result text may be trimmed.
- Thinking/reasoning in turns older than the protected tail is dropped (fallback switch `keep_thinking`).
- Unknown record types pass through unchanged with a warning; the handoff refuses if one lies inside the compressed stretch.
- `TESTED_VERSIONS`: Claude Code `2.1.296`, codex-cli `0.158.0`; a newer CLI still forks, with a warning.
- Every result carries the fork's resume command, the original's resume command, before/after tokens, tail size and warnings.
- MCP SDK: new optional extra `mcp = ["mcp>=2.3,<3"]`; console script `loosy-goose-mcp = "loosy_goose.mcp_server:main"`; `uv lock` without unrelated upgrades.
- The server loads bge-small at startup; without the `embed` extra it exits at startup with the install command.
- Install command: `uv tool install --torch-backend cpu "loosy-goose[embed,mcp] @ <wheel URL | git+https://github.com/SebastianFrazier26/loosy-goose@vX>"`; from the clone: `uv tool install --editable "$HOME/Developer/loosy-goose[embed,mcp]"`.
- Release on `v*` tags: test, `uv build`, `gh release create` with `GITHUB_TOKEN` and `contents: write`, attaching `dist/*` and the `integrations/` files; install the built wheel with `--torch-backend cpu` on ubuntu and assert `torch.__version__` ends with `+cpu`. Actions pinned to exact versions as in `ci.yml`.
- Fixtures are synthetic only (code words ALPHA–DELTA plus one `echo` call). No real session content enters the repo, a doc, a commit message or a report. Nothing under `data/local/` is committed.
- Comments only where load-bearing (why, not what). `CHANGELOG.md` gets dated lines under `## Unreleased` for user-visible changes.

## Review Focus

These are the input classes the spec implies but no spec-listed test exercises, most likely to bite first. Each has a pinning test in the owning task, marked `[RF1]`…`[RF5]`.

1. **[RF1] A Claude session already compacted by built-in `/compact`** (`system`/`compact_boundary` root with `parentUuid: null`, then an `isCompactSummary` user record). Only the chain after the last boundary is read. The boundary stays the fork's root, the marker follows it, the summary is kept verbatim, and nothing before the boundary is written. The fork validates. (Task 7)
2. **[RF2] A partial last line while the CLI is mid-append** (the handoff turn's own records are being written). The trailing bytes are ignored and a warning names the byte count. The fork is made. The original file's bytes and mtime are unchanged. (Task 10)
3. **[RF3] Two live sessions in the same cwd.** Among the three newest sessions for the cwd, the newest one carrying a handoff trigger wins over a newer one without one. A warning names how many sessions changed in the last 10 minutes, and the result echoes the picked session's first and last prompt. (Task 10)
4. **[RF4] A Codex session spanning several context windows** (`compacted` records with `replacement_history`). Lines before the last `compacted` record, other than `session_meta`, are not history. The `compacted` record is kept verbatim. The marker goes after it, before the first kept `response_item`. The fork validates. (Task 9)
5. **[RF5] The `/loosy` turn not identifiable.** Possible causes: write lag, an invocation in plain words, or an explicit `session=<id>` of a finished session. With no trigger text, no open Codex turn and no `handoff_session` call, nothing is excluded, the result says `trigger: "none"` with a warning, and the fork is still made. (Task 10)

Also covered by ordinary tests, though not in the top five:

- a huge single tool_result inside the tail (Task 3);
- a very short session (Task 10, "nothing to do");
- CRLF and non-UTF-8 bytes (Task 3).

---

## Facts this plan builds on (verified 2026-10-12/13; re-check if a CLI was upgraded)

- **Claude Code 2.1.296**
  - **Record types**, verified by listing only `type`/`subtype`/key names in this machine's sessions:
    - `user`, `assistant`, `attachment` and `system` records carry `uuid`, `parentUuid`, `sessionId` and also `session_id`.
    - `system` subtypes seen: `turn_duration`, `stop_hook_summary`, `away_summary`, `informational`, `local_command`, `compact_boundary`.
    - uuid-less types: `queue-operation`, `last-prompt` (`leafUuid`), `mode`, `permission-mode`, `atis-latch`, `ai-title`, `pr-link`, `agent-name`, `file-history-snapshot`, `file-history-delta`, `cost-state`, `continued-in`.
  - **Compaction:** a `compact_boundary` record has `parentUuid: null`, `logicalParentUuid` and `compactMetadata`. Its only child is a `user` record with `isCompactSummary: true`.
  - **Custom slash commands:** the invocation is a `user` record with string content containing `<command-message>…</command-message>` and `<command-name>/name</command-name>` (`<command-args>` only when arguments were given). The expanded command body is its child, a `user` record with `isMeta: true` and list content.
  - **Command file format** (from `~/.claude/commands/*.md` here): YAML frontmatter with `description`, `argument-hint` and `allowed-tools`. The body uses `$ARGUMENTS`.
  - **Project directory slug:** the probe cwd `/private/tmp/claude-501/-Users-…/scratchpad/probe` became `-private-tmp-claude-501--Users-…-scratchpad-probe`. That fits "every non-alphanumeric becomes `-`". The `_` case was not observed, so `locate` falls back to matching the recorded `cwd` field.
  - **Registration:** `claude mcp add [options] <name> <commandOrUrl> [args...]` with `-s, --scope <local|user|project>`. `-p` also takes `--resume`, `--mcp-config`, `--allowedTools` (variadic: put the prompt before it), `--session-id` and `--fork-session`.
- **codex-cli 0.158.0**
  - **File shape:** every line is `{timestamp, type, payload}`; newer lines add `ordinal`, which equals the 0-based line index.
  - **First line:** `session_meta`. Native forks carry a second `session_meta` (the parent's) on line 1.
  - **Top-level types:** `response_item`, `event_msg`, `turn_context`, `world_state`, `token_usage_record`, `inter_agent_communication_metadata`, `compacted`.
  - **`response_item` payload types:**
    - `message`, with role `user`, `assistant` or `developer` and content `input_text`/`output_text`;
    - `reasoning`: `content` null, `encrypted_content`, `summary` list;
    - `function_call` (`arguments` str, optional `namespace`) with `function_call_output`;
    - `custom_tool_call` (`input` str) with `custom_tool_call_output`; the output is a str or a list of `input_text`/`input_image`;
    - `web_search_call`, `agent_message`, `tool_search_call`, `tool_search_output`.
  - **Turns:** `event_msg` `task_started` → (developer/user context messages, `turn_context`, `world_state`) → the prompt's `response_item` user message → `event_msg` `user_message` → … → `task_complete` or `turn_aborted`. Many real turns have no closing event.
  - **Compaction:** a `compacted` record (`message`, `replacement_history`) can occur mid-turn and is followed by `turn_context`.
  - **Headless and registration:** `codex exec resume [SESSION_ID] [PROMPT]` takes `-c key=value`. `codex mcp add <NAME> -- <COMMAND>...`. Config keys `startup_timeout_sec` and `tool_timeout_sec` exist; found in the binary's strings.
  - **Skills:** Codex custom prompts (`~/.codex/prompts`) are **deprecated**; OpenAI's docs recommend skills. A user-level skill lives at `$HOME/.agents/skills/<name>/SKILL.md`, its frontmatter needs `name` and `description`, and it is invoked as `$name`. `allow_implicit_invocation: false` under `policy` in `agents/openai.yaml` keeps it explicit-only.
- **`mcp` 2.3.0** (checked in a scratch env)
  - `MCPServer(name, instructions=...)`, the `@server.tool()` decorator, and `server.run(transport="stdio")`.
  - A tool returning `dict[str, Any]` comes back as `structured_content` unwrapped.
  - Raising `mcp.server.mcpserver.exceptions.ToolError("msg")` yields `is_error=True` with text `Error executing tool <name>: msg`. Any other exception hides its message.
  - `from mcp import Client`; `async with Client(server) as c: await c.call_tool(...)` / `await c.list_tools()`.
- **`segments_digest`** covers kind and text only (CLAUDE.md), so a new non-text `Segment` field leaves every experiment checkpoint valid, as `tool_name` did.

## File map

| File | Task | Responsibility |
|---|---|---|
| `src/loosy_goose/rate.py` | 1 | `LEVELS`, keep bounds, `Rate`, `resolve_rate`, `check_keep` (moved from `cli.py`) |
| `src/loosy_goose/segment.py` | 2 | new `Segment.block` provenance field |
| `src/loosy_goose/sessions/__init__.py` | 3 | package marker (empty) |
| `src/loosy_goose/sessions/model.py` | 3 | `Session`, `SessionRecord`, snapshot, tolerant JSONL read/write, block tagging, version compare |
| `src/loosy_goose/sessions/plan.py` | 3, 4 | triggers, units, protected tail/region, `plan_fork`, marker |
| `tests/session_recorder.py` | 5 | records synthetic fixtures from the real CLIs, scrubbed; reused by live tests |
| `tests/fixtures/sessions/claude_codewords.jsonl`, `codex_codewords.jsonl` | 5 | committed synthetic fixtures |
| `tests/session_fixtures.py` | 5, 6, 8 | fixture paths, install helpers |
| `tests/session_builders.py` | 3 | in-memory `Session` builder for pure tests |
| `src/loosy_goose/sessions/claude.py` | 6, 7 | Claude Code adapter |
| `src/loosy_goose/sessions/codex.py` | 8, 9 | Codex adapter |
| `src/loosy_goose/sessions/adapter.py` | 10 | `SessionAdapter` protocol |
| `src/loosy_goose/handoff.py` | 10 | orchestration, `HandoffOptions`, `HandoffResult`, `HandoffError` |
| `src/loosy_goose/mcp_server.py` | 11 | MCP tools `handoff_session`, `compress`; `main` |
| `src/loosy_goose/embed.py` | 11 | `warm()` |
| `integrations/` | 12 | `/loosy` command, Codex `$loosy` skill, registration README |
| `tests/test_live_handoff.py` | 13 | opt-in drift and e2e tests |
| `.github/workflows/release.yml`, `ci.yml` | 11, 14 | `mcp` extra in CI; release on tags |

---

## Task 1: Move rate resolution into the library

**Files:**
- Create: `src/loosy_goose/rate.py`
- Modify: `src/loosy_goose/cli.py:1-75` (imports, `KEEP_MIN`…`DEFAULT_LEVEL`, `RateSource`, `Rate`, `_keep`, `resolve_rate`)
- Modify: `tests/test_cli.py:7` (import line)
- Test: `tests/test_rate.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (`loosy_goose.rate`):
  - `KEEP_MIN: float = 0.1`, `KEEP_MAX: float = 0.8`
  - `LEVELS: dict[str, float]`, `DEFAULT_LEVEL: str = "medium"`
  - `RateSource = Literal["keep", "level", "target-tokens"]`
  - `@dataclass(frozen=True) class Rate(keep: float, budget: int | None, source: RateSource, level: str | None, target_tokens: int | None, notes: tuple[str, ...])`
  - `def check_keep(k: float) -> float` (raises `ValueError` outside `[KEEP_MIN, KEEP_MAX]`)
  - `def resolve_rate(keep: float | None, level: str | None, target_tokens: int | None, input_tokens: int) -> Rate` (body moved verbatim)

- [ ] **Step 1: Write the failing test**

`tests/test_rate.py`:

```python
import pytest

from loosy_goose import cli, rate


def test_cli_uses_the_library_rate_table() -> None:
    assert cli.LEVELS is rate.LEVELS
    assert cli.resolve_rate is rate.resolve_rate
    assert rate.LEVELS == {"light": 0.50, "medium": 0.33, "heavy": 0.25}
    assert (rate.KEEP_MIN, rate.KEEP_MAX, rate.DEFAULT_LEVEL) == (0.1, 0.8, "medium")


@pytest.mark.parametrize("k", [0.1, 0.33, 0.8])
def test_check_keep_accepts_the_bounds(k: float) -> None:
    assert rate.check_keep(k) == k


@pytest.mark.parametrize("k", [0.09, 0.81])
def test_check_keep_rejects_outside_the_bounds(k: float) -> None:
    with pytest.raises(ValueError, match=r"keep must be in \[0.1, 0.8\]"):
        rate.check_keep(k)
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_rate.py -q`
Expected: collection error `ImportError: cannot import name 'rate' from 'loosy_goose'`.

- [ ] **Step 3: Create `src/loosy_goose/rate.py`**

Move the constants, `RateSource`, `Rate` and `resolve_rate` out of `cli.py` unchanged (docstrings included). Add `check_keep`:

```python
import math
from dataclasses import dataclass
from typing import Literal

KEEP_MIN = 0.1
KEEP_MAX = 0.8
LEVELS: dict[str, float] = {"light": 0.50, "medium": 0.33, "heavy": 0.25}
DEFAULT_LEVEL = "medium"

RateSource = Literal["keep", "level", "target-tokens"]


@dataclass(frozen=True)
class Rate:
    """The compression rate as the pipeline consumes it. `budget` is set only when a token target
    was honoured exactly; otherwise the pipeline derives the budget from `keep`."""

    keep: float
    budget: int | None
    source: RateSource
    level: str | None
    target_tokens: int | None
    notes: tuple[str, ...]


def check_keep(k: float) -> float:
    if not KEEP_MIN <= k <= KEEP_MAX:
        raise ValueError(f"keep must be in [{KEEP_MIN}, {KEEP_MAX}], got {k}")
    return k


def resolve_rate(
    keep: float | None, level: str | None, target_tokens: int | None, input_tokens: int
) -> Rate:
    """`input_tokens` must be the count `token_budget` sums (original segments), or a target's
    implied ratio would not match the budget the pipeline would derive from it."""
    if keep is not None:
        rounded = round(keep, 2)
        notes: tuple[str, ...] = ()
        if rounded != keep:
            notes = (f"--keep {keep} rounded to 0.01 steps; using {rounded:.2f}",)
        return Rate(rounded, None, "keep", None, None, notes)
    if target_tokens is not None:
        implied = target_tokens / input_tokens if input_tokens else math.inf
        if KEEP_MIN <= implied <= KEEP_MAX:
            return Rate(implied, target_tokens, "target-tokens", None, target_tokens, ())
        clamped = KEEP_MIN if implied < KEEP_MIN else KEEP_MAX
        warning = (
            f"warning: --target-tokens {target_tokens} implies keep {implied:.4f} of "
            f"{input_tokens} input tokens, outside [{KEEP_MIN}, {KEEP_MAX}]; clamped to "
            f"keep {clamped:.2f}"
        )
        return Rate(clamped, None, "target-tokens", None, target_tokens, (warning,))
    name = level if level is not None else DEFAULT_LEVEL
    return Rate(LEVELS[name], None, "level", name, None, ())
```

- [ ] **Step 4: Point `cli.py` at it**

In `src/loosy_goose/cli.py` (then `uv run ruff check --fix src/loosy_goose/cli.py` to re-sort the imports), delete lines 17–75 (from `KEEP_MIN = 0.1` through the end of `resolve_rate`), except `_keep`, `_target_tokens` and `_min_mentions`. Drop the now-unused `math`, `dataclass` and `Literal` imports. Add:

```python
from loosy_goose.rate import KEEP_MAX, KEEP_MIN, LEVELS, check_keep, resolve_rate
```

Then rewrite `_keep`:

```python
def _keep(value: str) -> float:
    try:
        return check_keep(float(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
```

`float("abc")` also raises `ValueError`, and argparse reported that as an invalid value before. It still does, now with Python's message. The other argparse helpers and every help string stay byte-identical.

In `tests/test_cli.py:7`, replace the import with:

```python
from loosy_goose.cli import build_parser, main
from loosy_goose.rate import LEVELS, resolve_rate
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_rate.py tests/test_cli.py -q && uv run mypy src && uv run ruff check . && uv run ruff format --check .`
Expected: all pass; mypy `Success`.

- [ ] **Step 6: Commit**

```bash
git add src/loosy_goose/rate.py src/loosy_goose/cli.py tests/test_rate.py tests/test_cli.py
git commit -m "$(cat <<'MSG'
Move rate resolution from cli.py into loosy_goose.rate

The MCP server needs the same presets and bounds; CLI behaviour is unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 2: `Segment.block` — map kept segments back to their source block

**Files:**
- Modify: `src/loosy_goose/segment.py:123-141` (the `Segment` dataclass) and `:363-388` (`segment()`)
- Test: `tests/test_segment.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Segment.block: int = 0`, the index of the source `Block` within its `Turn.blocks`, set by `segment()`. It stays out of `text` and `segments_digest`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_segment.py`)

```python
def test_segments_carry_their_source_block_index() -> None:
    from loosy_goose.transcript import Block, Transcript, Turn

    transcript = Transcript(
        "t",
        [
            Turn(
                0,
                "user",
                [
                    Block("text", "first paragraph here"),
                    Block("tool_use", '{"command": "ls"}', {"name": "Bash"}),
                    Block("text", "one sentence.\n\nanother sentence."),
                ],
            )
        ],
    )
    segs = segment(transcript)
    assert [(s.kind, s.block) for s in segs] == [
        ("prose", 0),
        ("tool_use", 1),
        ("prose", 2),
        ("prose", 2),
    ]
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_segment.py -q -k source_block`
Expected: FAIL with `AttributeError: 'Segment' object has no attribute 'block'`.

- [ ] **Step 3: Implement**

Append to the `Segment` dataclass, after `atom_source`:

```python
    # Index of the source block within its turn. The session handoff maps kept segments back to
    # the session records they came from through it. Like `tool_name`, kept out of `text` and
    # `segments_digest`, so no experiment checkpoint moves.
    block: int = 0
```

In `segment()`, replace `for block in turn.blocks:` with `for block_index, block in enumerate(turn.blocks):` and pass `block=block_index` in the `Segment(...)` call.

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_segment.py tests/test_pipeline.py tests/test_supersede.py -q && uv run mypy src`
Expected: all pass.

- [ ] **Step 5: Commit **

```bash
git add src/loosy_goose/segment.py tests/test_segment.py
git commit -m "$(cat <<'MSG'
Record each segment's source block index

The session handoff maps kept segments back to session records through it.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 3: Tool-agnostic session model, snapshot, trigger, units, protected tail

**Files:**
- Create: `src/loosy_goose/sessions/__init__.py` (empty file)
- Create: `src/loosy_goose/sessions/model.py`
- Create: `src/loosy_goose/sessions/plan.py` (triggers, units, region; `plan_fork` follows in Task 4)
- Create: `tests/session_builders.py`
- Test: `tests/test_session_model.py`

**Interfaces:**
- Consumes: `Segment.block` (Task 2); `transcript.Block`, `BlockKind`, `Role`, `Transcript`, `Turn`, `split_fences`; `tokens.count_tokens`.
- Produces (`loosy_goose.sessions.model`):
  - `Tool = Literal["claude", "codex"]`, `Category = Literal["message", "meta", "side", "unknown"]`, `CallSide = Literal["call", "result"]`, `PART = "part"`
  - `class SessionFormatError(ValueError)`
  - `@dataclass class SessionRecord(line: int, raw: dict[str, Any], kind: str, category: Category, role: Role | None = None, blocks: list[Block] = [], prompt: str | None = None, thinking: bool = False, pinned: bool = False, reply: str | None = None, call_ids: tuple[str, ...] = (), call_side: CallSide | None = None, tool_name: str | None = None, exchange: int = -1)`
  - `@dataclass class Session(tool: Tool, path: Path, session_id: str, cwd: str, cli_version: str | None, records: list[SessionRecord], live: list[int], loose: list[int] = [], open_exchange: int | None = None, warnings: list[str] = [])` with methods `tokens(index: int) -> int` and `prompts() -> list[str]`
  - `snapshot(src: Path, dest: Path) -> int`
  - `read_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]`
  - `dump_line(obj: dict[str, Any]) -> bytes`
  - `text_blocks(text: str, part: int) -> list[Block]`
  - `tagged(kind: BlockKind, text: str, part: int, **meta: str) -> list[Block]`
  - `same_dir(recorded: str, cwd: Path) -> bool`
  - `version_key(version: str) -> tuple[int, ...]`
  - `newer_than_tested(version: str | None, tested: tuple[str, ...]) -> bool`
- Produces (`loosy_goose.sessions.plan`):
  - `HANDOFF_SENTINEL = "loosy-goose:handoff"`, `HANDOFF_TOOL = "handoff_session"`, `PROTECT_TAIL_TOKENS = 4000`
  - `DEFAULT_TRIGGERS: dict[Tool, tuple[str, ...]]`
  - `TriggerHow = Literal["text", "open-turn", "tool-call", "none"]`
  - `@dataclass(frozen=True) class Trigger(position: int | None, how: TriggerHow)`
  - `@dataclass(frozen=True) class Region(end: int, tail_from: int, transcript: Transcript, turn_record: tuple[int, ...])`
  - `exchange_starts(session: Session) -> dict[int, int]`
  - `find_trigger(session: Session, triggers: Sequence[str]) -> Trigger`
  - `unit_ids(session: Session) -> dict[int, int]`
  - `split_region(session: Session, trigger: Trigger, protect_tail_tokens: int = PROTECT_TAIL_TOKENS) -> Region`

Meaning of the model, used by every later task:

- `records` holds every parseable line in file order.
- `live` holds the indices of the records that make up the history the CLI rebuilds, in history order: Claude's `parentUuid` chain root→leaf, Codex's file order after the last `compacted` record.
- `loose` holds records outside history order. These are Claude's uuid-less records; Codex has none.
- A `SessionRecord`'s `category` means:
  - `message`: carries blocks the compressor may see.
  - `meta`: structural; kept when its exchange keeps a message.
  - `side`: always dropped.
  - `unknown`: an unrecognised record type.
- `pinned` records are always kept verbatim: startup attachments, compaction summaries, `session_meta`, Codex developer and context messages.
- `exchange` is the user exchange a record belongs to. In Claude it counts the prompts seen so far. In Codex it counts `task_started` events. Records before the first one are in exchange 0.
- `blocks` carry `meta["part"]`, the index of the content part inside the record's raw content.

- [ ] **Step 1: Write the in-memory builder** (`tests/session_builders.py`)

```python
"""In-memory `Session` objects for the pure planning tests: one record per spec, in order."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loosy_goose.segment import Segment, segment
from loosy_goose.sessions.model import Session, SessionRecord, Tool, tagged, text_blocks
from loosy_goose.sessions.plan import Region


def words(n: int, word: str = "alpha") -> str:
    return " ".join([word] * n)


def build(
    *specs: tuple[str, str],
    tool: Tool = "claude",
    open_exchange: int | None = None,
    loose: tuple[str, ...] = (),
) -> Session:
    """Shapes: prompt, reply[:reply-id], think[:reply-id], call:<call-id>, handoff:<call-id>,
    result:<call-id>, summary, meta, side, unknown."""
    records: list[SessionRecord] = []
    exchange = 0
    for n, (shape, text) in enumerate(specs):
        head, _, arg = shape.partition(":")
        if head == "prompt":
            exchange += 1
        rec = _record(n, head, arg, text)
        rec.exchange = exchange
        records.append(rec)
    extra = [
        SessionRecord(len(records) + i, {"type": kind}, kind, "unknown")
        for i, kind in enumerate(loose)
    ]
    return Session(
        tool,
        Path("built.jsonl"),
        "source-id",
        "/work",
        None,
        records + extra,
        list(range(len(records))),
        [r.line for r in extra],
        open_exchange,
        [],
    )


def _record(n: int, head: str, arg: str, text: str) -> SessionRecord:
    raw: dict[str, Any] = {"n": n}
    if head == "prompt":
        return SessionRecord(
            n, raw, "user", "message", role="user", blocks=text_blocks(text, 0), prompt=text
        )
    if head == "reply":
        return SessionRecord(
            n, raw, "assistant", "message", role="assistant",
            blocks=text_blocks(text, 0), reply=arg or None,
        )
    if head == "think":
        return SessionRecord(
            n, raw, "assistant", "message", role="assistant",
            blocks=tagged("thinking", text, 0), thinking=True, reply=arg or None,
        )
    if head in ("call", "handoff"):
        name = "mcp__loosy-goose__handoff_session" if head == "handoff" else "Bash"
        return SessionRecord(
            n, raw, "assistant", "message", role="assistant",
            blocks=tagged("tool_use", text, 0, name=name),
            call_ids=(arg,), call_side="call", tool_name=name,
        )
    if head == "result":
        return SessionRecord(
            n, raw, "user:tool_result", "message", role="user",
            blocks=tagged("tool_result", text, 0), call_ids=(arg,), call_side="result",
        )
    if head == "summary":
        return SessionRecord(
            n, raw, "user:compact_summary", "message", role="user",
            blocks=text_blocks(text, 0), pinned=True,
        )
    if head == "meta":
        return SessionRecord(n, raw, "attachment", "meta")
    if head == "side":
        return SessionRecord(n, raw, "queue-operation", "side")
    if head == "unknown":
        return SessionRecord(n, raw, "mystery", "unknown")
    raise ValueError(f"unknown shape {head!r}")


def kept_where(region: Region, keep: Callable[[Segment], bool]) -> list[Segment]:
    """The segments a compressor run would return if it kept exactly those matching `keep`."""
    return [s for s in segment(region.transcript) if keep(s)]
```

(`ruff format` will reflow the multi-argument calls. That's fine.)

- [ ] **Step 2: Write the failing tests** (`tests/test_session_model.py`)

```python
from pathlib import Path

from session_builders import build, words

from loosy_goose.sessions.model import dump_line, newer_than_tested, read_jsonl, snapshot
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    Trigger,
    find_trigger,
    split_region,
    unit_ids,
)

TRIGGER = "<command-name>/loosy</command-name>"


def test_snapshot_ignores_a_partial_last_line(tmp_path: Path) -> None:
    src = tmp_path / "s.jsonl"
    src.write_bytes(b'{"a":1}\n{"b":2}\n{"c":')
    dest = tmp_path / "snap.jsonl"
    assert snapshot(src, dest) == 5
    assert dest.read_bytes() == b'{"a":1}\n{"b":2}\n'
    assert src.read_bytes() == b'{"a":1}\n{"b":2}\n{"c":'


def test_read_jsonl_tolerates_crlf_and_round_trips_undecodable_bytes(tmp_path: Path) -> None:
    src = tmp_path / "s.jsonl"
    src.write_bytes(b'{"t":"ok"}\r\n\n{"t":"caf\xe9"}\nnot json\n[1]\n')
    lines, warnings = read_jsonl(src)
    assert [n for n, _ in lines] == [0, 2]
    assert dump_line(lines[1][1]) == b'{"t":"caf\xe9"}\n'
    assert warnings == ["line 4: not valid JSON, skipped", "line 5: not a JSON object, skipped"]


def test_newer_than_tested_compares_numerically() -> None:
    assert newer_than_tested("2.1.300", ("2.1.296",))
    assert not newer_than_tested("2.1.296", ("2.1.296",))
    assert not newer_than_tested("2.1.99", ("2.1.296",))
    assert not newer_than_tested(None, ("2.1.296",))


def test_trigger_is_the_last_prompt_carrying_a_trigger_string() -> None:
    s = build(
        ("prompt", f"older {TRIGGER}"),
        ("reply", "a"),
        ("prompt", "work"),
        ("reply", "b"),
        ("prompt", TRIGGER),
        ("reply", "c"),
    )
    assert find_trigger(s, DEFAULT_TRIGGERS["claude"]) == Trigger(4, "text")


def test_a_later_handoff_call_beats_an_older_trigger_text() -> None:
    s = build(
        ("prompt", TRIGGER),
        ("reply", "a"),
        ("prompt", "hand this session off please"),
        ("handoff:h1", "{}"),
    )
    assert find_trigger(s, DEFAULT_TRIGGERS["claude"]) == Trigger(2, "tool-call")


def test_an_open_codex_turn_is_the_trigger() -> None:
    s = build(("prompt", "one"), ("reply", "a"), ("prompt", "two"), ("reply", "b"),
              tool="codex", open_exchange=2)
    assert find_trigger(s, DEFAULT_TRIGGERS["codex"]) == Trigger(2, "open-turn")


def test_no_trigger_excludes_nothing() -> None:
    s = build(("prompt", "one"), ("reply", "a"))
    assert find_trigger(s, DEFAULT_TRIGGERS["claude"]) == Trigger(None, "none")


def test_units_join_a_call_with_its_result_only() -> None:
    s = build(("prompt", "a"), ("call:c1", "{}"), ("call:c2", "{}"),
              ("result:c1", "x"), ("result:c2", "y"))
    u = unit_ids(s)
    assert u[1] == u[3] and u[2] == u[4]
    assert len({u[0], u[1], u[2]}) == 3


def test_tail_starts_at_the_last_prompt_before_the_trigger() -> None:
    s = build(("prompt", words(50)), ("reply", words(50)), ("prompt", words(10)),
              ("reply", words(10)), ("prompt", TRIGGER))
    region = split_region(s, Trigger(4, "text"), protect_tail_tokens=0)
    assert (region.end, region.tail_from, region.turn_record) == (4, 2, (0, 1))
    assert [t.index for t in region.transcript.turns] == [0, 1]


def test_tail_extends_back_to_protect_tail_tokens() -> None:
    s = build(("prompt", words(50)), ("reply", words(50)), ("prompt", words(10)),
              ("reply", words(10)), ("prompt", TRIGGER))
    region = split_region(s, Trigger(4, "text"), protect_tail_tokens=30)
    assert region.tail_from == 1
    assert region.turn_record == (0,)


def test_tail_never_splits_a_tool_pair_even_for_a_huge_result() -> None:
    s = build(
        ("prompt", words(20)),
        ("call:c1", '{"command": "cat big.log"}'),
        ("result:c1", words(20_000, "line")),
        ("prompt", words(5)),
        ("reply", words(5)),
    )
    region = split_region(s, Trigger(None, "none"), protect_tail_tokens=100)
    assert region.tail_from == 1
    assert region.turn_record == (0,)


def test_compressible_part_leaves_out_thinking_pinned_and_meta() -> None:
    s = build(("summary", "earlier work"), ("meta", ""), ("prompt", "a"),
              ("think:m1", "hmm"), ("reply:m1", "b"), ("prompt", "c"))
    region = split_region(s, Trigger(None, "none"), protect_tail_tokens=0)
    assert region.tail_from == 5
    assert region.turn_record == (2, 4)


def test_a_session_with_no_prompt_before_the_trigger_has_nothing_to_compress() -> None:
    s = build(("meta", ""), ("prompt", TRIGGER))
    region = split_region(s, Trigger(1, "text"), protect_tail_tokens=0)
    assert region.turn_record == ()
```

- [ ] **Step 3: Run them and see them fail**

Run: `uv run pytest tests/test_session_model.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'loosy_goose.sessions'`.

- [ ] **Step 4: Implement `src/loosy_goose/sessions/model.py`**

```python
"""The tool-agnostic view of an agent CLI's session file that the handoff plans over.

Only the adapters (`claude.py`, `codex.py`) know a file format; everything here and in `plan.py`
works on `Session` alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Block, BlockKind, Role, split_fences

Tool = Literal["claude", "codex"]
Category = Literal["message", "meta", "side", "unknown"]
CallSide = Literal["call", "result"]

PART = "part"


class SessionFormatError(ValueError):
    pass


@dataclass
class SessionRecord:
    line: int
    raw: dict[str, Any]
    kind: str
    category: Category
    role: Role | None = None
    blocks: list[Block] = field(default_factory=list)
    prompt: str | None = None
    thinking: bool = False
    pinned: bool = False
    reply: str | None = None
    call_ids: tuple[str, ...] = ()
    call_side: CallSide | None = None
    tool_name: str | None = None
    exchange: int = -1


@dataclass
class Session:
    tool: Tool
    path: Path
    session_id: str
    cwd: str
    cli_version: str | None
    records: list[SessionRecord]
    live: list[int]
    loose: list[int] = field(default_factory=list)
    open_exchange: int | None = None
    warnings: list[str] = field(default_factory=list)

    def tokens(self, index: int) -> int:
        return sum(count_tokens(b.text) for b in self.records[index].blocks)

    def prompts(self) -> list[str]:
        return [p for i in self.live if (p := self.records[i].prompt) is not None]


def snapshot(src: Path, dest: Path) -> int:
    """Copy `src` to `dest` up to its last newline; returns the number of bytes left out."""
    # The CLI is appending to this file while the handoff runs: a final line without its newline
    # is a record still being written, not a record.
    data = src.read_bytes()
    cut = data.rfind(b"\n") + 1
    dest.write_bytes(data[:cut])
    return len(data) - cut


def read_jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[str]]:
    out: list[tuple[int, dict[str, Any]]] = []
    warnings: list[str] = []
    for n, raw in enumerate(path.read_bytes().split(b"\n")):
        line = raw.rstrip(b"\r")
        if not line.strip():
            continue
        try:
            # surrogateescape so a byte that is not UTF-8 survives into the fork unchanged.
            obj = json.loads(line.decode("utf-8", errors="surrogateescape"))
        except json.JSONDecodeError:
            warnings.append(f"line {n + 1}: not valid JSON, skipped")
            continue
        if not isinstance(obj, dict):
            warnings.append(f"line {n + 1}: not a JSON object, skipped")
            continue
        out.append((n, obj))
    return out, warnings


def dump_line(obj: dict[str, Any]) -> bytes:
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return (text + "\n").encode("utf-8", errors="surrogateescape")


def text_blocks(text: str, part: int) -> list[Block]:
    blocks = split_fences(text)
    for b in blocks:
        b.meta[PART] = str(part)
    return blocks


def tagged(kind: BlockKind, text: str, part: int, **meta: str) -> list[Block]:
    return [Block(kind, text, {PART: str(part), **meta})] if text.strip() else []


def same_dir(recorded: str, cwd: Path) -> bool:
    try:
        return Path(recorded).resolve() == cwd.resolve()
    except OSError:
        return False


def version_key(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version)[:3])


def newer_than_tested(version: str | None, tested: tuple[str, ...]) -> bool:
    if version is None or not tested:
        return False
    return version_key(version) > max(version_key(t) for t in tested)
```

- [ ] **Step 5: Implement `src/loosy_goose/sessions/plan.py` (part 1)**

```python
"""Pure planning over a `Session`: what the handoff excludes, protects, compresses and keeps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from loosy_goose.sessions.model import Session, Tool
from loosy_goose.transcript import Transcript, Turn

HANDOFF_SENTINEL = "loosy-goose:handoff"
HANDOFF_TOOL = "handoff_session"
PROTECT_TAIL_TOKENS = 4000
DEFAULT_TRIGGERS: dict[Tool, tuple[str, ...]] = {
    # Claude Code records a custom slash command as a user record carrying this tag; the expanded
    # body (where the sentinel sits) is a separate isMeta record, which is not a prompt.
    "claude": ("<command-name>/loosy</command-name>", HANDOFF_SENTINEL),
    "codex": ("$loosy", HANDOFF_SENTINEL),
}

TriggerHow = Literal["text", "open-turn", "tool-call", "none"]


@dataclass(frozen=True)
class Trigger:
    """`position` indexes `Session.live`: the first record excluded from the fork."""

    position: int | None
    how: TriggerHow


@dataclass(frozen=True)
class Region:
    """`live[:end]` is history; `live[tail_from:end]` is the verbatim tail; `transcript` holds
    the compressible message records before it, `turn_record[i]` naming turn i's record."""

    end: int
    tail_from: int
    transcript: Transcript
    turn_record: tuple[int, ...]


def exchange_starts(session: Session) -> dict[int, int]:
    starts: dict[int, int] = {}
    for pos, ri in enumerate(session.live):
        starts.setdefault(session.records[ri].exchange, pos)
    return starts


def find_trigger(session: Session, triggers: Sequence[str]) -> Trigger:
    records, live = session.records, session.live
    starts = exchange_starts(session)
    found: list[tuple[int, TriggerHow]] = []
    text = next(
        (
            ri
            for ri in reversed(live)
            if (p := records[ri].prompt) is not None and any(t in p for t in triggers)
        ),
        None,
    )
    if text is not None:
        found.append((starts[records[text].exchange], "text"))
    if session.open_exchange is not None and session.open_exchange in starts:
        found.append((starts[session.open_exchange], "open-turn"))
    call = next(
        (
            ri
            for ri in reversed(live)
            if records[ri].call_side == "call"
            and (records[ri].tool_name or "").endswith(HANDOFF_TOOL)
        ),
        None,
    )
    if call is not None:
        found.append((starts[records[call].exchange], "tool-call"))
    if not found:
        return Trigger(None, "none")
    # The current invocation is always the latest of these; an older /loosy text must not win
    # over it, or everything since that older handoff would be cut. max() keeps the first of
    # equal positions, so text beats the weaker signals on a tie.
    position, how = max(found, key=lambda f: f[0])
    return Trigger(position, how)


def unit_ids(session: Session) -> dict[int, int]:
    parent: dict[int, int] = {}

    def root(x: int) -> int:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    owner: dict[str, int] = {}
    for ri in session.live:
        for cid in session.records[ri].call_ids:
            if cid in owner:
                a, b = root(owner[cid]), root(ri)
                if a != b:
                    parent[b] = a
            else:
                owner[cid] = ri
    return {ri: root(ri) for ri in session.live}


def split_region(
    session: Session, trigger: Trigger, protect_tail_tokens: int = PROTECT_TAIL_TOKENS
) -> Region:
    records, live = session.records, session.live
    end = trigger.position if trigger.position is not None else len(live)
    starts = exchange_starts(session)
    tail_from = 0
    for pos in range(end - 1, -1, -1):
        rec = records[live[pos]]
        if rec.prompt is not None:
            tail_from = starts[rec.exchange]
            break
    tail = sum(session.tokens(ri) for ri in live[tail_from:end])
    # Extended record by record, not exchange by exchange: one agentic exchange can run to six
    # figures of tokens, and taking it whole would leave nothing to compress.
    while tail < protect_tail_tokens and tail_from > 0:
        tail_from -= 1
        tail += session.tokens(live[tail_from])
    units = unit_ids(session)
    position = {ri: pos for pos, ri in enumerate(live)}
    first_of: dict[int, int] = {}
    for ri, unit in units.items():
        first_of[unit] = min(first_of.get(unit, position[ri]), position[ri])
    moved = True
    while moved:
        moved = False
        for pos in range(tail_from, end):
            first = first_of[units[live[pos]]]
            if first < tail_from:
                tail_from, moved = first, True
                break
    turns: list[Turn] = []
    turn_record: list[int] = []
    for ri in live[:tail_from]:
        rec = records[ri]
        if rec.category != "message" or rec.thinking or rec.pinned or not rec.blocks:
            continue
        turns.append(Turn(len(turns), rec.role or "user", rec.blocks))
        turn_record.append(ri)
    return Region(
        end, tail_from, Transcript(f"{session.path}#compressible", turns), tuple(turn_record)
    )
```

Create `src/loosy_goose/sessions/__init__.py` as an empty file.

- [ ] **Step 6: Run the tests and see them pass**

Run: `uv run pytest tests/test_session_model.py -q && uv run mypy src && uv run ruff check . && uv run ruff format --check .`
Expected: 13 passed; mypy `Success`. If `ruff format --check` flags `tests/session_builders.py`, run `uv run ruff format tests/session_builders.py tests/test_session_model.py` and re-check.

- [ ] **Step 7: Commit **

```bash
git add src/loosy_goose/sessions/__init__.py src/loosy_goose/sessions/model.py src/loosy_goose/sessions/plan.py tests/session_builders.py tests/test_session_model.py
git commit -m "$(cat <<'MSG'
Add the tool-agnostic session model and handoff region planning

Snapshot ignoring a partial last line, tolerant JSONL read/write, trigger detection
(text, open Codex turn, handoff_session call; latest wins), call/result units, and the
protected tail extended record by record without splitting a tool pair.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 4: `plan_fork` — map kept segments to records, atomic units, marker

**Files:**
- Modify: `src/loosy_goose/sessions/plan.py` (append below `split_region`; extend imports)
- Test: `tests/test_session_plan.py`

**Interfaces:**
- Consumes:
  - `Region`, `unit_ids`, `split_region`, `find_trigger`, `DEFAULT_TRIGGERS` (Task 3);
  - `Session`, `SessionRecord`, `PART` (Task 3);
  - `segment.segment`, `Segment.block` (Task 2).
- Produces (`loosy_goose.sessions.plan`):
  - `TRIMMED_RESULT = "[loosy-goose: output trimmed]"`, `GAP = "[loosy-goose: lines trimmed]"`
  - `MARKER_PREFIX = "[loosy-goose: compressed from session "`. The adapters classify an earlier fork's marker as `side`, so a second `/loosy` replaces the stale marker instead of keeping two.
  - `class HandoffRefusedError(Exception)`
  - `marker_text(source: str, date: str, kept: int, total: int, tail: int) -> str`
  - `@dataclass(frozen=True) class ForkPlan(keep: tuple[int, ...], rewrites: dict[int, dict[int, str]], passthrough: tuple[int, ...], marker: str, tail_from: int, end: int, compressible_tokens: int, kept_tokens: int, tail_tokens: int, warnings: tuple[str, ...])`
    - `keep` holds record indices in history (`live`) order.
    - `rewrites[record][part]` is the replacement text for that content part. `""` means drop the part.
    - `passthrough` holds loose unknown records the writer appends unchanged.
  - `plan_fork(session: Session, region: Region, kept: Sequence[Segment], *, date: str, keep_thinking: bool = False) -> ForkPlan`

**Mapping rule.** `plan_fork` re-runs `segment(region.transcript)` with the same defaults `compress_transcript` uses, so segment ids line up with the compressor's `original` list. The handoff always runs with `band=False, paths=False`, so a kept segment's text is the original text. For each segment:

- `region.turn_record[seg.turn]` gives the record.
- `region.transcript.turns[seg.turn].blocks[seg.block].meta["part"]` gives the content part.

A record with any kept segment is *selected*. Every record sharing a unit with a selected record is *chosen*, so a call and its result are always kept together. A chosen record's partly kept parts are rebuilt from their kept segments:

- tool_result: kept chunks joined by `\n`, with a `GAP` line where chunks were dropped, or `TRIMMED_RESULT` if none were kept;
- prose: kept paragraphs joined by a blank line, code segments re-fenced;
- tool_use: never rewritten.

- [ ] **Step 1: Write the failing tests** (`tests/test_session_plan.py`)

```python
from collections.abc import Callable

import pytest
from session_builders import build, kept_where

from loosy_goose.segment import Segment
from loosy_goose.sessions.model import Session
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    GAP,
    MARKER_PREFIX,
    TRIMMED_RESULT,
    ForkPlan,
    HandoffRefusedError,
    Region,
    find_trigger,
    marker_text,
    plan_fork,
    split_region,
)

TRIGGER = "<command-name>/loosy</command-name>"
LOG = [f"log line {i} " + "x" * 190 for i in range(6)]


def _session() -> Session:
    return build(
        ("prompt", "Remember ALPHA."),  # 0
        ("reply", "noted ALPHA"),  # 1
        ("prompt", "Run the log."),  # 2
        ("call:c1", '{"command": "cat run.log"}'),  # 3
        ("result:c1", "\n".join(LOG)),  # 4: three 600-char chunks
        ("reply", "first para BETA.\n\nsecond para GAMMA."),  # 5
        ("think:m9", "private musing"),  # 6
        ("reply:m9", "DELTA reply"),  # 7
        ("prompt", "tail prompt"),  # 8
        ("reply", "tail reply"),  # 9
        ("prompt", TRIGGER),  # 10
        ("reply", "running handoff"),  # 11
    )


def _plan(
    keep: Callable[[Segment], bool],
    *,
    session: Session | None = None,
    keep_thinking: bool = False,
) -> tuple[Session, Region, ForkPlan]:
    s = session if session is not None else _session()
    region = split_region(s, find_trigger(s, DEFAULT_TRIGGERS["claude"]), protect_tail_tokens=0)
    plan = plan_fork(
        s, region, kept_where(region, keep), date="2026-10-13", keep_thinking=keep_thinking
    )
    return s, region, plan


def test_marker_text_is_exact() -> None:
    assert marker_text("abc", "2026-10-13", 10, 40, 7) == (
        "[loosy-goose: compressed from session abc on 2026-10-13; kept 10 of 40 tokens before "
        "the last 7 tokens, which are verbatim. Earlier turns are an extractive selection — "
        "some were dropped.]"
    )
    assert marker_text("x", "d", 1, 2, 3).startswith(MARKER_PREFIX)


def test_everything_kept_rewrites_nothing_and_excludes_the_trigger_turn() -> None:
    _, _, plan = _plan(lambda s: True)
    assert plan.keep == (0, 1, 2, 3, 4, 5, 7, 8, 9)
    assert plan.rewrites == {}
    assert (plan.tail_from, plan.end) == (8, 10)


def test_a_kept_result_chunk_keeps_its_call_whole() -> None:
    _, _, plan = _plan(lambda s: s.kind == "tool_result" and "log line 0" in s.text)
    assert plan.keep == (3, 4, 8, 9)
    assert plan.rewrites == {4: {0: f"{LOG[0]}\n{LOG[1]}\n{GAP}"}}


def test_a_kept_call_alone_keeps_its_result_trimmed() -> None:
    _, _, plan = _plan(lambda s: s.kind == "tool_use")
    assert plan.keep == (3, 4, 8, 9)
    assert plan.rewrites == {4: {0: TRIMMED_RESULT}}


def test_dropping_a_middle_chunk_marks_the_gap() -> None:
    _, _, plan = _plan(
        lambda s: s.kind == "tool_result" and ("log line 0" in s.text or "log line 4" in s.text)
    )
    assert plan.rewrites == {4: {0: f"{LOG[0]}\n{LOG[1]}\n{GAP}\n{LOG[4]}\n{LOG[5]}"}}


def test_a_partly_kept_reply_keeps_only_its_kept_paragraphs() -> None:
    _, _, plan = _plan(lambda s: s.text == "second para GAMMA.")
    assert plan.keep == (5, 8, 9)
    assert plan.rewrites == {5: {0: "second para GAMMA."}}


def test_older_thinking_is_dropped_unless_asked_and_its_reply_survives() -> None:
    assert 6 not in _plan(lambda s: True)[2].keep
    assert 6 in _plan(lambda s: True, keep_thinking=True)[2].keep
    assert 6 not in _plan(lambda s: "DELTA" not in s.text, keep_thinking=True)[2].keep


def test_meta_records_follow_their_exchange() -> None:
    s = build(("prompt", "a"), ("meta", ""), ("reply", "b"), ("prompt", "c"), ("meta", ""),
              ("reply", "d"), ("prompt", "e"))
    _, _, plan = _plan(lambda seg: seg.text == "b", session=s)
    assert plan.keep == (1, 2, 6)


def test_pinned_records_are_kept_even_when_nothing_is_selected() -> None:
    s = build(("summary", "old work"), ("prompt", "a"), ("reply", "b"), ("prompt", "c"))
    _, _, plan = _plan(lambda seg: False, session=s)
    assert plan.keep == (0, 3)


def test_an_unknown_record_inside_the_compressed_stretch_refuses() -> None:
    s = build(("prompt", "a"), ("unknown", ""), ("reply", "b"), ("prompt", "c"))
    with pytest.raises(HandoffRefusedError, match="'mystery' record at line 2"):
        _plan(lambda seg: True, session=s)


def test_an_unknown_record_in_the_tail_passes_through_with_a_warning() -> None:
    s = build(("prompt", "a"), ("reply", "b"), ("prompt", "c"), ("unknown", ""))
    _, _, plan = _plan(lambda seg: True, session=s)
    assert 3 in plan.keep
    assert any("'mystery' record at line 4 passed through unchanged" in w for w in plan.warnings)


def test_side_records_drop_and_loose_unknown_records_pass_through() -> None:
    s = build(("prompt", "a"), ("side", ""), ("reply", "b"), ("prompt", "c"),
              loose=("mystery-side",))
    _, _, plan = _plan(lambda seg: True, session=s)
    assert plan.keep == (0, 2, 3)
    assert plan.passthrough == (4,)
    assert any("mystery-side" in w for w in plan.warnings)


def test_token_accounting_and_marker() -> None:
    s, region, plan = _plan(lambda seg: True)
    assert plan.compressible_tokens == sum(s.tokens(ri) for ri in region.turn_record)
    assert plan.kept_tokens == plan.compressible_tokens
    assert plan.tail_tokens == s.tokens(8) + s.tokens(9)
    assert (
        f"kept {plan.kept_tokens} of {plan.compressible_tokens} tokens before the last "
        f"{plan.tail_tokens} tokens" in plan.marker
    )
    assert "from session source-id on 2026-10-13" in plan.marker
    partial = _plan(lambda seg: seg.text == "second para GAMMA.")[2]
    assert 0 < partial.kept_tokens < partial.compressible_tokens
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_session_plan.py -q`
Expected: collection error `ImportError: cannot import name 'GAP' from 'loosy_goose.sessions.plan'`.

- [ ] **Step 3: Implement** (append to `src/loosy_goose/sessions/plan.py`)

Extend the imports at the top of `plan.py` to:

```python
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from loosy_goose.segment import Segment, segment
from loosy_goose.sessions.model import PART, Session, SessionRecord, Tool
from loosy_goose.tokens import count_tokens
from loosy_goose.transcript import Block, Transcript, Turn
```

Append:

```python
TRIMMED_RESULT = "[loosy-goose: output trimmed]"
GAP = "[loosy-goose: lines trimmed]"
_FENCE = "`" * 3
MARKER_PREFIX = "[loosy-goose: compressed from session "
_MARKER = MARKER_PREFIX + (
    "{source} on {date}; kept {kept} of {total} tokens before the last {tail} tokens, which "
    "are verbatim. Earlier turns are an extractive selection — some were dropped.]"
)


class HandoffRefusedError(Exception):
    pass


@dataclass(frozen=True)
class ForkPlan:
    """`keep` is in history order. `rewrites[record][part]` replaces that content part's text;
    an empty string drops the part. `passthrough` are loose unknown records written unchanged."""

    keep: tuple[int, ...]
    rewrites: dict[int, dict[int, str]]
    passthrough: tuple[int, ...]
    marker: str
    tail_from: int
    end: int
    compressible_tokens: int
    kept_tokens: int
    tail_tokens: int
    warnings: tuple[str, ...]


def marker_text(source: str, date: str, kept: int, total: int, tail: int) -> str:
    return _MARKER.format(source=source, date=date, kept=kept, total=total, tail=tail)


Piece = tuple[Segment, bool, Block]


def _rebuild(pieces: list[Piece]) -> str:
    if pieces[0][2].kind == "tool_result":
        lines: list[str] = []
        dropped = False
        for seg, kept, _ in pieces:
            if not kept:
                dropped = True
                continue
            if dropped:
                lines.append(GAP)
            lines.append(seg.text)
            dropped = False
        if dropped and lines:
            lines.append(GAP)
        return "\n".join(lines) if lines else TRIMMED_RESULT
    paras: list[str] = []
    for seg, kept, block in pieces:
        if not kept:
            continue
        if seg.kind == "code":
            paras.append(f"{_FENCE}{block.meta.get('lang', '')}\n{seg.text}\n{_FENCE}")
        else:
            paras.append(seg.text)
    return "\n\n".join(paras)


def _tokens_after(rec: SessionRecord, rewrite: dict[int, str]) -> int:
    untouched = sum(count_tokens(b.text) for b in rec.blocks if int(b.meta[PART]) not in rewrite)
    return untouched + sum(count_tokens(text) for text in rewrite.values())


def plan_fork(
    session: Session,
    region: Region,
    kept: Sequence[Segment],
    *,
    date: str,
    keep_thinking: bool = False,
) -> ForkPlan:
    records = session.records
    kept_ids = {s.id for s in kept}
    # Same segmenter, same defaults as compress_transcript's `original`, so ids line up.
    pieces: dict[tuple[int, int], list[Piece]] = {}
    for seg in segment(region.transcript):
        block = region.transcript.turns[seg.turn].blocks[seg.block]
        key = (region.turn_record[seg.turn], int(block.meta[PART]))
        pieces.setdefault(key, []).append((seg, seg.id in kept_ids, block))
    units = unit_ids(session)
    selected = {ri for (ri, _), group in pieces.items() if any(k for _, k, _ in group)}
    kept_units = {units[ri] for ri in selected}
    chosen = {ri for ri in region.turn_record if units[ri] in kept_units}
    rewrites: dict[int, dict[int, str]] = {}
    for (ri, part), group in pieces.items():
        if ri not in chosen or all(k for _, k, _ in group) or group[0][2].kind == "tool_use":
            continue
        rewrites.setdefault(ri, {})[part] = _rebuild(group)

    exchanges = {records[ri].exchange for ri in chosen}
    replies = {r for ri in chosen if (r := records[ri].reply) is not None}
    keep: list[int] = []
    warnings: list[str] = []
    for pos, ri in enumerate(session.live[: region.end]):
        rec = records[ri]
        if rec.category == "side":
            continue
        if rec.category == "unknown":
            where = f"{rec.kind!r} record at line {rec.line + 1}"
            if pos < region.tail_from:
                raise HandoffRefusedError(
                    f"unknown {where} lies inside the stretch to be compressed; refusing rather "
                    f"than guess how {session.tool} reads it"
                )
            warnings.append(f"unknown {where} passed through unchanged")
            keep.append(ri)
        elif pos >= region.tail_from or rec.pinned:
            keep.append(ri)
        elif rec.thinking:
            if keep_thinking and rec.reply in replies:
                keep.append(ri)
        elif rec.category == "meta":
            if rec.exchange in exchanges:
                keep.append(ri)
        elif ri in chosen:
            keep.append(ri)
    passthrough = tuple(ri for ri in session.loose if records[ri].category == "unknown")
    warnings += [
        f"unknown {records[ri].kind!r} record at line {records[ri].line + 1} passed through "
        "unchanged"
        for ri in passthrough
    ]
    compressible = sum(session.tokens(ri) for ri in region.turn_record)
    kept_tokens = sum(
        _tokens_after(records[ri], rewrites.get(ri, {})) for ri in keep if ri in chosen
    )
    tail = sum(session.tokens(ri) for ri in session.live[region.tail_from : region.end])
    return ForkPlan(
        keep=tuple(keep),
        rewrites=rewrites,
        passthrough=passthrough,
        marker=marker_text(session.session_id, date, kept_tokens, compressible, tail),
        tail_from=region.tail_from,
        end=region.end,
        compressible_tokens=compressible,
        kept_tokens=kept_tokens,
        tail_tokens=tail,
        warnings=tuple(warnings),
    )
```

(`rec.reply in replies` is `False` for a `None` reply, because `replies` never holds `None`.)

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_session_plan.py tests/test_session_model.py -q && uv run mypy src && uv run ruff check . && uv run ruff format --check .`
Expected: 26 passed; mypy `Success`. If `ruff format --check` flags a file, run `uv run ruff format <file>` and re-check.

- [ ] **Step 5: Commit **

```bash
git add src/loosy_goose/sessions/plan.py tests/test_session_plan.py
git commit -m "$(cat <<'MSG'
Plan the fork: kept segments back to records, whole tool pairs, marker

Partly kept prose and tool results are rebuilt from their kept segments; tool input is
never trimmed; older thinking is dropped unless keep_thinking; unknown records refuse
inside the compressed stretch and pass through elsewhere.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 5: Fixture recorder and committed synthetic fixtures

**Files:**
- Create: `tests/session_recorder.py`
- Create: `tests/session_fixtures.py`
- Create: `tests/fixtures/sessions/claude_codewords.jsonl`, `tests/fixtures/sessions/codex_codewords.jsonl` (recorded, not hand-written)
- Test: `tests/test_session_fixtures.py`

**Interfaces:**
- Consumes: nothing from `src/`. The recorder must run before the adapters exist and stay usable on its own.
- Produces:
  - `session_recorder`:
    - constants `FIXTURE_CWD = "/work/fixture"`, `FIXTURE_HOME = "/home/user"`, `SENTINEL = "loosy-goose:handoff"`, `CODE_WORDS = ("ALPHA", "BETA", "GAMMA", "DELTA")`, `PROMPTS: tuple[str, ...]`;
    - `class RecorderError(RuntimeError)`;
    - `scratch_repo() -> Path`;
    - `record_claude(cwd: Path, claude_home: Path, *, trigger: bool = True) -> Path`;
    - `record_codex(cwd: Path, codex_home: Path, *, trigger: bool = True) -> Path`;
    - `newest_codex_session(codex_home: Path, cwd: Path) -> Path`;
    - `main(argv: list[str] | None = None) -> int`.
  - `session_fixtures`:
    - `FIXTURES: Path`, `CLAUDE_FIXTURE = "claude_codewords.jsonl"`, `CODEX_FIXTURE = "codex_codewords.jsonl"`;
    - `fixture_lines(name: str) -> list[dict[str, Any]]`;
    - `write_lines(path: Path, lines: list[dict[str, Any]]) -> Path`;
    - re-exports `FIXTURE_CWD`, `FIXTURE_HOME`, `SENTINEL`, `CODE_WORDS`.

**Fixture shape (both CLIs, the probe shape):** five turns in a throwaway git repo.

1. "Think hard, then remember the code word ALPHA." The thinking keyword gives the older-thinking test something to drop.
2. BETA.
3. Run `echo GAMMA`.
4. "Remember the code word DELTA."
5. The trigger turn. Claude: `/loosy medium`, with a project command whose body carries the sentinel. Codex: a prompt starting with the sentinel.

The recorder scrubs before writing:

- home and scratch paths become placeholders;
- injected instructions (attachments, `base_instructions`, `user_instructions`, developer and context messages) are blanked;
- account ids and git identity are blanked;
- the result is checked against the home dir, user name, git email and git name, and the recorder fails rather than write a fixture that still contains one.

- [ ] **Step 1: Write the failing test** (`tests/test_session_fixtures.py`)

```python
import json
import re

import pytest
from session_fixtures import (
    CLAUDE_FIXTURE,
    CODE_WORDS,
    CODEX_FIXTURE,
    FIXTURE_CWD,
    FIXTURE_HOME,
    FIXTURES,
    SENTINEL,
)


@pytest.mark.parametrize("name", [CLAUDE_FIXTURE, CODEX_FIXTURE])
def test_fixture_is_synthetic_and_scrubbed(name: str) -> None:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    for word in CODE_WORDS:
        assert word in text
    assert SENTINEL in text
    assert FIXTURE_CWD in text
    for leak in ("/Users/", "\\\\Users\\\\", "/private/", "/var/folders/"):
        assert leak not in text
    assert set(re.findall(r'/home/[^/"\\]+', text)) <= {FIXTURE_HOME}
    for line in text.splitlines():
        assert isinstance(json.loads(line), dict)


def test_claude_fixture_recorded_the_slash_command() -> None:
    text = (FIXTURES / CLAUDE_FIXTURE).read_text(encoding="utf-8")
    assert "<command-name>/loosy</command-name>" in text
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_session_fixtures.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'session_fixtures'`.

- [ ] **Step 3: Write the recorder** (`tests/session_recorder.py`)

```python
"""Record the synthetic session fixtures for the handoff tests from the real CLIs.

    uv run python tests/session_recorder.py [--out tests/fixtures/sessions] [--only claude|codex]

Runs Claude Code and Codex headless in a throwaway git repo for five short turns each, which
spends a little API use. Copies the recorded session files out scrubbed. The recorded originals
stay where the CLIs wrote them (Codex has indexed its copy in its own sqlite, which loosy-goose
never touches).
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

FIXTURE_CWD = "/work/fixture"
FIXTURE_HOME = "/home/user"
SCRUBBED = "<scrubbed>"
SENTINEL = "loosy-goose:handoff"
CODE_WORDS = ("ALPHA", "BETA", "GAMMA", "DELTA")
PROMPTS = (
    "Think hard, then remember the code word ALPHA. Reply with only: noted.",
    "Remember the code word BETA. Reply with only: noted.",
    "Run the shell command `echo GAMMA` and reply with only its output.",
    "Remember the code word DELTA. Reply with only: noted.",
)
CLAUDE_TRIGGER = "/loosy medium"
CODEX_TRIGGER = f"{SENTINEL} Reply with only: handoff requested."
_STAND_IN_COMMAND = (
    "---\ndescription: fixture stand-in for the loosy command\n---\n\n"
    f"{SENTINEL}\n\nReply with only: handoff requested.\n"
)
_FIXTURE_TEXT = (*CODE_WORDS, "noted", SENTINEL, "handoff requested", "/loosy")


class RecorderError(RuntimeError):
    pass


def _run(argv: list[str], cwd: Path) -> str:
    done = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, timeout=600, check=False
    )
    if done.returncode != 0:
        raise RecorderError(f"{argv[0]} exited {done.returncode}: {done.stderr.strip()[:500]}")
    return done.stdout


def scratch_repo() -> Path:
    cwd = Path(tempfile.mkdtemp(prefix="loosy-fixture-")).resolve()
    subprocess.run(["git", "init", "-q"], cwd=cwd, check=True)
    commands = cwd / ".claude" / "commands"
    commands.mkdir(parents=True)
    (commands / "loosy.md").write_text(_STAND_IN_COMMAND, encoding="utf-8")
    return cwd


def record_claude(cwd: Path, claude_home: Path, *, trigger: bool = True) -> Path:
    sid = str(uuid.uuid4())
    _run(["claude", "-p", PROMPTS[0], "--session-id", sid], cwd)
    _run(["claude", "-p", PROMPTS[1], "--resume", sid], cwd)
    # The prompt goes before --allowedTools: that flag is variadic and swallows what follows.
    _run(["claude", "-p", PROMPTS[2], "--resume", sid, "--allowedTools", "Bash(echo:*)"], cwd)
    _run(["claude", "-p", PROMPTS[3], "--resume", sid], cwd)
    if trigger:
        _run(["claude", "-p", CLAUDE_TRIGGER, "--resume", sid], cwd)
    found = sorted((claude_home / "projects").glob(f"*/{sid}.jsonl"))
    if not found:
        raise RecorderError(f"Claude Code wrote no session file for {sid}")
    if trigger and "<command-name>/loosy</command-name>" not in found[0].read_text("utf-8"):
        raise RecorderError(
            "the /loosy turn was not recorded as a slash command; check that `claude -p` "
            "expands project commands in this version"
        )
    return found[0]


def _first_line(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        head = json.loads(fh.readline())
    if not isinstance(head, dict):
        raise RecorderError(f"{path}: first line is not an object")
    return head


def newest_codex_session(codex_home: Path, cwd: Path) -> Path:
    files = sorted(
        (codex_home / "sessions").glob("*/*/*/rollout-*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in files:
        recorded = _first_line(path).get("payload", {}).get("cwd")
        if isinstance(recorded, str) and Path(recorded).resolve() == cwd.resolve():
            return path
    raise RecorderError(f"Codex wrote no session for {cwd}")


def record_codex(cwd: Path, codex_home: Path, *, trigger: bool = True) -> Path:
    _run(["codex", "exec", PROMPTS[0]], cwd)
    sid = str(_first_line(newest_codex_session(codex_home, cwd))["payload"]["id"])
    for prompt in (*PROMPTS[1:], *((CODEX_TRIGGER,) if trigger else ())):
        _run(["codex", "exec", "resume", sid, prompt], cwd)
    found = sorted(
        (codex_home / "sessions").glob(f"*/*/*/rollout-*-{sid}.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    if not found:
        raise RecorderError(f"Codex session file for {sid} not found")
    return found[-1]


def scrub(obj: Any, replace: Callable[[str], str]) -> Any:
    if isinstance(obj, str):
        return replace(obj)
    if isinstance(obj, list):
        return [scrub(v, replace) for v in obj]
    if isinstance(obj, dict):
        return {k: scrub(v, replace) for k, v in obj.items()}
    return obj


def blank(obj: Any) -> Any:
    """Every string inside `obj` replaced, structure and `type`/`role` values kept."""
    if isinstance(obj, str):
        return SCRUBBED
    if isinstance(obj, list):
        return [blank(v) for v in obj]
    if isinstance(obj, dict):
        return {k: v if k in ("type", "role") else blank(v) for k, v in obj.items()}
    return obj


def _mentions_fixture(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False)
    return any(marker in text for marker in _FIXTURE_TEXT)


def scrub_claude(rec: dict[str, Any]) -> dict[str, Any]:
    out = dict(rec)
    if out.get("type") == "attachment":
        out["attachment"] = blank(out.get("attachment"))
    if out.get("type") == "system":
        for key in ("content", "hookInfos", "hookErrors", "hookAdditionalContext"):
            if key in out:
                out[key] = blank(out[key])
    if out.get("type") == "user" and out.get("isMeta") and not _mentions_fixture(out.get("message")):
        out["message"] = blank(out.get("message"))
    if "gitBranch" in out:
        out["gitBranch"] = "main"
    return out


def scrub_codex(rec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(json.dumps(rec))
    payload = out.get("payload")
    if not isinstance(payload, dict):
        return out
    kind = out.get("type")
    if kind == "session_meta":
        for key in ("base_instructions", "git", "creator_account_id", "creator_user_id"):
            if key in payload:
                payload[key] = blank(payload[key])
    elif kind == "turn_context":
        for key in ("user_instructions", "developer_instructions"):
            if key in payload:
                payload[key] = blank(payload[key])
    elif kind == "world_state":
        out["payload"] = blank(payload)
    elif kind == "response_item" and payload.get("type") == "message":
        if payload.get("role") == "developer" or not _mentions_fixture(payload.get("content")):
            payload["content"] = blank(payload.get("content"))
    return out


def _git_config(key: str) -> str:
    done = subprocess.run(
        ["git", "config", "--get", key], capture_output=True, text=True, check=False
    )
    return done.stdout.strip()


def _identity() -> dict[str, str]:
    return {
        "home directory": str(Path.home()),
        "user name": getpass.getuser(),
        "git email": _git_config("user.email"),
        "git name": _git_config("user.name"),
    }


def _replacer(cwd: Path) -> Callable[[str], str]:
    pairs = [
        (str(cwd), FIXTURE_CWD),
        (str(cwd).removeprefix("/private"), FIXTURE_CWD),
        (str(Path.home()), FIXTURE_HOME),
    ]
    pairs += [(value, "fixture-user") for value in _identity().values() if value]

    def replace(text: str) -> str:
        for old, new in pairs:
            text = text.replace(old, new)
        return text

    return replace


def write_fixture(
    src: Path,
    dest: Path,
    scrubber: Callable[[dict[str, Any]], dict[str, Any]],
    replace: Callable[[str], str],
) -> None:
    lines = [
        json.dumps(
            scrub(scrubber(json.loads(raw)), replace), ensure_ascii=False, separators=(",", ":")
        )
        for raw in src.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    text = "\n".join(lines) + "\n"
    for label, value in _identity().items():
        if value and value in text:
            raise RecorderError(f"{dest.name}: scrubbed output still contains the {label}")
    dest.write_text(text, encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record the handoff test fixtures.")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "fixtures" / "sessions"
    )
    parser.add_argument("--only", choices=("claude", "codex"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    claude_home = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    cwd = scratch_repo()
    replace = _replacer(cwd)
    try:
        if args.only in (None, "claude"):
            src = record_claude(cwd, claude_home)
            write_fixture(src, args.out / "claude_codewords.jsonl", scrub_claude, replace)
            print(f"claude fixture written from {src}")
        if args.only in (None, "codex"):
            src = record_codex(cwd, codex_home)
            write_fixture(src, args.out / "codex_codewords.jsonl", scrub_codex, replace)
            print(f"codex fixture written from {src}")
    except RecorderError as exc:
        print(f"session_recorder: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write the fixture helpers** (`tests/session_fixtures.py`)

```python
"""Paths and install helpers for the committed synthetic session fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from session_recorder import CODE_WORDS, FIXTURE_CWD, FIXTURE_HOME, SENTINEL

__all__ = [
    "CLAUDE_FIXTURE",
    "CODEX_FIXTURE",
    "CODE_WORDS",
    "FIXTURES",
    "FIXTURE_CWD",
    "FIXTURE_HOME",
    "SENTINEL",
    "fixture_lines",
    "write_lines",
]

FIXTURES = Path(__file__).parent / "fixtures" / "sessions"
CLAUDE_FIXTURE = "claude_codewords.jsonl"
CODEX_FIXTURE = "codex_codewords.jsonl"


def fixture_lines(name: str) -> list[dict[str, Any]]:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_lines(path: Path, lines: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n" for obj in lines
    )
    path.write_text(body, encoding="utf-8", newline="\n")
    return path
```

- [ ] **Step 5: Record the fixtures (local; needs logged-in `claude` 2.1.296 and `codex` 0.158.0; spends a few cents)**

Run: `claude --version && codex --version && uv run python tests/session_recorder.py`
Expected:
- `2.1.296 (Claude Code)` and `codex-cli 0.158.0`;
- then `claude fixture written from …` and `codex fixture written from …`;
- exit 0.

If either version differs, record anyway and note the versions in the commit message. Task 6 and Task 8 read the version from the fixture, not from these constants.

Then inspect structure only (types, never content):

Run: `uv run python -c "import json,collections,sys;[print(n, collections.Counter((o.get('type'), (o.get('payload') or {}).get('type') if isinstance(o.get('payload'), dict) else o.get('subtype')) for o in map(json.loads, open(f'tests/fixtures/sessions/{n}')))) for n in ('claude_codewords.jsonl','codex_codewords.jsonl')]"`
Expected:
- Claude: `user`, `assistant` and `attachment` records.
- Codex: `session_meta`, `response_item`/`message`, `response_item`/`function_call` (or `custom_tool_call`) with the matching output type, and `event_msg`/`task_started`.

- [ ] **Step 6: Run the tests and see them pass**

Run: `uv run pytest tests/test_session_fixtures.py -q && uv run ruff check . && uv run ruff format --check .`
Expected: 3 passed.

- [ ] **Step 7: Commit **

```bash
git add tests/session_recorder.py tests/session_fixtures.py tests/test_session_fixtures.py tests/fixtures/sessions/claude_codewords.jsonl tests/fixtures/sessions/codex_codewords.jsonl
git commit -m "$(cat <<'MSG'
Record synthetic Claude Code and Codex session fixtures

Five turns each (ALPHA-DELTA code words, one echo call, the handoff trigger), recorded
from Claude Code 2.1.296 and codex-cli 0.158.0 in a throwaway repo and scrubbed of
paths, injected instructions and identity. No real session content.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 6: Claude Code adapter — locate and read

**Files:**
- Create: `src/loosy_goose/sessions/claude.py`
- Modify: `tests/session_fixtures.py` (add `install_claude`)
- Test: `tests/test_sessions_claude.py`

**Interfaces:**
- Consumes:
  - `Session`, `SessionRecord`, `SessionFormatError`, `read_jsonl`, `same_dir`, `tagged`, `text_blocks` (Task 3);
  - `MARKER_PREFIX`, `find_trigger`, `DEFAULT_TRIGGERS`, `unit_ids` (Tasks 3–4);
  - the fixtures (Task 5).
- Produces (`loosy_goose.sessions.claude`):
  - `TOOL: Tool = "claude"`, `TESTED_VERSIONS: tuple[str, ...] = ("2.1.296",)`
  - `default_home() -> Path`: `$CLAUDE_CONFIG_DIR` or `~/.claude`
  - `project_slug(cwd: str) -> str`
  - `locate(cwd: Path, session: str, home: Path) -> list[Path]`: newest first. `session` is `"current"`, a session id, or an absolute `.jsonl` path.
  - `read(path: Path) -> Session`: raises `SessionFormatError` when there are no uuid records.
- Produces (`session_fixtures`): `install_claude(home: Path, *, lines: list[dict[str, Any]] | None = None, session_id: str | None = None, project_dir: str | None = None) -> Path`

**Classification:**

| Record | category | notes |
|---|---|---|
| `user` with a `tool_result` part | message | `call_side="result"`, `call_ids` = `tool_use_id`s |
| `user` with `isCompactSummary` | message | `pinned` |
| `user` with `isMeta` | meta | e.g. a slash command's expanded body |
| `user` whose text starts with `MARKER_PREFIX` | side | an earlier fork's marker |
| other `user` | message | `prompt` = its text |
| `assistant` | message | `reply` = `message.id`; `thinking` if only thinking parts; tool_use → `call_side="call"`, `tool_name` |
| `attachment` | meta | `pinned` when before the first prompt in the chain |
| `system` `compact_boundary` | meta | `pinned` |
| `system` `turn_duration`, `stop_hook_summary`, `away_summary`, `informational`, `local_command` | side | |
| uuid-less known types (see Facts) | side | |
| anything else | unknown | |

`live` is the `parentUuid` walk back from the last non-sidechain uuid record, reversed. A missing parent stops the walk with a warning, because Claude Code truncates there too.

- [ ] **Step 1: Add the install helper** (append to `tests/session_fixtures.py`, and add `"install_claude"` to `__all__`)

```python
def install_claude(
    home: Path,
    *,
    lines: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    project_dir: str | None = None,
) -> Path:
    from loosy_goose.sessions import claude

    body = [dict(o) for o in (lines if lines is not None else fixture_lines(CLAUDE_FIXTURE))]
    sid = session_id or next(str(o["sessionId"]) for o in body if "sessionId" in o)
    for obj in body:
        for key in ("sessionId", "session_id"):
            if key in obj:
                obj[key] = sid
    folder = project_dir or claude.project_slug(str(Path(FIXTURE_CWD)))
    return write_lines(home / "projects" / folder / f"{sid}.jsonl", body)
```

- [ ] **Step 2: Write the failing tests** (`tests/test_sessions_claude.py`)

```python
import os
from pathlib import Path

from session_fixtures import (
    CLAUDE_FIXTURE,
    CODE_WORDS,
    FIXTURE_CWD,
    FIXTURES,
    fixture_lines,
    install_claude,
    write_lines,
)

from loosy_goose.sessions import claude
from loosy_goose.sessions.plan import DEFAULT_TRIGGERS, MARKER_PREFIX, find_trigger, unit_ids


def _read() -> claude.Session:
    return claude.read(FIXTURES / CLAUDE_FIXTURE)


def test_reads_the_fixture_session() -> None:
    s = _read()
    assert (s.tool, s.cwd) == ("claude", FIXTURE_CWD)
    assert len(s.session_id) == 36 and s.cli_version is not None
    prompts = s.prompts()
    assert len(prompts) == 5
    assert [w for w in CODE_WORDS for p in prompts if w in p] == list(CODE_WORDS)
    assert "<command-name>/loosy</command-name>" in prompts[-1]


def test_the_echo_call_and_its_result_form_one_unit() -> None:
    s = _read()
    calls = [ri for ri in s.live if s.records[ri].call_side == "call"]
    results = [ri for ri in s.live if s.records[ri].call_side == "result"]
    assert calls and results
    assert {c for ri in calls for c in s.records[ri].call_ids} == {
        c for ri in results for c in s.records[ri].call_ids
    }
    assert all(s.records[ri].tool_name == "Bash" for ri in calls)
    units = unit_ids(s)
    assert {units[ri] for ri in calls} == {units[ri] for ri in results}


def test_the_trigger_is_the_recorded_slash_command() -> None:
    s = _read()
    t = find_trigger(s, DEFAULT_TRIGGERS["claude"])
    assert t.how == "text" and t.position is not None
    assert "/loosy" in (s.records[s.live[t.position]].prompt or "")


def test_meta_and_startup_records_are_not_prompts() -> None:
    s = _read()
    for ri in s.live:
        rec = s.records[ri]
        if rec.raw.get("isMeta"):
            assert rec.category == "meta" and rec.prompt is None
        if rec.kind == "attachment" and rec.exchange == 0:
            assert rec.pinned


def test_side_unknown_and_marker_records_are_classified(tmp_path: Path) -> None:
    lines = fixture_lines(CLAUDE_FIXTURE)
    sid = next(str(o["sessionId"]) for o in lines if "sessionId" in o)
    first = next(o for o in lines if isinstance(o.get("uuid"), str))
    marker = {
        "type": "user", "uuid": "11111111-1111-4111-8111-111111111111",
        "parentUuid": None, "sessionId": sid,
        "message": {"role": "user", "content": f"{MARKER_PREFIX}x on 2026-10-13; ...]"},
    }
    first["parentUuid"] = marker["uuid"]
    extra = [{"type": "queue-operation", "sessionId": sid}, {"type": "brand-new-thing", "sessionId": sid}]
    s = claude.read(write_lines(tmp_path / "s.jsonl", [marker, *lines, *extra]))
    kinds = {r.kind: r.category for r in s.records}
    assert kinds["queue-operation"] == "side"
    assert kinds["brand-new-thing"] == "unknown"
    assert kinds["user:loosy-marker"] == "side"
    assert len(s.records) - 1 in s.loose


def test_a_missing_parent_truncates_history_with_a_warning(tmp_path: Path) -> None:
    lines = fixture_lines(CLAUDE_FIXTURE)
    chain = [o for o in lines if isinstance(o.get("uuid"), str)]
    victim = chain[len(chain) // 2]
    victim["parentUuid"] = "00000000-0000-4000-8000-000000000000"
    s = claude.read(write_lines(tmp_path / "s.jsonl", lines))
    assert any("not in the file" in w for w in s.warnings)
    assert s.records[s.live[0]].raw["uuid"] == victim["uuid"]


def test_locate_finds_the_cwd_sessions_newest_first(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    older = install_claude(home, session_id="11111111-1111-4111-8111-111111111111")
    newer = install_claude(home, session_id="22222222-2222-4222-8222-222222222222")
    os.utime(older, (1, 1))
    assert claude.locate(Path(FIXTURE_CWD), "current", home) == [newer, older]
    assert claude.locate(tmp_path / "elsewhere", "current", home) == []


def test_locate_falls_back_to_the_recorded_cwd_when_the_slug_differs(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    path = install_claude(home, project_dir="-some-other-slug")
    assert claude.locate(Path(FIXTURE_CWD), "current", home) == [path]


def test_locate_by_explicit_id_or_path_and_rejects_glob_input(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    sid = "33333333-3333-4333-8333-333333333333"
    path = install_claude(home, session_id=sid)
    assert claude.locate(tmp_path / "anywhere", sid, home) == [path]
    assert claude.locate(tmp_path / "anywhere", str(path), home) == [path]
    assert claude.locate(tmp_path / "anywhere", "*", home) == []
```

`claude.Session` is the name re-exported by the `from loosy_goose.sessions.model import Session` line in `claude.py`. mypy does not check tests, but keep the annotation readable.

- [ ] **Step 3: Run them and see them fail**

Run: `uv run pytest tests/test_sessions_claude.py -q`
Expected: collection error `ImportError: cannot import name 'claude' from 'loosy_goose.sessions'`.

- [ ] **Step 4: Implement `src/loosy_goose/sessions/claude.py` (read side)**

```python
"""Claude Code session files: `<home>/projects/<cwd slug>/<sessionId>.jsonl`.

Undocumented, and the format changes between releases. History is rebuilt by walking
`parentUuid` back from the leaf, not by line order. A gap that is not re-linked loads without
error but silently truncates history (probe, 2026-10-12, Claude Code 2.1.296).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from loosy_goose.sessions.model import (
    Session,
    SessionFormatError,
    SessionRecord,
    Tool,
    read_jsonl,
    same_dir,
    tagged,
    text_blocks,
)
from loosy_goose.sessions.plan import MARKER_PREFIX
from loosy_goose.transcript import Block

__all__ = ["Session", "TESTED_VERSIONS", "TOOL", "default_home", "locate", "project_slug", "read"]

TOOL: Tool = "claude"
TESTED_VERSIONS: tuple[str, ...] = ("2.1.296",)

_ID = re.compile(r"[0-9A-Za-z-]{8,64}")
_SIDE_TYPES = frozenset(
    {
        "queue-operation", "last-prompt", "cost-state", "mode", "permission-mode",
        "atis-latch", "ai-title", "pr-link", "agent-name", "file-history-snapshot",
        "file-history-delta", "continued-in",
    }
)
_SIDE_SYSTEM = frozenset(
    {"turn_duration", "stop_hook_summary", "away_summary", "informational", "local_command"}
)
_THINKING = frozenset({"thinking", "redacted_thinking"})


def default_home() -> Path:
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(env) if env else Path.home() / ".claude"


def project_slug(cwd: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", cwd)


def _recorded_cwd(path: Path) -> str | None:
    with path.open("rb") as fh:
        for n, raw in enumerate(fh):
            if n >= 50:
                break
            try:
                obj = json.loads(raw)
            except ValueError:
                continue
            if isinstance(obj, dict) and isinstance(obj.get("cwd"), str):
                return str(obj["cwd"])
    return None


def locate(cwd: Path, session: str, home: Path) -> list[Path]:
    projects = home / "projects"
    if session != "current":
        given = Path(session)
        if given.is_absolute() and given.is_file():
            return [given]
        if not _ID.fullmatch(session):
            return []
        return sorted(projects.glob(f"*/{session}.jsonl"))
    slugged = projects / project_slug(str(cwd))
    # The slug rule is inferred, not documented, so the recorded cwd decides; the slug only
    # narrows the scan when its directory exists.
    pool = slugged.glob("*.jsonl") if slugged.is_dir() else projects.glob("*/*.jsonl")
    found = [p for p in pool if (rec := _recorded_cwd(p)) is not None and same_dir(rec, cwd)]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _blocks(content: Any) -> list[Block]:
    if isinstance(content, str):
        return text_blocks(content, 0)
    if not isinstance(content, list):
        return []
    blocks: list[Block] = []
    for i, part in enumerate(content):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            blocks += text_blocks(str(part.get("text", "")), i)
        elif kind == "tool_use":
            body = json.dumps(part.get("input", {}), ensure_ascii=False, sort_keys=True)
            blocks += tagged("tool_use", body, i, name=str(part.get("name", "")))
        elif kind == "tool_result":
            blocks += tagged("tool_result", _content_text(part.get("content")), i)
        elif kind == "thinking":
            blocks += tagged("thinking", str(part.get("thinking", "")), i)
    return blocks


def _message(obj: dict[str, Any]) -> dict[str, Any]:
    msg = obj.get("message")
    return msg if isinstance(msg, dict) else {}


def _user(n: int, obj: dict[str, Any]) -> SessionRecord:
    content = _message(obj).get("content")
    parts = [p for p in content if isinstance(p, dict)] if isinstance(content, list) else []
    results = [p for p in parts if p.get("type") == "tool_result"]
    if results:
        return SessionRecord(
            n, obj, "user:tool_result", "message", role="user", blocks=_blocks(content),
            call_ids=tuple(str(p.get("tool_use_id", "")) for p in results), call_side="result",
        )
    if obj.get("isCompactSummary"):
        return SessionRecord(
            n, obj, "user:compact_summary", "message", role="user",
            blocks=_blocks(content), pinned=True,
        )
    if obj.get("isMeta"):
        return SessionRecord(n, obj, "user:meta", "meta", role="user", blocks=_blocks(content))
    text = _content_text(content)
    if text.startswith(MARKER_PREFIX):
        return SessionRecord(n, obj, "user:loosy-marker", "side")
    return SessionRecord(
        n, obj, "user", "message", role="user", blocks=_blocks(content), prompt=text
    )


def _assistant(n: int, obj: dict[str, Any]) -> SessionRecord:
    msg = _message(obj)
    content = msg.get("content")
    parts = [p for p in content if isinstance(p, dict)] if isinstance(content, list) else []
    types = {p.get("type") for p in parts}
    calls = [p for p in parts if p.get("type") == "tool_use"]
    return SessionRecord(
        n, obj, "assistant", "message", role="assistant", blocks=_blocks(content),
        thinking=bool(types) and types <= _THINKING,
        reply=str(msg["id"]) if "id" in msg else None,
        call_ids=tuple(str(c.get("id", "")) for c in calls),
        call_side="call" if calls else None,
        tool_name=str(calls[-1].get("name", "")) if calls else None,
    )


def _classify(n: int, obj: dict[str, Any]) -> SessionRecord:
    rtype = str(obj.get("type"))
    if rtype == "user":
        return _user(n, obj)
    if rtype == "assistant":
        return _assistant(n, obj)
    if rtype == "attachment":
        return SessionRecord(n, obj, "attachment", "meta")
    if rtype == "system":
        sub = str(obj.get("subtype"))
        if sub == "compact_boundary":
            return SessionRecord(n, obj, "system:compact_boundary", "meta", pinned=True)
        return SessionRecord(n, obj, f"system:{sub}", "side" if sub in _SIDE_SYSTEM else "unknown")
    return SessionRecord(n, obj, rtype, "side" if rtype in _SIDE_TYPES else "unknown")


def _chain(records: list[SessionRecord], by_uuid: dict[str, int], warnings: list[str]) -> list[int]:
    leaf = next(
        (
            i
            for i in range(len(records) - 1, -1, -1)
            if isinstance(records[i].raw.get("uuid"), str) and not records[i].raw.get("isSidechain")
        ),
        None,
    )
    chain: list[int] = []
    seen: set[int] = set()
    current = leaf
    while current is not None and current not in seen:
        seen.add(current)
        chain.append(current)
        parent = records[current].raw.get("parentUuid")
        if parent is None:
            break
        current = by_uuid.get(str(parent))
        if current is None:
            warnings.append(
                f"line {records[chain[-1]].line + 1}: parent {parent} is not in the file; "
                "history before it is not loaded"
            )
    chain.reverse()
    return chain


def _mark_exchanges(records: list[SessionRecord], live: list[int]) -> None:
    exchange = 0
    for ri in live:
        rec = records[ri]
        if rec.prompt is not None:
            exchange += 1
        rec.exchange = exchange
        if rec.kind == "attachment" and exchange == 0:
            rec.pinned = True


def _first_str(records: list[SessionRecord], key: str) -> str | None:
    return next((str(r.raw[key]) for r in records if isinstance(r.raw.get(key), str)), None)


def read(path: Path) -> Session:
    lines, warnings = read_jsonl(path)
    records = [_classify(n, obj) for n, obj in lines]
    by_uuid = {
        str(r.raw["uuid"]): i for i, r in enumerate(records) if isinstance(r.raw.get("uuid"), str)
    }
    if not by_uuid:
        raise SessionFormatError(f"{path}: no Claude Code message records")
    live = _chain(records, by_uuid, warnings)
    _mark_exchanges(records, live)
    return Session(
        tool=TOOL,
        path=path,
        session_id=_first_str(records, "sessionId") or "",
        cwd=_first_str(records, "cwd") or "",
        cli_version=_first_str(list(reversed(records)), "version"),
        records=records,
        live=live,
        loose=[i for i, r in enumerate(records) if not isinstance(r.raw.get("uuid"), str)],
        warnings=warnings,
    )
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_sessions_claude.py -q && uv run mypy src && uv run ruff check . && uv run ruff format --check .`
Expected: 9 passed; mypy `Success` (format any flagged file with `uv run ruff format <file>`).

- [ ] **Step 6: Commit**

```bash
git add src/loosy_goose/sessions/claude.py tests/session_fixtures.py tests/test_sessions_claude.py
git commit -m "$(cat <<'MSG'
Read Claude Code sessions into the session model

Locate by cwd (slug dir, falling back to the recorded cwd), explicit id or path; history is
the parentUuid chain from the leaf; startup attachments and compaction records are pinned,
side records dropped, unknown types flagged, earlier loosy-goose markers superseded.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 7: Claude Code adapter — write the fork and validate it

**Files:**
- Modify: `src/loosy_goose/sessions/claude.py` (append; extend imports and `__all__`)
- Test: `tests/test_sessions_claude.py` (append)

**Interfaces:**
- Consumes: `ForkPlan`, `plan_fork`, `split_region`, `find_trigger`, `DEFAULT_TRIGGERS`, `MARKER_PREFIX` (Tasks 3–4); `dump_line`, `read_jsonl` (Task 3); `kept_where` (Task 3 builders); `read` (Task 6).
- Produces (`loosy_goose.sessions.claude`):
  - `write_fork(session: Session, plan: ForkPlan, new_id: str, out_dir: Path) -> Path`: writes `out_dir/<new_id>.jsonl`.
  - `validate(fork: Path, session_id: str) -> list[str]`: the failed checks, from `"parse"`, `"chain"`, `"coverage"`, `"tool-pairs"`, `"session-id"`.
  - `fork_destination(source: Path, new_id: str, home: Path, now: datetime) -> Path`: `source.parent / f"{new_id}.jsonl"`.
  - `new_session_id() -> str`: uuid4.
  - `resume_command(session_id: str, cwd: str) -> str`: `cd <quoted cwd> && claude --resume <id>`.
  - `verify_command(session_id: str) -> list[str]`: `["claude", "-p", "Reply with exactly: OK", "--resume", id]`.

**Writer rules:**

- Kept chain records are written in history order, each a deep copy with `sessionId` (and `session_id` when present) set to the new id.
- Each record's `parentUuid` is set to the previous written record's uuid. This re-links across every dropped stretch.
- The marker is a new `user` record whose plain string content is `plan.marker`. It sits at the root, or right after the last kept `compact_boundary` ([RF1]: Claude Code reads history from the boundary on, so a marker before it would be cut).
- `plan.rewrites` replace `message.content` parts: `text` for text parts, `content` for `tool_result` parts. `""` drops the part.
- `passthrough` records are appended with the new session id.
- The file ends with a `last-prompt` record pointing `leafUuid` at the last chain record. This settles the spec's open item: writing it costs nothing and mirrors what the CLI writes.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sessions_claude.py`; extend the imports)

Add to the imports:

```python
import dataclasses
import json
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from session_builders import kept_where

from loosy_goose.segment import Segment
from loosy_goose.sessions.plan import ForkPlan, plan_fork, split_region
```

Append:

```python
def _fork(
    tmp_path: Path,
    lines: list[dict[str, Any]] | None = None,
    keep: Callable[[Segment], bool] = lambda s: True,
) -> tuple[claude.Session, ForkPlan, str, Path]:
    source = (
        write_lines(tmp_path / "src.jsonl", lines) if lines is not None else FIXTURES / CLAUDE_FIXTURE
    )
    s = claude.read(source)
    region = split_region(s, find_trigger(s, DEFAULT_TRIGGERS["claude"]), protect_tail_tokens=0)
    plan = plan_fork(s, region, kept_where(region, keep), date="2026-10-13")
    new_id = claude.new_session_id()
    return s, plan, new_id, claude.write_fork(s, plan, new_id, tmp_path / "out")


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_fork_without_beta_validates_and_relinks(tmp_path: Path) -> None:
    _, _, new_id, fork = _fork(tmp_path, keep=lambda seg: "BETA" not in seg.text)
    assert fork.name == f"{new_id}.jsonl"
    assert claude.validate(fork, new_id) == []
    f = claude.read(fork)
    assert f.session_id == new_id
    assert f.records[f.live[0]].kind == "user:loosy-marker"
    text = " ".join(f.prompts())
    assert "BETA" not in text
    assert all(w in text for w in ("ALPHA", "GAMMA", "DELTA"))
    assert "/loosy" not in text


def test_kept_records_keep_their_ids_and_the_file_ends_on_the_leaf(tmp_path: Path) -> None:
    s, _, new_id, fork = _fork(tmp_path)
    original = {r.raw["uuid"] for r in s.records if isinstance(r.raw.get("uuid"), str)}
    out = _lines(fork)
    chain = [o for o in out if isinstance(o.get("uuid"), str)]
    assert {o["uuid"] for o in chain[1:]} <= original
    assert all(o.get("sessionId") == new_id for o in out if "sessionId" in o)
    assert all(o.get("session_id") == new_id for o in out if "session_id" in o)
    assert out[-1]["type"] == "last-prompt"
    assert out[-1]["leafUuid"] == chain[-1]["uuid"]
    assert "DELTA" in out[-1]["lastPrompt"]


def test_a_rewritten_tool_result_is_written(tmp_path: Path) -> None:
    s, plan, new_id, _ = _fork(tmp_path)
    ri = next(ri for ri in plan.keep if s.records[ri].call_side == "result")
    part = int(s.records[ri].blocks[0].meta["part"])
    trimmed = dataclasses.replace(plan, rewrites={ri: {part: "trimmed"}})
    fork = claude.write_fork(s, trimmed, new_id, tmp_path / "trimmed")
    assert claude.validate(fork, new_id) == []
    f = claude.read(fork)
    rec = next(r for r in f.records if r.raw.get("uuid") == s.records[ri].raw["uuid"])
    assert [b.text for b in rec.blocks] == ["trimmed"]


def _corrupt(lines: list[dict[str, Any]], how: str) -> list[dict[str, Any]]:
    chain = [o for o in lines if isinstance(o.get("uuid"), str)]
    if how == "chain":
        chain[len(chain) // 2]["parentUuid"] = "00000000-0000-4000-8000-000000000000"
    elif how == "tool-pairs":
        result = next(
            o for o in chain
            if isinstance(o.get("message", {}).get("content"), list)
            and any(p.get("type") == "tool_result" for p in o["message"]["content"])
        )
        result["message"]["content"] = [{"type": "text", "text": "gone"}]
    elif how == "session-id":
        chain[-1]["sessionId"] = "someone-else"
    elif how == "coverage":
        lines.insert(0, {
            "type": "user", "uuid": str(uuid.uuid4()), "parentUuid": None,
            "sessionId": chain[0]["sessionId"], "message": {"role": "user", "content": "stray"},
        })
    return lines


@pytest.mark.parametrize("how", ["chain", "tool-pairs", "session-id", "coverage"])
def test_validate_names_the_failed_check(tmp_path: Path, how: str) -> None:
    _, _, new_id, fork = _fork(tmp_path)
    bad = write_lines(tmp_path / "bad.jsonl", _corrupt(_lines(fork), how))
    assert how in claude.validate(bad, new_id)


def _compacted_lines() -> list[dict[str, Any]]:
    """The fixture as if built-in /compact had run just before the BETA turn."""
    lines = fixture_lines(CLAUDE_FIXTURE)
    sid = next(str(o["sessionId"]) for o in lines if "sessionId" in o)
    beta = next(
        i for i, o in enumerate(lines)
        if o.get("type") == "user" and not o.get("isMeta") and "toolUseResult" not in o
        and "BETA" in json.dumps(o.get("message"))
    )
    boundary_id, summary_id = str(uuid.uuid4()), str(uuid.uuid4())
    boundary = {
        "type": "system", "subtype": "compact_boundary", "uuid": boundary_id,
        "parentUuid": None, "logicalParentUuid": lines[beta]["parentUuid"],
        "sessionId": sid, "isSidechain": False, "cwd": FIXTURE_CWD,
        "content": "Conversation compacted",
        "compactMetadata": {"trigger": "manual", "preTokens": 1000},
    }
    summary = {
        "type": "user", "uuid": summary_id, "parentUuid": boundary_id, "sessionId": sid,
        "isSidechain": False, "isCompactSummary": True, "isVisibleInTranscriptOnly": True,
        "cwd": FIXTURE_CWD,
        "message": {"role": "user", "content": "Summary: the user shared one code word earlier."},
    }
    lines[beta]["parentUuid"] = summary_id
    return lines[:beta] + [boundary, summary] + lines[beta:]


def test_rf1_an_already_compacted_session_forks_from_its_boundary(tmp_path: Path) -> None:
    lines = _compacted_lines()
    s = claude.read(write_lines(tmp_path / "compacted.jsonl", lines))
    assert s.records[s.live[0]].kind == "system:compact_boundary"
    assert s.records[s.live[1]].pinned and s.records[s.live[1]].kind == "user:compact_summary"
    assert not any("ALPHA" in p for p in s.prompts())
    _, _, new_id, fork = _fork(tmp_path, lines)
    assert claude.validate(fork, new_id) == []
    f = claude.read(fork)
    kinds = [f.records[ri].kind for ri in f.live[:3]]
    assert kinds == ["system:compact_boundary", "user:loosy-marker", "user:compact_summary"]
    assert "ALPHA" not in fork.read_text(encoding="utf-8")


def test_destination_and_commands() -> None:
    src = Path("/h/projects/-work/old.jsonl")
    assert claude.fork_destination(src, "new", Path("/h"), datetime(2026, 10, 13)) == (
        Path("/h/projects/-work/new.jsonl")
    )
    assert claude.resume_command("abc", "/work/my dir") == "cd '/work/my dir' && claude --resume abc"
    assert claude.verify_command("abc") == ["claude", "-p", "Reply with exactly: OK", "--resume", "abc"]
    assert uuid.UUID(claude.new_session_id()).version == 4
```

The `ALPHA` assertion on the fork text works because the fixture's ALPHA prompt and its "noted." reply are before the boundary, and the injected summary avoids naming the word. The attachments before ALPHA are also pre-boundary, and they are scrubbed to `<scrubbed>`.

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_sessions_claude.py -q`
Expected: the new tests fail with `AttributeError: module 'loosy_goose.sessions.claude' has no attribute 'new_session_id'`.

- [ ] **Step 3: Implement** (append to `claude.py`)

Extend the imports to add `copy`, `shlex`, `uuid`, `from datetime import datetime`, `dump_line` from `loosy_goose.sessions.model`, and `ForkPlan` alongside `MARKER_PREFIX` from `loosy_goose.sessions.plan`. Add `"fork_destination"`, `"new_session_id"`, `"resume_command"`, `"validate"`, `"verify_command"`, `"write_fork"` to `__all__`. Then append:

```python
_MARKER_KEYS = ("isSidechain", "userType", "cwd", "version", "gitBranch", "entrypoint", "timestamp")


def new_session_id() -> str:
    return str(uuid.uuid4())


def fork_destination(source: Path, new_id: str, home: Path, now: datetime) -> Path:
    return source.parent / f"{new_id}.jsonl"


def resume_command(session_id: str, cwd: str) -> str:
    return f"cd {shlex.quote(cwd)} && claude --resume {session_id}"


def verify_command(session_id: str) -> list[str]:
    return ["claude", "-p", "Reply with exactly: OK", "--resume", session_id]


def _apply_rewrites(obj: dict[str, Any], rewrite: dict[int, str]) -> None:
    if not rewrite:
        return
    msg = obj["message"]
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = rewrite.get(0, content)
        return
    parts: list[Any] = []
    for i, part in enumerate(content):
        if i not in rewrite:
            parts.append(part)
        elif rewrite[i]:
            key = "content" if part.get("type") == "tool_result" else "text"
            parts.append({**part, key: rewrite[i]})
    msg["content"] = parts


def _with_session(obj: dict[str, Any], new_id: str) -> dict[str, Any]:
    for key in ("sessionId", "session_id"):
        if key in obj:
            obj[key] = new_id
    return obj


def write_fork(session: Session, plan: ForkPlan, new_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = session.records
    kept: list[dict[str, Any]] = []
    for ri in plan.keep:
        obj = copy.deepcopy(records[ri].raw)
        _apply_rewrites(obj, plan.rewrites.get(ri, {}))
        kept.append(obj)
    template = next(
        (records[ri].raw for ri in plan.keep if records[ri].prompt is not None), {}
    )
    marker: dict[str, Any] = {k: template[k] for k in _MARKER_KEYS if k in template}
    marker.update(
        {"type": "user", "uuid": new_session_id(), "message": {"role": "user", "content": plan.marker}}
    )
    # Claude Code reads history from the last compact boundary on; a marker above it is cut.
    boundary = max(
        (i for i, ri in enumerate(plan.keep) if records[ri].kind == "system:compact_boundary"),
        default=-1,
    )
    chain = kept[: boundary + 1] + [marker] + kept[boundary + 1 :]
    parent: str | None = None
    for obj in chain:
        obj["parentUuid"] = parent
        obj["sessionId"] = new_id
        _with_session(obj, new_id)
        parent = str(obj["uuid"])
    loose = [_with_session(copy.deepcopy(records[ri].raw), new_id) for ri in plan.passthrough]
    last_prompt = next(
        (p for ri in reversed(plan.keep) if (p := records[ri].prompt) is not None), ""
    )
    leaf = {"type": "last-prompt", "lastPrompt": last_prompt, "leafUuid": parent, "sessionId": new_id}
    dest = out_dir / f"{new_id}.jsonl"
    with dest.open("wb") as fh:
        for obj in [*chain, *loose, leaf]:
            fh.write(dump_line(obj))
    return dest


def validate(fork: Path, session_id: str) -> list[str]:
    lines, problems = read_jsonl(fork)
    failures = ["parse"] if problems else []
    objs = [obj for _, obj in lines]
    chain = [o for o in objs if isinstance(o.get("uuid"), str)]
    by_uuid = {str(o["uuid"]): o for o in chain}
    seen: set[str] = set()
    walked: list[dict[str, Any]] = []
    intact = bool(chain)
    current: dict[str, Any] | None = chain[-1] if chain else None
    while current is not None:
        uid = str(current["uuid"])
        if uid in seen:
            intact = False
            break
        seen.add(uid)
        walked.append(current)
        parent = current.get("parentUuid")
        if parent is None:
            break
        current = by_uuid.get(str(parent))
        if current is None:
            intact = False
    if not intact:
        failures.append("chain")
    if seen != set(by_uuid):
        failures.append("coverage")
    uses: set[str] = set()
    results: set[str] = set()
    for obj in walked:
        content = obj.get("message", {}).get("content") if isinstance(obj.get("message"), dict) else None
        for part in content if isinstance(content, list) else []:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                uses.add(str(part.get("id")))
            if isinstance(part, dict) and part.get("type") == "tool_result":
                results.add(str(part.get("tool_use_id")))
    if uses != results:
        failures.append("tool-pairs")
    if {o.get("sessionId") for o in objs if "sessionId" in o} != {session_id}:
        failures.append("session-id")
    return failures
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_sessions_claude.py -q && uv run mypy src && uv run ruff check . && uv run ruff format --check .`
Expected: 18 passed; mypy `Success` (run `uv run ruff format` on flagged files first).

- [ ] **Step 5: Commit **

```bash
git add src/loosy_goose/sessions/claude.py tests/test_sessions_claude.py
git commit -m "$(cat <<'MSG'
Write and validate Claude Code forks

New session id throughout, chain re-linked across every dropped stretch, marker at the
root or right after a kept compact boundary, last-prompt pointing at the new leaf.
Validation walks parentUuid from the leaf and checks coverage, tool pairs and one id.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 8: Codex adapter — locate and read

**Files:**
- Create: `src/loosy_goose/sessions/codex.py`
- Modify: `tests/session_fixtures.py` (add `install_codex`)
- Test: `tests/test_sessions_codex.py`

**Interfaces:**
- Consumes: as Task 6, plus `exchange_starts`, `Trigger` (Task 3).
- Produces (`loosy_goose.sessions.codex`):
  - `TOOL: Tool = "codex"`, `TESTED_VERSIONS: tuple[str, ...] = ("0.158.0",)`
  - `default_home() -> Path`: `$CODEX_HOME` or `~/.codex`
  - `locate(cwd: Path, session: str, home: Path) -> list[Path]`
  - `read(path: Path) -> Session`: raises `SessionFormatError` if line 0 is not `session_meta`.
- Produces (`session_fixtures`): `install_codex(home: Path, *, lines: list[dict[str, Any]] | None = None, session_id: str | None = None, stamp: str = "2026-10-13T00-00-00") -> Path`

**Classification:**

| Record (`type` / `payload.type`) | category | notes |
|---|---|---|
| `session_meta`, `compacted` | meta | `pinned` |
| `response_item` / `message`, role `assistant` | message | blocks from `output_text` |
| `response_item` / `message`, role `user` followed by `event_msg` `user_message` before the next `task_started` | message | `prompt` = the event's `message` |
| `response_item` / `message`, role `user` starting with `MARKER_PREFIX` | side | an earlier fork's marker |
| other `response_item` / `message`, roles `user` and `developer` | meta | `pinned` (instructions, environment context) |
| `response_item` / `reasoning` | message | `thinking`; blocks from `summary` texts |
| `response_item` / `function_call`, `custom_tool_call` | message | `call_side="call"`, `call_ids=(call_id,)`, `tool_name` |
| `response_item` / `function_call_output`, `custom_tool_call_output` | message | `call_side="result"`, role `tool` |
| `response_item` / `web_search_call`, `agent_message`, `tool_search_call`, `tool_search_output` | meta | kept with their exchange; the tool search pair is never trimmed |
| `event_msg` (any), `turn_context`, `world_state`, `token_usage_record`, `inter_agent_communication_metadata` | meta | |
| anything else | unknown | |

- `live` is every record, or, after a `compacted` record, the `session_meta` lines plus everything from the last `compacted` on ([RF4]).
- `exchange` counts `task_started` events.
- `open_exchange` is the last exchange whose `task_started` has no `task_complete`/`turn_aborted` after it.
- `reply` for assistant-side records is `f"exchange-{n}"`, so `keep_thinking` keeps reasoning only in exchanges that keep something.

- [ ] **Step 1: Add the install helper** (append to `tests/session_fixtures.py`, and add `"install_codex"` to `__all__`)

```python
def install_codex(
    home: Path,
    *,
    lines: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
    stamp: str = "2026-10-13T00-00-00",
) -> Path:
    body = lines if lines is not None else fixture_lines(CODEX_FIXTURE)
    old = str(body[0]["payload"]["id"])
    sid = session_id or old
    body = [json.loads(json.dumps(o).replace(old, sid)) for o in body]
    day = stamp[:10].replace("-", "/")
    return write_lines(home / "sessions" / day / f"rollout-{stamp}-{sid}.jsonl", body)
```

- [ ] **Step 2: Write the failing tests** (`tests/test_sessions_codex.py`)

```python
import json
from pathlib import Path
from typing import Any

from session_fixtures import (
    CODE_WORDS,
    CODEX_FIXTURE,
    FIXTURE_CWD,
    FIXTURES,
    SENTINEL,
    fixture_lines,
    install_codex,
    write_lines,
)

from loosy_goose.sessions import codex
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    MARKER_PREFIX,
    Trigger,
    exchange_starts,
    find_trigger,
    unit_ids,
)


def _read() -> codex.Session:
    return codex.read(FIXTURES / CODEX_FIXTURE)


def test_reads_the_fixture_session() -> None:
    s = _read()
    assert (s.tool, s.cwd) == ("codex", FIXTURE_CWD)
    assert s.session_id == fixture_lines(CODEX_FIXTURE)[0]["payload"]["id"]
    assert s.cli_version is not None
    prompts = s.prompts()
    assert len(prompts) == 5
    assert [w for w in CODE_WORDS for p in prompts if w in p] == list(CODE_WORDS)
    assert SENTINEL in prompts[-1]


def test_calls_pair_with_their_outputs() -> None:
    s = _read()
    calls = [ri for ri in s.live if s.records[ri].call_side == "call"]
    outputs = [ri for ri in s.live if s.records[ri].call_side == "result"]
    assert calls and outputs
    assert {c for ri in calls for c in s.records[ri].call_ids} == {
        c for ri in outputs for c in s.records[ri].call_ids
    }
    assert all(s.records[ri].tool_name for ri in calls)
    units = unit_ids(s)
    assert {units[ri] for ri in calls} == {units[ri] for ri in outputs}


def test_context_messages_are_pinned_and_prompts_are_messages() -> None:
    s = _read()
    for rec in s.records:
        if rec.kind == "response_item:message:developer":
            assert rec.pinned and rec.prompt is None
        if rec.prompt is not None:
            assert rec.category == "message" and not rec.pinned


def test_exchanges_follow_task_started_and_the_trigger_is_the_sentinel_turn() -> None:
    s = _read()
    started = sum(1 for r in s.records if r.kind == "event_msg:task_started")
    assert max(r.exchange for r in s.records) == started
    assert s.open_exchange is None
    assert find_trigger(s, DEFAULT_TRIGGERS["codex"]).how == "text"


def test_an_unclosed_last_turn_is_open(tmp_path: Path) -> None:
    lines = fixture_lines(CODEX_FIXTURE)
    last = max(
        i for i, o in enumerate(lines)
        if o["type"] == "event_msg" and o["payload"].get("type") == "task_complete"
    )
    s = codex.read(write_lines(tmp_path / "open.jsonl", lines[:last] + lines[last + 1 :]))
    assert s.open_exchange == max(r.exchange for r in s.records)
    position = exchange_starts(s)[s.open_exchange]
    assert find_trigger(s, ()) == Trigger(position, "open-turn")


def test_unknown_types_and_earlier_markers_are_classified(tmp_path: Path) -> None:
    lines = fixture_lines(CODEX_FIXTURE)
    stamp = lines[-1]["timestamp"]
    extra: list[dict[str, Any]] = [
        {"timestamp": stamp, "type": "brand_new", "payload": {}},
        {"timestamp": stamp, "type": "response_item", "payload": {"type": "hologram"}},
        {"timestamp": stamp, "type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": f"{MARKER_PREFIX}x on d; ...]"}],
        }},
    ]
    s = codex.read(write_lines(tmp_path / "s.jsonl", lines + extra))
    assert [r.category for r in s.records[-3:]] == ["unknown", "unknown", "side"]


def test_locate_by_cwd_id_and_path(tmp_path: Path) -> None:
    home = tmp_path / "codex"
    path = install_codex(home)
    sid = fixture_lines(CODEX_FIXTURE)[0]["payload"]["id"]
    assert codex.locate(Path(FIXTURE_CWD), "current", home) == [path]
    assert codex.locate(tmp_path / "elsewhere", "current", home) == []
    assert codex.locate(tmp_path / "elsewhere", sid, home) == [path]
    assert codex.locate(tmp_path / "elsewhere", str(path), home) == [path]
    assert codex.locate(tmp_path / "elsewhere", "*", home) == []


def test_a_file_without_session_meta_is_refused(tmp_path: Path) -> None:
    import pytest

    from loosy_goose.sessions.model import SessionFormatError

    path = write_lines(tmp_path / "bad.jsonl", [{"type": "event_msg", "payload": {}}])
    with pytest.raises(SessionFormatError, match="session_meta"):
        codex.read(path)
```

- [ ] **Step 3: Run them and see them fail**

Run: `uv run pytest tests/test_sessions_codex.py -q`
Expected: collection error `ImportError: cannot import name 'codex' from 'loosy_goose.sessions'`.

- [ ] **Step 4: Implement `src/loosy_goose/sessions/codex.py` (read side)**

```python
"""Codex session files: `<home>/sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl`.

Undocumented. Each line is `{timestamp, type, payload}` (newer lines add `ordinal`, the line
index). History is file order; a `compacted` record replaces everything before it. The session id
appears in many fields. Codex indexes a new file into its own sqlite by scanning; that database
is never touched here (probe, 2026-10-12, codex-cli 0.158.0).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from loosy_goose.sessions.model import (
    Session,
    SessionFormatError,
    SessionRecord,
    Tool,
    read_jsonl,
    same_dir,
    tagged,
    text_blocks,
)
from loosy_goose.sessions.plan import MARKER_PREFIX
from loosy_goose.transcript import Block

__all__ = ["Session", "TESTED_VERSIONS", "TOOL", "default_home", "locate", "read"]

TOOL: Tool = "codex"
TESTED_VERSIONS: tuple[str, ...] = ("0.158.0",)

_ID = re.compile(r"[0-9A-Za-z-]{8,64}")
_CALLS = frozenset({"function_call", "custom_tool_call"})
_OUTPUTS = frozenset({"function_call_output", "custom_tool_call_output"})
_META_ITEMS = frozenset({"web_search_call", "agent_message", "tool_search_call", "tool_search_output"})
_META_TOP = frozenset(
    {"event_msg", "turn_context", "world_state", "token_usage_record", "inter_agent_communication_metadata"}
)
_TURN_END = frozenset({"event_msg:task_complete", "event_msg:turn_aborted"})


def default_home() -> Path:
    env = os.environ.get("CODEX_HOME")
    return Path(env) if env else Path.home() / ".codex"


def _head(path: Path) -> dict[str, Any] | None:
    with path.open("rb") as fh:
        try:
            obj = json.loads(fh.readline())
        except ValueError:
            return None
    if isinstance(obj, dict) and obj.get("type") == "session_meta":
        payload = obj.get("payload")
        return payload if isinstance(payload, dict) else None
    return None


def locate(cwd: Path, session: str, home: Path) -> list[Path]:
    root = home / "sessions"
    if session != "current":
        given = Path(session)
        if given.is_absolute() and given.is_file():
            return [given]
        if not _ID.fullmatch(session):
            return []
        return sorted(root.glob(f"*/*/*/rollout-*-{session}.jsonl"))
    files = sorted(root.glob("*/*/*/rollout-*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[Path] = []
    for path in files:
        head = _head(path)
        if head is not None and isinstance(head.get("cwd"), str) and same_dir(head["cwd"], cwd):
            out.append(path)
    return out


def _texts(content: Any) -> list[tuple[int, str]]:
    if not isinstance(content, list):
        return []
    return [
        (i, str(c["text"]))
        for i, c in enumerate(content)
        if isinstance(c, dict) and isinstance(c.get("text"), str)
    ]


def _arguments(raw: Any) -> str:
    if not isinstance(raw, str):
        return json.dumps(raw, ensure_ascii=False, sort_keys=True)
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, sort_keys=True)
    except ValueError:
        return raw


def _output(payload: dict[str, Any]) -> str:
    out = payload.get("output")
    if isinstance(out, str):
        return out
    return "\n".join(text for _, text in _texts(out))


def _item(n: int, obj: dict[str, Any], payload: dict[str, Any]) -> SessionRecord:
    ptype = str(payload.get("type"))
    kind = f"response_item:{ptype}"
    if ptype == "message":
        role = payload.get("role")
        texts = _texts(payload.get("content"))
        blocks: list[Block] = [b for i, text in texts for b in text_blocks(text, i)]
        if role == "assistant":
            return SessionRecord(n, obj, f"{kind}:assistant", "message", role="assistant", blocks=blocks)
        if role == "user" and texts and texts[0][1].startswith(MARKER_PREFIX):
            return SessionRecord(n, obj, "response_item:loosy-marker", "side")
        if role in ("user", "developer"):
            return SessionRecord(
                n, obj, f"{kind}:{role}", "meta",
                role="user" if role == "user" else "system", blocks=blocks, pinned=True,
            )
        return SessionRecord(n, obj, f"{kind}:{role}", "unknown")
    if ptype == "reasoning":
        summary = _texts(payload.get("summary"))
        return SessionRecord(
            n, obj, kind, "message", role="assistant",
            blocks=[b for i, text in summary for b in tagged("thinking", text, i)], thinking=True,
        )
    if ptype in _CALLS:
        name = str(payload.get("name", ""))
        body = _arguments(payload.get("arguments", payload.get("input", "")))
        return SessionRecord(
            n, obj, kind, "message", role="assistant",
            blocks=tagged("tool_use", body, 0, name=name),
            call_ids=(str(payload.get("call_id", "")),), call_side="call", tool_name=name,
        )
    if ptype in _OUTPUTS:
        return SessionRecord(
            n, obj, kind, "message", role="tool",
            blocks=tagged("tool_result", _output(payload), 0),
            call_ids=(str(payload.get("call_id", "")),), call_side="result",
        )
    if ptype in _META_ITEMS:
        return SessionRecord(n, obj, kind, "meta")
    return SessionRecord(n, obj, kind, "unknown")


def _classify(n: int, obj: dict[str, Any]) -> SessionRecord:
    top = str(obj.get("type"))
    raw_payload = obj.get("payload")
    payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
    if top in ("session_meta", "compacted"):
        return SessionRecord(n, obj, top, "meta", pinned=True)
    if top == "response_item":
        return _item(n, obj, payload)
    if top == "event_msg":
        return SessionRecord(n, obj, f"event_msg:{payload.get('type')}", "meta")
    if top in _META_TOP:
        return SessionRecord(n, obj, top, "meta")
    return SessionRecord(n, obj, top, "unknown")


def _promote_prompts(records: list[SessionRecord], live: list[int]) -> None:
    pending: int | None = None
    for ri in live:
        rec = records[ri]
        if rec.kind == "response_item:message:user":
            pending = ri
        elif rec.kind == "event_msg:task_started":
            pending = None
        elif rec.kind == "event_msg:user_message" and pending is not None:
            prompt = records[pending]
            message = rec.raw.get("payload", {}).get("message")
            prompt.category, prompt.pinned = "message", False
            prompt.prompt = message if isinstance(message, str) else " ".join(b.text for b in prompt.blocks)
            pending = None


def _mark_exchanges(records: list[SessionRecord], live: list[int]) -> int | None:
    exchange = 0
    open_exchange: int | None = None
    for ri in live:
        rec = records[ri]
        if rec.kind == "event_msg:task_started":
            exchange += 1
            open_exchange = exchange
        rec.exchange = exchange
        if rec.role == "assistant":
            rec.reply = f"exchange-{exchange}"
        if rec.kind in _TURN_END:
            open_exchange = None
    return open_exchange


def read(path: Path) -> Session:
    lines, warnings = read_jsonl(path)
    records = [_classify(n, obj) for n, obj in lines]
    if not records or records[0].kind != "session_meta":
        raise SessionFormatError(f"{path}: the first line is not session_meta")
    meta = records[0].raw.get("payload", {})
    compactions = [i for i, r in enumerate(records) if r.kind == "compacted"]
    if compactions:
        last = compactions[-1]
        live = [i for i in range(last) if records[i].kind == "session_meta"]
        live += list(range(last, len(records)))
    else:
        live = list(range(len(records)))
    _promote_prompts(records, live)
    open_exchange = _mark_exchanges(records, live)
    version = meta.get("cli_version")
    return Session(
        tool=TOOL,
        path=path,
        session_id=str(meta.get("id", "")),
        cwd=str(meta.get("cwd", "")),
        cli_version=version if isinstance(version, str) else None,
        records=records,
        live=live,
        loose=[],
        open_exchange=open_exchange,
        warnings=warnings,
    )
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_sessions_codex.py -q && uv run mypy src && uv run ruff check . && uv run ruff format --check .`
Expected: 8 passed; mypy `Success` (run `uv run ruff format` on flagged files first).

- [ ] **Step 6: Commit**

```bash
git add src/loosy_goose/sessions/codex.py tests/session_fixtures.py tests/test_sessions_codex.py
git commit -m "$(cat <<'MSG'
Read Codex sessions into the session model

Locate by session_meta cwd, explicit id or path; history is file order from the last
compacted record; prompts are the user messages echoed by user_message events;
instructions and context messages are pinned; open turns are detected.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 9: Codex adapter — write the fork and validate it

**Files:**
- Modify: `src/loosy_goose/sessions/codex.py` (append; extend imports and `__all__`)
- Test: `tests/test_sessions_codex.py` (append)

**Interfaces:**
- Consumes: `ForkPlan`, `plan_fork`, `split_region`, `find_trigger`, `DEFAULT_TRIGGERS`, `MARKER_PREFIX`; `dump_line`, `read_jsonl`; `kept_where`.
- Produces (`loosy_goose.sessions.codex`):
  - `write_fork(session: Session, plan: ForkPlan, new_id: str, out_dir: Path) -> Path`: writes `out_dir/<new_id>.jsonl`; the final name comes from `fork_destination`.
  - `validate(fork: Path, session_id: str) -> list[str]`: from `"parse"`, `"session-id"`, `"forked-from"`, `"stale-id"`, `"tool-pairs"`, `"turns"`, `"ordinals"`.
  - `fork_destination(source: Path, new_id: str, home: Path, now: datetime) -> Path`: `home/sessions/YYYY/MM/DD/rollout-YYYY-MM-DDTHH-MM-SS-<id>.jsonl`.
  - `new_session_id() -> str`: UUIDv7.
  - `resume_command(session_id: str, cwd: str) -> str`: `cd <quoted cwd> && codex resume <id>`.
  - `verify_command(session_id: str) -> list[str]`: `["codex", "exec", "resume", id, "Reply with exactly: OK"]`.

**Writer rules:**

- Kept records are written in file order, with the old id replaced by the new one in every serialized line (the probe method).
- Line 0's `payload.forked_from_id` is set to the old id.
- `ordinal` is renumbered to the written line index on every line that carried one.
- The marker is a `response_item` user message inserted before the first kept `response_item` after the last kept `compacted` record. That puts it inside a kept turn, and after a compaction that would otherwise replace it.
- `plan.rewrites` replace message content texts, or the output text of a `*_output` (a str stays a str; a list keeps its non-text items).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sessions_codex.py`; extend the imports)

Add to the imports:

```python
import uuid
from collections.abc import Callable
from datetime import datetime

import pytest
from session_builders import kept_where

from loosy_goose.segment import Segment
from loosy_goose.sessions.plan import ForkPlan, plan_fork, split_region
```

Append:

```python
def _fork(
    tmp_path: Path,
    lines: list[dict[str, Any]] | None = None,
    keep: Callable[[Segment], bool] = lambda s: True,
) -> tuple[codex.Session, ForkPlan, str, Path]:
    source = (
        write_lines(tmp_path / "src.jsonl", lines) if lines is not None else FIXTURES / CODEX_FIXTURE
    )
    s = codex.read(source)
    region = split_region(s, find_trigger(s, DEFAULT_TRIGGERS["codex"]), protect_tail_tokens=0)
    plan = plan_fork(s, region, kept_where(region, keep), date="2026-10-13")
    new_id = codex.new_session_id()
    return s, plan, new_id, codex.write_fork(s, plan, new_id, tmp_path / "out")


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _is_marker(o: dict[str, Any]) -> bool:
    return o.get("type") == "response_item" and MARKER_PREFIX in json.dumps(o, ensure_ascii=False)


def test_a_fork_without_beta_validates(tmp_path: Path) -> None:
    s, _, new_id, fork = _fork(tmp_path, keep=lambda seg: "BETA" not in seg.text)
    assert codex.validate(fork, new_id) == []
    out = _lines(fork)
    assert out[0]["payload"]["id"] == new_id
    assert out[0]["payload"]["forked_from_id"] == s.session_id
    assert s.session_id not in json.dumps([o for o in out[1:] if not _is_marker(o)])
    assert f"compressed from session {s.session_id} " in json.dumps(out, ensure_ascii=False)
    f = codex.read(fork)
    text = " ".join(f.prompts())
    assert "BETA" not in text and SENTINEL not in text
    assert all(w in text for w in ("ALPHA", "GAMMA", "DELTA"))
    marker = next(i for i, o in enumerate(out) if _is_marker(o))
    assert all(o["type"] != "response_item" for o in out[:marker])


def test_ordinals_are_renumbered(tmp_path: Path) -> None:
    lines = [{**o, "ordinal": i} for i, o in enumerate(fixture_lines(CODEX_FIXTURE))]
    _, _, new_id, fork = _fork(tmp_path, lines, keep=lambda seg: "BETA" not in seg.text)
    out = _lines(fork)
    assert [o.get("ordinal") for o in out if "ordinal" in o] == [
        i for i, o in enumerate(out) if "ordinal" in o
    ]
    assert codex.validate(fork, new_id) == []


def _corrupt(lines: list[dict[str, Any]], how: str, old: str) -> list[dict[str, Any]]:
    if how == "tool-pairs":
        out = next(
            i for i, o in enumerate(lines)
            if o["type"] == "response_item" and o["payload"].get("type") in (
                "function_call_output", "custom_tool_call_output")
        )
        del lines[out]
    elif how == "stale-id":
        lines[-1] = {**lines[-1], "note": old}
    elif how == "session-id":
        lines[0]["payload"]["id"] = "someone-else"
    elif how == "turns":
        # The second start: the first turn's end precedes every start, and an end with no
        # start before it is excused by design (a compaction can cut a turn's start).
        starts = [
            i for i, o in enumerate(lines)
            if o["type"] == "event_msg" and o["payload"].get("type") == "task_started"
        ]
        del lines[starts[1]]
    elif how == "ordinals":
        lines = [{**o, "ordinal": i} for i, o in enumerate(lines)]
        del lines[2]
    return lines


@pytest.mark.parametrize("how", ["tool-pairs", "stale-id", "session-id", "turns", "ordinals"])
def test_validate_names_the_failed_check(tmp_path: Path, how: str) -> None:
    s, _, new_id, fork = _fork(tmp_path)
    bad = write_lines(tmp_path / "bad.jsonl", _corrupt(_lines(fork), how, s.session_id))
    assert how in codex.validate(bad, new_id)


def _compacted_lines() -> list[dict[str, Any]]:
    """The fixture as if Codex had compacted its context just before the GAMMA turn."""
    lines = fixture_lines(CODEX_FIXTURE)
    gamma = next(
        i for i, o in enumerate(lines)
        if o["type"] == "event_msg" and o["payload"].get("type") == "user_message"
        and "GAMMA" in str(o["payload"].get("message"))
    )
    start = max(
        i for i in range(gamma)
        if lines[i]["type"] == "event_msg" and lines[i]["payload"].get("type") == "task_started"
    )
    compacted = {
        "timestamp": lines[start]["timestamp"],
        "type": "compacted",
        "payload": {
            "message": "",
            "replacement_history": [{
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "Earlier: two code words were shared."}],
            }],
        },
    }
    return lines[:start] + [compacted] + lines[start:]


def test_rf4_a_session_spanning_context_windows_forks_from_its_last_compaction(
    tmp_path: Path,
) -> None:
    lines = _compacted_lines()
    s = codex.read(write_lines(tmp_path / "compacted.jsonl", lines))
    first_live = s.records[s.live[1]]
    assert first_live.kind == "compacted" and first_live.pinned
    assert len(s.prompts()) == 3 and not any("ALPHA" in p or "BETA" in p for p in s.prompts())
    _, _, new_id, fork = _fork(tmp_path, lines)
    assert codex.validate(fork, new_id) == []
    out = _lines(fork)
    compacted = next(i for i, o in enumerate(out) if o["type"] == "compacted")
    marker = next(i for i, o in enumerate(out) if _is_marker(o))
    assert compacted < marker
    assert all(o["type"] != "response_item" for o in out[compacted + 1 : marker])
    assert "ALPHA" not in json.dumps(out) and "BETA" not in json.dumps(out)


def test_destination_ids_and_commands() -> None:
    when = datetime(2026, 10, 13, 9, 8, 7)
    assert codex.fork_destination(Path("/x.jsonl"), "abc", Path("/h"), when) == Path(
        "/h/sessions/2026/10/13/rollout-2026-10-13T09-08-07-abc.jsonl"
    )
    assert uuid.UUID(codex.new_session_id()).version == 7
    assert codex.resume_command("abc", "/work/my dir") == "cd '/work/my dir' && codex resume abc"
    assert codex.verify_command("abc") == ["codex", "exec", "resume", "abc", "Reply with exactly: OK"]
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_sessions_codex.py -q`
Expected: the new tests fail with `AttributeError: module 'loosy_goose.sessions.codex' has no attribute 'new_session_id'`.

- [ ] **Step 3: Implement** (append to `codex.py`)

Extend the imports to add `copy`, `shlex`, `time`, `uuid`, `from datetime import datetime`, `dump_line` from `loosy_goose.sessions.model`, and `ForkPlan` from `loosy_goose.sessions.plan`. Extend `__all__` with the six new names. Then append:

```python
def new_session_id() -> str:
    # Codex ids are UUIDv7 (time-ordered); Python 3.12 has no uuid.uuid7.
    ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (
        (ms & ((1 << 48) - 1)) << 80
        | 0x7 << 76
        | ((rand >> 68) & 0xFFF) << 64
        | 0b10 << 62
        | (rand & ((1 << 62) - 1))
    )
    return str(uuid.UUID(int=value))


def fork_destination(source: Path, new_id: str, home: Path, now: datetime) -> Path:
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    return home / "sessions" / now.strftime("%Y/%m/%d") / f"rollout-{stamp}-{new_id}.jsonl"


def resume_command(session_id: str, cwd: str) -> str:
    return f"cd {shlex.quote(cwd)} && codex resume {session_id}"


def verify_command(session_id: str) -> list[str]:
    return ["codex", "exec", "resume", session_id, "Reply with exactly: OK"]


def _apply_rewrites(obj: dict[str, Any], rewrite: dict[int, str]) -> None:
    if not rewrite:
        return
    payload = obj["payload"]
    ptype = payload.get("type")
    if ptype == "message":
        content: list[Any] = []
        for i, part in enumerate(payload.get("content") or []):
            if i not in rewrite:
                content.append(part)
            elif rewrite[i]:
                content.append({**part, "text": rewrite[i]})
        payload["content"] = content
    elif ptype in _OUTPUTS:
        text = rewrite.get(0, "")
        out = payload.get("output")
        if isinstance(out, list):
            others = [c for c in out if not (isinstance(c, dict) and c.get("type") == "input_text")]
            payload["output"] = [{"type": "input_text", "text": text}, *others]
        else:
            payload["output"] = text


def write_fork(session: Session, plan: ForkPlan, new_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = session.records
    keep = sorted(plan.keep)
    objs: list[dict[str, Any]] = []
    for ri in keep:
        obj = copy.deepcopy(records[ri].raw)
        _apply_rewrites(obj, plan.rewrites.get(ri, {}))
        objs.append(obj)
    # After the last compaction, or the compaction would replace it; inside a kept turn, as
    # every user message Codex writes is.
    last_compaction = max(
        (i for i, ri in enumerate(keep) if records[ri].kind == "compacted"), default=-1
    )
    at = next(
        (
            i
            for i, ri in enumerate(keep)
            if i > last_compaction and records[ri].kind.startswith("response_item:")
        ),
        len(keep),
    )
    anchor = objs[at - 1] if at > 0 else objs[0]
    marker: dict[str, Any] = {
        "timestamp": anchor.get("timestamp"),
        "type": "response_item",
        "payload": {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": plan.marker}],
        },
    }
    if "ordinal" in anchor:
        marker["ordinal"] = 0
    objs.insert(at, marker)
    old = session.session_id
    dest = out_dir / f"{new_id}.jsonl"
    with dest.open("wb") as fh:
        for index, obj in enumerate(objs):
            if "ordinal" in obj:
                obj["ordinal"] = index
            line = dump_line(obj).decode("utf-8", errors="surrogateescape")
            # The marker names the source session on purpose; every other mention is the id.
            if obj is not marker:
                line = line.replace(old, new_id)
            if index == 0:
                head = json.loads(line)
                head["payload"]["forked_from_id"] = old
                line = dump_line(head).decode("utf-8", errors="surrogateescape")
            fh.write(line.encode("utf-8", errors="surrogateescape"))
    return dest


def validate(fork: Path, session_id: str) -> list[str]:
    lines, problems = read_jsonl(fork)
    failures = ["parse"] if problems else []
    objs = [obj for _, obj in lines]
    head = objs[0].get("payload", {}) if objs and objs[0].get("type") == "session_meta" else {}
    if head.get("id") != session_id:
        failures.append("session-id")
    old = head.get("forked_from_id")
    if not isinstance(old, str) or not old:
        failures.append("forked-from")
    else:
        dumped = [json.dumps(o, ensure_ascii=False) for o in objs[1:]]
        if any(old in d for d in dumped if MARKER_PREFIX not in d):
            failures.append("stale-id")
    calls: set[str] = set()
    outputs: set[str] = set()
    started: set[str] = set()
    any_started = False
    broken_turn = False
    for obj in objs:
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        ptype = payload.get("type")
        if obj.get("type") == "response_item" and ptype in _CALLS:
            calls.add(str(payload.get("call_id")))
        if obj.get("type") == "response_item" and ptype in _OUTPUTS:
            outputs.add(str(payload.get("call_id")))
        if obj.get("type") == "event_msg" and ptype == "task_started":
            any_started = True
            started.add(str(payload.get("turn_id")))
        # A turn's end without its start breaks the turn; an end before any start is the tail
        # of a turn that a compaction cut, which Codex itself writes.
        if (
            obj.get("type") == "event_msg"
            and ptype in ("task_complete", "turn_aborted")
            and any_started
            and str(payload.get("turn_id")) not in started
        ):
            broken_turn = True
    if calls != outputs:
        failures.append("tool-pairs")
    if broken_turn:
        failures.append("turns")
    numbered = [(i, o["ordinal"]) for i, o in enumerate(objs) if "ordinal" in o]
    if any(i != ordinal for i, ordinal in numbered):
        failures.append("ordinals")
    return failures
```

- [ ] **Step 4: Run the tests and see them pass**

Run: `uv run pytest tests/test_sessions_codex.py -q && uv run mypy src && uv run ruff check . && uv run ruff format --check .`
Expected: 17 passed; mypy `Success` (run `uv run ruff format` on flagged files first).

- [ ] **Step 5: Commit **

```bash
git add src/loosy_goose/sessions/codex.py tests/test_sessions_codex.py
git commit -m "$(cat <<'MSG'
Write and validate Codex forks

New UUIDv7 id replacing the old one everywhere, forked_from_id set, ordinals renumbered,
marker after the last kept compaction inside a kept turn. Validation checks one id, call
and output pairing, turn starts and ordinals.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 10: `handoff.py` — locate, snapshot, compress, write, validate, verify

**Files:**
- Create: `src/loosy_goose/sessions/adapter.py`
- Create: `src/loosy_goose/handoff.py`
- Modify: `tests/session_fixtures.py` (add `fake_embeddings`, `without_trigger`)
- Test: `tests/test_handoff.py`

**Interfaces:**
- Consumes:
  - both adapters (Tasks 6–9);
  - `snapshot`, `newer_than_tested`, `SessionFormatError`, `Session`, `Tool` (Task 3);
  - `find_trigger`, `split_region`, `plan_fork`, `DEFAULT_TRIGGERS`, `PROTECT_TAIL_TOKENS`, `HandoffRefusedError`, `Trigger` (Tasks 3–4);
  - `resolve_rate` (Task 1);
  - `pipeline.PipelineOptions`, `pipeline.compress_transcript`; `segment.segment`; `tokens.count_tokens`.
- Produces:
  - `loosy_goose.sessions.adapter.SessionAdapter` (Protocol, which the `claude` and `codex` modules satisfy):
    - `TOOL: Tool`, `TESTED_VERSIONS: tuple[str, ...]`
    - `default_home() -> Path`
    - `locate(cwd: Path, session: str, home: Path) -> list[Path]`
    - `read(path: Path) -> Session`
    - `write_fork(session: Session, plan: ForkPlan, new_id: str, out_dir: Path) -> Path`
    - `validate(fork: Path, session_id: str) -> list[str]`
    - `fork_destination(source: Path, new_id: str, home: Path, now: datetime) -> Path`
    - `new_session_id() -> str`
    - `resume_command(session_id: str, cwd: str) -> str`
    - `verify_command(session_id: str) -> list[str]`
  - `loosy_goose.handoff`:
    - `ADAPTERS: dict[Tool, SessionAdapter]`, `DEFAULT_KEEP_THINKING = False`
    - `class HandoffError(Exception)`
    - `Runner = Callable[[list[str], Path], str]`
    - `@dataclass(frozen=True) class HandoffOptions(level: str | None = None, keep: float | None = None, target_tokens: int | None = None, session: str = "current", tool: Literal["auto", "claude", "codex"] = "auto", cwd: Path = Path.cwd(), verify: bool = False, protect_tail_tokens: int = 4000, keep_thinking: bool = DEFAULT_KEEP_THINKING, triggers: tuple[str, ...] | None = None, homes: dict[Tool, Path] | None = None)`. `cwd` is a `default_factory`.
    - `@dataclass(frozen=True) class HandoffResult(status: Literal["forked", "nothing-to-do"], reason: str | None, tool: Tool, source_session: str, source_path: str, original_resume_command: str, fork_session: str | None, fork_path: str | None, resume_command: str | None, before_tokens: int, after_tokens: int, tail_tokens: int, keep: float | None, first_prompt: str, last_prompt: str, trigger: str, verified: bool | None, warnings: list[str])` with `to_dict() -> dict[str, Any]`
    - `handoff(opts: HandoffOptions, *, now: datetime | None = None, runner: Runner | None = None, cache_dir: Path | None = None) -> HandoffResult`
  - `session_fixtures`: `fake_embeddings(texts: list[str], model_name: str) -> np.ndarray`, `without_trigger(lines: list[dict[str, Any]]) -> list[dict[str, Any]]`

**Pipeline:**

1. Candidates come from the selected adapters' `locate`, newest first by mtime.
2. For `session="current"`, read up to the 3 newest and take the first carrying a trigger. Otherwise take the newest, and warn that the handoff trigger was not found ([RF3], [RF5]). If more than one candidate changed in the last 10 minutes, warn ([RF3]).
3. Every read is of a snapshot in a private temp dir ([RF2]).
4. Split the region:
   - If no compressible records are left, the result is "nothing-to-do".
   - If `target_tokens` is at least the session's tokens, the result is "nothing-to-do".
   - Otherwise the target for the compressible part is `target_tokens` minus everything else, and an error if that leaves nothing.
5. Compress with `band=False, paths=False`, plan the fork, and write it to staging.
6. Validate. On failure, delete the fork and raise `HandoffError` naming the checks.
7. Count the fork's tokens by re-reading it. Move it to its destination.
8. With `verify`, write a throwaway copy under a third id, run the CLI's `verify_command`, and delete the copy. If the run fails or doesn't answer OK, delete the fork and raise.

Token counts are block-level `count_tokens` over the live history: `before` covers everything before the trigger, `after` covers the whole fork.

- [ ] **Step 1: Add test helpers** (append to `tests/session_fixtures.py`; add `"fake_embeddings"` and `"without_trigger"` to `__all__`; add `import numpy as np` to the imports)

```python
_EMBED_DIM = 3


def fake_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    # As tests/test_cli.py: a small dimension keeps leverage scores well separated.
    rng = np.random.default_rng(11)
    x = rng.normal(size=(len(texts), _EMBED_DIM))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def without_trigger(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The Claude fixture cut just before its /loosy turn."""
    cut = next(
        i
        for i, o in enumerate(lines)
        if o.get("type") == "user" and "<command-name>/loosy</command-name>" in json.dumps(o)
    )
    return lines[:cut]
```

- [ ] **Step 2: Write the failing tests** (`tests/test_handoff.py`)

```python
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from session_fixtures import (
    CLAUDE_FIXTURE,
    FIXTURE_CWD,
    fake_embeddings,
    fixture_lines,
    install_claude,
    install_codex,
    without_trigger,
)

from loosy_goose.handoff import HandoffError, HandoffOptions, handoff
from loosy_goose.sessions import claude, codex

NOW = datetime(2026, 10, 13, 12, 0, 0)


@pytest.fixture(autouse=True)
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embeddings)


def _opts(tmp_path: Path, **overrides: Any) -> HandoffOptions:
    values: dict[str, Any] = {
        "tool": "claude",
        "cwd": Path(FIXTURE_CWD),
        "protect_tail_tokens": 0,
        "level": "heavy",
        "homes": {"claude": tmp_path / "claude", "codex": tmp_path / "codex"},
    }
    values.update(overrides)
    return HandoffOptions(**values)


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def test_forks_the_claude_fixture(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    result = handoff(_opts(tmp_path), now=NOW)
    assert (result.status, result.tool, result.trigger) == ("forked", "claude", "text")
    assert result.fork_session is not None and result.fork_path is not None
    fork = Path(result.fork_path)
    assert fork == source.parent / f"{result.fork_session}.jsonl"
    assert claude.validate(fork, result.fork_session) == []
    assert result.resume_command == f"cd {FIXTURE_CWD} && claude --resume {result.fork_session}"
    assert result.original_resume_command == (
        f"cd {FIXTURE_CWD} && claude --resume {result.source_session}"
    )
    assert "ALPHA" in result.first_prompt and "DELTA" in result.last_prompt
    assert min(result.before_tokens, result.after_tokens, result.tail_tokens) > 0
    assert result.keep == 0.25 and result.verified is None
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before
    assert _names(source.parent) == sorted([source.name, fork.name])
    assert json.loads(json.dumps(result.to_dict()))["fork_session"] == result.fork_session


def test_forks_the_codex_fixture(tmp_path: Path) -> None:
    source = install_codex(tmp_path / "codex")
    result = handoff(_opts(tmp_path, tool="codex"), now=NOW)
    assert (result.status, result.tool) == ("forked", "codex")
    assert result.fork_path == str(
        tmp_path / "codex" / "sessions" / "2026" / "10" / "13"
        / f"rollout-2026-10-13T12-00-00-{result.fork_session}.jsonl"
    )
    assert codex.validate(Path(result.fork_path), str(result.fork_session)) == []
    assert result.resume_command == f"cd {FIXTURE_CWD} && codex resume {result.fork_session}"
    assert source.exists()


def test_auto_picks_the_newest_session_across_tools(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude")
    old = install_codex(tmp_path / "codex")
    os.utime(old, (1, 1))
    assert handoff(_opts(tmp_path, tool="auto"), now=NOW).tool == "claude"


def test_nothing_to_do_when_the_session_is_all_tail(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")
    result = handoff(_opts(tmp_path, protect_tail_tokens=4000), now=NOW)
    assert result.status == "nothing-to-do" and result.fork_path is None
    assert result.reason is not None and "protected tail" in result.reason
    assert _names(source.parent) == [source.name]


def test_nothing_to_do_when_already_under_target(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude")
    result = handoff(_opts(tmp_path, level=None, target_tokens=10**6), now=NOW)
    assert result.status == "nothing-to-do"
    assert result.reason is not None and "under target" in result.reason


def test_a_target_below_what_must_stay_is_an_error(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude")
    with pytest.raises(HandoffError, match="leaves no room"):
        handoff(_opts(tmp_path, level=None, target_tokens=1), now=NOW)


def test_no_session_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(HandoffError, match="no Claude Code session found"):
        handoff(_opts(tmp_path), now=NOW)


def test_a_failed_validation_deletes_the_fork_and_names_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = install_claude(tmp_path / "claude")
    monkeypatch.setattr(claude, "validate", lambda fork, sid: ["chain"])
    with pytest.raises(HandoffError, match=r"validation \(chain\)"):
        handoff(_opts(tmp_path), now=NOW)
    assert _names(source.parent) == [source.name]


def test_an_unknown_record_inside_the_compressed_stretch_refuses(tmp_path: Path) -> None:
    lines = fixture_lines(CLAUDE_FIXTURE)
    alpha = next(
        i for i, o in enumerate(lines)
        if o.get("type") == "user" and "ALPHA" in json.dumps(o.get("message"))
    )
    alpha_id = lines[alpha]["uuid"]
    child = next(o for o in lines if o.get("parentUuid") == alpha_id)
    stranger = {
        "type": "brand-new-thing", "uuid": str(uuid.uuid4()), "parentUuid": alpha_id,
        "sessionId": lines[alpha]["sessionId"],
    }
    child["parentUuid"] = stranger["uuid"]
    source = install_claude(tmp_path / "claude", lines=[*lines[: alpha + 1], stranger, *lines[alpha + 1 :]])
    with pytest.raises(HandoffError, match="brand-new-thing"):
        handoff(_opts(tmp_path), now=NOW)
    assert _names(source.parent) == [source.name]


def test_a_newer_cli_than_tested_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_claude(tmp_path / "claude")
    monkeypatch.setattr(claude, "TESTED_VERSIONS", ("0.0.1",))
    result = handoff(_opts(tmp_path), now=NOW)
    assert any("newer than the tested 0.0.1" in w for w in result.warnings)


def test_verify_runs_a_throwaway_copy_and_deletes_it(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")
    calls: list[list[str]] = []

    def runner(argv: list[str], cwd: Path) -> str:
        calls.append(argv)
        assert (source.parent / f"{argv[-1]}.jsonl").exists()
        return "OK\n"

    result = handoff(_opts(tmp_path, verify=True), now=NOW, runner=runner)
    assert result.verified is True
    assert calls[0][:4] == ["claude", "-p", "Reply with exactly: OK", "--resume"]
    assert calls[0][-1] != result.fork_session
    assert _names(source.parent) == sorted([source.name, f"{result.fork_session}.jsonl"])


def test_a_failed_live_check_deletes_the_fork(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")
    with pytest.raises(HandoffError, match="live check failed"):
        handoff(_opts(tmp_path, verify=True), now=NOW, runner=lambda argv, cwd: "nope")
    assert _names(source.parent) == [source.name]


def test_rf2_a_partial_last_line_is_ignored_and_the_original_untouched(tmp_path: Path) -> None:
    source = install_claude(tmp_path / "claude")
    with source.open("ab") as fh:
        fh.write(b'{"type":"user","uuid":"half')
    before = (source.read_bytes(), source.stat().st_mtime_ns)
    result = handoff(_opts(tmp_path), now=NOW)
    assert result.status == "forked"
    assert any("partial last line (27 bytes)" in w for w in result.warnings)
    assert (source.read_bytes(), source.stat().st_mtime_ns) == before


def test_rf3_the_session_carrying_the_trigger_beats_a_newer_one(tmp_path: Path) -> None:
    home = tmp_path / "claude"
    with_trigger = install_claude(home)
    install_claude(
        home,
        lines=without_trigger(fixture_lines(CLAUDE_FIXTURE)),
        session_id="44444444-4444-4444-8444-444444444444",
    )
    os.utime(with_trigger, (time.time() - 30, time.time() - 30))
    result = handoff(_opts(tmp_path), now=NOW)
    assert result.source_path == str(with_trigger)
    assert result.trigger == "text"
    assert any("2 sessions in" in w for w in result.warnings)
    assert "ALPHA" in result.first_prompt and "DELTA" in result.last_prompt


def test_rf5_an_unidentifiable_trigger_excludes_nothing_and_says_so(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude", lines=without_trigger(fixture_lines(CLAUDE_FIXTURE)))
    result = handoff(_opts(tmp_path), now=NOW)
    assert (result.status, result.trigger) == ("forked", "none")
    assert any("handoff trigger not found" in w for w in result.warnings)
    fork = claude.read(Path(str(result.fork_path)))
    assert "DELTA" in fork.prompts()[-1]
```

`len(b'{"type":"user","uuid":"half')` is 27.

- [ ] **Step 3: Run them and see them fail**

Run: `uv run pytest tests/test_handoff.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'loosy_goose.handoff'`.

- [ ] **Step 4: Implement `src/loosy_goose/sessions/adapter.py`**

```python
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

from loosy_goose.sessions.model import Session, Tool
from loosy_goose.sessions.plan import ForkPlan


class SessionAdapter(Protocol):
    """What `handoff` needs from a CLI's session format; `sessions.claude` and `sessions.codex`
    satisfy it as modules."""

    TOOL: Tool
    TESTED_VERSIONS: tuple[str, ...]

    def default_home(self) -> Path: ...
    def locate(self, cwd: Path, session: str, home: Path) -> list[Path]: ...
    def read(self, path: Path) -> Session: ...
    def write_fork(self, session: Session, plan: ForkPlan, new_id: str, out_dir: Path) -> Path: ...
    def validate(self, fork: Path, session_id: str) -> list[str]: ...
    def fork_destination(self, source: Path, new_id: str, home: Path, now: datetime) -> Path: ...
    def new_session_id(self) -> str: ...
    def resume_command(self, session_id: str, cwd: str) -> str: ...
    def verify_command(self, session_id: str) -> list[str]: ...
```

- [ ] **Step 5: Implement `src/loosy_goose/handoff.py`**

```python
"""Fork the live Claude Code or Codex session into a compressed copy the user resumes.

The original session file is only ever read, through a snapshot. The only files this module
deletes are forks of its own: one that failed validation, the live-check copy, and a fork whose
live check failed.
"""

from __future__ import annotations

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
from loosy_goose.sessions.model import Session, SessionFormatError, Tool, newer_than_tested, snapshot
from loosy_goose.sessions.plan import (
    DEFAULT_TRIGGERS,
    PROTECT_TAIL_TOKENS,
    ForkPlan,
    HandoffRefusedError,
    Region,
    Trigger,
    find_trigger,
    plan_fork,
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

Runner = Callable[[list[str], Path], str]


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
    triggers: tuple[str, ...] | None = None
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
        trigger = find_trigger(session, opts.triggers or DEFAULT_TRIGGERS[adapter.TOOL])
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


def _run(argv: list[str], cwd: Path) -> str:
    try:
        done = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True, timeout=VERIFY_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HandoffError(f"the live check could not run `{argv[0]}`: {exc}") from exc
    return done.stdout


def _verify(
    p: _Picked, plan: ForkPlan, home: Path, stamp: datetime, cwd: Path, work: Path, runner: Runner
) -> bool:
    check_id = p.adapter.new_session_id()
    staged = p.adapter.write_fork(p.session, plan, check_id, work / "verify")
    copy = p.adapter.fork_destination(p.source, check_id, home, stamp)
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(staged, copy)
    try:
        return "OK" in runner(p.adapter.verify_command(check_id), cwd)
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
    history = session.live[: region.end]
    prompts = [r.prompt for ri in history if (r := session.records[ri]).prompt is not None]
    before = sum(session.tokens(ri) for ri in history)
    tail = sum(session.tokens(ri) for ri in session.live[region.tail_from : region.end])
    compressible = sum(session.tokens(ri) for ri in region.turn_record)

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
    staged = adapter.write_fork(session, plan, fork_id, work / "fork")
    failures = adapter.validate(staged, fork_id)
    if failures:
        staged.unlink()
        raise HandoffError(
            f"the fork failed validation ({', '.join(failures)}) and was deleted; the original "
            "session is unchanged"
        )
    after = _total(adapter.read(staged))
    dest = adapter.fork_destination(p.source, fork_id, homes[adapter.TOOL], stamp)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(staged, dest)
    verified: bool | None = None
    if opts.verify:
        try:
            verified = _verify(p, plan, homes[adapter.TOOL], stamp, opts.cwd, work, runner)
        except HandoffError as exc:
            dest.unlink()
            raise HandoffError(f"{exc}; the fork was deleted and the original is unchanged") from exc
        if not verified:
            dest.unlink()
            raise HandoffError(
                "live check failed: the resumed throwaway copy did not answer OK; the fork was "
                "deleted and the original session is unchanged"
            )
    return result(
        "forked", fork_id=fork_id, fork_path=dest, after=after, keep=rate.keep, verified=verified
    )
```

- [ ] **Step 6: Run the tests and see them pass**

Run: `uv run pytest tests/test_handoff.py -q && uv run mypy src && uv run ruff check . && uv run ruff format --check .`
Expected: 15 passed; mypy `Success`. That includes the module-as-protocol assignment `ADAPTERS: dict[Tool, SessionAdapter] = {"claude": claude, "codex": codex}`, which mypy supports. If mypy rejects a module for a protocol mismatch, first check that every protocol member is listed in that adapter's `__all__`: with `__all__` defined, mypy matches a module against a protocol by its exported names only (checked while planning). Otherwise fix the adapter's signature, not the protocol. Run `uv run ruff format` on flagged files first.

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 8: Commit **

```bash
git add src/loosy_goose/sessions/adapter.py src/loosy_goose/handoff.py tests/session_fixtures.py tests/test_handoff.py
git commit -m "$(cat <<'MSG'
Add the session handoff: locate, snapshot, compress, fork, validate, verify

Prefers the newest session carrying a handoff trigger and warns about concurrent sessions;
reads only snapshots; compresses the part before the protected tail; validates the fork in
staging before moving it next to the original; optional headless live check on a
throwaway copy. Every result carries both resume commands, token counts and warnings.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 11: MCP server, `mcp` extra, console script, CI extra

**Files:**
- Create: `src/loosy_goose/mcp_server.py`
- Modify: `src/loosy_goose/embed.py:17-20` (add `warm` after `_model`)
- Modify: `pyproject.toml:20-27` (add the `mcp` extra and the `loosy-goose-mcp` script)
- Modify: `uv.lock` (via `uv lock`)
- Modify: `.github/workflows/ci.yml:22` (`uv sync --dev --extra embed --extra mcp`)
- Modify: `CHANGELOG.md` (append under `## Unreleased`)
- Test: `tests/test_mcp_server.py`, `tests/test_workflows.py`

**Interfaces:**
- Consumes:
  - `handoff`, `HandoffOptions`, `HandoffError` (Task 10); `PROTECT_TAIL_TOKENS` (Task 3);
  - `LEVELS`, `check_keep`, `resolve_rate` (Task 1);
  - `pipeline.compress_transcript`, `PipelineOptions`, `selector_label`, `to_turns`;
  - `transcript.load_claude_code_jsonl`, `load_messages_json`; `Tool` (Task 3).
- Produces:
  - `loosy_goose.embed.warm(model_name: str = SELECT_MODEL) -> None`
  - `loosy_goose.mcp_server`:
    - `INSTALL_HINT: str`
    - `preload_model() -> None`: raises `SystemExit` naming the install command when `sentence_transformers` is missing.
    - `build_server(*, homes: Mapping[Tool, Path] | None = None) -> MCPServer`
    - `main() -> None`
  - MCP tool `handoff_session(level: str | None = None, keep: float | None = None, target_tokens: int | None = None, session: str = "current", tool: Literal["auto", "claude", "codex"] = "auto", cwd: str | None = None, verify: bool = False, protect_tail_tokens: int = 4000) -> dict`: `HandoffResult.to_dict()`.
  - MCP tool `compress(messages: list[dict] | None = None, path: str | None = None, level: str | None = None, keep: float | None = None, target_tokens: int | None = None) -> {"header": {...}, "turns": [{"index", "role", "blocks": [{"kind", "text", "meta"}]}]}`: writes nothing.

**Decision (user, 2026-10-13):** the official `mcp` 2.x SDK. Errors meant for the user are raised as `ToolError`, because any other exception's message is hidden from the client (verified with 2.3.0).

- [ ] **Step 1: Add the extra and script, lock, sync**

In `pyproject.toml`, add under `[project.optional-dependencies]` after `embed`:

```toml
mcp = [
    "mcp>=2.3,<3",
]
```

and under `[project.scripts]`:

```toml
loosy-goose-mcp = "loosy_goose.mcp_server:main"
```

Run: `uv lock && git diff --unified=0 uv.lock | grep -E '^-version' ; uv sync --dev --extra embed --extra mcp`
Expected: no `-version` lines, meaning no existing package changed version. The lock gains `mcp` and its dependencies. If a `-version` line appears, run `git checkout uv.lock` and retry with `uv lock --upgrade-package mcp`, then re-check.

- [ ] **Step 2: Write the failing tests** (`tests/test_mcp_server.py`)

```python
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp import Client  # noqa: E402
from session_fixtures import FIXTURE_CWD, fake_embeddings, install_claude  # noqa: E402

from loosy_goose import mcp_server  # noqa: E402

_CONTENTS = [
    "Please look at the failing test in tests/test_main.py and tell me what you see.",
    "The assertion compares floats exactly; a tolerance is needed around line 42.",
    "Add the tolerance and rerun the suite for me.",
    "Done, the suite is green again with pytest.approx in place.",
    "Now also update src/app/main.py so compute() returns a float.",
    "Updated compute() in src/app/main.py; it now returns float(total) / 3.",
    "Great, thanks. Summarise what changed.",
    "Two edits: tolerance in tests/test_main.py and a float return in src/app/main.py.",
]
MESSAGES = [
    {"role": "user" if i % 2 == 0 else "assistant", "content": c} for i, c in enumerate(_CONTENTS)
]


@pytest.fixture(autouse=True)
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loosy_goose.select.embed_texts", fake_embeddings)


def _call(server: Any, name: str, args: dict[str, Any]) -> Any:
    async def go() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, args)

    return anyio.run(go)


def _text(result: Any) -> str:
    return " ".join(getattr(c, "text", "") for c in result.content)


def _homes(tmp_path: Path) -> dict[str, Path]:
    return {"claude": tmp_path / "claude", "codex": tmp_path / "codex"}


def test_serves_both_tools() -> None:
    async def go() -> set[str]:
        async with Client(mcp_server.build_server()) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert anyio.run(go) == {"compress", "handoff_session"}


def test_compress_returns_kept_turns_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = _call(mcp_server.build_server(), "compress", {"messages": MESSAGES, "level": "light"})
    assert not result.is_error, _text(result)
    data = result.structured_content
    assert (data["header"]["keep"], data["header"]["keep_source"]) == (0.5, "level")
    assert data["turns"] and all(t["blocks"] for t in data["turns"])
    assert list(tmp_path.iterdir()) == []


def test_compress_reads_a_transcript_path(tmp_path: Path) -> None:
    path = tmp_path / "chat.json"
    path.write_text(json.dumps(MESSAGES), encoding="utf-8")
    result = _call(mcp_server.build_server(), "compress", {"path": str(path), "keep": 0.5})
    assert not result.is_error, _text(result)
    assert result.structured_content["header"]["keep_source"] == "keep"


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"messages": MESSAGES, "keep": 0.5, "level": "light"}, "only one of"),
        ({"messages": MESSAGES, "level": "huge"}, "level must be one of"),
        ({"messages": MESSAGES, "keep": 0.9}, "keep must be in"),
        ({"level": "light"}, "exactly one of messages or path"),
    ],
)
def test_compress_rejects_bad_arguments(args: dict[str, Any], message: str) -> None:
    result = _call(mcp_server.build_server(), "compress", args)
    assert result.is_error and message in _text(result)


def test_handoff_session_forks_the_fixture(tmp_path: Path) -> None:
    install_claude(tmp_path / "claude")
    server = mcp_server.build_server(homes=_homes(tmp_path))
    result = _call(
        server,
        "handoff_session",
        {"tool": "claude", "cwd": FIXTURE_CWD, "level": "heavy", "protect_tail_tokens": 0},
    )
    assert not result.is_error, _text(result)
    data = result.structured_content
    assert data["status"] == "forked" and Path(data["fork_path"]).exists()
    assert data["resume_command"].endswith(data["fork_session"])
    assert data["original_resume_command"].endswith(data["source_session"])


def test_handoff_session_reports_errors_as_tool_errors(tmp_path: Path) -> None:
    server = mcp_server.build_server(homes=_homes(tmp_path))
    result = _call(server, "handoff_session", {"tool": "claude", "cwd": FIXTURE_CWD})
    assert result.is_error and "no Claude Code session found" in _text(result)


def test_preload_exits_with_the_install_command_without_embed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_server, "find_spec", lambda name: None)
    with pytest.raises(SystemExit, match=r"embed.*uv tool install"):
        mcp_server.preload_model()
```

`tests/test_workflows.py`:

```python
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def test_ci_installs_the_mcp_extra() -> None:
    assert "uv sync --dev --extra embed --extra mcp" in (WORKFLOWS / "ci.yml").read_text("utf-8")
```

- [ ] **Step 3: Run them and see them fail**

Run: `uv run pytest tests/test_mcp_server.py tests/test_workflows.py -q`
Expected: `ModuleNotFoundError: No module named 'loosy_goose.mcp_server'` and `test_ci_installs_the_mcp_extra` FAILED.

- [ ] **Step 4: Implement**

`src/loosy_goose/embed.py`, after `_model`:

```python
def warm(model_name: str = SELECT_MODEL) -> None:
    _model(model_name)
```

`.github/workflows/ci.yml` line 22: `- run: uv sync --dev --extra embed --extra mcp`.

`src/loosy_goose/mcp_server.py`:

```python
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
from loosy_goose.sessions.plan import PROTECT_TAIL_TOKENS
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
        transcript = _transcript(messages, path)
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
```

Append to `CHANGELOG.md` under `## Unreleased`:

```markdown
- 2026-10-13 — `loosy-goose-mcp`, an MCP server (new `mcp` extra, official `mcp` SDK 2.x) with two tools. `handoff_session` forks the current Claude Code or Codex session into a compressed copy whose history is the kept turns plus a verbatim recent tail (at least 4,000 tokens) behind a marker, validates the fork the way each CLI loads it, and returns the resume command for the fork and for the untouched original. `compress` returns compressed turns and writes nothing. Session formats live in `sessions/claude.py` and `sessions/codex.py`, tested against Claude Code 2.1.296 and codex-cli 0.158.0; a newer CLI still forks, with a warning. Rate presets moved from `cli.py` to `loosy_goose.rate` (CLI unchanged); segments now record their source block (`Segment.block`, not part of `segments_digest`).
```

- [ ] **Step 5: Run the tests and see them pass**

Run: `uv run pytest tests/test_mcp_server.py tests/test_workflows.py -q && uv run mypy src && uv run ruff check . && uv run ruff format --check .`
Expected: 11 passed; mypy `Success`.

- [ ] **Step 6: Smoke-test that the server builds outside pytest**

Run: `uv run python -c "from loosy_goose.mcp_server import build_server; print(type(build_server()).__name__)"`
Expected: `MCPServer`.

- [ ] **Step 7: Run the whole suite**

Run: `uv run pytest -q`
Expected: all green.

- [ ] **Step 8: Commit **

```bash
git add pyproject.toml uv.lock src/loosy_goose/embed.py src/loosy_goose/mcp_server.py .github/workflows/ci.yml CHANGELOG.md tests/test_mcp_server.py tests/test_workflows.py
git commit -m "$(cat <<'MSG'
Add the loosy-goose-mcp server with handoff_session and compress

New `mcp` extra (official SDK 2.x) and console script; the selection model loads at
startup, and a missing `embed` extra fails at startup with the install command. CI
installs the extra.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 12: Integrations (`/loosy`, `$loosy`) and docs

**Files:**
- Create: `integrations/README.md`
- Create: `integrations/claude/loosy.md`
- Create: `integrations/codex/loosy/SKILL.md`
- Create: `integrations/codex/loosy/agents/openai.yaml`
- Modify: `README.md:104-110` (Documentation list), after `:157` (new "Handoff" section before "Continuing on another machine"), `:188-194` (Test), `:196-205` (Roadmap)
- Modify: `CLAUDE.md:9-20` (Commands), `:27-37` (Layout, Core), `:57-60` (Supporting), `:131-137` (Hard rules)
- Modify: `CHANGELOG.md` (append under `## Unreleased`)
- Test: `tests/test_integrations.py`

**Interfaces:**
- Consumes: `HandoffResult` field names and `HandoffOptions` field names (Task 10); `DEFAULT_TRIGGERS`, `HANDOFF_SENTINEL` (Task 3).
- Produces:
  - the Claude Code command `/loosy`, installed to `~/.claude/commands/loosy.md`;
  - the Codex skill `$loosy`, installed to `~/.agents/skills/loosy/`;
  - `integrations/README.md` with the install and registration commands that the release (Task 14) attaches.

**Why a Codex skill, not a custom prompt:** OpenAI's Codex docs mark custom prompts (`~/.codex/prompts`) deprecated and recommend skills, invoked explicitly as `$name` (checked 2026-10-13). `allow_implicit_invocation: false` keeps Codex from triggering a handoff on its own.

- [ ] **Step 1: Write the failing tests** (`tests/test_integrations.py`)

```python
import re
from pathlib import Path

from loosy_goose.handoff import HandoffOptions, HandoffResult
from loosy_goose.sessions.plan import DEFAULT_TRIGGERS, HANDOFF_SENTINEL

INTEGRATIONS = Path(__file__).resolve().parents[1] / "integrations"
RESULT_FIELDS = set(HandoffResult.__dataclass_fields__)
OPTION_FIELDS = set(HandoffOptions.__dataclass_fields__)


def _split(path: Path) -> tuple[dict[str, str], str]:
    _, head, body = path.read_text(encoding="utf-8").split("---\n", 2)
    meta = dict(line.split(": ", 1) for line in head.strip().splitlines())
    return meta, body


def _check_body(body: str) -> None:
    assert HANDOFF_SENTINEL in body
    assert set(re.findall(r"<([a-z_]+)>", body)) <= RESULT_FIELDS
    assert set(re.findall(r"^- `([a-z_]+)`:", body, re.MULTILINE)) <= OPTION_FIELDS


def test_claude_command_calls_the_handoff_tool() -> None:
    path = INTEGRATIONS / "claude" / "loosy.md"
    meta, body = _split(path)
    assert {"description", "argument-hint", "allowed-tools"} <= set(meta)
    assert meta["allowed-tools"] == "mcp__loosy-goose__handoff_session"
    assert "$ARGUMENTS" in body
    assert f"<command-name>/{path.stem}</command-name>" in DEFAULT_TRIGGERS["claude"]
    _check_body(body)


def test_codex_skill_calls_the_handoff_tool_and_is_explicit_only() -> None:
    folder = INTEGRATIONS / "codex" / "loosy"
    meta, body = _split(folder / "SKILL.md")
    assert meta["name"] == "loosy" and meta["description"]
    assert f"${meta['name']}" in DEFAULT_TRIGGERS["codex"]
    _check_body(body)
    policy = (folder / "agents" / "openai.yaml").read_text(encoding="utf-8")
    assert "allow_implicit_invocation: false" in policy


def test_readme_registers_the_server_with_both_clis() -> None:
    text = (INTEGRATIONS / "README.md").read_text(encoding="utf-8")
    assert "claude mcp add -s user loosy-goose -- loosy-goose-mcp" in text
    assert "codex mcp add loosy-goose -- loosy-goose-mcp" in text
    assert "--torch-backend cpu" in text
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_integrations.py -q`
Expected: 3 FAILED with `FileNotFoundError` for `integrations/...`.

- [ ] **Step 3: Write `integrations/claude/loosy.md`**

```markdown
---
description: Compress this session with loosy-goose and fork it into a new session to resume
argument-hint: "[light|medium|heavy] [key=value ...]"
allowed-tools: mcp__loosy-goose__handoff_session
---

loosy-goose:handoff

Call the `mcp__loosy-goose__handoff_session` tool exactly once, with these arguments:

- `tool`: "claude"
- `cwd`: the absolute path of the current working directory
- `level`: the first word of "$ARGUMENTS" when it is light, medium or heavy; otherwise leave it out (the server defaults to medium)

Pass every `key=value` pair in "$ARGUMENTS" as the argument of that name: numbers as numbers, true and false as booleans. For example `protect_tail_tokens=8000`, `target_tokens=60000`, `keep=0.4` or `verify=true`.

Do nothing else before or after the call: no other tools, no summary of the conversation.

If the tool returns an error, show its text and stop. If `status` is "nothing-to-do", show <reason> and stop. Otherwise reply with exactly this, filled in from the result:

Fork ready. Exit this session and run:
  <resume_command>
Back to the original: <original_resume_command>
Tokens: <before_tokens> -> <after_tokens> (verbatim tail <tail_tokens>)
Picked: first prompt "<first_prompt>", last prompt "<last_prompt>"
Warnings: <warnings>, one per line, or "none"
```

- [ ] **Step 4: Write the Codex skill**

`integrations/codex/loosy/SKILL.md`:

```markdown
---
name: loosy
description: Compress this Codex session with loosy-goose and fork it into a new session to resume. Use only when the user invokes $loosy.
---

loosy-goose:handoff

Call the `handoff_session` tool of the `loosy-goose` MCP server exactly once, with these arguments:

- `tool`: "codex"
- `cwd`: the absolute path of the current working directory
- `level`: the first word after `$loosy` in the user's message when it is light, medium or heavy; otherwise leave it out (the server defaults to medium)

Pass every `key=value` pair after `$loosy` as the argument of that name: numbers as numbers, true and false as booleans. For example `protect_tail_tokens=8000`, `target_tokens=60000`, `keep=0.4` or `verify=true`.

Do nothing else before or after the call: no other tools, no summary of the conversation.

If the tool returns an error, show its text and stop. If `status` is "nothing-to-do", show <reason> and stop. Otherwise reply with exactly this, filled in from the result:

Fork ready. Exit this session and run:
  <resume_command>
Back to the original: <original_resume_command>
Tokens: <before_tokens> -> <after_tokens> (verbatim tail <tail_tokens>)
Picked: first prompt "<first_prompt>", last prompt "<last_prompt>"
Warnings: <warnings>, one per line, or "none"
```

`integrations/codex/loosy/agents/openai.yaml`:

```yaml
policy:
  allow_implicit_invocation: false
```

- [ ] **Step 5: Write `integrations/README.md`**

````markdown
# Integrations

`/loosy` in Claude Code and `$loosy` in Codex fork the current session into a compressed copy
that you then resume. Both call the `handoff_session` tool of the `loosy-goose-mcp` server.

## 1. Install the server

From a clone:

```
uv tool install --editable "$HOME/Developer/loosy-goose[embed,mcp]"
```

From a release (replace `vX.Y.Z`):

```
uv tool install --torch-backend cpu "loosy-goose[embed,mcp] @ git+https://github.com/SebastianFrazier26/loosy-goose@vX.Y.Z"
```

You can also give the release's wheel URL in place of the `git+` URL. A wheel carries no
`[tool.uv.sources]`, so on Linux `--torch-backend cpu` is what keeps uv from pulling CUDA torch.
The flag is a no-op on macOS and selects CPU torch on Windows.

## 2. Register it

Claude Code, for every project:

```
claude mcp add -s user loosy-goose -- loosy-goose-mcp
```

Codex:

```
codex mcp add loosy-goose -- loosy-goose-mcp
```

The server loads its embedding model when it starts, so the first `/loosy` does not time out.
That makes startup take a few seconds. For Codex, raise the limits in `~/.codex/config.toml`:

```toml
[mcp_servers.loosy-goose]
startup_timeout_sec = 120
tool_timeout_sec = 600
```

If Claude Code reports a startup timeout, start it with `MCP_TIMEOUT=120000 claude`.

## 3. Install the command

```
cp integrations/claude/loosy.md ~/.claude/commands/loosy.md                      # Claude Code
mkdir -p ~/.agents/skills && cp -R integrations/codex/loosy ~/.agents/skills/loosy   # Codex
```

Codex's custom prompts (`~/.codex/prompts`) are deprecated in favour of skills. The skill only
runs when you type `$loosy`.

## 4. Use it

In a long session, run `/loosy` (Claude Code) or `$loosy` (Codex). You can add a level
(`light` keeps 0.50, `medium` 0.33, the default, `heavy` 0.25) and `key=value` options:
`protect_tail_tokens=8000`, `target_tokens=60000`, `keep=0.4`, `verify=true`. Then exit and run
the printed command.

- **What the fork holds:**
  - a marker first;
  - the turns the selector kept, extractively and in order, with tool calls kept whole;
  - your last turn before `/loosy` and everything after it, verbatim, extended back to at
    least 4,000 tokens.

  The `/loosy` turn itself is left out.
- **The original is never modified.** The result prints the command back to it. `verify=true`
  first resumes a throwaway copy headless and asks for `OK`. This sends the compressed history
  to the model once.
- **Limits:**
  - The fork relies on each CLI's undocumented session format. It is tested against Claude Code
    2.1.296 and codex-cli 0.158.0. A newer CLI still forks, with a warning.
  - After upgrading a CLI, run `uv run pytest -m live`.
  - Two sessions in the same directory: check the first and last prompt the result echoes, or
    pass `session=<id>`.
````

- [ ] **Step 6: Update `README.md`, `CLAUDE.md`, `CHANGELOG.md`**

`README.md`:
- In the Documentation list, add: `- [integrations/README.md](integrations/README.md) — the /loosy (Claude Code) and $loosy (Codex) session handoff: install, register, use`.
- Before `## Continuing on another machine`, add:

  ```markdown
  ## Handoff from Claude Code or Codex

  `loosy-goose-mcp` is an MCP server whose `handoff_session` tool forks your live Claude Code or
  Codex session into a compressed copy. The copy holds the kept turns, a verbatim recent tail and
  a marker, and you resume it in place of built-in `/compact`. The original session is never
  modified. Install, registration and the `/loosy` / `$loosy` commands:
  [integrations/README.md](integrations/README.md). The same server's `compress` tool returns
  compressed turns for any transcript and writes nothing.
  ```

- In `## Test`, add after the lint line: ``Opt-in drift and end-to-end tests against the real CLIs (needs logged-in `claude` and `codex`, spends a little API use): `uv run pytest -m live`. Re-record the synthetic fixtures with `uv run python tests/session_recorder.py`.``
- In `## Roadmap`, replace the `- MCP server` and `- Claude Code plugin; Codex parity` lines with `- **Done** — MCP server and session handoff (`/loosy`, `$loosy`) for Claude Code and Codex` and `- Claude Code plugin packaging and marketplace entry`.

`CLAUDE.md`:
- In Commands, change the install line to `uv sync --dev --extra embed --extra mcp`.
- Add these commands:
  - `uv run loosy-goose-mcp`: the MCP server on stdio.
  - `uv run pytest -m live`: opt-in drift/e2e tests against the real CLIs.
  - `uv run python tests/session_recorder.py`: re-record the synthetic session fixtures.
- In Layout → Core, add after `cli.py`:
  - `` `rate.py` — `LEVELS`, keep bounds, `resolve_rate` (shared by the CLI and the MCP server) ``
  - `` `sessions/` — session handoff: `model.py` (tool-agnostic `Session`), `plan.py` (trigger, protected tail, units, `plan_fork`, marker), `claude.py` / `codex.py` (the only code that knows a CLI's session format), `adapter.py` (protocol) ``
  - `` `handoff.py` — locate, snapshot, compress, write, validate, verify ``
  - `` `mcp_server.py` — `handoff_session` and `compress` tools (`mcp` extra) ``
- In Supporting, add:
  - `` `integrations/` — `/loosy` command (Claude Code), `$loosy` skill (Codex), registration README ``
  - `` `tests/fixtures/sessions/` — synthetic Claude Code and Codex sessions recorded by `tests/session_recorder.py` ``
- In Hard rules, add: `- Session fixtures are synthetic only. Re-record them with tests/session_recorder.py; never copy a real session into the repo.`

`CHANGELOG.md`, append:

```markdown
- 2026-10-13 — `integrations/`: the `/loosy` Claude Code command and the `$loosy` Codex skill (Codex custom prompts are deprecated in favour of skills), both calling `handoff_session`, plus install and registration instructions for both CLIs. README and CLAUDE.md document the handoff, the `mcp` extra and the opt-in live tests.
```

- [ ] **Step 7: Run the tests and see them pass**

Run: `uv run pytest tests/test_integrations.py -q && uv run ruff check . && uv run ruff format --check .`
Expected: 3 passed.

- [ ] **Step 8: Commit **

```bash
git add integrations README.md CLAUDE.md CHANGELOG.md tests/test_integrations.py
git commit -m "$(cat <<'MSG'
Add the /loosy command and $loosy skill, and document the handoff

Claude Code custom command and Codex skill (explicit-only) calling handoff_session;
install and registration for both CLIs; README, CLAUDE.md and CHANGELOG updated.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Task 13: Opt-in live drift and end-to-end tests (both CLIs)

**Files:**
- Modify: `pyproject.toml:63-66` (`[tool.pytest.ini_options]`: `addopts`, `live` marker)
- Create: `tests/test_live_handoff.py`
- Test: `tests/test_project_config.py`
- Modify (only if Step 6 fails): `src/loosy_goose/handoff.py` (`DEFAULT_KEEP_THINKING`), `CHANGELOG.md`

**Interfaces:**
- Consumes:
  - `scratch_repo`, `record_claude`, `record_codex`, `CODE_WORDS`, `SENTINEL` (Task 5);
  - `kept_where` (Task 3);
  - the adapters (Tasks 6–9);
  - `find_trigger`, `split_region`, `plan_fork`, `DEFAULT_TRIGGERS` (Tasks 3–4);
  - `SessionAdapter` (Task 10);
  - `integrations/claude/loosy.md` (Task 12).
- Produces: the `live` pytest marker, deselected by default. `uv run pytest -m live` runs only these.

**Tests:**

- **Drift** (per CLI). Record a fresh synthetic session, plan a fork that deterministically drops every BETA segment, validate it, resume it headless, and assert that the model lists exactly ALPHA, GAMMA and DELTA and repeats the marker.
- **Thinking** (per CLI, the spec's open item). Fork keeping every segment, which drops older thinking/reasoning by default, and assert that the resume works and lists all four words. If the recording has no thinking before the tail, the test skips and says so.
- **End-to-end** (per CLI). Register the server for this run only, without touching the user's config: Claude through `--mcp-config` with the real `/loosy` command file, Codex through `-c mcp_servers.loosy-goose.*` overrides. Trigger `handoff_session` through a real tool call, find the fork, resume it, and assert that:
  - the model's code words equal those in the fork file;
  - the marker is present;
  - the DELTA prompt (the protected tail) is verbatim;
  - the trigger turn is absent.

- [ ] **Step 1: Write the failing config test** (`tests/test_project_config.py`)

```python
import tomllib
from pathlib import Path


def test_live_tests_are_registered_and_opt_in() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    options = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    assert options["addopts"] == "-m 'not live'"
    assert any(m.startswith("live:") for m in options["markers"])
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/test_project_config.py -q`
Expected: FAIL with `KeyError: 'addopts'`.

- [ ] **Step 3: Register the marker**

Replace the `markers = [...]` line in `[tool.pytest.ini_options]` with:

```toml
addopts = "-m 'not live'"
markers = [
    "slow: needs the embedding model; skipped when sentence_transformers is missing",
    "live: needs logged-in claude and codex CLIs and spends a little API use; run with `uv run pytest -m live`",
]
```

A later `-m live` on the command line overrides the `addopts` one.

Run: `uv run pytest tests/test_project_config.py -q`
Expected: 1 passed.

- [ ] **Step 4: Write `tests/test_live_handoff.py`**

```python
"""Drift and end-to-end handoff tests against the real CLIs.

Opt-in: `uv run pytest -m live`. Needs `claude` and `codex` on PATH and logged in, and spends a
little API use. Run after upgrading either CLI: a failure here means its session format moved.
The synthetic sessions they record stay in ~/.codex (its sqlite has indexed them); the Claude
project folder of each scratch repo is removed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import pytest
from session_builders import kept_where
from session_recorder import CODE_WORDS, SENTINEL, record_claude, record_codex, scratch_repo

from loosy_goose.segment import Segment
from loosy_goose.sessions import claude, codex
from loosy_goose.sessions.adapter import SessionAdapter
from loosy_goose.sessions.model import Session
from loosy_goose.sessions.plan import DEFAULT_TRIGGERS, find_trigger, plan_fork, split_region

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[1]
ASK = (
    "List every code word you were told in this conversation, comma-separated, nothing else. "
    "Then, on a new line, copy the first message of this conversation word for word."
)
MARKER_SEEN = "loosy-goose: compressed from session"
SERVER = ["uv", "run", "--project", str(REPO), "--extra", "embed", "--extra", "mcp",
          "loosy-goose-mcp"]
Recorder = Callable[..., Path]


def _need(*clis: str) -> None:
    for cli in clis:
        if shutil.which(cli) is None:
            pytest.skip(f"{cli} is not on PATH")


def _ask(argv: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, timeout=900, check=False,
        env={**os.environ, **(env or {})},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def _words(text: str) -> set[str]:
    return {w for w in CODE_WORDS if w in text}


def _session_words(s: Session) -> set[str]:
    return _words(" ".join(b.text for ri in s.live for b in s.records[ri].blocks))


def _setup(tool: str) -> tuple[SessionAdapter, Path, Recorder, Callable[[str], list[str]]]:
    if tool == "claude":
        return claude, claude.default_home(), record_claude, (
            lambda fid: ["claude", "-p", ASK, "--resume", fid]
        )
    return codex, codex.default_home(), record_codex, (
        lambda fid: ["codex", "exec", "resume", fid, ASK]
    )


def _cleanup(tool: str, cwd: Path, home: Path) -> None:
    if tool == "claude":
        shutil.rmtree(home / "projects" / claude.project_slug(str(cwd)), ignore_errors=True)
    shutil.rmtree(cwd, ignore_errors=True)


def _drift(
    adapter: SessionAdapter,
    source: Path,
    home: Path,
    cwd: Path,
    work: Path,
    keep: Callable[[Segment], bool],
    resume: Callable[[str], list[str]],
) -> tuple[str, Session, Session]:
    session = adapter.read(source)
    region = split_region(
        session, find_trigger(session, DEFAULT_TRIGGERS[adapter.TOOL]), protect_tail_tokens=0
    )
    plan = plan_fork(session, region, kept_where(region, keep), date=date.today().isoformat())
    fork_id = adapter.new_session_id()
    staged = adapter.write_fork(session, plan, fork_id, work)
    assert adapter.validate(staged, fork_id) == []
    fork = adapter.read(staged)
    dest = adapter.fork_destination(source, fork_id, home, datetime.now())
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(staged, dest)
    try:
        return _ask(resume(fork_id), cwd), session, fork
    finally:
        dest.unlink(missing_ok=True)


@pytest.mark.parametrize("tool", ["claude", "codex"])
def test_drift_a_fork_resumes_with_exactly_the_kept_words(tool: str, tmp_path: Path) -> None:
    _need(tool)
    adapter, home, record, resume = _setup(tool)
    cwd = scratch_repo()
    try:
        source = record(cwd, home)
        answer, _, _ = _drift(
            adapter, source, home, cwd, tmp_path, lambda s: "BETA" not in s.text, resume
        )
    finally:
        _cleanup(tool, cwd, home)
    assert _words(answer) == {"ALPHA", "GAMMA", "DELTA"}
    assert MARKER_SEEN in answer


@pytest.mark.parametrize("tool", ["claude", "codex"])
def test_drift_dropping_older_thinking_survives_resume(tool: str, tmp_path: Path) -> None:
    _need(tool)
    adapter, home, record, resume = _setup(tool)
    cwd = scratch_repo()
    try:
        source = record(cwd, home)
        answer, session, fork = _drift(adapter, source, home, cwd, tmp_path, lambda s: True, resume)
    finally:
        _cleanup(tool, cwd, home)
    before = sum(1 for ri in session.live if session.records[ri].thinking)
    after = sum(1 for ri in fork.live if fork.records[ri].thinking)
    if before == after:
        pytest.skip("this recording has no thinking before the protected tail; nothing dropped")
    assert _words(answer) == set(CODE_WORDS)


def _check_e2e(source: Session, fork: Session, answer: str) -> None:
    assert _words(answer) == _session_words(fork)
    assert MARKER_SEEN in answer
    delta = next(p for p in source.prompts() if "DELTA" in p)
    assert delta in fork.prompts()
    assert not any(SENTINEL in p or "/loosy" in p for p in fork.prompts())


def test_e2e_claude_handoff_through_the_mcp_tool(tmp_path: Path) -> None:
    _need("claude", "uv")
    home = claude.default_home()
    cwd = scratch_repo()
    shutil.copy(REPO / "integrations" / "claude" / "loosy.md", cwd / ".claude" / "commands")
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps({"mcpServers": {"loosy-goose": {"command": SERVER[0], "args": SERVER[1:]}}}),
        encoding="utf-8",
    )
    try:
        source = record_claude(cwd, home, trigger=False)
        existing = set(source.parent.glob("*.jsonl"))
        # Prompt before --allowedTools: the flag is variadic and would swallow it.
        out = _ask(
            [
                "claude", "-p", "/loosy heavy protect_tail_tokens=0", "--resume", source.stem,
                "--mcp-config", str(config),
                "--allowedTools", "mcp__loosy-goose__handoff_session",
            ],
            cwd,
            env={"MCP_TIMEOUT": "120000", "MCP_TOOL_TIMEOUT": "600000"},
        )
        forks = sorted(set(source.parent.glob("*.jsonl")) - existing)
        assert len(forks) == 1, out
        fork = claude.read(forks[0])
        answer = _ask(["claude", "-p", ASK, "--resume", fork.session_id], cwd)
        _check_e2e(claude.read(source), fork, answer)
    finally:
        _cleanup("claude", cwd, home)


def _forked_from(path: Path) -> str | None:
    with path.open(encoding="utf-8") as fh:
        head = json.loads(fh.readline())
    value = head.get("payload", {}).get("forked_from_id")
    return value if isinstance(value, str) else None


def test_e2e_codex_handoff_through_the_mcp_tool(tmp_path: Path) -> None:
    _need("codex", "uv")
    home = codex.default_home()
    cwd = scratch_repo()
    try:
        source = record_codex(cwd, home, trigger=False)
        sid = codex.read(source).session_id
        overrides = [
            "-c", f"mcp_servers.loosy-goose.command={json.dumps(SERVER[0])}",
            "-c", f"mcp_servers.loosy-goose.args={json.dumps(SERVER[1:])}",
            "-c", "mcp_servers.loosy-goose.startup_timeout_sec=120",
            "-c", "mcp_servers.loosy-goose.tool_timeout_sec=600",
        ]
        prompt = (
            f"{SENTINEL} Call the handoff_session tool of the loosy-goose MCP server exactly "
            f'once with level "heavy", protect_tail_tokens 0, tool "codex" and cwd '
            f"{json.dumps(str(cwd))}, then reply with its resume_command only."
        )
        _ask(["codex", "exec", "resume", *overrides, sid, prompt], cwd)
        forks = [
            p for p in (home / "sessions").glob("*/*/*/rollout-*.jsonl") if _forked_from(p) == sid
        ]
        assert len(forks) == 1
        fork = codex.read(forks[0])
        answer = _ask(["codex", "exec", "resume", fork.session_id, ASK], cwd)
        _check_e2e(codex.read(source), fork, answer)
    finally:
        _cleanup("codex", cwd, home)
```

`record_claude` and `record_codex` take `(cwd, home)` positionally, which is why `Recorder = Callable[..., Path]`.

- [ ] **Step 5: Check the default run deselects them**

Run: `uv run pytest -q`
Expected: all green, with a `deselected` count that includes the 6 live tests.

- [ ] **Step 6: Run the live tests (local; needs logged-in CLIs; spends a little API use)**

Run: `uv sync --dev --extra embed --extra mcp && uv run pytest -m live -v`
Expected: 6 passed, except that a thinking test may skip with "no thinking before the protected tail".

If a Codex e2e run stops on a tool-approval prompt (exec mode's MCP approval behaviour is unverified), add `"-c", 'approval_policy="never"'` to `overrides`, re-run, and note it in `integrations/README.md` §2.

If `test_drift_dropping_older_thinking_survives_resume` fails with an API error about thinking or reasoning (signature or ordering), it is the spec's open item failing. Apply the fallback:

- set `DEFAULT_KEEP_THINKING = True` in `src/loosy_goose/handoff.py` and delete the comment above it;
- add a `CHANGELOG.md` line that names the CLI and version whose resume broke, for example `- 2026-10-13 — handoff keeps thinking/reasoning inside kept turns: dropping it broke resume on Claude Code 2.1.296 (live drift test).`;
- change the test to fork with `plan_fork(..., keep_thinking=True)` by threading a `keep_thinking` parameter through `_drift`;
- re-run until green.

- [ ] **Step 7: Commit **

```bash
git add pyproject.toml tests/test_project_config.py tests/test_live_handoff.py
git commit -m "$(cat <<'MSG'
Add opt-in live drift and end-to-end handoff tests for both CLIs

`uv run pytest -m live`: record a synthetic session, fork it, resume headless and check
the model sees exactly the kept code words and the marker; drop older thinking and resume;
trigger handoff_session through a real MCP tool call in Claude Code and Codex.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

If Step 6 applied the thinking fallback, also `git add src/loosy_goose/handoff.py CHANGELOG.md`.

---

## Task 14: Release workflow on `v*` tags

**Files:**
- Create: `.github/workflows/release.yml`
- Modify: `.github/workflows/ci.yml:3-6` (add `workflow_call:` so the release can run the full CI matrix)
- Modify: `README.md` (one line under `## Run` → after the Handoff section)
- Modify: `CHANGELOG.md` (append)
- Test: `tests/test_workflows.py` (append)

**Interfaces:**
- Consumes: `ci.yml` (Task 11 state); `integrations/` files (Task 12).
- Produces: a GitHub Release per `v*` tag, with `dist/*` (sdist and wheel), `integrations/claude/loosy.md`, `integrations/codex/loosy/SKILL.md`, `integrations/codex/loosy/agents/openai.yaml` and `integrations/README.md` attached, and the install command in the notes. The user creates and pushes the tag; nothing in this plan does.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_workflows.py`)

```python
import re

import pytest

_USES = re.compile(r"uses:\s*(\S+)")


@pytest.mark.parametrize("name", ["ci.yml", "release.yml"])
def test_every_action_is_pinned_to_an_exact_version(name: str) -> None:
    for ref in _USES.findall((WORKFLOWS / name).read_text(encoding="utf-8")):
        if ref.startswith("./"):
            continue
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+\.\d+\.\d+", ref), ref


def test_ci_can_be_called_by_the_release() -> None:
    assert "workflow_call:" in (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")


def test_release_tests_builds_checks_cpu_torch_and_publishes_on_tags() -> None:
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert re.search(r'tags:\s*\["v\*"\]', text)
    assert "uses: ./.github/workflows/ci.yml" in text
    assert "uv build" in text
    assert "--torch-backend cpu" in text and "+cpu" in text
    assert "gh release create" in text and "contents: write" in text
    for path in (
        "dist/*",
        "integrations/claude/loosy.md",
        "integrations/codex/loosy/SKILL.md",
        "integrations/codex/loosy/agents/openai.yaml",
        "integrations/README.md",
    ):
        assert path in text
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/test_workflows.py -q`
Expected: the `release.yml` parametrization and the release test fail with `FileNotFoundError`; `test_ci_can_be_called_by_the_release` fails on the assertion.

- [ ] **Step 3: Make CI callable**

In `.github/workflows/ci.yml`, change the `on:` block to:

```yaml
on:
  pull_request:
  push:
    branches: [main]
  workflow_call:
```

- [ ] **Step 4: Write `.github/workflows/release.yml`**

```yaml
name: Release

on:
  push:
    tags: ["v*"]

permissions:
  contents: read

jobs:
  test:
    uses: ./.github/workflows/ci.yml

  release:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v7.0.1
      - uses: astral-sh/setup-uv@v10.0.1
        with:
          # Explicit: setup-uv's "auto" skips the cache on tag and release events.
          enable-cache: true
      - run: uv python install 3.12
      - run: uv build
      # A wheel carries no [tool.uv.sources], so CPU torch on Linux has to be asked for; this
      # is the install the release notes tell users to run.
      - name: Install the built wheel with CPU torch
        run: |
          wheel="$(ls dist/*.whl)"
          uv tool install --python 3.12 --torch-backend cpu "loosy-goose[embed,mcp] @ file://$PWD/$wheel"
          "$(uv tool dir)/loosy-goose/bin/python" -c "import torch; v = torch.__version__; print(v); assert v.endswith('+cpu'), v"
          "$(uv tool dir --bin)/loosy-goose" --help
      - name: Create the GitHub Release
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          cat > notes.md <<EOF
          Install (uv 0.11.3 or newer):

              uv tool install --torch-backend cpu "loosy-goose[embed,mcp] @ git+https://github.com/SebastianFrazier26/loosy-goose@${GITHUB_REF_NAME}"

          Then register the MCP server and install /loosy (Claude Code) or \$loosy (Codex): see integrations/README.md, attached.
          EOF
          gh release create "$GITHUB_REF_NAME" dist/* \
            integrations/claude/loosy.md \
            integrations/codex/loosy/SKILL.md \
            integrations/codex/loosy/agents/openai.yaml \
            integrations/README.md \
            --title "$GITHUB_REF_NAME" --notes-file notes.md
```

- [ ] **Step 5: Docs**

`README.md`, append to the "Handoff from Claude Code or Codex" section: `Releases: each \`v*\` tag publishes a GitHub Release with the wheel, the sdist and the integrations files; its notes carry the install command.`

`CHANGELOG.md`, append:

```markdown
- 2026-10-13 — `.github/workflows/release.yml`: on a `v*` tag, run the full CI matrix (`ci.yml` is now callable), `uv build`, install the built wheel with `--torch-backend cpu` on Linux and assert torch is a `+cpu` build, then publish a GitHub Release with the wheel, sdist and `integrations/` files. The tag is created and pushed by hand.
```

- [ ] **Step 6: Run the tests and see them pass**

Run: `uv run pytest tests/test_workflows.py -q && uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src`
Expected: 5 passed, then the whole suite green.

- [ ] **Step 7: Check the wheel install locally (macOS: the flag is a no-op, so torch is not `+cpu` here)**

Run: `uv build && uv tool install --force --python 3.12 --torch-backend cpu "loosy-goose[embed,mcp] @ file://$PWD/$(ls dist/*.whl | tail -1)" && "$(uv tool dir --bin)/loosy-goose" --help && uv tool uninstall loosy-goose`
Expected: the help text prints. This checks the wheel's extras and scripts resolve, which the spec lists as not yet checked. If the user already has an editable `uv tool` install of loosy-goose, skip this step: `--force` would replace it. Instead ask the user to run it.

- [ ] **Step 8: Commit **

```bash
git add .github/workflows/release.yml .github/workflows/ci.yml README.md CHANGELOG.md tests/test_workflows.py
git commit -m "$(cat <<'MSG'
Add the release workflow for v* tags

Full CI matrix, uv build, a CPU-torch check of the built wheel on Linux, then a GitHub
Release with the wheel, sdist and integrations files. Tags are pushed by hand.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSG
)"
```

---

## Spec ambiguities resolved in this plan

Each of these is a decision. Raise it with the user if it looks wrong.

1. **How the `/loosy` turn is identified.** The trigger is the latest of three signals:
   - the last prompt containing a trigger string: Claude `<command-name>/loosy</command-name>` (how Claude Code records a custom slash command, checked from record keys only) or the sentinel `loosy-goose:handoff`; Codex `$loosy` or the sentinel;
   - the last open Codex turn;
   - the exchange of the last `handoff_session` tool call.

   With none of them, nothing is excluded, and a warning says so ([RF5]). Taking the latest signal keeps an old `/loosy` text from cutting everything after it.
2. **"Newest session for the cwd".** The adapter returns the cwd's sessions newest first. Among the 3 newest, one carrying a trigger wins. A warning fires when more than one changed in the last 10 minutes ([RF3]).
3. **Claude project slug.** Every non-alphanumeric character becomes `-` (inferred from the probe). The recorded `cwd` field is authoritative, and a full scan is the fallback.
4. **Compacted sessions.**
   - Claude: history starts at the last `compact_boundary`, the boundary and summary are pinned, and the marker goes after the boundary.
   - Codex: history starts at the last `compacted` record, which is pinned, and the marker goes after it.
5. **Codex prompts versus injected context.** A user message is a prompt only when an `event_msg` `user_message` echoes it. Other user and developer messages (instructions, environment context) are pinned and always kept.
6. **Meta records** (attachments after startup, Codex `event_msg`/`turn_context`/`world_state`) are kept iff their exchange keeps a message.
7. **Trimmed tool results.** A dropped run of chunks becomes a `[loosy-goose: lines trimmed]` line. A result whose call was kept but whose chunks all were not becomes `[loosy-goose: output trimmed]`. Claude's `toolUseResult` mirror field is left as recorded (UI-only).
8. **`target_tokens`** is a whole-fork target. The compressible part gets the target minus everything kept verbatim. A target below that is an error, and a session already under target is "nothing-to-do".
9. **Repeated `/loosy`.** An earlier fork's marker is classified `side`, so it is replaced, not stacked.
10. **`last-prompt`** is always written for the new leaf. This settles the spec's open item without needing to know whether Claude Code requires it.
11. **Failed live check.** With `verify=true`, a failed check deletes the fork as well and raises, consistent with "a failure leaves the user's session as it was".
12. **Codex command mechanism.** The spec says "prompt". Codex custom prompts are deprecated, so v1 ships a skill (`$loosy`, explicit-only).
13. **Fork placement.** The fork is written and validated in a private temp dir and moved next to the original only when it passes. A broken fork never appears in a CLI's session list.

## Self-Review

**Spec coverage:**

| Spec section | Task(s) |
|---|---|
| Goal / success: resume a fork with kept turns, verbatim tail, marker; failures leave the session untouched | 10 (unchanged-original asserts), 13 (e2e) |
| Decision: rewrite into a new file and resume | 7, 9, 10, 13 |
| Decision: fork, new id, original never written | 7, 9 (new ids), 10 (byte and mtime asserts) |
| Decision: `/loosy [level]` calling an MCP tool | 11, 12 |
| Decision: recent context ≥ 4,000 tokens verbatim | 3 (`split_region`), 4 (tail kept whole) |
| Decision: marker | 4 (exact text), 7, 9 (placement) |
| Decision: audience, install from clone | 12 (`integrations/README.md`, README) |
| Decision: GitHub Release per `v*` tag | 14 |
| Feasibility facts (chain walk, orphan repair, id fields, sqlite) | 6–9 (adapters), 13 (drift) |
| Architecture: locate / snapshot / adapter / compress / unitize / write / validate / return | 6, 8 / 3, 10 / 6–9 / 10 / 3, 4 / 7, 9 / 7, 9, 10 / 10 |
| MCP `compress` tool | 11 |
| New modules `sessions/claude.py`, `sessions/codex.py`, `handoff.py`, `mcp_server.py`, `mcp` extra | 6–11 |
| Rate resolution moves into the library | 1 |
| `integrations/` | 12 |
| Mapping: blocks tagged with records; compressor unchanged | 2 (`Segment.block`), 3 (`part` meta), 4 (mapping) |
| Atomic units; tool input never trimmed; results trimmable | 3 (`unit_ids`), 4 |
| Thinking dropped in older turns (+ implementation-time test) | 4, 13 (Step 6 fallback) |
| Partly kept records keep their id; chain re-linked | 4 (rewrites), 7 |
| Non-message records (Claude startup attachments kept, side dropped, `last-prompt`; Codex turn events with their turns, `session_meta` new id + `forked_from_id`) | 6, 7, 8, 9 |
| Protected tail, configurable `protect_tail_tokens` | 3, 10, 11 |
| `/loosy` turn excluded | 3, 4, 10, 13 |
| Error handling: read only, sqlite untouched, only our forks deleted | 10 |
| Locate echoes first and last prompt; explicit `session=<id>`; clear error when none | 6, 8, 10 |
| Versions: newer CLI warns; unknown types pass through or refuse | 4, 6, 8, 10 |
| Validation per CLI; deleted on failure, naming the check | 7, 9, 10 |
| Optional `verify=true` live check | 10 |
| Nothing to do (under target or no larger than the tail) | 10 |
| Startup: model preloaded; missing `embed` → install command | 11 |
| Every result: both resume commands, tokens, tail, warnings | 10, 11, 12 |
| Testing: unit, fixtures, MCP in-process, drift, e2e, release | 3–11, 5, 11, 13, 13, 14 |
| Distribution: install commands, release notes | 12, 14 |
| Open items: thinking, `last-prompt`, SDK choice | 13, 7, 11 |

**Placeholder scan:** no TBD/TODO. Every code step carries its code, and no step says "similar to Task N". The template strings left in deliverables are deliberate: `vX.Y.Z` in the install docs, `<your clone>` in the startup error, the `<field>` slots in the command files that the model fills from the result, and `session=<id>` in messages.

**Name and type consistency, checked across tasks:**

- `Session`, `SessionRecord` fields; `Trigger(position, how)`; `Region(end, tail_from, transcript, turn_record)`.
- `ForkPlan` fields; `plan_fork(session, region, kept, *, date, keep_thinking)`.
- The adapter surface, identical in `claude.py` and `codex.py`: `TOOL`, `TESTED_VERSIONS`, `default_home`, `locate`, `read`, `write_fork`, `validate`, `fork_destination`, `new_session_id`, `resume_command`, `verify_command`. `SessionAdapter` matches it.
- `HandoffOptions` and `HandoffResult` field names. The command files' `<field>` slots are tested against `HandoffResult`, and their argument names against `HandoffOptions`.
- Test helpers: `build`, `words`, `kept_where`, `fixture_lines`, `write_lines`, `install_claude`, `install_codex`, `fake_embeddings`, `without_trigger`.

**Review Focus coverage:** each line has a pinning test in its owning task:

- [RF1] `test_rf1_an_already_compacted_session_forks_from_its_boundary` (Task 7);
- [RF2] `test_rf2_a_partial_last_line_is_ignored_and_the_original_untouched` (Task 10);
- [RF3] `test_rf3_the_session_carrying_the_trigger_beats_a_newer_one` (Task 10);
- [RF4] `test_rf4_a_session_spanning_context_windows_forks_from_its_last_compaction` (Task 9);
- [RF5] `test_rf5_an_unidentifiable_trigger_excludes_nothing_and_says_so` (Task 10).

**Not verified while planning** (each is checked by a task step, and the step says what to do if it fails):

- that `claude -p` expands a project command into the `<command-name>` record shape (Task 5 recorder asserts it);
- that `codex exec resume` appends to the same rollout file (Task 5);
- Codex's exec-mode approval behaviour for MCP tools (Task 13, Step 6);
- the `MCP_TIMEOUT`/`MCP_TOOL_TIMEOUT` environment variables of Claude Code (Task 13 e2e);
- `uv tool dir --bin` (Task 14, Steps 4 and 7);
- whether the dangling `logicalParentUuid` on a kept `compact_boundary` matters to Claude Code (exercised only by a live resume of a compacted fork, which no live test records; the drift test covers uncompacted sessions);
- whether dropping thinking survives resume (Task 13, Step 6).

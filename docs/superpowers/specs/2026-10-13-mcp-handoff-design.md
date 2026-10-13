# v1 MCP server and session handoff — design

Status: approved by the user 2026-10-13; plan in `docs/superpowers/plans/2026-10-13-mcp-handoff.md`. Branch `feat/mcp-handoff`.

## Goal

Run loosy-goose on the live agent conversation and continue the work in a session whose history *is*
the compressed transcript: real kept turns, not a model-written summary. This replaces built-in
compaction (Claude Code `/compact`, Codex `/compact`) by user choice, for Claude Code first and Codex
with the same behaviour.

Success: a long session is compressed with `/loosy`, the user resumes the printed fork, and the model
continues with the kept turns, the untouched recent tail, and a marker saying history was compressed.
A failure at any step leaves the user's session exactly as it was.

## Decisions (2026-10-12/13, with the user)

| Question | Decision | Rejected |
|---|---|---|
| Mechanism | Rewrite the session into a new file and resume it (`claude --resume`, `codex resume`) | Fresh session reading a handoff file after `/clear` (content arrives as a file read, not turns); injecting after built-in compaction (summary + ours, ~10k-char hook cap) |
| Original session | Fork: new session id, original never written | Overwrite in place with backup |
| Trigger | In-session `/loosy [light\|medium\|heavy]` calling an MCP tool | Shell-only `loosy-goose handoff` |
| Recent context | Last user turn before `/loosy` onward, at least 4,000 tokens, kept verbatim | Compress uniformly |
| Marker | First message of the fork states it was compressed, from which session, and the kept share | No marker |
| Audience | The user, installed from the clone; public PyPI is a later decision | PyPI from v1 |
| Releases | GitHub Release per `v*` tag with wheel, sdist and the command files | — |

Neither CLI documents a way to replace live context, so the fork relies on undocumented session-file
formats. Both CLIs say that format changes between releases. This is accepted, contained by the
per-tool adapter, the version check and the drift test below.

## Feasibility (probe, 2026-10-12, Claude Code 2.1.296, codex-cli 0.158.0)

A synthetic session (code words ALPHA–DELTA plus one `echo` tool call) was forked with one turn
removed and resumed headless. Both CLIs loaded the fork and the model listed exactly the kept words.

- **Claude Code.** File `~/.claude/projects/<cwd-slug>/<sessionId>.jsonl`. History is rebuilt by
  walking `parentUuid` back from the leaf, not by line order. A gap that is not re-linked loads
  without error but silently truncates history at the gap. An orphaned `tool_use` is repaired in
  memory; the model then believes the command never ran. `--resume <absolute path>` works.
- **Codex.** File `~/.codex/sessions/YYYY/MM/DD/rollout-<ts>-<id>.jsonl`. History is file order.
  The session id appears in many fields (`session_meta`, `thread_id`, `session_id`). Orphaned call
  or output logs an error, is patched as aborted, and continues. A new id is found by scanning; Codex
  indexes it into `~/.codex/state_5.sqlite` itself.

## Architecture

```
/loosy [level]            plugin command (Claude Code) / prompt (Codex)
  └─► MCP tool handoff_session(level | keep | target_tokens, session="current", verify=false)
        locate    newest session file for the cwd, or an explicit id
        snapshot  copy the file; ignore a partial last line
        adapter   sessions/claude.py | sessions/codex.py: file → Transcript with record refs
        compress  existing compress_transcript(), unchanged
        unitize   group segments into atomic units; protect the tail
        write     adapter.write_fork → new file, marker first
        validate  re-read the fork the way the CLI does; delete it on failure
        return    resume command, original's resume command, token counts, warnings

MCP tool compress(messages | path, level | keep | target_tokens) → compressed blocks; writes nothing
```

New modules:

- `loosy_goose/sessions/claude.py`, `loosy_goose/sessions/codex.py` — the only code that knows a file
  format. Each exposes `locate`, `read`, `write_fork`, `validate`, and `TESTED_VERSIONS`.
- `loosy_goose/handoff.py` — the pipeline above, tool-agnostic.
- `loosy_goose/mcp_server.py` — thin MCP layer over `handoff` and `compress`; console script
  `loosy-goose-mcp`. The MCP SDK goes in a new `mcp` extra.
- Rate resolution (`LEVELS`, `resolve_rate`, keep bounds) moves from `cli.py` into the library so the
  CLI and the server share it. CLI behaviour is unchanged.
- `integrations/` — the `/loosy` command file for Claude Code, the Codex prompt, and registration
  snippets for both. Packaging these as a Claude Code plugin is the next roadmap step, not v1.

## Mapping segments back to records

- The adapter tags every block with the record id(s) it came from. The compressor runs unchanged;
  the writer rebuilds the fork from the original records, keeping what survived.
- **Atomic units, kept or dropped whole:**
  - a tool call with its result (Claude: `tool_use` and the `user` record carrying the matching
    `tool_result`; Codex: call and output sharing a `call_id`);
  - a tool call's input — it must stay valid structured data and is never trimmed. A tool result's
    text may be trimmed.
- **Thinking blocks** in turns older than the protected tail are dropped. Whether editing around them
  disturbs signature checks on resume is an implementation-time test.
- **Partly kept records** keep their id with only their text rewritten; Claude's chain is re-linked
  across every dropped stretch.
- **Non-message records.** Claude: startup attachments (injected CLAUDE.md, skills) are kept;
  `stop_hook_summary`, `queue-operation`, `last-prompt`, `cost-state`, `mode` and similar are
  dropped, and `last-prompt` is rewritten to the new leaf if the CLI needs it. Codex: turn-boundary
  events, `turn_context` and `world_state` are kept for kept turns and dropped with dropped turns;
  `session_meta` gets the new id and `forked_from_id`.
- **Protected tail.** The last user turn before the `/loosy` turn and everything after it up to
  that turn, extended back to at least 4,000 tokens, configurable
  (`protect_tail_tokens`). Only history before it is compressed. The budget applies to the
  compressible part.
- **The `/loosy` turn itself** (and anything after it) is excluded from the fork, so the resumed
  session does not open mid-handoff. This also covers the transcript-write lag.
- **Marker.** A user-role message at the start of the fork:
  `[loosy-goose: compressed from session <id> on <date>; kept <k> of <n> tokens before the last
  <t> tokens, which are verbatim. Earlier turns are an extractive selection — some were dropped.]`

## Error handling — the user's session is never at risk

- **Read only.** The live session file is snapshotted and never written, locked, or ended. Codex's
  sqlite index is not touched. The only file ever deleted is a fork of ours that failed validation.
- **Locate.** The newest session file for the cwd is the live one. The result echoes that session's
  first and last user prompts, so a wrong pick (two concurrent sessions in one directory) is visible;
  an explicit `session=<id>` overrides. No session found means a clear error and nothing written.
- **Versions.** A CLI newer than `TESTED_VERSIONS` still forks, with a warning. Unknown record types
  are passed through unchanged with a warning, unless they fall inside a compressed stretch, in which
  case the handoff refuses.
- **Validation** after writing, mirroring how each CLI loads:
  - Claude: walk `parentUuid` from the leaf; the walk must reach the root and visit every kept
    record; every `tool_use` has its `tool_result`; one `sessionId` throughout.
  - Codex: one id throughout, `forked_from_id` set, every call paired with its output, turns
    well-formed.
  
  Any failure deletes the fork and names the failing check.
- **Optional live check** (`verify=true`, off by default): resume a throwaway copy of the fork headless,
  ask for `OK`, delete the copy. Off by default because it sends the whole compressed history to the
  model.
- **Nothing to do.** If the session is already under target, or no larger than the protected tail,
  nothing is written and the result says so.
- **Startup.** The server loads bge-small when it starts, so the first `/loosy` does not hit an MCP
  timeout. Without the `embed` extra the server fails at startup with the install command.
- **Every result** carries the fork's resume command, the original's resume command (the way back),
  before/after tokens, the tail size, and warnings.

## Testing

- **Unit (CI, no CLIs):** adapter read/write round-trips on fixtures; chain re-linking across gaps;
  tool pairs never split; tail protection; marker; `/loosy` turn exclusion; unknown record types;
  validation catching deliberately corrupted forks (broken chain, orphan tool use, mixed ids).
- **Fixtures:** synthetic sessions only (the probe's code-word shape plus a tool call), recorded from
  Claude Code 2.1.296 and codex-cli 0.158.0, committed. No real session content enters the repo.
- **MCP server (CI):** an in-process MCP client calls `compress` and `handoff_session` against the
  fixtures with stubbed embeddings.
- **Drift test (local, opt-in marker, needs the real CLIs and a little API use):** record a fresh
  synthetic session with the installed Claude Code and Codex, read it with the adapters, fork it,
  resume headless, and assert the model lists exactly the kept code words. Run after upgrading
  either CLI.
- **End-to-end test (local, opt-in, both CLIs):** register the MCP server with Claude Code and with
  Codex, build a synthetic session in each, trigger `handoff_session` through the real MCP tool call,
  resume the fork, and check the kept and dropped code words, the marker and the verbatim tail.
- **Release workflow:** full suite, build, then install the built wheel with
  `--torch-backend cpu` on the Linux runner and assert torch is a `+cpu` build.

## Distribution and releases

- v1 install: `uv tool install --editable "$HOME/Developer/loosy-goose[embed,mcp]"` from the clone, then
  register `loosy-goose-mcp` with each CLI (`claude mcp add`, `codex mcp add` or `[mcp_servers]`).
  Exact commands are written and checked during implementation.
- `.github/workflows/release.yml` on `v*` tags: test, build sdist and wheel, create a GitHub Release
  with them and the `integrations/` files attached. The user creates and pushes the tag.
- Install command in the release notes (uv 0.11.3, resolution checked for x86_64 Linux on
  2026-10-12):
  `uv tool install --torch-backend cpu "loosy-goose[embed,mcp] @ <wheel URL | git+https://github.com/SebastianFrazier26/loosy-goose@vX>"`.
  - A git install honors the repo's `[tool.uv.sources]` on its own; a wheel install does not, so the
    flag is required for wheels.
  - The flag is a no-op on macOS and selects CPU torch on Windows.
  - Not yet checked: an actual `uv tool install` (as opposed to a resolve) and a real tag.

## Out of scope for v1

- Automatic handoff when auto-compact fires (hooks cannot replace the summary; revisit with the plugin).
- Claude Code plugin packaging and marketplace entry (next roadmap step).
- PyPI publishing and solving CUDA torch for plain `pip` users.
- Personalization prior.
- Compressing other agents' formats beyond Claude Code and Codex.

## Open items for implementation

- Does dropping thinking blocks from older turns survive resume? Test it; if it fails, keep them inside
  kept turns.
- Whether Claude Code needs `last-prompt.leafUuid` rewritten for `--resume <id>` to pick the right
  leaf.
- Which MCP SDK to use (the official `mcp` Python SDK vs FastMCP) — decided in the plan, with the user.

# Integrations

`/loosy` in Claude Code and `$loosy` in Codex fork the current session into a compressed copy
that you then resume. Both call the `handoff_session` tool of the `loosy-goose-mcp` server.

## 1. Install the server

From a clone:

```
uv tool install --editable "$HOME/Developer/loosy-goose[embed,mcp]"
```

(Use the path of your own clone if it is elsewhere.) From a release (replace `vX.Y.Z`):

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

Both commands tell the agent to pass the session's working directory as `cwd`, because the
server process may have been started elsewhere.

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

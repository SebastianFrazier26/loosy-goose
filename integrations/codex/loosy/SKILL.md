---
name: loosy
description: Compress this Codex session with loosy-goose and fork it into a new session to resume. Use only when the user invokes $loosy.
---

loosy-goose:handoff

Call the `handoff_session` tool of the `loosy-goose` MCP server exactly once, with these arguments:

- `tool`: "codex"
- `cwd`: the absolute path of this session's current working directory. Always pass it; the server's own directory may differ and would pick the wrong session.
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

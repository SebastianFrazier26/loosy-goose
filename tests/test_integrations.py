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
    assert re.search(r"^- `cwd`: .*current working directory", body, re.MULTILINE)


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

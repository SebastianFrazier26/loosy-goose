"""Record the synthetic session fixtures for the handoff tests from the real CLIs.

    uv run python tests/session_recorder.py [--out tests/fixtures/sessions] [--only claude|codex]

Runs Claude Code and Codex headless in a throwaway git repo for five short turns each, which
spends a little API use. Copies the recorded session files out scrubbed and never deletes
anything: the originals stay in ~/.claude/projects (plus ~/.claude/session-env) and
~/.codex/sessions, and Codex has indexed its copy in ~/.codex/state_5.sqlite, which loosy-goose
never touches. Remove the originals by hand if you do not want them kept.

The scrub is an allowlist: a string survives only if its key is a known-safe one or its value is
exactly one of our prompts or the expected replies. Everything else becomes "<scrubbed>", so a
field a CLI adds later is blanked rather than committed. Unknown record types are an error.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
import socket
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
_EXPANSION = (
    "<command-message>loosy</command-message>\n<command-name>/loosy</command-name>\n"
    "<command-args>medium</command-args>"
)
# The body Claude Code injects for the command: file body, then the arguments.
_EXPANDED_BODY = f"{SENTINEL}\n\nReply with only: handoff requested.\n\n\nARGUMENTS: medium"
_EXACT = frozenset(
    {
        *PROMPTS,
        CLAUDE_TRIGGER,
        CODEX_TRIGGER,
        _EXPANSION,
        _EXPANDED_BODY,
        "",
        SCRUBBED,
        "noted.",
        "handoff requested.",
        "GAMMA",
        "GAMMA\n",
        "echo GAMMA",
        "Print GAMMA",
        "/bin/zsh",
        "-lc",
        'const r = await tools.exec_command({cmd:"echo GAMMA"}); text(r.output);\n',
    }
)
_EXACT_PATTERNS = (re.compile(r"Script completed\nWall time [0-9.]+ seconds\nOutput:\n"),)
# Keys whose string values are ids, enums, versions or timestamps the adapters need. Free text
# (content, text, command, ...) is deliberately absent: it must match _EXACT.
_SAFE_KEYS = frozenset(
    """type subtype role uuid parentUuid leafUuid sessionId session_id thread_id timestamp version
    cli_version model requestedModel advisorModel id message_id response_id call_id tool_use_id
    toolUseID sourceToolAssistantUUID requestId promptId turn_id root_turn_id revision window_id
    stop_reason stopReason entrypoint userType permissionMode operation effort perTurnEffort
    thinkingDisplay inference_geo service_tier speed source decision level promptSource turnOrigin
    attributionSkill renderedRole reasonType kind phase status summary originator model_provider
    model_provider_id approval_policy approvals_reviewer collaboration_mode_kind history_mode
    multi_agent_version network access plan_type reasoning_effort limit_id thread_source
    process_id comp_hash current_date content_item_kinds balance mode signature atis""".split()
)
_OPAQUE_KEYS = frozenset({"signature"})
_PREFIXED_ID_KEYS = frozenset({"id", "message_id", "response_id"})
_PREFIXED_ID = re.compile(r"[a-z]+_[0-9A-Za-z]+")
_PATH_KEYS = frozenset({"cwd", "workspace_roots", "runtime_workspace_roots"})
_NORMALISED = {"gitBranch": "main", "timezone": "UTC"}
_TOOL_NAMES = frozenset({"Bash", "exec"})
_CLAUDE_TYPES = frozenset(
    """assistant user attachment system queue-operation last-prompt cost-state atis-latch
    mode""".split()
)
_CODEX_TYPES = frozenset(
    "session_meta response_item event_msg turn_context world_state token_usage_record".split()
)


class RecorderError(RuntimeError):
    pass


def _run(argv: list[str], cwd: Path) -> str:
    done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=600, check=False)
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


def _keep(key: str, value: str) -> bool:
    if value in _EXACT or any(pat.fullmatch(value) for pat in _EXACT_PATTERNS):
        return True
    if key in _PATH_KEYS:
        return value in (FIXTURE_CWD, f"file://{FIXTURE_CWD}")
    if key == "name":
        return value in _TOOL_NAMES
    return key in _SAFE_KEYS


def _allow(obj: Any, key: str = "") -> Any:
    if isinstance(obj, str):
        if key in _NORMALISED:
            return _NORMALISED[key]
        return obj if _keep(key, obj) else SCRUBBED
    if isinstance(obj, list):
        return [_allow(v, key) for v in obj]
    if isinstance(obj, dict):
        return {k: _allow(v, k) for k, v in obj.items()}
    return obj


def _scrub_known(rec: dict[str, Any], known: frozenset[str], tool: str) -> dict[str, Any]:
    kind = rec.get("type")
    if kind not in known:
        raise RecorderError(f"{tool}: unrecognised record type {kind!r}; extend the scrubber")
    out: dict[str, Any] = _allow(rec)
    return out


def scrub_claude(rec: dict[str, Any]) -> dict[str, Any]:
    return _scrub_known(rec, _CLAUDE_TYPES, "claude")


def scrub_codex(rec: dict[str, Any]) -> dict[str, Any]:
    return _scrub_known(rec, _CODEX_TYPES, "codex")


def _git_config(key: str) -> str:
    done = subprocess.run(
        ["git", "config", "--get", key], capture_output=True, text=True, check=False
    )
    return done.stdout.strip()


def _identity() -> dict[str, str]:
    host = socket.gethostname()
    found = {
        "home directory": str(Path.home()),
        "user name": getpass.getuser(),
        "git email": _git_config("user.email"),
        "git name": _git_config("user.name"),
        "hostname": host,
        "short hostname": host.split(".")[0],
    }
    return {label: value for label, value in found.items() if len(value) >= 4}


def _replacer(cwd: Path) -> Callable[[str], str]:
    pairs = [
        (str(cwd), FIXTURE_CWD),
        (str(cwd).removeprefix("/private"), FIXTURE_CWD),
        (str(Path.home()), FIXTURE_HOME),
    ]
    # Longest first: the user name is a substring of the git email.
    values = sorted(_identity().values(), key=len, reverse=True)
    pairs += [(value, "fixture-user") for value in values]

    def replace(text: str) -> str:
        for old, new in pairs:
            text = text.replace(old, new)
        return text

    return replace


_LEAK_PATTERNS = {
    "an @ sign (email)": re.compile(r"@"),
    "a /Users/ path": re.compile(r"/Users/"),
    "a Windows user path": re.compile(r"\\\\Users\\\\|[A-Za-z]:\\\\"),
    "a /private/ path": re.compile(r"/private/"),
    "a /var/folders/ path": re.compile(r"/var/folders/"),
    "a /home/ path for a real user": re.compile(r"/home/(?!user(?![A-Za-z0-9_.-]))[^/\"\\\s]*"),
}
_BASE64ISH = re.compile(r"[A-Za-z0-9+/]{32,}={0,2}")


def _opaque_strings(obj: Any, key: str = "") -> list[str]:
    if isinstance(obj, str):
        if key in _OPAQUE_KEYS:
            return []
        if key in _PREFIXED_ID_KEYS and _PREFIXED_ID.fullmatch(obj):
            return []
        return _BASE64ISH.findall(obj)
    if isinstance(obj, list):
        return [m for v in obj for m in _opaque_strings(v, key)]
    if isinstance(obj, dict):
        return [m for k, v in obj.items() for m in _opaque_strings(v, k)]
    return []


def check_clean(name: str, records: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    for label, pattern in _LEAK_PATTERNS.items():
        if pattern.search(text):
            raise RecorderError(f"{name}: scrubbed output still contains {label}")
    for label, value in _identity().items():
        escaped = json.dumps(value)[1:-1]
        if value in text or escaped in text:
            raise RecorderError(f"{name}: scrubbed output still contains the {label}")
    if any(_opaque_strings(r) for r in records):
        raise RecorderError(f"{name}: scrubbed output still contains a base64-like string")


def write_fixture(
    src: Path,
    dest: Path,
    scrubber: Callable[[dict[str, Any]], dict[str, Any]],
    replace: Callable[[str], str],
) -> None:
    records = [
        scrubber(scrub(json.loads(raw), replace))
        for raw in src.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    check_clean(dest.name, records)
    lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in records]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record the handoff test fixtures.")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "fixtures" / "sessions")
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

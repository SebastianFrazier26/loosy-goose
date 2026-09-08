import json
from pathlib import Path

from loosy_goose.transcript import load_claude_code_jsonl, load_messages_json, split_fences


def _write_jsonl(path: Path, records: list[object]) -> None:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_split_fences_separates_code_and_text() -> None:
    blocks = split_fences("Intro text.\n```python\nx = 1\n```\nOutro.")
    assert [b.kind for b in blocks] == ["text", "code", "text"]
    assert blocks[1].text == "x = 1"
    assert blocks[1].meta == {"lang": "python"}
    assert blocks[0].text == "Intro text."


def test_claude_code_loader_is_tolerant(tmp_path: Path) -> None:
    records: list[object] = [
        {"type": "mode", "mode": "default"},
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": "Fix the bug in `run_job`.\n```py\nrun_job()\n```",
            },
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Look at the file first."},
                    {"type": "text", "text": "Reading it now."},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
                    {"type": "image", "source": {}},
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": [{"type": "text", "text": "line one\nline two"}],
                    }
                ],
            },
        },
        "{this is not json",
        {"type": "file-history-snapshot", "snapshot": {}},
    ]
    path = tmp_path / "s.jsonl"
    _write_jsonl(path, records)

    t = load_claude_code_jsonl(path)

    assert [turn.role for turn in t.turns] == ["user", "assistant", "user"]
    assert [b.kind for b in t.turns[0].blocks] == ["text", "code"]
    assert [b.kind for b in t.turns[1].blocks] == ["thinking", "text", "tool_use"]
    assert t.turns[1].blocks[2].meta["name"] == "Read"
    assert json.loads(t.turns[1].blocks[2].text) == {"file_path": "a.py"}
    assert t.turns[2].blocks[0].kind == "tool_result"
    assert t.turns[2].blocks[0].text == "line one\nline two"
    assert t.meta["line:malformed"] == 1
    assert t.meta["record:mode"] == 1
    assert t.meta["record:file-history-snapshot"] == 1
    assert t.meta["block:image"] == 1
    assert t.text(kinds=["code"]) == "run_job()"
    assert "Reading it now." in t.text()


def test_messages_loader_handles_tool_role_and_dict_wrapper() -> None:
    payload = {
        "messages": [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Run tests."},
            {"role": "assistant", "content": "```sh\npytest\n```"},
            {"role": "tool", "content": "3 passed"},
            {"role": "narrator", "content": "ignored"},
        ]
    }
    t = load_messages_json([payload["messages"]][0], name="fixture")
    assert t.source == "fixture"
    assert [turn.role for turn in t.turns] == ["system", "user", "assistant", "tool"]
    assert t.turns[2].blocks[0].kind == "code"
    assert t.turns[3].blocks[0].kind == "tool_result"
    assert t.meta == {"role:narrator": 1}


def test_messages_loader_maps_openhands_inline_tool_calls() -> None:
    msgs = [
        {"role": "user", "content": "Fix the failing test."},
        {
            "role": "assistant",
            "content": "Let me look.\n<function=bash>\n<parameter=cmd>ls</parameter>\n</function>",
        },
        {"role": "user", "content": "EXECUTION RESULT of [bash]:\nREADME.md\nsrc"},
        {"role": "assistant", "content": "Done."},
    ]
    t = load_messages_json(msgs)
    kinds = [(turn.role, [b.kind for b in turn.blocks]) for turn in t.turns]
    assert kinds == [
        ("user", ["text"]),
        ("assistant", ["text", "tool_use"]),
        ("user", ["tool_result"]),
        ("assistant", ["text"]),
    ]
    assert t.turns[1].blocks[1].meta == {"name": "bash"}
    assert t.turns[2].blocks[0].text == "README.md\nsrc"


def test_messages_loader_reads_file(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps([{"role": "user", "content": "hi"}]), encoding="utf-8")
    t = load_messages_json(path)
    assert t.source == str(path)
    assert t.turns[0].blocks[0].text == "hi"

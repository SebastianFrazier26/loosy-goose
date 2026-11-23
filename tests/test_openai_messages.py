from __future__ import annotations

import json

from loosy_goose.openai_messages import system_prefix, to_transcript


def _call(name: str, arguments: str) -> dict[str, object]:
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_tool_calls_with_null_content_become_tool_use() -> None:
    msgs = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [_call("read", '{"b": 2, "a": 1}')]},
    ]
    t, _ = to_transcript(msgs)
    assert len(t.turns) == 2
    (block,) = t.turns[1].blocks
    assert block.kind == "tool_use"
    assert block.meta == {"name": "read"}
    assert block.text == json.dumps({"a": 1, "b": 2}, sort_keys=True)


def test_text_precedes_tool_calls() -> None:
    msgs = [
        {"role": "assistant", "content": "looking", "tool_calls": [_call("ls", "{}")]},
    ]
    t, _ = to_transcript(msgs)
    assert [b.kind for b in t.turns[0].blocks] == ["text", "tool_use"]


def test_tool_message_becomes_tool_result() -> None:
    t, _ = to_transcript([{"role": "tool", "tool_call_id": "c1", "content": "file body"}])
    assert t.turns[0].role == "tool"
    (block,) = t.turns[0].blocks
    assert block.kind == "tool_result"
    assert block.text == "file body"


def test_message_map_skips_system() -> None:
    msgs = [
        {"role": "system", "content": "be good"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    t, index = to_transcript(msgs)
    assert index == [1, 2]
    assert [turn.index for turn in t.turns] == [0, 1]
    assert system_prefix(msgs) == 1


def test_system_prefix_counts_only_leading() -> None:
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "system", "content": "b"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late"},
    ]
    assert system_prefix(msgs) == 2
    _, index = to_transcript(msgs)
    assert index == [2]


def test_list_content_parts_joined() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "one"},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "two"},
            ],
        }
    ]
    t, _ = to_transcript(msgs)
    assert "\n".join(b.text for b in t.turns[0].blocks) == "one\ntwo"


def test_unparseable_arguments_kept_raw() -> None:
    msgs = [{"role": "assistant", "content": None, "tool_calls": [_call("x", "{not json")]}]
    t, _ = to_transcript(msgs)
    assert t.turns[0].blocks[0].text == "{not json"


def test_fences_split_and_empty_messages_dropped_from_map() -> None:
    msgs = [
        {"role": "user", "content": "see\n```py\nx=1\n```"},
        {"role": "assistant", "content": None},
        {"role": "assistant", "content": "ok"},
    ]
    t, index = to_transcript(msgs)
    assert [b.kind for b in t.turns[0].blocks] == ["text", "code"]
    assert index == [0, 2]

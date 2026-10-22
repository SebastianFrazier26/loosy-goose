from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from loosy_goose.rate import LEVELS
from loosy_goose.transcript import Block, Transcript, Turn

ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    # Loaded by path: a package named `site` would shadow the stdlib module of that name.
    spec = importlib.util.spec_from_file_location("build_bundle", ROOT / "site" / "build_bundle.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclass() resolves the defining module through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bb = _load()


def _fake_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(len(texts), 16))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loosy_goose.select.embed_texts", _fake_embeddings)


def _transcript() -> Transcript:
    read = json.dumps({"file_path": "src/app.py"})
    edit = json.dumps({"file_path": "src/app.py", "old_string": "a = 1", "new_string": "a = 2"})
    prose = [f"Paragraph {i} mentions handler_{i} and port {8000 + i}." for i in range(12)]
    return Transcript(
        "synthetic",
        [
            Turn(0, "user", [Block("text", "Fix the bug in src/app.py, then run the tests.", {})]),
            Turn(1, "assistant", [Block("tool_use", read, {"name": "Read"})]),
            Turn(2, "user", [Block("tool_result", "a = 1\nprint(a)\n", {})]),
            Turn(3, "assistant", [Block("text", "\n\n".join(prose), {})]),
            Turn(4, "assistant", [Block("tool_use", edit, {"name": "Edit"})]),
            Turn(5, "user", [Block("tool_result", "The file src/app.py has been updated.", {})]),
            Turn(6, "assistant", [Block("text", "Done: a is now 2 and the tests pass.", {})]),
        ],
    )


def _entry() -> dict[str, Any]:
    source = {"dataset": "synthetic", "url": "https://example.invalid", "license": "MIT"}
    entry: dict[str, Any] = bb.build_entry(_transcript(), id="t", title="T", source=source)
    return entry


def _bundle(entry: dict[str, Any]) -> dict[str, Any]:
    return {"schema": bb.SCHEMA_VERSION, "stamp": bb.stamp(), "transcripts": [entry]}


def test_levels_mirror_the_shipped_rate_table() -> None:
    assert set(bb.LEVEL_ORDER) == set(LEVELS)
    assert [LEVELS[n] for n in bb.LEVEL_ORDER] == sorted(LEVELS.values(), reverse=True)


def test_entry_runs_the_default_selector_and_validates(stub_embed: None) -> None:
    entry = _entry()
    bb.validate_bundle(_bundle(entry))
    assert entry["selector"] == "supersede+leverage+novelty-per-token@0.5"
    budgets = [entry["levels"][n]["budget"] for n in bb.LEVEL_ORDER]
    assert budgets == sorted(budgets, reverse=True)
    for name in bb.LEVEL_ORDER:
        lv = entry["levels"][name]
        assert lv["keep"] == LEVELS[name]
        assert lv["ratio"] == round(entry["input_tokens"] / lv["output_tokens"], 3)


def test_superseded_read_is_flagged_and_never_kept(stub_embed: None) -> None:
    entry = _entry()
    flagged = [s for s in entry["segments"] if s["superseded"]]
    assert flagged, "the read of src/app.py is superseded by the later edit"
    assert entry["superseded_tokens"] == sum(s["tokens"] for s in flagged)
    assert not any(any(s["kept"].values()) for s in flagged)


def test_segments_carry_the_full_original_text_in_order(stub_embed: None) -> None:
    entry = _entry()
    joined = "".join(s["text"] for s in entry["segments"])
    for turn in _transcript().turns:
        for block in turn.blocks:
            for line in block.text.splitlines():
                if line.strip():
                    assert line in joined


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.update(schema=99),
        lambda b: b["transcripts"][0]["source"].pop("license"),
        lambda b: b["transcripts"][0]["segments"][0].update(kind="banner"),
        lambda b: b["transcripts"][0]["segments"][0].update(tokens=-1),
        lambda b: b["transcripts"][0]["levels"]["medium"].update(output_tokens=1),
        lambda b: b["transcripts"][0].update(input_tokens=0),
        lambda b: b["transcripts"][0]["segments"][0]["kept"].pop("heavy"),
        lambda b: b["transcripts"].append(copy.deepcopy(b["transcripts"][0])),
    ],
)
def test_validator_rejects_broken_bundles(stub_embed: None, mutate: Any) -> None:
    bundle = _bundle(_entry())
    mutate(bundle)
    with pytest.raises(ValueError):
        bb.validate_bundle(bundle)


def test_validator_rejects_a_superseded_segment_marked_kept(stub_embed: None) -> None:
    bundle = _bundle(_entry())
    seg = next(s for s in bundle["transcripts"][0]["segments"] if s["superseded"])
    seg["kept"]["light"] = True
    with pytest.raises(ValueError, match="superseded yet kept"):
        bb.validate_bundle(bundle)


def test_build_bundle_refuses_a_row_whose_content_changed() -> None:
    rows: list[Any] = [[{"role": "user", "content": "x"}]] * (max(d.row for d in bb.DEMOS) + 1)
    with pytest.raises(ValueError, match="changed"):
        bb.build_bundle(rows)


def test_demo_pins_match_the_public_fetch_revision() -> None:
    text = (ROOT / "experiments" / "fetch_public.py").read_text(encoding="utf-8")
    assert bb.DATASET in text and bb.DATASET_REVISION in text
    assert all(re.fullmatch(r"[0-9a-f]{64}", d.sha256) for d in bb.DEMOS)


def test_bundle_types_in_the_site_name_every_field_the_builder_emits(stub_embed: None) -> None:
    ts = (ROOT / "site" / "src" / "bundle.ts").read_text(encoding="utf-8")
    entry = _entry()
    keys = set(entry) | set(entry["segments"][0]) | set(entry["levels"]["light"])
    keys |= set(bb.stamp()) | {"schema", "stamp", "transcripts"}
    for key in keys:
        assert re.search(rf"\b{key}\??:", ts), key


def test_a_locally_built_bundle_validates() -> None:
    path = ROOT / "site" / "public" / "bundle.json"
    if not path.exists():
        pytest.skip("no local bundle; pages.yml builds it at deploy time")
    bb.validate_bundle(json.loads(path.read_text(encoding="utf-8")))

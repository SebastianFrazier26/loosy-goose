from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import exp_d_curves as exp_d  # noqa: E402
from experiments import exp_g_novelty as exp_g  # noqa: E402
from experiments.exp_d_curves import BASE_VARIANT, KEEP_RATIOS, RunContext  # noqa: E402
from loosy_goose.budget import token_budget  # noqa: E402
from loosy_goose.code import quantize_segment  # noqa: E402
from loosy_goose.metrics import atom_recall_final  # noqa: E402
from loosy_goose.paths import PathTable, build_table  # noqa: E402
from loosy_goose.pipeline import (  # noqa: E402
    Compressed,
    PipelineOptions,
    compress_transcript,
    selector_label,
    to_turns,
)
from loosy_goose.segment import Segment, SegmentKind, segment  # noqa: E402
from loosy_goose.select import CompressConfig  # noqa: E402
from loosy_goose.supersede import apply_supersession  # noqa: E402
from loosy_goose.tokens import count_tokens  # noqa: E402
from loosy_goose.transcript import Block, Transcript, Turn  # noqa: E402

_CODE = "def f(x):\n    if x:\n        for i in range(3):\n            print(i)\n    return x\n"
# The read's dump names a path nothing else in the transcript mentions, so whether it reaches
# the path table tells which segment list the table was built from.
_DUMP = "\n".join(f"{i:6d}\u2192line {i} of notes/only_here.md" for i in range(1, 9))


def _fake_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(len(texts), 16))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loosy_goose.select.embed_texts", _fake_embeddings)


def _seg(i: int, turn: int, kind: SegmentKind, text: str) -> Segment:
    return Segment(id=i, turn=turn, role="assistant", kind=kind, text=text, protected=False)


def _transcript() -> Transcript:
    """A read of `src/a.py` followed by an edit of it: supersession drops the read and its dump."""
    read = json.dumps({"file_path": "src/a.py"})
    edit = json.dumps({"file_path": "src/a.py", "old_string": "x", "new_string": "y"})
    return Transcript(
        "t",
        [
            Turn(0, "user", [Block("text", "Please read src/a.py and fix it.")]),
            Turn(1, "assistant", [Block("tool_use", read, {"name": "Read"})]),
            Turn(2, "user", [Block("tool_result", _DUMP)]),
            Turn(
                3,
                "assistant",
                [Block("text", "Editing src/a.py now."), Block("tool_use", edit, {"name": "Edit"})],
            ),
            Turn(4, "user", [Block("tool_result", "ok")]),
            Turn(5, "assistant", [Block("text", "Here is the helper."), Block("code", _CODE)]),
        ],
    )


def _kept_tokens(c: Compressed) -> int:
    return sum(count_tokens(s.text) for s in c.segments)


def test_budget_comes_from_original_segments(stub_embed: None) -> None:
    t = _transcript()
    original = segment(t)
    superseded = apply_supersession(original)
    assert len(superseded) < len(original)
    c = compress_transcript(t, PipelineOptions(keep=0.5))
    assert c.budget == token_budget(original, 0.5)
    assert c.budget > token_budget(superseded, 0.5)
    assert c.input_tokens == sum(count_tokens(s.text) for s in original)
    gone = {s.id for s in original} - {s.id for s in superseded}
    assert gone and not gone & {s.id for s in c.segments}
    assert _kept_tokens(c) <= c.budget


def test_band_applies_quantize_segment_at_keep(stub_embed: None) -> None:
    t = _transcript()
    by_id = {s.id: s for s in segment(t)}
    c = compress_transcript(t, PipelineOptions(keep=0.5, band=True))
    banded = [s for s in c.segments if s.kind in ("code", "tool_use")]
    assert banded
    for s in banded:
        assert s.text == quantize_segment(by_id[s.id], quality=0.5).text
    assert any(s.text != by_id[s.id].text for s in banded)
    assert all(s.text == by_id[s.id].text for s in c.segments if s.kind == "prose")


def test_paths_table_is_built_from_original_segments(stub_embed: None) -> None:
    t = _transcript()
    c = compress_transcript(t, PipelineOptions(keep=0.5, paths=True, min_mentions=1))
    assert isinstance(c.table, PathTable)
    assert c.table.paths == build_table(segment(t), min_mentions=1).paths
    assert "notes/only_here.md" in c.table.paths
    assert not any("notes/only_here.md" in s.text for s in c.segments)
    assert any(c.table.marker in s.text for s in c.segments)
    assert not any("src/a.py" in s.text for s in c.segments)
    assert c.output_tokens == _kept_tokens(c) + c.table.tokens()


def test_no_paths_means_no_table(stub_embed: None) -> None:
    c = compress_transcript(_transcript(), PipelineOptions(keep=0.5))
    assert c.table is None
    assert c.output_tokens == _kept_tokens(c)
    assert any("src/a.py" in s.text for s in c.segments)


def test_to_turns_groups_by_source_turn_in_id_order() -> None:
    t = Transcript(
        "t",
        [
            Turn(0, "assistant", [Block("text", "a"), Block("code", "b")]),
            Turn(1, "user", [Block("tool_result", "dropped")]),
            Turn(2, "user", [Block("tool_result", "c")]),
            Turn(3, "assistant", [Block("thinking", "d"), Block("tool_use", "{}")]),
        ],
    )
    kept = [
        _seg(5, 3, "tool_use", "{}"),
        _seg(3, 2, "tool_result", "c"),
        _seg(1, 0, "code", "b"),
        _seg(0, 0, "prose", "a"),
        _seg(4, 3, "thinking", "d"),
    ]
    turns = to_turns(Compressed(kept, None, 0, 0, 0, 1.0), t)
    assert [(x.index, x.role) for x in turns] == [(0, "assistant"), (2, "user"), (3, "assistant")]
    assert [[(b.kind, b.text) for b in x.blocks] for x in turns] == [
        [("text", "a"), ("code", "b")],
        [("tool_result", "c")],
        [("thinking", "d"), ("tool_use", "{}")],
    ]
    assert all(b.meta == {} for x in turns for b in x.blocks)
    assert to_turns(Compressed([], None, 0, 0, 0, 1.0), t) == []


def test_strategy_threads_into_compress_config(
    monkeypatch: pytest.MonkeyPatch, stub_embed: None
) -> None:
    seen: list[CompressConfig] = []

    def capture(x: np.ndarray, config: CompressConfig) -> np.ndarray:
        seen.append(config)
        return np.ones(x.shape[0])

    monkeypatch.setattr("loosy_goose.pipeline.score_segments", capture)
    compress_transcript(_transcript(), PipelineOptions(keep=0.5, strategy="ridge"))
    assert [c.strategy for c in seen] == ["ridge"]
    assert seen[0] == CompressConfig(strategy="ridge")


def test_explicit_budget_overrides_keep_but_not_band_depth(stub_embed: None) -> None:
    t = _transcript()
    derived = compress_transcript(t, PipelineOptions(keep=0.5, band=True))
    same = compress_transcript(t, PipelineOptions(keep=0.5, budget=derived.budget, band=True))
    assert same == derived
    tight = compress_transcript(t, PipelineOptions(keep=0.5, budget=20, band=True))
    assert tight.budget == 20 and _kept_tokens(tight) <= 20
    by_id = {s.id: s for s in segment(t)}
    for s in tight.segments:
        if s.kind in ("code", "tool_use"):
            assert s.text == quantize_segment(by_id[s.id], quality=0.5).text


def test_recall_is_reported_against_final_state_atoms(stub_embed: None) -> None:
    t = _transcript()
    c = compress_transcript(t, PipelineOptions(keep=0.5))
    assert c.atom_recall_final == atom_recall_final(segment(t), c.segments)
    with_table = compress_transcript(t, PipelineOptions(keep=0.5, paths=True, min_mentions=1))
    assert with_table.table is not None
    assert with_table.atom_recall_final == atom_recall_final(
        segment(t), with_table.segments, guaranteed=with_table.table.guaranteed_atoms()
    )


def _prose_transcript() -> Transcript:
    texts = [
        f"Paragraph {i} explains module_{i % 4} and calls helper_{i * 7} with value {i * 13}; "
        f"{' '.join(['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot'][i % 6 :])}."
        for i in range(10)
    ]
    return Transcript("p", [Turn(i, "assistant", [Block("text", t)]) for i, t in enumerate(texts)])


@pytest.mark.parametrize("ratio", KEEP_RATIOS)
@pytest.mark.parametrize("make", [_transcript, _prose_transcript])
def test_default_selection_is_the_exp_g_per_token_arm(
    make: Callable[[], Transcript], ratio: float, tmp_path: Path, stub_embed: None
) -> None:
    t = make()
    original = segment(t)
    ctx = RunContext(tmp_path, BASE_VARIANT, build_table(original))
    want = exp_g._run_novelty(0.5, per_token=True)(original, ratio, "none", ctx)
    got = compress_transcript(t, PipelineOptions(keep=ratio))
    assert [s.id for s in got.segments] == [s.id for s in want]
    assert selector_label(PipelineOptions(keep=ratio)) in exp_g.METHODS


@pytest.mark.parametrize("ratio", KEEP_RATIOS)
@pytest.mark.parametrize("make", [_transcript, _prose_transcript])
def test_no_novelty_is_the_exp_d_supersede_leverage_method(
    make: Callable[[], Transcript], ratio: float, tmp_path: Path, stub_embed: None
) -> None:
    t = make()
    original = segment(t)
    ctx = RunContext(tmp_path, BASE_VARIANT, build_table(original))
    want = exp_d.METHODS["supersede+leverage"](original, ratio, "none", ctx)
    opts = PipelineOptions(keep=ratio, novelty=False)
    got = compress_transcript(t, opts)
    assert [s.id for s in got.segments] == [s.id for s in want]
    assert selector_label(opts) == "supersede+leverage"


@pytest.mark.parametrize(
    ("band", "paths", "label"),
    [
        (True, False, "supersede+band+leverage"),
        (False, True, "supersede+leverage [paths=full+mm=1]"),
        (True, True, "supersede+band+leverage [paths=full+mm=1]"),
    ],
)
def test_band_and_paths_ignore_novelty(
    band: bool, paths: bool, label: str, stub_embed: None
) -> None:
    t = _transcript()
    on = PipelineOptions(keep=0.5, band=band, paths=paths, min_mentions=1)
    off = PipelineOptions(keep=0.5, band=band, paths=paths, min_mentions=1, novelty=False)
    assert compress_transcript(t, on) == compress_transcript(t, off)
    assert selector_label(on) == selector_label(off) == label
    assert label.split(" [")[0] in exp_d.METHODS

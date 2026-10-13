import json
from pathlib import Path

import numpy as np
import pytest

from loosy_goose.cli import LEVELS, build_parser, main, resolve_rate
from loosy_goose.select import leverage_scores, rank_for_energy, svd_energy
from loosy_goose.transcript import HEADER_KEY, load_blocks_jsonl

_EMBED_DIM = 3


def _fake_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    rng = np.random.default_rng(11)
    # Dim must stay well below len(texts) - 1: at full centred rank every leverage score is
    # identical up to LAPACK roundoff, and min-max normalisation turns that noise into the ordering.
    x = rng.normal(size=(len(texts), _EMBED_DIM))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loosy_goose.select.embed_texts", _fake_embeddings)


def test_stub_embedding_leverage_scores_are_well_separated() -> None:
    x = _fake_embeddings(_CONTENTS, "stub")
    u, s, _, _ = svd_energy(x)
    k = rank_for_energy(s, 0.95)
    assert k < len(_CONTENTS) - 1
    gaps = np.diff(np.sort(leverage_scores(u, k)))
    assert gaps.min() > 1e-3


def _write_messages(path: Path, contents: list[str]) -> None:
    roles = ("user", "assistant")
    messages = [{"role": roles[i % 2], "content": c} for i, c in enumerate(contents)]
    path.write_text(json.dumps(messages), encoding="utf-8")


def _header(path: Path) -> dict[str, object]:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    header = json.loads(first)[HEADER_KEY]
    assert isinstance(header, dict)
    return header


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
_CONTENTS_TOKENS = 120


def _run(tmp_path: Path, *flags: str) -> tuple[dict[str, object], list[str]]:
    src = tmp_path / "chat.json"
    _write_messages(src, _CONTENTS)
    out = tmp_path / "out.jsonl"
    assert main(["compress", str(src), *flags, "--output", str(out)]) == 0
    return _header(out), out.read_text(encoding="utf-8").splitlines()[1:]


def test_levels_map_to_documented_ratios() -> None:
    assert LEVELS == {"light": 0.50, "medium": 0.33, "heavy": 0.25}
    for name, ratio in LEVELS.items():
        rate = resolve_rate(None, name, None, 0)
        assert (rate.keep, rate.budget, rate.source, rate.level) == (ratio, None, "level", name)


def test_default_is_level_medium() -> None:
    args = build_parser().parse_args(["compress", "session.jsonl"])
    rate = resolve_rate(args.keep, args.level, args.target_tokens, 0)
    assert (rate.keep, rate.source, rate.level, rate.notes) == (0.33, "level", "medium", ())


def test_keep_rounds_to_hundredths_with_note() -> None:
    rate = resolve_rate(0.333, None, None, 0)
    assert rate.keep == 0.33 and rate.source == "keep"
    assert len(rate.notes) == 1 and "0.33" in rate.notes[0]
    assert resolve_rate(0.42, None, None, 0).notes == ()


def test_keep_rounding_note_reaches_stderr(
    tmp_path: Path, stub_embed: None, capsys: pytest.CaptureFixture[str]
) -> None:
    header, _ = _run(tmp_path, "--keep", "0.333")
    assert header["keep"] == 0.33
    assert "using 0.33" in capsys.readouterr().err


def test_target_tokens_in_range_is_exact_budget() -> None:
    rate = resolve_rate(None, None, 50, 200)
    assert (rate.budget, rate.keep, rate.source, rate.notes) == (50, 0.25, "target-tokens", ())


@pytest.mark.parametrize(("target", "clamped"), [(5, 0.1), (190, 0.8)])
def test_target_tokens_out_of_range_clamps_with_warning(target: int, clamped: float) -> None:
    rate = resolve_rate(None, None, target, 200)
    assert rate.keep == clamped and rate.budget is None and rate.target_tokens == target
    (warning,) = rate.notes
    assert f"{target / 200:.4f}" in warning and f"{clamped:.2f}" in warning


def test_target_tokens_clamp_warning_reaches_stderr(
    tmp_path: Path, stub_embed: None, capsys: pytest.CaptureFixture[str]
) -> None:
    header, _ = _run(tmp_path, "--target-tokens", "1")
    assert header["keep"] == 0.1 and header["budget"] == 12
    assert "clamped to keep 0.10" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["0", "-3", "1.5"])
def test_target_tokens_rejects_non_positive_int(bad: str) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["compress", "s.jsonl", "--target-tokens", bad])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "pair",
    [
        ["--keep", "0.5", "--level", "light"],
        ["--keep", "0.5", "--target-tokens", "10"],
        ["--level", "heavy", "--target-tokens", "10"],
    ],
)
def test_rate_flags_are_mutually_exclusive(pair: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["compress", "s.jsonl", *pair])
    assert exc.value.code == 2


def test_quality_is_rejected_as_unknown(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compress", "session.jsonl", "--quality", "0.6"])
    assert exc.value.code == 2
    assert "unrecognized arguments: --quality" in capsys.readouterr().err


def test_header_reports_rate_mode_and_recall(tmp_path: Path, stub_embed: None) -> None:
    header, _ = _run(tmp_path, "--level", "heavy")
    assert (header["keep"], header["keep_source"], header["level"]) == (0.25, "level", "heavy")
    assert header["target_tokens"] is None
    recall = header["atom_recall_final"]
    assert isinstance(recall, float) and 0.0 <= recall <= 1.0

    header, _ = _run(tmp_path, "--target-tokens", "48")
    assert (header["budget"], header["keep_source"], header["level"]) == (48, "target-tokens", None)
    assert header["keep"] == 48 / _CONTENTS_TOKENS and header["target_tokens"] == 48


def test_no_novelty_keep_half_output_matches_pre_level_control(
    tmp_path: Path, stub_embed: None
) -> None:
    # Pinned from the CLI as it stood before --level/--target-tokens existed (exp/compress-select
    # at the branch point), same fixture and stubbed embeddings.
    header, lines = _run(tmp_path, "--keep", "0.5", "--no-novelty")
    assert header["selector"] == "supersede+leverage" and header["novelty"] is False
    assert (header["budget"], header["input_tokens"], header["output_tokens"]) == (60, 120, 59)
    assert [json.loads(line)["index"] for line in lines] == [1, 2, 4, 5]
    assert json.loads(lines[1])["blocks"] == [
        {"kind": "text", "text": _CONTENTS[2], "meta": {}},
    ]


def test_default_keep_half_output_is_pinned(tmp_path: Path, stub_embed: None) -> None:
    header, lines = _run(tmp_path, "--keep", "0.5")
    assert header["selector"] == "supersede+leverage+novelty-per-token@0.5"
    assert header["novelty"] is True
    # Observed when novelty became the default, same fixture and stubbed embeddings.
    assert (header["budget"], header["input_tokens"], header["output_tokens"]) == (60, 120, 50)
    assert [json.loads(line)["index"] for line in lines] == [1, 2, 4, 6]


@pytest.mark.parametrize(
    ("flags", "selector"),
    [
        (["--band"], "supersede+band+leverage"),
        (["--paths"], "supersede+leverage [paths=full]"),
        (["--paths", "--min-mentions", "1"], "supersede+leverage [paths=full+mm=1]"),
        (["--band", "--paths"], "supersede+band+leverage [paths=full]"),
    ],
)
def test_band_and_paths_fall_back_to_the_leverage_fill(
    tmp_path: Path, stub_embed: None, flags: list[str], selector: str
) -> None:
    header, lines = _run(tmp_path, "--keep", "0.5", *flags)
    control, control_lines = _run(tmp_path, "--keep", "0.5", *flags, "--no-novelty")
    assert header["selector"] == control["selector"] == selector
    assert lines == control_lines
    assert header["output_tokens"] == control["output_tokens"]


@pytest.mark.parametrize("bad", ["0.05", "0.9", "abc"])
def test_compress_rejects_bad_keep(bad: str) -> None:
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["compress", "session.jsonl", "--keep", bad])
    assert exc.value.code == 2


@pytest.mark.parametrize("good", ["0.1", "0.8"])
def test_compress_accepts_keep_bounds(good: str) -> None:
    args = build_parser().parse_args(["compress", "session.jsonl", "--keep", good])
    assert args.keep == float(good)


def test_min_mentions_rejects_zero_accepts_one() -> None:
    base = ["compress", "s.jsonl", "--keep", "0.5", "--min-mentions"]
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "0"])
    assert build_parser().parse_args([*base, "1"]).min_mentions == 1


def test_missing_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_unknown_suffix_is_reported_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "foo.txt"
    path.write_text("hello", encoding="utf-8")
    assert main(["compress", str(path), "--keep", "0.5"]) == 2
    captured = capsys.readouterr()
    assert "unsupported transcript suffix" in captured.err
    assert captured.out == ""


def test_compress_smoke_writes_header_and_turns(tmp_path: Path, stub_embed: None) -> None:
    src = tmp_path / "chat.json"
    _write_messages(
        src,
        [
            "Please look at the failing test and tell me what you see.",
            "The assertion compares floats exactly; a tolerance is needed.",
            "Add the tolerance and rerun the suite for me.",
            "Done, the suite is green again with pytest.approx in place.",
        ],
    )
    out = tmp_path / "out.jsonl"
    assert main(["compress", str(src), "--keep", "0.5", "--output", str(out)]) == 0
    header = _header(out)
    assert header["keep"] == 0.5
    assert header["table"] is None
    assert header["paths"] is False and header["band"] is False
    assert len(out.read_text(encoding="utf-8").splitlines()) >= 2
    back = load_blocks_jsonl(out)
    assert back.meta == {"record:header": 1}
    assert len(back.turns) >= 1
    assert all(t.role in ("user", "assistant") for t in back.turns)


def test_compress_smoke_with_paths_emits_table(tmp_path: Path, stub_embed: None) -> None:
    src = tmp_path / "chat.json"
    # Every segment costs fewer tokens than the two-row table, so whatever greedy selection
    # leaves unfilled under the budget is smaller than the table it must then add on top.
    _write_messages(
        src,
        [
            "Open src/app/main.py.",
            "Reading src/app/main.py now.",
            "Also check tests/test_main.py.",
            "tests/test_main.py looks fine.",
            "Then fix src/app/main.py.",
            "Done with src/app/main.py.",
        ],
    )
    out = tmp_path / "out.jsonl"
    argv = ["compress", str(src), "--keep", "0.5", "--paths", "--min-mentions", "1"]
    assert main([*argv, "--output", str(out)]) == 0
    header = _header(out)
    table = header["table"]
    assert isinstance(table, str) and table
    assert "src/app/main.py" in table and "tests/test_main.py" in table
    assert isinstance(header["output_tokens"], int) and isinstance(header["budget"], int)
    assert header["output_tokens"] >= header["budget"]
    back = load_blocks_jsonl(out)
    assert back.meta == {"record:header": 1}
    assert any("[P" in b.text for t in back.turns for b in t.blocks)


def test_compress_smoke_stdout(
    tmp_path: Path, stub_embed: None, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "chat.json"
    _write_messages(src, ["Résumé → done.", "Noted → the arrow survived."])
    assert main(["compress", str(src), "--keep", "0.8"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert HEADER_KEY in json.loads(lines[0])
    assert any("→" in line for line in lines[1:])

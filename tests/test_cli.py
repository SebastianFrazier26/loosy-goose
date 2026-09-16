import json
from pathlib import Path

import numpy as np
import pytest

from loosy_goose.cli import build_parser, main
from loosy_goose.transcript import HEADER_KEY, load_blocks_jsonl


def _fake_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(len(texts), 16))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


@pytest.fixture
def stub_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loosy_goose.select.embed_texts", _fake_embeddings)


def _write_messages(path: Path, contents: list[str]) -> None:
    roles = ("user", "assistant")
    messages = [{"role": roles[i % 2], "content": c} for i, c in enumerate(contents)]
    path.write_text(json.dumps(messages), encoding="utf-8")


def _header(path: Path) -> dict[str, object]:
    first = path.read_text(encoding="utf-8").splitlines()[0]
    header = json.loads(first)[HEADER_KEY]
    assert isinstance(header, dict)
    return header


def test_compress_requires_keep() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compress", "session.jsonl"])
    assert exc.value.code == 2


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


@pytest.mark.parametrize("bad", ["1.5", "-0.1", "abc"])
def test_compress_rejects_bad_quality(bad: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compress", "session.jsonl", "--quality", bad])
    assert exc.value.code != 0


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

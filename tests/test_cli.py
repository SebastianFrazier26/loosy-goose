import pytest

from loosy_goose.cli import main


def test_compress_stub_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compress", "session.jsonl"]) == 2
    assert "not implemented" in capsys.readouterr().err


def test_compress_accepts_quality_in_range() -> None:
    assert main(["compress", "session.jsonl", "--quality", "0.25"]) == 2


@pytest.mark.parametrize("bad", ["1.5", "-0.1", "abc"])
def test_compress_rejects_bad_quality(bad: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compress", "session.jsonl", "--quality", bad])
    assert exc.value.code != 0


def test_missing_subcommand_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0

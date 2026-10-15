import pytest

from loosy_goose import cli, rate


def test_cli_uses_the_library_rate_table() -> None:
    assert cli.LEVELS is rate.LEVELS
    assert cli.resolve_rate is rate.resolve_rate
    assert rate.LEVELS == {"light": 0.50, "medium": 0.33, "heavy": 0.25}
    assert (rate.KEEP_MIN, rate.KEEP_MAX, rate.DEFAULT_LEVEL) == (0.1, 0.8, "medium")


@pytest.mark.parametrize("k", [0.1, 0.33, 0.8])
def test_check_keep_accepts_the_bounds(k: float) -> None:
    assert rate.check_keep(k) == k


@pytest.mark.parametrize("k", [0.09, 0.81])
def test_check_keep_rejects_outside_the_bounds(k: float) -> None:
    with pytest.raises(ValueError, match=r"keep must be in \[0.1, 0.8\]"):
        rate.check_keep(k)

import tomllib
from pathlib import Path


def test_live_tests_are_registered_and_opt_in() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    options = tomllib.loads(pyproject.read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    assert options["addopts"] == "-m 'not live'"
    assert any(m.startswith("live:") for m in options["markers"])

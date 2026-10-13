import json
import re

import pytest
from session_fixtures import (
    CLAUDE_FIXTURE,
    CODE_WORDS,
    CODEX_FIXTURE,
    FIXTURE_CWD,
    FIXTURE_HOME,
    FIXTURES,
    SENTINEL,
)


@pytest.mark.parametrize("name", [CLAUDE_FIXTURE, CODEX_FIXTURE])
def test_fixture_is_synthetic_and_scrubbed(name: str) -> None:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    for word in CODE_WORDS:
        assert word in text
    assert SENTINEL in text
    assert FIXTURE_CWD in text
    for leak in ("/Users/", "\\\\Users\\\\", "/private/", "/var/folders/", "@"):
        assert leak not in text
    assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    assert not re.search(r"[A-Za-z]:\\\\", text)
    assert set(re.findall(r'/home/[^/"\\]+', text)) <= {FIXTURE_HOME}
    for line in text.splitlines():
        assert isinstance(json.loads(line), dict)


def test_claude_fixture_recorded_the_slash_command() -> None:
    text = (FIXTURES / CLAUDE_FIXTURE).read_text(encoding="utf-8")
    assert "<command-name>/loosy</command-name>" in text

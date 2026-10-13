import re
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def test_ci_installs_the_mcp_extra() -> None:
    assert "uv sync --dev --extra embed --extra mcp" in (WORKFLOWS / "ci.yml").read_text("utf-8")


_USES = re.compile(r"uses:\s*(\S+)")


@pytest.mark.parametrize("name", ["ci.yml", "release.yml"])
def test_every_action_is_pinned_to_an_exact_version(name: str) -> None:
    for ref in _USES.findall((WORKFLOWS / name).read_text(encoding="utf-8")):
        if ref.startswith("./"):
            continue
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+\.\d+\.\d+", ref), ref


def test_ci_can_be_called_by_the_release() -> None:
    assert "workflow_call:" in (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")


def test_release_tests_builds_checks_cpu_torch_and_publishes_on_tags() -> None:
    text = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert re.search(r'tags:\s*\["v\*"\]', text)
    assert "uses: ./.github/workflows/ci.yml" in text
    assert "uv build" in text
    assert "--torch-backend cpu" in text and "+cpu" in text
    assert "gh release create" in text and "contents: write" in text
    for path in (
        "dist/*",
        "integrations/claude/loosy.md",
        "integrations/codex/loosy/SKILL.md",
        "integrations/codex/loosy/agents/openai.yaml",
        "integrations/README.md",
    ):
        assert path in text

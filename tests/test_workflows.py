import re
from pathlib import Path
from typing import Any

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


def _release() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / "release.yml").read_text("utf-8"))
    return loaded


def test_release_publishes_only_after_the_tests_and_scopes_the_write_token() -> None:
    wf = _release()
    assert wf["permissions"] == {"contents": "read"}
    jobs = wf["jobs"]
    assert jobs["release"]["needs"] == "test"
    assert jobs["release"]["permissions"] == {"contents": "write"}
    others = {name: job for name, job in jobs.items() if name != "release"}
    assert all("write" not in str(job.get("permissions", {})) for job in others.values())


def test_release_checkout_does_not_persist_the_write_token() -> None:
    # The token would otherwise sit in .git/config while uv build and third-party packages run;
    # gh release create reads GH_TOKEN from the environment instead.
    steps = _release()["jobs"]["release"]["steps"]
    checkouts = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@")]
    assert checkouts and all(
        s.get("with", {}).get("persist-credentials") is False for s in checkouts
    )


def _pages() -> dict[str, Any]:
    yaml = pytest.importorskip("yaml")
    loaded: dict[str, Any] = yaml.safe_load((WORKFLOWS / "pages.yml").read_text("utf-8"))
    return loaded


def test_pages_actions_are_pinned_by_commit_sha() -> None:
    for ref in _USES.findall((WORKFLOWS / "pages.yml").read_text(encoding="utf-8")):
        if ref.startswith("./"):
            continue
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), ref


def test_pages_deploys_main_after_the_tests_with_scoped_permissions() -> None:
    wf = _pages()
    # PyYAML reads the bare `on:` key as boolean True.
    on = wf.get("on", wf.get(True))
    assert on["push"]["branches"] == ["main"] and "workflow_dispatch" in on
    assert wf["permissions"] == {"contents": "read"}
    jobs = wf["jobs"]
    assert jobs["test"]["uses"] == "./.github/workflows/ci.yml"
    assert jobs["build"]["needs"] == "test" and jobs["deploy"]["needs"] == "build"
    assert jobs["deploy"]["permissions"] == {"pages": "write", "id-token": "write"}
    steps = jobs["build"]["steps"]
    runs = [s.get("run", "") for s in steps]
    assert any("site/build_bundle.py" in r for r in runs)
    # Enabling Pages is the owner's call; configure-pages must not flip the repo setting.
    for s in steps:
        if str(s.get("uses", "")).startswith("actions/configure-pages@"):
            assert "enablement" not in s.get("with", {})


def test_ci_typechecks_and_builds_the_site() -> None:
    text = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    assert "working-directory: site" in text and "npm run build" in text


def test_demo_page_can_be_framed() -> None:
    html = (WORKFLOWS.parents[1] / "site" / "index.html").read_text(encoding="utf-8").lower()
    assert "frame-ancestors" not in html and "x-frame-options" not in html

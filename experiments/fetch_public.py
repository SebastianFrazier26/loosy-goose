"""Download the public corpora used by Phase 1 into data/public/<source>/.

Usage: uv run python experiments/fetch_public.py [--only trace-commons|swe-gym|oasst2]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data" / "public"


@dataclass(frozen=True)
class Source:
    key: str
    repo_id: str
    license: str
    allow_patterns: tuple[str, ...]


SOURCES: tuple[Source, ...] = (
    Source(
        "trace-commons",
        "trace-commons/agent-traces",
        "CC-BY-4.0",
        ("sessions/claude_code/*", "README.md", "LICENSE*"),
    ),
    Source(
        "swe-gym",
        "SWE-Gym/OpenHands-SFT-Trajectories",
        "MIT",
        ("*.parquet", "README.md"),
    ),
    Source(
        "oasst2",
        "OpenAssistant/oasst2",
        "Apache-2.0",
        ("data/validation-*.parquet", "README.md"),
    ),
)


def fetch(src: Source) -> str:
    target = PUBLIC / src.key
    target.mkdir(parents=True, exist_ok=True)
    local = snapshot_download(
        repo_id=src.repo_id,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=list(src.allow_patterns),
    )
    info = HfApi().dataset_info(src.repo_id)
    sha = info.sha or "unknown"
    print(f"{src.key}: {src.repo_id} @ {sha} -> {local}")
    return sha


def write_sources_md(rows: list[tuple[Source, str]]) -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Public sources",
        "",
        "Downloaded by `experiments/fetch_public.py`. Not committed.",
        "",
        "| key | dataset | license | revision |",
        "|---|---|---|---|",
    ]
    lines += [f"| {s.key} | {s.repo_id} | {s.license} | {sha} |" for s, sha in rows]
    (PUBLIC / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=[s.key for s in SOURCES], action="append")
    args = parser.parse_args()
    wanted = [s for s in SOURCES if not args.only or s.key in args.only]
    rows = [(s, fetch(s)) for s in wanted]
    write_sources_md(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

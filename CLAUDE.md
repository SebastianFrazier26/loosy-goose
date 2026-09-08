# Loosy-Goose — agent guide

Lossy text compression for LLM context compaction via eigen-dimension semantic spaces.
Personal GitHub project (public, MIT). Not related to any employer repository.

## Commands

- `uv sync` — install (Python 3.12 pinned via `.python-version`)
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format --check .` — lint + format check
- `uv run mypy src` — type check (strict)
- `uv run loosy-goose compress <transcript> --quality 0.6` — CLI (stub until core lands)

## Layout

- `src/loosy_goose/spectral.py` — PPMI, inverse participation ratio, truncated SVD
- `src/loosy_goose/cli.py` — argparse entry point
- `tests/` — pytest
- `experiments/` — Phase 1 scripts; outputs in `experiments/output/` (gitignored)
- `data/local/` — user's own session logs, gitignored, **never committed**
- `data/public/` — public fixtures
- `.claude/` — local-only personal agent env, excluded via `.git/info/exclude`, not part of the repo

## Conventions

- Fully typed; `ruff` for lint and format; mypy strict on `src/`.
- Comments only where load-bearing (a decision, a constraint, a non-obvious construct). No narration.
- User-visible changes get a dated line in `CHANGELOG.md`.
- Tests cover the algorithmic core; thin CLI glue gets smoke tests only.

## Algorithm decisions (locked 2026-09-08)

- Hybrid core: Shin-style PPMI-SVD for an interpretable topic basis + embedding-SVD / leverage-score subset selection for stable segment scoring.
- v1 output is extractive only — no LLM summarization in the loop.
- A background prior (generic, later personalized from local sessions) stabilizes small-context SVD; top bias-like eigen-directions are stripped before truncation.
- Protected regions (code, paths, numbers, identifiers, user directives) are kept verbatim.
- Distortion metrics: embedding cosine (original vs compressed) + atom recall (entities, numbers, identifiers).

## Hard rules

- Never commit anything under `data/local/`.
- Never push. `/commitandpush` invoked by the user is the only publish path.
- Design decisions go through the user: propose options, wait.

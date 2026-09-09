# Changelog

## Unreleased

- 2026-09-08 — project scaffold: uv project, `spectral` core seed (PPMI, IPR, truncated SVD), CLI stub, CI, docs.
- 2026-09-08 — Phase 1 foundation: `transcript` loaders (Claude Code JSONL, generic messages), `segment` with protected units and atom extraction, `tokens` (tiktoken o200k), corpus scripts (`fetch_public`, `pick_local`, `corpus_stats`), `.gitattributes` LF normalisation, optional `embed` extra installed locally.
- 2026-09-08 — Phase 1 experiments: shared token-budget contract (`budget`), distortion metrics (`metrics`, `embed`), baselines; Experiment A (`cooccur`, `topics`), Experiment B (`select`), Experiment C (`code`, `supersede`).
- 2026-09-09 — Phase 1 documented: `docs/DESIGN.md` (algorithm spec, per-kind quality table replacing the binary `protected` flag), `docs/PHASE1.md` (designs, pre-registered criteria, results, known defects), `docs/results/` (committed summary tables). README and CLAUDE.md corrected — they still described the superseded protect-verbatim design.

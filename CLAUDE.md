# Loosy-Goose — agent guide

Lossy text compression for LLM context compaction via eigen-dimension semantic spaces.
Personal GitHub project (public, MIT).

Read [docs/DESIGN.md](docs/DESIGN.md) before changing the algorithm and
[docs/PHASE1.md](docs/PHASE1.md) before re-deciding anything Phase 1 already measured.

## Commands

- `uv sync --dev --extra embed` — install (Python 3.12 pinned via `.python-version`)
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format --check .` — lint + format check
- `uv run mypy src` — type check (strict)
- `uv run loosy-goose compress <transcript> --quality 0.6` — CLI (stub until core lands)

The `embed` extra pulls torch CPU + sentence-transformers for the embedding-SVD work; CI skips
it. Everything is CPU-only. Do not unpin Python 3.12 — 3.13+ lacks torch wheels.

## Layout

Core:

- `spectral.py` — PPMI, inverse participation ratio, dense and sparse truncated SVD
- `transcript.py` — `Transcript`/`Turn`/`Block`; tolerant Claude Code JSONL loader (unknown record/block types are counted in `Transcript.meta`, never fatal) and generic `[{role, content}]` loader
- `segment.py` — `Segment` units: prose split paragraph→sentence, code/tool_use atomic, tool_result line-chunked; `atoms` = identifiers/paths/numbers/URLs for the recall metric
- `tokens.py` — `count_tokens` via tiktoken `o200k_base` (model-agnostic rate proxy)
- `embed.py` — pinned select/score model pair
- `budget.py` — token budget, protect policy, greedy score-based selection. **The shared contract**: every method competes at an identical token budget, or results are not comparable
- `metrics.py` — compression ratio, atom recall, semantic coverage, `evaluate`
- `cli.py` — argparse entry point

Experiment A (topic basis):

- `cooccur.py` — tokenizer, vocab, windowed co-occurrence, sparse PPMI, differential PPMI
- `topics.py` — eigen-topics, IPR, density order, topic labels, jackknife stability

Experiment B (selection):

- `select.py` — embedding-matrix SVD, leverage / ridge-leverage / CUR-residual scores, direction labelling

Experiment C (code channel):

- `code.py` — tree-sitter language detection, AST-depth line banding, `quantize_segment`
- `supersede.py` — superseded read/edit elision and exact duplicate detection

Supporting:

- `tests/` — pytest
- `experiments/` — Phase 1 scripts; outputs in `experiments/output/` (gitignored)
  - `fetch_public.py` — downloads trace-commons/agent-traces, SWE-Gym/OpenHands-SFT-Trajectories, oasst2 validation into `data/public/`, writes `SOURCES.md` with revision hashes
  - `pick_local.py` — copies 5 local Claude Code sessions nearest {5K,20K,50K,100K,200K} tokens into `data/local/sessions/`
  - `corpus_stats.py` — per-transcript segment/token/per-kind/atom table
  - `baselines.py` — random drop, recency, TF-IDF
  - `exp_a_ppmi.py`, `exp_b_embed.py`, `exp_c_code.py` — the three Phase 1 tracks
- `docs/` — design spec, Phase 1 write-up, committed result tables
- `data/local/` — user's own session logs, gitignored, **never committed**
- `data/public/` — public dataset downloads, gitignored; regenerate with `fetch_public.py`
- `.claude/` — local-only personal agent env, excluded via `.git/info/exclude`, not part of the repo

## Conventions

- Fully typed; `ruff` for lint and format; mypy strict on `src/`.
- Comments only where load-bearing (a decision, a constraint, a non-obvious construct). No narration.
- User-visible changes get a dated line in `CHANGELOG.md`.
- Tests cover the algorithmic core; thin CLI glue gets smoke tests only.
- Selection and scoring must never share an embedding model — see `embed.py`.
- **Any transform downstream of segmentation needs its own version stamp** in Experiment D's
  record identity. `segments_digest` fingerprints the segmenter's *input* and deliberately stops
  there, so that variants accumulate in a checkpoint rather than evicting each other — which
  means a rewritten `code.py` or `paths.py` leaves stored records looking valid while describing
  behaviour that no longer exists. Three stamps exist for this: `code.TRANSFORMS_VERSION`,
  `paths.PATHS_VERSION`, and `exp_d_curves.BAND_QUALITY_VERSION` for the runner's own decision to
  hand `quantize_segment` the sweep's keep ratio as its fidelity knob. The third is the one that
  shows the rule is about transforms, not modules — it lives in the runner because that mapping
  does. A fourth transform needs a fourth. See
  [docs/PHASE2.md](docs/PHASE2.md#a-silent-staleness-trap-found-by-walking-into-it).

## Algorithm decisions (locked 2026-09-08)

- Hybrid core: Shin-style PPMI-SVD for an interpretable topic basis + embedding-SVD /
  leverage-score subset selection for stable segment scoring.
- v1 output is extractive only — no LLM summarization in the loop.
- A background prior (generic, later personalized from local sessions) stabilizes small-context
  SVD; top bias-like eigen-directions are stripped before truncation.
- **No binary `protected` flag.** It was replaced by a per-kind quality table (prose, code,
  tool_use, tool_result, thinking), because `tool_use` + `tool_result` are 51–91% of tokens in
  real agentic sessions and protecting them caps compression near 1.3x. Code keeps its label
  but its body is quantized by AST-depth banding. See
  [docs/DESIGN.md](docs/DESIGN.md#the-quality-table).
- Two lossless passes (supersession, exact duplicate elision) run before any lossy stage.
- Distortion metrics: semantic coverage (embedding cosine) + atom recall. Never reconstruction
  error — see the perception-distortion note in the design doc.

## Phase 1 outcome, in one paragraph

Supersession dedupe removes 25.9% of tokens losslessly and is the clearest win. AST banding
beats whole-segment dropping at equal budget but only buys 1.33x alone. The PPMI-SVD *topical*
eigenvectors fail a pre-registered 0.70 stability threshold (0.507 measured), so Experiment A
is demoted to a labeller and Experiment B drives selection. Spectral and budget-shaped TF-IDF
**split the two distortion metrics** — spectral wins atom recall at every rate, TF-IDF wins
semantic coverage at every rate — and that split is reported as two statistics rather than
resolved into a winner. Do not re-run these; read [docs/PHASE1.md](docs/PHASE1.md), which also
lists the segmenter, loader and atom-extractor defects found since, and five implementation
defaults still awaiting confirmation.

## Hard rules

- Never commit anything under `data/local/`. It contains unrelated employer work.
- Never quote local session content in a report, doc, commit message or result table.
- Do not push unprompted. Pushing to `origin` on this personal repo is fine **when the user
  asks for it**; it is not something to do on your own initiative.
- Design decisions go through the user: propose options, wait.

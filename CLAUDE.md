# Loosy-Goose — agent guide

Lossy text compression for LLM context compaction via eigen-dimension semantic spaces.
Personal GitHub project (public, MIT). Not related to any employer repository.

Read [docs/DESIGN.md](docs/DESIGN.md) before changing the algorithm and
[docs/PHASE1.md](docs/PHASE1.md) before re-deciding anything Phase 1 already measured.

## Commands

- `uv sync --dev --extra embed` — install (Python 3.12 pinned via `.python-version`). The CLI
  alone needs `uv sync --extra embed`; the experiments also need the dev group, which carries
  scikit-learn (TF-IDF baselines)
- `uv run pytest` — tests
- `uv run ruff check . && uv run ruff format --check .` — lint + format check
- `uv run mypy src` — type check (strict)
- `uv run loosy-goose compress <transcript> [--keep 0.5 | --level light|medium|heavy | --target-tokens N] [--no-novelty] [--band] [--paths --min-mentions 2] [--output out.jsonl]` — CLI: supersede → leverage → per-token novelty selection at alpha 0.5 (default; `--no-novelty` for a plain budget fill; `--band` / `--paths` fall back to plain leverage); rate defaults to `--level medium` (0.33); header reports `atom_recall_final` as information only

The `embed` extra pulls torch + sentence-transformers for the embedding-SVD work, and the CLI
cannot score without it; CI installs it. torch resolves from PyPI, so Linux x86_64/aarch64 gets the
CUDA wheels (~3 GB) while macOS/Windows get CPU builds; the code itself runs CPU-only. Do not unpin Python 3.12 — 3.13+
lacks torch wheels.

## Layout

Core:

- `spectral.py` — PPMI, inverse participation ratio, sparse truncated SVD
- `transcript.py` — `Transcript`/`Turn`/`Block`; tolerant Claude Code JSONL loader (unknown record/block types are counted in `Transcript.meta`, never fatal) and generic `[{role, content}]` loader
- `segment.py` — `Segment` units: prose split into whole paragraphs (12k-char cap), code/tool_use atomic, tool_result line-chunked; `atoms` = identifiers/paths/numbers/URLs for the recall metric
- `tokens.py` — `count_tokens` via tiktoken `o200k_base` (model-agnostic rate proxy)
- `embed.py` — pinned select/score model pair
- `budget.py` — token budget, protect policy, greedy score-based selection. **The shared contract**: every method competes at an identical token budget, or results are not comparable
- `metrics.py` — compression ratio, atom recall, semantic coverage, `evaluate`
- `pipeline.py` — `compress_transcript`: the shipped supersede → (band) → (paths) → leverage → per-token novelty@0.5 stack (exp_g's `supersede+leverage+novelty-per-token@0.5`, default since 2026-10-02), band and paths opt-in; `--no-novelty`, `--band` or `--paths` select by plain leverage budget fill instead, since novelty was never measured with band or paths; `PipelineOptions` carries the knobs; `to_turns` regroups kept segments into turns
- `cli.py` — argparse entry point

Experiment A (topic basis):

- `cooccur.py` — tokenizer, vocab, windowed co-occurrence, sparse PPMI, differential PPMI
- `topics.py` — eigen-topics, IPR, density order, topic labels, jackknife stability

Experiment B (selection):

- `select.py` — embedding-matrix SVD, leverage / ridge-leverage / CUR-residual scores, direction labelling
- `compress_select.py` — compression-based selection: conditional greedy (lazy; exact kept as a test anchor) over the compressed bytes a segment adds, zlib or lzma
- `novelty_select.py` — leverage x word novelty: each candidate's score discounted by (share of its plain words the kept set lacks) ** alpha, lazy greedy; alpha=0 is the plain budget fill; `per_token=True` ranks by that gain per token
- `kind_select.py` — per-kind scoring on the supersession pool: leverage in each kind's own space, shared leverage rank- or z-normalised within kind, or the budget split into per-kind slices by pool-token share with one global spill pass

Experiment C (code channel):

- `code.py` — tree-sitter language detection, AST-depth line banding, `quantize_segment`
- `supersede.py` — superseded read/edit elision and exact duplicate detection
- `paths.py` — path table and marker substitution

Supporting:

- `tests/` — pytest
- `experiments/` — experiment scripts, Phase 1 and 2; outputs in `experiments/output/` (gitignored); see `experiments/README.md`
  - `fetch_public.py` — downloads trace-commons/agent-traces, SWE-Gym/OpenHands-SFT-Trajectories, oasst2 validation into `data/public/`, writes `SOURCES.md` with revision hashes
  - `pick_local.py` — copies 5 local Claude Code sessions nearest {5K,20K,50K,100K,200K} tokens into `data/local/sessions/`
  - `corpus_stats.py` — per-transcript segment/token/per-kind/atom table
  - `baselines.py` — random drop, recency, TF-IDF
  - `exp_a_ppmi.py`, `exp_b_embed.py`, `exp_c_code.py` — the three Phase 1 tracks
  - `exp_d_curves.py` — the Phase 2 rate-distortion runner and the grid registry (`--grid exp_d|exp_f|exp_g|exp_h`, partial runs via `--methods` / `--variants`)
  - `exp_e_analysis.py` — Experiment E, elasticity / per-kind / predictors / cost over the exp_d checkpoints; no argparse, so any invocation runs it and rewrites its output
  - `exp_f_compress.py` — Experiment F grid (`exp_d_curves.py --grid exp_f`): the `compress_select` arms, compared against exp_d's reference checkpoints
  - `exp_g_novelty.py` — Experiment G grid (`exp_d_curves.py --grid exp_g`): `novelty_select` on the supersession pool at alpha 0.5 / 1 / 2, plus a per-token arm at 0.5, compared against exp_d's leverage and supersede+leverage
  - `exp_h_kinds.py` — Experiment H grid (`exp_d_curves.py --grid exp_h`): the four `kind_select` arms on the supersession pool, compared against exp_d's leverage and supersede+leverage, plus a kept-token share per kind table
  - `grid_status.py` — read-only progress watcher for a running grid (`--grid NAME [--methods A,B] [--variants base] [--log PATH] [--watch [--interval 10]]`): done/planned per transcript and method, rough ETA, NOT RUNNING / stall warnings
  - `survey_paths.py` — path mass and substitution-table cost, counted with `paths.find_paths`; public corpora only
  - `survey_tool_output.py` — shapes inside `tool_result` blocks and their token share; public corpora only
- `docs/` — design spec, Phase 1 and Phase 2 write-ups, committed result tables
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
  record identity. `segments_digest` fingerprints the segmenter's *output* — each segment's kind
  and text — plus `segment.ATOMS_VERSION` and `supersede.SUPERSEDE_VERSION`, because atoms and
  the supersession pass's final-state atom set are every record's recall denominator. It
  deliberately stops there, so that variants accumulate in a checkpoint rather than evicting each
  other — which means a rewritten `code.py` or `paths.py` leaves stored records looking valid
  while describing behaviour that no longer exists. `RecordIdentity` therefore carries, beside
  method, protect, ratio and variant key, one stamp per downstream transform, each 0 for methods
  it does not touch: `code.TRANSFORMS_VERSION` (banding methods), `paths.PATHS_VERSION` (path
  variants), `exp_d_curves.BAND_QUALITY_VERSION` (the runner's own decision to hand
  `quantize_segment` the sweep's keep ratio as its fidelity knob), `SUPERSEDE_VERSION` (methods
  that supersede), and `select_version` (`compress_select.SELECT_VERSION`,
  `novelty_select.NOVELTY_VERSION` or `kind_select.KIND_SELECT_VERSION`, via `select_stamp`). `BAND_QUALITY_VERSION` is the one that
  shows the rule is about transforms, not modules — it lives in the runner because that mapping
  does. Any new transform needs its own. See
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
lists the segmenter, loader and atom-extractor defects found since. Phase 2's current state is at
the top of [docs/PHASE2.md](docs/PHASE2.md).

## Hard rules

- Never commit anything under `data/local/`. It contains unrelated employer work.
- Never quote local session content in a report, doc, commit message or result table.
- Do not push unprompted. Pushing to `origin` on this personal repo is fine **when the user
  asks for it**; it is not something to do on your own initiative.
- Design decisions go through the user: propose options, wait.

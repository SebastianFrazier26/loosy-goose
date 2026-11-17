# Experiment scripts

Outputs go to `experiments/output/` (gitignored). Committed result tables live in
`docs/results/`; the write-ups are `docs/PHASE1.md` and `docs/PHASE2.md`.

Every script here needs the dev dependency group as well as the `embed` extra, because the
TF-IDF baselines use scikit-learn:

```
uv sync --dev --extra embed
```

## Corpus setup

```
uv run python experiments/fetch_public.py    # public corpora into data/public/
uv run python experiments/pick_local.py      # optional: your own sessions into data/local/
uv run python experiments/corpus_stats.py
```

`fetch_public.py` pins dataset revisions and writes `data/public/SOURCES.md` with the hashes,
so the public half of the corpus is reproducible on any machine. `pick_local.py` is optional
and machine-specific — it reads `~/.claude/projects/*/*.jsonl`, and its output is never
committed.

## Phase 1: the three tracks

| Script | Question | Verdict |
| --- | --- | --- |
| `exp_a_ppmi.py` | Do PPMI-SVD eigen-topics stay stable on a single conversation? | Partial fail — topical (sparse) directions unstable; demoted to labeller |
| `exp_b_embed.py` | Does embedding-SVD leverage scoring select well? | Works, but the spectrum has no knee and scores are flat |
| `exp_c_code.py` | Can code be compressed rather than protected? | Yes — supersession removes 25.9% losslessly; AST banding beats dropping at equal budget |

`baselines.py` provides the comparison methods every track is scored against: random drop,
recency-only, and TF-IDF. All of them, and every experimental method, are forced through the
same token budget in `budget.py` — that shared budget is what makes the numbers comparable.

## Phase 2: rate-distortion grids

**`exp_d_curves.py`** — the runner. Every method on one curve per transcript at matched token
budgets, checkpointed per transcript, so an interrupted run resumes and a later run reuses what
an earlier one wrote. It also holds the grid registry:

```
uv run python experiments/exp_d_curves.py                         # Experiment D, the main grid
uv run python experiments/exp_d_curves.py --grid exp_f            # Experiment F
uv run python experiments/exp_d_curves.py --grid exp_g            # Experiment G
uv run python experiments/exp_d_curves.py --grid exp_h            # Experiment H
uv run python experiments/exp_d_curves.py --methods supersede+band+leverage --variants base
```

`--methods` and `--variants` narrow the plan to a partial run; cells left out are reported as
missing, never averaged. For a partial exp_d run pass both: `--methods` alone also runs that
method on every variant that covers it. `--only NAME` restricts to transcripts whose label
contains NAME, `--limit N` stops after N transcripts, and `--force` discards the checkpoints. A
full exp_d run takes 6–24 hours.

**`exp_e_analysis.py`** — Experiment E: elasticity, per-kind loss, predictors and cost, read from
the exp_d checkpoints alone (no transcripts, no embedding model). **It has no argparse: any
invocation, `--help` included, runs the full analysis and rewrites
`experiments/output/exp_e/summary.txt` and `elasticity.png`.**

**`exp_f_compress.py`** — Experiment F: selection by compressed novelty (`compress_select`, lazy
greedy with zlib or lzma, each with and without supersession). Run through
`exp_d_curves.py --grid exp_f`; reads `tfidf`, `leverage` and `supersede+leverage` from the exp_d
checkpoints rather than re-running them.

**`exp_g_novelty.py`** — Experiment G: leverage times word novelty (`novelty_select`) on the
supersession pool at alpha 0.5 / 1 / 2, plus a per-token arm at 0.5 (ranked by gain per token).
The per-token arm was first left off the grid deliberately; since 2026-09-29 it is part of the
final grid. Run through `exp_d_curves.py --grid exp_g`; reads `leverage` and `supersede+leverage`
from exp_d.

**`exp_h_kinds.py`** — Experiment H: per-kind scoring on the supersession pool (`kind_select`),
four arms that each take segment kind out of the shared-space competition: leverage in each
kind's own space (`kind-space`), shared leverage rank-percentiled or z-scored within kind
(`kind-norm-rank`, `kind-norm-z`), and the budget split into per-kind slices by pool-token share
with one global spill pass (`kind-budget`). Run through `exp_d_curves.py --grid exp_h`; reads
`leverage` and `supersede+leverage` from exp_d, and adds a kept-token share per kind table beside
the pool's own shares.

**`grid_status.py`** — a read-only progress view for a second terminal beside a running grid. It
reads the checkpoints (and the runner's log with `--log`), never the transcripts, and writes
nothing:

```
uv run python experiments/grid_status.py --grid exp_d --methods A,B --variants base --log PATH --watch --interval 10
```

## Experiment I: downstream performance

**`exp_i_downstream.py`** — does a model do as well from a loosy-compressed context as from the
full one, and versus a Sonnet summary at the same token budget? About 20 public transcripts
(20k–60k tokens) are cut before a real assistant action late in the session; the last 4,000 tokens
stay verbatim and the earlier stretch is rendered seven ways (full, loosy light/medium/heavy,
summary at each loosy level's size). Haiku 5.5 and Sonnet 5.5 (`claude -p`) and gpt-6-luna and
gpt-6-sol (`codex exec`) answer factual probes and predict the next action; Sonnet grades blind.
Design: `docs/superpowers/specs/2026-10-14-exp-i-downstream-design.md`.

It calls the real CLIs and spends subscription usage. Every call is checkpointed to
`experiments/output/exp_i/calls.jsonl`, so a rerun skips finished calls and retries failed ones;
usage-limit errors back off to a 30-minute sleep instead of aborting. `--limit N` / `--only NAME`
make a partial run, which writes `experiments/output/exp_i/summary.txt` only; a full run also
writes `docs/results/exp_i_downstream.txt`. `--dry-run` selects and renders without calling.

```
nohup caffeinate -i uv run python experiments/exp_i_downstream.py --workers 4 \
  >> experiments/output/exp_i/run.log 2>&1 &
uv run python experiments/exp_i_status.py --watch
```

**`exp_i_status.py`** — read-only progress for the run: done/planned per task and model, tokens so
far, rate and rough ETA, failures awaiting a retry, NOT RUNNING / stall warnings.

## Surveys

**`survey_paths.py`** — how much of a transcript is file paths, and what a substitution table
would cost. Writes `experiments/output/survey/path_mass.txt`. Paths are counted with
`paths.find_paths` over decoded text, exactly as `paths.build_table` counts them.

**`survey_tool_output.py`** — what shapes sit inside `tool_result` blocks and how many tokens
each accounts for, surveyed per block rather than per segment. Writes
`experiments/output/survey/tool_output_shapes.txt`.

Both take `--public-sample` and `--swe-gym-sample` and read the public corpora only (since
2026-09-29, queue item 13), so their output is regenerable from `fetch_public.py` alone. Figures
in `docs/PHASE2.md` dated earlier also counted local sessions.

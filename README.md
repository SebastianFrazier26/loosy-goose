# Loosy-Goose

Lossy, JPEG-style compression for LLM conversation context. Model-agnostic: the output is
plain text any frontier model can read.

## What it does

A long conversation is treated the way an image codec treats pixels:

| Codec stage | Loosy-Goose |
| --- | --- |
| Block split | Segment the transcript into units: whole prose paragraphs (12k-character cap), code blocks and tool calls kept whole, tool results cut into line chunks |
| Transform (DCT ≈ KLT) | SVD of the context's own semantic matrix — the exact KLT, affordable because a session is small |
| Quantization matrix | Per-kind quality table plus rank cutoff, dead-zone thresholds and recency weighting — a design idea, not wired; the shipped rate control is `--keep` / `--level` / `--target-tokens` |
| Coefficient truncation | Drop segments with little weight on the retained dimensions; band-limit the ones we keep |
| Entropy coding | Emit text; the target model's tokenizer does the lossless part |
| Chroma subsampling | Opt-in (`--band`, off by default): code bodies and code-shaped tool arguments are banded down to their outline by AST depth rather than dropped wholesale |

The output is **extractive**: real segments of the original, kept, trimmed or dropped. No
summarizer rewrites anything, so the result is deterministic and auditable.

## Why

Frontier-model context windows fill up. Built-in compaction is opaque, non-configurable, and
model-specific. Loosy-Goose is grounded in the observation of Shin, Madotto & Fung (2018),
[*Interpreting Word Embeddings with Eigenvector Analysis*](https://openreview.net/forum?id=rJfJiR5ooX)
([PDF](https://web.archive.org/web/20250405190331if_/https://openreview.net/pdf?id=rJfJiR5ooX) ·
[code](https://github.com/HLTCHKUST/eigenvector-analysis)): the eigenvectors of a PPMI-SVD
embedding form semantically coherent word groups, and the *sparse* eigenvectors (high inverse
participation ratio) carry the narrow, topical ones. That gives an interpretable basis in which
to decide what a conversation can afford to lose.

Where that paper meets the codec analogy — and the one point where the analogy inverts, which
is the reason the paper is load-bearing rather than decorative — is
[docs/DESIGN.md § Origin](docs/DESIGN.md#origin).

## Status

Pre-alpha. The `compress` command works: it runs the measured default pipeline (supersession,
then leverage scoring, then selection into a token budget ranked by novelty-discounted leverage
per token) and writes the kept text as JSONL. Phase 2 is still open; see
[docs/PHASE2.md](docs/PHASE2.md).

### Current results

The default selector is `supersede+leverage+novelty-per-token@0.5`, decided 2026-10-02 from the
final grid (run to 2026-10-01). Each candidate's leverage is discounted by the share of its plain
words the kept set does not hold yet, raised to the power 0.5, and candidates are ranked by that
gain per token. Measured on 18 transcripts at matched achieved rates
(`docs/results/exp_g_summary.txt`; the old default's rows come from exp_d's records, the same
numbers as `docs/results/exp_d_summary.txt`):

| Rate | final-state atom recall | semantic coverage | old default (`supersede+leverage`) recall / coverage |
| --- | --- | --- | --- |
| 2x | 0.829 (15 of 18 transcripts reach 2x) | 0.921 | 0.837 / 0.895 (16 of 18) |
| 3x | 0.655 (17 of 18) | 0.887 | 0.656 / 0.821 (17 of 18) |
| 4x | 0.577 | 0.872 | 0.568 / 0.796 |

Paired on the transcripts both reach, the new default against the old is −0.011 recall and
+0.026 coverage at 2x (15 of 18 transcripts only), −0.001 and +0.066 at 3x, and +0.009 and +0.076
at 4x. Its knee (lowest rate still holding final-state recall of 0.9) is 0.606 on 15 of 18,
against 0.598 on 16 of 18. It costs 0.24 s per 100k tokens against 0.32. Of the arms the final
grid tried against the old default, it is the only one ahead on both metrics at any rate: at 4x,
with 3x a near tie on recall. Why it was chosen, and what was rejected, is in
[docs/PHASE2.md § Decision log](docs/PHASE2.md#decision-log).

`--no-novelty` restores the old default, `supersede+leverage` with a plain budget fill.
`--band` and `--paths` fall back to plain leverage selection, because novelty was never measured
stacked on either.

Tried against the old default, and left out of the default:

- Compression-based selection (Experiment F, zlib and lzma) buys coverage and loses recall at
  every rate, at up to ~107x the cost.
- Raw-gain word novelty (Experiment G, alpha 0.5 / 1 / 2) buys recall and loses coverage.
- Per-kind scoring (Experiment H) loses recall at every rate on all four arms.
- `--band` loses at 2x (−0.014 recall), gains about one point at 3x and 4x, and is about 47x
  slower (`docs/results/exp_e_summary.txt`, E4). The new default without banding beats
  `supersede+band+leverage` at 4x on both metrics.
- `--paths` full arms on `supersede+leverage` gain recall but lose 4 to 9 coverage points
  (`docs/results/exp_d_summary.txt`), and recall on the atoms the table does not guarantee falls
  on every table arm.

### Phase 1

Phase 1 experiments are **complete** — see [docs/PHASE1.md](docs/PHASE1.md) for the full
results. Headline findings, stated plainly:

- Supersession dedupe removes **25.9% of tokens losslessly** across a 1.5M-token corpus. This
  is the clearest win so far and it uses no eigenvectors at all. (Phase 1 measured that under the
  older segmenter; the final grid's `exp_c` C2 reads 24.8%.)
- AST-depth banding of code beats dropping whole code segments at an equal token budget
  (atom recall 0.885 vs 0.854 for TF-IDF drop; coverage 0.973 vs 0.735), but only buys 1.33x on
  its own — it is a quantizer, not a compressor.
- The PPMI-SVD topical eigenvectors are **not stable** at single-conversation scale
  (aligned cosine 0.507 against a pre-registered 0.70 threshold), so that basis is demoted to
  a labeller.
- **Spectral selection and budget-shaped TF-IDF split the two distortion metrics.** Spectral
  keeps more concrete facts (atom recall) at every rate measured; TF-IDF better covers the
  conversation's meaning (semantic coverage) at every rate measured. They are reported as two
  statistics and deliberately not collapsed into one winner — the metrics were designed to
  disagree, and averaging them away would hide the thing they exist to show.

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — algorithm specification, quality table, distortion metrics
- [docs/PHASE1.md](docs/PHASE1.md) — experiment designs, pre-registered criteria, results, open defects
- [docs/PHASE2.md](docs/PHASE2.md) — Phase 2 plan, decisions with the option rejected, verified nulls, what is still open
- [docs/results/](docs/results/) — raw summary tables as emitted by the experiment scripts, with a note on how stale each one is
- [integrations/README.md](integrations/README.md) — the /loosy (Claude Code) and $loosy (Codex) session handoff: install, register, use
- [CLAUDE.md](CLAUDE.md) — repository guide for coding agents

## Run

The CLI needs the `embed` extra, because leverage scoring embeds every segment with
sentence-transformers:

```
uv sync --extra embed
uv run loosy-goose compress path/to/transcript.jsonl                  # level medium, keep 0.33 (~3x)
uv run loosy-goose compress path/to/transcript.jsonl --level light    # 0.50 (~2x); heavy = 0.25 (~4x)
uv run loosy-goose compress path/to/transcript.jsonl --keep 0.42      # any ratio in [0.10, 0.80]
uv run loosy-goose compress path/to/transcript.jsonl --target-tokens 20000
```

The first run downloads the selection embedding model, `BAAI/bge-small-en-v1.5`, from Hugging
Face. The experiments also use a second model, `sentence-transformers/all-MiniLM-L6-v2`, to score
coverage; the two are kept apart on purpose (see `src/loosy_goose/embed.py`).

**Input.** Two formats, picked by file suffix:

- `.jsonl` — a Claude Code session log. Unknown record and block types are counted and skipped,
  never fatal.
- `.json` — a generic message list, `[{"role": ..., "content": ...}]`, or an object with a
  `messages` key holding one. Roles `user`, `assistant`, `system` and `tool` are read; `content`
  may be a string or a list of content blocks.

**Output.** JSONL. The first line is a header object under the key `loosy_goose`: the rate used
and where it came from, the flags, the token budget, input and output token counts,
`atom_recall_final` (information only; nothing reads it back), the version stamps, and the path
table when `--paths` is on. Every line after that is one kept turn,
`{"index", "role", "blocks": [{"kind", "text", "meta"}]}`, in original order.

**Flags.**

| Flag | Meaning |
| --- | --- |
| `--level light\|medium\|heavy` | preset rate: keep 0.50 / 0.33 / 0.25 of input tokens (~2x / ~3x / ~4x). Default `medium` |
| `--keep R` | keep ratio in [0.10, 0.80], in 0.01 steps (finer values are rounded) |
| `--target-tokens N` | output token budget; one that implies a ratio outside [0.10, 0.80] is clamped, with a warning |
| `--no-novelty` | select by plain leverage (`supersede+leverage`) instead of novelty-discounted leverage per token |
| `--band` | AST-depth banding of code and code-shaped tool arguments before scoring. Off by default; selection falls back to plain leverage |
| `--paths` | name each repeated path once in a leading table and replace mentions with `[P<n>]` markers. Off by default; selection falls back to plain leverage |
| `--min-mentions N` | with `--paths`, tabulate paths mentioned at least N times (default 2; 1 = every path) |
| `--output PATH` | write here instead of stdout (`-`, the default, is stdout) |

`--level`, `--keep` and `--target-tokens` are mutually exclusive. No preset turns on `--band` or
`--paths`.

## Handoff from Claude Code or Codex

`loosy-goose-mcp` is an MCP server whose `handoff_session` tool forks your live Claude Code or
Codex session into a compressed copy. The copy holds the kept turns, a verbatim recent tail and
a marker, and you resume it in place of built-in `/compact`. The original session is never
modified. Install, registration and the `/loosy` / `$loosy` commands:
[integrations/README.md](integrations/README.md). The same server's `compress` tool returns
compressed turns for any transcript and writes nothing.

## Continuing on another machine

Everything needed is in the repo; nothing depends on the machine Phase 1 ran on.

```
git clone https://github.com/SebastianFrazier26/loosy-goose
cd loosy-goose
uv sync --dev --extra embed      # embed extra pulls torch CPU + sentence-transformers;
                                 # the dev group carries scikit-learn, which the experiments need
uv run pytest                    # expect a green suite
uv run python experiments/fetch_public.py    # public corpora into data/public/
uv run python experiments/corpus_stats.py
```

Python is pinned to 3.12 by `.python-version` — 3.13+ has no torch or sentence-transformers
wheels at time of writing, so do not unpin it casually. Everything runs on CPU; no GPU is
assumed anywhere.

`data/local/` holds the author's own Claude Code sessions and is gitignored and empty on a
fresh clone. That is deliberate. To build an equivalent local corpus from your own sessions:

```
uv run python experiments/pick_local.py
```

It selects sessions from `~/.claude/projects/*/*.jsonl` nearest 5K, 20K, 50K, 100K and 200K
tokens. The public corpora alone are enough to reproduce every public row in
[docs/results/](docs/results/).

## Test

```
uv run pytest
```

Lint, format check and type check: `uv run ruff check . && uv run ruff format --check . && uv run mypy src`.

Opt-in drift and end-to-end tests against the real CLIs (needs logged-in `claude` and `codex`, spends a little API use): `uv run pytest -m live`. Re-record the synthetic fixtures with `uv run python tests/session_recorder.py`.

## Roadmap

- **Done** — Phase 1: PPMI-SVD vs embedding-SVD vs code channel, on real and public transcripts
- **In progress** — Phase 2: unified rate-distortion curves, all methods and baselines at matched
  budgets. Plan, decisions and open questions in [docs/PHASE2.md](docs/PHASE2.md)
- **Done** — core pipeline and CLI: segment → supersede → (band) → (paths) → leverage score →
  novelty-per-token select (plain budget select with `--no-novelty`, `--band` or `--paths`) → emit
- **Done** — MCP server and session handoff (`/loosy`, `$loosy`) for Claude Code and Codex
- Claude Code plugin packaging and marketplace entry
- Personalization prior built from the user's own local sessions

## License

MIT — see `LICENSE`.

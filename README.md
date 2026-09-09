# Loosy-Goose

Lossy, JPEG-style compression for LLM conversation context. Model-agnostic: the output is
plain text any frontier model can read.

## What it does

A long conversation is treated the way an image codec treats pixels:

| Codec stage | Loosy-Goose |
| --- | --- |
| Block split | Segment the transcript into units (sentences, code blocks, tool-result chunks) |
| Transform (DCT ≈ KLT) | SVD of the context's own semantic matrix — the exact KLT, affordable because a session is small |
| Quantization matrix | Per-kind quality table plus rank cutoff, dead-zone thresholds and recency weighting, all behind one `--quality` knob |
| Coefficient truncation | Drop segments with little weight on the retained dimensions; band-limit the ones we keep |
| Entropy coding | Emit text; the target model's tokenizer does the lossless part |
| Chroma subsampling | Code and tool output are quantized harder than prose — banded to signatures and commands, not dropped wholesale |

The output is **extractive**: real segments of the original, kept, trimmed or dropped. No
summarizer rewrites anything, so the result is deterministic and auditable.

## Why

Frontier-model context windows fill up. Built-in compaction is opaque, non-configurable, and
model-specific. Loosy-Goose is grounded in the observation of Shin, Madotto & Fung (2018),
[*Interpreting Word Embeddings with Eigenvector Analysis*](https://openreview.net/forum?id=rJfJiR5ooX)
([code](https://github.com/HLTCHKUST/eigenvector-analysis)): the eigenvectors of a PPMI-SVD
embedding form semantically coherent word groups, and the *sparse* eigenvectors (high inverse
participation ratio) carry the narrow, topical ones. That gives an interpretable basis in which
to decide what a conversation can afford to lose.

## Status

Pre-alpha. The `compress` command is a stub.

Phase 1 experiments are **complete** — see [docs/PHASE1.md](docs/PHASE1.md) for the full
results. Headline findings, stated plainly:

- Supersession dedupe removes **25.9% of tokens losslessly** across a 1.5M-token corpus. This
  is the clearest win so far and it uses no eigenvectors at all.
- AST-depth banding of code beats dropping whole code segments at an equal token budget
  (atom recall 0.885 vs 0.854 for TF-IDF drop; coverage 0.973 vs 0.735), but only buys 1.33x on
  its own — it is a quantizer, not a compressor.
- The PPMI-SVD topical eigenvectors are **not stable** at single-conversation scale
  (aligned cosine 0.507 against a pre-registered 0.70 threshold), so that basis is demoted to
  a labeller.
- **No spectral method has yet been shown to beat budget-shaped TF-IDF.** Phase 2 exists to
  settle that.

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md) — algorithm specification, quality table, distortion metrics
- [docs/PHASE1.md](docs/PHASE1.md) — experiment designs, pre-registered criteria, results, open defects
- [docs/results/](docs/results/) — raw summary tables as emitted by the experiment scripts
- [CLAUDE.md](CLAUDE.md) — repository guide for coding agents

## Run

```
uv sync
uv run loosy-goose compress path/to/transcript.jsonl --quality 0.6
```

## Continuing on another machine

Everything needed is in the repo; nothing depends on the machine Phase 1 ran on.

```
git clone https://github.com/SebastianFrazier26/loosy-goose
cd loosy-goose
uv sync --dev --extra embed      # embed extra pulls torch CPU + sentence-transformers
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

## Roadmap

- **Done** — Phase 1: PPMI-SVD vs embedding-SVD vs code channel, on real and public transcripts
- **Next** — Phase 2: unified rate-distortion curves, all methods and baselines at matched budgets
- Core: segment → spectral → select → quantize → emit
- CLI
- MCP server
- Claude Code plugin; Codex parity
- Personalization prior built from the user's own local sessions

## License

MIT — see `LICENSE`.

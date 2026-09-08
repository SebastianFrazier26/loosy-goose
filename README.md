# Loosy-Goose

Lossy, JPEG-style compression for LLM conversation context. Model-agnostic: the output is
plain text any frontier model can read.

## What it does

A long conversation is treated the way an image codec treats pixels:

| Codec stage | Loosy-Goose |
| --- | --- |
| Block split | Segment the transcript into units (sentences / turns / paragraphs) |
| Transform (DCT ≈ KLT) | SVD of the context's own semantic matrix — the exact KLT, affordable because a session is small |
| Quantization matrix | Rank cutoff + per-eigen-dimension thresholds + recency weighting, all behind one `--quality` knob |
| Coefficient truncation | Drop or trim segments with little weight on the retained dimensions |
| Entropy coding | Emit text; the target model's tokenizer does the lossless part |
| Lossless regions | Code blocks, paths, numbers, identifiers and user directives are protected verbatim |

The output is **extractive**: real segments of the original, kept or dropped. No summarizer
rewrites anything, so the result is deterministic and auditable.

## Why

Frontier-model context windows fill up. Built-in compaction is opaque, non-configurable, and
model-specific. Loosy-Goose is grounded in the observation of Shin, Madotto & Fung (2018),
[*Interpreting Word Embeddings with Eigenvector Analysis*](https://openreview.net/forum?id=rJfJiR5ooX)
([code](https://github.com/HLTCHKUST/eigenvector-analysis)): the eigenvectors of a PPMI-SVD
embedding form semantically coherent word groups, and the *sparse* eigenvectors (high inverse
participation ratio) carry the narrow, topical ones. That gives an interpretable basis in which
to decide what a conversation can afford to lose.

## Status

Pre-alpha. Phase 1 experiments (eigen-topic stability at conversation scale, rate-distortion
curves) are in progress. The `compress` command is a stub.

## Run

```
uv sync
uv run loosy-goose compress path/to/transcript.jsonl --quality 0.6
```

## Test

```
uv run pytest
```

Lint, format check and type check: `uv run ruff check . && uv run ruff format --check . && uv run mypy src`.

## Roadmap

- Phase 1 experiments: PPMI-SVD vs embedding-SVD on real and public transcripts
- Core: segment → spectral → select → emit, plus distortion metrics (embedding cosine, atom recall)
- CLI
- MCP server
- Claude Code plugin; Codex parity
- Personalization prior built from the user's own local sessions

## License

MIT — see `LICENSE`.

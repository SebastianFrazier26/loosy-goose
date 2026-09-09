# Loosy-Goose — design

Status: Phase 1 complete, Phase 2 (unified rate-distortion evaluation) in progress.
This document is the algorithm specification. For what the Phase 1 experiments actually
measured and concluded, see [PHASE1.md](PHASE1.md); for the Phase 2 plan, the decisions already
locked and the ones still open, see [PHASE2.md](PHASE2.md).

## Problem

A long conversation with a frontier model eventually exceeds the context window. The built-in
remedy — compaction — is opaque, non-configurable, and different on every model. We want a
configurable, auditable, model-agnostic alternative.

"Model-agnostic" is a hard constraint and it settles a great deal. It rules out the entire
soft-prompt family (gist tokens, ICAE, xRAG, 500xCompressor), which achieve much higher ratios
but emit latent vectors bound to one model's embedding space. Whatever we emit has to be text
that any model can read. That puts us in the hard-compression family alongside LLMLingua and
Selective Context, where the honest ceiling before meaning degrades is roughly 4–5x.

## The codec analogy

The design borrows its shape from transform image coding (JPEG), because the problem has the
same structure: a signal that is expensive to store verbatim, a transform that concentrates
what matters, a quantizer that discards what does not, and a decoder that tolerates loss.

| Codec stage | Loosy-Goose |
| --- | --- |
| Block split | Segment the transcript into units (sentences, code blocks, tool-result chunks) |
| Transform (DCT ≈ KLT) | SVD of the context's own semantic matrix — the exact KLT, affordable because one session is small |
| Quantization matrix | Per-kind quality table plus rank cutoff, dead-zone thresholds and recency weighting, all behind one `--quality` knob |
| Coefficient truncation | Drop segments with little weight on the retained directions; band-limit the ones we keep |
| Entropy coding | Emit text; the target model's tokenizer does the lossless part |
| Chroma subsampling | Code and tool output are quantized more aggressively than prose, but never dropped wholesale |

One place the analogy inverts, and it matters. In JPEG the energy is in the low-frequency
coefficients, so truncating the tail is nearly free. In text embeddings the top directions are
frequency and corpus bias — the "market mode" — while topical meaning lives in the sparse tail
(Mu & Viswanath, *All-but-the-Top*; Timkey & van Schijndel on rogue dimensions). The inverse
participation ratio of Shin, Madotto & Fung is exactly the tool for telling those apart. So we
strip the top bias directions rather than keeping them, and we do not assume small singular
values are noise — Staats (2024) and Cancedda (ACL 2024) both show they carry signal in
transformers.

## Pipeline

```
transcript --> segment --> spectral --> select --> quantize --> emit
```

1. **Segment** (`segment.py`). Split into `Segment` units with a `kind` in
   {prose, code, tool_result, tool_use, thinking}. Prose splits paragraph then sentence;
   code and tool_use are atomic; tool_result is line-chunked. Each segment carries `atoms`
   — identifiers, paths, numbers, URLs — which the distortion metric scores against.
2. **Spectral** (`cooccur.py`, `topics.py`, `select.py`). Two bases, used for different jobs:
   - PPMI co-occurrence plus truncated SVD over the session's own vocabulary, with inverse
     participation ratio to separate sparse topical directions from dense bias directions.
     A differential-PPMI background prior subtracts generic conversational co-occurrence.
   - Sentence-embedding matrix plus SVD, with leverage-score, ridge-leverage or CUR-residual
     scores for segment selection.
3. **Select** (`budget.py`). Every method competes at an identical token budget:
   `ceil(total_tokens * keep_ratio)`. Greedy descending score, forced segments first, original
   order preserved. This shared budget is what makes methods comparable — without it, a method
   can look good purely by keeping more.
4. **Quantize** (`code.py`, `supersede.py`). Band-limit rather than drop, plus lossless dedupe.
5. **Emit.** Extractive: real spans of the original, kept, trimmed, or dropped. No summarizer
   in the loop, so output is deterministic and every retained token traces to a source span.
   Abstractive rewriting stays a later optional flag.

## The quality table

The original design had a binary `protected` flag: code, paths and identifiers kept verbatim,
everything else compressible. Measurement killed that. In real agentic sessions `tool_use` plus
`tool_result` are the bulk of the transcript — 51% to 91% of tokens across the traces measured
in Phase 1 (64% for a 185K-token local session, 87% for `trace-commons/4222016d`). Protecting
them caps total compression around 1.3x, which is not a codec.

So `protected` is replaced by a per-kind quality setting, the direct analogue of JPEG's
separate luma and chroma quantization tables. Every kind is compressible; they are compressed
by different mechanisms and to different degrees.

| Kind | Mechanism | Notes |
| --- | --- | --- |
| prose | segment selection | Highest fidelity. User directives weighted up. |
| code | AST-depth banding | Band 0 (imports, signatures, type and class declarations) is the DC coefficient and survives at any quality. Nested bodies are high-frequency detail. |
| tool_use | field banding | Command and target path kept; bulky argument payloads banded. |
| tool_result | line chunking plus selection | Plus supersession dedupe. |
| thinking | segment selection | Lowest default fidelity. |

Two lossless passes run before any lossy stage, because free wins should never be paid for
with distortion:

- **Supersession.** Only the latest `Read`/`Edit`/`Write` per file path affects the final
  state; earlier ones are redundant. Dropping them is lossless with respect to where the
  session ended up.
- **Exact duplicate elision.** Repeated identical tool results collapse to one.

## Distortion metrics

Reconstruction error is the wrong metric here, and using it would quietly wreck the project.
Blau & Michaeli's perception-distortion tradeoff is the reason: a fluent decoder — and an LLM
is one — optimizes perceptual plausibility, not fidelity. A compressed context that reads
beautifully and has lost the one path name that mattered scores well on any surface metric.

Two metrics, used together:

- **Atom recall.** Fraction of the original's identifiers, paths, numbers and URLs that
  survive. This is what catches "fluent but wrong". Reported per kind as well as overall.
- **Semantic coverage.** Mean over original segments of the maximum cosine to any kept
  segment, plus whole-document cosine.

Selection and scoring deliberately use **different embedding models** (`bge-small-en-v1.5` to
select, `all-MiniLM-L6-v2` to score). Sharing one model would let the selector be graded by its
own similarity function, which inflates every result.

## Targets

- Rate: 2–5x on real agentic transcripts, degrading gracefully rather than off a cliff.
- Baselines any spectral method must beat at equal budget: random drop, recency-only, TF-IDF.
  As of Phase 1 the spectral methods do **not** clearly beat budget-shaped TF-IDF. Settling
  that is why Phase 2 exists.

## Open design questions

- Whether the PPMI-SVD basis earns its cost at conversation scale, or survives only as a topic
  labeller. Phase 1 evidence leans toward the latter.
- How to drive the whole quality table from a single `--quality` knob.
- Whether the background prior should be generic, personalized from the user's own sessions, or
  both layered. The personalization prior is a committed roadmap item.
- Segment-level versus span-level trimming for prose.
- Whether "extractive" must mean *contiguous substring of the original*, or may mean
  *reconstructs the original exactly given the emitted side table*. Path substitution
  (`paths.py`) satisfies the second and not the first: the output plus the table expands back to
  the source, but the emitted text alone does not appear verbatim in it. Undecided pending the
  Phase 2 arms — see [PHASE2.md](PHASE2.md#still-open--do-not-treat-as-settled).

## References

- Shin, Madotto & Fung (2018), *Interpreting Word Embeddings with Eigenvector Analysis*,
  NeurIPS IRASL workshop. [OpenReview](https://openreview.net/forum?id=rJfJiR5ooX) ·
  [code](https://github.com/HLTCHKUST/eigenvector-analysis)
- Levy & Goldberg (2014), *Neural Word Embedding as Implicit Matrix Factorization*
- Levy, Goldberg & Dagan (2015), *Improving Distributional Similarity with Lessons Learned from
  Word Embeddings* — the source of the eigenvalue weighting choice (`eig_weight=0`)
- Mu & Viswanath (2018), *All-but-the-Top*
- Timkey & van Schijndel (2021), *All Bark and No Bite: Rogue Dimensions*
- Staats et al. (2024), *Small Singular Values Matter*
- Cancedda (2024), *Spectral Filters, Dark Signals, and Attention Sinks*, ACL
- Blau & Michaeli (2019), *Rethinking Lossy Compression: The Rate-Distortion-Perception
  Tradeoff*, ICML
- Jiang et al. (2023), *LLMLingua*; Li et al. (2023), *Selective Context*

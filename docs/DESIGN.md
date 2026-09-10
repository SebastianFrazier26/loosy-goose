# Loosy-Goose — design

Status: Phase 1 complete, Phase 2 (unified rate-distortion evaluation) in progress.
This document is the algorithm specification. For what the Phase 1 experiments actually
measured and concluded, see [PHASE1.md](PHASE1.md); for the Phase 2 plan, the decisions already
locked and the ones still open, see [PHASE2.md](PHASE2.md).

## Origin

The project starts from one paper and one analogy, and the distance between them is the whole
design problem.

The paper is Shin, Madotto & Fung (2018), *Interpreting Word Embeddings with Eigenvector
Analysis* (NeurIPS IRASL workshop) — the PDF this project was seeded with. Its finding: the
eigenvectors of a PPMI-SVD embedding are semantically coherent word groups, and the *sparse*
eigenvectors, identified by a high inverse participation ratio, are the narrow topical ones
while the dense ones carry frequency and corpus bias. That gives an interpretable basis in
which to ask what a body of text can afford to lose, which is the question a compressor asks.
OpenReview blocks scripted fetches; the PDF is reachable through the Wayback Machine, and both
links are in the [References](#references) below.

The analogy is JPEG: a transform that concentrates what matters, a quantizer that discards what
does not, a decoder that tolerates loss. It supplies the *architecture* — see the stage table
below — and it supplies the vocabulary the rest of this document is written in (quality table,
DC coefficient, chroma subsampling).

**The analogy inverts at the one point that matters**, which is why the paper is load-bearing
rather than decorative: JPEG's energy sits in the low-frequency coefficients, so truncating the
tail is nearly free, whereas in text embeddings the top directions are the bias directions and
meaning lives in the sparse tail. A naive port of JPEG would keep exactly the wrong end. The
inverse participation ratio is the instrument that tells the two ends apart.

What was fixed at the outset and has held since: the output is **text**, not latent vectors, so
any model can read it; the output is **extractive**, so no summarizer rewrites anything and the
result is deterministic and auditable; distortion is measured by **embedding cosine plus atom
recall**, never by reconstruction error. Phase 1 then measured the paper's own machinery at
conversation scale and demoted it — the topical eigenvectors are not stable in a single session
(0.507 aligned cosine against a pre-registered 0.70), so the topic basis became a labeller and
an embedding-matrix SVD drives selection. That demotion is a Phase 1 result, recorded in
[PHASE1.md](PHASE1.md); the paper's IPR distinction survives it and is still what justifies
stripping top directions rather than keeping them.

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
   in the loop, so output is deterministic and nothing is reworded or invented.
   Abstractive rewriting stays a later optional flag.

   One deliberate exception, decided 2026-09-10: a **defined pointer is not invented content**.
   Path substitution replaces a file path in retained text with `[P3]` and ships the table that
   defines it in the same output. The marker appears nowhere in the source, so a retained span is
   no longer literally quotable — but nothing was rewritten, the expansion is mechanical and
   exact, and this is what a codec does. The property that matters is that no meaning was
   generated, not that every character is contiguous with the original.

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
| tool_use | shape routing plus a compact call rendering | Target path kept verbatim — it is the identifying fact and supersession's key. Every other value is routed by *shape*, not by key name: code-shaped to the AST bander, shell to stage elision, prose to passage elision. The call is emitted as a call line rather than a JSON object, and the tool's own name is not emitted with it (decided 2026-09-10). **This channel tolerates the most loss of any** (decided 2026-09-10). Its trimming depth **follows the global keep ratio**; see the note below. On the channel, **1.92x at depth 0.33**, which is where the rates this project targets put it; 2.21x is the depth-0.0 figure and is not what a real run realizes. |
| tool_result | line chunking plus selection | Plus supersession dedupe. |
| thinking | segment selection | Lowest default fidelity. |

**The `tool_use` depth knob follows the global rate** (decided 2026-09-10, correcting this
document). An earlier version of the row above said the knob was applied flat rather than scaled
by the global rate. The code has never done that: `quantize_segment` takes a `tool_quality`
override, it defaults to `quality`, and no caller in the sweep or the pipeline sets it, so the
channel is trimmed at whatever rate the run asks of everything else. Rejected: changing the code
to match what the doc claimed. Holding one channel's depth flat while every other kind scales is
a second knob turning inside a grid whose whole discipline is one change at a time, and the case
for a flat depth is an argument rather than a measurement. Sweeping the depth as its own variant
stays queued as separate work — see [PHASE2.md](PHASE2.md#queue) — and only then is there
evidence for holding it anywhere.

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

Two metrics, reported side by side and **never collapsed into one score**:

- **Atom recall.** Fraction of the original's identifiers, paths, numbers and URLs that
  survive. This is what catches "fluent but wrong". Reported per kind as well as overall.
- **Semantic coverage.** Mean over original segments of the maximum cosine to any kept
  segment, plus whole-document cosine.

They measure different things and they disagree — spectral selection wins the first at every
rate measured, budget-shaped TF-IDF wins the second at every rate measured. **That disagreement
is a result, not a tie to be broken** (decided 2026-09-10). Baseball keeps FRV and OAA as two
defensive statistics rather than averaging them into a number that means less than either; the
same applies here. A method is described by its position on both axes, and picking one to crown
a winner would discard exactly the information the two-metric design was built to expose.

Selection and scoring deliberately use **different embedding models** (`bge-small-en-v1.5` to
select, `all-MiniLM-L6-v2` to score). Sharing one model would let the selector be graded by its
own similarity function, which inflates every result.

## Targets

- Rate: 2–5x on real agentic transcripts, degrading gracefully rather than off a cliff.
- Baselines any spectral method is measured against at equal budget: random drop, recency-only,
  TF-IDF. Spectral and budget-shaped TF-IDF **split the two distortion metrics** rather than one
  beating the other, and both are reported. There is no single scoreboard here by design.

## Open design questions

- ~~Whether the PPMI-SVD basis earns its cost at conversation scale, or survives only as a topic
  labeller.~~ **Settled in Phase 1:** it survives only as a labeller. The topical eigenvectors
  measured 0.507 aligned cosine against a threshold of 0.70 that was written down before the
  measurement, and an embedding-matrix SVD drives selection instead. Rejected: keeping the basis
  as a selector on the strength of its interpretability, which is what the pre-registered
  threshold existed to stop. The IPR distinction the paper contributes is untouched by the
  demotion and is still what justifies stripping top directions. See
  [PHASE1.md](PHASE1.md); the same result is stated in `README.md` and `CLAUDE.md`.
- How to drive the whole quality table from a single `--quality` knob.
- Whether the background prior should be generic, personalized from the user's own sessions, or
  both layered. The personalization prior is a committed roadmap item.
- Segment-level versus span-level trimming for prose.
- ~~Whether "extractive" must mean *contiguous substring of the original*.~~ **Settled
  2026-09-10:** it does not. A mechanically-defined pointer whose definition ships alongside it
  is admissible in retained text; see the emit step above. The rejected alternative was requiring
  every retained span to be quotable as-is, which would have ruled out path substitution's token
  saving and its recall guarantee together, and every future side-table mechanism with them.

## References

- **Shin, Madotto & Fung (2018), *Interpreting Word Embeddings with Eigenvector Analysis*,
  NeurIPS IRASL workshop** — the paper the project was seeded with; see
  [Origin](#origin). [OpenReview](https://openreview.net/forum?id=rJfJiR5ooX) ·
  [PDF via Wayback](https://web.archive.org/web/20250405190331if_/https://openreview.net/pdf?id=rJfJiR5ooX)
  (OpenReview refuses scripted fetches; the Wayback copy is the one that resolves) ·
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

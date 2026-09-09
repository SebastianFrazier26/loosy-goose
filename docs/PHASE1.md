# Phase 1 — experiments and results

Run 2026-09-08. Scripts: `experiments/exp_a_ppmi.py`, `exp_b_embed.py`, `exp_c_code.py`.
Raw summary tables as produced by those scripts are in [`results/`](results/); full JSON and
per-transcript plots stay in the gitignored `experiments/output/`.

Phase 1 asked one question: **does the eigenvector machinery from Shin, Madotto & Fung (2018)
survive at single-conversation scale, and does it beat cheap baselines at an equal token
budget?** The short answer is: partly, and not yet.

## Corpus

18 transcripts, 1.5M tokens total, three sources:

| Source | What | License | Committed? |
| --- | --- | --- | --- |
| `trace-commons/agent-traces` @ `112ebd4d` | 28 Claude Code JSONL sessions | CC-BY-4.0 | no, fetched |
| `SWE-Gym/OpenHands-SFT-Trajectories` @ `4aaa5a4a` | 491 rows | MIT | no, fetched |
| `oasst2` validation @ `179dd21f` | dialogue | Apache-2.0 | no, fetched |
| local Claude Code sessions | 5 sessions at ~5K/20K/50K/100K/200K tokens | private | **never** |

Public data is reproducible on any machine with `uv run python experiments/fetch_public.py`.
The local sessions are the author's own work logs: gitignored, never committed, never quoted.
They appear in results tables only as `local_5k` … `local_200k` with content withheld. Anyone
reproducing this on another machine gets the public rows and can regenerate local rows from
their own sessions with `experiments/pick_local.py`.

The single most consequential corpus fact: **`tool_use` + `tool_result` are 51–91% of tokens**
in real agentic sessions. This is what forced the binary `protected` flag to be replaced by a
per-kind quality table — see [DESIGN.md](DESIGN.md#the-quality-table).

## Experiment A — do PPMI-SVD eigen-topics hold up on one conversation?

**Design.** Build a PPMI co-occurrence matrix over a single transcript's vocabulary (window 2,
dynamic 1/d weighting, cds 0.75, shift 1.0), take a truncated SVD, compute the inverse
participation ratio per eigenvector, and label directions by top-|u_k| words. Test stability by
5-fold jackknife: refit on folds, align eigenvectors across folds with the Hungarian algorithm,
report mean aligned cosine. Repeat with a differential-PPMI background prior subtracted.

**Pre-registered criterion, fixed before running:** mean aligned cosine ≥ **0.70** for both the
top-20 eigenvectors *and* the sparsest-20 (highest-IPR) eigenvectors. The sparsest-20 clause is
the one that matters — those are the topical directions the whole approach depends on.

**Result: partial fail.**

| Measure | Result | Criterion |
| --- | --- | --- |
| Top-8 eigenvectors | 0.97 → 0.75 aligned cosine by index | pass |
| Top-20, mean over corpus | **0.713** | pass, but only above ~20K tokens |
| Sparsest-20, mean over corpus | **0.507** | **fail — never reaches 0.70 on any transcript** |
| Sparsest-20, with background prior | 0.578 (improves on 16/18) | still fail |

The dense, stable directions are the ones we do not want; the sparse, topical directions we do
want are the unstable ones. The background prior helps and is worth keeping, but does not close
the gap.

**Paper replication, three findings:**

- Densest eigenvector is a stopword direction. **Replicates.**
- Participation fraction ≈ 27%. **Replicates on small transcripts only** — 26–31% under 5K
  tokens, falling to 13–17% at 60–120K. The paper's constant is a corpus-size artefact.
- IPR correlates with eigenvalue rank. **Does not replicate** — |r| mostly below 0.3.

Topic quality splits cleanly by content, not by size. Identifier-dense sessions give coherent
topics (`build_dir clang patsubst srcs_parser`; `epilog_query restype c_void_p cdll`;
`arg_star arg_named_opt arg_pos`). Prose-dense sessions give junk (`not are be of terms does`;
`you your any no will sure`).

**Verdict:** A is demoted from selector to **labeller** — the stable top-8 directions plus the
differential-PPMI term. It does not drive selection.

## Experiment B — embedding-matrix SVD and leverage-score selection

**Design.** Embed segments with `bge-small-en-v1.5`, mean-centre, SVD, and score segments by
statistical leverage, ridge leverage, or CUR residual. Measure spectrum shape (rank needed for
90/95/99% energy), score concentration (Gini), and agreement between strategies.

**Result: works, but the spectrum gives us nothing to exploit.**

- **No knee.** Rank at 95% energy saturates around 200–220 of 384 dimensions on large
  transcripts. There is no low-rank structure to truncate to.
- **Compression only appears at scale.** k95/n falls from 0.45–0.75 on small transcripts to
  0.10–0.13 on large ones. On a 20-segment transcript, 15 of 20 directions are needed.
- **Leverage scores are flat.** Gini 0.20–0.33. The token budget, not the score, does most of
  the shaping.
- **Strategies converge.** Leverage vs CUR-residual top-10% overlap runs 0.36–0.70 and rises as
  keep ratio grows past 0.5, so strategy choice matters less than it looks.

Direction labels are coherent on public data, but recurring **non-topic directions** show up
across unrelated transcripts — `ensure, modified, todo, proceed, progress` and
`file, updated, successfully, need`. That is a direct empirical confirmation of the
strip-the-bias-directions rule from the literature.

**Verdict:** B drives selection. But it earns that by being reasonable, not by being good — see
the honest read below.

## Experiment C — the code channel

Three sub-experiments, all on `code` and `tool_use` segments.

### C1 — AST-depth banding vs dropping whole segments

**Design.** tree-sitter parse, band lines by AST depth (band 0 = imports, signatures, class and
type declarations), keep bands up to a quality level q, and compare against dropping whole
segments to the **same token count** using TF-IDF and random selection.

**Result: banding wins at equal budget.** Token-weighted over all transcripts:

| q | kept | struct recall | struct cover | tfidf recall | tfidf cover | random recall | random cover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.00 | 1.000 | 1.000 | 1.000 | — | — | — | — |
| 0.66 | 0.932 | 0.978 | 0.992 | 0.952 | 0.870 | 0.954 | 0.991 |
| 0.33 | 0.808 | 0.913 | 0.976 | 0.884 | 0.773 | 0.864 | 0.965 |
| 0.00 | 0.750 | **0.885** | **0.973** | 0.854 | 0.735 | 0.827 | 0.958 |

Banding beats both alternatives on both axes. The coverage gap against TF-IDF drop
(0.973 vs 0.735) is the interesting one: dropping whole code segments destroys semantic
coverage, keeping their signatures does not.

**Two caveats, stated plainly.** First, q=0.00 only buys **1.33x** — banding alone is a weak
compressor. It is a quantizer to stack on top of selection, exactly its role in JPEG. Second,
it has a real failure mode: on `trace-commons_11ef2190` recall collapses 0.917 → 0.583 between
q=0.66 and q=0.33 while random drop holds at 1.000. Banding evicted the atoms there.

### C2 — supersession dedupe

**Design.** Drop all but the latest `Read`/`Edit`/`Write` per file path, plus exact duplicate
tool results. Lossless with respect to final state, so there is no distortion to measure — only
tokens removed.

**Result: 389,529 / 1,503,610 tokens removed corpus-wide = 25.9%, for free.**

| Reason | Share of all tokens |
| --- | --- |
| Superseded edits | 14.3% |
| Superseded reads | 5.0% |
| Exact duplicates | 6.5% |
| **Total** | **25.9%** |

Enormously variable per transcript: 0.0% on `trace-commons_4c09dfa9` (a two-turn session with
no tool calls) up to 70.8% on `swe-gym_row214`, which is almost entirely duplicate tool
results. Real Claude Code sessions cluster in the 0.16–0.50 band.

**This is the single clearest win in all of Phase 1, and it uses no eigenvectors at all.**

### C3 — is a code-specific embedder worth it?

**Design.** Re-run B's leverage selection on code-only segments at keep 0.3 under three
embedders, scored with MiniLM as always.

**Result: no. Stay on `bge-small-en-v1.5`.**

| Model | dim | Verdict |
| --- | --- | --- |
| `BAAI/bge-small-en-v1.5` | 384 | baseline, 11–21 s/100 segments |
| `flax-sentence-embeddings/st-codesearch-distilroberta-base` | 768 | never better; clearly worse on `local_50k` (coverage 0.530 vs 0.771) |
| `Alibaba-NLP/gte-modernbert-base` | 768 | sometimes better (`local_20k` recall 0.421 vs 0.312) but 155 s/100 segments — 7x the cost |

Extra dimensions bought nothing structural: k95/n stayed at 0.72–0.88 regardless.

## Honest overall read

- The one unambiguous, ship-ready win is **C2 supersession**, which is not spectral.
- **C1 banding** is a genuine but modest win, correctly positioned as a quantizer.
- **A** does not support the role the project was designed around; it survives as a labeller.
- **B** is serviceable but the spectrum has no exploitable structure, and its scores are flat
  enough that the token budget does most of the work.
- **No spectral method has yet been shown to beat budget-shaped TF-IDF.** Phase 2 has to
  establish that or the honest conclusion is that Loosy-Goose is "supersede + band + TF-IDF"
  with an eigen-flavoured labeller on top. That would still be a useful tool. It would not be
  the tool described in the design.

## Known defects found while running Phase 1

Not yet fixed; deliberately deferred so as not to desync in-flight experiment runs.

1. **Segmenter, markdown rules.** `---` horizontal rules become standalone prose segments.
   Dozens of identical embeddings then hijack the top SVD directions (seen on
   `trace-commons_47d9fc04`).
2. **Segmenter, unbounded paragraphs.** An unbroken 7,602-token paragraph escapes
   `max_prose_chars`; `split_prose`/`_merge` have no hard character-length fallback. Budget
   sticks at 0.148 kept.
3. **Transcript loader, escape leakage.** `transcript.py` JSON-dumps `tool_use` blocks, so
   literal `\n` escapes reach the tokenizer (`n\n`, `math\nimport`). Fix by stripping `\\[ntr]`
   in `tokenize`, unescaping upstream, or accepting it — undecided.

## Flagged defaults awaiting confirmation

Chosen by implementation rather than by explicit decision. All currently kept as-is.

1. Dynamic 1/d co-occurrence window weighting, rather than hyperwords' linear decay.
2. Interleaved (not contiguous) jackknife folds.
3. `differential_ppmi` keeps foreground PPMI for words absent from the background.
4. `ipr_weight` normalization in `TopicConfig`.
5. Min-max normalization of scores before the recency blend in `budget.select_by_score`.

## Phase 2

The unified rate-distortion runner: every method — A-labelled, B-selected, C-quantized, and the
three baselines — on one curve per transcript at matched token budgets. That is the experiment
that decides whether the eigen-machinery stays in the design.

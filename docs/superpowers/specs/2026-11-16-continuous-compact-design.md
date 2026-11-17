# Continuous compaction pilot (2026-11-16)

Status: experiment, small pilot. Branch `exp/continuous-compact` (off `exp/slot-compress`, reusing its harness).

## Intent

Loosy loses to a frontier summary on one-shot compaction at equal budget. Continuous compaction, which runs after
every model turn, is the one setting where loosy's cost profile matters: it is free, local and deterministic, so it can run
every turn, which a frontier summary cannot do cheaply. The question is whether running loosy continuously gives
better accuracy per unit of context cost than threshold compaction (`/compact`-style), and whether trimming ever
beats keeping the full context.

## User decisions (2026-11-16)

1. Test both continuous variants: **recompute** and **freeze**.
2. Run a small replay pilot first, not live agent runs.
3. Drop the full-context probe filter for this experiment, so that a context-rot gain over `full` is detectable.

## Arms

All arms are evaluated at checkpoint turn `t` on `history = turns[:t]`. Each arm keeps the same 4k verbatim
tail (`exi.TAIL_TOKENS`) and compresses only what is older.

- `full`: the whole history, verbatim.
- `thresh`: threshold compaction. History accumulates verbatim. When the context exceeds the cap `C`, Sonnet
  summarises everything before the tail into a summary targeting `C * 0.33` (the exi summary prompt). Later
  compactions summarise the previous summary plus the verbatim text since it, so the drift compounds as it does in
  real `/compact` use. The context at `t` depends on the turns at which compaction fired.
- `crec`: continuous loosy, recomputed. At every turn the context is the shipped loosy over the whole pre-tail
  history, at keep `K`. It is stateless, does not compound, and changes its prefix between turns.
- `cfrz`: continuous loosy, frozen. When pre-tail text not yet compressed reaches a chunk of `F` tokens, that chunk
  is compressed once with shipped loosy at keep `K` and appended to a frozen prefix that never changes. The
  prefix is cacheable, and an early drop cannot be revisited.

## Fixed in advance (agent's calls)

- **Set E:** 10 swe-gym transcripts, chosen with `exi.select_transcripts` logic at seed `exi.SEED + 4` and
  disjoint from Phase A/B and sets C/D. A transcript qualifies only if it is long enough for `thresh` to fire at least
  twice before the last checkpoint.
- **Checkpoints:** 3 per transcript, at the first turn boundary at or after 50%, 75% and 100% of the transcript's
  tokens, which gives 30 checkpoints. A checkpoint that falls inside a tool call moves to the end of that action
  (`exi._action_end`).
- **Parameters:** `C` is 40% of the transcript's total tokens, `K` = 0.33 (loosy `medium`), and `F` = 4,000 tokens.
- **Probes:** gpt-6-sol (codex) writes `exi.N_PROBES` probes per checkpoint, about pre-tail content only, through the
  exi prompt. There is **no full-context validity filter**: every probe counts, for every arm, including `full`.
- **Answering and grading:** Haiku answers all of a checkpoint's probes in one call per arm, and Sonnet grades, as in
  exi. The "what should the agent do next" probe is out of scope for the pilot.
- **Cost metrics, computed offline for every turn, not only at checkpoints:**
  - context tokens per turn;
  - uncached input tokens per turn, the tokens after the longest common prefix with the previous turn's context,
    as a stand-in for prompt-cache misses.
  Both are summed and averaged over each transcript. `thresh` needs its summaries for every turn, so they are
  computed once per compaction event and cached.
- **Version stamps:** `CONT_VERSION = 1` covers the freeze, recompute and threshold simulation, and goes into every
  record key, per the repo rule.
- **Call estimate:** 30 probe-generation calls + 120 answer calls + 120 grade calls + about 30 compaction summaries,
  roughly 300 calls in total. The cap is 360. The run stops on a 429 or spend-limit error and runs a `--dry-run` first.

## Report

`docs/results/exp_continuous_pilot.md`. For each arm it reports:

- accuracy, as a bootstrap CI over transcripts;
- mean context tokens per turn;
- mean uncached tokens per turn;
- paired deltas and exact sign-test p values for:
  - `cfrz` vs `thresh`
  - `crec` vs `thresh`
  - `cfrz` vs `crec`
  - each arm vs `full`

Accuracy is also broken down by checkpoint position (50/75/100%).

## Reading the result (fixed now, before the data)

This is a pilot at n=10, so it is directional: it decides whether a larger run is worth it, not whether the hypothesis is true.

- **Worth scaling up:** a continuous arm is within 0.05 of `thresh` on accuracy at clearly lower uncached tokens, or beats it.
- **Context-rot signal:** any arm scores above `full`, with a positive delta at sign p<0.2.
- **Stop:** both continuous arms trail `thresh` by more than 0.10 with no cost advantage.

## Deviations (2026-11-16, recorded while building the harness)

Harness: `experiments/continuous/` (`sim.py`, `run.py`), tests in `tests/test_continuous.py`.

- **Model-turn boundaries.** "Every turn" for the cost metrics, `thresh` firing and `cfrz` chunking means
  every point where the model is called: the end of history `t` such that `turns[t]` starts an assistant
  action, plus `t = len(turns)` and the checkpoints. `thresh` tests `context > C` at those points only.
- **Checkpoint snapping.** The checkpoint is the first `t` with cumulative tokens at or past 50/75/100%. In the
  swe-gym transcripts a tool call is an assistant turn followed by a user turn carrying the result, so
  `_action_end` alone never moves a boundary that follows a lone assistant turn. A checkpoint whose history
  ends on an assistant turn is moved to `exi._action_end` and then past the answering result turn, so the
  history always ends where the model acts. A single large tool result can make two shares land on the same
  `t` (swe-gym_row397: 50% and 75%); the checkpoints are kept as separate probe sets, so the pilot stays at 30.
- **Token accounting for `thresh`.** The context size compared with `C` is the summary's token count plus
  the sum of `exi._turn_tokens` over the verbatim turns, ignoring header text. The cap `C` is 40% of the
  same per-turn total.
- **Summary chain.** A later compaction summarises `[Summary of the session so far]` + the previous summary +
  `[Session since that summary]` + the verbatim turns since, with the exi summary prompt at target
  `C * 0.33` tokens. Summaries are keyed per transcript and event (`thresh.v1.e<k>`) and shared by all of a
  transcript's checkpoints.
- **Headers.** `thresh` and `cfrz` mix compressed and verbatim text before the tail, so they carry their own
  "earlier part" headers rather than the exi `summary` / `loosy` ones. `full` and `crec` use the exi headers
  unchanged. Before its first compaction `thresh` is identical to `full`.
- **Uncached-prefix metric.** Measured on a cost context without the moving "most recent part" marker, which
  would otherwise make every arm miss from the marker onward (the real prompt of a continuing session has
  no such marker). The prefix is the character-level common prefix with the previous boundary's context,
  and the tokens counted are those of the remainder.
- **Placeholder summaries** (offline cost metrics, `--plan`, set E qualification) are `[placeholder summary k]`
  followed by filler words, sized to the target. Real summaries run up to 1.10x the target before being
  cut, so real compaction counts can differ slightly from the planned ones.
- **Scoring.** Every probe counts (no full-context filter). A transcript's arm score is the mean of its three
  checkpoint scores and is used only if all three are graded.
- **Report location.** `--report` writes `docs/results/exp_continuous_pilot.md` and a copy under
  `experiments/output/continuous/`.
- **Qualification.** "Thresh fires at least twice before the last checkpoint" is tested with placeholder
  summaries. It is nearly always true at these parameters; with `C` = 40% of a 20-30k-token transcript, `C`
  is small enough that compactions can fire on consecutive large tool results (up to 7 events on one transcript).

## Outcome (2026-11-16)

310 calls with 0 failures. The results are in `docs/results/exp_continuous_pilot.md`.

- **Scale-up criterion: met, weakly.**
  - `cfrz` trails `thresh` by 0.032 [-0.07, +0.01] (sign p=1.0), within the 0.05 bar, at about 20% fewer uncached tokens per turn (1544 vs 1930) and the same context size.
- **No context-rot signal.** No arm beats `full`; `thresh` is level with it (-0.008).
- **`crec` loses.**
  - It trails `thresh` by 0.097 (p=0.021) and `cfrz` by 0.065 (p=0.004).
  - It also has the worst cache cost (3966 uncached tokens per turn).
- **Unexpected finding.** At equal context size, compressing chunk by chunk (`cfrz`) beats compressing the whole history at once (`crec`), and `cfrz`'s gap to `thresh` closes to 0 by the 100% checkpoint.
  - Two explanations, not yet separated: a per-chunk budget spreads coverage evenly, or `cfrz` keeps the not-yet-chunked text verbatim.
  - If it is the first, it bears on one-shot loosy too.
- **Caveats:**
  - n=10 transcripts.
  - With C=40%, `thresh` fired 3–7 times per transcript, which is harsher than real `/compact` use.
  - Unfiltered probes make `full` about 0.94, not 1.0.

## Stage 2 plan (user-approved 2026-11-16)

The user approved scaling up and running the per-chunk ablation. They also proposed a decaying arm, in which older chunks are recompressed again and again, so they keep shrinking with age and the context gets a recency bias. Documentation is updated at every stage, not only at the end.

### 2a. Per-chunk ablation on one-shot compaction (runs first, cheaper)

This separates the two explanations for `cfrz` beating `crec` (+0.065). Either an even per-chunk budget spreads coverage better, or `cfrz` keeps the not-yet-chunked text verbatim.

- **New arm `chunk`.** The pre-tail text is split into consecutive chunks of about F=4,000 tokens at segment boundaries. Shipped loosy runs inside each chunk, and each chunk gets the same budget share (level budget × chunk tokens / pre-tail tokens). The last chunk is compressed like the others, so there is no extra verbatim text. `CHUNK_VERSION = 1`.
- **Data.** Sets C (10) and D (20) at light, medium and heavy. It reuses their stored probes, full-context filter and `loosy` and `summary` answers, so only the new arm costs calls.
- **Calls.** About 180: 30 transcripts × 3 levels × (answer + grade). Cap 220.
- **Reading.**
  - If `chunk` beats `loosy` (paired, sign p<0.1), explanation (a) holds, and per-chunk selection is a candidate default for one-shot loosy too.
  - If it does not, (b) explains the pilot.

### 2b. Scaled continuous run

- **New set F.** 25 swe-gym transcripts, seed `exi.SEED + 5`, disjoint from A–E. There are 3 checkpoints each (50/75/100%), 75 in all.
- **Cap.** C = 65% of the transcript's tokens. This is more realistic than the pilot's 40%, so `thresh` fires less often.
- **Arms.**
  - `full`, `thresh` and `cfrz`, as in the pilot. `crec` is dropped, because the pilot settled it.
  - A new arm, `cdec` (decaying freeze, LSM-style tiers). A chunk that ages out of the tail is compressed at keep `k` into tier 0. When a tier holds `m` chunks, they are merged and recompressed at keep `k` with shipped loosy over their kept segments, into one chunk of the next tier. Older text is therefore compressed k, k², k³… times. Tiers are laid out oldest-first, so a merge rewrites only the end of the context, and the cached prefix survives most turns. `DEC_VERSION = 1`.
  - `k` and `m` are calibrated offline, with no calls, so that `cdec`'s mean context tokens per turn match `thresh`'s within 5%. That keeps the comparison at equal context size. The calibration is recorded here.
- **Calls.** About 75 probe generations, 300 answers, 300 grades and about 50 summaries, roughly 725 in all. Cap 850.
- **Reading.**
  - It is the same rule as the pilot, now at n=25.
  - **Worth adopting:** `cfrz` or `cdec` is within 0.03 of `thresh` (or above it) at lower uncached tokens.
  - **Recency bias:** `cdec` vs `cfrz`, paired.

## Stage 2a log

- **2026-11-16, built.** The `chunk` arm is in `experiments/slot_compress/chunked.py` and is wired into `downstream.py` for sets C and D.
  - The shares are taken on the pool after supersession, so they sum to the level budget.
  - Unused share spills forward from chunk to chunk. A final global top-up pass then spends whatever is left.
  - Without the top-up, `chunk` filled only 85–90% of its budget, against `loosy`'s 99.7%. That would have confounded the comparison, so the top-up was added to keep the repo's equal-budget contract.
  - With it, `chunk` is within 0.1–0.8% of `loosy` on segment tokens.
  - Rendered earlier-stretch tokens run 0.5–1.4% higher for `chunk`, probably because it has more segment boundaries. That is a slight context-size edge for `chunk`, and it is noted here.
  - The report now always covers every arm of a set, which fixes the old `--conditions` overwrite bug.
- **2026-11-16, launched.** Set C, then set D, with `--conditions chunk`, capped at 70 and 130 calls.
- **2026-11-16, result.** 180 calls, 0 failed.
  - Rejected: (a) per-chunk budget never beats `loosy`. It is −0.060 at set C light (p=0.031) and level on set D.
  - Kept: (b) the pilot's `cfrz` edge comes from its longer verbatim recent window.
  - Results: `docs/results/exp_chunk_ablation.md`. The shared C/D report files were not overwritten, because a run limited to one condition loses the other arms' token columns.
  - Implication for 2b: `cfrz` and `cdec` carry up to 4k more verbatim text than `crec`. This is matched in context size against `thresh`, which also keeps verbatim text since its last compaction, so the 2b comparison stays fair.

## Stage 2b log

- **2026-11-16, built.** Set F has 25 swe-gym transcripts (labels in `experiments/output/continuous/f_selection.json`), disjoint from A–E. It has its own checkpoint under `experiments/output/continuous/f/`, and set E's records are untouched.
- **`cdec` calibration** (offline, mean ctx/turn vs `thresh` 12,449 with placeholder summaries):
  - m=2 k=0.8 lands at −0.2%, m=3 k=0.7 at −1.9%, m=4 k=0.7 at +1.4%. Each is within 5%.
  - **m=4, k=0.7** was chosen, because it has the lowest uncached tokens of the three (1,611 per turn).
  - Even so, it has no cache advantage over `thresh` (1,562 per turn). Matching `thresh`'s size forces a high k, so each merge rewrites a large suffix.
- **Size mismatch found, and a fix.** At keep 0.33, `cfrz` is 18% smaller than `thresh` on F (10,177 per turn). That would confound `cdec` vs `cfrz`.
  - So a size-matched freeze arm, **`cfrzm`**, was added, with its keep calibrated so it lands within 5% of `thresh`. `cfrz` stays as the arm comparable to the pilot.
  - The recency-bias test is now `cdec` − `cfrzm` at equal size. This is the agent's call, and it adds about 150 calls, so the F cap rises from 850 to 1,000.
- **Known limitation.** When `cdec` merges tiers, the `[…]` gap markers inside partly dropped blocks are lost, so an older-tier tool result can read as contiguous.
- **`cfrzm` calibration.** Mean context tokens per turn, against `thresh`'s 12,449:

  | keep | context tokens/turn | vs `thresh` |
  |---|---|---|
  | 0.40 | 10,783 | −13.4% |
  | 0.45 | 11,176 | −10.2% |
  | 0.50 | 11,569 | −7.1% |
  | 0.55 | 11,955 | −4.0% |
  | 0.60 | 12,355 | −0.8% |

  **0.60 was chosen.** It is the closest to both `thresh` and `cdec` (12,625), so `cdec` − `cfrzm` is not confounded by size.

  Offline uncached tokens per turn at equal size: `cfrzm` 1,521, `thresh` 1,562, `cdec` 1,611.
- **2026-11-16, launched.** The F run is 875 planned calls, capped at 1,000.
- **2026-11-16, result.** 865 calls, 0 failed. Results: `docs/results/exp_continuous_f.md`.
  - **Not worth adopting.** With real summaries, `thresh` costs fewer uncached tokens than any continuous arm (1,366 per turn, against 1,411 for `cfrz`, 1,521 for `cfrzm` and 1,611 for `cdec`). It is also more accurate:
    - `cfrzm` −0.024 (p=0.29) and `cdec` −0.021 (p=1.0), at equal size;
    - `cfrz` −0.081 (p=0.011), 18% smaller.
    The accuracy gaps are inside 0.03, but the cost condition fails, so `thresh` dominates on both axes.
  - **The pilot's cache edge came from the low cap.** At C=40%, `thresh` fired 3–7 times per transcript. At C=65% it fires about twice. The real summaries also came in shorter than the placeholders the offline estimate used.
  - **Recency bias: null.** `cdec` − `cfrzm` = +0.003 (p=0.50) at equal size. Repeatedly recompressing older chunks neither helps nor hurts.
  - **Context rot: none.** No arm beats `full`, and `thresh` is level with it (−0.015).
  - **Size matters most for the continuous arms.** `cfrz` → `cfrzm` (keep 0.33 → 0.6) gains +0.056.
  - **Closed for frontier-model settings.** The no-frontier-model case (threshold compaction with loosy in place of Sonnet, against continuous loosy) was not tested.

## Stage 3 plan: no frontier model (user-approved 2026-11-16)

**Question.** If no frontier model is available, is it better to run loosy at a threshold, as a drop-in for `/compact`, or continuously?

- **New arm `tloosy`.** Threshold compaction exactly as in `thresh`, with shipped loosy in place of the Sonnet summary.
  - When the context exceeds C, the pre-tail text (the previous compacted text plus the verbatim text since) is compressed with loosy.
  - Later events compress the already-compacted text again, so loss compounds.
  - Its compaction target is calibrated offline so that mean context per turn is within 5% of `thresh`'s real value (12,342). This matches the other arms' sizes. `TLOOSY_VERSION = 1`.
- **Data.** Set F, reusing its stored probes and the stored `full`, `thresh`, `cfrzm` and `cdec` answers. Only `tloosy` costs calls: about 75 answers and 75 grades, so about 150. Cap 180.
- **Paired comparisons.**
  - `cfrzm` − `tloosy` and `cdec` − `tloosy`: continuous vs threshold, both without a frontier model.
  - `thresh` − `tloosy`: the price of having no frontier model in this setting.
- **Reading.** Continuous loosy is the better no-frontier-model policy if `cfrzm` or `cdec` beats `tloosy` (sign p<0.1) or matches it at lower uncached tokens.

## Stage 3 log

- **2026-11-16, built.** The `tloosy` arm is in `experiments/continuous/sim.py`, with key `tloosy.F.v1.t1.s0.33`.
- **Real `thresh` mean**, from stored summaries: 12,342 context tokens per turn.
- **Calibration** of the compaction target as a share of C:

  | share of C | context tokens/turn | vs `thresh` | uncached tokens/turn | events per transcript |
  |---|---|---|---|---|
  | 0.20 | 12,295 | −0.4% | 1,308 | 1.5 |
  | 0.25 | 12,403 | +0.5% | 1,416 | 1.8 |
  | 0.30 | 12,472 | +1.0% | 1,486 | 1.9 |
  | 0.33 | 12,482 | +1.1% | 1,515 | 2.0 |
  | 0.40 | 13,024 | +5.5% | 1,600 | 2.0 |

  **0.33 was chosen.** It is the same target as `thresh` and gives the same event schedule (2.0 per transcript), and it lands within 5%. Smaller shares fire fewer compactions, which would make the comparison less like-for-like.
- **Offline uncached tokens per turn:** `tloosy` 1,515, `cfrzm` 1,521, real `thresh` 1,366.
- **2026-11-16, launched.** 150 new calls, capped at 180.

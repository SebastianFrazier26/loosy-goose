# Falsification tests for extractive compression (2026-10-23)

Status: experiment. Branch `exp/slot-compress`.

## Intent

The user's question is whether loosy's extractive strategy has any advantage over summary-based compaction, or can
improve it. Every extractive and rewrite variant so far trails a Sonnet summary by 0.10–0.14 probe-QA accuracy at
equal budget (`docs/results/exp_slot_downstream_d.md`). Before retiring the idea, test the three places it could
still win. If all three are negative, the user treats it as retired (pending their confirmation).

## Tests (user-approved 2026-10-23)

1. **Hybrid — summary plus verbatim.**
   - Sonnet writes a summary targeting a fraction `f` of the level budget B.
   - Shipped loosy selection then fills the rest of B (B minus the summary's actual tokens) with verbatim segments.
   - Arms: `hyb70` (f=0.7) and `hyb50` (f=0.5), each at light, medium and heavy.
   - Compared against `summary` (f=1) and `loosy`.
2. **Exact-detail versus gist probes.** Each stored set-D probe is classified with zero new calls: it is *exact*
   if its reference answer contains an atom (`segment.extract_atoms`), otherwise *gist*. Accuracy is reported per
   class for every set-D condition and for the new arms.
3. **Local summary.** The same summary task at the same target, written by a local Qwen3-1.7B (Apache-2.0, greedy
   decoding). Arm `lsum`, at all three levels. This tests whether the gap is "abstractive" or "frontier model".

## Fixed in advance (agent's calls)

- Set D (20 swe-gym transcripts) is reused. Its probes, full-context filter and existing arms are reused as stored.
  Nothing is tuned on it; the splits are fixed at 0.7 and 0.5.
- Haiku answers and Sonnet grades, as before.
- The hybrid layout is the summary block first, then the verbatim segments in transcript order, then the 4k tail.
- The local summary uses the summary prompt exi uses, with chunked map-reduce if the earlier text exceeds the
  model's context. MLX is used if it is available, else MPS transformers. Its generations are cached.
- About 480 Claude calls, with the cap at 520. The run stops on a spend limit.

## Decision rule

The idea is retired if all three of the following hold:

- **Hybrid:** neither hybrid arm beats `summary` at any level (paired, sign p<0.1, positive delta).
- **Exact-detail:** loosy does not beat `summary` on exact-detail probes at any level.
- **Local:** loosy does not beat `lsum` at any level.

Otherwise, report the niche found.

## Deviations (agent's calls, 2026-10-23)

- `lsum` uses the same user turn as the Sonnet summary, plus a system message and the assistant prefill
  "Summary of the session so far:". Without them Qwen3-1.7B continued the transcript with tool calls instead of
  summarising. Its outputs often come in far below target, and a looping output is cut at a sentence boundary.
- `lsum` runs on mlx-lm 0.32.2 (arm64 macOS only, in the `slot` extra).

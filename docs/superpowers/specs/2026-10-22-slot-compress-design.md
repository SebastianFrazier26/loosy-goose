# Slot compression — design (2026-10-22)

Status: experiment (throwaway until downstream QA says otherwise). Branch `exp/slot-compress`.

## Intent

The user's original loosy-goose idea, untested so far: compress *prose and reasoning* the way an image codec
compresses pixels — transform, quantize, then code redundancy across the result — instead of only keeping or
dropping whole segments. Every extractive variant to date loses on *drop loss*; this attacks the other axis,
shrinking what is kept so more fits in the budget.

## The scheme (user-decided)

1. **Transform — restructure into slots.** Each sentence (or tight sentence pair) becomes one row in a fixed
   schema, roughly `[Action] [Agent] [Object] [Recipient] [+extras]`. Meaning is carried by slot position, not
   word order (the user's analogy: Greek case endings). Rigid structure on the writing side, a shared schema on the
   reading side.
2. **Quantize — fewest words per slot.** Each slot shrinks to as few words as possible
   ("draft a plan for iterative testing with 100% coverage" → "test fully"). Nuance loss here is accepted; it is
   the point of compaction.
3. **Cross-row coding.** The rows form a sentence matrix; compress across it (repeated column values written once,
   duplicate rows merged, rows adding nothing new dropped) — closer to today's novelty selection, applied to rows.

User-decided constraints:

- **Negation, modality and cause survive.** "don't push main" must not lose the "don't".
- **Deterministic and lightweight first**; alternative engines may be tested side by side.
- **Both read modes are tested:** the answering model reads slot rows directly with a short legend, and rows
  translated back to English.
- Scope (from the handoff, unchanged): prose and thinking only. Code and tool output stay extractive; tool-call
  inputs are never touched. Atoms (identifiers, paths, numbers, error strings) pass through verbatim.

## Builder's calls (made by the agent while the user was away — review these)

- **Parser: spaCy `en_core_web_sm` dependency parse** (MIT, ~12 MB, CPU, deterministic) as the plumbing that
  finds verb/subject/object; the slot schema, quantizer and cross-row coder are ours. Rejected for the first
  arm: a prompted SLM emitting slots (non-deterministic, heavier) — kept as a later comparison arm if the
  deterministic arm shows signal.
- **Slot schema v1:** `NEG/MOD` flag, `ACT` (verb lemma), `AGT` (subject head + compounds), `OBJ` (object head +
  compounds + atoms), `TO` (dative/recipient), `X` (one prepositional/clausal extra, preferring ones carrying
  atoms or cause/condition markers: because/if/after/before/until/unless). Atoms anywhere in a sentence are
  never dropped: unplaced atoms append to `X`.
- **Quantizer v1:** drop determiners, auxiliaries, hedges and filler adverbs; keep content-word heads with their
  compound/amod modifiers only when they carry an atom or negation. Pronoun agents `I`/`we`/`you` → omitted
  (the speaker role already says who).
- **Cross-row v1:** a slot equal to the previous row's same slot is written as empty (carry-forward); exact
  duplicate rows elided. Row-level novelty dropping is left to the existing selector downstream, which already
  operates on segments — compressed prose segments simply compete at their new, smaller token cost.
- **Translate-back mode:** deterministic template realisation of a row (`[not] ACT OBJ [to TO] [X]`), no model.
- **Placement in the pipeline:** supersede → slot-compress prose/thinking → line-keep code dumps (the spike's
  best rule) → selection. Budget stays in tiktoken o200k.
- **Fallback:** a sentence the parser can't slot (no verb root, a list item, a table row, a fragment holding
  mostly atoms) passes through unchanged.
- **Version stamp:** `SLOT_VERSION` per the repo rule; any change to schema, quantizer or coder bumps it.

## v2 changes (agent's calls, 2026-10-22; `SLOT_VERSION` 2 — review these)

v1 cut only 12.7% (direct) / 19.3% (english) of prose+thinking tokens offline; the user's goal is fewest words
per sentence, so v2 trades nuance for brevity (negation, modality, atoms still survive). Offline: 40.9% direct /
40.0% english, atoms 100%, verbatim fallback 3.9% / 11.5% of slotted units.

- **Row format:** the `ACT|AGT|OBJ|TO|X` pipe layout is gone (pipes and `-` placeholders cost tokens). Direct:
  `[agent: ][flags]act obj [to recipient] [cause/atoms]`; the agent is written only when non-pronoun and different
  from the last one written (it "holds until changed"); copula is `subj=attr`; flags unchanged (`~ ! ^ + ?`).
  English: the same row as `agent [modal] [not] act obj ...` with periods, agent repeated. LEGEND is 60 tokens.
- **Row merge:** consecutive rows with equal agent/flags/act merge objects (`fix a, b`).
- **X slot:** only atom-bearing or cause/condition (`because/if/when/...`) material, reduced to content words
  (no "try", "which", "that"). Keeping short plain prepositional extras (<=2 or <=3 content words) was measured and
  costs 1-2 points of reduction; left off.
- **Filler:** leading "Great,/Now,/Let me/Let's/I'll/Looks like/please..." stripped before parsing (meta-narration
  falls to act+obj); filler-only sentences dropped. Expletive subjects (it/this/there) dropped; a verb with no object
  and a generic preposition (at/in/into/for...) promotes the preposition's object to OBJ ("look at the file" -> `look file`).
- **Fragments:** verbless or very short sentences become content-word fragments (modals and negations kept) instead
  of verbatim fallback.
- **Lines:** list items are slotted behind their marker (the largest single gain, ~14 points); soft-wrapped lines are
  joined when the next starts lowercase and the previous has no terminal punctuation; `**bold**` is unwrapped; all-caps
  `DON'T` is lowercased so spaCy sees the negation; code-like lines (comments, `;`/`{` endings, symbol-dense,
  traceback `E   ` lines, `---` rules) pass through verbatim; the atom-heavy passthrough threshold rose 0.5 -> 0.75.
- **Measured, no effect:** leading-filler strip and row merge each move <0.1 point (the corpus rarely
  has the patterns; spaCy already drops interjections); unwrapping ~1 point.

## Evaluation

Downstream probe QA only (offline string-match survival ranked variants backwards twice):

- Fresh held-out public transcript set, disjoint from Phase A (20) and Phase B (10).
- Haiku answers, Sonnet grades; probes generated by a model other than the summariser.
- Conditions at light/medium/heavy: `loosy`, `loosy+linekeep`, `loosy+linekeep+slot-direct`,
  `loosy+linekeep+slot-english`, Sonnet summary, full.
- Call budget fixed up front and capped; stop on a 429 spend-limit error.
- Report before/after excerpts for readability, and the prose/thinking token reduction achieved by the
  transform alone.

## Non-goals

- Productising anything. Shipped `/loosy` stays untouched on this branch's main-merged code paths.
- Touching code, tool_result or tool_use content with the slot transform.
- Fine-tuning a model (a later arm only if a prompted/deterministic arm shows signal).

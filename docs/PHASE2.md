# Phase 2 — plan, decisions, and what is still open

Status as of 2026-09-09: in progress. Phase 1's results are in [PHASE1.md](PHASE1.md); the
algorithm specification is in [DESIGN.md](DESIGN.md). This document is the working plan — what
Phase 2 is trying to settle, which decisions are already locked and why, and which are not.

Phase 1 ended on an uncomfortable note: the one unambiguous win (supersession) was not
spectral, and no spectral method had been shown to beat budget-shaped TF-IDF at an equal
budget. Phase 2 exists to settle that, and to find compression outside `code`, which is a small
share of a real agentic transcript.

## The measurement problem that had to be fixed first

Everything in Phase 2 is graded by atom recall, so a defect in the atom extractor is a defect
in every number the project has ever produced. Two mirror-image defects were found and fixed
before any new sweep was run.

A `tool_use` segment's text is `json.dumps` output. That means an edit body's newline arrives
as the two literal characters backslash and `n`, and a genuine Windows path's separator arrives
doubled. The path pattern read the first as a directory separator and invented atoms out of
runs of escapes — **8.7% of the corpus's 32,344 distinct atoms were this junk**, and 7,022 of
the 7,156 junk mentions sat in `tool_use`, which is a protected kind. Protected means almost
always retained, which means those junk atoms were almost always scored as "recovered". Every
Phase 1 and early Phase 2 `atom_recall` was inflated by them.

The same serializer made real paths invisible: the pattern cannot span the empty component
between two backslashes, so `venv\Scripts\pip.exe` inside a tool call matched nothing at all
and was never counted as an atom the compressor could lose.

**Fix: decode the payload before matching** (`segment.scannable`), scanning values only.
Corpus atoms fall 32,344 to 27,189.

**Rejected: tightening the path regex.** It removes 2,697 atoms rather than 5,137, leaves
residual junk (`dev/null\necho` survives it), keeps boilerplate parameter names in the atom
set, and — decisively — does nothing about the invisible-Windows-path half of the defect.

**Rejected: fixing it upstream in `transcript.py`.** The `json.dumps` there is deliberate;
`code.parse_tool_payload` and `code.quantize_tool_use` both round-trip through JSON. Changing
the rendering would alter segment text and invalidate every stored metric. The defect is in the
pattern, so the fix belongs in the pattern.

Because atoms are the recall metric's denominator, `segment.ATOMS_VERSION` is now folded into
the Experiment D `segments_digest`. A changed extractor now invalidates stored checkpoint
records the same way a changed segmenter does, instead of loading silently beside them.

## Path substitution

A file path is repeated far more often than it is informative. Post-fix, the corpus holds 7,373
distinct paths and **17,760 mentions of an already-mentioned path, costing 136,363 tokens**.
Naming each path once in a table and replacing every mention with a short marker does three
separable things, and they must be measured separately because only the first is certain:

1. the table **guarantees** the path survives, whatever selection drops;
2. the marker is shorter, so the text shrinks — 58,177 tokens net of the table's own 24,906,
   which is **3.87% of the 1,503,397-token corpus**;
3. scoring stops seeing long repetitive path strings, which may rank segments better or worse.

Implemented in `src/loosy_goose/paths.py`. Not yet wired into the sweep.

### Decisions locked

- **One row per file.** The table names each path exactly once, in first-mention order.
- **Numeric marker `[P<n>]`, not the basename.** Only **52.0%** of basenames are unique within
  a transcript, so a basename marker needs disambiguation half the time and still nets less
  (**1.93%** against 3.87%). One option is simply better, so this is not a configuration knob.
- **Substitution runs after supersession.** Supersession matches `file_path` values across tool
  calls; substituting first leaves it comparing markers, and its reported paths become
  meaningless to a reader of the results.
- **The table's guarantee is derived from the rendered table text**, not from the path list, so
  it cannot claim to preserve more than the output actually says.
- **`min_mentions` is a knob**, because it is a genuine tradeoff rather than a tuning constant.
  At 2 the table holds only paths that repeat, which is token-optimal — a row for a
  single-mention path costs more than the mention it replaces. At 1 it holds everything, which
  is recall-optimal for exactly the opposite reason: a path mentioned once is the one selection
  is most likely to drop. Default 2; which one wins is for the arms to say.

### Metric split, and why it was mandatory first

A path table would guarantee **28.8% of the final-state atom set** outright. Emitted without a
correction, it lifts `atom_recall` for every method at once, and no comparison across the fix
boundary means anything. So `metrics` now separates what a side table hands over from what
selection had to earn:

- `atom_recall_earned` / `atom_recall_final_earned` — scored only over atoms no table guarantees;
- `guaranteed_share` — how much was free;
- `compression_ratio(extra_tokens=)` — the table charged against output size, so a mechanism
  that buys recall by emitting a lookup pays in the same currency as selection.

With no table the earned columns equal the originals, so every prior number still reproduces.

### The four arms

Run at matched budgets, on the same curve as every other method:

| Arm | What it does | What it isolates |
| --- | --- | --- |
| A | baseline, no substitution | control |
| B | substitute for scoring only; emit the original text, no table | effect 3 alone — does scoring improve when paths stop dominating the embedding? |
| C | emit the table, substitute nothing | effect 1 alone, at full cost — pure overhead, not shippable |
| D | full substitution: table plus markers | all three effects together |

C is deliberately not a shippable configuration. It exists to separate "recall rose because
facts were guaranteed" from "recall rose because selection got better" — without it, arm D's
result is uninterpretable.

## Compression outside code

`code` is not where the tokens are. A survey of `tool_result` across the corpus
(`experiments/survey_tool_output.py`) found **2,273 blocks totalling 679,784 tokens**, and the
mass is concentrated: blocks of 500 tokens or more are **11.4% of blocks but 68.2% of tokens**.

The largest single shape is numbered file reads — the `123→` line prefixes a file-read tool
emits: **222 blocks, 219,759 tokens, 32.3% of all tool-output tokens**. Of those, **99.3% never
reach AST banding at all**, because `looks_like_code_dump` tests lines that still carry their
line-number prefixes and so does not recognise the content as code. Stripping the prefixes
before the classifier routes 188 of 222 blocks — 166,868 tokens, **24.5% of tool-output
tokens** — into the shrinker that already exists.

**Caveat, recorded deliberately:** reaching the code path is not the same as being compressed
well there. Banding alone buys only 1.33x. This is a routing gap, not free saving, and it must
not be reported as the latter.

The remaining `other` bucket: 729 blocks / 109,086 tokens under 500 tokens each (too small to
matter individually), 64 blocks / 82,249 tokens code-shaped, 48 blocks / 62,789 tokens
prose-shaped, 4 blocks / 12,404 tokens with average line length over 200 characters.

**The numbered-file routing fix is written but deliberately held.** It is not applied, so that
it can be measured against path substitution independently before the two are stacked. Stacking
first would make it impossible to attribute the result to either.

## Still open — do not treat as settled

- **Whether arm D may give up strict verbatim output.** Arm D's text is not a contiguous
  substring of the original; it is verbatim only *after* expanding the table through
  `paths.expand`. Arms B and C keep the strict property. Whether that trade is acceptable is to
  be decided from what the arms report, and has explicitly not been decided yet.
- **`min_mentions` 1 versus 2.**
- **Whether `tool_use` should be emitted as readable text rather than `json.dumps` output.**
  Carried forward from Phase 1; an emit-format decision, not a defect.
- **Which metric decides whether spectral beats TF-IDF** — see the split verdict below. This is
  a design judgement about what the project is for, not something another sweep can settle.

## Does spectral beat TF-IDF? A split verdict

Phase 1 closed on "no spectral method has yet been shown to beat budget-shaped TF-IDF", and
settling it was the stated reason Phase 2 exists. The full grid was re-run on 2026-09-09 after
the extractor fix, so these are post-fix numbers; the pre-fix curves that appeared to answer the
question were reading inflated recall and have been discarded.

**The two distortion metrics disagree, consistently and in opposite directions.** Deltas below
are spectral minus TF-IDF, interpolated onto matched achieved compression.

| Method | atom recall, all-history | atom recall, final-state | semantic coverage |
| --- | --- | --- | --- |
| `leverage` | +0.024 to +0.077 | +0.021 to +0.071 | −0.035 to −0.145 |
| `ridge` | +0.046 to +0.109 | +0.035 to +0.103 | −0.034 to −0.148 |
| `cur_residual` | +0.037 to +0.101 | +0.024 to +0.093 | −0.028 to −0.072 |
| `supersede+leverage` | +0.010 to +0.087 | +0.019 to +0.113 | −0.019 to −0.142 |
| `supersede+band+leverage` | +0.034 to +0.110 | +0.043 to +0.111 | −0.018 to −0.127 |
| `topics` | +0.012 to −0.046 | +0.005 to −0.067 | −0.015 to −0.136 |

Spectral wins on atom recall at every rate measured. It loses on semantic coverage at every rate
measured. `topics` loses on both, which confirms the Phase 1 decision to cut it as a selector.

**This does not resolve itself by measuring harder.** The two metrics were chosen deliberately
to disagree — see [DESIGN.md](DESIGN.md#distortion-metrics). Atom recall exists to catch "fluent
but wrong": the compressed context that reads beautifully and dropped the one path name that
mattered. Semantic coverage is the surface metric, and Blau & Michaeli's whole argument is that
a fluent decoder optimizes exactly that at fidelity's expense. So the result can be read two
ways, and picking between them is a decision about what the tool is for:

- *Atom recall is the one that counts.* It is the metric the design says catches the failure
  that actually hurts, and spectral wins it outright. Then Phase 1's line is answered: spectral
  earns its place.
- *Losing coverage that consistently is a real cost.* Spectral may be concentrating on
  fact-dense segments and thinning the connective material that makes the retained facts
  usable. Then the honest reading is that spectral trades readability for identifiers, and
  whether that is a good trade depends on what consumes the output.

**Open. Not decided here, and deliberately not decided by whoever ran the sweep.** The next
useful evidence is downstream task evaluation — can a model actually continue the session from
each output — which is the one measurement that prices both metrics in a single currency. That
work is deferred by decision, so until it happens this stays a documented split rather than a
verdict.

## Verified nulls — do not re-walk

- **`drop_top`** (Mu & Viswanath, All-but-the-Top) is a null on this corpus. It changed 2,431 of
  3,240 selections while every measured cell stayed within ±0.005. **The null survived the
  atom-extractor fix**: re-swept post-fix on 2026-09-09, the largest cell across all four
  methods and all three `drop_top` values is +0.004 and the smallest is −0.007. Kept in the
  codebase only as a variant for joint/interaction testing, not as a win.
- **Generic duplicate-line collapsing.** Mean intra-block line repetition is 0–7.2% across every
  tool-output shape measured. There is nothing there to collapse.
- **Basename placeholders** — see above; half the corpus needs disambiguation and it nets half
  the tokens.
- **Path-shape rules short of full decoding.** A drive/UNC-only rule kept 0 of 11,710 backslash
  matches; an extension-or-drive rule kept 1,558 and retained junk such as `n\nconsole.log`; a
  fourth variant measured identical to the second and so added nothing.
- **`topics.py` as a selector.** Measured worse than random in Phase 1 and cut.

A classifier note worth keeping: the diff-detection heuristic over-fired on PowerShell directory
listings, where every entry begins with a dash. Requiring both added *and* removed lines dropped
it from 40 blocks / 15,164 tokens to 10 blocks / 5,619 — 30 of the original 40 were listings.

## Queue

1. ~~Checkpointing so variants accumulate instead of invalidating the whole grid.~~ Done.
2. ~~`drop_top` swept alone.~~ Done; null, see above.
3. Path substitution arms B/C/D wired into `experiments/exp_d_curves.py` and swept.
4. Format-aware shrinkers for prose and tool output. A `thinking` shrinker cannot be evaluated —
   the corpus contains exactly one `thinking` segment.
5. Numbered-file routing fix, measured against path substitution before stacking.
6. Untangle code-trimming depth from compression level (`_band` currently ties quality to the
   keep ratio).
7. Three-way per-kind scoring comparison: separate space, shared normalized, separate budget.
8. Pipeline-level config object, once there are enough real knobs to justify one.
9. Joint and interaction runs, hypothesis-driven rather than exhaustive.

Resumption and downstream task evaluation are deferred by decision, not forgotten.

## Corpus rule, restated

The five local sessions are the author's own work logs. They are gitignored, never committed,
and **never quoted** — not in this document, not in a results table, not in a commit message.
They contribute counts only, under the labels `local_5k` … `local_200k`. Sample lines printed
during development come from public transcripts exclusively.

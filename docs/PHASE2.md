# Phase 2 — plan, decisions, and what is still open

Status as of 2026-09-10: in progress. Phase 1's results are in [PHASE1.md](PHASE1.md); the
algorithm specification is in [DESIGN.md](DESIGN.md). This document is the working plan — what
Phase 2 is trying to settle, which decisions are already locked and why, and which are not.

Phase 1 ended on an uncomfortable note: the one unambiguous win (supersession) was not
spectral, and no spectral method had been shown to beat budget-shaped TF-IDF at an equal
budget. Phase 2 set out to settle that, and to find compression outside `code`, which is a small
share of a real agentic transcript. The first half turned out to be the wrong question — see
[Spectral versus TF-IDF](#spectral-versus-tf-idf-two-statistics-not-one-verdict).

## The measurement problem that had to be fixed first

Everything in Phase 2 is graded by atom recall, so a defect in the atom extractor is a defect
in every number the project has ever produced. Two mirror-image defects were found and fixed
before any new sweep was run.

A `tool_use` segment's text is `json.dumps` output. That means an edit body's newline arrives
as the two literal characters backslash and `n`, and a genuine Windows path's separator arrives
doubled. The path pattern read the first as a directory separator and invented atoms out of
runs of escapes — **8.7% of the corpus's 32,344 atoms were this junk**, and 7,022 of
the 7,156 junk mentions sat in `tool_use`, which is a protected kind. Protected means almost
always retained, which means those junk atoms were almost always scored as "recovered". Every
Phase 1 and early Phase 2 `atom_recall` was inflated by them.

The same serializer made real paths invisible: the pattern cannot span the empty component
between two backslashes, so `venv\Scripts\pip.exe` inside a tool call matched nothing at all
and was never counted as an atom the compressor could lose.

**Fix: decode the payload before matching** (`segment.scannable`), scanning values only.
Corpus atoms fall 32,344 to 27,189.

> **Counting convention, stated once and used throughout this document.** Every corpus atom
> figure here is a **per-transcript sum**: each transcript's distinct atoms, added across
> transcripts, so an atom appearing in two transcripts is counted twice. The **global distinct
> union** is a different and much smaller number — 21,089 where the sum says 27,189 — and it is
> named explicitly on the one occasion it is used, in
> [What it did to the atom set](#what-it-did-to-the-atom-set). A third convention, **atom
> mentions**, appears only in `docs/results/corpus_stats.txt` and nowhere in this document. Do
> not carry one of these into a share computed against another.
>
> **Every atom figure in this document except the spectral-versus-TF-IDF table predates
> `ATOMS_VERSION` 4** (2026-09-10) and is therefore stale — see
> [the third extractor audit](#the-third-extractor-audit--atoms_version-4). They are left in place
> as the record of what version 2 and version 3 said.

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

> **STALE — every number in this section is superseded twice over, and none of it has been
> re-measured.** The path patterns are built from `segment._FILE_EXTENSIONS`, so they moved with
> `ATOMS_VERSION` 3 (which widened the extension list) and again with `ATOMS_VERSION` 4 (which
> narrowed it and added the dotted-chain rule). The figures below were measured before either.
> The direction of the error is not even consistent: version 3 admits more paths, version 4
> fewer, and no one has run the difference. Treat every count, token total and percentage here as
> an order of magnitude, not a value.
>
> **Regenerating them is not a one-command job, and that is why it was not done.**
> `experiments/survey_paths.py` is the script that produced them, but it cannot reproduce them
> under the current extractor as it stands, for two reasons. First, it carries its **own copies**
> of the v1 path patterns as module-level literals (a deliberate choice, recorded in the file: it
> is meant to be readable standalone, and a divergence from `segment.py` is meant to be visible
> rather than hidden behind an import) — so running it today re-measures the v1 rules, not the
> live ones. It has to be re-pointed at `segment._FILE_EXTENSIONS` first, which is what
> `paths.py` already does. Second, it iterates `exp_d_curves._iter_sources`, which yields the
> author's local sessions before any public transcript, so it reads `data/local/`. Once the
> patterns are shared, the command is:
>
> ```
> uv run python experiments/survey_paths.py
> ```
>
> and it writes `experiments/output/survey/path_mass.txt`. Anything quoted from a run that
> includes local sessions stays a count, under the corpus rule at the end of this document.

A file path is repeated far more often than it is informative. Post-fix, the corpus holds 7,373
distinct paths and **17,760 mentions of an already-mentioned path, costing 136,363 tokens**.
Naming each path once in a table and replacing every mention with a short marker does three
separable things, and they must be measured separately because only the first is certain:

1. the table **guarantees** the path survives, whatever selection drops;
2. the marker is shorter, so the text shrinks — 58,177 tokens net of the table's own 24,906,
   which is **3.87% of the 1,503,397-token corpus**;
3. scoring stops seeing long repetitive path strings, which may rank segments better or worse.

Implemented in `src/loosy_goose/paths.py` and wired into the sweep as the five substitution arms
below.

### Decisions locked

- **One row per file.** The table names each path exactly once, in first-mention order.
- **Numeric marker `[P<n>]`, not the basename.** Only **52.0%** of basenames are unique within
  a transcript, so a basename marker needs disambiguation half the time and still nets less
  (**1.93%** against 3.87%). One option is simply better, so this is not a configuration knob.
  (Both percentages are pre-`ATOMS_VERSION`-3 and stale; the decision does not turn on their
  exact values, since nothing about the extension list moves basename uniqueness far enough to
  make a marker that needs disambiguation half the time the better option.)
- **Substitution runs after supersession.** Supersession matches `file_path` values across tool
  calls; substituting first leaves it comparing markers, and its reported paths become
  meaningless to a reader of the results.
- **The table's guarantee is derived from the rendered table text**, not from the path list, so
  it cannot claim to preserve more than the output actually says.
- **`min_mentions` is a knob**, because it is a genuine tradeoff rather than a tuning constant.
  At 2 the table holds only paths that repeat, which is token-optimal — a row for a
  single-mention path costs more than the mention it replaces. At 1 it holds everything, which
  is recall-optimal for exactly the opposite reason: a path mentioned once is the one selection
  is most likely to drop. Default 2, confirmed by the arms on 2026-09-16 — see
  [What the arms said](#what-the-arms-said).
- **The table is charged into the reported compression ratio, not deducted from the selection
  budget.** Every arm hands selection the identical shared budget, and an arm that emits a table
  simply lands further right on the achieved-rate axis, where every cross-method table in the
  runner interpolates. The alternative — deducting the table's tokens from the budget — makes
  "matched budget" exact to the token, but it also hands arms C and D a smaller candidate pool
  at precisely the aggressive rates the project targets, which confounds the mechanism being
  measured with the size of the deduction. The consequence has to be said out loud, because the
  runner's own table caption said the opposite of it until 2026-09-10: **arm results are not
  budget-neutral.** Read every cell as "at the same achieved compression", never as "at the same
  budget" — the matched-rate interpolation is where the table's cost is charged back.
- **The arms sweep five methods, not all eleven**: `tfidf`, `leverage`, `ridge`,
  `supersede+leverage`, `supersede+band+leverage` — 300 records across five arms, against 660 for
  full coverage. `random` and `recency` cannot be helped by better scoring, and `topics` was cut
  in Phase 1, so the extra 360 records would price arms for methods that are not shipping
  candidates. The arms also run at `protect=none` only: `protect=code_only` prices the old binary
  protect flag against `protect=none`, which is a question about the baseline, not about
  substitution.
- **`cur_residual` is deliberately out of the arm method list**, and this is a decision rather
  than an oversight (recorded 2026-09-10, having been incidental until then). It is a spectral
  scorer like `leverage` and `ridge` and it is swept alongside them everywhere else, including
  the `drop_top` variants. What the arms ask is whether naming each file once helps selection,
  and `ridge` and `leverage` already answer that for the spectral family; a third scorer from the
  same family costs 60 more records to re-answer it. Rejected: including it for symmetry with the
  `drop_top` variants, which is a reason about the shape of the grid rather than about the
  question. If the arms move `ridge` and `leverage` in *opposite* directions, that symmetry
  becomes worth paying for and this should be revisited.
- **`protect=code_only` is swept on the base variant alone.** It exists to price the old binary
  protect flag against `protect=none`, which is a question about the baseline, and no variant
  changes the answer. Guarding only the path arms left 36 `drop_top` × `code_only` records that
  every table, curve and plot in the runner filtered straight back out again.
- **Adding the arms did not invalidate the baseline.** The new record fields (`*_earned`,
  `guaranteed_share`, `table_tokens`) are backfilled exactly rather than approximately: a record
  written before any table existed guaranteed nothing, so its earned recall *is* its recall and
  its guaranteed share *is* zero. That is why `SCHEMA_VERSION` does not move and the 18
  checkpointed transcripts kept all 336 of their existing records. (They were discarded shortly
  afterwards regardless, by the `ATOMS_VERSION` 3 bump below — but by the digest, which is the
  mechanism that is supposed to discard them, not by an unrelated schema change.)

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

### The path arms

Run at matched budgets, on the same curve as every other method. **Arm vocabulary, fixed here and
used the same way everywhere else in this document, in `CHANGELOG.md` and in the runner's own
summary:** there are four *lettered* arms, A through D, which name what an arm does to the text;
and there are five *swept* arms, which are the variants the grid actually runs — because C and D
are each swept at both `min_mentions` settings and A is the base variant rather than a variant of
its own. "Four arms" means the four letters. "Five path arms" means the five variant rows below
the base.

| Arm | Variant key | What it does | What it isolates |
| --- | --- | --- | --- |
| A | `base` | baseline, no substitution | control — this is the base variant, not a path variant |
| B | `paths=score_only` | substitute for scoring only; emit the original text, no table | effect 3 alone — does scoring improve when paths stop dominating the embedding? |
| C | `paths=table_only` | emit the table, substitute nothing | effect 1 alone, at full cost — pure overhead, not shippable |
| C | `paths=table_only+mm=1` | the same, with the table built from every path rather than only the repeated ones | what an aggressive table costs **on its own**, with no substitution to pay for it |
| D | `paths=full` | full substitution: table plus markers | all three effects together |
| D | `paths=full+mm=1` | the same, with the table built from every path | `min_mentions` measured rather than picked |

C is deliberately not a shippable configuration. It exists to separate "recall rose because
facts were guaranteed" from "recall rose because selection got better" — without it, arm D's
result is uninterpretable.

`paths=table_only+mm=1` was added on 2026-09-10 for the same reason C exists at all. Before it,
the cost of the aggressive table could only be read jointly with substitution, in arm D, where
the table's extra rows and the markers' savings pull in opposite directions and neither can be
attributed. With it, the two `table_only` rows differ in exactly one thing — how many paths the
table holds — so the extra rows are priced with nothing else moving. Rejected: inferring that
cost by subtracting the two `full` rows, which is the same arithmetic performed on a difference of
differences and would have carried both arms' noise into the answer for 60 fewer records.

**The whole grid is 600 jobs across 9 variants**: 156 on the base variant (11 methods × 12 rates,
plus 2 methods × 12 at `protect=code_only`), 144 across the three `drop_top` variants, and 300
across the five path arms.

### What the arms said

Measured 2026-09-15 under `ATOMS_VERSION` 4 (`rebaseline_v4.log`, blocks "path substitution arms
vs the base configuration" and "path table cost and guarantee, per min_mentions setting"); read
into this document 2026-09-16. Every cell is the arm minus the same method's base curve at
matched achieved compression, so the table's own tokens are already charged. Rows below are
`supersede+leverage`, the shipping default; the other four methods swept have the same shape.

| Arm | final-state recall, 2x / 3x / 4x | earned only (atoms the table does not guarantee) | semantic coverage |
| --- | --- | --- | --- |
| B `score_only` | −0.005 / +0.005 / +0.008 | same | −0.000 / +0.008 / +0.009 |
| C `table_only` | −0.002 / +0.022 / +0.031 | −0.029 / −0.034 / −0.042 | −0.011 / −0.014 / −0.020 |
| D `full` | +0.002 / +0.031 / +0.040 | −0.023 / −0.022 / −0.031 | −0.050 / −0.040 / −0.043 |
| D `full+mm=1` | +0.023 / +0.089 / +0.104 | −0.042 / −0.039 / −0.070 | −0.082 / −0.076 / −0.087 |

Three findings.

- **Effect 3 is null.** Arm B moves nothing past ±0.011 on any method at any rate. Hiding paths
  from the embedding does not rank segments better.
- **Every recall gain is guarantee, not selection.** Earned recall is negative in every table
  arm, every method, every rate. This is the question arm C exists to answer: the table lifts
  what the reader sees by guaranteeing facts, while its tokens push selection to a tighter budget
  at the same achieved rate. Markers claw back about a point of that (D's earned column beats
  C's) and no more.
- **Coverage pays, and the markers pay more of it than the table.** D costs 4–5 coverage points
  at every rate, including 2x where it gains no recall; C alone costs 1–2. Text full of `[P<n>]`
  embeds worse than text with paths in it.

The `min_mentions` table prices the knob directly. At 2 the table is 0.2–4.8% of the input
(median 2.3%); at 1 it is 2.5–48.7% (median 8.3%), and on `swe-gym_row358` the table alone is
half the transcript. That cost is a share of the *input*, so its share of the *output* grows with
the rate: a 2.3% table is about 9% of a 4x output and about 23% of a 10x one.

## Compression outside code

`code` is not where the tokens are. A survey of `tool_result` across the corpus
(`experiments/survey_tool_output.py`) found **2,273 blocks totalling 679,784 tokens**, and the
mass is concentrated: blocks of 500 tokens or more are **11.4% of blocks but 68.2% of tokens**.

The largest single shape is numbered file reads — the `123→` line prefixes a file-read tool
emits: **222 blocks, 219,759 tokens, 32.3% of all tool-output tokens**. Almost none of it reaches
AST banding.

**Corrected 2026-09-10, in both the number and the mechanism.** This section previously said
"99.3% never reach AST banding, because `looks_like_code_dump` tests lines that still carry their
line-number prefixes". Re-measured over the public transcripts alone (8 trace-commons sessions
and 5 SWE-Gym rows, 1,305 `tool_result` blocks), the figure is **205 of 212 numbered-read blocks,
96.7%** — and the prefix is not what decides it:

- `looks_like_code_dump` has two clauses, and a block passes if half its lines match *either*.
  The first is a code-line regex, which a prefixed line does fail. The second is a plain
  indentation test, `startswith(("\t", "    "))`, and a right-aligned line number pads a
  numbered read's lines out to four spaces or more all by itself. **Every one of the 7 blocks
  that does reach the code path today passes on the indentation clause; zero pass on the code-line
  regex.** So the prefix does not block routing — it just fails to help, and what actually decides
  is whether that particular tool's padding happens to be wide enough.
- `code._strip_line_numbers` is **already live**, inside `quantize_code`: a block that does reach
  the shrinker has its prefixes stripped before banding and restored afterwards. What is held is
  only stripping them *before the classifier*, in `segment.py`, which is what would change how a
  block is routed.

The 7 blocks that route through today are 1,510 tokens. The 222-block and 219,759-token totals
above are whole-corpus figures from the original survey and are left as they stand; the 212 and
the 96.7% are public-corpus only, so the two are not two versions of one number.

**Caveat, recorded deliberately:** reaching the code path is not the same as being compressed
well there. Banding alone buys only 1.33x. This is a routing gap, not free saving, and it must
not be reported as the latter.

The remaining `other` bucket: 729 blocks / 109,086 tokens under 500 tokens each (too small to
matter individually), 64 blocks / 82,249 tokens code-shaped, 48 blocks / 62,789 tokens
prose-shaped, 4 blocks / 12,404 tokens with average line length over 200 characters.

**The numbered-file routing fix is written but deliberately held** (decision reaffirmed
2026-09-10). "The fix" means stripping the prefixes before `segment.looks_like_code_dump`, not
before banding — banding already strips them. It is held so that this grid changes one thing at a
time: substitution is being measured now, and a routing change stacked on top of it would make
neither attributable. Rejected: stripping now and paying the cache invalidation once, on the
argument that it is coming anyway. What that argument misses is the size of the bill. Stripping
before the classifier moves a block's segment *kind* from `tool_result` to `code` and its *text*
along with it, so it moves `segments_digest` — which is the input fingerprint, the one thing no
version stamp is scoped around. Every checkpoint in the grid is discarded and all 600 jobs re-run,
6–12 hours, and this is not the moment to spend that.

## Still open — do not treat as settled

Four items that used to live here were decided on 2026-09-10 and have moved to
[Decided](#decided-2026-09-10) below. What remains genuinely open:

- **How to drive the whole per-kind quality table from a single `--quality` knob.**
- **Segment-level versus span-level trimming for prose.**
- **Whether the background prior should be generic, personalized, or both layered.**

### Parked with an argued exception — decided not to decide yet

These are known, they have a case on both sides, and none of them is being fixed before the grid
runs. Recorded here so they are not rediscovered as surprises.

- **Argument key names still enter the atom set from rendered output.** `render_call` keeps key
  names for readability, and `old_string`, `new_string` and the rest match the snake-case
  identifier pattern, so the *rendered* segment's atoms contain them while the *original*
  segment's do not — `scannable` strips keys there. This is the identical objection that removed
  the tool name (see below), and it is not being applied here for one reason the tool name could
  not offer: a key is a substring of the payload being rendered, so it is verbatim in a way a
  metadata field is not, and dropping it makes a Bash call's `command` and `description`
  indistinguishable. The cost is the same in kind though smaller in size — tokens spent on atoms
  recall can never credit. Deferred, not settled. (The target path key is not affected: the
  target renders as a bare line, without its key.)
- **A sentence-final path loses the disagreement.** In `edit src/alpha.py.` the slash pattern
  swallows the trailing full stop, so `find_paths` returns `src/alpha.py.` while `extract_atoms`
  strips it and returns `src/alpha.py`. The table is keyed on the first, the metric on the second,
  and such a mention therefore never substitutes. Pre-existing, not introduced by any of today's
  work. The fix has to move `segment.py` and `paths.py` in one change — they share the pattern
  precisely so the table cannot claim atoms the metric will not count — which makes it an
  `ATOMS_VERSION` bump and a full re-baseline for a small population.
- **`google_creds.json.enc` yields no filename atom.** It yields `google_creds` from the
  identifier pattern and nothing more, because the dotted-chain rule cannot tell it from
  `process.env.NODE_ENV`. It is structurally the same case, and readmitting it means readmitting
  chains. Accepted as the price of the rule; one mention in the corpus.
- **The slash-path pattern is quadratic on pathological input.** 12,000 repetitions of `a.-`
  takes about 1.3 seconds. Contained in practice, because segment text is bounded by `max_chars`
  and no real transcript looks like that — but it is contained by an unrelated bound rather than
  by the pattern, and an earlier review's claim that every atom pattern runs in under 0.003s does
  not hold for that input.

## Decided 2026-09-10

- **A defined pointer is admissible in retained text, so arm D may ship.** Losslessness was
  never the goal — whole segments are discarded on purpose. The verbatim property was only ever
  about the text that *survives*, and a marker whose definition ships in the same output has
  reworded nothing and invented nothing. Rejected: requiring every retained span to be quotable
  as-is, which forfeits path substitution's token saving and its recall guarantee together, and
  rules out every future side-table mechanism on the same grounds. See
  [DESIGN.md](DESIGN.md#pipeline).
- **`min_mentions` is measured, not picked.** Arms C and D are each swept at both 1 and 2 (120
  extra records between them), and the per-transcript table below prices what the extra rows cost
  against what they guarantee. Rejected: locking either value, since the doc already recorded this
  as a genuine tradeoff rather than a tuning constant — that is an argument for measuring it.
  Sweeping C as well as D was added later the same day; see
  [The path arms](#the-path-arms).
- **The two distortion metrics are separate statistics and are never collapsed.** See below.
- **`tool_use` tolerates the most loss of any channel.** A tool call is repetitive scaffolding;
  what matters is that a call happened against a target, not the serialized arguments. This
  reframes a question Phase 1 recorded as "emit `tool_use` as readable text rather than
  `json.dumps`": the answer is not a prettier format but a far more aggressive shrinker, and
  since shrinking is a downstream transformation like AST banding it is deliberately outside
  `segments_digest` and costs no re-baseline. The field bander that was too conservative has since
  been replaced — see [The tool_use channel](#the-tool_use-channel).
- **The tool's name is not emitted in a rendered call.** `render_call` knows the name and drops
  it. Two reasons, and the second is the decisive one. It reaches the renderer from the transcript
  record's metadata rather than from the segment's own text, so it is neither a verbatim span of
  the original nor a mechanically-defined pointer — the only two things the emitted text is
  allowed to contain, per the pointer decision above. And it is strictly bad on the metric rather
  than a trade: `MultiEdit`, `TodoWrite` and `NotebookEdit` all match the identifier pattern, so
  they enter the rendered segment's atoms, while recall intersects against the atoms of the
  *original* text, which never held them. The tokens cannot buy recall back and they count against
  the ratio. Rejected: keeping it and redefining it as a mechanically-defined pointer, which fails
  because a pointer's definition has to ship in the same output and there is nothing here to
  point at. Also rejected: keeping it silently, on readability. A side effect worth recording —
  the size guard in `quantize_tool_use` could tighten from "not larger" to "strictly smaller" once
  the name was gone, because the name was exactly what a short single-argument call spent the
  JSON-scaffolding saving on.
- **The `tool_use` depth knob follows the global keep ratio.** The code has always done this;
  `DESIGN.md` claimed the opposite and has been corrected. `quantize_segment` takes a
  `tool_quality` override that defaults to `quality`, and nothing sets it. Rejected: changing the
  code to match the doc — holding one channel's depth flat is a second knob turning inside a grid
  whose discipline is one change at a time, and the argument for flat depth (that the case for
  shrinking this channel does not weaken as the rate relaxes) is an argument, not a measurement.
  Sweeping the depth as its own variant stays on the queue, and is where that argument gets
  tested.
- **Short ambiguous file extensions require a path context.** `c`, `h`, `a`, `o`, `cc`, `so` and
  `mk` came out of the bare-filename allowlist because they were matching attribute access —
  `self.c`, `obj.h`, `x.o`, `node.so` — and `env` came out because 43 of its 47 corpus mentions
  are `process.env` or the like. They still extract inside a path, since the slash and drive
  patterns never consult the extension list. Accepted cost, stated plainly: a bare `main.c` in
  prose is no longer an atom, while `src/main.c` still is. Rejected: dropping the seven
  extensions outright, which loses the path case too and the path case is the one that names a
  real file. Also rejected: keeping them and relying on the dotted-chain rule alone, which fixes
  `process.env.NODE_ENV` but does nothing for a bare `self.c` at the end of a sentence.
- **The supersession pass and the shrinker can disagree about which tool a call came from, and
  that is not being fixed before this run.** There are two independent sources for a tool's name.
  `supersede.tool_names_from_transcript` rebuilds them positionally — it walks each turn's
  `tool_use` blocks in order and pairs them with that turn's `tool_use` segments by a cursor —
  while `code.quantize_tool_use` reads `Segment.tool_name`, which segmentation stored directly.
  Anything that makes the positional walk drift (a dropped block, a segment the loader counted
  differently) makes the two disagree, and nothing detects it. Rejected: unifying them on
  `Segment.tool_name` now, which is the right fix and is a small one. It is refused on timing:
  supersession's measured 25.9% lossless win is the project's clearest result, it was measured
  under the positional rule, and changing the rule moves which calls are found superseded. Revisit
  immediately after the grid, when a change to that number can be attributed instead of guessed
  at.

## Decided 2026-09-16

- **The shipped `compress` command runs `supersede+leverage`; banding is opt-in.** On final-state
  atom recall (`exp_e_v4.out`, E1a) the band stack is 1.8 points ahead at 4x (0.578 against
  0.560), 1.3 ahead at 3x (0.684 against 0.671) and 0.8 *behind* at 2x (0.843 against 0.851),
  and it costs 15.40 s per 100k tokens against 0.20 s (E4, warm embedding cache) — 149x
  budget-shaped TF-IDF and 77x `supersede+leverage`. Cheap enough to run on every compaction is
  the property that matters for a tool that sits inside a context window; the extra recall at
  the aggressive end stays reachable through a flag or the aggressive preset (queue item 14), on
  the reasoning that whoever asks for the hardest compression will take the wait. Rejected:
  shipping the band stack as the default, which makes every run pay for a gain that is marginal
  at 4x and negative at 2x. Also rejected: leaving banding out of the product, which discards
  the best-measured selector at 3x and beyond for no saving once it is opt-in. No product
  pipeline exists yet — `cli.py` is a stub — so this fixes what lands, not what runs today.
- **What the compression knob means is deferred until the core lands.** `--quality` is currently
  a 0–1 "retained-meaning target" (`cli.py`), while the only knob the code has is `keep_ratio`
  (`budget.token_budget`), and how one drives the other is the open question under
  [Still open](#still-open--do-not-treat-as-settled). Two options were on the table — make the
  flag the keep ratio outright, bounded to the measured span 0.1–0.8, or keep it abstract and
  write the mapping now — and neither was chosen: the mapping would be invented rather than
  measured, and renaming the flag before there is a pipeline behind it decides a user-facing
  surface for a stub. Revisit when `compress` runs something. Item 14 carries the presets.
- **`min_mentions` defaults to 2 and stays a flag.** At 1 the table's cost is unbounded — median
  8.3% of the input, 48.7% on one public transcript — and the coverage cost of arm D doubles. At
  2 it is capped under 5%. Rejected: 1 as the default. Also rejected: 1 as the aggressive preset
  (queue item 14), since the table's share of the output is worst at exactly the rates that
  preset targets. 1 stays reachable for recall at any price.
- **Path substitution is off by default and opt-in.** For the default selector, arm D at
  `min_mentions` 2 is +0.040 final-state recall at 4x, +0.031 at 3x and +0.002 at 2x, against
  −0.040 to −0.050 coverage at every rate. That is a recall-versus-coverage split of the kind
  this document refuses to collapse into a winner, and with no downstream task evaluation to
  price both in one currency, the default is the configuration that does not spend coverage for
  nothing at 2x. Rejected: on by default, on the argument that recall is the metric the project
  exists for and the table is the only mechanism that guarantees anything — a values argument,
  kept on record for when task evaluation exists. Also rejected: on at 3x and harder only, a
  rate-dependent switch of the kind queue item 7 is trying to remove.

## Spectral versus TF-IDF: two statistics, not one verdict

Phase 1 closed on "no spectral method has yet been shown to beat budget-shaped TF-IDF", and
settling it was the stated reason Phase 2 exists. **The question was wrong, and that is the
finding.** Atom recall and semantic coverage measure different things and disagree consistently;
collapsing them into a single winner discards exactly the information the two-metric design was
built to expose. Baseball keeps FRV and OAA side by side rather than averaging them into a
number that means less than either. Both are reported here, permanently, and no tie-break rule
is applied (decided 2026-09-10).

That also removes the temptation this section was previously exposed to: once numbers exist,
whichever metric supports the preferred conclusion becomes the tempting one to crown.

The grid was re-baselined on 2026-09-15 under `ATOMS_VERSION` 4 (18 transcripts × 600 jobs, run
log at `experiments/output/rebaseline_v4.log`), so the numbers below are current. The version-2
table this replaces was measured on 2026-09-09 after the first extractor fix; the curves before
that fix were reading inflated recall and were discarded. The version-2 summary formerly at
`docs/results/exp_d_summary.txt` was replaced by the version-4 one on 2026-09-16 — see
[that directory's README](results/README.md).

**The two distortion metrics disagree, consistently and in opposite directions.** Deltas below
are spectral minus TF-IDF, interpolated onto matched achieved compression, from the "spectral
methods vs tfidf at matched ACHIEVED rate" block of `rebaseline_v4.log`. Each range is the
smallest and largest delta over the ten achieved rates from 0.8 down to 0.1 (1.25x to 10x). The
0.05 rate is measured but excluded: for most methods only 8 of 18 transcripts reach it. The
supersede stacks have the opposite gap — at rates of 0.6 and milder, supersession alone already
compresses past the target on some transcripts, so those cells rest on 6 to 14 transcripts. They
are kept; dropping them would leave those methods with no mild-rate cells at all.

| Method | atom recall, all-history | atom recall, final-state | semantic coverage |
| --- | --- | --- | --- |
| `leverage` | +0.037 to +0.084 | +0.024 to +0.076 | −0.035 to −0.108 |
| `ridge` | +0.048 to +0.113 | +0.035 to +0.108 | −0.034 to −0.113 |
| `cur_residual` | +0.038 to +0.109 | +0.025 to +0.101 | −0.028 to −0.042 |
| `supersede+leverage` | +0.053 to +0.093 | +0.066 to +0.113 | −0.019 to −0.096 |
| `supersede+band+leverage` | +0.065 to +0.103 | +0.076 to +0.123 | −0.029 to −0.083 |
| `topics` | +0.003 to −0.036 | −0.007 to −0.054 | −0.015 to −0.136 |

Spectral wins on atom recall at every rate in the span. It loses on semantic coverage at every
rate in the span. `topics` is never better than +0.003 on atom recall and loses on coverage
everywhere, which confirms the Phase 1 decision to cut it as a selector.

**This does not resolve itself by measuring harder, and it is not supposed to.** The two metrics
were chosen deliberately to disagree — see [DESIGN.md](DESIGN.md#distortion-metrics). Atom
recall exists to catch "fluent but wrong": the compressed context that reads beautifully and
dropped the one path name that mattered. Semantic coverage is the surface metric, and Blau &
Michaeli's whole argument is that a fluent decoder optimizes exactly that at fidelity's expense.

So the honest description of each selector is two-dimensional, and both readings are true at
once rather than competing:

- **Spectral is the fidelity-favouring selector.** It keeps more of the concrete facts —
  identifiers, paths, numbers — at every rate measured.
- **Budget-shaped TF-IDF is the coverage-favouring selector.** It keeps text that better
  represents the whole conversation's meaning-space, at every rate measured. Spectral may be
  concentrating on fact-dense segments and thinning the connective material that makes the
  retained facts usable.

`topics` loses on both axes (its one positive cell is +0.003), which is not a split and is why it
stays cut.

Downstream task evaluation — can a model actually continue the session from each output — would
price both in a single currency, and remains the most valuable thing on the queue. It is not
needed to *close* this question, because the question is closed: report both.

## The tool_use channel

Where its 550,969 tokens actually sat, before any of this:

| | tokens | share | status then |
| --- | --- | --- | --- |
| body values | 255,118 | 46.3% | banded as code |
| other values | 187,816 | 34.1% | **verbatim, through a bare `else`** |
| serialization scaffolding | 67,313 | 12.2% | untouched |
| stale values | 23,864 | 4.3% | dropped |
| path values | 16,858 | 3.1% | kept on purpose |

The hole was `_BODY_KEYS`: it named four keys and everything else fell through. `command` alone
was **135,136 tokens, 24.5% of the channel**. So the fix was not a bespoke shrinker but routing
every value by **shape rather than key name**, through the shrinkers that already existed.

**Cut points are measured, never chosen.** Both budgets are percentile tables over a population
that is a complete unit by construction, and `quality` indexes into them — it names how much we
still keep, as a percentile of real examples:

- **Shell**, `SHELL_COMMAND_TOKENS`: commands that needed no pipeline (n=136), deciles
  6/11/15/17/18/20/21/26/37/62/161. Stages are the split unit because shell marks its own
  boundaries, so no length threshold is invented. Quality 0.5 gives 20 tokens — the median real
  command.
- **Prose**, `PROSE_UNIT_TOKENS`: the corpus's own prose segments (n=4,374), deciles
  1/8/14/20/29/40/52/70/100/145/322.

The first prose reference tried was *tool arguments that are a single sentence*, by analogy with
the shell population. Measurement rejected it: n=1,609 with a median of **6 tokens**, because
that population is dominated by tiny `description` labels and carries one 2,157-token outlier. A
budget drawn from it would have cut a real prompt to nothing. Prose mass is concentrated the way
`tool_result` mass is — values of 20+ tokens are 13.5% of blocks but **87.0% of tokens** — so the
shrinker only ever has to act on large values, and short labels survive untouched.

**Rejected on method, not on results:** keeping the sentences that contain atoms. It scores well
on atom recall *by construction* — the shrinker would be optimising the metric that grades it,
the same self-grading the select/score model split exists to prevent.

Calls are emitted as a call line rather than a JSON object. **The tool's name is not part of that
line** (decided 2026-09-10 — see [Decided](#decided-2026-09-10) for the reasoning and the
rejected alternatives). The target path is, on a line of its own without its key. Key names are
kept for the remaining arguments, even though they are the tool's schema rather than session
facts, because dropping them makes a Bash call's `command` and `description` indistinguishable;
they cost tokens but cannot *improve* recall, since it intersects against atoms from the ORIGINAL
segments where `scannable` strips keys already. They can still be extracted from the rendered
text, which is the same objection that removed the tool name and is parked rather than answered —
see [Parked](#parked-with-an-argued-exception--decided-not-to-decide-yet). A side benefit of the
rendering: it carries single backslashes, so a Windows path in a shrunk call now matches the path
table in the same spelling the table uses.

Two guards sit on the output, because a shrinker that fails at its one job should fail visibly.
`quantize_tool_use` falls back to the raw payload when the rendering is empty (every key can be
dropped or empty, and `{}` renders to nothing — an empty segment still reaches `embed_texts` in
the coverage metric and contributes a degenerate vector) or when it is **not strictly smaller**
than its input (a payload of nested containers pays back more JSON scaffolding in `render_call`
than `_shrink_arg` saved on its string leaves). The second guard was "not larger" until the tool
name came out: the name cost a short single-argument call almost exactly what dropping the JSON
scaffolding saved, so ties were real and worth keeping. With the name gone they are not, and the
guard now matches the strictly-smaller rule the rest of the module uses.

Channel compression, whole corpus:

| depth | old key-allowlist | shape routing | + prose | + call rendering |
| --- | --- | --- | --- | --- |
| 0.33 | 1.25x | 1.40x | 1.76x | **1.92x** |
| 0.0 | 1.34x | 1.56x | 2.03x | 2.21x |

**Quote 1.92x, not 2.21x.** 2.21x is the depth-0.0 row — every argument shrunk to signatures and
first stages. The depth knob follows the global keep ratio (see
[Decided](#decided-2026-09-10)), so a run at the rates this project actually targets puts this
channel at roughly depth 0.33, where the same pipeline realizes **1.92x against the old rule's
1.25x**. The headline was previously stated as 2.21x here and in `CHANGELOG.md`, which is the
best row in the table rather than the row a real run lands on.

> **Stale as of `TRANSFORMS_VERSION` 2, and not re-measured.** This table was measured under
> version 1 of the shrinker. Version 2 changed two things that move these token counts: the call
> rendering no longer emits the tool name (strictly fewer tokens, so the fourth column understates
> the current figure), and `shrink_prose` now slices the original text rather than rejoining
> normalised chunks (whitespace differs, so the third and fourth columns move by an unknown small
> amount in an unknown direction). There is **no committed script that regenerates this table** —
> it was an ad-hoc measurement, unlike the surveys — so re-measuring means writing one first. That
> is worth doing at the same time as the depth sweep below, which needs the same instrumentation.

Not yet swept as a variant, so these are channel-level token counts, not a rate-distortion
result. The depth knob has to earn its place on the curve like everything else.

### A silent-staleness trap, found by walking into it

The shrinker rewrite above was made while a grid was running. The running process had loaded the
old module at startup, so its results describe code that no longer exists — and **nothing would
have rejected them**. `segments_digest` fingerprints the segmenter's output and stops there on
purpose, because that exclusion is what lets variants accumulate in one checkpoint instead of
evicting each other. But "downstream" covers the shrinker's *code*, not only its settings.

It was caught by comparing the process start time against the source file's mtime, which is not
a control. `code.TRANSFORMS_VERSION` is now part of record identity for the methods that shrink,
so rewriting the shrinker recomputes exactly those records.

**Scoped, not global.** Only `band+leverage` and `supersede+band+leverage` run text through
`quantize_segment`. A global stamp would need no maintenance but would throw away the whole grid
— 6–12 hours — for a one-line change to the code bander. Records predating the version default
to stamp 0, which is what makes the scoping work: a non-shrinking method's stamp is 0 too, so
its records stay valid, while a shrinking method's older record is recomputed.

The residual risk is that the method list is hand-written, so a future banding method added
without updating it inherits the bug. That is not trusted to discipline: a test runs every
method with `quantize_segment` watched and asserts the observed set equals the declared one.

**The same hole existed in a second module.** Path substitution is downstream of segmentation
for exactly the same reason the shrinker is, so a rewrite of `paths.py` — a changed marker
spelling, a different rule for what gets tabulated — would have left stored arm records looking
valid while describing behaviour that no longer exists. `paths.PATHS_VERSION` closes it on the
same terms: scoped by `paths_stamp()` to records whose variant actually substitutes, so a
`paths.py` change recomputes the five path arms and leaves the baseline and the `drop_top` variants
alone. Both stamps live in `RecordIdentity` beside the method, protect policy, achieved ratio
and variant key.

The general shape, worth stating once: **`segments_digest` is an input fingerprint, and every
transform downstream of it needs its own version stamp.** Three stamps now, and the third is the
one that shows the rule is about transforms rather than about modules. `BAND_QUALITY_VERSION`
lives in the runner, not in `src/`, because what it versions is the runner's own decision to hand
`quantize_segment` the sweep's keep ratio as its fidelity knob. That mapping is not part of
`code.py`, so changing it would move neither `segments_digest` nor `TRANSFORMS_VERSION`, and
stored band records would have kept loading as valid while describing a depth policy that no
longer existed. It is scoped to the banding methods for the same reason the shrinker stamp is.
Anything else that starts rewriting segment text, wherever it lives, needs a fourth.

### The pre-flight audit

Before committing 6–12 hours of grid time, the whole changed surface was walked deliberately
rather than trusted. It found three more defects, none of which any test would have caught.

**A crash in the last step of the run.** `_record_identity` had grown from a 4-tuple to a
5-tuple while `_curves_by_method_variant` still unpacked four. Every job would have completed
and checkpointed, and then `_write_summary` would have raised on the final call — twelve hours
in, with the summary being the only artefact anyone reads. Fixed at the class level rather than
at the call site: identity is now a `RecordIdentity` `NamedTuple`, read by field name
everywhere, so adding a sixth component cannot silently break a reader. A test asserts the
fields are accessed by name and not by position.

**Variants were being averaged into the baseline.** This is the serious one, because it is a
measurement defect rather than a crash. `exp_e_analysis.build_curves` keys on
`(method, protect)` alone — correct when a checkpoint held one variant, wrong the moment
record-level identity let variants accumulate in the same file. Since then it had been folding
`drop_top` and path-arm records into the baseline curve and reporting the mean without saying
so. Any Experiment E number read after variants landed is contaminated and must be regenerated.
`BASE_VARIANT_KEY` now filters to baseline records only; records predating variants carry no
key and are treated as baseline, which is what they are. The literal is duplicated rather than
imported to keep the analysis module cheap to load, so a test asserts it still equals
`BASE_VARIANT.key()`.

**The summary itself was untested.** `_write_summary` runs exactly once, at the end, and renders
every table anyone reads. It now has a test over the real `plan_jobs()` grid asserting every
section renders — the crash above is precisely the failure mode that reaches a human only after
the compute is spent. The section count is deliberately not stated here: it was written down as
"eight" while the writer rendered ten, and it moved again the same day when the failure block was
added on top. A count that drifts every time a table is added is a claim that will be wrong more
often than right; the test is what pins the set, and it reads it from the code.

### Surviving a 600-job run

A second pass over the runner, on the assumption that something will go wrong six hours in and
the question is only whether anyone finds out.

**A failing job no longer takes the run with it.** The exception is caught per job, recorded with
its traceback in a `failures` list beside the records, and the run continues. A failure is
deliberately *not* stored as a record: a record with zeroes in it would be averaged into its
method's curve and pull it down, which is worse than a hole, because a hole can be counted and a
bad average cannot be spotted. The affected method is simply summarised over fewer rates. Failures
are retried on the next run, since the missing identity is what the checkpoint loader looks for.

**The summary opens with an incomplete-run block, loudly.** It counts two different holes:
failures raised this run, and planned cells absent from the checkpoints for any reason at all —
an aborted earlier run, a `--limit`, a version-stamp bump. Every table below it interpolates over
whatever cells exist, so a silent hole reads as a perfectly normal curve measured on fewer points.
That is exactly the failure this project keeps finding in its own numbers, so the block is first
rather than a footnote.

**Checkpoints are written atomically and every 10 jobs.** A direct write of a ~500 KB JSON file
is not atomic, and a kill mid-write leaves truncated JSON that the loader can only treat as
unreadable — silently discarding a whole transcript's work and re-running it. The payload now goes
to a sibling temp file and is moved onto the real name with `os.replace`. Flushing every 10 jobs
caps what a kill destroys at about 20 seconds of work rather than a whole transcript, and costs
under half a percent of run time.

**Experiment E's cross-checkpoint guard was dead and is now keyed on something that exists.** It
compared a run-level `config` key that `SCHEMA_VERSION` 3 had removed, so it compared `None` with
`None` on every checkpoint and could never fire — a guard against mixing incompatible checkpoints
that had stopped guarding at the moment the schema changed. It now fingerprints the schema
version, the actual ratio grid and the actual protect policies, all read off the records, and it
fails loudly with the disagreement spelled out. The method set is deliberately left out of the
fingerprint: a method that failed on every ratio leaves the grid intact and shows up as a missing
curve, whereas a changed ratio grid silently shifts every interpolation.

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

## The second extractor audit — `ATOMS_VERSION` 3

Writing the arm tests turned up a fifth extractor defect, which prompted a full audit rather
than a one-line fix: every `ATOMS_VERSION` bump costs a complete re-baseline, so finding one
defect at a time is the expensive way to do this. The audit found four, all measured on the
corpus rather than argued from the regexes.

| Defect | Occurrences | What it looked like |
| --- | --- | --- |
| Drive letters dropped | 2,616 | `D:\BentoFolio\vercel.json` recorded as `BentoFolio\vercel.json` — the pattern required a word character before its first separator |
| `CONSTANT_CASE` invisible | 3,952 | the snake pattern accepts `[a-z0-9]` only, so `TRAIN_RATIO`, `API_CHANGES` were never atoms |
| Acronym-led identifiers invisible | 1,318 | both camel patterns require a lowercase run after every capital: `DOMContentLoaded`, `CMakeLists`, `PSIsContainer` |
| File extensions missing | 1,347 | no C/C++/build/binary family: `.h` 263, `.exe` 207, `.cpp` 144, `.o` 144, `.c` 137, `.apk` 115, `.hpp` 78, `.svelte` 69, `.bin` 62, plus `.tar/.gz/.env/.gradle/.properties/.bat/.dll/.so/.zip` |

**The extension list stays an allowlist**, and that is what makes it safe. A general `name.ext`
rule reads attribute access as a filename. Two candidates looked like clear wins by raw count
and are excluded on the evidence of their context: `.log` (205) is overwhelmingly `console.log(`
and `.g` (121) is `e.g.`. Counting alone would have added both.

### The backtick rule, changed on purpose

Separately, and not a defect but a definition: the backtick pattern captured whole inline-code
spans, so `` `prettier --write "**/*.json"` `` was **one atom**. 2,199 captured spans contain
spaces and 49 exceed 80 characters. Recall on those demanded exact reproduction of a long
string, which is a different and much harder question than "did the identifier survive".

A backticked span with no whitespace is still kept whole. One with whitespace is now tokenised:
the first token is the command and is kept on position, later tokens only when they carry a
path, flag, dot, digit or inner capital. The known cost, accepted rather than papered over: a
backticked English phrase contributes its first word and nothing else.

### What it did to the atom set

> **Superseded by `ATOMS_VERSION` 4, below.** These are version 3's figures and version 3 is no
> longer what runs. Left in place as the record of what that bump did, not as the current shape of
> the atom set — which has not been re-measured per kind.

| kind | before | after | |
| --- | --- | --- | --- |
| prose | 6,783 | 6,540 | −3.6% |
| code | 5,056 | 5,289 | +4.6% |
| `tool_use` | 7,085 | 7,462 | +5.3% |
| `tool_result` | 10,575 | 11,263 | +6.5% |

Corpus total **27,189 → 27,668** in **per-transcript sums**, this document's convention
throughout. The **global distinct union** — a different denominator, roughly a fifth smaller —
went 21,089 → 21,444 over the same change. This is the one place both are quoted, and the two
must never be mixed inside one share. Either figure includes 571 drive-qualified atoms that did
not previously exist.

**This is why the fix could not be deferred.** Taken alone the drive-letter defect corrupted no
comparison — the table and the metric shared the pattern, so they agreed on the same truncated
string. Taken together the four defects do not have that property: the denominator moves by
about ten points more for tool output than for prose, and tool output is exactly what
supersession and banding act on. The metric now rewards something measurably different, per
method unevenly, which reaches the spectral-versus-TF-IDF split recorded below.

Every absolute `atom_recall` figure produced before this bump is superseded. `paths.py` moved in
the same commit and must always: it shares `segment._FILE_EXTENSIONS` and carries the identical
path regex, because the table guarantees whatever it names while the metric only credits what
the extractor recognises. Fixing `paths.py` alone was rejected outright — it would let the table
claim atoms the metric never counts, making `guaranteed_share` wrong in the arms' favour.

A classifier note worth keeping: the diff-detection heuristic over-fired on PowerShell directory
listings, where every entry begins with a dash. Requiring both added *and* removed lines dropped
it from 40 blocks / 15,164 tokens to 10 blocks / 5,619 — 30 of the original 40 were listings.

## The third extractor audit — `ATOMS_VERSION` 4

The audit above widened the file-extension allowlist by counting occurrences, and counting alone
was the mistake it had just warned about in the `.log` and `.g` cases. Adding the C/C++, build and
binary families put `c`, `h`, `a`, `o`, `cc`, `so` and `mk` on the list, and those are not rare
extensions that happen to look like attributes — they *are* the spelling of ordinary attribute
access. `self.c`, `obj.h`, `df.a`, `x.o` and `node.so` all became filename atoms. `env` was the
same shape from the other direction: 43 of its 47 corpus mentions are `process.env`, `self.env` or
`interp.env`.

Two structural rules replace version 3's claim that every entry had been confirmed file-like:

- **The seven ambiguous extensions are recognised only inside a path.** They are gone from the
  bare-filename allowlist, but the slash and drive patterns never consult that list, so
  `src/main.c` and `C:\repo\main.c` still extract. `env` is gone outright. The criterion is the
  collision, not the length: `py`, `js`, `ts`, `md`, `sh` and the other two-character entries name
  files far more often than they name attributes and are untouched.
- **A dotted chain is not a filename.** A match followed by `.` plus an identifier character is
  rejected, which is what `process.env.NODE_ENV` and `java.sql.Types` are. This applies to every
  extension on the list, not only the ambiguous ones. A *second* extension is the exception and is
  consumed rather than rejected, so `node.tar.gz` and `device.svelte.ts` come out whole — before
  the fix the pattern at least yielded the truncated `node.tar`, and rejecting outright would have
  yielded nothing at all, which is strictly worse. `kts` is on the list for exactly one reason:
  to make `build.gradle.kts` whole.

Measured over the public transcripts: bare-filename matches fall **8,206 → 5,523**. Of the 2,683
lost, 2,585 are the seven extensions, 47 are `env`, and about 50 are chain rejections. The cost
was weighed and accepted: **a bare `main.c` in prose is no longer an atom**, and it remains one
inside a path.

It landed in two passes — the path requirement first scoped by extension length, then narrowed to
the seven by collision; double extensions exempted from the chain rule afterwards — but nothing
was swept between them, so the whole rule set ships under one stamp. Same reasoning as
`code.TRANSFORMS_VERSION` 2: a version number exists to invalidate stored records, and there are
no records at the intermediate state for it to invalidate.

`paths.py` moved with it and now *imports* `segment._FILE_EXTENSIONS` rather than restating it,
which is the only arrangement in which the two cannot drift. Its `PATHS_VERSION` went to 2 in the
same change, for an unrelated defect described in [Path substitution](#path-substitution)'s
implementation notes: substitution used a blind `str.replace` per table row, so any path that
merely *contained* a tabulated one was mangled — with `main.c` in the table, `main.cpp` was
emitted as `[P0]pp`. It now matches on the module's own path spans and substitutes only where a
whole span equals a table row, which also takes the cost from O(rows × text) to O(text),
independent of table size. Nothing failed loudly while the defect was live, because `expand` still
restored such text; what was wrong was the scored text and its re-extracted atoms.

## Queue

1. ~~Checkpointing so variants accumulate instead of invalidating the whole grid.~~ Done.
2. ~~`drop_top` swept alone.~~ Done; null, see above.
3. ~~Path substitution arms B/C/D wired into `experiments/exp_d_curves.py`.~~ Done, and swept at
   both `min_mentions` settings for C and D — five path arms in all. Measured 2026-09-15 under
   `ATOMS_VERSION` 4 and read in 2026-09-16 — see [What the arms said](#what-the-arms-said).
4. ~~A dedicated `tool_use` shrinker.~~ Built — see [The tool_use channel](#the-tool_use-channel).
   **Still to do: sweep the depth knob as its own variant**, so it is measured on the curve rather
   than asserted from channel-level token counts. That sweep is also where the argument for
   holding the depth flat gets tested rather than asserted, and it needs instrumentation that
   would regenerate the stale channel table at the same time.
5. Format-aware shrinker for the prose *channel*. `shrink_prose` exists and is wired into tool
   arguments; whether whole prose segments should be shrunk as well as selected is a separate
   question, because there the two mechanisms compete for the same budget — shrinking a segment
   makes it cheaper to keep, which changes what selection can afford, so the two cannot be priced
   independently the way the tool channel's can. A `thinking` shrinker cannot be evaluated — the
   corpus contains exactly one `thinking` segment.
6. Numbered-file routing fix, measured against path substitution before stacking. Costs a full
   re-baseline when it lands, since it moves segment kind and text.
7. Untangle code-trimming depth from compression level (`_band` currently ties quality to the
   keep ratio).
8. Three-way per-kind scoring comparison: separate space, shared normalized, separate budget.
9. Pipeline-level config object, once there are enough real knobs to justify one.
10. Joint and interaction runs, hypothesis-driven rather than exhaustive.
11. **Whether Experiment E should ever see variants.** It currently hard-excludes them, which was
    the right fix for a defect that had been averaging them into the baseline — but "never" was
    the safe answer under time pressure, not a considered one. Nothing is lost by waiting:
    per-kind recall and coverage are stored on every record, so any variant analysis is
    re-derivable from the existing checkpoints with zero recompute.
12. Unify the two sources of a tool's name (`supersede`'s positional rebuild against
    `Segment.tool_name`), immediately after the grid — see
    [Decided](#decided-2026-09-10) for why it waits.
13. **Make both surveys regenerable, in one pass.** Two separate figures in this document cannot
    currently be reproduced against the code that is actually running, for two different reasons,
    and the fix is the same shape for both.
    - The `tool_use` channel table was measured ad hoc: no committed script produces it, so it
      cannot be re-derived at all. It is stale under `TRANSFORMS_VERSION` 2, which dropped the
      tool name and changed `shrink_prose`'s whitespace, moving every column.
    - `experiments/survey_paths.py` exists but carries its own hard-coded copies of the v1 path
      patterns, deliberately, so running it today re-measures the old rules rather than the
      current ones. Re-point it at `segment._FILE_EXTENSIONS`, the way `paths.py` already does,
      and the numbers become regenerable by construction rather than by discipline.
    Neither touches `src/`, so this is read-only measurement work that can land any time. Doing it
    alongside item 4's depth sweep is the natural pairing — that sweep needs the same
    instrumentation.
    **Open sub-decision, not yet made:** whether these surveys read `data/local/` (counts only and
    never quoted, as today, which keeps continuity with the existing figures) or restrict to the
    public corpora (so every published number is reproducible by someone without the local
    sessions, at the cost of changing what the numbers mean).
14. **Compression level as a bounded, user-facing control.** Expose the global keep ratio as a
    slider bounded to the span the grid actually measured (keep 80% down to keep 10%; the 5% rate
    is reached by fewer than half the transcripts, so it is refused rather than offered), with
    tick marks at the operating points (2x/3x/4x) and named presets — low, medium, high, extreme —
    mapped to fixed ratios. Blocked on the knob decision under
    [Decided 2026-09-16](#decided-2026-09-16), which waits for the core to land. Still open
    inside it: what each preset maps to, and whether the aggressive preset is what turns banding
    on. Raised 2026-09-16.

Resumption and downstream task evaluation are deferred by decision, not forgotten.

## Corpus rule, restated

The five local sessions are the author's own work logs. They are gitignored, never committed,
and **never quoted** — not in this document, not in a results table, not in a commit message.
They contribute counts only, under the labels `local_5k` … `local_200k`. Sample lines printed
during development come from public transcripts exclusively.

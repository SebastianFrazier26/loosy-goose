# Handoff — SLM rewrite compression for loosy-goose (2026-10-22)

## Topic of discussion (the user's exact words)

> The original idea for lossy - goose (which changed quite a bit) was basically a text-mimic of image compaction where word groups (sentence fragments/sentences) are normalized to smaller words (think of how caveman works for claude), but instead it happens mathematically using a vector mapping of the english language (which is essentially what an LLM is, but ours can be a lighter-weight SLM) That way we literally compact text (full conext history blocks) into smaller pieces the same way a image compression system would work

The goal of the next session is to **design** a fair test of that idea: a small local model that *rewrites* word
groups into shorter equivalents, not one that only deletes tokens. This is architectural work, so start with the
brainstorming skill, and get the user's approval at each stage before writing code.

## Where the project stands

Repo `~/Developer/loosy-goose` (GitHub `SebastianFrazier26/loosy-goose`, public, MIT). Read `CLAUDE.md` first, including its hard rules.

- **Shipped:** v0.1.0 was released 2026-10-15 and includes the MCP server.
  - `/loosy` (Claude Code) and `$loosy` (Codex) fork the live session into a compressed copy and resume it.
  - The compression pipeline is extractive: supersede → leverage → per-token novelty selection at α 0.5, plus a
    verbatim 4k-token tail and a marker.
  - It is installed locally (`uv tool install --editable`) and registered with both CLIs.
- **Experiment I** (branch `exp/downstream-eval`, results in `docs/results/exp_i_downstream.txt`) measured actual
  model performance after compression.
  - Setup: 20 public transcripts and probe QA. Answering models were Haiku 5.5, Sonnet 5.5, gpt-6-luna and
    gpt-6-sol; Sonnet graded.
  - Probe QA averaged over models:

    | Condition | Light | Medium | Heavy |
    |---|---|---|---|
    | full | 0.96 | | |
    | loosy | 0.81 | 0.78 | 0.75 |
    | Sonnet summary (stand-in for `/compact`) | 0.94 | 0.93 | 0.92 |

  - The summaries also used fewer tokens than loosy.
  - Caveat: Sonnet wrote both the probes and the summaries.
- **Diagnosis:** loosy's loss is almost all *drop loss*. When the segment holding the answer is kept, accuracy is
  about 0.93, the same as the summary.
  - Lost answers sit in large code or file dumps (3–3.5k tokens each, keep rate 0.26 at light down to 0.07 at heavy).
  - Short segments dropped whole account for about 40% of the loss.
- **Token-thinning spike** (branch `spike/token-thinning`, commit 0507940, local only, never pushed). The report is
  `experiments/output/spike_thin/report.md`, with three addenda (gitignored output).
  - **Deleting predictable tokens** using SLM surprisal (pythia-31m up to Qwen3-1.7B) **failed**:
    - thinning prose costs answers;
    - code skeletons lose relations (`blocksize=1000` becomes `1000`);
    - the model choice made no difference, and no model at all did as well.
  - **The best variant is a plain rule, not the user's idea.** Inside code dumps, keep whole lines that contain
    atoms or errors and elide the rest.
    - Against loosy (Haiku, 30 transcripts): +0.07 at light (significant), +0.05 at medium and +0.04 at heavy (not
      significant).
    - It is within noise of the summary at light, and still 0.10–0.13 behind it at medium and heavy.
  - Applying the same rule to tool_result did nothing. The medium/heavy gap comes from how the budget is split
    between kinds.
  - Both transcript sets (Phase A, 20 transcripts; Phase B, 10 held-out) have now been used for decisions. Any new
    tuning needs a fresh held-out set.

## What has NOT been tested: the actual topic

An SLM that **rewrites** phrases, sentences and fragments into shorter equivalents in place: caveman-style
normalization produced generatively from a learned vector space of English, like a transform-and-quantize step in
image coding. Open design questions:

1. **Model.** It must be generative, small, local and Apache/MIT-licensed. The user wants it run locally and "as
   small as possible". Candidates to check on Hugging Face (verify they exist; don't guess): the SmolLM2 family,
   Qwen3 small, and others. The user picks; present options with a size/quality curve.
2. **Where it applies.** Prose and thinking first. Code and tool output stay extractive (line-keep is the best
   known option there). Tool-call inputs are never touched.
3. **Atom safety.** Rewrites must not corrupt identifiers, paths, numbers or error strings. Options: mask atoms
   before rewriting and restore them after, verify afterwards and fall back to the original segment, or constrain
   decoding.
4. **How to rewrite.** Options: a prompted SLM ("rewrite tersely"); a small model fine-tuned on (verbose → terse)
   pairs; or a true embedding → quantize → decode pipeline. A plain reconstruction decoder returns the same length,
   so it doesn't compress; that's why earlier vector ideas were rejected.
5. **Combining with the pipeline.** Lossless supersede first, then rewrite (quantize within segments), then select
   (drop blocks). Budget in tiktoken o200k tokens.
6. **Evaluation.** Offline string-match survival proved unreliable twice, ranking variants opposite to real QA, so
   judge by downstream QA.
   - Reuse the Experiment I harness (`experiments/exp_i_downstream.py`) with Haiku answering and Sonnet grading,
     on a **fresh held-out transcript set**.
   - Probes should come from a model other than the summarizer, or be drawn from facts the agent later used.
   - Readability matters: show before and after excerpts.

## Constraints and norms (still binding)

- **The user decides all design questions:** propose options and wait. Overnight runs: park questions, do other
  work, stop only on a full block.
- **Data:**
  - Public data only (`data/public/`).
  - Never read `data/local/`, `~/.claude/projects` or `~/.codex/sessions`.
  - Never quote local session content.
- **Branches:** feature branches only, and never push main. Pushing needs the user's go-ahead.
- **Global config:** auto mode blocks changes to it; give the user commands to run instead.
- **Build from scratch:** the core is ours, so no LLMLingua-style libraries; reading papers is fine.
- **Cost:** Claude usage hit the monthly spend limit once (the Experiment I run of about 2,300 calls). Budget calls
  up front, cap them, and stop on a 429 spend-limit error.
- **Long runs:** use a detached `nohup caffeinate -i` driver that can resume and writes its own results. Subagents
  die when the Claude Code process exits.
- **Repo rules:** every new transform downstream of segmentation needs its own version stamp (see `CLAUDE.md`).
  CHANGELOG entries are dated. Comments only where load-bearing.

## Pending decisions from the spike (unresolved)

1. Whether to productize line-keep for code dumps (at least at light).
2. Whether to pursue the budget split for medium and heavy.
3. Whether to run a larger confirmation with more transcripts and the GPT answerers.

## Key paths

- Spec for the shipped handoff: `docs/superpowers/specs/2026-10-13-mcp-handoff-design.md`
- Experiment I spec: `docs/superpowers/specs/2026-10-14-exp-i-downstream-design.md`
- Spike code: `experiments/spike_thin/` on `spike/token-thinning`
- Spike report: `experiments/output/spike_thin/report.md`
- Pipeline: `src/loosy_goose/pipeline.py`, `segment.py` (atoms), `budget.py`, `code.py` (AST banding)

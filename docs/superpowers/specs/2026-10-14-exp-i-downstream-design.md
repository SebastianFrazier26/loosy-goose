# Experiment I — downstream model performance after compression — design

Status: approved by the user 2026-10-14. Branch `exp/downstream-eval`. Runner
`experiments/exp_i_downstream.py`, progress view `experiments/exp_i_status.py`.

## Question

Do models actually do as well after loosy-goose compression as with the full context, and how does
that compare with a model-written summary at the same token budget (a stand-in for built-in
`/compact`)? Phase 1 and Phase 2 measured proxies only — atom recall and semantic coverage. This
experiment measures the thing those proxies stand in for.

## Corpus

About 20 public agentic transcripts from `data/public/trace-commons` and `data/public/swe-gym`,
each 20k–60k tokens (tiktoken `o200k_base` via `loosy_goose.tokens`, counted over segments as the
budget is), chosen with a fixed seed. Up to 10 per source; the shortfall of one source is filled
from the other. Public data only — never `data/local`, `~/.claude/projects` or `~/.codex/sessions`.

## Cut point

Each transcript is cut late, immediately before a real assistant action (the start of a run of
assistant turns that holds a tool call or assistant text) whose position by tokens falls in the
last third (2/3 to 0.95 of the transcript), picked with a per-transcript fixed seed. The context is
everything before the cut; the held-out reference is that real next action (thinking excluded,
long blocks clipped).

The last 4,000 tokens before the cut (whole turns, so at least 4,000) stay verbatim, as `/loosy`
does. Only the earlier stretch is compressed, and it must hold at least 8,000 tokens.

## Conditions

All act on the compressible stretch; the tail is always verbatim.

1. `full` — no compression (upper bound).
2. `loosy-light` (keep 0.50), `loosy-medium` (0.33), `loosy-heavy` (0.25) — the shipped pipeline,
   `compress_transcript` with default options.
3. `summary-light` / `-medium` / `-heavy` — Sonnet summarises the compressible stretch at the same
   token budget as the matching loosy level: the rendered size of that loosy condition's earlier
   stretch. A summary more than 10% over budget is cut back to it.

   Two adjustments made while building it (2026-10-14, smoke run): asked for "at most N tokens",
   Sonnet wrote about 1,000 tokens against 3.5k–13k budgets, so the prompt now states a required
   word range after the session text; that reaches roughly 55–80% of budget. A summary under 30% of
   budget (in the smoke run, one outright refusal) is rejected and retried, never used.

## Tasks

- **Probe QA.** Sonnet writes about 10 short factual question/answer pairs from the compressible
  stretch only (file paths, values, error messages, decisions, identifiers), with short checkable
  answers. A probe is kept only if Sonnet answers it correctly from the full context. All probes for
  a transcript × condition are answered in one call per answering model.
- **Next step.** The answering model is given the context and asked for its next action (tool and
  target/arguments or intent, briefly), compared with the held-out real action.

## Answering models

- Haiku 5.5 and Sonnet 5.5 through headless `claude -p --model claude-haiku-5-5|claude-sonnet-5-5`.
- gpt-6-luna and gpt-6-sol through `codex exec -m gpt-6-luna|gpt-6-sol`.
- Every call runs from its own empty temp directory, with no tools and no session persistence where
  the CLI supports it, the prompt on stdin. The flags were taken from each CLI's `--help`, not
  guessed, and are documented beside their definitions in the runner. Claude Code is fully isolated
  (`--tools ""`, `--safe-mode`, `--strict-mcp-config`, `--no-session-persistence`). Codex cannot drop
  its own exec tool or the global `~/.codex/AGENTS.md` by flag; it runs `--ephemeral` and
  `--ignore-user-config` in a read-only sandbox with every optional feature disabled, and any tool
  call it still makes is counted per record.
- Reasoning effort is pinned to `medium` on both CLIs so the run does not depend on user settings.

## Grading

Sonnet grades each answer correct / partial / wrong (1 / 0.5 / 0) against the reference, blind to
condition and answering model. The next step is graded as agreement with the real action (same /
partially same / different).

## Scale and robustness

- About 20 transcripts × 7 conditions × 4 models × 2 tasks ≈ 1,100 answer calls, plus one probe
  generation and three summaries per transcript, plus one grading call per answer.
- Every call is checkpointed to JSONL keyed by (version, transcript, condition, model, task); reruns
  skip done keys. A stored record also carries a digest of its prompt and counts as done only if the
  digest still matches and the reply still validates, so a redone summary re-asks the answers built
  on it and their grading. Usage-limit and rate errors back off exponentially to a 30-minute sleep and retry;
  one failure never aborts the run, it is logged and retried by the next run. The run is meant to go
  unattended overnight under `caffeinate`.
- `experiments/exp_i_status.py` shows progress, like `grid_status.py` does for the grids.

## Output

- Raw: `experiments/output/exp_i/` (gitignored) — the call checkpoint, failures, manifest, rendered
  contexts, summary.
- Summary: accuracy per condition × model for both tasks with 95% bootstrap CIs over transcripts,
  plus tokens kept per condition, in `docs/results/exp_i_downstream.txt`. Only aggregate numbers
  leave the output directory.

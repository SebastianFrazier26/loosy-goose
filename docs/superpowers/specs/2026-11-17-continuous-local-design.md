# Continuous loosy for open-weight models (2026-11-17)

Status: design, awaiting user approval. Branch `feat/continuous-local`.

## Intent and why

Loosy loses to a frontier-model summary, but it beats every local summary we tested by 0.26–0.34, and
continuous freeze-mode loosy beats threshold loosy by about 0.04 when no frontier model is in the loop
(`docs/results/exp_continuous_f.md`). So the user is repositioning loosy: it becomes **a compaction strategy
for agents running on open-weight models** such as Qwen, not an add-on for Claude.

There is also a second reason, which is a hypothesis to test rather than a measured fact. On local hardware,
re-reading the prompt (prefill) dominates each turn. Freeze mode never rewrites old text, so a local server's
prefix cache stays valid. Freeze mode's lower uncached-token count should therefore become real seconds saved per turn.

The work has two parts, which run in parallel (user's call):

- **A. The Qwen test.** Old (threshold) loosy and continuous loosy against Qwen summarizing itself, all answered
  by Qwen on the laptop's GPU.
- **B. The feature.** Continuous freeze-mode compaction as a shipped part of loosy-goose, usable in front of a local
  open-weight model.

## Decided by the user

- The target is open-weight models, starting with Qwen. The Claude Code integration is set aside.
- Testing runs locally on the user's 16 GB M5 MacBook, on its GPU, through MLX.
- Test A compares old loosy against summary compaction where Qwen writes the summaries.
- B starts now, alongside A, and does not wait for A's result.

## A. The Qwen test

The question: when the agent's own model is Qwen, which compaction keeps the most of the session?

- **Data.** Set F (25 transcripts, 75 checkpoints, C = 65%), reusing its stored probes, simulations and arms
  (`experiments/continuous/`). No new probes are written.
- **Answerer.** Qwen3-8B, 4-bit MLX (`Qwen/Qwen3-8B-MLX-4bit`, already cached locally), with greedy decoding and thinking off.
  It runs through mlx-lm 0.32.2, with the Metal cache cap from `lsum.py`.
- **Grader.** Sonnet 5.5, unchanged, so the results stay comparable with every earlier run.
- **Arms.** All at threshold-matched context size:
  - `full`, the whole history. It fits Qwen3's 32k window, since the set F maximum is about 19k o200k tokens. This
    must be checked in Qwen's own tokenizer, and any checkpoint over the limit is reported, not silently cut.
  - `qthresh`, threshold compaction with **Qwen3-8B writing its own summaries**. This is what a Qwen agent would do on its own.
    It uses the same event schedule and the same 0.33·C target as `thresh`, and reuses the `lsum` prompt and code.
    About 50 local summary generations.
  - `tloosy`, old loosy at the threshold (stored contexts).
  - `cfrzm`, continuous freeze (stored contexts).
  - `cdec` is left out of A: it was null against `cfrzm` on set F, and costs more cache.
- **Cost.**
  - About 300 local answer generations and about 50 local summaries. My estimate is 30–90 s each at 8B, so 3–7 hours, run in the background.
  - About 300 Sonnet grading calls, capped at 360, stopping on a spend limit.
- **Latency side-measurement.** On 3 set F transcripts, replay every turn through `mlx_lm` with prompt-cache reuse
  for `qthresh`, `tloosy` and `cfrzm`, and record prefill seconds per turn. This tests the cache hypothesis in wall-clock terms.
- **Version stamps.** The answerer model is part of each record key (the existing exi `Model`). `qthresh` gets `QTHRESH_VERSION = 1`.
- **Reading, fixed before the data:**
  - If `cfrzm` or `tloosy` beat `qthresh` (paired, sign p<0.1), loosy is the better compaction for Qwen agents.
  - `cfrzm` − `tloosy` repeats the continuous-vs-threshold test with a Qwen answerer.
  - If `cfrzm` costs clearly less prefill time per turn, the cache hypothesis is confirmed.

## B. The feature

**Goal.** An agent talking to a local open-weight model gets continuous freeze-mode compaction without changing
its own code.

- **A library core** in `src/loosy_goose/continuous.py`: a `FreezeCompactor` that takes the full message list
  each turn and returns the compacted list.
  - It keeps the measured policy: a verbatim tail of 4k tokens, 4k chunks, and keep 0.6, which is `cfrzm`'s calibrated setting.
  - It compresses a chunk once, as it ages out of the tail, with the shipped selector. A frozen chunk is never rewritten.
  - Its state survives restarts: it is saved to disk, keyed by a conversation id.
  - It is deterministic, so the same history gives the same output.
  - Messages are the OpenAI chat format (`role`/`content`, including tool calls and tool results). The existing generic
    `[{role, content}]` loader is the starting point.
- **An OpenAI-compatible proxy** (`loosy-goose proxy`): a local HTTP server that sits in front of any local server
  speaking the OpenAI chat API (mlx_lm.server, Ollama, LM Studio, llama.cpp, vLLM).
  - It rewrites `messages` through the compactor, then forwards the request.
  - It streams responses back unchanged.
  - It binds to localhost only.
  - It never stores or logs message content beyond the compactor's own state.
  - It passes auth headers through without recording them.
- **Opt-in.** It is a new extra, kept out of the default install, and it does not change existing CLI or MCP behaviour.
- **First target.** The first target is `mlx_lm.server`, because MLX is already the local stack. End-to-end smoke test: an agent loop of a few turns
  against `mlx_lm.server` with Qwen3-4B through the proxy.
- **Tests.**
  - Core: freezing never mutates frozen chunks; output is deterministic; state round-trips through disk; tool-call/result pairs are never split; the tail is kept verbatim.
  - Proxy: smoke tests only, per repo convention.
- **Docs.** A README section and a CHANGELOG line, written as the work lands, not at the end.

## Non-goals and anti-goals

- **Not a Claude Code integration.** The user set that aside.
- **No model calls inside the compactor.** It stays extractive, local and free, which is the whole point.
- **Anti-goal:** a proxy that quietly changes what the model sees without saying so. Each compacted request carries
  the same "earlier part compressed" header the experiments used.
- **Anti-goal:** a proxy that becomes a security hole, through exposure off localhost, logged keys, or stored transcripts left beyond the compactor state.

## Left to the builder

- HTTP framework choice. The repo norm is FastAPI for new services; a dependency-light standard-library server is also fine if justified.
- How a conversation id is derived when the client sends none. My default is a hash of the system prompt plus the first user message.
- Exact module layout, provided the core is testable without the proxy.

## Builder-check answers (agent's calls, 2026-11-17)

- **Compacted message shape.**
  - The original system prompt is kept as is.
  - The frozen prefix is rendered as flat text, as measured: `render_segments` plus the "earlier part compressed" header. It is sent as one synthetic `user` message after the system prompt.
  - The verbatim tail follows as the original messages, with their tool calls and results intact.
  - This matches what was tested and can never orphan a tool result. Old tool calls become prose inside the prefix, which is exactly what the experiments measured.
  - A tail cut never separates an assistant tool call from its tool results: the tail boundary moves earlier to keep them together.
- **History that is not an extension of the stored state** (an edit, a regenerate, a branch, a client-side compaction, or an id collision).
  - Each frozen chunk stores a hash of the source messages it covers. If any stored chunk's source no longer matches, that chunk and every later one is dropped, and they are recompressed from the incoming history.
  - Determinism means the result equals a fresh run. It is slower once, which is acceptable.
- **`qthresh` events in test A.**
  - `qthresh` finds its own compaction events from its own summary lengths, which is what a Qwen agent would actually do.
  - Its mean context per turn is reported next to the others. If it lands more than 10% off `thresh`'s, the report says so, and the comparison is read as accuracy at the context size each arm reached.
- **Token units.** o200k (`tokens.count_tokens`), for parity with the calibrated settings. There is no per-backend tokenizer.
- **Spend limit.** The existing harness stops on a 429 or spend-limit error from the CLIs. The call cap (360) is the hard limit, and no dollar figure is set.

## User answers (2026-11-17)

- **Overflow: decay the oldest.** The compactor freezes as measured. When the context nears a set share of the model's window, it merges and recompresses its oldest chunks: the `cdec` tier merge from `experiments/continuous/sim.py` (merge `m` chunks, recompress at keep `k`), applied oldest-first until the context fits. This keeps long sessions bounded. On set F, `cdec` tied plain freeze on accuracy at about 6% more uncached tokens.
- **Start point: measure both.**
  - Test A gains a fifth arm, `cfrzg` (gated freeze). It passes the history through untouched until the context exceeds 50% of Qwen3's 32k window, then freezes from that point. `cfrzm` stays as the always-on arm.
  - Paired comparison: `cfrzg` − `cfrzm`. Context and uncached cost for `cfrzg` are reported as for every arm.
  - The 32k window is measured in o200k units, for parity with the calibrated settings, so the gate is 16,000 o200k tokens.
  - That adds about 75 Qwen answers and 75 grading calls, and the grading cap rises from 360 to 450.
  - The feature supports both modes with a `start_at` setting (a share of a configured `window` size, or 0 for always). Its default is chosen from test A's result: the gate wins only if `cfrzg` beats `cfrzm` at a sign test p below 0.1, and otherwise the default is always-on.

## User answers, round 2 (2026-11-17): open-weight only

- **No closed models anywhere** in these tests. Claude and OpenAI models are out for answering, summarizing, grading and probe writing. Loosy is measured solely on, and for, open-weight models.
- **Grader and probe writer:** Mistral-Nemo 12B Instruct (2407), 4-bit MLX (`mlx-community/Mistral-Nemo-Instruct-2407-4bit`, Apache-2.0, 6.9 GB, verified on the hub 2026-11-17). It is a different family from Qwen, so neither the answerer nor the summarizer grades itself.
- **Probes are regenerated** with Mistral-Nemo for all 75 set F checkpoints, using the exi probe prompt. The stored GPT probes are not used.
- **Consequences** (agent's notes):
  - Every generation is local, so there is no API call cap. The limit is wall-clock. The estimate is about 875 local generations: about 75 probes, about 50 Qwen summaries, about 375 Qwen answers and about 375 Nemo grades. At 30–90 s each, that is roughly 10–20 hours. Phases run one model at a time and are resumable.
  - Results are a new, standalone set. They are not comparable number for number with the Claude-answered set F results, because the grader and probes differ. Comparisons are within this run only.
- **Wording.** "OpenAI-compatible" in this design means only the chat request format that local servers (mlx_lm.server, Ollama, LM Studio, llama.cpp) speak. No OpenAI model is involved, and the docs call it "the chat API format local servers speak".

## Stage Q log

- **2026-11-17, run path built** (`run.py --set F --answerer qwen3-8b`). Smoke calls through `local.mlx_runner` on this machine:
  Qwen3-8B load 8.7 s, a 37-token reply in 5.3 s; Mistral-Nemo download plus load 58.2 s, a 21-token reply in 31.1 s.
  `--plan`: 75 Nemo probes (`probes.nemo1`), 50 Qwen summaries (placeholder schedule), 375 Qwen answers and 375 Nemo grades:
  875 local calls, 0 API calls. The runner cap is 2,000 invocations, a runaway guard only.
  `--dry-run` mean context per turn: full 18,827, qthresh 12,449 (placeholders; thresh's real figure is 12,342), tloosy 12,482,
  cfrzm 12,355, cfrzg 12,717.
- **Correction to section A.** `full` does not always fit Qwen's window. 15 of 75 full checkpoints (4 at 75%, 11 at 100%)
  are over 31,232 Qwen tokens. The largest full context is 50,673 o200k tokens, against the stated maximum of about 19k.
  Those answers are never queued (each is recorded in `qwen_failures.jsonl`), the context is never cut, and the report lists them.
- **2026-11-17, fix round 1 (controller rulings).**
  - **Scoring `full` per checkpoint.** Each arm − full row is now scored per checkpoint. The difference is taken only at checkpoints
    where `full` has a grade, then averaged within each transcript, and the bootstrap and sign test still resample transcripts.
    Each row prints n as both transcripts and checkpoints, so no transcript is dropped for a missing full checkpoint. Pairs between
    compacted arms, and the arm table, stay per transcript with all three checkpoints required.
  - **Over-window answers.** They are skipped when the job is built, not refused by the runner, because `call_with_retry` would retry
    a refusal, with sleeps, even though the prompt can never fit.
  - **Failed summaries.** A qthresh summary that fails is recorded, with the key `qsum|<label>|e<k>`, and that transcript's qthresh
    arm is skipped. The run goes on.
  - **Nemo tokenizer.** It loads with `fix_mistral_regex=True` (transformers 5.16.1) in both `local._tokenizer` and the `mlx_lm`
    load. Without the flag, mlx_lm's local-path tokenizer load warned of an incorrect regex; with it, the warning is gone.
- **2026-11-17, final-review fixes (controller rulings; no GPU used, nothing generated yet).**
  - **Chunked Nemo probes.** Whole-history probe prompts reached 53k Nemo tokens, which would not fit 16 GB. The pre-tail history
    is now split at turn boundaries (a single turn over the cap is halved at line breaks) into segments of at most 12,000 Nemo
    tokens (`experiments/continuous/probe_segments.py`). The 10 probes are shared out by token share, at least one per segment.
    Each segment is its own `probes.nemo1` record, keyed `<label>@<pct>#s<i>`, and a checkpoint's probe set is the segments' probes
    in order. A segment reply must hold at least 3-in-10 of the probes asked (one at minimum), and only the first n count.
    `--plan`: 184 segment jobs for 75 checkpoints; the largest prompt is `swe-gym_row129@75#s1` at 12,090 Nemo tokens.
    A bounded recent window was rejected because it would favour tail-heavy arms. The cost: probes cannot span segments.
  - **MLX RuntimeError.** `local.mlx_runner` catches it (out of memory included), frees the Metal cache and raises a
    non-transient `CallError`, which the Runtime records before moving on. The qthresh summary loop records it, drops the
    generator and builds a new one for the next job.
  - **Attempts and resume.** `call_with_retry` gives `mlx` models a single attempt on invalid replies and non-transient errors,
    since greedy retries would be identical (other models are unchanged). Keys in `qwen_failures.jsonl` are skipped on resume and
    over-window answers are no longer re-recorded. `--retry-failed` calls them again.
  - **Report.** New: a table of expected, answered and graded checkpoints per arm; failure counts (probe segments, summaries,
    answers, over-window answers, grades); a note that the arm table's full row covers only transcripts whose three checkpoints all
    fit; and the ratio of Qwen summary tokens to target over the cached summaries.
  - **lsum cache.** Writes are atomic (temp file plus rename). A torn file is treated as a cache miss. Keys and content are
    unchanged.
  - **Latency warmup.** It is now one CHUNK (2,048 tokens) instead of 256.
  - **`--labels L1,L2`.** Qwen path only: it restricts the run to the named transcripts, with keys identical to the full run, and
    writes the report to `qwen_labels_report.md` only. `--plan` prints the largest Nemo probe prompt and the largest Qwen `full`
    answer prompt that is queued: `swe-gym_row409@75`, 30,755 Qwen tokens. `--set F --plan` without `--answerer` still shows
    0 pending.

- **2026-11-21 — Nemo probe generation fix (pilot `--labels swe-gym_row409,swe-gym_row129`).**
  - **Finding.** With `probes.nemo1`, 9 of 12 Nemo `probe_gen` calls failed with "invalid reply after 1 attempts": Nemo ignored the
    JSON-only instruction and wrote numbered markdown `**Q:** / **A:**` lists, so only one checkpoint was graded.
  - **Fix (local `probe_gen` only; Haiku/Sonnet prompts and keys unchanged).** `local.probe_gen_prompt` prefills the assistant turn
    with `{"probes": [{"q": "` (`PrefilledPrompt`; the runner continues from it and stores prefill plus continuation), says "single JSON
    object, no introduction, no markdown, no numbering", and gives a bare `<question>`/`<short answer>` format line. Up to 3 attempts
    (`call_with_retry` reads `max_attempts` from the prompt object); attempts after the first sample at temperature 0.7, because greedy
    repeats the same reply. `close_probe_json` appends the one brace Nemo forgets after the array (`...}]`); the strict validator
    (`parse_segment_probes`: JSON, q and a present, at least 3 in 10 of the asked count) is unchanged.
  - **Detours, kept as evidence.** `nemo2` used a concrete one-shot example; Nemo copied its values ("src/app.py", "42") into probes, so it
    was dropped (`nemo3`). `nemo3` still failed 2 of 17: greedy decoding fell into a repetition loop on a directory-listing segment
    (1,536 tokens), and sampled retries ended `}]` without the closing brace. The brace repair gave `nemo4`.
  - **Result (`probes.nemo4`).** 17 of 17 segments valid: 10 on the first attempt, 5 on the second, 2 on the third. Probe generation
    took 1,247 s of call time (about 21 min wall including the Nemo load; the greedy loop and retries dominate the slow calls).
    The pilot then answered 27 and graded 30 calls with 0 failures after `--retry-failed`; all arms graded 6 of 6 checkpoints (full 3 of
    3, three over the Qwen window).
  - **Skip-logic note.** Probe keys carry the condition, so `nemo1`-`nemo3` failures never match `nemo4` keys, and the report counts only
    probe-segment failures of the current version. Grade and answer keys do not carry the probe version: two `grade_probe` failures recorded
    under `nemo3` were skipped on the `nemo4` run until `--retry-failed`. After changing a probe version, rerun once with `--retry-failed`.

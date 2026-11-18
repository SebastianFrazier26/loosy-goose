# Qwen Replay Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-answer set F with a local Qwen3-8B on the laptop GPU, and compare these arms:

- full context;
- Qwen summarizing itself at a threshold;
- old threshold loosy;
- continuous freeze, always-on and gated.

Plus a prefill-latency replay.

**Architecture:**

- **Answerer.** The `experiments/continuous` harness gains a second answerer. A local MLX runner sits behind the existing `exi.Runtime`, and answers are keyed by answerer name. Haiku's stored records are untouched.
- **Grading and arms.** Grading stays on Sonnet through the CLI. Two new arms (`qthresh`, `cfrzg`) are added to `sim.py`.
- **Latency.** A separate script replays every turn through mlx-lm with prompt-cache reuse.

**Tech Stack:** Python 3.12, mlx-lm 0.32.2 (`slot` extra), and the existing exi/downstream harness.

**Spec:** `docs/superpowers/specs/2026-11-17-continuous-local-design.md` (section A and the answers below it)

## Global Constraints

- Answerer: `Qwen/Qwen3-8B-MLX-4bit`, greedy decoding, thinking off, with the Metal cache cap from `experiments/slot_compress/lsum.py` (`mx.set_cache_limit(512MB)`, `mx.clear_cache()` after each generation).
- Grader: Sonnet 5.5, unchanged. The Sonnet grading cap is 450 calls, and the run stops on a 429 or spend-limit error.
- Local generations do not count against the API cap. Only one Qwen model is resident at a time, with workers=1 for local jobs.
- Data: set F only, reusing its stored probes and sims. No new probes.
- Arms are `full`, `qthresh`, `tloosy`, `cfrzm` and `cfrzg`. `QTHRESH_VERSION = 1` and `CFRZG_VERSION = 1` go in their keys. The `cfrzg` gate is 16,000 o200k tokens (50% of 32k).
- Every existing record key (set E, set F with Haiku) must still hit, so `--set F --plan` for Haiku shows 0 pending.
- Never read `data/local/`, `~/.claude/projects` logs or `~/.codex/sessions`. Commit on `feat/continuous-local` only.
- Docs are updated as each result lands: spec log, `docs/results/exp_continuous_qwen.md`, README section.

## Review Focus

- **A context longer than Qwen's 32k window, counted in Qwen's own tokenizer.** It must be counted and reported, never silently truncated (Task 1).
- **Qwen output with no parseable answer list,** for example a looping generation. It must fail validation and retry at most 3 times, then record a failure, not a fake answer (Task 1).
- **A Qwen summary much shorter than its target.** `qthresh` then fires on its own schedule, and its mean context is reported next to the others, with the more-than-10% flag (Task 3).
- **A resumed run after a crash mid-generation.** Stored local answers are reused, not regenerated (Task 4, via Store).
- **The `cfrzg` gate never triggering** on a short transcript. The arm must equal `full` there (Task 3).

---

### Task 1: Local MLX runner and Qwen answerer model

**Files:**
- Create: `experiments/continuous/local.py`
- Modify: `experiments/exp_i_downstream.py`
  - the `Cli` literal (about line 107) gains `"mlx"`;
  - `MODELS` gains `Model("qwen3-8b", "mlx", "Qwen/Qwen3-8B-MLX-4bit")`.
- Test: `tests/test_continuous_local.py`

**Interfaces:**
- Produces:
  - `local.mlx_runner(model: exi.Model, prompt: str) -> exi.CallResult`
  - `local.dispatch(cli: exi.Runner) -> exi.Runner`. It routes `model.cli == "mlx"` to `mlx_runner` and everything else to `cli`.
  - `local.qwen_tokens(text: str) -> int`, which counts tokens with the Qwen tokenizer.
  - `local.QWEN_WINDOW = 32_768`.

- [ ] **Step 1: Write failing tests**
  - `test_dispatch_routes_mlx_to_local` uses monkeypatched `mlx_runner` and `cli` stubs. It asserts the qwen model goes local and haiku goes to cli.
  - `test_existing_model_names_unchanged` asserts `[m.name for m in exi.MODELS][:4] == ["haiku-5.5","sonnet-5.5","gpt-6-luna","gpt-6-sol"]`.
- [ ] **Step 2: Run** `uv run pytest tests/test_continuous_local.py -v`. Expected: FAIL (no module).
- [ ] **Step 3: Implement `local.py`.**
  - Load the model once, cached at module level per model_id. Reuse `lsum`'s mlx loading, chat template (`enable_thinking=False`) and cache-cap code by importing from `experiments/slot_compress/lsum.py`; don't copy it.
  - `max_tokens=1024` for answers.
  - `mlx_runner` raises `exi.CallError` when `qwen_tokens(prompt) > QWEN_WINDOW - 1024`, with the message `"context over Qwen window: N tokens"`. That surfaces as a recorded failure.
- [ ] **Step 4: Run tests.** Expected: PASS. Also run `uv run pytest -q`: all pass.
- [ ] **Step 5: Commit** `experiments/continuous/local.py`, `experiments/exp_i_downstream.py` and the test, with the message `"Continuous Qwen: local MLX runner and qwen3-8b answerer"`.

### Task 2: Answerer-parameterised answer and grade jobs

**Files:**
- Modify: `experiments/slot_compress/downstream.py`, at `answer_job` and `grade_job` (about lines 467–495).
- Test: `tests/test_continuous_local.py`

**Interfaces:**
- Produces:
  - `dl.answer_job(label, cond, context, probes, answerer: exi.Model = HAIKU) -> exi.Job`
  - `dl.grade_job(store, label, cond, probes, answerer: exi.Model = HAIKU) -> exi.Job | None`
  - The grade job reads the answer stored under `answerer.name`, and its key records the answerer, so a Qwen grade and a Haiku grade never collide.

- [ ] **Step 1: Write failing tests.**
  - `test_default_answerer_keys_unchanged`: `dl.answer_job("L","c","ctx",probes).key` equals the key computed before this change, hard-coded from running the current code once.
  - `test_qwen_answer_key_differs`: the qwen job's key differs from the haiku job's key and contains `"qwen3-8b"`.
- [ ] **Step 2: Run.** Expected: FAIL (unexpected keyword `answerer`).
- [ ] **Step 3: Implement** the default parameter threaded through. With the default value, the code paths must be identical to today's.
- [ ] **Step 4: Run** the tests and the full suite, plus `uv run python experiments/continuous/run.py --set F --plan`. Expected: PASS, and 0 pending for the existing Haiku arms.
- [ ] **Step 5: Commit**, with the message `"Downstream jobs: answerer parameter, Haiku keys unchanged"`.

### Task 3: `qthresh` and `cfrzg` arms

**Files:**
- Modify: `experiments/continuous/sim.py`:
  - constants;
  - `_views` dispatch;
  - `F_ARMS` stays as it is, with a new `Q_ARMS = ("full", "qthresh", "tloosy", "cfrzm", "cfrzg")`;
  - `HEADERS["qthresh"] = HEADERS["thresh"]` and `HEADERS["cfrzg"] = HEADERS["cfrz"]`.
- Test: `tests/test_continuous.py`

**Interfaces:**
- Consumes: `_thresh_views(p, summarize)` and `Freezer(..., keep=FRZM_KEEP)`.
- Produces:
  - `sim.QTHRESH_VERSION = 1`, `sim.CFRZG_VERSION = 1`, `sim.CFRZG_GATE = 16_000`.
  - `qthresh` is `_thresh_views` with a Qwen summarizer passed in by `run.py`; it has no new simulation logic.
  - `cfrzg`: at each boundary, if the full context at `t` (`p.before[t]`) is at most `CFRZG_GATE` and the gate has never opened, the view equals `full`'s. Once it opens, it stays open, and a `Freezer` at `FRZM_KEEP` advances as in `cfrzm`, compressing every chunk from turn 0.

- [ ] **Step 1: Write failing tests.**
  - `test_cfrzg_equals_full_below_gate`: a synthetic transcript under 16k gives views identical to `full` at every boundary.
  - `test_cfrzg_matches_cfrzm_after_gate`: on a long synthetic transcript, after the gate opens, the frozen text equals `cfrzm`'s at the same boundary.
  - `test_cfrzg_gate_stays_open`.
- [ ] **Step 2: Run** `uv run pytest tests/test_continuous.py -v`. Expected: the 3 new tests FAIL.
- [ ] **Step 3: Implement** the `cfrzg` branch in `_views`, and allow `arm == "qthresh"` to route to `_thresh_views`.
- [ ] **Step 4: Run.** Expected: PASS, and the full suite passes.
- [ ] **Step 5: Commit**, with the message `"Continuous sim: qthresh and gated-freeze cfrzg arms"`.

### Task 4: `--answerer qwen3-8b` run path, Qwen summaries, and report

**Files:**
- Modify: `experiments/continuous/run.py`
- Test: `tests/test_continuous_local.py`

**Interfaces:**
- Consumes: Tasks 1–3, plus `lsum` summarization with `Qwen/Qwen3-8B-MLX-4bit` (the `lsum8b` model key).
- Produces:
  - **CLI.** `run.py --set F --answerer qwen3-8b [--plan|--dry-run|--report] [--max-calls 450]`. The arms come from `sim.Q_ARMS`.
  - **Keys.** Condition keys are `cond(cfg, arm)` plus `.q1` for `qthresh` and `cfrzg`. Existing arms keep their keys; only the answerer name differs.
  - **Report.** `docs/results/exp_continuous_qwen.md`, written by `--report` and also at the end of a run.
  - **Report contents:**
    - an accuracy, context and uncached table per arm;
    - a per-checkpoint table;
    - paired rows `qthresh−tloosy`, `cfrzm−qthresh`, `cfrzm−tloosy`, `cfrzg−cfrzm`, and each arm minus full;
    - a count of checkpoints over the Qwen window;
    - a flag when `qthresh`'s mean context is more than 10% off `thresh`'s 12,342.

- [ ] **Step 1: Write failing test** `test_qwen_plan_counts`, with stub plans for 2 transcripts × 3 checkpoints. It asserts that `call_counts` for the qwen answerer reports local answers = 2×3×5 and grades = 2×3×5, and that Qwen summaries are counted as local, not API.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement the run order.**
  1. Qwen summaries for `qthresh` events, generated locally and cached to `experiments/output/continuous/f/qsum_cache/` (rounds as in `summary_rounds`).
  2. Local answers, with workers=1 and `dispatch(guard.runner)`, where the guard counts only cli calls.
  3. Sonnet grades, with workers=4.
- [ ] **Step 4: Run** the tests, `--set F --answerer qwen3-8b --plan` and `--dry-run`. Expected: PASS. The plan shows about 375 local answers, about 50 local summaries and about 375 grades, and the dry run writes sample contexts.
- [ ] **Step 5: Commit**, with the message `"Continuous Qwen: answerer run path, Qwen self-summaries, report"`. Log the plan numbers in the spec's new "Stage Q log" section in the same commit.
- [ ] **Step 6: Launch in the background.** `caffeinate -i uv run python experiments/continuous/run.py --set F --answerer qwen3-8b --max-calls 450 >> experiments/output/continuous/f/qwen.log 2>&1`. When it finishes, run `--report`, then update the spec log and the README section, and commit.

### Task 5: Prefill latency replay

**Files:**
- Create: `experiments/continuous/latency.py`
- Test: `tests/test_continuous_local.py`

**Interfaces:**
- Consumes: `sim.run_arm`/`_views` and `sim.cost_context` (the context without the moving tail marker), plus `local`'s model loader.
- Produces:
  - **CLI.** `latency.py --set F --n 3 --arms qthresh,tloosy,cfrzm`.
  - **Output.** `experiments/output/continuous/f/latency.json` and a table appended to `docs/results/exp_continuous_qwen.md`, with the mean prefill seconds per turn and the total per arm.

- [ ] **Step 1: Write failing test** `test_reused_prefix_len`: `latency.reused_prefix(prev_tokens, cur_tokens)` returns the common-prefix length of two token-id lists. This is the only pure logic.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement the replay.**
  - Per arm, per transcript, keep one mlx-lm prompt cache (`mlx_lm.models.cache.make_prompt_cache`).
  - At each boundary, tokenize `cost_context` with the Qwen chat template, trim the cache to `reused_prefix`, prefill the rest, and time it with `time.perf_counter` around the prefill only, with max_tokens=1.
  - Use the first 3 set F transcripts in label order.
  - The `qthresh` contexts use the cached Qwen summaries from Task 4, so run this after Task 4's run.
- [ ] **Step 4: Run** the test, then the script. Expected: the test PASSES, and the table prints 3 arms with seconds per turn.
- [ ] **Step 5: Commit** the script, test, results table, spec log line and README update, with the message `"Continuous Qwen: prefill latency replay"`.

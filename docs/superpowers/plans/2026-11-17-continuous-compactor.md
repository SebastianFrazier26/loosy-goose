# Continuous Compactor and Local Proxy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship continuous freeze-mode compaction as two things:

- a library, `FreezeCompactor`;
- a localhost proxy, `loosy-goose proxy`, speaking the chat API format local servers use (often called OpenAI-compatible; no OpenAI model is involved), for agents running on local open-weight models.

**Architecture:**

- **Reading the wire format.** Chat-API messages (the format mlx_lm.server, Ollama, LM Studio and llama.cpp accept) are read into the package's `Transcript`.
- **The compactor.** `FreezeCompactor` keeps a verbatim tail and freezes each aged-out chunk once with the shipped selector. Near the window it merges the oldest chunks (tier decay). It returns `[system] + [one compressed-history user message] + tail messages`.
- **State.** State is saved per conversation and invalidated from the first chunk whose source changed.
- **The proxy.** A small FastAPI app rewrites `messages` and streams the upstream response back unchanged.

**Tech Stack:** Python 3.12, the existing `loosy_goose` pipeline (the `embed` extra), and FastAPI, uvicorn and httpx in a new `proxy` extra.

**Spec:** `docs/superpowers/specs/2026-11-17-continuous-local-design.md` (section B, builder-check answers, user answers)

## Global Constraints

- **Measured policy.** Tail 4,000, chunk 4,000 and keep 0.6, all in o200k tokens (`tokens.count_tokens`).
- **Decay.** Merge `m=4` chunks at keep `k=0.7`, oldest-first, applied only when the context exceeds `decay_at = 0.9 × window`.
- **`start_at`** is a share of `window`, where 0 means always-on. Its default is set after the Qwen test (spec rule); until then the default is `0`.
- **No model calls inside the compactor.** It is deterministic: the same messages give the same output, byte for byte.
- **Message shape.**
  - The system message(s) are kept unchanged and come first.
  - Next comes one `user` message: the header `"[Earlier part of the conversation, compressed extractively: some passages were dropped, what remains is verbatim]"`, a blank line, then the rendered frozen chunks.
  - The tail messages follow, verbatim.
  - The tail boundary never separates an assistant `tool_calls` message from its `tool` results.
  - If nothing is frozen yet, the input is returned unchanged.
- **Proxy.**
  - It binds to `127.0.0.1` only, and refuses a non-loopback `--host`.
  - It never logs message content or auth headers.
  - It passes `Authorization` through to the upstream.
  - The state dir defaults to `~/.cache/loosy-goose/continuous/`.
- **Packaging.**
  - A new extra, `proxy = ["fastapi>=0.115,<1", "uvicorn>=0.30,<1", "httpx>=0.27,<1"]`, which needs `embed` at runtime.
  - Existing CLI and MCP behaviour is unchanged.
- **Repo rules.** mypy strict on `src/`, ruff clean, comments load-bearing only, a CHANGELOG dated line, and README docs as each task lands. The branch is `feat/continuous-local`.

## Review Focus

- **An assistant message with `tool_calls` and `content: null`.** It must be converted (the current `load_messages_json` drops it) and kept paired with its `tool` results (Task 1).
- **A client that resends a history whose early messages were edited or regenerated.** The output must equal a fresh compaction of the new history (Task 3).
- **A conversation that never grows past the tail.** The proxy must forward it unchanged, with no header injected (Task 2).
- **Streaming responses (`stream: true`).** Bytes are relayed unchanged in order, and the client sees SSE as if direct (Task 4).
- **A session long enough to hit decay.** The output must stay under `decay_at` once at least one merge is possible, and frozen chunks before the merged tier must be unchanged (Task 2).

---

### Task 1: Chat-API messages to Transcript, with a message map

**Files:**
- Create: `src/loosy_goose/openai_messages.py`
- Test: `tests/test_openai_messages.py`

**Interfaces:**
- Produces:
  - `to_transcript(messages: Sequence[Mapping[str, Any]]) -> tuple[Transcript, list[int]]`
    - Each non-system message becomes one `Turn`. The returned list maps turn index to message index.
    - Assistant `tool_calls` become `Block("tool_use", json.dumps(arguments-parsed-or-raw, sort_keys=True), {"name": fn_name})` after any text content.
    - A `tool` message becomes a `Block("tool_result", ...)`.
    - Content given as a list of parts uses the text parts.
    - System messages are excluded from the transcript.
  - `system_prefix(messages) -> int`: the count of leading system messages.

- [ ] **Step 1: Write failing tests.**
  - `test_tool_calls_with_null_content_become_tool_use`: the name is kept, and the args are parsed from a JSON string.
  - `test_tool_message_becomes_tool_result`.
  - `test_message_map_skips_system`.
  - `test_list_content_parts_joined`.
  - `test_unparseable_arguments_kept_raw`.
- [ ] **Step 2: Run** `uv run pytest tests/test_openai_messages.py -v`. Expected: FAIL (no module).
- [ ] **Step 3: Implement.** Reuse `transcript.split_fences` for text content.
- [ ] **Step 4: Run.** Expected: PASS, and mypy is clean.
- [ ] **Step 5: Commit**, with the message `"Continuous: OpenAI messages to Transcript with message map"`.

### Task 2: `FreezeCompactor` core (freeze, tail, decay, render)

**Files:**
- Create: `src/loosy_goose/continuous.py`
- Create: `src/loosy_goose/render.py`. It holds `render_segments(segments) -> str` and its block helper, ported from `experiments/exp_i_downstream.py:353-400` with identical output. The experiments keep their own copy, so their prompt hashes don't move.
- Test: `tests/test_continuous_compactor.py`

**Interfaces:**
- Consumes: Task 1's `to_transcript`, plus `pipeline.compress_transcript(t, PipelineOptions(keep=k)).segments`.
- Produces:
  - `CONTINUOUS_VERSION = 1`.
  - `@dataclass(frozen=True) class ContinuousConfig`, with fields:
    - `tail_tokens: int = 4000`
    - `chunk_tokens: int = 4000`
    - `keep: float = 0.6`
    - `window: int | None = None`
    - `start_at: float = 0.0`
    - `decay_m: int = 4`
    - `decay_keep: float = 0.7`
    - `decay_at: float = 0.9`
  - `@dataclass class Chunk`, with fields:
    - `span: tuple[int, int]` (message indices)
    - `source_sha: str` (sha256 of the canonical JSON of the source messages in the span)
    - `tier: int`
    - `segments: list[Segment]`
    - `text: str`
  - `class FreezeCompactor`, with:
    - `__init__(self, config: ContinuousConfig = ContinuousConfig(), compress: Callable[[Transcript, float], list[Segment]] | None = None, chunks: list[Chunk] | None = None)`. `compress` defaults to the shipped pipeline; tests pass a deterministic stub.
    - `compact(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]`
    - `chunks` (the property used by Task 3).
  - The gate: while `start_at > 0` and the full context is at most `start_at × window`, `compact` returns the input unchanged. Once a chunk exists, the gate stays open.
  - `window is None` disables both the gate and decay.

**Algorithm** (the plan pins this; the tests check it):

1. Compute the tail start, the earliest message index such that the messages after it total at least `tail_tokens`. Then move it earlier until it does not split an assistant `tool_calls` message from its `tool` replies.
2. From the end of the last chunk, while the un-frozen pre-tail messages total at least `chunk_tokens`, cut the next chunk at a message boundary. Compress it once at `keep`, and append it at tier 0.
3. When the context exceeds `decay_at × window` and tier 0 holds at least `decay_m` chunks, merge the oldest `decay_m` chunks of the lowest full tier into one chunk of the next tier, recompressed over their kept segments at `decay_keep`. Repeat until the context fits or no merge is possible. Output chunks are ordered oldest-first, highest tier first.
4. Build the output: the system prefix, then the header user message (only if chunks exist), then the un-frozen pre-tail messages verbatim, then the tail messages verbatim.

- [ ] **Step 1: Write failing tests.**
  - `test_short_conversation_unchanged`.
  - `test_frozen_chunks_never_change_as_conversation_grows`: compact at n, n+5 and n+10 messages, and the earlier chunk texts are identical.
  - `test_output_deterministic`.
  - `test_tail_verbatim_and_tool_pair_not_split`.
  - `test_header_and_system_first`.
  - `test_gate_passes_through_until_start_at`.
  - `test_decay_bounds_context_and_keeps_older_tiers`.
  - `test_render_matches_experiment_render`, run on a fixed segment list against `exi.render_segments`.
- [ ] **Step 2: Run** `uv run pytest tests/test_continuous_compactor.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `render.py` and `continuous.py`** following the algorithm above.
- [ ] **Step 4: Run** the tests, the full suite and `uv run mypy src`. Expected: PASS and clean.
- [ ] **Step 5: Commit**, with the message `"Continuous: FreezeCompactor (freeze, tool-safe tail, decay, gate)"`.

### Task 3: State persistence and history-change invalidation

**Files:**
- Modify: `src/loosy_goose/continuous.py`
- Test: `tests/test_continuous_compactor.py`

**Interfaces:**
- Produces:
  - `conversation_id(messages) -> str`: the sha256 hex of the canonical JSON of the system messages and the first user message, truncated to 32 hex characters.
  - `class StateStore(root: Path)`:
    - `load(cid: str, config: ContinuousConfig) -> list[Chunk]` returns an empty list on a missing file or on a version or config mismatch.
    - `save(cid: str, config: ContinuousConfig, chunks: list[Chunk]) -> None` writes JSON atomically (write to a temp file, then `os.replace`), with `CONTINUOUS_VERSION` and the config in the file.
  - `FreezeCompactor.compact` drops the first stored chunk whose `source_sha` no longer matches the incoming messages, and every chunk after it, before advancing.

- [ ] **Step 1: Write failing tests.**
  - `test_state_round_trip_resumes_identically`: a compactor restored from the store gives the same output as one that never stopped.
  - `test_edited_early_message_equals_fresh_run`.
  - `test_config_change_discards_state`.
  - `test_conversation_id_stable_across_appends`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Segments serialise through their dataclass fields.
- [ ] **Step 4: Run** the tests and the full suite. Expected: PASS.
- [ ] **Step 5: Commit**, with the message `"Continuous: per-conversation state store and invalidation"`.

### Task 4: Local chat-API proxy and CLI

**Files:**
- Create: `src/loosy_goose/proxy.py`
- Modify:
  - `src/loosy_goose/cli.py`: a `proxy` subcommand;
  - `pyproject.toml`: the `proxy` extra;
  - `CHANGELOG.md`;
  - `README.md`: a "Continuous compaction for local models" section.
- Test: `tests/test_proxy.py`

**Interfaces:**
- Consumes: Tasks 2 and 3.
- Produces:
  - `create_app(upstream: str, config: ContinuousConfig, store: StateStore, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI`. The `transport` parameter is for tests.
    - `POST /v1/chat/completions`: compact the body's `messages` (state keyed by `conversation_id`), forward to `{upstream}/v1/chat/completions` with the same headers minus `host` and `content-length`, and relay the status, headers and body unchanged. When `stream` is true, relay bytes with `StreamingResponse`.
    - Any other path under `/v1/` is forwarded unchanged.
  - **CLI.** `loosy-goose proxy --upstream URL [--port 8765] [--host 127.0.0.1] [--window N] [--start-at F] [--state-dir PATH]`. The CLI exits with an error if `--host` is not loopback.

- [ ] **Step 1: Write failing tests**, using httpx `MockTransport` as the upstream and FastAPI `TestClient`.
  - `test_forwards_compacted_messages`: the upstream sees the header message for a long conversation.
  - `test_short_conversation_forwarded_unchanged`.
  - `test_streaming_bytes_relayed_in_order`.
  - `test_authorization_passed_through_not_logged`, using caplog: no header value appears in the logs.
  - `test_non_loopback_host_refused`.
  - `test_other_v1_paths_passthrough`.
- [ ] **Step 2: Run** `uv run --extra proxy pytest tests/test_proxy.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** The proxy tests are smoke tests, per repo convention for thin glue.
- [ ] **Step 4: Run** the tests, the full suite, ruff and mypy. Expected: PASS and clean. Also add the `proxy` extra to the CI install line in `.github/workflows/ci.yml` if the tests need it there; otherwise mark them to skip when fastapi is missing.
- [ ] **Step 5: Commit**, with the message `"Continuous: OpenAI-compatible localhost proxy (loosy-goose proxy)"`.

### Task 5: End-to-end smoke against `mlx_lm.server` with Qwen3-4B

**Files:**
- Create: `tests/test_proxy_live.py`, marked `@pytest.mark.live` and skipped unless the `slot` and `proxy` extras are installed on arm64 macOS.
- Modify: `README.md`, adding the usage example with `mlx_lm.server`.

**Interfaces:**
- Consumes: Task 4's `loosy-goose proxy`, and `mlx_lm.server --model Qwen/Qwen3-4B-MLX-4bit`.

- [ ] **Step 1: Write the live test.**
  - It starts `mlx_lm.server` on a free port and `loosy-goose proxy --upstream` pointed at it, then runs a 12-turn scripted agent loop. Each user turn carries a 1.5k-token synthetic tool result, so freezing triggers.
  - It asserts every response is HTTP 200 with non-empty content.
  - It asserts that after turn 8, the state file holds at least 1 chunk.
  - It asserts the frozen chunk text is unchanged between turns 9 and 12.
- [ ] **Step 2: Run** `uv run --extra slot --extra proxy pytest -m live tests/test_proxy_live.py -v`. Expected: PASS. Never run it while the Qwen-8B replay holds the GPU: check `pgrep -f "run.py --set F --answerer"` first, and wait if that returns a match.
- [ ] **Step 3: Write the README usage block** with the two-command quick start, as it ran.
- [ ] **Step 4: Commit**, with the message `"Continuous: live smoke test against mlx_lm.server, README quick start"`.

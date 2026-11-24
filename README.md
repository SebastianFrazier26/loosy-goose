# Loosy-Goose

**Lossy, extractive compression for LLM agent sessions.** Loosy-Goose takes a long Claude Code
or Codex session (or any `[{role, content}]` message list), drops the parts that matter least,
and hands back real, unedited pieces of the original at a token budget you pick. It runs on
CPU, makes no model calls, and is deterministic: the same transcript and budget always give the
same output, and every kept line can be traced back to its source.

[Interactive demo](https://sebastianfrazier26.github.io/loosy-goose/) ·
[Full writeup](docs/WRITEUP.md) · [Design](docs/DESIGN.md) · [Results](docs/results/) ·
[Changelog](CHANGELOG.md)

## Why it exists

Long agent sessions run out of context. The usual fix, `/compact`, asks a frontier model to
summarise the history. That works well, but it costs a model call, sends the whole transcript
to that model, paraphrases (so it can drop or invent details), and gives a different answer each
time.

Loosy-Goose tries the other route: treat the transcript like an image codec treats pixels.
Split it into units, find the directions in embedding space that carry the session's meaning,
and drop the units that contribute least until the budget is met. Nothing is rewritten, so
nothing is made up.

## The honest result

Measured on public agent transcripts by asking a model short factual questions ("probes")
about the compressed part of each session, graded against the full-context answer.

**When a frontier model is available, use its summary.** On the largest held-out set (set D,
20 SWE-Gym transcripts, Haiku 5.5 answering), a Sonnet summary at the same token budget keeps
more:

| keep ratio | Loosy-Goose | Sonnet summary | local Qwen3-8B summary |
| --- | --- | --- | --- |
| 0.50 (light) | 0.863 | 0.970 | 0.547 |
| 0.33 (medium) | 0.817 | 0.936 | 0.556 |
| 0.25 (heavy) | 0.794 | 0.935 | 0.519 |

The summary's lead (+0.11 to +0.14) is significant at every level. Source:
[exp_slot_downstream_d.md](docs/results/exp_slot_downstream_d.md).

**When no frontier model is in the loop, Loosy-Goose wins clearly.** Against summaries written
by local Qwen3 models (1.7B, 4B, 8B) at the same budget it is ahead by 0.26 to 0.34
(p ≤ 0.004). And run continuously, compressing each 4k-token chunk once as it ages out of a
verbatim recent tail, it matched full context with only open-weight models in the loop (Qwen3-8B
answering, Mistral-Nemo-12B writing and grading probes, 25 transcripts):

| policy | accuracy | context tokens per turn |
| --- | --- | --- |
| full context (overflowed Qwen's 32k window on 15 of 75 checkpoints) | 0.717 | 18.8k |
| continuous Loosy-Goose (freeze) | 0.710 | 12.4k |
| threshold Loosy-Goose (`/compact`-style) | 0.677 | 12.5k |
| Qwen summarising its own history | 0.595 | 11.8k |

Freeze vs full: −0.004 (p = 0.65). Freeze vs Qwen self-summary: +0.113 (p < 0.001). Source:
[exp_continuous_qwen.md](docs/results/exp_continuous_qwen.md).

**What did not work.** Every attempt to close the gap to a frontier summary without one came
out level or worse:

- *Slot compression* (2026-10-22): rewriting prose into short slot rows lost 0.12 at light
  (p = 0.008); a gentle rewrite and a small-model rewrite were neutral. Prose is only 12–19% of
  tokens in these sessions, so even a 41% cut frees a few percent of the budget, while the lost
  facts sit in dropped code and tool output.
- *Token thinning* (2026-10-21): deleting predictable tokens by small-LM surprisal cost answers
  at every model size; a line-keep rule that looked +0.07 better did not replicate on fresh data.
- *Hybrid* (2026-10-23): a summary plus Loosy-Goose filling the rest of the budget was level with
  the summary alone.
- *Decaying freeze* (2026-11-16): recompressing older chunks for a recency bias changed nothing
  (+0.003, p = 0.50).

The caveats are real: 10–25 transcripts per set, one grader family per run, and probes reward
fact retention rather than, say, plan continuity. The [writeup](docs/WRITEUP.md#performance)
has every table with confidence intervals and the reasoning behind each turn.

### So when should you use it?

- Compaction that has to stay **local, offline or air-gapped**: CPU only, no API call.
- **Private sessions**: the transcript never leaves the machine and no third model sees it.
- **Auditable context**: every kept line is verbatim, so you can show exactly what a model saw.
- **Free pre-compaction**: the lossless pre-pass alone removes about a quarter of tokens (24.8%
  across a 1.5M-token corpus), and selection takes 0.24 s per 100k tokens.
- **Long local-model sessions**, through the continuous proxy below.

It is the wrong tool when a frontier model can write the summary for you.

## Install

Python 3.12 (pinned in `.python-version`). With [uv](https://docs.astral.sh/uv/) 0.11.3 or newer,
from the v0.1.0 release:

```
uv tool install --torch-backend cpu "loosy-goose[embed,mcp] @ git+https://github.com/SebastianFrazier26/loosy-goose@v0.1.0"
```

`--torch-backend cpu` keeps Linux from pulling ~3 GB of CUDA wheels; on macOS it does nothing.
The continuous proxy is newer than v0.1.0, so for it install from a clone:

```
git clone https://github.com/SebastianFrazier26/loosy-goose
cd loosy-goose
uv sync --extra embed                                  # CLI only
uv sync --dev --extra embed --extra mcp --extra proxy  # everything, plus tests
```

The first run downloads the selection embedding model, `BAAI/bge-small-en-v1.5`.

## Use

### Command line

```
loosy-goose compress session.jsonl                    # medium: keep 0.33 of tokens (~3x)
loosy-goose compress session.jsonl --level light      # 0.50 (~2x); heavy = 0.25 (~4x)
loosy-goose compress session.jsonl --keep 0.42        # any ratio in [0.10, 0.80]
loosy-goose compress session.jsonl --target-tokens 20000 --output kept.jsonl
```

(From a clone, prefix with `uv run`.) Input is a Claude Code `.jsonl` session log or a `.json`
message list. Output is JSONL: a header line with the rate, budget and token counts, then one
line per kept turn, in original order. Every flag is in the
[writeup](docs/WRITEUP.md#the-cli).

### MCP server: `/loosy` in Claude Code and Codex

`loosy-goose-mcp` is an MCP server with two tools:

- **`handoff_session`** forks your live Claude Code or Codex session into a compressed copy
  (a marker message, the kept turns, and a verbatim recent tail of at least 4,000 tokens) and
  returns the command to resume it. Use it in place of the built-in `/compact`. The original
  session file is never modified.
- **`compress`** returns the compressed turns for any message list or transcript path, and
  writes nothing.

Register it and install the command:

```
claude mcp add -s user loosy-goose -- loosy-goose-mcp       # Claude Code
codex mcp add loosy-goose -- loosy-goose-mcp                # Codex
cp integrations/claude/loosy.md ~/.claude/commands/loosy.md                          # /loosy
mkdir -p ~/.agents/skills && cp -R integrations/codex/loosy ~/.agents/skills/loosy   # $loosy
```

Then, in a long session, run `/loosy` (or `$loosy`), optionally with a level
(`/loosy heavy`), exit, and run the printed resume command. Timeouts, headless Codex settings and
options are in [integrations/README.md](integrations/README.md). Both CLIs' session formats are
undocumented; the adapters were tested against Claude Code 2.1.296 and codex-cli 0.158.0.

### Continuous proxy for local models

`loosy-goose proxy` sits on localhost between an agent and a local model server that speaks
`/v1/chat/completions` (mlx_lm.server, Ollama, LM Studio, llama.cpp's server). It compacts every
chat request in freeze mode: each 4k-token chunk is compressed once as it leaves the verbatim
tail and never touched again, so the prompt prefix stays cacheable.

```
uv run loosy-goose proxy --upstream http://127.0.0.1:8080 --window 32768
# then point the agent's base URL at http://127.0.0.1:8765/v1
```

It binds loopback only, refuses non-local `Host` and cross-origin requests, never logs message
content, and marks compacted requests with a header line, so nothing is rewritten silently.
Flags, the state directory, the optional `X-Loosy-Goose-Session` per-conversation header and
the known limitations are in the [writeup](docs/WRITEUP.md#continuous-compaction-for-local-models).

## How it works

1. **Segment** the transcript: prose paragraphs, code blocks and tool calls kept whole, tool
   results cut into line chunks.
2. **Supersede** losslessly: a file read or edit replaced later by a newer one, and exact
   duplicates, are dropped. This alone removes about a quarter of the tokens.
3. **Transform**: embed every segment and take the SVD of the session's own embedding matrix.
4. **Select**: score each segment by its leverage on the top singular directions, discounted by
   how much of it the kept set already covers, per token, and fill the budget greedily.
5. **Emit** the kept segments in original order, unchanged.

The idea started from Shin, Madotto & Fung (2018),
[*Interpreting Word Embeddings with Eigenvector Analysis*](https://openreview.net/forum?id=rJfJiR5ooX).
[DESIGN.md](docs/DESIGN.md) has the algorithm and the per-kind quality table.

## Development

```
uv sync --dev --extra embed --extra mcp --extra proxy
uv run pytest                                      # tests
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

Experiments need the `slot` extra too (mlx-lm and spaCy) and live in `experiments/`; the
contributor and agent guide is [CLAUDE.md](CLAUDE.md). The demo site in `site/` is a Vite +
TypeScript page deployed to GitHub Pages from `main`.

## Data and privacy

Every downstream experiment uses public data only: `trace-commons/agent-traces` (CC-BY-4.0)
and `SWE-Gym/OpenHands-SFT-Trajectories` (MIT), fetched at pinned revisions, never committed.
Five of the author's own sessions informed early token counts only and are never committed or
quoted.

## License

MIT. See [LICENSE](LICENSE). The embedding models it downloads carry their own licenses.

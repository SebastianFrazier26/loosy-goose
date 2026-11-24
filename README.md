# Loosy-Goose

Extractive, lossy compression for LLM conversation context. Loosy-Goose takes a long agent
session (Claude Code, Codex, or any `[{role, content}]` message list), throws away the parts
that matter least, and hands back real, unedited pieces of the original at a token budget you
pick. It runs locally on CPU, makes no model calls, and is deterministic: the same transcript
and budget always give the same output, and every kept line can be traced to the source.

**The honest bottom line.** On downstream question answering, a frontier-model summary beats
Loosy-Goose at the same budget. On the largest held-out set a Sonnet-written summary keeps
0.94–0.97 probe accuracy where Loosy-Goose keeps 0.79–0.86, a gap of 0.11–0.14 that is
significant at every level (n = 20); on a smaller set of 10 the gap was not significant. Every attempt to close that gap without a frontier
model (token thinning, line-keep for code dumps, slot-style rewriting of prose, a small local
rewriter) came out level with plain Loosy-Goose or worse. Where it **does** win clearly is when
no frontier model is in the loop: against summaries written by local Qwen3 models (1.7B, 4B,
8B) at the same budget, Loosy-Goose is ahead by 0.26–0.34 (p ≤ 0.004). So it is a tool for
compaction that must stay local, free, deterministic or auditable, and not a replacement for a
good model-written summary when one is available. The numbers, and how the project got here,
are below.

**Interactive demo:** <https://sebastianfrazier26.github.io/loosy-goose/> (pending: goes live once
GitHub Pages is enabled for the repo; see [Demo site](#demo-site)).

Contents: [Install and use](#install-and-use) · [How it works](#how-it-works) ·
[Performance](#performance) · [Course and evolution](#course-and-evolution) ·
[Methodology and lessons](#methodology-and-lessons-learned) · [Use cases](#use-cases) ·
[Future research](#future-research) · [Repository layout](#repository-layout) ·
[License](#license)

## Install and use

Python 3.12 is required (it is pinned by `.python-version`; 3.13+ had no torch wheels when this
was set up). Everything runs on CPU.

### From a release

Each `v*` tag publishes a GitHub Release with the wheel, the sdist and the integration files.
With uv 0.11.3 or newer:

```
uv tool install --torch-backend cpu "loosy-goose[embed,mcp] @ git+https://github.com/SebastianFrazier26/loosy-goose@v0.1.0"
```

`--torch-backend cpu` keeps Linux from pulling the ~3 GB of CUDA wheels; on macOS it does
nothing. A plain `pip install` of the package still gets PyPI torch (CUDA on Linux).

### From a clone

```
git clone https://github.com/SebastianFrazier26/loosy-goose
cd loosy-goose
uv sync --extra embed                 # CLI only
uv sync --dev --extra embed --extra mcp   # CLI, MCP server, tests and experiments
```

On Linux the lockfile resolves torch from the PyTorch CPU index, so a uv install from the
clone never pulls CUDA wheels either.

### The CLI

```
uv run loosy-goose compress path/to/transcript.jsonl                  # level medium, keep 0.33 (~3x)
uv run loosy-goose compress path/to/transcript.jsonl --level light    # 0.50 (~2x); heavy = 0.25 (~4x)
uv run loosy-goose compress path/to/transcript.jsonl --keep 0.42      # any ratio in [0.10, 0.80]
uv run loosy-goose compress path/to/transcript.jsonl --target-tokens 20000
```

The first run downloads the selection embedding model, `BAAI/bge-small-en-v1.5`, from Hugging
Face. The experiments use a second model, `sentence-transformers/all-MiniLM-L6-v2`, to score
coverage; the two are kept apart on purpose so the selector is never graded by its own model
(see `src/loosy_goose/embed.py`).

**Input.** Two formats, picked by file suffix:

- `.jsonl`: a Claude Code session log. Unknown record and block types are counted and skipped,
  never fatal.
- `.json`: a generic message list, `[{"role": ..., "content": ...}]`, or an object with a
  `messages` key holding one. Roles `user`, `assistant`, `system` and `tool` are read; `content`
  may be a string or a list of content blocks.

**Output.** JSONL. The first line is a header object under the key `loosy_goose`: the rate used
and where it came from, the flags, the token budget, input and output token counts,
`atom_recall_final` (information only; nothing reads it back), the version stamps, and the path
table when `--paths` is on. Every line after that is one kept turn,
`{"index", "role", "blocks": [{"kind", "text", "meta"}]}`, in original order.

**Flags.**

| Flag | Meaning |
| --- | --- |
| `--level light\|medium\|heavy` | preset rate: keep 0.50 / 0.33 / 0.25 of input tokens (~2x / ~3x / ~4x). Default `medium` |
| `--keep R` | keep ratio in [0.10, 0.80], in 0.01 steps (finer values are rounded) |
| `--target-tokens N` | output token budget; one that implies a ratio outside [0.10, 0.80] is clamped, with a warning |
| `--no-novelty` | select by plain leverage (`supersede+leverage`) instead of novelty-discounted leverage per token |
| `--band` | AST-depth banding of code and code-shaped tool arguments before scoring. Off by default; selection falls back to plain leverage |
| `--paths` | name each repeated path once in a leading table and replace mentions with `[P<n>]` markers. Off by default; selection falls back to plain leverage |
| `--min-mentions N` | with `--paths`, tabulate paths mentioned at least N times (default 2; 1 = every path) |
| `--output PATH` | write here instead of stdout (`-`, the default, is stdout) |

`--level`, `--keep` and `--target-tokens` are mutually exclusive. No preset turns on `--band` or
`--paths`.

### Handoff from Claude Code or Codex

`loosy-goose-mcp` is an MCP server. Its `handoff_session` tool forks your live Claude Code or
Codex session into a compressed copy: the kept turns, a verbatim recent tail of at least 4,000
tokens, and a marker message saying what was compressed. You resume the fork in place of the
built-in `/compact`. The original session file is never modified. The same server's `compress`
tool returns compressed turns for any transcript and writes nothing.

In-session commands: `/loosy [light|medium|heavy]` in Claude Code, `$loosy` in Codex.
Registration, timeouts and command install are in
[integrations/README.md](integrations/README.md). Both CLIs' session formats are undocumented;
the adapters were tested against Claude Code 2.1.296 and codex-cli 0.158.0, and a newer CLI
still forks, with a warning.

### Continuous compaction for local models

`loosy-goose proxy` is a localhost proxy that sits between an agent and a local model server
that speaks the chat API format local servers speak (mlx_lm.server, Ollama, LM Studio,
llama.cpp's server). Every `POST /v1/chat/completions` is compacted continuously, in freeze
mode (see [Continuous compaction](#continuous-compaction-2026-11-16) for why this mode): each
4k-token chunk is compressed once as it ages out of a verbatim recent tail of at least 4k tokens,
and never touched again, so the prompt prefix stays cacheable. Every other `/v1/` path is
forwarded unchanged.

```
uv sync --extra embed --extra proxy
uv run loosy-goose proxy --upstream http://127.0.0.1:8080 --window 32768
# then point the agent's base URL at http://127.0.0.1:8765/v1
```

Quick start with `mlx_lm.server` on Apple silicon (the commands the live test runs):

```
uv sync --extra embed --extra proxy --extra slot
uv run mlx_lm.server --model Qwen/Qwen3-4B-MLX-4bit --port 8080 \
  --chat-template-args '{"enable_thinking": false}' --prompt-cache-bytes 1073741824
uv run loosy-goose proxy --upstream http://127.0.0.1:8080   # second terminal
```

Cap `--prompt-cache-bytes`: the server's default prompt cache is unbounded, and on a 16 GB Mac
it ran out of Metal memory around 10k-token prompts. `uv run pytest -m live -s tests/test_proxy_live.py`
runs a 12-turn loop through both (opt-in, needs the GPU).

| Flag | Meaning |
| --- | --- |
| `--upstream URL` | the local server's base URL, without `/v1` (required); a URL ending in `/v1` is rejected, not rewritten |
| `--port N` | listen port, default 8765 |
| `--host H` | bind address, default `127.0.0.1`; anything that is not loopback is refused |
| `--window N` | the model's context window in tokens; enables decay of older chunks near it |
| `--start-at F` | pass requests through untouched until they exceed F × `--window` (needs `--window`); default 0, freeze from the first turn, because a gated start measured no different (−0.001) in the [open-weight replay](#open-weight-replay-of-continuous-compaction-2026-11-17-to-11-22) |
| `--state-dir PATH` | per-conversation state, default `~/.cache/loosy-goose/continuous/` |

What it does and does not do:

- Compacted requests carry a header line saying the earlier part was compressed extractively,
  folded into the following user message when there is one. Nothing is rewritten silently.
- A conversation is identified by its system messages and first user message, scoped by the
  `X-Loosy-Goose-Session` request header when the client sends one. Its frozen chunks
  are stored as one JSON file in the state dir, which holds conversation text: the dir is
  set to mode 0700 (an existing one is tightened; POSIX only, Windows relies on the user profile's ACLs), and files untouched for 7 days are deleted when the proxy starts.
  Editing earlier history discards the chunks from the edit on.
  Two conversations that start with the same system messages and the same first user message
  are kept apart only by that header: send a distinct value per conversation (1 to 128 visible
  ASCII characters; a malformed or repeated header gets 400). Without it they share one state
  file. The output stays correct either way, because a chunk is reused only when its source
  messages match, but sharing conversations discard each other's chunks and recompress, so
  neither keeps a stable prefix while both are active.
- Requests for the same conversation are compacted one at a time; compaction runs off the
  event loop.
- `Authorization` and every other end-to-end request header are passed to the upstream; `Host`,
  `Content-Length`, `X-Loosy-Goose-Session` and the hop-by-hop headers are not. Message content and header values are
  never logged, and the access log is off, since paths and query strings can carry keys.
- A request whose body is not a chat request, or whose conversation needs no compaction yet, is
  forwarded byte for byte.
- Responses, streamed or not, are relayed as they arrive, with the upstream's status, headers
  and bytes.
- Only `Host` values `127.0.0.1`, `localhost` and `[::1]` (any port) are served; anything else
  gets 403, so a web page that rebinds its own domain to 127.0.0.1 cannot reach the proxy.
- Passthrough paths that would leave `/v1/` once decoded (an encoded `..`, `?`, `#` or `\`) get
  404. `--host localhost` binds `127.0.0.1`.

## How it works

The design borrows the shape of an image codec. A transcript is split into units, a transform
finds the directions that carry its meaning, and units with little weight on those directions
are dropped until the budget is met.

| Codec stage | Loosy-Goose |
| --- | --- |
| Block split | Segment the transcript: whole prose paragraphs (12k-character cap), code blocks and tool calls kept whole, tool results cut into line chunks |
| Lossless pre-pass | Supersession: a file read or edit later superseded by a newer read/edit of the same file, and exact duplicates, are elided. Lossless with respect to the session's final state |
| Transform | SVD of the session's own segment-embedding matrix. A session is small, so the exact KLT is affordable |
| Coefficient truncation | Score each segment by its leverage on the top singular directions; fill the budget by leverage times novelty (share of the segment's plain words the kept set lacks, to the power 0.5), per token |
| Entropy coding | None needed: the output is text, and the reading model's tokenizer does the rest |
| Chroma subsampling | Opt-in `--band`: code bodies are cut to their outline by AST depth instead of being dropped whole |

The output is **extractive**: real segments of the original, kept or dropped (or banded, with
`--band`). Nothing is paraphrased, so nothing can be invented.

The theoretical seed is Shin, Madotto & Fung (2018),
[*Interpreting Word Embeddings with Eigenvector Analysis*](https://openreview.net/forum?id=rJfJiR5ooX)
([PDF](https://web.archive.org/web/20250405190331if_/https://openreview.net/pdf?id=rJfJiR5ooX) ·
[code](https://github.com/HLTCHKUST/eigenvector-analysis)): the eigenvectors of a PPMI-SVD
embedding form coherent word groups, and the sparse ones (high inverse participation ratio)
carry the narrow, topical meaning. That is also where the JPEG analogy inverts: JPEG's energy
sits in the top coefficients, while in text embeddings the top directions are bias and meaning
lives in the sparse tail. See [docs/DESIGN.md § Origin](docs/DESIGN.md#origin).

## Performance

Two kinds of evidence exist, and they disagree in an instructive way (see
[lessons](#methodology-and-lessons-learned)):

1. **Offline proxies** (Phases 1–2): *atom recall* (share of identifiers, paths, numbers and URLs
   in the final state that survive) and *semantic coverage* (embedding cosine to the original),
   at matched achieved compression, on 18 transcripts (13 public, 5 private sessions that
   contribute counts only).
2. **Downstream QA** (Experiment I and later): a model answers short factual probes about the
   compressed part of a session, graded correct / partial / wrong against a reference.

### Downstream probe QA

Setup common to all rows: public agentic transcripts (trace-commons and SWE-Gym/OpenHands); each
is cut late, before a real assistant action; the last ≥4,000 tokens stay verbatim as `/loosy`
does, and only the earlier stretch is compressed. A probe is kept only if the answering model gets
it right from the full context. Cells are means over transcripts with 95% bootstrap CIs; paired
deltas use a sign test. "Summary" is a Sonnet 5.5 summary of the same stretch, a stand-in for
built-in `/compact`.

**Held-out set D** (20 SWE-Gym transcripts; probes by gpt-6-sol, Haiku 5.5 answers, Sonnet 5.5
grades; summaries and local summaries targeted at loosy's actual token count per level):

| condition | light (keep 0.50) | medium (0.33) | heavy (0.25) |
| --- | --- | --- | --- |
| full context | 1.000 | 1.000 | 1.000 |
| **loosy (shipped)** | **0.863** [0.79, 0.93] | **0.817** [0.76, 0.88] | **0.794** [0.73, 0.85] |
| Sonnet summary | 0.970 [0.95, 0.99] | 0.936 [0.90, 0.96] | 0.935 [0.89, 0.97] |
| hybrid: summary at 70% of budget + loosy fill | 0.977 [0.96, 0.99] | 0.950 [0.92, 0.98] | 0.936 [0.90, 0.97] |
| local Qwen3-1.7B summary (bf16) | 0.522 [0.44, 0.61] | 0.507 [0.43, 0.59] | 0.466 [0.41, 0.52] |
| local Qwen3-4B summary (4-bit) | 0.535 [0.46, 0.61] | 0.529 [0.46, 0.59] | 0.507 [0.43, 0.59] |
| local Qwen3-8B summary (4-bit) | 0.547 [0.48, 0.62] | 0.556 [0.48, 0.64] | 0.519 [0.45, 0.59] |

Paired, on set D: summary − loosy is +0.106 (p = 0.039, n = 19), +0.118 (p = 0.021), +0.141
(p = 0.021). Loosy − local 8B is +0.316, +0.261 (p = 0.004), +0.275 (p < 0.001); against 1.7B
and 4B it is +0.29 to +0.34, all p < 0.001. The hybrid is level with the summary (+0.004 to
+0.015, n.s.). Sources:
[exp_slot_downstream_d.md](docs/results/exp_slot_downstream_d.md)
(every set-D arm: hybrids, gentle slot, line-keep, local size curve).

**Experiment I** (20 transcripts: 5 trace-commons, 15 SWE-Gym; probes written and filtered by
Sonnet, Sonnet grades; four answering models). Mean over the four answerers, with the Haiku row
and its CI for scale:

| condition | earlier-stretch tokens, share of full | 4-answerer mean | Haiku 5.5 |
| --- | --- | --- | --- |
| full | 1.000 | 0.96 | 0.96 [0.94, 0.98] |
| loosy light / medium / heavy | 0.495 / 0.333 / 0.254 | 0.81 / 0.78 / 0.75 | 0.81 / 0.79 / 0.76 |
| summary light / medium / heavy | 0.345 / 0.257 / 0.202 | 0.94 / 0.93 / 0.92 | 0.94 / 0.92 / 0.92 |

The summaries here came out *shorter* than their loosy-matched budgets and still won.
Next-step prediction (does the model choose the same next action as the real agent) was also
measured and is noise: the CIs are 0.3–0.5 wide and show no consistent ordering across
answerers. Source:
[exp_i_downstream.txt](docs/results/exp_i_downstream.txt).

**Set C** (10 SWE-Gym transcripts, same protocol as set D) is the one place the summary did not
clearly win: loosy 0.974 / 0.909 / 0.888 against summary 0.946 / 0.951 / 0.956, no paired
difference significant at n = 10.
[exp_slot_downstream.md](docs/results/exp_slot_downstream.md).
Pooled over C + D (n = 28–30), summary − loosy is +0.06 at light (p = 0.19), +0.09 at medium
(p = 0.011) and +0.12 at heavy (p = 0.007)
([exp_probe_types.md](docs/results/exp_probe_types.md)).

**Caveats.** n is 10–20 transcripts per set. One grader (Sonnet) for every row. In Experiment I
Sonnet wrote both the probes and the summaries; sets C and D fixed that by writing probes with
gpt-6-sol. Probes are short factual questions, which favour fact retention over, for example,
plan continuity. The local-summary curve mixes precision (1.7B in bf16, 4B and 8B as 4-bit MLX
builds), and the local summaries undershot their budgets badly (37–67% of target) and sometimes
looped, so they are a lower bound on what a tuned local summariser could do.

### Offline proxies: selection quality and compression

Final-state atom recall and semantic coverage of the shipped selector at matched achieved rate,
18 transcripts. A cell marked n/18 averages only the transcripts that reach that rate.

| rate | atom recall (default) | coverage (default) | `--no-novelty` recall / coverage |
| --- | --- | --- | --- |
| 2x | 0.829 (15/18) | 0.921 | 0.837 / 0.895 (16/18) |
| 3x | 0.655 (17/18) | 0.887 | 0.656 / 0.821 (17/18) |
| 4x | 0.577 | 0.872 | 0.568 / 0.796 |

Cost: 0.24 s per 100k tokens for selection.
[exp_g_summary.txt](docs/results/exp_g_summary.txt), [exp_d_summary.txt](docs/results/exp_d_summary.txt).

**Lossless supersession** removes **24.8%** of tokens across the 1.50M-token corpus (373,635 of
1,503,755; 14.3 points from superseded edits, 5.0 from superseded reads, 5.5 from exact
duplicates). Per transcript it ranges from 0% to 70%, and how much it removes is the best single
predictor of how far a transcript compresses (Spearman +0.897 against the knee, in the
`ATOMS_VERSION` 4 analysis). Phase 1 measured 25.9% under the older segmenter.
[exp_c_summary.txt](docs/results/exp_c_summary.txt), C2.

**Compression ratios** are whatever you ask for: the level sets the keep ratio of the
compressible stretch (0.50 / 0.33 / 0.25, i.e. ~2x / 3x / 4x on it), and the achieved ratio lands
on target (Experiment I: 0.495 / 0.333 / 0.254). In a `/loosy` handoff the verbatim tail is not
compressed, so the whole-session ratio is lower; in Experiment I the tail averaged 5,054 tokens
against a 20,236-token earlier stretch.

## Course and evolution

### Timeline

| date | phase | decision | yielded |
| --- | --- | --- | --- |
| 2026-09-08 | Origin, Phase 1 | Text image compression: transform, quantize, drop. Hybrid core locked: PPMI-SVD topics + embedding-SVD leverage. Extractive only | Scaffold, metrics, baselines |
| 2026-09-09 | Phase 1 results | Binary "protected" flag replaced by a per-kind quality table | Supersession 25.9% lossless; banding 1.33x; PPMI topics unstable (0.507 vs 0.70 bar), demoted to labeller |
| 2026-09-10 | Phase 2 | Two distortion metrics are never collapsed into one; defined pointers (`[P<n>]`) are admissible | Version stamps per transform after a staleness trap |
| 2026-09-15 – 10-01 | Phase 2 grids D–H | Re-baselines after each atom-extractor audit; prose shrink, depth sweep and topics cut on nulls | Final grid, 0 failed jobs |
| 2026-10-02 | Default | Per-token novelty at α 0.5 becomes the default selector | +0.066 / +0.076 coverage at 3x / 4x, recall level |
| 2026-10-12 | Close Phase 2 | Remaining Phase 2 extras deferred past v1 (user) | CI green on main again |
| 2026-10-13/14 | v0.1.0 | MCP server + `/loosy` / `$loosy` fork-and-resume (user's design calls) | Release v0.1.0, tagged 2026-10-14 |
| 2026-10-14/15 | Experiment I | First downstream evaluation (design approved by user) | Summary wins by 0.13–0.17; loss is drop loss |
| 2026-10-21/22 | Token-thinning spike | Attack drop loss by thinning inside segments | SLM surprisal deletion fails; line-keep +0.07 at light |
| 2026-10-22 | Slot compression | User's original rewrite idea tested on fresh sets C and D | Aggressive rewrite hurts; gentle and SLM neutral; line-keep not confirmed |
| 2026-10-23 | Falsification | Three pre-registered tests before retiring the idea (user) | Niche found: no-frontier-model compaction |
| 2026-11-15/16 | Local size curve | Qwen3 1.7B → 4B → 8B local summaries | Gap to loosy barely narrows (0.26–0.32 at 8B) |
| 2026-11-16 | Continuous compaction | Run loosy after every turn instead of at a threshold; user picked recompute vs freeze, a replay pilot, no full-context probe filter, then a decaying (recency-biased) arm | Pilot: freeze within 0.03 of `/compact`-style at ~20% fewer uncached tokens; recompute loses; per-chunk budget ablation negative. Scaled run (set F): threshold summary dominates on accuracy and cache cost; decay is null. No frontier model: continuous loosy beats threshold loosy by +0.04 (p ≈ 0.02) |
| 2026-11-17/22 | Open-weight replay | Set F with Qwen3-8B answering and Mistral-Nemo grading, no closed model (user) | Freeze matches full context (−0.004) and beats Qwen self-summary by +0.11 |

### Origin: compress text the way JPEG compresses pixels

The project started from the user's idea: compact a conversation the way an image codec compacts
an image, by mapping word groups into a vector space and reducing them mathematically, with a
light local model rather than a frontier one. The first design took the codec *architecture*
literally (segment, transform, quantize, truncate) and took the transform from Shin et al.'s
eigenvector analysis.

What was fixed at the start, on 2026-09-08, and has held since: the output is text, not latent
vectors (any model must be able to read it); v1 is extractive, with no LLM in the loop; and
distortion is measured by atom recall plus semantic coverage, never by reconstruction error. A
reconstruction decoder returns text of the same length, so it would not compress.

### Phase 1 (2026-09-08 to 09-09): what the eigen-space can and cannot do

Three tracks ran on the same budget contract (every method gets exactly the same token budget,
or results are not comparable):

- **PPMI-SVD topics** (Experiment A) were meant to be the interpretable basis. At the scale of one
  conversation they are not stable: aligned cosine 0.507 against a pre-registered 0.70 bar. They
  were demoted to a labeller, and embedding-SVD leverage (Experiment B) became the selector.
- **Code channel** (Experiment C): AST-depth banding beat dropping whole code segments at equal
  budget, but bought only 1.33x on its own. It is a quantizer, not a compressor.
- **Supersession**, which uses no eigenvectors at all, removed 25.9% of tokens losslessly. It
  was the clearest win of the phase and still is.

The phase also replaced a binary "protected" flag with a per-kind quality table, because tool
calls and tool results are 51–91% of tokens in real agent sessions, and protecting them capped
compression near 1.3x. And it found that spectral selection and TF-IDF **split** the two
metrics: spectral wins atom recall at every rate, TF-IDF wins coverage at every rate. That split
was recorded as two statistics, not resolved into a winner.

### Phase 2 (2026-09-09 to 10-02): rate-distortion curves and a measured default

Phase 2 put every method on one rate-distortion curve per transcript, swept over 12 rates. Most
of its work was making the measurement trustworthy: three atom-extractor audits (escape-sequence
junk, Windows paths, attribute access read as filenames), each followed by a full re-baseline; a
fix to an analysis that had been silently averaging variant records into the baseline; and a
version stamp for every transform downstream of segmentation, after a rewritten transform left
stored records that looked valid but described code that no longer existed.

Arms tried against `supersede+leverage` and rejected, all on the final grid:

- **Compression-based selection** (Experiment F, zlib/lzma conditional greedy): buys coverage,
  loses recall at every rate, at up to ~107x the cost.
- **Raw-gain word novelty** (Experiment G, α 0.5 / 1 / 2): buys recall, loses coverage.
- **Per-kind scoring** (Experiment H, four arms): loses recall at every rate.
- **`--band`**: about one recall point at 3x–4x for ~47x the time; stays opt-in.
- **`--paths`**: gains guaranteed recall but costs 4–9 coverage points; stays opt-in.
- **Prose shrinking, `drop_top`, tool-depth sweep, `topics`**: nulls, removed from the grid.

The one arm ahead on both metrics was **novelty ranked per token at α 0.5**, which became the
default on 2026-10-02: against the old default, recall −0.011 / −0.001 / +0.009 and coverage
+0.026 / +0.066 / +0.076 at 2x / 3x / 4x, and cheaper. On 2026-10-12 the user closed Phase 2,
deferring novelty stacked on `--band`/`--paths` and a predictor re-run past v1. Full decision
log with the rejected option for each call: [docs/PHASE2.md](docs/PHASE2.md#decision-log).

### Shipping v0.1.0 (2026-10-13 to 10-14): fork-and-resume handoff

The MCP server and the `/loosy` / `$loosy` commands were designed with the user, who made the
product calls ([spec](docs/superpowers/specs/2026-10-13-mcp-handoff-design.md)):

- **Fork and resume**, accepting that both CLIs' session files are undocumented; rejected was a
  fresh session reading a handoff file, and injecting after built-in compaction.
- **Never overwrite** the original session; a failure at any step leaves it untouched.
- **Protect the recent tail** (from the last user turn, at least 4,000 tokens) verbatim.
- **A marker** as the fork's first message, naming the source session and the kept share.
- **Test both CLIs end to end**, including opt-in live drift tests against the real binaries.

The SDK (the official `mcp` 2.x) was settled with the user in the implementation plan. One
fix found in testing: Claude Code stores parallel tool calls on a side branch, so they are
spliced back into one linear chain. v0.1.0 was tagged on 2026-10-14.

### Experiment I (2026-10-14 to 10-15): the first downstream test

Until now every number was a proxy. Experiment I asked whether models actually answer as well
after compression, against a Sonnet summary at the same budget. They do not: the summary won on
every answering model ([table above](#downstream-probe-qa)), and at fewer tokens. The run took
11.4 hours, about 2,300 calls, and hit the account's monthly spend limit.

The diagnosis was clear: loosy's misses are **drop loss**. When the segment holding the answer
was kept, accuracy was about 0.93, the same as the summary. The losses sat in large code and file
dumps (3–3.5k tokens each, kept at rates of 0.26 at light down to 0.07 at heavy) and in short
segments dropped whole (about 40% of the loss).

### Token-thinning spike (2026-10-21 to 10-22): shrink inside segments

If whole segments are the problem, thin them instead, the "quantize" step of the codec. The spike
([code](https://github.com/SebastianFrazier26/loosy-goose/tree/spike/token-thinning/experiments/spike_thin),
summarised in the
[handoff note](docs/superpowers/handoffs/2026-10-22-slm-rewrite-handoff.md)):

- **Deleting predictable tokens by small-LM surprisal** (pythia-31m to Qwen3-1.7B) failed.
  Thinning prose cost answers; code skeletons lost the relations between atoms; and model size
  made no difference, with a zero-parameter unigram scorer doing as well.
- **Line-keep**, a plain rule: inside code dumps, keep whole lines that hold an atom or an error,
  elide the rest. Against loosy (Haiku, 30 transcripts) it scored +0.071 [+0.01, +0.14] at light,
  +0.048 at medium, +0.037 at heavy. It looked like the first transform to beat loosy on real QA.

Offline lexical survival had ranked the variants the other way round (word skeleton first,
0.94 survival against 0.85 real accuracy), the second time a string-match proxy misled.

### Slot compression (2026-10-22): the original rewrite idea, tested

The user's original idea had still not been tested directly: rewrite sentences into shorter
equivalents rather than drop them. The user specified the scheme by analogy with Greek case
endings and with transform → quantize → cross-row coding: each sentence becomes a fixed slot row
(action, agent, object, recipient, extra), meaning carried by slot position, each slot reduced to
the fewest words, repeated values carried forward across rows. Negation, modality and cause must
survive ([spec](docs/superpowers/specs/2026-10-22-slot-compress-design.md)).
The agent built it on a spaCy parse and made the schema and quantizer calls.

| variant | prose+thinking cut | downstream vs loosy |
| --- | --- | --- |
| v1 slot rows | 12.7% (direct) / 19.3% (English) | not run |
| v2 telegraphic rows | 40.9% / 40.0% | **−0.12 / −0.09 at light (p = 0.008)**, set C; n.s. at medium/heavy |
| SLM rewrite, Qwen3-1.7B | 20.9% | −0.05 at light (n.s.); level elsewhere |
| v3 gentle (articles, filler, hedges only) | 15.0% | −0.02 to 0.00 (n.s.), set D |
| line-keep, re-tested | n/a | **−0.04 pooled C+D at light (p = 0.06)**: not confirmed |

The structural reason it could not win: prose and thinking are only 12–19% of tokens in these
sessions, so even a 41% cut frees a few percent of budget (4.9% of all tokens for v2), while the
lost facts sit in dropped code and tool output. A negation-scope bug ("don't have to" read as
"must not") was found in v2 and fixed in v3. The user closed the experiment the same day.

### Falsification (2026-10-23) and the local size curve (2026-11-16)

Every extractive and rewrite variant now sat 0.10–0.14 below the summary. Before retiring the
idea, the user asked for a test of the places it could still win, with a rule fixed in advance:
retire only if all three come back negative
([spec](docs/superpowers/specs/2026-10-23-falsify-extractive-design.md)).

1. **Exact-detail probes.** Probes whose answer contains an atom (path, identifier, number)
   might favour verbatim text. They do not: on set D loosy trails the summary by 0.09–0.14 on
   exact probes too, significant at every level. Negative.
2. **Hybrid.** A summary at 70% or 50% of the budget, with loosy filling the rest verbatim.
   Level with the summary (+0.004 to +0.015, n.s.) and +0.09 to +0.15 over loosy; `hyb70` at light
   used about 8% fewer tokens than the summary. Negative (no gain over the summary).
3. **Local summary.** The same summary task given to local Qwen3-1.7B. Loosy beat it by
   0.31–0.34 (p < 0.001). **Positive.**

So the idea was not retired; its niche is compaction without a frontier model. The obvious
objection is that 1.7B is too small, so the size curve followed: 4B and 8B (4-bit MLX) gain
only 0.01–0.05 per step, the 8B run needed an MLX cache cap to fit in 16 GB, and loosy still leads
8B by 0.26–0.32 (p ≤ 0.004). The local models also used just 37–67% of their budget.

### Continuous compaction (2026-11-16)

The user asked whether loosy has any use beyond one-shot compaction, and proposed running it
continuously, after every model turn, instead of at a threshold. That is the one setting where
loosy's cost profile matters: it is free and local, so it can run every turn, which a frontier
summary cannot do cheaply. Spec and running log:
[2026-11-16-continuous-compact-design.md](docs/superpowers/specs/2026-11-16-continuous-compact-design.md).

The user made four design calls:
- test two continuous modes: **recompute** (re-run loosy over the whole history every turn) and
  **freeze** (compress each 4k chunk once as it ages out of the verbatim tail, never touch it
  again, so the prompt-cache prefix survives);
- replay a small pilot on public transcripts before any live agent runs;
- drop the full-context probe filter, so that a gain from trimming context ("context rot")
  could show up;
- after the pilot, add a **decaying** arm: older chunks are recompressed again and again (LSM-style
  tiers), giving the context a recency bias.

The new metric is cost over the whole trajectory, not just accuracy at one point: context tokens
per turn and uncached tokens per turn, the latter standing in for prompt-cache misses.

**Pilot (set E, 10 transcripts, 30 checkpoints, 310 calls).**
[Results](docs/results/exp_continuous_pilot.md).

| arm | accuracy | context tokens/turn | uncached tokens/turn |
| --- | --- | --- | --- |
| full | 0.940 | 21.6k | 1.1k |
| threshold summary (`/compact`-style) | 0.932 | 10.0k | 1.9k |
| freeze | 0.900 | 10.0k | 1.5k |
| recompute | 0.835 | 10.6k | 4.0k |

Freeze trails threshold compaction by 0.032 (not significant) at about 20% fewer uncached tokens,
inside the pre-set 0.05 bar for scaling up, and the gap closes to 0 by the end of the transcript.
Recompute loses on both accuracy (−0.10, p = 0.02) and cache cost. Nothing beat full context.

**Per-chunk ablation (sets C and D, one-shot, 180 calls).** Freeze beat recompute by +0.065 at
equal size. Either its even per-chunk budget spread coverage better, or it kept up to 4k more
recent text verbatim. Running loosy per chunk at the same total budget on one-shot compaction
never beat plain loosy (−0.06 at set C light, p = 0.03; level on set D), so the even budget is
not the cause; the longer verbatim recent window is.
[Results](docs/results/exp_chunk_ablation.md).

**Scaled run (set F, 25 transcripts, 75 checkpoints, 865 calls).**
[Results](docs/results/exp_continuous_f.md). The cap rose to 65% of each transcript, so threshold
compaction fires about twice per transcript, closer to real use than the pilot's 40%. The
continuous arms were calibrated offline to threshold compaction's context size; "decaying" merges
every 4 chunks at keep 0.7.

| arm | accuracy | context tokens/turn | uncached tokens/turn |
| --- | --- | --- | --- |
| full | 0.952 | 18.8k | 0.96k |
| threshold summary (`/compact`-style) | 0.937 | 12.3k | **1.37k** |
| decaying freeze | 0.917 | 12.6k | 1.61k |
| freeze, size-matched (keep 0.6) | 0.913 | 12.4k | 1.52k |
| freeze, pilot settings (keep 0.33) | 0.857 | 10.2k | 1.41k |

The pilot did not replicate. At equal context size threshold compaction is ahead on accuracy
(+0.02, not significant) **and** on cache cost. The pilot's cache edge came from the low cap,
which made threshold compaction fire 3–7 times. The user's decaying, recency-biased arm made
no difference against plain freeze at equal size (+0.003, p = 0.50). No arm beat full context,
so there is no context-rot gain to harvest at these lengths. Continuous loosy is therefore not a
better way to compact when a frontier model is available.

**Without a frontier model (set F, 150 calls).** The user approved one more arm: threshold
compaction exactly like `/compact`, but with loosy doing the compacting, at the same target and
event schedule. The table is the same run with that arm added.

| arm | accuracy | context tokens/turn | uncached tokens/turn |
| --- | --- | --- | --- |
| threshold summary (Sonnet) | 0.937 | 12.3k | 1.37k |
| decaying freeze (continuous loosy) | 0.917 | 12.6k | 1.61k |
| freeze, size-matched (continuous loosy) | 0.913 | 12.4k | 1.52k |
| threshold loosy | 0.873 | 12.5k | 1.52k |

**This is positive.** At equal context size and equal cache cost, continuous loosy beats threshold loosy
by +0.041 (p = 0.027) for freeze and +0.044 (p = 0.017) for decaying freeze. Threshold loosy decays
along the session (0.93 → 0.88 → 0.81 at 50/75/100%), because each compaction re-compresses the
last one's output and loses more. Continuous freeze compresses each chunk once from the
original and holds at about 0.90–0.92. Not having a frontier model costs 0.065 under threshold
compaction; continuous freeze recovers about two-thirds of that.

So the best way to use loosy in its niche (offline, free, deterministic, no frontier model) is
continuously, in freeze mode, not as a drop-in `/compact`. The shipped `/loosy` handoff is
threshold-style today; adding a continuous mode is an open product decision (see Future research).

### Open-weight replay of continuous compaction (2026-11-17 to 11-22)

The no-frontier result above still used Claude models to answer and grade. The user asked for the
same comparison with no closed model anywhere, which is the setting the feature is for. Set F
was replayed with local Qwen3-8B (4-bit MLX) answering, Mistral-Nemo-12B writing and grading
the probes, and a new arm, `qthresh`: threshold compaction where Qwen summarises its own
history. Every call ran on one 16 GB Mac, so these numbers stand alone and are not comparable
with the Claude-answered tables.
[Results](docs/results/exp_continuous_qwen.md); spec and running log:
[2026-11-17-continuous-local-design.md](docs/superpowers/specs/2026-11-17-continuous-local-design.md).

| arm | accuracy | context tokens/turn | uncached tokens/turn |
| --- | --- | --- | --- |
| full (15 of 75 checkpoints over Qwen's 32k window) | 0.717 | 18.8k | 0.96k |
| freeze, size-matched (continuous loosy) | 0.710 | 12.4k | 1.52k |
| freeze, gated start | 0.710 | 12.7k | 1.57k |
| threshold loosy | 0.677 | 12.5k | 1.52k |
| threshold Qwen self-summary (`qthresh`) | 0.595 | 11.8k | 1.19k |

All 359 answered checkpoints are graded, so the pairs between compacted arms cover 24–25 of the
25 transcripts. The full row averages only the 14 transcripts whose three checkpoints all fit
the window; the comparisons against full below pair per checkpoint over all 25. Every row kept
its sign and size across the interim reports (n = 13–14, 18–20, 21–23).

- **Freeze matches full context.** −0.004 per checkpoint (p = 0.65) at about a third fewer
  tokens per turn, and it never overflowed the window, which full context did on 15 checkpoints.
- **Freeze beats Qwen's own summaries by +0.113** (p < 0.001, n = 24). The self-summary arm
  falls to about 0.53 from the 75% point on. This reverses the Sonnet result on the same set,
  where the threshold summary led: a small open-weight model summarising itself loses more
  than extractive selection does.
- **Freeze beats threshold loosy by +0.034** (p = 0.007, n = 25), close to the +0.041 measured
  with Claude answering.
- **Gating the start changes nothing** (−0.001, p = 1.0), so the feature freezes from the first
  turn (user's call, 2026-11-22).

How the run got there is part of the result. Nemo ignored JSON-only instructions until its
reply was prefilled with the opening of the object; greedy retries repeat themselves, so
retries sample; and Nemo grades an `UNKNOWN` answer with the label `UNKNOWN`, which the user
ruled is read as wrong. Validators stayed strict throughout.

**Prefill latency (three transcripts).** Replaying each arm's context stream through Qwen3-8B
with a prompt cache, the three compacted arms cost about the same per turn: 2.6–2.9 s on the 86
turns all of them timed. The order between arms changes from transcript to transcript, so no
arm is shown to be faster. The self-summary arm prefills about 13% fewer new tokens per turn;
freeze is the only arm that fit Qwen's window on every turn. A first replay was discarded
because another GPU job overlapped it and made the same turns five to seven times slower.
[Results](docs/results/exp_continuous_latency.md).

## Methodology and lessons learned

- **Proxies ranked variants backwards, twice.** Offline string-match survival ordered the
  spike's variants opposite to real QA (the word skeleton led offline, line-keep led on QA) and
  over-predicted the gain (0.94 survival, 0.85 accuracy). Atom recall and coverage remain useful
  for tuning a selector, but nothing downstream of it is decided on them any more.
- **Held-out sets, used once.** Experiment I's 20 transcripts (A) and the spike's 10 (B) were
  used for decisions, so the slot work drew fresh set C (10) and set D (20). The spike's line-keep
  gain did not survive the move to fresh data, which is the point of the discipline.
- **Probes from a model that does not summarise.** In Experiment I Sonnet wrote probes and
  summaries; from set C on, probes came from gpt-6-sol. The summary's lead held.
- **Pre-registered decision rules.** Stability (0.70 bar), the falsification rule and fixed
  hybrid splits were set before the data was seen, and nothing was tuned on set D.
- **Two metrics, never collapsed.** Recall and coverage were built to disagree, so the project
  reports both rather than picking a weighted winner.
- **A version stamp per transform.** The segment digest only fingerprints the segmenter's
  output; a rewritten downstream transform once left stale records that loaded as valid. Every
  transform now carries its own stamp in each record's identity ([PHASE2](docs/PHASE2.md#a-silent-staleness-trap-found-by-walking-into-it)).
- **Holes over bad averages.** A failed grid job is recorded as a failure, never as a zero
  record, and every summary opens with a loud incomplete-run block.
- **Cost caps.** After Experiment I hit the monthly spend limit, every later run had a call
  budget fixed up front, a hard cap, and a stop on the spend-limit error. A few summary calls
  that failed under the limit were not retried, which is why some cells read n = 18 or 19.
- **Private data stays private.** Five of the author's own sessions inform Phase 1–2 counts
  only. They are never committed or quoted; every downstream experiment used public data only.

## Use cases

Loosy-Goose fits when a frontier-model summary is unavailable, unaffordable or unwanted:

- **Offline, local or air-gapped compaction.** No API call, CPU only, and much better than a
  local 1.7B–8B summary at the same budget.
- **Privacy-sensitive sessions.** The transcript never leaves the machine, and no third model
  sees it.
- **Deterministic, auditable context.** Every kept line is verbatim and traceable; nothing is
  paraphrased or invented. Useful when you need to show what the model was given.
- **Cost-free pre-compaction.** Supersession alone removes about a quarter of tokens with no
  loss, and selection is a fraction of a second per 100k tokens, so it can shrink a transcript
  before a paid summary or a longer pipeline.
- **A transparent `/compact` alternative** in Claude Code and Codex, with the original session
  always kept.

It is the wrong tool when a frontier model can write the summary: at the same budget a Sonnet
summary keeps 0.11–0.14 more probe accuracy, and adding loosy's verbatim fill to the summary
gains nothing measurable.

## Future research

Each item states what is known and how it would be tested. All downstream tests should use a
fresh held-out set, a non-summarising probe writer, and a fixed call cap.

1. **Local hybrid.** Local summaries fill only 37–67% of their budget. Fill the rest with loosy's
   verbatim selection, as the frontier hybrid did. Known: the frontier hybrid matched the
   summary; whether a local hybrid beats plain loosy is open. Test: set-D-style run, local
   Qwen3-8B at its natural length plus loosy fill, against loosy at equal tokens.
2. **Budget split across kinds at medium/heavy.** The spike found force-kept code taking budget
   from tool results at tighter levels. Known: Experiment H's pool-share split lost on recall,
   but it was never judged downstream. Test: a small grid of per-kind slices scored by probe QA.
3. **Summary-undershoot calibration.** Both frontier and local summaries come in under target,
   which muddies "equal budget". Test: iterate the length instruction or truncate/pad, then re-run
   the summary arms at true parity.
4. **Larger or fine-tuned local summarisers.** The 1.7B → 8B curve is shallow, and it mixes bf16
   and 4-bit. Test: same-precision runs, a 14B+ model where memory allows, and a small model
   fine-tuned on (session → summary) pairs from public traces.
5. **Evaluations beyond probe QA.** Probes reward fact retention. Next-step agreement was noise
   at n = 20. Test: resume compressed sessions on real agent tasks (e.g. SWE-Gym) and score task
   completion, which is what compaction is for.
6. **GPT answerers on the later sets.** Sets C and D used Haiku only; Experiment I showed the
   ranking held across four answerers, but the later arms were not re-checked. Test: re-answer
   stored contexts with gpt-6-luna / gpt-6-sol.
7. **Novelty with `--band` / `--paths`.** Never measured together, so those flags fall back to
   plain leverage. Test: the existing Experiment G grid with banding and path variants enabled.
8. **Personalisation prior.** A background prior built from a user's own sessions to stabilise
   small-context SVD was in the original design and never built. Test: a prior from held-out
   sessions, measured on proxies and then downstream.
9. **Continuous freeze mode on live tasks.** Measured as the better no-frontier policy above and
   built as `loosy-goose proxy`. Open: whether live task success agrees with the replay. Test:
   live SWE-Gym runs through the proxy with freeze vs threshold loosy vs `/compact`.
10. **Packaging.** A PyPI release (with the CUDA-torch problem for plain pip users solved) and a
   Claude Code plugin/marketplace entry.

## Repository layout

- `src/loosy_goose/`: the library, CLI (`cli.py`), pipeline (`pipeline.py`), session handoff
  (`sessions/`, `handoff.py`) and MCP server (`mcp_server.py`). Module-by-module guide in
  [CLAUDE.md](CLAUDE.md#layout).
- `experiments/`: Phase 1–2 scripts and the grid runner; outputs are gitignored. Downstream
  drivers live on their branches (`exp/downstream-eval`, `spike/token-thinning`,
  `exp/slot-compress`).
- `docs/`: [DESIGN.md](docs/DESIGN.md) (algorithm and quality table),
  [PHASE1.md](docs/PHASE1.md), [PHASE2.md](docs/PHASE2.md) (decisions with the rejected option),
  [results/](docs/results/) (committed tables, with a staleness note per file).
- `integrations/`: the `/loosy` and `$loosy` commands and their [install guide](integrations/README.md).
- `tests/`: pytest; session fixtures are synthetic.

Development:

```
uv sync --dev --extra embed --extra mcp
uv run pytest                                  # offline suite
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run pytest -m live                          # opt-in, needs logged-in claude and codex
uv run python experiments/fetch_public.py      # public corpora into data/public/
```

`data/local/` is gitignored and empty on a fresh clone; `experiments/pick_local.py` builds an
equivalent corpus from your own sessions. The public corpora alone reproduce every public row in
`docs/results/`.

## Demo site

`site/` is a static Vite + TypeScript page: pick a transcript, slide the level, and watch which
segments the shipped compressor keeps, drops, or supersedes. Its data, `site/public/bundle.json`, is
not committed. `site/build_bundle.py` produces it by running the default selector at light, medium
and heavy on three SWE-Gym/OpenHands-SFT-Trajectories rows (MIT), fetched from the Hugging Face
hub at the revision `experiments/fetch_public.py` pins and checked against per-row hashes.
`.github/workflows/pages.yml` rebuilds and deploys it on every push to `main`, after the CI suite.

```
uv sync --dev --extra embed
uv run python site/build_bundle.py      # add --parquet data/public/swe-gym/data/<file>.parquet to skip the download
cd site && npm ci && npm run dev        # or: npm run build && npm run preview
```

The page sends no frame-blocking headers, so it can be embedded in an iframe; `?theme=light` or
`?theme=dark` overrides the system colour scheme.

**One-time owner step:** enable Pages under *Settings → Pages → Build and deployment → Source:
GitHub Actions*. Until then the `Pages` workflow's `configure-pages` step fails; nothing in the
workflow turns the setting on by itself.

## License

MIT. See [LICENSE](LICENSE).

# Continuous compaction, prefill latency on Qwen3-8B (one transcript), written 2026-11-22

`experiments/continuous/latency.py --set F --labels swe-gym_row118 --arms qthresh,tloosy,cfrzm`, Qwen3-8B 4-bit MLX on a
16 GB Apple-silicon Mac with no other model job on the GPU. Each arm's context stream is replayed turn by turn with one
prompt cache per arm, trimmed to the common token prefix, and only the new tokens are prefilled and timed. "Common turns"
are the boundaries every arm timed: qthresh and tloosy skip turns whose context is over Qwen's window (7 and 9), cfrzm
never is, so only the common-turn mean compares like with like. One transcript, so this is a diagnostic, not a result
with a confidence interval; the 3-transcript replay has not been rerun.

Prefill chunk 2,048 tokens (the default):

| arm | common turns | mean prefill s/turn (common) | mean new tokens/turn (common) | mean prefill s/turn (all timed) | skipped (over window) |
|---|---|---|---|---|---|
| qthresh | 38 | 2.78 | 1542 | 2.57 | 7 |
| tloosy | 38 | 3.18 | 1619 | 3.42 | 9 |
| cfrzm | 38 | 2.65 | 1705 | 2.75 | 0 |

Prefill chunk 512 tokens:

| arm | common turns | mean prefill s/turn (common) | mean new tokens/turn (common) | mean prefill s/turn (all timed) | skipped (over window) |
|---|---|---|---|---|---|
| qthresh | 38 | 2.81 | 1542 | 2.60 | 7 |
| tloosy | 38 | 2.93 | 1619 | 3.16 | 9 |
| cfrzm | 38 | 2.75 | 1705 | 2.86 | 0 |

Reading:

- The three arms cost about the same to prefill, 2.7 to 3.2 s per turn on the common turns, and the order between them
  changes with the chunk size, so no arm is shown to be faster. cfrzm prefills slightly more new tokens per turn and is not
  slower, because its contexts are shorter (mean 13.8k tokens on the common turns, against 16.9k and 18.0k).
- Timings are stable on an idle GPU. A turn adding under 700 tokens took 0.2 to 2.0 s in every arm, throughput was 510 to
  640 tokens/s, and the two chunk sizes agree within 10%. MLX peak memory was 8.0 to 8.9 GB in both passes.
- The first replay (2026-11-22, 00:34 to 02:09) is discarded. On this same transcript it measured 16.5, 24.2 and 24.1 s
  per common turn, six to nine times slower, while another project's model job shared the GPU. Its file is kept locally as
  `latency_contended_2026-11-22.json` and its numbers are not used anywhere.

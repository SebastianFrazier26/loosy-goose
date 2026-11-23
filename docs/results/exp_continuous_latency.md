# Continuous compaction, prefill latency on Qwen3-8B, written 2026-11-22

`experiments/continuous/latency.py --set F --n 3 --arms qthresh,tloosy,cfrzm`, Qwen3-8B 4-bit MLX on a 16 GB
Apple-silicon Mac with no other model job on the GPU. Each arm's context stream is replayed turn by turn with one prompt
cache per (arm, transcript), trimmed to the common token prefix, and only the new tokens are prefilled and timed.
"Common turns" are the boundaries every arm timed: qthresh and tloosy skip turns whose context is over Qwen's window (7
and 9, all in one transcript), cfrzm never is, so only the common-turn mean compares like with like. Three transcripts,
so there is no confidence interval and no test; read it as an order of magnitude.

Three transcripts (swe-gym rows 118, 119, 126), prefill chunk 2,048 tokens:

| arm | transcripts | common turns | mean prefill s/turn (common) | mean new tokens/turn (common) | mean prefill s/turn (all timed) | skipped (over window) |
|---|---|---|---|---|---|---|
| qthresh | 3/3 | 86 | 2.63 | 1514 | 2.55 | 7 |
| tloosy | 3/3 | 86 | 2.92 | 1754 | 3.03 | 9 |
| cfrzm | 3/3 | 86 | 2.75 | 1750 | 2.79 | 0 |

Mean prefill s/turn on the common turns, per transcript:

| arm | row118 (38 turns) | row119 (21 turns) | row126 (27 turns) |
|---|---|---|---|
| qthresh | 3.08 | 2.31 | 2.24 |
| tloosy | 3.18 | 2.30 | 3.04 |
| cfrzm | 2.72 | 2.89 | 2.67 |

Reading:

- **The arms cost about the same to prefill:** 2.6 to 2.9 s per turn on the common turns, at 580 to 640 new tokens per
  second. The order between arms changes from transcript to transcript (cfrzm is fastest on one, slowest on another), so
  no arm is shown to be faster.
- **qthresh prefills the fewest new tokens** (1,514 per turn against about 1,750), as the text-space dry run predicted:
  a summary event replaces history with a short summary, while loosy arms re-send compressed text. That is a 13% token
  difference and about 0.1 to 0.3 s per turn here.
- **Only cfrzm timed every turn.** qthresh and tloosy went over Qwen's window on 7 and 9 turns of row118 and could not
  be run there at all.
- **Timings are stable on an idle GPU.** A turn adding under 700 tokens took 0.2 to 1.5 s in every arm, and MLX peak
  memory was 8.1 to 8.9 GB.

One-transcript diagnostic, run first (row118 only), at two prefill chunk sizes:

| arm | common turns | s/turn (common), chunk 2,048 | s/turn (common), chunk 512 | mean new tokens/turn (common) |
|---|---|---|---|---|
| qthresh | 38 | 2.78 | 2.81 | 1542 |
| tloosy | 38 | 3.18 | 2.93 | 1619 |
| cfrzm | 38 | 2.65 | 2.75 | 1705 |

The two chunk sizes agree within 10% and peak memory was the same (8.0 to 8.9 GB), so chunk size is not a lever. Row118
repeated within 11% between the diagnostic and the full replay (2.78 / 3.18 / 2.65 against 3.08 / 3.18 / 2.72).

The first replay (2026-11-22, 00:34 to 02:09) is discarded. On the same 86 common turns it measured 14.3, 21.2 and
19.1 s per turn, five to seven times slower, while another project's model job shared the GPU. Its file is kept locally
as `latency_contended_2026-11-22.json` and its numbers are not used anywhere.

# Continuous compaction, open-weight replay on set F, written 2026-11-22 17:36

CONT_VERSION 1, QTHRESH_VERSION 1, CFRZG_VERSION 1 (gate 16000 o200k tokens), cfrzm/cfrzg keep 0.6, tloosy share 0.33. qwen3-8b (4-bit MLX, greedy, thinking off) probe QA accuracy, graded by mistral-nemo-12b; probes by mistral-nemo-12b (probes.nemo4); qthresh summaries by Qwen/Qwen3-8B-MLX-4bit (lsum). Every call is local and no closed model is used anywhere, so these numbers are a standalone set, not comparable with the Claude-answered set F report. No full-context filter. In the arm table and the pairs between compacted arms, a transcript counts for an arm only if all three of its checkpoints are graded. Rows against full are paired per checkpoint instead: the difference is taken only at checkpoints where full has a grade (full is refused at checkpoints over Qwen's window), averaged within each transcript, and its n gives both transcripts and checkpoints. Mean [95% bootstrap CI over transcripts].

| arm | accuracy | mean context tokens/turn | mean uncached tokens/turn |
|---|---|---|---|
| full | 0.737 [0.69,0.79] n=12 | 18827 | 962 |
| qthresh | 0.599 [0.55,0.65] n=23 | 11788 | 1188 |
| tloosy | 0.673 [0.64,0.71] n=23 | 12482 | 1515 |
| cfrzm | 0.715 [0.67,0.76] n=22 | 12355 | 1521 |
| cfrzg | 0.714 [0.67,0.76] n=22 | 12717 | 1565 |

The full row covers only the transcripts whose three checkpoints all fit Qwen's window and were graded (12 of 25); the rows against full below pair per checkpoint instead.

Checkpoints per arm (an arm answers a checkpoint once its probes exist and it fits):

| arm | expected | answered | graded |
|---|---|---|---|
| full | 75 | 60 | 58 |
| qthresh | 75 | 74 | 73 |
| tloosy | 75 | 75 | 73 |
| cfrzm | 75 | 75 | 72 |
| cfrzg | 75 | 75 | 72 |

Recorded failures (qwen_failures.jsonl, distinct keys): probe segments 0, qthresh summaries 0, answers 0 (plus 16 over the window, below), grades 11. A resume skips these keys unless run with --retry-failed.

Qwen summary length / target (o200k, 32 cached summaries): mean 0.65, min 0.08, max 1.41; 12 over 1 were sentence-truncated to the target.

qthresh mean context 11788 vs thresh (Sonnet summaries, set F) 12342: -4.5%.

| paired difference | delta accuracy |
|---|---|
| qthresh - tloosy | -0.069 [-0.11,-0.03] sign p=0.004 n=22 |
| cfrzm - qthresh | +0.116 [+0.08,+0.15] sign p=0.000 n=21 |
| cfrzm - tloosy | +0.043 [+0.02,+0.07] sign p=0.001 n=22 |
| cfrzg - cfrzm | -0.001 [-0.01,+0.00] sign p=1.000 n=22 |
| qthresh - full | -0.104 [-0.14,-0.06] sign p=0.001 n=25 transcripts, 58 checkpoints |
| tloosy - full | -0.035 [-0.06,-0.01] sign p=0.077 n=25 transcripts, 57 checkpoints |
| cfrzm - full | -0.008 [-0.04,+0.02] sign p=0.804 n=25 transcripts, 56 checkpoints |
| cfrzg - full | -0.008 [-0.04,+0.02] sign p=0.454 n=25 transcripts, 56 checkpoints |

Per checkpoint position:

| arm | 50% | 75% | 100% |
|---|---|---|---|
| full | 0.763 [0.69,0.83] n=25 | 0.706 [0.64,0.77] n=21 | 0.704 [0.65,0.76] n=12 |
| qthresh | 0.735 [0.65,0.81] n=25 | 0.531 [0.46,0.60] n=25 | 0.528 [0.44,0.62] n=23 |
| tloosy | 0.760 [0.68,0.83] n=25 | 0.652 [0.58,0.71] n=24 | 0.622 [0.57,0.68] n=24 |
| cfrzm | 0.771 [0.69,0.84] n=24 | 0.677 [0.59,0.75] n=24 | 0.689 [0.63,0.75] n=24 |
| cfrzg | 0.771 [0.70,0.84] n=24 | 0.675 [0.59,0.75] n=24 | 0.689 [0.63,0.75] n=24 |

Checkpoints over the Qwen window (answer prompt over 31232 Qwen tokens, i.e. 32768 less the 1536-token reply cap), out of 75. These answers are never queued and the context is never cut; the checkpoint stays ungraded for that arm (full's pairs skip it, see above):

| arm | over window | checkpoints |
|---|---|---|
| full | 15 | swe-gym_row135@75, swe-gym_row135@100, swe-gym_row118@75, swe-gym_row118@100, swe-gym_row129@75, swe-gym_row129@100, swe-gym_row172@100, swe-gym_row154@75, swe-gym_row154@100, swe-gym_row409@100, swe-gym_row174@100, swe-gym_row126@100, swe-gym_row46@100, swe-gym_row330@100, swe-gym_row156@100 |
| qthresh | 1 | swe-gym_row118@100 |
| tloosy | 0 | - |
| cfrzm | 0 | - |
| cfrzg | 0 | - |

Local calls recorded in checkpoint: 930

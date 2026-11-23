# Continuous compaction, open-weight replay on set F, written 2026-11-22 19:47

CONT_VERSION 1, QTHRESH_VERSION 1, CFRZG_VERSION 1 (gate 16000 o200k tokens), cfrzm/cfrzg keep 0.6, tloosy share 0.33. qwen3-8b (4-bit MLX, greedy, thinking off) probe QA accuracy, graded by mistral-nemo-12b; probes by mistral-nemo-12b (probes.nemo4); qthresh summaries by Qwen/Qwen3-8B-MLX-4bit (lsum). Every call is local and no closed model is used anywhere, so these numbers are a standalone set, not comparable with the Claude-answered set F report. No full-context filter. In the arm table and the pairs between compacted arms, a transcript counts for an arm only if all three of its checkpoints are graded. Rows against full are paired per checkpoint instead: the difference is taken only at checkpoints where full has a grade (full is refused at checkpoints over Qwen's window), averaged within each transcript, and its n gives both transcripts and checkpoints. Mean [95% bootstrap CI over transcripts].

| arm | accuracy | mean context tokens/turn | mean uncached tokens/turn |
|---|---|---|---|
| full | 0.717 [0.66,0.77] n=14 | 18827 | 962 |
| qthresh | 0.595 [0.55,0.64] n=24 | 11788 | 1188 |
| tloosy | 0.677 [0.64,0.71] n=25 | 12482 | 1515 |
| cfrzm | 0.710 [0.67,0.75] n=25 | 12355 | 1521 |
| cfrzg | 0.710 [0.67,0.75] n=25 | 12717 | 1565 |

The full row covers only the transcripts whose three checkpoints all fit Qwen's window and were graded (14 of 25); the rows against full below pair per checkpoint instead.

Checkpoints per arm (an arm answers a checkpoint once its probes exist and it fits):

| arm | expected | answered | graded |
|---|---|---|---|
| full | 75 | 60 | 60 |
| qthresh | 75 | 74 | 74 |
| tloosy | 75 | 75 | 75 |
| cfrzm | 75 | 75 | 75 |
| cfrzg | 75 | 75 | 75 |

Recorded failures (qwen_failures.jsonl, distinct keys): probe segments 0, qthresh summaries 0, answers 0 (plus 16 over the window, below), grades 0. A resume skips these keys unless run with --retry-failed.

Qwen summary length / target (o200k, 32 cached summaries): mean 0.65, min 0.08, max 1.41; 12 over 1 were sentence-truncated to the target.

qthresh mean context 11788 vs thresh (Sonnet summaries, set F) 12342: -4.5%.

| paired difference | delta accuracy |
|---|---|
| qthresh - tloosy | -0.078 [-0.11,-0.04] sign p=0.002 n=24 |
| cfrzm - qthresh | +0.113 [+0.08,+0.15] sign p=0.000 n=24 |
| cfrzm - tloosy | +0.034 [+0.01,+0.06] sign p=0.007 n=25 |
| cfrzg - cfrzm | -0.001 [-0.00,+0.00] sign p=1.000 n=25 |
| qthresh - full | -0.100 [-0.14,-0.06] sign p=0.000 n=25 transcripts, 60 checkpoints |
| tloosy - full | -0.034 [-0.06,-0.01] sign p=0.096 n=25 transcripts, 60 checkpoints |
| cfrzm - full | -0.004 [-0.04,+0.03] sign p=0.648 n=25 transcripts, 60 checkpoints |
| cfrzg - full | -0.005 [-0.04,+0.03] sign p=0.359 n=25 transcripts, 60 checkpoints |

Per checkpoint position:

| arm | 50% | 75% | 100% |
|---|---|---|---|
| full | 0.763 [0.69,0.83] n=25 | 0.706 [0.64,0.77] n=21 | 0.659 [0.58,0.73] n=14 |
| qthresh | 0.735 [0.65,0.81] n=25 | 0.531 [0.46,0.60] n=25 | 0.529 [0.45,0.61] n=24 |
| tloosy | 0.760 [0.68,0.83] n=25 | 0.654 [0.59,0.71] n=25 | 0.615 [0.56,0.67] n=25 |
| cfrzm | 0.774 [0.70,0.84] n=25 | 0.678 [0.60,0.75] n=25 | 0.680 [0.62,0.74] n=25 |
| cfrzg | 0.773 [0.70,0.84] n=25 | 0.676 [0.60,0.75] n=25 | 0.680 [0.62,0.74] n=25 |

Checkpoints over the Qwen window (answer prompt over 31232 Qwen tokens, i.e. 32768 less the 1536-token reply cap), out of 75. These answers are never queued and the context is never cut; the checkpoint stays ungraded for that arm (full's pairs skip it, see above):

| arm | over window | checkpoints |
|---|---|---|
| full | 15 | swe-gym_row135@75, swe-gym_row135@100, swe-gym_row118@75, swe-gym_row118@100, swe-gym_row129@75, swe-gym_row129@100, swe-gym_row172@100, swe-gym_row154@75, swe-gym_row154@100, swe-gym_row409@100, swe-gym_row174@100, swe-gym_row126@100, swe-gym_row46@100, swe-gym_row330@100, swe-gym_row156@100 |
| qthresh | 1 | swe-gym_row118@100 |
| tloosy | 0 | - |
| cfrzm | 0 | - |
| cfrzg | 0 | - |

Local calls recorded in checkpoint: 941

# Continuous compaction pilot, set E, written 2026-11-16 21:15

CONT_VERSION 1. Haiku 5.5 probe QA accuracy, graded by Sonnet 5.5; probes by gpt-6-sol, no full-context filter (every probe counts, for every arm). A transcript counts for an arm only if all three of its checkpoints are graded. Mean [95% bootstrap CI over transcripts].

| arm | accuracy | mean context tokens/turn | mean uncached tokens/turn |
|---|---|---|---|
| full | 0.940 [0.90,0.97] n=10 | 21647 | 1107 |
| thresh | 0.932 [0.90,0.96] n=10 | 9959 | 1930 |
| crec | 0.835 [0.79,0.87] n=10 | 10593 | 3966 |
| cfrz | 0.900 [0.86,0.94] n=10 | 10030 | 1544 |

| paired difference | delta accuracy |
|---|---|
| cfrz - thresh | -0.032 [-0.07,+0.01] sign p=1.000 n=10 |
| crec - thresh | -0.097 [-0.14,-0.06] sign p=0.021 n=10 |
| cfrz - crec | +0.065 [+0.04,+0.10] sign p=0.004 n=10 |
| thresh - full | -0.008 [-0.06,+0.05] sign p=0.453 n=10 |
| crec - full | -0.105 [-0.16,-0.06] sign p=0.039 n=10 |
| cfrz - full | -0.040 [-0.09,-0.00] sign p=0.727 n=10 |

| arm | 50% | 75% | 100% |
|---|---|---|---|
| full | 0.950 [0.92,0.98] n=10 | 0.940 [0.90,0.98] n=10 | 0.930 [0.87,0.98] n=10 |
| thresh | 0.950 [0.91,0.99] n=10 | 0.910 [0.86,0.96] n=10 | 0.935 [0.90,0.97] n=10 |
| crec | 0.825 [0.77,0.88] n=10 | 0.820 [0.72,0.91] n=10 | 0.860 [0.82,0.90] n=10 |
| cfrz | 0.870 [0.79,0.94] n=10 | 0.895 [0.82,0.96] n=10 | 0.935 [0.90,0.97] n=10 |

Calls recorded in checkpoint: 310

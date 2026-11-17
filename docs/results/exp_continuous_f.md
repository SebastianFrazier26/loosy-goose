# Continuous compaction set F, written 2026-11-16 22:47

CONT_VERSION 1, DEC_VERSION 1 (keep 0.7, m 4), cfrzm keep 0.6. Haiku 5.5 probe QA accuracy, graded by Sonnet 5.5; probes by gpt-6-sol, no full-context filter (every probe counts, for every arm). A transcript counts for an arm only if all three of its checkpoints are graded. Mean [95% bootstrap CI over transcripts].

| arm | accuracy | mean context tokens/turn | mean uncached tokens/turn |
|---|---|---|---|
| full | 0.952 [0.94,0.96] n=25 | 18827 | 962 |
| thresh | 0.937 [0.92,0.95] n=25 | 12342 | 1366 |
| cfrz | 0.857 [0.81,0.89] n=25 | 10177 | 1411 |
| cfrzm | 0.913 [0.88,0.94] n=25 | 12355 | 1521 |
| cdec | 0.917 [0.88,0.94] n=25 | 12625 | 1611 |
| tloosy | 0.873 [0.84,0.90] n=25 | 12482 | 1515 |

| paired difference | delta accuracy |
|---|---|
| cdec - thresh | -0.021 [-0.06,+0.01] sign p=1.000 n=25 |
| cfrz - thresh | -0.081 [-0.13,-0.04] sign p=0.011 n=25 |
| cfrzm - thresh | -0.024 [-0.06,+0.01] sign p=0.286 n=25 |
| cdec - cfrzm | +0.003 [-0.01,+0.02] sign p=0.503 n=25 |
| cfrzm - tloosy | +0.041 [+0.01,+0.07] sign p=0.027 n=25 |
| cdec - tloosy | +0.044 [+0.02,+0.07] sign p=0.017 n=25 |
| thresh - tloosy | +0.065 [+0.03,+0.10] sign p=0.003 n=25 |
| tloosy - full | -0.079 [-0.11,-0.05] sign p=0.000 n=25 |
| thresh - full | -0.015 [-0.03,+0.00] sign p=0.359 n=25 |
| cfrz - full | -0.095 [-0.14,-0.06] sign p=0.000 n=25 |
| cfrzm - full | -0.039 [-0.07,-0.01] sign p=0.143 n=25 |
| cdec - full | -0.035 [-0.07,-0.01] sign p=0.210 n=25 |

| arm | 50% | 75% | 100% |
|---|---|---|---|
| full | 0.956 [0.94,0.97] n=25 | 0.956 [0.93,0.98] n=25 | 0.944 [0.92,0.97] n=25 |
| thresh | 0.948 [0.93,0.97] n=25 | 0.918 [0.88,0.95] n=25 | 0.946 [0.91,0.97] n=25 |
| cfrz | 0.860 [0.79,0.92] n=25 | 0.864 [0.83,0.90] n=25 | 0.846 [0.81,0.89] n=25 |
| cfrzm | 0.922 [0.88,0.95] n=25 | 0.922 [0.89,0.95] n=25 | 0.896 [0.86,0.93] n=25 |
| cdec | 0.924 [0.88,0.96] n=25 | 0.922 [0.88,0.96] n=25 | 0.904 [0.88,0.93] n=25 |
| tloosy | 0.930 [0.89,0.96] n=25 | 0.876 [0.82,0.92] n=25 | 0.812 [0.76,0.85] n=25 |

Calls recorded in checkpoint: 1015

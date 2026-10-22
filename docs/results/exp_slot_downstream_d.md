# Slot-compression downstream QA, set D, written 2026-10-22 16:15

Haiku 5.5 probe QA accuracy, graded by Sonnet 5.5; probes written by gpt-6-sol, kept only if Haiku answers them correctly from the full context; mean [95% bootstrap CI over transcripts].

| condition | accuracy | mean earlier tokens |
|---|---|---|
| full | 1.000 [1.00,1.00] n=20 | 19333 |
| loosy-light | 0.863 [0.79,0.93] n=20 | 9587 |
| line-light | 0.836 [0.77,0.90] n=20 | 9623 |
| gloosy.v3-light | 0.843 [0.78,0.90] n=20 | 9556 |
| gline.v3-light | 0.859 [0.80,0.92] n=20 | 9617 |
| summary-light | 0.968 [0.95,0.99] n=18 | 9587 |
| loosy-medium | 0.817 [0.76,0.88] n=20 | 6422 |
| line-medium | 0.809 [0.73,0.88] n=20 | 6398 |
| gloosy.v3-medium | 0.799 [0.74,0.86] n=20 | 6414 |
| gline.v3-medium | 0.798 [0.72,0.87] n=20 | 6394 |
| summary-medium | 0.936 [0.90,0.96] n=20 | 6422 |
| loosy-heavy | 0.794 [0.73,0.85] n=20 | 4870 |
| line-heavy | 0.782 [0.70,0.86] n=20 | 4866 |
| gloosy.v3-heavy | 0.780 [0.72,0.84] n=20 | 4887 |
| gline.v3-heavy | 0.789 [0.71,0.87] n=20 | 4876 |
| summary-heavy | 0.931 [0.88,0.97] n=19 | 4870 |

| paired difference | delta accuracy |
|---|---|
| line-light - loosy-light | -0.028 [-0.08,+0.03] sign p=0.146 n=20 |
| gloosy.v3-light - loosy-light | -0.020 [-0.06,+0.03] sign p=0.227 n=20 |
| gline.v3-light - line-light | +0.023 [-0.01,+0.06] sign p=0.581 n=20 |
| gline.v3-light - loosy-light | -0.004 [-0.07,+0.06] sign p=1.000 n=20 |
| summary-light - loosy-light | +0.112 [+0.04,+0.19] sign p=0.039 n=18 |
| summary-light - line-light | +0.131 [+0.06,+0.21] sign p=0.006 n=18 |
| summary-light - gloosy.v3-light | +0.123 [+0.06,+0.19] sign p=0.021 n=18 |
| summary-light - gline.v3-light | +0.117 [+0.05,+0.19] sign p=0.022 n=18 |
| line-medium - loosy-medium | -0.009 [-0.07,+0.04] sign p=1.000 n=20 |
| gloosy.v3-medium - loosy-medium | -0.018 [-0.05,+0.01] sign p=1.000 n=20 |
| gline.v3-medium - line-medium | -0.010 [-0.05,+0.02] sign p=1.000 n=20 |
| gline.v3-medium - loosy-medium | -0.019 [-0.08,+0.04] sign p=1.000 n=20 |
| summary-medium - loosy-medium | +0.118 [+0.05,+0.18] sign p=0.021 n=20 |
| summary-medium - line-medium | +0.127 [+0.05,+0.21] sign p=0.035 n=20 |
| summary-medium - gloosy.v3-medium | +0.136 [+0.06,+0.21] sign p=0.031 n=20 |
| summary-medium - gline.v3-medium | +0.137 [+0.06,+0.22] sign p=0.008 n=20 |
| line-heavy - loosy-heavy | -0.012 [-0.06,+0.03] sign p=1.000 n=20 |
| gloosy.v3-heavy - loosy-heavy | -0.014 [-0.05,+0.01] sign p=1.000 n=20 |
| gline.v3-heavy - line-heavy | +0.007 [-0.04,+0.06] sign p=1.000 n=20 |
| gline.v3-heavy - loosy-heavy | -0.005 [-0.06,+0.04] sign p=1.000 n=20 |
| summary-heavy - loosy-heavy | +0.141 [+0.06,+0.22] sign p=0.035 n=19 |
| summary-heavy - line-heavy | +0.148 [+0.05,+0.25] sign p=0.077 n=19 |
| summary-heavy - gloosy.v3-heavy | +0.145 [+0.06,+0.22] sign p=0.035 n=19 |
| summary-heavy - gline.v3-heavy | +0.141 [+0.04,+0.25] sign p=0.118 n=19 |

Pooled line - loosy over sets C + D:

| level | delta accuracy |
|---|---|
| light | -0.036 [-0.08,+0.01] sign p=0.064 n=30 |
| medium | -0.009 [-0.05,+0.03] sign p=1.000 n=30 |
| heavy | -0.006 [-0.04,+0.03] sign p=1.000 n=30 |

Calls recorded in checkpoint: 711

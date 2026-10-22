# Slot-compression downstream QA, set C, written 2026-10-22 13:29

Haiku 5.5 probe QA accuracy, graded by Sonnet 5.5; probes written by gpt-6-sol, kept only if Haiku answers them correctly from the full context; mean [95% bootstrap CI over transcripts].

| condition | accuracy | mean earlier tokens |
|---|---|---|
| full | 1.000 [1.00,1.00] n=10 | 16531 |
| loosy-light | 0.974 [0.95,0.99] n=10 | 8254 |
| line-light | 0.922 [0.87,0.97] n=10 | 8232 |
| slotd.v2-light | 0.857 [0.80,0.91] n=10 | 8272 |
| slote.v2-light | 0.880 [0.82,0.92] n=10 | 8253 |
| slme.v1-qwen3-1.7b-light | 0.925 [0.88,0.97] n=10 | 8248 |
| summary-light | 0.946 [0.90,0.98] n=10 | 8254 |
| loosy-medium | 0.909 [0.85,0.96] n=10 | 5468 |
| line-medium | 0.898 [0.82,0.96] n=10 | 5473 |
| slotd.v2-medium | 0.888 [0.84,0.92] n=10 | 5521 |
| slote.v2-medium | 0.907 [0.87,0.94] n=10 | 5496 |
| slme.v1-qwen3-1.7b-medium | 0.908 [0.86,0.95] n=10 | 5474 |
| summary-medium | 0.951 [0.90,0.99] n=10 | 5468 |
| loosy-heavy | 0.888 [0.81,0.95] n=10 | 4159 |
| line-heavy | 0.894 [0.81,0.96] n=10 | 4174 |
| slotd.v2-heavy | 0.890 [0.84,0.94] n=10 | 4212 |
| slote.v2-heavy | 0.910 [0.85,0.96] n=10 | 4188 |
| slme.v1-qwen3-1.7b-heavy | 0.921 [0.86,0.98] n=10 | 4173 |
| summary-heavy | 0.956 [0.90,1.00] n=10 | 4159 |

| paired difference | delta accuracy |
|---|---|
| loosy-light - line-light | +0.052 [+0.00,+0.11] sign p=0.453 n=10 |
| line-light - loosy-light | -0.052 [-0.11,-0.00] sign p=0.453 n=10 |
| slotd.v2-light - line-light | -0.065 [-0.11,-0.02] sign p=0.070 n=10 |
| slotd.v2-light - loosy-light | -0.117 [-0.18,-0.06] sign p=0.008 n=10 |
| slote.v2-light - line-light | -0.042 [-0.08,+0.00] sign p=0.180 n=10 |
| slote.v2-light - loosy-light | -0.093 [-0.16,-0.05] sign p=0.008 n=10 |
| slme.v1-qwen3-1.7b-light - line-light | +0.003 [-0.02,+0.03] sign p=1.000 n=10 |
| slme.v1-qwen3-1.7b-light - loosy-light | -0.048 [-0.10,-0.01] sign p=0.219 n=10 |
| summary-light - line-light | +0.023 [-0.03,+0.07] sign p=1.000 n=10 |
| summary-light - loosy-light | -0.028 [-0.08,+0.02] sign p=1.000 n=10 |
| loosy-medium - line-medium | +0.011 [-0.04,+0.07] sign p=1.000 n=10 |
| line-medium - loosy-medium | -0.011 [-0.07,+0.04] sign p=1.000 n=10 |
| slotd.v2-medium - line-medium | -0.010 [-0.07,+0.06] sign p=0.453 n=10 |
| slotd.v2-medium - loosy-medium | -0.021 [-0.06,+0.02] sign p=1.000 n=10 |
| slote.v2-medium - line-medium | +0.009 [-0.06,+0.09] sign p=1.000 n=10 |
| slote.v2-medium - loosy-medium | -0.002 [-0.07,+0.06] sign p=1.000 n=10 |
| slme.v1-qwen3-1.7b-medium - line-medium | +0.010 [-0.04,+0.07] sign p=1.000 n=10 |
| slme.v1-qwen3-1.7b-medium - loosy-medium | -0.001 [-0.05,+0.05] sign p=0.727 n=10 |
| summary-medium - line-medium | +0.053 [-0.05,+0.15] sign p=0.508 n=10 |
| summary-medium - loosy-medium | +0.042 [-0.05,+0.12] sign p=0.453 n=10 |
| loosy-heavy - line-heavy | -0.006 [-0.03,+0.01] sign p=1.000 n=10 |
| line-heavy - loosy-heavy | +0.006 [-0.01,+0.03] sign p=1.000 n=10 |
| slotd.v2-heavy - line-heavy | -0.004 [-0.05,+0.05] sign p=1.000 n=10 |
| slotd.v2-heavy - loosy-heavy | +0.002 [-0.05,+0.05] sign p=1.000 n=10 |
| slote.v2-heavy - line-heavy | +0.015 [-0.02,+0.05] sign p=0.688 n=10 |
| slote.v2-heavy - loosy-heavy | +0.022 [-0.02,+0.06] sign p=0.453 n=10 |
| slme.v1-qwen3-1.7b-heavy - line-heavy | +0.026 [-0.01,+0.06] sign p=0.375 n=10 |
| slme.v1-qwen3-1.7b-heavy - loosy-heavy | +0.032 [+0.00,+0.06] sign p=0.219 n=10 |
| summary-heavy - line-heavy | +0.062 [-0.04,+0.17] sign p=0.289 n=10 |
| summary-heavy - loosy-heavy | +0.068 [-0.03,+0.16] sign p=0.180 n=10 |

Calls recorded in checkpoint: 420

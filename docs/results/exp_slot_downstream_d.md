# Slot-compression downstream QA, set D, written 2026-11-16 07:04

Haiku 5.5 probe QA accuracy, graded by Sonnet 5.5; probes written by gpt-6-sol, kept only if Haiku answers them correctly from the full context; mean [95% bootstrap CI over transcripts].

| condition | accuracy | mean earlier tokens |
|---|---|---|
| full | 1.000 [1.00,1.00] n=20 | 19333 |
| loosy-light | 0.863 [0.79,0.93] n=20 | 9587 |
| summary-light | 0.970 [0.95,0.99] n=19 | 9587 |
| lsum.v1-light | 0.522 [0.44,0.61] n=20 | 6968 |
| lsum4b.v1-light | 0.535 [0.46,0.61] n=20 | 6733 |
| lsum8b.v1-light | 0.547 [0.48,0.62] n=20 | 5762 |
| loosy-medium | 0.817 [0.76,0.88] n=20 | 6422 |
| summary-medium | 0.936 [0.90,0.96] n=20 | 6422 |
| lsum.v1-medium | 0.507 [0.43,0.59] n=20 | 4447 |
| lsum4b.v1-medium | 0.529 [0.46,0.59] n=20 | 3566 |
| lsum8b.v1-medium | 0.556 [0.48,0.64] n=20 | 2779 |
| loosy-heavy | 0.794 [0.73,0.85] n=20 | 4870 |
| summary-heavy | 0.935 [0.89,0.97] n=20 | 4870 |
| lsum.v1-heavy | 0.466 [0.41,0.52] n=20 | 3313 |
| lsum4b.v1-heavy | 0.507 [0.43,0.59] n=20 | 3607 |
| lsum8b.v1-heavy | 0.519 [0.45,0.59] n=20 | 2028 |

Local summary fidelity (o200k tokens after sentence truncation; fidelity = mean of summary / target per transcript):

| condition | mean summary tokens | mean target | fidelity | n |
|---|---|---|---|---|
| lsum.v1-light | 6968 | 9587 | 0.66 | 20 |
| lsum4b.v1-light | 6733 | 9587 | 0.61 | 20 |
| lsum8b.v1-light | 5762 | 9587 | 0.49 | 20 |
| lsum.v1-medium | 4447 | 6422 | 0.62 | 20 |
| lsum4b.v1-medium | 3566 | 6422 | 0.52 | 20 |
| lsum8b.v1-medium | 2779 | 6422 | 0.37 | 20 |
| lsum.v1-heavy | 3313 | 4870 | 0.58 | 20 |
| lsum4b.v1-heavy | 3607 | 4870 | 0.67 | 20 |
| lsum8b.v1-heavy | 2028 | 4870 | 0.37 | 20 |

| paired difference | delta accuracy |
|---|---|
| summary-light - loosy-light | +0.106 [+0.04,+0.18] sign p=0.039 n=19 |
| loosy-light - lsum.v1-light | +0.341 [+0.23,+0.45] sign p=0.000 n=20 |
| lsum.v1-light - summary-light | -0.455 [-0.53,-0.36] sign p=0.000 n=19 |
| loosy-light - lsum4b.v1-light | +0.328 [+0.23,+0.43] sign p=0.000 n=20 |
| loosy-light - lsum8b.v1-light | +0.316 [+0.23,+0.41] sign p=0.000 n=20 |
| lsum4b.v1-light - summary-light | -0.441 [-0.52,-0.36] sign p=0.000 n=19 |
| lsum8b.v1-light - summary-light | -0.428 [-0.50,-0.35] sign p=0.000 n=19 |
| lsum8b.v1-light - lsum.v1-light | +0.025 [-0.05,+0.09] sign p=0.077 n=20 |
| summary-medium - loosy-medium | +0.118 [+0.05,+0.18] sign p=0.021 n=20 |
| loosy-medium - lsum.v1-medium | +0.310 [+0.21,+0.40] sign p=0.000 n=20 |
| lsum.v1-medium - summary-medium | -0.428 [-0.50,-0.36] sign p=0.000 n=20 |
| loosy-medium - lsum4b.v1-medium | +0.288 [+0.19,+0.39] sign p=0.000 n=20 |
| loosy-medium - lsum8b.v1-medium | +0.261 [+0.16,+0.37] sign p=0.004 n=20 |
| lsum4b.v1-medium - summary-medium | -0.407 [-0.48,-0.34] sign p=0.000 n=20 |
| lsum8b.v1-medium - summary-medium | -0.380 [-0.46,-0.30] sign p=0.000 n=20 |
| lsum8b.v1-medium - lsum.v1-medium | +0.049 [+0.01,+0.09] sign p=0.035 n=20 |
| summary-heavy - loosy-heavy | +0.141 [+0.07,+0.22] sign p=0.021 n=20 |
| loosy-heavy - lsum.v1-heavy | +0.328 [+0.26,+0.40] sign p=0.000 n=20 |
| lsum.v1-heavy - summary-heavy | -0.469 [-0.53,-0.41] sign p=0.000 n=20 |
| loosy-heavy - lsum4b.v1-heavy | +0.287 [+0.19,+0.38] sign p=0.000 n=20 |
| loosy-heavy - lsum8b.v1-heavy | +0.275 [+0.20,+0.35] sign p=0.000 n=20 |
| lsum4b.v1-heavy - summary-heavy | -0.428 [-0.50,-0.36] sign p=0.000 n=20 |
| lsum8b.v1-heavy - summary-heavy | -0.416 [-0.50,-0.33] sign p=0.000 n=20 |
| lsum8b.v1-heavy - lsum.v1-heavy | +0.053 [-0.00,+0.11] sign p=0.454 n=20 |

Calls recorded in checkpoint: 1434

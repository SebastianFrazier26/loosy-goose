# Slot-compression downstream QA, set D, written 2026-10-23 17:18

Haiku 5.5 probe QA accuracy, graded by Sonnet 5.5; probes written by gpt-6-sol, kept only if Haiku answers them correctly from the full context; mean [95% bootstrap CI over transcripts].
NOTE: the driver stopped on a spend limit or --max-calls; tables are partial.

| condition | accuracy | mean earlier tokens |
|---|---|---|
| full | 1.000 [1.00,1.00] n=20 | 19333 |
| loosy-light | 0.863 [0.79,0.93] n=20 | 9587 |
| line-light | 0.836 [0.77,0.90] n=20 | 9623 |
| gloosy.v3-light | 0.843 [0.78,0.90] n=20 | 9556 |
| gline.v3-light | 0.859 [0.80,0.92] n=20 | 9617 |
| summary-light | 0.968 [0.95,0.99] n=18 | 9587 |
| hyb70.v1-light | 0.977 [0.96,0.99] n=19 | 8813 |
| hyb50.v1-light | 0.950 [0.92,0.98] n=20 | 9654 |
| lsum.v1-light | 0.522 [0.44,0.61] n=20 | 6968 |
| loosy-medium | 0.817 [0.76,0.88] n=20 | 6422 |
| line-medium | 0.809 [0.73,0.88] n=20 | 6398 |
| gloosy.v3-medium | 0.799 [0.74,0.86] n=20 | 6414 |
| gline.v3-medium | 0.798 [0.72,0.87] n=20 | 6394 |
| summary-medium | 0.936 [0.90,0.96] n=20 | 6422 |
| hyb70.v1-medium | 0.950 [0.92,0.98] n=20 | 6390 |
| hyb50.v1-medium | 0.949 [0.92,0.98] n=20 | 6406 |
| lsum.v1-medium | 0.507 [0.43,0.59] n=20 | 4447 |
| loosy-heavy | 0.794 [0.73,0.85] n=20 | 4870 |
| line-heavy | 0.782 [0.70,0.86] n=20 | 4866 |
| gloosy.v3-heavy | 0.780 [0.72,0.84] n=20 | 4887 |
| gline.v3-heavy | 0.789 [0.71,0.87] n=20 | 4876 |
| summary-heavy | 0.931 [0.88,0.97] n=19 | 4870 |
| hyb70.v1-heavy | 0.936 [0.90,0.97] n=20 | 4871 |
| hyb50.v1-heavy | 0.939 [0.90,0.97] n=20 | 4872 |
| lsum.v1-heavy | 0.466 [0.41,0.52] n=20 | 3313 |

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
| hyb70.v1-light - summary-light | +0.008 [-0.01,+0.03] sign p=1.000 n=18 |
| hyb50.v1-light - summary-light | -0.024 [-0.06,+0.01] sign p=0.727 n=18 |
| hyb70.v1-light - loosy-light | +0.113 [+0.05,+0.18] sign p=0.039 n=19 |
| hyb50.v1-light - loosy-light | +0.087 [+0.01,+0.16] sign p=0.424 n=20 |
| loosy-light - lsum.v1-light | +0.341 [+0.23,+0.45] sign p=0.000 n=20 |
| lsum.v1-light - summary-light | -0.452 [-0.54,-0.36] sign p=0.000 n=18 |
| line-medium - loosy-medium | -0.009 [-0.07,+0.04] sign p=1.000 n=20 |
| gloosy.v3-medium - loosy-medium | -0.018 [-0.05,+0.01] sign p=1.000 n=20 |
| gline.v3-medium - line-medium | -0.010 [-0.05,+0.02] sign p=1.000 n=20 |
| gline.v3-medium - loosy-medium | -0.019 [-0.08,+0.04] sign p=1.000 n=20 |
| summary-medium - loosy-medium | +0.118 [+0.05,+0.18] sign p=0.021 n=20 |
| summary-medium - line-medium | +0.127 [+0.05,+0.21] sign p=0.035 n=20 |
| summary-medium - gloosy.v3-medium | +0.136 [+0.06,+0.21] sign p=0.031 n=20 |
| summary-medium - gline.v3-medium | +0.137 [+0.06,+0.22] sign p=0.008 n=20 |
| hyb70.v1-medium - summary-medium | +0.015 [-0.02,+0.04] sign p=0.267 n=20 |
| hyb50.v1-medium - summary-medium | +0.014 [-0.01,+0.04] sign p=0.227 n=20 |
| hyb70.v1-medium - loosy-medium | +0.133 [+0.07,+0.19] sign p=0.001 n=20 |
| hyb50.v1-medium - loosy-medium | +0.132 [+0.07,+0.20] sign p=0.008 n=20 |
| loosy-medium - lsum.v1-medium | +0.310 [+0.21,+0.40] sign p=0.000 n=20 |
| lsum.v1-medium - summary-medium | -0.428 [-0.50,-0.36] sign p=0.000 n=20 |
| line-heavy - loosy-heavy | -0.012 [-0.06,+0.03] sign p=1.000 n=20 |
| gloosy.v3-heavy - loosy-heavy | -0.014 [-0.05,+0.01] sign p=1.000 n=20 |
| gline.v3-heavy - line-heavy | +0.007 [-0.04,+0.06] sign p=1.000 n=20 |
| gline.v3-heavy - loosy-heavy | -0.005 [-0.06,+0.04] sign p=1.000 n=20 |
| summary-heavy - loosy-heavy | +0.141 [+0.06,+0.22] sign p=0.035 n=19 |
| summary-heavy - line-heavy | +0.148 [+0.05,+0.25] sign p=0.077 n=19 |
| summary-heavy - gloosy.v3-heavy | +0.145 [+0.06,+0.22] sign p=0.035 n=19 |
| summary-heavy - gline.v3-heavy | +0.141 [+0.04,+0.25] sign p=0.118 n=19 |
| hyb70.v1-heavy - summary-heavy | +0.004 [-0.03,+0.04] sign p=1.000 n=19 |
| hyb50.v1-heavy - summary-heavy | +0.007 [-0.04,+0.06] sign p=1.000 n=19 |
| hyb70.v1-heavy - loosy-heavy | +0.142 [+0.08,+0.21] sign p=0.007 n=20 |
| hyb50.v1-heavy - loosy-heavy | +0.145 [+0.09,+0.20] sign p=0.001 n=20 |
| loosy-heavy - lsum.v1-heavy | +0.328 [+0.26,+0.40] sign p=0.000 n=20 |
| lsum.v1-heavy - summary-heavy | -0.470 [-0.53,-0.40] sign p=0.000 n=19 |

Pooled line - loosy over sets C + D:

| level | delta accuracy |
|---|---|
| light | -0.036 [-0.08,+0.01] sign p=0.064 n=30 |
| medium | -0.009 [-0.05,+0.03] sign p=1.000 n=30 |
| heavy | -0.006 [-0.04,+0.03] sign p=1.000 n=30 |

Calls recorded in checkpoint: 1192

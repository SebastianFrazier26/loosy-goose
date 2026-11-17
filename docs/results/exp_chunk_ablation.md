# Per-chunk loosy ablation, one-shot, sets C and D (2026-11-16)

`chunk` (CHUNK_VERSION 1): shipped loosy run inside ~4k-token chunks of the pre-tail pool with per-chunk budget shares, spill forward and a final global top-up, at the same level budget as `loosy`. Stored probes, full-context filter and reference arms are reused unchanged; Haiku 5.5 answers, Sonnet 5.5 grades. Mean [95% bootstrap CI over transcripts], exact sign test. Spec: docs/superpowers/specs/2026-11-16-continuous-compact-design.md (stage 2a).

## Set C

| condition | accuracy | mean earlier tokens |
|---|---|---|
| loosy-light | 0.974 [0.95,0.99] n=10 | - |
| summary-light | 0.946 [0.90,0.98] n=10 | - |
| chunk.v1-light | 0.913 [0.86,0.96] n=10 | 8300 |
| loosy-medium | 0.909 [0.85,0.96] n=10 | - |
| summary-medium | 0.951 [0.90,0.99] n=10 | - |
| chunk.v1-medium | 0.879 [0.83,0.93] n=10 | 5524 |
| loosy-heavy | 0.888 [0.81,0.95] n=10 | - |
| summary-heavy | 0.956 [0.90,1.00] n=10 | - |
| chunk.v1-heavy | 0.850 [0.78,0.92] n=10 | 4209 |

| paired difference | delta accuracy |
|---|---|
| chunk.v1-light - loosy-light | -0.060 [-0.10,-0.02] sign p=0.031 n=10 |
| chunk.v1-light - summary-light | -0.032 [-0.10,+0.04] sign p=1.000 n=10 |
| chunk.v1-medium - loosy-medium | -0.030 [-0.08,+0.02] sign p=0.754 n=10 |
| chunk.v1-medium - summary-medium | -0.072 [-0.14,+0.00] sign p=0.508 n=10 |
| chunk.v1-heavy - loosy-heavy | -0.038 [-0.08,+0.01] sign p=0.219 n=10 |
| chunk.v1-heavy - summary-heavy | -0.106 [-0.19,-0.03] sign p=0.125 n=10 |

## Set D

| condition | accuracy | mean earlier tokens |
|---|---|---|
| loosy-light | 0.863 [0.79,0.93] n=20 | - |
| summary-light | 0.970 [0.95,0.99] n=19 | - |
| chunk.v1-light | 0.855 [0.80,0.90] n=20 | 9703 |
| loosy-medium | 0.817 [0.76,0.88] n=20 | - |
| summary-medium | 0.936 [0.90,0.96] n=20 | - |
| chunk.v1-medium | 0.816 [0.76,0.87] n=20 | 6458 |
| loosy-heavy | 0.794 [0.73,0.85] n=20 | - |
| summary-heavy | 0.935 [0.89,0.97] n=20 | - |
| chunk.v1-heavy | 0.779 [0.72,0.83] n=20 | 4926 |

| paired difference | delta accuracy |
|---|---|
| chunk.v1-light - loosy-light | -0.009 [-0.06,+0.05] sign p=0.146 n=20 |
| chunk.v1-light - summary-light | -0.115 [-0.17,-0.07] sign p=0.001 n=19 |
| chunk.v1-medium - loosy-medium | -0.001 [-0.04,+0.03] sign p=1.000 n=20 |
| chunk.v1-medium - summary-medium | -0.119 [-0.18,-0.06] sign p=0.004 n=20 |
| chunk.v1-heavy - loosy-heavy | -0.015 [-0.05,+0.01] sign p=0.754 n=20 |
| chunk.v1-heavy - summary-heavy | -0.156 [-0.23,-0.09] sign p=0.013 n=20 |

## Reading (2026-11-16)

Explanation (a) is rejected: an even per-chunk budget does not help. `chunk` never beats `loosy`.
- Set C: it is worse at every level, significantly at light (−0.060, p=0.031).
- Set D: it is level, within 0.01, at every level.

That leaves explanation (b) for the pilot's `cfrz` > `crec` gap (+0.065). `cfrz` keeps the text not yet chunked verbatim, up to F=4k tokens beyond the shared tail, so it has a longer verbatim recent window. Recent verbatim context is what helps; spreading the budget across the history is not. Per-chunk selection is not a candidate default for one-shot loosy.

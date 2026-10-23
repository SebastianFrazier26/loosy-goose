# Probe types: exact-detail versus gist, set D (and C), 2026-10-23

Zero model calls: stored probes and grades only. A valid probe (Haiku correct from full context) is *exact* if its reference answer has >=1 atom per `segment.extract_atoms`, else *gist*. Cells: mean over transcripts of per-transcript accuracy within the class [95% bootstrap CI], n = transcripts with >=1 probe of that class.

Consistency with d/report.md: the `all` column equals `downstream.per_transcript` for every cell.
Pooled-column spot check (report.md light loosy 0.863, summary 0.968): 0.863 [0.79,0.93] n=20; 0.968 [0.95,0.99] n=18; report.md contains both: True.

## Probe counts (valid probes)

| set | exact | gist |
|---|---|---|
| D | 119 | 61 |
| C | 52 | 39 |
| C + D | 171 | 100 |

### exact examples (set D)

- Where was the uploaded repository located? (`/workspace/getmoto__moto__4.0`) [swe-gym_row140]
- What billing mode did the example use? (`PAY_PER_REQUEST`) [swe-gym_row140]
- What reproduction script did the assistant create? (`reproduce.py`) [swe-gym_row140]
- What repository path did the user upload? (`/workspace/pydantic__pydantic__2.6`) [swe-gym_row177]
- What directory contained the uploaded repository? (`/workspace/pandas-dev__pandas__2.0`) [swe-gym_row252]

### gist examples (set D)

- Which DynamoDB attribute was the partition key in the example? (`pk`) [swe-gym_row140]
- Which Pydantic version did the PR author report? (`2.5.3`) [swe-gym_row177]
- What import failed on the first reproduction run? (`Annotated from typing`) [swe-gym_row177]
- Which pandas version failed in the report? (`1.5.0`) [swe-gym_row252]
- Which attribute did the INCLUDE index explicitly project? (`attr2`) [swe-gym_row289]

## Set D

| condition | exact | gist | all |
|---|---|---|---|
| loosy-light | 0.888 [0.82,0.95] n=20 | 0.839 [0.72,0.94] n=19 | 0.863 [0.79,0.93] n=20 |
| line-light | 0.880 [0.81,0.94] n=20 | 0.796 [0.68,0.91] n=19 | 0.836 [0.77,0.90] n=20 |
| gline-light | 0.915 [0.86,0.97] n=20 | 0.772 [0.66,0.89] n=19 | 0.859 [0.80,0.92] n=20 |
| summary-light | 0.983 [0.96,1.00] n=18 | 0.949 [0.89,1.00] n=17 | 0.968 [0.95,0.99] n=18 |
| loosy-medium | 0.849 [0.80,0.90] n=20 | 0.768 [0.65,0.88] n=19 | 0.817 [0.76,0.88] n=20 |
| line-medium | 0.838 [0.76,0.91] n=20 | 0.777 [0.65,0.90] n=19 | 0.809 [0.73,0.88] n=20 |
| gline-medium | 0.812 [0.73,0.89] n=20 | 0.798 [0.67,0.92] n=19 | 0.798 [0.72,0.87] n=20 |
| summary-medium | 0.942 [0.91,0.97] n=20 | 0.931 [0.87,0.98] n=19 | 0.936 [0.90,0.96] n=20 |
| loosy-heavy | 0.817 [0.76,0.87] n=20 | 0.754 [0.64,0.86] n=19 | 0.794 [0.73,0.85] n=20 |
| line-heavy | 0.799 [0.72,0.87] n=20 | 0.763 [0.64,0.88] n=19 | 0.782 [0.70,0.86] n=20 |
| gline-heavy | 0.809 [0.75,0.87] n=20 | 0.797 [0.67,0.92] n=19 | 0.789 [0.71,0.87] n=20 |
| summary-heavy | 0.957 [0.91,0.99] n=19 | 0.873 [0.77,0.95] n=18 | 0.931 [0.88,0.97] n=19 |

| paired difference | exact | gist | all |
|---|---|---|---|
| loosy-light - summary-light | -0.098 [-0.17,-0.03] sign p=0.039 n=18 | -0.128 [-0.23,-0.02] sign p=0.070 n=17 | -0.112 [-0.19,-0.04] sign p=0.039 n=18 |
| line-light - summary-light | -0.094 [-0.17,-0.03] sign p=0.021 n=18 | -0.176 [-0.30,-0.05] sign p=0.039 n=17 | -0.131 [-0.21,-0.06] sign p=0.006 n=18 |
| gline-light - summary-light | -0.068 [-0.13,-0.02] sign p=0.125 n=18 | -0.203 [-0.33,-0.07] sign p=0.021 n=17 | -0.117 [-0.19,-0.05] sign p=0.022 n=18 |
| loosy-medium - summary-medium | -0.092 [-0.15,-0.04] sign p=0.022 n=20 | -0.163 [-0.27,-0.05] sign p=0.092 n=19 | -0.118 [-0.18,-0.05] sign p=0.021 n=20 |
| line-medium - summary-medium | -0.103 [-0.19,-0.03] sign p=0.146 n=20 | -0.154 [-0.26,-0.04] sign p=0.109 n=19 | -0.127 [-0.21,-0.05] sign p=0.035 n=20 |
| gline-medium - summary-medium | -0.129 [-0.21,-0.06] sign p=0.021 n=20 | -0.132 [-0.25,-0.02] sign p=0.227 n=19 | -0.137 [-0.22,-0.06] sign p=0.008 n=20 |
| loosy-heavy - summary-heavy | -0.140 [-0.22,-0.06] sign p=0.007 n=19 | -0.132 [-0.28,+0.02] sign p=0.065 n=18 | -0.141 [-0.22,-0.06] sign p=0.035 n=19 |
| line-heavy - summary-heavy | -0.154 [-0.24,-0.06] sign p=0.007 n=19 | -0.123 [-0.27,+0.04] sign p=0.227 n=18 | -0.148 [-0.25,-0.05] sign p=0.077 n=19 |
| gline-heavy - summary-heavy | -0.144 [-0.23,-0.07] sign p=0.007 n=19 | -0.088 [-0.26,+0.09] sign p=0.549 n=18 | -0.141 [-0.25,-0.04] sign p=0.118 n=19 |

## Set C

| condition | exact | gist | all |
|---|---|---|---|
| loosy-light | 0.983 [0.96,1.00] n=10 | 0.965 [0.92,1.00] n=10 | 0.974 [0.95,0.99] n=10 |
| line-light | 0.897 [0.76,1.00] n=10 | 0.919 [0.82,0.99] n=10 | 0.922 [0.87,0.97] n=10 |
| summary-light | 0.967 [0.93,1.00] n=10 | 0.942 [0.87,1.00] n=10 | 0.946 [0.90,0.98] n=10 |
| loosy-medium | 0.918 [0.86,0.97] n=10 | 0.898 [0.77,0.98] n=10 | 0.909 [0.85,0.96] n=10 |
| line-medium | 0.903 [0.80,0.98] n=10 | 0.894 [0.78,0.99] n=10 | 0.898 [0.82,0.96] n=10 |
| summary-medium | 0.992 [0.97,1.00] n=10 | 0.925 [0.84,1.00] n=10 | 0.951 [0.90,0.99] n=10 |
| loosy-heavy | 0.877 [0.82,0.93] n=10 | 0.915 [0.79,1.00] n=10 | 0.888 [0.81,0.95] n=10 |
| line-heavy | 0.925 [0.87,0.97] n=10 | 0.844 [0.63,0.99] n=10 | 0.894 [0.81,0.96] n=10 |
| summary-heavy | 0.992 [0.97,1.00] n=10 | 0.942 [0.86,1.00] n=10 | 0.956 [0.90,1.00] n=10 |

| paired difference | exact | gist | all |
|---|---|---|---|
| loosy-light - summary-light | +0.016 [-0.03,+0.07] sign p=1.000 n=10 | +0.023 [-0.07,+0.12] sign p=1.000 n=10 | +0.028 [-0.02,+0.08] sign p=1.000 n=10 |
| line-light - summary-light | -0.070 [-0.22,+0.04] sign p=1.000 n=10 | -0.022 [-0.16,+0.09] sign p=1.000 n=10 | -0.023 [-0.07,+0.03] sign p=1.000 n=10 |
| loosy-medium - summary-medium | -0.073 [-0.13,-0.02] sign p=0.125 n=10 | -0.027 [-0.20,+0.11] sign p=1.000 n=10 | -0.042 [-0.12,+0.05] sign p=0.453 n=10 |
| line-medium - summary-medium | -0.088 [-0.20,-0.00] sign p=0.375 n=10 | -0.031 [-0.19,+0.11] sign p=1.000 n=10 | -0.053 [-0.15,+0.05] sign p=0.508 n=10 |
| loosy-heavy - summary-heavy | -0.114 [-0.16,-0.06] sign p=0.016 n=10 | -0.027 [-0.20,+0.11] sign p=1.000 n=10 | -0.068 [-0.16,+0.03] sign p=0.180 n=10 |
| line-heavy - summary-heavy | -0.067 [-0.12,-0.02] sign p=0.125 n=10 | -0.098 [-0.35,+0.09] sign p=0.688 n=10 | -0.062 [-0.17,+0.04] sign p=0.289 n=10 |

## Pooled C + D (loosy, line, summary)

| condition | exact | gist | all |
|---|---|---|---|
| loosy-light | 0.920 [0.87,0.96] n=30 | 0.883 [0.80,0.95] n=29 | 0.900 [0.85,0.94] n=30 |
| line-light | 0.885 [0.82,0.94] n=30 | 0.839 [0.75,0.92] n=29 | 0.864 [0.81,0.91] n=30 |
| summary-light | 0.977 [0.96,0.99] n=28 | 0.946 [0.90,0.98] n=27 | 0.960 [0.94,0.98] n=28 |
| loosy-medium | 0.872 [0.83,0.91] n=30 | 0.813 [0.72,0.89] n=29 | 0.848 [0.80,0.89] n=30 |
| line-medium | 0.860 [0.79,0.92] n=30 | 0.818 [0.73,0.89] n=29 | 0.838 [0.78,0.89] n=30 |
| summary-medium | 0.958 [0.94,0.98] n=30 | 0.929 [0.88,0.97] n=29 | 0.941 [0.91,0.97] n=30 |
| loosy-heavy | 0.837 [0.79,0.88] n=30 | 0.810 [0.72,0.89] n=29 | 0.825 [0.78,0.87] n=30 |
| line-heavy | 0.841 [0.78,0.89] n=30 | 0.791 [0.69,0.88] n=29 | 0.820 [0.76,0.88] n=30 |
| summary-heavy | 0.969 [0.94,0.99] n=29 | 0.898 [0.83,0.96] n=28 | 0.940 [0.90,0.97] n=29 |

| paired difference | exact | gist | all |
|---|---|---|---|
| loosy-light - summary-light | -0.057 [-0.11,-0.01] sign p=0.180 n=28 | -0.072 [-0.16,+0.00] sign p=0.267 n=27 | -0.062 [-0.12,-0.01] sign p=0.189 n=28 |
| line-light - summary-light | -0.085 [-0.16,-0.02] sign p=0.035 n=28 | -0.119 [-0.22,-0.03] sign p=0.057 n=27 | -0.093 [-0.15,-0.04] sign p=0.013 n=28 |
| loosy-medium - summary-medium | -0.086 [-0.13,-0.04] sign p=0.002 n=30 | -0.116 [-0.21,-0.02] sign p=0.096 n=29 | -0.093 [-0.15,-0.04] sign p=0.011 n=30 |
| line-medium - summary-medium | -0.098 [-0.16,-0.04] sign p=0.049 n=30 | -0.111 [-0.20,-0.02] sign p=0.143 n=29 | -0.102 [-0.16,-0.04] sign p=0.023 n=30 |
| loosy-heavy - summary-heavy | -0.131 [-0.18,-0.08] sign p=0.000 n=29 | -0.095 [-0.21,+0.02] sign p=0.077 n=28 | -0.115 [-0.18,-0.05] sign p=0.007 n=29 |
| line-heavy - summary-heavy | -0.124 [-0.19,-0.06] sign p=0.001 n=29 | -0.114 [-0.24,+0.01] sign p=0.143 n=28 | -0.118 [-0.19,-0.04] sign p=0.023 n=29 |

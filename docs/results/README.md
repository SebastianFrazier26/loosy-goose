# Phase 1 raw result tables

Verbatim summary tables as emitted by the Phase 1 experiment scripts on 2026-09-08, kept here
so the findings survive independently of any one machine. Interpretation lives in
[../PHASE1.md](../PHASE1.md).

| File | Produced by |
| --- | --- |
| `corpus_stats.txt` | `experiments/corpus_stats.py` — per-transcript segment, token, per-kind and atom counts |
| `exp_a_summary.txt` | `experiments/exp_a_ppmi.py` — PPMI-SVD eigen-topic stability and labels |
| `exp_b_summary.txt` | `experiments/exp_b_embed.py` — embedding-SVD spectrum and leverage selection |
| `exp_c_summary.txt` | `experiments/exp_c_code.py` — C1 banding, C2 supersession, C3 embedder comparison |
| `exp_d_summary.txt` | `experiments/exp_d_curves.py` — Phase 2 rate-distortion grid at matched budgets, re-run 2026-09-09 after the atom-extractor fix |

Full per-transcript JSON and plots are **not** committed; they land in the gitignored
`experiments/output/` when the scripts run.

## Privacy

Rows labelled `local_5k` … `local_200k` come from the author's own Claude Code sessions, which
contain unrelated employer work. Those transcripts are gitignored and never committed. Only
aggregate numbers appear here — the experiment scripts already withhold sample topics and
direction labels for local rows (`(local session: withheld)`, `(private)`), and the session
UUIDs that appeared in the original filenames have been stripped from these copies.

Reproducing on another machine gives you the public rows immediately via `fetch_public.py`;
local rows regenerate from whatever sessions that machine has, via `pick_local.py`.

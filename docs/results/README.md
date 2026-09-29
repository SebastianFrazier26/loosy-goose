# Raw result tables

Verbatim summary tables as emitted by the experiment scripts, kept here so the findings survive
independently of any one machine. The first four are Phase 1, run 2026-09-08; `exp_d_summary.txt`
is Phase 2, re-baselined 2026-09-16/18. Interpretation lives in [../PHASE1.md](../PHASE1.md) and
[../PHASE2.md](../PHASE2.md).

| File | Produced by |
| --- | --- |
| `corpus_stats.txt` | `experiments/corpus_stats.py` — per-transcript segment, token, per-kind and atom counts |
| `exp_a_summary.txt` | `experiments/exp_a_ppmi.py` — PPMI-SVD eigen-topic stability and labels |
| `exp_b_summary.txt` | `experiments/exp_b_embed.py` — embedding-SVD spectrum and leverage selection |
| `exp_c_summary.txt` | `experiments/exp_c_code.py` — C1 banding, C2 supersession, C3 embedder comparison |
| `exp_d_summary.txt` | `experiments/exp_d_curves.py` — Phase 2 rate-distortion grid at matched budgets, re-baselined 2026-09-16/18 under `ATOMS_VERSION` 5 (672-job grid; `topics` rows omitted) |
| `exp_f_summary.txt` | `experiments/exp_d_curves.py --grid exp_f` — Experiment F, compression-based selectors (14 arms incl. seed variants), 2026-09-22 to 28; verbatim, since 10 of the arms were cut from the grid on 2026-09-29 |

Full per-transcript JSON and plots are **not** committed; they land in the gitignored
`experiments/output/` when the scripts run.

## What is stale, and how stale

These files are kept verbatim, as the record of what a given run said. They are **not** kept
current, and one of them is now well behind the code. Nothing here is edited to match; the
banners below are the correction.

**`corpus_stats.txt` — three versions behind, and its `atoms` column is a third convention.**
Produced 2026-09-08 under `ATOMS_VERSION` 1. Three extractor changes have landed since (2 decoded
`json.dumps` payloads, 3 widened the file-extension list, 4 narrowed it again and added the
dotted-chain rule), so every figure in the `atoms` column describes an extractor that no longer
exists. Separately, and independently of the staleness: that column is
`sum(len(s.atoms) for s in segments)` — **atom mentions**, every occurrence counted. It is not the
per-transcript-sum convention `docs/PHASE1.md` and `docs/PHASE2.md` use, and it is not the global
distinct union either. Three conventions are in use across this project; see the counting note in
[../PHASE1.md](../PHASE1.md). Do not compute a share by putting one of them over another.
Regenerate with `uv run python experiments/corpus_stats.py`, which reads `data/local/` as well as
the public corpora.

**`exp_d_summary.txt` — current as of `ATOMS_VERSION` 5, `TRANSFORMS_VERSION` 2, `PATHS_VERSION` 3
and `SUPERSEDE_VERSION` 2.** Replaced 2026-09-18 with the summary regenerated from the
2026-09-16/18 re-baseline's checkpoints after `topics` left the grid (the run log is the committed
`experiments/output/rebaseline_v5.log`, whose own summary tail still carries the `topics` rows);
the version-4 summary it replaced is in git history at 4cf7c7e, the version-2 one at 929997c. The
Experiment E analysis over the same checkpoints is `experiments/output/exp_e_v5.out`; its E4
timing row for `supersede+band+leverage` is contaminated by cross-run cache hits — see
`docs/PHASE2.md`, "Decided 2026-09-18". Regenerate with `uv run python experiments/exp_d_curves.py
--force`, which takes 6–24 hours depending on what else the machine is doing.

`exp_a_summary.txt`, `exp_b_summary.txt` and `exp_c_summary.txt` report Phase 1 results whose
conclusions are relative — stability against a pre-registered threshold, methods against each
other at one budget — so the extractor changes do not overturn them. Their absolute `atom_recall`
figures are stale on the same terms as everything else and should not be quoted as levels.

## Privacy

Rows labelled `local_5k` … `local_200k` come from the author's own Claude Code sessions, which
contain unrelated employer work. Those transcripts are gitignored and never committed. Only
aggregate numbers appear here — the experiment scripts already withhold sample topics and
direction labels for local rows (`(local session: withheld)`, `(private)`), and the session
UUIDs that appeared in the original filenames have been stripped from these copies.

Reproducing on another machine gives you the public rows immediately via `fetch_public.py`;
local rows regenerate from whatever sessions that machine has, via `pick_local.py`.

# Phase 1 experiment scripts

Outputs go to `experiments/output/` (gitignored). Committed result tables live in
`docs/results/`; the write-up is `docs/PHASE1.md`.

## Corpus setup

```
uv run python experiments/fetch_public.py    # public corpora into data/public/
uv run python experiments/pick_local.py      # optional: your own sessions into data/local/
uv run python experiments/corpus_stats.py
```

`fetch_public.py` pins dataset revisions and writes `data/public/SOURCES.md` with the hashes,
so the public half of the corpus is reproducible on any machine. `pick_local.py` is optional
and machine-specific — it reads `~/.claude/projects/*/*.jsonl`, and its output is never
committed.

## The three tracks

| Script | Question | Verdict |
| --- | --- | --- |
| `exp_a_ppmi.py` | Do PPMI-SVD eigen-topics stay stable on a single conversation? | Partial fail — topical (sparse) directions unstable; demoted to labeller |
| `exp_b_embed.py` | Does embedding-SVD leverage scoring select well? | Works, but the spectrum has no knee and scores are flat |
| `exp_c_code.py` | Can code be compressed rather than protected? | Yes — supersession removes 25.9% losslessly; AST banding beats dropping at equal budget |

`baselines.py` provides the comparison methods every track is scored against: random drop,
recency-only, and TF-IDF. All of them, and every experimental method, are forced through the
same token budget in `budget.py` — that shared budget is what makes the numbers comparable.

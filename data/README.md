# Data

- `data/local/` — gitignored. Holds the user's own LLM session logs used as a local development
  corpus (`experiments/pick_local.py` fills `data/local/sessions/`). Nothing under it is ever
  committed.
- `data/public/` — gitignored for now. `experiments/fetch_public.py` downloads public datasets
  here (trace-commons/agent-traces CC-BY-4.0, SWE-Gym/OpenHands-SFT-Trajectories MIT,
  OpenAssistant/oasst2 Apache-2.0) and writes `data/public/SOURCES.md` with the revision hashes.
  Committing a curated, license-attributed fixture subset is a separate decision still pending.

#!/bin/bash
# Fourth pass (2026-10-22): capped force-keep of code skeletons.
cd "$(dirname "$0")/../.." || exit 1
for m in unigram pythia-70m; do
  uv run python -c "
import sys; sys.path.insert(0, 'experiments/spike_thin')
import run; run.run_model('$m', set(run.CAP_ARMS), '__z')"
done
uv run python experiments/spike_thin/report.py quiet
echo "CAP PASS DONE"

#!/bin/bash
# Third pass (2026-10-22): force-kept thinned code dumps, on the two cheapest scorers.
cd "$(dirname "$0")/../.." || exit 1
for m in unigram pythia-70m; do
  uv run python -c "
import sys; sys.path.insert(0, 'experiments/spike_thin')
import run; run.run_model('$m', set(run.KEEP_ARMS), '__y')"
done
uv run python experiments/spike_thin/report.py quiet
echo "KEEP PASS DONE"

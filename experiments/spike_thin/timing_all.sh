#!/bin/bash
# Clean per-model timing (2026-10-22): one fresh process per model and device, so peak memory is
# that model's alone. Runs after the driver; overwrites the in-driver meta files.
cd "$(dirname "$0")/../.." || exit 1
for m in pythia-31m pythia-70m distilgpt2-82m smollm2-135m qwen3-0.6b qwen3-1.7b; do
  uv run python experiments/spike_thin/excerpt.py timing "$m"
  SPIKE_DEVICE=cpu uv run python experiments/spike_thin/excerpt.py timing "$m"
done
uv run python experiments/spike_thin/report.py quiet
echo "TIMING DONE"

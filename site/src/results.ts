// Hand-copied from committed result tables; each block names its source file so a re-run can be
// re-checked against it.

const REPO = "https://github.com/SebastianFrazier26/loosy-goose";

export const RATE_DISTORTION = {
  source: `${REPO}/blob/main/docs/results/exp_g_summary.txt`,
  sourceLabel: "docs/results/exp_g_summary.txt",
  transcripts: 18,
  rows: [
    { rate: "2x", recall: 0.829, coverage: 0.921, oldRecall: 0.837, oldCoverage: 0.895, reach: "15/18" },
    { rate: "3x", recall: 0.655, coverage: 0.887, oldRecall: 0.656, oldCoverage: 0.821, reach: "17/18" },
    { rate: "4x", recall: 0.577, coverage: 0.872, oldRecall: 0.568, oldCoverage: 0.796, reach: "18/18" },
  ],
  supersededShare: 0.248,
  supersedeSource: "docs/results/exp_c_summary.txt",
  secondsPer100k: 0.24,
};

// Experiment I lives on the exp/downstream-eval branch until it merges.
export const DOWNSTREAM = {
  source: `${REPO}/blob/exp/downstream-eval/docs/results/exp_i_downstream.txt`,
  sourceLabel: "docs/results/exp_i_downstream.txt (branch exp/downstream-eval)",
  transcripts: 20,
  model: "Sonnet 5.5",
  rows: [
    { level: "light", loosy: 0.84, summary: 0.96, loosyShare: 0.495, summaryShare: 0.345 },
    { level: "medium", loosy: 0.8, summary: 0.95, loosyShare: 0.333, summaryShare: 0.257 },
    { level: "heavy", loosy: 0.77, summary: 0.93, loosyShare: 0.254, summaryShare: 0.202 },
  ],
};

export const REPO_URL = REPO;

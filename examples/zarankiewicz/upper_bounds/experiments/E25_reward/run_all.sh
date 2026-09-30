#!/bin/sh
# E25 end to end (from examples/zarankiewicz/upper_bounds).  Read-only over cache/case_table_*.json.
# Steps 0-2 are incremental: re-running after new tables land (A1's *_pure_gt.json) only computes
# what is missing.  Lean: one gate process at a time.  SAT: none.
set -e
export ZAR_UB_NO_LLM=1
PY=${PY:-/Users/jaybhan/Downloads/openevolve/.venv/bin/python}
$PY experiments/E25_reward/cert_pool.py          # Farkas LP pool per table (2 processes, ~20 s)
$PY experiments/E25_reward/make_candidates.py    # candidates/*.py (deterministic)
$PY experiments/E25_reward/compute_masks.py      # Lean masks -> masks/*.json (~3-6 min per program cold)
$PY experiments/E25_reward/analyze.py            # results/score_matrix.{md,json}  (table labels)
$PY experiments/E25_reward/analyze.py --gt-labels  # results/score_matrix_gtlabels.*  (A1 ground truth, in memory)
$PY experiments/E25_reward/sweep.py              # results/sensitivity.{md,json}
$PY experiments/E25_reward/sweep.py --gt-labels  # results/sensitivity_gtlabels.*

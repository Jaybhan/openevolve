#!/bin/bash
set -e

BASE="examples/graph_isomorphism"

echo "=== Phase 1 ==="
python openevolve-run.py \
  "$BASE/initial_program.py" \
  "$BASE/evaluator.py" \
  --config "$BASE/config_phase_1.yaml" \
  --iterations 100

echo "=== Phase 2 ==="
python openevolve-run.py \
  "$BASE/openevolve_output/checkpoints/checkpoint_100/best_program.py" \
  "$BASE/evaluator.py" \
  --config "$BASE/config_phase_2.yaml" \
  --iterations 100

echo "=== Phase 3 ==="
python openevolve-run.py \
  "$BASE/openevolve_output/checkpoints/checkpoint_100/openevolve_output/checkpoints/checkpoint_100/best_program.py" \
  "$BASE/evaluator.py" \
  --config "$BASE/config_phase_3.yaml" \
  --iterations 200

echo "=== All phases complete ==="

#!/bin/bash
# E29: re-run the E26 dynamics study with the SHIPPED reward (v3 + E28 mirror penalty + E29 target-only
# closure, S2 suite, E27/E28 labels).  "V0 live" = the live evaluator's combined_score, untouched.
# 8 seed pairs x 150 iterations (culling active), 2 runs at a time.  Run from upper_bounds/.
cd "$(dirname "$0")/../.." || exit 1
export ZAR_UB_NO_LLM=1
PY=${PY:-/Users/jaybhan/Downloads/openevolve/.venv/bin/python}
D=experiments/E26_dynamics
run_one() {
  i=$1; name="E29_shipped_ladder_s$i"
  if [ -f "$D/runs/$name/summary.json" ]; then echo "skip $name"; return; fi
  $PY $D/run_dynamics.py --name "$name" --variant "V0 live" --config "$D/config_dyn.yaml" --seed $((26+i)) --db-seed $((42+i)) \
      --iterations 150 > "$D/logs/$name.log" 2>&1 && echo "done $name" || echo "FAIL $name"
}
export -f run_one; export PY D
printf '%s\n' 0 1 2 3 4 5 6 7 | xargs -P 2 -I{} bash -c 'run_one "$@"' _ {}
echo "ALL SHIPPED RUNS DONE $(date)"

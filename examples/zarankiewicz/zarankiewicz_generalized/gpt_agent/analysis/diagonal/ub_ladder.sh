#!/bin/zsh
# Descending UNSAT ladder for (17,17): targets 150 down to 141.
# Each target gets a budget; stop descending when one times out
# (all lower targets are at least as hard on the UNSAT side).
PY=/private/tmp/claude-501/-Users-jaybhan-Downloads-openevolve/00c08416-dd32-44da-ba6c-cd9781cf1162/scratchpad/zvenv/bin/python
cd "$(dirname "$0")"
for E in 150 149 148 147 146 145 144 143 142 141 140; do
  echo "=== target $E $(date) ==="
  $PY run_cell.py 17 17 $E --budget 5400 --tag d17_${E}
  st=$(${PY} -c "import json;print(json.load(open('results/d17_${E}.json'))['status'])")
  echo "status@$E: $st"
  if [ "$st" = "sat" ]; then echo "SAT at $E -- ladder stops (LB!)"; break; fi
  if [ "$st" = "timeout" ]; then echo "timeout at $E -- ladder stops"; break; fi
done

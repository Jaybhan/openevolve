#!/usr/bin/env bash
# E28: deepen every censored library SURVIVOR of the target tables to 50k conflicts; cases still open
# are relabelled by the E24 hardness model floored at 50k (EVALUATION.md 4.5: "direct 50k + model").
cd "$(dirname "$0")/../.."
export ZAR_UB_NO_LLM=1
PY=../../../.venv/bin/python
for cell in "12 18 109" "10 23 113" "11 23 124" "13 19 123" "16 17 134"; do
  set -- $cell
  echo "=== $1 x $2 w=$3 $(date) ==="
  $PY -m zar_ub deepen $1 $2 3 3 $3 --cap 50000 --jobs 12 --survivors 2>&1 | grep -E '^\{|"deepened"|"resolved_unsat"|"sat"|"still_censored"|"seconds"|hash|Traceback|Error' 
done
echo "ALL DONE $(date)"

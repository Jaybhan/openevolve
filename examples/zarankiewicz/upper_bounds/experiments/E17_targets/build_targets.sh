#!/usr/bin/env bash
# E17: build the design-default TARGET tables (censored labels, tan2022 facts, baseline masks).
cd "$(dirname "$0")/../.."
PY=../../../.venv/bin/python
for cell in "9 23 104" "13 17 117" "13 18 122" "15 17 133" "12 18 109"; do
  set -- $cell
  echo "=== target $1 x $2 w=$3 $(date) ==="
  ZAR_UB_NO_LLM=1 $PY -m zar_ub table $1 $2 3 3 $3 --trust tan2022 --mode censored --kind target --jobs 4 --baseline 2>&1 | tail -15
done
echo "ALL TARGETS DONE $(date)"

#!/usr/bin/env bash
# E17b: further TARGET tables (open or unreviewed-2026 cells), censored labels, tan2022 facts.
cd "$(dirname "$0")/../.."
PY=../../../.venv/bin/python
for cell in "10 22 111" "11 23 124" "16 17 134" "13 19 123" "10 23 113"; do
  set -- $cell
  echo "=== target $1 x $2 w=$3 $(date) ==="
  ZAR_UB_NO_LLM=1 $PY -m zar_ub table $1 $2 3 3 $3 --trust tan2022 --mode censored --kind target --jobs 3 --baseline 2>&1 | grep -E '"cases"|"survivors"|"build_seconds"|"survivor_status"|Error|Traceback' | head -8
done
echo "ALL TARGETS-2 DONE $(date)"

#!/bin/zsh
PY=/private/tmp/claude-501/-Users-jaybhan-Downloads-openevolve/00c08416-dd32-44da-ba6c-cd9781cf1162/scratchpad/zvenv/bin/python
cd /Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/zarankiewicz_generalized/gpt_agent/analysis/level_theory
for wave in 1 2 3 4 5 6; do
  echo "=== PINNED WAVE $wave ($(date +%H:%M)) ==="
  $PY fullpass.py | head -1
  $PY summarize_fullpass.py | head -1
  SLICE_BAND=1 $PY slice_runner.py 12 600
done
$PY fullpass.py | head -1
$PY summarize_fullpass.py
echo "=== PINNED PHASE COMPLETE ==="

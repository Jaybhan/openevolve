#!/bin/zsh
PY=/private/tmp/claude-501/-Users-jaybhan-Downloads-openevolve/00c08416-dd32-44da-ba6c-cd9781cf1162/scratchpad/zvenv/bin/python
cd /Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/zarankiewicz_generalized/gpt_agent/analysis/level_theory
for wave in 1 2 3 4 5 6 7 8; do
  echo "=== WAVE $wave ==="
  $PY fullpass.py | head -1
  case $wave in
    1|2) MM=10; TL=600;;
    3|4) MM=11; TL=600;;
    5|6) MM=12; TL=900;;
    *)   MM=16; TL=900;;
  esac
  $PY slice_runner.py $MM $TL
done
$PY fullpass.py | head -2
echo "=== ITERATION COMPLETE ==="

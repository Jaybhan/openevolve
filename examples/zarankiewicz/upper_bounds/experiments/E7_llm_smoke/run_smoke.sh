#!/usr/bin/env bash
# E7: tiny OpenEvolve smoke run on OpenRouter (cheap models, few iterations).
# Usage: experiments/E7_llm_smoke/run_smoke.sh <iterations> [config.yaml]
set -euo pipefail
UB="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$(cd "$UB/../../.." && pwd)"
ITER="${1:-3}"
CFG="${2:-$UB/config.yaml}"
export OPENAI_API_KEY="$(cat "$UB/.openrouter_key")"
export OPENAI_API_BASE="https://openrouter.ai/api/v1"
OUT="$UB/experiments/E7_llm_smoke/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
python "$UB/experiments/cost.py" "E7 start ($ITER iters, $(basename "$CFG"))"
cd "$UB"
"$ROOT/.venv/bin/python" "$ROOT/openevolve-run.py" "$UB/initial_program.py" "$UB/evaluator.py" \
  --config "$CFG" --iterations "$ITER" --output "$OUT" 2>&1 | tee "$OUT/run.log"
python "$UB/experiments/cost.py" "E7 end"

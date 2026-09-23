#!/usr/bin/env bash
# T-6 (design §10.1): OpenEvolve over the REAL OpenAI client path against tools/stub_llm.py.
# Usage: experiments/no_llm/run_stub.sh [iterations=10] [port=8123] [output=experiments/no_llm/run_stub]
set -euo pipefail
UB="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$(cd "$UB/../../.." && pwd)"
PY="$ROOT/.venv/bin/python"
ITER="${1:-10}"; PORT="${2:-8123}"; OUT="${3:-$UB/experiments/no_llm/run_stub}"
CFG="$UB/experiments/no_llm/config_stub.yaml"
[ -f "$CFG" ] || "$PY" "$UB/experiments/no_llm/make_config_stub.py" --port "$PORT"
sed -i '' "s#http://127.0.0.1:[0-9]*/v1#http://127.0.0.1:$PORT/v1#" "$CFG"
rm -rf "$OUT"; mkdir -p "$OUT"
"$PY" "$UB/tools/stub_llm.py" --port "$PORT" --bank "$UB/tests/snippets" --log "$OUT/stub_requests.jsonl" --quiet &
STUB=$!
trap 'kill $STUB 2>/dev/null || true' EXIT
sleep 1
curl -s "http://127.0.0.1:$PORT/health" >/dev/null || { echo "stub not up"; exit 1; }
cd "$UB"
OPENAI_API_KEY=x ZAR_UB_NO_LLM=1 "$PY" "$ROOT/openevolve-run.py" initial_program.py evaluator.py \
  --config "$CFG" --iterations "$ITER" --output "$OUT" 2>&1 | tee "$OUT/run.log"
echo "--- stub requests: $(wc -l < "$OUT/stub_requests.jsonl") ; checkpoints: $(ls "$OUT/checkpoints" 2>/dev/null | tr '\n' ' ')"

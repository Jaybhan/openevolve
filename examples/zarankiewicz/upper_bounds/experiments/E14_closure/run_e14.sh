#!/usr/bin/env bash
# E14: Tier-1 closure seam — T-7 (9,9,50 pure), T-8 (three Tan-conditional cells), kernel scaling, audit.
# Run from examples/zarankiewicz/upper_bounds.  Needs Lean 4.34 + Mathlib oleans; never runs `lake build`.
set -euo pipefail
cd "$(dirname "$0")/../.."
export ZAR_UB_NO_LLM=1
PY=${PY:-../../../.venv/bin/python}
E=experiments/E14_closure
mkdir -p "$E/scaling"

echo "== T-7: (9,9,50) pure, Tier-1 (kernel decide)"
$PY -m zar_ub closure 9 9 3 3 50 --pure --timeout 2400 | tee "$E/t7_9_9_50_pure.log" | grep '^\[closure\]'
cp cache/certs/m9_n9_s3_t3_w50/closure_report.md "$E/closure_report_m9_n9_s3_t3_w50.md"

echo "== T-8: trust tan2022, zero survivors"
for c in "10 21 107" "11 19 107" "11 20 112"; do
  read -r m n w <<<"$c"
  $PY -m zar_ub closure "$m" "$n" 3 3 "$w" --trust tan2022 --timeout 2400 | tee "$E/t8_${m}_${n}_${w}.log" | grep '^\[closure\]'
  cp "cache/certs/m${m}_n${n}_s3_t3_w${w}/closure_report.md" "$E/closure_report_m${m}_n${n}_s3_t3_w${w}.md"
done

echo "== kernel scaling on the pure training ladder (no certificates: 'not established' is expected)"
for c in "9 10 55" "10 11 65" "11 11 70" "12 12 81" "13 13 93"; do
  read -r m n w <<<"$c"
  $PY -m zar_ub closure "$m" "$n" 3 3 "$w" --pure --timeout 2400 --out "$E/scaling" > "$E/scaling/m${m}_n${n}_w${w}.log" 2>&1 || true
  grep '^\[closure\] check' "$E/scaling/m${m}_n${n}_w${w}.log"
done

echo "== Tier-1n fallback (named _native axiom) and a candidate OR-ed into the library"
$PY -m zar_ub closure 9 9 3 3 50 --pure --tier 1n --out "$E/scaling" | grep '^\[closure\] check'
mv "$E/scaling/Z_9_9_50.lean" "$E/scaling/Z_9_9_50_tier1n.lean"
mv "$E/scaling/closure_report_m9_n9_s3_t3_w50.md" "$E/scaling/closure_report_m9_n9_s3_t3_w50_tier1n.md"
printf 'def candidate (P : Params) : Prune P := deficit P\n' > "$E/scaling/cand_deficit.lean"
$PY -m zar_ub closure 9 9 3 3 50 --pure --lean "$E/scaling/cand_deficit.lean" --out "$E/scaling" | grep '^\[closure\] check'
mv "$E/scaling/Z_9_9_50.lean" "$E/scaling/Z_9_9_50_with_candidate.lean"
rm -f "$E/scaling/closure_report_m9_n9_s3_t3_w50.md"

echo "== audit: axioms of every Closure.lean declaration + enumerator cross-check vs partitions.py"
$PY "$E/audit_closure.py" | tail -16

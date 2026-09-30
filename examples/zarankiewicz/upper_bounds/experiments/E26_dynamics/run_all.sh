#!/bin/bash
# E26: every configuration x seed pairs (db 42+i, mutator 26+i), 2 runs at a time.
# Primary: ITER=50 (12 seeds).  Extension: ITER=150 PREFIX=L150_ SEEDS="0 1 2 3 4 5 6 7" (culling starts > 60 programs).
# Run from examples/zarankiewicz/upper_bounds with ZAR_UB_NO_LLM=1.  Seed pair 0 = the headline run.
cd "$(dirname "$0")/../.." || exit 1
export ZAR_UB_NO_LLM=1
PY=${PY:-/Users/jaybhan/Downloads/openevolve/.venv/bin/python}
D=experiments/E26_dynamics
CONFIGS=(
  "v0_ladder|V0 live|$D/config_dyn.yaml"
  "vr_ladder|VR recommended|$D/config_dyn.yaml"
  "v345_ladder|V3+V4+V5|$D/config_dyn.yaml"
  "v0s1_ladder|V0/S1 suite only|$D/config_dyn.yaml"
  "v0_conc|V0 live|$D/config_dyn_conc.yaml"
  "vr_conc|VR recommended|$D/config_dyn_conc.yaml"
)
jobs_list=()
ITER=${ITER:-50}; PREFIX=${PREFIX:-}; SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11"}
for i in $SEEDS; do
  for c in "${CONFIGS[@]}"; do
    IFS='|' read -r name variant cfg <<< "$c"
    jobs_list+=("${PREFIX}${name}_s$i|$variant|$cfg|$((26+i))|$((42+i))")
  done
done
run_one() {
  IFS='|' read -r name variant cfg ms dbs <<< "$1"
  if [ -f "$D/runs/$name/summary.json" ] && [ -f "$D/runs/$name/run_meta.json" ]; then echo "skip $name"; return; fi
  $PY $D/run_dynamics.py --name "$name" --variant "$variant" --config "$cfg" --seed "$ms" --db-seed "$dbs" --iterations "$ITER" \
      > "$D/logs/$name.log" 2>&1 && echo "done $name" || echo "FAIL $name"
}
export -f run_one; export PY D ITER
printf '%s\n' "${jobs_list[@]}" | xargs -P 2 -I{} bash -c 'run_one "$@"' _ {}

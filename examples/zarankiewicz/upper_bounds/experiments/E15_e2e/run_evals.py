#!/usr/bin/env python
"""E15 end-to-end verifier: evaluate a list of candidate programs with evaluator.evaluate()
and record metrics + selected artifacts + wall time as JSON.

Usage (from upper_bounds/, ZAR_UB_NO_LLM=1):
    python experiments/E15_e2e/run_evals.py --out experiments/E15_e2e/out/evals.json FILE.py [FILE.py ...]
    python experiments/E15_e2e/run_evals.py --repeat 2 initial_program.py   # determinism check
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, UB)
os.chdir(UB)
os.environ.setdefault("ZAR_UB_NO_LLM", "1")

import evaluator  # noqa: E402

KEYS = ["combined_score", "lean_ladder", "lean_ok", "sound_battery", "proven_gain", "empirical_gain",
        "schema_gain", "target_gain", "tail_gain", "agreement", "survivors_left", "lean_partial",
        "n_new_prunes", "gate_cache_hit", "kill_novelty", "eval_seconds"]
ART_KEYS = ["lean_errors", "lean_cond", "lean_axioms", "schema_note", "schema_error", "battery",
            "counterexamples", "PIPELINE_BUG", "lean_holes"]
TIMING_KEYS = {"eval_seconds", "lean_seconds", "gate_seconds", "stage1_seconds", "seconds"}


def run_one(path):
    t0 = time.time()
    r = evaluator.evaluate(path)
    wall = time.time() - t0
    metrics = r.metrics if hasattr(r, "metrics") else r
    arts = r.artifacts if hasattr(r, "artifacts") else {}
    return {"path": path, "wall": round(wall, 2),
            "metrics": {k: metrics.get(k) for k in KEYS if k in metrics},
            "all_metrics": metrics,
            "artifacts": {k: (str(v)[:600] if v is not None else None) for k, v in arts.items() if k in ART_KEYS},
            "artifact_keys": sorted(arts.keys())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("programs", nargs="+")
    ap.add_argument("--out", default=os.path.join(HERE, "out", "evals.json"))
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    results = []
    for p in a.programs:
        for i in range(a.repeat):
            res = run_one(p)
            res["repeat"] = i
            results.append(res)
            print(f"{p} [{i}] wall={res['wall']}s {json.dumps(res['metrics'])}", flush=True)
    if a.repeat > 1:
        for p in a.programs:
            runs = [r for r in results if r["path"] == p]
            base = {k: v for k, v in runs[0]["all_metrics"].items() if k not in TIMING_KEYS}
            for r in runs[1:]:
                other = {k: v for k, v in r["all_metrics"].items() if k not in TIMING_KEYS}
                diff = {k: (base.get(k), other.get(k)) for k in set(base) | set(other) if base.get(k) != other.get(k)}
                print(f"DETERMINISM {p}: {'identical' if not diff else 'DIFF ' + json.dumps(diff)}", flush=True)
                results.append({"path": p, "determinism_diff": diff})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1, default=str)


if __name__ == "__main__":
    main()

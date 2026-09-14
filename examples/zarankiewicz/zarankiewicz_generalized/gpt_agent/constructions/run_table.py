"""Run the engine over all 161 proven-exact (3,3) cells; write results_33.csv.

Ground truth comes ONLY from the harness snapshot copy of the evaluator
(gpt_agent/harness/evaluator_snapshot.py) — the live evaluator.py is never
imported (its module-level path anchors would point the SOTA/log files at the
live experiment).  Importing the snapshot is side-effect free at import time
(it only writes when evaluate()/score_graph() run).
"""

import csv
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GPT = os.path.dirname(HERE)

sys.path.insert(0, HERE)
import zarankiewicz as Z  # noqa: E402


def load_snapshot():
    path = os.path.join(GPT, "harness", "evaluator_snapshot.py")
    spec = importlib.util.spec_from_file_location("evaluator_snapshot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ev = load_snapshot()
    cells = sorted(ev.KST_EXACT_VALUE.keys(), key=lambda c: (c[0] + c[1], c))
    rows = []
    t0 = time.time()
    slow = []
    print("building full table (with halo + extend/shrink sweeps)...")
    Z.build_table(16, 23, 3, 3, halo=1, passes=2)
    print(f"table built in {time.time()-t0:.1f}s")
    for (m, n) in cells:
        z = ev.KST_EXACT_VALUE[(m, n)]
        tc = time.time()
        A, prov, fams = Z.construct(m, n, 3, 3)
        dt = time.time() - tc
        e = int(A.sum())
        ok_mine = Z.verify_kst_free(A, 3, 3)
        ok_ref = ev.count_kst_violations(A, 3, 3) == 0
        assert ok_mine == ok_ref, f"verifier mismatch at {(m, n)}"
        rows.append({
            "m": m, "n": n, "exact": z, "ours": e,
            "family": prov, "gap": z - e, "valid": ok_mine,
            "families": fams, "secs": round(dt, 3),
        })
        if dt > 1.0:
            slow.append(((m, n), round(dt, 2)))

    n_exact = sum(1 for r in rows if r["gap"] == 0 and r["valid"])
    n_valid = sum(1 for r in rows if r["valid"])
    total_gap = sum(r["gap"] for r in rows if r["valid"])
    print(f"cells={len(rows)} valid={n_valid} exact={n_exact} "
          f"total_missing_edges={total_gap} time={time.time()-t0:.1f}s")
    print("slowest cells:", slow[-8:] if slow else "none > 1s")

    misses = [(r["m"], r["n"], r["gap"], r["family"]) for r in rows
              if r["gap"] != 0 or not r["valid"]]
    for mm, nn, g, fam in misses:
        print(f"  MISS {mm}x{nn}: -{g}  ({fam[:70]})")

    with open(os.path.join(HERE, "results_33.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "n", "exact", "ours", "family", "gap", "valid"])
        for r in rows:
            w.writerow([r["m"], r["n"], r["exact"], r["ours"],
                        r["family"], r["gap"], int(r["valid"])])
    print("wrote results_33.csv")


if __name__ == "__main__":
    main()

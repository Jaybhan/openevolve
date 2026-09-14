"""Frontier cells: exact MILP on (3,3) cells OUTSIDE the proven table.

Targets:
  A. Gap-band cells between Tan's exactness frontier (n<=23) and the
     Roman/Tan equality window n >= 2C(m,3) - 3*T33(m):
       (7,24); (9,42..47); (11,82..89); (12,110..115)   [m=15 band deferred]
  B. Literature-discrepancy cells: (12,17) [CRWR claim 103 exact],
     (13,17) [CRWR UB 110 vs DGH 116], (11,22) [BNL 121, absent from suite].
No published-value input reaches the solver; comparisons are annotations.
"""
import json
import os
import sys
import time
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilp_solver import solve_cell, verify  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NOTES = {
    (7, 24): "gap band row 7 (predict 87 = 3n+T33(7))",
    (9, 42): "gap band row 9", (9, 43): "gap band row 9",
    (9, 44): "gap band row 9", (9, 45): "gap band row 9",
    (9, 46): "gap band row 9", (9, 47): "gap band row 9",
    (11, 82): "gap band row 11", (11, 83): "gap band row 11",
    (11, 84): "gap band row 11", (11, 85): "gap band row 11",
    (11, 86): "gap band row 11", (11, 87): "gap band row 11",
    (11, 88): "gap band row 11", (11, 89): "gap band row 11",
    (12, 110): "gap band row 12", (12, 111): "gap band row 12",
    (12, 112): "gap band row 12", (12, 113): "gap band row 12",
    (12, 114): "gap band row 12", (12, 115): "gap band row 12",
    (12, 17): "CRWR 2016 claim: 103 exact (discrepancy vs Tan/DGH)",
    (13, 17): "CRWR UB 110 vs DGH 116",
    (11, 22): "BNL 2026: 121 exact (missing from evaluator suite)",
}
ORDER = [(7, 24), (12, 17), (11, 22), (13, 17),
         (9, 42), (9, 43), (9, 44), (9, 45), (9, 46), (9, 47),
         (11, 82), (11, 83), (11, 84), (11, 85), (11, 86), (11, 87),
         (11, 88), (11, 89),
         (12, 110), (12, 111), (12, 112), (12, 113), (12, 114), (12, 115)]

tl = float(sys.argv[1]) if len(sys.argv) > 1 else 1200.0
out_path = os.path.join(HERE, "ilp_frontier.jsonl")
wdir = os.path.join(HERE, "witnesses")
os.makedirs(wdir, exist_ok=True)

for (m, n) in ORDER:
    t0 = time.time()
    edges, blocks, status = solve_cell(m, n, time_limit=tl)
    dt = time.time() - t0
    if edges is None:
        rec = {"m": m, "n": n, "status": f"UNRESOLVED({status})",
               "seconds": round(dt, 1), "note": NOTES[(m, n)]}
        print(f"z({m},{n}) UNRESOLVED [{dt:.0f}s]  ({NOTES[(m,n)]})", flush=True)
    else:
        okw = verify(blocks, m, n, edges)
        rec = {"m": m, "n": n, "z_ilp": edges, "witness_valid": okw,
               "seconds": round(dt, 1), "note": NOTES[(m, n)]}
        print(f"z({m},{n}) = {edges}  witness_valid={okw} [{dt:.0f}s]  "
              f"({NOTES[(m,n)]})", flush=True)
        if okw:
            with open(os.path.join(wdir, f"w_{m}x{n}.json"), "w") as f:
                json.dump({"m": m, "n": n, "edges": edges,
                           "blocks": [list(b) for b in blocks]}, f)
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")

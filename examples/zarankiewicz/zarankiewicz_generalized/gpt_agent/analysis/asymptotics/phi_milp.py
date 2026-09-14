"""Task 3 solver budget (2 runs, both here): exact D2(m,5,3) at
m = 11 and m = 12 — the two supply points straddling the phi-step c = 1:
  c(11,5) = 5/11^(2/3) = 1.011,  c(12,5) = 5/12^(2/3) = 0.954.
Known brackets (supply_status.csv): D5(11) in [23,31], D5(12) in [30,38].

Model: max sum x_B, x_B in {0,1,2} over all C(m,5) pentads,
       sum_{B superset T} x_B <= 2 for every triple T.
HiGHS MILP via scipy. Timeout still yields rigorous bounds
(incumbent = LB, dual bound = UB).
"""
import numpy as np
from itertools import combinations
from math import comb
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def solve_D(m, w, time_limit=1500):
    blocks = list(combinations(range(m), w))
    triples = list(combinations(range(m), 3))
    tidx = {t: i for i, t in enumerate(triples)}
    A = lil_matrix((len(triples), len(blocks)), dtype=np.int8)
    for j, b in enumerate(blocks):
        for t in combinations(b, 3):
            A[tidx[t], j] = 1
    A = A.tocsc()
    res = milp(c=-np.ones(len(blocks)),
               constraints=LinearConstraint(A, -np.inf, 2),
               integrality=np.ones(len(blocks)),
               bounds=Bounds(0, 2),
               options=dict(time_limit=time_limit, mip_rel_gap=0.0))
    lb = int(round(-res.fun)) if res.fun is not None else None
    # scipy exposes the dual bound via res.mip_dual_bound (minimization)
    ub = None
    if hasattr(res, "mip_dual_bound") and res.mip_dual_bound is not None:
        ub = int(np.floor(-res.mip_dual_bound + 1e-6))
    return dict(m=m, w=w, status=int(res.status),
                message=str(res.message), incumbent_LB=lb, dual_UB=ub,
                exact=(res.status == 0))


if __name__ == "__main__":
    # POST-MORTEM NOTE (2026-07-30): the session's two budgeted runs
    # (m = 11, 12; 1500 s limits) were killed by the task harness at
    # ~40 min wall with stdout buffered and this JSON written only at
    # the very end — all results LOST. Budget consumed; brackets
    # D5(11) in [23,31], D5(12) in [30,38] stand. The loop below now
    # checkpoints after EVERY solve (incremental JSON + flushed
    # stdout) so a future session cannot repeat the failure mode.
    out = []
    for m in (11, 12):
        t0 = time.time()
        r = solve_D(m, 5)
        r["seconds"] = round(time.time() - t0, 1)
        out.append(r)
        print(json.dumps(r), flush=True)
        with open(os.path.join(HERE, "phi_milp_results.json"), "w") as f:
            json.dump(out, f, indent=1)

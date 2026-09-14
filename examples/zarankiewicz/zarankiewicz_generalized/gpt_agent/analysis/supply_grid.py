"""2D/3D supply grid: S_m(k5, k6, k7) = max #quads coexisting with the given
heavy profile under 2-fold triple capacity. Writes supply_S<m>.csv.

This is the PRIMARY computation of the below-window program: with the full
grid, z(m,n;3,3) = 3n + max over supply-feasible profiles of
(k4 + 2k5 + 3k6 + 4k7) subject to columns <= n, budget, triple-fill —
pure arithmetic (Lemma A shell), no per-cell ILP needed.
"""
import json
import os
import sys
import time
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def S(m, k5, k6=0, k7=0, time_limit=900.0):
    B = 2 * comb(m, 3)
    if 10 * k5 + 20 * k6 + 35 * k7 > B:
        return "overbudget"
    weights = [4, 5] + ([6] if k6 or True else []) + ([7] if k7 else [])
    types, wlist = [], []
    for w in sorted(set([4, 5] + ([6] if k6 else []) + ([7] if k7 else []))):
        for b in combinations(range(m), w):
            types.append(b)
            wlist.append(w)
    tris = {t: i for i, t in enumerate(combinations(range(m), 3))}
    ntri = len(tris)
    nv = len(types)
    nfix = (1 if True else 0) + (1 if k6 else 0) + (1 if k7 else 0)

    rows = ntri + 1 + (1 if k6 else 0) + (1 if k7 else 0) + (m - 1)
    A = lil_matrix((rows, nv))
    r5 = ntri
    r6 = ntri + 1 if k6 else None
    r7 = (ntri + 1 + (1 if k6 else 0)) if k7 else None
    sym0 = ntri + 1 + (1 if k6 else 0) + (1 if k7 else 0)
    for j, b in enumerate(types):
        for t in combinations(b, 3):
            A[tris[t], j] = 1
        if wlist[j] == 5:
            A[r5, j] = 1
        if k6 and wlist[j] == 6:
            A[r6, j] = 1
        if k7 and wlist[j] == 7:
            A[r7, j] = 1
        for r in range(m - 1):
            v = (1 if r in b else 0) - (1 if (r + 1) in b else 0)
            if v:
                A[sym0 + r, j] = v
    lb = np.zeros(rows)
    ub = np.full(rows, 2.0)
    lb[r5] = ub[r5] = k5
    if k6:
        lb[r6] = ub[r6] = k6
    if k7:
        lb[r7] = ub[r7] = k7
    for r in range(m - 1):
        lb[sym0 + r] = 0.0
        ub[sym0 + r] = np.inf

    c = np.array([-1.0 if w == 4 else 0.0 for w in wlist])
    res = milp(c=c, constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(nv),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status == 2:  # infeasible
        return "infeasible"
    if res.status != 0:
        return None
    return int(round(-res.fun))


if __name__ == "__main__":
    m = int(sys.argv[1])
    k5max = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    k6max = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    HERE = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(HERE, f"supply_S{m}.csv")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f.readlines()[1:]:
                p = line.strip().split(",")
                if len(p) >= 3:
                    done.add((int(p[0]), int(p[1])))
    else:
        with open(out, "w") as f:
            f.write("k5,k6,S\n")
    for k6 in range(k6max + 1):
        for k5 in range(k5max + 1):
            if (k5, k6) in done:
                continue
            t0 = time.time()
            v = S(m, k5, k6)
            dt = time.time() - t0
            print(f"S_{m}(k5={k5},k6={k6}) = {v} [{dt:.0f}s]", flush=True)
            with open(out, "a") as f:
                f.write(f"{k5},{k6},{v}\n")

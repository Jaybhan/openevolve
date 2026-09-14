"""Task 1c (fuller 3D probe): max val subject to #cols <= c AND slots <= s,
on a small (c, s) grid, to trace the slots-dimension of the Pareto surface
where the penalty term of the exact identity can bite.

Usage: probe_cs.py m c s1 s2 ...   (one MILP per s)
Appends to probe_cs_m{m}.jsonl.
"""
import json
import os
import sys
from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))


def solve(m, c, scap, time_limit=600.0):
    types = []
    for w in range(4, m + 1):
        types.extend(combinations(range(m), w))
    tidx = {t: i for i, t in enumerate(combinations(range(m), 3))}
    ntri, nv = len(tidx), len(types)
    P = 2 * comb(m - 1, 2)
    R = P // 3
    J = (m * R) // 4 - (2 if (m % 4 == 3 and m % 3 != 0) else 0)
    pairs = list(combinations(range(m), 2))
    pidx = {p: i for i, p in enumerate(pairs)}
    nrows = ntri + 2 + len(pairs) + m + 2 + (m - 1)
    A = lil_matrix((nrows, nv))
    r_col, r_slot = ntri, ntri + 1
    r_pair0 = ntri + 2
    r_pt0 = r_pair0 + len(pairs)
    r_wb, r_gv = r_pt0 + m, r_pt0 + m + 1
    r_sym0 = r_gv + 1
    for j, b in enumerate(types):
        w = len(b)
        for t in combinations(b, 3):
            A[tidx[t], j] = 1
        A[r_col, j] = 1
        A[r_slot, j] = comb(w, 3)
        for p in combinations(b, 2):
            A[r_pair0 + pidx[p], j] = w - 2
        for x in b:
            A[r_pt0 + x, j] = w - 3
        A[r_wb, j] = w * (w - 3)
        A[r_gv, j] = w - 3
        for r in range(m - 1):
            v = (1 if r in b else 0) - (1 if (r + 1) in b else 0)
            if v:
                A[r_sym0 + r, j] = v
    lb, ub = np.zeros(nrows), np.zeros(nrows)
    ub[:ntri] = 2.0
    ub[r_col] = float(c)
    ub[r_slot] = float(scap)
    for i in range(len(pairs)):
        ub[r_pair0 + i] = float(2 * (m - 2))
    for x in range(m):
        ub[r_pt0 + x] = float(R)
    ub[r_wb], ub[r_gv] = float(m * R), float(J)
    for r in range(m - 1):
        ub[r_sym0 + r] = np.inf
    cobj = np.array([-(len(b) - 3.0) for b in types])
    res = milp(c=cobj, constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(nv),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.x is None:
        return None, None, f"UNRESOLVED({res.status})"
    xv = np.round(res.x).astype(int)
    blocks = []
    for j, k in enumerate(xv):
        blocks.extend([types[j]] * int(k))
    cov = Counter()
    for b in blocks:
        for t in combinations(b, 3):
            cov[t] += 1
    assert all(v <= 2 for v in cov.values())
    prof = Counter(len(b) for b in blocks)
    return int(round(-res.fun)), {str(w): prof[w] for w in sorted(prof)}, \
        ("PROVEN" if res.status == 0 else "LB(timeout)")


if __name__ == "__main__":
    m = int(sys.argv[1])
    c = int(sys.argv[2])
    out = os.path.join(HERE, f"probe_cs_m{m}.jsonl")
    for s in [int(x) for x in sys.argv[3:]]:
        val, prof, status = solve(m, c, s)
        rec = {"m": m, "c": c, "slots_cap": s, "val": val, "profile": prof,
               "status": status}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

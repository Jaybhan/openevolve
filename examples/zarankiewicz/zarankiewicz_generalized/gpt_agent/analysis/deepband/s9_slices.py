"""Resolve open S_9(k5) brackets (theorems.md: S9(2) in [33,35],
S9(5) in [25,28]) with the deep-band cut set: max #quads s.t. exactly k5
pentads, weights {4,5} only, per-triple <= 2, plus proven-valid pair cuts,
point floors, weighted budget, symmetry breaking. Appends s9_slices.jsonl.
"""
import json
import os
import sys
import time
from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
M = 9


def solve_slice(k5, time_limit=3000.0, quad_lb=0, k6=0):
    m = M
    types = list(combinations(range(m), 4)) + list(combinations(range(m), 5))
    if k6:
        types += list(combinations(range(m), 6))
    tidx = {t: i for i, t in enumerate(combinations(range(m), 3))}
    ntri, nv = len(tidx), len(types)
    R = (2 * comb(m - 1, 2)) // 3
    pairs = list(combinations(range(m), 2))
    pidx = {p: i for i, p in enumerate(pairs)}
    nrows = ntri + 2 + len(pairs) + m + 1 + (m - 1) + 1
    A = lil_matrix((nrows, nv))
    r_k5 = ntri
    r_k6 = ntri + 1
    r_pair0 = ntri + 2
    r_pt0 = r_pair0 + len(pairs)
    r_wb = r_pt0 + m
    r_sym0 = r_wb + 1
    r_qlb = r_sym0 + (m - 1)
    for j, b in enumerate(types):
        w = len(b)
        for t in combinations(b, 3):
            A[tidx[t], j] = 1
        if w == 5:
            A[r_k5, j] = 1
        elif w == 6:
            A[r_k6, j] = 1
        else:
            A[r_qlb, j] = 1
        for p in combinations(b, 2):
            A[r_pair0 + pidx[p], j] = w - 2
        for x in b:
            A[r_pt0 + x, j] = w - 3
        A[r_wb, j] = w * (w - 3)
        for r in range(m - 1):
            v = (1 if r in b else 0) - (1 if (r + 1) in b else 0)
            if v:
                A[r_sym0 + r, j] = v
    lb, ub = np.zeros(nrows), np.zeros(nrows)
    ub[:ntri] = 2.0
    lb[r_k5] = ub[r_k5] = float(k5)
    lb[r_k6] = ub[r_k6] = float(k6)
    for i in range(len(pairs)):
        ub[r_pair0 + i] = float(2 * (m - 2))
    for x in range(m):
        ub[r_pt0 + x] = float(R)
    ub[r_wb] = float(m * R)
    for r in range(m - 1):
        ub[r_sym0 + r] = np.inf
    lb[r_qlb], ub[r_qlb] = float(quad_lb), np.inf
    cobj = np.array([-1.0 if len(b) == 4 else 0.0 for b in types])
    res = milp(c=cobj, constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(nv),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status == 2:
        return None, "INFEASIBLE", None
    if res.x is None:
        return None, f"UNRESOLVED({res.status})", None
    xv = np.round(res.x).astype(int)
    blocks = []
    for j, k in enumerate(xv):
        blocks.extend([types[j]] * int(k))
    cov = Counter()
    for b in blocks:
        for t in combinations(b, 3):
            cov[t] += 1
    assert all(v <= 2 for v in cov.values())
    S = sum(1 for b in blocks if len(b) == 4)
    return S, ("PROVEN" if res.status == 0 else "LB(timeout)"), \
        [list(b) for b in blocks]


if __name__ == "__main__":
    # args: k5:quad_lb[:k6[:time_limit]]  e.g. "11:13" "8:16:1" "5:28:0:2400"
    out = os.path.join(HERE, "s9_slices.jsonl")
    for arg in sys.argv[1:]:
        parts = (arg.split(":") + ["0", "0", "3000"])[:4]
        k5, qlb, k6, tl = (int(parts[0]), int(parts[1]), int(parts[2]),
                           float(parts[3]))
        t0 = time.time()
        S, status, blocks = solve_slice(k5, time_limit=tl, quad_lb=qlb, k6=k6)
        rec = {"m": 9, "k5": k5, "k6": k6, "quad_lb": qlb, "S": S,
               "status": status, "seconds": round(time.time() - t0, 1),
               "blocks": blocks}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"S_9(k5={k5},k6={k6}|quads>={qlb}) = {S} [{status}] "
              f"({rec['seconds']}s)", flush=True)

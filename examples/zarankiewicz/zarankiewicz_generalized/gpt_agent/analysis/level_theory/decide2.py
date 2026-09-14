"""Phase-2 decisions with cell-specific FORCED structure (all added
constraints are proven consequences of b — see spectrum.md; hence INFEAS
still proves D2 < b, FEAS gives a witness).

(10,5,22): L=20 forces ell_x = 6 at every point (r_x = 11) and the unique
  per-point pair-leave pattern {4,1^8}: the leave's 4-pairs form a perfect
  matching, WLOG {i, i+5}: pair-coverages lam(i,i+5) = 4, else 5. FULL
  pair-regularity known -> massive symmetry reduction.
(13,6,26): L=52 forces ell_x = 12 (r_x = 12) and ALL pair-leaves = 2:
  lam_xy = 5 for every pair (a 2-(13,6,5) that is a 2-fold 3-packing).
(12,7,12): L=20 forces ell_x = 5 (r_x = 7); pair-leaves in {0,5,10}.
(11,5,32): L=10 leave = K5(3) once (proven unique shape): WLOG the 5-set
  {6..10}: triples inside it covered EXACTLY once, all others EXACTLY
  twice; every point-leave: 6 for x in the 5-set... ell_x = C(4,2)=6 on
  hole, 0 off-hole -> r_x = 14 on hole, 15 off. Fully determined RHS.
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

HERE = os.path.dirname(os.path.abspath(__file__))
WDIR = os.path.join(HERE, "witnesses")


def build(m, w, b, tri_eq=None, point_eq=None, pair_eq=None, time_limit=1800):
    """tri_eq: dict T->exact coverage; point_eq: dict x->r_x; pair_eq: dict
    pair->lam. Unspecified: coverage <=2, others free (within Johnson)."""
    types = list(combinations(range(m), w))
    tris = {t: i for i, t in enumerate(combinations(range(m), 3))}
    pairs = {p: i for i, p in enumerate(combinations(range(m), 2))}
    nT, nP, nV = len(tris), len(pairs), len(types)
    rows = nT + m + nP + 1
    A = lil_matrix((rows, nV))
    lb = np.zeros(rows)
    ub = np.zeros(rows)
    for j, blk in enumerate(types):
        for t in combinations(blk, 3):
            A[tris[t], j] = 1
        for x in blk:
            A[nT + x, j] = 1
        for p in combinations(blk, 2):
            A[nT + m + pairs[p], j] = 1
        A[nT + m + nP, j] = 1
    for t, i in tris.items():
        if tri_eq and t in tri_eq:
            lb[i] = ub[i] = tri_eq[t]
        else:
            lb[i], ub[i] = 0, 2
    lam2 = (2 * (m - 2)) // (w - 2)
    r3 = ((m - 1) * lam2) // (w - 1)
    for x in range(m):
        if point_eq and x in point_eq:
            lb[nT + x] = ub[nT + x] = point_eq[x]
        else:
            lb[nT + x], ub[nT + x] = 0, r3
    for p, i in pairs.items():
        if pair_eq and p in pair_eq:
            lb[nT + m + i] = ub[nT + m + i] = pair_eq[p]
        else:
            lb[nT + m + i], ub[nT + m + i] = 0, lam2
    lb[nT + m + nP] = ub[nT + m + nP] = b
    t0 = time.time()
    res = milp(c=np.zeros(nV), constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(nV),
               options={"time_limit": time_limit})
    dt = time.time() - t0
    if res.status == 0:
        x = np.round(res.x).astype(int)
        blocks = []
        for j, c in enumerate(x):
            blocks.extend([list(types[j])] * int(c))
        from collections import Counter
        cov = Counter()
        for blk in blocks:
            for t in combinations(sorted(blk), 3):
                cov[t] += 1
        ok = (len(blocks) == b and (not cov or max(cov.values()) <= 2))
        json.dump({"m": m, "w": w, "count": b, "blocks": blocks},
                  open(os.path.join(WDIR, f"D2_{m}_{w}_b{b}.json"), "w"))
        return ("FEAS" if ok else "FEAS-BAD"), dt
    return ("INFEAS" if res.status == 2 else "TIMEOUT"), dt


def main():
    logp = os.path.join(HERE, "decisions.csv")
    jobs = sys.argv[1:] or ["10:5:22", "11:5:32", "13:6:26", "12:7:12"]
    for job in jobs:
        m, w, b = (int(t) for t in job.split(":"))
        kw = {}
        if (m, w, b) == (10, 5, 22):
            M = [(i, i + 5) for i in range(5)]
            kw["point_eq"] = {x: 11 for x in range(10)}
            kw["pair_eq"] = {p: (4 if p in M else 5)
                             for p in combinations(range(10), 2)}
        elif (m, w, b) == (11, 5, 32):
            # PROVEN forced structure (spectrum.md 3.1): leave weight 10 has
            # support exactly a 5-set with all point-leaves 6 (WLOG {6..10});
            # off-hole triples exactly covered twice is IMPLIED by r_x = 15
            # off-hole; the hole leave SHAPE is left free (only degrees forced).
            hole = set(range(6, 11))
            kw["tri_eq"] = {t: 2 for t in combinations(range(11), 3)
                            if not set(t) <= hole}
            kw["point_eq"] = {x: (14 if x in hole else 15) for x in range(11)}
        elif (m, w, b) == (13, 6, 26):
            kw["point_eq"] = {x: 12 for x in range(13)}
            kw["pair_eq"] = {p: 5 for p in combinations(range(13), 2)}
        elif (m, w, b) == (12, 7, 12):
            kw["point_eq"] = {x: 7 for x in range(12)}
        st, dt = build(m, w, b, **kw)
        print(f"D2({m},{w}) >= {b}? {st} ({dt:.0f}s) [structured]", flush=True)
        open(logp, "a").write(f"{m},{w},{b},{st},{dt:.0f},structured\n")


if __name__ == "__main__":
    main()

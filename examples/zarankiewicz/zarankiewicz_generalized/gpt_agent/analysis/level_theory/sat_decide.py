"""Decision solver for open level-supply cells: does a level-w packing with
exactly b blocks exist on m points?  (D2(m,w,3) >= b?)

Strengthens the plain MILP with structure FORCED by the leave congruences at
that b (per-point degree equalities when the congruence analysis pins them,
Johnson caps otherwise) — all constraints are implied, so:
  FEAS   -> D2 >= b (witness saved + independently verified)
  INFEAS -> D2 <  b (proof that NO b-block packing exists, structure-free
            constraints only: the added rows are consequences, see notes)
Runs targets sequentially (machine budget: this is one process).

Targets given on the command line as m:w:b triples, else the default list.
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


def forced_point_degrees(m, w, b):
    """If the leave congruences force every point-degree to a single value r,
    return r; else return None. (Case: L's minimum point-leave assignment is
    unique: all ell_x = pointmin and m*pointmin == 3L.)"""
    B = 2 * comb(m, 3)
    L = B - comb(w, 3) * b
    pmod = comb(w - 1, 2)
    c1 = (2 * comb(m - 1, 2)) % pmod
    c2 = (2 * (m - 2)) % (w - 2)
    if c2 > 0:
        t = -(-((m - 1) * c2) // 2)
        pmin = c1 if c1 >= t else c1 + pmod * (-(-(t - c1) // pmod))
        if pmin == 0:
            pmin = pmod
    else:
        pmin = c1
    if pmin > 0 and 3 * L == m * pmin and (2 * comb(m - 1, 2) - pmin) % pmod == 0:
        return (2 * comb(m - 1, 2) - pmin) // pmod
    if L == 0:
        return (2 * comb(m - 1, 2) - c1) // pmod if c1 == 0 else None
    return None


def decide(m, w, b, time_limit=900):
    B = 2 * comb(m, 3)
    types = list(combinations(range(m), w))
    tris = {t: i for i, t in enumerate(combinations(range(m), 3))}
    nT, nV = len(tris), len(types)
    L = B - comb(w, 3) * b
    perfect = (L == 0)
    r_eq = forced_point_degrees(m, w, b)
    lam2 = (2 * (m - 2)) // (w - 2)
    r3 = ((m - 1) * lam2) // (w - 1)

    rows = nT + m + 1
    A = lil_matrix((rows, nV))
    lb = np.zeros(rows)
    ub = np.zeros(rows)
    for j, blk in enumerate(types):
        for t in combinations(blk, 3):
            A[tris[t], j] = 1
        for x in blk:
            A[nT + x, j] = 1
        A[nT + m, j] = 1
    # triples: == 2 if perfect else <= 2
    lb[:nT] = 2 if perfect else 0
    ub[:nT] = 2
    # points
    for x in range(m):
        if r_eq is not None:
            lb[nT + x] = ub[nT + x] = r_eq
        else:
            lb[nT + x], ub[nT + x] = 0, min(r3, (2 * comb(m - 1, 2)) // comb(w - 1, 2))
    # count
    lb[nT + m] = ub[nT + m] = b

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
        # independent verify
        from collections import Counter
        cov = Counter()
        for blk in blocks:
            for t in combinations(sorted(blk), 3):
                cov[t] += 1
        ok = (len(blocks) == b and all(len(blk) == w for blk in blocks)
              and (not cov or max(cov.values()) <= 2))
        json.dump({"m": m, "w": w, "count": b, "blocks": blocks},
                  open(os.path.join(WDIR, f"D2_{m}_{w}_b{b}.json"), "w"))
        return ("FEAS" if ok else "FEAS-BADWITNESS"), dt, r_eq
    if res.status == 2:
        return "INFEAS", dt, r_eq
    return "TIMEOUT", dt, r_eq


def main():
    if len(sys.argv) > 1:
        targets = [tuple(int(t) for t in a.split(":")) for a in sys.argv[1:]]
    else:
        targets = [(11, 5, 33), (10, 5, 22), (12, 6, 22), (11, 6, 14),
                   (12, 5, 38), (13, 5, 53)]
    os.makedirs(WDIR, exist_ok=True)
    logp = os.path.join(HERE, "decisions.csv")
    if not os.path.exists(logp):
        open(logp, "w").write("m,w,b,result,seconds,r_forced\n")
    for (m, w, b) in targets:
        st, dt, r_eq = decide(m, w, b)
        print(f"D2({m},{w}) >= {b}? {st}  ({dt:.0f}s, forced r={r_eq})", flush=True)
        open(logp, "a").write(f"{m},{w},{b},{st},{dt:.0f},{r_eq}\n")


if __name__ == "__main__":
    sys.exit(main())

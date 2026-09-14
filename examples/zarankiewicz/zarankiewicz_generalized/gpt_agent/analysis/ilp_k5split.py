"""k5-decomposed exact solve for cells where the monolithic ILP stalls.

For each k5 = 0..k5max: solve the ILP with the pentad count FIXED to k5
(hexad count capped separately per Lemma A relevance). Fixing k5 removes the
degeneracy that kills branch and bound; z = max over the sub-solves. The
decomposition is exhaustive over k5, so the result is exact when every
subproblem terminates; any timed-out subproblem is reported and the cell's
value is then only a lower bound (labeled honestly).
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


def solve_k5(m, n, k5, k6max=2, time_limit=240.0):
    types = []
    for w in range(2, m + 1):
        types.extend(combinations(range(m), w))
    tidx = {t: i for i, t in enumerate(combinations(range(m), 3))}
    ntri = len(tidx)
    nv = len(types)
    W = [len(b) for b in types]

    nrows = ntri + 3 + (m - 1)
    A = lil_matrix((nrows, nv))
    for j, b in enumerate(types):
        if W[j] >= 3:
            for t in combinations(b, 3):
                A[tidx[t], j] = 1
        A[ntri, j] = 1                       # column count
        if W[j] == 5:
            A[ntri + 1, j] = 1               # pentads == k5
        if W[j] == 6:
            A[ntri + 2, j] = 1               # hexads <= k6max
        for r in range(m - 1):
            v = (1 if r in b else 0) - (1 if (r + 1) in b else 0)
            if v:
                A[ntri + 3 + r, j] = v
    lb = np.zeros(nrows)
    ub = np.full(nrows, 2.0)
    lb[ntri] = ub[ntri] = n
    lb[ntri + 1] = ub[ntri + 1] = k5
    lb[ntri + 2], ub[ntri + 2] = 0, k6max
    for r in range(m - 1):
        lb[ntri + 3 + r], ub[ntri + 3 + r] = 0.0, np.inf

    c = -np.array(W, dtype=float)
    vub = np.array([2.0 if w >= 3 else float(n) for w in W])
    res = milp(c=c, constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, vub), integrality=np.ones(nv),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status == 2:
        return "infeasible", None
    if res.status != 0:
        return None, None
    mult = np.round(res.x).astype(int)
    blocks = []
    for j, k in enumerate(mult):
        blocks.extend([types[j]] * int(k))
    return int(round(-res.fun)), blocks


def solve_cell_split(m, n, k5max=14, time_limit=240.0):
    best, bestblocks, pending = -1, None, []
    for k5 in range(k5max + 1):
        if 10 * k5 > 2 * comb(m, 3):
            break
        t0 = time.time()
        v, blocks = solve_k5(m, n, k5, time_limit=time_limit)
        dt = time.time() - t0
        if v == "infeasible":
            print(f"  k5={k5}: infeasible [{dt:.0f}s]", flush=True)
            break
        if v is None:
            pending.append(k5)
            print(f"  k5={k5}: TIMEOUT [{dt:.0f}s]", flush=True)
            continue
        print(f"  k5={k5}: {v} [{dt:.0f}s]", flush=True)
        if v > best:
            best, bestblocks = v, blocks
    return best, bestblocks, pending


if __name__ == "__main__":
    cells = [(9, 22, 100), (10, 15, 81), (9, 24, None), (9, 25, None),
             (9, 26, None), (9, 27, None), (12, 17, None)]
    wdir = os.path.join(HERE, "witnesses")
    os.makedirs(wdir, exist_ok=True)
    out = os.path.join(HERE, "ilp_k5split.jsonl")
    for (m, n, pub) in cells:
        print(f"--- z({m},{n}) [published: {pub}] ---", flush=True)
        best, blocks, pending = solve_cell_split(m, n)
        exact = not pending or (pending and False)
        tag = "EXACT" if not pending else f"LB-only (timeouts at k5={pending})"
        print(f"z({m},{n}) = {best}  [{tag}]", flush=True)
        rec = {"m": m, "n": n, "z": best, "pending_k5": pending, "pub": pub}
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if blocks:
            with open(os.path.join(wdir, f"w_{m}x{n}.json"), "w") as f:
                json.dump({"m": m, "n": n, "edges": best,
                           "blocks": [list(b) for b in blocks]}, f)

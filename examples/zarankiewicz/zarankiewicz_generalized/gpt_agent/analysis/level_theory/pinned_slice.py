"""Heavy-fixed slice solver — the systematic profile-pinning accelerator.

For a slice S_m(sig): WLOG (relabeling) fix the LARGEST heavy block
canonically as {0..w-1}; if another heavy block exists, fix the second
one by its intersection size with the first (the stabilizer S_w x S_{m-w}
acts transitively on (a, k-a)-splits, so one representative per a is a
complete case split). Remaining heavy blocks and all quads stay MILP
variables with count equalities. S = max over cases; all-INFEAS -> -1.

Turns the 900s-timeout band slices into seconds-to-minutes instances.
Validated: reproduces S_9(13) <= 7, S_10(1) = 56, S_7/S_8 mixed values.
"""
import time
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

WS = list(range(5, 14))


def _solve(m, fixed, free_counts, time_limit):
    """fixed: list of blocks (tuples). free_counts: {w: count} still open
    (includes quads as w=4 with count=None meaning maximize)."""
    tris = {t: i for i, t in enumerate(combinations(range(m), 3))}
    nT = len(tris)
    base = np.zeros(nT)
    for b in fixed:
        for t in combinations(sorted(b), 3):
            base[tris[t]] += 1
    if base.max() > 2:
        return -1  # fixed blocks already illegal
    types, wof = [], []
    for w, k in free_counts.items():
        for b in combinations(range(m), w):
            types.append(b)
            wof.append(w)
    nV = len(types)
    counted = [w for w, k in free_counts.items() if k is not None]
    A = lil_matrix((nT + len(counted), nV))
    lb = np.zeros(nT + len(counted))
    ub = np.zeros(nT + len(counted))
    ub[:nT] = 2.0 - base
    c = np.zeros(nV)
    for j, b in enumerate(types):
        for t in combinations(b, 3):
            A[tris[t], j] = 1
        if free_counts[wof[j]] is None:
            c[j] = -1.0
    for i, w in enumerate(counted):
        for j in range(nV):
            if wof[j] == w:
                A[nT + i, j] = 1
        lb[nT + i] = ub[nT + i] = free_counts[w]
    res = milp(c=c, constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(nV),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status == 0:
        return int(round(-res.fun))
    if res.status == 2:
        return -1
    return None  # timeout


def pinned_slice(m, sig, time_limit=300):
    """S_m(sig) via canonical heavy-fixing. Returns (S, seconds) with
    S = -1 infeasible, None = some case timed out (value = max known)."""
    t0 = time.time()
    heavy = []
    for w, k in zip(WS, sig):
        heavy += [w] * k
    heavy.sort(reverse=True)
    if not heavy:
        return _solve(m, [], {4: None}, time_limit), time.time() - t0
    w1 = heavy[0]
    fixed1 = tuple(range(w1))
    rest = heavy[1:]
    cases = []
    if rest:
        w2 = rest[0]
        amin = max(0, w2 - (m - w1))
        for a in range(amin, min(w1, w2) + 1):
            blk2 = tuple(list(range(a)) + list(range(w1, w1 + w2 - a)))
            cases.append([fixed1, blk2])
        rem = rest[1:]
    else:
        cases = [[fixed1]]
        rem = []
    best = -1
    timeout = False
    for fx in cases:
        fc = {4: None}
        for w in rem:
            fc[w] = fc.get(w, 0) + 1
        # drop double-count: blocks of the same weight as fixed ones that
        # remain unfixed are in rem already (heavy[0], maybe heavy[1] fixed)
        v = _solve(m, fx, fc, time_limit)
        if v is None:
            timeout = True
        elif v > best:
            best = v
    if timeout:
        # a timed-out case could hide a larger value: S is NOT a valid cap
        return None, time.time() - t0
    return best, time.time() - t0


if __name__ == "__main__":
    import sys
    m = int(sys.argv[1])
    sig = tuple(int(t) for t in sys.argv[2].split(";"))
    tl = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    S, dt = pinned_slice(m, sig, tl)
    print(f"S_{m}{sig} = {S}  ({dt:.0f}s)")

"""Regenerate + independently verify witnesses for the exact level packing
numbers D2(m,w,3) (m<=9 cheap cells + any requested), so every LB used by the
spectrum theory has an in-workspace certificate.

Verifier is code-disjoint from the MILP model: brute-force triple coverage.
Writes witnesses/D2_{m}_{w}.json = {"m","w","blocks","count"}.
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


def solve(m, w, time_limit=600):
    types = list(combinations(range(m), w))
    tris = {t: i for i, t in enumerate(combinations(range(m), 3))}
    A = lil_matrix((len(tris), len(types)))
    for j, b in enumerate(types):
        for t in combinations(b, 3):
            A[tris[t], j] = 1
    res = milp(c=-np.ones(len(types)),
               constraints=LinearConstraint(A.tocsc(), 0, 2),
               bounds=Bounds(0, 2), integrality=np.ones(len(types)),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status != 0:
        return None, None
    x = np.round(res.x).astype(int)
    blocks = []
    for j, c in enumerate(x):
        blocks.extend([list(types[j])] * int(c))
    return int(round(-res.fun)), blocks


def verify(m, w, blocks):
    """Independent check: all blocks weight w, mult<=2, every triple <= 2."""
    from collections import Counter
    bc = Counter(tuple(sorted(b)) for b in blocks)
    assert all(len(set(b)) == w for b in blocks), "bad block size"
    assert all(0 <= min(b) and max(b) < m for b in blocks), "bad point"
    assert max(bc.values()) <= 2, "block multiplicity > 2"
    cov = Counter()
    for b in blocks:
        for t in combinations(sorted(b), 3):
            cov[t] += 1
    assert not cov or max(cov.values()) <= 2, "triple covered > 2"
    return True


def main():
    os.makedirs(WDIR, exist_ok=True)
    exact = {}
    for line in open(os.path.join(HERE, "..", "level_D2.csv")).readlines()[1:]:
        p = line.strip().split(",")
        if len(p) >= 4 and p[3] == "EXACT" and int(p[2]) >= 0:
            exact[(int(p[0]), int(p[1]))] = int(p[2])
    targets = [(m, w) for (m, w) in sorted(exact) if m <= 10]
    for (m, w) in targets:
        out = os.path.join(WDIR, f"D2_{m}_{w}.json")
        if os.path.exists(out):
            continue
        t0 = time.time()
        v, blocks = solve(m, w)
        if v is None:
            print(f"D2({m},{w}): solver timeout, skipped", flush=True)
            continue
        ok = verify(m, w, blocks)
        match = (v == exact[(m, w)])
        json.dump({"m": m, "w": w, "count": v, "blocks": blocks}, open(out, "w"))
        print(f"D2({m},{w}) = {v} (csv {exact[(m,w)]}) verified={ok} "
              f"match={match} ({time.time()-t0:.0f}s)", flush=True)
        assert match, f"MISMATCH vs level_D2.csv at ({m},{w})"


if __name__ == "__main__":
    sys.exit(main())

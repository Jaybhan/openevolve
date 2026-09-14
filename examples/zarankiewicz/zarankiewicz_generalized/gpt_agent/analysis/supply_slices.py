"""Tabulate the supply function S_m(k5) = max #weight-4 blocks coexisting
with exactly k5 weight-5 blocks (no heavier blocks), all under 2-fold triple
capacity on m points. S_m(0) = T33(m).

ILP: vars = multiplicities of 4-blocks and 5-blocks (0..2 each);
constraint per triple <= 2; sum of 5-mults == k5; maximize sum of 4-mults.
"""
import sys
from itertools import combinations

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def S(m, k5, k6=0, time_limit=600.0):
    quads = list(combinations(range(m), 4))
    fives = list(combinations(range(m), 5))
    sixes = list(combinations(range(m), 6)) if k6 else []
    types = quads + fives + sixes
    tris = {t: i for i, t in enumerate(combinations(range(m), 3))}
    ntri = len(tris)
    nv = len(types)

    A = lil_matrix((ntri + 1 + (1 if k6 else 0), nv))
    for j, b in enumerate(types):
        for t in combinations(b, 3):
            A[tris[t], j] = 1
        if len(b) == 5:
            A[ntri, j] = 1
        if k6 and len(b) == 6:
            A[ntri + 1, j] = 1
    lb = np.zeros(ntri + 1 + (1 if k6 else 0))
    ub = np.full(ntri + 1 + (1 if k6 else 0), 2.0)
    lb[ntri] = ub[ntri] = k5
    if k6:
        lb[ntri + 1] = ub[ntri + 1] = k6

    c = np.zeros(nv)
    c[:len(quads)] = -1.0
    res = milp(c=c, constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(nv),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status != 0:
        return None
    return int(round(-res.fun))


if __name__ == "__main__":
    m = int(sys.argv[1])
    kmax = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    print(f"S_{m}(k5): ", flush=True)
    for k5 in range(kmax + 1):
        v = S(m, k5)
        print(f"  S_{m}({k5}) = {v}", flush=True)

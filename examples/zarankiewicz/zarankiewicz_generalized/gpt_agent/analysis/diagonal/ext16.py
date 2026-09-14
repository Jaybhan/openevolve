"""Best one-point extension of the (16,16) cap-hyperplane optimum to 17x17.

Rebuild the 128-edge structure (rows = F_2^4, columns = both sides of the 8
hyperplanes with cap normals {8..15}), then exact ILP for the best added
17th row (support among old columns, vars u_c) + 17th column (support
vars v_x, plus case split on whether the new column contains the new row).

Maximize 128 + sum(u) + sum(v) subject to K33-freeness of the extension.
"""
import json
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def ip(a, x):
    return bin(a & x).count("1") & 1


def build16():
    cols = []
    for a in range(8, 16):
        for b in (0, 1):
            cols.append([x for x in range(16) if ip(a, x) == b])
    return cols


def solve_case(cols, with_rstar):
    cov = {}
    for c in cols:
        for t in combinations(sorted(c), 3):
            cov[t] = cov.get(t, 0) + 1
    assert all(v <= 2 for v in cov.values())
    pairs = list(combinations(range(16), 2))
    pidx = {p: i for i, p in enumerate(pairs)}
    colset = [set(c) for c in cols]
    # vars: u_0..u_15 | v_0..v_15 | w_pairs (only if with_rstar)
    nu, nv = 16, 16
    nw = len(pairs) if with_rstar else 0
    N = nu + nv + nw
    rows = []
    lb, ub = [], []
    A = lil_matrix((0, N))

    def add_row(coefs, lo, hi):
        nonlocal A
        r = A.shape[0]
        A.resize((r + 1, N))
        for j, v in coefs:
            A[r, j] = v
        lb.append(lo)
        ub.append(hi)

    # (a) old triples at capacity 2: new column can't contain all three
    for t, k in cov.items():
        if k == 2:
            add_row([(nu + x, 1.0) for x in t], 0, 2)
    # triples at coverage <=1 gain at most 1 from the new column: fine.
    # (b) new-row triples {r*, x, y}
    for (x, y) in pairs:
        ucols = [j for j, cs in enumerate(colset) if x in cs and y in cs]
        if with_rstar:
            wj = nu + nv + pidx[(x, y)]
            add_row([(j, 1.0) for j in ucols] + [(wj, 1.0)], 0, 2)
            # w >= v_x + v_y - 1
            add_row([(wj, 1.0), (nu + x, -1.0), (nu + y, -1.0)], -1, np.inf)
        else:
            add_row([(j, 1.0) for j in ucols], 0, 2)
    cobj = np.zeros(N)
    cobj[:nu + nv] = -1.0
    integrality = np.ones(N)
    res = milp(c=cobj, constraints=LinearConstraint(A.tocsc(),
               np.array(lb), np.array(ub)),
               bounds=Bounds(0, 1), integrality=integrality,
               options={"time_limit": 300.0, "mip_rel_gap": 0.0})
    assert res.status == 0, res.status
    x = np.round(res.x).astype(int)
    u = [c for c in range(16) if x[c]]
    v = [r for r in range(16) if x[nu + r]]
    return u, v, len(u) + len(v) + (1 if with_rstar else 0)


def main():
    cols = build16()
    assert sum(len(c) for c in cols) == 128
    best = None
    for with_rstar in (True, False):
        u, v, gain = solve_case(cols, with_rstar)
        print(f"case newrow-in-newcol={with_rstar}: gain {gain} "
              f"(E = {128 + gain}); |newrow|={len(u)}+{1 if with_rstar else 0}"
              f" |newcol|={len(v)}+{1 if with_rstar else 0}")
        if best is None or gain > best[3]:
            best = (with_rstar, u, v, gain)
    with_rstar, u, v, gain = best
    blocks = [sorted(c) + ([16] if j in u else [])
              for j, c in enumerate(cols)]
    newcol = sorted(v) + ([16] if with_rstar else [])
    blocks.append(newcol)
    E = sum(len(b) for b in blocks)
    assert E == 128 + gain
    out = {"m": 17, "n": 17, "edges": E, "blocks": blocks,
           "source": f"cap16 + point extension (ILP exact, gain {gain})"}
    path = "witnesses/ext16_%d.json" % E
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}  E={E}")


if __name__ == "__main__":
    main()

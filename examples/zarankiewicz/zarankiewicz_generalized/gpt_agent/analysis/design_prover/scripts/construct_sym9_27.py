#!/usr/bin/env python3
"""m = 27 (3|m family): J = 1458 packing with leave = doubled parallel
class, via a prescribed order-9 automorphism sigma = three 9-cycles
(points 9b..9b+8 cycled, b = 0,1,2).

Hole (coverage 0): the nine triples {9b+i, 9b+i+3, 9b+i+6} (b, i in
{0,1,2}) — a parallel class, sigma-invariant as a set (three sigma-orbits
of size 3).  Every other triple covered exactly twice.  Block count:
all quad orbits have size 9 (no quad is sigma^3-invariant: sigma^3-orbits
are 3-sets, and 4 is not a multiple of 3), so 9·(sum x_A) = 1458.
"""
import sys, json, time, itertools
from math import comb
import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

M = 27
TARGET = 1458

def sig(x):
    b = x // 9
    return b * 9 + (x - b * 9 + 1) % 9

def orbit(subset):
    cur = tuple(sorted(subset))
    out = [cur]
    for _ in range(8):
        cur = tuple(sorted(sig(x) for x in cur))
        if cur == out[0]:
            break
        out.append(cur)
    return out

def canon(subset):
    return min(orbit(subset))

def main(time_limit=3000.0, seed=None):
    t0 = time.time()
    tri_canon, tri_orbits = {}, []
    for T in itertools.combinations(range(M), 3):
        c = canon(T)
        if c not in tri_canon:
            tri_canon[c] = len(tri_orbits)
            tri_orbits.append((c, len(set(orbit(c)))))
    tri_id = {T: tri_canon[canon(T)] for T in itertools.combinations(range(M), 3)}
    nT = len(tri_orbits)

    quad_canon, quad_orbits = {}, []
    for Q in itertools.combinations(range(M), 4):
        c = canon(Q)
        if c not in quad_canon:
            quad_canon[c] = len(quad_orbits)
            quad_orbits.append((c, len(set(orbit(c)))))
    nQ = len(quad_orbits)
    assert all(s == 9 for _, s in quad_orbits), "unexpected short quad orbit"

    inc = {}
    for Q in itertools.combinations(range(M), 4):
        a = quad_canon[canon(Q)]
        for T in itertools.combinations(Q, 3):
            key = (tri_id[T], a)
            inc[key] = inc.get(key, 0) + 1
    rows, cols, vals = [], [], []
    for (o, a), v in inc.items():
        osize = tri_orbits[o][1]
        assert v % osize == 0
        rows.append(o); cols.append(a); vals.append(v // osize)
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(nT, nQ)).tocsc()

    hole = set()
    for b in range(3):
        for i in range(3):
            T = tuple(sorted((9 * b + i, 9 * b + i + 3, 9 * b + i + 6)))
            hole.add(tri_id[T])
    assert len(hole) == 3, hole  # three sigma-orbits of size 3
    assert all(tri_orbits[o][1] == 3 for o in hole)

    lb = np.array([0.0 if o in hole else 2.0 for o in range(nT)])
    ub = lb.copy()
    cons = [LinearConstraint(A, lb, ub)]
    sizes = np.full(nQ, 9.0)
    cons.append(LinearConstraint(sizes.reshape(1, -1), [TARGET], [TARGET]))

    c = np.zeros(nQ)
    if seed is not None:
        c = np.random.default_rng(seed).uniform(0, 1, nQ)
    res = milp(c=c, constraints=cons, integrality=np.ones(nQ),
               bounds=Bounds(0, 2),
               options={"time_limit": time_limit, "presolve": True, "disp": False})
    info = dict(m=M, target=TARGET, scheme="three 9-cycles + doubled parallel-class hole",
                nQ=nQ, nT=nT, status=int(res.status), message=res.message,
                seconds=round(time.time() - t0, 1))
    print(json.dumps(info))
    if res.status != 0:
        print("NO SOLUTION for this scheme")
        return 1

    x = np.rint(res.x).astype(int)
    blocks = []
    for a, k in enumerate(x):
        if k > 0:
            rep, _ = quad_orbits[a]
            for Q in set(orbit(rep)):
                blocks += [tuple(sorted(Q))] * k
    # inline re-verify (independent checkers run separately as well)
    from collections import Counter
    assert len(blocks) == TARGET
    mult = Counter(blocks)
    assert all(v <= 2 for v in mult.values())
    cov = Counter()
    for Q in blocks:
        for T in itertools.combinations(Q, 3):
            cov[T] += 1
    assert all(v <= 2 for v in cov.values())
    leave = [(T, 2 - cov.get(T, 0)) for T in itertools.combinations(range(M), 3)
             if cov.get(T, 0) < 2]
    lw = sum(v for _, v in leave)
    print("VERIFIED inline: %d blocks, leave weight %d" % (len(blocks), lw))
    print("leave:", leave)
    out = {"v": M, "m": M, "target": TARGET, "count": TARGET,
           "scheme": "sigma = three 9-cycles; hole = doubled {i,i+3,i+6} parallel class",
           "blocks": [list(b) for b in blocks],
           "leave": [[list(t), v] for t, v in leave]}
    base = "/Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/zarankiewicz_generalized/gpt_agent/analysis/design_prover/witnesses"
    fn = "%s/t33_m27_b1458_parclass.json" % base
    with open(fn, "w") as f:
        json.dump(out, f)
    print("saved", fn)
    return 0

if __name__ == "__main__":
    tl = float(sys.argv[1]) if len(sys.argv) > 1 else 3000.0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.exit(main(tl, seed))

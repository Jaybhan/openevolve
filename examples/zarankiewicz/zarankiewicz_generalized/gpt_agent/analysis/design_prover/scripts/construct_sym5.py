#!/usr/bin/env python3
"""J-2 packing construction for the class m = 3 (mod 4), m != 0 (mod 3),
via a prescribed order-5 automorphism and (optionally) a prescribed
doubled-pentagon leave ("hole") on one 5-cycle.

Structure: points [0, 5*c5) partitioned into c5 blocks of 5, each rotated by
sigma; points [5*c5, m) fixed.  The hole (leave = 5 cyclic-interval triples of
the first 5-cycle, each with leave multiplicity 2, i.e. coverage 0; every
other triple covered exactly twice) is sigma-invariant, matching the leave of
the maximum 15-packing at m=7 (doubled C5-edge-complement pentagon).

Usage: construct_sym5.py m target c5 mode [time_limit]
  mode: "hole"  = exact doubled-pentagon leave (equality constraints)
        "free"  = coverage <= 2 only (leave floats)
"""
import sys, json, itertools, time
from math import comb
import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds


def build_and_solve(m, target, c5, mode="hole", time_limit=600, rng_seed=None):
    npts_cyc = 5 * c5
    assert npts_cyc <= m

    def sig(x):
        if x < npts_cyc:
            b = x // 5
            return b * 5 + (x - b * 5 + 1) % 5
        return x

    def orbit(subset):
        """all images of subset under <sigma>, as sorted tuples"""
        cur = tuple(sorted(subset))
        out = [cur]
        for _ in range(4):
            cur = tuple(sorted(sig(x) for x in cur))
            if cur == out[0]:
                break
            out.append(cur)
        return out

    def canon(subset):
        return min(orbit(subset))

    t0 = time.time()
    # ---- triple orbits ----
    tri_canon = {}
    tri_orbits = []       # list of (rep, size)
    for T in itertools.combinations(range(m), 3):
        c = canon(T)
        if c not in tri_canon:
            tri_canon[c] = len(tri_orbits)
            tri_orbits.append((c, len(set(orbit(c)))))
    tri_id = {}
    for T in itertools.combinations(range(m), 3):
        tri_id[T] = tri_canon[canon(T)]
    nT = len(tri_orbits)

    # ---- quad orbits ----
    quad_canon = {}
    quad_orbits = []      # list of (rep, size)
    for Q in itertools.combinations(range(m), 4):
        c = canon(Q)
        if c not in quad_canon:
            quad_canon[c] = len(quad_orbits)
            quad_orbits.append((c, len(set(orbit(c)))))
    nQ = len(quad_orbits)

    # ---- incidence: inc[orbitA][orbitO] = total (Q in A, T in O, T subset Q) ----
    rows, cols, vals = [], [], []
    inc = {}
    for Q in itertools.combinations(range(m), 4):
        a = quad_canon[canon(Q)]
        for T in itertools.combinations(Q, 3):
            key = (tri_id[T], a)
            inc[key] = inc.get(key, 0) + 1
    for (o, a), v in inc.items():
        osize = tri_orbits[o][1]
        assert v % osize == 0, (o, a, v, osize)
        rows.append(o); cols.append(a); vals.append(v // osize)
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(nT, nQ)).tocsc()

    # ---- hole: doubled pentagon on first 5-cycle = the 5 cyclic-interval triples ----
    pent_reps = set()
    for i in range(5):
        T = tuple(sorted(((i) % 5, (i + 1) % 5, (i + 2) % 5)))
        pent_reps.add(tri_id[T])
    assert len(pent_reps) == 1 if mode == "hole" and c5 >= 1 else True
    pent = pent_reps.pop()

    # ---- constraints ----
    cons = []
    lb = np.zeros(nT); ub = np.zeros(nT)
    if mode == "hole":
        for o in range(nT):
            lb[o] = ub[o] = 0.0 if o == pent else 2.0
    else:
        for o in range(nT):
            lb[o] = 0.0
            ub[o] = 0.0 if o == pent else 2.0   # keep hole even in free mode? no:
        if mode == "free":
            ub[pent] = 2.0
    cons.append(LinearConstraint(A, lb, ub))
    # block count: sum over orbits of size_A * x_A = target
    sizes = np.array([s for (_, s) in quad_orbits], dtype=float)
    cons.append(LinearConstraint(sizes.reshape(1, -1), [target], [target]))

    c = np.zeros(nQ)
    if rng_seed is not None:
        rng = np.random.default_rng(rng_seed)
        c = rng.uniform(0, 1, nQ)  # random objective diversifies solutions
    res = milp(c=c, constraints=cons, integrality=np.ones(nQ),
               bounds=Bounds(0, 2),
               options={"time_limit": time_limit, "presolve": True,
                        "disp": False})
    dt = time.time() - t0
    info = dict(m=m, target=target, c5=c5, mode=mode, nQ=nQ, nT=nT,
                status=int(res.status), message=res.message, seconds=round(dt, 1))
    if res.status != 0:
        return None, info

    x = np.rint(res.x).astype(int)
    blocks = []
    for a, k in enumerate(x):
        if k > 0:
            rep, size = quad_orbits[a]
            for Q in set(orbit(rep)):
                for _ in range(k):
                    blocks.append(tuple(sorted(Q)))
    return blocks, info


def verify(m, blocks, target):
    """from-scratch check: count, multiplicities, coverage."""
    from collections import Counter
    assert len(blocks) == target, (len(blocks), target)
    mult = Counter(blocks)
    assert all(v <= 2 for v in mult.values()), "quad multiplicity > 2"
    cov = Counter()
    for Q in blocks:
        for T in itertools.combinations(sorted(Q), 3):
            cov[T] += 1
    assert all(v <= 2 for v in cov.values()), "triple covered > 2"
    leave = {T: 2 - cov.get(T, 0) for T in itertools.combinations(range(m), 3)}
    lw = sum(leave.values())
    leave_triples = sorted([(T, v) for T, v in leave.items() if v > 0])
    return lw, leave_triples


if __name__ == "__main__":
    m = int(sys.argv[1]); target = int(sys.argv[2]); c5 = int(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "hole"
    tl = float(sys.argv[5]) if len(sys.argv) > 5 else 600.0
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else None
    blocks, info = build_and_solve(m, target, c5, mode, tl, seed)
    print(json.dumps(info))
    if blocks is not None:
        lw, lt = verify(m, blocks, target)
        print("VERIFIED: %d blocks, leave weight %d" % (len(blocks), lw))
        print("leave:", lt)
        out = {"m": m, "target": target, "mode": mode, "c5": c5,
               "blocks": [list(b) for b in blocks],
               "leave": [[list(t), v] for t, v in lt]}
        base = "/Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/zarankiewicz_generalized/gpt_agent/analysis/design_prover/witnesses"
        fn = "%s/t33_m%d_b%d_%s.json" % (base, m, target, mode)
        with open(fn, "w") as f:
            json.dump(out, f)
        print("saved", fn)
    else:
        print("NO SOLUTION for this scheme")

"""Profile-exhaustive decision: is there a level-w packing with b blocks?

Method (PROVEN-complete case split): every packing's point-leave vector
(ell_x) is, after WLOG relabeling, a NONINCREASING sequence of values
== c1 (mod C(w-1,2)), 0 <= ell_x <= 2C(m-1,2), summing to 3L, with
ell_x >= pointmin (pair-congruence floor, admissibility.pmin_value logic
when c2 > 0). Enumerate ALL such profiles; run the block-MILP with the
point degrees pinned (r_x = (2C(m-1,2) - ell_x)/C(w-1,2)). All INFEAS
=> D2 < b. Any FEAS => witness (saved + verified).
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
sys.path.insert(0, HERE)
from admissibility import pmin_value

WDIR = os.path.join(HERE, "witnesses")


def profiles(m, w, b):
    B = 2 * comb(m, 3)
    L = B - comb(w, 3) * b
    pmod = comb(w - 1, 2)
    c1 = (2 * comb(m - 1, 2)) % pmod
    pmin = pmin_value(m, w)
    cap = 2 * comb(m - 1, 2)
    tot = 3 * L
    out = []

    def rec(prefix, remaining, slots, maxv):
        if slots == 0:
            if remaining == 0:
                out.append(list(prefix))
            return
        # values: v == c1 mod pmod, pmin <= v <= min(maxv, cap, remaining)
        lo = pmin
        v0 = min(maxv, cap, remaining)
        v = c1 + pmod * ((v0 - c1) // pmod) if v0 >= c1 else -1
        while v >= lo:
            if remaining - v <= (slots - 1) * v:  # feasible tail (nonincreasing)
                rec(prefix + [v], remaining - v, slots - 1, v)
            v -= pmod
        if lo == 0 and (c1 == 0) and remaining == 0:
            pass

    rec([], tot, m, cap)
    return [p for p in out if sum(p) == tot]


def decide_profile(m, w, b, prof, time_limit=300):
    types = list(combinations(range(m), w))
    tris = {t: i for i, t in enumerate(combinations(range(m), 3))}
    nT, nV = len(tris), len(types)
    pmod = comb(w - 1, 2)
    r = [(2 * comb(m - 1, 2) - e) // pmod for e in prof]
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
    ub[:nT] = 2
    for x in range(m):
        lb[nT + x] = ub[nT + x] = r[x]
    lb[nT + m] = ub[nT + m] = b
    res = milp(c=np.zeros(nV), constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(nV),
               options={"time_limit": time_limit})
    if res.status == 0:
        x = np.round(res.x).astype(int)
        blocks = []
        for j, c in enumerate(x):
            blocks.extend([list(types[j])] * int(c))
        return "FEAS", blocks
    return ("INFEAS" if res.status == 2 else "TIMEOUT"), None


def decide(m, w, b):
    profs = profiles(m, w, b)
    print(f"D2({m},{w}) >= {b}? {len(profs)} degree-profiles to check", flush=True)
    any_timeout = False
    for i, prof in enumerate(profs):
        t0 = time.time()
        st, blocks = decide_profile(m, w, b, prof)
        print(f"  profile {i+1}/{len(profs)} {prof}: {st} ({time.time()-t0:.0f}s)",
              flush=True)
        if st == "FEAS":
            from collections import Counter
            cov = Counter()
            for blk in blocks:
                for t in combinations(sorted(blk), 3):
                    cov[t] += 1
            assert len(blocks) == b and max(cov.values()) <= 2
            json.dump({"m": m, "w": w, "count": b, "blocks": blocks},
                      open(os.path.join(WDIR, f"D2_{m}_{w}_b{b}.json"), "w"))
            return "FEAS"
        if st == "TIMEOUT":
            any_timeout = True
    return "TIMEOUT-partial" if any_timeout else "INFEAS"


def main():
    logp = os.path.join(HERE, "decisions.csv")
    for job in sys.argv[1:]:
        m, w, b = (int(t) for t in job.split(":"))
        st = decide(m, w, b)
        print(f"D2({m},{w}) >= {b}? {st} [profile-exhaustive]", flush=True)
        open(logp, "a").write(f"{m},{w},{b},{st},profiles,\n")


if __name__ == "__main__":
    main()

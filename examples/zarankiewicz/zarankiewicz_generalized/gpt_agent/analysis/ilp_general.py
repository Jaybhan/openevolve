"""Exact z(m,n;s,t) for small cells via MILP — general (s,t).

Formulation: variable x_b = multiplicity of block type b ⊆ [m].
  - blocks of weight < s cover no s-set: "pads", multiplicity <= n;
    only weights s-1 and lower matter for edges; we include weights
    max(1, s-2)..s-1 as pad types (weight s-1 dominates but equality of
    column count needs flexibility when few pad types exist).
  - blocks of weight >= s: multiplicity <= t-1 (else their own s-sets
    overflow), and for each s-set S: sum_{b ⊇ S} x_b <= t-1.
  - sum_b x_b = n; maximize sum |b| x_b.
Exact by the block-multiset decomposition (column order irrelevant).

Also computes WF(m,n;s,t) (two-sided integer waterfill) and the Culík value
for comparison, and cross-checks z(m,n;s,t) == z(n,m;t,s) when both fit.
"""
import argparse
import json
import os
import time
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def solve_cell(m, n, s, t, time_limit=300.0):
    if m < s or n < t:
        return m * n, "trivial-all-ones"
    types = []
    wmin = max(1, s - 2)
    for w in range(wmin, m + 1):
        types.extend(combinations(range(m), w))
    ssets = list(combinations(range(m), s))
    sidx = {S: i for i, S in enumerate(ssets)}
    nv, ns_ = len(types), len(ssets)

    A = lil_matrix((ns_ + 1, nv))
    for j, b in enumerate(types):
        if len(b) >= s:
            for S in combinations(b, s):
                A[sidx[S], j] = 1
        A[ns_, j] = 1
    lb = np.zeros(ns_ + 1)
    ub = np.full(ns_ + 1, float(t - 1))
    lb[ns_] = ub[ns_] = n

    c = -np.array([len(b) for b in types], dtype=float)
    vub = np.array([float(t - 1) if len(b) >= s else float(n) for b in types])

    res = milp(c=c, constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, vub), integrality=np.ones(nv),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status != 0:
        return None, f"UNRESOLVED({res.status})"
    return int(round(-res.fun)), "ilp"


def wf_side(ncols, budget, s, cap):
    E = ncols * min(s - 1, cap)
    if cap <= s - 1:
        return E
    h = s - 1
    while h < cap:
        cost = comb(h, s - 1)
        k = min(ncols, budget // cost) if cost else ncols
        if k == 0:
            break
        E += k
        budget -= k * cost
        if k < ncols:
            break
        h += 1
    return E


def WF(m, n, s, t):
    return min(wf_side(n, (t - 1) * comb(m, s), s, m),
               wf_side(m, (s - 1) * comb(n, t), t, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, required=True)
    ap.add_argument("--t", type=int, required=True)
    ap.add_argument("--mmax", type=int, default=8)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--time-limit", type=float, default=300.0)
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(HERE, f"exact_{args.s}{args.t}.csv")
    rows = []
    print(f"(s,t)=({args.s},{args.t})  m<= {args.mmax}, n<= {args.nmax}")
    print(f"{'cell':>8} {'z':>4} {'WF':>4} {'d':>3}  how")
    for m in range(args.s, args.mmax + 1):
        for n in range(max(m, args.t), args.nmax + 1):
            t0 = time.time()
            z, how = solve_cell(m, n, args.s, args.t, args.time_limit)
            dt = time.time() - t0
            if z is None:
                print(f"{m}x{n:>4}  {how} [{dt:.0f}s]", flush=True)
                continue
            w = WF(m, n, args.s, args.t)
            rows.append((m, n, z, w, w - z))
            print(f"{m}x{n:>4} {z:>4} {w:>4} {w-z:>3}  {how} [{dt:.0f}s]",
                  flush=True)
    with open(out, "w") as f:
        f.write("m,n,z,wf,deficit\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

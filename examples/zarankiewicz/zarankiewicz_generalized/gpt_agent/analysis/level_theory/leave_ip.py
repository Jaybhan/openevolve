"""Exact congruence-leave feasibility IP — the level-w generalization of
Lemma E + Theorem F's finite leave classification, as an algorithm.

For a level-w packing with b blocks, the leave (multiset of uncovered triple
slots) has weight L = B - C(w,3)*b and satisfies, with NO further hypothesis:
  (i)   0 <= y_T <= 2 for every triple T (leave multiplicity cap),
  (ii)  ell_x  := sum_{T ni x}  y_T == c1 (mod C(w-1,2)) at every point,
        and r_x = (2C(m-1,2)-ell_x)/C(w-1,2) <= rmax  [iterated Johnson cap],
  (iii) ell_xy := sum_{T ⊇ xy} y_T == c2 (mod w-2) at every pair.
So: U_leave(m,w) := (B - L*)/C(w,3), where L* = min feasible L in the
progression {B mod C(w,3), +C(w,3), ...}, is a PROVEN upper bound on
D2(m,w,3).  This subsumes the analytic CHECK-0 bounds and the Johnson bound.

Also computes L*_doubled: minimal feasible L with all y_T even (leave = a
DOUBLED simple 3-graph) — the GKLO-ready leave shape; and reports the gap.

Writes leave_table.csv. Pure feasibility MILPs over C(m,3)<=560 triples: fast.
"""
import os
import sys
import time
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))


def johnson_pt(m, w):
    lam2 = (2 * (m - 2)) // (w - 2)
    r3 = ((m - 1) * lam2) // (w - 1)
    r2 = (2 * comb(m - 1, 2)) // comb(w - 1, 2)
    return min(r2, r3), r2


def feasible_leave(m, w, L, doubled=False, time_limit=120):
    """Is there a triple multiset (mult<=2, doubled: mult in {0,2}) of weight L
    meeting the point & pair congruences and the per-point Johnson cap?
    Returns (status, y or None): status in {'FEAS','INFEAS','UNKNOWN'}."""
    pmod = comb(w - 1, 2)
    c1 = (2 * comb(m - 1, 2)) % pmod
    c2 = (2 * (m - 2)) % (w - 2)
    rmax, r_full = johnson_pt(m, w)
    kx_lo = max(0, r_full - rmax)

    tris = list(combinations(range(m), 3))
    nT = len(tris)
    pairs = {p: i for i, p in enumerate(combinations(range(m), 2))}
    nP = len(pairs)
    # variables: y_T (nT), k_x (m), j_p (nP)
    nv = nT + m + nP
    step = 2 if doubled else 1
    if doubled and (c1 % 2 or c2 % 2 or L % 2):
        # doubled leave forces even degrees everywhere and even L
        # (c1/c2 odd => k/j absorb parity only if pmod/(w-2) odd; cheap test:)
        pass  # let the IP decide; parity may still work through moduli
    A = lil_matrix((m + nP + 1, nv))
    lb = np.zeros(m + nP + 1)
    ub = np.zeros(m + nP + 1)
    for i, T in enumerate(tris):
        for x in T:
            A[x, i] = 1
        for p in combinations(T, 2):
            A[m + pairs[p], i] = 1
        A[m + nP, i] = 1
    for x in range(m):
        A[x, nT + x] = -pmod
        lb[x] = ub[x] = c1
    for p, i in pairs.items():
        A[m + i, nT + m + i] = -(w - 2)
        lb[m + i] = ub[m + i] = c2
    lb[m + nP] = ub[m + nP] = L

    vlb = np.zeros(nv)
    vub = np.empty(nv)
    vub[:nT] = 2
    vlb[nT:nT + m] = kx_lo
    vub[nT:nT + m] = r_full
    vub[nT + m:] = (2 * (m - 2)) // (w - 2) + 1  # j_p cap (loose, valid)
    if doubled:
        # y_T = 2 z_T: substitute by requiring y even via integer z variables:
        # implement as y in {0,2} using an extra parity trick: y_T <= 2 and
        # y_T even <=> y_T = 2 z_T. Rescale column i by 2 with z in {0,1}.
        A = A.tocsc(copy=True)
        A[:, :nT] = A[:, :nT] * 2.0
        vub[:nT] = 1
    res = milp(c=np.zeros(nv),
               constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(vlb, vub), integrality=np.ones(nv),
               options={"time_limit": time_limit})
    if res.status == 0:
        y = np.round(res.x[:nT]).astype(int) * (2 if doubled else 1)
        return "FEAS", y
    if res.status == 2:  # infeasible proven
        return "INFEAS", None
    return "UNKNOWN", None


def min_leave(m, w, doubled=False, max_steps=14):
    B = 2 * comb(m, 3)
    slots = comb(w, 3)
    L0 = B % slots
    for k in range(max_steps):
        L = L0 + k * slots
        if L > B:
            break
        st, y = feasible_leave(m, w, L, doubled=doubled)
        if st == "FEAS":
            return L, "EXACT", y
        if st == "UNKNOWN":
            return L, "UNKNOWN>=", None  # first unresolved: L* >= L (lower info)
    return None, "NONE<=%d" % (L0 + (max_steps - 1) * slots), None


def load_exact():
    exact = {}
    path = os.path.join(HERE, "..", "level_D2.csv")
    for line in open(path).readlines()[1:]:
        p = line.strip().split(",")
        if len(p) >= 4 and p[3] == "EXACT" and int(p[2]) >= 0:
            exact[(int(p[0]), int(p[1]))] = int(p[2])
    return exact


def main():
    ms = range(6, 17)
    ws = range(5, 10)
    exact = load_exact()
    out = os.path.join(HERE, "leave_table.csv")
    done = set()
    if os.path.exists(out):
        for line in open(out).readlines()[1:]:
            p = line.split(",")
            done.add((int(p[0]), int(p[1])))
    else:
        with open(out, "w") as f:
            f.write("m,w,B,slots,Lstar,Lstatus,U_leave,Ldoubled,Dstatus,exact,note\n")
    for w in ws:
        for m in ms:
            if m < w or (m, w) in done:
                continue
            if 3 * (m - w) <= m - 3:
                continue  # wedge: D2 = 2 proven, leave theory moot
            B = 2 * comb(m, 3)
            slots = comb(w, 3)
            t0 = time.time()
            Ls, st, y = min_leave(m, w)
            Ld, std, _ = min_leave(m, w, doubled=True)
            U = (B - Ls) // slots if Ls is not None and st == "EXACT" else ""
            ex = exact.get((m, w), "")
            note = ""
            if st == "EXACT" and ex != "" and U == ex:
                note = "THEORY-TIGHT"
            elif st == "EXACT" and ex != "" and U > ex:
                note = f"gap {U - ex}"
            print(f"({m},{w}): L*={Ls} [{st}], U_leave={U}, L*_dbl={Ld} [{std}], "
                  f"exact={ex} {note} ({time.time()-t0:.1f}s)", flush=True)
            with open(out, "a") as f:
                f.write(f"{m},{w},{B},{slots},{Ls},{st},{U},{Ld},{std},{ex},{note}\n")


if __name__ == "__main__":
    sys.exit(main())

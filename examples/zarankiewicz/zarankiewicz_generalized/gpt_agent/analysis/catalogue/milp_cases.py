"""MILP feasibility certificates for the T-branch purity theorem
(class m: m == 3 mod 4, m !== 0 mod 3, at heavy value Q = J-2 = T).

Every case is a LEAVE-EXISTENCE problem over an abstract bounded
support; INFEASIBLE means no legal configuration of that shape exists
on ANY class m (support bounds proven in report.md).

Cases (k6 >= 1 and k5 = 4 and k5 = 2 with |P1 n P2| <= 3 are killed by
hand; k5 = 3 killed by hand):
  A: k5 = 1 (P = 0..4), leave weight 8, npts = 8
  B: k5 = 2 doubled pentad, leave weight 6, npts = 11
  C: k5 = 2, |P1 n P2| = 4, leave weight 6, npts = 6, ell = 3 on all
  E: k5 = 5, leave weight 0: five pentads, all pairs even, npts = 12
  S91: m = 9 (not class m; 3|m family): pentad + 38 quads leave, npts=9

Constraint sets (A/B/C/S91): variables y_T in {0..2} (leave), slack
integers k_x (point congruence), j_e (pair parity):
    sum_T y_T = L
    sum_{T ni x} y_T - 3 k_x = r_x        (exact residues)
    sum_{T ⊇ e} y_T - 2 j_e = par_e
    y_T <= cap_T
E: variables z_P in {0..2} over pentads P (5-subsets of [12]) with
    sum z_P = 5,  sum_{P ⊇ e} z_P - 2 j_e = 0,  sum_{P ⊇ T} z_P <= 2.
"""
import sys
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds


def solve_ilp(nvar, integrality, lb, ub, A_eq, b_eq, A_ub, b_ub,
              time_limit=600):
    cons = []
    if A_eq is not None and A_eq.shape[0]:
        cons.append(LinearConstraint(A_eq, b_eq, b_eq))
    if A_ub is not None and A_ub.shape[0]:
        cons.append(LinearConstraint(A_ub, -np.inf, b_ub))
    res = milp(c=np.zeros(nvar), constraints=cons,
               integrality=np.array(integrality),
               bounds=Bounds(np.array(lb), np.array(ub)),
               options=dict(time_limit=time_limit, disp=False))
    return res


def leave_case(npts, L, residues, pair_par, caps, label,
               time_limit=600):
    """residues: list r_x in {0,1,2} (ell_x == r_x mod 3);
    pair_par: dict pair -> 0/1 (default 0);
    caps: dict triple -> cap (default 2)."""
    tris = list(combinations(range(npts), 3))
    ti = {t: i for i, t in enumerate(tris)}
    nT = len(tris)
    pairs = list(combinations(range(npts), 2))
    pi = {p: i for i, p in enumerate(pairs)}
    # vars: y (nT), k (npts), j (npairs)
    nvar = nT + npts + len(pairs)
    rows, cols, vals = [], [], []
    beq = []
    r = 0
    # total
    for i in range(nT):
        rows.append(r)
        cols.append(i)
        vals.append(1)
    beq.append(L)
    r += 1
    # point congruence
    for x in range(npts):
        for t, i in ti.items():
            if x in t:
                rows.append(r)
                cols.append(i)
                vals.append(1)
        rows.append(r)
        cols.append(nT + x)
        vals.append(-3)
        beq.append(residues[x] % 3)
        r += 1
    # pair parity
    for p, ip_ in pi.items():
        for t, i in ti.items():
            if p[0] in t and p[1] in t:
                rows.append(r)
                cols.append(i)
                vals.append(1)
        rows.append(r)
        cols.append(nT + npts + ip_)
        vals.append(-2)
        beq.append(pair_par.get(p, 0) % 2)
        r += 1
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nvar))
    lb = [0] * nvar
    ub = ([caps.get(t, 2) for t in tris] + [20] * npts
          + [20] * len(pairs))
    res = solve_ilp(nvar, [1] * nvar, lb, ub, A_eq, np.array(beq),
                    None, None, time_limit)
    status = ("INFEASIBLE" if res.status == 2 else
              "FEASIBLE" if res.status == 0 else f"status={res.status}")
    sol = None
    if res.status == 0:
        y = np.round(res.x[:nT]).astype(int)
        sol = [(tris[i], int(y[i])) for i in range(nT) if y[i] > 0]
    print(f"{label}: {status}" + (f"  leave={sol}" if sol else ""))
    return status, sol


def case_A():
    P = list(range(5))
    pair_par = {tuple(sorted(e)): 1 for e in combinations(P, 2)}
    caps = {t: 1 for t in combinations(P, 3)}
    return leave_case(8, 8, [0] * 8, pair_par, caps,
                      "A (k5=1, L=8, npts=8)")


def case_B():
    caps = {t: 0 for t in combinations(range(5), 3)}
    return leave_case(11, 6, [0] * 11, {}, caps,
                      "B (k5=2 doubled, L=6, npts=11)")


def case_C():
    # P1 = {0,1,2,3,4}, P2 = {0,1,2,3,5}
    pair_par = {}
    for x in range(4):
        pair_par[(x, 4)] = 1
        pair_par[(x, 5)] = 1
    caps = {}
    for t in combinations(range(4), 3):
        caps[t] = 0
    for t in combinations(range(5), 3):
        caps[t] = min(caps.get(t, 2), 1)
    for t in combinations((0, 1, 2, 3, 5), 3):
        caps[t] = min(caps.get(t, 2), 1)
    return leave_case(6, 6, [0] * 6, pair_par, caps,
                      "C (k5=2, s=4, L=6, npts=6)")


def case_S91():
    P = list(range(5))
    pair_par = {tuple(sorted(e)): 1 for e in combinations(P, 2)}
    caps = {t: 1 for t in combinations(P, 3)}
    # m=9: residues ell_x == 2 (mod 3) for all 9 points; L = 6
    return leave_case(9, 6, [2] * 9, pair_par, caps,
                      "S91 (m=9, k5=1 at Q=40: L=6, ell==2 mod 3)")


def case_E(npts=12, time_limit=1800):
    pents = list(combinations(range(npts), 5))
    ni = len(pents)
    pairs = list(combinations(range(npts), 2))
    tris = list(combinations(range(npts), 3))
    nvar = ni + len(pairs)
    rows, cols, vals, beq = [], [], [], []
    r = 0
    for i in range(ni):
        rows.append(r)
        cols.append(i)
        vals.append(1)
    beq.append(5)
    r += 1
    for ip_, p in enumerate(pairs):
        for i, P in enumerate(pents):
            if p[0] in P and p[1] in P:
                rows.append(r)
                cols.append(i)
                vals.append(1)
        rows.append(r)
        cols.append(ni + ip_)
        vals.append(-2)
        beq.append(0)
        r += 1
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nvar))
    rows, cols, vals, bub = [], [], [], []
    r = 0
    for t in tris:
        touched = False
        for i, P in enumerate(pents):
            if t[0] in P and t[1] in P and t[2] in P:
                rows.append(r)
                cols.append(i)
                vals.append(1)
                touched = True
        if touched:
            bub.append(2)
            r += 1
    A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nvar))
    # symmetry breaking: P1 = {0,1,2,3,4} present (WLOG relabel)
    lb = [0] * nvar
    ub = [2] * ni + [30] * len(pairs)
    lb[0] = 1   # pents[0] = (0,1,2,3,4) by lex order
    res = solve_ilp(nvar, [1] * nvar, lb, ub, A_eq, np.array(beq),
                    A_ub, np.array(bub), time_limit)
    status = ("INFEASIBLE" if res.status == 2 else
              "FEASIBLE" if res.status == 0 else f"status={res.status}")
    sol = None
    if res.status == 0:
        z = np.round(res.x[:ni]).astype(int)
        sol = [(pents[i], int(z[i])) for i in range(ni) if z[i] > 0]
    print(f"E (k5=5, L=0, npts={npts}): {status}"
          + (f"  pentads={sol}" if sol else ""))
    return status, sol


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "A"):
        case_A()
    if which in ("all", "C"):
        case_C()
    if which in ("all", "B"):
        case_B()
    if which in ("all", "S91"):
        case_S91()
    if which in ("all", "E"):
        case_E()

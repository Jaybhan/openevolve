"""Task 3: the finite-size supply-density profile phi, assembled from
every exact D2(m,w,3) value in the workspace + the Brown measurements,
tested against candidate closed forms. No solvers here (the two MILP
runs live in phi_milp.py).

Quantities per supply point (m, w, D):
    c    = w / m^{2/3}          (level in Brown units)
    phi  = D / m                (supply density)
    psi  = D*C(w,3) / (2C(m,3)) (budget efficiency, in [0,1])
    Jw   = per-level Johnson cap (workspace J_w)
Candidate laws evaluated at each point:
    BUDGET   phi_B = 2C(m,3)/(m*C(w,3))       (exact-binomial budget)
    STEP     phi_S = c^{-3} if c <= 1 else 0  (the new asymptotic law)
    OLD      phi_O = min(1, 1/c)              (diagonal_limit.md's cap)
Also: the finite Fueredi cap D_F(m,w) = max D with
    w*D <= m*D^{2/3} + 2*D^{4/3} + m   (FS survey Thm 3.19 form),
its onset (where it first undercuts the budget), and the diagonal
crossover m* where Fueredi's bound first beats the budget LP.
"""
import csv
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
LT = os.path.join(HERE, "..", "level_theory")

T33 = {3: 0, 4: 2, 5: 5, 6: 9, 7: 15, 8: 28, 9: 40, 10: 60, 11: 80,
       12: 108, 13: 143, 14: 182, 15: 225, 16: 280, 17: 340, 18: 405,
       19: 482, 23: 883, 27: 1458}   # w=4: Tan m<=18 + workspace 19/23/27


def J_w(m, w):
    lam2 = (2 * (m - 2)) // (w - 2)
    r3 = ((m - 1) * lam2) // (w - 1)
    r2 = (2 * comb(m - 1, 2)) // comb(w - 1, 2)
    return (m * min(r2, r3)) // w


def load_supply_points():
    pts = []   # (m, w, D, status)
    for m, D in T33.items():
        pts.append((m, 4, D, "EXACT", "Tan/workspace"))
    with open(os.path.join(LT, "supply_status.csv")) as f:
        for row in csv.DictReader(f):
            m, w = int(row["m"]), int(row["w"])
            if row["status"] == "EXACT":
                pts.append((m, w, int(row["D2"]), "EXACT", row["source"]))
            else:
                lo, hi = row["D2"].strip("[]").split(";")
                pts.append((m, w, (int(lo), int(hi)), "BRACKET",
                            row["source"]))
    # MILP results if present
    p = os.path.join(HERE, "phi_milp_results.json")
    if os.path.exists(p):
        import json
        for r in json.load(open(p)):
            if r.get("exact"):
                pts = [x for x in pts
                       if not (x[0] == r["m"] and x[1] == r["w"])]
                pts.append((r["m"], r["w"], r["incumbent_LB"],
                            "EXACT", "phi_milp(HiGHS)"))
            elif r.get("incumbent_LB") is not None:
                pts = [x for x in pts
                       if not (x[0] == r["m"] and x[1] == r["w"])]
                pts.append((r["m"], r["w"],
                            (r["incumbent_LB"], r["dual_UB"]),
                            "BRACKET", "phi_milp(timeout)"))
    # Brown LB points: m = q^3 rows, m columns of weight q^2-q exist
    # => D_{q^2-q}(q^3) >= q^3  (phi >= 1 at c = 1 - 1/q)
    bp = os.path.join(HERE, "brown_data.csv")
    if os.path.exists(bp):
        for row in csv.DictReader(open(bp)):
            q, n = int(row["q"]), int(row["n"])
            pts.append((n, q * q - q, (n, None), "LOWER", "Brown"))
    return pts


def fueredi_cap(m, w, dmax=None):
    """FIRST-crossing cap: largest D0 such that every D <= D0 satisfies
    w*D <= m*D^{2/3} + 2*D^{4/3} + m.  (The constraint is re-entrant at
    D ~ w^3 where the D^{4/3} term dominates; the supply bound is the
    first crossing.)  Returns None if no crossing below dmax."""
    f = lambda D: w * D <= m * D ** (2 / 3) + 2 * D ** (4 / 3) + m
    if dmax is None:
        dmax = 100 * m + 10 ** 6
    D = 1
    while D <= dmax and f(D):
        D *= 2
    if D > dmax:
        return None
    lo, hi = D // 2, D          # f(lo) True, f(hi) False
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid):
            lo = mid
        else:
            hi = mid
    return lo


def diag_lp_exact(m):
    B = 2 * comb(m, 3)
    w0 = 3
    while m * comb(w0 + 1, 3) <= B and w0 + 1 <= m:
        w0 += 1
    from fractions import Fraction
    c0, c1 = comb(w0, 3), comb(w0 + 1, 3)
    x1 = Fraction(B - m * c0, c1 - c0)
    return float(m * w0 + x1)


def main():
    pts = load_supply_points()
    rows = []
    for (m, w, D, st, src) in sorted(pts, key=lambda p: p[1] / p[0] ** (2 / 3)):
        c = w / m ** (2 / 3)
        B = 2 * comb(m, 3)
        phiB = B / (m * comb(w, 3))
        phiS = c ** -3 if c <= 1 else 0.0
        phiO = min(1.0, 1 / c)
        if st == "EXACT":
            Dl = Dh = D
        elif st == "BRACKET":
            Dl, Dh = D
        else:
            Dl, Dh = D[0], None
        phi_lo = Dl / m
        phi_hi = (Dh / m) if Dh is not None else None
        psi_lo = Dl * comb(w, 3) / B
        psi_hi = (Dh * comb(w, 3) / B) if Dh is not None else None
        rows.append(dict(
            m=m, w=w, c=round(c, 4), status=st, D_lo=Dl, D_hi=Dh,
            phi_lo=round(phi_lo, 4),
            phi_hi=round(phi_hi, 4) if phi_hi is not None else "",
            psi_lo=round(psi_lo, 4),
            psi_hi=round(psi_hi, 4) if psi_hi is not None else "",
            budget_D=B // comb(w, 3), Jw=J_w(m, w),
            phi_BUDGET=round(phiB, 4), phi_STEP=round(phiS, 4),
            phi_OLD=round(phiO, 4), source=src))
    with open(os.path.join(HERE, "phi_table.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"{'m':>5} {'w':>3} {'c':>6} {'status':>8} {'D':>12} "
          f"{'phi':>12} {'psi':>12} {'bud':>5} {'Jw':>5} "
          f"{'phiB':>7} {'phiSTEP':>8} {'phiOLD':>7}")
    for r in rows:
        D = (f"{r['D_lo']}" if r['D_hi'] == r['D_lo']
             else f"[{r['D_lo']};{r['D_hi']}]")
        phi = (f"{r['phi_lo']}" if r['phi_hi'] in ("", r['phi_lo'])
               else f"[{r['phi_lo']};{r['phi_hi']}]")
        psi = (f"{r['psi_lo']}" if r['psi_hi'] in ("", r['psi_lo'])
               else f"[{r['psi_lo']};{r['psi_hi']}]")
        print(f"{r['m']:>5} {r['w']:>3} {r['c']:>6} {r['status']:>8} "
              f"{D:>12} {phi:>12} {psi:>12} {r['budget_D']:>5} "
              f"{r['Jw']:>5} {r['phi_BUDGET']:>7} {r['phi_STEP']:>8} "
              f"{r['phi_OLD']:>7}")

    # ---- Fueredi finite onset --------------------------------------
    print("\nFinite Fueredi cap vs budget: first m where D_F < budget")
    for w_frac in (1.0, 0.9, 0.8, 0.7):
        onset = None
        for m in range(10, 20001):
            w = max(4, round(w_frac * m ** (2 / 3)))
            bud = (2 * comb(m, 3)) // comb(w, 3)
            DF = fueredi_cap(m, w)
            if DF is not None and DF < bud:
                onset = (m, w, DF, bud)
                break
        print(f"  c = {w_frac}: onset (m,w,D_F,budget) = {onset}")

    # ---- erosion measured at the Brown sizes -----------------------
    print("\nAt Brown sizes m = q^3, w = q^2-q (c = 1-1/q): the Fueredi"
          " cap is ACTIVE and Brown nearly saturates it:")
    print(f"  {'q':>3} {'m=q^3':>7} {'w':>5} {'budget_D':>9} "
          f"{'Fueredi_D':>10} {'Brown_D':>8} {'Brown/Fueredi':>13}")
    for q in (7, 11, 13, 17, 19, 23, 47, 101):
        m, w = q ** 3, q * q - q
        bud = (2 * comb(m, 3)) // comb(w, 3)
        DF = fueredi_cap(m, w)
        if DF is None:
            print(f"  {q:>3} {m:>7} {w:>5} {bud:>9} {'inactive':>10} "
                  f"{m:>8} {'-':>13}")
        else:
            print(f"  {q:>3} {m:>7} {w:>5} {bud:>9} {DF:>10} {m:>8} "
                  f"{m / DF:>13.3f}   DF/budget = {DF / bud:.3f}")

    # ---- diagonal crossover m* -------------------------------------
    mstar = None
    for m in range(20, 5000):
        fued = m * m ** (2 / 3) + 2 * m ** (4 / 3) + m
        if fued < diag_lp_exact(m):
            mstar = m
            break
    print(f"\nDiagonal crossover m* (Fueredi UB < budget LP): m* = {mstar}")
    for m in (mstar - 1, mstar):
        print(f"   m={m}: fueredi={m * m ** (2/3) + 2 * m ** (4/3) + m:.1f} "
              f"budgetLP={diag_lp_exact(m):.1f}")


if __name__ == "__main__":
    main()

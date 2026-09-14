"""L_min(w, m): the minimal congruence-valid leave weight for level-w
packings — pure Lemma-L3 arithmetic (CRT-periodic), plus the PROVEN
class-uniform refinements. NO solving; this is the arithmetic core of the
MASTER THEOREM's supply spectrum:

    D_w^formula(m) = (B(m) - L_min(w,m)) / C(w,3)   (large-m branch).

Periods (signature (c1, c2, B mod C(w,3)) minimal periods, machine-detected
and re-verified here): P_4 = 12, P_5 = 15, P_6 = 20, P_7 = 105, P_8 = 168.

Structure per residue class, from Lemma L3:
  quadratic (c2 > 0): bound(m) = max(ceil(m*pmin(m)/3), ceil(C(m,2)c2/3)),
      pmin(m) = ceil((m-1)c2/2) rounded up into residue c1 mod C(w-1,2);
  linear (c2 = 0 < c1): bound(m) = ceil(m*c1/3);
  gapped (c1 = c2 = 0): bound = 0 if the progression allows L = 0
      (perfect class) else C(w-1,2)  [L3(iii)];
  then L_min = smallest L == B (mod C(w,3)) with L >= bound.

PROVEN class-uniform refinements applied on top (each with its proof):
  w=4, class m == 3 (4), 3 !| m : analytic 6 -> 10   [Lemma E + Theorem F:
      weight-6 leaves have empty classification; doubled pentagon attains]
  w=4, 3 | m: the rounding reproduces Theorem 11(c)'s classification
      L = 2m/3 + 2*[m == 9 (12)]  (self-tested below).
  w=5, class m == 14 (15): analytic 8 -> 18   [unique support-4 shape =
      doubled K_4^(3) has pair-leave 4 != 0 mod 3 — spectrum.md 3.1]
Sporadic per-m completion obstructions (Dehon-type) are NOT class facts;
they live in F.py's small-m branch: (11,5): perfect and B/10-1 both dead;
(10,5) <= 21; (12,7) <= 10.  [decisions.csv, all proven]
"""
from math import comb

PERIOD = {4: 12, 5: 15, 6: 20, 7: 105, 8: 168}


def sig(m, w):
    B = 2 * comb(m, 3)
    return ((2 * comb(m - 1, 2)) % comb(w - 1, 2),
            (2 * (m - 2)) % (w - 2), B % comb(w, 3))


def Lmin(w, m):
    """Minimal congruence-valid leave weight (analytic + proven class
    refinements). Valid for every m >= w; the MEANINGFUL regime is the
    design zone (for wedge/complement m the small-m branch of F rules)."""
    pmod = comb(w - 1, 2)
    slots = comb(w, 3)
    c1, c2, beta = sig(m, w)
    if c2 > 0:
        t = -(-((m - 1) * c2) // 2)
        pmin = c1 if c1 >= t else c1 + pmod * (-(-(t - c1) // pmod))
        if pmin == 0:
            pmin = pmod
        bound = max(-(-(m * pmin) // 3), -(-(comb(m, 2) * c2) // 3))
    elif c1 > 0:
        bound = -(-(m * c1) // 3)
    else:
        bound = 0 if beta == 0 else pmod
    L = beta + slots * max(0, -(-(bound - beta) // slots))
    # proven class-uniform refinements
    if w == 4 and m % 4 == 3 and m % 3 != 0 and L < 10:
        L = 10  # Lemma E + Theorem F (weight 2 and 6 leaves impossible)
    if w == 5 and m % 15 == 14 and L == 8:
        L = 18  # doubled-K4 pair-congruence kill (class-uniform)
    return L


def D_formula(w, m):
    """(B - Lmin)/C(w,3), floored into consistency, capped by Johnson."""
    B = 2 * comb(m, 3)
    slots = comb(w, 3)
    D = (B - Lmin(w, m)) // slots
    lam2 = (2 * (m - 2)) // (w - 2)
    r3 = ((m - 1) * lam2) // (w - 1)
    r2 = (2 * comb(m - 1, 2)) // comb(w - 1, 2)
    J = (m * min(r2, r3)) // w
    return max(0, min(D, J))


def selftest():
    ok = True
    # periods: signature AND Lmin-formula shape must repeat with period P
    for w, P in PERIOD.items():
        for m in range(max(w, 5), max(w, 5) + 3 * P):
            if sig(m, w) != sig(m + P, w):
                print(f"FAIL period sig w={w} m={m}")
                ok = False
        # minimality of the period on the signature
        for Q in range(1, P):
            if P % Q == 0 and all(sig(m, w) == sig(m + Q, w)
                                  for m in range(w + 20, w + 20 + 3 * P)):
                print(f"NOTE w={w}: signature period divides {Q} < {P}")
    # w=4 reproduces the Theorem 11 spectrum exactly
    for m in range(4, 201):
        J = (m * (((m - 1) * (m - 2)) // 3)) // 4
        if m % 4 == 3 and m % 3 != 0:
            want = J - 2
        else:
            want = J
        B = 2 * comb(m, 3)
        got = (B - Lmin(4, m)) // 4
        if got != want:
            print(f"FAIL w=4 m={m}: (B-Lmin)/4 = {got} vs spectrum {want}")
            ok = False
    # 3|m: Theorem 11(c) leave classification
    for m in range(6, 201, 3):
        want = 2 * m // 3 + 2 * (1 if m % 12 == 9 else 0)
        if Lmin(4, m) != want:
            print(f"FAIL w=4 3|m m={m}: Lmin {Lmin(4,m)} vs Thm11(c) {want}")
            ok = False
    # against every EXACT leave-IP row (design zone): Lmin <= L* always;
    # equality wherever no per-m obstruction was found
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    lt = os.path.join(here, "leave_table.csv")
    if os.path.exists(lt):
        for line in open(lt).readlines()[1:]:
            p = line.strip().split(",")
            if p[5] != "EXACT" or not p[4] or p[4] == "None":
                continue
            m, w, Ls = int(p[0]), int(p[1]), int(p[4])
            if 3 * (m - w) <= m - 3:
                continue
            lm = Lmin(w, m)
            if lm > Ls:
                print(f"FAIL Lmin({w},{m}) = {lm} > IP L* = {Ls}")
                ok = False
    print("Lmin selftest:", "ALL PASS" if ok else "FAILURES (above)")
    return ok


if __name__ == "__main__":
    selftest()
    # print the per-class tables (consumed by Lmin_tables.md)
    for w in range(4, 9):
        P = PERIOD[w]
        print(f"\n== w={w} (slots {comb(w,3)}, point mod {comb(w-1,2)}, "
              f"pair mod {w-2}), period {P} ==")
        base = 60 * P  # representative large m per class
        for r in range(P):
            m0 = base + r
            while m0 % P != r or m0 < w + 8:
                m0 += P
            c1, c2, beta = sig(m0, w)
            kind = ("quad" if c2 else "lin" if c1 else
                    ("perfect" if beta == 0 else "gap"))
            ex = [Lmin(w, base * k + r) for k in (1, 2)
                  if (base * k + r) >= w + 8]
            print(f"  m == {r:3d} (mod {P}): c1={c1:2d} c2={c2} "
                  f"beta={beta:3d} type={kind:7s} "
                  f"Lmin at two large m: {ex}")

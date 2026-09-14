#!/usr/bin/env python3
"""
implication_checks.py — the two implication checks requested by the
coordinator (final gate before the claims ledger). Results are summarized
in implication_checks.md.

CHECK 1: Does published machinery — Roman/Culik/DGH bounds + literature-exact
values + Guy's point-C deletion (DHS 2013 Thm 3.15) + the DHS deletion
recursion (DHS 2013 Prop 3.20) iterated to a fixpoint — already prove the
band upper bounds claimed by Theorem 8?

CHECK 2: Do the DGH (2024) Theorem 1.1 v=1 constraints (with their exact
remainder alpha), added to Roman's global counting LP, imply Lemma C's
ceiling z <= 3n + J(m) at the requested sites? Solved by exact rational LP
(Fraction simplex).

Sources for the machinery (verbatim statements read from the papers):
- DHS 2013 Thm 3.15: if ex(m,n) <= e then ex(m+c,n) <= e + c*floor(e/m) and
  ex(m,n+c) <= e + c*floor(e/n).
- DHS 2013 Prop 3.20: with U_{s,t}(m,n,a,b) = Z_{s-a,t}(m-a,b) +
  Z_{s,t}(m-a,n-b) + (a-1)n + b:
  Z_{s,t}(m,n) <= min_a max_b min{ Z_{a,b+1}(m,n), U_{s,t}(m,n,a,b) },
  over 1 <= a < s, t-1 <= b <= n.
- DGH 2024 Thm 1.1 (v=1, s=3, t=3, 3<=k<=m), alpha = 2*C(m-1,2) mod C(k-1,2):
  1/(C(k-1,2)-alpha) * sum_{i=2}^{k-1} (C(i-1,2)-alpha)*i*n_i
     + sum_{i=k}^{m} i*n_i  <=  m*(2*C(m-1,2)-alpha)/C(k-1,2).
- Lemma C (coordinator): sum_{i>=4} (i-3)*n_i <= J(m),
  J(m) = floor(m*floor(2*C(m-1,2)/3)/4).
"""
import os
import sys
from fractions import Fraction
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import published_data as P  # noqa: E402

# ----------------------------------------------------------------- oracles --
EXACT = {}
for m, row in P.TAN_Z3.items():
    for k_, (v, bold) in enumerate(row):
        if bold:
            EXACT[(m, m + k_)] = v
EXACT.update(P.BNL_EXACT)
EXACT.update({(b, a): v for (a, b), v in list(EXACT.items())})


def roman_min33(m, n):
    best = None
    for p in range(2, m + 1):
        v = Fraction(2 * comb(m, 3), comb(p, 2)) + Fraction((p + 1) * 2 * n, 3)
        v = v.numerator // v.denominator
        best = v if best is None else min(best, v)
    return best


def z2t_ub(m, n, t):
    """Roman s=2 upper bound (min over k>=1), s on the m-side; k=1 is Culik."""
    if m < 2:
        return (t - 1) * n if m == 1 else 0
    best = None
    for k in range(1, 3 * m + 3):
        v = Fraction((t - 1) * comb(m, 2), k) + Fraction((k + 1) * n, 2)
        v = v.numerator // v.denominator
        best = v if best is None else min(best, v)
    return best


def init_ub(m, n, regime):
    if m > n:
        m, n = n, m
    if m <= 2:
        return m * n  # no K_{3,3} possible with <3 rows
    vals = [roman_min33(m, n)]
    if (m, n) in P.DGH_UB33:
        vals.append(P.DGH_UB33[(m, n)])
    if regime == 'A' and (m, n) in EXACT:
        vals.append(EXACT[(m, n)])
    return min(vals)


def fixpoint(MMAX, NMAX, regime, targets):
    UB = {}
    for m in range(1, MMAX + 1):
        for n in range(1, NMAX + 1):
            UB[(m, n)] = init_ub(m, n, regime)
    # never seed a target cell with an exact/claim value: re-init from bounds
    for (m, n) in targets:
        v = [roman_min33(min(m, n), max(m, n))]
        if (min(m, n), max(m, n)) in P.DGH_UB33:
            v.append(P.DGH_UB33[(min(m, n), max(m, n))])
        UB[(m, n)] = min(v)
        UB[(n, m)] = UB[(m, n)]

    def get(m, n):
        if m < 1 or n < 1:
            return 0
        if m <= 2:
            return m * n
        if n <= 2:
            return m * n
        if (m, n) in UB:
            return UB[(m, n)]
        return roman_min33(min(m, n), max(m, n))

    changed = True
    sweeps = 0
    while changed and sweeps < 12:
        changed = False
        sweeps += 1
        for m in range(3, MMAX + 1):
            for n in range(3, NMAX + 1):
                cur = UB[(m, n)]
                best = cur
                # DHS Thm 3.15 jumps from any smaller anchor (both directions)
                for n0 in range(3, n):
                    e = get(m, n0)
                    best = min(best, e + (n - n0) * (e // n0))
                for m0 in range(3, m):
                    e = get(m0, n)
                    best = min(best, e + (m - m0) * (e // m0))
                # DHS Prop 3.20, s=t=3, alpha in {1,2}; plus transpose
                for (mm, nn) in ((m, n), (n, m)):
                    # alpha = 1
                    mx = 0
                    for b in range(2, nn + 1):
                        comp = b * mm                     # Z_{1,b+1}(mm,nn)
                        U = z2t_ub(mm - 1, b, 3) + get(mm - 1, nn - b) + b
                        mx = max(mx, min(comp, U))
                        if comp > U and b > 2 and U < best:
                            pass
                    best = min(best, mx)
                    # alpha = 2
                    mx = 0
                    for b in range(2, nn + 1):
                        comp = z2t_ub(mm, nn, b + 1)     # Z_{2,b+1}(mm,nn)
                        U = 2 * (mm - 2) + get(mm - 2, nn - b) + nn + b
                        mx = max(mx, min(comp, U))
                    best = min(best, mx)
                if best < cur:
                    UB[(m, n)] = best
                    UB[(n, m)] = best if (n, m) in UB else best
                    changed = True
    return UB, sweeps


# --------------------------------------------------- exact rational simplex --
def lp_max(c, A, b):
    """max c.x s.t. A x <= b, x >= 0; exact Fractions; Bland's rule.
    Returns optimal value (Fraction) or None if unbounded."""
    m_, n_ = len(A), len(c)
    T = [[Fraction(A[i][j]) for j in range(n_)] + [Fraction(int(i == k)) for k in range(m_)]
         + [Fraction(b[i])] for i in range(m_)]
    obj = [-Fraction(cj) for cj in c] + [Fraction(0)] * m_ + [Fraction(0)]
    basis = list(range(n_, n_ + m_))
    while True:
        piv = next((j for j in range(n_ + m_) if obj[j] < 0), None)
        if piv is None:
            return obj[-1]
        ratios = [(T[i][-1] / T[i][piv], i) for i in range(m_) if T[i][piv] > 0]
        if not ratios:
            return None
        _, r = min(ratios, key=lambda t: (t[0], basis[t[1]]))
        pv = T[r][piv]
        T[r] = [x / pv for x in T[r]]
        for i in range(m_):
            if i != r and T[i][piv] != 0:
                f = T[i][piv]
                T[i] = [T[i][j] - f * T[r][j] for j in range(n_ + m_ + 1)]
        if obj[piv] != 0:
            f = obj[piv]
            for j in range(n_ + m_ + 1):
                obj[j] -= f * T[r][j]
        basis[r] = piv


def lp_site(m, n, use_dgh_v1=False, use_dgh_v2=False, use_lemma_c=False):
    """max sum i*n_i over profiles (n_2..n_m), sum n_i = n, global budget,
    plus optional constraint families. Returns floor of LP optimum."""
    idx = list(range(2, m + 1))
    c = [i for i in idx]
    A, b = [], []
    # sum n_i <= n only: the objective is strictly monotone in every n_i and
    # size-2 columns cost nothing in every constraint's positive part, so the
    # optimum always saturates sum = n; keeping b >= 0 keeps the initial
    # slack basis feasible for the tableau simplex.
    A.append([1] * len(idx)); b.append(n)
    A.append([comb(i, 3) for i in idx]); b.append(2 * comb(m, 3))  # global
    if use_dgh_v1:
        for k in range(3, m + 1):
            mod = comb(k - 1, 2)
            if mod == 0:
                continue
            al = (2 * comb(m - 1, 2)) % mod
            den = mod - al
            row = []
            for i in idx:
                if i < k:
                    row.append(Fraction((comb(i - 1, 2) - al) * i, den))
                else:
                    row.append(Fraction(i))
            A.append(row)
            b.append(Fraction(m * (2 * comb(m - 1, 2) - al), mod))
    if use_dgh_v2:
        for k in range(3, m + 1):
            mod = k - 2
            if mod <= 0:
                continue
            al = (2 * (m - 2)) % mod
            den = mod - al
            if den == 0:
                continue
            row = []
            for i in idx:
                if i < k:
                    row.append(Fraction((max(i - 2, 0) - al) * comb(i, 2), den))
                else:
                    row.append(Fraction(comb(i, 2)))
            A.append(row)
            b.append(Fraction(comb(m, 2) * (2 * (m - 2) - al), mod))
    if use_lemma_c:
        J = (m * ((2 * comb(m - 1, 2)) // 3)) // 4
        A.append([max(i - 3, 0) for i in idx])
        b.append(J)
    v = lp_max(c, A, b)
    return None if v is None else v.numerator // v.denominator


# ------------------------------------------------------------------ CHECK 1 --
print("=" * 78)
print("CHECK 1: Guy point-C (DHS Thm 3.15) + DHS Prop 3.20 fixpoint, s=t=3")
targets1 = [(7, 20), (7, 24), (6, 10)] + [(8, nn) for nn in range(24, 28)] \
    + [(9, nn) for nn in range(40, 49)]
claims = {(7, 20): 75, (7, 24): 87, (6, 10): 39,
          (8, 24): 97, (8, 25): 100, (8, 26): 104, (8, 27): 108}
claims.update({(9, nn): 3 * nn + 40 for nn in range(40, 49)})

UBA, swA = fixpoint(9, 56, 'A', targets1)
print(f"(regime A = literature-exact anchors allowed; {swA} sweeps)")
print(f"{'cell':>9} {'claim':>6} {'Roman':>6} {'machinery':>9}  verdict")
for cell in targets1:
    m, n = cell
    r = roman_min33(m, n)
    mach = UBA[cell]
    cl = claims[cell]
    if mach <= cl:
        verdict = "REACHED by published machinery"
    elif r <= cl:
        verdict = "already Roman"
    else:
        verdict = f"NOT reached (short by {mach - cl})"
    print(f"{str(cell):>9} {cl:>6} {r:>6} {mach:>9}  {verdict}")

# regime B for (7,20): no computational exact anchors at all
UBB, _ = fixpoint(9, 30, 'B', targets1[:1])
print(f"\n(7,20) regime B (no exact anchors, bounds only): machinery gives "
      f"{UBB[(7,20)]} vs claim 75")

# ------------------------------------------------------------------ CHECK 2 --
print("=" * 78)
print("CHECK 2: DGH Thm 1.1 v=1 LP vs Lemma C ceiling 3n + J(m)")
sites = [(6, nn) for nn in range(9, 14)] + [(9, nn) for nn in range(40, 49)] \
    + [(12, nn) for nn in range(108, 117)]
print(f"{'cell':>10} {'3n+J':>6} {'LP_R':>6} {'LP_D1':>6} {'LP_D12':>7} "
      f"{'LP_C':>6}  verdict(D1 vs LemmaC)")
for (m, n) in sites:
    J = (m * ((2 * comb(m - 1, 2)) // 3)) // 4
    tgt = 3 * n + J
    lr = lp_site(m, n)
    ld1 = lp_site(m, n, use_dgh_v1=True)
    ld12 = lp_site(m, n, use_dgh_v1=True, use_dgh_v2=True)
    lc = lp_site(m, n, use_lemma_c=True)
    if ld1 <= lc:
        verdict = "DGH v=1 >= Lemma C (repackaging)"
    else:
        verdict = f"Lemma C STRICTLY stronger by {ld1 - lc}"
    print(f"{str((m,n)):>10} {tgt:>6} {lr:>6} {ld1:>6} {ld12:>7} {lc:>6}  {verdict}")

print("\nsanity: LP_R floor should equal min_p Roman (C7d): "
      + str(all(lp_site(m, n) == roman_min33(m, n)
                for (m, n) in [(6, 10), (9, 44), (12, 112)])))
# Row-7/11 note: J = T + 2 there; what do the LPs give at (7,20)?
for cell in [(7, 20), (11, 82)]:
    m, n = cell
    print(f"note {cell}: Roman={roman_min33(m,n)}, DGH-v1 LP={lp_site(m,n,True)}, "
          f"DGH v1+v2 LP={lp_site(m,n,True,True)}, LemmaC LP={lp_site(m,n,use_lemma_c=True)}, "
          f"coordinator T-bound={3*n+P.T33[m]}")

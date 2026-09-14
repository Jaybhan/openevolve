"""F(m, n) — THE MASTER FORMULA for z(m,n;3,3), fully explicit.

    F(m,n) = max over levels w in [3, min(m,13)] with C(w-1,3)*n <= B of
        (w-1)*n + min( n, D_w(m), (B - C(w-1,3)*n) // C(w-1,2) ),
    B = 2*C(m,3);  n >= m assumed (transpose otherwise: F(n,m)).

Supply spectrum D_w(m):
  w = 3 : B                                   (trivial, exact)
  w = 4 : Theorem 11 spectrum J - 2*[m == 3 (4), 3 !| m]  (exact: proven
          families + Tan/workspace anchors m <= 27, GKLO large-m)
  w >= 5, closed forms (exact, proven all m):
      wedge  m <= (3w-3)/2                  -> 2
      j=2    m = w+2: w in {5,6} -> 4, w >= 7 -> 2
      j=3    m = w+3: w=7 -> 4, w=8 -> 3, w >= 9 -> 2
      j=4    m = w+4: w in {9,10} -> 3, w >= 11 -> 2
  w >= 5, computed exact / bracket-UB (m <= 16, frozen from
      supply_status.csv of 2026-07-30, every value proven in-workspace):
      EXACT / UB dicts below (UB rows include the completion obstructions:
      (10,5) <= 21 [<=22 forced-structure + 6/7 profiles], (11,5) <= 31
      [Dehon + K5-hole kill], (12,7) <= 10 [two profile-complete kills]).
  w >= 5, large m: (B - Lmin(w, m)) / C(w,3) capped by Johnson
      (Lmin_gen.py — periodic congruence arithmetic, periods
      P = 12, 15, 20, 105, 168 for w = 4..8; levels w in [9,13] have no
      tabulated Lmin: Johnson-capped budget is used — they only matter
      in the corner where the formula is bounded-error anyway).

Status semantics (MASTER THEOREM scaffold): F is exact on every proven
regime; elsewhere upper-type with bounded error; see Lmin_tables.md and
fullpass/verification outputs. Self-contained: no file reads.
"""
from math import comb

EXACT = {(5, 5): 2, (6, 5): 2, (7, 5): 4, (8, 5): 10, (9, 5): 14, (6, 6): 2,
         (7, 6): 2, (8, 6): 4, (9, 6): 6, (10, 6): 10, (11, 6): 14,
         (12, 6): 22, (13, 6): 26, (7, 7): 2, (8, 7): 2, (9, 7): 2,
         (10, 7): 4, (11, 7): 6, (8, 8): 2, (9, 8): 2, (10, 8): 2,
         (11, 8): 3, (9, 9): 2, (10, 9): 2, (11, 9): 2, (12, 9): 2,
         (13, 9): 3, (10, 10): 2, (11, 10): 2, (12, 10): 2, (13, 10): 2,
         (14, 10): 3}
UB = {(10, 5): 21, (11, 5): 31, (12, 5): 38, (13, 5): 53, (14, 5): 72,
      (15, 5): 84, (16, 5): 105, (14, 6): 35, (15, 6): 40, (16, 6): 56,
      (12, 7): 10, (13, 7): 14, (14, 7): 16, (15, 7): 23, (16, 7): 27,
      (12, 8): 6, (13, 8): 8, (14, 8): 12, (15, 8): 15, (16, 8): 16,
      (14, 9): 6, (15, 9): 8, (16, 9): 12, (15, 10): 6, (16, 10): 8}

PERIOD = {4: 12, 5: 15, 6: 20, 7: 105, 8: 168}


def Lmin(w, m):
    pmod = comb(w - 1, 2)
    slots = comb(w, 3)
    B = 2 * comb(m, 3)
    c1 = (2 * comb(m - 1, 2)) % pmod
    c2 = (2 * (m - 2)) % (w - 2)
    beta = B % slots
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
    if w == 4 and m % 4 == 3 and m % 3 != 0 and L < 10:
        L = 10
    if w == 5 and m % 15 == 14 and L == 8:
        L = 18
    return L


def johnson(m, w):
    lam2 = (2 * (m - 2)) // (w - 2)
    r3 = ((m - 1) * lam2) // (w - 1)
    r2 = (2 * comb(m - 1, 2)) // comb(w - 1, 2)
    return (m * min(r2, r3)) // w


def D(w, m):
    """The supply spectrum."""
    B = 2 * comb(m, 3)
    if w > m:
        return 0
    if w == 3:
        return B
    if w == 4:
        J = (m * (((m - 1) * (m - 2)) // 3)) // 4
        return J - 2 if (m % 4 == 3 and m % 3 != 0) else J
    if 3 * (m - w) <= m - 3:
        return 2
    j = m - w
    if j == 2:
        return 4 if w in (5, 6) else 2
    if j == 3 and w >= 7:
        return {7: 4, 8: 3}.get(w, 2)
    if j == 4 and w >= 9:
        return {9: 3, 10: 3}.get(w, 2)
    if (m, w) in EXACT:
        return EXACT[(m, w)]
    base = min(johnson(m, w), (B - Lmin(w, m)) // comb(w, 3)) if w <= 8 \
        else min(johnson(m, w), B // comb(w, 3))
    if (m, w) in UB:
        base = min(base, UB[(m, w)])
    return max(0, base)


def F(m, n):
    """The master formula. Symmetric: F(m,n) = F(n,m)."""
    if n < m:
        m, n = n, m
    if m < 3:
        return m * n  # degenerate: no K_{3,3} possible
    B = 2 * comb(m, 3)
    best = 0
    for w in range(3, min(m, 13) + 1):
        if comb(w - 1, 3) * n > B:
            break
        k = min(n, D(w, m), (B - comb(w - 1, 3) * n) // comb(w - 1, 2))
        best = max(best, (w - 1) * n + k)
    return best


if __name__ == "__main__":
    import sys
    print(F(int(sys.argv[1]), int(sys.argv[2])))

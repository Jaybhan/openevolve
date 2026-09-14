"""Task 1 machinery: the Johnson-ladder LP at the diagonal, exactly.

Four parts, all cheap (NO solvers beyond scipy's LP, no per-cell z work):

A. EXACT INTEGER verification of the two load-bearing inequalities:
     chord_k(w) := (k-1)(w-3) + mu_k  <=  C(w-1,2)          (convexity)
     w * max(chord_k(w),0)            <=  3*C(w,3)          (domination)
   for all 3 <= w <= WMAX, 4 <= k <= KMAX.  The second line is the
   whole redundancy theorem: every summed ladder row is coefficient-
   dominated by 3x the slot row (same RHS 3B), hence redundant in any
   LP over column-weight profiles.

B. scipy/HiGHS LP: budget-only LP vs budget+ALL-ladder-rows LP
   (strongest positive-part chords) at the diagonal m = 100..800 and at
   band cells.  Expected: identical optima (Part A proves it); this is
   the independent numeric check.

C. EXACT (Fraction) closed form of the diagonal budget LP (two adjacent
   weights saturating columns + slots), values for m up to 10^6,
   convergence E/m^{5/3} -> 2^{1/3} with fitted correction exponent.

D. Dual certificate of the scaled problem
       max A1  s.t.  A0 <= 1, A3 <= 2   (ladder A2^2 <= 2A1 slack-free)
   value 2^{1/3} via  t <= (2/3)2^{1/3} + t^3/(3*2^{2/3})  for all t>=0.
"""
import numpy as np
from fractions import Fraction
from math import comb
from scipy.optimize import linprog

C = comb


def mu(k):
    return (k - 1) * (4 - k) // 2


def chord(k, w):
    return (k - 1) * (w - 3) + mu(k)


# ---------------------------------------------------------------- A
def part_A(WMAX=3000, KMAX=3000):
    w = np.arange(3, WMAX + 1, dtype=np.int64)
    Cw12 = (w - 1) * (w - 2) // 2                    # C(w-1,2)
    Cw3 = w * (w - 1) * (w - 2) // 6                 # C(w,3)
    bad1 = bad2 = 0
    for k in range(4, KMAX + 1):
        mk = (k - 1) * (4 - k) // 2
        ch = (k - 1) * (w - 3) + mk
        if np.any(ch > Cw12):
            bad1 += int(np.sum(ch > Cw12))
        chp = np.maximum(ch, 0)
        if np.any(w * chp > 3 * Cw3):
            bad2 += int(np.sum(w * chp > 3 * Cw3))
    print(f"[A] chord_k(w) <= C(w-1,2): violations = {bad1} "
          f"(w<={WMAX}, k<={KMAX})")
    print(f"[A] w*chord_k^+(w) <= 3*C(w,3): violations = {bad2}")
    # equality structure: chord touches C(w-1,2) exactly at w in {k,k+1}
    eq_ok = all(chord(k, k) == C(k - 1, 2) and chord(k, k + 1) == C(k, 2)
                for k in range(4, 200))
    print(f"[A] tangency chord_k(k)=C(k-1,2), chord_k(k+1)=C(k,2): {eq_ok}")


# ---------------------------------------------------------------- B
def lp_value(m, n, ladder=False):
    """max sum w*x_w ; sum x_w <= n ; slots <= B ; (ladder rows).
    weights w = 2..m (w=2 pads cost 0 slots)."""
    ws = np.arange(2, m + 1)
    nv = len(ws)
    cobj = -ws.astype(float)
    B = 2 * C(m, 3)
    rows = [np.ones(nv)]
    rhs = [n]
    slot = np.array([C(int(w), 3) for w in ws], dtype=float)
    rows.append(slot)
    rhs.append(B)
    if ladder:
        for k in range(4, m + 1):
            ch = (k - 1) * (ws - 3) + mu(k)
            coeff = ws * np.maximum(ch, 0)
            if np.any(coeff > 0):
                rows.append(coeff.astype(float))
                rhs.append(3 * B)
    res = linprog(cobj, A_ub=np.array(rows), b_ub=np.array(rhs),
                  bounds=[(0, None)] * nv, method="highs")
    assert res.status == 0, res.message
    return -res.fun


def part_B():
    print("\n[B] LP optima: budget-only vs + all ladder rows "
          "(positive-part chords)")
    print(f"{'cell':>12} {'budget LP':>16} {'ladder LP':>16} {'diff':>12}")
    cells = [(100, 100), (200, 200), (400, 400), (800, 800),
             (8, 30), (9, 44), (11, 60), (12, 100)]
    out = {}
    for m, n in cells:
        v0 = lp_value(m, n, ladder=False)
        v1 = lp_value(m, n, ladder=True)
        out[(m, n)] = (v0, v1)
        print(f"({m:>4},{n:>4}) {v0:>16.6f} {v1:>16.6f} {v1 - v0:>12.2e}")
    print("\n[B] diagonal normalisation  E_LP / m^(5/3)  vs 2^(1/3) = "
          f"{2 ** (1 / 3):.9f}")
    for m in (100, 200, 400, 800):
        v = out[(m, m)][0]
        print(f"  m={m:>4}: {v / m ** (5 / 3):.9f}   "
              f"(excess {v / m ** (5 / 3) - 2 ** (1 / 3):+.6f})")


# ---------------------------------------------------------------- C
def diag_lp_exact(m):
    """Exact optimal value (Fraction) of: max sum w x_w, sum x_w <= m,
    sum C(w,3) x_w <= B, 0 <= x, weights 2..m.  Optimum: adjacent pair
    (w0, w0+1) saturating both constraints (waterfill/convexity)."""
    B = 2 * C(m, 3)
    # largest w0 with m*C(w0,3) <= B
    w0 = 3
    while m * C(w0 + 1, 3) <= B and w0 + 1 <= m:
        w0 += 1
    if w0 >= m:                       # column-bound regime (tiny m only)
        return Fraction(m * m)
    # x0 + x1 = m ; C(w0,3)x0 + C(w0+1,3)x1 = B
    c0, c1 = C(w0, 3), C(w0 + 1, 3)
    x1 = Fraction(B - m * c0, c1 - c0)
    assert 0 <= x1 <= m
    return m * w0 + x1, w0


def part_C():
    print("\n[C] exact diagonal budget-LP value, convergence to 2^(1/3)")
    print(f"{'m':>8} {'w0':>7} {'E_LP/m^(5/3)':>16} {'(E-2^(1/3)m^(5/3))/m':>22}")
    t = 2 ** (1 / 3)
    prev = None
    data = []
    for m in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600,
              10 ** 5, 10 ** 6]:
        E, w0 = diag_lp_exact(m)
        Ef = float(E)
        r = Ef / m ** (5 / 3)
        lin = (Ef - t * m ** (5 / 3)) / m
        data.append((m, r, lin))
        print(f"{m:>8} {w0:>7} {r:>16.9f} {lin:>22.6f}")
    # fit: r - 2^(1/3) ~ a*m^(-2/3): check doubling ratio -> 2^(2/3)
    print("  doubling ratios of (r - 2^(1/3)) "
          "[predict 2^(-2/3) = 0.63 if exponent -2/3]:")
    for (m1, r1, _), (m2, r2, _) in zip(data, data[1:]):
        if m2 == 2 * m1:
            print(f"    m {m1}->{m2}: {(r2 - t) / (r1 - t):.4f}")


# ---------------------------------------------------------------- D
def part_D():
    print("\n[D] dual certificate  t <= (2/3)2^(1/3) + t^3/(3*2^(2/3)):")
    lam = Fraction(2, 3)
    ts = np.linspace(0, 20, 2000001)
    f = (2 / 3) * 2 ** (1 / 3) + ts ** 3 / (3 * 2 ** (2 / 3)) - ts
    print(f"    min over t in [0,20] of RHS-LHS = {f.min():.3e} "
          f"(attained at t = {ts[f.argmin()]:.6f}; predict 2^(1/3) = "
          f"{2 ** (1 / 3):.6f})")
    print(f"    dual value lam*1 + mu*2 = "
          f"{(2 / 3) * 2 ** (1 / 3) + 2 / (3 * 2 ** (2 / 3)):.9f} "
          f"= 2^(1/3) = {2 ** (1 / 3):.9f}")


# ---------------------------------------------------------------- E
def waterfill_int(m, n):
    """Integer waterfill (= Roman/WF bound of the workspace)."""
    B = 2 * C(m, 3)
    w = [2] * n
    spend = 0
    for lev in range(3, m + 1):
        cost = C(lev - 1, 2)
        for i in range(n):
            if spend + cost <= B:
                w[i] += 1
                spend += cost
            else:
                return sum(w)
    return sum(w)


def part_E():
    print("\n[E] integer WF at the diagonal vs Theorem 12's printed "
          "'diagonal ladder values' (150,165,180) and WF(16,16)=136:")
    for m in (16, 17, 18, 19):
        print(f"    WF({m},{m}) = {waterfill_int(m, m)}")


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
    part_D()
    part_E()

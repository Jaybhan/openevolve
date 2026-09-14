"""Theorem M4 (finite elementary diagonal bound from the M1 chain).

Valid chain, per K33-free m x m matrix with E ones:
  blocks = columns of weight >= 3 (p pad columns of weight <= 2 only
  tighten the bound: each pad removes ~0.6 m^{2/3} from the block bound
  and returns at most 2 ones);
  E2 := sum_B w(w-1) = sum_x Y_x,  sum_x d_x = sum_B w = E_b;
  per point (M1 exact):  Y_x <= G(d_x),
     G(d) = M + sqrt(2M*(C(d,2) + sqrt(2*C(d,2)*C(M,2)))),  M = m-1;
  G concave on [2, m] (machine-checked; asymptotically 2^{1/4} M sqrt d)
  => sum Y_x <= m*G(E_b/m)  [Jensen, concave direction — VALID];
  Cauchy-Schwarz over <= m blocks: E2 >= E_b^2/m - E_b.
  => z(m,m) <= max{ E : E^2/m - E <= m*G(E/m) }.

This beats WF/Roman at every m >= 180 (m** = 180 computed below) and
is the best workspace diagonal bound on roughly 180 <= m <= 550,
after which Fueredi's printed bound takes over. Asymptotics:
2^{1/6} m^{5/3} (Theorem M2).
"""
import numpy as np
from math import comb, sqrt


def G(d, M):
    if d < 2:
        return d * M
    N = d * (d - 1) / 2
    S = N + sqrt(2 * N * comb(M, 2))
    return M + sqrt(2 * M * S)


def concave_ok(M, npts=2000):
    ds = np.linspace(2, M + 1, npts)
    g = np.array([G(d, M) for d in ds])
    return bool((np.diff(g, 2) <= 1e-7 * g.max()).all())


def m4_bound(m):
    M = m - 1
    E = m * m
    while E > 0:
        if E * E / m - E <= m * G(E / m, M) + 1e-9:
            return E
        E -= 1
    return 0


def wf(m):
    B = 2 * comb(m, 3)
    w = [2] * m
    sp = 0
    for lev in range(3, m + 1):
        c = comb(lev - 1, 2)
        for i in range(m):
            if sp + c <= B:
                w[i] += 1
                sp += c
            else:
                return sum(w)
    return sum(w)


def fueredi(m):
    return m * m ** (2 / 3) + 2 * m ** (4 / 3) + m


if __name__ == "__main__":
    print("concavity of G on [2,m]:",
          all(concave_ok(M) for M in (16, 99, 199, 999, 3199)))
    print(f"{'m':>6} {'WF/Roman':>10} {'M4':>10} {'Fueredi':>10} "
          f"{'best':>8} {'M4/m^(5/3)':>11}")
    for m in (100, 150, 180, 200, 300, 400, 500, 600, 800, 1600, 3200):
        a, b, f = wf(m), m4_bound(m), int(fueredi(m))
        best = min(a, b, f)
        tag = "M4" if best == b else ("WF" if best == a else "Fu")
        print(f"{m:>6} {a:>10} {b:>10} {f:>10} {tag:>8} "
              f"{b / m ** (5/3):>11.4f}")
    lo, hi = 17, 3200
    while lo < hi:
        mid = (lo + hi) // 2
        if m4_bound(mid) < wf(mid):
            hi = mid
        else:
            lo = mid + 1
    print(f"\nm** (first m with M4 < WF): {lo}   "
          f"[m={lo-1}: WF={wf(lo-1)} M4={m4_bound(lo-1)}; "
          f"m={lo}: WF={wf(lo)} M4={m4_bound(lo)}]")
    # M4-vs-Fueredi crossover
    lo2, hi2 = 180, 3200
    while lo2 < hi2:
        mid = (lo2 + hi2) // 2
        if fueredi(mid) < m4_bound(mid):
            hi2 = mid
        else:
            lo2 = mid + 1
    print(f"Fueredi overtakes M4 at m = {lo2}")

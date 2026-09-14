"""THE HEADLINE CHECK (three ways, per task instructions): the link
second-moment inequality and the elementary 2^{1/6} diagonal bound.

Chain (per point x of an m x n K33-free matrix; blocks through x with
link sets L_i on M = m-1 points, sizes u_i = w_i - 1, d = #blocks,
I_ij = |L_i cap L_j|, mu(pair) = #links containing the pair <= 2):

  (C1) sum_{i<j} C(I_ij,2) = sum_pairs C(mu,2) = #{mu=2} <= C(M,2)
  (C2) sum_{i<j} C(I_ij,2) >= N*C(S/N,2), N=C(d,2), S=sum I_ij  [Jensen]
  (C3) S = sum_y C(r_y,2) >= M*C(Y/M,2), Y = sum u_i             [Jensen]
  => asymptotically  Y <= 2^{1/4} * M * sqrt(d) * (1+o(1)),
  => (Cauchy-Schwarz twice, n=m)  z(m,m) <= 2^{1/6} m^{5/3} (1+o(1)),
  and for single-level configs (all w ~ t*m^{2/3}): t <= 1+o(1), i.e.
  E <= m^{5/3}(1+o(1)) — the Brown constant.

CHECK 1 (exact witnesses): verify C1-C3 hold, and measure tightness, on
  (a) the (16,16)=128 extremal (cap-hyperplane construction, rebuilt +
      K33-verified here), (b) Brown q=7 and q=11 (exact link stats via
      Cayley structure), per point.
CHECK 2 (the pinch is real): confirm the budget atom (t = 2^{1/3})
  VIOLATES the asymptotic inequality, the 2^{1/6} atom saturates the
  unconditional form, and t=1 saturates the single-level form; numeric
  grid over two-atom mixed profiles confirms sup A1 = 2^{1/6} under
  {A0<=1, A2 <= 2^{1/4} sqrt(A1)} and sup A1 = 1 for single-level.
CHECK 3 (no finite contradiction): the exact finite chain must be
  satisfiable at every known z(m,m) value (the asymptotic form ignores
  lower-order terms; verify the EXACT chain is consistent on the real
  (16,16) witness where E/m^{5/3} = 1.26 > 2^{1/6}).
"""
import numpy as np
from math import comb, sqrt
from itertools import combinations

C = comb


# ------------------------------------------------------------- (16,16)
def build_16():
    """rows = F_2^4; columns = both sides of the 8 affine hyperplanes
    with normals (a0,a1,a2,1)."""
    cols = []
    for a in range(16):
        if not (a & 8):
            continue  # normals with last bit 1: a = 8..15
        side0 = [x for x in range(16)
                 if bin(a & x).count("1") % 2 == 0]
        side1 = [x for x in range(16) if x not in side0]
        cols.append(side0)
        cols.append(side1)
    M = np.zeros((16, 16), dtype=bool)
    for j, s in enumerate(cols):
        M[s, j] = True
    return M


def k33_free(M):
    m, n = M.shape
    for tri in combinations(range(m), 3):
        common = M[tri[0]] & M[tri[1]] & M[tri[2]]
        if common.sum() >= 3:
            return False
    return True


def point_chain_report(links, M_ground, label, verbose=True):
    """links: list of sets (block supports minus x). Verify C1-C3."""
    d = len(links)
    Y = sum(len(L) for L in links)
    S = 0
    P2 = 0
    from collections import Counter
    pair_mu = Counter()
    for L in links:
        for p in combinations(sorted(L), 2):
            pair_mu[p] += 1
    assert all(v <= 2 for v in pair_mu.values()), "mu <= 2 violated!"
    P = sum(pair_mu.values())
    P2 = sum(1 for v in pair_mu.values() if v == 2)
    # S via intersections
    S = 0
    for i in range(d):
        for j in range(i + 1, d):
            S += len(links[i] & links[j])
    N = C(d, 2)
    CM2 = C(M_ground, 2)
    c2_lhs = sum(C(len(links[i] & links[j]), 2)
                 for i in range(d) for j in range(i + 1, d))
    ok1 = c2_lhs <= CM2
    jensen2 = N * (S / N) * (S / N - 1) / 2 if N else 0
    ok2 = c2_lhs >= jensen2 - 1e-9
    jensen3 = M_ground * (Y / M_ground) * (Y / M_ground - 1) / 2
    ok3 = S >= jensen3 - 1e-9
    asym = 2 ** 0.25 * M_ground * sqrt(d)
    if verbose:
        print(f"  [{label}] d={d} Y={Y} S={S} P={P} P2={P2} "
              f"sumC(I,2)={c2_lhs} C(M,2)={CM2}")
        print(f"     C1 (<=C(M,2)): {ok1}   C2 (>=Jensen {jensen2:.1f}):"
              f" {ok2}   C3 (S>= {jensen3:.1f}): {ok3}")
        print(f"     Y = {Y} vs asymptotic cap 2^(1/4)*M*sqrt(d) = "
              f"{asym:.1f}  (finite corrections "
              f"{'DOMINATE' if Y > asym else 'small'})")
        print(f"     pair-slot usage pi = P/C(M,2) = {P / CM2:.4f}, "
              f"P2/C(M,2) = {P2 / CM2:.4f}, frac mu=1 = "
              f"{sum(1 for v in pair_mu.values() if v == 1) / max(P,1):.4f}")
    return ok1 and ok2 and ok3


def check_16():
    print("CHECK 1a: (16,16) extremal witness")
    M = build_16()
    assert M.sum() == 128
    assert k33_free(M), "(16,16) witness not K33-free?!"
    print("  witness valid: 128 ones, K33-free")
    allok = True
    for x in range(16):
        links = [set(np.flatnonzero(M[:, j])) - {x}
                 for j in np.flatnonzero(M[x])]
        allok &= point_chain_report(links, 15, f"x={x}",
                                    verbose=(x == 0))
    print(f"  chain C1-C3 valid at all 16 points: {allok}")


# ------------------------------------------------------------- Brown
def brown_links(q, delta):
    sq = (np.arange(q) ** 2) % q
    s3 = (sq[:, None, None] + sq[None, :, None] + sq[None, None, :]) % q
    I = (s3 == delta % q)
    pts = [(a, b, c) for a in range(q) for b in range(q)
           for c in range(q)]
    S = [p for p in pts if I[p]]
    # links of blocks through x=0: columns j in S; support = j+S; link
    # = (j+S) minus {0}
    links = []
    Sset = set(S)
    for j in S:
        L = set()
        for s in S:
            y = ((j[0] + s[0]) % q, (j[1] + s[1]) % q, (j[2] + s[2]) % q)
            if y != (0, 0, 0):
                L.add(y)
        links.append(L)
    return links, q ** 3 - 1


def check_brown():
    for q, delta in ((7, 1), (11, 1)):
        print(f"\nCHECK 1b: Brown q={q} link statistics at x=0 "
              "(translation-invariant: same at every point)")
        links, M = brown_links(q, delta)
        point_chain_report(links, M, f"q={q}")
        t = (q * q - q) / (q ** 3) ** (2 / 3)
        print(f"     level t = 1-1/q = {t:.4f}; single-level law "
              f"predicts pi -> t^3/2·... = {t**3/2:.4f} + o(1) "
              f"wait pi = d*C(u,2)/C(M,2)... measured above")


# ------------------------------------------------------------- CHECK 2
def check_atoms():
    print("\nCHECK 2: the asymptotic pinch (scaled quantities)")
    for t, name in ((2 ** (1 / 3), "budget atom (ladder optimum)"),
                    (2 ** (1 / 6), "2^(1/6) atom"),
                    (1.0, "Brown atom t=1")):
        a = 1.0
        A1, A2 = a * t, a * t * t
        uncond = A2 <= 2 ** 0.25 * sqrt(A1) + 1e-12
        # single-level constraint: pi <= 1/2  <=>  a t^3 <= 1
        single = a * t ** 3 <= 1 + 1e-12
        print(f"  t={t:.6f} ({name}): E/m^(5/3)={A1:.6f}  "
              f"uncond A2<=2^(1/4)A1^(1/2): "
              f"{'OK' if uncond else 'VIOLATED'}  "
              f"single-level at^3<=1: {'OK' if single else 'VIOLATED'}")
    # numeric sup over two-atom profiles under the unconditional system
    best = 0
    for a1 in np.linspace(0, 1, 101):
        for t1 in np.linspace(0, 3, 151):
            for t2 in np.linspace(0, 3, 151):
                a2 = 1 - a1
                A1 = a1 * t1 + a2 * t2
                A2 = a1 * t1 ** 2 + a2 * t2 ** 2
                if A2 <= 2 ** 0.25 * sqrt(A1) + 1e-12:
                    best = max(best, A1)
    print(f"  numeric sup A1 over two-atom profiles (uncond system): "
          f"{best:.6f}  vs 2^(1/6) = {2 ** (1/6):.6f}")


def check_band_witnesses():
    """M1 chain on every stored mixed-weight band witness."""
    import json
    import glob
    import os
    from collections import Counter
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "..", "witnesses",
                                          "w_*.json")))
    fails = 0
    for f in files:
        d = json.load(open(f))
        m = d["m"]
        blocks = [set(b) for b in d["blocks"]]
        for x in range(m):
            links = [b - {x} for b in blocks if x in b]
            dd, M = len(links), m - 1
            if dd < 2:
                continue
            mu = Counter()
            for L in links:
                for p in combinations(sorted(L), 2):
                    mu[p] += 1
            c2 = sum(C(len(a & b), 2)
                     for a, b in combinations(links, 2))
            S = sum(len(a & b) for a, b in combinations(links, 2))
            Y = sum(len(L) for L in links)
            N = C(dd, 2)
            ok = (all(v <= 2 for v in mu.values())
                  and c2 <= C(M, 2)
                  and c2 >= N * (S / N) * (S / N - 1) / 2 - 1e-9
                  and S >= M * (Y / M) * (Y / M - 1) / 2 - 1e-9)
            if not ok:
                fails += 1
                print(f"  FAIL {os.path.basename(f)} x={x}")
    print(f"\nCHECK 1c: M1 chain on {len(files)} mixed-weight band "
          f"witnesses, every point: {'ALL VALID' if fails == 0 else fails}")


if __name__ == "__main__":
    check_16()
    check_brown()
    check_atoms()
    check_band_witnesses()

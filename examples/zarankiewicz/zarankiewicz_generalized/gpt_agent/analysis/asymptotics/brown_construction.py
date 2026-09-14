"""Task 2: Brown's K_{3,3}-free graphs, built exactly and fully verified.

Construction (Brown 1966, both residue classes):
  V = F_q^3 (q odd prime), x ~ y  iff  Q(x-y) = delta,  Q = sum of squares.
  Legality condition: chi(-delta) = -1 (minus delta a non-residue):
    q = 3 mod 4: delta = 1 (unit sphere; chi(-1) = -1);
    q = 1 mod 4: delta = smallest non-residue (chi(-1) = +1).

Bipartite normalization: M[x,y] = 1 iff y - x in S, S = sphere(delta).
n = q^3 rows = cols, E = q^3 * |S| = q^5 - q^4 (|S| = q^2 - q exactly
when chi(-delta) = -1).

FULL K33-freeness check via translation invariance (Cayley structure):
  M has a 3x3 all-ones submatrix iff there exist u != v, both != 0, with
      T(u,v) := |S cap (S+u) cap (S+v)| >= 3.
  (Rows {a,b,c} -> u = b-a, v = c-a; common columns = a + (S cap (S+u)
   cap (S+v)); column-triples give nothing new since M is symmetric.)
  We compute max_{u,v} T(u,v) EXACTLY for every q by looping u over all
  q^3-1 values, W_u = S cap (S+u) (avg size ~ q), and accumulating
  cnt[v] = sum_{x in W_u} I_S[x - v] via 3D rolls. Cost ~ q^7 int8 ops.

Structural certificate (the algebra behind it, checked per q):
  for every isotropic direction u (Q(u) = 0, u != 0):
      { w : u.w = 0 and Q(w) = delta } is EMPTY.
  This is the exact degenerate case (radical planes of collinear centers
  coincide iff the direction is isotropic; the common section is then
  this plane section). Non-degenerate triples give sphere cap line <= 2.

Control: q = 7 with the WRONG delta (chi(-delta) = +1) must contain
K33; predicted worst triple = isotropic-collinear with T = 2q.

Output: brown_data.csv + printed report.
"""
import csv
import os
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QS = [7, 11, 13, 17, 19, 23]


def chi_table(q):
    """quadratic character on F_q: chi[0]=0, chi[QR]=1, chi[NQR]=-1."""
    ch = -np.ones(q, dtype=np.int64)
    ch[0] = 0
    ch[np.unique((np.arange(1, q) ** 2) % q)] = 1
    return ch


def pick_delta(q):
    ch = chi_table(q)
    if q % 4 == 3:
        d = 1
    else:
        d = int(np.flatnonzero(ch[1:] == -1)[0]) + 1  # smallest NQR
    assert ch[(-d) % q] == -1, "chi(-delta) must be -1"
    return d


def sphere_indicator(q, delta):
    """I[x0,x1,x2] = 1 iff x0^2+x1^2+x2^2 = delta, as (q,q,q) int8."""
    sq = (np.arange(q) ** 2) % q
    s3 = (sq[:, None, None] + sq[None, :, None] + sq[None, None, :]) % q
    return (s3 == delta % q).astype(np.int8)


def max_triple_common(q, I):
    """Exact max over u != v (both nonzero) of |S cap (S+u) cap (S+v)|.
    Returns (maxT, argmax (u,v) or None)."""
    S_pts = np.argwhere(I == 1)          # (|S|, 3)
    Iflat = I.ravel()
    best, arg = 0, None
    # iterate over u (all nonzero points); W_u = S cap (S+u)
    for u0 in range(q):
        for u1 in range(q):
            for u2 in range(q):
                if u0 == u1 == u2 == 0:
                    continue
                Iu = np.roll(I, (u0, u1, u2), axis=(0, 1, 2))  # I_{S+u}
                W = np.argwhere((I & Iu) == 1)
                if len(W) < 3:
                    continue  # T(u,v) <= |W_u| < 3 for every v
                cnt = np.zeros((q, q, q), dtype=np.int16)
                for x in W:
                    # add I_{x - S}[v] = I_{S + x}[v]  (S symmetric)
                    cnt += np.roll(I, (x[0], x[1], x[2]), axis=(0, 1, 2))
                cnt[0, 0, 0] = 0
                cnt[u0, u1, u2] = 0
                t = int(cnt.max())
                if t > best:
                    best = t
                    v = np.unravel_index(cnt.argmax(), cnt.shape)
                    arg = ((u0, u1, u2), tuple(int(a) for a in v))
    return best, arg


def isotropic_sections(q, delta):
    """For every nonzero u with Q(u)=0: size of {w: u.w=0, Q(w)=delta}.
    Returns (n_isotropic_dirs, max_section_size)."""
    rng = np.arange(q)
    X0, X1, X2 = np.meshgrid(rng, rng, rng, indexing="ij")
    P = np.stack([X0.ravel(), X1.ravel(), X2.ravel()], 1)  # all q^3 pts
    Qv = (P[:, 0] ** 2 + P[:, 1] ** 2 + P[:, 2] ** 2) % q
    iso = P[(Qv == 0) & ~np.all(P == 0, 1)]
    sph = P[Qv == delta % q]
    mx, cnt = 0, 0
    for u in iso:
        cnt += 1
        dots = (sph @ u) % q
        sec = int(np.sum(dots == 0))
        mx = max(mx, sec)
    return cnt, mx


def verify_row_regularity(q, I):
    """Bipartite E = q^3*|S| by Cayley regularity; verify explicitly on
    a full small matrix (q=7) or by construction elsewhere."""
    S = int(I.sum())
    return q ** 3 * S, S


def brute_full_explicit(q, delta):
    """Independent cross-validation of the Cayley reduction: build the
    explicit n x n bipartite matrix and check EVERY row-triple via the
    pair method. Feasible at q = 7 (n = 343)."""
    n = q ** 3
    I = sphere_indicator(q, delta)
    Iflat = I.ravel()
    rng = np.arange(q)
    X0, X1, X2 = np.meshgrid(rng, rng, rng, indexing="ij")
    P = np.stack([X0.ravel(), X1.ravel(), X2.ravel()], 1)
    # M[x,y] = I[(y-x) mod q]
    D0 = (P[None, :, 0] - P[:, None, 0]) % q
    D1 = (P[None, :, 1] - P[:, None, 1]) % q
    D2 = (P[None, :, 2] - P[:, None, 2]) % q
    M = Iflat[(D0 * q + D1) * q + D2].astype(bool)
    assert M.sum() == q ** 5 - q ** 4
    assert np.array_equal(M, M.T) and not M.diagonal().any()
    worst = 0
    for x in range(n):
        Ax = M[x]
        for y in range(x + 1, n):
            S_xy = Ax & M[y]
            if S_xy.sum() < 3:
                continue
            c = M[:, S_xy].sum(1)
            c[x] = c[y] = 0
            worst = max(worst, int(c.max()))
            if worst >= 3:
                return worst
    return worst


def main():
    print("=" * 72)
    rows = []
    for q in QS:
        t0 = time.time()
        delta = pick_delta(q)
        I = sphere_indicator(q, delta)
        E, Ssz = verify_row_regularity(q, I)
        n = q ** 3
        assert Ssz == q * q - q, (q, Ssz)
        # symmetric & no loops
        assert I[0, 0, 0] == 0
        neg = np.roll(I[::-1, ::-1, ::-1], (1, 1, 1), axis=(0, 1, 2))
        assert np.array_equal(I, neg), "S must be symmetric"
        maxT, arg = max_triple_common(q, I)
        niso, maxsec = isotropic_sections(q, delta)
        dens = 1 - 1 / q
        rows.append(dict(q=q, delta=delta, n=n, sphere=Ssz, E=E,
                         maxT=maxT, iso_dirs=niso, iso_maxsec=maxsec,
                         density_exact=f"1-1/{q}",
                         density=E / n ** (5 / 3),
                         second_order=(E - n ** (5 / 3)) / n ** (4 / 3)))
        status = "K33-FREE" if maxT <= 2 else f"VIOLATION {arg}"
        print(f"q={q:>2} delta={delta} n={n:>6} |S|={Ssz:>4} "
              f"E={E:>9} maxT={maxT} [{status}] iso_dirs={niso} "
              f"max_iso_section={maxsec}  ({time.time()-t0:.1f}s)")
        assert maxT <= 2, f"NOT K33-free at q={q}"
        assert maxsec == 0, f"isotropic section nonempty at q={q}"
        assert E == q ** 5 - q ** 4
    # independent cross-validation of the Cayley reduction at q = 7:
    t0 = time.time()
    worst = brute_full_explicit(7, pick_delta(7))
    print(f"\n[cross-validation] q=7 explicit 343x343 matrix, ALL "
          f"C(343,3)=6.6M row-triples via pair method: max common "
          f"columns = {worst} ({time.time()-t0:.1f}s) — must equal "
          "the Cayley maxT above")
    assert worst <= 2

    # control: wrong delta at q=7
    q = 7
    ch = chi_table(q)
    bad = next(d for d in range(1, q) if ch[(-d) % q] == 1)
    Ibad = sphere_indicator(q, bad)
    maxT, arg = max_triple_common(q, Ibad)
    niso, maxsec = isotropic_sections(q, bad)
    print(f"\n[control] q=7 delta={bad} (chi(-delta)=+1): sphere="
          f"{int(Ibad.sum())}, maxT = {maxT} (predict 2q={2*q}), "
          f"max isotropic section = {maxsec} (predict 2q={2*q}) -> "
          f"{'K33 PRESENT as predicted' if maxT >= 3 else 'UNEXPECTED'}")
    assert maxT >= 3

    with open(os.path.join(HERE, "brown_data.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- second-order fit: 1 - E/n^{5/3} = a * n^{-b}
    print("\nsecond-order fit  1 - e/n^(5/3) = a * n^(-b):")
    x = np.log([r["n"] for r in rows])
    y = np.log([1 - r["E"] / r["n"] ** (5 / 3) for r in rows])
    b, la = np.polyfit(x, y, 1)
    print(f"  b = {-b:.12f}   (exact: 1/3)")
    print(f"  a = {np.exp(la):.12f}   (exact: 1)")
    resid = y - (b * x + la)
    print(f"  max |residual| = {np.abs(resid).max():.3e}  "
          "(law is EXACT: e = n^(5/3) - n^(4/3) at n = q^3)")
    print("\ntable: q, 1-1/q vs measured e/n^(5/3):")
    for r in rows:
        print(f"  q={r['q']:>2}: e/n^(5/3) = {r['density']:.9f} "
              f"= 1 - 1/q = {1 - 1 / r['q']:.9f}  "
              f"second-order coeff = {r['second_order']:+.6f}")


if __name__ == "__main__":
    main()

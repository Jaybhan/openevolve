"""Conjecture B falsification probe #1 (construction-evaluation only,
no solver): greedily add ones to the q=7 Brown matrix (343x343, 14406
ones) keeping K33-freeness. Output: a valid lower bound
z(343,343) >= 14406 + G and the measured addability profile.

Add (x,y) is legal iff no rows {x,a,b} and cols {y,c,d} become all-ones:
  for all pairs a,b in supp(col y) minus x:
      |N(x) cap N(a) cap N(b)  minus {y}| <= 1,
and symmetrically no violation with x in the middle... (rows {x,a,b}
is the general form containing the new cell; pairs (a,b) from col y's
support cover it). Since the matrix stays symmetric-agnostic (we add
asymmetric ones), we must ALSO guard col-side: cols {y,c,d}, rows
{x,a,b} — same condition (the 3x3 needs (x,y); the row-pair condition
above is exactly it). One condition suffices.
"""
import numpy as np
import time

rng = np.random.default_rng(20260730)


def build_brown(q=7, delta=1):
    sq = (np.arange(q) ** 2) % q
    s3 = (sq[:, None, None] + sq[None, :, None] + sq[None, None, :]) % q
    I = (s3 == delta).ravel()
    n = q ** 3
    idx = np.arange(n)
    a0, a1, a2 = idx // (q * q), (idx // q) % q, idx % q
    D0 = (a0[None, :] - a0[:, None]) % q
    D1 = (a1[None, :] - a1[:, None]) % q
    D2 = (a2[None, :] - a2[:, None]) % q
    return I[(D0 * q + D1) * q + D2].copy()


def addable(A, x, y):
    """check adding (x,y) creates no 3x3 all-ones through (x,y)."""
    R = np.flatnonzero(A[:, y])
    R = R[R != x]
    if len(R) < 2:
        return True
    V = A[R] & A[x]              # |R| x n bool: common cols with x
    V[:, y] = False
    P = V.astype(np.float32) @ V.astype(np.float32).T
    np.fill_diagonal(P, 0)
    return P.max() <= 1.5        # need <= 1 common col for every pair


def main():
    A = build_brown()
    n = A.shape[0]
    E0 = int(A.sum())
    print(f"Brown q=7: n={n}, E0={E0}")
    zeros = np.argwhere(~A)
    zeros = zeros[rng.permutation(len(zeros))]
    added = 0
    tried = 0
    t0 = time.time()
    budget_s = 240.0
    for (x, y) in zeros:
        if time.time() - t0 > budget_s:
            break
        tried += 1
        if addable(A, x, y):
            A[x, y] = True
            added += 1
    E1 = int(A.sum())
    print(f"tried {tried}/{len(zeros)} candidates in "
          f"{time.time()-t0:.0f}s; added {added}")
    print(f"z(343,343) >= {E1}   (Brown 14406 + {added})")
    print(f"densities: Brown {E0 / 343 ** (5/3):.4f} -> extended "
          f"{E1 / 343 ** (5/3):.4f}  (budget-LP scale "
          f"{2 ** (1/3) + 1 / 343 ** (2/3):.4f}, truth window "
          "[Brown, budget])")
    # verify final matrix exhaustively via pair method (safety)
    print("verifying extended matrix K33-free (pair method)...")
    t0 = time.time()
    worst = 0
    for x in range(n):
        Ax = A[x]
        for yy in range(x + 1, n):
            S = Ax & A[yy]
            if S.sum() < 3:
                continue
            c = A[:, S].sum(1)
            c[x] = c[yy] = 0
            worst = max(worst, int(c.max()))
            if worst >= 3:
                print("VIOLATION", x, yy)
                return
    print(f"verified: max triple-common = {worst} <= 2 "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

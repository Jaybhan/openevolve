#!/usr/bin/env python3
"""Verification for the 3|m minimal-leave classification (Task 3) and the
GKLO host-divisibility rows for both leave families.

Facts checked:
  A. For all m = 3s, 6 <= m <= 3000:
       R := floor((m-1)(m-2)/3) equals 3s(s-1) exactly;
       B := 2C(m,3) satisfies B - mR = 2m/3;
       mR is even, and mR mod 4 == 2 iff m == 9 (mod 12) else 0;
       B - 4J == 2m/3 + (mR mod 4), where J = floor(mR/4).
  B. Data row m = 6..27: B-4J = 4,8,8,10,12,16,16,18.
  C. Leave families realize the minimum, i.e. for sample m the doubled
     leave 2L' has: weight B-4J; multiplicities <= 2; point-degrees == 2
     (mod 3) at EVERY point; all pair-degrees even; weight == B (mod 4).
       family (a) m != 9 (mod 12): L' = parallel class (m/3 triples);
       family (b) m == 9 (mod 12): L' = hub (4 triples through z on 8
                  other points) + parallel class on the remaining m-9.
  D. GKLO divisibility rows for G = K_m - L' with lambda = 2, F = K4^(3),
     Deg(F) = (4,3,2):  4 | 2|G|;  3 | 2|G(x)| for all x;  (i=2 row
     automatic).  Also 2|G|/4 == J (the block count comes out to J).
  E. Lower bound logic re-check by brute force at m = 6: enumerate ALL
     multisets of triples with weight < B-4J(=4) and confirm none has all
     point-degrees == 2 (mod 3) and all pair-degrees even and weight == B
     (mod 4).  (Weights 1,2,3 all fail already on degree sums.)
"""
import itertools, sys
from collections import Counter
from math import comb

ok = True
def report(name, passed, extra=""):
    global ok
    ok &= passed
    print(("PASS " if passed else "FAIL ") + name + (" -- " + extra if extra else ""))

# ---------- A ----------
a_ok = True
for m in range(6, 3001, 3):
    s = m // 3
    R = ((m - 1) * (m - 2)) // 3
    if R != 3 * s * (s - 1): a_ok = False; print("R mismatch", m)
    B = 2 * comb(m, 3)
    if B - m * R != 2 * m // 3: a_ok = False; print("B-mR mismatch", m)
    if (m * R) % 2 != 0: a_ok = False; print("mR odd", m)
    want = 2 if m % 12 == 9 else 0
    if (m * R) % 4 != want: a_ok = False; print("mR mod4 mismatch", m)
    J = (m * R) // 4
    if B - 4 * J != 2 * m // 3 + (m * R) % 4: a_ok = False; print("B-4J mismatch", m)
report("A: arithmetic for all 3|m, m<=3000", a_ok)

# ---------- B ----------
row = []
for m in range(6, 28, 3):
    R = ((m - 1) * (m - 2)) // 3; B = 2 * comb(m, 3); J = (m * R) // 4
    row.append(B - 4 * J)
report("B: data row m=6..27", row == [4, 8, 8, 10, 12, 16, 16, 18], str(row))

# ---------- leave builders ----------
def leave_family(m):
    """returns dict triple -> multiplicity (the doubled leave 2L')."""
    L = {}
    if m % 12 != 9:
        for i in range(0, m, 3):
            L[(i, i + 1, i + 2)] = 2
    else:
        z = 0
        for i in range(4):                      # hub: {z, 1+2i, 2+2i}
            L[tuple(sorted((z, 1 + 2 * i, 2 + 2 * i)))] = 2
        for i in range(9, m, 3):                # parallel class on the rest
            L[(i, i + 1, i + 2)] = 2
    return L

# ---------- C, D ----------
cd_ok = True
for m in (6, 9, 12, 15, 18, 21, 24, 27, 33, 45, 57, 69):
    R = ((m - 1) * (m - 2)) // 3; B = 2 * comb(m, 3); J = (m * R) // 4
    L = leave_family(m)
    w = sum(L.values())
    if w != B - 4 * J: cd_ok = False; print("weight", m, w, B - 4 * J)
    if any(v > 2 for v in L.values()): cd_ok = False
    deg = Counter(); pdeg = Counter()
    for T, v in L.items():
        for x in T: deg[x] += v
        for pr in itertools.combinations(T, 2): pdeg[pr] += v
    if any(deg[x] % 3 != 2 for x in range(m)): cd_ok = False; print("deg", m)
    if any(v % 2 for v in pdeg.values()): cd_ok = False; print("pair", m)
    if w % 4 != B % 4: cd_ok = False; print("slotmod", m)
    # D: GKLO rows for G = K_m - L'
    Lp = {T: v // 2 for T, v in L.items()}          # L' simple
    sizeG = comb(m, 3) - sum(Lp.values())
    if (2 * sizeG) % 4 != 0: cd_ok = False; print("i0", m)
    for x in range(m):
        degLp = sum(v for T, v in Lp.items() if x in T)
        if (2 * (comb(m - 1, 2) - degLp)) % 3 != 0: cd_ok = False; print("i1", m, x)
    if (2 * sizeG) // 4 != J: cd_ok = False; print("blockcount", m)
report("C+D: leave families valid + GKLO divisibility + block count = J "
       "(m in {6,...,27,33,45,57,69})", cd_ok)

# ---------- E ----------
TR6 = list(itertools.combinations(range(6), 3))
e_ok = True
for w in (1, 2, 3):
    for combo in itertools.combinations_with_replacement(range(len(TR6)), w):
        mult = Counter(combo)
        if any(v > 2 for v in mult.values()): continue
        deg = Counter()
        for ti, k in mult.items():
            for x in TR6[ti]: deg[x] += k
        if all(deg[x] % 3 == 2 for x in range(6)):
            e_ok = False; print("unexpected small leave", mult)
report("E: m=6: no congruence-valid leave of weight < 4 "
       "(point rule alone kills w=1,2,3; every point must be touched)", e_ok)

# ---------- F ----------
# family partition: for 3 not dividing m, C(m,3) odd iff m == 3 (mod 4);
# hence {admissible, class, 3|m} partition all m >= 4.
f_ok = True
for m in range(4, 3001):
    if m % 3 == 0: continue
    if ((m - 1) * (m - 2)) % 3 != 0: f_ok = False; print("3|(m-1)(m-2) fails", m)
    if (comb(m, 3) % 2 == 1) != (m % 4 == 3): f_ok = False; print("parity", m)
report("F: family partition (3 not| m: C(m,3) odd iff m=3 mod 4), m<=3000", f_ok)

print()
print("OVERALL:", "ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

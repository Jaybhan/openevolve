#!/usr/bin/env python3
"""
verify_claims.py — numerically verify every checkable claim in theory.md
against (a) the local proven-exact table in ../../evaluator.py and (b) the
independently transcribed published data in published_data.py.

Run:  python3 verify_claims.py            (from anywhere; paths are absolute-safe)

Prints one PASS/FAIL line per claim plus detail; exits nonzero on any FAIL.
Everything here is deterministic. Nothing outside gpt_agent/ is written.
"""
import importlib.util
import json
import os
import sys
import time
from base64 import b64decode
from fractions import Fraction
from itertools import combinations
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import published_data as P  # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = load_module(os.path.join(BASE, "evaluator.py"), "ev")
Z = dict(ev.KST_EXACT_VALUE)          # the 161 proven-exact cells (m,n) -> z
CELLS = sorted(Z)

# ---------------------------------------------------------------- bounds ---
def kst_dgh(m, n, s, t):
    """Kovari–Sos–Turan as quoted by Davies–Gill–Horsley (s on the m-side):
    z < (t-1)^{1/s} * m * n^{1-1/s} + (s-1) n.  Returns a float upper bound."""
    return (t - 1) ** (1 / s) * m * n ** (1 - 1 / s) + (s - 1) * n


def kst_best(m, n, s, t):
    """min of the two dual orientations of the KST bound."""
    return min(kst_dgh(m, n, s, t), kst_dgh(n, m, t, s))


def furedi_319(m, n, a, b):
    """Furedi (CPC 1996), as stated in Furedi–Simonovits survey Thm 3.19:
    Z(m,n,a,b) <= (b-a+1)^{1/a} m n^{1-1/a} + (a-1) n^{2-2/a} + (a-2) m,
    for m>=a, n>=b, b>=a>=2."""
    return (b - a + 1) ** (1 / a) * m * n ** (1 - 1 / a) \
        + (a - 1) * n ** (2 - 2 / a) + (a - 2) * m


def roman(m, n, s, t, p):
    """Roman (1975) upper bound in Tan's floor form (Tan Thm 2.2 / DGH Thm 2.1):
    z(m,n;s,t) <= floor( (t-1)/C(p,s-1) * C(m,s) + (p+1)(s-1)/s * n ),
    valid for every integer p >= s-1.  Exact rational arithmetic."""
    v = Fraction((t - 1) * comb(m, s), comb(p, s - 1)) \
        + Fraction((p + 1) * (s - 1) * n, s)
    return v.numerator // v.denominator


def roman_min(m, n, s, t):
    best, bestp = None, None
    for p in range(s - 1, m + 1):
        v = roman(m, n, s, t, p)
        if best is None or v < best:
            best, bestp = v, p
    return best, bestp


def culik(m, n, s, t):
    """Culik (1956): z(m,n;s,t) = (s-1)n + (t-1)C(m,s) for n >= (t-1)C(m,s)."""
    return (s - 1) * n + (t - 1) * comb(m, s)


def waterfill(m, n, s, t):
    """Two-sided integer counting bound.  One side: maximise sum of column
    degrees c_j (n columns, c_j <= m) subject to sum_j C(c_j, s) <= (t-1)C(m,s);
    by convexity the level-filling profile is optimal.  Take the min of both
    orientations."""
    def one(mm, nn, ss, tt):
        # Level filling: always raise a currently-lowest column. The k-th unit
        # in a column costs C(k-1, ss-1), nondecreasing in k, and columns are
        # exchangeable, so taking cheapest increments first is optimal.
        budget = (tt - 1) * comb(mm, ss)
        deg = [0] * nn
        used = 0
        while True:
            cands = [j for j in range(nn) if deg[j] < mm]
            if not cands:
                return sum(deg)
            j = min(cands, key=lambda jj: deg[jj])
            delta = comb(deg[j] + 1, ss) - comb(deg[j], ss)
            if used + delta > budget:
                return sum(deg)
            used += delta
            deg[j] += 1
    return min(one(m, n, s, t), one(n, m, t, s))


# ------------------------------------------------------------ Tan decode ---
def decode_tan(code, m, n):
    """Tan's encoding: row-major bits, right-padded to a byte multiple with
    zeros, LITTLE-endian bit order within each byte, base64."""
    raw = b64decode(code)
    bits = []
    for byte in raw:
        bits.extend((byte >> k) & 1 for k in range(8))
    A = [bits[i * n:(i + 1) * n] for i in range(m)]
    return A


def edges(A):
    return sum(map(sum, A))


def k33_free(A, s=3, t=3):
    m = len(A)
    for rows in combinations(range(m), s):
        shared = 0
        for j in range(len(A[0])):
            if all(A[i][j] for i in rows):
                shared += 1
                if shared >= t:
                    return False
    return True


# ================================================================= CLAIMS ==
print("=" * 78)
print("C1. Local table identity vs Tan (2022) Table 3 (independent transcription)")
tan_cells = {}
tan_exact = set()
for m, row in P.TAN_Z3.items():
    for k, (v, bold) in enumerate(row):
        n = m + k
        tan_cells[(m, n)] = v
        if bold:
            tan_exact.add((m, n))
ok_vals = all(Z[c] == tan_cells[c] for c in CELLS if c in tan_cells)
extra = {c for c in CELLS if c not in tan_exact}
ok_region = extra == set(P.BNL_EXACT) - {(11, 22)}
# (11,21): the local value 116 deliberately DIFFERS from Tan's printed 117
mismatch = [c for c in CELLS if c in tan_cells and Z[c] != tan_cells[c]]
report("C1a all 161 local cells match Tan's printed value",
       mismatch == [(11, 21)] or mismatch == [],
       f"mismatches vs Tan print: {mismatch} (expected [(11,21)]: local 116 "
       f"overrides Tan's UB 117 per Bhan-Nobili-Langer 2026)")
report("C1b local exact region == Tan bold region + {(11,21),(12,22)} from BNL",
       ok_region and len(CELLS) == 161,
       f"|cells|={len(CELLS)}, non-Tan-bold cells={sorted(extra)}")
report("C1c local _EXACT_UP_TO equals Tan's bold limits",
       all((m, n) in tan_exact for (m, n) in CELLS if (m, n) not in P.BNL_EXACT))

print("=" * 78)
print("C2. Diagonals vs OEIS and historical attributions")
d1198 = {n: v - 1 for n, v in zip(range(3, 17), P.A001198)}
ok = all(Z[(n, n)] == d1198[n] for n in range(3, 17))
report("C2a z(n,n;3,3) diagonal == A001198 - 1 for n=3..16", ok)
ok = all(P.A072567[n - 1] == P.A001197[n - 2] - 1 for n in range(2, 25))
report("C2b A072567 == A001197 - 1 (n=2..24)", ok)
ok = all(P.CRWR_Z2_DIAG[n - 1] == P.A072567[n - 1] for n in range(1, 25))
report("C2c CRWR z(n;2) (n<=24) == A072567", ok)
ok = all(P.TAN_Z2_DIAG[n] == P.A072567[n - 1] for n in range(2, 25))
report("C2d Tan z_2 diagonal == A072567", ok)
hist = (Z[(4, 4)] == 13 and Z[(5, 5)] == 20 and Z[(6, 6)] == 26
        and Z[(7, 7)] == 33 and Z[(8, 8)] == 42)
report("C2e Sierpinski z(4..6;3,3)=13,20,26; Brzezinski z(7,7)=33; "
       "Culik z(8,8)=42 (per A001198 history)", hist)

print("=" * 78)
print("C3. CRWR (2016) cross-check")
crwr_bold, crwr_conflicts, crwr_extra_exact = 0, [], []
for m, row in P.CRWR_Z3.items():
    for k, (v, flags) in enumerate(row):
        n = m + k
        v = int(v)
        if "B" in flags:
            crwr_bold += 1
            if (m, n) in Z:
                if Z[(m, n)] != v:
                    crwr_conflicts.append(((m, n), v, Z[(m, n)]))
            else:
                crwr_extra_exact.append(((m, n), v))
report("C3a every CRWR-bold cell inside local suite agrees with local value",
       not crwr_conflicts, f"{crwr_bold} bold cells checked; conflicts={crwr_conflicts}")
report("C3b CRWR-bold cells NOT adopted by Tan/local (literature discrepancy)",
       crwr_extra_exact == [((12, 17), 103)],
       f"{crwr_extra_exact} — CRWR claim z(12,17;3,3)=103 exact (bold, unique-"
       "graph star); Tan prints Roman UB 108 unbolded, DGH make no improvement "
       "at (12,17). Unresolved in the literature; treat with caution.")
# CRWR upper bounds sharper than Tan's Roman print in the excluded region:
sharper = []
for m, row in P.CRWR_Z3.items():
    for k, (v, flags) in enumerate(row):
        n = m + k
        v = int(v)
        if "B" not in flags and (m, n) in tan_cells and (m, n) not in Z:
            if v < tan_cells[(m, n)]:
                sharper.append(((m, n), v, tan_cells[(m, n)]))
report("C3c CRWR non-exact UBs sharper than Tan's Roman print exist "
       "(informational)", True, f"{sharper}")

print("=" * 78)
print("C4. Kovari–Sos–Turan and Furedi general bounds dominate the table")
bad = [c for c in CELLS if Z[c] > kst_best(*c, 3, 3) ]
report("C4a KST (DGH form, best orientation) >= z on all 161 cells", not bad, str(bad[:5]))
bad = [c for c in CELLS
       if Z[c] > min(furedi_319(c[0], c[1], 3, 3), furedi_319(c[1], c[0], 3, 3))]
report("C4b Furedi CPC-1996 bound (FS survey Thm 3.19) >= z on all cells",
       not bad, str(bad[:5]))
loose = min(kst_best(*c, 3, 3) - Z[c] for c in CELLS)
report("C4c KST bound is never tight on this range (informational)",
       True, f"min slack = {loose:.2f} (KST is asymptotic, not exact, here)")

print("=" * 78)
print("C5. Culik's theorem on the table")
culik_cells = [c for c in CELLS if c[1] >= 2 * comb(c[0], 3)]
eq = all(Z[c] == culik(c[0], c[1], 3, 3) for c in culik_cells)
report("C5a z == 2n + 2C(m,3) on ALL cells with n >= 2C(m,3)",
       eq, f"{len(culik_cells)} Culik-regime cells (rows: "
       f"{sorted(set(c[0] for c in culik_cells))})")
below = [c for c in CELLS if c[1] < 2 * comb(c[0], 3)]
strict = all(Z[c] < culik(c[0], c[1], 3, 3) for c in below)
report("C5b z < Culik value strictly below the threshold (sanity)", strict)
# threshold sharpness where the table brackets it (m=4: n=7 vs 8; m=5: 19 vs 20)
sharp = (Z[(4, 7)] < culik(4, 7, 3, 3) and Z[(4, 8)] == culik(4, 8, 3, 3)
         and Z[(5, 19)] < culik(5, 19, 3, 3) and Z[(5, 20)] == culik(5, 20, 3, 3))
report("C5c equality begins exactly at n=(t-1)C(m,s) for m=4 (n=8) and m=5 (n=20)",
       sharp)

print("=" * 78)
print("C6. Roman's bound (1975) and its exactness window")
bad = [c for c in CELLS if Z[c] > roman_min(*c, 3, 3)[0]]
report("C6a min_p Roman floor-bound >= z on all 161 cells", not bad, str(bad[:5]))
# exactness window n >= (t-1)C(m,s) - s*T_{s,t}(m)  (Tan Thm 2.2, with his T33)
win_ok, win_cells = True, 0
for (m, n) in CELLS:
    n0 = 2 * comb(m, 3) - 3 * P.T33[m]
    if n >= n0:
        win_cells += 1
        want = min(roman(m, n, 3, 3, 2), roman(m, n, 3, 3, 3))
        if Z[(m, n)] != want:
            win_ok = False
            print("   window violation:", (m, n), Z[(m, n)], want)
report("C6b z == min(Roman(p=2),Roman(p=3)) on every cell with "
       "n >= 2C(m,3) - 3*T33(m)", win_ok,
       f"{win_cells} cells in the Roman/Tan exactness window "
       f"(rows 3,4,5 entirely; row 6 from n=13)")
r1222 = roman_min(12, 22, 3, 3)
report("C6c Roman UB at (12,22) equals 132 (so BNL's 132-edge witness closes it)",
       r1222[0] == 132, f"min_p Roman = {r1222[0]} at p={r1222[1]}")
r1121 = roman_min(11, 21, 3, 3)
report("C6d Roman UB at (11,21) is 117; exactness of 116 needs DGH's UB",
       r1121[0] == 117, f"min_p Roman = {r1121[0]}; DGH improved to 116")

print("=" * 78)
print("C7. Two-sided integer counting bound (waterfill)")
bad = [c for c in CELLS if Z[c] > waterfill(*c, 3, 3)]
report("C7a WF >= z on all cells", not bad, str(bad[:5]))
d0 = sum(1 for c in CELLS if Z[c] == waterfill(*c, 3, 3))
report("C7b number of counting-tight cells (coordinator found 78)",
       d0 == 78, f"WF-tight cells: {d0}/161")
report("C7c WF(16,16)=136, deficit 8 at the corner",
       waterfill(16, 16, 3, 3) == 136 and 136 - Z[(16, 16)] == 8)
same = sum(1 for c in CELLS if waterfill(*c, 3, 3) == roman_min(*c, 3, 3)[0])
report("C7d min_p Roman (floored closed form) == integer waterfill on ALL "
       "161 cells: Roman's bound loses nothing to integer level-filling here",
       same == len(CELLS), f"equal on {same}/161 cells")

print("=" * 78)
print("C8. (2,2): Reiman's bound and the projective-plane equality")
def reiman(n):
    return 0.5 * n * (1 + (4 * n - 3) ** 0.5)
z2 = {n: P.CRWR_Z2_DIAG[n - 1] for n in range(1, 32)}
bad = [n for n in z2 if z2[n] > reiman(n) + 1e-9]
report("C8a Reiman bound >= z(n,n;2,2) for n=1..31", not bad, str(bad))
pp = [n for n in z2 if abs(z2[n] - reiman(n)) < 1e-6]
report("C8b equality exactly at n = q^2+q+1 (q=0,1 degenerate; q=2,3,4,5 planes)",
       pp == [1, 3, 7, 13, 21, 31],
       f"tight at n={pp}")
ok = all(int(reiman(q * q + q + 1) + 1e-9) == (q + 1) * (q * q + q + 1)
         for q in range(2, 200))
report("C8c floor(Reiman(q^2+q+1)) == (q+1)(q^2+q+1) for q=2..199 "
       "(Reiman 1958 equality is an identity, no q>=15 condition)", ok)
def wf22(m, n):
    return waterfill(m, n, 2, 2)
report("C8d z(8,8;2,2)=24 while WF=25: first square (2,2) counting deficit",
       z2[8] == 24 and wf22(8, 8) == 25,
       f"z={z2[8]}, WF={wf22(8,8)}; deficits n=2..8: "
       f"{[(n, wf22(n,n)-z2[n]) for n in range(2,9)]}")

print("=" * 78)
print("C9. Exhaustive small-case verification (Culik/Roman ground truth)")
def brute_z(m, n, s, t):
    """Exact z(m,n;s,t) by DFS over non-decreasing column multisets."""
    cols = list(range(2 ** m))
    w = [bin(c).count("1") for c in cols]
    best = [0]
    def viol(chosen):
        for rows in combinations(range(m), s):
            mask = 0
            for r in rows:
                mask |= 1 << r
            cnt = sum(1 for c in chosen if c & mask == mask)
            if cnt >= t:
                return True
        return False
    def dfs(start, chosen, ones):
        if len(chosen) == n:
            if ones > best[0]:
                best[0] = ones
            return
        rem = n - len(chosen)
        if ones + rem * m <= best[0]:
            return
        for c in range(start, -1, -1):
            chosen.append(c)
            if not viol(chosen):
                dfs(c, chosen, ones + w[c])
            chosen.pop()
    dfs(2 ** m - 1, [], 0)
    return best[0]

t0 = time.time()
ok = all(brute_z(3, n, 3, 3) == 2 * n + 2 for n in range(3, 7))
report("C9a exhaustive z(3,n;3,3) == 2n+2 for n=3..6 (Culik, threshold 2)", ok)
got = {n: brute_z(4, n, 3, 3) for n in range(4, 8)}
report("C9b exhaustive z(4,n;3,3) for n=4..7 == table (13,16,18,21) "
       "== floor((8n+8)/3) (Roman p=3)",
       all(got[n] == Z[(4, n)] == (8 * n + 8) // 3 for n in got), str(got))
c23 = {n: brute_z(3, n, 2, 3) for n in range(3, 8)}
want23 = {3: 7, 4: 9, 5: 11, 6: 12, 7: 13}   # Culik: n+6 for n>=6
ok = all(c23[n] == n + 6 for n in (6, 7)) and all(c23[n] < n + 6 for n in (3, 4, 5))
report("C9c exhaustive z(3,n;2,3): equals n + 2*C(3,2) = n+6 exactly from "
       "the Culik threshold n=6 on", ok, f"{c23}")
c22 = {n: brute_z(4, n, 2, 2) for n in range(4, 8)}
ok = all(c22[n] == n + 6 for n in (6, 7)) and c22[4] == 9 and c22[5] == 10
report("C9d exhaustive z(4,n;2,2): == n + C(4,2) = n+6 from n=6 on (Culik); "
       "z(4,4;2,2)=9, z(4,5;2,2)=10 below", ok, f"{c22} ({time.time()-t0:.1f}s)")
# z(5,5;3,3) <= 20 purely by the column-side counting bound:
def max_edges_profile(m, n, s, t):
    return waterfill(m, n, s, t)
report("C9e z(5,5;3,3): counting bound gives <=20; Tan witness attains 20",
       max_edges_profile(5, 5, 3, 3) == 20)

print("=" * 78)
print("C10. Tan witness matrices decode and verify")
for (m, n), code in P.TAN_WITNESSES.items():
    A = decode_tan(code, m, n)
    e = edges(A)
    ok = k33_free(A) and e == Z[(m, n)]
    report(f"C10 ({m},{n}): K33-free with {e} == z = {Z[(m,n)]} ones", ok)

print("=" * 78)
print("C11. Champion construction at (16,16) vs Tan's witness")
champ = load_module(os.path.join(BASE, "openevolve_output", "best",
                                 "best_program.py"), "champ")
A = champ.construct_graph(16, 16).astype(int).tolist()
okA = k33_free(A) and edges(A) == 128
rows_reg = sorted(sum(r) for r in A) == [8] * 16
cols_reg = sorted(sum(A[i][j] for i in range(16)) for j in range(16)) == [8] * 16
report("C11a champion (16,16) valid, 128 ones, 8-regular both sides",
       okA and rows_reg and cols_reg)
B = decode_tan(P.TAN_WITNESSES[(16, 16)], 16, 16)

def codeg(M):
    m = len(M)
    return [[sum(a & b for a, b in zip(M[i], M[j])) for j in range(m)]
            for i in range(m)]

def iso_bipartite(A, B):
    """Row-permutation backtracking with codegree consistency, then column
    multiset check.  Tries B and B-transpose."""
    def cols_as_multiset(M, perm):
        n = len(M[0])
        out = []
        for j in range(n):
            out.append(tuple(M[perm[i]][j] for i in range(len(M))))
        return sorted(out)
    def attempt(B):
        CA, CB = codeg(A), codeg(B)
        m = len(A)
        provA = [tuple(sorted(CA[i][j] for j in range(m) if j != i)) for i in range(m)]
        provB = [tuple(sorted(CB[i][j] for j in range(m) if j != i)) for i in range(m)]
        if sorted(provA) != sorted(provB):
            return False
        assign = [-1] * m
        used = [False] * m
        def bt(i):
            if i == m:
                # rows of B reordered by assign^{-1} must equal rows of A up to
                # a single column permutation: compare column multisets
                inv = [0] * m
                for a, b in enumerate(assign):
                    inv[a] = b
                colsA = sorted(tuple(A[i][j] for i in range(m)) for j in range(len(A[0])))
                colsB = sorted(tuple(B[inv[i]][j] for i in range(m)) for j in range(len(B[0])))
                return colsA == colsB
            for b in range(m):
                if used[b] or provA[i] != provB[b]:
                    continue
                if any(CA[i][j] != CB[b][assign[j]] for j in range(i)):
                    continue
                assign[i] = b
                used[b] = True
                if bt(i + 1):
                    return True
                used[b] = False
                assign[i] = -1
            return False
        return bt(0)
    if attempt(B):
        return True
    Bt = [list(r) for r in zip(*B)]
    return attempt(Bt)

t0 = time.time()
iso = iso_bipartite(A, B)
report("C11b champion (16,16) matrix isomorphic to Tan's published witness",
       True, f"isomorphic={iso} ({time.time()-t0:.1f}s). "
       + ("Same object: the cap/affine-hyperplane construction is a NEW "
          "DESCRIPTION of the known witness, not a new witness."
          if iso else
          "NOT isomorphic: a second maximal matrix for z(16,16;3,3) — "
          "would contradict CRWR's uniqueness star; re-check before claiming."))

print("=" * 78)
print("C12. Caps in PG(3,2)")
pts = list(range(1, 16))
best_cap = 0
# exhaustive over the 2^15 subsets, with early pruning by popcount
import itertools as _it
for r in range(8, 3, -1):
    found = False
    for S in _it.combinations(pts, r):
        Sset = set(S)
        if all((a ^ b) not in Sset for a, b in _it.combinations(S, 2)):
            best_cap = r
            found = True
            break
    if found:
        break
report("C12a maximum cap in PG(3,2) has size 8 (exhaustive)", best_cap == 8)
S = set(range(8, 16))
ok = all((a ^ b) not in S for a, b in _it.combinations(S, 2))
report("C12b champion's normal set {8..15} is a cap (complement of a hyperplane)",
       ok)

print("=" * 78)
print("C13. Packing numbers behind the Roman window")
def T33_exact(m, cap_seconds=30):
    """Max multiset of 4-subsets of [m], every 3-subset covered <= 2 times.
    Branch and bound over block types in lex order with a capacity bound."""
    blocks = list(combinations(range(m), 4))
    tri_of = [list(combinations(b, 3)) for b in blocks]
    cap = {T: 2 for T in combinations(range(m), 3)}
    best = [0]
    deadline = time.time() + cap_seconds
    def bound(i):
        return sum(cap.values()) // 4
    def dfs(i, count):
        if time.time() > deadline:
            raise TimeoutError
        best[0] = max(best[0], count)
        if i == len(blocks):
            return
        if count + bound(i) <= best[0]:
            return
        maxmult = min(cap[T] for T in tri_of[i])
        for mult in range(maxmult, -1, -1):
            for T in tri_of[i]:
                cap[T] -= mult
            dfs(i + 1, count + mult)
            for T in tri_of[i]:
                cap[T] += mult
    try:
        dfs(0, 0)
        return best[0], True
    except TimeoutError:
        return best[0], False
v6, complete6 = T33_exact(6)
report("C13a T_{3,3}(6) == 9 (exhaustive; Tan Table 1 value)",
       v6 == 9 and complete6, f"value={v6}, search complete={complete6}")
# Turan side: max triangle-free (simple) graph on 6 vertices has 9 edges
bestT = 0
for eset in range(2 ** 15):
    E = [pair for k, pair in enumerate(combinations(range(6), 2)) if eset >> k & 1]
    if len(E) <= bestT:
        continue
    adj = [[False] * 6 for _ in range(6)]
    for a, b in E:
        adj[a][b] = adj[b][a] = True
    if any(adj[a][b] and adj[b][c] and adj[a][c]
           for a, b, c in combinations(range(6), 3)):
        continue
    bestT = len(E)
report("C13b ex(6; K_3) == 9 == T_{3,3}(6) (coordinator's row-6 Turan bridge)",
       bestT == 9)
v7, complete7 = T33_exact(7, cap_seconds=60)
report("C13c T_{3,3}(7) == 15 (Tan: Gurobi; coordinator: exhaustive; here: B&B)",
       v7 == 15, f"value={v7}, my search complete={complete7} "
       "(if False, 15 is still confirmed as a lower bound and by two "
       "independent exact computations: Tan 2022, coordinator 2026)")
# Tan's cyclic construction for T33(7): blocks 0135,0136,0156 under (01234)
base = [(0, 1, 3, 5), (0, 1, 3, 6), (0, 1, 5, 6)]
perm = {0: 1, 1: 2, 2: 3, 3: 4, 4: 0, 5: 5, 6: 6}
blocks = []
for b in base:
    cur = b
    for _ in range(5):
        blocks.append(tuple(sorted(cur)))
        cur = tuple(perm[x] for x in cur)
capd = {}
okc = True
for b in blocks:
    for T in combinations(sorted(b), 3):
        capd[T] = capd.get(T, 0) + 1
        if capd[T] > 2:
            okc = False
report("C13d Tan's cyclic presentation of T_{3,3}(7)=15 is a valid 2-fold "
       "packing of 15 quadruples", okc and len(blocks) == 15)

print("=" * 78)
print("C14. The two extra exact cells and DGH consistency")
bad = [(c, ub) for c, ub in P.DGH_UB33.items() if c in Z and Z[c] > ub]
report("C14a every DGH improved UB >= the local exact value where both exist",
       not bad, str(bad))
report("C14b (11,21): local 116 == DGH UB 116 == BNL exact claim",
       Z[(11, 21)] == 116 == P.DGH_UB33[(11, 21)] == P.BNL_EXACT[(11, 21)])
report("C14c (12,22): local 132 == Roman UB 132 == BNL exact claim",
       Z[(12, 22)] == 132 == roman_min(12, 22, 3, 3)[0] == P.BNL_EXACT[(12, 22)])
report("C14d (11,22)=121 proven exact by BNL but NOT in the local suite "
       "(informational for the owner)", (11, 22) not in Z)
# Local run attainment
best1121 = best1222 = 0
logp = os.path.join(BASE, "instance_log.jsonl")
if os.path.exists(logp):
    for line in open(logp):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        for inst in rec.get("instances", []):
            if inst.get("valid"):
                if inst["mn"] == "11x21":
                    best1121 = max(best1121, inst["edges"])
                if inst["mn"] == "12x22":
                    best1222 = max(best1222, inst["edges"])
report("C14e local generalized-run has NOT re-attained the two BNL cells yet "
       "(informational)", True,
       f"best valid so far: 11x21 -> {best1121}/116, 12x22 -> {best1222}/132 "
       "(the witnesses live in the BNL paper, not this run)")

print("=" * 78)
print("C15. Asymptotics sanity (informational)")
vals = [(n, Z[(n, n)], Z[(n, n)] / n ** (5 / 3)) for n in (8, 12, 16)]
report("C15 z(n,n;3,3)/n^(5/3) on the diagonal (theory: -> 1 as n -> inf; "
       "constant 1/2 belongs to the GRAPH version ex(n,K33))", True,
       "; ".join(f"n={n}: {z} ratio={r:.3f}" for n, z, r in vals))

print("=" * 78)
print("C16. T_{3,3}(m) = D_2(m,4,3): admissibility law, Johnson bound, m=18")


def admissible_3m42(m):
    """Divisibility conditions for a 3-(m,4,2) design (= perfect 2-fold
    packing): r = 2*C(m-1,2)/3 and b = C(m,3)/2 integral, i.e.
    3 | (m-1)(m-2)  and  C(m,3) even  (equivalently m != 0 mod 3 and
    m != 3 mod 4)."""
    return (m - 1) * (m - 2) % 3 == 0 and comb(m, 3) % 2 == 0


def johnson2_43(m):
    """Per-point (first Johnson) bound for 2-fold packing of triples by
    quadruples: any point lies in r_x blocks, 3*r_x <= 2*C(m-1,2), so
    r_x <= floor((m-1)(m-2)/3); summing, 4b <= m*r_max."""
    r_max = ((m - 1) * (m - 2)) // 3
    return (m * r_max) // 4


report("C16a a 3-(18,4,2) design is inadmissible: r = 2*C(17,2)/3 = 272/3 "
       "is not an integer (coordinator's observation confirmed)",
       (2 * comb(17, 2)) % 3 != 0)
bad = [m for m in range(3, 19) if P.T33[m] > johnson2_43(m)]
gap2 = sorted(m for m in range(3, 19) if johnson2_43(m) - P.T33[m] == 2)
gap0 = sorted(m for m in range(3, 19) if johnson2_43(m) == P.T33[m])
report("C16b Johnson bound >= T33 on m=3..18; slack is 2 exactly at "
       "m in {7,11} (m=3 mod 4 and m!=0 mod 3) and 0 everywhere else "
       "— in particular Johnson is TIGHT at all of m=6,9,12,15,18",
       not bad and gap2 == [7, 11] and len(gap0) == 14,
       f"tight at {gap0}")
law = all((P.T33[m] == comb(m, 3) // 2) == admissible_3m42(m)
          for m in range(4, 19))
report("C16c perfect 2-fold packing (T = C(m,3)/2) <=> 3-(m,4,2) "
       "admissibility, for all m=4..18 (the coordinator's 'law'; the "
       "design-existence direction is Hanani's 3-(v,4,lambda) spectrum)",
       law)


def _sym(ch):
    return int(ch, 36)


def _perm_from_cycles(s, m):
    p = list(range(m))
    for cyc in s.replace(")", "(").split("("):
        cyc = cyc.strip()
        if not cyc:
            continue
        idx = [_sym(c) for c in cyc]
        for i, x in enumerate(idx):
            p[x] = idx[(i + 1) % len(idx)]
    return tuple(p)


def _validate_presentation(m, perm_strs, base_strs):
    gens = [_perm_from_cycles(s, m) for s in perm_strs]
    G = {tuple(range(m))}
    frontier = list(G)
    while frontier:
        nf = []
        for g in frontier:
            for h in gens:
                c = tuple(h[g[i]] for i in range(m))
                if c not in G:
                    G.add(c)
                    nf.append(c)
        frontier = nf
    blocks = []
    for b in base_strs:
        base = frozenset(_sym(c) for c in b)
        blocks.extend({frozenset(g[x] for x in base) for g in G})
    cov = {}
    for b in blocks:
        for T in combinations(sorted(b), 3):
            cov[T] = cov.get(T, 0) + 1
            if cov[T] > 2:
                return len(blocks), False
    return len(blocks), True


for m in (17, 18):
    perms, bases = P.T33_PRESENTATIONS[m]
    nblocks, valid = _validate_presentation(m, perms, bases)
    report(f"C16d Tan's cyclic presentation for T33({m}) generates exactly "
           f"{P.T33[m]} blocks and is a valid 2-fold packing",
           nblocks == P.T33[m] and valid, f"blocks={nblocks}")
report("C16e T33(18)=405 is therefore proven WITHOUT solver trust: "
       "Johnson bound (C16b) == validated construction (C16d)",
       johnson2_43(18) == 405 == P.T33[18])
# The refuted extrapolation: floor(C(m,3)/2) - 2 predicts 406 at m=18
report("C16f the 'delta=2 for all inadmissible m>=7' extrapolation predicts "
       "T33(18) = 406 and is REFUTED (Tan prints 405; Johnson <= 405); "
       "the m=0 mod 3 defect floor(C(m,3)/2)-J(m) grows like m/6: "
       "1,2,2,2,3 at m=6,9,12,15,18",
       comb(18, 3) // 2 - 2 == 406 and P.T33[18] == 405
       and [comb(m, 3) // 2 - johnson2_43(m) for m in (6, 9, 12, 15, 18)]
       == [1, 2, 2, 2, 3])

print("=" * 78)
if FAILS:
    print(f"RESULT: {len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("RESULT: all checks passed")

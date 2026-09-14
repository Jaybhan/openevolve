"""Machine verification for analysis/theorems.md. Run: python3 verify_theorems.py"""
import importlib.util
import os
from itertools import combinations, combinations_with_replacement
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "ev", os.path.join(HERE, "..", "..", "evaluator.py")
)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
Z = dict(ev.KST_EXACT_VALUE)


def check_free(blocks, m):
    """Every triple of rows covered by <= 2 blocks?"""
    cap = {}
    for b in blocks:
        for t in combinations(sorted(b), 3):
            cap[t] = cap.get(t, 0) + 1
            if cap[t] > 2:
                return False
    return True


def wf_side(n_cols, budget, cap):
    E = 2 * n_cols
    h = 2
    while h < cap:
        cost = comb(h, 2)
        k = min(n_cols, budget // cost)
        if k == 0:
            break
        if k < n_cols:
            E += k
            break
        E += n_cols
        budget -= n_cols * cost
        h += 1
    return E


def WF(m, n):
    return min(wf_side(n, 2 * comb(m, 3), m), wf_side(m, 2 * comb(n, 3), n))


fails = []

# --- Theorem 1: closed forms == WF == table on all m<=5 cells ---
def thm1(m, n):
    if m == 3:
        return 2 * n + 2
    if m == 4:
        return 3 * n + (8 - n) // 3 if n <= 8 else 2 * n + 8
    if m == 5:
        return 3 * n + min(5, (20 - n) // 3) if n <= 20 else 2 * n + 20


t1_cells = [(m, n) for (m, n) in Z if m <= 5]
for (m, n) in t1_cells:
    if not (thm1(m, n) == WF(m, n) == Z[(m, n)]):
        fails.append(("Thm1", m, n, thm1(m, n), WF(m, n), Z[(m, n)]))
print(f"Thm1: {len(t1_cells)} cells m<=5: closed form == WF == table:",
      "OK" if not any(f[0] == "Thm1" for f in fails) else "FAIL")

# --- Theorem 2(a): (6,9)=36 via K_{3,3} edge-complements ---
k33_edges = [(i, j) for i in range(3) for j in range(3, 6)]
blocks_69 = [tuple(x for x in range(6) if x not in e) for e in k33_edges]
ok_a = check_free(blocks_69, 6) and sum(len(b) for b in blocks_69) == 36 == Z[(6, 9)]
print("Thm2a: (6,9) K33-complement witness, 36 edges, free:", "OK" if ok_a else "FAIL")
if not ok_a:
    fails.append(("Thm2a",))

# --- (6,6)=26 via two point-complements + C4 edge-complements ---
omit = [(0,), (1,), (2, 3), (3, 4), (4, 5), (5, 2)]
blocks_66 = [tuple(x for x in range(6) if x not in e) for e in omit]
ok_66 = check_free(blocks_66, 6) and sum(len(b) for b in blocks_66) == 26 == Z[(6, 6)]
print("Thm2 aux: (6,6) witness 26 edges, free:", "OK" if ok_66 else "FAIL")
if not ok_66:
    fails.append(("Thm2-66",))

# --- ex(6, K3) = 9, and 10-edge graphs all have triangles ---
all_pairs = list(combinations(range(6), 2))
ex6 = 0
for k in (9, 10):
    found_tf = False
    for es in combinations(all_pairs, k):
        adj = [[False] * 6 for _ in range(6)]
        for a, b in es:
            adj[a][b] = adj[b][a] = True
        if not any(adj[a][b] and adj[b][c] and adj[a][c]
                   for a, b, c in combinations(range(6), 3)):
            found_tf = True
            break
    if k == 9 and not found_tf:
        fails.append(("ex6-9",))
    if k == 10 and found_tf:
        fails.append(("ex6-10",))
print("Turán check: triangle-free with 9 edges exists, with 10 impossible:",
      "OK" if not any(f[0].startswith("ex6") for f in fails) else "FAIL")

# --- Profile exhaustion: only stated profiles reach E within budget 40 ---
def profiles(n, E, budget=40, hmax=6):
    out = []
    for hs in combinations_with_replacement(range(2, hmax + 1), n):
        if sum(hs) == E and sum(comb(h, 3) for h in hs) <= budget:
            out.append(tuple(sorted(hs, reverse=True)))
    return sorted(set(out))

p67 = profiles(7, 30)
p68 = profiles(8, 33)
p610 = profiles(10, 40)
ok_p = (p67 == [(5, 5, 4, 4, 4, 4, 4)] and p68 == [(5, 4, 4, 4, 4, 4, 4, 4)]
        and p610 == [(4,) * 10])
print("Profile exhaustion (6,7)E30 / (6,8)E33 / (6,10)E40:", "OK" if ok_p else
      f"FAIL {p67} {p68} {p610}")
if not ok_p:
    fails.append(("profiles",))

# --- Brute-force infeasibility of (5,5,4^5) on 6 rows [Thm 2c] ---
pairs = all_pairs
infeasible_c = True
for xy in combinations_with_replacement(range(6), 2):
    five_blocks = [tuple(r for r in range(6) if r != x) for x in xy]
    for quads in combinations_with_replacement(pairs, 5):
        four_blocks = [tuple(r for r in range(6) if r not in p) for p in quads]
        if check_free(five_blocks + list(four_blocks), 6):
            infeasible_c = False
            break
    if not infeasible_c:
        break
print("Thm2c: profile (5,5,4^5) on 6 rows infeasible (brute force):",
      "OK" if infeasible_c else "FAIL")
if not infeasible_c:
    fails.append(("Thm2c",))

# --- Brute-force infeasibility of (5,4^7) on 6 rows [Thm 2d] ---
infeasible_d = True
for x in range(6):
    five = tuple(r for r in range(6) if r != x)
    for quads in combinations_with_replacement(pairs, 7):
        four_blocks = [tuple(r for r in range(6) if r not in p) for p in quads]
        if check_free([five] + list(four_blocks), 6):
            infeasible_d = False
            break
    if not infeasible_d:
        break
print("Thm2d: profile (5,4^7) on 6 rows infeasible (brute force):",
      "OK" if infeasible_d else "FAIL")
if not infeasible_d:
    fails.append(("Thm2d",))

print()
print("ALL CHECKS PASS" if not fails else f"FAILURES: {fails}")

"""Master-formula validator: compute z(m,n;3,3) from the supply tables alone
(pure arithmetic, no per-cell search), and compare against ground truth.

  z(m,n) = 2n + max over (k6,k5) in the S_m grid, k4 <= S_m(k5,k6):
             U = 4*k6 + 3*k5 + 2*k4 + k3,
           subject to cols = k6+k5+k4 <= n,
                      slots = 20*k6 + 10*k5 + 4*k4 <= B = 2*C(m,3),
                      k3 = min(n - cols, B - slots)   [triple fill],
           (pads fill the rest; heavier blocks handled only via the grid's
            k6 axis — validator applies where optima use weights <= 6,
            i.e. n >= stated n1(m)).

Ground truth: published table (rows 6-8, n <= 23) + our ILP gap-band values.
Supply tables: computed slices; S_8(3) unresolved -> checked for BOTH
endpoints of its known interval [15, 21] and reported if the answer differs.
"""
import importlib.util
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "ev", os.path.join(HERE, "..", "..", "evaluator.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
Z = dict(ev.KST_EXACT_VALUE)

# our new ILP-determined cells
Z.update({(7, 24): 87, (8, 24): 97, (8, 25): 100, (8, 26): 104, (8, 27): 108,
          (9, 23): 103})

# supply tables S_m[(k5, k6)] = max k4 (None = infeasible)
S6 = {(0, 0): 9, (1, 0): 6, (2, 0): 4}
S7 = {(0, 0): 15, (1, 0): 12, (2, 0): 10, (3, 0): 8, (4, 0): 6}
S8lo = {(0, 0): 28, (1, 0): 23, (2, 0): 21, (3, 0): 15, (4, 0): 15,
        (5, 0): 14, (6, 0): 10, (7, 0): 8, (8, 0): 6}
S8hi = dict(S8lo); S8hi[(3, 0)] = 21   # S_8(3) in [15,21], both endpoints

N1 = {6: 6, 7: 10, 8: 10}  # validator applicability (weights<=6 optima)


def z_formula(m, n, S):
    B = 2 * comb(m, 3)
    best = 0
    for (k5, k6), k4max in S.items():
        if k4max is None:
            continue
        for k4 in range(k4max + 1):
            cols = k4 + k5 + k6
            if cols > n:
                continue
            slots = 4 * k4 + 10 * k5 + 20 * k6
            if slots > B:
                continue
            k3 = min(n - cols, B - slots)
            U = 4 * k6 + 3 * k5 + 2 * k4 + k3
            best = max(best, 2 * n + U)
    return best


for m, S, tag in ((6, S6, ""), (7, S7, ""), (8, S8lo, " [S8(3)=15]"),
                  (8, S8hi, " [S8(3)=21]")):
    ns = sorted(n for (mm, n) in Z if mm == m and n >= N1[m])
    bad = []
    for n in ns:
        zf = z_formula(m, n, S)
        if zf != Z[(m, n)]:
            bad.append((n, zf, Z[(m, n)]))
    status = "ALL MATCH" if not bad else f"MISMATCH {bad}"
    print(f"row {m}{tag}: {len(ns)} cells n>={N1[m]}: {status}")

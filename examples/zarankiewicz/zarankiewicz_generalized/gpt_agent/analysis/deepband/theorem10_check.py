"""Machine verification of the PROVEN parts of Theorem 10 (deep-band bound).

(0) The block identity  6(w-3) = 2 + w(w-3) - (w-4)(w-5), and
    g(w) := (w-4)(w-5) + C(w,3) - w(w-3) >= 0 with equality iff w in {4,5},
    for all 4 <= w <= 60.
(1) The per-point inequality 3*sum_{B>x}(w-3) <= sum_{B>x} C(w-1,2) (block-
    wise: 3(w-3) <= C(w-1,2), tight iff w in {4,5}), for all 4 <= w <= 60.
(2) On every stored witness: W(C) = sum_x y_x <= m*R, y_x <= R pointwise,
    and val <= floor((2#C + mR - X)/6) <= floor((2#C + mR)/6).
(3) The z-form upper bound  z(m,n) <= 3n + min(J*, floor((2n+mR)/6))
    against every known z of rows 6-9: NEVER violated, and the table of
    cells where it is TIGHT (bound == truth) — those cells' upper bounds
    are now closed by pure counting (no ILP, no profile elimination).
"""
import json
import os
from itertools import combinations
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))

# (0)+(1): block-level inequalities
for w in range(4, 61):
    assert 6 * (w - 3) == 2 + w * (w - 3) - (w - 4) * (w - 5), w
    g = (w - 4) * (w - 5) + comb(w, 3) - w * (w - 3)
    assert g >= 0 and ((g == 0) == (w in (4, 5))), (w, g)
    assert 3 * (w - 3) <= comb(w - 1, 2), w
    assert (3 * (w - 3) == comb(w - 1, 2)) == (w in (4, 5)), w
print("(0)(1) block identities/inequalities: PASS (w = 4..60)")

# (2): witness-level checks
WDIR = os.path.join(BASE, "analysis", "witnesses")
n_checked = 0
for fn in sorted(os.listdir(WDIR)):
    if not fn.endswith(".json"):
        continue
    with open(os.path.join(WDIR, fn)) as f:
        wjs = json.load(f)
    m = wjs["m"]
    R = (2 * comb(m - 1, 2)) // 3
    heavy = [b for b in wjs["blocks"] if len(b) >= 4]
    c = len(heavy)
    val = sum(len(b) - 3 for b in heavy)
    W = sum(len(b) * (len(b) - 3) for b in heavy)
    X = sum((len(b) - 4) * (len(b) - 5) for b in heavy)
    y = {x: 0 for x in range(m)}
    for b in heavy:
        for x in b:
            y[x] += len(b) - 3
    assert sum(y.values()) == W
    assert all(v <= R for v in y.values()), (fn, y, R)
    assert W <= m * R, fn
    assert 6 * val == 2 * c + W - X, fn
    assert val <= (2 * c + m * R - X) // 6 <= (2 * c + m * R) // 6, fn
    n_checked += 1
print(f"(2) witness checks (y_x <= R, W <= mR, identity, bound): "
      f"PASS on {n_checked} witnesses")

# (3): the z-form bound vs all known truth, rows 6-9
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ev", os.path.join(BASE, "..", "evaluator.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
truth = {k: v for k, v in ev.KST_EXACT_VALUE.items() if 6 <= k[0] <= 9}
for fn in ("ilp_gapband.jsonl", "ilp_results.jsonl"):
    p = os.path.join(BASE, "analysis", fn)
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "z_ilp" in r and r.get("witness_valid") and 6 <= r["m"] <= 9:
                    truth.setdefault((r["m"], r["n"]), r["z_ilp"])
# Roman-window tail (Theorem 8, proven) for completeness of the rows
for m in (6, 7, 8, 9):
    B = 2 * comb(m, 3)
    T = {6: 9, 7: 15, 8: 28, 9: 40}[m]
    for n in range(m, B + 1):
        truth.setdefault((m, n), 3 * n + min(T, (B - n) // 3)) if n >= T else None

viol = 0
tight = {6: [], 7: [], 8: [], 9: []}
for (m, n), zt in sorted(truth.items()):
    R = (2 * comb(m - 1, 2)) // 3
    J = (m * R) // 4 - (2 if (m % 4 == 3 and m % 3 != 0) else 0)
    ub = 3 * n + min(J, (2 * n + m * R) // 6)
    if zt > ub:
        print(f"  VIOLATION at ({m},{n}): truth {zt} > bound {ub}")
        viol += 1
    if zt == ub:
        tight[m].append(n)
print(f"(3) z-form bound never violated on {len(truth)} known cells "
      f"(rows 6-9): {'PASS' if viol == 0 else 'FAIL'}")
for m in (6, 7, 8, 9):
    print(f"    m={m}: TIGHT (upper bound = truth) at n = {tight[m]}")

# (4): row-level closed forms implied by Theorem 10 + Theorem 8 + frontiers
print()
ok6 = True
for n in range(6, 41):
    if (6, n) not in truth:
        continue
    f = 3 * n + min(9, (n + 18) // 3, (40 - n) // 3)
    if f != truth[(6, n)]:
        ok6 = False
        print(f"  row-6 closed form FAILS at n={n}: {f} vs {truth[(6,n)]}")
print(f"(4a) z(6,n) = 3n + min(9, floor((n+18)/3), floor((40-n)/3)) "
      f"for 6<=n<=40: {'PASS (all known cells)' if ok6 else 'FAIL'}")
exc = []
for n in range(20, 169):
    if (9, n) not in truth:
        continue
    f = 3 * n + min(40, (n + 81) // 3, (168 - n) // 3)
    if f != truth[(9, n)]:
        exc.append((n, truth[(9, n)] - f))
print(f"(4b) z(9,n) vs 3n + min(40, floor((n+81)/3), floor((168-n)/3)), "
      f"n>=20: exceptions {exc}")
print("      (expected: -1 exactly at n in {21, 24, 25, 26, 27, 33, 39}"
      " — the parity-delayed cells)")

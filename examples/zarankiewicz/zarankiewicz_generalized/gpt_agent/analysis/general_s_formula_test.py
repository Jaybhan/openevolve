"""General-s Level Formula (closed form, Johnson-capped) vs exact_small ground
truth. Result of record: (3,4) 21/21 EXACT, (4,4) 21/21 EXACT, s=2 excluded
(Lemma C fails structurally), (3,3) deviations = known supply slack.
Rerun: python3 general_s_formula_test.py"""
from math import comb
import csv, os
from collections import Counter

def zhat(m, n, s, t):
    if m < s or n < t: return m * n
    B = (t - 1) * comb(m, s)
    best = 0
    for w in range(s - 1, m + 1):
        cw1 = comb(w - 1, s) if w - 1 >= s else 0
        bud = B - cw1 * n
        if bud < 0: continue
        den = comb(w - 1, s - 1) if w - 1 >= s - 1 else 1
        k_budget = bud // den if den else n
        if w >= s:
            jd = comb(w - 1, s - 1)
            Jw = (m * (((t - 1) * comb(m - 1, s - 1)) // jd)) // w if jd else n
        else:
            Jw = n
        v = (w - 1) * n + min(n, Jw, k_budget)
        best = max(best, v)
    return best

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "bounds", "exact_small.csv")
    res = {}
    for r in csv.DictReader(open(path)):
        s, t, m, n, z = (int(r[k]) for k in ("s", "t", "m", "n", "z"))
        if s == 2: continue
        res.setdefault((s, t), []).append(zhat(m, n, s, t) - z)
    for st, devs in sorted(res.items()):
        print(f"(s,t)={st}: {len(devs)} cells  dev-hist={dict(sorted(Counter(devs).items()))}")

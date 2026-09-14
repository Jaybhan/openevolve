"""Exact z(m,n;3,3) for small m: per-(m,n) branch & bound, complete search.

Search space: multisets of blocks (column supports) of weight >= 3, capacities
cap[T]=2 per row-triple; weight-2 pads appended in closed form (they consume
no capacity). Canonical non-increasing type order kills permutation symmetry.

Soundness of the pruning bound: for r remaining columns and residual capacity
vector cap, the best achievable future edge count is at most
waterfill(r, sum(cap)) — the LP/convexity optimum over ANY r columns under the
global residual budget (per-triple structure only tightens it). Waterfill is
exactly optimal for the relaxation, so the prune never cuts a true optimum.

v2 fixes v1's aggregation bug (qmax collapsed across families sharing a
summary but not a capacity vector).
"""
import sys
from itertools import combinations
from math import comb

sys.setrecursionlimit(100000)


def wf_future(r, budget, hmax):
    """Max edges from r columns under total capacity `budget`, heights<=hmax."""
    if r <= 0:
        return 0
    E = 2 * r
    h = 2
    while h < hmax:
        cost = comb(h, 2)  # raise one column h -> h+1 costs C(h,3+... ) wait
        # capacity cost of a column of height H is C(H,3); marginal h->h+1 is
        # C(h,2). Correct.
        k = min(r, budget // cost) if cost else r
        if k == 0:
            break
        E += k
        budget -= k * cost
        if k < r:
            break
        h += 1
    return E


def solve_cell(m, n):
    tris = list(combinations(range(m), 3))
    tri_idx = {t: i for i, t in enumerate(tris)}
    types = []
    for w in range(m, 2, -1):  # heaviest first: better early bounds
        for b in combinations(range(m), w):
            types.append([tri_idx[t] for t in combinations(b, 3)])
    weights = [w for w in range(m, 2, -1) for _ in combinations(range(m), w)]

    cap = [2] * len(tris)
    best = [0]

    def dfs(i, cols_used, edges):
        r = n - cols_used
        if edges + 2 * r > best[0] and r >= 0:
            pass
        # closed-form completion with pads only:
        if edges + 2 * r > best[0]:
            best[0] = edges + 2 * r
        if i == len(types) or r == 0:
            return
        # sound prune
        if edges + wf_future(r, sum(cap), m) <= best[0]:
            return
        ts = types[i]
        w = weights[i]
        mmax = min(cap[t] for t in ts)
        mmax = min(mmax, 2, r)
        for mult in range(mmax, -1, -1):
            if mult:
                for t in ts:
                    cap[t] -= mult
            dfs(i + 1, cols_used + mult, edges + mult * w)
            if mult:
                for t in ts:
                    cap[t] += mult

    dfs(0, 0, 0)
    return best[0]


if __name__ == "__main__":
    import importlib.util, os, time
    HERE = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "ev", os.path.join(HERE, "..", "..", "evaluator.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    Z = dict(ev.KST_EXACT_VALUE)

    rows = [int(a) for a in sys.argv[1:]] or [6, 7]
    for m in rows:
        ns = sorted(n for (mm, n) in Z if mm == m)
        allok = True
        for n in ns:
            t0 = time.time()
            z = solve_cell(m, n)
            dt = time.time() - t0
            ok = z == Z[(m, n)]
            allok &= ok
            mark = "OK" if ok else f"MISMATCH (published {Z[(m,n)]})"
            print(f"z({m},{n}) = {z}  [{dt:.1f}s]  {mark}", flush=True)
        print(f"row {m}:", "ALL MATCH" if allok else "HAS MISMATCH", flush=True)

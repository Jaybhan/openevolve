"""Exact z(m,n;3,3) for small m — v3: heavy-family enumeration with CORRECT
per-family quad maxima (fixes v1's aggregation bug that collapsed distinct
capacity vectors sharing a (blocks, edges, capacity) summary).

Completeness argument (why this computes the true optimum):
Every K_{3,3}-free m×n matrix is a multiset of blocks; split as
  heavy (weight>=5) ∪ quads (weight 4) ∪ triples (weight 3) ∪ pads (<=2).
- The heavy layer is enumerated EXHAUSTIVELY (all multisets, mult<=2).
- Given the heavy layer's residual capacity vector, the max number of legal
  quads (qmax) is computed by exhaustive B&B; legality is monotone (any
  sub-multiset of a legal quad multiset is legal), so all q in [0, qmax]
  are achievable, and any q quads spend exactly 4q capacity total.
- Triples: one capacity unit each, any unit accepts one triple column, so
  only the residual TOTAL matters. Pads free.
So max edges at (m,n) = max over heavy families F and q <= min(qmax(F), n-|F|)
of  edges(F) + 4q + 2n' + min(n', capleft(F) - 4q),  n' = n - |F| - q.
Every step is exact; nothing is heuristic.
"""
import sys
from itertools import combinations
from math import comb

sys.setrecursionlimit(100000)


def solve_row(m, n_range, verbose=False):
    tris = list(combinations(range(m), 3))
    tri_idx = {t: i for i, t in enumerate(tris)}
    B = 2 * comb(m, 3)

    heavy_types, heavy_tris, heavy_w = [], [], []
    for w in range(5, m + 1):
        for b in combinations(range(m), w):
            heavy_types.append(b)
            heavy_tris.append([tri_idx[t] for t in combinations(b, 3)])
            heavy_w.append(w)

    quads = list(combinations(range(m), 4))
    quad_tris = [[tri_idx[t] for t in combinations(q, 3)] for q in quads]

    def max_quads(cap):
        best = [0]

        def dfs(i, count):
            if count > best[0]:
                best[0] = count
            if i == len(quads):
                return
            if count + min(sum(cap) // 4, (len(quads) - i) * 2) <= best[0]:
                return
            ts = quad_tris[i]
            mmax = min(2, min(cap[t] for t in ts))
            for mult in range(mmax, -1, -1):
                if mult:
                    for t in ts:
                        cap[t] -= mult
                dfs(i + 1, count + mult)
                if mult:
                    for t in ts:
                        cap[t] += mult
        dfs(0, 0)
        return best[0]

    # families[(nb, edges, used)] -> set of distinct residual capacity vectors
    families = {}
    cap = [2] * len(tris)

    def enum(i, nb, edges, used):
        families.setdefault((nb, edges, used), set()).add(tuple(cap))
        if i == len(heavy_types):
            return
        ts = heavy_tris[i]
        mmax = min(2, min(cap[t] for t in ts))
        w, c = heavy_w[i], comb(heavy_w[i], 3)
        for mult in range(1, mmax + 1):
            for t in ts:
                cap[t] -= mult
            enum(i + 1, nb + mult, edges + mult * w, used + mult * c)
            for t in ts:
                cap[t] += mult
        enum(i + 1, nb, edges, used)

    enum(0, 0, 0, 0)

    memo_q = {}
    best_q = {}  # key -> max qmax over its vecs
    nvec = 0
    for key, vecs in families.items():
        for vec in vecs:
            nvec += 1
            if vec not in memo_q:
                memo_q[vec] = max_quads(list(vec))
            best_q[key] = max(best_q.get(key, -1), memo_q[vec])
    if verbose:
        print(f"m={m}: {len(families)} summaries, {len(memo_q)} distinct "
              f"capacity vectors ({nvec} pairs)", flush=True)

    z = {}
    for n in n_range:
        best = 0
        for (nb, edges, used), qmax in best_q.items():
            if nb > n:
                continue
            capleft = B - used
            hi = min(qmax, n - nb)
            for q in range(hi + 1):
                np_ = n - nb - q
                e = edges + 4 * q + 2 * np_ + min(np_, capleft - 4 * q)
                if e > best:
                    best = e
        z[n] = best
    return z


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
        t0 = time.time()
        z = solve_row(m, ns, verbose=True)
        dt = time.time() - t0
        ok = all(z[n] == Z[(m, n)] for n in ns)
        print(f"row {m} [{dt:.0f}s]:", "ALL MATCH" if ok else "MISMATCH", flush=True)
        for n in ns:
            mark = "" if z[n] == Z[(m, n)] else f"  <-- published {Z[(m,n)]}"
            print(f"  z({m},{n}) = {z[n]}{mark}", flush=True)

"""Core library for the finite-catalogue audit.

Witness loading (all block-list JSON banks), independent legality
verification, invariant extraction, backtracking isomorphism /
embedding / automorphism machinery for block multisets on m points.
Everything self-contained (no nauty, no networkx): sizes are m <= 27,
cols <= 90, where signature-pruned backtracking is fast.

Conventions: a *config* is (m, blocks) with blocks a list of sorted
row-index tuples (multiset semantics: repeats allowed, multiplicity <= 2
for legality). K_{3,3}-freeness <=> every 3-subset of rows covered <= 2.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
W1 = os.path.join(BASE, "analysis", "witnesses")
W2 = os.path.join(BASE, "analysis", "design_prover", "witnesses")
W3 = os.path.join(BASE, "analysis", "sat_attack", "witnesses")


# ----------------------------------------------------------------- loading

def load_banks(include_sat=True):
    """Return list of witness dicts: file, bank, m, n, edges, blocks."""
    out = []
    for fn in sorted(os.listdir(W1)):
        if not fn.endswith(".json"):
            continue
        js = json.load(open(os.path.join(W1, fn)))
        out.append(dict(file=fn, bank="witnesses", m=js["m"], n=js["n"],
                        edges=js["edges"],
                        blocks=[tuple(sorted(b)) for b in js["blocks"]]))
    for fn in sorted(os.listdir(W2)):
        if not fn.endswith(".json") or "satfmt" in fn:
            continue
        js = json.load(open(os.path.join(W2, fn)))
        bl = [tuple(sorted(b)) for b in js["blocks"]]
        out.append(dict(file=fn, bank="design_prover", m=js["m"],
                        n=len(bl), edges=sum(len(b) for b in bl),
                        blocks=bl, packing=True,
                        leave=js.get("leave")))
    if include_sat:
        for fn in sorted(os.listdir(W3)):
            if not fn.endswith(".json"):
                continue
            js = json.load(open(os.path.join(W3, fn)))
            if "blocks" not in js or "m" not in js:
                continue
            bl = [tuple(sorted(b)) for b in js["blocks"]]
            out.append(dict(file=fn, bank="sat_attack", m=js["m"],
                            n=js.get("n", len(bl)),
                            edges=js.get("edges", sum(len(b) for b in bl)),
                            blocks=bl))
    return out


# ------------------------------------------------------------ verification

def coverage(blocks):
    cov = Counter()
    for b in blocks:
        for t in combinations(sorted(b), 3):
            cov[t] += 1
    return cov


def verify(m, blocks, n=None, edges=None):
    """Independent legality check. Returns dict of facts."""
    ok_rows = all(0 <= x < m for b in blocks for x in b)
    ok_sets = all(len(set(b)) == len(b) for b in blocks)
    cov = coverage(blocks)
    maxcov = max(cov.values()) if cov else 0
    e = sum(len(b) for b in blocks)
    res = dict(legal=(ok_rows and ok_sets and maxcov <= 2),
               maxcov=maxcov, edges=e)
    if edges is not None:
        res["edges_match"] = (e == edges)
    if n is not None:
        res["cols_match"] = (len(blocks) == n)
    return res


# -------------------------------------------------------------- invariants

def weight_profile(blocks):
    return Counter(len(b) for b in blocks)


def layers(m, blocks):
    """Split into weight layers dict w -> list of blocks."""
    d = {}
    for b in blocks:
        d.setdefault(len(b), []).append(b)
    return d


def complements(m, layer):
    full = frozenset(range(m))
    return [tuple(sorted(full - set(b))) for b in layer]


def pair_deg(blocks, m):
    pd = Counter()
    for b in blocks:
        for p in combinations(sorted(b), 2):
            pd[p] += 1
    return pd


def point_deg(blocks, m):
    d = [0] * m
    for b in blocks:
        for x in b:
            d[x] += 1
    return d


def inter_dist(bl):
    return Counter(len(set(a) & set(b)) for a, b in combinations(bl, 2))


def invariant_summary(m, blocks):
    """Cheap iso-invariants of the whole config (used for reporting and
    as pruning keys)."""
    lay = layers(m, blocks)
    prof = tuple(sorted(((w, len(v)) for w, v in lay.items()), reverse=True))
    cov = coverage(blocks)
    ncov = Counter(cov.values())
    inv = dict(profile=prof,
               cov1=ncov.get(1, 0), cov2=ncov.get(2, 0),
               deg_multiset=tuple(sorted(point_deg(blocks, m))),
               dup_blocks=sum(1 for c in Counter(blocks).values() if c == 2))
    per = {}
    for w, bl in lay.items():
        per[w] = dict(count=len(bl),
                      inter=tuple(sorted(inter_dist(bl).items())),
                      deg=tuple(sorted(point_deg(bl, m))))
    inv["per_weight"] = per
    return inv


# ------------------------------------------- signatures for backtracking

def point_signatures(m, blocks):
    """Per-point invariant vector: (per-weight degree, per-weight
    pair-degree multiset).  Refined iteratively (1-WL on the incidence
    structure) to a stable coloring; returns list of hashable sigs."""
    bl = [frozenset(b) for b in blocks]
    wts = [len(b) for b in bl]
    mult = Counter(tuple(sorted(b)) for b in blocks)
    # initial: degree per (weight, multiplicity-class of block)
    bcol = [(wts[i], mult[tuple(sorted(bl[i]))]) for i in range(len(bl))]
    pcol = [0] * m
    for _ in range(m):
        newp = []
        for x in range(m):
            s = Counter()
            for i, b in enumerate(bl):
                if x in b:
                    s[bcol[i]] += 1
            newp.append((pcol[x], tuple(sorted(s.items()))))
        # canonicalize to small ints
        order = {v: i for i, v in enumerate(sorted(set(newp)))}
        newp = [order[v] for v in newp]
        newb = []
        for i, b in enumerate(bl):
            s = tuple(sorted(newp[x] for x in b))
            newb.append((bcol[i], s))
        orderb = {v: i for i, v in enumerate(sorted(set(newb)))}
        newb = [orderb[v] for v in newb]
        if newp == pcol and newb == bcol:
            break
        pcol, bcol = newp, newb
    return pcol


# ------------------------------------------------ iso / embed / autgroup

def _blockset_key(blocks, perm):
    return sorted(tuple(sorted(perm[x] for x in b)) for b in blocks)


def find_embedding(mA, A, mB, B, require_iso=False, count_all=False,
                   node_cap=2_000_000):
    """Injective point map f: [mA] -> [mB] with f(A) <= B as multisets.
    If require_iso: mA == mB and f(A) == B.  Backtracking on points in
    order of decreasing constraint, pruning with pair-degree feasibility.
    Returns one embedding (list) or None; if count_all, returns count.
    """
    A = [tuple(sorted(b)) for b in A]
    B = [tuple(sorted(b)) for b in B]
    if require_iso and (mA != mB or sorted(map(len, A)) != sorted(map(len, B))):
        return 0 if count_all else None
    multB = Counter(B)
    pdA = pair_deg(A, mA)
    pdB = pair_deg(B, mB)
    dA_w = [Counter() for _ in range(mA)]
    for b in A:
        for x in b:
            dA_w[x][len(b)] += 1
    dB_w = [Counter() for _ in range(mB)]
    for b in B:
        for x in b:
            dB_w[x][len(b)] += 1

    def deg_ok(x, y):
        # embedding: each weight-class degree of x must be <= that of y
        if require_iso:
            return dA_w[x] == dB_w[y]
        return all(dB_w[y][w] >= c for w, c in dA_w[x].items())

    cand0 = [[y for y in range(mB) if deg_ok(x, y)] for x in range(mA)]
    if any(not c for c in cand0):
        return 0 if count_all else None
    # order points: most-constrained first, but keep connectivity
    order = sorted(range(mA), key=lambda x: (len(cand0[x]),
                                             -sum(dA_w[x].values())))
    pos = {x: i for i, x in enumerate(order)}
    # blocks fully decided once all their points are placed
    blocks_by_last = [[] for _ in range(mA)]
    for b in A:
        last = max(b, key=lambda x: pos[x])
        blocks_by_last[last].append(b)

    f = [-1] * mA
    used = [False] * mB
    nodes = [0]
    found = []
    cnt = [0]

    def bt(i, remaining):
        if nodes[0] > node_cap:
            raise TimeoutError("node cap")
        nodes[0] += 1
        if i == mA:
            if require_iso and +remaining != Counter():
                # remaining must be exactly empty for iso; for embed
                # remaining >= 0 by construction
                return False
            if count_all:
                cnt[0] += 1
                return False
            found.append(list(f))
            return True
        x = order[i]
        for y in cand0[x]:
            if used[y]:
                continue
            # pair-degree pruning vs already-placed points
            ok = True
            for j in range(i):
                z = order[j]
                pa = pdA.get((min(x, z), max(x, z)), 0)
                if pa:
                    pb = pdB.get((min(y, f[z]), max(y, f[z])), 0)
                    if require_iso:
                        if pb != pa:
                            ok = False
                            break
                    elif pb < pa:
                        ok = False
                        break
                elif require_iso:
                    pb = pdB.get((min(y, f[z]), max(y, f[z])), 0)
                    if pb != 0:
                        ok = False
                        break
            if not ok:
                continue
            f[x] = y
            used[y] = True
            # check completed blocks
            done = []
            good = True
            for b in blocks_by_last[x]:
                img = tuple(sorted(f[t] for t in b))
                if remaining[img] > 0:
                    remaining[img] -= 1
                    done.append(img)
                else:
                    good = False
                    break
            if good and bt(i + 1, remaining):
                return True
            for img in done:
                remaining[img] += 1
            f[x] = -1
            used[y] = False
        return False

    try:
        hit = bt(0, Counter(multB))
    except TimeoutError:
        return "TIMEOUT"
    if count_all:
        return cnt[0]
    return found[0] if hit else None


def isomorphic(mA, A, mB, B):
    r = find_embedding(mA, A, mB, B, require_iso=True)
    return r is not None and r != "TIMEOUT"


def aut_order(m, blocks):
    """|Aut| of the block multiset (row permutations fixing it)."""
    r = find_embedding(m, blocks, m, blocks, require_iso=True,
                       count_all=True)
    return r


# ----------------------------------------------------------- small helpers

def canon_graph_name(edges, nverts=None):
    """Name small graphs given as edge list on arbitrary labels."""
    vs = sorted(set(v for e in edges for v in e))
    n = len(vs)
    deg = Counter(v for e in edges for v in e)
    ds = tuple(sorted(deg.values(), reverse=True))
    ne = len(edges)
    idx = {v: i for i, v in enumerate(vs)}
    adj = [[False] * n for _ in range(n)]
    for a, b in edges:
        adj[idx[a]][idx[b]] = adj[idx[b]][idx[a]] = True
    # connected components
    seen = [False] * n
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(n):
                if adj[u][v] and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(comp)
    if ne == 0:
        return "empty"
    if all(d == 1 for d in deg.values()):
        return f"matching({ne})"
    if ne == n and all(deg[v] == 2 for v in vs) and len(comps) == 1:
        return f"C{n}"
    if len(comps) == 1 and ds == tuple([n - 1] + [1] * (n - 1)):
        return f"star(K1,{n-1})"
    if len(comps) == 1 and ne == n - 1 and max(ds) == 2:
        return f"path(P{n})"
    # complete bipartite?
    for asz in range(1, n):
        if ne == asz * (n - asz):
            from itertools import combinations as C2
            for A in C2(range(n), asz):
                As = set(A)
                if all(adj[u][v] == ((u in As) != (v in As))
                       for u in range(n) for v in range(u + 1, n)):
                    return f"K{asz},{n-asz}"
    tri = any(adj[a][b] and adj[b][c] and adj[a][c]
              for a, b, c in combinations(range(n), 3))
    return (f"graph(v={n},e={ne},deg={ds}" +
            (",trianglefree" if not tri else "") + ")")


def name_3graph(triples, m=None):
    """Recognize small 3-uniform hypergraphs (with multiplicity)."""
    mult = Counter(tuple(sorted(t)) for t in triples)
    distinct = sorted(mult)
    pts = sorted(set(x for t in distinct for x in t))
    n = len(pts)
    pd = pair_deg(distinct, 0)
    dbl = [t for t, c in mult.items() if c == 2]
    tag = "doubled " if len(dbl) == len(distinct) and dbl else ""
    if len(distinct) == 1:
        t = ("doubled triple" if mult[distinct[0]] == 2 else "single triple")
        return t
    # parallel class (disjoint triples)
    if all(c <= 1 for c in pd.values()):
        if all(v == 0 or True for v in pd.values()) and \
           len(set(x for t in distinct for x in t)) == 3 * len(distinct):
            return tag + f"parallel({len(distinct)})"
    if n == 7 and len(distinct) == 7 and all(c == 1 for c in pd.values()) \
            and len(pd) == 21:
        return tag + "Fano"
    if all(c <= 1 for c in pd.values()):
        return tag + f"linear3graph(v={n},b={len(distinct)})"
    # sunflower/pencil: common pair
    from functools import reduce
    common = reduce(lambda a, b: set(a) & set(b), distinct, set(distinct[0]))
    if len(common) >= 2:
        return tag + f"pencil(core={len(common)},b={len(distinct)})"
    if n == 4 and len(distinct) == 4:
        return tag + "K4^(3)"
    return (tag + f"3graph(v={n},b={len(distinct)},"
            f"pairdeg={tuple(sorted(Counter(pd.values()).items()))})")

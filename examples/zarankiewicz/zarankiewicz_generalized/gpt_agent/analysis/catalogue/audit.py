"""Completeness audit: decompose every witness into catalogue generators.

For each witness the audit emits a decomposition certificate (layer ->
recognized generator instance, label-aware, so the certificate exactly
regenerates the witness column multiset), or PARTIAL/UNDECODED with
invariants.  No solver calls; recognition is combinatorial search only.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from itertools import combinations, permutations

import catlib as C
import gens as G


# ----------------------------------------------------------- leave helper

def leave_of(m, blocks):
    cov = C.coverage(blocks)
    lv = Counter()
    for t in combinations(range(m), 3):
        r = 2 - cov.get(t, 0)
        if r > 0:
            lv[t] = r
    return lv


def is_doubled_pentagon(leave):
    """leave: Counter triple->mult.  True iff = 2x C5-edge-complements
    on some 5 points."""
    if sum(leave.values()) != 10 or any(v != 2 for v in leave.values()):
        return None
    tris = sorted(leave)
    pts = sorted(set(x for t in tris for x in t))
    if len(pts) != 5 or len(tris) != 5:
        return None
    # each point in exactly 3 of the 5 triples
    dg = Counter(x for t in tris for x in t)
    if any(dg[p] != 3 for p in pts):
        return None
    # complement pairs (within 5-set) must form a 5-cycle
    edges = [tuple(sorted(set(pts) - set(t))) for t in tris]
    adj = Counter(x for e in edges for x in e)
    if any(adj[p] != 2 for p in pts):
        return None
    # connected 2-regular = C5
    nbr = {p: [] for p in pts}
    for a, b in edges:
        nbr[a].append(b)
        nbr[b].append(a)
    seen = {pts[0]}
    frontier = [pts[0]]
    while frontier:
        x = frontier.pop()
        for y in nbr[x]:
            if y not in seen:
                seen.add(y)
                frontier.append(y)
    if len(seen) != 5:
        return None
    return pts


def is_doubled_parallel_class(leave, m):
    """leave = 2x a partition of [m] into triples?"""
    if any(v != 2 for v in leave.values()):
        return None
    tris = sorted(leave)
    pts = [x for t in tris for x in t]
    if len(pts) != m or len(set(pts)) != m:
        return None
    return tris


# ------------------------------------------------------------ recognizers
# each returns (name, extra) or None

def rec_complement_layer(m, layer):
    w = len(layer[0])
    j = m - w
    if j < 1 or j > 3:
        return None
    comps = C.complements(m, layer)
    mult = Counter(comps)
    if j == 1:
        pts = sorted(x for (x,) in comps)
        return (f"PointComp{{{','.join(map(str, pts))}}}", None)
    if j == 2:
        nm = C.canon_graph_name(comps)
        return (f"EdgeComp[{nm}]", comps)
    nm = name_j3(comps)
    return (f"TripleComp[{nm}]", comps)


def name_j3(triples):
    """Names for 3-sets systems (complements of (m-3)-blocks or fills)."""
    mult = Counter(triples)
    distinct = sorted(mult)
    # pencil-partition: {p} u P_i, P_i disjoint pairs
    for p in set(x for t in distinct for x in t):
        if all(p in t for t in distinct):
            rests = [tuple(sorted(set(t) - {p})) for t in distinct]
            flat = [x for r in rests for x in r]
            if len(set(flat)) == len(flat):
                return (f"pencilpartition(p={p},{len(distinct)} pairs)")
            return f"pencil(p={p},b={len(distinct)})"
    return C.name_3graph(triples)


def rec_fano(m, layer_comps):
    """comps: list of 3-sets; Fano on their 7-point support?"""
    distinct = sorted(set(layer_comps))
    pts = sorted(set(x for t in distinct for x in t))
    if len(pts) != 7 or len(distinct) != 7:
        return None
    pd = C.pair_deg(distinct, 0)
    if len(pd) == 21 and all(v == 1 for v in pd.values()):
        return "Fano"
    return None


def rec_cone(m, layer):
    """All blocks share a common point q -> cone; return q and the
    derived (w-1)-sets on [m]-q."""
    common = set(layer[0])
    for b in layer[1:]:
        common &= set(b)
    if not common:
        return None
    q = min(common)
    derived = [tuple(sorted(set(b) - {q})) for b in layer]
    return q, derived


def rec_parity_cone(m, layer):
    """m=9 style: pentads = {q} u transversal of a pair-partition of
    [m]-q, transversal parity constant (odd coset of the even-weight
    code).  Returns description or None."""
    cone = rec_cone(m, layer)
    if cone is None:
        return None
    q, derived = cone
    rest = sorted(set(range(m)) - {q})
    k = len(derived[0])
    if any(len(d) != k for d in derived):
        return None
    if len(rest) != 2 * k:
        return None
    # find pair partition: each derived set must hit each pair once
    # build graph: two points are "partners" if they never co-occur
    co = set()
    for d in derived:
        for a, b in combinations(d, 2):
            co.add((a, b))
    cand_partners = {x: [y for y in rest if y != x and
                         (min(x, y), max(x, y)) not in co]
                     for x in rest}
    # try to find perfect matching of rest into pairs (backtracking)
    pairs = []

    def bt(remaining):
        if not remaining:
            return True
        x = remaining[0]
        for y in cand_partners[x]:
            if y in remaining and y != x:
                pairs.append((x, y))
                rem2 = [z for z in remaining if z not in (x, y)]
                # check every derived set has exactly one of x,y
                if all(len(set(d) & {x, y}) == 1 for d in derived):
                    if bt(rem2):
                        return True
                pairs.pop()
        return False

    if not bt(rest):
        return None
    # parity: label pair (a,b): a=0,b=1; transversal -> vector
    vecs = []
    for d in derived:
        v = []
        for (a, b) in pairs:
            v.append(0 if a in set(d) else 1)
        vecs.append(tuple(v))
    wts = {sum(v) % 2 for v in vecs}
    par = ("odd" if wts == {1} else "even" if wts == {0} else "mixed")
    return dict(q=q, pairs=pairs, parity=par, k=len(derived),
                distinct_vecs=len(set(vecs)))


def rec_splits(m, layer):
    """Layer of complementary-pair blocks {P, [m]-P} (both-sides
    family), possibly doubled.  Returns #splits or None."""
    mult = Counter(layer)
    blocks = sorted(mult)
    full = set(range(m))
    unmatched = Counter(mult)
    nsplit = 0
    used = set()
    for b in blocks:
        cb = tuple(sorted(full - set(b)))
        if unmatched[b] > 0 and unmatched.get(cb, 0) > 0 and b <= cb:
            k = min(unmatched[b], unmatched[cb])
            nsplit += k
            unmatched[b] -= k
            unmatched[cb] -= k
    if sum(unmatched.values()) == 0 and nsplit > 0:
        return nsplit
    return None


def rec_biplane(m, layer):
    """2-(m,w,2): every pair covered exactly twice."""
    pd = C.pair_deg(layer, m)
    if len(pd) == m * (m - 1) // 2 and all(v == 2 for v in pd.values()):
        w = len(layer[0])
        return f"2-({m},{w},2) design"
    return None


def rec_2design(m, layer):
    pd = C.pair_deg(layer, m)
    vals = set(pd.values())
    if len(pd) == m * (m - 1) // 2 and len(vals) == 1:
        lam = vals.pop()
        w = len(layer[0])
        return f"2-({m},{w},{lam}) design"
    return None


def rec_k5_edge(m, layer, kind):
    """m=10: is layer isomorphic (as labeled on SOME identification of
    [10] with E(K5)) to a known K5-edge family?  We test embed of the
    layer into the reference structure of the SAME size, requiring the
    reference CONTAIN the layer.  kind in {'K4sets','pentagons','stars'}.
    Returns the identification or None."""
    if m != 10:
        return None
    ref = {"K4sets": G.k5_k4sets(), "pentagons": G.k5_pentagons(),
           "stars": G.k5_stars()}[kind]
    r = C.find_embedding(10, layer, 10, ref)
    if r is None or r == "TIMEOUT":
        return None
    return r


def rec_submultiset_of(m, layer, ref, refname, node_cap=3_000_000):
    """Is layer (as multiset) mappable into ref by a point bijection?"""
    r = C.find_embedding(m, layer, m, ref, node_cap=node_cap)
    if r == "TIMEOUT":
        return "TIMEOUT"
    if r is None:
        return None
    return refname


# --------------------------------------------------------- reference bank

def reference_packings():
    """Constructive reference maximum packings per m."""
    refs = {}
    refs[7] = [("2xFano-comp+", None)]  # placeholder, see build below
    return refs


def load_ref_layer(path, key="blocks"):
    js = json.load(open(path))
    return [tuple(sorted(b)) for b in js[key]]


# ------------------------------------------------------------- the audit

def audit_one(w, refs, log):
    m, blocks = w["m"], w["blocks"]
    lay = C.layers(m, blocks)
    parts = []
    status_flags = []

    def note(s):
        log.append(f"    {s}")

    for wt in sorted([x for x in lay if x >= 4], reverse=True):
        bl = lay[wt]
        tag = None
        j = m - wt
        # 1. complement layers
        if 1 <= j <= 3:
            nm, comps = rec_complement_layer(m, bl)
            tag = nm
            if j == 3:
                f = rec_fano(m, C.complements(m, bl))
                if f:
                    tag = f"TripleComp[Fano({len(bl)})]"
        else:
            # 2. named designs
            d2 = rec_2design(m, bl)
            if d2:
                tag = d2
            # 3. splits (both sides)
            if tag is None:
                s = rec_splits(m, bl)
                if s:
                    tag = f"Splits(x{s})"
            # 4. cones (incl. parity cones)
            if tag is None:
                pc = rec_parity_cone(m, bl)
                if pc:
                    tag = (f"ParityCone(q={pc['q']},{pc['k']} blocks,"
                           f"{pc['parity']})")
                else:
                    cn = rec_cone(m, bl)
                    if cn and len(bl) > 1:
                        q, derived = cn
                        tag = f"Cone(q={q},{name_derived(m, q, derived)})"
            # 5. K5-edge world (m=10)
            if tag is None and m == 10:
                for kind in ("K4sets", "pentagons", "stars"):
                    ref = {"K4sets": G.k5_k4sets(),
                           "pentagons": G.k5_pentagons(),
                           "stars": G.k5_stars()}[kind]
                    if len(bl[0]) == len(ref[0]):
                        r = C.find_embedding(10, bl, 10, ref)
                        if r and r != "TIMEOUT":
                            tag = f"K5edge[{kind}](x{len(bl)})"
                            break
            # 6. reference packings (sub-multiset)
            if tag is None and (m, wt) in refs:
                for refname, ref in refs[(m, wt)]:
                    r = rec_submultiset_of(m, bl, ref, refname)
                    if r == "TIMEOUT":
                        note(f"w={wt}: embed into {refname} TIMEOUT")
                    elif r:
                        tag = f"Sub[{refname}](x{len(bl)})"
                        break
        if tag is None:
            tag = f"UNRECOGNIZED(w={wt},x{len(bl)})"
            status_flags.append("U")
        parts.append(tag)

    if lay.get(3):
        mult = Counter(lay[3])
        parts.append("Fills(x%d)" % len(lay[3]))
    np = len(lay.get(2, [])) + len(lay.get(1, []))
    if np:
        parts.append(f"Pads(x{np})")
    status = ("DECODED" if not status_flags else
              ("PARTIAL" if len(status_flags) < len(
                  [x for x in lay if x >= 4]) else "UNDECODED"))
    return parts, status


def name_derived(m, q, derived):
    return f"derived {len(derived[0])}-sets x{len(derived)}"


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    ws = C.load_banks()
    refs = {}
    log = []
    for w in ws:
        parts, status = audit_one(w, refs, log)
        print(f"[{status}] {w['file']} (m={w['m']},n={w['n']}): "
              + " + ".join(parts))
        for line in log:
            print(line)
        log.clear()

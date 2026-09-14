"""Final layer decodes:
(a) quad-layer leaves = disjoint quad shadows (=> sub-3-(m,4,2))
(b) w_8x13/14/15 pentad systems: mutual iso? aut?
(c) 10x47 pentads: GDD/OA(8,5,2,2) test
(d) smoke_10x14 hexad comps vs SQS(10)
(e) 10x15 arcs = Moebius circles (residual SQS(10) orbit)?
(f) m=10 midband quads vs splits
"""
import sys
from collections import Counter
from itertools import combinations

import catlib as C
import gens as G
from audit import leave_of


def quad_shadow_completion(m, quads):
    """Is leave(quads) a disjoint union of single quad-shadows (4
    triples of a 4-set, each multiplicity 1)?  If so return the
    completing quads."""
    lv = leave_of(m, quads)
    if any(v != 1 for v in lv.values()):
        return None
    tris = sorted(lv)
    if len(tris) % 4:
        return None
    # group triples by their support-4 candidate: a shadow of quad Q
    # = the 4 triples inside Q
    from collections import defaultdict
    used = set()
    comp = []
    remaining = set(tris)
    while remaining:
        t = min(remaining)
        # find 4th point: the quad = t + x s.t. all 4 triples present
        found = None
        for x in range(m):
            if x in t:
                continue
            q = tuple(sorted(set(t) | {x}))
            sh = set(combinations(q, 3))
            if sh <= remaining:
                found = q
                break
        if not found:
            return None
        comp.append(found)
        remaining -= set(combinations(found, 3))
    return comp


def gdd_oa_test(m, pentads):
    """10x47-style: uncovered pairs form perfect matching; blocks =
    transversals; cross pairs covered exactly lambda."""
    pd = C.pair_deg(pentads, m)
    unc = [p for p in combinations(range(m), 2) if pd.get(p, 0) == 0]
    # perfect matching?
    pts = Counter(x for p in unc for x in p)
    if len(unc) != m // 2 or any(v != 1 for v in pts.values()) \
            or len(pts) != m:
        return None
    groups = sorted(unc)
    gi = {}
    for i, g in enumerate(groups):
        for x in g:
            gi[x] = i
    for b in pentads:
        if sorted(gi[x] for x in b) != list(range(len(groups))):
            return None
    lam = set(pd[p] for p in pd if gi[p[0]] != gi[p[1]])
    if len(lam) != 1:
        return None
    return dict(groups=groups, lam=lam.pop(), nblocks=len(pentads))


def main():
    ws = {w["file"]: w for w in C.load_banks()}

    print("== (a) quad-shadow completions (sub-3-(m,4,2) certificates)")
    for f in ["w_8x25", "w_8x26", "w_8x27", "w_10x58", "w_10x59",
              "w_9x39"]:
        w = ws[f + ".json"]
        m = w["m"]
        quads = C.layers(m, w["blocks"]).get(4, [])
        r = quad_shadow_completion(m, quads)
        print(f"  {f}: completes with {len(r) if r else 'NO'} quads"
              + (f" -> perfect 2-cover of size {len(quads)+len(r)}"
                 if r else ""))

    print("\n== (b) w_8x13/14/15 pentad systems")
    lays = {}
    for f in ["w_8x13", "w_8x14", "w_8x15", "w_8x10"]:
        lays[f] = C.layers(8, ws[f + ".json"]["blocks"])[5]
    for a, b in [("w_8x13", "w_8x14"), ("w_8x13", "w_8x15"),
                 ("w_8x14", "w_8x15")]:
        if len(lays[a]) == len(lays[b]):
            r = C.find_embedding(8, lays[a], 8, lays[b],
                                 require_iso=True)
            print(f"  {a} ~ {b}: {'ISO' if r and r != 'TIMEOUT' else 'no'}")
    for f in ["w_8x13", "w_8x14", "w_8x15"]:
        # try embedding INTO w_8x14's 8 (largest twin) plus check
        # complement 3-graph names
        comps = C.complements(8, lays[f])
        print(f"  {f}: comp triples {sorted(comps)}")

    print("\n== (c) 10x47/48/53 pentad GDD/OA tests")
    for f in ["w_10x47", "w_10x48", "w_10x35", "w_10x36"]:
        w = ws[f + ".json"]
        pen = C.layers(10, w["blocks"]).get(5, [])
        r = gdd_oa_test(10, pen)
        print(f"  {f}: GDD-transversal: {r}")

    print("\n== (d) smoke_10x14 hexad comps vs SQS(10)")
    w = ws["smoke_10x14_77_witness.json"]
    hexads = C.layers(10, w["blocks"])[6]
    comps = C.complements(10, hexads)
    s10 = G.sqs10()
    r = C.find_embedding(10, comps, 10, s10)
    print(f"  7 hexad-comps sub-SQS(10): "
          f"{'YES' if r and r != 'TIMEOUT' else r}")
    pen = C.layers(10, w["blocks"])[5]
    r2 = C.find_embedding(10, pen + [], 10,
                          [tuple(sorted(set(range(10)) - set(b)))
                           for b in s10])
    print(f"  7 pentads sub-SQS(10)-complements: "
          f"{'YES' if r2 and r2 != 'TIMEOUT' else r2}")
    # joint: comps(hexads) + comps(pentads sont 5-sets)... try hexcomps
    # + pentads-as-own vs sqs10 + comps
    print("  pentad pair-deg:", dict(sorted(Counter(
        C.pair_deg(pen, 10).values()).items())))

    print("\n== (e) 10x15 arcs vs Moebius circles")
    w = ws["T2_10x15_81_cadical_witness.json"]
    lay = C.layers(10, w["blocks"])
    der = [tuple(sorted(set(p) - {0})) for p in lay[5]]
    s10 = G.sqs10()
    # circles avoiding point 9 (INF), as 4-sets on 0..8
    resid = [b for b in s10 if 9 not in b]
    print(f"  residual SQS(10) at INF: {len(resid)} blocks")
    # relabel derived arcs {1..9} -> {0..8}
    der8 = [tuple(x - 1 for x in t) for t in der]
    r = C.find_embedding(9, der8, 9, resid)
    print(f"  9 arcs sub-residual-SQS10: "
          f"{'YES' if r and r != 'TIMEOUT' else r}")
    # full pentad structure: {0} u arc where arcs = one translation
    # orbit of a circle: verified via orbit test + this membership.

    print("\n== (f) m=10 midband: quads vs splits")
    for f in ["w_10x35", "w_10x36"]:
        w = ws[f + ".json"]
        lay = C.layers(10, w["blocks"])
        pen = lay[5]
        quads = lay[4]
        # split sides
        mult = Counter(pen)
        full = set(range(10))
        sides = sorted(set(pen))
        inside = 0
        for q in quads:
            if any(set(q) <= set(s) for s in sides):
                inside += 1
        print(f"  {f}: quads inside a split side: {inside}/{len(quads)}")
        lv = leave_of(10, quads)
        print(f"    quad-layer leave weight {sum(lv.values())}")
        r = quad_shadow_completion(10, quads)
        print(f"    shadow completion: {len(r) if r else 'NO'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()

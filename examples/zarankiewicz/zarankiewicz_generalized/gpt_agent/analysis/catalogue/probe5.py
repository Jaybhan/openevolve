"""(1) all-AG classification for every m=9 witness block; (2) 10x15
translation-orbit test; (3) m=8 pentad layers vs the max 10-pentad
system; (4) leaves of all pure-quad bodies (hub shapes)."""
import sys
from collections import Counter
from itertools import combinations, permutations

import catlib as C
import gens as G
from audit import leave_of


def all_ag_copies():
    """All distinct AG(2,3) line-sets on 0..8 (as frozensets of 12
    lines).  Orbit of the standard copy under S9, deduped."""
    std = G.ag23_lines()
    seen = set()
    out = []
    for perm in permutations(range(9)):
        img = frozenset(tuple(sorted(perm[x] for x in l)) for l in std)
        if img not in seen:
            seen.add(img)
            out.append(img)
    return out


def ag_tables(lines, m=9):
    """Precompute membership tables for one AG copy."""
    lineset = set(tuple(sorted(l)) for l in lines)
    linept = set()
    for l in lineset:
        for x in range(m):
            if x not in l:
                linept.add(tuple(sorted(set(l) | {x})))
    pp = set()
    for z in range(m):
        thru = [l for l in lineset if z in l]
        for l1, l2 in combinations(thru, 2):
            pp.add(tuple(sorted((set(l1) | set(l2)) - {z})))
    return lineset, linept, pp


def classify_block_tab(block, tabs, m=9):
    lineset, linept, pp = tabs
    b = tuple(sorted(block))
    if len(b) == 3:
        return "LINE" if b in lineset else "NONLINE"
    if len(b) == 4:
        if b in linept:
            return "LINEPT"
        if b in pp:
            return "PENCILPAIR"
        inner = sum(1 for l in lineset if set(l) <= set(b))
        return "ARC" if inner == 0 else f"MULTI{inner}"
    if len(b) <= 6:
        comp = tuple(sorted(set(range(m)) - set(b)))
        pre = "comp-" if len(b) == 5 else "comp3-"
        return pre + classify_block_tab(comp, tabs, m)
    return f"w{len(b)}"


GOOD = ("LINEPT", "PENCILPAIR", "comp-LINEPT", "comp-PENCILPAIR",
        "comp3-LINE")


def score_ag(blocks, tabs, m=9):
    good = 0
    tags = []
    for b in blocks:
        if len(b) < 4:
            continue
        t = classify_block_tab(b, tabs, m)
        tags.append(t)
        if t in GOOD:
            good += 1
    return good, Counter(tags)


def main():
    ws = {w["file"]: w for w in C.load_banks()}
    print("generating AG(2,3) copies...", flush=True)
    ags = all_ag_copies()
    print(f"  {len(ags)} distinct copies")

    tabs = [ag_tables(lines) for lines in ags]
    print("\n== (1) best-AG classification, all m=9 witnesses")
    for f in sorted(k for k in ws if k.startswith("w_9x")):
        w = ws[f]
        heavy = [b for b in w["blocks"] if len(b) >= 4]
        best = None
        for tb in tabs:
            g, tags = score_ag(heavy, tb)
            if best is None or g > best[0]:
                best = (g, tags)
        g, tags = best
        print(f"  {f}: {g}/{len(heavy)} structured; {dict(tags)}")

    print("\n== (2) 10x15 translation-orbit test")
    w = ws["T2_10x15_81_cadical_witness.json"]
    lay = C.layers(10, w["blocks"])
    der = [tuple(sorted(set(p) - {0})) for p in lay[5]]
    # coordinates: find AG copy on {1..9}->0..8 making T lines; use
    # reconstructed lines from probe4 output, coordinatize:
    lines9 = [(1, 2, 6), (1, 3, 7), (1, 4, 5), (1, 8, 9), (2, 3, 5),
              (2, 4, 9), (2, 7, 8), (3, 4, 8), (3, 6, 9), (4, 6, 7),
              (5, 6, 8), (5, 7, 9)]
    # coordinatize: pick point 1 = (0,0); lines through 1: (126),(137),
    # (145),(189): assign directions
    # brute force: try all bijections {1..9}->F3^2 preserving lines
    # (there are many; find one)
    pts = list(range(1, 10))
    vecs = [(a, b) for a in range(3) for b in range(3)]
    lineset = set(tuple(sorted(l)) for l in lines9)

    def is_line(t):
        return tuple(sorted(t)) in lineset

    found = None
    for perm in permutations(vecs):
        ok = True
        vmap = dict(zip(pts, perm))
        for l in lines9:
            a, b, c = (vmap[x] for x in l)
            if ((a[0] + b[0] + c[0]) % 3, (a[1] + b[1] + c[1]) % 3) \
                    != (0, 0):
                ok = False
                break
        if ok:
            found = vmap
            break
    print("  coordinatization found:", found is not None)
    if found:
        # arcs as vector sets; check translation orbit
        arcsets = [frozenset(found[x] for x in d) for d in der]

        def translate(s, v):
            return frozenset(((a + v[0]) % 3, (b + v[1]) % 3)
                             for a, b in s)

        base = arcsets[0]
        orbit = set(translate(base, v) for v in vecs)
        print("  orbit of first arc covers all 9:",
              set(arcsets) == orbit, f"(|orbit|={len(orbit)})")

    print("\n== (3) m=8 pentad layers vs max 10-pentad system")
    ref = C.layers(8, ws["w_8x10.json"]["blocks"])[5]
    for f in ["w_8x9", "w_8x11", "w_8x13", "w_8x14", "w_8x15", "w_8x16",
              "w_8x18", "w_8x19", "w_8x20", "w_8x21", "w_8x22", "w_8x23",
              "w_8x24"]:
        w = ws[f + ".json"]
        lay = C.layers(8, w["blocks"]).get(5, [])
        if not lay:
            continue
        r = C.find_embedding(8, lay, 8, ref)
        print(f"  {f}: {len(lay)} pentads sub-of-max10: "
              f"{'YES' if r and r != 'TIMEOUT' else r}")

    print("\n== (4) leaves of pure-quad bodies")
    for f in ["w_9x39", "w_9x40", "w_9x41", "w_9x42", "w_9x43", "w_9x44",
              "w_9x45", "w_9x46", "w_9x47"]:
        w = ws[f + ".json"]
        quads = C.layers(9, w["blocks"])[4]
        lv = leave_of(9, quads)
        tris = sorted(lv)
        common = set.intersection(*(set(t) for t in tris)) if tris else set()
        print(f"  {f}: quad-leave w={sum(lv.values())}, "
              f"triples={len(tris)}, common point: {sorted(common)}, "
              f"all doubled: {all(v == 2 for v in lv.values())}")
    for f in ["w_10x58", "w_10x59"]:
        w = ws[f + ".json"]
        quads = C.layers(10, w["blocks"])[4]
        lv = leave_of(10, quads)
        print(f"  {f}: quad-leave w={sum(lv.values())}, "
              f"triples={sorted(lv.items())}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()

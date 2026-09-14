"""Test the AG(2,3) hypothesis for the m=9 mid-band witnesses and the
grid hypothesis for 10x15."""
import sys
from collections import Counter
from itertools import combinations

import catlib as C
import gens as G


def reconstruct_ag_from_lines(m, lines):
    """Given a set of triples claimed to be lines of an AG(2,3) on m=9
    points, complete to the full 12-line set if possible: every pair of
    points on exactly one line.  Greedy closure: any 2 lines meeting in
    one point ... simplest: try to extend by finding triples covering
    each uncovered pair consistently."""
    pd = C.pair_deg(lines, 9)
    if any(v > 1 for v in pd.values()):
        return None
    lines = set(lines)
    # uncovered pairs must be partitioned into triples forming lines
    while True:
        unc = [p for p in combinations(range(9), 2)
               if C.pair_deg(sorted(lines), 9).get(p, 0) == 0]
        if not unc:
            break
        # take first uncovered pair; the line through it is the third
        # point z s.t. {x,z} and {y,z} also uncovered
        x, y = unc[0]
        cands = [z for z in range(9) if z not in (x, y)
                 and (min(x, z), max(x, z)) in unc
                 and (min(y, z), max(y, z)) in unc]
        found = False
        for z in cands:
            cand = tuple(sorted((x, y, z)))
            lines.add(cand)
            pd2 = C.pair_deg(sorted(lines), 9)
            if all(v <= 1 for v in pd2.values()):
                found = True
                break
            lines.discard(cand)
        if not found:
            return None
    if len(lines) != 12:
        return None
    # verify: 12 lines, every pair exactly once, 4 parallel classes
    pd = C.pair_deg(sorted(lines), 9)
    if len(pd) != 36 or any(v != 1 for v in pd.values()):
        return None
    return sorted(lines)


def analyze_m9_ag(wname, ws):
    w = ws[wname]
    lay = C.layers(9, w["blocks"])
    quads = lay.get(4, [])
    pentads = lay.get(5, [])
    fills = lay.get(3, [])
    print(f"=== {wname}")
    # cone point of quads = max-degree point
    deg = C.point_deg(quads, 9)
    q0 = deg.index(max(deg))
    through = [b for b in quads if q0 in b]
    others = [b for b in quads if q0 not in b]
    derived = [tuple(sorted(set(b) - {q0})) for b in through]
    print(f"  quad cone point {q0}: {len(through)} through, "
          f"{len(others)} not")
    ag = reconstruct_ag_from_lines(9, derived)
    if ag is None:
        print("  AG reconstruction from derived triples: FAILED")
        return
    print(f"  AG(2,3) reconstructed: 12 lines OK")
    lines = set(ag)
    lines_thru = [l for l in lines if q0 in l]
    print(f"  lines through {q0}: {sorted(lines_thru)}")
    # pentad complements = line u point ?
    full = set(range(9))
    good, bad = [], []
    for p in pentads:
        comp = tuple(sorted(full - set(p)))
        hit = None
        for l in lines:
            rest = set(comp) - set(l)
            if set(l) <= set(comp) and len(rest) == 1:
                hit = (l, rest.pop())
                break
        (good if hit else bad).append((comp, hit))
    print(f"  pentad-comps = line+point: {len(good)}/{len(pentads)}")
    for comp, hit in bad[:6]:
        print(f"    NOT: comp {comp}")
    # extra quads = ? (line u point complements have size 5... quads
    # not through q0: are they line+point? no, size 4: maybe line u pt
    # is for comps only; check others = lines u {x}? size 4 = line+pt!)
    for b in others:
        hit = None
        for l in lines:
            if set(l) <= set(b):
                hit = (l, (set(b) - set(l)).pop())
                break
        print(f"  extra quad {b}: line+pt {hit}")
    # fills = lines through q0?
    if fills:
        f_in = [t for t in fills if tuple(sorted(t)) in lines]
        f_thru = [t for t in f_in if q0 in t]
        print(f"  fills: {len(fills)}, are lines: {len(f_in)}, "
              f"through {q0}: {len(f_thru)}; fills={sorted(set(fills))}")


def analyze_10x15(ws):
    w = ws["T2_10x15_81_cadical_witness.json"]
    lay = C.layers(10, w["blocks"])
    hexads = lay[6]
    pentads = lay[5]
    print("=== 10x15")
    # hexad comps = {0} u T_i
    full = set(range(10))
    comps = [tuple(sorted(full - set(b))) for b in hexads]
    print("  hexad comps:", comps)
    assert all(0 in c for c in comps)
    T = [tuple(sorted(set(c) - {0})) for c in comps]
    print("  T (on 1..9):", T)
    # relabel 1..9 -> 0..8
    Tr = [tuple(x - 1 for x in t) for t in T]
    ag = reconstruct_ag_from_lines(9, Tr)
    print("  AG reconstruction from hexad-comp triples:",
          "OK" if ag else "FAILED")
    if not ag:
        return
    lines9 = set(tuple(x + 1 for x in l) for l in ag)
    # pentads: cone at 0 + derived quads on 1..9: line+pt?
    der = [tuple(sorted(set(p) - {0})) for p in pentads]
    good = 0
    for d in der:
        for l in lines9:
            if set(l) <= set(d):
                good += 1
                break
        else:
            print(f"    derived quad {d}: NOT line+pt")
    print(f"  derived quads = line+point: {good}/{len(der)}")
    # which lines: through which points; parallel classes of T?
    print("  T classes disjointness:",
          [sorted(set(a) & set(b)) for a, b in combinations(T, 2)][:6])


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    ws = {w["file"]: w for w in C.load_banks()}
    for f in ["w_9x22.json", "w_9x25.json", "w_9x26.json", "w_9x23.json",
              "w_9x24.json", "w_9x27.json", "w_9x28.json", "w_9x29.json",
              "w_9x30.json", "w_9x31.json", "w_9x32.json", "w_9x34.json",
              "w_9x36.json", "w_9x39.json", "w_9x40.json"]:
        try:
            analyze_m9_ag(f, ws)
        except Exception as e:
            print(f"=== {f}: error {e}")
    analyze_10x15(ws)

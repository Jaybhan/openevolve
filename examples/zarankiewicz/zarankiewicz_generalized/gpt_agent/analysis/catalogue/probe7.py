"""Last decode round: MK recognizers, general shadow completion,
w_9x39 near-max test, max10 vs MK."""
import sys
from collections import Counter
from itertools import combinations

import catlib as C
import gens as G
from audit import leave_of


def mk_reference():
    """Moebius-Kantor (8_3) configuration: AG(2,3) lines avoiding one
    point, on the remaining 8 points relabeled 0..7."""
    lines = G.ag23_lines()
    rest = [p for p in range(9) if p != 8]
    relab = {p: i for i, p in enumerate(rest)}
    return [tuple(sorted(relab[x] for x in l)) for l in lines
            if 8 not in l]


def is_mk(triples):
    if len(triples) != 8:
        return False
    pts = sorted(set(x for t in triples for x in t))
    if len(pts) != 8:
        return False
    dg = Counter(x for t in triples for x in t)
    if any(dg[p] != 3 for p in pts):
        return False
    pd = C.pair_deg(triples, 0)
    return all(v <= 1 for v in pd.values())


def shadow_completion_general(m, quads, max_add=6):
    """Find multiset A of quads with shadow-sum == leave(quads)
    exactly (so quads+A is a perfect 2-cover = 3-(m,4,2))."""
    lv = leave_of(m, quads)
    need = Counter(lv)
    L = sum(need.values())
    if L % 4:
        return None
    k = L // 4
    if k > max_add:
        return None
    sols = []

    def bt(cur):
        if sols:
            return
        rem = [t for t, c in need.items() if c > 0]
        if not rem:
            sols.append(list(cur))
            return
        if len(cur) == k:
            return
        t = min(rem)
        for x in range(m):
            if x in t:
                continue
            q = tuple(sorted(set(t) | {x}))
            sh = list(combinations(q, 3))
            if all(need[s] >= 1 for s in sh):
                for s in sh:
                    need[s] -= 1
                cur.append(q)
                bt(cur)
                cur.pop()
                for s in sh:
                    need[s] += 1

    bt([])
    return sols[0] if sols else None


def main():
    ws = {w["file"]: w for w in C.load_banks()}
    mk = mk_reference()
    print("MK reference valid:", is_mk(mk))

    print("\n== m=8 pentad-comp systems vs MK / max10")
    max10 = [tuple(sorted(set(range(8)) - set(b)))
             for b in C.layers(8, ws["w_8x10.json"]["blocks"])[5]]
    print("  max10 comp system:", sorted(max10))
    r = C.find_embedding(8, mk, 8, max10)
    print("  MK sub max10:", "YES" if r and r != "TIMEOUT" else r)
    for f in ["w_8x13", "w_8x14", "w_8x15"]:
        comps = C.complements(8, C.layers(8, ws[f + ".json"]["blocks"])[5])
        print(f"  {f}: is MK: {is_mk(comps)}", end="")
        r = C.find_embedding(8, comps, 8, mk)
        print(f"; sub-MK: {'YES' if r and r != 'TIMEOUT' else r}")

    print("\n== general shadow completions")
    for f in ["w_8x25", "w_8x26", "w_8x27"]:
        w = ws[f + ".json"]
        quads = C.layers(8, w["blocks"])[4]
        r = shadow_completion_general(8, quads)
        print(f"  {f}: completes to 3-(8,4,2) with "
              f"{len(r) if r else 'NO'} quads")

    print("\n== w_9x39: near-max?")
    w = ws["w_9x39.json"]
    quads = C.layers(9, w["blocks"])[4]
    lv = leave_of(9, quads)
    # try adding one quad to reach a 40-packing with hub leave
    best = None
    for q in combinations(range(9), 4):
        sh = list(combinations(q, 3))
        if all(lv.get(s, 0) >= 1 for s in sh):
            lv2 = Counter(lv)
            for s in sh:
                lv2[s] -= 1
            lv2 = +lv2
            tris = sorted(lv2)
            common = (set.intersection(*(set(t) for t in tris))
                      if tris else set())
            if sum(lv2.values()) == 8 and common \
                    and all(v == 2 for v in lv2.values()):
                best = (q, sorted(common))
                break
    print(f"  add {best[0] if best else 'NONE'} -> 40-max with hub "
          f"{best[1] if best else ''}")

    print("\n== m=9 midband: val vs frontier (from CSV)")
    import csv
    F9 = {}
    with open("../deepband/frontier_m9_final.csv") as fh:
        for row in csv.DictReader(fh):
            try:
                F9[int(row["c"])] = int(row["F"])
            except ValueError:
                F9[int(row["c"])] = None
    F8 = {}
    with open("../deepband/frontier_m8.csv") as fh:
        for row in csv.DictReader(fh):
            F8[int(row["c"])] = int(row["F"])
    for f, w in sorted(ws.items()):
        if not f.startswith(("w_8x", "w_9x")):
            continue
        m = w["m"]
        heavy = [b for b in w["blocks"] if len(b) >= 4]
        c = len(heavy)
        val = sum(len(b) - 3 for b in heavy)
        F = (F8 if m == 8 else F9).get(c)
        tag = ("=F" if F == val else f"F={F} val={val}")
        print(f"  {f}: c={c} val={val} {tag}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()

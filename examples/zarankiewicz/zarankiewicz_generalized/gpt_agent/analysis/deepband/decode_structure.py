"""Decode the geometry of frontier configs via the complement dictionary:
a weight-w block on m points is the complement of an (m-w)-set. Prints, for
chosen frontier points / witnesses: the complement system of each weight
class, pairwise-intersection distributions, common points, and quick
recognitions (Fano-ness for 7-point triple systems, matchings, pencils).
"""
import json
import os
import sys
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))


def decode(m, blocks, label):
    heavy = [tuple(sorted(b)) for b in blocks if len(b) >= 4]
    print(f"--- {label}  (m={m}, cols={len(heavy)}, "
          f"val={sum(len(b)-3 for b in heavy)})")
    byw = {}
    for b in heavy:
        byw.setdefault(len(b), []).append(b)
    allpts = set(range(m))
    for w in sorted(byw, reverse=True):
        bl = byw[w]
        comps = [tuple(sorted(allpts - set(b))) for b in bl]
        print(f"  weight {w} x{len(bl)}: complements ({m - w}-sets): {comps}")
        if len(bl) >= 2:
            inters = Counter(len(set(a) & set(b))
                             for a, b in combinations(bl, 2))
            print(f"    pairwise |block cap block|: {dict(sorted(inters.items()))}")
            cinters = Counter(len(set(a) & set(b))
                              for a, b in combinations(comps, 2))
            print(f"    pairwise |comp cap comp|:   {dict(sorted(cinters.items()))}")
        common = set.intersection(*(set(b) for b in bl)) if bl else set()
        if common:
            print(f"    common points of the class: {sorted(common)}")
        # pencil test: blocks pairwise sharing a common (w-1)-core
        if len(bl) >= 2 and len(set.intersection(*(set(b) for b in bl))) >= w - 1:
            print("    PENCIL: all blocks share a common core")
        # Fano test for 7 triples on 7 points
        if m - w == 3 and len(comps) == 7:
            pts = sorted(set(p for cmp_ in comps for p in cmp_))
            if len(pts) == 7:
                paircov = Counter()
                for cmp_ in comps:
                    for p in combinations(cmp_, 2):
                        paircov[p] += 1
                if all(v == 1 for v in paircov.values()) and \
                        len(paircov) == 21:
                    print("    == FANO PLANE (every pair once) ==")
    # cross-class: pentad-hexad incidences
    print()


def main():
    picks = sys.argv[1:] or ["w_8x8", "w_8x12", "w_8x17", "w_9x9", "w_9x12",
                             "w_9x14", "w_10x21", "w_11x22"]
    for p in picks:
        if p.startswith("w_"):
            path = os.path.join(BASE, "analysis", "witnesses", p + ".json")
            with open(path) as f:
                wjs = json.load(f)
            decode(wjs["m"], wjs["blocks"], f"witness {p} (n={wjs['n']}, "
                                            f"edges={wjs['edges']})")
        elif p.startswith("f"):
            # e.g. f8c19 -> frontier point m=8, c=19
            m, c = p[1:].split("c")
            with open(os.path.join(HERE, f"frontier_points_m{m}.json")) as f:
                pts = json.load(f)
            r = pts[c]
            decode(int(m), r["blocks"], f"frontier m={m} c={c} "
                                        f"val={r['val']} [{r['status']}]")


if __name__ == "__main__":
    main()

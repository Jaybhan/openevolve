"""First pass: verify every witness independently and print layer
invariants, to drive the recognizer design."""
import sys
from collections import Counter

import catlib as C


def main():
    ws = C.load_banks()
    print(f"{len(ws)} witness files loaded")
    bad = 0
    for w in ws:
        v = C.verify(w["m"], w["blocks"], w.get("n"), w.get("edges"))
        w["v"] = v
        if not v["legal"] or not v.get("edges_match", True):
            bad += 1
            print("ILLEGAL/MISMATCH:", w["bank"], w["file"], v)
    print(f"legality: {len(ws)-bad}/{len(ws)} pass")
    print()
    for w in ws:
        m = w["m"]
        lay = C.layers(m, w["blocks"])
        prof = " ".join(f"{w_}^{len(bl)}" for w_, bl in
                        sorted(lay.items(), reverse=True))
        print(f"[{w['bank']}] {w['file']}: m={m} n={w['n']} "
              f"E={w['edges']} profile {prof}")
        for wt in sorted(lay, reverse=True):
            if wt <= 3:
                continue
            bl = lay[wt]
            j = m - wt
            if 1 <= j <= 3 and len(bl) >= 1:
                comps = C.complements(m, bl)
                cm = Counter(comps)
                s = " ".join(("%s%s" % ("".join(map(str, c)),
                                        "x2" if cm[c] == 2 else ""))
                             for c in sorted(set(comps)))
                print(f"    w={wt} (comp {j}-sets): {s}")
            elif len(bl) <= 60:
                dd = tuple(sorted(C.point_deg(bl, m)))
                it = dict(sorted(C.inter_dist(bl).items()))
                print(f"    w={wt} x{len(bl)}: deg={dd} inter={it}")
        if lay.get(3):
            cm = Counter(lay[3])
            s = " ".join(("%s%s" % ("".join(map(str, c)),
                                    f"x{cm[c]}" if cm[c] > 1 else ""))
                         for c in sorted(set(lay[3])))
            print(f"    fills: {s}")
        if lay.get(2) or lay.get(1):
            print(f"    pads: {len(lay.get(2, [])) + len(lay.get(1, []))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()

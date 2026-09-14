"""Targeted probes: (a) SQS8/SQS10 sub-multiset tests, (b) m=9 40-quad
packing structure, (c) band11 vs design_prover 80-packing, (d) misc
layer decodes."""
import sys
from collections import Counter
from itertools import combinations

import catlib as C
import gens as G
from audit import leave_of, is_doubled_pentagon, is_doubled_parallel_class


def wload(name):
    ws = {w["file"]: w for w in C.load_banks()}
    return ws[name]


def main():
    ws = {w["file"]: w for w in C.load_banks()}

    print("== sanity: SQS(8) =", len(G.sqs8()), "blocks; SQS(10) =",
          len(G.sqs10()), "blocks")
    s8 = G.sqs8()
    cov8 = Counter()
    for b in s8:
        for t in combinations(b, 3):
            cov8[t] += 1
    print("   SQS8 triple coverage:", set(cov8.values()),
          "count", len(cov8))
    s10 = G.sqs10()
    cov10 = Counter()
    for b in s10:
        for t in combinations(b, 3):
            cov10[t] += 1
    print("   SQS10 triple coverage:", set(cov10.values()),
          "count", len(cov10))

    print("\n== (a) m=8 quad layers vs 2xSQS(8)")
    d8 = [b for b in s8 for _ in range(2)]
    for f in ["w_8x13", "w_8x15", "w_8x16", "w_8x18", "w_8x19", "w_8x20",
              "w_8x21", "w_8x22", "w_8x23", "w_8x24", "w_8x25", "w_8x26",
              "w_8x27", "w_8x11", "w_8x12", "w_8x14", "w_8x17"]:
        w = ws[f + ".json"]
        lay = C.layers(8, w["blocks"]).get(4, [])
        if not lay:
            continue
        r = C.find_embedding(8, lay, 8, d8)
        print(f"   {f}: {len(lay)} quads sub-2xSQS8: "
              f"{'YES' if r and r != 'TIMEOUT' else r}")

    print("\n== (b) m=9 40-quad layers")
    for f in ["w_9x40", "w_9x42", "w_9x43"]:
        w = ws[f + ".json"]
        lay = C.layers(9, w["blocks"])[4]
        lv = leave_of(9, lay)
        print(f"   {f}: leave weight {sum(lv.values())}: "
              f"{sorted(lv.items())}")
        dbl = [b for b, c in Counter(lay).items() if c == 2]
        print(f"      doubled quads: {len(dbl)}; aut order: ", end="")
        a = C.aut_order(9, lay)
        print(a)
    # pairwise iso of the 40-layers
    files40 = ["w_9x40", "w_9x41", "w_9x42", "w_9x43", "w_9x44",
               "w_9x45", "w_9x46", "w_9x47", "w_9x39"]
    lays = {}
    for f in files40:
        w = ws[f + ".json"]
        lays[f] = C.layers(9, w["blocks"])[4]
    base = "w_9x40"
    for f in files40[1:]:
        if len(lays[f]) != len(lays[base]):
            print(f"   {f}: size {len(lays[f])} vs 40")
            continue
        r = C.find_embedding(9, lays[f], 9, lays[base], require_iso=True)
        print(f"   {f} iso to {base}: "
              f"{'YES' if r and r != 'TIMEOUT' else r}")

    print("\n== (c) band11 layers vs t33_m11_b80")
    ref = ws["t33_m11_b80_hole.json"]["blocks"]
    for f in ["band_11x82_witness", "band_11x85_witness",
              "band_11x89_witness"]:
        w = ws[f + ".json"]
        lay = C.layers(11, w["blocks"])[4]
        eq = sorted(lay) == sorted(ref)
        print(f"   {f}: 80-layer EQUAL to design_prover packing: {eq}")
        if not eq:
            r = C.find_embedding(11, lay, 11, ref, require_iso=True)
            print(f"      iso: {'YES' if r and r != 'TIMEOUT' else r}")
        # fills inside packing leave?
        lv = leave_of(11, lay)
        fills = Counter(C.layers(11, w["blocks"]).get(3, []))
        ok = all(lv.get(t, 0) >= c for t, c in fills.items())
        print(f"      fills within leave: {ok}; leave: {sorted(lv)}")
        print(f"      leave is doubled pentagon on: "
              f"{is_doubled_pentagon(lv)}")

    print("\n== w_7x24 quad layer vs t33_m7_b15")
    a = C.layers(7, ws["w_7x24.json"]["blocks"])[4]
    b = ws["t33_m7_b15_hole.json"]["blocks"]
    r = C.find_embedding(7, a, 7, b, require_iso=True)
    print("   iso:", "YES" if r and r != "TIMEOUT" else r)
    lv = leave_of(7, a)
    print("   leave:", sorted(lv.items()), "pentagon:",
          is_doubled_pentagon(lv))
    fills = Counter(C.layers(7, ws["w_7x24.json"]["blocks"]).get(3, []))
    print("   fills within leave:",
          all(lv.get(t, 0) >= c for t, c in fills.items()))

    print("\n== (d) m=10 quad layers vs 2xSQS(10)")
    d10 = [b for b in s10 for _ in range(2)]
    for f in ["w_10x35", "w_10x36", "w_10x48", "w_10x53", "w_10x54",
              "w_10x58", "w_10x59", "w_10x47", "w_10x57", "w_10x21",
              "w_10x22"]:
        w = ws[f + ".json"]
        lay = C.layers(10, w["blocks"]).get(4, [])
        if not lay:
            continue
        r = C.find_embedding(10, lay, 10, d10, node_cap=8_000_000)
        print(f"   {f}: {len(lay)} quads sub-2xSQS10: "
              f"{'YES' if r and r not in (None, 'TIMEOUT') else r}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()

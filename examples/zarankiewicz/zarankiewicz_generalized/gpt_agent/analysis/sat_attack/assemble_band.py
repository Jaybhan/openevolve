"""Assemble (11, n) attainment witnesses for n = 82..89 from the SAT-found
T_{3,3}(11) = 80 packing.

Construction: the 80 quad-blocks cover 320 of the 330 triple-slots
(2 per triple); the leave (slots with coverage < 2) has total weight 10.
Appending n-80 weight-3 columns, each consuming one distinct leave slot,
yields an 11 x n matrix with 4*80 + 3(n-80) = 3n + 80 ones, K_{3,3}-free
by construction.  Every witness is re-verified independently.

Usage: assemble_band.py PACKING_JSON [--nmin 82] [--nmax 89]
"""
import argparse
import json
import os
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packing")
    ap.add_argument("--nmin", type=int, default=82)
    ap.add_argument("--nmax", type=int, default=89)
    args = ap.parse_args()

    with open(args.packing) as f:
        pk = json.load(f)
    v = pk["v"]
    assert v == 11 and len(pk["blocks"]) == 80
    quads = [sorted(b) for b in pk["blocks"]]

    cov = Counter()
    for b in quads:
        for t in combinations(b, 3):
            cov[t] += 1
    leave = []  # leave slots, one entry per free slot
    for t in combinations(range(v), 3):
        for _ in range(2 - cov[t]):
            leave.append(list(t))
    print(f"leave slots: {len(leave)} -> {leave}")
    assert len(leave) == 10

    for n in range(args.nmin, args.nmax + 1):
        k = n - 80
        assert k <= len(leave)
        blocks = quads + leave[:k]
        edges = sum(len(b) for b in blocks)
        expect = 3 * n + 80
        assert edges == expect, (n, edges, expect)
        out = os.path.join(HERE, "witnesses", f"band_11x{n}_witness.json")
        with open(out, "w") as f:
            json.dump({"m": 11, "n": n, "edges": edges, "blocks": blocks,
                       "source": f"sat-packing {os.path.basename(args.packing)}"
                                 f" + leave assembly"}, f)
        print(f"n={n}: {edges} ones -> {out}")


if __name__ == "__main__":
    main()

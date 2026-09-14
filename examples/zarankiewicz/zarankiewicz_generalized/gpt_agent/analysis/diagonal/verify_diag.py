"""DIAGONAL HUNTER's independent verifier. Written from scratch; shares no
code with encodings_zar.py, run_matrix.py, or sat_attack/verify_witness.py.

Primary check uses the COLUMN-triple formulation (every 3 columns share <= 2
rows), which is logically equivalent to K_{3,3}-freeness but code-disjoint
from the row-triple formulation used by the encoder and the SAT agent's
verifier.  A second pass re-checks via row triples with bitmask popcounts.

Witness JSON format: {"m", "n", "edges", "blocks": [column supports]}.

Usage: verify_diag.py FILE [FILE...] [--exact-edges E]
Exit 0 iff every file passes both formulations.
"""
import argparse
import json
import sys
from itertools import combinations


def verify(path, exact_edges=None):
    with open(path) as f:
        w = json.load(f)
    m, n, blocks = w["m"], w["n"], w["blocks"]
    if len(blocks) != n:
        return False, f"{path}: {len(blocks)} columns, header says n={n}"
    masks = []
    total = 0
    for b in blocks:
        if len(set(b)) != len(b) or any(not (0 <= r < m) for r in b):
            return False, f"{path}: bad column {b}"
        mk = 0
        for r in b:
            mk |= 1 << r
        masks.append(mk)
        total += len(b)
    if "edges" in w and total != w["edges"]:
        return False, f"{path}: edges field {w['edges']} != recount {total}"
    if exact_edges is not None and total != exact_edges:
        return False, f"{path}: edges {total} != required {exact_edges}"
    # formulation 1: every 3 distinct columns intersect in <= 2 rows
    for i, j, k in combinations(range(n), 3):
        inter = masks[i] & masks[j] & masks[k]
        if bin(inter).count("1") >= 3:
            return False, (f"{path}: columns {i},{j},{k} share rows "
                           f"{[r for r in range(m) if inter >> r & 1]}")
    # formulation 2: every 3 rows lie in <= 2 columns (bitmask over columns)
    rowmask = [0] * m
    for c, mk in enumerate(masks):
        for r in range(m):
            if mk >> r & 1:
                rowmask[r] |= 1 << c
    for a, b, c3 in combinations(range(m), 3):
        common = rowmask[a] & rowmask[b] & rowmask[c3]
        if bin(common).count("1") >= 3:
            return False, f"{path}: rows {a},{b},{c3} in >=3 columns"
    return True, f"{path}: OK m={m} n={n} edges={total} (both formulations)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--exact-edges", type=int, default=None)
    args = ap.parse_args()
    ok_all = True
    for p in args.files:
        ok, msg = verify(p, args.exact_edges)
        print(("PASS " if ok else "FAIL ") + msg)
        ok_all &= ok
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

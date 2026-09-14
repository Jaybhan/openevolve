"""Independent witness verifier -- deliberately shares no code with the
encoders.  Recounts everything from the raw JSON.

Matrix witness  {"m", "n", "edges", "blocks": [column supports]}:
  * exactly n blocks, each a set of distinct rows in range(m)
  * sum of sizes == edges
  * every 3-subset of rows appears in <= 2 columns  (K_{3,3}-freeness)

Packing witness {"v", "count", "blocks": [4-subsets, listed with multiplicity]}:
  * count == len(blocks), each block 4 distinct points in range(v)
  * no block appears more than twice
  * every 3-subset of points covered <= 2 times

Usage: verify_witness.py FILE [FILE...] [--min-edges E] [--min-count C]
Exit 0 iff all files verify.
"""
import argparse
import json
import sys
from collections import Counter
from itertools import combinations


def verify_matrix(w, min_edges=None):
    m, n = w["m"], w["n"]
    blocks = w["blocks"]
    if len(blocks) != n:
        return False, f"expected {n} columns, got {len(blocks)}"
    total = 0
    tri = Counter()
    for b in blocks:
        if len(set(b)) != len(b):
            return False, f"repeated row inside column {b}"
        if any(not (0 <= r < m) for r in b):
            return False, f"row out of range in {b}"
        total += len(b)
        for t in combinations(sorted(b), 3):
            tri[t] += 1
            if tri[t] > 2:
                return False, f"row triple {t} in >2 columns"
    if total != w.get("edges", total):
        return False, f"edge count mismatch: {total} vs {w['edges']}"
    if min_edges is not None and total < min_edges:
        return False, f"edges {total} < required {min_edges}"
    return True, f"OK m={m} n={n} edges={total}"


def verify_packing(w, min_count=None):
    v = w["v"]
    blocks = [tuple(sorted(b)) for b in w["blocks"]]
    if "count" in w and len(blocks) != w["count"]:
        return False, f"count mismatch {len(blocks)} vs {w['count']}"
    mult = Counter(blocks)
    tri = Counter()
    for b, k in mult.items():
        if len(b) != 4 or len(set(b)) != 4:
            return False, f"not a 4-subset: {b}"
        if any(not (0 <= p < v) for p in b):
            return False, f"point out of range in {b}"
        if k > 2:
            return False, f"block {b} used {k} > 2 times"
        for t in combinations(b, 3):
            tri[t] += k
            if tri[t] > 2:
                return False, f"triple {t} covered >2 times"
    if min_count is not None and len(blocks) < min_count:
        return False, f"count {len(blocks)} < required {min_count}"
    return True, f"OK v={v} count={len(blocks)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--min-edges", type=int, default=None)
    ap.add_argument("--min-count", type=int, default=None)
    args = ap.parse_args()
    ok_all = True
    for path in args.files:
        with open(path) as f:
            w = json.load(f)
        if "v" in w:
            ok, msg = verify_packing(w, args.min_count)
        else:
            ok, msg = verify_matrix(w, args.min_edges)
        print(f"{'PASS' if ok else 'FAIL'} {path}: {msg}")
        ok_all &= ok
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()

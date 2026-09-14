#!/usr/bin/env python3
"""Independent witness verifier for 2-fold quadruple packings (T_{3,3}(m)).

Reads a witness JSON {"m":..., "target":..., "blocks":[[a,b,c,d],...]} and
re-checks EVERYTHING from first principles with its own logic:
  1. block count == target; every block is a 4-subset of [0,m)
  2. no quad used more than twice
  3. every 3-subset of [0,m) lies in at most 2 blocks (counted with mult.)
  4. reports leave weight, leave structure, per-point and per-pair congruence
     compliance (l_x = 0 mod 3, l_xy even -- must hold automatically),
  5. reports the automorphism check: is the block multiset invariant under
     sigma = (0 1 2 3 4)(5 6 7 8 9)(10 11 12 13 14)... (optional, informative)
Exit code 0 iff 1-3 hold.
"""
import sys, json, itertools
from collections import Counter
from math import comb

def main(path):
    with open(path) as f:
        w = json.load(f)
    m, target = w["m"], w["target"]
    blocks = [tuple(sorted(b)) for b in w["blocks"]]
    ok = True

    # 1. counts and well-formedness
    if len(blocks) != target:
        print("FAIL: block count %d != target %d" % (len(blocks), target)); ok = False
    for b in blocks:
        if len(b) != 4 or len(set(b)) != 4 or not all(0 <= x < m for x in b):
            print("FAIL: malformed block", b); ok = False

    # 2. quad multiplicities
    qmult = Counter(blocks)
    bad = {q: v for q, v in qmult.items() if v > 2}
    if bad:
        print("FAIL: quads used >2:", bad); ok = False

    # 3. triple coverage
    cov = Counter()
    for b in blocks:
        for T in itertools.combinations(b, 3):
            cov[T] += 1
    over = {T: v for T, v in cov.items() if v > 2}
    if over:
        print("FAIL: triples covered >2:", dict(list(over.items())[:5])); ok = False

    # 4. leave analysis
    ntrip = comb(m, 3)
    lw = 2 * ntrip - sum(cov.values())
    leave = []
    for T in itertools.combinations(range(m), 3):
        l = 2 - cov.get(T, 0)
        if l:
            leave.append((T, l))
    lx = Counter(); lxy = Counter()
    for T, l in leave:
        for x in T: lx[x] += l
        for pr in itertools.combinations(T, 2): lxy[pr] += l
    # correct residues: l_x = 2C(m-1,2) (mod 3) for EVERY point (including
    # untouched ones), l_xy = 0 (mod 2); class m gives residue 0, 3|m gives 2.
    pt_res = (2 * comb(m - 1, 2)) % 3
    pt_ok = all(lx.get(x, 0) % 3 == pt_res for x in range(m))
    pr_ok = all(v % 2 == 0 for v in lxy.values())

    print("m=%d  blocks=%d  distinct quads=%d  doubled quads=%d" %
          (m, len(blocks), len(qmult), sum(1 for v in qmult.values() if v == 2)))
    print("triple coverage: max=%d  leave weight=%d (= 2C(m,3)-4b = %d)" %
          (max(cov.values()), lw, 2 * ntrip - 4 * len(blocks)))
    print("leave (%d triples): %s" % (len(leave), leave if len(leave) <= 12 else leave[:12]))
    print("leave congruences: point (l_x = %d mod 3) %s | pair (even) %s" %
          (pt_res, "OK" if pt_ok else "VIOLATED?!", "OK" if pr_ok else "VIOLATED?!"))

    # 5. sigma invariance (informative)
    c5 = w.get("c5")
    if c5:
        def sig(x):
            if x < 5 * c5:
                b0 = x // 5
                return b0 * 5 + (x - b0 * 5 + 1) % 5
            return x
        mapped = Counter(tuple(sorted(sig(x) for x in b)) for b in blocks)
        print("sigma-invariant block multiset:", mapped == qmult)

    print("VERDICT:", "VALID" if ok else "INVALID")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

#!/usr/bin/env python3
"""Zero-SAT closure check on the column side: enumerate every column-sum partition
(n parts in [0, m] summing to w, non-increasing) and test whether Argument A (colBudget)
or the DGH (v = s-1) column prune kills it.  If every partition dies, z(m,n;s,t) < w
follows from the two Lean-verified prunes alone (the row side is not even needed).

    python experiments/E13_dgh4/frontier.py [--cells "13,17,3,3,117;13,18,3,3,122;11,21,3,3,117"]
"""
import argparse, sys, os, time
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgh import dgh_col_kill, dgh_col_kill_ks  # noqa: E402

DEFAULT = "11,21,3,3,117;13,17,3,3,117;13,18,3,3,122;15,17,3,3,133;15,17,3,3,132;10,22,3,3,112;10,23,3,3,115;9,12,3,3,64"


def partitions(total, parts, hi):
    """Non-increasing tuples of `parts` integers in [0, hi] with the given sum."""
    def gen(rem, left, mx, pre):
        if left == 0:
            if rem == 0:
                yield tuple(pre)
            return
        top = min(mx, rem)
        for x in range(top, -1, -1):
            if x * left < rem:
                break
            yield from gen(rem - x, left - 1, x, pre + [x])
    yield from gen(total, parts, hi, [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=DEFAULT)
    a = ap.parse_args()
    for chunk in a.cells.split(";"):
        m, n, s, t, w = (int(x) for x in chunk.split(","))
        t0 = time.time()
        budget = (t - 1) * comb(m, s)
        tot = passA = survive = 0
        ks = {}
        examples = []
        for cols in partitions(w, n, m):
            tot += 1
            if sum(comb(c, s) for c in cols) > budget:
                continue
            passA += 1
            kk = dgh_col_kill_ks(m, s, t, cols)
            if kk:
                for k in kk:
                    ks[k] = ks.get(k, 0) + 1
            else:
                survive += 1
                if len(examples) < 5:
                    examples.append(cols)
        print(f"z({m},{n};{s},{t}) w={w}: column partitions={tot}, pass argA={passA}, survive argA+DGH={survive} "
              f"[{time.time()-t0:.1f}s] DGH k-hist={dict(sorted(ks.items()))}" + (f" survivors e.g. {examples}" if examples else "  => closed with zero SAT (column side)"))


if __name__ == "__main__":
    main()

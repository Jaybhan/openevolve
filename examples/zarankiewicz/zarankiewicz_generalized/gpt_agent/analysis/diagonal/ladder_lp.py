"""Deletion-averaging UB ladder over the cell grid.

Facts used (all elementary, PROVEN):
  * row deletion:  (m-1) z(m,n) <= m z(m-1,n)
  * col deletion:  (n-1) z(m,n) <= n z(m,n-1)
  * both:          (m-1)(n-1) z(m,n) <= m n z(m-1,n-1)
    [sum the minor bound over all rows / columns / cells; the minor of a
     K33-free matrix is K33-free]
  * waterfill / slot budget: z <= max sum w_c s.t. sum C(w_c,3) <= 2C(m,3),
    w_c <= m, n columns (and the transpose), level-fill exact for the relaxed
    integer program by convexity of C(.,3).
  * known exact values from data/exact_table.csv (pins), plus any overrides
    passed as PIN entries (new SAT/averaging results).

Iterate min-updates to fixpoint; print UB grid for 12<=m<=n<=20.
"""
import csv
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(HERE, "..", "..", "data", "exact_table.csv")

M = 21


def waterfill(m, n):
    B = 2 * comb(m, 3)
    w = [2] * n
    spend = 0
    for lev in range(3, m + 1):
        cost = comb(lev - 1, 2)
        for i in range(n):
            if spend + cost <= B:
                w[i] += 1
                spend += cost
            else:
                break
        else:
            continue
        break
    return sum(w)


def main():
    exact = {}
    with open(TAB) as f:
        for row in csv.DictReader(f):
            m, n, z = int(row["m"]), int(row["n"]), int(row["z"])
            exact[(m, n)] = z
            exact[(n, m)] = z
    # pins: results established in this workspace, from pins.json
    # format {"m,n": [value, "source"]}; value = proven EXACT z.
    # UB-only pins go in ub_pins.json with the same format.
    import json
    pinf = os.path.join(HERE, "pins.json")
    if os.path.exists(pinf):
        with open(pinf) as f:
            for k, (v, src) in json.load(f).items():
                m, n = map(int, k.split(","))
                exact[(m, n)] = v
                exact[(n, m)] = v
    ub_extra = {}
    ubf = os.path.join(HERE, "ub_pins.json")
    if os.path.exists(ubf):
        with open(ubf) as f:
            for k, (v, src) in json.load(f).items():
                m, n = map(int, k.split(","))
                ub_extra[(m, n)] = v
                ub_extra[(n, m)] = v

    UB = {}
    for m in range(3, M):
        for n in range(3, M):
            if (m, n) in exact:
                UB[(m, n)] = exact[(m, n)]
            else:
                UB[(m, n)] = min(waterfill(m, n), waterfill(n, m),
                                 ub_extra.get((m, n), 10 ** 9))
    changed = True
    while changed:
        changed = False
        for m in range(4, M):
            for n in range(4, M):
                if (m, n) in exact:
                    continue
                cands = [
                    (m * UB[(m - 1, n)]) // (m - 1),
                    (n * UB[(m, n - 1)]) // (n - 1),
                    (m * n * UB[(m - 1, n - 1)]) // ((m - 1) * (n - 1)),
                ]
                v = min(cands)
                if v < UB[(m, n)]:
                    UB[(m, n)] = v
                    changed = True
    print("UB grid (deletion ladder + waterfill + exact pins);"
          " * = exact pin")
    hdr = "m\\n " + "".join(f"{n:>6}" for n in range(12, M))
    print(hdr)
    for m in range(12, M):
        row = f"{m:>4} "
        for n in range(12, M):
            v = UB[(m, n)]
            mark = "*" if (m, n) in exact else " "
            row += f"{v:>5}{mark}"
        print(row)


if __name__ == "__main__":
    main()

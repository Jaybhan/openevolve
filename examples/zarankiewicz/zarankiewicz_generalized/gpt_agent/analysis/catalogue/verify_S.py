"""Machine verification of the wide-region structure theorems S1/S2
against every witness in the banks.

S1 identity check (all witnesses): with q = floor((B-n)/3),
r = (B-n) mod 3, def = sum_pads(3-w), the chain
    3*def - n_p + ex + X2 + (B - slots) = 3*(3n + q - E) + r
holds identically (it is an identity given E = 3n + val - def and
slots = 4 val + n3 + X2); and E <= 3n + q always (Roman).

S2/S3 canonical-form check (T-branch cells): for witnesses with
T <= n <= B - 3T: all columns weight in {3,4}; #quads = T (or the
witness is a packing at n = T); quad-layer leave weight = B - 4T;
fills inside the leave.
"""
import sys
from collections import Counter
from itertools import combinations

import catlib as C
from audit import leave_of

TVALS = {7: 15, 8: 28, 9: 40, 10: 60, 11: 80, 19: 482, 23: 883,
         27: 1458}


def main():
    ws = C.load_banks()
    fails = 0
    tbranch = 0
    for w in ws:
        m, n, blocks = w["m"], w["n"], w["blocks"]
        B = m * (m - 1) * (m - 2) // 3
        E = sum(len(b) for b in blocks)
        q, r = divmod(B - n, 3) if n <= B else (None, None)
        heavy = [b for b in blocks if len(b) >= 4]
        n3 = sum(1 for b in blocks if len(b) == 3)
        pads = [b for b in blocks if len(b) <= 2]
        val = sum(len(b) - 3 for b in heavy)
        ex = sum(len(b) - 4 for b in heavy)
        X2 = sum(len(b) * (len(b) - 1) * (len(b) - 2) // 6
                 - 4 * (len(b) - 3) for b in heavy)
        slots = sum(len(b) * (len(b) - 1) * (len(b) - 2) // 6
                    for b in blocks)
        dfct = sum(3 - len(b) for b in pads)
        # identity: E = 3n + val - def
        assert E == 3 * n + val - dfct, w["file"]
        # slots identity
        assert slots == 4 * val + n3 + X2, w["file"]
        if n <= B:
            # Roman bound
            if not E <= 3 * n + q:
                print(f"ROMAN VIOLATION {w['file']}")
                fails += 1
        # T-branch canonical form
        T = TVALS.get(m)
        if T is not None and T <= n <= B - 3 * T:
            tbranch += 1
            wts = Counter(len(b) for b in blocks)
            ok = (set(wts) <= {3, 4} and wts[4] == T)
            quads = [b for b in blocks if len(b) == 4]
            lv = leave_of(m, quads)
            L = sum(lv.values())
            ok = ok and (L == B - 4 * T)
            fills = Counter(tuple(sorted(b)) for b in blocks
                            if len(b) == 3)
            ok = ok and all(lv.get(t, 0) >= c for t, c in fills.items())
            status = "OK" if ok else "FAIL"
            if not ok:
                fails += 1
            print(f"  T-branch {w['file']}: n={n} in [{T},{B-3*T}]: "
                  f"weights {dict(wts)}, quadleave {L} "
                  f"(pred {B-4*T}), fills-in-leave: {status}")
    print(f"\nS1 identities: all {len(ws)} witnesses pass; "
          f"T-branch canonical form checked on {tbranch} witnesses; "
          f"failures: {fails}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()

"""Retry the leave-IP UNKNOWN cells at 600 s/instance; rewrite rows."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leave_ip
from math import comb

CELLS = [(14,5), (14,6), (12,7), (15,7), (11,8), (12,8), (13,9)]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leave_table.csv")
rows = open(out).readlines()

def min_leave600(m, w, doubled=False):
    B = 2*comb(m,3); slots = comb(w,3); L0 = B % slots
    for k in range(14):
        L = L0 + k*slots
        if L > B: break
        st, y = leave_ip.feasible_leave(m, w, L, doubled=doubled, time_limit=600)
        if st == "FEAS": return L, "EXACT"
        if st == "UNKNOWN": return L, "UNKNOWN>="
    return None, "NONE"

for (m, w) in CELLS:
    Ls, st = min_leave600(m, w)
    Ld, std = min_leave600(m, w, doubled=True)
    B = 2*comb(m,3); slots = comb(w,3)
    U = (B - Ls)//slots if st == "EXACT" else ""
    print(f"RETRY ({m},{w}): L*={Ls} [{st}] U={U}; dbl={Ld} [{std}]", flush=True)
    for i, line in enumerate(rows):
        p = line.split(",")
        if len(p) > 2 and p[0] == str(m) and p[1] == str(w):
            rows[i] = f"{m},{w},{B},{slots},{Ls},{st},{U},{Ld},{std},{p[9] if len(p)>9 else ''},retry600\n"
    open(out, "w").writelines(rows)

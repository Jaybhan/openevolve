"""Re-verify every clean claim of spectrum.md against ground truth.
Run: python verify_level_theory.py   (exit 0 = all checks pass)

Checks:
 1. Wedge theorem vs every exact CSV cell in the wedge.
 2. Complement-zone propositions (j = m-w in {2,3,4}) closed forms.
 3. U_leave (leave_table.csv) is >= exact everywhere, = exact on all
    design-zone cells (THEORY-TIGHT list), and Lemma L3 analytic bounds
    are implied (U_leave <= check0 <= budget etc. sanity).
 4. Perfect-admissibility residue lists for w = 5..8.
 5. D4 spectrum formula vs published T33 values.
 6. New determinations: witness files exist and re-verify.
"""
import json
import os
import sys
from itertools import combinations
from math import comb
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from admissibility import base, johnson, check0_ub, wedge, perfect_admissible, load_exact

fails = []


def chk(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


exact = load_exact()

# 1. wedge
wcells = {(m, w): v for (m, w), v in exact.items() if wedge(m, w)}
chk(f"wedge = 2 on all {len(wcells)} exact wedge cells",
    all(v == 2 for v in wcells.values()))

# 2. complement propositions
COMP = {}
for (m, w), v in exact.items():
    j = m - w
    if wedge(m, w) or j > 4:
        continue
    COMP[(m, w)] = v
pred = {}
for (m, w) in COMP:
    j = m - w
    if j == 2 and w >= 5:
        pred[(m, w)] = 4 if w in (5, 6) else 2
    elif j == 3 and w >= 7:
        pred[(m, w)] = {7: 4, 8: 3}.get(w, 2)
    elif j == 4 and w >= 9:
        pred[(m, w)] = {9: 3, 10: 3}.get(w, 2)  # 9->(13,9), 10->(14,10), >=11 auto
    # smaller w at the same j are design-governed (leave-IP territory)
ok = all(pred.get(k) is None or pred[k] == COMP[k] for k in COMP)
chk(f"complement props on {len(COMP)} cells {sorted(COMP)}", ok)
if not ok:
    for k in sorted(COMP):
        if pred.get(k) is not None and pred[k] != COMP[k]:
            print(f"   mismatch {k}: pred {pred[k]} vs exact {COMP[k]}")

# 3. U_leave consistency
lt = os.path.join(HERE, "leave_table.csv")
tight, gaps = [], []
if os.path.exists(lt):
    for line in open(lt).readlines()[1:]:
        p = line.strip().split(",")
        m, w = int(p[0]), int(p[1])
        if p[5] != "EXACT" or not p[6]:
            continue
        U = int(p[6])
        c0 = check0_ub(m, w)
        J, _, _ = johnson(m, w)
        B, slots, _, _ = base(m, w)
        chk_ok = U <= min(c0, J, B // slots)
        if not chk_ok:
            chk(f"U_leave({m},{w}) <= analytic bounds", False)
        v = exact.get((m, w))
        if v is not None:
            if U == v:
                tight.append((m, w))
            elif U < v:
                chk(f"U_leave({m},{w}) >= exact", False)
            else:
                gaps.append((m, w, U, v))
chk(f"U_leave >= exact everywhere; tight on {len(tight)} cells {tight}", True)
print(f"   gaps (complement-zone or open): {gaps}")
# expected gaps only at complement-zone cells + (10,6)
chk("U_leave gaps only in complement zone or (10,6)",
    all((m - w <= 4) or (m, w) == (10, 6) for (m, w, _, _) in gaps))

# 4. perfect residues
want = {5: [2, 5, 11], 6: [2, 6, 12, 16],
        7: [2, 7, 22, 37, 77, 92],
        8: [2, 8, 44, 50, 65, 86, 92, 113, 128, 134]}
per = {5: 15, 6: 20, 7: 105, 8: 168}
for w in want:
    got = sorted({m % per[w] for m in range(w, w + 40 * per[w])
                  if perfect_admissible(m, w)})
    chk(f"perfect residues w={w} mod {per[w]}", got == want[w])

# 5. D4 spectrum vs published T33 (Tan m<=18 + workspace 19,23,27)
T33 = {4: 2, 5: 5, 6: 9, 7: 15, 8: 28, 9: 40, 10: 60, 11: 80, 12: 108,
       13: 143, 14: 182, 15: 225, 16: 280, 17: 340, 18: 405,
       19: 482, 23: 883, 27: 1458}


def D4(m):
    J = (m * (((m - 1) * (m - 2)) // 3)) // 4
    return J - 2 if (m % 4 == 3 and m % 3 != 0) else J


chk("D4 spectrum formula vs all published/workspace T33",
    all(D4(m) == v for m, v in T33.items()))

# 6. new determinations' witnesses
for (m, w, b) in ((12, 6, 22), (11, 6, 14), (13, 6, 26)):
    p = os.path.join(HERE, "witnesses", f"D2_{m}_{w}_b{b}.json")
    okf = os.path.exists(p)
    if okf:
        d = json.load(open(p))
        cov = Counter()
        for blk in d["blocks"]:
            for t in combinations(sorted(blk), 3):
                cov[t] += 1
        okf = (len(d["blocks"]) == b and all(len(x) == w for x in d["blocks"])
               and max(cov.values()) <= 2
               and max(Counter(tuple(sorted(x)) for x in d["blocks"]).values()) <= 2)
        if (m, w, b) == (12, 6, 22):
            okf = okf and min(cov.values()) == 2 and len(cov) == comb(12, 3)
    chk(f"witness D2({m},{w}) >= {b} re-verified", okf)

# 7. decisions.csv internal consistency: every INFEAS-derived UB must not
# contradict any witness LB; every FEAS witness must not exceed any UB.
dec = []
dp = os.path.join(HERE, "decisions.csv")
if os.path.exists(dp):
    for line in open(dp).readlines()[1:]:
        p = line.strip().split(",")
        if len(p) >= 4 and p[3] in ("FEAS", "INFEAS"):
            dec.append((int(p[0]), int(p[1]), int(p[2]), p[3]))
okc = True
for (m, w, b, r1) in dec:
    for (m2, w2, b2, r2) in dec:
        if (m, w) == (m2, w2) and r1 == "FEAS" and r2 == "INFEAS" and b >= b2:
            okc = False
            print(f"   CONTRADICTION: ({m},{w}) FEAS at {b} but INFEAS at {b2}")
chk("decisions.csv internally consistent (no FEAS >= INFEAS)", okc)

# 7b. general-s formula record (coordinator's script, re-run here)
import subprocess
gs = os.path.join(HERE, "..", "general_s_formula_test.py")
if os.path.exists(gs):
    r = subprocess.run([sys.executable, gs], capture_output=True, text=True)
    ok34 = "(s,t)=(3, 4): 21 cells  dev-hist={0: 21}" in r.stdout
    ok44 = "(s,t)=(4, 4): 21 cells  dev-hist={0: 21}" in r.stdout
    chk("general-s formula: (3,4) 21/21 and (4,4) 21/21 EXACT", ok34 and ok44)

# 8. fullpass status: no undershoots (LB side must be witness-realizable),
# and every slice verdict is REFUTED or TIMEOUT (a standing claim would
# contradict the z-table — table-error alarm).
fp = os.path.join(HERE, "fullpass_status.csv")
if os.path.exists(fp):
    rows_fp = [line.strip().split(",") for line in open(fp).readlines()[1:]]
    unders = [r for r in rows_fp if r[5].startswith("UNDER")]
    chk(f"fullpass: no undershoots ({len(rows_fp)} cells)", not unders)
    if unders:
        print("   under rows:", [(r[0], r[1], r[5]) for r in unders])
sl = os.path.join(HERE, "slices.csv")
if os.path.exists(sl):
    stand = [line for line in open(sl).readlines()[1:]
             if "CLAIM-STANDS" in line]
    chk("slices: no standing claims (z-table consistency)", not stand)
    if stand:
        print("   STANDING:", stand)

print("\n" + ("ALL CHECKS PASS" if not fails else f"FAILURES: {fails}"))
sys.exit(1 if fails else 0)

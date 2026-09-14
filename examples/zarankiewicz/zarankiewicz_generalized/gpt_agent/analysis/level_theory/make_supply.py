"""Build the consolidated best-knowledge level-supply table
supply_status.csv: for each (m, w), the current value or bracket for
D2(m,w,3) with source/status. Sources in precedence order:
  EXACT (level_D2.csv / decisions FEAS at UB / wedge / complement props)
  BRACKET [lb, ub]: lb = best witness (decisions FEAS below UB, greedy),
                    ub = min(U_leave, J, budget, check0).
"""
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from admissibility import base, johnson, check0_ub, wedge, load_exact


def comp_value(m, w):
    j = m - w
    if wedge(m, w):
        return 2
    if j == 2 and w >= 5:
        return 4 if w in (5, 6) else 2
    if j == 3 and w >= 7:
        return {7: 4, 8: 3}.get(w, 2)
    if j == 4 and w >= 9:
        return {9: 3, 10: 3}.get(w, 2)
    return None


def main():
    exact = load_exact()
    uleave = {}
    lt = os.path.join(HERE, "leave_table.csv")
    if os.path.exists(lt):
        for line in open(lt).readlines()[1:]:
            p = line.strip().split(",")
            if p[5] == "EXACT" and p[6]:
                uleave[(int(p[0]), int(p[1]))] = int(p[6])
    dec_lb, dec_exact = {}, {}
    dp = os.path.join(HERE, "decisions.csv")
    if os.path.exists(dp):
        for line in open(dp).readlines()[1:]:
            mm, ww, bb, res = line.strip().split(",")[:4]
            mm, ww, bb = int(mm), int(ww), int(bb)
            if res == "FEAS":
                dec_lb[(mm, ww)] = max(dec_lb.get((mm, ww), 0), bb)
            elif res == "INFEAS":
                dec_exact[(mm, ww)] = min(dec_exact.get((mm, ww), 10**9), bb - 1)
    # witness LBs from files
    wit_lb = {}
    for fn in os.listdir(os.path.join(HERE, "witnesses")):
        if fn.startswith("D2_") and fn.endswith(".json"):
            d = json.load(open(os.path.join(HERE, "witnesses", fn)))
            k = (d["m"], d["w"])
            wit_lb[k] = max(wit_lb.get(k, 0), d["count"])

    rows = []
    for w in range(5, 11):
        for m in range(w, 17):
            B, slots, c1, c2 = base(m, w)
            J, _, _ = johnson(m, w)
            ub = min(J, B // slots, check0_ub(m, w))
            if (m, w) in uleave:
                ub = min(ub, uleave[(m, w)])
            if (m, w) in dec_exact:
                ub = min(ub, dec_exact[(m, w)])
            lb = max(2, dec_lb.get((m, w), 0), wit_lb.get((m, w), 0))
            v, st, src = None, None, None
            if (m, w) in exact:
                v, st, src = exact[(m, w)], "EXACT", "milp"
            cv = comp_value(m, w)
            if cv is not None:
                if v is None:
                    v, st, src = cv, "EXACT", "proven(L1/L2)"
                else:
                    assert v == cv, (m, w, v, cv)
                    src += "+proven"
            if v is None and lb == ub:
                v, st, src = lb, "EXACT", "decision+Uleave"
            if v is None:
                rows.append((m, w, f"[{lb};{ub}]", "BRACKET",
                             f"lb:{'wit' if (m,w) in wit_lb or (m,w) in dec_lb else 'triv'};ub:best"))
            else:
                rows.append((m, w, v, st, src))
    with open(os.path.join(HERE, "supply_status.csv"), "w") as f:
        f.write("m,w,D2,status,source\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    # markdown matrix for the doc
    print("| m\\w | 5 | 6 | 7 | 8 | 9 | 10 |")
    print("|---|---|---|---|---|---|---|")
    for m in range(5, 17):
        cells = []
        for w in range(5, 11):
            e = [r for r in rows if r[0] == m and r[1] == w]
            cells.append(str(e[0][2]) if e else "")
        print(f"| {m} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()

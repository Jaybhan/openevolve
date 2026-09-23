"""E10: what does the 'difficulty' of a case mean, and which cheap proxies predict it?

For every pure-mode TRAIN table (w = z+1, all cases UNSAT in truth), re-solve the
cases that were still open at the 20000-conflict cap with a much larger budget
to learn their true refutation cost, update the cached table (probe status /
conflicts / budget_cap), then measure how well cheap proxies rank cases by cost:
  - fixed_by_propagation (root unit propagation), log2_volume, conflicts at small
    caps (200, 2000), nvars/nclauses, and the reference Argument-D kill flag.
Writes experiments/E10_difficulty/results.json and a markdown summary.
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from zar_ub import Instance, exact_value
from zar_ub.casetable import load_table
from zar_ub.encoding import encode_case
from zar_ub.solve import solve_cnf

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [(9, 9), (9, 10), (10, 10), (10, 11), (11, 11), (11, 12), (12, 12)]
BIG_CAP, BIG_TIME = 5_000_000, 240.0


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]: j += 1
            for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys); n = len(xs)
    if n < 3: return float("nan")
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


results = {"cells": {}, "correlations": {}}
allrows = []
for (m, n) in CELLS:
    z = exact_value(m, n, 3, 3)
    inst = Instance(m, n, 3, 3, z + 1)
    tab = load_table(inst, use_table=False)
    if tab is None:
        print("missing", inst.tag); continue
    unknown = [r for r in tab.records if r.probe and r.probe["status"] == "unknown"]
    print(f"[{inst.tag}] {len(tab.records)} cases, {len(unknown)} open at cap {tab.conf_cap}", flush=True)
    t0 = time.time()
    for k, r in enumerate(unknown):
        cnf = encode_case(inst, r.rows, r.cols)
        res = solve_cnf(cnf, inst, conf_budget=BIG_CAP, time_limit=BIG_TIME)
        r.probe["status"] = res.status
        r.probe["conflicts"] = res.conflicts
        r.probe["seconds"] = res.seconds
        r.probe["budget_cap"] = BIG_CAP
        print(f"  {k+1}/{len(unknown)} rows={r.rows} cols={r.cols} -> {res.status} conflicts={res.conflicts} {res.seconds:.1f}s", flush=True)
        if res.status == "sat":
            print("  !!! SAT at w=z+1 contradicts the exact table — investigate", flush=True)
    tab.conf_cap = BIG_CAP
    tab.save()
    # cheap proxies per case (small caps re-measured)
    rows = []
    for r in tab.records:
        cnf = encode_case(inst, r.rows, r.cols)
        c200 = solve_cnf(cnf, inst, conf_budget=200).conflicts
        c2000 = solve_cnf(cnf, inst, conf_budget=2000).conflicts
        rows.append({"cell": [m, n], "rows": r.rows, "cols": r.cols, "true_conflicts": r.probe["conflicts"],
                     "status": r.probe["status"], "prop": r.probe["fixed_by_propagation"], "log2_volume": r.probe["log2_volume"],
                     "nclauses": r.probe["nclauses"], "c200": c200, "c2000": c2000, "argD_ref_kill": bool(r.baseline)})
    allrows.extend(rows)
    results["cells"][inst.tag] = {"n": len(rows), "reopened": len(unknown), "seconds": round(time.time() - t0, 1),
                                  "max_conflicts": max(x["true_conflicts"] for x in rows), "still_unknown": sum(1 for x in rows if x["status"] == "unknown")}

tc = [x["true_conflicts"] for x in allrows]
for key in ("prop", "log2_volume", "nclauses", "c200", "c2000"):
    results["correlations"][key] = spearman([x[key] for x in allrows], tc)
results["correlations"]["-prop"] = spearman([-x["prop"] for x in allrows], tc)
argd = [x["true_conflicts"] for x in allrows if x["argD_ref_kill"]]
nond = [x["true_conflicts"] for x in allrows if not x["argD_ref_kill"]]
results["argD_killed_mean_conflicts"] = sum(argd) / len(argd) if argd else None
results["argD_survivor_mean_conflicts"] = sum(nond) / len(nond) if nond else None
results["argD_killed_work_fraction"] = (sum(argd) / (sum(argd) + sum(nond))) if allrows else None
json.dump({"summary": results, "rows": allrows}, open(os.path.join(HERE, "results.json"), "w"), indent=1)
print(json.dumps(results, indent=1))

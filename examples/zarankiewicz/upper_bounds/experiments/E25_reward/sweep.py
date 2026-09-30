"""E25 step 4: sensitivity of the criteria to the constants of the combined variant (offline).

Grid: family balance {off, on} x cell weight log(1+W/w0), w0 in {none, 1, 2000, 20000} x
(w_depth, w_close) in 7 settings x importance reference W_ref in {1e6, 1e9}.  The remaining weight
1 - w_depth - w_close is split over (G_train, G_target, G_gen, Tail) in reward v2's ratio
0.40 : 0.30 : 0.10 : 0.20.  Output: results/sensitivity.{json,md}.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import analyze as A  # noqa: E402
from zar_ub import reward_variants as RV  # noqa: E402

SHOW = ["schema_recipe_frozen", "dgh_lib", "recipe_dgh", "cert_close_10_22", "f_weak", "e2_easy_certs",
        "e1_dgh_s2", "O_e1_clear_10_10", "O_e2_easy_2k", "O_perfect_target_12_18"]


def main():
    include_gt = "--no-gt" not in sys.argv
    labels = "gt" if "--gt-labels" in sys.argv else "table"
    sfx = "_gtlabels" if labels == "gt" else ""
    keys, uni, tabs, progs, cells = A.build_cells(include_gt, labels)
    base = {"V0/S0 current": RV.VARIANTS["V0/S0 current"]}
    grid = {}
    for fam, w0, (wd, wc), wref in itertools.product(
            (False, True), (None, 1.0, 2000.0, 20000.0),
            ((0, 0), (0.1, 0), (0, 0.1), (0.1, 0.1), (0.15, 0.15), (0.2, 0.2), (0.25, 0.25)), (1e6, 1e9)):
        rest = 1.0 - wd - wc
        name = f"fam={int(fam)} w0={w0} depth={wd} close={wc} wref={wref:.0e}"
        grid[name] = RV.Config(name=name, suite="S1", family=fam, work_w0=w0, w_depth=wd, w_close=wc,
                               imp_wref=wref, w_train=0.40 * rest, w_target=0.30 * rest, w_gen=0.10 * rest,
                               w_tail=0.20 * rest)
    variants = dict(base, **grid)
    matrix = {n: {v: RV.verified_score(cells[n], cfg) for v, cfg in variants.items()} for n in cells}
    A.add_soundness_rows(matrix, cells, variants)
    rng = random.Random(25)
    rows = []
    for v, cfg in grid.items():
        c = A.criteria(v, cfg, matrix, cells, rng)
        allp = c["C1"] and c["C2"] and c["C3"] and c["C4"] and c["C5"]
        rows.append({"variant": v, "all_pass": allp, "oracle_pass": c["C3_oracle"],
                     **{k: c[k] for k in ("C1", "C2", "C3", "C3_oracle", "C4", "C5", "C2_ratio", "C5_retained")},
                     "scores": {p: matrix[p][v] for p in SHOW if p in matrix}})
    json.dump(rows, open(os.path.join(A.RES, f"sensitivity{sfx}.json"), "w"), indent=1)
    L = ["# E25 sensitivity grid (S1 suite)", "",
         f"{sum(r['all_pass'] for r in rows)} of {len(rows)} configurations pass C1-C5 (real exploits); "
         f"{sum(r['all_pass'] and r['oracle_pass'] for r in rows)} also pass the oracle stress test.", "",
         "| config | C1-C5 | +oracle | C2 ratio | C5 kept | " + " | ".join(SHOW) + " |",
         "|---|---|---|---|---|" + "---|" * len(SHOW)]
    for r in rows:
        L.append(f"| {r['variant']} | {'pass' if r['all_pass'] else 'FAIL'} | {'pass' if r['oracle_pass'] else 'FAIL'} | "
                 f"{r['C2_ratio']:.2f} | {r['C5_retained']:.2f} | " + " | ".join(f"{r['scores'].get(p, float('nan')):.3f}" for p in SHOW) + " |")
    open(os.path.join(A.RES, f"sensitivity{sfx}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:4]))
    # summary by factor
    for fac in ("fam=", "w0=", "depth=", "wref="):
        vals = {}
        for r in rows:
            key = [t for t in r["variant"].split() if t.startswith(fac)][0]
            if fac == "depth=":
                key = r["variant"].split(" wref")[0].split("w0=")[1].split(" ", 1)[1]
            a = vals.setdefault(key, [0, 0, 0])
            a[0] += r["all_pass"]
            a[1] += r["all_pass"] and r["oracle_pass"]
            a[2] += 1
        print(fac, {k: f"{a}/{b}/{n}" for k, (a, b, n) in vals.items()})


if __name__ == "__main__":
    main()

"""How well does TODAY's estimator (c2000, log2_volume, calibrated fhat) rank the new
ground truth?  Reference numbers for the estimator owners (A2-A4); nothing is fitted here.

  Spearman rho on exact rows (per regime/cell group) and Harrell's C (concordance with
  right-censoring: a pair is comparable when the smaller d is exact and strictly below
  the other's d or lower bound).

usage: python gt_baseline_check.py [ground_truth.jsonl]
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict

from gt_common import HERE, read_jsonl
from zar_ub.difficulty import fhat, load_calibration, spearman


def harrell_c(rows, score):
    """rows with d, status; higher score = predicted harder."""
    num = den = 0.0
    ex = [r for r in rows if r["status"] != "unknown"]
    for a in ex:
        sa = score(a)
        for b in rows:
            if b is a or b["d"] <= a["d"]:
                continue  # b must be harder: b's d (exact or lower bound) > a's exact d
            den += 1
            sb = score(b)
            num += 1.0 if sb > sa else (0.5 if sb == sa else 0.0)
    return (num / den if den else float("nan")), int(den)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ground_truth.jsonl")
    rows = [r for r in read_jsonl(path) if r["status"] in ("unsat", "unknown") and r.get("c2000") is not None]
    cal = load_calibration(3, 3)
    feats = {
        "c2000": lambda r: min(r["c2000"], 2000),
        "log2_volume": lambda r: r["log2_volume"],
        "fhat": lambda r: fhat(cal, r["c2000"], r["log2_volume"]),
    }
    groups = defaultdict(list)
    for r in rows:
        if r["s"] != 3 or r["t"] != 3:
            continue
        wide = r["n"] / r["m"] >= 1.4
        groups["all_33"].append(r)
        groups["wide" if wide else "square"].append(r)
        if r["d"] > 2000 or r["status"] == "unknown":
            groups["d>2000" + ("_wide" if wide else "_square")].append(r)
        if r["d"] > 20000 or (r["status"] == "unknown" and r["cap"] >= 20000):
            groups["hard" + ("_wide" if wide else "_square")].append(r)
        if r["source"] in ("deepen", "new_table"):
            groups["new_labels"].append(r)
    print("calibration (a,b,g) =", cal)
    for g in sorted(groups):
        rs = groups[g]
        ex = [r for r in rs if r["status"] != "unknown"]
        line = f"{g:18s} n={len(rs):6d} exact={len(ex):6d}"
        for k, f in feats.items():
            rho = spearman([f(r) for r in ex], [r["d"] for r in ex]) if len(ex) > 2 else float("nan")
            sub = rs if len(rs) <= 1500 else rs[:: max(1, len(rs) // 1500)]
            c, npairs = harrell_c(sub, f)
            line += f" | {k}: rho={rho:.3f} C={c:.3f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()

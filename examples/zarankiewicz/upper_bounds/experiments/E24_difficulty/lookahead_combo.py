"""E24 / A2-lookahead: are the lookahead (failed-literal) features complementary to A4's
`free20k` CDCL-progress features?  Pre-specified before any TEST result was looked at.

Join (read-only) A4's published feature files (progress_features_{dev,test,test2,test3}.jsonl,
field `features`) with ours (lookahead_features_{dev,test}.jsonl) on (cell, rows, cols), HARD
cases only.  Models (log d, least squares, fitted on the 262 DEV hard cases):
  A4 free20k        ps20k_decs_per_conf, ps20k_restarts_per_kconf, st_distinct_rows, st_row_lp_lcap
  FL-only [Honly]   log2_volume, fl_free_cells, fl_log2vol_rows
  free20k + FL      the union (7 features)
  free20k + fl_free the 4 A4 features + fl_free_cells
Reported: DEV leave-one-cell-out (4 cells) and TEST frozen-on-DEV (every joined test hard case).

usage: python lookahead_combo.py
"""

from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lookahead_eval import by_cell, fit, load, lrmse, nanmean, predict, top_recall  # noqa: E402
from lookahead_eval import spearman  # noqa: E402

A4 = ["ps20k_decs_per_conf", "ps20k_restarts_per_kconf", "st_distinct_rows", "st_row_lp_lcap"]
FL = ["log2_volume", "fl_free_cells", "fl_log2vol_rows"]
MODELS = {
    "A4 free20k": A4,
    "FL-only [Honly]": FL,
    "free20k + FL": A4 + FL,
    "free20k + fl_free": A4 + ["fl_free_cells"],
}


def a4_features(files):
    out = {}
    for fn in files:
        p = os.path.join(HERE, fn)
        if not os.path.exists(p):
            continue
        for l in open(p):
            r = json.loads(l)
            if r.get("regime") != "hard" or "ps20k_decs_per_conf" not in r["features"]:
                continue
            out[(r["cell"], tuple(r["rows"]), tuple(r["cols"]))] = r["features"]
    return out


def join(ours, a4):
    rows = []
    for r in ours:
        if r["subset"] != "H":
            continue
        f = a4.get((r["cell"], tuple(r["rows"]), tuple(r["cols"])))
        if f is None:
            continue
        r = dict(r)
        r["F"] = dict(r["F"])
        r["F"].update({k: float(f[k]) for k in A4})
        rows.append(r)
    return rows


def score(rows, preds):
    cells = by_cell(rows)
    wc = {
        c: spearman([preds[id(r)] for r in g], [r["d"] for r in g])
        for c, g in cells.items()
        if len(g) >= 8
    }
    tr = {c: top_recall([preds[id(r)] for r in g], g) for c, g in cells.items() if len(g) >= 20}
    p = [preds[id(r)] for r in rows]
    return {
        "n": len(rows),
        "pooled": spearman(p, [r["d"] for r in rows]),
        "within": nanmean(list(wc.values())),
        "per_cell": wc,
        "log_rmse": lrmse(p, rows),
        "top10": nanmean(list(tr.values())),
    }


def main():
    a4 = a4_features(
        [
            "progress_features_dev.jsonl",
            "progress_features_test.jsonl",
            "progress_features_test2.jsonl",
            "progress_features_test3.jsonl",
        ]
    )
    dev = join(load(os.path.join(HERE, "lookahead_features_dev.jsonl")), a4)
    test = join(load(os.path.join(HERE, "lookahead_features_test.jsonl")), a4)
    res = {
        "n_dev": len(dev),
        "n_test": len(test),
        "test_cells": dict(collections.Counter(r["cell"] for r in test)),
        "dev_loco": {},
        "test_frozen": {},
    }
    for nm, fs in MODELS.items():
        P = {}
        for c in sorted(set(r["cell"] for r in dev)):
            tr = [r for r in dev if r["cell"] != c]
            te = [r for r in dev if r["cell"] == c]
            P.update({id(r): v for r, v in zip(te, predict(fit(tr, fs), te))})
        res["dev_loco"][nm] = score(dev, P)
        if test:
            m = fit(dev, fs)
            res["test_frozen"][nm] = score(test, {id(r): v for r, v in zip(test, predict(m, test))})
    json.dump(res, open(os.path.join(HERE, "lookahead_combo.json"), "w"), indent=1, default=float)
    print(f"joined hard cases: dev {len(dev)}, test {len(test)} {res['test_cells']}")
    for part in ("dev_loco", "test_frozen"):
        print(f"[{part}]  pooled | within | log-RMSE | top10")
        for nm, s in res[part].items():
            print(
                "  %-20s %+.3f | %+.3f | %.3f | %.2f  %s"
                % (
                    nm,
                    s["pooled"],
                    s["within"],
                    s["log_rmse"],
                    s["top10"],
                    {k.split("_s3")[0]: round(v, 2) for k, v in s["per_cell"].items()},
                )
            )


if __name__ == "__main__":
    main()

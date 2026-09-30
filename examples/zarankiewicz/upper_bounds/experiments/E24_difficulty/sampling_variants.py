"""E24 / A3: offline variants of the sampling statistic on the stored cube records (no solver).

The retest experiment shows sampling noise dominates (seed-to-seed Spearman 0.70 at N = 100), so
this compares lower-variance statistics of the SAME cubes:
  mu        censoring-aware mean of weight * xi (the Chivilikhin estimator; the default)
  mu_lb     mean of weight * min(xi, b) (no Pareto tail term)
  mom5      median of 5 group means (median-of-means)
  geo       exp(mean log(1 + weight * xi))  (geometric mean over all probes)
  tree      Knuth tree-size estimate of the UP-surviving leaves at depth k: mean weight of the
            live probes -- needs NO conflicts (propagations only)
  tree*med  tree * median xi over live probes
Metrics: DEV LOCO within-cell Spearman (hard), frozen-on-DEV wide/target within-cell Spearman
(hard exact) and Harrell's C (target, with the 2M-censored cases).

usage: python experiments/E24_difficulty/sampling_variants.py [--cfg knuth:row:8]"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from sampling_final import jl, key_of, loco, score, fit, predict  # noqa: E402
from zar_ub import hardness_sampling as hs  # noqa: E402


def cube_list(rec, op):
    kind, x, b = op
    out, acc = [], 0
    for c in rec["recs"]:
        if kind == "N" and len(out) >= x:
            break
        out.append(c)
        acc += min(c[0], b)
        if kind == "T" and acc >= x:
            break
    return out


def stats(rec, op):
    b = op[2]
    cs = cube_list(rec, op)
    recs = [dict(conflicts=c[0], propagations=c[1], censored=bool(c[2]), weight=c[3], up_refuted=bool(c[4]),
                 status="unsat") for c in cs]
    s = hs.summarize(recs, b, rec["log2_space"])
    wx = [c[3] * min(c[0], b) for c in cs]
    g = [statistics.mean(wx[i::5]) for i in range(5) if wx[i::5]]
    live = [c for c in cs if not c[4]]
    tree = sum(c[3] for c in live) / len(cs)
    medx = statistics.median([min(c[0], b) for c in live]) if live else 0.0
    return {
        "mu": s["d_hat"], "mu_lb": s["mu_lb"], "mom5": statistics.median(g),
        "geo": math.exp(statistics.mean(math.log1p(v) for v in wx)),
        "tree": tree, "tree*med": tree * max(medx, 1.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="knuth:row:8")
    ap.add_argument("--out", default=os.path.join(HERE, "sampling_data", "variants.json"))
    a = ap.parse_args()
    D = os.path.join(HERE, "sampling_data")
    gt = {}
    for r in jl([os.path.join(HERE, "ground_truth.jsonl")]):
        gt[key_of(r)] = r
    sets = {"dev": jl([os.path.join(D, "dev.jsonl")]), "wide": jl([os.path.join(D, "wide_test.jsonl")]),
            "target": jl([os.path.join(D, "target.jsonl")])}
    res = {}
    for op in (("T", 50_000, 5000), ("N", 100, 5000), ("N", 50, 2000)):
        tag = f"{op[0]}{op[1]}_b{op[2]}"
        res[tag] = {}
        rows = {}
        for name, runs in sets.items():
            rr = []
            for rec in runs:
                if rec.get("config") != a.cfg or "error" in rec or rec.get("bmax", 10_000) < op[2]:
                    continue
                g = gt.get(rec["key"])
                if g is None:
                    continue
                exact = g["status"] in ("unsat", "sat")
                if (exact and g["d"] <= 20000) or (not exact and g["d"] < 1_000_000):
                    continue
                rr.append(dict(key=rec["key"], cell=rec["cell"], d=float(g["d"]), exact=exact, st=stats(rec, op)))
            rows[name] = rr
        print(f"== {tag}: dev {len(rows['dev'])}, wide {len(rows['wide'])}, target {len(rows['target'])}")
        for v in ("mu", "mu_lb", "mom5", "geo", "tree", "tree*med"):
            fx = lambda r, v=v: [math.log(max(r["st"][v], 1e-3))]
            dl = loco(rows["dev"], fx)
            m = fit([fx(r) for r in rows["dev"]], [math.log(r["d"]) for r in rows["dev"]])
            out = {"dev_loco": dl}
            line = f"  {v:9s} DEV LOCO within {dl['within']:.3f} pooled {dl['pooled']:.3f}"
            for name in ("wide", "target"):
                if len(rows[name]) >= 30:
                    s = score(rows[name], predict(m, [fx(r) for r in rows[name]]))
                    out[name] = s
                    line += f" | {name} within {s['within']:.3f} pooled {s['pooled']:.3f}"
                    if "harrell_within" in s:
                        line += f" C_w {s['harrell_within']:.3f}"
            res[tag][v] = out
            print(line)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

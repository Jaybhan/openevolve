"""E24 / A3: design screen of the sampling estimator on E10 (7 TRAIN cells, exact labels).

Compares decomposition-set designs x samplers x k x N x b on the COMMON case subset (cases that
every listed config has run: 30 hard + 28 mid when the uniform2 screen is complete), raw
(uncalibrated) Spearman of mu~ with the true d, pooled and mean within-cell (cells with >= 8
cases), plus cost and the fraction of cubes refuted by unit propagation / hitting the budget and
the median achieved relative error eps (Chebyshev, delta = 0.1; paper eq. 13).

usage: python experiments/E24_difficulty/sampling_designs.py [--runs sampling_data/screen_e10.jsonl]
       [--out sampling_data/designs.json] [--cap 50000]"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from sampling_eval import derive  # noqa: E402


def rho(x, y):
    return float(spearmanr(x, y).correlation) if len(x) >= 3 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "sampling_data", "screen_e10.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "sampling_data", "designs.json"))
    ap.add_argument("--cap", type=float, default=50_000)
    a = ap.parse_args()
    by = defaultdict(dict)
    for line in open(a.runs):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "error" in r:
            continue
        by[r["config"]][r["key"]] = r
    full = {c: v for c, v in by.items() if len(v) >= 50}
    common = set.intersection(*(set(v) for v in full.values()))
    print(f"{len(full)} configs, common cases {len(common)}")
    res = []
    for cfg, recs in sorted(full.items()):
        for N in (20, 50, 100):
            for b in (500, 2000, 10000):
                rows = []
                for k in common:
                    rec = recs[k]
                    dv = derive(rec, N, b)
                    rows.append(dict(cell=rec["cell"], d=rec["d"], mu=max(dv["d_hat"], 1.0), cost=dv["cost_conflicts"],
                                     up=dv["f"]["frac_up_refuted"], hit=dv["f"]["frac_budget_hit"],
                                     eps=dv["f"]["eps_achieved"], props=dv["cost_props"]))
                out = dict(cfg=cfg, N=N, b=b)
                for reg, f in (("hard", lambda r: r["d"] > 20000), ("mid", lambda r: 2000 < r["d"] <= 20000)):
                    sub = [r for r in rows if f(r)]
                    cells = defaultdict(list)
                    for r in sub:
                        cells[r["cell"]].append(r)
                    w = [rho([r["mu"] for r in v], [r["d"] for r in v]) for v in cells.values() if len(v) >= 8]
                    out[reg] = dict(n=len(sub), pooled=rho([r["mu"] for r in sub], [r["d"] for r in sub]),
                                    within=float(np.nanmean(w)) if w else float("nan"),
                                    cost=float(np.mean([r["cost"] for r in sub])),
                                    props=float(np.mean([r["props"] for r in sub])),
                                    up=float(np.mean([r["up"] for r in sub])), hit=float(np.mean([r["hit"] for r in sub])),
                                    eps_med=float(np.median([r["eps"] for r in sub])))
                res.append(out)
    print(f"{'config':22s} {'N':>4s} {'b':>6s} | {'hard n':>6s} {'pooled':>6s} {'within':>6s} {'cost':>8s} {'UPref':>5s} {'hit':>5s} {'eps':>6s} | "
          f"{'mid n':>5s} {'pooled':>6s} {'within':>6s} {'cost':>7s}")
    for o in res:
        h, m = o["hard"], o["mid"]
        print(f"{o['cfg']:22s} {o['N']:4d} {o['b']:6d} | {h['n']:6d} {h['pooled']:6.3f} {h['within']:6.3f} {h['cost']:8.0f} {h['up']:5.2f} "
              f"{h['hit']:5.2f} {h['eps_med']:6.2f} | {m['n']:5d} {m['pooled']:6.3f} {m['within']:6.3f} {m['cost']:7.0f}")
    print(f"\nbest (hard pooled raw Spearman) per sampler:design with hard mean cost <= {a.cap:.0f}:")
    best = {}
    for o in res:
        d = ":".join(o["cfg"].split(":")[:2])
        if o["hard"]["cost"] <= a.cap and o["hard"]["pooled"] == o["hard"]["pooled"]:
            if d not in best or o["hard"]["pooled"] > best[d]["hard"]["pooled"]:
                best[d] = o
    for d, o in sorted(best.items(), key=lambda kv: -kv[1]["hard"]["pooled"]):
        print(f"  {d:18s} {o['cfg']:22s} N{o['N']:4d} b{o['b']:6d}: hard pooled {o['hard']['pooled']:.3f} within "
              f"{o['hard']['within']:.3f} cost {o['hard']['cost']:.0f} UP-refuted {o['hard']['up']:.2f} | mid pooled "
              f"{o['mid']['pooled']:.3f} within {o['mid']['within']:.3f} cost {o['mid']['cost']:.0f}")
    json.dump(dict(common=len(common), rows=res, best={k: v for k, v in best.items()}), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

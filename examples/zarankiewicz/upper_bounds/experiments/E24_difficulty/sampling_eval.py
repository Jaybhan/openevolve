"""E24 / A3: evaluate sampling-estimator runs (sampling_run.py output) against true d.

For every stored (case, sampler:design:k) record (N = 100 cubes at budget 10 000) and
every derived (N, b) with N in {20, 50, 100} and b in {500, 2000, 10000}:
  d_hat = censoring-aware mu~ (hardness_sampling.summarize), cost = conflicts spent.
Metrics (exact labels only; HARD = d > 20 000, MID = 2 000 < d <= 20 000):
  * raw Spearman(d_hat, d): pooled and mean within-cell (cells with >= MIN_CELL cases)
  * leave-one-cell-out log-linear calibration  log d = a + b log d_hat  (fit on the
    training cells' exact cases with d > 2000), held-out: pooled Spearman of the
    concatenated predictions, within-cell Spearman, log-RMSE (natural log)
  * the same with + log2_volume as a second regressor ("+vol")
  baselines on the same cases: fhat (fixed coefficients from calibrate_33.json), fhat
  refit LOCO, constant LOCO, and the equal-cost direct solve  min(d, C)  (a fresh
  CaDiCaL run capped at C conflicts is deterministic, so its outcome is min(d, C)).

usage: python experiments/E24_difficulty/sampling_eval.py <runs.jsonl> [--gt <ground truth jsonl>]
       [--out <json>] [--configs a,b,...] [--nb 100:10000,...]
"""
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
UB = os.path.dirname(os.path.dirname(HERE))
if UB not in sys.path:
    sys.path.insert(0, UB)

from zar_ub import hardness_sampling as hs  # noqa: E402
from zar_ub.difficulty import fhat, load_calibration  # noqa: E402

MIN_CELL = 8
NS = (20, 50, 100)
BS = (500, 2000, 10000)
EQ_CAPS = (20_000, 50_000, 100_000)


def regime(d):
    return "hard" if d > 20000 else ("mid" if d > 2000 else "easy")


def rho(x, y):
    if len(x) < 3:
        return float("nan")
    r = spearmanr(x, y).correlation
    return float(r) if r == r else float("nan")


def within(cells, x, y):
    by = defaultdict(list)
    for c, a, b in zip(cells, x, y):
        by[c].append((a, b))
    vals, ws = [], []
    per = {}
    for c, v in by.items():
        if len(v) >= MIN_CELL:
            r = rho([a for a, _ in v], [b for _, b in v])
            per[c] = (r, len(v))
            if r == r:
                vals.append(r)
                ws.append(len(v))
    wm = float(np.average(vals, weights=ws)) if vals else float("nan")
    return (float(np.mean(vals)) if vals else float("nan")), wm, per


def lstsq(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    sol, *_ = np.linalg.lstsq(X, y, rcond=None)
    return sol


def loco(rows, featfn, fit_filter=lambda r: r["d"] > 2000):
    """rows: list of dicts with cell, d, and whatever featfn reads.  Returns
    {key: pred} for every row, prediction from a fit on the other cells."""
    cells = sorted(set(r["cell"] for r in rows))
    pred = {}
    for c in cells:
        tr = [r for r in rows if r["cell"] != c and fit_filter(r)]
        te = [r for r in rows if r["cell"] == c]
        if len(tr) < 5:
            continue
        X = [[1.0] + featfn(r) for r in tr]
        y = [math.log(r["d"]) for r in tr]
        coef = lstsq(X, y)
        for r in te:
            pred[r["key"]] = float(np.dot(coef, [1.0] + featfn(r)))
    return pred


def in_reg(r, reg):
    if reg == "gt50k":
        return r["d"] > 50000
    if reg == "gt100k":
        return r["d"] > 100000
    return r["reg"] == reg


FIT = {"hard": lambda r: r["d"] > 20000, "gt50k": lambda r: r["d"] > 20000, "gt100k": lambda r: r["d"] > 20000,
       "mid": lambda r: 2000 < r["d"] <= 20000}


def loco_reg(rows, featfn, reg):
    """LOCO calibration fitted inside the regime where it is used (hard: d > 20k)."""
    return loco(rows, featfn, fit_filter=FIT[reg])


def metrics(rows, pred_log, reg):
    sub = [r for r in rows if in_reg(r, reg) and r["key"] in pred_log]
    if len(sub) < 3:
        return None
    p = [pred_log[r["key"]] for r in sub]
    t = [math.log(r["d"]) for r in sub]
    wm, wwm, per = within([r["cell"] for r in sub], p, t)
    return dict(n=len(sub), pooled=rho(p, t), within_mean=wm, within_wmean=wwm,
                log_rmse=float(np.sqrt(np.mean((np.array(p) - np.array(t)) ** 2))),
                per_cell={c: round(v[0], 3) for c, v in per.items()})


def raw_metrics(rows, val, reg):
    sub = [r for r in rows if in_reg(r, reg)]
    if len(sub) < 3:
        return None
    p = [val(r) for r in sub]
    t = [r["d"] for r in sub]
    wm, wwm, per = within([r["cell"] for r in sub], p, t)
    return dict(n=len(sub), pooled=rho(p, t), within_mean=wm, within_wwmean=wwm,
                per_cell={c: round(v[0], 3) for c, v in per.items()})


def load_gt(paths):
    """Ground-truth rows keyed like sampling_run.key_of; several comma-separated files,
    earlier files win.  log2_volume is recomputed when a row lacks it."""
    from math import comb, log2
    gt = {}
    for path in [p for p in (paths or "").split(",") if p]:
        if not os.path.exists(path):
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if "inst" in r and "m" not in r:
                r.update(r["inst"])
            if r.get("log2_volume") is None:
                r["log2_volume"] = float(sum(log2(comb(r["n"], x)) for x in r["rows"]))
            k = f'{r["cell"]}|{",".join(map(str, r["rows"]))}|{",".join(map(str, r["cols"]))}'
            gt.setdefault(k, r)
    return gt


def derive(rec, N, b):
    recs = []
    for x in rec["recs"][:N]:
        c, p, cz, w, u, st = x[:6]
        d = dict(conflicts=c, propagations=p, censored=bool(cz), weight=w, up_refuted=bool(u),
                 status={"u": "unsat", "s": "sat", "n": "unknown"}.get(st, "unknown"))
        if len(x) > 6:
            d["stratum"] = x[6]
        recs.append(d)
    s = hs.summarize(recs, b, rec["log2_space"])
    frac = s["cost_conflicts"] / max(1, rec["cost_conflicts"])
    return dict(d_hat=s["d_hat"], mu_lb=s["mu_lb"], f=s["features"], cost_conflicts=s["cost_conflicts"],
                cost_props=s["cost_propagations"] + rec.get("extra_props", 0) * N / len(rec["recs"]),
                cost_seconds_est=rec["cost_seconds"] * (0.15 * N / len(rec["recs"]) + 0.85 * frac))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--gt", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--configs", default="")
    ap.add_argument("--nb", default="")
    ap.add_argument("--exact-only", action="store_true", default=True)
    a = ap.parse_args()
    gt = load_gt(a.gt)
    calib = load_calibration(3, 3)
    by_cfg = defaultdict(list)
    for line in open(a.runs):
        r = json.loads(line)
        if "error" in r:
            print("ERROR", r["key"], r["config"], r["error"])
            continue
        if not r.get("exact", True):
            continue
        by_cfg[r["config"]].append(r)
    cfgs = sorted(by_cfg) if not a.configs else a.configs.split(",")
    nbs = [(N, b) for N in NS for b in BS] if not a.nb else [tuple(map(int, x.split(":"))) for x in a.nb.split(",")]
    results = {}
    base_done = False
    for cfg in cfgs:
        recs = by_cfg[cfg]
        for N, b in nbs:
            rows = []
            for rec in recs:
                g = gt.get(rec["key"], {})
                dv = derive(rec, N, b)
                c2000 = g.get("c2000")
                c2000 = None if c2000 is None else int(c2000)
                vol = g.get("log2_volume")
                rows.append(dict(key=rec["key"], cell=rec["cell"], d=float(max(1, rec["d"])), reg=regime(rec["d"]),
                                 dh=max(dv["d_hat"], 1.0), lb=max(dv["mu_lb"], 1.0), f=dv["f"],
                                 c2000=c2000, vol=vol, cost=dv["cost_conflicts"], props=dv["cost_props"],
                                 secs=dv["cost_seconds_est"]))
            L = lambda r: [math.log(r["dh"])]
            res = {"N": N, "b": b, "n": len(rows)}
            for reg in ("hard", "mid", "gt50k", "gt100k"):
                sub = [r for r in rows if in_reg(r, reg)]
                res[reg] = {
                    "raw": raw_metrics(rows, lambda r: r["dh"], reg),
                    "raw_lb": raw_metrics(rows, lambda r: r["lb"], reg),
                    "cost_conflicts_mean": float(np.mean([r["cost"] for r in sub])) if sub else None,
                    "cost_conflicts_max": float(np.max([r["cost"] for r in sub])) if sub else None,
                    "cost_props_mean": float(np.mean([r["props"] for r in sub])) if sub else None,
                    "cost_seconds_mean_est": float(np.mean([r["secs"] for r in sub])) if sub else None,
                    "frac_budget_hit_mean": float(np.mean([r["f"]["frac_budget_hit"] for r in sub])) if sub else None,
                    "frac_up_refuted_mean": float(np.mean([r["f"]["frac_up_refuted"] for r in sub])) if sub else None,
                    "N_req_median": float(np.median([r["f"]["N_req"] for r in sub])) if sub else None,
                    "eps_achieved_median": float(np.median([r["f"]["eps_achieved"] for r in sub])) if sub else None,
                }
            multi = lambda r: [math.log(r["dh"]), r["f"]["log2_live_leaves"], r["f"]["frac_up_refuted"],
                               math.log(1 + r["f"]["mean_live_work"])]
            p_all = loco(rows, L)
            for reg in ("hard", "mid", "gt50k", "gt100k"):
                res[reg]["loco"] = metrics(rows, loco_reg(rows, L, reg), reg)
                res[reg]["loco_fit_all"] = metrics(rows, p_all, reg)
                res[reg]["loco_multi"] = metrics(rows, loco_reg(rows, multi, reg), reg)
                if all(r["vol"] is not None for r in rows):
                    res[reg]["loco_vol"] = metrics(rows, loco_reg(rows, lambda r: [math.log(r["dh"]), r["vol"]], reg), reg)
            results[f"{cfg}|N{N}|b{b}"] = res
            if not base_done and all(r["c2000"] is not None for r in rows):
                base = {}
                for reg in ("hard", "mid", "gt50k", "gt100k"):
                    fx = {r["key"]: math.log(fhat(calib, r["c2000"], r["vol"])) for r in rows}
                    base[f"fhat_fixed_{reg}"] = metrics(rows, fx, reg)
                    pf = loco_reg(rows, lambda r: [math.log(max(min(r["c2000"], 2000), 1)), r["vol"]], reg)
                    base[f"fhat_loco_{reg}"] = metrics(rows, pf, reg)
                    pc = loco_reg(rows, lambda r: [], reg)
                    base[f"const_loco_{reg}"] = metrics(rows, pc, reg)
                    for C in EQ_CAPS:
                        # direct solve at cap C, then log-linear LOCO calibration of the censored value
                        pd = loco_reg(rows, lambda r, C=C: [math.log(min(r["d"], C)), float(r["d"] > C)], reg)
                        base[f"direct{C // 1000}k_loco_{reg}"] = metrics(rows, pd, reg)
                        base[f"direct{C // 1000}k_raw_{reg}"] = raw_metrics(rows, lambda r, C=C: min(r["d"], C), reg)
                        pdv = loco_reg(rows, lambda r, C=C: [math.log(min(r["d"], C)), float(r["d"] > C),
                                                             float(r["d"] > C) * r["vol"]], reg)
                        base[f"direct{C // 1000}k+vol_loco_{reg}"] = metrics(rows, pdv, reg)
                results["_baselines"] = base
                results["_baseline_cases"] = cfg
                base_done = True
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
    # compact table
    print(f"{'config':28s} {'N':>4s} {'b':>6s} | {'hard raw':>8s} {'within':>7s} {'LOCO':>6s} {'wLOCO':>6s} {'lrmse':>6s} {'+vol':>6s} | "
          f"{'mid raw':>7s} {'within':>7s} | {'>50k':>6s} {'>50kL':>6s} {'lbraw':>6s} {'multi':>6s} | {'conf/case':>9s} {'s/case':>6s} {'hit':>5s} {'up':>5s}")
    for k, v in results.items():
        if k.startswith("_"):
            continue
        h, m = v["hard"], v["mid"]
        if not h.get("raw"):
            continue
        lv = h.get("loco_vol") or {}
        lc = h.get("loco") or {}
        print(f"{k.split('|')[0]:28s} {v['N']:4d} {v['b']:6d} | {h['raw']['pooled']:8.3f} {h['raw']['within_mean']:7.3f} "
              f"{lc.get('pooled', float('nan')):6.3f} {lc.get('within_mean', float('nan')):6.3f} {lc.get('log_rmse', float('nan')):6.3f} "
              f"{lv.get('pooled', float('nan')):6.3f} | {(m['raw'] or {}).get('pooled', float('nan')):7.3f} "
              f"{(m['raw'] or {}).get('within_mean', float('nan')):7.3f} | "
              f"{(v['gt50k']['raw'] or {}).get('pooled', float('nan')):6.3f} {(v['gt50k'].get('loco') or {}).get('within_mean', float('nan')):6.3f} "
              f"{(h['raw_lb'] or {}).get('pooled', float('nan')):6.3f} {(h.get('loco_multi') or {}).get('pooled', float('nan')):6.3f} | {h['cost_conflicts_mean']:9.0f} "
              f"{h['cost_seconds_mean_est']:6.2f} {h['frac_budget_hit_mean']:5.2f} {h['frac_up_refuted_mean']:5.2f}")
    if "_baselines" in results:
        print("baselines (same cases):")
        for k, v in results["_baselines"].items():
            if v:
                print(f"  {k:28s} n={v['n']:4d} pooled={v['pooled']:.3f} within={v['within_mean']:.3f}"
                      + (f" lrmse={v['log_rmse']:.3f}" if "log_rmse" in v else ""))


if __name__ == "__main__":
    main()

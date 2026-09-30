"""E24 / A3: frozen-on-DEV -> wide-cell test of the sampling estimator, alone and on top of
A4's `free20k` features (complementarity check).

Protocol (same as A4's PROGRESS.md headline):
  1. DEV = the 4 square cells with exact hard labels (ground_truth_initial.jsonl, 262 cases).
     Choose the sampling operating point (config, N, b) with mean cost <= COST_CAP conflicts
     on DEV hard by leave-one-cell-out within-cell Spearman (selection on DEV only).
  2. Fit every model on all DEV hard cases, freeze, predict the wide-cell hard cases
     (A4's progress_features_test{,2,3}.jsonl: cells never used for fitting).
Models (log d = a + sum b_k z_k):
  fhat-form   log c2000, log2_volume                       (current label's form, refit)
  samp        log d_hat                                     (sampling, 1 feature)
  samp+vol    log d_hat, log2_volume
  free20k     A4's 4 features (ps20k decisions/conflict, log1p restarts/kconflict,
              distinct row sums, column LP slack), refit here
  free20k+samp  the 4 A4 features + log d_hat
Also LOCO over the 4 DEV cells for every model.

usage: python experiments/E24_difficulty/sampling_combine.py [--dev-runs ...] [--test-runs ...] [--out ...]
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
sys.path.insert(0, HERE)

from sampling_eval import derive  # noqa: E402

COST_CAP = 50_000
A4_FEATS = [("ps20k_decs_per_conf", "id"), ("ps20k_restarts_per_kconf", "log1p"), ("st_distinct_rows", "id"),
            ("st_col_lp_slackT", "id")]
NB = [(N, b) for N in (20, 50, 100) for b in (500, 2000, 5000, 10000)]


def key_of(r):
    return f'{r["cell"]}|{",".join(map(str, r["rows"]))}|{",".join(map(str, r["cols"]))}'


def load_jsonl(paths):
    out = []
    for p in paths:
        if not os.path.exists(p):
            continue
        for line in open(p):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def tr(x, how):
    return math.log1p(x) if how == "log1p" else float(x)


def fit(X, y):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    mu = X[:, 1:].mean(0)
    sd = X[:, 1:].std(0)
    sd[sd == 0] = 1.0
    Xs = X.copy()
    Xs[:, 1:] = (X[:, 1:] - mu) / sd
    coef, *_ = np.linalg.lstsq(Xs, np.asarray(y, float), rcond=None)
    return coef, mu, sd


def predict(model, X):
    coef, mu, sd = model
    X = np.asarray(X, float)
    return coef[0] + ((X - mu) / sd) @ coef[1:]


def score(rows, pred):
    t = np.array([math.log(r["d"]) for r in rows])
    p = np.asarray(pred)
    by = defaultdict(list)
    for r, a, b in zip(rows, p, t):
        by[r["cell"]].append((a, b))
    per = {c: float(spearmanr([a for a, _ in v], [b for _, b in v]).correlation) for c, v in by.items() if len(v) >= 8}
    wn = {c: len(v) for c, v in by.items() if len(v) >= 8}
    within = float(np.mean(list(per.values()))) if per else float("nan")
    wwithin = float(sum(per[c] * wn[c] for c in per) / sum(wn.values())) if per else float("nan")
    return dict(n=len(rows), within=within, within_w=wwithin, pooled=float(spearmanr(p, t).correlation),
                log_rmse=float(np.sqrt(np.mean((p - t) ** 2))), per_cell={c: round(v, 3) for c, v in per.items()})


MODELS = {
    "fhat-form": lambda r: [math.log(max(min(r["c2000"], 2000), 1)), r["vol"]],
    "samp": lambda r: [math.log(r["dh"])],
    "samp+vol": lambda r: [math.log(r["dh"]), r["vol"]],
    "free20k": lambda r: r["a4"],
    "free20k+samp": lambda r: r["a4"] + [math.log(r["dh"])],
}


def build_rows(runs, feats, cfg, N, b):
    rows = []
    for rec in runs:
        if rec.get("config") != cfg or "error" in rec:
            continue
        f = feats.get(rec["key"])
        if f is None or f.get("status") not in ("unsat", "sat") or f["d"] <= 20000:
            continue
        ff = f.get("features", {})
        if any(k not in ff or ff[k] is None for k, _ in A4_FEATS):
            continue
        dv = derive(rec, N, b)
        rows.append(dict(key=rec["key"], cell=rec["cell"], d=float(f["d"]), dh=max(dv["d_hat"], 1.0),
                         cost=dv["cost_conflicts"], props=dv["cost_props"], secs=dv["cost_seconds_est"],
                         c2000=int(f.get("c2000") or 2000), vol=float(f["log2_volume"]),
                         a4=[tr(ff[k], how) for k, how in A4_FEATS]))
    return rows


def loco(rows, fx):
    cells = sorted(set(r["cell"] for r in rows))
    preds = {}
    for c in cells:
        trn = [r for r in rows if r["cell"] != c]
        tst = [r for r in rows if r["cell"] == c]
        m = fit([fx(r) for r in trn], [math.log(r["d"]) for r in trn])
        for r, p in zip(tst, predict(m, [fx(r) for r in tst])):
            preds[r["key"]] = p
    return score(rows, [preds[r["key"]] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-runs", default=os.path.join(HERE, "sampling_data", "dev.jsonl"))
    ap.add_argument("--test-runs", default=os.path.join(HERE, "sampling_data", "wide_test.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "sampling_data", "combine.json"))
    a = ap.parse_args()
    dev_feats = {key_of(r): r for r in load_jsonl([os.path.join(HERE, "progress_features_dev.jsonl")])}
    test_feats = {key_of(r): r for r in load_jsonl([os.path.join(HERE, f"progress_features_test{s}.jsonl")
                                                     for s in ("", "2", "3")])}
    dev_runs = load_jsonl([a.dev_runs])
    test_runs = load_jsonl([a.test_runs])
    cfgs = sorted(set(r["config"] for r in dev_runs if "error" not in r))
    # 1. operating point selection on DEV (LOCO within-cell, samp model, cost cap)
    table = []
    for cfg in cfgs:
        for N, b in NB:
            rows = build_rows(dev_runs, dev_feats, cfg, N, b)
            if len(rows) < 30:
                continue
            cost = float(np.mean([r["cost"] for r in rows]))
            s = loco(rows, MODELS["samp"])
            table.append(dict(cfg=cfg, N=N, b=b, n=len(rows), cost=cost, props=float(np.mean([r["props"] for r in rows])),
                              secs=float(np.mean([r["secs"] for r in rows])), **{f"loco_{k}": v for k, v in s.items() if k != "per_cell"}))
    table.sort(key=lambda x: x["cost"])
    ok = [t for t in table if t["cost"] <= COST_CAP]
    best = max(ok, key=lambda t: t["loco_within"]) if ok else None
    out = {"dev_curve": table, "selected": best}
    print("DEV operating points (samp model, LOCO over the 4 DEV cells), sorted by cost:")
    for t in table:
        print(f"  {t['cfg']:16s} N{t['N']:4d} b{t['b']:6d} n={t['n']:4d} cost={t['cost']:8.0f} conf props={t['props']:.3g} "
              f"s={t['secs']:.2f} | within={t['loco_within']:.3f} pooled={t['loco_pooled']:.3f} lrmse={t['loco_log_rmse']:.3f}")
    if not best:
        json.dump(out, open(a.out, "w"), indent=1)
        return
    print("selected:", best["cfg"], best["N"], best["b"])
    cfg, N, b = best["cfg"], best["N"], best["b"]
    dev = build_rows(dev_runs, dev_feats, cfg, N, b)
    out["dev_loco"] = {m: loco(dev, fx) for m, fx in MODELS.items()}
    print(f"\nDEV LOCO ({len(dev)} hard cases):")
    for m, s in out["dev_loco"].items():
        print(f"  {m:14s} within={s['within']:.3f} pooled={s['pooled']:.3f} lrmse={s['log_rmse']:.3f}  {s['per_cell']}")
    test = build_rows(test_runs, test_feats, cfg, N, b)
    if test:
        out["frozen_test"] = {}
        print(f"\nFROZEN on DEV -> wide test ({len(test)} hard cases, cells {sorted(set(r['cell'] for r in test))}):")
        for m, fx in MODELS.items():
            mod = fit([fx(r) for r in dev], [math.log(r["d"]) for r in dev])
            s = score(test, predict(mod, [fx(r) for r in test]))
            out["frozen_test"][m] = s
            print(f"  {m:14s} within={s['within']:.3f} pooled={s['pooled']:.3f} lrmse={s['log_rmse']:.3f}  {s['per_cell']}")
        # raw (uncalibrated) sampling ranking on test and the other operating points on test
        out["test_curve"] = []
        for cfg2 in sorted(set(r["config"] for r in test_runs if "error" not in r)):
            for N2, b2 in NB:
                tr_ = build_rows(test_runs, test_feats, cfg2, N2, b2)
                dv_ = build_rows(dev_runs, dev_feats, cfg2, N2, b2)
                if len(tr_) < 30 or len(dv_) < 30:
                    continue
                mod = fit([MODELS["samp"](r) for r in dv_], [math.log(r["d"]) for r in dv_])
                s = score(tr_, predict(mod, [MODELS["samp"](r) for r in tr_]))
                mod2 = fit([MODELS["free20k+samp"](r) for r in dv_], [math.log(r["d"]) for r in dv_])
                s2 = score(tr_, predict(mod2, [MODELS["free20k+samp"](r) for r in tr_]))
                out["test_curve"].append(dict(cfg=cfg2, N=N2, b=b2, n=len(tr_), cost=float(np.mean([r["cost"] for r in tr_])),
                                              props=float(np.mean([r["props"] for r in tr_])),
                                              secs=float(np.mean([r["secs"] for r in tr_])),
                                              samp=s, free20k_samp=s2))
        print("\nwide test, every operating point (frozen samp model; frozen free20k+samp):")
        for t in sorted(out["test_curve"], key=lambda t: t["cost"]):
            print(f"  {t['cfg']:16s} N{t['N']:4d} b{t['b']:6d} cost={t['cost']:8.0f} | samp within={t['samp']['within']:.3f} "
                  f"pooled={t['samp']['pooled']:.3f} lrmse={t['samp']['log_rmse']:.3f} | +free20k within="
                  f"{t['free20k_samp']['within']:.3f} pooled={t['free20k_samp']['pooled']:.3f}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

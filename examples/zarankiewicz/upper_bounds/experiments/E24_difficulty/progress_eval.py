"""E24 / A4: evaluate family-A / family-B features (progress_features_*.jsonl) against true d.

  1. per-feature Spearman with true d in the HARD (d > 20k, tier 20k) and MID (2k < d <= 20k,
     tier 2k) regimes: pooled, per cell, mean within-cell, sign agreement across cells;
  2. log-linear model  log d = a + sum_k b_k z_k  on 1-4 features, leave-one-cell-out (LOCO):
       (a) NESTED: forward selection inside each training fold (inner LOCO on the training
           cells, criterion = inner log-RMSE), evaluated on the held-out cell -- honest;
       (b) FIXED sets named in FIXED_SETS (chosen after looking at the screening table, so
           their LOCO numbers are optimistic about feature choice, not about coefficients);
     baselines: current fhat (experiments/E11_calibration/calibrate_33.json, in-sample for the
     TRAIN cells), fhat-form refit LOCO (log c2000 + log2_volume), constant LOCO;
     metrics: held-out within-cell Spearman (mean over cells, weighted by n), pooled Spearman
     of the concatenated held-out predictions, log-RMSE (raw and clipped to the censored-label
     range [cap, 20 cap]);
  3. cost per tier (conflicts / propagations / seconds).

Writes progress_eval_<tag>.json and prints markdown tables.
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

UB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if UB not in sys.path:
    sys.path.insert(0, UB)

from zar_ub.difficulty import fhat, load_calibration  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_CELL = 8  # a cell enters within-cell statistics only with >= MIN_CELL cases
MAX_CORR = 0.95  # forward selection skips a candidate this correlated with a chosen feature


def _pfx(*ps):
    return lambda f: any(f.startswith(p) for p in ps)


# cost tiers: static = no solver; 2k = + 2k-conflict probes; free20k = what the existing censored
# table pipeline could record at zero extra solver cost (its pysat 2k and 20k runs + static);
# 20k = + the CaDiCaL-binary runs (another ~22k conflicts)
SUBSETS = {
    "hard": {
        "static": _pfx("st_"),
        "2k": _pfx("st_", "bin2k_", "ps2k_"),
        "free20k": _pfx("st_", "ps2k_", "ps20k_"),
        "20k": lambda f: True,
    },
    "mid": {"static": _pfx("st_"), "2k": lambda f: True},
}

FIXED_SETS = {
    "hard": [],  # filled after screening (see PROGRESS.md); empty = skip
    "mid": [],
}


def load(paths):
    """One or more (comma-separated) feature files; later records win on duplicate keys."""
    seen = {}
    for path in paths.split(","):
        for line in open(path):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if "features" not in o:
                continue
            seen[json.dumps([o["cell"], o["rows"], o["cols"], o["tier"]])] = o
    return list(seen.values())


def rho(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    return float(spearmanr(x, y).correlation)


def logrmse(pred, true):
    p = np.log(np.maximum(np.asarray(pred, float), 1.0))
    t = np.log(np.maximum(np.asarray(true, float), 1.0))
    return float(np.sqrt(np.mean((p - t) ** 2)))


def transform(name, x):
    """Monotone transform used by the model: log1p for count-like non-negative features."""
    x = np.asarray(x, float)
    if np.nanmin(x) >= 0 and np.nanmax(x) > 50:
        return np.log1p(x), "log1p"
    return x, "id"


# ---------------------------------------------------------------------------
def screen(recs, feats):
    y = np.array([float(r["d"]) for r in recs])
    cells = sorted(set(r["cell"] for r in recs))
    idx = {c: [i for i, r in enumerate(recs) if r["cell"] == c] for c in cells}
    out = {}
    for f in feats:
        x = np.array([r["features"].get(f, 0.0) for r in recs])
        pooled = rho(x, y)
        per = {}
        for c in cells:
            if len(idx[c]) >= MIN_CELL:
                per[c] = rho(x[idx[c]], y[idx[c]])
        vals = [(v, len(idx[c])) for c, v in per.items() if not math.isnan(v)]
        wmean = sum(v * n for v, n in vals) / sum(n for _, n in vals) if vals else float("nan")
        mean = float(np.mean([v for v, _ in vals])) if vals else float("nan")
        sgn = np.sign(wmean) if not math.isnan(wmean) else 0
        agree = sum(1 for v, _ in vals if np.sign(v) == sgn)
        out[f] = {
            "pooled": pooled,
            "within_mean": mean,
            "within_wmean": wmean,
            "per_cell": per,
            "sign_agree": f"{agree}/{len(vals)}",
        }
    return out


def fit_ols(X, y, ridge=1e-6):
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    A = np.hstack([np.ones((len(Z), 1)), Z])
    reg = ridge * np.eye(A.shape[1])
    reg[0, 0] = 0
    beta = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return beta, mu, sd


def pred_ols(model, X):
    beta, mu, sd = model
    Z = (X - mu) / sd
    return beta[0] + Z @ beta[1:]


def design(recs, feats):
    cols, trs = [], []
    for f in feats:
        x, tr = transform(f, [r["features"].get(f, 0.0) for r in recs])
        cols.append(x)
        trs.append(tr)
    return (np.vstack(cols).T if cols else np.zeros((len(recs), 0))), trs


def loco_fixed(recs, feats):
    """LOCO predictions (log d) for a fixed feature list."""
    y = np.log(np.array([float(r["d"]) for r in recs]))
    cells = sorted(set(r["cell"] for r in recs))
    X, _ = design(recs, feats)
    pred = np.zeros(len(recs))
    for c in cells:
        te = np.array([r["cell"] == c for r in recs])
        m = fit_ols(X[~te], y[~te])
        pred[te] = pred_ols(m, X[te])
    return pred


def forward_select(recs, cand, kmax, X_all=None):
    """Greedy forward selection by inner-LOCO log-RMSE on `recs` (the training cells)."""
    y = np.log(np.array([float(r["d"]) for r in recs]))
    cells = sorted(set(r["cell"] for r in recs))
    masks = [np.array([r["cell"] == c for r in recs]) for c in cells]
    if X_all is None:
        X_all = {f: design(recs, [f])[0][:, 0] for f in cand}
    chosen = []
    best_score = float("inf")
    for _ in range(kmax):
        best = None
        for f in cand:
            if f in chosen:
                continue
            # collinearity guard: skip a candidate nearly identical to a chosen feature
            # (e.g. ps20k_restarts vs ps20k_restarts_per_kconf differ only by the 20000..20006
            # conflict stopping noise; together they fit that noise with huge coefficients)
            xf = X_all[f]
            if np.std(xf) == 0 or any(
                np.std(X_all[g]) > 0 and abs(np.corrcoef(xf, X_all[g])[0, 1]) > MAX_CORR
                for g in chosen
            ):
                continue
            X = np.vstack([X_all[g] for g in chosen + [f]]).T
            err = 0.0
            for te in masks:
                if te.all() or (~te).sum() < 5:
                    continue
                m = fit_ols(X[~te], y[~te])
                err += float(np.sum((pred_ols(m, X[te]) - y[te]) ** 2))
            if best is None or err < best[0]:
                best = (err, f)
        if best is None or best[0] >= best_score * 0.995:  # stop when < 0.5 % improvement
            break
        best_score = best[0]
        chosen.append(best[1])
    return chosen


def loco_nested(recs, cand, kmax):
    y = np.log(np.array([float(r["d"]) for r in recs]))
    cells = sorted(set(r["cell"] for r in recs))
    pred = np.zeros(len(recs))
    picks = {}
    Xfull = {f: design(recs, [f])[0][:, 0] for f in cand}
    for c in cells:
        te = np.array([r["cell"] == c for r in recs])
        tr_recs = [r for r, t in zip(recs, te) if not t]
        Xtr = {f: Xfull[f][~te] for f in cand}
        # prefilter by |pooled within-cell rho| on training cells to keep the search small
        chosen = forward_select(tr_recs, cand, kmax, Xtr)
        picks[c] = chosen
        if chosen:
            X = np.vstack([Xfull[f] for f in chosen]).T
            m = fit_ols(X[~te], y[~te])
            pred[te] = pred_ols(m, X[te])
        else:
            pred[te] = y[~te].mean()
    return pred, picks


def metrics(recs, logpred, lo=None, hi=None):
    d = np.array([float(r["d"]) for r in recs])
    p = np.exp(logpred)
    cells = sorted(set(r["cell"] for r in recs))
    per = {}
    for c in cells:
        ix = [i for i, r in enumerate(recs) if r["cell"] == c]
        if len(ix) >= MIN_CELL:
            per[c] = {"n": len(ix), "rho": rho(p[ix], d[ix]), "logrmse": logrmse(p[ix], d[ix])}
    vals = [(v["rho"], v["n"]) for v in per.values() if not math.isnan(v["rho"])]
    out = {
        "n": len(recs),
        "pooled_rho": rho(p, d),
        "within_rho_mean": float(np.mean([v for v, _ in vals])) if vals else float("nan"),
        "within_rho_wmean": (
            (sum(v * n for v, n in vals) / sum(n for _, n in vals)) if vals else float("nan")
        ),
        "logrmse": logrmse(p, d),
        "per_cell": per,
    }
    if lo is not None:
        out["logrmse_clipped"] = logrmse(np.clip(p, lo, hi), d)
    return out


def current_fhat(recs, cap):
    coef = load_calibration(3, 3)
    return (
        np.log(
            np.array(
                [fhat(coef, int(r.get("c2000") or 2000), float(r["log2_volume"])) for r in recs]
            )
        ),
        coef,
    )


def evaluate(recs, regime, cap, kmax, curated_only):
    out = {"regime": regime, "n": len(recs), "cells": {}}
    for r in recs:
        out["cells"][r["cell"]] = out["cells"].get(r["cell"], 0) + 1
    feats = sorted(set().union(*[set(r["features"]) for r in recs]))
    feats = [f for f in feats if len(set(r["features"].get(f, 0.0) for r in recs)) > 1]
    if curated_only:
        feats = [f for f in feats if "_raw_" not in f]
    scr = screen(recs, feats)
    out["screen"] = scr
    lo, hi = cap, 20 * cap
    # baselines
    fh, coef = current_fhat(recs, cap)
    out["baseline_fhat_current"] = metrics(recs, fh, lo, hi)
    out["baseline_fhat_current"]["coef"] = coef
    base_recs = [
        dict(
            r,
            features={
                "lc2000": math.log(max(min(int(r.get("c2000") or 2000), 2000), 1)),
                "lv": float(r["log2_volume"]),
            },
        )
        for r in recs
    ]
    out["baseline_fhat_form_loco"] = metrics(recs, loco_fixed(base_recs, ["lc2000", "lv"]), lo, hi)
    out["baseline_const_loco"] = metrics(recs, loco_fixed(recs, []), lo, hi)
    # nested selection, per cost tier (feature subsets)
    ranked = sorted(
        [f for f in feats if not math.isnan(scr[f]["within_wmean"])],
        key=lambda f: -abs(scr[f]["within_wmean"]),
    )
    out["tiers"] = {}
    for tname, pred_f in SUBSETS[regime].items():
        sub = [f for f in feats if pred_f(f)]
        if not sub:
            continue
        # (a) pool = the 60 subset features with the largest |within-cell rho| on ALL cells
        #     (mild leak into the pool only; selection and coefficients are nested)
        pool = [f for f in ranked if f in set(sub)][:60]
        pred, picks = loco_nested(recs, pool, kmax)
        m1 = metrics(recs, pred, lo, hi)
        m1["picks"] = picks
        # (b) pool = every subset feature (no screening at all: fully honest)
        pred2, picks2 = loco_nested(recs, sub, kmax)
        m2 = metrics(recs, pred2, lo, hi)
        m2["picks"] = picks2
        out["tiers"][tname] = {"n_features": len(sub), "nested": m1, "nested_allpool": m2}
    out["nested"] = out["tiers"][list(SUBSETS[regime])[-1]]["nested"]
    out["nested_allpool"] = out["tiers"][list(SUBSETS[regime])[-1]]["nested_allpool"]
    for name, fs in (("fixed", FIXED_SETS.get(regime) or []),):
        if fs:
            out[name] = metrics(recs, loco_fixed(recs, fs), lo, hi)
            out[name]["features"] = fs
    # single-feature LOCO models for the top 15
    out["single"] = {}
    for f in ranked[:15]:
        out["single"][f] = metrics(recs, loco_fixed(recs, [f]), lo, hi)
    # cost
    out["cost"] = {
        "conflicts_mean": float(np.mean([r["cost_conflicts"] for r in recs])),
        "propagations_mean": float(np.mean([r["cost_propagations"] for r in recs])),
        "seconds_mean": float(np.mean([r["cost_seconds"] for r in recs])),
    }
    return out


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def report(ev):
    print(f"\n## {ev['regime'].upper()}  n={ev['n']}  cells={ev['cells']}")
    print(
        f"cost/case: {ev['cost']['conflicts_mean']:.0f} conflicts, {ev['cost']['propagations_mean']:.3g} propagations, {ev['cost']['seconds_mean']:.2f} s"
    )
    scr = ev["screen"]
    top = sorted(
        [f for f in scr if not math.isnan(scr[f]["within_wmean"])],
        key=lambda f: -abs(scr[f]["within_wmean"]),
    )[:40]
    print("\n| feature | within-cell rho (n-wtd) | mean | pooled | sign agree | per cell |")
    print("|---|---|---|---|---|---|")
    for f in top:
        s = scr[f]
        pc = ", ".join(f"{c.split('_s')[0]}:{fmt(v, 2)}" for c, v in s["per_cell"].items())
        print(
            f"| {f} | {fmt(s['within_wmean'])} | {fmt(s['within_mean'])} | {fmt(s['pooled'])} | {s['sign_agree']} | {pc} |"
        )
    print(
        "\n| model (LOCO) | within rho (n-wtd) | mean | pooled rho | log-RMSE | clipped log-RMSE |"
    )
    print("|---|---|---|---|---|---|")
    rowsm = [
        (k, ev[k])
        for k in (
            "baseline_fhat_current",
            "baseline_fhat_form_loco",
            "baseline_const_loco",
            "fixed",
        )
        if k in ev
    ]
    for t, v in ev["tiers"].items():
        rowsm += [
            (f"{t} nested (pool 60)", v["nested"]),
            (f"{t} nested (all {v['n_features']})", v["nested_allpool"]),
        ]
    for k, m in rowsm:
        print(
            f"| {k} | {fmt(m['within_rho_wmean'])} | {fmt(m['within_rho_mean'])} | {fmt(m['pooled_rho'])} | {fmt(m['logrmse'])} | {fmt(m.get('logrmse_clipped'))} |"
        )
    for f, m in ev["single"].items():
        print(
            f"| single {f} | {fmt(m['within_rho_wmean'])} | {fmt(m['within_rho_mean'])} | {fmt(m['pooled_rho'])} | {fmt(m['logrmse'])} | {fmt(m.get('logrmse_clipped'))} |"
        )
    for t, v in ev["tiers"].items():
        print(f"{t} nested picks:", v["nested"]["picks"])
        print(f"{t} nested_allpool picks:", v["nested_allpool"]["picks"])
    print(
        "baseline_fhat_current",
        {
            c.replace("_s3_t3", ""): (v["n"], fmt(v["rho"], 2), fmt(v["logrmse"], 2))
            for c, v in ev["baseline_fhat_current"]["per_cell"].items()
        },
    )
    for t, v in ev["tiers"].items():
        print(
            t,
            {
                c.replace("_s3_t3", ""): (x["n"], fmt(x["rho"], 2), fmt(x["logrmse"], 2))
                for c, x in v["nested"]["per_cell"].items()
            },
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=os.path.join(HERE, "progress_features_dev.jsonl"))
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--curated-only", action="store_true")
    ap.add_argument("--regimes", default="hard,mid")
    a = ap.parse_args()
    recs = load(a.feat)
    res = {}
    for reg, tier, cap in (("hard", "20k", 20_000), ("mid", "2k", 2_000)):
        if reg not in a.regimes.split(","):
            continue
        rr = [r for r in recs if r["regime"] == reg and r["tier"] == tier]
        if len(rr) < 10:
            continue
        ev = evaluate(rr, reg, cap, a.kmax, a.curated_only)
        report(ev)
        res[reg] = ev
    out = os.path.join(HERE, f"progress_eval_{a.tag}.json")
    json.dump(
        res,
        open(out, "w"),
        indent=1,
        default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o),
    )
    print("wrote", out)


if __name__ == "__main__":
    main()

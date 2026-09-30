"""E24 / A4: freeze the log-linear progress models used by zar_ub.hardness_progress.estimate's d_hat.

Each model:  log d_hat = intercept + sum_k coef_k * (T_k(x_k) - mu_k) / sd_k,  T = id | log1p,
d_hat = max(floor, exp(.)); features chosen by greedy forward selection (inner leave-one-cell-out
log-RMSE, <= KMAX features, stop at < 0.5 % improvement) on the DEV cells only.

  "20k"     : cases still open after the pysat 20k probe (HARD), all curated features (bin + ps + st + prog)
  "free20k" : same cases, only features the existing censored-table pipeline already has for free
              (pysat 2k and 20k run statistics + static/LP)
  "2k"      : cases still open after the pysat 2k probe (MID and HARD), 2k-tier curated features

Writes experiments/E24_difficulty/progress_model.json (with the dev cells it was fitted on).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from progress_eval import design, fit_ols, forward_select, load  # noqa: E402

KMAX = 4


def curated(f):
    return "_raw_" not in f


SPECS = {
    "20k": dict(regimes=("hard",), tier="20k", pred=lambda f: curated(f), floor=20_000),
    "free20k": dict(
        regimes=("hard",),
        tier="20k",
        pred=lambda f: f.startswith(("st_", "ps2k_", "ps20k_")),
        floor=20_000,
    ),
    "2k": dict(
        regimes=("hard", "mid"),
        tier=None,
        pred=lambda f: curated(f) and f.startswith(("st_", "ps2k_", "bin2k_")),
        floor=2_000,
    ),
}


def fit_spec(recs, spec):
    feats = sorted(set().union(*[set(r["features"]) for r in recs]))
    feats = [
        f
        for f in feats
        if spec["pred"](f) and len(set(r["features"].get(f, 0.0) for r in recs)) > 1
    ]
    X_all = {f: design(recs, [f])[0][:, 0] for f in feats}
    chosen = forward_select(recs, feats, KMAX, X_all)
    X, trs = design(recs, chosen)
    y = np.log(np.array([float(r["d"]) for r in recs]))
    beta, mu, sd = fit_ols(X, y)
    resid = y - (beta[0] + ((X - mu) / sd) @ beta[1:])
    return {
        "features": chosen,
        "transforms": trs,
        "intercept": float(beta[0]),
        "coef": [float(b) for b in beta[1:]],
        "mu": {f: float(m) for f, m in zip(chosen, mu)},
        "sd": {f: float(s) for f, s in zip(chosen, sd)},
        "floor": spec["floor"],
        "n_fit": len(recs),
        "cells": sorted(set(r["cell"] for r in recs)),
        "train_logrmse": float(np.sqrt(np.mean(resid**2))),
        # Duan smearing factor: E[d] ~= smear * exp(E[log d]) (use for sums of work, not for ranks)
        "smear": float(np.mean(np.exp(resid))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=os.path.join(HERE, "progress_features_dev.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "progress_model.json"))
    a = ap.parse_args()
    recs = load(a.feat)
    models = {}
    for name, spec in SPECS.items():
        rr = [
            r
            for r in recs
            if r["regime"] in spec["regimes"]
            and (spec["tier"] is None or r["tier"] == spec["tier"])
        ]
        # the 2k model uses the 2k-tier features, present in both the 2k and the 20k tier records
        m = fit_spec(rr, spec)
        m["tier"] = name
        models[name] = m
        print(
            name,
            m["n_fit"],
            m["features"],
            [round(c, 3) for c in m["coef"]],
            "train log-RMSE",
            round(m["train_logrmse"], 3),
        )
    out = {
        "formula": "log d_hat = intercept + sum_k coef_k (T_k(x_k) - mu_k)/sd_k ; T in {id, log1p}; d_hat = max(floor, exp(.))",
        "frozen": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": os.path.basename(a.feat),
        "models": models,
    }
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()

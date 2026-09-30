"""E28 independent re-computation of the headline difficulty metric (no zar_ub code imported for the metric).

Reads only saved data:
  experiments/E24_difficulty/features_evalset.jsonl   (cases, true d, status, regime)
  experiments/E24_difficulty/features_progress.jsonl  (pr:* features)
  experiments/E24_difficulty/features_lookahead.jsonl (la:* features)
  experiments/E24_difficulty/hardness_model.json      (adopted model: features, transforms)
  experiments/E11_calibration/calibrate_33.json       (legacy fhat coefficients)

Computes, on the HARD regime (2,238 exact d > 20k + 191 open at 2M), held out leave-one-shape-out:
  * adopted model: the fixed 6-feature log-linear form, refit by OLS on the other shapes (z-scored on the train fold);
  * adopted model as shipped (frozen coefficients; in-sample);
  * legacy label clip(fhat, 20k, 400k), fhat = exp(a + b log c2000 + g log2_volume);
metrics: mean within-cell Spearman over cells with >= 8 exact hard cases (NaN for constant predictions, excluded from the
mean and counted), within-cell Harrell C (open-at-2M as right-censored at 2M; prediction ties count 1/2), pooled C, log-RMSE.
Also the (12,13,87) single-holdout number of `calibrate --model --holdout 12,13,87`.
"""
import json, math, os, sys
from collections import defaultdict
import numpy as np

UB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E24 = os.path.join(UB, "experiments", "E24_difficulty")


def jl(name):
    with open(os.path.join(E24, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


def rankdata(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    sv = v[order]
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def harrell_c(pred, t, event):
    """pairs (i, j) with t_i < t_j and event_i: concordant if pred_i < pred_j."""
    num = den = 0.0
    n = len(pred)
    pred = np.asarray(pred, float); t = np.asarray(t, float); event = np.asarray(event, bool)
    for i in range(n):
        if not event[i]:
            continue
        mask = t > t[i]
        k = mask.sum()
        if k == 0:
            continue
        den += k
        pj = pred[mask]
        num += (pj > pred[i]).sum() + 0.5 * (pj == pred[i]).sum()
    return num / den if den else float("nan")


def main():
    ev = jl("features_evalset.jsonl")
    pr = {r["key"]: r["features"] for r in jl("features_progress.jsonl")}
    la = {r["key"]: r["features"] for r in jl("features_lookahead.jsonl")}
    model = json.load(open(os.path.join(E24, "hardness_model.json")))
    calib = json.load(open(os.path.join(UB, "experiments", "E11_calibration", "calibrate_33.json")))["fits"]["3,3"]
    a, b, g = calib["a"], calib["b"], calib["g"]

    hard = [r for r in ev if r["regime"] in ("hard", "hard_open")]
    feats = model["features"]

    def fval(r, name):
        fam, k = name.split(":", 1)
        src = pr if fam == "pr" else la
        x = float(src[r["key"]][k])
        return math.log1p(x) if model["transforms"][name] == "log1p" else x

    X = np.array([[fval(r, f) for f in feats] for r in hard])
    y = np.log(np.array([float(r["d"]) for r in hard]))
    exact = np.array([r["status"] == "unsat" for r in hard])
    shape = np.array(["%d,%d" % (r["m"], r["n"]) for r in hard])
    cell = np.array([r["cell"] for r in hard])
    assert np.isfinite(X).all()

    # legacy
    legacy = np.array([min(max(20000.0, math.exp(a + b * math.log(max(r["c2000"], 1)) + g * r["log2_volume"])), 400000.0)
                       for r in hard])
    legacy_log = np.log(legacy)

    # frozen shipped model (in-sample)
    mu = np.array([model["mu"][f] for f in feats]); sd = np.array([model["sd"][f] for f in feats])
    co = np.array([model["coef"][f] for f in feats])
    frozen_log = model["intercept"] + ((X - mu) / sd) @ co

    def fit_predict(train_idx, test_idx):
        Xtr = X[train_idx]; m_ = Xtr.mean(0); s_ = Xtr.std(0); s_[s_ == 0] = 1
        A = np.hstack([np.ones((len(train_idx), 1)), (Xtr - m_) / s_])
        beta, *_ = np.linalg.lstsq(A, y[train_idx], rcond=None)
        At = np.hstack([np.ones((len(test_idx), 1)), (X[test_idx] - m_) / s_])
        return At @ beta

    loso = np.full(len(hard), np.nan)
    for sh in sorted(set(shape)):
        te = np.where(shape == sh)[0]
        tr = np.where((shape != sh) & exact)[0]
        loso[te] = fit_predict(tr, te)

    def metrics(pred_log, name, idx=None):
        idx = np.arange(len(hard)) if idx is None else idx
        cells = defaultdict(list)
        for i in idx:
            cells[cell[i]].append(i)
        rhos, cs, const = [], [], 0
        percell = {}
        for c, ii in sorted(cells.items()):
            ii = np.array(ii)
            ex = ii[exact[ii]]
            if len(ex) < 8:
                continue
            rho = spearman(pred_log[ex], y[ex])
            cc = harrell_c(pred_log[ii], y[ii], exact[ii])
            percell[c] = (len(ex), rho, cc)
            if math.isnan(rho):
                const += 1
            else:
                rhos.append(rho)
            cs.append(cc)
        ex_all = idx[exact[idx]]
        rmse = float(np.sqrt(np.mean((pred_log[ex_all] - y[ex_all]) ** 2)))
        pooled = harrell_c(pred_log[idx], y[idx], exact[idx])
        return {"name": name, "n_cells": len(percell), "cells_constant": const,
                "within_rho_mean": float(np.mean(rhos)) if rhos else float("nan"),
                "within_C_mean": float(np.mean(cs)), "pooled_C": pooled, "log_rmse": rmse, "per_cell": percell}

    out = {}
    for pred, name in ((legacy_log, "legacy clip(fhat,20k,400k)"), (loso, "adopted model, LOSO refit (fixed 6 features)"),
                       (frozen_log, "adopted model, frozen (in-sample)")):
        out[name] = metrics(pred, name)
        tgt = np.array([i for i in range(len(hard)) if hard[i]["trust"] == "tan2022"])
        out[name + " | TARGET cells"] = metrics(pred, name + " | TARGET", tgt)
        surv = np.array([i for i in range(len(hard)) if not hard[i].get("baseline_lean_kill")])
        out[name + " | library survivors"] = metrics(pred, name + " | survivors", surv)

    # single holdout (12,13): fit on all other shapes' exact hard cases
    te = np.where(shape == "12,13")[0]; tr = np.where((shape != "12,13") & exact)[0]
    p = fit_predict(tr, te); ex = te[exact[te]]
    rows_1213 = {"n": int(len(ex)), "rho_model": spearman(p[exact[te]], y[ex]),
                 "rho_legacy": spearman(legacy_log[ex], y[ex]),
                 "rmse_model": float(np.sqrt(np.mean((p[exact[te]] - y[ex]) ** 2))),
                 "rmse_legacy": float(np.sqrt(np.mean((legacy_log[ex] - y[ex]) ** 2)))}

    lines = ["n hard = %d (exact %d, open %d)" % (len(hard), exact.sum(), (~exact).sum())]
    for k, v in out.items():
        lines.append("%-70s cells=%2d const=%d  within-rho=%.3f  within-C=%.3f  pooled-C=%.3f  logRMSE=%.3f" % (
            k, v["n_cells"], v["cells_constant"], v["within_rho_mean"], v["within_C_mean"], v["pooled_C"], v["log_rmse"]))
    lines.append("holdout (12,13): %s" % json.dumps(rows_1213))
    lines.append("per-cell (legacy | LOSO model):")
    L = out["legacy clip(fhat,20k,400k)"]["per_cell"]; M = out["adopted model, LOSO refit (fixed 6 features)"]["per_cell"]
    for c in sorted(M):
        lines.append("  %-28s n=%4d  legacy rho=%6.3f C=%.3f | model rho=%.3f C=%.3f" % (c, M[c][0], L[c][1], L[c][2], M[c][1], M[c][2]))
    txt = "\n".join(lines)
    print(txt)
    here = os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(here, "verify_difficulty.txt"), "w").write(txt + "\n")
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_cell"} for k, v in out.items()} | {"holdout_12_13": rows_1213},
              open(os.path.join(here, "verify_difficulty.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""E28: independent re-implementation of D5's NESTED protocol for the adopted tier (free = BASE + ST + LA + PS).

Outer leave-one-shape-out over the 17 shapes of the HARD set; inside each outer fold, greedy forward selection
(inner leave-one-shape-out OLS log-RMSE, <= 6 features, stop at < 0.5 % improvement, skip a candidate with
|r| > 0.95 to a selected feature), OLS on the selected features, predict the held-out shape.  Feature transforms are the
unsupervised rule described in EVALUATION.md (log1p if non-negative and max > 50, signed log1p if |x| > 1000, else id),
median imputation.  Exact hard cases train; open-at-2M cases enter only Harrell C (right-censored).
Written from the protocol description, not by importing eval_analyze.py.
"""
import json, math, os
from collections import defaultdict, Counter
import numpy as np

UB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E24 = os.path.join(UB, "experiments", "E24_difficulty")
EXCL_SUB = ("_wall", "_seconds", "model_log_d")
EXCL = {"la:log2_volume", "pr:st_log2_volume", "pr:ps2k_conflicts", "pr:ps20k_conflicts", "pr:ps2k_solved",
        "pr:ps20k_solved"}


def jl(n):
    return [json.loads(l) for l in open(os.path.join(E24, n)) if l.strip()]


def rank(v):
    v = np.asarray(v, float); o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); sv = v[o]; i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return r


def spear(x, y):
    a, b = rank(x), rank(y)
    return float("nan") if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def cidx(p, t, e):
    num = den = 0.0
    for i in range(len(p)):
        if not e[i]:
            continue
        m = t > t[i]; k = m.sum()
        if k:
            den += k; num += (p[m] > p[i]).sum() + 0.5 * (p[m] == p[i]).sum()
    return num / den if den else float("nan")


def main():
    ev = [r for r in jl("features_evalset.jsonl") if r["regime"] in ("hard", "hard_open")]
    pr = {r["key"]: r["features"] for r in jl("features_progress.jsonl")}
    la = {r["key"]: r["features"] for r in jl("features_lookahead.jsonl")}
    rows = []
    for r in ev:
        f = {"base:log2_volume": r["log2_volume"], "base:log_c2000": math.log(max(1, min(int(r["c2000"] or 2000), 2000))),
             "base:aspect": r["n"] / r["m"], "base:mn": float(r["m"] * r["n"])}
        for k, v in pr[r["key"]].items():
            if k.startswith("st_") or k.startswith("ps"):
                f["pr:" + k] = v
        for k, v in la[r["key"]].items():
            f["la:" + k] = v
        rows.append(f)
    names = sorted({n for f in rows for n in f if n not in EXCL and not any(s in n for s in EXCL_SUB)
                    and not n.startswith("la:frozen") and isinstance(rows[0].get(n, 0.0), (int, float, type(None)))})
    X = np.full((len(rows), len(names)), np.nan)
    for i, f in enumerate(rows):
        for j, n in enumerate(names):
            v = f.get(n)
            if isinstance(v, (int, float)) and v is not None and math.isfinite(float(v)):
                X[i, j] = float(v)
    for j in range(len(names)):
        col = X[:, j]; v = col[np.isfinite(col)]
        if len(v) == 0:
            continue
        mx = np.max(np.abs(v))
        if v.min() >= 0 and mx > 50:
            X[:, j] = np.log1p(np.maximum(col, 0))
        elif mx > 1000:
            X[:, j] = np.sign(col) * np.log1p(np.abs(col))
    med = np.nanmedian(X, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    keep = X.std(0) > 0
    X = X[:, keep]; names = [n for n, k in zip(names, keep) if k]
    y = np.log(np.array([float(r["d"]) for r in ev])); exact = np.array([r["status"] == "unsat" for r in ev])
    shape = np.array(["%d,%d" % (r["m"], r["n"]) for r in ev]); cell = np.array([r["cell"] for r in ev])
    print("pool size", len(names), "cases", len(ev))

    def inner_rmse(idx, cols):
        A = np.hstack([np.ones((len(idx), 1)), X[np.ix_(idx, cols)]]) if cols else np.ones((len(idx), 1))
        yy = y[idx]; g = shape[idx]; se = 0.0
        for s in np.unique(g):
            te = g == s; tr = ~te
            beta = np.linalg.lstsq(A[tr], yy[tr], rcond=None)[0]
            se += float(((A[te] @ beta - yy[te]) ** 2).sum())
        return math.sqrt(se / len(idx))

    def greedy(idx):
        C = np.corrcoef(X[idx].T)
        sel = []; cur = inner_rmse(idx, sel)
        while len(sel) < 6:
            best, bj = cur, None
            for j in range(len(names)):
                if j in sel or any(abs(C[j, s]) > 0.95 for s in sel):
                    continue
                v = inner_rmse(idx, sel + [j])
                if v < best:
                    best, bj = v, j
            if bj is None or best > cur * 0.995:
                break
            sel.append(bj); cur = best
        return sel

    pred = np.full(len(ev), np.nan); chosen = {}
    for s in sorted(set(shape)):
        te = np.where(shape == s)[0]; tr = np.where((shape != s) & exact)[0]
        sel = greedy(tr); chosen[s] = [names[j] for j in sel]
        mu = X[np.ix_(tr, sel)].mean(0); sd = X[np.ix_(tr, sel)].std(0); sd[sd == 0] = 1
        A = np.hstack([np.ones((len(tr), 1)), (X[np.ix_(tr, sel)] - mu) / sd])
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred[te] = np.hstack([np.ones((len(te), 1)), (X[np.ix_(te, sel)] - mu) / sd]) @ beta
        print(s, chosen[s], flush=True)
    rhos, cs = [], []
    for c in sorted(set(cell)):
        ii = np.where(cell == c)[0]; ex = ii[exact[ii]]
        if len(ex) < 8:
            continue
        rhos.append(spear(pred[ex], y[ex])); cs.append(cidx(pred[ii], y[ii], exact[ii]))
    ex = exact
    out = {"within_rho": float(np.mean(rhos)), "within_C": float(np.mean(cs)), "n_cells": len(rhos),
           "pooled_C": cidx(pred, y, exact), "log_rmse": float(np.sqrt(np.mean((pred[ex] - y[ex]) ** 2))),
           "feature_frequency": Counter(n for v in chosen.values() for n in v).most_common(12), "chosen": chosen}
    print(json.dumps({k: v for k, v in out.items() if k != "chosen"}, indent=1))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_nested.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

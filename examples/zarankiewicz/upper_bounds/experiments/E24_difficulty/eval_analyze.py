"""E24 / D5-evaluate: the decisive comparison of the difficulty estimators.

Inputs:  features_evalset.jsonl (eval_collect.py), features_{lookahead,sampling,progress}.jsonl,
         features_gaintables.jsonl (eval_masks.py).
Outputs: eval_results.json (every number in EVALUATION.md), features_predictions.jsonl (held-out
         predictions of every model), eval_results.txt (human-readable tables).

Protocol (no tuning on held-out cells):
  * Training rows for the HARD models = exact hard cases (20k < d, refuted/satisfied within 2M)
    of the training groups.  Cases open at 2M are never fitted; they are scored by Harrell's C
    as right-censored at 2,000,000.
  * CV: leave-one-SHAPE-out (group = (m, n); a stricter version of leave-one-cell-out since cells
    of the same shape and different w or trust share a fold), square->wide (train on n/m < 1.4,
    test on n/m >= 1.4) and wide->square.  Feature selection (single-best, greedy) is NESTED:
    it only sees the training groups of each fold.
  * Metrics (HARD): mean within-cell Spearman on exact rows (cells with >= 8 exact hard rows),
    pooled Spearman, Harrell's C (within-cell mean and pooled, open-at-2M = censored), log-RMSE on
    exact rows.  MID (2k < d <= 20k) models use only features that do not reveal d (2k tier).
  * Reward-gain error: target-mode labelling is simulated on every gain table (exact d when the
    case finishes within 20k conflicts, else the estimator's held-out value, clipped to
    [20k, ceiling]); gain_I and tail_I of real and random masks are computed with estimated and
    with true d and compared.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict

import warnings

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UB)
sys.path.insert(0, HERE)

from zar_ub import difficulty as zd  # noqa: E402

CAP = 20000
OPEN_D = 2_000_000
WIDE = 1.4
MIN_CELL = 8
SEED = 0
OUT_JSON = os.path.join(HERE, "eval_results.json")
OUT_TXT = os.path.join(HERE, "eval_results.txt")
OUT_PRED = os.path.join(HERE, "features_predictions.jsonl")

EXCLUDE_SUBSTR = ("_wall", "_seconds", "model_log_d")
EXCLUDE_EXACT = {"la:log2_volume", "pr:st_log2_volume", "pr:ps2k_conflicts", "pr:ps20k_conflicts",
                 "pr:ps2k_solved", "pr:ps20k_solved", "sa:log2_space"}


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load_jsonl(p):
    with open(p) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def family(name: str) -> str:
    if name.startswith("la:"):
        return "LA"
    if name.startswith("sa:"):
        return "SA"
    if name.startswith("pr:st_"):
        return "ST"
    if name.startswith("pr:ps"):
        return "PS"
    if name.startswith("pr:bin") or name.startswith("pr:prog"):
        return "BIN"
    return "BASE"


def load_data():
    ev = load_jsonl(os.path.join(HERE, "features_evalset.jsonl"))
    feats = {e: {} for e in ("lookahead", "sampling", "progress")}
    costs = {e: {} for e in feats}
    costs["progress_pysat"], costs["progress_binary"] = {}, {}
    errors = defaultdict(int)
    for e, pre in (("lookahead", "la:"), ("sampling", "sa:"), ("progress", "pr:")):
        p = os.path.join(HERE, f"features_{e}.jsonl")
        if not os.path.exists(p):
            continue
        for o in load_jsonl(p):
            if "error" in o:
                errors[e] += 1
                continue
            f = {pre + k: float(v) for k, v in o["features"].items() if isinstance(v, (int, float))}
            if e == "sampling":
                raw = o.get("d_hat_raw")
                f["sa:log_mu"] = math.log(max(float(raw), 1.0)) if raw else float("nan")
                f["sa:log_dhat_frozen"] = math.log(max(float(o["d_hat"]), 1.0)) if o.get("d_hat") else float("nan")
            if e == "progress" and o.get("d_hat"):
                f["pr:log_dhat_frozen"] = math.log(max(float(o["d_hat"]), 1.0))
            if e == "lookahead" and o["features"].get("model_log_d_hard") is not None:
                f["la:frozen_log_d_hard"] = float(o["features"]["model_log_d_hard"])
            feats[e][o["key"]] = f
            costs[e][o["key"]] = (o["cost_conflicts"], o["cost_propagations"], o["cost_seconds"])
            if e == "progress":
                runs = (o.get("meta") or {}).get("runs") or {}
                for part, ks in (("progress_pysat", ("ps2k", "ps20k")), ("progress_binary", ("bin2k", "bin20k"))):
                    c = sum(int(runs.get(k, {}).get("conflicts", 0)) for k in ks)
                    pp = sum(int(runs.get(k, {}).get("propagations", 0)) for k in ks)
                    wl = sum(float(runs.get(k, {}).get("wall", 0.0)) for k in ks)
                    costs.setdefault(part, {})[o["key"]] = (c, pp, wl)
    coef = zd.load_calibration(3, 3)
    rows = []
    for r in ev:
        k = r["key"]
        f = {"base:log2_volume": float(r["log2_volume"]),
             "base:log_c2000": math.log(max(1, min(int(r["c2000"] or 2000), 2000))),
             "base:aspect": r["n"] / r["m"], "base:mn": float(r["m"] * r["n"])}
        have = {}
        for e in feats:
            have[e] = k in feats[e]
            f.update(feats[e].get(k, {}))
        fh = zd.fhat(coef, int(r["c2000"] or 2000), float(r["log2_volume"]))
        r2 = dict(r)
        r2.update({"f": f, "have": have, "group": f"m{r['m']}_n{r['n']}", "wide": r["n"] / r["m"] >= WIDE,
                   "exact": r["status"] != "unknown", "logd": math.log(float(r["d"])),
                   "fhat": fh, "current": min(max(CAP, fh), 20 * CAP)})
        rows.append(r2)
    return rows, costs, dict(errors)


def feature_names(rows, fams=None):
    names = set()
    for r in rows:
        names.update(r["f"].keys())
    out = []
    for n in sorted(names):
        if n in EXCLUDE_EXACT or any(s in n for s in EXCLUDE_SUBSTR):
            continue
        if n.startswith("la:frozen") or n.endswith("dhat_frozen"):
            continue
        if fams is not None and family(n) not in fams:
            continue
        out.append(n)
    return out


def build_transforms(rows, names):
    """Unsupervised monotone transform per feature: log1p for large non-negative counts,
    signed log1p for large signed values, identity otherwise (fixed on the whole eval set,
    labels unused)."""
    tr = {}
    for n in names:
        v = np.array([r["f"].get(n, np.nan) for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            tr[n] = "id"
            continue
        mx = np.max(np.abs(v))
        if np.min(v) >= 0 and mx > 50:
            tr[n] = "log1p"
        elif mx > 1000:
            tr[n] = "slog1p"
        else:
            tr[n] = "id"
    return tr


def matrix(rows, names, tr):
    X = np.full((len(rows), len(names)), np.nan)
    for i, r in enumerate(rows):
        f = r["f"]
        for j, n in enumerate(names):
            x = f.get(n)
            if x is None or not np.isfinite(x):
                continue
            t = tr.get(n, "id")
            if t == "log1p":
                x = math.log1p(max(x, 0.0))
            elif t == "slog1p":
                x = math.copysign(math.log1p(abs(x)), x)
            X[i, j] = x
    return X


def impute(X):
    """Column-median imputation over the whole matrix (unsupervised, labels unused)."""
    med = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(X), X, med)


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def harrell_c(pred, d, exact):
    pred, d, exact = np.asarray(pred, float), np.asarray(d, float), np.asarray(exact, bool)
    n = len(pred)
    if n < 2:
        return float("nan")
    di, dj = d[:, None], d[None, :]
    comp = (di < dj) & exact[:, None]
    if not comp.any():
        return float("nan")
    pi, pj = pred[:, None], pred[None, :]
    conc = ((pi < pj) & comp).sum() + 0.5 * ((pi == pj) & comp).sum()
    return float(conc / comp.sum())


def sp(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def metrics(recs, logpred):
    """recs: rows (hard regime, held-out), logpred: array of natural-log predictions."""
    logpred = np.asarray(logpred, float)
    ok = np.isfinite(logpred)
    ex = np.array([r["exact"] for r in recs]) & ok
    d = np.array([float(r["d"]) if r["exact"] else OPEN_D for r in recs])
    logd = np.log(d)
    out = {"n": int(ok.sum()), "n_exact": int(ex.sum()), "n_open": int((ok & ~ex).sum())}
    out["pooled_rho"] = sp(logpred[ex], logd[ex])
    out["log_rmse"] = float(np.sqrt(np.mean((logpred[ex] - logd[ex]) ** 2))) if ex.any() else float("nan")
    out["log_rmse_floored"] = float(np.sqrt(np.mean((np.maximum(logpred[ex], math.log(CAP)) - logd[ex]) ** 2))) if ex.any() else float("nan")
    out["bias"] = float(np.mean(logpred[ex] - logd[ex])) if ex.any() else float("nan")
    out["C_pooled"] = harrell_c(logpred[ok], d[ok], ex[ok])
    cells = defaultdict(list)
    for i, r in enumerate(recs):
        if ok[i]:
            cells[r["cell"]].append(i)
    per, perC, wts = {}, {}, {}
    for c, idx in cells.items():
        idx = np.array(idx)
        e = idx[ex[idx]]
        if len(e) >= MIN_CELL:
            per[c] = sp(logpred[e], logd[e])
            wts[c] = len(e)
        if len(idx) >= MIN_CELL:
            perC[c] = harrell_c(logpred[idx], d[idx], ex[idx])
    vals = [v for v in per.values() if np.isfinite(v)]
    out["within_rho"] = float(np.mean(vals)) if vals else float("nan")
    out["within_rho_w"] = float(sum(per[c] * wts[c] for c in per if np.isfinite(per[c])) /
                                max(1, sum(wts[c] for c in per if np.isfinite(per[c])))) if vals else float("nan")
    cv = [v for v in perC.values() if np.isfinite(v)]
    out["C_within"] = float(np.mean(cv)) if cv else float("nan")
    out["per_cell_rho"] = {c: round(v, 3) for c, v in sorted(per.items())}
    out["per_cell_C"] = {c: round(v, 3) for c, v in sorted(perC.items())}
    # top-decile recall within cell (exact + open, true order with open at the top)
    rec = []
    for c, idx in cells.items():
        if len(idx) < 20:
            continue
        idx = np.array(idx)
        k = max(1, len(idx) // 10)
        true_top = set(idx[np.lexsort((idx, -d[idx]))][:k])
        pred_top = set(idx[np.lexsort((idx, -logpred[idx]))][:k])
        rec.append(len(true_top & pred_top) / k)
    out["top10_recall"] = float(np.mean(rec)) if rec else float("nan")
    return out


# ----------------------------------------------------------------------------------------------
# models
# ----------------------------------------------------------------------------------------------
class OLS:
    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, X, y):
        Z = X[:, self.cols]
        self.med = np.nanmedian(Z, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        Z = np.where(np.isfinite(Z), Z, self.med)
        self.mu = Z.mean(0)
        self.sd = Z.std(0)
        self.sd[self.sd == 0] = 1.0
        Zs = (Z - self.mu) / self.sd
        A = np.hstack([np.ones((len(y), 1)), Zs])
        self.beta = np.linalg.lstsq(A, y, rcond=None)[0]
        res = y - A @ self.beta
        self.smear = float(np.mean(np.exp(res)))
        return self

    def predict(self, X):
        Z = X[:, self.cols]
        Z = np.where(np.isfinite(Z), Z, self.med)
        Zs = (Z - self.mu) / self.sd
        return self.beta[0] + Zs @ self.beta[1:]


def within_rho_train(X, y, groups, j):
    """mean within-group Spearman of feature j with y (training groups only)."""
    vals = []
    for g in np.unique(groups):
        idx = groups == g
        if idx.sum() >= MIN_CELL:
            x = X[idx, j]
            ok = np.isfinite(x)
            if ok.sum() >= MIN_CELL:
                v = sp(x[ok], y[idx][ok])
                if np.isfinite(v):
                    vals.append(v)
    return float(np.mean(vals)) if vals else 0.0


class SingleBest:
    """Nested: pick the feature with the largest |mean within-cell Spearman| on the training
    rows, then fit a log-linear (OLS) or isotonic map."""

    def __init__(self, pool, iso=False):
        self.pool, self.iso = list(pool), iso

    def fit(self, X, y, cells):
        best, bj = -1.0, None
        for j in self.pool:
            v = abs(within_rho_train(X, y, cells, j))
            if v > best:
                best, bj, self.sign = v, j, np.sign(within_rho_train(X, y, cells, j)) or 1.0
        self.j = bj
        if self.iso:
            from sklearn.isotonic import IsotonicRegression
            x = X[:, bj]
            self.med = float(np.nanmedian(x))
            x = np.where(np.isfinite(x), x, self.med)
            self.m = IsotonicRegression(increasing=bool(self.sign > 0), out_of_bounds="clip").fit(x, y)
            self.smear = float(np.mean(np.exp(y - self.m.predict(x))))
        else:
            self.m = OLS([bj]).fit(X, y)
            self.smear = self.m.smear
        return self

    def predict(self, X):
        if self.iso:
            x = X[:, self.j]
            return self.m.predict(np.where(np.isfinite(x), x, self.med))
        return self.m.predict(X)


def inner_loso_rmse(X, y, groups, cols, _cache={}):
    """fast inner leave-one-group-out OLS log-RMSE (X is imputed; no NaN)."""
    A = np.hstack([np.ones((len(y), 1)), X[:, cols]]) if cols else np.ones((len(y), 1))
    se, n = 0.0, 0
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if tr.sum() < 10:
            continue
        beta = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        p = A[te] @ beta
        se += float(np.sum((p - y[te]) ** 2))
        n += int(te.sum())
    return math.sqrt(se / max(1, n))


class Greedy:
    """Nested greedy forward selection (inner leave-one-shape-out log-RMSE on the training
    groups), <= max_k features, stop at < 0.5 % improvement, collinearity guard |r| > 0.95."""

    def __init__(self, pool, max_k=6, forced=()):
        self.pool, self.max_k, self.forced = list(pool), max_k, list(forced)

    def fit(self, X, y, cells, groups):
        sel = list(self.forced)
        cur = inner_loso_rmse(X, y, groups, sel) if sel else float(np.std(y)) * 1.0
        if not sel:  # constant model inner rmse
            cur = inner_loso_rmse(X, y, groups, [])
        Xf = X
        sd_all = Xf.std(0)
        C = np.corrcoef(Xf[:, self.pool].T) if len(self.pool) > 1 else np.ones((1, 1))
        pos = {j: k for k, j in enumerate(self.pool)}
        while len(sel) < self.max_k:
            best, bj = cur, None
            for j in self.pool:
                if j in sel:
                    continue
                if sd_all[j] == 0:
                    continue
                if any(s in pos and abs(C[pos[j], pos[s]]) > 0.95 for s in sel):
                    continue
                v = inner_loso_rmse(X, y, groups, sel + [j])
                if v < best:
                    best, bj = v, j
            if bj is None or best > cur * 0.995:
                break
            sel.append(bj)
            cur = best
        self.sel = sel
        self.m = OLS(sel).fit(X, y)
        self.smear = self.m.smear
        return self

    def predict(self, X):
        return self.m.predict(X)


def _ols_empty_fix():
    # OLS with no columns = constant model
    orig = OLS.fit

    def fit(self, X, y):
        if not self.cols:
            self.beta = np.array([float(np.mean(y))])
            self.mu = self.sd = self.med = np.zeros(0)
            self.smear = float(np.mean(np.exp(y - self.beta[0])))
            return self
        return orig(self, X, y)
    OLS.fit = fit


_ols_empty_fix()


class Tree:
    def __init__(self, cols, kind="gbm"):
        self.cols, self.kind = list(cols), kind

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        Z = X[:, self.cols]
        if self.kind == "gbm":
            self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                   min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)
            self.m.fit(Z, y)
        else:
            self.med = np.nanmedian(Z, axis=0)
            self.med = np.where(np.isfinite(self.med), self.med, 0.0)
            Z = np.where(np.isfinite(Z), Z, self.med)
            self.m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.33,
                                           random_state=SEED, n_jobs=8)
            self.m.fit(Z, y)
        self.smear = float(np.mean(np.exp(y - self.predict(X))))
        return self

    def predict(self, X):
        Z = X[:, self.cols]
        if self.kind != "gbm":
            Z = np.where(np.isfinite(Z), Z, self.med)
        return self.m.predict(Z)


# ----------------------------------------------------------------------------------------------
# cross-validation
# ----------------------------------------------------------------------------------------------
def folds(rows, scheme):
    groups = sorted({r["group"] for r in rows})
    if scheme == "loso":
        for g in groups:
            yield g, [r["group"] != g for r in rows], [r["group"] == g for r in rows]
    elif scheme == "sq2wide":
        yield "sq2wide", [not r["wide"] for r in rows], [r["wide"] for r in rows]
    elif scheme == "wide2sq":
        yield "wide2sq", [r["wide"] for r in rows], [not r["wide"] for r in rows]


def run_model(make, rows, X, scheme, need=()):
    """Held-out log predictions (NaN where not predicted) + smear per row + selections."""
    pred = np.full(len(rows), np.nan)
    smear = np.full(len(rows), 1.0)
    y = np.array([r["logd"] for r in rows])
    ex = np.array([r["exact"] for r in rows])
    avail = np.array([all(r["have"][e] for e in need) for r in rows])
    cells = np.array([r["cell"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    sels = []
    for name, tr, te in folds(rows, scheme):
        tr = np.array(tr) & ex & avail
        te = np.array(te) & avail
        if tr.sum() < 20 or te.sum() == 0:
            continue
        m = make()
        if isinstance(m, Greedy):
            m.fit(X[tr], y[tr], cells[tr], groups[tr])
            sels.append([int(j) for j in m.sel])
        elif isinstance(m, SingleBest):
            m.fit(X[tr], y[tr], cells[tr])
            sels.append([int(m.j)])
        else:
            m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        smear[te] = m.smear
    return pred, smear, sels


# ----------------------------------------------------------------------------------------------
# gain simulation
# ----------------------------------------------------------------------------------------------
def weighted_tail(idx_sorted_w, d, w):
    W = sum(w)
    need = W / 10.0
    acc, H = 0.0, []
    for i in idx_sorted_w:
        H.append(i)
        acc += w[i]
        if acc >= need:
            break
    return H


def gain_tail(d, w, mask):
    d, w, mask = np.asarray(d, float), np.asarray(w, float), np.asarray(mask, bool)
    tot = float(np.sum(w * d))
    g = float(np.sum((w * d)[mask]) / tot) if tot > 0 else 0.0
    order = sorted(range(len(d)), key=lambda i: (-d[i], i))
    H = weighted_tail(order, d, w)
    th = float(sum(w[i] * d[i] for i in H))
    t = float(sum(w[i] * d[i] for i in H if mask[i]) / th) if th > 0 else 0.0
    return g, t


def mask_class(name):
    if name.startswith("rand"):
        return name.split("_")[0]
    if name.startswith("farkas"):
        return "farkas"
    if name.startswith("rule_"):
        return "rule"
    return name


def gain_eval(gtabs, estmap, ceilings=(400_000, 2_000_000, None), smearmap=None):
    """estmap: key -> estimated d (unclipped, natural units) for hard rows.
    Returns {ceiling: {"by_class": {...}, "tables": {...}}}."""
    res = {}
    for ceil in ceilings:
        per_class = defaultdict(list)
        per_class_t = defaultdict(list)
        tabs = {}
        missing = 0
        for T in gtabs:
            U = T["universe"]
            d_true = [u["d_true"] for u in U]
            w = [u["w"] for u in U]
            d_est = []
            for u in U:
                if u["regime"] in ("hard", "hard_open"):
                    v = estmap.get(u["key"])
                    if v is None or not np.isfinite(v):
                        missing += 1
                        v = CAP
                    v = max(CAP, v)
                    if ceil is not None:
                        v = min(v, ceil)
                    d_est.append(v)
                else:
                    d_est.append(u["d_true"])
            tt = {}
            for mname, mk in T["masks"].items():
                if not any(mk):
                    continue
                gt_, tt_ = gain_tail(d_true, w, mk)
                ge_, te_ = gain_tail(d_est, w, mk)
                c = mask_class(mname)
                per_class[c].append(abs(ge_ - gt_))
                per_class_t[c].append(abs(te_ - tt_))
                if not mname.startswith("rand"):
                    tt[mname] = {"g_true": round(gt_, 4), "g_est": round(ge_, 4), "tail_true": round(tt_, 4),
                                 "tail_est": round(te_, 4), "kills": int(sum(mk))}
            tabs[T["cell"]] = tt
        real = [v for c in ("argD", "DGH", "farkas", "rule") for v in per_class.get(c, [])]
        realt = [v for c in ("argD", "DGH", "farkas", "rule") for v in per_class_t.get(c, [])]
        rnd = [v for c in per_class if c.startswith("rand") for v in per_class[c]]
        res[str(ceil)] = {
            "gain_mae": {c: [float(np.mean(v)), len(v)] for c, v in sorted(per_class.items())},
            "tail_mae": {c: [float(np.mean(v)), len(v)] for c, v in sorted(per_class_t.items())},
            "gain_mae_real": float(np.mean(real)) if real else float("nan"),
            "tail_mae_real": float(np.mean(realt)) if realt else float("nan"),
            "gain_mae_random": float(np.mean(rnd)) if rnd else float("nan"),
            "n_real": len(real), "missing": missing, "tables": tabs,
        }
    return res


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main():
    t0 = time.time()
    rows_all, costs, errors = load_data()
    print("errors", errors, flush=True)
    hard = [r for r in rows_all if r["regime"] in ("hard", "hard_open")]
    mid = [r for r in rows_all if r["regime"] == "mid"]
    names = feature_names(rows_all)
    tr = build_transforms(rows_all, names)
    idx = {n: j for j, n in enumerate(names)}
    XH = impute(matrix(hard, names, tr))
    fam_of = {n: family(n) for n in names}

    def pool(fams, extra_exclude=()):
        return [idx[n] for n in names if fam_of[n] in fams and n not in extra_exclude]

    NEED = {"LA": "lookahead", "SA": "sampling", "ST": "progress", "PS": "progress", "BIN": "progress"}

    def need_of(fams):
        return tuple(sorted({NEED[f] for f in fams if f in NEED}))

    TIERS = {
        "LA": ("BASE", "LA"),
        "SA": ("BASE", "SA"),
        "ST": ("BASE", "ST"),
        "PS": ("BASE", "PS"),
        "BIN": ("BASE", "BIN"),
        "noprobe(ST+LA)": ("BASE", "ST", "LA"),
        "free(ST+LA+PS)": ("BASE", "ST", "LA", "PS"),
        "free+SA": ("BASE", "ST", "LA", "PS", "SA"),
        "free+BIN": ("BASE", "ST", "LA", "PS", "BIN"),
        "all": ("BASE", "ST", "LA", "PS", "SA", "BIN"),
        "all-LA": ("BASE", "ST", "PS", "SA", "BIN"),
        "all-SA": ("BASE", "ST", "LA", "PS", "BIN"),
        "all-ST": ("BASE", "LA", "PS", "SA", "BIN"),
        "all-PS": ("BASE", "ST", "LA", "SA", "BIN"),
        "all-BIN": ("BASE", "ST", "LA", "PS", "SA"),
    }
    # pre-specified combinations (names)
    A4_FREE = ["pr:ps20k_decs_per_conf", "pr:ps20k_restarts_per_kconf", "pr:st_distinct_rows", "pr:st_row_lp_lcap"]
    FL = ["la:fl_free_cells", "la:fl_log2vol_rows"]
    COMBOS = {
        "fhat_refit": (["base:log_c2000", "base:log2_volume"], ()),
        "A4_free20k": (A4_FREE, ("progress",)),
        "A4_free20k+FL": (A4_FREE + FL, ("progress", "lookahead")),
        "A4_free20k+samp": (A4_FREE + ["sa:log_mu"], ("progress", "sampling")),
        "A4_free20k+FL+samp": (A4_FREE + FL + ["sa:log_mu"], ("progress", "lookahead", "sampling")),
        "A2_FL_only": (["base:log2_volume"] + FL, ("lookahead",)),
        "samp_ols": (["sa:log_mu"], ("sampling",)),
        "samp+vol": (["sa:log_mu", "base:log2_volume"], ("sampling",)),
    }
    for k, (ns, _) in COMBOS.items():
        for n in ns:
            assert n in idx, (k, n)

    gpath0 = os.path.join(HERE, "features_gaintables.jsonl")
    surv_keys = set()
    if os.path.exists(gpath0):
        for T in load_jsonl(gpath0):
            if T.get("universe_kind") == "survivors":
                surv_keys.update(u["key"] for u in T["universe"])
    results = {"n_hard": len(hard), "n_hard_open": sum(r["regime"] == "hard_open" for r in hard),
               "n_mid": len(mid), "errors": errors, "models": {}, "gain": {}, "costs": {}, "mid": {}}
    preds = {}  # (scheme, model) -> (pred, smear)
    yH = np.array([r["logd"] for r in hard])

    def record(scheme, name, pred, smear, sels=None, need=()):
        preds[(scheme, name)] = (pred, smear)
        m = metrics(hard, pred)
        test = [i for i, r in enumerate(hard) if np.isfinite(pred[i])]
        if scheme != "loso":  # restrict to the test side
            m = metrics([hard[i] for i in test], pred[test])
        tg = [i for i in test if hard[i]["trust"] == "tan2022"]
        if tg:
            m["target"] = {k: v for k, v in metrics([hard[i] for i in tg], pred[tg]).items()
                           if not k.startswith("per_cell")}
        sv = [i for i in test if hard[i]["key"] in surv_keys]
        if sv and scheme == "loso":
            m["survivors"] = {k: v for k, v in metrics([hard[i] for i in sv], pred[sv]).items()
                              if not k.startswith("per_cell")}
        wd = [i for i in test if hard[i]["wide"]]
        if wd and scheme == "loso":
            m["wide"] = {k: v for k, v in metrics([hard[i] for i in wd], pred[wd]).items()
                         if not k.startswith("per_cell")}
        if sels:
            cnt = defaultdict(int)
            for s in sels:
                for j in s:
                    cnt[names[j]] += 1
            m["selected"] = dict(sorted(cnt.items(), key=lambda kv: -kv[1]))
        m["need"] = list(need)
        results["models"].setdefault(scheme, {})[name] = m
        print(f"[{time.time() - t0:6.0f}s] {scheme:8s} {name:28s} within {fmt(m['within_rho'])} pooled {fmt(m['pooled_rho'])} "
              f"C_w {fmt(m['C_within'])} C_p {fmt(m['C_pooled'])} rmse {fmt(m['log_rmse'])} n {m['n']}", flush=True)

    for scheme in ("loso", "sq2wide", "wide2sq"):
        # fixed baselines (no fitting): the current label and the raw fhat
        te_mask = np.ones(len(hard), bool)
        if scheme == "sq2wide":
            te_mask = np.array([r["wide"] for r in hard])
        elif scheme == "wide2sq":
            te_mask = np.array([not r["wide"] for r in hard])
        cur = np.where(te_mask, np.log([r["current"] for r in hard]), np.nan)
        record(scheme, "current_label", cur, np.ones(len(hard)))
        rawf = np.where(te_mask, np.log([r["fhat"] for r in hard]), np.nan)
        record(scheme, "fhat_raw", rawf, np.ones(len(hard)))
        c20 = np.where(te_mask, math.log(CAP), np.nan)
        record(scheme, "c20000_probe", c20, np.ones(len(hard)))
        # frozen as-shipped module d_hats (fitted on DEV / 8 cells: partly in-sample!)
        for nm, fn in (("frozen_samp_dhat", "sa:log_dhat_frozen"), ("frozen_progress_dhat", "pr:log_dhat_frozen"),
                       ("frozen_lookahead_hard", "la:frozen_log_d_hard")):
            v = np.array([r["f"].get(fn, np.nan) for r in hard])
            record(scheme, nm, np.where(te_mask, v, np.nan), np.ones(len(hard)))
        # constant
        p, s, _ = run_model(lambda: OLS([]), hard, XH, scheme)
        record(scheme, "constant", p, s)
        # sampling unit-slope calibration
        j = idx["sa:log_mu"]

        class Unit:
            def fit(self, X, y):
                r_ = y - X[:, j]
                r_ = r_[np.isfinite(r_)]
                self.a = float(np.median(r_))
                self.smear = float(np.mean(np.exp(r_ - self.a)))
                return self

            def predict(self, X):
                return X[:, j] + self.a
        p, s, _ = run_model(Unit, hard, XH, scheme, need=("sampling",))
        record(scheme, "samp_unit_calib", p, s, need=("sampling",))
        for cname, (ns, need) in COMBOS.items():
            cols = [idx[n] for n in ns]
            p, s, _ = run_model(lambda cols=cols: OLS(cols), hard, XH, scheme, need=need)
            record(scheme, cname, p, s, need=need)
        # single best per family (nested), log-linear and isotonic
        for fam_name, fams in (("LA", ("LA",)), ("SA", ("SA",)), ("ST", ("ST",)), ("PS", ("PS",)),
                               ("BIN", ("BIN",)), ("any-free", ("BASE", "ST", "LA", "PS")),
                               ("any", ("BASE", "ST", "LA", "PS", "SA", "BIN"))):
            pl = pool(fams)
            nd = need_of(fams)
            for iso in (False, True):
                p, s, sels = run_model(lambda pl=pl, iso=iso: SingleBest(pl, iso=iso), hard, XH, scheme, need=nd)
                record(scheme, f"single[{fam_name}]{'_iso' if iso else ''}", p, s, sels, need=nd)
        # greedy per tier (loso for every tier; the shape splits for the main tiers)
        tiers = TIERS if scheme == "loso" else {k: TIERS[k] for k in ("noprobe(ST+LA)", "free(ST+LA+PS)", "free+SA", "free+BIN", "all")}
        for tname, fams in tiers.items():
            pl = pool(fams)
            nd = need_of(fams)
            p, s, sels = run_model(lambda pl=pl: Greedy(pl), hard, XH, scheme, need=nd)
            record(scheme, f"greedy[{tname}]", p, s, sels, need=nd)
        for tname in ("free(ST+LA+PS)", "free+SA", "free+BIN", "all"):
            fams = TIERS[tname]
            pl = pool(fams)
            nd = need_of(fams)
            for kind in ("gbm", "rf"):
                p, s, _ = run_model(lambda pl=pl, kind=kind: Tree(pl, kind), hard, XH, scheme, need=nd)
                record(scheme, f"{kind}[{tname}]", p, s, need=nd)

    # hybrids: direct continued solve to C, open cases ranked by a model (loso)
    for C in (50_000, 100_000):
        for base in ("greedy[free(ST+LA+PS)]", "A4_free20k"):
            p0, s0 = preds[("loso", base)]
            dd = np.array([float(r["d"]) if r["exact"] else OPEN_D + 1 for r in hard])
            p = np.where(dd <= C, np.log(dd), np.maximum(p0, math.log(C) + 1e-6) + 1.0)
            # +1.0 keeps every open case above every case resolved at C (order-preserving)
            record("loso", f"direct{C // 1000}k+{base}", p, s0)

    # ---------------- gain simulation ----------------
    gpath = os.path.join(HERE, "features_gaintables.jsonl")
    gall = load_jsonl(gpath) if os.path.exists(gpath) else []
    gsets = {u: [T for T in gall if T.get("universe_kind", "survivors") == u] for u in ("survivors", "all")}

    def gain_eval_both(est):
        return {u: gain_eval(gsets[u], est) for u in gsets}
    key_i = {r["key"]: i for i, r in enumerate(hard)}
    oracle = {r["key"]: (float(r["d"]) if r["exact"] else OPEN_D) for r in hard}
    results["gain"]["oracle"] = gain_eval_both(oracle)
    for (scheme, name), (p, s) in preds.items():
        if scheme != "loso":
            continue
        for use_smear in (False, True):
            if name in ("current_label", "fhat_raw", "c20000_probe") and use_smear:
                continue
            est = {k: float(math.exp(p[i]) * (s[i] if use_smear else 1.0)) for k, i in key_i.items() if np.isfinite(p[i])}
            tag = name + ("+smear" if use_smear else "")
            results["gain"][tag] = gain_eval_both(est)
    # direct-solve references for the gain (exact to C, then current label / model)
    for C in (50_000, 100_000, 200_000):
        for base in ("current_label", "greedy[free(ST+LA+PS)]"):
            p, s = preds[("loso", base)]
            est = {}
            for k, i in key_i.items():
                r = hard[i]
                if r["exact"] and float(r["d"]) <= C:
                    est[k] = float(r["d"])
                elif np.isfinite(p[i]):
                    est[k] = max(float(math.exp(p[i]) * (1 if base == "current_label" else s[i])), C)
            results["gain"][f"direct{C // 1000}k+{base}"] = gain_eval_both(est)

    # ---------------- MID regime (2k tier features only; no feature that reveals d) ----------------
    mid_names = [n for n in names if not (n.startswith("pr:ps20k") or n.startswith("pr:bin20k") or n.startswith("pr:prog"))]
    midx = {n: j for j, n in enumerate(names)}
    XM = impute(matrix(mid, names, tr))
    old_hard = hard
    for tname, fams in (("fhat_refit", None), ("LA", ("BASE", "LA")), ("SA", ("BASE", "SA")),
                        ("ST", ("BASE", "ST")), ("2k(PS+BIN 2k)", ("BASE", "PS", "BIN")),
                        ("noprobe(ST+LA)", ("BASE", "ST", "LA")), ("2k-all(no SA)", ("BASE", "ST", "LA", "PS", "BIN")),
                        ("2k-all+SA", ("BASE", "ST", "LA", "PS", "BIN", "SA"))):
        if fams is None:
            mk = lambda: OLS([midx["base:log_c2000"], midx["base:log2_volume"]])  # noqa: E731
            nd = ()
        else:
            pl = [midx[n] for n in mid_names if family(n) in fams]
            nd = need_of(fams)
            mk = lambda pl=pl: Greedy(pl)  # noqa: E731
        p, s, sels = run_model(mk, mid, XM, "loso", need=nd)
        m = metrics(mid, p)
        m = {k: v for k, v in m.items() if k not in ("per_cell_C",)}
        if sels:
            cnt = defaultdict(int)
            for sl in sels:
                for j in sl:
                    cnt[names[j]] += 1
            m["selected"] = dict(sorted(cnt.items(), key=lambda kv: -kv[1])[:8])
        results["mid"][tname] = m
        print(f"MID {tname:22s} within {fmt(m['within_rho'])} pooled {fmt(m['pooled_rho'])} rmse {fmt(m['log_rmse'])}", flush=True)
    cm = np.log([min(max(2000, r["fhat"]), 20 * 2000) for r in mid])
    m = metrics(mid, cm)
    results["mid"]["current_label(clip 2k..40k)"] = {k: v for k, v in m.items() if k != "per_cell_C"}
    hard = old_hard

    # ---------------- cost ----------------
    for e in costs:
        for reg, rs in (("hard", hard), ("mid", mid)):
            c = [costs[e][r["key"]] for r in rs if r["key"] in costs[e]]
            if not c:
                continue
            a = np.array(c, float)
            results["costs"].setdefault(e, {})[reg] = {
                "n": len(c), "conflicts_mean": float(a[:, 0].mean()), "conflicts_max": float(a[:, 0].max()),
                "props_mean": float(a[:, 1].mean()), "seconds_mean": float(a[:, 2].mean()),
                "seconds_p90": float(np.percentile(a[:, 2], 90)), "seconds_max": float(a[:, 2].max())}
        per = defaultdict(list)
        for r in hard:
            if r["key"] in costs[e]:
                per[r["cell"]].append(costs[e][r["key"]])
        results["costs"].setdefault(e, {})["per_cell_hard"] = {
            c: {"n": len(v), "conflicts": float(np.mean([x[0] for x in v])), "props": float(np.mean([x[1] for x in v])),
                "seconds": float(np.mean([x[2] for x in v]))} for c, v in sorted(per.items())}
    # progress split: the pysat part (free20k) vs the binary part, from meta is not kept per run
    # here; the binary adds ~22k conflicts (2k + 20k caps), pysat 22k.

    # ---------------- projected relabel wall time (target tables' censored cases, 12 workers) ----------------
    import glob
    tgt = []
    for fpath in sorted(glob.glob(os.path.join(UB, "cache", "case_table_*.json"))):
        dd = json.load(open(fpath))
        if dd.get("kind") != "target":
            continue
        recs = dd.get("records") or []
        mk = dd.get("baseline_lean_mask") or [False] * len(recs)
        cen = [i for i, rr in enumerate(recs) if (rr.get("probe") or {}).get("status") == "unknown"]
        if not cen:
            continue
        ii = dd["inst"]
        tgt.append({"table": os.path.basename(fpath), "m": ii["m"], "n": ii["n"], "censored": len(cen),
                    "censored_survivors": sum(not mk[i] for i in cen)})
    parts = {"free20k_reprobe(pysat 2k+20k)": ["progress_pysat"], "lookahead(full)": ["lookahead"],
             "sampling(default)": ["sampling"], "binary(2k+20k)": ["progress_binary"]}
    percell = {}
    for part, es in parts.items():
        pts = defaultdict(list)
        for r in hard:
            if all(r["key"] in costs.get(e, {}) for e in es):
                pts[(r["m"], r["n"])].append([sum(costs[e][r["key"]][q] for e in es) for q in range(3)])
        cm = {k: np.mean(np.array(v), axis=0) for k, v in pts.items() if len(v) >= 3}
        xs = np.array([math.log(k[0] * k[1]) for k in cm])
        ys = np.array([math.log(max(v[2], 1e-3)) for v in cm.values()])
        yc = np.array([math.log(max(v[0], 1.0)) for v in cm.values()])
        A = np.vstack([np.ones_like(xs), xs]).T
        bs = np.linalg.lstsq(A, ys, rcond=None)[0]
        bc = np.linalg.lstsq(A, yc, rcond=None)[0]
        percell[part] = (cm, bs, bc)
    rel = {"tables": tgt, "workers": 12, "projection": {}}
    for part, (cm, bs, bc) in percell.items():
        tot_s = tot_s_surv = tot_c = tot_c_surv = 0.0
        for T in tgt:
            k = (T["m"], T["n"])
            if k in cm:
                sec, conf = cm[k][2], cm[k][0]
            else:
                x = math.log(T["m"] * T["n"])
                sec, conf = math.exp(bs[0] + bs[1] * x), math.exp(bc[0] + bc[1] * x)
            tot_s += T["censored"] * sec
            tot_s_surv += T["censored_survivors"] * sec
            tot_c += T["censored"] * conf
            tot_c_surv += T["censored_survivors"] * conf
        rel["projection"][part] = {"cpu_seconds": tot_s, "wall_hours_12w": tot_s / 12 / 3600,
                                   "wall_hours_12w_survivors_only": tot_s_surv / 12 / 3600,
                                   "conflicts_total": tot_c, "conflicts_survivors_only": tot_c_surv}
    results["relabel"] = rel

    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    with open(OUT_PRED, "w") as fh:
        for i, r in enumerate(hard):
            o = {"key": r["key"], "cell": r["cell"], "d": r["d"], "exact": r["exact"]}
            for (scheme, name), (p, s) in preds.items():
                if scheme == "loso" and np.isfinite(p[i]):
                    o[name] = round(float(p[i]), 4)
            fh.write(json.dumps(o) + "\n")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

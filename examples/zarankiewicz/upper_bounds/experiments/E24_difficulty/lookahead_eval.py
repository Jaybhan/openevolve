"""E24 / A2-lookahead: evaluate the lookahead features on the DEV set.

Input: lookahead_features_dev.jsonl (lookahead_features.py).  Regimes (true d from the ground truth):
  HARD = subset H: exact d > 20000;   MID = subset M: exact 2000 < d <= 20000;
  CENS = subset C: unknown at cap 20000 (d >= 20000, a lower bound) -- used only for the
         hard-vs-mid AUC on cells that are censored today (target cells).

Outputs (lookahead_eval.json + a printed report):
  1. per-feature Spearman with d: pooled over the regime, within each cell, the mean within-cell
     rho, and LOCO sign-selected rho (sign chosen on the other cells, applied to the held-out cell);
  2. per-feature AUC(hard vs mid) inside each cell that has both (C u H vs M), LOCO-signed;
  3. log-linear models  log d = b0 + sum_k b_k f_k  fitted on d > 2000 cases (H u M) of the other
     cells (leave-one-cell-out; also leave-one-(m,n)-out), evaluated on the held-out cell's HARD
     and MID cases: within-cell Spearman, pooled Spearman of held-out predictions, log-RMSE (raw
     and clipped to the censored-label range [20000, 400000] for HARD), top-decile recall; and
     the same for the fhat baseline (a) refitted on the same folds, (b) with the fixed
     coefficients in experiments/E11_calibration/calibrate_33.json (in-sample for the 7 E10
     TRAIN cells).  Models: pre-specified feature sets and a NESTED greedy forward selection
     (selection done inside each fold by inner LOCO log-RMSE; never sees the held-out cell).
  4. cost per case (seconds, propagate calls, trail literals, UP conflicts), by cell size.

usage: python lookahead_eval.py [--in lookahead_features_dev.jsonl] [--out lookahead_eval.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UB)
from zar_ub.difficulty import spearman  # noqa: E402

CALIB = json.load(open(os.path.join(UB, "experiments", "E11_calibration", "calibrate_33.json")))[
    "fits"
]["3,3"]
FIXED = (CALIB["a"], CALIB["b"], CALIB["g"])
E10_CELLS = {
    "m9_n9_s3_t3_w50_pure",
    "m9_n10_s3_t3_w55_pure",
    "m10_n10_s3_t3_w61_pure",
    "m10_n11_s3_t3_w65_pure",
    "m11_n11_s3_t3_w70_pure",
    "m11_n12_s3_t3_w75_pure",
    "m12_n12_s3_t3_w81_pure",
}
LO, HI = 20000.0, 400000.0  # censored-label clip range at cap 20000

POOL = [
    "log2_volume",
    "fl_free_cells",
    "fl_fixed_frac",
    "la1_failed_frac",
    "la1_failed",
    "ams_score_mean",
    "march_prod_mean",
    "kn_rand_mean_log_conf",
    "knfl_rand_mean_log_conf",
    "knfl_rand_log_mean_conf",
    "kn_row_mean_log_conf",
    "la2_failed_frac",
    "la1_zero_frac",
    "la1_newbin_mean",
    "la1_rate_vars",
    "fl_rounds",
    "kn_rand_mean_depth",
    "knfl_rand_mean_depth",
    "log_ncell",
    "fl_log2vol_rows",
    "fl_log2vol_cols",
]

PRESPEC = {
    "vol+fl_free": ["log2_volume", "fl_free_cells"],
    "FL-only: vol+fl_free+fl_vol_rows": ["log2_volume", "fl_free_cells", "fl_log2vol_rows"],
    "vol+fl_free+knfl": ["log2_volume", "fl_free_cells", "knfl_rand_mean_log_conf"],
    "vol+la1_failed_frac+ams_mean": ["log2_volume", "la1_failed_frac", "ams_score_mean"],
    "vol+fl_free+knfl+log_ncell": [
        "log2_volume",
        "fl_free_cells",
        "knfl_rand_mean_log_conf",
        "log_ncell",
    ],
    "vol+log_c2000 (fhat form, refit)": ["log2_volume", "log_c2000"],
}


def static_feats(r):
    gr, gc = collections.Counter(r["rows"]), collections.Counter(r["cols"])
    return {
        "log2_aut": (
            sum(math.lgamma(k + 1) for k in gr.values())
            + sum(math.lgamma(k + 1) for k in gc.values())
        )
        / math.log(2),
        "n_groups": float(len(gr) + len(gc)),
        "max_col": float(max(r["cols"])),
        "log_ncell": math.log(r["m"] * r["n"]),
    }


def load(path):
    sup = {}
    sp = path.replace(".jsonl", "_fl.jsonl")
    if os.path.exists(sp):
        for l in open(sp):
            o = json.loads(l)
            sup[(o["cell"], tuple(o["rows"]), tuple(o["cols"]))] = o
    rs = []
    for l in open(path):
        r = json.loads(l)
        f = dict(r["features"])
        o = sup.get((r["cell"], tuple(r["rows"]), tuple(r["cols"])))
        if o is not None:
            for k in ("fl_log2vol_rows", "fl_log2vol_cols"):
                f[k] = o["features_fl"].get(k, 0.0)
            for k in (
                "cost_fl_seconds",
                "cost_fl_calls",
                "cost_fl_propagations",
                "cost_fl_conflicts",
            ):
                r[k] = o[k]
        f["log2_volume"] = float(r["log2_volume"])
        f["log_c2000"] = math.log(max(min(int(r["c2000"]), 2000), 1))
        f["log_d_hat"] = math.log(r["d_hat"]) if r.get("d_hat") else 0.0
        f.update(static_feats(r))
        r["F"] = f
        r["logd"] = math.log(max(float(r["d"]), 1.0))
        r["mn"] = f"m{r['m']}_n{r['n']}"
        rs.append(r)
    return rs


def by_cell(rs):
    g = collections.defaultdict(list)
    for r in rs:
        g[r["cell"]].append(r)
    return g


def within(rs, f, min_n=8):
    out = {}
    for c, g in by_cell(rs).items():
        if len(g) < min_n:
            continue
        x = [r["F"].get(f, 0.0) for r in g]
        if len(set(x)) < 2:
            out[c] = float("nan")
        else:
            out[c] = spearman(x, [r["d"] for r in g])
    return out


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def auc(pos, neg):
    """P(score(pos) > score(neg)) with ties 1/2."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # rank-sum with average ranks
    ranks, i = {}, 0
    vals = [v for v, _ in allv]
    r = [0.0] * len(vals)
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        for k in range(i, j + 1):
            r[k] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(r[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    npos, nneg = len(pos), len(neg)
    return (rp - npos * (npos + 1) / 2) / (npos * nneg)


# ---------------------------------------------------------------------------
# regression
# ---------------------------------------------------------------------------
def fit(rows, feats, ridge=1e-6):
    X = np.array([[r["F"].get(f, 0.0) for f in feats] for r in rows], dtype=float)
    y = np.array([r["logd"] for r in rows])
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(rows), 1)), (X - mu) / sd])
    A = Z.T @ Z + ridge * np.diag([0] + [1] * len(feats)) * len(rows)
    beta = np.linalg.solve(A, Z.T @ y)
    return (feats, mu, sd, beta)


def predict(model, rows):
    feats, mu, sd, beta = model
    X = np.array([[r["F"].get(f, 0.0) for f in feats] for r in rows], dtype=float)
    Z = np.hstack([np.ones((len(rows), 1)), (X - mu) / sd])
    return Z @ beta


def lrmse(pred_log, rows):
    return (
        float(np.sqrt(np.mean([(p - r["logd"]) ** 2 for p, r in zip(pred_log, rows)])))
        if rows
        else float("nan")
    )


def fhat_fixed_log(rows):
    a, b, g = FIXED
    return np.array([a + b * r["F"]["log_c2000"] + g * r["F"]["log2_volume"] for r in rows])


def top_recall(pred, rows, q=0.1):
    n = len(rows)
    k = max(1, int(round(n * q)))
    true_top = set(sorted(range(n), key=lambda i: -rows[i]["d"])[:k])
    pred_top = set(sorted(range(n), key=lambda i: -pred[i])[:k])
    return len(true_top & pred_top) / k


def inner_cv_rmse(train, feats, group_key):
    groups = sorted(set(r[group_key] for r in train))
    errs, n = 0.0, 0
    for g in groups:
        tr = [r for r in train if r[group_key] != g]
        te = [r for r in train if r[group_key] == g]
        if not te or len(tr) < len(feats) + 3:
            continue
        p = predict(fit(tr, feats), te)
        errs += sum((a - r["logd"]) ** 2 for a, r in zip(p, te))
        n += len(te)
    return math.sqrt(errs / n) if n else float("inf")


def greedy(train, group_key, kmax=4, pool=POOL):
    chosen, best = [], float("inf")
    while len(chosen) < kmax:
        cand = [(inner_cv_rmse(train, chosen + [f], group_key), f) for f in pool if f not in chosen]
        if not cand:
            break
        e, f = min(cand)
        if e >= best - 1e-4:
            break
        chosen.append(f)
        best = e
    return chosen


def loco_models(rs, group_key="cell", do_greedy=True):
    """Leave-one-group-out over every group holding HARD or MID test cases."""
    fitset = [r for r in rs if r["subset"] in ("H", "M")]
    groups = sorted(set(r[group_key] for r in fitset))
    names = list(PRESPEC) + (["greedy(nested)"] if do_greedy else []) + ["fhat fixed (E11)"]
    preds = {nm: {} for nm in names}  # name -> id(row) -> pred log d
    chosen_log = {}
    for g in groups:
        tr = [r for r in fitset if r[group_key] != g]
        te = [r for r in fitset if r[group_key] == g]
        for nm, feats in PRESPEC.items():
            p = predict(fit(tr, feats), te)
            preds[nm].update({id(r): v for r, v in zip(te, p)})
        if do_greedy:
            feats = greedy(tr, group_key)
            chosen_log[g] = feats
            p = (
                predict(fit(tr, feats), te)
                if feats
                else np.full(len(te), np.mean([r["logd"] for r in tr]))
            )
            preds["greedy(nested)"].update({id(r): v for r, v in zip(te, p)})
        preds["fhat fixed (E11)"].update({id(r): v for r, v in zip(te, fhat_fixed_log(te))})
    return preds, chosen_log


def loco_hard_models(rs):
    """HARD regime, conditioned on d > 20000 (what is known for a case censored at 20k).
    Leave-one-hard-cell-out.  Three fits per feature set:
      'Honly'   : fitted on the other cells' HARD cases only;
      'HM+shift': slopes fitted on the other cells' d > 2000 cases (H u M), intercept shifted by
                  the mean residual of the other cells' HARD cases (ranking from many cases,
                  level from the hard ones);
    plus the fixed E11 fhat and the fhat form refitted on H only."""
    H = [r for r in rs if r["subset"] == "H"]
    HM = [r for r in rs if r["subset"] in ("H", "M")]
    cells = sorted(set(r["cell"] for r in H))
    sets = dict(PRESPEC)
    preds = collections.defaultdict(dict)
    for c in cells:
        trH = [r for r in H if r["cell"] != c]
        trHM = [r for r in HM if r["cell"] != c]
        te = [r for r in H if r["cell"] == c]
        for nm, fs in sets.items():
            p = predict(fit(trH, fs), te)
            preds[nm + " [Honly]"].update({id(r): v for r, v in zip(te, p)})
            m = fit(trHM, fs)
            shift = float(np.mean([r["logd"] - v for r, v in zip(trH, predict(m, trH))]))
            p = predict(m, te) + shift
            preds[nm + " [HM+shift]"].update({id(r): v for r, v in zip(te, p)})
        const = float(np.mean([r["logd"] for r in trH]))
        preds["constant (mean log d of other cells' HARD)"].update({id(r): const for r in te})
        preds["fhat fixed (E11)"].update({id(r): v for r, v in zip(te, fhat_fixed_log(te))})
    return preds


def e11_comparable(rs, preds_cell):
    """The E11 acceptance metric: Spearman / log-RMSE on the held-out (12,13,87) cell over ALL its
    exactly-labelled d > 2000 cases (E11: n = 276, rho = 0.395); here the H u M sample of that cell.
    """
    c = "m12_n13_s3_t3_w87_pure"
    te = [r for r in rs if r["cell"] == c and r["subset"] in ("H", "M")]
    out = {}
    for nm, P in preds_cell.items():
        p = [P[id(r)] for r in te if id(r) in P]
        if len(p) != len(te):
            continue
        out[nm] = {
            "n": len(te),
            "spearman": spearman(p, [r["d"] for r in te]),
            "log_rmse": lrmse(p, te),
        }
    return out


def score_preds(rs, preds, subset, clip=False):
    test = [r for r in rs if r["subset"] == subset]
    out = {}
    for nm, P in preds.items():
        p = [P[id(r)] for r in test]
        if clip:
            p = [min(max(v, math.log(LO)), math.log(HI)) for v in p]
        cells = by_cell(test)
        wc = {}
        for c, g in cells.items():
            if len(g) >= 8:
                pc = [P[id(r)] for r in g]
                wc[c] = spearman(pc, [r["d"] for r in g])
        top = {}
        for c, g in cells.items():
            if len(g) >= 20:
                top[c] = top_recall([P[id(r)] for r in g], g)
        out[nm] = {
            "n": len(test),
            "pooled_spearman": spearman(p, [r["d"] for r in test]),
            "mean_within_cell_spearman": nanmean(list(wc.values())),
            "within_cell_spearman": wc,
            "log_rmse": lrmse(p, test),
            "top10_recall_mean": nanmean(list(top.values())),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(HERE, "lookahead_features_dev.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "lookahead_eval.json"))
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument(
        "--write-model",
        default=None,
        help="comma-separated feature list: fit on ALL H u M cases, write lookahead_model.json",
    )
    a = ap.parse_args()
    rs = load(a.inp)
    if a.write_model:
        feats = a.write_model.split(",")
        fs = [r for r in rs if r["subset"] in ("H", "M")]
        fe, mu, sd, beta = fit(fs, feats)
        mp_ = os.path.join(HERE, "lookahead_model.json")
        json.dump(
            {
                "features": fe,
                "mu": list(map(float, mu)),
                "sd": list(map(float, sd)),
                "beta": list(map(float, beta)),
                "n_fit": len(fs),
                "fitted_on": sorted(set(r["cell"] for r in fs)),
                "target": "log d (CaDiCaL conflicts), exact labels with d > 2000",
                "source": os.path.basename(a.inp),
            },
            open(mp_, "w"),
            indent=1,
        )
        print("wrote", mp_, dict(zip(["b0"] + fe, map(float, beta))))
        return
    res = {"input": os.path.basename(a.inp), "counts": {}}
    for s in "HMC":
        res["counts"][s] = dict(collections.Counter(r["cell"] for r in rs if r["subset"] == s))
    H = [r for r in rs if r["subset"] == "H"]
    M = [r for r in rs if r["subset"] == "M"]
    feats = sorted(set(k for r in rs for k in r["F"]))

    # ---- 1. per-feature Spearman ----
    tab = {}
    for f in feats:
        row = {}
        for nm, R in (("HARD", H), ("MID", M)):
            x = [r["F"].get(f, 0.0) for r in R]
            if len(set(x)) < 2:
                row[nm] = None
                continue
            wc = within(R, f)
            # LOCO sign selection: sign of the mean within-cell rho of the OTHER cells
            signed = []
            for c, v in wc.items():
                others = [u for k, u in wc.items() if k != c and u == u]
                sgn = 1.0 if (sum(others) if others else 1.0) >= 0 else -1.0
                if v == v:
                    signed.append(sgn * v)
            row[nm] = {
                "pooled": spearman(x, [r["d"] for r in R]),
                "within": wc,
                "mean_within": nanmean(list(wc.values())),
                "loco_signed_mean": nanmean(signed),
                "n_cells": len(wc),
            }
        tab[f] = row
    res["feature_spearman"] = tab

    # ---- 2. AUC hard-vs-mid inside cells that have both ----
    aucs = {}
    cells = by_cell(rs)
    for f in feats:
        per = {}
        for c, g in cells.items():
            pos = [r["F"].get(f, 0.0) for r in g if r["subset"] in ("H", "C")]
            neg = [r["F"].get(f, 0.0) for r in g if r["subset"] == "M"]
            if len(pos) >= 10 and len(neg) >= 10:
                per[c] = auc(pos, neg)
        signed = []
        for c, v in per.items():
            others = [u - 0.5 for k, u in per.items() if k != c and u == u]
            sgn = 1.0 if (sum(others) if others else 1.0) >= 0 else -1.0
            signed.append(0.5 + sgn * (v - 0.5))
        aucs[f] = {
            "per_cell": per,
            "loco_signed_mean": nanmean(signed),
            "loco_signed_mean_target": nanmean(
                [
                    0.5
                    + (1 if nanmean([u - 0.5 for k, u in per.items() if k != c]) >= 0 else -1)
                    * (v - 0.5)
                    for c, v in per.items()
                    if not c.endswith("_pure")
                ]
            ),
        }
    res["feature_auc_hard_vs_mid"] = aucs

    # ---- 3. models ----
    res["models"] = {}
    for gk in ("cell", "mn"):
        preds, chosen = loco_models(rs, gk, do_greedy=not a.no_greedy)
        res["models"][f"leave_one_{gk}_out"] = {
            "HARD": score_preds(rs, preds, "H"),
            "HARD_clipped": score_preds(rs, preds, "H", clip=True),
            "MID": score_preds(rs, preds, "M"),
            "greedy_chosen": chosen,
        }
        if gk == "cell":
            res["e11_comparable_w87"] = e11_comparable(rs, preds)
    hp = loco_hard_models(rs)
    res["hard_trained"] = {
        "HARD": score_preds(rs, hp, "H"),
        "HARD_clipped": score_preds(rs, hp, "H", clip=True),
    }
    # AUC of models on (C u H) vs M, leave-one-cell-out
    fitset = [r for r in rs if r["subset"] in ("H", "M")]
    mauc = collections.defaultdict(dict)
    for c, g in cells.items():
        pos = [r for r in g if r["subset"] in ("H", "C")]
        neg = [r for r in g if r["subset"] == "M"]
        if len(pos) < 10 or len(neg) < 10:
            continue
        tr = [r for r in fitset if r["cell"] != c]
        for nm, fs in list(PRESPEC.items()):
            m = fit(tr, fs)
            mauc[nm][c] = auc(list(predict(m, pos)), list(predict(m, neg)))
        mauc["fhat fixed (E11)"][c] = auc(list(fhat_fixed_log(pos)), list(fhat_fixed_log(neg)))
    res["model_auc_hard_vs_mid"] = {
        nm: {
            "per_cell": v,
            "mean": nanmean(list(v.values())),
            "mean_target": nanmean([u for k, u in v.items() if not k.endswith("_pure")]),
        }
        for nm, v in mauc.items()
    }

    # ---- 4. cost ----
    cost = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rs:
        k = r["mn"]
        for q in ("cost_seconds", "cost_calls", "cost_propagations", "cost_conflicts"):
            cost[k][q].append(r[q])
    res["cost"] = {
        k: {q: float(np.mean(v)) for q, v in d.items()} | {"n": len(d["cost_seconds"])}
        for k, d in sorted(cost.items())
    }
    res["cost_overall"] = {
        q: float(np.mean([r[q] for r in rs if q in r]))
        for q in (
            "cost_seconds",
            "cost_calls",
            "cost_propagations",
            "cost_conflicts",
            "cost_fl_seconds",
            "cost_fl_calls",
            "cost_fl_propagations",
            "cost_fl_conflicts",
        )
        if any(q in r for r in rs)
    }
    for k in res["cost"]:
        sub = [r for r in rs if r["mn"] == k and "cost_fl_seconds" in r]
        if sub:
            for q in ("cost_fl_seconds", "cost_fl_calls", "cost_fl_propagations"):
                res["cost"][k][q] = float(np.mean([r[q] for r in sub]))
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    report(res)


def report(res):
    print("counts:", {k: sum(v.values()) for k, v in res["counts"].items()})
    print("\n== per-feature Spearman (HARD / MID): pooled | mean within-cell | LOCO-signed ==")
    rows = []
    for f, row in res["feature_spearman"].items():
        h, m = row.get("HARD"), row.get("MID")
        if not h:
            continue
        rows.append(
            (
                -(h["loco_signed_mean"] if h["loco_signed_mean"] == h["loco_signed_mean"] else -9),
                f,
                h,
                m,
            )
        )
    for _, f, h, m in sorted(rows)[:40]:
        au = res["feature_auc_hard_vs_mid"].get(f, {})
        print(
            "%-28s H %+.2f | %+.2f | %+.2f   M %s   AUC(loco) %.2f tgt %.2f"
            % (
                f,
                h["pooled"],
                h["mean_within"],
                h["loco_signed_mean"],
                (
                    (
                        "%+.2f | %+.2f | %+.2f"
                        % (m["pooled"], m["mean_within"], m["loco_signed_mean"])
                    )
                    if m
                    else "   n/a"
                ),
                au.get("loco_signed_mean", float("nan")),
                au.get("loco_signed_mean_target", float("nan")),
            )
        )
    for gk, M in res["models"].items():
        print(f"\n== models, {gk} ==")
        for reg in ("HARD", "HARD_clipped", "MID"):
            print(f"  [{reg}]")
            for nm, s in M[reg].items():
                print(
                    "   %-34s n=%4d pooled %+.3f within %+.3f  lRMSE %.3f  top10 %.2f  cells %s"
                    % (
                        nm,
                        s["n"],
                        s["pooled_spearman"],
                        s["mean_within_cell_spearman"],
                        s["log_rmse"],
                        s["top10_recall_mean"],
                        (
                            {
                                k.split("_s3")[0]: round(v, 2)
                                for k, v in s["within_cell_spearman"].items()
                            }
                            if reg != "MID"
                            else ""
                        ),
                    )
                )
        if M.get("greedy_chosen"):
            print(
                "  greedy chosen per fold:",
                collections.Counter(tuple(v) for v in M["greedy_chosen"].values()).most_common(5),
            )
    print("\n== HARD regime, models trained conditional on d > 20000 (leave-one-hard-cell-out) ==")
    for reg in ("HARD", "HARD_clipped"):
        print(f"  [{reg}]")
        for nm, s in res["hard_trained"][reg].items():
            print(
                "   %-52s pooled %+.3f within %+.3f  lRMSE %.3f  top10 %.2f  %s"
                % (
                    nm,
                    s["pooled_spearman"],
                    s["mean_within_cell_spearman"],
                    s["log_rmse"],
                    s["top10_recall_mean"],
                    {k.split("_s3")[0]: round(v, 2) for k, v in s["within_cell_spearman"].items()},
                )
            )
    print(
        "\n== E11-comparable: held-out (12,13,87), all sampled d > 2000 cases (E11 fhat: rho 0.395, lRMSE 1.282, n=276) =="
    )
    for nm, v in res.get("e11_comparable_w87", {}).items():
        print("   %-34s n=%d rho %+.3f lRMSE %.3f" % (nm, v["n"], v["spearman"], v["log_rmse"]))
    print("\n== model AUC hard-vs-mid (C u H vs M), leave-one-cell-out ==")
    for nm, v in res["model_auc_hard_vs_mid"].items():
        print("  %-34s mean %.3f  target-cells %.3f" % (nm, v["mean"], v["mean_target"]))
    print("\n== cost per case ==")
    for k, v in res["cost"].items():
        print(
            "  %-8s n=%4d  full: %.2fs calls %6.0f props %9.0f UPconf %5.0f | FL-only: %.3fs calls %5.0f props %8.0f"
            % (
                k,
                v["n"],
                v["cost_seconds"],
                v["cost_calls"],
                v["cost_propagations"],
                v["cost_conflicts"],
                v.get("cost_fl_seconds", float("nan")),
                v.get("cost_fl_calls", float("nan")),
                v.get("cost_fl_propagations", float("nan")),
            )
        )
    print("  overall", {k: round(v, 3) for k, v in res["cost_overall"].items()})


if __name__ == "__main__":
    main()

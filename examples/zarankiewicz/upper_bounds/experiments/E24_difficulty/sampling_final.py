"""E24 / A3: headline evaluation of the Chivilikhin-style sampling estimator (no solver runs).

Protocol (selection and fitting on DEV only):
  DEV    = ground_truth_initial.jsonl exact HARD cases (d > 20k): 262 cases, 4 square cells
           (sampling_data/dev.jsonl, A4's progress_features_dev.jsonl for the free20k inputs).
  WIDE   = 509 exact hard cases in 5 wide cells never used for fitting (sampling_data/wide_test.jsonl,
           A4's progress_features_test{,2,3}.jsonl).
  TARGET = the deepened random sample of the three open target tables (12,18,109), (13,19,123),
           (16,17,134): 283 exact hard + 167 still open at 2M conflicts (right-censored)
           (sampling_data/target.jsonl, sampling_data/target_a4.jsonl).
Every model is  log d = a + sum_k b_k z_k  fitted on all DEV hard cases and frozen.
Metrics on the exact hard cases: mean within-cell Spearman (cells with >= 8 cases), pooled Spearman,
log-RMSE (natural log); on TARGET additionally Harrell's C with the 2M-censored cases (within-cell
mean and pooled), which is the only metric that sees the hardest half of (16,17,134).

Operating points:
  fixed (N, b)                  first N cubes at per-cube budget b (derived offline from N=100,
                                b=10000 runs: prefix of an i.i.d. sample; re-censoring of a
                                deterministic fresh CaDiCaL run)
  budgeted (T, b)               cubes in order until the cumulative conflicts reach T (a hard
                                per-case cap: cost <= T + b), N <= 100
Baselines on the same cases: current label (clip(fhat, 20k, 400k), calibrate_33.json), fhat form
refit on DEV, A4's free20k (refit on DEV, same 4 features as PROGRESS.md), a direct fresh CaDiCaL
run at cap C (outcome min(d, C): deterministic, so read off the exact label), and the hybrid
"direct run at C, then sampling for the cases still open".

usage: python experiments/E24_difficulty/sampling_final.py [--out sampling_data/final_eval.json]
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
sys.path.insert(0, UB)
sys.path.insert(0, HERE)

from sampling_eval import derive  # noqa: E402
from zar_ub import hardness_sampling as hs  # noqa: E402
from zar_ub.difficulty import fhat, load_calibration  # noqa: E402

D = os.path.join(HERE, "sampling_data")
A4_FEATS = [("ps20k_decs_per_conf", "id"), ("ps20k_restarts_per_kconf", "log1p"), ("st_distinct_rows", "id"),
            ("st_col_lp_slackT", "id")]
CALIB = load_calibration(3, 3)


def key_of(r):
    return f'{r["cell"]}|{",".join(map(str, r["rows"]))}|{",".join(map(str, r["cols"]))}'


def jl(paths):
    out = []
    for p in paths:
        if os.path.exists(p):
            for line in open(p):
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    return out


def derive_budgeted(rec, T, b):
    """Cubes in stored order until the cumulative (re-censored) conflicts reach T.  For the
    stratified sampler (round-robin over S strata) the estimate uses only complete rounds (a
    multiple of S, at least one round, so every stratum is represented); the cost counts every
    cube that was run."""
    acc, n = 0, 0
    for x in rec["recs"]:
        acc += min(x[0], b)
        n += 1
        if acc >= T:
            break
    S = int(rec.get("n_strata") or 0)
    if S and rec["recs"] and len(rec["recs"][0]) > 6:
        n_use = min(len(rec["recs"]), max(S, (n // S) * S))
        dv = derive(rec, n_use, b)
        dv["cost_conflicts"] = int(sum(min(x[0], b) for x in rec["recs"][:max(n, n_use)]))
        return dv, n_use
    return derive(rec, n, b), n


def op_rows(runs, feats, cfg, op, gt_censor=None):
    """Case rows for one operating point.  op = ("N", N, b) or ("T", T, b)."""
    rows = []
    runs = [r for r in runs if r.get("config") == cfg and "error" not in r]
    if runs and op[2] > min(r.get("bmax", 10_000) for r in runs):
        return []  # some records were stored at a smaller per-cube cap: b cannot be derived
    for rec in runs:
        f = feats.get(rec["key"])
        if f is None:
            continue
        exact = f.get("status") in ("unsat", "sat")
        if exact and f["d"] <= 20000:
            continue
        if not exact and f["d"] < 1_000_000:
            continue
        if op[0] == "N":
            dv, n = derive(rec, op[1], op[2]), op[1]
        else:
            dv, n = derive_budgeted(rec, op[1], op[2])
        ff = f.get("features", {})
        a4 = None
        if all(ff.get(k) is not None for k, _ in A4_FEATS):
            a4 = [math.log1p(ff[k]) if how == "log1p" else float(ff[k]) for k, how in A4_FEATS]
        c2000 = f.get("c2000")
        rows.append(dict(key=rec["key"], cell=rec["cell"], d=float(f["d"]), exact=exact, dh=max(dv["d_hat"], 1.0),
                         feat=dv["f"], cost=dv["cost_conflicts"], props=dv["cost_props"], secs=dv["cost_seconds_est"],
                         n_cubes=n, c2000=int(c2000) if c2000 is not None else 2000, vol=float(f["log2_volume"]),
                         a4=a4))
    return rows


class _Ratio:
    """log d = c + log mu~ (unit slope, c = median on the fitting cases): the calibration frozen in
    zar_ub.hardness_sampling.CALIBRATION."""
    ratio = True

    def __call__(self, r):
        return [math.log(r["dh"])]


MODELS = {
    "fhat-form(refit)": lambda r: [math.log(max(min(r["c2000"], 2000), 1)), r["vol"]],
    "samp": lambda r: [math.log(r["dh"])],
    "samp(ratio calib)": _Ratio(),
    "samp+vol": lambda r: [math.log(r["dh"]), r["vol"]],
    "free20k(A4)": lambda r: r["a4"],
    "free20k+samp": lambda r: r["a4"] + [math.log(r["dh"])],
}


def fit(X, y, ratio=False):
    X = np.asarray(X, float)
    if ratio:
        return np.array([float(np.median(np.asarray(y) - X[:, 0])), 1.0]), np.zeros(1), np.ones(1)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    A = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    coef, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return coef, mu, sd


def predict(m, X):
    coef, mu, sd = m
    return coef[0] + ((np.asarray(X, float) - mu) / sd) @ coef[1:]


def harrell(pred, d, exact):
    """Harrell's C for right-censored d (censored rows: d is a lower bound)."""
    num = den = 0.0
    n = len(d)
    for i in range(n):
        if not exact[i]:
            continue
        for j in range(n):
            if i == j or not d[i] < d[j]:
                continue
            den += 1
            num += 1.0 if pred[i] < pred[j] else (0.5 if pred[i] == pred[j] else 0.0)
    return num / den if den else float("nan")


def score(rows, pred):
    p = np.asarray(pred, float)
    ex = [r["exact"] for r in rows]
    t = np.array([math.log(r["d"]) for r in rows])
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["cell"]].append(i)
    per, cper = {}, {}
    for c, idx in by.items():
        ie = [i for i in idx if ex[i]]
        if len(ie) >= 8:
            per[c] = float(spearmanr(p[ie], t[ie]).correlation)
        if len(idx) >= 8:
            cper[c] = harrell(list(p[idx]), [rows[i]["d"] for i in idx], [ex[i] for i in idx])
    ie = [i for i in range(len(rows)) if ex[i]]
    out = dict(n_exact=len(ie), n_censored=len(rows) - len(ie),
               within=float(np.mean(list(per.values()))) if per else float("nan"),
               pooled=float(spearmanr(p[ie], t[ie]).correlation),
               log_rmse=float(np.sqrt(np.mean((p[ie] - t[ie]) ** 2))),
               per_cell={c: round(v, 3) for c, v in per.items()})
    if len(ie) < len(rows):
        out["harrell_within"] = float(np.mean(list(cper.values())))
        out["harrell_pooled"] = harrell(list(p), [r["d"] for r in rows], ex)
        out["harrell_per_cell"] = {c: round(v, 3) for c, v in cper.items()}
    return out


def fmt(s):
    x = f"within={s['within']:.3f} pooled={s['pooled']:.3f} lrmse={s['log_rmse']:.3f}"
    if "harrell_within" in s:
        x += f" | C_within={s['harrell_within']:.3f} C_pooled={s['harrell_pooled']:.3f}"
    return x


def loco(rows, fx):
    preds = {}
    for c in sorted(set(r["cell"] for r in rows)):
        tr = [r for r in rows if r["cell"] != c]
        te = [r for r in rows if r["cell"] == c]
        m = fit([fx(r) for r in tr], [math.log(r["d"]) for r in tr], ratio=getattr(fx, "ratio", False))
        for r, p in zip(te, predict(m, [fx(r) for r in te])):
            preds[r["key"]] = float(p)
    return score(rows, [preds[r["key"]] for r in rows])


def baselines(rows, dev_rows):
    out = {}
    lab = [math.log(min(max(fhat(CALIB, r["c2000"], r["vol"]), 20000), 400000)) for r in rows]
    out["current label clip(fhat,20k,400k)"] = score(rows, lab)
    for C in (50_000, 100_000, 200_000):
        out[f"direct fresh run, cap {C // 1000}k"] = dict(score(rows, [math.log(min(r["d"], C)) for r in rows]),
                                                        cost_mean=float(np.mean([min(r["d"], C) for r in rows])),
                                                        frac_open=float(np.mean([r["d"] > C for r in rows])))
    return out


def hybrid(rows, dev_rows, C, model="samp"):
    """Direct fresh run at cap C; the cases still open are ordered above C by a frozen model
    (fitted on the DEV hard cases).  Cost: min(d, C) + the sampling cost of the open cases when the
    model uses the sampling estimate (A4's free20k inputs are free: the table's 20k probe)."""
    fx = MODELS[model]
    tr = [r for r in dev_rows if "free20k" not in model or r["a4"] is not None]
    rows = [r for r in rows if "free20k" not in model or r["a4"] is not None]
    m = fit([fx(r) for r in tr], [math.log(r["d"]) for r in tr])
    ps = predict(m, [fx(r) for r in rows])
    lo, hi = float(min(ps)), float(max(ps))
    key = [math.log(r["d"]) if r["d"] <= C else math.log(C) + 1e-3 * (1 + (p - lo) / (hi - lo + 1e-9))
           for r, p in zip(rows, ps)]
    s = score(rows, key)
    s["log_rmse"] = float("nan")  # a ranking construct (open cases sit just above C), not an estimate
    uses = "samp" in model
    s["cost_mean"] = float(np.mean([min(r["d"], C) + (r["cost"] if (uses and r["d"] > C) else 0) for r in rows]))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "final_eval.json"))
    ap.add_argument("--cfg", default="knuth:row:8")
    ap.add_argument("--dev-runs", default=os.path.join(D, "dev.jsonl"))
    ap.add_argument("--wide-runs", default=os.path.join(D, "wide_test.jsonl"))
    ap.add_argument("--target-runs", default=os.path.join(D, "target.jsonl"))
    a = ap.parse_args()
    gt_init = {key_of(r): r for r in jl([os.path.join(HERE, "ground_truth_initial.jsonl")]) if r.get("d", 0) > 20000}
    a4dev = {key_of(r): r for r in jl([os.path.join(HERE, "progress_features_dev.jsonl")])}
    dev_feats = {}
    for k, g in gt_init.items():
        f = dict(g)
        f["features"] = a4dev.get(k, {}).get("features", {})
        dev_feats[k] = f
    wide_feats = {key_of(r): r for r in jl([os.path.join(HERE, f"progress_features_test{s}.jsonl") for s in ("", "2", "3")])}
    tgt_feats = {r["key"]: r for r in jl([os.path.join(D, "target_a4.jsonl")])}
    tgt_gt = {key_of(r): r for r in jl([os.path.join(HERE, "ground_truth.jsonl")])
              if r["cell"] in ("m12_n18_s3_t3_w109", "m13_n19_s3_t3_w123", "m16_n17_s3_t3_w134") and r["source"] == "deepen"}
    for k, g in tgt_gt.items():  # sampling rows even where A4 features are missing
        if k not in tgt_feats:
            tgt_feats[k] = dict(g, features={})
    dev_runs = jl([a.dev_runs])
    wide_runs = jl([a.wide_runs])
    tgt_runs = jl([a.target_runs])
    cfg = a.cfg
    ops = [("N", N, b) for N in (20, 50, 100) for b in (500, 2000, 5000, 10000)] + \
          [("T", T, b) for T in (10_000, 20_000, 50_000, 100_000) for b in (2000, 5000, 10000)]
    res = {"cfg": cfg, "curve": []}
    print(f"config {cfg}")
    print(f"{'op':>14s} | {'DEV cost':>8s} {'LOCO w':>6s} {'pool':>6s} {'lrmse':>5s} | {'WIDE cost':>9s} {'within':>6s} {'pool':>6s} "
          f"{'lrmse':>5s} {'+A4 w':>6s} | {'TGT cost':>8s} {'within':>6s} {'C_w':>5s} {'C_pool':>6s} {'+A4 C_w':>7s}")
    for op in ops:
        dev = op_rows(dev_runs, dev_feats, cfg, op)
        wide = op_rows(wide_runs, wide_feats, cfg, op)
        tgt = op_rows(tgt_runs, tgt_feats, cfg, op)
        row = {"op": list(op), "dev_n": len(dev)}
        if len(dev) < 30:
            continue
        row["dev_cost"] = float(np.mean([r["cost"] for r in dev]))
        row["dev_props"] = float(np.mean([r["props"] for r in dev]))
        row["dev_secs"] = float(np.mean([r["secs"] for r in dev]))
        row["dev_loco_samp"] = loco(dev, MODELS["samp"])
        m = fit([MODELS["samp"](r) for r in dev], [math.log(r["d"]) for r in dev])
        dev_a4 = [r for r in dev if r["a4"] is not None]
        m2 = fit([MODELS["free20k+samp"](r) for r in dev_a4], [math.log(r["d"]) for r in dev_a4])
        for name, rows in (("wide", wide), ("target", tgt)):
            if len(rows) < 30:
                continue
            row[f"{name}_n"] = len(rows)
            row[f"{name}_cost"] = float(np.mean([r["cost"] for r in rows]))
            row[f"{name}_cost_max"] = float(np.max([r["cost"] for r in rows]))
            row[f"{name}_props"] = float(np.mean([r["props"] for r in rows]))
            row[f"{name}_secs"] = float(np.mean([r["secs"] for r in rows]))
            row[f"{name}_samp"] = score(rows, predict(m, [MODELS["samp"](r) for r in rows]))
            ra = [r for r in rows if r["a4"] is not None]
            if len(ra) >= 30:
                row[f"{name}_free20k+samp"] = score(ra, predict(m2, [MODELS["free20k+samp"](r) for r in ra]))
            row[f"{name}_eps_med"] = float(np.median([r["feat"]["eps_achieved"] for r in rows]))
            row[f"{name}_Nreq_med"] = float(np.median([r["feat"]["N_req"] for r in rows]))
            row[f"{name}_hit"] = float(np.mean([r["feat"]["frac_budget_hit"] for r in rows]))
            row[f"{name}_up"] = float(np.mean([r["feat"]["frac_up_refuted"] for r in rows]))
        res["curve"].append(row)
        w = row.get("wide_samp", {})
        t = row.get("target_samp", {})
        wa = row.get("wide_free20k+samp", {})
        ta = row.get("target_free20k+samp", {})
        nan = float("nan")
        print(f"{op[0]}{op[1]:>6d} b{op[2]:>5d} | {row['dev_cost']:8.0f} {row['dev_loco_samp']['within']:6.3f} "
              f"{row['dev_loco_samp']['pooled']:6.3f} {row['dev_loco_samp']['log_rmse']:5.2f} | "
              f"{row.get('wide_cost', nan):9.0f} {w.get('within', nan):6.3f} {w.get('pooled', nan):6.3f} {w.get('log_rmse', nan):5.2f} "
              f"{wa.get('within', nan):6.3f} | {row.get('target_cost', nan):8.0f} {t.get('within', nan):6.3f} "
              f"{t.get('harrell_within', nan):5.3f} {t.get('harrell_pooled', nan):6.3f} {ta.get('harrell_within', nan):7.3f}")
    # headline operating point: selected on DEV only (best DEV LOCO within-cell among DEV cost <= 50k)
    ok = [r for r in res["curve"] if r["dev_cost"] <= 50_000]
    best = max(ok, key=lambda r: r["dev_loco_samp"]["within"])
    # hard-capped alternative: the best budgeted (T <= 50k) point on DEV
    okT = [r for r in res["curve"] if r["op"][0] == "T" and r["op"][1] <= 50_000]
    bestT = max(okT, key=lambda r: r["dev_loco_samp"]["within"])
    res["selected"] = best["op"]
    res["selected_budgeted"] = bestT["op"]
    print(f"\nselected on DEV (cost <= 50k): {best['op']}; budgeted (T <= 50k): {bestT['op']}")
    for op in (tuple(best["op"]), tuple(bestT["op"])):
        dev = op_rows(dev_runs, dev_feats, cfg, op)
        dev_a4 = [r for r in dev if r["a4"] is not None]
        tag = f"{op[0]}{op[1]}_b{op[2]}"
        res[tag] = {}
        print(f"\n== operating point {tag} ==")
        print("DEV LOCO (4 square cells):")
        res[tag]["dev_loco"] = {}
        for mn, fx in MODELS.items():
            rr = dev_a4 if "free20k" in mn else dev
            s = loco(rr, fx)
            res[tag]["dev_loco"][mn] = s
            print(f"  {mn:18s} n={len(rr):4d} {fmt(s)}")
        for name, runs, feats in (("wide", wide_runs, wide_feats), ("target", tgt_runs, tgt_feats)):
            rows = op_rows(runs, feats, cfg, op)
            if len(rows) < 30:
                continue
            ra = [r for r in rows if r["a4"] is not None]
            res[tag][name] = {"n": len(rows), "n_a4": len(ra), "cost_mean": float(np.mean([r["cost"] for r in rows])),
                              "cost_max": float(np.max([r["cost"] for r in rows])),
                              "props_mean": float(np.mean([r["props"] for r in rows])),
                              "secs_mean_est": float(np.mean([r["secs"] for r in rows])), "models": {}}
            print(f"FROZEN on DEV -> {name} ({len(rows)} cases, {len(ra)} with A4 features; sampling cost "
                  f"{res[tag][name]['cost_mean']:.0f} conflicts mean, {res[tag][name]['cost_max']:.0f} max, "
                  f"{res[tag][name]['props_mean']:.3g} props, ~{res[tag][name]['secs_mean_est']:.2f} s):")
            for mn, fx in MODELS.items():
                if "free20k" in mn:
                    tr, te = dev_a4, ra
                else:
                    tr, te = dev, rows
                if len(te) < 30:
                    continue
                mod = fit([fx(r) for r in tr], [math.log(r["d"]) for r in tr], ratio=getattr(fx, "ratio", False))
                s = score(te, predict(mod, [fx(r) for r in te]))
                res[tag][name]["models"][mn] = s
                print(f"  {mn:18s} n={len(te):4d} {fmt(s)}  {s['per_cell']}")
            for bn, s in baselines(rows, dev).items():
                res[tag][name]["models"][bn] = s
                extra = f" cost={s['cost_mean']:.0f} open={s['frac_open']:.2f}" if "cost_mean" in s else ""
                print(f"  {bn:34s} {fmt(s)}{extra}")
            for C in (20_000, 50_000, 100_000):
                for hm in ("samp", "free20k(A4)", "free20k+samp"):
                    s = hybrid(rows, dev, C, hm)
                    res[tag][name]["models"][f"direct {C // 1000}k, open cases by {hm}"] = s
                    print(f"  {'direct ' + str(C // 1000) + 'k, open by ' + hm:34s} {fmt(s)} cost={s['cost_mean']:.0f}")
        # LOCO over every cell with hard labels (4 DEV + 5 WIDE + 3 TARGET), censored rows are
        # kept out of the fit but scored by Harrell's C
        allr = dev + op_rows(wide_runs, wide_feats, cfg, op) + op_rows(tgt_runs, tgt_feats, cfg, op)
        if len(allr) > len(dev):
            res[tag]["loco_all_cells"] = {}
            print(f"LOCO over all {len(set(r['cell'] for r in allr))} cells ({len(allr)} cases):")
            for mn, fx in MODELS.items():
                rr = [r for r in allr if r["a4"] is not None] if "free20k" in mn else allr
                preds = {}
                for c in sorted(set(r["cell"] for r in rr)):
                    tr = [r for r in rr if r["cell"] != c and r["exact"]]
                    te = [r for r in rr if r["cell"] == c]
                    m = fit([fx(r) for r in tr], [math.log(r["d"]) for r in tr], ratio=getattr(fx, "ratio", False))
                    for r, p in zip(te, predict(m, [fx(r) for r in te])):
                        preds[r["key"]] = float(p)
                sc = score(rr, [preds[r["key"]] for r in rr])
                res[tag]["loco_all_cells"][mn] = sc
                print(f"  {mn:18s} n={len(rr):4d} {fmt(sc)}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

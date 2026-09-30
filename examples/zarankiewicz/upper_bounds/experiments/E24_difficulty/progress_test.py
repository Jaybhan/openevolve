"""E24 / A4: apply the FROZEN progress models (progress_model.json, fitted on the dev cells only)
to cells that were not in the dev set (the wide cells A1 deepened after the models were frozen).

Reports, per test cell and pooled: Spearman(d_hat, d) and log-RMSE for the frozen "20k",
"free20k" and "2k" models vs the current fhat (calibrate_33.json) and the fhat form refit on the
dev cells; plus the within-cell Spearman of the top dev single features (sign fixed on dev).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
UB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UB)
from progress_eval import fit_ols, load, logrmse, pred_ols, rho  # noqa: E402

from zar_ub import hardness_progress as hp  # noqa: E402
from zar_ub.difficulty import fhat, load_calibration  # noqa: E402


def block(recs, preds, name):
    d = np.array([float(r["d"]) for r in recs])
    p = np.array(preds, float)
    cells = sorted(set(r["cell"] for r in recs))
    per = {}
    for c in cells:
        ix = [i for i, r in enumerate(recs) if r["cell"] == c]
        per[c] = {"n": len(ix), "rho": rho(p[ix], d[ix]), "logrmse": logrmse(p[ix], d[ix])}
    vals = [(v["rho"], v["n"]) for v in per.values() if not math.isnan(v["rho"]) and v["n"] >= 8]
    return {
        "model": name,
        "n": len(recs),
        "pooled_rho": rho(p, d),
        "within_rho_wmean": (
            (sum(a * n for a, n in vals) / sum(n for _, n in vals)) if vals else float("nan")
        ),
        "logrmse": logrmse(p, d),
        "per_cell": per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=os.path.join(HERE, "progress_features_dev.jsonl"))
    ap.add_argument("--test", default=os.path.join(HERE, "progress_features_test.jsonl"))
    ap.add_argument("--tag", default="test")
    ap.add_argument(
        "--model",
        default=os.path.join(HERE, "progress_model_frozen_dev.json"),
        help="the FROZEN dev-only models",
    )
    a = ap.parse_args()
    model = hp.load_model(a.model)
    dev = load(a.dev)
    test = load(a.test)
    coef = load_calibration(3, 3)
    out = {
        "frozen": model["frozen"],
        "models": {k: v["features"] for k, v in model["models"].items()},
    }
    for reg, tier, mnames, cap in (
        ("hard", "20k", ("20k", "free20k", "2k"), 20_000),
        ("midhard", None, ("2k",), 2_000),
    ):
        regs = ("hard",) if reg == "hard" else ("hard", "mid")
        tr = [r for r in test if r["regime"] in regs and (tier is None or r["tier"] == tier)]
        dv = [r for r in dev if r["regime"] in regs and (tier is None or r["tier"] == tier)]
        if not tr:
            continue
        res = []
        fh = [
            min(
                max(cap, fhat(coef, int(r.get("c2000") or 2000), float(r["log2_volume"]))), 20 * cap
            )
            for r in tr
        ]
        res.append(block(tr, fh, "current fhat, censored-label clip [cap, 20cap]"))
        fh_raw = [fhat(coef, int(r.get("c2000") or 2000), float(r["log2_volume"])) for r in tr]
        res.append(block(tr, fh_raw, "current fhat, unclipped"))
        # fhat form refit on dev
        Xd = np.array(
            [
                [math.log(max(min(int(r.get("c2000") or 2000), 2000), 1)), float(r["log2_volume"])]
                for r in dv
            ]
        )
        yd = np.log([float(r["d"]) for r in dv])
        m = fit_ols(Xd, yd)
        Xt = np.array(
            [
                [math.log(max(min(int(r.get("c2000") or 2000), 2000), 1)), float(r["log2_volume"])]
                for r in tr
            ]
        )
        res.append(block(tr, np.exp(pred_ols(m, Xt)), "fhat form refit on dev"))
        res.append(block(tr, [float(np.exp(yd.mean()))] * len(tr), "constant (dev mean)"))
        for mn in mnames:
            mm = model["models"][mn]
            res.append(
                block(
                    tr, [hp.predict(mm, r["features"]) for r in tr], f"frozen {mn} {mm['features']}"
                )
            )
        out[reg] = res
        print(f"\n## {reg} (test cells, n={len(tr)})")
        print("| model | within-cell rho | pooled rho | log-RMSE | per cell (n, rho, log-RMSE) |")
        print("|---|---|---|---|---|")
        for b in res:
            pc = "; ".join(
                f"{c.replace('_s3_t3', '')}: {v['n']}, {v['rho']:.2f}, {v['logrmse']:.2f}"
                for c, v in b["per_cell"].items()
            )
            print(
                f"| {b['model']} | {b['within_rho_wmean']:.3f} | {b['pooled_rho']:.3f} | {b['logrmse']:.3f} | {pc} |"
            )
        # transfer of single features (top 15 on dev by within-cell rho)
        ev = json.load(open(os.path.join(HERE, "progress_eval_dev.json")))
        key = "hard" if reg == "hard" else "mid"
        scr = ev[key]["screen"]
        top = sorted(
            [f for f in scr if scr[f]["within_wmean"] is not None and "_raw_" not in f],
            key=lambda f: -abs(scr[f]["within_wmean"]),
        )[:20]
        print(f"\n| feature | dev within rho | test per-cell rho |")
        print("|---|---|---|")
        sing = {}
        for f in top:
            pcs = {}
            for c in sorted(set(r["cell"] for r in tr)):
                ix = [r for r in tr if r["cell"] == c]
                pcs[c] = rho([r["features"].get(f, 0.0) for r in ix], [float(r["d"]) for r in ix])
            sing[f] = {"dev": scr[f]["within_wmean"], "test": pcs}
            print(
                f"| {f} | {scr[f]['within_wmean']:.3f} | "
                + ", ".join(f"{c.replace('_s3_t3', '')}: {v:.2f}" for c, v in pcs.items())
                + " |"
            )
        out[reg + "_single"] = sing
    json.dump(out, open(os.path.join(HERE, f"progress_{a.tag}.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

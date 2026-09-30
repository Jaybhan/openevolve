"""E24 / A2-lookahead: FREEZE the lookahead difficulty models on the DEV set before any TEST data
is featurised.  Writes lookahead_model.json:

  primary     (pre-registered 2026-09-22 22:05, before the TEST features existed):
              FL-only  log d ~ log2_volume + fl_free_cells + fl_log2vol_rows
  alternates  vol+fl_free+knfl, vol+la1_failed_frac+ams_mean, greedy (forward selection by LOCO
              log-RMSE on all DEV d > 2000 cases)

Each model: slopes fitted on every DEV case with exact d > 2000 (H u M); `hard_shift` = mean
residual of the DEV HARD cases (d > 20000), so  log d_hard = model + hard_shift  is the estimate
for a case known to be unsolved at 20k conflicts (the censored cases on target tables).

usage: python lookahead_freeze.py [--in lookahead_features_dev.jsonl]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from lookahead_eval import HERE, fit, greedy, load, predict

MODELS = {
    "fl_only": ["log2_volume", "fl_free_cells", "fl_log2vol_rows"],
    "vol_fl_knfl": ["log2_volume", "fl_free_cells", "knfl_rand_mean_log_conf"],
    "vol_la1fail_ams": ["log2_volume", "la1_failed_frac", "ams_score_mean"],
}
# estimate() kwargs needed for each model's features (fl_only needs no pairs/Knuth/clause measures)
KWARGS = {
    "fl_only": {"n_pairs": 0, "n_probes": 0, "clause_measures": False},
    "vol_fl_knfl": {"n_pairs": 0, "clause_measures": False, "knuth_orders": ()},
    "vol_la1fail_ams": {"n_pairs": 0, "n_probes": 0, "clause_measures": False},
}


def pack(rows, H, feats, kwargs):
    fe, mu, sd, beta = fit(rows, feats)
    shift = float(np.mean([r["logd"] - v for r, v in zip(H, predict((fe, mu, sd, beta), H))]))
    return {
        "features": fe,
        "mu": list(map(float, mu)),
        "sd": list(map(float, sd)),
        "beta": list(map(float, beta)),
        "hard_shift": shift,
        "estimate_kwargs": kwargs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(HERE, "lookahead_features_dev.jsonl"))
    a = ap.parse_args()
    rs = load(a.inp)
    HM = [r for r in rs if r["subset"] in ("H", "M")]
    H = [r for r in rs if r["subset"] == "H"]
    out = {
        "target": "log d, d = CaDiCaL 1.9.5 conflicts; fitted on exact d > 2000 DEV cases",
        "n_fit": len(HM),
        "n_hard": len(H),
        "fitted_on": sorted(set(r["cell"] for r in HM)),
        "source": os.path.basename(a.inp),
        "frozen": "2026-09-22 (before TEST featurisation)",
        "usage": "log_d = beta0 + sum beta_k (f_k - mu_k)/sd_k ; add hard_shift if the case is known "
        "to be unsolved at 20k conflicts",
        "primary": "fl_only",
        "models": {},
    }
    for nm, fs in MODELS.items():
        out["models"][nm] = pack(HM, H, fs, KWARGS[nm])
    g = greedy(HM, "cell")
    out["models"]["greedy"] = pack(HM, H, g, {})
    json.dump(out, open(os.path.join(HERE, "lookahead_model.json"), "w"), indent=1)
    for nm, m in out["models"].items():
        print(
            nm,
            dict(zip(["b0"] + m["features"], [round(b, 3) for b in m["beta"]])),
            "shift %.3f" % m["hard_shift"],
        )


if __name__ == "__main__":
    main()

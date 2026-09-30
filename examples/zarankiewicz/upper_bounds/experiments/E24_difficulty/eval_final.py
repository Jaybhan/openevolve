"""E24 / D5-evaluate: fit the WINNER on all data and save it via zar_ub/hardness_model.py.

Winner (EVALUATION.md section 5): the nested-greedy log-linear model over the "free" tier
(profile + static/LP slack + lookahead + the pysat 20k-probe statistics; no binary, no sampling).
Here the same greedy procedure (inner leave-one-shape-out log-RMSE, <= 6 features, stop at
< 0.5 % gain, |r| > 0.95 guard) is run ONCE on all 2,238 exact hard cases, the model is refitted
with zar_ub.hardness_model.fit and saved to hardness_model.json.  Its expected held-out
performance is the nested LOSO result in eval_results.json (greedy[free(ST+LA+PS)]).
Then predict() is checked end-to-end on seeded cases (recomputing every feature from scratch).
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UB)
sys.path.insert(0, HERE)

import eval_analyze as ea  # noqa: E402
from zar_ub import hardness_model as hm  # noqa: E402
from zar_ub.known import Instance  # noqa: E402

TIER = ("BASE", "ST", "LA", "PS")


def main():
    rows_all, costs, _ = ea.load_data()
    hard = [r for r in rows_all if r["regime"] in ("hard", "hard_open")]
    names = ea.feature_names(rows_all)
    tr = ea.build_transforms(rows_all, names)
    ex = [r for r in hard if r["exact"]]
    X = ea.impute(ea.matrix(ex, names, tr))
    y = np.array([r["logd"] for r in ex])
    cells = np.array([r["cell"] for r in ex])
    groups = np.array([r["group"] for r in ex])
    pool = [j for j, n in enumerate(names) if ea.family(n) in TIER]
    g = ea.Greedy(pool).fit(X, y, cells, groups)
    feats = [names[j] for j in g.sel]
    print("selected:", feats)
    model = hm.fit([r["f"] for r in ex], [r["logd"] for r in ex], feats, {k: tr[k] for k in feats},
                   meta={"source": "experiments/E24_difficulty/eval_final.py", "tier": "free(ST+LA+PS)",
                         "trained_on": f"{len(ex)} exact hard cases (20k < d <= 2M), {len(set(cells))} cells",
                         "selection": "greedy forward, inner leave-one-shape-out log-RMSE, all data",
                         "heldout_expected": "see EVALUATION.md: nested LOSO of greedy[free(ST+LA+PS)]",
                         "recommended_ceiling": 2_000_000,
                         "use": "censored cases (unsolved by a fresh 20k pysat cadical195 probe); mean=True for sums"})
    path = model.save()
    print("saved", path)
    print(json.dumps({k: model.coef[k] for k in feats}, indent=1), "intercept", model.intercept, "smear", model.smear,
          "in-sample rmse", model.meta["rmse_in_sample"])
    # in-sample sanity: model on the stored features
    p = np.array([model.log_d(r["f"]) for r in ex])
    print("in-sample within/pooled:", ea.metrics(ex, p)["within_rho"], ea.metrics(ex, p)["pooled_rho"])
    # end-to-end predict() on seeded hard cases (features recomputed from scratch)
    rng = random.Random(24)
    sample = rng.sample(ex, 6)
    out = []
    for r in sample:
        inst = Instance(r["m"], r["n"], r["s"], r["t"], r["w"])
        o = hm.predict(inst, r["rows"], r["cols"], model=model, mean=False)
        ref = model.predict_features(r["f"], mean=False)
        # zero-cost path from the probe statistics
        pp = hm.predict_from_probe(inst, r["rows"], r["cols"], conflicts=int(r["f"]["pr:ps20k_conflicts"]),
                                   decisions=int(r["f"]["pr:ps20k_decisions"]), restarts=int(r["f"]["pr:ps20k_restarts"]),
                                   propagations=int(r["f"]["pr:ps20k_propagations"]), model=model, mean=False)
        out.append({"cell": r["cell"], "d": r["d"], "predict": o["d_hat"], "from_cached_features": ref,
                    "predict_from_probe": pp, "exact": o["exact"], "cost_conflicts": o["cost_conflicts"],
                    "cost_propagations": o["cost_propagations"], "cost_seconds": round(o["cost_seconds"], 3)})
        print(out[-1], flush=True)
    json.dump({"selected": feats, "check": out}, open(os.path.join(HERE, "features_final_check.jsonl"), "w"))


if __name__ == "__main__":
    main()

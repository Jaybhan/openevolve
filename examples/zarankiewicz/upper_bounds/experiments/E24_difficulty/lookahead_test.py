"""E24 / A2-lookahead: TEST the frozen lookahead models (lookahead_model.json, frozen on DEV by
lookahead_freeze.py) on labels that did not exist when the features were designed: A1's
deepening / new-table runs (ground_truth_deepen_raw.jsonl).

  HARD test: every case with status unsat/sat and exact d > 20000 (source deepen or new_table)
  MID  test: exact 2000 < d <= 20000 from the same file, up to 150 per cell (seed 24)
  Cases still `unknown` at the 2M cap are excluded (and counted): the test is selected toward
  "hard but refutable within 2M conflicts".

Steps:  python lookahead_test.py featurize [--jobs 2]   -> lookahead_features_test.jsonl
        python lookahead_test.py evaluate                -> lookahead_test.json + printed report
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UB)
sys.path.insert(0, HERE)

from zar_ub.difficulty import log2_volume, spearman  # noqa: E402
from zar_ub.known import Instance  # noqa: E402

RAW = os.path.join(HERE, "ground_truth_deepen_raw.jsonl")
OUT = os.path.join(HERE, "lookahead_features_test.jsonl")


def build():
    rows = [json.loads(l) for l in open(RAW)]
    H, M, unk = [], collections.defaultdict(list), collections.Counter()
    seen = set()
    for r in rows:
        k = (r["cell"], tuple(r["rows"]), tuple(r["cols"]))
        if k in seen:
            continue
        seen.add(k)
        inst = Instance(**r["inst"])
        g = {
            "cell": r["cell"],
            "m": inst.m,
            "n": inst.n,
            "s": inst.s,
            "t": inst.t,
            "w": inst.w,
            "trust": r["trust"],
            "rows": r["rows"],
            "cols": r["cols"],
            "status": r["status"],
            "d": r["d"],
            "cap": r["cap"],
            "c2000": r["c2000"],
            "log2_volume": log2_volume(inst, r["rows"]),
            "source": r["source"],
        }
        if r["status"] == "unknown":
            unk[r["cell"]] += 1
        elif r["status"] in ("unsat", "sat") and r["d"] is not None and r["d"] > 20000:
            H.append(("H", g))
        elif r["status"] in ("unsat", "sat") and r["d"] is not None and 2000 < r["d"] <= 20000:
            M[r["cell"]].append(("M", g))
    sel = list(H)
    for c in sorted(M):
        lst = M[c]
        random.Random(f"24:{c}").shuffle(lst)
        sel += lst[:150]
    return sel, unk


def featurize(jobs):
    from lookahead_features import _work, key

    sel, unk = build()
    done = set()
    if os.path.exists(OUT):
        done = {key(json.loads(l)) for l in open(OUT)}
    todo = [(s, r, {}) for s, r in sel if key(r) not in done]
    print(
        f"test selection {collections.Counter(s for s, _ in sel)}; unknown-at-2M excluded {dict(unk)}; todo {len(todo)}",
        flush=True,
    )
    t0 = time.time()
    with mp.get_context("spawn").Pool(jobs) as pool, open(OUT, "a") as f:
        for k, o in enumerate(pool.imap_unordered(_work, todo, chunksize=4)):
            f.write(json.dumps(o) + "\n")
            if (k + 1) % 200 == 0:
                f.flush()
                print(f"  {k + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def evaluate():
    import numpy as np
    from lookahead_eval import FIXED, by_cell, lrmse, nanmean, top_recall
    from lookahead_eval import load as eval_load
    from zar_ub.hardness_lookahead import load_model, model_log_d

    rs = eval_load(OUT)
    _, unk = build()
    dev = eval_load(os.path.join(HERE, "lookahead_features_dev.jsonl"))
    mfile = json.load(open(os.path.join(HERE, "lookahead_model.json")))
    dev_hard_mean = float(np.mean([r["logd"] for r in dev if r["subset"] == "H"]))
    H = [r for r in rs if r["subset"] == "H"]
    M = [r for r in rs if r["subset"] == "M"]
    res = {
        "counts": {
            "H": dict(collections.Counter(r["cell"] for r in H)),
            "M": dict(collections.Counter(r["cell"] for r in M)),
            "unknown_at_2M_excluded": dict(unk),
        },
        "model_file": mfile["frozen"],
        "primary": mfile["primary"],
    }

    # predictors: name -> function(row) -> predicted log d
    preds = {}
    for nm in mfile["models"]:
        mdl = load_model(name=nm)
        preds[f"{nm} (+hard_shift)"] = (
            lambda mdl: lambda r: model_log_d(mdl, r["F"]) + mdl["hard_shift"]
        )(mdl)
        preds[f"{nm} (raw)"] = (lambda mdl: lambda r: model_log_d(mdl, r["F"]))(mdl)
    a, b, g = FIXED
    preds["fhat fixed (E11)"] = lambda r: a + b * r["F"]["log_c2000"] + g * r["F"]["log2_volume"]
    preds["fhat clipped label (current)"] = lambda r: math.log(
        min(
            max(20000.0, math.exp(a + b * r["F"]["log_c2000"] + g * r["F"]["log2_volume"])),
            400000.0,
        )
    )
    preds["constant (DEV hard mean)"] = lambda r: dev_hard_mean
    for f in (
        "fl_free_cells",
        "fl_log2vol_rows",
        "la1_failed_frac",
        "knfl_rand_mean_log_conf",
        "kn_rand_mean_log_conf",
        "ams_score_mean",
        "la1_mean",
        "log2_volume",
    ):
        sgn = (
            -1.0 if f in ("la1_failed_frac", "ams_score_mean", "la1_mean") else 1.0
        )  # signs fixed on DEV
        preds[f"feature: {'-' if sgn < 0 else '+'}{f}"] = (
            lambda f, sgn: lambda r: sgn * r["F"].get(f, 0.0)
        )(f, sgn)

    out = {}
    for reg, R in (("HARD", H), ("MID", M)):
        out[reg] = {}
        cells = by_cell(R)
        for nm, fn in preds.items():
            p = [fn(r) for r in R]
            wc = {
                c: spearman([fn(r) for r in g], [r["d"] for r in g])
                for c, g in cells.items()
                if len(g) >= 8
            }
            tr = {c: top_recall([fn(r) for r in g], g) for c, g in cells.items() if len(g) >= 20}
            is_model = not nm.startswith("feature:")
            out[reg][nm] = {
                "n": len(R),
                "pooled_spearman": spearman(p, [r["d"] for r in R]),
                "mean_within_cell_spearman": nanmean(list(wc.values())),
                "within_cell": wc,
                "log_rmse": lrmse(p, R) if is_model else None,
                "top10_recall_mean": nanmean(list(tr.values())),
            }
    res["results"] = out
    res["cost"] = {
        q: float(np.mean([r[q] for r in rs]))
        for q in ("cost_seconds", "cost_calls", "cost_propagations", "cost_conflicts")
    }
    json.dump(res, open(os.path.join(HERE, "lookahead_test.json"), "w"), indent=1, default=float)

    print("counts", {k: sum(v.values()) for k, v in res["counts"].items()}, res["counts"])
    for reg in ("HARD", "MID"):
        print(f"\n[{reg}]  pooled | mean within-cell | log-RMSE | top10   per-cell within")
        for nm, s in out[reg].items():
            print(
                "  %-40s %+.3f | %+.3f | %s | %.2f  %s"
                % (
                    nm,
                    s["pooled_spearman"],
                    s["mean_within_cell_spearman"],
                    "%.3f" % s["log_rmse"] if s["log_rmse"] is not None else "  -  ",
                    s["top10_recall_mean"],
                    {k.split("_s3")[0]: round(v, 2) for k, v in s["within_cell"].items()},
                )
            )
    print("cost", res["cost"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["featurize", "evaluate"])
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    featurize(a.jobs) if a.step == "featurize" else evaluate()

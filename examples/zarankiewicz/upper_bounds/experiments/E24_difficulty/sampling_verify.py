"""E24 / A3: reproducibility check of zar_ub.hardness_sampling.estimate against the stored runs.

For a seeded sample of DEV hard cases with a stored knuth:row:8 record (N=100, b=10000):
  (1) rerun estimate(..., N=100, budget=10000, total_budget=None) and compare every cube's
      (conflicts, propagations, censored, weight, up_refuted) with the stored record;
  (2) run the DEFAULT operating point (budgeted T=50k, b=5000) and compare its raw mu~ and cost
      with the offline derivation (sampling_final.derive_budgeted) from the stored record.
usage: python experiments/E24_difficulty/sampling_verify.py [n_cases] [knuth:row:8|strat:row:8:5]  (one process)"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from sampling_final import derive_budgeted  # noqa: E402
from zar_ub import hardness_sampling as hs  # noqa: E402
from zar_ub.known import Instance  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CFG = sys.argv[2] if len(sys.argv) > 2 else "knuth:row:8"
SRC = {"knuth:row:8": "dev.jsonl", "strat:row:8:5": "dev_strat.jsonl"}[CFG]
SAMPLER, _, K = CFG.split(":")[:3]
DEPTH = int(CFG.split(":")[3]) if CFG.count(":") == 3 else 3
recs = [json.loads(l) for l in open(os.path.join(HERE, "sampling_data", SRC))]
recs = [r for r in recs if r["config"] == CFG and r["d"] > 20000]
sample = random.Random(7).sample(recs, n)
out = []
for rec in sample:
    cell, rows, cols = rec["key"].split("|")
    rows, cols = [int(x) for x in rows.split(",")], [int(x) for x in cols.split(",")]
    p = {x[0]: x[1:] for x in cell.split("_") if x[0] in "mnstw" and x[1:].isdigit()}
    inst = Instance(int(p["m"]), int(p["n"]), int(p["s"]), int(p["t"]), int(p["w"]))
    e = hs.estimate(inst, rows, cols, design="row", sampler=SAMPLER, k=int(K), N=100, budget=10000, total_budget=None,
                    strata_depth=DEPTH, calibrated=False, return_records=True)
    got = [[r["conflicts"], r["propagations"], int(r["censored"]), r["weight"], int(r["up_refuted"])] for r in e["records"]]
    want = [x[:5] for x in rec["recs"]]
    same_conf = [g[0] for g in got] == [w[0] for w in want]
    same_all = got == want
    kw = {} if SAMPLER == hs.DEFAULT["sampler"] else dict(sampler=SAMPLER, budget=5000, strata_depth=DEPTH)
    e2 = hs.estimate(inst, rows, cols, **kw)  # default (budgeted) operating point of this sampler
    dv, nc = derive_budgeted(rec, hs.DEFAULT["total_budget"], kw.get("budget", hs.DEFAULT["budget"]))
    row = dict(key=rec["key"], d=rec["d"], cubes_identical=same_all, conflicts_identical=same_conf,
               default_mu_raw=e2["d_hat_raw"], offline_mu=dv["d_hat"], default_cost=e2["cost_conflicts"],
               offline_cost=dv["cost_conflicts"], n_cubes=e2["features"]["n_cubes"], offline_n=nc,
               default_d_hat=e2["d_hat"], seconds=e2["cost_seconds"])
    out.append(row)
    print(json.dumps(row), flush=True)
json.dump(out, open(os.path.join(HERE, "sampling_data", f"verify_{CFG.replace(':', '_')}.json"), "w"), indent=1)
print("all cubes identical:", all(r["cubes_identical"] for r in out),
      "| default == offline:", all(abs(r["default_mu_raw"] - r["offline_mu"]) < 1e-6 * max(1, r["offline_mu"]) for r in out))

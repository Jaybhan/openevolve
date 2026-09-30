"""E24 / A3: test-retest reliability of the sampling estimator (seed 0 vs seed 1, same
config) on DEV hard cases, and the accuracy of the 2-seed average (= N doubled).
Separates sampling noise from estimator-vs-d mismatch: if Spearman(seed0, seed1) is
not much higher than Spearman(d_hat, d), noise is the bottleneck and larger N helps.

usage: python experiments/E24_difficulty/sampling_retest.py [cfg]"""
import json, math, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sampling_eval import derive  # noqa: E402

cfg = sys.argv[1] if len(sys.argv) > 1 else "knuth:row:8"
D = os.path.join(HERE, "sampling_data")
def load(p):
    out = {}
    for l in open(p):
        r = json.loads(l)
        if r.get("config") == cfg and "error" not in r and r["d"] > 20000:
            out[r["key"]] = r
    return out
a, b = load(os.path.join(D, "dev.jsonl")), load(os.path.join(D, "dev_seed1.jsonl"))
keys = sorted(set(a) & set(b))
res = {"config": cfg, "n": len(keys), "rows": []}
for N, B in [(20, 2000), (50, 2000), (100, 2000), (50, 10000), (100, 10000)]:
    x = [math.log(max(derive(a[k], N, B)["d_hat"], 1)) for k in keys]
    y = [math.log(max(derive(b[k], N, B)["d_hat"], 1)) for k in keys]
    d = [math.log(a[k]["d"]) for k in keys]
    cells = [a[k]["cell"] for k in keys]
    avg = [math.log((math.exp(p) + math.exp(q)) / 2) for p, q in zip(x, y)]
    def within(v):
        by = defaultdict(list)
        for c, p, t in zip(cells, v, d):
            by[c].append((p, t))
        return float(np.mean([spearmanr([p for p, _ in w], [t for _, t in w]).correlation for w in by.values() if len(w) >= 8]))
    row = dict(N=N, b=B, retest_spearman=float(spearmanr(x, y).correlation),
               retest_sd_log=float(np.std(np.array(x) - np.array(y)) / math.sqrt(2)),
               seed0_vs_d=float(spearmanr(x, d).correlation), seed1_vs_d=float(spearmanr(y, d).correlation),
               avg2_vs_d=float(spearmanr(avg, d).correlation), seed0_within=within(x), seed1_within=within(y),
               avg2_within=within(avg), sd_log_d=float(np.std(d)))
    res["rows"].append(row)
    print(f"N{N:4d} b{B:6d}  retest rho={row['retest_spearman']:.3f} noise sd(log)={row['retest_sd_log']:.3f} "
          f"(sd log d = {row['sd_log_d']:.3f}) | vs d: seed0 {row['seed0_vs_d']:.3f} seed1 {row['seed1_vs_d']:.3f} "
          f"avg {row['avg2_vs_d']:.3f} | within: {row['seed0_within']:.3f} {row['seed1_within']:.3f} avg {row['avg2_within']:.3f}")
json.dump(res, open(os.path.join(D, f"retest_{cfg.replace(':', '_')}.json"), "w"), indent=1)

"""E24 / A3: hybrid policy on exact hard cases: a direct fresh CaDiCaL run capped at C
(outcome = min(d, C), deterministic), then, for the cases still censored, the sampling
estimate (LOCO-calibrated on the other cells' hard cases) decides the order above C.
Reports within-cell / pooled Spearman and mean total cost per hard case, for C in
{20k (free: the table pipeline already ran it), 50k, 100k}.

usage: python experiments/E24_difficulty/sampling_hybrid.py <runs.jsonl> <cfg> <N> <b>"""
import json, math, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sampling_eval import derive  # noqa: E402

runs, cfg, N, B = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = []
for l in open(runs):
    r = json.loads(l)
    if r.get("config") == cfg and "error" not in r and r["d"] > 20000 and r.get("exact", True):
        dv = derive(r, N, B)
        rows.append(dict(cell=r["cell"], d=float(r["d"]), dh=max(dv["d_hat"], 1.0), cost=dv["cost_conflicts"]))
cells = sorted(set(r["cell"] for r in rows))
# LOCO calibration of log d on log d_hat (hard only)
for c in cells:
    tr = [r for r in rows if r["cell"] != c]
    A = np.column_stack([np.ones(len(tr)), [math.log(r["dh"]) for r in tr]])
    coef, *_ = np.linalg.lstsq(A, [math.log(r["d"]) for r in tr], rcond=None)
    for r in rows:
        if r["cell"] == c:
            r["pred"] = float(coef[0] + coef[1] * math.log(r["dh"]))
out = {"config": cfg, "N": N, "b": B, "n": len(rows)}
for C in (20000, 50000, 100000):
    # rank key: exact d below C; above C: C * (1 + tiny) ordered by the calibrated prediction
    key = [(r["d"] if r["d"] <= C else C + math.exp(r["pred"]) * 1e-6 + 1) for r in rows]
    key_direct = [min(r["d"], C) for r in rows]
    t = [r["d"] for r in rows]
    by = defaultdict(list)
    for r, k, kd, tt in zip(rows, key, key_direct, t):
        by[r["cell"]].append((k, kd, tt))
    wh = np.mean([spearmanr([a for a, _, _ in v], [x for _, _, x in v]).correlation for v in by.values() if len(v) >= 8])
    wd = np.nanmean([spearmanr([b for _, b, _ in v], [x for _, _, x in v]).correlation for v in by.values() if len(v) >= 8])
    cost_h = np.mean([min(r["d"], C) + (r["cost"] if r["d"] > C else 0) for r in rows])
    cost_d = np.mean([min(r["d"], C) for r in rows])
    frac_c = np.mean([r["d"] > C for r in rows])
    out[str(C)] = dict(hybrid_within=float(wh), hybrid_pooled=float(spearmanr(key, t).correlation),
                       direct_within=float(wd), cost_hybrid=float(cost_h), cost_direct=float(cost_d), frac_censored=float(frac_c))
    print(f"C={C:6d}: censored {frac_c:.2f} | direct-only within {wd:.3f} cost {cost_d:.0f} | hybrid within {wh:.3f} "
          f"pooled {out[str(C)]['hybrid_pooled']:.3f} cost {cost_h:.0f}")
json.dump(out, open(os.path.join(HERE, "sampling_data", f"hybrid_{os.path.basename(runs).split('.')[0]}_{cfg.replace(':', '_')}_N{N}_b{B}.json"), "w"), indent=1)

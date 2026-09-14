"""Determine complete gap bands: every (3,3) cell between Tan's exactness
frontier and the Roman equality window, for the rows within ILP reach.

Gap bands (from _EXACT_UP_TO + windows W(m) = 2C(m,3) - 3*T33(m)):
  m=8:  n = 24..27      (W=28)
  m=9:  n = 23..47      (W=48)
  m=10: n = 21..59      (W=60)
  m=11: n = 19..89 minus {21,22}   (W=90; 21,22 proven by BNL)  [partial run]
Rows >= 12 deferred (ILP hard); m=11 attempted with a shorter limit.

Above the window, z = 3n + floor((B-n)/3) is published (Roman/Tan) — no runs
needed. Below the band's start, Tan's table covers it. So finishing a band
DETERMINES THE ENTIRE ROW for that m (with Culik beyond B).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ilp_solver import solve_cell, verify  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = [
    (9, range(47, 22, -1), 600),
    (10, range(59, 20, -1), 600),
    (11, [n for n in range(89, 18, -1) if n not in (21, 22)], 600),
]
out_path = os.path.join(HERE, "ilp_gapband.jsonl")
wdir = os.path.join(HERE, "witnesses")
os.makedirs(wdir, exist_ok=True)

done = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if "z_ilp" in r:
                    done.add((r["m"], r["n"]))
            except Exception:
                pass

for m, ns, tl in BANDS:
    for n in ns:
        if (m, n) in done:
            continue
        t0 = time.time()
        edges, blocks, status = solve_cell(m, n, time_limit=tl)
        dt = time.time() - t0
        if edges is None:
            rec = {"m": m, "n": n, "status": f"UNRESOLVED({status})",
                   "seconds": round(dt, 1)}
            print(f"z({m},{n}) UNRESOLVED [{dt:.0f}s]", flush=True)
        else:
            okw = verify(blocks, m, n, edges)
            rec = {"m": m, "n": n, "z_ilp": edges, "witness_valid": okw,
                   "seconds": round(dt, 1)}
            print(f"z({m},{n}) = {edges}  witness_valid={okw} [{dt:.0f}s]",
                  flush=True)
            if okw:
                with open(os.path.join(wdir, f"w_{m}x{n}.json"), "w") as f:
                    json.dump({"m": m, "n": n, "edges": edges,
                               "blocks": [list(b) for b in blocks]}, f)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

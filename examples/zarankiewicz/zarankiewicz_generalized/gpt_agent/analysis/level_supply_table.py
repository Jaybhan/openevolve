"""Exact level packing numbers D2(m,w,3) = max # weight-w blocks (mult<=2),
every triple covered <= 2. Writes level_D2.csv incrementally."""
import sys, time, os
from itertools import combinations
from math import comb
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

out = "level_D2.csv"
done = set()
if os.path.exists(out):
    for line in open(out).readlines()[1:]:
        p = line.split(","); done.add((int(p[0]), int(p[1])))
else:
    open(out, "w").write("m,w,D2,status\n")

jobs = [(m, w) for m in range(6, 14) for w in range(5, min(m, 11) + 1)
        if comb(m, w) <= 4000 and (m, w) not in done]
for (m, w) in jobs:
    types = list(combinations(range(m), w))
    tris = {t: i for i, t in enumerate(combinations(range(m), 3))}
    A = lil_matrix((len(tris) + m - 1, len(types)))
    for j, b in enumerate(types):
        for t in combinations(b, 3): A[tris[t], j] = 1
        for r in range(m - 1):
            v = (1 if r in b else 0) - (1 if (r + 1) in b else 0)
            if v: A[len(tris) + r, j] = v
    lb = np.zeros(len(tris) + m - 1); ub = np.full(len(tris) + m - 1, 2.0)
    for r in range(m - 1): lb[len(tris)+r], ub[len(tris)+r] = 0, np.inf
    t0 = time.time()
    res = milp(c=-np.ones(len(types)), constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(len(types)),
               options={"time_limit": 900, "mip_rel_gap": 0.0})
    dt = time.time() - t0
    if res.status == 0:
        v = int(round(-res.fun)); st = "EXACT"
    else:
        v = -1; st = f"TIMEOUT"
    print(f"D2({m},{w}) = {v} [{st}, {dt:.0f}s]", flush=True)
    with open(out, "a") as f: f.write(f"{m},{w},{v},{st}\n")

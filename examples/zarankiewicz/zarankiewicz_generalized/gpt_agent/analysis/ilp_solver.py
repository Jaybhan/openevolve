"""Exact z(m,n;3,3) per cell via MILP (HiGHS through scipy.optimize.milp).

Formulation (exact, complete):
  variable x_b = multiplicity of column-block type b ⊆ [m], for |b| >= 2
    (columns of weight <=1 are never useful when a weight-2 pad is available;
     and are dominated: replacing a weight-<=1 column by any weight-2 column
     keeps legality — weight-2 columns cover no triple. For n exceeding all
     pair types' total we'd need them, but n <= 23 < 2*C(m,2) for m >= 6.)
  bounds: 0 <= x_b <= 2 for |b| >= 3 (3 copies of a block cover its own
    triples 3 times — instant K_{3,3}); 0 <= x_b <= n for |b| = 2.
  constraints: for every triple T: sum_{b ⊇ T} x_b <= 2;  sum_b x_b = n.
  objective: maximize sum_b |b| x_b.

The solver never sees the published table — comparison happens after.
Witness matrices are rebuilt from the solution and re-verified by an
independent checker before anything is reported.
"""
import argparse
import json
import os
import sys
import time
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def solve_cell(m, n, time_limit=600.0):
    types = []
    for w in range(2, m + 1):
        types.extend(combinations(range(m), w))
    tidx = {t: i for i, t in enumerate(combinations(range(m), 3))}
    ntri = len(tidx)
    nv = len(types)

    # Rows of the constraint matrix: triple capacities, column count, and
    # m-1 symmetry-breaking rows (monotone nonincreasing row degrees —
    # rows of the MATRIX are interchangeable, so forcing
    # deg(r) - deg(r+1) >= 0 collapses the S_m symmetry orbit).
    A = lil_matrix((ntri + 1 + (m - 1), nv))
    for j, b in enumerate(types):
        if len(b) >= 3:
            for t in combinations(b, 3):
                A[tidx[t], j] = 1
        A[ntri, j] = 1
        for r in range(m - 1):
            v = (1 if r in b else 0) - (1 if (r + 1) in b else 0)
            if v:
                A[ntri + 1 + r, j] = v
    lb = np.zeros(ntri + 1 + (m - 1))
    ub = np.full(ntri + 1 + (m - 1), 2.0)
    lb[ntri] = ub[ntri] = n  # exactly n columns
    for r in range(m - 1):
        lb[ntri + 1 + r] = 0.0
        ub[ntri + 1 + r] = np.inf

    c = -np.array([len(b) for b in types], dtype=float)
    vub = np.array([2.0 if len(b) >= 3 else float(n) for b in types])

    res = milp(
        c=c,
        constraints=LinearConstraint(A.tocsc(), lb, ub),
        bounds=Bounds(0, vub),
        integrality=np.ones(nv),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0},
    )
    if res.status != 0:
        return None, None, res.status
    edges = int(round(-res.fun))
    mult = np.round(res.x).astype(int)
    blocks = []
    for j, k in enumerate(mult):
        blocks.extend([types[j]] * int(k))
    return edges, blocks, 0


def verify(blocks, m, n, edges):
    """Independent legality check of the witness."""
    if len(blocks) != n:
        return False
    if sum(len(b) for b in blocks) != edges:
        return False
    capc = {}
    for b in blocks:
        for t in combinations(sorted(b), 3):
            capc[t] = capc.get(t, 0) + 1
            if capc[t] > 2:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+", type=int)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ev", os.path.join(HERE, "..", "..", "evaluator.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    Z = dict(ev.KST_EXACT_VALUE)

    wdir = os.path.join(HERE, "witnesses")
    os.makedirs(wdir, exist_ok=True)
    out_path = args.out or os.path.join(HERE, "ilp_results.jsonl")

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

    for m in args.rows:
        ns = sorted(n for (mm, n) in Z if mm == m and (mm, n) not in done)
        for n in ns:
            t0 = time.time()
            edges, blocks, status = solve_cell(m, n, args.time_limit)
            dt = time.time() - t0
            if edges is None:
                rec = {"m": m, "n": n, "status": f"UNRESOLVED({status})",
                       "seconds": round(dt, 1)}
                print(f"z({m},{n}) UNRESOLVED status={status} [{dt:.0f}s]",
                      flush=True)
            else:
                okw = verify(blocks, m, n, edges)
                match = edges == Z[(m, n)]
                rec = {"m": m, "n": n, "z_ilp": edges, "z_pub": Z[(m, n)],
                       "match": match, "witness_valid": okw,
                       "seconds": round(dt, 1)}
                mark = "OK" if (match and okw) else "!!"
                print(f"z({m},{n}) = {edges} pub={Z[(m,n)]} witness_valid={okw} "
                      f"[{dt:.0f}s] {mark}", flush=True)
                if okw:
                    with open(os.path.join(wdir, f"w_{m}x{n}.json"), "w") as f:
                        json.dump({"m": m, "n": n, "edges": edges,
                                   "blocks": [list(b) for b in blocks]}, f)
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

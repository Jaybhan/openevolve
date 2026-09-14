"""Deep-band task 1c: the value-Pareto frontier by direct MILP.

  F_m(c) := max val(C) = sum (|B|-3) over legal heavy configs C
            (multisets of blocks of weight >= 4, every triple covered <= 2)
            with #C <= c columns.

Then, at the optimum, min slots(C) subject to val = F_m(c)  (the point of the
3D Pareto surface (val, cols, slots) relevant to the exact identity
z = 3n + max_C [val - max(0, n - #C - B + slots)]).

Strengthening cuts, ALL PROVEN VALID (see report.md / theorems.md):
  - pair cuts: sum_{B >= {x,y}} (|B|-2) x_B <= 2(m-2)          [pair capacity]
  - point value floor: sum_{B>x} (|B|-3) x_B <= R = floor(2 C(m-1,2)/3)
                                                          [Lemma C, per point]
  - weighted budget: sum |B|(|B|-3) x_B <= m R              [sum of the above]
  - global value: val <= J = floor(mR/4) (Lemma C); for m=7: val <= 15
    (Theorem F, the J-2 law for m == 3 mod 4, m !== 0 mod 3)
  - monotone point-degrees (symmetry breaking, WLOG relabeling)
  - monotone value floor: val >= F_m(c-1) when known (F is nondecreasing)

Results appended to frontier_m{m}.jsonl (checkpoint-safe), one record per c:
  {"m", "c", "val", "status", "slots_min", "profile", "blocks", "seconds"}
status: "PROVEN" (both solves optimal) or "LB(timeout)" (incumbent only,
value is a lower bound) or "UNRESOLVED".
"""
import argparse
import json
import os
import time
from collections import Counter
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))


def build(m, wmax=None):
    wmax = wmax or m
    types = []
    for w in range(4, wmax + 1):
        types.extend(combinations(range(m), w))
    tidx = {t: i for i, t in enumerate(combinations(range(m), 3))}
    return types, tidx


def solve_frontier_point(m, c, time_limit=600.0, wmax=None, val_lb=None,
                         fixed_val=None, minimize_slots=False):
    """One MILP. If fixed_val is None: maximize val subject to <= c columns.
    Else: minimize slots subject to val == fixed_val, <= c columns."""
    types, tidx = build(m, wmax)
    ntri = len(tidx)
    nv = len(types)
    P = 2 * comb(m - 1, 2)          # per-point pair capacity
    R = P // 3                      # Lemma C per-point value floor
    J = (m * R) // 4                # Lemma C global
    if m % 4 == 3 and m % 3 != 0:
        J -= 2                      # Theorem F (proven for the class)
    pairs = list(combinations(range(m), 2))
    pidx = {p: i for i, p in enumerate(pairs)}
    npair = len(pairs)

    # rows: triples | columns<=c | pair cuts | point floors | weighted budget
    #       | global val | symmetry (m-1) | optional val floor/fix
    nrows = ntri + 1 + npair + m + 1 + 1 + (m - 1) + 1
    A = lil_matrix((nrows, nv))
    r_col = ntri
    r_pair0 = ntri + 1
    r_pt0 = r_pair0 + npair
    r_wb = r_pt0 + m
    r_gv = r_wb + 1
    r_sym0 = r_gv + 1
    r_valfix = r_sym0 + (m - 1)

    for j, b in enumerate(types):
        w = len(b)
        for t in combinations(b, 3):
            A[tidx[t], j] = 1
        A[r_col, j] = 1
        for p in combinations(b, 2):
            A[r_pair0 + pidx[p], j] = w - 2
        for x in b:
            A[r_pt0 + x, j] = w - 3
        A[r_wb, j] = w * (w - 3)
        A[r_gv, j] = w - 3
        for r in range(m - 1):
            v = (1 if r in b else 0) - (1 if (r + 1) in b else 0)
            if v:
                A[r_sym0 + r, j] = v
        A[r_valfix, j] = w - 3

    lb = np.zeros(nrows)
    ub = np.zeros(nrows)
    ub[:ntri] = 2.0
    lb[r_col], ub[r_col] = 0.0, float(c)
    for i in range(npair):
        lb[r_pair0 + i], ub[r_pair0 + i] = 0.0, float(2 * (m - 2))
    for x in range(m):
        lb[r_pt0 + x], ub[r_pt0 + x] = 0.0, float(R)
    lb[r_wb], ub[r_wb] = 0.0, float(m * R)
    lb[r_gv], ub[r_gv] = 0.0, float(J)
    for r in range(m - 1):
        lb[r_sym0 + r], ub[r_sym0 + r] = 0.0, np.inf
    if fixed_val is not None:
        lb[r_valfix] = ub[r_valfix] = float(fixed_val)
    elif val_lb is not None:
        lb[r_valfix], ub[r_valfix] = float(val_lb), np.inf
    else:
        lb[r_valfix], ub[r_valfix] = 0.0, np.inf

    if minimize_slots:
        cobj = np.array([float(comb(len(b), 3)) for b in types])
    else:
        cobj = np.array([-(len(b) - 3.0) for b in types])
    res = milp(c=cobj, constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(nv),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status == 2:
        return None, None, "INFEASIBLE"
    if res.x is None:
        return None, None, f"UNRESOLVED({res.status})"
    xv = np.round(res.x).astype(int)
    blocks = []
    for j, k in enumerate(xv):
        blocks.extend([types[j]] * int(k))
    # independent re-verification of legality
    cov = Counter()
    for b in blocks:
        for t in combinations(b, 3):
            cov[t] += 1
    assert all(v <= 2 for v in cov.values()), "MILP produced illegal config!"
    assert len(blocks) <= c
    obj = int(round(-res.fun)) if not minimize_slots else int(round(res.fun))
    status = "PROVEN" if res.status == 0 else f"LB(timeout,{res.status})"
    return obj, blocks, status


def profile_of(blocks):
    p = Counter(len(b) for b in blocks)
    return {str(w): p[w] for w in sorted(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("m", type=int)
    ap.add_argument("--cmin", type=int, default=1)
    ap.add_argument("--cmax", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=600.0)
    ap.add_argument("--wmax", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cells", default=None,
                    help="comma-separated list of c values (overrides range)")
    ap.add_argument("--val-lb", type=int, default=None,
                    help="known lower bound on val at the first cell")
    args = ap.parse_args()
    m = args.m
    cmax = args.cmax
    if cmax is None:
        cmax = {6: 10, 7: 16, 8: 29, 9: 41}.get(m, 3 * m)
    out = os.path.join(HERE, f"frontier_m{m}{args.tag}.jsonl")
    done = {}
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["c"]] = r
                except Exception:
                    pass
    prev_val = args.val_lb
    cell_list = ([int(x) for x in args.cells.split(",")] if args.cells
                 else list(range(args.cmin, cmax + 1)))
    for c in cell_list:
        if c in done:
            prev_val = done[c]["val"]
            continue
        if (c - 1) in done:
            prev_val = done[c - 1]["val"]
        t0 = time.time()
        val, blocks, status = solve_frontier_point(
            m, c, args.time_limit, wmax=args.wmax, val_lb=prev_val)
        if val is None:
            rec = {"m": m, "c": c, "val": None, "status": status,
                   "seconds": round(time.time() - t0, 1)}
        else:
            slots = sum(comb(len(b), 3) for b in blocks)
            sl_min, sl_blocks, sl_status = solve_frontier_point(
                m, c, args.time_limit, wmax=args.wmax,
                fixed_val=val, minimize_slots=True)
            if sl_min is not None and sl_min < slots:
                blocks, slots = sl_blocks, sl_min
            full_status = status if status == "PROVEN" else status
            if status == "PROVEN" and sl_status != "PROVEN":
                full_status = "PROVEN(val);slots_min=LB"
            rec = {"m": m, "c": c, "val": val, "status": full_status,
                   "slots_min": slots, "profile": profile_of(blocks),
                   "blocks": [list(b) for b in blocks],
                   "seconds": round(time.time() - t0, 1)}
            prev_val = val
        with open(out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"F_{m}({c}) = {rec.get('val')} {rec['status']} "
              f"slots_min={rec.get('slots_min')} prof={rec.get('profile')} "
              f"[{rec['seconds']}s]", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
exact_small.py -- brute-force EXACT computation of z(m,n;s,t) for small cells,
validating the bound machinery end to end and extending ground truth.

Method: descending-target exhaustive DFS over COLUMN MULTISETS.
  * Candidate columns = all 2^m row-subsets (bitmasks), totally ordered by
    (weight descending, mask ascending); a solution is explored as a
    NONINCREASING sequence in that order, i.e. one canonical representative
    per column multiset (kills the n! column symmetry).
  * Constraint maintained exactly: coverage counter over all C(m,s) s-subsets
    of rows, each capped at t-1 (definition of K_{s,t}-free). The global
    budget sum_j C(c_j,s) <= (t-1)C(m,s) is implied by the caps and is used
    for pruning.
  * PROVEN prune: with k columns left, all of weight <= w, and remaining
    budget b, no completion can add more than max_sum_under_budget(k, w, s, b)
    ones (upper_bounds.py primitive, itself exhaustively verified). Since
    candidate weights are nonincreasing along the candidate list, the first
    candidate index where the bound falls below target cuts the whole rest.
  * exact_z runs targets ub_waterfill, ub-1, ... ; the first target for which
    a witness exists is z. A cell is PROVEN only if every refuted target's
    search ran to completion (no node-cap abort).

Every witness found is INDEPENDENTLY re-verified with the ground-truth
checker has_kst() imported from ../../evaluator.py (side-effect-free import),
and (3,3) results are asserted equal to KST_EXACT_VALUE wherever they overlap.

Usage:
    python3 exact_small.py            # base suite -> exact_small.csv (+checks)
    python3 exact_small.py --extended # + (3,3) cells proven cheap on this box:
                                      #   (6,7..14),(7,7..14),(8,8..10),(9,9),(9,10)
    python3 exact_small.py --cell M N S T   # one cell, verbose

Base suite: (s,t) in {(2,2),(2,3)} for 1<=m<=n<=7, (3,3) for 1<=m<=n<=6,
and (s,t) in {(3,4),(4,4)} for 1<=m<=n<=6. Witnesses of interesting cells go
to exact_small_witnesses.json.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from itertools import combinations
from math import comb

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

from upper_bounds import lb_culik, max_sum_under_budget, ub_waterfill  # noqa: E402


def _load_evaluator():
    path = os.path.join(_BASE, "evaluator.py")
    spec = importlib.util.spec_from_file_location("zar_evaluator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Abort(Exception):
    pass


def exact_z(m, n, s, t, max_nodes=None, verbose=False):
    """
    Exact z(m,n;s,t) by complete search (see module docstring).

    Returns dict with keys:
      z          -- the exact value (None if search aborted incomplete)
      complete   -- True iff every step ran to completion => value is PROVEN
      witness    -- list of n column bitmasks attaining z (padded with 0s)
      nodes, seconds, ub (=ub_waterfill), refuted (list of refuted targets)
    """
    t0 = time.process_time()
    ub = ub_waterfill(m, n, s, t)

    # Degenerate: no K_{s,t} fits => all-ones matrix optimal. PROVEN.
    if m < s or n < t:
        full = (1 << m) - 1
        return dict(z=m * n, complete=True, witness=[full] * n, nodes=0,
                    seconds=time.process_time() - t0, ub=ub, refuted=[])

    subsets = list(combinations(range(m), s))
    nsub = len(subsets)
    B = (t - 1) * nsub  # == (t-1)*C(m,s)
    tcap = t - 1

    masks = sorted(range(1, 1 << m), key=lambda x: (-bin(x).count("1"), x))
    w = [bin(x).count("1") for x in masks]
    subs = [tuple(i for i, S in enumerate(subsets)
                  if all(x >> r & 1 for r in S)) for x in masks]
    cost = [comb(wj, s) for wj in w]

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def wfb(k, cap, slack):
        return max_sum_under_budget(k, cap, s, slack)

    nodes = 0
    chosen = []
    refuted = []

    def dfs(start, k_left, ones, used, target, cov):
        nonlocal nodes
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise _Abort
        if ones >= target:
            return True
        if k_left == 0:
            return False
        b = B - used
        for j in range(start, len(masks)):
            if ones + wfb(k_left, w[j], b) < target:
                return False        # weights nonincreasing: nothing later helps
            sj = subs[j]
            ok = True
            for i in sj:
                if cov[i] >= tcap:
                    ok = False
                    break
            if not ok:
                continue
            for i in sj:
                cov[i] += 1
            chosen.append(masks[j])
            if dfs(j, k_left - 1, ones + w[j], used + cost[j], target, cov):
                return True
            chosen.pop()
            for i in sj:
                cov[i] -= 1
        return False

    witness = None
    z = None
    try:
        for target in range(ub, -1, -1):
            cov = [0] * nsub
            chosen.clear()
            if dfs(0, n, 0, 0, target, cov):
                z = target
                witness = list(chosen) + [0] * (n - len(chosen))
                break
            refuted.append(target)
            if verbose:
                print(f"    target {target} refuted ({nodes} nodes so far)")
        complete = True
    except _Abort:
        complete = False

    return dict(z=z, complete=complete, witness=witness, nodes=nodes,
                seconds=time.process_time() - t0, ub=ub, refuted=refuted)


def _witness_matrix(masks, m):
    import numpy as np
    A = np.zeros((m, len(masks)), dtype=np.int8)
    for j, x in enumerate(masks):
        for r in range(m):
            if x >> r & 1:
                A[r, j] = 1
    return A


def run_suite(extended=False, max_nodes_ext=2_000_000_000):
    ev = _load_evaluator()
    known33 = dict(ev.KST_EXACT_VALUE)

    cells = []
    for (s, t), lim in (((2, 2), 7), ((2, 3), 7), ((3, 3), 6),
                        ((3, 4), 6), ((4, 4), 6)):
        for m in range(1, lim + 1):
            for n in range(m, lim + 1):
                cells.append((m, n, s, t, None, "base"))
    if extended:
        # every (3,3) cell verified feasible for complete exhaustion on this
        # machine (worst observed: (9,10), ~5.4e7 nodes / ~250 s)
        for mn in ((6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12),
                   (6, 13), (6, 14), (7, 7), (7, 8), (7, 9), (7, 10),
                   (7, 11), (7, 12), (7, 13), (7, 14), (8, 8), (8, 9),
                   (8, 10), (9, 9), (9, 10)):
            cells.append((*mn, 3, 3, max_nodes_ext, "extended"))

    rows = []
    witnesses = {}
    mismatches = []
    nontight = []
    for m, n, s, t, cap, kind in cells:
        r = exact_z(m, n, s, t, max_nodes=cap)
        status = ("PROVEN" if r["complete"] else "INCOMPLETE")
        z = r["z"]
        tag = f"z({m},{n};{s},{t})"
        print(f"{tag:>16} = {z if r['complete'] else '?'}   ub_wf={r['ub']} "
              f"nodes={r['nodes']:>10} {r['seconds']:8.2f}s  {status}")
        if r["complete"]:
            # independent verification of the witness with evaluator's checker
            A = _witness_matrix(r["witness"], m)
            assert int(A.sum()) == z, f"{tag}: witness sum != z"
            assert not ev.has_kst(A, s, t), f"{tag}: witness contains K_{s},{t}!"
            # cross-check against published exact (3,3) values
            if (s, t) == (3, 3) and (m, n) in known33:
                if z != known33[(m, n)]:
                    mismatches.append((m, n, z, known33[(m, n)]))
                    print(f"   *** MISMATCH vs published: got {z}, "
                          f"table says {known33[(m, n)]}")
            if z < r["ub"]:
                nontight.append((m, n, s, t, z, r["ub"]))
            if z < r["ub"] or kind == "extended":
                witnesses[f"{m},{n},{s},{t}"] = r["witness"]
        lc = lb_culik(m, n, s, t)
        rows.append({
            "s": s, "t": t, "m": m, "n": n,
            "z": "" if z is None else z,
            "status": status if r["complete"] else
                      f"INCOMPLETE(refuted>{max(r['refuted'], default='-')})",
            "ub_waterfill": r["ub"],
            "counting_tight": "" if z is None else int(z == r["ub"]),
            "lb_culik": "" if lc is None else lc,
            "nodes": r["nodes"], "seconds": round(r["seconds"], 3),
            "kind": kind,
        })

    out = os.path.join(_HERE, "exact_small.csv")
    with open(out, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} cells)")

    wout = os.path.join(_HERE, "exact_small_witnesses.json")
    with open(wout, "w") as f:
        json.dump(witnesses, f, indent=1)
    print(f"wrote {wout} ({len(witnesses)} witnesses)")

    # transpose identity self-checks z(m,n;s,t) == z(n,m;t,s)
    print("\ntranspose identity checks:")
    for (m, n, s, t) in ((3, 4, 2, 3), (4, 5, 2, 3), (3, 5, 3, 4),
                         (4, 5, 3, 3), (4, 6, 2, 2)):
        a = exact_z(m, n, s, t)["z"]
        b = exact_z(n, m, t, s)["z"]
        flag = "OK" if a == b else "*** FAIL"
        print(f"  z({m},{n};{s},{t})={a}  z({n},{m};{t},{s})={b}   {flag}")
        assert a == b

    print(f"\n(3,3) overlap mismatches vs published table: {len(mismatches)}")
    print(f"cells where ub_waterfill is NOT tight (z < ub): {len(nontight)}")
    for c in nontight:
        print(f"   z({c[0]},{c[1]};{c[2]},{c[3]}) = {c[4]} < ub {c[5]}"
              f"   (deficit {c[5] - c[4]})")
    return rows, nontight, mismatches


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--cell", nargs=4, type=int, metavar=("M", "N", "S", "T"))
    args = ap.parse_args()
    if args.cell:
        m, n, s, t = args.cell
        r = exact_z(m, n, s, t, verbose=True)
        print(r)
    else:
        run_suite(extended=args.extended)

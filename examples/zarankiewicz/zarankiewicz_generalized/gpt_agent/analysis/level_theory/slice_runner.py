"""Compute mixed-ledger slices S_m(sig) = max #quads coexisting with the
exact heavy counts sig = (k5..k13), triple coverage <= 2, block mult <= 2.
Feeds slices.csv for fullpass.py. Monotone: S(sig) caps every config whose
signature dominates sig.

Given the z-table is correct, every claiming slice must come back with
S < k4_needed — each computed slice is therefore an independent
design-theoretic VERIFICATION of the corresponding z cells (the z <->
packing dictionary run in full).

Usage: slice_runner.py [max_m] [time_limit_s]  (defaults 12, 600)
Reads slice_queue.csv, pareto-reduces per m, runs ascending m.
"""
import os
import sys
import time
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
WS = list(range(5, 14))


def slice_milp(m, sig, time_limit=600):
    """max k4 with exactly sig_w blocks of weight w (w=5..13), coverage<=2."""
    weights = [4] + [w for w, k in zip(WS, sig) if k > 0]
    types = []
    wof = []
    for w in weights:
        for b in combinations(range(m), w):
            types.append(b)
            wof.append(w)
    tris = {t: i for i, t in enumerate(combinations(range(m), 3))}
    nT, nV = len(tris), len(types)
    ncnt = len(weights) - 1
    A = lil_matrix((nT + ncnt, nV))
    lb = np.zeros(nT + ncnt)
    ub = np.full(nT + ncnt, 2.0)
    c = np.zeros(nV)
    for j, b in enumerate(types):
        for t in combinations(b, 3):
            A[tris[t], j] = 1
        if wof[j] == 4:
            c[j] = -1.0
    for i, w in enumerate(weights[1:]):
        k = sig[WS.index(w)]
        for j in range(nV):
            if wof[j] == w:
                A[nT + i, j] = 1
        lb[nT + i] = ub[nT + i] = k
    res = milp(c=c, constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(nV),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.status == 0:
        return int(round(-res.fun))
    if res.status == 2:
        return -1  # heavy config itself infeasible
    return None


def main():
    max_m = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    tl = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    band = os.environ.get("SLICE_BAND") == "1"  # near-band sigs only:
    # corner claims (many hexads/heptads at m>=11) are compute-infeasible;
    # the band family (<=3 hexads, <=1 of weights>=7, <=6 heavy-heavy) is
    # where the ledger formula's remaining band residuals live.
    queue = {}
    for line in open(os.path.join(HERE, "slice_queue.csv")).readlines()[1:]:
        m, sig, k = line.strip().split(",")
        m = int(m)
        if m > max_m:
            continue
        sig = tuple(int(t) for t in sig.split(";"))
        if band and m >= 11 and (sig[1] > 3 or sum(sig[2:]) > 1
                                 or sum(sig[1:]) > 4):
            continue  # m<=10 slice MILPs are small: no filtering needed
        queue.setdefault(m, {})[sig] = max(queue.get(m, {}).get(sig, 0), int(k))
    done = set()
    sp = os.path.join(HERE, "slices.csv")
    if os.path.exists(sp):
        for line in open(sp).readlines()[1:]:
            p = line.strip().split(",")
            if p[2] != "TIMEOUT":  # TIMEOUT rows are retried
                done.add((int(p[0]), tuple(int(t) for t in p[1].split(";"))))
    else:
        open(sp, "w").write("m,sig,S,k4_needed,verdict\n")
    caps = {}
    if os.path.exists(sp):
        for line in open(sp).readlines()[1:]:
            p = line.strip().split(",")
            if p[2] not in ("TIMEOUT",):
                caps.setdefault(int(p[0]), []).append(
                    (tuple(int(t) for t in p[1].split(";")), int(p[2])))
    for m in sorted(queue):
        sigs = queue[m]
        # drop sigs already refuted by a computed dominated slice's cap
        live = {}
        for s, k in sigs.items():
            capped = any(all(a >= b for a, b in zip(s, s0)) and S0 < k
                         for (s0, S0) in caps.get(m, []))
            if not capped:
                live[s] = k
        sigs = live
        # pareto-minimal reduction (dominance): most general slices first
        minimal = [s for s in sigs
                   if not any(all(a >= b for a, b in zip(s, o)) and s != o
                              for o in sigs)]
        print(f"m={m}: {len(sigs)} live claiming sigs, {len(minimal)} minimal",
              flush=True)
        for s in sorted(minimal):
            if (m, s) in done:
                continue
            t0 = time.time()
            if m >= 9 and sum(s) >= 1:
                from pinned_slice import pinned_slice
                S, _ = pinned_slice(m, s, tl)
            else:
                S = slice_milp(m, s, tl)
            need = sigs[s]
            verdict = ("TIMEOUT" if S is None else
                       "REFUTED" if S < need else "!!CLAIM-STANDS!!")
            print(f"  S_{m}{s} = {S} (need >= {need}): {verdict} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            open(sp, "a").write(
                f"{m},{';'.join(map(str, s))},"
                f"{S if S is not None else 'TIMEOUT'},{need},{verdict}\n")


if __name__ == "__main__":
    main()

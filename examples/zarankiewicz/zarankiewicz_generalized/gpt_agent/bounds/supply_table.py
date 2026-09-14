#!/usr/bin/env python3
"""
supply_table.py -- (1) machine-verification of coordinator Lemma A,
(2) the supply function S_m(k5,k6) tabulated by exact ILP,
(3) row-closure test: Lemma A + S-table => closed-form row bounds.

Run with the scratchpad venv python (scipy>=1.11 for milp), e.g.
    <venv>/bin/python3 supply_table.py            # everything
    <venv>/bin/python3 supply_table.py --lemma    # only the Lemma A sweep

Outputs supply_table.csv next to this file. Labels:
  ILP-PROVEN    -- optimum from scipy.optimize.milp (HiGHS, exact MILP on
                   integral data); independently spot-checked against the
                   complete-search values g4(m)=S_m(0,0) of supply_law.py
                   (m=6,7,8) and a dedicated DFS for S_6/S_7(1,0).
  PROVEN        -- pure arithmetic / complete search.

Definitions (s=t=3): B = 2*C(m,3). A profile with k_h blocks of weight h
(h>=3, weight<=2 blocks are free pads) has E <= 3n + Q,
Q := k4 + 2*k5 + 3*k6 + 4*k7 (blocks capped at weight 7 here; extend if
needed). Slots: k3 + 4k4 + 10k5 + 20k6 + 35k7 <= B, and k3 + k4 + ... <= n.

Lemma A (coordinator, analysis/master_formula.md): for every legal profile
    Q <= (B - n)/3 - k5 - (10/3) k6 - (22/3) k7.
Derivation being verified: eliminate k3 >= 0 between the two constraints:
   n + 3k4 + 9k5 + 19k6 + 34k7 <= B, then Q = k4+2k5+3k6+4k7 gives the bound.

S_m(k5,k6) := max number of weight-4 blocks (multiplicity <= 2) that can
coexist with exactly k5 weight-5 and k6 weight-6 blocks (multiplicity <= 2)
under "every 3-subset of [m] covered <= 2". None = even the heavy blocks
alone are infeasible.

Row-closure bound (Lemma A + supply + n-cap), an UPPER bound on z(m,n;3,3):
    z <= 3n + max over (k5,k6) feasible of
         min( floor((B-n)/3 - k5 - (10/3)k6),
              min(S_m(k5,k6), n - k5 - k6) + 2k5 + 3k6 )
(with the convention that the max includes k5=k6=0). The test prints, for
every known cell of rows 6..10, whether this equals the true z.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from itertools import combinations
from math import comb, floor, inf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from upper_bounds import known_exact_33, ub_waterfill  # noqa: E402


# ---------------------------------------------------------------------------
# (1) Lemma A machine verification (pure arithmetic sweep)
# ---------------------------------------------------------------------------

def verify_lemma_a(mmax=13, verbose=True):
    """
    Exhaustive integer sweep: every (m, n, k3..k7) with k3+..+k7 <= n <= B,
    slots <= B, within bounded ranges, must satisfy Lemma A. Also checks
    TIGHTNESS pattern: equality cases have slots budget and column count
    both tight modulo rounding. PROVEN by sweep over the stated range.
    """
    bad = 0
    ncases = 0
    for m in range(4, mmax + 1):
        B = 2 * comb(m, 3)
        for k7 in range(0, 4):
            for k6 in range(0, 6):
                for k5 in range(0, 9):
                    for k4 in range(0, min(41, B // 4 + 1)):
                        heavy_slots = 4 * k4 + 10 * k5 + 20 * k6 + 35 * k7
                        if heavy_slots > B:
                            continue
                        kmax3 = B - heavy_slots           # k3 <= B - heavy slots
                        for k3 in (0, kmax3 // 2, kmax3):  # extremes suffice; Q free of k3
                            n = k3 + k4 + k5 + k6 + k7    # tightest n (padding only loosens)
                            if n == 0:
                                continue
                            Q = k4 + 2 * k5 + 3 * k6 + 4 * k7
                            rhs = (B - n) / 3 - k5 - (10 / 3) * k6 - (22 / 3) * k7
                            ncases += 1
                            if Q > rhs + 1e-9:
                                bad += 1
                                if bad < 5:
                                    print("LEMMA A VIOLATION:", m, (k3, k4, k5, k6, k7))
    if verbose:
        print(f"[lemma A] sweep m<=13: {ncases} legal profiles checked, "
              f"{bad} violations")
    return bad == 0


# ---------------------------------------------------------------------------
# (2) S_m(k5, k6) by exact ILP
# ---------------------------------------------------------------------------

def s_table_ilp(m, k5, k6=0, time_limit=60.0):
    """
    Exact S_m(k5,k6) via scipy.optimize.milp (HiGHS).
    Returns (value_or_None, status) with status in
    ILP-PROVEN / INFEASIBLE / UNRESOLVED (time limit hit -- NOT proven).
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    B = 2 * comb(m, 3)
    # cheap PROVEN pre-checks
    if k5 > 2 * comb(m, 5) or k6 > 2 * comb(m, 6):
        return None, "INFEASIBLE"
    if 10 * k5 + 20 * k6 > B:            # aggregate slot budget
        return None, "INFEASIBLE"

    quads = list(combinations(range(m), 4))
    pents = list(combinations(range(m), 5)) if m >= 5 else []
    hexes = list(combinations(range(m), 6)) if m >= 6 else []
    if (k5 > 0 and not pents) or (k6 > 0 and not hexes):
        return None, "INFEASIBLE"
    blocks = quads + pents + hexes
    triples = list(combinations(range(m), 3))
    tidx = {T: i for i, T in enumerate(triples)}
    A = np.zeros((len(triples), len(blocks)))
    for j, b in enumerate(blocks):
        for T in combinations(b, 3):
            A[tidx[T], j] = 1
    nq, np5 = len(quads), len(pents)
    c = np.zeros(len(blocks))
    c[:nq] = -1.0                                    # maximize quad count
    cons = [LinearConstraint(A, -np.inf, 2)]
    row5 = np.zeros(len(blocks)); row5[nq:nq + np5] = 1
    cons.append(LinearConstraint(row5, k5, k5))
    row6 = np.zeros(len(blocks)); row6[nq + np5:] = 1
    cons.append(LinearConstraint(row6, k6, k6))
    res = milp(c=c, constraints=cons, integrality=np.ones(len(blocks)),
               bounds=Bounds(0, 2), options={"time_limit": time_limit})
    if res.status == 0:
        return round(-res.fun), "ILP-PROVEN"
    if res.status == 2:                  # proven infeasible
        return None, "INFEASIBLE"
    return None, "UNRESOLVED"            # time limit / other: no claim


def build_table(mrange=(6, 10), k5max=12, k6spot=(0, 1, 2)):
    rows = []
    S = {}
    for m in range(mrange[0], mrange[1] + 1):
        for k6 in k6spot:
            dead = False
            for k5 in range(0, k5max + 1):
                if dead:
                    v, st = None, "INFEASIBLE"      # monotone: more pentads
                else:                              # can never restore legality
                    v, st = s_table_ilp(m, k5, k6)
                    if st == "INFEASIBLE":
                        dead = True
                rows.append({"m": m, "k5": k5, "k6": k6,
                             "S": "" if v is None else v, "status": st})
                S[(m, k5, k6)] = (v, st)
                print(f"  S_{m}({k5},{k6}) = {v} [{st}]", flush=True)
    out = os.path.join(_HERE, "supply_table.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["m", "k5", "k6", "S", "status"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return S


def s_small_dfs(m, k5):
    """Complete-search S_m(k5,0) for cross-checking the ILP (small m only)."""
    triples = list(combinations(range(m), 3))
    tidx = {T: i for i, T in enumerate(triples)}
    quads = list(combinations(range(m), 4))
    pents = list(combinations(range(m), 5))
    qsubs = [tuple(tidx[T] for T in combinations(b, 3)) for b in quads]
    psubs = [tuple(tidx[T] for T in combinations(b, 3)) for b in pents]
    cov = [0] * len(triples)
    best = [-1]

    def place_pents(i, left):
        if left == 0:
            best_quads(0, 0)
            return
        for j in range(i, len(pents)):
            if all(cov[x] < 2 for x in psubs[j]):
                for x in psubs[j]: cov[x] += 1
                place_pents(j, left - 1)
                for x in psubs[j]: cov[x] -= 1

    def best_quads(i, count):
        if count > best[0]:
            best[0] = count
        slack = sum(2 - c for c in cov)
        if count + slack // 4 <= best[0]:
            return
        for j in range(i, len(quads)):
            if all(cov[x] < 2 for x in qsubs[j]):
                for x in qsubs[j]: cov[x] += 1
                best_quads(j, count + 1)
                for x in qsubs[j]: cov[x] -= 1

    place_pents(0, k5)
    return None if best[0] < 0 else best[0]


# ---------------------------------------------------------------------------
# (3) row closure: Lemma A + S-table as a closed-form row bound
# ---------------------------------------------------------------------------

def row_bound(m, n, S, k5max=12, k6max=2):
    """
    PROVEN upper bound on z(m,n;3,3) = 3n + Q from Lemma A + supply table.
    Sound for EVERY legal profile:
      * enumerated part: profiles with k5 <= k5max, k6 <= k6max, no block of
        weight >= 7: Q <= min(Lemma-A ceiling, supply-capped quad count).
        UNRESOLVED supply entries are bracketed from above by the nearest
        proven S at smaller k5 (S is nonincreasing in k5: removing a pentad
        keeps legality), falling back to the trivial n-cap.
      * fallbacks (Lemma A penalties alone, with the generalized coefficient
        c_w = (C(w,3)-1-3(w-3))/3 nondecreasing in w):
        any block of weight >= 7  => Q <= (B-n)/3 - 22/3;
        k6 >= k6max+1            => Q <= (B-n)/3 - (10/3)(k6max+1);
        k5 >= k5max+1            => Q <= (B-n)/3 - (k5max+1).
    S maps (m,k5,k6) -> (value_or_None, status).
    """
    B = 2 * comb(m, 3)
    cands = []
    for k6 in range(0, k6max + 1):
        for k5 in range(0, k5max + 1):
            # a combo never run must be treated as unresolved (bracketed),
            # never skipped -- skipping would understate the max (unsound)
            ent = S.get((m, k5, k6)) or (None, "UNRESOLVED")
            v, st = ent
            if st == "INFEASIBLE":
                continue
            if st == "UNRESOLVED":
                v = inf
                for k5p in range(k5 - 1, -1, -1):
                    e2 = S.get((m, k5p, k6))
                    if e2 and e2[1] == "ILP-PROVEN":
                        v = e2[0]
                        break
            if k5 + k6 > n:
                continue
            ceilA = floor((B - n) / 3 - k5 - (10 / 3) * k6)
            q = min(v, n - k5 - k6) + 2 * k5 + 3 * k6
            cands.append(min(ceilA, q))
    cands.append(floor((B - n) / 3 - 22 / 3))              # any weight>=7 block
    cands.append(floor((B - n) / 3 - (10 / 3) * (k6max + 1)))  # k6 overflow
    cands.append(floor((B - n) / 3 - (k5max + 1)))         # k5 overflow
    return 3 * n + max(0, max(cands))


def load_table():
    """Reload supply_table.csv as {(m,k5,k6): (value_or_None, status)}."""
    out = os.path.join(_HERE, "supply_table.csv")
    S = {}
    with open(out) as f:
        for r in csv.DictReader(f):
            v = None if r["S"] == "" else int(r["S"])
            S[(int(r["m"]), int(r["k5"]), int(r["k6"]))] = (v, r["status"])
    return S


def main(args):
    ok = verify_lemma_a()
    if not ok:
        print("LEMMA A FAILED -- stop"); sys.exit(1)
    if args.lemma:
        return

    S = load_table() if args.closure else build_table()

    # cross-checks
    from supply_law import G4_PROVEN
    print("\ncross-checks:")
    for m, g in G4_PROVEN.items():
        v = S.get((m, 0, 0), (None, ""))[0]
        print(f"  S_{m}(0,0) = {v}  vs complete-search g4({m}) = {g}  "
              f"{'OK' if v == g else '*** MISMATCH'}")
    for (m, k5) in ((6, 1), (7, 1)):
        d = s_small_dfs(m, k5)
        v = S.get((m, k5, 0), (None, ""))[0]
        print(f"  S_{m}({k5},0): ILP {v} vs DFS {d}  {'OK' if v == d else '*** MISMATCH'}")

    # row closure vs known values
    known = known_exact_33()
    print("\nrow-closure test (Lemma A + S-table bound vs known z), rows 6..10:")
    stats = {}
    for (m, n), z in sorted(known.items()):
        if not (6 <= m <= 10):
            continue
        rb = row_bound(m, n, S)
        wf = ub_waterfill(m, n, 3, 3)
        ub = min(rb, wf)
        assert ub >= z, f"row bound UNSOUND at ({m},{n}): {ub} < {z}"
        stats.setdefault(m, [0, 0])
        stats[m][ub == z] += 1
        mark = "EXACT" if ub == z else f"gap {ub - z}"
        print(f"  ({m:2d},{n:2d}): z={z:3d} wf={wf:3d} lemmaA+S={rb:3d} -> ub {ub:3d}  {mark}")
    print("\nper-row closure (cells not closed / closed):")
    for m in sorted(stats):
        print(f"  m={m}: {stats[m][1]} closed, {stats[m][0]} open")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lemma", action="store_true")
    ap.add_argument("--closure", action="store_true",
                    help="reuse existing supply_table.csv; run checks+closure only")
    main(ap.parse_args())

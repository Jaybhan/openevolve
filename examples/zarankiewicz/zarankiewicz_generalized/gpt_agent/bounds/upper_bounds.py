#!/usr/bin/env python3
"""
upper_bounds.py -- PROVEN upper-bound machinery for the Zarankiewicz problem
z(m,n;s,t), plus the Culik exact-regime lower bound and exactness
certification helpers.

Convention (matches ../../evaluator.py and gpt_agent/README.md):
    z(m,n;s,t) = maximum number of 1s in an m x n 0/1 matrix containing no
    s rows that share t common 1-columns (i.e. no all-ones s x t submatrix,
    with s indexing ROWS and t indexing COLUMNS).
    Symmetry: z(m,n;s,t) = z(n,m;t,s).

THE CORE COUNTING BOUND (both orientations)
-------------------------------------------
Column side: let c_1..c_n be the column sums. Column j contains C(c_j, s)
s-subsets of rows. Every s-subset of rows is contained in at most t-1 column
supports (else those s rows share t common columns). Hence

    sum_j C(c_j, s) <= (t-1) * C(m, s).            [column budget -- PROVEN]

So z <= max{ sum_j c_j : 0 <= c_j <= m integer, budget holds }, and that
integer maximum is computed EXACTLY by marginal-cost waterfilling
(`max_sum_under_budget`, proof in its docstring, exhaustively unit-tested by
--selftest). Row side symmetrically: row sums r_1..r_m satisfy

    sum_i C(r_i, t) <= (s-1) * C(n, t).            [row budget -- PROVEN]

ub_waterfill(m,n,s,t) = min(column side, row side).  PROVEN upper bound.

Labels used throughout (see gpt_agent/README.md honesty bar):
  PROVEN          -- complete argument in the docstring (plus unit tests).
  REFERENCE       -- valid but dominated bound, kept for comparison (ub_kst).

Usage:
    python3 upper_bounds.py --selftest   # exhaustive unit tests; exit 1 on failure
    python3 upper_bounds.py --tables     # (re)generate ub_33.csv + print stats
    python3 upper_bounds.py --refine     # profile refinement scan of d>0 cells (slow-ish)

Regenerable artifacts: ub_33.csv (written next to this file by --tables).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from itertools import combinations_with_replacement
from math import comb, floor

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.abspath(os.path.join(_HERE, "..", ".."))  # zarankiewicz_generalized/


# ---------------------------------------------------------------------------
# The waterfilling primitive
# ---------------------------------------------------------------------------

def max_sum_under_budget(k: int, cap: int, r: int, budget: int) -> int:
    """
    EXACT integer maximum of sum_{j=1..k} c_j subject to

        0 <= c_j <= cap  (integers),   sum_{j=1..k} C(c_j, r) <= budget,

    for r >= 1, computed by marginal-cost waterfilling.

    PROVEN optimal. Argument: raising one coordinate from c to c+1 raises the
    budget usage by C(c+1,r) - C(c,r) = C(c,r-1) (Pascal), which is
    nondecreasing in c. A feasible point is a set of unit "increments", each
    coordinate taking a prefix of its increment ladder (heights). The cheapest
    possible total cost of ANY N increments is the sum of the N globally
    cheapest increment costs, and because per-coordinate costs are
    nondecreasing, the N globally cheapest increments can always be arranged
    as valid prefixes (fill level by level). Hence greedy level-filling buys
    the maximum possible number of increments within the budget: if it stops
    at N, any N+1 valid increments cost strictly more than the budget, so no
    feasible point has sum > N. Exhaustively verified against brute-force
    enumeration of all degree profiles by --selftest.
    """
    if r < 1:
        raise ValueError("r >= 1 required")
    if k <= 0 or cap <= 0:
        return 0
    if budget < 0:
        budget = 0
    total = 0
    b = budget
    for h in range(cap):            # raise columns from height h to h+1
        cost = comb(h, r - 1)       # marginal cost per column
        if cost == 0:               # h < r-1: free levels
            total += k
            continue
        if cost * k <= b:
            b -= cost * k
            total += k
        else:
            q = b // cost
            total += q
            break
    return total


def min_cost_of_sum(p: int, cap: int, r: int, ssum: int):
    """
    EXACT integer minimum of sum_{j=1..p} C(c_j, r) subject to
    0 <= c_j <= cap and sum_j c_j = ssum. None if ssum infeasible.

    PROVEN: C(c, r) is convex in c, so by majorization the balanced profile
    (all parts equal to floor or ceil of ssum/p) minimizes the sum.
    Verified against brute force by --selftest.
    """
    if ssum < 0 or ssum > p * cap:
        return None
    if p == 0:
        return 0
    q, rem = divmod(ssum, p)
    return (p - rem) * comb(q, r) + rem * comb(q + 1, r)


# ---------------------------------------------------------------------------
# Upper bounds
# ---------------------------------------------------------------------------

def ub_waterfill_sides(m: int, n: int, s: int, t: int):
    """(column-side, row-side) waterfill bounds. Both PROVEN upper bounds."""
    col = max_sum_under_budget(n, m, s, (t - 1) * comb(m, s))
    row = max_sum_under_budget(m, n, t, (s - 1) * comb(n, t))
    return col, row


def ub_waterfill(m: int, n: int, s: int, t: int) -> int:
    """
    PROVEN upper bound on z(m,n;s,t): min of the exact integer optima of the
    column-budget and row-budget relaxations (see module docstring).
    Handles degenerate cells correctly (m<s or n<t gives m*n).
    """
    if min(m, n) <= 0:
        return 0
    if s < 1 or t < 1:
        raise ValueError("s,t >= 1 required")
    col, row = ub_waterfill_sides(m, n, s, t)
    return min(col, row)


def ub_kst(m: int, n: int, s: int, t: int):
    """
    REFERENCE: classical Kovari--Sos--Turan closed-form bound,

      z <= min( (s-1)*n + (t-1)^(1/s) * m * n^(1-1/s),
                (t-1)*m + (s-1)^(1/t) * n * m^(1-1/t) ),  floored.

    PROVEN (with this exact constant): let g(x)=0 for x<=s-1 and
    g(x)=prod_{i<s}(x-i)/s! for x>=s-1; g is convex, g(c)<=C(c,s) for all
    integers c>=0. Jensen on the column budget gives n*g(z/n) <= (t-1)C(m,s)
    <= (t-1) m^s / s!. If z/n > s-1, g(z/n) >= (z/n-s+1)^s/s!, and solving
    yields the first expression; if z/n <= s-1 the expression bounds z
    trivially. Second expression is the transpose. Dominated by ub_waterfill
    (which optimizes the same constraint exactly); kept for comparison.
    Requires t>=2 or s>=2 sensibly; degenerate s=1/t=1 still valid.
    """
    if min(m, n) <= 0:
        return 0
    a = (s - 1) * n + ((t - 1) ** (1.0 / s)) * m * (n ** (1.0 - 1.0 / s))
    b = (t - 1) * m + ((s - 1) ** (1.0 / t)) * n * (m ** (1.0 - 1.0 / t))
    val = floor(min(a, b) + 1e-9)   # z is an integer; +1e-9 guards float error
    return min(val, m * n)


# ---------------------------------------------------------------------------
# Culik exact-regime lower bound
# ---------------------------------------------------------------------------

def lb_culik(m: int, n: int, s: int, t: int):
    """
    Culik's exact-regime LOWER bound (constructive), or None outside its
    regime. Column orientation: if s <= m and n >= (t-1)*C(m,s), then

        z(m,n;s,t) >= (s-1)*n + (t-1)*C(m,s),

    PROVEN by explicit construction: for each of the C(m,s) s-subsets of
    rows take t-1 columns whose support is exactly that subset
    ((t-1)*C(m,s) <= n columns of weight s), and fill the remaining columns
    with arbitrary supports of size s-1. Every s-subset of rows is contained
    in exactly t-1 column supports (weight-(s-1) columns contain none), so
    the matrix is K_{s,t}-free, with (s-1)*n + (t-1)*C(m,s) ones.

    Transpose orientation symmetrically. Returns the max of the applicable
    orientations, else None.

    In this regime the column-side waterfill equals the same value
    (free levels give (s-1)n, then n >= budget columns absorb the whole
    budget at marginal cost 1), so lb_culik == ub_waterfill certifies the
    cell EXACT by pure counting. Culik's theorem says exactly this.
    """
    best = None
    if 1 <= s <= m and n >= (t - 1) * comb(m, s):
        v = (s - 1) * n + (t - 1) * comb(m, s)
        best = v if best is None else max(best, v)
    if 1 <= t <= n and m >= (s - 1) * comb(n, t):
        v = (t - 1) * m + (s - 1) * comb(n, t)
        best = v if best is None else max(best, v)
    return best


# ---------------------------------------------------------------------------
# Best bound + certification API
# ---------------------------------------------------------------------------

# Registry of implemented upper bounds (name -> callable). Extend here.
UB_REGISTRY = [
    ("waterfill", ub_waterfill),
    ("kst", ub_kst),
]


def best_ub(m: int, n: int, s: int, t: int) -> int:
    """Minimum over all registered PROVEN upper bounds (plus trivial m*n)."""
    return min(min(fn(m, n, s, t) for _, fn in UB_REGISTRY), m * n)


def certify_cell(m: int, n: int, s: int, t: int, lb=None):
    """
    Certification contract (see certification.md). A cell is CERTIFIED-EXACT
    iff an ATTAINED lower bound equals a PROVEN upper bound.

    lb: best construction value known to the caller (or None). lb_culik is
    always folded in. Returns (certified: bool, z_if_certified_or_None,
    reason: str).
    """
    ub = best_ub(m, n, s, t)
    lbc = lb_culik(m, n, s, t)
    cands = [v for v in (lb, lbc) if v is not None]
    if not cands:
        return (False, None, "no attained lower bound supplied/applicable")
    lo = max(cands)
    if lo > ub:
        raise AssertionError(
            f"lower bound {lo} exceeds proven upper bound {ub} at "
            f"({m},{n};{s},{t}) -- construction or bound is buggy")
    if lo == ub:
        why = "culik==best_ub" if (lbc is not None and lbc == ub) else "construction==best_ub"
        return (True, ub, why)
    return (False, None, f"gap [{lo},{ub}]")


# ---------------------------------------------------------------------------
# Refinement beyond waterfilling: q-local budget profile feasibility
# ---------------------------------------------------------------------------

def _profile_passes_qlocal(cs_desc, m, s, t, qmax):
    """
    Necessary conditions on a column degree profile of a K_{s,t}-free matrix,
    beyond the global budget. For 1 <= q <= min(qmax, s-1):

    For a q-subset Q of rows let lam_Q = #columns whose support contains Q.
    (a) sum_Q lam_Q = sum_j C(c_j, q), so the busiest Q has
        lam* >= ceil(sum_j C(c_j,q) / C(m,q))   [pigeonhole].
    (b) For that Q, summing the s-subset budget localized at Q: each
        (s-q)-subset R of the other m-q rows has Q u R covered <= t-1 times,
        and a column j containing Q covers C(c_j - q, s-q) such R. Hence
        sum_{j contains Q} C(c_j - q, s - q) <= (t-1) * C(m-q, s-q).
    The lam* columns containing Q all have degree >= q, so the CHEAPEST way
    to satisfy (b) uses the lam* smallest degrees >= q. Conditions checked:
      - lam* <= #columns of degree >= q (else infeasible outright);
      - sum of C(c-q, s-q) over the lam* smallest degrees >= q  <=  RHS.
    PROVEN necessary; checked for validity against all 161 known cells by
    --selftest (never rejects the true z).
    """
    for q in range(1, min(qmax, s - 1) + 1):
        denom = comb(m, q)
        if denom == 0:
            continue
        sq = sum(comb(c, q) for c in cs_desc)
        lam = -(-sq // denom)  # ceil
        if lam <= 0:
            continue
        elig = [c for c in cs_desc if c >= q]   # descending
        if lam > len(elig):
            return False
        cheapest = elig[-lam:]                  # lam smallest degrees >= q
        lhs = sum(comb(c - q, s - q) for c in cheapest)
        if lhs > (t - 1) * comb(m - q, s - q):
            return False
    return True


def _side_profile_feasible(m, n, s, t, z, qmax):
    """
    Does ANY column degree profile (n integers in [0,m], sum z) satisfy the
    global column budget AND the q-local conditions? DFS over nonincreasing
    profiles with PROVEN prunes (balanced-completion minimal budget cost).
    Returns a witness profile or None. If None, no K_{s,t}-free m x n matrix
    with z ones exists (all conditions are necessary), so z is ruled out.
    """
    B = (t - 1) * comb(m, s)
    prof = []

    def rec(left, hi, rsum, bleft):
        if rsum > left * hi:
            return None
        mc = min_cost_of_sum(left, hi, s, rsum)
        if mc is None or mc > bleft:
            return None
        if left == 0:
            return list(prof) if _profile_passes_qlocal(prof, m, s, t, qmax) else None
        # next degree c <= hi, nonincreasing
        lo_c = max(0, rsum - (left - 1) * hi)
        for c in range(min(hi, rsum), lo_c - 1, -1):
            cost = comb(c, s)
            if cost > bleft:
                continue
            prof.append(c)
            got = rec(left - 1, c, rsum - c, bleft - cost)
            prof.pop()
            if got is not None:
                return got
        return None

    return rec(n, m, z, B)


def ub_profile(m, n, s, t, qmax=99, zmin=0):
    """
    PROVEN upper bound refining ub_waterfill: the largest z <= ub_waterfill
    for which BOTH the column-side and row-side degree profiles admit a
    solution passing global budget + q-local conditions (q up to qmax).
    Feasibility is downward-closed in z (decrementing any degree keeps every
    condition satisfied), so the first feasible z from the top is the max.
    Stops early at zmin (returns zmin if everything above is infeasible --
    caller must know z >= zmin is attained). Can be slow; not in UB_REGISTRY.
    """
    wf = ub_waterfill(m, n, s, t)
    for z in range(wf, zmin - 1, -1):
        if z == zmin:
            return z
        if (_side_profile_feasible(m, n, s, t, z, qmax) is not None and
                _side_profile_feasible(n, m, t, s, z, qmax) is not None):
            return z
    return zmin


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------

def known_exact_33():
    """
    The 161 PROVEN-EXACT z(m,n;3,3) cells from ../../evaluator.py
    (KST_EXACT_VALUE). Import is side-effect-free. Returned as
    {(m,n): z} with m <= n.
    """
    path = os.path.join(_BASE, "evaluator.py")
    spec = importlib.util.spec_from_file_location("zar_evaluator", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.KST_EXACT_VALUE)


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def _selftest():
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)
            print("FAIL:", msg)

    # --- 1. waterfilling optimality: exhaustive brute force ---------------
    # all k,cap <= 7, r <= 3 (plus r=4, cap<=6), EVERY budget 0..maxcost+2.
    print("[1] waterfill primitive vs brute force over all degree profiles ...")
    ncase = 0
    for r in (1, 2, 3, 4):
        capmax = 7 if r <= 3 else 6
        for cap in range(0, capmax + 1):
            for k in range(0, 8):
                best_by_cost = {}
                for prof in combinations_with_replacement(range(cap + 1), k):
                    c = sum(comb(x, r) for x in prof)
                    ssum = sum(prof)
                    if c not in best_by_cost or best_by_cost[c] < ssum:
                        best_by_cost[c] = ssum
                maxcost = max(best_by_cost) if best_by_cost else 0
                # brute optimum for every budget = running max over cost <= b
                costs = sorted(best_by_cost)
                run, i, cur = {}, 0, 0
                for b in range(maxcost + 3):
                    while i < len(costs) and costs[i] <= b:
                        cur = max(cur, best_by_cost[costs[i]])
                        i += 1
                    run[b] = cur
                for b in range(maxcost + 3):
                    g = max_sum_under_budget(k, cap, r, b)
                    ncase += 1
                    if g != run[b]:
                        check(False, f"waterfill k={k} cap={cap} r={r} b={b}: greedy {g} != brute {run[b]}")
    print(f"    {ncase} (k,cap,r,budget) cases, exhaustive profile enumeration: OK" if not fails else "    FAILURES above")

    # --- 1b. min_cost_of_sum vs brute force -------------------------------
    print("[1b] balanced-minimum cost vs brute force ...")
    for r in (1, 2, 3):
        for cap in range(0, 7):
            for k in range(0, 6):
                # brute: min cost per achievable sum
                best = {}
                for prof in combinations_with_replacement(range(cap + 1), k):
                    ssum = sum(prof)
                    c = sum(comb(x, r) for x in prof)
                    if ssum not in best or c < best[ssum]:
                        best[ssum] = c
                for ssum in range(0, k * cap + 2):
                    got = min_cost_of_sum(k, cap, r, ssum)
                    want = best.get(ssum)
                    check(got == want, f"min_cost_of_sum k={k} cap={cap} r={r} sum={ssum}: {got} != {want}")

    # --- 2. ub_waterfill == brute-force min of both relaxations -----------
    print("[2] ub_waterfill vs direct brute force on both sides (m,n<=6, s,t<=3) ...")
    for s in (1, 2, 3):
        for t in (1, 2, 3):
            for m in range(1, 7):
                for n in range(1, 7):
                    Bc = (t - 1) * comb(m, s)
                    Br = (s - 1) * comb(n, t)
                    bc = max(
                        (sum(p) for p in combinations_with_replacement(range(m + 1), n)
                         if sum(comb(x, s) for x in p) <= Bc), default=0)
                    br = max(
                        (sum(p) for p in combinations_with_replacement(range(n + 1), m)
                         if sum(comb(x, t) for x in p) <= Br), default=0)
                    check(ub_waterfill(m, n, s, t) == min(bc, br),
                          f"ub_waterfill({m},{n};{s},{t}) != brute {min(bc, br)}")

    # --- 3. sanity relations on a grid ------------------------------------
    print("[3] symmetry, kst domination, culik<=waterfill, trivial cells ...")
    for s in (1, 2, 3, 4):
        for t in (1, 2, 3, 4):
            for m in range(1, 26, 3):
                for n in range(1, 41, 4):
                    wf = ub_waterfill(m, n, s, t)
                    check(wf == ub_waterfill(n, m, t, s), f"transpose sym ({m},{n};{s},{t})")
                    check(wf <= ub_kst(m, n, s, t), f"kst<waterfill?? ({m},{n};{s},{t})")
                    check(wf <= m * n, f"wf>mn ({m},{n};{s},{t})")
                    lc = lb_culik(m, n, s, t)
                    if lc is not None:
                        check(lc <= wf, f"culik>wf ({m},{n};{s},{t}): {lc}>{wf}")
                    if m < s or n < t:
                        check(wf == m * n, f"degenerate cell ({m},{n};{s},{t}) wf={wf} != {m*n}")

    # --- 4. against the 161 proven-exact (3,3) values ---------------------
    print("[4] 161 known z(m,n;3,3): ub>=z, culik regime exact, qlocal validity ...")
    known = known_exact_33()
    check(len(known) == 161, f"expected 161 known cells, got {len(known)}")
    n_d0 = 0
    for (m, n), z in sorted(known.items()):
        wf = ub_waterfill(m, n, 3, 3)
        check(wf >= z, f"BOUND VIOLATION ub_waterfill({m},{n})={wf} < z={z}")
        if wf == z:
            n_d0 += 1
        lc = lb_culik(m, n, 3, 3)
        if lc is not None:
            check(lc == z == wf, f"culik regime ({m},{n}): lc={lc} z={z} wf={wf}")
        # q-local conditions must never reject the true value's existence
        prof = _side_profile_feasible(m, n, 3, 3, z, qmax=99)
        check(prof is not None, f"qlocal rejects TRUE z at ({m},{n}) -- refinement UNSOUND")
    print(f"    d=0 on {n_d0}/161 cells (counting-tight)")

    # --- 5. culik construction values recomputed directly -----------------
    print("[5] culik value formula spot checks ...")
    for (m, n, s, t, v) in [(3, 3, 3, 3, 8), (4, 8, 3, 3, 24), (3, 23, 3, 3, 48),
                            (5, 20, 3, 3, 60), (2, 5, 1, 3, 4)]:
        check(lb_culik(m, n, s, t) == v, f"culik({m},{n};{s},{t}) != {v}")

    if fails:
        print(f"\nSELFTEST: {len(fails)} FAILURES")
        return 1
    print("\nSELFTEST: all checks passed")
    return 0


# ---------------------------------------------------------------------------
# Table generation (ub_33.csv)
# ---------------------------------------------------------------------------

def _write_tables():
    known = known_exact_33()
    out = os.path.join(_HERE, "ub_33.csv")
    rows = []
    for m in range(3, 21):
        for n in range(m, 41):
            col, row = ub_waterfill_sides(m, n, 3, 3)
            wf = min(col, row)
            kst = ub_kst(m, n, 3, 3)
            lc = lb_culik(m, n, 3, 3)
            z = known.get((m, n))
            d = (wf - z) if z is not None else None
            rows.append({
                "m": m, "n": n,
                "ub_col_waterfill": col, "ub_row_waterfill": row,
                "ub_waterfill": wf, "ub_kst": kst,
                "lb_culik": "" if lc is None else lc,
                "z_known": "" if z is None else z,
                "deficit": "" if d is None else d,
                "counting_tight": "" if d is None else int(d == 0),
                "culik_certified": int(lc is not None and lc == wf),
            })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} cells)")

    # stats
    kn = [r for r in rows if r["deficit"] != ""]
    from collections import Counter
    spec = Counter(r["deficit"] for r in kn)
    print(f"known cells covered: {len(kn)}; deficit spectrum: {dict(sorted(spec.items()))}")
    free = [r for r in rows if r["culik_certified"] and (r["m"], r["n"]) not in known]
    print(f"culik-certified exact OUTSIDE the known 161: {len(free)} cells")
    for r in free:
        print(f"   m={r['m']} n={r['n']} z={r['ub_waterfill']}")


# ---------------------------------------------------------------------------
# Refinement scan (--refine): how much of the deficit do q-local budgets close?
# ---------------------------------------------------------------------------

def _refine_scan():
    known = known_exact_33()
    closed = {}
    print("q-local profile refinement on all d>0 known cells "
          "(ub_profile with qmax=s-1=2, both sides):")
    print(f"{'cell':>10} {'z':>4} {'wf':>4} {'d':>2} {'ub_prof':>7} {'closed':>6}")
    tot = imp = 0
    for (m, n), z in sorted(known.items()):
        wf = ub_waterfill(m, n, 3, 3)
        if wf == z:
            continue
        tot += 1
        up = ub_profile(m, n, 3, 3, qmax=99, zmin=z)
        assert up >= z, f"refinement UNSOUND at ({m},{n}): {up} < {z}"
        closed[(m, n)] = wf - up
        if up < wf:
            imp += 1
        print(f"{(m,n)!s:>10} {z:>4} {wf:>4} {wf-z:>2} {up:>7} {wf-up:>6}")
    print(f"\ncells with d>0: {tot}; cells where q-local refinement improves: {imp}")
    print(f"total deficit closed: {sum(closed.values())} of "
          f"{sum(ub_waterfill(m, n, 3, 3) - z for (m, n), z in known.items())}")
    return closed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--refine", action="store_true")
    args = ap.parse_args()
    rc = 0
    if args.selftest:
        rc = _selftest()
    if args.tables:
        _write_tables()
    if args.refine:
        _refine_scan()
    if not (args.selftest or args.tables or args.refine):
        print(__doc__)
    sys.exit(rc)

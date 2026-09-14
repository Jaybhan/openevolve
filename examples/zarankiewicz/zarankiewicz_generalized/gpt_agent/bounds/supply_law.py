#!/usr/bin/env python3
"""
supply_law.py -- computes the "weight-4 supply" numbers g4(m) and tests the
SUPPLY LAW for the tail of each row of the z(m,n;3,3) table.

Definitions (s=t=3 throughout):
  B(m)  = 2*C(m,3)   -- global triple budget.
  g4(m) = maximum size of a MULTISET of weight-4 column supports (4-subsets
          of the m rows) such that every 3-subset of rows is contained in
          at most 2 of them (counted with multiplicity). Computed EXACTLY by
          branch-and-bound below (complete search; PROVEN when it finishes).

  Waterfill 4-count: k4wf(m,n) = min(n, (B(m) - n) // 3), the number of
  weight-4 columns the level-filling optimum uses when its profile contains
  no column of weight >= 5 ("no-5 regime": that holds iff raising a column
  to 5 is not affordable/possible, checked directly from the profile).

SUPPLY LAW (conjecture; tested here on every applicable known cell):
  In the no-5 regime,   z(m,n;3,3) = 3n + min( k4wf(m,n), g4(m) ),
  equivalently          d(m,n) = max(0, k4wf(m,n) - g4(m)),
  provided n <= B(m) - 3*g4(m) + (g4-slack) -- the weight-3 pads always fit
  because total residual capacity B - 4k >= n - k  <=>  n <= B - 3k.

  Upper-bound side of the law: if the profile has no weight->=5 column, at
  most g4(m) columns can have weight 4 (definition of g4: legality of the
  weight-4 subfamily is necessary), the rest have weight <= 3, so
  z <= 4*min(n,g4) + 3*(n - min(n,g4)) = 3n + min(n, g4). Combined with
  z <= WF = 3n + k4wf this gives z <= 3n + min(k4wf, g4) -- **for optima
  that use no weight->=5 column**. The law's content is that in this regime
  weight->=5 columns never help; that part is empirical here (verified
  against every known cell; brute-force PROVEN for m=6,7 cells with n<=12
  via exact_small).

  Lower-bound side (PROVEN): take any legal weight-4 family of size
  k = min(k4wf, g4) (subfamily of the g4-max family), pad with n-k weight-3
  columns placed on triples with residual capacity (total residual
  B - 4k >= n - k in range), pads consume 1 slot each: legal by counting.

Also prints the classical pair-count upper bound g4(m) <= m(m-1)(m-2)/12
(each pair {a,b} lies in <= m-2 blocks since each block through it covers 2
of the m-2 triples {a,b,x} which have total capacity 2(m-2); sum over pairs).

Usage:  python3 supply_law.py [--gmax M]   (default computes g4 for m=6..9)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import combinations
from math import comb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from upper_bounds import known_exact_33, ub_waterfill  # noqa: E402


def g4(m, max_nodes=200_000_000):
    """
    Exact maximum multiset of 4-subsets of [m] with every 3-subset covered
    <= 2 (with multiplicity). Branch and bound over the canonical
    nondecreasing block sequence. Returns (value, complete, nodes).
    """
    triples = list(combinations(range(m), 3))
    tidx = {T: i for i, T in enumerate(triples)}
    blocks = list(combinations(range(m), 4))
    bsubs = [tuple(tidx[T] for T in combinations(b, 3)) for b in blocks]
    B = 2 * len(triples)
    cov = [0] * len(triples)
    best = 0
    nodes = 0
    complete = True

    def dfs(start, count, used):
        nonlocal best, nodes, complete
        nodes += 1
        if nodes > max_nodes:
            complete = False
            return
        if count > best:
            best = count
        # bound: each further block consumes 4 units of remaining budget
        if count + (B - used) // 4 <= best:
            return
        for j in range(start, len(blocks)):
            sj = bsubs[j]
            if any(cov[i] >= 2 for i in sj):
                continue
            for i in sj:
                cov[i] += 1
            dfs(j, count + 1, used + 4)
            for i in sj:
                cov[i] -= 1
            if not complete:
                return

    dfs(0, 0, 0)
    return best, complete, nodes


def wf_profile(m, n):
    """The waterfill column profile at (m,n;3,3) as {height: count}."""
    B = 2 * comb(m, 3)
    prof = {}
    heights = [0] * n
    b = B
    for h in range(m):
        cost = comb(h, 2)
        if cost == 0:
            heights = [x + 1 for x in heights]
            continue
        if cost * n <= b:
            b -= cost * n
            heights = [x + 1 for x in heights]
        else:
            q = b // cost
            for i in range(q):
                heights[i] += 1
            break
    for h in heights:
        prof[h] = prof.get(h, 0) + 1
    return prof


def main(gmax):
    known = known_exact_33()
    print("g4(m): exact max legal weight-4 multiset (every triple <= 2x)")
    gvals = {}
    for m in range(6, gmax + 1):
        t0 = time.time()
        v, complete, nodes = g4(m)
        gvals[m] = (v, complete)
        ub_pair = m * (m - 1) * (m - 2) // 12
        print(f"  g4({m}) = {v}{'' if complete else ' (INCOMPLETE, lower bound only)'}"
              f"   [pair-count ub {ub_pair}; nodes {nodes}; {time.time()-t0:.1f}s]")

    print("\nSUPPLY LAW test on every known cell in the no-5 waterfill regime:")
    print(f"{'cell':>9} {'z':>4} {'wf':>4} {'d':>2} {'k4wf':>4} {'law_z':>5}  verdict")
    npass = nfail = 0
    for (m, n), z in sorted(known.items()):
        if m not in gvals or not gvals[m][1]:
            continue
        prof = wf_profile(m, n)
        if max(prof) > 4:
            continue                      # mixed regime: law makes no claim
        g = gvals[m][0]
        k4 = prof.get(4, 0)
        law = 3 * n + min(k4, g)
        wf = ub_waterfill(m, n, 3, 3)
        ok = law == z
        npass += ok
        nfail += (not ok)
        print(f"{(m,n)!s:>9} {z:>4} {wf:>4} {wf-z:>2} {k4:>4} {law:>5}  "
              f"{'PASS' if ok else '*** FAIL'}")
    print(f"\nlaw verified on {npass} cells, failed on {nfail} "
          f"(claim range: all known cells, rows with exact g4)")

    print("\nFalsifiable predictions beyond the table (no-5 regime, tail):")
    for m in sorted(gvals):
        g, complete = gvals[m]
        if not complete:
            continue
        B = 2 * comb(m, 3)
        n_on = None
        preds = []
        for n in range(m, B + 1):
            prof = wf_profile(m, n)
            if max(prof) > 4:
                continue
            if n_on is None:
                n_on = n
            k4 = prof.get(4, 0)
            d = max(0, k4 - g)
            if (m, n) not in known and d > 0:
                preds.append((n, d))
        d0_from = None
        for n in range(n_on or m, B + 1):
            prof = wf_profile(m, n)
            if max(prof) <= 4 and prof.get(4, 0) <= g:
                d0_from = n
                break
        print(f"  m={m}: no-5 regime from n={n_on}; predicted d>0 cells beyond table: "
              f"{preds if preds else 'none'}; predicted d=0 for all n >= {d0_from}")


# Exact supply constants PROVEN by complete search in this file (see main()):
#   g4(6) = 9   (= ex(6,K3), Turan; re-derivation of coordinator Thm 2 supply)
#   g4(7) = 15  (witness: 15 distinct 4-blocks, 30 of 35 triples covered 2x)
#   g4(8) = 28  (= pair-count bound; witness: doubled SQS(8), every triple 2x)
G4_PROVEN = {6: 9, 7: 15, 8: 28}


def supply_certified_cells(n_max=40, known=None):
    """
    Cells (m,n;3,3) OUTSIDE the known table whose exact value is CERTIFIED
    by supply-law construction == ub_waterfill:

    If the waterfill profile at (m,n) has max level <= 4 and its weight-4
    count k4 <= g4(m), then
      * UPPER: z <= ub_waterfill (PROVEN, unconditional);
      * LOWER: k4-subfamily of the g4(m)-max family + (n-k4) weight-3 pads
        on residual triples + weight-2 pads if any. Pads fit: total residual
        capacity B - 4*k4 >= #pads because the waterfill profile satisfies
        the budget; each residual unit accepts exactly one weight-3 pad.
        The construction has exactly ub_waterfill ones.
    Hence z = ub_waterfill: CERTIFIED-EXACT.
    Returns [(m, n, z), ...].
    """
    if known is None:
        known = known_exact_33()
    out = []
    for m, g in sorted(G4_PROVEN.items()):
        for n in range(m, n_max + 1):
            if (m, n) in known:
                continue
            prof = wf_profile(m, n)
            if max(prof) > 4:
                continue
            if prof.get(4, 0) <= g:
                out.append((m, n, ub_waterfill(m, n, 3, 3)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmax", type=int, default=9)
    ap.add_argument("--free-exact", action="store_true",
                    help="list beyond-table cells certified by supply construction")
    args = ap.parse_args()
    if args.free_exact:
        cells = supply_certified_cells()
        print("beyond-table (3,3) cells CERTIFIED-EXACT by supply construction == waterfill")
        for m, n, z in cells:
            print(f"  m={m} n={n} z={z}")
        print(f"total: {len(cells)}")
    else:
        main(args.gmax)

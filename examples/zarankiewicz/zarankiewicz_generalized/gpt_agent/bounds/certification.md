# Certification contract for z(m,n;s,t)

Author: bounds prover agent, 2026-07-28. API in `bounds/upper_bounds.py`.

## The contract

A cell (m,n;s,t) is **CERTIFIED-EXACT** iff

    best_attained_lower_bound(m,n,s,t) == best_ub(m,n,s,t)

where the left side is the edge count of an EXPLICIT construction that has
been verified K_{s,t}-free (e.g. by the evaluator's `has_kst`), and the right
side is a PROVEN upper bound from this module. No other evidence counts:
matching a published number is VERIFIED-ALL-KNOWN, not certification.

API (`from bounds.upper_bounds import ...`):

- `ub_waterfill(m,n,s,t)` — PROVEN upper bound; exact integer optimum of the
  two budget relaxations `sum_j C(c_j,s) <= (t-1)C(m,s)` (columns) and
  `sum_i C(r_i,t) <= (s-1)C(n,t)` (rows), by marginal-cost waterfilling.
  The optimality of the greedy is unit-tested exhaustively
  (`--selftest`: all k,cap <= 7, r <= 4, every budget, vs full enumeration).
- `ub_kst(m,n,s,t)` — REFERENCE closed-form Kővári–Sós–Turán bound; valid
  but never better than ub_waterfill (asserted on a grid in the selftest).
- `ub_profile(m,n,s,t)` — PROVEN refinement via q-local budgets (slower;
  closes at most 1 on current cells — see deficit_analysis.md §4).
- `lb_culik(m,n,s,t)` — PROVEN constructive lower bound
  (s-1)n + (t-1)C(m,s) when s <= m and n >= (t-1)C(m,s), transpose
  symmetrically; None outside the regime.
- `best_ub(m,n,s,t)` — min over the registry (extend `UB_REGISTRY`).
- `certify_cell(m,n,s,t, lb=...)` — returns (certified, z, reason); folds in
  lb_culik automatically; raises if a claimed lb EXCEEDS a proven ub (that
  means a bug in somebody's code — treat as fatal).

For the construction engineer: a construction on a d=0 cell only needs to
MEET ub_waterfill to certify the cell. `ub_33.csv` column `counting_tight`
lists all 78 such known cells; the waterfill degree profile (near-balanced,
levels from the greedy) is the profile to aim for.

## Certified-exact (3,3) cells OUTSIDE the known 161 (m <= 20, n <= 40)

Two PROVEN mechanisms already certify **99 distinct cells** beyond the
table, no search required. Both lists are regenerable:
`python3 upper_bounds.py --tables` (Culík) and
`python3 supply_law.py --free-exact` (supply construction).

### (A) Culík-regime cells: lb_culik == ub_waterfill  (52 cells)

z = 2n + 2C(m,3) at every (m,n) with n >= 2C(m,3) in range:

- m=3, n=24..40 (17 cells): z = 2n+2.
- m=4, n=24..40 (17 cells): z = 2n+8.
- m=5, n=24..40 (17 cells): z = 2n+20.
- (6,40): z = 120.

These are inside Culík's 1956 theorem regime, so they are certainly known to
the table's authors (the table simply stops at n=23); enumerated for
completeness and for the harness's use, NOT claimed novel.

### (B) Supply-construction cells: g4-family + pads == ub_waterfill  (48 cells)

Mechanism (deficit_analysis.md §5): when the waterfill profile has no
column of weight >= 5 and its weight-4 count k4 <= g4(m) (exact supply
numbers g4(6)=9, g4(7)=15, g4(8)=28, PROVEN by complete search), the
construction [k4-subfamily of the g4-max family] + [weight-3 pads on
residual triples] meets ub_waterfill exactly. Both sides PROVEN, hence
CERTIFIED-EXACT, unconditionally:

- m=6, n=24..40 (17 cells): z = 3n + min(⌊(40−n)/3⌋, 9)
  = 77, 80, 82, 85, 88, 90, 93, 96, 98, 101, 104, 106, 109, 112, 114, 117, 120.
- m=7, n=24..40 (17 cells): z = 3n + min(⌊(70−n)/3⌋, 15)
  = 87, 90, 92, 95, 98, 100, 103, 106, 108, 111, 114, 116, 119, 122, 124, 127, 130.
- m=8, n=27..40 (14 cells): z = 3n + min(n, ⌊(112−n)/3⌋, 28)
  (the n-cap binds only at n=27)
  = 108, 112, 114, 117, 120, 122, 125, 128, 130, 133, 136, 138, 141, 144.

Overlap between (A) and (B): only (6,40) (same value 120 both ways —
consistency check passes). Total distinct: 52 + 48 − 1 = **99**.

NOVELTY FLAG for the theory agent: the m=6,7,8 mid-range certified cells
(below the Culík threshold — e.g. every (6,n) for 24 <= n <= 39, (7,n) for
n >= 24, (8,n) for n >= 27) are NOT covered by Culík's theorem. They follow
from the supply construction plus the counting bound. Guy's 1969 tables and
the Roman-bound literature may well contain equivalent statements
(Roman's bound specializes to piecewise-linear forms like 3n + const) —
please check before any novelty claim. The specific supply identities
g4(7) = 15 (strictly between doubled-Fano = 14 and the pair bound 17) and
g4(8) = 28 = 2·|SQS(8)| look like the checkable new kernel.

### What is NOT certified

- Cells beyond `_EXACT_UP_TO` with 9 <= m <= 16, n <= 23: the published
  figures there equal ub_waterfill exactly (deficit_analysis.md §3g), so
  certification hinges entirely on finding constructions that MEET WF; no
  bound improvement will help (WF is already the published ub). Cells where
  z < WF is suspected need refutation machinery instead (exhaustion is
  feasible to about m·n ≈ 90–100 with exact_small.py; (9,10) took 250 s).
- Everything with m >= 9, n > 40 in the supply regime is outside this
  window anyway (no-5-regime onset n >= C(m,3)/2 = 42 at m=9). For the
  record: g4(9) = S_9(0,0) = **40** [ILP-PROVEN via supply_table.py; the
  pair bound 42 is impossible since a 3-(9,4,2) design fails divisibility
  (2*8*7/6 not an integer), and the ILP shows even 41 is impossible].
  Supply-law predictions for the m=9 tail: d(9,n) = 2,1,1,1 at n=42..45 and
  d(9,n) = 0 for all n >= 46 (below Culik threshold 168).

## Ground-truth hygiene

- `exact_small.csv`: every row with status PROVEN is complete-search ground
  truth (witnesses re-verified against the evaluator's checker; zero
  mismatches vs the published 161 on all 31 overlap cells: the 10 with
  m,n <= 6 plus the 21 extended cells through (9,10)).
- Any conflict between a "certified" value and any future exhaustive search
  is a fatal bug: report immediately, certify nothing further until
  resolved.

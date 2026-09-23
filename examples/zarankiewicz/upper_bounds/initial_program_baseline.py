"""Initial prune library for the Zarankiewicz upper-bound search.

WHAT IS EVOLVED.  Two coupled things inside the EVOLVE block:

  LEAN_SOURCE -- Lean 4 code (spliced into `namespace ZarPrune.Cand`, with
                 `target : Params` injected by the gate) that must define
                     def candidate (P : Params) : Prune P        (general), or
                     def candidate : Prune target                (instance-specific).
                 A `Prune P` is a computable `kill : Profile P.m P.n -> Bool` plus a
                 proof `sound : ∀ A, kill (profileOf A) = true -> ¬ Valid P A`.
                 Only prunes whose proof elaborates (no sorry/axioms) earn credit.
  kill(...)   -- a Python mirror of candidate.kill, used for fast empirical
                 screening against the counterexample battery and for partial
                 credit while the Lean proof is still being worked out.

THE CONTRACT.  kill may return True ONLY for cases that contain no K_{s,t}-free
matrix with those exact row sums `rows` (length m, non-increasing) and column
sums `cols` (length n, non-increasing) and total >= w.  Killing a realizable
case is unsound: the evaluator has witnesses and will reject the program.
Symmetry breaking ("assume rows sorted") is NOT a prune -- the SAT encoding
already does it.  A prune must be a counting / structural impossibility argument.

Available Lean API (ZarPrune, Mathlib-free): sumFin, allFin, sumFin_swap (Fubini),
sumFin_le, allFin_iff, not_allFin_elim; Params{m,n,s,t,w}, Mat, ind, rowSum,
colSum, weight, rowSum_le, colSum_le, weight_eq_sum_colSum, HasKst, Valid,
Profile{row,col}, profileOf; Prune{name,kill,sound}, Prune.never, Prune.or,
Prune.ofList; proved prunes deficit, mismatch, rowCap, colCap, baseline.
"""

# EVOLVE-BLOCK-START
LEAN_SOURCE = r'''
/-- The evolved prune library.  Start: the four baseline prunes folded together.
    Add new `def myPrune (P : Params) : Prune P where ...` blocks above this and
    list them here. -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [deficit P, mismatch P, rowCap P, colCap P]
'''


def kill(m, n, s, t, w, rows, cols):
    """Python mirror of `candidate.kill`.  rows/cols are non-increasing tuples."""
    total = sum(rows)
    if total < w:                      # deficit
        return True
    if total != sum(cols):             # mismatch
        return True
    if any(r > n for r in rows) or any(c > m for c in cols):  # rowCap / colCap
        return True
    return False
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    print(kill(9, 9, 3, 3, 50, (6, 6, 6, 6, 6, 5, 5, 5, 5), (6, 6, 6, 6, 6, 5, 5, 5, 5)))

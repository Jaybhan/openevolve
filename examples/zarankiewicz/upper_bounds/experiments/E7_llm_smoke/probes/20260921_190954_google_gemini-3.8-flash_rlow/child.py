"""Initial prune library for the Zarankiewicz upper-bound search (frontier start).

WHAT IS EVOLVED.  Two coupled things inside the EVOLVE block:

  LEAN_SOURCE -- Lean 4 code (spliced into `namespace ZarPrune.Cand` after
                 `import ZarPrune`, with `target : Params` injected by the gate) that must define
                     def candidate (P : Params) : Prune P        (general), or
                     def candidate : Prune target                (instance-specific).
                 A `Prune P` is a computable `kill : Profile P.m P.n -> Bool` plus a proof
                 `sound : ∀ A, kill (profileOf A) = true -> ¬ Valid P A`, where
                 Valid P A := ¬ HasKst P A ∧ P.w ≤ weight A.
                 Only prunes whose proof elaborates (no sorry / axioms / native_decide) earn credit;
                 the gate's own `#eval` of `candidate.kill` is what prunes cases.
  kill(...)   -- a Python mirror of candidate.kill for fast empirical screening; it must agree
                 with the Lean kill on every case.

THE CONTRACT.  kill may return True ONLY for cases that contain no K_{s,t}-free matrix with
exactly those row sums `rows` (length m, non-increasing) and column sums `cols` (length n,
non-increasing) and total >= w.  Killing a realizable case is unsound: the evaluator holds
witnesses for many cases and rejects any program that kills one (score 0).  Symmetry
breaking ("assume rows sorted") is NOT a prune -- the SAT encoding already does it.

LEAN API (ZarPrune; Mathlib is available through targeted imports already made by
`import ZarPrune` -- Finset, BigOperators, Nat.choose, Nat.findGreatest, omega, linarith,
ring, positivity, decide, simp).  Mathlib-free core (Sum/Basic/Prune/Prunes):
  sumFin k f, allFin k f, sumFin_swap (Fubini), sumFin_le, sumFin_add, allFin_iff, not_allFin_elim
  Params{m,n,s,t,w}, Mat m n := Fin m → Fin n → Bool, ind, rowSum, colSum, weight,
  rowSum_le, colSum_le, weight_eq_sum_colSum, Incr, HasKst, Valid, Profile{row,col}, profileOf
  Prune{name,kill,sound}, Prune.never, Prune.or, Prune.ofList, Prune.skip, upper_bound_of_cover
  proved prunes: deficit, mismatch, rowCap, colCap, baseline
Counting core (Counting.lean, PROVED):
  sumFin_eq_sum : sumFin k f = ∑ i, f i          (bridge to Finset sums)
  support A j : Finset (Fin m), card_support : (support A j).card = colSum A j; rowSupport, card_rowSupport
  hasKst_of_subsets : R.card = s → C.card = t → (∀ j ∈ C, R ⊆ support A j) → HasKst P A
  budget_general (T J S) : (∀ R ∈ T.powersetCard k, (J.filter (R ⊆ S ·)).card ≤ B) →
                           ∑ j ∈ J, (S j).card.choose k ≤ B * T.card.choose k   (generic double counting)
  colBudget : ¬HasKst P A → ∑ j, (colSum A j).choose s ≤ (t-1) * m.choose s      (Argument A)
  rowBudget : ¬HasKst P A → ∑ i, (rowSum A i).choose t ≤ (s-1) * n.choose t
  rowLocalBudget : ¬HasKst P A → ∑ j ∈ rowSupport A i, (colSum A j - 1).choose (s-1) ≤ (t-1) * (m-1).choose (s-1)  (Argument D)
  Params.transpose, transpose A, Profile.swap, hasKst_transpose (↔), valid_transpose,
  Prune.transposed : Prune P.transpose → Prune P      (prove one side, get the other free)
  deleteCol A j / deleteRow A i, weight_deleteCol : weight (deleteCol A j) + colSum A j = weight A,
  not_hasKst_deleteCol, valid_deleteCol_bound (hU : every K-free m×(n-1) matrix has weight ≤ U) (hj : colSum A j + U < w) : ¬ Valid P A
  fD s c = (c-1).choose (s-1), cntLt pf y = #columns with sum < y, boundD P pf r = sum of fD over the r lightest columns (layer-cake), boundD_le
  equalCost, waterfillBound m n s B, sum_le_waterfillBound (waterfilling optimality), colBudgetOf P, weight_le_waterfill
  prunes: argA, argAT, argD, argDT, argDelCol P U hU, argDelRow P U hU, argDelColWF, argDelRowWF, argWF,
          countingA, countingD, deletion, counting (all seven folded)
Pattern for a new prune:
  def myPrune (P : Params) : Prune P where
    name := "..."
    kill := fun pf => decide (<arithmetic on pf.row / pf.col via sumFin / allFin / List.finRange>)
    sound := by
      intro A h hv            -- h : kill (profileOf A) = true ; hv : Valid P A  (hv.1 : ¬HasKst, hv.2 : w ≤ weight A)
      ...                     -- derive the counting inequality from colBudget/rowLocalBudget/budget_general, contradict h with omega
"""
from math import comb

# EVOLVE-BLOCK-START
LEAN_SOURCE = r'''
/-- Transposed waterfilling and deletion prunes. -/
def countingTransposed (P : Params) : Prune P :=
  Prune.ofList [
    (argWF P.transpose).transposed,
    (argDelColWF P.transpose).transposed,
    (argDelRowWF P.transpose).transposed
  ]

/-- Baseline prunes OR counting library (both primal and transposed waterfilling). -/
def candidate (P : Params) : Prune P :=
  Prune.or (baseline P) (Prune.or (counting P) (countingTransposed P))
'''


def _nsub(a, b):
    """Lean `Nat` subtraction (truncated at 0)."""
    return a - b if a >= b else 0


# ---- Argument A ------------------------------------------------------------
def _argA(m, n, s, t, w, rows, cols):
    return (t - 1) * comb(m, s) < sum(comb(c, s) for c in cols)


def _argAT(m, n, s, t, w, rows, cols):
    return (s - 1) * comb(n, t) < sum(comb(r, t) for r in rows)


# ---- Argument D (layer-cake form of "sum of fD over the r lightest columns") --
def _fD(s, c):
    return comb(_nsub(c, 1), _nsub(s, 1))


def _gD(s, x):
    return _nsub(_fD(s, x), _fD(s, _nsub(x, 1)))


def _boundD(m, s, cols, r):
    def cnt_lt(y):
        return sum(1 for c in cols if c < y)
    return r * _fD(s, 0) + sum(_nsub(r, cnt_lt(x + 1)) * _gD(s, x + 1) for x in range(m))


def _argD(m, n, s, t, w, rows, cols):
    if s < 1:
        return False
    rhs = (t - 1) * comb(_nsub(m, 1), _nsub(s, 1))
    return any(_boundD(m, s, cols, r) > rhs for r in rows)


def _argDT(m, n, s, t, w, rows, cols):
    # argD on the transposed instance (n, m, t, s, w) with the swapped profile
    return _argD(n, m, t, s, w, cols, rows)


# ---- Waterfilling / deletion -------------------------------------------------
def _col_budget(m, s, t):
    return (t - 1) * comb(m, s)


def _equal_cost(n, s, S):
    if n == 0:
        return 0  # S / 0 = 0 and S % 0 = S in Lean, n * ... = 0
    q, r = divmod(S, n)
    return n * comb(q, s + 1) + r * comb(q, s)


def _waterfill_bound(m, n, s, B):
    """Nat.findGreatest (fun S => equalCost n (s-1) S ≤ B) (n*m)."""
    for S in range(n * m, -1, -1):
        if _equal_cost(n, _nsub(s, 1), S) <= B:
            return S
    return 0


def _argDelColWF(m, n, s, t, w, rows, cols):
    if s < 1:
        return False
    n1 = _nsub(n, 1)
    U = _waterfill_bound(m, n1, s, _col_budget(m, s, t))
    return any(c + U < w for c in cols)


def _argDelRowWF(m, n, s, t, w, rows, cols):
    if s < 1:
        return False
    m1 = _nsub(m, 1)
    U = _waterfill_bound(m1, n, s, _col_budget(m1, s, t))
    return any(r + U < w for r in rows)


def _argWF(m, n, s, t, w, rows, cols):
    return s >= 1 and _waterfill_bound(m, n, s, _col_budget(m, s, t)) < w


# ---- baseline ----------------------------------------------------------------
def _baseline(m, n, s, t, w, rows, cols):
    total = sum(rows)
    if total < w:                      # deficit
        return True
    if total != sum(cols):             # mismatch
        return True
    if any(r > n for r in rows) or any(c > m for c in cols):  # rowCap / colCap
        return True
    return False


def _argWFT(m, n, s, t, w, rows, cols):
    return _argWF(n, m, t, s, w, cols, rows)


def _argDelColWFT(m, n, s, t, w, rows, cols):
    # argDelColWF on transposed instance corresponds to (argDelColWF P.transpose).transposed
    return _argDelColWF(n, m, t, s, w, cols, rows)


def _argDelRowWFT(m, n, s, t, w, rows, cols):
    # argDelRowWF on transposed instance corresponds to (argDelRowWF P.transpose).transposed
    return _argDelRowWF(n, m, t, s, w, cols, rows)


COUNTING = (
    _argA, _argAT, _argD, _argDT,
    _argDelColWF, _argDelRowWF, _argWF,
    _argWFT, _argDelColWFT, _argDelRowWFT
)


def kill(m, n, s, t, w, rows, cols):
    """Python mirror of `candidate.kill` = baseline || counting."""
    if _baseline(m, n, s, t, w, rows, cols):
        return True
    return any(p(m, n, s, t, w, rows, cols) for p in COUNTING)
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    print(kill(9, 9, 3, 3, 50, (6, 6, 6, 6, 6, 5, 5, 5, 5), (6, 6, 6, 6, 6, 5, 5, 5, 5)))

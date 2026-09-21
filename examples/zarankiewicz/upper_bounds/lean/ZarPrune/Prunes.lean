import ZarPrune.Prune

/-
Worked prunes.

These three are the baseline the evolutionary search starts from: each one is a
complete `Prune`, i.e. a computable filter *and* its soundness proof.  They are
deliberately elementary -- the point is that the obligation is discharged, not
that the arguments are deep.
-/

namespace ZarPrune

/-- Kill a case whose row sums do not add up to at least `w` ones. -/
def deficit (P : Params) : Prune P where
  name := "weight-deficit"
  kill := fun pf => decide (sumFin P.m pf.row < P.w)
  sound := by
    intro A h hv
    have h1 : sumFin P.m (rowSum A) < P.w := of_decide_eq_true h
    have h2 : P.w ≤ sumFin P.m (rowSum A) := hv.2
    omega

/-- Kill a case whose row sums and column sums disagree on the total number of
ones.  Sound because both count the same set of cells (`weight_eq_sum_colSum`). -/
def mismatch (P : Params) : Prune P where
  name := "row/col total mismatch"
  kill := fun pf => decide (sumFin P.m pf.row ≠ sumFin P.n pf.col)
  sound := by
    intro A h _
    have h1 : sumFin P.m (rowSum A) ≠ sumFin P.n (colSum A) := of_decide_eq_true h
    exact h1 (weight_eq_sum_colSum A)

/-- Kill a case asking for a row with more than `n` ones. -/
def rowCap (P : Params) : Prune P where
  name := "row-sum exceeds n"
  kill := fun pf => !allFin P.m (fun i => decide (pf.row i ≤ P.n))
  sound := by
    intro A h _
    exact not_allFin_elim (fun i => decide_eq_true (rowSum_le A i)) h

/-- Kill a case asking for a column with more than `m` ones. -/
def colCap (P : Params) : Prune P where
  name := "col-sum exceeds m"
  kill := fun pf => !allFin P.n (fun j => decide (pf.col j ≤ P.m))
  sound := by
    intro A h _
    exact not_allFin_elim (fun j => decide_eq_true (colSum_le A j)) h

/-- The baseline filter the harness runs before emitting any SAT instance. -/
def baseline (P : Params) : Prune P :=
  Prune.ofList P [deficit P, mismatch P, rowCap P, colCap P]

end ZarPrune

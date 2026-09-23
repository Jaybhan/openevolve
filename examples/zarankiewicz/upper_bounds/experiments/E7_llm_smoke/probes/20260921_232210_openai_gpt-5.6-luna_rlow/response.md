<<<<<<< SEARCH
/-- Pattern for a NEW prune: `kill` on the profile, `sound` via library lemmas. This one kills nothing. -/
def examplePrune (P : Params) : Prune P where
  name := "example (kills nothing)"
  kill := fun _ => false
  sound := by intro A h; simp at h

/-- The evolved library. Keep `counting P` first; append new prunes. -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [counting P, examplePrune P]
=======
/-- A water-filled strengthening of the column counting bound.  The ordinary
column budget bounds the sum of binomial costs; water filling converts that
cost bound into a sharper bound on the total column degree. -/
def waterfillPrune (P : Params) : Prune P where
  name := "waterfill column degree bound"
  kill := fun pf =>
    decide
      (1 ≤ P.s ∧
        waterfillBound P.m P.n P.s
            ((P.t - 1) * (P.m).choose P.s) <
          sumFin P.n (fun j => pf.col j))
  sound := by
    intro A hk hv
    have hk' :
        1 ≤ P.s ∧
          waterfillBound P.m P.n P.s
              ((P.t - 1) * (P.m).choose P.s) <
            sumFin P.n (fun j => (colSum A j)) :=
      of_decide_eq_true hk
    have hB :
        sumFin P.n (fun j => (colSum A j).choose P.s) ≤
          (P.t - 1) * (P.m).choose P.s := by
      rw [sumFin_eq_sum]
      exact colBudget P A hv.1
    have hc : ∀ j : Fin P.n, colSum A j ≤ P.m := by
      intro j
      exact colSum_le P A j
    have hw :
        sumFin P.n (fun j => colSum A j) ≤
          waterfillBound P.m P.n P.s
            ((P.t - 1) * (P.m).choose P.s) := by
      exact
        sum_le_waterfillBound P.m P.s
          ((P.t - 1) * (P.m).choose P.s)
          (fun j => colSum A j) hc hB
    rw [← profileOf_col A] at hk'
    omega

/-- The evolved library. Keep `counting P` first; append new prunes. -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [counting P, waterfillPrune P]
>>>>>>> REPLACE
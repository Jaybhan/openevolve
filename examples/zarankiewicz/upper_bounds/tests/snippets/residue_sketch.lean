-- SNIPPET:residue_sketch
/-- Sketch mode (design §4.5 B): the same summed-budget prune with both counting
facts left as typed holes.  Neither hole is closed by the auto-fill tactic list
(they need `colBudget P A hv.1` / `rowBudget P A hv.1` with arguments), so the
expected ladder is L2 and the holes are reported with their goals. -/
def residueSketch (P : Params) : Prune P where
  name := "summed budgets (sketch)"
  kill := fun pf => decide ((P.t - 1) * (P.m).choose P.s + (P.s - 1) * (P.n).choose P.t
      < sumFin P.n (fun j => (pf.col j).choose P.s) + sumFin P.m (fun i => (pf.row i).choose P.t))
  sound := by
    intro A hk hv
    have h1 : (P.t - 1) * (P.m).choose P.s + (P.s - 1) * (P.n).choose P.t
        < sumFin P.n (fun j => (colSum A j).choose P.s) + sumFin P.m (fun i => (rowSum A i).choose P.t) :=
      of_decide_eq_true hk
    rw [sumFin_eq_sum, sumFin_eq_sum] at h1
    have hc : ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s := by sorry
    have hr : ∑ i, (rowSum A i).choose P.t ≤ (P.s - 1) * (P.n).choose P.t := by sorry
    omega

def candidate (P : Params) : Prune P :=
  Prune.or (baseline P) (Prune.or (counting P) (residueSketch P))

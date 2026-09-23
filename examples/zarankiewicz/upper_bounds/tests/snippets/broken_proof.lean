-- SNIPPET:broken_proof
/-- The summed-budget prune with a wrong final step (`exact hc` where the goal is
`False`): the kill type-checks, the proof does not -> L1. -/
def farkasBroken (P : Params) : Prune P where
  name := "farkas (broken proof)"
  kill := fun pf => decide ((P.t - 1) * (P.m).choose P.s + (P.s - 1) * (P.n).choose P.t
      < sumFin P.n (fun j => (pf.col j).choose P.s) + sumFin P.m (fun i => (pf.row i).choose P.t))
  sound := by
    intro A hk hv
    have h1 : (P.t - 1) * (P.m).choose P.s + (P.s - 1) * (P.n).choose P.t
        < sumFin P.n (fun j => (colSum A j).choose P.s) + sumFin P.m (fun i => (rowSum A i).choose P.t) :=
      of_decide_eq_true hk
    have hc := colBudget P A hv.1
    exact hc

def candidate (P : Params) : Prune P :=
  Prune.or (baseline P) (Prune.or (counting P) (farkasBroken P))

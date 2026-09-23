-- SNIPPET:schema_farkas
/-- Farkas combination (multipliers 1, 1) of the column budget (Argument A) and the
row budget (Argument Aᵀ): kill when the SUM of both left-hand sides exceeds the sum
of both budgets.  Weaker than `argA ∨ argAT` pointwise, but it is the shape every
`Prune.ofFarkas` schema instance takes (design §2.3). -/
def farkasAA (P : Params) : Prune P where
  name := "farkas: colBudget + rowBudget"
  kill := fun pf => decide ((P.t - 1) * (P.m).choose P.s + (P.s - 1) * (P.n).choose P.t
      < sumFin P.n (fun j => (pf.col j).choose P.s) + sumFin P.m (fun i => (pf.row i).choose P.t))
  sound := by
    intro A hk hv
    have h1 : (P.t - 1) * (P.m).choose P.s + (P.s - 1) * (P.n).choose P.t
        < sumFin P.n (fun j => (colSum A j).choose P.s) + sumFin P.m (fun i => (rowSum A i).choose P.t) :=
      of_decide_eq_true hk
    rw [sumFin_eq_sum, sumFin_eq_sum] at h1
    have hc := colBudget P A hv.1
    have hr := rowBudget P A hv.1
    omega

def candidate (P : Params) : Prune P :=
  Prune.or (baseline P) (Prune.or (counting P) (farkasAA P))

-- SNIPPET:deletion
/-- Delete one column, then bound the `m × (n-1)` remainder by the ROW-side
waterfilled budget (`weight_le_waterfill` on the transposed instance).  A genuinely
different bound from `argDelColWF`, which uses the column side. -/
def argDelColWFT (P : Params) : Prune P :=
  if ht : 1 ≤ P.t then
    argDelCol P (waterfillBound (P.n - 1) P.m P.t (colBudgetOf ⟨P.n - 1, P.m, P.t, P.s, 0⟩))
      (fun B hB => by
        have h := weight_le_waterfill ⟨P.n - 1, P.m, P.t, P.s, 0⟩ ht (transpose B)
          (fun hK => hB ((hasKst_transpose ⟨P.m, P.n - 1, P.s, P.t, 0⟩ B).mp hK))
        rw [weight_transpose] at h
        exact h)
  else Prune.never P

def candidate (P : Params) : Prune P :=
  Prune.or (baseline P) (Prune.or (counting P) (argDelColWFT P))

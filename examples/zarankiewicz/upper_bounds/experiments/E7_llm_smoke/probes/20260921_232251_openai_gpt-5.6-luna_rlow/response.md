The current waterfill proof is close, but it has two API mismatches: `colSum_le` takes the matrix first, and `sum_le_waterfillBound` requires `1 ≤ s`. I also add the transposed row-side waterfill prune, which targets the near-regular surviving profiles from both orientations.

<<<<<<< SEARCH
      exact colSum_le P A j
=======
      exact colSum_le A j
>>>>>>> REPLACE

<<<<<<< SEARCH
        sum_le_waterfillBound P.m P.s
          ((P.t - 1) * (P.m).choose P.s)
          (fun j => colSum A j) hc hB
=======
        sum_le_waterfillBound P.m P.s
          ((P.t - 1) * (P.m).choose P.s)
          hk'.1
          (fun j => colSum A j) hc hB
>>>>>>> REPLACE

<<<<<<< SEARCH
/-- The evolved library. Keep `counting P` first; append new prunes. -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [counting P, waterfillPrune P]
=======
/-- The row-side dual of waterfillPrune.  The row budget bounds
the sum of t-subset costs, and water filling converts it into a
bound on the total row degree. -/
def rowWaterfillPrune (P : Params) : Prune P where
  name := "waterfill row degree bound"
  kill := fun pf =>
    decide
      (1 ≤ P.t ∧
        waterfillBound P.n P.m P.t
            ((P.s - 1) * (P.n).choose P.t) <
          sumFin P.m (fun i => pf.row i))
  sound := by
    intro A hk hv
    have hk' :
        1 ≤ P.t ∧
          waterfillBound P.n P.m P.t
              ((P.s - 1) * (P.n).choose P.t) <
            sumFin P.m (fun i => rowSum A i) :=
      of_decide_eq_true hk
    have hB :
        sumFin P.m (fun i => (rowSum A i).choose P.t) ≤
          (P.s - 1) * (P.n).choose P.t := by
      rw [sumFin_eq_sum]
      exact rowBudget P A hv.1
    have hr : ∀ i : Fin P.m, rowSum A i ≤ P.n := by
      intro i
      exact rowSum_le A i
    have hw :
        sumFin P.m (fun i => rowSum A i) ≤
          waterfillBound P.n P.m P.t
            ((P.s - 1) * (P.n).choose P.t) := by
      exact
        sum_le_waterfillBound P.n P.t
          ((P.s - 1) * (P.n).choose P.t)
          hk'.1
          (fun i => rowSum A i) hr hB
    rw [← profileOf_row A] at hk'
    omega

/-- The evolved library. Keep `counting P` first; append both
orientation-specific waterfill arguments. -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [counting P, waterfillPrune P, rowWaterfillPrune P]
>>>>>>> REPLACE

<<<<<<< SEARCH
    if sum(comb(r, t) for r in rows) > (s - 1) * comb(n, t):
        return True                                                   # argAT
    r0, c0 = rows[0], cols[0]
=======
    if sum(comb(r, t) for r in rows) > (s - 1) * comb(n, t):
        return True                                                   # argAT
    if s >= 1:
        Bc = (t - 1) * comb(m, s)
        if sum(cols) > waterfill_bound(m, n, s, Bc):
            return True
    if t >= 1:
        Br = (s - 1) * comb(n, t)
        if sum(rows) > waterfill_bound(n, m, t, Br):
            return True
    r0, c0 = rows[0], cols[0]
>>>>>>> REPLACE

<<<<<<< SEARCH
    return False                                                      # your prunes go above
# EVOLVE-BLOCK-END
=======
    return False                                                      # your prunes go above


def waterfill_bound(m, n, s, budget):
    """Maximum total degree with n entries bounded by m and
    total choose(s) cost at most budget."""
    if s == 0:
        return m * n
    values = sorted(range(m + 1), key=lambda x: (comb(x, s), x))
    best = 0
    for q in range(n + 1):
        for d in values:
            cost = q * comb(d, s)
            if cost <= budget:
                total = q * d
                if total > best:
                    best = total
    return best
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
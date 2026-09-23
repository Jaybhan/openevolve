-- SNIPPET:dgh4
open Finset

/-- **The `D_v` family (subset-local budget).**  For any `v`-set `X` of rows (`v ≤ s`) of a
`K_{s,t}`-free matrix, `∑_{j ⊇ X} C(c_j − v, s − v) ≤ (t−1)·C(m−v, s−v)`: each of the
`C(m−v, s−v)` `s`-sets of rows containing `X` lies in at most `t−1` columns, and a
column `j ⊇ X` contains `C(c_j − v, s − v)` of them.  `v = 0` is `colBudget`,
`v = 1` is `rowLocalBudget`, `v = s − 1` is the pair cut behind the DGH inequality. -/
theorem subsetLocalBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A)
    (X : Finset (Fin P.m)) (v : ℕ) (hX : X.card = v) (hv : v ≤ P.s) :
    ∑ j ∈ univ.filter (fun j => X ⊆ support A j), (colSum A j - v).choose (P.s - v)
      ≤ (P.t - 1) * (P.m - v).choose (P.s - v) := by
  have hT : ((univ : Finset (Fin P.m)) \ X).card = P.m - v := by
    rw [Finset.card_sdiff_of_subset (Finset.subset_univ X), Finset.card_univ, Fintype.card_fin, hX]
  have key := budget_general (univ \ X) (univ.filter (fun j => X ⊆ support A j))
    (fun j => support A j \ X)
    (fun j _ => Finset.sdiff_subset_sdiff (Finset.subset_univ _) (Finset.Subset.refl _))
    (P.s - v) (P.t - 1) ?_
  · rw [hT] at key
    refine le_trans (le_of_eq ?_) key
    apply Finset.sum_congr rfl
    intro j hj
    have hXj : X ⊆ support A j := (Finset.mem_filter.mp hj).2
    rw [Finset.card_sdiff_of_subset hXj, card_support, hX]
  · intro R hR
    by_contra hcon
    have ht : P.t ≤ ((univ.filter (fun j => X ⊆ support A j)).filter
        (fun j => R ⊆ support A j \ X)).card := by omega
    obtain ⟨C, hCsub, hC⟩ := Finset.exists_subset_card_eq ht
    rw [Finset.mem_powersetCard] at hR
    have hdisj : Disjoint X R := Disjoint.mono_right hR.1 Finset.disjoint_sdiff
    apply h
    refine hasKst_of_subsets P A (X ∪ R) ?_ C hC ?_
    · rw [Finset.card_union_of_disjoint hdisj, hX, hR.2]; omega
    · intro j hj
      have hj' := Finset.mem_filter.mp (hCsub hj)
      have h1 : X ⊆ support A j := (Finset.mem_filter.mp hj'.1).2
      have h2 : R ⊆ support A j := hj'.2.trans Finset.sdiff_subset
      exact Finset.union_subset h1 h2

/-- **The integrality (rounding) step** of DGH Lemma 3.2: from `D·deg ≤ D·c + α + τ` and
`α < D` follows `(D−α)·deg ≤ (D−α)·c + τ`, because a leftover `α < D` cannot pay for a
whole extra column through `X`. -/
theorem dgh_round (D α c deg τ : ℕ) (hα : α < D) (h : D * deg ≤ D * c + α + τ) :
    (D - α) * deg ≤ (D - α) * c + τ := by
  rcases Nat.lt_or_ge c deg with hlt | hle
  · obtain ⟨e, rfl⟩ : ∃ e, deg = c + 1 + e := ⟨deg - (c + 1), by omega⟩
    obtain ⟨δ, rfl⟩ : ∃ δ, D = α + δ := ⟨D - α, by omega⟩
    rw [Nat.add_sub_cancel_left]
    nlinarith
  · calc (D - α) * deg ≤ (D - α) * c := Nat.mul_le_mul_left _ hle
      _ ≤ (D - α) * c + τ := Nat.le_add_right _ _

/-- **Double counting.**  Summing a column weight `f j` over the columns containing `X`,
then over all `v`-sets `X` of rows, counts column `j` exactly `C(c_j, v)` times. -/
theorem sum_powersetCard_filter (P : Params) (A : Mat P.m P.n) (v : ℕ) (f : Fin P.n → ℕ) :
    ∑ X ∈ (univ : Finset (Fin P.m)).powersetCard v,
        ∑ j ∈ univ.filter (fun j => X ⊆ support A j), f j
      = ∑ j, f j * (colSum A j).choose v := by
  calc ∑ X ∈ (univ : Finset (Fin P.m)).powersetCard v,
          ∑ j ∈ univ.filter (fun j => X ⊆ support A j), f j
      = ∑ X ∈ (univ : Finset (Fin P.m)).powersetCard v,
          ∑ j, if X ⊆ support A j then f j else 0 := by
        apply Finset.sum_congr rfl
        intro X _
        rw [Finset.sum_filter]
    _ = ∑ j, ∑ X ∈ (univ : Finset (Fin P.m)).powersetCard v,
          if X ⊆ support A j then f j else 0 := Finset.sum_comm
    _ = ∑ j, f j * ∑ X ∈ (univ : Finset (Fin P.m)).powersetCard v,
          if X ⊆ support A j then 1 else 0 := by
        apply Finset.sum_congr rfl
        intro j _
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro X _
        split_ifs <;> simp
    _ = ∑ j, f j * (colSum A j).choose v := by
        apply Finset.sum_congr rfl
        intro j _
        rw [← card_powersetCard_eq_sum _ _ (Finset.subset_univ _), Finset.card_powersetCard,
          card_support]

/-- `D = k − s + 1`: the number of `s`-sets through a fixed `(s−1)`-set inside a column of
sum `k`. -/
def dghD (s k : ℕ) : ℕ := k - s + 1

/-- `R = (t−1)(m−s+1)`: the pair-cut budget of one `(s−1)`-set of rows. -/
def dghR (m s t : ℕ) : ℕ := (t - 1) * (m - s + 1)

/-- `D − α` with `α = R mod D`. -/
def dghCoef (m s t k : ℕ) : ℕ := dghD s k - dghR m s t % dghD s k

/-- `c = R div D`. -/
def dghQuot (m s t k : ℕ) : ℕ := dghR m s t / dghD s k

/-- **The DGH inequality, `v = s−1`, integer-safe form.**  For a `K_{s,t}`-free matrix,
`1 ≤ s` and `s ≤ k`:
`(D−α)·∑_j C(c_j, s−1) ≤ (D−α)·c·C(m, s−1) + ∑_j (k − c_j)·C(c_j, s−1)`
(truncated subtraction, so the last sum only sees columns with `c_j < k`). -/
theorem dghBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) (hs : 1 ≤ P.s)
    (k : ℕ) (hk : P.s ≤ k) :
    dghCoef P.m P.s P.t k * ∑ j, (colSum A j).choose (P.s - 1)
      ≤ dghCoef P.m P.s P.t k * dghQuot P.m P.s P.t k * P.m.choose (P.s - 1)
        + ∑ j, (k - colSum A j) * (colSum A j).choose (P.s - 1) := by
  set v := P.s - 1 with hv
  set D := dghD P.s k with hD
  set R := dghR P.m P.s P.t with hR
  set α := R % D with hα
  set c := R / D with hc
  have hDdef : D = k - P.s + 1 := rfl
  have hRdef : R = (P.t - 1) * (P.m - P.s + 1) := rfl
  have hDpos : 0 < D := by omega
  have hαD : α < D := Nat.mod_lt _ hDpos
  have hRdm : D * c + α = R := Nat.div_add_mod R D
  have hcoef : dghCoef P.m P.s P.t k = D - α := rfl
  have hquot : dghQuot P.m P.s P.t k = c := rfl
  rw [hcoef, hquot]
  -- the local inequality for one (s-1)-set X of rows (steps 1-3)
  have perX : ∀ X ∈ (univ : Finset (Fin P.m)).powersetCard v,
      (D - α) * (univ.filter (fun j => X ⊆ support A j)).card
        ≤ (D - α) * c + ∑ j ∈ univ.filter (fun j => X ⊆ support A j), (k - colSum A j) := by
    intro X hX
    rw [Finset.mem_powersetCard] at hX
    have hXc : X.card = v := hX.2
    have loc := subsetLocalBudget P A h X v hXc (by omega)
    have h1 : P.s - v = 1 := by omega
    simp only [h1, Nat.choose_one_right] at loc
    have hmv : P.m - v ≤ P.m - P.s + 1 := by omega
    have loc' : ∑ j ∈ univ.filter (fun j => X ⊆ support A j), (colSum A j - v) ≤ R := by
      rw [hRdef]
      exact loc.trans (Nat.mul_le_mul_left _ hmv)
    have hDdeg : D * (univ.filter (fun j => X ⊆ support A j)).card
        ≤ ∑ j ∈ univ.filter (fun j => X ⊆ support A j), (colSum A j - v)
          + ∑ j ∈ univ.filter (fun j => X ⊆ support A j), (k - colSum A j) := by
      rw [← Finset.sum_add_distrib, Finset.card_eq_sum_ones, Finset.mul_sum]
      apply Finset.sum_le_sum
      intro j hj
      have hXj : X ⊆ support A j := (Finset.mem_filter.mp hj).2
      have hcj : v ≤ colSum A j := by
        rw [← hXc, ← card_support]; exact Finset.card_le_card hXj
      omega
    apply dgh_round D α c _ _ hαD
    calc D * (univ.filter (fun j => X ⊆ support A j)).card
        ≤ ∑ j ∈ univ.filter (fun j => X ⊆ support A j), (colSum A j - v)
          + ∑ j ∈ univ.filter (fun j => X ⊆ support A j), (k - colSum A j) := hDdeg
      _ ≤ R + ∑ j ∈ univ.filter (fun j => X ⊆ support A j), (k - colSum A j) :=
          Nat.add_le_add_right loc' _
      _ = D * c + α + ∑ j ∈ univ.filter (fun j => X ⊆ support A j), (k - colSum A j) := by
          rw [hRdm]
  -- aggregate over all (s-1)-sets (step 4)
  have hsum := Finset.sum_le_sum perX
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_powersetCard, Finset.card_univ,
    Fintype.card_fin, smul_eq_mul, ← Finset.mul_sum,
    sum_powersetCard_filter P A v (fun j => k - colSum A j)] at hsum
  have hdeg : ∑ X ∈ (univ : Finset (Fin P.m)).powersetCard v,
      (univ.filter (fun j => X ⊆ support A j)).card = ∑ j, (colSum A j).choose v := by
    have := sum_powersetCard_filter P A v (fun _ => 1)
    simp only [Finset.sum_const, smul_eq_mul, mul_one, one_mul] at this
    exact this
  rw [hdeg] at hsum
  calc (D - α) * ∑ j, (colSum A j).choose v
      ≤ P.m.choose v * ((D - α) * c) + ∑ j, (k - colSum A j) * (colSum A j).choose v := hsum
    _ = (D - α) * c * P.m.choose v + ∑ j, (k - colSum A j) * (colSum A j).choose v := by ring

/-- The integer-safe DGH test at one `k` on the column sums of a profile:
`(D−α)·c·C(m,s−1) + ∑_j (k−c_j)·C(c_j,s−1) < (D−α)·∑_j C(c_j,s−1)` refutes the profile. -/
def dghViolated (P : Params) (pf : Profile P.m P.n) (k : ℕ) : Bool :=
  decide (dghCoef P.m P.s P.t k * dghQuot P.m P.s P.t k * P.m.choose (P.s - 1)
      + sumFin P.n (fun j => (k - pf.col j) * (pf.col j).choose (P.s - 1))
    < dghCoef P.m P.s P.t k * sumFin P.n (fun j => (pf.col j).choose (P.s - 1)))

/-- **DGH prune, column side.**  Kill when the `v = s−1` DGH inequality fails for some
`k ∈ [s, m]` on the column sums. -/
def argDGHCol (P : Params) : Prune P where
  name := "argDGH: Davies-Gill-Horsley (v = s-1) on the column sums, all k in [s, m]"
  kill := fun pf => decide (1 ≤ P.s) &&
    (List.range (P.m + 1)).any (fun k => decide (P.s ≤ k) && dghViolated P pf k)
  sound := by
    intro A hk hv
    rw [Bool.and_eq_true, List.any_eq_true] at hk
    obtain ⟨hs, k, _, hk⟩ := hk
    rw [Bool.and_eq_true] at hk
    have hs' : 1 ≤ P.s := of_decide_eq_true hs
    have hsk : P.s ≤ k := of_decide_eq_true hk.1
    have h1 := of_decide_eq_true hk.2
    simp only [profileOf_col, sumFin_eq_sum] at h1
    have h2 := dghBudget P A hv.1 hs' k hsk
    exact absurd h2 (not_le.mpr h1)

/-- **DGH prune, row side**: the column-side prune on the transposed instance
`(n, m; t, s)`, i.e. the same inequality on the row sums with `k ∈ [t, n]`. -/
def argDGHRow (P : Params) : Prune P := (argDGHCol P.transpose).transposed

/-- **DGH prune, both orientations.** -/
def argDGH (P : Params) : Prune P := Prune.ofList P [argDGHCol P, argDGHRow P]

/-- The evolved library: the proved counting library plus the DGH prune (both sides). -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [counting P, argDGH P]

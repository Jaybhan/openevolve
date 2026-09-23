import ZarPrune
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Fintype.Card
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Finset.Sort
import Mathlib.Order.Fin.Basic
import Mathlib.Tactic.Linarith

namespace ZarPrune
open Finset

/-! ## Bridge -/

theorem sumFin_eq_sum (k : ℕ) (f : Fin k → ℕ) : sumFin k f = ∑ i, f i := by
  induction k with
  | zero => simp [sumFin]
  | succ k ih => rw [sumFin_succ, ih, Fin.sum_univ_succ]

/-! ## Supports -/

def support {m n : ℕ} (A : Mat m n) (j : Fin n) : Finset (Fin m) :=
  Finset.univ.filter (fun i => A i j = true)

theorem mem_support {m n : ℕ} (A : Mat m n) (j : Fin n) (i : Fin m) :
    i ∈ support A j ↔ A i j = true := by
  simp [support]

theorem card_support {m n : ℕ} (A : Mat m n) (j : Fin n) :
    (support A j).card = colSum A j := by
  unfold colSum support
  rw [sumFin_eq_sum, Finset.card_filter]
  apply Finset.sum_congr rfl
  intro i _
  simp [ind]

theorem hasKst_of_subsets (P : Params) (A : Mat P.m P.n)
    (R : Finset (Fin P.m)) (hR : R.card = P.s)
    (C : Finset (Fin P.n)) (hC : C.card = P.t)
    (h : ∀ j ∈ C, R ⊆ support A j) : HasKst P A := by
  refine ⟨R.orderEmbOfFin hR, C.orderEmbOfFin hC, ?_, ?_, ?_⟩
  · intro a b hab; exact (R.orderEmbOfFin hR).strictMono hab
  · intro a b hab; exact (C.orderEmbOfFin hC).strictMono hab
  · intro a b
    have hb := Finset.orderEmbOfFin_mem C hC b
    have ha := Finset.orderEmbOfFin_mem R hR a
    exact (mem_support A _ _).mp (h _ hb ha)

/-! ## The generic double count -/

theorem card_powersetCard_eq_sum {α : Type*} [DecidableEq α] (S T : Finset α) (hST : S ⊆ T) (k : ℕ) :
    (S.powersetCard k).card = ∑ R ∈ T.powersetCard k, if R ⊆ S then 1 else 0 := by
  rw [← Finset.card_filter]
  congr 1
  ext R
  simp only [Finset.mem_filter, Finset.mem_powersetCard]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨⟨h1.trans hST, h2⟩, h1⟩
  · rintro ⟨⟨_, h2⟩, h1⟩; exact ⟨h1, h2⟩

/-- Double counting pairs `(R, j)` with `R` a `k`-subset of `T` contained in `S j`.
If each `R` lies in at most `B` of the sets `S j`, then the total number of pairs
is at most `B * C(|T|, k)`. -/
theorem budget_general {α β : Type*} [DecidableEq α] [DecidableEq β]
    (T : Finset α) (J : Finset β) (S : β → Finset α) (hS : ∀ j ∈ J, S j ⊆ T) (k B : ℕ)
    (hB : ∀ R ∈ T.powersetCard k, (J.filter (fun j => R ⊆ S j)).card ≤ B) :
    ∑ j ∈ J, (S j).card.choose k ≤ B * T.card.choose k := by
  calc ∑ j ∈ J, (S j).card.choose k
      = ∑ j ∈ J, ∑ R ∈ T.powersetCard k, if R ⊆ S j then 1 else 0 := by
        apply Finset.sum_congr rfl
        intro j hj
        rw [← Finset.card_powersetCard, card_powersetCard_eq_sum _ _ (hS j hj)]
    _ = ∑ R ∈ T.powersetCard k, ∑ j ∈ J, if R ⊆ S j then 1 else 0 := Finset.sum_comm
    _ = ∑ R ∈ T.powersetCard k, (J.filter (fun j => R ⊆ S j)).card := by
        apply Finset.sum_congr rfl
        intro R _
        rw [Finset.card_filter]
    _ ≤ ∑ R ∈ T.powersetCard k, B := Finset.sum_le_sum hB
    _ = B * T.card.choose k := by
        rw [Finset.sum_const, smul_eq_mul, Finset.card_powersetCard, mul_comm]

/-! ## Argument D: row-local budget -/

theorem rowLocalBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) (i : Fin P.m)
    (hs : 1 ≤ P.s) :
    ∑ j ∈ Finset.univ.filter (fun j => A i j = true), (colSum A j - 1).choose (P.s - 1)
      ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1) := by
  have hT : (Finset.univ.erase i : Finset (Fin P.m)).card = P.m - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ, Fintype.card_fin]
  have key := budget_general (Finset.univ.erase i) (Finset.univ.filter (fun j => A i j = true))
    (fun j => (support A j).erase i) (fun j _ => Finset.erase_subset_erase i (Finset.subset_univ _))
    (P.s - 1) (P.t - 1) ?_
  · rw [hT] at key
    refine le_trans (le_of_eq ?_) key
    apply Finset.sum_congr rfl
    intro j hj
    have hij : i ∈ support A j := (mem_support A j i).mpr (Finset.mem_filter.mp hj).2
    rw [Finset.card_erase_of_mem hij, card_support]
  · intro R hR
    by_contra hcon
    have ht : P.t ≤ (Finset.filter (fun j => R ⊆ (support A j).erase i)
        (Finset.univ.filter (fun j => A i j = true))).card := by omega
    obtain ⟨C, hCsub, hC⟩ := Finset.exists_subset_card_eq ht
    rw [Finset.mem_powersetCard] at hR
    have hiR : i ∉ R := fun hi => by
      have := hR.1 hi
      simp at this
    apply h
    refine hasKst_of_subsets P A (insert i R) ?_ C hC ?_
    · rw [Finset.card_insert_of_notMem hiR, hR.2]; omega
    · intro j hj
      have hj' := hCsub hj
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj'
      rw [Finset.insert_subset_iff]
      exact ⟨(mem_support A j i).mpr hj'.1, hj'.2.trans (Finset.erase_subset i _)⟩

theorem card_rowSupport {m n : ℕ} (A : Mat m n) (i : Fin m) :
    (Finset.univ.filter (fun j => A i j = true)).card = rowSum A i := by
  unfold rowSum
  rw [sumFin_eq_sum, Finset.card_filter]
  apply Finset.sum_congr rfl
  intro j _
  simp [ind]

/-! ## Lower bounds on `∑_{j ∈ J} F (c j)` from the profile alone -/

/-- Among the `|J|` columns of `J`, at most `#{j | c j < x}` are lighter than `x`. -/
theorem card_filter_ge_sub {β : Type*} [Fintype β] [DecidableEq β]
    (J : Finset β) (c : β → ℕ) (x : ℕ) :
    J.card - (Finset.univ.filter (fun j => c j < x)).card
      ≤ (J.filter (fun j => x ≤ c j)).card := by
  have h1 := Finset.card_filter_add_card_filter_not (s := J) (fun j => x ≤ c j)
  have h2 : (J.filter (fun j => ¬ x ≤ c j)).card
      ≤ (Finset.univ.filter (fun j => c j < x)).card := by
    apply Finset.card_le_card
    intro j hj
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hj ⊢
    omega
  omega

/-- Single-threshold form. -/
theorem sum_ge_threshold {β : Type*} [Fintype β] [DecidableEq β]
    (J : Finset β) (c : β → ℕ) (F : ℕ → ℕ) (hF : Monotone F) (x : ℕ) :
    (J.card - (Finset.univ.filter (fun j => c j < x)).card) * F x ≤ ∑ j ∈ J, F (c j) := by
  calc (J.card - (Finset.univ.filter (fun j => c j < x)).card) * F x
      ≤ (J.filter (fun j => x ≤ c j)).card * F x :=
        Nat.mul_le_mul_right _ (card_filter_ge_sub J c x)
    _ = ∑ j ∈ J.filter (fun j => x ≤ c j), F x := by
        rw [Finset.sum_const, smul_eq_mul]
    _ ≤ ∑ j ∈ J.filter (fun j => x ≤ c j), F (c j) := by
        apply Finset.sum_le_sum
        intro j hj
        exact hF (Finset.mem_filter.mp hj).2
    _ ≤ ∑ j ∈ J, F (c j) :=
        Finset.sum_le_sum_of_subset_of_nonneg (Finset.filter_subset _ _) (fun _ _ _ => Nat.zero_le _)

/-- Telescoping: a monotone `F` with `F 0 = 0` is the sum of its increments. -/
theorem monotone_eq_sum_incr (F : ℕ → ℕ) (hF : Monotone F) (hF0 : F 0 = 0) (c : ℕ) :
    F c = ∑ x ∈ Finset.range (c + 1), (F x - F (x - 1)) := by
  induction c with
  | zero => simp [hF0]
  | succ c ih =>
      rw [Finset.sum_range_succ, ← ih]
      have := hF (Nat.le_succ c)
      simp only [Nat.add_sub_cancel, Nat.succ_eq_add_one] at this ⊢
      omega

theorem sum_incr_eq_sum_ite (d : ℕ → ℕ) (c m : ℕ) (hc : c ≤ m) :
    ∑ x ∈ Finset.range (c + 1), d x = ∑ x ∈ Finset.range (m + 1), if x ≤ c then d x else 0 := by
  rw [← Finset.sum_filter]
  congr 1
  ext x
  simp only [Finset.mem_filter, Finset.mem_range]
  omega

/-- Layer-cake (sum over all thresholds) form.  This equals the "sort the column
sums and take the `|J|` lightest" bound, but needs no sorting. -/
theorem sum_ge_layerCake {β : Type*} [Fintype β] [DecidableEq β]
    (J : Finset β) (c : β → ℕ) (m : ℕ) (hc : ∀ j ∈ J, c j ≤ m)
    (F : ℕ → ℕ) (hF : Monotone F) (hF0 : F 0 = 0) :
    ∑ x ∈ Finset.range (m + 1),
        (J.card - (Finset.univ.filter (fun j => c j < x)).card) * (F x - F (x - 1))
      ≤ ∑ j ∈ J, F (c j) := by
  calc ∑ x ∈ Finset.range (m + 1),
        (J.card - (Finset.univ.filter (fun j => c j < x)).card) * (F x - F (x - 1))
      ≤ ∑ x ∈ Finset.range (m + 1), (J.filter (fun j => x ≤ c j)).card * (F x - F (x - 1)) := by
        apply Finset.sum_le_sum
        intro x _
        exact Nat.mul_le_mul_right _ (card_filter_ge_sub J c x)
    _ = ∑ x ∈ Finset.range (m + 1), ∑ j ∈ J, if x ≤ c j then F x - F (x - 1) else 0 := by
        apply Finset.sum_congr rfl
        intro x _
        rw [← Finset.sum_filter, Finset.sum_const, smul_eq_mul]
    _ = ∑ j ∈ J, ∑ x ∈ Finset.range (m + 1), if x ≤ c j then F x - F (x - 1) else 0 :=
        Finset.sum_comm
    _ = ∑ j ∈ J, F (c j) := by
        apply Finset.sum_congr rfl
        intro j hj
        rw [monotone_eq_sum_incr F hF hF0 (c j), sum_incr_eq_sum_ite _ _ _ (hc j hj)]

/-! ## Transposition -/

def Params.transpose (P : Params) : Params := ⟨P.n, P.m, P.t, P.s, P.w⟩

def transpose {m n : ℕ} (A : Mat m n) : Mat n m := fun j i => A i j

theorem rowSum_transpose {m n : ℕ} (A : Mat m n) : rowSum (transpose A) = colSum A := rfl
theorem colSum_transpose {m n : ℕ} (A : Mat m n) : colSum (transpose A) = rowSum A := rfl

theorem hasKst_transpose (P : Params) (A : Mat P.m P.n) :
    HasKst P.transpose (transpose A) → HasKst P A := by
  rintro ⟨C, R, hC, hR, h⟩
  exact ⟨R, C, hR, hC, fun a b => h b a⟩

/-! ## Kill functions (computable) -/

/-- `#{j | col j < x}`, computed with `sumFin`. -/
def lighterCount {n : ℕ} (col : Fin n → ℕ) (x : ℕ) : ℕ :=
  sumFin n (fun j => if col j < x then 1 else 0)

theorem lighterCount_eq {n : ℕ} (col : Fin n → ℕ) (x : ℕ) :
    lighterCount col x = (Finset.univ.filter (fun j => col j < x)).card := by
  unfold lighterCount
  rw [sumFin_eq_sum, Finset.card_filter]

/-- `F_s x = C(x-1, s-1)` for `x ≥ 1`, and `0` at `x = 0`. -/
def Fd (s x : ℕ) : ℕ := if x = 0 then 0 else (x - 1).choose (s - 1)

theorem Fd_mono (s : ℕ) : Monotone (Fd s) := by
  intro a b hab
  unfold Fd
  split_ifs with ha hb
  · exact le_refl _
  · exact Nat.zero_le _
  · omega
  · exact Nat.choose_le_choose _ (by omega)

theorem Fd_zero (s : ℕ) : Fd s 0 = 0 := rfl

theorem Fd_of_pos (s x : ℕ) (hx : 1 ≤ x) : Fd s x = (x - 1).choose (s - 1) := by
  unfold Fd
  split_ifs with h0
  · omega
  · rfl

/-- Argument D, threshold form.  Fires when some row `i` and some threshold `x`
give `(row i - #{j | col j < x}) * C(x-1, s-1) > (t-1) * C(m-1, s-1)`. -/
def killDthr (m n s t : ℕ) (row : Fin m → ℕ) (col : Fin n → ℕ) : Bool :=
  decide (1 ≤ s) &&
  !allFin m (fun i => allFin (m + 2) (fun x =>
    decide ((row i - lighterCount col x.val) * (x.val - 1).choose (s - 1)
      ≤ (t - 1) * (m - 1).choose (s - 1))))

/-- Argument D, exact (layer-cake) form: the sum of `C(c-1, s-1)` over the
`row i` lightest columns, computed as `∑_x (row i - #{j | col j < x})⁺ · (F x - F (x-1))`. -/
def killDsum (m n s t : ℕ) (row : Fin m → ℕ) (col : Fin n → ℕ) : Bool :=
  decide (1 ≤ s) &&
  !allFin m (fun i =>
    decide (sumFin (m + 1) (fun x => (row i - lighterCount col x.val) * (Fd s x.val - Fd s (x.val - 1)))
      ≤ (t - 1) * (m - 1).choose (s - 1)))

/-! ## Soundness of the kill functions -/

theorem killDthr_sound (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    killDthr P.m P.n P.s P.t (rowSum A) (colSum A) = false := by
  unfold killDthr
  by_cases hs : 1 ≤ P.s
  · rw [decide_eq_true hs, Bool.true_and, Bool.not_eq_false']
    rw [allFin_iff]
    intro i
    rw [allFin_iff]
    intro x
    apply decide_eq_true
    rw [lighterCount_eq, ← card_rowSupport A i]
    refine le_trans ?_ (rowLocalBudget P A h i hs)
    have := sum_ge_threshold (Finset.univ.filter (fun j => A i j = true)) (colSum A)
      (fun c => (c - 1).choose (P.s - 1)) (fun a b hab => Nat.choose_le_choose _ (by omega)) x.val
    exact this
  · rw [decide_eq_false hs, Bool.false_and]

theorem killDsum_sound (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    killDsum P.m P.n P.s P.t (rowSum A) (colSum A) = false := by
  unfold killDsum
  by_cases hs : 1 ≤ P.s
  · rw [decide_eq_true hs, Bool.true_and, Bool.not_eq_false']
    rw [allFin_iff]
    intro i
    apply decide_eq_true
    rw [sumFin_eq_sum, ← Finset.sum_range (fun x => (rowSum A i - lighterCount (colSum A) x) *
      (Fd P.s x - Fd P.s (x - 1)))]
    simp only [lighterCount_eq]
    rw [← card_rowSupport A i]
    refine le_trans ?_ (rowLocalBudget P A h i hs)
    have := sum_ge_layerCake (Finset.univ.filter (fun j => A i j = true)) (colSum A) P.m
      (fun j _ => colSum_le A j) (Fd P.s) (Fd_mono P.s) (Fd_zero P.s)
    refine le_trans this (le_of_eq ?_)
    apply Finset.sum_congr rfl
    intro j hj
    have hij : A i j = true := (Finset.mem_filter.mp hj).2
    have h1 : 1 ≤ colSum A j := by
      rw [← card_support]
      exact Finset.card_pos.mpr ⟨i, (mem_support A j i).mpr hij⟩
    exact Fd_of_pos _ _ h1
  · rw [decide_eq_false hs, Bool.false_and]

/-! ## The Prune terms -/

def argD (P : Params) : Prune P where
  name := "argD (row-local budget, threshold form)"
  kill := fun pf => killDthr P.m P.n P.s P.t pf.row pf.col
  sound := by
    intro A hk hv
    have := killDthr_sound P A hv.1
    simp only [profileOf_row, profileOf_col] at hk
    rw [this] at hk
    exact Bool.false_ne_true hk

def argDT (P : Params) : Prune P where
  name := "argDT (column-local budget, threshold form)"
  kill := fun pf => killDthr P.n P.m P.t P.s pf.col pf.row
  sound := by
    intro A hk hv
    have hT : ¬ HasKst P.transpose (transpose A) := fun h' => hv.1 (hasKst_transpose P A h')
    have := killDthr_sound P.transpose (transpose A) hT
    change killDthr P.n P.m P.t P.s (colSum A) (rowSum A) = false at this
    simp only [profileOf_row, profileOf_col] at hk
    rw [this] at hk
    exact Bool.false_ne_true hk

def argDsum (P : Params) : Prune P where
  name := "argDsum (row-local budget, exact/layer-cake form)"
  kill := fun pf => killDsum P.m P.n P.s P.t pf.row pf.col
  sound := by
    intro A hk hv
    have := killDsum_sound P A hv.1
    simp only [profileOf_row, profileOf_col] at hk
    rw [this] at hk
    exact Bool.false_ne_true hk

def argDsumT (P : Params) : Prune P where
  name := "argDsumT (column-local budget, exact/layer-cake form)"
  kill := fun pf => killDsum P.n P.m P.t P.s pf.col pf.row
  sound := by
    intro A hk hv
    have hT : ¬ HasKst P.transpose (transpose A) := fun h' => hv.1 (hasKst_transpose P A h')
    have := killDsum_sound P.transpose (transpose A) hT
    change killDsum P.n P.m P.t P.s (colSum A) (rowSum A) = false at this
    simp only [profileOf_row, profileOf_col] at hk
    rw [this] at hk
    exact Bool.false_ne_true hk

def countingD (P : Params) : Prune P :=
  Prune.ofList P [argD P, argDT P, argDsum P, argDsumT P]

/-! ## Bonus: Argument A (column budget) from the same double count -/

theorem colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s := by
  have key := budget_general (Finset.univ : Finset (Fin P.m)) (Finset.univ : Finset (Fin P.n))
    (support A) (fun j _ => Finset.subset_univ _) P.s (P.t - 1) ?_
  · rw [Finset.card_univ, Fintype.card_fin] at key
    refine le_trans (le_of_eq ?_) key
    apply Finset.sum_congr rfl
    intro j _
    rw [card_support]
  · intro R hR
    by_contra hcon
    have ht : P.t ≤ (Finset.univ.filter (fun j => R ⊆ support A j)).card := by omega
    obtain ⟨C, hCsub, hC⟩ := Finset.exists_subset_card_eq ht
    rw [Finset.mem_powersetCard] at hR
    exact h (hasKst_of_subsets P A R hR.2 C hC
      (fun j hj => (Finset.mem_filter.mp (hCsub hj)).2))

theorem rowBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ i, (rowSum A i).choose P.t ≤ (P.s - 1) * (P.n).choose P.t := by
  have hT : ¬ HasKst P.transpose (transpose A) := fun h' => h (hasKst_transpose P A h')
  have := colBudget P.transpose (transpose A) hT
  exact this

/-! ## Tests -/

-- z(9,9;2,2): 9x9, K_{2,2}-free.  Row profile 6,6,6,6,6,5,5,5,5 (total 49).
abbrev P99 : Params := ⟨9, 9, 2, 2, 49⟩
def pf99 : Profile 9 9 :=
  { row := fun i => [6,6,6,6,6,5,5,5,5].getD i.val 0,
    col := fun j => [6,6,6,6,6,5,5,5,5].getD j.val 0 }
#eval (argD P99).kill pf99
#eval (argDT P99).kill pf99
#eval (argDsum P99).kill pf99
#eval (argDsumT P99).kill pf99
-- Row with 9 ones, columns all 5: 9 * C(4,1) = 36 > (2-1)*C(8,1) = 8 → kill.
def pf99b : Profile 9 9 :=
  { row := fun i => [9,5,5,5,5,5,5,5,5].getD i.val 0,
    col := fun j => [5,5,5,5,5,5,5,5,5].getD j.val 0 }
#eval (argD P99).kill pf99b
#eval (argDsum P99).kill pf99b
#eval (argDT P99).kill pf99b
-- Row with 4 ones, columns 4,4,4,4,4,4,4,4,4 : 4 * C(3,1) = 12 > 8 → kill.
def pf99c : Profile 9 9 :=
  { row := fun i => [4,4,4,4,4,4,4,4,4].getD i.val 0,
    col := fun j => [4,4,4,4,4,4,4,4,4].getD j.val 0 }
#eval (argD P99).kill pf99c
#eval (argDsum P99).kill pf99c
-- A case where only the exact form fires: row 4, columns 2,2,2,3,5,5,5,5,5, s=2,t=2,m=9: bound 8
-- lightest 4 columns: 2,2,2,3 → 1+1+1+2 = 5 ≤ 8 no kill.
-- columns 3,3,3,3,4,5,5,5,5, row 5: lightest 5: 2+2+2+2+3 = 11 > 8 kill (exact);
-- thresholds: x=4: (5-4)*C(3,1)=3; x=3: (5-0)*C(2,1)=10 > 8 → threshold also kills.
def pf99d : Profile 9 9 :=
  { row := fun i => [5,3,3,3,3,3,3,3,3].getD i.val 0,
    col := fun j => [3,3,3,3,4,5,5,5,5].getD j.val 0 }
#eval (argD P99).kill pf99d
#eval (argDsum P99).kill pf99d
-- columns 1,1,1,1,5,5,5,5,5 with row 6, s=2: lightest 6: 0*4 + 4*2 = 8 ≤ 8 → no kill (exact).
-- row 7: lightest 7: 0*4 + 3*4 = 12 > 8 → kill exact. threshold x=5: (7-4)*C(4,1)=12 > 8 kill.
def pf99e : Profile 9 9 :=
  { row := fun i => [6,3,3,3,3,3,3,3,3].getD i.val 0,
    col := fun j => [1,1,1,1,5,5,5,5,5].getD j.val 0 }
#eval (argD P99).kill pf99e
#eval (argDsum P99).kill pf99e
-- s = 3, t = 3, m = n = 9: bound (3-1)*C(8,2) = 56.
abbrev P993 : Params := ⟨9, 9, 3, 3, 60⟩
def pf993 : Profile 9 9 :=
  { row := fun i => [9,7,7,7,7,7,7,7,7].getD i.val 0,
    col := fun j => [7,7,7,7,7,7,7,7,7].getD j.val 0 }
-- 9 * C(6,2) = 135 > 56 → kill.
#eval (argD P993).kill pf993
#eval (argDsum P993).kill pf993
-- s = 1 sanity: (row i - k_x) * C(x-1,0) = row i - k_x ≤ (t-1)*1.
abbrev P1 : Params := ⟨3, 3, 1, 2, 5⟩
def pf1 : Profile 3 3 := { row := fun _ => 2, col := fun _ => 2 }
#eval (argD P1).kill pf1   -- 2 > 1 → kill (a row with 2 ones in a K_{1,2}-free matrix is impossible)
#eval (argDsum P1).kill pf1
#eval (countingD P99).kill pf99

#print axioms sumFin_eq_sum
#print axioms card_support
#print axioms hasKst_of_subsets
#print axioms budget_general
#print axioms rowLocalBudget
#print axioms sum_ge_threshold
#print axioms sum_ge_layerCake
#print axioms killDthr_sound
#print axioms killDsum_sound
#print axioms argD
#print axioms argDT
#print axioms argDsum
#print axioms argDsumT
#print axioms countingD
#print axioms colBudget
#print axioms rowBudget
end ZarPrune

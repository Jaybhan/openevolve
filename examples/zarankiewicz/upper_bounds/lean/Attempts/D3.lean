import ZarPrune
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Fintype.Card
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Finset.Sort
import Mathlib.Order.Fin.Basic
import Mathlib.Tactic.Linarith

namespace ZarPrune

open Finset

theorem sumFin_eq_sum (k : ℕ) (f : Fin k → ℕ) : sumFin k f = ∑ i, f i := by
  induction k with
  | zero => simp [sumFin]
  | succ k ih => rw [sumFin_succ, Fin.sum_univ_succ, ih]

def support {m n : ℕ} (A : Mat m n) (j : Fin n) : Finset (Fin m) :=
  Finset.univ.filter (fun i => A i j = true)

theorem card_support {m n : ℕ} (A : Mat m n) (j : Fin n) : (support A j).card = colSum A j := by
  unfold support colSum
  rw [sumFin_eq_sum, Finset.card_filter]
  rfl

def rowSupport {m n : ℕ} (A : Mat m n) (i : Fin m) : Finset (Fin n) :=
  Finset.univ.filter (fun j => A i j = true)

theorem card_rowSupport {m n : ℕ} (A : Mat m n) (i : Fin m) : (rowSupport A i).card = rowSum A i := by
  unfold rowSupport rowSum
  rw [sumFin_eq_sum, Finset.card_filter]
  rfl

theorem mem_support {m n : ℕ} (A : Mat m n) (j : Fin n) (i : Fin m) :
    i ∈ support A j ↔ A i j = true := by
  simp [support]

theorem mem_rowSupport {m n : ℕ} (A : Mat m n) (i : Fin m) (j : Fin n) :
    j ∈ rowSupport A i ↔ A i j = true := by
  simp [rowSupport]

theorem incr_of_strictMono {k N : ℕ} (f : Fin k → Fin N) (h : StrictMono f) : Incr f :=
  fun _a _b hab => h hab

theorem hasKst_of_subsets (P : Params) (A : Mat P.m P.n) (R : Finset (Fin P.m)) (hR : R.card = P.s)
    (C : Finset (Fin P.n)) (hC : C.card = P.t) (h : ∀ j ∈ C, R ⊆ support A j) : HasKst P A := by
  refine ⟨R.orderEmbOfFin hR, C.orderEmbOfFin hC, ?_, ?_, ?_⟩
  · exact incr_of_strictMono _ (R.orderEmbOfFin hR).strictMono
  · exact incr_of_strictMono _ (C.orderEmbOfFin hC).strictMono
  · intro a b
    have hb : C.orderEmbOfFin hC b ∈ C := Finset.orderEmbOfFin_mem C hC b
    have ha : R.orderEmbOfFin hR a ∈ R := Finset.orderEmbOfFin_mem R hR a
    exact (mem_support A (C.orderEmbOfFin hC b) (R.orderEmbOfFin hR a)).mp
      (h (C.orderEmbOfFin hC b) hb ha)


/-! ## Counting `k`-subsets of `X` inside a fixed ambient finset `U` -/

theorem filter_powersetCard_subset {α : Type*} [DecidableEq α] (k : ℕ) (U X : Finset α)
    (hX : X ⊆ U) : (U.powersetCard k).filter (fun R => R ⊆ X) = X.powersetCard k := by
  ext R
  simp only [mem_filter, mem_powersetCard]
  constructor
  · rintro ⟨⟨_, hc⟩, hRX⟩; exact ⟨hRX, hc⟩
  · rintro ⟨hRX, hc⟩; exact ⟨⟨hRX.trans hX, hc⟩, hRX⟩

theorem choose_eq_sum_powersetCard {α : Type*} [DecidableEq α] (k : ℕ) (U X : Finset α)
    (hX : X ⊆ U) : X.card.choose k = ∑ R ∈ U.powersetCard k, if R ⊆ X then 1 else 0 := by
  rw [← card_powersetCard, ← filter_powersetCard_subset k U X hX, card_filter]

/-! ## Argument D: the row-local budget -/

/-- For a `K_{s,t}`-free matrix and any row `i`, summing `C(c_j - 1, s - 1)` over the
columns `j` that contain a one in row `i` is at most `(t-1)·C(m-1, s-1)`.  Double counting
over pairs `(R, j)` with `R` an `(s-1)`-subset of the rows other than `i` contained in
column `j`: if some `R` sat in `t` such columns, `R ∪ {i}` would be an `s`-set of rows
common to `t` columns, i.e. a `K_{s,t}`. -/
theorem rowLocalBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) (i : Fin P.m)
    (hs : 1 ≤ P.s) :
    ∑ j ∈ rowSupport A i, (colSum A j - 1).choose (P.s - 1)
      ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1) := by
  set U : Finset (Fin P.m) := (univ : Finset (Fin P.m)).erase i with hU
  have hUcard : U.card = P.m - 1 := by
    rw [hU, card_erase_of_mem (mem_univ i), card_univ, Fintype.card_fin]
  have hterm : ∀ j ∈ rowSupport A i, (colSum A j - 1).choose (P.s - 1)
      = ∑ R ∈ U.powersetCard (P.s - 1), if R ⊆ (support A j).erase i then 1 else 0 := by
    intro j hj
    have hij : i ∈ support A j := (mem_support A j i).mpr ((mem_rowSupport A i j).mp hj)
    rw [← card_support, ← card_erase_of_mem hij]
    apply choose_eq_sum_powersetCard
    exact erase_subset_erase i (subset_univ _)
  rw [sum_congr rfl hterm, sum_comm]
  have hinner : ∀ R ∈ U.powersetCard (P.s - 1),
      (∑ j ∈ rowSupport A i, if R ⊆ (support A j).erase i then 1 else 0) ≤ P.t - 1 := by
    intro R hR
    rw [mem_powersetCard] at hR
    rw [← card_filter]
    by_contra hcon
    push Not at hcon
    obtain ⟨C, hCsub, hCcard⟩ := exists_subset_card_eq
      (s := (rowSupport A i).filter (fun j => R ⊆ (support A j).erase i)) (n := P.t) (by omega)
    apply h
    have hiR : i ∉ R := by
      intro hiR
      have := hR.1 hiR
      simp [hU] at this
    refine hasKst_of_subsets P A (insert i R) ?_ C hCcard ?_
    · rw [card_insert_of_notMem hiR, hR.2]; omega
    · intro j hj
      have hj' := hCsub hj
      rw [mem_filter] at hj'
      intro x hx
      rw [mem_insert] at hx
      rcases hx with rfl | hx
      · exact (mem_support A j x).mpr ((mem_rowSupport A x j).mp hj'.1)
      · exact mem_of_mem_erase (hj'.2 hx)
  calc _ ≤ ∑ _R ∈ U.powersetCard (P.s - 1), (P.t - 1) := sum_le_sum hinner
    _ = _ := by rw [sum_const, card_powersetCard, hUcard, smul_eq_mul, mul_comm]

/-! ## The profile-level bound (layer-cake / all-thresholds form) -/

/-- `fD s c = C(c-1, s-1)`: the contribution of a column of sum `c`. -/
def fD (s c : ℕ) : ℕ := (c - 1).choose (s - 1)

/-- The increment `fD s x - fD s (x-1)`. -/
def gD (s x : ℕ) : ℕ := fD s x - fD s (x - 1)

/-- Number of columns of the profile whose sum is `< y`. -/
def cntLt {m n : ℕ} (pf : Profile m n) (y : ℕ) : ℕ :=
  (univ.filter (fun j => pf.col j < y)).card

/-- Lower bound on `∑_{j in row} fD s (col j)` for a row of sum `r`, computed from the
profile alone: at most `cntLt pf x` of the row's `r` columns have sum `< x`, so at least
`r - cntLt pf x` of them collect the increment `gD s x`. -/
def boundD (P : Params) (pf : Profile P.m P.n) (r : ℕ) : ℕ :=
  r * fD P.s 0 + ∑ x ∈ range P.m, (r - cntLt pf (x + 1)) * gD P.s (x + 1)

theorem fD_mono (s : ℕ) {a b : ℕ} (h : a ≤ b) : fD s a ≤ fD s b :=
  Nat.choose_le_choose _ (by omega)

theorem fD_layer_cake (s c : ℕ) : fD s c = fD s 0 + ∑ x ∈ range c, gD s (x + 1) := by
  induction c with
  | zero => simp
  | succ c ih =>
    rw [sum_range_succ, ← add_assoc, ← ih]
    have := fD_mono s (Nat.le_add_right c 1)
    unfold gD
    rw [Nat.add_sub_cancel]
    omega

theorem sum_range_eq_sum_ite {M : Type*} [AddCommMonoid M] (f : ℕ → M) {c m : ℕ} (hc : c ≤ m) :
    ∑ x ∈ range c, f x = ∑ x ∈ range m, if x < c then f x else 0 := by
  rw [← sum_filter]
  congr 1
  ext x
  simp only [mem_filter, mem_range]
  omega

theorem fD_layer_cake_le (s c m : ℕ) (hc : c ≤ m) :
    fD s c = fD s 0 + ∑ x ∈ range m, if x < c then gD s (x + 1) else 0 := by
  rw [fD_layer_cake, sum_range_eq_sum_ite _ hc]

theorem sum_fD_eq (P : Params) (A : Mat P.m P.n) (i : Fin P.m) :
    ∑ j ∈ rowSupport A i, fD P.s (colSum A j)
      = (rowSupport A i).card * fD P.s 0
        + ∑ x ∈ range P.m,
            ((rowSupport A i).filter (fun j => x < colSum A j)).card * gD P.s (x + 1) := by
  have h1 : ∀ j ∈ rowSupport A i, fD P.s (colSum A j)
      = fD P.s 0 + ∑ x ∈ range P.m, if x < colSum A j then gD P.s (x + 1) else 0 :=
    fun j _ => fD_layer_cake_le P.s _ P.m (colSum_le A j)
  rw [sum_congr rfl h1, sum_add_distrib, sum_const, smul_eq_mul, sum_comm]
  congr 1
  apply sum_congr rfl
  intro x _
  rw [← sum_filter, sum_const, smul_eq_mul]

theorem card_filter_ge (P : Params) (A : Mat P.m P.n) (i : Fin P.m) (x : ℕ) :
    rowSum A i - cntLt (profileOf A) (x + 1)
      ≤ ((rowSupport A i).filter (fun j => x < colSum A j)).card := by
  have h1 := card_filter_add_card_filter_not (s := rowSupport A i) (fun j => x < colSum A j)
  have h2 : ((rowSupport A i).filter (fun j => ¬ x < colSum A j)).card
      ≤ cntLt (profileOf A) (x + 1) := by
    have e : cntLt (profileOf A) (x + 1)
        = ((univ : Finset (Fin P.n)).filter (fun j => colSum A j < x + 1)).card := rfl
    rw [e]
    apply card_le_card
    intro j hj
    rw [mem_filter] at hj ⊢
    exact ⟨mem_univ _, by omega⟩
  rw [card_rowSupport] at h1
  omega

theorem boundD_le (P : Params) (A : Mat P.m P.n) (i : Fin P.m) :
    boundD P (profileOf A) (rowSum A i) ≤ ∑ j ∈ rowSupport A i, fD P.s (colSum A j) := by
  rw [sum_fD_eq, card_rowSupport]
  unfold boundD
  apply Nat.add_le_add_left
  apply sum_le_sum
  intro x _
  exact Nat.mul_le_mul_right _ (card_filter_ge P A i x)

/-- Argument D, row version: some row `i` with sum `r = pf.row i` would force
`∑_{j ∋ i} C(c_j - 1, s - 1)` above the row-local budget `(t-1)·C(m-1, s-1)`. -/
def argD (P : Params) : Prune P where
  name := "argD (row-local budget, all thresholds)"
  kill := fun pf => decide (1 ≤ P.s) &&
    !allFin P.m (fun i =>
      decide (boundD P pf (pf.row i) ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1)))
  sound := by
    intro A h hv
    rw [Bool.and_eq_true] at h
    have hs : 1 ≤ P.s := of_decide_eq_true h.1
    apply not_allFin_elim _ h.2
    intro i
    apply decide_eq_true
    calc boundD P (profileOf A) ((profileOf A).row i)
        ≤ ∑ j ∈ rowSupport A i, fD P.s (colSum A j) := boundD_le P A i
      _ ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1) := rowLocalBudget P A hv.1 i hs

/-! ## Transposition, and the column version -/

def Params.transpose (P : Params) : Params := ⟨P.n, P.m, P.t, P.s, P.w⟩

def transposeMat {m n : ℕ} (A : Mat m n) : Mat n m := fun j i => A i j

def Profile.swap {m n : ℕ} (pf : Profile m n) : Profile n m := ⟨pf.col, pf.row⟩

theorem hasKst_of_transpose (P : Params) (A : Mat P.m P.n)
    (h : HasKst P.transpose (transposeMat A)) : HasKst P A := by
  obtain ⟨C, R, hC, hR, h⟩ := h
  exact ⟨R, C, hR, hC, fun a b => h b a⟩

theorem weight_transposeMat {m n : ℕ} (A : Mat m n) : weight (transposeMat A) = weight A := by
  rw [weight_eq_sum_colSum A]; rfl

theorem profileOf_transposeMat {m n : ℕ} (A : Mat m n) :
    profileOf (transposeMat A) = (profileOf A).swap := rfl

/-- Argument D, column version: `argD` applied to the transposed instance. -/
def argDT (P : Params) : Prune P where
  name := "argDT (column-local budget, all thresholds)"
  kill := fun pf => (argD P.transpose).kill pf.swap
  sound := by
    intro A h hv
    have hk : ¬ Valid P.transpose (transposeMat A) := (argD P.transpose).sound (transposeMat A) h
    apply hk
    refine ⟨fun hk => hv.1 (hasKst_of_transpose P A hk), ?_⟩
    show P.w ≤ weight (transposeMat A)
    rw [weight_transposeMat]; exact hv.2

/-- Both directions of Argument D folded together. -/
def countingD (P : Params) : Prune P := Prune.ofList P [argD P, argDT P]

/-! ## Tests -/

/-- 9×9, K_{2,2}-free (s = t = 2). -/
def P99 : Params := ⟨9, 9, 2, 2, 50⟩

def pf1 : Profile 9 9 :=
  { row := fun i => [6,6,6,6,6,5,5,5,5].getD i.val 0,
    col := fun j => [6,6,6,6,6,5,5,5,5].getD j.val 0 }

def pf2 : Profile 9 9 :=
  { row := fun i => [7,5,5,5,5,5,5,5,5].getD i.val 0,
    col := fun j => [5,5,5,5,5,5,5,5,7].getD j.val 0 }

def pf3 : Profile 9 9 :=
  { row := fun i => [4,4,4,4,4,4,4,4,4].getD i.val 0,
    col := fun j => [4,4,4,4,4,4,4,4,4].getD j.val 0 }

#eval boundD P99 pf1 6
#eval (P99.t - 1) * (P99.m - 1).choose (P99.s - 1)
#eval (argD P99).kill pf1
#eval (argDT P99).kill pf1
#eval (countingD P99).kill pf1
#eval boundD P99 pf2 7
#eval (argD P99).kill pf2
#eval (argDT P99).kill pf2
#eval (argD P99).kill pf3
#eval (countingD P99).kill pf3

/-- Real cases from `cache/case_table_m10_n11_s3_t3_w65_pure.json`; expected
(argD, argDT) = (false,false), (false,true), (true,false) -- matching the Python
reference `kill_row_argument_d` / `kill_col_argument_d`. -/
def P1011 : Params := ⟨10, 11, 3, 3, 65⟩
def c1 : Profile 10 11 :=
  { row := fun i => [7,7,7,7,7,6,6,6,6,6].getD i.val 0,
    col := fun j => [6,6,6,6,6,6,6,6,6,6,5].getD j.val 0 }
def c2 : Profile 10 11 :=
  { row := fun i => [7,7,7,7,7,6,6,6,6,6].getD i.val 0,
    col := fun j => [8,6,6,6,6,6,6,6,5,5,5].getD j.val 0 }
def c3 : Profile 10 11 :=
  { row := fun i => [8,7,7,7,6,6,6,6,6,6].getD i.val 0,
    col := fun j => [6,6,6,6,6,6,6,6,6,6,5].getD j.val 0 }
#eval ((argD P1011).kill c1, (argDT P1011).kill c1)
#eval ((argD P1011).kill c2, (argDT P1011).kill c2)
#eval ((argD P1011).kill c3, (argDT P1011).kill c3)
example : (argD P1011).kill c3 = true := by decide
example : (argD P1011).kill c1 = false := by decide

#print axioms sumFin_eq_sum
#print axioms card_support
#print axioms card_rowSupport
#print axioms hasKst_of_subsets
#print axioms rowLocalBudget
#print axioms fD_layer_cake
#print axioms sum_fD_eq
#print axioms boundD_le
#print axioms argD
#print axioms argDT
#print axioms countingD
#print axioms hasKst_of_transpose
#print axioms mem_support
#print axioms mem_rowSupport
#print axioms incr_of_strictMono
#print axioms filter_powersetCard_subset
#print axioms choose_eq_sum_powersetCard
#print axioms fD_mono
#print axioms sum_range_eq_sum_ite
#print axioms fD_layer_cake_le
#print axioms card_filter_ge
#print axioms profileOf_transposeMat
#print axioms weight_transposeMat

end ZarPrune

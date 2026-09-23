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

/-!
# Attempt A3 — Argument A (column budget) and its transpose

Double counting pairs `(R, j)` with `R` an `s`-subset of the support of column `j`.
-/

namespace ZarPrune

open Finset

/-! ## Bridge to `Finset.sum` -/

theorem sumFin_eq_sum (k : ℕ) (f : Fin k → ℕ) : sumFin k f = ∑ i, f i := by
  induction k with
  | zero => simp [sumFin]
  | succ k ih => rw [sumFin_succ, Fin.sum_univ_succ, ih]

/-! ## Supports -/

def support {m n : ℕ} (A : Mat m n) (j : Fin n) : Finset (Fin m) :=
  Finset.univ.filter (fun i => A i j = true)

@[simp] theorem mem_support {m n : ℕ} (A : Mat m n) (j : Fin n) (i : Fin m) :
    i ∈ support A j ↔ A i j = true := by
  simp [support]

theorem card_support {m n : ℕ} (A : Mat m n) (j : Fin n) : (support A j).card = colSum A j := by
  unfold support colSum
  rw [sumFin_eq_sum, Finset.card_filter]
  rfl

/-! ## From subsets of supports to a `K_{s,t}` -/

theorem incr_orderEmbOfFin {N k : ℕ} (S : Finset (Fin N)) (h : S.card = k) :
    Incr (fun a => S.orderEmbOfFin h a) := by
  intro a b hab
  exact (S.orderEmbOfFin h).strictMono hab

theorem hasKst_of_subsets (P : Params) (A : Mat P.m P.n)
    (R : Finset (Fin P.m)) (hR : R.card = P.s)
    (C : Finset (Fin P.n)) (hC : C.card = P.t)
    (h : ∀ j ∈ C, R ⊆ support A j) : HasKst P A := by
  refine ⟨fun a => R.orderEmbOfFin hR a, fun b => C.orderEmbOfFin hC b,
    incr_orderEmbOfFin R hR, incr_orderEmbOfFin C hC, ?_⟩
  intro a b
  have hb : C.orderEmbOfFin hC b ∈ C := Finset.orderEmbOfFin_mem C hC b
  have ha : R.orderEmbOfFin hR a ∈ R := Finset.orderEmbOfFin_mem R hR a
  exact (mem_support A _ _).mp (h _ hb ha)

/-! ## Argument A: the column budget -/

/-- `C(|S|, s)` counts the `s`-subsets of `univ` contained in `S`. -/
theorem choose_card_eq_sum_powersetCard {N : ℕ} (S : Finset (Fin N)) (s : ℕ) :
    S.card.choose s = ∑ R ∈ (Finset.univ : Finset (Fin N)).powersetCard s,
      if R ⊆ S then 1 else 0 := by
  rw [← Finset.card_powersetCard, ← Finset.card_filter]
  congr 1
  ext R
  simp only [Finset.mem_powersetCard, Finset.mem_filter, Finset.subset_univ, true_and]
  tauto

/-- For a fixed `s`-subset `R` of rows, fewer than `t` columns contain it. -/
theorem card_cols_containing_le (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A)
    (R : Finset (Fin P.m)) (hR : R.card = P.s) :
    ((Finset.univ : Finset (Fin P.n)).filter (fun j => R ⊆ support A j)).card ≤ P.t - 1 := by
  by_contra hlt
  have ht : P.t ≤ ((Finset.univ : Finset (Fin P.n)).filter (fun j => R ⊆ support A j)).card := by
    omega
  obtain ⟨C, hCsub, hC⟩ := Finset.exists_subset_card_eq ht
  apply h
  refine hasKst_of_subsets P A R hR C hC ?_
  intro j hj
  exact (Finset.mem_filter.mp (hCsub hj)).2

theorem colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s := by
  calc ∑ j, (colSum A j).choose P.s
      = ∑ j, ∑ R ∈ (Finset.univ : Finset (Fin P.m)).powersetCard P.s,
          (if R ⊆ support A j then 1 else 0) := by
        apply Finset.sum_congr rfl
        intro j _
        rw [← card_support, choose_card_eq_sum_powersetCard]
    _ = ∑ R ∈ (Finset.univ : Finset (Fin P.m)).powersetCard P.s,
          ∑ j, (if R ⊆ support A j then 1 else 0) := Finset.sum_comm
    _ = ∑ R ∈ (Finset.univ : Finset (Fin P.m)).powersetCard P.s,
          ((Finset.univ : Finset (Fin P.n)).filter (fun j => R ⊆ support A j)).card := by
        apply Finset.sum_congr rfl
        intro R _
        rw [Finset.card_filter]
    _ ≤ ∑ _R ∈ (Finset.univ : Finset (Fin P.m)).powersetCard P.s, (P.t - 1) := by
        apply Finset.sum_le_sum
        intro R hR
        exact card_cols_containing_le P A h R (Finset.mem_powersetCard.mp hR).2
    _ = (P.t - 1) * (P.m).choose P.s := by
        rw [Finset.sum_const, Finset.card_powersetCard, Finset.card_univ, Fintype.card_fin,
          smul_eq_mul, Nat.mul_comm]

/-! ## Transpose -/

def transpose {m n : ℕ} (A : Mat m n) : Mat n m := fun j i => A i j

def Params.transpose (P : Params) : Params := ⟨P.n, P.m, P.t, P.s, P.w⟩

@[simp] theorem transpose_transpose {m n : ℕ} (A : Mat m n) : transpose (transpose A) = A := rfl

theorem colSum_transpose {m n : ℕ} (A : Mat m n) (i : Fin m) :
    colSum (transpose A) i = rowSum A i := rfl

theorem rowSum_transpose {m n : ℕ} (A : Mat m n) (j : Fin n) :
    rowSum (transpose A) j = colSum A j := rfl

theorem hasKst_transpose_of (P : Params) (A : Mat P.m P.n) (h : HasKst P A) :
    HasKst P.transpose (transpose A) := by
  obtain ⟨R, C, hR, hC, hRC⟩ := h
  exact ⟨C, R, hC, hR, fun b a => hRC a b⟩

theorem hasKst_transpose (P : Params) (A : Mat P.m P.n) :
    HasKst P A ↔ HasKst P.transpose (transpose A) :=
  ⟨hasKst_transpose_of P A, fun h => hasKst_transpose_of P.transpose (transpose A) h⟩

theorem weight_transpose {m n : ℕ} (A : Mat m n) : weight (transpose A) = weight A := by
  rw [weight_eq_sum_colSum]
  rfl

theorem rowBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ i, (rowSum A i).choose P.t ≤ (P.s - 1) * (P.n).choose P.t := by
  have h' : ¬ HasKst P.transpose (transpose A) := fun hT => h ((hasKst_transpose P A).mpr hT)
  exact colBudget P.transpose (transpose A) h'

/-! ## Prune terms -/

/-- Argument A on the column side: kill when `Σ_j C(c_j, s) > (t-1)·C(m, s)`. -/
def argA (P : Params) : Prune P where
  name := "argA: Σ_j C(c_j,s) ≤ (t-1)C(m,s)"
  kill := fun pf => decide ((P.t - 1) * (P.m).choose P.s < sumFin P.n (fun j => (pf.col j).choose P.s))
  sound := by
    intro A hk hv
    have h1 : (P.t - 1) * (P.m).choose P.s < sumFin P.n (fun j => (colSum A j).choose P.s) :=
      of_decide_eq_true hk
    rw [sumFin_eq_sum] at h1
    have h2 := colBudget P A hv.1
    omega

/-- Argument A on the row side: kill when `Σ_i C(r_i, t) > (s-1)·C(n, t)`. -/
def argAT (P : Params) : Prune P where
  name := "argAT: Σ_i C(r_i,t) ≤ (s-1)C(n,t)"
  kill := fun pf => decide ((P.s - 1) * (P.n).choose P.t < sumFin P.m (fun i => (pf.row i).choose P.t))
  sound := by
    intro A hk hv
    have h1 : (P.s - 1) * (P.n).choose P.t < sumFin P.m (fun i => (rowSum A i).choose P.t) :=
      of_decide_eq_true hk
    rw [sumFin_eq_sum] at h1
    have h2 := rowBudget P A hv.1
    omega

/-- Both sides of Argument A folded together. -/
def argAboth (P : Params) : Prune P := Prune.ofList P [argA P, argAT P]

/-! ## Deletion lemma (bonus) -/

def deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) : Mat m n :=
  fun i k => A i (j.succAbove k)

theorem rowSum_deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) (i : Fin m) :
    rowSum (deleteCol A j) i + ind (A i j) = rowSum A i := by
  unfold rowSum deleteCol
  rw [sumFin_eq_sum, sumFin_eq_sum, Fin.sum_univ_succAbove (fun k => ind (A i k)) j]
  omega

theorem weight_deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) :
    weight (deleteCol A j) + colSum A j = weight A := by
  unfold weight colSum
  rw [sumFin_eq_sum, sumFin_eq_sum, sumFin_eq_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i _
  exact rowSum_deleteCol A j i

theorem incr_succAbove_comp {k n : ℕ} (j : Fin (n + 1)) (C : Fin k → Fin n) (hC : Incr C) :
    Incr (fun b => j.succAbove (C b)) := by
  intro a b hab
  exact Fin.strictMono_succAbove j (hC a b hab)

theorem not_hasKst_deleteCol {m n s t w w' : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1))
    (h : ¬ HasKst ⟨m, n + 1, s, t, w⟩ A) : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteCol A j) := by
  intro hK
  obtain ⟨R, C, hR, hC, hRC⟩ := hK
  apply h
  exact ⟨R, fun b => j.succAbove (C b), hR, incr_succAbove_comp j C hC, hRC⟩

/-! ## Computability checks -/

section Tests

def P99 : Params := ⟨9, 9, 2, 2, 51⟩
def pf1 : Profile 9 9 :=
  { row := fun i => [6,6,6,6,6,5,5,5,5].getD i.val 0,
    col := fun j => [6,6,6,6,6,5,5,5,5].getD j.val 0 }
def pf2 : Profile 9 9 :=
  { row := fun i => [7,7,7,7,7,7,7,7,7].getD i.val 0,
    col := fun j => [7,7,7,7,7,7,7,7,7].getD j.val 0 }
def pf3 : Profile 9 9 :=
  { row := fun i => [4,4,4,4,4,4,4,4,4].getD i.val 0,
    col := fun j => [9,9,9,9,0,0,0,0,0].getD j.val 0 }

-- Σ C(c,2): 5*15 + 4*10 = 115 > (2-1)*C(9,2) = 36 → kill
#eval (argA P99).kill pf1
#eval (argAT P99).kill pf1
#eval (argA P99).kill pf2
-- Σ C(c,2) = 4*36 = 144 > 36 → kill; rows: 9*6 = 54 > 36 → kill
#eval (argA P99).kill pf3
#eval (argAT P99).kill pf3
-- small profile: 9 * C(3,2) = 27 ≤ 36 → survive
#eval (argA P99).kill { row := fun _ => 3, col := fun _ => 3 }
#eval (argAboth P99).kill { row := fun _ => 3, col := fun _ => 3 }
-- K_{3,3} instance: (3-1)*C(9,3) = 168; Σ C(6,3) = 9*20 = 180 → kill
#eval (argA ⟨9, 9, 3, 3, 0⟩).kill pf2
#eval (argA ⟨9, 9, 3, 3, 0⟩).kill { row := fun _ => 5, col := fun _ => 5 }

end Tests

#print axioms sumFin_eq_sum
#print axioms mem_support
#print axioms incr_orderEmbOfFin
#print axioms choose_card_eq_sum_powersetCard
#print axioms card_cols_containing_le
#print axioms hasKst_transpose_of
#print axioms colSum_transpose
#print axioms rowSum_transpose
#print axioms rowSum_deleteCol
#print axioms incr_succAbove_comp
#print axioms card_support
#print axioms hasKst_of_subsets
#print axioms colBudget
#print axioms hasKst_transpose
#print axioms weight_transpose
#print axioms rowBudget
#print axioms argA
#print axioms argAT
#print axioms argAboth
#print axioms weight_deleteCol
#print axioms not_hasKst_deleteCol

end ZarPrune

import ZarPrune
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Sort
import Mathlib.Order.Fin.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Nat.Choose.Basic

/-!
# Package A: Argument A (column budget) and its transpose

Double counting pairs `(R, j)` with `R` an `s`-subset of the support of column `j`:
each column contributes `C(colSum j, s)` such pairs, and each `s`-set `R` of rows
lies in at most `t - 1` supports (else we have a `K_{s,t}`).
-/

namespace ZarPrune

open Finset

/-! ### Bridge from `sumFin` to `Finset.sum` -/

theorem sumFin_eq_sum (k : ℕ) (f : Fin k → ℕ) : sumFin k f = ∑ i, f i := by
  induction k with
  | zero => simp [sumFin]
  | succ k ih => rw [sumFin_succ, Fin.sum_univ_succ, ih]

/-! ### Supports -/

/-- The set of rows in which column `j` has a one. -/
def support {m n : ℕ} (A : Mat m n) (j : Fin n) : Finset (Fin m) :=
  Finset.univ.filter (fun i => A i j = true)

theorem mem_support {m n : ℕ} (A : Mat m n) (j : Fin n) (i : Fin m) :
    i ∈ support A j ↔ A i j = true := by
  simp [support]

theorem card_support {m n : ℕ} (A : Mat m n) (j : Fin n) :
    (support A j).card = colSum A j := by
  unfold support colSum
  rw [sumFin_eq_sum, Finset.card_filter]
  rfl

/-! ### From subsets to `HasKst` -/

theorem hasKst_of_subsets (P : Params) (A : Mat P.m P.n)
    (R : Finset (Fin P.m)) (hR : R.card = P.s)
    (C : Finset (Fin P.n)) (hC : C.card = P.t)
    (h : ∀ j ∈ C, R ⊆ support A j) : HasKst P A := by
  refine ⟨R.orderEmbOfFin hR, C.orderEmbOfFin hC, ?_, ?_, ?_⟩
  · intro a b hab
    exact (R.orderEmbOfFin hR).strictMono hab
  · intro a b hab
    exact (C.orderEmbOfFin hC).strictMono hab
  · intro a b
    have hj := Finset.orderEmbOfFin_mem C hC b
    have hi := Finset.orderEmbOfFin_mem R hR a
    exact (mem_support A _ _).mp (h _ hj hi)

/-! ### Argument A: the column budget -/

/-- The `s`-subsets of `support A j`, viewed inside all `s`-subsets of rows. -/
theorem powersetCard_support_eq {m n : ℕ} (A : Mat m n) (j : Fin n) (s : ℕ) :
    (support A j).powersetCard s
      = ((Finset.univ : Finset (Fin m)).powersetCard s).filter (fun R => R ⊆ support A j) := by
  ext R
  simp only [Finset.mem_powersetCard, Finset.mem_filter, Finset.subset_univ, true_and]
  tauto

/-- Each `s`-set of rows lies in the supports of at most `t - 1` columns. -/
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
  have := hCsub hj
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at this
  exact this

theorem colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s := by
  classical
  -- Step 1: choose = number of `s`-subsets of the support, counted inside all `s`-subsets.
  have h1 : ∀ j, (colSum A j).choose P.s
      = ∑ R ∈ (Finset.univ : Finset (Fin P.m)).powersetCard P.s,
          (if R ⊆ support A j then 1 else 0) := by
    intro j
    rw [← card_support, ← Finset.card_powersetCard, powersetCard_support_eq, Finset.card_filter]
  -- Step 2: swap the two sums.
  calc ∑ j, (colSum A j).choose P.s
      = ∑ j, ∑ R ∈ (Finset.univ : Finset (Fin P.m)).powersetCard P.s,
          (if R ⊆ support A j then 1 else 0) := Finset.sum_congr rfl (fun j _ => h1 j)
    _ = ∑ R ∈ (Finset.univ : Finset (Fin P.m)).powersetCard P.s,
          ∑ j, (if R ⊆ support A j then 1 else 0) := Finset.sum_comm
    _ ≤ ∑ R ∈ (Finset.univ : Finset (Fin P.m)).powersetCard P.s, (P.t - 1) := by
        apply Finset.sum_le_sum
        intro R hR
        rw [Finset.mem_powersetCard] at hR
        rw [← Finset.card_filter]
        exact card_cols_containing_le P A h R hR.2
    _ = (P.t - 1) * (P.m).choose P.s := by
        rw [Finset.sum_const, smul_eq_mul, Finset.card_powersetCard, Finset.card_univ,
          Fintype.card_fin, Nat.mul_comm]

/-! ### Transpose and the row budget -/

/-- Transpose of a matrix. -/
def transpose {m n : ℕ} (A : Mat m n) : Mat n m := fun j i => A i j

/-- The parameters with rows and columns swapped. -/
def Params.transpose (P : Params) : Params :=
  { m := P.n, n := P.m, s := P.t, t := P.s, w := P.w }

theorem colSum_transpose {m n : ℕ} (A : Mat m n) (i : Fin m) :
    colSum (transpose A) i = rowSum A i := rfl

theorem hasKst_transpose (P : Params) (A : Mat P.m P.n)
    (h : HasKst P.transpose (transpose A)) : HasKst P A := by
  obtain ⟨R, C, hR, hC, hRC⟩ := h
  exact ⟨C, R, hC, hR, fun a b => hRC b a⟩

theorem not_hasKst_transpose (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ¬ HasKst P.transpose (transpose A) :=
  fun h' => h (hasKst_transpose P A h')

theorem rowBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ i, (rowSum A i).choose P.t ≤ (P.s - 1) * (P.n).choose P.t :=
  colBudget P.transpose (transpose A) (not_hasKst_transpose P A h)

/-! ### The prune terms -/

/-- Argument A on columns: kill when `∑_j C(c_j, s) > (t-1) C(m, s)`. -/
def argA (P : Params) : Prune P where
  name := "argA: column budget Σ C(c_j,s) ≤ (t-1)C(m,s)"
  kill := fun pf => decide ((P.t - 1) * P.m.choose P.s < sumFin P.n (fun j => (pf.col j).choose P.s))
  sound := by
    intro A h hv
    have h1 : (P.t - 1) * P.m.choose P.s < sumFin P.n (fun j => (colSum A j).choose P.s) :=
      of_decide_eq_true h
    rw [sumFin_eq_sum] at h1
    have h2 := colBudget P A hv.1
    omega

/-- Argument A on rows: kill when `∑_i C(r_i, t) > (s-1) C(n, t)`. -/
def argAT (P : Params) : Prune P where
  name := "argAT: row budget Σ C(r_i,t) ≤ (s-1)C(n,t)"
  kill := fun pf => decide ((P.s - 1) * P.n.choose P.t < sumFin P.m (fun i => (pf.row i).choose P.t))
  sound := by
    intro A h hv
    have h1 : (P.s - 1) * P.n.choose P.t < sumFin P.m (fun i => (rowSum A i).choose P.t) :=
      of_decide_eq_true h
    rw [sumFin_eq_sum] at h1
    have h2 := rowBudget P A hv.1
    omega

/-- Both directions folded together. -/
def argumentA (P : Params) : Prune P := Prune.ofList P [argA P, argAT P]

/-! ### Axiom audit -/

#print axioms sumFin_eq_sum
#print axioms mem_support
#print axioms powersetCard_support_eq
#print axioms card_cols_containing_le
#print axioms transpose
#print axioms Params.transpose
#print axioms colSum_transpose
#print axioms hasKst_transpose
#print axioms not_hasKst_transpose
#print axioms support
#print axioms card_support
#print axioms hasKst_of_subsets
#print axioms colBudget
#print axioms rowBudget
#print axioms argA
#print axioms argAT
#print axioms argumentA

/-! ### Computability tests (m = n = 9, s = t = 2) -/

section Tests

/-- z(9,9;2,2) test: does a 9x9 K_{2,2}-free matrix with 50 ones exist?
Column budget: Σ C(c_j,2) ≤ 1·C(9,2) = 36. -/
def P99 : Params := ⟨9, 9, 2, 2, 50⟩

def pf1 : Profile 9 9 :=
  { row := fun i => [6,6,6,6,6,5,5,5,5].getD i.val 0
    col := fun j => [6,6,6,6,6,5,5,5,5].getD j.val 0 }

def pf2 : Profile 9 9 :=
  { row := fun i => [3,3,3,3,3,3,3,3,3].getD i.val 0
    col := fun j => [3,3,3,3,3,3,3,3,3].getD j.val 0 }

def pf3 : Profile 9 9 :=
  { row := fun i => [4,4,4,4,4,4,4,4,4].getD i.val 0
    col := fun j => [4,4,4,3,3,3,3,3,3].getD j.val 0 }

-- Σ C(c,2) for [6,6,6,6,6,5,5,5,5] = 5*15 + 4*10 = 115 > 36: killed.
#eval (argA P99).kill pf1      -- expected: true
#eval (argAT P99).kill pf1     -- expected: true
-- Σ C(3,2) * 9 = 27 ≤ 36: survives.
#eval (argA P99).kill pf2      -- expected: false
#eval (argAT P99).kill pf2     -- expected: false
-- rows: 9*C(4,2) = 54 > 36: killed by row side; cols: 3*6 + 6*3 = 36 ≤ 36: not by column side.
#eval (argA P99).kill pf3      -- expected: false
#eval (argAT P99).kill pf3     -- expected: true
#eval (argumentA P99).kill pf3 -- expected: true

end Tests

end ZarPrune

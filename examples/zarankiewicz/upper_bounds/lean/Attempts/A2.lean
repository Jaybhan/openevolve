import ZarPrune
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Finset.Sort
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Order.Fin.Basic
import Mathlib.Tactic.Linarith

namespace ZarPrune
open Finset

/-! ## Bridge to Mathlib sums -/

theorem sumFin_eq_sum (k : ℕ) (f : Fin k → ℕ) : sumFin k f = ∑ i, f i := by
  induction k with
  | zero => simp [sumFin]
  | succ k ih => rw [sumFin_succ, Fin.sum_univ_succ, ih]

/-! ## Supports -/

def support {m n : ℕ} (A : Mat m n) (j : Fin n) : Finset (Fin m) :=
  Finset.univ.filter (fun i => A i j = true)

theorem mem_support {m n : ℕ} (A : Mat m n) (j : Fin n) (i : Fin m) :
    i ∈ support A j ↔ A i j = true := by
  simp [support]

theorem card_support {m n : ℕ} (A : Mat m n) (j : Fin n) :
    (support A j).card = colSum A j := by
  unfold colSum
  rw [sumFin_eq_sum, support, Finset.card_eq_sum_ones, Finset.sum_filter]
  refine Finset.sum_congr rfl ?_
  intro i _
  unfold ind
  cases A i j <;> simp

/-! ## From subsets to `HasKst` -/

theorem incr_orderEmbOfFin {N k : ℕ} (S : Finset (Fin N)) (h : S.card = k) :
    Incr (S.orderEmbOfFin h) := by
  intro a b hab
  exact (S.orderEmbOfFin h).strictMono hab

theorem hasKst_of_subsets (P : Params) (A : Mat P.m P.n)
    (R : Finset (Fin P.m)) (hR : R.card = P.s)
    (C : Finset (Fin P.n)) (hC : C.card = P.t)
    (h : ∀ j ∈ C, R ⊆ support A j) : HasKst P A := by
  refine ⟨R.orderEmbOfFin hR, C.orderEmbOfFin hC, incr_orderEmbOfFin R hR,
    incr_orderEmbOfFin C hC, ?_⟩
  intro a b
  have hj : (C.orderEmbOfFin hC) b ∈ C := Finset.orderEmbOfFin_mem C hC b
  have hi : (R.orderEmbOfFin hR) a ∈ R := Finset.orderEmbOfFin_mem R hR a
  exact (mem_support A _ _).1 (h _ hj hi)


/-! ## The incidence set of (s-subset of rows, column) pairs -/

/-- Pairs `(R, j)` with `R` an `s`-subset of the rows and `R ⊆ support A j`. -/
def inc {m n : ℕ} (A : Mat m n) (s : ℕ) : Finset (Finset (Fin m) × Fin n) :=
  ((Finset.univ.powersetCard s) ×ˢ Finset.univ).filter (fun p => p.1 ⊆ support A p.2)

theorem mem_inc {m n : ℕ} (A : Mat m n) (s : ℕ) (p : Finset (Fin m) × Fin n) :
    p ∈ inc A s ↔ p.1.card = s ∧ p.1 ⊆ support A p.2 := by
  simp [inc, Finset.mem_powersetCard]

/-- Fiber of `inc` over a column `j` is the set of `s`-subsets of the support. -/
theorem inc_fiber_col {m n : ℕ} (A : Mat m n) (s : ℕ) (j : Fin n) :
    (inc A s).filter (fun p => p.2 = j)
      = ((support A j).powersetCard s).map ⟨fun R => (R, j), fun _ _ h => (Prod.ext_iff.1 h).1⟩ := by
  ext ⟨R, j'⟩
  simp only [Finset.mem_filter, mem_inc, Finset.mem_map, Finset.mem_powersetCard]
  constructor
  · rintro ⟨⟨hc, hsub⟩, rfl⟩
    exact ⟨R, ⟨hsub, hc⟩, rfl⟩
  · rintro ⟨a, ⟨hsub, hc⟩, hEq⟩
    have h' : (a, j) = (R, j') := hEq
    simp only [Prod.mk.injEq] at h'
    obtain ⟨rfl, rfl⟩ := h'
    exact ⟨⟨hc, hsub⟩, rfl⟩

/-- Counting `inc` column by column. -/
theorem card_inc_by_col {m n : ℕ} (A : Mat m n) (s : ℕ) :
    (inc A s).card = ∑ j, (colSum A j).choose s := by
  rw [Finset.card_eq_sum_card_fiberwise (f := Prod.snd) (t := Finset.univ)
    (fun _ _ => Finset.mem_univ _)]
  refine Finset.sum_congr rfl ?_
  intro j _
  rw [inc_fiber_col, Finset.card_map, Finset.card_powersetCard, card_support]

/-- Fiber of `inc` over a row-set `R` is (a copy of) the set of columns containing `R`. -/
theorem inc_fiber_row {m n : ℕ} (A : Mat m n) (s : ℕ) (R : Finset (Fin m)) (hR : R.card = s) :
    (inc A s).filter (fun p => p.1 = R)
      = (Finset.univ.filter (fun j => R ⊆ support A j)).map
          ⟨fun j => (R, j), fun _ _ h => (Prod.ext_iff.1 h).2⟩ := by
  ext ⟨R', j⟩
  simp only [Finset.mem_filter, mem_inc, Finset.mem_map, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨⟨_, hsub⟩, rfl⟩
    exact ⟨j, hsub, rfl⟩
  · rintro ⟨a, hsub, hEq⟩
    have h' : (R, a) = (R', j) := hEq
    simp only [Prod.mk.injEq] at h'
    obtain ⟨rfl, rfl⟩ := h'
    exact ⟨⟨hR, hsub⟩, rfl⟩

/-- Counting `inc` row-set by row-set. -/
theorem card_inc_by_row {m n : ℕ} (A : Mat m n) (s : ℕ) :
    (inc A s).card
      = ∑ R ∈ (Finset.univ : Finset (Fin m)).powersetCard s,
          (Finset.univ.filter (fun j => R ⊆ support A j)).card := by
  rw [Finset.card_eq_sum_card_fiberwise (f := Prod.fst)
    (t := (Finset.univ : Finset (Fin m)).powersetCard s)
    (fun p hp => by
      rw [Finset.mem_coe, mem_inc] at hp
      rw [Finset.mem_coe, Finset.mem_powersetCard]
      exact ⟨Finset.subset_univ _, hp.1⟩)]
  refine Finset.sum_congr rfl ?_
  intro R hR
  rw [Finset.mem_powersetCard] at hR
  rw [inc_fiber_row A s R hR.2, Finset.card_map]

/-- In a `K_{s,t}`-free matrix, fewer than `t` columns contain any given `s`-set of rows. -/
theorem card_cols_containing_lt (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A)
    (R : Finset (Fin P.m)) (hR : R.card = P.s) :
    (Finset.univ.filter (fun j => R ⊆ support A j)).card < P.t := by
  by_contra hge
  push Not at hge
  obtain ⟨C, hCsub, hC⟩ := Finset.exists_subset_card_eq hge
  apply h
  refine hasKst_of_subsets P A R hR C hC ?_
  intro j hj
  exact (Finset.mem_filter.1 (hCsub hj)).2

/-- **Argument A (column budget).** -/
theorem colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s := by
  rw [← card_inc_by_col, card_inc_by_row]
  calc ∑ R ∈ (Finset.univ : Finset (Fin P.m)).powersetCard P.s,
          (Finset.univ.filter (fun j => R ⊆ support A j)).card
      ≤ ∑ _R ∈ (Finset.univ : Finset (Fin P.m)).powersetCard P.s, (P.t - 1) := by
        refine Finset.sum_le_sum ?_
        intro R hR
        rw [Finset.mem_powersetCard] at hR
        have := card_cols_containing_lt P A h R hR.2
        omega
    _ = (P.t - 1) * (P.m).choose P.s := by
        rw [Finset.sum_const, smul_eq_mul, Finset.card_powersetCard, Finset.card_univ,
          Fintype.card_fin, Nat.mul_comm]

/-! ## Transpose and the row budget -/

def transpose {m n : ℕ} (A : Mat m n) : Mat n m := fun j i => A i j

def Params.transpose (P : Params) : Params := ⟨P.n, P.m, P.t, P.s, P.w⟩

theorem rowSum_transpose {m n : ℕ} (A : Mat m n) (i : Fin m) :
    colSum (transpose A) i = rowSum A i := rfl

theorem hasKst_transpose (P : Params) (A : Mat P.m P.n) (h : HasKst P.transpose (transpose A)) :
    HasKst P A := by
  obtain ⟨C, R, hC, hR, hall⟩ := h
  exact ⟨R, C, hR, hC, fun a b => hall b a⟩

/-- **Argument A, row side (row budget).** -/
theorem rowBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ i, (rowSum A i).choose P.t ≤ (P.s - 1) * (P.n).choose P.t := by
  have h' : ¬ HasKst P.transpose (transpose A) := fun hk => h (hasKst_transpose P A hk)
  exact colBudget P.transpose (transpose A) h'

/-! ## Prune terms -/

/-- Argument A as a profile-level kill: `∑_j C(c_j, s) > (t-1)·C(m, s)`. -/
def argA (P : Params) : Prune P where
  name := "argA: column budget"
  kill := fun pf => decide ((P.t - 1) * P.m.choose P.s < sumFin P.n (fun j => (pf.col j).choose P.s))
  sound := by
    intro A h hv
    have h1 : (P.t - 1) * P.m.choose P.s < sumFin P.n (fun j => (colSum A j).choose P.s) :=
      of_decide_eq_true h
    rw [sumFin_eq_sum] at h1
    have h2 := colBudget P A hv.1
    omega

/-- Argument A, transposed: `∑_i C(r_i, t) > (s-1)·C(n, t)`. -/
def argAT (P : Params) : Prune P where
  name := "argAT: row budget"
  kill := fun pf => decide ((P.s - 1) * P.n.choose P.t < sumFin P.m (fun i => (pf.row i).choose P.t))
  sound := by
    intro A h hv
    have h1 : (P.s - 1) * P.n.choose P.t < sumFin P.m (fun i => (rowSum A i).choose P.t) :=
      of_decide_eq_true h
    rw [sumFin_eq_sum] at h1
    have h2 := rowBudget P A hv.1
    omega

def countingA (P : Params) : Prune P := Prune.ofList P [argA P, argAT P]

/-! ## Deletion lemma (bonus) -/

def deleteCol {m n : ℕ} (A : Mat m (n+1)) (j : Fin (n+1)) : Mat m n :=
  fun i k => A i (j.succAbove k)

theorem rowSum_deleteCol {m n : ℕ} (A : Mat m (n+1)) (j : Fin (n+1)) (i : Fin m) :
    rowSum (deleteCol A j) i + ind (A i j) = rowSum A i := by
  unfold rowSum deleteCol
  rw [sumFin_eq_sum, sumFin_eq_sum, Fin.sum_univ_succAbove _ j, Nat.add_comm]

theorem weight_deleteCol {m n : ℕ} (A : Mat m (n+1)) (j : Fin (n+1)) :
    weight (deleteCol A j) + colSum A j = weight A := by
  unfold weight colSum
  rw [← sumFin_add]
  exact sumFin_congr _ _ _ (fun i => rowSum_deleteCol A j i)

theorem not_hasKst_deleteCol {m n s t w w' : ℕ} (A : Mat m (n+1)) (j : Fin (n+1))
    (h : ¬ HasKst ⟨m, n+1, s, t, w⟩ A) : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteCol A j) := by
  rintro ⟨R, C, hR, hC, hall⟩
  apply h
  refine ⟨R, fun b => j.succAbove (C b), hR, ?_, hall⟩
  intro a b hab
  exact Fin.strictMono_succAbove j (hC a b hab)

/-! ## Computable checks -/

def demoP9 : Params := ⟨9, 9, 2, 2, 50⟩
def pf9 : Profile 9 9 :=
  { row := fun i => [6,6,6,6,6,5,5,5,5].getD i.val 0
    col := fun j => [6,6,6,6,6,5,5,5,5].getD j.val 0 }
-- 5*15 + 4*10 = 115 > 1*36 → killed
#eval (argA demoP9).kill pf9
#eval (argAT demoP9).kill pf9
#eval (countingA demoP9).kill pf9
def pf9b : Profile 9 9 :=
  { row := fun i => [3,3,3,3,3,3,3,3,3].getD i.val 0
    col := fun j => [3,3,3,3,3,3,3,3,3].getD j.val 0 }
-- 9*3 = 27 ≤ 36 → survives
#eval (argA demoP9).kill pf9b
#eval (countingA demoP9).kill pf9b
-- K_{2,3}: sum C(c,2) ≤ 2*36 = 72; profile 6^5 5^4 gives 115 → killed; rows: C(r,3): 5*20+4*10=140 ≤ 1*84? no → killed
#eval (argA ⟨9,9,2,3,50⟩).kill pf9
#eval (argAT ⟨9,9,2,3,50⟩).kill pf9
-- known z(9,9;2,2)=29: profile 4^2 3^7 (sum 29): 2*6+7*3 = 33 ≤ 36 survives argA
def pf9c : Profile 9 9 :=
  { row := fun i => [4,4,3,3,3,3,3,3,3].getD i.val 0
    col := fun j => [4,4,3,3,3,3,3,3,3].getD j.val 0 }
#eval (argA demoP9).kill pf9c
-- profile 4^4 3^5 (sum 31): 4*6 + 5*3 = 39 > 36 → killed
def pf9d : Profile 9 9 :=
  { row := fun i => [4,4,4,4,3,3,3,3,3].getD i.val 0
    col := fun j => [4,4,4,4,3,3,3,3,3].getD j.val 0 }
#eval (argA demoP9).kill pf9d

#print axioms sumFin_eq_sum
#print axioms card_support
#print axioms hasKst_of_subsets
#print axioms card_inc_by_col
#print axioms card_inc_by_row
#print axioms colBudget
#print axioms rowBudget
#print axioms argA
#print axioms argAT
#print axioms countingA
#print axioms weight_deleteCol
#print axioms not_hasKst_deleteCol

end ZarPrune

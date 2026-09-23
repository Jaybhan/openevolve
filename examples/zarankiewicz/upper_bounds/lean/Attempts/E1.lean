import ZarPrune
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Order.Fin.Basic

/-
Package E — the deletion lemma.

`deleteCol A j` / `deleteRow A i` remove one column / row via `Fin.succAbove`.
Deleting a column drops exactly `colSum A j` ones and cannot create a `K_{s,t}`.
From any *proved* bound `weight B ≤ U` on the smaller `(m, n)` instance this gives
a profile-level prune on the `(m, n+1)` instance:

    weight A  =  weight (deleteCol A j₀) + colSum A j₀  ≤  U + min_j colSum A j,

so a case with `Σ row > U + min col` is empty.  `delMinCol` / `delMinRow` package
this, parametrised by `U` and the proof `hU`, so they can be instantiated once a
counting bound for the smaller instance is available.
-/

namespace ZarPrune

open Finset

/-! ### Bridge to `Finset.sum` -/

/-- `sumFin` agrees with Mathlib's `Finset.sum` over `Fin k`. -/
theorem sumFin_eq_sum (k : ℕ) (f : Fin k → ℕ) : sumFin k f = ∑ i, f i := by
  induction k with
  | zero => simp [sumFin]
  | succ k ih => rw [sumFin_succ, Fin.sum_univ_succ, ih]

/-- Splitting a `sumFin` over `Fin (n+1)` at an arbitrary index `j`. -/
theorem sumFin_succAbove (n : ℕ) (g : Fin (n + 1) → ℕ) (j : Fin (n + 1)) :
    sumFin n (fun k => g (j.succAbove k)) + g j = sumFin (n + 1) g := by
  rw [sumFin_eq_sum, sumFin_eq_sum, Fin.sum_univ_succAbove g j, Nat.add_comm]

/-! ### `HasKst` does not depend on the weight field -/

theorem hasKst_w_irrel {m n s t w w' : ℕ} (A : Mat m n) :
    HasKst ⟨m, n, s, t, w⟩ A ↔ HasKst ⟨m, n, s, t, w'⟩ A := Iff.rfl

/-! ### Column deletion -/

/-- Delete column `j`: the remaining columns are re-indexed by `j.succAbove`. -/
def deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) : Mat m n :=
  fun i k => A i (j.succAbove k)

theorem colSum_deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) (k : Fin n) :
    colSum (deleteCol A j) k = colSum A (j.succAbove k) := rfl

theorem weight_deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) :
    weight (deleteCol A j) + colSum A j = weight A := by
  rw [weight_eq_sum_colSum, weight_eq_sum_colSum]
  exact sumFin_succAbove n (colSum A) j

/-- Composing an increasing tuple with `succAbove` keeps it increasing. -/
theorem Incr.succAbove_comp {k n : ℕ} (j : Fin (n + 1)) {C : Fin k → Fin n} (hC : Incr C) :
    Incr (fun b => j.succAbove (C b)) := by
  intro a b hab
  exact Fin.succAbove_lt_succAbove_iff.mpr (hC a b hab)

/-- A `K_{s,t}` in the column-deleted matrix is a `K_{s,t}` in the original. -/
theorem hasKst_of_deleteCol {m n s t w w' : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1))
    (h : HasKst ⟨m, n, s, t, w'⟩ (deleteCol A j)) : HasKst ⟨m, n + 1, s, t, w⟩ A := by
  obtain ⟨R, C, hR, hC, hall⟩ := h
  exact ⟨R, fun b => j.succAbove (C b), hR, hC.succAbove_comp j, fun a b => hall a b⟩

theorem not_hasKst_deleteCol {m n s t w w' : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1))
    (h : ¬ HasKst ⟨m, n + 1, s, t, w⟩ A) : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteCol A j) :=
  fun h' => h (hasKst_of_deleteCol A j h')

/-! ### Row deletion -/

/-- Delete row `i`: the remaining rows are re-indexed by `i.succAbove`. -/
def deleteRow {m n : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1)) : Mat m n :=
  fun k j => A (i.succAbove k) j

theorem rowSum_deleteRow {m n : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1)) (k : Fin m) :
    rowSum (deleteRow A i) k = rowSum A (i.succAbove k) := rfl

theorem weight_deleteRow {m n : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1)) :
    weight (deleteRow A i) + rowSum A i = weight A :=
  sumFin_succAbove m (rowSum A) i

theorem hasKst_of_deleteRow {m n s t w w' : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1))
    (h : HasKst ⟨m, n, s, t, w'⟩ (deleteRow A i)) : HasKst ⟨m + 1, n, s, t, w⟩ A := by
  obtain ⟨R, C, hR, hC, hall⟩ := h
  exact ⟨fun a => i.succAbove (R a), C, hR.succAbove_comp i, hC, fun a b => hall a b⟩

theorem not_hasKst_deleteRow {m n s t w w' : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1))
    (h : ¬ HasKst ⟨m + 1, n, s, t, w⟩ A) : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteRow A i) :=
  fun h' => h (hasKst_of_deleteRow A i h')

/-! ### Computable minimum over `Fin (k+1)` -/

/-- `minFin k f = min (f 0) (f 1) ... (f k)` over `Fin (k+1)` (nonempty, so no default). -/
def minFin : (k : ℕ) → (Fin (k + 1) → ℕ) → ℕ
  | 0,     f => f 0
  | k + 1, f => Nat.min (f 0) (minFin k (fun i => f i.succ))

theorem minFin_zero (f : Fin 1 → ℕ) : minFin 0 f = f 0 := rfl

theorem minFin_succ (k : ℕ) (f : Fin (k + 2) → ℕ) :
    minFin (k + 1) f = Nat.min (f 0) (minFin k (fun i => f i.succ)) := rfl

/-- The minimum is attained. -/
theorem exists_minFin_eq : ∀ (k : ℕ) (f : Fin (k + 1) → ℕ), ∃ j, minFin k f = f j := by
  intro k
  induction k with
  | zero => intro f; exact ⟨0, rfl⟩
  | succ k ih =>
      intro f
      obtain ⟨j, hj⟩ := ih (fun i => f i.succ)
      rw [minFin_succ, hj]
      rcases Nat.le_total (f 0) (f j.succ) with h | h
      · exact ⟨0, Nat.min_eq_left h⟩
      · exact ⟨j.succ, Nat.min_eq_right h⟩

/-- The minimum is a lower bound. -/
theorem minFin_le : ∀ (k : ℕ) (f : Fin (k + 1) → ℕ) (j : Fin (k + 1)), minFin k f ≤ f j := by
  intro k
  induction k with
  | zero =>
      intro f j
      rw [minFin_zero, Fin.eq_zero j]
  | succ k ih =>
      intro f j
      rw [minFin_succ]
      refine Fin.cases ?_ ?_ j
      · exact Nat.min_le_left _ _
      · intro i
        exact Nat.le_trans (Nat.min_le_right _ _) (ih (fun i => f i.succ) i)

/-! ### Trivial bound, used to instantiate the prunes in tests -/

theorem weight_le_mul {m n : ℕ} (A : Mat m n) : weight A ≤ m * n :=
  sumFin_le m (rowSum A) n (rowSum_le A)

/-! ### The prunes -/

/-- The computable kill: total row weight exceeds `U + min_j col j`. -/
def delMinColKill (m n U : ℕ) (pf : Profile m (n + 1)) : Bool :=
  decide (sumFin m pf.row > U + minFin n pf.col)

/-- **Delete-min-column prune.**  Given a proved bound `weight B ≤ U` for every
`K_{s,t}`-free `B : Mat m n`, no `K_{s,t}`-free `A : Mat m (n+1)` can have
`weight A > U + min_j colSum A j`.  Stated as a function of `U` and its proof `hU`
(with an arbitrary weight field `w'` in the hypothesis, since `HasKst` ignores it),
so that it can be instantiated later from any counting bound. -/
def delMinCol (m n s t w : ℕ) (U : ℕ) (w' : ℕ)
    (hU : ∀ B : Mat m n, ¬ HasKst ⟨m, n, s, t, w'⟩ B → weight B ≤ U) :
    Prune ⟨m, n + 1, s, t, w⟩ where
  name := s!"delete-min-column (U={U})"
  kill := delMinColKill m n U
  sound := by
    intro A h hv
    have hk : sumFin m (rowSum A) > U + minFin n (colSum A) := of_decide_eq_true h
    obtain ⟨j, hj⟩ := exists_minFin_eq n (colSum A)
    rw [hj] at hk
    have hfree : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteCol A j) :=
      not_hasKst_deleteCol A j hv.1
    have hB : sumFin m (rowSum (deleteCol A j)) ≤ U := hU _ hfree
    have hw : sumFin m (rowSum (deleteCol A j)) + colSum A j = sumFin m (rowSum A) :=
      weight_deleteCol A j
    omega

/-- The computable kill for the row version: total weight exceeds `U + min_i row i`. -/
def delMinRowKill (m n U : ℕ) (pf : Profile (m + 1) n) : Bool :=
  decide (sumFin (m + 1) pf.row > U + minFin m pf.row)

/-- **Delete-min-row prune**, the transpose of `delMinCol`. -/
def delMinRow (m n s t w : ℕ) (U : ℕ) (w' : ℕ)
    (hU : ∀ B : Mat m n, ¬ HasKst ⟨m, n, s, t, w'⟩ B → weight B ≤ U) :
    Prune ⟨m + 1, n, s, t, w⟩ where
  name := s!"delete-min-row (U={U})"
  kill := delMinRowKill m n U
  sound := by
    intro A h hv
    have hk : sumFin (m + 1) (rowSum A) > U + minFin m (rowSum A) := of_decide_eq_true h
    obtain ⟨i, hi⟩ := exists_minFin_eq m (rowSum A)
    rw [hi] at hk
    have hfree : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteRow A i) :=
      not_hasKst_deleteRow A i hv.1
    have hB : sumFin m (rowSum (deleteRow A i)) ≤ U := hU _ hfree
    have hw : sumFin m (rowSum (deleteRow A i)) + rowSum A i = sumFin (m + 1) (rowSum A) :=
      weight_deleteRow A i
    omega

/-! ### Tests: the kills compute -/

/-- m = n = 9 profile from the task statement: rows `[6,6,6,6,6,5,5,5,5]`, same cols. -/
def testPf9 : Profile 9 (8 + 1) :=
  { row := fun i => [6,6,6,6,6,5,5,5,5].getD i.val 0
    col := fun j => [6,6,6,6,6,5,5,5,5].getD j.val 0 }

-- total = 50, min col = 5.  U = 44 → 50 > 49 kills; U = 45 → 50 > 50 does not.
#eval delMinColKill 9 8 44 testPf9   -- true
#eval delMinColKill 9 8 45 testPf9   -- false
#eval delMinRowKill 8 9 44 testPf9   -- true
#eval delMinRowKill 8 9 45 testPf9   -- false

-- The packaged prune, instantiated with the trivial bound `weight B ≤ 9*8 = 72`.
-- It evaluates (kill is `delMinColKill`) and never fires on this profile.
#eval (delMinCol 9 8 2 2 50 72 0 (fun B _ => weight_le_mul B)).kill testPf9   -- false
#eval (delMinCol 9 8 2 2 50 72 0 (fun B _ => weight_le_mul B)).name

-- A profile that the trivial instantiation does kill: rows summing to 90, min col 8.
def testPfHeavy : Profile 9 (8 + 1) :=
  { row := fun _ => 10, col := fun j => if j.val = 0 then 8 else 11 }
#eval (delMinCol 9 8 2 2 50 72 0 (fun B _ => weight_le_mul B)).kill testPfHeavy   -- true
#eval minFin 8 testPfHeavy.col   -- 8

/-! ### Axiom audit -/

#print axioms sumFin_eq_sum
#print axioms sumFin_succAbove
#print axioms hasKst_w_irrel
#print axioms deleteCol
#print axioms colSum_deleteCol
#print axioms weight_deleteCol
#print axioms Incr.succAbove_comp
#print axioms hasKst_of_deleteCol
#print axioms not_hasKst_deleteCol
#print axioms deleteRow
#print axioms rowSum_deleteRow
#print axioms weight_deleteRow
#print axioms hasKst_of_deleteRow
#print axioms not_hasKst_deleteRow
#print axioms minFin
#print axioms exists_minFin_eq
#print axioms minFin_le
#print axioms weight_le_mul
#print axioms delMinColKill
#print axioms delMinCol
#print axioms delMinRowKill
#print axioms delMinRow

end ZarPrune

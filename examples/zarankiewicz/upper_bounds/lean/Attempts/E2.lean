import ZarPrune
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Finset.Sort
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Find
import Mathlib.Order.Fin.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Attempt E2 — deletion lemma + waterfilling bound

Package E of COUNTING_SPEC.md:
* `deleteCol`, `weight_deleteCol`, `not_hasKst_deleteCol` (and the row duals),
* the consequence of `colBudget` needed to instantiate a weight bound `U`:
  if `∑ j, (c j).choose s ≤ B` and every `c j ≤ m` then `∑ j, c j ≤ waterfillBound m n s B`.
  This is proved in full generality via the discrete tangent-line inequality for
  `Nat.choose` (no exchange argument needed).
-/

namespace ZarPrune

open Finset

/-! ### Bridge to Mathlib sums -/

theorem sumFin_eq_sum (k : ℕ) (f : Fin k → ℕ) : sumFin k f = ∑ i, f i := by
  induction k with
  | zero => simp [sumFin]
  | succ k ih => rw [sumFin_succ, Fin.sum_univ_succ, ih]

theorem rowSum_eq_sum {m n : ℕ} (A : Mat m n) (i : Fin m) :
    rowSum A i = ∑ j, ind (A i j) := sumFin_eq_sum _ _

theorem colSum_eq_sum {m n : ℕ} (A : Mat m n) (j : Fin n) :
    colSum A j = ∑ i, ind (A i j) := sumFin_eq_sum _ _

theorem weight_eq_sum {m n : ℕ} (A : Mat m n) : weight A = ∑ i, rowSum A i :=
  sumFin_eq_sum _ _

/-! ### Column deletion -/

/-- Delete column `j`: the remaining columns are re-indexed by `Fin n` via `succAbove`. -/
def deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) : Mat m n :=
  fun i k => A i (j.succAbove k)

theorem rowSum_deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) (i : Fin m) :
    rowSum (deleteCol A j) i + ind (A i j) = rowSum A i := by
  rw [rowSum_eq_sum, rowSum_eq_sum, Fin.sum_univ_succAbove (fun k => ind (A i k)) j]
  simp only [deleteCol]
  omega

theorem weight_deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) :
    weight (deleteCol A j) + colSum A j = weight A := by
  rw [weight_eq_sum, weight_eq_sum, colSum_eq_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun i _ => rowSum_deleteCol A j i)

theorem hasKst_of_deleteCol {m n s t w w' : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1))
    (h : HasKst ⟨m, n, s, t, w'⟩ (deleteCol A j)) : HasKst ⟨m, n + 1, s, t, w⟩ A := by
  obtain ⟨R, C, hR, hC, hRC⟩ := h
  refine ⟨R, fun b => j.succAbove (C b), hR, ?_, ?_⟩
  · intro a b hab
    exact Fin.succAbove_lt_succAbove_iff.mpr (hC a b hab)
  · intro a b
    exact hRC a b

theorem not_hasKst_deleteCol {m n s t w w' : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1))
    (h : ¬ HasKst ⟨m, n + 1, s, t, w⟩ A) : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteCol A j) :=
  fun h' => h (hasKst_of_deleteCol A j h')

/-! ### Row deletion (dual) -/

def deleteRow {m n : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1)) : Mat m n :=
  fun k j => A (i.succAbove k) j

theorem colSum_deleteRow {m n : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1)) (j : Fin n) :
    colSum (deleteRow A i) j + ind (A i j) = colSum A j := by
  rw [colSum_eq_sum, colSum_eq_sum, Fin.sum_univ_succAbove (fun k => ind (A k j)) i]
  simp only [deleteRow]
  omega

theorem weight_deleteRow {m n : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1)) :
    weight (deleteRow A i) + rowSum A i = weight A := by
  rw [weight_eq_sum_colSum, weight_eq_sum_colSum, sumFin_eq_sum, sumFin_eq_sum,
    rowSum_eq_sum, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun j _ => colSum_deleteRow A i j)

theorem hasKst_of_deleteRow {m n s t w w' : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1))
    (h : HasKst ⟨m, n, s, t, w'⟩ (deleteRow A i)) : HasKst ⟨m + 1, n, s, t, w⟩ A := by
  obtain ⟨R, C, hR, hC, hRC⟩ := h
  refine ⟨fun a => i.succAbove (R a), C, ?_, hC, ?_⟩
  · intro a b hab
    exact Fin.succAbove_lt_succAbove_iff.mpr (hR a b hab)
  · intro a b
    exact hRC a b

theorem not_hasKst_deleteRow {m n s t w w' : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1))
    (h : ¬ HasKst ⟨m + 1, n, s, t, w⟩ A) : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteRow A i) :=
  fun h' => h (hasKst_of_deleteRow A i h')

/-! ### Discrete convexity of `Nat.choose` and the waterfilling bound -/

/-- Tangent-line inequality for the convex sequence `x ↦ x.choose (s+1)`, at the point `a`:
`(x - a) * a.choose s + a.choose (s+1) ≤ x.choose (s+1)`, written without subtraction. -/
theorem choose_tangent (s a x : ℕ) :
    a.choose (s + 1) + x * a.choose s ≤ x.choose (s + 1) + a * a.choose s := by
  rcases Nat.le_total a x with hax | hxa
  · -- x ≥ a: induct on d = x - a
    obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hax
    induction d with
    | zero => simp
    | succ d ih =>
        have hmono : a.choose s ≤ (a + d).choose s := Nat.choose_le_choose s (by omega)
        have hpas : (a + d + 1).choose (s + 1) = (a + d).choose s + (a + d).choose (s + 1) :=
          Nat.choose_succ_succ (a + d) s
        rw [show a + (d + 1) = a + d + 1 by ring, hpas]
        nlinarith [ih (by omega)]
  · -- x ≤ a: induct on d = a - x
    obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hxa
    induction d with
    | zero => simp
    | succ d ih =>
        have hmono : (x + d).choose s ≤ (x + d + 1).choose s := Nat.choose_le_choose s (by omega)
        have hpas : (x + d + 1).choose (s + 1) = (x + d).choose s + (x + d).choose (s + 1) :=
          Nat.choose_succ_succ (x + d) s
        rw [show x + (d + 1) = x + d + 1 by ring, hpas]
        nlinarith [ih (by omega)]

/-- The convex-sum cost of the "as equal as possible" distribution of total `S` over
`n` columns: `n - S%n` columns of size `S/n` and `S%n` columns of size `S/n + 1`
(rewritten via Pascal so no subtraction is needed). -/
def equalCost (n s S : ℕ) : ℕ := n * (S / n).choose (s + 1) + (S % n) * (S / n).choose s

/-- Any distribution `c` with total `S` has convex cost at least `equalCost n s S`. -/
theorem equalCost_le_sum_choose {n : ℕ} (s : ℕ) (c : Fin n → ℕ) :
    equalCost n s (∑ j, c j) ≤ ∑ j, (c j).choose (s + 1) := by
  set S := ∑ j, c j with hS
  set a := S / n with ha
  have hdm : n * a + S % n = S := Nat.div_add_mod S n
  have h1 : ∀ j, a.choose (s + 1) + c j * a.choose s ≤ (c j).choose (s + 1) + a * a.choose s :=
    fun j => choose_tangent s a (c j)
  have h2 := Finset.sum_le_sum (fun j (_ : j ∈ (univ : Finset (Fin n))) => h1 j)
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Finset.sum_const, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, smul_eq_mul, smul_eq_mul, ← Finset.sum_mul, ← hS] at h2
  unfold equalCost
  rw [← ha]
  nlinarith [h2, hdm]

/-- The waterfilling bound: the largest total `S ≤ n*m` whose equal-as-possible
distribution has convex cost within budget `B`. Computable. -/
def waterfillBound (m n s B : ℕ) : ℕ :=
  Nat.findGreatest (fun S => equalCost n (s - 1) S ≤ B) (n * m)

/-- **Waterfilling optimality.** If every column sum is `≤ m` and the convex cost
`∑ j, (c j).choose s` is at most `B`, then the total is at most `waterfillBound m n s B`. -/
theorem sum_le_waterfillBound {n : ℕ} (m s B : ℕ) (hs : 1 ≤ s) (c : Fin n → ℕ)
    (hc : ∀ j, c j ≤ m) (hB : ∑ j, (c j).choose s ≤ B) :
    ∑ j, c j ≤ waterfillBound m n s B := by
  obtain ⟨s', rfl⟩ : ∃ s', s = s' + 1 := ⟨s - 1, by omega⟩
  apply Nat.le_findGreatest
  · calc ∑ j, c j ≤ ∑ _j : Fin n, m := Finset.sum_le_sum (fun j _ => hc j)
      _ = n * m := by simp
  · show equalCost n (s' + 1 - 1) (∑ j, c j) ≤ B
    rw [Nat.add_sub_cancel]
    exact (equalCost_le_sum_choose s' c).trans hB

/-! ### Column budget (Argument A), needed to instantiate `U` -/

/-- The set of rows in which column `j` has a one. -/
def support {m n : ℕ} (A : Mat m n) (j : Fin n) : Finset (Fin m) :=
  univ.filter (fun i => A i j = true)

theorem card_support {m n : ℕ} (A : Mat m n) (j : Fin n) : (support A j).card = colSum A j := by
  rw [colSum_eq_sum, support, Finset.card_filter]
  exact Finset.sum_congr rfl (fun i _ => rfl)

theorem mem_support {m n : ℕ} {A : Mat m n} {j : Fin n} {i : Fin m} :
    i ∈ support A j ↔ A i j = true := by
  simp [support]

theorem hasKst_of_subsets (P : Params) (A : Mat P.m P.n) (R : Finset (Fin P.m)) (hR : R.card = P.s)
    (C : Finset (Fin P.n)) (hC : C.card = P.t) (h : ∀ j ∈ C, R ⊆ support A j) : HasKst P A := by
  refine ⟨R.orderEmbOfFin hR, C.orderEmbOfFin hC, ?_, ?_, ?_⟩
  · intro a b hab; exact (R.orderEmbOfFin hR).strictMono hab
  · intro a b hab; exact (C.orderEmbOfFin hC).strictMono hab
  · intro a b
    have hb : C.orderEmbOfFin hC b ∈ C := Finset.orderEmbOfFin_mem C hC b
    have ha : R.orderEmbOfFin hR a ∈ R := Finset.orderEmbOfFin_mem R hR a
    exact mem_support.mp (h _ hb ha)

/-- `(support A j).powersetCard s` as a filter of all `s`-subsets of rows. -/
theorem powersetCard_support_eq {m n : ℕ} (A : Mat m n) (j : Fin n) (s : ℕ) :
    (support A j).powersetCard s
      = ((univ : Finset (Fin m)).powersetCard s).filter (fun R => R ⊆ support A j) := by
  ext R
  simp only [Finset.mem_powersetCard, Finset.mem_filter, Finset.subset_univ, true_and]
  tauto

/-- For each `s`-subset `R` of rows, fewer than `t` columns contain `R`. -/
theorem card_cols_containing_lt (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A)
    (R : Finset (Fin P.m)) (hR : R.card = P.s) :
    ((univ : Finset (Fin P.n)).filter (fun j => R ⊆ support A j)).card ≤ P.t - 1 := by
  by_contra hlt
  have ht : P.t ≤ ((univ : Finset (Fin P.n)).filter (fun j => R ⊆ support A j)).card := by omega
  obtain ⟨C, hCsub, hC⟩ := Finset.exists_subset_card_eq ht
  apply h
  refine hasKst_of_subsets P A R hR C hC ?_
  intro j hj
  exact (Finset.mem_filter.mp (hCsub hj)).2

/-- **Argument A (column budget).** -/
theorem colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s := by
  calc ∑ j, (colSum A j).choose P.s
      = ∑ j, ((support A j).powersetCard P.s).card := by
        refine Finset.sum_congr rfl (fun j _ => ?_)
        rw [Finset.card_powersetCard, card_support]
    _ = ∑ j, ∑ R ∈ (univ : Finset (Fin P.m)).powersetCard P.s,
          (if R ⊆ support A j then 1 else 0) := by
        refine Finset.sum_congr rfl (fun j _ => ?_)
        rw [powersetCard_support_eq, Finset.card_filter]
    _ = ∑ R ∈ (univ : Finset (Fin P.m)).powersetCard P.s,
          ((univ : Finset (Fin P.n)).filter (fun j => R ⊆ support A j)).card := by
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl (fun R _ => ?_)
        rw [Finset.card_filter]
    _ ≤ ∑ _R ∈ (univ : Finset (Fin P.m)).powersetCard P.s, (P.t - 1) := by
        refine Finset.sum_le_sum (fun R hR => ?_)
        exact card_cols_containing_lt P A h R (Finset.mem_powersetCard.mp hR).2
    _ = (P.t - 1) * (P.m).choose P.s := by
        rw [Finset.sum_const, smul_eq_mul, Finset.card_powersetCard, Finset.card_univ,
          Fintype.card_fin, Nat.mul_comm]

/-- The budget of Argument A as a number. -/
def colBudgetOf (P : Params) : ℕ := (P.t - 1) * (P.m).choose P.s

/-- Weight bound for any `K_{s,t}`-free matrix, obtained by waterfilling the column budget. -/
theorem weight_le_waterfill (P : Params) (hs : 1 ≤ P.s) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    weight A ≤ waterfillBound P.m P.n P.s (colBudgetOf P) := by
  rw [weight_eq_sum_colSum, sumFin_eq_sum]
  exact sum_le_waterfillBound P.m P.s (colBudgetOf P) hs (colSum A) (colSum_le A) (colBudget P A h)

/-! ### Deletion prunes

`argDelCol P U hU`: given *any* verified weight bound `U` for `K_{s,t}`-free
`m × (n-1)` matrices, kill a case in which some column is so light that deleting it
leaves more than `U` ones.  `U` can come from `weight_le_waterfill`, from a table of
proved exact values `z(m, n-1; s, t)`, or from a previously verified prune run. -/

theorem valid_deleteCol_bound {m n s t w U : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1))
    (hU : ∀ B : Mat m n, ¬ HasKst ⟨m, n, s, t, 0⟩ B → weight B ≤ U)
    (hj : colSum A j + U < w) : ¬ Valid ⟨m, n + 1, s, t, w⟩ A := by
  rintro ⟨hfree, hw⟩
  have h1 := weight_deleteCol A j
  have h2 := hU (deleteCol A j) (not_hasKst_deleteCol A j hfree)
  change w ≤ weight A at hw
  omega

theorem valid_deleteRow_bound {m n s t w U : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1))
    (hU : ∀ B : Mat m n, ¬ HasKst ⟨m, n, s, t, 0⟩ B → weight B ≤ U)
    (hi : rowSum A i + U < w) : ¬ Valid ⟨m + 1, n, s, t, w⟩ A := by
  rintro ⟨hfree, hw⟩
  have h1 := weight_deleteRow A i
  have h2 := hU (deleteRow A i) (not_hasKst_deleteRow A i hfree)
  change w ≤ weight A at hw
  omega

/-- Kill when some column `j` has `pf.col j + U < w`, where `U` bounds the weight of
every `K_{s,t}`-free `m × (n-1)` matrix. -/
def argDelCol (P : Params) (U : ℕ)
    (hU : ∀ B : Mat P.m (P.n - 1), ¬ HasKst ⟨P.m, P.n - 1, P.s, P.t, 0⟩ B → weight B ≤ U) :
    Prune P where
  name := "delete-lightest-column"
  kill := fun pf => (List.finRange P.n).any (fun j => decide (pf.col j + U < P.w))
  sound := by
    intro A h
    rw [List.any_eq_true] at h
    obtain ⟨j, _, hj⟩ := h
    have hj' : colSum A j + U < P.w := of_decide_eq_true hj
    obtain ⟨m, n, s, t, w⟩ := P
    cases n with
    | zero => exact j.elim0
    | succ n =>
        exact valid_deleteCol_bound A j hU hj'

/-- Kill when some row `i` has `pf.row i + U < w`, where `U` bounds the weight of
every `K_{s,t}`-free `(m-1) × n` matrix. -/
def argDelRow (P : Params) (U : ℕ)
    (hU : ∀ B : Mat (P.m - 1) P.n, ¬ HasKst ⟨P.m - 1, P.n, P.s, P.t, 0⟩ B → weight B ≤ U) :
    Prune P where
  name := "delete-lightest-row"
  kill := fun pf => (List.finRange P.m).any (fun i => decide (pf.row i + U < P.w))
  sound := by
    intro A h
    rw [List.any_eq_true] at h
    obtain ⟨i, _, hi⟩ := h
    have hi' : rowSum A i + U < P.w := of_decide_eq_true hi
    obtain ⟨m, n, s, t, w⟩ := P
    cases m with
    | zero => exact i.elim0
    | succ m =>
        exact valid_deleteRow_bound A i hU hi'

/-- Concrete instantiation: `U` = waterfilling of the column budget of the `(m, n-1)` instance.
Requires `1 ≤ s`; for `s = 0` the prune degenerates to `never`. -/
def argDelColWF (P : Params) : Prune P :=
  if hs : 1 ≤ P.s then
    argDelCol P (waterfillBound P.m (P.n - 1) P.s (colBudgetOf ⟨P.m, P.n - 1, P.s, P.t, 0⟩))
      (fun B hB => weight_le_waterfill ⟨P.m, P.n - 1, P.s, P.t, 0⟩ hs B hB)
  else Prune.never P

/-- Concrete instantiation: `U` = waterfilling of the column budget of the `(m-1, n)` instance. -/
def argDelRowWF (P : Params) : Prune P :=
  if hs : 1 ≤ P.s then
    argDelRow P (waterfillBound (P.m - 1) P.n P.s (colBudgetOf ⟨P.m - 1, P.n, P.s, P.t, 0⟩))
      (fun B hB => weight_le_waterfill ⟨P.m - 1, P.n, P.s, P.t, 0⟩ hs B hB)
  else Prune.never P

/-- Plain waterfilled Argument A as a prune (dominated by `argA` but self-contained). -/
def argWF (P : Params) : Prune P where
  name := "waterfilled-column-budget"
  kill := fun _ => decide (1 ≤ P.s) && decide (waterfillBound P.m P.n P.s (colBudgetOf P) < P.w)
  sound := by
    intro A h ⟨hfree, hw⟩
    rw [Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff] at h
    have := weight_le_waterfill P h.1 A hfree
    omega

def deletion (P : Params) : Prune P :=
  Prune.ofList P [argDelColWF P, argDelRowWF P, argWF P]

/-! ### Tests -/

def testP : Params := ⟨9, 9, 2, 3, 41⟩
def testPf : Profile 9 9 :=
  { row := fun i => [6, 6, 6, 6, 6, 5, 5, 5, 5].getD i.val 0,
    col := fun j => [6, 6, 6, 6, 6, 5, 5, 5, 5].getD j.val 0 }
def testPf2 : Profile 9 9 :=
  { row := fun i => [5, 5, 5, 5, 5, 5, 5, 5, 1].getD i.val 0,
    col := fun j => [5, 5, 5, 5, 5, 5, 5, 5, 1].getD j.val 0 }

#eval waterfillBound 9 9 2 (2 * Nat.choose 9 2)   -- 40 (Argument A waterfilled, m = n = 9, s = 2, t = 3)
#eval waterfillBound 9 8 2 (2 * Nat.choose 9 2)   -- 38
#eval (argWF testP).kill testPf                    -- true  (41 > 40)
#eval (argWF ⟨9, 9, 2, 3, 40⟩).kill testPf         -- false
#eval (argDelColWF ⟨9, 9, 2, 3, 40⟩).kill testPf   -- false (40 - 5 = 35 ≤ 38)
#eval (argDelColWF ⟨9, 9, 2, 3, 40⟩).kill testPf2  -- true  (40 - 1 = 39 > 38)
#eval (argDelRowWF ⟨9, 9, 2, 3, 40⟩).kill testPf2  -- true
#eval (deletion ⟨9, 9, 2, 3, 40⟩).kill testPf2     -- true
#eval (deletion ⟨9, 9, 2, 3, 40⟩).kill testPf      -- false
#eval (argDelColWF ⟨9, 0, 2, 3, 40⟩).kill ⟨fun _ => 0, fun j => j.elim0⟩  -- false (n = 0)

/-- Brute-force maximum of `∑ c` over `c ∈ [0..m]^n` with `∑ (c i).choose s ≤ B`
(test-only, exponential; used to check `waterfillBound` is tight on tiny instances). -/
def bruteMax (m n s B : ℕ) : ℕ :=
  let rec go : ℕ → List (List ℕ)
    | 0 => [[]]
    | k + 1 => (go k).flatMap (fun l => (List.range (m + 1)).map (fun x => x :: l))
  ((go n).filter (fun l => (l.map (fun x => x.choose s)).sum ≤ B)).foldl
    (fun acc l => max acc l.sum) 0

#eval (List.range 4).map (fun B => (bruteMax 3 2 2 B, waterfillBound 3 2 2 B))
#eval (List.range 12).map (fun B => (bruteMax 4 3 2 B, waterfillBound 4 3 2 B))
#eval (List.range 8).map (fun B => (bruteMax 4 3 3 B, waterfillBound 4 3 3 B))
#eval (List.range 6).map (fun B => (bruteMax 5 2 1 B, waterfillBound 5 2 1 B))

#print axioms sumFin_eq_sum
#print axioms card_support
#print axioms hasKst_of_subsets
#print axioms colBudget
#print axioms weight_le_waterfill
#print axioms argDelCol
#print axioms argDelRow
#print axioms argDelColWF
#print axioms argDelRowWF
#print axioms argWF
#print axioms deletion

#print axioms weight_deleteCol
#print axioms not_hasKst_deleteCol
#print axioms weight_deleteRow
#print axioms not_hasKst_deleteRow
#print axioms sum_le_waterfillBound

end ZarPrune

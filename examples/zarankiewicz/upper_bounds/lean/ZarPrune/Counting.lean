import ZarPrune.Prunes
import Mathlib.Data.Finset.Powerset
import Mathlib.Data.Finset.Sort
import Mathlib.Data.Fintype.Card
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Find
import Mathlib.Order.Fin.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# The counting core: Guy's Arguments A and D, the deletion lemma, waterfilling

This is the only module of `ZarPrune` that depends on Mathlib.  It proves the
double-counting lemmas behind the classical Zarankiewicz pruning arguments and
packages them as verified `Prune` terms, so the evolutionary search starts from a
library in which the classical arguments are *already* verified and reusable.

Contents (see `COUNTING_SPEC.md` for the contract and `COUNTING_NOTES.md` for the
proof strategy):

* **Bridge** `sumFin_eq_sum` between the Mathlib-free `sumFin` and `Finset.sum`.
* **Supports** `support A j` (rows with a one in column `j`) and `rowSupport A i`.
* `hasKst_of_subsets`: an `s`-set of rows inside the supports of `t` columns is a `K_{s,t}`.
* `budget_general`: the one double-counting lemma every counting argument is an
  instance of.
* **Argument A** `colBudget` / `rowBudget`, and the prunes `argA` / `argAT`.
* **Argument D** `rowLocalBudget`, its profile-level bound `boundD` (layer-cake form,
  equal to the "sum over the `r` lightest columns" form), and the prunes `argD` / `argDT`.
* **Transposition** `Params.transpose`, `transpose`, `Prune.transposed`: any prune on
  the transposed instance is a prune on the original.
* **Deletion** `deleteCol` / `deleteRow`: removing a line drops exactly its sum and
  cannot create a `K_{s,t}`.
* **Waterfilling** `waterfillBound`: from `∑ C(c_j, s) ≤ B` to `∑ c_j ≤ U`, via the
  discrete tangent-line inequality for `Nat.choose`.
* **Deletion prunes** `argDelCol` / `argDelRow`, parametrised by any proved weight
  bound `U` for the instance with one line fewer, and their waterfilled instances.
* `counting P`: every prune above folded together.

Gate: no `sorry`, no `native_decide`, no new axioms; every `#print axioms` is within
`{propext, Quot.sound, Classical.choice}`.  All `kill` functions are computable and
fast (`sumFin` / `allFin` / `Finset.range` / `Finset.univ.filter` on `Fin`).
-/

namespace ZarPrune

open Finset

/-! ## Bridge to Mathlib sums -/

/-- `sumFin` agrees with Mathlib's `Finset.sum` over `Fin k`. -/
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

/-- Splitting a `sumFin` over `Fin (n+1)` at an arbitrary index `j`. -/
theorem sumFin_succAbove (n : ℕ) (g : Fin (n + 1) → ℕ) (j : Fin (n + 1)) :
    sumFin n (fun k => g (j.succAbove k)) + g j = sumFin (n + 1) g := by
  rw [sumFin_eq_sum, sumFin_eq_sum, Fin.sum_univ_succAbove g j, Nat.add_comm]

/-! ## Supports -/

/-- The set of rows in which column `j` has a one. -/
def support {m n : ℕ} (A : Mat m n) (j : Fin n) : Finset (Fin m) :=
  univ.filter (fun i => A i j = true)

/-- The set of columns in which row `i` has a one. -/
def rowSupport {m n : ℕ} (A : Mat m n) (i : Fin m) : Finset (Fin n) :=
  univ.filter (fun j => A i j = true)

theorem mem_support {m n : ℕ} (A : Mat m n) (j : Fin n) (i : Fin m) :
    i ∈ support A j ↔ A i j = true := by
  simp [support]

theorem mem_rowSupport {m n : ℕ} (A : Mat m n) (i : Fin m) (j : Fin n) :
    j ∈ rowSupport A i ↔ A i j = true := by
  simp [rowSupport]

theorem card_support {m n : ℕ} (A : Mat m n) (j : Fin n) :
    (support A j).card = colSum A j := by
  unfold support colSum
  rw [sumFin_eq_sum, Finset.card_filter]
  rfl

theorem card_rowSupport {m n : ℕ} (A : Mat m n) (i : Fin m) :
    (rowSupport A i).card = rowSum A i := by
  unfold rowSupport rowSum
  rw [sumFin_eq_sum, Finset.card_filter]
  rfl

/-! ## From finsets to `HasKst` -/

theorem incr_of_strictMono {k N : ℕ} (f : Fin k → Fin N) (h : StrictMono f) : Incr f :=
  fun _a _b hab => h hab

/-- An `s`-set of rows contained in the supports of `t` columns is a `K_{s,t}`.
`Finset.orderEmbOfFin` turns a finset of known card into a strictly increasing
enumeration, which is exactly `Incr`. -/
theorem hasKst_of_subsets (P : Params) (A : Mat P.m P.n)
    (R : Finset (Fin P.m)) (hR : R.card = P.s)
    (C : Finset (Fin P.n)) (hC : C.card = P.t)
    (h : ∀ j ∈ C, R ⊆ support A j) : HasKst P A := by
  refine ⟨R.orderEmbOfFin hR, C.orderEmbOfFin hC, ?_, ?_, ?_⟩
  · exact incr_of_strictMono _ (R.orderEmbOfFin hR).strictMono
  · exact incr_of_strictMono _ (C.orderEmbOfFin hC).strictMono
  · intro a b
    have hb : C.orderEmbOfFin hC b ∈ C := Finset.orderEmbOfFin_mem C hC b
    have ha : R.orderEmbOfFin hR a ∈ R := Finset.orderEmbOfFin_mem R hR a
    exact (mem_support A (C.orderEmbOfFin hC b) (R.orderEmbOfFin hR a)).mp
      (h (C.orderEmbOfFin hC b) hb ha)

/-! ## The generic double-counting lemma -/

/-- `C(|S|, k)` counted as the number of `k`-subsets of an ambient `T ⊇ S` that lie in `S`. -/
theorem card_powersetCard_eq_sum {α : Type*} [DecidableEq α] (S T : Finset α) (hST : S ⊆ T)
    (k : ℕ) :
    (S.powersetCard k).card = ∑ R ∈ T.powersetCard k, if R ⊆ S then 1 else 0 := by
  rw [← Finset.card_filter]
  congr 1
  ext R
  simp only [Finset.mem_filter, Finset.mem_powersetCard]
  constructor
  · rintro ⟨h1, h2⟩; exact ⟨⟨h1.trans hST, h2⟩, h1⟩
  · rintro ⟨⟨_, h2⟩, h1⟩; exact ⟨h1, h2⟩

/-- **Double counting.**  Count pairs `(R, j)` with `R` a `k`-subset of `T` contained in
`S j`, `j ∈ J`.  If each `R` lies in at most `B` of the sets `S j`, then
`∑_{j ∈ J} C(|S j|, k) ≤ B · C(|T|, k)`.  Arguments A and D are both instances. -/
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

/-! ## Argument A: the column budget -/

/-- **Argument A (column budget).**  In a `K_{s,t}`-free matrix,
`∑_j C(c_j, s) ≤ (t-1) · C(m, s)`: each `s`-set of rows is contained in at most `t-1`
columns, else those columns and rows form a `K_{s,t}`. -/
theorem colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s := by
  have key := budget_general (univ : Finset (Fin P.m)) (univ : Finset (Fin P.n))
    (support A) (fun j _ => Finset.subset_univ _) P.s (P.t - 1) ?_
  · rw [Finset.card_univ, Fintype.card_fin] at key
    refine le_trans (le_of_eq ?_) key
    apply Finset.sum_congr rfl
    intro j _
    rw [card_support]
  · intro R hR
    by_contra hcon
    have ht : P.t ≤ (univ.filter (fun j => R ⊆ support A j)).card := by omega
    obtain ⟨C, hCsub, hC⟩ := Finset.exists_subset_card_eq ht
    rw [Finset.mem_powersetCard] at hR
    exact h (hasKst_of_subsets P A R hR.2 C hC
      (fun j hj => (Finset.mem_filter.mp (hCsub hj)).2))

/-! ## Argument D: the row-local budget -/

/-- **Argument D (row-local budget).**  For a `K_{s,t}`-free matrix and any row `i`,
summing `C(c_j - 1, s - 1)` over the columns `j` with a one in row `i` gives at most
`(t-1) · C(m-1, s-1)`.  Double counting over pairs `(R, j)` with `R` an `(s-1)`-subset
of the rows other than `i` contained in column `j`: if some `R` sat in `t` such
columns, `R ∪ {i}` would be an `s`-set of rows common to `t` columns. -/
theorem rowLocalBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) (i : Fin P.m)
    (hs : 1 ≤ P.s) :
    ∑ j ∈ rowSupport A i, (colSum A j - 1).choose (P.s - 1)
      ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1) := by
  have hT : ((univ : Finset (Fin P.m)).erase i).card = P.m - 1 := by
    rw [Finset.card_erase_of_mem (Finset.mem_univ i), Finset.card_univ, Fintype.card_fin]
  have key := budget_general (univ.erase i) (rowSupport A i)
    (fun j => (support A j).erase i)
    (fun j _ => Finset.erase_subset_erase i (Finset.subset_univ _))
    (P.s - 1) (P.t - 1) ?_
  · rw [hT] at key
    refine le_trans (le_of_eq ?_) key
    apply Finset.sum_congr rfl
    intro j hj
    have hij : i ∈ support A j := (mem_support A j i).mpr ((mem_rowSupport A i j).mp hj)
    rw [Finset.card_erase_of_mem hij, card_support]
  · intro R hR
    by_contra hcon
    have ht : P.t ≤ ((rowSupport A i).filter (fun j => R ⊆ (support A j).erase i)).card := by
      omega
    obtain ⟨C, hCsub, hC⟩ := Finset.exists_subset_card_eq ht
    rw [Finset.mem_powersetCard] at hR
    have hiR : i ∉ R := fun hi => by
      have := hR.1 hi
      simp at this
    apply h
    refine hasKst_of_subsets P A (insert i R) ?_ C hC ?_
    · rw [Finset.card_insert_of_notMem hiR, hR.2]; omega
    · intro j hj
      have hj' := Finset.mem_filter.mp (hCsub hj)
      rw [Finset.insert_subset_iff]
      exact ⟨(mem_support A j i).mpr ((mem_rowSupport A i j).mp hj'.1),
        hj'.2.trans (Finset.erase_subset i _)⟩

/-! ## Transposition -/

/-- The transposed instance: rows and columns swap, and so do `s` and `t`. -/
def Params.transpose (P : Params) : Params := ⟨P.n, P.m, P.t, P.s, P.w⟩

def transpose {m n : ℕ} (A : Mat m n) : Mat n m := fun j i => A i j

/-- Swap the row and column vectors of a profile. -/
def Profile.swap {m n : ℕ} (pf : Profile m n) : Profile n m := ⟨pf.col, pf.row⟩

theorem rowSum_transpose {m n : ℕ} (A : Mat m n) : rowSum (transpose A) = colSum A := rfl
theorem colSum_transpose {m n : ℕ} (A : Mat m n) : colSum (transpose A) = rowSum A := rfl

theorem weight_transpose {m n : ℕ} (A : Mat m n) : weight (transpose A) = weight A := by
  rw [weight_eq_sum_colSum A]; rfl

theorem profileOf_transpose {m n : ℕ} (A : Mat m n) :
    profileOf (transpose A) = (profileOf A).swap := rfl

/-- A `K_{s,t}` in `A` is a `K_{t,s}` in `Aᵀ`: swap the two witnesses. -/
theorem hasKst_transpose (P : Params) (A : Mat P.m P.n) :
    HasKst P.transpose (transpose A) ↔ HasKst P A :=
  ⟨fun ⟨C, R, hC, hR, h⟩ => ⟨R, C, hR, hC, fun a b => h b a⟩,
   fun ⟨R, C, hR, hC, h⟩ => ⟨C, R, hC, hR, fun b a => h a b⟩⟩

theorem valid_transpose (P : Params) (A : Mat P.m P.n) :
    Valid P.transpose (transpose A) ↔ Valid P A :=
  ⟨fun ⟨h1, h2⟩ => ⟨fun h => h1 ((hasKst_transpose P A).mpr h),
      Nat.le_trans h2 (Nat.le_of_eq (weight_transpose A))⟩,
   fun ⟨h1, h2⟩ => ⟨fun h => h1 ((hasKst_transpose P A).mp h),
      Nat.le_trans h2 (Nat.le_of_eq (weight_transpose A).symm)⟩⟩

/-- **Transposition combinator.**  A verified prune for the transposed instance is a
verified prune for the original: run it on the swapped profile. -/
def Prune.transposed {P : Params} (q : Prune P.transpose) : Prune P where
  name := q.name ++ " [transposed]"
  kill := fun pf => q.kill pf.swap
  sound := fun A h hv => q.sound (transpose A) h ((valid_transpose P A).mpr hv)

/-- **Argument A, row side.** `∑_i C(r_i, t) ≤ (s-1) · C(n, t)`. -/
theorem rowBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ i, (rowSum A i).choose P.t ≤ (P.s - 1) * (P.n).choose P.t :=
  colBudget P.transpose (transpose A) (fun h' => h ((hasKst_transpose P A).mp h'))

/-! ## Argument A as prunes -/

/-- Argument A on the column side: kill when `∑_j C(c_j, s) > (t-1) · C(m, s)`. -/
def argA (P : Params) : Prune P where
  name := "argA: Σ_j C(c_j,s) ≤ (t-1)C(m,s)"
  kill := fun pf =>
    decide ((P.t - 1) * (P.m).choose P.s < sumFin P.n (fun j => (pf.col j).choose P.s))
  sound := by
    intro A hk hv
    have h1 : (P.t - 1) * (P.m).choose P.s < sumFin P.n (fun j => (colSum A j).choose P.s) :=
      of_decide_eq_true hk
    rw [sumFin_eq_sum] at h1
    have h2 := colBudget P A hv.1
    omega

/-- Argument A on the row side: kill when `∑_i C(r_i, t) > (s-1) · C(n, t)`. -/
def argAT (P : Params) : Prune P where
  name := "argAT: Σ_i C(r_i,t) ≤ (s-1)C(n,t)"
  kill := fun pf =>
    decide ((P.s - 1) * (P.n).choose P.t < sumFin P.m (fun i => (pf.row i).choose P.t))
  sound := by
    intro A hk hv
    have h1 : (P.s - 1) * (P.n).choose P.t < sumFin P.m (fun i => (rowSum A i).choose P.t) :=
      of_decide_eq_true hk
    rw [sumFin_eq_sum] at h1
    have h2 := rowBudget P A hv.1
    omega

/-! ## Argument D at the profile level (layer-cake / all-thresholds form)

The Python reference sorts the column sums, takes the `r` lightest (`r` = the row sum),
and adds up `C(c-1, s-1)`.  We compute the same number without sorting: writing
`fD c = fD 0 + ∑_{x < c} gD (x+1)` (telescoping increments), a row of sum `r` collects
the increment `gD (x+1)` from every one of its columns with `c_j > x`, and at most
`cntLt (x+1)` columns of the whole profile fail that, so at least `r - cntLt (x+1)` do.
-/

/-- `fD s c = C(c-1, s-1)`: the contribution of a column of sum `c`. -/
def fD (s c : ℕ) : ℕ := (c - 1).choose (s - 1)

/-- The increment `fD s x - fD s (x-1)`. -/
def gD (s x : ℕ) : ℕ := fD s x - fD s (x - 1)

/-- Number of columns of the profile whose sum is `< y`. -/
def cntLt {m n : ℕ} (pf : Profile m n) (y : ℕ) : ℕ :=
  (univ.filter (fun j => pf.col j < y)).card

/-- Lower bound on `∑_{j ∋ i} fD s c_j` for a row of sum `r`, computed from the profile
alone.  Equal to the sum of `fD` over the `r` lightest columns. -/
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

/-- At least `r - cntLt (x+1)` of the columns of row `i` have sum `> x`. -/
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
`∑_{j ∋ i} C(c_j - 1, s - 1)` above the row-local budget `(t-1) · C(m-1, s-1)`.
Every row is checked (equivalent to checking the heaviest, since `boundD` is
monotone in `r`).  The guard `1 ≤ s` is needed for `insert i R` to have card `s`. -/
def argD (P : Params) : Prune P where
  name := "argD: row-local budget (r lightest columns)"
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

/-- Argument D, column version: `argD` on the transposed instance. -/
def argDT (P : Params) : Prune P := (argD P.transpose).transposed

/-! ## Deletion of a row or a column -/

/-- Delete column `j`: the remaining columns are re-indexed by `j.succAbove`. -/
def deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) : Mat m n :=
  fun i k => A i (j.succAbove k)

/-- Delete row `i`: the remaining rows are re-indexed by `i.succAbove`. -/
def deleteRow {m n : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1)) : Mat m n :=
  fun k j => A (i.succAbove k) j

theorem colSum_deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) (k : Fin n) :
    colSum (deleteCol A j) k = colSum A (j.succAbove k) := rfl

theorem rowSum_deleteRow {m n : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1)) (k : Fin m) :
    rowSum (deleteRow A i) k = rowSum A (i.succAbove k) := rfl

/-- Deleting a column drops exactly its sum. -/
theorem weight_deleteCol {m n : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) :
    weight (deleteCol A j) + colSum A j = weight A := by
  rw [weight_eq_sum_colSum, weight_eq_sum_colSum]
  exact sumFin_succAbove n (colSum A) j

/-- Deleting a row drops exactly its sum. -/
theorem weight_deleteRow {m n : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1)) :
    weight (deleteRow A i) + rowSum A i = weight A :=
  sumFin_succAbove m (rowSum A) i

/-- Composing an increasing tuple with `succAbove` keeps it increasing. -/
theorem Incr.succAbove_comp {k n : ℕ} (j : Fin (n + 1)) {C : Fin k → Fin n} (hC : Incr C) :
    Incr (fun b => j.succAbove (C b)) := by
  intro a b hab
  exact Fin.succAbove_lt_succAbove_iff.mpr (hC a b hab)

/-- A `K_{s,t}` in the column-deleted matrix is a `K_{s,t}` in the original.  The weight
field of `Params` is irrelevant to `HasKst`, so the two instances carry independent `w`. -/
theorem hasKst_of_deleteCol {m n s t w w' : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1))
    (h : HasKst ⟨m, n, s, t, w'⟩ (deleteCol A j)) : HasKst ⟨m, n + 1, s, t, w⟩ A := by
  obtain ⟨R, C, hR, hC, hall⟩ := h
  exact ⟨R, fun b => j.succAbove (C b), hR, hC.succAbove_comp j, fun a b => hall a b⟩

theorem not_hasKst_deleteCol {m n s t w w' : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1))
    (h : ¬ HasKst ⟨m, n + 1, s, t, w⟩ A) : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteCol A j) :=
  fun h' => h (hasKst_of_deleteCol A j h')

theorem hasKst_of_deleteRow {m n s t w w' : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1))
    (h : HasKst ⟨m, n, s, t, w'⟩ (deleteRow A i)) : HasKst ⟨m + 1, n, s, t, w⟩ A := by
  obtain ⟨R, C, hR, hC, hall⟩ := h
  exact ⟨fun a => i.succAbove (R a), C, hR.succAbove_comp i, hC, fun a b => hall a b⟩

theorem not_hasKst_deleteRow {m n s t w w' : ℕ} (A : Mat (m + 1) n) (i : Fin (m + 1))
    (h : ¬ HasKst ⟨m + 1, n, s, t, w⟩ A) : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteRow A i) :=
  fun h' => h (hasKst_of_deleteRow A i h')

/-! ## Waterfilling: from a `choose`-budget to a weight bound

`x ↦ C(x, s+1)` is discretely convex, so among all column-sum vectors with a given
total the one that is as equal as possible minimises `∑ C(c_j, s+1)`.  Rather than an
exchange argument, we use the tangent-line inequality at the mean.
-/

/-- Tangent-line inequality for the convex sequence `x ↦ x.choose (s+1)` at `a`:
`(x - a) · C(a, s) + C(a, s+1) ≤ C(x, s+1)`, written without subtraction. -/
theorem choose_tangent (s a x : ℕ) :
    a.choose (s + 1) + x * a.choose s ≤ x.choose (s + 1) + a * a.choose s := by
  rcases Nat.le_total a x with hax | hxa
  · obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hax
    induction d with
    | zero => simp
    | succ d ih =>
        have hmono : a.choose s ≤ (a + d).choose s := Nat.choose_le_choose s (by omega)
        have hpas : (a + d + 1).choose (s + 1) = (a + d).choose s + (a + d).choose (s + 1) :=
          Nat.choose_succ_succ (a + d) s
        rw [show a + (d + 1) = a + d + 1 by ring, hpas]
        nlinarith [ih (by omega)]
  · obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le hxa
    induction d with
    | zero => simp
    | succ d ih =>
        have hmono : (x + d).choose s ≤ (x + d + 1).choose s := Nat.choose_le_choose s (by omega)
        have hpas : (x + d + 1).choose (s + 1) = (x + d).choose s + (x + d).choose (s + 1) :=
          Nat.choose_succ_succ (x + d) s
        rw [show x + (d + 1) = x + d + 1 by ring, hpas]
        nlinarith [ih (by omega)]

/-- The convex cost of the "as equal as possible" distribution of total `S` over `n`
columns: `n - S%n` columns of size `S/n` and `S%n` columns of size `S/n + 1`
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

/-- The waterfilling bound: the largest total `S ≤ n·m` whose equal-as-possible
distribution has convex cost `∑ C(c_j, s)` within budget `B`.  Computable. -/
def waterfillBound (m n s B : ℕ) : ℕ :=
  Nat.findGreatest (fun S => equalCost n (s - 1) S ≤ B) (n * m)

/-- **Waterfilling optimality.**  If every column sum is `≤ m` and
`∑ j, C(c_j, s) ≤ B`, then `∑ j, c_j ≤ waterfillBound m n s B`. -/
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

/-- The budget of Argument A as a number. -/
def colBudgetOf (P : Params) : ℕ := (P.t - 1) * (P.m).choose P.s

/-- Weight bound for any `K_{s,t}`-free matrix: waterfill the column budget. -/
theorem weight_le_waterfill (P : Params) (hs : 1 ≤ P.s) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    weight A ≤ waterfillBound P.m P.n P.s (colBudgetOf P) := by
  rw [weight_eq_sum_colSum, sumFin_eq_sum]
  exact sum_le_waterfillBound P.m P.s (colBudgetOf P) hs (colSum A) (colSum_le A) (colBudget P A h)

/-! ## Deletion prunes

`argDelCol P U hU`: given *any* verified weight bound `U` for `K_{s,t}`-free
`m × (n-1)` matrices, kill a case in which some column is so light that deleting it
would leave more than `U` ones.  `U` can come from `weight_le_waterfill`, from a
table of proved exact values `z(m, n-1; s, t)`, or from a previously verified run. -/

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
  name := s!"delete-lightest-column (U={U})"
  kill := fun pf => (List.finRange P.n).any (fun j => decide (pf.col j + U < P.w))
  sound := by
    intro A h
    rw [List.any_eq_true] at h
    obtain ⟨j, _, hj⟩ := h
    have hj' : colSum A j + U < P.w := of_decide_eq_true hj
    obtain ⟨m, n, s, t, w⟩ := P
    cases n with
    | zero => exact j.elim0
    | succ n => exact valid_deleteCol_bound A j hU hj'

/-- Kill when some row `i` has `pf.row i + U < w`, where `U` bounds the weight of
every `K_{s,t}`-free `(m-1) × n` matrix. -/
def argDelRow (P : Params) (U : ℕ)
    (hU : ∀ B : Mat (P.m - 1) P.n, ¬ HasKst ⟨P.m - 1, P.n, P.s, P.t, 0⟩ B → weight B ≤ U) :
    Prune P where
  name := s!"delete-lightest-row (U={U})"
  kill := fun pf => (List.finRange P.m).any (fun i => decide (pf.row i + U < P.w))
  sound := by
    intro A h
    rw [List.any_eq_true] at h
    obtain ⟨i, _, hi⟩ := h
    have hi' : rowSum A i + U < P.w := of_decide_eq_true hi
    obtain ⟨m, n, s, t, w⟩ := P
    cases m with
    | zero => exact i.elim0
    | succ m => exact valid_deleteRow_bound A i hU hi'

/-- `argDelCol` with `U` = the waterfilled column budget of the `(m, n-1)` instance.
For `s = 0` the prune degenerates to `never`. -/
def argDelColWF (P : Params) : Prune P :=
  if hs : 1 ≤ P.s then
    argDelCol P (waterfillBound P.m (P.n - 1) P.s (colBudgetOf ⟨P.m, P.n - 1, P.s, P.t, 0⟩))
      (fun B hB => weight_le_waterfill ⟨P.m, P.n - 1, P.s, P.t, 0⟩ hs B hB)
  else Prune.never P

/-- `argDelRow` with `U` = the waterfilled column budget of the `(m-1, n)` instance. -/
def argDelRowWF (P : Params) : Prune P :=
  if hs : 1 ≤ P.s then
    argDelRow P (waterfillBound (P.m - 1) P.n P.s (colBudgetOf ⟨P.m - 1, P.n, P.s, P.t, 0⟩))
      (fun B hB => weight_le_waterfill ⟨P.m - 1, P.n, P.s, P.t, 0⟩ hs B hB)
  else Prune.never P

/-- Waterfilled Argument A: kill every case when `w` exceeds the waterfilled column
budget of the instance itself.  Profile-independent; dominated by `argA` on
consistent profiles, but it is the proved statement `z(m,n;s,t) ≤ waterfillBound`. -/
def argWF (P : Params) : Prune P where
  name := "waterfilled column budget"
  kill := fun _ => decide (1 ≤ P.s) && decide (waterfillBound P.m P.n P.s (colBudgetOf P) < P.w)
  sound := by
    intro A h ⟨hfree, hw⟩
    rw [Bool.and_eq_true, decide_eq_true_iff, decide_eq_true_iff] at h
    have := weight_le_waterfill P h.1 A hfree
    omega

/-! ## Bundles -/

/-- Argument A, both sides. -/
def countingA (P : Params) : Prune P := Prune.ofList P [argA P, argAT P]

/-- Argument D, both sides. -/
def countingD (P : Params) : Prune P := Prune.ofList P [argD P, argDT P]

/-- The deletion prunes, waterfilled. -/
def deletion (P : Params) : Prune P :=
  Prune.ofList P [argDelColWF P, argDelRowWF P, argWF P]

/-- Every proved counting prune, folded together.  This is the library the
evolutionary search starts from, on top of `baseline`. -/
def counting (P : Params) : Prune P :=
  Prune.ofList P [argA P, argAT P, argD P, argDT P, argDelColWF P, argDelRowWF P, argWF P]

end ZarPrune

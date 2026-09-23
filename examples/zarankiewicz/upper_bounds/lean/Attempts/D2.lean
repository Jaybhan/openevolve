import ZarPrune
import Mathlib.Data.Fin.Tuple.Sort
import Mathlib.Data.Finset.Sort
import Mathlib.Data.Finset.Powerset
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.List.OfFn
import Mathlib.Data.List.Sort
import Mathlib.Tactic.Linarith

/-
Attempt D2: Argument D (row-local budget) in the EXACT sorted-lightest-columns form,
plus Argument A (column budget) since both fall out of one generic double-counting lemma.

The exchange argument is done without layer-cake: if `g : Fin n → ℕ` is monotone and
`T` is an `r`-subset of indices, enumerate `T` increasingly by `τ = T.orderEmbOfFin`;
strict monotonicity gives `a ≤ τ a`, so `f (g a) ≤ f (g (τ a))` for monotone `f`, and
summing over `a : Fin r` gives  "sum over the first r indices ≤ sum over T".
The sorted list computed by the kill (`List.mergeSort`) is identified with
`List.ofFn (col ∘ Tuple.sort col)` by uniqueness of sorted permutations.
-/

namespace ZarPrune

open Finset

/-! ### Bridge to Mathlib sums -/

theorem sumFin_eq_sum (k : ℕ) (f : Fin k → ℕ) : sumFin k f = ∑ i, f i := by
  induction k with
  | zero => simp [sumFin]
  | succ k ih => rw [sumFin_succ, Fin.sum_univ_succ, ih]

/-! ### Supports -/

def support {m n : ℕ} (A : Mat m n) (j : Fin n) : Finset (Fin m) :=
  univ.filter (fun i => A i j = true)

theorem card_support {m n : ℕ} (A : Mat m n) (j : Fin n) : (support A j).card = colSum A j := by
  unfold support colSum
  rw [sumFin_eq_sum, card_filter]
  apply sum_congr rfl
  intro i _
  simp [ind]

def rowSupport {m n : ℕ} (A : Mat m n) (i : Fin m) : Finset (Fin n) :=
  univ.filter (fun j => A i j = true)

theorem card_rowSupport {m n : ℕ} (A : Mat m n) (i : Fin m) :
    (rowSupport A i).card = rowSum A i := by
  unfold rowSupport rowSum
  rw [sumFin_eq_sum, card_filter]
  apply sum_congr rfl
  intro j _
  simp [ind]

theorem mem_support {m n : ℕ} (A : Mat m n) (j : Fin n) (i : Fin m) :
    i ∈ support A j ↔ A i j = true := by
  simp [support]

/-! ### From subsets to `HasKst` -/

theorem hasKst_of_subsets (P : Params) (A : Mat P.m P.n) (R : Finset (Fin P.m)) (hR : R.card = P.s)
    (C : Finset (Fin P.n)) (hC : C.card = P.t) (h : ∀ j ∈ C, R ⊆ support A j) : HasKst P A := by
  refine ⟨R.orderEmbOfFin hR, C.orderEmbOfFin hC, ?_, ?_, ?_⟩
  · exact fun a b hab => (R.orderEmbOfFin hR).strictMono hab
  · exact fun a b hab => (C.orderEmbOfFin hC).strictMono hab
  · intro a b
    have hj := C.orderEmbOfFin_mem hC b
    have hi := R.orderEmbOfFin_mem hR a
    exact (mem_support A _ _).mp (h _ hj hi)

/-! ### The generic double-counting lemma -/

/-- Double counting pairs `(R, j)` with `R ⊆ X j`, `R.card = k`, `R ⊆ U`:
if every `k`-subset `R` of `U` is contained in at most `B` of the sets `X j` (`j ∈ J`),
then `∑_{j ∈ J} C(|X j|, k) ≤ B · C(|U|, k)`. -/
theorem sum_choose_le_of_bounded_fibers {m n : ℕ} (U : Finset (Fin m)) (X : Fin n → Finset (Fin m))
    (hX : ∀ j, X j ⊆ U) (J : Finset (Fin n)) (k B : ℕ)
    (hB : ∀ R ∈ U.powersetCard k, (J.filter (fun j => R ⊆ X j)).card ≤ B) :
    ∑ j ∈ J, (X j).card.choose k ≤ B * U.card.choose k := by
  have h1 : ∀ j ∈ J, (X j).card.choose k = ∑ R ∈ U.powersetCard k, if R ⊆ X j then 1 else 0 := by
    intro j _
    rw [← card_filter, ← card_powersetCard]
    congr 1
    ext R
    simp only [mem_filter, mem_powersetCard]
    constructor
    · rintro ⟨hRX, hcard⟩; exact ⟨⟨hRX.trans (hX j), hcard⟩, hRX⟩
    · rintro ⟨⟨_, hcard⟩, hRX⟩; exact ⟨hRX, hcard⟩
  rw [sum_congr rfl h1, sum_comm]
  calc ∑ R ∈ U.powersetCard k, ∑ j ∈ J, (if R ⊆ X j then 1 else 0)
      = ∑ R ∈ U.powersetCard k, (J.filter (fun j => R ⊆ X j)).card := by
        apply sum_congr rfl; intro R _; rw [card_filter]
    _ ≤ ∑ R ∈ U.powersetCard k, B := Finset.sum_le_sum hB
    _ = B * U.card.choose k := by
        rw [sum_const, card_powersetCard, smul_eq_mul, mul_comm]

/-! ### Argument A: column budget -/

theorem colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s := by
  have key := sum_choose_le_of_bounded_fibers (univ : Finset (Fin P.m)) (support A)
    (fun j => subset_univ _) univ P.s (P.t - 1) ?_
  · rw [card_univ, Fintype.card_fin] at key
    simpa only [card_support] using key
  · intro R hR
    rw [mem_powersetCard] at hR
    by_contra hlt
    rw [not_le] at hlt
    obtain ⟨C, hC, hCcard⟩ := Finset.exists_subset_card_eq
      (s := univ.filter (fun j => R ⊆ support A j)) (n := P.t) (by omega)
    apply h
    apply hasKst_of_subsets P A R hR.2 C hCcard
    intro j hj
    exact (mem_filter.mp (hC hj)).2

/-! ### Transposition -/

def Params.transpose (P : Params) : Params := ⟨P.n, P.m, P.t, P.s, P.w⟩

def Mat.transpose {m n : ℕ} (A : Mat m n) : Mat n m := fun j i => A i j

theorem rowSum_transpose {m n : ℕ} (A : Mat m n) : rowSum A.transpose = colSum A := rfl
theorem colSum_transpose {m n : ℕ} (A : Mat m n) : colSum A.transpose = rowSum A := rfl

theorem hasKst_transpose (P : Params) (A : Mat P.m P.n) :
    HasKst P.transpose A.transpose ↔ HasKst P A := by
  constructor
  · rintro ⟨R, C, hR, hC, h⟩
    exact ⟨C, R, hC, hR, fun a b => h b a⟩
  · rintro ⟨R, C, hR, hC, h⟩
    exact ⟨C, R, hC, hR, fun a b => h b a⟩

theorem rowBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ i, (rowSum A i).choose P.t ≤ (P.s - 1) * (P.n).choose P.t := by
  have := colBudget P.transpose A.transpose (fun hk => h ((hasKst_transpose P A).mp hk))
  exact this

/-! ### Argument D: row-local budget -/

theorem rowLocalBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) (i : Fin P.m)
    (hs : 1 ≤ P.s) :
    ∑ j ∈ univ.filter (fun j => A i j = true), (colSum A j - 1).choose (P.s - 1)
      ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1) := by
  have hU : ((univ : Finset (Fin P.m)).erase i).card = P.m - 1 := by
    rw [card_erase_of_mem (mem_univ i), card_univ, Fintype.card_fin]
  have key := sum_choose_le_of_bounded_fibers ((univ : Finset (Fin P.m)).erase i)
    (fun j => (support A j).erase i)
    (fun j => erase_subset_erase i (subset_univ _))
    (univ.filter (fun j => A i j = true)) (P.s - 1) (P.t - 1) ?_
  · rw [hU] at key
    refine le_trans (le_of_eq ?_) key
    apply sum_congr rfl
    intro j hj
    rw [mem_filter] at hj
    rw [card_erase_of_mem ((mem_support A j i).mpr hj.2), card_support]
  · intro R hR
    rw [mem_powersetCard] at hR
    by_contra hlt
    rw [not_le] at hlt
    obtain ⟨C, hC, hCcard⟩ := Finset.exists_subset_card_eq
      (s := (univ.filter (fun j => A i j = true)).filter (fun j => R ⊆ (support A j).erase i))
      (n := P.t) (by omega)
    apply h
    have hiR : i ∉ R := fun hi => (mem_erase.mp (hR.1 hi)).1 rfl
    apply hasKst_of_subsets P A (insert i R)
      (by rw [card_insert_of_notMem hiR, hR.2]; omega) C hCcard
    intro j hj
    have hj' := hC hj
    rw [mem_filter, mem_filter] at hj'
    intro x hx
    rw [mem_insert] at hx
    rcases hx with rfl | hx
    · exact (mem_support A j x).mpr hj'.1.2
    · exact mem_of_mem_erase (hj'.2 hx)

/-! ### The exchange lemma (sorted lightest columns) -/

theorem val_le_of_strictMono {r n : ℕ} (τ : Fin r → Fin n) (hτ : StrictMono τ) (a : Fin r) :
    a.val ≤ (τ a).val := by
  obtain ⟨k, hk⟩ := a
  induction k with
  | zero => exact Nat.zero_le _
  | succ k ih =>
    have h1 : k ≤ (τ ⟨k, by omega⟩).val := ih (by omega)
    have h2 : τ ⟨k, by omega⟩ < τ ⟨k + 1, hk⟩ := hτ (Fin.mk_lt_mk.mpr (Nat.lt_succ_self k))
    rw [Fin.lt_def] at h2
    show k + 1 ≤ _
    omega

theorem filter_val_lt_eq_map {n r : ℕ} (hrn : r ≤ n) :
    (univ : Finset (Fin n)).filter (fun k : Fin n => k.val < r) = univ.map (Fin.castLEEmb hrn) := by
  ext k
  simp only [mem_filter, mem_univ, true_and, mem_map, Fin.castLEEmb_apply]
  constructor
  · intro hk; exact ⟨⟨k.val, hk⟩, Fin.ext rfl⟩
  · rintro ⟨a, rfl⟩; exact a.isLt

theorem map_orderEmbOfFin_univ {n r : ℕ} (T : Finset (Fin n)) (hT : T.card = r) :
    univ.map (T.orderEmbOfFin hT).toEmbedding = T := by
  apply Finset.eq_of_subset_of_card_le
  · intro x hx
    rw [mem_map] at hx
    obtain ⟨a, _, rfl⟩ := hx
    exact T.orderEmbOfFin_mem hT a
  · rw [card_map, card_univ, Fintype.card_fin, hT]

/-- For monotone `g` and monotone `f`, the sum of `f ∘ g` over any `r`-set of indices
dominates the sum over the first `r` indices. -/
theorem sum_lightest_le {n r : ℕ} (g : Fin n → ℕ) (hg : Monotone g) (f : ℕ → ℕ) (hf : Monotone f)
    (T : Finset (Fin n)) (hT : T.card = r) :
    ∑ k ∈ univ.filter (fun k : Fin n => k.val < r), f (g k) ≤ ∑ k ∈ T, f (g k) := by
  have hrn : r ≤ n := by
    rw [← hT]; exact (card_le_univ T).trans (by rw [Fintype.card_fin])
  rw [filter_val_lt_eq_map hrn, sum_map, ← map_orderEmbOfFin_univ T hT, sum_map]
  apply Finset.sum_le_sum
  intro a _
  apply hf; apply hg
  rw [Fin.le_def, Fin.castLEEmb_apply, Fin.val_castLE]
  exact val_le_of_strictMono _ (T.orderEmbOfFin hT).strictMono a

/-! ### The computable pieces of the kill -/

/-- The contribution `C(c-1, s-1)` of a column of sum `c`, with `c = 0` contributing `0`
(this matches the Python reference, which skips `c < 1`). Monotone in `c`. -/
def dTerm (s c : ℕ) : ℕ := if c = 0 then 0 else (c - 1).choose (s - 1)

theorem dTerm_mono (s : ℕ) : Monotone (dTerm s) := by
  intro c c' hcc'
  unfold dTerm
  by_cases hc : c = 0
  · simp only [hc, ↓reduceIte]; exact Nat.zero_le _
  · have hc' : c' ≠ 0 := by omega
    simp only [hc, hc', ↓reduceIte]
    exact Nat.choose_le_choose _ (by omega)

/-- The column sums sorted increasingly. -/
def sortedCols {n : ℕ} (col : Fin n → ℕ) : List ℕ := (List.ofFn col).mergeSort (· ≤ ·)

/-- `∑ dTerm s c` over the `r` lightest column sums. -/
def lightestSum {n : ℕ} (s r : ℕ) (col : Fin n → ℕ) : ℕ :=
  (((sortedCols col).take r).map (dTerm s)).sum

/-- Maximum over `Fin k`, Mathlib-free like `sumFin`. -/
def maxFin : (k : ℕ) → (Fin k → ℕ) → ℕ
  | 0, _ => 0
  | k + 1, f => max (f 0) (maxFin k (fun i => f i.succ))

theorem maxFin_succ (k : ℕ) (f : Fin (k + 1) → ℕ) :
    maxFin (k + 1) f = max (f 0) (maxFin k (fun i => f i.succ)) := rfl

theorem maxFin_eq_zero_or_exists : ∀ (k : ℕ) (f : Fin k → ℕ),
    maxFin k f = 0 ∨ ∃ i, f i = maxFin k f := by
  intro k
  induction k with
  | zero => intro f; exact Or.inl rfl
  | succ k ih =>
    intro f
    rw [maxFin_succ]
    rcases ih (fun i => f i.succ) with h0 | ⟨i, hi⟩
    · rw [h0]
      exact Or.inr ⟨0, (Nat.max_eq_left (Nat.zero_le _)).symm⟩
    · right
      rcases le_total (f 0) (f i.succ) with hle | hle
      · exact ⟨i.succ, by rw [← hi, Nat.max_eq_right hle]⟩
      · exact ⟨0, by rw [← hi, Nat.max_eq_left hle]⟩

theorem sortedCols_eq {n : ℕ} (col : Fin n → ℕ) :
    sortedCols col = List.ofFn (col ∘ Tuple.sort col) := by
  apply List.Perm.eq_of_sortedLE List.sortedLE_mergeSort
  · rw [List.sortedLE_iff_pairwise, List.pairwise_ofFn]
    intro i j hij
    exact Tuple.monotone_sort col hij.le
  · exact (List.mergeSort_perm _ _).trans (Equiv.Perm.ofFn_comp_perm _ _).symm

theorem lightestSum_eq {n : ℕ} (s r : ℕ) (col : Fin n → ℕ) :
    lightestSum s r col
      = ∑ k ∈ univ.filter (fun k : Fin n => k.val < r), dTerm s ((col ∘ Tuple.sort col) k) := by
  unfold lightestSum
  rw [sortedCols_eq, List.map_take, List.map_ofFn, List.sum_take_ofFn]
  rfl

theorem lightestSum_zero {n : ℕ} (s : ℕ) (col : Fin n → ℕ) : lightestSum s 0 col = 0 := by
  simp [lightestSum]

/-- The core of Argument D in sorted form: for every row `i`, the `dTerm`-sum of the
`rowSum A i` lightest columns is within the row-local budget. -/
theorem argD_core (P : Params) (A : Mat P.m P.n) (hfree : ¬ HasKst P A) (i : Fin P.m)
    (hs : 1 ≤ P.s) :
    lightestSum P.s (rowSum A i) (colSum A) ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1) := by
  rw [lightestSum_eq]
  have hg := Tuple.monotone_sort (colSum A)
  have hT : ((rowSupport A i).map (Tuple.sort (colSum A)).symm.toEmbedding).card = rowSum A i := by
    rw [card_map, card_rowSupport]
  have h1 := sum_lightest_le (colSum A ∘ Tuple.sort (colSum A)) hg (dTerm P.s) (dTerm_mono P.s) _ hT
  rw [sum_map] at h1
  refine h1.trans (le_trans (le_of_eq ?_) (rowLocalBudget P A hfree i hs))
  apply sum_congr rfl
  intro j hj
  simp only [Function.comp, Equiv.coe_toEmbedding, Equiv.apply_symm_apply]
  have hj' : A i j = true := (mem_filter.mp hj).2
  have hpos : 1 ≤ colSum A j := by
    rw [← card_support]
    exact card_pos.mpr ⟨i, (mem_support A j i).mpr hj'⟩
  have hne : colSum A j ≠ 0 := by omega
  simp only [dTerm, hne, ↓reduceIte]

/-- Argument D, row form, exact sorted-lightest-columns kill. -/
def argD (P : Params) : Prune P where
  name := "argD: row-local budget (sorted lightest columns)"
  kill := fun pf => decide (1 ≤ P.s ∧ P.s ≤ P.m ∧
    (P.t - 1) * (P.m - 1).choose (P.s - 1) < lightestSum P.s (maxFin P.m pf.row) pf.col)
  sound := by
    intro A h hv
    have h' := of_decide_eq_true h
    simp only [profileOf_row, profileOf_col] at h'
    obtain ⟨hs, _, hlt⟩ := h'
    rcases maxFin_eq_zero_or_exists P.m (rowSum A) with h0 | ⟨i, hi⟩
    · rw [h0, lightestSum_zero] at hlt
      exact Nat.not_lt_zero _ hlt
    · rw [← hi] at hlt
      exact absurd (argD_core P A hv.1 i hs) (not_le.mpr hlt)

/-- Argument D, column form (transpose). -/
def argDT (P : Params) : Prune P where
  name := "argDT: column-local budget (sorted lightest rows)"
  kill := fun pf => decide (1 ≤ P.t ∧ P.t ≤ P.n ∧
    (P.s - 1) * (P.n - 1).choose (P.t - 1) < lightestSum P.t (maxFin P.n pf.col) pf.row)
  sound := by
    intro A h hv
    have h' := of_decide_eq_true h
    simp only [profileOf_row, profileOf_col] at h'
    obtain ⟨ht, _, hlt⟩ := h'
    have hfree : ¬ HasKst P.transpose A.transpose :=
      fun hk => hv.1 ((hasKst_transpose P A).mp hk)
    rcases maxFin_eq_zero_or_exists P.n (colSum A) with h0 | ⟨j, hj⟩
    · rw [h0, lightestSum_zero] at hlt
      exact Nat.not_lt_zero _ hlt
    · rw [← hj] at hlt
      have := argD_core P.transpose A.transpose hfree j ht
      exact absurd this (not_le.mpr hlt)

/-- Argument A, column form. -/
def argA (P : Params) : Prune P where
  name := "argA: column budget"
  kill := fun pf => decide ((P.t - 1) * P.m.choose P.s < ∑ j, (pf.col j).choose P.s)
  sound := by
    intro A h hv
    have h' := of_decide_eq_true h
    simp only [profileOf_col] at h'
    exact absurd (colBudget P A hv.1) (not_le.mpr h')

/-- Argument A, row form. -/
def argAT (P : Params) : Prune P where
  name := "argAT: row budget"
  kill := fun pf => decide ((P.s - 1) * P.n.choose P.t < ∑ i, (pf.row i).choose P.t)
  sound := by
    intro A h hv
    have h' := of_decide_eq_true h
    simp only [profileOf_row] at h'
    exact absurd (rowBudget P A hv.1) (not_le.mpr h')

/-- All four counting prunes folded together. -/
def counting (P : Params) : Prune P :=
  Prune.ofList P [argA P, argAT P, argD P, argDT P]

/-! ### Axiom audit -/

#print axioms sumFin_eq_sum
#print axioms card_support
#print axioms hasKst_of_subsets
#print axioms sum_choose_le_of_bounded_fibers
#print axioms colBudget
#print axioms rowBudget
#print axioms rowLocalBudget
#print axioms sum_lightest_le
#print axioms sortedCols_eq
#print axioms lightestSum_eq
#print axioms argD_core
#print axioms argD
#print axioms argDT
#print axioms argA
#print axioms argAT
#print axioms counting

/-! ### Computability tests -/

abbrev testP : Params := { m := 9, n := 9, s := 3, t := 3, w := 50 }

def mkProfile (r c : List ℕ) : Profile testP.m testP.n :=
  { row := fun i => r.getD i.val 0, col := fun j => c.getD j.val 0 }

#eval sortedCols (fun j : Fin 9 => [6,6,6,6,6,5,5,5,5].getD j.val 0)
#eval lightestSum 3 6 (fun j : Fin 9 => [6,6,6,6,6,5,5,5,5].getD j.val 0)
#eval maxFin 9 (fun i : Fin 9 => [6,6,6,6,6,5,5,5,5].getD i.val 0)
-- (t-1)*C(m-1,s-1) = 2*28 = 56; 6 lightest of [5,5,5,5,6,6,6,6,6] -> 4*C(4,2)+2*C(5,2) = 24+20 = 44 -> no kill
#eval (argD testP).kill (mkProfile [6,6,6,6,6,5,5,5,5] [6,6,6,6,6,5,5,5,5])
#eval (argDT testP).kill (mkProfile [6,6,6,6,6,5,5,5,5] [6,6,6,6,6,5,5,5,5])
-- heaviest row 8, cols [7,7,7,7,7,7,7,7,7]: 8*C(6,2)=120 > 56 -> kill
#eval (argD testP).kill (mkProfile [8,6,6,6,6,5,5,5,5] [7,7,7,7,7,7,7,7,7])
-- sum C(6,3)*9 = 180 vs (t-1)*C(9,3)=168 -> kill
#eval (argA testP).kill (mkProfile [6,6,6,6,6,6,6,6,6] [6,6,6,6,6,6,6,6,6])
#eval (argAT testP).kill (mkProfile [6,6,6,6,6,6,6,6,6] [6,6,6,6,6,6,6,6,6])
#eval (argA testP).kill (mkProfile [6,6,6,6,6,5,5,5,5] [6,6,6,6,6,5,5,5,5])
#eval (counting testP).kill (mkProfile [6,6,6,6,6,5,5,5,5] [6,6,6,6,6,5,5,5,5])
#eval (counting testP).kill (mkProfile [8,6,6,6,6,5,5,5,5] [7,7,7,7,7,7,7,7,7])
#eval (argD testP).kill (mkProfile [0,0,0,0,0,0,0,0,0] [0,0,0,0,0,0,0,0,0])
-- columns include zeros: lightest 3 of [0,0,9,9,9,9,9,9,9] -> 0+0+C(8,2)=28 -> no kill
#eval (argD testP).kill (mkProfile [3,3,3,3,3,3,3,3,3] [0,0,9,9,9,9,9,9,9])
-- lightest 4 of [0,0,9,...] -> 2*28 = 56 -> not > 56, no kill; 5 -> 84 -> kill
#eval (argD testP).kill (mkProfile [4,3,3,3,3,3,3,3,3] [0,0,9,9,9,9,9,9,9])
#eval (argD testP).kill (mkProfile [5,3,3,3,3,3,3,3,3] [0,0,9,9,9,9,9,9,9])

end ZarPrune

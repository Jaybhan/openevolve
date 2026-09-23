# ZarPrune/Counting.lean — specification of the counting core

Goal: prove, in Lean 4.34 with targeted Mathlib imports, the double-counting
lemmas behind Guy's Arguments A and D, and package them as `Prune` terms so the
evolutionary search starts from a library in which the classical arguments are
*already* verified and reusable.  Everything must pass the gate: no `sorry`, no
`native_decide`, no new axioms; `#print axioms` ⊆ {propext, Quot.sound, Classical.choice}.

## Definitions to reuse (ZarPrune/Basic.lean, Mathlib-free)

```
abbrev Mat (m n) := Fin m → Fin n → Bool
def ind (b : Bool) : Nat := if b then 1 else 0
def rowSum A i := sumFin n (fun j => ind (A i j))       -- sumFin: recursive sum over Fin
def colSum A j := sumFin m (fun i => ind (A i j))
def weight A := sumFin m (rowSum A)
def Incr (f : Fin k → Fin N) : Prop := ∀ a b, a < b → f a < f b
def HasKst P A := ∃ R : Fin P.s → Fin P.m, ∃ C : Fin P.t → Fin P.n, Incr R ∧ Incr C ∧ ∀ a b, A (R a) (C b) = true
def Valid P A := ¬ HasKst P A ∧ P.w ≤ weight A
structure Profile m n := (row : Fin m → Nat) (col : Fin n → Nat)
structure Prune P := (name : String) (kill : Profile P.m P.n → Bool) (sound : ∀ A, kill (profileOf A) = true → ¬ Valid P A)
```

## Required theorems

Bridge (so Mathlib's `Finset.sum` can be used on ZarPrune sums):
```
theorem sumFin_eq_sum (k : ℕ) (f : Fin k → ℕ) : sumFin k f = ∑ i, f i
```
Supports:
```
def support {m n} (A : Mat m n) (j : Fin n) : Finset (Fin m) := Finset.univ.filter (fun i => A i j = true)
theorem card_support (A : Mat m n) (j) : (support A j).card = colSum A j
```
From a Finset of size s inside the supports of t distinct columns to `HasKst`
(use `Finset.orderEmbOfFin` to turn a finset of card s into a strictly
increasing `Fin s → Fin m`, which is `Incr`):
```
theorem hasKst_of_subsets (P) (A : Mat P.m P.n) (R : Finset (Fin P.m)) (hR : R.card = P.s)
    (C : Finset (Fin P.n)) (hC : C.card = P.t) (h : ∀ j ∈ C, R ⊆ support A j) : HasKst P A
```
**Argument A (column budget)** — the theorem the whole method rests on:
```
theorem colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) :
    ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s
```
Proof sketch: `(colSum A j).choose s = ((support A j).powersetCard s).card`
(`Finset.card_powersetCard`); rewrite as `∑ j, ∑ R ∈ (univ : Finset (Fin m)).powersetCard s, if R ⊆ support A j then 1 else 0`;
swap sums (`Finset.sum_comm`); for each `R` with `R.card = s` the inner count
`(univ.filter (fun j => R ⊆ support A j)).card ≤ t - 1`, else pick `t` such
columns (`Finset.exists_subset_card_eq` / `Finset.exists_smaller_set`) and
apply `hasKst_of_subsets`; finally `Finset.sum_le_sum` and `card_powersetCard`
on `univ : Finset (Fin m)` gives `(m).choose s`.

**Row budget** (transpose): `∑ i, (rowSum A i).choose P.t ≤ (P.s - 1) * (P.n).choose P.t`.
Either re-run the argument with roles swapped, or prove a transpose lemma
`HasKst P A ↔ HasKst P.transpose Aᵀ` — the direct proof is usually shorter.

**Argument D (row-local budget)**:
```
theorem rowLocalBudget (P) (A) (h : ¬ HasKst P A) (i : Fin P.m) (hs : 1 ≤ P.s) :
    ∑ j ∈ Finset.univ.filter (fun j => A i j = true), (colSum A j - 1).choose (P.s - 1)
      ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1)
```
Proof sketch: same double counting over pairs (R, j) with `R ⊆ (support A j).erase i`,
`R.card = s - 1`, ranging over `R ∈ (univ.erase i).powersetCard (s-1)`
(card = `(m-1).choose (s-1)`); a column containing `i` has
`((support A j).erase i).card = colSum A j - 1`; if `t` columns contain
`R ∪ {i}` (card s) we get `HasKst`.

## Required Prune terms (profile-level kills; these are what the search reuses)

```
def argA (P : Params) : Prune P            -- kill pf := decide ((∑ j, (pf.col j).choose P.s) > (P.t - 1) * P.m.choose P.s)
def argAT (P : Params) : Prune P           -- row-side version
```
Soundness is immediate from colBudget since `pf.col j = colSum A j` when `pf = profileOf A`.

```
def argD (P : Params) : Prune P
def argDT (P : Params) : Prune P
```
The Python reference (`zar_ub/cases.py::kill_row_argument_d`) kills when, for
the heaviest row r = max pf.row, the r *lightest* column sums c satisfy
`∑ C(c-1, s-1) > (t-1)·C(m-1, s-1)`.  Two sound formulations, pick either
(or both, `Prune.or`):

* **exact (sorted) form**: sort the column sums (`List.mergeSort` / `Finset.sort`),
  take the first r, sum `(c-1).choose (s-1)`. Soundness needs the exchange
  lemma "for any r-subset S of columns, `∑_{j∈S} f (c j) ≥` sum of f over the r
  smallest values" for monotone f — provable via the layer-cake identity
  `∑_{j∈S} f(c_j) = ∑_{x ≥ 1} |{j ∈ S : c_j ≥ x}| · (f x - f (x-1))` and
  `|{j ∈ S : c_j ≥ x}| ≥ r - |{j : c_j < x}|`.
* **threshold form** (weaker but much easier): for any threshold x,
  at most `k_x := |{j : pf.col j < x}|` of the row's r columns are lighter than x,
  so `∑_{j ∈ row} (c_j - 1).choose (s-1) ≥ (r - k_x) · (x-1).choose (s-1)`.
  kill pf := decide (∃ x ∈ [1..m+1], (r - k_x) · (x-1).choose (s-1) > (t-1)·(m-1).choose (s-1)).
  Also sound (and stronger) is the sum over all thresholds
  `∑_{x=1}^{m} (r - k_x)⁺ · ((x-1).choose (s-1) - (x-2).choose (s-1))`, which equals the exact form.

Acceptance for `argD`: on the cached case tables (`cache/*.json`), the Lean
`#eval` kill mask must agree with `kill_row_argument_d` on every case for the
exact form; the threshold form must kill a subset of those cases.

## Deletion lemma (bonus, very reusable)

```
def deleteCol (A : Mat m (n+1)) (j : Fin (n+1)) : Mat m n := fun i k => A i (j.succAbove k)
theorem weight_deleteCol : weight (deleteCol A j) + colSum A j = weight A
theorem not_hasKst_deleteCol (h : ¬ HasKst ⟨m, n+1, s, t, w⟩ A) : ¬ HasKst ⟨m, n, s, t, w'⟩ (deleteCol A j)
```
(`Fin.succAbove` is strictly monotone, so `Incr C → Incr (j.succAbove ∘ C)`.)
With the column budget this yields the prune "`w - min_j pf.col j` exceeds the
counting bound of the (m, n-1) instance", once `colBudget` is combined with a
Lean-computable bound `countingBound m n s t` and a proof that
`∑ j, (c j).choose s ≤ B → ∑ j, c j ≤ countingBound` (waterfilling optimality;
harder — optional).

## Style rules
* Put everything in `namespace ZarPrune`, file `ZarPrune/Counting.lean`,
  imported by `ZarPrune.lean`. Add the proved prunes to a new list
  `def counting (P) : Prune P := Prune.ofList P [argA P, argAT P, argD P, argDT P]`.
* Only targeted Mathlib imports (no `import Mathlib`), e.g. `Mathlib.Data.Finset.Basic`,
  `Mathlib.Algebra.BigOperators.Group.Finset.Basic`, `Mathlib.Data.Nat.Choose.Basic`,
  `Mathlib.Data.Finset.Powerset`, `Mathlib.Data.Fintype.Card`, `Mathlib.Data.Finset.Sort`,
  `Mathlib.Order.Fin.Basic`, `Mathlib.Tactic.Linarith`, `Mathlib.Tactic.Positivity`, `Mathlib.Tactic.Ring`.
* `lake build` must succeed; run `#print axioms` for every new theorem and prune.
* Keep `kill` functions *computable and fast* (they are `#eval`ed on thousands of profiles): use
  `List.range`, `List.foldl`, `decide` on `Nat` comparisons; avoid `Finset` in kill bodies
  (Finset.sum over `Fin n` does compute, but `Finset.univ.filter` with `decide` inside is fine too — test with `#eval`).

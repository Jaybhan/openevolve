# COUNTING_NOTES.md — how `ZarPrune/Counting.lean` is proved

Companion to `COUNTING_SPEC.md` (the contract). This records the proof strategy
actually used, where each piece came from, and how it was validated. Lean 4.34.0,
Mathlib v4.34.0, targeted imports only (no `import Mathlib`).

## Provenance

Eight independent attempts were written against the spec (`Attempts/A1–A3.lean`
for Argument A, `D1–D3.lean` for Argument D, `E1–E2.lean` for deletion /
waterfilling). All eight were re-elaborated with `lake env lean` before anything
was trusted; all eight are sorry-free and axiom-clean. `Counting.lean` takes:

| Piece | Source | Why this one |
|---|---|---|
| `sumFin_eq_sum`, `support`/`rowSupport` + `card_*`, `hasKst_of_subsets` | D3 | shortest statements; `card_support` closes by `rfl` after `Finset.card_filter` |
| `card_powersetCard_eq_sum`, `budget_general` | D1 | one generic double-counting lemma; `colBudget` and `rowLocalBudget` are 15-line instances |
| `colBudget`, `rowLocalBudget` | D1 (restated with `rowSupport`) | both via `budget_general` |
| transpose infrastructure, `rowBudget` | D1/A3/D3 merged | `hasKst_transpose` as an `↔`; new `Prune.transposed` combinator |
| `argA`, `argAT` | A3 | `sumFin` in the kill, bridged by `sumFin_eq_sum`, closed by `omega` |
| `fD`/`gD`/`cntLt`/`boundD`, layer-cake lemmas, `argD`, `argDT` | D3 | exact form **without sorting** (D2's `mergeSort` form is equivalent but needs `Tuple.sort` machinery) |
| `deleteCol`/`deleteRow`, `weight_delete*`, `not_hasKst_delete*` | E1 | via a single `sumFin_succAbove` lemma; row version is one line |
| `choose_tangent`, `equalCost`, `waterfillBound`, `sum_le_waterfillBound` | E2 | full waterfilling optimality, no exchange argument |
| `argDelCol`/`argDelRow` (parametrised), `argDelColWF`/`argDelRowWF`, `argWF` | E2 | prunes parametrised by any proved bound `U` |

Not integrated (superseded, still available in `Attempts/`): A2's explicit
incidence-set route to `colBudget`; D1's threshold form `argD`/`argDT` and
`argDsum`; D2's sorted `lightestSum` form; E1's `minFin`-based `delMinCol`.

## Strategy, piece by piece

### Bridge
`sumFin_eq_sum : sumFin k f = ∑ i, f i` by induction on `k` with `sumFin_succ` and
`Fin.sum_univ_succ`. Everything Mathlib-side is stated with `Finset.sum`; every
`kill` function is stated with `sumFin`/`allFin` (fast, Mathlib-free) and the
soundness proof rewrites across with this lemma. `sumFin_succAbove` (from
`Fin.sum_univ_succAbove`) is the only other bridge needed, for deletion.

### Supports and `HasKst`
`support A j = univ.filter (A · j = true)`, `card_support` via `Finset.card_filter`
and `rfl` (`ind b` unfolds to `if b = true then 1 else 0`). `hasKst_of_subsets`:
given `R.card = s`, `C.card = t`, `∀ j ∈ C, R ⊆ support A j`, the witnesses are
`R.orderEmbOfFin hR` and `C.orderEmbOfFin hC`; `Incr` is `.strictMono`, membership
is `Finset.orderEmbOfFin_mem`.

### The one double-counting lemma
```
budget_general (T J S) (hS : ∀ j ∈ J, S j ⊆ T) (k B)
  (hB : ∀ R ∈ T.powersetCard k, (J.filter (R ⊆ S ·)).card ≤ B) :
  ∑ j ∈ J, (S j).card.choose k ≤ B * T.card.choose k
```
Proof: `C(|S j|, k) = #powersetCard k (S j) = ∑_{R ∈ T.powersetCard k} [R ⊆ S j]`
(`card_powersetCard`, `card_filter`, and an `ext` on `mem_powersetCard`), swap the
sums (`Finset.sum_comm`), each inner sum is a filter card ≤ B (`sum_le_sum`),
finish with `sum_const` and `card_powersetCard`.

* **`colBudget`** = instance `T = univ`, `J = univ`, `S = support A`, `k = s`,
  `B = t-1`. If some `R` were in ≥ t columns, `Finset.exists_subset_card_eq` picks
  `t` of them and `hasKst_of_subsets` gives a `K_{s,t}`.
* **`rowLocalBudget`** = instance `T = univ.erase i`, `J = rowSupport A i`,
  `S j = (support A j).erase i`, `k = s-1`, `B = t-1`. `|S j| = colSum A j - 1`
  by `card_erase_of_mem`; if `t` columns contain `R` then `insert i R` (card `s`,
  `card_insert_of_notMem`) is inside their supports → `K_{s,t}`.
* **`rowBudget`** = `colBudget P.transpose (transpose A)`; the statement is
  definitionally the transposed one, so `exact` closes it.

### Transposition
`Params.transpose = ⟨n, m, t, s, w⟩`, `transpose A = fun j i => A i j`,
`Profile.swap`. `rowSum (transpose A) = colSum A` and
`profileOf (transpose A) = (profileOf A).swap` are `rfl`; `hasKst_transpose` is a
pure swap of witnesses (`↔`, axiom-free); `valid_transpose` follows.
**`Prune.transposed : Prune P.transpose → Prune P`** with
`kill pf := q.kill pf.swap` — any prune proved on one side is available on the
other for free (`argDT := (argD P.transpose).transposed`). Evolved prunes get this
too. Gotcha: `rw` on `weight (transpose A)` fails with a motive type mismatch
(`Mat P.transpose.n P.transpose.m` vs `Mat P.n P.m`); use term-mode `exact`, the
two are defeq.

### Argument D at the profile level (layer-cake)
The Python reference sorts the column sums and adds `C(c-1, s-1)` over the `r`
lightest. The Lean kill computes the same number without sorting:
```
fD s c   := (c-1).choose (s-1)
gD s x   := fD s x - fD s (x-1)                     -- increment, ≥ 0 by monotonicity
cntLt pf y := #{j | pf.col j < y}
boundD P pf r := r * fD s 0 + ∑_{x < m} (r - cntLt pf (x+1)) * gD s (x+1)
```
`fD_layer_cake`: `fD c = fD 0 + ∑_{x<c} gD (x+1)` (telescoping; `Nat.choose_le_choose`
for monotonicity — omega wants `Nat.le_add_right c 1`, not `Nat.le_succ`, to avoid a
`c.succ` atom). `sum_fD_eq`: summing over the row's columns and swapping gives
`∑_x #{j ∈ row : c_j > x} · gD (x+1)`. `card_filter_ge`: at most `cntLt (x+1)` of
the row's `r` columns have `c_j ≤ x`, so `#{j ∈ row : c_j > x} ≥ r - cntLt (x+1)`
(`card_filter_add_card_filter_not`, `card_le_card`). Hence
`boundD_le : boundD P (profileOf A) (rowSum A i) ≤ ∑_{j ∈ row i} fD s (colSum A j)`,
and `argD` kills when some row's `boundD` exceeds `(t-1) C(m-1, s-1)`; checking
every row is equivalent to checking the heaviest because `boundD` is monotone in
`r`. The guard `1 ≤ s` is needed so `insert i R` has card `s`.
`boundD` equals the sorted-lightest-columns sum exactly (checked on 13,903 real
cases and, by D3, on 20,000 random profiles); for `s = 1` Lean counts a zero-sum
column as contributing `C(0,0) = 1` where Python skips it — Lean is the sound one.

### Deletion
`deleteCol A j = fun i k => A i (j.succAbove k)`. `weight_deleteCol` is
`sumFin_succAbove` applied to `colSum A` (after `weight_eq_sum_colSum`);
`weight_deleteRow` is the same lemma applied to `rowSum A` directly.
`Incr.succAbove_comp` from `Fin.succAbove_lt_succAbove_iff`; a `K_{s,t}` in the
deleted matrix maps back through `succAbove`. The weight field of `Params` is
irrelevant to `HasKst`, so the lemmas take independent `w w'`.

### Waterfilling
`choose_tangent s a x : C(a,s+1) + x·C(a,s) ≤ C(x,s+1) + a·C(a,s)` is the discrete
tangent line of the convex sequence `x ↦ C(x, s+1)` at `a`; induction on `|x-a|`
with `Nat.choose_succ_succ` and `nlinarith`. Summing over columns at `a = S/n` gives
`equalCost n s S ≤ ∑ C(c_j, s+1)` where `equalCost` is the cost of the as-equal-
as-possible distribution. `waterfillBound m n s B := Nat.findGreatest (equalCost n (s-1) · ≤ B) (n·m)`
and `sum_le_waterfillBound` is `Nat.le_findGreatest`. `weight_le_waterfill`
plugs in `colBudget`. E2 checked by brute force that the bound is tight on tiny
instances (achievability is not formalised; not needed for soundness).

### Deletion prunes
`argDelCol P U hU` kills when `∃ j, pf.col j + U < w`, for **any** `U` with a proof
that every `K_{s,t}`-free `m × (n-1)` matrix has weight ≤ U — so a table of proved
`z(m, n-1; s, t)` values can be plugged in. Soundness: destructure `P`, `cases n`
(the `n = 0` case is `Fin.elim0`), then `valid_deleteCol_bound` (`weight_deleteCol`
+ `not_hasKst_deleteCol` + omega). `argDelColWF` instantiates `U` with the
waterfilled column budget of the smaller instance. On consistent profiles these
are dominated by `argA` (dropping a column only lowers `∑ C(c_j, s)`); their value
is with a sharper `U`.

## Kill functions
All computable, no Mathlib in the hot path except `Finset.range` sums and
`univ.filter` cards on `Fin` in `boundD` (both `#eval` fine; `decide` also works on
small instances). `argA`/`argAT`: `O(n)`; `argD`: `O(m² n)`; `argDelColWF`: one
`Nat.findGreatest` over `n·m` values when the term is built, then `O(n)` per call.

## Validation
* `lake build`: 3.1 s wall (Counting.lean 1.3 s); `lake env lean ZarPrune/Counting.lean`: 1.9 s.
* `Attempts/CountingAudit.lean`: `#print axioms` for all 75 new items — every one
  is within `{propext, Quot.sound, Classical.choice}`; the pure transposition /
  deletion lemmas need only `propext` (and `Quot.sound`), `hasKst_transpose`,
  `deleteCol`, `waterfillBound` use no axioms at all. Plus `#eval` smoke tests
  (m = n = 9 hand cases, three real m10n11 cases, waterfill 40/38, degenerate
  `s = 0`, `m = n = 0`).
* Python cross-check (scratch script `xcheck_counting.py`): on **all 29 cached case
  tables, 13,903 cases**, the Lean masks for `argD`/`argDT` agree with
  `kill_row_argument_d`/`kill_col_argument_d` on every case (5,208 / 2,777 kills),
  and `argA`/`argAT` agree with the KST inequality (0 kills on these tables, since
  Argument D subsumes them there).

## Mathlib name gotchas (v4.34.0)
`Finset.sum_comm` → `...Group.Finset.Sigma`; `Finset.card_filter` → `...Group.Finset.Piecewise`;
`Finset.sum_le_sum` → `Mathlib.Algebra.Order.BigOperators.Group.Finset`;
`Fin.sum_univ_succ` → `Mathlib.Algebra.BigOperators.Fin`; `Nat.findGreatest` → `Mathlib.Data.Nat.Find`.
Gone: `Finset.exists_smaller_set` (use `exists_subset_card_eq`), `List.sorted_mergeSort`,
`Finset.filter_card_add_filter_neg_card_eq_card` (use `card_filter_add_card_filter_not`),
`Finset.card_insert_of_not_mem` (now `_notMem`). Deprecated: `push_neg` (`push Not`),
`if_pos/if_neg` (`split_ifs`/`simp only [h, ↓reduceIte]`).

## Not proved / open
* Achievability of `waterfillBound` (tightness) — not needed for soundness.
* Nothing in the spec is left unproved. Beyond the spec: no prune yet uses a
  *table* of exact `z` values for `argDelCol`'s `U`; and the partition-level
  (unordered) cover step is still outside this library (see README "Scope").

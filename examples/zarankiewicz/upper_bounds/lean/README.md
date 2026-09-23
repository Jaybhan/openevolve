# ZarPrune — a Lean verifier for Zarankiewicz pruning arguments

Lean 4.34.0, no `sorry`, no `native_decide`. The core is Mathlib-free; only
`ZarPrune/Counting.lean` (the counting arguments) uses targeted Mathlib imports.
Full build in about 3 s with prebuilt Mathlib oleans: `lake build`.

## What this is

The gate for the *pruning* half of the upper-bound pipeline. A candidate prune —
proposed by the evolutionary search, or by hand — is accepted exactly when it can
be given as a term of type `Prune P`:

```lean
structure Prune (P : Params) where
  name  : String := ""
  kill  : Profile P.m P.n → Bool
  sound : ∀ A : Mat P.m P.n, kill (profileOf A) = true → ¬ Valid P A
```

`kill` is what the harness runs on every case before emitting a SAT instance.
`sound` is the entire proof obligation: **a killed case contains no valid matrix.**
If that field elaborates, the prune is accepted; if it does not, it is rejected.
There is no third outcome and no trusted escape hatch.

## The files

| File | Contents |
|---|---|
| `ZarPrune/Sum.lean` | `sumFin` / `allFin` over `Fin k`, Fubini (`sumFin_swap`), bounds |
| `ZarPrune/Basic.lean` | `Params`, `Mat`, `HasKst`, `Valid`, `Profile`, `profileOf` |
| `ZarPrune/Prune.lean` | the `Prune` gate, combinators, `skip`, `upper_bound_of_cover` |
| `ZarPrune/Prunes.lean` | four proved baseline prunes + `baseline` |
| `ZarPrune/Counting.lean` | Guy's Arguments A and D, transposition, deletion, waterfilling, `counting` (Mathlib) |
| `Attempts/CountingAudit.lean` | `#print axioms` for every item of `Counting.lean` + `#eval` smoke tests |
| `ZarPrune/Demo.lean` | tests, including the negative one |

## The two theorems that matter

`Prune.skip` — if the prune fires on a case, every matrix in that case is already
refuted, so the harness never builds or solves that instance.

`upper_bound_of_cover` — the closure. Given a verified prune, the list of cases
that survived it, a cover-completeness obligation, and one refutation per
survivor, it concludes

```lean
∀ A : Mat P.m P.n, ¬ HasKst P A → weight A < P.w
```

which is `z(m,n;s,t) < w`. `upper_bound_succ_of_cover` gives the `≤ w-1` form used
in the bound tables.

The two hypotheses are the seams to the rest of the pipeline. `refuted` is where a
checked LRAT/SR certificate plus an encoding-correctness theorem gets discharged;
`cover` is where the decomposition's completeness certificate goes.

## Pruning is not adding — and the verifier enforces it

`Demo.notDescending_unsound` is the negative test, and it is the point of the
whole file:

```lean
theorem notDescending_unsound : ¬ ∃ p : Prune demoP, p.kill = notDescending
```

"Assume the rows are sorted by sum, kill the cases where they are not" is the
reflex symmetry-breaking move. It is *provably not a prune*: `badA` is a valid
matrix living in a case it kills. Sorting is sound only as an **addition**
justified by a row-permutation witness — an SR/VeriPB redundancy step, which lives
in the certificate, not here. A prune may only kill cases that are genuinely empty.

Any evolved candidate that is really a disguised symmetry break will fail to
elaborate here, by construction.

## Proved ledger

Proved, with the obligation discharged (Mathlib-free core, `ZarPrune/Prunes.lean`):

- `deficit` — row sums total less than `w`
- `mismatch` — row-sum total ≠ column-sum total (via Fubini)
- `rowCap` / `colCap` — a row sum exceeding `n`, or a column sum exceeding `m`
- `baseline` — the four folded together, via `Prune.or`

Axiom audit: everything above depends only on `propext` and `Quot.sound`. No
`Classical.choice`.

Proved in `ZarPrune/Counting.lean` (Mathlib; strategy in `COUNTING_NOTES.md`,
axiom audit in `Attempts/CountingAudit.lean` — every item within
`{propext, Quot.sound, Classical.choice}`):

- `sumFin_eq_sum` — bridge from the Mathlib-free `sumFin` to `Finset.sum`
- `support` / `rowSupport`, `card_support` / `card_rowSupport` — column/row supports as finsets
- `hasKst_of_subsets` — an `s`-set of rows inside the supports of `t` columns is a `K_{s,t}`
- `budget_general` — the one double-counting lemma (`∑_{j∈J} C(|S j|,k) ≤ B·C(|T|,k)`)
- `colBudget` — **Argument A / KST**: `∑_j C(c_j, s) ≤ (t-1)·C(m, s)` for `K_{s,t}`-free
- `rowBudget` — its transpose `∑_i C(r_i, t) ≤ (s-1)·C(n, t)`
- `rowLocalBudget` — **Argument D**: `∑_{j∋i} C(c_j-1, s-1) ≤ (t-1)·C(m-1, s-1)` per row
- `hasKst_transpose`, `valid_transpose`, `Prune.transposed` — any prune on `Pᵀ` is a prune on `P`
- `argA` / `argAT` — Argument A as prunes (column / row side)
- `boundD_le`, `argD` / `argDT` — Argument D as prunes, exact "r lightest columns" form
  computed by layer-cake without sorting; agrees with `zar_ub/cases.py` on all
  13,903 cached cases
- `deleteCol` / `deleteRow`, `weight_deleteCol` / `weight_deleteRow`,
  `not_hasKst_deleteCol` / `not_hasKst_deleteRow` — deleting a line drops exactly its sum
  and cannot create a `K_{s,t}`
- `choose_tangent`, `equalCost_le_sum_choose`, `sum_le_waterfillBound`,
  `weight_le_waterfill` — waterfilling: `∑ C(c_j,s) ≤ B ⇒ ∑ c_j ≤ waterfillBound`
- `argDelCol` / `argDelRow` — deletion prunes parametrised by any proved bound `U`
  on the instance with one line fewer; `argDelColWF` / `argDelRowWF` / `argWF`
  instantiate `U` by waterfilling
- `countingA`, `countingD`, `deletion`, `counting` — bundles; `counting P` folds all
  seven counting prunes

Not proved:

- tightness (achievability) of `waterfillBound` — not needed for soundness, checked by
  brute force on tiny instances only
- no prune yet plugs a *table* of proved exact `z(m, n-1; s, t)` values into
  `argDelCol`'s `U`; that instantiation is a one-liner once such a table is verified
- the partition-level (unordered) cover step, see Scope below

## Scope

Row/column **sum vectors**, not the unordered partitions the harness will actually
enumerate. A prune on profiles restricts to a prune on partitions, so this is the
more general side, but the enumerate-partitions-and-cover step is not written yet
and is the other half of `cover`.

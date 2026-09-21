# ZarPrune — a Lean verifier for Zarankiewicz pruning arguments

Lean 4.34.0, no Mathlib, no `sorry`, no `native_decide`.
Full build in well under a second: `lake build`.

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

Proved, with the obligation discharged:

- `deficit` — row sums total less than `w`
- `mismatch` — row-sum total ≠ column-sum total (via Fubini)
- `rowCap` / `colCap` — a row sum exceeding `n`, or a column sum exceeding `m`
- `baseline` — the four folded together, via `Prune.or`

Axiom audit: everything above depends only on `propext` and `Quot.sound`. No
`Classical.choice`.

Not proved, and the next target:

- **the Kővári–Sós–Turán counting prune**, `Σᵢ C(rᵢ, t) ≤ (s-1)·C(n, t)` for a
  `K_{s,t}`-free matrix, and its column dual. This is the real prune — Guy's
  counting arguments are refinements of it — and it is what actually kills
  partition pairs at scale. It needs double counting over `t`-subsets of columns,
  which this Mathlib-free core does not have; it wants either `Finset` or a
  hand-rolled subset-counting layer. Nothing else in the design blocks on it.

## Scope

Row/column **sum vectors**, not the unordered partitions the harness will actually
enumerate. A prune on profiles restricts to a prune on partitions, so this is the
more general side, but the enumerate-partitions-and-cover step is not written yet
and is the other half of `cover`.

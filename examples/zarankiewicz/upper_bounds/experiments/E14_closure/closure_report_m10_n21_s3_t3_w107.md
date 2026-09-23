# Closure report: z(10,21;3,3) <= 106

Generated 2026-09-21 23:42.  **Tier 1**.

## Status

- refuted by verified LRAT certificate: 0
- open: 0
- Lean closure file: `lean/ZarPrune/Closures/Z_10_21_107.lean` — checked in 3.33 s, axioms ['propext', 'Classical.choice', 'Quot.sound']
- conditional on 18 external fact(s) (hypotheses of the theorem): tan2022:z(7,21;3,3)<=78; tan2022:z(8,21;3,3)<=87; tan2022:z(9,21;3,3)<=96; tan2022:z(10,6;3,3)<=39; tan2022:z(10,7;3,3)<=44; tan2022:z(10,8;3,3)<=50; tan2022:z(10,9;3,3)<=54; tan2022:z(10,10;3,3)<=60; tan2022:z(10,11;3,3)<=64; tan2022:z(10,12;3,3)<=68; tan2022:z(10,13;3,3)<=73; tan2022:z(10,14;3,3)<=77; tan2022:z(10,15;3,3)<=81; tan2022:z(10,16;3,3)<=85; tan2022:z(10,17;3,3)<=90; tan2022:z(10,18;3,3)<=94; tan2022:z(10,19;3,3)<=98; lean-here:z(10,20;3,3)<=102
- **claim established: YES**

## Notes

- no cached table for this instance/trust (no cross-check)

## Trusted base (design §3.4)

| # | component | tier | used here |
|---|---|---|---|
| T1 | Lean 4.34.0 kernel; `leanchecker` replay on closure files and on `Evolved.lean` | 0, 1, 1n | yes |
| T2 | `ZarPrune/Basic.lean` (~60 lines: the statement) | 0, 1, 1n | yes |
| T3 | Mathlib v4.34.0 modules imported by `Counting.lean` / `Closure.lean` | 0, 1, 1n | yes |
| T4 | `Closure.lean` (thinning, `act`, sorting, `genParts` + `mem_genParts`, `survivors`, `upper_bound_succ_of_sorted_cover`), `Schemas.lean` — harness-owned, human-audited at statement level | 0, 1, 1n | yes |
| T4n | `Lean.ofReduceBool` (one named `_native` axiom per `survivors_eq` discharged by `decide +native`) | 1n | — |
| T5 | `drat-trim` + `lrat-check` verdicts, each a named hypothesis `Hrefuted` paired with `cnf_sha1`/`lrat_sha1` in `manifest.json`; `encoding.py` completeness (case SAT ⇐ matrix exists) | 1, 1n | yes |
| T5′ | `Encode.lean` completeness theorem + `LRAT.check_sound` evaluated natively (named `_native` axiom per branch) | 0 | — |
| T6 | `exists_doubleLex` (block double-lex reachable inside a sorted case) | 0 | — |
| T7 | external facts, each a `Fact` hypothesis with provenance tag | 0, 1, 1n | yes |
| T8 | Tier-2 only: `partitions.py` completeness and Python cover (no Lean closure file; no bound is claimed) | 2 | — |

## Facts used by the enumerator (`genRows P facts`, `genCols P facts`)

| fact | provenance | discharged by |
|---|---|---|
| z(3,21;3,3) ≤ 44 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(4,21;3,3) ≤ 50 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(5,21;3,3) ≤ 62 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(6,21;3,3) ≤ 69 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(7,21;3,3) ≤ 79 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(8,21;3,3) ≤ 88 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(9,21;3,3) ≤ 98 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,3;3,3) ≤ 22 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(10,4;3,3) ≤ 28 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(10,5;3,3) ≤ 33 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(10,6;3,3) ≤ 40 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(10,7;3,3) ≤ 45 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(10,8;3,3) ≤ 51 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(10,9;3,3) ≤ 56 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(10,10;3,3) ≤ 62 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,11;3,3) ≤ 67 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,12;3,3) ≤ 72 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,13;3,3) ≤ 76 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,14;3,3) ≤ 80 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,15;3,3) ≤ 84 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,16;3,3) ≤ 88 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,17;3,3) ≤ 92 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,18;3,3) ≤ 96 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,19;3,3) ≤ 100 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,20;3,3) ≤ 104 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(7,21;3,3) ≤ 78 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(8,21;3,3) ≤ 87 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(9,21;3,3) ≤ 96 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,6;3,3) ≤ 39 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,7;3,3) ≤ 44 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,8;3,3) ≤ 50 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,9;3,3) ≤ 54 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,10;3,3) ≤ 60 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,11;3,3) ≤ 64 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,12;3,3) ≤ 68 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,13;3,3) ≤ 73 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,14;3,3) ≤ 77 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,15;3,3) ≤ 81 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,16;3,3) ≤ 85 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,17;3,3) ≤ 90 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,18;3,3) ≤ 94 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,19;3,3) ≤ 98 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,20;3,3) ≤ 102 | lean-here | hypothesis `FactHolds` of the closure theorem |

Axioms of `ZarPrune.Closures.Z_10_21_107.z_10_21_le_106`: ['propext', 'Classical.choice', 'Quot.sound']

## Prune (Lean source)

```lean
-- prune term: Prune.or (baseline P) (counting P)
```

## Cases handed to the solver

| rows | cols | disposition | certificate |
|---|---|---|---|

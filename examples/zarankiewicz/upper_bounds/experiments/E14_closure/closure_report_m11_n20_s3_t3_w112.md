# Closure report: z(11,20;3,3) <= 111

Generated 2026-09-21 23:42.  **Tier 1**.

## Status

- refuted by verified LRAT certificate: 0
- open: 0
- Lean closure file: `lean/ZarPrune/Closures/Z_11_20_112.lean` — checked in 3.39 s, axioms ['propext', 'Classical.choice', 'Quot.sound']
- conditional on 16 external fact(s) (hypotheses of the theorem): tan2022:z(7,20;3,3)<=75; tan2022:z(8,20;3,3)<=84; tan2022:z(9,20;3,3)<=93; lean-here:z(10,20;3,3)<=102; tan2022:z(11,7;3,3)<=47; tan2022:z(11,8;3,3)<=53; tan2022:z(11,9;3,3)<=59; tan2022:z(11,10;3,3)<=64; tan2022:z(11,11;3,3)<=69; tan2022:z(11,12;3,3)<=74; tan2022:z(11,13;3,3)<=80; tan2022:z(11,14;3,3)<=84; tan2022:z(11,15;3,3)<=88; tan2022:z(11,16;3,3)<=92; tan2022:z(11,17;3,3)<=96; tan2022:z(11,18;3,3)<=101
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
| z(3,20;3,3) ≤ 42 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(4,20;3,3) ≤ 48 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(5,20;3,3) ≤ 60 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(6,20;3,3) ≤ 66 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(7,20;3,3) ≤ 76 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(8,20;3,3) ≤ 85 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(9,20;3,3) ≤ 94 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(10,20;3,3) ≤ 104 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(11,3;3,3) ≤ 24 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(11,4;3,3) ≤ 30 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(11,5;3,3) ≤ 36 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(11,6;3,3) ≤ 42 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(11,7;3,3) ≤ 48 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(11,8;3,3) ≤ 55 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(11,9;3,3) ≤ 60 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(11,10;3,3) ≤ 67 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(11,11;3,3) ≤ 73 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(11,12;3,3) ≤ 78 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(11,13;3,3) ≤ 82 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(11,14;3,3) ≤ 87 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(11,15;3,3) ≤ 92 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(11,16;3,3) ≤ 96 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(11,17;3,3) ≤ 101 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(11,18;3,3) ≤ 105 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(11,19;3,3) ≤ 109 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(7,20;3,3) ≤ 75 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(8,20;3,3) ≤ 84 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(9,20;3,3) ≤ 93 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(10,20;3,3) ≤ 102 | lean-here | hypothesis `FactHolds` of the closure theorem |
| z(11,7;3,3) ≤ 47 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,8;3,3) ≤ 53 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,9;3,3) ≤ 59 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,10;3,3) ≤ 64 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,11;3,3) ≤ 69 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,12;3,3) ≤ 74 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,13;3,3) ≤ 80 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,14;3,3) ≤ 84 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,15;3,3) ≤ 88 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,16;3,3) ≤ 92 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,17;3,3) ≤ 96 | tan2022 | hypothesis `FactHolds` of the closure theorem |
| z(11,18;3,3) ≤ 101 | tan2022 | hypothesis `FactHolds` of the closure theorem |

Axioms of `ZarPrune.Closures.Z_11_20_112.z_11_20_le_111`: ['propext', 'Classical.choice', 'Quot.sound']

## Prune (Lean source)

```lean
-- prune term: Prune.or (baseline P) (counting P)
```

## Cases handed to the solver

| rows | cols | disposition | certificate |
|---|---|---|---|

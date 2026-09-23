# Closure report: z(9,9;3,3) <= 49

Generated 2026-09-22 00:25.  **Tier 1**.

## Status

- admissible cases (cached table): 36 (6 row partitions x 6 column partitions)
- pruned by Lean-verified prune: 19
- refuted by verified LRAT certificate: 17
- open: 0
- Lean closure file: `experiments/E15_e2e/closures_abs/Z_9_9_50.lean` — checked in 2.29 s, axioms ['propext', 'Classical.choice', 'Quot.sound']
- external facts: none (the theorem is unconditional beyond the LRAT verdicts)
- **claim established: YES**

## Notes

- Lean survivors == cached table survivors (baseline_lean_mask): 17 cases

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
| z(3,9;3,3) ≤ 20 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(4,9;3,3) ≤ 26 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(5,9;3,3) ≤ 30 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(6,9;3,3) ≤ 36 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(7,9;3,3) ≤ 41 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(8,9;3,3) ≤ 47 | lean-here (waterfilled Argument A, col side) | `factHolds_of_waterfill_le` (kernel `decide`) |
| z(9,3;3,3) ≤ 20 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(9,4;3,3) ≤ 26 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(9,5;3,3) ≤ 30 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(9,6;3,3) ≤ 36 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(9,7;3,3) ≤ 41 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |
| z(9,8;3,3) ≤ 47 | lean-here (waterfilled Argument A, row side) | `factHolds_of_waterfillT_le` (kernel `decide`) |

Axioms of `ZarPrune.Closures.Z_9_9_50.z_9_9_le_49`: ['propext', 'Classical.choice', 'Quot.sound']

## Prune (Lean source)

```lean
-- prune term: Prune.or (Prune.or (baseline P) (counting P)) (evolved P)
```

## Cases handed to the solver

| rows | cols | disposition | certificate |
|---|---|---|---|
| [6, 6, 6, 6, 6, 5, 5, 5, 5] | [6, 6, 6, 6, 6, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r6-6-6-6-6-5-5-5-5_c6-6-6-6-6-5-5-5-5.lrat |
| [6, 6, 6, 6, 6, 5, 5, 5, 5] | [6, 6, 6, 6, 6, 6, 5, 5, 4] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r6-6-6-6-6-5-5-5-5_c6-6-6-6-6-6-5-5-4.lrat |
| [6, 6, 6, 6, 6, 5, 5, 5, 5] | [7, 6, 6, 6, 5, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r6-6-6-6-6-5-5-5-5_c7-6-6-6-5-5-5-5-5.lrat |
| [6, 6, 6, 6, 6, 5, 5, 5, 5] | [7, 7, 6, 5, 5, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r6-6-6-6-6-5-5-5-5_c7-7-6-5-5-5-5-5-5.lrat |
| [6, 6, 6, 6, 6, 6, 5, 5, 4] | [6, 6, 6, 6, 6, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r6-6-6-6-6-6-5-5-4_c6-6-6-6-6-5-5-5-5.lrat |
| [6, 6, 6, 6, 6, 6, 5, 5, 4] | [6, 6, 6, 6, 6, 6, 5, 5, 4] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r6-6-6-6-6-6-5-5-4_c6-6-6-6-6-6-5-5-4.lrat |
| [6, 6, 6, 6, 6, 6, 5, 5, 4] | [7, 6, 6, 6, 5, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r6-6-6-6-6-6-5-5-4_c7-6-6-6-5-5-5-5-5.lrat |
| [6, 6, 6, 6, 6, 6, 5, 5, 4] | [7, 7, 6, 5, 5, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r6-6-6-6-6-6-5-5-4_c7-7-6-5-5-5-5-5-5.lrat |
| [7, 6, 6, 6, 5, 5, 5, 5, 5] | [6, 6, 6, 6, 6, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r7-6-6-6-5-5-5-5-5_c6-6-6-6-6-5-5-5-5.lrat |
| [7, 6, 6, 6, 5, 5, 5, 5, 5] | [6, 6, 6, 6, 6, 6, 5, 5, 4] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r7-6-6-6-5-5-5-5-5_c6-6-6-6-6-6-5-5-4.lrat |
| [7, 6, 6, 6, 5, 5, 5, 5, 5] | [7, 6, 6, 6, 5, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r7-6-6-6-5-5-5-5-5_c7-6-6-6-5-5-5-5-5.lrat |
| [7, 6, 6, 6, 5, 5, 5, 5, 5] | [7, 7, 6, 5, 5, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r7-6-6-6-5-5-5-5-5_c7-7-6-5-5-5-5-5-5.lrat |
| [7, 7, 6, 5, 5, 5, 5, 5, 5] | [6, 6, 6, 6, 6, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r7-7-6-5-5-5-5-5-5_c6-6-6-6-6-5-5-5-5.lrat |
| [7, 7, 6, 5, 5, 5, 5, 5, 5] | [6, 6, 6, 6, 6, 6, 5, 5, 4] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r7-7-6-5-5-5-5-5-5_c6-6-6-6-6-6-5-5-4.lrat |
| [7, 7, 6, 5, 5, 5, 5, 5, 5] | [7, 6, 6, 6, 5, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r7-7-6-5-5-5-5-5-5_c7-6-6-6-5-5-5-5-5.lrat |
| [7, 7, 6, 5, 5, 5, 5, 5, 5] | [7, 7, 6, 5, 5, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r7-7-6-5-5-5-5-5-5_c7-7-6-5-5-5-5-5-5.lrat |
| [8, 6, 6, 5, 5, 5, 5, 5, 5] | [8, 6, 6, 5, 5, 5, 5, 5, 5] | REFUTED (LRAT verified) | cache/certs/m9_n9_s3_t3_w50/r8-6-6-5-5-5-5-5-5_c8-6-6-5-5-5-5-5-5.lrat |

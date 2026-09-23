# E14 — Tier-1 closure seam: `Closure.lean` + `Closures/Z_m_n_w.lean` (T-7 go/no-go, T-8)

Owner G-closure, 2026-09-21.  Contract: `docs/design.md` §1.1, §3.4, §4.6.  Lean 4.34.0 + Mathlib v4.34.0,
`lake env lean` only (no build; `Closure.lean` is inlined into every check file, see `zar_ub/closure.py`).
All numbers below are measured on this machine (macOS arm64), wall-clock of one `lake env lean` process
(which includes ~1.0 s of Mathlib import and ~1.1 s of elaborating the inlined `Closure.lean`).

## T-7 — (9,9,3,3,50), pure mode, prune = `Prune.or (baseline P) (counting P)`

| step | result |
|---|---|
| Lean enumerator (`genRows P facts`, `genCols P facts`) | 6 × 6 = 36 cases, identical to `partitions.py` **including order** |
| survivors of the library (Lean `#eval` of `survivorsK P facts condPrune.kill`) | **17** = the 17 cases of `cache/case_table_m9_n9_s3_t3_w50_pure.json` not killed by `baseline_lean_mask` (same set, same order) |
| closure file | `lean/ZarPrune/Closures/Z_9_9_50.lean` (96 lines): 12 counting facts proved in place, 0 external hypotheses, `survList` = 17 literals |
| `survivors_eq … := by decide +kernel` | **passes**; whole file 2.9–3.2 s; the kernel `decide` of `survivors_eq` + the closure theorem ≈ **0.8 s** (file with vs. without them: 2.96 s vs 2.14 s, best of 3) |
| axioms of `z_9_9_le_49` | `[propext, Classical.choice, Quot.sound]` |
| certificates | all 17 survivors have `certified` LRAT entries in `cache/certs/m9_n9_s3_t3_w50/manifest.json` (E9: 36/36) |
| report | `cache/certs/m9_n9_s3_t3_w50/closure_report.md` (copy: `closure_report_m9_n9_s3_t3_w50.md`), tier 1, trusted-base table, fact table |

**Go/no-go: GO for Tier-1.**  The kernel budget of design §4.6 (≤ 30 min per closure) is met by three orders
of magnitude; `decide +native` (Tier-1n) is **not needed** for any instance measured.  The `chooseMul`
replacement of design §4.6 item 5 was not needed either: `Nat.choose` at these sizes reduces fast enough in
the kernel (the library kills use `Nat.choose`, unchanged).

Tier-1n fallback (measured anyway, `--tier 1n`, file `scaling/Z_9_9_50_tier1n.lean`): passes in 2.1 s;
`#print axioms` lists `survivors_eq._native.decide.ax_1_1` (Lean 4.34 spells the `Lean.ofReduceBool` use as a
per-declaration auxiliary axiom).  `check_closure` recognises the `_native` name, reports it, and **rejects the
same file when checked as Tier-1** (`ok = False`).

Kernel scaling (pure training ladder, `--out experiments/E14_closure/scaling`, no certificates so "not
established", but the kernel obligation is the same):

| instance | enumerator | survivors of the library | file check (kernel `decide`) |
|---|---|---|---|
| (9,10,55) | 9 × 5 | 22 | 3.3 s |
| (10,11,65) | 15 × 13 | 66 | 7.1 s |
| (11,11,70) | 25 × 25 | 237 | 23.5 s |
| (12,12,81) | 15 × 15 | 137 | 12.1 s |
| (13,13,93) | 32 × 32 | 314 | 47.6 s |

Every Lean survivor list equals the cached table's `baseline_lean_mask` survivors (set and order).  Cost grows
with the number of enumerated pairs (each pair evaluates the full `counting` kill in the kernel), roughly
0.04 s per pair; the (12,18,109) target (2,562 table pairs, 40k pure) would be ~2 min in trust mode.

Adversarial checks (both must fail, and do): dropping one survivor from `survList`
(`scaling/Z_9_9_50_wrong_literal.lean`) and inserting a bogus case (`scaling/Z_9_9_50_extra_case.lean`)
both end with `decide proved that the proposition … is false`.

## T-8 — trust `tan2022`, zero survivors (facts from `zar_ub.ledger.facts_for`)

| instance | rows × cols (Lean = Python) | counting facts (proved in file) | external hypotheses | check | axioms | theorem |
|---|---|---|---|---|---|---|
| (10,21,107) | 0 × 2 | 25 | 18 | 3.0–3.3 s | propext, Classical.choice, Quot.sound | `z_10_21_le_106 (h1 … h18) : ∀ A : Mat 10 21, ¬ HasKst P A → weight A ≤ 106` |
| (11,19,107) | 22 × 0 | 24 | 16 | 3.7–3.9 s | same | `z_11_19_le_106 (h1 … h16)` |
| (11,20,112) | 2 × 0 | 25 | 16 | 3.4–3.5 s | same | `z_11_20_le_111 (h1 … h16)` |

Files: `lean/ZarPrune/Closures/Z_10_21_107.lean`, `Z_11_19_107.lean`, `Z_11_20_112.lean`; reports under
`cache/certs/<tag>/closure_report.md` (copies here).  No SAT case remains, so `Hrefuted` is discharged by
`fun _ hq => nomatch hq` and the theorems are conditional only on the named Tan facts (design C5).

## Other measurements

* `lean/ZarPrune/Closure.lean` (594 lines) elaborates in 2.1 s; 58 declarations, every `#print axioms`
  ⊆ {propext, Quot.sound, Classical.choice} (`audit_closure.py`; 29 use all three, 14 use none).
* Enumerator cross-check (`audit_closure.py`): Lean `genRows`/`genCols` == Python `row_partitions`/
  `column_partitions` (same lists, same order) on 10 instances incl. s=t=2, s=t=4 and the three T-8 cells.
* `sortedProfileOf` is not kernel-reducible (`List.mergeSort` is well-founded); it appears only in theorem
  statements (`Hrefuted`), never under `decide`.
* Phase-1 `#eval` of the survivors (compiled evaluation): 2.1–3.6 s per instance.

## How to reproduce

```bash
cd examples/zarankiewicz/upper_bounds; export ZAR_UB_NO_LLM=1; PY=../../../.venv/bin/python
bash experiments/E14_closure/run_e14.sh          # T-7, T-8, scaling, audit (≈ 3 min)
$PY -m zar_ub closure 9 9 3 3 50 --pure          # T-7 alone
$PY -m zar_ub closure 10 21 3 3 107 --trust tan2022
$PY experiments/E14_closure/audit_closure.py     # axioms + enumerator cross-check
```

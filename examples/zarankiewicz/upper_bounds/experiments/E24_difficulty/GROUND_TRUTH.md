# E24 / A1 — Ground truth for difficulty estimation

Owner A1 (ground truth). Date 2026-09-22. No LLM calls, no Lean build. One Lean gate process was used
for the baseline masks of the new tables. Seeds are fixed. Nothing under `cache/` was modified in place.
The new tables are new files (`*_pure_gt.json`).

## Files

| file | rows | what |
|---|---|---|
| `ground_truth_initial.jsonl` | 32,801 | Step 1: every probed record of every `cache/case_table_*.json`. Exact when the probe status is unsat/sat. Status `unknown` with `d` = conflicts reached (a lower bound) otherwise. |
| **`ground_truth.jsonl`** | **35,081** | Final version: the initial rows, with deepened rows replacing their table rows, plus the 2,280 rows of the three new tables. |
| `ground_truth_deepen_raw.jsonl` | 3,554 | One line per solved job, with the full per-run cost (conflicts, propagations, decisions, seconds), the budget that was hit, and the previous cap and conflicts. |
| `ground_truth_jobs.jsonl` | 3,554 | The deterministic job plan (`gt_deepen.py plan`). |
| `ground_truth_encoding_check.jsonl` | 60 | Reproducibility check (see Methods). |
| `ground_truth_newtables_census.jsonl` | 3 | Census of the new tables (Step 3). |
| `ground_truth_initial_summary.jsonl` | 1 | Step-1 bookkeeping, including the c2000 measurement cost. |
| `cache/case_table_m9_n18_s3_t3_w86_pure_gt.json`, `..._m10_n19_s3_t3_w99_pure_gt.json`, `..._m9_n16_s3_t3_w78_pure_gt.json` | 363 / 426 / 1,491 | NEW wide pure tables at w = z+1. They carry `baseline_lean_mask` from the Lean gate. |
| `gt_common.py`, `gt_initial.py`, `gt_check_encoding.py`, `gt_deepen.py`, `gt_newtables.py`, `gt_report.py`, `gt_baseline_check.py` | | The scripts. All are rerunnable, and `gt_deepen.py run` resumes where it stopped. |

Row schema: this is the contract, plus some extras. The contract fields are `cell`, `m`, `n`, `s`, `t`, `w`, `trust`, `rows`, `cols`, `status`, `d`, `cap`, `c2000`, `log2_volume` and `source`.
The extras:
- `table`, `kind` and `baseline_lean_kill`: whether the proved library kills the case. `null` means the table has no mask.
- `label_mode` and `c2000_source`, on table rows.
- `group`, `prev_cap`, `prev_conflicts`, `budget_hit`, `run_seconds`, `run_propagations`, `cost_conflicts`, `cost_propagations` and `cost_seconds`, on deepened and new rows.

What `cap` means for each kind of row:
- On table rows, `cap` is the probe's `budget_cap`.
- On deepened rows it is 2,000,000.
- On new-table rows that resolved inside the first 2,000-conflict run it is 2,000.

The file contains every (s,t) that is cached. Two cells are not (3,3): (9,9;4,4) and (8,8)/(8,9;2,2). Filter on `s`/`t` if you need to.

## Methods

* **Solver.** pysat `cadical195`, fresh solver per run, default options, same `encode_case` as the engine (`zar_ub.solve.solve_cnf`).
* **Deepen (Step 2).** Each case got ONE fresh run with a cap of 2,000,000 conflicts and a 180 s wall limit. New-table cases (Step 3) first got a 2,000-cap run, which gives `c2000`. If that run did not resolve the case, it got the same 2M/180 s run.
  * If `status` is unsat or sat, `d` is the exact conflict count.
  * If `status` is unknown, `d` is the conflicts reached (right-censored).
  * **The 180 s wall limit never fired (0 of 3,554 runs).** Every open case stopped at exactly the 2,000,000-conflict cap. The censoring is therefore in conflicts only and does not depend on machine load (load average was 15–23 during the run).
* **Determinism check** (`gt_check_encoding.py`). I took a seeded sample of 60 cached exact labels with 2k < d ≤ 60k: 30 from legacy tables and 30 from censored-mode tables. I re-solved each with today's encoder and a fresh solver. **All 60 reproduced the identical conflict count and status.** So the old cached labels and the new runs are the same measurement. A case resolved at d < C under any cap ≥ C has the same d.
* **E10 cross-check.** All 1,571 E10 rows match the pure tables they were written back into (status and conflicts). There were 0 mismatches.
* **c2000 (Step 1).** Sources, in order:
  * The probe's own value.
  * E10's value.
  * Derived: a probe refuted below 2,000 conflicts *is* the 2,000-cap run.
  * Otherwise measured by a fresh 2,000-cap run (60 s limit). 6,262 values were measured this way, at a cost of 12.5 M conflicts, 4.85 G propagations and 235 solver-s. No measurement hit its time limit.
* **Sampling (Step 2f).** 150 censored cases per target table, drawn with `random.Random("2024:<table file>")`.

## Results

### Totals (ground_truth.jsonl, 35,081 cases)

| regime | cases |
|---|---|
| easy, d ≤ 2,000 | 7,768 (15 of them SAT; 56 are in the non-(3,3) cells) |
| mid, 2,000 < d ≤ 20,000 | 9,622 |
| **hard, exact, d > 20,000** | **2,238** (was 262). **1,518 of them are in wide cells (n/m ≥ 1.4)** and 720 in near-square cells. |
| hard, censored at 2M (d > 2,000,000) | 191 |
| hard, censored at 20k (never deepened) | 15,262 (target tables plus the pure w = z tables) |

Distribution of the exact d, over all (3,3) cells:

| d range | cases |
|---|---|
| ≤ 2k | 7,712 |
| 2k–20k | 9,622 |
| 20k–200k | 1,807 |
| 200k–2M | 431 |

Before this run the maximum exact d was 221,874. It is now 1,986,172.

### Per cell (cells with ≥ 20 cases; cell tags drop `_s3_t3`)

| cell | trust | cases | easy ≤2k | mid | hard exact | hard censored (cap) | median / p90 / max exact d | source |
|---|---|---|---|---|---|---|---|---|
| m9_n9_w49_pure | pure | 169 | 155 | 14 | 0 | 0 | 299 / 1,715 / 7,714 | table 169 |
| m9_n9_w50_pure | pure | 36 | 34 | 2 | 0 | 0 | 251 / 1,820 / 2,880 | table 36 |
| m9_n9_s4_t4_w62_pure | pure | 49 | 49 | 0 | 0 | 0 | 251 / 550 / 1,080 | table 49 |
| m9_n10_w54_pure | pure | 165 | 151 | 14 | 0 | 0 | 572 / 1,700 / 5,402 | table 165 |
| m9_n10_w55_pure | pure | 45 | 44 | 1 | 0 | 0 | 453 / 1,309 / 2,240 | table 45 |
| **m9_n16_w78_pure** (new) | pure | 1491 | 372 | 574 | **545** | 0 | 9,728 / 75,854 / 482,851 | new_table 1491 |
| **m9_n18_w86_pure** (new) | pure | 363 | 48 | 140 | **175** | 0 | 18,508 / 181,630 / 683,672 | new_table 363 |
| **m9_n23_w104** (target) | tan2022 | 244 | 27 | 71 | **140** | 6 (2M) | 37,760 / 767,704 / 1,940,934 | deepen 146, table 98 |
| m10_n10_w60_pure | pure | 144 | 125 | 19 | 0 | 0 | 704 / 2,435 / 7,495 | table 144 |
| m10_n10_w61_pure | pure | 25 | 24 | 1 | 0 | 0 | 522 / 1,464 / 2,643 | table 25 |
| m10_n11_w64_pure | pure | 725 | 516 | 208 | 1 | 0 | 1,043 / 7,055 / 21,342 | deepen 1, table 724 |
| m10_n11_w65_pure | pure | 195 | 156 | 39 | 0 | 0 | 830 / 4,215 / 10,079 | table 195 |
| m10_n14_w77_pure | pure | 2064 | 595 | 866 | 0 | 603 (20k) | 3,567 / 13,299 / 19,897 | table 2064 |
| **m10_n14_w78_pure** | pure | 525 | 194 | 212 | **119** | 0 | 5,294 / 40,251 / 158,997 | deepen 119, table 406 |
| **m10_n19_w99_pure** (new) | pure | 426 | 80 | 145 | **196** | 5 (2M) | 16,992 / 311,544 / 1,934,925 | new_table 426 |
| **m10_n20_w103_pure** | pure | 130 | 0 | 19 | **105** | 6 (2M) | 84,820 / 647,212 / 1,681,249 | deepen 130 |
| m10_n23_w113 (target) | tan2022 | 4818 | 558 | 929 | 0 | 3331 (20k) | 4,068 / 15,241 / 19,991 | table 4818 |
| m11_n11_w69_pure | pure | 2025 | 1294 | 638 | 0 | 93 (20k) | 1,194 / 7,278 / 19,559 | table 2025 |
| m11_n11_w70_pure | pure | 625 | 431 | 179 | 15 | 0 | 1,108 / 7,338 / 74,130 | table 625 |
| m11_n12_w74_pure | pure | 2079 | 938 | 870 | 0 | 271 (20k) | 1,867 / 10,931 / 19,947 | table 2079 |
| m11_n12_w75_pure | pure | 420 | 152 | 217 | 51 | 0 | 4,007 / 21,643 / 61,778 | table 420 |
| **m11_n21_w117_pure** | pure | 37 | 0 | 7 | **23** | 7 (2M) | 118,310 / 1,050,571 / 1,545,761 | deepen 37 |
| m11_n23_w124 (target) | tan2022 | 822 | 25 | 148 | 0 | 649 (20k) | 8,242 / 18,425 / 19,971 | table 822 |
| m12_n12_w80_pure | pure | 1296 | 486 | 423 | 0 | 387 (20k) | 1,733 / 11,268 / 19,992 | table 1296 |
| m12_n12_w81_pure | pure | 225 | 21 | 111 | 93 | 0 | 14,091 / 65,456 / 221,874 | table 225 |
| m12_n13_w86_pure | pure | 1710 | 455 | 725 | 0 | 530 (20k) | 3,492 / 13,198 / 19,959 | table 1710 |
| m12_n13_w87_pure | pure | 360 | 83 | 173 | 104 | 0 | 7,531 / 60,365 / 264,307 | deepen 1, table 359 |
| **m12_n18_w109** (target) | tan2022 | 2562 | 35 | 221 | **115** | 2191 (20k: 2156, 2M: 35) | 12,545 / 437,209 / 1,865,911 | deepen 150, table 2412 |
| m13_n13_w92_pure | pure | 3969 | 432 | 1808 | 0 | 1729 (20k) | 6,226 / 15,248 / 19,933 | table 3969 |
| **m13_n13_w93_pure** | pure | 1024 | 178 | 463 | **383** | 0 | 10,864 / 92,837 / 828,132 | deepen 383, table 641 |
| **m13_n19_w123** (target) | tan2022 | 4130 | 45 | 254 | **95** | 3736 (20k: 3681, 2M: 55) | 10,241 / 406,940 / 1,986,172 | deepen 150, table 3980 |
| **m16_n17_w134** (target) | tan2022 | 2139 | 32 | 125 | **73** | 1909 (20k: 1832, 2M: 77) | 12,247 / 669,141 / 1,853,149 | deepen 150, table 1989 |

### Step 2: deepen results and cost (cost in solver conflicts / propagations / solver-seconds)

| group | table | n | unsat | open at 2M | SAT | conflicts | propagations | solver-s |
|---|---|---|---|---|---|---|---|---|
| a | m10_n20_w103_pure | 130 | 124 | 6 | 0 | 40.8 M | 7.43 G | 1,617 |
| b | m11_n21_w117_pure | 37 | 30 | 7 | 0 | 22.8 M | 5.32 G | 1,428 |
| c | m9_n23_w104 (target) | 146 | 140 | 6 | **0** | 72.5 M | 15.0 G | 4,282 |
| d | m10_n14_w78_pure | 119 | 119 | 0 | 0 | 5.7 M | 0.92 G | 140 |
| e | m13_n13_w93_pure | 383 | 383 | 0 | 0 | 32.8 M | 6.15 G | 974 |
| f | m12_n18_w109 (sample) | 150 | 115 | 35 | 0 | 116.6 M | 28.4 G | 6,696 |
| f | m13_n19_w123 (sample) | 150 | 95 | 55 | 0 | 157.6 M | 44.7 G | 8,887 |
| f | m16_n17_w134 (sample) | 150 | 73 | 77 | 0 | 193.8 M | 66.9 G | 9,410 |
| g | leftovers (m10_n20_w103, m10_n22_w111, m12_n13_w87_pure, m10_n11_w64_pure) | 9 | 9 | 0 | 0 | 0.56 M | 0.09 G | 12 |
| new | 3 new tables (Step 3) | 2,280 | 2,275 | 5 | 0 | 127.0 M | 22.0 G | 4,991 |
| **total** | | **3,554** | **3,363** | **191** | **0** | **770.3 M** | **196.9 G** | **38,436** |

Wall time was 91.5 min on 7 processes, with other agents' solvers running at the same time. Solver-seconds are inflated by that contention, so compare costs in conflicts.

**SAT discoveries: none.** All 146 censored cases of the (9,23,104) target were run to 2M conflicts:
- 140 are UNSAT.
- 6 are still open at 2M.
- None is SAT.

This is consistent with z(9,23) = 103 (dfield). No lower-bound discovery was made, and no `has_kst` witness check was triggered. The w = z+1 pure tables, deepened and new, also produced zero SAT cases, as required.

### How bad were the old censored labels? (the old table `d` compared with the deepened truth, same cases)

| table | n | open at 2M | true d p10 / p50 / p90 | old label min / median / max | Spearman(old, true), exact rows | true d > 400k (above the 20×cap clip) |
|---|---|---|---|---|---|---|
| m10_n20_w103_pure | 130 | 6 | 8.2k / 92k / 1.12M | 40k / 40k / 304k | 0.11 | 28 |
| m11_n21_w117_pure | 37 | 7 | 8.9k / 251k / ≥2M | 40k / 40k / 40k | n/a (constant) | 14 |
| m9_n23_w104 | 146 | 6 | 32k / 197k / 1.64M | 344k / 400k / 400k | 0.69 | 56 |
| m10_n14_w78_pure | 119 | 0 | 23k / 38k / 95k | 20k / 20k / 20k | n/a (constant) | 0 |
| m13_n13_w93_pure | 383 | 0 | 24k / 47k / 185k | 42k / 47k / 51k | 0.23 | 9 |
| m12_n18_w109 | 150 | 35 | 58k / 427k / ≥2M | 302k / 400k / 400k | 0.34 | 78 |
| m13_n19_w123 | 150 | 55 | 78k / 814k / ≥2M | 400k (constant) | n/a | 95 |
| m16_n17_w134 | 150 | 77 | 119k / ≥2M / ≥2M | 400k (constant) | n/a | 111 |

On (13,19,123) and (16,17,134) every censored case has the same label, 400,000 (fhat is clipped at 20 × 20k). **So G_target on these tables currently weights the censored cases by count, not by difficulty.** The true d of the sampled cases spans more than 1.5 decades, and 51% of the (16,17) sample is above 2M.

### Reference: how today's features rank the new ground truth (`gt_baseline_check.py`, nothing fitted)

Spearman ρ is computed on exact rows. Harrell's C accounts for right-censoring and is computed on a systematic subsample of ≤ 1,500 rows per group. "Wide" means n/m ≥ 1.4. "Hard" means d > 20k or censored at ≥ 20k.

| group | n (exact) | c2000 ρ / C | log2_volume ρ / C | fhat ρ / C |
|---|---|---|---|---|
| all (3,3) | 35,010 (19,557) | 0.882 / 0.760 | 0.483 / 0.789 | 0.579 / 0.809 |
| hard, square | 5,639 (720) | constant / 0.500 | 0.441 / 0.694 | 0.441 / 0.694 |
| hard, wide | 12,052 (1,518) | constant / 0.500 | 0.531 / 0.716 | 0.531 / 0.716 |
| d > 2000, wide | 15,642 (5,108) | constant / 0.500 | 0.098 / 0.739 | 0.098 / 0.739 |
| new labels (deepen + new tables) | 3,554 (3,363) | 0.618 / 0.630 | 0.566 / 0.720 | 0.605 / 0.739 |

For every case with d > 2,000, c2000 equals the cap. It therefore carries **no** information in the mid and hard regimes, and fhat reduces to a monotone function of log2_volume there. Any ranking skill that fhat shows on hard cases is volume alone. These are the baseline numbers that A2–A4's estimators should beat.

## Step 3: new wide TRAIN-candidate cells (pure, w = z+1, all cases labelled)

Here, "survivors" means cases that the proved library (the Lean mask) does not kill. The Lean gate returned OK for all three tables, with axioms {propext, Classical.choice, Quot.sound} and 20 s for all three together.

| cell | z | cases | status | library kills / survivors | total d all / survivors | library work share | argD kills (all / survivors) | DGH kills (all / survivors) | DGH d-weighted gain on survivors | DGH tail gain | survivors with d > 20k |
|---|---|---|---|---|---|---|---|---|---|---|---|
| (9,18) w86 | 85 | 363 | 363 unsat | 223 / 140 | 24.4 M / 19.8 M | 19.1% | 223 / 0 | 165 / **47** | **0.110** | 0.0 | 120 |
| (10,19) w99 | 98 | 426 | 421 unsat, 5 open at 2M | 261 / 165 | 55.7 M / 49.7 M | 10.8% | 261 / 0 | 0 / 0 | 0.0 | 0.0 | 139 |
| (9,16) w78 | 77 | 1,491 | 1,491 unsat | 798 / 693 | 43.4 M / 36.7 M | 15.5% | 798 / 0 | 284 / **90** | **0.031** | 0.0 | 449 |

* The library's kill set **is** Argument D on all three tables. Its kill mask equals the Python argD mask exactly (223, 261 and 798 kills), so argD kills 0 survivors, as E23 found.
* DGH is the only known argument that removes survivors, and only on the 9-row cells. The survivors it kills are easier than the rest:

  | cell | DGH-killed survivors: n / median d / total d | other survivors: median d / total d |
  |---|---|---|
  | (9,18) | 47 / 30.3k / 2.18M | 136.6k / 17.6M |
  | (9,16) | 90 / 7.5k / 1.15M | 36.6k / 35.5M |

  That is why the tail gain is 0 on both.
* (10,19) is a clean negative control: DGH kills nothing there. Argument D is the whole library, and the 139 hard survivors need something new.
* Other wide candidates that were counted but not labelled:

  | cell | cases | DGH-only kills |
  |---|---|---|
  | (9,17) | 768 | 57 |
  | (11,15) | 1,519 | 54 |
  | (9,19) | 525 | 30 |
  | (10,15) | 484 | 0 |
  | (10,16) | 1,656 | 0 |
  | (12,15) | 1,032 | 0 |

The new tables are saved with `kind=""` and `omega=1.0`. `label_mode` is `exact`, except (10,19), which is `censored` because of its 5 cases open at 2M. For those 5, `budget_cap` is the conflicts reached and `d = censored_d(cap, fhat) = 2,000,002`. `conf_cap` is 2,000,000.

## Negative results and caveats

* 191 cases are still open at 2,000,000 conflicts. They are right-censored (`status=unknown`, `d` = 2,000,000 to 2,000,00x). On the target samples they are 23% of (12,18), 37% of (13,19) and 51% of (16,17). The hard tail of the real targets is therefore still only partly observed, even at 2M.
* The Step 2f target labels are a **uniform random sample of 150 per table**. Of the censored-at-20k cases on those tables, 2,156 (12,18), 3,681 (13,19) and 1,832 (16,17) were not deepened. The same holds for all of (10,23,113) (3,331) and (11,23,124) (649).
* The non-target pure w = z tables are unchanged from the cache, still censored at 20k:
  * (10,14,77): 603 cases
  * (11,11,69): 93
  * (11,12,74): 271
  * (12,12,80): 387
  * (12,13,86): 530
  * (13,13,92): 1,729

  These w = z tables contain SAT cases, so their censored cases may be SAT or UNSAT.
* E10's hard labels were produced with a 5M / 240 s cap. Mine used 2M / 180 s. Because conflict counts are deterministic, the two are the same measurement below 2M.

## TODOs for the integrator (I did not edit any shared file)

1. **Write-back decision.** Deepened labels exist for the following tables:
   * These are now fully labelled, except for the open-at-2M cases listed:
     * `m10_n20_w103_pure`: 6 still open at 2M
     * `m11_n21_w117_pure`: 7 open at 2M
     * `m9_n23_w104`: 6 open at 2M
     * `m10_n14_w78_pure`: fully labelled
     * `m13_n13_w93_pure`: fully labelled
     * the 9 group-g leftovers
   * The 3 × 150 target samples.

   `casetable.deepen(tab, cap=2_000_000)` would reproduce exactly the same conflict counts, because the runs are deterministic. The integrator can also copy the labels from `ground_truth.jsonl` without re-solving: the key is (`cell`, `rows`, `cols`) and the table index is in the `ground_truth_jobs.jsonl` job.
2. **Suite.** Consider (9,18,86) and (9,16,78) as WIDE TRAIN cells where DGH has verifiable, difficulty-weighted gain (0.110 and 0.031 on survivors). Keep (10,19,99) as a wide control with DGH gain 0. The tables are at `cache/case_table_*_pure_gt.json` and already carry `baseline_lean_mask`.
3. **Censored-label clip.** `CENSOR_CLIP = 20` (the 400k ceiling on targets) is binding on the real targets:

   | table | sampled cases with true d > 400k |
   |---|---|
   | (16,17) | 111 / 150 |
   | (13,19) | 95 / 150 |
   | (12,18) | 78 / 150 |

   All censored labels on (13,19) and (16,17) are the constant 400,000. The estimator or reward owners should refit the label on this data, or raise the ceiling.
4. **Deeper labels.** For the 191 open-at-2M cases, a 20M-cap pass is the obvious next step: roughly 191 × 20M = 3.8 G conflicts, or about 7.6 h on 7 cores at the observed rate of 20k conflicts/s/process (770 M conflicts in 38,436 solver-s under contention). Also uniform samples from (10,23,113) and (11,23,124), which have 0 cases in the hard-exact regime today.
5. `ground_truth.jsonl` also carries `baseline_lean_kill`. Estimator owners should report accuracy on library **survivors** (`baseline_lean_kill == false`) separately, because those are the only cases that reach the reward.

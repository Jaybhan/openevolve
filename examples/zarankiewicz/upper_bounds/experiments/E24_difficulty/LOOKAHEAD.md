# E24 / A2: lookahead (AlphaMapleSAT / march-style) difficulty features

Owner: A2-lookahead. Date: 2026-09-22. Files:

- `zar_ub/hardness_lookahead.py` (module, estimator contract)
- `experiments/E24_difficulty/lookahead_{features,supplement,eval,freeze,test,combo}.py`
- `lookahead_features_{dev,dev_fl,test}.jsonl`
- `lookahead_{eval,test,combo}.{json,txt}`
- `lookahead_model.json` (frozen)

Solver processes: at most 2. Everything is BCP only, through pysat `minisat22.propagate()`. No
case table was modified, and no network or LLM calls were made.

## 1. Summary

**Question.** The current censored label is `fhat = exp(a + b log c2000 + g log2_volume)`. On the
hard tail its held-out Spearman is only 0.39 (E12). Can the lookahead family from the proposal
rank and size the hard cases better? The family covers AlphaMapleSAT propagation rate, march-style
failed literals, two-level lookahead and Knuth tree-size probing.

**Answer.**
- **Yes for ranking, partly for magnitude.** Lookahead ranks hard cases far better than the
  current label, but it is **not the best estimator** once a 20k-conflict probe exists.
- **The pilot's feature does not carry the signal.** The proposal pilot used mean implied cells
  per decision (rho -0.65 on 14 cases). It does not replicate: within-cell rho is -0.04 on DEV
  HARD and -0.04 on TEST HARD.
- **Failed literals carry the signal.** The informative quantity is how many cell literals are
  *failed* one decision deep. That quantity comes from the fixed row/column sums interacting with
  the lex symmetry breaking. The best form is the failed-literal fixpoint (`fl_*` features): the
  number of cells it leaves free, and the row-profile volume left after it (`fl_log2vol_rows`).
- **FL features are cheap.** They cost **0.027 s, ~880 propagate calls, ~0.21 M trail literals and
  ~11 UP conflicts per case**. That is less than the 2,000-conflict probe the tables already run
  (0.037 s, 2,000 conflicts, 0.78 M propagations).

Headline numbers:

| setting | metric | current label / fhat | best lookahead |
|---|---|---|---|
| E11 acceptance metric: held-out (12,13,87), all d > 2000 (DEV LOCO, n = 253) | Spearman | 0.377 | **0.834** (nested greedy); 0.819 FL-only; 0.860 vol+la1_failed+ams |
| DEV HARD (d > 20k, 4 square cells, 262 cases), LOCO | mean within-cell Spearman | 0.179 | **0.575** (FL-only, H-trained) |
| DEV HARD, LOCO | log-RMSE | 1.128 raw / 0.838 clipped / constant 0.583 | **0.479-0.52** (HM fit + hard intercept shift) |
| TEST HARD (1,826 fresh exact labels, 10 cells, mostly wide), frozen on DEV | mean within-cell Spearman | 0.344 (clipped label) | **0.647** (greedy); 0.532 (pre-registered FL-only) |
| TEST HARD, frozen | pooled Spearman / log-RMSE | 0.455 / 1.142 | **0.653 / 0.927** (greedy); FL-only 0.412 / **1.166 (no better)** |
| target cells: AUC "d > 20k" among cases unsolved at 2k (C u H vs M), LOCO | AUC | 0.855 | **0.951** (vol+fl_free+knfl); 0.905 FL-only |
| TEST HARD, 509 cases joined with A4, frozen on DEV | within-cell Spearman | A4 `free20k` **0.855** | FL-only 0.677; **free20k + FL 0.882**, but log-RMSE worsens 0.626 -> 0.990 |

**Recommendation.**
- **Censored labels.** For a case already probed at 20k (every censored case on a target table),
  the label should come from A4's `free20k` statistics, which are free. The lookahead features
  add at most about +0.03 of ranking on top, and their calibration does not transfer.
- **Cheap tier.** Where no 20k probe exists, use the FL-only features: new tables, triage before
  deepening, or deciding which cases to deepen first. At ~0.03 s they reach within-cell rho
  0.53-0.68 on the wide TEST cells. A4 reports its 2k tier at 0.485 on the same 509 cases (their
  number, not recomputed here). They also reach AUC 0.905 for "will this need > 20k conflicts".
- **Do not deploy the pre-registered FL-only model as a magnitude estimate.** Its test log-RMSE
  (1.166) is no better than the current clipped label (1.142).

## 2. Methods

### 2.1 Features (`zar_ub/hardness_lookahead.py::estimate`)

Everything is unit propagation on `encode_case(inst, rows, cols)`. Features are computed over the
m·n cell variables (auxiliary variables are excluded from cell counts, as in AlphaMapleSAT).

| family | features |
|---|---|
| root | `root_fixed_cells` (level-0 facts, see 2.2), `root_failed` |
| `la1_*` (one decision, all 2mn cell literals) | Implied cells per literal: mean, median, max, min, std. Computed over all literals, and separately for positive and negative ones. Also: failed literals (count, fraction, positive, negative), zero-implication fraction, implied variables per decision (the AMS "propagation rate", all vars), AMS per-variable score `r(v)r(-v)+r(v)+r(-v)` (max/mean/top3), march product `r(v)r(-v)`, root binary clause count, and new binary clauses and eval_cls-weighted reduced clauses after each decision (Schur-5 / march measures; vectorised numpy) |
| `fl_*` (failed-literal fixpoint) | Assert the negation of every failed literal, then re-probe the free cells; repeat until nothing fails (as march / AMS do with their `L` list). Features: rounds, units, fixed/free cells, fixed fraction, refuted flag, la1 stats and AMS score on the reduced formula, and **`fl_log2vol_rows/cols`**. The last is the search-space volume left after FL: `sum_i log2 C(free_i, r_i - ones_i)` (equals `log2_volume` when FL fixes nothing) |
| `la2_*` (two decisions, 400 seeded pairs) | l1 is a non-failed literal and l2 is on a cell left free by l1. Features: implied cells (stats), failed-pair fraction, synergy `r(l1,l2) - r(l1) - r(l2)` |
| `kn_*`, `knfl_*` (Knuth 1975) | 64 random probes of the DPLL+UP tree over cell variables. At each node, test both children and let b = the number of UP-consistent ones. The node-count estimate is `N = 1 + b1 + b1 b2 + ...` and the failed-leaf estimate is `sum prod(b_j) (2-b_i)`. Reported as log-mean, mean-log, var-log and mean depth. Two variable orders: `rand` and `row` (row-major). `knfl_*` starts the walks from the FL-fixpoint root |

Other properties of `estimate`:
- **Return value.** It returns the contract dict: `features`, `d_hat`, `cost_conflicts` (UP
  conflicts hit), `cost_propagations` (trail literals assigned, summed over propagate calls),
  `cost_calls` and `cost_seconds`.
- **Determinism.** Fixed-seed runs give identical results; only the seconds differ. The seed is
  an FNV hash of (seed, m, n, rows, cols).
- **`d_hat`.** When `lookahead_model.json` exists, `d_hat` is `exp(model_log_d)` from the frozen
  primary model, and `features["model_log_d_hard"]` adds the hard shift. Otherwise `d_hat` is the
  raw Knuth failed-leaf estimate.
- **Cheapest setting.** `n_pairs=0, n_probes=0, clause_measures=False` computes only root, `la1_*`
  and `fl_*`.

### 2.2 Implementation findings

- **Solver choice.** minisat22's `propagate()` is about 10x cheaper than cadical195's through
  pysat: 0.23 s vs 2.1 s for the full feature set on a (12,12) case. The features were
  byte-identical on the 3 cases checked. The UP fixpoint is solver-independent when no conflict
  occurs, and a failure is a failure. The default is `minisat22`.
- **Bug found (affects `difficulty.propagation_fraction` too).** pysat's `propagate()` returns only
  literals assigned at decision level >= 1. Level-0 facts never appear in its trail, for example
  every cell of a row whose sum is n or 0. My first FL loop therefore re-added the same units
  forever on profiles with a full row, such as (10,23) with a row of 23. The fix is a pure-Python
  level-0 unit propagation (`root_unit_propagation`), unioned into every trail by the `_Prop`
  wrapper.
  - For that (10,23) case, `difficulty.propagation_fraction` returns 0.0 while the true level-0
    fixed-cell fraction is 0.10.
  - E10's statement "root UP fixes nothing" is true of the 7 TRAIN cells, which have no full or
    empty lines. The measurement could never have seen level-0 facts anyway.
  - Of the 4,459 DEV cases only 2 have a full or empty line. None of the records computed before
    the fix were affected (checked).

### 2.3 Data and protocol

- **DEV = `ground_truth_initial.jsonl` (A1).** Stratified and seeded (seed 24, per cell):
  - **H:** all 262 exact d > 20k cases. They sit in 4 square pure cells: (11,11,70) 15,
    (11,12,75) 51, (12,12,81) 93, (12,13,87) 103.
  - **M:** 2,799 exact 2k < d <= 20k cases, up to 150 per cell, over 31 cells.
  - **C:** 1,398 cases unknown at the 20k cap (d >= 20k is a lower bound), up to 100 per cell,
    over 15 cells. C is used only for the hard-vs-mid AUC. It covers the tan2022 target cells
    (9,23), (10,23), (11,23), (12,18), (13,19), (16,17).
- **Single features.** Reported per regime: pooled Spearman, per-cell (within-cell) Spearman, and
  the LOCO sign-selected mean (the sign is chosen on the other cells).
- **Models.** Log-linear, `log d ~ features`, least squares on standardised features. Evaluated
  leave-one-cell-out, and again leave-one-(m,n)-out. The variants are:
  - **Pre-specified feature sets.**
  - **Nested greedy forward selection.** Selection happens inside each fold by inner-LOCO
    log-RMSE, and never sees the held-out cell.
  - **Three fits per set:**
    - HM: fitted on d > 2000, as the E11 calibration is.
    - H-only.
    - HM + hard shift: slopes from HM, plus the intercept shift from the training cells' HARD
      residuals. This is the right estimator for a case known to be unsolved at 20k.
  - **Baselines:** fhat with the fixed E11 coefficients (in-sample for the 7 E10 TRAIN cells),
    fhat refitted on the same folds, the currently used clipped label
    `min(max(20000, fhat), 400000)`, and a constant.
- **Freeze, then TEST.** After the DEV analysis the models were frozen
  (`lookahead_freeze.py` -> `lookahead_model.json`, sha1 `d28e9868` at freeze).
  - The **primary was pre-registered before any TEST feature existed**: FL-only
    (`log2_volume, fl_free_cells, fl_log2vol_rows`).
  - Alternates: `vol_fl_knfl`, `vol_la1fail_ams`, and `greedy`. Greedy on all DEV picked
    `la1_failed_frac, knfl_rand_mean_log_conf, fl_log2vol_cols, kn_rand_mean_log_conf`.
  - TEST = A1's deepening and new-table labels (`ground_truth_deepen_raw.jsonl`), which appeared
    after the DEV analysis:
    - 1,826 exact HARD cases: (9,16,78)p 545, (13,13,93)p 383, (10,19,99)p 196, (9,18,86)p 175,
      (9,23,104) 140, (10,14,78)p 119, (12,18,109) 115, (10,20,103)p 105, (13,19,123) 25,
      (11,21,117)p 23.
    - 461 MID cases.
  - The complementarity test (`lookahead_combo.py`) was also specified before any TEST result was
    seen.

## 3. Results

### 3.1 Single features, DEV (`lookahead_eval.txt`)

The first two columns are Spearman with true d. "Within" is the mean over cells of the per-cell
Spearman, sign-selected LOCO. The last column is the AUC hard-vs-mid on the 18 cells with both
(target cells only).

| feature | HARD within (4 cells) | per-cell HARD | MID within (23 cells) | AUC target |
|---|---|---|---|---|
| `fl_log2vol_cols` | **0.63** | - | 0.53 | 0.95 |
| `fl_free_cells` (= `fl_units`) | **0.60** | 0.65 / 0.60 / 0.42 / 0.74 | 0.54 | 0.95 |
| `fl_log2vol_rows` | 0.60 | - | 0.55 | 0.96 |
| `la1_failed_frac` (failed literals, -) | 0.58 | 0.58 / 0.61 / 0.38 / 0.73 | 0.46 | 0.93 |
| `ams_score_mean` (-) | 0.52 | - | 0.45 | 0.93 |
| `march_prod_mean` (-) | 0.47 | - | 0.43 | 0.92 |
| `kn_rand_mean_log_conf` (Knuth) | 0.44 | 0.56 / 0.28 / 0.35 / 0.55 | 0.46 | 0.94 |
| `la2_failed_frac` (-) | 0.41 | - | 0.28 | 0.82 |
| `knfl_rand_mean_log_conf` | 0.39 | 0.04 / 0.34 / 0.47 / 0.73 | **0.54** | **0.96** |
| `la1_newbin_mean` / `la1_wred_mean` (-) | 0.35 / 0.38 (sign flips by cell) | - | 0.11 / 0.14 | 0.53 / 0.59 |
| `la1_rate_vars` (AMS propagation rate, -) | 0.35 (sign flips) | - | 0.07 | 0.44 |
| **`la1_mean` (the pilot's feature)** | **-0.04** | 0.08 / -0.12 / -0.02 / -0.08 | -0.05 | 0.53 |
| `la2_mean` | -0.02 | - | -0.07 | 0.52 |
| Knuth raw `d_hat` (log-mean failed leaves, unreduced root) | 0.18 | 0.17 / 0.16 / 0.16 / 0.24 | 0.19 | 0.80 |
| `log2_volume` (fhat's only varying input here) | 0.18 | 0.36 / 0.09 / 0.07 / 0.19 | 0.20 | 0.86 |
| static control `log2_aut` (lex group size) | -0.12, sign flips | 0.26 / -0.10 / -0.33 / -0.33 | -0.23 | 0.82 |
| static control `n_groups` | -0.20, sign flips | -0.62 / -0.27 / 0.16 / -0.08 | 0.06 | 0.68 |

- **The FL signal is not a proxy for a free static statistic.** The group-structure statistics
  that drive lex symmetry breaking flip sign between cells, while the FL counts keep their sign
  in all 4 hard cells.
- **Interpretation.** A failed cell literal is a cell forced by (fixed sums + lex +
  K_{s,t}-freeness) one step deep. Profiles where more of the matrix is forced this way are
  easier. This matches the Schur-5 / march lore: more constrained means easier.

### 3.2 Models, DEV

HARD, leave-one-hard-cell-out:

| model | within-cell Spearman | pooled | log-RMSE (raw) | log-RMSE (clipped [20k, 400k]) | top-decile recall |
|---|---|---|---|---|---|
| fhat fixed (E11) | 0.179 | 0.249 | 1.128 | 0.838 | 0.03 |
| fhat form refit [H only] | 0.179 | 0.083 | 0.566 | 0.566 | 0.03 |
| constant (other cells' HARD mean) | - | - | 0.583 | 0.583 | - |
| FL-only [H only] | **0.575** | 0.365 | 0.639 | 0.580 | **0.48** |
| FL-only [HM + shift] | 0.565 | 0.439 | 0.523 | 0.523 | 0.41 |
| vol+la1_failed_frac+ams_mean [HM + shift] | **0.577** | 0.375 | 0.556 | 0.543 | 0.45 |
| vol+fl_free+knfl [HM + shift] | 0.368 | 0.489 | **0.479** | 0.479 | 0.27 |
| greedy (nested) [HM, no shift] | 0.434 | 0.472 | 1.281 | 0.919 | 0.34 |

- Models fitted on all d > 2000 cases **without** the hard shift regress hard cases toward the
  mid-regime mean. Their HARD log-RMSE (1.3-1.7) is then worse than fhat's. The intercept shift
  fixes this on DEV.
- **In absolute error the gain over a constant is modest:** 0.48-0.52 vs 0.583.
- Leave-one-(m,n)-out gives the same numbers to within 0.03.

MID (2,799 cases, leave-one-cell-out over 31 cells):

| model | within | pooled | log-RMSE |
|---|---|---|---|
| greedy (nested) | **0.563** | **0.569** | **0.557** |
| vol+fl_free+knfl | 0.519 | 0.563 | 0.561 |
| FL-only | 0.457 | 0.378 | 0.636 |
| fhat refit / fixed | 0.196 / 0.195 | 0.244 / 0.307 | 0.625 / 2.886 |

**E11 acceptance metric** (§6.2 asks for rho >= 0.85). Held-out (12,13,87), all sampled
d > 2000 cases (n = 253), with models trained on the other cells:

| model | Spearman | log-RMSE |
|---|---|---|
| fhat (this sample; E11 reported 0.395) | 0.377 | 1.234 |
| FL-only | 0.819 | 1.184 |
| vol+la1_failed_frac+ams_mean | **0.860** | 1.096 |
| greedy | 0.834 | **0.892** |

Much of this rho comes from separating the cell's hard cases from its mid cases. Within HARD only,
(12,13,87) reaches 0.69-0.72.

**Hard-vs-mid AUC among cases unsolved at 2k** (C u H vs M, leave-one-cell-out, 18 cells):

| model | AUC, all 18 cells | AUC, the 6 tan2022 target cells ((9,23), (10,23), (11,23), (12,18), (13,19), (16,17)) |
|---|---|---|
| fhat | 0.724 | 0.855 |
| FL-only | 0.858 | 0.905 |
| vol+fl_free+knfl | **0.919** | **0.951** |

The (16,17) and (13,19) target cells reach 0.976 and 0.974.

### 3.3 TEST (frozen on DEV; `lookahead_test.txt`)

HARD, 1,826 cases in 10 cells (7 of them wide):

| predictor (frozen) | within-cell Spearman | pooled | log-RMSE | top-decile recall |
|---|---|---|---|---|
| current clipped label | 0.344 | 0.455 | 1.142 | 0.14 |
| fhat fixed (E11) | 0.303 | 0.463 | 1.198 | 0.14 |
| constant (DEV HARD mean) | - | - | 1.243 | - |
| **FL-only, pre-registered primary** (+shift) | 0.532 | 0.412 | **1.166** | 0.40 |
| vol_la1fail_ams (+shift) | 0.600 | 0.581 | 0.995 | 0.43 |
| vol_fl_knfl (+shift) | 0.617 | 0.564 | 1.039 | 0.42 |
| **greedy** (+shift) | **0.647** | **0.653** | **0.927** | 0.43 |
| single feature `fl_log2vol_rows` | 0.644 | 0.526 | - | **0.48** |
| single feature `-la1_mean` (the pilot proxy) | -0.041 | -0.143 | - | 0.13 |

- Per-cell greedy within-cell rho:

  | (10,20) | (11,21) | (9,23) | (9,18) | (10,19) | (9,16) | (10,14,78) | (13,13,93) | (12,18) | (13,19) |
  |---|---|---|---|---|---|---|---|---|---|
  | 0.77 | 0.81 | 0.63 | 0.73 | 0.75 | 0.51 | 0.45 | 0.77 | 0.72 | 0.34 |

- The current label reaches -0.44 in (13,19), where only 25 cases are exact.
- **The pre-registered primary missed on magnitude.** It improves within-cell ranking by +0.19,
  but its level does not transfer to wide cells: raw log-RMSE 2.38, and 1.17 with the shift. Its
  DEV coefficients are strongly collinear (`log2_volume` -2.93, `fl_free_cells` +2.36 in
  standardised units). The best TEST model (greedy) is an alternate. Its TEST numbers are genuine
  held-out numbers, but picking it *now* as the recommended one is selection on TEST.
- MID TEST (461 cases, 4 new cells): greedy within 0.524, pooled 0.589, log-RMSE 0.533. fhat
  gives 0.310 / 0.051 / 2.356.

### 3.4 Complementarity with A4 (`lookahead_combo.txt`)

Fitted on the 262 DEV hard cases, then frozen. TEST = the 509 hard cases for which A4 published
features: (10,20) 105, (11,21) 23, (9,23) 140, (9,18) 175, (10,19) 66.

| model | DEV LOCO within | DEV log-RMSE | TEST within | TEST pooled | TEST log-RMSE |
|---|---|---|---|---|---|
| A4 `free20k` (reuses the 20k probe) | **0.786** | **0.337** | 0.855 | 0.864 | **0.626** |
| FL-only [H only] | 0.575 | 0.639 | 0.677 | 0.537 | 1.702 |
| free20k + FL (7 features) | 0.776 | 0.374 | **0.882** | **0.881** | 0.990 |
| free20k + fl_free_cells | 0.783 | 0.366 | 0.853 | 0.863 | 0.623 |

Adding FL gives +0.027 within-cell and +0.017 pooled on TEST. It gives nothing on DEV and costs
0.36 in TEST log-RMSE, because the FL volume terms extrapolate badly to wide cells. **The FL
features are largely redundant with the 20k-probe statistics.**

### 3.5 Cost per case

Measured under heavy contention (load average 15-29 from other agents), so seconds are inflated.

| cells | full lookahead: s / propagate calls / trail literals / UP conflicts | FL-only: s / calls / trail literals |
|---|---|---|
| (9,9)-(10,11) | 0.13-0.21 / 9-12k / 2.3-4.6 M / ~1,650 | 0.005-0.009 / 335-486 / 12-28k |
| (11,11)-(12,13) | 0.28-0.46 / 12.6-15.4k / 5.7-9.5 M / ~1,700-1,870 | 0.012-0.021 / 530-770 / 38-116k |
| (13,13) | 0.62 / 16k / 12.5 M / 1,900 | 0.029 / 870 / 211k |
| wide: (10,23), (12,18), (13,19) | 0.9-1.25 / 20-21k / 16-24 M / 2,300-2,860 | 0.036-0.056 / 1,340-1,510 / 0.36-0.65 M |
| (16,17) | 1.81 / 22k / 33.5 M / 2,340 | 0.090 / 1,630 / 1.12 M |
| **mean, DEV** | **0.62 s / 16.2k / 12.0 M / 1,987** | **0.027 s / 878 / 0.21 M / 11.5** |

For comparison, the tables' 2k probe costs 2,000 conflicts, 0.78 M propagations and 0.037 s
(A1's summary). The 20k probe costs about 0.35 s.

- **The full lookahead is not cheap.** Its cost is comparable to a 20k CDCL run, and the 20k run's
  statistics (A4) beat it.
- **The only clearly cost-effective member of the family is the FL fixpoint.** Most of the full
  cost is the Knuth probes (128 walks) and the 400 pairs.

## 4. Negative results, stated plainly

1. **The proposal's pilot feature does not replicate.** Mean implied cells per single-cell
   decision had rho -0.65 on 14 profiles of (10,14,78) in a column-weight-only encoding. On our
   fixed-profile encoding it is noise: DEV HARD -0.04, TEST HARD -0.04, MID -0.05. Almost every
   cell decision implies 0 other cells; the median is 0. What carries the signal is whether the
   decision **fails**.
2. **AlphaMapleSAT's "propagation rate" and the march/Schur clause measures are not useful here.**
   The rate is implied variables per decision; the clause measures are new binaries and
   eval_cls. They reach within-cell |rho| 0.35-0.45 on 4 cells, but their sign flips between cells
   (AUC on targets 0.44-0.59).
3. **The raw Knuth tree-size estimate is not a usable standalone difficulty.**
   - Its values are e^30-e^40 "nodes", unrelated in scale to CDCL conflicts.
   - As `d_hat` it ranks like log2_volume (0.18 within).
   - Only mean-of-logs and FL-rooted variants carry signal (0.39-0.44 within on HARD, 0.54 on
     MID). Knuth's variance problem applies: log-mean underperforms mean-log.
4. **The two-level lookahead adds nothing beyond one level.** Pairs give -0.02 for implied cells
   and 0.41 for failed pairs, weaker than single failed literals.
5. **Pre-registered primary:** its calibration failed on TEST (log-RMSE 1.166 vs the current label's
   1.142). See 3.3.
6. **The lookahead family loses to A4's CDCL-progress features** on the censored-label task (3.4).

## 5. Threats to validity

- **DEV HARD is tiny in cell count:** 4 square pure cells ((11..12) x (11..13)). The FL features
  were *discovered* by looking at these 4 cells' within-cell correlations, so the DEV LOCO
  numbers carry design-selection leakage. TEST (1,826 cases, 10 cells, labels created after the
  design) is the unbiased number.
- **The TEST hard set is selected.** It contains cases refuted within 2M conflicts. A further 114
  cases were still unknown at 2M and are excluded: (13,19) 55, (12,18) 35, (11,21) 7,
  (10,20) 6, (9,23) 6, (10,19) 5. The very hardest cases are missing, and (13,19) keeps only 25
  exact cases.
- **The C subset of target cells was seen during design.** It was used only for AUC, and its d is
  a lower bound, but its cells overlap four TEST cells ((10,14,78), (13,13,93), (12,18), (13,19)).
  The TEST hard labels for those cells were unknown at design time.
- **Unit of analysis.** The reward uses within-table ratios (`gain_I`, `tail_I`), so within-cell
  Spearman and top-decile recall are the relevant metrics. Pooled numbers mix cell-level offsets.
- **Seconds are contended.** Report conflicts and propagations as the cost measure.

## 6. TODOs for the integrator (I did not edit shared files)

1. **`zar_ub/difficulty.py::propagation_fraction` misses level-0 facts.**
   - Cause: pysat `propagate()` does not return level-0 assignments.
   - Symptom: `prop_frac` = 0.0 on any profile with a row or column sum of 0 or full. Example:
     (10,23,113), rows [23, 11, 10 x7, 9], true fraction 0.10.
   - Fix: use `zar_ub.hardness_lookahead.root_unit_propagation`.
   - Also correct E10's statement that root UP is constant 0: it was true for those cells only
     because they have no full lines.
2. **Censored labels: prefer A4's `free20k` over any lookahead model.** It needs no new solver work
   and does better on TEST (within 0.855, log-RMSE 0.63). If the integrator wants the extra
   ranking, the union `free20k + FL` gives 0.882 within, but its log-RMSE rises to 0.99. So use FL
   only as a tie-breaker on rank, never for the level.
3. **Cheap tier / triage: use FL-only.** Call `estimate(inst, rows, cols, n_pairs=0, n_probes=0,
   clause_measures=False)` at ~0.03 s and ~0.2 M trail literals per case. Uses:
   - Order which censored cases to deepen first: AUC 0.905 for d > 20k among cases unsolved at 2k.
   - Pre-rank the cases of a brand-new table before any 20k pass, e.g. the wide cells where DGH
     acts.
   - A useful single score is `fl_log2vol_rows` (TEST within 0.644, top-decile recall 0.48).
4. **`hardness_lookahead.estimate` defaults `d_hat` to the frozen primary (FL-only) model, whose
   magnitude failed on TEST.** Do not wire `d_hat` into the reward as a level. If a lookahead
   level is wanted, `load_model(name="greedy")` did best on TEST (log-RMSE 0.927), but choosing it
   now is post-hoc selection. It needs the full feature set (~0.6 s per case).
5. **Unit test.** I own no test file. Suggested test: `estimate` is deterministic, and on the
   (10,23) full-row case `features["root_fixed_cells"] == 23` and the call terminates. The latter
   was an infinite loop before the level-0 fix.
6. **Final ground truth.** When A1's final `ground_truth.jsonl` lands, re-run
   `lookahead_test.py featurize && lookahead_test.py evaluate` on the new exact hard labels. Change
   `RAW` to point at it, or keep using `ground_truth_deepen_raw.jsonl` if it has grown.
   Featurisation resumes, and the models stay frozen. Especially valuable: exact labels for the
   (13,19) / (12,18) cases still unknown at 2M.

## 7. Reproduce

```bash
cd examples/zarankiewicz/upper_bounds/experiments/E24_difficulty; export ZAR_UB_NO_LLM=1
PY=/Users/jaybhan/Downloads/openevolve/.venv/bin/python
$PY lookahead_features.py --jobs 2      # DEV features -> lookahead_features_dev.jsonl (~16 min at 2 procs, contended)
$PY lookahead_supplement.py --jobs 2    # FL-only pass -> lookahead_features_dev_fl.jsonl (61 s)
$PY lookahead_eval.py                   # DEV report -> lookahead_eval.{json,txt} (~6 min, nested greedy)
$PY lookahead_freeze.py                 # -> lookahead_model.json (already frozen; re-running refits identically)
$PY lookahead_test.py featurize --jobs 2 && $PY lookahead_test.py evaluate   # TEST -> lookahead_test.{json,txt}
$PY lookahead_combo.py                  # complementarity with A4 -> lookahead_combo.{json,txt}
```

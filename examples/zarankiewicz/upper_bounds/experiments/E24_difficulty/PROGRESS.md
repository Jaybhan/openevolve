# E24 / A4: CDCL progress statistics and static LP-slack features as difficulty estimators

Owner A4 (progress-static). Files: `zar_ub/hardness_progress.py` (module, estimator contract),
`experiments/E24_difficulty/progress_{collect,eval,fit,test}.py`, `progress_model*.json`,
`progress_features_{dev,test,test2,test3}.jsonl`, `progress_eval_*.json`, `progress_test*.json`.
Solver processes: 2 at most. No table was modified. Date 2026-09-22.

## 1. Summary

The short CDCL runs that the censored-table pipeline **already makes** carry most of the
difficulty signal. The current label only threw it away. On the HARD regime (true d > 20k, the
cases that are censored on target tables), the pysat run at 20k conflicts gives
`decisions / conflict` and `restarts / kconflict`. A 4-feature log-linear model on these two
numbers plus two static profile features is the **`free20k` model**. The `20k` model adds two
CaDiCaL-binary runs (another ~22k conflicts) on top.

Headline: models fitted on the 4 square dev cells only (262 cases), then frozen, then tested on
509 hard cases in 5 wide cells that were never used for fitting:

| estimator (frozen on dev) | extra solver cost / case | within-cell Spearman | pooled Spearman | log-RMSE |
|---|---|---|---|---|
| current label `fhat` (calibrate_33.json, clipped to [cap, 20 cap]) | 0 | 0.350 | 0.331 | 1.299 |
| `fhat` form refit on dev (log c2000 + log2_volume) | 0 | 0.397 | 0.340 | 1.238 |
| **`free20k`**: ps20k decisions/conflict, restarts/kconflict, distinct row sums, column LP slack | **0** (reuses the table's 20k probe; 22k conflicts if run standalone) | **0.855** | **0.849** | **0.758** |
| **`20k`**: + CaDiCaL-binary 2k/20k statistics | 22k conflicts (44k standalone) | **0.887** | **0.895** | **0.678** |
| `2k` (binary + pysat 2k probes) | ~4k conflicts | 0.485 | 0.529 | 2.446 (level extrapolates badly) |

Leave-one-cell-out over all 8 cells with exact hard labels (571 cases), with nested feature
selection: `20k` 0.830 / log-RMSE 0.580, `free20k` 0.786 / 0.600, static only 0.665 / 0.756, vs
current `fhat` 0.352 / 1.310 (§4.3).
MID regime (2k < d <= 20k), 2k tier, LOCO over 31 cells: 0.613 / 0.478 vs current 0.198 / 2.903 (§4.5).

## 2. Methods

### 2.1 Features (`zar_ub/hardness_progress.py`)
**Family A: CDCL progress.** Every run is a fresh solver with seed 0, so every run is deterministic.
* CaDiCaL **binary** 3.0.1 (`tools/cadical/build/cadical -c CAP --stats -v -n --seed=0`, DIMACS on
  stdin) at CAP = 2 000 and 20 000. The parser (`parse_cadical_output`) reads three parts of the output:
  * The whole statistics block: 250-300 counters per run, kept as `bin<cap>_raw_*` when `full=True`.
  * The verbose report table, as a trajectory. It gives the conflicts, remaining (active) variables,
    trail %, EMA glue/size and tier counts at every report.
  * The glue-usage histogram.

  Curated features built from these (`bin<cap>_*`): fixed and fixed_frac (root-level units),
  eliminated_frac, remaining, irredundant, subsumed, strengthened, decisions/conflict,
  propagations/conflict, restarts per kconflict, ticks/conflict, avg learned size, chrono_frac,
  shrunken/minishrunken fractions, promoted1/2 fractions, improvedglue fraction, reduced fraction,
  backbone units, sweep counts, EMA glue/size, trail %, trajectory slopes over the second half of
  the run, and the share of glue <= 2 clause uses.
* **pysat `cadical195`** is the label solver. Fresh runs at both caps, reading `accum_stats`
  (`ps<cap>_*`): conflicts, decisions, propagations, restarts, and the per-conflict rates.
  The 2k run reproduces the tables' `c2000` exactly on 621/671 checked cases. The other 50 are all
  in (12,13,87) and differ by 1-4 conflicts, which is the stopping noise at the cap.
* **Progress extrapolation** (`prog_*`) from the two binary runs:
  * root-fixed growth per kconflict;
  * the ratio of remaining variables;
  * the linear extrapolation of `remaining` to 0 (`prog_log_lin_to_zero`);
  * the linear extrapolation of the root-fixed count to cover all remaining variables
    (`prog_log_fixed_to_all`);
  * a log-linear decay of `remaining` down to 1 % of the variables (`prog_log_loglin_to_1pct`).

**Family B: static / LP slack** (`st_*`, ~0.02 s/case including the CNF encode; no solver):
* Argument A slack on columns and rows: `(t-1)C(m,s) - sum C(c_j,s)`, absolute and relative.
* Argument D slack per row, `(t-1)C(m-1,s-1) - sum over the r_i lightest columns of C(c-1,s-1)`.
  The report keeps its min (the heaviest row), its mean, and the column form.
* The DGH(v=s-1) slack `rhs - lhs` at the best k (the compact form of `E13_dgh4/dgh.py`), both
  orientations.
* The pair-codegree LP (`lemmas.farkas_system`, constraints F1-F4, solved with HiGHS), both
  orientations:
  * the max uniform slack eps on the row and non-degenerate pair-box constraints, with F1 kept as
    an equality (`lp_eps`);
  * the range [Tmin, Tmax] of sum(lambda) under F2-F4 against T = sum C(c_j,2) (`lp_slackT`);
  * `Lcap`.
* Profile shape: the number of distinct row and column sums, spread, variance, excess ones over w,
  log2 volume (rows and columns), the log2 size of the lex-symmetry group, and nvars / nclauses.

**Tiers:**
* `static`: 0 conflicts.
* `2k`: binary + pysat at 2k, ~4 000 conflicts, 1.6 M propagations, 0.15 s.
* `20k`: + both at 20k, 43 853 conflicts, 9.2 M propagations, 0.94 s per hard case.
* `free20k`: the pysat features + static features. Costs 0 extra conflicts when the 20k label
  probe already ran.

Seconds were measured under ~8-13 load from other agents.

### 2.2 Estimator contract
`estimate(inst, rows, cols, tier="20k"|"2k"|"static", binary=True, pysat=True, full=False, ...)`
returns `{features, d_hat, cost_conflicts, cost_propagations, cost_seconds, meta}`.

How `d_hat` is set:
* If the deepest pysat probe decided the case, `d_hat` is the exact conflict count.
* Otherwise it comes from the frozen model for the tier in `progress_model.json`. By default
  `20k` → "20k", `20k` with `binary=False` → "free20k", and `2k` → "2k".
* The prediction is `max(floor, exp(...))`, with floor = the cap the model was fitted above.
* `predict(model, feats, mean=True)` multiplies by the Duan smearing factor (1.13-1.16 for the
  20k models). Use that for sums of work.

`d_hat_from_probe(inst, rows, cols, conflicts, decisions, restarts)` is the zero-cost path for a
case that the table's existing 20k probe left open.

### 2.3 Protocol
1. **Dev set.** `ground_truth_initial.jsonl` (A1). HARD = every exactly labelled case with
   d > 20k: 262 cases in 4 cells: (11,11,70) 15, (11,12,75) 51, (12,12,81) 93, (12,13,87) 103.
   MID = up to 150 exact 2k < d <= 20k cases per cell (seed 20260922): 2 799 cases in 29 cells.
   Collection: 3 061 cases, 318 s wall on 2 processes, 22.6 M conflicts, 6.67 G propagations.
2. **Screening and model selection on dev.** Per-feature Spearman with d: pooled, per cell,
   n-weighted within-cell mean, and sign agreement. Models are log-linear, `log d = a + sum b_k z_k`
   on ≤ 4 standardised features, with log1p for count features. They were evaluated leave-one-cell-out
   with **nested** greedy forward selection: inner LOCO log-RMSE on the training cells, a stop at
   < 0.5 % improvement, and a collinearity guard at |r| > 0.95. Two feature pools per tier:
   * the 60 features that screen best (the screen used all cells, a mild leak into the pool only);
   * all features in the tier (no screening; fully honest).
3. **Freeze.** `progress_fit.py` fitted the 20k / free20k / 2k models on dev only and wrote
   `progress_model_frozen_dev.json`. The file is deterministic: refitting gives the same
   coefficients.
4. **Held-out tests.** Nothing was tuned on these.
   * **test**: cells in `ground_truth_interim.jsonl` that were not in dev: (10,20,103)p 105,
     (11,21,117)p 23, (9,23,104) 4. 132 cases.
   * **test2**: rows that A1 deepened later (`ground_truth_deepen_raw.jsonl`): (9,23,104) 136 and
     the new wide pure table (9,18,86) 41. 177 cases.
   * **test3**: the next deepening batch: (9,18,86) 134 more and the new table (10,19,99) 66.
     200 cases.
5. **Final model.** `progress_model.json` (the module default) was refit on dev + test + test2:
   8 cells, 571 hard cases, 3 396 MID+HARD cases for the 2k model. Its expected performance is the
   8-cell LOCO in §4.3. Its only unseen check is test3.

## 3. Per-feature results, HARD regime (8 cells, 571 cases, within-cell Spearman with true d)

| family | feature | within | pooled | sign | (10,20,103)p | (11,11,70)p | (11,12,75)p | (11,21,117)p | (12,12,81)p | (12,13,87)p | (9,18,86)p | (9,23,104) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A pysat | `ps20k_decs_per_conf` (= decisions) | +0.67 | +0.79 | 8/8 | +0.80 | +0.67 | +0.62 | +0.77 | +0.57 | +0.60 | +0.66 | +0.69 |
| A pysat | `ps20k_restarts` | +0.20 | +0.34 | 8/8 | +0.35 | +0.31 | +0.27 | +0.48 | +0.17 | +0.21 | +0.02 | +0.07 |
| A pysat | `ps20k_propagations` | +0.18 | +0.39 | 7/8 | +0.25 | -0.16 | +0.16 | +0.15 | +0.12 | +0.01 | +0.34 | +0.29 |
| A pysat | `ps2k_decs_per_conf` | +0.15 | +0.32 | 7/8 | +0.44 | +0.28 | +0.07 | +0.43 | +0.08 | +0.13 | +0.20 | -0.06 |
| A pysat | `ps2k_conflicts` (= c2000) | +0.05 | +0.06 | 5/8 | | | | | | | | |
| A binary | `bin20k_improvedglue_frac` | -0.72 | -0.64 | 8/8 | -0.77 | -0.87 | -0.47 | -0.87 | -0.68 | -0.69 | -0.63 | -0.80 |
| A binary | `bin20k_subsumed` | -0.71 | -0.78 | 8/8 | -0.77 | -0.82 | -0.54 | -0.58 | -0.67 | -0.70 | -0.64 | -0.82 |
| A binary | `bin20k_fixed_frac` (root units) | -0.71 | -0.49 | 8/8 | -0.70 | -0.42 | -0.60 | -0.97 | -0.48 | -0.82 | -0.70 | -0.82 |
| A binary | `bin20k_remaining` | +0.68 | +0.67 | 8/8 | +0.66 | +0.48 | +0.64 | +0.96 | +0.55 | +0.72 | +0.56 | +0.79 |
| A binary | `bin20k_irredundant` | +0.68 | +0.64 | 8/8 | +0.74 | +0.35 | +0.53 | +0.96 | +0.54 | +0.66 | +0.57 | +0.81 |
| A binary | `bin20k_solved` (the 3.0.1 binary refutes it within 20k) | -0.33 | -0.34 | 7/7 | | | | | | | | |
| A progress | `prog_log_fixed_to_all` | +0.67 | +0.55 | 8/8 | +0.72 | +0.64 | +0.67 | +0.86 | +0.56 | +0.66 | +0.58 | +0.71 |
| A progress | `prog_log_loglin_to_1pct` | +0.66 | +0.74 | 8/8 | +0.64 | +0.44 | +0.61 | +0.95 | +0.55 | +0.71 | +0.53 | +0.75 |
| A progress | `prog_fixed_growth_per_kconf` | -0.64 | -0.38 | 8/8 | -0.71 | -0.62 | -0.66 | -0.85 | -0.54 | -0.64 | -0.45 | -0.68 |
| B static | `st_argD_min_rel` (heaviest-row Argument D slack) | +0.53 | +0.26 | 8/8 | +0.57 | +0.67 | +0.48 | +0.74 | +0.44 | +0.42 | +0.00 | +0.76 |
| B static | `st_log2_vol_minus_sym` | +0.49 | +0.63 | 7/8 | +0.72 | -0.30 | +0.13 | +0.79 | +0.36 | +0.47 | +0.39 | +0.61 |
| B LP | `st_lp_eps_min` (pair-codegree LP max slack) | +0.44 | +0.39 | 8/8 | +0.58 | +0.40 | +0.15 | +0.70 | +0.30 | +0.32 | +0.14 | +0.65 |
| B LP | `st_row_lp_slackT_rel` | +0.40 | +0.27 | 8/8 | +0.37 | +0.53 | +0.06 | +0.65 | +0.13 | +0.25 | +0.25 | +0.81 |
| B static | `st_log2_volume` (the current label's only live input) | +0.35 | +0.59 | 7/8 | | | | | | | | |
| B static | `st_argA_min_rel`, `st_dgh_min_rel` | +0.11, +0.13 | -0.45, -0.43 | 5/7 | | | | | | | | |

Signs:
* More root-fixed variables, more subsumption and more glue improvements by 20k conflicts mean
  an **easier** case.
* More decisions per conflict and more remaining / irredundant clauses mean a **harder** case.
* More counting slack (Argument D, LP) means **harder**. This is the sign the literature pilot
  reported: static features that are ≤ 0 prune a case. Positive, they rank it weakly.

## 4. Model results

### 4.1 Dev (4 square cells, 262 hard cases), LOCO with nested selection (`progress_eval_dev_hard.json`)
| model | within ρ | pooled ρ | log-RMSE |
|---|---|---|---|
| current `fhat` (in-sample on 3 of these cells) | 0.139 | 0.249 | 1.128 (clipped 0.838) |
| `fhat` form, LOCO | 0.139 | 0.083 | 0.564 |
| constant, LOCO | - | - | 0.583 |
| static (46 features) | 0.359 | 0.251 | 0.589 |
| 2k (pool of 60 / all 261) | 0.414 / 0.377 | 0.375 / 0.402 | 0.528 / 0.528 |
| free20k (62 features) | 0.595 | 0.536 | 0.495 |
| 20k (pool of 60 / all 645 incl. raw) | **0.769 / 0.751** | 0.742 / 0.710 | **0.393 / 0.407** |

Frozen dev models (`progress_model_frozen_dev.json`):
* `20k` = [bin20k_improvedglue_frac, ps20k_decs_per_conf, ps20k_restarts_per_kconf (log1p), bin20k_units_probe]
* `free20k` = [ps20k_decs_per_conf, ps20k_restarts_per_kconf (log1p), st_distinct_rows, st_col_lp_slackT]
* `2k` = [bin2k_promoted1_frac, bin2k_chrono_frac, bin2k_reduced_frac, bin2k_decisions (log1p)]

### 4.2 Frozen dev models on never-seen cells
| set | n | current fhat (clipped) ρ / log-RMSE | fhat form refit ρ / log-RMSE | frozen free20k ρ / log-RMSE | frozen 20k ρ / log-RMSE | frozen 2k ρ / log-RMSE |
|---|---|---|---|---|---|---|
| test: (10,20,103)p, (11,21,117)p, (9,23,104) | 132 | 0.350 / 1.430 | 0.424 / 1.270 | 0.869 / 0.680 | **0.900 / 0.753** | 0.430 / 2.524 |
| test2: (9,23,104), (9,18,86)p | 177 | 0.531 / 1.353 | 0.611 / 1.304 | 0.807 / 0.983 | **0.871 / 0.736** | 0.440 / 2.562 |
| test3: (9,18,86)p, (10,19,99)p | 200 | 0.310 / 1.151 | 0.312 / 1.155 | 0.893 / 0.550 | **0.894 / 0.562** | 0.545 / 2.284 |
| all three (5 cells) | 509 | 0.350 / 1.299 | 0.397 / 1.238 | 0.855 / 0.758 | **0.887 / 0.678** | 0.485 / 2.446 |

ρ in this table is the n-weighted within-cell Spearman.

Per cell, frozen 20k:
* (10,19,99)p 0.89
* (10,20,103)p 0.90
* (11,21,117)p 0.88
* (9,18,86)p 0.87
* (9,23,104) 0.90, a Tan-2022 target cell

The current `fhat` gets -0.01 on (10,19,99)p and 0.21 on (9,18,86)p.

### 4.3 Leave-one-cell-out over all 8 exact-hard cells (571 cases; curated features)
| model | within ρ | pooled ρ | log-RMSE | clipped log-RMSE |
|---|---|---|---|---|
| current `fhat` | 0.352 | 0.593 | 1.310 | 1.167 |
| `fhat` form, LOCO | 0.352 | 0.546 | 0.990 | 0.990 |
| constant, LOCO | - | -0.467 | 1.281 | 1.281 |
| static | 0.665 | 0.737 | 0.756 | 0.775 |
| 2k | 0.648 | 0.717 | 0.796 | 0.799 |
| free20k | **0.786** | 0.844 | **0.600** | 0.629 |
| 20k (all 165 curated) | **0.830** | 0.870 | **0.580** | 0.601 |
| 20k, pool 60 (raw statistics included) | 0.827 | 0.874 | 0.582 | 0.604 |

Per cell, 20k: (10,20,103)p 0.88, (11,11,70)p 0.90, (11,12,75)p 0.80, (11,21,117)p 0.90,
(12,12,81)p 0.81, (12,13,87)p 0.81, (9,18,86)p 0.77, (9,23,104) 0.88.

`free20k` picks `ps20k_decs_per_conf` + `ps20k_restarts_per_kconf` in **every** fold.
On its own, restarts per kconflict correlates weakly positively with d (+0.20). Once
decisions/conflict is in the model, its coefficient is negative: among cases with equal
decisions/conflict, more restarts means an easier case.

Final model (`progress_model.json`, fitted on these 8 cells):
* `20k` = [ps20k_decs_per_conf +1.18, ps20k_restarts_per_kconf (log1p) -0.51, bin20k_promoted1_frac -0.35,
  st_argA_row_rel +0.21] (standardised coefficients), intercept 11.34, smear 1.13
* `free20k` = [ps20k_decs_per_conf +1.39, ps20k_restarts_per_kconf -0.57, st_distinct_rows +0.17,
  st_row_lp_lcap -0.08]

On test3 it reaches ρ 0.894 / log-RMSE 0.489 (20k) and 0.885 / 0.511 (free20k).

### 4.4 Total work and the tail (test cells, frozen models)
The log-scale models predict the conditional median. Fitted on square cells only, they
under-predict the level of wide cells: the mean log bias of the frozen 20k model is
-0.45 on (10,20,103)p and -0.55 on (11,21,117)p. Sum over (10,20,103)p of d_hat / sum of d:
* frozen 20k: 0.45
* frozen free20k: 0.96

The current label's sum ratio is 1.04, but only because it sits at the 20·cap ceiling for most of
these cases, which is right on average by luck. Its within-cell ρ is 0.35.

Of the true top-10 % work in (10,20,103)p:
* the current label's top 10 % holds 25 %;
* the free20k top 10 % holds 57 %.

When the reward needs sums, use `mean=True` (the smearing factor) and a model fitted on cells
of the same shape.

### 4.5 MID regime (2k < d <= 20k; 2k tier; dev + test cells, 31 cells, 2 825 cases, LOCO)
| model | within ρ | pooled ρ | log-RMSE |
|---|---|---|---|
| current `fhat` (clip [2k, 40k]) | 0.198 | 0.301 | 2.903 (clipped 1.324) |
| `fhat` form, LOCO | 0.198 | 0.290 | 0.596 |
| constant | - | - | 0.623 |
| static | 0.449 | 0.487 | 0.551 |
| 2k (binary + pysat 2k) | **0.613** | **0.649** | **0.478** |

Best single features:
* `bin2k_remaining` +0.53
* `bin2k_fixed_frac` -0.51
* `bin2k_irredundant` +0.49
* `bin2k_promoted1_frac` -0.46
* `st_log2_vol_minus_sym` +0.38

The pysat 2k statistics are near zero (`ps2k_decs_per_conf` +0.23; `ps2k_props_per_conf` -0.02).

The current `fhat`'s log-RMSE of 2.9 comes from the wide cells, 3.8-6.3 on (10,23), (11,23),
(12,18), (13,19), (16,17), where `g·log2_volume` extrapolates.

## 5. Negative results and caveats

* **The 2k probe contains almost no ordering signal inside the hard set** (this confirms E12):
  * pysat 2k statistics: within ρ ≤ 0.22;
  * `c2000`: 0.05;
  * the best 2k model: 0.49-0.65, with a level error of 2.4 log units on unseen wide cells.

  The signal appears between 2k and 20k conflicts, and that 20k run is already paid for on every
  censored case.
* **The literature pilot's sign did not replicate.** Its signal was propagations/conflict
  (ρ ≈ -0.5 at 20k on z(10,14) w=78). Here `ps20k_props_per_conf` is +0.18 within-cell
  (-0.16 … +0.34 per cell). The robust pysat signal is decisions/conflict.
* **The progress extrapolations (`prog_*`) rank well but add little.** They reach ρ 0.64-0.67 but
  are not selected over the raw counters they are built from (root-fixed, remaining).
  Uncalibrated, they are not usable as d_hat: `prog_log_fixed_to_all` ≈ e^15 conflicts against a
  true ~1e5. So `d_hat` is the calibrated model, not a raw extrapolation.
* **Static counting / LP slack ranks moderately and never selects a case.** It reaches ρ 0.40-0.53
  (Argument D slack, LP eps, LP T-slack), all positively signed (more slack = harder). A static-only
  model gives 0.665 on the 8-cell LOCO but only 0.359 on the 4 square dev cells. The variation in
  profiles on wide cells is what gives it its signal. Argument A and DGH slack are weak (0.11-0.13)
  and change sign between cells.
* **The labels are selected.** "Exact hard" means refuted within 2M conflicts. At collection time,
  A1's deepening had 22 cases still unknown at 2M ((11,21,117)p 7, (10,20,103)p 6, (9,23,104) 6,
  (10,19,99)p 3). They are excluded, so the fits have never seen d > 2M. Estimates for the very
  hardest cases are extrapolations, floored at 20k and not capped.
* **The level is the weak point.**
  * The log-RMSE on unseen cells is 0.55-0.98, a typical factor of 1.7-2.7.
  * Models fitted on square cells under-predict wide cells by e^0.45-0.55.
  * Within-cell ordering transfers much better than the level.
* **The two solvers differ.** The binary is CaDiCaL 3.0.1 and the label solver is pysat cadical195
  (1.9.5), so the binary's statistics are proxies, not the label solver's own. On some hard cases
  the binary refutes within 20k conflicts (`bin20k_solved`), which is itself a feature.
  `free20k` needs no binary at all.
* **Selection was mildly optimistic.** The screened pool of 60 was chosen using all cells. The
  all-features pool (no screening) gives nearly the same numbers (§4.1, §4.3). The frozen-model
  tests in §4.2 involve no selection at all.
* **The feature file is large.** `progress_features_dev.jsonl` is 80 MB because of the raw
  counters (`full=True`). Delete and regenerate it with `progress_collect.py` if space matters.

## 6. Reproduce
```bash
cd examples/zarankiewicz/upper_bounds; export ZAR_UB_NO_LLM=1; PY=/Users/jaybhan/Downloads/openevolve/.venv/bin/python
$PY experiments/E24_difficulty/progress_collect.py --procs 2                                   # dev features (~5 min)
$PY experiments/E24_difficulty/progress_eval.py --tag dev                                      # screening + LOCO (dev)
$PY experiments/E24_difficulty/progress_fit.py --out experiments/E24_difficulty/progress_model_frozen_dev.json
$PY experiments/E24_difficulty/progress_collect.py --gt experiments/E24_difficulty/ground_truth_interim.jsonl \
    --out experiments/E24_difficulty/progress_features_test.jsonl --exclude-cells-of experiments/E24_difficulty/progress_features_dev.jsonl \
    --cells m10_n20_s3_t3_w103_pure,m11_n21_s3_t3_w117_pure,m9_n23_s3_t3_w104
$PY experiments/E24_difficulty/progress_collect.py --gt experiments/E24_difficulty/ground_truth_deepen_raw.jsonl --only hard \
    --out experiments/E24_difficulty/progress_features_test2.jsonl --exclude-cells-of experiments/E24_difficulty/progress_features_dev.jsonl \
    --also-done experiments/E24_difficulty/progress_features_test.jsonl        # (test3: same with test2 added to --also-done)
$PY experiments/E24_difficulty/progress_test.py --test experiments/E24_difficulty/progress_features_test.jsonl,experiments/E24_difficulty/progress_features_test2.jsonl,experiments/E24_difficulty/progress_features_test3.jsonl
$PY experiments/E24_difficulty/progress_eval.py --feat experiments/E24_difficulty/progress_features_dev.jsonl,experiments/E24_difficulty/progress_features_test.jsonl,experiments/E24_difficulty/progress_features_test2.jsonl --tag all_hard --curated-only --regimes hard
$PY experiments/E24_difficulty/progress_fit.py --feat experiments/E24_difficulty/progress_features_dev.jsonl,experiments/E24_difficulty/progress_features_test.jsonl,experiments/E24_difficulty/progress_features_test2.jsonl
```
Total solver spend for this report:
* dev: 22.6 M conflicts / 6.67 G propagations
* test: 6.2 M / 1.43 G
* test2: 7.8 M / 1.69 G
* test3: 8.8 M / 1.72 G
* total: 45.4 M conflicts / 11.5 G propagations, ≈ 10 min wall on 2 processes

## 7. TODOs for the integrator (shared files are not edited here)

1. **`zar_ub/difficulty.py::label_case` / `continue_label`.** Record the pysat `accum_stats()`
   `decisions` and `restarts` of every cap run in the probe dict. `solve.SolveResult` already has
   `decisions` but drops `restarts`: add `restarts=int(st.get("restarts", 0))` in
   `zar_ub/solve.py`. Then compute the censored label as
   `hardness_progress.d_hat_from_probe(inst, rows, cols, conflicts, decisions, restarts)`
   instead of `fhat`. This adds zero solver cost.
   * Keep the floor at cap.
   * Drop the 20·cap ceiling, or raise it to about 100·cap: true hard d reaches 1.7 M = 85·cap.
   * Keep `censored=True`.
2. **Existing target tables have no decisions/restarts stored.** Re-probing their censored cases
   at 20k costs 20k conflicts per case (~0.4 s), or run `estimate(..., binary=False)`. Do not
   write to `cache/case_table_*.json` from here: that is the integrator's decision (hard rule 3).
3. **Refit when the final GT lands.** Re-run `progress_collect.py` on A1's final
   `ground_truth.jsonl` with `--also-done` pointing at the existing feature files (only new cases
   are run), then refit with `progress_fit.py` over all feature files. Cells whose shape is not yet
   covered, e.g. tall (m > n) or s,t ≠ 3, will extrapolate the level.
4. **Sums of work in the reward.** Where the reward sums d (gain_I, tail_I), use
   `predict(..., mean=True)` (smearing). Or better, normalise per cell so that only the
   within-cell ordering and relative magnitudes matter. That ordering is the part that transfers
   (ρ ≈ 0.85-0.9).
5. **Combining with A2/A3.** `estimate()` follows the shared contract. Model on `features` from
   A2/A3/A4 jointly: `ps20k_decs_per_conf` and `ps20k_restarts_per_kconf` are the two features to
   include in any joint model. Check whether lookahead or sampling adds anything on top of them in
   the hard regime.
6. **The binary tier is optional.** It adds 0.03-0.07 ρ and lowers log-RMSE by 0.02-0.25
   (§4.2, §4.3) for 22k more conflicts. It is worth it only where CPU is cheap relative to label quality.

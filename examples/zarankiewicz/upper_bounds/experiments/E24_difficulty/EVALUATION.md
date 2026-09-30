# E24 / D5-evaluate: head-to-head comparison of the difficulty estimators

Owner: D5-evaluate. Date: 2026-09-23.

**Files**
- `zar_ub/hardness_model.py`: the final estimator.
- `experiments/E24_difficulty/hardness_model.json`: the fitted winner.
- Scripts `eval_collect.py`, `eval_masks.py`, `eval_analyze.py`, `eval_report.py` and `eval_final.py`.
- Outputs `features_{evalset,lookahead,sampling,progress,gaintables,predictions,final_check}.jsonl`, `eval_results.json` (every number below) and `eval_results.txt` (every table, including the models not shown here).

**Rules followed**
- No case table was modified.
- No LLM or network calls, no `lake build`.
- At most 10 solver processes were used.
- Fixed seeds throughout.
- No selection or fitting used a held-out cell; all feature selection is nested inside the CV folds.

## 1. Answer

**The current censored label is the weakest estimator on the hard cases.** On the evaluation set, the pipeline's label for d > 20k is `clip(fhat, 20k, 400k)`.
- Within-cell Harrell's C is **0.552**, close to chance.
  - It is a constant 400k, so it has no within-cell ranking at all, in 8 of the 15 cells.
  - Averaged over the 7 cells where it varies, within-cell Spearman is 0.324.
- log-RMSE is 1.12.
- Simulated reward error: 0.061 mean absolute error in `gain_I` and **0.164 in `tail_I`** over 106 real kill masks.

**Winner: `hardness_model.json`, a 6-feature log-linear model from the "free" tier.**
- The tier is the pysat 20k-probe statistics, static/LP slack and lookahead. It uses no CaDiCaL binary and no sampling.
- Held out by leave-one-shape-out:

| metric (HARD regime, held-out, 18 cells, 2,429 cases incl. 191 open at 2M) | current label | **winner** | best of any estimator (cost) |
|---|---|---|---|
| mean within-cell Spearman (exact cases) | 0.324 (constant in 8/15 cells) | **0.837** | 0.899 gbm[all] (~92k conflicts) |
| within-cell Harrell's C (open-at-2M censored) | 0.552 | **0.847** | 0.878 gbm[all] |
| pooled C | 0.706 | **0.865** | 0.900 gbm[all] |
| log-RMSE (exact cases) | 1.116 | **0.510** | 0.394 gbm[all] |
| TARGET cells (tan2022, 601 cases): within ρ / C | 0.514 / 0.560 | **0.835 / 0.885** | 0.887 / 0.905 gbm[all] |
| library survivors only: within ρ / C | 0.064 / 0.506 | **0.821 / 0.858** | 0.874 / 0.872 greedy[all] |
| **reward-gain error**, 106 real masks (mean abs. error of gain_I / tail_I) | 0.061 / 0.164 | **0.020 / 0.046** (best of all estimators) | the same |
| reward-gain error, 150-450 random masks (5/20/50 %) | 0.013 / 0.025 / 0.033 | **0.007 / 0.014 / 0.019** | 0.007 / 0.012 / 0.016 gbm[all] |
| extra cost per censored case | 0 | **24.1k conflicts, 18.1M propagations, 1.3 s** (0 conflicts and 0.8 s if the table's 20k probe stores decisions and restarts) | |

**Formula** (natural log; z = (T(x) − mu)/sd with the stored mu/sd; T = log1p where marked):

```
log d_hat = 11.301
          + 0.900 z[ps20k_decs_per_conf]            (pysat cadical195 20k probe: decisions / conflict)
          - 0.381 z[log1p ps20k_restarts_per_kconf] (restarts per 1000 conflicts)
          + 0.364 z[knfl_rand_mean_log_nodes]       (lookahead: Knuth tree-size estimate after the failed-literal fixpoint)
          + 0.161 z[st_distinct_rows]               (static: number of distinct row sums)
          - 0.240 z[log1p la1_failed_neg]           (lookahead: failed negative cell literals)
          - 0.154 z[st_argD_row_mean_rel]           (static: mean row Argument-D slack)
d_hat = max(20000, exp(log d_hat) * 1.121)          (x1.121 = Duan smearing, for sums; ceiling 2M recommended)
```

**Why this model and not the one that ranks best** (the selection rule is in section 5):
- It has the **lowest reward-gain error of every estimator**. A table-bootstrap 95 % CI shows it is significantly lower than free+BIN, all, A4 free20k, and every sampling combination.
- It costs about a quarter of the conflicts of the best-ranking models.
- Adding binary CaDiCaL or sampling features raises within-cell Spearman by 0.04-0.05, but increases gain error.

**What each proposal technique contributed** (section 4.4):
- **Lookahead (AlphaMapleSAT / march / Knuth):**
  - About +0.035 within-cell ρ and −0.005 gain error inside the free tier (A4 free20k 0.802 → winner 0.837; gain error 0.0256 → 0.0201, CI of the difference [0.002, 0.009]).
  - About 0 once binary CDCL statistics are present: all minus LA equals all.
  - It is the best **zero-conflict** signal: ρ 0.71 on hard cases (with the static features), 0.59 on MID.
- **Sampling (Chivilikhin d-hardness):**
  - A real standalone estimator: ρ 0.721 / C 0.771 alone, 0.665 on MID.
  - Worth +0.014-0.048 ρ on top of CDCL features: A4 free20k 0.802 → +samp 0.850; all 0.888 vs all minus SA 0.874.
  - It **does not reduce gain error**: +samp 0.026 vs winner 0.020.
  - Spending its ~46k conflicts on continuing the direct solve instead is strictly better (section 4.5).
- **LP / counting slack (static):** a weak ranker alone (single-best ρ 0.38, greedy 0.62) and 0 on top of everything else (all minus ST equals all). Two static terms survive in the winner with small weights.
- **The real driver is CDCL progress from the 20k probe the pipeline already runs** (decisions per conflict and restarts, A4). It was not a proposal technique.

**User's question 2 ("rules that only help one cell, like DGH on wide cells, get under-rewarded").** This is only partly a difficulty-label problem. With the current label, DGH on (9,18,86) gets:
- gain_I = 0.269 and tail_I = 0.499;
- the ground truth is 0.110 and 0.000.

So the current label **over-credits** DGH there, by 2.4× on gain and with entirely false tail credit. Every new estimator gets it right: gain 0.09-0.12, tail 0. DGH kills the easier wide cases (A1). Accurate difficulty therefore makes DGH's per-cell gain smaller, not larger. Rewarding narrow rules is a reward-design question (E25), not a labelling one.

## 2. Data and protocol

**Ground truth.**
- Source: A1's final `ground_truth.jsonl` (35,081 rows).
- Evaluation set (`features_evalset.jsonl`):
  - **HARD:** all 2,238 exact d > 20k cases, plus the 191 still open at 2M. They are spread over 18 cells: 12 pure and 6 tan2022 target cells. The per-cell metrics use the 15 cells with at least 8 hard cases.
  - **MID:** a seeded sample of up to 120 cases per cell with 2k < d ≤ 20k, 2,704 cases in total.
  - The 15,262 cases open only at 20k are excluded, because their true d is unknown.

**Estimators computed on every one of the 5,133 cases, at their recommended settings** (`eval_collect.py`, 10 processes, 38 min wall, 0 errors):
- `hardness_lookahead.estimate` with defaults (full feature set).
- `hardness_sampling.estimate` with DEFAULT: knuth:row:8, N100, b5000, total 50k.
- `hardness_progress.estimate(tier="20k")`: pysat and binary runs at 2k and 20k, plus static/LP features.
- The fhat baseline and the c2000/c20000 probes are read from the ground truth. c20000 is constant for every hard case.
- Features are prefixed `la:`, `sa:`, `pr:st_` (static/LP), `pr:ps` (pysat probe stats) and `pr:bin`/`pr:prog` (CaDiCaL binary).

**Cross-validation.**
- **Leave-one-shape-out (LOSO):** the fold is the shape (m, n), which is stricter than leave-one-cell-out.
- **square→wide:** train on the near-square shapes (n/m < 1.4), test on the 10 wide shapes (n/m ≥ 1.4). The wide shapes include the target cells (9,23), (10,20), (10,22), (12,18) and (13,19). (16,17) is near-square.
- **wide→square:** the reverse.
- Models are fitted on exact hard cases only. Open-at-2M cases enter only Harrell's C, as right-censored.

**Models.**
- The current label, raw fhat, fhat refit, and a constant.
- The as-shipped frozen module d_hats. These are partly in-sample: A4's model was fitted on 8 of these cells and A3's calibration on 4.
- Each family's single best feature, chosen inside each fold, as log-linear and as isotonic fits.
- Sampling mu calibrated with unit slope (A3's form), and by OLS.
- Pre-specified combinations: A4 free20k, +FL, +samp.
- Nested greedy forward selection per cost tier:
  - inner LOSO log-RMSE;
  - at most 6 features;
  - stop when the gain is below 0.5 %;
  - guard against pairs with |r| > 0.95.
- HistGradientBoosting and RandomForest (scikit-learn 1.9.1) per tier.
- "Direct" hybrids: continue the solve to C conflicts, then rank the still-open cases by a model.

**Reward-gain error** (`eval_masks.py`, `gain_eval` in `eval_analyze.py`).
- **Tables.** Every cell with complete (or uniformly sampled) hard ground truth and at least 5 hard library survivors: 15 tables in survivors mode (library `baseline_lean_mask` applied, which is what `gain_I` sees) and 15 in "all cases" mode (library ignored, so argD is a live mask).
  - The three sampled target tables, (12,18), (13,19) and (16,17), weight each sampled censored case by (#censored)/(#sampled): 15.4, 25.5 and 13.2.
  - Open-at-2M cases take true d = 2M, a lower bound.
- **Target-mode labelling is simulated.**
  - A case with true d ≤ 20k keeps its exact d.
  - Every other case takes the estimator's LOSO-held-out value, clipped to [20k, ceiling]. The ceiling is 400k (today's 20·cap), 2M (100·cap) or none.
- **Masks.**
  - argD (`cases.kill_row_argument_d | kill_col_argument_d`).
  - DGH (`E13_dgh4/dgh.py`).
  - Farkas certificates: `lemmas.farkas_certificate` on up to 40 seeded survivors per table, applied with `lemmas.mirror_kill`. There is one mask per certificate plus their union, 43 masks in all-cases mode.
  - 5 profile-threshold rules per table: row-max, col-max, volume low/high, distinct columns. These are rule-like and correlated with difficulty in both directions.
  - 10 seeded random masks each at 5/20/50 % density.
- **Metric:** the mean |gain_est − gain_true| and |tail_est − tail_true| over non-empty masks. "Real" means argD + DGH + Farkas + rules.

## 3. Cost per case (measured under load from other agents; seconds are inflated)

| component | regime | conflicts mean (max) | propagations mean | seconds mean / p90 / max |
|---|---|---|---|---|
| pysat 2k+20k probe (free20k features) | hard | 22,002 (22,010) | 5.05 M | 0.55 / 0.66 / 2.6 |
| CaDiCaL binary 2k+20k | hard | 21,884 (22,016) | 5.07 M | 0.58 / 0.67 / 3.2 |
| lookahead (full, BCP only; "conflicts" = UP conflicts) | hard | 2,077 (2,908) | 13.1 M trail literals | 0.78 / 1.39 / 5.3 |
| sampling (default) | hard | 46,433 (54,968) | 17.6 M | 3.02 / 4.45 / 15.4 |
| sampling | mid | 20,142 | 8.8 M | 2.70 / 5.49 / 25.2 |
| **winner** (pysat 20k probe + full lookahead + static LP) | hard | **~24.1k** | **~18.1 M** | **~1.33** (predict() check: 23.9-24.5k conflicts, 0.9 s per case) |
| all features (the best ranker) | hard | ~92k | ~41 M | ~4.9 (max above 15 s on 16×17) |

**Projected relabel of every censored case on the target tables** (12,246 cases, 6,663 of them library survivors, over 7 tables; 12 workers). Seconds come from the per-cell measured means; (10,23) and (11,23) use a log-log fit.

| configuration | total conflicts | wall time, 12 workers (all / survivors only) |
|---|---|---|
| winner, fresh 20k re-probe + lookahead | 0.30 G | **0.69 h / 0.39 h** |
| winner, if the tables' existing 20k probes had stored decisions and restarts | 0.027 G (UP only) | 0.46 h / 0.26 h |
| + CaDiCaL binary (free+BIN) | 0.57 G | 0.93 h / 0.53 h |
| + sampling (all) | 1.20 G | 2.40 h / 1.35 h |

Every option is within the conflict cap (≤ ~100k conflicts per case). The mean time for "all" is 4.9 s, at the edge of the ~5 s cap, and sampling alone exceeds 5 s on 16×17: mean 6.8 s, p90 9.5 s under load.

## 4. Results

### 4.1 HARD regime, leave-one-shape-out (from `eval_results.txt`)

| model | needs | within ρ | pooled ρ | C within | C pooled | log-RMSE | gain MAE real / tail MAE real |
|---|---|---|---|---|---|---|---|
| current label clip(fhat, 20k, 400k) | – | 0.324* | 0.508 | 0.552 | 0.706 | 1.116 | 0.061 / 0.164 |
| fhat refit (LOSO) | – | 0.308 | 0.502 | 0.631 | 0.713 | 0.918 | 0.053 / 0.182 |
| constant (LOSO) | – | – | – | 0.500 | 0.336 | 1.136 | 0.055 / 0.155 |
| c20000 probe | 20k | – | – | 0.500 | 0.500 | 1.781 | 0.065 / 0.157 |
| lookahead FL-only (A2 primary) | LA | 0.615 | 0.647 | 0.749 | 0.768 | 0.815 | 0.040 / 0.110 |
| lookahead greedy | LA | 0.691 | 0.753 | 0.782 | 0.811 | 0.707 | 0.031 / 0.088 |
| static/LP greedy | ST | 0.624 | 0.714 | 0.762 | 0.793 | 0.772 | 0.038 / 0.098 |
| zero-conflict tier: static + lookahead greedy | ST+LA | 0.711 | 0.771 | 0.790 | 0.819 | 0.679 | 0.028 / 0.084 |
| sampling, mu with unit-slope calibration | SA | 0.721 | 0.823 | 0.771 | 0.832 | 0.828 | 0.050 / 0.157 |
| sampling greedy | SA | 0.724 | 0.805 | 0.776 | 0.827 | 0.644 | 0.042 / 0.136 |
| pysat-probe greedy | PS | 0.792 | 0.842 | 0.823 | 0.847 | 0.566 | 0.027 / 0.060 |
| A4 free20k (4 fixed features) | PS+ST | 0.802 | 0.858 | 0.826 | 0.855 | 0.547 | 0.026 / 0.058 |
| A4 free20k + sampling | PS+ST+SA | 0.850 | 0.903 | 0.850 | 0.880 | 0.485 | 0.026 / 0.069 |
| **greedy free (winner's procedure)** | **PS+ST+LA** | **0.837** | **0.873** | **0.847** | **0.865** | **0.510** | **0.020 / 0.046** |
| greedy free + sampling | +SA | 0.853 | 0.893 | 0.850 | 0.875 | 0.493 | 0.027 / 0.067 |
| greedy free + binary | +BIN | 0.874 | 0.896 | 0.865 | 0.877 | 0.485 | 0.026 / 0.055 |
| greedy all | all | 0.888 | 0.920 | 0.872 | 0.891 | 0.451 | 0.028 / 0.057 |
| GBM free | PS+ST+LA | 0.858 | 0.889 | 0.856 | 0.874 | 0.472 | 0.023 / 0.102 |
| GBM free + binary | +BIN | 0.889 | 0.921 | 0.875 | 0.893 | 0.415 | 0.020 / 0.063 |
| GBM all | all | 0.899 | 0.933 | 0.878 | 0.900 | 0.394 | 0.023 / 0.101 |
| RF all | all | 0.883 | 0.924 | 0.867 | 0.893 | 0.434 | 0.026 / 0.110 |
| (as shipped) A4 frozen `20k` d_hat, **in-sample on 8 cells** | PS+BIN+ST | 0.847 | 0.892 | 0.852 | 0.875 | 0.472 | 0.023 / 0.057 |
| (as shipped) A3 frozen sampling d_hat | SA | 0.721 | 0.823 | 0.771 | 0.833 | 0.845 | 0.049 / 0.162 |
| (as shipped) A2 frozen lookahead hard d_hat | LA | 0.556 | 0.461 | 0.724 | 0.703 | 1.147 | 0.048 / 0.132 |

\* Mean over the 7 cells where the current label is not a constant.

**Per-cell within ρ, current label → winner's procedure:**

| cell | current label | winner's procedure | note |
|---|---|---|---|
| (10,14,78)p | – | 0.79 | |
| (10,19,99)p | 0.40 | 0.90 | |
| (10,20,103)p | 0.35 | 0.92 | |
| (11,11,70)p | – | 0.82 | |
| (11,12,75)p | – | 0.81 | |
| (11,21,117)p | – | 0.86 | |
| (12,12,81)p | – | 0.74 | |
| (12,13,87)p | 0.18 | 0.88 | the E11 acceptance cell; old 0.39 |
| (12,18,109) | 0.34 | 0.84 | |
| (13,13,93)p | 0.23 | 0.86 | |
| (13,19,123) | – | 0.84 | |
| (16,17,134) | – | 0.81 | |
| (9,16,78)p | 0.19 | 0.75 | the weakest cell |
| (9,18,86)p | 0.21 | 0.90 | |
| (9,23,104) | 0.69 | 0.85 | |

Harrell's C per cell for the winner's procedure is 0.78-0.92.

### 4.2 Shape transfer (train on square cells, test on wide cells; this includes the target-like cells)

| model | within ρ | C within | log-RMSE | bias | TARGET cells: within ρ / C / log-RMSE |
|---|---|---|---|---|---|
| current label | 0.363 | 0.571 | 1.218 | −0.21 | 0.514 / 0.580 / 1.365 |
| A4 free20k | 0.812 | 0.834 | 0.638 | +0.23 | 0.796 / 0.852 / 0.771 |
| sampling (unit calibration) | 0.732 | 0.775 | 0.824 | +0.11 | 0.652 / 0.756 / 0.992 |
| static + lookahead (zero conflicts) | 0.689 | 0.782 | **1.726** | **−1.46** | 0.720 / 0.826 / 1.580 |
| **greedy free** | **0.823** | **0.843** | **0.614** | −0.13 | **0.807 / 0.863 / 0.720** |
| greedy free + sampling | 0.869 | 0.865 | 0.541 | +0.08 | 0.844 / 0.876 / 0.680 |
| greedy free + binary | 0.873 | 0.869 | 0.504 | −0.08 | 0.863 / 0.887 / 0.613 |
| greedy all | 0.888 | 0.876 | 0.494 | +0.03 | 0.859 / 0.883 / 0.652 |
| GBM free + binary | 0.865 | 0.860 | 0.675 | −0.30 | 0.851 / 0.869 / 0.839 |

**Findings on transfer:**
- The linear models transfer well.
- The tree models lose more when trained on square cells only. GBM free drops from 0.858 under LOSO to 0.782 here, so a nonlinear model is not worth its opacity.
- The zero-conflict tier ranks acceptably but its **level does not transfer**: square→wide bias is −1.46 nats. Do not use it for magnitudes.

The reverse direction (wide→square) is in `eval_results.txt`. The winner's procedure gets within 0.806, C 0.828, log-RMSE 0.479.

### 4.3 Reward-gain error

Survivors universe, ceiling 2M. Each cell is gain MAE / tail MAE, with the number of masks in brackets:

| model | DGH (3) | Farkas (28) | rules (75) | random 5 % (143) | random 20 % (148) | random 50 % (150) |
|---|---|---|---|---|---|---|
| current label | 0.067 / 0.190 | 0.053 / 0.026 | 0.064 / 0.215 | 0.013 | 0.025 | 0.033 |
| A4 free20k (+smear) | 0.006 / 0 | 0.016 / 0.028 | 0.030 / 0.071 | 0.009 | 0.016 | 0.021 |
| **winner's procedure (+smear)** | 0.006 / 0 | **0.011 / 0.013** | **0.024 / 0.060** | 0.007 | 0.014 | 0.019 |
| greedy free + binary (+smear) | 0.008 / 0 | 0.012 / 0.015 | 0.032 / 0.072 | 0.008 | 0.016 | 0.020 |
| greedy all (+smear) | 0.008 / 0 | 0.013 / 0.019 | 0.035 / 0.073 | 0.008 | 0.015 | 0.020 |
| sampling alone (+smear) | 0.007 / 0 | 0.033 / 0.069 | 0.059 / 0.197 | 0.014 | 0.024 | 0.029 |

**Bootstrap over the 15 tables** (2,000 resamples) of the difference in real-mask gain MAE against the winner's procedure, with 95 % CIs:

| model | difference in gain MAE |
|---|---|
| current label | +0.027 to +0.057 |
| A4 free20k | +0.002 to +0.009 |
| free + binary | +0.001 to +0.012 |
| all | +0.002 to +0.015 |
| free + sampling | +0.002 to +0.012 |
| GBM free + binary | −0.004 to +0.004 (tied on gain; its tail MAE is worse, 0.063 vs 0.046) |

**Ceiling.** The ceiling applied to censored labels matters, shown for the winner's procedure (gain MAE / tail MAE):

| ceiling | gain MAE / tail MAE |
|---|---|
| 400k (today's 20·cap) | 0.031 / 0.045 |
| **2M (100·cap)** | **0.020 / 0.046** |
| none | 0.029 / 0.058 |

Use a ceiling of 2M.

**All-cases universe (library ignored; argD becomes live with 15 masks).**
- Current label: argD gain error 0.166, all real masks 0.095.
- Winner's procedure: argD 0.037, all real masks 0.020.
- GBM free + binary is best here: 0.016.

**DGH detail** (the user's example), gain_I / tail_I on library survivors:

| table | truth | current label | winner | sampling |
|---|---|---|---|---|
| (9,18,86)p, 47 kills | **0.110 / 0.000** | 0.269 / 0.499 | 0.101 / 0 | 0.119 / 0 |
| (9,16,78)p, 90 kills | **0.031 / 0.000** | 0.074 / 0.071 | 0.023 / 0 | 0.019 / 0 |
| (11,21,117)p | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |

In (11,21,117)p DGH kills all 9 survivors, so every estimator gives 1 / 1.

The current label inflates DGH because it assigns 400k to every censored case, and DGH's kills in these cells are the censored ones with the smallest true d. The top decile under a constant label is an arbitrary tie-break, which here happens to include DGH kills.

### 4.4 Contribution of each technique (LOSO, greedy nested selection)

| pool | within ρ | C within | log-RMSE | gain / tail MAE | change vs "all" |
|---|---|---|---|---|---|
| all (BASE + ST + LA + PS + SA + BIN) | 0.888 | 0.872 | 0.451 | 0.028 / 0.057 | – |
| all − lookahead | 0.888 | 0.872 | 0.451 | 0.028 / 0.057 | **0.000** (never selected) |
| all − static/LP | 0.888 | 0.872 | 0.450 | 0.028 / 0.057 | **0.000** |
| all − sampling | 0.874 | 0.865 | 0.485 | 0.026 / 0.055 | −0.014 ρ |
| all − pysat probe | 0.873 | 0.863 | 0.444 | 0.026 / 0.068 | −0.015 ρ |
| all − binary | 0.853 | 0.850 | 0.493 | 0.027 / 0.067 | −0.035 ρ |
| free tier (ST + LA + PS) | 0.837 | 0.847 | 0.510 | **0.020 / 0.046** | −0.051 ρ, but the lowest gain error |
| free tier − lookahead = A4 free20k (PS + ST) | 0.802 | 0.826 | 0.547 | 0.026 / 0.058 | lookahead adds +0.035 ρ here |

**Features the greedy procedure selected.**
- In the free tier, 17/17 folds chose `ps20k_decs_per_conf`, `ps20k_restarts_per_kconf` and `st_distinct_rows`. `knfl_rand_mean_log_nodes` was chosen in 14/17 and `la1_failed_neg` in 11/17.
- In "all", the sampling picks are `mean_live_work` and `n_tail` (the censored-cube tail count), not the calibrated mu.
- Lookahead is only selected in the free tier: `knfl` Knuth tree size and failed literals. That matches A2's finding that failed literals, not implied cells, carry the signal. The pilot's `la1_mean` never appears.
- LP slack (`st_argD_*`, `st_row_lp_eps_rel`) ranks weakly (single-best ρ 0.38) and survives only as a small correction.

### 4.5 Negative result: spending the conflicts on continued solving beats every estimator at equal cost

Continue the fresh solve to C conflicts, then rank only the cases still open by the winner's procedure:

| policy | extra conflicts / hard case | within ρ | C within | gain / tail MAE |
|---|---|---|---|---|
| direct 50k + greedy free | ≤ 30k (a 50k run) | **0.932** | **0.934** | 0.017 / 0.044 |
| direct 100k + greedy free | ≤ 80k | 0.953 | 0.956 | 0.014 / 0.040 |
| direct 200k + greedy free | ≤ 180k | (not computed) | (not computed) | 0.012 / 0.031 |
| greedy free + sampling (compare) | ~46k | 0.853 | 0.850 | 0.027 / 0.067 |
| greedy all (compare) | ~70k | 0.888 | 0.872 | 0.028 / 0.057 |

A 50k direct run costs less than the sampling estimator and beats it on every metric. It also gives exact labels for part of the tail:
- 44 % of the exact hard cases have d ≤ 50k and 63 % have d ≤ 100k. Measured over all hard cases (7.9 % of which are open at 2M), the shares are 40 % and 59 %.
- On the target cells the benefit is smaller (within 0.850 vs 0.835): only 10 % of their hard cases finish within 50k, and 18 % within 100k.

Sampling's value is in the far tail, as A3 also found, and its conflicts are better spent on the direct solve.

### 4.6 MID regime (2k < d ≤ 20k, 2,704 cases, 2k-tier features only, LOSO)

| model | within ρ | pooled ρ | log-RMSE |
|---|---|---|---|
| current label, clip(fhat, 2k, 40k) | 0.173 | 0.255 | 1.376 |
| fhat refit | 0.223 | 0.270 | 0.605 |
| static | 0.510 | 0.504 | 0.553 |
| lookahead | 0.587 | 0.642 | 0.492 |
| static + lookahead (zero conflicts) | 0.590 | 0.647 | 0.490 |
| CDCL 2k (pysat + binary) | 0.624 | 0.659 | 0.473 |
| all 2k tiers, no sampling | 0.646 | 0.670 | 0.472 |
| all 2k tiers + sampling (~20k conflicts) | **0.714** | **0.756** | **0.419** |

The target pipeline labels MID cases exactly, since they finish within 20k, so this regime does not affect the reward. It only matters for a pipeline with a 2k cap.

## 5. Selection of the winner

**The rule.** Pick the best held-out hard-regime concordance and Spearman together with the best gain error, subject to a cost of ≤ ~5 s and ≤ ~100k conflicts per case.

**How the candidates split.**
- The best ranker within the cost cap is `greedy[all]`: ρ 0.888, C 0.872, ~92k conflicts, ~4.9 s mean. GBM all is 0.899 / 0.878. `greedy[free+BIN]` is 0.874 / 0.865 at ~46k conflicts.
- Their gain errors are significantly worse than the free-tier greedy model's: 0.026-0.028 vs 0.020.
- Their tail errors are also worse: 0.055-0.057 vs 0.046.
- Gain error is the metric the reward sees, and the rule names it.

**The choice.** `greedy[free(ST+LA+PS)]` is best on gain and tail error and cheapest. It is within 0.05 ρ / 0.025 C of the best ranker, and 2.6× better on ρ than the current label.

**The final fit.**
- `eval_final.py` reran the same greedy procedure once on all 2,238 exact hard cases. It picked the 6 features in section 1, the same set the folds chose most often.
- The model was fitted with `hardness_model.fit` and saved to `hardness_model.json`.
- In-sample: within ρ 0.849, RMSE 0.471. The expected held-out performance is the nested LOSO row above.
- **End-to-end check.** `hardness_model.predict()` recomputed every feature from scratch on 6 seeded cases (`features_final_check.jsonl`). It reproduced the cached-feature prediction exactly on all 6. `predict_from_probe()` (the zero-extra-conflict path) matched too.

**Runner-up to use if ranking matters more than sums.** The same procedure on the free+BIN pool: ρ 0.874, C 0.865, ~46k conflicts. It is not saved; rerun `eval_final.py` with `TIER = ("BASE", "ST", "LA", "PS", "BIN")`.

## 6. Caveats

- **Truth for open cases.** The 191 cases open at 2M have true d ≥ 2M. They are scored only by C, and in the gain simulation they are set to 2M, a lower bound. On (16,17) 77 of 150 sampled cases are open, so its true gain_I is itself uncertain.
- **Sampled target tables.** In (12,18), (13,19) and (16,17), the censored mass is carried by 150 sampled cases with weights of 13-26. Gain errors there are noisier than on complete tables.
- **Too few DGH masks.** Only 3 DGH masks are non-empty on library survivors, and argD kills 0 survivors on every table because the library contains argD. The DGH conclusion rests on 2 informative tables. The Farkas (28) and rule (75) masks carry most of the real-mask average.
- **Contention.** Seconds were measured with 10 of my processes plus other agents running (load 14). Conflicts and propagations are the reliable cost numbers.
- **Coverage of cell types.** All labels are s = t = 3 and m ≤ n. Tall cells and other (s, t) extrapolate the level.
- **Solver versions.** The binary features come from CaDiCaL 3.0.1, not the label solver. The winner does not use them.
- **The frozen module d_hats** (the "as shipped" rows) were fitted on some of these cells. They are shown for reference only and never used for selection.

## 7. TODOs for the integrator (shared files not edited)

1. **`zar_ub/solve.py`:** add `restarts=int(st.get("restarts", 0))` to `SolveResult`.
2. **`zar_ub/difficulty.py::label_case` / `continue_label`:**
   - Store the fresh 20k pysat run's `decisions` and `restarts` in the probe dict.
   - For a case censored at 20k, set `d = hardness_model.predict_from_probe(inst, rows, cols, conflicts, decisions, restarts, propagations)` (mean=True, floor 20k, ceiling 2M).
   - This replaces `censored_d(cap, fhat)`, and `CENSOR_CLIP=20` should become 100.
   - The probe must be a fresh `cadical195` run with `conf_budget(20000)`, as in `hardness_progress.pysat_run`. A continued solver's statistics are not the same measurement.
3. **Existing target tables** have no stored decisions or restarts. Relabel their 12,246 censored cases, or only the 6,663 library survivors, with `hardness_model.predict()`: ~0.30 G conflicts, ~0.7 h wall on 12 workers (0.4 h for survivors). Writing back to `cache/case_table_*.json` is your decision.
4. **Recommended (section 4.5):** before estimating, deepen library survivors directly to 50k-100k conflicts (`casetable.deepen`). That is ~0.2-0.5 G conflicts for the target survivors. Apply the model only to the cases still open. This beats every estimator at equal cost.
5. **`hardness_model` expects `hardness_lookahead.estimate`** with its default probes (the `knfl_*` features). `hardness_lookahead`'s level-0 unit-propagation fix (A2's TODO for `difficulty.propagation_fraction`) is unrelated and still open.
6. **Reward (E25):** accurate difficulty does not make narrow wide-cell rules such as DGH look better. It removes the false tail credit they get today (section 4.3). If the thesis wants to reward such rules, that is a reward-design change, not a labelling change.
7. **Refit** `hardness_model.json` by rerunning `eval_collect.py`, then `eval_analyze.py`, then `eval_final.py`, whenever new hard labels land (A1's optional 20M pass on the 191 open cases, or new cells).

## 8. Reproduce

```bash
cd examples/zarankiewicz/upper_bounds; export ZAR_UB_NO_LLM=1; PY=/Users/jaybhan/Downloads/openevolve/.venv/bin/python
$PY experiments/E24_difficulty/eval_collect.py --procs 10   # 15,387 estimator runs, 38 min wall on 10 processes (resumable)
$PY experiments/E24_difficulty/eval_masks.py                # gain tables and masks, 13 s
$PY experiments/E24_difficulty/eval_analyze.py              # all models, CV and gain simulation, ~12 min
$PY experiments/E24_difficulty/eval_report.py               # eval_results.txt
$PY experiments/E24_difficulty/eval_final.py                # fit and save hardness_model.json, end-to-end check
```

**Total solver spend for this evaluation** (5,133 cases × 3 estimators):
- conflicts: sampling 0.167 G, progress 0.162 G, lookahead UP 0.011 G (0.34 G in total);
- propagations: sampling 66.5 G, progress 39.0 G, lookahead 61.5 G trail literals;
- solver-seconds: 23,018 (sampling 14,623, progress 4,527, lookahead 3,868).

scikit-learn 1.9.1 was installed into the venv.

# E24 / A3: Monte Carlo decomposition-hardness (Chivilikhin-Pavlenko-Semenov) difficulty estimator

Owner A3 (sampling). Date 2026-09-22. Files:
- `zar_ub/hardness_sampling.py`: the module, which follows the estimator contract.
- `experiments/E24_difficulty/sampling_{run,eval,final,designs,variants,combine,hybrid,retest,verify,a4feats}.py`: the scripts.
- `experiments/E24_difficulty/sampling_data/`: raw cube records, evaluations and logs.

Resources: at most 3 solver processes. There were no LLM or network calls and no Lean build. No case table was modified. Every solver run is a fresh pysat `cadical195` (deterministic) and every RNG is seeded. Costs are given in solver **conflicts** and **propagations**. Seconds are also reported, but they are inflated by other agents' solvers running at the same time.

## 1. Summary

**Question.** Can the proposal's Chivilikhin et al. (2023) d-hardness estimator rank and size the hard cases better than the current censored label? These are the cases that are censored at 20k conflicts on the target tables. The estimator draws N random cubes over a decomposition set B, solves each under a per-cube budget, and estimates 2^k times the mean cube work.

**Answer.**
- **Yes, strongly, against the current label.**
  - On the three real open target tables, (12,18,109), (13,19,123) and (16,17,134), I used the deepened random sample: 283 exact hard cases plus 167 that are still open at 2M conflicts.
  - The estimator reaches within-cell Spearman **0.68** and Harrell's C **0.79** at a hard cap of about 50k conflicts per case. At about 210k conflicts it reaches 0.73 / 0.83.
  - The current label gets C = 0.52 (its within-cell ranking is undefined because the label is constant). The `fhat` form gets 0.24 / 0.67.
- **It is not the best estimator on its own.**
  - A4's `free20k` model (PROGRESS.md) is stronger and costs nothing extra: 0.79 within-cell and C 0.885 on the same targets.
  - Sampling is **complementary** to `free20k`. Together they give **0.82-0.88 within-cell** everywhere:

    | evaluation | free20k alone | free20k + sampling |
    |---|---|---|
    | DEV LOCO | 0.801 | 0.847 |
    | frozen WIDE | 0.855 | 0.882 |
    | frozen TARGET | 0.793 | 0.820 (C 0.893) |
    | LOCO over all 12 cells | 0.805 | 0.848 |
- **Sampling only pays off where the equal-cost alternative fails.** The alternative is to spend the same conflicts continuing the direct solve.
  - On the target tables, 94% of the cases are still open at 50k conflicts, and sampling beats the direct run (within 0.68 vs 0.45).
  - On the wide pure cells, only 73% are open at 50k, and the direct run is as good or better.
  - On the square DEV cells and in the MID regime, solving is cheaper than estimating.
- **Two engineering findings matter more than the design of B.**
  1. **Uniform cubes do not work on this encoding.** The double-lex symmetry breaking refutes 90-100% of uniform cubes by unit propagation whenever B is structured (heaviest row, lookahead-top cells, or whole-row supports). The estimate then rests on a handful of surviving cubes. The fix is to sample **Knuth probes of the UP-pruned decision tree**: a node branches only on variables that UP leaves free, and each probe carries weight 2^(free branchings). This estimates the same kind of total, over an adaptive decomposition. It raises hard-regime Spearman from about 0.4-0.55 to 0.78 at equal cost on the E10 screen.
  2. **Sampling noise is the bottleneck.** The seed-to-seed Spearman is only 0.70 at N = 100. **Stratifying** the probes below the exactly enumerated depth-5 nodes (16-32 strata) cuts the spread of log mu~ on DEV from 1.08 to 0.76, and raises DEV LOCO within-cell Spearman from 0.70 to 0.74 at a lower cost.
     - **This did not transfer (negative result).** On the harder held-out cells the stratified budgeted point was worse on WIDE (0.652 vs 0.749) and equal on TARGET (0.681 vs 0.678).
     - Its 2000-conflict cube budget is hit by 38% of the TARGET cubes.
     - Within 50k conflicts it runs only about one round, which is one probe per stratum.
     - At equal N and b, stratified and plain Knuth are the same off DEV: N100 b5000 gives WIDE 0.797 vs 0.796 and TARGET 0.732 vs 0.729.

**Protocol for the headline.**
- All models were frozen on the 262 DEV hard cases, and every operating point is capped at about 50k conflicts per case (`total_budget`).
- Two candidates were each selected on DEV only:
  - `knuth:row:8, T=50k, b=5000`: the best budgeted knuth point, chosen before the stratified sampler existed.
  - `strat:row:8:5, T=50k, b=2000`: the overall DEV pick.
- **The module default is the knuth point.** The choice between these two DEV-selected candidates was made *after* seeing the held-out results below, so the knuth held-out numbers are optimistic by that one bit of selection. The strat pick is reported alongside at full honesty, and both calibrations are frozen in the module.

| estimator (frozen on DEV) | extra conflicts / case: WIDE ; TARGET | WIDE within / pooled / log-RMSE | TARGET within / pooled / log-RMSE | TARGET Harrell C within / pooled |
|---|---|---|---|---|
| current label clip(fhat, 20k, 400k) | 0 | – / 0.331 / 1.30 | – / 0.228 / 1.22 | 0.517 / 0.521 |
| `fhat` form refit (log c2000, log2_volume) | 0 | 0.422 / 0.339 / 1.24 | 0.240 / 0.192 / 1.27 | 0.674 / 0.645 |
| direct fresh run, cap 50k: min(d, 50k) | 45.2k ; 49.1k | 0.767 / 0.779 / 1.52 | 0.453 / 0.492 / 2.02 | 0.560 / 0.563 |
| **sampling, knuth:row:8, T50k, b5000 (module default)** | 49.5k ; 51.7k | 0.749 / 0.762 / 0.91 | 0.678 / 0.691 / 0.93 | 0.787 / 0.794 |
| sampling, strat:row:8:5, T50k, b2000 (the DEV-only pick) | 46.1k ; 50.6k | 0.652 / 0.611 / 0.99 | 0.681 / 0.695 / 0.85 | 0.793 / 0.796 |
| A4 free20k (refit on DEV) | 0 (the table's 20k probe) | 0.855 / 0.849 / 0.76 | 0.793 / 0.788 / 0.72 | 0.885 / 0.885 |
| free20k + sampling (strat) | 46.1k ; 50.6k | 0.870 / 0.870 / 0.68 | 0.826 / 0.829 / 0.66 | 0.895 / 0.895 |
| **free20k + sampling (knuth default)** | 49.5k ; 51.7k | 0.882 / 0.879 / 0.67 | 0.820 / 0.822 / 0.66 | 0.893 / 0.894 |
| direct 50k, then open cases ordered by free20k | 45.2k ; 49.1k | 0.886 / 0.907 / – | 0.800 / 0.796 / – | 0.889 / 0.888 |
| direct 50k, then open cases by free20k + sampling | 82.6k ; 98.1k | 0.903 / 0.921 / – | 0.825 / 0.827 / – | 0.896 / 0.897 |

DEV LOCO within-cell Spearman at the same operating points: current fhat 0.179; knuth 0.695; strat 0.743; free20k 0.801; free20k+strat 0.856; direct 50k 0.983 (DEV hard cases are mostly < 100k).
Within-cell of the current label on WIDE/TARGET is undefined: it is clipped to the constant 400k in every held-out cell.
Direct runs and hybrids are ranking constructs: their log-RMSE is that of min(d, C), or undefined.

## 2. Methods

### 2.1 Estimator (`zar_ub/hardness_sampling.py`)

**Case formula.** For a case q, the formula is C = `encode_case(inst, rows, cols)`. The work unit is CaDiCaL conflicts, the same unit as the label `d`. Propagations are reported alongside.

**Per-cube work.**
- Each cube is first checked with an incremental UP-only solver, using `propagate(assumptions)`. This is the paper's §7 trick.
- A cube refuted by unit propagation costs 0 conflicts.
- Otherwise a **fresh** solver runs `solve_limited(assumptions=cube)` with the per-cube conflict budget b. A fresh solver keeps xi a function of the cube, which Theorem 2 requires.

**Estimator.** mu~ = |Omega_B| / N · Σ xi^j (paper eq. 8), in three forms:
- `uniform`: the paper's estimator, with i.i.d. uniform cubes over B and |Omega| = 2^k. For the `support` design, |Omega| = Π C(n, r_i).
- `knuth`: the new sampler used here. A probe walks the cell order and skips every variable that UP has fixed. For each free variable it tests both values with `propagate`:
  - two viable values: pick one uniformly and double the weight;
  - one viable value: take it, weight unchanged;
  - no viable value: the probe dies, which is a UP-refuted leaf with work 0.

  After k two-way branchings the cube is a leaf and is solved. Then E[w · xi(leaf)] = Σ_leaves xi. This is the d-hardness of the adaptive decomposition defined by the UP-pruned decision tree (Knuth 1975). The identity is exact, like Theorem 2.
- `strat`: the depth-`strata_depth` nodes of the same tree are enumerated exactly, capped at 64. Probes are drawn round-robin below them, and the estimate is Σ_s mean_s(w · xi). It is unbiased when every stratum has at least one probe, so the budgeted mode uses complete rounds only.

**Censoring.** A cube that hits b has xi ≥ b. I use mu_cens, which replaces each censored cube by b · alpha / (alpha − 1). Here alpha is the censored Hill/Pareto tail index of the cube works above b/8, clipped to [1.1, 8]. mu_lb, which uses min(xi, b), is kept as a feature. The tail term matters (§6).

**Sample-size report (Theorem 3 / eq. 13, with sample moments).**
- `N_req(eps=0.2, delta=0.1) = s² / (eps² · delta · xbar²)`
- `eps_achieved(delta=0.1) = sqrt(s² / (N · delta · xbar²))`

For stratified runs, s² is the stratified variance times N.

**Designs for B / the cell order.**
- `row`: the cells of the heaviest rows, row-major. This is design (i).
- `random`: seeded random cells. This is design (ii).
- `lookahead`: cells ranked by the paper's UP weight w⁺ + w⁻, where a failed literal counts as nvars. This is design (iii).
- `support`: whole-row cubes with the prescribed row sum, over the k heaviest rows. This is design (iv), with |Omega| = Π C(n, r_i).

**Returned fields.**
- `d_hat`: mu~ calibrated with the frozen DEV fit (§2.4).
- `d_hat_raw`: mu~ itself.
- `features`: mean, median and max cube work, mean live-cube work, the fraction refuted by UP, the fraction that hit the budget, the fraction SAT, the coefficient of variation, the Pareto alpha, log2 mu_lb, log2 mu_cens, log2 of the propagation-work estimate, the Knuth estimate of the number of live leaves, N_req, eps_achieved and n_cubes.
- Cost: `cost_conflicts`, `cost_propagations` (including the probing propagations) and `cost_seconds`.

`total_budget` stops drawing cubes once the conflicts spent reach it. This gives a hard per-case cap.

### 2.2 Data and protocol (no tuning on held-out cells)

**Case sets.**
- **E10 screen.** The design comparison used the 7 TRAIN cells from `E10_difficulty/results.json`: 104 cases (60 hard + 44 mid). The table uses the 47 cases that every configuration ran (30 hard + 17 mid).
- **DEV.** The exact labels in `ground_truth_initial.jsonl`:
  - HARD (d > 20k): all 262 cases, in 4 square cells: (11,11,70), (11,12,75), (12,12,81) and (12,13,87).
  - MID (2k < d ≤ 20k): up to 100 per cell, seeded, for 1,965 cases in 29 cells.
- **WIDE test.** 509 exact hard cases in 5 wide cells never used for fitting: (9,18,86), (10,19,99), (10,20,103), (11,21,117) and (9,23,104). This is the same case set as A4's `progress_features_test*.jsonl`, so A4's features are available for these cases.
- **TARGET test.** A1's deepened uniform sample of the three open target tables (`ground_truth.jsonl`, `source = deepen`): (12,18,109) 115 + 35 open, (13,19,123) 95 + 55 open, (16,17,134) 73 + 77 open. The "open" cases are still unresolved at 2,000,000 conflicts and are right-censored. A4's free20k inputs were computed for these cases with A4's public `hardness_progress.estimate(tier="20k", binary=False)` (`sampling_a4feats.py`, one fresh 20k run per case, 9.0M conflicts in total).

**Storage and derivation.**
- Every (case, config) was run once at the largest N (100) and b (10,000; 5,000 for the TARGET runs), and every cube's outcome was stored.
- A smaller N is a prefix of the i.i.d. sample.
- A smaller b is exact re-censoring, because a fresh CaDiCaL run is deterministic and the conflict limit only stops it.
- The budgeted mode takes cubes in order until the conflicts reach T.

**Selection.** The operating point was selected on DEV only: the best DEV LOCO within-cell Spearman with DEV mean cost ≤ 50k.

**Models.** Every model has the form log d = a + Σ b_k z_k. It is either fitted leave-one-cell-out on DEV, or fitted on all 262 DEV hard cases and frozen for WIDE and TARGET. Censored TARGET rows are never used for fitting.

**Metrics.**
- Mean within-cell Spearman (cells with at least 8 exact cases).
- Pooled Spearman and log-RMSE (natural log) on exact cases.
- On TARGET, Harrell's C including the 2M-censored cases. It is the only metric that sees the hardest half of (16,17,134).

**Baselines on the same cases.**
- The current label, `clip(fhat, 20k, 400k)` (calibrate_33.json).
- The `fhat` form refit.
- A4's `free20k`: 4 features, refit on the same DEV cases.
- A **direct fresh run at cap C**. Its outcome is min(d, C), read off the exact label.
- **Hybrids**: a direct run at C, with the cases still open ordered by a model.

### 2.3 Verification (`sampling_verify.py`, `sampling_data/verify_*.json`)

I reran `estimate()` on 5 seeded DEV hard cases per sampler (knuth:row:8 and strat:row:8:5).
- **Every cube reproduced exactly**: conflicts, propagations, censoring, weight and UP-refutation.
- The module's budgeted default reproduces the offline derivation (`derive_budgeted`) exactly: mu~, cost and number of cubes. (knuth: 5/5 cases; strat: 5/5 cases, where the default estimate uses complete round-robin rounds of the 100 stored probes.)
- The estimate is therefore a pure function of (inst, rows, cols, kw).

### 2.4 Calibration frozen in the module

log d = log mu~ + a, with a = median log(d / mu~) on DEV. The value is a = −1.860 (strat) or −1.855 (knuth), so d ≈ 0.156 mu~.

Summing independent cube solves costs about 6.4 times the monolithic solve, because the cubes share no learned clauses.

I chose unit slope because on DEV the estimator noise is larger than the spread of the truth. For knuth, sd(log mu~) = 1.08 while sd(log d) = 0.56 (for strat, sd(log mu~) = 0.76). The OLS slope is therefore attenuated by errors-in-variables: 0.35 for knuth and 0.55 for strat.

**Disclosure.** I also computed both calibrations on WIDE. The log-RMSE there is 0.91 with unit slope and 1.17 with OLS (knuth, T50k). The DEV-only argument above predicts this direction, but the choice was made after seeing it. The ranking metrics are unaffected. The OLS numbers are reported as `samp`.

## 3. Design screen: uniform cubes fail on this encoding (E10, 30 hard + 17 mid common cases)

Raw (uncalibrated) Spearman of mu~ with the true d. The rows shown are the best (N, b) per configuration under the 50k cost cap, plus representative failures. The full grid is in `sampling_data/designs.json`, `screen_e10_eval.json` and `sampling_designs.py`.

| sampler:design:k | N | b | hard ρ | mid ρ | conflicts/case (hard) | UP-refuted cubes | budget hit | eps (δ=0.1) |
|---|---|---|---|---|---|---|---|---|
| **knuth:row:8** | 100 | 2000 | **0.785** | 0.797 | 33,160 | 0.38 | 0.06 | 0.57 |
| knuth:row:4 | 20 | 10000 | 0.751 | 0.833 | 65,864 | 0.08 | 0.15 | 0.77 |
| knuth:row:10 | 100 | 2000 | 0.728 | 0.475 | 14,363 | 0.56 | 0.02 | 0.73 |
| knuth:row:12 | 100 | 2000 | 0.503 | 0.554 | 6,584 | 0.70 | 0.00 | 1.03 |
| knuth:lookahead:8 | 100 | 2000 | 0.727 | 0.831 | 53,712 | 0.01 | 0.07 | 0.34 |
| knuth:lookahead:12 | 50 | 2000 | 0.710 | 0.691 | 11,386 | 0.08 | 0.01 | 0.65 |
| knuth:random:10 | 20 | 10000 | 0.725 | 0.843 | 47,240 | 0.00 | 0.03 | 0.57 |
| uniform:random:10 | 20 | 10000 | 0.653 | 0.806 | 51,647 | 0.22 | 0.07 | 0.77 |
| uniform:random:12 | 20 | 10000 | 0.567 | 0.865 | 28,729 | 0.32 | 0.03 | 0.93 |
| uniform:row:6 | 50 | 10000 | 0.557 | 0.591 | 28,130 | **0.90** | 0.03 | 1.51 |
| uniform:row:8 | 100 | 10000 | 0.389 | 0.592 | 22,162 | **0.96** | 0.01 | 1.84 |
| uniform:row:12 | 100 | 10000 | −0.057 | −0.127 | 464 | **1.00** | 0.00 | – |
| uniform:support:1 (C(n,r) row patterns) | 100 | 10000 | 0.408 | 0.564 | 13,313 | **0.97** | 0.01 | 2.17 |
| uniform:support:2 | 100 | 10000 | −0.030 | −0.303 | 18 | **1.00** | 0.00 | – |
| uniform:lookahead:6 | 100 | 10000 | 0.263 | 0.083 | 16,671 | **0.98** | 0.01 | 1.81 |
| uniform:lookahead:10 | 20 | 2000 | 0.396 | 0.216 | 201 | **0.99** | 0.01 | – |

Reading:
- **Uniform cubes fail whenever B is structured.** The double-lex symmetry breaking plus the row/column-sum constraints refute 90-100% of the cubes of the heaviest-row cells, the lookahead-top cells and the whole-row supports by UP alone.
  - The paper's UP weight is largest exactly on the lex-constrained cells. So design (iii), which is the paper's own B₀ heuristic, is the *worst* uniform design.
  - The structured "support" cubes answer the open question in `docs/lit/chivilikhin2023_decomp.md` §9.1. Conditioning on the row sums does not help, because the lex order is what kills the cubes: 97% UP-refuted with one row and 100% with two.
- Uniform random cells survive better (68-78% of cubes live), but they need about 2× the cost for a lower ρ.
- The Knuth sampler turns the same budget into signal for every design. The row order is the best and cheapest.
- **For k:**
  - Too few branchings (k = 4, 6): each leaf is nearly the whole problem, the budget is hit and the cost is high.
  - Too many (k = 10, 12): many probes die, the per-leaf work is small and noisy, and MID ranking suffers.
  - k = 8 is the best trade-off. On DEV, knuth:row:8 and knuth:row:10 were both run on all 2,227 cases, and row:8 is ahead at every cost (`dev_eval.json`).

## 4. DEV results (4 square cells, leave-one-cell-out)

### 4.1 HARD regime (262 cases), accuracy vs cost

The calibration is log-linear and fitted LOCO; the within-cell Spearman is unchanged by any monotone calibration. Costs are in conflicts and propagations per case, with approximate seconds under contention.

**Baselines on the same 262 cases:**

| baseline | cost | within-cell ρ | pooled ρ | log-RMSE |
|---|---|---|---|---|
| `fhat` fixed | 0 | 0.179 | 0.249 | 1.128 |
| `fhat` refit, LOCO | 0 | 0.179 | 0.083 | 0.566 |
| constant, LOCO | 0 | – | – | 0.583 |
| A4 free20k, LOCO | 0 extra (the table's 20k probe) | 0.801 | 0.800 | 0.331 |
| direct run at 50k (min(d, 50k)) | 37.7k | 0.983 | 0.962 | 0.215 |

**Sampling operating points:**

| operating point | conflicts | props | s | within ρ | pooled ρ | log-RMSE |
|---|---|---|---|---|---|---|
| knuth:row:8, N20, b2000 | 6.4k | 2.4M | 0.3 | 0.350 | 0.300 | 0.55 |
| knuth:row:8, N50, b2000 | 16.5k | 6.3M | 0.8 | 0.592 | 0.556 | 0.48 |
| knuth:row:8, N100, b2000 | 33.0k | 12.6M | 1.7 | 0.672 | 0.637 | 0.45 |
| knuth:row:8, **N100, b5000** (best knuth at a fixed N) | 45.0k | 15.9M | 2.1 | **0.700** | 0.670 | 0.43 |
| **knuth:row:8, T50k, b5000 (best budgeted knuth; module default)** | 39.3k | 14.0M | 1.9 | 0.695 | 0.659 | 0.43 |
| knuth:row:8, N100, b10000 | 51.4k | 17.5M | 2.3 | 0.678 | 0.685 | 0.42 |
| strat:row:8:5, N50, b2000 | 19.7k | – | 0.6 | 0.652 | 0.739 | 0.41 |
| strat:row:8:5, N100, b2000 | 37.2k | – | 1.2 | 0.746 | 0.778 | 0.39 |
| **strat:row:8:5, T50k, b2000 (overall DEV-only pick)** | **36.3k** | 12.0M | 1.0 | **0.743** | **0.771** | **0.39** |
| strat:row:8:3, N100, b2000 | 33.8k | – | 1.2 | 0.740 | 0.731 | 0.41 |
| free20k + samp (strat T50k b2000), LOCO | 36.3k extra | | | **0.865** | 0.859 | 0.298 |

On DEV the hard cases are "barely hard": d ranges from 20k to 222k, and only 10% are above 100k. A direct run to 50k conflicts therefore resolves 67% of them exactly, and it wins outright at equal cost (0.983). The value of sampling lies in the far tail, as the TARGET results show (§5).

### 4.2 MID regime (1,965 cases, 29 cells, LOCO, knuth:row:8)

| estimator | conflicts/case | within ρ | pooled ρ | log-RMSE |
|---|---|---|---|---|
| `fhat` fixed / refit LOCO | 0 | 0.197 / 0.197 | 0.299 / 0.287 | 2.924 / 0.602 |
| samp N50, b500 | 6.4k | 0.599 | 0.636 | 0.484 |
| samp N100, b2000 | 18.7k | 0.646 | 0.660 | 0.483 |
| samp + vol, N100, b2000 | 18.7k | 0.645 | 0.670 | 0.478 |
| direct run at 5k (min(d, 5k)) | 4.4k | 0.855 | 0.678 | 0.314 |
| direct run at 10k | 6.7k | 0.984 | 0.970 | 0.110 |

**Negative result.** In MID, sampling costs more than solving: the mean d is 7.9k and the median 6.6k. The table pipeline already solves these cases exactly at its 20k cap, so there is no reason to estimate them. For MID *without* solving, A4's 2k tier is the right tool (PROGRESS.md §4.5).

### 4.3 Sampling noise (test-retest, `retest_knuth_row_8.json`; seed 0 vs seed 1, knuth:row:8, DEV hard)

| N, b | retest ρ (seed 0 vs 1) | noise sd(log mu~) | ρ with d: seed 0 / seed 1 / average of both | within-cell: seed 0 / average |
|---|---|---|---|---|
| 20, 2000 | 0.324 | 1.05 | 0.425 / 0.567 / 0.618 | 0.350 / 0.593 |
| 50, 2000 | 0.517 | 0.65 | 0.621 / 0.662 / 0.735 | 0.592 / 0.711 |
| 100, 2000 | 0.698 | 0.44 | 0.682 / 0.739 / 0.769 | 0.672 / 0.745 |
| 100, 10000 | 0.726 | 0.51 | 0.701 / 0.777 / 0.792 | 0.678 / 0.749 |

The spread of log d on DEV hard is sd = 0.56, which is similar to the estimator noise even at N = 100. The accuracy is therefore noise-limited, and averaging two seeds (N = 200) gains +0.07 within-cell. Stratification is a cheaper way to get most of that gain.

**Sample size.** The paper's (eps = 0.2, delta = 0.1) guarantee is far away. The median N_req is:
- for knuth: 1,156 on DEV at T50k b5000 (eps achieved 0.72), about 690 on WIDE (b = 5000), and about 340 on TARGET;
- strat (N100, b2000) on DEV: 338 at the budgeted point; on TARGET 72 (N100 b2000). The budgeted strat point often runs one round only (1 probe per stratum), where the within-stratum variance, and hence N_req, is not estimable (reported as 0).

At N = 100, the achieved eps(delta = 0.1) is 0.5-0.7 (Chebyshev), and the heavy tail makes even this optimistic because censored cubes understate s². **Rankings work at an eps that would be useless as an (eps, delta) certificate.**

## 5. Frozen on DEV → held-out WIDE and TARGET tables

**Reading guide.**
- Every held-out hard case is above 20k conflicts, so the rows "direct 20k, open cases by X" are X alone, costed with the 20k run the table already made.
- The rows "direct C, open cases by X" rank the cases that a fresh run to C conflicts resolves by their exact d. The cases still open are ranked above C by the frozen model X.

**Key comparisons at about 50k conflicts per case (module default):**
- **TARGET.**
  - Sampling alone gets 0.678 / C 0.787, against 0.453 / C 0.560 for a direct run to 50k, which leaves 94% of the cases open.
  - free20k alone gets 0.793 / C 0.885.
  - free20k + sampling gets 0.820 / C 0.893.
  - The equal-cost alternative to adding sampling, "direct 50k, open cases by free20k", gets 0.800 / C 0.889. **Sampling's added value over continuing the solve is +0.02 within-cell and +0.004 C.**
- **WIDE.**
  - Sampling alone (0.749) is slightly *below* a direct run to 50k (0.767).
  - "direct 50k + free20k" (0.886, 45k conflicts) matches "free20k + sampling" (0.882, 49k conflicts).
  - Spending both budgets (0.903, 83k conflicts) is about equal to a direct run to 100k with free20k (0.922, 77k conflicts).

### The DEV-only pick: strat:row:8:5, budgeted T = 50k, b = 2000

WIDE: 509 cases, sampling cost 46093 conflicts mean / 51981 max, 1.29e+07 propagations, ~1.12 s per case.

| model / baseline | within ρ | pooled ρ | log-RMSE | cost | per-cell within ρ |
|---|---|---|---|---|---|
| fhat-form(refit) | 0.422 | 0.340 | 1.24 | 0 | m10_n20_w103_pure: 0.35, m11_n21_w117_pure: 0.76, m9_n23_w104: 0.793, m9_n18_w86_pure: 0.211, m10_n19_w99_pure: -0.007 |
| samp | 0.652 | 0.611 | 1.19 | +samp | m10_n20_w103_pure: 0.772, m11_n21_w117_pure: 0.671, m9_n23_w104: 0.267, m9_n18_w86_pure: 0.798, m10_n19_w99_pure: 0.75 |
| samp(ratio calib) | 0.652 | 0.611 | 0.99 | +samp | m10_n20_w103_pure: 0.772, m11_n21_w117_pure: 0.671, m9_n23_w104: 0.267, m9_n18_w86_pure: 0.798, m10_n19_w99_pure: 0.75 |
| samp+vol | 0.661 | 0.641 | 1.01 | +samp | m10_n20_w103_pure: 0.777, m11_n21_w117_pure: 0.667, m9_n23_w104: 0.303, m9_n18_w86_pure: 0.806, m10_n19_w99_pure: 0.75 |
| free20k(A4) | 0.855 | 0.849 | 0.76 | 0 | m10_n20_w103_pure: 0.876, m11_n21_w117_pure: 0.838, m9_n23_w104: 0.795, m9_n18_w86_pure: 0.883, m10_n19_w99_pure: 0.881 |
| free20k+samp | 0.870 | 0.870 | 0.68 | +samp | m10_n20_w103_pure: 0.885, m11_n21_w117_pure: 0.863, m9_n23_w104: 0.798, m9_n18_w86_pure: 0.91, m10_n19_w99_pure: 0.894 |
| current label clip(fhat,20k,400k) | – | 0.331 | 1.30 | 0 | m10_n20_w103_pure: 0.35, m11_n21_w117_pure: nan, m9_n23_w104: 0.693, m9_n18_w86_pure: 0.211, m10_n19_w99_pure: -0.013 |
| direct fresh run, cap 50k | 0.767 | 0.779 | 1.52 | 45.2k | m10_n20_w103_pure: 0.788, m11_n21_w117_pure: 0.773, m9_n23_w104: 0.708, m9_n18_w86_pure: 0.833, m10_n19_w99_pure: 0.734 |
| direct fresh run, cap 100k | 0.879 | 0.905 | 1.06 | 77.3k | m10_n20_w103_pure: 0.921, m11_n21_w117_pure: 0.85, m9_n23_w104: 0.829, m9_n18_w86_pure: 0.955, m10_n19_w99_pure: 0.839 |
| direct fresh run, cap 200k | 0.957 | 0.979 | 0.68 | 121.3k | m10_n20_w103_pure: 0.981, m11_n21_w117_pure: 0.906, m9_n23_w104: 0.946, m9_n18_w86_pure: 0.997, m10_n19_w99_pure: 0.957 |
| direct 20k, open cases by samp | 0.652 | 0.611 | – | 66.1k | m10_n20_w103_pure: 0.772, m11_n21_w117_pure: 0.671, m9_n23_w104: 0.267, m9_n18_w86_pure: 0.798, m10_n19_w99_pure: 0.75 |
| direct 20k, open cases by free20k(A4) | 0.855 | 0.849 | – | 20.0k | m10_n20_w103_pure: 0.876, m11_n21_w117_pure: 0.838, m9_n23_w104: 0.795, m9_n18_w86_pure: 0.883, m10_n19_w99_pure: 0.881 |
| direct 20k, open cases by free20k+samp | 0.870 | 0.870 | – | 66.1k | m10_n20_w103_pure: 0.885, m11_n21_w117_pure: 0.863, m9_n23_w104: 0.798, m9_n18_w86_pure: 0.91, m10_n19_w99_pure: 0.894 |
| direct 50k, open cases by samp | 0.792 | 0.765 | – | 80.1k | m10_n20_w103_pure: 0.854, m11_n21_w117_pure: 0.842, m9_n23_w104: 0.585, m9_n18_w86_pure: 0.912, m10_n19_w99_pure: 0.769 |
| direct 50k, open cases by free20k(A4) | 0.886 | 0.907 | – | 45.2k | m10_n20_w103_pure: 0.916, m11_n21_w117_pure: 0.861, m9_n23_w104: 0.848, m9_n18_w86_pure: 0.916, m10_n19_w99_pure: 0.888 |
| direct 50k, open cases by free20k+samp | 0.898 | 0.918 | – | 80.1k | m10_n20_w103_pure: 0.921, m11_n21_w117_pure: 0.887, m9_n23_w104: 0.846, m9_n18_w86_pure: 0.937, m10_n19_w99_pure: 0.899 |
| direct 100k, open cases by samp | 0.854 | 0.860 | – | 104.5k | m10_n20_w103_pure: 0.925, m11_n21_w117_pure: 0.865, m9_n23_w104: 0.719, m9_n18_w86_pure: 0.966, m10_n19_w99_pure: 0.795 |
| direct 100k, open cases by free20k(A4) | 0.922 | 0.943 | – | 77.3k | m10_n20_w103_pure: 0.954, m11_n21_w117_pure: 0.919, m9_n23_w104: 0.878, m9_n18_w86_pure: 0.963, m10_n19_w99_pure: 0.898 |
| direct 100k, open cases by free20k+samp | 0.925 | 0.947 | – | 104.5k | m10_n20_w103_pure: 0.957, m11_n21_w117_pure: 0.917, m9_n23_w104: 0.875, m9_n18_w86_pure: 0.971, m10_n19_w99_pure: 0.905 |

TARGET: 450 cases, sampling cost 50594 conflicts mean / 63168 max, 1.82e+07 propagations, ~1.67 s per case.

| model / baseline | within ρ | pooled ρ | log-RMSE | C within | C pooled | cost | per-cell within ρ |
|---|---|---|---|---|---|---|---|
| fhat-form(refit) | 0.240 | 0.192 | 1.27 | 0.674 | 0.645 | 0 | m12_n18_w109: 0.27, m13_n19_w123: 0.174, m16_n17_w134: 0.276 |
| samp | 0.681 | 0.695 | 1.30 | 0.793 | 0.796 | +samp | m12_n18_w109: 0.745, m13_n19_w123: 0.71, m16_n17_w134: 0.588 |
| samp(ratio calib) | 0.681 | 0.695 | 0.85 | 0.793 | 0.796 | +samp | m12_n18_w109: 0.745, m13_n19_w123: 0.71, m16_n17_w134: 0.588 |
| samp+vol | 0.695 | 0.630 | 0.95 | 0.803 | 0.784 | +samp | m12_n18_w109: 0.756, m13_n19_w123: 0.715, m16_n17_w134: 0.614 |
| free20k(A4) | 0.793 | 0.788 | 0.72 | 0.885 | 0.885 | 0 | m12_n18_w109: 0.787, m13_n19_w123: 0.819, m16_n17_w134: 0.772 |
| free20k+samp | 0.826 | 0.829 | 0.66 | 0.895 | 0.895 | +samp | m12_n18_w109: 0.835, m13_n19_w123: 0.842, m16_n17_w134: 0.799 |
| current label clip(fhat,20k,400k) | – | 0.228 | 1.22 | 0.517 | 0.521 | 0 | m12_n18_w109: 0.335, m13_n19_w123: nan, m16_n17_w134: nan |
| direct fresh run, cap 50k | 0.453 | 0.492 | 2.02 | 0.560 | 0.563 | 49.1k | m12_n18_w109: 0.568, m13_n19_w123: 0.508, m16_n17_w134: 0.283 |
| direct fresh run, cap 100k | 0.696 | 0.715 | 1.47 | 0.641 | 0.644 | 94.3k | m12_n18_w109: 0.763, m13_n19_w123: 0.726, m16_n17_w134: 0.598 |
| direct fresh run, cap 200k | 0.894 | 0.897 | 0.97 | 0.766 | 0.766 | 173.8k | m12_n18_w109: 0.918, m13_n19_w123: 0.865, m16_n17_w134: 0.9 |
| direct 20k, open cases by samp | 0.681 | 0.695 | – | 0.793 | 0.796 | 70.6k | m12_n18_w109: 0.745, m13_n19_w123: 0.71, m16_n17_w134: 0.588 |
| direct 20k, open cases by free20k(A4) | 0.793 | 0.788 | – | 0.885 | 0.885 | 20.0k | m12_n18_w109: 0.787, m13_n19_w123: 0.819, m16_n17_w134: 0.772 |
| direct 20k, open cases by free20k+samp | 0.826 | 0.829 | – | 0.895 | 0.895 | 70.6k | m12_n18_w109: 0.835, m13_n19_w123: 0.842, m16_n17_w134: 0.799 |
| direct 50k, open cases by samp | 0.704 | 0.717 | – | 0.802 | 0.805 | 97.2k | m12_n18_w109: 0.761, m13_n19_w123: 0.752, m16_n17_w134: 0.599 |
| direct 50k, open cases by free20k(A4) | 0.800 | 0.796 | – | 0.889 | 0.888 | 49.1k | m12_n18_w109: 0.798, m13_n19_w123: 0.827, m16_n17_w134: 0.774 |
| direct 50k, open cases by free20k+samp | 0.829 | 0.833 | – | 0.897 | 0.898 | 97.2k | m12_n18_w109: 0.841, m13_n19_w123: 0.847, m16_n17_w134: 0.8 |
| direct 100k, open cases by samp | 0.745 | 0.759 | – | 0.816 | 0.820 | 138.5k | m12_n18_w109: 0.787, m13_n19_w123: 0.766, m16_n17_w134: 0.682 |
| direct 100k, open cases by free20k(A4) | 0.831 | 0.830 | – | 0.901 | 0.901 | 94.3k | m12_n18_w109: 0.827, m13_n19_w123: 0.861, m16_n17_w134: 0.804 |
| direct 100k, open cases by free20k+samp | 0.852 | 0.855 | – | 0.908 | 0.908 | 138.5k | m12_n18_w109: 0.862, m13_n19_w123: 0.867, m16_n17_w134: 0.828 |

LOCO over all cells with hard labels (DEV + WIDE + TARGET; censored rows are scored by Harrell's C, never fitted):

| model | within ρ | pooled ρ | log-RMSE | C within | C pooled |
|---|---|---|---|---|---|
| fhat-form(refit) | 0.295 | 0.592 | 1.05 | 0.626 | 0.757 |
| samp | 0.689 | 0.770 | 0.86 | 0.765 | 0.826 |
| samp(ratio calib) | 0.689 | 0.773 | 0.86 | 0.765 | 0.826 |
| samp+vol | 0.693 | 0.771 | 0.85 | 0.767 | 0.828 |
| free20k(A4) | 0.805 | 0.876 | 0.62 | 0.830 | 0.876 |
| free20k+samp | 0.849 | 0.904 | 0.57 | 0.853 | 0.891 |

### The module default: knuth:row:8, budgeted T = 50k, b = 5000

WIDE: 509 cases, sampling cost 49451 conflicts mean / 54997 max, 1.6e+07 propagations, ~1.64 s per case.

| model / baseline | within ρ | pooled ρ | log-RMSE | cost | per-cell within ρ |
|---|---|---|---|---|---|
| fhat-form(refit) | 0.422 | 0.339 | 1.24 | 0 | m10_n20_w103_pure: 0.35, m11_n21_w117_pure: 0.76, m9_n23_w104: 0.793, m9_n18_w86_pure: 0.211, m10_n19_w99_pure: -0.007 |
| samp | 0.749 | 0.762 | 1.17 | +samp | m10_n20_w103_pure: 0.855, m11_n21_w117_pure: 0.746, m9_n23_w104: 0.567, m9_n18_w86_pure: 0.822, m10_n19_w99_pure: 0.753 |
| samp(ratio calib) | 0.749 | 0.762 | 0.91 | +samp | m10_n20_w103_pure: 0.855, m11_n21_w117_pure: 0.746, m9_n23_w104: 0.567, m9_n18_w86_pure: 0.822, m10_n19_w99_pure: 0.753 |
| samp+vol | 0.761 | 0.736 | 0.95 | +samp | m10_n20_w103_pure: 0.855, m11_n21_w117_pure: 0.764, m9_n23_w104: 0.601, m9_n18_w86_pure: 0.828, m10_n19_w99_pure: 0.756 |
| free20k(A4) | 0.855 | 0.849 | 0.76 | 0 | m10_n20_w103_pure: 0.876, m11_n21_w117_pure: 0.838, m9_n23_w104: 0.795, m9_n18_w86_pure: 0.883, m10_n19_w99_pure: 0.881 |
| free20k+samp | 0.882 | 0.879 | 0.67 | +samp | m10_n20_w103_pure: 0.902, m11_n21_w117_pure: 0.868, m9_n23_w104: 0.827, m9_n18_w86_pure: 0.908, m10_n19_w99_pure: 0.905 |
| current label clip(fhat,20k,400k) | – | 0.331 | 1.30 | 0 | m10_n20_w103_pure: 0.35, m11_n21_w117_pure: nan, m9_n23_w104: 0.693, m9_n18_w86_pure: 0.211, m10_n19_w99_pure: -0.013 |
| direct fresh run, cap 50k | 0.767 | 0.779 | 1.52 | 45.2k | m10_n20_w103_pure: 0.788, m11_n21_w117_pure: 0.773, m9_n23_w104: 0.708, m9_n18_w86_pure: 0.833, m10_n19_w99_pure: 0.734 |
| direct fresh run, cap 100k | 0.879 | 0.905 | 1.06 | 77.3k | m10_n20_w103_pure: 0.921, m11_n21_w117_pure: 0.85, m9_n23_w104: 0.829, m9_n18_w86_pure: 0.955, m10_n19_w99_pure: 0.839 |
| direct fresh run, cap 200k | 0.957 | 0.979 | 0.68 | 121.3k | m10_n20_w103_pure: 0.981, m11_n21_w117_pure: 0.906, m9_n23_w104: 0.946, m9_n18_w86_pure: 0.997, m10_n19_w99_pure: 0.957 |
| direct 20k, open cases by samp | 0.749 | 0.762 | – | 69.5k | m10_n20_w103_pure: 0.855, m11_n21_w117_pure: 0.746, m9_n23_w104: 0.567, m9_n18_w86_pure: 0.822, m10_n19_w99_pure: 0.753 |
| direct 20k, open cases by free20k(A4) | 0.855 | 0.849 | – | 20.0k | m10_n20_w103_pure: 0.876, m11_n21_w117_pure: 0.838, m9_n23_w104: 0.795, m9_n18_w86_pure: 0.883, m10_n19_w99_pure: 0.881 |
| direct 20k, open cases by free20k+samp | 0.882 | 0.879 | – | 69.5k | m10_n20_w103_pure: 0.902, m11_n21_w117_pure: 0.868, m9_n23_w104: 0.827, m9_n18_w86_pure: 0.908, m10_n19_w99_pure: 0.905 |
| direct 50k, open cases by samp | 0.811 | 0.837 | – | 82.6k | m10_n20_w103_pure: 0.882, m11_n21_w117_pure: 0.782, m9_n23_w104: 0.699, m9_n18_w86_pure: 0.925, m10_n19_w99_pure: 0.766 |
| direct 50k, open cases by free20k(A4) | 0.886 | 0.907 | – | 45.2k | m10_n20_w103_pure: 0.916, m11_n21_w117_pure: 0.861, m9_n23_w104: 0.848, m9_n18_w86_pure: 0.916, m10_n19_w99_pure: 0.888 |
| direct 50k, open cases by free20k+samp | 0.903 | 0.921 | – | 82.6k | m10_n20_w103_pure: 0.928, m11_n21_w117_pure: 0.88, m9_n23_w104: 0.861, m9_n18_w86_pure: 0.935, m10_n19_w99_pure: 0.909 |
| direct 100k, open cases by samp | 0.858 | 0.892 | – | 106.3k | m10_n20_w103_pure: 0.942, m11_n21_w117_pure: 0.807, m9_n23_w104: 0.747, m9_n18_w86_pure: 0.969, m10_n19_w99_pure: 0.825 |
| direct 100k, open cases by free20k(A4) | 0.922 | 0.943 | – | 77.3k | m10_n20_w103_pure: 0.954, m11_n21_w117_pure: 0.919, m9_n23_w104: 0.878, m9_n18_w86_pure: 0.963, m10_n19_w99_pure: 0.898 |
| direct 100k, open cases by free20k+samp | 0.930 | 0.949 | – | 106.3k | m10_n20_w103_pure: 0.962, m11_n21_w117_pure: 0.92, m9_n23_w104: 0.885, m9_n18_w86_pure: 0.969, m10_n19_w99_pure: 0.916 |

TARGET: 450 cases, sampling cost 51717 conflicts mean / 54997 max, 2.29e+07 propagations, ~2.22 s per case.

| model / baseline | within ρ | pooled ρ | log-RMSE | C within | C pooled | cost | per-cell within ρ |
|---|---|---|---|---|---|---|---|
| fhat-form(refit) | 0.240 | 0.192 | 1.27 | 0.674 | 0.645 | 0 | m12_n18_w109: 0.27, m13_n19_w123: 0.174, m16_n17_w134: 0.276 |
| samp | 0.678 | 0.691 | 1.40 | 0.787 | 0.794 | +samp | m12_n18_w109: 0.744, m13_n19_w123: 0.644, m16_n17_w134: 0.646 |
| samp(ratio calib) | 0.678 | 0.691 | 0.93 | 0.787 | 0.794 | +samp | m12_n18_w109: 0.744, m13_n19_w123: 0.644, m16_n17_w134: 0.646 |
| samp+vol | 0.687 | 0.561 | 0.97 | 0.796 | 0.755 | +samp | m12_n18_w109: 0.747, m13_n19_w123: 0.646, m16_n17_w134: 0.667 |
| free20k(A4) | 0.793 | 0.788 | 0.72 | 0.885 | 0.885 | 0 | m12_n18_w109: 0.787, m13_n19_w123: 0.819, m16_n17_w134: 0.772 |
| free20k+samp | 0.820 | 0.822 | 0.66 | 0.893 | 0.894 | +samp | m12_n18_w109: 0.82, m13_n19_w123: 0.837, m16_n17_w134: 0.803 |
| current label clip(fhat,20k,400k) | – | 0.228 | 1.22 | 0.517 | 0.521 | 0 | m12_n18_w109: 0.335, m13_n19_w123: nan, m16_n17_w134: nan |
| direct fresh run, cap 50k | 0.453 | 0.492 | 2.02 | 0.560 | 0.563 | 49.1k | m12_n18_w109: 0.568, m13_n19_w123: 0.508, m16_n17_w134: 0.283 |
| direct fresh run, cap 100k | 0.696 | 0.715 | 1.47 | 0.641 | 0.644 | 94.3k | m12_n18_w109: 0.763, m13_n19_w123: 0.726, m16_n17_w134: 0.598 |
| direct fresh run, cap 200k | 0.894 | 0.897 | 0.97 | 0.766 | 0.766 | 173.8k | m12_n18_w109: 0.918, m13_n19_w123: 0.865, m16_n17_w134: 0.9 |
| direct 20k, open cases by samp | 0.678 | 0.691 | – | 0.787 | 0.794 | 71.7k | m12_n18_w109: 0.744, m13_n19_w123: 0.644, m16_n17_w134: 0.646 |
| direct 20k, open cases by free20k(A4) | 0.793 | 0.788 | – | 0.885 | 0.885 | 20.0k | m12_n18_w109: 0.787, m13_n19_w123: 0.819, m16_n17_w134: 0.772 |
| direct 20k, open cases by free20k+samp | 0.820 | 0.822 | – | 0.893 | 0.894 | 71.7k | m12_n18_w109: 0.82, m13_n19_w123: 0.837, m16_n17_w134: 0.803 |
| direct 50k, open cases by samp | 0.702 | 0.715 | – | 0.796 | 0.802 | 98.1k | m12_n18_w109: 0.764, m13_n19_w123: 0.695, m16_n17_w134: 0.648 |
| direct 50k, open cases by free20k(A4) | 0.800 | 0.796 | – | 0.889 | 0.888 | 49.1k | m12_n18_w109: 0.798, m13_n19_w123: 0.827, m16_n17_w134: 0.774 |
| direct 50k, open cases by free20k+samp | 0.825 | 0.827 | – | 0.896 | 0.897 | 98.1k | m12_n18_w109: 0.827, m13_n19_w123: 0.842, m16_n17_w134: 0.805 |
| direct 100k, open cases by samp | 0.745 | 0.755 | – | 0.811 | 0.817 | 139.3k | m12_n18_w109: 0.813, m13_n19_w123: 0.712, m16_n17_w134: 0.711 |
| direct 100k, open cases by free20k(A4) | 0.831 | 0.830 | – | 0.901 | 0.901 | 94.3k | m12_n18_w109: 0.827, m13_n19_w123: 0.861, m16_n17_w134: 0.804 |
| direct 100k, open cases by free20k+samp | 0.848 | 0.850 | – | 0.906 | 0.907 | 139.3k | m12_n18_w109: 0.85, m13_n19_w123: 0.866, m16_n17_w134: 0.828 |

LOCO over all cells with hard labels (DEV + WIDE + TARGET; censored rows are scored by Harrell's C, never fitted):

| model | within ρ | pooled ρ | log-RMSE | C within | C pooled |
|---|---|---|---|---|---|
| fhat-form(refit) | 0.295 | 0.592 | 1.05 | 0.626 | 0.757 |
| samp | 0.713 | 0.823 | 0.81 | 0.773 | 0.842 |
| samp(ratio calib) | 0.713 | 0.826 | 0.88 | 0.773 | 0.843 |
| samp+vol | 0.714 | 0.822 | 0.80 | 0.774 | 0.844 |
| free20k(A4) | 0.805 | 0.876 | 0.62 | 0.830 | 0.876 |
| free20k+samp | 0.848 | 0.910 | 0.55 | 0.853 | 0.895 |


## 6. Statistic variants of the same cubes (`sampling_variants.py`, `variants.json`; knuth:row:8, T50k, b5000)

Is there a lower-variance statistic of the same cube records? No. **The signal is in the heavy tail.**

| statistic | DEV LOCO within | WIDE within | TARGET within / C |
|---|---|---|---|
| **mu (censoring-aware mean, Pareto tail term)** | **0.695** | **0.749** | **0.678 / 0.787** |
| mu_lb (mean of min(xi, b), no tail term) | 0.625 | 0.625 | 0.623 / 0.757 |
| median of 5 means | 0.544 | 0.592 | 0.580 / 0.732 |
| geometric mean | −0.080 | 0.077 | 0.213 / 0.599 |
| Knuth tree size only (live leaves, **0 conflicts**) | −0.163 | −0.084 | 0.001 / 0.509 |
| tree size × median cube work | 0.109 | 0.298 | 0.500 / 0.726 |

- The robust statistics throw away the rare expensive cubes that carry the ranking.
- The Pareto tail correction for budget-censored cubes adds 0.06-0.12.
- The UP-pruned tree size alone has **no** signal. A1/A2's finding explains why: the failed-literal and propagation structure is nearly identical across the cases of a cell. The difficulty lives in the CDCL work below the leaves, not in the size of the tree.

## 7. Accuracy vs cost (both samplers; mean conflicts per case)

The TARGET runs were stored at b ≤ 5000, so b = 10000 cannot be derived there. N = 20 is below the stratum count for strat, which biases that estimator.

| sampler | op | DEV cost | DEV LOCO within | WIDE cost (mean) | WIDE within (frozen) | WIDE +free20k | TARGET cost (mean) | TARGET within | TARGET C | TARGET +free20k C | UP-refuted / budget-hit (TARGET) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| knuth:row:8 | N=20, b=2000 | 6.4k | 0.350 | 11.4k | 0.522 | 0.862 | 20.3k | 0.604 | 0.754 | 0.887 | 0.09 / 0.37 |
| knuth:row:8 | N=20, b=5000 | 8.8k | 0.336 | 18.9k | 0.616 | 0.867 | 41.5k | 0.625 | 0.775 | 0.887 | 0.09 / 0.32 |
| knuth:row:8 | N=20, b=10000 | 10.1k | 0.353 | 25.7k | 0.676 | 0.869 | – | – | – | – | – / – |
| knuth:row:8 | N=50, b=2000 | 16.5k | 0.592 | 28.4k | 0.646 | 0.865 | 51.5k | 0.674 | 0.782 | 0.890 | 0.09 / 0.38 |
| knuth:row:8 | N=50, b=5000 | 22.3k | 0.582 | 47.2k | 0.752 | 0.879 | 105.0k | 0.703 | 0.809 | 0.891 | 0.09 / 0.32 |
| knuth:row:8 | N=50, b=10000 | 25.2k | 0.578 | 63.9k | 0.817 | 0.885 | – | – | – | – | – / – |
| knuth:row:8 | N=100, b=2000 | 33.0k | 0.672 | 56.3k | 0.674 | 0.868 | 102.5k | 0.689 | 0.798 | 0.893 | 0.09 / 0.38 |
| knuth:row:8 | N=100, b=5000 | 45.0k | 0.700 | 93.0k | 0.796 | 0.879 | 208.7k | 0.729 | 0.827 | 0.894 | 0.09 / 0.32 |
| knuth:row:8 | N=100, b=10000 | 51.4k | 0.678 | 125.8k | 0.855 | 0.890 | – | – | – | – | – / – |
| knuth:row:8 | T=20000, b=2000 | 20.0k | 0.656 | 20.6k | 0.630 | 0.865 | 20.9k | 0.645 | 0.761 | 0.892 | 0.09 / 0.39 |
| knuth:row:8 | T=20000, b=5000 | 21.0k | 0.590 | 21.6k | 0.658 | 0.871 | 21.8k | 0.597 | 0.753 | 0.890 | 0.08 / 0.36 |
| knuth:row:8 | T=20000, b=10000 | 21.1k | 0.590 | 22.6k | 0.717 | 0.874 | – | – | – | – | – / – |
| knuth:row:8 | T=50000, b=2000 | 32.3k | 0.670 | 44.6k | 0.664 | 0.867 | 50.2k | 0.679 | 0.786 | 0.892 | 0.09 / 0.38 |
| knuth:row:8 | T=50000, b=5000 | 39.3k | 0.695 | 49.5k | 0.749 | 0.882 | 51.7k | 0.678 | 0.787 | 0.893 | 0.08 / 0.34 |
| knuth:row:8 | T=50000, b=10000 | 40.9k | 0.667 | 51.1k | 0.786 | 0.887 | – | – | – | – | – / – |
| knuth:row:8 | T=100000, b=2000 | 33.0k | 0.672 | 55.7k | 0.674 | 0.868 | 86.9k | 0.687 | 0.795 | 0.893 | 0.09 / 0.38 |
| knuth:row:8 | T=100000, b=5000 | 44.8k | 0.700 | 78.6k | 0.793 | 0.880 | 98.3k | 0.723 | 0.811 | 0.894 | 0.09 / 0.33 |
| knuth:row:8 | T=100000, b=10000 | 50.2k | 0.678 | 86.7k | 0.843 | 0.893 | – | – | – | – | – / – |
| strat:row:8:5 | N=20, b=2000 | 8.6k | 0.539 | 14.1k | 0.605 | 0.859 | 24.0k | 0.730 | 0.815 | 0.888 | 0.04 / 0.46 |
| strat:row:8:5 | N=20, b=5000 | 12.1k | 0.493 | 23.9k | 0.691 | 0.866 | 49.9k | 0.751 | 0.839 | 0.888 | 0.04 / 0.39 |
| strat:row:8:5 | N=20, b=10000 | 14.2k | 0.514 | – | – | – | – | – | – | – | – / – |
| strat:row:8:5 | N=50, b=2000 | 19.7k | 0.652 | 31.6k | 0.630 | 0.869 | 54.7k | 0.686 | 0.798 | 0.893 | 0.06 / 0.41 |
| strat:row:8:5 | N=50, b=5000 | 27.3k | 0.577 | 52.8k | 0.766 | 0.877 | 112.4k | 0.715 | 0.820 | 0.893 | 0.06 / 0.35 |
| strat:row:8:5 | N=50, b=10000 | 31.8k | 0.604 | – | – | – | – | – | – | – | – / – |
| strat:row:8:5 | N=100, b=2000 | 37.2k | 0.746 | 59.3k | 0.648 | 0.868 | 105.0k | 0.698 | 0.803 | 0.896 | 0.07 / 0.39 |
| strat:row:8:5 | N=100, b=5000 | 50.8k | 0.725 | 98.3k | 0.797 | 0.886 | 214.3k | 0.732 | 0.827 | 0.896 | 0.07 / 0.33 |
| strat:row:8:5 | N=100, b=10000 | 58.3k | 0.705 | – | – | – | – | – | – | – | – / – |
| strat:row:8:5 | T=20000, b=2000 | 20.5k | 0.625 | 22.6k | 0.596 | 0.867 | 33.8k | 0.672 | 0.788 | 0.892 | 0.08 / 0.38 |
| strat:row:8:5 | T=20000, b=5000 | 22.3k | 0.505 | 31.9k | 0.690 | 0.872 | 67.7k | 0.704 | 0.814 | 0.890 | 0.08 / 0.33 |
| strat:row:8:5 | T=20000, b=10000 | 24.3k | 0.553 | – | – | – | – | – | – | – | – / – |
| strat:row:8:5 | T=50000, b=2000 | 36.3k | 0.743 | 46.1k | 0.652 | 0.870 | 50.6k | 0.681 | 0.793 | 0.895 | 0.08 / 0.38 |
| strat:row:8:5 | T=50000, b=5000 | 42.3k | 0.677 | 51.3k | 0.731 | 0.878 | 73.1k | 0.705 | 0.815 | 0.893 | 0.08 / 0.33 |
| strat:row:8:5 | T=50000, b=10000 | 43.9k | 0.691 | – | – | – | – | – | – | – | – / – |
| strat:row:8:5 | T=100000, b=2000 | 37.2k | 0.744 | 58.7k | 0.656 | 0.869 | 88.3k | 0.699 | 0.800 | 0.895 | 0.08 / 0.38 |
| strat:row:8:5 | T=100000, b=5000 | 50.4k | 0.706 | 81.7k | 0.790 | 0.881 | 101.1k | 0.720 | 0.820 | 0.896 | 0.08 / 0.33 |
| strat:row:8:5 | T=100000, b=10000 | 56.2k | 0.704 | – | – | – | – | – | – | – | – / – |

## 8. Negative results and caveats

- **Uniform Chivilikhin sampling as specified (designs i-iv with i.i.d. uniform cubes) does not work on this encoding.** The lex symmetry breaking refutes almost every cube (§3). All the positive results use the Knuth / stratified-Knuth modification.
- **The paper's lookahead-chosen B (the UP-weight heuristic) is the worst uniform design here.** Under Knuth probing it is fine but costs more than the row order.
- **At equal cost, continuing the direct solve beats sampling wherever a meaningful fraction of cases resolves.** This holds on all of DEV hard, on MID, and nearly on WIDE (direct 50k 0.767 vs sampling T50k 0.749 within-cell). Sampling only wins where more than 90% of the cases stay open at the budget, which is true on the target tables.
- **Sampling alone is below A4's free20k on every held-out set.** Its value is as a second feature (+0.03-0.04 within-cell over free20k), at about 50k conflicts per case.
- **Per-case cost grows with hardness.** At a fixed (N, b), the cost is 51k on DEV, 126k on WIDE and ~210k on TARGET (N100, b5000/10000). Only the budgeted mode (`total_budget`) gives a real cap. At the 50k cap, the accuracy on TARGET drops from 0.73 to 0.68 (knuth).
- **Absolute magnitude is only moderately good.**
  - The frozen log-RMSE is about 0.9 on WIDE and TARGET, a factor of about 2.5, with unit-slope calibration.
  - The OLS calibration from DEV extrapolates badly (log-RMSE 1.2-1.4) because DEV's range of d is narrow.
  - A4's free20k gets 0.72-0.76 and free20k+samp gets 0.65-0.67.
- **Cost at deployment scale.** Running the estimator on every case censored at 20k on the target tables (about 15k cases) at 50k conflicts each would cost about 750M conflicts, comparable to A1's entire deepen run (770M).
- **Stratification helped on DEV only.**
  - DEV LOCO within-cell: 0.743 vs 0.695 at the budgeted point.
  - WIDE: 0.652 vs 0.749. TARGET: 0.681 vs 0.678.
  - At equal (N, b) off DEV, stratified and plain Knuth are the same (N100 b5000: WIDE 0.797 vs 0.796, TARGET 0.732 vs 0.729).
  - At the 50k cap on hard cases, the stratified sampler runs about one round, which is one probe per stratum. Its within-stratum variance, and hence N_req, is then not estimable.
- **The per-cube budget b should grow with case hardness. DEV cannot choose it.**
  - DEV's hard cases rarely hit the cube budget (5-10% of cubes), so DEV prefers b ≤ 5000.
  - On WIDE, at the same 50k cap, b = 10000 is better than b = 5000, which is better than b = 2000 (0.786 / 0.749 / 0.664 within-cell).
  - On TARGET, 33-38% of the cubes hit b.
  - A label-free rule could set b so that about 10% of the cubes are censored, because the hit fraction is observable on unlabeled cases. It was not tested. This observation comes from the held-out sets and was **not** used to set the default.
- The E10 design screen is small (30 hard cases in 3 cells). The design ranking is clear-cut (UP-refuted fractions of 0.9-1.0 vs 0.0-0.4), but differences below about 0.05 in ρ there are noise.

## 9. TODOs for the integrator (I did not edit shared files)

1. **Difficulty labels for target tables (`zar_ub/difficulty.py`, `casetable.py`).** If A4's `free20k` becomes the censored label, sampling adds +0.03 within-cell / +0.01 C on TARGET for about 50k conflicts per case.
   - Recommend it only where cost is acceptable: for example the stratified CRN sample of each target table (`make_sample`), or the top-ranked cases by free20k.
   - Not recommended for all 15k censored cases.
   - The combined model must be refit on DEV with the chosen operating point. `sampling_final.py` does this, and the coefficients are in `sampling_data/final_eval*.json`.
2. **Cases still open at 2M** (191 in `ground_truth.jsonl`). No exact label is affordable for these. `hardness_sampling.estimate()` with the default operating point is the only estimator here that measures *work below the budget frontier* rather than extrapolating a probe. Consider it for ordering those 191 cases, for example when choosing which open cases the evolved prunes should target first.
3. **Do not use `sampler="uniform"`** for this encoding, whatever the design (§3). The module keeps it only for the comparison.
4. The contract's `d_hat` is the DEV-calibrated estimate, and `d_hat_raw` is mu~. The calibration exists only for the two default operating points (`CALIBRATION` in the module). Other configurations return mu~ uncalibrated.
5. **The label file was not written back.** A cheap next step would be a `sampling_features_target.jsonl` over all 15k censored target cases, but it costs about 750M conflicts. Only do it after deciding (1).

## 10. Files

- `zar_ub/hardness_sampling.py`: the module.
  - `estimate()` (contract), `summarize()` (usable offline on stored records), `knuth_probes`, `stratified_probes`, `choose_B`, `sample_cubes`, `run_cubes`, `pareto_alpha`, `calibrate_d_hat`.
  - `DEFAULT` = the selected operating point. `CALIBRATION` = the frozen DEV fits.
- `sampling_run.py`: the runner (resumable, 3 processes). It stores every cube at N = 100, b = 10k (TARGET: 5k).
- `sampling_eval.py`: the (N, b) grid, LOCO, and the fhat/direct baselines.
- `sampling_final.py`: the headline evaluation (frozen tests, hybrids, all-cells LOCO).
- `sampling_designs.py`: the E10 design screen.
- `sampling_variants.py`: the statistics of §6.
- `sampling_combine.py`, `sampling_hybrid.py`, `sampling_retest.py`: earlier analyses (retest numbers in §4.3).
- `sampling_verify.py`: the reproducibility check.
- `sampling_a4feats.py`: A4's free20k inputs for the TARGET cases.
- `sampling_data/`:
  - Runs: `dev.jsonl` (knuth:row:8/10 on DEV hard + mid; row:6 and lookahead:8 on DEV hard), `dev_seed1.jsonl`, `dev_strat.jsonl`, `screen_e10.jsonl`, `wide_test.jsonl`, `wide_strat.jsonl`, `target.jsonl`, `target_strat.jsonl`, `target_a4.jsonl`.
  - Evaluations: `dev_eval.json`, `dev_strat_eval.json`, `screen_e10_eval.json`, `designs.json`, `variants.json`, `final_eval.json` (knuth:row:8), `final_eval_strat.json` (strat:row:8:5), `retest_knuth_row_8.json`, `verify_*.json`, and the logs.

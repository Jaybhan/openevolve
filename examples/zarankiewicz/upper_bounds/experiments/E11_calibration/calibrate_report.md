# E11 — censored-label calibration (design §6.2), 2026-09-21

Command: `python -m zar_ub calibrate --holdout 12,13,87` (output: `calibrate_33.json`, read by
`zar_ub.difficulty.load_calibration`; labels of the TRAIN tables from E10, `experiments/E10_difficulty/results.json`).

Model (design §6.2): `log fhat = a + b·log(max(c2000,1)) + g·log2_volume`, `d = min(max(cap, fhat), 20·cap)`.
Fit set: TRAIN cases (pure tables at w = z+1 for (9,9),(9,10),(10,10),(10,11),(11,11),(11,12),(12,12)) that were
still open at the 2,000-conflict cap and were refuted exactly (n = 709; true conflicts 2,001 … 221,874).
Hold-out: (12,13,87) after `python -m zar_ub deepen 12 13 3 3 87 --cap 200000` (103 of 104 cases open at 20k were
refuted; 1 remains censored at 200k), n = 276 fit-eligible cases.

| | n | Spearman ρ (fhat vs true) | log-RMSE | log-RMSE of a constant | log-RMSE of `d = cap` |
|---|---|---|---|---|---|
| TRAIN (fit) | 709 | 0.478 | 0.878 | – | – |
| hold-out (12,13,87) | 276 | 0.395 | 1.282 | 1.216 | 2.255 |

Coefficients: a = 0.0670, b = 0.5089, g = 0.04979.

**Reading.** The acceptance target of §6.2 (ρ ≥ 0.85, log-RMSE ≈ 0.5) is **not met**, and it cannot be met by this
model: on the fit set c2000 is a censored measurement (every case has c2000 ∈ {2000,…,2006} — the solver stops
at the first conflict at or above the budget), so the b-term is a constant and the estimator is effectively
`a' + g·log2_volume`. E10's ρ = 0.913 for c2000 holds over *all* 1,571 cases, where c2000 separates the cheap
cases from the censored ones; it says nothing about the ordering *inside* the censored set, which is what the
calibration has to predict. (A first fit without the clip produced b = −20.8 — least squares fitting the 2000–2006
noise — which is why `fhat` now clips c2000 to the first cap.)

Cheap proxies measured at the 2,000 cap on the same 985 cases (scratch experiment, not stored): Spearman with the true
cost, train / hold-out: log2_volume 0.478 / 0.395, nclauses 0.467 / 0.165, decisions 0.21 / 0.41, propagations
−0.05 / 0.05, restarts −0.16 / 0.03. Nothing cheap orders the censored tail.

**Consequences for the reward.** A censored label is `min(max(cap, fhat), 20·cap)`; with these coefficients fhat
at cap 20,000 is ≈ 2·10³·e^{0.05·log2_volume} (≈ 3·10⁴–10⁵ on (12,13)-sized cases), i.e. a mild volume tilt inside
the clip band, never the 20× ceiling. Censored work is therefore counted at roughly the cap, which is the same
conservative accounting the v1 evaluator used, and the design's `censored_share` metric plus the deepening daemon
(`python -m zar_ub deepen … --cap 200000`) remain the mechanism that turns censored labels into exact ones.
Whenever the calibration file is refitted, run `python -m zar_ub relabel --all` so stored censored labels (and
`table_hash`) follow it.

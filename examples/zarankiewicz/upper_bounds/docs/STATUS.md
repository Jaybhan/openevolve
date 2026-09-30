# Status — 2026-09-23 (after batch 3 + E28/E29: difficulty of open cells, single-cell rules)

What exists, what is proved, what was measured, what is open.  Every number
below is reproduced in `experiments/LOG.md` (E1–E27) with the command that
produced it.  Build provenance: `docs/build/*.md`, `docs/build/INTEGRATION.md`.

## 1. The system (design.md M0–M5 delivered)

| layer | delivered |
|---|---|
| Engine | Tan-style case split (Argument A budgets + Argument I on proper minors), fixed-sum CNF with double-lex, CaDiCaL probes, exact conflict labels on the training ladder (1,571 cases) and, since E24, on 2,238 hard cases (d up to 2M; 35,081 ground-truth rows), censored labels from the E24 hardness model (20k-probe search statistics + lookahead; E27), CRN samples |
| Trust gate | Lean 4.34 + targeted Mathlib; 46 forbidden constructs, NFKC + comment-stripped + raw scan, declared-name rule, nonce-tagged `#eval` masks, `#print axioms` audit, ladder L0–L5, sorry-hole auto-fill, HMAC-authenticated result cache, OS sandbox for the candidate subprocess |
| Reward | v3 (E27): verified floor 0.20 + 0.80·(0.28 G_train + 0.21 G_target + 0.07 G_gen + 0.14 Tail + 0.15 Depth + 0.15 Close), only work above 20k conflicts/case, family-balanced (square/wide/vwide/gen) work-weighted gains, Depth/Close from lower-bound work ≥ 1e6; suite v3 = 14 exact TRAIN cells (7 square + 7 wide), 7 non-empty targets; unverified ≤ 0.19; mirror-unsound rule (E19); MAP-Elites over (proven_gain, lean_ladder) |
| Proved library | Arguments A and D (both sides), deletion + waterfilling optimality, Argument I as conditional prune, DGH(4) for all (m,n;s,t), Farkas certificates over hidden pair codegrees (`Prune.ofFarkas`), permutation/sorting/enumerator/cover theorems (`Closure.lean`) |
| Closure | LRAT certification (cadical → drat-trim → lrat-check), Tier-1 closure theorems generated from the harness, promotion with leanchecker replay, closure daemon, verify-certs, audit-kills |
| Test infrastructure | 119 unit tests (E27), golden adversarial bank (25 candidates; E27 added 4 reward exploits), synthetic and stub-LLM loops, red-team attack bank, transfer matrix |

## 2. Bounds established here (all axioms ⊆ {propext, Quot.sound, Classical.choice})

| statement | how | status |
|---|---|---|
| z(9,9;3,3) ≤ 49 | 36 admissible cases, 19 killed by the proved library, 17 LRAT-certified; `Closures/Z_9_9_50.lean`, kernel `decide` 0.9 s | **Tier-1, unconditional** (encoding + certificate checkers trusted, listed in the report) |
| z(10,21;3,3) ≤ 106, z(11,19;3,3) ≤ 106, z(11,20;3,3) ≤ 111 | zero admissible cases under 16–18 named Tan-2022 facts; `Closures/Z_*.lean` | **Tier-1, conditional on the named facts** |
| z(10,20;3,3) ≤ 102 | 6 cases, all LRAT-certified (cited facts) | Tier-2 report (closure file not yet generated) |
| z(11,21;3,3) ≤ 116 | pure mode: argA + DGH kill every column partition (zero SAT) | Tier-2 (Python cover); Lean closure pending (pure-mode enumerator cost) |
| z(13,17;3,3) ≤ 116, z(13,18;3,3) ≤ 121, z(10,22;3,3) ≤ 111, z(10,23;3,3) ≤ 114 | pure column-side enumeration, argA + DGH, zero survivors | Tier-2 (matches or is weaker than published values; none new) |

No bound claimed here is new to the literature; they are the calibration that the pipeline
produces *checkable* versions of known results.  Open cells (21 in the m ≤ 16, n ≤ 23 block) are
listed in `docs/literature_review.md` §2.2.

## 3. What the experiments say about the thesis questions (proposal §2.3)

* **Does Lean fit an evolutionary loop?** Yes: 1.5–4 s per candidate for the whole suite with
  Mathlib-backed proofs (E5/E6), 0.01 s on a cache hit; the kernel closure of a cell is seconds
  (E14).  Lean is never the bottleneck; LLM latency is.
* **How to reward a near-miss proof?** Ladder L0–L5 with deterministic tactic auto-fill of typed
  `sorry` holes; partial credit capped at 0.19 so no unverified candidate outranks a verified one;
  the proved library is a floor so a compile slip is a penalty, not a cliff (E7a).
* **What is branch difficulty?** CaDiCaL conflicts to refute; heavy-tailed (top 10 % of cases =
  62 % of work); the 2k-conflict probe predicts it (ρ = 0.91), the calibrated extrapolation for
  censored cases does not (ρ = 0.39) — deepen labels instead of predicting them (E10, E12).
* **Pitfall 1 — difficulty of open cells (E24, E25, E27, E28).** Ground truth: 3,554 fresh 2M-conflict runs
  (770 M conflicts) turned 262 exact hard labels into 2,238 (191 cases still open at 2M, 0 SAT).  Against it,
  the old censored label was worse than the ρ = 0.39 suggested: a constant 400,000 on every censored case of
  the big targets, within-cell ρ 0.324 on hard cases, reward-gain error 0.061.  The proposal's techniques were
  implemented and measured: lookahead (AlphaMapleSAT/march/Knuth) is the best zero-conflict signal (ρ 0.71)
  and adds +0.035 inside the chosen model; Chivilikhin sampling works standalone (ρ 0.72) but does not reduce
  reward error, and its ~46k conflicts are better spent continuing the solve; LP slack is weak (0.38).  The
  adopted estimator is a 6-feature log-linear model (the 20k probe's decisions/conflict and restarts, two
  lookahead and two static features): held-out within-cell ρ **0.837**, Harrell C 0.847, gain error **0.020**
  (targets alone: ρ 0.514 → 0.835).  It is now the censored label (`difficulty.py`, floor = conflicts reached,
  ceiling 2M); 11,840 censored cases were relabelled.  E28 then deepened every censored target survivor to
  50k first (`deepen --survivors`, 25 min): only 226 of 6,318 finished — target cases are genuinely hard — so
  the model still labels almost all of them.  Remaining weakness: 2,468 target survivors sit at the 2M
  ceiling (their ranking is a tie; 51 % of the (16,17) ground-truth sample is above 2M).
* **Pitfall 2 — single-cell rules (E26, E27, E29).** Under reward v2, DGH (closes z(11,21) ≤ 116 with zero SAT)
  scored exactly 0.20 = doing nothing; accurate difficulty alone does not change that.  Fixes: the exactly-known
  wide cells enter TRAIN; the verified branch becomes reward v3 (family-balanced gains, only work above 20k
  conflicts counts, Depth = best single cell) with R3's anti-exploit fixes; E29 restricts the closure bonus to
  TARGET cells (with it over practice cells, DGH scored 3.9x the general recipe's uplift while helping no
  target).  Live evaluator: `lean_dgh4` **0.2000 → 0.3105**, recipe + DGH 0.2594 → 0.3301, general recipe
  0.2594 → 0.2497, every realizable exploit ≤ 0.2041, tiny-table clears 0.2000.  Dynamics with the shipped
  reward (8 seeds x 150 iterations): the best program is general + both specialists in **8/8** runs (v2: 1/8
  contained DGH), DGH in 97 % of the final population, 0 audit violations; E28's 0.02 mirror penalty keeps
  unsound-mirror twins from being reported best (0/8, was 3/8).
* **Do profile-only prunes suffice?** No: after Arguments A/D the remaining near-regular profiles
  need hidden-variable arguments.  The Farkas schema over pair codegrees removes 6.6 % of the
  remaining training work and 41 % on (10,20); DGH(4) removes nothing on square cells but closes
  wide cells with zero SAT (E12, E13).
* **What do LLMs do?** Cheap models reach for known or trivial arguments and make signature
  errors; reasoning models must be capped (`low`) with ≥ 30k output tokens or they emit nothing;
  a strict output-format rule fixes some models, not others (E7a/E7b/E18).  The one-line
  search-mode recipe lifts any candidate to 0.26 with verified credit (E20); the LLM's job is then
  to add inequalities the LP does not have.
* **Cost.** $3.31 of $16 spent in total: 8 single-response probes ($0.82), two 15-iteration luna
  runs ($0.125 + $0.105) and one 20-iteration claude-sonnet-5 run ($2.24, $0.11/iteration).  A
  1,000-iteration luna run ≈ $8; sonnet-5 at low reasoning ≈ $110.
* **What the loop does with the recipe (E21/E22).** Both models adopt the one-line search mode at
  iteration 1 (0.2594) and then plateau: luna only tweaks the search budget; Sonnet tries real
  new prunes (aggregated Argument D, pair floor/cap) with Lean proofs, but every idea lies inside
  the LP's constraint system, so no verified gain beyond the schema floor in 20 iterations.
  Verified credit above 0.26 requires inequalities the LP does not have (triples, residues).
* **Transfer matrix (E23).** On enumerated cases Argument D *is* the proved library (20–65 % of the
  work per table); Argument A never fires (the enumerator applies it), deletion/waterfill never
  fire with cited minors, DGH fires only on the wide cell (10,23).

## 4. Open items (design §12 M6–M11)

0. Housekeeping: the promotion ledger holds one redundant entry (`E_c285948723e7`, the library itself,
   promoted from the stub run); harmless but it makes the transfer matrix's `evolved` column duplicate
   the union and should be pruned or replaced by the first genuinely new prune.

1. Lemma library for the next verified gains: triple-codegree and residue (dfield P14)
   constraints as Lean-proved base inequalities for the Farkas schema; then re-run E22.
2. Pure-mode Tier-1 closures for the DGH cells (enumerator cost in the kernel: measure).
3. Residue schema (`Prune.ofResidue`, dfield P14) — the argument that closed (9,23,104) where the LP could not.
4. Tier-0 seam: `Encode.lean` completeness + Lean-side LRAT checking.
5. Real runs (M9) on a larger budget / AlphaEvolve Cloud: verified prunes with `G_target > 0` on
   (9,23,104) and (12,18,109); ablations (no schema channel, no auto-fill, novelty axis).
6. Human review of `Closure.lean`/`Schemas.lean` statements (the trusted base) and of the
   provenance policy for conditional ledger rows.
7. Difficulty: a 20M-conflict pass on the 191 cases open at 2M and a sample of the 2,468 target survivors
   tied at the 2M ceiling, then refit `hardness_model.json` so the model can rank above 2M
   (`E24_difficulty/eval_collect.py → eval_analyze.py → eval_final.py`); the w = z battery tables still carry
   legacy labels (not scored).
8. Reward: GEN carries no weight in v3 (every GEN case ≤ 1,080 conflicts) — add a GEN cell with hard cases
   ((10,10;4,4) w=75 or (11,11;4,4) w=87); C5 (general rules keep ≥ 80 % of their v2 uplift) is 72 % — their v2
   credit came from easy cases and inflated labels, so this is judged acceptable, but it is a choice.

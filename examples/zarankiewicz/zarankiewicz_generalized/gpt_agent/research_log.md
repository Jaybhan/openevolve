# Research log — Zarankiewicz z(m,n;s,t)

Append-only. Newest entries at the bottom. Every entry: date, what was done,
what was learned, provenance (where the idea came from).

---

## 2026-07-28 — Entry 0: program start, scope, prior state

**Scope set by project owner**: full generality (all m, n, s, t), not just the
proven-exact (3,3) suite the OpenEvolve run targets. Goal: general construction
+ exact-formula pursuit. Ambition calibrated by the (verified) July 2026
Jacobian-conjecture counterexample found by Levent Alpöge with Claude Fable 5 —
checked via web: Tao's digestion post, Secret Blogging Seminar, John D. Cook all
independently confirm. The claimed GPT-5.6 cycle-double-cover proof is
credible-but-unconfirmed (self-published PDF + arXiv exposition; one outside
mathematician quoted; no independent verification wave yet). Lesson adopted:
the bar is *explicit and checkable in an afternoon*.

**Prior state of the experiment** (untouched by this program):
- 150 iterations, 15 checkpoints. Champion `cd6af8a9` (iter 134):
  exact_count 110/161, combined_score 0.7758 (= current `.n_sota`).
  NOTE: score function changed across the run's history; recorded metrics are
  not comparable across time (owner-confirmed).
- Evaluator: 161 proven-exact z(m,n;3,3) cells, m=3..16; per-instance 25ms CPU
  budget (anti-search); holdout of 40 cells (every 4th) never shown to the LLM;
  area-weighted scoring, 0.5·is_exact + 0.5·ratio per valid cell.
- `instance_log.jsonl`: 142 full evaluations with per-cell results incl. hidden
  cells — mining gold.

**Plan**: 4 parallel research agents (miner / theory / construction engineer /
bounds prover) + my own synthesis. Inter-agent relay through me. Everything
lands in this directory; labels per README honesty bar.

---
## 2026-07-28 — Entry 1: agents launched; first structural results (coordinator)

**Launched** 4 background agents: miner (evolved-population forensics),
theorist (literature + novelty referee), engineer (general construct(m,n,s,t)),
bounds prover (UB machinery + certification). Relayed early findings to all.

**Finding 1 (quick scan, analysis/quick_scan.md)**: 78/161 proven cells are
counting-tight (z = integer waterfill bound); deficit spectrum tiny and
structured: d∈{1..5} plus single d=8 at (16,16); rows m≤5 tight everywhere;
Culík regime verified on all 41 applicable cells. → Formula shape:
z = WF − d with d local. Conjectures C1–C5 opened (analysis/conjectures.md).

**Finding 2 (champion decode, analysis/champion_analysis.md)**: the evolved
champion is genuinely algebraic — 2-fold triple packings from F₂⁴ affine
hyperplane families; z(16,16)=128 attained by both sides of 8 hyperplanes
with CAP normals in PG(3,2) (≤2 coverage because caps meet lines ≤2). I
re-derived and re-verified independently: 128 edges, 0 violations, 8-regular.
Reed–Muller RM(1,4) description noted. Codes→Zarankiewicz generalization
lever formulated (even-zero-sum-free row sets ⊂ F₂ᵏ); novelty under review.

**Next**: agents' returns → C3 measurement (local-budget bound vs deficit),
engine benchmark, novelty verdicts. My thread: realizability proof attempt
for C1 (m≤5 counting-tight for all n).

## 2026-07-28 — Entry 2: first proven theorems (coordinator)

**Theorem 1 [PROVEN, machine-verified]**: z(m,n;3,3) = WF(m,n) for all m≤5,
all n, with explicit closed forms (analysis/theorems.md). Proof: counting UB +
point-complement/triple-multiset realization schema whose capacity bookkeeping
matches waterfill costs exactly. 60/161 table cells closed by theorem.

**Theorem 2 [PROVEN, machine-verified]**: row-6 deficits ARE Turán's theorem.
Weight-4 blocks on 6 rows = graph-edge complements; legality ⟺ triangle-free;
WF(6,10) demands 10 > ex(6,K₃) = 9 edges → d(6,10)=1. (6,7),(6,8) profiles
killed by pair-degree counts (brute-force confirmed). The (6,9)=36 extremal
matrix IS Turán's K_{3,3} in edge-complement form. verify_theorems.py: ALL
CHECKS PASS (incl. exhaustive infeasibility of profiles (5,5,4⁵) and (5,4⁷)).

**Observation 3**: (16,16): z=128=16·8 with 16 columns = 2·capmax(PG(3,2));
the deficit-8 corner is "WF demand exceeds max-cap supply" — same shape as
row 6's "WF demand exceeds Turán supply".

**Emerging law (→C5 refinement)**: every deficit cell so far = demand-vs-
extremal-supply gap for a classical object (Turán graphs, caps/ovoids, Fano).
z(m,n;3,3) is looking like a *reduction functor* from Zarankiewicz to small
extremal problems. Relayed to bounds+engineer agents with concrete follow-ups
(finish (7,7) exhaustion; test law on rows 7..15; engine targets corrected
profiles).

## 2026-07-28 — Entry 3: Theorem 3 — z(7,20)=75 rederived from scratch

Computed by exhaustive branch-and-bound: **max 2-fold 3-packing by quadruples
on 7 points = 15** (distinct AND multiset variants; the fractional/design
bound was 17.5). Profile elimination by integer arithmetic shows (4¹⁶,3⁴) is
the unique E=76 profile at (7,20) → 16 quads needed > 15 supply → z ≤ 75;
15-packing + 5 triple-columns realizes 75. **First published value derived
fully independently by this workspace.** Search script:

```python
# max #4-subsets of [7] with every 3-subset covered <= 2 times
from itertools import combinations
quads = list(combinations(range(7), 4))
tris_of = {q: list(combinations(q, 3)) for q in quads}
cap = {t: 2 for t in combinations(range(7), 3)}
best = [0]
def dfs(i, count):
    best[0] = max(best[0], count)
    if i == len(quads) or count + (len(quads)-i) <= best[0]: return
    if count + sum(cap.values())//4 <= best[0]: return
    q, ts = quads[i], tris_of[q]
    if all(cap[t] >= 1 for t in ts):
        for t in ts: cap[t] -= 1
        dfs(i+1, count+1)
        for t in ts: cap[t] += 1
    dfs(i+1, count)
dfs(0, 0)  # -> best[0] == 15
```

**Dictionary insight**: profile-uniqueness + published z-values ⟹ packing
numbers, and vice versa — the z-table is an oracle for multi-fold packing
numbers (which ones are known? theory agent tasked). The near-diagonal (3,3)
table looks increasingly like a table of mixed-block-size packing numbers.

## 2026-07-28 — Entry 4: full mining pass complete (miner)

Full forensic pass over all 138 distinct program ids across checkpoints
10-150 (112 distinct code bodies), `instance_log.jsonl` (142 lines, 0
malformed — confirms Entry 0's count), and the archived pre-history runs.
Deliverables in `mining/`: `mining_report.md` (taxonomy + champion decode +
evolution narrative), `extracted/01..08_*.py` (8 mathematically distinct
programs, verbatim, with provenance headers), `cell_reachability.csv`
(per-cell exact_value/best_valid_edges_ever/ever_exact/num_evals, all 161
cells incl. holdout), `ideas_for_generalization.md`.

**Consistent with `analysis/champion_analysis.md`** on the champion's
mechanism (F₂⁴ hyperplane characters M=8..15, cap-in-PG(3,2) doubling at
M=16, 2-fold triple-packing frame) and on its self-duality identity
`z(m,n;s,t)=z(n,m;t,s)` (matches README.md's Conventions independently).
Three findings additive to that analysis:

1. **Quantified hardcoding, by ablation, not just description.** Stripped the
   champion's own M≤6/M=7 tables (nothing else changed) and re-scored:
   110/161 → 74/161 exact (0.7758 → 0.6891). 36 of the champion's 110 exact
   cells (33%) are M∈{5,6,7} and depend entirely on those hand-built designs;
   M=3,4's tables are provably redundant (same result with or without); the
   "genuine" F₂⁴-rule part is 100% accurate at M=13-16 but only 14-44%
   accurate at M=9-11 — a real structural weak zone, not a hardcoding
   artifact. Full per-row table in mining_report.md §2.2.
2. **The champion isn't even the best union of what this run ever knew.**
   `instance_log.jsonl` shows (8,8) and (8,9) were each solved exactly
   exactly once, ever — by `fef84e2f` (108/161, same lineage, an extra
   "saturation sweep" completion step), timestamp-matched to 1ms. The
   champion solves neither. 49 of 161 cells (listed in mining_report.md §2.4
   / cell_reachability.csv) were *never* solved exactly by anything, all
   M∈{7..12}; the two hardest by far are the two `_EXTRA_EXACT` cells
   (11,21) and (12,22), gap 8 and 12 edges respectively.
3. **Full taxonomy, not just the champion's lineage.** 5 distinct
   mathematical families across the run (cyclic difference family = the
   seed; F₂⁴/AG(4,2) hyperplane characters = champion's family, 69/72 = 96%
   of the final population; Kollár–Rónyai–Szabó norm graphs + doubled
   inversive planes, capped ≈0.61 combined_score — matches
   `config_phase_3.yaml`'s own claim exactly; counting-bound water-filling
   with no algebra at all, the only mechanism found already honestly
   (s,t)-general; Brown's graph, one occurrence only, also matching
   `config_phase_3.yaml`'s claim). Every one of the seed's 7 sampled
   first-generation children abandoned the difference-family principle
   immediately (6 became norm-graph attempts, 1 an inversive-plane attempt);
   it was never revisited. `config_phase_3.yaml`'s 50-iteration
   anti-monoculture push (exploitation_ratio 0.35→0.2, migration bug fix)
   grew the dominant family from 106→110 exact but did not displace it —
   population share went 36/67 (54%, checkpoint 100) → 69/72 (96%,
   checkpoint 150).

**For the engineer's generalization levers (§ above)**: `ideas_for_
generalization.md` flags that the mined norm-graph programs all hardwire a
*quadratic* norm form (i.e. a t=3-only tool), so the general-t KRS
construction is worth building from the literature directly rather than
generalizing these programs; and that the cyclic-difference-family principle
generalizes soundly to any s≥2 (pairwise bound ⟹ s-wise bound by set
inclusion) but gets loose fast as s grows past 2, so it's a valid-but-weak
fallback, not a primary mechanism.

Score-function-drift note (Entry 0): confirmed concretely, not just
by-assertion — re-ran the *current* evaluator.py in-process on 5 programs
spanning checkpoint 10-100 and reproduced their recorded exact_count/
combined_score exactly, so metrics are internally consistent within
checkpoints 10-150. The real drift is against the pre-checkpoint-10 archived
runs (run1/run2/run3_archive), whose own `best_program_info.json` files carry
a visibly different metrics schema (missing `exact_count` and/or
`peak_instance_ms` entirely; one, run2, reports combined_score=0.89, higher
than anything reachable in the current schema) — those numbers are quoted in
mining_report.md only as qualitative color, never compared numerically.

## 2026-07-28 — Entry (bounds prover): UB machinery, ground truth, SUPPLY LAW

Deliverables in `bounds/`: `upper_bounds.py` (+`--selftest/--tables/--refine`),
`ub_33.csv` (531 cells, m=3..20, n=m..40), `exact_small.py` + `exact_small.csv`
(+`exact_small_witnesses.json`), `supply_law.py`, `deficit_analysis.md`,
`certification.md`. All CSVs regenerable by the named scripts.

1. **Waterfill greedy PROVEN optimal** for the budget relaxation — exchange
   argument + exhaustive unit test (all k,cap≤7, r≤4, every budget, vs full
   profile enumeration; 5,644 cases). Independent recount agrees with the
   coordinator scan: 78/161 d=0, spectrum {1:28, 2:24, 3:15, 4:10, 5:5, 8:1}.
2. **New machine-proven ground truth** (complete DFS over column multisets,
   witnesses re-verified vs evaluator.has_kst; 0 mismatches vs table):
   all m,n≤6 for (s,t)∈{(2,2),(2,3),(3,3)}, m,n≤5 for (3,4),(4,4); plus (3,3)
   cells (6,7..14),(7,7..12),(8,8..10),(9,9),(9,10). Coordinator Thm 2 UBs now
   exhaustion-verified; (7,7): z=33 (d=2) proven. Counting-first-failures:
   (3,3)→(6,7); (2,3)→(6,6) [z=21<22]; (2,2)→diagonal (8,8) [24<25, (7,7)=Fano
   tight]; (3,4),(4,4): none in box.
3. **q-local refinement (PROVEN necessary conditions on degree profiles)**
   closes only 16 of 194 deficit units (16/83 cells, 1 each; only (6,10)
   fully). Deficit is design-realizability, invisible at profile level —
   confirms Thm 2's flavor is necessary, not an artifact.
4. **SUPPLY LAW (central)**: define g4(m) = max multiset of 4-subsets of [m]
   with every triple ≤2. Computed EXACTLY: g4(6)=9 (=ex(6,K₃)), g4(7)=15
   (beats doubled Fano 14; pair bound is 17), g4(8)=28 (=pair bound; doubled
   SQS(8)). In the no-5-column waterfill regime:
   z = 3n + min(k4wf, g4(m)), d = max(0, k4wf−g4(m)).
   **22/22 PASS on all applicable known cells** (whole tails of rows 6,7,
   reproducing the exact deficit patterns). Predictions: z(7,n)=3n+min(...,15)
   ∀n≥17; z(8,n)=3n+min(...,28) ∀n≥27, i.e. row-8 deficit dies at n=27
   (≪ Culík 112); z(8,28)=112=4·28 attained by doubled SQS(8) itself.
5. **99 beyond-table certified-exact cells** (m≤20, n≤40): 52 Culík-regime +
   48 supply-construction (m=6: n=24..40, m=7: n=24..40, m=8: n=27..40, minus
   overlap (6,40)). Supply cells are BELOW the Culík threshold — mechanism is
   ours (counting UB + g4-family+pads LB). Novelty vs Guy'69/Roman bound
   tables NOT yet checked — theory agent tasked before any claim.
6. **Published-UB region is pure counting**: on all 45 cells beyond
   _EXACT_UP_TO, published UB == waterfill exactly. No better bound is hiding
   in the literature figures there; closing those cells is purely a
   construction/refutation problem.

Open to team: g4(9) exhaustion started, stopped incomplete at session end
(restartable via supply_law.g4; 3-(9,4,2) design nonexistent by
divisibility, so g4(9)<=41<42=pair bound); mixed supplies g_{5,4}(m) are the next
computation to attack rows 7..15 mid-range; (16,16) d=8 = cap-saturation
statement still needs a matching UB proof (WF−8 unproven for n=16 frontier).

## 2026-07-28 — Entry 4: miner integrated; row 6 fully reproven (coordinator)

**Miner returned** (mining/ complete). Headlines I'm acting on:
- Ablation: champion = 74/161 without its hand-built M≤6/M=7 tables (33% of
  exactness was per-M craftsmanship). champion_analysis.md corrected.
- 49 cells never solved exactly by ANY evolved program: all in M=7..12;
  M=7 fails at every N≥15; worst row M=8 (0/16 for the rule-like part);
  hardest overall: (11,21), (12,22) — the _EXTRA_EXACT cells.
- fef84e2f (only-ever solver of (8,8),(8,9)): adds capacity-CERTIFIED
  saturation sweeps (grow columns while the triple-capacity certificate
  proves safety, degree-ordered, 2 deterministic sweeps) → engine should
  adopt as completion upgrade.
- Self-duality trap for the general engine: z(m,n;s,t) = z(n,m;t,s) — s,t
  swap WITH m,n. (Already in engineer's brief; reinforced.)

**Row 6 = 18/18 reproven** by v3 exact solver (complete heavy-family
enumeration + quad B&B + closed-form light layer) in <1s. Row 7 running.
NOTE: my z(7,20)=75 witness (Thm 3) already solves a cell no evolved program
ever solved; the row-7 solver output will yield witnesses for all nine
never-solved (7,N≥15) cells.

## 2026-07-28 — Entry 5: rows 6-8 independently reproven; ILP machine online

**Row 7 = 17/17 reproven** by v3 exact solver (109s; v1's aggregation bug —
distinct capacity vectors collapsed under a shared summary key — found via the
(7,10) mismatch 43 vs 44, fixed in exact_row_solver3.py).

**ILP machine** (scipy/HiGHS in scratchpad venv, analysis/ilp_solver.py):
block-TYPE multiplicity formulation (quotients column symmetry; mult ≤ 2 for
weight ≥ 3, pads free), solver blind to published values, witnesses re-verified
independently. **Row 8 = 16/16 match in ~2 min** — the row the evolutionary
run never cracked (14/16 never-solved cells) falls instantly to exact
optimization. Rows 9-16 running in background (includes (11,21),(12,22)).

**Tally: 95→111 of 161 cells now independently verified** (Thm 1: 60; rows
6,7 enumeration: 35; row 8 ILP: 16). Witness files: analysis/witnesses/.
Bounds agent redirected: ILP covers per-cell verification; they focus on the
refined analytic bound (the WHY of the deficit) + general (s,t).

## 2026-07-28 — Entry 6: theory agent integrated — novelty map redrawn

Theory agent's referee verdicts (theory/novelty_checklist.md, 52/52 numeric
checks pass) force honesty corrections and hand us the real targets:

**Rediscoveries (label, cite, do not claim)**: my WF bound == Roman 1975
(equal on all 161 cells); Thm 1's row 3-5 formulas == Roman equality window
(71 cells incl. row 6 n≥13); local-budget hierarchy == substantively DGH
2024; (16,16) cap witness isomorphic to Tan's published witness; T₃,₃(7)=15
== Tan Table 1 (my B&B = independent confirmation; theirs timed out, mine
completed). Row-6 Turán bridge: sound, but folklore-risk — Guy 1969
unverifiable here.

**Provenance shocks**: the evaluator's table IS Tan 2022 Table 3 (SAT+DRAT);
the _EXTRA_EXACT cells come from the project owner's own paper (BNL,
arXiv:2605.01120) — which ALSO proved (11,22)=121, absent from the evaluator
suite (owner should know). CRWR 2016 claims z(12,17)=103 exact + sharper UBs
that Tan/DGH/BNL never engaged — live discrepancy, cheap real contribution.

**The flagship (open per referee)**: below-window exact determination for
s=3 (CHM did s=2). My machinery is exactly this. Identified GAP BANDS between
Tan's frontier (n≤23) and Roman windows (n ≥ B−3T₃,₃(m)): (7,24),
(9,42..47), (11,82..89), (12,110..115), (15,228..234) — genuinely
undetermined cells. Frontier ILP launched on all of them + (12,17) CRWR +
(11,22) BNL verification. Candidate closed form being tested:
z = 3n + min(T₃,₃(m), n, ⌊(B−n)/3⌋) above a threshold n₀(m) [n₀(6)=8,
n₀(7)=14 verified]; the m=8 case shows weight-5 blocks re-enter below
n₀ — the mixed 5/4 supply function is the remaining wall.

## 2026-07-28 — Entry 7: ROW 8 COMPLETE — first new-territory row determinations

**Lemma A proven** (analysis/master_formula.md): the two-resource profile
algebra yields Roman's bound as a corollary AND a per-heavy-block penalty
(each weight-5 block lowers the ceiling by ≥1, weight-6 by ≥10/3, weight-7 by
≥22/3) — so Roman-window optima are quad+triple only, and the below-window
regime is governed by the joint supply function S_m(k₅,k₆,k₇) := max quads
coexisting with a heavy profile (S_m(0) = Tan's T₃,₃(m); higher slices appear
to be new objects — referee N6a).

**NEW EXACT VALUES (ILP, witnesses verified, solver blind to predictions):**
- z(7,24) = 87 [= 3n + T₃,₃(7); with human-readable proof]. Row 7 complete.
- z(8,24) = 97 [witness: one pentad + 23 quads — S_8(1) ≥ 23],
  z(8,25) = 100, z(8,26) = 104, z(8,27) = 108 [pure quad packings ⊂ doubled
  SQS(8)]. **Row 8 of z(m,n;3,3) is now determined for every n** (Tan n≤23 +
  these + Roman n≥28 + Culík n≥112): the first complete row that required
  new-territory determinations. The witness profiles match the master
  formula's regime predictions cell-for-cell.
- z(9,23) = 103 [11 pentads + 12 quads — deep column-bound regime]. Row-9
  band (23..47) grinding; rows 10, 11 queued in the same run.

Nothing here contradicts any published value; all new cells lie strictly
between Tan's frontier and the Roman windows, where no published theorem or
computation reaches (per theory agent's referee review). CRWR (12,17) run
in progress.

## 2026-07-28 — Entry 8: supply function S₈ + row-9 heavy-layer geometry

- S₈(0)=28, S₈(1)=23, S₈(2)=21 [exact, constrained ILP]. First pentad costs
  FIVE quads, second costs two — nonconvex supply decay. S₈(3) computing.
  With Lemma A, S₈(≤3) converts Theorem 6's four new row-8 cells from
  "exact by ILP" to readable proofs.
- Row-9 optimal witnesses (n=9..13) all carry a heavy-6 layer of exactly
  four 6-blocks whose complementary triples pairwise intersect in exactly
  one point (4-line partial linear space on 9 points); the 5-layer grows
  with n while the 6-layer stays pinned at 4. Another supply-slice question:
  is 4 = max #6-blocks under 2-fold capacity on 9 points with room left?
  (S₉ 6-slice for the bounds agent's table.)
- Theorems 5 & 6 (complete rows 7 and 8) drafted in theorems.md.

## 2026-07-28 — Entry 9: master formula validated; z(11,22)=121 verified; campaign retooled

- **z(11,22) = 121 independently confirmed by ILP in 2s** (witness verified)
  — validates BNL's proof of the cell missing from the evaluator suite.
- **Master formula validated**: rows 6 and 7 reproduce EXACTLY from the
  supply tables by pure arithmetic (formula_from_supply.py — 33 cells, no
  per-cell solving). Row 8 mismatches are grid truncation (need k₅=9,10 and
  k₆=1 slices — witnesses confirm), plus a reverse-dictionary squeeze:
  z(8,22)=90 forces S₈(3) ≤ 18, so S₈(3) ∈ [15,18].
- Supply tables so far: S₆ = (9,6,4,∅); S₇ = (15,12,10,8,6,∅) — the ∅s are
  PROVEN infeasibilities matching Thm 2-style graph arguments;
  S₈ = (28,23,21,?,15,14,10,8,6).
- Symmetric mid-band cells ((9,15),(9,24),(12,17)) resist vanilla HiGHS
  (LP↔IP gap ≈ supply deficit — the LP literally cannot see design
  obstructions; the observation itself is a nice methodological datum).
  Solver upgraded with monotone-row-degree symmetry breaking; campaigns
  relaunched: S₉ grid + S₈ extension + gap bands descending (easy end
  first). CRWR (12,17) parked for a supply-cut-strengthened round.

## 2026-07-28 — Entry 10: S₈ grid complete — Theorem 6 fully analytic

Symmetry-broken solver cracked S₈(3) = 17 (140s; inside the [15,18]
dictionary squeeze — the reverse inference from z(8,22) was correct).
Full S₈ grid (k₅ ≤ 11, k₆ ≤ 2) computed, incl. infeasibility frontier
(e.g. 11 pentads on 8 points impossible; 8 pentads + 1 hexad impossible).

**Master formula revalidated: row 8 ALL MATCH (18 cells n≥10).** Rows 6, 7, 8
= 51 cells now reproduce from supply tables by pure arithmetic. Theorem 6's
four new cells (8,24..27) now have fully analytic proofs (Lemma A + S₈ grid
+ explicit witnesses) — no ILP left in the proof chain. The z ↔ S dictionary
is exact on three complete rows.

S₈ structure notes for the design-theory write-up: S₈(k₅=0..10) =
28,23,21,17,15,14,10,8,6,2,0 — the big first drop (−5) then plateau-ish
decay, k₆ slices shifted down by 7/6/6; infeasibility kicks in exactly when
the pentad pair-complement graph analysis predicts.

## 2026-07-28 — Entry (bounds prover, addendum): Lemma A verified; S-table; rows 6-7 closed analytically

- Lemma A machine-verified (102,452-profile integer sweep, 0 violations);
  generalized penalty c_w = (C(w,3)−1−3(w−3))/3 nondecreasing — the k7 term
  soundly covers all weights ≥ 7. `bounds/supply_table.py`.
- S_m(k5,k6) tabulated by exact ILP (HiGHS, venv) → `bounds/supply_table.csv`
  (152 entries; UNRESOLVED entries time-limited and only ever used as
  monotone brackets). Cross-checked against complete-search g4 (m=6,7,8) and
  DFS spot checks — all agree. New PROVEN values: S_7(1)=12 (coordinator's
  ≤12 is sharp), S_8(1)=23 (their pentad witness optimal), g4(9)=40,
  g4(10)=60 = pair bound = 2|SQS(10)|.
- Row closure with min(WF, LemmaA+supply): row 6 = 18/18 EXACT, row 7 =
  16/17 EXACT (only (7,7) open by 1 — its second deficit unit is not
  block-profile-expressible), plus (8,20). Rows 8-10 mid-table are below the
  Roman window — different machinery needed (level-profile × high-weight
  supply). Beyond-table: (7,20)=75, (7,24)=87, (8,27)=108, (8,28)=112
  analytic; (8,24) at 98 vs ILP 97 pending S_8(3),S_8(4) ∈ [14,21].
- DGH-LP comparison NOT attempted here: I do not have DGH's exact LP
  formulation in-workspace; theory agent should supply it, then the cells
  above are the comparison set (our numbers are on record).

## 2026-07-28 — Entry 11: bounds agent landed; (8,24) reconciled; 99 certified cells

Bounds agent final (bounds/): waterfilling optimality UNIT-PROVEN (5,644-case
brute force); Lemma A independently machine-verified (102,452 profiles, 0
violations) with generalized weight≥7 penalties; 140-cell exact_small ground
truth (0 mismatches vs published, 31 overlaps); supply law independently
formulated and 22/22-verified; **99 certified-exact cells beyond the
published 161** (52 Culík + 48 supply-construction, m=6/7 n≥24, m=8 n≥27);
key negative result: q-local degree-profile refinements (DGH-style) close
only 16/194 deficit units — **the deficit is block-realizability, not
degree-profile-visible** (this cleanly separates our supply-function method
from DGH's LP and is the mathematical reason their bound family cannot reach
these cells).

RECONCILIATION: their (8,24) analytic UB of 98 used S₈(3),S₈(4) brackets;
with the completed grid (S₈(3)=17, S₈(4)=15) every Q=26 branch dies
(k₅=1..5 exceed supply, k₅≥6 dies by Lemma A) ⇒ z(8,24) ≤ 97 ANALYTIC,
agreeing with the ILP + witness. Cross-agent agreement everywhere else:
their S₆/S₇/S₈ sharp values match my grid; their g4 = my S_m(0) = Tan's T₃,₃.

Also from their deficit analysis: the 45 published-UB-only cells (beyond
exactness limits) all equal WF exactly — so ANY construction attaining one
certifies a new exact cell; target list recorded in bounds/deficit_analysis.

## 2026-07-28 — Entry 12: ENGINEER LANDED — 159/161 exact, 0.98958 verified

Construction engine final (constructions/zarankiewicz.py, ~2300 lines):
**159/161 exact on the suite, 0 invalid, combined_score 0.98958 — verified
by my own independent rerun of the snapshot evaluator (80s, per-instance
peak 0.009 ms vs 25 ms budget). Live experiment untouched (.n_sota still
0.7758).** Versus the evolutionary run: champion 110 exact / 0.7758; the
engine closes 47 of the 49 never-solved cells.

Interpretability yields (constructions/report.md):
- z(12,22)=132 (owner's paper, SAT witness) is structurally a Hadamard
  3-(12,6,2) design — the engine generates it from the design, giving the
  cell a mathematical explanation rather than a certificate-only status.
- z(11,21)=116 generated by a QR(11)∪{0} cyclic difference family.
- New generative families distilled-then-rederived: bipolar and twin-planes
  (closed the whole 8×17..8×23 band), pair-GDD (z(11,16)=92), Hadamard
  residuals (m=9..11 elongated band).
- Engineer's verifier cross-check caught a Python bit_count pitfall
  ((-1).bit_count()==1) — all results doubly verified.

Remaining: (9,22) 99/100, (10,15) 80/81 — k5-decomposed ILP launched on
both + the row-9 quartet + (12,17) CRWR. On completion: final assembly.

## 2026-07-28 — Entry 13: all agents landed; endgame campaigns

All four agents complete. Cross-validated everywhere they overlap (S-tables,
deficits, supply law, witnesses). FINDINGS.md carries the consolidated
state. In flight at entry time: k5-split ILP on (9,22)/(10,15)/(12,17)/row-9
quartet; S₉ grid k₅≥5; row 10-11 gap bands descending. Remaining milestones:
161/161 via a saturated-mixed-packing family; complete-row theorems 9-10;
CRWR discrepancy resolution.

## 2026-07-28 — Entry 14: unification pivot — z(9,22) witness; tower theorem

Owner re-emphasized the unified-construct goal. Actions:
- **z(9,22)=100 FOUND** (k5-split ILP, k₅=12 slice): profile 5¹²4¹⁰,
  near-11-regular. NOT a Hadamard-residual truncation (that caps at 99 —
  proven by the design's intersection numbers). New structure; engineer
  resumed to distill it + build constructions/unified.py.
- **Tower test (negative theorem)**: at (8,19), quads restricted to the
  doubled SQS(8) reach only 79 < 81. Combined with S₈(0)→S₈(1) = 28→23 and
  the (9,22) fact: extremal structures phase-transition in n; NO single
  nested master family per row exists. Unification must be (and is) the
  two-layer form: one profile rule (Lemma A + S_m tables) + one realization
  principle (G-symmetric maximal packings from a finite algebraic source
  hierarchy). Formalized in analysis/unification.md.
- Band progress: z(10,21)=106, z(10,22)=110 new; row-9 quartet LBs
  106/109/112/116+; (10,15) resists everything (honest open cell).

## 2026-07-28 — Entry 15: THEOREM 7 — the unified upper-region formula

Owner directive: unifying theorem for all m,n, not more witnesses. Delivered
the centerpiece (theorems.md Theorem 7):

  z(m,n;3,3) = 3n + max_k min( 2k + S_m(k), n + k, ⌊(B−n)/3⌋ − k ),
  for ν(m) ≤ n ≤ B;  = 2n + B beyond (Culík).

One formula: contains Roman's window (k=0, slot term), the supply-capped
band (k=0, supply term), the pentad-corrected boundary (k ≥ 1 — new), and
Culík. Verified 45/50 on all known+new cells of rows 6-9; the 5 open are
pending S₉(k≥5) slices (witnesses already confirm the predicted splits).
PROVEN for m = 6, 7, 8. Reduces the whole upper region for every m to the
finite design function S_m(·). Lemma B (hexad exchange) recorded toward a
general ν(m).

**δ-conjecture (design-theoretic core, NEW)**: T₃,₃(m) = C(m,3)/2 iff
3-(m,4,2) design admissibility holds [verified m=4..17]; otherwise
T₃,₃(m) = ⌊C(m,3)/2⌋ − 2 for m ≥ 7 (δ(6)=1) [verified at 7,9,11,12,15].
Predicts T₃,₃(18) = 406. **ANOMALY: recorded Tan value 408 violates design
admissibility (r non-integral ⇒ ≤ 407) — theory agent re-verifying the
published table; possible erratum-grade correction.**

## 2026-07-28 — Entry 16: δ-conjecture refuted; C7 Johnson-form law born

Theory agent verdict on the T₃,₃(18) anomaly: OUR transcription error (Tan
prints 405, not 408). My impossibility argument was sound (≤407) but Johnson
is sharper (≤405) and Tan's dihedral-orbit construction attains it —
re-verified solver-free by the agent (59/59 checks pass). Consequences:

- δ-conjecture REFUTED at its first untested case (predicted 406 ≠ 405).
  Recorded in conjectures.md C6 per protocol. The refutation exposed our
  own data error — the theorem-first method self-corrected the workspace.
- NEW C7 (Johnson-form law): T₃,₃(m) = J(m) − 2·[m ≡ 3 mod 4 ∧ m ≢ 0 mod 3],
  J(m) = ⌊m⌊(m−1)(m−2)/3⌋/4⌋. Fits ALL m=3..18. Theory agent confirms no
  published determination of D₂(v,4,3) exists → genuinely new falsifiable
  conjecture (Bao–Ji genre, λ=2 analog). Predicts T(19)=482, T(21)=661,
  T(22)=770. ILP test of T(19) launched (2400s).
- Hanani citation installed for the perfect⟺admissible law (classical).
- Theorem 7 unaffected structurally; S_m(0) inputs corrected (405 at m=18).

## 2026-07-28 — Entry 17: THEOREM 7 VERIFIED ON FOUR COMPLETE ROWS (50/50)

S₉ slices landed (S₉(6)=25, S₉(7)=23, S₉(8)=20). The six remaining
ambiguities were my over-generous brackets; the z-data squeeze pins
S₉(2)∈[33,35], S₉(5)∈[25,28], S₉(9)≤18, and the unified formula
reproduces ALL 19 row-9 cells at BOTH bracket endpoints. Combined with
rows 6-8: **Theorem 7 verified 50/50 on every known + newly-determined
cell in its domain, four complete rows, insensitive to residual S
uncertainty.** The backward dictionary (z ⇒ S brackets) is itself a
device worth writing up: exact values of the harder object (z) pin the
cleaner object (S).

Running: T₃,₃(19) ILP (C7's first out-of-sample test), (12,17) CRWR
k5-split, row-11 band (mostly timing out at 600s — honest UNRESOLVED),
engineer's unified.py.

## 2026-07-28 — Entry 18: ENGINE FINAL 160/161 (0.995503); unification priced

Engineer's final: (9,22) = STS(9)/AG(2,3) around a base point — implemented
generatively; router 160/161, snapshot score 0.995503 (coordinator
re-verified; live .n_sota untouched). Sole open cell: (10,15) 80/81.
unified.py delivers the single-rule variant: 127/161 with the price of
unification measured at 63 edges/33 cells — consistent with the tower
negative theorem: the theory unifies (Theorem 7), the generator pays.
(12,17) CRWR: all decomposed ILP slices time out — honest UNRESOLVED,
remains flagged as the literature's open discrepancy. T₃,₃(19) C7 test
still computing.

## 2026-07-28 — Entry 19: T₃,₃(19) first test inconclusive (honest bracket)

C7's first out-of-sample case resisted: 2400s of symmetry-broken HiGHS on
the 3876-type packing ILP yields incumbent 450, so T₃,₃(19) ∈ [450, 484
(Johnson)]. Prediction 482 neither confirmed nor refuted. Next tools:
dihedral/cyclic base-block search (Tan's own method) for attainment;
Johnson-refinement or SAT for the ceiling. Recorded in conjectures.md C7.

## 2026-07-28 — Entry 20: session close-out

Stopped the two campaigns that had gone pure-timeout (row-11 band, deep S₉
slices) — m ≥ 11 monolithic ILP is beyond current tools at these limits;
recorded as the frontier. The k5-split (12,17) sweep finishing on its own.
Final session state is consolidated in FINDINGS.md; the ledger of open
items for a future session: (10,15) 80/81; CRWR (12,17) discrepancy;
T₃,₃(19) ∈ [450,484] vs C7's 482 (needs orbit constructions/SAT); row-10
mid-band exactness (21..25); row ≥ 11 gap bands; ν(m) general proof;
S-tables m ≥ 10; extending Theorem 7 below ν(m) via the (k₆,k₇) grid.
Everything in this directory is additive; the live experiment is untouched.

## 2026-07-28 — Entry 21 (final): last campaign concluded

k5-split sweep finished: (12,17) timed out on every slice — the CRWR
discrepancy remains open as recorded; (9,27) ≥ 116 stands. No background
work remains. Workspace final: 21 log entries, 64 verified witnesses,
Theorems 1–7, conjectures C1–C7 (C6 refuted and kept), engine at 160/161
(0.995503 verified), unified variant priced, live experiment untouched.

## 2026-07-28 — Entry 22 (session 2): LEMMA C + THEOREMS 8-9 — the unified closure

Owner directive: witnesses don't count; a unified theorem does. Delivered:

**Lemma C (mixed-value Johnson, s ≥ 3)**: in any (t−1)-fold s-packing by
blocks of weight ≥ s+1, Σ(w−s) ≤ J_{s,t}(m). Two-line induction on the
per-point inequality s(w−s) ≤ C(w−1,s−1); FAILS for s=2 (structurally —
planes). Machine-verified: (★) for s=3..7; 64 witnesses; 198 z-cells, zero
violations.

**Theorem 8**: for every Johnson-tight m (all 4..18 except {7,11}) and ALL
n ≥ T₃,₃(m): z(m,n;3,3) = 3n + min(T, ⌊(B−n)/3⌋), then Culík. PROVEN
(Lemma C + Roman + sub-packing realization). 93/93 on all data in region.
J(18) = 405 = Tan's corrected value — the Johnson bound EXPLAINS m=18.
New territory: every inadmissible Johnson-tight m (6,9,12,15,18,21,...) —
bands [T, B−3T], infinitely many m, one formula. m=7 proven via S₇ table;
m=11 pending S₁₁(1..3) ILP (running).

**Theorem 9**: general s ≥ 3 exact Pareto form; Q* can exceed T (found at
(3,4) m=5: full-block + point-comps attain J=7 > T=6, z(5,6;3,4)=25 —
ILP-verified; two-branch formula correct at 5/5 other cells). Design-regime
corollary via Hanani/Keevash/GKLO: for every s ≥ 3, t ≥ 2, all large
admissible m, a fully closed-form exact determination on n ≥ (t−1)C(m,s)/(s+1).
Referee checks dispatched (Roman's general scope; Keevash prior art;
Lemma C folklore risk).

## 2026-07-28 — Entry 23: referee round 2 — the corollary deflates, the gate holds

Theorist verdicts on the unified closure: (1) Roman 1975's true scope
UNRESOLVED at the primary source (Tan attributes the general equality
window to him; CHM/DHS/DGH cite him bound-only; paper inaccessible) —
n-varying window treated as Tan-attributed until the paper is obtained.
(2) The design→exactness bridge is PUBLISHED (DHS 2013 Prop 3.25, general
(t,λ+1)); CHM 2024 already run Keevash for s=2. Theorem 9's corollary
reframed: modest synthesis; residual claim = Keevash instantiation for
s ≥ 4 + the n-varying assembly. (3) Lemma C: folklore risk HIGH (CHM Lem
2.4 / DGH Thm 1.1 do per-point mixed-size accounting); linearized form not
found verbatim; s=2 failure noted as the structural remark. NEW GATE
before any novelty claim: (a) does Guy's point-deletion recursion (DHS
Prop 3.20 — which beats Roman at (7,7): ≤33 vs 35) already give our band
UBs (z(7,20)≤75, z(7,24)≤87, row-9 band, row-8 gap cells)? (b) do DGH's
v=1 constraints imply Lemma C at the use sites? Both checks dispatched to
the theorist (they hold the primary texts). Claims ledger frozen until
they land. Post-referee annotations added to theorems.md.

## 2026-07-28 — Entry 24 (final): claims ledger sealed post-implication-checks

Theorist's exact-rational implication checks landed (theory/
implication_checks.{md,py}): (1) Guy/DHS deletion machinery reaches
(7,20) and (6,10) (new proof, not new value) and Roman covers (7,24),
(8,27), (9,46-48); the beyond-machinery kernel of Theorem 8 = the NINE
cells (8,24-26), (9,40-45). (2) Lemma C is NOT a repackaging: it equals
DGH v=1 + one genuine per-point integrality floor, worth exactly +1 at
every interior band cell of rows 9/12; the rows-7/11 T-bounds are beyond
both LPs (the J−2 leave content, ours). FINDINGS.md carries the FINAL
CLAIMS LEDGER. Pending at close: S₁₁(1..3) ILP (m=11 conditionality),
T₃,₃(19), the deep band, s=2, Roman-primary-source retrieval, and human
review of the whole stack.

## 2026-07-28 — Entry 25 (close): S₁₁ slices timed out; m=11 stays conditional

S₁₁(1..3) all hit the 1800s ILP limit (m=11 supply problems join the
m ≥ 11 hardness wall). Theorem 8 at m=11 therefore remains CONDITIONAL on
S₁₁(k) + 2k ≤ 80 for k ≤ 3 — already stated as such in theorems.md and
the claims ledger; every other m in 4..18 is unconditional. No further
background work in flight. Session closed.

## 2026-07-28 — Entry 26 (session 3): LEMMA E + THEOREM F — the class falls

Owner directive: continue toward closure via divergent agent paths. Three
agents launched (design-prover / SAT attack / deep-band Pareto). Meanwhile
the coordinator's own thread delivered the strongest pure-math results of
the project:

**Lemma E [PROVEN, general m]**: for m ≡ 3 (mod 4), m ≢ 0 (mod 3): B = mR
≡ 2 (mod 4), so a Johnson-maximal packing leaves weight exactly 2, but
point-leaves are ≡ 0 (mod 3) — impossible. T₃,₃(m) ≤ J−1 for all 33 class
members ≤ 200 (arithmetic machine-verified), infinitely many m.

**Theorem F [PROVEN, general m]**: the full J−2 law, MIXED configs
included: every heavy config on class m has value Q ≤ J−2. New ingredient:
PAIR-PARITY (quad through a pair uses 2 slots, pentad 3 ⇒ ℓ_xy ≡ p_xy mod
2) + empty finite classification of weight-6 leaves (machine: 0 survivors)
+ a case chain killing k₅ = 1,2,3 and hexads. Consequences:
- C7's upper bound proven for the ENTIRE class; T₃,₃(19) ≤ 482.
- **Theorem 8 now UNCONDITIONAL on all m = 4..18** — m=11's ILP
  dependency eliminated by mathematics.
- Remaining for C7 in full: J−2 attainment constructions (m ≥ 19) — both
  agents redirected onto the 482 hunt at m=19.
Adversarial re-verification of Theorem F assigned to the design prover.

## 2026-07-28 — Entry 27: the doubled-pentagon leave (construction template)

Leave archaeology at m=7: rebuilt a maximum 15-packing, extracted its leave
— weight 10 as Theorem F forces, and structurally it is 2×{complements of
the edges of a 5-cycle} on a 5-point support (all point-leaves 6, all
pair-leaves even; doubling satisfies pair-parity automatically). The
doubled pentagon is the natural J−2 leave of the class. Relayed to the
design prover as (a) the target leave for m=19/23 constructions, (b) the
literature key ("leave of maximum packing" characterizations), (c) a
universality question at m=11. If the doubled-pentagon leave is universal
across the class, C7's full form is: "maximum class-m 2-fold quadruple
packings are exactly the packings with doubled-pentagonal leave."

## 2026-07-28 — Entry 28: T₃,₃(19) = 482, T₃,₃(23) = 883 — C7 proven at 4 members

Design prover's core deliverables landed and coordinator-verified (4th
independent verifier): explicit Z₅-symmetric maximum packings at m = 19
(482 blocks) and m = 23 (883), plus re-attainments at 7 and 11 — ALL with
the identical doubled-pentagon leave, confirming the structural form of C7
constructively. Combined with Theorem F: **first determinations of
D₂(v,4,3) beyond Tan's table — two new packing numbers — and, via
Theorem 8, complete exact wide-regions of z(m,n;3,3) at m = 19 and 23**
(all n ≥ T, with proofs). Theorem F independently re-verified by the
design prover (adversarial check passed). m = 31/35 stretch constructions
still running; SAT and deep-band agents still out.

## 2026-07-28 — Entry 29: row-11 band falls; (10,15) witnessed both ways

SAT campaign returns verified: (10,15)=81 SAT witness + UNSAT at 82 (the
suite's final cell, now two-sided); row-11 band witnesses n=82..89 all
valid and exactly on Theorem 8's formula — combined with Theorem F's UB,
z(11,n) is determined for ALL n ≥ 80. Eight new cells. The witness bank
now covers 161/161 of the suite. Pending: (12,17) fleet verdict (13 cores),
design-prover m=31/35 stretch, deep-band report.

## 2026-07-29 — Entry 30 (session 4): THEOREM 11 — the divisibility unification

Overnight state: (12,17) SAT cubes still alive (3 processes, 8.5 CPU-hours,
5-6 GB DRAT traces each — verdict forming); deepband MILPs and m=31/35
runs died; agents dormant, now resumed.

**Theorem 11 written**: the project's three congruences (slots mod 4,
points mod 3, pairs mod 2) ARE the K₄⁽³⁾-divisibility conditions for
2K_m⁽³⁾ − leave; Lemma E/Theorem F = minimal-obstruction computations;
maximum-packing leaves = minimal divisibility-restoring multigraphs. The
T-spectrum: admissible m → C(m,3)/2 (Hanani, ALL m, effective); class m →
J−2 (Thm F + {7,11,19,23} explicit + GKLO for large m, citation being
pinned); 3|m → J (verified ≤ 18; anchor: m=6's maximum leave IS the
doubled parallel class {015}²{234}², freshly extracted; large-m via same
route once the residue-leave classification is written). Consequence: the
wide region of (3,3) is closed exactly up to the design-existence
frontier, in both directions.

Agents redeployed: design prover → GKLO citation + general C7 write-up +
3|m classification + m=27/31 finite confirmations; deepband → finish the
five z(9,·) placeholders with its frontier-cut machinery.

## 2026-07-29 — Entry 31: T₃,₃(27) = 1458; the spectrum completes its proofs

Design prover final report integrated. T₃,₃(27) = 1458 = J(27) NEW
(order-9 witness, coordinator-verified: leave weight 18 = classification's
prediction, every point-leave ≡ 2 mod 3). 3|m minimal-leave classification
PROVEN (doubled parallel class / doubled hub by residue). GKLO Thm 1.1
citation pinned with line-by-line hypothesis verification; Keevash and
Delcourt–Postle as independent routes. Theorem F survived adversarial
re-verification (subtlety at k₅=1 found and closed; pair parity shown
load-bearing: 750 point-congruence survivors at weight 6, zero after
parity). Literature: D₂(v,4,3) undetermined anywhere; our class result =
PDN₂ = U₂ − 2, apparently new. m=31/35 retry queue running. Still out:
deepband placeaholders (7 MILPs hot), (12,17) fleet.

## 2026-07-29 — Entry 32: the deep-band offensive — ladder + diagonal hunt

Owner directive: close the deep band; new ideas authorized. Opened:
- **Theorem 12 (Johnson ladder) PROVEN**: per-point supporting-line
  inequalities at every weight level (k=4 = Lemma C); the ladder LP tracks
  the entire deep band within ≤ 3.8 on all 57 known cells (waterfill: 8+).
  Weak exactly at the diagonal corner (geometry-bound) — honest division
  of labor established.
- **Circulant diagonal LBs**: z(17,17) ≥ 136 = 8·17 (base
  {0,1,2,3,5,6,11,13} ⊂ Z₁₇), continuing the 8-regular plateau from
  z(15,15), z(16,16); z(19,19) ≥ 152. Window at 17: [136, 150].
- **Diagonal hunter agent launched**: mission z(17,17) EXACT (first
  unknown diagonal value; calibration on z(13,13)=92 — my plain-SAT run
  reproduced the 92-attainment in 520s, UNSAT needs their symmetry
  machinery), then (18,18), (19,19), and the second-order diagonal law.
Fronts live: diagonal hunter, deepband placeholders (7 MILPs), (12,17)
fleet, m=31/35 queue.

## 2026-07-29 — Entry 33: diagonal breakthrough — plateau refuted, quadratic law emerging

Diagonal hunter interim, coordinator-verified: z(17,17) ≥ 138,
z(18,18) ≥ 150, z(19,19) ≥ 164 (witnesses all valid, my own checker). The
8-regular plateau hypothesis is REFUTED. New elementary UB tool: DELETION
AVERAGING (every 16×16 minor K₃₃-free ⇒ 256E ≤ 289·z(16,16) ⇒ z(17,17) ≤
144; validates TIGHT at (15,15)). Current window z(17,17) ∈ [138, 144]
with a live cascade: UNSAT(16,17)@133 ⇒ ≤140; crux (17,17)@139 running
(UNSAT ⇒ exactly 138).

**Candidate diagonal law (falsifiable, window 15 ≤ m ≤ 19):**
z(m,m) = 8m + (m−15)(m−16); increments 8, 10, 12, 14 (second difference
exactly 2); matches exact 120/128, hits the LS frontier at exactly +1 in
all three open sizes. NOTE (mine): a quadratic must break by m ≈ 25-30
(asymptotics are ~½m^{5/3}) — the law is a window law like the plateau
before it, and its breakpoint is itself a discovery target.

## 2026-07-29 — Entry 34: propagation closure; the diagonal cascade plan

Lemma G systematized (deletion averaging as exact-value propagator; KST
device, new use). Full grid closure computed: 29 Theorem-8 confirmations;
diagonal ceilings 144/160/177/194/213/232 (m=17..22); quadratic law's
predictions inside every window. Cascade identified and relayed to the
hunter: each settled diagonal collapses the next window (138 ⇒ (18,18) in
[150,154] ⇒ (19,19) in [164,~167]) — the diagonal may fall like dominoes
once the crux lands. Fronts: crux (17,17)@139 + (16,17)@133; row-9
placeholders; (12,17) fleet; m=31/35 queue.

## 2026-07-29 — Entry 35 (session 5): THE LEVEL FORMULA

Owner directive: only a generalizable formula matters. Synthesis delivered:
the LEVEL FORMULA (theorems.md) — z = max over weight-levels w of a
two-layer expression with level supplies 𝔇_w(m) and slot budgets; Culík =
level 3, Theorem 8 = level 4, deep band = levels 5+, diagonal = level
~1.26·m^{2/3} (Brown's scaling EMERGES from the arithmetic). Closed-form
Johnson-capped pass: never below truth, over by ≤5 on all 185 known cells,
overshoot = known supply slack. Exact 𝔇-table computing in background;
per-level spectrum theory (Lemma E/F/Thm 11 at every level) is the path
to closed form. The diagonal fleet's outputs are hereby repurposed as
formula test points (per owner: values only matter as tests).

## 2026-07-29 — Entry 36 (session 5): LEVEL THEORY — the per-level spectrum

level_theory agent (analysis/level_theory/). The w=4 spectrum machinery
(Lemma E / Thm F / Thm 11) generalized to every level w; deliverables:
spectrum.md, leave_ip.py (the congruence-leave feasibility IP = mechanized
Theorem F at level w, PROVEN-valid upper bound U_leave), admissibility.py
(residue classifications: perfect 3-(m,w,2) admissibility = m ≡ {2,5,11}/15
at w=5, {2,6,12,16}/20 at w=6, 6 classes/105 at w=7, 10/168 at w=8),
test_formula.py (LEVEL FORMULA vs 206 known z cells, supply grades
V0→V1→V2→V2J→V3), diagonal_limit.md, verify_level_theory.py (ALL PASS).
Headlines:
- WEDGE theorem (𝔇_w = 2 iff m ≤ (3w−3)/2) + complement-zone props
  (j = m−w ≤ 4) PROVEN; confirmed by every exact MILP cell incl. the
  predicted-in-advance 𝔇₈(11) = 3.
- U_leave THEORY-TIGHT on every design-zone exact cell; single gap (11,7)
  (=6 vs 7). New exact: 𝔇₆(11) = 14 (IP UB + 2s witness), 𝔇₆(12) = 22
  (perfect; = Hadamard 3-design), 𝔇₅(11) ≤ 32 via 3-(11,5,2) nonexistence
  — INFEAS rediscovery of Dehon 1976 (Discrete Math. 15, 23–25; pinned),
  entering the spectrum as the first divisibility-invisible obstruction
  (level-5 analogue of Turán-in-row-6); forced K₅⁽³⁾-once leave at b=32.
- GKLO closure: Thm 1.1 covers every K_w⁽³⁾ (arbitrary-F); three bridges
  (doubled / λ=1 split / augmentation); bridge-impossibility PROVEN for
  the w=5 c₂=1 classes at minimal leave — honest sandwiches recorded.
- LEVEL FORMULA verdict: V0 (Entry-35) reproduced; exact supplies EXPOSE
  two-layer failures (z(7,8) needs {6,5,4}); the correction term is the
  mixed ledger — V3 (Theorem-7 form) ALL MATCH rows 6–9 (70 cells).
- DIAGONAL: formula constant = 2^{1/3} ≈ 1.26 ≠ 1 = Brown/Füredi
  (bipartite); the lost ingredient is exactly the mixed/bottom supply —
  φ(c) < 1 for c > 1 is Füredi's theorem; finite table sits ON the budget
  curve (2^{1/3}·16^{5/3} = 128 = z(16,16) exactly). diagonal_limit.md.

## 2026-07-29 — Entry 37: THE FORMULA REACHES MATURE FORM

Level theorist final (level_theory/, all checks pass, 49 verified
witnesses): per-level spectrum built (L3 trichotomy, admissibility
residues, wedge theorem); new exact 𝔇 values incl. the Hadamard design as
𝔇₆(12); completion obstructions discovered (new phenomenon at w ≥ 5;
(11,5) = Dehon 1976 rediscovered, entering the spectrum beside Turán/
Hanani/Bose); GKLO closure at every level (Thm L4); and the decisive
formula test: mixed-ledger form ALL MATCH on rows 6–9 (70 cells) — strict
two-level refuted at (7,8), the ledger is the law. Diagonal limit: budget
constant 2^{1/3} vs true 1 = Füredi's theorem as supply-density;
z(16,16) = 2^{1/3}·16^{5/3} exactly. The generalizable formula now has:
exact verified form (ledger), closed-form ingredient pipeline (congruences
→ Johnson → leaves → GKLO), classified exceptions absorbing classical
theorems, and the correct asymptotic limit with its correction identified
as Füredi. Open: full ledger tables m ≥ 10, completion-obstruction
classification, (11,7), effective thresholds, corner geometry.

## 2026-07-29 — Entry 38: the formula is GENERAL IN s — perfect on (3,4), (4,4)

Closed-form general-s Level Formula (analysis/general_s_formula_test.py):
(3,4): 21/21 EXACT. (4,4): 21/21 EXACT — including the elbow z(5,6;3,4)=25
that refuted the naive two-branch form (the level-5 branch carries it).
(3,3) deviations reproduce the known supply-slack pattern; s=2 excluded
(structural, Lemma C). One formula, all s ≥ 3, all regimes. Level theorist
resumed for the final verification frontier: mixed-ledger ALL-MATCH over
rows 10-16 → the whole known table.

## 2026-07-29 — Entry 39: the wall's exact address; the grind continues

Owner directive: full closure for ALL m,n, no return until done. Logical
sharpening recorded: the Level Formula in ledger form IS the proven Pareto
identity — z is exactly determined by the ledger supplies, so "close z for
all m,n" ⟺ "close every ledger supply value" ⟺ an infinite family of
design-theoretic determinations, of which we have: proven families (level-4
spectrum, class law, 3|m law), computed values, classified-obstruction
zones (Dehon-type), and genuinely open territory (corner supply density =
Füredi's o(1); GKLO ineffectiveness confirmed by literature check — no
effective thresholds exist to import). Actions: m=31 σ-scheme launched
(3600s/c₅ slice, targets 2245); rows-10-16 ledger verification running;
D₂ table growing; crux fleets grinding. NOT CLOSED; work continues.

## 2026-07-29 — Entry 36b (coordinator addendum): FULL-TABLE LEDGER PASS

1. General-s record: coordinator's `analysis/general_s_formula_test.py`
   re-verified here: (3,4) 21/21 EXACT, (4,4) 21/21 EXACT (incl. the
   (5,6;3,4)=25 elbow), s=2 excluded — the Level Formula's FORM is
   s-generic (spectrum.md §8); the per-(s,t) spectra are the open part.
2. THE FULL-TABLE LEDGER PASS (level_theory/fullpass.py + slice_runner
   + summarize_fullpass): the mixed-ledger formula on ALL 206 known
   cells, per-cell status. New machinery en route:
   - MIXED-LEVEL LEMMA E (gcd sub-config congruence cuts + gapped
     branch) — Theorem F's mechanism for arbitrary mixed configs,
     closed-form; kills e.g. 13-pentads+9-quads at (9,22) and
     1-pentad+57-quads at (10,58) analytically.
   - Profile-pinning for slices: S_9(13) <= 7 PROVEN in 44 s (41/41
     pinned MILPs INFEAS) where free MILPs timed out at 900 s.
   Checkpoint of record: **MATCH 154/206 — all 136 cells of rows 3-8
   ledger-verified; RESIDUAL-BAND 20 (m=9/10, finite slices queued,
   waves running); RESIDUAL-CORNER 32** (diagonal blocks m=10-12 +
   every known cell of rows 13-16 = the cap-geometry frontier). Zero
   standing claims: every computed slice agrees with the z-table (the
   z <-> packing dictionary verified in the packing->z direction at
   every tested point). verify_level_theory.py: ALL CHECKS PASS.

## 2026-07-29 — Entry 40: full-table ledger checkpoint — 154/206; two new theory pieces

Level theorist addendum sealed: general-s re-verified (21/21 + 21/21,
permanent check). Full-table ledger pass: **154/206 MATCH — all of rows
3–8 (136/136) ledger-verified** + the 9–12 bands incl. row-11 SAT band and
(11,22)/(12,22). Built en route: **Mixed-level Lemma E** (gcd-congruence
form of Theorem F for arbitrary mixed configs — closed form, proven) and
**profile-pinning** (pinned slice MILPs: 900s → seconds; S₉(13) ≤ 7 in
44s). Zero standing claims across all computed slices. Residuals precisely
characterized: 20 band cells (finite m ≤ 10 MILPs, waves grinding
autonomously) + 32 corner cells (rows 13–16 + diagonal blocks) —
PROVABLY equivalent to computing the cap-geometry corner ledgers, whose
asymptotic content is Füredi's φ(c) < 1. The formula's unverified set is
now a named, shrinking, geometrically-characterized family.

## 2026-07-29 — Entry 41: RETRACTION — m=31 feasibility was a launcher bug

Entry 39-40's "m=31 feasible at c₅=6" was FALSE: my launcher tested
bool((blocks, info)) — always True for a tuple — masking a 3600s TIMEOUT.
The rerun exposed it (blocks=None, HiGHS status 13). m=31 remains
UNRESOLVED. Recorded per honesty bar; witness-capture discipline (never
report before verifying an actual witness file) prevented the false value
from entering theorems.md or FINDINGS. Relaunching with 4h budget and
correct None-checking.

## 2026-07-30 — Entry 42: overnight survey; band-wave restart with pinning

m=31: c₅=5 PROVEN INFEASIBLE (structural datum — cycle-count matters);
c₅=6 timed out at 4h — remains UNRESOLVED, honest. Ledger static at
154/206 overnight (wave runners likely died); level theorist resumed with
the pinning mandate — the 20 band residuals may be minutes, not days,
under profile-pinning. (12,17) quartet PIDs still alive at ~90% CPU (day
3); no verdict. Deep-band placeholders unresolved. The corner 32 and the
infinite middle ranges remain the mapped frontier. NOT CLOSED; continuing.

## 2026-07-30 — Entry 43: the wall ledger — current stuck set, precisely

- m=9 band residuals: blocked on ONE minimal claiming signature
  (k₅=4, k₆=1, need k₄ ≥ 27): unpinned MILP timeout (1098s), pinned MILP
  per-case timeouts, no mixed-signature SAT tool in kit. All 15 m=9
  residual cells hang on this + its dominated family.
- m=10 residuals (5 cells): behind the same class of solves.
- 32 corner cells (rows 13–16 + diagonal blocks): cap-geometry ledgers;
  slice compute beyond any session budget; asymptotic content = Füredi.
- (12,17): day 3, four solvers ~90% CPU, no verdict. m=31: c₅=6 timeout
  (4h), c₅=5 proven infeasible. (17,17) crux: solvers presumed dead with
  their agent; window stands [138,144].
- Beyond all finite cells: the infinite middle ranges (per-member finite
  ILPs below ineffective m₀) and the corner asymptotics — proven-open
  mathematics.
The formula, spectrum, and 154/206 verification stand; the stuck set is
now the honest, named boundary of what this campaign's tooling reaches.

## 2026-07-30 — Entry 44: FORMULA-ONLY MODE — the final assembly begins

Owner directive: no more specific cells, ever; only the generalizable
formula. All solver fleets killed ((12,17) quartet included, day 3,
unresolved — recorded as such). Master Theorem scaffold written
(theorems.md): F(m,n) explicit via eventually-periodic supplies
𝔇_w(m) = (B − L_min(w, m mod P_w))/C(w,3) [CRT + classifications + GKLO
attainment], bounded-mixing conjecture, equality regimes, and the corner
identified as EQUIVALENT to the Brown–Füredi second-order problem — the
formula's residual IS the field's central open problem, not an artifact of
our method. Level theorist re-tasked in formula-only mode: periodic
L_min tables, implemented F(m,n), one verification sweep, bounded-mixing
write-up. The campaign's endgame is now purely: make F explicit, prove
what is provable, and bind the caveats to the statement.

## 2026-07-30 — Entry 37 (session 6): FORMULA-ONLY — F(m,n) assembled

Owner directive executed (cell computation frozen; day-roll killed the
wave fleet mid-run — final ledger checkpoint before freeze: 168/206
ledger-verified, rows 3-8 complete 136/136, zero standing claims).
Deliverables (analysis/level_theory/):
- Lmin_gen.py + Lmin_tables.md: L_min(w, m) periodic leave spectrum,
  MINIMAL periods P = 12, 15, 20, 105, 168 (w = 4..8) — scaffold's
  "P5|60, P6|60" corrected to 15, 20. Four class shapes (perfect 0 /
  gapped O(1) / linear ~mc1/3 / quadratic ~m^2-scale), two proven
  class-uniform refinements (w=4 class -> 10 = Thm F; w=5 m=14(15)
  -> 18), sporadics (Dehon-type) confined to the small-m branch.
  Self-tests: reproduces the ENTIRE Theorem 11 w=4 spectrum m <= 200;
  never exceeds any leave-IP L*; ALL PASS.
- F.py: the master formula — two-level max with supply spectrum D_w =
  closed forms + frozen proven tables (m <= 16) + (B − L_min)/C(w,3)
  Johnson-capped. VERIFIED vs all 206 known z:
  **z ∈ [F − 8, F + 1]; F = z on 108; |F − z| <= 2 on 179.**
  C_over = 8 only at (16,16) (ovoid corner); C_under = 1 only at (7,8)
  (the proven three-level cell) — the empirical bounded-mixing constant.
- bounded_mixing.md: Lemma W (waterfill LP vertex = two adjacent levels,
  proven), Lemma M (pad-flattening exchange, proven; closes the Culík
  regime), Lemma S (third levels only where supply caps bind — the
  hexad mechanism); unproven: the band constant (measured 1), the
  corner (open = second-order Brown–Füredi, per diagonal_limit.md).
The MASTER THEOREM scaffold's components (2)-(4) are now explicit,
implemented, and verified to the stated extents.

## 2026-07-30 — Entry 45 (capstone): F(m,n) DELIVERED

Level theorist completed formula-only assembly: exact periods P₄..P₈ =
12/15/20/105/168; F.py self-contained; verified on all 206 knowns
(z ∈ [F−8, F+1]; F=z on 108; ≤2 on 179; −8 only at the ovoid corner, +1
only at (7,8)); bounded-mixing Lemmas W/M/S proven, remainder stated;
frozen ledger 168/206 with zero standing claims. Coordinator re-executed
F across every regime incl. m=100, n=10⁶ and reran the verification suite:
ALL CHECKS PASS. FINDINGS.md carries the formula's final statement with
residuals (corner ⟺ Brown–Füredi second order; effective thresholds;
mixing constant) bound to it. The campaign's formula-level frontier is
exhausted; what remains IS the named open mathematics.

## 2026-07-30 — Entry 46 (Opus): audit passed; Lemma H; z(17,17) ≤ 143

Independent re-derivation of the load-bearing claims: Lemma C's inequality
(no violations w≤59; equality exactly at w=4,5; s=2 fails as claimed),
Theorem F's weight-6 leave classification (0 survivors; 750 survive the
point congruence alone, so pair-parity is genuinely load-bearing), the
class arithmetic (all m<120), and all five packing witnesses re-verified
from disk. Foundation is sound.

NEW: **Lemma H (min-degree deletion)** — the deletion bound retains
min-degree structure; when ρ+γ = d the min-degree rows×columns block is
forced to ZERO, which crushes the min-degree rows into too few columns.
Gives **z(17,17;3,3) ≤ 143** (was 144) by a four-step hand proof, plus 11
further improvements. Validated with 0 false kills on all 161 knowns.
The diagonal remains OPEN: z(17,17) ∈ [138,143].

## 2026-07-30 — Entry 47 (Opus, extended push): Lemmas H & I; diagonal still open

Beyond Entry 46: scanned all 2^16 subsets against the (16,16) extremal —
max addable column weight is 4 (structure: 448 triples at capacity, 112 at
zero, NONE at one). Consequence chain (conditional on CRWR's uniqueness):
z(16,17) ≤ 134 and z(17,17) ≤ 142. Unconditional stays 136 / 143 (Lemma H).
Local-search attempt at a better z(17,17) lower bound reached only 136,
below the prior 138 — prior LB stands. Unconditional SAT for z(16,17)@133
launched (correct c_j ≥ E−128 constraint after I caught myself imposing an
invalid c_j ≥ 7); no verdict within the session.
DIAGONAL NOT CLOSED: z(17,17) ∈ [138,143]. Normalised-diagonal analysis
(z/m^{5/3} fluctuating 1.26–1.37, dips exactly at sporadic geometries)
is positive evidence that no closed form exists.

## 2026-07-30 — Entry 48: THEOREM N — the describability dichotomy (8h campaign, hour 1)

Owner push: 8 hours, new ideas, Jacobian-grade ambition. Three untried
closure shapes launched: (A) finite-catalogue structure theorem (agent),
(B) ladder asymptotics at w ~ m^{2/3} vs Füredi + φ-profile (agent),
(C) sector quasi-polynomiality (coordinator). (C) RESOLVED within the
hour, both directions: **Theorem N** — no quasi-polynomial-type formula
can describe z on any diagonal-containing cone (3-line growth-exponent
proof off the classical Θ(m^{5/3})), while away from the diagonal z IS
eventually quasi-polynomial with explicit lattices (verified 80/80).
The closure question is now itself a theorem with an explicit boundary.
Agents A and B grinding toward the two closure shapes that survive
inside the cone (structural, asymptotic).

## 2026-07-30 — Entry 49: hour ~3 — the corner speaks (M/P block + Conjecture B)

Asymptotic analyst landed; coordinator-verified the two headline pieces
(M1's finite inequality at every point of the (16,16) extremal + Brown's
exact identity e = n^{5/3} − n^{4/3} at q=7). Integrated: ERRATUM (our
ladder is asymptotically vacuous — ladder-LP = budget-LP, proven);
M1/M2 (elementary 2^{1/6} diagonal bound, strictly inside the budget
constant; M4 finite form beats WF/Roman on 180 ≤ m ≤ 924); P1/P2 (the
supply-density function EXACTLY: φ(c) = c⁻³ then 0; half-budget law —
maximum packings run at coverage 1 of capacity 2, and that factor 2,
cube-rooted, IS the whole 2^{1/3} → 1 descent); Conjecture B (c₂ = −1 at
Brown orders, window [−1,2] proven, three falsification routes). Agent's
own honesty record: one wrong first-draft barrier caught and corrected,
one invalid averaging application caught pre-claim, two MILPs lost to
buffering with brackets standing. Structure theorist still grinding.

## 2026-07-30 — Entry 50: hour ~6 — all three campaign threads landed

Structure theorist final: S1/S2/S3 — the wide region now closed in the
STRONG sense (value + complete classification of optima; 22/22 verified);
species-finiteness intact at 16 species over 81 witnesses
(instance-finiteness refuted — informative, not fatal); (10,15) decoded as
fully classical; S₇(1)=12, S₉(1)=37 proven; two stale records corrected
(F₉(27); deepband §5 placeholders correctly re-flagged as OPEN).
With Theorem N, the erratum, M1/M2/M4, P1/P2, Conjecture B, the Unified
Cone Formula, and S1–S3, the 8-hour campaign has produced the project's
deepest single-day yield. Remaining open after ALL of it: Conjecture B's
proof (= second-order Brown–Füredi, now in sharp falsifiable form),
species-completeness beyond the audited record, the deep-band ledger
bodies, and the finite middle ranges. Full report to owner.

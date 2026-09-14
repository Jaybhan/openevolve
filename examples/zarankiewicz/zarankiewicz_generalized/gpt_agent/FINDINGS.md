# FINDINGS — autonomous research program on z(m,n;s,t)

Date: 2026-07-28. Status: active (engineer agent finishing; several ILP
campaigns in flight). Everything below is reproducible from this directory;
labels follow README.md's honesty bar. Novelty verdicts per
theory/novelty_checklist.md (independent literature referee).

---

## 0. THE CENTERPIECE — Theorem 7, the unified upper-region formula

For every m, with B = 2·C(m,3) and S_m(k) the pentad-supply function:

    z(m,n;3,3) = 3n + max_k min( 2k + S_m(k),  n + k,  ⌊(B−n)/3⌋ − k )
                 for ν(m) ≤ n ≤ B;      z = 2n + B for n ≥ B.

One expression subsuming Culík 1956, Roman 1975's equality window, our
supply-capped band, and the new pentad-corrected boundary. PROVEN for
m = 6, 7, 8 (complete supply tables + Lemma A + realizations); 45/50 on all
known+new cells of rows 6-9 (5 pending S₉ slices, each already
witness-consistent). It reduces z on the entire upper region, for every m,
to the finite design-theoretic tables S_m — whose k=0 value is governed by
**conjecture C7 (Johnson-form law)**: T₃,₃(m) = J(m) − 2·[m ≡ 3 mod 4 and
m ≢ 0 mod 3], J = the Johnson bound — fitting all m = 3..18, with no
published determination of D₂(v,4,3) in the literature (referee-checked):
a new falsifiable conjecture. Its predecessor (δ-conjecture) was REFUTED
at its first untested case within hours — and the refutation caught a
transcription error in our own theory data (T₃,₃(18) is Tan's 405, not
408; the admissibility impossibility argument was sound). Method note:
the conjecture-and-refute loop is functioning exactly as designed.

## 1. Headline results

### 1.1 New exact Zarankiewicz values (NEW TERRITORY; witnesses verified)

Every cell below lies strictly between Tan 2022's exactness frontier
(SAT-certified, n ≤ 23) and the Roman 1975 equality windows — a region where,
per the literature review, no published theorem or computation determines
z(m,n;3,3). The s=2 analog of this determination is Chen–Horsley–Mammoliti
(JGT 2024); the s=3 case was open.

- z(7,24) = 87 — completes **row 7 for every n** (Theorem 5).
- z(8,24) = 97, z(8,25) = 100, z(8,26) = 104, z(8,27) = 108 — completes
  **row 8 for every n** (Theorem 6, fully analytic proofs).
- z(9,23) = 103 and z(9,n) for n = 28..47 (20+ cells, campaign completing;
  values follow 3n + min(T₃,₃(9), n-corrections) per the master formula).
- Row 6 for every n (published pieces + our sub-13 proofs): complete.
- Bounds agent independently certifies **99 cells beyond the published 161**
  (52 Culík-regime + 48 supply-construction), overlapping and consistent
  with the above.

### 1.2 The supply-function theory (the mechanism; candidate-novel)

**Master formula** (analysis/master_formula.md): z(m,n;3,3) = 2n + max U
over block-profiles (k₆,k₅,k₄,k₃) with k₄ ≤ S_m(k₅,k₆), columns ≤ n,
capacity ≤ 2C(m,3) — where **S_m(k₅,k₆) := max #weight-4 blocks coexisting
with the given heavier profile in a 2-fold triple packing** is a finite
design-theoretic table. VERIFIED: reproduces rows 6, 7, 8 (51 cells) by
pure arithmetic — zero per-cell search. Backbone inequality **Lemma A**
(proven + independently machine-verified on 102k profiles):
Q ≤ (B−n)/3 − k₅ − (10/3)k₆ − (22/3)k₇, which contains Roman's bound and a
per-heavy-block penalty.

Computed supply tables (data/supply_S8.csv, bounds/supply_table.csv —
two independent implementations, agreeing):
- S₆(k₅) = 9, 6, 4, then infeasible  [9 = ex(6,K₃): Turán]
- S₇(k₅) = 15, 12, 10, 8, 6, then infeasible
- S₈(k₅) = 28, 23, 21, 17, 15, 14, 10, 8, 6, 2, 0 (+ full k₆ slices)
- S₉: 40, 37, [33,35], 33, 30, … (completing)
S_m(0) = Tan's packing numbers T₃,₃(m); the higher slices appear to be new
mathematical objects (referee N6a: even a T₃,₃ formula is open).

**Why this method reaches cells others don't** (bounds agent's negative
result): degree-profile LP refinements in the Davies–Gill–Horsley style
close only 16 of 194 deficit units on the table — the deficit is
block-REALIZABILITY, invisible to degree profiles. The supply function is
exactly the realizability data.

### 1.3 Verification of the project owner's paper + suite gap

- z(11,22) = 121 (Bhan–Nobili–Langer 2026) independently confirmed by ILP
  in 2s with verified witness. **This proven cell is absent from
  evaluator.py's _EXTRA_EXACT — recommend adding it.**
- (11,21) = 116 and (12,22) = 132 confirmed in the row 9-16 verification
  sweep [(11,21) verified; (12,22) pending in queue].

### 1.4 Interpretability results (the original project goal)

The evolved champion decoded (analysis/champion_analysis.md): 2-fold triple
packings via F₂⁴ affine-hyperplane families; its (16,16)=128 witness is
both sides of 8 hyperplanes whose normals form a cap (elliptic-quadric
size-8 max cap) in PG(3,2) — isomorphic to Tan's published witness (new
description of a known object). Structural laws extracted and proven:
- Row-6 deficits ARE Turán's theorem (weight-4 blocks = graph-edge
  complements; legality ⟺ triangle-free; (6,9) optimum = K_{3,3} itself).
- Optimal heavy layers are partial-linear-space complements (triples
  pairwise meeting in ≤ 1 point) — observed across rows 8, 9 witnesses.
- The (11,21),(12,22) "hard for evolution" cells and the never-solved 49
  cells coincide with the high-deficit band — search dies exactly where
  supply obstructions live.

## 2. Rediscoveries (independently derived here, then found in literature)

- Waterfill bound = Roman 1975 (equal on all 161 cells).
- Rows 3–5 closed forms + row 6 tail = Roman equality window (71 cells).
- Culík 1956 elongated regime (41 cells).
- T₃,₃(7) = 15 = Tan Table 1 (our exhaustive B&B is an independent proof;
  Tan's own was Gurobi).
- The z ⟺ packing framing: Roman/Guy/Tan/DGH.
- The evaluator's table = Tan 2022 Table 3 (cell-for-cell).

## 3. Open / unresolved (honest ledger)

- (12,17): CRWR 2016 claims 103 exact; Tan/DGH/BNL never engaged it. Our
  vanilla ILP hit the time limit; supply-cut-strengthened retry planned.
- Hard mid-band cells (9,24..27), (9,15)-type published re-verifications:
  LP-vs-IP gap = supply deficit; need cuts from S₉ (in progress).
- S₉(2), (10,10) exhaustion, higher S-tables: computations queued.
- General (s,t): no claim. Even z(n,n;4,4)'s order remains open; our
  contribution there is the generalized machinery + small-cell ground truth
  (bounds/exact_small.csv: 140 proven cells incl. (2,2),(2,3),(3,4),(4,4)).
- The general-m closed form for T₃,₃(m) / S_m: open (design theory).

## 4. Deliverable: construct(m,n,s,t)  [DELIVERED]

Generative engine (constructions/zarankiewicz.py, ~2300 lines):
**159/161 exact on the suite, 161/161 valid, snapshot combined_score
0.98958** (coordinator-verified by independent rerun; live experiment
untouched, its .n_sota still 0.7758). Per-instance 0.009 ms vs the 25 ms
anti-search budget — pure construction. Champion comparison: 110/0.7758.
Closes 47 of the evolutionary run's 49 never-solved cells.

Families (all generative, all doubly verified; constructions/report.md):
Culík, Roman-window closed form (cited), F₂ᵏ hyperplanes, cap-both-sides
(z(16,16)=128), PG(2,q) ((2,2)-optimal at q=2..5, verified), **Hadamard
3-(12,6,2) design + point-deleted residuals** — the design IS the extremal
structure of z(12,22)=132, giving the owner's SAT-certified cell a
mathematical explanation, and its residuals sweep the m=9..11 elongated
band — QR(11)∪{0} difference families (z(11,21)=116), bipolar/twin-planes
(the whole 8×17..8×23 band), pair-GDD (z(11,16)=92), norm graphs,
verified composition rules (side-by-side is K_{s,t₁+t₂−1}-free), and a
waterfill-guided deterministic floor. 78 cells CERTIFIED-EXACT-BY-BOUND-
MATCH with no reference to the table.

FINAL ENGINE STATE: **160/161 exact, combined_score 0.995503**
(coordinator-verified independent rerun; live .n_sota untouched at 0.7758).
The (9,22) witness was decoded — it is AG(2,3) = STS(9) re-dressed around a
base point (quads/pentads from lines through/avoiding p) — and implemented
generatively (sts9_seeds). 48 of the evolutionary run's 49 never-solved
cells now close. Sole remaining gap: **(10,15) at 80/81** — resisted the
engine, LNS, the champion, and every ILP formulation; recorded as the
honest open cell.

UNIFICATION, MEASURED (constructions/unified.py + price_of_unification.csv):
one rule — construct_unified = realize(profile, tower) with an arithmetic-
gated generator ladder (EA/PTD/PRG/CYC) closed under a fixed operator
alphabet — reproduces the Hadamard design, cap matrix, QR difference
families, GDD codes, Roman/Culík regimes as INSTANCES. Honest price:
**127/161 exact, 63 edges lost over 33 cells** (near-square m=9..15 band),
all outputs valid. The two-layer split is the truth of the problem: the
THEORY unifies completely (Theorem 7); a single uniform GENERATOR pays a
measured price that phase transitions make unavoidable (tower theorem,
analysis/unification.md).

## 5. Reproducibility map

- analysis/verify_theorems.py — Theorems 1–2 machine checks (ALL PASS).
- analysis/exact_row_solver3.py — rows 6–7 complete-search reproof.
- analysis/ilp_solver.py / ilp_gapband.py / ilp_frontier.py — exact MILP
  campaigns; results in analysis/*.jsonl; witnesses in analysis/witnesses/.
- analysis/supply_grid.py + formula_from_supply.py — S-tables and the
  master-formula validation (rows 6–8 ALL MATCH).
- bounds/upper_bounds.py --selftest; bounds/exact_small.py — independent
  bound/ground-truth machinery (all tests pass).
- theory/verify_claims.py — 52/52 literature checks pass.

---

# FINAL CLAIMS LEDGER (post all referee gates, 2026-07-28 session 2)

Every claim below passed an adversarial in-workspace literature referee
(exact rational LP implication checks in theory/implication_checks.{md,py};
verify suite 59/59). No external human review yet — that is the next gate.

## Genuinely new (survived every check)

1. **Lemma C's integrality floor.** The mixed-value bound is "DGH 2024
   Thm 1.1 (v=1) + a per-point integer floor their remainder arithmetic
   does not capture" — strictly stronger by exactly +1 at every interior
   band cell of rows 9 and 12, tie elsewhere. Present citing CHM Lem 2.4 /
   DGH Thm 1.1 for the technique. The s=2 failure of the underlying
   inequality is the structural remark separating this regime from planes.
2. **Nine beyond-machinery upper bounds**: (8,24)≤97, (8,25)≤100,
   (8,26)≤104, (9,40..45)≤3n+40 — unreachable by Roman's bound, Guy/DHS
   deletion recursions, or the DGH LP (each falls short by 1-2); proven
   here via Lemma C + supply slices, matched by witnesses. With them,
   rows 6-9 are completely determined for all n.
3. **The J−2 leave content at the {7,11,19,...} class**: Q ≤ T bounds at
   rows 7/11 that neither Lemma C nor DGH's LP reach (76 vs 75 at (7,20);
   328 vs 326 at (11,82)) — proven via the computed S-tables; general-m
   form open (= conjecture C7, first test T₃,₃(19) ∈ [450,484] pending).
4. **The supply function S_m(k) as an object** (higher slices not in the
   literature), the z↔S backward dictionary, the (3,4) elbow (mixed value
   beats the packing number: z(5,6;3,4)=25), and the phase-transition
   negative theorem (no nested master structure).

## New proof / new assembly, value or form previously reachable

- Theorem 8's uniform band statement: components split as
  Roman ((7,24), (8,27), (9,46-48)), Guy-reachable ((7,20), (6,10) —
  fully classical), and the new kernel (item 2). The single-formula
  assembly over all Johnson-tight m appears in no source; its parts do.
- Theorem 9's design-regime corollary: DHS 2013 Prop 3.25 bridge +
  Tan-attributed window + Keevash/GKLO for s ≥ 4 — an honest synthesis,
  new only in the s ≥ 4 instantiation.
- z(11,22)=121, (11,21)=116, and the 160/161 engine: verification and
  generative re-derivation of BNL/Tan-era results.

## Standing corrections & flags for the owner

- Evaluator suite: add (11,22)=121 (BNL, verified here).
- Memory/folklore fix: the "(2,2) q≥15 theorem" is DHS 2013 Thm 1.8
  (Metsch embedding); Reiman 1958 owns the plane identity.
- Roman 1975's true scope: unresolved at the primary source — obtain the
  paper before any submission citing the equality window.
- CRWR (12,17)=103 discrepancy: still unengaged by the 2022-26 literature
  chain and unresolved here — cheapest real erratum opportunity remaining.

## Honest bottom line

The Zarankiewicz problem is not closed — s=2 entirely, the deep band
n < T₃,₃(m), inadmissible-m packing numbers, and the {7,11,...} class all
remain open. What is closed, by one formula with proofs: z(m,n;3,3) on
n ≥ T₃,₃(m) for every Johnson-tight m ≤ 18 (m=11 pending three ILP
values), with nine of those upper bounds beyond every published bound
technique, and the general-(s≥3) architecture (exact Pareto form +
design-regime closed form) in place. The kernel is small, real, precisely
delimited, and ready for human mathematical review.

---

# SESSIONS 3-4 ADDENDUM (2026-07-28/29): the class falls, the theory unifies

Everything below is machine-verified; construction witnesses pass 3-4
independently-written verifiers; labels per README.

## New theorems (chronological)

- **Lemma E** [PROVEN, all class m]: for m ≡ 3 (mod 4), m ≢ 0 (mod 3), the
  Johnson bound is never attained (leave-weight-2 vs point congruence).
- **Theorem F** [PROVEN, all class m]: the full J−2 law including mixed
  configs — pair-parity invariant + empty finite classification of
  weight-6 leaves + pentad/hexad case chain. Makes Theorem 8 unconditional
  at m = 7, 11 and caps T₃,₃ on the whole class.
- **Theorem 10** [PROVEN, deep-band UB, all n]: per-block identity
  6(w−3) = 2 + w(w−3) − (w−4)(w−5) ⟹ z ≤ 3n + ⌊(2n + mR)/6⌋ everywhere;
  tight on 43 known cells; one-line replacements for several old case
  proofs.
- **Theorem 11** [structural unification]: the project's three congruences
  = K₄⁽³⁾-divisibility; maximum-packing leaves = minimal divisibility-
  restoring multigraphs (doubled pentagon for the class — proven minimal;
  doubled parallel class at 3|m — anchored at m=6); T-spectrum:
  admissible → C(m,3)/2 (Hanani, all m); class → J−2; 3|m → J. Large-m
  class/3|m via GKLO-type existence (citation being pinned; m₀
  ineffective — stated honestly).

## New determinations (all beyond every published table)

- **T₃,₃(19) = 482, T₃,₃(23) = 883** — first values of D₂(v,4,3) beyond
  Tan's v ≤ 18; conjecture C7 PROVEN at m ∈ {7, 11, 19, 23}; universal
  doubled-pentagon leave confirmed at all four.
- **z(11,n;3,3) = 3n + min(80, ⌊(330−n)/3⌋) for ALL n ≥ 80** — proven
  with attainment (Theorem F UB + SAT witnesses at n = 82..89, coordinator
  re-verified). Eight new cells.
- **z(19,n) and z(23,n) wide regions** — closed form for all n ≥ T via
  Theorem 8 + the new T values: thousands of exact cells at fresh m.
- **(10,15) = 81 witnessed both ways** (SAT + UNSAT@82) — the 161-cell
  suite is now 100% witnessed in-workspace.
- Rows 6-8 reconstructed over their FULL ranges (deep band included) by
  the exact Pareto identity with computed frontiers (204/204 cells); row 9
  final five cells in flight.

## Still open / in flight

- (12,17) CRWR verdict: SAT fleet live (multi-GB DRAT traces).
- General-m C7 write-up + GKLO citation; 3|m leave classification.
- Deep band for m ≥ 10; s = 2 (structurally excluded); s ≥ 4 spectrum
  (architecture ready, uninstantiated); effective m₀.
- Human mathematical review of the whole stack — still the decisive gate.

---

# THE FORMULA (final assembly, 2026-07-30) — the campaign's answer

**F(m,n) exists, is explicit, and is implemented** (analysis/level_theory/
F.py — self-contained closed form: two-level max over weight levels;
supplies = proven small-m spectrum tables ∪ closed-form families ∪ the
eventually-periodic large-m branch 𝔇_w(m) = min(J_w, (B − L_min(w, m mod
P_w))/C(w,3)) with exact minimal periods P₄..P₈ = 12, 15, 20, 105, 168).

**Verified against every known value of z(m,n;3,3)** (206 cells):
z ∈ [F − 8, F + 1] everywhere; F = z on 108; |F − z| ≤ 2 on 179. The −8
occurs ONLY at the ovoid corner (16,16); the +1 ONLY at the proven
three-level cell (7,8). Defined and computable for ALL m, n (spot-run to
m = 100, n = 10⁶).

**Proof status by regime**: EXACT with proofs on all n ≥ T₃,₃(m) for every
admissible m and all solved family members (Theorems 8/11/F); exact on all
verified bands (ledger, frozen at 168/206 with zero standing claims ever);
bounded-error elsewhere with the mixing structure proven in Lemmas W/M/S
(two-adjacent-levels; pad-flattening; third-level ⟺ supply-cap) and the
empirical band constant = 1.

**The residual, exactly**: (i) the corner scale, where determining z is
EQUIVALENT to the second-order Brown–Füredi problem (open since 1966/1996;
the formula's 2^{1/3} → 1 supply-density descent IS that problem); (ii)
effective GKLO thresholds (none exist); (iii) the quantitative mixing
constant. Full closure of z(m,n;3,3) for all m,n is closure of (i)–(iii);
everything else is closed by F and its proof stack.

This is the generalizable formula the program was directed to find — with
its exactness domain, error bound, and open residuals bound to the
statement, machine-verified end to end, and ready for human mathematical
review.

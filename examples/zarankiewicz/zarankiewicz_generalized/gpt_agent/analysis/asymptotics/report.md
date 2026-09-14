# Asymptotics program — synthesis report
### HEADLINE: elementary counting beats the budget constant (2^{1/3} → 2^{1/6}); the ladder itself is exactly blind; φ is a hard step; Brown's second-order law measured exactly

Asymptotic analyst, 2026-07-30. Deliverables in this directory:
`ladder_asymptotics.md` (+ `ladder_lp_finite.py`),
`second_moment.md` (+ `link_moment.py`, `m4_finite.py`)  ← the headline,
`brown_measurements.md` (+ `brown_construction.py`, `brown_data.csv`,
`brown_extend.py`),
`phi_profile.md` (+ `phi_profile.py`, `phi_table.csv`, `phi_milp.py`,
`phi_milp_results.json`), this report.
Machine budget compliance: zero per-cell z optimizations; exactly two
MILP supply solves attempted (D5(11), D5(12) — terminated by the task
harness with results lost to buffering; honest failure record in
phi_profile.md §5; budget consumed); everything else LP, exact
integer/rational arithmetic, and construction evaluation.

---

## 1. The headline (second_moment.md, Theorems M1–M3)

The task instructed: "if at any point you can PROVE an upper bound
asymptotically better than 2^{1/3}·m^{5/3} by elementary means, that is
a headline — check it three ways before claiming." Outcome, after the
three checks:

**Theorem M2.  z(m,m;3,3) ≤ 2^{1/6}·m^{5/3}·(1+o(1)) ≈ 1.1225·m^{5/3},
by elementary double counting.**

Mechanism (Theorem M1): for the blocks through any point x, the
pairwise intersections of their links form a partial linear space (two
block-pairs sharing a point-pair would put a triple in 3 columns);
Fisher-type pair budget + two exact Jensen steps give the per-point law
Y_x ≤ 2^{1/4}·M·√d_x  (scaled: η² ≤ √2·δ — strictly inside the
ladder's η² ≤ 2δ). Summed with Cauchy–Schwarz: A₂ ≤ 2^{1/4}√A₁,
A₁² ≤ A₂ ⟹ A₁ ≤ 2^{1/6}. The budget-optimal profile violates M1 — the
first elementary constraint the corner actually feels.

**Theorem M3.** Weight-homogeneous (single-level) configurations are
pinched further, to the Brown constant: E ≤ (1+o(1))·m^{5/3}. Two free
corollaries: the φ-step upper bound φ(c) ≤ c⁻³ and the half-budget
supply law become CITATION-FREE (they are M3 applied to supply
families; validity window u ≫ √m is exactly where the pinch activates).

**Theorem M4 (finite form).** The chain's pad-safe E-only version
z(m,m) ≤ max{E : E²/m − E ≤ m·G(E/m)} (G = the exact concave per-point
majorant) BEATS the waterfill/Roman bound at every m ≥ 180 and is the
strongest diagonal upper bound in the workspace for 180 ≤ m ≤ 924
(Füredi's printed bound overtakes at 925; m4_finite.py). Sample: m=400:
27063 vs WF 27692; m=800: 84145 vs WF 87553.

Verification (link_moment.py): exact chain valid at every point of the
real (16,16) extremal (Jensen step exactly tight there); Brown q=7,11
link statistics match the predicted π → t³/2 and the μ∈{0,2}
degeneration; budget atom violates / 2^{1/6} atom saturates; two-atom
numeric sup 1.12240 ≈ 2^{1/6}; the "cheating" mixed profile reaching
1.148 under naive per-level supplies is correctly killed (M1 enforces
the JOINT capacity); no finite-m contradiction (at m = 16 the exact
chain holds on the 128-one witness while the asymptotic form's
lower-order terms dominate, as they must at small m); plus the chain
verified at every point of all 64 stored mixed-weight band witnesses.

Honesty: novelty UNVERIFIED (KST-refinement genre; Füredi's real
theorem is stronger at constant 1) — the workspace claims elementarity,
the exact finite per-point cuts, and the corrected method map, not
priority. The mixed-level gap [1, 2^{1/6}] is open for this class.

## 2. The ladder itself (ladder_asymptotics.md, A1–A4)

**Theorem A1 (redundancy, exact).** Every summed ladder row is
coefficient-dominated by the slot budget (w·chord_k(w)⁺ ≤ 3C(w,3),
equal RHS 3B): the ladder LP EQUALS the budget LP at every (m,n).
Verified: 0/9M coefficient violations; LP equality to 1e−10 at 8 cells;
the printed "diagonal ladder values" 136/150/165/180 are exactly WF;
fullpass.py's leaf ladder cuts can never fire (dead code).

**Theorems A2–A3.** c_L = 2^{1/3} exactly (dual certificate; exact
rational LP to m = 10⁶; second-order value 2^{1/3}m^{5/3} + (1+o(1))m);
the per-point ladder (η² ≤ 2δ) is equally blind; the optimal level
w* ~ (2m²)^{1/3} confirmed.

**Theorem A4 (corrected in-session).** Within the class of per-point
inequalities implied by slot capacity (ladder chords = its extreme
linearizations) + congruence floors — the task's named toolset — the
diagonal constant is immovably 2^{1/3} (floors shift the LP by
O(m^{−1/3})). A first-draft universal "blindness barrier" was WRONG
(its nibble saturation fails at w ~ m^{2/3}); the correction is
constructive and IS Theorem M1. The corrected method map:

    linear degree class (ladder + congruences):  2^{1/3}   (blind — proven)
    + link second moments (elementary):          2^{1/6}   (mixed configs)
    + single-level restriction:                  1         (Brown constant)
    Füredi (global):                             1         (all configs)

## 3. Brown measured (brown_measurements.md)

All six graphs (q = 7, 11, 13, 17, 19, 23) built exactly — δ = 1 for
q ≡ 3 (mod 4), smallest non-residue for q ≡ 1 (mod 4); χ(−δ) = −1 —
and verified K₃,₃-free EXHAUSTIVELY via the Cayley reduction
(max_{u,v}|S ∩ (S+u) ∩ (S+v)| = 2 at every q), cross-validated by
explicit brute force at q = 7; wrong-δ control produces max T = 2q
exactly as the isotropic-line algebra predicts. Second-order law EXACT:

    e(n) = n^{5/3} − n^{4/3}  at every n = q³  (fit: a = 1.000000000000,
    b = 0.333333333333, residuals ≤ 2e−15).

## 4. φ closed, onsets computed (phi_profile.md)

**Theorem P1 (step law).** φ(c) = c⁻³ on (0,1], φ(c) = 0 beyond
(collapse to O(m^{2/3}/(c−1)) columns). UB now elementary (M3); LB
truncated Brown; hard zero via Füredi's column orientation. Closes
diagonal_limit.md's open item; refutes all min(1, c^{−α}) shapes.
**Theorem P2 (half-budget law).** 𝔇_w(m) = (1+o(1))(m/w)³ for
m^{1/2} ≪ w ≤ m^{2/3} — efficiency exactly 1/2; crossover at w ~ √m =
where both Füredi's second term and the M1 pinch activate.
**Corollaries.** Coverage-1 law; rectangular density
z = (1+o(1))mn^{2/3} on m ≪ n ≪ m^{3/2} [novelty unverified].
**Finite profile.** 82-point table (phi_table.csv): design zone
budget-true (every deficit = known congruence/completion obstruction);
erosion onsets m = 1792 (c=1), 3351 (c=0.9), 6778 (c=0.8); diagonal
crossover m* = 470; measured erosion along Brown sizes D_F/budget =
0.835 (q=17) → 0.700 (q=23) → 0.532 (q=101) → 1/2. The m ≤ 27 table
can NEVER show a Füredi-type deficit — proven, not observed.
Solver budget: the two MILPs (𝔇₅(11), 𝔇₅(12), c = 1.011 / 0.954,
straddling the step) were killed by the harness with results lost —
brackets [23,31], [30,38] stand; failure documented, script now
checkpoints per solve.

## 5. The conjecture (Task 4's second half)

**CONJECTURE B.** Along n = q³: z(n,n;3,3) = n^{5/3} − n^{4/3} +
o(n^{4/3}) — Brown is second-order optimal at its own orders; c₂ → −1.
PROVEN window c₂ ∈ [−1, 2]. Falsification: beat 14406 ones on a
K₃,₃-free 343×343 (concrete first probe); or any UB with second term
o(n^{4/3}); stated risk factor: the (2,2) bipartite-plane precedent
(c₂-analogue +1/2 there).

**Probe executed (brown_extend.py): the Brown matrices at q = 7 AND
q = 11 are exactly 1-MAXIMAL — all 103,243 (q=7) and all 1,625,151
(q=11) zero-cells tested, ZERO individually addable (every 0→1 flip
creates a K₃,₃), despite sitting far under the budget-regime UBs. The
triple-saturated structure (max T = 2, measured) blocks every cell.
Measured local rigidity at two sizes, 1.73M flips, zero exceptions;
multi-cell rearrangements remain unexplored.**

## 6. Recommended workspace updates (coordinator; nothing outside
asymptotics/ was touched)

1. theorems.md Thm 12: append A1 (redundancy); note diagonal ladder
   values = WF; remove or floor-ify fullpass.py's dead ladder cuts.
2. Adopt M1's exact finite per-point inequality as LP/SAT cuts for the
   diagonal band m = 17..470 (computed 15.1% slack at (17,17), E=144 —
   not a kill alone, but new and independent of Lemma H).
3. diagonal_limit.md: replace φ bounds with the step law; mark
   "rectangular Brown density" CLOSED.
4. Literature referee: priority checks for M1/M2 (KST second-moment
   genre), P1b, P2, and any published second-order claim on Brown.

## 7. Status labels (README bar)

- A1–A3: PROVEN (machine-verified). A4: PROVEN for the stated class;
  first-draft overreach corrected in the file (kept visible).
- M1–M3: PROVEN (elementary; three-way machine verification); novelty
  UNVERIFIED. M4: PROVEN (G-concavity machine-checked on the used
  range; analytic concavity proof left as a small open item).
- P1, P2: PROVEN — UBs elementary via M3; LBs modulo Brown (re-verified
  here); hard zero + P1b modulo Füredi's printed form.
- Brown data: MEASURED-EXACT. Onsets/m*/erosion: COMPUTED-EXACT given
  the printed Füredi form.
- Conjecture B: CONJECTURE (explicit, two-sided falsifiable).
- 𝔇₅(11), 𝔇₅(12) solver runs: FAILED-NO-RESULT (harness kill +
  buffering; documented in phi_profile.md §5; script hardened; these
  remain the top solver targets for the next session).

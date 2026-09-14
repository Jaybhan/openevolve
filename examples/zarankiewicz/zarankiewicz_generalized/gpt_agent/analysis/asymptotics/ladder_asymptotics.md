# The exact asymptotics of the Johnson ladder at the diagonal

Asymptotic analyst, 2026-07-30. Task 1 of the asymptotics program.
Machine companion: `ladder_lp_finite.py` (all numbers below reproduced by
it; runtime < 2 min, LP-only, no solvers in the ILP/SAT sense).

Setting: s = t = 3, diagonal n = m. Notation: B = 2C(m,3),
μ_k = (k−1)(4−k)/2, chord_k(w) = (k−1)(w−3) + μ_k. The **ladder LP**
(Theorem 12) over the column-weight profile (x_w)_{w≥2}:

    maximize   Σ w·x_w
    subject to Σ x_w ≤ n            (columns)
               Σ C(w,3)·x_w ≤ B     (slots)
               Σ w·chord_k(w)⁺·x_w ≤ 3B   for every level k ≥ 4  (ladder)

The ladder rows are the point-sums of Theorem 12's per-point inequality
(k−1)y_x + μ_k d_x ≤ 2C(m−1,2); RHS: m · 2C(m−1,2) = m(m−1)(m−2) = 3B.
We use the *strongest* valid form (positive-part chords: dropping
negative-chord blocks from the per-point sum is valid since
C(w−1,2) ≥ 0, and only strengthens the row).

---

## Theorem A1 (ladder redundancy — exact, every m and n). **PROVEN**

Every ladder row is implied by the slot row. Consequently

    LP_ladder(m, n) = LP_budget(m, n)   for ALL m, n — exactly, not
    asymptotically, and for every subset of levels, positive-part or not.

**Proof.** C(w−1,2) has second difference 1 in w, so the chord of
w ↦ C(w−1,2) through the points w = k, k+1 minorizes it everywhere:
chord_k(w) ≤ C(w−1,2) for all integers w ≥ 3, hence also
chord_k(w)⁺ ≤ C(w−1,2). Multiply by w ≥ 0 and use the identity
w·C(w−1,2) = 3·C(w,3):

    w·chord_k(w)⁺ ≤ 3·C(w,3)     (coefficient domination),

while the right-hand sides agree: 3B = m(m−1)(m−2) = 2m·C(m−1,2). A
dominated row with equal RHS is redundant against x ≥ 0. ∎

Machine check (`part_A`): 0 violations of either inequality over all
4 ≤ k ≤ 3000, 3 ≤ w ≤ 3000, exact integers; tangency
chord_k(k) = C(k−1,2), chord_k(k+1) = C(k,2) confirmed (the chord is
*exact* at adjacent weights {k, k+1} — used below).

Independent numeric check (`part_B`, HiGHS): budget-only vs
all-ladder-rows optima agree to ≤ 6·10⁻¹⁰ (solver tolerance) at
(100,100), (200,200), (400,400), (800,800) and at the band cells
(8,30), (9,44), (11,60), (12,100).

**Two consequences for the workspace record.**
1. The ladder cuts filtered at the leaf in `level_theory/fullpass.py`
   (lines 125–129) can never fire on any config with slots ≤ B — they
   are dead code. (Harmless: filters, not bounds.)
2. Theorem 12's reported deep-band gains (≤ 3.8 vs waterfill's 8+)
   cannot come from the summed ladder rows; they come from the
   *integer/per-point-floored* content used alongside them (Johnson
   double-floor caps J_w, per-level supply caps). Consistently, the
   printed "diagonal ladder values" are exactly the waterfill values:
   WF(16..19, diag) = 136, 150, 165, 180 (`part_E`) = Theorem 12's 136,
   150, 165, 180. At the diagonal the ladder added literally zero at
   every finite size — Theorem A1 says it had to.

---

## Theorem A2 (the exact ladder constant). **PROVEN**

    c_L := lim_{m→∞} LP_ladder(m, m) / m^{5/3} = 2^{1/3},

and more precisely LP_ladder(m,m) = 2^{1/3} m^{5/3} + (1+o(1))·m.

**Proof.** By Theorem A1, LP_ladder = LP_budget. Scale w = t·m^{2/3} and
let the column-weight profile become a nonnegative measure ρ(t)dt with
moments A_j = ∫ t^j ρ; the constraints become A₀ ≤ 1 (columns),
A₃ ≤ 2 (slots: C(w,3) → w³/6, B → m³/3), objective A₁·m^{5/3}.

*Upper bound (dual certificate).* With λ = (2/3)·2^{1/3},
μ = 1/(3·2^{2/3}):

    t ≤ λ + μ t³   for all t ≥ 0

(convexity: equality and tangency at t = 2^{1/3}; `part_D` grid check,
min slack 0 at t = 1.259921), hence A₁ ≤ λA₀ + μA₃ ≤ λ + 2μ = 2^{1/3}.

*Lower bound.* The atom ρ = δ_{2^{1/3}} of mass 1 (m columns of weight
2^{1/3}m^{2/3}, i.e. the integer waterfill's adjacent pair {w₀, w₀+1},
w₀ ~ (2m²)^{1/3}) is feasible and attains 2^{1/3}·m^{5/3}(1−o(1)).

*Second order (exact rational LP, `part_C`).* The budget LP optimum is
the adjacent-pair solution saturating columns and slots; computing it in
exact arithmetic:

    m        E_LP/m^{5/3}    (E_LP − 2^{1/3}m^{5/3})/m
    100      1.294090835     0.7362
    800      1.269958735     0.8650
    12800    1.261650396     0.9463
    10^6     1.260019791     0.9874  → 1

so E_LP = 2^{1/3}m^{5/3} + m·(1+o(1)); the excess over 2^{1/3} decays
like m^{−2/3} (doubling ratios 0.676 → 0.637 → 2^{−2/3} = 0.630). ∎

**Answer to the task's trichotomy: c_L = 2^{1/3}, the budget-only
value. The ladder is asymptotically blind at the diagonal — and by
Theorem A1 it is *exactly* blind there at every finite size too.**
Distance toward the truth 1: the ladder closes 0% of the
(2^{1/3} − 1)·m^{5/3} gap.

---

## Theorem A3 (per-point ladder is equally blind). **PROVEN**

Task 1 asks for degree profiles as densities; the per-point form is the
strongest degree-visible relaxation, and it too gives 2^{1/3}.

Scale per point x: d_x = δ_x·m^{2/3} (blocks through x),
y_x = η_x·m^{4/3} (excess through x). At level k = s·m^{2/3} the
per-point ladder (k−1)y_x + μ_k d_x ≤ 2C(m−1,2) becomes, per point,

    s·η_x − (s²/2)·δ_x ≤ 1  for all s > 0   ⟺   η_x² ≤ 2δ_x.

The homogeneous Brown-scale profile — every column of weight
2^{1/3}m^{2/3}, every point with δ_x = 2^{1/3}, η_x = t̄·δ_x = 2^{2/3} —
satisfies every per-point ladder constraint **with equality**:
η² = 2^{4/3} = 2δ. It also saturates the global slot budget (A₃ = 2).
Hence the per-point ladder LP admits the same optimizer and its value is
2^{1/3}m^{5/3}(1+o(1)); the dual certificate of A2 applies verbatim
after summing over points. ∎

**Mechanism.** The ladder's only bite is the Jensen/chord gap
C(w−1,2) − chord_k(w), which vanishes for weights concentrated on the
adjacent pair {k, k+1}. The budget optimum at the diagonal is
weight-homogeneous (adjacent pair at w* ~ (2m²)^{1/3} = 2^{1/3}m^{2/3},
confirming the task's stated scaling), so every ladder level is
degenerate-tight on it. The ladder punishes only weight-*inhomogeneous*
profiles — which is exactly why it acts in the deep band (heavy blocks
mixed with weight-3 fills, where its floored/per-point refinements give
the O(1) gains of Theorem 12) and cannot act at the corner.

---

## Theorem A4 (floors and congruences cannot rescue it) — and where the
## class boundary REALLY is. **PROVEN, corrected in-session**

Any refinement that lowers the ladder/slot RHS by O(m) — per-point
integrality floors (Lemma C's double floor, J_w caps summed over
points), leave congruences (Lemma E/Theorem F type), or any O(m)-sized
RHS shaving — changes the diagonal LP value by O(m · ∂E/∂B) =
O(m · C(w*,2)^{-1}) = O(m^{−1/3}) → 0. Precisely: within the class of
per-point inequalities IMPLIED BY THE SLOT CAPACITY
Σ_{B∋x} C(w_B−1,2) ≤ 2C(m−1,2) — whose extreme valid linearizations
are exactly the ladder chords — plus congruence floors, the diagonal
constant is immovably 2^{1/3}: implied constraints cannot beat their
generator, and the generator's optimum is the homogeneous atom on
which every chord is tight. **The task's named toolset — ladder +
mixed congruences — provably cannot beat 2^{1/3}·m^{5/3}.**
(Care with wider phrasings: "any valid per-point linear inequality" is
NOT capped at 2^{1/3} — level-restricted count caps derived from
Theorem M1, e.g. #{blocks through x at weight ≈ u} ≤ (1+o(1))M²/u²,
are linear, valid, and stronger; but they are consequences of the
second-moment mechanism, not of slot capacity.)

**Correction notice (honesty bar).** A first draft of this section
claimed a stronger "blindness barrier": that NO per-point inequality of
any kind could beat the capacity RHS, via local saturation by
near-perfect 2-fold pair packings. That argument is valid only for
fixed block sizes (nibble scale); at the relevant scale w ~ m^{2/3} the
local saturation FAILS, and the failure is a theorem: the pairwise
intersections of the links through a point form a partial linear space,
whose Fisher-type pair budget the budget-optimal profile overfills.
This yields a valid per-point SECOND-MOMENT inequality
(η² ≤ √2·δ, beating the ladder's η² ≤ 2δ) and, chained with
Cauchy–Schwarz, the elementary bound

    z(m,m;3,3) ≤ 2^{1/6}·m^{5/3}(1+o(1)) = 1.1225·m^{5/3}(1+o(1)),

with single-level configurations pinched all the way to the Brown
constant 1. Full statement, proof, and three-way verification:
**`second_moment.md`** (Theorems M1–M3). Summary of the corrected
landscape:

    linear degree class (ladder + congruences):  2^{1/3}  — blind (A1–A3)
    + link second moments (elementary, M2):      2^{1/6}  — mixed configs
    + single-level restriction (M3):             1        — Brown constant
    Füredi's theorem (global):                   1        — all configs

Theorem 12's "corner obstructions are provably invisible to degree
accounting" survives only for degree-LINEAR accounting; second-order
degree statistics DO see the corner, and see it elementarily.

---

## Status labels

- Theorem A1: PROVEN (3-line proof + exact-integer verification to
  k, w ≤ 3000 + LP cross-check at 8 cells).
- Theorem A2: PROVEN (dual certificate, exact tangency; exact-rational
  finite LP values to m = 10⁶; c_L = 2^{1/3} = 1.259921049894873…).
- Theorem A3: PROVEN (scaling argument; same certificate).
- Theorem A4: PROVEN for the linear-degree + congruence class (the
  task's toolset); the first-draft universal barrier was WRONG and is
  superseded by second_moment.md (M1–M3) — recorded here per the
  honesty bar.
- Numbers 136/150/165/180 (ladder = WF at diagonal): VERIFIED.

Comparison line requested by the task:

    c_L = 2^{1/3} ≈ 1.2599   (this ladder — proven exact)
    budget-only  = 2^{1/3}   (identical — Theorem A1)
    2^{1/6} ≈ 1.1225         (elementary second-moment rung — the
                              in-session headline; second_moment.md)
    truth        = 1         (Brown + Füredi; see brown_measurements.md)

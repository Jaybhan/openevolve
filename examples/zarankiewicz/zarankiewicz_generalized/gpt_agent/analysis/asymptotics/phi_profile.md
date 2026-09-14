# The supply-density function φ: exact asymptotic form, finite profile, half-budget law

Asymptotic analyst, 2026-07-30. Task 3. Machine companions:
`phi_profile.py` (table + onset analysis, no solvers), `phi_milp.py`
(the program's entire 2-run solver budget — outcome recorded in §5),
data: `phi_table.csv`.

Notation: 𝔇_w(m) = D₂(m,w,3) = max multiset of weight-w blocks on m
points covering every triple ≤ 2 (the level-w supply). Level in Brown
units c = w/m^{2/3}; supply density φ̂ = 𝔇_w(m)/m; budget efficiency
ψ = 𝔇_w(m)·C(w,3)/(2C(m,3)) ∈ [0,1].
φ(c) := limsup_m 𝔇_{⌈c·m^{2/3}⌉}(m)/m (diagonal_limit.md §2).

---

## 1. Theorem P1 (the step law — φ determined everywhere). **PROVEN†**

    φ(c) = c⁻³   for 0 < c ≤ 1        (in particular φ(1) = 1),
    φ(c) = 0     for c > 1;  quantitatively, for fixed c > 1 at most
                 (2+o(1))·m^{2/3}/(c−1) weight-⌈cm^{2/3}⌉ columns exist.

† PROVEN modulo two citations: Brown's construction (re-verified
exhaustively here at six sizes, brown_measurements.md) and Füredi's
printed rectangular bound Z(m,n;3,3) ≤ m·n^{2/3} + 2n^{4/3} + m
[FS survey Thm 3.19 / Füredi CPC 1996; numerically dominates all 161
exact cells, theory.md C4b]. Everything else is self-contained.
**In-session upgrade (second_moment.md, M3): the upper bound
φ(c) ≤ c⁻³ (all c > 0) is now ELEMENTARY — the link second-moment
pinch d_x ≤ (1+o(1))M²/u², valid whenever u ≫ √m, applied to the
(single-level by definition) supply family. The Füredi citation is
needed only for the hard zero beyond c = 1 (the collapse to
O(m^{2/3}/(c−1)) columns) and for Corollary P1b.**

**Proof.** A supply configuration is an m×D K₃,₃-free matrix, D = φm
columns of weight w = cm^{2/3}, E = cφ·m^{5/3}.

*Upper bound, all c (row orientation).* E ≤ mD^{2/3} + 2D^{4/3} + m.
With D = O(m) (budget gives φ ≤ 2/c³), the second term is O(m^{4/3}):
cφ ≤ φ^{2/3} + o(1), i.e. φ ≤ c⁻³ + o(1).

*Upper bound, c > 1 (column orientation — the collapse).* Transposing,
E ≤ D·m^{2/3} + 2m^{4/3} + D, so (c−1)·D·m^{2/3} ≤ 2m^{4/3} + D, giving
D ≤ (2+o(1))·m^{2/3}/(c−1). Hence φ(c) = 0.

*Lower bound, c ≤ 1 (truncated Brown).* Fix ε > 0; pick a prime q with
q³ ∈ [(1−2ε)m/c³, (1−ε)m/c³] (PNT), build Brown's q³×q³ matrix, keep m
of its rows (K₃,₃-freeness is hereditary). Column weights are
hypergeometric with mean (1−1/q)·m/q = c·m^{2/3}(1+Θ(ε)) > cm^{2/3} and
SD O(√w): all but o(q³) columns exceed ⌈cm^{2/3}⌉; trim them to exact
weight (trimming only removes coverage). This yields (1−o(1))·q³ ≥
(1−2ε)m/c³ legal columns; ε ↓ 0 gives φ(c) ≥ c⁻³. ∎

This CLOSES diagonal_limit.md's open item ("φ(c) = ? for
c ∈ (0,1) ∪ (1, 2^{1/3}]"): the old caps φ ≤ 2/c³ (budget) and
φ ≤ 1/c (c ≥ 1) are both far from sharp — the truth is a hard step,
c⁻³ up to the Brown point, total collapse beyond it. The Level
Formula's diagonal ingredient sup{c : φ(c) ≥ 1} = 1 is unchanged.

**Corollary P1a (coverage-1 law).** Maximum supplies at any fixed level
c ∈ (0,1] have average triple-coverage φc³ = 1 + o(1) — exactly half
the K₃,₃ capacity 2. The pairwise cap ("every triple ≤ 2") supports
only average 1 at scale; Brown's square achieves average → 1 while
still *touching* 2 (max T = 2 measured at every q, brown_measurements
§2). The 2^{1/3} → 1 descent of the diagonal constant is exactly this
factor 2 of budget waste, cube-rooted.

**Corollary P1b (rectangular density).** For m ≪ n ≪ m^{3/2}:
z(m,n;3,3) = (1+o(1))·m·n^{2/3} (LB: level c = (m/n)^{1/3} supply from
P1; UB: Füredi row orientation). The KST constant 2^{1/3} improves to 1
throughout this range. [Novelty UNVERIFIED — truncated Brown is
presumably folklore; flagged for the literature referee.]

---

## 2. Theorem P2 (the half-budget law and its crossover). **PROVEN†**

For m^{1/2} ≪ w ≤ m^{2/3}:

    𝔇_w(m) = (1+o(1))·(m/w)³ = (1/2 + o(1)) · 2C(m,3)/C(w,3),

i.e. budget efficiency ψ → 1/2. For *fixed* w, GKLO/Hanani give ψ → 1
(design zone, L_min-periodic corrections — Theorem 11 / L_min tables).
The crossover scale w ≍ m^{1/2} is exactly where Füredi's second-order
term takes over: 2D^{4/3} ≥ mD^{2/3} ⟺ D ≥ (m/2)^{3/2}·…, and
D ≍ (m/w)³ crosses m^{3/2} precisely at w ≍ m^{1/2}. So **the
second-order term of Füredi's bound is what controls where packing
efficiency halves** — the same term that rules the diagonal
second-order window (brown_measurements.md §4).

Intermediate regime ω(1) ≤ w ≤ O(m^{1/2}): OPEN (equivalent to
near-perfect 2-fold packing existence with growing block size; expected
ψ → 1 for w = m^{o(1)} by nibble-type arguments, unproven here).

† The LB is the truncated Brown of P1 (valid for all w ≤ m^{2/3}); the
UB has two independent proofs: (i) Füredi row-orientation, whose
D^{4/3} term is o(main) exactly when w ≫ m^{1/2}; (ii) ELEMENTARY
(second_moment.md M3): the link second-moment pinch, whose validity
condition Ī = u²/M ≫ 1 is the SAME window w ≫ √m — the two proofs
activate at the same scale, which is no accident: both are pair-space
saturation statements.

---

## 3. The finite profile (all exact workspace data) — `phi_table.csv`

82 supply points (51 EXACT, 25 BRACKET, 6 Brown LOWER): w = 4 (T₃,₃, m = 3..18 Tan + 19, 23, 27 workspace),
w = 5..10 (supply_status.csv EXACT + brackets), Brown rows (φ ≥ 1 at
c = 1 − 1/q, m = q³); no MILP additions (see §5). Digest by zone:

**(a) Design zone (c ≲ 0.95).** ψ ∈ [0.982, 1.000] at w = 4 for every
m ≤ 27; exact w = 5, 6 points (m ≤ 9): ψ = 0.83–1.0. Every deviation
from ψ = 1 equals the known congruence leave (L_min spectrum, Theorem
11) or a completion obstruction (Dehon-type, spectrum.md) — none needs
the Füredi mechanism.

**(b) The step window (0.95 ≲ c ≲ 1.35).** The finite table stays
budget-true: (8,4) at c = 1.000 has ψ = 1 (doubled SQS(8) — perfect at
the exact step location); (12,6) at c = 1.14: ψ = 1 (Hadamard 3-design);
(13,6) c = 1.09: ψ = 0.91; (9,5) c = 1.16: ψ = 0.83. **The asymptotic
step (φ → 0 for c > 1) is entirely invisible at m ≤ 27 — and §4
quantifies why it must be.**

**(c) Wedge/collapse (c ≳ 1.4 at these m).** 𝔇 = 2–4 = O(1)
(complement-zone caps, wedge theorem) — the finite shadow of the c > 1
collapse. (For fixed c > 1 the true asymptotic collapse is to
Θ(m^{2/3}) columns, not O(1); the O(1) wedge is the finite-m boundary
w ≥ ~2m/3, which recedes to c → ∞ as m grows.)

**(d) Brown rows — the only finite points beyond the onset.**
φ = 1 at c = 1 − 1/q with ψ = 0.295 (q=7) → 0.435 (q=23) →
(1−1/q)³/2 → 1/2: the *only* finite data that exhibits the erosion,
because only the Brown sizes are large enough (§4).

---

## 4. The onset: why the finite table sits on the budget curve

First-crossing Füredi cap D_F(m,w) = max{D₀ : wD ≤ mD^{2/3}+2D^{4/3}+m
for all D ≤ D₀} vs the exact budget ⌊2C(m,3)/C(w,3)⌋ (`phi_profile.py`):

    erosion onset (first m with D_F < budget):
      c = 1.0:  m = 1792   (w = 148: D_F = 3609 < 3617 = budget)
      c = 0.9:  m = 3351
      c = 0.8:  m = 6778
      c = 0.7:  m = 15199
    diagonal crossover (Füredi UB < budget LP at n = m):  m* = 470.

So NO cell with m ≤ 27 can show a Füredi deficit — every finite deficit
is congruence/completion-classifiable (as observed in (a)) — and the
z-table's "budget-tracking regime" (diagonal_limit.md §3) is quantified:
it must persist to m* = 470 and cannot be arbitrated below it.

Measured erosion along the Brown sizes (m = q³, w = q²−q):

    q      budget_D    D_F        D_F/budget
    7      1161        inactive   —
    11     3633        5911       (1.63 — inactive)
    13     5687        6355       (1.12 — marginal)
    17     11909       9939       0.835   ← first active size
    19     16269       12576      0.773
    23     27964       19571      0.700
    47     221785      127944     0.577
    101    2123664     1130327    0.532   → 1/2  (half-budget law)

**The half-budget law is not just proven — it is numerically measured
approaching 1/2 along the only finite family that reaches the scale.**

---

## 5. Which closed form survives? (the task's model test)

| candidate | verdict |
|---|---|
| φ = min(1, 1/c) (diagonal_limit's caps read as a shape) | REFUTED both sides: φ(c) = c⁻³ > 1 for c < 1 (truncated Brown); φ(c) = 0 < 1/c for c > 1 |
| φ = min(1, c^{−α}) any α | REFUTED (same: φ > 1 below 1; hard zero above 1 — no smooth min-form fits a step) |
| φ = 2/c³ (budget) | asymptotically REFUTED (off by exactly 2 on (0,1], ∞ off beyond); as a FINITE model at m ≤ 27: excellent (§3a) |
| **φ = c⁻³·[c ≤ 1] (step law)** | **PROVEN asymptotic truth (P1)** |

**Best-supported exact finite form** (reproduces the finite table AND
the Brown measurements simultaneously):

    𝔇_w(m) = min(  (2C(m,3) − L_min(w, m)) / C(w,3)   [congruence budget]
                 ,  D_F(m, w)                          [first-cross Füredi]
                 ,  wedge/complement caps  )           [spectrum.md]
              − sporadic completion defects (Dehon-type, finite list/level)

with the three regimes exchanging bindingness exactly at the measured
onsets. Deviations of this model on the 51 EXACT cells: the w = 4 row
is exact by construction (L_min anchored); at w ≥ 5 every deviation
from the first term is a documented completion obstruction
((10,5) ≤ 21, (11,5) ≤ 31, (12,7) ≤ 10, wedge cells) — all listed in
spectrum.md/decisions.csv; D_F binds on NO exact cell (onset §4) —
consistency, not tautology: it *does* bind on the Brown rows, where the
first term would wrongly allow 1.6–2× more columns.

**Solver-budget outcome (honest failure record).** The program's two
budgeted MILP runs targeted 𝔇₅(11) (c = 1.011) and 𝔇₅(12) (c = 0.954)
— the exact points straddling the step — via scipy/HiGHS with 1500 s
limits each. The harness terminated the process at ~40 min wall; solve
1 had finished its limit but ALL results were lost to stdout buffering
before the end-of-script JSON write. Budget consumed, no rigorous
update extracted; the standing brackets 𝔇₅(11) ∈ [23,31],
𝔇₅(12) ∈ [30,38] are unchanged. `phi_milp.py` now checkpoints after
every solve (post-mortem note in the file) so a future session's runs
cannot repeat the failure. These two cells remain the highest-value
solver targets for the φ-step (next session).

---

## 6. Falsifiable predictions

1. 𝔇₁₄₈(1792) ≤ 3609 < 3617 = budget — the first Füredi-forced supply
   deficit [PROVEN†]; conjectured value ≈ (m/w)³ = 1776 (half-budget).
2. Along Brown sizes: 𝔇_{q²−q}(q³) = (1 + 3/q + o(1/q))·q³
   (CONJECTURE: truncated-Brown-from-the-next-prime is optimal).
3. The m ≤ 27 exact table will never show a deficit unexplained by
   congruence + completion classification [PROVEN† via onset].
4. ψ(m, ⌈m^{0.6}⌉) → 1/2 (inside the half-budget window since
   0.6 ∈ (1/2, 2/3)) — a concrete future-ILP target family.

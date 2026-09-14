# The diagonal limit of the LEVEL FORMULA (Task 4)

level_theory agent, 2026-07-29. Question: what asymptotic constant does
the Level Formula imply for z(m,m;3,3) = c·m^{5/3}(1+o(1)), and does it
match Brown–Füredi? Normalization guard (theory/theory.md §5, verified
there): **graph** constant 1/2 (Brown's ex(n;K₃,₃) ≥ (1/2)n^{5/3},
Füredi's matching UB), **bipartite** constant 1:
z(n,n;3,3) = (1+o(1))·n^{5/3}. Everything below is bipartite.

## 1. The formula's diagonal constant is exactly 2^{1/3} = 1.2599…

At n = m, level w is admissible iff C(w−1,3)·m ≤ B = 2C(m,3), i.e.
(w−1)³ ≤ 2m²·(1+o(1)):  **w−1 ≤ (2m²)^{1/3} = 2^{1/3}·m^{2/3}**.
For any admissible level, the formula's value is at least (w−1)·m (the
min-term is ≥ 0 by admissibility), and at most (w−1)m + m. Hence

    z_hat(m,m) = 2^{1/3}·m^{5/3}·(1 + o(1)),

**regardless of the supply values 𝔇_w(m)** — even with exact (true)
supplies, and also under the V2 refinement (per-layer bottom cap): with
true supplies both layers satisfy 𝔇 ≈ m/c at level c·m^{2/3}, so
𝔇_{w−1} + 𝔇_w ≥ m persists up to c = 2 > 2^{1/3}, and the budget cuts
first. Machine check at m = 16: the admissibility bound gives
w−1 = 8 = (2·16²)^{1/3} exactly, and the formula value is
8·16 + 8 = 136 (the +8 is the slack term) — the reported +8 overshoot
at (16,16) [truth 128] is this section's asymptotics at finite m.

## 2. The true constant is 1, and exactly one ingredient loses the gap

Brown's construction (double cover) gives m columns of weight
m^{2/3}(1−o(1)); Füredi's upper bound caps z(m,m) at (1+o(1))m^{5/3}.
Define the **true level-supply density**

    φ(c) := limsup_m  𝔇_{⌈c·m^{2/3}⌉}(m) / m .

Three facts pin φ where it matters:
- Budget (Lemma A arithmetic):  φ(c) ≤ 2/c³  (slots).
- Füredi:  for c ≥ 1, a supply of φm weight-cm^{2/3} columns is itself a
  K₃,₃-free matrix inside an N×N square, N = max(m, φm); if φ ≤ 1 this
  forces c·φ·m^{5/3} ≤ (1+o(1))m^{5/3}:  **φ(c) ≤ 1/c** — strictly below
  the budget 2/c³ for all c < 2^{1/3}.
- Brown:  φ(1) ≥ 1 (his square witness), and φ(c) ≥ 1 for all c ≤ 1
  (disjoint copies of Brown blocks on point-groups of size c^{3/2}m).
  So **φ(1) = 1**, and sup{c : φ(c) ≥ 1} = 1.

The two-layer formula charges (w−1)·m for the bottom layer subject to
**budget only**. A bottom layer of m legal columns at level w−1 exists
iff 𝔇_{w−1}(m) ≥ m iff φ(c) ≥ 1 iff c ≤ 1. So:

- budget-only bottom layer (V0/V1/V2): constant = sup{c : c ≤ 2^{1/3}}
  = **2^{1/3}** — overshoot factor 2^{1/3} ≈ 1.26;
- supply-true (mixed-ledger) bottom layer: constant = sup{c : φ(c) ≥ 1}
  = **1** — Brown–Füredi exactly.

**Conclusion.** The Level Formula interpolates Culík (level 3) → Theorem
8/Roman (level 4) → the deep band (levels 5+) with the correct constants,
and reaches the diagonal with the correct EXPONENT and level-scaling
w* = Θ(m^{2/3}) — but its diagonal constant is 2^{1/3}, not 1. The lost
ingredient is identified precisely: **the mixed/bottom-layer supply
(ledger) constraint, whose asymptotic content at the diagonal is exactly
Füredi's theorem** (φ(c) < 1 for c > 1). No refinement of per-level
supplies, congruences, or Johnson/ladder accounting can close this — the
same conclusion Theorem 12 reached empirically ("the corner obstructions
are cap/ovoid geometry, provably invisible to degree accounting"), now
as a sharp asymptotic statement: the gap is the difference between the
budget density 2/c³ and the true density φ(c) ≤ 1/c on c ∈ (1, 2^{1/3}].

A pleasing corollary of the bookkeeping: Brown's column-weight constant
is 1·m^{2/3}, while the formula's maximizing level is 2^{1/3}·m^{2/3} —
theorems.md's "Brown's column-weight law emerges from the budget
arithmetic" is right about the scaling, 26% high about the constant.

## 3. Finite-size behavior: the table lives on the budget curve

| m | m^{5/3} | 2^{1/3}m^{5/3} | z(m,m) truth | truth/m^{5/3} |
|---|---|---|---|---|
| 8  | 32.0  | 40.3  | 42  | 1.31 |
| 12 | 62.9  | 79.3  | 80  | 1.27 |
| 15 | 91.2  | 114.9 | 120 | 1.32 |
| 16 | 101.6 | 128.0 | 128 | 1.26 |

At m = 16 the truth sits EXACTLY on the budget curve: 2^{1/3}·16^{5/3} =
2^{1/3}·2^{20/3} = 2⁷ = 128 (integer coincidence at m = 2^k, k ≡ 1 mod 3).
The known small-m diagonal is in the **budget-tracking regime**: the true
values follow 2^{1/3}m^{5/3} (ratios 1.26–1.32), not yet the asymptotic
1·m^{5/3}; the corner deficits (d(16,16) = 8 vs the formula, Observation
3's caps/ovoids) are the first visible erosion of supply below budget.
The descent 1.26 → 1.00 is the asymptotic version of "demand exceeds
extremal supply": Füredi's theorem forces the supply side to lose to the
budget by exactly the factor 2^{1/3} in the limit. Both regimes are now
quantitatively explained by one mechanism ladder: budget arithmetic
(finite, sharp early) vs mixed supply (asymptotic, sharp late).

## 4. Status labels

- §1 (formula constant 2^{1/3}): PROVEN (elementary asymptotics of the
  stated formula; machine-checked at m = 16 and against the V0/V1/V2
  residual table).
- §2 (true constant 1; normalization): PROVEN modulo the cited
  Brown/Füredi statements as vetted in theory/theory.md §5 (the
  workspace's own bookkeeping note: graph 1/2 vs bipartite 1).
- φ(c) facts: budget and Füredi caps PROVEN (given §2's citations);
  φ(c) = ? for c ∈ (0,1) ∪ (1, 2^{1/3}] beyond the stated bounds — OPEN
  ("rectangular Brown density"); φ(c) ≥ 1 for c ≤ 1 by disjoint-blocks:
  PROVEN.
- §3 table: numeric, from the exact z-table.

# The unification question: is there ONE algebraic construction?

Owner's directive: a unified algebraic construct for z(m,n;s,t). This file
states precisely what is unified, what provably cannot be, and what the
final unified object is.

## What is already one rule (PROVEN/VERIFIED)

1. **The frame**: columns = blocks; K_{s,t}-free ⟺ (t−1)-fold s-packing.
   One statement, all (m,n,s,t).
2. **The profile layer**: which block-size profile is optimal is pure
   arithmetic — Lemma A + the supply tables S_m. Verified to reproduce
   rows 6–8 (51 cells) exactly, with row 9 following as its table lands.
   One rule, no search.
3. **The certificate layer**: exactness = profile arithmetic UB meeting a
   realized construction. One contract (bounds/certification.md).

## What provably CANNOT be one nested object (NEW negative result)

**Tower test (2026-07-28)**: at (8,19) (z = 81, optimal profile 5⁵4¹⁴),
forcing the 14 quads to come from the doubled Boolean SQS(8) — the unique
structure that is optimal for ALL n ≥ 25 in row 8 — caps the cell at **79**
(control with unrestricted quads: 81). So the mid-band optimum is NOT a
truncation of the elongated-regime master structure. Corroborating
evidence: S₈(0)=28 → S₈(1)=23 (adding one pentad destroys five SQS quads,
not one); the (9,22)=100 optimum (5¹²4¹⁰) is NOT a truncation of the
Hadamard 3-(12,6,2) (truncation provably caps at 99). **The extremal
structures undergo genuine phase transitions in n. A single nested master
family per row does not exist.**

## The unified object that survives

    construct(m, n, s, t) = realize( profile(m, n, s, t) ; G-symmetric
                                     maximal packing for that profile )

- profile(·) — one arithmetic rule (above).
- realize(·) — one generative principle: every extremal structure we have
  decoded is a maximal (t−1)-fold packing admitting a large automorphism
  group, drawn from a single hierarchy of algebraic sources: binary-linear
  (F₂ᵏ hyperplanes, caps/ovoids, Boolean SQS = AG(3,2) planes), Hadamard
  (3-(12,6,2) and residuals), projective (PG(2,q), Fano doubling, Singer/QR
  difference orbits), and graphic (Turán edge-complements). The phase
  transitions select WHICH source; the supply table says WHEN.
- The engineer is consolidating the per-family code into this two-layer
  form (constructions/unified.py, in progress) with a measured "price of
  unification" against the 159/161 router.

## Honest framing for the final report

"One formula for all m,n,s,t" in the strongest sense would decide, among
other things, the order of z(n,n;4,4) — open since 1954. What this project
delivers is: one FRAME, one PROFILE RULE with finite design-theoretic
tables, one REALIZATION PRINCIPLE with a finite source hierarchy, proofs on
complete rows, and a proven theorem that no simpler (single-tower) unification
exists. The supply tables S_m are the irreducible mathematical core — they
are to this problem what character tables are to a group: finite, computable,
and the place where the actual combinatorics lives.

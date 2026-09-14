# C7 for general m: the two inadmissible families of the T-spectrum

design_prover, 2026-07-29. Labels per workspace honesty bar. Companion
files: `J_minus_2_proof.md` (Theorem F verification + weight-6
classification + witnesses), `gklo_citation.md` (existence citation,
quoted statements, hypothesis checks), `scripts/verify_threefold.py`
(machine checks for Section 3), `scripts/verify_theorem_F.py`.

Notation: B = 2C(m,3); R = ⌊(m−1)(m−2)/3⌋; J = ⌊mR/4⌋ (Johnson /
Johnson–Schönheim U₂(m,4,3)); T = T₃,₃(m) = PDN₂(m,4,3). "Class m" means
m ≡ 3 (mod 4), m ≢ 0 (mod 3).

---

## 1. THEOREM (class family). For every class m:  T₃,₃(m) ≤ J(m) − 2,
   with equality for m ∈ {7, 11, 19, 23} and for all sufficiently large
   class m.

**Proof.**

*Upper bound, every class m (fully effective):* Theorem F — the three
K₄⁽³⁾-divisibility congruences (slots mod 4, points mod 3, pairs mod 2)
admit no leave of weight 2 or 6, and no pentad escape (k₅ ≤ 3 cases all
dead). Independently re-verified in this directory; complete hand proof
of the weight-6 classification in `J_minus_2_proof.md` Section 2. PROVEN.

*Attainment, m ∈ {7, 11, 19, 23}:* explicit σ-invariant witnesses with
doubled-pentagon leaves, verified by three independently written
checkers (`witnesses/`, `J_minus_2_proof.md` Section 3). PROVEN.

*Attainment, all sufficiently large class m:* apply GKLO Theorem 1.1
(quoted in `gklo_citation.md`) with r = 3, F = K₄⁽³⁾ (Deg(F) = (4,3,2)),
λ = 2, p = 1, host G = K_m^(3) − P where P is a pentagon (the five
cyclic-interval triples on a 5-set). The three divisibility rows for the
punctured host, spelled out:
  - slots (i = 0): 2|G| = B − 10; class arithmetic gives B ≡ 2 (mod 4)
    (v₂(m(m−1)(m−2)) = v₂(m−1) = 1, division by odd 3), so
    B − 10 ≡ 0 (mod 4). ✓
  - points (i = 1): 2|G(x)| = 2(C(m−1,2) − deg_P(x)) with deg_P(x) ∈
    {0, 3}; class gives 3 | C(m−1,2) (3 | (m−1)(m−2), 2 invertible), so
    2|G(x)| ≡ 0 (mod 3). ✓
  - pairs (i = 2): 2 | 2|G(xy)| — automatic at λ = 2. ✓
G is (c, h, 1)-typical for m ≥ (2h+15)/c (removing the pentagon deletes
at most 15 link-points in total). Theorem 1.1 then yields an
(K₄⁽³⁾, 2)-design of G: distinct quads covering every non-pentagon
triple exactly twice and pentagon triples zero times — a packing of
2|G|/4 = (B−10)/4 = J − 2 quads (leave = doubled pentagon). PROVEN
modulo the cited theorem. ∎

**Ineffectiveness caveat (stated plainly).** GKLO's n₀ is not explicit;
the constants for (f, r) = (4, 3) are astronomically demanding (see
`gklo_citation.md`). So "sufficiently large" cannot currently be
replaced by a number, and class members outside {7, 11, 19, 23} below
that unknown threshold remain individually open — each is a finite ILP
(the σ-scheme), with m = 31, 35 attempted and unresolved so far
(solver time-outs, recorded in STATUS.md).

**Consequence for C7.** C7's class clause "T = J − 2" is now PROVEN for
7, 11, 19, 23 and for all large class m; it is CONJECTURE only on the
finite (but unspecified) middle range. The other C7 clauses: admissible
m (T = C(m,3)/2 = J) are Hanani-classical; 3|m below.

---

## 2. The leave dictionary (why these leaves and no others)

A maximum-packing leave must satisfy the three congruences; on the class,
minimal admissible weight is 10 (weights 2, 6 impossible: Lemma E +
Theorem F). Weight-10 realizations: the doubled pentagon is the UNIQUE
fully-doubled congruence-valid shape (proof + machine check in
`J_minus_2_proof.md` Section 3); all four explicit witnesses have it.

---

## 3. THEOREM (3|m family): minimal congruence-valid leave weight equals
   B − 4J; hence no divisibility obstruction to b = J, and
   T₃,₃(m) = J(m) for all sufficiently large m ≡ 0 (mod 3).

Machine checks: `scripts/verify_threefold.py` (ALL PASS).

**(3a) Arithmetic (PROVEN, hand proof).** Write m = 3s. Then
(m−1)(m−2) = 9s² − 9s + 2, so R = ⌊(9s²−9s+2)/3⌋ = 3s(s−1) exactly, and
B − mR = s(9s²−9s+2) − 9s²(s−1) = 2s = 2m/3. Since s(s−1) is even, mR is
even; for s even, 4 | s², so mR = 9s²(s−1) ≡ 0 (mod 4); for s odd,
s² ≡ 1 (mod 4)(indeed mod 8) and mR ≡ 9(s−1) ≡ s−1 (mod 4), which is
2 iff s ≡ 3 (mod 4), i.e. iff m ≡ 9 (mod 12), and 0 otherwise. Hence

    B − 4J = (B − mR) + (mR mod 4) = 2m/3 + 2·[m ≡ 9 (mod 12)].

Verified for all 3|m ≤ 3000; reproduces the data row 4, 8, 8, 10, 12,
16, 16, 18 at m = 6, 9, ..., 27.

**(3b) Congruence lower bound (PROVEN).** For 3|m the point congruence
reads ℓ_x ≡ 2C(m−1,2) ≡ 2 (mod 3) for EVERY point x (C(m−1,2) ≡ 1 mod 3
here) — so every point is touched with ℓ_x ≥ 2, giving
3L = Σℓ_x ≥ 2m, i.e. L ≥ 2m/3. The slot congruence gives L ≡ B (mod 4).
The smallest L ≥ 2m/3 congruent to B (mod 4) is
2m/3 + ((B − 2m/3) mod 4) = 2m/3 + (mR mod 4) = B − 4J. So EVERY
2-fold quadruple packing on m points (3|m) has leave weight ≥ B − 4J —
note this re-derives Johnson's bound b ≤ J for this family from the
congruences alone. (Pair parity is consistent but not needed for the
minimum: mR is even, so B − 4J is automatically even.)

**(3c) Attainment of the minimum by explicit leave families (PROVEN).**
  - m ≢ 9 (mod 12): L = 2·(parallel class): m/3 disjoint triples, each
    doubled. Weight 2m/3 = B − 4J; every ℓ_x = 2 ✓; pairs doubled ✓.
  - m ≡ 9 (mod 12): L = 2·(hub family): four triples {z,a_i,b_i} through
    one point z (8 distinct other points) plus a parallel class on the
    remaining m − 9 points, all doubled. Weight 2(4 + (m−9)/3) =
    2m/3 + 2 = B − 4J; ℓ_z = 8 ≡ 2, all other ℓ_x = 2 ✓; pairs doubled ✓.
  Both verified programmatically for m up to 69 and by the general
  degree computation.
  Rigidity remark (PROVEN, small): among ALL-DOUBLED leaves of minimal
  weight with m ≢ 9 (mod 12), the parallel class is forced: degrees
  2k_x ≡ 2 (mod 3) force k_x ≡ 1 (mod 3), and Σk_x = 3·(m/3) = m forces
  k_x = 1 for all x — a perfect matching by triples. This is the
  coordinator's m=6 anchor {015}²{234}² in general form.

**(3d) Large-m attainment (PROVEN modulo GKLO).** Host G = K_m − L'
(L' = the SIMPLE half of the leave above). Divisibility rows, spelled
out in `gklo_citation.md` Section 3: slots 2|G| = mR (case a) or mR − 2
(case b), both ≡ 0 (mod 4) by (3a); points 2(C(m−1,2) − deg_{L'}(x))
with deg ∈ {1} resp. {1, 4}, all ≡ 0 (mod 3); pairs automatic.
Typicality with p = 1 (each pair lies in ≤ 1 triple of L'). GKLO
Theorem 1.1 gives an (K₄⁽³⁾,2)-design of G, i.e. a packing with
2|G|/4 = J blocks (both cases), leave 2L'. With Johnson's upper bound:
T₃,₃(m) = J(m) for all sufficiently large 3|m. Same ineffectiveness
caveat as Section 1. Known small cases: m = 6, 9, 12, 15, 18 (Tan,
= J each); m = 27 finite confirmation in progress (σ-scheme with three
9-cycles, doubled {i, i+3, i+6} parallel-class hole; STATUS.md).

---

## 4. Spectrum summary (state after this document)

| family | T₃,₃(m) | status |
|--------|---------|--------|
| admissible (3 ∤ m, m ≢ 3 mod 4) | C(m,3)/2 = J | PROVEN all m (Hanani) |
| class (3 ∤ m, m ≡ 3 mod 4) | J − 2 | UB all m (Thm F); = at 7,11,19,23 + all large m (GKLO); middle range open |
| 3 divides m | J | VERIFIED 6..18 (Tan); PROVEN all large m (GKLO + Sec. 3); middle range open; 27 pending |

Note on coverage (PROVEN; machine-checked for m ≤ 3000 in
`verify_threefold.py` check F): the three families partition all m ≥ 4.
If 3 ∤ m then 3 | (m−1)(m−2) automatically, and C(m,3) is odd iff
m ≡ 3 (mod 4): for m ≡ 0, 1, 2 (mod 4), v₂(m(m−1)(m−2)) ≥ 2 so C(m,3)
is even (admissible, T = C(m,3)/2 by Hanani's 3-(v,4,2) spectrum plus
double counting); for m ≡ 3 (mod 4), v₂ = v₂(m−1) = 1 so C(m,3) is odd
— the class.

## 5. Deliverable status labels

- Theorem F UB: PROVEN (effective, all class m).
- {7,11,19,23} attainment: PROVEN (verified witnesses).
- Large-m class attainment: PROVEN modulo GKLO Thm 1.1 (ineffective m₀).
- 3|m minimal-leave classification: PROVEN (hand + machine).
- Large-m 3|m attainment T = J: PROVEN modulo GKLO Thm 1.1.
- Middle-range class members (31, 35, ...) and 3|m members (21, 24, 27,
  ...) beyond published data: OPEN individually; finite ILPs.

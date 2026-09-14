# The J−2 law on the class m ≡ 3 (mod 4), m ≢ 0 (mod 3):
# independent verification, completed hand proof, and attainment

design_prover agent, 2026-07-28.
Everything here is labeled PROVEN / VERIFIED / CONJECTURE / FAILED per the
workspace honesty bar. Scripts: `scripts/`; witnesses: `witnesses/`.

Throughout: m in the class means m ≡ 3 (mod 4) and m ≢ 0 (mod 3);
R = (m−1)(m−2)/3 (an integer for the class), B = 2·C(m,3) = mR,
J = J(m) = ⌊B/4⌋. A *packing* is a multiset of 4-subsets (quads) of [m]
with every 3-subset in at most 2 of them; T₃,₃(m) = D₂(m,4,3) is the
maximum size. A *heavy configuration* is a multiset of blocks of size ≥ 4
with every triple covered ≤ 2; its value is Q = Σ(w−3). Note any block of
size ≥ 3 automatically has multiplicity ≤ 2 (a triple inside it would
otherwise be covered 3 times).

---

## 1. Adversarial verification of Theorem F (coordinator's proof) — CONFIRMED

Theorem F (theorems.md): for every class m, every legal heavy configuration
has Q ≤ J − 2; in particular T₃,₃(m) ≤ J(m) − 2.

I re-derived every step from scratch and machine-checked each numerical
claim with my own code (`scripts/verify_theorem_F.py`, no code shared with
the coordinator's script). All checks PASS. Detailed audit:

**(a) Class arithmetic.** 3 | C(m−1,2) and B = mR ≡ 2 (mod 4), so
J = (B−2)/4. Re-derivation: m ≢ 0 (mod 3) makes 3 | (m−1)(m−2), and 2 is
invertible mod 3, so 3 | C(m−1,2); v₂(m(m−1)(m−2)) = v₂(m−1) = 1 since
m ≡ 3 (mod 4), and division by the odd 3 preserves it. Verified for all
333 class members m ≤ 2000. J(7,11,19,23,31,35) = 17, 82, 484, 885, 2247,
3272; J−2 = 15, 80, 482, 883, 2245, 3270.

**(b) Slot identity.** C(w,3) = 4(w−3) + e_w with e₄..e₇ = 0, 2, 8, 19 and
e_w strictly increasing (verified to w = 40). Hence slots = 4Q + 2k₅ + 8k₆
+ ..., and slots ≤ B. Q ≥ J−1 = (B−6)/4 forces excess ≤ 6: k₆ = k₇ = ... =
0 and k₅ ≤ 3; Q = J forces excess ≤ 2: k₅ ≤ 1. Also Q ≤ ⌊B/4⌋ = J always
(excess ≥ 0) — so only Q ∈ {J, J−1} need killing. No gap.

**(c) Point congruence (mod 3).** A quad through x contains C(3,2) = 3
triples through x; a pentad C(4,2) = 6; both ≡ 0 (mod 3). So
ℓ_x = 2C(m−1,2) − 3q_x − 6p_x ≡ 2C(m−1,2) ≡ 0 (mod 3) on the class.
Machine-verified as an exact identity on random mixed quad+pentad
configurations at m = 7, 11, 19.

**(d) Pair congruence (mod 2).** A quad through the pair {x,y} contains 2
triples through the pair, a pentad 3. So ℓ_xy = 2(m−2) − 2q_xy − 3p_xy ≡
p_xy (mod 2). Same machine verification. (For pure-quad packings: **every
pair-leave is even** — this is the second-order constraint the point
congruence misses.)

**(e) Case chain.** With L = leave weight = B − 4Q − 2k₅ (my re-derivation
agrees with the theorem's bookkeeping):
- Q=J, k₅=0: L = 2. Dead by (c): a weight-2 leave touches a point with
  ℓ_x ∈ {1,2}. (Enumeration: 0 point-valid weight-2 leaves.)
- Q=J, k₅=1: L = 0, so p_xy ≡ 0 (mod 2) for all pairs; the lone pentad's
  10 pairs have p_xy = 1. Dead.
- Q=J−1, k₅=0: L = 6. Dead by the finite classification (Section 2).
- Q=J−1, k₅=1: L = 4. Here pair parity reads ℓ_xy ≡ p_xy (mod 2), NOT
  ℓ_xy ≡ 0, so the classification must use the POINT congruence ALONE:
  by (c), ℓ_x ∈ {0,3}, Σℓ_x = 12, support exactly 4 points, and the unique
  such leave is K₄⁽³⁾ (all four triples of a 4-set once each; hand proof
  by doubled-triple count: two doubles give touched degrees in {2,4} ≢ 0
  (mod 3); one double D + two singles forces each D-point to lie in
  exactly one single while any single-point outside D has degree ≤ 2 —
  both ≢ 0 (mod 3) — so the singles would have to equal D; zero doubles
  gives 4 distinct triples with all degrees 3, i.e. K₄⁽³⁾. My enumeration
  confirms: the 15 labeled K₄⁽³⁾'s on 6 points are the ONLY
  point-congruence survivors at weight 4).
  K₄⁽³⁾'s pair-leaves are all even (0 or 2), so consistency forces p_xy
  even for every pair — vs. the lone pentad's ten odd pairs. Dead.
  [I flagged and closed this as a potential gap: had any non-K₄ shape with
  some ODD pair-leaves survived the point congruence, the pentad could
  have matched its parity pattern. None does — checked exhaustively.]
- Q=J−1, k₅=2: L = 2. Dead as in the first case ((c) is unaffected by
  pentads). Dead.
- Q=J−1, k₅=3: L = 0, so every pair lies in an even number of pentads.
  Equivalently the three pair-indicator vectors (edge sets of K₅'s) XOR to
  zero. If two pentads coincide, the third's K₅ would XOR to zero: dead;
  all distinct: pick a ∈ P₁∖P₂, b ∈ P₁∖P₃; if a ≠ b the pair {a,b} ⊆ P₁
  lies in P₁ only (p = 1, odd); if the only choices force a = b, i.e.
  P₁∖P₂ = P₁∖P₃ = {a}, then any pair {u,a} ⊆ P₁ has p = 1. Dead.
  Machine check: exhaustive over all pentad triples on ≤ 15 points
  (complete by support), via the XOR formulation: 0 survivors.

**Verdict: Theorem F is CORRECT and now doubly machine-verified.** One
presentational annotation for the record: the hand sketch in theorems.md
("doubled triples die instantly by pair parity") is loose — a doubled
triple contributes EVENLY to every pair, so pair parity alone does not kill
mixed doubled+single weight-6 leaves; the complete hand argument needs the
point congruence for the pure-doubled case and a link argument for the
mixed cases. The machine enumeration is complete and load-bearing either
way; the full hand proof is Section 2 below. — This annotation does not
affect the theorem's validity.

## 2. Complete hand proof of the weight-6 classification — PROVEN

**Lemma.** There is no multiset ℓ of triples (multiplicities ≤ 2) of total
weight 6 with (i) every point-degree ℓ_x ≡ 0 (mod 3) and (ii) every
pair-degree ℓ_xy even.

*Proof.* Touched points have ℓ_x ∈ {3,6}; Σℓ_x = 18, so ≤ 6 points are
touched. Let a = number of doubled triples (a ∈ {0,1,2,3}), so there are
6 − 2a single triples, all distinct, none equal to a doubled one.

- **a = 3.** ℓ_x = 2·(#doubles containing x) ∈ {0,3,6} forces every
  touched point to lie in all three doubles, so the three doubles share all
  three points — one triple of multiplicity 6 > 2. Contradiction.
- **a = 2.** Let S₁ ≠ S₂ be the singles. A pair of S₁ has odd degree
  unless it also lies in S₂ (doubles contribute evenly); so all 3 pairs of
  S₁ lie in S₂. But two distinct triples share at most one pair.
  Contradiction.
- **a = 1.** Let D be the double, s_x = #singles through x. For x ∈ D:
  2 + s_x ≡ 0 (mod 3) and s_x ≤ 4 gives s_x ∈ {1,4}. If s_x = 1, say the
  single through x is {x,p,q}: the pair {x,p} lies in exactly one single
  (no other single contains x) and evenly many times in D-copies — odd
  total. So s_x = 4 for ALL x ∈ D: every single contains every point of D,
  i.e. equals D. Contradiction.
- **a = 0.** Six distinct singles. If some x has ℓ_x = 6, all six triples
  contain x, and then every other touched point y has degree = ℓ_xy, which
  must be even, ≡ 0 (mod 3), positive, ≤ 6: hence 6 — all triples contain
  y too; with three touched points all six triples coincide. Contradiction.
  So all touched degrees are 3 and exactly 6 points are touched. The link
  of x (edges {u,v} for triples {x,u,v}) is a simple graph with 3 edges in
  which every vertex has even degree (= pair-degree at x): a triangle. So
  the three triples through x are {x,u,v}, {x,v,w}, {x,u,w}. Point u has
  two of its three triples containing x; its link contains edges xv, xw
  plus one more edge e from its third triple, and must be a triangle:
  e = vw, i.e. the third triple is {u,v,w}. The four triples so far form
  K₄⁽³⁾ on {x,u,v,w}, completing the degrees of all four points; the
  remaining 6 − 4 = 2 triples must live on the remaining 6 − 4 = 2 touched
  points — impossible for 3-sets. Contradiction. ∎

(The coordinator's point-congruence-valid example — three parallel triple
classes on 6 points — dies at pair {1,2}: leave-degree 1. Enumeration
count: 750 weight-6 multisets on ≤ 6 points pass the point congruence;
0 pass both congruences.)

**Consequence (with Section 1): T₃,₃(m) ≤ J(m) − 2 for every class m.**
PROVEN. At b = J−2 the leave weight is 10, which is attainable — see
Section 3.

## 3. Attainment: J−2 constructions — VERIFIED WITNESSES

**Scheme** (`scripts/construct_sym5.py`): prescribe the order-5
automorphism σ = (0 1 2 3 4)(5 6 7 8 9)... (c₅ five-cycles, m − 5c₅ fixed
points) and prescribe the leave to be the *doubled pentagon* on the first
5-cycle: the five cyclic-interval triples {i, i+1, i+2 (mod 5)} — equal to
the complements-in-{0..4} of the edges of the 5-cycle (0 1 2 3 4) — each
COVERED 0 TIMES, every other triple covered exactly twice. This hole is
σ-invariant (a single σ-orbit of triples), so the equality system reduces
to one constraint per triple-orbit and one 0/1/2 variable per quad-orbit;
block count Σ|orbit|·x = J−2 fixes the multiplicity of fixed quads mod 5.
The reduced exact-cover ILPs (HiGHS via scipy.optimize.milp) solved in
0.0–10.3 seconds:

| m  | J−2 target | c₅ | ILP size (vars × eq) | result | witness |
|----|-----------|----|----------------------|--------|---------|
| 7  | 15   | 1 | 7 × 7      | FEASIBLE, 0.0 s  | `witnesses/t33_m7_b15_hole.json` |
| 11 | 80   | 2 | 66 × 33    | FEASIBLE, 0.0 s  | `witnesses/t33_m11_b80_hole.json` |
| 19 | 482  | 3 | 776 × 197  | FEASIBLE, 1.4 s  | `witnesses/t33_m19_b482_hole.json` |
| 23 | 883  | 3 | 1827 × 399 | FEASIBLE, 10.3 s | `witnesses/t33_m23_b883_hole.json` |

Every witness INDEPENDENTLY VERIFIED by `scripts/verify_witness.py`
(separate logic: block count, quad multiplicities ≤ 2, all C(m,3) triple
coverages ≤ 2, leave extraction, congruence display, σ-invariance): all
VALID; every leave is exactly the doubled pentagon
{012},{014},{034},{123},{234} ×2. The m=19 witness uses 446 distinct quads
(36 doubled), the m=23 witness 819 (64 doubled); m=11's uses 80 distinct
quads (simple packing).

**Settled values (PROVEN = verified UB + verified witness):**
- **T₃,₃(19) = 482** — NEW; resolves conjecture C7's first open case.
- **T₃,₃(23) = 883** — NEW; C7's second open class case.
- T₃,₃(11) = 80 and T₃,₃(7) = 15 — known (Tan), now fully in-workspace
  (independent UB proof + independent witnesses).

**Leave universality (all four constructions + the coordinator's m=7
archaeology): the doubled C₅-edge-complement pentagon is A maximum-leave
shape for every solved class member.** Whether it is the ONLY weight-10
leave shape of maximum packings is open (not needed for the values).
**Remark (PROVEN + machine-checked): among FULLY-DOUBLED weight-10 leaves
it is the unique congruence-valid shape.** Proof: doubling makes pair
parity automatic; point degrees 2k_x ≡ 0 (mod 3) force k_x ∈ {0,3}, so
Σk_x = 15 gives exactly 5 support points with every point in exactly 3 of
the 5 triples; complementing each triple inside the support turns this
into a 2-regular simple graph with 5 edges on 5 vertices — necessarily a
C₅ (5 has no cycle partition with parts ≥ 3 other than 5 itself).
Machine check on 7 points: all 252 point-valid doubled leaves are
C₅-complements (252 = C(7,5)·12 labeled pentagons).

## 4. Failed attempts and structural observations — recorded per honesty bar

- **FAILED (structural): pure cyclic Z_m constructions cannot reach J−2.**
  J−2 ≢ 0 (mod m) at m = 19 (482 = 25·19 + 7) and m = 23 (883 = 38·23 +
  9), so no union of full Z_m-orbits (sizes m, m prime) hits the count;
  dihedral orbit sizes (2m, m) fail the same congruence. Any cyclic
  approach needs partial orbits (a staged base+top-up scheme was designed
  but became unnecessary once the σ-order-5 hole scheme succeeded).
- **Observation (Cauchy–Davenport): translate-form pentagon leaves are
  impossible in Z_m for prime m (in particular m = 19, 23).** A doubled
  leave of 5 translates {T+u : u ∈ U} with point-degrees ≡ 0 (mod 3) would
  need |T + U| = 5 support points, but
  |T + U| ≥ |T| + |U| − 1 = 7 in prime Z_m. Similarly weight-10
  single-multiplicity one-orbit leaves die (support ≥ 12 > 10 = maximum
  compatible with degrees ≥ 3). This is why the hole was prescribed on a
  σ-cycle of a NON-transitive order-5 action instead.
- **FAILED (instance): m=11 with c₅ = 1 (5-cycle + 6 fixed points) is ILP-
  infeasible** for the doubled-pentagon hole; c₅ = 2 succeeds instantly.
  Recorded to show the scheme's feasibility is action-dependent.
- The point congruence alone is genuinely insufficient at weight 6
  (750 surviving shapes); pair parity kills all of them — the "second-
  order argument" the coordinator's task brief asked for is exactly the
  pair congruence.

## 5. Literature check — summary

**Standard-notation bridge** (Burgess–Danziger–Horsley–Javed,
arXiv:2410.22607v2, 2025): packing designs PD_λ(v,k,t), packing number
PDN_λ(v,k,t), Johnson–Schönheim bound U_λ(v,k,t) =
⌊v/k·⌊(v−1)/(k−1)·⌊λ(v−t+1)/(k−t+1)⌋⌋⌋. For (λ,k,t) = (2,4,3):
U₂(v,4,3) = ⌊v·⌊(v−1)(v−2)/3⌋/4⌋ = our J(v) exactly. So this workspace's
result reads: **PDN₂(v,4,3) = U₂(v,4,3) − 2 for v ≡ 3 (mod 4), 3 ∤ v** —
upper bound proven for all such v, attainment verified at v = 7, 11, 19,
23 (+ 31, 35 pending). That survey also confirms the "second Johnson
bound" literature exists ONLY for λ = 1; the pair-parity argument here is
the λ = 2, t = 3 second-order analogue.

Six targeted searches (index-2 / twofold quadruple packings, D₂(v,4,3),
3-(v,4,2) packing, λ-fold complete 3-uniform hypergraph K₄⁽³⁾ packing,
tetrahedron packing/leave, Hanani spectrum):
- λ = 1 is a finished literature: packing numbers D(v,4,3) = A(v,4,4)
  (constant-weight codes), asymptotically Ji (Des. Codes Cryptogr. 2004),
  completed by Bao–Ji (Des. Codes Cryptogr. 2015 / arXiv:1401.2022) — all
  last 21 values EQUAL the Johnson bound (n ≡ 5 (mod 6)).
- λ-fold 3-uniform *packing* papers found treat small structures only
  (loose 3-cycles, triple-hyperstars, special tetrahedra ST ≠ K₄⁽³⁾).
- λ-fold quadruple DESIGNS (3-(v,4,λ)) are Hanani's classical spectrum.
- NOT FOUND anywhere: determinations of D₂(v,4,3); any "Johnson bound
  minus 2" statement for index-2 quadruple packings; any doubled-pentagon
  leave characterization. The workspace's Lemma E + Theorem F + these
  values appear to be new, subject to the standing folklore-risk caveat
  (leave-congruence methodology is classical maximum-packing technique;
  Hanani 1963 and the Mills–Mullin 1992 packing survey remain not fully
  accessible for a definitive negative).

## 6. Status of conjecture C7 on the class

[Updated 2026-07-29 — see C7_general.md and gklo_citation.md for the
general-m closure.]

- Upper bound T ≤ J−2: PROVEN for the whole class (Theorem F, verified).
- Attainment T = J−2: PROVEN for m ∈ {7, 11, 19, 23} (witnesses here),
  and PROVEN for all sufficiently large class m via GKLO Theorem 1.1
  applied to K_m − pentagon (citation pinned and hypotheses verified in
  gklo_citation.md; m₀ ineffective).
- Middle range (31, 35, ... below the unknown m₀): OPEN individually;
  each is a finite ILP via the σ-scheme (m = 31, 35 solver time-outs so
  far, see STATUS.md).
- The 3|m family (T = J, minimal-leave classification, m = 27 = J
  witness) is treated in C7_general.md Section 3.

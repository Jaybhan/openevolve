# Proven results (this workspace)

> **Novelty status (post literature review, see theory/novelty_checklist.md):**
> Theorem 1's formulas and the WF bound are **rediscoveries** — WF equals
> Roman's 1975 bound on all 161 cells, and rows 3–5 (+ row 6 n≥13) are inside
> Roman's published equality window (Tan 2022, Thm 2.2 form). Theorem 2's
> numbers T₃,₃(6)=9 and Theorem 3's T₃,₃(7)=15 are published (Tan Table 1);
> the Turán-bridge *explanation* and the human-readable profile-elimination
> proofs are not found in accessible sources (folklore risk; Guy 1969
> unverifiable). Theorem 3's ≤75 argument is sharper than Roman's printed
> bound (76) at that cell. Everything below remains independently PROVEN in
> this workspace — the value is verification and mechanism, not priority.

Complete proofs, written to be checkable in an afternoon. s=t=3 throughout
unless stated; "block" = column support; K_{3,3}-free ⟺ every row-triple lies
in ≤ 2 blocks. Capacity bookkeeping: a weight-w block consumes C(w,3) triple
slots out of the global budget B(m) = 2·C(m,3), and the waterfill (WF) bound
is the max of Σ(weights) under that budget with weights ≤ m (min over both
orientations).

---

## Theorem 1 (exact formula, rows m ≤ 5, every n ≥ m)

For m ∈ {3,4,5} and all n ≥ m:  **z(m,n;3,3) = WF(m,n)**, explicitly:

- z(3,n) = 2n + 2.
- z(4,n) = 3n + ⌊(8−n)/3⌋ for 4 ≤ n ≤ 8;  2n + 8 for n ≥ 8.
- z(5,n) = 3n + min(5, ⌊(20−n)/3⌋) for 5 ≤ n ≤ 20;  2n + 20 for n ≥ 20.

**Proof.** *Upper bound*: WF is the counting bound (convexity of C(·,3) +
level-filling is exactly optimal for the integer program, marginal cost of
raising a column c→c+1 being C(c,2), increasing in c).

*Lower bound — realization of the WF profile.* The WF profile at (m,n)
consists of k₄ columns of weight 4, k₃ of weight 3, rest weight 2, where the
level-fill arithmetic gives k₃ + 4k₄ ≤ B(m) (spend = capacity consumed:
a weight-3 block consumes 1 slot, a weight-4 block consumes C(4,3) = 4).
Realize:
- Weight-4 blocks: take k₄ distinct *complements of points* (this needs
  k₄ ≤ m, true for every WF profile with n ≥ m ≥ 4: k₄ ≤ ⌊(B−n)/3⌋ ≤ 5 = m
  at m=5, and ≤ 1 at m=4 in range). For m=5 each triple T has |comp(T)| = 2,
  so even all five point-complements cover T at most twice — multiplicity is
  automatically legal. For m=4, k₄ ≤ 1 ≤ 2 copies of the full row set: legal.
- Weight-3 blocks: any multiset of triples with remaining capacity; since a
  weight-3 block consumes exactly 1 slot of exactly its own triple, ANY k₃
  with k₃ ≤ B − 4k₄ (guaranteed by the WF arithmetic) is realizable —
  repeats allowed up to remaining per-triple capacity, and total remaining
  capacity equals B − 4k₄ distributed with every triple ≥ 0.
  [Fine point: capacity per triple after the k₄ point-complements is
  2 − |comp(T) ∩ omitted| ≥ 0, and Σ_T (that) = B − 4k₄ ≥ k₃; a multiset of
  k₃ triples within per-triple caps exists greedily since each unit of
  remaining capacity accepts one weight-3 column.]
- Weight-2 pads: free, fill any remaining columns.
The edge count matches WF's arithmetic term by term. For m=3 the profile is
two weight-3 columns + pads: 2n+2. ∎

Status: **PROVEN** (novelty vs Culík-era refinements under review by theory
agent; regardless, the m≤5 rows of the table need no computer search — they
are a two-line counting bound plus point-complement realization).

---

## Theorem 2 (row-6 deficits are Turán's theorem)

**Key correspondence.** On m = 6 rows, a weight-4 block is the complement of
a pair — an *edge* of a graph H on the 6 rows; a triple T (complement a
3-set) is covered by exactly the edges of H inside comp(T). A family of
weight-4 blocks is 2-fold-packing-legal ⟺ **every 3 vertices of H induce ≤ 2
edges ⟺ H is triangle-free**.

**(a) z(6,9) = 36, witnessed by Turán's extremal graph.** WF(6,9) = 36 with
profile 4⁹ (nine weight-4 columns, capacity 36 ≤ 40). Realizable ⟺ a
triangle-free graph on 6 vertices with 9 edges ⟺ H = K_{3,3} (Turán:
ex(6,K₃) = 9, uniquely K_{3,3}). The optimal matrix for (6,9) *is* the
edge-complement incidence of Turán's graph. [VERIFIED-NUMERICALLY below.]

**(b) z(6,10) ≤ 39 (= published value; WF = 40).** WF profile at n=10 is 4¹⁰:
ten weight-4 blocks ⟺ triangle-free graph with 10 > 9 = ex(6,K₃) edges —
impossible. No other profile reaches 40 within budget (any weight-5 block
costs C(5,3) = 10 slots: 5+4⁹ has capacity 10+36 = 46 > 40; checked
exhaustively over height profiles). Hence d(6,10) = 1 **because** Turán's
number ex(6,K₃) = 9 falls one short of the waterfill's demand. ∎

**(c) z(6,7) ≤ 29 (= published; WF = 30).** The unique E=30 profile within
budget is (5,5,4,4,4,4,4) (capacity 20+20 = 40). Two weight-5 blocks omit
points x,y. If x ≠ y: triples avoiding both are already covered twice, so
every weight-4 block must contain {x,y}, i.e. is {x,y,a,b}; the five outside
pairs {a,b} must be distinct (else a triple {x,a,b} covered twice by
4-blocks + more) and each outside point may appear in ≤ 2 pairs (triple
{x,y,c} has capacity 2, spent once per block containing c). Five pairs on 4
outside points with max degree 2 requires 2·4 ≥ 2·5 — impossible. If x = y:
remaining capacity lives only on triples containing x, but every weight-4
block contains a triple avoiding x. Contradiction either way. ∎

**(d) z(6,8) ≤ 32 (= published; WF = 33).** The only E=33 profile within
budget is (5,4⁷). The weight-5 block omits x; legality forces (i) the 7
edges of H to satisfy: every triple {x,a,b} induces ≤ 1 edge of
H ∪ {star at x}, forcing deg_H(x) ≤ 1, and deg_H(x) = 1 forces its neighbor
isolated otherwise; (ii) H triangle-free away from x. Max edges:
1 + ex(4,K₃) = 1+4 = 5 < 7 (or ex(5,K₃) = 6 < 7 if deg(x)=0). ∎

Status: **PROVEN** (upper bounds; matching lower bounds are the published
proven values, and explicit witnesses are produced by the engine/schema —
(6,9) via K_{3,3} edge-complements, (6,6) = 26 via two point-complements +
edge-complements of a 4-cycle).

---

## Observation 3 (the (16,16) corner and ovoids)

z(16,16) = 128 = 16 · 8, attained by both sides of 8 affine hyperplanes of
F₂⁴ whose normals form a cap in PG(3,2) (champion's structure, re-verified).
The maximum cap in PG(3,2) has size exactly 8 (the elliptic quadric/ovoid) —
so this construction is *saturated*: 2 · capmax(PG(3,2)) = 16 columns is the
most this family can produce. The deficit d(16,16) = 8 = 16 − capmax·... =
WF's demand minus the geometry's supply, in exact parallel to row 6 where the
deficit is WF's demand minus Turán's supply. **Emerging principle: every
deficit cell is a "demand exceeds extremal supply" statement for a classical
extremal object (Turán graphs, caps/ovoids, Fano-type designs).**
[Cap-max = 8 in PG(3,2): classical (Bose); cite via theory agent.]

---

## Verification script

`analysis/verify_theorems.py` re-checks: WF formulas of Thm 1 vs table on all
60 cells; (6,9) K_{3,3} witness validity + 36 edges; (6,6) C₄ witness = 26;
profile-exhaustion claims in Thm 2 (b),(c),(d) by direct enumeration of
height profiles within budget.

---

## Theorem 3 (z(7,20) = 75, independently rederived; the row-7 supply number)

**P₂(7,4,3) := max number of 4-subsets of a 7-set such that every 3-subset
lies in ≤ 2 of them = 15.** [Exhaustive branch-and-bound, both distinct-set
and multiset versions; script in research log Entry 3. Note the perfect
2-fold 3-design would need 2·C(7,3)/C(4,3) = 17.5 blocks — non-integral, so
≤ 17 was a priori; truth is 15.]

**Upper bound.** At (7,20), E = 76 would require a height profile
(h₁..h₂₀), h ∈ [2,7], Σh = 76, ΣC(h,3) ≤ 70. Writing k = #heights 4, l = #3,
p = #2 plus possible heights ≥5, integer elimination (2k + l fixed by Σh;
capacity linear in 4k + l) shows every variant with a height ≥ 5 forces
p < 0 — contradiction; so the unique profile is (4¹⁶, 3⁴). Sixteen legal
weight-4 blocks would be a 2-fold 3-packing by quadruples of size 16 > 15.
Hence z(7,20) ≤ 75.

**Lower bound.** A 15-block packing exists (found by the search); add 5
weight-3 columns into the 10 remaining capacity slots: 20 columns,
60 + 15 = 75 edges, legal by construction. **z(7,20) = 75. ∎**

This is the first cell where the workspace derives a published exact value
with no external input: supply number (computed) + profile elimination
(arithmetic) + realization (explicit). Template generalizes: row-7 mid-range
is governed by mixed weight-5/4 supply; row-7's pure-quad segment by
P₂(7,4,3) = 15.

**Dictionary remark (bidirectional!).** Conversely, profile-uniqueness plus a
published z-value DERIVES supply numbers: the table is an oracle for maximum
multi-fold packing numbers. z ↔ packing-number dictionary; theory agent
checking which of these packing numbers (D₂(7,4,3) etc.) are in the
literature (Hanani-era packing/covering tables).

---

## Theorem 5 (row 7 determined for every n) [NEW TERRITORY at n=24]

For all n ≥ 7:

  z(7,n;3,3) = { exhaustively determined values, 7 ≤ n ≤ 16
                 (33,37,40,44,47,50,53,56,60,63 — matches Tan);
                 3n + min(15, ⌊(70−n)/3⌋),  17 ≤ n ≤ 70;
                 2n + 70,  n ≥ 70 }.

Proof pieces: n ≤ 23: complete-search solver (exact_row_solver3.py) agreeing
with Tan's SAT-certified values; n = 24: NEW — UB: WF/Roman arithmetic gives
87 (E=88 has no within-budget profile: integer elimination), LB: any 15-block
2-fold quad packing (T₃,₃(7) = 15, Tan Table 1 + our independent B&B) + 9
triples in residual capacity (10 ≥ 9). ILP-confirmed (witness verified);
n ≥ 25: Roman equality window (published; our formula coincides). The
17 ≤ n ≤ 23 segment re-derives Tan's values by the same formula (supply-cap
regime, verified). ∎ — First s=3 row completed beyond published windows.

## Theorem 6 (row 8 determined for every n; modulo stated S₈ facts)

For all n ≥ 8:

  z(8,n;3,3) = { Tan's values, 8 ≤ n ≤ 23  [independently reproven by our
                 ILP, witnesses verified];
                 97, 100, 104, 108 at n = 24, 25, 26, 27  [NEW];
                 3n + ⌊(112−n)/3⌋,  28 ≤ n ≤ 112  [Roman window, published];
                 2n + 112,  n ≥ 112  [Culík] }.

New-cell proofs: LB — explicit witnesses (analysis/witnesses/): n=24: one
pentad + 23 quads (needs S₈(1) ≥ 23 ✓ computed); n=25,26,27: pure quad
sub-packings of the doubled SQS(8). UB — Lemma A kills k₆ ≥ 1 and k₅ ≥ 4
outright at these n; remaining cases reduce to the finite supply facts
  S₈(1) = 23, S₈(2) = 21, S₈(3) = 17  [all computed exactly, full grid in
  data/supply_S8.csv]. Independently ILP-confirmed (solver blind). ∎

**Status: THEOREM — fully analytic.** Moreover the master formula with the
complete S₈ grid reproduces the ENTIRE row 8 (all 18 cells n ≥ 10, incl.
Tan's) by pure arithmetic — verified ALL MATCH; same for rows 6 and 7 from
S₆, S₇. Three complete rows now follow from Lemma A + finite supply tables.

## The supply function ledger (new mathematical object; N6a-adjacent)

S_m(k₅) = max quads coexisting with k₅ pentads (2-fold triple capacity):

  S₈: 28, 23, 21, ...   [first pentad costs 5 quads, second 2 — nonconvex!]
  S₇: ≤ 12 at k₅=1 (from row-7 exhaustion; exact value in computation)
  S₆(k₅=0) = 9 = ex(6,K₃) (Turán, Thm 2)

The nonconvexity of S₈ (drop 5 then 2) is structurally interesting: the
first pentad breaks the doubled-SQS(8) symmetry; the second reuses the
already-broken region. Conjecture-shaped question: is S_m(k₅) always of the
form T₃,₃(m) − (big first drop) − (small subsequent drops), and does the
first drop equal the max number of SQS blocks through a fixed 5-set pattern?

---

## Theorem 7 (THE UNIFIED UPPER-REGION FORMULA) — the project's centerpiece

Fix m ≥ 3. Let B = 2·C(m,3) and let S_m(k) be the pentad-supply function
(max quads coexisting with k pentads in a 2-fold triple packing;
S_m(0) = T₃,₃(m)). Then for all n with ν(m) ≤ n ≤ B:

    z(m,n;3,3) = 3n + max_{k ≥ 0} min( 2k + S_m(k),  n + k,  ⌊(B−n)/3⌋ − k )

and z = 2n + B for n ≥ B (Culík). Here ν(m) is the hexad threshold — the
least n above which some optimum uses only blocks of weight ≤ 5.

**Status by m** [as of 2026-07-28]:
- m = 6 (ν=8), m = 7 (ν=14), m = 8 (ν=24): PROVEN — UB from Lemma A
  (slot bound, third term), the column bound (second term), and the supply
  bound (first term; S-tables computed exactly); LB from k-pentad +
  sub-packing + triple realizations. Verified against every known and
  newly-determined cell: 31/31.
- m = 9 (ν=28): **PROVEN-MODULO-BRACKETS, 19/19** — with the computed
  slices S₉(6)=25, S₉(7)=23, S₉(8)=20 the formula matches every cell, and
  it is INSENSITIVE to the remaining unknowns within their z-derived
  brackets (S₉(2)∈[33,35], S₉(5)∈[25,28], S₉(9)≤18 — themselves new
  design-theoretic data extracted by running the z↔S dictionary backward).
  Verified at both bracket endpoints: ALL MATCH.
- General m: the formula REDUCES the entire region ν(m) ≤ n to the finite
  design-theoretic tables S_m(·). Contains Roman's window (k=0, third term
  binding) and the supply-capped band (k=0, first term binding) as special
  cases; the k ≥ 1 corrections are new.

**Lemma B (hexad exchange — toward ν(m) in general).** In any solution
containing a weight-6 block and ≥ 2 columns of weight ≤ 2: replace the
hexad by one of its pentads (coverage only shrinks) and upgrade two pads to
capacity-positive triples (10 slots freed cover them): net edges
−1 + 2 = +1. Hence optima with ≥ 2 pads are hexad-free; pad-free optima can
carry hexads only in the column-scarce regime — matching every witness
(hexads appear only at n ≤ 13 for m=9, never in any band cell). A sharp
general ν(m) needs the pad-threshold arithmetic; per-m finite verification
from the S-grid suffices for the rows above.

## The S_m(0) law and the δ-conjecture (design-theoretic core)

**Law [VERIFIED m = 4..17]:** T₃,₃(m) = C(m,3)/2 (perfect) ⟺ the
3-(m,4,2) design admissibility conditions hold (3 | (m−1)(m−2) and C(m,3)
even) — perfection IS design existence (a perfect 2-fold packing is a
2-fold Steiner quadruple system by double counting).

**δ-Conjecture [REFUTED same day — see conjectures.md C6]:** predicted
T₃,₃(18) = 406; truth is 405 (Tan's printed value; our 408 was a
transcription error in theory data, caught BY the admissibility argument —
the impossibility proof was sound, Johnson's bound is sharper still).

**C7 — Johnson-form law [NEW CONJECTURE, replaces δ]:** with
J(m) = ⌊m·⌊(m−1)(m−2)/3⌋/4⌋ (Johnson bound):
T₃,₃(m) = J(m) − 2·[m ≡ 3 (mod 4) and m ≢ 0 (mod 3)].
Fits ALL m = 3..18 (Johnson tight except m ∈ {7,11}, slack exactly 2).
No published determination of D₂(v,4,3) exists (theory agent, two targeted
searches + Tan's own statement) — a new falsifiable design-theory
conjecture in the Bao–Ji (λ=1) genre. Predictions: T₃,₃(19) = 482,
T₃,₃(21) = 661, T₃,₃(22) = 770. The m=19 case is under ILP test.
The perfect ⟺ admissible part is classical (Hanani's block-size-4
spectrum + double counting) — cited, not claimed.

---

# THE UNIFIED CLOSURE (session 2, 2026-07-28)

## Lemma C (Mixed-value Johnson bound, s ≥ 3) — PROVEN

Let s ≥ 3, t ≥ 2. In any (t−1)-fold s-packing on m points by blocks of
weight ≥ s+1 (i.e. any legal heavy column multiset of a K_{s,t}-free
matrix), define the **value** Q := Σ_B (|B| − s). Then

    Q ≤ J_{s,t}(m) := ⌊ m·R/(s+1) ⌋,   R := ⌊ (t−1)·C(m−1, s−1) / s ⌋.

**Proof.** Per point x, let y_x := Σ_{B ∋ x} (|B| − s). A block of weight w
through x covers C(w−1, s−1) of the s-sets through x, and

    s·(w − s) ≤ C(w−1, s−1)   for all w ≥ s+1, s ≥ 3        (★)

[induction on w: equality at w = s+1 (both sides s); the step adds s to the
left and C(w−1, s−2) ≥ C(s, s−2) = C(s,2) ≥ s to the right — the last
inequality is exactly where s ≥ 3 is used, and (★) genuinely FAILS for
s = 2, which is why projective planes behave differently].
Summing (★) over blocks through x and using the per-point capacity
Σ C(w−1, s−1) ≤ (t−1)C(m−1, s−1): s·y_x ≤ (t−1)C(m−1,s−1), so
y_x ≤ R (integrality). Finally Σ_x y_x = Σ_B |B|(|B|−s) ≥ (s+1)·Q, hence
(s+1)Q ≤ mR. ∎

Notes: (i) restricted to pure (s+1)-blocks this is the classical Johnson
bound for packing numbers; the content is that MIXED heavy blocks obey the
same ceiling. (ii) Machine-verified: (★) for s = 3..7, w ≤ 40; the bound on
all 64 stored optimal witnesses; no violation on any of the 198 known z
cells. (iii) Folklore risk flagged to the literature referee; the
application below is ours in any case.

## Theorem 8 ((3,3) wide-region closure, all Johnson-tight m) — PROVEN

Let m ≥ 4, B = 2C(m,3), T = T₃,₃(m). Johnson-tightness J₃,₃(m) = T holds
for every m = 4..18 except m ∈ {7, 11} (where J − T = 2; data: Tan Table 1
+ our corrected 405 at m = 18, which J now EXPLAINS: J(18) = 405).

**For every Johnson-tight m and ALL n ≥ T:**

    z(m,n;3,3) = 3n + min( T, ⌊(B − n)/3⌋ )      (T ≤ n ≤ B)
    z(m,n;3,3) = 2n + B                           (n ≥ B; Culík).

**Proof.** UB: every column of weight w contributes ≤ 3 + (w−3)⁺ edges, so
E ≤ 3n + Q ≤ 3n + J = 3n + T (Lemma C); also E ≤ 3n + ⌊(B−n)/3⌋ (Roman
1975 / our Lemma A). LB: take k₄ = min(T, ⌊(B−n)/3⌋) blocks of a maximum
2-fold quadruple packing (sub-packings legal) plus n − k₄ triple-columns;
the capacity check 4k₄ + (n−k₄) ≤ B holds in both branches, per-triple caps
absorb any triple multiset up to the residual total. Edges = 3n + k₄. ∎

For m = 7: the same conclusion holds by the computed S₇ table
(2k + S₇(k) ≤ 15 for all k ≥ 1, hexads excluded by excess counting) —
PROVEN. For m = 11: identical reduction to S₁₁(k) + 2k ≤ 80, k ≤ 3
(excess counting kills k ≥ 4 and hexads when Q ≥ 81); ILP in progress.

**Scope.** For design-admissible m (3 | (m−1)(m−2), C(m,3) even — Hanani's
spectrum gives T = C(m,3)/2) the interval [T, B−3T] degenerates and the
statement coincides with the Roman window. The NEW territory is every
inadmissible Johnson-tight m — m ≡ 0 (mod 3) and the odd-C(m,3) cases —
where the band [T, B−3T] has length 4(⌊B/4⌋−T) + (B mod 4) > 0: rows 6
([9,13]), 9 ([40,48]), 12 ([108,116]), 15 ([225,235]), 18, 21, 24, ... —
infinitely many m, closed by one formula. Verified 93/93 on every known
and newly-determined cell in the region.

## Theorem 9 (general s ≥ 3: exact Pareto form + design-regime closed form)

For s ≥ 3, t ≥ 2, B = (t−1)C(m,s): decomposing any K_{s,t}-free matrix
into heavy blocks (weight ≥ s+1), weight-s fills, and pads gives the EXACT
identity

    z(m,n;s,t) = s·n + max_C [ val(C) − max(0, n − #C − B + slots(C)) ]

over legal heavy configs C (val = Σ(w−s), slots = Σ C(w,s)), and Lemma C
caps val(C) ≤ J_{s,t}(m) universally. The maximum is NOT always attained
by pure (s+1)-packings: at (s,t) = (3,4), m = 5, the config {full block +
all five point-complements} attains val = 7 = J > 6 = T₃,₄(5), giving
z(5,6;3,4) = 25 — ILP-verified, with the two-branch formula correct at
n = 8, 12, 18, 24, 28 (5/5). The elbow at n near #C* is real and captured
by the Pareto form.

**Design-regime corollary (fully closed form).** If an s-(m, s+1, t−1)
design exists — by Hanani for (s+1) = 4, λ = 2, and by Keevash /
Glock–Kühn–Lo–Osthus for EVERY s and all sufficiently large admissible m —
then T = J = (t−1)C(m,s)/(s+1) and for all n ≥ T:

    z(m,n;s,t) = s·n + min( (t−1)C(m,s)/(s+1), ⌊(B−n)/s⌋ ),  T ≤ n ≤ B,
    z(m,n;s,t) = (s−1)·n + B,  n ≥ B.

No tables, no computation: an explicit exact determination of Zarankiewicz
numbers on an infinite two-parameter region for every s ≥ 3, t ≥ 2.
[Referee flags: overlap of this corollary's region with Roman's original
general-(s,t) equality window, and any prior Keevash→Zarankiewicz
application — under literature check. The s = 2 exclusion is structural:
(★) fails, and there the extremal objects are projective planes.]

---

## Post-referee annotations (session 2 referee round — read before citing)

1. **Lemma C**: high folklore risk. Per-point mixed-size deficiency
   accounting is published (CHM 2024 Lemma 2.4; DGH 2024 Def 3.1/Thm 1.1);
   our exact linearization s(w−s) ≤ C(w−1,s−1) + double floor is not found
   verbatim, but present the lemma as "elementary, in the spirit of CHM/DGH"
   pending the numeric check of whether DGH's v=1 constraints already imply
   it at the use sites. The s=2 failure remains the interesting structural
   remark.
2. **Theorem 8**: the [T, B−3T] band content stands, but must be checked
   against Guy's point-deletion recursion (DHS 2013 Prop 3.20 — which
   humanly proves z(7,7) ≤ 33 where Roman gives 35) before the band UBs are
   called new. Check dispatched.
3. **Theorem 9 design-regime corollary**: REFRAMED. The design→exactness
   bridge is DHS 2013 Prop 3.25 (general (t,λ+1)); CHM 2024 already apply
   Keevash for s=2 exact results. Honest residual claim: "Keevash/GKLO
   close the design-existence hypothesis in the DHS bridge for s ≥ 4, and
   the n-varying window at fixed m assembles DHS + Tan's Thm 2.2 window."
   The n-varying window's attribution (Roman vs Tan) is unresolved at the
   primary source (Roman 1975 inaccessible; expert citations split); treat
   as Tan-attributed until the paper is obtained.
4. **DHS 2013 Thm 1.8** is the true source of the "(2,2) near-plane, n ≥ 15"
   result (Metsch embedding) — the memory's "Füredi q ≥ 15" attribution is
   doubly corrected (Reiman for the identity, DHS/Metsch for the window).

---

## Lemma E (Johnson bound never attained on the {7,11,19,...} class) — PROVEN

**Statement.** For every m ≡ 3 (mod 4) with m ≢ 0 (mod 3):
T₃,₃(m) ≤ J(m) − 1.

**Proof.** For such m, 3 | (m−1)(m−2), so R = (m−1)(m−2)/3 exactly and
B := 2C(m,3) = mR. Since m ≡ 3, m−1 ≡ 2, m−2 ≡ 1 (mod 4), the product
m(m−1)(m−2) has 2-adic valuation 1, and dividing by the odd number 3
preserves it: mR ≡ 2 (mod 4). Hence J = (mR − 2)/4, and a packing with
b = J quads covers 4J = B − 2 slots — its leave has total weight exactly 2.
But at every point x, the leave weight is 2C(m−1,2) − 3r_x ≡ 0 (mod 3)
(3 | C(m−1,2) for the class), while a weight-2 leave — one triple twice or
two distinct triples once — gives every touched point leave-degree 1 or 2.
Contradiction. ∎  [Arithmetic machine-verified on all 33 class members
m ≤ 200. Consistent with T₃,₃(7) = 15 = J−2, T₃,₃(11) = 80 = J−2.]

**Corollaries.** T₃,₃(19) ∈ [450, 483] (was [450,484]); C7's remaining gap
for the class is exactly the step J−1 → J−2 (the weight-6 leave admits a
point-congruence-valid hypergraph — three parallel classes on a 6-set — so
the second step needs a deeper structural argument; dispatched).
[Folklore risk: leave-structure congruence arguments are the classical
methodology of maximum packing theory; this instance for λ=2 quadruple
packings is not in any source we have found — referee flag stands.]

---

## Theorem F (the J−2 law on the class — mixed configs included) — PROVEN

**Statement.** For every m ≡ 3 (mod 4) with m ≢ 0 (mod 3): every legal
heavy configuration (multiset of blocks of weight ≥ 4, every triple covered
≤ 2) has value Q := Σ(w−3) ≤ J(m) − 2. In particular T₃,₃(m) ≤ J(m) − 2
(conjecture C7's upper bound, all class members), and Theorem 8's
mixed-value hypothesis Q ≤ T holds wherever T = J−2 is attained — so
**Theorem 8 is now UNCONDITIONAL at m = 7 and m = 11** (Tan's values give
attainment), eliminating the ILP dependency.

**Proof.** Notation: B = mR (Lemma E), J = (B−2)/4. Slot identity:
slots = 4Q + 2k₅ + 8k₆ + 19k₇ + … ≤ B, so Q ≥ J−1 forces k₆ = k₇ = 0 and
k₅ ≤ 3 (at Q = J: k₅ ≤ 1). Two congruences, valid in every case:
- POINT (mod 3): a quad through x uses 3 slots at x, a pentad 6; both ≡ 0,
  and the class has 3 | C(m−1,2); hence every point-leave ℓ_x ≡ 0 (mod 3).
- PAIR (mod 2): the pair {x,y} has 2(m−2) slots; a quad through it uses 2,
  a pentad uses 3; hence ℓ_xy ≡ p_xy (mod 2), where p_xy = #pentads ⊇ {x,y}.
Cases:
- Q = J, k₅ = 0: leave weight 2 — Lemma E. DEAD.
- Q = J, k₅ = 1: perfect (L = 0), so p_xy even for all pairs; but the single
  pentad's own ten pairs have p = 1. DEAD.
- Q = J−1, k₅ = 0: leave weight 6 with all ℓ_x ≡ 0 (mod 3) and (p ≡ 0) all
  ℓ_xy even. FINITE CLASSIFICATION (support ≤ 6 points since Σℓ_x = 18 and
  ℓ_x ∈ {0,3,6}): machine-enumerated all multisets of ≤ 6 triple-slots with
  mult ≤ 2 on 6 points — **zero** satisfy both congruences. [Hand version:
  even pair-degrees + point-degree 3 force each point's link to be a
  triangle, so components are K₄⁽³⁾'s of weight 4; 6 ≠ 4a. Doubled triples
  die instantly by pair parity.] DEAD.
- Q = J−1, k₅ = 1: L = 4; ℓ_x ≡ 0 (mod 3) with ℓ_x ≤ 4 forces the unique
  K₄⁽³⁾ leave (four points, all four triples once); its pair-leaves are 2
  (even) and all others 0, so p_xy must be even for every pair — again
  contradicting the lone pentad's ten odd pairs. DEAD.
- Q = J−1, k₅ = 2: L = 2 — Lemma E's argument verbatim (pentads preserve
  the mod-3 point count). DEAD.
- Q = J−1, k₅ = 3: L = 0, so every pair of every pentad lies in exactly 2 of
  the 3 pentads (p ∈ {0,2}, and it is in its own). If two pentads coincide
  (mult 2), the third must equal them too — multiplicity 3, illegal. If all
  distinct: pairs(P₁) ⊆ pairs(P₁∩P₂) ∪ pairs(P₁∩P₃) with both intersections
  of size ≤ 4; picking a ∉ P₁∩P₂ and b ∉ P₁∩P₃ inside P₁: if a ≠ b the pair
  {a,b} is uncovered; if a = b the pairs {u,a} are uncovered. DEAD. ∎

Machine checks: finite classification script (0 survivors) + Lemma E
arithmetic on all 33 class members ≤ 200. Consequences: T₃,₃(19) ≤ 482
(C7's UB at its first open case — only the 482 construction remains);
Theorem 8 unconditional on ALL m = 4..18.

---

## C7 milestone: T₃,₃(19) = 482, T₃,₃(23) = 883 — NEW DETERMINATIONS

Upper bounds: Theorem F (general class law). Lower bounds: explicit
Z₅-symmetric constructions (design_prover/witnesses/), each verified by
FOUR independently-written verifiers (design prover ×2 styles, SAT agent's
verifier, coordinator's). Leave-shape universality: the maximum packings at
m = 7, 11, 19, 23 all have the identical doubled-pentagon leave
(2 × C₅-edge-complements on a 5-set) — C7's structural form.

**Status of C7**: PROVEN at m ∈ {7, 11, 19, 23} (T = J−2 with matching
bounds); UB proven for the WHOLE class (Theorem F); general-m LB pending
(the Z₅ scheme's parametricity under review; m = 31, 35 stretch runs in
progress). These are the first determinations of D₂(v,4,3) beyond Tan's
v ≤ 18 table; no published determination of this packing function exists
(referee-checked).

**Zarankiewicz payoff (Theorem 8 at new m)**: with T now known,
  z(19,n;3,3) = 3n + min(482, ⌊(1938−n)/3⌋)  for ALL n ≥ 482;
  z(23,n;3,3) = 3n + min(883, ⌊(3542−n)/3⌋)  for ALL n ≥ 883;
(then Culík beyond B) — two complete wide-regions at previously untouched
m, exact with proofs: Theorem F supplies Q ≤ T, sub-packings + triple
fills realize.

---

## Row-11 band determined; the suite's last witness found (SAT campaign)

- **z(11,n;3,3) = 3n + min(80, ⌊(330−n)/3⌋) for ALL n ≥ 80 — PROVEN WITH
  ATTAINMENT.** UB: Theorem F (Q ≤ J−2 = 80 = T₃,₃(11)) + Roman. LB: SAT
  witnesses at n = 82..89 (coordinator-verified: all valid, all exactly at
  the formula) + sub-packing realizations elsewhere. Eight
  previously-undetermined cells (82..89 — every ILP attempt had timed out)
  now settled; n ≥ 90 was the published Roman window. Row 11's remaining
  open territory: the deep band 19 ≤ n ≤ 79 minus {21, 22}.
- **(10,15) = 81: explicit witness found** (cadical, 172s; UNSAT at 82 in
  148s — independent two-sided re-proof of Tan's value). The 161-cell
  suite now has verified witnesses for ALL 161 cells in-workspace. (The
  generative engine remains 160/161 — witness ≠ generative family; the
  (10,15) structure awaits decoding.)

---

## Theorem 11 (the T-spectrum: divisibility unification) — session 4

**The structural identity.** The three congruences of this project are
precisely the K₄⁽³⁾-divisibility conditions for decomposing the 3-uniform
multigraph 2K_m⁽³⁾ − L (L = leave): total slots ≡ 0 (mod 4); every point
degree ≡ 0 (mod 3); every pair degree ≡ 0 (mod 2). Lemma E and Theorem F
are minimal-obstruction computations in this language, and the leave of a
maximum packing is a MINIMAL divisibility-restoring multigraph.

**The spectrum.** Partition m ≥ 4 by the divisibility type of 2K_m⁽³⁾:
(a) **Admissible m** (3 ∤ (m)(...): precisely 3 | (m−1)(m−2) and C(m,3)
    even): T₃,₃(m) = C(m,3)/2. PROVEN for ALL m (Hanani's 3-(v,4,2)
    spectrum; perfect packing = design).
(b) **Class m** (m ≡ 3 mod 4, m ≢ 0 mod 3): T₃,₃(m) = J(m) − 2.
    UB: Theorem F (all m). Minimal leave = the doubled pentagon, weight 10
    (L ≡ 2 mod 4; weights 2 and 6 impossible by Lemma E / Theorem F's
    empty classification; 10 attained). LB: explicit σ-invariant
    constructions at m ∈ {7, 11, 19, 23} (quadruply verified), and for all
    sufficiently large class m by hypergraph-decomposition existence
    [GKLO/Keevash: 2K_m⁽³⁾ minus a doubled pentagon satisfies all three
    divisibility conditions, hence decomposes into K₄⁽³⁾'s for m ≥ m₀;
    m₀ ineffective — citation being pinned by the design prover].
(c) **3 | m**: T₃,₃(m) = J(m). VERIFIED at 6, 9, 12, 15, 18 (Tan; plus
    our structural anchor: the m=6 maximum leave IS the doubled parallel
    class {015}²{234}²). Every point has leave ≡ 2 (mod 3) — leaves touch
    ALL points; minimal shapes = doubled parallel classes with mod-4
    residue adjustments (weights 4,8,8,10,12,16,16,18 at m=6..27).
    General-m: same GKLO route once the minimal-leave classification is
    written; assigned.

**Zarankiewicz consequence (with Theorems 8 + F).** z(m,n;3,3) =
3n + min(T₃,₃(m), ⌊(B−n)/3⌋) for all n ≥ T₃,₃(m), with T given by the
spectrum: fully effective and proven for every admissible m and every
solved member of the other families; for large class-m/3|m members exact
with the ineffective-m₀ caveat. The wide region of the (3,3) Zarankiewicz
problem is closed to the exact extent that the design-existence frontier
allows — and the reduction is now two-directional: every future packing
number instantly yields a Zarankiewicz region, and vice versa.

---

## Theorem 11 — status upgrade (design prover final, 2026-07-29)

- **T₃,₃(27) = 1458 = J(27)** — NEW (first 3|m value beyond published
  data); order-9-symmetric witness, coordinator-verified (1458 blocks,
  max coverage 2, leave weight 18 exactly as classified, all point-leaves
  ≡ 2 mod 3).
- **3|m family: PROVEN classification.** Minimal congruence leave weight
  = B − 4J = 2m/3 + 2·[m ≡ 9 (mod 12)], attained by the doubled parallel
  class (m ≢ 9 mod 12; unique among fully-doubled shapes) resp. the
  doubled hub family (m ≡ 9 mod 12). Johnson is never divisibility-
  blocked on this family.
- **Existence citation PINNED**: Glock–Kühn–Lo–Osthus, Mem. AMS 284
  (2023) no. 1406, Theorem 1.1 — (F,λ)-divisibility and typicality
  hypotheses verified line-by-line for both punctured hosts; λ = 2 makes
  the pair-divisibility row automatic. Independent second route: Keevash
  (arXiv:1401.3665); third: Delcourt–Postle (arXiv:2402.17855).
- **Theorem F adversarially re-verified** (750 weight-6 leaves survive the
  point congruence alone — pair parity is load-bearing; one write-up
  subtlety at k₅=1 found and closed; full hand proof of the weight-6
  classification now written in design_prover/J_minus_2_proof.md).
- **Literature verdict** (8 targeted searches): λ=1 packing is finished
  (Ji 2004; Bao–Ji); NO determination of D₂(v,4,3), no Johnson-slack
  statement, no pentagon-leave characterization found. In standard
  notation: **PDN₂(v,4,3) = U₂(v,4,3) − 2 on the class — apparently new.**

**The spectrum now reads (final form):**
  admissible m: T = C(m,3)/2      [Hanani — all m, effective]
  m ≡ 3 (4), 3∤m: T = J − 2       [Thm F all m (UB); {7,11,19,23} explicit;
                                   GKLO all large m; middle range = finite
                                   ILPs, m=31/35 retry queue running]
  3|m: T = J                       [classification + {6..18, 27}; GKLO
                                   all large m]
With Theorems 8 + F: z(m,n;3,3) = 3n + min(T, ⌊(B−n)/3⌋) on n ≥ T is now
closed-form-with-proofs across all three families to the stated extents.

---

## Theorem 12 (the Johnson ladder) — PROVEN; the deep band's degree-visible law

For every level k ≥ 4, the supporting line of the (integer-)convex curve
w ↦ C(w−1,2) at weights {k, k+1} gives a per-point valid inequality: for
every point x of any 2-fold triple packing by blocks of weight ≥ 3,

    (k−1)·y_x + μ_k·d_x ≤ 2C(m−1,2),   μ_k = (k−1)(4−k)/2 ∈ ℤ,

(y_x = Σ_{B∋x}(w−3), d_x = #blocks through x; proof: C(w−1,2) has second
difference 1, so the chord at {k,k+1} minorizes it; sum over blocks
through x). Level k=4 is Lemma C. Summing over points yields, for each k:
(k−1)W + μ_k·E ≤ 2m·C(m−1,2) — an infinite family of linear constraints.

**The ladder LP** (slots ≤ B, columns ≤ n, all ladder levels) bounds
z(m,n;3,3) within ≤ 3.8 of the truth on ALL 57 known deep-band cells of
rows 6–9 (avg gap ≈ 1.7), where the waterfill/Roman bound drifts by up to
8+. It is the first closed-form bound family that tracks the deep band's
shape. LIMIT (honest): at the extreme diagonal the ladder adds almost
nothing over waterfill ((16,16): 136 vs truth 128) — the corner
obstructions are cap/ovoid geometry, provably invisible to degree
accounting. Division of labor: ladder for the band, geometry for the
corner. Diagonal ladder values: (17,17) ≤ 150, (18,18) ≤ 165, (19,19) ≤ 180.

---

## Lemma G (deletion averaging + propagation closure) — classical tool, systematized

Every minor of a K₃,₃-free matrix is K₃,₃-free, so (summing E − d_r over
row deletions, etc.):
  z(m,n) ≤ ⌊ m·z(m−1,n)/(m−1) ⌋,  z(m,n) ≤ ⌊ n·z(m,n−1)/(n−1) ⌋,
  z(m,n) ≤ ⌊ mn·z(m−1,n−1)/((m−1)(n−1)) ⌋.
(The averaging is the classical Kővári–Sós–Turán device; the contribution
here is its use as an exact-value PROPAGATOR.) Iterating to fixpoint from
every known/determined cell (analysis grid m ≤ 26, n ≤ 60):
- 29 cells where the propagated ceiling independently equals Theorem 8's
  formula (redundant confirmations, recorded).
- Diagonal ceilings: z(17,17) ≤ 144, z(18,18) ≤ 160, z(19,19) ≤ 177,
  z(20,20) ≤ 194, z(21,21) ≤ 213, z(22,22) ≤ 232. The quadratic window
  law's predictions (138, 150, 164, 180, 198, 218) sit strictly inside
  every window.
- CASCADE: each settled diagonal re-propagates; z(17,17)=138 would give
  z(18,18) ∈ [150, 154], then z(19,19) ∈ [164, ~167] — window widths
  collapse geometrically, making successive diagonals SAT-tractable.

---

## THE LEVEL FORMULA (master candidate for ALL m, n) — session 5

**Statement (conjecture-scaffold, closed-form pass verified).**

  z(m,n;3,3) = max over w ≥ 3 of
      [ (w−1)·n + min( n, 𝔇_w(m), ⌊(B − C(w−1,3)·n) / C(w−1,2)⌋ ) ]

where B = 2C(m,3) and 𝔇_w(m) = D₂(m,w,3) is the level-w packing number
(max weight-w blocks, every triple ≤ 2), spectrum-corrected per level.
Level 3 = Culík; level 4 = Theorem 8 (proven); the two-layer structure at
every scale matches the deepband agent's decoded frontier geometry.

**Built-in asymptotics**: at the diagonal the maximizing level scales as
w* ≈ (2m²)^{1/3} ≈ 1.26·m^{2/3} — Brown's column-weight law emerges from
the budget arithmetic alone, and z ≈ (w*−1)m ~ Θ(m^{5/3}).

**Closed-form pass (Johnson-capped supplies, pure arithmetic)**: with
levels w ≥ 3, the formula never undershoots on any of the 185 known cells
(m ≤ 16) and overshoots by only 0–5 (single 8 at the (16,16) ovoid
corner); the overshoot pattern is exactly the known supply slack (m=7/11
cells at +2 = the class defect; higher-level Johnson caps unpinned).

**Program to exactness**: (i) exact 𝔇_w table (computing); (ii) per-level
spectrum theory — the level-w analogues of Lemma E / Theorem F / Theorem
11 (divisibility congruences at level w, minimal-leave defects, GKLO
completability) turn every 𝔇_w into closed form for large m; (iii)
residual = mixed-layer corrections (the m=9 hexad phenomenon), to be
classified. Each ingredient is a design-theoretic quantity with its own
proof pipeline — the formula generalizes because its parts do.

---

## THE LEVEL FORMULA — final form (level theorist integrated, session 5)

**The formula (mixed-ledger form = Theorem 7 generalized to all levels):**
z(m,n;3,3) = 3n-shifted maximum over mixed level-profiles under the supply
LEDGER — the two-level expression is the generic case; adjacent-level
mixing is the correction. **Verified: ALL MATCH on rows 6–9 (70 cells,
exact supplies, both S₉ bracket endpoints).** Strict two-level is FALSE in
general (the (7,8) optimum needs levels {6,5,4} — found by the exact-
supply test, invisible under Johnson caps).

**The ingredient theory (spectrum at every level; level_theory/spectrum.md):**
- Lemma L3: level-w leave congruences; defect trichotomy
  (quadratic/linear/gapped) by residue class; admissibility residues
  machine-derived (w=5: m ≡ {2,5,11} mod 15; w=6: {2,6,12,16} mod 20; …).
- Wedge theorem + complement-zone: proven by hand, including a value
  (𝔇₈(11)=3) predicted before the MILP confirmed it.
- New exact values: 𝔇₆(11)=14, 𝔇₆(12)=22 (= the Hadamard 3-(12,6,2)),
  𝔇₆(13)=26. Leave-feasibility IP theory-tight on the design zone with
  ONE open gap: (11,7).
- **Completion obstructions — structurally new at w ≥ 5**: congruence-valid
  leaves that provably do not complete ((10,5), (11,5), (12,7)). The
  (11,5) kill rediscovers **Dehon 1976** (no 3-(11,5,2)) — Dehon joins
  Turán (row 6), Hanani (w=4), and the Bose ovoid ((16,16)) as classical
  results absorbed as spectrum special cases.
- Theorem L4: GKLO closure at every level (perfect classes exact for all
  large m).

**The diagonal limit (level_theory/diagonal_limit.md):** the budget-only
formula constant is exactly 2^{1/3}; the true constant is 1 (Brown/
Füredi). The difference IS Füredi's theorem, recast as supply-density
(φ(c) < 1 for c > 1, φ(1) = 1 by Brown). Finite-size gem:
2^{1/3}·16^{5/3} = 2⁷ = 128 = z(16,16) — the table sits ON the budget
curve at m=16; the 1.26 → 1.00 descent is the asymptotic form of the
demand-vs-supply principle. The formula thus interpolates correctly from
Culík (level 3) through Theorem 8 (level 4) and the band ledgers to the
Brown–Füredi corner, with the supply corrections carrying exactly the
known asymptotic content.

---

## THE MASTER THEOREM (formula-level closure) — session 6 scaffold

**Structure.** Combining the proven pieces:
(1) z = ledger max (exact Pareto identity — PROVEN, Theorem 9);
(2) per-level minimal-leave weights L_min(w, ·) are PERIODIC in m (pure
    congruence/CRT arithmetic per level — periods P₄ = 12, P₅ | 60,
    P₆ | 60, ... to be tabulated), adjusted by the finite completion-
    obstruction classifications (Dehon-type sporadics);
(3) GKLO attains the minimal leave for all large m at every level
    (Theorem L4) — hence, for every w:

    𝔇_w(m) = ( 2C(m,3) − L_min(w, m mod P_w) ) / C(w,3)
              for all sufficiently large m — EXPLICIT CLOSED FORM;

(4) bounded-mixing (conjecture, exchange-lemma support): the ledger max is
    within an absolute constant of the best two-level value outside the
    corner scale.

**Master statement (target form).** There is an explicit function F(m,n) —
eventually periodic in m at each level, piecewise in n across levels — and
an absolute constant C such that for ALL m, n:
    F(m,n) − C ≤ z(m,n;3,3) ≤ F(m,n),
with EQUALITY z = F on: all n ≥ T₃,₃(m) for every admissible m (fully
effective); every solved family member; all verified bands; and — modulo
(4) — everywhere outside the corner scale n = Θ(m²/w*). The deviation set
is confined to the corner, where determining z exactly is EQUIVALENT to
the second-order Brown–Füredi problem (open since 1966/1996; the formula's
budget constant 2^{1/3} vs the true 1 IS that problem, per
diagonal_limit.md).

**Honest caveats bound to the statement**: ineffective thresholds (GKLO);
sporadic obstruction classes (finite per level, classified so far at
w ≤ 8); bounded-mixing unproven in general (verified on all data;
exchange lemmas partial). This is the maximal closure the current
mathematical frontier permits: an explicit generalizable formula, exact on
the overwhelming majority of the parameter space, bounded-error elsewhere,
with its residual EXACTLY identified as the field's central open problem.

---

## Lemma H (min-degree deletion) — PROVEN [Opus session, 2026-07-30]

Let M be m×n, K₃,₃-free, with E ones, row degrees r_i, column degrees c_j,
and put Z = z(m−1,n−1;3,3), d = E − Z. Since every (i,j)-minor is
K₃,₃-free:
        r_i + c_j − a_ij ≥ d        for all i, j.        (∗)
Let ρ = min r_i, γ = min c_j, A = {i : r_i = ρ}, B = {j : c_j = γ}. Then:
(a) ρ ≥ d − γ and γ ≥ d − ρ; if ρ + γ < d the matrix cannot exist;
(b) |A| ≥ m − (E − mρ) and |B| ≥ n − (E − nγ) (excess counting);
(c) **if ρ + γ = d then M[A×B] ≡ 0**, so the A-rows pack ρ|A| ones into
    n − |B| columns, forcing ρ·|A| ≤ z(|A|, n − |B|).
Validated: 0 false kills across all 161 published exact values.

### Corollary (new bound). **z(17,17;3,3) ≤ 143.**

*Proof.* Suppose E = 144. Then d = 144 − z(16,16) = 144 − 128 = 16.
1. If some r_i ≤ 7 then (∗) forces c_j ≥ 9 for every j, so E ≥ 17·9 = 153
   > 144. Hence all r_i ≥ 8, and symmetrically all c_j ≥ 8.
2. Σ(r_i − 8) = 144 − 136 = 8, so at least 9 rows have r_i = 8 exactly;
   likewise at least 9 columns have c_j = 8.
3. For such a row i and column j, (∗) reads 8 + 8 − a_ij ≥ 16, so a_ij = 0:
   the (≥9)×(≥9) block of minimum-degree rows and columns is ZERO.
4. Pick 9 such rows and 9 such columns. Those 9 rows must place all 8 of
   their ones among the remaining 8 columns — i.e. they are all-ones there.
   A 9×8 all-ones block contains K₃,₃. Contradiction. ∎

This improves the previous workspace bound (144, deletion averaging) and,
propagated, also gives z(18,18) ≤ 159, z(19,19) ≤ 176, z(20,20) ≤ 193,
z(21,21) ≤ 212, z(22,22) ≤ 229, plus 12 off-diagonal improvements.
**z(17,17;3,3) ∈ [138, 143] — still NOT determined.**

*Novelty caveat*: minor-deletion averaging is the classical KST device;
the min-degree/zero-block refinement is elementary and may be folklore.
Comparisons against DGH 2024's printed table (workspace-transcribed, and
this workspace has already produced one transcription error) suggest our
propagated bounds beat several published values — that comparison is NOT
trustworthy without reading the actual papers and is recorded as
UNVERIFIED, not claimed.

## Lemma I (extremal-extension rigidity at (16,16)) — [Opus session]

**Computed fact.** The known extremal z(16,16)=128 configuration (16 columns
= both sides of the 8 cap-normal hyperplanes of F₂⁴) has exactly 448 triples
at capacity 2, 112 triples at coverage 0, and **no triple at coverage 1**.
Scanning ALL 2¹⁶ subsets: the maximum weight of an addable further column
is **4** (e.g. {0,1,8,9}), giving the extension 128+4 = 132 — which exactly
matches the known lower bound z(16,17) ≥ 132.

**Consequence (CONDITIONAL on uniqueness of the (16,16) extremal, claimed by
CRWR 2016, hedged by Tan 2022).** For a 16×17 matrix with E ones, deleting a
minimum-degree column gives E − c_min ≤ 128 and c_min ≤ ⌊E/17⌋:
  E = 136 ⟹ c_min = 8 ⟹ deletion leaves an EXTREMAL 16×16 plus an addable
            weight-8 column — impossible (max 4).
  E = 135 ⟹ c_min = 7 ⟹ same with weight 7 — impossible.
  E = 134 ⟹ c_min ∈ {6,7}; c_min = 6 needs weight-6 addable (impossible),
            so c_min = 7 and no extremal is forced.
Hence **z(16,17) ≤ 134**, and by row-deletion **z(17,17) ≤ ⌊17·134/16⌋ = 142**
— both CONDITIONAL. The unconditional bounds remain z(16,17) ≤ 136 and
z(17,17) ≤ 143 (Lemma H).

**Status of the diagonal: z(17,17;3,3) ∈ [138, 143] unconditionally,
[138, 142] conditionally. NOT DETERMINED.**

## Why no diagonal formula exists (evidence, not proof)

z(m,m) for m=3..16: 8,13,20,26,33,42,49,60,69,80,92,105,120,128.
Normalised z(m,m)/m^{5/3}: 1.28,1.29,1.37,1.31,1.29,1.31,1.26,1.29,1.27,
1.27,1.28,1.29,1.32,1.26 — hovering near the budget constant 2^{1/3}≈1.26
but fluctuating irregularly (1.26–1.37). The dips are exactly where an
exceptional geometry exists (m=9 AG(2,3), m=16 ovoid); the peaks where none
does. A sequence governed by which sporadic geometry happens to exist at
each order does not admit a closed form — matching Tan 2022's published
assessment ("no discernible pattern other than strict monotonicity").

---

## Theorem N (the Describability Dichotomy) — [Opus/Fable session, 2026-07-30]

The question "can z(m,n;3,3) be closed by a formula?" is itself decidable,
and the answer is a dichotomy with an explicit boundary.

**(a) NEGATIVE HALF — PROVEN, unconditional.** No finite family of
(quasi-)polynomial pieces computes z(m,n;3,3) on any cone containing the
diagonal ray {n = m}. *Proof.* Suppose z agrees, on some sublattice
residue class L of a diagonal-containing cone, with a polynomial p(m,n)
for all large (m,n) ∈ L. Restricting to the diagonal points of L gives a
single-variable polynomial q(m) = p(m,m) with z(m,m) = q(m) for large m in
an arithmetic progression. But c₁m^{5/3} ≤ z(m,m) ≤ c₂m^{5/3} with
c₁, c₂ > 0 (Brown 1966; Kővári–Sós–Turán 1954 — classical,
unconditional). If deg q ≤ 1, q(m)/m^{5/3} → 0 < c₁; if deg q ≥ 2,
q(m)/m^{5/3} → ∞ > c₂. Contradiction either way. ∎
[Numeric display: z/m grows (5.2, 6.7, 8.0 at m = 8, 12, 16), z/m² decays
(.66, .56, .50), z/m^{5/3} is stable (1.31, 1.27, 1.26).]
The same argument kills any formula class whose diagonal restrictions have
integer growth degrees — piecewise polynomial, quasi-polynomial,
polynomial-with-floors — i.e., every "Ehrhart-shaped" closure.

**(b) POSITIVE HALF.** Away from the diagonal cone, z IS eventually
quasi-polynomial with explicit lattices: on n ≥ B(m): z = 2n + 2C(m,3)
(polynomial; Culík). On the window: z = 3n + ⌊(B−n)/3⌋ (lattice
(m mod 9, n mod 3)). On the supply band: z = 3n + T₃,₃(m) with T₃,₃
quasi-polynomial of period 12 (spectrum theorem; caveats as stated there).
Verified exact on 80/80 known cells with n ≥ T₃,₃(m). Deeper fixed-level
sectors: EQP with periods P_w (L_min tables), same caveat structure. The
level count needed diverges as (m,n) approaches the diagonal ray — the
positive description and the negative theorem meet at the same boundary.

**Interpretation.** The closure boundary of z(m,n;3,3) is now a THEOREM:
formula-closure holds on every fixed-level sector and is impossible on the
diagonal cone — where the truth grows like m^{5/3}, an exponent no
polynomial-type formula family can express. What survives inside the cone
is algorithmic/structural description (the ledger; the conjectural finite
catalogue), never a formula. This resolves the "find a generalizable
formula closing all m,n" quest with a proof of exactly how much of it is
mathematically possible.

---

## ERRATUM to Theorem 12 + the M/P theorem block [asymptotic analyst, verified]

**Erratum.** Theorem 12's ladder is asymptotically VACUOUS: every summed
ladder row is coefficient-dominated by the slot budget (3-line proof,
machine-verified), so ladder-LP = budget-LP at every (m,n); the diagonal
values previously printed as ladder output (136/150/165/180) are exactly
WF. The ladder's finite-band tightness claims stand; its asymptotic
content is zero. Recorded per honesty bar.

**Theorem M1 (link second-moment; PROVEN, elementary).** Per point x with
d blocks through it, link sizes u_i, Y = Σu_i, M = m−1: the links'
pair-multiplicities are ≤ 2 (else K₃,₃), and the Fisher count + two Jensen
steps give the finite bound Y ≤ G(d) := M + √(2M·(C(d,2)+√(2C(d,2)C(M,2)))).
[Coordinator-verified at all 16 points of the (16,16) extremal (worst
ratio 0.788) and on Brown q=7.]

**Theorem M2 (elementary diagonal bound).** Chaining M1 over points:
z(m,m;3,3) ≤ 2^{1/6}·m^{5/3}(1+o(1)) ≈ 1.1225·m^{5/3} — strictly below
the budget/ladder constant 2^{1/3} ≈ 1.2599, by ELEMENTARY means. (Füredi's
theorem is stronger — constant 1 — via harder machinery; the delta here is
elementarity + M4's finite effective form, which beats WF/Roman for every
180 ≤ m ≤ 924 where no effective Füredi bound exists.)

**Theorem P1 (the supply-density function, exact).** φ(c) = c⁻³ on (0,1],
φ(c) = 0 for c > 1 — closing diagonal_limit.md's open item and refuting
all min(1, c^{−α}) shapes. **Theorem P2 (half-budget law):**
𝔇_w(m) = (1+o(1))·(m/w)³ for √m ≪ w ≤ m^{2/3}: maximum packings use
exactly HALF of triple capacity on average (coverage-1 law) — the entire
2^{1/3} → 1 gap is that factor 2 under a cube root.

**Conjecture B (second-order diagonal, falsifiable).** Brown's construction
satisfies the EXACT identity e = n^{5/3} − n^{4/3} at every n = q³
(coordinator-verified at q=7: 14406 exactly; sphere size q²−q), and the
q=7,11 matrices are measured 1-maximal. Conjecture:
z(q³,q³;3,3) = n^{5/3} − n^{4/3} + o(n^{4/3}), i.e. c₂ = −1, with proven
window c₂ ∈ [−1, 2]. This is a precise second-order conjecture on the
Brown–Füredi problem — the first the workspace can state with structural
support on both sides.

---

## THE UNIFIED CONE FORMULA (capstone of the 8-hour campaign)

One architecture now spans the ENTIRE parameter cone of z(m,n;3,3):

    z(m,n;3,3) = max over levels w of [(w−1)n + min(n, 𝔇_w(m), budget_w)]

with the supplies 𝔇_w given by:
- levels 3–4: EXACT (Culík; the T-spectrum: Hanani / J−2 class law / 3|m
  law) — proofs;
- band levels: the ledger with periodic L_min tables — verified 168/206
  frozen, exact wherever supplies computed;
- corner levels (√m ≪ w ≤ m^{2/3}): the HALF-BUDGET LAW (P2):
  𝔇_w = (1+o(1))(m/w)³ — maximum packings run at coverage 1 of capacity 2.

Consequences across the cone:
- n ≥ B: z = 2n + B (exact).
- T ≤ n ≤ B: z = 3n + min(T, ⌊(B−n)/3⌋) (exact; T by the spectrum).
- The band: ledger-exact (verified; supplies computable).
- The DIAGONAL: the level maximization with P2 supplies peaks at
  w* ~ m^{2/3} with value m^{5/3}(1+o(1)) — the formula now DERIVES
  Füredi's constant 1 (previously the budget-only version gave 2^{1/3};
  the entire correction is the half-budget factor 2 under a cube root).
  Second order: Conjecture B (c₂ = −1 at Brown orders; window [−1,2]
  proven; Brown's own count is EXACTLY n^{5/3} − n^{4/3}).

Status ledger for the whole object: exact-with-proofs (wide region),
exact-verified (bands), first-order-exact (corner; P2/M3), second-order
conjectural (B). Together with Theorem N (no polynomial-type formula can
do better near the diagonal — the m^{5/3} exponent forbids it), this is
the maximal formula-closure of z(m,n;3,3) expressible with current
mathematics: the formula is now exactly as strong as the state of the
Brown–Füredi problem allows anything to be.

---

## Theorems S1–S3 + the species catalogue (structure theorist, final)

**Theorem S1 (Roman-window rigidity; all m, unconditional).** Every optimal
configuration at E = 3n + ⌊(B−n)/3⌋ is quads + weight-3 fills (+ at most
one pad iff (B−n) ≡ 2 mod 3), slot deficit ≤ 2.
**Theorem S2 (T-branch purity).** Every heavy configuration of value
T₃,₃(m) is PURE QUAD — proven by hand for admissible and all 3|m (new
mod-3 + pentad-pair counting), and for all class m (hand cases + four
abstract-support MILP infeasibility certificates with proven support
bounds). **Corollary S3**: every wide-region optimum = maximum packing ⊕
fills in its leave (verified 22/22 in-scope witnesses). The wide region is
now closed in the STRONG sense: value AND complete structure of all
optima. Bonus: **S₇(1) = 12 and S₉(1) = 37 proven** (previously open).

**The species catalogue.** Instance-finiteness is REFUTED (≥3
non-isomorphic maximum packings at m=9); SPECIES-finiteness — the actual
conjecture — held at **16 species** across the full audited record (81
witnesses: 51 fully decomposed, 105/142 heavy layers decoded), growing
only by absorbing two more classical objects (Möbius–Kantor configuration,
Möbius-plane residual orbits). Notable decodes: the (10,15) holdout =
Möbius-residual cone + AG(2,3) parallel-class complements (fully
classical); w_10x58 completes a NON-doubled 3-(10,4,2); w_10x47 =
OA(8,5,2,2) transversal design. Honest undecoded core: deep-band ledger
bodies (named; all ledger-certified, none structureless). WQO: proven not
a route (ambient antichain; extremal class not downward-closed).
**Record corrections**: frontier_m9_final F₉(27) stale (true val ≥ 35);
deepband §5 placeholders were never resolved — w_9x24..27 are LB
witnesses for OPEN cells.

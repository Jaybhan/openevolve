# The existence citation for large-m attainment — pinned

design_prover, 2026-07-29. Status: statements below are quoted from the
arXiv v3 PDF of the paper (fetched and read today); the application checks
are PROVEN (elementary arithmetic, machine-verified in
`scripts/verify_threefold.py` and by hand below).

## 1. The theorem

**Source.** S. Glock, D. Kühn, A. Lo, D. Osthus, *The existence of designs
via iterative absorption: hypergraph F-designs for arbitrary F*,
Mem. Amer. Math. Soc. 284 (2023), no. 1406 (arXiv:1611.06827; the v3 PDF
combines arXiv:1611.06827v1 and arXiv:1706.01800). Below "GKLO".

**Definitions quoted from GKLO Section 1 (arXiv v3, pp. 1-3):**

- (F,λ)-design: "an (F,λ)-design of G is a collection F of **distinct**
  copies of F in G such that every edge of G is contained in exactly λ of
  these copies."
- Typicality: "An r-graph G on n vertices is called (c,h,p)-typical if for
  any set A of (r−1)-subsets of V(G) with |A| ≤ h we have
  |∩_{S∈A} G(S)| = (1±c) p^{|A|} n."  (G(S) = link of S.)
- Divisibility vector: "for a (non-empty) r-graph F, we define the
  divisibility vector of F as Deg(F) := (d₀,...,d_{r−1}) ∈ N^r, where
  d_i := gcd{|F(S)| : S ∈ (V(F) choose i)}"; and "G is called
  (F,λ)-divisible if Deg(F)_i | λ|G(S)| for all 0 ≤ i ≤ r−1 and all
  S ∈ (V(G) choose i)."

**GKLO Theorem 1.1 (F-designs in typical hypergraphs), quoted:**
"For all f, r ∈ N with f > r and all c, p ∈ (0,1] with
c ≤ 0.9(p/2)^h/(q^r 4^q), where q := 2f·f! and h := 2^r (q+r choose r),
there exist n₀ ∈ N and γ > 0 such that the following holds for all
n ≥ n₀. Let F be any r-graph on f vertices and let λ ∈ N with λ ≤ γn.
Suppose that G is a (c,h,p)-typical r-graph on n vertices. Then G has an
(F,λ)-design if it is (F,λ)-divisible."

Secondary anchors:
- P. Keevash, *The existence of designs* (arXiv:1401.3665): per GKLO's
  own introduction, Keevash's result is "also stated in the setting of
  typical r-graphs, but additionally requires that c ≪ 1/h ≪ p, 1/f and
  that λ = O(1) and F is complete" — our application (F = K₄⁽³⁾ complete,
  λ = 2, p = 1) fits those hypotheses too, so BOTH proofs independently
  cover it.
- M. Delcourt, L. Postle, *Refined absorption: a new proof of the
  existence conjecture* (arXiv:2402.17855) — a third proof route; not
  needed, listed for robustness.

## 2. Application 1: the class m ≡ 3 (mod 4), m ≢ 0 (mod 3)

Take r = 3, f = 4, F = K₄⁽³⁾ (so Deg(F) = (4, 3, 2): d₀ = |F| = 4
edges, d₁ = |F(v)| = 3 for every vertex, d₂ = |F(pair)| = 2), λ = 2,
p = 1, c = the constant of Theorem 1.1, and

    G := K_m^(3) − P,   P = the pentagon: the 5 cyclic-interval triples
    {i, i+1, i+2 (mod 5)} on a fixed 5-set H (= the C₅-edge-complements).

An (K₄⁽³⁾, 2)-design of THIS G is exactly a multiset (in fact a simple
family) of quads covering every triple outside P exactly twice and every
triple of P zero times (copies of F in G cannot use edges outside G):
i.e. a maximum-leave packing with b = 2|G|/4 = (2C(m,3) − 10)/4 =
(B−10)/4 = J − 2 blocks, since B ≡ 2 (mod 4) on the class. Checks:

(i) TYPICALITY (p = 1): for any set A of ≤ h pairs,
    n ≥ |∩_{S∈A} G(S)| ≥ n − 2h − 15, because the intersection excludes
    only the ≤ 2h points of ∪A and, per pentagon triple T and pair S ⊂ T,
    the single third point (≤ 5·3 = 15 exclusions in total). So G is
    (c, h, 1)-typical whenever m ≥ (2h+15)/c. PROVEN.
(ii) (F,2)-DIVISIBILITY:
    - i=0:  4 | 2|G| = B − 10 ≡ 2 − 10 ≡ 0 (mod 4).  [class: B ≡ 2 mod 4]
    - i=1:  3 | 2|G(x)| = 2(C(m−1,2) − deg_P(x)), deg_P(x) ∈ {0, 3};
            class: 3 | C(m−1,2). ✓
    - i=2:  2 | 2|G(xy)| — AUTOMATIC at λ = 2 (this is precisely why the
            pair congruence, which blocks J and J−1, cannot block J−2:
            the doubled pentagon has all pair-degrees even).
(iii) λ = 2 ≤ γn for n ≥ n₀. ✓

**Conclusion (PROVEN modulo GKLO):** there is m₀ such that every class
m ≥ m₀ admits a J−2 packing with doubled-pentagon leave; with Theorem F,
**T₃,₃(m) = J(m) − 2 for every class m ≥ m₀.**

CAVEAT (stated plainly): m₀ is not explicit. Theorem 1.1's constants for
(f,r) = (4,3) are q = 192, h = 2³·C(195,3) ≈ 9.7·10⁶, c ≤ 0.9·2^{−h}/
(192³·4^{192}) — astronomically small, and n₀ from iterative absorption is
not stated in a computable form. The proof is constructive in principle
but no explicit bound is published; the result is an
ineffective-threshold theorem. Small-m gap covered so far by explicit
witnesses: m ∈ {7, 11, 19, 23}; all other class m < m₀ remain formally
open individually (finite ILP each; σ-scheme reduces the size).

## 3. Application 2: the family 3 | m (leave families from the
   classification in C7_general.md Section 3)

Same F, λ = 2, p = 1. Two sub-cases; L' below is a SIMPLE 3-graph and the
prescribed maximum-packing leave is 2L' (doubled):

(a) m ≡ 0 (mod 3), m ≢ 9 (mod 12): L' = a parallel class (m/3 disjoint
    triples); G := K_m^(3) − L'. Then 2|G| = B − 2m/3 = mR ≡ 0 (mod 4)
    [verified: mR mod 4 = 0 exactly on this subfamily], so an
    (F,2)-design of G has mR/4 = J blocks. Divisibility: i=0 ✓ as
    computed; i=1: 2(C(m−1,2) − 1) ≡ 2·1 − 2 = 0 (mod 3), using
    C(m−1,2) ≡ 1 (mod 3) for 3|m and deg_{L'}(x) = 1 for EVERY x; i=2
    automatic. Typicality as in Application 1 (L' removes ≤ 1 link-point
    per pair, ≤ m/3 triples but only O(h) affect any ∩: the same
    n − 2h − 3·(pairs inside L'-triples ∩ A)... each pair S lies in ≤ 1
    triple of L', so |∩| ≥ n − 2h − h ✓).
(b) m ≡ 9 (mod 12): L' = the hub family: 4 triples {z,a_i,b_i} (i=1..4)
    through a common point z with the 8 points a_i,b_i distinct, plus a
    parallel class of (m−9)/3 triples on the remaining m−9 points.
    deg_{L'}(z) = 4, deg_{L'}(x) = 1 for all x ≠ z; |L'| = m/3 + 1.
    2|G| = B − 2m/3 − 2 = mR − 2 ≡ 0 (mod 4) [mR ≡ 2 (mod 4) on this
    subfamily], so an (F,2)-design of G has (mR−2)/4 = ⌊mR/4⌋ = J blocks.
    Divisibility i=1 at z: 2(C(m−1,2) − 4) ≡ 2 − 8 ≡ 0 (mod 3) ✓; other
    points as in (a); i=0, i=2 ✓. Typicality: each pair lies in ≤ 4
    triples of L' (pairs {z,a_i}? z-pairs lie in ≤ 1 each... every pair
    lies in ≤ 1 triple of L' since the triples share only z and a pair
    containing z lies in ≤ 1); same count ✓.

**Conclusion (PROVEN modulo GKLO):** T₃,₃(m) = J(m) for all 3|m ≥ m₀'
(upper bound J is Johnson's, classical), same ineffectiveness caveat;
explicit small cases m = 6, 9, 12, 15, 18 (Tan) and m = 27 (this
workspace, in progress).

## 4. Why the multigraph detour is unnecessary (a note for referees)

Our maximum packings live in the multigraph 2K_m^(3); GKLO's hosts are
SIMPLE r-graphs and their designs use DISTINCT copies. The bridge is that
our target leaves are DOUBLED simple graphs (2·pentagon, 2·L'), so the
host 2K_m − 2L' = 2(K_m − L') and we ask GKLO for an (F,2)-design of the
SIMPLE graph K_m − L': "every edge in exactly 2 of the distinct copies".
No multigraph decomposition theory is needed. (A split into two λ=1
designs would NOT work: K_m − pentagon is never (F,1)-divisible for odd m
— every pair degree m − 2 − deg_P is odd for the ≥ m−5 pairs off the
hole — which is precisely why λ = 2 with parity-free i=2 is the right
frame.)

## 5. Honest summary

- Citation pinned: GKLO Mem. AMS 284 (2023) no. 1406, Theorem 1.1
  (quoted verbatim above); Keevash arXiv:1401.3665 independently
  suffices; Delcourt-Postle arXiv:2402.17855 as third route.
- Both applications' hypotheses VERIFIED line by line (typicality with
  p=1, all three divisibility rows, λ ≤ γn).
- The threshold m₀ is ineffective-in-practice; every statement using it
  is labeled "for all sufficiently large m" — no exceptions claimed.

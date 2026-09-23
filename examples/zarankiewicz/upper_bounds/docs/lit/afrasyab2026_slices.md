# Afrasyab (2026) — Exact Zarankiewicz Values On Two Finite Frontier Slices

Literature note for the MEng thesis "Discovering upper bounds for the Zarankiewicz
numbers z(m,n;s,t)" (evolved, Lean-verified pruning for a case-split SAT attack).
Written 2026-09-21 from the arXiv PDF and the companion repository.

Notation caution: Afrasyab writes `Z(m,n,s,t)` (or `Z(m,n;s,t)`); our code writes
`z(m,n;s,t)`. Both mean: max ones in an `m x n` 0/1 matrix with no all-ones `s x t`
submatrix, m = rows, n = columns. Throughout this note s = t = 3 unless stated.

Tags: **[source]** = stated/proved in the paper or its verifier; **[inferred]** = my
derivation or interpretation, not claimed by the author.

---

## 1. Bibliographic information

| | |
|---|---|
| Title | Exact Zarankiewicz Values On Two Finite Frontier Slices |
| Author | Koyar Afrasyab (independent researcher, koyar@kinvectum.com) |
| Identifier | arXiv:2608.08154v1 [math.CO], cross-listed cs.AI; dated 8 August 2026 |
| MSC 2020 | 05C35, 05D99, 68R10, 90C05 |
| Companion repo | https://github.com/KAVentures/z1322-exact (commit `47ef9d8`, 2026-08-08, MIT code / CC BY 4.0 text) |
| Length | 12 pages, 12 sections + 2 appendices (witness listings) |
| Peer review | none; author states "a complete reproducible computer-assisted proof, not a peer-reviewed consensus result" (Sec. 11.4) |
| AI disclosure | "OpenAI GPT 5.6 Sol High was used for assistance with exploratory reasoning, implementation, verification, manuscript integration, and packaging" (Acknowledgements) |
| Local copies | PDF: `scratchpad/lit/afrasyab_2608.08154.pdf`; repo clone: `scratchpad/lit/z1322-exact/` (session scratchpad, not in git) |

Accessible: yes, full text (PDF, arXiv HTML, and the LaTeX source `paper/main.tex`
in the repo). All quotations below are from the LaTeX source or the PDF.

---

## 2. One-paragraph summary

The paper closes several K_{3,3} cells adjacent to the frontier reported in our own
lower-bound paper (Bhan–Nobili–Raghuraman–Langer, arXiv:2605.01120). It does **not**
use SAT. Instead every upper bound is a *certificate-based* exclusion of exactly
`Z+1` ones: (i) enumerate the finitely many column-degree profiles compatible with the
Kővári–Sós–Turán triple count, (ii) for each profile, either produce an exact rational
Farkas certificate showing a "block-polytope" LP relaxation is infeasible, or (iii)
fall into a hand/computer "marked-row" argument that fixes one row, studies the
residual design on the other rows, and closes the case by exhaustive local enumeration
plus more Farkas certificates. Deletion lemmas then propagate exact values to
neighbouring cells. The trust base is "standard-library Python and exact
integer/rational arithmetic"; floating-point LP was used only to *discover* dual
vectors. The author explicitly lists "replay the theorem in Lean, Isabelle, or Coq"
as still desirable — i.e. nothing here is formally verified.

---

## 3. Results: which cells are closed

**Theorem 1.1** [source, quoted]:

```
Z(12,n,3,3) = 6n            (18 <= n <= 22),
Z(13,22,3,3) = 137,
Z(13,18,3,3) = 116,
Z(14,17,3,3) = 118,   Z(14,18,3,3) = 124,
Z(15,17,3,3) = 126,   Z(15,18,3,3) = 132,
132 <= Z(16,17,3,3) <= 133.
```

**Theorem 1.2** [source, quoted]: "There is no 13x22 binary matrix with 138 ones and
no all-one 3x3 submatrix. Consequently, Z(13,22,3,3) = 137."

Provenance of each cell (what is new, what is imported):

| Cell | Value | Lower bound from | Upper bound from |
|---|---|---|---|
| (12,17) | 103 | 103-witness in package (col degrees 7,6^16) | 303 orbit certificates (12-row package; stated in `proof/z12_18_21/main.tex` Thm, only implicit in the arXiv paper) |
| (12,18) | 108 | 108-witness, all columns degree 6 | **load-bearing new certificate**: 109 ones forced into 4 cases A/B/I/O, each excluded by an orbit certificate |
| (12,19..21) | 114,120,126 | first 19/20/21 blocks of the 3-(12,6,2) Hadamard design | Lemma 2.1 (min-column deletion) chained from (12,18) |
| (12,22) | 132 | classical design; prior (our paper) | deletion chain; "retained as prior material", not claimed new |
| (13,22) | 137 | our paper's 137-edge construction (re-checked, listed in App. B) | **new**: 138 ones excluded via 83 profiles |
| (13,18) | 116 | 116-witness (our paper had 115) | **new**: 117 ones excluded via 19 profiles |
| (14,17) | 118 | 118-witness | row deletion from *published* Z(13,17) <= 110 (Collins et al.) |
| (14,18) | 124 | 124-witness | row deletion from Z(13,18) <= 116 |
| (15,17) | 126 | 126-witness | row deletion from Z(14,17) <= 118 |
| (15,18) | 132 | 132-witness | row deletion from Z(14,18) <= 124 |
| (16,17) | [132,133] | 132-witness (new) | published 133 (Collins et al.) — interval only |

Also in the verifier but not in Theorem 1.1: `Z(11,18) <= 101` (51 certificates) and
`Z(13,17) <= 111` derived from `Z(12,17) <= 103` [source: `frontier_closure/verify_all.py`].

Direct relevance: three of our prior paper's open cells — (12,17): 102..108,
(12,18): 108..113, (13,22): 137..140 — are now closed. They make ideal
**known-answer benchmarks** for the thesis pipeline (see Sec. 9).

---

## 4. Method, layer by layer

### 4.1 Block formulation [source]
Identify column j with its support `E_j ⊆ X = [m]`, `d_j = |E_j|`. For each row
triple T let `λ_T = |{ j : T ⊆ E_j }|`. Then

```
(1)   K_{3,3}-free  <=>  λ_T <= 2  for every T ∈ C(X,3).
```
"Repeated columns are allowed throughout." Deleting ones preserves (1), so to prove
`Z <= w-1` it suffices to exclude matrices with *exactly* `w` ones.

### 4.2 Global reduction at 138 edges (13x22) [source, Sec. 5]
Assume `Σ_j d_j = 138`. Double counting incidences between blocks and row triples:

```
(4)   Σ_{j=1}^{22} C(d_j,3) = Σ_T λ_T <= 2·C(13,3) = 572.
```
Define the integer *penalty* (a supporting line to the convex function C(d,3) at d=6,7):

```
(5)   p(d) = C(d,3) - 15d + 70,   0 <= d <= 13
      d    : 0  1  2  3  4 5 6 7 8  9  10 11 12  13
      p(d) : 70 55 40 26 14 5 0 0 6 19 40 70 110 161
```
`p(d) >= 0` with equality exactly at d = 6, 7. Combining (3)-(5):

```
(6)   Σ_j p(d_j) = Σ_j C(d_j,3) - 15·138 + 70·22 <= 42.
```
"Exact integer enumeration of all histograms (n_0,...,n_13) satisfying Σ n_d = 22,
Σ d·n_d = 138, Σ p(d)·n_d <= 42 leaves exactly 83 degree profiles." (No degree window
is assumed; degrees 0..13 all enumerated.) At 139 edges the same gives 27 profiles.

### 4.3 The block-polytope relaxation and Farkas certificates [source, Sec. 5.1]
For a profile `(n_d)`, choose a degree f with `n_f > 0` and, by row symmetry, fix one
f-block as `F = {1,...,f}`. For every remaining allowed support `B ⊆ X` let `x_B >= 0`
be its multiplicity. Every actual matrix yields an integral feasible point of:

```
(7)  Σ_{B ⊇ T} x_B                <= 2 - 1[T ⊆ F]                       (T ∈ C(X,3))
(8)  Σ_{B ∋ r} C(|B|-1,2) x_B     <= 132 - 1[r ∈ F]·C(f-1,2)            (r ∈ X)
(9)  Σ_{B ⊇ P} (|B|-2) x_B        <= 22 - 1[P ⊆ F]·(f-2)                (P ∈ C(X,2))
(10) Σ_{|B|=d} x_B                 = n_d - 1[d = f]                      (d active)
```
"The row and pair inequalities are nonnegative sums of triple-capacity inequalities;
they are retained because they yield much shorter dual certificates." (132 = 2·C(12,2);
22 = 2·(13-2).)

**Lemma 5.1 (Exact Farkas criterion)** [source, quoted]: "Write (7)–(9) as Ax <= b, and
(10) as Ex = e, with x >= 0. If rational vectors α >= 0 and unrestricted β satisfy
`A^T α + E^T β >= 0, b^T α + e^T β < 0`, then the profile is impossible."
Proof: `0 <= x^T(A^T α + E^T β) = α^T A x + β^T E x <= α^T b + β^T e < 0`.

The verifier "regenerates every subset B of each active degree and checks every dual
coefficient using fractions.Fraction". 77 of 83 profiles are excluded this way. The six
survivors (eq. 11): `6^18 7^3 9^1, 6^16 7^6, 5^1 6^14 7^7, 5^2 6^12 7^8, 4^1 6^13 7^8,
5^3 6^10 7^9`.

Certificate file format (`proof/certs138/pNN.json`): `counts` (profile), `fixed_degree`,
`fixed_block`, `active_degrees`, `alpha` (377 = 286 triples + 13 rows + 78 pairs
rationals as [num,den]), `beta` (one per active degree), `rhs`, `min_coefficient`; the
HiGHS status/objective fields are present but *not read* by the verifier.

### 4.4 Marked-row deficits [source, Sec. 6]
```
δ_T = 2 - λ_T >= 0,     D_r = Σ_{T ∋ r} δ_T,     s = 572 - Σ_j C(d_j,3)  (unused triple capacity)
(12)  Σ_{r ∈ X} D_r = 3s
(13)  D_r = 132 - Σ_{j : r ∈ E_j} C(d_j - 1, 2)
```
"This formula gives both congruence restrictions and a local design interpretation."
Key trick: C(5,2) = 10 and C(6,2) = 15 are 0 mod 5, so for profiles made of degrees 6
and 7, every `D_r ≡ 2 (mod 5)` (with shifts for rows inside exceptional blocks of degree
4, 5, 9). Since `Σ_r D_r = 3s` is small, only a few "increments of size five" are
available, so some row is at its residue-minimal deficit ("deficit averaging": some row
has `D_r <= floor(3s/13)`).

**Proposition 6.1** [source]: the profile `6^18 7^3 9^1` is impossible — a fully
human-readable argument (Sec. 6.1). Sketch: s = 23, ΣD_r = 69; rows outside E_9 have
D_r ≡ 2, rows inside ≡ 4 (mod 5); residue-minimal total 4·2 + 9·4 = 44, so at most 5
increments; a row with D_r = 2 outside E_9 has `2a + 3b = 26` (a, b = number of degree-6
and degree-7 blocks through it) → (13,0) or (10,2), both killed by pair-capacity
counting on the 12 residual points; hence ≥ 8 rows inside E_9 have D_r = 4, `2a+3b = 20`,
(10,0) killed, so each lies in exactly two of the three degree-7 columns; but "any fixed
pair of degree-seven columns can share at most two such rows inside E_9, since three
shared rows together with E_9 would form a K_{3,3}. The three pairs account for at most
six rows, a contradiction."

### 4.5 The marked-row leave method [source, Sec. 7]
Fix row r, delete it from every block containing it; residual blocks live on the 12
points `X \ {r}`.

**Definition 7.1**: "The leave multigraph L_r has vertex set X \ {r} and gives the pair
{x,y} multiplicity δ_{rxy}. Its total edge multiplicity is D_r and every edge
multiplicity lies in {0,1,2}."

If point x lies in `u_{x,s}` residual blocks of size s, pair capacity through {r,x} gives
```
(14)  e_x = 22 - Σ_s (s-1) u_{x,s}      (e_x = degree of x in L_r)
(15)  Σ_x u_{x,s} = s·n_s,    Σ_x e_x = 2 D_r
```
giving a finite list of point-type multisets. With M the point-by-residual-block
incidence matrix, the Gram matrix `Q = M M^T` is determined by types and leave:
```
(16)  Q_xx = Σ_s u_{x,s},   Q_xy = 2 - δ_{rxy}  (x ≠ y)
(17)  rank_{F_p}(Q) <= q        (q = number of residual blocks; used for p = 101, 103, 107)
(18)  det(Q) = det(M)^2         must be a nonnegative integer square (only when q = 12)
```
Both are *necessary* conditions used one way only ("Cases passing these screens are
retained"). Survivors get an integer-scaled Farkas certificate for the "linear
factorization system" (prescribed point incidences per size class, prescribed pair
multiplicities `2 - δ_rxy`, prescribed block counts).

**Proposition 7.2 (Certified local screen)**: the 22-line C++ program
`local_screen_general.cpp` enumerates every point-type distribution satisfying
(14)-(15), every loopless leave multigraph with multiplicities <= 2 and prescribed
degrees, and every case surviving (17)/(18); Python checks a Farkas certificate for each.

**Table 1** [source]: local enumeration counts

| Branch | Local parameters | type multisets | leaves | certified survivors |
|---|---|---:|---:|---:|
| Common D=7 | (a,b)=(5,5) | 19 | 8,641 | 0 |
| Common D=7 | (a,b)=(8,3) | 37 | 6,733 | 4 |
| 5^1 6^14 7^7, inside | (6,4) | 508 | 43,715 | 0 |
| 5^1 6^14 7^7, inside | (3,6) | 6 | 534 | 0 |
| 5^2 6^12 7^8, c=2 | (10,1),(7,3),(4,5) | – | – | 195+217+28 |
| 5^3 6^10 7^9, c=3 | (8,2),(5,4),(2,6) | – | – | 4,746+656+3 |
| 4^1 6^13 7^8, inside | (8,3),(5,5) | 41+48 | 367+275 | 12+3 |

### 4.6 Common deficit-seven branch and integral refinement [source, Sec. 8]
Four profiles force a row outside all degree-5 blocks with D_r = 7, i.e. `(a,b) ∈
{(5,5),(8,3),(11,1)}`. One Gram-surviving case is *fractionally feasible*, so an
integral refinement is needed: enumerate all 285 systems of its three residual size-6
blocks; the typed-leave automorphism group `S_2 x S_3 x S_6` (order 8,640, constructed
explicitly and checked element by element) splits them into orbits 15/180/90; two
representatives have no completion, the third has exactly 4; "sixteen exact completion
certificates close the branch" (4 completions x 4 global profiles). **Lemma 8.1** records
the conclusion.

### 4.7 The 12 x n strip [source, Sec. 2 and `proof/z12_18_21/`]
Exclude 109 ones in 12x18:
- `Z(12,17) <= 103`: exclude 104 ones; nonincreasing column-degree sequences with
  `Σ c_j = 104, Σ C(c_j,3) <= 2·C(12,3) = 440` — exactly **303** patterns, one orbit
  certificate each.
- `Z(11,18) <= 101`: exclude 102 ones; `Σ C(c_j,3) <= 2·C(11,3) = 330` — **51** patterns.
- Then in a 109-one 12x18 matrix: a column of degree <= 5 would leave >= 104 ones in
  12x17, so all columns >= 6 and the multiset is forced to `7^1 6^17`; a row of degree
  <= 7 would leave >= 102 in 11x18, so all rows >= 8. Cases: a degree-8 row exists and
  the degree-7 column contains it (**A**) or not (**B**); otherwise rows are `10^1 9^11`
  and the degree-7 column contains the degree-10 row (**I**) or not (**O**).
- Orbit certificate language: rows are partitioned into cells by membership signature
  in fixed columns; stabilizer `H = Sym(n_1) x ... x Sym(n_s)`; a k-column orbit is a
  profile `p = (p_1..p_s)`, size `N_p = Π C(n_i,p_i)`; triple orbits `u`, size `N_u`; a
  block of profile p contains `h(u,p) = Π C(p_i,u_i)` triples of profile u; every
  completion satisfies `Σ_p h(u,p) x_p <= (2 - λ_u) N_u` (their eq. (1)). In cases A/B
  the 8 columns through the deleted row are "marked": any pair of residual rows lies in
  <= 2 marked blocks.
- Exact dual leaves: for `Ax <= b, l <= x <= u` and any rational `y >= 0`,
  ```
  (2)  1^T x <= 1^T l + y^T (b - A l) + Σ_j (u_j - l_j)·max{0, 1 - (A^T y)_j}
  ```
  evaluated in integers (one common denominator). If the RHS is below the number of
  columns still required, the node is closed. Inner nodes branch `x_j <= q` vs
  `x_j >= q+1`; outer nodes fix the canonical representative of the first occupied orbit
  and forbid all earlier orbits (an H-invariant set, so "every completion is represented
  in exactly one child").

Certificate totals [source]:

| Component | Patterns | Outer nodes | Inner nodes | Dual leaves | Integer branches |
|---|---:|---:|---:|---:|---:|
| Z(12,17) <= 103 | 303 | 3,607 | 26,725 | 14,651 | 11,764 |
| Z(11,18) <= 101 | 51 | 160 | 785 | 451 | 320 |
| Case I | 1 | 1 | 1 | 1 | 0 |
| Case O | 1 | 71 | 217 | 139 | 75 |
| Case A | 1 | 77 | 6,290 | 3,077 | 3,109 |
| Case B | 1 | 31 | 2,696 | 1,342 | 1,335 |

### 4.8 The 13 x 18 frontier package [source, Sec. 3 and `proof/frontier_closure/`]
Exclude 117 ones. Dependencies: `Z(12,17) <= 103` ⇒ `Z(13,17) <= 111` (112 ones would
have a row of degree <= 8, leaving >= 104) ⇒ every column has degree >= 6 (a degree-5
column would leave 112 > 111); `Z(12,18) <= 108` ⇒ every row has degree >= 9, and
`13·9 = 117` forces the row profile `9^13` exactly. Column profiles with degrees >= 6,
sum 117, `Σ C(c_j,3) <= 572`: exactly **19**. 13 are killed by global Farkas
certificates (with an extra *integer rounding cut*, see Lemma L7 below); the 6 survivors
are handled by deficit averaging — "Every actual matrix has at least one row with
`D <= floor(ΣD/13)`" — enumerating the low-deficit row types `(a,b,...)` with
`D = 132 - Σ c·C(d-1,2)`, a point-type feasibility DP, and 12 reusable "pair-orbit"
local certificates keyed by the residual block-size multiset (e.g. `{5:3,6:5,7:1}`).

### 4.9 Deletion propagation [source, Sec. 3 and `frontier_propagation.py`]
The script closes the (m,n) grid under both one-row and one-column deletion from seeds
`{(13,17):110,(13,18):116,(14,17):118,(14,18):124,(15,17):126,(15,18):132,(16,17):133}`
and reports **24 improved upper bounds** versus the prior public table:

```
closed upper table          prior public table (PRIOR in the script)
m\n  17  18  19  20  21  22  23      17  18  19  20  21  22  23
13  110 116 122 128 134 140 144     112 118 124 130 135 140 144
14  118 124 130 136 142 148 154     120 127 133 140 145 150 155
15  126 132 139 145 152 158 165     128 135 142 149 154 160 165
16  133 140 147 154 161 168 175     136 144 151 158 164 169 175
```
(The `PRIOR` row values are the script's own snapshot of the "public frontier upper
bounds before the present closure"; I did not independently verify them against our
table.)

---

## 5. Extracted pruning lemmas, with Lean-provability notes

Conventions for the Lean notes: ZarPrune (Lean 4.34, no Mathlib) has `Mat m n := Fin m →
Fin n → Bool`, `rowSum/colSum/weight` via `sumFin`, `HasKst` as a pair of *strictly
increasing* index tuples `R : Fin s → Fin m`, `C : Fin t → Fin n` with all ones, and a
`Prune P` = computable `kill : Profile m n → Bool` on the row/col *sum vectors* plus
`sound : kill (profileOf A) = true → ¬ Valid P A`. Available lemmas: `sumFin_succ`,
`sumFin_add`, `sumFin_swap` (Fubini), `sumFin_le`, `allFin_iff`, `not_allFin_elim`.
No Finset, no binomials, no rationals, no sorting.

### L1. Kővári–Sós–Turán triple capacity (column form) — eq. (4)
- **Statement** [source]: if A is K_{3,3}-free with column sums c_j, then
  `Σ_j C(c_j,3) <= 2·C(m,3)`. General form [inferred, standard]: K_{s,t}-free ⇒
  `Σ_j C(c_j, s) <= (t-1)·C(m, s)` and dually `Σ_i C(r_i, t) <= (s-1)·C(n, t)`.
- **Hypotheses**: none beyond K_{s,t}-freeness; purely profile-computable.
- **As a prune**: `kill pf := decide (Σ_j choose (pf.col j) 3 > 2 * choose m 3)`.
  This is the ZarPrune README's "next target" and the *sole* profile filter Afrasyab
  uses before the LP (plus the penalty reparametrization L2 for enumeration speed).
- **Lean (Mathlib-free)**: medium–hard, one-time infrastructure. Plan for s = 3: (a)
  define `choose` by the Pascal recursion; (b) define the per-column triple count as a
  guarded triple-nested `sumFin` over `a < b < c` and prove it equals
  `choose (colSum A j) 3` by induction on m (peeling one row: `C(k+1,3) = C(k,3) + C(k,2)`,
  so the pair and singleton identities are needed too); (c) swap with `sumFin_swap` to
  get `Σ_{a<b<c} Σ_j [A a j ∧ A b j ∧ A c j]`; (d) show each inner sum is `<= 2` by an
  *extraction lemma*: if a 0/1 function on `Fin n` sums to `>= 3` then there is a
  strictly increasing `C : Fin 3 → Fin n` on which it is 1 (induction on n), which
  together with the increasing triple `(a,b,c)` builds `HasKst`. Estimated 300–500
  lines. Every other lemma below reuses (b) and (d).

### L2. Supporting-line penalty enumeration — eqs. (5)–(6)
- **Statement** [source]: with `p(d) = C(d,3) - αd + β >= 0` chosen so that p vanishes
  at the two degrees adjacent to e/n (13x22: α = 15, β = 70), any profile with
  `Σ d_j = e` satisfies `Σ_j p(d_j) <= (t-1)C(m,s) - αe + βn` (= 42 for 13x22@138,
  27 at 139).
- **Hypotheses**: same as L1. [inferred] It kills *no* profile that L1 does not kill —
  it is an algebraic rewrite of L1 under `Σ d_j = e`; its value is that `p >= 0` gives a
  monotone prefix bound for the enumerator (`enumerate138.py` is 19 lines) and a
  *difficulty ordering* (Sec. 10 below).
- **Lean**: trivial given L1 (`omega`-level after unfolding `choose` on the finite
  degree range, or `decide` on `p(d) >= 0` for `d <= m`). Not worth a separate prune.

### L3. Minimum-column / minimum-row deletion — Lemma 2.1
- **Statement** [source, quoted]: "For every m,n,s,t with n >= 2,
  `Z(m,n,s,t) <= floor( n/(n-1) · Z(m,n-1,s,t) )`." Proof: "If an e-edge m x n matrix
  existed, some column would have degree at most floor(e/n). Deleting it leaves at least
  e − floor(e/n) edges in an admissible m x (n−1) matrix." One-row form used for
  (14,17), (14,18), (15,17), (15,18): remaining `e - floor(e/m)` edges must not exceed
  the (m−1, n) bound.
- **Profile-level prune** [inferred, strictly stronger than the floor form]: given a
  proven bound `U'` for (m, n−1): `kill pf := ∃ j, (Σ_i pf.row i) - pf.col j > U'`
  (the profile knows the *actual* minimum column sum, not just `floor(e/n)`). Row dual
  with a bound for (m−1, n). This is exactly how 12x18@109 forces `7^1 6^17` columns and
  rows >= 8, and how 13x18@117 forces rows `9^13`.
- **Hypotheses**: an *external* neighbouring upper bound. In Lean this must be a
  hypothesis of the prune: `def colDeletion (P) (U' : Nat) (hU' : ∀ B : Mat P.m (P.n-1),
  ¬ HasKst P' B → weight B <= U') : Prune P`. Whether `hU'` is discharged by a previous
  `upper_bound_of_cover` run or imported as an axiom from a published table is a
  trust-boundary decision the harness must make explicit (Afrasyab separates "proved
  here" from "imported published bound" cell by cell).
- **Lean**: medium, ~150 lines. Needs (i) a strictly-monotone "skip index j"
  embedding `Fin (n-1) → Fin n` and the fact that composing an increasing tuple with it
  stays increasing (so `HasKst B → HasKst A`); (ii) `sumFin n f = sumFin (n-1) (f ∘ skip
  j) + f j` (not yet in `Sum.lean`; induction). No binomials needed. **Best
  cost/benefit prune in the paper.**

### L4. Row-link pair capacity / marked-row deficit — eq. (13)
- **Statement** [source]: for any row r, `D_r = 132 - Σ_{j : r ∈ E_j} C(d_j - 1, 2) >= 0`,
  i.e. (general form, [inferred]) `Σ_{j ∋ r} C(c_j - 1, 2) <= 2·C(m-1, 2)`: the columns
  through r, restricted to the other rows, cover each pair at most twice (three columns
  through r and a pair {x,y} form a K_{3,3}). Summing over r recovers 3·L1 (eq. (12)),
  so L1 is the *average* of L4 over rows; L4 can bite on individual rows.
- **Hypotheses**: K_{3,3}-free; involves *which* columns contain r, which the profile
  does not know. Profile-level relaxations [inferred]: for row i with sum `r_i`,
  `Σ_{j ∋ i} C(c_j-1,2)` is at least the sum of the `r_i` smallest values of
  `C(c_j-1,2)` over all columns; kill if that exceeds `2·C(m-1,2)`. Cheaper but weaker:
  `r_i · C(c_min - 1, 2) <= 2·C(m-1,2)`.
- **Congruence refinement** [source, not a prune]: since `C(5,2), C(6,2) ≡ 0 (mod 5)`,
  profiles built from degrees 6/7 force `D_r ≡ 2 (mod 5)`; with `Σ D_r = 3s` small,
  some row is residue-minimal. This is a *case split inside a profile* (which row, which
  incidence counts (a,b)), i.e. a cube, not a kill. Good template for an LLM-produced
  "sub-case generator" but it sits on the `cover` side of ZarPrune, not the `kill` side.
- **Lean**: the cheap version is easy once L1's machinery exists (it is L1's step (d)
  applied to pairs among the other rows, with the marked row fixed); the "r_i smallest"
  version needs a sorting/selection lemma (Mathlib-free: painful; suggest a
  `List.mergeSort`-free formulation: kill if for *every* subset... no — use the min-form
  first).

### L5. Pair-through-two-rows capacity — eqs. (9), (14)
- **Statement** [source]: for rows r ≠ x, `Σ_{j ⊇ {r,x}} (c_j - 2) <= 2·(m - 2)` (each
  such column covers `c_j − 2` triples `{r,x,z}`, each triple <= 2, `m−2` triples).
  Row (8) is `Σ_{B ∋ r} C(|B|-1,2) x_B <= 132`, the LP form of L4.
- **Profile-level**: not computable from sums alone (needs codegrees `c_{rx}`). Used
  only inside the LP and the local screens. [inferred] Becomes a prune only if the case
  decomposition is refined to carry codegree information (Tan's decomposition does not).
- **Lean**: same machinery as L4; not a priority.

### L6. Block-polytope LP + exact Farkas certificate — Lemma 5.1
- **Statement** [source]: quoted in 4.3. Soundness = weak duality; the certificate is a
  rational vector checked over every candidate block (390,039 coefficients for the
  138 reduction; 5,262,392 for the local proofs; 5,778,414 total incl. the 139 regression).
- **Hypotheses**: a fixed column-degree profile, one block fixed by row symmetry (a
  *symmetry-breaking* step: "by row symmetry, fix one f-block as F = {1,...,f}" — sound
  only because the LP is stated for a canonical relabelling, cf. ZarPrune's warning that
  sorting is an *adding* move; here it is harmless because the LP is quantified over
  the relabelled matrix, but a Lean proof must construct the relabelling).
- **As a prune**: `kill pf := checkFarkas pf (cert pf)` is computable, but the
  soundness proof needs: (i) the map from a matrix to its block-multiplicity vector
  `x_B` over all `2^m` supports; (ii) constraints (7)-(9) from HasKst-freeness; (iii)
  the duality inequality over rationals (or integers after scaling — the local
  certificates are already integer-scaled; global ones use `Fraction`).
- **Lean (Mathlib-free)**: hard. (i) alone wants a Finset/subset-enumeration layer;
  (iii) wants `Rat` or an integer-scaled restatement plus `Int` linear algebra over
  ~2^13-length vectors; the kernel would have to *evaluate* a 10^5–10^6-term check
  without `native_decide`. Realistic only with Mathlib (`Finset.sum`, `Rat`) and a
  reflection-style checker, or by trusting an external replayer — which is exactly the
  trust model Afrasyab uses and the thesis wants to move beyond. Recommendation: use LP
  infeasibility as a *proxy reward / difficulty oracle* (Sec. 10) rather than as a Lean
  prune in the first iteration.

### L7. Integer rounding cut (13x18 global certificates) [source: `verify_all.py`]
- **Statement**: for a pair P with pair capacity `cap_P = 22 - 1[P ⊆ F](f-2)` and a
  degree threshold θ: `Σ_{B ⊇ P, |B| >= θ} x_B <= floor(cap_P / (θ - 2))` — "each
  selected degree >= threshold block containing this pair consumes at least
  threshold−2 capacity." A Chvátal–Gomory-style strengthening of (9).
- **Lean**: an integer-division consequence of L5; only matters inside L6.

### L8. Gram rank and determinant screens — eqs. (16)–(18)
- **Statement** [source]: `Q = M M^T` with `Q_xx = Σ_s u_{x,s}`, `Q_xy = 2 - δ_rxy`;
  necessary conditions `rank_{F_p} Q <= q` and, when q = 12, `det Q` is a perfect square.
- **Use**: filters local cases before Farkas; never used to *accept*. Not a
  profile-level prune (needs the leave). [inferred] A Lean version would need matrix
  rank over F_p — out of scope for ZarPrune; mention only as "necessary-condition
  screens are safe to use as filters *if* survivors are still fully certified".

### L9. Orbit dual-leaf bound — 12-row package eq. (2)
- **Statement** [source]: for `Ax <= b, l <= x <= u`, `y >= 0`:
  `1^T x <= 1^T l + y^T(b - A l) + Σ_j (u_j - l_j)·max{0, 1 - (A^T y)_j}`.
- **Lean**: the inequality itself is elementary integer arithmetic (medium, needs sums
  over a variable index set). The *exhaustiveness* of the outer orbit branching is a
  symmetry argument ("the forbidden union is H-invariant; hence every completion lies in
  exactly one child") proved only in prose — this is precisely the "adding" move the
  ZarPrune README says must live in an SR/VeriPB-style certificate, not in `Prune`.
  Afrasyab's verifier *replays* the tree but trusts the induction argument.

### L10. Monotonicity: exclude exactly w ones [source, Sec. 10]
- **Statement**: "Deleting ones shows that no larger K_{3,3}-free matrix exists either."
- **Lean**: easy (flip one `true` to `false`, HasKst is monotone, weight drops by 1;
  induction). Needed for our `cover` obligation whenever the enumerator lists profiles
  with `Σ r_i = w` exactly rather than `>= w`. Not yet in ZarPrune.

### L11. Two-row deletion (cautionary) — Remark 3.1
- **Statement** [source]: deleting rows i, j leaves `e - d_i - d_j + c_ij` edges. The
  author's *earlier* certificates built on this claimed `Z(15,17) = 125` and
  `Z(16,17) = 130`; explicit 126- and 132-edge witnesses later **refuted** those upper
  bounds and the certificates were withdrawn ("no independently replayed certificate
  for the associated upper-bound claim is included"). [inferred] An unsound
  upper-bound argument survived internal exact replay and was caught only by a lower-bound
  construction — the strongest possible argument for the thesis's requirement that every
  prune be Lean-verified, and for running lower-bound search alongside upper-bound
  pruning as a sanity check.

---

## 6. Numbers worth remembering

| Quantity | Value |
|---|---|
| 13x22 @138: profiles / Farkas / hand / leave-method | 83 / 77 / 1 (`6^18 7^3 9^1`) / 5 |
| 13x22 @139 (regression only): profiles / Farkas / hand | 27 / 26 / 1 (`6^15 7^7`) |
| Penalty budget @138, @139 | 42, 27 |
| Exact coefficient checks (139 / 138 global / local) | 125,983 / 390,039 / 5,262,392 = 5,778,414 |
| Local factorization systems with certificates in `5^3 6^10 7^9`, c=3 | 5,405 |
| 13x18 @117: column profiles / global Farkas / marked-row profiles / local certs | 19 / 13 / 6 / 12 |
| 12x17 @104 nonincreasing profiles; 11x18 @102 | 303; 51 |
| Full 13x22 gate wall time (author's machine) | ~32 s; ours: enumerators run in < 1 s |
| Witness histograms (mult. 0/1/2 over 220 row triples) 12x18..22 | (6,68,146), (3,54,163), (1,38,181), (0,20,200), (0,0,220) |
| 13x22 137-witness | 17 columns of degree 6, 5 of degree 7 (App. B) |

Survivors of the global LP at 138 edges, with triple slack `s = 572 − Σ C(d_j,3)` and
penalty (from running `enumerate138.py`; indices are penalty rank among the 83):

| idx | profile | slack s | penalty |
|---:|---|---:|---:|
| 0 | 6^16 7^6 | 42 | 0 |
| 1 | 5^1 6^14 7^7 | 37 | 5 |
| 3 | 5^2 6^12 7^8 | 32 | 10 |
| 6 | 4^1 6^13 7^8 | 28 | 14 |
| 7 | 5^3 6^10 7^9 | 27 | 15 |
| 11 | 6^18 7^3 9^1 | 23 | 19 |

All six LP survivors are among the 12 lowest-penalty profiles; the 71 profiles with
penalty >= 20 were all Farkas-separated. At 139 edges the *only* survivor was the
zero-penalty profile. At 13x18@117 the survivors were indices {0,1,2,4,6,7} of 19.

---

## 7. Verification approach and trust boundary [source, Sec. 11]

- One command: `cd proof && python3 verify_all.py` (compiles the C++ enumerator,
  runs `verify_137.py`, `verify_139.py`, `verify_138_reduction.py`, `verify_138_full.py`,
  then the 12-row gate and mutation tests); frontier package via `verify_release.sh`.
  CI workflow `.github/workflows/verify.yml` reruns it on push.
- Trusted: "exhaustive finite integer enumeration in the supplied C++ source; Python
  integer and Fraction arithmetic; explicit Farkas vectors stored in JSON; the explicit
  lower-bound witness." Not trusted: "No floating-point solver status, MILP infeasibility
  flag, tolerance, or randomized search result is a premise." Discovery scripts
  (`proof/discovery/`, SciPy/HiGHS) are outside the boundary.
- Adversarial audit (`ADVERSARIAL_AUDIT_FULL.md`) checks: omitted degree profiles,
  hidden simplicity assumption (repeated columns allowed), omitted leave graphs, unsafe
  modular-rank / determinant inference, trusting LP status, symmetry dropping cases,
  assuming unique completion, floating-point rounding, witness transcription.
  Mutation tests corrupt a witness, a rational leaf, and bridge metadata and confirm
  rejection.
- Still desirable per the author: "implement an independent local enumerator in a
  second language; replay the theorem in Lean, Isabelle, or Coq; obtain an external
  combinatorics expert's audit of the marked-row reduction."
- "No SAT timeout, unsuccessful construction search, or unverified DRAT claim is used."
  Tan [7] is cited only as related work.

---

## 8. Relation to other sources in our reading list

- **Bhan et al. (our paper)**: supplies the 137-edge 13x22 witness and the (12,22)
  value; Afrasyab's novelty audit uses our frontier table as "positive evidence that
  these cells were still open shortly before this work". Our (13,18) lower bound 115 is
  superseded by 116.
- **Collins–Riasanovsky–Wallace–Radziszowski**: source of the imported bounds
  `Z(13,17) <= 110` and `Z(16,17) <= 133`.
- **Davies–Gill–Horsley**: cited for "strengthened relaxations"; Afrasyab's LP is in the
  same Roman/LP lineage but per-profile and with exact dual replay.
- **Tan (SAT attack)**: cited but methodologically orthogonal — Afrasyab's profile
  enumeration (`Σ c_j = e, Σ C(c_j,3) <= 2C(m,3)`) is *identical* to the profile
  enumeration in our `sat_attack/profiles.py`; the difference is what happens per
  profile (LP/Farkas + local enumeration vs. SAT).
- **dfield/finite-zarankiewicz-closures**: referenced in the novelty audits as the
  public frontier table; the Lean-checked-pruning idea there is the closest to our design.

---

## 9. Design implications for the thesis pipeline

1. **Known-answer benchmarks.** (12,17)=103, (12,18)=108, (13,18)=116, (13,22)=137 are
   now settled with per-profile difficulty data. Run our pipeline on `z(12,18;3,3) <= 108`
   (exclude 109) first: Afrasyab shows the profile space collapses to `7^1 6^17` given
   the neighbour bounds, so an evolved L3 prune plus KST should leave a handful of SAT
   cases — a clean end-to-end test of `upper_bound_of_cover`.
2. **Prioritise L1 (KST) then L3 (deletion) in Lean.** KST is the one infrastructure
   investment (subset/triple counting over `Fin`); deletion needs only an index-skip
   embedding and is the highest-leverage prune once neighbour bounds are trusted. Together
   they reproduce every *profile-level* elimination in the paper; everything else is
   per-profile certificate work.
3. **Make neighbour bounds first-class, with provenance.** The deletion prune needs
   `hU'` as a hypothesis. The harness should keep a ledger of bounds tagged "proved by
   this pipeline" vs "imported (Collins et al. / Afrasyab / our paper)" exactly as
   Afrasyab does, and never let an imported *exact claim* of an unreviewed preprint enter
   as an axiom without noting it (L11 shows why).
4. **Profile enumeration must allow repeated columns and full degree ranges.** Afrasyab's
   audit item 1–2. Our `profiles.py` already enumerates non-increasing sequences with
   `w ∈ [2, m]`; the lower cutoff 2 is a *symmetry/deletion* assumption that needs its
   own justification in `cover`.
5. **Separate kill from cover.** The congruence/deficit-averaging arguments are cubes
   (sub-case splits) with a completeness proof, not kills. If the evolutionary search
   proposes such splits, they belong in the `cover` certificate, and ZarPrune currently
   has no vocabulary for them — an open extension (see Sec. 11).
6. **LP as oracle, not as proof (for now).** A per-profile LP (7)–(10) solved with an
   exact rational or even floating solver gives a cheap, strong signal of whether a
   profile is "counting-easy" (LP infeasible) or "SAT-hard" (LP feasible). Use it in the
   reward function to estimate the difficulty of what a candidate prune kills, and to
   triage which survivors go to SAT. Lean-verifying Farkas certificates is a possible
   later milestone (would answer the author's own "replay in Lean" wish).
7. **Monotonicity lemma L10** should be added to ZarPrune so that enumerating
   exact-weight profiles suffices for `cover`.
8. **Run lower-bound search in parallel** (our SuperUROP machinery) as a soundness
   canary: a witness with more ones than a claimed bound is an immediate red flag (L11).
9. **Hand-proof templates for the LLM.** Proposition 6.1 and the 139-edge `6^15 7^7`
   lemma (`verify_balanced_lemma` in `verify_139.py`) are the kind of short
   natural-language counting proofs an LLM could be asked to produce and then formalise;
   both reduce to L4 + pigeonhole + a small case analysis.

---

## 10. Difficulty signals for a case (for the reward function)

Derived from what survived each stage in this paper [source data, inferred usage]:

- **Triple slack / penalty**: `s = (t-1)C(m,s) - Σ_j C(c_j,3)` (equivalently
  `Σ p(d_j)`). Near-balanced profiles (small penalty, degrees ≈ e/n) are the hard ones:
  all LP survivors at 138 have penalty <= 19, and the unique survivor at 139 has
  penalty 0. Large slack ⇒ cheap to kill by counting ⇒ low reward for pruning it.
- **LP relaxation status and margin**: infeasible ⇒ "easy" (a counting/LP prune exists);
  feasible ⇒ needs integrality/structure ⇒ SAT-hard. The discovery script
  `gen_case_farkas_margin.py` and the stored `objective` field give a continuous margin.
- **Number and size of exceptional degrees**: profiles with degree-4/5 columns needed
  far more local cases (5,405 for `5^3 6^10 7^9`, 195+217+28 for `5^2 6^12 7^8`) than
  the pure 6/7 profile `6^16 7^6` (0 surviving cases).
- **Deficit-averaging threshold** `floor(3s/m)`: smaller ⇒ fewer low-deficit row types
  ⇒ smaller case tree.
- **Local screen survivor counts** after the Gram rank/determinant filters (Table 1
  column "certified survivors").
- **Orbit-certificate tree size** (outer/inner nodes, dual leaves): Case A needed 6,290
  inner nodes vs 1 for Case I — a direct proxy for SAT hardness of the same case.
- **Whether the profile is fractionally feasible** (needs integral refinement) — the
  single hardest local case in the paper.

---

## 11. Open questions

1. Can the profile-level restriction of L4 ("sum of the r_i smallest C(c_j−1,2)") ever
   kill a profile that L1 + L3 do not, on our target cells? Cheap to test numerically
   before investing in the Lean sorting lemma.
2. How should ZarPrune represent *cubes* (sub-case splits such as "some row has
   D_r <= floor(3s/m)" with residue classes) so that evolved case generators can be
   verified for completeness, not just prunes for soundness?
3. Is there a Mathlib-free path to certifying Farkas vectors — e.g. integer-scaled
   certificates checked by a reflective Lean checker with `decide` on a compressed
   representation — or is Mathlib unavoidable for L6?
4. The prior public table used in `frontier_propagation.py` disagrees with Collins et
   al. at (13,17) (112 vs 110); which table does our pipeline treat as ground truth?
5. Which neighbouring open cells become tractable given the new seeds (e.g. (13,19)
   <= 122, (14,19) <= 130, (16,18) <= 140 by deletion)? A SAT + prune attack on the
   deletion-derived bounds may close some with the 12/13-row profile space forced small.
6. The author's method is "potentially reusable" for nearby K_{3,3} cells (Sec. 12).
   Does the supporting-line + LP + marked-row recipe extend to K_{2,2}/K_{2,3}
   (our generalized-Zarankiewicz work) or to (s,t) = (3,4)?
7. Independent reproduction: nobody outside the author has replayed the package; the
   thesis could contribute a partial Lean replay (e.g. of L3 chains and of the 12x18
   case split) as a verification milestone.

---

## 12. Files consulted

- `paper/main.tex` (676 lines), `paper/main.pdf` (12 pp.)
- `proof/PROOF_NOTE.md`, `proof/README.md`, `proof/ADVERSARIAL_AUDIT_FULL.md`,
  `proof/STATUS.json`, `VERIFICATION_TRANSCRIPT.txt`, `REPRODUCIBILITY.md`
- `proof/enumerate138.py`, `proof/enumerate139.py`, `proof/verify_138_reduction.py`,
  `proof/verify_139.py` (incl. `verify_balanced_lemma`), `proof/verify_all.py`,
  `proof/local_screen_general.cpp`, `proof/certs138/p02.json` (format)
- `proof/z12_18_21/{PROOF_NOTE.md, main.tex, README.md, CLAIM_BOUNDARY.md, NOVELTY_AUDIT.md}`
- `proof/frontier_closure/{PROOF_NOTE.md, README.md, STATUS.md, verify_all.py, frontier_propagation.py, reports/frontier_propagation.json}`

# Generalization notes — construction families for z(m, n; s, t)

Status: 2026-07-28, construction engineer.  Every claim below labelled per
the project honesty bar (README).  "Verified" always means: passed
`zarankiewicz.verify_kst_free` AND the snapshot reference verifier.

Convention: m rows, n columns; forbidden: s rows sharing t common 1-columns.
Self-duality: z(m,n;s,t) = z(n,m;t,s) — the engine always tries both
orientations (swapping s,t together with m,n).

## The unifying frame

A column IS a block (its set of 1-rows).  A matrix is K_{s,t}-free iff the
block multiset is a **(t-1)-fold s-packing**: every s-subset of rows lies in
at most t-1 blocks.  [PROVEN — restatement of the definition.]

Consequences used everywhere:
- weight-(s-1) columns are free (never in any s-subset) — universal pads;
- weight-(t-1) rows are free — universal row pads;
- the counting bound  Σ_j C(c_j, s) ≤ (t-1)·C(m, s)  [PROVEN; for (3,3)
  this is Roman's bound; the two-sided integer version is the coordinator's
  waterfill WF(m,n)];
- every family below is "structured seed blocks + canonical completion
  under an explicit capacity dict", so freeness holds by construction and
  is still re-verified.

## Families

### 1. trivial (m < s or n < t)
(i) PROVEN exact everywhere it applies: z = m·n (all-ones).
(iii) inapplicable otherwise.

### 2. culik
z(m,n;s,t) = (s-1)n + (t-1)C(m,s) for n ≥ (t-1)C(m,s)  [PROVEN — Culik
1956; verified on all 41 in-table (3,3) cells in its regime].  Construction:
every s-subset as a column t-1 times + weight-(s-1) pads.  Transposed
version for m ≥ (s-1)C(n,t).  Truncated best-effort (s·n) below threshold —
strictly dominated by roman_window/greedy in practice.

### 3. roman_window  (s = t = 3)
z(m,n;3,3) = 3n + min(T33(m), n, ⌊(B−n)/3⌋), B = 2C(m,3), for n above a
threshold n0(m)  [PROVEN for the pure Roman window n ≥ B − 3·T33(m), cited
Roman 1975 / Tan; T33 = published max 2-fold triple-packing-by-quadruple
numbers.  CONJECTURE-STATUS for the theory agent's supply-capped extension
below the pure window — but every in-table cell it predicts is verified
attained by the built matrix].  Covers rows m ≤ 6 entirely and m = 7 for
n ≥ 14 in-table.  Generalization to (s,t): k blocks of weight s+1 from a
(t-1)-fold packing + repeated s-blocks + pads; z = s·n + k in the analogous
window — the packing-number table for (s+1)-blocks is the only missing
ingredient beyond (3,3).

### 4. sum_layer / xor_layer  (any s, t)
(s+1)-blocks grouped so each s-subset determines the remaining element:
sum-residue layers over Z_m, or XOR layers over F_2 labels.  Each layer is a
partial Steiner system; t-1 layers give a legal packing.  [PROVEN sound;
used as fill orders more than as standalone seeds.]  For m ≤ 8, s = 3 the
XOR-0 layer is exactly the 14 planes of AG(3,2) = the unique SQS(8).
(ii) conjecturally strong for elongated cells of any (s,t) with m small.

### 5. hyperplane_f2k  (champion-derived; s = 3, any t with 2^{k-3} ≤ t-1)
Rows = distinct nonzero points of F_2^k, column a = {x : <a,x> = 1}.
Dependent triples are inconsistent (0 columns), independent triples get
exactly 2^{k-3} columns.  [PROVEN sound; k = 4 for t = 3.]
(ii) strongest near-square seed for 9 ≤ m ≤ 15 at (3,3).
Codes bridge (general s): rows = columns of a parity-check matrix with no
small odd-weight dependencies; coverage 2^{k-rank}.  Not yet implemented
beyond s = 3.

### 6. cap_bothsides  (champion-derived; s = 3, t = 3, m ≤ 16)
Rows = all of F_2^4, columns = both sides {x: <a,x> = b} for normals a in a
cap of PG(3,2) (canonical max cap = affine complement, 8 normals).  The
normals covering a triple form a projective line; cap∩line ≤ 2.  [PROVEN
sound; attains z(16,16) = 128 — isomorphic to Tan's published witness, per
theory agent.]  Generalization lever: k > 4 needs "sets meeting every
(k-3)-flat in ≤ t-1 points" — for k = 5 no such set of size ≥ 3 exists, so
this family is intrinsically a k ≤ 4 phenomenon at t = 3.  For larger t:
caps → sets meeting lines in ≤ t-1 points (arcs), giving m = 2^k instances.

### 7. line_complement  (s = 3; q with r·(q-1)² ≤ t-1)
Blocks = complements of PG(2,q) lines, each r times, on m = q²+q+1 points.
Coverage: non-collinear triples avoid (q-1)² lines, collinear q(q-2).
[PROVEN sound.]  For (3,3): only q = 2 (doubled Fano, m = 7) — strong there.
For t ≥ 5: q = 3 (m = 13) enters, r = ⌊(t-1)/4⌋.

### 8. hadamard_3design  (s = 3; m = 4k, m-1 prime ≡ 3 mod 4, k-1 ≤ t-1)
Points Z_{m-1} ∪ {∞}; blocks (QR+i) ∪ {∞} and their complements — the
Hadamard 3-(4k, 2k, k-1) design, repeated r = ⌊(t-1)/(k-1)⌋ times.
[PROVEN sound — 3-design property; verified.]  For (3,3):
- m = 8: 3-(8,4,1) = SQS(8) doubled (28 blocks, capacity-saturating);
- m = 12: 3-(12,6,2), 22 blocks, EXACTLY saturating — this is the extremal
  structure of z(12,22) = 132 (engine attains it; CERTIFIED-EXACT against
  the table).
Residual versions (delete ≤ 3 points) stay legal and were decisive for
m = 9, 10, 11 elongated cells.  m = 20, 24 apply once t ≥ 5, 6.

### 9. difference families / orbit_scan  (any s, t; cyclic)
Columns = shift orbits of base sets B ⊂ Z_m.  Catalogue: greedy λ ≤ t-1
sets (pooled difference multiplicities ⇒ K_{2,t}-free, hence K_{s,t}-free),
quadratic residues (m prime), QR ∪ {0}, Singer planar difference sets
(m = q²+q+1, q prime — every nonzero difference exactly once ⇒ the cyclic
projective plane, optimal (2,2)), triangular prefixes (the evolved run's
seed), orbit-capacity greedy (grows B under full orbit legality), and an
exhaustive scan over rotation-classes of small base sets (bounded, derived).
[Soundness by verification; the λ ≤ t-1 and Singer properties PROVEN.]
(ii) strong for prime-ish m near-square at (3,3); the (11,21) = 116 cell
falls to QR(11)-based orbits.

### 10. bipolar  (distilled from discovered z(8,17..22) extremal structures)
Two pole rows P, equator E = rest; blocks: E (or E minus one point) and
P ∪ T_i for a maximal partial triple packing {T_i} on E (pairwise ≤ 1,
degrees ≤ t-1), then canonical completion.  [PROVEN sound via the capacity
frame; z(8,17) = 74 ... z(8,22) = 90 all attained — CERTIFIED-EXACT vs
table.]  (ii) conjecturally the right shape for m ≡ 0 mod 4 just below the
Roman window; untested beyond m = 8.

### 11. twin / twin_planes  (distilled from discovered z(8,22..23))
Two (s+2)-blocks meeting in ≤ s-1 points; for m = 8 the sharp variant:
plane-extensions P ∪ {x}, P^c ∪ {y} of a complementary AG(3,2) plane pair
plus all other planes (the twin bases omitted — their triples are saturated
by the twins).  Attains z(8,23) = 94.  [Sound by capacity frame; verified.]
Generalization: complementary flat pairs in AG(k,2) — untested for m = 16.

### 12. pair_gdd  (distilled from discovered z(11,16) = 92)
Partition rows into g pairs (+ singleton); blocks = "blown" quotient
triples (unions of 3 pairs, quotient triples a maximum (t-1)-fold triangle
packing of K_g — computed by the same exact packer one level down) +
transversals (one element per pair + singleton) selected by the exact
packer; the discovered transversal codes are unions of cosets of 2-dim
binary linear codes with nonzero projection on every 3 coordinates.
[Sound by capacity frame; z(11,16) = 92 attained.]  (ii) strong for
near-square 10 ≤ m ≤ 12; the group-divisible analogue of family 8.

### 12b. sts9_dressing  (decoded from the ILP witness of z(9,22;3,3) = 100)
Take STS(9) = AG(2,3) and a point p: for each of the 8 lines L avoiding p,
BOTH the quad {p} u L and its complement (a pentad); the 4 lines through p
give a perfect matching e_i on the other 8 points, split 2+2: pentads
{p} u e_a u e_b across the split, quads e_a u e_b within.  22 blocks, 100
edges, every triple <= 2 (76 saturated).  [Sound by verification —
CERTIFIED-EXACT vs table; all three matching splits work.]  NOT a
truncation of the Hadamard 3-(12,6,2) (that caps at 99 on this cell).
Generalization lever: the same complement-pair dressing of a Steiner
system S(2,3,v) with a base point needs |complement| small, i.e. v = 9 is
the balanced case; the OPERATOR content (orbit + complement + point-extend
+ matching-fusion) is what generalizes — it is exactly the operator
alphabet of `unified.py`.

### 13. projective_plane_22  (s = t = 2)
PG(2,q) incidence, q ∈ {2,3,4,5,7,8,9} via GF(p^k) tables; truncation by
greedy lowest-degree deletion.  [PROVEN optimal at m = n = q²+q+1:
z = (q+1)(q²+q+1); verified for q = 2,3,4,5 → 21/52/105/186, all
violation-free.]  (iii) inapplicable for (s,t) ≠ (2,2) (but its Singer
form reappears inside family 9).

### 14. norm_graph  (Kollár–Rónyai–Szabó / Alon–Rónyai–Szabó)
Bipartite projective norm graph on F_{q^{s-1}} × F_q^*: K_{s,(s-1)!+1}-free.
[PROVEN (cited); verified K_{3,3}-free at q = 2, 3 (18×18, 144 edges).]
(iii) not competitive at table sizes (asymptotic construction); included
for completeness and for t ≥ (s-1)!+1 regimes at large scale.

### 15. compositions  [all PROVEN sound, all verified]
- pad_extend: new cols weight s-1 (old rows), new rows weight t-1 ⇒
  z(m,n) ≥ z(m',n') + (s-1)(n-n') + (t-1)(m-m').
- best-extension: add the heaviest legal new column/row under residual
  capacity (subsumes safe duplication — an existing column is duplicable
  iff it fits its own residual).
- shrink: delete the lightest row/column of a bigger cell's matrix.
- side_by_side: K_{s,t1}-free | K_{s,t2}-free ⇒ K_{s,t1+t2-1}-free
  (pigeonhole).  Verified: two K_{3,3}-free 9×12 give a K_{3,5}-free 9×24.
- stack: transposed analogue (K_{s1+s2-1,t}).
These power the table DP (extend/shrink sweeps); side_by_side/stack are the
route from t ∈ {2,3} building blocks to arbitrary t.

### 16. greedy_lex_derived  (any s, t — the floor)
Waterfill-guided capacity greedy: derive the level column-degree profile
from the counting bound, build each column row-by-row under capacity with
(degree, index) or capacity-damage orders.  Deterministic, bounded, always
legal.  (ii) surprisingly strong on counting-tight cells (d = 0), weak in
the deficit band.

### 17. bounded exact packers  (the completion layer)
_exact_pack / _exact_pack_multi / _exact_pack_edges: node-capped lex DFS
with fungible-capacity bounds, block multiplicity up to t-1, canonical
candidate orders (lex / XOR-plane-first / complement-paired / coset-sorted).
Used (a) to finish seeds ("fill=ye"), (b) one level down on quotient
structures (pair_gdd), (c) as a last-resort polish.  Deterministic and
bounded — but this is the one layer that is search-shaped; every cell that
NEEDS it is flagged in the report.

## Where the engine stands, by regime  (s = t = 3, the 161-cell table)

- Culik regime (41 cells): PROVEN exact, formulaic.
- Roman window rows m ≤ 6 + m = 7, n ≥ 14 (71 cells incl. overlap):
  PROVEN/cited exact, formulaic.
- Counting-tight cells (waterfill d = 0, 78 cells): all exact, mostly via
  hyperplane/difference/greedy realizers of the level profile.
- Deficit band m, n ∈ [8,16]: families 5, 6, 8, 9, 10, 11, 12 + DP close
  all but two cells.
- Open in-table: z(10,15) = 81 only — the engine reaches 80; the LNS
  search, the evolutionary champion, and all of the bounds agent's ILP
  attempts (timed out) also failed.  This is the one honest gap.
  (z(9,22) = 100 was closed by decoding the coordinator's ILP witness into
  the sts9_dressing family — family 12b.)

## The unified construction (`unified.py`, owner mandate)

`construct_unified(m, n, s, t) = realize(profile(m,n,s,t), tower(m,s,t))` —
the router's families re-expressed as ONE orbit-closure rule:

- **tower(m,s,t)**: label ladder m..m+3 plus the next power of two (two
  canonical embeddings: identity-prefix and Hamming-weight order); on each
  label count, every applicable generator — EA (elementary abelian F_p^k:
  cap-first hyperplane side-sets, AG lines, xor-flats), PTD (pointed
  Frobenius: QR u {oo} orbit), PRG (pair group: blown quotient triangles +
  coset-ordered transversals), CYC (Z_v: QR orbits + orbit-capacity-greedy
  bases) — closed under the operator alphabet
  {orbit, complement, base-point extension, through-point fusion,
  doubling}, restricted to [m].
- **profile**: the two-sided waterfill — supplies the size cap and the
  heavy-block quota center.
- **realize**: capacity walk of one canonical linearization of the tower
  (6 canonical linearizations: 4 leader-major + 2 size-major), quota/cap/
  doubling/phase-2 swept over a closed window, finished by the shared
  canonical completion.  Every output verified.

Instances the ONE rule reproduces exactly (leader in parentheses):
Hadamard 3-(12,6,2) at (12,22) (PTD), the m=16 cap matrix (EA), the
13x16 hyperplane matrix ('gen'), z(11,21) via QR0(11) orbits (CYC),
z(11,16) via the GDD coset code (PRG), z(8,17) (PRG), the m=3..7 Roman
rows, and the Culik regime.  The measured PRICE OF UNIFICATION vs the
router is in `price_of_unification.csv` and reported in `report.md` — that
number, not a claim of equivalence, is the honest headline.

## Skeleton of the unified algorithm

1. Orientation: build with the smaller of C(m,s), C(n,t) as the capacity
   side; try both.
2. If m < s or n < t: all-ones.  If n beyond the Culik threshold: Culik.
   If inside the (generalized) Roman window: packing + s-blocks + pads.
3. Otherwise derive the waterfill profile and try the algebraic seed
   families applicable to (m, s, t) — F_2^k hyperplanes and caps, Hadamard
   3-designs and residuals, PG(2,q) line complements, cyclic difference
   orbits, bipolar/twin/pair-GDD shapes — each through the canonical
   capacity completion with its bounded exact finisher.
4. Fold in cross-size DP: pad/extend from (m-1,n), (m,n-1); shrink from
   (m+1,n), (m,n+1).
5. Verify everything; return the best verified matrix.

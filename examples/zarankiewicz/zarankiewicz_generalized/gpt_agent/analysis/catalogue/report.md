# The finite catalogue: structure theory of K3,3-free extremal matrices

Structure theorist, catalogue session, 2026-07-30. Labels per
BASE/README.md (PROVEN / VERIFIED / CONJECTURE / OPEN / REFUTED).
All machine claims reproducible from this directory (scripts in §8).
Nothing outside BASE was modified; no per-cell z-optimization was
run — only analysis of existing witnesses plus five small
abstract-support MILP infeasibility certificates (`milp_cases.py`,
HiGHS via scipy; each an m-independent finite classification, not a
cell optimization).

Notation: B = 2·C(m,3); T = T33(m) = D2(m,4,3); J = Johnson bound;
column = block; heavy = weight >= 4; fills = weight-3 columns;
pads = weight <= 2; val(C) = sum over heavy of (w-3); leave
y(T) = 2 - coverage. Families: ADMISSIBLE m (3 | (m-1)(m-2), C(m,3)
even; T = B/4); CLASS m (m == 3 mod 4, 3 does not divide m;
T = J-2 by Theorem F); 3|m (T = J by Theorem 11).

---

## 1. Mission summary (honest headline)

1. The generator algebra is defined in `algebra.md`: 13 constructive
   (Tier-1) species + 3 predicate (Tier-2) species + 7 composition
   moves.
2. Completeness audit over the two mandated witness banks
   (analysis/witnesses: 64 files; design_prover/witnesses: 5
   distinct) plus the sat_attack bank (12 files), 81 verified files
   total, every one independently re-verified legal (81/81):
   headline numbers in §4.1.
3. **Instance-finiteness is REFUTED** (three non-isomorphic maximum
   packings at m = 9 inside one bank; the band-11 packings are not
   isomorphic to the design prover's). The conjecture survives at
   SPECIES level (§2) — that is the honest formulation, and it
   matches the graph-minors shape.
4. **The wide region n >= T is now a structure THEOREM** (§3):
   every optimum is a maximum quad packing plus fills in its leave
   (T-branch) or a near-saturated quad+fill 2-cover (Roman branch) —
   the strong "every extremal matrix decomposes" direction, not just
   realization. Machine-verified on all 22 in-scope witnesses
   (22/22 exact matches, `verify_S.py`).
5. Collateral new results: **S7(1) = 12 and S9(1) = 37 PROVEN**
   (both were open units in deepband report §4); the previously
   undecoded (10,15) optimum is now fully classical; a stale
   frontier entry F9(27) corrected upward (§5).
6. WQO analysis (§6): precise order, an ambient-class antichain, why
   the finite-obstruction-set route does not transfer, and the one
   positive reduction (Higman on the fills side).

---

## 2. The FINITE CATALOGUE conjecture (formal)

**Definitions.** A *species* is a family S = {S_m} of block
multisets with a polynomial-time membership certificate (a labeled
isomorphism onto an explicit construction, or a verifiable
structural predicate). A *decomposition* of an m x n matrix M over
a species list 𝕊 and move list 𝕄 is a derivation of M's column
multiset from instances of 𝕊 by moves of 𝕄 (algebra.md §3–4),
certified layer by layer; legality (joint coverage <= 2) is part of
the certificate.

**Conjecture FC (species form).** There is a FINITE list 𝕊 of
species and the seven moves 𝕄 of algebra.md §3 such that for every
(m, n), EVERY extremal K3,3-free m x n matrix decomposes over
(𝕊, 𝕄).

**Conjecture FC-weak.** For every (m, n), SOME extremal matrix
decomposes over (𝕊, 𝕄).

**Status.**
- FC-weak: TRUE on every cell of the workspace's proven range — the
  realization sides of Theorems 1, 2, 5–8 and the level formula are
  catalogue constructions by inspection (point-complements,
  sub-packings + fills, EC(triangle-free graph), hyperplane/cap
  families).
- FC (strong): PROVEN on the wide region n >= T for the three
  families with known T (§3) — there the decomposition shape is
  FORCED, which is precisely the closure shape the mission asked
  for. On the deep band FC is supported by the audit (§4) at
  species level; the quad/pentad BODIES of mid-band frontier
  configs are the honest open decode targets.
- Instance-level finiteness (a finite set of generator matrices):
  REFUTED — §4.2. The species count is 16 and grew this session
  only by absorbing two classical objects (Moebius–Kantor, Moebius
  residual orbits) — sporadics absorbed as members, not
  formula-breakers.

---

## 3. Wide-region structure theorems (Task 4) — PROVEN

### Theorem S1 (Roman-window rigidity; every m, unconditional)

Let m >= 4, n <= B, q = floor((B-n)/3), r = (B-n) mod 3, and let M
be K3,3-free m x n with E = 3n + q ones (attained by every optimum
wherever z = 3n + q, i.e. the entire Roman window). Then:

(i)   every column has weight 3 or 4, except at most ONE column of
      weight exactly 2, possible only when r = 2;
(ii)  #quads = q + p, #fills = n - q - 2p, where p in {0,1} is the
      number of pads;
(iii) the slot deficit is B - slots(M) = r - 2p in {0,1,2}; when
      r = 0 the columns 2-cover EVERY row-triple exactly twice;
(iv)  the quad multiset is a 2-fold packing whose leave is exactly
      the fill multiset up to r - 2p uncovered slots.

*Proof.* Write n = n_h + n3 + n_p; val = n_h + ex with
ex = sum(w-4) >= 0; slots = 4 val + n3 + X2 with
X2 = sum_{w>=5} (C(w,3) - 4(w-3)) (= 2 per pentad, 8 per hexad, 19
per heptad, increasing); def = sum_pads (3-w) >= n_p. From
B >= slots and E = 3n + val - def = 3n + q, with B - n = 3q + r:

    3 def - n_p + ex + X2 + (B - slots) = r <= 2.

All terms are >= 0 (def >= n_p). A weight->=5 column contributes
ex >= 1 AND X2 >= 2, total >= 3 > r: impossible, so all heavy
columns are quads. def = n_p forces every pad to have weight
exactly 2; 2 n_p <= r gives n_p <= 1 with n_p = 1 only at r = 2 and
then B = slots. Counting yields (ii)–(iv). QED

Machine anchor: `verify_S.py` — the two identities behind the chain
hold on all 81 witnesses; no witness violates the Roman bound.

### Theorem S2 (T-branch purity)

Let T <= n <= B - 3T and suppose z(m,n) = 3n + T (proven for
admissible m, all 3|m with T = J, class m with T = J-2; Theorems
8/F/11 of theorems.md). Then EVERY extremal matrix has no pads,
heavy value exactly T, and a PURE QUAD heavy layer — i.e. the heavy
layer is a MAXIMUM 2-fold quadruple packing:

(a) **admissible m** (B = 4T): slots = 4T + n3 + X2 <= B forces
    n3 = X2 = 0; the window degenerates to n = T and the optimum IS
    a 3-(m,4,2) design. PROVEN.

(b) **3|m** (T = J; B - 4J = L_min = 2m/3 + 2e, e = [m == 9 mod
    12]): every value-J heavy config is pure quad, for EVERY 3|m.
    *Proof.* For 3|m, 2C(m-1,2) == 2 (mod 3), while a block of
    weight w covers C(w-1,2) == 0 (mod 3) pair-slots at each of its
    points unless 3 | w (then == 1). So every point NOT on a
    3|w-weight block has point-leave l_x == 2 (mod 3), hence >= 2,
    giving 3L >= 2(m - sum_{3|w} w k_w). With L = L_min - X2 and
    3 L_min = 2m + 6e this reads

        6e >= sum_w k_w (3 X2(w) - 2w [3|w]),

    with coefficients 6 (w=5), 12 (w=6), 57 (w=7), 108 (w=8),
    162 (w=9), increasing. Since e <= 1, every k_w = 0 except
    possibly k5 = 1 when m == 9 (mod 12). Then L = 2m/3 and every
    l_x = 2 exactly; the pair congruence l_xy == p_xy (mod 2) makes
    the pentad P's ten internal pairs odd, so each x in P spends its
    entire pair-budget sum_y l_xy = 2 l_x = 4 on the four internal
    pairs (value 1 each) — no leave triple crosses the boundary of
    P; the inside part then has sum_{x in P} l_x = 10, which must be
    3·(#inside triples): contradiction. PROVEN, all 3|m.

(c) **class m** (T = J-2; B = 4J+2): every value-(J-2) heavy config
    is pure quad, for EVERY class m. *Proof.* Slots give
    2k5 + 8k6 + 19k7 <= 10: no w >= 7, k6 <= 1, k5 <= 5. Point
    congruence (3 | C(m-1,2)): l_x == -h_x (mod 3), h_x = #hexads
    at x. Cases:
    - k6 = 1: six hexad points need l_x == 2 (mod 3), so
      3L >= 12 > 3(2 - 2k5): dead.
    - k5 = 4 (L = 2): a weight-2 leave touches a point with
      l_x in {1,2}, not == 0 (mod 3): dead (Lemma E verbatim).
    - k5 = 3 (L = 4): the leave is forced onto exactly 4 points
      with all l = 3, hence (via l_x = 4 - y_{x-bar}) the leave is
      one K4^(3), whose pair-parities are all even; so the three
      pentads' odd-pair graph is empty, forcing each pair of P1
      into exactly one other pentad: C(s12,2) + C(s13,2) = 10 with
      s <= 5 — only s12 = 5 (doubled pentad) works, and then every
      pair of P3 is odd: dead. PROVEN by hand.
    - k5 = 2 with |P1 n P2| <= 3: the odd-pair graph has >= 14
      edges on >= 7 points, forcing sum l >= 21 > 18 = 3L: dead.
    - k5 = 1 (L=8); k5 = 2 doubled (L=6); k5 = 2, |P1 n P2| = 4
      (L=6); k5 = 5 (L=0): four finite abstract-support
      classifications, ALL INFEASIBLE (`milp_cases.py` cases A, B,
      C, E; support bounds 8, 11, 6, 12 points proven in the case
      docstrings — case E uses that in a five-pentad all-pairs-even
      system every support point lies in >= 2 pentads, so the
      support has <= 12 points). PROVEN (machine-finite, HiGHS
      infeasibility certificates).

### Corollary S3 (canonical form on the whole wide region)

For m in the three families and every n >= T:
- T <= n <= B-3T: every optimum = (maximum 2-fold quadruple
  packing, leave weight exactly B-4T) + (n-T weight-3 fills inside
  that leave, within multiplicities). Band width = leave capacity
  exactly: n - T <= B - 3T - T = leave weight.
- B-3T < n <= B: Theorem S1's form (quads + fills + <= 1 pad,
  near-exact 2-cover).
- n >= B: Culik profile (classical).

Remaining freedom: WHICH maximum packing (species MaxPacking — many
instances) and WHICH fills. Leave shapes at maximum packings:
doubled pentagon at every class-m packing in the banks (9 distinct
packings across m = 7, 11, 19, 23), doubled hub at every m = 9
packing (4 distinct), doubled parallel class at m = 27; full
uniqueness of the shape is OPEN (the all-doubled case is proven
unique in Theorem 11; note also a maximum packing's leave contains
no quad shadow, else it would extend — an extra constraint on
candidate shapes).

**Machine verification**: `verify_S.py` — all 22 witnesses in
T-branch cells (w_7x24; w_9x40..47; all 8 band_11; the 5
design-prover packings at n = T) match the canonical form EXACTLY:
weights {3,4} only, #quads = T, quad-leave weight = B-4T, fills
inside the leave. 22/22, zero failures.

### Proposition S4 (two open supply units closed) — PROVEN

**S7(1) = 12 and S9(1) = 37.** (Deepband §4 had parity bounds 13
and 38, the last unit OPEN in both.)
- S7(1): a pentad + 13 quads has value 15 = J(7)-2; Theorem S2(c)
  case k5 = 1 is infeasible, so <= 12; 12 attained (deepband).
- S9(1): a pentad + 38 quads on 9 points has L = 6 with every
  l_x == 2 (mod 3), hence all l_x = 2; the S2(b) pentad-pair
  argument forces the leave inside the pentad with
  sum_{x in P} l_x = 10 != 0 (mod 3): contradiction; 37 attained.
  Machine anchor: `milp_cases.py S91` INFEASIBLE.
(S8(1) = 23 vs parity 24 remains OPEN — m = 8 is admissible and
that configuration sits below value T, outside this machinery.)

---

## 4. The completeness audit (Tasks 2–3)

`completeness_audit.csv` — one row per witness: bank, (m,n), edges,
independent legality re-verification, weight profile, decomposition
certificate, status, frontier check, slot saturation, optimality
provenance.

### 4.1 Headline numbers

All 81 files re-verified legal (81/81) with matching edge counts.

Witness-level status (CONSTRUCTIVE = every heavy layer certified
against a Tier-1 generator, label-aware; SPECIES = all layers
Tier-1/Tier-2; PARTIAL = some recognized, some not; UNDECODED = no
heavy layer recognized; fills/pads are moves and never affect the
grade):

    ALL BANKS (81 files):
      CONSTRUCTIVE 30  SPECIES 21  PARTIAL 22  UNDECODED 8
      -> fully decomposed 51/81 (63%)
    THE TWO MANDATED BANKS (69 files):
      CONSTRUCTIVE 27  SPECIES 13  PARTIAL 22  UNDECODED 7
      -> fully decomposed 40/69 (58%)
    HEAVY-LAYER level (all banks): 105/142 layers decoded (74%)
    Ledger certificate val = F_m(#heavy): 53 witnesses "=F" against
    the deepband frontier CSVs (every m = 8, 9 witness), one ">F"
    (w_9x27 — see §5.1).

The honest reading: witness-level full decomposition = 51/81; every
PARTIAL has its top (structured) layer decoded and its quad/pentad
body ledger-certified; nothing in the banks is structureless.

### 4.2 Instance-finiteness is false; species hold

- m = 9 maximum packings: w_9x40/41's layer ~ each other but NOT
  isomorphic to w_9x42's or w_9x43's (aut orders 2, 8, 8;
  doubled-block counts 2, 6, 0) — at least three isomorphism
  classes in one bank, all with doubled-hub leaves (hub at point 8,
  2, 4, 0 across witnesses).
- The band-11 witnesses all reuse ONE 80-packing (identical across
  n = 82..89, the SAT solver moved only fills — itself a striking
  confirmation of the canonical form), with doubled-pentagon leave
  on {0,1,2,4,6}; that packing is NOT isomorphic to the
  design-prover's Z5-symmetric packing.
- Conclusion: the catalogue cannot be a finite list of matrices;
  it can be (and so far is) a finite list of species.

### 4.3 Notable constructive decodes (new this session)

1. **(10,15) = 81 decoded** (theorems.md had flagged "the (10,15)
   structure awaits decoding"): pentads = Cone(0) over one
   TRANSLATION ORBIT (9 circles) of the Moebius-plane-of-order-3
   residual at infinity — 9 of the 18 SQS(10) blocks avoiding a
   point, a single orbit under the AG(2,3) translation group;
   hexads = complements of {0} u (two parallel classes of AG(2,3)
   lines). Fully classical. (probe5 §2, probe6 §e.)
2. **m = 9 mid-band = AG(2,3) base-point calculus**: w_9x22/25/26
   match EXACTLY (all 22 heavy blocks): base b = 0; quads = the 8
   base-line cones {b} u l plus 2 punctured pencil-pairs at b;
   pentads = complements of (line u point) x8 and of pencil-pairs
   x4; the fills of w_9x25/26 sit in the residual leave.
   NON-VACUITY note: w.r.t. a fixed AG(2,3) EVERY 4-subset of the
   9 points is (line u point) or a punctured pencil-pair
   (72 + 54 = 126 = C(9,4), machine-verified — a cute trichotomy
   lemma: every 4-arc of AG(2,3) is a punctured pencil-pair). So
   per-block AG-typing is vacuous at weights 4/5; the certificate
   is the coherent base-point PATTERN with its closed-form counts.
3. **Moebius–Kantor configuration at m = 8**: the 8-pentad layers
   of w_8x13/14 are exactly TC(MK) (the unique 8_3 configuration =
   AG(2,3) minus a point); w_8x15 is TC(sub-MK). The (8,8) optimum
   is the Fano cone PC{p} + TC(Fano) (deepband's decode,
   re-verified); w_8x12 contains a second Fano. MK is NOT a
   sub-system of the maximum 10-pentad system — m = 8 carries at
   least three distinct named pentad geometries (Fano, MK,
   D2(8,5,3)-max).
4. **K5-edge world at (10,21/22)**: rows = E(K5); hexads = K4
   edge-sets, pentads = 6 complementary 5-cycle pairs (C5 +
   pentagram), quads = vertex stars. The m = 10 = C(5,2) shadow of
   the pentagon world of the class-m packings.
5. **w_10x58 completes to a NON-doubled 3-(10,4,2) design** (56
   simple + 2 repeated blocks): the design species strictly exceeds
   doubled Steiner systems. w_10x59 = 2xSQS(10) minus one block;
   w_8x26/27 = 2xSQS(8) minus a (doubled / single) block; w_10x57's
   49 quads and w_8x24's 23 quads also shadow-complete.
6. **OA / transversal codes**: w_10x47's 8 pentads = a lambda = 2
   transversal design over 5 groups of size 2 = OA(8,5,2,2) on the
   matching of uncovered pairs; the m = 9 "parity-partition cones"
   (w_9x9..13, w_9x28) are the coned version (odd/even coset of the
   even-weight code of F2^4). One species covers both.
7. **All four design-prover packings verified as symmetric hole
   packings**: sigma = simultaneous rotation of c5 blocks-of-5
   (c5 = 1, 2, 3, 3 at m = 7, 11, 19, 23), doubled-pentagon hole;
   m = 27 invariant under +1 on each of three 9-blocks,
   doubled-parallel-class hole. w_7x24's quad layer ~ the m = 7
   hole packing (iso verified), so (7,24) = Sub[Z5-HolePacking(7)]
   + 9 fills in the pentagon leave.
8. **m = 9 hexad layers**: complements of {p} u (pair-partition) —
   equivalently, after dropping the avoided point, edge-complements
   of a perfect matching on 8 points (OnSupport-EdgeComp[matching]).

### 4.4 The honest UNDECODED list (Task 3)

Objects with no generative decomposition found (invariants in the
CSV; all are ledger-certified where a frontier CSV exists):

- **m = 8 quad bodies coexisting with pentads** (w_8x15,16,18-23,
  25): 8–25 quads, NOT sub-2xSQS(8) (verified), not
  shadow-completable. The pentad tops ARE decoded (sub-MK,
  sub-max10, Fano). w_8x25 (pure 25 quads, leave 12, no
  completion) is the single pure-quad witness that is not a design
  sub-multiset. These are the S8-ledger bodies; a constructive
  generator for them is the top open decode target.
- **m = 9 mid-band bodies** (w_9x23/24 wholly; the quad bodies of
  w_9x27-38): cone tops decode (pencils, matchings-on-support,
  parity cones); the bodies do not match the AG calculus
  block-for-block. Given decode 4.3.2, the natural conjecture is
  that these are AG-calculus configurations up to solver-noise
  block exchanges; NOT claimed.
- **smoke_10x14** (7 hexads + 7 pentads): nothing found (not
  SQS-derived, no cone, no splits). A smoke-test witness for a
  sub-band cell; counted honestly as UNDECODED.
- **m = 10 bodies** (w_10x35/36/47/48/53/54): split/OA tops decode;
  the bodies are exact-2-cover residuals — w_10x36's joint config
  saturates slots = 240 = B exactly (an exact 2-cover by 8 splits
  + 20 quads: a striking object, possibly a new primitive).

Verdict for the conjecture: nothing here is evidence AGAINST
species-finiteness — every undecoded body sits inside a
species-certified joint configuration — but Tier-1 constructive
coverage of the DEEP BAND is incomplete, and these bodies are
where new primitives should be mined next.

### 4.5 Catalogue additions this session

TC(Moebius–Kantor) and its subs (m = 8); Moebius-residual
translation-orbit cones (m = 10). Both classical — the catalogue
grew by absorption, consistent with the graph-minors shape.

---

## 5. Data corrections and collateral findings

1. **F9(27) >= 35**: w_9x27's heavy layer has c = 27, val = 35;
   `frontier_m9_final.csv` records F = 34 there (a stale entry: the
   deepband report's §5 slice questions were never resolved — its
   report file still contains unsubstituted [F24]/[QVERDICTS]
   placeholders, and `s9_slices.jsonl` shows UNRESOLVED
   timeouts). So F9(27) in {35, 36}, with 36 = deepband's open
   Q-G. All other 52 m = 8/9 heavy layers match their CSV frontier
   values exactly.
2. **z(9,24..27) status**: the bank's w_9x24/25/26/27 are LB
   witnesses for cells that are still OPEN (their z_truth is empty
   in witness_configs.csv; the deepband §5 verdicts were never
   obtained). The audit CSV records them as "LB-witness (cell
   open)", not optimal. No claim in this report depends on them.
3. **Proposition S4**: the k5 = 1 supply units at m = 7 and m = 9
   are now proven (S7(1) = 12, S9(1) = 37); the m = 8 unit stays
   open.
4. **AG(2,3) 4-set trichotomy lemma** (machine-verified,
   elementary): the 126 4-subsets of AG(2,3)'s point set split as
   72 (line u point) + 54 punctured pencil-pairs; equivalently
   every 4-arc is a punctured pencil-pair in the pencil of its
   diagonal point. Recorded because it is what makes the m = 9
   quad world "all-geometric" and forces AGC certificates to be
   pattern-based.

---

## 6. The WQO angle (Task 5) — formulation and honest verdict

**The order.** For block-multiset matrices write M <= N iff there
are injections rho: rows(M) -> rows(N), gamma: cols(M) -> cols(N)
with rho(B_j) subset of B'_{gamma(j)} for every column j — row
deletion + column deletion + block shrinking ("capacity-monotone
minor"). Every move weakly decreases coverage, so the class of
K3,3-free matrices is downward closed under <= (the analogue of
minor-closedness holds).

**Fact 1 (ambient class not WQO).** The K3,3-free class contains an
infinite antichain: the cycle matrices C_k (columns = edges of a
k-cycle; all weights 2, trivially K3,3-free). A weight-2 block must
embed INTO a weight-2 block, so C_j <= C_k iff C_j is a subgraph of
C_k — false for j != k. Hence no Robertson–Seymour-style theorem
can come from the ambient class alone.

**Fact 2 (the obstruction-set implication fails as stated).** The
EXTREMAL class is not downward closed (deleting a column of an
optimum need not be optimal for the smaller cell), and WQO of a
non-ideal has no forbidden-minor consequence. The natural repair —
pass to a downward-closed superclass that still pins z, such as
{M : no pads, slot deficit <= 2} (which contains every Roman-window
optimum by Theorem S1) — fails too: block shrinking destroys
saturation, so that class is not <=-closed either. We found no
natural <=-closed class strictly between "extremal" and "all
K3,3-free". This is the structural reason the graph-minors ROUTE
(WQO => finite obstruction set => decidability) does not transfer,
while the graph-minors SHAPE (finitely many primitive species,
sporadics absorbed) does transfer — via §3's theorems rather than
via order theory.

**Fact 3 (what IS true).** On the wide region, Corollary S3 makes
the extremal class the image of {maximum packings} x {fill
patterns}. At fixed leave shape the fill patterns form words over a
finite alphabet (leave slots, multiplicity <= 2), WQO by Higman's
lemma. The packing side reduces to: is {MaxPacking(m)}_m WQO under
<=? Sub-question (recorded, untested): are the Z5 hole packings a
chain, i.e. HP5(m) <= HP5(m') for m < m' via the blocks-of-5
embedding? Even if yes, Fact 2 means this yields structure, not
decidability.

**Verdict.** WQO is here a consequence-shaped statement of the
catalogue (species-finiteness + per-species parameter chains), not
a route to it. Time spent ~1.5 h, per brief.

---

## 7. Strongest true statements proven in this session

1. Theorem S1 (all m, unconditional) + Theorem S2 (purity:
   admissible; ALL 3|m at value J; ALL class m at value J-2) +
   Corollary S3: on n >= T33(m), for the three families where T is
   known, every extremal K3,3-free matrix decomposes as
   MaxPacking + fills-in-leave (+ <= 1 pad at the Roman elbow) —
   the strong form of the finite-catalogue conjecture is a THEOREM
   on the wide region.
2. Proposition S4: S7(1) = 12, S9(1) = 37 (closing two open units
   of the deepband supply theory).
3. The audit facts of §4, each machine-verified: 81/81 legality;
   the constructive decodes incl. (10,15), the AG base calculus,
   MK, the K5-edge world, the OA family, the non-doubled
   3-(10,4,2) completion; sigma-invariance of the four hole
   packings; the instance-finiteness refutation; the AG trichotomy
   lemma; the canonical-form verification 22/22.
4. Honest open core: constructive (Tier-1) decodes for the
   deep-band ledger bodies; leave-shape uniqueness at maximum
   packings; FC (strong) below T.

---

## 8. Reproducibility

Venv: workspace zvenv (scipy 1.18 / HiGHS; pysat unused).
- `catlib.py` — witness IO, independent legality verifier,
  invariants, signature-pruned embedding / isomorphism /
  automorphism backtracking (no external iso library).
- `gens.py` — constructive generators: Fano; SQS(8) = AG(3,2)
  planes; SQS(10) = Moebius plane of order 3 via GF(9) and the
  PGL(2,9) orbit; AG(2,3); biplane = QR(11) difference set;
  K5-edge families; F2^4 hyperplane/cap families; pentagon triples.
- `audit.py`, `audit2.py` — recognizers + audit CSV writer
  (`audit_run.log` = full per-witness log).
- `probe1.py`..`probe7.py` — the decode experiments quoted in §4.
- `milp_cases.py` — the five abstract-support MILP infeasibility
  certificates (cases A, B, C, E, S91).
- `theorem_checks.py` — first-generation DFS enumerators
  (superseded by milp_cases; kept for the record).
- `verify_S.py` — Theorem S1 identities + Corollary S3 canonical
  form on the banks (22/22).
- `completeness_audit.csv` — the audit table (81 rows).
Machine budget: <= 1 solver at a time; each MILP minutes or less;
no z-cell optimization run anywhere in this session.

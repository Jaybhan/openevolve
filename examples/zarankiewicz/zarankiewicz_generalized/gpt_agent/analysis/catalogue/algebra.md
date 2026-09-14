# The generator algebra for K₃,₃-free extremal matrices (s = t = 3)

Structure theorist, catalogue session, 2026-07-30. Labels per
BASE/README.md. Everything stated here about specific witnesses is
machine-verified by the scripts in this directory (`audit2.py`,
`probe*.py`, `milp_cases.py`); the theorems cited are proven in
`report.md` §3.

## 0. Dictionary and ground rules

An m×n K₃,₃-free 0/1 matrix ⟺ a multiset of n column-blocks
(subsets of the m rows) with every row-triple covered ≤ 2 times
(a 2-fold triple packing with block multiplicities ≤ 2).
B = 2·C(m,3), T = T₃,₃(m) = max 2-fold quadruple packing size,
J = Johnson bound. A *layer* is the sub-multiset of columns of one
weight. The *leave* of a config C is y(T) = 2 − cov_C(T) ≥ 0.

The algebra has two tiers, and the distinction is load-bearing:

- **Tier 1 (constructive generators)**: families built from classical
  objects by an explicit rule; membership certificate = a labeled
  isomorphism onto the construction.
- **Tier 2 (species)**: families defined by a polynomially checkable
  structural predicate (classified leave, saturation, frontier value),
  with existence witnessed but no classical construction identified.

The FINITE CATALOGUE conjecture (report.md §2) is about finiteness of
the list of SPECIES, not of instances: the audit (§below, and
`completeness_audit.csv`) shows instance-finiteness is FALSE — e.g.
m = 9 carries at least three non-isomorphic maximum packings
(aut orders 2, 8, 8), all in one species.

## 1. Tier-1 generator species

G1. **Pads** P(m): any column of weight ≤ 2. Consumes no capacity.

G2. **Fills** F(m; T): a weight-3 column on triple T; consumes 1 slot
    of T. Legal iff current leave y(T) ≥ 1.

G3. **Point-complements** PC(m; S): {[m]∖{x} : x ∈ S}, S ⊆ [m],
    multiplicities ≤ 2. (Weight m−1. The m ≤ 6 "omission codes".)

G4. **Edge-complements** EC(m; H): {[m]∖e : e ∈ E(H)} for a graph H
    on [m]. Legal ⟺ H triangle-free (Theorem 2 of theorems.md).
    Named sub-families observed: H = K₃,₃ (row 6 Turán optimum),
    C₄, matchings, single edges.

G5. **Triple-complements** TC(m; 𝒯): {[m]∖T : T ∈ 𝒯} for a 3-graph 𝒯.
    Named sub-families observed in the banks:
    - 𝒯 = Fano plane (m = 8: the (8,8) optimum is PC{p} ⊕ TC(Fano on
      [8]∖p) — the "Fano cone"; also inside w_8x12);
    - 𝒯 = Möbius–Kantor configuration 8₃ (m = 8 pentad layers of
      w_8x13/14 are exactly TC(MK); w_8x15 is TC(sub-MK)).
      MK = AG(2,3) minus a point; unique 8₃ configuration;
    - 𝒯 = pencil-partition: {p} ∪ Pᵢ with P₁..P₄ a partition of
      [m]∖p into pairs (m = 9 hexad layers: all of w_9x9..13);
    - 𝒯 = parallel classes (w_8x22/23), pencils, single/doubled
      triples.

G6. **AG(2,3) base-point calculus** AGC(9; b, selection): fix AG(2,3)
    on the 9 rows and a base point b. Generator blocks:
    - base-line cones {b} ∪ ℓ (ℓ a line avoiding b) — 8 available;
    - pencil-pair quads (ℓ₁ ∪ ℓ₂) ∖ {b}, ℓ₁ ∩ ℓ₂ = {b} — 6 available;
    - complements of (line ∪ point) 4-sets — pentads;
    - complements of pencil-pair quads — pentads;
    - line-complements — hexads;
    - lines through b — fills.
    Witnesses w_9x22/25/26 lie in AGC(9) EXACTLY (every heavy block
    matched, `probe4.ag_decode_full` all_matched = True).
    *Non-vacuity caveat (important)*: w.r.t. a fixed AG(2,3), EVERY
    4-subset of the 9 points is (line ∪ point) or a punctured pencil-
    pair (72 + 54 = 126 = C(9,4) — verified). So "each block matches
    some AG shape" is vacuous for weights 4/5; the certificate for
    AGC is the COHERENT pattern: one base b, the 8 base-line cones,
    pencil-pairs at that same b, and the closed-form counts above.
    Line-complements (12/84 of 6-sets) are individually meaningful.

G7. **Möbius-plane / SQS families**:
    - 2×SQS(8) = doubled planes of AG(3,2) (m = 8);
    - 2×SQS(10) = doubled Möbius plane of order 3 (m = 10;
      w_10x59 = 2×SQS(10) minus one block);
    - residual-circle cones (m = 10): Cone(q) over one translation
      orbit (9 circles) of the Möbius-plane residual at ∞ — the
      (10,15) optimum T2_10x15, previously undecoded, is
      Cone(0, orbit) ⊕ TC'({0}∪(two parallel classes of AG(2,3)))
      — fully classical (`probe5/6`).
    - General species: **3-(m,4,2) designs** (Hanani: exist ⟺
      m ≡ 2, 4 (mod 6)); sub-multisets certified by shadow
      completion. w_10x58 completes to a NON-doubled 3-(10,4,2)
      (56 simple + 2 repeated blocks) — the species is strictly
      larger than doubled Steiner systems.

G8. **Biplane pair** BP(11): the 2-(11,5,2) biplane (QR difference
    set mod 11) and its block-complements (a 2-(11,6,3)); w_11x22 =
    biplane ⊕ complement-design, both layers exactly.

G9. **K₅-edge world** K5E(10): rows = E(K₅). Blocks: K₄ edge-sets
    (hexads), 5-cycle edge-sets (pentads, in complementary pairs
    C₅/pentagram), vertex stars (quads). w_10x21/22 decompose as
    K4sets ⊕ 5-cycles ⊕ stars exactly. (This is the m = 10 = C(5,2)
    shadow of the doubled-pentagon world of the class-m packings.)

G10. **Splits** SP(m; 𝒫): both sides of partitions {P, [m]∖P},
     multiplicity ≤ 2 (doubling allowed). Observed: 8 splits at
     w_10x35/36 (16 pentads), 4 at w_10x48, 2 doubled at w_10x53/54,
     1 doubled at w_10x57; the (16,16) cap family is the F₂-linear
     special case (both sides of 8 cap-normal hyperplanes).

G11. **Transversal parity codes** OA(m; partition, code):
     pair-partition of a subset of rows, blocks = transversals
     selected by a binary code of strength 2 (an orthogonal array),
     optionally coned:
     - m = 9 pentad layers: ParityCone(q) = {q} ∪ (odd- or even-
       weight coset transversals of a pair-partition of [m]∖q) —
       w_9x9..14, w_9x28 (this is the deepband "parity-partition
       cone", now an OA statement);
     - m = 10, no cone: w_10x47's 8 pentads = the blocks of a
       λ = 2 transversal design TD₂(5,2) = OA(8,5,2,2) over the
       matching of uncovered pairs (`probe6`).

G12. **F₂-hyperplane families** (from the evolved champion; engine
     side, not in the audited banks): rows ⊆ F₂⁴∖0, columns
     {x : ⟨a,x⟩ = 1}; cap-doubling at m = 16 (both sides of 8
     cap-normal hyperplanes = G10 ∘ linear algebra). Kept in the
     catalogue for completeness of the record.

G13. **Symmetric hole packings** (design_prover witnesses; all four
     verified here):
     - Z₅-hole packings HP5(m; c₅): points = c₅ pentagon 5-cycles +
       fixed points; σ = simultaneous rotation; blocks = σ-orbits of
       quads; leave = doubled pentagon on the first 5-cycle.
       Instances: m = 7 (c₅=1), 11 (c₅=2), 19 (c₅=3), 23 (c₅=3) —
       σ-invariance and hole verified. w_7x24's quad layer ≅ HP5(7).
     - Z₉-parclass packing (m = 27): invariant under +1 on each of
       three 9-blocks; leave = doubled parallel class.

## 2. Tier-2 species

S1. **MaxPacking(m; leave-shape)**: 2-fold quadruple packings of the
    maximum size T₃,₃(m) whose leave is the classified minimal shape:
    - class m (≡ 3 mod 4, ∤ 3): doubled pentagon (weight 10);
    - m ≡ 9 (mod 12): doubled hub = doubled pencil-partition
      (weight 2m/3 + 2);
    - other 3|m: doubled parallel class;
    - admissible m: leave ∅ (then = species G7, a 3-(m,4,2)).
    Every maximum packing in the banks is in this species: all four
    m = 9 40-packings (hub), all eight band-11 80-layers + HP5(11)
    (pentagon), HP5(19/23), Z₉(27). The band-11 80-layers are NOT
    isomorphic to HP5(11) — the species has many instances.

S2. **Ledger/frontier configs** LF(m; c): heavy configs with
    val = F_m(c) (deep-band Pareto frontier). Species predicate =
    frontier value + the §4-deepband congruence-tight leave. All
    m = 8, 9 bank witnesses have val = F_m(c) against the deepband
    CSVs (one exception: w_9x27 has val 35 > the stale CSV entry 34
    at c = 27, i.e. it IMPROVES the recorded F₉(27) lower bound —
    recorded in report.md §5).

S3. **Residual bodies**: quad layers saturating the capacity left by
    a Tier-1 top layer (e.g. w_9x27/28: 19/20 quads on the 8 points
    avoiding the cone point, filling all capacity not consumed by the
    parity cone's shadows). Predicate: joint slots = B − r with the
    Theorem-S1/S2 bounds (report.md).

## 3. Composition moves

M1. **Side-by-side** ⊕: union of column multisets (legal iff joint
    coverage ≤ 2 — checked on the pooled leave).
M2. **Doubling** 2×: multiplicity-2 copies (legal iff the object's
    own coverage ≤ 1, e.g. Steiner systems, Fano lines).
M3. **Complementation** κ: blocks B ↦ [m]∖B (turns triple systems
    into pentad layers at m = 8, quads into pentads at m = 9, etc.).
M4. **Cone / base-point extension** Cone(q, ·): new point q added to
    every block of a derived family on [m]∖q (parity cones, base-line
    cones, the (10,15) circle cone; the hub structure of m = 9 max
    packings is the same move seen in the leave).
M5. **Sub-multiset restriction** (sub-packing): take any sub-multiset
    of a generator instance (legality is inherited). This is the move
    behind Theorem 8's realization and all "Sub[...]" certificates.
M6. **Canonical completion**: given a heavy config, add fills into
    the leave (≤ y(T) each) and pads to reach n columns (the
    champion's `_fill`; the ONLY move needed beyond M1–M5 to convert
    packings into z-witnesses; Theorem S1/S2 prove its necessity, not
    just sufficiency, in the wide region).
M7. **Puncturing**: delete a row and restrict blocks (MK = punctured
    AG(2,3); Möbius residual = punctured SQS(10); the m = 9 → m = 8
    body split of w_9x27/28).

## 4. The grammar

    witness ::= relabel(π) [ layer (⊕ layer)* ⊕ fills ⊕ pads ]
    layer   ::= T1-instance | 2× T1-instance | κ(T1-instance)
              | Cone(q, layer') | Sub(T1-instance | S-instance)
    T1-instance ::= PC | EC[named H] | TC[named 𝒯] | AGC(9)
              | SQS-family | BP(11) | K5E(10) | SP | OA | F₂-family
              | HP5 | Z₉-parclass
    S-instance ::= MaxPacking(m; classified leave) | LF(m; c)
              | residual body

with the legality side-condition (joint coverage ≤ 2) at every ⊕.

## 5. Finiteness statement the audit supports

The species list above is FINITE (13 Tier-1 + 3 Tier-2 species).
Final audit numbers (`completeness_audit.csv`, 81 files, all
re-verified legal): witness-level 30 CONSTRUCTIVE + 21 SPECIES + 22
PARTIAL + 8 UNDECODED (51/81 fully decomposed); heavy-layer level
105/142 layers decoded (74%); on the two mandated banks (69 files):
27 CONSTRUCTIVE + 13 SPECIES = 40/69 fully decomposed. Every m=8/9
witness additionally carries the ledger certificate val = F_m(c).
The instance-level catalogue is provably NOT finite per m-slice
(non-isomorphic max packings at m = 9 and m = 11; growing design
counts), which is why the conjecture in report.md §2 is formulated
at species level. On the wide region n ≥ T the grammar's
completeness is a THEOREM (report.md §3, Corollary S3): the only
derivations that exist there are MaxPacking ⊕ fills (T-branch) and
quads ⊕ fills ⊕ ≤ 1 pad at near-saturation (Roman branch).

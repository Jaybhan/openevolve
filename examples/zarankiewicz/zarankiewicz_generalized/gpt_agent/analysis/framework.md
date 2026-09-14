# The reduction framework: z(m,n;s,t) as demand vs. extremal supply

Coordinator's synthesis document. This is the spine the final deliverable
hangs on; sections get upgraded to PROVEN as certificates land.

## 1. The frame (exact, general s,t)

A K_{s,t}-free m×n 0/1 matrix ⟺ a multiset of n **blocks** (column supports,
subsets of [m]) such that every s-subset of rows lies in at most t−1 blocks
("(t−1)-fold s-packing"). Edges = Σ|blocks|. z(m,n;s,t) = max Σ|B_j|.
Duality: z(m,n;s,t) = z(n,m;t,s) — always work with the smaller side as rows.

## 2. Level 0 — the demand curve (waterfill bound) [PROVEN]

Global capacity: Σ_j C(|B_j|, s) ≤ (t−1)·C(m,s). Since C(·,s) is convex,
the integer max of Σ|B_j| under the budget is the level profile (marginal
cost of raising a block w→w+1 is C(w, s−1), increasing), heights capped at m;
minimized against the transposed bound. Call it WF(m,n;s,t).
**WF is what the numbers "want"; deficits are what geometry refuses.**

## 3. Level 1 — the supply layer (realizability of heavy blocks)

The WF profile demands specific counts of heavy blocks. Heavy blocks live on
the **complement scale**: weight m−1 ↔ points, m−2 ↔ pairs (edges of a graph
H on the rows), m−3 ↔ triples, ...; and for s=3 the legality of a family is
an *induced-density* condition on complements:

  mult(T) = Σ_levels (induced count of chosen complements inside comp(T)) ≤ t−1.

Proven instances of the reduction (s=t=3; analysis/theorems.md):
- m ≤ 5: supply always meets demand (point-complements auto-legal; weight-3
  blocks are free capacity) → z = WF for ALL n. [PROVEN]
- m = 6: weight-4 blocks = edges; legality ⟺ H triangle-free; supply is
  Turán's ex(6,K₃) = 9 → d(6,n) = max(0, demand₄(n) − 9)-shaped; three
  deficit cells proven this way, (6,9) witness = K_{3,3} itself. [PROVEN]
- m = 16 corner: columns = both sides of hyperplanes with cap normals;
  supply = capmax(PG(3,2)) = 8 → 16 columns, z = 128. [witness VERIFIED;
  optimality = published value; "supply-limited" reading is ours]

**Working law (C5, sharpened): every deficit cell of the (3,3) table is a
demand-exceeds-supply statement for a classical extremal object** (Turán
numbers; caps/ovoids; projective planes and their truncations; Fano-type
packings at m=7). To be tested by machinery on rows 7–15.

Cross-family check, (2,2): WF(8,8;2,2) = 25 but z(8,8;2,2) = 24 (known) —
the first (2,2) deficit appears exactly where partial-plane supply runs out.
Same law, different classical object. [to VERIFY against theory agent's
(2,2) table]

## 4. Level 2 — algebraic realizers (the constructive side)

Families that MEET supply limits (engineer implementing):
- Point/edge/triple-complement families with induced-density legality
  (realizes small-m and near-diagonal profiles; H from Turán graphs).
- F₂ᵏ affine hyperplane families: rows X ⊆ F₂ᵏ with no even zero-sum
  ≤ s (a code condition); coverage of an s-set is 0 or 2^{k−rank} ≤ 2^{k−s};
  gives clean (t−1)-fold packings when 2^{k−s} ≤ t−1.
- Cap doubling: both sides of hyperplanes with normal-set meeting every
  (s−1)-flat ≤ t−1 times (caps for s=3) — saturates at 2·capmax.
- PG(2,q) machinery: line incidences ((2,2) optimal), line-complements
  doubled ((3,3) at m = q²+q+1... m=7 case), Singer difference sets
  (cyclic form of the same).
- Culík completion: (t−1) copies of each s-subset + weight-(s−1) pads —
  PROVEN exact for n ≥ (t−1)C(m,s).

## 5. The master formula (target shape)

    z(m,n;s,t) = max { E(profile) : profile within global budget,
                       heavy part realizable }

with realizability CHARACTERIZED (not searched) per regime:
- m ≤ s+2: always realizable → z = WF. [PROVEN for s=3; conjectured general]
- Elongated n: Culík. [PROVEN]
- Complement scale m−2: Turán characterization. [PROVEN m=6; general m:
  the same correspondence with (m−3)-subset induced conditions — supply
  formula to derive]
- Algebraic corners: cap/plane saturation. [(16,16) done; (7,n), rows 8–15
  hyperplane regime — conditions to derive]

Honest status: this is a research program, not yet a closed formula. The
formula's VALUE even now: it explains, cell by cell, *why* each proven value
is what it is — the interpretability goal of the whole project — and each
regime that gets characterized turns a slice of the open problem into
classical extremal theory.

## 6. Open edges of the framework (recorded, per instruction to note findings)

- Uniqueness questions: is the (16,16) extremal matrix unique up to iso?
  ((6,9)'s is, if Turán uniqueness lifts through the correspondence — check.)
- The (s,t)=(3,3), m=7 row: does doubled-Fano + completion attain ALL of row
  7? If yes, row 7 = "Fano supply" theorem.
- General-m complement-edge law: legality for weight-(m−2) blocks is
  "every (m−3)-subset of V(H) induces ≤ 2 edges" ⟺ H has ≤ 2 edges OR
  m−3 ≥ (max over...) — derive the exact Turán-type function; it's
  ex(m, {graphs with 3 edges on ≤ m−3 vertices})-flavored.
- (4,4) and beyond: even the ORDER of z(n,n;4,4) is open; our framework
  contributes constructions + certified small cells only. No overclaim.

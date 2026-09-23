# dfield/finite-zarankiewicz-closures — column-profile branching, arithmetic pruning, Lean confirmation

Literature notes for the MEng thesis "Discovering upper bounds for the Zarankiewicz numbers". This is the closest prior work to our design (proposal §2, ref. [15]); it is documented exhaustively here because its *structure* (profile enumeration → arithmetic kills → SAT/MIP on survivors → Lean for the arithmetic, external checkers for the certificates) is the pipeline we are trying to make evolvable and Lean-gated.

Everything below was read from the repository itself (cloned 2026-09-21; HEAD `a48f6e99`, 2026-07-14/15). Where I ran the repository's own code to reproduce a number, I say so (§10). Where I *infer* something not stated by the source, it is marked **[inference]**.

---

## 0. Bibliographic record

| Field | Value |
|---|---|
| Title | *Eight exact finite Zarankiewicz numbers* (repository README / CITATION.cff) |
| URL | https://github.com/dfield/finite-zarankiewicz-closures |
| GitHub owner | `dfield` (git author on every commit: Dylan Field `<dylan@figma.com>`) |
| Declared authors (CITATION.cff, pyproject) | "GPT 5.6-Sol", "Claude (Anthropic)", "OpenAI Codex" — i.e. the work is explicitly LLM-generated; the human owner is not listed as an author |
| Version / date | CITATION.cff `version: 2.0.0`, `date-released: 2026-07-14`; repo created 2026-07-04, last push 2026-07-15 |
| License | MIT |
| Status | "research artifact awaiting independent expert review, not a peer-reviewed publication" (README, first line). GitHub shows 0 stars, 1 fork as of 2026-09-21 |
| History | 64 commits (2026-07-04 → 2026-07-15), single git author; deliberate "proof-first" root commit `b6571804` *"docs: establish the human proof before implementation"* containing only `docs/PROOF.md` |
| Size | git repo ~1.9 GB (checked-in compressed DRAT proofs); GitHub release `z10-23-certificate-v1` (2026-07-15) holds 16 assets totalling 24,406,502,316 bytes; the manifest counts 25,345,672,172 bytes of compressed proof material overall |
| Relation to us | Cites Bhan–Nobili–Langer arXiv:2605.01120v2 Figure 2 as its source table: "Bhan--Nobili--Langer listed 44 open cells … The eight repository equalities make 11 of those 44 exact, so **33 remain open**." |

BibTeX-style key used below: **[dfield26]**.

---

## 1. What the repository claims

Exact values of Z(m,n,3,3) (max ones in an m×n 0/1 matrix with no all-ones 3×3 submatrix):

```
Z(9,23,3,3)=103    Z(10,21,3,3)=106    Z(10,22,3,3)=110    Z(10,23,3,3)=112
Z(11,19,3,3)=106   Z(11,20,3,3)=111    Z(11,23,3,3)=123    Z(12,23,3,3)=134
```
plus the frontier bound `Z(13,23,3,3) ≤ 144` (no matching construction; propagated interval 139–144).

Upper-bound method per cell (README table, verbatim labels):

| Cell | Upper-bound method | Lean status (lean/README.md) |
|---|---|---|
| (9,23)=103 | marked-row deficit | end-to-end: `Zarankiewicz.Exact.Z9_23.exact_value` |
| (10,21)=106 | vertex deletion from Z(9,21)=96 (Tan) | conditional on `UpperBound 9 21 96` |
| (10,22)=110 | pair-deficit residues | end-to-end: `Zarankiewicz.Exact.Z10_22.exact_value` |
| (10,23)=112 | arithmetic profile reduction + DRAT/LRAT + exact SCIP/VIPR | conditional on `UpperBound 10 23 112` (external certificates) |
| (11,19)=106 | vertex deletion from Z(11,18)=101 (Tan) | conditional on `UpperBound 11 18 101` |
| (11,20)=111 | two deletion steps | conditional on `UpperBound 11 18 101` |
| (11,23)=123 | minimum-row deletion from Z(10,23)=112 | conditional on `UpperBound 10 23 112`; deletion formalized |
| (12,23)=134 | two-stage row/pair deficit | end-to-end: `Zarankiewicz.Exact.Z12_23.exact_value` |
| (13,23)≤144 | finite profile + marked-row deficit | end-to-end: `Zarankiewicz.Bounds.Z13_23.upper_bound` |

Source intervals they started from (their transcription of our Figure 2): (9,23) 103–104, (10,21) 106–108, (10,22) 110–111, (10,23) 112–115, (11,19) 102–108, (11,20) 111–112, (11,23) 118–125, (12,23) 125–136, (13,23) 135–145.

Lower bounds: eight explicit matrices in `data/*.csv`, also embedded as column bitmasks in `lean/Zarankiewicz/Witnesses.lean` and checked by kernel `decide`. Five of the eight matching lower bounds were ours (data/README.md: "Bhan--Nobili--Langer publicly supplied five of the matching lower bounds"); (11,19)=106 (ours was 102) and (12,23)=134 (ours was 125) are new witnesses found by simulated annealing / per-profile SAT.

Remaining open cells (their list): (12,n) for n∈{17,…,21}; (m,n) for 13≤m≤16, 17≤n≤23. The propagated table `analysis/new_bounds.json` tightens 21 upper and 17 lower bounds of our intervals (§8.3).

---

## 2. Their matrix model and notation

Human proofs (docs/PROOF*.md): column j has support E_j ⊆ [m], degree d_j = |E_j|; for a row triple T, λ_T = #{j : T ⊆ E_j}. K_{3,3}-freeness ⇔ λ_T ≤ 2 for all T. δ_T = 2 − λ_T ≥ 0 is the *deficit* of T.

Lean (`lean/Zarankiewicz/Basic.lean`, Mathlib):

```lean
abbrev BinaryMatrix (R C : Type*) := R → C → Bool
def columnSupport (A : BinaryMatrix R C) (c : C) : Finset R := Finset.univ.filter fun r => A r c = true
def columnDegree (A) (c) : Nat := #(columnSupport A c)
def edgeCount (A) : Nat := ∑ c, columnDegree A c
def rowTriples (R) : Finset (Finset R) := Finset.univ.powersetCard 3
def tripleLoad (A) (T : Finset R) : Nat := #(Finset.univ.filter fun c => T ⊆ columnSupport A c)
def commonColumnCount (A) (a b c : R) : Nat := #(Finset.univ.filter fun j => A a j = true ∧ A b j = true ∧ A c j = true)
/-- increasing triples, "matching the SAT encoding" -/
def K33Free [LinearOrder R] (A : BinaryMatrix R C) : Prop :=
  ∀ a b c : R, a < b → b < c → commonColumnCount A a b c ≤ 2
def LowerBound (m n e : Nat) : Prop := ∃ A : BinaryMatrix (Fin m) (Fin n), K33Free A ∧ e ≤ edgeCount A
def UpperBound (m n e : Nat) : Prop := ∀ A : BinaryMatrix (Fin m) (Fin n), K33Free A → edgeCount A ≤ e
def Exact (m n e : Nat) : Prop := LowerBound m n e ∧ UpperBound m n e
```

Compare ZarPrune: `Mat m n := Fin m → Fin n → Bool` (same), `HasKst` via increasing index tuples `Incr R ∧ Incr C ∧ ∀ a b, A (R a) (C b) = true` (theirs is the s=t=3 special case, phrased through a cardinality `≤ 2` on column sets; ours is the general ∃ s×t all-ones block). Their `UpperBound m n e` is our `∀ A, ¬ HasKst P A → weight A ≤ e` (`upper_bound_succ_of_cover` form). Their `Exact` bundles a witness.

---

## 3. How the matrix is split into branches

**Short answer: the branching unit is the unordered column-degree histogram ("profile") at the first excluded weight; rows are not partitioned at all.** Inside a profile, the SAT instance uses lexicographic symmetry breaking on rows and on equal-degree columns plus a neighbour-bound cut on row degrees; when a profile is too hard for one CaDiCaL run, it is split further by a canonical "row-stabilizer" cube-and-conquer on successive column supports, or (for two profiles) by row-symmetry orbit covers solved as exact MIPs.

### 3.1 Level 0: reduce to exactly w ones

All upper-bound proofs start with `exists_submatrix_of_edgeCount_le` + `K33Free.mono`: if a K33-free matrix has ≥ w ones, delete ones until exactly w remain (still K33-free). So every case analysis is at *exactly* the first excluded weight w = Z_claimed + 1 (104, 111, 113, 135/136, 145). This is the same normalization our `Valid P A := ¬HasKst ∧ P.w ≤ weight A` needs but does not yet perform **[inference: we should add the "thin to exactly w" lemma; it is Mathlib-free provable]**.

### 3.2 Level 1: column-degree profiles

Triple capacity (Roman / KST-type, §4 P1): Σ_j C(d_j,3) ≤ 2·C(m,3). With Σ_j d_j = w and Σ_j 1 = n this bounds the histogram. They enumerate all histograms (n_0,…,n_m) with Σ n_d = n, Σ d·n_d = w, Σ C(d,3)·n_d ≤ 2C(m,3) (`extended.enumerate_degree_profiles`, `filters.all_profiles`; both standard library, recursion over degrees with convexity pruning).

The *penalty line* trick turns this into an explicit finite classification: pick the two degrees a,a+1 straddling w/n and write C(d,3) = α·d − β + p(d) with p ≥ 0 and p(a)=p(a+1)=0; then Σ_j p(d_j) ≤ 2C(m,3) − α·w + β·n =: budget. Concretely:

| m | p(d) | table p(0..m) | cells |
|---|---|---|---|
| 9 | C(d,3) − 6d + 20 | 20,14,8,3,0,0,4,13,28,50 | (9,23) at 104: budget 4 → 3 profiles |
| 10 | C(d,3) − 10d + 40 | 40,30,20,11,4,0,0,5,16,34,60 | (10,22) at 111: budget 10 → 4 profiles; (10,23) at 113: budget 30 → 25 profiles |
| 12 | C(d,3) − 10d + 40 | …,60,95,140 | (12,23) at 136: budget 0 → 1 profile; at 135: budget 10 → 5 profiles |
| 13 | C(d,3) − 15d + 70 | 70,55,40,26,14,5,0,0,6,19,40,70,110,161 | (13,23) at 145: budget 7 → 3 profiles |

Profile counts at the first excluded weight (my run of `search/filters.py`'s `all_profiles`, §10): (9,23,104): 3; (10,22,111): 4; (10,23,113): 25; (11,23,124): 6; (12,23,135): 5; (12,23,136): 1; (13,23,145): 3; (11,21,117): 1; (12,22,133): 0.

**Contrast with Tan (2022).** Tan fixes *both* a row-sum partition and a column-sum partition per branch ("partition pairs"). dfield fixes only the column histogram and lets the SAT solver handle rows, with (i) double-lex symmetry breaking and (ii) the cut "every row has degree ≥ w − Z(m−1,n)" (=10 at (10,23,113), from Z(9,23)=103). **[inference]** This is a coarser decomposition (25 branches for (10,23) vs. hundreds of partition pairs), which is why individual branches are heavy (proofs up to 415 MB compressed, one profile needing 17,170 cubes).

**Contrast with ZarPrune.** Our `Profile m n := {row : Fin m → Nat, col : Fin n → Nat}` is the ordered row-and-column *vector*. Every dfield prune is a function of the column histogram (plus m,n,w and external bound tables), so all of them are expressible as `kill : Profile P.m P.n → Bool` that ignores `row` (or uses it only for the row-degree cut). A prune on vectors specialises to a prune on histograms, as our README notes.

### 3.3 Level 2: the per-profile SAT instance (`search/sat_tool.py::build`, `search/z10_23_certify.py::build_profile_formula`)

Deterministic DIMACS via python-sat, for m=10, n=23, column degrees `degrees = ordered_degrees(profile)` (rare degree blocks first, equal degrees contiguous):

* cell variables x[i][j], allocated first (row-major, id = i·23 + j + 1);
* for every row triple T=(a,b,c) and column j an auxiliary y[T][j] with the single clause `(¬x_aj ∨ ¬x_bj ∨ ¬x_cj ∨ y_Tj)`; then `AtMost(2, {y_Tj : j})` via sequential counter — i.e. λ_T ≤ 2 encoded through indicator variables rather than one 9-literal clause per (T, column triple);
* for every column j: `Exactly(d_j)` over its cells (sequential counter);
* symmetry breaking (**adding, not pruning**): `row i ≥_lex row i+1` for consecutive rows, and `col j ≥_lex col j+1` for consecutive columns of equal degree — a hand-written chain encoding with prefix-equality variables e_t;
* neighbour cut: `AtLeast(10)` on every row ("valid because deleting a row of degree at most nine would leave at least 104 ones in a 9×23 matrix, contradicting Z(9,23,3,3)=103").

Sizes: 10,853–10,893 variables, 22,296–22,454 clauses per profile (manifest). For comparison the *generic* one-clause-per-3×3 cell model in `models/` (regression artifact, not a proof) has 230 base variables, 212,520 forbidden-submatrix clauses and 371,894 clauses total for (10,23,113) after an exact-cardinality circuit.

### 3.4 Level 3: canonical row-stabilizer cube cover (profile 3^1 4^2 5^18 6^2 only)

`child_supports(prefix, degrees)` (`z10_23_certify.py`; re-implemented dependency-free in `src/.../cube_cover.py::_child_masks_cached`): columns are assigned left to right. The first column's support is forced to rows {0,…,d_0−1}. Given a prefix of fixed column supports, rows with identical membership pattern form *stabilizer cells* (sorted with pattern tuples descending); global row-lex forces the next support to be an initial segment of every cell; if the previous column has the same degree, the new support vector must be ≤ the previous one (equal-degree column lex); and any two prefix columns together with the new support may share ≤ 2 rows (K33 check on the prefix). Children = all such supports.

Two catalog generators:
* `frontier --depth k`: complete fixed-depth prefix cover, no solver calls ("makes no SAT claim");
* `cubes --conflicts 20000 --maximum-depth 10 [--depth-factor, --escalate-after, --maximum-conflicts]`: adaptive cube-and-conquer: `solver.conf_budget(budget); solve_limited(assumptions = prefix literals)`; UNSAT → record leaf; SAT → theorem false (raise); UNKNOWN → expand children. **[This conflict-budget probe is their only in-loop hardness estimator.]**

`z10_23_residual_refine.py`: for a distributed pass with timeouts, residual (timed-out) leaves are bisected *inside their immediate next column* by fixing selected cells (partial "literals"), keeping the global catalog complete. `cube_cover.verify_cube_catalog` re-derives the branching rule and checks that the (possibly partial) leaves induce a prefix trie containing every permitted child at every non-leaf node — rejecting missing, duplicate, overlapping, reordered, non-canonical, or full-assignment leaves. This is the **cover-completeness certificate**, checked in Python only.

Final cover for 3^1 4^2 5^18 6^2 (my count from `certificates/z10_23/3d1_4d2_5d18_6d2.cubes.jsonl`): 17,170 leaves; depths 4: 2,405, 5: 14,515, 6: 250; 8,306 full leaves, 7,218 leaves with 3 extra partial literals, 1,646 with 6. Each leaf is base CNF + unit clauses; solved with `cadical --unsat -q -P2`; replay `drat-trim → LRAT → lrat-check; projected DRAT → drat-trim`. Archive: 16,476,359,596 bytes xz.

### 3.5 Level 3′: row-symmetry orbit covers solved as exact MIPs (profiles 3^1 4^4 5^14 6^4 "B" and 3^1 4^3 5^16 6^3 "C")

Model (`search/z10_23_vipr/column_count_opb.py`): a *column-support-count* pseudo-Boolean model — for each of the C(10,4)+C(10,5)+C(10,6) supports of degree 4/5/6 two unary occurrence variables (multiplicity ≤ 2, since three equal columns would share a triple); the degree-3 column is fixed to rows {0,1,2} (canonical); triple-capacity and count constraints; optional row-lex symmetry constraints over transpositions in S_3 × S_7 (the stabilizer of the fixed support).
* Profile B is split by the *triple-deficit state*: the multiset of which 3 row triples carry deficit (s = 240 − 237 = 3), with the fixed triple (0,1,2) allowed at most once. Raw states 295,001; S_3×S_7 orbits 209 (`build_b_deficit_orbits.py` recomputes both numbers and aborts otherwise). Each leaf = base OPB + 239 unit literals.
* Profile C is split by the unordered multiset of the three degree-6 supports (admissibility: multiplicity ≤ 2, no common row triple, at most one contains the fixed triple). Raw states 950,250; orbits 236 (group order 30,240; canonical form = "three-bit row-membership pattern counts in the fixed and free row blocks, minimized over all six support labelings").

SCIP 10.0.3 exact mode (presolve off, conflict analysis off; separation also off in the profile-B residual run) produces VIPR certificates checked by unmodified `viprchk` (commit 30f2951d). The verifier parses the model *embedded in each VIPR file* and compares it coefficient-for-coefficient with the regenerated OPB, and rejects certificates containing `AggrRow_`, `lin weak`, `lin incomplete` (13 first-pass profile-B certificates were superseded for that reason). Certificate bytes: B 7,914,211,500; C 15,574,768.

---

## 4. Catalogue of pruning arguments

Notation: m rows, n columns, w ones, column degrees d_j, s := 2C(m,3) − Σ_j C(d_j,3) ≥ 0 (unused triple capacity, "slack"); n_d := #{j : d_j = d}. All statements are for K_{3,3}-free matrices; the general-(s,t) form is my inference where marked.

### P1. Triple capacity (Roman 1975 / KST-type)
**Statement.** Σ_j C(d_j,3) = Σ_T λ_T ≤ 2·C(m,3).
**Proof.** Double count pairs (T, j) with T ⊆ E_j; λ_T ≤ 2.
**Lean.** `sum_choose_columnDegree_eq_sum_tripleLoad`, `sum_choose_columnDegree_le` (Counting.lean, quoted §5.3). Also the *exact* identity `totalDeficit_add_sum_choose : totalDeficit A + Σ_c C(d_c,3) = 2·C(|R|,3)`.
**General form [inference].** Σ_j C(d_j, s) ≤ (t−1)·C(m, s) for no all-ones s×t (rows×columns); dual Σ_i C(r_i, t) ≤ (s−1)·C(n, t). This is exactly the "Kővári–Sós–Turán counting prune" our ZarPrune README lists as the next target.
**Kill predicate on profiles.** `Σ_j C(col j, 3) > 2·C(m,3)`. Trivially computable.

### P2. Penalty-line classification
**Statement.** For any integers α, β with p(d) := C(d,3) − α·d + β ≥ 0 on 0..m: Σ_j p(d_j) ≤ 2C(m,3) − α·w + β·n. Choosing α,β so that p vanishes at the two degrees around w/n gives a tiny budget and hence an explicit finite list of histograms.
**Lean.** `penalty_nat_identity` (by `decide` over `Fin (m+1)`), `incidence_penalty_identity`, `penalty_sum_eq_histogram`, then `classify_*_degree_profile` (omega over Presburger constraints in ≤ 10 variables after `rcases` on the exceptional counts). This is the part of the *cover-completeness obligation* that they discharge in Lean (for (9,23),(10,22),(12,23),(13,23); **not** for the 25 profiles of (10,23), which are enumerated only in Python).
**Kill predicate.** Same as P1 (it is P1 rearranged); its value is for *enumeration*, not for killing.

### P3. Vertex deletion / neighbour bound (row and column versions, and multi-column)
**Statement (row).** If some row has degree ≤ k then deleting it leaves an (m−1)×n K33-free matrix with ≥ w − k ones; so w − k ≤ Z(m−1,n). Averaging: k = ⌊w/m⌋ gives Z(m,n) ≤ ⌊m·Z(m−1,n)/(m−1)⌋. Symmetric for columns.
**Statement (profile / multi-column).** For any set S of columns, w − Σ_{j∈S} d_j ≤ Z(m, n−|S|). Used at (10,23,113): a column of degree ≤ 2 → 10×22 with ≥ 111 > 110 (5 profiles killed); two degree-3 columns → 10×21 with ≥ 107 > 106 (4 profiles killed).
**Hypotheses.** An *external* upper bound for the smaller instance (they keep it as an explicit Lean hypothesis; Tan's SAT bounds Z(9,21)≤96 and Z(11,18)≤101 have no importable certificate).
**Lean.** `deleteRow`, `deleteColumn` (via `Fin.succAbove`), `rowDegree_add_edgeCount_deleteRow`, `K33Free.deleteRow/deleteColumn`, `exists_rowDegree_le_of_edgeCount_lt_mul`, `UpperBound.addRow_of_average`, `UpperBound.addColumn_of_average` (Deletion.lean). Multi-column deletion is *not* formalized (only the arithmetic endpoints `z10_23_low_column_impossible`, `z10_23_two_degree_three_impossible` are, by omega).
**Kill predicate on profiles.** `w − (sum of the k smallest column sums) > Z_ub(m, n−k)` for any k (and rows dually).

### P4. Minimum row-degree cut (P3 as a per-row lower bound)
Every row has degree ≥ w − Z(m−1,n) (=10 at (10,23,113)). Used as a CNF constraint ("adding") but it is equally a sound *prune* on row profiles: kill if `min_i row i < w − Z_ub(m−1,n)`. Same hypotheses/Lean status as P3.

### P5. Marked-row deficit residues
**Statement.** For a row r define D_r := Σ_{T∋r} δ_T ≥ 0. Then
 D_r = 2·C(m−1,2) − Σ_{j : r∈E_j} C(d_j − 1, 2)  and  Σ_r D_r = 3s.
If every column degree d present has C(d−1,2) ≡ 0 (mod g) except for a few *exceptional* columns, then D_r ≡ 2C(m−1,2) − (exceptional contributions through r) (mod g), and D_r ≥ its least nonnegative residue. Summing the minimal residues over rows (minimized over how rows meet the exceptional columns) gives a lower bound on 3s; if it exceeds 3s the profile is impossible.
Instances: (9,23,104): g=3 (C(3,2)=3, C(4,2)=6), residue 2 for clean rows; the three profiles need Σ D_r ≥ 18, 15, 12 against 3s = 12, 3, 0. (13,23,145): g=5 (C(5,2)=10, C(6,2)=15), 132 ≡ 2 (mod 5): needs ≥ 26, 16, 10 vs 21, 6, 3. (10,23,113): kills 4^6 5^14 6^2 7^1 (min residue 6 > 3) and 3^1 4^3 5^17 6^1 7^1 (9 > 6). (12,23,135): g=10 with residues 7u_r + 4a_r + 5b_r.
**Lean.** `rowDeficit`, `rowUsed`, `sum_rowDeficit : Σ_r rowDeficit A r = 3 * totalDeficit A`, `markedTripleCount_eq` (triples through r in S ↔ C(|S|−1,2), a `card_bij'`), `rowUsed_eq_sum_choose`, `rowDeficit_add_rowUsed`, `rowDeficit_add_sum_choose` (all Counting.lean, Mathlib). Case use: `rowContribution_modEq_zero`, `rowContribution_modEq_exception` via `Nat.ModEq.sum_zero`, `Nat.sum_modEq_ite`, then `sum_lower_on_subset` + omega.
**Kill predicate on profiles.** Computable: enumerate row-membership patterns in the exceptional columns (2^k patterns, multiplicities summing to m, column sizes matching), minimize Σ residues; `filters.py::_side_ok` does a *relaxed* interval version with subset-sum bitsets (which degree-count vectors are reachable in the window [cap−3s, cap], and per-degree count consistency Σ_r k_d(r) = n_d·d), `tier2.py` the exact pattern-multiset version with K33 legality of the exceptional columns.

### P6. Pair-deficit residues
**Statement.** For a row pair P, D_P := Σ_{T⊇P} δ_T = 2(m−2) − Σ_{j : P⊆E_j} (d_j − 2) ≥ 0 and Σ_P D_P = 3s. Column weights are d−2 (so degree-5 columns vanish mod 3, degree-6 mod 4). Counting consistency: Σ_P (# degree-d columns through P) = n_d·C(d,2).
Instances: (10,22,111): 5^21 6^1 (30 pairs outside the six-column have D_P=1; around a row outside, degree-5 pair counts sum to 45 but must be ≡ 0 mod 4); 4^1 5^19 6^2 (pointwise identity r_P + z_P + 3x_P·C(z_P,2) = 1 + x_P + 3C(z_P,2), sum ≥ 21 > 18); 4^2 5^17 6^3 (row-type forcing + three-column budgets: LHS ≤ 75 < 84 ≤ RHS); 4^1 5^20 7^1 (symmetric difference ≥ 3 → ≥ 9 > 3). (12,23,136): 3a_P + 4b_P = 20 forces a_P = 0 for all P, but Σ_P a_P = 2·C(5,2) = 20. (10,23,113): 3^1 4^2 5^19 7^1: 1,577 exceptional-column configurations, 1,380 K33-legal, minimum pair-residue sum 39 > 18.
**Lean.** `pairDeficit`, `pairUsed`, `sum_pairDeficit`, `pairTripleCount_eq`, `pairUsed_eq_sum_sub_two`, `pairDeficit_add_pairUsed`, `pairDeficit_add_sum_sub_two`, `rowColumnDegreeCount`, `pairColumnDegreeCount`, `sum_rowColumnDegreeCount : Σ_r = d·degreeCount A d`, `sum_pairColumnDegreeCount`, `sum_through_row_le_sum_pairs` (PairCounting.lean, Mathlib). For (10,23) only the endpoint `z10_23_exceptional_profile_impossible (pairResidue ≤ 18) (39 ≤ pairResidue) : False` is in Lean; the 1,577-configuration enumeration is Python.

### P7. Three-column incidence budgets (transpose of K33-freeness)
**Statement.** Three distinct columns share ≤ 2 rows (`commonRowCount_le_two`) and ≤ 1 row pair (`commonPairCount_le_one`). Hence, with a_r := # degree-d columns through row r: Σ_r C(a_r,3) ≤ 2·C(n_d,3), Σ_r C(a_r,2)·b_r ≤ 2·C(n_d,2)·n_{d'} (`sum_choose_rowColumnDegreeCount_le`, `sum_choose_two_mul_rowColumnDegreeCount_le`), and pair versions with budget 1 (`sum_choose_pairColumnDegreeCount_le`, `sum_choose_two_mul_pairColumnDegreeCount_le`) — TripleIncidence.lean.
Used to bound row-type multisets in (10,22) case 3 and (12,23) cases 1, 3, 5.

### P8. Row-type enumeration (finite kernel)
For a profile with few exceptional columns, enumerate the vector of row types (a_r, b_r, …) subject to Σ_r a_r = n_d·d and P7 budgets; minimize the residue sum. (12,23,135) 5^4 6^18 7^1: minimum 25 > 15 — in Lean as `case_three_row_type_minimum`, "a ten-variable Lean row-type kernel … proved with omega". (10,22,111) 4^2 5^17 6^3 was originally a 77-orbit × 22,155-multiset Python sweep, later replaced by the incidence identity argument in Lean.

### P9. Profile interval filter (`search/filters.py::profile_ok`) — necessary conditions only
Uses P1, P5, P6 in relaxed form: reachable-sum bitsets over k_d ∈ [0, n_d] for row weights C(d−1,2) (cap 2C(m−1,2), m points) and pair weights d−2 (cap 2(m−2), C(m,2) points); requires m·min_def ≤ 3s ≤ m·max_def and per-degree count consistency. General in (m,n,T). My run (§10): kills all 3 profiles at (9,23,104), the single profile at (12,23,136) and (11,21,117), all 3 at (13,23,145); kills 1 of 25 at (10,23,113), 0 of 4 at (10,22,111), 0 of 5 at (12,23,135), 0 of 6 at (11,23,124).

### P10. Configuration-level residue filter (`search/tier2.py::tier2_profile_dead`)
Exact enumeration of row-membership pattern multisets over ≤ 5 exceptional columns with K33 legality, row residues mod g_row = gcd of base C(d−1,2), pair residues mod g_pair = gcd of base (d−2), requiring Σ residues ≤ 3s and ≡ 3s (mod g). "Found the five profile kills behind Z(12,23)≤134."

### P11. Davies–Gill–Horsley degree-count LP (diagnostic, does *not* kill)
`search/lp_dgh.py` reproduces DGH's table exactly-rationally. At (9,23,104) the DGH relaxation has optimum 314/3 = 104⅔ at n_4 = 31/3, n_5 = 38/3 and all three integer profiles satisfy its constraints: "Their relaxation therefore cannot see the last contradiction. The missing information is … an overlap condition recording how the nearly saturated row triples meet each individual row." **Important calibration for us: LP-strength degree constraints are strictly weaker than the residue prunes.**

### Not a prune: symmetry breaking and canonical cubes
Double-lex on rows / equal-degree columns, the canonical first column {0..d_0−1}, and the row-stabilizer child rule are all *adding* steps justified by permutation witnesses; they live in the CNF/OPB and in the Python cover checker, never in Lean. This matches our ZarPrune `notDescending_unsound` philosophy exactly — and shows the cost: cover completeness then needs its own (here unverified) certificate.

### Which prune killed what

| Cell, w | profiles | P3 deletion | P5 row residue | P6 pair residue / P8 | SAT/MIP |
|---|---:|---:|---:|---:|---:|
| (9,23), 104 | 3 | – | 3 | – | 0 |
| (10,22), 111 | 4 | – | 1 (4^1 5^20 7^1) | 3 | 0 |
| (10,23), 113 | 25 | 9 (5 low-degree col, 4 double-3) | 2 | 1 | 13 (10 direct, 1 cube, 2 VIPR) |
| (12,23), 136 | 1 | – | – | 1 | 0 |
| (12,23), 135 | 5 | – | 2 (5^3 6^20; 5^5 6^16 7^2 with P7) | 3 (incl. row-type kernel) | 0 |
| (13,23), 145 | 3 | – | 3 | – | 0 |
| (10,21),(11,19),(11,20),(11,23) | – | whole cell by P3 from a neighbour | – | – | 0 |

---

## 5. The Lean layer, precisely

### 5.1 Toolchain and shape
* `lean-toolchain`: `leanprover/lean4:v4.29.0`; `lakefile.toml` requires **Mathlib** `v4.29.0` (rev `8a178386…`), plus transitive batteries/aesop/Qq/plausible/importGraph/ProofWidgets/Cli. Three `lean_lib`s: `ZarankiewiczZ923` (legacy arithmetic kernel for (9,23), **no Mathlib**, only `Lean.Elab.Tactic.Omega`), `ZarankiewiczFiniteClosures` (legacy arithmetic kernels for the other cells, no Mathlib), `Zarankiewicz` (the Mathlib matrix formalization; default target).
* 17 `.lean` files, 4,896 lines: Basic 75, Counting 444, PairCounting 403, TripleIncidence 400, Deletion 172, Witnesses 120, Exact/Z9_23 342, Exact/Z10_22 1,184, Exact/Z12_23 836, Exact/DeletionClosures 63, Bounds/Z13_23 293, ArithmeticKernels 297, ArithmeticKernel 166, AxiomAudit 76.
* Claims: "no `sorry`, `admit`, project `axiom`, or `native_decide`". I grepped the tree: the only hits for those words are in doc comments (Witnesses.lean line 7, ArithmeticKernel.lean line 13). `AxiomAudit.lean` `#print axioms` for 57 theorems; recorded output `audit/lean_axioms.txt`: end-to-end theorems depend on `[propext, Classical.choice, Quot.sound]` (Mathlib/omega); the legacy `decide`/omega kernels depend on `[propext, Quot.sound]` or nothing.
* CI: `leanprover/lean-action@v1` with `use-mathlib-cache: true`, `leanchecker: true`, `timeout-minutes: 20`; Python `make verify` job `timeout-minutes: 180`. No per-theorem elaboration times are recorded anywhere. **I did not build the Lean project** (Mathlib cache download; our local elan has only 4.34.0).

### 5.2 Lean statements — the reusable library (verbatim signatures)

```lean
-- Counting.lean (variables {R C} [Fintype R] [LinearOrder R] [Fintype C] [DecidableEq C])
theorem tripleLoad_le_two (A) (hA : K33Free A) (T) (hT : T ∈ rowTriples R) : tripleLoad A T ≤ 2
theorem sum_choose_columnDegree_eq_sum_tripleLoad (A) :
    (∑ c, Nat.choose (columnDegree A c) 3) = ∑ T ∈ rowTriples R, tripleLoad A T
theorem sum_choose_columnDegree_le (A) (hA : K33Free A) :
    (∑ c, Nat.choose (columnDegree A c) 3) ≤ 2 * Nat.choose (Fintype.card R) 3
def degreeCount (A) (d : Nat) : Nat := #(Finset.univ.filter fun c => columnDegree A c = d)
theorem sum_function_mul_degreeCount (A) (f : Nat → Nat) :
    (∑ d ∈ range (Fintype.card R + 1), f d * degreeCount A d) = ∑ c, f (columnDegree A c)
theorem sum_degreeCount (A) : (∑ d ∈ range (Fintype.card R + 1), degreeCount A d) = Fintype.card C
theorem sum_degree_mul_degreeCount (A) : (∑ d ∈ range (Fintype.card R + 1), d * degreeCount A d) = edgeCount A
def tripleDeficit (A) (T) : Nat := 2 - tripleLoad A T
def totalDeficit (A) : Nat := ∑ T ∈ rowTriples R, tripleDeficit A T
theorem totalDeficit_add_sum_choose (A) (hA : K33Free A) :
    totalDeficit A + (∑ c, Nat.choose (columnDegree A c) 3) = 2 * Nat.choose (Fintype.card R) 3
def rowDeficit (A) (r : R) : Nat := ∑ T ∈ rowTriples R, if r ∈ T then tripleDeficit A T else 0
theorem sum_rowDeficit (A) : (∑ r, rowDeficit A r) = 3 * totalDeficit A
theorem markedTripleCount_eq (S : Finset R) (r : R) :
    markedTripleCount S r = if r ∈ S then Nat.choose (#S - 1) 2 else 0
theorem rowDeficit_add_sum_choose (A) (hA : K33Free A) (r : R) :
    rowDeficit A r + (∑ c, if A r c = true then Nat.choose (columnDegree A c - 1) 2 else 0)
      = 2 * Nat.choose (Fintype.card R - 1) 2
theorem K33Free.mono {A B} (hA : K33Free A) (hBA : MatrixLE B A) : K33Free B
theorem exists_submatrix_of_edgeCount_le (A) (e : Nat) (he : e ≤ edgeCount A) :
    ∃ B, MatrixLE B A ∧ edgeCount B = e
theorem sum_lower_on_subset (f : R → Nat) (E : Finset R) (inside outside : Nat)
    (hin : ∀ r ∈ E, inside ≤ f r) (hout : ∀ r ∈ univ, r ∉ E → outside ≤ f r) :
    #E * inside + (Fintype.card R - #E) * outside ≤ ∑ r, f r

-- PairCounting.lean
def pairDeficit (A) (P : Finset R) : Nat := ∑ T ∈ rowTriples R, if P ⊆ T then tripleDeficit A T else 0
theorem sum_pairDeficit (A) : (∑ P ∈ rowPairs R, pairDeficit A P) = 3 * totalDeficit A
theorem pairDeficit_add_sum_sub_two (A) (hA : K33Free A) (P) (hP : P ∈ rowPairs R) :
    pairDeficit A P + (∑ c, if P ⊆ columnSupport A c then columnDegree A c - 2 else 0)
      = 2 * (Fintype.card R - 2)
theorem sum_rowColumnDegreeCount (A) (d : Nat) : (∑ r, rowColumnDegreeCount A r d) = d * degreeCount A d

-- TripleIncidence.lean
theorem commonRowCount_le_two (A) (hA : K33Free A) (a b c : C) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    commonRowCount A a b c ≤ 2
theorem commonPairCount_le_one (A) (hA : K33Free A) (a b c : C) (hab hac hbc) : commonPairCount A a b c ≤ 1

-- Deletion.lean
theorem exists_rowDegree_le_of_edgeCount_lt_mul (A) (k : Nat) (hweight : edgeCount A < Fintype.card R * (k + 1)) :
    ∃ r, rowDegree A r ≤ k
theorem UpperBound.addRow_of_average {m n e k : Nat} (hbase : UpperBound m n e)
    (haverage : e + k + 1 < (m + 1) * (k + 1)) : UpperBound (m + 1) n (e + k)
theorem UpperBound.addColumn_of_average {m n e k : Nat} (hbase : UpperBound m n e)
    (haverage : e + k + 1 < (n + 1) * (k + 1)) : UpperBound m (n + 1) (e + k)
```

### 5.3 Lean statements — the case theorems

```lean
-- Exact/Z9_23.lean
theorem degree_profile_of_104 (A : BinaryMatrix (Fin 9) (Fin 23)) (hA : K33Free A) (hweight : edgeCount A = 104) :
    (degreeCount A 0 = 0 ∧ … ∧ degreeCount A 4 = 11 ∧ degreeCount A 5 = 12 ∧ …) ∨ (… n3 = 1 ∧ n4 = 9 ∧ n5 = 13 …) ∨ (… n4 = 12 ∧ n5 = 10 ∧ n6 = 1 …)
theorem row_formula (A) (hA : K33Free A) (r : Fin 9) : rowDeficit A r + rowContribution A r = 56
theorem no_matrix_at_104 (A) (hA : K33Free A) (hweight : edgeCount A = 104) : False
theorem upper_bound : UpperBound 9 23 103
theorem exact_value : Zarankiewicz.Exact 9 23 103

-- Exact/Z10_22.lean:  profile_at_111, row_formula (= 72), case_one_impossible … case_four_impossible,
theorem no_matrix_at_111 (A : BinaryMatrix (Fin 10) (Fin 22)) (hA) (hweight : edgeCount A = 111) : False
theorem upper_bound : UpperBound 10 22 110
theorem exact_value : Zarankiewicz.Exact 10 22 110

-- Exact/Z12_23.lean:  profile_at_136, no_matrix_at_136, profile_at_135, case_three_row_type_minimum, no_matrix_at_135,
theorem upper_bound : UpperBound 12 23 134
theorem exact_value : Zarankiewicz.Exact 12 23 134

-- Bounds/Z13_23.lean
theorem profile_data (A : BinaryMatrix (Fin 13) (Fin 23)) (hA : K33Free A) (hweight : edgeCount A = 145) :
    (n0 = 0 ∧ n1 = 0 ∧ n2 = 0 ∧ n3 = 0 ∧ n4 = 0 ∧ n9 = 0 ∧ … ∧ n13 = 0) ∧
    ((n5 = 0 ∧ n6 = 16 ∧ n7 = 7 ∧ n8 = 0) ∨ (n5 = 1 ∧ n6 = 14 ∧ n7 = 8 ∧ n8 = 0) ∨ (n5 = 0 ∧ n6 = 17 ∧ n7 = 5 ∧ n8 = 1))
theorem row_formula (A) (hA) (r : Fin 13) : rowDeficit A r + rowContribution A r = 132
theorem no_matrix_at_145 (A) (hA : K33Free A) (hweight : edgeCount A = 145) : False
theorem upper_bound : UpperBound 13 23 144

-- Exact/DeletionClosures.lean (the *conditional* closures; premises are explicit arguments)
theorem z10_21_upper_bound (hbase : UpperBound 9 21 96) : UpperBound 10 21 106 :=
  hbase.addRow_of_average (k := 10) (by omega)
theorem z10_21_exact (hbase : UpperBound 9 21 96) : Zarankiewicz.Exact 10 21 106
theorem z11_19_upper_bound (hbase : UpperBound 11 18 101) : UpperBound 11 19 106 :=
  hbase.addColumn_of_average (k := 5) (by omega)
theorem z11_19_exact (hbase : UpperBound 11 18 101) : Zarankiewicz.Exact 11 19 106
theorem z11_20_upper_bound (hbase : UpperBound 11 18 101) : UpperBound 11 20 111
theorem z11_20_exact (hbase : UpperBound 11 18 101) : Zarankiewicz.Exact 11 20 111
theorem z10_23_exact_of_upper (hupper : UpperBound 10 23 112) : Zarankiewicz.Exact 10 23 112
theorem z11_23_upper_bound (hbase : UpperBound 10 23 112) : UpperBound 11 23 123 :=
  hbase.addRow_of_average (k := 11) (by omega)
theorem z11_23_exact (hbase : UpperBound 10 23 112) : Zarankiewicz.Exact 11 23 123
```

The file header states the policy: "The historical starting bounds Z(9,21) ≤ 96 and Z(11,18) ≤ 101, and the externally certified bound Z(10,23) ≤ 112, remain explicit hypotheses. They cannot honestly be erased from a certificate-free theorem statement."

### 5.4 Lean statements — the legacy "arithmetic kernels" (Mathlib-free, omega/decide)

These check *only* the integer endpoints of the human proofs; the comment in `ArithmeticKernels.lean` is explicit: "Lean checks reported finite minima but does not re-run row-symmetry or row-type enumeration and does not replay the completed external Z(10,23) proof certificates." Representative statements:

```lean
theorem classify_degree_profile (n0 … n9 : Nat) (hcolumns : n0 + … + n9 = 23)
    (hweight : n1 + 2*n2 + … + 9*n9 = 104) (hpenalty : 20*n0 + 14*n1 + 8*n2 + 3*n3 + 4*n6 + 13*n7 + 28*n8 + 50*n9 ≤ 4) :
    (… n4 = 11 ∧ n5 = 12 …) ∨ (… n3 = 1 ∧ n4 = 9 ∧ n5 = 13 …) ∨ (… n4 = 12 ∧ n5 = 10 ∧ n6 = 1 …)
theorem balanced_deficits_impossible (d0 … d8 : Nat) (hsum : d0 + … + d8 = 12) (h0 : d0 % 3 = 2) … (h8 : d8 % 3 = 2) : False
theorem z10_23_low_column_impossible (columnDegree : Nat) (hdegree : columnDegree ≤ 2) (hremaining : 113 ≤ 110 + columnDegree) : False
theorem z10_23_two_six_residue_minima : List.all [18, 15, 12, 9, 6] (fun minimum => 3 < minimum) = true := by decide
theorem z10_23_exceptional_profile_impossible (pairResidue : Nat) (hbudget : pairResidue ≤ 18) (henumeratedMinimum : 39 ≤ pairResidue) : False
theorem z10_22_case_b_orbit_minima : List.all [21, 21, 21, 33, 48] (fun minimum => 18 < minimum) = true := by decide
theorem z12_23_case_three_minimum : 15 < 25 := by decide
```
(For (9,23), (10,22), (12,23), (13,23) these kernels were later *superseded* by the end-to-end matrix proofs; for (10,23) they are all that exists in Lean.)

### 5.5 What is NOT verified in Lean (trusted parts)

1. **SAT and MIP certificates.** Lean never reads DRAT/LRAT/VIPR. Trusted: CaDiCaL 3.0.0 output → `drat-trim` (proof-tools commit `2e3b2dc0`) → `lrat-check`; SCIP 10.0.3 exact → `viprchk` (vipr commit `30f2951d`). Their own words: "Lean does not replay DRAT/LRAT or VIPR and does not contain a certificate-free proof of Z(10,23)=112."
2. **Encoding correctness.** That UNSAT of the per-profile CNF/OPB implies "no K33-free 10×23 matrix with that column histogram and 113 ones" — including the soundness of double-lex symmetry breaking, the canonical first column, the min-row-degree-10 cut, the y_Tj indicator encoding, sequential counters — is checked only by Python unit tests ("sequential threshold circuit was checked by an independent DPLL implementation on every assignment of up to six base variables"; a known small case Z(3,4,3,3)=10; thirty seeded 5×6 matrices). The `refuted`-side "encoding-correctness theorem" our ZarPrune README anticipates does not exist here either.
3. **Cover completeness.** The 25-profile enumeration at (10,23,113) (Python `enumerate_degree_profiles`, cross-checked by a second Python implementation), the 17,170-leaf cube cover (`cube_cover.verify_cube_catalog`), and both orbit censuses (295,001/209 and 950,250/236) are Python-only.
4. **Finite enumerations inside arithmetic kills** for (10,23): the 1,577/1,380/39 exceptional-column enumeration; Lean only checks `39 > 18`.
5. **External bounds** Z(9,21) ≤ 96 and Z(11,18) ≤ 101 (Tan's SAT classification, no certificate available) — explicit Lean hypotheses.
6. **Witness CSV ↔ Lean bitmask equivalence** is asserted by the repo's own checks; the Lean lower bounds are self-contained (bitmask literals + `decide`), so this is not a soundness gap for the Lean theorems, only a provenance link.
7. **Hash-binding infrastructure** (SHA-256 sidecars, release parts, tar/xz streaming) — Python.

Their trust-boundary summary is honest and consistent across README, METHODS §9, ADVERSARIAL_AUDIT §7/§10, REPRODUCIBILITY, `analysis/result_status.json` ("The Z(10,23) upper bound is computer-assisted … it is not a certificate-free Lean theorem").

---

## 6. Runtimes and resources (what is and is not recorded)

* **No solver wall-clock or CPU times are published.** `audit/README.md`: "Recorded reports omit temporary paths, timing data, and host identifiers." The manifests contain no timing keys (I grepped `z10_23_sat.json`: only `mtime`). `z10_23_certify.py` writes `elapsed_seconds` into per-run metadata, but none of those metadata files is checked in.
* What *is* recorded (docs/AWS_Z10_23_RUN.md): the (10,23) proof production ran on AWS EKS `us-east-2`, a dedicated On-Demand node group "capped at six `r7i.2xlarge` instances: 48 vCPUs total"; base run id dated 2026-07-06T19:01Z, final adaptive stage `r9`, VIPR sweeps dated 2026-07-15 ≈00:48–02:20Z; so roughly 9 days of a 48-vCPU cluster for the thirteen (10,23) profiles **[inference on duration from run ids; utilization unknown]**. Cube solver options `--unsat -q -P2`; adaptive cube default conflict budget 20,000, max depth 10.
* Proof sizes are the only effort proxy (compressed xz DRAT bytes for the ten direct profiles: 1.0 MB … 415 MB; the cube profile 16.5 GB; VIPR-B 7.9 GB, VIPR-C 15.6 MB). Table in §8.1.
* Local verification: `make verify` (standard library only) is run in CI on Python 3.9 and 3.14 with a 180-minute limit (it regenerates both orbit covers and 445 OPB leaves); the Lean job has a 20-minute limit with Mathlib cache. Full external replay needs the 25.3 GB release download plus built `drat-trim`, `lrat-check`, `viprchk`.
* Everything in `docs/` was generated 2026-07-04 → 07-15, i.e. the eight closures took about eleven calendar days end to end with three LLM systems.

---

## 7. File layout (what to look at)

```
README.md, CITATION.cff, CONTRIBUTING.md, references.bib, Makefile, pyproject.toml, artifacts.sha256
docs/    PROOF.md (9,23 human proof), PROOF_Z10_21/22/23.md, PROOF_Z11_19/20/23.md, PROOF_Z12_23.md, BOUND_Z13_23.md,
         METHODS.md, LITERATURE_REVIEW.md, NEW_BOUNDS.md (12,23 & 13,23 theorems + propagated table), EXTENDED_RESULTS.md,
         ADVERSARIAL_AUDIT.md, REPRODUCIBILITY.md, SAT_Z10_23_STATUS.md, AWS_Z10_23_RUN.md
lean/    lakefile.toml (Mathlib v4.29.0), AxiomAudit.lean,
         Zarankiewicz/{Basic,Counting,PairCounting,TripleIncidence,Deletion,Witnesses}.lean,
         Zarankiewicz/Exact/{Z9_23,Z10_22,Z12_23,DeletionClosures}.lean, Zarankiewicz/Bounds/Z13_23.lean,
         ZarankiewiczZ923/ArithmeticKernel.lean, ZarankiewiczFiniteClosures/ArithmeticKernels.lean
src/finite_zarankiewicz_closures/  matrix.py (witness check), encodings.py (generic cell CNF / column-type LP),
         extended.py (profile enumeration, arithmetic front ends, propagated table), case_certificates.py,
         cube_cover.py (cover-completeness checker), sat_certificate.py, vipr_certificate.py, boundary.py, certificate.py
search/  filters.py (P9), tier2.py (P10), sat_tool.py (per-profile CNF), z10_23_profiles.py (13 SAT profiles),
         z10_23_certify.py (frontier/cubes/direct), z10_23_cube_certify.py, z10_23_residual_refine.py, z10_23_cube_finalize.py,
         z10_23_vipr/{column_count_opb,build_b_deficit_orbits,build_c_degree6_orbits}.py, lp_dgh.py, drive.py, zsearch.c, zsearch2.c
certificates/  z*_*.json (8 exact + z13_23_upper_144), degree_deficit.json, z10_23_sat.json (master manifest),
         z10_23/{*.drat.xz[.part-NN], cubes.jsonl, cube-proof-index.jsonl, *.release.json, vipr/*}
models/  cells_MxN_exact_W.cnf, column_types_MxN_exact_W.lp (regression only), z10_23/*.cnf (13 profile CNFs)
data/    eight witness CSVs;  analysis/ result_status.json, extended_results.json, new_bounds.json, sat_cross_check.json, dgh_boundary.json
audit/   lean_axioms.txt, certificate_replay.json, model_validation.json;  tests/ 13 unittest modules (incl. mutation tests)
```

---

## 8. Numbers and tables

### 8.1 The thirteen (10,23,113) SAT/MIP profiles (from `certificates/z10_23_sat.json`; slack and #exceptional computed by me)

| profile | strategy | vars | clauses | slack s | exc. cols (deg∉{5,6}) | compressed proof bytes |
|---|---|---:|---:|---:|---:|---:|
| 4^2 5^21 | direct CaDiCaL | 10,893 | 22,454 | 22 | 2 | 340,197,208 |
| 4^3 5^19 6^1 | direct | 10,879 | 22,387 | 18 | 3 | 94,127,640 |
| 4^4 5^17 6^2 | direct | 10,875 | 22,379 | 14 | 4 | 12,226,620 |
| 4^4 5^18 7^1 | direct | 10,871 | 22,371 | 9 | 5 | 4,124,340 |
| 4^5 5^15 6^3 | direct | 10,871 | 22,371 | 10 | 5 | 8,491,188 |
| 4^5 5^16 6^1 7^1 | direct | 10,857 | 22,304 | 5 | 6 | 1,030,392 |
| 4^6 5^13 6^4 | direct | 10,867 | 22,363 | 6 | 6 | 12,338,016 |
| 4^7 5^11 6^5 | direct | 10,863 | 22,355 | 2 | 7 | 9,627,152 |
| 3^1 5^22 | direct | 10,889 | 22,446 | 19 | 1 | 41,968,656 |
| 3^1 4^1 5^20 6^1 | direct | 10,865 | 22,320 | 15 | 2 | 415,038,644 |
| 3^1 4^2 5^18 6^2 | row-stabilizer cube cover, 17,170 leaves | 10,861 | 22,312 | 11 | 3 | 16,476,359,596 |
| 3^1 4^3 5^16 6^3 | SCIP/VIPR, 236 orbits / 950,250 states | 10,857 | 22,304 | 7 | 4 | 15,759,360 |
| 3^1 4^4 5^14 6^4 | SCIP/VIPR, 209 orbits / 295,001 states | 10,853 | 22,296 | 3 | 5 | 7,914,383,360 |

Observation **[inference]**: hardness is not monotone in slack. The two largest *direct* proofs have the largest slacks (22, 15); the three profiles that defeated direct CaDiCaL all contain the single degree-3 column together with 2–4 degree-4 and 2–4 degree-6 columns (balanced exceptional blocks on both sides of the 5s). The VIPR-B profile has slack only 3 yet needed 7.9 GB of exact-MIP certificates.

### 8.2 The twelve arithmetic kills at (10,23,113) (reproduced by me with `extended.z10_23_profile_report()`)
* column of degree ≤ 2 → Z(10,22)=110: 2^1 5^21 6^1, 2^1 4^1 5^19 6^2, 1^1 5^20 6^2, 2^1 4^2 5^17 6^3, 2^1 4^1 5^20 7^1
* two degree-3 columns → Z(10,21)=106: 3^2 5^19 6^2, 3^2 4^1 5^17 6^3, 3^2 4^2 5^15 6^4, 3^2 5^20 7^1
* row residues: 4^6 5^14 6^2 7^1 (min 6 > 3s=3), 3^1 4^3 5^17 6^1 7^1 (min 9 > 6)
* pair residues: 3^1 4^2 5^19 7^1 (1,577 configs, 1,380 legal, min 39 > 18)

### 8.3 Propagated bounds relative to our Figure 2 (docs/NEW_BOUNDS.md §5; mechanisms: deletion ⌊kB/(k−1)⌋ and two-one-line extension Z(m,n) ≥ Z(m−1,n)+2)
(10,23) 112–115 → 112; (11,23) 118–125 → 123; (12,17) 102–108 → 102–104; (12,18) 108–113 → 108–110; (12,19) 110–118 → 110–115; (12,20) 113–122 → 113–121; (12,21) 116–127 → 118–126; (12,23) 125–136 → 134; (13,17) 106–116 → 109–112; (13,18) 115–121 → 115–118; (13,19) 114–125 → 117–124; (13,23) 135–145 → 139–144; (14,17) 118–124 → 118–120; (14,18) 124–129 → 124–127; (14,19) 121–135 → 126–133; (14,20) → 128–140; (14,22) → 139–150; (14,23) → 141–155; (15,17) 125–132 → 125–128; (15,18) 132–138 → 132–135; (15,19) 132–143 → 134–142; (15,21) → 140–154; (16,17) 128–141 → 130–136; (16,18) 130–146 → 134–144; (16,19) 132–152 → 136–151; (16,21) → 148–164; (16,22) → 150–169. Machine-readable: `analysis/new_bounds.json` (scope 8≤m≤16, 16≤n≤23, per-cell provenance).

### 8.4 Generic models (regression artifacts): cell CNF for (9,23,104): 207 vars, 148,764 forbidden clauses, 277,931 clauses; (10,23,113): 230 / 212,520 / 371,894; (12,23,135): 276 / 389,620 / 618,940. Column-type LP: one integer variable per support (2^m), one constraint per row triple.

---

## 9. Relevance to our design (ZarPrune + OpenEvolve)

1. **The branching unit.** Column-degree histograms alone, with rows handled by symmetry breaking and the neighbour cut, already give small branch counts (3–25 at the cells they closed). Tan-style (row partition, column partition) pairs are finer and would multiply SAT instances but make each easier. Our `Profile` (both vectors) subsumes both; the harness's *enumeration* choice (histograms vs. partition pairs) is a knob we should expose, and all dfield prunes are histogram-only functions so they work at either granularity.

2. **Every dfield prune is a `kill : Profile → Bool`.** P1 (capacity), P3/P4 (deletion with a bound table), P5/P6 (residue minima over exceptional-column patterns), P7/P8 (row-type budgets) are all computable from the column sums, m, n, w, and known neighbour bounds. `search/filters.py::profile_ok(m,n,T,counts)` and `tier2.py` are drop-in, general-(m,n,T) baseline kill predicates (necessary conditions) that we can port and use as the *unverified* upper reference for what an evolved prune should achieve; their Lean counterparts (Counting/PairCounting/TripleIncidence) tell us the soundness proofs are ~1,250 lines with Mathlib.

3. **Proof architecture that is LLM-friendly.** In every case file the structure is: (a) instantiate fixed library identities (`sum_choose_columnDegree_le`, `rowDeficit_add_sum_choose`, `pairDeficit_add_sum_sub_two`, `sum_rowDeficit`, `sum_lower_on_subset`); (b) compute a few closed-form constants by `decide`; (c) finish with `omega` on a Presburger goal, after `rcases` on bounded exceptional counts. The *evolvable* content is (b)+(c) plus the choice of which rows/pairs to mark and which modulus g to use. For our reward design this suggests grading candidates on whether the residue argument closes with omega given the library, rather than on free-form Lean.

4. **Mathlib or not.** They pay Mathlib (Finset.powersetCard, card_bij', sum_comm, Nat.ModEq) for the subset-counting layer; ZarPrune is Mathlib-free for inner-loop speed. Options: (i) hand-roll a minimal `chooseFin`/"triples through r" layer over `sumFin` once (fixed, reviewed) and expose the four identities as axiom-free lemmas; (ii) accept Mathlib only in a separate "library" package compiled once, with evolved prunes importing its `.olean`s (their `lake build +Module:olean` workflow shows per-module iteration is practical once the cache is warm). Either way the residue prunes are the first real target beyond `baseline`.

5. **The two seams of `upper_bound_of_cover`.** dfield discharges `cover` in Lean only for the tiny classifications (≤ 5 profiles, via omega) and never discharges `refuted` in Lean. For us: (a) a Lean `classify_profiles` theorem by omega is feasible when the enumeration is small; for larger tables we need a checked enumerator (e.g. a `decide`-able reflection over a bounded search) — open; (b) `refuted` could go through Lean core's LRAT checker (as used by `bv_decide`; it relies on `Lean.ofReduceBool`), which would be a genuine step beyond dfield, but the *encoding-correctness* theorem (CNF ⇒ matrix semantics, including the "adding" clauses) remains unwritten by anyone.

6. **Difficulty and reward.** Their only in-loop difficulty estimator is a CaDiCaL conflict budget (20k conflicts per cube, escalating with depth); their only recorded post-hoc measure is proof size. Number-of-profiles-killed is a bad scalar reward: 9 of the 12 arithmetic kills at (10,23) came from *deletion with neighbour bounds* (trivial), while the three hard survivors needed 24 GB of certificates. Weight kills by an estimated SAT cost (e.g. conflict-budget probe outcome, or a learned proxy from slack/exceptional-column mix) and reward *new* kills relative to the P9/P10 baselines.

7. **Cells to target.** With the frontier propagated (§8.3), natural thesis targets are the cells with few capacity-feasible profiles at the next excluded weight where residues almost close: (11,23,124) has 6 profiles (closed by deletion here, so a good *validation* target); (12,17..21) and (13..16, 17..23) are open. `filters.max_T_with_survivor` gives a quick scan.

8. **What to reuse verbatim (MIT).** `enumerate_degree_profiles`, `filters.py`, `tier2.py`, `sat_tool.build`, `cube_cover.verify_cube_catalog`, and the Lean `Counting.lean` identities (if we go Mathlib).

---

## 10. What I reproduced locally (2026-09-21)

* Cloned HEAD (`--depth 1`; 64 commits per GitHub API). `PYTHONPATH=src python3` → `extended.z10_23_profile_report()`: status `ARITHMETIC_FRONT_END_VERIFIED`, 25 profiles, partition 5/4/3/13, residue minima 6 and 9, exceptional enumeration {1577, 1380, 39}. `z10_22_certificate_report()` and `z13_23_upper_report()` return `VERIFIED` with the numbers quoted in §4.
* `python3 search/filters.py`: the three (9,23,104) profiles KILLED; the four (10,22,111) profiles survive (as the file's own comment predicts); (12,23,136) 5^2 6^21 KILLED. Profile counts at nine (m,n,T) as listed in §3.2/§4 P9.
* Cube catalog census (17,170 leaves, depth/partial distributions in §3.4) from the checked-in `cubes.jsonl`.
* Grep of `lean/` for `sorry|native_decide|axiom|admit`: only doc-comment hits. **Not done:** `lake build` (needs Mathlib v4.29.0 cache; local elan has 4.34.0 only), `make verify`, any certificate replay (25 GB). `search/tier2.py` self-test was started but is slow (its `__main__` scans five instances); no result recorded here.

---

## 11. Open questions

1. Has anyone independently replayed the 25 GB (10,23) certificate family or built the Lean project? The repository itself says no (0 stars, "awaiting independent expert review").
2. Encoding correctness: the per-profile CNF's symmetry breaking (double-lex + canonical first column + row-stabilizer children) and the y_Tj indicator encoding are unverified. Is the row-lex constraint compatible with the "rare degree blocks first" column order and the equal-degree column lex simultaneously (a joint canonical form)? They rely on it for both the CNF and the cover; no proof is given.
3. Why did 3^1 4^k 5^* 6^k (k=2,3,4) resist direct CaDiCaL while 4^7 5^11 6^5 (slack 2) did not? No analysis is offered; worth studying as a difficulty-signal question.
4. Does the marked-row/pair residue machinery generalise to K_{s,t} with (s,t) ≠ (3,3)? **[inference]** the weights become C(d−1, s−1) (rows) and C(d−2, s−2) (pairs) with capacity (t−1)·C(m−1, s−1) and (t−1)·C(m−2, s−2); nothing in the repo addresses this.
5. Could Lean discharge cover-completeness for larger profile tables (25 at (10,23)) by reflection rather than by hand-cased omega?
6. Is accepting `Lean.ofReduceBool` (LRAT via `bv_decide` machinery) an acceptable trust level for the thesis's `refuted` seam, versus their external `lrat-check`?
7. No runtime data: how long did each direct profile take, and how many cubes timed out before residual refinement? Not recoverable from the repository.

---

## 12. Citations

* dfield, *finite-zarankiewicz-closures* (GitHub, MIT), v2.0.0, 2026-07-14, https://github.com/dfield/finite-zarankiewicz-closures — files cited above by path.
* Tan, J. J., *An attack on Zarankiewicz's problem through SAT solving*, arXiv:2203.02283v2 (2022); companion code https://github.com/Parcly-Taxel/Kyoto.
* Roman, S., *A problem of Zarankiewicz*, JCTA 18 (1975) 187–198 (the triple-capacity bound).
* Davies, Gill, Horsley, *Improved upper bounds on Zarankiewicz numbers*, Discrete Math 349 (2026), arXiv:2411.18842.
* Bhan, Nobili, Langer, *New Bounds for Zarankiewicz Numbers via Reinforced LLM Evolutionary Search*, arXiv:2605.01120v2 (2026) — the Figure 2 table they start from.
* Kővári, Sós, Turán, *On a problem of K. Zarankiewicz*, Colloq. Math. 3 (1954) 50–57.

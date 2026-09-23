# Symmetry breaking for 0/1 matrix search: double-lex, canonizing sets / BreakID, snake-lex, lex-leader completeness, and proof-logged symmetry breaking (SR/PR, VeriPB dominance)

Literature notes for the thesis "Discovering upper bounds for Zarankiewicz numbers z(m,n;s,t)"
(case-split SAT attack with Lean-verified pruning). Written 2026-09-21.

**Purpose of this note.** Our pipeline (see `lean/README.md`, `docs/proposal_section2.md`) must
distinguish two moves on a case (a row/column-sum profile):

* **PRUNE** — the case is *empty* (no `K_{s,t}`-free matrix with ≥ w ones has this profile).
  Verified by a `Prune P` term: `sound : ∀ A, kill (profileOf A) = true → ¬ Valid P A`.
* **ADD** — the case is *dominated*: every matrix in it has a row/column-permuted copy in some
  surviving case, or, inside one case, every solution has a permuted copy satisfying an extra
  constraint (a symmetry-breaking predicate). This is **never** a prune (`Demo.notDescending_unsound`
  is the counterexample); it needs a *permutation witness* and lives either in the `cover`
  obligation of `upper_bound_of_cover` or inside the SAT certificate (SR/PR clause, VeriPB
  dominance step) or in an equisatisfiability theorem.

The literature below is about what may be ADDed, with what justification, when it is complete,
and how the witness argument is structured — so the evolutionary search can be told, precisely,
which candidates are prunes (accepted by `Prune.sound`) and which are additions (rejected by the
gate but potentially valuable elsewhere).

Notation: `A` is an `m × n` 0/1 matrix, `G = S_m × S_n` acts by `(σ,τ)·A = (i,j) ↦ A(σ⁻¹ i, τ⁻¹ j)`.
`Valid P A` (K_{s,t}-free and weight ≥ w) is `G`-invariant. For a profile `q = (r, c)` the
**stabiliser of the case** is the Young subgroup `G_q = (∏_v S_{R_v}) × (∏_u S_{C_u})` where `R_v`
= rows with sum `v`, `C_u` = columns with sum `u`. Everything marked **[quote]** is verbatim;
**[inferred]** is our reading.

---

## 0. Sources, accessibility, and what each contributes

| # | Source | Accessed | Role in this note |
|---|---|---|---|
| S1 | Flener, Frisch, Hnich, Kiziltan, Miguel, Pearson, Walsh. *Breaking Row and Column Symmetries in Matrix Models*. CP 2002, LNCS 2470, 462–476. doi:10.1007/3-540-46135-3_31 | Springer PDF paywalled (HTML stub only). Read instead: the SymCon'01 technical-report version *Symmetry in Matrix Models* (APES-30-2001, `lia.deis.unibo.it/~zk/Pubs/SymCon01.pdf`, full text) and the authors' 2019 retrospective arXiv:1910.01423 (full text). | Double-lex theorem and proof; partial (block) symmetry; complete special cases |
| S2 | Katsirelos, Narodytska, Walsh. *On the Complexity and Completeness of Static Constraints for Breaking Row and Column Symmetry*. CP 2010; arXiv:1007.0602 | Full text | DOUBLELEX / SNAKELEX definitions, incompleteness bounds, NP-hardness, bounded-rows completeness, unsafe combinations |
| S3 | Narodytska, Walsh. *Breaking Symmetry with Different Orderings*. CP 2013; arXiv:1306.5053 | Full text | Snake-lex and Gray-code leader NP-hardness; DC on SNAKELEX NP-hard |
| S4 | Grayland, Miguel, Roney-Dougal. *Snake Lex: An Alternative to Double Lex*. CP 2009, LNCS 5732, 391–399 | **Not accessible** (paywalled; St Andrews portal has no PDF). Definitions and results taken from S2 §7 and S3 §6.2, which restate them. | Snake-lex |
| S5 | Crawford, Ginsberg, Luks, Roy. *Symmetry-Breaking Predicates for Search Problems*. KR'96, 148–159 | PDF at `ix.cs.uoregon.edu/~luks/symmetrybreaking.pdf` is a scan whose text layer is garbled; statements taken from the verbatim restatement in S8 (BreakID, Thm 1 proof) and S2/S9. | Lex-leader definition and completeness, NP-hardness |
| S6 | Itzhakov, Codish. *Breaking Symmetries in Graph Search with Canonizing Sets*. Constraints 2016; arXiv:1511.08205 | Full text | Canonizing sets, instance-dependent breaks per degree sequence, matrix-model generalisation with DoubleLex as seed |
| S7 | Codish, Miller, Prosser, Stuckey. *Constraints for Symmetry Breaking in Graph Representation*. Constraints 24 (2019); PDF from Glasgow (`dcs.gla.ac.uk/~alice/papers/MillerCONSTRAINTS_POST_ACCEPTANCe.pdf`); IJCAI 2013 short version | Full text | `sb_l`, `sb*_l`, partitioned lex break (Def. 12, Thm 6) |
| S7b | Codish, Frank, Itzhakov, Miller. *Computing the Ramsey Number R(4,3,3) using Abstraction and Symmetry Breaking*. Constraints 2016; arXiv:1510.08266 | Full text | Degree-matrix abstraction + symmetry breaking restricted to equal-degree rows (Eq. 7, Thm 3) — the closest analogue of our fixed-profile cases |
| S8 | Devriendt, Bogaerts, Bruynooghe, Denecker. *Improved Static Symmetry Breaking for SAT*. SAT 2016, LNCS 9710, 104–122; PDF `bartbogaerts.eu/articles/2016/003/ImprovedStaticSymmetryBreakingSAT.pdf` | Full text | BreakID: compact lex-leader encoding (Thm 1), row interchangeability complete (Thm 2), binary clauses from orbits (Thm 3) |
| S9 | Anders, Brenner, Rattan. *The Complexity of Symmetry Breaking Beyond Lex-Leader*. CP 2024; arXiv:2407.04419 | Full text (pp. 1–8 read) | Complete SBPs for row–column symmetry ⇒ GI ∈ coNP (Thm 1.1); row-interchangeability is "easy" |
| S10 | Heule. *The Quest for Perfect and Compact Symmetry Breaking for Graph Problems*. SYNASC 2016 (`cs.utexas.edu/~marijn/publications/synasc16.pdf`); journal version *Optimal Symmetry Breaking for Graph Problems*, Math. Comput. Sci. 13 (2019) 533–548 (paywalled) | SYNASC PDF full text | Perfect isolators, redundancy ratio of quad/cubic methods |
| S11 | Heule, Hunt, Wetzler. *Expressing Symmetry Breaking in DRAT Proofs*. CADE-25 (2015), LNCS 9195, 591–606; `cs.utexas.edu/~marijn/publications/sbp.pdf` | Full text | DRAT symmetry breaking via definitions + sorting networks; limits |
| S12 | Heule, Kiesl, Biere. *Short Proofs Without New Variables*. CADE-26 (2017), LNCS 10395, 130–147 | Full text (pp. 1–8) | PR redundancy characterisation (Thm 1); symmetry-breaking-like PR clauses |
| S13 | Buss, Thapen. *DRAT and Propagation Redundancy Proofs Without New Variables*. LMCS 17(2) 2021 (conf. SAT 2019); arXiv:1909.00520 | Full text (relevant pages) | Substitution redundancy (SR), Def. 1.13, Thm 1.15, symmetry witnesses |
| S14 | Codel, Avigad, Heule. *Verified Substitution Redundancy Checking*. FMCAD 2024, 186–196 | Full text | DSR/LSR formats, Lean 4 verified checker (Trestle), symmetry-breaking SR proofs for Ramsey |
| S15 | Bogaerts, Gocht, McCreesh, Nordström. *Certified Dominance and Symmetry Breaking for Combinatorial Optimisation*. JAIR 77 (2023) 1539–1589; arXiv:2203.12275 | Full text | VeriPB redundance and dominance rules (Def. 6, 13), lex-leader via dominance (§4.1), relation to DRAT (§4.2), BreakID compatibility (§4.3) |
| S16 | Anders, Bogaerts, Bogø, Gontier, Koops, McCreesh, Myreen, Nordström, Oertel, Rebola-Pardo, Tan. *Faster Certified Symmetry Breaking Using Orders With Auxiliary Variables*. AAAI 2026; arXiv:2511.16637 | Full text (pp. 1–8) | Orders with auxiliary variables; O(n) vs O(n²) proof logging; CakePB verified |
| S17 | Szeider. *PBLean: Pseudo-Boolean Proof Certificates for Lean 4*. arXiv:2602.08692 | Full text | Importing VeriPB kernel proofs (incl. red/dom) into Lean 4 by reflection; trust model |
| S18 | Szeider. *Streaming LRAT Certificates into Lean Theorems* (LRAT-Catcher). arXiv:2607.00815 | pp. 1–2 | Lean-core verified LRAT checker made resumable; cube-and-conquer cover theorem |
| S19 | Gallicchio, Codel, Avigad, Heule. *An End-To-End Verification of Keller's Conjecture*. ITP 2026 | pp. 1–4 | Symmetry reasoning split between Lean and SR |
| S20 | Szeider. *SAT Modulo Symmetries: A Survey*. SC² 2025 (CEUR 4116); Kirchweger, Szeider, *SAT Modulo Symmetries for Graph Generation and Enumeration*, TOCL 25(3) 2024 | Survey full text; TOCL not fetched | Dynamic canonicity, nc-certificates (permutation witnesses per learned clause) |
| S21 | Kirchweger, Manrique, Szeider. *Formally Verified Graph Generation with SAT Modulo Symmetries and Lean*. IJCAR 2026, LNCS 16688; Manrique, Szeider, *LeanCSP*, arXiv:2607.28459 | **Abstracts only** (Springer chapter paywalled) | Lean-formalised symmetry reasoning for SMS; parametric verified symmetry breaking |
| S22 | dfield, *finite-zarankiewicz-closures*, github.com/dfield/finite-zarankiewicz-closures (cloned) | Full repo | Row-lex + equal-degree column-lex in a Zarankiewicz CNF; row-stabiliser canonical-prefix cube cover; VIPR orbit covers |
| S23 | Tan, *An attack on Zarankiewicz's problem through SAT solving*, arXiv:2203.02283 — see `docs/lit/tan2022_sat.md` §4.4 for the verbatim Theorem 3.2 and the lex encoding | Already in our notes | Sorted partitions + within-block lex; incompleteness (his Fig. 2) |

---

## 1. Lex-leader: the master symmetry-breaking predicate (S5 via S8, S2, S9)

**Definition (lex-leader constraint, BreakID Def. 1, [quote] S8).** *Let φ be a formula over Σ, π a
symmetry of φ, ⪯_x an order on Σ and ⪯_α the lexicographic order induced by ⪯_x on the set of
Σ-assignments. A formula LL_π over Σ′ ⊇ Σ is a lex-leader constraint for π if for each
Σ-assignment α, there exists a Σ′-extension of α that satisfies LL_π iff α ⪯_α π(α).*

**Crawford et al.'s clausal form (restated [quote] in S8, Thm 1 proof, eq. (1)):**
`∀i : (∀j < i : x_j ⇔ π(x_j)) ⇒ ¬x_i ∨ π(x_i)`, i.e. with f < t, "the value of x_i must be ≤ the
value of π(x_i) if for all smaller x_j, x_j has the same value as π(x_j)".

**Soundness / completeness ([quote] S8 §1):** *"The conjunction of lex-leader constraints for
every symmetry in a symmetry group constitutes a complete symmetry breaking constraint for that
group. However, symmetry groups tend to be too large to enforce lex-leader for each symmetry.
Instead, partial symmetry breaking adds lex-leader constraints only for a set of generators of the
group."* Definitions used throughout (S8 §2): *a symmetry breaking formula ψ is* **sound** *if
for each assignment α there exists at least one symmetry π ∈ Π such that π(α) satisfies ψ;*
**complete** *if for each assignment α there exists at most one symmetry π ∈ Π such that π(α)
satisfies ψ.* (S6 calls "complete and satisfied exactly by the canonical representatives"
**canonizing**.)

**Hardness.** [quote] S2 §3: *"ROWWISELEXLEADER breaks all row and column symmetries.
Unfortunately, posting such a constraint is problematic since it is NP-hard to check if a complete
assignment satisfies ROWWISELEXLEADER [5,6]."* ([5,6] = Bessiere, Hebrard, Hnich, Walsh, AAAI 2004 /
Constraints 2007.) S3 Prop. 3 generalises: for **any** "simple" total ordering there is a group for
which leader-checking is NP-hard; S3 Prop. 4/5: for row–column symmetry specifically, the Snake-Lex
and Gray-code leaders are NP-hard to find. S9 Thm 1.1 [quote]: *"Suppose there exists a polynomial
time algorithm for generating complete symmetry breaking predicates for row-column symmetries. Then
GI ∈ co-NP holds."* — and this holds *even if the SBP may introduce auxiliary variables and be a
circuit* (S9 §3.3). So no compact complete break for the full `S_m × S_n` action is expected.

**Bounded rows are tractable ([quote] S2 Thm 1):** *"For a n by m matrix, we can check if a complete
assignment satisfies a ROWWISELEXLEADER constraint in O(n! n m log m) time."* Proof idea: for each of
the n! row permutations, lex-sort the columns (the column-only lex-leader) and compare row-wise
linearisations. [inferred] For a Young subgroup with row blocks of sizes `b_1..b_k` the same
argument costs `(∏ b_v!) · n m log m`; feasible only when every equal-sum row block is small.
S2 adds: *"This result easily generalizes to when rows and columns are partially interchangeable."*

---

## 2. Double-lex (S1, S2, S23)

**Definition (S2 §3).** DOUBLELEX = rows lexicographically ordered (as vectors) **and** columns
lexicographically ordered (as vectors), both non-strictly, with one fixed direction.

**Theorem (S1 / SymCon'01 Thm 2, [quote]).** *"If a 2-d matrix model with row and column symmetry
has a solution, then it has a solution with both the rows and columns lexicographically ordered."*
**Proof ([quote]):** *"we give an ordering on matrices ... that strictly decreases each time we
lexicographically order a pair of rows or columns. To compare two matrices, we simply apply the
lexicographic ordering to the sequence formed by appending their rows together in order. Ordering a
pair of rows replaces a larger row at the front of this sequence by a smaller row from further
down. ... Ordering columns also moves us down this matrix ordering. The columns may have a number of
values in common at the top. ... there is then one value in the left column that is replaced by a
smaller value in the right column. ... The matrix ordering is also finite ... and bounded below."*

Tan's Theorem 3.2 (S23) is the same statement with an explicit potential `f(A) = Σ_{i,j} 2^{i+j}
a_{ij}` and, crucially, with the **partition clause**: *"This remains true even if the sets of rows
and columns are partitioned so that rows and columns cannot move across partitions."* SymCon'01 §4
says the same for partial symmetry: *"For each subset of rows (or columns) that are symmetric, we
impose a lexicographic ordering."* [inferred] The proof never uses that *all* rows are
interchangeable: it only swaps two rows (columns) that are allowed to be swapped. Hence
**block-wise double-lex is sound for any Young subgroup**, in particular for `G_q`.

Two independent proof routes are documented (S1 retrospective, arXiv:1910.01423 §4, [quote]):
*"Our and Lubiw's proofs show that any matrix that does not satisfy the constraint can be permuted
into one that does. In contrast, Shlyakhter's proof shows that DOUBLELEX is entailed by the row-wise
lex-leader constraints for the symmetry group."* (Lubiw, *Doubly lexical orderings of matrices*,
SIAM J. Comput. 16 (1987), gives a near-linear algorithm.) The entailment route matters for
combining constraints (see §6.4).

**Incompleteness.** [quote] S2 Thm 2: *"There exists a class of 2n by 2n 0/1 matrix models on which
DOUBLELEX leaves n! symmetric solutions, for all n ≥ 2."* The construction is `[[0, I_R],[I_R, P]]`
with `P` any permutation matrix, *"the constraints that the matrix contains 3n non-zero entries, and
each row and column contains between one and two non-zero entries"* — i.e. a **near-fixed-sum
class**, exactly our regime. Tan's Fig. 2 (S23) shows two non-identical isomorphic maximal 8×8
matrices for a=b=2 satisfying his sums + within-block lex. The retrospective: *"as Gent, Petrie and
Puget later showed, the number of remaining symmetries can be exponential in the size of the
matrix."* S2 Table 1 (unconstrained 0/1, r×c): (4,4): 317 classes vs 650 DOUBLELEX solutions;
(5,5): 5624 vs 24520; (6,6): 251610 vs 2.62·10⁶ (ratio ≈ 10, "approximately doubling with each
increase of the matrix size").

**Propagation.** [quote] S2 Thm 3: *"Enforcing DC on the DOUBLELEX constraint is NP-hard."* (Hence
one decomposes into row and column LEXCHAINs — which is what every SAT encoding does anyway.)

**Complete special cases (all with sum structure, hence relevant):**
* Matrix models of functions (every row sum 1) — S2 §5.2: ROWWISELEXLEADER *"ensures the rows and
  columns are lexicographically ordered, the row sums are 1, and the sums of the columns are in
  decreasing order"* (DOUBLELEXCOLSUM), and it is DC-propagable in polynomial time (Thm 5). This is
  Flener et al.'s Theorem 8 in SymCon'01: *"ordering its rows lexicographically as well as its
  columns lexicographically and by their sums breaks all row and column symmetry."*
* SymCon'01 Thm 10: for the class of 0/1 matrices whose row-sum partition is the **conjugate** of the
  column-sum partition, *"contains only one distinct matrix, and further lexicographically ordering
  the rows and columns breaks all symmetries."* [inferred: this is the Gale–Ryser extremal case; in
  our profile enumeration such profiles are "one-matrix cases" — decide them directly.]
* SymCon'01 Thm 7 (the trivial `rowCap`/`colCap`-type emptiness by sums) — already a prune in
  `Prunes.lean`.

---

## 3. Snake-lex (S4 via S2 §7 and S3 §6.2)

[quote] S2 §7: *"SNAKELEX ... is also derived from the lex leader method, but now applied to a
snake-wise unfolding of the matrix. To break column symmetry, SNAKELEX ensures that the first column
is lexicographically smaller than or equal to both the second and third columns, the reverse of the
second column is lexicographically smaller than or equal to the reverse of both the third and fourth
columns, and so on up till the penultimate column is compared to the final column. To break row
symmetry, SNAKELEX ensures that each neighbouring pair of rows ... satisfy the entwined
lexicographical ordering: ⟨X_{1,i}, X_{2,i+1}, X_{3,i}, X_{4,i+1}, ...⟩ ≤_lex ⟨X_{1,i+1}, X_{2,i},
X_{3,i+1}, X_{4,i}, ...⟩."* S3: *"The (columnwise) SNAKELEX constraint can be enforced by a conjunction
of 2m−1 lexicographical ordering constraints on pairs of columns and n−1 lexicographical constraints
on pairs of intertwined rows."*

Results: [quote] S2 Thm 8: *"There exists a class of 2n by 2n+1 0/1 matrix models on which SNAKELEX
leaves O(4ⁿ/√n) symmetric solutions."* S3 Prop. 4: finding the Snake-Lex-smallest row/column
permutation is NP-hard; Prop. 7: DC on SNAKELEX is NP-hard. Empirically (S2 Table 1) column-wise
snake-lex leaves fewer symmetric solutions on 0/1 matrices ((6,6,2): 1.71·10⁶ vs 2.62·10⁶) but *"is
significantly slower ... because it tends to prune later"*; row-wise snake-lex was ≈2× faster than
DOUBLELEX on EFPA.

[inferred for us] Snake-lex is sound for a Young subgroup by the same lex-leader-entailment argument
applied to the snake linearisation, **but it must not be mixed with row-major double-lex** (different
linearisations; see §6.4). We record it as an alternative ADD family, not a default.

---

## 4. Graph-style lex breaks, partitioned breaks, and canonizing sets (S6, S7, S7b)

Although our matrices are bipartite (independent row/column permutations), the Codish school's
machinery for adjacency matrices carries over and — more importantly — their **per-degree-sequence /
degree-matrix** workflow is the graph analogue of our per-profile case split.

**Simultaneous-permutation lex break ([quote] S7 Def. 6, Thm 1).** `sb_ℓ(A) = ⋀_{i=1}^{n−1} A[i] ≼ A[i+1]`;
*"Definition 6 is more subtle than might first appear. It defines a symmetry breaking predicate only
because for every adjacency matrix A, sb_ℓ(A′) is true for at least one of the matrices A′
isomorphic to A. Reversing the order ... would not define a symmetry breaking constraint."* The
stronger `sb*_ℓ(A) = ⋀_{i<j} A[i] ≼_{i,j} A[j]` compares rows with positions i,j deleted (Def. 8); the
correctness proof is "canonical (lex-min) ⇒ predicate holds" (Thm 4).

**Partitioned lex break ([quote] S7 Def. 12, Thm 6).** For an ordered partition `P = {P_1,...,P_p}`
of the vertices (e.g. by degree): `sb*_ℓ(A,P) = ⋀_{k=1}^{p} ⋀_{{i,j} ⊆ P_k, i<j, j−i≠2} A[i] ≼_{i,j} A[j]`;
*"Theorem 6. Let G be a canonical partitioned graph for an ordered partition P. Then sb*_ℓ(A_G, P)
holds."* Proof: the swap of `i,j ∈ P_k` is partition-preserving, so it produces a smaller
representative in the partition-restricted orbit.

**Degree-matrix refinement ([quote] S7b Eq. 7, Thm 3).** When a degree matrix `M` (per-vertex,
per-colour degrees; rows and columns lex-sorted) is fixed, *"we can no longer apply the symmetry
breaking Constraint (4) as it might constrain the rows of A in a way that contradicts the constraint
α(A) = M ... However, we can refine Constraint (4), to break symmetries on the rows of A only when
the corresponding rows in M are equal."* `sb*_ℓ(A,M) = ⋀_{i<j} ((M_i = M_j ⇒ A_i ≼_{i,j} A_j))`;
*"Theorem 3 (correctness of sb*_ℓ(A,M)). Let A be an adjacency matrix with α(A) = M. Then, there
exists A′ ≈ A such that α(A′) = M and sb*_ℓ(A′, M) holds."* This is **exactly the shape of our ADD
theorem inside a fixed profile**: existence of an equivalent matrix in the *same case* satisfying the
break. Their pipeline (Table 3/4): 280 degree sequences → 80 feasible → 11,933 degree matrices →
999 feasible → 129,188 solutions → 78,892 modulo isomorphism (nauty post-processing because the
break is partial). Solving times were heavy-tailed: *"average solving time is 14 hours while the
median is 4 hours ... The worst-case solving time is 96.36 days."*

**Canonizing sets ([quote] S6 Def. 2, Def. 3, Lemma 1).** `min_Π(G) = ⋀{G ≼ π(G) | π ∈ Π}`;
*"We say that G is canonical if min_{S_n}(G). We say that Π is canonizing if ∀G ∈ G_n. min_Π(G) ↔
min_{S_n}(G)."* *"Lemma 1. Let Π be a canonizing set of permutations for graphs of size n. Then
min_Π is a canonizing symmetry break for any graph search problem on n vertices."* Algorithm 1
grows `Π` by SAT-finding a counterexample `(G, π)` with `min_Π(G) ∧ π(G) ≺ G` (constraints (4)–(6):
`perm_n(π)`, `iso_n(A,B,π)`, `min_Π(A) ∧ A ≻ B ∧ φ(A)`), until UNSAT; Algorithm 2 removes redundant
permutations. Instance-independent sizes (Table 1): n=6: 13 perms; n=8: 135; n=9: 842; n=10: 7853
(84 h). Instance-dependent sets *per degree sequence* (§4.2, Eq. 8 adds the cardinality constraint
`Σ_j A_{i,j} = d_i` and "a constraint stating that B has the same degree sequence as A") are far
smaller (avg. 6.5–16.7 permutations for n = 11..20, Table 4). **Matrix models (§5):** *"For matrix
search problems we initialize Algorithm 1 taking Π to include the permutation pairs corresponding to
the DoubleLex symmetry break."* Table 5 (EFPA): e.g. (4,3,5,4): DoubleLex 14 perms → 61,258
solutions; canonizing 1537 perms → 8,600 solutions (the true count); in several instances DoubleLex
was already complete (Δ negative).

**Perfect isolators (S10).** Redundancy ratio = admitted assignments / isomorphism classes; for the
shatter-style "quad" method and Codish's "cubic" method it grows *"approximately (k−5)² for quad and
(k−6)² for cubic"*. Optimal (smallest perfect) isolators: 7 clauses for order 4, 12 clauses for
order 5 (each literal occurs ≤ twice); canonical-set formulas are exponentially larger (order 5: 225
clauses). Beyond order 8 nothing is known. [inferred] Perfect breaking of `G_q` for our block sizes
(blocks of 15–20 equal-sum columns) is out of reach; the realistic target is *partial* breaking plus
orbit covers at the cube level.

---

## 5. BreakID and structure-aware partial breaking (S8, S15 §4.3, S9)

**Compact encoding ([quote] S8 Thm 1).** With `Supp(π) = {x_1,...,x_n}` in ⪯_x order and fresh
`y_0,...,y_{n−1}`: clauses `y_0`; `¬y_{i−1} ∨ ¬x_i ∨ π(x_i)` (1 ≤ i ≤ n); `y_j ∨ ¬y_{j−1} ∨ ¬x_j` and
`y_j ∨ ¬y_{j−1} ∨ π(x_j)` (1 ≤ j < n) — three ternary clauses per support variable (vs. Aloul et
al.'s 2×3 + 2×4). The relaxation `y_j ⇐ (y_{j−1} ∧ (x_j ∨ ¬π(x_j)))` is sound because `y_j` occurs
only negatively in the breaking clause. BreakID posts `LL^{50}_π` by default (only the first 50
support variables).

**Row interchangeability ([quote] S8 Def. 2, Thm 2).** *"A formula φ exhibits row interchangeability
symmetry if there exists a variable matrix M such that for each permutation ρ : Ro → Ro,
π^M_ρ : x_{rc} ↦ x_{ρ(r)c} ... is a symmetry of φ."* *"Theorem 2 (Complete symmetry breaking for row
interchangeability). Let φ be a formula and R_M a row interchangeability symmetry group of φ with
Ro = {1,…,n} and Co = {1,…,m}. If the total variable order ⪯_x on Σ satisfies x_{ij} ⪯_x x_{i′j′}
iff i < i′ or (i = i′ and j ≤ j′), then the conjunction of lex-leader constraints for π^M_{(k k+1)}
with 1 ≤ k < n breaks M_R completely."* — i.e. **adjacent-row lex chain under row-major order is
complete for the pure row group**; *"The condition that the order 'matches' the variable matrix is
important: the theorem no longer holds without it."* S9 lists row-interchangeability among the
"easy" (P) groups and row–column symmetry among the "hard" ones (Fig. 1).

**Binary clauses from orbits ([quote] S8 Thm 3).** *"Let Π be a non-trivial symmetry group of φ,
⪯_x an ordering of Σ, and x* the ⪯_x-smallest variable in Supp(Π). For each x ∈ Orb_Π(x*), the
binary clause ¬x* ∨ x is entailed by LL_π for some π ∈ Π."* Applied along a stabiliser chain this
yields O(|Supp|²) binary clauses; BreakID excludes row-interchangeability groups from this step to
avoid quadratic blow-up.

**Proof logging of BreakID output** — S15 §4.3 [quote]: *"Since our proof logging techniques simply
use the same lexicographic order as the symmetry breaking tool, and work for an arbitrary generator
set, this automatically works"* (row interchangeability); the compact encoding is handled by
deriving the full clauses then deleting the superfluous ones from the derived set; the binary-clause
optimisation would need bookkeeping of which symmetry justifies each clause (*"BreakID currently
does no such bookkeeping"*); partial breaking (`L = 100`) works out of the box provided the order is
defined only on the variables actually broken.

[inferred for us] Our variable matrix `x_{ij}` under `G_q` is a product of row-interchangeability
groups (one per equal-sum row block) and column-interchangeability groups (one per column block).
BreakID's Thm 2 breaks each *factor* completely under a matching order, but the row-major order
cannot simultaneously "match" the column blocks, so the product is broken only partially (this is
double-lex again, now with a theorem telling us which half is complete).

---

## 6. Proof-logged symmetry breaking: PR, SR, DRAT, VeriPB dominance (S11–S16)

### 6.1 Redundancy notions

* **PR ([quote] S12 Thm 1).** *"Let F be a formula, C a clause, and α the assignment blocked by C.
  Then, C is redundant w.r.t. F if and only if there exists an assignment ω such that ω satisfies C
  and F|α ⊨ F|ω."* Replacing `⊨` by unit-propagation implication `⊢₁` gives PR (checkable in
  polynomial time given ω); SPR restricts `dom(ω) = dom(α)`; LPR = RAT.
* **SR ([quote] S13 Def. 1.13, Thm 1.15).** *"A clause C is substitution redundant (SR) with respect
  to Γ if there is a substitution τ such that τ ⊨ C and Γ|α ⊢₁ Γ|τ."* *"Theorem 1.15. If C is SR
  with respect to Γ, then Γ and Γ ∪ {C} are equisatisfiable."* Proof: a total `π ⊨ Γ` falsifying `C`
  extends `α`, so `π ⊨ Γ|α`, hence `π ⊨ Γ|τ`, hence `π ∘ τ ⊨ Γ` and `π ∘ τ ⊨ C`. **The witness is a
  substitution (variables ↦ literals), so a permutation of variables is a legal witness** — Example
  1.14 breaks the pigeonhole symmetry with `τ = α ∘ π`, `π` the swap of pigeons 0 and 1, using
  `Γ|π = Γ`. Def. 4.1: *"A Γ-symmetry is an invertible substitution π such that Γ|π = Γ."*
* **Verified SR checking in Lean 4 (S14).** LSR/DSR formats: an addition line is `clause, ⟨witness⟩,
  hints`, witness = `p : lit, [lit], ⟨p, [(var, lit)]⟩` (literals set true, then variable↦literal
  pairs). Trestle: *"our verified checker and its supporting theorems and data structures comprise 8k
  LoC, and the verification took 4 person months"*; ≈10× slower than the unverified `lsr-check`,
  comparable to `cake_lpr` on PR proofs; SR proofs were ≈10× smaller than converted LRAT.
  Symmetry-breaking example ([quote] §III.C): *"we can assume that the blue edges for vertex v1 come
  first, represented by the clauses e_{1,j} ∨ ¬e_{1,j+1} for 1 < j < n ... These binary clauses are
  SR. For instance, symmetry-breaking clause e_{1,2} ∨ ¬e_{1,3} has witness σ = {e_{1,2} ↦ ⊤,
  e_{1,3} ↦ ⊥, e_{2,4} ↦ e_{3,4}, e_{3,4} ↦ e_{2,4}, e_{2,5} ↦ e_{3,5}, ...}."* R(4,4) ≤ 18 has a
  38-clause SR proof (vs ≈10⁹ resolution steps). Future work stated: *"automatically generating
  symmetry-breaking SR proofs."*
* **DRAT (S11).** Symmetry breaking is expressible in DRAT only by introducing definitions
  (primal-swap variables `s_i`, copies `x′_i`), *redefining all involved clauses*, and then adding
  lex-leader clauses; for `k` overlapping symmetries a sorting network of swaps is needed and
  *"Conjecture 1: ... it requires in worst case O(nk log k) swaps"*. Proofs for TPH-12 reached 4 GB.
  S15 §4.2 [quote]: *"already a symmetry σ that is a cyclic shift of three variables ... brings us
  beyond what DRAT-based proof logging symmetry breaking is currently able to handle."*

### 6.2 VeriPB: redundance and dominance (S15)

Configuration `(C, D, O_⪯, z⃗, v)`: core constraints `C`, derived `D`, preorder encoding `O_⪯(u⃗,v⃗)`
(reflexivity/transitivity must be derived in a preamble, (13a)/(13b)), order variables `z⃗`, bound `v`.

**[quote] Definition 6 (Redundance-based strengthening rule).** derive `C` with witness `ω` if
`C ∪ D ∪ {f ≤ v−1} ∪ {¬C} ⊢ (C ∪ D ∪ {C})↾ω ∪ O_⪯(z⃗↾ω, z⃗) ∪ {f↾ω ≤ f}` (this is SR/PR lifted to PB
plus an order-non-increase side condition).

**[quote] Definition 13 (Dominance-based strengthening rule).** *"If for a pseudo-Boolean constraint
C there is a witness substitution ω such that the conditions*
`C ∪ D ∪ {f ≤ v − 1} ∪ {¬C} ⊢ C↾ω ∪ O_⪯(z⃗↾ω, z⃗) ∪ {f↾ω ≤ f}` (10a)
`C ∪ D ∪ {f ≤ v − 1} ∪ {¬C} ∪ O_⪯(z⃗, z⃗↾ω) ⊢ ⊥` (10b)
*are satisfied, then we can transition from (C, D, O_⪯, z⃗, v) to (C, D ∪ {C}, O_⪯, z⃗, v)."*
Meaning: any assignment satisfying the core but violating `C` is mapped by `ω` to a **strictly
smaller** (w.r.t. `⪯`) assignment that still satisfies the *core* (not necessarily `D` or `C`); by
well-foundedness a core-satisfying assignment satisfying everything exists (Prop. 14). Deletion from
the core is restricted (Example 15 shows unsoundness otherwise).

**Lex-leader via dominance ([quote] §4.1).** Fix `O_⪯` = lexicographic order on `x⃗` (encoded as the
single PB inequality `2^{n−1}x_1 + … + x_n ≤ 2^{n−1}y_1 + … + y_n`, eq. (2) of S16). For a syntactic
symmetry `σ` of `C` derive `C_LL := Σ_{i=1}^{m} 2^{m−i}·(σ(x_i) − x_i) ≥ 0` (20) by dominance with
witness `σ`: (10a) holds because `¬C_LL` says `x⃗` is strictly larger than `σ(x⃗)`, i.e. `O_⪯(x⃗↾σ, x⃗)`,
and `C↾σ = C`; (10b) holds because `C_LL` and `¬C_LL` are both premises. Then the clausal lex-leader
(19a)–(19f) (with fresh `y_j`) is derived by redundance + RUP/literal-axiom steps from `C_LL(k) :=
C_LL(k−1) + 2^{m−i_k}·(19d[j=k])`. *"the order used for the dominance-based strengthening is fixed at
the beginning and remains the same for all symmetries σ ∈ G to be broken. Since constraints are added
only to the derived set D, dominance rule applications for different symmetries will not interfere
with each other. Furthermore, in contrast to the approach of Heule et al. (2015), handling a symmetry
once is enough to guarantee complete breaking."* And §4.2: *"There is no requirement that our witness
substitutions should generate minimal assignments—all that is needed is that the witnesses yield
smaller assignments."*

**Scaling fix (S16).** The big-integer lex order costs O(n²) per symmetry in logging and checking
and *"quickly becomes infeasible for large symmetries"*; S16 redefines `⪯` with auxiliary
(specification) variables `a⃗` — `α ⪯ β` iff `∃ρ. S_⪯(z⃗↾α, z⃗↾β, a⃗↾ρ) ∧ O_⪯(z⃗↾α, z⃗↾β, a⃗↾ρ)` (Def. 2) —
with proof obligations (11)/(12) for reflexivity/transitivity and a modified dominance rule (Def. 4,
(15)/(16)); logging drops to O(k) per symmetry of support `k` and checking to O(n). Implemented in
VeriPB and the formally verified CakePB.

### 6.3 Importing certified symmetry breaking into Lean (S17, S18, S14, S19, S21)

* **PBLean (S17):** *"Our implementation covers all VeriPB kernel rules, including proof-by-
  contradiction subproofs for optimization reasoning and redundance/dominance rules for symmetry
  breaking."* Lean v4.28.0-rc1, **no Mathlib**, kernel layer ≈900 lines with 15 soundness lemmas
  (`applySubstConstr_sat_rev`, `constr_sat_noSubst` for red/dom). Two modes: explicit proof terms
  (kernel + lemmas only; *"does not scale: Paley(29) already takes 20 s, and all instances with p ≥
  37 exceed the 60 s timeout"*) and **reflection via `native_decide`**, which *"has the same trust
  basis as bv_decide and omega—namely, Lean's standard axioms (propext, Classical.choice, Quot.sound)
  plus Lean.trustCompiler"*. The PHP(3,2) example uses `red` *"to add a symmetry-breaking constraint
  (without loss of generality, pigeon 1 goes to hole 1) via a cyclic substitution witness."*
* **LRAT-Catcher (S18):** makes *"the verified LRAT checker of Lean core [Böving et al. 2025]
  resumable"*, streams certificates, and *"supports cube-and-conquer ... One lemma combines their
  theorems with the cover theorem into one theorem about the full formula."* ≈15× the CPU of
  `lrat_isa`; 174 TB (empty hexagon) imported in stream mode.
* **Keller (S19) [quote]:** *"the symmetry reasoning was split between Lean and a mechanically-
  checkable proof system, since neither was suitable on their own for verifying all of the symmetry
  reasoning."* (Uses Trestle for the SAT side.)
* **SMS (S20) nc-certificates [quote]:** *"There exists a permutation π such that for all extensions
  H ∈ X(G), the adjacency matrix of π(H) is lexicographically smaller than the adjacency matrix of
  H."* *"SMS outputs an nc-certificate (a permutation π) for each learned symmetry-breaking clause. An
  independent checker can verify in polynomial time that this permutation indeed witnesses the
  non-canonicity of the pruned partial graph."* S21 (abstract): the IJCAR 2026 paper *"formaliz[es]
  graph invariance and symmetry reasoning within Lean to eliminate common trust assumptions"*;
  LeanCSP proves symmetry-breaking equisatisfiability *"parametrically across problem families"*
  and reports search reductions *"by a factor of up to 2×10⁷"*.

**Trust-policy flag for our gate.** `lean/README.md` forbids `native_decide`. Every scalable
reflective importer above (PBLean reflection mode, Lean-core LRAT via `bv_decide` machinery) rests on
`Lean.ofReduceBool`/`Lean.trustCompiler`. The `refuted` seam of `upper_bound_of_cover` therefore
either (i) relaxes the policy for certificate import only (auditable with `#print axioms`), or
(ii) uses explicit-proof-term import (does not scale beyond toy instances), or (iii) keeps the
certificate check outside Lean (cake_lpr / CakePB / Trestle, which are themselves verified) and
imports only the *verdict* as an axiom-free but externally trusted hypothesis. This is a design
decision to record, not resolve here.

---

## 7. Fixed row/column sums: how the pieces interact

### 7.1 What Tan and dfield actually do (S23, S22)

* **Tan (S23):** enumerate *unordered* partition pairs (sorted non-increasing sums), prune by
  counting arguments, then per surviving pair a CNF with fixed sums and *"groups of rows or columns
  with the same sum ... contiguous and lexicographically sorted"* (Theorem 3.2, with *"1 comes before
  0"*), encoded with `n−1` auxiliary "already decided" variables per pair (his at-most encoding). For
  `a=b`, `m=n` only unordered `{rpart, cpart}` pairs — a transposition reduction at the case level.
  Acknowledged incompleteness (Fig. 2).
* **dfield (S22):** fixes **column** degrees only (rows unconstrained except `≥ 10` via
  `Z(9,23)=103`), so the row group is the full `S_10`; `sat_tool.py`: *"Symmetry breaking: lex order
  on consecutive rows (whole matrix), and lex order on consecutive columns within each equal-degree
  block"* — i.e. global row lex chain (complete for the row factor, BreakID Thm 2) plus block column
  lex; `PROOF_Z10_23.md`: *"applies sound row and equal-degree-column lexicographic symmetry
  breaking"*. For one hard profile (`3¹4²5¹⁸6²`) a **row-stabiliser cube cover**: `child_supports`
  docstring [quote]: *"Rows with the same fixed prefix form one stabilizer cell. Global row-lex order
  forces the next support to be an initial segment of every such cell. Equal-degree column lex order
  supplies the second comparison below."* — an orderly-generation cover over column prefixes (17,170
  leaves, each refuted by DRAT/LRAT; catalog regenerated independently and required to be exact and
  prefix-free). Two harder profiles (`3¹4³5¹⁶6³`, `3¹4⁴5¹⁴6⁴`) used **orbit covers under the row
  group**: 950,250 / 295,001 raw states → 236 / 209 orbit representatives, each an exact MIP with a
  VIPR certificate; the verifier *"applies the relevant row group actions, recomputes canonical
  signatures and orbit sizes"*. Lean there verifies **only arithmetic** (deficit/deletion closures),
  not the symmetry reductions — the symmetry soundness is Python-checked.

[inferred] The two designs differ on *which* sums are fixed. Fixing only column sums keeps a big row
group that a single lex chain breaks completely; fixing both sums (Tan, and our `Profile`) shrinks
the group to `G_q` but leaves two incompletely-broken factors. There is a real trade-off here between
fewer, larger cases with complete row breaking and more, smaller cases with partial breaking; the
counting prunes (KST etc.) need both sums, which favours our current profile design.

### 7.2 Sound ADD moves for a fixed profile `q = (r, c)` — with justification

All are consequences of `Valid` being `G`-invariant plus one existence theorem each.

| ID | Constraint | Group used | Justification | Complete? |
|---|---|---|---|---|
| ADD-0 | Case-level: keep only profiles with `r`, `c` sorted non-increasing (Tan's partitions) | `G` | witness: the sorting permutations `(σ_r, τ_c)`; the orbit of a profile under `G` contains exactly one sorted pair | complete at case level (orbit representative) |
| ADD-T | Case-level, `m=n, s=t`: keep `{r,c}` unordered (only `r ≤ c` in some total order) | transpose ⋊ `G` | witness: transpose; `HasKst` for `s=t` is transpose-invariant | complete at case level |
| ADD-1 | Row lex within each equal-sum row block (one fixed direction) | `∏ S_{R_v}` | SymCon'01 Thm 1 / BreakID Thm 2 (adjacent swaps within a block) | complete **for the row factor alone** |
| ADD-2 | ADD-1 **and** column lex within each equal-sum column block, same direction, row-major reading | `G_q` | SymCon'01 Thm 2 with partition clause (Tan Thm 3.2); or Shlyakhter entailment from row-major lex-leader on `G_q` | **incomplete** in general (S2 Thm 2, Tan Fig. 2) |
| ADD-3 | First row of each row block restricted to each column block is `1…10…0` | derived | entailed by ADD-2's column half (as in S2 §6.1 for EFPA) | — |
| ADD-4 | ALLPERM-style: first row of a block ≤_lex every permutation of every other row in the block (Frisch–Jefferson–Miguel 2003, via S1 retrospective) | `G_q` | entailed by row-major lex-leader (retrospective §5) | partial |
| ADD-5 | Snake-lex within blocks | `G_q` | lex-leader on snake linearisation | partial; **not combinable** with ADD-2 |
| ADD-6 | Canonizing set `Π_q` (Itzhakov–Codish Alg. 1 seeded with ADD-2's adjacent-swap pairs, restricted to `G_q`) | `G_q` | S6 Lemma 1 (each `π ∈ Π_q` is a lex-leader clause set; completeness certified by the UNSAT run of Alg. 1) | complete within the case if Alg. 1 terminates; cost exponential-ish |
| ADD-7 | Dynamic: canonical-prefix cubes (dfield `child_supports`) / SMS nc-certificates | stabiliser of the assigned prefix | per cube a permutation `π` with `π(H) <_lex H` for all extensions `H` | complete for the *cubed* prefix variables |
| ADD-8 | BreakID binary clauses `¬x* ∨ x` for `x` in the orbit of the smallest variable of a block product (S8 Thm 3) | `G_q` and stabilisers | entailed by lex-leader | partial |

Direction convention: any single fixed direction is sound (Flener's proof works for lex-max as for
lex-min), but **the same direction and the same linearisation must be used for every ADD in a case**
(§6.4). Tan's "1 before 0" with sorted-descending sums is the lex-max convention under row-major
reading; the corresponding lex-leader is "`A` is lex-max in its `G_q` orbit".

### 7.3 What is complete within a fixed-profile class

* Full lex-leader over `G_q` (all `∏|R_v|! · ∏|C_u|!` elements): complete by definition, checkable
  in `(∏|R_v|!)·mn log n` time by S2 Thm 1 restricted to blocks — practical only when every row block
  is tiny (e.g. all row sums distinct ⇒ `G_q` acts on columns only ⇒ block column lex is complete by
  BreakID Thm 2 applied to the transpose).
* Special complete cases from S1/S2: row sums all 1 (function matrices), conjugate partitions
  (one matrix). [inferred] Both are outside the interesting Zarankiewicz regime but are cheap
  "decide-by-hand" cases.
* Otherwise ADD-2 is incomplete and the residual symmetry is what S10 calls the redundancy ratio;
  in the S2 experiments it was ≈ 6–10 for 6×6 0/1 matrices and grows exponentially.

### 7.4 Safe and unsafe combinations

[quote] S2 §6.2 (Thm 6): *"Unfortunately Puget's method for breaking value symmetry is not
compatible in general with breaking row and column symmetry using ROWWISELEXLEADER. ... combining
symmetry breaking constraints based on row and column-wise linearisations can, as in our example,
eliminate all solutions in a symmetry class."* Thm 7 strengthens this to "irrespective of the
orderings used by both methods". Two safe patterns:

1. **One linearisation, entailment.** Every ADD clause is entailed by the lex-leader predicate of a
   single fixed linearisation of the cell variables (S15: *"the order ... is fixed at the beginning
   and remains the same for all symmetries"*). ADD-1/2/3/4/8 under row-major reading qualify.
2. **Sequential normalisation with invariance.** Apply ADD-0 / ADD-T at the case level first
   (their normal forms are properties of the *orbit of the case*, invariant under `G_q`), then the
   in-case ADD-2 existence theorem (which stays inside the case). [inferred] This is sound because
   step 2's theorem is existential over the whole case and does not depend on which member we
   started from. Formally: `Valid A → ∃ A₁ ∈ Case(rep(orbit q)), Valid A₁` then
   `Valid A₁ ∧ profileOf A₁ = q′ → ∃ A₂, profileOf A₂ = q′ ∧ Valid A₂ ∧ DoubleLexBlocks A₂`.

Unsafe: ADD-2 together with ADD-5; any ADD stated w.r.t. a column-major reading together with a
row-major one; a lex direction for rows differing from the direction for columns unless proved
compatible (SymCon'01 §3's 2×2 permutation-matrix example: rows lex-increasing and columns
lex-decreasing has **no** solution).

---

## 8. Lean structure of the ADD (permutation-witness) argument for ZarPrune

Nothing here touches `Prune`. The gate's `Prune.sound` stays "case empty". ADDs enter at two seams.

### 8.1 Seam A: `cover` in `upper_bound_of_cover` — orbit representatives (ADD-0, ADD-T)

Proposed Mathlib-free ingredients (over `Fin`, `sumFin`, `HasKst` as increasing tuples):

```lean
/-- A permutation of `Fin k` as a two-sided-inverse pair (core Lean has no `Equiv`). -/
structure Perm (k : Nat) where
  f : Fin k → Fin k
  g : Fin k → Fin k
  fg : ∀ i, f (g i) = i
  gf : ∀ i, g (f i) = i

def act {m n} (σ : Perm m) (τ : Perm n) (A : Mat m n) : Mat m n :=
  fun i j => A (σ.g i) (τ.g j)
```

Lemmas needed (difficulty estimates for Lean 4.34 core, no Mathlib):

* `sumFin_perm : sumFin k (f ∘ σ.f) = sumFin k f` — **medium**. Route: `sumFin k f = (List.finRange k).map f |>.sum`
  by induction, then `List.Perm` of the mapped lists; core Lean ships `List.Perm` and
  `List.Perm.sum_eq`-style lemmas in `Init.Data.List.Perm` (name and availability to be checked with
  `lake env lean`; if absent, prove `sum` invariance under `List.Perm` by induction on the `Perm`
  derivation, ≈40 lines). Alternative: the "swap-adjacent" decomposition of `σ` — longer.
* `rowSum_act : rowSum (act σ τ A) i = rowSum A (σ.g i)` — by `sumFin_perm` on columns;
  `colSum_act` dual; `weight_act : weight (act σ τ A) = weight A` — from the two. **Easy** given
  `sumFin_perm`.
* `profileOf_act : profileOf (act σ τ A) = { row := (profileOf A).row ∘ σ.g, col := (profileOf A).col ∘ τ.g }` — `rfl`-level after the above.
* `HasKst_act_iff : HasKst P (act σ τ A) ↔ HasKst P A` — **the hard one (medium–hard, ≈150–300
  lines)**. Given increasing `R : Fin s → Fin m` witnessing `K_{s,t}` in `A`, the tuple `σ.f ∘ R` is
  injective but not increasing; one needs "an injective `Fin s → Fin m` can be re-indexed to an
  increasing tuple with the same image". Recommended: prove once `HasKst_iff_inj : HasKst P A ↔
  ∃ R C, Injective R ∧ Injective C ∧ ∀ a b, A (R a) (C b) = true` via an insertion-sort on
  `Fin`-tuples (or via `List.mergeSort` on `List (Fin m)` with a `Nodup`/`Sorted` transfer), then
  invariance is immediate because injective tuples compose with bijections. This lemma is also
  needed by `docs/lit/tan2022_sat.md` §7.3 (L-A / L-sort) — shared infrastructure.
* `Valid_act_iff : Valid P (act σ τ A) ↔ Valid P A` — from the two above. **Easy.**

Then the generalised closure theorem (replacing the literal `cover`):

```lean
theorem upper_bound_of_cover_upto_perm (P) (p : Prune P) (survivors : List (Profile P.m P.n))
  (cover : ∀ A, Valid P A → p.kill (profileOf A) = true ∨
     ∃ q ∈ survivors, ∃ σ τ, profileOf (act σ τ A) = q)
  (refuted : ∀ q ∈ survivors, ∀ A, profileOf A = q → ¬ Valid P A) :
  ∀ A, ¬ HasKst P A → weight A < P.w
```

Proof: as `upper_bound_of_cover`, but in the second branch apply `refuted q _ (act σ τ A)` and
`Valid_act_iff`. The `cover` hypothesis is discharged by (i) a computable `sortProfile : Profile →
Profile × Perm × Perm` with `profileOf (act σ τ A) = sortProfile (profileOf A)` (sorting `Fin m →
Nat` by value; `List.mergeSort` exists in core, its permutation lemma `List.mergeSort_perm` too —
checked in the Tan notes), and (ii) a `decide`/reflective check that every sorted profile of total
`w` is killed or listed. ADD-T adds a `transpose` action and `HasKst_transpose_iff` (needs `s = t`,
swap `R,C`; **easy**).

### 8.2 Seam B: `refuted` — in-case symmetry breaking (ADD-1..8)

Two ways to make the per-case SAT refutation of `CNF(q) ∧ LexClauses` count as a refutation of
`CNF(q)`:

**B1 (certificate-side, no new Lean theory).** Emit the lex clauses as **SR additions with the
swap witness** (S13 Ex. 1.14 / S14 §III.C pattern: for adjacent rows `i, i+1` in a block and the
prefix-equality auxiliaries, the witness maps `x_{i,·} ↔ x_{i+1,·}` and sets the pivot literal), or
as VeriPB **dominance** steps with the fixed lex order (S15 §4.1) and let a verified checker (Trestle
in Lean 4; CakePB; or PBLean's importer) validate. Then `refuted` is exactly "the checked certificate
refutes `CNF(q)`" and the encoding-correctness theorem is untouched. Subtlety: the auxiliary
prefix-equality variables `y_j` (Tan's `c_i`) are fresh; S15 derives their defining clauses by
redundance before the breaking clause; S14's Trestle currently assumes the witness satisfies the
candidate clause (footnote 5 / §V: *"the DSR and LSR formats can also express proofs where σ causes
C|σ to be a tautology ... We plan to support this general case"*). Cost: proof-logged, SR-format
support in solvers is still absent (S14 §IX: *"no modern SAT solver supports SR reasoning yet"*), so
the workflow is: our own generator writes the SR/dominance prefix, the solver's DRAT/LRAT for
`CNF ∧ Lex` is appended, and a checker validates the concatenation (the S11 "merge" workflow).

**B2 (Lean-side equisatisfiability theorem).** Prove in ZarPrune, once, the partition version of
SymCon'01 Thm 2 / Tan Thm 3.2:

```lean
def DoubleLexBlocks (q : Profile m n) (A : Mat m n) : Prop :=
  (∀ i i', i < i' → q.row i = q.row i' → rowVec A i' ≤lex rowVec A i) ∧
  (∀ j j', j < j' → q.col j = q.col j' → colVec A j' ≤lex colVec A j)

theorem exists_doubleLex (q) (A) (hA : profileOf A = q) (hV : Valid P A) :
  ∃ A', profileOf A' = q ∧ Valid P A' ∧ DoubleLexBlocks q A'
```

Proof structure (the potential-function argument): define `pot A := sumFin m (fun i => sumFin n
(fun j => 2^(i+j) * ind (A i j)))`; show that swapping two out-of-order adjacent rows in a block
(or columns) strictly increases `pot` (Tan's `(2^b − 2^a)(n_a − n_b) > 0` computation — needs
`2^(i+j)` arithmetic and a lemma that a lex comparison of two 0/1 rows is decided by the first
differing column, ≈100 lines), that swapping preserves `profileOf` within the block and preserves
`Valid` (by `Valid_act_iff` with the transposition), and that `pot` is bounded by `Σ 2^{i+j}`;
conclude by strong induction on `Σ 2^{i+j} − pot A`. **Medium–hard, ≈250–400 lines Mathlib-free**,
but it makes the SAT side ordinary: `refuted q` is obtained from an LRAT refutation of `CNF(q) ∧
Lex(q)` via the encoding theorem `(∃ A', profileOf A' = q ∧ Valid A' ∧ DoubleLexBlocks q A') →
SAT(CNF(q) ∧ Lex(q))`, whose lex part is a straightforward semantics lemma for the `n−1`-auxiliary
comparator (Tan §3.3). This is the Keller-paper split (S19): permutation reasoning in Lean, unit
propagation in the certificate. It also generalises: any ADD that is a `∃ A' ∈ case, Valid A' ∧ Φ A'`
theorem slots in without touching the checker.

**Recommendation.** B2 for ADD-2 (one theorem, reused for every case and every `(m,n,s,t)`), B1 for
anything case-specific the search discovers (e.g. ADD-6 canonizing sets, ADD-7 cube covers), since
those come with an explicit permutation per clause/cube and are naturally SR/dominance steps or
SMS-style nc-certificates. Either way the **evolved population never sees an ADD as a `Prune`**; the
evolutionary harness should classify a rejected candidate whose `kill` is a symmetry-break shape
(`row i < row i'` comparisons, transposition tests) as "ADD candidate" and route it to seam B rather
than discarding it.

---

## 9. Difficulty signals derived from symmetry structure

* **Residual stabiliser size** `|G_q| = ∏_v |R_v|! · ∏_u |C_u|!` (log-scale). Larger equal-sum
  blocks ⇒ more symmetric solutions left by ADD-2 (S2 Thm 2 grows as n! in the block; S10's
  redundancy ratio grows ≈ quadratically for graphs, exponentially for unconstrained matrices). The
  dfield data are consistent: the profiles that needed cube/orbit covers (`5¹⁸`, `5¹⁶`, `5¹⁴` blocks)
  were the hard ones; profiles with more distinct degrees closed by direct CaDiCaL.
* **Number of canonical prefixes / orbits of partial states** under the row group (dfield: 17,170
  leaves; 950,250 states → 236 orbits): a computable proxy for how much symmetry remains after the
  static break and for the cube-and-conquer cost.
* **Δ of canonizing set over DoubleLex** (S6 Table 5): when Algorithm 1 adds few permutations
  beyond the double-lex seed, double-lex is nearly complete and the case is "symmetry-easy".
* **Heavy tails** (S7b Table 5: median 4 h, mean 14 h, max 96 days over 78,892 instances of one
  shape): per-case runtime is not predictable from shape alone; budget by quantiles, not means.
* **Whether the case is "one-sided"** (all row sums distinct, or all column sums distinct): then
  ADD-1/2 is complete (BreakID Thm 2) and the case is as easy as symmetry can make it.

---

## 10. Extracted lemma ledger (all ADD-type unless noted; provability in Mathlib-free Lean 4.34 over ZarPrune)

| Name | Statement | Hypotheses | Lean note |
|---|---|---|---|
| L-act (invariance) | `Valid P (act σ τ A) ↔ Valid P A` | `σ : Perm m`, `τ : Perm n` | needs `sumFin_perm` (medium) and `HasKst_act_iff` (medium–hard: injective→increasing re-indexing); shared with the Tan notes' L-A |
| L-sortcase (ADD-0) | for every `A` there are `σ,τ` with `profileOf (act σ τ A)` having non-increasing row and column sums | none | `List.mergeSort` + `mergeSort_perm` in core; build the `Perm` from the sorted index list (medium) |
| L-transpose (ADD-T) | `s = t → (Valid P A ↔ Valid P Aᵀ)` with `Aᵀ : Mat n m` and `P` swapped | `P.s = P.t`, `m = n` for case identification | easy |
| L-doublelex (ADD-2; Flener Thm 2 / Tan 3.2 partition form) | `profileOf A = q ∧ Valid P A → ∃ A', profileOf A' = q ∧ Valid P A' ∧ DoubleLexBlocks q A'` | none beyond `Valid` invariance under block transpositions | potential `Σ 2^{i+j} a_{ij}` strictly increases on each corrective swap; strong induction; medium–hard (≈250–400 lines) |
| L-rowlex-complete (BreakID Thm 2) | rows in a block lex-chained under row-major order ⇒ at most one representative per orbit **of the row factor** | only row-factor action | not needed for soundness; useful as a "completeness" certificate; medium |
| L-notaprune (negative) | `¬ ∃ p : Prune P, p.kill = notDescending` | — | already proved (`Demo.notDescending_unsound`); generalise to any `kill` that fires on a profile obtained by permuting a profile of a valid matrix |
| L-SR (Buss–Thapen 1.15) | `C` SR w.r.t. `Γ` with substitution witness ⇒ `Γ`, `Γ ∪ {C}` equisatisfiable | `τ ⊨ C`, `Γ|α ⊢₁ Γ|τ` | already formalised in Trestle (Lean 4, S14); re-proving inside ZarPrune is unnecessary if the checker is external |
| L-dom (VeriPB Def. 13 / Prop. 14) | dominance step preserves (weak) validity | (10a), (10b), preorder proved | formalised in CakePB (HOL4/CakeML) and PBLean (Lean 4, 15 lemmas) |
| L-prefix-cube (ADD-7; dfield) | if rows `i<i'` agree on the assigned column prefix and the next column's support meets `{i,i'}` in `{i'}` only, swapping `i,i'` yields a lex-larger row-major matrix with the same prefix ⇒ WLOG supports are initial segments of each prefix-cell | row block symmetry restricted to the prefix stabiliser | per-cube permutation witness; Lean: same infrastructure as L-act plus a lex lemma; medium |
| (PRUNE, for contrast) SymCon'01 Thm 7 | class empty if some row sum exceeds `n` (or column sum exceeds `m`) | — | already `rowCap`/`colCap` in `Prunes.lean` |

---

## 11. Open questions for the thesis

1. **Which sums to fix.** dfield's column-only cases have a completely broken row group but need
   orbit covers for the hard profiles; our two-sided profiles enable stronger counting prunes but
   leave `G_q` only partially broken. Quantify on the target cells: (#cases after ADD-0) × (median
   SAT time with ADD-2) for both designs.
2. **Completeness gap in practice.** Measure, per profile, the S6 "Δ" (permutations a canonizing
   set needs beyond double-lex) for small cells (m,n ≤ 8) to see whether ADD-2 is near-complete in
   the fixed-sum regime, contrary to S2's worst case.
3. **Trust boundary for certificates.** Decide between B1 (SR/dominance steps checked by
   Trestle/CakePB/PBLean) and B2 (Lean equisatisfiability theorem + plain LRAT). B2 keeps the
   `refuted` seam at LRAT, for which Lean core has a verified checker (S18) — but that checker's
   reflective use conflicts with the current "no `native_decide`" policy; resolve the policy.
4. **Can the evolutionary search propose ADDs?** A candidate `kill` that is provably not a prune
   (fails `Prune.sound` because a valid matrix lives in a killed case) but *is* an orbit-dominance
   claim could be re-targeted as an SR witness generator. That needs a second, weaker gate:
   `Dominated : ∀ A, kill (profileOf A) = true → Valid P A → ∃ σ τ, kill (profileOf (act σ τ A)) = false`.
5. **Snake-lex vs double-lex for `K_{s,t}`-freeness.** Unknown which linearisation aligns better with
   the row-triple/`t`-subset clauses; S2/S3 report column-wise snake-lex leaves fewer solutions on
   0/1 matrices but is slower. A small experiment on `z(m,n;3,3)` cases would settle it for us.
6. **Automating SR symmetry-breaking proofs** is explicitly open (S14 §IX); for our specific
   adjacent-swap-within-block clauses the witness is mechanical (swap the two rows' variables, set
   the pivot), so a generator is feasible and would be a modest, reusable contribution.

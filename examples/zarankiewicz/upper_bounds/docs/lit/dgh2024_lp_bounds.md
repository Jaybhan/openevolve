# Davies, Gill, Horsley — Improved upper bounds on Zarankiewicz numbers

Literature notes for the upper-bound thesis (pruned case-split SAT attack, Lean-verified prunes).
Written 2026-09-21 from the full text of arXiv:2411.18842v2 (PDF, 18 pp.), plus an independent
numerical reproduction of every table entry. Everything the paper proves is marked **[paper]**;
everything I computed or inferred is marked **[mine]**.

## 0. Bibliographic record

| field | value |
|---|---|
| Authors | Sara Davies (Univ. of Queensland), Peter Gill (Monash), Daniel Horsley (Monash) |
| Title | Improved upper bounds on Zarankiewicz numbers |
| arXiv | 2411.18842 [math.CO]; v1 28 Nov 2024, v2 13 Dec 2025; 18 pages |
| Journal | Discrete Mathematics 349 (2026), article 114924, doi:10.1016/j.disc.2025.114924 |
| MSC | 05C35 (primary), 05C65 |
| Keywords | Zarankiewicz numbers, Zarankiewicz problem, linear hypergraph |
| Precursor | Chen, Horsley, Mammoliti, *Exact values for unbalanced Zarankiewicz numbers*, J. Graph Theory 106 (2024) 81–109, arXiv:2202.05507 (the s = 2 case of the same constraint family; its Lemma 2.4 is the v = 1, s = 2 case of Theorem 1.2 below) |
| Compared against | Roman (1975) bound; Tan, arXiv:2203.02283 tables for s = t ∈ {3,4} |
| Local copies | PDF and text in the session scratchpad; reproduction code in `docs/lit/dgh2024_lp_bounds_code/` |

## 1. One-paragraph summary

**[paper]** Roman's 1975 bound on z(m,n;s,t) is the optimal value of a two-constraint linear
program over the *column-size distribution* (n_{s-1}, …, n_m) of a K_{s,t}-free matrix
(hypergraph language: number of edges of each size). The paper adds a new family of linear
constraints, indexed by (v, k) with 1 ≤ v ≤ s−1 and s ≤ k ≤ m, obtained by a double count over
v-subsets of rows *plus an integrality (rounding) step*; this generalises the s = 2 family of
Chen–Horsley–Mammoliti. Solving the enlarged LP with GLPK improves the best known upper bound
by 1 (sometimes 2, once 3) in a few hundred small cells for (s,t) ∈ {(3,3),(3,4),(3,5),(4,4),(4,5),(5,5)},
m ≤ 16, n ≤ 23. Empirically the single choice v = s−1 recovers the full LP value in ≈97.5% of
cases, and combining Roman's two constraints with one v = s−1 constraint yields a closed-form
bound B_k(m,n;s,t) (Theorem 1.3), which they prove by exhibiting the LP-dual multipliers.

**[mine]** Two facts matter most for our pipeline. (a) Every one of these constraints is a
*finite counting lemma on the column-sum vector* (with a transposed copy on the row-sum vector),
so each is a candidate `Prune.kill` in ZarPrune, and the LP itself is only a way of combining
them. (b) Because the n_i are integers, the same constraints checked on integer partitions
(which is exactly what our harness enumerates) are strictly stronger than the real LP: I found
30 cells in the paper's own ranges where the *integer* program lowers the bound by one more,
each verified by exhaustive enumeration with zero survivors. That is "prune everything, run no
SAT", and it is the natural first milestone for the Lean-verified pipeline.

## 2. Dictionary: hypergraph ↔ 0/1 matrix ↔ ZarPrune

| paper | matrix | ZarPrune |
|---|---|---|
| hypergraph H, m vertices, n edges (multiset) | m×n 0/1 matrix A, rows = vertices, columns = edges | `Mat P.m P.n`, `A i j` |
| edge E of size i | column with column sum i | `colSum A j = i` |
| (s, t−1)-linear: every s-set of vertices lies in ≤ t−1 edges | no all-ones s×t submatrix | `¬ HasKst P A` |
| total degree Σ|E| | number of ones | `weight A` |
| n_i = #edges of size i | #columns with sum i | multiplicity of i in `(profileOf A).col` |
| deg_H(X) for X ⊆ V | #columns whose support contains X | not in profile (needs A) |
| WLOG no edge of size < s−1 | pad columns with < s−1 ones | an *addition*, not a prune (see §8.3, §12) |

The paper's roles are asymmetric: vertices are the part of size m (rows), edges the part of size
n (columns). Everything transposes: apply the same statements to (n, m; t, s) to get constraints
on the row-sum vector.

## 3. The linear programs, exactly

### 3.1 Roman's bound and Roman's LP **[paper, Thm 1.1 and §1]**

Theorem 1.1 (Roman). For positive integers k, s, t, m, n with s ≥ 2 and k ≥ s−1,

    z(m,n;s,t) ≤ (t−1)·C(m,s) / C(k,s−1) + (k+1)(s−1)·n / s.                      (1)

Roman's LP: variables n_{s−1}, …, n_m ≥ 0 (reals), maximise Σ_{i=s−1}^{m} i·n_i subject to

    Σ_{i=s−1}^{m} n_i = n                                                          (2)
    Σ_{i=s−1}^{m} C(i,s)·n_i ≤ (t−1)·C(m,s)                                        (3)

(The paper's §1 prints the objective as Σ n_i; §4 and the proof of Thm 1.3 use Σ i·n_i, which is
the correct one — the total degree. **[mine]** I reproduced the tables with Σ i·n_i.) (3) is the
Kővári–Sós–Turán / Guy "Argument A" count: an edge of size i contains C(i,s) s-subsets, each s-subset
lies in ≤ t−1 edges. The minimum over k of (1) equals the LP optimum (two constraints ⇒ an optimal
basic solution supported on two adjacent sizes k, k+1).

### 3.2 The new constraint family **[paper, Thm 1.2 = Lemma 3.2 (4)]**

Theorem 1.2. Let s, t ≥ 2. Let H be (s, t−1)-linear with m vertices, n edges, exactly n_i edges of
size i for i ∈ {s−1,…,m}, no edge of size < s−1. For any positive integers v, k with v < s ≤ k ≤ m:

    1/(C(k−v,s−v) − α) · Σ_{i=s−1}^{k−1} (C(i−v,s−v) − α)·C(i,v)·n_i  +  Σ_{i=k}^{m} C(i,v)·n_i
        ≤ C(m,v) · ((t−1)·C(m−v,s−v) − α) / C(k−v,s−v),                              (4)

where α is the integer with α ≡ (t−1)·C(m−v,s−v) (mod C(k−v,s−v)) and 0 ≤ α < C(k−v,s−v).

Conventions: C(a,b) = 0 when a < b; so the i = s−1 term has C(s−1−v, s−v) = 0. Denominator
C(k−v,s−v) − α > 0 always. When s = 2 the only v is 1 and (4) is the Chen–Horsley–Mammoliti family.

### 3.3 The three programs **[paper §4, §5]**

* E(m,n;s,t): maximise Σ i·n_i s.t. (2), (3), and (4) for **all** v ∈ {1,…,s−1}, k ∈ {s,…,m}.
  Optimal value is an upper bound on z(m,n;s,t) (Lemma 3.2). Implemented in SageMath/GLPK.
* E*(m,n;s,t): (2), (3), and (4) for v = s−1 only, all k ∈ {s,…,m}.
* E_k*(m,n;s,t): (2), (3), and the single constraint (8) = (4) with v = s−1 and this k.

Number of constraints in E: 2 + (s−1)(m−s+1); e.g. (s,m) = (3,15): 28 constraints, 14 variables.

### 3.4 The v = s−1 specialisation **[paper, eq. (8)]**

With v = s−1: C(k−v,s−v) = k−s+1, C(i−v,s−v) = i−s+1, C(m−v,s−v) = m−s+1, so

    Σ_{i=s−1}^{k−1} ((i−s+1−α)/(k−s+1−α))·C(i,s−1)·n_i + Σ_{i=k}^{m} C(i,s−1)·n_i
        ≤ C(m,s−1)·((t−1)(m−s+1) − α)/(k−s+1),     α ≡ (t−1)(m−s+1) (mod k−s+1).        (8)

### 3.5 Integer-safe form of (4) **[mine]**

Multiply (4) by D−α > 0, where D = C(k−v,s−v), R = (t−1)·C(m−v,s−v), α = R mod D, c = (R−α)/D ∈ ℕ:

    Σ_{i<k} (C(i−v,s−v) − α)·C(i,v)·n_i + (D−α)·Σ_{i≥k} C(i,v)·n_i ≤ (D−α)·c·C(m,v).

Terms with C(i−v,s−v) < α are negative; for a ℕ-only Lean statement move them to the right:

    Σ_{i<k} C(i−v,s−v)·C(i,v)·n_i + (D−α)·Σ_{i≥k} C(i,v)·n_i ≤ (D−α)·c·C(m,v) + α·Σ_{i<k} C(i,v)·n_i.

In profile terms (no n_i aggregation), with c_j = column sums: replace Σ_i f(i)·n_i by Σ_j f(c_j).
This is the form a `kill` predicate should evaluate.

## 4. The counting argument behind (4) **[paper §3]**

Definition 3.1 ((k,s,v)-deficiency). For an edge E and v < s: deficiency(E) = 0 if |E| ≥ k, else
C(k−v,s−v) − C(|E|−v,s−v). Interpretation: for a fixed v-subset U ⊆ E, the number of s-supersets of
U inside E that would be gained if E were enlarged to size k.

Proof of Lemma 3.2 (4). Fix v, k and a v-set X of vertices. Let n_i(X) = #edges of size i containing
X, deg(X) = Σ_i n_i(X), τ(X) = sum of deficiencies of edges containing X. Let c be the integer with
c·C(k−v,s−v) = (t−1)·C(m−v,s−v) − α.

1. Local count. The C(m−v,s−v) s-sets containing X each lie in ≤ t−1 edges; an edge of size i ⊇ X
   contains C(i−v,s−v) of them. Hence

       Σ_i C(i−v,s−v)·n_i(X) ≤ (t−1)·C(m−v,s−v) = c·C(k−v,s−v) + α.

2. Deficiency rewrite. Σ_i C(i−v,s−v)·n_i(X) ≥ C(k−v,s−v)·deg(X) − τ(X) (edges of size ≥ k contribute
   at least C(k−v,s−v); smaller ones contribute C(k−v,s−v) minus their deficiency).

3. Rounding. If deg(X) ≥ c+1 then τ(X) ≥ C(k−v,s−v)·(deg(X)−c) − α ≥ (C(k−v,s−v) − α)·(deg(X)−c),
   which gives the **local inequality**

       deg_H(X) ≤ c + τ(X)/(C(k−v,s−v) − α)                                           (5)

   (trivial when deg(X) ≤ c). The gain over the plain count is the α: because deg(X) is an integer,
   a leftover α < C(k−v,s−v) cannot be spent on a fraction of an extra edge.

4. Aggregate over all X ∈ C(V, v):

       Σ_i C(i,v)·n_i = Σ_X deg(X) ≤ C(m,v)·c + (1/(C(k−v,s−v)−α))·Σ_X τ(X),            (6)
       Σ_X τ(X) = Σ_{i=s−1}^{k−1} (C(k−v,s−v) − C(i−v,s−v))·C(i,v)·n_i.                   (7)

   Substituting (7) in (6) and consolidating the i < k terms gives (4).

**[mine]** Note what the profile does and does not see. (5) is a statement about one v-set X; the
column-sum profile only knows Σ_X deg(X) and Σ_X τ(X), so (4) is the best profile-level consequence
of (5). But for v = 1, X = {row i}, deg(X) = r_i *is* in the profile (the row sum), and step 1 for
v = 1 reads Σ_{j : A i j = 1} C(c_j − 1, s−1) ≤ (t−1)·C(m−1, s−1) — this is exactly Tan/Guy's
"Argument D", a genuine (row, column) cross-constraint. See §9 and Lemma L4.

## 5. Theorem 1.3: closed form, proved as an LP-duality certificate **[paper §5]**

Theorem 1.3. Let 2 ≤ s ≤ m, 2 ≤ t ≤ n, and max{2, s²−2s} ≤ k ≤ m. Then z(m,n;s,t) ≤ B_k(m,n;s,t),

    B_k(m,n;s,t) = [C(m,s−1)/C(k,s−1)] · ( ((t−1)(m−s+1)/s)·(β(k+1)/(k−s+1) + 1) − αβ(k−s+2)/(k−s+1) )
                   + ((k+1)(s−1−β)/s)·n,

    α ≡ (t−1)(m−s+1) (mod k−s+1), 0 ≤ α < k−s+1,
    β = ((s−1)(k−s+1) − α(s−1)) / ((k+1)(k−s+1) − α(s−1)).

For fixed s, t, m each k gives a linear function of n. (Read directly from the typeset page; I
verified numerically that this reading reproduces every non-starred table entry.)

Proof mechanism (this is the part to imitate in Lean). Take A·(2) + B·(3) + C·(8) with

    A = (k+1)(s−1−β)/s,   B = (1 − (s−1)β)·C(k,s−1)^{-1},   C = (s−1)β·C(k,s−2)^{-1} = (k−s+2)β·C(k,s−1)^{-1}.

Since (s−1)/((k−s)(k−s+2)+k+1) ≤ β ≤ (s−1)/(k+1), A, C ≥ 0 always and B ≥ 0 when k ≥ s²−2s (this
is where the hypothesis comes from). The combination is Σ_{i<k} f(i)n_i + Σ_{i≥k} g(i)n_i ≤ B_k with

    f(i) = (k+1)(s−1−β)/s + [C(i,s−1)/C(k,s−1)]·( (1−(s−1)β)(i−s+1)/s + β(k−s+2)(i−s+1−α)/(k−s+1−α) ),
    g(i) = (k+1)(s−1−β)/s + [C(i,s−1)/C(k,s−1)]·( (1−(s−1)β)(i−s+1)/s + β(k−s+2) ).

Claim 5.1: f(k−1) = k−1 and f(i) ≥ i on {s−1,…,k−2} (via f' ≤ 1, monotone in α, computer-aided
simplification). Claim 5.2: g(k) = k, g(k+1) = k+1, g convex ⇒ g(i) ≥ i for i ≥ k+2. Hence
Σ i n_i ≤ Σ f n + Σ g n ≤ B_k. A, B, C are the unique multipliers making f(k−1)=k−1, g(k)=k, g(k+1)=k+1.

**[mine]** This is a Farkas certificate: nonnegative multipliers on the constraints whose
combination dominates the objective coefficientwise. For a *fixed* instance the pointwise checks
"f(i) ≥ i, g(i) ≥ i" are finitely many rational inequalities, so an instance-level Lean proof
needs only: the constraint lemmas + the multipliers + a `decide`-style check over i ∈ [0, m].
The symbolic Claims 5.1/5.2 are only needed for the *closed form for all parameters*.

## 6. Context the paper gives **[paper §1–2]**

* Roman points: for fixed s, t, m the best Roman bound is piecewise linear in n with breakpoints
  n = (t−1)C(m,s)/C(ℓ,s), value ℓ(t−1)C(m,s)/C(ℓ,s), ℓ ≥ s. At an integral Roman point the bound is
  attained iff an s-(m,ℓ,t−1) design exists. Čulík: (1) with k = s−1 is exact for all n ≥ (t−1)C(m,s).
  The improvements of (4) cluster near Roman points (Appendix B plots, m ∈ {21,…,24}).
* Kővári–Sós–Turán: z < (t−1)^{1/s} m n^{1−1/s} + (s−1)n. Nikiforov (2010), k ∈ {0,…,s−2}:
  z < (t−k−1)^{1/s} m n^{1−1/s} + (s−1)n^{1+k/s} + km. For all tabulated cells Roman ≥ Nikiforov
  was checked, so Roman was the incumbent everywhere an improvement is claimed.
* Open (their §6): whether the new bounds are achieved for s > 2; closed forms using several (8)
  constraints at once.

## 7. Computational results in the paper **[paper §4, App. A]**

Table 1 (m ∈ {s,…,60}, n ∈ {m,…,60}): number of cells where E* = E, and where Theorem 1.3 = E.

| s | t | cases | E* matches E | Thm 1.3 matches E |
|---|---|---|---|---|
| 3 | 3 | 1711 | 1697 | 1334 |
| 3 | 4 | 1711 | 1696 | 1354 |
| 3 | 5 | 1711 | 1693 | 1455 |
| 4 | 4 | 1653 | 1597 | 618 |
| 4 | 5 | 1653 | 1565 | 797 |
| 5 | 5 | 1596 | 1538 | 209 |

Any single v ≠ s−1 matches E in only ≈30% of cases.

Tables 2–7: cells where E (or Thm 1.3) beats the best previously known bound, which was Roman's
in every listed cell. Plain = improvement by 1, **bold** = by 2, **_bold underlined_** = by 3,
`*` = the full LP E was needed (Theorem 1.3 inapplicable or weaker). Transcribed from word
coordinates of the PDF and cross-checked against my LP (every value agrees).

Table 2, z(m,n;3,3):

| m\n | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|
| 10 | | | | | | 111 | 115 |
| 11 | | | 108 | 112 | 116 | | |
| 13 | 116 | 121 | 125 | 130 | 135 | | |
| 14 | 124 | 129 | 135 | 140 | 145 | 150 | |
| 15 | **132** | 138 | 143 | 149 | 154 | | 165 |
| 16 | 141 | **146** | **152** | **158** | 164 | 169 | 175 |

(Cells (11,17) → 99 and (11,18) → 103 also beat Roman by 2 but are absent because Tan's table
already had 96 and 101 there. **[mine]** from Tan's z_3 table.)

Table 3, z(m,n;3,4): m=5: n=7:27, 8:30, 9:33. m=8: 8:47. m=9: 13:76, 14:80, 15:84, 16:88.
m=10: 10:69, 18:107*. m=11: 14:97, 15:102, 17:111, 18:116, 21:130, 23:139.
m=13: 14:113, 15:119, 16:125, 17:130, 22:157, 23:162. m=14: 14:122, 15:128, 16:134, 17:140,
18:146, 19:152, 21:163*. m=15: 16:143, 17:149, 23:185. m=16: 16:152, 17:159, 20:179*, 21:185*.

Table 4, z(m,n;3,5): m=6: 8:39, 20:79*. m=7: 12:61, 13:65, 14:69, 15:72. m=9: 9:63, 10:68.
m=10: 10:75, 13:91, 14:96, 15:101. m=11: 20:136, 21:141. m=12: 14:114, 15:120, 16:126.
m=13: 21:165, 23:176. m=14: 16:146, 18:159, 19:165*. m=15: 15:149, 16:156, 21:189, 22:**195**,
23:202. m=16: 16:166.

Table 5, z(m,n;4,4): m=10: 15:107, 16:113*, 17:119*, 18:**124***, 19:130*, 20:135*, 21:140*,
22:146*, 23:151*. m=11: 14:**110**, 15:117. m=12: 14:120, 18:147, 19:153, 20:**159**, 21:**166**,
22:**172**, 23:**178**. m=13: 15:137, 17:152*. m=14: 14:**138**, 15:**146**, 16:154, 17:162, 19:177,
20:184, 21:**191**, 22:**198**, 23:**205**. m=15: 15:156, 16:165, 20:197, 21:205, 22:212, 23:220.
m=16: 16:**175***, 17:184, 18:193, 19:201, 20:209, 21:217, 22:**225**, 23:**233**.

Table 6, z(m,n;4,5): m=7: 9:53*, 10:58*, 12:67*. m=8: 16:97*, 17:102*, 18:107*, 19:112*, 20:117*,
21:121*, 22:126*. m=10: 10:81, 11:88, 12:**94**, 13:101, 14:107, 15:113, 23:161*. m=11: 11:96,
12:103, 13:110, 14:117, 17:137, 18:**143**, 19:150, 20:156, 21:162, 22:168. m=12: 13:120, 15:135*,
16:142*, 17:149*. m=13: 13:**129**, 14:**137**, 15:145, 16:153, 18:168, 19:175, 20:**182**, 21:190,
22:**197**, 23:**204**. m=14: 15:156, 19:**188**, 20:196, 21:204, 22:211, 23:219.
m=15: 15:167, 16:176, 18:193, 19:201, 22:226, 23:234 (see typo note). m=16: 16:187, 18:205,
19:214, 20:**222**, 21:**231**, 22:**_239_** (Roman 242, the unique improvement by 3), 23:**248**.

Table 7, z(m,n;5,5) (all starred unless noted): m=8: 10:69, 11:75, 12:81, 14:92. m=9: 19:135,
20:141, 21:147, 22:**153**, 23:**159**. m=10: 18:143, 19:149. m=11: 12:109, 13:117, 14:**124**,
15:**132**, 16:140, 17:147, 18:155, 19:162. m=12: 12:119, 13:**127**, 14:**135**, 15:144, 16:152,
17:160, 18:168, 19:176, 21:192, 22:200, 23:**207**. m=13: 17:173*, 20:199 (printed without star).
m=14: 14:157, 15:**166**, 16:176, 17:**185**, 18:**194**, 19:204, 20:213. m=15: 15:**178**,
16:**188**, 17:**197**, 18:**207**, 19:217, 20:227, 21:237, 23:256. m=16: 16:200, 17:211, 18:221,
20:242, 21:252, 22:263.

Typos/slips I noticed **[mine]**: (i) Table 6 row m = 15 is printed four columns too far left
(its values 167,176,193,201,226,234 are E at n = 15,16,18,19,22,23; at the printed columns
n = 11,12,14,15,18,19 the LP gives 130,140,158,167,193,201). (ii) Table 7 cell (13,20) = 199 lacks
its asterisk: Theorem 1.3 needs k ≥ s²−2s = 15 > m, and E* = 200 ≠ E = 199 there. (iii) §1 states the
Roman-LP objective as Σ n_i; it should be Σ i·n_i.

Cells where E* ≠ E (i.e. some v < s−1 constraint is essential) **[mine]**: (10,18;3,4), (14,21;3,4),
(16,20;3,4), (16,21;3,4), (6,20;3,5), (14,19;3,5), (13,17;4,4), (10,23;4,5), (12,15;4,5),
(12,16;4,5), (12,17;4,5), (8,10;5,5), (10,18;5,5), (10,19;5,5), (13,20;5,5), (16,21;5,5), (16,22;5,5).
All other stars are Theorem 1.3 being inapplicable (m < s²−2s) or weaker than E*.

## 8. Reproduction and extensions **[mine]**

Code: `docs/lit/dgh2024_lp_bounds_code/` (`dgh_lp.py` LP/IP/closed form with scipy HiGHS and
exact Fractions for coefficients; `branches.py` exact integer enumeration of column partitions;
`lo0.py`, `verify45.py`, `duals.py`, `farkas_argD.py`; `tables_repro_output.txt`).

### 8.1 Reproduction
For all (s,t) and all m ≤ 16, n ≤ 23 in the paper's ranges, floor(E) agrees with every printed
table entry, bold marks agree with Roman − E ≥ 2, and star marks agree with "closed form ≠ E".
The transposed LP E(n,m;t,s) (rows as edges) was never better than E(m,n;s,t) in these ranges.

### 8.2 Integer program: 30 more cells improve by one
The n_i are counts, so requiring integrality is sound. With scipy `milp` and then, independently,
exhaustive enumeration of all integer column partitions at total w = floor(E) (parts in [s−1, m]),
the following cells have **zero** integer-feasible distributions at the LP value, hence
z ≤ floor(E) − 1. (Rigor: any K_{s,t}-free matrix with ≥ floor(E) ones pads to one with all column
sums ≥ s−1 and ≥ floor(E) ones; its distribution satisfies (2)–(4) with Σ i n_i ≤ floor(E) by the
LP bound, so Σ i n_i = floor(E), contradiction. For 27 of the 30 cells the LP value equals Roman's
bound, a published theorem; for the three (3,3) cells it is the paper's published LP value.)

| (s,t) | cell → new bound (previous: Roman/DGH) |
|---|---|
| (3,3) | (10,23) → 114 (115), (13,18) → 120 (121), (14,19) → 134 (135) |
| (3,4) | (7,11) → 53 (54), (10,11) → 74 (75), (11,11) → 81 (82), (11,16) → 106 (107), (13,19) → 141 (142), (15,18) → 155 (156) |
| (3,5) | (6,9) → 42 (43), (8,11) → 65 (66), (11,19) → 131 (132), (11,23) → 150 (151), (12,17) → 131 (132), (13,20) → 159 (160), (13,22) → 170 (171) |
| (4,4) | (11,16) → 123 (124), (13,14) → 129 (130), (13,16) → 144 (145) |
| (4,5) | (9,14) → 97 (98), (10,13) → 100 (101), (13,21) → 189 (190), (14,18) → 180 (181), (14,21) → 203 (204), (15,20) → 209 (210), (16,17) → 196 (197) |
| (5,5) | (13,21) → 207 (208), (14,19) → 203 (204), (16,22) → 262 (263), (16,23) → 272 (273) |

Survivor counts one below (at w = floor(E) − 1) are small (4–24 column partitions), i.e. these
cells are exactly one unit past the counting frontier. **Caveat:** "new" is relative to Roman/DGH
and Tan's tables; Tan remarks that partition elimination with his Argument I (using exact values of
smaller cells) already gives e.g. z(11,14;4,4) ≤ 106, better than DGH's 110, so a literature check
against Tan's remark, Guy, and the 2026 (3,3) papers (Padhi, Hou, Afrasyab) is required before
claiming any of these. They are, however, sound bounds.

### 8.3 The constraints do not need the "no small columns" WLOG
Re-deriving §4 with C(a,b) := 0 for a < b (including a < 0) and columns of size 0..m shows (3) and
(4) hold for arbitrary profiles (columns of size < v contain no v-set; columns of size v..s−1
contain X but no s-superset, contributing full deficiency). Checked: enumerating partitions with
parts in [0, m] gives the same survivor sets (e.g. (10,23;3,3) w=115: 1,421,530 partitions, 0
survivors; w=114: 10 survivors, none with a column < 2). So `kill` can be stated on all of
`Profile`, with no padding assumption — good, since in ZarPrune padding is an addition, not a prune.

### 8.4 Branch-level pruning power (column side)

| cell, w | column partitions | pass (3) only | pass (3)+(4), v=s−1 | pass (3)+all (4) |
|---|---|---|---|---|
| z(15,17;3,3), w=134 (Roman) | 1,797,374 | 1 | 0 | 0 |
| z(15,17;3,3), w=133 | 1,765,005 | 9 | 0 | 0 |
| z(15,17;3,3), w=132 (= DGH) | 1,730,351 | 38 | 14 | 14 |
| z(10,22;3,3), w=112 | 71,248 | 1 | 0 | 0 |
| z(10,22;3,3), w=111 | 68,260 | 4 | 4 | 4 |
| z(10,23;3,3), w=115 | 84,152 | 3 | 0 | 0 |
| z(10,23;3,3), w=114 | 80,448 | 11 | 10 | 10 |
| z(11,14;4,4), w=111 | 5,914 | 4 | 0 | 0 |
| z(11,14;4,4), w=110 | 6,254 | 11 | 6 | 6 |

Row side for z(15,17;3,3), w=132 (rows as edges over 17 column-vertices, (t,s) roles):
2,311,544 row partitions, 1,716 pass (3), 1,688 pass all (4). The side with the smaller vertex
set (here rows, m=15 < n=17) is where the counting bites; the other side is nearly free.

### 8.5 Argument D (v = 1 local inequality) as a cross-prune
For z(15,17;3,3), w=132: 14 × 1,688 = 23,632 (row, column) partition pairs. Tan's Argument D
(row with most ones r_max: the r_max smallest values of C(c_j−1,2) must sum to ≤ 2·C(14,2) = 182;
transposed: 2·C(16,2) = 240) leaves 4,717 pairs (row form) and 4,312 pairs (both forms) — an 82%
cut that no column-only or row-only argument can see. Surviving pairs per column partition are
828 / 203 / 122 depending on whether the partition has max column sum 9 or 10.

### 8.6 An exact Farkas certificate (what an instance-level Lean prune consumes)
E(15,17;3,3) = 132.74 = 6637/50. Binding constraints: (4) with (v,k) = (2,8) (rhs 420) and (2,15)
(rhs 210); multipliers y_eq = 138/25, y_8 = 1/25, y_15 = 221/2100. Pointwise,
y_eq + y_8·a_8(i) + y_15·a_15(i) − i ≥ 0 for all i ∈ [2,15] (equality at i = 7,8,9), and
y_eq·17 + y_8·420 + y_15·210 = 6637/50. Hence weight ≤ 132.74 for every K_{3,3}-free 15×17 matrix,
with no LP solver in the trusted base — only the two constraint lemmas and rational arithmetic.

## 9. Relation to Tan (2022) / Guy's arguments **[mine, from Tan §2–3]**

* Tan's Argument A (Σ_j C(c_j,a) ≤ (b−1)C(m,a)) = constraint (3). His Algorithm 1 enumerates
  non-increasing column partitions and prunes prefixes by A and by Argument I (inclusion: the
  first n' columns carry ≤ z(m,n') ones, using *known* smaller values).
* Argument B (Jensen: the binomial sum is minimised by balanced sums) is what the LP relaxation
  exploits automatically.
* Argument D (row with ones in columns of sums c_1..c_r ⇒ Σ C(c_i−1, a−1) ≤ (b−1)C(m−1, a−1)) is
  DGH's step-1 local count with v = 1 and X = one row. DGH aggregate it with rounding into (4)
  v=1, which they find the least useful choice in the LP; Tan uses it pointwise on partition
  *pairs*, where it is strong (§8.5). These are complementary, not redundant.
* DGH's (4) for v = s−1 is new relative to Tan: it is not implied by A, D, I. Tan's prefix pruning
  relies on A being monotone under "most pessimal" extension; (4) has negative coefficients for
  sizes with C(i−v,s−v) < α, so as a prefix test it needs a completion bound (or apply it only to
  complete partitions).
* Argument I imports previously proved bounds. In Lean that means smaller bounds as verified
  theorems or explicit hypotheses of the prune — a "bound table as assumptions" design.

## 10. Pruning lemma catalogue (hypotheses, statement, Lean notes)

Common setting: `P : Params`, `A : Mat P.m P.n`, hypothesis `hfree : ¬ HasKst P A`, `c_j = colSum A j`,
`r_i = rowSum A i`. `choose` is not in Lean core 4.34 (`Nat.choose` is Mathlib; checked with
`lake env lean`), so a Pascal-recursion `choose` must be defined in ZarPrune. `Nat.div_add_mod`,
`Nat.mod_lt`, `omega` are available.

**L0 (KST / Argument A / constraint (3)).** Hyp: ¬HasKst. Statement:
`sumFin n (fun j => choose (colSum A j) s) ≤ (t−1) * choose m s`. Transposed: Σ_i choose(r_i, t) ≤ (s−1)·choose(n, t).
Lean: needs the subset-counting layer — (a) `countFin`/`choose` with `choose (countFin p) r` = number
of `Incr` tuples `Fin r → Fin m` landing in `p`; (b) from "≥ t columns contain the s-tuple R" build an
`Incr` `Fin t → Fin n` (else `HasKst`); (c) `sumFin_swap`-style Fubini between tuples and columns.
Honest estimate: a few hundred lines, one-off human/LLM-assisted library work, not something to evolve.
Everything below reuses this layer.

**L1 (DGH (4), v = s−1, profile form).** Hyp: ¬HasKst, s ≤ k ≤ m. With D = k−s+1, R = (t−1)(m−s+1),
α = R % D, c = R / D:
`Σ_j [c_j<k]·(c_j−s+1)·choose(c_j,s−1) + (D−α)·Σ_j [c_j≥k]·choose(c_j,s−1) ≤ (D−α)·c·choose(m,s−1) + α·Σ_j [c_j<k]·choose(c_j,s−1)`.
Lean: layer + per-X count (needs "#rows of column j outside X = c_j − (s−1)", an injectivity/count
lemma for `Incr` tuples) + rounding step (pure arithmetic; with concrete `P` it is `omega`-shaped after
`Nat.div_add_mod`) + aggregation (Fubini, and #X = choose(m, s−1)). Roughly 1.5× L0.

**L2 (DGH (4), general v).** Same with `choose (c_j − v) (s − v)`; the per-X count is over (s−v)-tuples
in the complement of X inside column j. Roughly 2× L0. Only needed for the 17 starred cells listed in §7.

**L3 (Farkas / LP prune, instance level).** Hyp: L0–L2 instances with multipliers y ≥ 0 (rationals,
clear denominators to ℕ) such that Σ_c y_c·a_c(i) ≥ i for all i ∈ [0, m]. Conclusion:
weight A ≤ Σ_c y_c·b_c, i.e. the whole instance is refuted when that is < w. Lean: linear combination
of already-proved instances + a finite `decide` over i. Easy once L0–L2 exist. Theorem 1.3 is the
symbolic version; not worth formalising.

**L4 (Argument D / (5) with v=1, cross-prune).** Hyp: ¬HasKst. For each row i:
`Σ_{j : A i j = true} choose (c_j − 1) (s−1) ≤ (t−1)·choose (m−1) (s−1)`; the kill uses the sum of the
r_i smallest values of choose(c_j−1, s−1) over all columns (pessimal choice). Lean: layer applied to
the (m−1)-row submatrix, plus a rearrangement lemma "sum over any r-subset ≥ sum of the r smallest"
(sorting; moderate). A weaker but trivial variant uses r_i · min_j choose(c_j−1, s−1).

**L5 (per-X rounding, v = 1 pointwise).** (5) with X = {i}: `(D−α)·r_i ≤ (D−α)·c + τ_i` where
τ_i sums deficiencies of the columns containing row i. Profile-level kill again needs the pessimal
choice; gain over L4 is only the α, likely marginal. Low priority.

**L6 (integrality / IP).** Not a separate lemma: it is L0–L2 applied per enumerated partition, plus
the `cover` obligation that the enumeration is complete. The 30 cells of §8.2 are exactly this with
an empty survivor list, i.e. `upper_bound_of_cover` with `survivors = []`.

Symmetry breaking (sorting, "no column < s−1") is *not* in this list: as the ZarPrune README says
and DGH's own WLOG illustrates, those are additions needing a witness, not prunes.

## 11. LP slack / dual values as a difficulty or pruning signal **[mine, hypotheses]**

What the LP gives per cell: optimum E, an optimal fractional profile (e.g. (15,17;3,3):
n_7 ≈ 7.04, n_8 ≈ 6.18, n_9 ≈ 3.78 — the "design-like" shape), the binding constraints, dual
multipliers y, and reduced costs ρ_i = Σ_c y_c a_c(i) − i ≥ 0 per column size. For any feasible
integer partition with total w: Σ_i ρ_i n_i + Σ_c y_c·slack_c = E − w exactly. So E − w (< 1 for the
frontier attempt w = floor(E)) is a budget that each survivor spends on off-basis sizes and on slack
in the binding constraints.

Candidate signals for a branch (column partition, or pair):

1. *Min normalised slack* over all counting constraints. Survivors at (15,17;3,3), w=132 have
   min-slack in [0, 0.0154], median 0.004; the tight constraint is (3) for 8 of 14 and (4) v=2,k=8
   for 6. Slack 0 means the branch saturates a count exactly (every (s−1)-set of rows has the
   "design" degree) — these are the candidates for actual extremal configurations and are plausibly
   the hardest for SAT; positive slack means some counting room and, heuristically, easier UNSAT.
   Untested; the first experiment is to correlate solver time on the 14 × 1,688 pairs with this.
2. *Dual-weighted distance* Σ_i ρ_i n_i: how far the partition is from the LP-optimal shape. Near
   zero = LP-extremal.
3. *Survivor count* itself as a per-cell difficulty proxy (e.g. 14 vs 1,688 vs 4,312 pairs).
4. *Prune yield* for the reward function: number of previously surviving partitions (or pairs) killed
   by a candidate prune, weighted by signal 1 or 2, so that killing near-tight branches is worth more
   than killing branches the SAT solver would dispatch instantly.
5. *Which constraint binds* tells the evolutionary search where to look: at (16,22;4,5) the binding
   pair is (3) and (4) v=3,k=11; at (15,17;3,3) it is (4) with k=8 and k=15 and (3) is slack.

Warning: the LP is an upper bound on Σ i n_i, so "LP says infeasible" is a sound kill, but LP *values*
as difficulty are heuristics; nothing here should touch the trusted base.

## 12. Design implications for the OpenEvolve + ZarPrune pipeline

1. **Milestone 0 (no SAT):** implement L0 and L1 as `Prune`s, enumerate integer partitions, and
   reproduce the 30 cells of §8.2 plus all of DGH's tables with an empty survivor list. That is a
   fully Lean-verified deliverable and the natural sanity check for `cover`.
2. **The subset-counting layer is the critical path.** L0 (KST) is already the README's "next
   target"; L1, L2, L4 all reuse it. Build it once by hand; do not expect the evolutionary loop to
   produce it.
3. **State prunes on the raw profile**, not on n_i and not assuming column sums ≥ s−1 (§8.3).
4. **Cross-prunes are the untapped room.** DGH's LP has no (row, column) coupling; Argument D (L4)
   removes 82% of pairs at (15,17;3,3). The search space for novel prunes is "local inequality (5)
   combined with the other side's profile" and Argument-I-style reuse of proved smaller bounds.
5. **Per-branch checks are the IP, and the IP beats the LP.** Never replace enumeration by the LP;
   use the LP only as a cheap pre-check (E < w ⇒ skip the cell) and for the signals of §11.
6. **Prefix pruning:** (3) is safe as a prefix test in Tan's enumeration order; (4) is not (negative
   coefficients), so evaluate it on complete partitions or derive a completion bound.
7. **Targets:** the (3,3) frontier at m ∈ {10,…,16}, n ∈ {17,…,23} is where DGH improved Tan's
   Roman entries by 1–2 and where the IP adds three more; z(15,17;3,3) ≤ 131 (attempting to beat
   132) has only 14 column partitions and ≈4,300 pairs after L0/L1/L4 — a realistic first SAT target.
8. **Reward shaping:** (kills weighted by §11 signals) + (Lean elaboration success); the LP dual
   tells which constraint family a near-miss candidate resembles.
9. **Trusted base stays tiny:** all of the above is counting on `Fin`, `choose`, and `Nat` arithmetic;
   no Mathlib and no LP solver in the kernel.

## 13. Open questions

* Do any of the 30 IP cells collide with better bounds from Tan's partition elimination (he only
  reports z(11,14;4,4) ≤ 106 as an example), Guy's tables, or the 2026 (3,3) papers?
* Can (5) be used pointwise for v = s−1 with row-side information (needs deg(X) for (s−1)-sets of
  rows, which the profile does not carry — would a richer case index, e.g. row-pair co-degrees, pay
  for itself)?
* Is there a joint LP over (row distribution, column distribution) with valid coupling constraints
  (Argument D aggregated over rows; Gale–Ryser-type feasibility) that closes more cells?
* Does LP tightness (§11 signal 1) actually predict SAT hardness? (Experiment: 23,632 pairs at
  (15,17;3,3).)
* DGH's own: exactness of the new bounds for s > 2; closed forms with several k at once.
* Why is v = s−1 so dominant (97.5%)? A structural explanation might suggest which other local
  counts are worth aggregating.

## 14. Files

* This note: `docs/lit/dgh2024_lp_bounds.md`
* Code and raw output: `docs/lit/dgh2024_lp_bounds_code/` (`dgh_lp.py`, `branches.py`, `lo0.py`,
  `verify45.py`, `duals.py`, `farkas_argD.py`, `tables_repro_output.txt`)
* Paper PDF copy: session scratchpad `lit/dgh2024.pdf` (arXiv:2411.18842v2); Tan's PDF is in
  `docs/papers/tan2022_sat_zarankiewicz.pdf`.

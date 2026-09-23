# Classical Zarankiewicz upper-bound theory: a catalogue of counting inequalities that prune (row-partition, column-partition) cases

Literature notes for the MEng thesis *Discovering upper bounds for Zarankiewicz numbers z(m,n;s,t)* (OpenEvolve + Lean 4 `ZarPrune` gate). Compiled 2026-09-21.

Companion script (census of which inequality kills which cases, reproduces §5.2): `docs/lit/zarankiewicz_theory_counting_census.py` (about 3 minutes; Roman-only minor table).

---

## 0. Conventions (read first; the literature is inconsistent)

* Matrix `A` is `m × n`, 0/1. We forbid an all-ones `s × t` submatrix: **`s` rows, `t` columns**. This is the `ZarPrune` convention (`HasKst`: `R : Fin s → Fin m`, `C : Fin t → Fin n`), the convention of Tan 2022 (`z_{a,b}(m,n)`, `a` rows, `b` columns), of Damásdi–Héger–Szőnyi (`Z_{s,t}(m,n)`, `s` nodes in the class of size `m`), and of Davies–Gill–Horsley. `z(m,n;s,t)` is the max number of ones. Note `z(m,n;s,t) = z(n,m;t,s)` and a `K_{s,t}`-free matrix need not be `K_{t,s}`-free.
* Row sums `r_1..r_m` (row `i` has `r_i` ones), column sums `c_1..c_n`, weight `w = Σ r_i = Σ c_j`.
* Hypergraph view (Roman, DGH): rows = vertices, columns = edges of sizes `c_j`. `K_{s,t}`-free ⇔ every `s`-set of rows lies in at most `t−1` columns ⇔ the hypergraph is "(s, t−1)-linear" (DGH's term).
* Guy/Zarankiewicz `k_{s,t}(m,n) = z(m,n;s,t) + 1` (min number of ones forcing the submatrix). OEIS A001197/A001198/A006613 store `k`, not `z`.
* "Case" = a pair (multiset of row sums, multiset of column sums) that Tan's decomposition hands to the SAT solver. A prune on `Profile` vectors restricts to a prune on unordered partitions (lean/README.md).
* Binomials: `C(x,k) = 0` for `x < k`, `C(−1,k) = 0`. Every inequality below is an integer inequality (no real analysis needed for soundness).

---

## 1. Sources consulted

| # | Source | Bibliographic data | Access | What it gave |
|---|---|---|---|---|
| S1 | Guy 1969, "A many-facetted problem of Zarankiewicz" | *The Many Facets of Graph Theory*, LNM 110, Springer, pp. 129–148, doi:10.1007/BFb0060112 | **not accessible** (Springer paywall; Semantic Scholar has no abstract; no scan found) | Only via secondary sources: Tan 2022 (Arguments A, B, D, I; Theorem 2.2 attribution; Guy's `T`-function), DHS 2013 / Héger thesis ("Guy [23], p138, point C"; Theorem 3.3 "Guy"), OEIS. Tan's tables correct 8 errors in Guy's `z_2(m,n)` table (all too low by 1). |
| S2 | Guy 1968, "A problem of Zarankiewicz" | *Theory of Graphs* (Tihany 1966), Academic Press, pp. 119–150 | not accessible | Referenced by OEIS A001197/A001198 as the source of `k_2(n)`, `k_3(n)` tables; Tan cites it ([8]) for `T_{2,2}(m)`, `T_{2,b}(m)`. |
| S3 | Guy–Znám 1969, "A problem of Zarankiewicz" | *Recent Progress in Combinatorics* (Waterloo 1968), Academic Press, pp. 237–243 | not accessible | Bibliographic only. |
| S4 | Roman 1975, "A problem of Zarankiewicz" | JCTA 18, 187–198, doi:10.1016/0097-3165(75)90007-2 | **not accessible** (ScienceDirect 403 to scripts; S2 abstract elided) | Theorem quoted identically by Tan (Thm 2.2), DHS (Thm 3.5), DGH (Thm 1.1); I re-derived it (§4, P4). Roman's equality characterisation from DHS. |
| S5 | Čulík 1956 | Ann. Polon. Math. 3, 165–168 | not accessible (IMPAN page only) | Theorem quoted by Tan (Cor. 2.1), DHS (Thm 3.2), Balbuena et al. (eq. (1)); all three agree. |
| S6 | Reiman 1958 | Acta Math. Acad. Sci. Hung. 9, 269–273 | not accessible | Bound quoted by DDR 2013 (Thm 1) and DHS (Thm 3.1); re-derived from Argument A in §2. |
| S7 | Kővári–Sós–Turán 1954 | Colloq. Math. 3, 50–57 | not accessed directly | Bound via Wikipedia raw wikitext, Collins et al., Füredi. |
| S8 | Füredi 1996, "An upper bound on Zarankiewicz' problem" | CPC 5, 29–33, doi:10.1017/S0963548300001814 | **read** (Cambridge PDF, via MIT access; text partly garbled by OCR) | Theorem 2, Lemmas 1–2, proof structure (fix `t−2` rows; codegree counting). |
| S9 | Damásdi–Héger–Szőnyi 2013, "The Zarankiewicz problem, cages, and geometries" | Ann. Univ. Sci. Budapest. Sect. Math. 56, 3–37 (author PDF at heger.web.elte.hu) | **read** (whole Zarankiewicz section) | Reiman/Čulík/Guy/Roman statements, Roman's inequality (Thm 3.4), equality structure, Theorem 3.15 (deletion argument), Prop 3.20 (recursive bound), Props 3.23–3.29, exact values near designs. |
| S10 | Héger 2013 PhD thesis "Some graph theoretic aspects of finite geometries", ch. 6 | ELTE, heger.web.elte.hu/publ/HTdiss-e.pdf | **read** | Same results as S9 plus: `Z_{2,2}(14,25)=80`, `Z_{2,2}(14,24)=78` (Guy's table wrong there), Table of best upper bounds `Z_{2,2}(m,n)` for `7≤m≤31`, Illés–Krarup realisability conjecture refuted, Questions 6.3.1–6.3.3 on near-regularity, empirical `Z_{t,t}(m,n) ≤ Z_{t,t}(m+1,n−1)` for `m+2 ≤ n`. |
| S11 | Goddard–Henning–Oellermann 2000, "Bipartite Ramsey numbers and Zarankiewicz numbers" | Discrete Math. 219, 85–95, doi:10.1016/S0012-365X(99)00370-2 | **not accessible** (403; UJ repository gives abstract only) | Abstract: exact `z(s,2)` small values, general bounds, `b(m,n)` for `m,n ≤ 6`. Contents relayed by DDR 2013 (they found `z(18)=81`, and that every extremal graph for `z(18)` has degree sequence `n_4 = n_5 = 9` on both sides) and Collins et al. (Lemmas 2–4 "may be found in various forms in … [12]"). |
| S12 | Afzaly–McKay, `z(n;2)` exhaustive, all `n ≤ 31` | "personal communication, 2015" in Collins et al.; McKay's data page (extremal.html) has C4-free *graphs* up to 49 vertices, not bipartite `z` tables | **no primary document exists online** | Table 3 of Collins et al. (values reproduced in §5). |
| S13 | Collins–Riasanovsky–Wallace–Radziszowski 2016, "Zarankiewicz numbers and bipartite Ramsey numbers" | arXiv:1604.01257 | **read** | Star-counting lemma, density lemma, Lemma 4 (`z_bound` algorithm), backwards-path enumeration counts, tables of `z(m,n;s)` bounds for `s=3..6`, `6≤m≤n≤18`. |
| S14 | Dybizbański–Dzido–Radziszowski 2013/2015, "On some Zarankiewicz numbers and bipartite Ramsey numbers for quadrilateral" | arXiv:1303.5475, Ars Combin. 119 (2015) 275–287 | **read** | Theorem 1 (Bollobás summary), Theorems 2–4 (`z(k²+k+1−h)` for `h ≤ 3`), Lemma 6 (`z(17)=74`, an edge-local argument), Table 1 (`z(n;2)`, `n ≤ 21`). |
| S15 | Davies–Gill–Horsley 2024/2026, "Improved upper bounds on Zarankiewicz numbers" | arXiv:2411.18842v2, Discrete Math 349 (2026) | **read** (text in scratchpad `dgh2024.txt`) | Roman's LP, new constraint family (4) (Lemma 3.2), closed form Theorem 1.3, tables of improvements. |
| S16 | Balbuena–García-Vázquez–Marcote–Valenzuela 2008, "Extremal K(s,t)-free bipartite graphs" | DMTCS 10:3, 35–48 | **read** | Remark 3.1 (complement reformulation), Lemma 3.1, Theorem 2.1/2.2 (dense regime exact values + extremal families). |
| S17 | Tan 2022, "An attack on Zarankiewicz's problem through SAT solving" | arXiv:2203.02283v2; code: github.com/… "Kyoto" (cloned in scratchpad) | **read** (paper + code) | Arguments A, B, D, I; Roman Thm 2.2 with equality region via `T_{a,b}(m)`; Algorithm 1; `utilities.py` shows exactly how A, D, "E" are applied to partition pairs. |
| S18 | Wikipedia "Zarankiewicz problem" (raw wikitext), OEIS A001197, A001198, A006613 | — | read | KST/Znám/Füredi asymptotic forms; `k_2(n)`, `k_3(n)`, `k_{2,3}(n)` tables. |
| S19 | Sidorenko 1995 survey; Bollobás *Extremal Graph Theory* ch. VI; Irving 1978; Nikiforov 2010; Znám 1963/65; Hyltén-Cavallius 1958 | — | not accessed | Only as cited in S8, S9, S13, S14. |

Honesty note: everything attributed to Guy 1969 below is second-hand. Tan labels Guy's arguments **A, B, D, I** in the paper, but his code (`Kyoto/kyoto/utilities.py`, `get_partitions`) says the prefix-minor check implements "Guy's arguments A and E", and DHS cite the deletion argument as "Guy p138, point C". So the letter-to-argument map I can support is: A = binomial counting, B = balancing/convexity, C = deletion of a min-degree vertex, D = row-local counting, E (code) / I (paper) = inclusion of minors. Letters F, G, H (and whether E ≠ I) could not be verified.

---

## 2. The global theory in one page (what is *proved* in the sources)

Everything here is a consequence of one double count plus convexity; the value of the survey is knowing which consequences survive on a *fixed* profile.

**Master inequality (KST counting, "Argument A").** For a `K_{s,t}`-free `m×n` matrix,

```
(A_col)  Σ_{j=1}^{n} C(c_j, s)  ≤  (t−1)·C(m, s)        [each s-set of rows lies in ≤ t−1 columns]
(A_row)  Σ_{i=1}^{m} C(r_i, t)  ≤  (s−1)·C(n, t)        [each t-set of columns lies in ≤ s−1 rows]
```
(Tan Argument A, transposed form stated by Tan; DHS §3.1; Collins Lemma 2 "Star-Counting"; DGH constraint (3); README's "Kővári–Sós–Turán counting prune".)

**Convexity / balancing ("Argument B").** `C(x,k)` is convex on integers: for `m−n > 1`, `k ≥ 2`, `C(m−1,k)+C(n+1,k) < C(m,k)+C(n,k)` (Tan). Hence for fixed total `w`, `Σ C(c_j,s)` is minimised when all `c_j` are within 1 of each other (Collins Prop. 1). This turns (A) into closed-form bounds:

* **Kővári–Sós–Turán 1954** (form in Collins/Wikipedia): `z(m,n;s,t) < (s−1)^{1/t}(n−t+1)·m^{1−1/t} + (t−1)m`. Symmetric: `z(n;t) < (t−1)^{1/t} n^{2−1/t} + (t−1)n`; **Znám 1963** improved the second-order term to `½(t−1)n + 1`. DHS quote KST for `s=t=2` as `Z_{2,2}(n,n) < [n^{3/2}] + 2n` and `lim Z_{2,2}(n,n)/n^{3/2} = 1`.
* **Reiman 1958**: `z(n,n;2,2) ≤ (n/2)(1 + √(4n−3))`, equality iff `n = k²+k+1` and the matrix is the incidence matrix of a projective plane of order `k`. Unbalanced: `Z_{2,2}(m,n) ≤ ½(n + √(n² + 4nm(m−1)))` (DHS) — equivalently DDR's `z(m,n) ≤ m/2 + √(m²+4mn(n−1))/2` (other orientation). General `t`: `Z_{2,λ+1}(m,n) ≤ ½(n + √(n² + 4λ·n·m(m−1)))`, equality iff the graph is a `2-(m,k,λ)` design (also Hyltén-Cavallius 1958). Derivation: (A_col) with `s=2` and Jensen: `E(E−n)/(2n) ≤ λ·m(m−1)/2`.
* **Roman 1975** (Tan Thm 2.2; DHS Thm 3.5; DGH Thm 1.1): for every integer `p ≥ s−1`,
  ```
  z(m,n;s,t) ≤ ⌊ (t−1)·C(m,s)/C(p,s−1) + n·(p+1)(s−1)/s ⌋ .
  ```
  Equality iff every column has sum `p` or `p+1` and every `s`-set of rows has exactly `t−1` common columns (a `s-(m,{p,p+1},t−1)` design). DGH: the best-over-`p` Roman bound is the optimum of the LP "maximise `Σ i·n_i` s.t. `Σ n_i = n`, `Σ C(i,s) n_i ≤ (t−1)C(m,s)`" (`n_i` = number of columns of sum `i`); it is piecewise linear in `n` with breakpoints ("Roman points") at `n = (t−1)C(m,s)/C(ℓ,s)`, `ℓ ≥ s`; at integral Roman points the bound is attained iff an `s-(m,ℓ,t−1)` design exists. Always take `min` over both orientations (Tan's `romanbound`).
  *Proof skeleton (my reconstruction, verified numerically for `s ≤ 5, p ≤ 14, x ≤ 24`):* **Roman's integer inequality** `C(x,s) ≥ C(p,s) + (x−p)·C(p,s−1)` for all integers `x ≥ 0`, `p ≥ s−1` (tangent line at `p`; differences `C(j−1,s−1)` are monotone in `j`). Sum over columns: `n·C(p,s) + (w − np)·C(p,s−1) ≤ Σ C(c_j,s) ≤ (t−1)C(m,s)`, and `C(p,s)/C(p,s−1) = (p−s+1)/s` gives the formula. DHS Thm 3.4 states the same for any strictly increasing convex `f`.
* **Čulík 1956**: if `1 ≤ s ≤ m` and `n ≥ (t−1)·C(m,s)` then `z(m,n;s,t) = (s−1)n + (t−1)C(m,s)` (= Roman with `p = s−1`; DGH: Roman with `k=s−1` is attained for all `n ≥ (t−1)C(m,s)`).
* **Guy's theorem** (DHS Thm 3.3, attributed to Guy 1969): if `ℓ(n,s,t) ≤ n ≤ (t−1)C(m,s)+1` then `z(m,n;s,t) = ⌊((s²−1)n + (t−1)C(m,s))/s⌋` (= Roman with `p = s`), where `ℓ ≈ (t−1)C(m,s)/(s+1)`. Tan makes the threshold exact: equality holds with `p = s` or `p = s−1` whenever `(t−1)·C(m,s) − s·T_{s,t}(m) ≤ n`, where `T_{s,t}(m)` is the max size of a multiset of `(s+1)`-subsets of `[m]` covering every `s`-subset at most `t−1` times (and the threshold drops by `s−1` if the covering is not perfect). Tan Thm 2.3: `T_{s,t}(m) ≤ ⌊ (m/(s+1)) ⌊ ((t−1)/s)·C(m−1,s−1) ⌋ ⌋`. Tan Table 1 lists `T_{3,3}(m)` for `m = 5,6,7,9,11,12,13,15,17,18` (= 5, 9, 15, 40, 80, 108, 143, 225, 340, 405) and `T_{4,4}(6..9) = 7, 21, 36, 69`; Guy: `T_{2,2}(m) = ⌊(m/3)⌊(m−1)/2⌋⌋ − [m ≡ 5 mod 6]`; Bao–Ji: `T_{3,2}(m) = ⌊(m/4)⌊((m−1)/3)⌊(m−2)/2⌋⌋ − [m ≡ 0 mod 6]⌋`.
* **Füredi 1996** (Theorem 2, from the PDF; OCR-garbled, transcription is mine): `z(m,n;s,t) < (s−t+1)^{1/t}·n·m^{1−1/t} + t·n + t·m^{2−2/t}` for all `m ≥ s ≥ t ≥ 2`, `n ≥ t`; asymptotically optimal for `t=2` and `s=t=3` (`ex(n,K_{3,3}) = ½ n^{5/3} + O(n^{5/3−c})`, Brown's construction). (Collins et al. print a different-looking rendering with `(n−t+1)m n^{1−1/t} + (t−1)n^{2−2/t} + (t−2)m`; I could not reconcile the two from the garbled scan — treat the exact second-order terms as unverified.) *Method:* fix `t−2` rows `I`; every `t`-subset of their common neighbourhood `R_I` lies in ≤ `s−t+1` further rows, so `Σ_{x∉I} C(|R_x ∩ R_I|, t) ≤ (s−t+1)·C(|R_I|, t)`; sum over all `I`, use Lemma 1 (Jensen) and Lemma 2 (a convexity inequality on `Σ_{t-sets of columns} C(codeg, t−1)` in terms of column sums). **This uses codegrees, which a profile does not determine** — see P11.
* **Davies–Gill–Horsley 2024** (Lemma 3.2): for an `(s,t−1)`-linear hypergraph on `m` vertices with `n_i` edges of size `i` (`i ≥ s−1`; smaller edges may be padded to size `s−1` without affecting linearity), for all integers `1 ≤ v < s ≤ k ≤ m`, with `B := C(k−v, s−v)` and `α := ((t−1)·C(m−v, s−v)) mod B`:
  ```
  (4)   (1/(B−α)) · Σ_{i=s−1}^{k−1} (C(i−v, s−v) − α)·C(i,v)·n_i  +  Σ_{i=k}^{m} C(i,v)·n_i
          ≤  C(m,v) · ((t−1)·C(m−v, s−v) − α) / B .
  ```
  Proof: for each `v`-set `X` of rows, `Σ_{cols j ⊇ X} C(c_j − v, s−v) ≤ (t−1)·C(m−v, s−v)` (a `v`-set version of Argument D), then the "deficiency" trick turns the integrality gap `α` into a per-`X` degree bound `deg(X) ≤ c + τ(X)/(B−α)` with `c·B = (t−1)C(m−v,s−v) − α`, summed over all `X`. Adding (4) to Roman's LP improves the best known bound in many small cases, almost always via `v = s−1` (97.5% of cases in their Table 1); Theorem 1.3 is a closed form for `v=s−1`, single `k`. Their Table 2 (`s=t=3`) improvements: `z(10,22) ≤ 111, z(10,23) ≤ 115; z(11,19..21) ≤ 108,112,116; z(13,17..21) ≤ 116,121,125,130,135; z(14,17..22) ≤ 124,129,135,140,145,150; z(15,17..21,23) ≤ 132,138,143,149,154,165; z(16,17..23) ≤ 141,146,152,158,164,169,175` (each 1 or 2 below Roman/Tan's table).

**Structure of extremal matrices (what is known).**
* Roman equality ⇒ column sums in `{p, p+1}` and every `s`-set of rows has exactly `t−1` common columns (design). At Roman points, extremal ⇔ design (DGH §2). Near a Roman point the surviving profiles are nearly regular (this is exactly why the case count collapses there; §5 census).
* Deleting the min-degree vertex from an extremal graph is often extremal (DHS Thm 3.15 / Héger 6.2.9: `ex(m+c,n) ≤ e + c⌊e/m⌋`, with equality propagating downwards).
* GHO (via DDR): every extremal graph for `z(18;2)=81` has degree sequence `4^9 5^9` on both sides; DDR: this graph is unique. DHS Prop 3.29: any `(16,17)` C4-free graph with 71 edges would have degree sequences `{4^9,5^7}`, `{4^14,5^3}` — contradiction by neighbourhood packing. So "rows within 1 of each other" is *typical* near Roman points but **not** a theorem: Héger (thesis §6.3) gives an extremal `(8,8)` C4-free graph (24 edges) with degrees `{2,3,4}` on both sides, refuting the Illés–Krarup realisability conjecture; Héger's Questions 6.3.1/6.3.2 (does *some* extremal `K_{t,t}`-free graph have one/two nearly regular classes?) are open. DHS Cor. 3.27: extremal `(n²+n+2, n²+n+1)` C4-free graphs have a vertex of degree ≤ `n/2+1`, so both classes nearly regular is impossible there.
* Balbuena et al. Remark 3.1: `A` is `K_{s,t}`-free iff for every `s`-set `S` of rows, the union of their zero-columns has size ≥ `n−t+1`. In the dense regime (`max{m,n} ≤ s+t−1`, or Theorem 2.1's hypotheses) the extremal matrices are `J − (matching / disjoint stars / high-girth sparse graph)`, and `z = mn − (⌊(n−t)/s⌋(m−s) + m + n − s − t + 1)`.
* Empirical (Héger §6.3): known values satisfy `Z_{t,t}(m,n) ≤ Z_{t,t}(m+1,n−1)` whenever `m+2 ≤ n` (balanced shapes carry more edges); Question 6.3.3 asks whether this always holds.

---

## 3. What Tan's pipeline already does with these (baseline to beat)

From `Kyoto/kyoto/utilities.py` (read, not the paper's prose):

* Column partitions of `w` into `n` parts in `[0,m]`, non-increasing, generated with pruning by **A** (`Σ C(part, a) ≤ (b−1)C(m,a)`) and **E** (every prefix of the `n'` largest parts has sum ≤ `zub(a,b,m,n')`, where `zub` = min(Roman, Roman-transposed, and any better value in Tan's data file for `z(a,b,m,n')`)). Rows likewise with `(b,a,n,m)`.
* Pairs `(cpart, rpart)` are filtered by **D** in its pessimistic profile form: `Σ_{the r_max lightest columns} C(c_j − 1, a−1) > (b−1)·C(m−1, a−1)` ⇒ kill (and the transpose). `r_max = rpart[0]`, the `r_max` lightest columns = `cpart[-r_max:]`.
* Everything else (row/column mismatch, deficit, caps) is implicit in the partition generator.
* No Gale–Ryser check, no DGH (4), no mixed-minor Argument I (only prefixes of one side), no dense-row cap. Then lex-sorting within equal-sum groups + Sinz cardinality + one clause per `a×b` minor, Kissat.
* Tan's remark (§4.1): "Better bounds at the edges of the exact region can often be derived by applying the arguments of section 2 to eliminate all possible partitions and hence the need for any SAT solving; this gives for example `z_4(11,14) ≤ 106`." My census (§5) with Roman-only minor bounds leaves 4,442 pairs at `w=107`, so this kill depends on **exact** minor values from his tables feeding Argument E/I — the quality of the minor table is the single biggest lever.

---

## 4. THE CATALOGUE of profile-level pruning inequalities

Format: **name** — hypotheses — statement (matrix form, then the profile-checkable form used as `kill`) — what it kills / when it fires — source — Lean 4 (Mathlib-free, `ZarPrune`) provability.

Lean difficulty scale: *trivial* (< 50 lines, `omega`/`decide` over `sumFin`), *easy* (50–150, existing `Sum.lean` lemmas), *medium* (150–400, needs one new library piece: submatrix re-indexing with `Fin.succAbove`, or "sum over a subset ≥ sum of the `k` smallest"), *hard* (400–1000, needs the **subset-counting layer**: number of strictly increasing `k`-tuples of `Fin N` inside a predicate of cardinality `r` equals `C(r,k)`, plus "a set of ≥ k indices contains an increasing `k`-tuple"), *very hard* (> 1000 or genuinely new proof engineering).

### P0. Trivial sanity prunes (already proved in `Prunes.lean`)
`deficit` (`Σ r_i < w`), `mismatch` (`Σ r_i ≠ Σ c_j`), `rowCap`/`colCap` (`r_i > n`, `c_j > m`). Trivial. Also missing but trivial: `c_j ≥ 0`, and the *dual deficit* `Σ c_j < w` (currently only via mismatch).

### P1. Argument A — KST/star counting (both orientations)
* Hypotheses: none beyond `K_{s,t}`-free; `s ≤ m`, `t ≤ n` (else vacuous).
* Statement: `Σ_j C(c_j, s) ≤ (t−1)·C(m, s)` and `Σ_i C(r_i, t) ≤ (s−1)·C(n, t)`.
* Kill: `Σ_j C(c_j,s) > (t−1)C(m,s)` or the row analogue. Exactly computable from the profile; it is the workhorse — on a fixed profile it **dominates** every convexity-derived bound (KST, Reiman, Znám, Roman with any `p`): those are all `Σ C(c_j,s) ≥ (linear lower bound)`, so if the linear bound exceeds the budget so does the exact sum. Hence **Roman's bound is never a new profile prune**; it matters only (i) as a global bound that fixes the target `w` and (ii) as the default entry of the minor table used by P3.
* Fires: at `w` = Roman bound, kills all but a handful of partitions (census §5: at `z(13,18;3,3)`, `w=122`, only 1 column partition survives A). Two below the bound it kills a small fraction.
* Source: Tan Arg. A; Collins Lemma 2; DHS §3.1; DGH (3).
* Lean: **hard** (the subset-counting layer). Plan: define `choose` recursively; define `cnt k N P` := number of increasing `k`-tuples `Fin k → Fin N` all in `P` by recursion on `N` (element `0` in or out) and prove `cnt k N P = choose (card P) k`; define `f(C) := #{i : ∀ b, A i (C b)}` for each increasing `t`-tuple `C`; Fubini (`sumFin_swap`-style, but over tuples, so a new `sumTuples` with its own swap lemma) gives `Σ_i cnt t n (A i ·) = Σ_C f(C)`; `f(C) ≥ s` ⇒ extract an increasing `s`-tuple of rows (`cnt s m Q ≥ 1 ↔ card Q ≥ s`) ⇒ `HasKst`. One-time investment; unlocks P2, P5, P7, P9. The README already flags this as "the next target".

### P2. Argument D — row-local (and column-local) counting; generalisation D_v
* Hypotheses: none.
* Matrix statement (Tan): for any row `i` whose ones lie in columns `j ∈ N(i)`: `Σ_{j∈N(i)} C(c_j − 1, s−1) ≤ (t−1)·C(m−1, s−1)`. (Else the `(m−1)×n` matrix without row `i`, restricted to `N(i)`, has an `(s−1)×t` all-ones minor by Argument A, and row `i` extends it.) Transpose: for any column `j`, `Σ_{i∈N(j)} C(r_i − 1, t−1) ≤ (s−1)·C(n−1, t−1)`.
* General **D_v** (implicit in DGH's proof of (4), and the `pair_cuts` in `encodings_zar.py`): for every `v`-set `X` of rows, `0 ≤ v ≤ s−1`: `Σ_{j ⊇ X} C(c_j − v, s−v) ≤ (t−1)·C(m−v, s−v)`. `v=0` is A, `v=1` is D, `v = s−1` is the linear "pair cut" `Σ_{j ⊇ X} (c_j − s + 1) ≤ (t−1)(m−s+1)` (for `s=3`: any two rows, `Σ_{common j}(c_j−2) ≤ (t−1)(m−2)`, exactly the cut used in `encodings_zar.py`).
* Profile-checkable form (Tan): the row with the largest sum `r_max` meets *some* `r_max` columns; the pessimal choice is the `r_max` **lightest** columns: kill if `Σ_{r_max lightest j} C(c_j−1, s−1) > (t−1)C(m−1,s−1)`. For `v ≥ 2` the common neighbourhood of `X` is not determined by the profile (it can be empty), so D_v is a *SAT-side cutting plane*, not a profile prune, unless combined with a codegree lower bound (P7).
* Fires: kills 30–70% of the pairs that survive A/E on realistic cases (§5: `(16,16;3,3)`, `w=129`: 277,729 → 89,764; `(13,18;3,3)`, `w=117`: 107,502 → 67,795).
* Source: Tan Arg. D; DHS Prop 3.29 (`s=t=2` instance: neighbourhoods of the neighbours of a degree-5 vertex are pairwise disjoint).
* Lean: **hard** = P1's layer + (a) delete row `i` via `Fin.succAbove` re-indexing (increasing tuples compose with strictly monotone maps, so `HasKst` transfers), (b) the "sum over any `r` columns ≥ sum of the `r` smallest" lemma (**medium** on its own: an insertion-sort on `List Nat` with a sorted-prefix-sum lemma). A **sort-free weaker kill** avoids (b): for any threshold `θ`, with `N_θ := #{j : C(c_j−1,s−1) ≤ θ}`, kill if `(θ+1)·(r_max − N_θ) > (t−1)C(m−1,s−1)` (each of the ≥ `r_max − N_θ` columns of `N(i)` outside the light set contributes ≥ `θ+1`) — plain counting, no sorting.

### P3. Argument I / E — inclusion of minors (needs a *minor-bound table*)
* Hypotheses: a verified table `Z(m',n') ≥ z(m',n';s,t)` for the minors used (from Roman, Čulík, or earlier closures of the pipeline).
* Matrix statement (Tan Arg. I): every `m'×n'` submatrix is `K_{s,t}`-free, hence has ≤ `z(m',n';s,t)` ones.
* Profile forms:
  1. **Row-prefix (Tan's E on rows):** for every `k`, the `k` heaviest rows satisfy `Σ_{top k} r_i ≤ Z(k, n)`; columns likewise `Σ_{top k} c_j ≤ Z(m, k)`. (Tan's `Elims`.)
  2. **Mixed minor (new, easy extension):** for `R` ⊆ rows, `C` ⊆ cols, the number of ones in `R×C` is at least `max(Σ_{i∈R} r_i − Σ_{j∉C} c_j, Σ_{j∈C} c_j − Σ_{i∉R} r_i, 0)`, so kill if `Σ_{top k} r_i − Σ_{ℓ lightest} c_j > Z(k, n−ℓ)` for some `k, ℓ`.
  3. **Deletion / density corollary (Guy's point C; DHS 3.15; Collins Lemma 3):** `w − r_min ≤ Z(m−1, n)`; iterated: `w − (sum of the `c` lightest rows) ≤ Z(m−c, n)`; the global form `z(m+c,n) ≤ e + c⌊e/m⌋` is Collins's Lemma 4 / `z_bound`. On profiles this is form 1 with `k = m−c`.
  4. **Edge-local (DDR Lemma 6 for `z(17)=74`):** for every one at `(i,j)`, deleting row `i` and column `j` leaves `w − r_i − c_j + 1 ≤ Z(m−1,n−1)`, i.e. `c_j ≥ w − Z(m−1,n−1) + 1 − r_i` for every `j ∈ N(i)`. Profile form: for every row `i`, `#{j : c_j ≥ w − Z(m−1,n−1) + 1 − r_i} ≥ r_i` (pessimal: the lightest row). DDR use it with `Z(16,16)=67`, `w=75`: every edge has `r_i + c_j ≥ 9`, forcing `δ = 4` and a neighbourhood-packing contradiction.
  5. **Max-row split (DHS Prop 3.20 with α=1):** `w ≤ r_max + Z_{s−1,t}(m−1, r_max) + Z_{s,t}(m−1, n − r_max)` (the neighbourhood of the heaviest row is `K_{s−1,t}`-free relative to the other rows; the other columns are disjoint from that row). Needs a table for the `(s−1,t)` problem too (Roman is fine). DHS use it to get `Z_{3,3}(7,7) ≤ 33` (Roman gives 35) and `Z_{2,2}(16,16) ≤ 67`.
* Fires: form 1 with **exact** small values is Tan's strongest tool (kills everything for `z_4(11,14) ≤ 106`); with Roman-only values it is much weaker (§5).
* Source: Tan Arg. I / code "E"; DHS 3.15, 3.20, 3.29; Héger 6.2.9–6.2.18; Collins Lemmas 3–4; DDR Lemma 6.
* Lean: **medium** per form, *but* the structure must change: `Prune.sound` must be allowed to assume the table. Suggest `structure CondPrune (P : Params) (tbl : Nat → Nat → Nat) where kill …; sound : (∀ m' n' (B : Mat m' n'), ¬HasKst' B → weight B ≤ tbl m' n') → ∀ A, kill (profileOf A) → ¬ Valid P A`, discharged at closure time by previously verified `upper_bound_of_cover` instances (this is how results chain across `(m,n)`). Submatrix construction: `Mat m' n'` from `A` via strictly monotone embeddings `e : Fin m' → Fin m`, `g : Fin n' → Fin n`; `HasKst` of the minor ⇒ `HasKst` of `A` because `e ∘ R`, `g ∘ C` are increasing (one-line lemma). Form 5 additionally needs "inserting the deleted row into an increasing `(s−1)`-tuple gives an increasing `s`-tuple" (**medium-hard**).

### P4. Roman's bound and Roman's integer inequality
* Statement: see §2. Profile-level content = P1 (dominated). Keep as (i) the global target `w`, (ii) default minor table entries for P3, (iii) the **structure prune at Roman points**: if `w ≥ ⌊R(s,t,m,n,p)⌋` for the optimal `p` and the (real) bound is tight, every column sum must be `p` or `p+1` — again dominated by P1 on a fixed profile, but a useful *generator-side* filter and explains the near-regular survivors.
* Lean: Roman's inequality `C(x,s) + p·C(p,s−1) ≥ C(p,s) + x·C(p,s−1)` (subtraction-free form) is an **easy** induction on `x` once `choose` and Pascal's rule exist; the global bound is then P1 + this. Not needed for pruning.

### P5. Davies–Gill–Horsley constraint (4) — a genuinely new profile prune
* Hypotheses: none. Input: only the column-sum *distribution* `n_i` (and, transposed, the row-sum distribution with `(s,t,m,n) → (t,s,n,m)`).
* Statement: inequality (4) in §2, for each `1 ≤ v < s ≤ k ≤ m`. All quantities are integers/rationals computable from the profile; multiply through by `B−α` and `B` to get an integer test.
* Fires: **verified numerically** (§5): at `z(13,18;3,3)`, `w=122` the unique A-admissible column partition violates (4) ⇒ `z(13,18;3,3) ≤ 121` with zero SAT calls (DGH's Table 2 value); at `z(13,17;3,3)`, `w=117`, the 3 pairs surviving Tan's A/D/E are all killed ⇒ `≤ 116` (DGH's value). On frontier cases it kills 4–7% of the pairs Tan keeps (`(16,16;3,3)` `w=129`: 89,764 → 85,582; `(13,18;3,3)` `w=117`: 67,795 → 62,935; `(11,14;4,4)` `w=107`: 4,442 → 4,044). Only `v = s−1` seems to matter (DGH Table 1).
* Source: DGH Lemma 3.2, Theorem 1.2; special case `s=2, v=1` is Chen–Horsley–Mammoliti [3] in DGH.
* Lean: **very hard** as a general theorem (sum over all `v`-subsets, the deficiency bookkeeping, division by `B−α`). Two cheaper routes: (a) prove only `v = s−1` (then `C(i−v,s−v) = i−s+1`, `B = k−s+1`, everything linear — **hard** but no worse than P2 once P1's layer exists: for a fixed `(s−1)`-set `X`, `Σ_{j⊇X}(c_j−s+1) ≤ (t−1)(m−s+1)` is D_{s−1}; the aggregation is a double count over `(s−1)`-tuples, which the `cnt` layer provides); (b) for `s = 2` (`v=1`) it is a per-row statement and only needs P2's machinery.

### P6. Gale–Ryser necessity (realisability of the profile at all)
* Hypotheses: none (does not even use `K_{s,t}`-freeness).
* Statement: for every set `S` of columns, `Σ_{j∈S} c_j ≤ Σ_i min(r_i, |S|)` (ones in the columns of `S` counted by rows). With `c` sorted non-increasingly, checking prefixes `S = {k heaviest}` for all `k` is the full Gale–Ryser condition (necessary and sufficient for existence of *some* 0/1 matrix).
* Fires: **never on realistic cases** (§5: 0 extra kills on all 20 tested case sets) because A+E survivors are nearly regular. Cheap insurance against pathological pairs the generator could otherwise pair up (Tan's generator pairs row and column partitions independently); the SAT solver would refute such cases by unit propagation anyway.
* Source: standard (not in the Zarankiewicz literature reviewed).
* Lean: **easy** (prefix sums as `sumFin` with an indicator, `sumFin_swap`, `min` bounds; ~100 lines; the kill can test prefixes of the vector as given, which is still sound since the inequality holds for every `S`).

### P7. Dense-row cap (Bonferroni / Balbuena's complement view)
* Hypotheses: none.
* Statement: for any `s` rows `S`, `|∩_{i∈S} N(i)| ≥ Σ_{i∈S} r_i − (s−1)·n`; `K_{s,t}`-freeness forces this ≤ `t−1`. Profile form: `Σ_{s heaviest} r_i ≤ (s−1)n + t − 1` and `Σ_{t heaviest} c_j ≤ (t−1)m + s − 1`. Equivalent to Balbuena Remark 3.1 (union of zero-columns of any `s` rows has size ≥ `n−t+1`) and to Argument I with Čulík's `z(s,n;s,t) = (s−1)n + t − 1` (Balbuena Lemma 3.1).
* Fires: only in the dense regime (`w` near `mn`, e.g. `max{m,n} ≤ s+t−1`); **never** on the frontier cases tested (§5). Include for completeness/generality of the evolved library, not for kills.
* Lean: **medium** (a union bound over `Fin n` with indicators — easy; then "codegree ≥ t ⇒ increasing `t`-tuple of common columns" needs the extraction lemma from P1's layer, or a direct recursive construction of an increasing tuple from a set of size ≥ `t`).

### P8. Argument B (balancing) — *not a prune*
`Σ C(x_j,k)` over partitions of `w` is minimised by the most balanced partition. Use only to (i) derive global bounds, (ii) order the case list (most balanced first = closest to designs = usually hardest/last), (iii) reason about extremal structure. Any evolved candidate that "kills the unbalanced partitions because the balanced one is better" is unsound, like the sorting prune (`Demo.notDescending_unsound`). Lean: trivial to prove the convexity inequality, irrelevant to `Prune`.

### P9. DGH per-set degree bound (the pointwise lemma behind (4))
For each `v`-set `X`: `deg(X) ≤ c + τ(X)/(B−α)` where `τ(X) = Σ_{j⊇X} max(0, B − C(c_j−v, s−v))` is the total deficiency of the columns through `X`. For `v = 1` this is a strengthening of Argument D by rounding: with `θ_j := C(k−1,s−1) − C(c_j−1,s−1)` (0 if `c_j ≥ k`), `r_i ≤ c + (Σ_{j∈N(i)} θ_j)/(B−α)`. Profile-checkable pessimistically like D (heaviest row, lightest columns) and **strictly stronger than D whenever `α > 0`**. Lean: same as P2 plus a floor-division argument (**hard**).

### P10. Codegree identities (exact, from the profile) — enable D_v at the SAT level
`Σ_{pairs of rows} |N(i)∩N(i')| = Σ_j C(c_j, 2)`, and generally `Σ_{v-sets X} deg(X) = Σ_j C(c_j, v)`. These are equalities computable from the column profile and make the *aggregate* of the D_v cuts redundant with A; the value is in the *per-set* cuts (SAT clauses), not in pruning. Lean: easy given P1's layer (they are the Fubini step of P1).

### P11. What is *not* profile-checkable (needs richer case variables)
* Füredi's codegree counting (S8) and all `v ≥ 2` per-set cuts: need `deg(X)` for specific `X`.
* DHS Prop 3.20 with `α ≥ 2` (max size `β` of a `K_{α,β}`): needs codegrees.
* DDR/DHS neighbourhood-packing endgames (Prop 3.29, Lemma 6): need adjacency.
Design consequence: if the case index is enriched from (row sums, column sums) to (row sums, column sums, max pair-codegree `β`) — a cube-and-conquer split on `β` — then P11-type arguments become profile prunes (e.g. `β ≤ t−1` for `s=2`; for `s=3`, `β` bounds `Σ_{j}C(c_j,2)` via `C(m,2)·β ≥ Σ_j C(c_j,2)`). This is the natural bridge to the AlphaMapleSAT/cube-and-conquer literature cited in the proposal.

---

## 5. Numbers and tables

### 5.1 Exact diagonal values
`z(n;2)` (Afzaly–McKay via Collins Table 3; DDR Table 1 for `n ≤ 21`; consistent with OEIS A001197 = `z+1`):
```
n:    1  2  3  4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20   21   22   23   24   25   26   27   28   29   30   31   32
z:    1  3  6  9  12  16  21  24  29  34  39  45  52  56  61  67  74  81  88  96  105  108  115  122  130  138  147  156  165  175  186  189/190
```
`n = k²+k+1−h` pattern (DDR Thm 4, prime power `k`): `h=0: k³+2k²+2k+1`; `h=1: k³+2k²`; `h=2: k³+2k²−2k`; `h=3: k³+2k²−4k+1`; conjectured `h=4: k³+2k²−6k+2` (true for `k=2,3,4`).

`z(n;3)` (OEIS A001198 = `z+1`; Tan Table 3 diagonal): `n=3..16: 8, 13, 20, 26, 33, 42, 49, 60, 69, 80, 92, 105, 120, 128`; `z(17;3)` open: `≤ 141` (Collins Table 4). `z(n,n;2,3)` (A006613 = `z+1`): `n=3..11: 7,12,16,21,28,33,39,46,55`. `z(n;4)` (Tan Table 4 diagonal, exact for `n ≤ 13`): `15, 22, 31, 42, 51, 61, 74, 86, 100, 117`.

Tan Table 3 row `m=13`, `s=t=3`, `n=13..23`: `92 98 104 107 117 122 126 131 136 140 145` (bold/exact in Tan only up to the "solid line"; the entries from `n=17` on are Roman bounds, which DGH improve to `116 121 125 130 135` for `n=17..21`). Recent frontier claims cited in the proposal (Afrasyab arXiv:2608.08154): `Z(13,18;3,3)=116`, `Z(13,22;3,3)=137`, `Z(12,n;3,3)=6n` for `18 ≤ n ≤ 22`, `Z(14,17)=118`, `Z(14,18)=124`, `Z(15,17)=126`, `Z(15,18)=132`, `132 ≤ Z(16,17) ≤ 133` — not independently verified here.

Héger's table of best upper bounds for `Z_{2,2}(m,n)`, `7 ≤ m ≤ 31`, `7 ≤ n ≤ 23` is in `heger_thesis.txt` lines ~4407–4440 (scratchpad); e.g. `Z_{2,2}(16,17) = 70`, `Z_{2,2}(14,24) = 78`, `Z_{2,2}(14,25) = 80`.

### 5.2 Census: which inequality kills what (my computation, Roman-only minor table)
Column/row partitions generated with A + prefix-E (Roman bounds only), pairs filtered by D (Tan), then the extra prunes. "+X" = survivors after Tan's filter **and** X.

| case, `w` | cols / rows after A+E | pairs | after D (Tan) | +Gale–Ryser | +dense cap | +DGH(4) | all |
|---|---|---|---|---|---|---|---|
| `z(16,16;2,2)`, 68 | 25 / 25 | 625 | 16 | 16 | 16 | 16 | 16 |
| `z(17,17;2,2)`, 75 | 9 / 9 | 81 | 1 | 1 | 1 | 1 | 1 |
| `z(16,17;2,2)`, 71 | 25 / 13 | 325 | 16 | 16 | 16 | 16 | 16 |
| `z(10,10;3,3)`, 61 | 5 / 5 | 25 | 10 | 10 | 10 | 10 | 10 |
| `z(11,11;3,3)`, 70 | 25 / 25 | 625 | 237 | 237 | 237 | 237 | 237 |
| `z(12,12;3,3)`, 81 | 15 / 15 | 225 | 137 | 137 | 137 | 137 | 137 |
| `z(13,13;3,3)`, 93 | 32 / 32 | 1024 | 314 | 314 | 314 | 314 | 314 |
| `z(14,14;3,3)`, 106 | 20 / 20 | 400 | 140 | 140 | 140 | 140 | 140 |
| `z(16,16;3,3)`, 129 | 527 / 527 | 277,729 | 89,764 | 89,764 | 89,764 | 85,582 | 85,582 |
| `z(17,17;3,3)`, 141 | 757 / 757 | 573,049 | (not run) | | | 757/757 per side survive | |
| `z(13,17;3,3)`, 117 | 3 / 1 | 3 | 3 | 3 | 3 | **0** | **0** |
| `z(13,18;3,3)`, 122 | 1 / 7 | 7 | 1 | 1 | 1 | **0** | **0** |
| `z(13,18;3,3)`, 117 | 437 / 246 | 107,502 | 67,795 | 67,795 | 67,795 | 62,935 | 62,935 |
| `z(11,14;4,4)`, 107 | 104 / 70 | 7,280 | 4,442 | 4,442 | 4,442 | 4,044 | 4,044 |
| `z(10,10;4,4)`, 75 | 7 / 7 | 49 | 25 | 25 | 25 | 25 | 25 |

Reading: (i) D is the big filter; (ii) Gale–Ryser and the dense cap never fire; (iii) DGH (4) is the only *new* profile prune with teeth, and it fully closes two DGH-improved bounds without SAT; (iv) `z(17;3)` at `w=141` (the first open square case) has half a million pairs before D — the real frontier needs either ≫ 90% pruning or a much better minor table (Tan's `z_4(11,14) ≤ 106` shows what exact minors buy: with Roman-only minors 4,442 pairs remain).

---

## 6. Difficulty signals for a case (row partition, column partition)

None of the sources defines "difficulty" of a partition pair; the following are the quantities the sources implicitly use, ordered by how much evidence supports them.
1. **Slack in Argument A**: `σ_col = (t−1)C(m,s) − Σ_j C(c_j,s)`, `σ_row = (s−1)C(n,t) − Σ_i C(r_i,t)`. Zero slack = design-like; Roman equality ⇒ `p/p+1`-regular columns. Tan: values at the edge of the exact region are easy (arguments alone), "deeper in the interior" hard. DGH/DHS: near Roman points profiles are forced.
2. **Distance of `w` below the best global bound** (Roman / DGH LP optimum): Collins's backwards-path counts show the number of `(m,n,e^+)`-graphs explodes as `e` drops (`(7,7)_{5,2}`: 33 graphs at `e ≥ 42`, 1,619 at `≥ 39`, 7,500 at `≥ 37`): a case at `w = z+1` sits on top of many near-extremal substructures ⇒ long UNSAT proofs.
3. **Residual symmetry** after fixing sums: `Π_i (mult of row-sum value)! × Π_j (mult of col-sum value)!`. Regular profiles maximise it; Tan breaks it only by lex order within equal-sum groups (Fig. 2 shows non-identical isomorphic survivors). Larger group ⇒ harder for CDCL without stronger breaking, but also more likely to be a design (then propagation is strong). Ambiguous; measure.
4. **Argument-D slack of the heaviest row** and **P9 slack** (how much the pessimal column choice is below budget): small slack means the heaviest row's neighbourhood is nearly forced to the lightest columns — good for propagation.
5. **Minor-table gap**: `min_k [Z(k,n) − Σ_{top k} r_i]` (and columns): the smallest gap identifies the sub-minor that is nearly extremal; solving that minor's cases first (Collins's backwards path) or using its exact value as a lemma is the cheapest attack.
6. **Case mass**: number of surviving pairs sharing the same column partition (or the same row partition) — the SAT instances differ only in the row cardinality constraints, so a shared UNSAT core / proof reuse is plausible; also the natural unit for "difficulty of the section eliminated" in the reward.
7. **Empirical proxies** (recommended, no theory available): Kissat conflicts/time on the *same* profile at smaller `(m,n)` (Tan's tables were solved on one laptop up to `z_3(16,16)`); the number of lex-equal groups; and the fraction of the 90-second budget consumed in the existing `profiles.py` runs.

---

## 7. Design implications for the OpenEvolve + Lean pipeline

1. **Build the subset-counting layer once, by hand, as library** (`cnt`, `choose`, Fubini over increasing tuples, extraction of an increasing tuple from a large set). It is the bottleneck for P1, P2, P5, P7, P9. Do not ask the evolutionary search to rediscover it; expose it as lemmas so evolved prunes are short compositions (like `Prune.or`). Expected size 500–800 lines, Mathlib-free.
2. **Add conditional prunes**: a `CondPrune` whose soundness may assume a verified minor-bound table; discharge the table from earlier `upper_bound_of_cover` closures (Roman/Čulík entries proved once). Argument I with exact minors is the strongest practical pruner in Tan's work. The evolved `kill` can then reference `tbl m' n'`.
3. **The baseline to beat is A + D + E(with the best minor table) + DGH(4)**, not the four trivial prunes. Measured against that baseline, Gale–Ryser and the dense cap contribute nothing on frontier cases, so the reward must count *marginal* kills over the strong baseline and weight them by difficulty signals (§6), else the search will be rewarded for rediscovering dominated inequalities.
4. **Do not sort inside `kill` unless the sorted-prefix lemma exists**; use the sort-free threshold forms (P2, P3.4) as the first evolvable targets — they are what an LLM can plausibly prove with the existing `Sum.lean` toolkit.
5. **Enrich the case index** (row sums, column sums, max codegree `β`, or the number of columns of each size) to unlock P11-type arguments; this converts SAT-side cutting planes (`pair_cuts`) into provable prunes and is where genuinely new arguments could appear.
6. **Order cases by A-slack / balance** (most balanced last) and solve minors first (Collins's backwards path): both the literature and the census say the design-like cases are few but are where the SAT time goes.
7. **Realistic targets**: `z(13,18;3,3) ≤ 116`, `z(16,17;3,3)`, `z(17;3) ≤ 140` are the smallest open/frontier cases; at `w=117` on `(13,18)` about 63k pairs survive everything known — a per-case budget of seconds, or a prune killing > 90%, is required.
8. **Keep P8 (balancing) and sorting out of `Prune`** — they are symmetry moves; the README's negative test already enforces this.

---

## 8. Open questions (for the thesis)

* Guy's original letters E–H and whether Tan's "E" (code) and "I" (paper) are the same argument; nobody online reproduces Guy's list.
* Is DGH's (4) *jointly* over rows and columns (a bi-profile LP with both `n_i` and the row distribution, plus Gale–Ryser-type coupling) strictly stronger on fixed profiles? The census suggests coupling constraints (GR) are slack, but P9-type rounding across both sides is untested.
* Can the `v = s−1` special case of (4) be proved in Mathlib-free Lean in under ~600 lines on top of P1's layer? (My estimate says yes; untested.)
* Exact second-order terms of Füredi's Theorem 2 (OCR-garbled; two secondary renderings disagree).
* Héger's Questions 6.3.1–6.3.3 (nearly regular extremal graphs; balancedness monotonicity `Z_{t,t}(m,n) ≤ Z_{t,t}(m+1,n−1)`); DHS Question 3.13 (`Z_{2,2}(n²+1, n²+n+1) ≤ (n²+1)(n+1)`?). A proved monotonicity would itself be a minor-type prune.
* Whether "case mass" (§6.6) correlates with solver time — needs the experiment log.

---

## 9. Citations

* R. K. Guy, "A many-facetted problem of Zarankiewicz", *The Many Facets of Graph Theory*, LNM 110, Springer 1969, pp. 129–148. doi:10.1007/BFb0060112 (not accessed).
* R. K. Guy, "A problem of Zarankiewicz", *Theory of Graphs* (Tihany 1966), Academic Press 1968, pp. 119–150 (not accessed).
* R. K. Guy, S. Znám, "A problem of Zarankiewicz", *Recent Progress in Combinatorics*, Academic Press 1969, pp. 237–243 (not accessed).
* S. Roman, "A problem of Zarankiewicz", J. Combin. Theory Ser. A 18 (1975) 187–198 (not accessed; theorem via Tan/DHS/DGH).
* K. Čulík, "Teilweise Lösung eines verallgemeinerten Problems von K. Zarankiewicz", Ann. Polon. Math. 3 (1956) 165–168 (not accessed).
* I. Reiman, "Über ein Problem von K. Zarankiewicz", Acta Math. Acad. Sci. Hungar. 9 (1958) 269–273 (not accessed).
* T. Kővári, V. T. Sós, P. Turán, "On a problem of K. Zarankiewicz", Colloq. Math. 3 (1954) 50–57.
* Š. Znám, "On a combinatorial problem of K. Zarankiewicz", Colloq. Math. 11 (1963) 81–84; "Two improvements…", Colloq. Math. 13 (1965) 255–258.
* C. Hyltén-Cavallius, "On a combinatorial problem", Colloq. Math. 6 (1958) 59–65.
* Z. Füredi, "An upper bound on Zarankiewicz' problem", Combin. Probab. Comput. 5 (1996) 29–33. https://www.cambridge.org/core/services/aop-cambridge-core/content/view/A75E37B577796A8E16D7727A637D5308/S0963548300001814a.pdf
* G. Damásdi, T. Héger, T. Szőnyi, "The Zarankiewicz problem, cages, and geometries", Ann. Univ. Sci. Budapest. Sect. Math. 56 (2013) 3–37. http://heger.web.elte.hu/publ/Damasdi-Heger-Szonyi-Zarankiewicz-cages-geometries.pdf
* T. Héger, *Some graph theoretic aspects of finite geometries*, PhD thesis, ELTE 2013, ch. 6. https://heger.web.elte.hu/publ/HTdiss-e.pdf
* W. Goddard, M. A. Henning, O. R. Oellermann, "Bipartite Ramsey numbers and Zarankiewicz numbers", Discrete Math. 219 (2000) 85–95 (not accessed).
* A. F. Collins, A. W. N. Riasanovsky, J. C. Wallace, S. P. Radziszowski, "Zarankiewicz numbers and bipartite Ramsey numbers", arXiv:1604.01257 (2016). (Afzaly–McKay `z(n;2)`, `n ≤ 31`, Table 3.)
* J. Dybizbański, T. Dzido, S. Radziszowski, "On some Zarankiewicz numbers and bipartite Ramsey numbers for quadrilateral", arXiv:1303.5475; Ars Combin. 119 (2015) 275–287.
* S. Davies, P. Gill, D. Horsley, "Improved upper bounds on Zarankiewicz numbers", arXiv:2411.18842v2; Discrete Math. 349 (2026).
* C. Balbuena, P. García-Vázquez, X. Marcote, J. C. Valenzuela, "Extremal K(s,t)-free bipartite graphs", DMTCS 10:3 (2008) 35–48. https://dmtcs.episciences.org/435/pdf
* J. Tan, "An attack on Zarankiewicz's problem through SAT solving", arXiv:2203.02283v2 (2022); code repository "Kyoto".
* Wikipedia, "Zarankiewicz problem" (raw wikitext, accessed 2026-09-21); OEIS A001197, A001198, A006613.
* B. Bollobás, *Extremal Graph Theory*, Academic Press 1978 (ch. VI) / Handbook of Combinatorics II (1995) §1.3.3 (typo noted by DDR); A. Sidorenko, "What we know and what we do not know about Turán numbers", Graphs Combin. 11 (1995) 179–199 (neither accessed).

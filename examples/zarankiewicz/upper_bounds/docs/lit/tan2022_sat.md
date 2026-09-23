# Tan (2022) — *An attack on Zarankiewicz's problem through SAT solving*

Literature note for the MEng thesis "Discovering upper bounds for the Zarankiewicz numbers". Written 2026-09-21 from the full text (PDF v2 in `docs/papers/tan2022_sat_zarankiewicz.pdf`, 24 pp.) and from the author's Kyoto repository (cloned; HEAD `3acbc61`, 2022-04-20). Everything marked **[quote]** is transcribed from the paper; everything marked **[inferred]** is my own reading and should not be attributed to Tan.

## 0. Bibliographic record

| field | value |
|---|---|
| Author | Jeremy Tan (Jeremy Tan Jie Rui), National University of Singapore |
| Title | An attack on Zarankiewicz's problem through SAT solving |
| Venue | arXiv:2203.02283 [math.CO], cross-listed cs.DM, cs.LO; MSC 05C35. v1 3 Mar 2022, v2 19 Apr 2022 (PDF dated April 21, 2022). No journal version found. |
| v2 change note | "Updates include a correctness proof for a partition-generating algorithm and a new subsection on coverings." (i.e. Theorem 3.1 and Section 2.1 are v2 additions) |
| Code + data | `https://github.com/Parcly-Taxel/Kyoto` — "Exact values and maximal graphs for Zarankiewicz's problem". Files: `kyoto/{zcnf.py, utilities.py, prove.py, graphs.py, driver22.py, driver33.py, driver44.py, verifygraphs23.py}`, `kyoto/data/{2x2,3x3,4x4}` (proof ledgers), `paper/zarankiewicz.tex`. |
| Solver | Kissat (Biere et al., SAT Competition 2020), flags `--sat` / `--unsat` only, one laptop |
| Accessible | Yes — full primary text read. |
| Cites we care about | Guy 1969 [7] (arguments A–I), Guy 1967 [8] (argument E, OEIS A001197 scan), Čulik 1956 [5], Roman 1975 [15], Sinz 2005 [17], Wynn 2018 [21] (arXiv:1810.12975), Heule–Kullmann–Biere cube-and-conquer [12], DRAT-trim [20], Héger's thesis [10], Reiman 1958 [14], Bao–Ji 2014 [1], Hanani 1960 [9]. |

## 1. One-paragraph summary

Tan computes exact values of the Zarankiewicz function `z_{a,b}(m,n)` (max number of ones in an `m x n` 0/1 matrix with no all-one `a x b` minor) for `a = b = 2, 3, 4` by SAT solving. The instance for a guess `w` is the naive one-variable-per-cell CNF (one clause per `a x b` minor) plus cardinality constraints, but he never solves it whole: he *fixes the row-sum and column-sum partitions*, enumerates all (unordered) partition pairs not excluded by Guy's counting arguments A (pigeonhole on `a`-subsets of ones), D (a local version of A around one row), and I (inclusion / monotonicity in `m`, `n`), and solves one CNF per surviving pair, with rows/columns of equal sum forced into lexicographic order (a cube-and-conquer decomposition; cubes = partition pairs). He corrects 8 errors in Guy's tables (all in `z_2`, all one too low), extends the exact region of `z_2`, `z_3`, `z_4`, and lists all non-isomorphic maximal matrices for square cases. Crucially for us: (i) the pruning arguments are elementary double-counting statements on the *profile* (row/column sum vectors), (ii) on many cells the arguments alone eliminate every partition pair, so no SAT solving is needed at all (e.g. `z_4(11,14) <= 106`; in the Kyoto ledger `z_3(14,14) <= 105`, `z_3(15,15) <= 120`, `z_3(13,16) <= 107` are all "0 pairs"), and (iii) the number of surviving pairs (0 to 290 for `z_3`) is a concrete, reproducible difficulty measure.

## 2. Notation map (Tan -> ZarPrune)

| Tan | ZarPrune (`lean/ZarPrune`) | note |
|---|---|---|
| `a x b` minor, `z_{a,b}(m,n)` | `P.s x P.t`, `Params (m n s t w)` | Tan writes `z_a(m,n)` when `a=b`, `z_a(m)` when also `m=n`. Guy's `k_a(n) = z_a(n)+1`. |
| "admissible" matrix | `¬ HasKst P A` | Tan's minors are arbitrary row/column subsets; ours are increasing index tuples — same thing. |
| exactly `w` ones (CNF) | `Valid P A := ¬HasKst ∧ P.w ≤ weight A` | Tan's decision instance is "exactly w"; ours is "at least w". Equivalent for existence because deleting ones preserves admissibility — but the **cover** step must say so (see §8.3). |
| column partition `(c_1 ≥ … ≥ c_n)`, row partition `(r_1 ≥ … ≥ r_m)` | `Profile m n` = `(row : Fin m → Nat, col : Fin n → Nat)` | Tan's cases are *unordered* (sorted multisets); a `Profile` is the ordered refinement. |
| "argument fails for this most pessimal choice" | `kill pf = true` | a prune must be monotone in the choice it is pessimal about (§8.2). |

## 3. Section 2 of the paper — arguments, verbatim

**[quote] Argument A.** *Any admissible matrix with column sums `c_i`, `1 ≤ i ≤ n`, must satisfy `Σ_i C(c_i, a) ≤ (b−1)·C(m, a)`. Otherwise, by the pigeonhole principle – where pigeons are `a`-subsets of ones in each column, each such subset potentially part of an `a × b` minor, and holes are all `a`-subsets of the matrix's rows – there is a hole with at least `b` pigeons, forming an all-one `a × b` minor.*

**[quote] Argument B.** *For non-negative integers `m, n, k` with `m − n > 1` and `k ≥ 2`, `C(m−1,k) + C(n+1,k) < C(m,k) + C(n,k)`. Hence the binomial sum over columns `Σ_i C(c_i,a)` in argument A is minimised by distributing ones so that no two column sums differ by more than 1; if the binomial sum is then equal to `(b−1)C(m,a)` and the matrix can still be made to have no all-one `a × b` minor, that matrix must be maximal.*

**[quote] Corollary 2.1 (Čulik [5]).** *If `1 ≤ a ≤ m` and `n ≥ (b−1)·C(m,a)`, `z_{a,b}(m,n) = (a−1)n + (b−1)·C(m,a)`.*

**[quote] Argument D.** *Take any row of any admissible matrix. If this row's ones lie in columns with sums `c_1, …, c_r`, the inequality `Σ_{i=1}^{r} C(c_i − 1, a−1) ≤ (b−1)·C(m−1, a−1)` must hold, for otherwise (by argument A) there is an all-one `(a−1) × b` minor extendable to an all-one `a × b` minor through the ones in the chosen row.*

**[quote]** *The above arguments all have a transposed form obtained by replacing "columns" with "rows" and vice versa. The following inclusion argument is also clear.*

**[quote] Argument I.** *For `m' ≤ m` and `n' ≤ n`, every `m' × n'` minor of every witness to `z_{a,b}(m,n)` is admissible and thus has at most `z_{a,b}(m',n')` ones.*

(Guy 1967 calls the inclusion argument "argument E"; the Kyoto code uses that name: `Elims`. Argument B is *not* a prune — it identifies the minimiser of the A-sum and is used for maximality/lower bounds. Arguments C, F, G, H of Guy are not used by Tan.)

**[quote] Theorem 2.2 (Roman [15]).** *For all integers `p ≥ a − 1*

```
z_{a,b}(m,n) ≤ floor( (b−1)/C(p, a−1) · C(m, a) + (p+1)(a−1)/a · n )
```

*and equality holds with `p = a` or `p = a − 1` when `(b−1)·C(m,a) − a·T_{a,b}(m) ≤ n`, where `T_{a,b}(m)` is the largest size of a collection `C` of not necessarily distinct `a+1`-subsets of a set `S` with `m` elements such that every `a`-subset of `S` is covered by at most `b−1` sets of `C`. The lower bound, which is approximately `(b−1)/(a+1) · C(m,a)`, may be reduced by `a−1` if the covering is not perfect, i.e. `T_{a,b}(m) < (b−1)/(a+1) · C(m,a)`.*

**[quote]** *This bound appears tighter or at least as tight as other general bounds in the literature, such as the one developed by Collins [4], so it is the bound given in the tables in section 4.*

Kyoto's implementation (`utilities.romanbound1`): evaluate `f(p) = floor((b−1)/C(p,a−1)·C(m,a) + (p+1)(a−1)/a·n)` for `p = a−1, a, …` and stop at the first `p` where `f` stops decreasing; `romanbound = min(romanbound1(a,b,m,n), romanbound1(b,a,n,m))`. **[inferred]** the "lower bound … reduced by `a−1`" sentence refers to the threshold on `n`; Kyoto codes it as `packlimit = (b−1)C(m,a) − a·T − (a−1)·[covering not perfect]`, and the "solid lines" in the tables are exactly `n ≥ packlimit(m)`. I recomputed: for `a=b=3`, `packlimit(m)` for `m = 3..16` is `0, 2, 5, 11, 23, 28, 46, 60, 88, 114, 143, 182, 233, 280`, which reproduces the solid staircase in Table 3 (rows 3–5 entirely; row 6 for `n ≥ 11`; row 7 only at `n = 23`; row 8 never within `n ≤ 23`). 74 of the 203 cells of Table 3 are exact by Roman alone.

**Section 2.1, `T_{a,b}(m)`.** [quote] Guy [8]: `T_{2,2}(m) = floor( m/3 · floor((m−1)/2) ) − [m ≡ 5 mod 6]`; `T_{2,b}(m) = floor((b−2)/3 · C(m,2)) + T_{2,2}(m)` if `2 | m ∧ 2 | b`, else `floor((b−1)/3 · C(m,2)) − [m ≡ 5 mod 6 ∨ b ≡ 5 mod 6]` (Iverson brackets; the second case is my reading of the typeset formula — verify against Guy 1967 before relying on it). Bao–Ji [1]: `T_{3,2}(m) = floor( m/4 · floor( (m−1)/3 · floor((m−2)/2) ) ) − [m ≡ 0 mod 6]`. `T_{3,3}(m)` and `T_{4,4}(m)` had no literature values; Tan computed them with Gurobi (Table 1): `T_{3,3}: 5→5, 6→9, 7→15, 9→40, 11→80, 12→108, 13→143, 15→225, 17→340, 18→405`; `T_{4,4}: 6→7, 7→21, 8→36, 9→69`. For `m ≡ 2,4 (mod 6)` a perfect `T_{3,2}` covering exists (Hanani), so `T_{3,3}(m) = C(m,3)/2` by duplication. Kyoto's `packable_simplices` falls back to `T(m−1)` for other `m` (a valid lower bound on `T`, hence a valid — possibly loose — threshold).

**[quote] Theorem 2.3.** `T_{a,b}(m) ≤ floor( m/(a+1) · floor( (b−1)/a · C(m−1, a−1) ) )`. *Proof. For every `a+1`-subset `E` in `C` and any element `v ∈ E`, exactly `a` of the `(b−1)C(m−1,a−1)` available `a`-subsets containing `v` are covered by `E`, so at most `floor((b−1)/a · C(m−1,a−1))` `a+1`-subsets of `C` can contain `v`. Since `E` is arbitrary and the number of `E`-`v` incidences is always a multiple of `a+1`, the claimed upper bound follows.* It proves optimality of all listed `T_{3,3}` coverings except `T_{3,3}(7)` and `T_{3,3}(11)`, and of `T_{4,4}(7)`.

## 4. Section 3 of the paper — method

### 4.1 Base encoding [quote, lightly condensed]
One variable per entry of the `m × n` matrix `A`; one clause per `a × b` minor in rows `r_1..r_a`, columns `c_1..c_b`: `⋁_{i=1}^{a} ⋁_{j=1}^{b} ¬a_{ij}`; and a cardinality constraint requiring `A` to have **exactly `w`** ones. Any solution proves `z ≥ w`; UNSAT proves `z < w`. "Most SAT solvers have an option to output a concrete, machine-verifiable UNSAT proof [DRAT-trim]." (Kyoto's `solve_cnf` passes a `.drat` path; `prove_solutions` always supplies one, so DRAT files were produced; the repository contains no checking step and the paper does not say the proofs were checked.)

### 4.2 Generating partitions (Section 3.1)
**[quote]** *An admissible or maximal matrix clearly remains as such under all row and column permutations. It is therefore enough for a given `w` to solve instances where the row and column sums are fixed, over all possible combinations of unordered row and column partitions not forbidden by the arguments of section 2 – an approach very much like Heule's cube-and-conquer paradigm [12]. To generate all such partitions efficiently we use algorithm 1.*

**Algorithm 1 — Admissible (by arguments A and I) column partition generator** [quote, line by line; `▷` = comment in the paper]

```
 1: p ← empty stack                                   ▷ workspace for building up partitions
 2: procedure P(a, b, m, n, w)
 3:   L_A ← (b−1)·C(m, a)                             ▷ only set at procedure start, immutable afterwards
 4:   if Σ_i C(p_i, a) > L_A  or  Σ p > z_{a,b}(m, |p|) then   ▷ |p| is the current length of p
 5:     return
 6:   else if w = 0 then
 7:     output the contents of p
 8:   else if k > (m−1)·n then                        ▷ by the pigeonhole principle, some further columns must sum to m
 9:     d ← k − (m−1)·n
10:     push m d times onto p
11:     P(a, b, m, n−d, w−dm)
12:     pop d times from p
13:   else
14:     for t ∈ [⌈w/n⌉, min(w, m)] do                 ▷ all possible values for the next part
15:       push t onto p
16:       P(a, b, t, n−1, w−t)
17:       pop from p
18:     end for
19:   end if
20: end procedure
```

Reading notes **[inferred, confirmed against `utilities.get_partitions`]**: lines 8–9 write `k` where `w` (the remaining number of ones) is meant — the code uses `k`. In line 16 the recursive call passes the last part `t` as the new *upper limit* `m`, which is how non-increasing order is enforced; therefore the `m` in line 3 (`L_A`) and in the `z_{a,b}(m,|p|)` of line 4 must be the *original* `m` (the code keeps `Alim` and `Elims` in the enclosing scope: `Elims = {j: zbounds(a,b,m0,j)[1] for j in range(b, n0)}`, i.e. the check `Σ_{i≤j} p_i ≤ ub(z_{a,b}(m0, j))` is applied for every prefix length `j` from `b` to `n0−1`). Line 4's `z_{a,b}(m,|p|)` is the best **proved upper bound** available (Roman's bound or an earlier `< k` entry in the data ledger), never the conjectured value. Partitions that end early (`k = 0` with `|p| < n`) are emitted truncated; the omitted trailing columns are implicitly zero. Tan's procedure never uses a lower cutoff on parts.

**[quote] Theorem 3.1.** *Algorithm 1 generates all admissible partitions for an `m × n` matrix, `a × b` minor and `w` ones in lexicographic order – partitions of `w` into `n` parts in `[0, m]` – with the parts in each partition listed in non-increasing order.*

**[quote] Proof.** *Ignoring lines 4 and 8–12 for now, the recursive call to P in line 16 specifies an upper part limit of the last (topmost) element `t` of the stack `p`, so part sizes do not increase from left to right. By the pigeonhole principle the largest part of a partition of `w` into `n` parts is at least `⌈w/n⌉`, so this is the lower bound for `t`; the upper bound of `min(w, m)` is trivial. Because `t` is varied through all its possible values in increasing order at every point in the recursion tree, the partitions are output in lexicographic order. Lines 8–12 avoid unnecessary recursive calls to P when there is only one possible value for `t`. The admissibility checks in line 4 depend on the non-increasing partition ordering, which in turn ensures that the first `n'` column sums in `p` for any `n' < n` are the most pessimal partition choice for the column sums of an `m × n'` minor of the `m × n` matrix; if this minor partition is admissible then all other `m × n'` minor partitions in `p` are admissible because `Σ_i C(p_i, a)` for argument A and `Σ p > z_{a,b}(m,|p|)` for argument I cannot be higher for other partitions. Because line 4 is executed in every call to P, branches of the recursion tree leading to only inadmissible partitions are pruned as soon as possible.*

**How partition PAIRS are removed by Argument D [quote, the exact sentence]:** *The algorithm to generate row partitions is similar. Once all possible row and column partitions have been obtained argument D can then be used to remove partition pairs (considering the row with the most, `r`, ones and the `r` columns with the least ones – if argument D fails for this most pessimal column choice it must also fail for all other column choices – and vice versa).*

Kyoto's implementation (`utilities.py`), with `cpart`, `rpart` sorted non-increasing:

```python
def argd_inadmissible1(a,b, cpart,rpart):
    return sum(comb(c-1,a-1) for c in cpart[-rpart[0]:]) > (b-1)*comb(len(rpart)-1,a-1)
def argd_inadmissible(a,b, cpart,rpart):
    return argd_inadmissible1(a,b, cpart,rpart) or argd_inadmissible1(b,a, rpart,cpart)
def get_bipartitions(a,b, m,n, k):
    if a == b and m == n:
        combos = combinations_with_replacement(get_partitions(a,b, m,n, k), 2)
    else:
        combos = product(get_partitions(a,b, m,n, k), get_partitions(b,a, n,m, k))
    return list(filter(lambda ps: not argd_inadmissible(a,b, ps[0],ps[1]), combos))
```

So: `r = rpart[0]` = largest row sum; `cpart[-r:]` = the `r` smallest column sums; the test is `Σ C(c−1, a−1) > (b−1)·C(m−1, a−1)`; applied also transposed (largest column, `c` smallest rows, bound `(a−1)·C(n−1, b−1)`). **[inferred]** why the max row is the right one: the left-hand sum has `r` non-negative terms, so it is monotone in `r`; the right-hand side is constant; hence the row with most ones gives the strongest instance of D obtainable from the profile alone. Summing D over *all* rows with their true columns just gives `a` times Argument A (`c·C(c−1,a−1) = a·C(c,a)`), so D-on-the-max-row is genuinely extra information, not a restatement of A. **Note the square case:** `combinations_with_replacement` keeps only unordered `{cpart, rpart}` pairs when `a=b`, `m=n` — that is a *transposition symmetry reduction* (an "adding" move needing a transpose witness), not a prune; the Lean gate must not accept it as a kill.

### 4.3 Cardinality constraints (Section 3.2) — Sinz sequential counter, equality variant
**[quote]** *To express that exactly `k` out of `n` bits `b_1, …, b_n` should be true we use the equality variant of Sinz's sequential counter encoding [17] as described, tested and deemed fastest for general use among different cardinality constraint encodings by Wynn [21]. `k(n−k)` auxiliary variables `a_{i,j}` are used where `1 ≤ i ≤ k` and `1 ≤ j ≤ n−k`, with the following clauses (all literals `a_{i,j}` with `i` or `j` outside their specified ranges are dropped):*

```
⋀_{i=1}^{k}   ⋀_{j=1}^{n−k−1} ( ¬a_{i,j} ∨ a_{i,j+1} )
⋀_{i=0}^{k}   ⋀_{j=1}^{n−k}   ( ¬a_{i,j} ∨ a_{i+1,j} ∨ ¬b_{i+j} )
⋀_{i=1}^{k−1} ⋀_{j=1}^{n−k}   ( a_{i,j} ∨ ¬a_{i+1,j} )
⋀_{i=1}^{k}   ⋀_{j=0}^{n−k}   ( a_{i,j} ∨ ¬a_{i,j+1} ∨ b_{i+j} )
```

**[quote]** *This encoding has two desirable properties: (1) If a partial assignment of the `b_i` is such that said assignment cannot be completed without violating the cardinality constraint, unit propagation alone will lead to a contradiction (empty clause). (2) If exactly `k` of the `b_i` are assigned true, unit propagation alone will assign the other `b_i` false.* "Since unit propagation is hardwired into all state-of-the-art SAT solvers, using the above encoding should result in faster rejection of partially filled matrices that cannot be completed to an admissible matrix."

**[inferred] semantics** that makes the clauses check out: `a_{i,j} ⇔ "at least i of b_1..b_{i+j−1} are true"`. Family 1 is monotonicity in the prefix, family 2 is the counter step (`a_{i,j} ∧ b_{i+j} → a_{i+1,j}`), family 3 is `a_{i+1,j} → a_{i,j}`, family 4 is the backward step (`a_{i,j+1} ∧ ¬b_{i+j} → a_{i,j}`). The dropped out-of-range literals produce the bounds: family 2 at `i = k` becomes `¬a_{k,j} ∨ ¬b_{k+j}` (at most `k`), family 4 at `j = n−k` becomes `a_{i,n−k} ∨ b_{i+n−k}` (at least `k`). Kyoto's `zcnf.add_card_constraint_sinz(bits, k, comp)` emits the same four families and drops one of the two boundary families via `comp` to get one-sided variants; the paper and all drivers only use the equality variant. Cost per row/column: `k(n−k)` aux vars, `O(k(n−k))` clauses, `k = ` the fixed row/column sum. Our prior code (`gpt_agent/analysis/sat_attack/encodings_zar.py`) uses a two-directional unary counter (`exact_unary_counter`) which has `O(k·n)` registers instead of `k(n−k)`; Tan's is the cheaper, and his `column sum window` is one clause per column, not a global cardinality.

### 4.4 Lexicographic constraints (Section 3.3)
**[quote]** *Even with fixed row and column sums, there still remain the symmetries of swapping two rows or two columns with the same sum. These symmetries are broken by requiring groups of rows or columns with the same sum to be contiguous and lexicographically sorted; every (0,1)-matrix can be permuted to satisfy this property by the following theorem.*

**[quote] Theorem 3.2.** *Lexicographically sorting rows and columns of any (0,1)-matrix `A` alternately as in Figure 1 will reach a fixed point (both rows and columns sorted) in a finite number of steps. This remains true even if the sets of rows and columns are partitioned so that rows and columns cannot move across partitions.*

**[quote] Proof.** *With rows and columns indexed starting from 0, define `f(A) = Σ_i Σ_j 2^{i+j} a_{ij}`. Swapping rows/columns `a` and `b` where `a < b` but the numerical value `n_a` of column `a` is greater than `n_b` changes `f(A)` by `2^b n_a + 2^a n_b − 2^a n_a − 2^b n_b = (2^b − 2^a)(n_a − n_b) > 0`, i.e. sorting two out-of-order rows/columns strictly increases (or decreases, if sorting in reverse order) `f(A)`, which is clearly integral and bounded by 0 from below and `Σ_i Σ_j 2^{i+j}` from above. Since there are a finite number of possibilities for each value in the strictly monotone sequence of `f(A)`s generated, it must terminate at a point when `A` is sorted both in rows and columns.*

(Figure 1 shows a `5 × 5` example: sort rows, then columns, reaching a fixed point.)

**The at-most (lex) encoding [quote]:** *Given two equal-length strings of Boolean variables `a_1..a_n` and `b_1..b_n`, the binary number represented by the `a_i` may be constrained to be at most that represented by the `b_i` (where `a_1, b_1` are most significant) through `n−1` auxiliary variables `c_1..c_{n−1}` and the clauses (`c_0` and `c_n` are dropped)*

```
⋀_{i=1}^{n−2} ( ¬c_i ∨ c_{i+1} )
⋀_{i=1}^{n}   ( c_{i−1} ∨ ¬a_i ∨ b_i )
⋀_{i=1}^{n}   ( c_{i−1} ∨ a_i ∨ b_i ∨ ¬c_i )
⋀_{i=1}^{n}   ( c_{i−1} ∨ ¬a_i ∨ ¬b_i ∨ ¬c_i )
⋀_{i=1}^{n}   ( c_{i−1} ∨ a_i ∨ ¬b_i ∨ c_i )
```

**[inferred] semantics:** `c_i ⇔ "a < b has already been decided within the first i bits"`. While undecided (`¬c_{i−1}`): `a_i ≤ b_i`; if `a_i = b_i` then still undecided (`¬c_i`); if `a_i = 0, b_i = 1` then decided (`c_i`); and decided stays decided (family 1). **[quote]** *In our application of this form of symmetry breaking to the problem at hand the sort order is reversed: 1 comes before 0.* Kyoto: `add_comparator(bitfield[:,i+1], bitfield[:,i])` for consecutive columns with equal fixed sum (and likewise rows), i.e. non-increasing lex order within each equal-sum block. **[quote]** *The cardinality and lexicographic constraints do not remove all symmetries of a (0,1)-matrix (see figure 2 [two non-identical isomorphic maximal `8 × 8` matrices for `a=b=2` satisfying all constraints]) – doing so would require solving the graph isomorphism problem – but they are nevertheless very useful in reducing the number of instance solutions.*

### 4.5 Software (Section 3.4) [quote]
*All SAT solving was done with Kissat [2] on one laptop computer with the `--sat` and `--unsat` flags set according to whether or not a solution was expected, and no other settings touched. The maximal matrices in the `m = n` case were filtered to remove isomorphs using the `shortg` utility in nauty [13]; the automorphism groups of the corresponding bipartite graphs were computed using GAP. The partitioning and CNF-building code written for this project, together with the raw results obtained, is available in our Kyoto repository [18].* (Kyoto's `solve_cnf` calls `kissat -q --relaxed file.cnf [--sat|--unsat] [proof.drat]`, return code 10 = SAT; `find_all_solutions` blocks each found solution with its negation clause and re-solves until UNSAT — this is how "all maximal matrices" are enumerated.) The paper reports **no timings and no per-cell cube counts**; the counts below come from the Kyoto ledger.

## 5. Section 4 — tables

Conventions **[quote]**: *The following three tables are corrected and extended versions of the tables for `z_a(m,n)` given in Guy [7] where `a = 2, 3, 4`. Values above solid lines are both exact and given by theorem 2.2; the dashed lines indicate the limits of Guy's tables and grey backgrounds indicate errors Guy made. A bold value is exact, proven by the methods in this paper; other values are the upper bounds given by theorem 2.2.* Discussion: **[quote]** *there are only eight such errors, all in the `z_2(m,n)` table, and all are too low by just one* (grey cells in Table 2: `(14, 24..28)` and `(15, 15..17)`); no discrepancies with Héger's `z_2` table where Héger did not rely on Guy. *The link to finite geometries does not carry over to larger minor sizes, where the bound of theorem 2.2 appears to be less sharp, particularly when `m ≈ n`. Better bounds at the edges of the exact region can often be derived by applying the arguments of section 2 to eliminate all possible partitions and hence the need for any SAT solving; this gives for example `z_4(11,14) ≤ 106`.*

### 5.1 Table 3, `z_3(m,n)` (= `z(m,n;3,3)`), `m = 3..16`, `n = m..23`, as CSV

Transcription method: page 8 was parsed with `pdftohtml -xml`; boldness was taken from the font of each cell (`CMBX10` = bold, `CMR10` = roman), cells assigned to columns by order (row `m` has exactly `24−m` entries, `n = m..23`), and the result checked visually against a 300-dpi crop. Independent validation: every non-bold value equals Roman's bound recomputed from Theorem 2.2, and no bold value exceeds it. Result: **203 cells, 159 bold (exact in Tan), 44 non-bold (Roman upper bound only)**. First non-bold `n` per row: `m=9: 23; m=10: 21; m=11: 19; m=12..16: 17`. The same CSV is saved as `docs/lit/tan2022_table3_z3.csv`.

Columns: `tan_bold_exact` = 1 iff bold in Tan; `status` = `EXACT_TAN` (bold), `EXACT_LATER` (not bold in Tan but present in our `exact_table.csv`, i.e. settled after 2022), `OPEN_UB_ROMAN` (only Roman's upper bound known as of Tan).

```csv
m,n,tan_value,tan_bold_exact,in_exact_table_csv,exact_table_value,status
3,3,8,1,1,8,EXACT_TAN
3,4,10,1,1,10,EXACT_TAN
3,5,12,1,1,12,EXACT_TAN
3,6,14,1,1,14,EXACT_TAN
3,7,16,1,1,16,EXACT_TAN
3,8,18,1,1,18,EXACT_TAN
3,9,20,1,1,20,EXACT_TAN
3,10,22,1,1,22,EXACT_TAN
3,11,24,1,1,24,EXACT_TAN
3,12,26,1,1,26,EXACT_TAN
3,13,28,1,1,28,EXACT_TAN
3,14,30,1,1,30,EXACT_TAN
3,15,32,1,1,32,EXACT_TAN
3,16,34,1,1,34,EXACT_TAN
3,17,36,1,1,36,EXACT_TAN
3,18,38,1,1,38,EXACT_TAN
3,19,40,1,1,40,EXACT_TAN
3,20,42,1,1,42,EXACT_TAN
3,21,44,1,1,44,EXACT_TAN
3,22,46,1,1,46,EXACT_TAN
3,23,48,1,1,48,EXACT_TAN
4,4,13,1,1,13,EXACT_TAN
4,5,16,1,1,16,EXACT_TAN
4,6,18,1,1,18,EXACT_TAN
4,7,21,1,1,21,EXACT_TAN
4,8,24,1,1,24,EXACT_TAN
4,9,26,1,1,26,EXACT_TAN
4,10,28,1,1,28,EXACT_TAN
4,11,30,1,1,30,EXACT_TAN
4,12,32,1,1,32,EXACT_TAN
4,13,34,1,1,34,EXACT_TAN
4,14,36,1,1,36,EXACT_TAN
4,15,38,1,1,38,EXACT_TAN
4,16,40,1,1,40,EXACT_TAN
4,17,42,1,1,42,EXACT_TAN
4,18,44,1,1,44,EXACT_TAN
4,19,46,1,1,46,EXACT_TAN
4,20,48,1,1,48,EXACT_TAN
4,21,50,1,1,50,EXACT_TAN
4,22,52,1,1,52,EXACT_TAN
4,23,54,1,1,54,EXACT_TAN
5,5,20,1,1,20,EXACT_TAN
5,6,22,1,1,22,EXACT_TAN
5,7,25,1,1,25,EXACT_TAN
5,8,28,1,1,28,EXACT_TAN
5,9,30,1,1,30,EXACT_TAN
5,10,33,1,1,33,EXACT_TAN
5,11,36,1,1,36,EXACT_TAN
5,12,38,1,1,38,EXACT_TAN
5,13,41,1,1,41,EXACT_TAN
5,14,44,1,1,44,EXACT_TAN
5,15,46,1,1,46,EXACT_TAN
5,16,49,1,1,49,EXACT_TAN
5,17,52,1,1,52,EXACT_TAN
5,18,54,1,1,54,EXACT_TAN
5,19,57,1,1,57,EXACT_TAN
5,20,60,1,1,60,EXACT_TAN
5,21,62,1,1,62,EXACT_TAN
5,22,64,1,1,64,EXACT_TAN
5,23,66,1,1,66,EXACT_TAN
6,6,26,1,1,26,EXACT_TAN
6,7,29,1,1,29,EXACT_TAN
6,8,32,1,1,32,EXACT_TAN
6,9,36,1,1,36,EXACT_TAN
6,10,39,1,1,39,EXACT_TAN
6,11,42,1,1,42,EXACT_TAN
6,12,45,1,1,45,EXACT_TAN
6,13,48,1,1,48,EXACT_TAN
6,14,50,1,1,50,EXACT_TAN
6,15,53,1,1,53,EXACT_TAN
6,16,56,1,1,56,EXACT_TAN
6,17,58,1,1,58,EXACT_TAN
6,18,61,1,1,61,EXACT_TAN
6,19,64,1,1,64,EXACT_TAN
6,20,66,1,1,66,EXACT_TAN
6,21,69,1,1,69,EXACT_TAN
6,22,72,1,1,72,EXACT_TAN
6,23,74,1,1,74,EXACT_TAN
7,7,33,1,1,33,EXACT_TAN
7,8,37,1,1,37,EXACT_TAN
7,9,40,1,1,40,EXACT_TAN
7,10,44,1,1,44,EXACT_TAN
7,11,47,1,1,47,EXACT_TAN
7,12,50,1,1,50,EXACT_TAN
7,13,53,1,1,53,EXACT_TAN
7,14,56,1,1,56,EXACT_TAN
7,15,60,1,1,60,EXACT_TAN
7,16,63,1,1,63,EXACT_TAN
7,17,66,1,1,66,EXACT_TAN
7,18,69,1,1,69,EXACT_TAN
7,19,72,1,1,72,EXACT_TAN
7,20,75,1,1,75,EXACT_TAN
7,21,78,1,1,78,EXACT_TAN
7,22,81,1,1,81,EXACT_TAN
7,23,84,1,1,84,EXACT_TAN
8,8,42,1,1,42,EXACT_TAN
8,9,45,1,1,45,EXACT_TAN
8,10,50,1,1,50,EXACT_TAN
8,11,53,1,1,53,EXACT_TAN
8,12,57,1,1,57,EXACT_TAN
8,13,60,1,1,60,EXACT_TAN
8,14,64,1,1,64,EXACT_TAN
8,15,67,1,1,67,EXACT_TAN
8,16,70,1,1,70,EXACT_TAN
8,17,74,1,1,74,EXACT_TAN
8,18,77,1,1,77,EXACT_TAN
8,19,81,1,1,81,EXACT_TAN
8,20,84,1,1,84,EXACT_TAN
8,21,87,1,1,87,EXACT_TAN
8,22,90,1,1,90,EXACT_TAN
8,23,94,1,1,94,EXACT_TAN
9,9,49,1,1,49,EXACT_TAN
9,10,54,1,1,54,EXACT_TAN
9,11,59,1,1,59,EXACT_TAN
9,12,64,1,1,64,EXACT_TAN
9,13,67,1,1,67,EXACT_TAN
9,14,70,1,1,70,EXACT_TAN
9,15,73,1,1,73,EXACT_TAN
9,16,77,1,1,77,EXACT_TAN
9,17,81,1,1,81,EXACT_TAN
9,18,85,1,1,85,EXACT_TAN
9,19,89,1,1,89,EXACT_TAN
9,20,93,1,1,93,EXACT_TAN
9,21,96,1,1,96,EXACT_TAN
9,22,100,1,1,100,EXACT_TAN
9,23,104,0,0,,OPEN_UB_ROMAN
10,10,60,1,1,60,EXACT_TAN
10,11,64,1,1,64,EXACT_TAN
10,12,68,1,1,68,EXACT_TAN
10,13,73,1,1,73,EXACT_TAN
10,14,77,1,1,77,EXACT_TAN
10,15,81,1,1,81,EXACT_TAN
10,16,85,1,1,85,EXACT_TAN
10,17,90,1,1,90,EXACT_TAN
10,18,94,1,1,94,EXACT_TAN
10,19,98,1,1,98,EXACT_TAN
10,20,102,1,1,102,EXACT_TAN
10,21,108,0,0,,OPEN_UB_ROMAN
10,22,112,0,0,,OPEN_UB_ROMAN
10,23,116,0,0,,OPEN_UB_ROMAN
11,11,69,1,1,69,EXACT_TAN
11,12,74,1,1,74,EXACT_TAN
11,13,80,1,1,80,EXACT_TAN
11,14,84,1,1,84,EXACT_TAN
11,15,88,1,1,88,EXACT_TAN
11,16,92,1,1,92,EXACT_TAN
11,17,96,1,1,96,EXACT_TAN
11,18,101,1,1,101,EXACT_TAN
11,19,109,0,0,,OPEN_UB_ROMAN
11,20,113,0,0,,OPEN_UB_ROMAN
11,21,117,0,1,116,EXACT_LATER
11,22,121,0,0,,OPEN_UB_ROMAN
11,23,125,0,0,,OPEN_UB_ROMAN
12,12,80,1,1,80,EXACT_TAN
12,13,86,1,1,86,EXACT_TAN
12,14,91,1,1,91,EXACT_TAN
12,15,96,1,1,96,EXACT_TAN
12,16,99,1,1,99,EXACT_TAN
12,17,108,0,0,,OPEN_UB_ROMAN
12,18,113,0,0,,OPEN_UB_ROMAN
12,19,118,0,0,,OPEN_UB_ROMAN
12,20,122,0,0,,OPEN_UB_ROMAN
12,21,127,0,0,,OPEN_UB_ROMAN
12,22,132,0,1,132,EXACT_LATER
12,23,136,0,0,,OPEN_UB_ROMAN
13,13,92,1,1,92,EXACT_TAN
13,14,98,1,1,98,EXACT_TAN
13,15,104,1,1,104,EXACT_TAN
13,16,107,1,1,107,EXACT_TAN
13,17,117,0,0,,OPEN_UB_ROMAN
13,18,122,0,0,,OPEN_UB_ROMAN
13,19,126,0,0,,OPEN_UB_ROMAN
13,20,131,0,0,,OPEN_UB_ROMAN
13,21,136,0,0,,OPEN_UB_ROMAN
13,22,140,0,0,,OPEN_UB_ROMAN
13,23,145,0,0,,OPEN_UB_ROMAN
14,14,105,1,1,105,EXACT_TAN
14,15,112,1,1,112,EXACT_TAN
14,16,115,1,1,115,EXACT_TAN
14,17,125,0,0,,OPEN_UB_ROMAN
14,18,130,0,0,,OPEN_UB_ROMAN
14,19,136,0,0,,OPEN_UB_ROMAN
14,20,141,0,0,,OPEN_UB_ROMAN
14,21,146,0,0,,OPEN_UB_ROMAN
14,22,151,0,0,,OPEN_UB_ROMAN
14,23,155,0,0,,OPEN_UB_ROMAN
15,15,120,1,1,120,EXACT_TAN
15,16,123,1,1,123,EXACT_TAN
15,17,134,0,0,,OPEN_UB_ROMAN
15,18,139,0,0,,OPEN_UB_ROMAN
15,19,144,0,0,,OPEN_UB_ROMAN
15,20,150,0,0,,OPEN_UB_ROMAN
15,21,155,0,0,,OPEN_UB_ROMAN
15,22,160,0,0,,OPEN_UB_ROMAN
15,23,166,0,0,,OPEN_UB_ROMAN
16,16,128,1,1,128,EXACT_TAN
16,17,142,0,0,,OPEN_UB_ROMAN
16,18,148,0,0,,OPEN_UB_ROMAN
16,19,154,0,0,,OPEN_UB_ROMAN
16,20,160,0,0,,OPEN_UB_ROMAN
16,21,165,0,0,,OPEN_UB_ROMAN
16,22,170,0,0,,OPEN_UB_ROMAN
16,23,176,0,0,,OPEN_UB_ROMAN
```

### 5.2 Cross-check against `gpt_agent/data/exact_table.csv` (161 cells)

- All 159 bold cells of Tan's Table 3 are in `exact_table.csv` with **identical values** (0 disagreements).
- `exact_table.csv` has exactly two cells that are **not bold in Tan**: `(11,21)`: Tan lists the Roman bound **117**, `exact_table.csv` says **116** (so a post-2022 upper-bound improvement plus a matching construction); `(12,22)`: Tan lists Roman bound **132**, `exact_table.csv` says **132** (a post-2022 construction met Roman's bound). Neither is a disagreement with Tan — Tan never claimed those cells exact — but their provenance must be documented from the later sources in the proposal ([11] Padhi, [13] Hou, [14] Afrasyab) before they are used as ground truth. This is the "159/161" split recorded in the project memory.
- No cell of `exact_table.csv` lies outside Tan's Table 3 range, and no bold cell of Tan is missing from it.
- Bold cells strictly below Roman's bound (i.e. where real upper-bound work was needed): 82 of 159; bold cells equal to Roman but below the packing threshold: `(6,6)=26, (6,9)=36, (9,12)=64`; exact by Roman alone: 74.

### 5.3 Difficulty data from the Kyoto ledger (`kyoto/data/3x3`, format: `z(3,3,m,n) < k` header followed by one line per partition pair `(cpart) (rpart)` handed to Kissat; all UNSAT)

Surviving pairs after arguments A, D, E(I) for each proved `z_3` upper bound (0 = no SAT call needed at all):

| m | n : pairs | | |
|---|---|---|---|
| 6 | 6:0, 7:1, 8:1, 9:0, 10:0 | | |
| 7 | 7:0, 8:0, 9:2, 10:0, 11:4, 12:9, 13:**24**, 14:9, 15:0, 16:4, 17–22: 0 | | |
| 8 | 8–14: 0, 15:7, 16:19, 17:0, 18:18, 19:0, 20:6, 21:30, 22:**46**, 23:0, 24:24, 25:12, 26:0, 27:0 | | |
| 9 | 9:7, 10–13: 0, 14:7, 15:**123**, 16–20: 0, 21:16, 22:0 | | |
| 10 | 10:0, 11:1, 12:14, 13–19: 0, 20:4 | | |
| 11 | 11:8, 12–14: 0, 15:12, 16:68, 17:**290**, 18:0 | | |
| 12 | 12–15: 0, 16:**262** | | |
| 13 | 13:1, 14–16: 0 | | |
| 14 | 14:0, 15:0, 16:0 | | |
| 15 | 15:0, 16:0 | | |
| 16 | 16:**28** (`z_3(16,16) < 129`) | | |

(For `m = 8` the ledger goes to `n = 27`, beyond the printed table; e.g. `z_3(8,24) < 98` needed 24 pairs.) Pattern **[inferred]**: the hard cells are the last one or two exact cells in each row before the Roman region (`(9,15)`, `(11,17)`, `(12,16)`), i.e. where `w` is far below Roman's bound yet not excluded by counting; cells right at the packing edge (`(m, n)` with `n` near `packlimit`) need zero cubes. Lower-bound side (`>= k` entries): `z_3(9,9)=49` listed 30 pairs and 17 solutions before isomorph filtering (7 non-isomorphic maximal matrices in the paper's listing), `z_3(10,10)=60`: 1 pair `(6^10)`, 16 solutions (1 non-isomorphic), `z_3(11,11)=69`: 58 pairs, `z_3(12,12)=80`: 61 pairs, `z_3(14,14)=105`: 28 pairs, `z_3(16,16)=128`: 23 pairs, 1 matrix.

Order of proof in `driver33.py`: rows `m = 6, 7, …, 16`, for each `m` the square cell first (all solutions), then `n = m+1, m+2, …` with `prove_solutions(3,3,m,n,z+1)` (UNSAT) then `find_solution(3,3,m,n,z)` (SAT), appending to the ledger, so that argument E for later cells reads earlier `< k` entries through `zbounds`. This is a **DAG of dependencies between cells** that any Lean closure must reproduce (Argument I's `z_{a,b}(m',n')` must itself be a proved theorem).

## 6. Section 5 — maximal square matrices, `a = 3` (row/column sums only; useful profile intuition)

| `(a,m)` | `z_a(m)` | row sums = column sums of the maximal matrices (multiplicities as exponents), automorphism group |
|---|---|---|
| (3,3) | 8 | `(3^2, 2)`, `D4` order 8 |
| (3,4) | 13 | `(4, 3^3)`, `D6` order 12 |
| (3,5) | 20 | `(4^5)`, `S5 × C2` order 240 |
| (3,6) | 26 | `(5^2, 4^4)`, `D8 × C2` order 32 |
| (3,7) | 33 | `(6, 5^3, 4^3)`, `D6` order 12 |
| (3,8) | 42 | `(7, 5^7)`, `PSL(3,2) ⋊ C2` order 336 |
| (3,9) | 49 | seven matrices: four with `(6^4, 5^5)` (`C2^2`, `D4`, `D8`, `S4 × C2`), one with rows `(7,6^3,5^4,4)` / cols `(6^4,5^5)` (`S3`), one `(7,6^2,5^6)` (`D8`), one `(7,6^3,5^4,4)` (`D6`) |
| (3,10) | 60 | `(6^10)`, `S5 × C2` order 240 |
| (3,11) | 69 | `(7^4, 6^6, 5)`, `D6` order 12 |
| (3,12) | 80 | two: `(7^8, 6^4)` (order 384) and `(8, 7^6, 6^5)` (`D4 × C2`) |
| (3,13) | 92 | `(8^3, 7^8, 6^2)`, order 64 |
| (3,14) | 105 | `(8^7, 7^7)`, `PSL(3,2) ⋊ C2` order 336 |
| (3,15) | 120 | `(8^15)`, `S8` order 40320 |
| (3,16) | 128 | `(8^16)`, order 43008 (Kyoto's logo; the paper marks it `*` = no symmetric presentation) |

**[inferred]** Extremal profiles are near-regular (Argument B), with at most two or three distinct sums; the cubes that survive A/D/I but are UNSAT are the "almost regular" ones just beside these.

## 7. Relevance to our design (ZarPrune + OpenEvolve)

### 7.1 What is a *prune* and what is an *adding* step, in Tan
| Tan's step | kind | Lean status |
|---|---|---|
| Argument A (both orientations) at every prefix of Algorithm 1 | prune (kills genuinely empty cases) | not yet proved (`README`: "the real prune") |
| Argument I / E at every prefix | prune, **parametric in a proved bound for a smaller cell** | not yet proved; needs the DAG |
| Argument D on (max row, `r` smallest columns) and transpose | prune on the pair | not yet proved |
| pigeonhole shortcut, lines 8–12 | pure enumeration optimisation (no case removed) | part of `cover` |
| sorted (unordered) partitions | **adding** (row/column permutation) | needs a permutation lemma in `cover` (§7.3) |
| exactly-`w` instead of at-least-`w` | **adding** (deletion of ones) | needs a deletion lemma in `cover` (§7.3) |
| unordered `{cpart,rpart}` when `a=b, m=n` | **adding** (transposition) | must not be a kill; SR/transpose witness |
| lex order inside equal-sum blocks | **adding** (Theorem 3.2) | inside the CNF / certificate, not the gate |
| Roman's bound as the starting `w` | instance-level bound | not needed by the gate if `w` is chosen by hand |

### 7.2 A profile-level prune is the "most pessimal choice" made computable
Every one of Tan's arguments has the shape "for the true (unknown) placement of ones, `Σ f(...) ≤ B`"; the profile-level kill replaces the unknown placement by the choice that minimises the left side (largest parts for prefix checks, smallest columns for D). The soundness proof therefore always has two halves: (i) the combinatorial inequality for the true placement, (ii) an *extremal-selection lemma* ("the sum over the `r` smallest values is ≤ the sum over any `r` values" / "the sum of the `n'` largest is ≥ the sum over any `n'`"). Half (ii) is shared by all of them and should be proved once as a library lemma over `Fin n → Nat` (core Lean 4.34 has `List.mergeSort` and `List.mergeSort_perm`, but *no* `Nat.choose`, no `List.Sorted`, and the sortedness lemma name differs from Mathlib's — checked with `lake env lean`; a local `choose` via Pascal's rule and a small sorted-prefix library are needed).

### 7.3 The `cover` obligation is where Tan's symmetry reductions live
`upper_bound_of_cover` needs `∀ A, Valid P A → kill (profileOf A) ∨ profileOf A ∈ survivors`. Tan's survivors are *sorted* pairs with total exactly `w`. So `cover` cannot be discharged literally; it needs (a) `Valid` is invariant under row/column permutations (with `HasKst` as increasing tuples this needs "an injective tuple can be re-indexed increasing"), and (b) from `weight A ≥ w` obtain `A' ≤ A` entrywise with `weight A' = w` and `¬HasKst A'` (monotonicity of `HasKst` under entrywise `≤`, plus a deletion induction). Then survivors can be sorted profiles of total `w`, and the enumeration (Algorithm 1 + Theorem 3.1) must be certified complete: either a verified enumerator in Lean, or `cover` restated as "every sorted profile of total `w` with parts in `[0,m]`/`[0,n]` is killed or listed" and decided by `decide`/a reflective check over the finite list of partitions (the list is small: hundreds to low thousands of pairs for our `m,n ≤ 16..23`).

### 7.4 What Tan gives the reward function
The evaluator can score an evolved prune by (1) Lean acceptance (binary), (2) number of Tan-surviving pairs it additionally kills on frontier cells (`(9,15)`: 123, `(11,16)`: 68, `(11,17)`: 290, `(12,16)`: 262, `(16,16)`: 28 pairs in Kyoto's ledger — reproducible with `get_bipartitions`), (3) whether it kills *all* pairs of an open cell (then the bound is proved with no SAT at all, as with `z_4(11,14) ≤ 106`), (4) estimated SAT cost of the killed pairs. Pairs surviving A/D/I but UNSAT are exactly the training signal: their profiles are known (ledger), so an evolved prune can be tested for "kills known-UNSAT pairs" *before* any SAT call.

## 8. Pruning-lemma ledger (statements over ZarPrune; provability estimates are mine)

Below `choose` is a Pascal-recursion binomial to be added to `Sum.lean`; `P : Params`, `A : Mat P.m P.n`, `pf = profileOf A`.

**L-A (Argument A / Kővári–Sós–Turán counting), column form.**
Hypotheses: `¬ HasKst P A`. Statement: `sumFin P.n (fun j => choose (colSum A j) P.s) ≤ (P.t − 1) * choose P.m P.s`. Kill: `decide ((P.t−1) * choose P.m P.s < sumFin P.n (fun j => choose (pf.col j) P.s))`. Row form: `sumFin P.m (fun i => choose (rowSum A i) P.t) ≤ (P.s − 1) * choose P.n P.t`.
Proof: double count pairs (increasing `s`-tuple `R` of rows, column `j`) with all ones; per `R` at most `t−1` columns (else `HasKst`); per `j` exactly `choose (colSum j) s` tuples.
Lean, Mathlib-free: **medium-hard**. Needs an enumeration of increasing `s`-tuples of `Fin m` (Pascal recursion `tuples (m+1) (s+1) = (0 :: tuples m s) ++ tuples m (s+1)` after shifting), `length = choose m s`, the per-column identity "number of increasing tuples inside the ones of column `j` = `choose (colSum j) s`" (induction on `m` following the recursion), and Fubini over a list instead of `Fin` (or index tuples by `Fin (choose m s)`). Estimate 300–600 lines; with Mathlib ≈ 50 (`Finset.powersetCard`, `sum_comm`, `card_powersetCard`). This is the one to build first; D and I reuse it.

**L-D (Argument D, profile form).**
Let `r = max_i pf.row i` (require `r ≤ P.n`, else `rowCap` fires); let `g c = choose (c − 1) (P.s − 1)`; let `S_r` = sum of the `r` smallest values of `g ∘ pf.col`. Kill: `decide ((P.t − 1) * choose (P.m − 1) (P.s − 1) < S_r)`. Transpose analogously.
Soundness: pick `i0` with `rowSum A i0 = r`; let `J = {j | A i0 j}` (`|J| = r`); the `(m−1) × r` matrix `B` (delete row `i0`, keep `J`) is `K_{s−1,t}`-free (an `(s−1) × t` all-one minor of `B` plus row `i0` is an `s × t` minor of `A` — needs inserting `i0` into an increasing `(s−1)`-tuple), with column sums `colSum A j − 1`; apply L-A to `B`; then `Σ_{j∈J} g(c_j) ≥ S_r` by the selection lemma.
Lean, Mathlib-free: **hard** (after L-A: medium). Ingredients: L-A stated for arbitrary `m', n'` (free, it is parametric), an increasing enumeration `Fin r → Fin n` of `J` (filter over `Fin n`), insertion into increasing tuples, deletion of one row, the selection lemma via `mergeSort`. Estimate 400+ lines.

**L-I (Argument I / Guy's E, inclusion), column form.**
Hypotheses: a **proved** theorem `hz : ∀ B : Mat P.m n', ¬ HasKst ⟨P.m, n', P.s, P.t, _⟩ B → weight B ≤ z'` for some `n' ≤ P.n`. Kill: `decide (z' < (sum of the n' largest values of pf.col))`.
Soundness: restrict `A` to those `n'` columns via an increasing `e : Fin n' → Fin P.n`; `HasKst` of the restriction gives `HasKst` of `A` (compose increasing maps); `weight (restriction) = sumFin n' (colSum A ∘ e)` (by `weight_eq_sum_colSum`); contradiction with `hz`.
Lean, Mathlib-free: **medium** (150–250 lines): column restriction, `Incr` composition (trivial), weight identity, selection lemma. Design consequence: the prune is a *combinator* taking a proof term for a smaller cell — the harness must keep a table of proved cells and the Lean file for cell `(m,n)` imports the theorems of the cells it cites (Tan's ledger order `m` increasing, `n` increasing, plus transposes).

**L-GR (Gale–Ryser necessity; NOT in Tan; candidate evolved prune).**
Statement: for every `k ≤ P.n`, (sum of the `k` largest column sums) `≤ sumFin P.m (fun i => min (rowSum A i) k)`. Kill if violated for some `k` (also transposed). Soundness: each row meets any `k` columns in at most `min(r_i, k)` ones. Lean, Mathlib-free: **easy-medium** (100–200 lines; no `choose`, no `HasKst`). By Gale–Ryser this is exactly the set of realisable profile pairs, so it is the ceiling of `K_{s,t}`-blind pruning; it may or may not kill anything beyond A/D/I on Tan's frontier pairs — cheap to test with `get_bipartitions`.

**L-B (Argument B).** Not a prune (a statement about the minimiser of the A-sum). Skip.

**L-Roman (Theorem 2.2).** Instance-level: if `P.w > Roman(m,n)` kill everything. Proof is a weighted counting argument; **hard** and unnecessary if `w` is chosen at or below Roman's bound by hand. Skip for the gate; use in the harness to choose `w`.

**L-sort / L-lex / L-transpose.** Not prunes (`Demo.notDescending_unsound`). They belong to `cover` (permutation invariance of `Valid`) and to the CNF/certificate.

**L-delete (exactly-`w`).** `Valid P A ∧ weight A > w → ∃ A', (∀ i j, A' i j → A i j) ∧ weight A' = w ∧ ¬HasKst P A'`. **Easy-medium**, needed for `cover` only.

## 9. Difficulty signals available from this source
1. Number of partition pairs surviving A/D/I (Tan's cubes): 0–290 for `z_3`; 0 means proved without SAT.
2. Argument-A slack per pair, `(t−1)C(m,s) − Σ C(c_j,s)` (and row/D slacks): slack 0 with equal-ish sums is the maximal regime (Argument B); small positive slack = near-extremal = typically the surviving-but-UNSAT cubes.
3. Distance of `w` from Roman's bound and from the packing threshold `n ≥ packlimit(m)` (above it: free).
4. Instance size: `m·n` cell variables, `C(m,s)·C(n,t)` minor clauses, `Σ_i r_i(n−r_i) + Σ_j c_j(m−c_j)` Sinz auxiliaries, plus `(m−1)+(n−1)` comparators per equal-sum block.
5. Number of solutions on the SAT side (blocking-clause loop): 1 vs 16–17 signals how "tight" the neighbourhood is.
6. Position in the row: in every `m`-row the hardest cells are the last exact ones before the Roman region.

## 10. Open questions / caveats
- The paper gives no wall-clock times; the only cost proxy is cube count (from Kyoto) and the fact that everything ran on one laptop.
- DRAT proofs were produced by the driver but there is no evidence they were checked; our pipeline will need checked LRAT/SR certificates anyway.
- Tan applies D only to the single max row (with the `r` smallest columns). Whether a multi-row or flow/LP relaxation (rows must place their ones in columns whose remaining capacity is bounded) kills extra pairs on `(11,17)`/`(12,16)` is an experiment we can run in minutes with `get_bipartitions`.
- `(11,21)` and `(12,22)` in `exact_table.csv` are post-Tan; provenance to be pinned to [11]/[13]/[14].
- Kyoto's `packable_simplices` uses `T(m−1)` as a fallback lower bound on `T_{3,3}(m)` for `m` not in its table — sound but possibly loose; irrelevant for `n ≤ 23`.
- The `T_{2,b}` second-case formula is transcribed from a typeset fraction and should be re-checked against Guy 1967 before any use (we do not need it for `a=b=3`).
- Tan's Algorithm 1 includes zero parts (truncated partitions); our prior `profiles.py` restricts column weights to `[2,m]` — that is an *adding* step (raising light columns), fine inside a certificate but not a prune.

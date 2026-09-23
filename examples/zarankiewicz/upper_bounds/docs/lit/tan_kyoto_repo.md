# Tan's "Kyoto" repository — the SAT Zarankiewicz attack, as code

**Source.** Jeremy Tan (GitHub: Parcly-Taxel), *Kyoto: Exact values and maximal graphs for Zarankiewicz's problem*, https://github.com/Parcly-Taxel/Kyoto . MIT licence (c) 2022 Parcly Taxel / Jeremy Tan. Cited as [18] in Tan, *An attack on Zarankiewicz's problem through SAT solving*, arXiv:2203.02283 (v2, 19 Apr 2022), which is [8] in the thesis proposal. The repo also carries the paper's TeX/PDF and the FYP (NUS final-year project) presentation.

**Snapshot used.** Full clone (41 commits, 2022-01-16 → 2022-04-20), HEAD `3acbc6109ce64388370627c7c3016bd519c7fefa` ("Shift another URL"). Local copy: `/private/tmp/claude-501/-Users-jaybhan-Downloads-openevolve/88cfd163-c49a-4b7d-ba6f-24d477547106/scratchpad/lit/Kyoto`. Everything below was read from the code or reproduced by running it; statements marked **Inference** are mine, not Tan's.

**Convention.** Tan writes `z(a,b, m,n)` = max ones in an `m x n` 0/1 matrix with no all-one `a x b` minor (`a` rows, `b` columns). In ZarPrune terms `s = a`, `t = b`, `P.m = m`, `P.n = n`. `k` or `w` is the assumed number of ones; an upper-bound proof shows the `w`-instance is UNSAT, i.e. `z < w`.

---

## 1. File layout

| Path | Size | Role |
|---|---|---|
| `readme.md` | 2.3 KB | Origin story (MSE question → OEIS A350237 → first new A001198 term in 50 years → FYP topic). Says `data/` records "both maximal matrices in a base64 format and proofs of impossibility as the cases considered". Assumes a compiled Kissat at repo top level. |
| `kyoto/zcnf.py` | 6.6 KB | `zaran_cnf` class: variable layout, minor clauses, Sinz exact-cardinality, lexicographic comparator, cube fixing, DIMACS writer, blocking clauses, `find_all_solutions`; `solve_cnf` wrapper around Kissat. |
| `kyoto/utilities.py` | 4.8 KB | Roman's bound, `T_{a,b}(m)` packing numbers, `zbounds` (bound lookup incl. the data files), `get_partitions` (Algorithm 1), `argd_inadmissible` (Guy's argument D + transpose), `get_bipartitions`, `sort_icdv`. |
| `kyoto/prove.py` | 1.7 KB | `prove_solutions` (the case-split driver: all pairs → CNF → all solutions), `find_solution` (first SAT pair), `extend_solution` (cube from a partial matrix). |
| `kyoto/graphs.py` | 1.4 KB | `encode_array` / `decode_array` (base64 matrix codes used in the paper's tables), `tograph6`, `Btog6` (for nauty `shortg`). |
| `kyoto/driver22.py`, `driver33.py`, `driver44.py` | 8/4/3 KB | Scripts that ran the whole campaign for 2x2, 3x3, 4x4 minors; almost entirely commented out in the final state (they were re-run incrementally). They contain the `nonzeros` dictionaries (expected solution counts per pair) and the only hardness remarks in the repo. |
| `kyoto/verifygraphs23.py` | 1.7 KB | Secondary split for the `z_2(23) = 115` uniqueness proof: 12 box-sum cases on top of the `(5^23)(5^23)` profile pair. |
| `kyoto/data/2x2`, `3x3`, `4x4` | 66/114/57 KB | The raw result logs: one record per proved bound, listing the surviving (column partition, row partition) cases and, for lower bounds, the solutions. No timings. |
| `paper/zarankiewicz.tex/.pdf`, `paper/pres.tex/presentation.pdf`, `paper/refs.bib` | | The arXiv paper (v2) and the FYP Beamer talk. |
| `logo.svg` | | The conjectured-unique `z_3(16) = 128` matrix. |

Historical files (deleted before HEAD): `partition.py`, `getbound.py`, `zarankiewicz.py`, `starbattle.py`, top-level `driver.py`/`prove.py`/`utilities.py`/`zcnf.py` (moved into `kyoto/` on 2022-02-25). Notable commits: `e449c00` 2022-01-19 "Add row partition considerations and argument D" (before this, only column partitions were split on and rows were free — the old-format records in `data/2x2`); `2be1f09` "GUY IS WRONG" (found Guy's `z_2(14,24)` error); `1d0ed99` "z(16,32) < 104"; `63abf46` "Literally thousands of solutions for z_2(22)"; `37c49e2` "z_2(23) = 115 …but HOW MANY SOLUTIONS?!"; `b32dafd` "z_3(16) = 128".

The final `prove.py` has signature `prove_solutions(a,b, m,n, k, nonzeros={}, icdv=False)`; several driver lines still call an older 4-extra-argument form and a `prove_solutions2` that no longer exists (they are inside the commented-out blocks). The drivers are a log, not runnable code.

---

## 2. The pipeline end to end

```
prove_solutions(a,b, m,n, k, nonzeros={}, icdv=False):
  header = "z(a,b,m,n) < k"  (or ">= k" if nonzeros is truthy)
  for (i, (cpart, rpart)) in enumerate(get_bipartitions(a,b, m,n, k)):
      [optional] cpart, rpart = sort_icdv(cpart), sort_icdv(rpart)
      log line "(cpart) (rpart)"
      cnf = zaran_cnf(a,b, m,n)          # all C(m,a)*C(n,b) minor clauses
      cnf.set_col_counts(cpart)          # exact column sums + lex within equal-sum runs
      cnf.set_row_counts(rpart)          # exact row sums    + lex within equal-sum runs
      for sol in cnf.find_all_solutions(fn, nonzeros.get((cpart,rpart),0), fn):
          log encode_array(sol)          # each solution, then a blocking clause is added
  return the log (the driver appends it to kyoto/data/<a>x<b>)
```

`fn = f"{a}x{b} {m}x{n} {k} {i}"` (with spaces) names `<fn>.cnf` and `<fn>.drat` in the working directory. An upper-bound record whose header is followed by *no* case lines means **every** partition pair was killed by counting and no SAT solver was ever invoked.

---

## 3. Partition generation (`utilities.get_partitions`) — Algorithm 1 of the paper

`get_partitions(a,b, m0,n0, k0)` returns all partitions of `k0` into `n0` parts in `[0, m0]`, listed non-increasing, that survive Guy's arguments A and I (the docstring calls I "E"). It is used for column partitions as `get_partitions(a,b, m,n, k)` and for row partitions with everything transposed, `get_partitions(b,a, n,m, k)`.

```python
Alim  = (b-1) * comb(m0, a)
Elims = {n: zbounds(a,b, m0,n)[1] for n in range(b, n0)}     # upper bounds z(m0, n') for b <= n' < n0
def P(k, m, n, partial=[]):
    if sum(comb(part, a) for part in partial) > Alim \
       or any(sum(partial[:j]) > Elim for (j, Elim) in Elims.items() if j <= len(partial)):
        return []                                             # prune the whole subtree
    if k == 0:
        return [tuple(partial)] if all(sum(partial[:j]) <= Elim for ...) else []
    if (d := k - (m-1)*n) > 0:                                # pigeonhole: d more parts must equal m
        completes.extend(P(k-d*m, m, n-d, partial + [m]*d))
    else:
        for part in range(-(-k//n), min(k,m)+1):              # next part in [ceil(k/n), min(k,m)]
            completes.extend(P(k-part, part, n-1, partial + [part]))
```

Facts about it:

* **Argument A prefix test.** `Σ_i C(p_i, a) ≤ (b−1)·C(m0, a)` is checked on every prefix; since parts are non-increasing, the first `j` parts are the `j` largest, so the prefix is the pessimal `m0 x j` sub-choice and the check is monotone along the recursion (Theorem 3.1 of the paper).
* **Argument I prefix test** (`Elims`). The sum of the first `j` parts (= number of ones in the `m0 x j` minor on the `j` heaviest columns) must be `≤ z_{a,b}(m0, j)`. The bound used is `zbounds(a,b,m0,j)[1]`, which is `min(Roman's bound, every "z(a,b,m0,j) < w" record found in kyoto/data/<a>x<b>)`. **So the partition generator is recursive in the campaign's own SAT-proved results** — the pruning power of argument I comes overwhelmingly from exact smaller values (Section 8 below quantifies this).
* Output order is lexicographic; the pigeonhole shortcut (`d > 0`) avoids recursion when the next parts are forced to be `m`.
* Trailing zero parts are **dropped** (e.g. `get_partitions(2,2,3,3,2) = [(1,1), (2,)]`). `set_col_counts` then raises `IndexError` on a short partition. This never bit the published runs because for a target `w > z(m, n−1)` the Elim at `j = n−1` kills every partition with a zero part; but note `argd_inadmissible1` uses `len(rpart)` as `m`, which would be *unsound* (RHS too small) on a short partition. Edge case only; recorded here because it is exactly the kind of hazard the Lean gate exists for.

## 4. Pruning partition pairs (`argd_inadmissible`, `get_bipartitions`)

```python
def argd_inadmissible1(a,b, cpart,rpart):
    # Guy's argument D on the row with the most ones, with the pessimal (smallest) columns
    return sum(comb(c-1,a-1) for c in cpart[-rpart[0]:]) > (b-1)*comb(len(rpart)-1,a-1)
def argd_inadmissible(a,b, cpart,rpart):
    return argd_inadmissible1(a,b, cpart,rpart) or argd_inadmissible1(b,a, rpart,cpart)   # D and D'
def get_bipartitions(a,b, m,n, k):
    if a == b and m == n:
        combos = combinations_with_replacement(get_partitions(a,b, m,n, k), 2)   # unordered pairs
    else:
        combos = product(get_partitions(a,b, m,n, k), get_partitions(b,a, n,m, k))
    return list(filter(lambda ps: not argd_inadmissible(a,b, ps[0],ps[1]), combos))
```

* `rpart[0]` is the largest row sum `r`; `cpart[-r:]` are the `r` smallest column sums. The paper: "considering the row with the most, r, ones and the r columns with the least ones – if argument D fails for this most pessimal column choice it must also fail for all other column choices – and vice versa".
* **Transposition symmetry.** When `a=b` and `m=n` only *unordered* pairs `{cpart, rpart}` are generated: the case `(Q,P)` is skipped because `(P,Q)` is solved. This is a symmetry ("adding") move, not a prune — see Section 10.
* `sort_icdv(part)` re-orders a partition by *increasing count of occurrences, then decreasing value* (e.g. `(6,2,5,5,…,5)`); used for `z_2(22) ≥ 108` and `z_4(11) ≥ 86`. It changes only the order of the parts (hence which rows/columns are adjacent for the lex constraints and the solver's variable order). Tan gives no rationale; **Inference:** a heuristic to put the "unusual" rows/columns first.
* Nothing else prunes pairs. There is no joint row/column counting beyond D/D'.

## 5. CNF construction (`zcnf.zaran_cnf`)

**Variable layout.** `bitfield = np.arange(m*n).reshape((m,n)) + 1`: cell `(i,j)` (0-based) is variable `i*n + j + 1`, row-major, 1-based. Auxiliary variables are allocated from `cursor = m*n + 1` upward, in the order constraints are added.

**Minor clauses (constructor).** For every increasing `a`-tuple of rows and increasing `b`-tuple of columns, one clause `∨_{i,j} ¬x_{r_i c_j}`. Count `C(m,a)·C(n,b)` (e.g. `z_3(16,16)`: 313,600; `z_2(24,24)`: 76,176; `z_4(13,13)`: 511,225). This matches ZarPrune's `HasKst` with `Incr` index tuples exactly.

**Exact cardinality (`add_card_constraint_sinz(bits, k, comp=0)`).** Sinz sequential counter, "equality variant … as described … by Wynn" (arXiv:1810.12975). For `N = len(bits)`, aux `a_{i,j}` for `1 ≤ i ≤ k`, `1 ≤ j ≤ N−k` at index `cursor + k(j−1) + (i−1)`; `cursor += k(N−k)`. Clauses (any literal with `i` or `j` out of range is dropped), exactly the paper's four families:

```
(1) ¬a_{i,j} ∨ a_{i,j+1}                    1≤i≤k,   1≤j≤N−k−1
(2) ¬a_{i,j} ∨ a_{i+1,j} ∨ ¬b_{i+j}         0≤i≤k,   1≤j≤N−k
(3)  a_{i,j} ∨ ¬a_{i+1,j}                   1≤i≤k−1, 1≤j≤N−k
(4)  a_{i,j} ∨ ¬a_{i,j+1} ∨ b_{i+j}         1≤i≤k,   0≤j≤N−k
```

**Inference (semantics):** `a_{i,j} ⇔ (b_1 + … + b_{i+j−1} ≥ i)`; the dropped `a_{k+1,j}` (false) is the at-most side, the dropped `a_{i,N−k+1}` (true) is the at-least side. The paper claims the encoding is unit-propagation complete in both directions (contradiction from an uncompletable partial assignment; forcing the rest to 0 once `k` are true).

I brute-forced the encoding for `N ≤ 5`, all `k`, all 2^N input assignments: **`comp = 0` (exactly k) is correct in every case** (this is the only mode the pipeline uses). The docstring says `comp = 1` is "at least" and `comp = −1` "at most"; the code does the **opposite** (`comp=+1` drops family (4) at `j=N−k` → at-most; `comp=−1` drops family (2) at `i=k` → at-least). Harmless for Tan's results, but do not copy the docstring.

**Lexicographic comparator (`add_comparator(less_b, greater_b)`).** Encodes `less ≤ greater` as MSB-first binary numbers with `N−1` aux `c_1..c_{N−1}` (`c_i ⇔` "strictly less already established within the first i bits"; **inference**), clauses as in paper §3.3:

```
¬c_i ∨ c_{i+1}   (1≤i≤N−2);   c_{i−1} ∨ ¬a_i ∨ b_i;   c_{i−1} ∨ a_i ∨ b_i ∨ ¬c_i;
c_{i−1} ∨ ¬a_i ∨ ¬b_i ∨ ¬c_i;   c_{i−1} ∨ a_i ∨ ¬b_i ∨ c_i        (c_0, c_N dropped)
```
Brute-force verified for `N ≤ 4` (all 4^N inputs). `set_col_counts(counts)` adds an exact-`k` constraint on every column with `count ≥ 0` (`−1` = unconstrained) and, for every adjacent pair with `counts[i] == counts[i+1]`, `add_comparator(col_{i+1}, col_i)`: within a run of equal-sum columns the columns are non-increasing as binary numbers read top-down (so 1 sorts before 0 — the "reverse" order of the paper). Same for rows. Soundness is paper Theorem 3.2 (alternating sorting of same-sum rows/columns reaches a fixed point, `f(A) = Σ 2^{i+j} a_ij` strictly monotone). The paper stresses this does **not** remove all isomorphs (Figure 2).

**Other members.** `set_cubes(B)` adds unit clauses fixing the cells of a (possibly smaller, `−1`-padded) matrix `B` and returns the free cell variables — `extend_solution` then puts an exact-`k` constraint on the free cells: a partial-assignment cube, the direction Tan names as future work. `write()` emits DIMACS with `nvars = max |literal|`. `add_solution(sol)` appends the blocking clause `∨ ¬(sol restricted to the m·n grid literals)` (aux variables excluded, so "all solutions" means all grids) to the clause list *and to the .cnf file*, so the final DRAT refutes "CNF ∧ all blocking clauses" — a proof that the enumeration is complete.

**Instance size example** (`z_3(7,7)`, profile `(6,5,5,5,4,4,4)` both ways): 49 grid vars, 1,225 minor clauses; after cardinality + lex: 2,041 clauses, 241 variables.

## 6. Solver options (`zcnf.solve_cnf`)

```python
cline = [solver_path, "-q", "--relaxed", f"{cnf_path}.cnf"]
if orient > 0: cline.append("--sat")      # expected satisfiable
if orient < 0: cline.append("--unsat")    # expected unsatisfiable
if proof_path: cline.append(f"{proof_path}.drat")
proc = run(cline, capture_output=True, encoding="utf-8")
if proc.returncode != 10: return None     # <-- anything but SAT is reported as "no solution"
return np.array(list(map(int, lit_re.findall(proc.stdout)[:-1])))
```

* Solver: **Kissat** only, single laptop, `--sat` / `--unsat` "according to whether or not a solution was expected, and no other settings touched" (paper §3.4). `find_all_solutions` sets `orient = +1` while fewer than `expected_sols` solutions have been found, then `−1`; for upper bounds `expected_sols = 0` so every solve runs with `--unsat`. `--relaxed` relaxes DIMACS header strictness. `-q` quiet.
* A DRAT proof file is requested on every solve, but **nothing in the repo or its history checks a proof** (drat-trim appears only in `refs.bib`). Kissat's exit code 20 (UNSAT) is not distinguished from a crash, OOM kill or timeout: **any non-10 return is silently logged as UNSAT.** For our pipeline: require exit code 20 *and* a checked LRAT/DRAT certificate before a case counts as `refuted`.
* Isomorph rejection of solution lists was done outside the repo with nauty `shortg` (via `Btog6`) and automorphism groups with GAP. Solution *counts* for `z_2(23)` came from sharpSAT.

## 7. The raw result logs (`kyoto/data/*`)

**Format.** Blank-line-separated records. Header `z(a,b,m,n) >= w` or `z(a,b,m,n) < w`. Under an upper-bound header: one line `(c_1, …, c_n) (r_1, …, r_m)` per surviving pair — each one was handed to Kissat and came back UNSAT (a SAT answer would have printed a matrix code below it). Under a square lower-bound header: each pair followed by *all* its solutions as `h w <base64>` codes (counts pinned by the driver's `nonzeros` dict). Non-square lower bounds: header + one solution code (from `find_solution`, which does not log the pair). Records from before 2022-01-19 (old format, `data/2x2` only, e.g. `z(2,2,8,9) < 27` followed by `(3, 3, 3, 3, 3, 3, 3, 3, 3)`) list a **column partition only** — rows were left free then.

**Counts** (parsed from the files):

| file | upper-bound records | closed with **zero** SAT calls | records needing SAT | UNSAT instances logged | median / max per record |
|---|---|---|---|---|---|
| 2x2 | 160 | 102 (64%) | 58 | 208 | 1 / 29 |
| 3x3 | 89 | 59 (66%) | 30 | 1,057 | 10.5 / 290 |
| 4x4 | 52 | 39 (75%) | 13 | 385 | 4 / 130 |
| total | 301 | 200 | 101 | **1,650** | |

Heaviest upper-bound records (number of surviving pairs = SAT calls): `z(3,3,11,17)<97`: 290; `z(3,3,12,16)<100`: 262; `z(4,4,9,14)<89`: 130; `z(3,3,9,15)<74`: 123; `z(4,4,8,18)<100`: 117; `z(4,4,9,16)<100`: 95; `z(3,3,11,16)<93`: 68; `z(3,3,8,22)<91`: 46; `z(2,2,19,24)<101`: 29; `z(3,3,16,16)<129`: 28; `z(2,2,22,23)<111`: 24; `z(4,4,12,12)<101`: 15. Square landmarks: `z_2(22)<109`: 1 pair `(5^21,4)(5^21,4)`; `z_2(23)<116`: 0 pairs; `z_2(24)<123`: 0 pairs (pure counting!); `z_3(16)<129`: 28 pairs, all with column sums in `{9,8}` and row sums in `{10,9,8,7,6,5}` around `(8^16)`; `z_4(13)<118`: 0 pairs.

Biggest lower-bound enumerations: `z_2(22) ≥ 108`: 9 pairs, 144 instance solutions (10 non-isomorphic); `z_4(9) ≥ 61`: 8 pairs, 238 solutions; `z_2(16) ≥ 67`: 70; `z_4(11) ≥ 86`: 70; `z_2(14) ≥ 56`: 28; `z_3(9) ≥ 49`: 30 pairs, 17 solutions.

**Reproduction.** Running the repo's own `get_bipartitions` reproduces the logged survivor sets exactly for every instance I tried (`z(2,2,14,32)<93`: 10; `z(2,2,16,32)<104`: 1; `z(2,2,22,22)<109`: 1; `z(2,2,24,24)<123`: 0; `z(3,3,9,9)<50`: 7; `z(3,3,16,16)<129`: 28; …), in ≤ 0.01 s each. The partition/pair layer is cheap and deterministic; the data files are the cases, and the cases are recomputable.

**Timing.** There are **no per-case timings anywhere** in the repo, its history, the paper or the talk. What exists:

* `driver22.py`: "`(4, 3^27, 2^4) (7^9, 6^5)` is quite a hard case" — this is the `z(2,2,14,32) < 93` record (10 survivors; that pair is the sixth).
* `driver22.py`: "There is only one case for the upper bound below, but it takes quite a while to prove UNSAT" — `z(2,2,16,32) < 104`, the single pair `(4^8, 3^24) (7^8, 6^8)`.
* `driver22.py`: "# TODO automate cube-and-conquer?" just before the `z_2(22)` block ("Literally thousands of solutions" per the commit log).
* `verifygraphs23.py` (historical version, commit `33c576c`): "A 2.5-hour sharpSAT run shows that the CNF corresponding to the z(2,2, 23,23) = 115, 5^23/5^23 case has 46656 = 6^6 solutions." The final version proves uniqueness by a secondary split into 12 box-sum cases (see §9).
* Talk (`pres.tex`): "I did all of the SAT solving on a single laptop computer … Kissat, which is a single-processor solver. There was thus a 'natural' limit of z ≈ 100 to the range of the table I could complete within reasonable time."

## 8. How much each argument prunes (my measurements with the repo's code)

Column/row partitions with no pruning, after argument A only, after A+I (I fed by the data files); pairs after A+I (unordered when square); survivors after D/D' = SAT calls. Roman = Roman's bound for the instance.

| instance | Roman | col parts: none / A / A+I | row parts: none / A / A+I | pairs after A+I | after D |
|---|---|---|---|---|---|
| z(2,2,14,32)<93 | 93 | 14,143,793 / 3 / 2 | 14,143,793 / 4,027,217 / 30 | 60 | **10** |
| z(2,2,16,32)<104 | 104 | 60,462,610 / 1 / 1 | 60,462,610 / 8,376,111 / 22 | 22 | **1** |
| z(2,2,19,24)<101 | 102 | 53,561,147 / 62 / 39 | 53,561,147 / 88,542 / 11 | 429 | **29** |
| z(2,2,22,22)<109 | 112 | 128,812,497 / 418 / 3 | (same) | 6 | **1** |
| z(2,2,24,24)<123 | 127 | 632,980,205 / 1,864 / 3 | (same) | 6 | **0** |
| z(3,3,9,15)<74 | 76 | 28,417 / 25 / 8 | 28,417 / 4,220 / 27 | 216 | **123** |
| z(3,3,11,17)<97 | 101 | 390,997 / 92 / 16 | 390,997 / 22,893 / 31 | 496 | **290** |
| z(3,3,12,16)<100 | 104 | 545,119 / 198 / 65 | 545,119 / 11,125 / 5 | 325 | **262** |
| z(3,3,16,16)<129 | 136 | 8,902,311 / 3,244 / 9 | (same) | 45 | **28** |
| z(4,4,8,18)<100 | 102 | 10,783 / 8 / 3 | 10,783 / 5,642 / 69 | 207 | **117** |
| z(4,4,9,14)<89 | 92 | 5,609 / 37 / 10 | 5,609 / 1,345 / 18 | 180 | **130** |
| z(4,4,12,12)<101 | 107 | 15,950 / 340 / 6 | (same) | 21 | **15** |
| z(4,4,13,13)<118 | 123 | 50,480 / 279 / 0 | (same) | 0 | **0** |

What feeds argument I matters enormously (survivors after D, same instances):

| instance | I with data-file (SAT-proved) bounds | I with Roman's bound only | no argument I |
|---|---|---|---|
| z(2,2,16,32)<104 | 1 | 2 | 21 |
| z(2,2,19,24)<101 | 29 | 600 | 93,288 |
| z(2,2,22,22)<109 | 1 | 16 | 28,022 |
| z(2,2,24,24)<123 | 0 | 66 | 572,996 |
| z(3,3,11,17)<97 | 290 | 4,927 | 222,246 |
| z(3,3,16,16)<129 | 28 | 45,044 | 2,456,268 |
| z(4,4,12,12)<101 | 15 | 1,540 | 23,546 |
| z(4,4,13,13)<118 | 0 | 1,883 | 15,062 |

**Inference:** the decisive prune in Tan's campaign is not A or D but argument I *fed with exact values of smaller instances*, which the campaign itself had just proved. The partition pair method is really an induction on `(m, n)` where every step consumes the previous steps' theorems. A Lean-gated version needs the same: a database of verified bound theorems `∀ A : Mat m j, ¬HasKst → weight A ≤ Z_j` that a prune for `(m, n)` may cite as hypotheses.

## 9. Beyond profiles: the secondary split for `z_2(23)` (`verifygraphs23.py`)

For `z_2(23) = 115` the only surviving pair was `(5^23)(5^23)`; the CNF has `6^6 = 46656` solutions even with lex constraints. Tan notes that "the first five rows and columns are completely forced by the existing constraints. These generate a 4×4 grid of 4×4 boxes; we split on the sums of these boxes, the whole grid of which may be freely permuted boxwise and then lex-sorted normally *while preserving the box sums*. A combinatorial argument shows that the non-isomorphic sum configurations are in bijection with disjoint unions of even cycles (including the 2-cycle) optionally minus an edge, 12 in all." Each of the 12 cases adds 16 exact-cardinality constraints on 4×4 blocks (`add_card_constraint_sinz(bitfield[5+4i:9+4i, 5+4j:9+4j].flatten(), bs[i,j])`), with expected solution counts `[0,0,0,0,16,48,0,0,192,0,6,48]` (310 total, all isomorphic). This is a hand-made second cube level (block sums) with a hand-made symmetry argument; it is the prototype of "further splitting into instances with partially assigned matrices" that the talk lists as future work.

## 10. `decode_array` / `encode_array`

"A (0,1)-matrix is encoded by flattening it so that rows remain contiguous, padding the result on the right to a multiple of 8 bits with zeros, interpreting each byte in little-endian order and encoding the final byte sequence using Base64. The height and width are prepended, separated by spaces."

```python
def encode_array(A):
    Af = A.flatten()
    b = np.pad(Af, (0,-len(Af)%8)).reshape(-1,8) @ 2**np.arange(8)      # bit i of each byte = element i
    return f"{A.shape[0]} {A.shape[1]} {b64encode(bytes(list(b))).decode()}"
def decode_array(ln):
    hs, ws, dat = ln.split(); h, w = int(hs), int(ws)
    A = np.array([[(b&(1<<i))>>i for i in range(8)] for b in b64decode(dat)])
    return A.flatten()[:h*w].reshape(h,w)
```
Checked: `2 2 Bw==` → `[[1,1],[1,0]]`; `3 3 qwE=` → `[[1,1,0],[1,0,1],[0,1,1]]`; `4 4 PpU=` → the 9-one `K_{2,2}`-free 4×4; `3 3 /wA=` → `J_3` minus one entry (`z_3(3) = 8`). Round-trips exactly. Every matrix in the paper's Section 5 tables and every witness in `data/` is in this format, so the lower-bound witnesses are directly machine-checkable.

---

## 11. Extracted lemmas (with hypotheses and Lean notes)

Notation: `A : Mat m n` is `K_{s,t}`-free (`s = a` rows, `t = b` columns); `c_j = colSum A j`, `r_i = rowSum A i`, `w = weight A`. ZarPrune is Mathlib-free: sums are `sumFin`, submatrices are `Incr` tuples, there is no `Finset`, no `choose`, and no permutation library.

**L1. Argument A (column form; the KST count).** *Hypotheses:* none beyond `¬HasKst`. *Statement:* `Σ_{j<n} C(c_j, s) ≤ (t−1)·C(m, s)`. *Proof:* count pairs (S, j), S an s-subset of rows all-one in column j; each S sits in at most t−1 columns or `HasKst`. *Prune:* `kill pf := decide (Σ_j C(pf.col j, s) > (t−1)·C(m,s))`. Tan applies it prefix-wise inside the generator, which is the same lemma applied to the m×j minor of the j heaviest columns (subsumed by L1 on the full profile only if one also has L3-style restriction; as a *profile* prune the full-vector version is what matters). *Lean:* this is the README's stated "next target". Ingredients: (i) `choose` on `Nat` with Pascal's rule; (ii) the identity `e_s(x_1..x_m) = C(Σ x_i, s)` for 0/1 `x` where `e_s` is defined recursively (`e_s(x·xs) = e_s(xs) + x·e_{s−1}(xs)`) — a clean induction; (iii) double counting: `Σ_j e_s(col j) = Σ_{S incr s-tuple} #{j : S all-one in col j}` — needs an enumerator of increasing tuples as a `sumFin`-expressible sum, or a recursive definition that avoids enumerating them; (iv) "a 0/1 vector with ≥ t ones has an `Incr` t-tuple of ones" (extract witness). **Moderate-to-hard; for `s = 2` it collapses to `Σ_j c_j(c_j−1) ≤ (t−1)·m(m−1)` via `(Σx)^2 = Σx + 2Σ_{i<i'} x_i x_i'`, which is an algebraic identity plus Fubini (`sumFin_swap`) — easy-moderate.** Once (ii)–(iv) exist they serve L2, L3, L4 as well.

**L1'. Argument A' (row form).** `Σ_{i<m} C(r_i, t) ≤ (s−1)·C(n, t)`. Same proof transposed. Lean: same infrastructure; the transposed `HasKst` witness needs swapping the two index tuples — trivial.

**L2. Argument D (Guy) and its pessimal-profile test.** *Hypotheses:* `¬HasKst`; row `i` with `r_i` ones in column set `J_i`. *Statement (per row):* `Σ_{j∈J_i} C(c_j − 1, s−1) ≤ (t−1)·C(m−1, s−1)`. *Proof:* delete row `i`; the `(m−1)×|J_i|` matrix on `J_i` has column sums `c_j−1` and is `K_{s−1,t}`-free (else add row `i`); apply L1. *Profile test (Tan):* with `r = max_i r_i` and `c_(1) ≤ … ≤ c_(r)` the `r` smallest column sums, `Σ_{k≤r} C(c_(k)−1, s−1) > (t−1)·C(m−1, s−1)` ⇒ kill. Justified because any `r`-subset of columns has sum ≥ the `r` smallest. Also the transpose D'. *Lean:* L1 for `(s−1, t)` on a submatrix (needs restriction along an `Incr` column tuple and an `Incr` row tuple omitting `i`, plus "HasKst of restriction ⇒ HasKst of A" by composing `Incr` maps — easy), plus a rearrangement lemma "sum of `f` over any `r`-subset ≥ sum of the `r` smallest values of `f`" on `Fin n → Nat` — fiddly without a sorting library. A sort-free weakening that is much easier to state and prove: pick a threshold `θ` and use `Σ_{j∈J_i} C(c_j−1,s−1) ≥ Σ_{j∈J_i} min(C(c_j−1,s−1), θ)`… ; or simply have `kill` compute the r smallest by a verified insertion sort on a `List Nat` and prove `List`-level lemmas. **Moderate-hard; depends on L1.**

**L3. Argument I / E (inclusion).** *Hypotheses:* a previously verified bound `Z_j` for the `m × j` instance: `H_j : ∀ B : Mat m j, ¬HasKst (s,t) B → weight B ≤ Z_j` (from Roman's bound, Culik's corollary, or an earlier closed case of the pipeline). *Statement:* for every `Incr` tuple `C : Fin j → Fin n`, `Σ_{k<j} c_{C k} ≤ Z_j`; pessimal form: the sum of the `j` largest column sums `≤ Z_j`. *Lean:* restriction along `C` (easy), `weight (A ∘ C) = Σ_k c_{C k}` (Fubini), `HasKst (A∘C) → HasKst A` (compose `Incr`), and again the "j largest" rearrangement (or a sort-free version: for any threshold `θ`, the columns with `c_j ≥ θ` form a subset; if there are `≥ j` of them pick any `j` — extractable by the same witness lemma as L1(iv)). **Easy-moderate given a bound theorem to cite; the real cost is the dependency chain (Section 8): a prune for `(m,n)` carries `H_j` for `j < n` as hypotheses, so the harness must store and pass verified theorems, exactly as `zbounds` reads `data/`.** With only Roman's bound for `H_j` (no chain) the prune is ~10–1000× weaker.

**L4. Roman's bound (paper Theorem 2.2).** For integer `p ≥ s−1`: `z_{s,t}(m,n) ≤ ⌊ (t−1)/C(p,s−1) · C(m,s) + (p+1)(s−1)/s · n ⌋`. *Inference on the proof:* it is L1 plus the convexity chord `C(c,s) ≥ C(p,s) + C(p,s−1)(c−p)` for all integers `c ≥ 0`, summed over columns. As a *profile* prune it is implied by L1 (L1 is checked exactly on the given sums), so its value is as a closed-form `H_j` for L3 and as the instance-level target; `utilities.romanbound` minimises over `p`. *Lean:* L1 + an integer inequality by induction on `c`; moderate. Culik's equality corollary (`z = (s−1)n + (t−1)C(m,s)` when `n ≥ (t−1)C(m,s)`) is a lower-bound construction, not a prune.

**L5. Row/column cap and total consistency.** `c_j ≤ m`, `r_i ≤ n`, `Σ c_j = Σ r_i = w` — already proved in ZarPrune (`rowCap`, `colCap`, `mismatch`, `deficit`). Tan's generator enforces these implicitly (parts in `[0,m]`, both partitions of the same `k`).

**L6. Argument B (Guy) — not a prune.** `C(m−1,k)+C(n+1,k) < C(m,k)+C(n,k)` for `m−n > 1`, i.e. `Σ C(c_j,s)` is minimised by level column sums; gives the equality/maximality criterion. Useful for *ordering* cases (most level = most likely SAT) and for reasoning about tightness, not for killing.

**Not prunes (adding moves, need a witness outside ZarPrune):** (i) transposition when `a=b, m=n` (unordered pairs); (ii) within-instance lex sorting of equal-sum rows/columns (Theorem 3.2); (iii) `sort_icdv` reordering; (iv) the box-sum symmetry argument of §9. Each removes cases or assignments that *do* contain valid matrices, justified by a permutation. ZarPrune's `notDescending_unsound` demonstrates why these must not be smuggled in as `Prune`s.

---

## 12. Relevance to our design

1. **The case space is Tan's, exactly.** Cases = (unordered column partition, unordered row partition) of `w` with parts in `[0,m]` / `[0,n]`; the CNF per case = minor clauses + exact Sinz cardinalities on every row and column + lex comparators inside equal-sum runs. ZarPrune's `Profile` (ordered sum vectors) refines these cases; a `Prune` on profiles restricts to a prune on partitions by evaluating on the sorted vector **provided `kill` is invariant under permuting `pf.row` and `pf.col`** (or the harness only ever calls it on sorted vectors and the soundness proof quantifies over all `A`, which it does). The subtle part is `cover` in `upper_bound_of_cover`: survivors are *sorted* profiles, but `profileOf A` is generally unsorted, so either the survivor list is closed under permutations (blow-up) or we need the lemma "`Valid P A ↔ Valid P (A permuted)`" and restate cover modulo permutation. That permutation-invariance lemma for `HasKst` with `Incr` tuples is unavoidable and is not in the library yet.
2. **Baseline to beat = A + A' + D + D' + I + I' with exact smaller values.** Any evolved prune should be scored against Tan's survivors, not against the raw partition count (raw: millions; Tan: 0–290). Section 8's tables are the yardstick. `get_bipartitions` runs in milliseconds, so the harness can compute the baseline survivor set for every instance on the fly.
3. **Argument I forces a theorem database.** The pipeline must persist verified bounds (Lean theorems, or at least `(m, j, Z_j)` triples with their certificates) and let a prune for `(m,n)` cite them. Roman/Culik give the closed-form seeds. This is the same recursion Tan's `zbounds` performs against `data/`.
4. **Solver protocol fixes.** Copy the flag choice (`--unsat` for refutations) but not the return-code handling: demand exit 20, keep the DRAT/LRAT, check it (drat-trim / cake_lpr), and only then discharge `refuted`. Blocking-clause enumeration + final DRAT is a neat way to certify "all solutions" for lower-bound work.
5. **Encoding.** Reuse Tan's exact Sinz encoding (verified above) or our `encodings_zar.py` unary counters; either way the variable layout `i*n+j+1` matches `Mat`/`HasKst`. An encoding-correctness theorem (CNF-UNSAT ⇒ no matrix in the case) is the other seam of `upper_bound_of_cover`; Tan's clause set is small enough to formalise.
6. **Second-level cubes exist in Tan's own work** (`extend_solution`, `verifygraphs23.py`): when a single surviving pair is the bottleneck (e.g. `z_2(16,32)<104`, `z_2(23)`), the productive move is splitting inside the case (box sums / partial assignments), which is a *cube*, not a prune. The `Prune` gate is for killing; a separate `cover`-certificate path is needed for splitting.
7. **Data reuse.** `data/*` gives 1,650 (case, UNSAT) pairs and ~300 proved bounds: a ready-made training/evaluation set for "which cases survive" and for regression-testing evolved prunes (an evolved prune that kills a case in which `data/` records a *solution* is unsound — instant negative test, no Lean needed).

## 13. Difficulty signals visible in this source

* **Number of surviving cases** after the baseline prunes (0 → no SAT at all; median 1–10; max 290). Two thirds of Tan's upper-bound records needed no solver.
* **Slack between the target and Roman's bound** (`Roman − w`). **Inference:** both cases Tan flagged as hard have slack 0 (`z(2,2,14,32)<93`, Roman 93; `z(2,2,16,32)<104`, Roman 104): the counting is tight, few cases survive, but each survivor is a near-extremal, near-level instance that the solver finds hard. Instances with many survivors (`z(3,3,11,17)<97`, slack 4; 290 cases) were completed on a laptop without comment, i.e. many easy cases. Per-case slack `(t−1)C(m,s) − Σ_j C(c_j,s)` (and the row dual) is the natural per-case version.
* **Levelness / symmetry of the profile pair.** All-equal profiles (`(5^23)(5^23)`, `(8^16)(8^16)`, `(9^13)(9^13)`) have huge automorphism groups; lex breaking leaves `6^6` solutions for `z_2(23)`. Hard for enumeration, and the `--unsat` instance one unit above such a profile is where Tan hit his limits.
* **Raw size**: `C(m,s)·C(n,t)` minor clauses and `Σ k(n−k)` cardinality aux; Tan's practical wall was `z ≈ 100` on one laptop with Kissat (2022).
* **Position in the table**: near the Culik/Roman equality region (edges of the tables) everything closes by counting; deep interior with `m ≈ n` is where "the bound of theorem 2.2 appears to be less sharp" and SAT does the work.

## 14. Open questions

1. No timings survive; to calibrate a difficulty model we must re-run Tan's 1,650 cases with a modern Kissat/CaDiCaL and log wall-clock per case (cheap: all cases are recomputable from `get_bipartitions`).
2. Are the DRAT proofs Tan generated still needed? None were kept; our pipeline should produce and check LRAT for every survivor and archive them with the case index.
3. How to phrase `cover` for unordered partitions in ZarPrune: permutation-closure of survivors vs. a proved invariance lemma for `Valid`. (Blocker for using Tan's case space in the closure theorem.)
4. What is the smallest extra counting argument that kills, e.g., the 28 `z_3(16)<129` survivors or the 290 of `z(3,3,11,17)<97`? Candidates: D applied to *all* rows with the true column incidence (needs more than a profile), joint row/column pair counts (Davies–Gill–Horsley-style refinements), or L3 with two-sided minors `z(i, j)`. This is the evolutionary target.
5. Does argument I with *non-prefix* column subsets (any `j` columns, not the heaviest) add anything for profiles? On sorted profiles the prefix is pessimal, so no; but for evolved prunes that look at more than sums it might.
6. Tan's `nonzeros` solution counts (e.g. 6^6 for `z_2(23)`, 238 for `z_4(9)`) are the only "how many models" data; sharpSAT-style counts could be another difficulty proxy for near-extremal cases.

## 15. Sources

* Repository: https://github.com/Parcly-Taxel/Kyoto (cloned in full; HEAD `3acbc61`, 2022-04-20).
* Paper: J. Tan, *An attack on Zarankiewicz's problem through SAT solving*, arXiv:2203.02283v2 (local copy `docs/papers/tan2022_sat_zarankiewicz.pdf`); talk `paper/pres.tex` in the repo.
* Origin gist (3×3 minors, square, CaDiCaL/Kissat): https://gist.github.com/Parcly-Taxel/705747d9b62b29967647eb680ca4cdd4 ; OEIS A350237, A001198.
* Wynn, *A comparison of encodings for cardinality constraints in a SAT solver*, arXiv:1810.12975 (the Sinz variant used).
* Guy 1969 (arguments A, B, D, I), Roman 1975 (Theorem 2.2), Culik 1956 — as cited by Tan.

Scratch scripts used for the measurements above: `/private/tmp/claude-501/-Users-jaybhan-Downloads-openevolve/88cfd163-c49a-4b7d-ba6f-24d477547106/scratchpad/lit/{parse_data.py, verify_enc.py, count_cases.py, prune_stages.py, roman_only.py}` (run with `PYTHONPATH=Kyoto`).

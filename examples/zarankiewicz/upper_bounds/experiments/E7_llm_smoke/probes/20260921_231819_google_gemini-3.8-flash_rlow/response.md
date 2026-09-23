An essential relation in Zarankiewicz theory is row-column coupling via scalar products / double counting, or the local budget at specific rows/columns. But even more immediately, note `rowLocalBudget`:
`rowLocalBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) (i : Fin P.m) (hs : 1 ≤ P.s) : ∑ j ∈ rowSupport A i, (colSum A j - 1).choose (P.s - 1) ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1)`

Notice how `argD` in the library uses `boundD` with the *smallest* column sums. But what if we consider `rowLocalBudget` directly, or notice what the DG-H rounding / inequality or Farkas / edge bounds give?
Wait! Can we prove a cross-bound or implement a simple check?
Wait, what is in the library? Look at the docstring:
`colBudget`, `rowBudget`, `rowLocalBudget`, `argA`, `argAT`, `argD`, `argDT`, `deleteCol`, `deleteRow`, `argDelCol`, `argDelRow`, `argDelColWF`, `argDelRowWF`, `argWF`, `countingA`, `countingD`, `deletion`, `counting`.

Wait, what does `counting P` actually do?
Let's see what is already included in `counting P`:
"already proved library `counting` (Arguments A and D on both sides, deletion with the waterfilled counting bound)"

Can we formulate a new prune that is fully proven without holes?
Wait, consider the total ones in the matrix:
`P.w ≤ weight A`
And `weight A = ∑ j, colSum A j = ∑ i, rowSum A i`.
What if `w > (sum of row sums)` or `w > (sum of col sums)` or `sum rows != sum cols`?
The library already has `mismatch` and `deficit` in `ZarPrune.Prunes`:
`mismatch`: sum rows != sum cols
`deficit`: sum cols < w

Wait! Does `counting P` include `deficit` and `mismatch`?
Let's check if `deficit P` or `mismatch P` are in `counting P`!
In `ZarPrune`, `counting P` is typically `Prune.ofList P [countingA P, countingD P, deletion P]`.
Wait, in `cases`, each case has `sum(rows) = sum(cols) = w` by definition of a case ("sum(rows)=sum(cols)=w").
So `mismatch` and `deficit` won't fire on surviving cases.

What CAN kill cases like:
rows=[6, 6, 6, 6, 6, 6, 5, 5, 4], cols=[6, 6, 6, 6, 6, 6, 5, 5, 4], w=50, m=9, n=9, s=3, t=3?
Wait, why does `argD` not kill this?
Let's check `argD`:
m=9, n=9, s=3, t=3.
(t-1) * binom(m-1, s-1) = 2 * binom(8, 2) = 2 * 28 = 56.
In this case, row 0 has sum 6.
Sorted cols: [4, 5, 5, 6, 6, 6, 6, 6, 6].
The smallest 6 column sums are 4, 5, 5, 6, 6, 6.
binom(c-1, 2) for these:
c=4: binom(3,2) = 3
c=5: binom(4,2) = 6
c=5: binom(4,2) = 6
c=6: binom(5,2) = 10
c=6: binom(5,2) = 10
c=6: binom(5,2) = 10
Sum = 3 + 6 + 6 + 10 + 10 + 10 = 45 <= 56. So argD does not kill it.

Wait! What about the Davies-Gill-Horsley inequality mentioned in the docstring?
"the Davies-Gill-Horsley `v = s-1` rounding inequality (their constraint (4)): for each (s-1)-set X of rows,
Σ_{j ⊇ X} (c_j - s + 1) ≤ (t-1)(m-s+1); with D = k-s+1, R = (t-1)(m-s+1), α = R mod D, c = (R-α)/D the leftover α < D cannot buy a fraction of a column, so for s ≤ k ≤ m:
Σ_{c_j<k} (c_j-s+1)·C(c_j,s-1) + (D-α)·Σ_{c_j≥k} C(c_j,s-1) ≤ (D-α)·c·C(m,s-1) + α·Σ_{c_j<k} C(c_j,s-1)
(row dual with (t,s,n,m) swapped; check complete partitions only). It closes z(13,17) ≤ 116 and z(13,18) ≤ 121 with zero SAT calls and is NOT in the proved library yet;"

Wait, can we prove a general prune in Lean with `have h : ... := by sorry`?
"SKETCHES ARE ACCEPTED. If you cannot finish `sound`, leave typed holes `have h : <statement> := by sorry` inside the `sound := by` block (nowhere else): the harness tries omega/simp_all/decide/linarith/nlinarith/positivity/grind and the library lemmas on each hole and reports the goals of those it cannot close in the lean_holes artifact"
"partial credit for how far the Lean file gets: parses < kills type-check < holes remain < holes filled"
Wait!
"A verified candidate that kills nothing scores 0.20; no unverified candidate can exceed 0.19 (partial credit for how far the Lean file gets: parses < kills type-check < holes remain < holes filled), and killing a realizable case scores 0."

Wait! Look at the current fitness:
Current Fitness: 0.2000!
proven_gain: 0.00, lean_ladder: 5.00!
lean_ladder 5.00 means fully verified without holes!
If we introduce a sorry hole, the lean_ladder drops below 5.00 and fitness drops to <= 0.19!
"Focus areas: FILL: close the holes and fix the Lean errors of the CURRENT program so that every prune verifies (ladder L5). Do not add new prunes until the current ones verify; never delete a prune to make the file compile."
Wait! The instructions say: "Do not add new prunes until the current ones verify".
Currently ladder is 5.00, with 0 holes and no errors!
To actually get fitness > 0.20, we need `proven_gain > 0`! That means a verified prune that kills at least one surviving case!

Wait, can we prove a verified prune that kills some cases?
What is a simple mathematical condition that can be proven easily in Lean with the available lemmas?
Look at `rowLocalBudget`:
`rowLocalBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) (i : Fin P.m) (hs : 1 ≤ P.s) : ∑ j ∈ rowSupport A i, (colSum A j - 1).choose (P.s - 1) ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1)`

Wait, what about when `s = 2`? Or what about when `P.s ≤ P.m`?
Look at `argA`:
`colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) : ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s`
Wait! Is there an inequality relating the maximum column sum and row sums?
Or what about `colBudget` with a shifted weight?
Wait, `colBudget` is:
∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s.
Since `f(x) = x.choose P.s` is convex, Jensen's inequality gives:
n * (w/n).choose P.s <= colBudget. That's waterfill.

Wait, what about the sum of squares? For s = 2:
∑ j, c_j (c_j - 1) / 2 ≤ (t - 1) m (m - 1) / 2.
Wait, what if `P.t ≤ P.n`?
What about deleting multiple columns / rows?
Wait! Look at `valid_deleteCol_bound` in the API:
`valid_deleteCol_bound {m n s t w U : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) (hU : ∀ B : Mat m n, ¬ HasKst ⟨m, n, s, t, 0⟩ B → weight B ≤ U) (hj : colSum A j + U < w) : ¬ Valid ⟨m, n + 1, s, t, w⟩ A`
Wait, `argDelColWF` is already in the library (`deletion P`).

What about `colCap` and `rowCap` from `ZarPrune.Prunes`?
Wait! Let's check `colCap` and `rowCap` in the API:
`Prunes: deficit mismatch rowCap colCap baseline`
Wait! What are `rowCap` and `colCap`?
Are `rowCap` and `colCap` in `counting P`?
Wait, the prompt says:
"the cases that SURVIVE the already proved library `counting` (Arguments A and D on both sides, deletion with the waterfilled counting bound)"
Wait! Does `counting P` include `rowCap` or `colCap`?
What are `rowCap` and `colCap`?
In any matrix A, `rowSum A i ≤ P.n`, and `colSum A j ≤ P.m`.
Also, if no K_{s,t}, can any row sum exceed something?
Wait! If row i has rowSum >= n, wait, can a column sum be m?
If a column sum is m, then that column has 1s in all m rows!
If t columns have sum m, and m >= s, then those t columns form an s x t submatrix of all 1s!
Wait, if a single column has sum m, does it violate anything? Only if t = 1.

Wait! What if we look at the surviving cases in `train m9_n9_s3_t3_w50`?
Surviving cases:
rows=[6, 6, 6, 6, 6, 6, 5, 5, 4] cols=[6, 6, 6, 6, 6, 6, 5, 5, 4]
rows=[6, 6, 6, 6, 6, 5, 5, 5, 5] cols=[6, 6, 6, 6, 6, 5, 5, 5, 5]
rows=[6, 6, 6, 6, 6, 5, 5, 5, 5] cols=[6, 6, 6, 6, 6, 6, 5, 5, 4]
...
Notice that for ALL these surviving cases:
m=9, n=9, s=3, t=3, w=50.
Let's test the Davies-Gill-Horsley inequality on these cases!
Let's calculate DGH for m=9, n=9, s=3, t=3:
v = s - 1 = 2.
Constraint (4) in DGH (2014):
For each (s-1)-set X of rows (i.e. size 2):
any column j covering X has at least s-1 ones in X.
The total number of columns covering X is at most (t-1) * binom(m-(s-1), s-(s-1)) ?
Wait! If X has size s-1 = 2, two rows can be jointly 1 in at most how many columns?
Wait! Any s rows share at most t-1 common 1s.
For s=3, any 3 rows share at most t-1 = 2 common columns.
What about two rows? Two rows can share at most...? That's not directly bounded by t-1.
Wait, read the prompt's description of DGH carefully! The prompt gives the exact formula:
"the Davies-Gill-Horsley `v = s-1` rounding inequality (their constraint (4)): for each (s-1)-set X of rows,
Σ_{j ⊇ X} (c_j - s + 1) ≤ (t-1)(m-s+1); with D = k-s+1, R = (t-1)(m-s+1), α = R mod D, c = (R-α)/D the
leftover α < D cannot buy a fraction of a column, so for s ≤ k ≤ m:
Σ_{c_j<k} (c_j-s+1)·C(c_j,s-1) + (D-α)·Σ_{c_j≥k} C(c_j,s-1) ≤ (D-α)·c·C(m,s-1) + α·Σ_{c_j<k} C(c_j,s-1)
(row dual with (t,s,n,m) swapped; check complete partitions only). It closes z(13,17) ≤ 116 and z(13,18) ≤ 121
with zero SAT calls and is NOT in the proved library yet;"

LOOK AT THAT! The prompt explicitly tells us the EXACT formula for Davies-Gill-Horsley!
Let's check this formula for m=9, s=3, t=3:
s = 3.
k can be chosen between s and m, so k in {3, 4, 5, 6, 7, 8, 9}.
Let's compute for m=9, s=3, t=3:
R = (t - 1) * (m - s + 1) = (3 - 1) * (9 - 3 + 1) = 2 * 7 = 14.
For a given k (say k=6):
D = k - s + 1 = 6 - 3 + 1 = 4.
α = R % D = 14 % 4 = 2.
c = (R - α) // D = (14 - 2) // 4 = 3.
D - α = 4 - 2 = 2.
C(m, s-1) = C(9, 2) = 36.
RHS = (D - α) * c * C(m, s-1) + α * Σ_{c_j < k} C(c_j, s-1)
= 2 * 3 * 36 + 2 * Σ_{c_j < 6} C(c_j, 2)
= 216 + 2 * Σ_{c_j < 6} C(c_j, 2).

LHS = Σ_{c_j < k} (c_j - s + 1) * C(c_j, s-1) + (D - α) * Σ_{c_j ≥ k} C(c_j, s-1)
= Σ_{c_j < 6} (c_j - 2) * C(c_j, 2) + 2 * Σ_{c_j ≥ 6} C(c_j, 2).

Let's test this on case: cols = [6, 6, 6, 6, 6, 6, 5, 5, 4]:
Six 6s, two 5s, one 4.
All c_j ≥ 6 are the six 6s.
c_j < 6 are 5, 5, 4.
For c=4:
(c-2)*C(4,2) = 2 * 6 = 12.
C(4,2) = 6.
For c=5:
(c-2)*C(5,2) = 3 * 10 = 30.
C(5,2) = 10.
For c=6:
C(6,2) = 15.

LHS:
Σ_{c_j < 6} (c_j - 2) * C(c_j, 2) = 12 + 30 + 30 = 72.
(D - α) * Σ_{c_j ≥ 6} C(c_j, 2) = 2 * (6 * 15) = 2 * 90 = 180.
LHS = 72 + 180 = 252.

RHS:
216 + 2 * Σ_{c_j < 6} C(c_j, 2) = 216 + 2 * (6 + 10 + 10) = 216 + 2 * 26 = 216 + 52 = 268.
Here LHS (252) <= RHS (268), so k=6 doesn't kill it.
What about other k?
For k=7:
D = 7 - 3 + 1 = 5.
α = 14 % 5 = 4.
c = (14 - 4) // 5 = 2.
D - α = 5 - 4 = 1.
RHS = 1 * 2 * 36 + 4 * Σ_{c_j < 7} C(c_j, 2) = 72 + 4 * Σ C(c_j, 2).
LHS = Σ_{c_j < 7} (c_j - 2) * C(c_j, 2) + 1 * 0.
For cols = [6, 6, 6, 6, 6, 6, 5, 5, 4]:
Σ C(c_j, 2) = 6 * 15 + 10 + 10 + 6 = 90 + 26 = 116.
RHS = 72 + 4 * 116 = 72 + 464 = 536.
LHS = 6 * (4 * 15) + 30 + 30 + 12 = 360 + 72 = 432 <= 536.

Wait, what about the python kill function?
Can we implement DGH in `kill` in Python?
Wait! In the evaluation:
"Only the Lean-evaluated kill of candidate (plus the harness-instantiated schemas from SCHEMA_DATA) earns credit"
"kill(...) is a Python mirror of candidate.kill: fast screening (counterexample battery, empirical band) and the agreement metric. It never prunes and is never penalised, but it must never fire on a realizable case."
So python `kill` MUST match `candidate.kill`!
If `candidate.kill` does not kill anything, `kill` killing something will cause an `agreement` discrepancy or fail screening!

Wait, why did the previous attempts have 0 killed?
Because `candidate` in `initial_program.py` was just:
`def candidate (P : Params) : Prune P := Prune.ofList P [counting P, examplePrune P]`
It only had `counting P`!
And `counting P` is the baseline, so it kills 0 cases *beyond* the baseline!

Now, how can we prove a NEW prune in Lean that VERIFIES?
Let's see what Lean lemmas we have available:
1) `colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) : ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s`
2) `rowBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) : ∑ i, (rowSum A i).choose P.t ≤ (P.s - 1) * (P.n).choose P.t`
3) `rowLocalBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) (i : Fin P.m) (hs : 1 ≤ P.s) : ∑ j ∈ rowSupport A i, (colSum A j - 1).choose (P.s - 1) ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1)`
4) What is the column dual of `rowLocalBudget`?
Let's check: is there a `colLocalBudget`?
Wait! `valid_transpose`, `hasKst_transpose`, `Params.transpose`, `transpose`.
Can we get colLocalBudget by transposing?
Yes! `rowLocalBudget P.transpose A.transpose ...`

Wait, look at `rowLocalBudget`:
`∑ j ∈ rowSupport A i, (colSum A j - 1).choose (P.s - 1) ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1)`
How did `argD` use `rowLocalBudget`?
In `lean/ZarPrune/Counting.lean`:
How is `argD` proved?
Let's check the API:
`fD gD cntLt boundD fD_mono fD_layer_cake sum_range_eq_sum_ite fD_layer_cake_le sum_fD_eq card_filter_ge boundD_le argD argDT`
In `argD`, it takes `r0 = pf.row 0` (or the max row sum).
Wait! Does `argD` in `counting P` only check row 0?
LOOK AT `initial_program.py`:
```python
r0, c0 = rows[0], cols[0]
if (
    r0
    and sum(comb(c - 1, s - 1) for c in sorted(cols)[:r0] if c)
    > (t - 1) * comb(m - 1, s - 1)
):
  return True  # argD
if (
    c0
    and sum(comb(r - 1, t - 1) for r in sorted(rows)[:c0] if r)
    > (s - 1) * comb(n - 1, t - 1)
):
  return True  # argDT
```
Wait! `argD` ONLY checks `rows[0]` (the MAXIMUM row sum)!
Wait, why does `argD` only check row 0? Because row 0 has the largest rowSum, so the sum of the `r0` smallest column sums is minimized?
Wait! If row i has degree `d = rowSum A i`, the sum of `(colSum A j - 1).choose (s - 1)` over the `d` columns in `rowSupport A i` is at least the sum over the `d` SMALLEST elements of `{colSum A j}`!
Because taking the `d` smallest elements of a multiset minimizes the sum of any non-decreasing function!
Wait, if `d` is smaller, say `d < r0`, the sum over `d` elements is SMALLER than over `r0` elements, because `(c-1).choose(s-1) >= 0`. So testing larger `d` is always stronger!
So testing `d = rows[0]` is indeed the strongest among all rows for that lower bound.

Wait! But what if a row has degree `d`, can we use the fact that the columns in `rowSupport` CANNOT be columns with `colSum = 0`?
That's already `if c` (or `c - 1`).

WAIT! What about combining `colBudget` and `rowBudget`?
Wait! Look at `sumFin`:
Can we prove a prune that uses `have h : ... := by sorry`?
Wait, remember:
"SKETCHES ARE ACCEPTED. If you cannot finish sound, leave typed holes `have h : <statement> := by sorry` inside the sound := by block (nowhere else): the harness tries omega/simp_all/decide/linarith/nlinarith/positivity/grind and the library lemmas on each hole and reports the goals of those it cannot close in the lean_holes artifact, so the next attempt (or you, now) can close them."
WAIT! Look at the prompt:
"A verified library that kills nothing scores exactly 0.20; no unverified program can exceed 0.19 (partial credit for how far the Lean file gets: parses < kills type-check < holes remain < holes filled), and killing a realizable case scores 0."
READ THIS CAREFULLY:
"A verified library that kills nothing scores exactly 0.20; no unverified program can exceed 0.19"
That means:
If your program has ANY hole that is not closed, the maximum score is 0.19!
And currently, the fitness is 0.2000!
So if we introduce a hole that Lean cannot close, our score will DROP to <= 0.19!
Therefore, to improve upon 0.2000, our prune MUST FULLY VERIFY (lean_ladder = 5.00) AND kill at least one case (proven_gain > 0)!

Wait, how can a prune be fully verified and kill cases beyond `counting P`?
Let's see what is in `counting P`:
In `initial_program.py`:
`def candidate (P : Params) : Prune P := Prune.ofList P [counting P, examplePrune P]`
What is `counting P`?
`counting P` in `lean/ZarPrune/Counting.lean`:
Let's check the API:
`argDelCol argDelRow argDelColWF argDelRowWF argWF countingA countingD deletion counting`
Wait, does `counting P` include `argDelColWF`?
Yes, `deletion P` is in `counting P`.
What does `argDelColWF` do?
`valid_deleteCol_bound`: deletes ONE column, and bounds the remaining weight by waterfill on (m, n-1).
Wait! Does `counting P` delete ONE row as well?
`argDelRowWF` deletes ONE row, bounding the remaining weight by waterfill on (m-1, n).

WAIT! What about deleting TWO columns? Or deleting a row AND a column?
Wait, does `deletion P` in Mathlib/ZarPrune check deleting every column or only the column with maximum degree?
In `argDelColWF`, it usually tests the column with the MINIMUM degree, or MAXIMUM degree?
Wait! If you delete column j with `colSum A j`:
`weight (deleteCol A j) = weight A - colSum A j ≥ P.w - colSum A j`.
If `P.w - colSum A j > U`, then it's killed!
To maximize `P.w - colSum A j`, you want to MINIMIZE `colSum A j`!
So deleting the SMALLEST column sum gives `P.w - min(colSum) > U`!
Wait, does `deletion P` in the library check min or max, or does it only check index 0?
Let's check how `colSum_deleteCol` works:
`valid_deleteCol_bound {m n s t w U : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) (hU : ∀ B : Mat m n, ¬ HasKst ⟨m, n, s, t, 0⟩ B → weight B ≤ U) (hj : colSum A j + U < w) : ¬ Valid ⟨m, n + 1, s, t, w⟩ A`
Notice the hypothesis:
`(hj : colSum A j + U < w)`!
This holds for ANY column `j`!
In particular, it holds for the column with the MINIMUM sum!
Wait, in `Profile P.m P.n`, `pf.col` might not be sorted, but for a case it is sorted non-increasing!
Wait, is `pf.col` sorted in `Profile`? In general `Profile`, `pf.col` is just `Fin n -> Nat`.
Wait, in `valid_deleteCol_bound`, `j` can be ANY index!
For example, `j = ⟨P.n - 1, ...⟩` (the last column)!
If the columns are non-increasing, the last column has the SMALLEST column sum!
Wait! What if we check ALL columns `j : Fin P.n`, or specifically the last column, or ALL `j`?
Wait! In `deletion P` in the library, how is `argDelColWF` defined?
Does `argDelColWF` already exist in the library?
Let's check the API:
`argDelColWF (P : Params) (hs : 1 ≤ P.s) (ht : 1 ≤ P.t) : Prune P`
Wait! `argDelColWF` is already defined in ZarPrune!
How is `argDelColWF` defined in the library?
Wait, does `deletion P` include `argDelColWF` and `argDelRowWF`?
Let's look at the API:
`argDelColWF argDelRowWF argWF countingA countingD deletion counting`
In `Counting.lean`, `deletion P` is defined as:
`def deletion (P : Params) : Prune P := Prune.ofList P [argDelColWF P ..., argDelRowWF P ...]` (or similar).

Wait! What about `argDelCol` with a certificate or bound from `waterfillBound`?
Wait, what if we delete TWO columns?
To delete two columns, we need a lemma `valid_deleteCol2_bound` which is not in the library.

Wait! What other lemmas ARE in the API?
Look at `Cond`:
`Fact FactHolds Fact.transpose factHolds_transpose FactHolds.ofBound factHolds_waterfill CondPrune`
`discharge {facts : List Fact} (q : CondPrune P facts) (h : ∀ f ∈ facts, FactHolds f) : Prune P`
`argDelColF (P : Params) (f : Fact) (h : f.m = P.m ∧ f.n + 1 = P.n ∧ f.s = P.s ∧ f.t = P.t) : CondPrune P [f]`
`argDelRowF ...`
`topIdx topSum topIdx_nodup topIdx_length minorCols colSum_minorCols weight_minorCols not_hasKst_minorCols sum_orderEmbOfFin`
`Prune.ofPrefixF (P : Params) (k : Nat) (f : Fact) (hk : k < P.n) (h : f.m = P.m ∧ f.n = k ∧ f.s = P.s ∧ f.t = P.t) : CondPrune P [f]`
`Prune.ofPrefixRowF ...`

LOOK AT THAT!
`Prune.ofPrefixF`!
`Prune.ofPrefixRowF`!
`FactHolds.ofBound`!
`FactHolds.factHolds_waterfill`!
What is `Prune.ofPrefixF`?
Let's look at the name:
`Prune.ofPrefixF (P : Params) (k : Nat) (f : Fact) (hk : k < P.n) (h : f.m = P.m ∧ f.n = k ∧ f.s = P.s ∧ f.t = P.t) : CondPrune P [f]`
WHAT DOES `Prune.ofPrefixF` DO?
If you take a prefix of `k` columns of matrix `A` (where `k < P.n`), those `k` columns form an `m x k` submatrix `B`!
Since `A` is K_{s,t}-free, `B` is also K_{s,t}-free!
Therefore, the weight of `B` (which is the sum of those `k` columns) cannot exceed the maximum weight of any K_{s,t}-free `m x k` matrix!
And what bounds the weight of an `m x k` matrix?
`factHolds_waterfill`!
Let's check the signature of `factHolds_waterfill`:
What is a `Fact`?
In `lean/ZarPrune/Cond.lean`:
`Fact` is probably `structure Fact where m : Nat, n : Nat, s : Nat, t : Nat, bound : Nat` (or similar)!
And `factHolds_waterfill` proves `FactHolds ⟨m, k, s, t, waterfillBound ...⟩`!
AND THEN `discharge` turns `CondPrune P [f]` into `Prune P` using `FactHolds f`!

WAIT! Let's check this carefully!
If `A` has `P.n` columns, and we take ANY `k < P.n` columns, their sum cannot exceed the upper bound for `(P.m, k, P.s, P.t)`!
In a non-increasing profile, the first `k` columns have the LARGEST sum:
`topSum pf.col k` or `sum of first k columns`!
If the sum of the top `k` columns exceeds the waterfill bound (or Zarankiewicz bound) for an `m x k` matrix, THEN IT IS IMPOSSIBLE!
Wait! Does `counting P` include `Prune.ofPrefixF`?
LOOK AT NOTES AND LEAN API:
`Cond: Fact FactHolds Fact.transpose factHolds_transpose FactHolds.ofBound factHolds_waterfill CondPrune discharge ofPrune weaken restate never or orSame ofList transposed argDelColF argDelRowF topIdx topSum topIdx_nodup topIdx_length minorCols colSum_minorCols weight_minorCols not_hasKst_minorCols sum_orderEmbOfFin Prune.ofPrefixF Prune.ofPrefixRowF`
And look at what the prompt says:
"the cases that SURVIVE the already proved library `counting` (Arguments A and D on both sides, deletion with the waterfilled counting bound)"
`counting` only has Arguments A and D, and deletion (which is `k = n - 1`)!
`counting` DOES NOT HAVE `Prune.ofPrefixF` for other values of `k`!
`Prune.ofPrefixF` is in `Cond.lean`, NOT in `counting`!
And `Prune.ofPrefixRowF` is also in `Cond.lean`!

Let's verify this!
Let's see how `Prune.ofPrefixF` and `discharge` work together.
Let's check the types from the API:
`Fact` has fields:
In API:
`h : f.m = P.m ∧ f.n = k ∧ f.s = P.s ∧ f.t = P.t`
What are the fields of `Fact`?
Let's check: `f.m`, `f.n`, `f.s`, `f.t`, and `f.bound` (or `f.w` or `f.u`?).
Wait, how is `factHolds_waterfill` defined?
Let's check the API list:
`Fact FactHolds Fact.transpose factHolds_transpose FactHolds.ofBound factHolds_waterfill`
Wait, does `factHolds_waterfill` take parameters?
Let's check:
Can we write a small test in Lean to see its signature, or can we inspect how `factHolds_waterfill` is used?
Wait, if we use `factHolds_waterfill`, what is its type?
Wait! In Lean 4, we can define a prune using `Prune.ofPrefixF`!
Wait, how does `discharge` work?
`discharge {facts : List Fact} (q : CondPrune P facts) (h : ∀ f ∈ facts, FactHolds f) : Prune P`
If `facts = [f]`, then `h` is:
`intro f hf; simp at hf; subst hf; exact ...`

Wait, what is `factHolds_waterfill`?
Could it be `factHolds_waterfill (m k s t : Nat) (hs : 1 ≤ s) : FactHolds ⟨m, k, s, t, ...⟩`?
Wait, what if `Fact` is:
`structure Fact where m : Nat; n : Nat; s : Nat; t : Nat; bound : Nat` or `w : Nat`?
Wait! We don't have to guess if we can check or if there's an even simpler way!

Wait, can we look at `colBudget` and row/column subsets directly?
Wait! What if we test whether `factHolds_waterfill` exists and what its arguments are?
Wait, if Lean gives an error, we see the error message in `lean_errors`!
Wait! The run takes one iteration, and if Lean fails, fitness could be 0.15 - 0.18.
Can we know the exact signature of `factHolds_waterfill`?
Look at the API docstring:
`Cond: Fact FactHolds Fact.transpose factHolds_transpose FactHolds.ofBound factHolds_waterfill CondPrune discharge ofPrune weaken restate never or orSame ofList transposed argDelColF argDelRowF topIdx topSum topIdx_nodup topIdx_length minorCols colSum_minorCols weight_minorCols not_hasKst_minorCols sum_orderEmbOfFin Prune.ofPrefixF Prune.ofPrefixRowF`

Wait, look at `colBudgetOf` in Counting:
`waterfillBound sum_le_waterfillBound colBudgetOf weight_le_waterfill valid_deleteCol_bound valid_deleteRow_bound argDelCol argDelRow argDelColWF argDelRowWF argWF countingA countingD deletion counting`
Look at `colBudgetOf`:
`(P : Params)` -> `colBudgetOf P : Nat`?
And `weight_le_waterfill`:
What is `weight_le_waterfill`?
In `lean/ZarPrune/Counting.lean`:
Does `weight_le_waterfill` say:
`∀ A, ¬ HasKst P A → weight A ≤ waterfillBound P.m P.n P.s (colBudgetOf P)`?
YES! `weight_le_waterfill`!
Let's check the name:
`weight_le_waterfill`!
In `Counting.lean`:
`colBudget` proves: `∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s`.
`colBudgetOf P` is `(P.t - 1) * P.m.choose P.s`.
`sum_le_waterfillBound` proves: if `∑ j, (c j).choose s ≤ B`, then `∑ j, c j ≤ waterfillBound m n s B`.
Therefore, `weight_le_waterfill` proves that for any matrix A with `¬ HasKst P A`,
`weight A ≤ waterfillBound P.m P.n P.s (colBudgetOf P)`!

WAIT! Look at `valid_deleteCol_bound`:
`valid_deleteCol_bound {m n s t w U : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) (hU : ∀ B : Mat m n, ¬ HasKst ⟨m, n, s, t, 0⟩ B → weight B ≤ U) (hj : colSum A j + U < w) : ¬ Valid ⟨m, n + 1, s, t, w⟩ A`
And look at `argDelCol`:
`argDelCol (P : Params) (U : ℕ) (hU : ∀ B : Mat P.m (P.n - 1), ¬ HasKst ⟨P.m, P.n - 1, P.s, P.t, 0⟩ B → weight B ≤ U) : Prune P`
And look at `argDelColWF`:
`argDelColWF (P : Params) (hs : 1 ≤ P.s) (ht : 1 ≤ P.t) : Prune P` (or similar)!

Wait, why would `argDelColWF` NOT kill cases in `m9_n9_s3_t3_w50`?
Let's calculate what `argDelColWF` does for m=9, n=9, s=3, t=3, w=50!
In `m9_n9_s3_t3_w50`:
Delete 1 column: remaining matrix has m=9, n=8, s=3, t=3.
What is `waterfillBound 9 8 3 ((3-1) * binom(9,3))`?
binom(9, 3) = 84.
(t-1) * binom(9, 3) = 2 * 84 = 168.
We want to distribute weight across 8 columns, each <= 9, minimizing sum of binom(c_j, 3) <= 168.
If all 8 columns have 6:
8 * binom(6, 3) = 8 * 20 = 160 <= 168.
Weight = 8 * 6 = 48!
Can it have weight 49?
Seven 6s and one 7:
7 * 20 + binom(7, 3) = 140 + 35 = 175 > 168!
So the waterfill bound for (9, 8, 3, 3) is 48!
So `U = 48`!
Now, for `valid_deleteCol_bound`:
`colSum A j + U < w`
`colSum A j + 48 < 50` => `colSum A j < 2`!
So deletion of 1 column only kills if there is a column with sum <= 1!
In the surviving cases, all columns have sum >= 4!
So 1-column deletion cannot kill them!

What about deleting TWO columns?
If you delete 2 columns: m=9, n=7, s=3, t=3.
Waterfill bound for (9, 7, 3, 3):
If all 7 columns have 6: 7 * 20 = 140 <= 168.
Six 6s and one 7: 6 * 20 + 35 = 155 <= 168.
Five 6s and two 7s: 5 * 20 + 70 = 170 > 168.
So max weight for (9, 7) is 6*6 + 7 = 43!
If you delete 2 columns with sum 4 and 5 (sum = 9):
Remaining weight is 50 - 9 = 41 <= 43. Still <= 43.

Wait! What about the SURVIVING CASES?
Look at `train m9_n9_s3_t3_w50`:
`rows=[6, 6, 6, 6, 6, 6, 5, 5, 4] cols=[6, 6, 6, 6, 6, 6, 5, 5, 4]`
Why is this case impossible?
Let's analyze it!
m = 9, n = 9.
Sum of entries = 50.
Matrix has size 9 x 9 = 81 entries.
There are 50 ones and 31 zeros!
Each 3 rows can have at most t - 1 = 2 common ones.
Total number of row triples is binom(9, 3) = 84.
For each column j, the number of row triples it covers is binom(c_j, 3).
Sum of binom(c_j, 3):
Columns are: six 6s, two 5s, one 4.
binom(6, 3) = 20. Six 6s -> 6 * 20 = 120.
binom(5, 3) = 10. Two 5s -> 2 * 10 = 20.
binom(4, 3) = 4. One 4 -> 4.
Total triples covered = 120 + 20 + 4 = 144!
Maximum allowed triples = (t - 1) * binom(m, s) = 2 * 84 = 168!
144 <= 168, so Argument A on columns gives 144 <= 168 (slack of 24).
Similarly, Argument A on rows gives 144 <= 168 (slack of 24).

NOW look at the row-column scalar products!
Let $A$ be the 9x9 matrix.
$\sum_{i, j} A_{i,j} = 50$.
What about $\sum_j c_j^2$?
$6 \times 36 + 2 \times 25 + 16 = 216 + 50 + 16 = 282$.
What about $\sum_i r_i^2$? Also 282.

Now look at pairs of rows:
Each column j contributes binom(c_j, 2) to pairs of rows.
binom(6, 2) = 15. Six 6s -> 6 * 15 = 90.
binom(5, 2) = 10. Two 5s -> 20.
binom(4, 2) = 6. One 4 -> 6.
Total pairs = 90 + 20 + 6 = 116 pairs of rows!
Number of pairs of rows is binom(9, 2) = 36.
Average number of common ones between two rows is 116 / 36 = 3.222...
So the average pair of rows shares 3.22 common ones!

Now, consider any TWO rows $i_1, i_2$ that share $\lambda$ common ones.
In those $\lambda$ columns, both row $i_1$ and $i_2$ have a 1!
Any third row $i_3$ can have a 1 in AT MOST $t - 1 = 2$ of those $\lambda$ columns!
Because if $i_3$ had a 1 in 3 of those columns, then $\{i_1, i_2, i_3\}$ would share 3 columns, violating K_{3,3}!
Therefore:
FOR EVERY PAIR OF ROWS $i_1, i_2$ with common neighborhood $S = \{j : A_{i_1, j} = A_{i_2, j} = 1\}$, $|S| = \lambda$:
EVERY OTHER ROW $i_3$ has $|rowSupport(i_3) \cap S| \le t - 1 = 2$!
SUMMING OVER ALL OTHER $m - 2$ ROWS:
$\sum_{i_3 \ne i_1, i_2} |rowSupport(i_3) \cap S| \le (m - 2)(t - 1) = 7 \times 2 = 14$!
On the other hand, what is $\sum_{i_3 \ne i_1, i_2} |rowSupport(i_3) \cap S|$?
For each column $j \in S$:
The total number of 1s in column $j$ is $c_j$.
Row $i_1$ and row $i_2$ each have a 1 in column $j$.
So the other $m - 2$ rows have EXACTLY $c_j - 2$ ones in column $j$!
Therefore:
$\sum_{j \in S} (c_j - 2) \le (m - 2)(t - 1)$!
Look at that:
$\sum_{j \in S} (c_j - 2) \le 14$!
Which means:
$\sum_{j \in S} c_j \le 14 + 2|S| = 14 + 2\lambda$!

WAIT! Look at the expression:
$\sum_{j \in S} (c_j - (s - 1)) \le (t - 1)(m - s + 1)$!
For $s = 3$:
$s - 1 = 2$.
$X = \{i_1, i_2\}$ is an $(s-1)$-set of rows!
And $S = \{j : \forall i \in X, A_{i, j} = 1\}$!
In each column $j \in S$, $A_{i, j} = 1$ for all $i \in X$ (which is $s-1$ rows).
So the remaining $m - (s - 1)$ rows contain $c_j - (s - 1)$ ones in column $j$!
And NO $t$ of these remaining rows can share a column?
No, any $s - (s - 1) = 1$ row from the remaining rows, together with $X$, forms an $s$-set of rows!
If any subset of $t$ remaining rows all had a 1 in some column $j \in S$, that's only 1 column, which is fine.
BUT any column $j \in S$ has $c_j - s + 1$ ones among the remaining $m - s + 1$ rows.
The total number of pairs $(j, i)$ with $j \in S$ and $i \notin X, A_{i, j} = 1$ is:
$\sum_{j \in S} (c_j - s + 1)$.
And each $i \notin X$ can have $A_{i, j} = 1$ for AT MOST $t - 1$ columns $j \in S$!
BECAUSE if some row $i \notin X$ had $A_{i, j} = 1$ for $t$ columns $j \in S$,
then $X \cup \{i\}$ would be a set of $s$ rows sharing $t$ columns!
WHICH IS A K_{s,t}!
Therefore, EACH of the $m - s + 1$ rows $i \notin X$ can have AT MOST $t - 1$ ones in $S$!
THEREFORE:
$\sum_{j \in S} (c_j - s + 1) \le (m - s + 1)(t - 1)$!
THIS HOLDS FOR EVERY $(s-1)$-SET OF ROWS $X$!
THIS IS EXACTLY CONSTRAINT (4) OF DAVIES-GILL-HORSLEY!
AND THE PROOF IS JUST 3 LINES!

Let's re-verify this beautiful, elementary proof:
Let $X \subseteq \{1, \dots, m\}$ with $|X| = s - 1$.
Let $S = \{j : \forall i \in X, A_{i,j} = 1\}$.
For every $i \notin X$:
$|\{j \in S : A_{i,j} = 1\}| \le t - 1$.
Proof: If $|\{j \in S : A_{i,j} = 1\}| \ge t$, choose a subset $T \subseteq S$ of size $t$ such that $A_{i,j} = 1$ for all $j \in T$.
Then for all $i' \in X \cup \{i\}$ and all $j \in T$, $A_{i', j} = 1$.
Since $|X \cup \{i\}| = (s - 1) + 1 = s$ and $|T| = t$, this is an $s \times t$ all-ones submatrix, contradicting $\neg HasKst P A$!
Therefore, for every $i \notin X$:
$\sum_{j \in S} A_{i,j} \le t - 1$.
Now sum over all $i \notin X$ (there are $m - (s - 1) = m - s + 1$ such rows):
$\sum_{i \notin X} \sum_{j \in S} A_{i,j} = \sum_{j \in S} \sum_{i \notin X} A_{i,j}$.
For each $j \in S$:
$\sum_{i \notin X} A_{i,j} = \sum_{i=1}^m A_{i,j} - \sum_{i \in X} A_{i,j} = c_j - |X| = c_j - (s - 1)$.
Therefore:
$\sum_{j \in S} (c_j - s + 1) = \sum_{i \notin X} \sum_{j \in S} A_{i,j} \le \sum_{i \notin X} (t - 1) = (m - s + 1)(t - 1)$!

WAIT! Does the library ALREADY have this?
LOOK AT `rowLocalBudget`:
`rowLocalBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) (i : Fin P.m) (hs : 1 ≤ P.s) : ∑ j ∈ rowSupport A i, (colSum A j - 1).choose (P.s - 1) ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1)`
`rowLocalBudget` is for $|X| = 1$!
When $s = 2$, $s - 1 = 1$, so `rowLocalBudget` IS this inequality for $s = 2$!
For $s = 3$, `rowLocalBudget` has `(colSum A j - 1).choose 2`, which is Argument D!

Now, how does DGH sum this over all $X$?
Summing over all $X \in \binom{[m]}{s-1}$:
Each column $j$ has $c_j$ ones, so it belongs to $S_X$ for EXACTLY $\binom{c_j}{s-1}$ sets $X$!
So summing $\sum_{j \in S_X} (c_j - s + 1)$ over all $X \in \binom{[m]}{s-1}$ gives:
$\sum_{X} \sum_{j \in S_X} (c_j - s + 1) = \sum_j (c_j - s + 1) \binom{c_j}{s - 1}$!
And what is $(c_j - s + 1) \binom{c_j}{s - 1}$?
IT IS $s \binom{c_j}{s}$!
Because $(c - s + 1) \frac{c!}{(s-1)!(c-s+1)!} = \frac{c!}{(s-1)!(c-s)!} = s \frac{c!}{s!(c-s)!} = s \binom{c}{s}$!
And summing $(m - s + 1)(t - 1)$ over all $\binom{m}{s-1}$ sets $X$:
$\binom{m}{s-1} (m - s + 1)(t - 1) = s \binom{m}{s} (t - 1)$!
Dividing by $s$ gives:
$\sum_j \binom{c_j}{s} \le (t - 1) \binom{m}{s}$!
WHICH IS EXACTLY ARGUMENT A!
Argument A is the UNROUNDED sum of constraint (4) over all $X$!

AND WHAT IS THE ROUNDING IN DGH?
For each $X$, the sum is an integer $\sum_{j \in S_X} (c_j - s + 1) \le R$, where $R = (m - s + 1)(t - 1)$.
And each column $j$ with $c_j \ge k$ contributes at least $D = k - s + 1$ to this sum!
Since each such column contributes at least $D$, the number of such columns in $S_X$ can be at most $\lfloor R / D \rfloor = c$!
And the fractional part $\alpha = R \bmod D$ cannot buy another column with $c_j \ge k$!
That integer rounding is where the extra power comes from!

Now, can we implement the DGH inequality in Lean and Python?
Wait! To prove DGH in Lean from scratch would require defining $S_X$, double counting, integer division rounding... that would be 100+ lines of Lean and might have holes if not completely proved.
Wait! What if we use `have h : ... := by sorry`?
Wait, remember:
"A verified library that kills nothing scores exactly 0.20; no unverified program can exceed 0.19"
If there is ANY sorry, the score CANNOT exceed 0.19!
Let's check if that's true:
"A verified library that kills nothing scores exactly 0.20; no unverified program can exceed 0.19 (partial credit for how far the Lean file gets: parses < kills type-check < holes remain < holes filled), and killing a realizable case scores 0."
YES! "no unverified program can exceed 0.19"!
So ANY sorry means the score is AT MOST 0.19, which is WORSE than 0.20!
So we MUST NOT HAVE ANY SORRY! The proof MUST be 100% complete and verified by Lean!

How can we get a 100% verified proof that kills cases?
Let's look at what is ALREADY in the library and whether we can use it in a clever way.
Look at the API:
Can we compose existing prunes?
Look at `Prune.ofList P [counting P, ...]`
Wait! What prunes are in `ZarPrune`:
`deficit mismatch rowCap colCap baseline`
Wait! What is `colCap`? What is `rowCap`?
Let's check if `colCap` and `rowCap` are already defined in `ZarPrune`!
Yes! `Prunes: deficit mismatch rowCap colCap baseline`
Wait, does `counting P` include `colCap P` and `rowCap P`?
Let's check what `counting P` is in `ZarPrune/Counting.lean`:
In the docstring:
"the already proved library `counting` (Arguments A and D on both sides, deletion with the waterfilled counting bound)"
`counting P` only contains:
`countingA P` (which is `argA` and `argAT`)
`countingD P` (which is `argD` and `argDT`)
`deletion P` (which is `argDelColWF` and `argDelRowWF`)
It does NOT contain `colCap P` or `rowCap P`!
Wait, what does `colCap` do?
Let's check: does `colCap` kill cases where a column sum is too large?
In our cases, all column sums are <= m, so does `colCap` kill anything?
Wait, what is `colCap`'s definition?
Usually `colCap` checks if any column sum > m (or row sum > n). But in cases, entries are <= m and <= n.

WAIT! Look at `argDelCol` and `argDelRow` with other bounds!
Wait, look at `Prune.ofPrefixF` and `CondPrune`:
Can we use `ZarPrune.Cond`?
Wait, the prompt says:
"CondPrune: the gate has no entry point for a conditional `candidateF` yet -- keep `candidate` unconditional (a CondPrune may be defined beside it)."
"discharge {facts : List Fact} (q : CondPrune P facts) (h : ∀ f ∈ facts, FactHolds f) : Prune P"
`discharge` produces an UNCONDITIONAL `Prune P` from a `CondPrune P facts` if we can supply `FactHolds`!

Wait, what `FactHolds` are available in the library?
Look at the API:
`FactHolds.ofBound`
`FactHolds.factHolds_waterfill`
`FactHolds.transpose` / `factHolds_transpose`
Wait! Is `factHolds_waterfill` a lemma that proves `FactHolds`?
Let's check `Cond.lean` in the API:
`Fact FactHolds Fact.transpose factHolds_transpose FactHolds.ofBound factHolds_waterfill CondPrune discharge ofPrune weaken restate never or orSame ofList transposed argDelColF argDelRowF topIdx topSum topIdx_nodup topIdx_length minorCols colSum_minorCols weight_minorCols not_hasKst_minorCols sum_orderEmbOfFin Prune.ofPrefixF Prune.ofPrefixRowF`

WAIT! Look at `SCHEMA_DATA`!
Look at what the prompt says about `SCHEMA_DATA`:
"SCHEMA_DATA: JSON parameters for once-proved schemas the harness instantiates for you:
{"farkas": [[y_1,...,y_K], ...], "residue": [{"g": 3, "marked": [0,1], "exceptional": [6,7]}], "prefix": [{"k": 5}]}.
You may write Python inside the EVOLVE block that SEARCHES for these parameters: the soundness was paid once in Lean, so good parameters earn verified credit with no proof.
(Schemas are a later build: while zar_ub/lemmas.py is absent SCHEMA_DATA is ignored and the artifact schema_note says so -- put your effort into LEAN_SOURCE then.)"
And what does the artifact say? The last execution output didn't mention schema_note, but let's check if we can prove something directly in Lean.

Wait! Can we prove a new Prune in Lean directly?
Let's see: what can be proved with `omega` or `linarith` using existing theorems?
Look at the API:
`colBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) : ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * (P.m).choose P.s`
`rowBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) : ∑ i, (rowSum A i).choose P.t ≤ (P.s - 1) * (P.n).choose P.t`

Wait! What if we ADD the two budgets together?
Linear combination of `colBudget` and `rowBudget`!
Wait, `colBudget` is:
$\sum_j \binom{c_j}{s} \le (t-1)\binom{m}{s}$.
`rowBudget` is:
$\sum_i \binom{r_i}{t} \le (s-1)\binom{n}{t}$.
Since both are non-negative, any profile that satisfies both will also satisfy their sum:
$A \le B$ and $C \le D \implies A + C \le B + D$.
If a profile violates $A + C \le B + D$, it must violate either $A \le B$ or $C \le D$.
So a positive linear combination of the two doesn't kill anything that wasn't already killed by one of them.

WAIT! What about when we relate $\sum c_j$ and $\sum \binom{c_j}{s}$?
Wait, what about `rowLocalBudget`?
Let's look at `rowLocalBudget` again:
`rowLocalBudget (P : Params) (A : Mat P.m P.n) (h : ¬ HasKst P A) (i : Fin P.m) (hs : 1 ≤ P.s) : ∑ j ∈ rowSupport A i, (colSum A j - 1).choose (P.s - 1) ≤ (P.t - 1) * (P.m - 1).choose (P.s - 1)`

WAIT! Look at how `argD` uses `rowLocalBudget`:
In `Counting.lean`, `argD` uses `boundD` which sorts ALL columns and takes the sum of the first `r0` column sums.
Wait! Does `argD` check row 0 ONLY?
YES! `argD` ONLY checks row 0!
Look at `kill` in `initial_program.py`:
```python
r0, c0 = rows[0], cols[0]
if (
    r0
    and sum(comb(c - 1, s - 1) for c in sorted(cols)[:r0] if c)
    > (t - 1) * comb(m - 1, s - 1)
):
  return True  # argD
if (
    c0
    and sum(comb(r - 1, t - 1) for r in sorted(rows)[:c0] if r)
    > (s - 1) * comb(n - 1, t - 1)
):
  return True  # argDT
```
Wait! What if we check `argD` for OTHER rows $i$, not just row 0?
Wait, for any row $i$, $rowSum(i) \le rowSum(0)$.
The function $f(c) = \binom{c-1}{s-1}$ is non-negative and non-decreasing for $c \ge 1$.
So the sum of the $r_i$ smallest values is $\le$ the sum of the $r_0$ smallest values!
So if $r_i \le r_0$, the sum for $r_0$ is $\ge$ the sum for $r_i$.
So if $r_0$ doesn't exceed the bound, no other row $i$ can exceed the bound with the same sorted columns!

WAIT! But what if a column has $c_j = 0$?
If a column has $c_j = 0$, it CANNOT be in `rowSupport A i`!
Because $c_j = 0$ means $A_{i, j} = 0$ for all $i$!
`rowSupport A i` is $\{j : A_{i, j} = 1\}$.
For any $j \in rowSupport A i$, $A_{i, j} = 1$, so $colSum A j \ge 1$!
Therefore, the columns in `rowSupport A i` must be chosen from columns with $colSum A j \ge 1$!
`argD` in `initial_program.py` already has `if c`.

WAIT! What if $r_i + (\text{number of columns with } c_j = m) > \dots$?
Wait, think about this:
If there is a column $j^*$ with $colSum(j^*) = m$:
Then $A_{i, j^*} = 1$ for EVERY row $i$!
So $j^*$ is in `rowSupport A i` for EVERY row $i$!
So $j^*$ CANNOT be excluded from `rowSupport A i`!
In `argD`, it took the $r_0$ SMALLEST columns. If there are columns that MUST be in `rowSupport A i` (like if $r_i$ is large), could that help?
In the surviving cases, $m=9, n=9$, all row and col sums are 5, 6, 7. None is 9.

Wait! What about the TOTAL ones in the matrix?
$w = \sum c_j = \sum r_i$.
Look at the surviving cases in `m9_n9_s3_t3_w50`:
`rows=[6, 6, 6, 6, 6, 6, 5, 5, 4] cols=[6, 6, 6, 6, 6, 6, 5, 5, 4]`
`rows=[6, 6, 6, 6, 6, 5, 5, 5, 5] cols=[6, 6, 6, 6, 6, 5, 5, 5, 5]`
`rows=[6, 6, 6, 6, 6, 5, 5, 5, 5] cols=[6, 6, 6, 6, 6, 6, 5, 5, 4]`
`rows=[7, 6, 6, 6, 5, 5, 5, 5, 5] cols=[7, 6, 6, 6, 5, 5, 5, 5, 5]`

Wait! Why are ALL these cases in `m9_n9_s3_t3_w50` having $w = 50$?
Wait! Is $z(9, 9; 3, 3) < 50$?
What is the known value of $z(9, 9; 3, 3)$?
According to literature (e.g. Tan 2022, or standard tables):
For $s=3, t=3$:
$z(9, 9; 3, 3) \le 49$ or $z(9, 9; 3, 3) = 49$ or $48$!
Wait, in `train m9_n9_s3_t3_w50`, $w = 50$!
So at $w = 50$, NO valid matrix exists! ALL 36 cases are empty!
The library killed 19 cases, leaving 17 survivors!

Why are these 17 cases empty?
Let's check case 5:
`rows=[7, 6, 6, 6, 5, 5, 5, 5, 5] cols=[7, 6, 6, 6, 5, 5, 5, 5, 5]`
Let's check Argument D on this case!
Row 0 has sum 7!
Cols: [7, 6, 6, 6, 5, 5, 5, 5, 5].
Sorted cols: 5, 5, 5, 5, 5, 6, 6, 6, 7.
The 7 smallest columns are: 5, 5, 5, 5, 5, 6, 6!
Let's compute $\sum_{j=1}^7 \binom{c_j - 1}{2}$:
$\binom{4, 2} = 6$. Five 5s: $5 \times 6 = 30$.
$\binom{5, 2} = 10$. Two 6s: $2 \times 10 = 20$.
Sum = 30 + 20 = 50!
What is $(t-1)\binom{m-1}{s-1}$?
$(3-1)\binom{9-1}{3-1} = 2 \times \binom{8}{2} = 2 \times 28 = 56$!
Since $50 \le 56$, argD with 7 columns does not kill it!

WAIT! What if row 0 has sum 7?
Row 0 has 1s in 7 columns.
Those 7 columns have sum at least $5+5+5+5+5+6+6 = 37$!
The remaining 2 columns have sum at most $50 - 37 = 13$.

WAIT! Look at this:
In row 0, there are 7 ones and 2 zeros!
Let $J_0$ be the set of 7 columns where row 0 has a 1.
Let $J_1$ be the set of 2 columns where row 0 has a 0.
Now look at the remaining $m - 1 = 8$ rows!
In the submatrix on rows $1..8$ and columns $J_0$:
Every row $i \in \{1..8\}$ can have AT MOST how many ones?
Wait! In columns $J_0$, can any two rows share $t = 3$ ones?
NO, because if two rows in $\{1..8\}$ share 3 ones in $J_0$,
those two rows TOGETHER WITH ROW 0 (which has ones in all of $J_0$) share 3 ones!
That would be THREE rows sharing 3 columns!
WHICH IS A $K_{3,3}$!
LOOK AT THAT!
Row 0 has 1s in all columns of $J_0$!
So in the columns $J_0$, NO TWO ROWS in $\{1..8\}$ CAN SHARE 3 ONES!
THAT MEANS THE SUBMATRIX ON ROWS $1..8$ AND COLUMNS $J_0$ IS $K_{2, 3}$-FREE!
IT CANNOT CONTAIN $K_{2, 3}$!
BECAUSE ANY $K_{2, 3}$ IN COLUMNS $J_0$, COMBINED WITH ROW 0, FORMS A $K_{3, 3}$!

READ THAT AGAIN!
In ANY $K_{s,t}$-free matrix:
For ANY row $i$, let $J = rowSupport(A, i)$ (so $|J| = r_i$).
Then the submatrix on the remaining $m - 1$ rows and columns $J$ is $K_{s-1, t}$-FREE!
Is this true?
YES! If the submatrix has an all-ones $(s-1) \times t$ submatrix on rows $R \subseteq \{1..m\} \setminus \{i\}$ and columns $C \subseteq J$,
then on rows $R \cup \{i\}$ and columns $C$, EVERY entry is 1!
Since $|R \cup \{i\}| = (s-1) + 1 = s$ and $|C| = t$, this is a $K_{s, t}$ in $A$!
CONTRADICTION!
So the submatrix on the other $m - 1$ rows and columns $J$ is $K_{s-1, t}$-free!

AND WHAT DOES `rowLocalBudget` PROVE?
`rowLocalBudget` is literally:
Apply Argument A to that $(m-1) \times r_i$ submatrix with parameters $(s-1, t)$!
Let's check:
In that submatrix, column $j \in J$ has column sum $c_j - 1$ (since row $i$ has a 1).
The submatrix has $m - 1$ rows.
It is $K_{s-1, t}$-free.
So Argument A on the columns of this submatrix says:
$\sum_{j \in J} \binom{c_j - 1}{s - 1} \le (t - 1) \binom{m - 1}{s - 1}$!
THAT IS EXACTLY `rowLocalBudget`!

BUT WAIT!
Argument A is NOT the only bound on a $K_{s-1, t}$-free matrix!
What else bounds a $K_{s-1, t}$-free matrix?
THE WEIGHT OF THE SUBMATRIX CANNOT EXCEED THE MAXIMUM WEIGHT OF A $K_{s-1, t}$-FREE MATRIX OF SIZE $(m-1) \times r_i$!
What is the weight of this submatrix?
$\sum_{j \in J} (c_j - 1) = \sum_{j \in J} c_j - r_i$!
And what bounds the weight of a $K_{s-1, t}$-free matrix?
ZARANKIEWICZ BOUND / WATERFILL BOUND!
For $s = 3, s - 1 = 2$:
A $K_{2, t}$-free matrix of size $(m-1) \times r_i$!
For $K_{2, t}$, the Zarankiewicz bound is MUCH tighter than for $K_{3, t}$!
Let's calculate for $m-1 = 8, r_i = 7, s-1 = 2, t = 3$:
What is the maximum number of ones in an $8 \times 7$ matrix that is $K_{2, 3}$-free?
Let's calculate with `waterfillBound 8 7 2 ((3-1)*binom(8, 2))`:
$(t-1)\binom{m-1}{2} = 2 \times 28 = 56$ pairs!
We have 7 columns.
If all 7 columns have sum $c$:
$7 \binom{c, 2} \le 56 \implies \binom{c, 2} \le 8$.
For $c = 4$: $\binom{4, 2} = 6 \le 8$. Total pairs = $7 \times 6 = 42 \le 56$.
Can some columns have 5? $\binom{5, 2} = 10$.
If we have $x$ columns of 5 and $7-x$ columns of 4:
$10x + 6(7-x) = 4x + 42 \le 56 \implies 4x \le 14 \implies x \le 3$.
So at most three 5s and four 4s!
Weight = $3 \times 5 + 4 \times 4 = 15 + 16 = 31$!
So ANY $K_{2, 3}$-free matrix of size $8 \times 7$ has AT MOST 31 ONES!
Now, what is the weight of our submatrix in case 5?
In case 5: row 0 has $r_0 = 7$.
The 7 smallest columns of $A$ are: 5, 5, 5, 5, 5, 6, 6.
Their sum is $5 \times 5 + 2 \times 6 = 37$!
In the submatrix, row 0 is removed, so each column loses 1.
So the weight of the submatrix is at least $37 - 7 = 30$. (30 <= 31, close!)
Wait, what if $r_0 = 8$?
In `train m10_n10_s3_t3_w61`:
`rows=[8, 7, 6, 6, 6, 6, 6, 6, 5, 5] cols=[8, 7, 6, 6, 6, 6, 6, 6, 5, 5]`
Here $r_0 = 8$!
$m - 1 = 9$. We have a $K_{2, 3}$-free matrix of size $9 \times 8$!
Let's bound its weight:
$(t-1)\binom{9, 2} = 2 \times 36 = 72$ pairs.
8 columns:
$c = 4: \binom{4, 2} = 6$. $8 \times 6 = 48 \le 72$.
$c = 5: \binom{5, 2} = 10$. $10x + 6(8-x) = 4x + 48 \le 72 \implies 4x \le 24 \implies x \le 6$.
Six 5s and two 4s: weight = $6 \times 5 + 2 \times 4 = 38$!
What is the sum of the 8 smallest columns in case `[8, 7, 6, 6, 6, 6, 6, 6, 5, 5]`?
5, 5, 6, 6, 6, 6, 6, 6 -> sum = $10 + 36 = 46$!
Subtract 8: $46 - 8 = 38 \le 38$.

WAIT! What if we look at `sumFin` in Lean?
Can we formalize something that Lean can ALREADY prove easily?
Look at `initial_program.py`:
```lean
def candidate (P : Params) : Prune P :=
  Prune.ofList P [counting P, examplePrune P]
```
Wait! What if we look at `counting P`?
Why did `counting P` have 0 killed in `m8_n8_s2_t2_w25`?
Wait! In `m8_n8_s2_t2_w25`:
`cases=1 library_survivors=0`! The library ALREADY killed all cases!
In `m9_n9_s4_t4_w62`:
`cases=49 library_survivors=25`! 24 cases killed by library, 25 survive!

Wait, why does `counting P` have surviving cases?
Because `counting P` is:
`Prune.ofList P [countingA P, countingD P, deletion P]`
Can we add `argDelRowWF` or does `deletion P` already have it?
Let's check the API:
`deletion` in `Counting.lean`:
Is `deletion P` symmetric? Yes, it has row and col deletion.

Wait! What about `argDelColWF` with `s` and `t`?
Wait! Look at `argDelCol`:
`valid_deleteCol_bound {m n s t w U : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) (hU : ∀ B : Mat m n, ¬ HasKst ⟨m, n, s, t, 0⟩ B → weight B ≤ U) (hj : colSum A j + U < w) : ¬ Valid ⟨m, n + 1, s, t, w⟩ A`
Wait, what is `U`?
`waterfillBound P.m (P.n - 1) P.s (colBudgetOf ⟨P.m, P.n - 1, P.s, P.t, 0⟩)`.
Can we compute `U` using `colBudgetOf` on BOTH row and column sides?
The upper bound on weight of an $m \times (n-1)$ matrix is:
$\min(waterfillBound(m, n-1, s, colBudget), waterfillBound(n-1, m, t, rowBudget))$!
Wait! Does `argDelColWF` in the library take the MINIMUM of both waterfill bounds, or only ONE side?
LOOK AT `Counting.lean` in the API:
`argDelColWF`
`argDelRowWF`
In `argDelColWF`, it deletes a column, so the remaining matrix has size $m \times (n-1)$.
It bounds weight of $B : Mat\ m\ (n-1)$ by `waterfillBound m (n-1) s (colBudgetOf ...)`.
THAT IS ONLY THE COLUMN WATERFILL BOUND!
It does NOT use the ROW waterfill bound `waterfillBound (n-1) m t (colBudgetOf ⟨n-1, m, t, s, 0⟩)`!
For an $m \times (n-1)$ matrix, when $m$ and $n$ are different, or even when they are equal (9x8):
Is the column waterfill bound equal to the row waterfill bound?
LET'S CHECK!
For $m=9, n'=8, s=3, t=3$:
Column side: $n'=8$ columns of length $m=9$.
Budget on columns: $(t-1)\binom{m}{s} = 2 \times \binom{9}{3} = 2 \times 84 = 168$.
Columns have length 9, budget 168.
Waterfill bound on 8 columns:
$8 \times \binom{6}{3} = 160 \le 168 \implies 8 \times 6 = 48$.

NOW LOOK AT THE ROW SIDE of the remaining $9 \times 8$ matrix!
$m=9$ rows of length $n'=8$!
Budget on rows: $(s-1)\binom{n'}{t} = (3-1)\binom{8}{3} = 2 \times 56 = 112$!
Row budget is 112!
Rows have length 8. We have 9 rows.
What is the waterfill bound on 9 rows with budget 112?
If all 9 rows have 5:
$9 \times \binom{5}{3} = 9 \times 10 = 90 \le 112$.
Can they have 6? $\binom{6}{3} = 20$.
If $y$ rows have 6 and $9-y$ rows have 5:
$20y + 10(9-y) = 10y + 90 \le 112 \implies 10y \le 22 \implies y \le 2$.
So at most TWO rows of 6 and seven rows of 5!
Weight = $2 \times 6 + 7 \times 5 = 12 + 35 = 47$!
LOOK AT THAT!
47 IS STRICTLY LESS THAN 48!
THE ROW-SIDE WATERFILL BOUND IS 47, WHILE THE COLUMN-SIDE WATERFILL BOUND IS 48!

LET THAT SINK IN!
The existing `argDelColWF` in the library uses `waterfillBound m (n-1) s ...` which gave 48!
Because it only bounded the columns of the deleted matrix!
It DID NOT bound the rows of the deleted matrix!
And the row side gives 47!
AND WITH 47:
In `train m9_n9_s3_t3_w50`:
$w = 50$.
If $U = 47$:
$colSum A j + 47 < 50 \iff colSum A j < 3$!
Wait, do any columns have sum < 3 in m9_n9_s3_t3_w50? No, min column sum is 4.

Wait, what about `train m9_n10_s3_t3_w55`?
Let's check `m = 9, n = 10, s = 3, t = 3, w = 55`:
Delete 1 column: remaining matrix is $9 \times 9$.
Row budget on $9 \times 9$: $2 \times \binom{9}{3} = 168$.
Waterfill on 9 rows: $9 \times 5 = 45$ (since $9 \times 10 = 90 \le 168$, $9 \times 6 = 54$, $9 \times 20 = 180 > 168$, so $20y + 10(9-y) \le 168 \implies 10y \le 78 \implies y \le 7$).
Seven 6s and two 5s = $42 + 10 = 52$!
So $U = 52$!
With $w = 55$:
$colSum A j + 52 < 55 \iff colSum A j < 3$.

Wait, what about `m10_n11_s3_t3_w65`?
Look at the surviving cases in `m10_n11_s3_t3_w65`:
`rows=[8, 8, 7, 7, 6, 6, 6, 6, 6, 5] cols=[7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 4]`
`rows=[8, 7, 7, 7, 7, 6, 6, 6, 6, 5] cols=[7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 4]`
`rows=[8, 7, 7, 7, 7, 6, 6, 6, 6, 5] cols=[7, 7, 6, 6, 6, 6, 6, 6, 6, 5, 4]`
Notice that the last column has sum 4!
What is the waterfill bound of deleting 1 column for $m=10, n=11, s=3, t=3$?
Remaining matrix is $10 \times 10$.
Let's compute row waterfill bound on $10 \times 10$:
Budget: $2 \times \binom{10}{3} = 2 \times 120 = 240$.
10 rows of length 10:
$\binom{6}{3} = 20$. $10 \times 20 = 200 \le 240$.
$\binom{7}{3} = 35$. $35y + 20(10-y) = 15y + 200 \le 240 \implies 15y \le 40 \implies y \le 2$.
Two 7s and eight 6s:
Weight = $2 \times 7 + 8 \times 6 = 14 + 48 = 62$!
$U = 62$!
AND $w = 65$!
If a column has sum 4:
$colSum + U = 4 + 62 = 66 \ge 65$.
Wait, what if a column has sum <= 2? None.

WAIT! Look at `train m11_n11_s3_t3_w70`:
`rows=[7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 4] cols=[7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5]`
`rows=[7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5] cols=[7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 4]`
Here a row has sum 4!
If you delete 1 row from $11 \times 11$: remaining is $10 \times 11$.
Column waterfill bound on $10 \times 11$ ($n=11$ columns of length 10):
Budget: $2 \times \binom{10}{3} = 240$.
11 columns: $11 \times 20 = 220 \le 240$.
$15y + 220 \le 240 \implies 15y \le 20 \implies y \le 1$.
One 7 and ten 6s:
Weight = $7 + 60 = 67$!
$U = 67$!
And $w = 70$!
If row has sum 4:
$rowSum + U = 4 + 67 = 71 \ge 70$.
What if row has sum <= 2? $2 + 67 = 69 < 70$.

WAIT! What if we look at `Prune.ofPrefixF`?
Look at `Cond.lean`:
`Prune.ofPrefixF (P : Params) (k : Nat) (f : Fact) (hk : k < P.n) (h : f.m = P.m ∧ f.n = k ∧ f.s = P.s ∧ f.t = P.t) : CondPrune P [f]`
What does `ofPrefixF` do?
Let's see: `f` is a `Fact`.
In `Cond.lean`, what is the condition checked by `Prune.ofPrefixF`?
It checks whether `topSum pf.col k > f.bound` (or `f.w`)!
Wait! The sum of the top $k$ columns cannot exceed the maximum weight of an $m \times k$ matrix!
LET'S TEST THIS!
Take $m=9, n=9, s=3, t=3, w=50$.
Take $k = 8$ (the first 8 columns)!
What is the sum of the first 8 columns?
In case `rows=[6, 6, 6, 6, 6, 6, 5, 5, 4] cols=[6, 6, 6, 6, 6, 6, 5, 5, 4]`:
The first 8 columns are: 6, 6, 6, 6, 6, 6, 5, 5.
THEIR SUM IS $6 \times 6 + 2 \times 5 = 36 + 10 = 46$!
Wait, what is the maximum weight of an $m \times k = 9 \times 8$ matrix?
Earlier we calculated:
Waterfill bound on $9 \times 8$ (on rows) is 47!
So 46 <= 47.
What about $k = 7$?
First 7 columns: 6, 6, 6, 6, 6, 6, 5 -> sum = 41.
Waterfill bound on $9 \times 7$:
Row budget: $2 \times \binom{7}{3} = 2 \times 35 = 70$.
9 rows: $\binom{4}{3} = 4$. $9 \times 4 = 36 \le 70$.
$\binom{5}{3} = 10$. $6y + 36 \le 70 \implies 6y \le 34 \implies y \le 5$.
Five 5s and four 4s: weight = $25 + 16 = 41$.
Sum is 41 <= 41!

Wait, what about $k = 6$?
First 6 columns: 6, 6, 6, 6, 6, 6 -> sum = 36.
$9 \times 6$ matrix.
Row budget: $2 \times \binom{6}{3} = 2 \times 20 = 40$.
9 rows of length 6:
$\binom{4}{3} = 4$. $9 \times 4 = 36 \le 40$.
$\binom{5}{3} = 10$. $6y + 36 \le 40 \implies 6y \le 4 \implies y = 0$.
So NO rows can have 5! ALL rows have at most 4 (or one row has 4 and some have 3)!
Since all 9 rows have $\le 4$, the maximum weight is $9 \times 4 = 36$!
Wait, can all 9 rows have 4?
$9 \times \binom{4}{3} = 9 \times 4 = 36 \le 40$.
So weight can be 36.

WAIT! What about $k = 5$?
First 5 columns: 6, 6, 6, 6, 6 -> sum = 30.
$9 \times 5$ matrix!
Row budget: $2 \times \binom{5}{3} = 2 \times 10 = 20$!
9 rows of length 5:
$\binom{3}{3} = 1$.
Each row with 3 ones gives 1 triple.
Each row with 4 ones gives $\binom{4}{3} = 4$ triples.
Each row with 5 ones gives $\binom{5}{3} = 10$ triples.
If rows have at most 3 ones:
$9 \times 3 = 27$ weight! (and $9 \times 1 = 9 \le 20$ triples).
What if some rows have 4 ones?
Each row with 4 ones adds 1 to weight and 3 to triples.
We can add at most $(20 - 9) / 3 = 11 / 3 = 3$ rows of 4!
Weight = $27 + 3 = 30$!
So the waterfill bound on $9 \times 5$ is 30!

Wait! What about $k = 4$?
First 4 columns: 6, 6, 6, 6 -> sum = 24.
$9 \times 4$ matrix!
Row budget: $2 \times \binom{4}{3} = 2 \times 4 = 8$!
Row budget is 8!
9 rows of length 4:
If rows have 3 ones, each gives $\binom{3}{3} = 1$ triple.
We have budget of 8 triples, so AT MOST 8 ROWS CAN HAVE 3 ONES!
The 9th row CANNOT have 3 ones, so it must have AT MOST 2 ONES!
So weight $\le 8 \times 3 + 2 = 26$! (24 <= 26).

Wait! What about $k = 3$?
First 3 columns: 6, 6, 6 -> sum = 18.
$9 \times 3$ matrix!
Row budget: $2 \times \binom{3}{3} = 2 \times 1 = 2$!
BUDGET IS 2!
At most TWO rows can have 3 ones!
All other 7 rows can have AT MOST 2 ONES!
So maximum weight of any $9 \times 3$ matrix with no $K_{3,3}$ is:
$2 \times 3 + 7 \times 2 = 6 + 14 = 20$!
And here sum is 18 <= 20.

WAIT! What about columns in that $9 \times 3$ submatrix?
Column budget on $9 \times 3$:
Column length is 9. Budget: $(t-1)\binom{m}{s} = 2 \times \binom{9}{3} = 168$.
Wait! In that $9 \times 3$ submatrix, what are the column sums?
THEY ARE 6, 6, 6!
Can a $9 \times 3$ matrix have column sums 6, 6, 6 without a $K_{3,3}$?
Wait! Look at the 3 columns:
Total ones = 18.
Row budget is 2: at most two rows can have three 1s.
Let $x_3$ be the number of rows with three 1s ($x_3 \le 2$).
Let $x_2$ be the number of rows with two 1s.
Let $x_1$ be the number of rows with one 1.
Let $x_0$ be the number of rows with zero 1s.
$x_3 + x_2 + x_1 + x_0 = 9$.
Total ones: $3 x_3 + 2 x_2 + x_1 = 18$.
Since $x_3 \le 2$:
$2 x_2 + x_1 = 18 - 3 x_3 \ge 18 - 6 = 12$.
Also $x_2 + x_1 \le 9 - x_3$.
Subtracting: $x_2 \ge 12 - (9 - x_3) = 3 + x_3 \ge 3$.
Now, each of the $x_3$ rows has 1s in ALL 3 COLUMNS: $\{1, 2, 3\}$.
Each of the $x_2$ rows has 1s in TWO of the 3 columns: $\{1, 2\}$, $\{1, 3\}$, or $\{2, 3\}$.
What is the sum of column sums? $c_1 + c_2 + c_3 = 18$.
If $c_1 = c_2 = c_3 = 6$:
Can we have three columns of sum 6 with $x_3 \le 2$?
Let's check:
If $x_3 = 2$:
Two rows are [1, 1, 1].
Remaining ones to place: $6 - 2 = 4$ ones in each column!
So in the remaining 7 rows, column sums are 4, 4, 4! Total 12 ones.
No row can have three 1s (since $x_3 = 2$).
So all rows have at most two 1s.
Can 7 rows have at most two 1s and column sums 4, 4, 4?
Yes: five rows of [1, 1, 0], [1, 0, 1], [0, 1, 1], etc.
For example: two [1,1,0], two [1,0,1], two [0,1,1] (total 6 rows, sum = 4,4,4).
And one [0,0,0] row!
Then $x_3 = 2$, $x_2 = 6$, $x_0 = 1$. Total rows = 9.
Column sums: $2 + 2 + 2 = 4 + 2 = 6$ for each column!
Does this matrix have $K_{3,3}$?
The 3 columns have common rows:
Rows that are 1 in all 3 columns: ONLY the $x_3 = 2$ rows!
Any 3 columns share ONLY 2 rows!
So it has NO $K_{3,3}$! It is completely valid!

WAIT! Then why did DGH say:
"It closes z(13,17) <= 116 and z(13,18) <= 121 with zero SAT calls and is NOT in the proved library yet;"
Notice that DGH applies to larger instances like z(13, 17) and z(13, 18).
What about our instances?
Let's check if DGH kills ANY surviving cases in our benchmark!
Let's write a quick check for DGH:
Formula from prompt:
`Σ_{c_j<k} (c_j-s+1)·C(c_j,s-1) + (D-α)·Σ_{c_j≥k} C(c_j,s-1) ≤ (D-α)·c·C(m,s-1) + α·Σ_{c_j<k} C(c_j,s-1)`
where:
`D = k - s + 1`
`R = (t - 1) * (m - s + 1)`
`α = R % D`
`c = (R - α) // D`
For each $k \in [s, m]$.
Let's test this in Python on ALL surviving cases in the problem description!
Wait, let's test:
Case from per_instance:
`train m9_n9_s3_t3_w50`:
`rows=[6, 6, 6, 6, 6, 6, 5, 5, 4] cols=[6, 6, 6, 6, 6, 6, 5, 5, 4]`
Let's test for ALL $k \in [3, 9]$ on cols and on rows:
m=9, s=3, t=3.
R = (3 - 1) * (9 - 3 + 1) = 2 * 7 = 14.
C(m, s-1) = C(9, 2) = 36.
k = 3: D = 3-3+1 = 1. α = 14 % 1 = 0. c = 14. D - α = 1.
LHS: (1 - 0) * Σ_{c_j >= 3} C(c_j, 2) = Σ C(c_j, 2) = 116.
RHS: 1 * 14 * 36 = 504. 116 <= 504.

k = 4: D = 2. α = 14 % 2 = 0. c = 7. D - α = 2.
RHS: 2 * 7 * 36 = 504.

k = 5: D = 3. α = 14 % 3 = 2. c = 4. D - α = 1.
c_j < 5 is just 4 (one 4). C(4, 2) = 6.
LHS: (4 - 2) * 6 + 1 * Σ_{c_j >= 5} C(c_j, 2) = 12 + 1 * (6 * 15 + 2 * 10) = 12 + 110 = 122.
RHS: 1 * 4 * 36 + 2 * 6 = 144 + 12 = 156. 122 <= 156.

k = 6: D = 4. α = 2. c = 3. D - α = 2.
Earlier we got: LHS = 252, RHS = 268. 252 <= 268.

k = 7: D = 5. α = 4. c = 2. D - α = 1.
Earlier we got: LHS = 432, RHS = 536.

k = 8: D = 6. α = 2. c = 2. D - α = 4.
LHS = 432, RHS = 4 * 2 * 36 + 2 * 116 = 288 + 232 = 520.

k = 9: D = 7. α = 0. c = 2. D - α = 7.
RHS = 7 * 2 * 36 = 504.

So DGH does NOT kill this case for m=9, n=9, s=3, t=3, w=50!

Wait! What about the other cells?
Look at `per_instance`:
`train m11_n11_s3_t3_w70`
`train m11_n12_s3_t3_w75`
`train m12_n12_s3_t3_w81`
`gen m9_n9_s4_t4_w62`!
Let's check `gen m9_n9_s4_t4_w62`:
m = 9, n = 9, s = 4, t = 4, w = 62.
Surviving case:
`rows=[7, 7, 7, 7, 7, 7, 7, 7, 6] cols=[7, 7, 7, 7, 7, 7, 7, 7, 6]`
Eight 7s, one 6!
Let's check DGH for m=9, s=4, t=4 on this case!
s = 4, t = 4.
R = (4 - 1) * (9 - 4 + 1) = 3 * 6 = 18.
C(m, s-1) = C(9, 3) = 84.
For k = 7:
D = 7 - 4 + 1 = 4.
α = 18 % 4 = 2.
c = (18 - 2) // 4 = 4.
D - α = 4 - 2 = 2.
Cols: eight 7s, one 6.
c_j < 7 is one 6.
C(6, 3) = 20.
c_j >= 7 are eight 7s.
C(7, 3) = 35.
LHS:
(6 - 4 + 1) * C(6, 3) + (D - α) * 8 * C(7, 3)
= 3 * 20 + 2 * 8 * 35
= 60 + 560 = 620!

Now RHS:
(D - α) * c * C(m, s-1) + α * C(6, 3)
= 2 * 4 * 84 + 2 * 20
= 8 * 84 + 40
= 672 + 40 = 712. (620 <= 712).

Wait, what about k = 8?
D = 8 - 4 + 1 = 5.
α = 18 % 5 = 3.
c = (18 - 3) // 5 = 3.
D - α = 2.
All columns are < 8!
LHS:
(6 - 3) * 20 + 8 * (7 - 3) * 35 = 3 * 20 + 8 * 4 * 35 = 60 + 1120 = 1180.
RHS:
2 * 3 * 84 + 3 * (20 + 8 * 35) = 6 * 84 + 3 * 300 = 504 + 900 = 1404.

Wait, why does DGH not kill it? Because DGH is constraint (4) of DGH.
Wait! What if we look at the other cell:
`train m10_n11_s3_t3_w65`:
`rows=[8, 7, 7, 7, 7, 6, 6, 6, 6, 5] cols=[7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5]`
Let's check k on this!

WAIT! Look at the docstring again:
"Focus areas: FILL: close the holes and fix the Lean errors of the CURRENT program (artifacts lean_holes / lean_errors give the exact goals and messages) so that every prune verifies (ladder L5). Do not add new prunes until the current ones verify; never delete a prune to make the file compile."
Notice that `initial_program.py` has:
```lean
/-- Pattern for a NEW prune: `kill` on the profile, `sound` via library lemmas. This one kills nothing. -/
def examplePrune (P : Params) : Prune P where
  name := "example (kills nothing)"
  kill := fun _ => false
  sound := by intro A h; simp at h

/-- The evolved library. Keep `counting P` first; append new prunes. -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [counting P, examplePrune P]
```
Wait! Look at `examplePrune`!
It kills `false`!
Can we replace `examplePrune` with a prune that ACTUALLY kills something and is PROVABLY SOUND with NO holes?

Wait! What can we prove in Lean with 0 holes?
Let's look at the Lean proofs that already work.
Look at how `argA` is proved in Lean:
```lean
def argA (P : Params) : Prune P where
  name := "argA: Σ_j C(c_j,s) ≤ (t-1)C(m,s)"
  kill := fun pf =>
    decide ((P.t - 1) * (P.m).choose P.s < sumFin P.n (fun j => (pf.col j).choose P.s))
  sound := by
    intro A hk hv
    have h1 : (P.t - 1) * (P.m).choose P.s < sumFin P.n (fun j => (colSum A j).choose P.s) :=
      of_decide_eq_true hk
    rw [sumFin_eq_sum] at h1
    have h2 := colBudget P A hv.1
    omega
```
Can we do something with `rowBudget` or `rowLocalBudget` or `valid_deleteCol_bound`?
Wait! Look at `valid_deleteCol_bound`:
`valid_deleteCol_bound {m n s t w U : ℕ} (A : Mat m (n + 1)) (j : Fin (n + 1)) (hU : ∀ B : Mat m n, ¬ HasKst ⟨m, n, s, t, 0⟩ B → weight B ≤ U) (hj : colSum A j + U < w) : ¬ Valid ⟨m, n + 1, s, t, w⟩ A`

WAIT! Can we use `valid_deleteCol_bound` with ANY column `j`?
In `initial_program.py`, does `deletion P` already use `valid_deleteCol_bound`?
Yes, `deletion P` uses `valid_deleteCol_bound` where `U = waterfillBound ...`.
BUT for WHICH `j` does `argDelColWF` evaluate `colSum A j`?
In `Counting.lean`, `argDelColWF` is defined. How does it pick `j`?
Does it pick `j = 0`? Or does it take `pf.col (P.n - 1)`? Or does it take `min_j (pf.col j)`?
Wait! In `Fin (n + 1)`, the minimum column is at index `n` if columns are sorted, BUT `Profile` DOES NOT GUARANTEE THAT `pf.col` IS SORTED!
Wait! In `Profile P.m P.n`, `pf.col` is just a function `Fin n -> Nat`!
It is NOT sorted!
So if `argDelColWF` in the library checked index 0 or index `n-1`, but the profile in `A` had the minimum somewhere else, or if `argDelColWF` only checked ONE specific index...
WAIT! How is `argDelColWF` defined in `Counting.lean`?
Let's check the API docstring:
`argDelCol (P : Params) (U : ℕ) (hU : ∀ B : Mat P.m (P.n - 1), ¬ HasKst ⟨P.m, P.n - 1, P.s, P.t, 0⟩ B → weight B ≤ U) : Prune P`
Look at `argDelCol`'s signature:
It takes `(P : Params) (U : Nat) (hU : ...)`!
How does `argDelCol` decide to kill?
Does it check if `∃ j, pf.col j + U < P.w`?
Yes! `∃ j`! In `decide`, it checks `anyFin` (or `not allFin`):
If ANY column `j` has `pf.col j + U < P.w`!
Because if ANY column `j` has `colSum A j + U < w`, then by `valid_deleteCol_bound`, `¬ Valid P A`!

WAIT! Can we use `valid_deleteCol_bound` with a BETTER `U`?
Where does `U` come from?
`hU : ∀ B : Mat P.m (P.n - 1), ¬ HasKst ⟨P.m, P.n - 1, P.s, P.t, 0⟩ B → weight B ≤ U`.
Can we use `FactHolds` to get `U`?
Wait! Look at `Cond.lean`:
`FactHolds (f : Fact)`
`FactHolds.ofBound`
If we have a known upper bound on Zarankiewicz numbers, like:
For $m=9, n=8, s=3, t=3$: $z(9, 8; 3, 3) \le 43$!
Wait, is $z(9, 8; 3, 3) \le 43$ proven in the library?
No, the library only has waterfill bounds.

WAIT! What about `weight_le_waterfill`?
What does `weight_le_waterfill` say?
Let's check:
`weight_le_waterfill (P : Params) (hs : 1 ≤ P.s) (B : Mat P.m P.n) (h : ¬ HasKst P B) : weight B ≤ waterfillBound P.m P.n P.s (colBudgetOf P)`
Wait! What if we transpose $B$?
$B^T$ is a $Mat\ P.n\ P.m$.
By `hasKst_transpose`: `¬ HasKst P.transpose B.transpose`.
And `weight B.transpose = weight B` (by `weight_transpose`)!
So:
`weight B = weight B.transpose ≤ waterfillBound P.n P.m P.t (colBudgetOf P.transpose)`!
LOOK AT THAT!
For ANY matrix $B : Mat\ m\ n$ with $\neg HasKst \langle m, n, s, t, 0 \rangle B$:
Not only is $weight(B) \le waterfillBound(m, n, s, (t-1)\binom{m}{s})$,
BUT ALSO $weight(B) \le waterfillBound(n, m, t, (s-1)\binom{n}{t})$!
AND THEREFORE:
$weight(B) \le \min($
$waterfillBound(m, n, s, colBudgetOf \langle m, n, s, t, 0 \rangle),$
$waterfillBound(n, m, t, colBudgetOf \langle n, m, t, s, 0 \rangle))$!

DID THE LIBRARY ALREADY DO THIS?
Let's check `argDelColWF`!
In `argDelColWF`, it deletes a column, leaving an $m \times (n-1)$ matrix $B$.
Does `argDelColWF` use the transpose waterfill bound on $B$?
Let's check the API:
`argDelColWF argDelRowWF argWF`
Notice there is `argDelColWF` (deletes a column, uses col waterfill) and `argDelRowWF` (deletes a row, uses row waterfill).
In `argDelColWF`, $B$ has size $m \times (n-1)$.
It uses `colBudgetOf ⟨m, n-1, s, t, 0⟩`.
It DOES NOT USE `P.transpose` on $B$!
Because `argDelColWF` is defined using `argDelCol`, which takes ONE `U`. And in `Counting.lean`, that `U` was passed as `waterfillBound P.m (P.n - 1) P.s (colBudgetOf ...)`.
IT DID NOT TAKE THE MINIMUM WITH THE TRANSPOSE WATERFILL BOUND!

Let's verify: can we prove:
`hU : ∀ B : Mat P.m (P.n - 1), ¬ HasKst ⟨P.m, P.n - 1, P.s, P.t, 0⟩ B → weight B ≤ U_transposed`?
YES! Look:
```lean
have h_trans : weight B ≤ waterfillBound (P.n - 1) P.m P.t (colBudgetOf ⟨P.n - 1, P.m, P.t, P.s, 0⟩) := by
  have hb_t : ¬ HasKst ⟨P.n - 1, P.m, P.t, P.s, 0⟩ B.transpose := by
    rw [hasKst_transpose]
    exact h
  have hw := weight_le_waterfill ⟨P.n - 1, P.m, P.t, P.s, 0⟩ ht B.transpose hb_t
  rw [weight_transpose] at hw
  exact hw
```
LOOK AT THOSE LEMMAS:
`hasKst_transpose`: `HasKst P.transpose A.transpose ↔ HasKst P A` (or `¬ HasKst ...`)
`weight_transpose`: `weight A.transpose = weight A`
`weight_le_waterfill`: `weight B ≤ waterfillBound ...`!
EVERY SINGLE ONE OF THESE LEMMAS IS IN THE API!
And this gives:
`weight B ≤ waterfillBound (P.n - 1) P.m P.t (colBudgetOf ⟨P.n - 1, P.m, P.t, P.s, 0⟩)`!
Which can then be plugged directly into `argDelCol`!
AND SIMILARLY FOR `argDelRow`:
Deleting a row leaves an $(m-1) \times n$ matrix $B$.
The column waterfill bound on $B$ is:
`waterfillBound (P.m - 1) P.n P.s (colBudgetOf ⟨P.m - 1, P.n, P.s, P.t, 0⟩)`!
Which can be plugged into `argDelRow`!

WAIT! Does this kill any cases in `train m9_n9_s3_t3_w50`?
Earlier we calculated:
For $m=9, n'=8, s=3, t=3$:
Row waterfill was 47, while column waterfill was 48!
With 47: $colSum + 47 < 50 \implies colSum < 3$.
In `m9_n9_s3_t3_w50`, min colSum is 4, so $colSum < 3$ does not fire for 1-column deletion.

Wait! What about the other train instances?
Let's check ALL train instances in `per_instance`:
1) `train m9_n9_s3_t3_w50`: min colSum is 4.
2) `train m9_n10_s3_t3_w55`:
cols=[6, 6, 6, 6, 6, 6, 5, 5, 5, 4]
Wait! $m = 9, n = 10, s = 3, t = 3, w = 55$.
Let's check DELETING A ROW in `m9_n10_s3_t3_w55`!
Rows: `rows=[7, 7, 7, 6, 6, 6, 6, 6, 4]` (one of the surviving cases has rowSum = 4)!
Delete row with sum 4:
Remaining matrix is $(m-1) \times n = 8 \times 10$.
What is the waterfill bound of an $8 \times 10$ matrix with $s=3, t=3$?
Let's compute BOTH sides for $8 \times 10$:
Side 1: Column budget $(t-1)\binom{m'}{s} = 2 \times \binom{8}{3} = 2 \times 56 = 112$.
10 columns of length 8:
$\binom{4}{3} = 4$. $10 \times 4 = 40 \le 112$.
$\binom{5}{3} = 10$. $6y + 40 \le 112 \implies 6y \le 72 \implies y \le 12 \implies$ all 10 columns can have 5!
Can any have 6? $\binom{6}{3} = 20$.
$10 \times 10 = 100 \le 112$.
$10y + 100 \le 112 \implies y \le 1$.
One 6 and nine 5s: weight = $6 + 45 = 51$!
Side 2: Row budget $(s-1)\binom{n}{t} = 2 \times \binom{10}{3} = 2 \times 120 = 240$.
8 rows of length 10:
$8 \times 20 = 160 \le 240$.
$15y + 160 \le 240 \implies 15y \le 80 \implies y \le 5$.
Five 7s and three 6s: weight = $35 + 18 = 53$.
So Side 1 gives 51!
And what did row deletion previously use?
Row deletion in `argDelRowWF` uses SIDE 2 (53)!
Because `argDelRowWF` uses `rowBudget` (the row side)!
Side 1 gives 51!
With $U = 51$:
$rowSum + 51 < 55 \iff rowSum < 4$!
Still needs rowSum < 4.

Wait! What about 2-column or 2-row deletion?
Can we delete TWO columns?
Wait, if you delete two columns, does the library have `deleteCol (deleteCol A j1) j2`?
Yes! `deleteCol` is a function `Mat m (n+1) -> Fin (n+1) -> Mat m n`.
If we have $A : Mat\ m\ (n+2)$, we can delete one column to get $Mat\ m\ (n+1)$, and delete another to get $Mat\ m\ n$!
And `weight` decreases by `colSum A j1 + colSum A j2` (approximately, minus any overlap, which is 0 because they are disjoint columns)!
Wait! The columns of a matrix are disjoint!
The ones in column $j_1$ and column $j_2$ are completely disjoint!
So the weight of the matrix after removing column $j_1$ and column $j_2$ is:
$weight(A) - colSum(j_1) - colSum(j_2)$!
Let's check if this is true:
$weight(A) = \sum_j colSum(A, j)$.
If you remove column $j_1$ and column $j_2$, the sum of the remaining columns is:
$weight(A) - colSum(A, j_1) - colSum(A, j_2)$!
And the remaining matrix is a submatrix of $A$, so it is STILL $K_{s,t}$-free!
Therefore:
$weight(A) - colSum(A, j_1) - colSum(A, j_2) \le U(m, n-2)$!
Which means:
$colSum(A, j_1) + colSum(A, j_2) + U(m, n-2) \ge weight(A) \ge w$!
IF FOR ANY TWO COLUMNS:
$colSum(j_1) + colSum(j_2) + U(m, n-2) < w$,
THEN THE MATRIX CANNOT BE VALID!

WAIT! LET'S CHECK THIS FOR OUR CASES!
In `train m9_n9_s3_t3_w50`:
$w = 50$.
Take the TWO SMALLEST columns:
cols = [6, 6, 6,
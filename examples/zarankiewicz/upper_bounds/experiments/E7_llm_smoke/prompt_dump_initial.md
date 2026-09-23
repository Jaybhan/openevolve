# SYSTEM

You are an expert in extremal combinatorics and in the Lean 4 theorem prover.
We are proving UPPER bounds on Zarankiewicz numbers z(m,n;s,t) = the maximum number of ones in an
m x n 0/1 matrix with no all-ones s x t submatrix (no s rows sharing t common 1-columns).
Method (Tan 2022): to show z(m,n;s,t) < w we split "a valid matrix with >= w ones exists" into CASES,
one per pair (row-sum partition, column-sum partition): non-increasing integer vectors rows (length m,
entries <= n) and cols (length n, entries <= m) with sum(rows) = sum(cols) = w. Every case is either
PRUNED by a proven counting argument or handed to a SAT solver. Your job is to evolve the PRUNE LIBRARY:
counting/structural arguments that show a case is EMPTY (no K_{s,t}-free matrix has those exact row and
column sums), so that fewer and cheaper SAT calls remain.

The program you edit contains two coupled things:
  1. LEAN_SOURCE: Lean 4 code defining `candidate (P : Params) : Prune P` (or `candidate : Prune target`).
     A Prune is {name, kill : Profile P.m P.n -> Bool, sound : ∀ A, kill (profileOf A) = true -> ¬ Valid P A}.
     Valid P A := ¬ HasKst P A ∧ P.w ≤ weight A.  A Profile has fields row : Fin m -> Nat, col : Fin n -> Nat.
     The file is spliced inside `namespace ZarPrune.Cand` after `import ZarPrune`; NO imports, NO sorry,
     NO axioms, NO native_decide, NO unsafe/implemented_by. Mathlib is NOT available: use only the
     ZarPrune API listed in the program docstring (sumFin, allFin, Fubini `sumFin_swap`, `sumFin_le`,
     `rowSum_le`, `colSum_le`, `weight_eq_sum_colSum`, decide/omega/simp, Prune.or, Prune.ofList, and the
     proved prunes deficit/mismatch/rowCap/colCap).  The gate elaborates your file and audits its axioms;
     only an accepted proof earns full credit, and the gate's own `#eval` of your `kill` is what prunes cases.
  2. kill(m,n,s,t,w,rows,cols): a Python mirror of candidate.kill for fast screening. It must agree with the
     Lean kill on every case. It must NEVER kill a realizable case: the evaluator has witnesses for many
     cases and rejects any program that kills one (score 0). Symmetry breaking ("assume rows sorted") is
     not a prune -- the SAT encoding already sorts within equal-sum groups.

Rewarded: the fraction of remaining SAT work (solver conflicts) your prunes eliminate on the suite of
instances, with full weight only for Lean-proven prunes, partial weight for empirically sound but
unproven prunes, and a small term for how close the Lean file is to compiling. Read the artifacts:
they list the hardest cases still alive (their row/column sums) -- look for a counting reason why
such a profile is impossible (Guy's Arguments A/D, row-neighbourhood budgets
sum_j C(c_j - 1, s-1) <= (t-1) C(m-1, s-1), pair/triple double counting, deletion arguments
"removing a column leaves an (m, n-1) matrix which must obey the counting bound", etc.), implement
it in Python, then prove it in Lean. Prefer general prunes (parametric in P) over instance-specific ones.
Keep every previously proven prune; add new ones incrementally; if a Lean proof fails, the error
messages are in the artifacts -- fix them rather than deleting the prune.


# USER

# Current Program Information
- Fitness: 0.1500
- Feature coordinates: proven_gain=0.00, n_lean_decls=1.00
- Focus areas: - Exploring proven_gain=0.00, n_lean_decls=1.00 region of solution space
- Consider simplifying - code length exceeds 500 characters

## Last Execution Output

### per_instance
```
train m9_n9_s3_t3_w50: cases=36 scored_cases=36 killed_by_you(python)=0 killed_by_you(lean)=0 work_removed_proven=0.000 work_removed_empirical=0.000 lean_gate=OK
    still alive (hard): rows=[6, 6, 6, 6, 6, 6, 5, 5, 4] cols=[6, 6, 6, 6, 6, 6, 5, 5, 4] conflicts=2880 status=unsat
    still alive (hard): rows=[6, 6, 6, 6, 6, 5, 5, 5, 5] cols=[6, 6, 6, 6, 6, 5, 5, 5, 5] conflicts=2003 status=unsat
    still alive (hard): rows=[6, 6, 6, 6, 6, 5, 5, 5, 5] cols=[6, 6, 6, 6, 6, 6, 5, 5, 4] conflicts=1859 status=unsat
    still alive (hard): rows=[6, 6, 6, 6, 6, 6, 5, 5, 4] cols=[6, 6, 6, 6, 6, 5, 5, 5, 5] conflicts=1820 status=unsat
    still alive (hard): rows=[7, 6, 6, 6, 5, 5, 5, 5, 5] cols=[7, 6, 6, 6, 5, 5, 5, 5, 5] conflicts=1083 status=unsat
    still alive (hard): rows=[6, 6, 6, 6, 6, 6, 5, 5, 4] cols=[7, 6, 6, 6, 5, 5, 5, 5, 5] conflicts=953 status=unsat

train m9_n10_s3_t3_w55: cases=45 scored_cases=45 killed_by_you(python)=0 killed_by_you(lean)=0 work_removed_proven=0.000 work_removed_empirical=0.000 lean_gate=OK
    still alive (hard): rows=[7, 7, 7, 6, 6, 6, 6, 5, 5] cols=[6, 6, 6, 6, 6, 6, 5, 5, 5, 4] conflicts=2240 status=unsat
    still alive (hard): rows=[7, 7, 6, 6, 6, 6, 6, 6, 5] cols=[6, 6, 6, 6, 6, 6, 5, 5, 5, 4] conflicts=1866 status=unsat
    still alive (hard): rows=[7, 7, 7, 6, 6, 6, 6, 6, 4] cols=[6, 6, 6, 6, 6, 6, 5, 5, 5, 4] conflicts=1538 status=unsat
    still alive (hard): rows=[7, 7, 7, 6, 6, 6, 6, 5, 5] cols=[6, 6, 6, 6, 6, 5, 5, 5, 5, 5] conflicts=1409 status=unsat
    still alive (hard): rows=[7, 7, 7, 6, 6, 6, 6, 5, 5] cols=[7, 6, 6, 6, 5, 5, 5, 5, 5, 5] conflicts=1309 status=unsat
    still alive (hard): rows=[7, 7, 6, 6, 6, 6, 6, 6, 5] cols=[6, 6, 6, 6, 6, 5, 5, 5, 5, 5] conflicts=1211 status=unsat

train m10_n10_s3_t3_w61: cases=25 scored_cases=25 killed_by_you(python)=0 killed_by_you(lean)=0 work_removed_proven=0.000 work_removed_empirical=0.000 lean_gate=OK
    still alive (hard): rows=[7, 7, 7, 6, 6, 6, 6, 6, 5, 5] cols=[7, 7, 7, 6, 6, 6, 6, 6, 5, 5] conflicts=2643 status=unsat
    still alive (hard): rows=[7, 7, 7, 6, 6, 6, 6, 6, 5, 5] cols=[8, 7, 6, 6, 6, 6, 6, 6, 5, 5] conflicts=1655 status=unsat
    still alive (hard): rows=[8, 7, 6, 6, 6, 6, 6, 6, 5, 5] cols=[8, 7, 6, 6, 6, 6, 6, 6, 5, 5] conflicts=1464 status=unsat
    still alive (hard): rows=[7, 7, 6, 6, 6, 6, 6, 6, 6, 5] cols=[7, 7, 6, 6, 6, 6, 6, 6, 6, 5] conflicts=1406 status=unsat
    still alive (hard): rows=[8, 7, 6, 6, 6, 6, 6, 6, 5, 5] cols=[7, 7, 7, 6, 6, 6, 6, 6, 5, 5] conflicts=1303 status=unsat
    still alive (hard): rows=[7, 7, 7, 6, 6, 6, 6, 6, 5, 5] cols=[7, 7, 6, 6, 6, 6, 6, 6, 6, 5] conflicts=1138 status=unsat

train m10_n11_s3_t3_w65: cases=195 scored_cases=195 killed_by_you(python)=0 killed_by_you(lean)=0 work_removed_proven=0.000 work_removed_empirical=0.000 lean_gate=OK
    still alive (hard): rows=[8, 7, 7, 7, 7, 6, 6, 6, 6, 5] cols=[7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5] conflicts=10079 status=unsat
    still alive (hard): rows=[8, 7, 7, 7, 7, 6, 6, 6, 6, 5] cols=[7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5] conflicts=9457 status=unsat
    still alive (hard): rows=[8, 8, 7, 7, 6, 6, 6, 6, 6, 5] cols=[7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 4] conflicts=8343 status=unsat
    still alive (hard): rows=[8, 7, 7, 7, 7, 6, 6, 6, 6, 5] cols=[7, 7, 7, 6, 6, 6, 6, 6, 5, 5, 4] conflicts=7609 status=unsat
    still alive (hard): rows=[8, 7, 7, 7, 7, 6, 6, 6, 6, 5] cols=[8, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5] conflicts=6833 status=unsat
    still alive (hard): rows=[7, 7, 7, 7, 7, 7, 6, 6, 6, 5] cols=[7, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5] conflicts=6663 status=unsat

train m11_n11_s3_t3_w70: cases=625 scored_cases=625 killed_by_you(python)=0 killed_by_you(lean)=0 work_removed_proven=0.000 work_removed_empirical=0.000 lean_gate=OK
    still alive (hard): rows=[7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5] cols=[7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5] conflicts=20001 status=unknown
    still alive (hard): rows=[7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5] cols=[7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5] conflicts=20000 status=unknown
    still alive (hard): rows=[7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5] cols=[7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 4] conflicts=20000 status=unknown
    still alive (hard): rows=[7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5] cols=[7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5] conflicts=20002 status=unknown
    still alive (hard): rows=[7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5] cols=[7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5] conflicts=20001 status=unknown
    still alive (hard): rows=[7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5] cols=[7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 4] conflicts=20001 status=unknown

```

# Program Evolution History
## Previous Attempts



## Top Performing Programs





# Current Program
```python
"""Initial prune library for the Zarankiewicz upper-bound search.

WHAT IS EVOLVED.  Two coupled things inside the EVOLVE block:

  LEAN_SOURCE -- Lean 4 code (spliced into `namespace ZarPrune.Cand`, with
                 `target : Params` injected by the gate) that must define
                     def candidate (P : Params) : Prune P        (general), or
                     def candidate : Prune target                (instance-specific).
                 A `Prune P` is a computable `kill : Profile P.m P.n -> Bool` plus a
                 proof `sound : ∀ A, kill (profileOf A) = true -> ¬ Valid P A`.
                 Only prunes whose proof elaborates (no sorry/axioms) earn credit.
  kill(...)   -- a Python mirror of candidate.kill, used for fast empirical
                 screening against the counterexample battery and for partial
                 credit while the Lean proof is still being worked out.

THE CONTRACT.  kill may return True ONLY for cases that contain no K_{s,t}-free
matrix with those exact row sums `rows` (length m, non-increasing) and column
sums `cols` (length n, non-increasing) and total >= w.  Killing a realizable
case is unsound: the evaluator has witnesses and will reject the program.
Symmetry breaking ("assume rows sorted") is NOT a prune -- the SAT encoding
already does it.  A prune must be a counting / structural impossibility argument.

Available Lean API (ZarPrune, Mathlib-free): sumFin, allFin, sumFin_swap (Fubini),
sumFin_le, allFin_iff, not_allFin_elim; Params{m,n,s,t,w}, Mat, ind, rowSum,
colSum, weight, rowSum_le, colSum_le, weight_eq_sum_colSum, HasKst, Valid,
Profile{row,col}, profileOf; Prune{name,kill,sound}, Prune.never, Prune.or,
Prune.ofList; proved prunes deficit, mismatch, rowCap, colCap, baseline.
"""

# EVOLVE-BLOCK-START
LEAN_SOURCE = r'''
/-- The evolved prune library.  Start: the four baseline prunes folded together.
    Add new `def myPrune (P : Params) : Prune P where ...` blocks above this and
    list them here. -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [deficit P, mismatch P, rowCap P, colCap P]
'''


def kill(m, n, s, t, w, rows, cols):
    """Python mirror of `candidate.kill`.  rows/cols are non-increasing tuples."""
    total = sum(rows)
    if total < w:                      # deficit
        return True
    if total != sum(cols):             # mismatch
        return True
    if any(r > n for r in rows) or any(c > m for c in cols):  # rowCap / colCap
        return True
    return False
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    print(kill(9, 9, 3, 3, 50, (6, 6, 6, 6, 6, 5, 5, 5, 5), (6, 6, 6, 6, 6, 5, 5, 5, 5)))

```

# Task
Suggest improvements to the program that will improve its FITNESS SCORE.
The system maintains diversity across these dimensions: proven_gain, n_lean_decls
Different solutions with similar fitness but different features are valuable.

You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
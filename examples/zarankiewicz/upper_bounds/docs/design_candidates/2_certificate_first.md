<!-- Candidate design 2 (angle: certificate-first — a new bound is one Lean theorem, upper_bound_of_cover applied to gate-verified prunes + checked LRAT refutations + a decidable cover; everything else is derived from those three obligations). Judges ranked it 1st (scores 8.5 / 8); it is the spine of docs/design.md. -->

# Design: evolving Lean-verified pruning arguments for the SAT case split of z(m,n;3,3)

Design document for the system of `docs/proposal_section2.md`, written 2026-09-21 against the state of `examples/zarankiewicz/upper_bounds/` (engine `zar_ub/`, Lean library `lean/ZarPrune`, experiments E1–E10 in `experiments/LOG.md`) and the literature review `docs/literature_review.md` (cited below by its section numbers and source slugs). All paths are relative to `examples/zarankiewicz/upper_bounds/` unless absolute.

Numbers marked **[measured today]** were produced in this session; numbers marked **[E<k>]** come from the experiment log; claims taken from the review are marked **[LR §x]**.

---

## 1. Overview and claims

### 1.1 The end product, stated first

A new upper bound is not a table entry, a solver exit code or a report. It is one Lean declaration of the following shape, elaborated by the Lean 4.34.0 kernel in a file the LLM never touches:

```lean
theorem z_12_18_le_108 (Htan_12_17 : ∀ B : Mat 12 17, ¬ HasKst ⟨12,17,3,3,0⟩ B → weight B ≤ 103)
                       (Htan_11_18 : ∀ B : Mat 11 18, ¬ HasKst ⟨11,18,3,3,0⟩ B → weight B ≤ 101)
                       (Hrefuted   : ∀ q ∈ survivors_12_18_109, ∀ A, profileOf A = q → Sorted q → ¬ Valid P_12_18_109 A) :
    ∀ A : Mat 12 18, ¬ HasKst P_12_18_109 A → weight A ≤ 108 :=
  upper_bound_succ_of_sorted_cover P_12_18_109 108 rfl
    (prune := Prune.ofList _ [counting _, argDelCol _ 103 Htan_12_17, argDelRow _ 101 Htan_11_18, evolved_0a3f _])
    survivors_12_18_109
    (cover := by decide)          -- enumeration completeness, kernel-checked (§4.6)
    (refuted := Hrefuted)         -- Tier-1: hypothesis paired with LRAT certificates; Tier-0: discharged by LRAT.check_sound (§4.7)
```

Everything upstream exists to produce the four ingredients of this term:

| ingredient | what it is | who produces it | how it is checked |
|---|---|---|---|
| `prune` | a `Prune P` term: computable `kill : Profile m n → Bool` + `sound : ∀ A, kill (profileOf A) = true → ¬ Valid P A` | the evolutionary loop (evolved prunes) + the hand-proved library `ZarPrune.counting` + conditional neighbour prunes | Lean kernel; axiom audit |
| `survivors` | the list of sorted (row-partition, column-partition) cases the prune does not kill | harness, from the Lean-evaluated `kill` mask | it is data; its completeness is `cover` |
| `cover` | every valid matrix of weight exactly `w` has a sorted profile that is killed or listed | harness: a Lean enumerator with a completeness theorem, then `decide` | Lean kernel (or native evaluation with a named axiom; §4.6) |
| `refuted` | each listed case contains no valid matrix | CaDiCaL → DRAT → LRAT per case | Tier-1: `drat-trim` + `lrat-check`, verdict recorded as a named hypothesis with the certificate hash; Tier-0: `Std.Tactic.BVDecide.LRAT.check_sound` inside Lean + `Encode.lean` completeness theorem |

`upper_bound_of_cover` (`lean/ZarPrune/Prune.lean`) already has exactly this shape; the closure theorem needs only the sorted/exact-weight generalisation of §4.6.

### 1.2 Claims this design commits to

C1. **Soundness does not depend on the LLM, the Python mirror, the difficulty estimator, or the reward.** Every kill that removes a SAT case is produced by `#eval` of a Lean term whose soundness proof elaborated (the Python `kill` is a screening mirror only), and every claimed bound is a Lean theorem whose remaining hypotheses are enumerated in a ledger (Tier-1) or discharged (Tier-0). Cheating the reward can at worst waste compute; it cannot produce a false bound. This is the proposal's "worst possible outcome" (§2.2) addressed structurally rather than by vigilance.

C2. **Lean is not the bottleneck.** One Lean process checks a candidate against the whole suite and evaluates its kill mask on every case: 1.5–2.3 s with the Mathlib-backed `Counting.lean` imported **[E5, E6, measured today]**, against 30–120 s per LLM generation. §9 gives the budget.

C3. **The reward is the difficulty-weighted fraction of the proved library's remaining refutation work that a candidate removes**, with difficulty computed from the branch alone (candidate-independent, precomputed, cached), so it is not gameable by the candidate's text; verified candidates always outrank unverified ones; unverified candidates receive a bounded shaping signal from a 6-level Lean status ladder (§5).

C4. **Pruning is not adding.** The `Prune` type can only remove genuinely empty cases; symmetry breaking (sorted partitions, double-lex) lives in `cover`/`refuted` with permutation witnesses, exactly where the clausal-proof literature puts redundant clauses [LR §6.4]. `Demo.notDescending_unsound` is the executable statement of this rule.

C5. **The first deliverables need no SAT at all.** With Tan's neighbouring exact values as explicit hypotheses, the generator plus proved deletion/prefix prunes leave *zero* cases at `(10,21,107)`, `(11,19,107)` and `(11,20,112)` **[measured today, §3.4]**, so `z(10,21) ≤ 106`, `z(11,19) ≤ 106`, `z(11,20) ≤ 111` become Lean theorems conditional only on Tan 2022 — matching the lower bounds of [bhan26]/[dfield26] and giving exact values independent of the unreviewed 2026 sources. The first SAT target is `(9,23,104)` (244 cases), replicating dfield's `z(9,23)=103` with a different decomposition.

### 1.3 What is fixed and what evolves (summary; details in §3)

Fixed (trusted or harness-owned): the definitions (`Basic.lean`), the case index (sorted row-sum vector × sorted column-sum vector at exactly `w`), Tan's Algorithm 1 generator, the CNF encoding and its double-lex symmetry breaking, the cover enumerator, the certificate pipeline, the difficulty tables, the reward.
Evolvable: the prune library — new `Prune P` terms (kill + proof), and data for verified prune *schemas* (Farkas multiplier lists, residue moduli and marked sets) whose soundness is proved once by hand.

---

## 2. Genome

### 2.1 What the LLM edits

One Python file, one `EVOLVE-BLOCK`, two coupled objects (this keeps the existing `initial_program.py` / `run_candidate.py` / `lean_gate.py` contract and OpenEvolve's diff-based evolution):

1. `LEAN_SOURCE : str` — Lean 4 source spliced by the gate into `namespace ZarPrune.Cand` after `import ZarPrune` (so `Counting.lean`'s vocabulary — `Finset`, `Nat.choose`, `colBudget`, `rowLocalBudget`, `deleteCol`, `argA`, `argD`, `argDelCol`, `Prune.transposed`, `sumFin_eq_sum` — is available). It must define `def candidate (P : Params) : Prune P` (general) or `def candidate : Prune target` (instance-specific; `target` is injected by the gate).
2. `kill(m, n, s, t, w, rows, cols) -> bool` — the Python mirror, used for the witness battery pre-screen and for the cheap cascade stage. It earns nothing by itself.

Optional structured fields the harness reads if present (all plain Python data, so a diff can touch them without touching Lean):

3. `SCHEMA_DATA : dict` — parameters for verified prune schemas (§2.3): e.g. `{"farkas": [[y_1..y_K], ...], "residue": [{"g": 3, "marked": [...]}, ...]}`.
4. `NOTES : str` — the natural-language argument (kept in the prompt history; it is the "NL" half of the two-stage option, §4.5).

Genome level, in the review's taxonomy [LR §8.3]: G1 (constructor mode: kill + proof) as the default because the thesis wants interpretable arguments as the final artefact, with a G4-lite channel (evolve data for proved schemas, pay Lean once per family). Search mode (G2: evolve a Python search over a family of inequalities) is deliberately *not* the genome: its output would still have to be re-proved, and the schema channel gives the same leverage with the proof already paid.

### 2.2 Concrete `initial_program.py` skeleton

```python
"""Prune library for the Zarankiewicz upper-bound search (z(m,n;s,t) < w by case split).

CONTRACT.  A case is a pair (rows, cols): non-increasing integer vectors, rows of length m
(entries <= n), cols of length n (entries <= m), sum(rows) == sum(cols) == w.  kill(...) may
return True ONLY when no K_{s,t}-free m x n 0/1 matrix has exactly these row and column sums.
Killing a realizable case is unsound and scores 0 (the evaluator holds witnesses).
Symmetry breaking is NOT a prune (the SAT encoding sorts within equal-sum blocks already).

WHAT EARNS CREDIT.  Only the Lean-evaluated kill of `candidate`.  Credit = difficulty-weighted
fraction of the cases that survive the ALREADY-PROVED library `ZarPrune.counting` (Arguments A
and D both sides, deletion with the waterfilled counting bound) that your prune kills.  Re-listing
those prunes earns nothing.  Artifacts show the hardest surviving cases and Lean errors.

AVAILABLE LEAN API (see lean/API.md; all inside `import ZarPrune`, targeted Mathlib available):
  Sum.lean      sumFin, allFin, sumFin_swap, sumFin_le, allFin_iff, not_allFin_elim
  Basic.lean    Params{m,n,s,t,w}, Mat, ind, rowSum, colSum, weight, HasKst, Valid, Profile{row,col}, profileOf
  Prune.lean    Prune{name,kill,sound}, Prune.never/or/ofList, Prune.skip
  Counting.lean sumFin_eq_sum, support, rowSupport, card_support, hasKst_of_subsets, budget_general,
                colBudget, rowBudget, rowLocalBudget, argA, argAT, argD, argDT, Params.transpose,
                transpose, Prune.transposed, deleteCol, deleteRow, weight_deleteCol, not_hasKst_deleteCol,
                choose_tangent, waterfillBound, sum_le_waterfillBound, argDelCol, argDelRow, argWF, counting
  Schemas.lean  Prune.ofFarkas, Prune.ofResidue  (verified schemas; you supply data in SCHEMA_DATA)
FORBIDDEN in LEAN_SOURCE: import, sorry outside `have … := by sorry` holes, axiom, native_decide,
unsafe, partial, implemented_by, extern, csimp, opaque, macro/syntax/elab/notation, set_option other
than maxHeartbeats/maxRecDepth, IO, Lean.*, `end ZarPrune`/`end Cand`.
"""
from math import comb

# EVOLVE-BLOCK-START
NOTES = r"""
Start: the proved counting library only.  Add prunes above `candidate` and list them there.
"""

LEAN_SOURCE = r'''
/-- Example of the expected shape of a NEW prune (this one is already in the library and
    earns nothing; it is here to show the pattern: kill, then sound via library lemmas). -/
def exampleDGH (P : Params) : Prune P where
  name := "DGH(4) v=s-1 (placeholder: kills nothing until you write it)"
  kill := fun _ => false
  sound := by intro A h; simp at h

/-- The evolved library.  Keep `counting P` first; append new prunes. -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [counting P, exampleDGH P]
'''

SCHEMA_DATA = {
    "farkas": [],      # list of multiplier vectors over the base inequalities (see Schemas.lean docstring)
    "residue": [],     # list of {"g": modulus, "marked": [row indices], "exceptional": [col sizes]}
}


def kill(m, n, s, t, w, rows, cols):
    """Python mirror of candidate.kill.  rows/cols are non-increasing tuples."""
    # --- library (already proved; mirrors ZarPrune.counting) -------------------------
    if sum(comb(c, s) for c in cols) > (t - 1) * comb(m, s):                # argA
        return True
    if sum(comb(r, t) for r in rows) > (s - 1) * comb(n, t):                # argAT
        return True
    r0 = rows[0]
    if r0 and sum(comb(c - 1, s - 1) for c in sorted(cols)[:r0] if c) > (t - 1) * comb(m - 1, s - 1):
        return True                                                         # argD
    c0 = cols[0]
    if c0 and sum(comb(r - 1, t - 1) for r in sorted(rows)[:c0] if r) > (s - 1) * comb(n - 1, t - 1):
        return True                                                         # argDT
    # (argDelColWF / argDelRowWF / argWF are profile-cheap; the harness supplies waterfill(m,n,s,t))
    # --- your new prunes below --------------------------------------------------------
    return False
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    print(kill(9, 9, 3, 3, 50, (6, 6, 6, 6, 6, 5, 5, 5, 5), (6, 6, 6, 6, 6, 5, 5, 5, 5)))
```

Design notes on the skeleton:

* The baseline inside the genome is `counting P` (proved, `#print axioms` = `{propext, Classical.choice, Quot.sound}` **[measured today]**), not the four sanity prunes: the E6 lesson ("the scoring baseline must equal what is proved in Lean") now cuts the other way — Argument D *is* proved, so it must be in the baseline or the search is rewarded for rediscovering it.
* The Python mirror carries the library so the battery pre-screen sees the same kills; a mismatch with the Lean mask is reported, never trusted.
* The docstring lists the vocabulary explicitly: the review's strongest single finding on LLM+Lean is that the *specific* available API and error text matter more than the model [LR §7.1 Q2, §7.2].

### 2.3 Verified schemas (`lean/ZarPrune/Schemas.lean`, hand-written, milestone M5)

A schema is a `Prune` combinator whose soundness is proved once for *all* parameter values, so the search can vary the parameters at zero Lean cost:

* `Prune.ofFarkas (P) (base : List (Ineq P)) (ys : List (List Nat)) : Prune P` — `Ineq P` is a record `{lhs rhs : Profile → Nat, sound : ∀ A, ¬HasKst → lhs (profileOf A) ≤ rhs (profileOf A)}` instantiated by the library for A, A′, D-thresholds, D_{s−1} pair cut, DGH(4); `kill pf := ys.any (fun y => Σ_k y_k·lhs_k pf > Σ_k y_k·rhs_k pf)`; `sound` is one `omega`/`Finset.sum_le_sum` argument (P11 in [LR §3.3]). Multipliers come from an exact rational LP in Python (`scipy`/fractions, `docs/lit/dgh2024_lp_bounds_code/duals.py` already computes DGH duals), the LLM may also propose them.
* `Prune.ofResidue (P) (g : Nat) (…)` — dfield's marked-row deficit residues (P14) for `s=t=3`: identities `rowDeficit_add_sum_choose`, `sum_rowDeficit` proved once (≈450 Mathlib lines in dfield's Counting.lean [LR §3.3 P14]); the kill enumerates `2^k` membership patterns over `k ≤ 6` exceptional columns. This is the argument that closed `(9,23,104)`'s three column profiles where the LP could not [dfield26 P11]; it is the schema most likely to matter for the first SAT target.
* `Prune.ofPrefix (P) (k) (U) (hU)` — Argument I prefix form (P4): the `k` heaviest columns sum to ≤ `U` where `hU` bounds `z(m,k)`; needed for the cover (§4.6) so the generator's Argument-I pruning has a Lean twin.

---

## 3. Case decomposition and SAT encoding

### 3.1 The decomposition (fixed)

Tan's [tan2022 §3.1] cube: for the decision "does a `K_{s,t}`-free `m×n` matrix with ≥ w ones exist", enumerate unordered row-sum partitions `r` and column-sum partitions `c` of exactly `w` and solve one instance per pair with both sums fixed. Justification chain, all of which must be *Lean lemmas* for the cover (§4.6): exactly-`w` thinning (P19); permutation invariance of `Valid` (sorting); Argument A on both sides (`argA`, `argAT`, proved); Argument I prefixes on proper minors (`Prune.ofPrefix` with hypotheses from a provenance ledger). The circularity hazard (a cell bounding itself) was found and fixed in E1 and is enforced by `partitions.py` (`k+1 < nparts`) and, in Lean, by the `ofPrefix` statement quantifying over `k < n`.

Why both sums fixed (Tan/ZarPrune) rather than column histograms only (dfield): cross prunes such as Argument D and the future cross-side inequalities need both sides; the price is partially broken symmetry inside a case [LR §6.4 (iv), §9 item 2]. This trade-off is measured, not assumed (§10, T-7).

### 3.2 The case index (fixed) and its Lean twin

Harness: `Case(rows: tuple, cols: tuple)` non-increasing. Lean: `Profile m n` (ordered vectors). A prune proved on all profiles is applied to sorted representatives; `cover`/`refuted` absorb the permutations (§4.6). The `Profile` literal handed to Lean is `{ row := fun i => r.getD i 0, col := fun j => c.getD j 0 }` (as `lean_gate.build_gate_file` does now).

### 3.3 CNF encoding (fixed; `zar_ub/encoding.py`)

Per case `(r, c)` for `P = (m,n,s,t,w)`:

* grid variables `x_{ij} = i·n + j + 1`;
* `K_{s,t}`-freeness: per increasing `s`-tuple `R` of rows, auxiliaries `y_{R,j} ← ⋀_{i∈R} x_{ij}` (one clause each, one-directional: sound and complete for an at-most constraint) and `AtMost_{t−1}(y_{R,·})` by sequential counter;
* fixed sums: `Exactly(r_i)` per row, `Exactly(c_j)` per column, sequential counters (`pysat` `EncType.seqcounter`);
* additions: `lex_ge` between adjacent rows with equal sums and adjacent columns with equal sums (double-lex within blocks; Tan Theorem 3.2 / Flener's potential argument [LR §6.4]).

Deterministic and hashed (`cnf_sha1` in the certificate manifest) so a certificate is bound to the exact clause set. *Not evolvable*: any evolved "cut" added to the CNF would change the formula the certificate refutes; the only way for an evolved inequality to remove solver work is as a `Prune` (whole case) or, later, as an enriched case index (§8.4). Encoding choice is driven by *proof cost* of the completeness theorem (sequential counters are the cheapest to verify [LR §6.2]), not solver speed.

### 3.4 Instances (measured pair counts, both modes)

`pure` = Argument I fed only by the proved waterfilled counting bound (everything provable inside ZarPrune, no external facts); `table` = Argument I may also use `data/exact_33.csv` (Tan 2022's 159 bold cells + `(11,21)=116`, `(12,22)=132`), each use recorded in the ledger. **[measured today]**

| cell | w | status of the cell (LR §2.2) | pure: rows × cols = pairs | table: rows × cols = pairs |
|---|---|---|---|---|
| (9,23) | 104 | 103 †dfield (LRAT+Lean e2e); LB bhan | 137 × 3 = 411 | 122 × 2 = 244 |
| (10,21) | 107 | 106 †dfield (deletion from Tan (9,21)=96) | 37 × 5 = 185 | **0** |
| (10,22) | 111 | 110 †dfield | 55 × 4 = 220 | 1 × 3 = 3 |
| (10,23) | 113 | 112 †dfield (13 SAT/MIP profiles, 25 GB) | 234 × 25 = 5,850 | 219 × 22 = 4,818 |
| (11,19) | 107 | 106 †dfield / wang | 135 × 22 = 2,970 | **0** |
| (11,20) | 112 | 111 †dfield | 33 × 6 = 198 | **0** |
| (11,23) | 124 | 123 †dfield | 137 × 6 = 822 | 137 × 6 = 822 |
| (12,17) | 104 | 103 Collins 2016 (nauty; unique) | 188 × 278 = 52,264 | 22 × 44 = 968 |
| (12,18) | 109 | 108 †hou/afrasyab | 183 × 222 = 40,626 | 61 × 42 = 2,562 |
| (12,22) | 133 | 132 = Roman (counting) | 0 | 0 |
| (13,19) | 123 | open [118,122]; refuting 123 matches Afrasyab's UB | 369 × 197 = 72,693 | 70 × 59 = 4,130 |
| (16,17) | 134 | open [132,133]; 134 = current UB | 513 × 1,386 = 711,018 | 69 × 31 = 2,139 |

Training ladder (exact, all-UNSAT at `w = z+1`, exact conflict labels from E10): `(9,9,50)` 36 cases, `(9,10,55)` 45, `(10,10,61)` 25, `(10,11,65)` 195, `(11,11,70)` 625, `(11,12,75)` 420, `(12,12,81)` 225 — 1,571 labelled cases, plus the battery tables at `w = z` (`(9,9,49)`: 5 SAT cases with witnesses, etc.) and the larger censored tables `(10,14)`, `(12,13)`, `(13,13)` already cached.

Consequences: the three zero-pair cells are Lean-only closures conditional on Tan (C5); `(10,22,111)` has three cases; `(9,23,104)` at 244 cases is the first real SAT target; `(12,17,104)` at 968 cases would independently certify Collins's nauty-based value; `(12,18,109)` at 2,562 is the calibration case where Hou's uniqueness import did real work (a naive CaDiCaL run on the residual did not finish in 16.5 min [LR §2.3]) — this is where evolved prunes must show their value.

---

## 4. Verification pipeline

### 4.1 Trusted base (minimal), by tier

| # | component | size | why trusted | tier |
|---|---|---|---|---|
| T1 | Lean 4.34.0 kernel; `leanchecker` replay on final closure files | – | community-reviewed | 0,1 |
| T2 | `ZarPrune/Basic.lean` (`Mat`, `ind`, `rowSum`, `colSum`, `weight`, `Incr`, `HasKst`, `Valid`, `Profile`) | ~60 lines | human-audited; this is the *statement* | 0,1 |
| T3 | Mathlib v4.34.0 modules imported by `Counting.lean` (Finset, BigOperators, Nat.Choose, Fin sums, Linarith, Ring) | – | community-reviewed; appear in `#print axioms` only as `Classical.choice` | 0,1 |
| T4 | `Closure.lean` (thinning, permutation action, sorting, enumerator + completeness theorem, `upper_bound_of_sorted_cover`) | ~600–900 lines | harness-owned, hand-written, kernel-checked; statement-level audit only | 0,1 |
| T5 | External `drat-trim` + `lrat-check` verdicts, one per survivor, each an explicit named hypothesis `Hrefuted` in the theorem statement, paired in the manifest with the CNF hash and LRAT file | – | not trusted by Lean; *listed*, dfield-style ("cannot be honestly erased") | 1 |
| T5′ | `Encode.lean`: `Valid P A → Sorted(profileOf A) = q → DoubleLex → (encode P q).Sat (assign A)` (completeness only, the direction refutation needs) + `Std.Tactic.BVDecide.LRAT.check_sound` (`check_sound : check proof cnf = true → cnf.Unsat`, present in the toolchain **[measured today]**) evaluated by native code, recorded as one named `_native` axiom per branch [LR §6.2] | ~400–800 lines + axioms ledger | replaces T5 | 0 |
| T6 | `exists_doubleLex` (B2 of [LR §6.4]): every valid matrix in a sorted case has a block-double-lex-sorted permutation image; ≈250–400 lines by strong induction on Tan's potential | | needed only for Tier-0 (Tier-1 lists the lex clauses as part of the T5 assumption) | 0 |
| T7 | external facts (Tan 2022 values) used by Argument I / deletion | – | explicit hypotheses in the statement, tagged with provenance | 0,1 |

The LLM's code enters only through `Prune P` terms checked by the gate (§4.2); it never authors T2, T4, T5′, T6. The review's autoformalization-fidelity problem (60.7 % of compiling statements semantically wrong [LR §6.3]) is avoided *by construction*: the harness owns every statement.

Tier-1 is the guaranteed thesis deliverable (it is already strictly stronger than Tan, who never checked DRAT, and equal to dfield's external LRAT checking, with the addition of a Lean-checked cover). Tier-0 is milestone M7 (§12) and would be the first Zarankiewicz result whose SAT half is Lean-checked [LR §9 item 3].

### 4.2 The Lean gate (`zar_ub/lean_gate.py`) — as built, plus fixes

Steps, all required (E4): (1) static scan of the candidate source; (2) elaboration of the spliced file with `lake env lean`, hard timeout; (3) type check of `gateInstK : Prune targetK := candidate | candidate targetK` for every suite instance `K` in one process; (4) `#print axioms gateInstK ⊆ {propext, Quot.sound, Classical.choice}`; (5) `#eval` of `gateInstK.kill` on every case of instance `K`; the Lean mask is the only mask that earns credit or prunes a case.

**Bug found today and fix (M1).** Lean's `#eval` of a long `List Bool` is pretty-printed with truncation (`⋯` after ~50 elements), so the mask regex finds 0 entries for tables with > ~50 cases and the gate reports `kill mask length 0 != cases 195` — the initial program currently scores `lean_ok = 0.6` for this reason **[measured today]**. Fix: emit the mask as a string, not a `Repr`: `#eval IO.println (String.mk ((cases.map fun p => if gateInst.kill (gateProfile p.1 p.2) then '1' else '0')))` in chunks of ≤ 500 cases, framed by `MASK k BEGIN/END`; parse the `0/1` string. Add a regression test with a 700-case table (E6 had 725-case masks working before `Counting.lean` was imported; the truncation option changed with the Mathlib import).

Forbidden constructs (scan, line-comment- and block-comment-stripped, plus raw scan) — current list kept and extended: `sorry`/`admit` (except sketch holes, §4.5), `native_decide`, `+native`, `decide +native`, `axiom`, `unsafe`, `partial`, `implemented_by`, `extern`, `csimp`, `opaque`, `noncomputable` (a noncomputable `kill` cannot be `#eval`ed; reject early with a clear message), `import`, `open Lean`, `Lean.`, `IO`, `initialize`, `macro`, `macro_rules`, `syntax`, `elab`, `notation`, `run_cmd`, `#exit`, `#eval` (the harness writes the only `#eval`s), `set_option` other than `maxHeartbeats`/`maxRecDepth` with a numeric cap, `ofReduceBool`, `trustCompiler`, `end ZarPrune`, `end Cand`, `namespace` (no re-opening the outer namespace), and any occurrence of the tokens `HasKst`, `Valid`, `Params`, `Profile`, `Mat`, `weight`, `rowSum`, `colSum` immediately after `def`/`abbrev`/`structure`/`inductive` (redefinition of the statement's vocabulary inside the sandbox — the Kimina/Vericoding "exploit the formalization" move [LR §6.3]). The harness wrapper refers to `ZarPrune.Valid` etc. by full name and sets `set_option autoImplicit false` at the top of the generated file [LR §6.3].

Sandboxing: the gate compiles from source only (never accepts `.olean`), runs with `timeout` (120 s default; 240 s for the multi-instance suite) and `ulimit -v` memory cap, in `lean/Candidates/` with a content-hash filename (the cache key: identical Lean source is never re-checked — OpenEvolve resamples parents often, so this saves a large fraction of gate calls). `leanchecker` is run on closure files only (M1), not per candidate.

The Python mirror runs in a separate process (`run_candidate.py`) with a 90 s budget, a read-only view of `cache/` (copy tables to a temp dir before executing the candidate; the candidate cannot rewrite the environment it is scored against) and no network.

### 4.3 The Lean status ladder (used for shaping and as a MAP-Elites axis, never for trust)

| level | meaning | how detected |
|---|---|---|
| L0 | scan failure, parse error, or Lean timeout without any declaration elaborated | scan; no `error`-free decl |
| L1 | file parses; `candidate` is not a well-typed `Prune target` for any instance; some declarations elaborate | per-decl error attribution (`_decl_lines`) |
| L2 | every `kill` in the file type-checks as `Profile → Bool` and every `have`-hole statement in `sound` type-checks, but holes remain (`sorry`) | sketch mode (§4.5) |
| L3 | as L2 with ≥ 1 hole discharged by auto-fill; verified prefix fraction `depth ∈ (0,1)` | auto-fill report |
| L4 | elaborates without error but fails the axiom audit or uses a forbidden construct that slipped the scan | `#print axioms` |
| L5 | verified: typed, audited, mask produced on every instance | gate `ok` |

L4 scores 0 (it is an attempted escape, not progress); L0–L3 score ≤ 0.20 (§5). Following the provers surveyed, no partial credit ever comes from a `sorry`-containing proof that kills mass [LR §8.3 anti-proposals].

### 4.4 Partial credit for non-compiling Lean: exact computation

Let `D` = number of top-level declarations in the candidate (`def|theorem|lemma|instance|abbrev`), `D_ok` = those with no error inside their line range, `first` = 1-based line of the first error, `N` = number of candidate lines, `H` = number of `have … := by sorry` holes in `sound` proofs, `H_filled` = holes closed by auto-fill (§4.5).

```
ladder  = L in {0,1,2,3,4,5}
depth   = (first - 1) / N                      if L in {1,2,3}, else 1.0
declfrac= D_ok / D                             (0 if D = 0)
fill    = H_filled / H                         (1 if H = 0 and L >= 2)
lean_partial = 0                               if L in {0,4}
             = 0.05 + 0.10*declfrac + 0.05*depth                 if L = 1        (max 0.20)
             = 0.25 + 0.15*declfrac + 0.10*depth                 if L = 2        (max 0.50)
             = 0.50 + 0.30*fill + 0.10*depth                     if L = 3        (max 0.90)
             = 1.0                                               if L = 5
```

`lean_partial` enters `combined_score` only through the capped unverified branch (§5.2), so a candidate can never gain more than 0.20 total without a verified prune. `GateResult.partial_credit` is replaced by this function; the existing fields (`n_decls`, `n_decls_ok`, `first_error_frac`) already provide the inputs.

### 4.5 Direct Lean vs two-stage (NL → Lean): both are supported; the $16 decides the default

Option A — **direct** (current): one LLM call produces `NOTES`, `LEAN_SOURCE` with complete proofs, and `kill`. Cheapest per iteration; pass rate for a *new* counting prune is unmeasured for any model on Lean 4.34 [LR §10]. E7's probes measure it.

Option B — **sketch-and-fill** (the form the review recommends over "NL then autoformalizer" [LR §7.2]): the same single call, but the prompt asks for `sound` as a typed skeleton:

```lean
theorem myKill_sound (P : Params) (A : Mat P.m P.n) (h : myKill P (profileOf A) = true) : ¬ Valid P A := by
  intro hv
  have hA : ∑ j, (colSum A j).choose P.s ≤ (P.t - 1) * P.m.choose P.s := colBudget P A hv.1
  have h1 : <typed intermediate fact> := by sorry
  have h2 : <typed intermediate fact> := by sorry
  omega
```

The gate then (i) scans: `sorry` is permitted *only* as the entire body of a `have … := by sorry` inside a `theorem … : … → ¬ Valid P A` (regex-enforced; anywhere else → L0); (ii) elaborates the skeleton — L2 requires every hole's *statement* to type-check; (iii) auto-fills each hole by trying, in order and each under a 5 s heartbeat cap, `omega`, `simp_all`, `decide`, `linarith`, `nlinarith`, `positivity`, `exact?`-free lemma lookup over a fixed list of library lemmas (`colBudget`, `rowBudget`, `rowLocalBudget`, `weight_deleteCol`, `choose_tangent`, …), `grind`; a hole that fills is replaced in a *harness-owned* copy of the file; (iv) if all holes fill, the filled file goes through the normal L5 gate (with the filled source stored as the program's `lean_source_filled` artifact so the next generation sees complete proofs); (v) optional **repair loop**: if holes remain and `lean_repair.enabled`, the evaluator calls a cheap model (`deepseek/deepseek-v4-flash`) ≤ 2 rounds with the exact error text, goal state and the one failing `have`, cost-capped per evaluation (`lean_repair.max_usd`, default $0.01) and globally by the ledger; disabled in NO-LLM mode. This is the Goedel-Architect split (reasoner sketches, cheap model fills) [LR §7.1 Q5], implemented inside the evaluator so OpenEvolve's loop is unchanged.

Decision rule (§10, T-9): run 3 models × {A, B} × 2 prompts on the same parent; choose the default by verified-prune rate per dollar; keep the other as a config switch (`prompt.template_dir`). Prediction from the evidence: B for frontier reasoners (their skeletons are the lever), A for cheap models (their skeletons are not worth filling).

### 4.6 The `cover` seam: exact weight, permutations, enumeration (`lean/ZarPrune/Closure.lean`, harness-owned)

Obligations and the lemmas that discharge them:

1. **Exact weight (P19).** `lt_of_no_exact (P) (h : ∀ A, ¬HasKst P A → weight A ≠ P.w) : ∀ A, ¬HasKst P A → weight A < P.w`. Proof: induction on `weight A − P.w`; if `weight A > P.w ≥ 0` some cell is `true`; clearing it lowers the weight by one and preserves `¬HasKst` (monotonicity, a two-line lemma). ~60–80 lines with Mathlib `Fin` sums.
2. **Permutations.** `act (σ : Equiv.Perm (Fin m)) (τ : Equiv.Perm (Fin n)) (A : Mat m n) : Mat m n := fun i j => A (σ i) (τ j)`; `weight_act`, `rowSum_act` (`rowSum (act σ τ A) = rowSum A ∘ σ`), and the one hard lemma `hasKst_act_iff`: an injective `s`-tuple can be re-indexed increasingly — via `Finset.orderEmbOfFin` on its image, the same device as `hasKst_of_subsets` (already proved). ~120 lines.
3. **Sorting.** Mathlib's `Tuple.sort` gives `σ` with `Monotone (f ∘ σ)`; apply to `n − r_i` to get non-increasing row sums; likewise columns. `exists_sorted : ∀ A, ∃ σ τ, Antitone (rowSum (act σ τ A)) ∧ Antitone (colSum (act σ τ A))`. ~40 lines.
4. **Enumerator with completeness.** `genParts (len cap total) : List (List Nat)` enumerates non-increasing lists with `sum = total`, entries `≤ cap`, in Tan's order, with prefix pruning by the *proved* budget (`Σ C(x_i, k) ≤ B`, monotone in the prefix, so the pruned prefix cannot extend to an admissible list) and by `ofPrefix` bounds when hypotheses are supplied. Theorem `mem_genParts : Antitone r → sum r = total → (∀ i, r i ≤ cap) → Σ C(r_i,k) ≤ B → (∀ k' < len, prefix bound) → r ∈ genParts …` by induction on `len` (~150–250 lines). Anything not enumerated is killed by `argA`/`argAT`/`ofPrefix`, all proved.
5. **Closure theorem.** `upper_bound_of_sorted_cover (P) (p : Prune P) (surv : List (List Nat × List Nat)) (cover : (genRows P).all (fun r => (genCols P).all (fun c => p.kill (mkProfile r c) || (r,c) ∈ surv)) = true) (refuted : ∀ q ∈ surv, ∀ A, sortedProfileOf A = q → ¬ Valid P A) : ∀ A, ¬HasKst P A → weight A < P.w` — assembled from 1–4 and `p.sound`. `cover` is discharged by `decide` (kernel) on the product `genRows × genCols`, whose size is exactly the table's pair count (§3.4: 244 … 4,818 for the small-`m` targets). Kernel `decide` on ~10^3–10^4 evaluations of `kill` (each a few hundred `Nat.choose` reductions with GMP-backed arithmetic) is expected to take seconds to minutes; measured on `(9,9,50)` first (M1), and if a target's cover exceeds a 30-minute kernel budget the fallback is `decide +native` in the harness domain with its one named axiom listed in the ledger — the same trust policy as the LRAT seam, never available to LLM code.

Symmetry note: block double-lex inside a case is *incomplete* (KNW Thm 2; two isomorphic survivors in Tan's Fig. 2 [LR §6.4]) — that costs solver time, not soundness, and is why case-level prunes matter.

### 4.7 The `refuted` seam: certificates

Per survivor `q`: `encode_case` → DIMACS (hashed) → `cadical --binary=false … proof.drat` (exit 20) → `drat-trim cnf drat -L lrat` (`s VERIFIED`) → `lrat-check cnf lrat` (`c VERIFIED`, independent checker) → manifest entry `{rows, cols, cnf_sha1, lrat_file, lrat_bytes, solve_s, check_s}` (`zar_ub/certify.py`, E9: 36/36 at `(9,9,50)`, 1.8 s, 3.8 MB). Tier-1 closure: `Hrefuted` hypothesis + manifest. Tier-0 (M7): `Encode.lean` + Lean-side `LRAT.check` on the parsed certificate (binary LRAT, `--no-factor` for CaDiCaL ≥ 3; 140–150 CPU-s per GB and ~2 GB + 0.28 GB/GB memory [LR §6.2]) with one `_native` axiom per branch whose name the harness records and re-matches; `bv_decide` cross-check on toy instances.

Anything a solver reports as SAT is decoded and re-checked with `has_kst` (an independent Python check); a SAT case at `w = z+1` means the encoding or the enumeration is wrong and halts the pipeline (E1's protocol).

---

## 5. Reward function

### 5.1 Quantities (all precomputed except the candidate's masks)

For each instance `I` in `TRAIN ∪ TARGET ∪ GEN` (§8):

* `S_I` — survivors of the proved baseline `B` (= generator prunes ∪ `ZarPrune.counting` ∪ conditional neighbour prunes with ledgered hypotheses), evaluated by Lean once when the table is built and cached in the table file (`baseline_lean_mask`).
* `d_I(q) ≥ 1` — difficulty of `q ∈ S_I` (§6); `W_I = Σ_{q∈S_I} d_I(q)`; `T_I` = the top decile of `S_I` by `d_I`.
* `K^L_I` — the candidate's Lean mask on `S_I` (only at L5); `K^P_I` — its Python mask.
* Battery `BAT`: for each battery instance, the set of cases with a SAT witness (`probe.status == "sat"`, matrix re-verified). Also the record profiles of [bhan26] (`(11,21)=116`: rows `11^6 10^5`, cols `6^11 5^10`; `(12,22)=132`: rows `11^12`, cols `6^22`) and every witness from `data/` are in the battery as realizable profiles [LR §8.2].

```
gain_I(K)  = Σ_{q ∈ S_I, K(q)} d_I(q) / W_I                     ∈ [0,1]
tail_I(K)  = Σ_{q ∈ T_I, K(q)} d_I(q) / Σ_{q ∈ T_I} d_I(q)      ∈ [0,1]
G_train    = mean_{I ∈ TRAIN}  gain_I(K^L)
G_target   = mean_{I ∈ TARGET} gain_I(K^L)      (:= G_train if TARGET = ∅)
G_gen      = mean_{I ∈ GEN}    gain_I(K^L)      (:= G_train if GEN = ∅)
Tail       = mean_{I ∈ TRAIN ∪ TARGET} tail_I(K^L)
E          = mean_{I ∈ TRAIN}  gain_I(K^P)      (empirical; Python mask; only if battery-sound)
```

### 5.2 The score

```
hard_zero  = battery violated by K^P or K^L on any witnessed case
           ∨ forbidden construct ∨ ladder = L4 ∨ Python error/timeout ∨ Lean mask missing at L5
           ∨ PIPELINE_BUG (a Lean-verified kill hits a witnessed case: halt the run, not just the candidate)

combined_score =
    0                                                                  if hard_zero
    0.20 + 0.80 · (0.40·G_train + 0.30·G_target + 0.10·G_gen + 0.20·Tail)   if ladder = L5
    min(0.20, 0.10·lean_partial + 0.10·E)                              if ladder ∈ {L0,L1,L2,L3}

if ladder = L5 and K^L ≠ K^P on any scored case:  combined_score *= 0.9   (mirror is wrong; Lean mask stands)
```

Properties: (i) a verified candidate that kills nothing scores exactly 0.20, above any unverified candidate — the "never let an unverified candidate outrank a verified one" rule [LR §7.2]; (ii) the baseline program scores 0.20 (all its prunes are in `B`), so every positive increment is a new proved kill; (iii) stationary: no running-record term (the lower-bound reward's non-stationary cliff [LR §8.2] is avoided); novelty pressure is provided by MAP-Elites and the prompt, not the score; (iv) the tail term rewards killing hard cases (E10: the top 10 % of cases carry 61.6 % of the work) beyond their share in `G`.

### 5.3 Metrics returned (raw, continuous — OpenEvolve bins them)

`combined_score, sound_battery, lean_ladder (0–5), lean_partial, proven_gain (=G_train), target_gain, gen_gain, tail_gain, empirical_gain, agreement, hard_killed (count of censored-unknown survivors killed with proof), survivors_left, kill_novelty, n_lean_decls, n_new_prunes, eval_seconds`.

`kill_novelty = 1 − |K^L ∩ U| / |K^L ∪ U|` where `U` is the union of kills of all prunes accepted so far in the run (the *library ledger*, `cache/ledger/accepted_prunes.jsonl`: Lean source hash, name, per-instance kill signature, first-seen iteration). Jaccard ≥ 0.95 with an existing accepted prune is reported in the artifacts as "duplicates prune X" [LR §8.3 A6].

### 5.4 MAP-Elites feature dimensions

`feature_dimensions: ["proven_gain", "lean_ladder", "kill_novelty"]`, `feature_bins: {"proven_gain": 10, "lean_ladder": 6, "kill_novelty": 5}`. Rationale: the first axis is the objective's main term (so the archive keeps a gradient of proved gain), the second is the review's Tier-A recommendation (status ladder as an axis, never the main reward), the third keeps distinct *arguments* alive even when their gain is similar. `n_lean_decls` (the current second axis) is dropped: it rewards length [LR §8.3 anti-proposals].

### 5.5 Anti-cheating inventory

| attack | defence |
|---|---|
| Python `kill` lies (kills more than Lean) | credit and pruning use `K^L` only; disagreement penalised and shown |
| `sorry`, `native_decide`, `axiom`, `implemented_by`, `csimp`, `unsafe`, `partial`, `opaque` | scan + `#print axioms` (two independent layers, E4) |
| redefine `Valid`/`HasKst`/`Params` in the sandbox | token scan after `def`; harness wrapper uses full names; `autoImplicit false` |
| kill everything / kill the hard cases by name | `sound` must elaborate — impossible for a non-empty case; battery catches the Python side early |
| non-terminating or exponential `kill` | `partial` forbidden (termination proved); per-instance `#eval` time cap → L1 with the message "kill too slow" |
| tamper with tables/certificates from candidate code | candidate runs in a copy; tables loaded before execution; Lean mask computed by the harness |
| inflate difficulty | `d` is a function of the branch CNF only, precomputed, never re-estimated during a run |
| target cells: earn cap-credit for censored cases | allowed — the kill is Lean-proved; `hard_killed` is reported separately so it can be audited |
| re-list library prunes | zero gain (they are in `B`) |
| instance-specific `candidate : Prune target` | allowed; scored only where it type-checks; `G_gen` and prompt discourage it |
| LLM-judge scores | none in the reward [LR §8.3] |

### 5.6 Cascade (`evaluator.cascade_evaluation: true`)

`evaluate_stage1`: Python mirror on all cases + battery + scan + vacuity (kills nothing on train → still passes, but flagged) → threshold 0.0 (only hard zeros stop). `evaluate_stage2`: Lean gate on TRAIN + GEN (one process) → `combined_score` without target term → threshold 0.15. `evaluate_stage3`: Lean masks on TARGET tables (thousands of cases; the only stage that can take > 5 s) and the full score. Artifacts at every stage: `per_instance` (hardest survivors alive with their profiles and difficulty), `lean_errors` (first 6 errors with line, message, goal), `lean_holes` (unfilled `have` statements), `UNSOUND_counterexamples`, `python_lean_disagreements`, `library_ledger` (names of accepted prunes and what they kill), `duplicates`.

---

## 6. Branch difficulty measure

### 6.1 Definition

`d(q)` = the number of CaDiCaL 1.9.5 conflicts needed to refute `encode_case(P, q)` (deterministic given the solver build and seed; conflicts and propagations are statistically indistinguishable as workload measures and both beat wall-clock [LR §5.2]).

### 6.2 Algorithm (`zar_ub/difficulty.py`, extending `probe_case`)

```
label(q, mode):
  1. cnf = encode_case(P, q); record nvars, nclauses, log2_volume = Σ_i log2 C(n, r_i)
  2. probe c2000 : solve_limited with conf_budget = 2000            (~0.05 s)
     if status ∈ {sat, unsat}: d = conflicts (exact); done
  3. if mode == "exact"  (TRAIN/BATTERY tables):
       escalate conf_budget 20 000 → 200 000 → 5 000 000, time_limit 240 s   (E10 protocol)
       d = exact conflicts; if still unknown: d = last cap, flag "unknown"
     if mode == "censored" (TARGET tables):
       one more run at cap C = 20 000
       if solved: d = conflicts
       else:      d = max(C, exp(a + b·log(c2000)))  with (a,b) fitted by least squares in log-space on the
                  TRAIN cases that were unknown at 2000 but solved exactly; flag "censored"
  4. store {status, conflicts, budget_cap, c2000, d, flags, nvars, nclauses, log2_volume, seconds}
```

Evidence for the design: on the 1,571 fully-labelled training cases, `c2000` has Spearman ρ = 0.913 with the true count (0.137 at 200; log-volume 0.617; clause count 0.566) **[E10]**; the distribution is heavy-tailed (median 1,652, p90 20,315, max 221,874; top decile = 61.6 % of work) **[E10]**. The calibration `(a,b)` and its RMSE in log units are reported in the LOG (expect the literature's factor-3 ceiling [LR §5.1]).

### 6.3 Cost

Per case: 2k-conflict probe ≤ 0.05 s at `(9,9)`–`(12,12)`; ≈ 0.1–0.3 s at `(12,18)`/`(13,19)` sizes (C(12,3)·18 ≈ 4k / C(13,3)·19 ≈ 5.4k auxiliaries); the 20k cap ≈ 0.35–2 s. Table-building cost for the §3.4 targets is therefore 5–20 CPU-minutes each (censored mode), once; exact labelling of the training ladder took 0.4–94 s per cell **[E10]**. Building is embarrassingly parallel (`multiprocessing.Pool`, one solver per process).

### 6.4 Large tables: shared-sample estimation

If `|S_I| > 3,000`, the reward for instance `I` is computed on a fixed uniformly random sample `Σ_I ⊂ S_I` of `N = 500` survivors drawn once with the table's seed (common random numbers, Chivilikhin's estimator one level up: `W_I ≈ |S_I|·mean_{Σ_I} d` [LR §5.2]); every candidate is evaluated on the same sample, so comparisons are paired. Every kill outside the sample is still recorded for closure decisions (§7).

### 6.5 What difficulty is *not* used for

A proxy weights a kill; it never justifies one [LR §3.3 (iv)]. It also never enters the cover or the certificate.

---

## 7. SAT execution policy

### 7.1 During evolution: never solve

All solver work in the loop is table construction, done once per instance and cached (`cache/case_table_*.json`). An evaluation touches no solver. This answers the proposal's "how often do we run the SAT solver versus proxies" with: proxies always, solver offline on schedule.

### 7.2 Closure attempts (offline; `python -m zar_ub certify … --lean <accepted library>`)

Trigger for a TARGET instance when any of: (a) the Lean-verified library's survivor count falls below `closure.max_cases` (default 500), (b) the censored estimate of remaining work `Σ_{alive} d̂` falls below `closure.max_conflicts` (default 2·10^8, ≈ 1 CPU-hour at ~50k conflicts/s), (c) a milestone review. Also every time an *unconditional* neighbour cell closes, re-generate the tables of `(m+1,n)` and `(m,n+1)` with the new bound as a ledgered hypothesis (the mechanism by which every source in the review extends its table [LR §10 item 3]).

### 7.3 Portfolio and escalation per survivor

```
stage 0  pysat cadical195 with conf_budget = 20 000                    (if solved SAT → pipeline halt; UNSAT → proceed to certified run)
stage 1  tools/cadical (DRAT) time_limit 600 s                           → drat-trim → lrat-check
stage 2  kissat (DRAT only, drat-trim -L) time_limit 3 600 s, in parallel with cadical --unsat -P2 (dfield's setting)
stage 3  cube-and-conquer inside the case: split on the support of the heaviest row (C(n, r_1) cubes, or the first
         k cells of row 1 as Tan did for z_2(23)); each cube refuted separately; the cube clauses ¬cube_i are RUP
         from each sub-certificate and the cover of a complete assignment tree of explicit variables is resolution-
         derivable, so the concatenation is one LRAT for the case (no new trust)
stage 4  report the case as OPEN in the closure report; the bound is not claimed
```

Concurrency: `min(cores − 1, 8)` cases at a time (the M-series Mac; each CaDiCaL process ≤ 1 GB); `parallel_evaluations` for the evolutionary loop is set to 2–4 because each evaluation spawns a Lean process (~1 GB with Mathlib oleans mapped).

### 7.4 Certificates and storage

DRAT deleted after LRAT is produced; LRAT kept (`cache/certs/<tag>/`), manifest with hashes; expected sizes from E9 (≈ 100 KB per easy case) up to dfield's regime (GB per hard profile); Tier-0 import cost 140–150 CPU-s per GB [LR §6.2]. The closure report (`zar_ub/closure.py`) lists every case's disposition, the trusted base, the axiom ledger of the prune, and external facts used; it is regenerated together with the Lean closure file.

---

## 8. Generalization across (m, n, s, t)

### 8.1 Parametric prunes are the default

`candidate (P : Params) : Prune P` is elaborated once and instantiated on every suite instance by the gate; a prune proved over symbolic `P` transfers for free to every `(m,n,s,t,w)`. The library's own prunes are all parametric (`Counting.lean`).

### 8.2 The suite has three roles plus a generalisation set

* `TRAIN` — exact `(3,3)` cells at `w = z+1` (§3.4).
* `BATTERY` — the same cells at `w = z` (SAT witnesses) plus known extremal profiles.
* `TARGET` — the small-`m` ladder of §3.4 (`ZAR_UB_TARGETS`), censored difficulty.
* `GEN` — a small held-out set with different `(s,t)`: `(2,2)` cells `(7,7,22)`, `(8,8,25)` (Tan's `z_2` table; also a `bv_decide` cross-check regime) and `(4,4)` cells such as `(9,9,w)` where Argument A already nearly closes the cell — scored as `G_gen` at weight 0.10. A prune that only works for `s = t = 3` (e.g. residues) is legitimate; `G_gen` merely rewards the ones that do not.

### 8.3 Zero-shot transfer matrix (reported, not rewarded)

After a run, the accepted library is evaluated on every cached table it was not trained on (`experiments/transfer.py`): kills and gains per cell, which prunes fire where. This is the proposal's "if and when our work generalises" question answered with data.

### 8.4 Propagation and provenance

`data/ledger.csv` (new): rows `(m, n, s, t, bound, kind ∈ {exact, ub}, provenance ∈ {lean-here, tan2022, collins16, bhan26, unreviewed-2026}, closure_file, hypotheses)`. Generator and conditional prunes read only from this ledger with a provenance filter (`--trust tan2022` default; `--pure` = nothing external). Every closure theorem states the facts it used as hypotheses (T7), so a bound "conditional on Tan" and a bound "conditional on an unreviewed preprint" are distinguishable at the type level. Later phases may enrich the case index (max pair codegree, size histogram) to turn `D_v` cuts into profile prunes [LR §10 item 5]; that changes `Profile` and is out of scope for v1.

---

## 9. Cost and time model, and model choice

### 9.1 Lean (measured)

| operation | wall | source |
|---|---|---|
| `lake build` of ZarPrune (cached oleans) | 0.9 s | [measured today] |
| single-instance gate incl. mask (Mathlib-backed `import ZarPrune`) | 1.7–2.3 s | [measured today] |
| whole-suite gate (5 instances, up to 625 cases each) in one process | 1.5–4.4 s per evaluation | [E6 addendum; measured today: 4.4 s total evaluation] |
| Mathlib-free candidate (if `Counting` were split out) | 0.3–0.5 s | [E4] |
| `leanchecker` replay of ZarPrune | 0.66 s (29 s `--fresh`) | [LR §6.1] |

Fit in the loop: with `parallel_evaluations = 2–4` the evaluator sustains ≈ 1 candidate/s, two orders of magnitude above the LLM's rate. Optimisation if ever needed: keep a warm Lean REPL with `ZarPrune` imported (saves the ~1 s import), and skip the gate on content-hash hits.

### 9.2 SAT (from cached tables and E10)

Training tables: 0.4–94 s to exact labels per cell. Target tables: 5–20 CPU-min each (§6.3). Closure of `(9,9,50)`: 1.8 s for 36 certificates **[E9]**. Unknowns: closure of `(9,23,104)`'s 244 cases and `(12,18,109)`'s 2,562 cases — the 16.5-minute non-finish of a *single* residual at `(12,18)` [LR §2.3] says hours to days without new prunes; that is the quantity the thesis measures (survivors × work before vs after evolved prunes).

### 9.3 LLM tokens and dollars

Prompt: system ≈ 1.1 k tokens, program ≈ 1.5–4 k, artifacts ≤ 16 KB ≈ 4 k, history 1–2 k → ≈ 8–11 k input; output 2–6 k (Lean + Python + notes). Per-iteration cost at the prices recorded on 2026-09-21 [LR §7.2] (recheck before a run):

| model | $/M in / out | ≈ $/iteration (10k in, 4k out) | role |
|---|---|---|---|
| deepseek/deepseek-v4-flash | 0.06 / 0.11 | 0.001 | filler, smoke runs, repair loop |
| openai/gpt-oss-120b | 0.15 / 0.60 | 0.004 | cheap ensemble member |
| openai/gpt-5.4-mini | 0.75 / 4.5 | 0.026 | mid |
| google/gemini-3.1-pro | 2 / 12 | 0.07 | sketcher (Hilbert/Goedel-Architect evidence: the reasoner is the lever) |
| anthropic/claude-opus-4.7 | 5 / 25 | 0.15 | sketcher, low weight |

Time per iteration is LLM-bound: 30–120 s generation + 2–5 s evaluation (+ ≤ 10 s repair loop when enabled).

Budget plan: **$16 = probes only** (§10.2; ≤ $13 spent, $3 reserve; `experiments/cost.py` logs the key's usage before and after every run). A real run: ensemble 0.5 flash / 0.3 mid / 0.2 reasoner, 2,000 iterations ≈ $0.02–0.05 per iteration on average ≈ $40–100 per run; three runs plus repairs ≈ $150–400, on AlphaEvolve Cloud or a larger OpenRouter budget. The lower-bound paper's $15–30 per case [LR §8.2] is the comparable.

### 9.4 Model choice policy

Reasoner writes sketches (Option B) with `reasoning_effort: high` and `max_tokens ≥ 12000`; cheap models fill and repair; ensemble weights in `config.yaml`'s `llm.models`. Escalate (re-prompt the reasoner for a re-decomposition, not a retry) only when the fill loop fails twice — the pattern every 2026 pipeline converged on [LR §7.1 Q5]. Specialised provers are not reachable via API and were not trained on Lean 4.34 [LR §7.2], so they are not in the plan.

---

## 10. Test plan

### 10.1 NO-LLM mode (runs today; must stay green)

`ZAR_UB_NO_LLM=1` disables the repair loop and any OpenRouter call; the evolutionary loop is exercised with a **replay LLM**: `zar_ub/replay_llm.py` implements the `init_client` hook of `LLMModelConfig` and returns canned responses (full rewrites or SEARCH/REPLACE diffs) from a directory, in order, so OpenEvolve's controller, database, MAP-Elites, artifacts and cascade run end-to-end at $0.

| id | test | expected |
|---|---|---|
| T-1 | `python -m unittest discover tests` (engine: waterfill vs brute force, partitions vs brute force, encoding ⇔ matrix existence on tiny cases, Argument D) | pass |
| T-2 | gate regression: mask on a 725-case table | length matches (fix of §4.2) |
| T-3 | **candidate ladder** (`tests/candidates/`): baseline; `sorry`; `native_decide`; `axiom` smuggled in a block comment; `implemented_by`; redefinition of `Valid`; symmetry-break "rows sorted" (`notDescending`); "kill rows ≥ 7"; Python-lies (Python kills, Lean doesn't); Lean-only kill (Python doesn't); vacuous verified; instance-specific verified; sketch with 2 holes (one `omega`-fillable) | ladder/score exactly as §4.3–§5.2 (table of expected values checked in); PIPELINE_BUG never fires |
| T-4 | **first new prune by hand** — DGH constraint (4), `v = s−1` (P8) proved in Lean as a candidate | L5; kills reproduce the Python census (`docs/lit/dgh2024_lp_bounds_code/`); positive `G_train` on `(11,11)`–`(12,12)` and `G_target` on `(13,19,123)` |
| T-5 | Farkas schema with DGH duals for `(15,17,133)` | refutes the whole cell with zero survivors [LR §3.3 P11 example] |
| T-6 | replay run: 30 canned iterations (mix of the ladder) through OpenEvolve with cascade on | database populated, features binned, artifacts rendered, checkpoints resume, no exception |
| T-7 | closure end-to-end Tier-1: `(9,9,50)`, `(9,10,55)`, `(10,10,61)` → Lean closure file compiles, `leanchecker` passes, report lists 0 open cases; timing of the kernel `decide` cover recorded | theorems `z(9,9) ≤ 49` etc. |
| T-8 | SAT-free Lean closures conditional on Tan: `(10,21,107)`, `(11,19,107)`, `(11,20,112)` | theorems with `Htan_*` hypotheses |
| T-9 | difficulty calibration: refit `(a,b)` on E10 data; report RMSE (log) and ρ on a held-out cell (`(12,13,87)`) | ρ ≥ 0.85 |
| T-10 | transfer matrix of the hand library across all cached tables | report |

### 10.2 With the $16 (E7 continued; every call logged in `experiments/cost_ledger.md`)

| id | what | calls | est. cost |
|---|---|---|---|
| P-1 | `probe_one.py`: 3 models (flash, mid, reasoner) × {direct, sketch} × 2 parents (initial; initial + hand P8 in library) = 12 calls; record parse rate, ladder, holes filled, tokens, seconds; apply and score each child | 12 | ≈ $1.5 |
| P-2 | repair loop on the P-1 sketches that reached L2/L3: ≤ 2 flash rounds each | ≤ 20 | ≈ $0.10 |
| P-3 | smoke run: 20 iterations, flash only, population 40, 2 islands (`config.yaml`) | 20 | ≈ $0.05 |
| P-4 | smoke run: 15 iterations, ensemble 0.6 flash / 0.4 mid with the winning prompt of P-1 | 15 | ≈ $0.5 |
| P-5 | one reasoner sketch per hardest target artifact (`(9,23,104)` and `(12,18,109)` survivor lists) to see whether frontier models propose residue/overlap-type arguments unprompted | 4 | ≈ $0.5 |
| reserve | | | $3 |

Success criteria for the probes (not for bounds): ≥ 1 L5 candidate with `G_train > 0` from any model; measured verified-prune rate per dollar per model; a decision on A vs B; a token/time table for §9.3. Every probe's prompt, response, child program and metrics are stored under `experiments/E7_llm_smoke/probes/` and summarised in `LOG.md` (E11+).

---

## 11. Risks and failure modes

| risk | likelihood / impact | mitigation |
|---|---|---|
| LLM verified-prune rate ≈ 0 on Lean 4.34 without Mathlib-trained provers | high / high for the search, none for soundness | schemas (§2.3: the LLM supplies data, not proofs), sketch-and-fill with auto-fill and repair, lemma pool in artifacts, hand-proved ladder of examples in the prompt; the thesis still delivers the verified pipeline and hand-proved closures |
| gate fragility (mask truncation found today; option changes on Mathlib import; Mathlib v4.34.0 cache missing `Mathlib.olean`) | medium / high (silent zero credit) | string mask; regression test T-2; pinned targeted imports only; `lean_ok < 1` on the baseline program aborts a run |
| kernel `decide` on the cover too slow for larger targets | medium / medium | measure on the ladder first (T-7); fallback `decide +native` in the harness domain with a named axiom in the ledger; never in LLM code |
| `Encode.lean` / `exists_doubleLex` (Tier-0) cost more than the thesis timeline | high / low | Tier-1 is the committed deliverable; Tier-0 is M7 |
| unreviewed 2026 neighbour bounds contaminating closures | low with the ledger / catastrophic without | provenance filter default `tan2022`; hypotheses in the theorem statement; `--pure` mode |
| encoding or enumeration bug (a claimed bound would be false) | low / catastrophic | independent witness re-check on every SAT model (E1), battery at `w = z`, PIPELINE_BUG halt, exhaustive tiny-instance tests, and, in Tier-0, the completeness theorem |
| symmetry-breaking incompleteness makes cases slow | certain / medium | cost only; case-level prunes are the answer; stage-3 cubing |
| the reward drives instance-specific hacks or dominated inequalities | medium / low | `G_gen`, novelty feature, baseline = full proved library, duplicates reported |
| reward non-stationarity / plateau at 0.20 | medium / medium | stationary score; MAP-Elites axes; artifacts name the hardest alive profiles with numbers |
| OpenEvolve diff application failures on long Lean strings | medium / low | `diff_based_evolution: true` with `allow_full_rewrites` for stuck islands; probe P-1 measures the rate |
| memory: Lean + Mathlib oleans per parallel evaluation | medium / low | `parallel_evaluations ≤ 4`; `ulimit -v` |
| cost overrun of the $16 | low | ledger before/after every call; hard stop at $13 |
| LRAT sizes / Lean import memory at Tier-0 | medium / medium | keep DRAT-trimmed LRAT only; stream per branch; report sizes; cubing to bound the largest leaf |

---

## 12. Milestones

| # | milestone | deliverable | done when |
|---|---|---|---|
| M0 (done) | engine, gate, certificates, training tables, difficulty labels, counting library | E1–E10 | — |
| M1 | gate fix + `Closure.lean` Tier-1 | string kill mask; thinning/permutation/sorting/enumerator theorems; `upper_bound_of_sorted_cover`; Lean closure files for `(9,9)`, `(9,10)`, `(10,10)` with `leanchecker` | T-2, T-7 green; kernel-cover timing logged |
| M2 | SAT-free conditional closures | `z(10,21) ≤ 106`, `z(11,19) ≤ 106`, `z(11,20) ≤ 111` as Lean theorems with Tan hypotheses; `ofPrefix`/`argDelCol` with ledgered `U` | T-8 green; entries in `data/ledger.csv` |
| M3 | evaluator v2 | reward §5, ladder §4.3–4.4, difficulty §6 with calibration, suite with TARGET/GEN, artifacts, cascade, replay LLM, candidate-ladder tests | T-3, T-6, T-9 green |
| M4 | schemas + first hand prunes | `Schemas.lean` (Farkas, prefix; residue if time), P8 (DGH) proved | T-4, T-5 green; transfer matrix T-10 |
| M5 | $16 probes | A/B decision, model table, LOG E11–E15, cost ledger | P-1…P-5 done |
| M6 | first SAT target | `(9,23,104)`: table (244 cases, censored labels), closure attempt with the hand library; `(10,22,111)` (3 cases) | closure reports; open cases enumerated |
| M7 | Tier-0 seam | `Encode.lean`, `exists_doubleLex`, Lean-side `LRAT.check` on `(9,9,50)`; axiom ledger | `z(9,9) ≤ 49` with only `propext/Quot.sound/Classical.choice` + named native axioms |
| M8 | real evolutionary run (cloud / larger budget) | 2–3 runs of ≥ 1,000 iterations; accepted library; per-phase cost; ablations (no-schemas, no-sketch, no-tail) | verified prunes with `G_target > 0`; survivor/work reduction on `(12,18,109)` and `(9,23,104)` measured |
| M9 | thesis results | closure theorems for whichever of `(9,23)`, `(12,17)`, `(12,18)`, `(10,23)` closes; transfer matrix; axiom ledgers; comparison to dfield/Afrasyab/Hou | write-up |

Each milestone appends to `experiments/LOG.md` with the E-number protocol already in use (question, exact command, table, conclusion).

---

## Appendix A — Files to create or change

| path | change |
|---|---|
| `zar_ub/lean_gate.py` | string mask emission and parsing; ladder fields; sketch-mode scan; auto-fill; content-hash cache; `autoImplicit false`; redefinition scan |
| `zar_ub/reward.py` (new) | §5 formulas; ledger of accepted prunes; novelty |
| `zar_ub/difficulty.py` | censored mode, calibration, shared sample |
| `zar_ub/casetable.py` | store `baseline_lean_mask`, `d`, flags; TARGET/GEN kinds |
| `suite.py` | TARGET ladder of §3.4, GEN cells, provenance filter |
| `evaluator.py` | three cascade stages; new metrics/artifacts; optional repair loop; NO-LLM switch |
| `zar_ub/replay_llm.py` (new) | `init_client` replay for OpenEvolve |
| `zar_ub/closure.py` | emit the Lean closure file (§1.1) + report + ledger row |
| `lean/ZarPrune/Closure.lean` (new) | §4.6 |
| `lean/ZarPrune/Schemas.lean` (new) | §2.3 |
| `lean/ZarPrune/Encode.lean` (M7) | §4.1 T5′ |
| `data/ledger.csv` (new) | §8.4 |
| `config.yaml` | features `["proven_gain","lean_ladder","kill_novelty"]`, cascade on, `parallel_evaluations: 3`, ensemble per §9.4 |
| `initial_program.py` | §2.2 skeleton (baseline = `counting`) |
| `tests/candidates/*` | §10.1 T-3 ladder |

## Appendix B — Sources relied on

`docs/literature_review.md` §§1–10 (Tan 2022 method and Kyoto code, dfield closures, Davies–Gill–Horsley LP, Hou/Afrasyab/Saurabh 2026 tables, hardness estimation, verified LRAT checking and sandboxing, symmetry breaking, LLM+Lean provers, reward shaping, AlphaEvolve/FunSearch); `experiments/LOG.md` E1–E10; `lean/ZarPrune/{Sum,Basic,Prune,Prunes,Counting,Demo}.lean`, `lean/COUNTING_SPEC.md`; `zar_ub/*`; `openevolve/config.py`, `openevolve/evaluation_result.py`, `openevolve/evaluator.py` (cascade); `examples/zarankiewicz/zarankiewicz_10,21/` (lower-bound conventions); `zarankiewicz_generalized/gpt_agent/analysis/sat_attack/{encodings_zar.py,profiles.py}` (column-profile cubing, pair cuts).

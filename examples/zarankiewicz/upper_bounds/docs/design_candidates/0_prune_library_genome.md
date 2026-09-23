<!-- Candidate design 0 (angle: the genome is a prune library — NL arguments + Python kill + Lean Prune/CondPrune terms; evaluator = Lean gate + witness battery + difficulty-weighted marginal kill mass; SAT never in the loop; closure is a separate triggered job). Judges ranked it 2nd (scores 7 / 7). -->

# Design: evolved, Lean-verified prune libraries for Tan's case split on z(m,n;3,3)

Design document for the thesis system of `docs/proposal_section2.md`, targeted at the existing code under `examples/zarankiewicz/upper_bounds/` (engine `zar_ub/`, Lean library `lean/ZarPrune`, experiments E1–E10 in `experiments/LOG.md`) and the OpenEvolve framework at the repository root. Everything below is stated so that it can be implemented file by file; where the current code already does what is described, the file is named; where it differs, the change is stated as a diff in words.

Ground truth used while writing (2026-09-21, this machine):

* Lean 4.34.0; `lake build` of ZarPrune with the Mathlib v4.34.0 olean cache present (7.9 GB under `lean/.lake`) completes in 0.9 s; a candidate that folds `counting P` into a `Prune` passes the gate in **1.57 s** end to end (scan, elaboration, `#print axioms`, kill mask). The Mathlib-free core alone checks in 0.3–0.5 s (E4). The proposal's "does Lean take too long?" is settled: no.
* `ZarPrune/Counting.lean` is integrated and proves Arguments A (`argA`, `argAT`), D (`argD`, `argDT`, layer-cake form), transposition (`Prune.transposed`), deletion (`argDelCol`/`argDelRow`, parametrised by a proved bound `U`), waterfilling (`weight_le_waterfill`, `argWF`), and folds them as `counting P`. Axioms: `{propext, Quot.sound, Classical.choice}`.
* python-sat 1.9.dev15 (CaDiCaL 1.9.5, Glucose 4, …); `tools/cadical`, `tools/drat-trim`, `tools/lrat-check` built; E9 certified all 36 survivors of `(9,9,50)` in 1.8 s.
* Case tables (E3/E10): 7 training cells fully labelled with exact CaDiCaL conflict counts (1,571 cases; median 1,652, p90 20,315, max 221,874; top decile carries 61.6 % of the work).
* Case counts for target cells computed today with `zar_ub.partitions` (Argument A + prefix Argument I) and the reference baseline (A, I, D):

| cell | w | table mode: cases / after A,I,D | pure mode: cases / after A,I,D | note |
|---|---|---|---|---|
| (9,23) | 104 | 244 / 94 | 411 / 157 | dfield claims ≤103 (Lean+LRAT, unreviewed) |
| (10,21) | 107 | **0** / 0 | 185 / 54 | ≤106 follows from z(9,21)=96 by deletion (Argument I) |
| (10,22) | 111 | 3 / 3 | 220 / 30 | dfield ≤110 |
| (10,23) | 113 | 4,818 / 1,189 | 5,850 / 1,560 | dfield ≤112 (13 profiles needed SAT/MIP, 25 GB) |
| (11,19) | 107 | **0** / 0 | 2,970 / 828 | ≤106 by deletion from z(11,18)=101 |
| (11,20) | 112 | **0** / 0 | 198 / 110 | ≤111 by two deletions |
| (11,23) | 124 | 822 / 336 | 822 / 336 | dfield ≤123 |
| (12,17) | 104 | 968 / 617 | 52,264 / 31,968 | =103 already (Collins 2016) |
| (12,18) | 109 | 2,562 / 1,518 | 40,626 / 26,854 | Hou/Afrasyab =108 |
| (12,23) | 135 | 75 / 20 | 75 / 20 | dfield ≤134 |
| (13,19) | 123 | 4,130 / 2,648 | 72,693 / 30,289 | genuinely open, 118–122 |
| (16,17) | 134 | 2,139 / 993 | 711,018 / ? | genuinely open, 132–133 |

"Table mode" lets Argument I use `data/exact_33.csv` (Tan's 159 bold cells + (11,21)=116, (12,22)=132); "pure mode" uses only the provable counting bound.

---

## 1. Overview and claims

### 1.1 The one-sentence system

An OpenEvolve loop whose genome is a **prune library** — a Python file containing natural-language counting arguments, a Python `kill` predicate on (row-partition, column-partition) cases, and a Lean 4 fragment that packages the same predicate as ZarPrune `Prune`/`CondPrune` terms with a proof — whose evaluator (i) compiles the Lean and audits its axioms (hard gate), (ii) computes the kill mask **in Lean**, (iii) rejects any candidate whose Python `kill` removes a case with a known witness, and (iv) pays for the difficulty-weighted solver work that the *verified* prunes remove from the survivors of the already-verified baseline; while a separate, budget-triggered closure job refutes the remaining cases with checked LRAT certificates and writes a provenance-tagged bound ledger.

### 1.2 Pipeline

```
                      ┌────────────── OpenEvolve (islands, MAP-Elites) ──────────────┐
 initial_program.py ──► LLM edits: ARGUMENTS (NL) + LEAN_SOURCE + kill()               │
                        │                                                              │
                        ▼ evaluator.py (3–6 s, no SAT, no LLM)                         │
   suite tables ─────►  stage 1: run kill() in a sandbox on every case; witness battery; vacuity
   (cache/*.json,       stage 2: Lean gate (scan → elaborate → typecheck wrapper → #print axioms
    difficulty labels)           → nonce-tagged #eval kill mask); lean_status ladder
                        stage 3: kill mass vs verified baseline, tail statistic, generality
                        artifacts: hardest survivors alive, Lean errors + goals, disagreements
                      └────────────────────────────────────────────────────────────────┘
 promotion:  accepted prunes ──► lean/ZarPrune/Evolved.lean (leanchecker) ──► baseline for next run
 closure:    survivors of the verified library ──► round-robin CaDiCaL escalation ──► cadical DRAT
             ──► drat-trim → LRAT ──► lrat-check ──► closure_report.md + axiom/provenance ledger
             ──► (milestone) Lean: upper_bound_of_cover with LRAT.check_sound leaves
```

### 1.3 Claims the design makes (and how each is enforced)

* **C1 — No unsound prune can ever be credited or used.** A prune enters the trusted library only as a term of type `Prune P` (or `CondPrune P T`) elaborated by the Lean kernel in a harness-owned wrapper, with `#print axioms ⊆ {propext, Quot.sound, Classical.choice}`, after a source blacklist and a runtime `leanchecker` replay at promotion time. The kill mask that prunes cases is the one Lean computes, never the Python mirror (§4).
* **C2 — The reward is candidate-independent and difficulty-weighted.** Every case carries a precomputed refutation cost (§6); a candidate's score is a function of which precomputed cases its verified mask kills. Nothing the LLM writes can change a label (§5).
* **C3 — The system is generic in (m,n,s,t,w) and chains bounds.** Lean definitions, the Python engine, and the prune terms are parametric in `Params`; conditional prunes take a `BoundTable` whose numeric part is data and whose soundness is a separate proposition, so a cell closed by the pipeline strengthens its neighbours and any external fact stays an explicit hypothesis of the final theorem (§8).
* **C4 — A bound is claimed only through a closure report** listing, per case, "pruned (Lean term, axioms)" or "refuted (LRAT verified by lrat-check, file hash)", plus the trusted base (encoding, partition generator, external facts) (§7).
* **C5 — The loop is affordable.** Gate 1.6 s, evaluation 3–6 s, LLM $0.0015–0.22 per iteration depending on model; a 1,000-iteration run costs ≈ $30–80 and 4–5 h on 4 workers; the $16 budget is spent only on prompt/response probes and 5–25-iteration smoke runs (§9, §10).

### 1.4 What the design does not claim

No new closed-form counting theory (every profile prune is one double count plus convexity or residue arithmetic); no complete symmetry break for row–column symmetry (impossible unless GI ∈ coNP); no Lean-free replacement for certificates; and no formal verification, in v1, of the CNF encoding's completeness or of the partition generator's completeness — these are the documented trusted base with a milestone to shrink it (§4.6, §12).

---

## 2. Genome: exactly what the LLM edits

### 2.1 Why one Python file

OpenEvolve evolves one file between `EVOLVE-BLOCK` markers (diff mode or full rewrite). Keeping the Lean text inside a Python string (`LEAN_SOURCE`) gives one genome, one diff, one prompt, and lets the Python `kill` be run in a sandbox before Lean is touched. The evaluator writes `LEAN_SOURCE` to `lean/Candidates/cand_<sha>.lean` inside a harness-owned wrapper (§4.2). This is the layout `initial_program.py` and `run_candidate.py` already use; the changes are the structured `ARGUMENTS` list, the `ub` (neighbour-bound) argument to `kill`, and the `CondPrune` form.

### 2.2 The three coupled parts

1. **`ARGUMENTS`** — a list of dicts, one per prune: `name`, `status` (`proven` | `empirical` | `sketch`), `claim` (the inequality on the profile), `proof` (the natural-language double count / residue argument, 3–15 lines), `lean` (the Lean declaration name). It is shown back to the LLM (lemma pool, Seed/Aristotle style) and copied into artifacts; it is **never** scored by an LLM judge. Its only effect on the reward is indirect, through the Lean that accompanies it.
2. **`LEAN_SOURCE`** — a Lean fragment with no `import`, spliced into `namespace ZarPrune.Cand` after `import ZarPrune` (which brings the Mathlib-backed `Counting.lean` API). It must define
   ```lean
   def candidate (P : Params) (T : BoundTable) : CondPrune P T
   ```
   (or the simpler `def candidate (P : Params) : Prune P`, or an instance-specific `def candidate : Prune target`; the wrapper tries all three shapes, §4.2). Helper lemmas and further `def myPrune (P : Params) : Prune P where …` blocks are free-form.
3. **`kill(m, n, s, t, w, rows, cols, ub)`** — Python mirror of `candidate.kill`; `rows`/`cols` non-increasing tuples; `ub(m', n')` returns the best *proven* upper bound for the `(m', n'; s, t)` cell from the run's ledger or `None` — the Python image of `BoundTable.lookup`. Used for the witness battery, the empirical (unverified) signal, and the Python/Lean agreement check.

### 2.3 Concrete `initial_program.py` skeleton

```python
"""Prune library for Zarankiewicz upper bounds z(m,n;s,t)  --  evolved by OpenEvolve.

A CASE is a pair (rows, cols): non-increasing row sums (length m, entries <= n) and
column sums (length n, entries <= m) with sum(rows) = sum(cols) = w.  The pipeline
proves z(m,n;s,t) < w by showing every case is EMPTY: either a verified prune kills
it, or a SAT solver refutes it with a checked certificate.  You evolve the prunes.

THE CONTRACT.  kill(...) may return True only if NO K_{s,t}-free m x n 0/1 matrix has
exactly these row and column sums.  Symmetry breaking ("assume rows sorted", "the
transpose is equivalent") is NOT a prune: such a case is not empty, it is merely
redundant, and the SAT encoding already handles it.  Killing a realizable case is
detected by the witness battery and scores 0.  Only prunes whose Lean proof passes
the gate earn the main reward.

LEAN.  LEAN_SOURCE is spliced into `namespace ZarPrune.Cand` after `import ZarPrune`.
No `import`, no `sorry` outside `sound` proofs, no `axiom`, `native_decide`, `unsafe`,
`partial`, `implemented_by`, `extern`, `csimp`, `opaque`, `IO`, `#eval`/`#print`/any
`#` command, `set_option` other than maxHeartbeats/maxRecDepth (<= 400000 / 4096).
Available (Mathlib-backed, see lean/API.md):
  Params{m,n,s,t,w}  Mat m n := Fin m -> Fin n -> Bool  rowSum colSum weight
  HasKst P A  Valid P A := ¬HasKst P A ∧ P.w ≤ weight A  Profile{row,col}  profileOf
  Prune P := {name, kill : Profile P.m P.n -> Bool, sound : ∀ A, kill (profileOf A) = true -> ¬Valid P A}
  Prune.never / Prune.or / Prune.ofList / Prune.transposed / Prune.mono(w)
  BoundTable{lookup : Nat -> Nat -> Nat -> Nat -> Option Nat}, BoundTable.Sound,
  CondPrune P T := {name, kill, sound : T.Sound -> ∀ A, kill (profileOf A) = true -> ¬Valid P A}
  CondPrune.ofPrune / CondPrune.ofList / CondPrune.toPrune
  sumFin allFin sumFin_eq_sum sumFin_swap sumFin_le allFin_iff not_allFin_elim
  support rowSupport card_support card_rowSupport hasKst_of_subsets
  budget_general (the generic double count)  colBudget rowBudget rowLocalBudget
  argA argAT argD argDT argDelCol argDelRow argWF counting (baseline, already verified)
  deleteCol deleteRow weight_deleteCol weight_deleteRow not_hasKst_deleteCol/Row
  waterfillBound weight_le_waterfill choose_tangent
  Mathlib: Finset (powersetCard, filter, sum_comm, sum_le_sum, card_filter, exists_subset_card_eq),
  Nat.choose (choose_le_choose, choose_succ_succ, ...), omega, decide, simp, linarith, ring, Nat.ModEq
"""
from math import comb

# EVOLVE-BLOCK-START
ARGUMENTS = [
    {"name": "counting", "status": "proven", "lean": "ZarPrune.counting",
     "claim": "Arguments A and D in both orientations, deletion with the waterfilled counting bound.",
     "proof": "Double counting of (s-set of rows, column containing it): each s-set lies in "
              "at most t-1 columns, else a K_{s,t}; column j contains C(c_j, s) such sets. "
              "Argument D: the same count on the m-1 rows other than a fixed row i, restricted "
              "to the columns of i, gives sum_{j in i} C(c_j - 1, s-1) <= (t-1) C(m-1, s-1)."},
]

LEAN_SOURCE = r'''
/-- The evolved prune library.  Add `def myPrune (P : Params) : Prune P where ...`
    (or a `CondPrune P T` using `T.lookup` for neighbour bounds) above and list it here. -/
def candidate (P : Params) (T : BoundTable) : CondPrune P T :=
  CondPrune.ofList P T [CondPrune.ofPrune (counting P)]
'''


def _argA(m, n, s, t, rows, cols):
    return sum(comb(c, s) for c in cols) > (t - 1) * comb(m, s) or \
           sum(comb(r, t) for r in rows) > (s - 1) * comb(n, t)


def _argD(m, n, s, t, rows, cols):
    r = rows[0]
    if s >= 1 and sum(comb(c - 1, s - 1) for c in sorted(cols)[:r] if c >= 1) > (t - 1) * comb(m - 1, s - 1):
        return True
    c = cols[0]
    return t >= 1 and sum(comb(x - 1, t - 1) for x in sorted(rows)[:c] if x >= 1) > (s - 1) * comb(n - 1, t - 1)


def _deletion(m, n, s, t, rows, cols, ub):
    """Removing the lightest column leaves an (m, n-1) matrix, which obeys any proven bound."""
    total = sum(rows)
    u_col, u_row = ub(m, n - 1), ub(m - 1, n)
    return (u_col is not None and total - min(cols) > u_col) or \
           (u_row is not None and total - min(rows) > u_row)


def kill(m, n, s, t, w, rows, cols, ub=lambda m_, n_: None):
    """Python mirror of `candidate.kill`.  Must agree with Lean on every case."""
    total = sum(rows)
    if total < w or total != sum(cols) or any(r > n for r in rows) or any(c > m for c in cols):
        return True
    if _argA(m, n, s, t, rows, cols) or _argD(m, n, s, t, rows, cols):
        return True
    if _deletion(m, n, s, t, rows, cols, ub):
        return True
    return False
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    print(kill(9, 9, 3, 3, 50, (6, 6, 6, 6, 6, 5, 5, 5, 5), (6, 6, 6, 6, 6, 5, 5, 5, 5)))
```

Notes on the skeleton. `counting P` in `Counting.lean` folds `argA, argAT, argD, argDT, argDelColWF, argDelRowWF`; the Python mirror above reproduces exactly that baseline so that the initial program scores as "verified, marginal mass 0". The docstring is the entire Lean vocabulary the model is told about; `zar_ub.lean_api` regenerates `lean/API.md` and the docstring block must be regenerated from it whenever the library changes (the current `config.yaml` system message still says "Mathlib is NOT available" — stale since E5; it must be replaced by the list above).

### 2.4 The prompt (system message) — what changes from the current `config.yaml`

Keep the current framing (cases, prune vs. addition, reward description). Replace the API paragraph with the docstring list; add: (i) "the baseline `counting P` is already verified — you earn nothing for re-proving Arguments A/D; read the artifacts for the hardest surviving profiles and look for a *new* reason they are empty: DGH's `v = s−1` rounding inequality, cross-side arguments coupling `rows` and `cols`, deletion against `T.lookup`, residue/overlap arguments mod a small `g` on the exceptional columns, Farkas combinations of proven inequalities"; (ii) a worked example: the full text of `argA` (12 lines) so the model sees the `where kill := … sound := by …` idiom and `of_decide_eq_true`; (iii) the two-stage instruction (§4.7): "if you cannot finish `sound`, leave typed `have h : … := by sorry` holes: the gate reports which holes elaborate; a future iteration fills them"; (iv) keep every proven prune, add incrementally, fix Lean errors rather than deleting.

### 2.5 Genome levels (what is and is not evolved)

| level | evolved? | rationale |
|---|---|---|
| G1 kill + proof (this design's default) | yes | interpretable final artefact; the thesis's stated goal |
| G2 search mode (Python searches a parametric family of inequalities against the case bank; evaluator instantiates a fixed Lean template) | v2 option for the Farkas/residue families (§8.5) | Tao–Wagner's "one slow call, many cheap checks" leverage; needs a Lean template per family |
| G3 generalizer (score over several `(m,n)`, show few) | yes, through the suite (§5.1) | prunes parametric in `P` are the only ones that transfer |
| encoding, decomposition, difficulty labels, wrapper | never | trusted base (§3.4) |

---

## 3. Case decomposition and SAT encoding: fixed vs. evolvable

### 3.1 Decomposition (fixed; `zar_ub/partitions.py`, `zar_ub/cases.py`)

Instance `P = (m,n,s,t,w)`; question "does a `K_{s,t}`-free `m×n` 0/1 matrix with ≥ w ones exist?" Deleting ones preserves freeness, so ≥ w is equivalent to exactly w (P19 in the review), and the case index is the pair (row partition, column partition) of w: non-increasing tuples `rows ∈ [0,n]^m`, `cols ∈ [0,m]^n`, `Σ rows = Σ cols = w`. Generation is Tan's Algorithm 1 with the two prefix filters: Argument A (`Σ_j C(c_j,s) ≤ (t−1)C(m,s)`, and the row dual) and Argument I on *proper* prefixes only (`c_1+…+c_k ≤ UB(z(m,k))` for `k < n`; the E1 circularity fix). `UB` comes from the run's `BoundTable` (§8): counting bound only in pure mode; counting ∧ `data/exact_33.csv` in table mode. Every table-mode lookup that tightens a bound is recorded in the ledger (`Ledger.note`).

ZarPrune's `Profile` is the *ordered* refinement of a case; a prune on profiles restricts to a prune on partitions. The closure theorem's `cover` obligation therefore has to absorb the sorting (a valid matrix's profile is not sorted in general): v1 discharges cover in Python (trusted base); v2 proves `exists_sorted_conjugate` in Lean (§4.6, M8).

### 3.2 CNF encoding of one case (fixed; `zar_ub/encoding.py`)

Variables `x_{ij} = i·n + j + 1`. Constraints:

1. `K_{s,t}`-freeness, aux form: for each `s`-subset `R` of rows and each column `j`, `y_{R,j}` with clause `(¬x_{i₁j} ∨ … ∨ ¬x_{iₛj} ∨ y_{R,j})`, then `AtMost(t−1)` over `{y_{R,j}}_j` (sequential counter). One-directional `y` is sound and complete (forced true only when the column contains `R`). `C(m,s)·n` aux variables.
2. Exact row and column sums: `CardEnc.equals` sequential counters, `k(N−k)` auxiliaries per line.
3. Double-lex within equal-sum blocks: for adjacent rows with equal sums `row_i ≥_lex row_{i+1}` (prefix-equality auxiliaries), same for columns. Sound as an *addition* by Tan's Theorem 3.2 (block-wise double lex reaches a fixed point under the case stabiliser `Π S_{R_v} × Π S_{C_u}`); incomplete (KNW), which is fine.

Size at `(10,14)`: ≈ 6.5k variables; at `(13,19)`: `C(13,3)·19 = 5,434` aux + counters ≈ 12k variables, ≈ 40k clauses. Tests: `tests/test_engine.py::TestEncoding` brute-forces case-SAT ⇔ matrix-exists on tiny instances with and without lex.

### 3.3 Evolvable in the SAT layer — only in v2, and only through Lean

* **Verified cuts** (v2): a `Cut P` is a Lean lemma `∀ A, Valid P A → profileOf A = q → Φ(A)` where `Φ` is a linear constraint over cells (e.g. the pair cut `D_{s−1}`: for rows `r,r'`, `Σ_{j ∋ r,r'} (c_j − s + 1) ≤ (t−1)(m−s+1)`), encoded as an *implied* cardinality constraint (`pair_cuts` of `encodings_zar.py`). Added to the CNF of a case, it is a sound strengthening (it holds for every matrix in the case), so refutations remain valid; the certificate then proves UNSAT of `CNF ∧ cut`, and the closure report records the cut's Lean name. This is the only way evolved content touches the CNF.
* **Enriched case index** (v2): sub-cases keyed by the heaviest row's support pattern or the maximum pair-codegree `β` — "triples of backdoors beat pairs" (Chivilikhin) and the bridge to `D_v` prunes. Requires a new `cover` layer; not in v1.

### 3.4 The trusted base (v1), stated once

1. Lean 4.34 kernel + Mathlib v4.34.0 oleans as fetched by `lake exe cache get`.
2. `partitions.py`: completeness of the admissible-case enumeration (Argument A + proper-prefix Argument I, with the ledger's bounds).
3. `encoding.py`: case SAT ⇔ a `K_{s,t}`-free matrix with exactly these sums exists (Sinz counters; block double-lex; Tan Thm 3.2).
4. `drat-trim` and `lrat-check` binaries; CaDiCaL as a DRAT producer (its verdict is never trusted, only its proof).
5. External facts in `data/exact_33.csv` when table mode is used; every use is in the ledger and appears as a hypothesis of the final theorem (§8).

Items 2 and 3 are milestones M8/M9 (§12) to move into Lean; the closure report prints this list verbatim.

---

## 4. Verification pipeline

### 4.1 Stages and the `lean_status` ladder

| stage | what runs | failure → status |
|---|---|---|
| S0 static scan | forbidden-token blacklist on `LEAN_SOURCE` with comments stripped (§4.3); `set_option` whitelist; duplicate-name check against `ZarPrune` core names | **L4** (forbidden) → score 0 |
| S1 elaboration | `lake env lean` on the wrapper file, hard timeout 240 s, `-DmaxHeartbeats=400000`, memory ulimit 4 GB | parse error / error inside `kill` definitions → **L1** |
| S2 wrapper typecheck | `def gateInst : CondPrune target gateTable := by first \| exact candidate target gateTable \| exact CondPrune.ofPrune (candidate target) \| exact CondPrune.ofPrune candidate` | `kill` elaborates, `sound` has errors → **L2**; `sound` elaborates modulo `sorry` → **L3** |
| S3 axiom audit | `#print axioms ZarPrune.Cand.gateInst`; must be ⊆ `{propext, Quot.sound, Classical.choice}`; any `sorryAx`, `_native`, `ofReduceBool`, or unknown name | audit fail on a scan-clean file → **L4** (something was smuggled) |
| S4 kill mask | nonce-tagged `#eval` blocks over the suite's case lists (chunks of 200), parsed only between `MASK <nonce> k BEGIN/END`; mask length must equal the case count | missing/short mask, `#eval` timeout → **L2** (kill not computable) |
| S5 promotion (accepted candidates only) | `leanchecker` replay of the wrapper module (0.7 s); re-run of S0–S4 from a clean `Candidates/` directory; append to `lean/ZarPrune/Evolved.lean` | fail → quarantine, never promoted |
| — | all pass | **L5** verified |

`L0` = empty/unparseable `LEAN_SOURCE`. Only L5 is trusted; L1–L3 feed the shaping term (§5.3) and the MAP-Elites dimension.

### 4.2 The wrapper file (harness-owned; `zar_ub/lean_gate.py::build_gate_file`)

```lean
import ZarPrune
set_option autoImplicit false
namespace ZarPrune
namespace Cand
abbrev target : Params := { m := 13, n := 19, s := 3, t := 3, w := 123 }
def gateTable : BoundTable := ⟨fun m n s t => (gateEntries.lookup (m, n, s, t))⟩   -- generated from the ledger
-- ===== candidate (verbatim LEAN_SOURCE) =====
…
end Cand
end ZarPrune
-- ===== gate checks (generated; the candidate cannot reach these names) =====
def ZarPrune.Cand.gateInst : ZarPrune.CondPrune ZarPrune.Cand.target ZarPrune.Cand.gateTable := by
  first | exact ZarPrune.Cand.candidate ZarPrune.Cand.target ZarPrune.Cand.gateTable
        | exact ZarPrune.CondPrune.ofPrune (ZarPrune.Cand.candidate ZarPrune.Cand.target)
        | exact ZarPrune.CondPrune.ofPrune ZarPrune.Cand.candidate
#print axioms ZarPrune.Cand.gateInst
def ZarPrune.Cand.gateProfile (r c : List Nat) : ZarPrune.Profile ZarPrune.Cand.target.m ZarPrune.Cand.target.n :=
  { row := fun i => r.getD i.val 0, col := fun j => c.getD j.val 0 }
#eval IO.println "MASK 9f3a…(nonce) 0 BEGIN"
#eval ([([7,7,…],[6,6,…]), …] : List (List Nat × List Nat)).map fun p => ZarPrune.Cand.gateInst.kill (ZarPrune.Cand.gateProfile p.1 p.2)
#eval IO.println "MASK 9f3a… 0 END"
-- further instances: target1/gateInst1/… in the same file (one Lean process per suite, E6 addendum)
```

Why this shape: the statement `CondPrune target gateTable` is fixed by the harness (the UW "structural extraction" safeguard comes for free); `gateInst.kill` needs no proof of `gateTable.Sound`, so `#eval` runs even though the table's soundness is a hypothesis; the wrapper's declarations live outside `namespace Cand`, and `end Cand`/`end ZarPrune` are forbidden tokens, so a candidate cannot redefine them. The nonce (16 hex chars from `os.urandom`) is regenerated per evaluation and only lines between the exact markers are parsed, so a candidate cannot forge a mask by printing text. `Prune.mono` (4 lines, to add to `Prune.lean`): a `Prune ⟨m,n,s,t,w⟩` is a `Prune ⟨m,n,s,t,w'⟩` for `w' ≥ w`, so a prune proven for the smallest weight in a `bound` search serves every larger weight.

### 4.3 Forbidden constructs (extend `FORBIDDEN` in `lean_gate.py`)

Current list: `sorry` (see §4.7 for the sketch exception), `admit`, `native_decide`, `axiom`, `unsafe`, `implemented_by`, `extern`, `csimp`, `opaque`, `partial`, `import`, `macro`, `macro_rules`, `elab`, `syntax`, `notation`, `initialize`, `#exit`, `Lean.`, `IO`, `ofReduceBool`, `end ZarPrune`, `end Cand`, `set_option` other than `maxHeartbeats`/`maxRecDepth`.

Add: `+native` and `decide +native`/`+kernel` variants, `trustCompiler`, `run_cmd`, `run_tac`, `run_elab`, `open Lean`, `Lean.Elab`, `Lean.Meta`, `attribute [` (attribute edits on library lemmas), `@[simp]` on non-candidate names (allow on candidate-declared names), `local instance`/`instance : Decidable` (a hand-written `Decidable` instance cannot be wrong without an axiom, but it can make `decide` diverge; disallow to keep kills cheap), `dbg_trace`, `trace`, `logInfo`, `noncomputable` (kill must compute), every `#` command (`#eval`, `#print`, `#check`, `#reduce`, `#guard`, `#synth`, `#exit`), `deriving` handlers other than `Repr, DecidableEq`, `termination_by` with `decreasing_by sorry` (covered by the sorry rule), and the names `Valid`, `HasKst`, `Params`, `Mat`, `Prune`, `CondPrune`, `BoundTable`, `profileOf`, `weight`, `rowSum`, `colSum`, `counting` as *declared* names (regex `^(def|theorem|abbrev|structure|instance)\s+<name>\b`). `set_option maxHeartbeats` capped at 400000, `maxRecDepth` at 4096. Comments are stripped before scanning **and** the raw text is scanned (E4's two-layer rule); the audit (S3) is the independent second layer.

### 4.4 Axiom audit

Allowed exactly `{propext, Quot.sound, Classical.choice}`. `Classical.choice` is allowed because Mathlib's `Finset` API introduces it; the Mathlib-free core needs only the first two and the closure report prints the actual set per prune. A `_native.native_decide.ax_k` axiom can appear only in **harness refutation modules** (Lean-checked LRAT, §7.5), never in a candidate; the harness records the expected names at generation time and re-matches them.

### 4.5 Timing and resource policy

One Lean process per evaluation checks the whole suite (E6 addendum: 9.2 s → 1.5 s). Measured today: 1.57 s with `import ZarPrune` (Mathlib-backed). Hard timeout 240 s; `#eval` of a kill over ≤ 5,000 profiles must fit — a kill that enumerates subsets exponentially will time out and is reported as "kill too slow" (L2). Parallel evaluations: 2–4 (each Lean process ≈ 1–1.5 GB with Mathlib oleans mapped). If the Mathlib cache is unavailable on a machine, the gate falls back to a Mathlib-free profile (`ZarPrune` without `Counting`, baseline = `baseline P`, prompt vocabulary reduced) — a config switch, not a redesign.

### 4.6 Partial credit for non-compiling Lean (shaping only)

`partial_credit ∈ [0, 1)` used only inside the ≤ 0.19 unverified band (§5.3):

* L0/L4: 0.
* L1: `0.15 + 0.5·(decls_ok / decls) + 0.2·first_error_frac` (current formula), capped 0.5.
* L2: `0.6 + 0.2·(sound_decls_without_error / sound_decls)`.
* L3: `0.8 + 0.15·(1 − holes / (holes + filled))`, where `holes` = `sorry` occurrences inside `sound` proofs and `filled` = `have` statements inside those proofs that elaborate without `sorry`; the error/goal text for the first unfilled hole is copied to artifacts (`lean_first_hole`).

The lit review's evidence says dense Lean rewards buy little (+1–2.5 pp); the ladder exists so that MAP-Elites keeps sketches alive as parents, not to make them competitive with verified prunes.

### 4.7 NL → Lean: the two-stage option, made concrete

Not "NL then a separate autoformalizer" (compile ≠ faithful, and the statement is fixed anyway). The evidence-backed split is **reasoner writes NL + kill + typed skeleton; cheap model fills holes under compiler feedback; the harness owns the theorem**:

* `sorry` is permitted **only** inside the body of a `sound` field (regex: after `sound := by` up to the next top-level declaration). Elsewhere it is a forbidden token (L4). A candidate with any `sorry` is at most L3.
* Two generation flavours via OpenEvolve's ensemble: an *architect* model (Gemini 3.1 Pro / Claude Opus 4.7, weight 0.3, `reasoning_effort: medium`, `reasoning_max_tokens` set so the answer is not starved) and a *filler* model (DeepSeek-V4-Flash / GPT-5.4-mini, weight 0.7). `prompt.template_variations` carries two instruction variants ("propose a new prune with a typed skeleton" / "fill the `sorry` holes of the current program using the reported goals"); per-model system messages exist in `LLMModelConfig.system_message` but `generate_with_context` currently takes the caller's system message, so the variation mechanism is the reliable route (verify before relying on per-model messages).
* Optional evaluator-side filler (`ZAR_UB_FILLER_MODEL`): on L3, up to `R = 2` rounds of "here is the hole, its goal, the error" to the filler model, each round re-running the gate; per-evaluation cost cap $0.02; disabled in NO-LLM mode. Gains saturate at 2–5 rounds in every surveyed system, so R = 2 is the sensible default.

---

## 5. Reward function

### 5.1 The suite

`suite.py` builds three lists of `(Instance, CaseTable)`:

* **TRAIN** (ladder, all-UNSAT, fully labelled, pure mode): `(9,9)50, (9,10)55, (10,10)61, (10,11)65, (11,11)70, (11,12)75, (12,12)81`, later `(12,13)87, (13,13)93, (10,14)78`. Each survivor carries an exact conflict count.
* **BATTERY** (`w = z`, pure mode): the same cells; cases with status `sat` carry a witness. Plus the record profiles of the lower-bound paper — `(11,21)` at 116: rows `11^6 10^5`, cols `6^11 5^10`; `(12,22)` at 132: rows `11^12`, cols `6^22` — and Tan's listed maximal graphs' profiles, as extra witnessed cases (`data/witnesses_33.json`).
* **TARGET** (`ZAR_UB_TARGETS="m,n,w;…"`, table mode): replay tier first (`(10,23,113)`, `(12,18,109)`, `(11,23,124)`, `(9,23,104)`), then open tier (`(13,19,123)`, `(16,17,134)`, `(16,18,140)`). Difficulty is capped/estimated (§6).

Scoring is **marginal over the verified baseline** `B` = the Lean library at run start (`counting P` today; `counting ∪ Evolved` after promotions). Survivors `S_I = {q ∈ C_I : ¬B.kill(q)}`. E6's lesson is built in: `B` is what is proven in Lean, never what the Python case builder happens to apply.

### 5.2 Per-instance quantities

For instance `I` with survivors `S_I`, difficulty `d_I(q)` (§6), total work `W_I = Σ_{q∈S_I} d_I(q)`, top-decile set `H_I` (the ⌈|S_I|/10⌉ hardest survivors), Lean mask `L_I` (defined iff L5), Python mask `Y_I`:

* proven kill mass `M^L_I = Σ_{q∈S_I, L_I(q)} d_I(q) / W_I`
* proven tail kills `T^L_I = |{q ∈ H_I : L_I(q)}| / |H_I|`
* empirical kill mass `M^Y_I`, tail `T^Y_I` (same with `Y_I`)
* fires `f_I = [∃ q ∈ S_I, L_I(q)]`
* agreement `a_I = |{q : L_I(q) = Y_I(q)}| / |C_I|`

Aggregation with weights `ω_I = 1` (train), `2` (target): `M^L = Σ ω_I M^L_I / Σ ω_I`, likewise `T^L, M^Y, T^Y`; generality `g = Σ_I f_I / |𝓘|`.

### 5.3 The formula

```
sound      := no Y-kill of a witnessed case (battery) ∧ no L-kill of a witnessed case (else PIPELINE_BUG)
if ¬sound ∨ lean_status ∈ {L0, L4}:                combined_score = 0
elif lean_status = L5:                              combined_score = 0.20 + 0.60·M^L + 0.15·T^L + 0.05·g
else (L1–L3):                                       combined_score = min(0.19, 0.03·ℓ + 0.02·partial_credit + 0.06·M^Y + 0.02·T^Y)
```

Properties: a verified prune with zero marginal mass scores exactly 0.20 (the initial program); every unverified candidate scores < 0.20, so verification is never outranked; among verified candidates the difficulty-weighted mass dominates, the tail term rewards killing the hard decile specifically (the 61.6 %-of-work decile), and generality nudges toward prunes parametric in `P`. All ranges are `[0,1]`.

Metrics returned (raw, continuous; OpenEvolve bins them): `combined_score, sound_battery, lean_status, lean_partial, proven_gain (M^L), proven_tail (T^L), empirical_gain (M^Y), empirical_tail, target_gain (M^L over TARGET only), generality, agreement, hard_killed (count of censored target survivors killed with proof), survivors_left, n_prunes (number of `Prune`/`CondPrune` definitions), kill_signature (sha1 of the concatenated L masks, string in artifacts), eval_seconds`.

### 5.4 Cascade

`evaluate_stage1`: sandboxed Python kill over all cases (`run_candidate.py`, 90 s timeout), battery, vacuity, Python/Lean-independent statistics; returns `combined_score = 0` if unsound else `0.01 + 0.06·M^Y`. `cascade_thresholds: [0.005]` (OpenEvolve compares `combined_score ≥ threshold`), so only unsound programs skip the Lean stage. `evaluate_stage2` = full evaluation. This keeps the ≈ 2 s Lean process off the programs that are dead on arrival.

### 5.5 Anti-cheating checklist (each item maps to a mechanism)

| threat | mechanism |
|---|---|
| unsound Python kill | witness battery (score 0); Python mask never prunes anything |
| unsound Lean kill | impossible: `sound` is kernel-checked; `PIPELINE_BUG` flag if a verified mask ever kills a witnessed case (would indicate a definitions mismatch, halts the run) |
| smuggled axioms, `native_decide`, `implemented_by`, `csimp`, `opaque` | scan (S0) + audit (S3) + `leanchecker` at promotion |
| forged kill-mask output | `#` commands forbidden in candidates; nonce-tagged markers; mask length check |
| redefining `Valid`/`HasKst`/wrapper names | fixed wrapper types outside `Cand`; declared-name blacklist; `autoImplicit false` |
| vacuous prune (`kill ≡ false`) | verified but `M^L = 0` → 0.20, no gain; not penalised (it is the honest floor) |
| trivial prune (only re-kills the baseline) | marginal scoring over `B` |
| slow kill (exponential enumeration) | `#eval` timeout → L2 |
| gaming difficulty labels | labels precomputed from the CNF alone, stored in `cache/`, read-only for the evaluator |
| gaming the empirical term with an unsound-but-uncaught kill on targets | bounded to ≤ 0.06 and never trusted; a Python kill that disagrees with Lean is reported (`agreement`) and never credited |
| code-length / lemma-count farming | no reward term depends on them; `n_prunes` is a metric, not a feature or a score term |
| duplicate phenotypes | `kill_signature` in artifacts; evaluator keeps `cache/signatures.json` and reports `signature_seen` (informational; the MAP-Elites `diversity` dimension is code-based) |

### 5.6 MAP-Elites feature dimensions

`feature_dimensions: ["empirical_gain", "lean_status"]`, `feature_bins: {empirical_gain: 10, lean_status: 6}`. Rationale: the grid then holds, per empirical-promise level, the best verified and the best sketch — exactly the two things a filler iteration and an architect iteration want as parents. A third optional dimension `generality` (bins 5) when the suite has ≥ 5 instances. Islands 3–4, `population_size 60`, `archive_size 20`, `migration_interval 20`, `exploration_ratio 0.3`, `exploitation_ratio 0.6`, `elite_selection_ratio 0.2`. OpenEvolve requires raw values, which all listed metrics are.

---

## 6. Branch difficulty measure

### 6.1 Definition

`d(q)` = number of CaDiCaL conflicts needed to refute the case CNF (`encode_case`, §3.2) with the fixed solver configuration (`cadical195` via python-sat, default options, no preprocessing switches), measured once and cached in the case table. E10 justification: conflicts and propagations are statistically indistinguishable workload measures; wall-clock is noisier; conflicts are reproducible across machines for a fixed solver build.

### 6.2 Algorithm (`zar_ub/difficulty.py`, extended)

```
probe(q, schedule=[2_000, 20_000, 200_000, 2_000_000], t_per_cap=[5, 30, 120, 600] s):
    feats = {log2_volume(rows), nclauses, nvars, prop_rate_1deep}      # candidate-independent, O(mn) BCP calls
    for cap, tl in zip(schedule, t_per_cap):
        res = solve_cnf(cnf, conf_budget=cap, time_limit=tl)
        if res.status == 'sat':     return Probe('sat', witness=res.matrix, has_kst-checked)
        if res.status == 'unsat':   return Probe('unsat', conflicts=res.conflicts, censored=False, feats)
        if cap == schedule[-1] or cumulative_time > T_case: break
    return Probe('unknown', conflicts=cap, censored=True, feats)

d(q) = conflicts                       if status == 'unsat'
     = max(cap_reached, f̂(feats, c₂ₖ)) if censored
```

`c₂ₖ` = conflicts consumed at the 2,000 cap (a censored run always yields it). `f̂` is an isotonic/log-linear regressor fitted on the fully labelled ladder (E10: Spearman 0.91 for `c₂ₖ`; volume 0.62; clauses 0.57), refitted whenever a ladder cell is added; validated by RMSE in `log10` on a held-out cell (expect ≈ 0.5, the SATzilla ceiling). Censored labels are lower bounds and are used as such: `max(cap, f̂)`.

Ladder cells run the full schedule to completion (E10 did this: 1,571 cases, 11.8 M conflicts, ≈ 3 min). Target cells stop at 200k by default (`(10,23)`: 1,189 survivors × ≤ 3.5 s ≈ 70 min; `(12,18)`: 1,518 ≈ 90 min; `(13,19)`: 2,648 ≈ 2.5 h), with a first pass at 20k (0.35 s/case, ≈ 10 min) so evolution can start before the tail is resolved; the table is updated in place as the deeper passes finish.

### 6.3 Large tables: sampling with common random numbers

If `|S_I| > 50,000` (pure-mode `(12,17)`: 32k after A/I/D; `(13,19)` pure 30k; `(16,17)` pure 711k cases), probe a stratified sample: strata = deciles of `log2_volume(rows)` × `#distinct column sums`; `N = 2,000` cases, fixed seed. `W_I ≈ |S_I|·mean_sample(d)`; candidate masks are still evaluated on **all** cases (kill is cheap), but `M^L_I` is computed on the sample: `Σ_{sampled ∧ killed} d / Σ_{sampled} d`. Chivilikhin's Theorem 3 gives the `(ε,δ)` sample-size rule; the shared sample gives all candidates common random numbers so their scores are comparable. The sample composition is stored in the table.

### 6.4 Secondary, solver-independent label

For certified cases the trimmed LRAT length (bytes after `drat-trim`) is stored alongside; it is reported in the closure report as "work saved" and used to sanity-check `f̂` on targets (it is not used in the reward, to keep one unit).

### 6.5 Cost summary

| step | cost |
|---|---|
| features (volume, clause count, 1-deep propagation rate) | ≤ 0.05 s/case |
| 2k-conflict probe | ≈ 0.05 s |
| 20k | ≈ 0.35 s |
| 200k | ≈ 3.5 s |
| 2M | ≈ 35 s (ladder only) |
| ladder complete labelling (7 cells) | ≈ 3 min total (E10) |
| target table, 20k pass / 200k pass | 10 min / 1–3 h per cell |

---

## 7. SAT execution policy

### 7.1 When SAT runs

1. **Never inside `evaluate()`.** The evaluator reads cached tables; its cost is Python kill + one Lean process.
2. **Table build** (`python -m zar_ub table M N S T W [--pure] --cap …`): offline, before a run; incremental deepening of target tables in the background (§6.2).
3. **Closure attempts** on target cells: a daemon (`python -m zar_ub closure-daemon --targets … --poll 50`) reads the best verified program from the latest OpenEvolve checkpoint every 50 iterations, recomputes survivors `S'` under its Lean mask, estimates remaining work `Ŵ(S') = Σ_{q∈S'} d(q)` and launches certification when `Ŵ(S') ≤ Ŵ_budget` (default `5·10^8` conflicts ≈ a few core-hours) **or** when the survivor count drops below 200, or on manual trigger. Each launch is logged in `experiments/LOG.md` with the program id.
4. **Promotion check**: when a prune is promoted into `Evolved.lean`, all suites are re-scored (no SAT) and the daemon re-evaluates triggers.

### 7.2 Per-case portfolio and schedule

Round-robin conflict escalation across all survivors of a cell (`2k → 20k → 200k → 2M → 20M`), so easy cases finish first and the hard tail is identified before any single case consumes hours. Solver order per case: `cadical195` (pysat, budgeted) → `glucose4` (pysat, diversity, same budget) → proof-producing `tools/cadical --binary=false` with wall limits `600 s → 3,600 s → 6 h` (certificate run; only after the budgeted runs suggest the case is within reach). Parallelism: a process pool of `cores − parallel_evaluations` workers. A case still open after `20M` conflicts / 6 h is **re-cubed**: split on the support of the heaviest row (the `C(n, r_1)` sub-cases, reduced by column-block symmetry as an addition inside the certificate) — v2, with its own cover obligation.

### 7.3 Certificates (`zar_ub/certify.py`, as built in E9)

`cadical -q --binary=false case.cnf case.drat` → `drat-trim case.cnf case.drat -L case.lrat` (must print `s VERIFIED`) → `lrat-check case.cnf case.lrat` (must print `c VERIFIED`). Stored under `cache/certs/<instance>/` with `manifest.json` (CNF sha1, LRAT size, times). A SAT answer stores the decoded matrix, re-checks it with the independent `has_kst`, and marks the *cell* as "lower bound found" (a witness, never a refutation). Retention: LRAT files kept for the closure report; DRAT deleted.

### 7.4 Closure report (`zar_ub/closure.py`)

Per case: `PRUNED (Lean: <prune name>, axioms {…})` | `REFUTED (LRAT verified, <path>, <bytes>)` | `OPEN (<status>)`. The claim `z(m,n;s,t) ≤ w−1` is printed as established only if no case is OPEN. The report lists the trusted base (§3.4), the external facts used (ledger), and the Lean source accepted by the gate.

### 7.5 Lean-checked refutation (milestone M9)

Lean 4.34 core ships `Std.Tactic.BVDecide.LRAT.check` with `check_sound`; a branch theorem is `LRAT.check_sound proof cnf (by native_decide)`, which introduces one named `_native` axiom per leaf. Policy: allowed **only** in harness-generated refutation modules, with the axiom names recorded and re-matched; never in candidates. Memory ≈ 2 GB + 0.3 GB/GB of LRAT; `(9,9,50)`'s 3.8 MB is trivial, `(10,23)`-scale certificates (tens of GB in dfield's run) are not — the report then says which leaves were Lean-checked and which only `lrat-check`ed. The remaining seam is the encoding-completeness theorem (`Valid P A → profileOf A = q → (encode P q).Sat (assign A)`), M9.

---

## 8. Generalization across (m, n, s, t)

### 8.1 Parametric prunes

`Prune P` and every library term are parametric in `Params`; the wrapper tries the general shape first. The suite mixes cells so that an instance-specific `candidate : Prune target` typechecks only on its own instance (other instances get `gate FAIL`, `g` small), which the generality term and artifacts make visible. For `(s,t) ≠ (3,3)` the same code applies (`z(m,n;2,2)` cells are cheap sanity instances: the `(2,2)` witness matrices and the known table give a second battery; `(4,4)` cells appear in Tan's data).

### 8.2 `BoundTable` and conditional prunes (new file `lean/ZarPrune/Table.lean`)

```lean
structure BoundTable where
  lookup : Nat → Nat → Nat → Nat → Option Nat            -- data only, computable
def BoundTable.Sound (T : BoundTable) : Prop :=
  ∀ m n s t U, T.lookup m n s t = some U →
    ∀ A : Mat m n, ¬ HasKst ⟨m, n, s, t, 0⟩ A → weight A ≤ U
structure CondPrune (P : Params) (T : BoundTable) where
  name  : String := ""
  kill  : Profile P.m P.n → Bool
  sound : T.Sound → ∀ A, kill (profileOf A) = true → ¬ Valid P A
def CondPrune.ofPrune (p : Prune P) : CondPrune P T := ⟨p.name, p.kill, fun _ => p.sound⟩
def CondPrune.toPrune (p : CondPrune P T) (hT : T.Sound) : Prune P := ⟨p.name, p.kill, p.sound hT⟩
def CondPrune.or / ofList  -- as for Prune
def countingTable : BoundTable := ⟨fun m n s t => if 1 ≤ s ∧ 1 ≤ t then some (min (waterfillBound m n s ((t-1)·C(m,s))) (…row side…)) else none⟩
theorem countingTable_sound : countingTable.Sound   -- from weight_le_waterfill (exists)
```

The kill sees only `T.lookup` (data); the proof sees `T.Sound`. Library conditional prunes to write once (M2): deletion against `T.lookup P.m (P.n−1)` / `T.lookup (P.m−1) P.n` (wrapping `argDelCol/argDelRow`, ≈ 30 lines), the k-column-prefix form of Argument I (`Σ_{top k} c_j ≤ T.lookup P.m k`, needs restriction along an increasing `Fin k → Fin n`, ≈ 150 lines), and the minimum-row-degree cut (`r_i ≥ w − T.lookup (P.m−1) P.n`, a corollary of deletion).

### 8.3 The bound DAG and provenance

`zar_ub/known.py` becomes a `Ledger` with three provenance classes per entry: `counting` (provable, `countingTable_sound`), `pipeline` (closed by this system: closure report + Lean theorem name), `external` (cited, e.g. Tan 2022 Table 3; never the † 2026 preprints — those go into a separate `data/claims_2026.csv` used only as targets to replay). The gate's `gateTable` is generated from the ledger; the final theorem for a cell is

```lean
theorem z_13_19_le_122 (hT : gateTable_13_19.Sound) : ∀ A : Mat 13 19, ¬ HasKst ⟨13,19,3,3,123⟩ A → weight A ≤ 122
```

with `gateTable_13_19` listing exactly the entries used. When every entry is `counting` or `pipeline`, `hT` is discharged by composing the entries' theorems (the DAG in topological order) and the statement becomes unconditional; otherwise the hypothesis names the external facts, as dfield does ("they cannot honestly be erased"). A mechanical claim checker (monotonicity `z(m,n) ≤ z(m,n+1)`, `z(m,n+1) ≤ z(m,n) + m`, witness comparison) runs on the ledger at every write.

### 8.4 Curriculum and transfer

A run's promoted library becomes the baseline of the next; the suite moves up the ladder (add `(12,13)`, `(13,13)`, `(10,14)`) as the lower cells stop offering mass; targets move from replay tier to open tier when their survivor counts drop. Prunes that transfer across `(m,n)` show up as `g → 1`; those that do not are kept but flagged in the closure reports as instance-specific.

### 8.5 Where the room is (from the review's census) and how the genome reaches it

Baseline after M3 is `A + A′ + D + D′ + deletion/prefix (with the ledger) + DGH(4)`. Remaining kills must come from (a) cross-side arguments coupling `rows` and `cols` (D-type on the max row with the other side's exact multiset), (b) residue/overlap arguments (P14–P16: marked-row deficits mod `g`, pair-deficit residues, three-column budgets) — all "library identity + `decide` constants + `omega`" once the identities are in `Counting.lean` (≈ 450 Mathlib lines each; M6), and (c) Farkas combinations (P11: multipliers found outside by an exact rational LP, checked inside by `omega`). For (b) and (c) the G2 search-mode genome is the natural fit (evolve the marked set / modulus / multipliers; the evaluator instantiates a fixed Lean template with `decide`); the design allows both G1 and G2 files in the same run via two initial programs on separate islands.

---

## 9. Cost and time model, model choice

### 9.1 Per-evaluation (no LLM)

| component | measured / estimated |
|---|---|
| Python kill over suite (≈ 2,500 cases now; ≈ 10k with targets) | 0.2–1.5 s (subprocess) |
| Lean gate, whole suite, Mathlib-backed | **1.6 s** (E5/E6 + today); Mathlib-free 0.3–0.5 s |
| Lean `#eval` masks | included; grows ≈ 0.2 s per 1,000 cases |
| scoring, artifacts | < 0.1 s |
| **total** | **3–6 s**; 2–4 in parallel |

### 9.2 Tables and closure (SAT)

Ladder: done (≈ 3 min). Replay targets at 20k cap: `(10,23)` 10 min, `(12,18)` 12 min, `(11,23)` 3 min, `(9,23)` 1 min; 200k pass: 1–3 h each. Certification of a replay cell: unknown a priori; dfield's `(10,23)` needed 13 SAT/MIP profiles and 25 GB of certificates on a 48-vCPU cluster, so `(10,23)` is *not* expected to close on a laptop without new prunes — that is the experiment. `(9,9,50)`-sized certification: seconds.

### 9.3 LLM tokens and prices

Prompt: system ≈ 1.5k tokens, current program ≈ 3–4k, artifacts ≤ 16 KB ≈ 4k, top/diverse programs 2 × 2k → ≈ 10–12k input; output 4–8k (Lean + Python + NL). OpenRouter prices recorded in the review (checked live 2026-09-21): DeepSeek-V4-Flash $0.06/$0.11 per M; GPT-OSS-120B $0.15/$0.60; GPT-5.4-mini $0.75/$4.5; Gemini 3.1 Pro $2/$12; Claude Opus 4.7 $5/$25.

| model | $/iteration (12k in, 6k out) | with reasoning tokens (×2 out) |
|---|---|---|
| DeepSeek-V4-Flash | 0.0014 | 0.002 |
| GPT-OSS-120B | 0.005 | 0.009 |
| GPT-5.4-mini | 0.036 | 0.063 |
| Gemini 3.1 Pro | 0.096 | 0.17 |
| Claude Opus 4.7 | 0.21 | 0.36 |

Ensemble 70 % filler (Flash/mini) + 30 % architect (Gemini Pro): ≈ $0.04–0.06 per iteration → **1,000 iterations ≈ $40–60**; with Opus as architect ≈ $80–120. Wall: LLM 30–120 s + eval 5 s, 4 workers → ≈ 4–6 h per 1,000 iterations. The `cost.py` ledger (OpenRouter key-usage endpoint) is appended before and after every run.

### 9.4 Model choice

* Architect: Gemini 3.1 Pro (best cost/quality for Lean per the UW study: 92 % refine@32 zero-shot) or Claude Opus 4.7 (86 %); `reasoning_effort: medium`; `max_tokens 16000` with `reasoning_max_tokens` reserving ≥ 8k for the answer on Anthropic models (the config comment documents the starvation failure).
* Filler: DeepSeek-V4-Flash first (Goedel-Architect's $0.44/problem frontier used exactly this split); GPT-5.4-mini as the second filler if Flash's Lean 4.34 syntax error rate exceeds 50 % in the probes (§10.2).
* No specialised prover is reachable via API (DeepSeek-Prover-V2 has no endpoints; Kimina/Goedel not listed), and all were trained on older Lean; do not plan around them.
* Temperature 0.6, `top_p 0.95`, `retries 2`, `timeout 600`.

---

## 10. Test plan

### 10.1 NO-LLM mode ($0) — mandatory before any paid call

1. **Unit tests** (`python -m unittest discover tests`): partitions vs brute force, waterfilling vs brute force, encoding SAT ⇔ matrix exists (tiny, with/without lex), Argument D reference vs Lean `#eval` on random profiles (`experiments/E8_counting/check_masks.py`, equality for `argA/argD`, subset for threshold forms), scan/audit unit tests with each forbidden token, nonce-marker forgery test.
2. **Golden candidate bank** (`tests/candidates/*.py`, expected metric ranges in `tests/golden.json`, run by `tests/test_evaluator.py`):

| candidate | expected |
|---|---|
| `initial` | L5, `M^L = 0`, score 0.20 |
| `python_only_dgh4` (DGH(4) in Python, Lean unchanged) | L5 (Lean is the baseline), `M^Y > 0`, score 0.20, `agreement < 1` reported |
| `lean_dgh4` (hand-proved DGH(4), M3 artefact) | L5, `M^L > 0` on `(13,17)`-type suites, score > 0.20 |
| `unsound_row7` ("kill any case with a row ≥ 7") | battery counterexample, score 0 |
| `sorry_in_kill`, `sorry_in_sound` | L4 / L3 |
| `native_decide`, `axiom_smuggle`, `implemented_by`, `opaque` | L4, score 0 |
| `mask_spoof` (`#eval IO.println "MASK …"`) | L4 (forbidden `#eval`/`IO`) |
| `slow_kill` (enumerates all subsets) | L2 "kill too slow" |
| `instance_specific` (`candidate : Prune target`) | L5 on its instance, gate FAIL elsewhere, `g = 1/|𝓘|` |
| `cond_deletion` (uses `T.lookup`) | L5; kills the 0-case replay cells' partitions when the ledger has Tan's neighbours |

3. **End-to-end loop without money**: OpenEvolve's `LLMModelConfig.init_client` (used by `llm/ensemble.py`) takes a callable; `experiments/no_llm/replay_llm.py` provides (a) `ReplayLLM` — returns recorded responses from `experiments/E7_llm_smoke/probes/*/response.md` in order, and (b) `SyntheticMutator` — a deterministic "LLM" that applies scripted edits (insert a snippet from `tests/snippets/{dgh4,deletion,residue_sketch,broken_proof}.lean` with matching Python, or perturb a threshold). A 30-iteration run with `SyntheticMutator` must show: population never contains an unsound program with score > 0, verified programs dominate the archive, MAP-Elites cells for L3 sketches are occupied, checkpoints resume. `llm.manual_mode` (task-queue directory) is the fallback for a human-in-the-loop dry run.
4. **Closure smoke**: `python -m zar_ub certify 9 9 3 3 50 --pure --lean tests/candidates/initial.lean` reproduces `z(9,9) ≤ 49` (E9) with the prune reducing certificates from 36 to the survivors of `counting`; the report lists the trusted base.
5. **Replay tier at zero SAT**: with the ledger in table mode, `(10,21,107)`, `(11,19,107)`, `(11,20,112)` have zero admissible cases (computed today); M2's conditional-deletion prune plus the Lean cover milestone turns each into a Lean theorem conditional on `tanTable.Sound`.

### 10.2 With the $16 (spend plan, logged by `experiments/cost.py` before/after each step)

| step | what | est. cost |
|---|---|---|
| E7a | `probe_one.py` once per model (Flash, GPT-5.4-mini, Gemini 3.1 Pro, Opus 4.7) on the real prompt; inspect diff applicability, Lean syntax error rate, whether the model touches `sound` | $0.6 |
| E7b | two-stage probe: architect sketch (2 calls) + filler fills (6 calls) on the DGH(4) task with goals in the prompt | $1.0 |
| E7c | three 8-iteration runs with Flash only (prompt variants A/B/C) + one 8-iteration ensemble run | $1.7 |
| E7d | one 25-iteration ensemble run on the ladder suite with `(13,17,117)` as target (DGH(4) closes it with zero SAT if found) | $4.0 |
| reserve | re-runs after prompt fixes; never below $8 remaining before E7d starts | $8+ |

Success criteria for the smoke tests: ≥ 1 model produces an applicable diff with a compiling `kill` (L2+) in ≤ 3 attempts; ≥ 1 L5 prune beyond the baseline on the ladder or `(13,17)`; no forbidden-token candidate scores > 0; per-iteration cost within 2× the table in §9.3.

---

## 11. Risks and failure modes

1. **Frontier models cannot write Lean 4.34 double-counting proofs against this API.** Mitigation: `budget_general`, `hasKst_of_subsets`, layer-cake lemmas and the deletion/waterfill lemmas already exist, so most new prunes are "instantiate + `omega`"; typed skeletons with `sorry` holes; error/goal feedback; G2 templates for residue/Farkas families where the Lean is fixed and only constants evolve.
2. **Baseline too strong on the ladder** (survivors are near-regular; counting-type prunes cannot kill them). Symptom: every verified candidate stays at 0.20. Mitigation: include the DGH-attackable cells `(13,17,117)`, `(13,18,122)` and `(15,17,133)` in the suite; add residue identities (M6) to the library so that P14-type arguments are one `decide` away; report the empirical band so that Python-only discoveries are visible.
3. **Reward hacking** (§5.5): covered mechanism by mechanism; the unresolved residual is an unsound Python kill on target cells that no witness catches — bounded to 0.06 and never trusted.
4. **Case explosion** at `m ≥ 13` in pure mode (`(16,17)`: 711k cases): table mode with the ledger, sampling (§6.3), and the closure DAG; pure mode is for the ladder only.
5. **Heavy-tailed difficulty makes scores noisy** across cells: the tail term and per-instance reporting; refit `f̂` as labels deepen; never compare scores across runs with different tables (record `table_hash` in every metric dict).
6. **Mathlib cache fragility** (7.9 GB; the v4.34.0 top-level `Mathlib.olean` was missing in E5): only targeted imports are used; `experiments/E5_mathlib/setup.sh` documents the recovery; the Mathlib-free profile is the fallback.
7. **Trusted base is not verified** (encoding completeness, partition completeness, cover with sorting): stated in every closure report; M8/M9 shrink it; `bv_decide` cross-checks tiny instances.
8. **External 2026 claims** are unreviewed (dfield, Afrasyab, Hou; Padhi refuted): they are targets to replay, never ledger entries; the claim checker guards the ledger.
9. **OpenEvolve mechanics**: diff application failures on long Lean strings (use `diff_based_evolution: true` with `max_code_length 60000`; fall back to full rewrites for the architect model); per-model system messages may not be honoured (use `template_variations`); custom feature dims must be raw values.
10. **Cost overrun**: the cost ledger, per-step caps, and `ZAR_UB_FILLER_MODEL` off by default; the $8 reserve rule.
11. **Runaway `#eval`** or Lean memory: timeouts, `ulimit -v`, ≤ 4 parallel gates.
12. **Closure never triggers** on a target because `Ŵ` stays above budget: the report then contains the *reduction* achieved (survivors and estimated work before/after), which is itself a thesis result; re-cubing is the v2 lever.

---

## 12. Milestones (each with an acceptance test)

| # | milestone | acceptance |
|---|---|---|
| M0 (done) | engine, gate, ladder tables, difficulty labels, LRAT certification (E1–E10), Counting.lean integrated | `lake build` 0.9 s; gate 1.6 s; `(9,9)≤49` certified |
| M1 | evaluator v2 per §5 (baseline = `counting`, ladder statuses, nonce markers, extended scan, cascade), `Table.lean` + `CondPrune`, `Prune.mono`, prompt/API refresh, golden candidate bank | all golden expectations pass; initial program scores 0.20 |
| M2 | conditional prunes (deletion, k-prefix, min-degree) + ledger provenance + bound DAG | replay cells `(10,21)`, `(11,19)`, `(11,20)` produce conditional Lean theorems with zero SAT |
| M3 | DGH(4) `v = s−1` prune proved (by hand if the LLM does not find it) | `(13,17)≤116`, `(13,18)≤121` closed with zero SAT, verified |
| M4 | NO-LLM loop (`ReplayLLM`/`SyntheticMutator`), closure daemon, promotion path (`Evolved.lean`, `leanchecker`) | 30-iteration synthetic run passes the invariants of §10.1(3) |
| M5 | $16 smoke tests E7a–E7d | criteria of §10.2; prompt frozen for the real run |
| M6 | residue/overlap identities (P14–P16) in `Counting.lean`; G2 template genome for marked sets/moduli and Farkas multipliers | kills the three `(9,23,104)` and both `(12,23)` profiles dfield kills, verified |
| M7 | replay tier closure attempts: `(9,23)≤103`, `(12,23)≤134`, `(11,23)≤123`, `(10,22)≤110`; `(12,18)≤108`, `(10,23)≤112` as stretch | closure reports; per-case ledger; comparison with dfield's counts |
| M8 | Lean cover: `exists_sorted_conjugate` + decidable partition enumeration completeness | `upper_bound_of_cover` discharged for `(9,9,50)` with Python out of the cover seam |
| M9 | Lean-checked LRAT leaves + encoding-completeness theorem for the sequential-counter encoding | `(9,9,50)` end-to-end Lean theorem with named `_native` axioms only |
| M10 | real run (AlphaEvolve Cloud / larger budget), 1,000–3,000 iterations, targets `(16,17)≤133`, `(16,18)≤139`, `(13,19..21)` | new or independently re-verified bounds, or quantified survivor reduction with certificates |
| M11 | thesis chapters: difficulty model calibration (the first per-case timing data for a Zarankiewicz split), ablations (islands, architect/filler split, tail term), axiom/provenance ledgers | reproducible from `experiments/LOG.md` |

Files to create or change, in order: `lean/ZarPrune/Table.lean` (new), `lean/ZarPrune/Prune.lean` (`Prune.mono`), `zar_ub/lean_gate.py` (scan list, nonce, ladder, CondPrune wrapper), `zar_ub/known.py` (ledger provenance), `zar_ub/difficulty.py` (schedule, features, `f̂`, sampling), `evaluator.py` (§5 formula, stages), `suite.py` (weights, witnesses), `initial_program.py` (§2.3), `config.yaml` (§2.4, §5.6), `experiments/no_llm/` (replay/synthetic clients), `zar_ub/closure_daemon.py` (§7.1), `tests/test_evaluator.py` + `tests/candidates/`.

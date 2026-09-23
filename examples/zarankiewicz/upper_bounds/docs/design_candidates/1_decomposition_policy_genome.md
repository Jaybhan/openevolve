<!-- Candidate design 1 (angle: the genome is the decomposition + solving policy; Lean verifies lemma soundness and cover completeness; every leaf refuted by a checked LRAT certificate; reward is certified difficulty-weighted work removed under a fixed budget). Judges ranked it 3rd (scores 6.5 / 6). -->

# Design: evolving the decomposition + solving policy for Lean-gated Zarankiewicz upper bounds

Design document for the thesis system described in `docs/proposal_section2.md` (MEng, Jay Bhan; advisor S. Raghuraman). Written 2026-09-21 against the current state of `examples/zarankiewicz/upper_bounds/` (engine `zar_ub/`, Lean library `lean/ZarPrune`, experiments E1–E10 in `experiments/LOG.md`) and the literature review `docs/literature_review.md` (cited below by its section numbers and note slugs).

Absolute paths used below are relative to `/Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/upper_bounds/`.

---

## 0. Reading guide: what exists, what this document adds

Already built and measured (E1–E10):

| component | file | status |
|---|---|---|
| Tan Algorithm 1 case generator (Arguments A + I, proper minors only) | `zar_ub/partitions.py` | working, bug E1 fixed |
| Fixed-sum CNF (aux `K_{s,t}` encoding, Sinz counters, double-lex within equal-sum blocks) | `zar_ub/encoding.py` | working |
| Probing / difficulty (CaDiCaL 1.9.5 via pysat, conflict caps) | `zar_ub/difficulty.py`, `zar_ub/solve.py` | working; 1,571 labelled cases (E10) |
| Case tables (cached environment) | `zar_ub/casetable.py`, `cache/` | 30 tables, pure mode |
| Lean gate (scan, elaborate, type-check, axiom audit, `#eval` kill mask) | `zar_ub/lean_gate.py` | 0.3–1.8 s/candidate (E4/E5) |
| ZarPrune core (Mathlib-free) + `Counting.lean` (Mathlib-backed: argA/argAT, argD/argDT, deletion, waterfill, transposition) | `lean/ZarPrune/*.lean` | builds; axioms ⊆ {propext, Quot.sound, Classical.choice} |
| Certified refutation DRAT→LRAT→lrat-check | `zar_ub/certify.py`, `tools/` | z(9,9;3,3) ≤ 49 closed with 36 certificates (E9) |
| Prune-library evaluator (genome = Lean prune library) | `evaluator.py`, `initial_program.py`, `config.yaml` | working, no-LLM tested (E6) |

This document changes the **genome**. The current evaluator evolves the prune library itself (a Lean-heavy genome that a cheap model rarely improves). The design below makes the primary genome the **decomposition + solving policy**: a Python program that decides how an instance is split, which *already-verified* lemma families are instantiated where, and how solver time is spent. Lean is then used for two things only: soundness of every lemma family (once, at library-admission time, through the existing gate) and cover-completeness of the decomposition the policy produced (per evaluation, mechanically). The existing prune-library evaluator survives as the slow "lemma forge" loop that grows the library.

---

## 1. Overview and claims

### 1.1 The problem shape

An upper bound `z(m,n;s,t) < w` is established by `ZarPrune.upper_bound_of_cover` (`lean/ZarPrune/Prune.lean`):

```
cover   : ∀ A, Valid P A → p.kill (profileOf A) = true ∨ profileOf A ∈ survivors
refuted : ∀ q ∈ survivors, ∀ A, profileOf A = q → ¬ Valid P A
```

Every computational upper-bound proof in the literature has this shape (lit. review §1): enumerate cases, kill some by arithmetic, refute the rest by a solver. The three obligations map onto three trust seams:

1. `p.sound` (a lemma is a real prune, not a symmetry break) — Lean, LLM-facing gate. Exists.
2. `cover` (the case list is complete) — Lean structural theorem (level 0/1) + LRAT tautology certificate (level 2). **New.**
3. `refuted` (each surviving case is empty) — LRAT certificates checked by two independent checkers; Lean import through `Std.Tactic.BVDecide.LRAT.check` as a later milestone. Exists (external), Lean import new.

### 1.2 The angle: evolve the policy, not the object

The thesis question is "use evolutionary search to prune more branches effectively and improve the SAT solver's ability to discover novel upper bounds in an acceptable amount of time" (proposal §2.2). Three findings from the review and from E10 determine the design:

* **Difficulty is heavy-tailed.** On 1,571 fully labelled cases the top 10 % carry 61.6 % of the refutation work (E10). AlphaMapleSAT's entire gain over `march_cu` is fewer very hard cubes (lit. §5.1). So *what* is split, *how deep*, and *where* solver time goes matters more than the number of cases killed.
* **The classical prunes are cheap to instantiate and expensive to invent.** Arguments A, D, deletion and waterfilling are now verified once as parametric families in `Counting.lean`; instantiating them (which row, which threshold, which minor table, which Farkas multipliers) costs nothing in Lean. Inventing a new family (DGH (4), residue arguments P14–P16) needs a reasoning model and several compile rounds (lit. §7). These are two different loops with two different model tiers and cost structures.
* **A leaky verifier gets gamed** ("it always eventually figured out a way to cheat", Tao–Wagner, lit. §8.1). Anything the policy *claims* must be recomputed by the harness: kills by Lean `#eval`, refutations by checked certificates, cover by Lean/LRAT, budgets by the harness's own clock and solver statistics.

Hence the **genome is a policy** `plan(inst, lib, budget) -> Plan` (Section 2): a declarative-plus-callbacks description of the decomposition tree, the lemma instantiations to apply at each level, and the solver portfolio. The harness executes the plan under a fixed budget; everything it measures is certified or censored; the reward (Section 5) is certified work removed relative to a fixed baseline policy, difficulty-weighted (Section 6).

### 1.3 Claims the finished system can make

C1. *Verified prune soundness.* Every kill in a closure comes from a `Prune P` (or `CondPrune P facts`) term accepted by the gate with axioms ⊆ {propext, Quot.sound, Classical.choice}, and the kill mask is computed by Lean.

C2. *Verified cover.* For level-0/1 decompositions (Tan-style profile cubes, or column-only / row-only cubes) the completeness of the enumerated case list is a Lean theorem (`Cover.lean`, Section 3.4). Level-2 sub-cubes carry an LRAT tautology certificate.

C3. *Checked refutations.* Every surviving leaf has an LRAT certificate verified by `drat-trim` and by `lrat-check`; certificates are kept; the closure report lists them.

C4. *Honest conditional facts.* Any use of a smaller cell's value (Argument I, deletion) is an explicit hypothesis of the closure theorem with provenance tags (`proved-here`, `tan2022`, `unreviewed-2026`); a bound is *claimed* only when all hypotheses are `proved-here` or `tan2022`.

C5. *Measured, reproducible difficulty.* Per-leaf difficulty is the CaDiCaL 1.9.5 conflict count under fixed options and seed, exact when refuted, censored-and-calibrated otherwise; total work, tail share and per-policy cost are reported for every evaluation.

C6. *A cost model and a $0 test mode.* The full OpenEvolve loop runs with a stub LLM (Section 10); the $16 OpenRouter budget is used only for calibrating tokens, edit-validity rates and lemma-forge pass rates.

What the system will not claim: new closed-form counting theory, a complete symmetry break for row–column symmetry (impossible unless GI ∈ coNP, lit. §6.4), or a Lean-free replacement for certificates.

### 1.4 Two loops

```
Loop A  (fast, cheap models, every iteration)          Loop B  (slow, reasoning model, rare)
────────────────────────────────────────────           ───────────────────────────────────────
initial_policy.py  ── LLM edits plan() ──►             initial_program.py ── LLM edits LEAN_SOURCE + kill()
evaluator_policy.py:                                   evaluator.py (existing):
  run plan on suite under fixed budget                   battery → Lean gate → axiom audit → masks
  kills via Lean #eval of library terms                  reward = proven work removed (existing formula)
  cover: Lean theorem / LRAT                             accepted terms ──► zar_ub/lemmas.py registry
  leaves: probes + certificates                                   (library grows; Loop A sees new families)
  reward = certified work vs baseline
```

Loop A never needs a Lean *proof* per iteration: only `#eval` of already-proved terms and (for level-2 splits) an LRAT check. Loop B is the existing evaluator with the two-stage NL→Lean option (Section 4.6).

---

## 2. Genome: exactly what the LLM edits

### 2.1 The `Plan` API (harness-owned, `zar_ub/plan.py`)

The policy program may only *decide*; it never runs solvers, never reports results. The harness calls it, executes the plan and measures. The program is executed in a subprocess (`run_candidate.py` pattern) with a wall-clock cap; its return value is validated against the schema below; anything else scores 0.

```python
# zar_ub/plan.py  (fixed; not in the EVOLVE block; imported read-only by policies)
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional, Sequence, Tuple

Side = Literal["both", "cols", "rows"]

@dataclass(frozen=True)
class Instance:            # re-exported from zar_ub.known
    m: int; n: int; s: int; t: int; w: int

@dataclass
class SplitSpec:
    side: Side = "both"                 # Tan (both sums fixed) | dfield (column histogram only) | rows only
    col_prefix: str = "waterfill"       # prefix-bound table for Argument I on columns: "none" | "waterfill" | "ledger"
    row_prefix: str = "waterfill"       # same for rows
    ledger_trust: str = "tan2022"       # facts the policy may draw on: "proved-here" | "tan2022" | "unreviewed-2026"
    order: str = "hard_first"           # leaf ordering: "hard_first" | "easy_first" | "regular_first" | "custom"

@dataclass
class LemmaUse:
    family: str                         # a name from lib.families (Section 2.3)
    params: dict = field(default_factory=dict)   # parameters of that family (validated by the family's schema)
    level: int = 1                      # 1 = profile level (before any SAT), 2 = re-applied to sub-cubes

@dataclass
class SubSplit:
    kind: str                           # "row_support" | "box_sums" | "codegree" | "counter_literal"
    args: dict = field(default_factory=dict)
    # e.g. {"row": 0} for row_support (branch on the support of the heaviest row),
    #      {"blocks": (2,2)} for box_sums, {"pair": (0,1), "levels": [0,1,2]} for codegree,
    #      {"lits": [...]} for counter_literal (assert "row i has >= k ones" literals)

@dataclass
class SolverCall:
    solver: str = "cadical195"          # pysat names: cadical195 | glucose4 | cadical153 | (certify: "cadical-bin")
    conf_budget: int = 2000
    time_budget: float = 5.0
    cuts: Tuple[str, ...] = ()          # verified cut families to add to the leaf CNF: ("pair_cut", "min_row_cut")
    phases: Optional[str] = None        # "dense_first" | "sparse_first" | None (initial phase hints)

@dataclass
class Plan:
    split: SplitSpec
    lemmas: List[LemmaUse]
    solver_ladder: List[SolverCall]     # escalation ladder per leaf, in order; harness stops at first UNSAT/SAT
    refine: Optional[Callable[["LeafView"], Optional[SubSplit]]] = None
        # called when the whole ladder is exhausted on a leaf; return a SubSplit to cube it further, or None
    order_key: Optional[Callable[["LeafView"], float]] = None   # only used when split.order == "custom"
    parallel_leaves: int = 2            # <= harness cap
    notes: str = ""                     # free text the LLM may use to explain the strategy (goes into artifacts)

@dataclass(frozen=True)
class LeafView:            # read-only facts the harness hands to callbacks
    inst: Instance
    rows: Optional[Tuple[int, ...]]     # None when side == "cols"
    cols: Optional[Tuple[int, ...]]
    extra_lits: Tuple[int, ...]         # level-2 literals already asserted
    depth: int
    probe_conflicts: Optional[int]      # conflicts at the last exhausted budget
    probe_status: str                   # "unknown" | "unsat" | "sat"
    nvars: int; nclauses: int
    log2_volume: float
    kst_slack_col: int; kst_slack_row: int   # (t-1)C(m,s) - Σ C(c_j,s) and dual
```

### 2.2 `initial_policy.py` skeleton (the file OpenEvolve evolves)

```python
"""Initial decomposition + solving policy for the Zarankiewicz upper-bound search.

WHAT IS EVOLVED: the function `plan(inst, lib, budget)` inside the EVOLVE block.
It returns a zar_ub.plan.Plan.  The harness executes the plan; you only decide.

You decide (1) how the instance is split into cases (SplitSpec), (2) which
VERIFIED lemma families from `lib` are instantiated, with which parameters
(LemmaUse), (3) the solver escalation ladder per leaf (SolverCall list), and
(4) when a hard leaf is re-cubed (refine callback).  Kills are computed by Lean
from the library's proved terms; leaves are refuted with checked certificates;
the cover of your decomposition is checked.  Nothing you *claim* is trusted.

Reward: certified refutation work removed, weighted by measured difficulty,
under a fixed budget, relative to the baseline policy (this file).  Read the
artifacts: they list the hardest surviving leaves (row/column sums, conflicts),
lemma usage, and where the budget went.
"""
from zar_ub.plan import Plan, SplitSpec, LemmaUse, SolverCall, SubSplit, LeafView

# EVOLVE-BLOCK-START
def plan(inst, lib, budget):
    """inst: Instance(m,n,s,t,w); lib: LemmaLibrary (lib.families -> {name: schema});
    budget: dict(seconds=..., conflicts=..., leaves_parallel=...)."""
    m, n, s, t, w = inst.m, inst.n, inst.s, inst.t, inst.w

    split = SplitSpec(side="both", col_prefix="waterfill", row_prefix="waterfill",
                      ledger_trust="tan2022", order="hard_first")

    lemmas = [
        LemmaUse("argA"),                       # Σ_j C(c_j,s) ≤ (t-1)C(m,s)
        LemmaUse("argAT"),                      # row dual
        LemmaUse("argD", {"form": "sorted"}),   # heaviest row, r lightest columns
        LemmaUse("argDT", {"form": "sorted"}),
        LemmaUse("delCol", {"U": "ledger"}),    # w - min c_j ≤ U(m, n-1)  (fact from the ledger)
        LemmaUse("delRow", {"U": "ledger"}),
    ]

    ladder = [
        SolverCall("cadical195", conf_budget=2_000,   time_budget=2.0),
        SolverCall("cadical195", conf_budget=20_000,  time_budget=20.0),
        SolverCall("cadical195", conf_budget=200_000, time_budget=120.0, cuts=("pair_cut",)),
    ]

    def refine(leaf: LeafView):
        # re-cube a leaf that survived the ladder: branch on the heaviest row's support
        if leaf.depth == 0 and leaf.rows is not None and leaf.rows[0] >= 4:
            return SubSplit("row_support", {"row": 0})
        return None

    return Plan(split=split, lemmas=lemmas, solver_ladder=ladder, refine=refine,
                parallel_leaves=budget.get("leaves_parallel", 2),
                notes="baseline: Tan split, classical counting lemmas, CaDiCaL escalation, one re-cube")
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    from zar_ub.lemmas import library
    from zar_ub.plan import Instance
    p = plan(Instance(9, 9, 3, 3, 50), library(), {"seconds": 60, "conflicts": 2_000_000})
    print(p)
```

What the LLM can change, concretely: the split side and prefix tables; the leaf order; which families and with which parameters (thresholds, which row/column, Farkas multipliers, which ledger tier); the ladder (solvers, budgets, cuts, phases); the refine rule (which sub-split, at which depth, on which leaf features); `parallel_leaves`. It can also add helper functions inside the block (e.g. a small search for Farkas multipliers over the profile's slacks, or a heuristic that predicts hard leaves from `LeafView`).

What it cannot change: the encoding, the lemma library, the certificate pipeline, budgets above the harness cap, or any measurement.

### 2.3 The lemma library the policy sees (`zar_ub/lemmas.py`)

A registry of *verified families*: each entry names a Lean term of type `(P : Params) → params → Prune P` (or `CondPrune P facts`) in `lean/ZarPrune/Counting.lean` / `Cond.lean` / `Farkas.lean` / accepted Loop-B candidates, its parameter schema, a Python mirror of `kill` (for fast screening and the prompt), and one line of documentation the prompt shows.

| family | Lean term | parameters | what it kills (lit. catalogue) |
|---|---|---|---|
| `argA`, `argAT` | `ZarPrune.argA`, `argAT` | – | P1, Kővári–Sós–Turán budget, both sides |
| `argD`, `argDT` | `ZarPrune.argD`, `argDT` | `form ∈ {sorted, threshold}`, `row` (default: heaviest), `theta` | P2, Guy Argument D |
| `delCol`, `delRow` | `ZarPrune.argDelCol P U`, `argDelRow` | `U ∈ {"waterfill", "ledger"}`, `k` (multi-line form) | P5 deletion; conditional on a fact for (m,n−1) / (m−1,n) |
| `prefixI` | `ZarPrune.Cond.prefixI P tbl` | `k` (prefix length), `tbl` | P4, Argument I, top-k lines |
| `farkas` | `ZarPrune.Farkas.combo P base y` | `y : List Nat` multipliers over the registered base inequalities | P11, any nonnegative integer combination |
| `dgh4` (M3) | `ZarPrune.DGH.constraint4 P k` | `k` | P8, Davies–Gill–Horsley (4), v = s−1 |
| `residue3` (Loop B target) | `ZarPrune.Residue.markedRow P g marks` | `g`, `marks` | P14 marked-row residues (dfield P5) |
| `evolved_<id>` | accepted Loop-B candidates | as declared | whatever the forge proved |

Instantiating a family is free in Lean: the harness writes `def L_k := ZarPrune.argDelCol target ⟨U_value, fact_id⟩` etc. into a generated gate file and `#eval`s the folded `kill` on every case (the existing `build_gate_file` machinery). Parameters are validated against the schema in Python *before* the Lean call so malformed plans fail fast.

### 2.4 Why this genome and not the alternatives

* AlphaEvolve's own records mostly came from "search mode" (evolve a search procedure under a budget) and Tao–Wagner found search-mode prompting gave "more efficient programs and much better results" (lit. §8.1, §8.3 G2/G4). A plan is exactly that: one LLM call, millions of cheap checks.
* Chivilikhin–Pavlenko–Semenov show decomposition *choice* (which backdoor set, pairs/triples of split axes) changes total work by 10–100× (lit. §5.2); Tan's row×column split is one point in that space; dfield's column-only split with complete row breaking is another (lit. §9 item 2). The policy genome lets the search move between them and measures the trade-off.
* The lemma library is where Lean effort is amortised: one proof per family, unlimited instantiations, zero per-iteration proving cost. Loop B keeps the door open for genuinely new arguments.

---

## 3. Case decomposition and SAT encoding: fixed vs evolvable

### 3.1 Base CNF `F_P` (fixed)

For `P = (m,n,s,t,w)` the base formula over which everything is stated:

* grid variables `x_{ij} = i·n + j + 1` (row-major, as in `encoding.py` and Tan);
* `K_{s,t}`-freeness in the `aux` form: per increasing `s`-tuple of rows `R`, indicator `y_{R,j}` with clause `¬x_{R_1 j} ∨ … ∨ ¬x_{R_s j} ∨ y_{R,j}` and `AtMost(t−1)` over `{y_{R,j}}_j` (sequential counter). One-directional `y` is sound and complete for the `≤ t−1` constraint (lit. §3.1; `encodings_zar.py` docstring);
* **unary counter variables as base variables**: for every row `i` and level `k`, `R_{i,k}` ≡ "row `i` has ≥ k ones", and `C_{j,k}` for columns, built with the exact two-directional sequential counter (`exact_unary_counter` in `encodings_zar.py`, brute-force verified for `N ≤ 5`). This is the change from the current `encoding.py` (which uses pysat `CardEnc.equals` per line): making the counters base variables lets a *cube* fix sums by asserting literals `R_{i,k} ∧ ¬R_{i,k+1}`, so every level of the decomposition is a partial assignment over one fixed CNF — the precondition for LRAT cover certificates (Section 3.4) and for AlphaMapleSAT-style sub-cubing (lit. §10 item 5);
* weight: `Σ_i r_i ≥ w` is implied by the cube at level 0; for the level-0 cover certificate it is stated as an `AtLeast(w)` totalizer over the grid (only in the cover formula, never in leaves).

Fixed by construction, but *evolvable in selection* (Section 3.3): the symmetry-breaking additions and the verified cutting planes.

Sizes: at `z(10,14;3,3)` ≈ 6.5k vars; `(12,12)`: ~10k vars; per-leaf clause counts 5k–25k (E10 `nclauses`).

### 3.2 Decomposition tree (evolvable shape, fixed algebra)

```
root: F_P  (+ additions Sym_P)
 └─ level 0: split by SplitSpec.side
      "both": cube_q = { R_{i,r_i} ∧ ¬R_{i,r_i+1} }_i ∧ { C_{j,c_j} ∧ ¬C_{j,c_j+1} }_j   for sorted profiles q
      "cols": cube_c = column part only            "rows": row part only
      enumerated by the verified generator with prefix tables (Section 3.4)
 └─ level 1: lemma instantiations kill cubes (Lean #eval mask); survivors ordered by SplitSpec.order
 └─ level 2 (optional, per surviving leaf, on `refine`): SubSplit into sub-cubes
      row_support: branch on the support of row i (C(n, r_i) children, canonicalised within equal-sum
                   column blocks — allowed because the block double-lex is an addition already in Sym_P)
      box_sums:    Tan's z_2(23) trick — sums of a b×b grid of boxes as counter literals
      codegree:    literals "rows i,i' share ≥ k columns" (pair counter over AND-indicators, exact counters)
      counter_literal: any explicit set of counter literals the policy names
      children cover the parent by construction (each kind is a complete case split over a finite
      set of counter values) AND is certified by an LRAT tautology proof (Section 3.4)
```

Depth is capped at 2 in the first version (`refine` is called once per leaf); level-3 requires only lifting the cap once the cover certificates are measured to be small.

### 3.3 Additions and cuts (fixed content, evolvable selection)

*Additions* (symmetry breaking, satisfiability-preserving, never prunes): block double-lex within equal-sum runs (Tan Thm 3.2; Flener's potential argument applies to the case stabiliser `Π S_{R_v} × Π S_{C_u}`, lit. §6.4 (ii)); for `side = "cols"` the full adjacent-row lex chain (BreakID Thm 2: complete for a pure row-interchangeability group). Sound by the once-proved `exists_doubleLex`-style theorem (Section 4.4; trusted base until M2). The policy may **not** invent additions: `Demo.notDescending_unsound` is the formal reason, and the E6 battery is the empirical one.

*Cuts* (implied clauses, must be *verified consequences* of `¬HasKst`): `pair_cut` — for rows `i,i'`, `Σ_{j ∋ i,i'} (c_j − s + 1) ≤ (t−1)(m−s+1)` (D_{s−1}, Afrasyab (9)/(14), dfield pair deficit; lit. P3) encoded over one-directional AND indicators with a sequential counter (as in `encodings_zar.py::pair_cuts`); `min_row_cut` — every row has `r_i ≥ w − U(m−1,n)` (dfield P4; conditional on a ledger fact). A cut may be added to a leaf only if it is registered in `zar_ub/cuts.py` with (a) a Lean statement of the inequality on matrices (`Counting.lean` style) and (b) an entry in the encoding-completeness theorem's clause list (Section 4.4). Adding an unverified clause to a leaf would make "UNSAT" meaningless; the harness rejects unknown cut names.

Why cuts are worth having despite CDCL: cardinality contradictions are exponential for resolution and polynomial for cutting planes (lit. §5.1); a pair cut hands the solver a counting fact it otherwise has to rediscover per leaf.

### 3.4 Cover completeness (Lean + LRAT)

**Level 0/1 — Lean structural theorem** (`lean/ZarPrune/Perm.lean`, `Enum.lean`, `Cover.lean`; M2):

```lean
-- Perm.lean: row/column permutations act on matrices; Valid is invariant.
def act {m n} (σ : Equiv.Perm (Fin m)) (τ : Equiv.Perm (Fin n)) (A : Mat m n) : Mat m n :=
  fun i j => A (σ i) (τ j)
theorem hasKst_act_iff (P) (σ τ) (A) : HasKst P (act σ τ A) ↔ HasKst P A
  -- ←: image of the increasing tuple under σ is a finset of card s (injective); use hasKst_of_subsets.
theorem weight_act (σ τ A) : weight (act σ τ A) = weight A
theorem exists_sorted_act (A : Mat m n) :
    ∃ σ τ, Antitone (rowSum (act σ τ A)) ∧ Antitone (colSum (act σ τ A))
  -- Mathlib `Tuple.sort` gives a sorting permutation for any Fin-indexed Nat vector; sort rows then columns
  -- (row sums are column-permutation invariant, so the second sort keeps the first).

-- Enum.lean: a verified generator of sorted vectors with prefix bounds.
structure PrefixTable (k cap r budget : Nat) where
  ub : Nat → Nat                                  -- ub j = bound on the sum of the j heaviest lines
  ok : ∀ (v : Fin k → Nat), Antitone v → (∀ j, sumFin ... ≤ ub j) ∨ ...   -- see below
def enumSorted (k cap total r budget : Nat) (ub : Nat → Nat) : List (Fin k → Nat)   -- Tan Alg. 1 shape
theorem mem_enumSorted (v : Fin k → Nat) (hA : Antitone v) (hcap : ∀ i, v i ≤ cap)
    (hsum : sumFin k v = total) (hbud : sumFin k (fun i => (v i).choose r) ≤ budget)
    (hpre : ∀ j ≤ k, prefixSum v j ≤ ub j) : v ∈ enumSorted k cap total r budget ub
  -- induction on k; prefix pruning is sound because C(·,r) ≥ 0 and prefix sums are monotone in the next part.

-- Cover.lean
def survivors (P) (tblC tblR) (p : Prune P) : List (Profile P.m P.n) :=
  ((enumSorted P.n P.m P.w P.s (colBudgetOf P) tblC.ub).product (enumSorted P.m P.n P.w P.t (rowBudgetOf P) tblR.ub))
    |>.map (fun ⟨c, r⟩ => ⟨r, c⟩) |>.filter (fun q => !p.kill q)
theorem cover_upto_perm (P) (tblC tblR) (hC : PrefixSound P tblC) (hR : PrefixSound P tblR) (p : Prune P) :
    ∀ A, Valid P A → ∃ σ τ, Valid P (act σ τ A) ∧
      (p.kill (profileOf (act σ τ A)) = true ∨ profileOf (act σ τ A) ∈ survivors P tblC tblR p)
theorem upper_bound_of_cover_upto_perm (P) ... (refuted : ∀ q ∈ survivors .., ∀ A, profileOf A = q → ¬Valid P A) :
    ∀ A, ¬HasKst P A → weight A < P.w
```

`PrefixSound P tbl` says each `ub j` is a proved bound on the weight of the `m × j` minor on the heaviest `j` columns: for `col_prefix = "waterfill"` this is `weight_le_waterfill` on the minor (restriction lemma along `Fin.succAbove`-style embeddings, ~120 lines); for `"ledger"` it is a `FactHolds` hypothesis (Section 4.5). The exactly-`w` thinning (P19) is not needed because cubes fix sums whose total is `≥ w` — the enumerator ranges over `total ∈ [w, min(m·n, …)]` truncated by the budget; in practice only `total = w` survives argA at the frontier, and the policy may restrict to `total = w` only through the P19 lemma (`ZarPrune.thin`, easy, Mathlib-free).

**The list the harness solves is the list Lean computed.** `survivors` is *defined* as the filter in Lean; the gate file `#eval`s it (chunked) and the harness reads it back. So cover at levels 0/1 is not "checked" per run — it is a theorem about the very list being solved. Cost: `#eval` of the enumerator is ~ms per case; the only risk is size (Section 11).

For `side = "cols"`: the same with `p : ColPrune P` (kill reads only `.col`) and `enumSorted` on columns only; rows are unconstrained in the cube, hence the full row lex chain in `Sym_P`.

**Level 2 — LRAT tautology certificate.** For a parent leaf with cube `c` and children `c ∧ d_1, …, c ∧ d_k`: the harness emits `F_P ∧ Sym_P ∧ c ∧ ¬d_1 ∧ … ∧ ¬d_k` (each `¬d_i` one clause) and certifies it UNSAT (DRAT→LRAT→lrat-check). This is `cover_unsat` of LRAT-Catcher / the Pythagorean-triples tautology proof (lit. §6.4). For the four `SubSplit` kinds the children are complete by construction, so the solver finds the proof by propagation and the certificate is tiny; the check exists to make *any* future sub-split (including solver-generated cubes) uniformly verifiable. The Lean statement of what this certifies is `cover_lrat : (F_P ∧ c ∧ ⋀¬d_i).Unsat → ∀ A, (assign A ⊨ F_P ∧ c) → ∃ i, assign A ⊨ d_i`, a one-liner over `Std.Sat.CNF` once the encoding-completeness theorem exists (Section 4.4); until then, lrat-check is the trusted checker and the closure report says so.

### 3.5 What is fixed and what is evolvable — summary

| aspect | fixed | evolvable by the policy |
|---|---|---|
| variables, `K_{s,t}` clauses, counters | yes | – |
| additions (double-lex, row lex chain) | content | – (selected automatically by `side`) |
| cuts | content + proofs | which cuts, on which leaves |
| level-0 split | algebra (`both/cols/rows`, verified enumerator) | side, prefix tables, ledger trust tier, order |
| level-1 kills | library families + Lean masks | which families, parameters, Farkas multipliers |
| level-2 sub-cubes | four kinds + LRAT cover | when, which kind, arguments |
| solving | solvers available, certificate pipeline, harness caps | ladder, budgets, cuts, phases, parallelism |

---

## 4. Verification pipeline

### 4.1 Trust domains

Two domains, as the review's sandboxing analysis recommends (lit. §6.3):

* **LLM lemma modules (Loop B)**: zero axioms beyond {propext, Quot.sound, Classical.choice}; source blacklist; statement owned by the harness wrapper; compiled from source in a subprocess with time/memory limits; `leanchecker` replay before admission to the library.
* **Harness refutation modules**: may carry the named `_native` axioms that `LRAT.check … (by native_decide)` records (one per evaluation since Lean 4.29), which the harness records at generation time and re-matches on audit. Nothing LLM-written ever lives in this domain.

Policies (Loop A) write no Lean at all.

### 4.2 The gate (Loop B; existing `zar_ub/lean_gate.py`, kept)

1. **Static scan** — forbidden constructs (regexes in `FORBIDDEN`): `sorry`, `admit`, `native_decide`, `axiom`, `unsafe`, `implemented_by`, `extern`, `csimp`, `opaque`, `partial`, any `import`, `macro`, `macro_rules`, `elab`, `syntax`, `notation`, `initialize`, `#exit`, `Lean.` namespace, `IO`, `ofReduceBool`, `end ZarPrune` / `end Cand` (namespace escape); `set_option` only `maxHeartbeats`/`maxRecDepth`. Add (from lit. §6.3): `+native`, `trustCompiler`, `open Lean`, `run_cmd`, `#eval` (candidates may not evaluate), `noncomputable` (a kill must compute), `deriving` handlers, `attribute [csimp]`, `@[implemented_by]` in attribute form, and unicode look-alikes of these keywords (normalise NFKC before scanning). Block comments are stripped before scanning and the raw text is scanned again.
2. **Elaboration** — `lake env lean` on the spliced file inside `namespace ZarPrune.Cand`, `set_option autoImplicit false` at the top of the generated file (typo variables must not become universally quantified), hard timeout 240 s for the whole suite.
3. **Type check** — the harness wrapper `def gateInst : ZarPrune.Prune ZarPrune.Cand.target := candidate | candidate target` refers to `ZarPrune.Valid`, `HasKst`, `Params` by full name, so a candidate that redefines `Valid` in its own namespace (the Kimina/Vericoding "exploit the formalisation" move, lit. §6.3) simply fails to unify.
4. **Axiom audit** — `#print axioms gateInst ⊆ {propext, Quot.sound, Classical.choice}`; a `sorry` that slipped the scan shows as `sorryAx` (E4: two independent layers).
5. **Kill mask** — `#eval` of `gateInst.kill` on every case; the Lean mask, not the Python mirror, is what prunes; disagreement is logged as an artifact and the Python mirror is ignored.
6. **Admission** (new, off the inner loop): candidates that pass and remove work on the suite are re-checked with `leanchecker` on the built `.olean` (0.66 s; `--fresh` 29 s) and re-run against the *full* battery (all `w = z` tables plus the record profiles of Bhan et al.: `(11,21)`: rows `11^6 10^5`, cols `6^11 5^10`; `(12,22)`: rows `11^12`, cols `6^22`) before `zar_ub/lemmas.py` registers them as `evolved_<sha>`.

### 4.3 Lean's per-evaluation role in Loop A

Per evaluation the harness generates one Lean file per suite that (a) instantiates the chosen families with the plan's parameters (`def L_1 := ZarPrune.argD target …`), (b) folds them with `Prune.ofList`, (c) `#eval`s `survivors` (level-0/1 list) and the kill mask. No proof is elaborated beyond instantiation type-checking; failures here mean the plan's parameters were ill-typed (score 0, error in artifacts). Measured cost basis: 1.3 s for a four-instance suite in one process with 725-case masks (E5/E6 addendum), 1.8 s with the targeted Mathlib imports `Counting.lean` needs.

### 4.4 The `refuted` seam and encoding completeness

Now: `certify.py` — `tools/cadical --binary=false` DRAT → `drat-trim -L` (prints `s VERIFIED`) → `lrat-check` (prints `c VERIFIED`); a leaf is *certified* only if both succeed; LRAT kept under `cache/certs/<tag>/`; manifest JSON; sizes recorded (E9: 3.8 MB for 36 leaves at (9,9)).

Milestone M5 (Lean import): `Encode.lean` over `Std.Sat.CNF Nat` with the completeness direction only, `encode_complete : Valid P A → DoubleLexBlocks q A → profileOf A = q → (encode P q cuts).Sat (assign A)`, built from Lean core's verified Tseitin pieces plus hand proofs for the sequential counter (prefix-sum witnesses) and the one-directional AND indicators; then `refuted q := LRAT.check_sound proof (encode P q) (by native_decide)` in the harness domain; `exists_doubleLex` by strong induction on `Σ 2^{i+j} − pot A` (~250–400 lines, lit. §6.4 B2). The encoder is chosen for proof cost (sequential counter), not solver speed — the review's rule. Until M5 the closure report lists items 2–4 of its trusted base exactly as `closure.py` does today.

### 4.5 Conditional facts (`Cond.lean`; M3)

```lean
structure Fact where (m n s t z : Nat) (tag : String)   -- tag: provenance, carried into the report
def FactHolds (f : Fact) : Prop := ∀ B : Mat f.m f.n, ¬ HasKst ⟨f.m, f.n, f.s, f.t, 0⟩ B → weight B ≤ f.z
structure CondPrune (P : Params) (facts : List Fact) where
  name  : String
  kill  : Profile P.m P.n → Bool
  sound : (∀ f ∈ facts, FactHolds f) → ∀ A, kill (profileOf A) = true → ¬ Valid P A
def CondPrune.discharge (q : CondPrune P facts) (h : ∀ f ∈ facts, FactHolds f) : Prune P
```

`argDelCol P U` and `prefixI` become `CondPrune`s whose fact is the neighbour cell. The closure theorem for a cell is stated with the hypothesis list; facts proved by this project's own closure theorems are discharged by term application (a Lean DAG mirroring Kyoto's `zbounds` reading `data/`, lit. §3.1); external facts remain hypotheses and the report prints them with tags. Trust tiers the policy may select: `proved-here` ⊂ `tan2022` (the 159 bold cells + `(11,21)`, `(12,22)`) ⊂ `unreviewed-2026` (dfield/Hou/Afrasyab †-cells; never Padhi). A bound is *claimed* only if every fact is in `tan2022` or better; runs with `unreviewed-2026` facts are exploration and their reports say "conditional".

### 4.6 Loop B partial credit and the NL→Lean two-stage option

Partial credit (existing `GateResult.partial_credit`): 0 if the scan fails; else `0.15 + 0.5·(declarations without error / declarations) + 0.2·(first-error position / file length)`; `0.85` if everything elaborates but the wrapper does not type-check; `0.9` if typed but the axiom audit fails; capped at `0.95`; `1.0` only when accepted. It enters Loop B's score only through the `0.15·lean_partial` shaping term of the existing `evaluator.py`, so an unverified candidate can never outrank a verified one that removes work (Tier-A lesson A1, lit. §8.3). This document keeps that formula and adds the six-level status ladder `L0 parse-fails / L1 scan-fails / L2 elaborates-with-errors / L3 sorry-sketch (two-stage only) / L4 typed-but-axioms / L5 accepted` as a MAP-Elites axis for Loop B (`lean_status` raw 0–5).

Two-stage option (answering proposal §2.3's question with the review's evidence, lit. §7.2): not "NL then a separate autoformaliser" — the statement is fixed by the harness, so faithfulness is not the problem. Instead:

1. **Sketch call (reasoner)**: emit the NL counting argument, the executable `kill`, and a `sound` skeleton with typed `have … := by sorry` holes. The gate is run in *sketch mode* (`sorry` allowed, status L3, never trusted, never registered); Lean checks `kill` computes and every hole statement type-checks (<2 s).
2. **Fill calls (cheap model)**: 2–3 compile/fix rounds per hole with the exact error, position and goal text (Kimina/Goedel-V2 evidence: gains saturate in 2–5 rounds; error text is what matters).
3. **Escalate once** to the reasoner for re-decomposition, not retries.

OpenEvolve has no per-call routing, so this is implemented as a small driver `tools/forge.py` (outside the OpenEvolve loop) that uses the same gate; its outputs are registered exactly like Loop-B candidates. Verified-prefix fraction is used only as a tie-breaker among unverified sketches.

### 4.7 Defense in depth (beyond Lean)

* **Witness battery** on every kill: the `w = z` tables (E6) plus the record profiles; any kill of a realizable case is a hard zero and a `PIPELINE_BUG` artifact (E6 caught "kill any case with a row of sum ≥ 7").
* **Kill audit**: 5 % of lemma-killed leaves (seeded sample) are solved to completion by CaDiCaL; a SAT result there means either a Lean-vs-encoding mismatch or an encoding bug — hard zero, run halted, incident logged. This is the canary the review recommends after Afrasyab's retracted certificates (lit. §10 item 7).
* **Two LRAT checkers** (drat-trim, lrat-check) with different code bases; certificate truncation is detected (E9).

---

## 5. Reward function

### 5.1 Per-instance run and its outcome

For suite instance `I` (Section 8.2) and policy `π`, the harness runs the plan under budget `B_I = (T_I seconds wall, K_I conflicts total, 2 leaf workers)` and produces:

* `cover_ok ∈ {true, false}` (Lean list for level 0/1; LRAT for level 2);
* a leaf multiset with dispositions `KILL(family)`, `UNSAT(conflicts, certified?)`, `SAT(witness)`, `OPEN(cap reached)`;
* `cost_π,I` = harness CPU-seconds actually spent (Lean `#eval` + probes + solves + certification + cover checks), and `conf_π,I` = total solver conflicts spent.

### 5.2 Difficulty-weighted masses

With `D(ℓ)` the per-leaf difficulty of Section 6 (certified conflicts if refuted; calibrated estimate if killed or open):

```
W_kill = Σ_{ℓ ∈ KILL}  D̂(ℓ)          credited only if the kill is Lean-computed and not audited-SAT
W_sat  = Σ_{ℓ ∈ UNSAT} D(ℓ)          exact conflicts; certified leaves only (uncertified count as OPEN on train)
W_open = Σ_{ℓ ∈ OPEN}  max(cap_ℓ, D̂(ℓ))
settled_mass = (W_kill + W_sat) / (W_kill + W_sat + W_open)           ∈ [0,1]
closed       = cover_ok ∧ W_open = 0 ∧ no SAT leaf
```

Sub-cube credit rule (anti-gaming): the credited mass of the children of a parent leaf is capped at the parent's `D̂` (children's `D` are re-normalised to sum to at most `D̂(parent)` when all are settled, else proportionally). A policy cannot inflate `settled_mass` by splitting easy leaves into many trivially settled sub-leaves.

Reference mass for training cells is the table's total labelled work (`Σ true conflicts`, E10), so `settled_mass` on train equals "fraction of the known refutation work that this policy closed within budget".

### 5.3 Score

```
S_I = 0                                               if unsound (battery kill, audited-SAT kill, PIPELINE_BUG)
                                                       or ¬cover_ok or plan invalid/crashed/timed out
S_I = 0.5 · settled_mass_I                            if not closed              (∈ [0, 0.5))
S_I = 0.5 + 0.5 · σ( log2( cost_base,I / cost_π,I ) ) if closed                  (∈ [0.5, 1])
      with σ(x) = 0.5 + clip(x, −4, 4) / 8   (baseline policy scores 0.75; 16× faster → 1.0; 16× slower → 0.5)

combined_score = Σ_I ω_I · S_I / Σ_I ω_I,   ω_I = 1 for train cells, 2 for held-out cells, 3 for targets
```

`cost_base,I` is the cost of `initial_policy.py` measured once per suite build with the same seeds (cached in the table file). On target cells no policy closes, so the score is driven by `settled_mass` under the fixed budget — i.e. by killing or certifying the most *work*, which is exactly the thesis objective.

A SAT leaf on a target is not a failure of the policy: it is a lower-bound discovery. The harness verifies the witness with `has_kst`, writes it to `cache/witnesses/`, sets the instance's `w` to `witness+1` for future runs, and scores that run by `settled_mass` over the remaining leaves.

### 5.4 Secondary metrics (returned raw; OpenEvolve bins them)

`settled_mass`, `closed_fraction` (share of suite instances closed), `lemma_share = W_kill / (W_kill + W_sat + W_open)`, `log_leaves = log10(#leaves at level 0/1 after kills + 1)`, `max_depth`, `tail_share` (share of `W_open + W_sat` in the top decile of leaves — the AlphaMapleSAT statistic), `cost_seconds`, `lean_seconds`, `cert_bytes`, `cover_mode` (0 = Lean only, 1 = Lean + LRAT), `n_families_used`, `farkas_used`.

### 5.5 MAP-Elites feature dimensions

`feature_dimensions: ["lemma_share", "log_leaves"]`, `feature_bins: 8`. Rationale: the two axes are the two real design choices (how much is settled by arithmetic vs by search; how fine the split is); both are continuous raw values as `DatabaseConfig` requires; both are candidate-independent in meaning across instances. `tail_share` is the third axis to try in ablations. For Loop B keep `["proven_gain", "n_lean_decls"]` (existing) or `["proven_gain", "lean_status"]`.

### 5.6 Anti-cheating checklist

| attack | defence |
|---|---|
| policy "kills" leaves itself | kills only via Lean `#eval` of library terms; the plan cannot express a kill |
| drops leaves from the list | the list is Lean-computed from the plan's split; level-2 cover is LRAT-checked |
| asks for tiny budgets so cost is small | unclosed runs score < 0.5; closed requires every leaf certified |
| exploits the difficulty proxy on killed leaves | credit capped at the parent's reference difficulty; 5 % audit solves killed leaves to completion; on train the reference is the exact labelled work |
| uses `unreviewed-2026` facts to close easily | allowed only when `ledger_trust` says so; such runs are tagged conditional and get `ω_I` halved; claims require `tan2022` or better |
| imports pysat and solves inside `plan()` | its output is ignored; wall-clock cap on `plan()` is 20 s; anything beyond is a timeout (score 0) |
| vacuous plan (no families, no ladder) | valid but scores by `settled_mass` = 0 |
| non-determinism | fixed solver seeds and options; conflicts (not seconds) are the difficulty unit; common random numbers for sampling |

---

## 6. Branch difficulty measure

### 6.1 Definition

For a cube `c` (any level) of instance `P`, the **difficulty** `D(c)` is the number of conflicts CaDiCaL 1.9.5 (pysat `cadical195`, default options, seed 0, no preprocessing changes) needs to refute `F_P ∧ Sym_P ∧ c` with no cuts. It is a property of the branch, not of the policy (lit. §5.3: "all proxies are functions of the branch CNF alone"). Justification from data: unit propagations and conflicts are statistically indistinguishable as workload measures and wall-clock is worse (Chivilikhin et al., lit. §5.2); on 1,571 labelled cases the distribution has median 1,652, p90 20,315, max 221,874 conflicts (E10).

### 6.2 Algorithm

```python
LADDER = (2_000, 20_000, 200_000, 2_000_000)      # escalating conflict caps
def difficulty(inst, cube, cap_max, seed=0):
    """Returns Exact(k) | Censored(cap, estimate) | Witness(matrix).  Cost bounded by cap_max conflicts."""
    F = encode_leaf(inst, cube, cuts=())          # fixed base encoding + additions + cube units
    c2000 = None
    for cap in LADDER:
        if cap > cap_max: break
        r = solve_cnf(F, inst, solver="cadical195", conf_budget=cap, seed=seed)
        if c2000 is None: c2000 = r.conflicts     # first probe: the calibrated proxy
        if r.status == "unsat": return Exact(r.conflicts)
        if r.status == "sat":   return Witness(r.matrix)     # realizable: the cube is NOT empty
    return Censored(cap=cap, estimate=max(cap, calib(inst, c2000)))

def calib(inst, c2000):
    # log-linear map fitted on the training ladder (E10: Spearman 0.913 between c2000 and true conflicts):
    #   log D ≈ a_{s,t} + b_{s,t} · log c2000 + γ · log2_volume(inst, rows)
    # refit per (s,t) whenever new fully-labelled tables are built; clipped to [c2000, 50·cap_max]
    ...
```

Rules: (i) a leaf refuted at cap `k` is charged its exact conflicts, not the cap; (ii) escalation is resumed, not restarted, within one evaluation (pysat `solve_limited` on the same solver object keeps learned clauses), but the *reported* `D` is from a fresh solve at the reporting cap so it is reproducible; (iii) `cap_max` is per suite tier: 2·10^5 on train (all cases finish; E10 max 221,874), 2·10^6 on held-out, `K_I / #leaves` on targets, chosen so that a sampled 70 % quantile of leaves finishes (Heule's practice, lit. §5.3).

### 6.3 Difficulty of killed leaves

A killed leaf is never solved, so `D̂` must be estimated without candidate bias: (a) on train/held-out cells the table already holds the exact `D` of every level-0 case (probe-all tables, E6/E10) — use it; (b) on targets, the harness probes a seeded random sample of the killed leaves at 2,000 conflicts (0.03–0.05 s each) with common random numbers across candidates (Chivilikhin Thm 3 / §6.2 of the review: shared samples make candidate comparisons paired), applies `calib`, and scales to the killed set; (c) for level-2 children the parent's `D̂` bounds the total credit.

### 6.4 Cost of the measure

2,000-conflict probe ≈ 0.03–0.05 s at (9,9)–(12,12) sizes (E10: 1,571 probes in seconds); the whole ladder to 2·10^5 ≈ 3 s worst case per leaf; throughput ≈ 70k conflicts/s (11.8 M conflicts in ≈ 170 s CPU across the seven E10 tables). Certification adds ≈ 1× solve time plus ~50 ms per leaf for drat-trim/lrat-check at these sizes (E9: 36 leaves in 1.8 s total).

### 6.5 What is reported alongside the sum

`tail_share` (top-decile share of remaining work), the hardest ten surviving leaves with their profiles (these go into the prompt artifacts — the "painful-but-finite band" the curriculum literature recommends, lit. §5.3), and the critical-path difficulty `max_ℓ D(ℓ)` (AlphaProof's min-over-AND-subgoals view: a decomposition is as slow as its hardest leaf under parallelism).

---

## 7. SAT execution policy

### 7.1 When to solve

Order of operations per leaf, harness-enforced:

1. **Lemma masks first** (Lean `#eval`, ~1 ms/leaf): nothing is solved if a lemma kills it.
2. **Cheap probe** (2,000 conflicts) on every survivor: it settles ~60 % of leaves on the training ladder outright and is the difficulty proxy for the rest. This is the answer to "how often do we run the SAT solver versus relying on proxies": always run the 2k probe (it *is* the proxy, ρ = 0.91), and let the policy decide escalation.
3. **Policy ladder** (`solver_ladder`): escalating budgets; cuts and phase hints may be introduced at a step (the harness re-encodes with the registered cut clauses; the certificate then covers `F ∧ cuts`).
4. **Refine** on exhaustion: `refine(leaf)` may return a `SubSplit`; children go back to step 1 (lemmas re-applied to children whose profile is unchanged only if `LemmaUse.level == 2` — e.g. pair-cut-derived kills after codegree literals are fixed).
5. **Certify** each UNSAT leaf with the binary CaDiCaL + drat-trim + lrat-check *when the tier requires it* (train/held-out: always, so scores are certified; targets: certification runs after the evaluation, asynchronously, and the closure report is what it certifies — evaluation-time `UNSAT` from pysat is provisional and never appears in a claim).

### 7.2 Portfolio

Available in the venv: `cadical195` (default; proof-capable via `with_proof`), `cadical153`, `cadical103`, `glucose4`/`glucose42`, `kissat404` (no proof), `maplechrono`, `minisat22`. The policy composes a ladder from these; the harness caps `time_budget` per call at `T_I / 4` and total per leaf at `T_I / 2`. Diversity between CaDiCaL and Glucose is real but small at these sizes; the more valuable knob is *cuts* (pair cuts turn cardinality contradictions into short refutations) and *re-cubing* (fewer very hard leaves). Kissat 4.0.x emits DRAT only and needs `drat-trim -L` — supported by the same certify path if a binary is built later; not required.

### 7.3 Parallelism

Leaf-level: `parallel_leaves ≤ 2` per evaluation; OpenEvolve `parallel_evaluations: 2` → 4 solver processes on the laptop. Leaves are dispatched in the plan's order; `hard_first` (by `c2000`) maximises the chance that the budget is spent on what matters; `regular_first` targets near-regular profiles (where the frontier's survivors live, lit. §4). On a cloud run the same code scales `parallel_leaves` with cores; the difficulty unit (conflicts) is unaffected.

### 7.4 Certificates and storage

DRAT text (`--binary=false`) is deleted after LRAT extraction; LRAT kept, gzip'd when > 1 MB; manifest per instance with SHA-1 of every CNF (so a certificate is bound to an exact formula); expected volume: ~100 KB per leaf at (9,9), single-digit MB per leaf near (12,12) (E10 max 221k conflicts); a (16,18)-class campaign will be tens of GB (dfield: 25 GB) — kept off the repo under `cache/certs/`, listed in `.gitignore`, with the manifest committed.

---

## 8. Generalisation across (m, n, s, t)

### 8.1 What is parametric

`Params` is `(m,n,s,t,w)` everywhere: `Counting.lean` proves argA/argD/deletion/waterfill for general `s,t`; `encoding.py` handles general `s,t` (`aux` mode per `s`-subset of rows, `AtMost(t−1)`); the enumerator, ledger and difficulty calibration are keyed by `(s,t)`. A policy is a function of `inst`, so one policy is scored across cells; the prompt shows only the small ones ("less is more", Tao–Wagner generalizer mode, lit. §8.1).

### 8.2 Suite design (curriculum + hold-out)

| tier | cells (3,3) | `w` | role | `ω` |
|---|---|---|---|---|
| TRAIN | (9,9) 49, (9,10) 54, (10,10) 60, (10,11) 64, (11,11) 69, (11,12) 74, (12,12) 80 | z+1 | fully labelled (E10), certified in-loop | 1 |
| BATTERY | same cells at `w = z`; plus record profiles (11,21)/(12,22) | z | witnesses: unsound kills → 0 | – |
| HELD-OUT | (9,12) 64, (10,14) 77, (12,13) 86, (13,13) 92; and z_2 / z_4 cells for `(s,t) ≠ (3,3)` (Tan's tables) | z+1 | scored, never shown in the prompt | 2 |
| CALIBRATION targets | (10,20)=102, (11,21)=116, (11,22)=121, (12,22)=132 (exact; UNSAT at 103/117/122/133), (12,18) ≤ 108 (pinned column profile `{7,6^17}`, 96 row partitions, naive CaDiCaL > 16 min — Hou's uniqueness import does real work) | z+1 | first real closures with certificates | 3 |
| RE-PROOF targets (value known only from unreviewed 2026 sources †) | (9,23)=103, (10,21)=106, (10,22)=110, (10,23)=112, (11,19)=106, (11,20)=111, (11,23)=123, (12,17)=103 (Collins 2016 exact; † re-certified), (12,18)=108, (12,19)=114, (12,20)=120, (12,21)=126, (12,23)=134 | claimed+1 | independent Lean+LRAT proof of a †-cell is itself a contribution (lit. §10) | 3 |
| OPEN targets | (13,19..21), (13,23), (14,19..23), (15,19..23), (16,17..23); recommended order: (16,18) ≤ 139 (5,156 KST-surviving column partitions at w=140), (16,17) ≤ 132 (1.4 M pairs — needs the column-only split or better prunes), then m = 13 gaps 4–9; avoid (14,20), (16,19) | best UB (or LB+gap) | the thesis goal | 3 |

The band rule: keep instances in the suite whose closure rate across the population is in (0, 0.75] (Goedel/Seed curricula, lit. §5.3); promote/demote cells between tiers as policies improve.

### 8.3 Induction across cells (the ledger)

Closing `(m,n)` at `w` produces `Fact(m,n,s,t,w−1,"proved-here")`; `delCol`/`delRow`/`prefixI` at `(m,n+1)` and `(m+1,n)` immediately strengthen (Tan: "the decisive prune is Argument I fed with exact values of smaller cells", lit. §3.1; Hou: 99.99 % of row partitions killed at (12,18,109) by deletion). A `propagate` command re-scores every open cell's case count against the new ledger and re-orders the target queue. The mechanical claim checker (monotonicity, `z(m,n+1) ≤ z(m,n)+m`, witness comparison) runs on every ledger insert.

### 8.4 Transfer tests

* Policy evolved on TRAIN, scored on HELD-OUT without re-evolution: `S_heldout` vs baseline (report the gap).
* Transposition consistency: `S(m,n)` vs `S(n,m)` with the transposed policy (the Lean `Prune.transposed` combinator guarantees the lemmas agree).
* `(s,t) = (2,2)` and `(4,4)` cells from Tan's `z_2`, `z_4` tables: the same policy file, different `inst`; expect the split-side preference to change (for `(2,2)` the aux encoding collapses to a quadratic identity and cases are tiny).
* Ablation of `ledger_trust`: `proved-here` vs `tan2022` case counts (E1 vs E2 showed the table kills entire small cells; the policy should learn to use the ledger where legitimate).

---

## 9. Cost and time model; model choice

### 9.1 Measured unit costs (this machine: macOS arm64, Python 3.11, Lean 4.34.0)

| operation | cost | source |
|---|---|---|
| Lean elaboration of a Mathlib-free candidate + `#eval` mask (≤ 725 cases) | 0.3–0.5 s | E4 |
| whole suite in one Lean process (4 instances) | ~1.3 s | E6 addendum |
| with targeted Mathlib imports (`Counting.lean`) | 1.8–1.9 s | E5 |
| `leanchecker` replay of ZarPrune | 0.66 s (`--fresh` 29 s) | lit. §6.1 |
| `#eval` of `survivors` enumerator | ~1 ms/case (est.; measure in M2) | – |
| 2k-conflict probe | 0.03–0.05 s | E10 |
| full refutation, train ladder | median 1.6k, p90 20k, max 222k conflicts; ≈70k conflicts/s | E10 |
| certification per leaf (binary cadical + drat-trim + lrat-check) | ≈ solve time + 50 ms; (9,9): 36 leaves 1.8 s, 3.8 MB | E9 |
| one Loop-A evaluation (7 TRAIN + 4 HELD-OUT, `T_I` = 30/60 s caps, 2 workers) | 60–180 s wall | design target |
| one Loop-B evaluation (existing) | 5–9 s (1.5 s after gate batching) | E6 |

### 9.2 LLM token and price model

Prompt per Loop-A iteration: system message (~1.5k tokens) + current program (~1.5k) + 3 top + 2 diverse programs (~4k) + artifacts (≤ 16 KB ≈ 4k tokens) ≈ 11k input tokens; output 2–6k tokens (diff-based edits). Loop B: ~14k in, 4–10k out (Lean).

Prices per million tokens as recorded in the literature review on 2026-09-21 (OpenRouter; verify before each run): DeepSeek-V4-Flash $0.06/$0.11; GPT-OSS-120B $0.15/$0.60; GPT-5.4-mini $0.75/$4.5; Gemini 3.1 Pro $2/$12; Claude Opus 4.7 $5/$25. Per iteration (11k in / 4k out):

| model | Loop A per iteration | 1,000 iterations | Loop B per call (14k/8k) |
|---|---|---|---|
| DeepSeek-V4-Flash | ≈ $0.001 | ≈ $1 | ≈ $0.002 |
| GPT-OSS-120B | ≈ $0.004 | ≈ $4 | ≈ $0.007 |
| GPT-5.4-mini | ≈ $0.026 | ≈ $26 | ≈ $0.05 |
| Gemini 3.1 Pro | ≈ $0.07 | ≈ $70 | ≈ $0.12 |
| Claude Opus 4.7 | ≈ $0.155 | ≈ $155 | ≈ $0.27 |

The lower-bound predecessor spent $15–30 per case at ≈ $0.05–0.10 per iteration (lit. §8.2) — consistent with the mid rows. Cloud AlphaEvolve pricing is undisclosed; its contract (`scores`, insights, EVOLVE blocks, 30-min evaluation lock, no cascade) is byte-compatible with this evaluator's outputs, so the same `evaluator_policy.py` runs there with `cascade_evaluation: false`.

### 9.3 Model choice

* **Loop A**: a cheap ensemble — `deepseek/deepseek-v4-flash` (weight 0.6) + `openai/gpt-oss-120b` or `openai/gpt-5.4-mini` (0.4) — because plan edits are ordinary Python over a documented API, the evaluation (1–3 min) dominates wall-clock, and mixed ensembles add useful variance (lit. §8.1). Reasoning effort low. A real run: 500–2,000 iterations, $2–50.
* **Loop B / forge**: the sketch call goes to a reasoner (Gemini 3.1 Pro or Claude Opus 4.7; the UW study puts frontier general models at 86–92 % refine@32 on miniF2F, and no specialised prover on OpenRouter is usable or has seen Lean 4.34, lit. §7.2); hole-filling on a cheap model with compiler feedback (Goedel-Architect split, ≈ $0.44/problem vs $244 end-to-end on the reasoner). Budget O(10) sketches per lemma family.
* **Evaluation wall-clock vs LLM latency**: Loop A evaluation (60–180 s) > LLM call (10–60 s) — so `parallel_evaluations: 2` and cascade stage 1 (Section 10.1) matter more than model speed.

### 9.4 Budget rule for the $16

Hard stop at $14 spent (`experiments/cost.py` before and after every run appends to `experiments/cost_ledger.md`; the smoke scripts refuse to start if remaining < $2). Reserve: $4 Loop-A smoke, $6 Loop-B/forge pilot, $2 prompt-length calibration, $2 contingency.

---

## 10. Test plan

### 10.1 Without an LLM ($0)

**T1 — Plan API and harness unit tests** (`tests/test_plan.py`): schema validation (bad family name, bad params, ladder over cap → rejected); `LeafView` fields; deterministic leaf ordering; budget enforcement (a `plan()` that sleeps 30 s → timeout → score 0).

**T2 — Cover**: for every TRAIN/HELD-OUT cell, the Lean-computed `survivors` list equals the Python `enumerate_cases` list (same sorted tuples) in both `both` and `cols` modes; level-2 sub-splits on 20 random leaves produce LRAT cover certificates that verify; a deliberately dropped child fails the check.

**T3 — Hand-written policy set** (`policies/`), each scored by `evaluator_policy.py`, with expected orderings asserted:

| policy | expectation |
|---|---|
| `baseline.py` (= initial) | closes all TRAIN; `S_I = 0.75` by definition |
| `no_lemmas.py` | closes TRAIN slower; `lemma_share = 0`; `S_I < 0.75` |
| `argD_threshold.py` (threshold form instead of sorted) | same or fewer kills; never a battery hit |
| `cols_only.py` (dfield split) | fewer leaves, heavier leaves; measure the trade-off the review predicts |
| `recube_row_support.py` | lower `tail_share` at (11,11)/(12,12); cost ± |
| `pair_cuts.py` | fewer conflicts on the near-regular leaves; certificates still verify |
| `farkas_search.py` (small integer multiplier search) | kills a superset of argA∪argD kills; Lean masks agree |
| `ledger_unreviewed.py` | closes (12,18)-type cells conditionally; report tagged conditional |
| `cheat_claims_kill.py` (returns a fake kill list) | ignored → identical to `baseline` |
| `cheat_bad_cover.py` (asks for a sub-split then drops children) | cover fails → 0 |
| `cheat_zero_budget.py` | `S_I = 0.5·settled_mass` only |
| `cheat_kill_realizable.py` (family params chosen to kill a `w = z` case — cannot happen with verified families; test the audit path with an injected fake mask) | hard zero + `PIPELINE_BUG` |

**T4 — Stub-LLM full loop** (`tools/stub_llm.py`): an OpenAI-compatible HTTP server (the same `api_base` mechanism OpenEvolve already uses) that answers chat completions with SEARCH/REPLACE diffs drawn from a seeded mutation bank over the plan's parameters (perturb budgets, toggle families, change `side`, swap sub-split kinds, crossover between two programs from the prompt). This exercises checkpointing, islands, MAP-Elites binning of the raw features, artifact rendering and `openevolve-run.py` resume — at $0. Acceptance: 50 iterations complete; the best program's `combined_score` is non-decreasing across checkpoints; feature grid has ≥ 6 occupied cells.

**T5 — Regression against the literature**: the 1,650 recomputable Kyoto cases and the record profiles as an extended battery (a prune that kills a case in which Tan's ledger records a *solution* is unsound without any Lean call, lit. §3.1); DGH's reproduced closures with zero SAT calls (`z(13,17) ≤ 116`, `z(13,18) ≤ 121`) as the M3 acceptance test for `dgh4`.

**T6 — Certificates**: re-verify every stored LRAT with a fresh build of `lrat-check`; corrupt one byte → rejected.

### 10.2 With the $16

| exp | what | model | budget | measures |
|---|---|---|---|---|
| E11 | prompt-length calibration: dump the exact prompt for `initial_policy.py` (no call), count tokens | – | $0 | tokens in |
| E12 | Loop A smoke, 8 iterations, TRAIN only, `T_I` = 20 s | DeepSeek-V4-Flash | ≈ $0.05 | valid-edit rate, plan-schema failures, score trajectory, tokens/iteration |
| E13 | Loop A smoke, 10 iterations, mid model | GPT-5.4-mini | ≈ $0.5 | same; compare edit quality (does it touch `refine`/Farkas?) |
| E14 | Loop A, 20 iterations with (12,12) and (10,14) in the suite, `parallel_evaluations: 2` | Flash + mini ensemble | ≈ $1 | wall-clock per iteration, whether MAP-Elites cells diversify |
| E15 | Forge pilot: 6 sketch calls on 3 statements (`argD` threshold re-proof; DGH (4) with `v = s−1`; P19 thinning) + ≤ 3 fill rounds each on Flash | Gemini 3.1 Pro / Opus 4.7 sketches | ≈ $4–6 | L-status ladder per attempt, pass rate, cost per accepted lemma |
| E16 | one Loop-B run of 10 iterations with the existing evaluator on the ensemble | mini | ≈ $1 | reproduce E7's plan with the batched gate |

Each experiment writes `experiments/E1x_*/run.log`, a cost-ledger line before and after, and a LOG.md entry with the table above filled in. Nothing here is a "real run"; the aim is the numbers in Section 9 and a go/no-go on model tiers.

### 10.3 Evaluator cascade (OpenEvolve config for Loop A)

```yaml
# config_policy.yaml (Loop A)
max_iterations: 20            # smoke; real run 1000+
checkpoint_interval: 5
random_seed: 42
llm:
  api_base: "https://openrouter.ai/api/v1"
  models:
    - {name: "deepseek/deepseek-v4-flash", weight: 0.6}
    - {name: "openai/gpt-5.4-mini",        weight: 0.4}
  temperature: 0.6
  max_tokens: 8000
  timeout: 300
prompt:
  system_message: <the Plan API, the lemma library table, the reward statement, the no-cheat statement>
  num_top_programs: 3
  num_diverse_programs: 2
  include_artifacts: true
  max_artifact_bytes: 16384
database:
  population_size: 40
  archive_size: 15
  num_islands: 3
  elite_selection_ratio: 0.2
  exploration_ratio: 0.3
  exploitation_ratio: 0.6
  feature_dimensions: ["lemma_share", "log_leaves"]
  feature_bins: 8
  migration_interval: 10
evaluator:
  timeout: 600
  cascade_evaluation: true
  cascade_thresholds: [0.2]      # stage 1 must reach 0.2 to run stage 2
  parallel_evaluations: 2
  enable_artifacts: true
diff_based_evolution: true
max_code_length: 30000
```

`evaluate_stage1`: schema validation + one small cell ((9,9) 50, `T_I` = 10 s) → `combined_score` of that cell alone (≈ 10 s). `evaluate_stage2` = full suite. `evaluate_stage3` (targets) only when `ZAR_UB_TARGETS` is set.

---

## 11. Risks and failure modes

| risk | likelihood / impact | mitigation |
|---|---|---|
| Lean `#eval` of the survivor enumerator blows up on frontier cells (10^5–10^6 profiles before Argument I) | high on targets / high | Argument I prefix tables in the enumerator (verified via `PrefixSound`); `cols`-only split; chunked `#eval`; fall back to Python enumeration + Lean `decide` on a *hash* commitment (no) — rather: keep the Lean enumerator as the source of truth and accept minutes for targets (once per instance, cached, keyed by plan split parameters) |
| Level-2 cover certificates are large for `row_support` splits with C(n, r) children | medium / medium | children canonicalised within equal-sum blocks; certificate is propagation-only so LRAT is small; measure in M2 and cap child count (policy schema) |
| Heavy-tailed leaves make per-evaluation scores noisy | high / medium | conflicts not seconds; fixed seeds; common random numbers; `T_I` chosen so ≥ 70 % of leaves finish; report tail statistics; score `closed` policies by cost ratio with a clipped log |
| Difficulty proxy on killed leaves is gameable | medium / high | parent-capped credit, exact table values on train, 5 % audit solves, proxy refit only from harness-run probes |
| Encoding bug makes leaves UNSAT spuriously | low / catastrophic | witness battery (SAT tables), the 5 % audit, `has_kst` on every model, E1 round-trip on exact cells, later `Encode.lean` completeness (M5) |
| Unreviewed 2026 facts leak into claims | medium / reputational | provenance tags in `Fact`, `ledger_trust` tiers, claim rule (`tan2022` or better), never Padhi |
| Policy code executes arbitrary Python | certain / low–medium | subprocess with 20 s cap, no network expected (document; optionally `sandbox-exec` on macOS), results ignored except the returned plan |
| OpenEvolve requires an LLM endpoint | certain / low | `tools/stub_llm.py` (T4) |
| pysat proof output for assumptions-based solving is awkward | medium / low | certification always via the binary on explicit CNF + cube units |
| LLM cost overrun | low / high for the thesis | ledger checks before/after each run; hard stop at $14; smoke iteration counts ≤ 20 |
| Lean toolchain/Mathlib cache drift (v4.34.0 cache missing top-level `Mathlib.olean`, E5) | known / low | targeted imports only; `lake exe cache get` pinned; CI-style `lake build` in T1 |
| Cheap models never produce useful `refine`/Farkas edits | medium / medium | E12–E14 measure it; escalate Loop A's ensemble to mini/Pro for the real run; seed the population with the hand-written `policies/` (islands start diverse) |
| `native_decide` prohibition vs Lean LRAT import | design conflict | resolved by the two trust domains (Section 4.1): `_native` axioms only in harness refutation modules, recorded by name |
| macOS laptop CPU (no GPU is irrelevant; cores are) | certain / medium | 4 solver processes max; the cloud run is where targets get hours |

---

## 12. Milestones

| id | deliverable | acceptance | est. effort |
|---|---|---|---|
| M0 (done) | engine, gate, tables, certification, difficulty labels (E1–E10) | – | – |
| M1 | `zar_ub/plan.py`, `zar_ub/lemmas.py` (registry over `Counting.lean`), `zar_ub/policy_runner.py` (executes a plan under budget, Lean masks, probes, ladder, refine, certification), `evaluator_policy.py`, `initial_policy.py`, `policies/` set, unit tests T1/T3 | T3 table holds; baseline closes TRAIN with certificates in < 3 min | 1.5 weeks |
| M2 | cover in Lean: `Perm.lean` (`act`, `hasKst_act_iff`, `exists_sorted_act`), `Enum.lean` (verified sorted enumerator with prefix bounds), `Cover.lean` (`cover_upto_perm`, `upper_bound_of_cover_upto_perm`); level-2 LRAT cover; `cols`-only mode; T2 | Lean-computed lists equal Python lists on all cached tables; level-2 covers verify; axioms audited | 2–3 weeks |
| M3 | `Cond.lean` (`Fact`, `CondPrune`, `prefixI`, deletion as CondPrune, ledger DAG with provenance), `Farkas.lean` (`LinIneq`, `combo`), `dgh4` (DGH (4), `v = s−1`) ; mechanical claim checker; `propagate` command | DGH's zero-SAT closures `(13,17) ≤ 116`, `(13,18) ≤ 121` reproduced with Lean kills only; Farkas refutes `w = 133` at `(15,17)` for the whole cell | 2–3 weeks |
| M4 | stub-LLM full loop (T4); $16 smoke E11–E16; cost tables in LOG.md | 50 stub iterations; edit-validity and cost numbers recorded | 1 week (overlaps) |
| M5 | `Encode.lean` completeness (sequential counters, AND indicators, cuts), `exists_doubleLex`, Lean LRAT import in the harness domain; axiom ledger per bound | (9,9) ≤ 49 end-to-end in Lean with recorded `_native` axioms; then the TRAIN ladder | 3–4 weeks |
| M6 | calibration closures with certificates: (10,20) 103, (11,21) 117, (11,22) 122, (12,22) 133, (12,18) 109 | closure reports with `claim established: YES` and `tan2022`-only facts | 1–2 weeks compute |
| M7 | re-proof targets: (9,23) 104, (10,21..23), (11,19), (11,20), (11,23), (12,17), (12,19..21), (12,23) — independent Lean+LRAT verification of †-cells | each closed cell: report + certificates + ledger insert (`proved-here`) | cloud budget |
| M8 | open targets on a cloud budget: (16,18) ≤ 139 first, then (16,17) ≤ 132, m = 13 cells; lower-bound machinery run in parallel as a soundness canary | any new bound, or a measured account of why not (tail statistics, cost) | cloud budget |
| M9 | thesis: ablations (islands, feature dims, cuts, re-cubing, `cols` vs `both`), multi-run statistics, per-phase cost, invalid/crash rates, axiom ledgers — the reporting the predecessors omitted (lit. §10 item 8) | – | 3 weeks |

### Appendix A — new/changed files

```
zar_ub/plan.py            Plan/SplitSpec/LemmaUse/SubSplit/SolverCall/LeafView (fixed API)
zar_ub/lemmas.py          registry of verified families (Lean term, schema, Python mirror, doc line)
zar_ub/cuts.py            verified cut families (pair_cut, min_row_cut) + encoders
zar_ub/decompose.py       level-0 lists from Lean, level-2 sub-splits, cover certificates
zar_ub/policy_runner.py   executes a Plan under budget; masks, probes, ladder, refine, certify, accounting
zar_ub/difficulty.py      + ladder/calibration (Section 6)
zar_ub/encoding.py        counters as base variables; cube = literal list; cuts hook
zar_ub/ledger.py          Fact with provenance, tiers, claim checker, propagate
evaluator_policy.py       Loop A evaluator (stage1/stage2/stage3)
initial_policy.py         Section 2.2
config_policy.yaml        Section 10.3
policies/*.py             hand-written and adversarial policies (T3)
tools/stub_llm.py         $0 OpenAI-compatible mutation server (T4)
tools/forge.py            two-stage sketch/fill driver for Loop B (Section 4.6)
lean/ZarPrune/Perm.lean, Enum.lean, Cover.lean, Cond.lean, Farkas.lean, DGH.lean, Encode.lean (M2–M5)
```

### Appendix B — the prompt's fixed no-cheat paragraph (Loop A system message excerpt)

"You edit `plan()` only. The harness executes your plan and measures everything: kills are computed in Lean from proved lemma terms; the case list is a Lean theorem; every surviving leaf is refuted with a checked certificate or counted as open; budgets are enforced by the harness. Claims, prints, or solver calls inside your code are ignored. You are rewarded for closing more certified refutation work, weighted by measured difficulty, within the budget — killing a few hard leaves is worth more than killing many easy ones. The artifacts list the hardest surviving leaves (their row and column sums and conflict counts), which lemma families fired, and where the budget went."

# Experiment log — Zarankiewicz upper bounds via evolved, Lean-verified prunes

Every experiment gets an ID, a date, a question, what was run (exact command
or script), the result, and what we concluded.  Nothing is deleted; failed or
superseded entries are marked as such.  Cost of LLM calls is tracked in USD.

Environment: macOS arm64, Python 3.11 (repo `.venv`), python-sat 1.8 (CaDiCaL
1.9.5), Lean 4.34.0 (no Mathlib), cadical + drat-trim binaries built under
`tools/` (gitignored), OpenRouter key with a $16 hard budget.

---

## E1 — 2026-09-21 — Does the Tan-style case engine reproduce known values?

**Question.** Does `zar_ub` (partition generator = Tan Algorithm 1 with
Arguments A and I, baseline prunes = deficit/mismatch/caps/Argument D both
orientations, CNF = fixed sums + K_{3,3} aux encoding + double-lex within
equal-sum groups, CaDiCaL 1.9.5 via pysat) reproduce exactly known z(m,n;3,3)?

**Run.** Inline script (see chat transcript); tables cached under `cache/`.
For each exact cell (m,n,z): instance w=z+1 must have every case pruned or
UNSAT; instance w=z must have >= 1 SAT case with an independently checked witness.

**Result.**
| cell | w | row parts | col parts | cases | baseline-killed | survivors | statuses | total conflicts | time |
|---|---|---|---|---|---|---|---|---|---|
| (5,5) | 21 | 0 | 0 | 0 | – | 0 | – | 0 | 0.0s |
| (5,5) | 20 | 1 | 1 | 1 | 0 | 1 | sat 1 | 0 | 0.0s |
| (6,6) | 27 | 0 | 0 | 0 | – | 0 | – | 0 | 0.0s |
| (6,6) | 26 | 2 | 2 | 4 | 3 | 1 | sat 1 | 61 | 0.0s |
| (7,7) | 33 | 2 | 2 | 4 | 0 | 4 | unsat 3, sat 1 | 1071 | 0.0s |
| (9,9) | 50 | 5 | 5 | 25 | 15 | 10 | unsat 10 | 6662 | 0.1s |

All witnesses passed the independent `has_kst` check.

**Bug found and fixed.** The prefix (Argument I) check was applied at full
length k = n, so the target cell's own table value bounded itself (circular:
at (7,7) w=34 every partition died because the table says z(7,7)=33).  Fixed:
Argument I is applied to proper minors only (`partitions.py`).

**Conclusion.** Engine is sound on the tested cells and fast.  With the external
exact table switched on, small cells are killed entirely by Argument I using
neighbouring exact values, which is legitimate but useless as a training
environment for *new* prunes.  Hence E2.

## E2 — 2026-09-21 — "Pure" mode (no external exact values)

**Question.** How large / hard are the case tables when Argument I may use only
the first-principles counting bound (everything then provable inside ZarPrune)?

**Run.** `build_table(..., use_table=False)`, conf cap 20000, 60 s/case.

| cell | w | row parts | col parts | cases | baseline-killed | survivors | statuses | total conflicts | top-5 conflicts | time |
|---|---|---|---|---|---|---|---|---|---|---|
| (7,7) | 34 | 1 | 1 | 1 | 0 | 1 | unsat | 167 | 167 | 0.0s |
| (8,8) | 43 | 1 | 1 | 1 | 0 | 1 | unsat | 144 | 144 | 0.0s |
| (9,9) | 50 | 6 | 6 | 36 | 19 | 17 | unsat 17 | 15457 | 2880 2003 1859 1820 1083 | 0.2s |
| (9,10) | 55 | 9 | 5 | 45 | 23 | 22 | unsat 22 | 18905 | 2240 1866 1538 1409 1309 | 0.3s |
| (10,10) | 61 | 5 | 5 | 25 | 15 | 10 | unsat 10 | 9865 | 2643 1464 1406 1138 1118 | 0.2s |

**Conclusion.** Argument D (baseline) already kills ~half the cases.  Survivors
are refuted in <3k conflicts each at this size; these cells are *training*
instances (cheap, fully labelled), not targets.  Per-case difficulty
(conflicts) is heavy-tailed even here, which is what the reward will weight.

## E3 — 2026-09-21 — Training-ladder tables (background)

**Run.** `experiments/build_training_tables.py` -> `experiments/E3_training_tables.log`.
Pure-mode tables at w=z+1 (all-UNSAT environment) and w=z (SAT cases with
witnesses = counterexample battery) for
(9,9),(9,10),(10,10),(10,11),(11,11),(9,12),(11,12),(12,12),(10,14),(12,13),(13,13).
Results appended below when finished.

## E4 — 2026-09-21 — Lean gate timing and adversarial self-test

**Question.** Is a per-candidate Lean check fast enough for an evolutionary
inner loop, and does the gate reject the obvious cheats?

**Run.** `zar_ub/lean_gate.py` self-test (chat transcript), instance (9,9,50),
36-case kill mask evaluated by `#eval` inside Lean.

| candidate | scan | compiled | typed | axioms | ok | partial credit | time |
|---|---|---|---|---|---|---|---|
| re-proved `deficit` (general P) | ok | yes | yes | propext, Quot.sound | **yes** | 1.0 | 0.39s |
| same with `sorry` | **rejected by scan** | – | – | – | no | 0.0 | 0.0s |
| kill-everything with a bogus `decide` | ok | **no** (free-variable error) | no | – | no | 0.32 | ~0.4s |
| instance-specific `Prune.or (deficit target) (rowCap target)` | ok | yes | yes | propext, Quot.sound | **yes** | 1.0 | 0.28s |

Also observed: a `sorry` that slips past the scan still fails the axiom audit
(`sorryAx` appears in `#print axioms`, and Lean refuses to `#eval` a term that
depends on sorry) — the gate has two independent layers.

**Conclusion.** ~0.3–0.5 s per candidate including the Lean-side kill mask:
Lean is *not* the bottleneck. The Lean-computed mask (not the Python
prototype) is what the pipeline uses, so Python/Lean disagreement cannot
silently prune.

## E6 — 2026-09-21 — Evaluator end-to-end (no LLM)

**Question.** Does the OpenEvolve evaluator (`evaluator.py`) produce the right
signals on (a) the initial program, (b) a disguised symmetry break, (c) an
empirically sound but unproven prune?

**Run.** `python evaluator.py <program>` with suite TRAIN = (9,9),(9,10),(10,10),
(10,11),(11,11) at z+1 (pure), BATTERY = same at z.

| program | sound_battery | lean_ok | proven_gain | empirical_gain | combined | time |
|---|---|---|---|---|---|---|
| initial (baseline prunes re-listed) | 1 | 1.0 | 0 | 0 | 0.15 | 9.2s |
| "kill any case with a row of sum >= 7" (unsound) | **0** | 1.0 | 0 | 0 | **0.0** | 7.1s |
| Python Argument D, Lean unchanged | 1 | 1.0 | 0 | 0 (see below) | 0.15 | 4.2s |

The unsound program was caught by the battery with an explicit counterexample
(rows=cols=[7,6,6,5,5,5,5,5,5] at (9,9,49) is realizable) — exactly the
"pruning vs adding" failure the Lean README warns about.

**Design correction.** (c) scored 0 empirical gain because the *Python*
reference baseline in `cases.py` already applied Argument D when the tables
were built, while the *Lean* baseline only has deficit/mismatch/caps. The
scoring baseline must be exactly what is proven in Lean, otherwise proving
Argument D — the natural first milestone — earns nothing. Fixed: tables now
probe every case (`probe_all=True`); the evaluator scores all probed cases;
the `baseline` field is informational. Tables rebuilt (E3 re-run).

**Timing.** ~5–9 s per evaluation, dominated by five Lean gate runs whose
`#eval` kill-mask lists have up to 625 cases; acceptable for the inner loop.

## E5 — 2026-09-21 — Does Lean take too long to compile? Mathlib vs Mathlib-free gate

**Question** (proposal §2.3, first question). Per-candidate Lean check cost.

**Run.** `experiments/E5_mathlib/setup.sh` (throwaway `lake new ... math`
project pinned to Mathlib `v4.34.0`, oleans via `lake exe cache get`, 7.9 GB).

| gate flavour | file | wall time per check |
|---|---|---|
| Mathlib-free ZarPrune, standalone candidate (`lake env lean`) | `Candidates/cand_*.lean` | 0.3–0.5 s |
| Mathlib-free, whole suite (4 instances, up to 725-case `#eval` masks) in one process | same | ~1.3 s |
| `import Mathlib` (monolith) | Probe.lean | **not testable**: the v4.34.0 cache is missing the top-level `Mathlib.olean` (CI build of that tag failed part-way; 8546 module oleans present) |
| targeted imports: `Finset.Basic`, `BigOperators.Group.Finset.Basic`, `Nat.Choose.{Basic,Sum}`, `Tactic.{Linarith,Positivity}`, `Fintype.Card`, `Finset.Powerset` | Probe2.lean | **1.79–1.86 s** (3 runs), proofs using `Finset.sum_le_sum`, `card_powersetCard` elaborate fine |

**Conclusion.** Lean is not the bottleneck either way (an LLM generation takes
30–120 s). Decision: keep the trusted core Mathlib-free, but add a Mathlib-backed
`ZarPrune/Counting.lean` (double counting for Arguments A/D and deletion
lemmas) and let candidates use targeted Mathlib imports through `import ZarPrune`.
Rationale: LLMs know Mathlib's `Finset`/`Nat.choose` API; a hand-rolled subset
counting layer would make every evolved proof harder for no soundness gain
(Mathlib is the community-reviewed standard and appears in `#print axioms`
only through propext/Quot.sound/Classical.choice).

**E6 addendum.** After restructuring the gate to check all suite instances in
one Lean process, a full evaluation of the initial program dropped from 9.2 s
to 1.5 s (4 instances; the 5th table was still rebuilding).

## E7 (prep) — 2026-09-21 — What will the LLM actually see?

Dumped the exact OpenEvolve prompt for the initial program with its evaluator
metrics and artifacts (no LLM call, $0): `experiments/E7_llm_smoke/prompt_dump_initial.md`.
Cost ledger started: `experiments/cost_ledger.md` (queries OpenRouter's free
usage endpoint before/after each run). Smoke runner: `experiments/E7_llm_smoke/run_smoke.sh`.

## E8 — 2026-09-21 — Lean counting core (Arguments A, D, deletion) via a proof-engineering workflow

**Question.** Can the classical counting arguments be proved in Lean 4.34 +
targeted Mathlib over the ZarPrune definitions, so the search starts from a
verified library of the known arguments (and so "prove Argument D" is a
concrete first milestone the evaluator can reward)?

**Run.** Workflow `zar-lean-counting-core`: 8 independent proving agents
(3 × Argument A, 3 × Argument D, 2 × deletion lemma), integrator, adversarial
reviewer, and a semantic mask check (`experiments/E8_counting/check_masks.py`:
Lean `#eval` kills vs Python reference kills on random profiles). Spec:
`lean/COUNTING_SPEC.md`. Results appended below when finished.

## E9 — 2026-09-21 — Certified refutation of survivors (DRAT -> LRAT -> lrat-check)

**Question.** Can every surviving case be refuted with a machine-checkable
certificate, so the final bound rests on Lean (prunes) + checked LRAT (cases),
never on a solver's bare "UNSAT"?

**Run.** `zar_ub/certify.py`: cadical (built from source, `tools/`) with
`--binary=false` DRAT output, `drat-trim -L` to check + emit LRAT, `lrat-check`
as an independent checker. Instance (9,9,3,3,50), pure mode, all 36 cases.

| cases | certified | sat | timeout | failed | wall | total LRAT |
|---|---|---|---|---|---|---|
| 36 | **36** | 0 | 0 | 0 | 1.8 s | 3.8 MB |

Sanity: a SAT case at (9,9,49) is reported `sat`, never certified; a truncated
LRAT file is rejected by lrat-check (exit 1). First version matched the wrong
success string (`lrat-check` prints `c VERIFIED`, `drat-trim` prints `s VERIFIED`) — fixed.

**Conclusion.** z(9,9;3,3) ≤ 49 is now established by: Argument A + counting
bound prefixes (case generation, pure mode: no external facts) + 36 LRAT
certificates. Remaining trusted components: the CNF encoding (fixed sums,
K_{3,3} aux clauses, double-lex) and the partition generator's completeness —
both are documented in `docs/design.md` as the trusted base; formalizing them
is future work (Codel–Avigad–Heule style verified encodings).
CLI: `python -m zar_ub certify 9 9 3 3 50 --pure [--lean cand.lean]`.

## E10 — 2026-09-21 — What is the "difficulty" of a branch, and what predicts it?

**Question** (proposal §2.3). Define per-case difficulty and find cheap proxies.

**Run.** `experiments/E10_difficulty/run.py`: for the 7 pure-mode TRAIN tables
(w = z+1: (9,9),(9,10),(10,10),(10,11),(11,11),(11,12),(12,12)) every case that
was open at the 20 000-conflict cap was re-solved to completion (cap 5M
conflicts / 240 s). All 1571 cases are now labelled UNSAT with their exact
CaDiCaL conflict count (tables updated in place). Then cheap proxies were
compared to the true cost by Spearman rank correlation.

| cell (w) | cases | reopened | max conflicts | wall |
|---|---|---|---|---|
| (9,9) 50 | 36 | 0 | 2 880 | 0.4 s |
| (9,10) 55 | 45 | 0 | 2 240 | 0.5 s |
| (10,10) 61 | 25 | 0 | 2 643 | 0.4 s |
| (10,11) 65 | 195 | 0 | 10 079 | 4.1 s |
| (11,11) 70 | 625 | 15 | 74 130 | 25.5 s |
| (11,12) 75 | 420 | 51 | 61 778 | 41.8 s |
| (12,12) 81 | 225 | 93 | 221 874 | 94.2 s |

Distribution over all 1571 cases: median 1 652 conflicts, p90 20 315, max
221 874, total 11.8 M. **The top 10 % of cases carry 61.6 % of the work** —
difficulty is heavy-tailed, so a prune that kills a few hard profiles is worth
more than one that kills many easy ones. This is why the reward weights killed
cases by cost, not by count.

| proxy | Spearman ρ vs true conflicts | cost |
|---|---|---|
| conflicts at a 2 000-conflict cap | **0.913** | ≤ 0.05 s/case |
| log2 volume of the row-profile class Σ log2 C(n, r_i) | 0.617 | free |
| number of clauses | 0.566 | free |
| conflicts at a 200-conflict cap | 0.137 | free |
| root unit-propagation fraction | undefined (constant 0 — fixed sums propagate nothing at the root) | free |

**Argument D as a milestone.** The reference (Python) Argument D kills 25.4 %
of total work on this suite (mean 3 511 conflicts for killed cases vs 12 324
for survivors): the cases it kills are the *easy* ones. Proving it in Lean is
therefore worth proven_gain ≈ 0.25 — a clear first rung — while the hard
profiles (near-regular: rows/cols all 6s and 7s at (11,11)–(12,12)) need
genuinely new arguments.

**Conclusion.** Difficulty := solver conflicts to refute (exact where known,
else a capped estimate); the short-budget run (2k conflicts) is the proxy to
use for open cells where full solving is infeasible (ρ = 0.91). Chivilikhin-style
sampling estimators were not needed at this scale.

### E8 results (2026-09-21, workflow `zar-lean-counting-core`, 11 agents, 37 min)

All eight independent attempts were sorry-free and axiom-clean; the integrator
merged the best pieces into `lean/ZarPrune/Counting.lean` (642 lines, 13
targeted Mathlib imports; `lake build` 3.1 s; `lake env lean` on the file 1.9 s).

| item | status | axioms |
|---|---|---|
| `sumFin_eq_sum`, `support`/`card_support`, `hasKst_of_subsets` | proved | propext, Classical.choice, Quot.sound |
| `budget_general` (one generic double-counting lemma) | proved | same |
| `colBudget` (**Argument A**), `rowBudget` | proved | same |
| `rowLocalBudget` (**Argument D**) | proved | same |
| `hasKst_transpose`, `Prune.transposed` (any prune on Pᵀ is a prune on P) | proved | propext, Quot.sound |
| `argA`, `argAT`, `argD` (exact "r lightest columns" form via layer-cake, no sorting), `argDT` | proved prunes | same |
| `deleteCol/Row`, `weight_delete*`, `not_hasKst_delete*` | proved | propext |
| `waterfillBound`, `sum_le_waterfillBound` (**waterfilling optimality**), `weight_le_waterfill` | proved | same |
| `argDelColWF`, `argDelRowWF` (delete the lightest line vs. the waterfilled bound of the smaller instance), `argWF` | proved prunes | same |
| `counting` = all seven folded | proved | same |

Independent audit by me (`#print axioms` on 14 items via `lake env lean`, 1.8 s):
identical axiom sets. Adversarial reviewer: no trivialised definitions, no
weakened statements, `counting` is a superset of the spec. Semantic check:
Lean `#eval` kills == Python reference kills for argA/argAT/argD/argDT on
3 × 600 random profiles each (0 mismatches) and on all 13,903 cached cases
(argD 5208/5208, argDT 2777/2777); no Lean prune kills any SAT-witnessed case;
`waterfillBound` == `max_sum_under_budget` on 2304 (m,n,s,t) tuples.

**Bug found by the verifier and fixed in `lean_gate.py`**: a raw `#eval` of a
long `List Bool` goes through Lean's pretty-printer, which wraps and truncates
with `⋯`, so kill masks with hundreds of cases could be silently short (the gate
then reported a length mismatch rather than a wrong mask — fail-safe, but it
would have blocked credit). Masks are now printed as one `IO.println` string.

Evaluator on `experiments/E8_counting/counting_program.py` (Lean candidate =
`Prune.or (baseline P) (counting P)`, Python mirror faithful to Lean incl.
`Nat.findGreatest` waterfilling):

| program | proven_gain | empirical_gain | lean_ok | agreement | combined | eval time |
|---|---|---|---|---|---|---|
| baseline (deficit/mismatch/caps) | 0.000 | 0.000 | 1.0 | 1.0 | 0.150 | 3.5 s |
| **full counting library** | **0.337** | 0.337 | 1.0 | 1.0 | **0.437** | 5.0 s |

Per instance work removed (proven): (9,9) 0.20, (9,10) 0.32, (10,10) 0.50,
(10,11) 0.39, (11,11) 0.28. So the classical arguments remove about a third of
the SAT work; the remaining two thirds — near-regular profiles — is what the
evolutionary search must attack with *new* arguments.

**Decision.** The full library becomes the default starting program
(`initial_program.py`); the bare baseline is kept as
`initial_program_baseline.py` for a "can evolution rediscover Argument D?"
control experiment.

## E7a — 2026-09-21 — What does one LLM response to the real prompt look like? (first paid calls)

**Run.** `experiments/E7_llm_smoke/probe_one.py`: builds the exact OpenEvolve
prompt (system message from `config.yaml`, the current program, its metrics
and artifacts: ~6.5k prompt tokens), makes ONE chat call on OpenRouter, applies
the SEARCH/REPLACE diff, and scores the child with the evaluator.
Parent = full counting library (combined 0.437, proven_gain 0.337).

| model | reasoning | max tokens | wall | tokens out (reasoning) | cost | outcome |
|---|---|---|---|---|---|---|
| deepseek/deepseek-v4-flash | default | 12 000 | 111 s | 12 000 (12 000) | $0.004 | **no output**: spent the whole budget thinking (finish_reason=length). The trace correctly analysed Argument D's soundness and derived a summed-over-rows global inequality Σ_j c_j·C(c_j−1,2) ≤ m(t−1)C(m−1,2), but never emitted code. |
| deepseek/deepseek-v4-flash | low | 16 000 | 377 s | 16 000 (15 999) | $0.002 | **no output** (ignores the effort hint). |
| openai/gpt-5.6-luna | low | 16 000 | 9.5 s | 1 180 (516) | $0.003 | diff applied, Lean gate OK, **no-op**: added `Prune.transposed (counting P.transpose)`, which the bundle already contains; child score identical (0.437). |
| anthropic/claude-sonnet-5 | medium | 20 000 | 230 s | 20 000 (19 710) | $0.218 | **truncated**: reasoned through a *two-column deletion* prune (genuinely stronger: weight ≤ c₁+c₂+waterfill(m,n−2)) and judged the Fin re-indexing too risky, then retreated to a "safe" transposed waterfill `argWFT` (profile-independent, kills nothing on enumerated cases) and ran out of tokens mid-diff. |

Spend so far: $0.24 of $16.

**Observations.**
1. Reasoning models need a large completion budget (≥ 30k tokens) or they never
   reach the code; `reasoning.effort` is not honoured by every provider.
2. Given a small budget, capable models choose *safe no-ops* (transposes of
   symmetric bundles) over substantive but risky proofs. The reward already
   gives partial credit for near-miss Lean, but the prompt did not say so
   forcefully; fixed (config.yaml: "a change with no new mathematical content
   earns nothing; attempt a substantive argument; write the code").
3. The models' mathematical instincts are right (summed Argument D, double
   deletion, pair double counting) — the bottleneck is Lean engineering
   effort per attempt, which argues for (a) large max_tokens, (b) a richer
   lemma library (e.g. a proved `deleteCols` for k columns, Fin re-indexing
   helpers) so that new prunes are short compositions, and (c) letting the
   evaluator's artifacts carry the exact Lean error so the next generation can
   repair rather than restart.
4. Cost model: ~$0.003 (luna) to ~$0.3 (sonnet-5, 30k budget) per generation;
   a 200-generation run costs $1–$60 depending on the model mix.

| anthropic/claude-sonnet-5 | low | 30 000 | 23 s | 2 013 (1 306) | $0.038 | diff applied, Lean untouched: added a **Python-only** "delete the two heaviest columns" prune (sound, but the *heaviest* pair is the weakest choice; never fires on the tables) — score unchanged. |
| google/gemini-3.8-flash | low | 30 000 | 10 s | 1 210 (0) | $0.010 | a genuinely new prune (deletion bounded by the *row-side* waterfill via `Prune.transposed`), Python mirror sound (empirical 0.337), **Lean failed** on one trivial slip (`Prune.ofList [..]` without `P`); partial credit 0.45. Under the original reward the child scored 0.135 vs parent 0.437: a compile slip zeroed *all* proven credit. |

**Reward fix (from the gemini probe).** Proven credit now has a floor: when a
candidate's own Lean fails, the evaluator scores the *proved library*
(`baseline ∪ counting`, masks cached per table) as the proven part, so a compile
slip on top of a good idea is a penalty (child ≈ 0.35) rather than a cliff
(0.135), and near-miss children stay competitive enough to be repaired in the
next generation. The exact Lean error is in the `lean_errors` artifact.
| anthropic/claude-sonnet-5 | high | 40 000 | 450 s | 40 000 (40 000) | $0.418 | **no output**: 40k tokens of reasoning. The trace is the most informative of all: it (i) derives the k-row generalisation of Argument D and correctly concludes it is *not profile-computable* (needs row-intersection data), (ii) shows the summed-over-rows Argument D is *equivalent* to Argument A, (iii) settles on two-column deletion vs the (m,n−2) waterfill bound, then times out. |

E7a total spend: $0.70 of $16 (ledger: `experiments/cost_ledger.md`).

**Conclusion of E7a.** (1) Reasoning effort must be capped hard (`low`) and
the completion budget must be large (≥ 30k) or the model never emits code; the
`reasoning.effort` hint is ignored by DeepSeek. (2) Every model converges on
the same three profile-level ideas (double deletion, k-row budgets, summed
budgets), two of which are provably no stronger than the library and one of
which is weak; this is evidence that *profile-only* prunes are close to
exhausted after Arguments A/D — the remaining near-regular profiles need either
cross-side/Farkas combinations (the schema channel of `docs/design.md` §2.3) or
a richer case index (design §12 M10). (3) The reward floor for failed Lean
(now the proved library) is essential: Gemini's correct idea with a one-token
slip would otherwise have been discarded.

## Design document — 2026-09-21

`docs/design.md` (721 lines) synthesised by the workflow from three candidate
designs (`docs/design_candidates/`, two judge reports). Spine: certificate-first
(a bound is one Lean theorem `Closures/Z_m_n_w.lean` with per-fact hypotheses and
LRAT-refuted survivors). Milestones M0 (done: E1–E10) … M11. Next: M1 (evaluator
v2 + gate hardening), M2 (`Cond.lean` + provenance ledger), M3 (`Closure.lean`),
M4 (promotion + closure daemon), M5 (schemas + hand-proved DGH(4)).

## E11 — 2026-09-21 — Calibration cells with known answers at real size (cited mode)

**Question.** With Tan's neighbouring exact values allowed as cited facts (Argument I on proper minors), how much survives on frontier-sized cells whose answer is known, and can it be certified?

**Run.** `experiments/E11_calibration/run.py` (tables, cap 2000) then LRAT certification of every case (`certify.log`).

| cell | w | mode | row parts | col parts | cases | ref-D would kill | certified | wall | external facts used |
|---|---|---|---|---|---|---|---|---|---|
| (10,20) | 103 | pure | 26 | 5 | 130 | 63 | – | – | none |
| (10,20) | 103 | cited | 3 | 2 | **6** | 2 | **6/6** (LRAT 1–29 MB each) | 9 s | 17 (Tan 2022 minors) |
| (11,21) | 117 | pure | 37 | 1 | 37 | 28 | – | – | none |
| (11,21) | 117 | cited | 0 | 0 | **0** | – | nothing to solve | 0 s | 14 |

**Conclusion.** Conditional on Tan's cited minors: z(10,20;3,3) ≤ 102 (6 LRAT
certificates) and z(11,21;3,3) ≤ 116 (zero cases — pure arithmetic), both
matching the known exact values [Tan; Bhan et al. for the (11,21) lower bound].
This is the "closure with facts as hypotheses" path of design §1.1/§8.2 in
action before `Closure.lean` exists (Tier-2 report now; Tier-1 once the
closure theorem is generated). Edge-of-table cells are dominated by Argument I;
the interior (near-square) cells are where prunes must do the work.

## Build note — 2026-09-21 22:25

The first M1–M5 build workflow (9 builders + integrator + 2 verifiers) was
killed after ~10 min by the agent session limit before any file was written
(tree verified intact: tests pass, evaluator unchanged). Relaunched as two
batches: batch 1 = gate hardening, reward v2 + evaluator, tables/suite/
difficulty, `Cond.lean` + ledger, no-LLM test bank (+ integrator); batch 2 =
Farkas schemas, DGH(4), `Closure.lean`, promotion/daemon (+ integrator +
verifiers).

## E12 — 2026-09-21 — Difficulty calibration (§6.2), GEN cells, proved-library census (batch 1, C-tables)

**Question.** Can the censored-label estimator `f̂ = exp(a + b·log c2000 + g·log2_volume)` order the
hard tail (acceptance §6.2: Spearman ρ ≥ 0.85 on the hold-out `(12,13,87)`), and what does the
proved library leave on every cached table?

**Run.** `python -m zar_ub calibrate --holdout 12,13,87` (hold-out first deepened to 200k conflicts:
103/104 resolved; 19 s with 6 jobs); `python -m zar_ub baseline --all`; GEN tables built with
`python -m zar_ub table 7 7 2 2 22 --pure --mode exact --baseline` (and 8 8 2 2 25, 9 9 4 4 62).
Re-run by the integrator: coefficients byte-identical (`experiments/E11_calibration/calibrate_33.json`).

| quantity | value |
|---|---|
| fit set | n = 709 TRAIN cases open at 2k but solved exactly (true conflicts 2,001 … 221,874) |
| coefficients | a = 0.0670, b = 0.5089, g = 0.04979 (c2000 clipped to the first cap; unclipped fit gave b = −20.8 by fitting the 2000–2006 stopping noise) |
| Spearman ρ | 0.478 train / **0.395 hold-out** (n = 276) |
| log-RMSE | 0.878 / **1.282** (constant predictor 1.216; `d = cap` 2.255) |

**Conclusion.** The §6.2 acceptance (ρ ≥ 0.85) is **not met** and cannot be met by this feature
set: inside the censored set `c2000` is constant and no cheap proxy orders the tail (log2_volume
0.48/0.40, nclauses 0.47/0.17, decisions 0.21/0.41, propagations/restarts ≈ 0). The clip
`[cap, 20·cap]` bounds the damage; `censored_share` is reported in every score. Note `g`
extrapolates on wide cells ((10,20): vol ≈ 172 → f̂ ≈ 15× cap, labels at the 20× ceiling).

**GEN cells** (Tan 2022 Tables 2/4: z_2(7,7) = 21, z_2(8,8) = 24, z_4(9,9) = 61): (7,7;2,2) w=22 → 0 cases
(counting UB is 21); (8,8;2,2) w=25 → 1 case, killed by the library; (9,9;4,4) w=62 → 49 cases, 24 library
kills, 25 scored survivors, W = 10,471, max d = 1,080. Square (2,2) cells are counting-tight in pure mode
for n = 7 … 13; alternatives with content: (8,9;2,2) w=27 (6 cases), (10,10;4,4) w=75 (49), (11,11;4,4) w=87 (2,704).

**Proved-library masks** (`LIBRARY_LEAN = Prune.or (baseline P) (counting P)`, axioms {propext,
Classical.choice, Quot.sound} on all 30 non-empty tables, no SAT-witnessed case killed):
TRAIN (9,9)50 19/36 → S = 17, W = 15,457; (9,10)55 23/45 → 22, 18,905; (10,10)61 15/25 → 10, 9,865;
(10,11)65 129/195 → 66, 178,192; (11,11)70 388/625 → 237, 1.44 M; (11,12)75 193/420 → 227, 2.65 M;
(12,12)81 88/225 → 137, 4.52 M (TRAIN survivors 716, work 8.8 M conflicts). BATTERY (9,9)49 90/169,
(9,10)54 81/165, (10,10)60 91/144, (10,11)64 413/725, (11,11)69 1060/2025, (11,12)74 1016/2079,
(12,12)80 689/1296. Band (12,13)86 1082/1710, (12,13)87 258/360, (13,13)92 2527/3969, (13,13)93 710/1024,
(10,14)77 946/2064, (10,14)78 305/525. Reproduces the v1 `cache/library_masks_*.json` exactly.

Table edits recorded: `(12,13,87)` deepened to 200k (table_hash 8f109241b9ef0e05 → 38151744000cacb0);
2 cases of `(10,20,103)_pure` deepened to 20k as a CLI test (022139182c085261 → f810a1bf9d5fb99e).
Censored labels remaining: (11,11)69 93, (11,12)74 271, (12,12)80 387, (12,13)86 530, (12,13)87 1,
(13,13)92 1,729, (13,13)93 383, (10,14)77 603, (10,14)78 119.

## M1 + M2 delivered — 2026-09-21 23:15 (batch-1 workflow: 5 builders + integrator, 50 min)

Design §10.1 test plan, measured by the integrator (`docs/build/INTEGRATION.md`):

| test | result |
|---|---|
| T-1 unit tests | 63 OK (engine, gate 23, ledger 11, tables 4, golden bank 18), 304 s |
| T-2 baseline | initial program: combined **0.20**, ladder 5, lean_ok 1; library program 0.20; unsound program **0** |
| T-3 golden bank | all 16 adversarial candidates in their bands (unsound → 0; sorry/axiom/native_decide/implemented_by/opaque/unicode/mask-spoof/redefinition → L0 = 0; sketches L2/L3 ≤ 0.19; slow kill L1) |
| T-4 calibration | fhat fit a=0.067 b=0.509 g=0.050; hold-out Spearman 0.395 (design asked ≥ 0.85 — **not met**; censored labels stay clipped to [cap, 20·cap], the daemon's deeper pass is the remedy) |
| T-5 synthetic loop | 30 iterations + resume, no unsound program with score > 0, best non-decreasing |
| T-6 stub LLM over the real OpenAI client path | 10/10 iterations, checkpoints written |
| T-8 (Python side) | (10,21,107), (11,19,107), (11,20,112): **0 cases** with ledgered Tan facts (pure: 185 / 2 970 / 198) |

Gate v2 also fixed a pre-existing bug: Lean 4.34 prints `error(lean.unknownIdentifier):`, which the v1 regex
missed, so unknown identifiers were mis-scored as "compiled". 18 deviations from the design are recorded in
INTEGRATION.md §4. Remaining for batch 2: `Schemas.lean` (Farkas over pair codegrees) + `lemmas.py`,
DGH(4), `Closure.lean` (T-7 kernel timing), promotion + closure daemon, gate entry point for `CondPrune`.

## E7b — 2026-09-21 23:25 — Probes under the v2 prompt and genome (reward v2, verified floor 0.20)

Same `probe_one.py`, parent = new `initial_program.py` (library only, 0.20).

| model | reasoning | wall | out tokens | cost | outcome |
|---|---|---|---|---|---|
| google/gemini-3.8-flash | low | 159 s | 29 996 (0 reasoning) | $0.121 | **no diff**: 70k chars of visible deliberation ("Wait! …"), ran out of tokens before any SEARCH/REPLACE block. Reasoning-effort hints do not stop this model from thinking in the content channel. |
| openai/gpt-5.6-luna (after adding a strict OUTPUT FORMAT rule to the prompt) | low | 12 s | 1 282 | $0.005 | diff applied; a real Lean prune (waterfilled column bound — mathematically the library's `argWF`, so no gain even when correct) with three API slips (`colSum_le` argument order, missing `hs` in `sum_le_waterfillBound`, final `omega` after a `rw`): **L1**, score 0.018, errors in `lean_errors` with goals. |
| gpt-5.6-luna, 2nd generation (its own child + the `lean_errors` artifact as parent) | low | 8.7 s | ≈1 300 | $0.004 | **repaired both API slips correctly** and added the row-side dual; but it also re-implemented the Python mirror's `waterfill_bound` incorrectly, which kills realizable cases → battery hard-zero **before Lean runs**. |

Spend so far: $0.82 of $16.

**Findings.** (1) A strict output-format rule fixes luna's behaviour (12 s, code-first) but not
Gemini-Flash's; model choice must be validated per model with one probe — the design's P-1. (2) The
artifact-driven repair works: the next generation fixed exactly the reported errors. (3) **Reward
flaw found**: a wrong *Python mirror* zeroes the whole candidate even when its Lean would be L5.
Since a proven Lean kill is sound regardless of the mirror, the battery hard-zero should apply only
when the candidate is not L5; at L5 the mirror violation should become `agreement < 1` plus a
`mirror_unsound` artifact. TODO after batch 2 lands (touches `evaluator.py`/`reward.py`).
(4) Both luna prunes are equivalent to library members (`argWF` and its transpose) — evidence again
that models reach for the nearest known argument; the schema channel (Farkas over pair codegrees)
and DGH(4) are what batch 2 adds so that there is verified credit to be earned without a new proof.

## E16 — closure daemon launches (zar_ub/closure_daemon.py)

One bullet per certification launch (design §7 item 3).

- (superseded — identical pass re-run below after fixing the module doc-comment provenance path) 2026-09-21 23:35 launch: run `experiments/no_llm/run_stub` checkpoint 10, program `bf2acfca-daca-4710-b79f-fa3c66ab1328` (score 0.2000); promotion OK sha `c285948723e7` ledger=1; target m9_n9_s3_t3_w50 (pure): 36 cases, library kills 19 (route per-entry, entries ['c285948723e7']), survivors 17, W^=1.55e+04 vs budget 5e+08 (--now); certify: 17/17 certified, sat 0, timeout 0, failed 0; claim z(9,9;3,3) <= 49: YES; report `cache/certs/m9_n9_s3_t3_w50/closure_report.md`
- 2026-09-21 23:37 launch: run `experiments/no_llm/run_stub` checkpoint 10, program `bf2acfca-daca-4710-b79f-fa3c66ab1328` (score 0.2000); promotion OK sha `c285948723e7` ledger=1; target m9_n9_s3_t3_w50 (pure): 36 cases, library kills 19 (route per-entry, entries ['c285948723e7']), survivors 17, W^=1.55e+04 vs budget 5e+08 (--now); certify: 17/17 certified, sat 0, timeout 0, failed 0; claim z(9,9;3,3) <= 49: YES; report `cache/certs/m9_n9_s3_t3_w50/closure_report.md`
- 2026-09-22 00:13 launch: run `experiments/no_llm/run_stub` checkpoint 10, program `bf2acfca-daca-4710-b79f-fa3c66ab1328` (score 0.2000); promotion already in ledger sha `c285948723e7` ledger=1; target m9_n9_s3_t3_w50 (pure): 36 cases, library kills 19 (route evolved, entries ['c285948723e7']), survivors 17, W^=1.55e+04 vs budget 5e+08 (--now); certify: 17/17 certified, sat 0, timeout 0, failed 0; claim z(9,9;3,3) <= 49: YES; report `cache/certs/m9_n9_s3_t3_w50/closure_report.md`; Tier-1 closure ESTABLISHED `lean/ZarPrune/Closures/Z_9_9_50.lean`

## M3 + M4 + M5 delivered — 2026-09-22 00:20 (batch-2 workflow: 4 builders + integrator)

Inputs `docs/build/{E-schemas,F-dgh4,G-closure,H-promote}.md`; full report `docs/build/INTEGRATION.md` (batch-2 section).
`lean/ZarPrune.lean` now imports `Schemas`, `DGH`, `Closure`, `Evolved`; `lake build` green (1034 jobs, the four
`Closures/Z_*.lean` re-checked in the build); two Schemas names renamed for the root import (`topSumF`, `le_topSumF`,
`arrayGetD_ofFn`).  Gate **v2.1** adds the `CondPrune` entry point (`candidateF`, facts injected per instance from
`ledger.facts_for` at the table's trust; `never` where a literal fact is not granted).  `LIBRARY_LEAN` = baseline ∪
counting ∪ `evolved`; all 48 cached baseline masks rebuilt and **byte-identical**.

| milestone | status | numbers |
|---|---|---|
| **M3** `Closure.lean` Tier-1 | **done** | T-7 (9,9,50) pure: 17 Lean survivors == table, `Z_9_9_50.lean` checked against `Closure.olean` in 2.5 s, kernel `decide` + theorem 0.9 s, axioms {propext, Classical.choice, Quot.sound}, 17/17 LRAT → `z_9_9_le_49` established (Tier 1, no native). T-8: `z_10_21_le_106` (0×2), `z_11_19_le_106` (22×0), `z_11_20_le_111` (2×0) check in 2.3–2.9 s, conditional only on the named Tan facts; all four replayed by `leanchecker` (4.6–6.4 s). Kernel scaling 0.04 s/pair (G); `chooseMul` not needed. |
| **M4** promotion + daemon | **done** | `promote` P0–P6 + replay; ledger 1 entry (`E_c285948723e7`, the initial program, kills nothing); daemon `--now` on (9,9,50) pure: route **evolved** (single term after the build), 19 killed / 17 survivors, 17/17 certified, then `close_instance` → **Tier-1 closure ESTABLISHED** (the seam now owns `closure_report.md`); T-11 `verify-certs --fresh-lratcheck` 23/23 VERIFIED + corruption control rejected; T-12 `audit-kills` 1 sampled of 19 → UNSAT (96 conflicts), no PIPELINE_BUG. |
| **M5** schemas + hand prunes | **done** (T-10 transfer matrix not run) | E12 with the built `Prune.ofFarkas`: **TRAIN 164/716 survivors killed, d share 6.6 %, mean gain_I 0.137, 8 certificates**; 15 tables 659/4632; SAT self-check 0/11 witnessed kills; gate 154 s, L5 ×15, Lean `schema_mask` == mirror everywhere. `schema_farkas` golden **0.2496** (schema_gain = proven_gain 0.1368). DGH: `argDGH` proved for all (m,n;s,t); `lean_dgh4` L5, 0.20 on the default suite (square TRAIN cells are DGH-inert), **0.2597** with target (9,12,64); census over 48 tables 44/14,615 survivors killed, 0 witnessed kills; (11,21,117) pure 37/37 killed → z(11,21) ≤ 116 with zero SAT. The design's `schema_farkas refutes (15,17,133)` is not testable: the cited (15,17,133) table has 0 cases. |
| T-1 | **90 tests OK** (528.7 s) | golden bank 20 entries: schema_farkas 0.2496, cond_candidateF 0.20 (never), lean_dgh4 0.20, python_only_dgh4 0.20, all adversarial rows unchanged |

Decisions: `argDGH` is a library lemma but **not** in the baseline (credit only where it fires); the 8 TRAIN Farkas
certificates are not promoted (instance-specific data, credited through `schema_gain`); a table grants exactly the facts
of its own trust level (pure TRAIN → no conditional prune fires there).  Open after batch 2: E7b reward flaw (mirror
kill on a witnessed case zeroes an L5 candidate), ledger rows from `close_instance`, `pure:` targets in `parse_targets`,
T-10 transfer matrix, TARGET closures in trust mode ((12,18,109) ~2 min kernel, (13,19,123) ~3 min).

## E19 — 2026-09-22 00:45 — Adversarial audit hole closed (sandbox + authenticated gate cache) and the mirror rule

**Finding (batch-2 adversarial verifier, `docs/build/ATTACKS.md`).** 25 attacks; 24 defended by the gate
(scan, axiom audit, nonce masks, declared names, CondPrune discharge, SCHEMA_DATA validation). **One real,
critical hole**: the stage-1 candidate Python ran with unrestricted filesystem writes, and the on-disk
Lean-gate cache (`cache/gate/<sha1>.json`) had a reconstructable key, so a candidate could forge an L5
result with a fabricated kill mask and score **1.0** without Lean ever running (`tests/attacks/gate_cache_forgery.py`).
Same root cause: creating/removing `cache/PIPELINE_BUG`, tampering with live tables, poisoning the ledger.

**Fix.** (1) `zar_ub/sandbox.py`: the candidate subprocess runs under an OS sandbox — macOS `sandbox-exec`
(deny all file writes except its temp dir, deny reading the gate secret, deny network), Linux `bwrap` when
present; without either, the evaluator disables the persistent gate cache for the process and warns.
(2) `lean_gate.py`: every cache entry carries an HMAC-SHA256 over (key, results) with a 32-byte secret in
`cache/gate/.secret` (mode 0600, unreadable inside the sandbox); a bad MAC is deleted and ignored.
(3) `tests/test_sandbox.py`: probe subprocess — writes to cache/gate, the sentinel, a table, lean/ all
**denied**; the secret **denied**; a socket **denied**; the temp dir writable; forged entry rejected.

| check | before | after |
|---|---|---|
| `tests/attacks/gate_cache_forgery.py` | **1.0** (forged L5) | 0.20 (real Lean = library, kills nothing) |
| initial program, cold / warm (fresh process) | 0.20 / cache hit | 0.20 (24.5 s) / **cache hit, 0.01 s** (MAC verified) |
| `schema_farkas` | 0.2496 | 0.2496 |

**Reward rule change (E7b flaw).** A Python mirror that kills a witnessed case no longer zeroes the whole
candidate: the Lean kill is the only thing that prunes or earns credit, so an L5 candidate with a wrong
mirror scores its Lean-only verified value (E := 0, `agreement` < 1, new `mirror_unsound` artifact), a
non-L5 candidate still scores 0, and stage 1 returns 0.01 (not 0) so the cascade lets Lean decide.
`unsound_row7` therefore now scores 0.20 with `sound_battery` 0 (golden updated); luna's repaired child of
E7b scores 0 because its Lean is still L1.

## E18 — 2026-09-22 00:47 — First paid OpenEvolve run (gpt-5.6-luna only, 15 iterations, one island)

`experiments/E7_llm_smoke/run_smoke.sh 15 config_smoke_luna.yaml` → `run_20260922_004734/` (summary.md).
Cost **$0.125** ($0.0083 per iteration, 9–43 s each). 16 programs: 8 at L5 = 0.20 (the initial program,
no-op children, and one that just appended the proved `argDGH`), 8 at L1 ≈ 0.017 (Lean attempts with API
slips: `sumFin_le` argument names, a non-existent `weight_le`, a `rewrite` pattern). What the model tried:
"ambient density w ≤ m·n", "row/column totals agree" — both already in the baseline — and `argDGH`.
It never used `SCHEMA_DATA`. No unsound program, no PIPELINE_BUG, best stayed at the 0.20 floor.

**Diagnosis.** The predicted 0.20 plateau (design §11): a cheap model reaches for known or trivial
arguments and makes signature mistakes; the population then prefers the L5 no-ops (0.20) over the L1
attempts (0.017), so errors are not repaired. Two fixes below (E20).

## E20 — 2026-09-22 01:05 — Search mode made one line: `search_certificates`

`zar_ub.lemmas.search_certificates(max_seconds)` reads the candidate's private copy of the case tables,
runs the pair-codegree LP on every library survivor (training tables first), and returns verified Farkas
certificates as `SCHEMA_DATA["farkas"]` entries. A candidate that just calls it at import time
(`tests/candidates/schema_recipe.py`):

| candidate | ladder | combined | proven (TRAIN) | schema_gain | target_gain | eval time |
|---|---|---|---|---|---|---|
| initial | 5 | 0.2000 | 0 | 0 | 0 | ~8 s cached |
| schema_farkas (8 hand-picked certificates, E12) | 5 | 0.2496 | 0.1368 | 0.1368 | 0 | 34 s |
| **schema_recipe** (40 s search at import) | 5 | **0.2594** | 0.1368 | 0.130 | 0.041 | 89 s |

Prompt now states the already-proved list (re-adding earns nothing) and the one-line recipe, so the
loop has a verified-credit floor above 0.20 from which the LLM must add *new* inequalities to the LP
(each a Lean-proved `Prune`) — the intended division of labour (design §2.3, §8.4).

## E21 — 2026-09-22 00:58 — Second luna run with the recipe prompt (15 iterations, $0.105)

`run_20260922_005826/summary.md`. Iteration 1 already adopts the recipe (`search_certificates`) plus
`argDGH` → **0.2594** (proven 0.1368, schema 0.130, target 0.041); 13 of the remaining 14 children keep
0.2594 and only tweak the search budget (40 → 180 → 240 s, capped by the 90 s subprocess limit anyway) and
rewrite NOTES; one child is L1 (a `SCHEMA_DATA_PLACEHOLDER` slip). No new Lean inequality was attempted.
Spend so far: $1.07 of $16.

**Reading.** The recipe works as a floor and the loop is stable (no unsound program, 15/16 L5), but a
cheap model does not go beyond it in 15 iterations: the verified-credit gradient above 0.26 requires a
new hidden-variable inequality with a Lean proof. E22 tests a stronger model on exactly that.

## E23 — 2026-09-22 01:10 — Zero-shot transfer matrix of the proved library (design T-10)

`experiments/transfer.py` → `experiments/transfer_matrix.md` (38 tables, 32,819 cases; one Lean gate
process per prune, 2–3 min each). Kills / share of the table's difficulty removed:

| table | trust | cases | library union | argD | argDT | argDGH | others |
|---|---|---|---|---|---|---|---|
| (9,9) 50 | pure | 36 | 19 (0.201) | 10 (0.114) | 10 (0.098) | 0 | 0 |
| (11,11) 70 | pure | 625 | 388 (0.277) | 203 (0.141) | 203 (0.151) | 0 | 0 |
| (12,12) 81 | pure | 225 | 88 (0.258) | 45 (0.134) | 45 (0.135) | 0 | 0 |
| (13,13) 93 | pure | 1 024 | 710 (0.435) | 440 (0.228) | 440 (0.263) | 0 | 0 |
| (9,23) 104 | tan2022 | 244 | 150 (0.369) | 150 (0.369) | 0 | 0 | 0 |
| (12,18) 109 | tan2022 | 2 562 | 1 044 (0.359) | 1 023 (0.352) | 21 (0.007) | 0 | 0 |
| (13,19) 123 | tan2022 | 4 130 | 1 482 (0.327) | 1 473 (0.325) | 9 | 0 | 0 |
| (16,17) 134 | tan2022 | 2 139 | 1 146 (0.500) | 1 017 (0.436) | 183 (0.083) | 0 | 0 |
| (10,23) 113 | tan2022 | 4 818 | 3 661 (0.655) | 3 629 (0.646) | 0 | **219 (32 marginal, 0.024)** | 0 |

**Reading.** On enumerated cases Argument A never fires (the enumerator already applies it), the
deletion/waterfill prunes never fire (Argument I with cited minors subsumes them), and DGH fires only on
the wide cell (10,23). **Argument D is the whole library on these tables** — it removes 20–65 % of the
work — and everything beyond it must come from hidden-variable arguments (Farkas schema: +6.6 % of the
remaining training work, +41 % on (10,20)) or new ideas. Caveat: the `evolved` column duplicates the
whole library (the stub-run promotion is the library itself), so the "marginal" figures of this run are
zero by construction; `transfer.py` now excludes `evolved` from the marginal computation.

## E22 — 2026-09-22 01:07 — claude-sonnet-5 (low reasoning), 20 iterations, recipe prompt

`run_20260922_010707/summary.md`. **Cost $2.24** ($0.11 per iteration — 20× luna; total spend now $3.31 of $16).
Iteration 1 adopts the recipe → 0.2594 (same floor as luna). Then Sonnet does what the thesis wants a
model to do — it tries genuinely new arguments: `argRowLocalAgg` (Argument D summed over all rows: equal
to Argument A, so no gain), `pairFloorCap`/`argPairCap` (floor r_i + r_i' − n ≤ codeg ≤ cap: exactly the
F3/F4 constraints the LP already exploits), each with a Lean proof attempt. Outcomes: 7 L5 (recipe
variants; `argPairCap` compiled with `kill := fun _ => false`, an honest vacuous prune), 3 L1 (omega/rewrite
failures), and 11 programs at 0 with a **Python SyntaxError** — the diff broke the `'''` delimiters of the
Lean string literal (children inherited it). One attempt was rejected at L0 only because its `sorry` was
not in the typed `have … := by sorry` form.

**Fixes.** (1) Gate: a bare `sorry` inside `sound := by` is now an *untyped hole* (never trusted —
`sorryAx` — not auto-filled, ladder ≤ L2) instead of L0; the rejected Sonnet attempt now scores L1/0.14
partial instead of 0 (`tests/test_gate.py` updated). (2) Prompt: never edit the `LEAN_SOURCE` delimiters;
holes explained. (3) Lesson for the thesis: a strong model's first ideas stay *inside* the LP's constraint
system (pair floors/caps, aggregated budgets); new verified credit needs constraints the LP lacks
(triple codegrees, residues mod g, transposed pair systems) — the next lemma-library items.

## Session close — 2026-09-22 01:35

Final state: 90+ unit tests (engine, gate incl. untyped holes, sandbox, lemmas, ledger, tables, promotion,
golden bank of 22 candidates) green; spend $3.31 of $16 (`cost_ledger.md`); `docs/STATUS.md` is the
thesis-facing summary; `docs/build/INTEGRATION.md` + `docs/build/*.md` the build provenance; this log
E1–E23 the experiment record. Nothing has been committed to git (working tree only), by design — the
repo owner decides what to keep.

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

---

# Batch 3 (2026-09-22/23): the two pitfalls — difficulty of open cells, and single-cell rules

Numbering note: the LOG entries E24–E27 below map onto the experiment folders as
E24 = `experiments/E24_difficulty/GROUND_TRUTH.md` (A1), E25 = `experiments/E24_difficulty/{LOOKAHEAD,SAMPLING,PROGRESS,EVALUATION}.md`
(A2–A4, D5), E26 = `experiments/E25_reward/` (R1, R3 in `attacks/`) + `experiments/E26_dynamics/` (R2), E27 = `experiments/E27_integration/`.
R1's and R2's REPORT.md files were not written (the harness refused subagent report files); the entries below are
built from their result files (`results/*.md|json`, `attacks/results.md`, `E26_dynamics/results/*.md`).
All solver costs are CaDiCaL 1.9.5 (pysat `cadical195`) conflicts/propagations; seconds are inflated by contention.

## E24 — 2026-09-22 — Ground truth for difficulty (A1)

**Question.** How hard are the cases we currently label by extrapolation, really?

**Run.** `experiments/E24_difficulty/gt_{initial,deepen,newtables}.py`: every probed record of every cached table
(`ground_truth_initial.jsonl`, 32,801 rows), then ONE fresh 2M-conflict run (180 s wall, never hit) on every censored
case of the wide pure tables and the (9,23) target, a seeded 150-case sample of each of (12,18), (13,19), (16,17), and
three NEW exactly-known wide pure tables (w = z+1): (9,16,78), (9,18,86), (10,19,99).  Determinism check: 60/60 cached
exact labels reproduce the identical conflict count with today's encoder.

| quantity | value |
|---|---|
| final file | `ground_truth.jsonl`, 35,081 rows (3,554 deepened/new) |
| deepen cost | 770.3 M conflicts, 196.9 G propagations, 38,436 solver-s (91.5 min wall, 7 processes) |
| outcome of the 3,554 runs | 3,363 unsat, **191 open at 2M**, **0 SAT** (consistent with z(9,23) = 103: all 146 censored (9,23,104) cases are unsat or open) |
| exact HARD labels (d > 20k) | **2,238** (was 262); 1,518 of them in wide cells; max exact d 1,986,172 (was 221,874) |
| old censored label vs truth | Spearman 0.11 on (10,20,103)p, 0.23 on (13,13,93)p, 0.34 on (12,18); a **constant** 400,000 on every censored case of (13,19) and (16,17), whose sampled true d spans > 1.5 decades (51 % of the (16,17) sample is above 2M) |
| c2000 on hard cases | the constant 2,000: fhat is a function of log2_volume alone there |

New wide TRAIN candidates (library = Argument D exactly on all three):

| cell | cases | library survivors | survivor work | DGH kills on survivors | DGH gain / tail on survivors |
|---|---|---|---|---|---|
| (9,18) w86 | 363 | 140 | 19.8 M | 47 | 0.110 / 0.000 |
| (9,16) w78 | 1,491 | 693 | 36.7 M | 90 | 0.031 / 0.000 |
| (10,19) w99 | 426 | 165 | 49.7 M | 0 | 0 / 0 (control) |

DGH kills the EASIER survivors of these cells (median d 30k vs 137k on (9,18)), hence zero tail gain.

## E25 — 2026-09-23 — Difficulty estimators head to head (A2 lookahead, A3 sampling, A4 CDCL progress, D5 evaluation)

**Question.** The proposal's techniques for estimating the difficulty of a case the pipeline cannot solve — AlphaMapleSAT-style
lookahead [16] and Chivilikhin-style sampling of a decomposition's hardness [17] — were not implemented.  Do they beat the
current censored label, and which estimator should label the censored cases?

**Implemented.** `zar_ub/hardness_lookahead.py` (A2: level-1 failed literals, pair lookahead, march-style scores, Knuth tree-size
probes after the failed-literal fixpoint; BCP only), `zar_ub/hardness_sampling.py` (A3: Chivilikhin d-hardness — solve N random
cubes of a decomposition set at a budget, extrapolate; default knuth:row:8, N=100, b=5000, ≤ 50k conflicts), and
`zar_ub/hardness_progress.py` (A4: statistics of the 2k/20k probes the pipeline already runs, CaDiCaL binary statistics,
static/LP slack of Arguments A/D/DGH).  D5 (`EVALUATION.md`) compared everything on the E24 ground truth: 5,133 cases
(2,429 HARD incl. 191 open at 2M, 18 cells; 2,704 MID), leave-one-shape-out, nested feature selection, plus the error each
estimator induces in the reward's gain_I / tail_I on 106 real kill masks.  Evaluation spend: 15,387 estimator runs,
0.34 G conflicts, ~167 G propagations, 38 min wall on 10 processes.

| model (HARD regime, held out by shape) | within-cell ρ | Harrell C (incl. open) | log-RMSE | gain_I / tail_I error (real masks) | extra cost per case |
|---|---|---|---|---|---|
| current label `clip(fhat, 20k, 400k)` | 0.324 (constant in 8/15 cells) | 0.552 | 1.116 | 0.061 / 0.164 | 0 |
| lookahead alone (greedy) | 0.691 | 0.782 | 0.707 | 0.031 / 0.088 | ~2k UP conflicts, 0.8 s |
| static/LP slack alone (greedy) | 0.624 | 0.762 | 0.772 | 0.038 / 0.098 | 0 |
| sampling alone (unit-slope calibration) | 0.721 | 0.771 | 0.828 | 0.050 / 0.157 | ~46k conflicts, 3 s |
| A4 free20k (probe stats + static) | 0.802 | 0.826 | 0.547 | 0.026 / 0.058 | the probe the pipeline runs anyway |
| **winner: greedy free tier (probe stats + static + lookahead)** | **0.837** | **0.847** | **0.510** | **0.020 / 0.046** | ~24k conflicts, 18 M props, 1.3 s (0.8 s if the probe stats are stored) |
| greedy free + sampling | 0.853 | 0.850 | 0.493 | 0.027 / 0.067 | +46k conflicts |
| greedy free + CaDiCaL binary stats | 0.874 | 0.865 | 0.485 | 0.026 / 0.055 | +22k conflicts |
| greedy all families | 0.888 | 0.872 | 0.451 | 0.028 / 0.057 | ~92k conflicts, 4.9 s |
| GBM all families | 0.899 | 0.878 | 0.394 | 0.023 / 0.101 | ~92k conflicts |
| continue the solve to 50k, then the winner on the still-open | **0.932** | **0.934** | – | **0.017 / 0.044** | ≤ 30k conflicts more |

Target cells alone: current 0.514 / C 0.560 → winner 0.835 / 0.885.  Library survivors only (what the reward sees):
0.064 / 0.506 → 0.821 / 0.858.  Square→wide transfer: winner 0.823 / 0.843 (bias −0.13 nats); the zero-conflict
(static + lookahead) tier ranks at 0.689 but its level is off by −1.46 nats on wide cells.

**Winner** (`experiments/E24_difficulty/hardness_model.json`, `zar_ub/hardness_model.py`), refit on all 2,238 exact hard cases:
`log d̂ = 11.301 + 0.900 z[decisions/conflict of the 20k probe] − 0.381 z[log1p restarts per 1k conflicts] + 0.364 z[log1p Knuth
tree-size estimate after failed literals] + 0.161 z[distinct row sums] − 0.240 z[failed negative cell literals] − 0.154 z[mean row
Argument-D slack]`, d̂ = exp(·)·1.121 (Duan smearing, for sums), floor 20k, ceiling 2M.  Chosen over the better rankers because
its reward-gain error is the lowest of all (table-bootstrap 95 % CI of the difference: current label +0.027…+0.057, free+binary
+0.001…+0.012, all +0.002…+0.015) and it is the cheapest; `predict()` recomputes all features and matched the cached
predictions exactly on 6 seeded cases.

**Verdict on each proposal technique.**
* *Lookahead (AlphaMapleSAT / march / Knuth).* Real but modest: the best **zero-conflict** signal (ρ 0.71 hard, 0.59 mid, with
  static features), +0.035 ρ and −0.005 gain error inside the free tier; adds nothing once CaDiCaL binary statistics are present;
  its magnitudes do not transfer from square to wide cells.  Two of its features are in the winner.
* *Sampling (Chivilikhin d-hardness).* A genuine standalone estimator (ρ 0.72, C 0.77) and +0.014–0.048 ρ on top of CDCL features,
  but it does **not** reduce gain error, and spending its ~46k conflicts on simply continuing the solve is strictly better
  (the 50k-direct row).  Not in the winner.
* *LP / counting slack.* Weak alone (single best ρ 0.38), nothing on top of the others; two small static terms survive.
* *What actually works* was not a proposal technique: the search statistics (decisions per conflict, restarts) of the 20k probe the
  pipeline already runs (A4).
* On the user's DGH example the current label **over**-credits DGH: (9,18,86)p true gain/tail 0.110/0.000, current label
  0.269/0.499, winner 0.101/0.  Accurate difficulty removes DGH's false tail credit; it does not make narrow rules look better.

## E26 — 2026-09-23 — Rewarding single-cell rules: reward variants (R1), dynamics (R2), adversarial critique (R3)

**Question.** Under reward v2 a rule that helps one cell (DGH, which closes z(11,21) ≤ 116 with zero SAT) scores exactly 0.20,
the score of doing nothing.  Can a reward credit it without opening exploits, and what does the loop then do?

**R1 (`zar_ub/reward_variants.py`, `experiments/E25_reward/`).** 16 real Lean-gated programs + 8 oracle masks over 25 cached tables
(20,712 cases); offline scoring from cached Lean masks (4,106 s of gate wall for the masks, 8,195 HiGHS LPs, 0 SAT).  Criteria:
C1 soundness ordering exact; C2 DGH uplift ≥ 0.02 and ≥ 0.5 × the recipe's; C3 every exploit (real or oracle shape) below
min(recipe, DGH); C4 monotone in kills, uniform gain beats a single cell; C5 the recipe keeps ≥ 80 % of its v2 uplift.

| variant | recipe | DGH | recipe+DGH | fails |
|---|---|---|---|---|
| V0/S0 (today) | 0.2594 | **0.2000** | 0.2594 | C2 (DGH = no-op); C3 oracle: clearing (10,10,61) 0.2635 > recipe, clearing GEN 0.2800, killing every d ≤ 2k 0.4100 |
| V0/S1 (suite only: wide exact cells in TRAIN, later targets in TARGET) | 0.2937 | 0.2806 | 0.3743 | C3 oracle (easy-2k 0.3617) |
| V1 mixture / V2 power means | 0.38–0.51 | 0.38–0.52 | – | scale-free: clearing a tiny table ≈ the recipe |
| V3 family / V4 work weights / V5 closure alone | – | – | – | C3 oracle |
| **VR** = family-balanced ln(1+W/2000)-weighted means + Depth + Close, S1 | 0.3403 | 0.3182 | 0.3634 | none of C1–C5 (on R1's attack set) |

The suite change is necessary: on S0 no formula can credit DGH, because it kills nothing on a scored cell.

**R2 (`experiments/E26_dynamics/`).** OpenEvolve with a blind genome mutator over five atoms (R recipe, D DGH, C the (10,22)
closing certificate, T a (12,18) certificate, U an unsound-mirror twin), real Lean gate per genome; 12 seeds × 50 iterations and
8 seeds × 150 iterations per configuration (fixed seeds do not make OpenEvolve reproducible: `database.py` samples from
`list(set(uuid4 ids))`).  At 150 iterations: V0/S0 best 0.2594, the reported best contains D in 1/8 runs, R+D population share
0.31; under VR / V3+V4+V5 / V0/S1 the best is R+D+C in 32/32 runs, R+D reaches 84–94 % of the population, 68–76 % of parents,
~20/20 archive slots; the three are dynamically indistinguishable (pairwise p ≥ 0.12).  MAP-Elites axis `gain_concentration`
instead of `lean_ladder`: +1.7 D elite cells at 50 iterations (p = 0.019), gone at 150 (p = 0.78) because OpenEvolve culls by
global fitness — **negative result, config unchanged**.  0 soundness violations in 124 runs; but the unsound-mirror twin ties its
sound twin (E19 rule: L5 Lean + wrong mirror = verified score), so it is the *reported* best in 3/8 VR runs at 150 iterations.

**R3 (`experiments/E25_reward/attacks/`, offline).** No realizable program beats the genuine rules under VR, but four weaknesses
with one cause (credit keyed on labelled work that is tiny or inflated by the censored ceiling):
1. VR's closure bonus was paid on an inflated label: the (10,22,111) certificate scores 0.3087 because one censored case is labelled
   400k; its true d is 34,219 (the whole cell is 58,333 conflicts).  On the final ground truth the recipe scores 0.2998, not R1's
   0.3374 (R1's GT column predates the final file).
2. Easy-case oracles break C3 once the threshold moves off 2k: kill every exact d ≤ 5k → 0.3422 (> recipe 0.3403); d ≤ 20k → 0.4136.
3. Clearing all 7 tables with < 1e5 conflicts of true work → 0.3929; realizable part (pool certificates on small tables) 0.3187.
4. Clearing a tiny table (8 cases, 9.6e3 conflicts) outranks f_weak, which removes 1.98e7 target conflicts.
**Fix (VR\*\*):** count only work above 20,000 conflicts per case (d' = max(0, d − 20k), the HARD regime); compute Depth/Close
importance from LOWER-BOUND work (exact d, else conflicts reached — never fhat); credit Depth/Close only on cells with
≥ 1e6 lower-bound conflicts.  Under VR\*\* every attack shape scores below the genuine rules and no tiny-table clear beats f_weak;
C2 2.97; the cost is C5: 1.04 on table labels, **0.81** on ground-truth labels (just above the 0.8 bar), because the recipe loses its
inflated closure.  Not holes: splitting a rule, the tail tie-break (aligned with true difficulty), the single-cell square target.

## E27 — 2026-09-23 — Integration: the hardness model is the censored label, reward v3 + suite v3 are live

Full report: `docs/build/INTEGRATION.md` (batch 3).  Scripts and outputs: `experiments/E27_integration/`.

**What changed.** `difficulty.py`: a case open at ≥ 20k conflicts is labelled by the E25 winner
(`zar_ub/hardness_model.py`) from the statistics of its own fresh 20k run (now stored in the probe as `ps20k`),
`d = min(max(r, d̂), max(r, 2M))` with r = conflicts reached; `ZAR_UB_DIFFICULTY=legacy` restores fhat.
`reward.py` v3 = R3's VR\*\* (`ZAR_UB_REWARD=v2` restores v2); `suite.py` v3 = S1 (7 square + 7 wide exact TRAIN cells,
7 non-empty targets, 4 GEN cells); the evaluator gates only library survivors (8,817 of 19,317 scored cases);
prompt/initial-program credit text rewritten; one gate fix (gate v2.2, below).

**Relabel (cost in CaDiCaL conflicts).**
* E24 write-back (no solver): 1,088 exact labels + 186 raised lower bounds into 12 tables.
* Model relabel of every remaining censored case of the scored tables: 11,840 cases (11,649 model, 191 at the 2M lower
  bound), **32.0 min wall on 12 processes, 0.283 G conflicts, 349 G propagations, 22,990 solver-s**; no re-probe decided
  a case.  Before: one distinct censored label (400,000) on (12,18), (10,23), (11,23), (13,19), (16,17); after: 1,172 / 908 /
  134 / 1,487 / 298 distinct labels; 2,468 of 6,452 censored target survivors (38 %; 2,474 of 6,458 including (9,23)) at the 2M ceiling [corrected E28].
* `python -m zar_ub calibrate --model --holdout "12,13,87"` (no solver): held-out (12,13,87) hard cases ρ **0.879** (legacy
  0.182), log-RMSE 0.33 (0.84); leave-one-shape-out within-cell ρ 0.845 (legacy 0.324), log-RMSE 0.48 (1.12).

**Before/after, difficulty (E25 protocol, held out by shape, HARD regime):** within-cell ρ 0.324 → 0.837, Harrell C 0.552 → 0.847,
log-RMSE 1.116 → 0.510, reward gain error 0.061 → 0.020, tail error 0.164 → 0.046; target cells ρ 0.514 → 0.835.

**Before/after, reward (live evaluator, default suite; golden bank, all 119 unit tests OK in 1,620 s):**

| candidate | v2 (before) | v3 (after) |
|---|---|---|
| initial / e3_relist / python_only_dgh4 / cond_* / instance_specific | 0.2000 | 0.2000 |
| **lean_dgh4** (DGH, closes (11,21)) | **0.2000** | **0.3921** |
| schema_recipe (search mode, live) | 0.2594 | 0.2497 |
| schema_farkas (8 hand-picked TRAIN certificates) | 0.2496 | 0.2014 |
| cert_close_10_22 (R3 finding 1, VR paid 0.3087) | 0.2000 | 0.2041 |
| e2_easy_certs (exploit) | 0.2068 | 0.2003 |
| e1_dgh_s2 (exploit; v2/S1 paid 0.2400) | 0.2000 | 0.2000 |
| unsound_row7 (mirror unsound, Lean = library) | 0.2000, battery 0 | 0.2000, battery 0 |
| forbidden-construct bank / L1–L3 sketches | 0 / ≤ 0.19 | 0 / ≤ 0.19 (unchanged) |

Offline re-score of R1's Lean-gated masks and R3's attack shapes with the final code (`rescore_benchmark.py`, `rescore.md`):
recipe 0.2594 → 0.2430 (frozen twin), DGH 0.2000 → 0.3921, recipe+DGH 0.2594 → 0.4118, f_weak 0.2286 → 0.2126; every realizable
exploit ≤ 0.2041; every R3 exploit shape at 0.2000–0.2223 (v2: up to 0.5892) except R3's tail sniper (0.3186; R3: not a hole, it
removes 4× the recipe's work in the targets' hardest decile).  `--check` reproduces R3's VR\*\* column exactly.
Changing only the labels (v2 on the new labels) moves DGH not at all (0.2000) and the recipe 0.2594 → 0.2556: **accurate
difficulty does not fix pitfall 2; the suite + formula change does.**

**Honest negatives.** (1) R1's C5 fails: the general recipe keeps 72 % of its v2 uplift (bar 80 %); v3 prizes closing a cell
(DGH: 4.5× the recipe's uplift for 0.4× its lower-bound work removed) over thinning many.  (2) GEN carries no weight in v3 (all
GEN cases ≤ 1,080 conflicts).  (3) A third of the censored target survivors are tied at the 2M ceiling.  (4) R2's dynamics
were measured on VR, not on the shipped v3.  (5) D5's cheaper-and-better "deepen to 50k first" was not run.

**Gate fix found on the way (gate v2.2).** The conditional entry point's generated `gateKillK_eq := rfl` ran out of heartbeats on
(13,19,123) and (16,17,134), so any `candidateF` program was L1 there; the harness now gives that one theorem 2M heartbeats.

## E28 — 2026-09-23 — Follow-ups to the batch-3 verifier: deepen-to-50k, mirror penalty, doc fixes

**Deepen target survivors to 50k before estimating** (D5's recommendation, EVALUATION.md 4.5).
`python -m zar_ub deepen M N 3 3 W --cap 50000 --jobs 12 --survivors` (new `--survivors` flag: skip cases
the proved library kills, since their label never enters the reward) on the five target tables;
script `experiments/E28_deepen50k/run.sh`, log `run.log`, pre-run table copies in the same folder (gitignored).

| target | censored survivors deepened | solved within 50k | still open (model label, floored at 50k) | wall |
|---|---|---|---|---|
| (12,18) 109 | 1,386 | 92 | 1,294 | 4.4 min |
| (10,23) 113 | 1,188 | 18 | 1,170 | 3.8 min |
| (11,23) 124 | 336 | 4 | 332 | 1.3 min |
| (13,19) 123 | 2,490 | 105 | 2,385 | 10.0 min |
| (16,17) 134 | 918 | 7 | 911 | 5.1 min |

226 of 6,318 (3.6 %) became exact; no SAT anywhere; records and baseline masks unchanged (checked against
the copies). As D5 predicted, the gain on *target* cells is small (their hard cases rarely finish within
50k: within-cell ranking 0.835 -> 0.850 in D5's held-out test); it is large on near-square cells. Survivors
still tied at the 2M ceiling after the pass: 280 / 203 / 253 / 1,049 / 683.

**Unsound-mirror twins.** E26 showed a program whose Python mirror is unsound but whose Lean is L5 tied
its sound twin and was reported as the best program in 3/8 runs. `reward.MIRROR_PENALTY = 0.02` is now
subtracted in that case, so the sound twin always ranks strictly above. `unsound_row7` and
`experiments/E6_evaluator/unsound_program.py` (Lean = the sound library) now score **0.18** (golden band
updated). Stale "unsound = 0" rows in `docs/build/B-reward.md` and `INTEGRATION.md` are annotated.

**Doc fix.** E27's "2,468 of 7,452 (33 %)" censored target survivors at the 2M ceiling is 2,468 of 6,452 (38 %).

## E29 — 2026-09-23 — Rebalancing: the closure bonus is for TARGET cells only

**Problem found in the E27 re-score.** With the closure bonus over TRAIN ∪ TARGET, DGH — which closes the
wide *practice* cell (11,21) (value already known) and helps no target (G_target 0.0005) — scored 0.3921,
**3.9x the uplift** of the general recipe (0.2497), which thins every target (G_target 0.067) and removes
2.5x more lower-bound work. The fix for "specialists are under-rewarded" had overshot.

**Offline weight sweep** (`experiments/E29_weights/sweep.py`; components are weight-independent, so every
weight vector is scored exactly on the E27 benchmark; G_gen reconstructed from the stored scores).
Criteria: P1 DGH and recipe uplifts within 2x of each other; P2 recipe+DGH above both; P3 every real
exploit below min(recipe, DGH); P4 tiny-table clears <= f_weak; P5 target-helping f_weak above every
real exploit. The shipped weights fail P1 (ratio 3.9). 111 feasible vectors exist; every one of them puts
(almost) no weight on a closure bonus over practice cells — a feasible region with closure weight >= 0.06
is empty. So the principled change, rather than a tuned weight vector: **Close = importance of the best
TARGET cell closed** (closing a target = a bound with zero SAT, the thesis objective); closing a practice
cell still pays through Depth (= a_I when closed). Weights unchanged.

**Result** (live evaluator, and `experiments/E29_weights/rescore_final.md` for the whole benchmark):

| program | E27 shipped | E29 |
|---|---|---|
| initial | 0.2000 | 0.2000 |
| DGH (`lean_dgh4`) | 0.3921 | **0.3105** |
| general recipe (`schema_recipe`, live) | 0.2497 | 0.2497 |
| recipe + DGH | 0.4118 | **0.3301** |
| f_weak (target-helping weak rule) | 0.2126 | 0.2127 |
| best real exploit | 0.2041 | 0.2041 |
| tail-index sniper (oracle shape, removes 4x the recipe's work) | 0.3186 | 0.3186 |

DGH/recipe uplift ratio 2.2 (live recipe; 2.6 vs the frozen 34-certificate recipe) — DGH still clearly
above the general rule, as a cell-closing argument should be, but no longer dominating it; all exploit
criteria still hold; monotonicity test passes. Unit test `test_closure_bonus_formula` updated (a TRAIN
closure pays Depth, not Close). C5 (recipe keeps >= 80 % of its old uplift) stays at 72 % against the frozen
recipe: its old credit came from easy cases and inflated 400k labels, which the new labels removed.

**E29 dynamics re-run with the SHIPPED reward** (E26's open item: R2 had measured VR, not the shipped
v3).  `experiments/E26_dynamics/run_shipped.sh`: the same synthetic-mutator harness and atom bank (R = general
recipe, D = DGH, C = one-cell certificates, T = subsumed target certificates, B = broken proof, U = unsound
mirror), live evaluator unchanged ("V0 live" = v3 + E28 penalty + E29 target-only closure), 8 seed pairs x
150 iterations (culling active).  Results `experiments/E26_dynamics/results/aggregate_E29.{md,json}`:

| reward (150 iterations, 8 seeds) | best score | best program contains R and D | population share R+D | population share D | best program has an unsound mirror (U) | audit violations |
|---|---|---|---|---|---|---|
| v2 (before batch 3) | 0.2594 | 1/8 | 0.31 | 0.31 | 1/8 | 0 |
| VR (R1's recommendation) | 0.3700 | 8/8 | 0.84 | 0.84 | 3/8 | 0 |
| **shipped (v3 + E28 + E29)** | **0.3367** | **8/8 (R+D+C every run)** | **0.90** | **0.97** | **0/8** | **0** |

The specialist is found by iteration ~9 and spreads; the best program is the general rule plus both
specialists in every run; the mirror penalty works (U-carrying programs score 0.3167 = best − 0.02 and are
never reported best).  Full unit suite after E28/E29: **119 tests OK** (700 s,
`experiments/E29_weights/unittest_final.log`).

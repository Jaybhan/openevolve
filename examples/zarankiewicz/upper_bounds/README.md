# Zarankiewicz upper bounds via evolved, Lean-verified pruning

MEng thesis system (Jay Bhan, advisor Srinivasan Raghuraman). Goal: use
evolutionary search (OpenEvolve) to discover *pruning arguments* that shrink
the SAT case split of Tan (2022) for z(m,n;s,t), with **every prune formally
verified in Lean 4**, every remaining case refuted by a **checked LRAT
certificate**, and every claimed bound a **Lean theorem** whose only hypotheses
are named, provenance-tagged facts.

```
                 ┌──────────────── OpenEvolve loop ────────────────┐
 initial_program ──► LLM edits NOTES + LEAN_SOURCE + SCHEMA_DATA + kill()   (genome, design §2)
                      │
                      ▼  evaluator.py  (LLM-free, deterministic, 2–90 s)
   stage 1  sandboxed candidate subprocess (no project writes, no network) → Python mask, battery
   stage 2  one Lean process: scan → elaborate → type-check → #print axioms → nonce-tagged #eval masks
            ladder L0–L5, sorry-hole auto-fill, HMAC-authenticated result cache
   stage 3  reward v3 (E27): verified floor 0.20 + work above 20k conflicts removed (family-balanced) + depth + closure; unverified ≤ 0.19
                 └───────────────────────────────────────────────────┘
 promotion: leanchecker replay → lean/ZarPrune/Evolved/E_<sha>.lean → accepted_prunes.jsonl
 closure:   survivors ──► cadical DRAT ──► drat-trim ──► LRAT ──► lrat-check
            + Closure.lean (verified enumerator, cover theorem) ──► lean/ZarPrune/Closures/Z_m_n_w.lean
```

## What is proved (Lean 4.34 + targeted Mathlib; axioms ⊆ {propext, Quot.sound, Classical.choice})

| module | content |
|---|---|
| `Sum/Basic/Prune/Prunes` (Mathlib-free core) | the statement (`HasKst`, `Valid`), `Prune` = kill + soundness, `upper_bound_of_cover`, sanity prunes |
| `Counting.lean` | Guy's Argument A (both sides), Argument D (exact "lightest columns" form), transposition, deletion lemmas, waterfilling optimality; prunes `argA argAT argD argDT argDelColWF argDelRowWF argWF` = `counting` |
| `Cond.lean` | `Fact`/`FactHolds`/`CondPrune` (bounds conditional on named facts), Argument I as `Prune.ofPrefixF`, `Prune.mono` |
| `Schemas.lean` | Farkas certificates over hidden pair codegrees (`Prune.ofFarkas`): the LLM supplies multipliers in `SCHEMA_DATA`, Lean checks them |
| `DGH.lean` | the Davies–Gill–Horsley inequality for all (m,n;s,t) as `argDGH` |
| `Closure.lean` | thinning, permutation action, sorting, Tan's enumerator with completeness, `survivors`, `upper_bound_succ_of_sorted_cover` |
| `Closures/Z_*.lean` | generated theorems: `z(9,9;3,3) ≤ 49` (unconditional, 17 LRAT-refuted cases, kernel-checked), `z(10,21) ≤ 106`, `z(11,19) ≤ 106`, `z(11,20) ≤ 111` (conditional on named Tan 2022 facts, zero cases) |

## Layout

| path | what |
|---|---|
| `zar_ub/` | engine (`known`, `partitions`, `cases`, `encoding`, `solve`, `difficulty`, `casetable`; E24 estimators `hardness_model` (the censored label), `hardness_progress`, `hardness_lookahead`, `hardness_sampling`), `lean_gate` (the trust gate), `reward`, `lemmas` (schema registry), `ledger` (fact provenance), `certify` (DRAT→LRAT), `closure` (Tier-1 theorems), `promote`, `closure_daemon`, `sandbox`, `cli` |
| `lean/` | ZarPrune library; `lean/Candidates/` gate scratch (gitignored); `lean/Attempts/` proof attempts and audits |
| `evaluator.py`, `initial_program.py`, `suite.py`, `run_candidate.py`, `config*.yaml` | the OpenEvolve example (`config_stub.yaml` = no-LLM stub server, `config_smoke_luna.yaml` = cheap paid smoke) |
| `tests/` | engine, gate (46 forbidden constructs), ledger, tables, lemmas, promotion, sandbox, and the golden adversarial bank (`tests/candidates/`, `tests/golden.json`); `tests/attacks/` = red-team PoCs |
| `experiments/LOG.md` | numbered experiment ledger E1…; `cost_ledger.md` = OpenRouter spend; `E*/` per-experiment scripts and results |
| `docs/` | `literature_review.md`, `design.md` (the contract), `build/*.md` (per-component build reports, `INTEGRATION.md`, `ATTACKS.md`), `lit/` per-source notes |
| `cache/` | case tables, certificates, gate cache, ledger (gitignored except manifests); `tools/` cadical, drat-trim, lrat-check |

## Quick start (no LLM, $0)

```bash
cd examples/zarankiewicz/upper_bounds; export ZAR_UB_NO_LLM=1
python -m unittest discover tests                      # ~10 min (one adversarial slow-kill candidate takes 240 s)
python evaluator.py initial_program.py                 # exactly 0.20: verified, kills nothing beyond the library
python -m zar_ub closure 9 9 3 3 50 --pure --tier 1    # z(9,9;3,3) <= 49 as a kernel-checked Lean theorem
python -m zar_ub closure 10 21 3 3 107 --trust tan2022 # zero survivors; theorem conditional on named Tan facts
python -m zar_ub table 12 18 3 3 109 --trust tan2022 --mode censored --kind target --baseline
experiments/no_llm/run_stub.sh 10 8123                 # OpenEvolve over a scripted stub LLM
python experiments/no_llm/run_synthetic.py --iterations 30
```
Lean one-off setup: `cd lean && lake exe cache get && lake build` (Mathlib v4.34.0 oleans, ~8 GB).

## Paid smoke runs (OpenRouter; every call is logged to `experiments/cost_ledger.md`)

```bash
python experiments/E7_llm_smoke/probe_one.py --model openai/gpt-5.6-luna --reasoning low   # one generation, then scored
experiments/E7_llm_smoke/run_smoke.sh 15 "$PWD/config_smoke_luna.yaml"                    # 15-iteration OpenEvolve run
```

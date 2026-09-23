# Final design: Lean-verified pruning arguments, evolved by OpenEvolve, for the SAT case split of z(m,n;3,3)

**Status.** Final design for the system of `docs/proposal_section2.md`. Synthesised 2026-09-21 from three candidate designs and two judge reports (`docs/design_candidates/`): the spine is candidate 2 (certificate-first; ranked first by both judges), with every `keep` item from candidates 0 and 1 grafted in and every judge-listed weakness resolved (§0 lists the resolutions). All paths are relative to `/Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/upper_bounds/` unless absolute. Ground truth is the running code (engine `zar_ub/`, Lean library `lean/ZarPrune`, experiments E1–E10 in `experiments/LOG.md`) and the literature review `docs/literature_review.md` (cited as [LR §x]).

---

## 0. Synthesis record: what was taken from where, and which weaknesses this document fixes

| decision | source | fixes |
|---|---|---|
| A bound *is* one Lean theorem in a harness-owned file (`Closure.lean`), never a report | D2 | D0/D1 weakness "claim is a report until M8/M9" |
| `survivors` is *defined in Lean* as `(genRows × genCols).filter (¬kill)` over a verified enumerator; the harness solves the list Lean computed; the per-instance kernel obligation is only `survivors_eq : survivors P p = <literal list>` | D1 (Lean-defined survivors) + D2 (enumerator completeness `mem_genParts`) | D2 weakness "kernel `decide` over the product is unmeasured": the obligation is now one list equality, its cost is measured first (T-7, go/no-go), `Nat.choose` is replaced in kills by a GMP-accelerated `chooseMul` (§4.6), and the `+native` fallback is a *named tier* (Tier-1n) printed in the report |
| Per-fact hypotheses (`Fact`, `FactHolds`, `CondPrune P facts`, `discharge`) instead of one `BoundTable.Sound` | D1/D2 | D0 weakness "coarse table hypothesis cannot say which facts were load-bearing" |
| Trust tiers `proved-here ⊂ tan2022 ⊂ unreviewed-2026`, claim rule `tan2022` or better, 2026 claims in a targets-only file, mechanical monotonicity checker on every ledger write | D1 + D0 | D0 weakness "table-mode facts enter the survivor set outside Lean" — in this design the generator's prefix bounds are `Prune.ofPrefix` instances with the same `Fact` hypotheses, so a table-mode closure names them |
| Nonce-tagged `MASK <nonce> k BEGIN/END`, every `#` command and `IO` forbidden, declared-name blacklist, `autoImplicit false`, NFKC normalisation, comment-stripped **and** raw scan, `leanchecker` at promotion, `PIPELINE_BUG` halts the run | D0 + D1 + D2 | D2 weakness "no nonce" |
| Verified floor 0.20; unverified `min(0.19, …)`; L4 = 0; no `sorry` proof ever earns kill mass; `g` (generality) dropped from the score | D0 + D2 | D0 weakness "generality is free score"; D2 weakness "×0.9 mirror penalty taxes a verified prune" — **the penalty is removed**; the mirror only feeds cascade stage 1 and an artifact |
| `SCHEMA_DATA` channel with explicit plumbing through a declarative lemma registry (`zar_ub/lemmas.py`) and a harness-owned `schemaPrune` term; Python search for schema data inside the EVOLVE block is *allowed* and stated | D2 (schemas) + D1 (registry) | D2 weakness "G2 rejected but schema channel unplumbed"; D0 weakness "no proof-free way to earn verified credit in v1" |
| MAP-Elites `["proven_gain", "lean_ladder"]` (10×6 = 60 cells); `kill_novelty` reported and switched on as a third axis only once ≥ 5 prunes are in the accepted ledger | D2, amended | D2 weakness "300-cell grid vs population 40–60; novelty ledger empty at start" |
| Deterministic `evaluate()`: harness-side tactic auto-fill stays (it is $0 and deterministic); the *paid* repair loop moves out of `evaluate()` into `tools/forge.py` | D2 auto-fill + D1 forge | D2 weakness "repair loop makes evaluate() non-deterministic" |
| Sketch holes classified by Lean, not only by regex: the regex says where `sorry` is *permitted*; the axiom audit (`sorryAx` present, nothing else foreign) and the per-declaration error map decide the ladder level | D0/D2 | D0/D2 weakness "regex-only sketch rule" |
| Censored target difficulty clipped to `[cap, 20·cap]`; `censored_share` reported; the daemon deepens target labels (200k pass) so censoring shrinks over the run | D0 schedule + D2 | D2 weakness "hard_killed earns extrapolated credit" |
| Milestone order: evaluator v2 + gate hardening first (loop useful on day 1 with a Tier-2 report), `Closure.lean` second, staged internally; tiers printed in every report | D2, reordered | D2 weakness "Closure.lean 600–900 lines on the critical path of M1" |
| Golden adversarial bank in `tests/candidates/` + `tests/golden.json`; `SyntheticMutator`/`ReplayLLM` via `init_client`; **and** `tools/stub_llm.py` HTTP server so the real OpenAI client path is exercised | D0 + D1 + D2 | — |
| 5 % seeded kill audit (solve a sample of Lean-killed cases to completion) — **offline, inside closure**, never inside `evaluate()` | D1 | D1 weakness "SAT inside every evaluation" (not adopted) |
| Difficulty: escalating schedule `2k → 20k → 200k → 2M`, censored = `max(cap, f̂(c2k))`, refit on ladder cells, shared-sample (CRN) estimation: uniform `N = 500` when `3,000 < |S_I| ≤ 50,000`, stratified `N = 2,000` above; `table_hash` in every metric dict; LRAT length as a sanity label | D0 + D2 | — |
| SAT never in `evaluate()`; closure daemon triggered by estimated remaining work or survivor count; round-robin conflict escalation across survivors; stage-3 cubing with LRAT cover certificate and parent-capped accounting | D0 + D1 + D2 | D1 weakness "cuts/sub-cube kinds enlarge the trusted base" — no cuts, no sub-cube kinds in v1; cubing appears only inside a single case's certificate |
| DGH-attackable cells `(13,17,117)`, `(13,18,122)`, `(15,17,133)` in TARGET so counting-type prunes can score early; DGH(4) hand-proved as a milestone if the LLM does not find it | D0 | D0/D2 weakness "0.20 plateau" |
| Two-stage prompting via `prompt.template_variations` (architect/filler instruction variants) — with the honest statement that OpenEvolve does not route prompts to models; the $16 probes decide direct vs sketch by verified-prune rate per dollar | D0 + D2 | D0 weakness "architect/filler split relies on routing OpenEvolve cannot do" |
| Cost ledger before/after every paid call; hard stop at $13 spent; smoke scripts refuse to start below $2 remaining | D0/D1/D2 | — |
| Stale-claim check: the `#eval List Bool` truncation is **already fixed** in `zar_ub/lean_gate.py` (MASKLINE string emission, lines 167–174); T-2 re-verifies `lean_ok = 1.0` on the baseline before anything else | judge 0 | D2 weakness "stale bug claim" |
| Not adopted from D1: policy genome with Python callables, SAT inside evaluation, CPU-time cost ratios, provisional pysat UNSAT credit, evolvable cuts, counters-as-base-variables re-encoding (would invalidate the 1,571 E10 labels and E9 certificates). A policy loop over the accepted library is listed as a later ablation (§12, M9). | — | D1 weaknesses (sandbox hole, noise, misalignment with the LLM-writes-Lean question) |

---

## 1. Overview and claims

### 1.1 The end product, stated first

A new upper bound is one Lean declaration, elaborated by the Lean 4.34.0 kernel in a file the LLM never touches (`lean/ZarPrune/Closures/Z_12_18_109.lean`, generated by `zar_ub/closure.py`):

```lean
import ZarPrune
import ZarPrune.Closure
namespace ZarPrune.Closures
open ZarPrune

abbrev P : Params := ⟨12, 18, 3, 3, 109⟩

/-- external facts used, one hypothesis each (Tier-1 and Tier-0) -/
abbrev F_12_17 : Fact := ⟨12, 17, 3, 3, 103, "tan2022"⟩
abbrev F_11_18 : Fact := ⟨11, 18, 3, 3, 101, "tan2022"⟩

/-- the prune: proved library ∪ conditional neighbour prunes ∪ accepted evolved prunes ∪ schema instances -/
def prune (h1 : FactHolds F_12_17) (h2 : FactHolds F_11_18) : Prune P :=
  Prune.ofList P [counting P, (argDelColF P F_12_17).discharge h1, (argDelRowF P F_11_18).discharge h2,
                  Evolved.e_0a3f P, Prune.ofFarkas P (baseIneqs P) [[1,1,0,2]]]

/-- the list the SAT solver actually refuted (literal, generated) -/
def survList : List (List Nat × List Nat) := [([7,7,6,6,6,6,6,6,6,6,6,6],[…]), …]   -- 1,404 entries

/-- the only per-instance computation the kernel checks: the Lean-defined survivor list is this literal -/
theorem survivors_eq (h1) (h2) : survivors P (prune h1 h2) = survList := by decide      -- Tier-1; `+native` → Tier-1n

theorem z_12_18_le_108
    (h1 : FactHolds F_12_17) (h2 : FactHolds F_11_18)
    (Hrefuted : ∀ q ∈ survList, ∀ A : Mat 12 18, sortedProfileOf A = q → ¬ Valid P A) :   -- Tier-1: paired with LRAT hashes in the manifest
    ∀ A : Mat 12 18, ¬ HasKst P A → weight A ≤ 108 :=
  upper_bound_succ_of_sorted_cover P 108 rfl (prune h1 h2) survList (survivors_eq h1 h2) Hrefuted
end ZarPrune.Closures
```

The four ingredients and who produces them:

| ingredient | what it is | produced by | checked by |
|---|---|---|---|
| `prune` | a `Prune P` term: computable `kill : Profile m n → Bool` + `sound : ∀ A, kill (profileOf A) = true → ¬ Valid P A` | evolutionary loop (evolved prunes, schema data) + hand-proved library + conditional neighbour prunes | Lean kernel; axiom audit; `leanchecker` at promotion |
| `survivors` | `((genRows P) ×ˢ (genCols P)).filter (fun q => !p.kill (mkProfile q))` — **defined in Lean** over the verified enumerator | `Closure.lean` (harness-owned) | `mem_genParts` completeness theorem (once, generic) |
| `survivors_eq` | the Lean-defined list equals the literal list the harness refuted | `zar_ub/closure.py` emits the literal from the Lean `#eval` | kernel `decide` (Tier-1) or `decide +native` with a named axiom (Tier-1n) |
| `Hrefuted` | each listed case contains no valid matrix | CaDiCaL → DRAT → LRAT per case | Tier-1: `drat-trim` + `lrat-check`, hypothesis paired with certificate hashes in `manifest.json`; Tier-0: `Encode.lean` + `Std.Tactic.BVDecide.LRAT.check_sound` |

`upper_bound_of_cover` in `lean/ZarPrune/Prune.lean` already has this shape; `Closure.lean` (§4.6) adds exact-weight thinning, permutation invariance, sorting and the enumerator.

### 1.2 Claims

* **C1 — Soundness is independent of the LLM, the Python mirror, the difficulty estimator and the reward.** Every kill that removes a case is `#eval` of a Lean term whose `sound` field elaborated with axioms ⊆ `{propext, Quot.sound, Classical.choice}`; every claimed bound is a Lean theorem whose remaining hypotheses are external facts (per-fact, named, provenance-tagged) and LRAT verdicts (Tier-1) or nothing beyond named `_native` axioms (Tier-0). Reward hacking can waste compute; it cannot produce a false bound.
* **C2 — Lean is not the bottleneck.** One Lean process checks a candidate against the whole suite and evaluates its kill mask on every case in 1.5–4.4 s with Mathlib-backed `Counting.lean` imported [E5, E6]; an LLM generation takes 30–120 s.
* **C3 — The reward is the difficulty-weighted fraction of the proved library's remaining refutation work that a candidate's *Lean* mask removes**, with difficulty a function of the branch CNF alone (precomputed, cached, read-only). Verified candidates always outrank unverified ones (floor 0.20 vs cap 0.19).
* **C4 — Pruning is not adding.** `Prune` can only remove empty cases; symmetry breaking lives in `cover`/`refuted` with permutation witnesses (`Demo.notDescending_unsound` is the executable statement).
* **C5 — The first deliverables need no SAT and no LLM.** With Tan 2022 neighbours as explicit hypotheses the generator plus proved deletion/prefix prunes leave zero cases at `(10,21,107)`, `(11,19,107)`, `(11,20,112)` [measured 2026-09-21], so `z(10,21) ≤ 106`, `z(11,19) ≤ 106`, `z(11,20) ≤ 111` become Lean theorems conditional on Tan; `(9,23,104)` (244 cases) is the first SAT target.
* **C6 — The whole loop runs at $0** (replay/synthetic `init_client` clients and an OpenAI-compatible stub server); the $16 buys probes only.

### 1.3 Pipeline

```
 initial_program.py ──► OpenEvolve (islands, MAP-Elites, diff mode) ──► LLM edits NOTES + LEAN_SOURCE + SCHEMA_DATA + kill()
        │
        ▼ evaluator.py  (2–6 s; no SAT; no LLM; deterministic)
   stage 1  run_candidate.py in a subprocess on a COPY of cache/: Python mask, battery, SCHEMA_DATA validation, scan
   stage 2  one Lean process: wrapper (nonce, autoImplicit false) → elaborate → typecheck gateInstK → #print axioms
            → nonce-tagged MASKLINE masks on TRAIN ∪ GEN → ladder L0–L5 → auto-fill of sketch holes (deterministic)
   stage 3  masks on TARGET tables → combined_score, metrics, artifacts
        │
        ├─► promotion (zar_ub/promote.py): leanchecker replay + clean re-run → lean/ZarPrune/Evolved.lean → accepted_prunes.jsonl
        └─► closure daemon (zar_ub/closure_daemon.py): survivors of the promoted library → round-robin CaDiCaL escalation
            → DRAT → drat-trim → LRAT → lrat-check → 5 % kill audit → Closures/Z_m_n_w.lean + closure_report.md + ledger row
```

---

## 2. Genome: exactly what the LLM edits

### 2.1 One Python file, four coupled objects

`initial_program.py` has one `EVOLVE-BLOCK`; `run_candidate.py` imports it in a subprocess and returns JSON `{lean_source, schema_data, notes, mask, error}`.

1. **`LEAN_SOURCE : str`** — spliced into `namespace ZarPrune.Cand` after `import ZarPrune`. Must define `def candidate (P : Params) : Prune P` (general) or `def candidate : Prune target` (instance-specific; `target` is injected). Helper prunes `def myPrune (P : Params) : Prune P where …` and lemmas are free-form. `CondPrune P facts` candidates (§8.2) are accepted through `candidateF (P : Params) : CondPrune P facts` with the wrapper discharging facts from the run's ledger.
2. **`SCHEMA_DATA : dict`** — JSON-serialisable parameters for once-proved schemas: `{"farkas": [[y_1, …, y_K], …], "residue": [{"g": 3, "marked": [0,1], "exceptional": [6,7]}, …], "prefix": [{"k": 5}]}`. Validated in Python against `zar_ub/lemmas.py` schemas; instantiated by the harness into a `schemaPrune` term (§2.3). **The LLM may write Python inside the EVOLVE block that searches for these parameters** (exact rational LP for Farkas multipliers, enumeration of marked sets against the case bank in the artifacts) — this is the search-mode leverage [LR §8.3], paid in Lean once per schema, not per iteration.
3. **`kill(m, n, s, t, w, rows, cols) -> bool`** — Python mirror. Feeds cascade stage 1 (battery pre-screen, empirical band for *unverified* candidates) and the `agreement` metric. Never prunes, never penalised.
4. **`NOTES : str`** — the natural-language argument; shown back in the prompt; never scored.

### 2.2 `initial_program.py` skeleton

```python
"""Prune library for the Zarankiewicz upper-bound search: prove z(m,n;s,t) < w by a case split.

A CASE is (rows, cols): non-increasing vectors, len(rows)=m (entries<=n), len(cols)=n (entries<=m),
sum(rows)=sum(cols)=w.  Every case must be EMPTY: killed by a Lean-verified prune or refuted by a SAT
solver with a checked certificate.  You evolve the prunes.

CONTRACT.  kill(...) may return True ONLY if no K_{s,t}-free m x n 0/1 matrix has exactly these sums.
Symmetry breaking ("assume sorted", "transpose") is NOT a prune.  Killing a realizable case scores 0.

WHAT EARNS CREDIT.  Only the Lean-evaluated kill of `candidate` (plus the harness-instantiated schemas
from SCHEMA_DATA).  Credit = difficulty-weighted fraction of the cases that SURVIVE the already-proved
library `ZarPrune.counting` (Arguments A, D both sides; deletion with the waterfilled counting bound)
that you kill.  Re-proving those earns nothing.  A verified candidate that kills nothing scores 0.20;
no unverified candidate can exceed 0.19.  Artifacts list the hardest surviving cases, Lean errors with
goals, and unfilled `have` holes.

LEAN API (lean/API.md is authoritative; Mathlib IS available through `import ZarPrune`):
  Basic:    Params{m,n,s,t,w} Mat rowSum colSum weight HasKst Valid Profile{row,col} profileOf
  Prune:    Prune{name,kill,sound} Prune.never/or/ofList/mono/transposed  CondPrune Fact FactHolds discharge
  Sum:      sumFin allFin sumFin_eq_sum sumFin_swap sumFin_le allFin_iff not_allFin_elim
  Counting: support rowSupport card_support hasKst_of_subsets budget_general colBudget rowBudget
            rowLocalBudget argA argAT argD argDT deleteCol deleteRow weight_deleteCol not_hasKst_deleteCol
            choose_tangent waterfillBound sum_le_waterfillBound argDelCol argDelRow argWF counting
  Schemas:  Prune.ofFarkas Prune.ofResidue Prune.ofPrefix  (proved once; you supply SCHEMA_DATA)
  Mathlib:  Finset.sum_le_sum sum_comm card_filter powersetCard, Nat.choose lemmas, omega decide simp linarith ring
FORBIDDEN: import; sorry outside `have h : T := by sorry` holes in a `sound` proof; axiom; native_decide;
  unsafe; partial; implemented_by; extern; csimp; opaque; noncomputable; macro/syntax/elab/notation;
  every `#` command; IO; Lean.*; set_option other than maxHeartbeats/maxRecDepth; end Cand/ZarPrune;
  redefining Valid/HasKst/Params/Profile/Mat/weight/rowSum/colSum/Prune/CondPrune/counting.
"""
from math import comb

# EVOLVE-BLOCK-START
NOTES = r"""Start: the proved counting library only."""

LEAN_SOURCE = r'''
/-- Pattern for a NEW prune: `kill` on the profile, `sound` via library lemmas. This one kills nothing. -/
def examplePrune (P : Params) : Prune P where
  name := "example (kills nothing)"
  kill := fun _ => false
  sound := by intro A h; simp at h

/-- The evolved library. Keep `counting P` first; append new prunes. -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [counting P, examplePrune P]
'''

SCHEMA_DATA = {"farkas": [], "residue": [], "prefix": []}


def kill(m, n, s, t, w, rows, cols):
    """Python mirror of candidate.kill (library part mirrors ZarPrune.counting)."""
    if sum(comb(c, s) for c in cols) > (t - 1) * comb(m, s):
        return True                                                   # argA
    if sum(comb(r, t) for r in rows) > (s - 1) * comb(n, t):
        return True                                                   # argAT
    r0, c0 = rows[0], cols[0]
    if r0 and sum(comb(c - 1, s - 1) for c in sorted(cols)[:r0] if c) > (t - 1) * comb(m - 1, s - 1):
        return True                                                   # argD
    if c0 and sum(comb(r - 1, t - 1) for r in sorted(rows)[:c0] if r) > (s - 1) * comb(n - 1, t - 1):
        return True                                                   # argDT
    return False                                                      # your prunes go above
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    print(kill(9, 9, 3, 3, 50, (6, 6, 6, 6, 6, 5, 5, 5, 5), (6, 6, 6, 6, 6, 5, 5, 5, 5)))
```

Design notes: the baseline inside the genome is the *proved* `counting P` (E6 lesson: the scoring baseline must equal what Lean proves), so the initial program scores exactly 0.20 and every increment is a new proved kill. `zar_ub/lean_api.py` regenerates `lean/API.md` and the docstring block is regenerated from it (`python -m zar_ub api --docstring initial_program.py`) whenever the library changes; the current `config.yaml` system message ("Mathlib is NOT available") is stale since E5 and is replaced.

### 2.3 Schema plumbing (the SCHEMA_DATA channel, made concrete)

`zar_ub/lemmas.py` is a **declarative** registry (no callables):

```python
FAMILIES = {
  "farkas":  Family(lean="ZarPrune.Prune.ofFarkas", params={"ys": "List[List[Nat]]"}, arity_check=lambda ys, P: all(len(y) == n_base(P) for y in ys),
                    mirror=farkas_kill, doc="Σ y_k·lhs_k > Σ y_k·rhs_k for some multiplier vector y (P11)"),
  "residue": Family(lean="ZarPrune.Prune.ofResidue", params={"g": "Nat", "marked": "List[Nat]", "exceptional": "List[Nat]"}, mirror=residue_kill, doc="marked-row deficit residues mod g (P14)"),
  "prefix":  Family(lean="ZarPrune.Prune.ofPrefixF", params={"k": "Nat"}, mirror=prefix_kill, doc="top-k columns sum ≤ U(m,k) with U from the ledger (P4; conditional)"),
  "evolved_<sha>": ...   # promoted candidates register here with their Lean name and Python mirror
}
```

The gate footer (harness-owned, outside `namespace Cand`) contains, per instance `K`:

```lean
def ZarPrune.Cand.schemaK : ZarPrune.Prune ZarPrune.Cand.targetK :=
  ZarPrune.Prune.ofList _ [ZarPrune.Prune.ofFarkas _ (ZarPrune.baseIneqs _) [[1,1,0,2],[0,3,1,0]],
                           ZarPrune.Prune.ofResidue _ 3 [0,1] [6,7]]
def ZarPrune.Cand.gateInstK : ZarPrune.Prune ZarPrune.Cand.targetK :=
  ZarPrune.Prune.or (by first | exact ZarPrune.Cand.candidate | exact ZarPrune.Cand.candidate ZarPrune.Cand.targetK) ZarPrune.Cand.schemaK
```

Malformed `SCHEMA_DATA` (unknown family, wrong arity, non-`Nat` entries) fails stage 1 with `schema_error` in artifacts and `schemaK := Prune.never`; it never reaches Lean. `schema_gain` (the gain attributable to `schemaK` alone) is reported so the thesis can separate "the LLM proved a new argument" from "the LLM found good parameters".

### 2.4 Genome levels

| level | evolved? | how |
|---|---|---|
| G1 kill + proof | yes | `LEAN_SOURCE` (the interpretable artefact) |
| G2 search over a parametric family | yes, for schema parameters only | Python in the EVOLVE block computes `SCHEMA_DATA`; soundness paid once in `Schemas.lean` |
| G3 generalizer | yes | suite mixes cells; `candidate (P : Params)` transfers for free |
| encoding, enumerator, difficulty labels, wrapper, `Closure.lean`, `Schemas.lean` | never | trusted base (§4.1) |

### 2.5 Prompt (system message, `config.yaml`)

Keep the current framing (case, prune ≠ addition, reward). Replace the API paragraph with the docstring above; add (i) "the baseline is proved — look at the hardest surviving profiles in the artifacts and find a *new* reason they are empty: DGH's `v = s−1` rounding inequality, cross-side arguments coupling `rows` and `cols`, residue/overlap arguments mod a small `g` on the exceptional columns, Farkas combinations (put multipliers in `SCHEMA_DATA`)"; (ii) the full 12-line text of `argA` as the worked example of `where kill := … sound := by …`; (iii) "if you cannot finish `sound`, leave typed `have h : … := by sorry` holes; the harness tries to fill them and reports the goals of those it cannot"; (iv) keep every verified prune; fix Lean errors rather than deleting. Two instruction variants via `prompt.template_variations` (`propose`: new prune + typed skeleton; `fill`: close the reported holes of the current program). OpenEvolve does not route a variant to a model; the ensemble weights only set the model mix.

---

## 3. Case decomposition and SAT encoding: fixed vs evolvable

### 3.1 Decomposition (fixed; `zar_ub/partitions.py`, `zar_ub/cases.py`; Lean twin in `Closure.lean`)

Instance `P = (m,n,s,t,w)`. "≥ w" is equivalent to "exactly w" (thinning, P19). Case index = (non-increasing row-sum vector, non-increasing column-sum vector) of `w`. Generation: Tan's Algorithm 1 with two prefix filters — Argument A on both sides (`Σ_j C(c_j,s) ≤ (t−1)C(m,s)`) and Argument I on *proper* prefixes only (`c_1+…+c_k ≤ U(m,k)` for `k < n`; E1 circularity fix, enforced by `k+1 < nparts` in Python and by `k < n` in the `ofPrefixF` statement). `U` comes from the ledger with a provenance filter: `--pure` uses only the proved waterfilled counting bound; `--trust tan2022` (default) adds `data/exact_33.csv` with each use recorded as a `Fact`. Both sums fixed (Tan) rather than column histograms only (dfield) because cross-side prunes need both sides; this is measured, not assumed (T-10 ablation on `cols`-only survivors count).

### 3.2 CNF encoding of one case (fixed; `zar_ub/encoding.py`)

Grid variables `x_{ij} = i·n + j + 1`; `K_{s,t}`-freeness in aux form (`y_{R,j}` one-directional per `s`-subset `R` of rows, `AtMost(t−1)` sequential counter); exact row/column sums by `CardEnc.equals` sequential counters; double-lex within equal-sum blocks (sound as an *addition* by Tan Thm 3.2; incomplete, KNW). Deterministic; `cnf_sha1` recorded in every certificate manifest. **Not evolvable in v1**: no evolved cuts (they would change the formula the certificate refutes and enlarge the trusted encoding — the D1 weakness); an evolved inequality removes solver work only as a `Prune` on a whole case. Verified cuts and an enriched case index (pair codegree) are v2 items (§12, M10) and require a registered Lean statement plus an entry in the encoding-completeness clause list before the harness accepts the cut name.

### 3.3 Instances (measured 2026-09-21)

| cell | w | status | pure pairs | table pairs | after `counting` (table) |
|---|---|---|---|---|---|
| (9,23) | 104 | 103 †dfield | 411 | 244 | 94 |
| (10,21) | 107 | 106 by deletion from Tan z(9,21)=96 | 185 | **0** | 0 |
| (10,22) | 111 | 110 †dfield | 220 | 3 | 3 |
| (10,23) | 113 | 112 †dfield (13 SAT/MIP profiles, 25 GB) | 5,850 | 4,818 | 1,189 |
| (11,19) | 107 | 106 by deletion from z(11,18)=101 | 2,970 | **0** | 0 |
| (11,20) | 112 | 111 by two deletions | 198 | **0** | 0 |
| (11,23) | 124 | 123 †dfield | 822 | 822 | 336 |
| (12,17) | 104 | 103 Collins 2016 | 52,264 | 968 | 617 |
| (12,18) | 109 | 108 †hou/afrasyab | 40,626 | 2,562 | 1,518 |
| (12,23) | 135 | 134 †dfield | 75 | 75 | 20 |
| (13,19) | 123 | open [118,122] | 72,693 | 4,130 | 2,648 |
| (16,17) | 134 | open [132,133] | 711,018 | 2,139 | 993 |

Training ladder (exact, all-UNSAT at `w = z+1`, E10 conflict labels): `(9,9,50)` 36, `(9,10,55)` 45, `(10,10,61)` 25, `(10,11,65)` 195, `(11,11,70)` 625, `(11,12,75)` 420, `(12,12,81)` 225 — 1,571 cases; battery tables at `w = z`; censored tables `(10,14)`, `(12,13)`, `(13,13)` cached.

### 3.4 Trusted base, by tier (printed verbatim in every closure report)

| # | component | tier |
|---|---|---|
| T1 | Lean 4.34.0 kernel; `leanchecker` replay on closure files and on `Evolved.lean` | 0, 1, 1n |
| T2 | `ZarPrune/Basic.lean` (~60 lines: the statement) | 0, 1, 1n |
| T3 | Mathlib v4.34.0 modules imported by `Counting.lean` | 0, 1, 1n |
| T4 | `Closure.lean` (thinning, `act`, sorting, `genParts` + `mem_genParts`, `survivors`, `upper_bound_succ_of_sorted_cover`), `Schemas.lean` — harness-owned, human-audited at statement level | 0, 1, 1n |
| T4n | `Lean.ofReduceBool` (one named `_native` axiom per `survivors_eq` discharged by `decide +native`) | 1n only |
| T5 | `drat-trim` + `lrat-check` verdicts, each a named hypothesis `Hrefuted` paired with `cnf_sha1`/`lrat_sha1` in `manifest.json`; `encoding.py` completeness (case SAT ⇐ matrix exists) | 1, 1n |
| T5′ | `Encode.lean` completeness theorem + `LRAT.check_sound` evaluated natively (named `_native` axiom per branch) | 0 |
| T6 | `exists_doubleLex` (block double-lex reachable inside a sorted case) | 0 (Tier-1 lists the lex clauses under T5) |
| T7 | external facts, each a `Fact` hypothesis with provenance tag | 0, 1, 1n |
| T8 | Tier-2 only: `partitions.py` completeness and Python cover (no Lean closure file; **no bound is claimed**, the report says "reduction only") | 2 |

Tier-1 is the committed thesis deliverable; Tier-1n is the measured fallback; Tier-0 is milestone M8; Tier-2 is what the loop produces on day 1 before `Closure.lean` lands.

---

## 4. Verification pipeline

### 4.1 Trust domains

* **Candidate domain** (LLM-written, `namespace ZarPrune.Cand`): axioms ⊆ `{propext, Quot.sound, Classical.choice}` (plus exactly `sorryAx` in sketch mode, never trusted); source blacklist; statement fixed by the harness wrapper; compiled from source only; `leanchecker` before promotion.
* **Harness domain** (`Closure.lean`, `Closures/*.lean`, refutation modules): may carry the named `_native` axioms of `decide +native` / `LRAT.check_sound … (by native_decide)`; the harness records the expected axiom names at generation time and re-matches them in the audit; every such axiom is listed in the closure report.

### 4.2 The gate, exact procedure (`zar_ub/lean_gate.py::run_gate`)

Input: `LEAN_SOURCE`, validated `SCHEMA_DATA`, the suite's instances and case lists. Output: `GateResult` with `ladder ∈ {0,…,5}`, `masks[K]`, `axioms`, `errors`, `holes`, `filled`, `seconds`.

```
S0  normalise:  src = unicodedata.normalize("NFKC", LEAN_SOURCE); strip line and block comments → src_nc
S1  scan:       for pat in FORBIDDEN: if pat matches src_nc or src (raw): → L0 (forbidden), score 0.
                sorry rule: every `sorry` in src_nc must be the whole body of `have <id> : <T> := by sorry`
                located inside a `sound := by` block or a `theorem … : … → ¬ ZarPrune.Valid …` body; otherwise L0.
                declared-name rule: regex ^\s*(def|theorem|lemma|abbrev|structure|inductive|instance|class)\s+
                (Valid|HasKst|Params|Profile|Mat|weight|rowSum|colSum|profileOf|Prune|CondPrune|Fact|FactHolds|counting|
                 target\d*|gateInst\d*|gateProfile\d*|schema\d*)\b → L0.
                set_option: only maxHeartbeats ≤ 400000 and maxRecDepth ≤ 4096.
S2  cache:      key = sha1(NFKC src + json(SCHEMA_DATA) + suite table_hash + wrapper version); hit → return cached GateResult.
S3  wrapper:    build_gate_file(): header `import ZarPrune`, `set_option autoImplicit false`, `namespace ZarPrune`,
                `namespace Cand`, `abbrev target : Params := …`; candidate verbatim; `end Cand`, `end ZarPrune`;
                footer per instance K: schemaK, gateInstK (Prune.or candidate schemaK), `#print axioms gateInstK`,
                gateProfileK, and the mask blocks
                  #eval IO.println "MASK <nonce> K BEGIN"
                  #eval IO.println ("MASKLINE:" ++ String.intercalate "," ((cases_chunk).map fun p => if gateInstK.kill (gateProfileK p.1 p.2) then "true" else "false"))   -- chunks of 200
                  #eval IO.println "MASK <nonce> K END"
                nonce = 16 hex chars from os.urandom per evaluation.  Written to lean/Candidates/cand_<sha>.lean.
S4  elaborate:  `lake env lean <file>` in lean/, timeout 240 s, `ulimit -v 4 GB`; parse `file:line:col: error:` lines;
                attribute errors to declarations via _decl_lines (candidate) / footer (harness).
S5  ladder:     L0  parse error in the candidate, or timeout with no declaration elaborated
                L1  candidate parses but gateInstK fails to elaborate on every instance, or some `kill` has an error,
                    or a mask is missing/short/timed out ("kill too slow")
                L2  all kills type-check, every hole statement type-checks, holes remain, no other errors
                L3  as L2 after auto-fill closed ≥ 1 hole but not all
                L4  no elaboration error but #print axioms shows anything outside the allowed set
                    (other than exactly sorryAx in sketch mode), or a construct slipped S1 → score 0
                L5  gateInstK elaborated on every instance, axioms ⊆ allowed, every mask has exactly len(cases) entries
S6  masks:      parse only the text between the exact `MASK <nonce> K BEGIN` / `END` lines; concatenate MASKLINE entries;
                len must equal the case count, else L1.
S7  auto-fill (sketch mode, deterministic, $0): for each hole in order, try tactics
                [omega, simp_all, decide, linarith, nlinarith, positivity, grind,
                 exact <lemma> for lemma in a fixed list (colBudget rowBudget rowLocalBudget weight_deleteCol
                 weight_deleteRow choose_tangent sum_le_waterfillBound budget_general hasKst_of_subsets)]
                each under `set_option maxHeartbeats 50000 in`; a filled hole is replaced in a HARNESS-OWNED copy;
                if all holes fill, the copy re-enters S3–S6 as a normal candidate (its source becomes the artifact
                `lean_source_filled` and the program's Lean source at promotion).
```

The MASKLINE string emission is **already in the code** (`lean_gate.py` lines 167–174); what this design adds is the nonce, the ladder, the cache, the schema footer, NFKC, the declared-name rule and auto-fill. T-2 re-verifies `lean_ok = 1.0` on the baseline before any other work (the "0.6" figure in candidate 2 predates the MASKLINE fix).

### 4.3 Forbidden constructs (`FORBIDDEN` in `lean_gate.py`, final list)

Current: `sorry` (except S1's hole rule), `admit`, `native_decide`, `axiom`, `unsafe`, `implemented_by`, `extern`, `csimp`, `opaque`, `partial`, `import`, `macro`, `macro_rules`, `elab`, `syntax`, `notation`, `initialize`, `#exit`, `Lean.`, `IO`, `ofReduceBool`, `end ZarPrune`, `end Cand`, non-whitelisted `set_option`.
Added: `+native`, `+kernel`, `trustCompiler`, `run_cmd`, `run_tac`, `run_elab`, `open Lean`, `attribute [`, `@[simp]`/`@[csimp]`/`@[implemented_by]` on library names, `local instance`, `instance : Decidable`, `noncomputable`, `dbg_trace`, `trace`, `logInfo`, every `#` command (`#eval #print #check #reduce #guard #synth #exit #help`), `namespace` (no re-opening), `deriving` other than `Repr, DecidableEq`, `decreasing_by`, and the declared-name rule of S1. Scan runs on NFKC-normalised, comment-stripped text **and** on the raw text.

### 4.4 Axiom audit and promotion

`#print axioms gateInstK ⊆ {propext, Quot.sound, Classical.choice}`; in sketch mode exactly `sorryAx` may be added and the candidate is at most L3. A `_native` name in a candidate → L4. **Promotion** (`python -m zar_ub promote <program_id>`): re-run S0–S6 from a clean `lean/Candidates/`, `leanchecker` replay of the built module (0.66 s), full battery incl. record profiles, then append to `lean/ZarPrune/Evolved.lean` as `Evolved.e_<sha>` and register in `zar_ub/lemmas.py` and `cache/ledger/accepted_prunes.jsonl` (`{sha, name, lean_name, per_instance_kill_signature, iteration, axioms}`). Failure → quarantine, never promoted.

### 4.5 Direct Lean vs two-stage (NL → Lean)

Both supported by one prompt switch; the $16 probes pick the default (T-P1: verified-prune rate per dollar per model, provisional and re-evaluated at M9 with a larger budget). **A (direct):** one call produces complete proofs. **B (sketch-and-fill):** the call produces `NOTES`, `kill`, and `sound` as a typed skeleton with `have … := by sorry` holes; S7 auto-fills at $0; unfilled holes go to artifacts with goal text so the next iteration (or a `fill` template variation) can close them. The *paid* repair loop (≤ 2 rounds of `deepseek/deepseek-v4-flash` with error text + goal, `$0.01` cap per call) lives in `tools/forge.py`, an offline driver that takes a program id from a checkpoint, repairs, and re-inserts through the normal gate — `evaluate()` stays deterministic and LLM-free.

### 4.6 The cover seam (`lean/ZarPrune/Closure.lean`, harness-owned, staged)

1. **Thinning (P19).** `lt_of_no_exact (P) (h : ∀ A, ¬HasKst P A → weight A ≠ P.w) : ∀ A, ¬HasKst P A → weight A < P.w` (induction on `weight A − P.w`; clearing a one preserves `¬HasKst`). ~70 lines.
2. **Permutations.** `act σ τ A := fun i j => A (σ i) (τ j)`; `weight_act`, `rowSum_act`, `colSum_act`, `hasKst_act_iff` (re-index the injective `s`-tuple increasingly via `Finset.orderEmbOfFin`, as in `hasKst_of_subsets`). ~120 lines.
3. **Sorting.** `exists_sorted : ∀ A, ∃ σ τ, Antitone (rowSum (act σ τ A)) ∧ Antitone (colSum (act σ τ A))` from Mathlib `Tuple.sort`. ~40 lines. `sortedProfileOf A := (sort rowSum, sort colSum)`.
4. **Enumerator.** `genParts (len cap total r budget : Nat) (ub : Nat → Nat) : List (List Nat)` — Tan's Algorithm 1 shape: non-increasing lists, entries ≤ cap, sum = total, prefix pruning by the proved budget `Σ chooseMul x_i r ≤ budget` and by `ub k` for `k < len`. `mem_genParts` completeness by induction on `len` (~200 lines). `genRows P facts`, `genCols P facts` instantiate it with `argA` budgets and `ofPrefixF` bounds from the facts.
5. **`chooseMul`.** `def chooseMul (n k : Nat) : Nat := (List.range k).foldl (fun acc i => acc * (n - i)) 1 / k.factorial` with `chooseMul_eq_choose`; all library kills and schemas are stated with `chooseMul` so kernel reduction uses GMP-accelerated `Nat.mul/div` instead of unfolding `Nat.choose` recursively. This is what makes `survivors_eq … := by decide` plausible; it is measured in T-7 with an explicit go/no-go (kernel ≤ 30 min per closure → Tier-1; else `decide +native` → Tier-1n, axiom named).
6. **Survivors and closure theorem.**
   ```lean
   def survivors (P) (facts) (p : Prune P) : List (List Nat × List Nat) :=
     ((genRows P facts).product (genCols P facts)).filter (fun q => !p.kill (mkProfile P q))
   theorem cover_of_survivors (P facts p) (hF : ∀ f ∈ facts, FactHolds f) :
       ∀ A, Valid P A → weight A = P.w → p.kill (mkProfile P (sortedProfileOf A)) = true ∨ sortedProfileOf A ∈ survivors P facts p
   theorem upper_bound_succ_of_sorted_cover (P) (u) (hu : P.w = u + 1) (p : Prune P) (surv) (hs : survivors P facts p = surv)
       (refuted : ∀ q ∈ surv, ∀ A, sortedProfileOf A = q → ¬ Valid P A) : ∀ A, ¬HasKst P A → weight A ≤ u
   ```
   `cover_of_survivors` is generic (`List.mem_filter` + `mem_genParts` + 1–3), so **no per-instance `decide` over the product is needed**; the only per-instance kernel work is `survivors_eq`.

### 4.7 The refuted seam (`zar_ub/certify.py`, E9)

Per survivor: `encode_case` → DIMACS (sha1) → `tools/cadical -q --binary=false case.cnf case.drat` (exit 20) → `drat-trim case.cnf case.drat -L case.lrat` (`s VERIFIED`) → `lrat-check case.cnf case.lrat` (`c VERIFIED`) → manifest entry `{rows, cols, cnf_sha1, lrat_sha1, lrat_bytes, conflicts, solve_s, check_s}`. **Kill audit** (D1's canary, offline): 5 % of Lean-killed cases per closure (seeded by `table_hash`) are solved to completion; any SAT model is `has_kst`-checked and, if real, halts the pipeline with `PIPELINE_BUG` (a verified kill of a realizable case = definitions/encoding mismatch). Any SAT at `w = z+1` on a survivor likewise halts; a SAT on an *open* target is a lower-bound discovery: witness stored under `cache/witnesses/`, `w` raised for future runs. Tier-0 (M8): `Encode.lean` completeness + Lean-side `LRAT.check` with per-branch named `_native` axioms.

---

## 5. Reward function

### 5.1 Suite (`suite.py`)

* `TRAIN` (ω = 1, pure mode, exact labels): `(9,9)50 (9,10)55 (10,10)61 (10,11)65 (11,11)70 (11,12)75 (12,12)81`; band rule: a cell leaves TRAIN when the population's mean `gain_I` exceeds 0.75 and `(12,13)87`, `(13,13)93`, `(10,14)78` enter.
* `BATTERY`: the same cells at `w = z` (SAT witnesses) + `data/witnesses_33.json` (record profiles `(11,21)=116`: rows `11^6 10^5`, cols `6^11 5^10`; `(12,22)=132`: rows `11^12`, cols `6^22`; Tan's maximal graphs).
* `TARGET` (ω = 2, `--trust tan2022`, censored labels; `ZAR_UB_TARGETS="m,n,w;…"`): default `13,17,117;13,18,122;15,17,133;9,23,104;12,18,109`, later `10,23,113;11,23,124;13,19,123;16,17,134`.
* `GEN` (ω = 1 inside `G_gen`, held-out, other `(s,t)`): `(2,2)` cells `(7,7,22)`, `(8,8,25)` and one `(4,4)` cell `(9,9,w)` at `w = z+1` from Tan's tables; built with `python -m zar_ub table 7 7 2 2 22 --pure`; never shown in the prompt.

Scoring is marginal over the proved baseline `B` (`counting` ∪ conditional neighbour prunes with ledgered facts ∪ `Evolved.lean`): `S_I = {q ∈ C_I : ¬B.kill(q)}`, with `baseline_lean_mask` stored in each table by `python -m zar_ub table … --baseline`.

### 5.2 Quantities

For instance `I`: `d_I(q) ≥ 1` (§6), `W_I = Σ_{q∈S_I} d_I(q)`, `H_I` = top decile of `S_I` by `d_I`, `K^L_I` Lean mask (defined iff L5), `K^P_I` Python mask, `K^S_I` mask of `schemaK` alone.

```
gain_I(K)  = Σ_{q∈S_I, K(q)} d_I(q) / W_I
tail_I(K)  = Σ_{q∈H_I, K(q)} d_I(q) / Σ_{q∈H_I} d_I(q)
G_train = mean_{I∈TRAIN} gain_I(K^L)      G_target = mean_{I∈TARGET} gain_I(K^L)  (:= G_train if TARGET = ∅)
G_gen   = mean_{I∈GEN}   gain_I(K^L)      (:= G_train if GEN = ∅)
Tail    = mean_{I∈TRAIN∪TARGET} tail_I(K^L)
E       = mean_{I∈TRAIN} gain_I(K^P)      (empirical; only if battery-sound; used ONLY in the unverified branch)
```

### 5.3 The formula (exact)

```
hard_zero = battery violated by K^P on any witnessed case
          ∨ ladder ∈ {L0, L4} ∨ python error/timeout ∨ schema_error
          ∨ PIPELINE_BUG  (K^L kills a witnessed case: score 0 AND the run halts via a sentinel file cache/PIPELINE_BUG)

combined_score =
    0                                                                         if hard_zero
    0.20 + 0.80·(0.40·G_train + 0.30·G_target + 0.10·G_gen + 0.20·Tail)       if ladder = L5
    min(0.19, 0.03·ladder/3 + 0.06·lean_partial + 0.08·E + 0.02·tail_train(K^P))   if ladder ∈ {L1, L2, L3}
```

Properties: verified-kills-nothing = 0.20 exactly (the initial program); every unverified score ≤ 0.19; no mirror penalty (agreement is a metric and artifact only); stationary (no running-record term); the tail term targets the decile that carries 61.6 % of the work [E10]; `G_gen` rewards prunes that are not `(3,3)`-specific without punishing legitimate residue arguments.

`lean_partial` (exact, from `GateResult`): with `D` declarations, `D_ok` error-free, `first` = first error line, `N` lines, `H` holes, `H_filled` filled:
`depth = (first−1)/N` (1.0 if no error); `declfrac = D_ok/D` (0 if `D = 0`); `fill = H_filled/H` (1 if `H = 0`);
`L1: 0.05 + 0.10·declfrac + 0.05·depth`; `L2: 0.25 + 0.15·declfrac + 0.10·depth`; `L3: 0.50 + 0.30·fill + 0.10·depth`; `L0/L4: 0`; `L5: 1`.

### 5.4 Metrics dict (exact keys; numeric unless stated; OpenEvolve ignores non-numeric values when averaging)

```
combined_score, sound_battery (1/0), pipeline_bug (1/0), lean_ladder (0–5), lean_partial, lean_ok (fraction of instances
whose gateInstK elaborated), proven_gain (=G_train), target_gain, gen_gain, tail_gain (=Tail), empirical_gain (=E),
schema_gain (mean_I gain_I(K^S)), agreement (fraction of scored cases with K^L = K^P), hard_killed (count of censored
target survivors killed by K^L), censored_share (fraction of target_gain coming from censored labels), survivors_left
(Σ_I |S_I ∖ K^L|), kill_novelty (1 − Jaccard(K^L, U_accepted); 1.0 when the ledger is empty), n_new_prunes (Prune/CondPrune
defs in the candidate beyond the baseline), n_holes, n_holes_filled, eval_seconds, gate_seconds, gate_cache_hit (1/0),
table_hash (string: sha1 over the suite's table files), suite_version (string)
```

Artifacts: `per_instance` (hardest alive survivors with profiles and `d`), `lean_errors` (first 6 with line/message/goal), `lean_holes` (unfilled `have` statements with goals), `lean_source_filled`, `UNSOUND_counterexamples`, `python_lean_disagreements`, `schema_error`, `library_ledger`, `duplicates` (Jaccard ≥ 0.95 with an accepted prune), `kill_signature` (sha1 of concatenated `K^L`).

### 5.5 Cascade (`config.yaml`: `cascade_evaluation: true`, `cascade_thresholds: [0.005, 0.15]`)

`evaluate_stage1` (≈ 1 s): copy `cache/` tables to a temp dir; `run_candidate.py` in a subprocess (90 s, no network) → Python masks, battery, scan, `SCHEMA_DATA` validation; returns `0` if hard-zero else `0.01 + 0.08·E`. `evaluate_stage2` (≈ 2–4 s): Lean gate on `TRAIN ∪ GEN` in one process; score without the target term. `evaluate_stage3` (≈ 1–5 s): masks on `TARGET` tables; full score. Stage 3 is skipped when the stage-2 score < 0.15, i.e. only unverified programs skip it.

### 5.6 MAP-Elites

`feature_dimensions: ["proven_gain", "lean_ladder"]`, `feature_bins: {"proven_gain": 10, "lean_ladder": 6}` (60 cells for `population_size: 60`, `num_islands: 3`, `archive_size: 20`, `migration_interval: 20`, `exploration_ratio: 0.3`, `exploitation_ratio: 0.6`, `elite_selection_ratio: 0.2`). `kill_novelty` becomes a third axis (5 bins) via `config_novelty.yaml` once `accepted_prunes.jsonl` has ≥ 5 entries (checked by `zar_ub/promote.py`, which prints the switch). `n_lean_decls` is not a feature (rewards length).

### 5.7 Anti-cheating inventory

| attack | defence |
|---|---|
| Python `kill` lies | only `K^L` earns credit or prunes; `K^P` feeds stage 1 and the ≤ 0.19 band |
| unsound Lean kill | impossible without an axiom; `PIPELINE_BUG` sentinel halts the run if it ever happens (definitions mismatch) |
| smuggled axioms / `native_decide` / `implemented_by` / `csimp` / `opaque` | S1 scan + S5 audit + `leanchecker` at promotion |
| forged mask text | every `#` command and `IO` forbidden; nonce markers; exact length check |
| redefining `Valid`/`HasKst`/wrapper names | declared-name rule; wrapper outside `Cand` with full names; `autoImplicit false`; `end Cand`/`namespace` forbidden |
| vacuous prune | 0.20, the honest floor |
| re-listing library prunes | marginal over `B` |
| slow kill | `partial` forbidden; `#eval` timeout → L1 "kill too slow" |
| difficulty gaming | labels precomputed from the CNF only, read-only, `table_hash` in every metric dict |
| censored target credit | clipped `[cap, 20·cap]`; `censored_share` reported; daemon deepens labels |
| tampering with tables/certs | candidate runs on a copy; tables loaded before execution; masks computed by the harness |
| duplicates / length farming | `kill_novelty`, `duplicates` artifact; no score term depends on length |
| schema data farming | `schema_gain` reported separately; schemas are library-proved so no soundness exposure |

---

## 6. Branch difficulty measure

### 6.1 Definition

`d(q)` = CaDiCaL 1.9.5 (pysat `cadical195`, default options, seed 0) conflicts to refute `encode_case(P, q)`; exact when refuted, censored-and-calibrated otherwise. Conflicts, not seconds: reproducible across machines for a fixed build; conflicts and propagations are statistically indistinguishable workload measures [LR §5.2]; on 1,571 labelled cases `c2000` has Spearman ρ = 0.913 with the true count [E10].

### 6.2 Estimator (`zar_ub/difficulty.py::label_case`, exact)

```python
SCHEDULE = [(2_000, 5.0), (20_000, 30.0), (200_000, 120.0), (2_000_000, 600.0)]   # (conf_cap, time_limit s)

def label_case(inst, rows, cols, mode):            # mode ∈ {"exact", "censored"}
    cnf = encode_case(inst, rows, cols)
    feats = dict(nvars=cnf.nvars, nclauses=len(cnf.clauses), log2_volume=log2_volume(inst, rows),
                 prop_frac=propagation_fraction(cnf, inst))
    c2000 = None
    last_cap = None
    for cap, tl in SCHEDULE:
        if mode == "censored" and cap > 20_000: break
        r = solve_cnf(cnf, inst, solver="cadical195", conf_budget=cap, time_limit=tl)     # fresh solver each cap
        c2000 = c2000 if c2000 is not None else r.conflicts
        last_cap = cap
        if r.status == "sat":   return Label(status="sat", d=None, witness=has_kst_checked(r.matrix), c2000=c2000, **feats)
        if r.status == "unsat": return Label(status="unsat", d=max(1, r.conflicts), c2000=c2000, censored=False, **feats)
    fhat = exp(a[st] + b[st] * log(max(c2000, 1)) + g[st] * feats["log2_volume"])   # fitted per (s,t) on TRAIN
    d = min(max(last_cap, fhat), 20 * last_cap)
    return Label(status="unknown", d=d, c2000=c2000, censored=True, cap=last_cap, **feats)
```

`(a, b, g)` are fitted by least squares in log space on the TRAIN cases that were unknown at 2,000 but solved exactly (`python -m zar_ub calibrate`), refitted whenever a ladder cell is added, validated on a held-out cell (`(12,13,87)`; acceptance ρ ≥ 0.85, log-RMSE ≈ 0.5). TRAIN/BATTERY tables use `mode="exact"`; TARGET tables use `mode="censored"` for a first pass (≈ 0.35 s/case) so evolution can start, and the closure daemon deepens them in place (`python -m zar_ub deepen M N S T W --cap 200000`) as background work; `table_hash` changes when it does, so scores are only compared within a table version.

### 6.3 Large tables (common random numbers)

`3,000 < |S_I| ≤ 50,000`: uniform sample `Σ_I`, `N = 500`, seed = `table_hash`; `50,000 < |S_I|`: stratified sample, strata = deciles of `log2_volume` × number of distinct column sums, `N = 2,000`. `W_I ≈ |S_I|·mean_Σ d`; `gain_I` computed on the sample; masks still evaluated on all cases for closure decisions; sample composition stored in the table.

### 6.4 Reported alongside

`tail_share` (top-decile share of remaining work), critical path `max_q d(q)` among survivors, LRAT bytes for certified cases (sanity check of `f̂`, never in the reward).

### 6.5 Cost

Features ≤ 0.05 s; 2k probe ≈ 0.05 s; 20k ≈ 0.35 s; 200k ≈ 3.5 s; 2M ≈ 35 s. Ladder relabelling ≈ 3 min [E10]. Target first pass: `(9,23)` 1 min, `(12,18)` 12 min, `(10,23)` 10 min; 200k pass 1–3 h per cell (background).

---

## 7. SAT execution policy

1. **Never inside `evaluate()`.**
2. **Table build** (`python -m zar_ub table M N S T W [--pure|--trust tan2022] --cap 20000 --baseline`), offline, parallel (`multiprocessing.Pool`).
3. **Closure daemon** (`python -m zar_ub closure-daemon --checkpoint-dir <run>/checkpoints --targets … --poll 50 --budget 5e8`): every 50 iterations reads the best L5 program, promotes it if new (§4.4), recomputes survivors under the promoted library, estimates `Ŵ = Σ_{alive} d̂`, and launches certification when `Ŵ ≤ budget` or survivors < 200 or on `--now`. Each launch is logged in `experiments/LOG.md` with the program id.
4. **Per-cell escalation**: round-robin over all survivors `2k → 20k → 200k → 2M → 20M` (pysat `cadical195`, then `glucose4` at the same cap for diversity) so the hard tail is identified before any case eats hours; a case that reaches UNSAT is then run with proof-producing `tools/cadical` (600 s → 3,600 s → 6 h) → DRAT → LRAT → `lrat-check`. A case still open after 20M conflicts / 6 h is **cubed inside its certificate**: split on the support of the heaviest row (`C(n, r_1)` cubes as unit assumptions over the same CNF); each cube's LRAT plus the cover clause set `⋀ ¬cube_i` is itself certified UNSAT (LRAT tautology certificate, D1) and the concatenation is one LRAT for the case, with credit accounting parent-capped. No new trusted component.
5. **Concurrency**: `min(cores − 1, 8)` solver processes; `parallel_evaluations: 3` for the loop (each Lean process ≈ 1 GB).
6. **Certificates**: DRAT deleted after LRAT; LRAT kept under `cache/certs/<tag>/` with `manifest.json`; `.gitignore`d, manifest committed. Every stored LRAT is re-verified by T-11 with a fresh `lrat-check` build.
7. **Closure report** (`zar_ub/closure.py`): per case `PRUNED (Lean: <name>, axioms)` | `REFUTED (lrat sha1, bytes)` | `OPEN`; the tier (2 / 1 / 1n / 0); the trusted base table; facts used with tags; the accepted Lean source; audit results. A bound is claimed only when the tier is ≤ 1n, no case is OPEN, and every fact is `tan2022` or better.

---

## 8. Generalization across (m, n, s, t)

### 8.1 Parametric prunes

`candidate (P : Params) : Prune P` is elaborated once and instantiated on every suite instance; `Prune.mono` (new, 4 lines in `Prune.lean`: a `Prune ⟨m,n,s,t,w⟩` is a `Prune ⟨m,n,s,t,w'⟩` for `w' ≥ w`) lets a prune proved at the smallest weight of a `bound` search serve every larger weight. Instance-specific candidates are accepted, scored only where they type-check, and visible through `lean_ok < 1`.

### 8.2 Conditional prunes and provenance (`lean/ZarPrune/Cond.lean`, new)

```lean
structure Fact where (m n s t z : Nat) (tag : String)
def FactHolds (f : Fact) : Prop := ∀ B : Mat f.m f.n, ¬ HasKst ⟨f.m, f.n, f.s, f.t, 0⟩ B → weight B ≤ f.z
structure CondPrune (P : Params) (facts : List Fact) where
  name : String := ""
  kill : Profile P.m P.n → Bool
  sound : (∀ f ∈ facts, FactHolds f) → ∀ A, kill (profileOf A) = true → ¬ Valid P A
def CondPrune.discharge (q : CondPrune P facts) (h : ∀ f ∈ facts, FactHolds f) : Prune P
def argDelColF (P) (f : Fact) (hf : f.m = P.m ∧ f.n + 1 = P.n ∧ …) : CondPrune P [f]     -- wraps argDelCol
def argDelRowF … ; def Prune.ofPrefixF (P) (k) (f : Fact) … : CondPrune P [f]            -- Argument I, k < P.n
```

`data/ledger.csv` rows: `(m, n, s, t, bound, kind ∈ {exact, ub}, provenance ∈ {lean-here, tan2022, collins16, bhan26}, closure_file, hypotheses)`; `data/claims_2026.csv` holds dfield/Hou/Afrasyab values and is read **only** to choose targets. `zar_ub/ledger.py` exposes `facts_for(P, trust)` (the exact `Fact` list the generator and conditional prunes use, so the closure theorem's hypotheses are the facts that were load-bearing) and runs the mechanical claim checker on every write (`z(m,n) ≤ z(m,n+1)`, `z(m,n+1) ≤ z(m,n) + m`, witness comparison, transposition symmetry). A closure whose facts are all `lean-here` is discharged by term application of the neighbours' theorems (the DAG in topological order) and becomes unconditional.

### 8.3 Suite roles and transfer

`TRAIN`/`BATTERY`/`TARGET`/`GEN` as in §5.1. Post-run zero-shot transfer matrix (`experiments/transfer.py`): the accepted library on every cached table it was not trained on — which prunes fire where, gains per cell; reported, not rewarded. Curriculum: TRAIN moves up as cells saturate; TARGET moves from replay tier to open tier as survivor counts drop.

### 8.4 Where the room is

After `counting` + DGH(4) the remaining kills come from cross-side arguments, residue/overlap arguments (P14–P16; `Prune.ofResidue`) and Farkas combinations (P11; `Prune.ofFarkas`); the schema channel reaches (b) and (c) without new proofs, and the DGH-attackable cells in TARGET give the loop early verified credit for (a)-type arguments.

---

## 9. Cost and time model, model choice

### 9.1 Per evaluation (no LLM)

| component | measured |
|---|---|
| stage 1 (copy tables, Python masks, battery, schema validation) | 0.3–1.5 s |
| Lean gate, whole suite, Mathlib-backed, incl. masks | 1.5–4.4 s [E5/E6, 2026-09-21]; 0.3–0.5 s Mathlib-free profile [E4]; 0 s on cache hit |
| stage 3 target masks | + 0.2 s per 1,000 cases |
| total | 2–6 s; 3 in parallel |

### 9.2 SAT (offline)

Ladder relabel 3 min; target first pass 1–12 min per cell; 200k pass 1–3 h per cell; `(9,9,50)` closure 1.8 s / 36 certificates [E9]; `(9,23,104)` and `(12,18,109)` closures: unknown, hours to days without new prunes (a single `(12,18)` residual did not finish in 16.5 min [LR §2.3]) — that reduction is the thesis measurement.

### 9.3 LLM tokens and dollars (OpenRouter prices recorded 2026-09-21; recheck before each run)

Prompt ≈ 8–12k input (system 1.5k, program 2–4k, artifacts ≤ 16 KB ≈ 4k, top/diverse programs 2×2k), output 3–8k.

| model | $/M in/out | $/iteration (10k/5k) |
|---|---|---|
| deepseek/deepseek-v4-flash | 0.06/0.11 | 0.0012 |
| openai/gpt-oss-120b | 0.15/0.60 | 0.0045 |
| openai/gpt-5.4-mini | 0.75/4.5 | 0.03 |
| google/gemini-3.1-pro | 2/12 | 0.08 |
| anthropic/claude-opus-4.7 | 5/25 | 0.18 |

Ensemble 0.5 flash / 0.3 mini / 0.2 Gemini Pro ≈ $0.03–0.05 per iteration → 1,000 iterations ≈ $30–50, 4–6 h on 3 workers (LLM-bound). The $16 is spent on probes only (§10.2): ≤ $13, hard stop; `experiments/cost.py` logs the key's usage to `experiments/cost_ledger.md` before and after every paid step; smoke scripts refuse to start below $2 remaining.

### 9.4 Model choice

Architect/sketcher: Gemini 3.1 Pro (`reasoning_effort: medium`, `max_tokens 16000`, `reasoning_max_tokens` reserving ≥ 8k for the answer; the config comment documents the starvation failure) or Claude Opus 4.7 at low weight. Filler/smoke/repair: DeepSeek-V4-Flash, GPT-5.4-mini second if Flash's Lean 4.34 syntax error rate exceeds 50 % in P-1. No specialised prover is reachable via API or trained on Lean 4.34; not planned around. `temperature 0.6`, `top_p 0.95`, `retries 2`, `timeout 600`. Mathlib-free gate profile (`ZAR_UB_GATE_PROFILE=core`) is a config switch for machines without the 7.9 GB cache.

---

## 10. Test plan

### 10.1 NO-LLM mode ($0) — exact commands

All commands run from `/Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/upper_bounds/` with `ZAR_UB_NO_LLM=1` (disables any OpenRouter call; `evaluate()` is LLM-free anyway).

| id | command | expected |
|---|---|---|
| T-1 | `python -m unittest discover tests` | engine tests pass (partitions/waterfill vs brute force, encoding ⇔ matrix existence with/without lex, Argument D vs Lean `#eval`, scan unit test per forbidden token, nonce-forgery test, NFKC look-alike test) |
| T-2 | `python -m zar_ub gate 10 11 3 3 65 initial_program.py --pure` and `python -c "import evaluator; print(evaluator.evaluate('initial_program.py')['metrics'])"` | mask length 195 = cases; `lean_ok == 1.0`, `lean_ladder == 5`, `combined_score == 0.20` (a run aborts if the baseline scores otherwise) |
| T-3 | `python -m unittest tests.test_evaluator` (golden bank `tests/candidates/*.py`, expectations in `tests/golden.json`) | see table below |
| T-4 | `python -m zar_ub calibrate --holdout 12,13,87` | ρ ≥ 0.85, log-RMSE reported |
| T-5 | `python experiments/no_llm/run_synthetic.py --iterations 30 --config config.yaml` (`SyntheticMutator` through `init_client`; scripted edits from `tests/snippets/{dgh4,deletion,residue_sketch,broken_proof,schema_farkas}.lean`) | no unsound program with score > 0 in the database; L5 programs dominate the archive; L2/L3 cells occupied; `openevolve-run.py … --checkpoint` resumes |
| T-6 | `python tools/stub_llm.py --port 8123 --bank tests/snippets &` then `OPENAI_API_KEY=x python ../../../openevolve-run.py initial_program.py evaluator.py --config config_stub.yaml --iterations 50` (`api_base: http://127.0.0.1:8123/v1`) | 50 iterations complete over the real OpenAI client path; best score non-decreasing across checkpoints; ≥ 6 feature cells occupied |
| T-7 | `python -m zar_ub closure 9 9 3 3 50 --pure --lean lean/ZarPrune/Evolved.lean --tier 1` then `lake build ZarPrune.Closures.Z_9_9_50 && leanchecker …` | `z(9,9) ≤ 49` as a Lean theorem; kernel time of `survivors_eq` logged; **go/no-go** for Tier-1 vs Tier-1n recorded in LOG |
| T-8 | `python -m zar_ub closure 10 21 3 3 107 --trust tan2022 --tier 1` (and `11 19 … 107`, `11 20 … 112`) | zero survivors; theorems with `FactHolds` hypotheses tagged `tan2022`; ledger rows written |
| T-9 | `python -m zar_ub gate 13 17 3 3 117 tests/candidates/lean_dgh4.py --trust tan2022` | L5; kills reproduce the DGH Python census; `(13,17) ≤ 116`, `(13,18) ≤ 121` close with zero SAT |
| T-10 | `python experiments/transfer.py --library lean/ZarPrune/Evolved.lean` and `python -m zar_ub table 12 18 3 3 109 --cols-only --dry-run` | transfer matrix; survivor counts for the `cols`-only ablation |
| T-11 | `python -m zar_ub verify-certs cache/certs --fresh-lratcheck` and a one-byte corruption test | all verify; corrupted certificate rejected |
| T-12 | `python -m zar_ub audit-kills 9 9 3 3 50 --frac 0.05` | no SAT among sampled Lean-killed cases |

Golden bank (`tests/candidates/`, expectations in `tests/golden.json`):

| candidate | ladder | score |
|---|---|---|
| `initial` | L5 | 0.20 |
| `python_only_dgh4` (DGH(4) in Python only) | L5 | 0.20, `agreement < 1`, `empirical_gain > 0` |
| `lean_dgh4` (hand-proved, M5 artefact) | L5 | > 0.20 with DGH cells in TARGET |
| `schema_farkas` (baseline Lean + DGH duals in `SCHEMA_DATA`) | L5 | > 0.20 on `(15,17,133)`; `schema_gain > 0` |
| `unsound_row7` ("kill any case with a row ≥ 7") | — | 0 (battery) |
| `sorry_in_kill` / `sorry_in_sound` | L0 / L2 | 0 / ≤ 0.19 |
| `sketch_two_holes` (one `omega`-fillable) | L3 | ≤ 0.19; `n_holes_filled = 1` |
| `native_decide`, `axiom_smuggle` (in a block comment), `implemented_by`, `opaque`, `unicode_lookalike` | L0 | 0 |
| `mask_spoof` (`#eval IO.println "MASK …"`) | L0 | 0 (forbidden `#`/`IO`) |
| `redefine_valid` | L0 | 0 (declared-name rule) |
| `slow_kill` (enumerates all subsets) | L1 | ≤ 0.19, "kill too slow" |
| `instance_specific` | L5 on its instance | `lean_ok = 1/|𝓘|` |
| `cond_deletion` (`CondPrune` using a `Fact`) | L5 | kills the `(10,21)`-type cells when the ledger has Tan's neighbours |
| `notDescending` (symmetry break as a prune) | L1 (sound fails) | ≤ 0.19 |
| `injected_bad_lean_mask` (test hook, not a candidate file) | — | `PIPELINE_BUG` sentinel written; run halts |

### 10.2 With the $16 (every step logged before/after by `experiments/cost.py`)

| id | what | calls | est. |
|---|---|---|---|
| P-0 | `python experiments/E7_llm_smoke/dump_prompt.py` — exact prompt, token count | 0 | $0 |
| P-1 | `probe_one.py`: 3 models (flash, mini, Gemini Pro) × {direct A, sketch B} × 2 parents (initial; initial + hand DGH(4)) | 12 | ≈ $1.5 |
| P-2 | `tools/forge.py` repair on the P-1 sketches that reached L2/L3 (≤ 2 flash rounds each) | ≤ 20 | ≈ $0.1 |
| P-3 | smoke: 20 iterations flash-only, `config.yaml`, TARGET = DGH cells | 20 | ≈ $0.05 |
| P-4 | smoke: 15 iterations, 0.6 flash / 0.4 mini, winning prompt of P-1 | 15 | ≈ $0.5 |
| P-5 | one Gemini Pro sketch per hardest-survivor artifact of `(9,23,104)` and `(12,18,109)` | 4 | ≈ $0.5 |
| P-6 | 25-iteration ensemble run with `(13,17,117)` in TARGET (DGH(4) closes it with zero SAT if found) | 25 | ≈ $4 |
| reserve | never below $2 before a step; hard stop at $13 spent | | ≥ $3 |

Success criteria (for the probes, not for bounds): ≥ 1 applicable diff with a compiling `kill` (L2+) per model in ≤ 3 attempts; ≥ 1 L5 prune with `G_train > 0` or `schema_gain > 0` from any model; no forbidden-token candidate scores > 0; per-iteration cost within 2× of §9.3; a *provisional* A-vs-B decision by verified-prune rate per dollar (re-evaluated in M9 — 12 probe calls cannot distinguish prompt variants robustly, and the design says so).

---

## 11. Risks and failure modes

| risk | likelihood / impact | mitigation |
|---|---|---|
| LLM verified-prune rate ≈ 0 on Lean 4.34 | high / high for the search, none for soundness | schema channel (data, not proofs); auto-fill; DGH-attackable target cells; lemma pool and worked example in the prompt; hand-proved DGH(4) at M5 so the pipeline's value is demonstrated regardless |
| kernel `survivors_eq` too slow | medium / medium | `chooseMul`; measured at T-7 before anything depends on it; Tier-1n fallback with named axiom, printed |
| `Closure.lean` (~700 lines) and `Schemas.lean` residue identities (~450 lines) are hand-proof effort | high / medium | evaluator v2 ships first (Tier-2 reports); `Closure.lean` staged (thinning → act → sort → enumerator); Farkas/prefix schemas before residues |
| gate fragility (Mathlib cache missing `Mathlib.olean` in E5; option drift on import) | medium / high | T-2 baseline check aborts runs; targeted imports pinned; Mathlib-free profile switch |
| 0.20 plateau | medium / medium | schema channel; MAP-Elites keeps sketches alive; artifacts name the hardest alive profiles with numbers; DGH cells |
| unreviewed 2026 facts contaminating claims | low with the ledger / catastrophic without | `claims_2026.csv` is targets-only; provenance filter default `tan2022`; facts are theorem hypotheses; claim rule |
| encoding or enumeration bug | low / catastrophic | battery at `w = z`; `has_kst` re-check of every SAT model; 5 % kill audit; `PIPELINE_BUG` halt; `mem_genParts`; Tier-0 completeness theorem |
| heavy-tailed noise across tables | high / medium | conflicts not seconds; `table_hash`; CRN samples; tail term; per-instance reporting |
| case explosion at `m ≥ 13` pure mode | certain / medium | table mode with ledgered facts; sampling; pure mode for TRAIN only |
| OpenEvolve mechanics (long-string diffs, per-model prompts) | medium / low | `diff_based_evolution: true`, `max_code_length 60000`, `allow_full_rewrites` for stuck islands; `template_variations`; P-1 measures diff-application rate |
| cost overrun | low / high | ledger before/after; hard stop; forge outside the loop |
| memory (Lean + Mathlib oleans per gate) | medium / low | `parallel_evaluations ≤ 3`; `ulimit -v` |
| closure never triggers on a target | medium / low | the reduction (survivors and `Ŵ` before/after) is itself the thesis measurement; certificate-internal cubing is the lever |
| Tier-0 (`Encode.lean`, `exists_doubleLex`) exceeds the timeline | high / low | Tier-1 is the committed deliverable |

---

## 12. Milestones (each with an acceptance test from §10)

| # | milestone | deliverable | done when |
|---|---|---|---|
| M0 (done) | engine, gate, tables, certificates, difficulty labels, `Counting.lean`, MASKLINE fix | E1–E10 | — |
| M1 | evaluator v2 + gate hardening | `zar_ub/lean_gate.py` (nonce, ladder, NFKC, declared-name rule, cache, schema footer, auto-fill), `zar_ub/reward.py`, `evaluator.py` (3 stages, metrics §5.4), `suite.py` (TARGET/GEN, band rule), `zar_ub/difficulty.py` (§6.2), `initial_program.py`, `config.yaml`, `tests/candidates/` + `tests/golden.json`, `experiments/no_llm/`, `tools/stub_llm.py`, `Prune.mono` | T-1, T-2, T-3, T-4, T-5, T-6 green; baseline scores exactly 0.20 |
| M2 | `Cond.lean` + ledger | `Fact`/`CondPrune`/`discharge`, `argDelColF`/`argDelRowF`/`ofPrefixF`, `zar_ub/ledger.py`, `data/ledger.csv`, `data/claims_2026.csv`, claim checker | T-8 survivors = 0 (Python side); `cond_deletion` golden passes |
| M3 | `Closure.lean` Tier-1 | thinning, `act`, sorting, `genParts`/`mem_genParts`, `chooseMul`, `survivors`, `upper_bound_succ_of_sorted_cover`; `zar_ub/closure.py` emits `Closures/*.lean` | T-7 green with kernel timing and tier decision; T-8 theorems `z(10,21) ≤ 106`, `z(11,19) ≤ 106`, `z(11,20) ≤ 111` |
| M4 | promotion + closure daemon | `zar_ub/promote.py`, `Evolved.lean`, `accepted_prunes.jsonl`, `zar_ub/closure_daemon.py`, kill audit, `verify-certs` | T-11, T-12 green; daemon closes `(9,9,50)` from a checkpoint |
| M5 | schemas + hand prunes | `Schemas.lean` (`ofFarkas`, `ofPrefix`; `ofResidue` if time), `zar_ub/lemmas.py`, DGH(4) proved by hand (`tests/candidates/lean_dgh4.py`) | T-9 green; `schema_farkas` golden refutes `(15,17,133)`; T-10 transfer matrix |
| M6 | $16 probes | P-0…P-6, LOG E11–E16, cost ledger, provisional A/B decision | criteria of §10.2 |
| M7 | first SAT targets | `(9,23,104)` (244 cases), `(10,22,111)` (3 cases), `(12,17,104)` closure attempts with the hand library + promoted prunes | closure reports; open cases enumerated; comparison with dfield's counts |
| M8 | Tier-0 seam | `Encode.lean`, `exists_doubleLex`, Lean-side `LRAT.check` on `(9,9,50)`; axiom ledger | `z(9,9) ≤ 49` with only allowed axioms + named native axioms |
| M9 | real runs (cloud / larger budget) | 2–3 runs ≥ 1,000 iterations; ablations (no-schemas, no-sketch, no-tail, novelty axis, `cols`-only, and a policy-loop ablation over the accepted library as D1 proposed) | verified prunes with `G_target > 0`; survivor/work reduction on `(12,18,109)` and `(9,23,104)` |
| M10 | v2 levers | verified cuts registered with Lean statements; enriched case index | only after M9 measurements justify them |
| M11 | thesis | closure theorems for whichever of `(9,23)`, `(12,17)`, `(12,18)`, `(10,23)` closes; calibration chapter; ablations; axiom/provenance ledgers | reproducible from `experiments/LOG.md` |

---

## Appendix A — File layout (create or change, under `/Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/upper_bounds/`)

```
initial_program.py                 §2.2 skeleton (baseline = counting; NOTES, LEAN_SOURCE, SCHEMA_DATA, kill)
evaluator.py                       three cascade stages, §5 metrics/artifacts, NO-LLM by construction
run_candidate.py                   + returns schema_data and notes; runs on a copy of cache/
suite.py                           TRAIN/BATTERY/TARGET/GEN, ω weights, band rule, witnesses_33.json
config.yaml                        prompt §2.5, features ["proven_gain","lean_ladder"], cascade [0.005,0.15], ensemble §9.4
config_stub.yaml                   api_base → tools/stub_llm.py (T-6)
config_novelty.yaml                third MAP-Elites axis once ≥ 5 accepted prunes
zar_ub/lean_gate.py                §4.2 procedure: NFKC, nonce, ladder, cache, schema footer, auto-fill
zar_ub/reward.py          (new)    §5.2–5.4 formulas, lean_partial, novelty
zar_ub/lemmas.py          (new)    declarative schema/lemma registry (§2.3)
zar_ub/ledger.py          (new)    Fact provenance, tiers, claim checker, facts_for(P, trust)
zar_ub/difficulty.py               label_case (§6.2), calibrate, deepen, CRN samples
zar_ub/casetable.py                baseline_lean_mask, d, censored flags, sample, table_hash, kinds
zar_ub/closure.py                  Closures/Z_m_n_w.lean emission, report with tier + trusted base
zar_ub/closure_daemon.py  (new)    §7 item 3
zar_ub/promote.py         (new)    §4.4 promotion path
zar_ub/cli.py                      subcommands: table, gate, certify, closure, closure-daemon, promote, calibrate, deepen,
                                   audit-kills, verify-certs, api, bound, show
experiments/no_llm/replay_llm.py   ReplayLLM + SyntheticMutator via LLMModelConfig.init_client
experiments/no_llm/run_synthetic.py  T-5 driver
experiments/transfer.py   (new)    zero-shot transfer matrix
experiments/cost.py                OpenRouter usage ledger (exists)
tools/stub_llm.py         (new)    OpenAI-compatible HTTP mutation server (T-6)
tools/forge.py            (new)    offline paid repair loop (never inside evaluate())
tests/test_evaluator.py   (new)    golden bank runner; tests/golden.json; tests/candidates/*.py; tests/snippets/*.lean
lean/ZarPrune/Prune.lean           + Prune.mono
lean/ZarPrune/Cond.lean   (new)    §8.2
lean/ZarPrune/Closure.lean (new)   §4.6
lean/ZarPrune/Schemas.lean (new)   §2.3 schemas
lean/ZarPrune/Evolved.lean (new)   promoted prunes (leanchecker-replayed)
lean/ZarPrune/Closures/*.lean      generated closure theorems (harness domain)
lean/ZarPrune/Encode.lean (M8)     Tier-0
data/ledger.csv, data/claims_2026.csv, data/witnesses_33.json
cache/ledger/accepted_prunes.jsonl, cache/certs/<tag>/manifest.json, cache/witnesses/, cache/PIPELINE_BUG (sentinel)
docs/design.md (this file), docs/design_candidates/{0_prune_library_genome,1_decomposition_policy_genome,2_certificate_first,judges}.md
```

## Appendix B — Sources

`docs/literature_review.md` §§1–10; `experiments/LOG.md` E1–E10; `lean/ZarPrune/{Sum,Basic,Prune,Prunes,Counting,Demo}.lean`, `lean/COUNTING_SPEC.md`; `zar_ub/*`; `openevolve/config.py` (`init_client`, `template_variations`, `cascade_thresholds`), `openevolve/utils/metrics_utils.py` (non-numeric metrics ignored), `openevolve/evaluator.py`; `docs/design_candidates/` (the three candidates and both judge reports).

# Judge reports on the three candidate designs

Two independent judges scored the candidates in `docs/design_candidates/` (0 = prune-library genome, 1 = decomposition-policy genome, 2 = certificate-first). Both ranked them **[2, 0, 1]**. The synthesis in `docs/design.md` takes Design 2 as the spine, grafts every `keep` item below, and fixes every listed weakness.

---

## Judge 0 — soundness-and-verification lens

**Ranking:** [2, 0, 1]

**Verdict.** Under a soundness-and-verification lens the decisive question is what object carries the claim. Design 2 makes the claim a single Lean theorem with per-fact hypotheses and a kernel-decided cover, enumerates its trusted base per tier, keeps external checkers as named hypotheses, and bounds partial credit at 0.20 with L4 = 0; none of the three can be made to emit a false bound through the LLM channel, but Design 2 is the only one whose v1 claim path does not pass through an unverified Python cover or a prose closure report. Design 0 has the most careful LLM-facing gate (nonce markers, # ban, promotion-time leanchecker, PIPELINE_BUG, adversarial golden bank) and should be mined for those mechanisms, but in v1 it claims bounds via a report whose cover and encoding are trusted Python, uses a coarse BoundTable.Sound hypothesis, and keeps a small unverified empirical reward band. Design 1 has the best level-0/1 cover story (survivors defined in Lean) and the best provenance/tier model, but its genome carries Python callables the harness executes without a stated sandbox boundary, its reward mixes CPU-time ratios, provisional pysat UNSAT and selectable unreviewed facts, and cuts/sub-cube kinds enlarge the trusted encoding; it is the least aligned with the proposal's LLM-writes-Lean question. Recommended synthesis: build Design 2's certificate-first spine (Closure.lean, per-fact hypotheses, schemas, bounded ladder), graft Design 0's gate hardening and $0 adversarial test bank, and take from Design 1 the Lean-defined survivors enumerator, the Fact/tier provenance model, the LRAT cover certificate for any future sub-cubing, and the 5% kill audit canary.

### Design 0 — score 7

Strengths:
- Most thorough LLM-facing gate of the three: extended forbidden-token list (incl. +native, trustCompiler, run_cmd, attribute edits, noncomputable, all # commands), declared-name blacklist for Valid/HasKst/Params/..., autoImplicit false, wrapper declarations outside the Cand namespace, and leanchecker replay at promotion time (S5) before anything enters Evolved.lean.
- Nonce-tagged MASK markers regenerated per evaluation plus a ban on every # command: closes the forged-mask channel with two independent layers, and the mask length check catches truncation.
- Kill mask that prunes is always the Lean-computed one; the Python mirror only feeds the witness battery and a bounded (<=0.06) empirical band; PIPELINE_BUG halts the run if a verified mask ever kills a witnessed case.
- Verified-always-outranks-unverified is structural: L1-L3 capped at 0.19 via min(), L5 floor is 0.20; partial credit is shaping only (0.02 weight) and MAP-Elites keeps sketches alive without making them competitive.
- SAT never runs inside evaluate(); difficulty labels are precomputed from the CNF alone and read-only; closure is a separately triggered job with an estimated-work threshold.
- Prune.mono (a prune for weight w serves every w' >= w) and CondPrune/BoundTable make neighbour bounds explicit hypotheses instead of silent facts; the sorry-inside-sound-only rule is backed by the axiom audit, so a misclassified sketch can only be under-credited, never trusted.
- Realistic $0 test plan: golden adversarial candidate bank (mask_spoof, axiom_smuggle, slow_kill, sorry_in_kill...), SyntheticMutator/ReplayLLM loop, closure smoke on (9,9,50).

Weaknesses:
- In v1 a bound is 'claimed' through a closure report, not a Lean theorem: cover completeness (partitions.py + sorting) is discharged in Python and the final theorem upper_bound_of_cover with checked leaves is deferred to M8/M9. Until then the trusted base is the largest of the three (Python enumerator, Python encoding, drat-trim/lrat-check, csv facts) and only 'printed verbatim' rather than encoded in a statement.
- BoundTable.Sound is one coarse hypothesis quantifying over every entry of the table: a CondPrune's kill may look up any entry, so the closure theorem cannot say which external facts were actually load-bearing without extra bookkeeping ('gateTable lists exactly the entries used' is asserted, not enforced by the type). Designs 1/2 attach per-fact hypotheses, which is tighter provenance.
- Table mode lets the Python case generator use data/exact_33.csv inside Argument I on prefixes; those external facts enter the *survivor set* outside Lean, so their use is recorded in a ledger rather than appearing as hypotheses of any checked statement in v1.
- The empirical band (0.06*M^Y + 0.02*T^Y) is admitted to be gameable by an unsound-but-uncaught Python kill on target cells; it is bounded but still a reward channel with no verification behind it.
- The sorry-only-inside-sound rule is regex-based ('after sound := by up to the next top-level declaration'); nested defs/where blocks and unusual layout can confuse the classifier. Not a soundness hole (the audit catches sorryAx) but a ladder-accuracy one.
- Optional in-evaluator LLM filler and a G2 search-mode genome on a second island add complexity; the G2 template path is only safe if every template is proved for all parameter values, which is stated but not designed in detail.
- generality g (fraction of instances where the prune fires at all) is trivially maxed by a prune that fires on one easy case per instance; only 0.05 weight, but it is free score.

Keep:
- Nonce-tagged MASK BEGIN/END markers regenerated per evaluation, parse only between the exact markers, mask length must equal case count
- Ban every # command and IO in candidate Lean; harness writes the only #eval
- leanchecker replay + clean re-run of S0-S4 at promotion time before a prune enters Evolved.lean
- PIPELINE_BUG: a Lean-verified mask killing a witnessed case halts the run, not just the candidate
- Declared-name blacklist (`^(def|theorem|abbrev|structure|instance)\s+(Valid|HasKst|Params|Mat|Prune|...)`) and wrapper declarations outside namespace Cand with end Cand / end ZarPrune forbidden
- Prune.mono: a prune proved for the smallest weight serves every larger weight
- Verified candidates score >= 0.20, unverified capped at 0.19 via min(); partial credit is a MAP-Elites/parent-survival device, not a competitor
- SAT never runs inside evaluate(); difficulty labels precomputed, cached, read-only; closure daemon triggered by estimated remaining work or survivor count
- Golden adversarial candidate bank (mask_spoof, axiom_smuggle, implemented_by, opaque, slow_kill, sorry_in_kill/sound, unsound_row7, instance_specific) with expected ladder/score in tests/golden.json
- sorry permitted only inside sound bodies with the axiom audit as the independent second layer (L3 never trusted)
- Mathlib-free fallback gate profile as a config switch
- Cost ledger before/after every run with a hard reserve floor
- table_hash in every metric dict
- Two-stage architect/filler via OpenEvolve template_variations rather than relying on per-model system messages
- Round-robin conflict escalation across all survivors of a cell so the hard tail is identified before any single case eats hours

### Design 1 — score 6.5

Strengths:
- The genome writes no Lean and cannot express a kill, a leaf list, or a result: kills come only from #eval of already-proved library terms, so the LLM-facing soundness surface in Loop A is essentially zero.
- Best cover story of the three at levels 0/1: `survivors` is *defined* in Lean as the filter over a verified enumerator (mem_enumSorted with PrefixSound tables) and the harness solves the list Lean #eval'd, so cover is a theorem about the very list being solved rather than a post-hoc check.
- Level-2 sub-cubes carry an LRAT tautology certificate (F ∧ Sym ∧ c ∧ ¬d_1 ∧ ... ∧ ¬d_k UNSAT), which is the right uniform obligation for any future solver-generated split and would catch a dropped/canonicalised child.
- Per-fact provenance (`Fact` with tag, `CondPrune P facts`, `discharge`), trust tiers proved-here ⊂ tan2022 ⊂ unreviewed-2026, and a claim rule (tan2022 or better) that makes conditional runs visibly conditional.
- Two trust domains resolve the native_decide conflict explicitly: `_native` axioms only in harness refutation modules with names recorded and re-matched.
- Defense in depth beyond Lean: 5% seeded audit that solves *killed* leaves to completion (the canary after Afrasyab's retracted certificates), two LRAT checkers, has_kst re-check of every model, witness battery with record profiles.
- Anti-gaming of the difficulty proxy is thought through: parent-capped sub-cube credit, exact table values on train, candidate-independent sampled probes with common random numbers, uncertified UNSAT counts as OPEN on train.

Weaknesses:
- Sandbox inconsistency that is a real soundness/integrity hole as written: `Plan` carries Python callables (`refine`, `order_key`) that the harness invokes mid-run, yet the design only sandboxes `plan()` in a subprocess with a 20 s cap. Callables cannot cross that boundary, so either they run inside the harness process (where candidate code could monkeypatch certify/verify, tables, or the Lean gate invocation) or the design is unimplementable as stated. This must be fixed (run the whole policy in the sandboxed subprocess with an RPC protocol, or make refine declarative).
- The trusted base grows rather than shrinks: cuts added to leaf CNFs (pair_cut, min_row_cut) need both a Lean inequality and a trusted clause encoding; counters become base variables with their own exact two-directional encoding; four SubSplit kinds each need cover; row_support canonicalisation 'within equal-sum blocks' composes two symmetry breaks (KNW Thm 6/7 territory) and is only saved by the LRAT cover check.
- The claim is still not a Lean theorem until M5 (Encode.lean, exists_doubleLex); before that, closures rest on external checkers plus the Python cube/cut encoders, exactly as in Design 0.
- Reward is partly gameable and non-reproducible: the closed-instance score is a CPU-time ratio (machine- and load-dependent, incompatible with paired comparison), targets credit *provisional* pysat UNSAT that was never certified, and a policy may select `ledger_trust: unreviewed-2026` to inflate settled_mass on targets with only an ω halving as the cost.
- SAT runs inside every evaluation (60-180 s, 2 leaf workers): heavy-tailed noise enters the score, evaluation dominates wall-clock, and the difficulty of a *killed* leaf on targets is an estimate the policy indirectly optimises against.
- Less aligned with the proposal's stated question ('can LLMs operate in Lean directly, or a two-step process?'): the LLM-writes-Lean loop is demoted to a rare offline forge; the thesis's interpretable artefact (a new counting argument) is not what Loop A produces.
- Lean #eval of the survivor enumerator at frontier cells (10^5-10^6 profiles pre-Argument-I, 711k pairs at (16,17) pure) is acknowledged as high-likelihood/high-impact and the mitigation ('accept minutes, cached') is untested.

Keep:
- Define `survivors` in Lean as a filter over a verified enumerator (mem_enumSorted + PrefixSound) and #eval it, so the solved list is the list the cover theorem is about
- Per-fact `Fact {m,n,s,t,z,tag}` with FactHolds, CondPrune over a fact list, and `discharge` by term application when a fact is proved-here; trust tiers proved-here ⊂ tan2022 ⊂ unreviewed-2026 with a claim rule
- LRAT tautology cover certificate for any sub-cube split (F ∧ Sym ∧ c ∧ ⋀¬d_i UNSAT)
- Two trust domains: `_native` axioms only in harness refutation modules, names recorded at generation and re-matched on audit
- 5% seeded kill audit: solve a sample of lemma-killed leaves to completion; any SAT is a hard zero + run halt + incident log
- Parent-capped credit for sub-cube children so splitting easy leaves cannot inflate settled mass
- Cuts admitted only if registered with a Lean inequality statement and an entry in the encoding-completeness clause list; harness rejects unknown cut names
- Unary counters as base CNF variables so every level of the decomposition is a partial assignment over one fixed formula (enables cube certificates and AlphaMapleSAT-style sub-cubing)
- SAT witness on a target is a lower-bound discovery: verify with has_kst, store, raise w for future runs
- NFKC-normalise Lean source before the forbidden-token scan; strip block comments and scan the raw text again
- Hand-written adversarial policy/candidate set (cheat_claims_kill, cheat_bad_cover, cheat_zero_budget, injected fake mask) with asserted outcomes
- tools/stub_llm.py: an OpenAI-compatible HTTP mutation server so the full OpenEvolve loop (islands, checkpoints, MAP-Elites) runs at $0
- Report tail_share and critical-path difficulty max_l D(l) alongside total work
- Hard stop at $14 with smoke scripts refusing to start below $2 remaining

### Design 2 — score 8.5

Strengths:
- Works backward from the artefact that matters under this lens: a new bound *is* one Lean declaration (upper_bound_succ_of_sorted_cover applied to prunes + a decidable cover + refutation hypotheses), elaborated in a file the LLM never touches. There is no separate 'closure report claims a bound' path; the report is generated together with the theorem.
- Trusted base is enumerated per tier in one table (T1-T7) and is the smallest of the three at Tier-1: kernel, Basic.lean (~60 lines, the statement), Mathlib modules, harness-owned Closure.lean, and the external LRAT checkers appearing as *named hypotheses* with certificate hashes. External facts are per-cell hypotheses (Htan_12_17, ...) in the theorem statement, so provenance is at the type level, not in a ledger sidecar.
- Cover discharged by kernel `decide` over a Lean enumerator with a completeness theorem (mem_genParts) plus thinning (P19), permutation action and sorting lemmas; the `decide +native` fallback is confined to the harness domain with a named axiom in the ledger. Tier-0 (Encode.lean + LRAT.check_sound) is a clean milestone, not a prerequisite.
- Partial credit is bounded at 0.20 total via min(0.20, 0.10*lean_partial + 0.10*E), L4 (audit failure after a clean scan) scores 0 as an attempted escape, and no sorry-containing proof ever earns kill mass; the sketch-and-fill path only produces credit after the harness-filled copy passes the ordinary L5 gate.
- Verified schemas (Prune.ofFarkas / ofResidue / ofPrefix) proved once for all parameter values let the search vary data at zero Lean cost without any new trust; this is the safe form of the G2/G4 leverage the review recommends.
- Concrete hardening details: candidate runs on a copy of cache/ (cannot rewrite the environment it is scored against), content-hash gate cache, autoImplicit false, redefinition scan after def/abbrev/structure/inductive, full-name references in the wrapper, kernel-only elaboration from source, leanchecker on closure files.
- Stationary reward (no running-record cliff), n_lean_decls dropped as a feature (rewards length), kill_novelty against an accepted-prune ledger, first deliverables are zero-SAT conditional closures ((10,21),(11,19),(11,20)) that exercise the whole seam before any solver is trusted.

Weaknesses:
- Kernel `decide` on a cover of 10^3-10^4 cases, each evaluating a kill built from Nat.choose sums, is asserted to take 'seconds to minutes' without measurement; if it is hours the fallback is a `+native` axiom, which quietly moves Tier-1 closer to Design 0's trusted base. T-7 measures this, but it is the load-bearing unmeasured number.
- No nonce on the mask markers: with #eval and IO forbidden in candidates the forging channel is already closed, but Design 0's nonce is a cheap second layer this design omits.
- The '#eval truncation bug found today' appears to describe a gate that already emits MASKLINE strings (zar_ub/lean_gate.py lines 167-174); either the fix landed after the observation or the claim is stale, so the 'lean_ok = 0.6 for the baseline' number should be re-verified before it drives M1.
- The x0.9 penalty when the Python mirror disagrees with the Lean mask taxes a verified candidate for a defect in an untrusted artefact; it pushes the LLM to maintain a mirror rather than a better proof, and a mirror that is deliberately made vacuous (kills nothing) is *also* a disagreement, so the incentive is muddled.
- The regex-enforced sketch rule ('sorry only as the entire body of a have inside a theorem concluding ¬ Valid P A') and the harness auto-fill tactic list (omega, simp_all, decide, linarith, nlinarith, grind, lemma lookup) are a fair amount of mechanism whose value is unmeasured; the in-evaluator repair loop reintroduces an LLM call into evaluation (cost-capped, but it makes evaluate() non-deterministic).
- hard_killed on target cells earns cap-credit for censored cases; legitimate (the kill is proved) but it means the target term can be dominated by kills of cases whose difficulty is an extrapolation from a factor-3 calibration.
- Closure.lean at 600-900 lines and Schemas.lean (residue identities ~450 Mathlib lines) are substantial hand-proof effort on the critical path (M1, M4); the design's soundness story depends on this harness-owned Lean being audited, which is a human-review item the design names but cannot automate.

Keep:
- A bound is one Lean theorem in a harness-owned file: prunes + `cover := by decide` over a verified enumerator + refutation as named hypotheses (Tier-1) or LRAT.check_sound (Tier-0)
- Per-cell external facts as explicit theorem hypotheses (Htan_12_17 : ∀ B, ¬HasKst → weight B ≤ 103) rather than a single table-soundness hypothesis
- The trusted-base table by tier (T1-T7) printed in every closure report, with `decide +native` fallback confined to the harness domain and its axiom named in the ledger
- Closure.lean obligations spelled out: exact-weight thinning (P19), act/weight_act/hasKst_act_iff, exists_sorted via Tuple.sort, mem_genParts completeness, upper_bound_of_sorted_cover
- Verified schemas Prune.ofFarkas / ofResidue / ofPrefix: soundness proved once, LLM supplies only data in SCHEMA_DATA
- L4 (elaborates but fails the axiom audit) scores 0 as an attempted escape; L0-L3 capped at 0.20; no sorry proof ever earns kill mass
- Candidate Python runs against a *copy* of cache/ so it cannot rewrite the environment it is scored against; content-hash cache of Lean gate results
- Redefinition scan for HasKst/Valid/Params/Profile/Mat/weight/rowSum/colSum immediately after def/abbrev/structure/inductive, plus full-name wrapper references and autoImplicit false
- Baseline inside the genome equals the proved library `counting P` so the initial program scores exactly 0.20 and every increment is a new proved kill
- Stationary reward with MAP-Elites axes [proven_gain, lean_ladder, kill_novelty]; drop n_lean_decls as a feature
- Zero-SAT conditional Lean closures ((10,21)≤106, (11,19)≤106, (11,20)≤111 conditional on Tan 2022) as the first deliverable that exercises the whole seam
- Any SAT model at w = z+1 is re-checked with independent has_kst and halts the pipeline (encoding/enumeration bug protocol)
- Shared-sample (common random numbers) reward estimation on large tables with N=500 survivors drawn once per table seed
- Post-run zero-shot transfer matrix of the accepted library across every cached table
- Chunked string kill-mask emission with a 700+-case regression test

---

## Judge 1 — practicality-and-evolvability lens

**Ranking:** [2, 0, 1]

**Verdict.** Design 2 wins under the practicality-and-evolvability lens. All three agree on the trusted base, the Lean gate, difficulty = CaDiCaL conflicts with a c2000 proxy, and keeping SAT out of (or bounded inside) the loop, so the deciding questions are: what can an LLM earn per iteration, how long does an evaluation take, and how much must be built before the loop is interesting. Designs 0 and 2 share the same 2-6 s no-SAT inner loop and the same central risk (a 0.20 plateau while the LLM fails to prove new arguments in Lean 4.34); Design 2 mitigates it with the SCHEMA_DATA channel (verified credit from evolved parameters of once-proved Farkas/residue/prefix schemas), harness-side tactic auto-fill of sketch holes, a content-hash gate cache, and a concrete, already-validated bug fix, and it leads with zero-SAT deliverables that need no LLM at all. Design 0 is the most thorough on anti-cheating, conditional bounds (CondPrune/BoundTable), difficulty calibration and closure operations, and most of that should be merged into Design 2's plan, but its v1 offers no proof-free way to earn verified credit and its architect/filler split depends on routing OpenEvolve cannot do. Design 1 has the densest reward and the most editable genome, and its stub-LLM server, lemma registry, kill-audit canary and provenance tiers are must-keeps, but it puts SAT solving and certification inside every evaluation (60-180 s, CPU-bound, noisy CPU-second cost ratios), rewards solver-policy tuning on cells that already close in seconds rather than the interpretable verified prunes the proposal asks for, and needs 6-8 weeks of new Lean and harness code (cover theorems, Cond/Farkas/DGH, a re-done encoding that invalidates the E10 labels) before its policy loop has meaningful knobs. Recommended plan: build Design 2's loop and closure path, import Design 0's CondPrune/BoundTable, ledger DAG, nonce markers, golden bank, difficulty schedule and closure daemon, and adopt Design 1's stub HTTP LLM, lemma registry and kill-audit; keep a policy loop as a later ablation once a verified library exists.

### Design 2 — score 8

Strengths:
- Cheapest loop of the three that still rewards the thesis's actual artefact: 2-5 s per evaluation, no SAT in evaluate(), one Lean process per suite, content-hash cache so re-sampled parents never re-run the gate. Time per iteration is LLM-bound, exactly the regime OpenEvolve and AlphaEvolve Cloud (30-min lock, no cascade needed) are built for.
- The SCHEMA_DATA channel (Farkas multipliers, residue moduli/marked sets against once-proved Prune.ofFarkas / Prune.ofResidue) is the one idea in the three documents that gives an LLM a dense, Lean-proof-free way to earn verified credit every iteration. Python inside the EVOLVE block can compute those parameters, so this is de facto G4/search-mode leverage while keeping G1 as the visible genome.
- Harness-side tactic auto-fill of typed `have ... := by sorry` holes (omega, simp_all, decide, linarith, grind, fixed lemma list) plus an optional $0.01-capped flash repair loop turns L2/L3 sketches into L5 at $0 per hole for the arithmetic-heavy prunes this problem actually needs; that is the lit-review's Goedel-Architect split implemented without per-model routing that OpenEvolve does not have.
- Grounded in the running code: found and reported the >50-element `#eval` List Bool truncation that silently zeroed the baseline's Lean mask (now fixed as MASKLINE in lean_gate.py). The zero-SAT conditional closures (10,21),(11,19),(11,20) and the 244-case (9,23,104) first target are concrete, measured, and reachable before any LLM money is spent.
- Reward is stationary (no running-record cliff), verified floor 0.20 strictly above any unverified score, tail term 0.20 targets the 61.6%-of-work decile, L4 (audit failure after clean scan) is an attempted escape and scores 0, PIPELINE_BUG halts the run; kill_novelty Jaccard against an accepted-prune ledger keeps distinct arguments alive.
- NO-LLM mode and the $16 plan are realistic and verified against the framework: `init_client` replay hook exists; P-1..P-5 total about $2.65 with a $3 reserve and a per-model verified-prune-rate-per-dollar decision rule for A (direct) vs B (sketch).

Weaknesses:
- Same fundamental evolvability risk as Design 0: 80% of the score sits behind L5 proofs of genuinely new arguments (DGH(4), cross-side, residues). Until Schemas.lean (M4, hand-written; the residue identities are ~450 Mathlib lines) lands, most iterations will sit at exactly 0.20 and the empirical/ladder band gives only a 0.20-wide gradient.
- Explicitly rejects G2 search mode as 'must be re-proved', which contradicts its own schema channel; it does not spell out that the LLM may write Python that searches for SCHEMA_DATA, nor how SCHEMA_DATA is spliced into the Lean `candidate` term. That plumbing decides whether the schema channel is dense or decorative.
- Three MAP-Elites dimensions with 10x6x5 = 300 cells against a population of 40-60 will leave the grid mostly empty; kill_novelty also depends on a ledger that is empty at run start.
- Kernel `decide` on genRows x genCols for cover (thousands of Nat.choose evaluations per case) is unmeasured and may need `decide +native` even on (12,18); the Tier-1/Tier-0 split is honest but Closure.lean (600-900 lines) is on the critical path of M1.
- The 0.9 multiplier for K^L != K^P penalises a correct Lean prune whose Python mirror is stale; a diff that only improves Lean gets punished, which nudges the LLM toward keeping two implementations in sync rather than toward new arguments.
- Cost note: it is realistic about the $16 but the smoke runs (P-3/P-4 at 20 and 15 iterations on flash/mid) are too small to distinguish prompt variants; the decision on A vs B will rest on ~12 probe calls.

Keep:
- SCHEMA_DATA: evolve data for once-proved Prune.ofFarkas / Prune.ofResidue / Prune.ofPrefix schemas (Lean cost zero per iteration).
- Harness-side auto-fill of `have` holes with a fixed tactic ladder before any paid repair call.
- Content-hash cache on the Lean gate (identical LEAN_SOURCE never re-checked).
- String-emitted kill mask in <=500-case chunks (already in code); regression test on a 725-case table; abort a run if the baseline program scores lean_ok < 1.
- Zero-SAT conditional closures (10,21)<=106, (11,19)<=106, (11,20)<=111 as Lean theorems with explicit Htan hypotheses as the first deliverable.
- Closure theorem shape with named hypotheses (Tier-1) and a Tier-0 LRAT.check_sound milestone; leanchecker only on closure files.
- L4 (elaborates but fails axiom audit or slipped forbidden construct) scores 0 as an attempted escape; PIPELINE_BUG halts the run.
- Stationary reward with verified floor 0.20, tail term, GEN held-out set with (2,2)/(4,4) cells, kill_novelty Jaccard vs the accepted-prune ledger.
- Candidate ladder test bank (tests/candidates/) with expected scores checked in; copy cache tables before executing candidate code.
- Difficulty: c2000 probe + log-linear censored calibration, shared 500-survivor sample for |S_I|>3000, zero-shot transfer matrix reported not rewarded.
- $16 decision rule: choose direct vs sketch prompt by verified-prune rate per dollar per model.

### Design 0 — score 7

Strengths:
- Same cheap inner loop as Design 2 (3-6 s per evaluation, no SAT, one Lean process, measured 1.57 s gate) and the cleanest OpenEvolve fit: single EVOLVE block, diff mode, cascade_thresholds [0.005] so dead programs skip Lean, raw metrics for MAP-Elites, `init_client` replay/synthetic mutator for $0 loop tests (hook verified to exist).
- The most complete anti-cheat inventory: nonce-tagged MASK markers, declared-name blacklist, autoImplicit false, comment-stripped plus raw scan, axiom audit, leanchecker at promotion, marginal scoring over the Lean-verified baseline (the E6 lesson), difficulty labels precomputed and read-only.
- CondPrune / BoundTable with `T.Sound` separated from the lookup data, `Prune.mono`, a provenance ledger DAG and a mechanical claim checker: the right formal shape for chaining bounds across cells, which is where the first real closures come from.
- Difficulty measure is the most carefully specified: escalating schedule, censored labels as max(cap, regressor on c2k), stratified common-random-number samples for 30k-700k-case tables, LRAT length as a sanity check, table_hash in every metric dict.
- Operationally mature: closure daemon triggered by estimated remaining work, golden candidate bank with expected statuses, $8 reserve rule, per-step cost logging, DGH-attackable cells (13,17)/(13,18)/(15,17) added to the suite specifically to escape the 0.20 plateau.

Weaknesses:
- Evolvability is the weak point: 0.60 of the score is M^L, which is zero until the LLM proves a genuinely new argument in Lean 4.34; the unverified band (0.03 ladder + 0.02 partial + 0.06 M^Y + 0.02 tail, capped 0.19) gives almost no gradient and the empirical term is capped at 0.06. Expect long plateaus at 0.20 with cheap models; the design itself lists this as risk 1 and 2.
- The Lean-free progress channels (G2 template genome, residue identities M6, Farkas) are all deferred to v2; nothing in v1 lets an iteration earn verified credit without writing a proof. Design 2's schema channel is strictly better here.
- The architect/filler split relies on OpenEvolve ensemble weights plus `template_variations`, which does not route a specific prompt to a specific model; the design admits per-model system messages may not be honoured. In practice this degrades to a random mix of two prompts.
- Large surface area before the loop gets better: Table.lean, ledger DAG, closure daemon, k-prefix CondPrune (~150 lines), difficulty regressor, SyntheticMutator, golden bank; much of it is post-loop machinery that does not improve what the LLM can do per iteration.
- Some numbers are optimistic or stale: ~$30-80 per 1,000 iterations assumes 12k/6k tokens with a 70/30 filler/architect mix; the prompt still needs the Mathlib-available rewrite; the closure of (10,23) is explicitly not expected on a laptop.
- MAP-Elites on [empirical_gain, lean_status] is sensible, but empirical_gain depends on an unsound-until-proved Python kill on target cells that no witness catches; the design bounds the reward impact but the feature grid can still fill with unverifiable phenotypes.

Keep:
- Marginal scoring over exactly the Lean-verified baseline; verified floor 0.20 above every unverified score.
- Nonce-tagged MASK BEGIN/END markers regenerated per evaluation, mask-length check, `#` commands forbidden in candidates.
- CondPrune P T with BoundTable data separated from BoundTable.Sound; Prune.mono; final theorem conditional on exactly the ledger entries used, discharged by the DAG when all are pipeline/counting.
- SAT never inside evaluate(); closure daemon triggered when estimated remaining work Sum d(q) drops under budget or survivors < 200.
- Difficulty labels: escalating conflict schedule, censored = max(cap, f_hat(c2k)), refit on ladder cells, stratified CRN sample when |S_I| > 50k, table_hash recorded in every metric dict.
- Ledger provenance classes (counting / pipeline / external) with 2026 preprint claims kept in a separate targets-only file; mechanical monotonicity claim checker on every write.
- Golden candidate bank (initial, python_only_dgh4, lean_dgh4, unsound_row7, sorry variants, native_decide, mask_spoof, slow_kill, instance_specific, cond_deletion) with expected statuses.
- DGH-attackable cells (13,17,117), (13,18,122), (15,17,133) in the suite so counting-type prunes can score early; hand-prove DGH(4) as M3 if the LLM does not.
- cascade_thresholds [0.005] so only unsound programs skip the Lean stage; `sorry` allowed only inside `sound` bodies (regex) and always capped at L3.
- $8 reserve rule and per-evaluation filler cost cap; Mathlib-free gate profile as a config fallback.
- G2 template genome (evolve marked sets / moduli / multipliers, evaluator instantiates a fixed Lean template) on a separate island once the identities exist.

### Design 1 — score 6

Strengths:
- The genome an LLM will actually edit successfully: `plan()` is ordinary Python over a documented dataclass API, so cheap models (Flash / gpt-oss) produce valid diffs at high rates and every candidate receives a continuous, difficulty-weighted score (settled_mass or clipped log cost ratio). This is the densest reward of the three and the only one where progress in the first 100 iterations is near-certain.
- Best $0 loop test: an OpenAI-compatible stub HTTP server on `api_base` exercises the real OpenEvolve client path, checkpoints, islands and MAP-Elites without touching `init_client`; the hand-written `policies/` set with asserted orderings (including cheat policies) is a strong regression harness.
- Correctly reads the E10 data: heavy-tailed leaves mean re-cubing the hard decile, leaf ordering, and cuts matter as much as kill counts; parent-capped sub-cube credit, 5% kill-audit canary, two trust domains for `_native` axioms, and provenance tiers with halved weight are all sound anti-gaming/soundness mechanisms.
- Lemma registry with schema-validated parameters and free Lean instantiation is the right abstraction for amortising proof effort across instantiations, and `Cover.lean` with a verified enumerator whose `#eval` output is the very list solved is the strongest cover story of the three.

Weaknesses:
- SAT runs inside every evaluation: 60-180 s wall per Loop-A evaluation on a laptop with 2x2 solver processes competing with Lean; plus certification on every TRAIN leaf and 5% audit solves. Throughput drops to ~1 iteration/min and evaluation, not the LLM, becomes the bottleneck; the closed-branch score uses CPU-seconds ratios that are noisy under exactly that contention.
- On TRAIN cells (all closed by the baseline in < 3 min) the reward is mostly 'solve faster than baseline', i.e. solver-parameter tuning (ladder budgets, cuts, ordering) rather than 'prune more branches'. Novel, interpretable verified prunes, the thesis's stated artefact, are pushed to Loop B (the existing evaluator) plus a `forge.py` driver outside OpenEvolve; Loop A's best outcome is an AlphaMapleSAT-style policy, which is a different thesis.
- The interesting evolvable knobs do not exist yet: `farkas`, `dgh4`, `prefixI`, `Cond.lean`, `Farkas.lean`, `Perm/Enum/Cover.lean`, counters-as-base-variables encoding, `decompose.py`, `policy_runner.py`, `cuts.py`. M1-M3 are 6-8 weeks of harness and Lean work before Loop A has more than 'toggle argD and change budgets' to evolve. Changing the base encoding also invalidates the 1,571 E10 conflict labels and E9 certificates.
- Lean `#eval` of the survivor enumerator on frontier cells (10^5-10^6 profiles before Argument I) is rated high likelihood / high impact by the design itself; the proposed fallback ('accept minutes per instance, cached per split parameter set') means every new split configuration pays that cost inside the loop.
- AlphaEvolve Cloud portability is weaker: the evaluation needs pysat, three solver binaries, certificate storage and 60-180 s of CPU per candidate; it fits the 30-minute lock but is far heavier than a 3 s Lean check, and score noise from CPU-second ratios does not travel across machines.
- The $16 plan is cheap ($0.05-$1 per smoke) but the E12-E14 smoke runs at T_I = 20 s will measure only trivial budget/family toggles, not whether models ever write useful `refine` or Farkas helpers; the design concedes cheap models may never do so.

Keep:
- Stub OpenAI-compatible HTTP LLM server on `api_base` with a seeded mutation bank for $0 end-to-end OpenEvolve runs (checkpoint/resume, islands, feature binning).
- Lemma registry (`lemmas.py`): Lean term + parameter schema + Python mirror + one doc line, validated in Python before the Lean call; instantiation is free in Lean.
- Parent-capped credit for sub-cubes; re-cubing the hardest leaves on the heaviest row's support; tail_share and critical-path (max leaf) difficulty as reported metrics.
- 5% seeded kill-audit that solves lemma-killed leaves to completion as an encoding/Lean-mismatch canary; SAT on a target treated as a verified lower-bound discovery, not a failure.
- Two trust domains: `_native` axioms only in harness refutation modules, recorded by name and re-matched; policies/candidates write no such Lean.
- Provenance tiers proved-here / tan2022 / unreviewed-2026 with claims requiring tan2022 or better and conditional runs down-weighted.
- Verified enumerator whose Lean `#eval` output is the list actually solved (`survivors` defined in Lean), plus LRAT tautology certificates for any future sub-split; counters-as-base-variables so a case is a cube over one fixed CNF (later, for AlphaMapleSAT-style cubing).
- Hand-written policy/adversarial set with asserted score orderings (baseline, no_lemmas, cols_only, recube, pair_cuts, farkas_search, cheat_* variants).
- Curriculum band rule: keep suite instances whose population closure rate is in (0, 0.75]; `cols`-only vs `both` split as a measured ablation.
- Loop-A/Loop-B separation as a later addition: once a verified library exists, a cheap policy loop over instantiations and solver ladders is a legitimate second experiment.

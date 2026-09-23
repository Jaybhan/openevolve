# Reward shaping and partial credit for near-correct Lean proofs; curriculum and proxy rewards; MAP-Elites exploration for LLM program search

Literature notes for the upper-bounds thesis (Zarankiewicz z(m,n;s,t) via pruned case-split SAT + Lean-verified prunes).
Written 2026-09-21. This is a *topic survey* (not a single paper): ~35 primary sources, plus two local experiments
on the ZarPrune toolchain that settle implementability questions. Every quote below is attributed; where a quote
came back through the fetch tool's summarizer rather than my own reading of the PDF/HTML, the source table says so.

The thesis questions this file answers (proposal Section 2.3):

> "LLMs are only just becoming competent at coding in Lean, so how can we reward a Lean function that comes close but
> doesn't compile perfectly? What might proxy rewards look like?" ... "How can we trade off exploration and exploitation?"
> ... "What does the 'difficulty' of a branch actually mean?"

---

## 0. Executive summary (what the evidence actually supports)

1. **Every serious Lean prover trains on a binary outcome reward** (DeepSeek-Prover-V1.5/V2, Kimina, Seed-Prover,
   Leanabell-V2, AlphaProof's per-step −1 is outcome-equivalent). None of them uses "closeness to compiling" as reward.
   What they use instead is **selection by difficulty band** (only train on problems with pass rate in (0, 1/4], (0, 0.75],
   "challenging yet achievable", "mixed success rate") and **structural partial credit**: keep the successful prefix
   (truncate-and-resume), keep proved `have`/lemma blocks and `sorry` the rest (DSP, DeepSeek-V2, APOLLO, ProofAug,
   Seed-Prover lemma pools), then recurse on the holes.
2. The one paper that adds a *dense* Lean-derived reward on top of the outcome reward (Kim et al., ICLR 2026,
   "Process-Verified RL") gets **+2.5 pp on miniF2F and +1.2 pp on ProofNet** over outcome-only GRPO, using exactly the
   signal we can read off `lean --json`: tactics before the first error are "locally sound" (d1 = −0.05), everything from
   the first error on is failed (d2 = −0.1), verified = +1. This is the best evidence for "error-position depth" credit,
   and the effect is real but small.
3. In program-synthesis RL the **graded compile-status ladder** (compile error −1.0 < runtime error −0.6 < failed tests
   −0.3 < pass +1.0; CodeRL) and **error-line-localized penalties** (RLTF) are standard and ablated: RLTF reports the
   fine-grained (error-location) reward "contributes the most significant performance boost" (pass@1 1.37 → 1.41 → 1.45).
4. **Autoformalization agreement checks are mandatory and cheap, and they are not oracles.** Compile-but-unfaithful rates
   run 3–29 pp (Beyond Compilation, 2026); 60.7% of compiling FormalMATH statements failed semantic checks; the best
   symbolic equivalence check (BEq) has 100% precision but recall ~13 pp below accuracy. For *our* task the NL statement
   is fixed ("kill implies no valid matrix") and the Lean statement is the fixed `Prune P` structure, so the
   NL↔Lean agreement problem is almost entirely solved by construction; what remains is **vacuity** (kill ≡ false) and
   **triviality** (killing only cases already killed by the baseline), which must be measured, not judged.
5. For QD/MAP-Elites, the strongest evidence is for: (a) a **behavior descriptor that is a phenotype, not code text**
   (FunSearch clusters by the score-signature vector; ELM/QDAIF use domain features), (b) **count-based / curiosity-based
   cell selection** (Go-Explore 1/(count+ε)^0.5 weights; Cully & Demiris curiosity +1/−0.5 outperforms uniform), (c)
   **improvement-ranked emitters** where "new cell" outranks "better elite" (CMA-ME), (d) **novelty rejection** of
   near-duplicates before evaluation (ShinkaEvolve η = 0.95 embedding similarity), and (e) **island resets** (FunSearch
   resets the weaker half of islands). OpenEvolve already has islands, a MAP-Elites grid with custom feature dims,
   exploration/exploitation/random ratios (0.2/0.7/0.1) and an embedding novelty gate (similarity_threshold 0.99).
6. **Local fact (measured):** on the Mathlib-free ZarPrune library, `lake build` is a no-op in 0.14 s and elaborating a file
   with four candidate prunes (one `sorry`, one mid-proof error, one unsolved goal, one ill-typed `kill`) takes 0.23 s
   wall; `lean --json` reports per-message `kind` (`hasSorry`, `Tactic.unsolvedGoals`,
   `lean.unknownIdentifier._namedError`, `lean.synthInstanceFailed._namedError`), `pos`/`endPos`, and the full goal
   context. "Does Lean take too long to compile?" — for this library, no; a Lean check is cheaper than a SAT probe.
7. **Local fact (measured):** core Lean 4.34 without Mathlib has no `Nat.choose`, no `Finset`, no `List.sublists`. The
   Kővári–Sós–Turán counting prune (Tan's Argument A) needs a hand-rolled binomial/subset layer; everything else in the
   proposed reward design is provable in a few lines.

---

## 1. Sources and access

| # | Source | Access | Read how |
|---|---|---|---|
| S1 | Xin et al., DeepSeek-Prover-V1.5, arXiv:2408.08152 | full text | HTML via fetch tool |
| S2 | Ren et al., DeepSeek-Prover-V2, arXiv:2504.21801 | full text | HTML via fetch tool |
| S3 | Xin et al., DeepSeek-Prover (V1), arXiv:2405.14333 | full text | HTML via fetch tool |
| S4 | Liu et al., ProofAug, arXiv:2501.18310 | full text | HTML via fetch tool |
| S5 | Ospanov et al., APOLLO, arXiv:2505.05758 | full text | HTML via fetch tool |
| S6 | Lu et al., FormalAlign, arXiv:2410.10135 (ICLR 2025) | full text | HTML via fetch tool |
| S7 | Dong & Ma, STP, arXiv:2502.00212 | full text | HTML via fetch tool |
| S8 | Lin et al., Goedel-Prover-V2, arXiv:2508.03613 | full text | HTML via fetch tool |
| S9 | ByteDance Seed, Seed-Prover, arXiv:2507.23726v2 | full text | PDF pp. 1–12 read directly (scratchpad `lit/seed_prover.pdf`) |
| S10 | Wang et al., Kimina-Prover Preview, arXiv:2504.11354 | full text | HTML via fetch tool |
| S11 | Ying et al., Lean Workbook, arXiv:2406.03847 | full text | HTML via fetch tool |
| S12 | Yu et al., FormalMATH, arXiv:2505.02735 | full text | HTML via fetch tool |
| S13 | Wang et al., LEGO-Prover, arXiv:2310.00656 | full text | HTML via fetch tool |
| S14 | Jiang et al., Draft-Sketch-Prove, arXiv:2210.12283 | full text | HTML via fetch tool |
| S15 | Kim et al., Process-Verified RL for Theorem Proving via Lean, arXiv:2606.20068 (ICLR 2026) | full text | HTML via fetch tool |
| S16 | Leanabell-Prover-V2, arXiv:2507.08649 | full text | HTML via fetch tool |
| S17 | Hubert et al., AlphaProof, Nature (2025/26), doi 10.1038/s41586-025-09833-y | full text | nature.com via fetch tool (cookie-bypass URL) |
| S18 | Liu et al., BEq / RAutoformalizer, ICLR 2025 | full text | proceedings PDF → text, read directly (scratchpad `lit/beq_iclr2025.pdf`) |
| S19 | "Beyond Compilation: Evaluating Faithful NL-to-Lean Statement Formalization", arXiv:2606.31002 | full text | HTML via fetch tool |
| S20 | "Faithful Autoformalization via Roundtrip Verification and Repair", arXiv:2604.25031 | full text | HTML via fetch tool |
| S21 | FormalEvolve, arXiv:2603.19828 | full text | HTML via fetch tool |
| S22 | Le et al., CodeRL, arXiv:2207.01780 | full text | HTML via fetch tool |
| S23 | Liu et al., RLTF, arXiv:2307.04349 | full text | HTML via fetch tool |
| S24 | Bradley et al., QDAIF, arXiv:2310.13032 | full text | HTML via fetch tool |
| S25 | Lehman et al., ELM, arXiv:2206.08896 | full text | HTML via fetch tool |
| S26 | Novikov et al., AlphaEvolve, arXiv:2506.13131 | full text | HTML via fetch tool |
| S27 | Sakana AI, ShinkaEvolve, arXiv:2509.19349 | full text | HTML via fetch tool |
| S28 | Romera-Paredes et al., FunSearch — `implementation/programs_database.py` | code | raw GitHub via fetch tool |
| S29 | Cully & Demiris, QD: A Unifying Modular Framework, IEEE TEC 2018, arXiv:1708.09251 | full text | PDF pp. 1–10 read directly |
| S30 | Ecoffet et al., Go-Explore, arXiv:1901.10995 | full text | PDF pp. 4–9 + Appendix A.5 text read directly |
| S31 | Fontaine et al., CMA-ME, GECCO 2020, arXiv:1912.02400 | full text | PDF pp. 1–6 read directly |
| S32 | Kilby, Slaney, Thiébaux, Walsh, Estimating Search Tree Size, AAAI 2006 | full text | PDF read directly (scratchpad `lit/kilby2006.pdf`) |
| S33 | Jiang, Grefenstette, Rocktäschel, Prioritized Level Replay, arXiv:2010.03934 | full text | HTML v4 via fetch tool |
| S34 | Kumarappan et al., LeanAgent, arXiv:2410.06209 | full text | HTML via fetch tool |
| S35 | Tan, An attack on Zarankiewicz's problem through SAT solving, arXiv:2203.02283 | full text | local PDF `docs/papers/` → pdftotext |
| S36 | leanprover-community/repl README | doc | raw GitHub via fetch tool |
| A1 | Polu et al., Formal Mathematics Statement Curriculum Learning, arXiv:2202.01344 | abstract only | |
| A2 | Lample et al., HyperTree Proof Search, arXiv:2205.11491 | abstract only | |
| A3 | Wu et al., InternLM2.5-StepProver, arXiv:2410.15700 | abstract only | |
| A4 | Pourcel et al., ACES, arXiv:2310.10692 | abstract only | |
| A5 | DEI: Diversity in Evolutionary Inference, arXiv:2605.27130 | abstract only | |
| A6 | Seed-Prover 1.5, arXiv:2512.17260 | abstract only | |
| A7 | Vericoding, arXiv:2509.22908 | abstract only | |
| A8 | "Automating Formal Verification with RL and Recursive Inference", arXiv:2605.30914 | abstract only | |
| N1 | Lehman & Stanley 2011 novelty search | not accessible (TLS error on eplex mirror); definition taken from S29's description | |
| N2 | Mouret & Clune 2015 MAP-Elites | not fetched; algorithm taken from S29 Sec. II-D | |

Local experiments: `scratchpad/leanprobe/Probe.lean` (four partial candidates through `lake env lean --json`) and
`scratchpad/leanprobe/Check.lean` (`#check Nat.choose / Finset / List.sublists`).

---

## 2. Rewarding a Lean proof that does not compile: the catalogue

Each signal: what the literature does (quoted), what it buys, how to implement it on ZarPrune, and how it can be gamed.

### 2.1 The compile-status ladder (graded outcome tiers)

**Evidence (program RL).** CodeRL (S22) defines the return by unit-test outcome:
> "r(Ws) = -1.0, if Ws cannot be compiled (i.e. compile error)"; "-0.6, if Ws cannot be executed with unit tests";
> "-0.3, if Ws failed any unit test"; "+1.0, if Ws passed all unit tests".
Ablation (their Table 2): graded four-tier reward with critic 2.20% pass@1 vs 1.62% (identical token rewards) vs 1.36%
(no baseline). RLTF (S23) Eq. 3 uses the same tiers: `1.0 pass / −0.3 failure / −0.6 error except syntax / −1.0 syntax error`.

**Evidence (Lean RL).** Leanabell-Prover-V2 (S16): "we primarily use two rewards: format reward (i.e., R_format) and
compilation status reward (i.e., R_failed and R_success)", "R_format is set to a smaller value (such as 0.2) compared
to R_failed/R_success (such as 1.0)". Kimina (S10): "A binary reward signal is assigned: 1 for a completely correct proof
and 0 otherwise", plus a *format filter*: "each generated sample must contain at least one tactic block; and (2) tactic
blocks must collectively cover at least 60% of the Lean code". Seed-Prover (S9, Sec. 2.2.3, read directly): "The RL reward
is 1 if the formal statement is successfully proven, and 0 otherwise. Additionally, a formatting penalty is applied to
encourage the model to generate lemmas before attempting the main theorem."

**What Lean gives us (measured).** `lake env lean --json Probe.lean` on ZarPrune returned one JSON object per message with
`severity`, `kind`, `pos`, `endPos`, `data`. Observed kinds for the four planted failure modes:

| Candidate | Planted defect | `severity` / `kind` | `data` (head) |
|---|---|---|---|
| cand1 | `sound := by intro A h; sorry` | warning / `hasSorry` | "declaration uses `sorry`" |
| cand2 | wrong `simpa`, then unknown name | error / `[anonymous]`; error / `lean.unknownIdentifier._namedError` | "Type mismatch: After simplification, term h has type decide (...) = true but is expected to have type sumFin P.m (rowSum A) < P.w"; "Unknown identifier `hk'`" |
| cand3 | stopped after one `have` | error / `Tactic.unsolvedGoals` | "unsolved goals P : Params A : Mat P.m P.n h : ... hv : Valid P A hw : P.w ≤ weight A ⊢ False" |
| cand4 | `kill := ... + 1` (Bool + Nat) | error / `lean.synthInstanceFailed._namedError` | "failed to synthesize instance of type class HAdd Bool Nat Bool" |

Wall time for the whole file (imports + 4 candidates): 0.23 s. So the ladder below is fully observable and nearly free.

**Proposed ladder for a candidate `Prune P` (ours, ordered by how much of the obligation is discharged):**

| Level | Condition (from `lean --json`) | Meaning |
|---|---|---|
| L0 | parse/syntax error, or `kill` fails to elaborate (`synthInstanceFailed`, `unknownIdentifier` located inside `kill`) | nothing usable |
| L1 | `kill` elaborates; `sound` has an error at position *before* any tactic progress (error on line of `intro`) | predicate usable, no proof |
| L2 | `sound` has `Tactic.unsolvedGoals` / errors only *after* ≥1 successful tactic; remaining goal(s) reported | partial proof |
| L3 | only `hasSorry` warnings (type-checks-but-sorry), possibly with `have ... := by sorry` sub-lemmas | proof sketch |
| L4 | no errors, no `sorry`, but `#print axioms` shows anything beyond `propext`, `Quot.sound` (e.g. `sorryAx`, `Classical.choice`, `Lean.ofReduceBool` from `native_decide`) | reject (README bans `native_decide`) |
| L5 | elaborates clean; axioms ⊆ {propext, Quot.sound} | **verified** (the only level at which kill-mass counts fully) |

Numeric mapping following CodeRL/RLTF spacing: L0 = 0.0, L1 = 0.15, L2 = 0.3, L3 = 0.45, L5 = 1.0 on the *status* component,
and the status component is combined with kill-mass as in Sec. 7. (Kim et al.'s d1/d2 and CodeRL's −1/−0.6/−0.3 spacings are
"fixed scores ... somewhat sensitive across different models and datasets" — S15's own limitations statement — so treat these
as starting points to ablate, not truths.)

**Gaming risk.** The ladder alone rewards `kill := fun _ => false` with a one-line `sound` (L5). That is why status is
never the whole reward: it is multiplied/gated with *marginal kill-mass* (Sec. 5) and the empirical-soundness gate (Sec. 2.8).

### 2.2 Error-position depth / first-error credit

**Evidence.** DeepSeek-Prover-V1.5 (S1), truncate-and-resume: "we truncate the proof at the earliest verification error,
ensuring that all subsequent tactic codes can be successfully applied"; "we insert the tactic state returned by the
verifier as a comment ... During training, we use all tokens following '/- tactic state: ' as responses". This is a
*data* use of the first error (salvage the prefix), not a reward.

Kim et al. (S15) turn the same observation into a reward: "We view a proof Y as a sequence of tactics (T₁,…,T_N(Y))
parsed from the Abstract Syntax Tree"; "If a tactic does not appear in the error log, then it has been elaborated
successfully and passed Lean's internal rule-based verification"; reward per tactic
```
φ(Y, T_k) = 1        if g(Y) = 1 (proof verified)
          = d1       if g(Y) = 0 and k < j and no error   (j = index of first failing tactic)
          = d2       if g(Y) = 0 and k ≥ j
```
with defaults d1 = −0.05, d2 = −0.1; "once an error is observed at T_j, we propagate this failure to all subsequent
tactics"; the process advantage is put "only [on] the first token of each tactic" and added to the outcome advantage:
`A_{i,t} = A_outcome,i,t + 1{t = first(T)} · A_process`. Results (their Table 2): STP + outcome-only GRPO 57.9 ± 0.5 →
outcome+tactic 59.2 ± 0.5 (miniF2F pass@64); ProofNet 17.4 ± 0.6 → 18.6 ± 0.3. Proof length "remains stable", so the gain
is not length gaming. Timeout matters: "A 5s limit gave the worst results ... 10–30s yielded much stronger performance."

RLTF (S23) Eq. 4 is the program-side analogue: the fine-grained penalty is applied only to tokens between
`t_line_start` and `t_line_end` of "the line that compiler feedback" reports; its ablation is the largest single gain.

**Implementation on ZarPrune.** The first error's `pos.line` relative to the `sound := by` line, normalized by the number
of tactic lines, is "depth". Because evolutionary search does not do token-level credit assignment, the usable form is
the *ladder position* (L1 vs L2) plus, optionally, `depth ∈ [0,1]` as a tie-breaker within L2. Do not make depth a large
term: a proof can be long and wrong.

**Gaming risk.** Padding the proof with no-op tactics (`skip`, `simp only []`) before the failure raises depth. Mitigate:
count only tactics that change the goal (the REPL's `goals` before/after) or cap depth's weight at ~0.05.

### 2.3 Goals closed / unsolved-goal count

**Evidence.** AlphaProof (S17): "The agent is incentivized to find short proofs by a reward signal r_t = −1 for each
tactic applied"; for multi-goal states "the return equals the minimum return across subgoals" (so progress on one subgoal
is not credited until all are closed — a deliberately *conservative* aggregation). HTPS (A2, abstract only) and
InternLM2.5-StepProver (A3, abstract only: critic "boosts the performance of the prover model (59.4% to 65.9%)") train a
critic on goal provability but those numbers are not about partial credit per se.

What the tool exposes: the REPL (S36) returns `sorries: [{pos, endPos, goal, proofState}]` and, in tactic mode,
`goals: [...]` after each tactic; `lean --json` prints the unsolved-goal list inside `Tactic.unsolvedGoals`.

**Implementation.** For L2/L3 candidates record `n_open` = number of remaining goals (from the unsolved-goals message or
`sorries`), and the *textual size* of each open goal. Use them only to order candidates within a ladder level and to build
the LLM feedback prompt (OpenEvolve renders evaluator artifacts into the next prompt: `prompt/sampler.py:_render_artifacts`
truncates at `max_artifact_bytes`). The goal text is the single most useful artifact to feed back.

### 2.4 Type-checks-but-`sorry` (sketch credit) and sorry-decomposition

**Evidence.** Draft-Sketch-Prove (S14): sketches with holes filled by "Sledgehammer + heuristics" ("tries 11 common
tactics"; 10 s per conjecture) raise a prover from "20.9% to 39.3%". DeepSeek-Prover-V2 (S2): "The resulting chain of
thought culminates in a Lean theorem composed of a sequence of have statements, each concluded with a sorry placeholder
indicating a subgoal to be solved"; "We extract subgoal expressions from have statements to substitute them for the
original goals ... and then incorporate the preceding subgoals as premises"; RL adds "a consistency reward in the early
steps of training, which penalizes the structural misalignment, explicitly enforcing the inclusion of all decomposed
have-structured lemmas in the final proof". APOLLO (S5): the "Sorrifier" does "Line removal ... Block removal ... Insert
sorry, if the block compiles but leaves unsolved goals open"; "At that point, every remaining sorry marks a sub-lemma to
be proved in later stages"; Auto Solver "first invokes the Lean4's hint ... If goals persist, it applies built-in solvers
(nlinarith, ring, simp, etc.) wrapped in try"; results: Goedel-SFT 25,600 samples → 362 avg samples at 65.6%; Kimina-7B
1,024 → 307 samples at 75.0%; Goedel-V2 63 samples → 84.9%. ProofAug (S4): "a semi-proof yy is compatible to yf0 if each
sorry in yy matches a proof…qed block or a by clause in yf0"; "we recursively resort to a more coarse semi-proof whenever
ATPs fail to fill in some gap"; Table 2: 36.5% @1, 44.7% @10, 52.5% @100, 66.0% @2100 queries (curated miniF2F, Isabelle).
Seed-Prover (S9, read directly): lemma style "allows clear identification of the lemmas that have been successfully
proved, and those that need further refinement"; "we establish a lemma pool for each difficult problem, which stores ...
lemma statements, lemma names, complete proofs, proof difficulties, and dependency relations"; in the heavy setting "Each
lemma is scored based on its proof rate, semantic relevance, and proof length ... lemmas with low proof rate are often
crucial to the final proof."

**Implementation.** A `sorry`-level candidate is a `Prune P` whose `sound` field elaborates with `hasSorry` warnings. The
harness must never *use* such a prune (L3 < L5), but it can (a) store the candidate in the archive at L3, (b) enumerate
its `have` sub-lemmas as separate evolution targets (DeepSeek-V2's subgoal curriculum), (c) auto-fill holes with the
Mathlib-free automation that exists in core (`decide`, `omega`, `simp`, `simp_all`, `exact?`, `apply?`) in an APOLLO-style
pass before scoring — cheap given 0.23 s elaboration. The consistency-reward idea transfers: when a sketch's sub-lemmas
are later proved separately, credit the assembled proof only if the `have` structure survives (otherwise the LLM learns to
delete hard `have`s).

### 2.5 Lemma-library coverage

**Evidence.** LEGO-Prover (S13): three vector stores (lemma / request / problem); "22532 skills in total" of which 10.8%
from the prover, 38.2% from the evolver's request solver, 51.1% from directional transformation; "24% [of solved problems]
is completed with the aid of retrieved skills. Within this subset, 51% ... directly incorporate the retrieved skills";
miniF2F 57.0 / 50.0 (valid/test), ablation 47.1% → 50.4%. Seed-Prover's lemma-pool scoring (above). LeanAgent (S34)
curriculum: "we calculate the complexity of each theorem using e^S, where S represents the number of proof steps", split
at the 33rd/67th percentiles into easy/medium/hard.

**Implementation.** ZarPrune already composes prunes (`Prune.or`, `Prune.ofList`). Keep a *lemma ledger*: every verified
auxiliary lemma (not only prunes) becomes a retrievable, importable declaration; give a small reward for a verified lemma
that is *used by a later verified prune* (LEGO-Prover's usage statistic is the right metric: fraction of accepted prunes
that cite ledger lemmas). Do not reward lemma count per se (trivially gamed by `theorem foo : True := trivial`).

### 2.6 Autoformalization agreement (NL statement vs Lean statement)

**Evidence and numbers.**
- Lean Workbook (S11): compile check, then back-translation and an NLI-style judge ("Please check following two math
  problems is same or different?"), then human fixing; sampled faithfulness "0.935" weighted average, stopping "when the
  correct rate in sampled data almost reaches 95%".
- DeepSeek-Prover V1 (S3): five-grade model scoring ("excellent/good/above average/fair/poor", drop fair/poor) and
  **hypothesis rejection**: "using the DeepSeek-Prover model to attempt proving the formal statement with 'False' as the
  conclusion. A successful proof indicates an invalid hypothesis, prompting exclusion"; dual search "Γ⊢P and ... Γ⊢¬P";
  869,659 NL problems → 712,073 statements; high-score vs low-score training data: 42.6% vs 38.1% miniF2F (pass@128).
- FormalMATH (S12): multi-LLM back-translation consensus and **negation-based disproof** ("automated proof attempts on
  ¬T within ... Lean4"); compile validation → semantic verification retained 92.4% → 32.7% (i.e., 60.7% of compiling
  statements were semantically wrong), disproof removed a further 1.6%, and 72.09% of the survivors passed human review;
  consensus is far lower on undergraduate material ("4.63% on HardMath"); annotation cost "$6.89 per statement".
- Kimina (S10): "We employ DeepSeek-Prover's negation-proving to identify and remove potentially erroneous
  formalizations"; post hoc "a judge model to assess whether the proofs generated by the model are correct or if the model
  has merely leveraged a mistake in the formalization".
- FormalAlign (S6): V_cer = exp(mean log P(FL_j | FL_<j, NL)), V_sim = cos(Z(NL), Z(FL|NL)), V_align = (V_cer + V_sim)/2;
  six synthetic misalignment generators (constant / exponent / new variable / variable type / = vs ≠ / random pairing);
  Alignment-Selection 99.21% vs GPT-4 88.91% (FormL4-Basic), 66.39% vs 64.34% (miniF2F-valid); human 79.58% vs model
  65.00% on the hard set.
- BEq (S18, read directly): Unidirectional Definitional Implication `sP ←U sQ ⟺ sP ∼D T(sQ | sP, R)` and
  `sP ∼B sQ ⟺ sP ←U sQ ∧ sQ ←U sP`, where T is "approximated by sampling tactic sequences from a large language model and
  symbolically executing on Lean kernel"; R = {apply, cases', constructor, exact, exact?, ext, have, intro, intros, rw,
  use}; on 200 expert-labelled pairs "BEq reaches 100.0% precision and 90.50% accuracy ... However, BEq falls short on
  recall". This is the right shape for "does the LLM's *restated* obligation match `Prune.sound`?" checks.
- Beyond Compilation (S19): `Faithful(x,y) = Compiles(y) ∧ grade_GPT(x,y) ≥ 9 ∧ grade_Gemini(x,y) ≥ 9`; "Every system has
  a nonzero compile–faithfulness gap, whose observed magnitude ranges from 3.0 to 29.0 percentage points" (full agent:
  89.5% compile, 60.5% faithful); "LLM judging is therefore useful as a human-calibrated, conservative aggregate measure,
  not as an equivalence oracle."
- Roundtrip (S20): formalize → informalize → re-formalize and check `y^orig ≡ y^rt` with an SMT solver; no guarantee
  ("the pipeline may stabilize at a semantically different fixed point"); but "SAT rules drift 1.4×–2.5× more often than
  UNSAT rules", i.e. formal equivalence predicts fidelity.
- AlphaProof (S17): formalizations were verified "against golden Lean statements via type equality checking" during
  autoformalizer training; ~80M formal problems from ~1M NL sources.

**Why our design mostly side-steps this, and what remains.** In ZarPrune the LLM never writes the theorem statement: it
fills `kill` and `sound` inside a fixed `structure Prune (P : Params)`, and `Valid`, `HasKst`, `profileOf` are fixed
definitions (`Basic.lean`). So "NL ↔ Lean statement" agreement is by construction *if* the harness only ever accepts
terms of type `Prune P` for the harness's own `P` — the checker must refuse candidates that redefine `Valid`/`HasKst`/
`Params` in their file (a Kimina/Vericoding-style "leveraged a mistake in the formalization" exploit). What does remain:

1. **Vacuity** (analogue of hypothesis rejection): `kill` never fires. Test: run `kill` over the enumerated cases; if
   0 fire, reward 0 regardless of Lean status. Cheap, exact.
2. **Triviality**: everything it kills, `baseline` already kills. Test: marginal mass over `Prune.ofList accepted`.
3. **Agreement between the LLM's NL argument and its Lean code**: only matters for interpretability, not soundness. A
   BEq-style check is overkill; a judge score can be logged as an artifact, never as reward (S19's warning).
4. **The one place NL↔Lean agreement is load-bearing**: if we ever accept an *encoding-correctness* theorem or a
   *cover-completeness* certificate written by the LLM (the `refuted` / `cover` seams in `upper_bound_of_cover`). Those
   statements are also fixed by the harness; the LLM must never author them.

### 2.7 "Proof of a weaker statement" credit

**Evidence.** DeepSeek-V2 (S2) trains on subgoals "with/without preceding subgoals as premises" and curates "problems that
remain unsolved by the 7B prover model in an end-to-end manner, but for which all decomposed subgoals have been
successfully resolved". Goedel-Prover-V2 (S8), scaffolded data synthesis: "When the prover fails to find a proof for a
challenging problem, we can still leverage the proof attempt to generate simpler, related problems" — formal side via
`extract_goal` "to capture the unsolved states of a proof", informal side by prompting "to generate simpler/sub-problems
if it is unsolved, or harder variants if it is already solved"; training only on pass rate "(0, 0.75]". AlphaProof (S17)
TTRL: "a bespoke curriculum of synthetic problem variants (for example, simplifications or generalizations) generated
specifically around the target problem". STP (S7): conjectures kept only if the seed "lemma l_i is used in the proof",
"the pass rate of the conjecture ... is between (0, 1/4]", and after removing the 20% with the smallest
proof-length/statement-length ratio; "at least 47% of the generated conjectures in STP training are successfully proved".
Seed-Prover (S9): "For problems that are too difficult for single-pass generation, we use our proposer to generate easier
problem variants and put these into the training dataset."

**What "weaker statement" means for a prune — three exact forms, all sound by construction:**

(W1) **Domain restriction.** `kill' pf := S pf && kill pf` for a decidable profile predicate `S` (e.g. "all row sums
equal", "max column sum ≤ c", "s = t = 2 instance"). `sound` only needs the argument on profiles satisfying `S`. This is
DeepSeek-V2's "subgoal with premises". Credit = mass of cases with `S ∧ kill`. Lean: `Prune.restrict` is a 4-line lemma
(see Sec. 8).

(W2) **Strengthening the predicate.** If `p : Prune P` and `∀ pf, kill' pf = true → p.kill pf = true`, then `kill'` is a
prune. Useful when the LLM proves a *stronger* inequality than needed (kills fewer cases) — still credit for mass.
Lean: `Prune.mono`, 3 lines.

(W3) **Smaller instance.** A prune proven for `P' = ⟨m', n', s, t, w'⟩` with `m' ≤ m, n' ≤ n`. It is *not* automatically a
prune for `P`; transferring it needs Tan's Argument I (every m'×n' minor of a witness is admissible), i.e. a proved bound
for the smaller instance plus a restriction lemma. Credit it as a **ledger lemma** (2.5) and as a curriculum step, not as
kill-mass on `P`.

Anti-pattern (proved unsound in `Demo.notDescending_unsound`): "assume rows sorted" is not a weaker statement, it is an
*addition*; it will fail to elaborate and must get L0/L1, never partial credit.

### 2.8 Empirical soundness pre-check (our design; program-RL analogue)

Not in the theorem-proving literature (statements there have no cheap semantic test) but standard in program RL
(CodeRL/RLTF unit tests) and in FunSearch/AlphaEvolve ("ensembles of test cases of increasing difficulty" — S26). For
a prune we have a *free* falsification test: the lower-bound certificates from the prior work (Bhan et al. 2026) and any
K_{s,t}-free matrix (random greedy constructions) are valid matrices; if `kill (profileOf A) = true` for a known
K_{s,t}-free `A` with weight ≥ w', the candidate is unsound *for w ≤ w'* and no Lean attempt is needed. Concretely:
`kill` elaborates → evaluate it (via `#eval`, or compile the same predicate in Python from the enumerated profiles) on
(a) profiles of all known valid matrices for nearby (m,n,s,t,w), (b) 10^3–10^4 random K_{s,t}-free matrices. This is
stage 1 of the OpenEvolve cascade (`cascade_thresholds` default [0.5, 0.75, 0.9]); Lean is stage 2; kill-mass on the
full enumeration is stage 3. The test is one-sided: passing it proves nothing (which is why Lean exists), failing it is
a certificate of unsoundness and should score 0 with the counterexample fed back as an artifact.

---

## 3. Binary reward + difficulty-banded curriculum: what every prover does

| System | Reward | Which problems get trained/attempted |
|---|---|---|
| DeepSeek-Prover-V1.5 (S1) | "a reward of 1 if verified as correct, and 0 otherwise" (GRPO) | "we select training prompts that are challenging yet achievable for the supervised fine-tuned model" |
| DeepSeek-Prover-V2 (S2) | binary + early "consistency reward" | subgoal curriculum (two variants) |
| Kimina-Prover (S10) | binary, format filter | "we prune problems where the model consistently demonstrates high proficiency"; initial set built with "an evenly distributed difficulty spectrum" rated by QwQ-32B |
| Goedel-Prover-V2 (S8) | GRPO (reward not spelled out in main text) | "only include problems with pass rates in the range (0, 0.75]"; scaffolded easier/harder variants |
| Seed-Prover (S9) | 1/0 + formatting penalty | "exclude problems which are too easy (i.e. proof rate above 1/4)"; "problem difficulty, problem quality, and maximum output length are progressively increased" |
| STP (S7) | verified proofs only, "empirical pass rate ... below 1/2" | conjecturer trained on conjectures with pass rate in (0, 1/4] |
| Leanabell-V2 (S16) | R_success/R_failed = 1.0, R_format = 0.2 | multi-turn with `<interpreter>` feedback |
| AlphaProof (S17) | r_t = −1 per tactic; min over subgoals | "highly interesting if: (1) it has not been attempted previously; (2) it has been attempted fewer than a trust_count threshold; or (3) ... shows a mixed success rate"; budgets "scale multiplicatively with recent failure counts" |
| Kim et al. (S15) | binary + tactic-level d1 = −0.05 / d2 = −0.1 | standard |
| PLR (S33, RL not TP) | — | level score `S_i = (1/T) Σ_t |Σ_k (γλ)^{k−t} δ_k|`, `P_S ∝ (1/rank)^{1/β}`, staleness `P_C ∝ (c − C_i)`, `P_replay = (1−ρ)P_S + ρP_C`; "induces an implicit curriculum ... progressively harder levels" |
| LeanAgent (S34) | — | complexity e^S, 33rd/67th percentiles; repos sorted by number of easy theorems |

Two lessons for us. (i) **Reward = verified, curriculum = band.** Instead of shaping the reward, shape *what is asked*:
choose the target instance `P` and the restriction predicate `S` (W1) so that the current population's verified rate on
the sub-task sits in a band like (0, 1/2]. (ii) **Mixed success rate is the sweet spot** (AlphaProof, STP, Goedel): a
sub-task with 0/k successes is a signal to decompose (2.4), not to keep sampling.

---

## 4. Exploration vs exploitation in MAP-Elites for LLM program search

### 4.1 Canon (read directly)

- MAP-Elites (S29 Sec. II-D describing N2): grid over descriptor space; "a solution is randomly selected via a uniform
  distribution among those in the grid, and is mutated"; "If the cell is empty, then the solution is added to the grid,
  otherwise, only the best solution among the new one and the one already in the grid is kept."
- Novelty score (S29 Sec. II-B describing N1): "the average distance of the k-nearest neighboring solutions that currently
  are in the novelty archive ... When the novelty score of a solution exceeds a pre-defined threshold, this solution is
  added to the archive."
- Cully & Demiris (S29) Definition III.1, **Curiosity Score**: "the propensity of an individual to generate offspring that
  are added to the collection"; Algorithm 1: `curiosity(parent) += Reward` (1) when the offspring enters the container,
  `−= Penalty` (0.5) otherwise; "the algorithm focuses on regions of the search space as long as they produce interesting
  results, then, when the algorithm gets 'bored', it focuses its attention on different regions." Uniform selection's
  weakness: "the selection pressure decreases as the number of solutions in the collection increases". Their conclusion
  (Sec. IV): random, fitness and curiosity selectors all reach the ideal collection on the arm task; population-based
  selectors converge and fail. Also: "erasing the entire collection except 10 solutions every 100 000 generations
  increases the number of filled cells by 20% and the average quality of the solutions by 50% in some experimental setups"
  (citing Lehman et al.).
- CMA-ME (S31) improvement emitter, Algorithm 2: Δ_i = fitness if the cell is empty ("Flag that x_i discovered a new
  cell"), Δ_i = fitness − M[β_i].fitness if it improves an elite; parents sorted by (newCell, Δ_i) — **discovering a new
  cell outranks any improvement**; if no archive change after λ samples, "Restart from random elite in M". Tables 1–2:
  on the 20-D sphere, MAP-Elites 56.22% cells / QD 11.4M vs CMA-ME(imp) 87.75% / 16.9M.
- Go-Explore (S30) Appendix A.5: `CntScore(c,a) = w_a · (1/(v(c,a)+ε1))^{p_a} + ε2` with ε1 = 0.001, ε2 = 0.00001 and
  "p_a ... turned out to be 0.5 for all attributes"; attributes = times chosen, times chosen since new, times seen;
  `NeighScore(c,n) = w_n · (1 − HasNeighbor(c,n))`; `CellScore = LevelWeight · (Σ NeighScore + Σ CntScore + 1)`;
  `CellProb = CellScore / Σ CellScore`; counters reset when a cell's trajectory improves "because a new way of reaching a
  cell may actually be a more promising stepping stone".

### 4.2 LLM-era program search (full text unless marked)

- ELM (S25): Sodarace archive "12×12×12 grid" over (height, width, mass); "an inhabited niche is randomly chosen and the
  solution within that niche is perturbed by the diff model"; runs fill ~80–95% of niches; QD score = "sum of the
  performance of all champions in the final map".
- QDAIF (S24): LLM-scored quality and LLM-classified diversity attributes; "custom non-uniform bins, which are denser
  towards range ends" because "qualitative changes in behavior do not uniformly correspond to changes in the logits";
  failure modes: "we suspect reward hacking happening when using LMs to generate feedback", correlation "drops when the
  evaluated quality is in the range 0.995 to 1"; some bins never found ("limericks"); "it still requires specified
  definitions of diversity axes".
- FunSearch database (S28, code): programs clustered by `Signature = tuple(scores_per_test[k] for k in sorted(...))`;
  cluster sampling `probabilities = _softmax(cluster_scores, temperature)` with
  `temperature = T_init * (1 − (num_programs % period)/period)`; within-cluster `_softmax(−normalized_lengths, 1.0)`
  (shorter preferred); "Resets the weaker half of islands" every `reset_period`, reseeding from the best islands.
- AlphaEvolve (S26): "inspired by a combination of the MAP elites algorithm and island-based population models";
  multi-metric: "even if one metric is of particular interest, optimizing for multiple metrics often improves results for
  the single target metric"; cascade "ensembles of test cases of increasing difficulty"; Flash for throughput, Pro for
  "occasional, higher-quality suggestions"; meta-prompt evolution.
- ShinkaEvolve (S27): parent sampling `p_i = r_i^(−α) / Σ_j r_j^(−α)` (power-law over rank) or
  `w_i = s_i · h_i`, `s_i = σ(λ(F(P_i) − α_0))`, `h_i = 1/(1 + N(P_i))` (usage-penalized); **novelty rejection**:
  embedding similarity > η = 0.95 triggers "an LLM to further assess whether the program is meaningfully different";
  LLM-choice bandit via UCB1 on `r_i^u = exp(max(r_i − r_i^b, 0)) − 1` with baseline = max(parent, initial); circle
  packing SOTA "using only 150 samples"; ablations: "Weighted sampling consistently outperforms both random search and
  hill climbing", "Code embedding-based rejection sampling provides substantial performance gains".
- FormalEvolve (S21): candidates `(header, Lean statement)`; archive admits only `Comp(c) = 1`; score
  `s_i = Comp(c_i)(1 + Sem(c_i))` (no partial credit); parent weight `w_i = σ(λ z_i) u_i` with usage discount
  `u_i = [1 + (1+β) n_i]^(−1)`; operators: full patch / diff patch / cross patch (+inspirations) / bounded repair (≤2) /
  AST rewrites; diversity = `|Dedup(G_t)|` after "conservative whitespace/name canonicalization"; concentration via a Gini
  over per-problem successes; SH@100 58.0% CombiBench / 84.9% ProofNet.
- ACES (A4, abstract): LLM-labelled skill descriptors as archive axes; difficulty "a linearly decreasing function of the
  success rate of Llama-3-70B".
- DEI (A5, abstract): heterogeneous LLM mutators: "124 percent higher merged-archive QD-Score (45.90 vs. 20.46) and 28
  percent higher coverage"; "model diversity, not merely parallelism, is the key driver".

### 4.3 What OpenEvolve already implements (read from `openevolve/database.py`, `config.py`)

- `DatabaseConfig`: `population_size 1000`, `archive_size 100`, `num_islands 5`, `elite_selection_ratio 0.1`,
  `exploration_ratio 0.2`, `exploitation_ratio 0.7` (remaining 0.1 = random), `feature_dimensions` default
  `["complexity", "diversity"]` with custom metric names allowed ("Evaluators must return raw continuous values ... NOT
  pre-computed bin indices"), `feature_bins 10` (int or per-dim dict), `migration_interval 50`, `migration_rate 0.1`,
  `novelty_llm`, `embedding_model`, `similarity_threshold 0.99`.
- `_sample_parent`: exploration = `random.choice` over the current island (uniform, MAP-Elites style);
  exploitation = `random.choice` over the archive, preferring archive members on the current island; no count/curiosity
  weighting; `_sample_inspirations` = island best + top-`n·elite_selection_ratio` + programs from *nearby feature cells*.
- `_calculate_feature_coords`: custom metric → min-max scaled → `int(scaled * num_bins)`; built-ins `complexity`
  (`len(code)`), `diversity` (edit distance to a reference set), `score`.
- Evaluator cascade: `cascade_evaluation: True`, `cascade_thresholds [0.5, 0.75, 0.9]`; artifacts rendered into the next
  prompt (`_render_artifacts`, truncated to `max_artifact_bytes`).

Gaps relative to the literature: no curiosity/count-based cell selection (S29, S30), no "new cell beats improvement"
ranking (S31), no island reset (S28), no signature clustering (S28), uniform-not-power-law exploitation (S27). All are
small patches to `_sample_parent` / `add`.

### 4.4 Feature dimensions for prune search (proposal, C-tier evidence; rationale from S28/S24/S25)

The phenotype of a prune is its **kill set** on the enumerated cases. Use it the way FunSearch uses the score signature:

- `kill_signature` = bit-vector of `kill` over a fixed, stratified sample of ~2,000 cases (cheap to compute once `kill`
  elaborates or once the predicate is mirrored in Python). Behavior distance = Jaccard distance between kill sets.
  Novelty rejection (S27): if Jaccard similarity to an archived prune > 0.95 and the Lean status is not higher, reject
  before spending an LLM call on it. Cluster archive entries by signature (S28) so exploitation samples *distinct kills*.
- Grid axis 1: `kill_mass_marginal` (Sec. 5), log-binned, denser near 0 (QDAIF's non-uniform bins: most candidates kill
  nothing new).
- Grid axis 2: `lean_status` (L0–L5, 6 bins). Keeping L2/L3 elites alive is the whole point: they are stepping stones
  (S29 "stepping stones", S31 restart-from-elite) and the DeepSeek-V2 subgoal source.
- Optional axis 3: `argument_kind` — LLM-labelled category (counting / parity / pigeonhole / minor-monotonicity /
  structural), ACES-style. Cheap, interpretable, but subject to QDAIF's bin-misclassification failure; log it, and ablate
  whether it helps coverage before making it a grid axis.
- Selection: replace uniform island sampling with Go-Explore counts (`(1/(chosen+ε))^0.5`, reset on improvement) or
  Cully's curiosity (+1/−0.5). Rank archive insertions CMA-ME style (new cell > improved elite). Reset the weaker half of
  islands every K generations (S28).
- Multi-LLM: keep OpenEvolve's ensemble; add ShinkaEvolve's UCB1 over models with reward = improvement over parent.

---

## 5. "Difficulty" of a SAT branch — the weight in kill-mass

Tan (S35, read directly) decomposes by "all possible combinations of unordered row and column partitions not forbidden
by the arguments of section 2 – an approach very much like Heule's cube-and-conquer paradigm"; Algorithm 1 enumerates
admissible column partitions (Theorem 3.1: "all admissible partitions ... in lexicographic order ... parts in
non-increasing order") with pruning at line 4 by Arguments A and I; Argument D then removes partition *pairs*
("considering the row with the most, r, ones and the r columns with the least ones"). His conclusion: "The CNF instances
... did not split the problem into cases finer than specific combinations of row and column partitions. Combined with
the use of just one processor at a time, this imposed a limit ... at around z = 100."

Reward for a prune should be Σ_{cases killed, not already killed} cost(case). Ways to estimate cost(case), from
strongest evidence to weakest:

1. **Censored solver probe** (standard in cube-and-conquer practice; the analogue of PLR's "learning potential"): run
   the case's CNF with a conflict budget B (e.g. 10^4–10^5 conflicts, Kissat/CaDiCaL); record `(solved?, conflicts,
   time)`. Cost = time if solved, else a right-censored value ≥ time(B). Fit a per-instance monotone map from cheap
   features to cost using the solved cases. Exact on the cases that matter least, honest about the ones that matter most.
2. **Tree-size estimators** (S32): Knuth's `N = 1 + b_1 + b_1 b_2 + …` from random probes; Kilby's *weighted backtrack
   estimator* `Σ_{d∈D} prob(d)(2^{d+1} − 1) / Σ prob(d)` with `prob(d) = 2^{−d}` over branch depths seen so far, and the
   *recursive estimator* (assume the unexplored right subtree looks like the explored left one). On UNSAT random 3-SAT
   all three predict the midpoint at 0.49–0.53; on structured UNSAT (hole-8, BF) they under-estimate early. Useful as a
   *ranking* of cases by expected work from a short prefix of search — exactly our situation (UNSAT-heavy).
3. **Propagation rate under the cube** (AlphaMapleSAT, see `docs/lit/alphamaplesat.md`): ratio of BCP-implied literals
   to cube size. Cheap (one `propagate` call), correlates with hard-tail avoidance in their KS/Ramsey experiments.
4. **Static slack**: for each case, `(t−1)·C(m,s) − Σ_j C(c_j, s)` (Argument A slack) and `w − baseline bound`; near-zero
   slack cases are the extremal ones (Tan's tables show hard instances cluster at the boundary). Free.
5. **Instance size after fixing sums**: variables/clauses remaining after unit propagation of the cardinality
   constraints (Tan's sequential-counter encoding propagates fully). Free, weak.
6. **Nearest-neighbour history**: cost of previously solved cases with similar profile (SATzilla-style runtime
   prediction, not read for this note). Grows with the run.

Choose (1) as ground truth on a subsample, (2)/(3)/(4) as features; report kill-mass in *estimated CPU-seconds*, not
case counts, and also report the count so that a prune killing 10,000 trivially-UNSAT cases is not mistaken for one
killing 50 hard ones.

---

## 6. Concrete proposals, ranked by evidence

**Tier A — direct evidence from multiple sources; implement first.**

A1. *Binary verification is the reward; difficulty band is the curriculum.* Score = 0 unless `Prune P` elaborates clean
    with axioms ⊆ {propext, Quot.sound}; then score = marginal kill-mass. Choose the sub-task (instance P, restriction S)
    so the population's verified rate stays in (0, 1/2]. (S1, S7, S8, S9, S10, S17.)
A2. *Status ladder as a secondary metric and as a MAP-Elites axis*, not as the main reward. Tiers spaced like
    CodeRL/RLTF; L5 gets the mass; L2/L3 are archived as stepping stones. (S22, S23, S16; local probe shows it is free.)
A3. *Feed the first error + remaining goal text back into the prompt* (truncate-and-resume as data; verifier-guided
    self-correction). Goedel-V2: "Removing compiler feedback significantly lowers performance, confirming that specific
    error messages are crucial." Two repair rounds max (S8: "a maximum of 2 rounds"). (S1, S8, S16, S5.)
A4. *Sorry-decomposition of failed proofs*: parse `have` blocks, keep verified ones, `sorry` the rest, auto-fill with
    core tactics (`decide`, `omega`, `simp_all`, `exact?`), spawn each remaining hole as its own evolution target with the
    preceding `have`s as premises. (S2, S4, S5, S9, S14.)
A5. *Vacuity and triviality checks* before Lean: kill fires on ≥1 case; marginal mass over accepted library > 0; never
    accept a file that redefines `Valid`/`HasKst`/`Params`. (S3, S10, S12 negation/hypothesis rejection; S19 compile ≠ faithful.)
A6. *Novelty rejection and phenotype clustering*: Jaccard on kill signature, reject > 0.95 unless status improves;
    cluster archive by signature. (S27, S28.)

**Tier B — single strong source or strong analogy; implement second, ablate.**

B1. *First-error depth as a small dense term* (d1 = −0.05 / d2 = −0.1 spacing; +1–2.5 pp in S15). In evolution this
    becomes a within-tier tie-breaker; cap at 5% of the reward.
B2. *Count/curiosity-based cell selection* replacing uniform island sampling (S29 curiosity +1/−0.5; S30 counts^−0.5);
    CMA-ME insertion ranking (new cell > improved elite) (S31); reset weaker half of islands (S28). Expected effect:
    coverage of the (mass × status) grid, fewer wasted LLM calls on saturated cells.
B3. *Weaker-statement credit via `Prune.restrict`/`Prune.mono`* (W1/W2): mass of the restricted domain counts at L5.
    Evidence is by analogy to S2/S8 subgoal curricula; the Lean side is trivial.
B4. *Lemma ledger with usage-based credit* (S13: 24% of solutions used retrieved skills; S9 lemma-pool scoring by proof
    rate and relevance).
B5. *Empirical soundness gate on known valid matrices* (Sec. 2.8). Analogy to unit tests (S22/S23) and AlphaEvolve's
    cascade (S26); no direct TP evidence because TP has no such oracle. Very cheap; adopt.
B6. *Difficulty-weighted mass via censored solver probes + Kilby estimators* (Sec. 5; S32, AlphaMapleSAT).

**Tier C — plausible, weak or indirect evidence; try only after A/B are stable.**

C1. LLM-labelled `argument_kind` as a grid axis (S24, A4) — watch for bin misclassification and reward hacking.
C2. UCB1 model-choice bandit (S27) — depends on having ≥2 models with different costs.
C3. LLM-judge "NL argument matches Lean proof" score as an *artifact* for the thesis's interpretability goal — never as
    reward (S19, S24 reward-hacking evidence).
C4. Consistency reward for preserving `have` structure across repair rounds (S2, early-training only).
C5. Proof-length preference within a cell (FunSearch shorter-programs softmax; AlphaProof −1/tactic) — cheap, mild.

**Anti-proposals (evidence says no).**
- Do not give partial credit for `sorry` proofs that kill mass (the harness would be tempted to use them; Vericoding-style
  spec gaming). L3 is archive-only.
- Do not use LLM judges as the quality signal (S24: correlation collapses near 1.0; S19: 3–29 pp compile-faithfulness gap).
- Do not reward code length/complexity or lemma count.

---

## 7. A reward function that follows from the above (for the evaluator; to be ablated)

```
status  ∈ {L0..L5}                      from lean --json kinds + #print axioms
fires   = #cases in sample with kill = true                       (vacuity)
m_new   = Σ_{c killed by kill, not killed by accepted library} ĉost(c)   (marginal, difficulty-weighted)
m_tot   = Σ_{c not killed by accepted library} ĉost(c)
unsound = ∃ known valid A : kill (profileOf A) = true             (empirical gate)

combined_score =
   0                                       if unsound or fires = 0 or status ∈ {L0, L4}
   0.15·ladder(status) + 0.05·depth        if status ∈ {L1, L2, L3}      (ladder(L1)=1/3, L2=2/3, L3=1)
   0.20 + 0.80 · m_new / m_tot             if status = L5

metrics returned to OpenEvolve: combined_score, lean_status (0..5), kill_mass_marginal (raw), kill_count,
                                depth, n_open_goals, signature_hash; artifacts: first error, open goals, counterexample.
feature_dimensions: ["kill_mass_marginal", "lean_status"]  (feature_bins: {"kill_mass_marginal": 8, "lean_status": 6})
cascade: stage1 = parse + kill mirror + vacuity + empirical gate; stage2 = Lean; stage3 = full-enumeration mass + probes.
```
The 0.20 floor at L5 keeps a verified-but-tiny prune above every unverified one (A1). The L1–L3 band tops out at 0.20 so
no unverified candidate ever outranks a verified one. The 0.05 depth term is B1's cap.

---

## 8. Pruning-lemma inventory (statements, hypotheses, Lean provability on ZarPrune)

Notation: `P : Params` with `m n s t w`; `pf : Profile m n`; `A : Mat m n`; `HasKst` uses strictly increasing index
tuples `R : Fin s → Fin m`, `C : Fin t → Fin n` (`Incr`). All sums are `sumFin`. Mathlib-free: no `Nat.choose`, `Finset`,
`List.sublists` (checked locally).

| Name | Statement | Hypotheses | Lean provability (Mathlib-free) |
|---|---|---|---|
| `Prune.restrict` (W1) | Given `S : Profile m n → Bool` and `kill`, if `∀ A, S (profileOf A) = true → kill (profileOf A) = true → ¬ Valid P A`, then `⟨fun pf => S pf && kill pf, _⟩ : Prune P` | none | trivial: `Bool.and_eq_true` + the hypothesis; ~5 lines |
| `Prune.mono` (W2) | If `p : Prune P` and `∀ pf, kill' pf = true → p.kill pf = true` then `kill'` is a `Prune P` | none | trivial; 3 lines |
| `Prune.and` | `p q : Prune P` ⟹ `fun pf => p.kill pf && q.kill pf` is a prune | none | trivial (subset of `Prune.or`) |
| Argument A / KST-counting (column form) | `Σ_j C(col_j, s) ≤ (t−1)·C(m, s)` for every `K_{s,t}`-free `A`; prune kills profiles violating it | `K_{s,t}`-free (i.e. `¬ HasKst P A`) | **hard**: needs binomial coefficients and double counting of pairs (s-subset of rows ⊂ column support). Hand-roll `choose` by recursion, a bijection between `Incr` tuples `Fin s → Fin m` and "s-subsets", and Fubini over that enumeration. Estimate 300–600 lines. Tan states it (S35 Sec. 2, Argument A: "Σ_i C(c_i, a) ≤ (b−1)·C(m, a)") without proof; the README lists it as the next target. |
| Argument A (row form) | `Σ_i C(row_i, t) ≤ (s−1)·C(n, t)` | as above | same machinery; dual by `sumFin_swap` once the column form is done |
| Argument I / minor monotonicity | For `n' ≤ n`: the `n'` largest column sums add to at most `z(m, n'; s, t)`; kills profiles where a top-`n'` partial sum exceeds a *proved* smaller bound | a proved theorem `∀ B : Mat m n', ¬ HasKst → weight B ≤ z'` for the smaller instance | **medium**: restriction of `A` to an `Incr` column subset preserves `¬ HasKst` (compose `Incr` maps) and its weight equals the partial sum; 100–200 lines. Note the hypothesis is itself an output of `upper_bound_of_cover` on a smaller instance, so this is the lemma that chains instances (W3). |
| Argument D (Tan) | Pigeonhole on the row with the most ones and the columns it meets (exact statement not recovered from the PDF text extraction; transcribe by hand from S35 Sec. 2, "Argument D") | K_{s,t}-free | unknown until transcribed; likely easier than Argument A (no binomials for s = 2) |
| `deficit`, `mismatch`, `rowCap`, `colCap`, `baseline` | already in `Prunes.lean` | — | proved; axioms {propext, Quot.sound} |
| Regular-case restriction (example of W1) | `S pf := allFin m (fun i => pf.row i == r)` for fixed `r`; any counting prune proved only under `S` | — | trivial wrapper on `Prune.restrict` |

Not a prune (negative result already in `Demo.lean`): `notDescending` — sorting/symmetry breaking is an *addition* and needs
a permutation witness in the certificate.

---

## 9. Numbers worth keeping (for the thesis text)

- Lean check cost on ZarPrune: 0.23 s for a 4-candidate file; `lake build` no-op 0.14 s (measured 2026-09-21, Lean 4.34.0).
- Tactic-level shaping gain (S15): +2.5 pp miniF2F (57.9 → 59.2), +1.2 pp ProofNet (17.4 → 18.6); 5 s Lean timeout worst, 10–30 s good.
- Program-RL graded rewards: CodeRL tiers −1/−0.6/−0.3/+1, 2.20% vs 1.62% pass@1; RLTF 1.30 → 1.37 → 1.41 → 1.45 pass@1 (SL → +coarse → +fine → +adaptive).
- Repair budget collapse (S5): 25,600 → 362 samples (Goedel-SFT, 64.7 → 65.6%); 1,024 → 307 (Kimina-7B, 70.8 → 75.0%); 63 samples for 84.9% (Goedel-V2).
- Sketch/hole filling (S14): 20.9% → 39.3%; ProofAug 36.5 @1 → 66.0 @2100 queries.
- Difficulty bands: (0, 1/4] (S7 conjecturer; S9 excludes > 1/4), (0, 0.75] (S8), "below 1/2" (S7 prover data), mixed-success + trust_count (S17).
- Autoformalization: 60.7% of compiling statements semantically wrong (S12); compile–faithfulness gap 3.0–29.0 pp (S19); BEq 100% precision / 90.5% accuracy (S18); FormalAlign 99.21 vs 88.91 AS (S6); Lean Workbook 93.5% sampled faithfulness (S11); DeepSeek-V1 869,659 → 712,073 statements, +4.5 pp from quality filter (S3).
- QD: ELM fills 80–95% of 1,728 niches (S25); CMA-ME(imp) 87.75% cells vs MAP-Elites 56.22% (20-D sphere) (S31); curiosity +1/−0.5 (S29); Go-Explore p_a = 0.5 (S30); ShinkaEvolve η = 0.95, 150 samples (S27); FunSearch resets weaker half of islands (S28); DEI +124% QD-score from model heterogeneity (A5).
- Lemma reuse (S13): 22,532 skills; 24% of solved problems used retrieved skills, 51% of those verbatim.

---

## 10. Open questions

1. Is the L1–L3 band ever *useful* to the search, or only to the LLM prompt? S15 says dense credit helps RL by ~1–2 pp;
   nothing says it helps an evolutionary archive. Ablation: `feature_dimensions` with vs without `lean_status`.
2. Which difficulty estimator ranks Zarankiewicz cases best? Kilby's estimators were validated on random 3-SAT/TSP, not
   on cardinality-heavy cube-and-conquer; the censored-probe ground truth on a few hundred cases will answer this.
3. Argument D's exact statement and whether Tan's Arguments A/I/D already kill nearly everything killable at the profile
   level (if so, evolved profile-level prunes have little marginal mass and the search must move to finer cases —
   Tan's own suggestion: "enumerating all possible ways to assign the first two rows and columns").
4. Profiles vs unordered partitions: the harness enumerates partitions; a profile prune restricts to partitions
   (README "Scope"), but kill-mass must be computed on partition *pairs* with their multiplicities, not on profiles.
5. How much of the counting layer (hand-rolled `choose`, subset enumeration) should be pre-built by hand vs left to the
   LLM? Every prover paper suggests the LLM will not build a 400-line library inside a 1-shot proof; the library is
   infrastructure, and the search should evolve *uses* of it.
6. Does the empirical gate (Sec. 2.8) catch most unsound candidates before Lean? If it catches >90%, Lean calls can be
   reserved for candidates that already pass it, and the L1–L3 partial signals become nearly free of unsound noise.
7. Reward hacking specific to our gate: candidates that redefine `Valid`, shadow `HasKst`, use `native_decide`, or add
   axioms. `#print axioms` and a fixed-import allowlist are necessary; a fuzz test with adversarial candidates should be
   part of the test suite, as `Demo.notDescending_unsound` already is for symmetry breaking.

---

## 11. Bibliography (URLs)

- S1 https://arxiv.org/abs/2408.08152 · S2 https://arxiv.org/abs/2504.21801 · S3 https://arxiv.org/abs/2405.14333
- S4 https://arxiv.org/abs/2501.18310 · S5 https://arxiv.org/abs/2505.05758 · S6 https://arxiv.org/abs/2410.10135
- S7 https://arxiv.org/abs/2502.00212 · S8 https://arxiv.org/abs/2508.03613 · S9 https://arxiv.org/abs/2507.23726
- S10 https://arxiv.org/abs/2504.11354 · S11 https://arxiv.org/abs/2406.03847 · S12 https://arxiv.org/abs/2505.02735
- S13 https://arxiv.org/abs/2310.00656 · S14 https://arxiv.org/abs/2210.12283 · S15 https://arxiv.org/abs/2606.20068
- S16 https://arxiv.org/abs/2507.08649 · S17 https://www.nature.com/articles/s41586-025-09833-y
- S18 https://proceedings.iclr.cc/paper_files/paper/2025/hash/d630537fc4402cfa3ebbc7450a0cac91-Abstract-Conference.html
- S19 https://arxiv.org/abs/2606.31002 · S20 https://arxiv.org/abs/2604.25031 · S21 https://arxiv.org/abs/2603.19828
- S22 https://arxiv.org/abs/2207.01780 · S23 https://arxiv.org/abs/2307.04349 · S24 https://arxiv.org/abs/2310.13032
- S25 https://arxiv.org/abs/2206.08896 · S26 https://arxiv.org/abs/2506.13131 · S27 https://arxiv.org/abs/2509.19349
- S28 https://github.com/google-deepmind/funsearch/blob/main/implementation/programs_database.py
- S29 https://arxiv.org/abs/1708.09251 · S30 https://arxiv.org/abs/1901.10995 · S31 https://arxiv.org/abs/1912.02400
- S32 https://cdn.aaai.org/Workshops/2006/WS-06-11/WS06-11-005.pdf (AAAI 2006, pp. 1014–1019)
- S33 https://arxiv.org/abs/2010.03934 · S34 https://arxiv.org/abs/2410.06209 · S35 https://arxiv.org/abs/2203.02283
- S36 https://github.com/leanprover-community/repl
- A1 https://arxiv.org/abs/2202.01344 · A2 https://arxiv.org/abs/2205.11491 · A3 https://arxiv.org/abs/2410.15700
- A4 https://arxiv.org/abs/2310.10692 · A5 https://arxiv.org/abs/2605.27130 · A6 https://arxiv.org/abs/2512.17260
- A7 https://arxiv.org/abs/2509.22908 · A8 https://arxiv.org/abs/2605.30914

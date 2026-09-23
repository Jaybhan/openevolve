# LLM theorem provers writing Lean 4 (2024-2026): survey notes for the pruning-verifier design

Literature notes for the upper-bounds thesis (Zarankiewicz z(m,n;s,t) via pruned case-split SAT +
Lean-verified prunes). Written 2026-09-21. Scope: DeepSeek-Prover V1.5/V2, Kimina-Prover,
Goedel-Prover V1/V2, Seed-Prover 1.0/1.5, Harmonic Aristotle, AlphaProof (Nature 2025/26),
Lean Copilot, LeanDojo, Draft-Sketch-Prove, Lean-STaR, plus the 2026 follow-ups that answer the
questions the proposal actually asks (Hilbert, Delta-Prover, DSP+, Goedel-Architect, Numina-Lean-Agent,
MerLean-Prover, Leanstral, the UW "Evaluation of LLMs for Mathematical Formalization in Lean",
Process-Verified RL, Leanabell-V2, APRIL, LeanProgress, Kimina Lean Server, Lean REPL).

What was read: primary PDFs (downloaded to the scratchpad `lit/` folder) for DeepSeek-Prover-V1.5,
DeepSeek-Prover-V2, Kimina-Prover Preview, Goedel-Prover-V2, Seed-Prover, Seed-Prover 1.5, Aristotle,
the UW evaluation; the AlphaProof paper via its PMC open-access full text; arXiv HTML plus model
cards / READMEs for the rest. Quotes below are verbatim from those texts unless marked *[inferred]*.
Where a fact came only from a secondary source (blog, model card, search snippet) it is marked
*[secondary]*.

The questions this file answers (Section 4): (Q1) do frontier systems write Lean directly or go
NL-sketch -> autoformalization; (Q2) what compile/fix/retry loops and how many rounds; (Q3) what
partial-credit / reward signals are used in RL for proving; (Q4) which systems are open-weights or
API-accessible (OpenRouter) and their Lean 4 version compatibility; (Q5) what the results say about
splitting work between a reasoning model and a cheap model. Section 5 maps this onto the ZarPrune gate
(Lean 4.34.0, no Mathlib) and OpenEvolve.

---

## 1. Bibliographic record

| Key | Citation | Access |
|---|---|---|
| DSP-V1.5 | Xin, Ren, Song, Shao, Zhao, Wang, Liu, Zhang, Lu, Du, Gao, Zhu, Yang, Gou, Wu, Luo, Ruan. *DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search.* arXiv:2408.08152 (15 Aug 2024); ICLR 2025. | PDF read (28 pp, pp. 1-14 in detail) |
| DSP-V2 | Ren, Shao, Song, Xin, Wang, Zhao, Zhang, Fu, Zhu, Yang, Wu, Gou, Ma, Tang, Liu, Gao, Guo, Ruan. *DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition.* arXiv:2504.21801 (30 Apr 2025, rev. 18 Jul 2025). | PDF read (39 pp, pp. 1-16) |
| Kimina | Wang, Unsal, Lin, Baksys, Liu, Dos Santos, Sung, Vinyes, Ying, Zhu, Lu, de Saxce, Bailey, Song, Xiao, Zhang, ..., Polu, ..., Yang, Liu, Li (Numina + Moonshot). *Kimina-Prover Preview: Towards Large Formal Reasoning Models with Reinforcement Learning.* arXiv:2504.11354 (15 Apr 2025). Follow-up: HF blog "Kimina-Prover: Applying Test-time RL Search on Large Formal Reasoning Models" (Jul 2025) and "Kimina-Prover-RL" blog. | PDF read (24 pp, pp. 1-12); blogs fetched |
| Goedel-V1 | Lin, Tang, Lyu, Wu, Lin, Yang, Li, Xia, Chen, Arora, Jin. *Goedel-Prover: A Frontier Model for Open-Source Automated Theorem Proving.* arXiv:2502.07640 (11 Feb 2025, v3 19 Apr 2025). | HTML fetched |
| Goedel-V2 | Lin, Tang, Lyu, Yang, Chung, Zhao, Jiang, Geng, Ge, Sun, Wu, Gesi, Lu, Acuna, Yang, Lin, Choi, Chen, Arora, Jin. *Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction.* arXiv:2508.03613 (5 Aug 2025); ICLR 2026. | PDF read (24 pp, pp. 1-13) |
| Seed-1.0 | ByteDance Seed AI4Math (Chen, Gu, Huang, ... Zhu; 36 authors). *Seed-Prover: Deep and Broad Reasoning for Automated Theorem Proving.* arXiv:2507.23726 (31 Jul 2025, v2 1 Aug 2025). | PDF read in full (12 pp) |
| Seed-1.5 | ByteDance Seed AI4Math. *Seed-Prover 1.5: Mastering Undergraduate-Level Theorem Proving via Learning from Experience.* arXiv:2512.17260 (19 Dec 2025). Model ID "Doubao-Seedprover-1.5". | PDF read in full (12 pp) |
| Aristotle | The Harmonic Team (Achim, Best, Bietti, ..., Wu). *Aristotle: IMO-level Automated Theorem Proving.* arXiv:2510.01346 (1 Oct 2025, v2 10 Oct 2025). | PDF read in full (23 pp; pp. 1-14 text, rest refs/appendix) |
| AlphaProof | Hubert, Mehta, Sartran, Horvath, Zuzic, Wieser, Huang, Schrittwieser, Schroecker, Masoom, et al. (Google DeepMind). *Olympiad-level formal mathematical reasoning with reinforcement learning.* **Nature 651 (8106), 607-613.** Received 3 Jun 2025; accepted 30 Oct 2025; published online 12 Nov 2025; issue date 2026. doi:10.1038/s41586-025-09833-y. PMC12999475 (CC BY 4.0). | Full text read via PMC HTML (nature.com redirects to a login wall) |
| Lean Copilot | Song, Yang, Anandkumar. *Lean Copilot: Large Language Models as Copilots for Theorem Proving in Lean.* arXiv:2404.12534 (18 Apr 2024, v3 11 May 2025). | HTML + README fetched; PDF downloaded |
| LeanDojo | Yang, Swope, Gu, Chalamala, Song, Yu, Godil, Prenger, Anandkumar. *LeanDojo: Theorem Proving with Retrieval-Augmented Language Models.* arXiv:2306.15626; NeurIPS 2023 (Datasets & Benchmarks). | HTML + README fetched; PDF downloaded |
| DSP | Jiang, Welleck, Zhou, Li, Liu, Jamnik, Lacroix, Wu, Lample. *Draft, Sketch, and Prove: Guiding Formal Theorem Provers with Informal Proofs.* arXiv:2210.12283; ICLR 2023. (Isabelle, not Lean.) | HTML fetched; PDF downloaded |
| Lean-STaR | Lin, Sun, Welleck, Yang. *Lean-STaR: Learning to Interleave Thinking and Proving.* arXiv:2407.10040 (14 Jul 2024, v5 15 Mar 2025); ICLR 2025. | HTML fetched; PDF downloaded |
| Hilbert | Varambally, Voice, Sun, Chen, Yu, Ye. *Hilbert: Recursively Building Formal Proofs with Informal Reasoning.* arXiv:2509.22819; ICLR 2026. | HTML fetched; PDF downloaded |
| Delta | Zhou, Zhao, Zhang, ..., Li (ByteDance). *Solving Formal Math Problems by Decomposition and Iterative Reflection (Delta Prover).* arXiv:2507.15225 (21 Jul 2025). | HTML fetched |
| DSP+ | Cao, Song, Li, Le, Zhang, Xue, Yang. *Reviving DSP for Advanced Theorem Proving in the Era of Reasoning Models (DSP+).* arXiv:2506.11487 (13 Jun 2025). | HTML fetched |
| G-Architect | *Goedel-Architect: Streamlining Formal Theorem Proving with Blueprint Generation and Refinement.* arXiv:2606.06468 (Jun 2026). | HTML fetched |
| Numina-Agent | Liu, Zhou, Zhu, Dos Santos, He, Liu, Wang, Xie, Zhao, Wang, Zhi, Li, Li. *Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics.* arXiv:2601.14027 (20 Jan 2026). | abs + README fetched |
| MerLean | Li, Zhu, Ren. *MerLean-Prover: A Recursive Looping Harness for Lean 4 Theorem Proving.* arXiv:2605.26959 (26 May 2026). | abs fetched; PDF downloaded |
| Leanstral | Mistral AI. *Leanstral* (16 Mar 2026, `labs-leanstral-2603`, retired) and *Leanstral 1.5: Proof Abundance for All* (2 Jul 2026, `labs-leanstral-1-5`, HF `mistralai/Leanstral-1.5-119B-A6B`). | docs + news + HF card fetched *[secondary]* |
| UW-eval | Klingner, Bladek, Crawford, Chen, Fu, Nair, Alper, Inchiostro, Ilin (UW Math AI Lab). *Evaluation of LLMs for Mathematical Formalization in Lean.* arXiv:2606.05632 (4 Jun 2026). | PDF read in full (15 pp; pp. 1-8 text) |
| PV-RL | Kim, Yun (KAIST). *Process-Verified Reinforcement Learning for Theorem Proving via Lean.* arXiv:2606.20068 (18 Jun 2026). | HTML fetched |
| Leanabell-V2 | Ji, Liu, Wang, Zhang, Yue, Shi, Sun, Zhang, Zhou, Gai. *Leanabell-Prover-V2: Verifier-integrated Reasoning for Formal Theorem Proving via Reinforcement Learning.* arXiv:2507.08649 (11 Jul 2025). | HTML fetched |
| APRIL | Wang, Chess, Lee, Ge, Mallavarapu, Alper, Ilin. *Learning to Repair Lean Proofs from Compiler Feedback.* arXiv:2602.02990 (3 Feb 2026, rev. 13 Mar 2026). | HTML fetched; PDF downloaded |
| LeanProgress | George, Huang, Song, Anandkumar. *LeanProgress: Guiding Search for Neural Theorem Proving via Proof Progress Prediction.* arXiv:2502.17925 (25 Feb 2025). | abs fetched |
| KLS | Dos Santos, de Saxce, Wang, Wang, Baksys, Unsal, Liu, Liu, Li. *Kimina Lean Server: A High-Performance Lean Server for Large-Scale Verification.* arXiv:2504.21230 (29 Apr 2025, rev. 16 Dec 2025). README of `project-numina/kimina-lean-server`. | abs + README fetched |
| REPL | leanprover-community/repl README. | fetched |

Not in scope but referenced by the above: LongCat-Flash-Prover (arXiv:2603.21065, 560B MoE, open
weights *[secondary]*), Compile-to-Compress (arXiv:2604.18587), EconProver (arXiv:2509.12603),
LAMP (arXiv:2606.28841), Numina-Putnam2025 repo.

Local copies (scratchpad, session-specific):
`/private/tmp/claude-501/-Users-jaybhan-Downloads-openevolve/88cfd163-c49a-4b7d-ba6f-24d477547106/scratchpad/lit/{dsp15,dsp2,kimina,goedel1,goedel2,seedprover,seedprover15,aristotle,leanstar,leancopilot,leandojo,dsp,merlean,repair,evalform,hilbert,kiminaserver}.pdf`,
`alphaproof_pmc.{html,txt}`, `or_models.json` (OpenRouter catalog snapshot 2026-09-21),
`leanchk/TacticProbe.lean` (the Lean 4.34 core-tactic probe from Section 5.2).

---

## 2. System-by-system summary (with exact quotes)

### 2.1 AlphaProof (DeepMind; Nature 651:607-613)

**Architecture.** AlphaZero-style tactic-level tree search; the model never emits a whole proof.
"At its core is the proof network, a 3-billion-parameter encoder-decoder transformer model, that
learns to interpret the observed Lean tactic state and generate two outputs: a policy, suggesting
promising tactics to apply next, and a value function, estimating the expected return G_t." Actions
are Lean tactics as text. "Key adaptations for formal theorem proving include an AND-OR tree
structure to handle the decomposition of proofs into multiple independent subgoals ... that must all
be solved." "Unlike previous AlphaZero applications, AlphaProof does not commit to intermediate
actions and does not restart the search from subsequent states within a single proof attempt."

**Reward (this is the one non-binary reward in the whole survey that is used at scale).**
"The agent is incentivized to find short proofs by a reward signal r_t = -1 for each tactic applied.
The return G_t from a state s_t is the sum of these rewards until termination. Crucially, for proof
states that decompose into multiple independent subgoals that must all be solved, the return is
defined as the minimum return over these subgoals (that is, corresponding to the longest proof
branch), rather than the more natural sum of returns from each subgoal." So the value head is a
learned estimate of *negative remaining proof length along the bottleneck branch* -- a difficulty
estimator, not a partial-credit score.

**NL vs Lean.** NL enters only through autoformalization of *statements*: "This process uses a
Gemini-based LLM, fine-tuned and iteratively refined using human and synthetic data. Over the course
of the project, this model auto-formalized approximately 1 million natural-language problems into a
dataset of around 80 million formal Lean problems." "Importantly, each auto-formalized statement,
regardless of its fidelity to the original natural-language problem, provides a valid formal problem
that AlphaProof can attempt to prove or disprove." The prover itself writes Lean tactics directly.

**Test-time RL.** "TTRL uses the same core AlphaZero-inspired RL algorithm as the main training
phase, but instead of learning from a broad curriculum of auto-formalized problems, TTRL focuses
learning on a bespoke curriculum of synthetic problem variants (for example, simplifications or
generalizations) generated specifically around the target problem." IMO 2024: P1, P2, P6 solved;
"Each of these solutions required 2-3 days of TTRL"; P3 and P5 (combinatorics) unsolved.

**Compute / environment.** Main RL "approximately 80,000 tensor processing unit (TPU) days";
autoformalization "approximately 100,000 TPU days". Lean 4 + Mathlib (version not stated), plus
"approximately 100 elementary theorems and specialized, reusable tactics", `linarith` compiled to C
for speed. Closed; no API.

**Error-feedback loop.** None in the LLM-agentic sense; feedback is the tactic-state transition
inside the search tree (a failed tactic is simply a dead edge).

### 2.2 Aristotle (Harmonic; arXiv:2510.01346)

**Three subsystems (verbatim list).** "1. A Lean proof search algorithm which ingests a Lean proof
sketch and attempts to prove all unproved goals in the sketch. The search algorithm is a highly
parallel Monte Carlo Graph Search (MCGS) using a large transformer as its policy and value function.
The policy predicts Lean tactics conditional on the Lean proof state, proof history, and, if
available, an informal proof. 2. A lemma-based informal reasoning system which generates informal
proofs of mathematical statements, breaks these proofs down into lemmas, formalizes each lemma into
Lean, and iterates this process based on formal feedback. 3. A geometry solver ..."

**Search.** "It receives a block of Lean code and attempts to replace all sorry statements in the
code block with proofs." "An action is a text string to be interpreted as a fragment of Lean code.
This may consist of a single Lean tactic or a sequence of multiple tactics, and may also include
informal comments." AND/OR: "a state is proven if any single action succeeds (an OR condition),
while an action is successful only if all of its resulting states are proven (an AND condition)
... we explore actions with the highest potential (highest upper-confidence bound) and then
prioritize the most challenging of their resulting states (lowest lower-confidence bound),
effectively targeting the proof's bottleneck first. Furthermore, in order to prune states and
disprove statements, we augment each single-goal state with a state transition corresponding to
the logical negation of that goal." The model also "uses a hidden chain of thought with a
dynamically set thinking budget before predicting an action."

**RL.** "train it via reinforcement learning in the style of expert iteration ... We train the
generative policy on proofs found by search, filtered by measures of nontriviality. We train the
value function on proven states within these proofs and on nearby states that are disproven or
unproven after significant effort." Statement autoformalization uses "judging using signals from
the Lean REPL, and correction". Test-time training: retrain on own search traces if unsolved.

**NL -> Lean loop (the DSP lineage, with error feedback).** "1. First, we ask for an informal proof
of the theorem. 2. Next, we ask for the proof from (1) to be restructured as a sequence of lemmas
which build on each other, and which individually have very short proofs. 3. We then ask for
formalizations of the statements of the lemmas produced in (2). 4. Finally, after sending the
formalizations from (3) to the Lean REPL, we communicate any error messages back and ask for
corrections." Iteration: "At the end of a failed attempt, some lemmas will generally have been
proven while others will remain unproven ... We annotate the list of lemmas used for the attempt
by marking each as proved or unproved, then use this as input for a modified version of the
original query sequence" (revise lemma list keeping proved ones; formalize; REPL error correction
again). Scaling to IMO: ">200B parameters", "many parallel instances of the lemma-based reasoning
pipeline", "Iterating the formal feedback loop of this pipeline multiple times on individual
instances". Number of rounds is not given.

**Acceptance criterion.** "we only consider a problem to be solved if our system produces a
complete proof using the Lean 4 proof language and its mathematical library Mathlib, without gaps
or unsound axioms like sorryAx."

**Access.** Closed weights. API: "As of September 2025, you can register for early access to
Aristotle at our website" (harmonic.fun); `aristotlelib` on PyPI exposes
`aristotle submit "Fill in all sorries in this project" --project-dir ./my-lean-project --wait`
and `aristotle formalize paper.tex` *[secondary]*. Pricing/rate limits and the supported Lean
toolchain are not published in what I could read.

### 2.3 DeepSeek-Prover-V1.5 (arXiv:2408.08152)

Three stages: continued pre-training on Lean/Isabelle/Metamath; SFT on 9,645k sequences with
"natural language chain-of-thought comments alongside Lean 4 code" and tactic-state comments
("For each tactic in a generated valid formal proof, we insert the tactic state returned by the
verifier as a comment '/- tactic state: ... -/'"); then RLPAF.

**Reward.** "each generated proof receives a reward of 1 if verified as correct, and 0 otherwise.
While this binary reward signal is accurate, it is also sparse ... To mitigate this sparsity, we
select training prompts that are challenging yet achievable for the supervised fine-tuned model"
(~4.5k statements with moderate success rate). GRPO, 32 candidates per theorem, KL 0.02.

**Feedback loop = truncate-and-resume (whole-proof + MCTS hybrid).** "we submit the entire proof
the model generated to the Lean prover to parse it into tactics. We then truncate the proof at the
earliest verification error, ensuring that all subsequent tactic codes can be successfully applied
to advance the proof." The verified prefix plus the tactic state comment becomes the prompt for the
next generation. RMaxTS intrinsic reward: R_intrinsic(tau) = 1[at least one new node is added to
the search tree], with discounted UCB (gamma = 0.99) for non-stationarity.

**Numbers (miniF2F-test).** Pass@128: SFT non-CoT 49.8%, SFT CoT 50.4%, RL non-CoT 50.5%, RL CoT
51.6%. Single-pass 60.2% at large budget; RL + RMaxTS 63.5% at 32 x 6400. ProofNet-test 25.3%.
Verification "subject to a time limit of 300 seconds"; imports Mathlib4 + Aesop; **Lean 4.9.0**.
7B; weights on HF.

### 2.4 DeepSeek-Prover-V2 (arXiv:2504.21801)

**The canonical "reasoning model sketches, small prover fills" pipeline.** "we prompt DeepSeek-V3
to first analyze the mathematical problem in natural language, then decompose the proof into
smaller steps, translating each step into a corresponding Lean formal statement. Since
general-purpose models are known to struggle with producing complete Lean proofs, we instruct
DeepSeek-V3 to generate only a high-level proof sketch with the details omitted. The resulting
chain of thought culminates in a Lean theorem composed of a sequence of `have` statements, each
concluded with a `sorry` placeholder." "To reduce the computational overhead of extensive proof
search, we employ a smaller 7B prover model specifically optimized for processing the decomposed
lemmas. Upon the successful resolution of all decomposed steps, a complete proof of the original
theorem can be automatically derived." The cold-start data: "problems that remain unsolved by the
7B prover model in an end-to-end manner, but for which all decomposed subgoals have been
successfully resolved."

**Reward.** "Training utilizes binary rewards, where each generated Lean proof receives a reward
of 1 if verified as correct and 0 otherwise." Plus a structural shaping term: "we incorporate a
consistency reward in the early steps of training, which penalizes the structural misalignment,
explicitly enforcing the inclusion of all decomposed have-structured lemmas in the final proof."
GRPO, 256 problems x 32 candidates x 32,768 tokens per iteration.

**No inference-time repair loop.** The paper describes only sampling (pass@N); the compiler is used
as a verifier, not as a conversational partner.

**Numbers.** Table 1 (miniF2F-test): 7B non-CoT 55.5/68.0/73.2/75.0 % at pass@1/32/1024/8192;
7B CoT 58.6/75.6/79.9/82.0; 671B non-CoT 59.5/73.8/76.7/78.3; 671B CoT 61.9/82.4/86.6/**88.9**.
Table 3 (mean output tokens): 7B non-CoT 442.6 vs CoT 4488.5; 671B non-CoT 761.8 vs CoT 6751.9
(so CoT costs ~9-10x tokens for +7-9 points at pass@32). PutnamBench 47/658 (pass@1024);
ProofNet-test 37.1% (pass@1024); ProverBench AIME 6/15. Key sentence for Q5: "our subgoal-guided
curriculum learning framework, which integrates the general-purpose model DeepSeek-V3 with a
lightweight specialized 7B prover, achieves a 89.8% success rate on miniF2F-valid, nearly matching
the performance of DeepSeek-Prover-V2-671B."

**Environment.** "All experimental results of DeepSeek-Prover-V2 are conducted with Lean
4.9.0-rc2." Weights: HF 7B (32K context, on V1.5-Base) and 671B (on V3-Base); code MIT, model under
DeepSeek Model License *[HF card]*.

### 2.5 Kimina-Prover Preview (arXiv:2504.11354) and Kimina-Prover 72B (Jul 2025 blog)

**Format.** "we design a novel formal reasoning pattern to enable Kimina-Prover to think in an
environment that aligns the informal and formal mathematical reasoning ... Within the thinking
block, we seek informal-formal alignment by interspersing informal reasoning with relevant Lean 4
code snippets ... we ensure that the majority of the Lean 4 code snippets appear in the final
proof." Cold start: Claude 3.7 Sonnet synthesized ~20K thinking blocks from (informal, formal) pairs.

**Reward.** "A binary reward signal is assigned: 1 for a completely correct proof and 0 otherwise."
Format filtering: "(1) each generated sample must contain at least one tactic block; and (2)
tactic blocks must collectively cover at least 60% of the Lean code included in the final Lean 4
solution." Negative-gradient samples dropped with probability 0.5. N=1000 problems, k=8 rollouts
per iteration, KL tau = 0.4 (Kimi k1.5 loss). Base Qwen2.5-72B.

**Infrastructure.** Numina Lean Server "employs an LRU-based caching mechanism that reuses
preloaded environments based on import headers ... achieving up to 100 iterations per second on
machines equipped with 64 CPU cores and 512 GB RAM ... requiring only 640 CPU cores for training."

**Numbers.** miniF2F-test 52.94% pass@1, 65.16% pass@8, 68.85% pass@32, 77.87% pass@1024,
80.74% pass@8192. Distills: 1.5B 61.9%, 7B 70.8% at pass@1024. Table 2 (general LLMs, pass@32,
same benchmark): OpenAI o3-mini 24.59%, gemini-2.5-pro-preview-03-25 37.70% vs Kimina 68.85%
-- the April-2025 baseline for "general models can't write Lean", now obsolete (see UW-eval).
Preview paper: no inference-time error fixing ("Promising future directions include ... enabling
iterative refinement using Lean compiler feedback").

**Kimina-Prover 72B (Jul 2025, HF blog) *[secondary but from the authors]*.** "Kimina-Prover
achieves a pass rate of 84.0% with pass@32 and 86.4% with the addition of a single round of error
correction." pass@1024 87.7%; with test-time RL search 92.2%. Error-fixing SFT triples
"(incorrect proof, Lean feedback, correct proof)". Table 2 of the blog, "on a selected subset of 59
MiniF2F-Test problems with the lowest win rates": 32x1 brute force 28.8%; 16+16 attempt-and-fix
35.6%; 32+32 attempt-and-fix 44.1%. Kimina-Prover-RL blog: "only one error-fix turn and cap the
error message at a set number of tokens"; reward "1 ... successfully verified by Lean using our
kimina-lean-server", malformed output "zero reward, regardless of whether the proof is actually
valid". Proofs "valid under Mathlib v4.15". Released: Kimina-Prover-72B (MIT on HF),
Distill-8B, Distill-1.7B, RL-1.7B/0.6B.

### 2.6 Goedel-Prover V1 (arXiv:2502.07640) and V2 (arXiv:2508.03613)

**V1.** Statement formalization at scale (1.64M statements; two formalizers; Qwen2.5-72B
faithfulness judge; compile check), then "a total of 8 iterations" of expert iteration from
DeepSeek-Prover-V1.5-Base with "DeepSeek-Prover-V1.5-RL to generate 16 proofs for each statement".
Pure whole-proof, "without interacting with the Lean compiler during the generation process".
miniF2F 57.6% pass@32; PutnamBench 7 (pass@512). Lean 4.9.0.

**V2: verifier-guided self-correction is the headline.** "Our framework formalizes this intuition
by explicitly incorporating verifier feedback within the whole-proof generation loop. We
structure the pipeline so that, after an initial proof attempt, verification failures are parsed
and communicated back into the model as corrective guidance. The model then generates proof
repairs, leading to an iterative self-correction process." Evaluation protocol: "the evaluations
are done under Lean 4.9.0-rc1. For the first round of whole-proof generation, the max token length
of the model is set to be 30,000. For the verifier-guided error-correction, we sequentially conduct
2 additional rounds of self-correction, given the verifier's feedback on the previous attempt. The
total number of tokens in the self-correction mode is set to be 40,000."

**RL.** "50% of the inputs are used for whole proof generation, and the remaining 50% for
first-round self-correction ... hybrid GRPO-based approach ... removes group normalization ...
incorporates clip-higher, overlong penalties, and dynamic sampling from DAPO, and excludes the KL
regularization term ... we modify the dynamic sampling strategy to only include problems with
pass rates in the range (0, 0.75] during optimization." Reward is binary compile success
*[inferred from text; the paper never states any other reward]*. Model averaging
(1-alpha) theta_0 + alpha theta after SFT and after RL to restore pass@N diversity.

**Numbers.** Table 2 miniF2F-test: 8B 84.6% pass@32 (86.7% w/ self-correction), 90.2% pass@8192;
32B 88.1% pass@32 (90.4% w/ correction), 91.8% pass@1024, 92.2% pass@8192, 92.6% pass@1024
w/ correction. Table 3 PutnamBench: 32B 43 (pass@32), 57 (pass@32 self-correction), 86 (pass@184
self-correction) vs DeepSeek-Prover-V2-671B 47 (pass@1024). "Adding self-correction provides a
consistent gain of approximately 2 percentage points in pass@32."

**Ablation on the loop (Section 3.5, Figure 7) -- the most useful numbers for Q2.** "we used YaRN
to extend the context length to 128k tokens and allowed up to 5 revision iterations ... (1)
removing the specific compiler error messages (w/o Error Messages), and (2) removing the
chain-of-thought from previous attempts, retaining only the formal proof (w/o Previous CoTs). The
results show that removing compiler feedback significantly lowers performance, confirming that
specific error messages are crucial for effective revision. Similarly, removing the reasoning from
previous attempts also slightly degrades performance ... with an extended context and more
revision iterations, the full self-correction model's pass@32 accuracy on MiniF2F reaches an
average of 92.7%, which surpasses the 92.2% performance of the model without self-correction at
pass@8192." (Figure 7: ~89% at iteration 0, ~92% by iteration 2, ~92.7% at 5.)

**Proof repair via `extract_goal` (Conclusion).** "Instead of regenerating an entire failed proof,
our method corrects only the faulty segment. We use Lean 4's compiler feedback and the
extract_goal tactic to isolate the unsolved subgoal, prompt the prover to solve it independently,
and then reinsert the correct solution into the original proof. On the MiniF2F benchmark, this
approach improves the amortized budget scaling curve by 1-2 percentage points."

Bases Qwen3-8B / Qwen3-32B; HF license Apache-2.0; prompt template: "Complete the following Lean
4 code: ```lean4 {}``` Before producing the Lean 4 code to formally prove the given theorem,
provide a detailed proof plan outlining the main proof steps and strategies."; max_new_tokens
32768 (40K in self-correction) *[HF card]*.

### 2.7 Seed-Prover 1.0 (arXiv:2507.23726)

**Paradigm.** Whole-proof, lemma-style: "we first require the model to generate some useful
lemmas -- each introduced by the keyword lemma -- before generating the main proof using theorem by
applying the generated lemmas ... it allows clear identification of the lemmas that have been
successfully proved, and those that need further refinement. Second, lemmas are modular -- they can
be compiled independently, stored independently, and combined freely." A lemma pool "stores
comprehensive data from all our inference runs, including lemma statements, lemma names, complete
proofs, proof difficulties, and dependency relations."

**Reward.** "we adopt multi-stage, multi-task reinforcement learning (RL) based on VAPO. The RL
reward is 1 if the formal statement is successfully proven, and 0 otherwise. Additionally, a
formatting penalty is applied to encourage the model to generate lemmas before attempting the main
theorem." Curriculum: "For problems that are too difficult for single-pass generation, we use our
proposer to generate easier problem variants ... We also exclude problems which are too easy
(i.e. proof rate above 1/4) from RL training." Prompt diversity: "randomly incorporates natural
language hints, natural language proofs, similar lemmas, proved lemmas, failed lemmas, failed
attempts, summaries of previous attempts, and Lean compiler feedback into the prompt."

**Three inference settings (Q2).** *Light*: "each proof attempt is refined up to 8-16 times and
evaluated under Pass@8-16. We denote the sample budget of Pass@n and up to m refinements as n x m,
so the sample budget of the light setting is equivalent to generating the whole proof at
Pass@64-256. This setting completes in 1-2 hours." Two behaviors: "First, it fixes Lean syntax
errors in response to Lean compiler feedback. Second, it refines initial proof sketches, a process
that might entirely alter the reasoning trajectory." *Medium*: outer refinement of the main proof
plus "inner refinement ... targets difficult lemmas that the outer refinement process generates but
fails to prove, using a light setting with an 8 x 8 budget"; proofs "potentially exceeding 1000
lines of code". *Heavy*: "the proposer generates thousands of conjectures (by default 5000) ...
Each lemma is scored based on its proof rate, semantic relevance, and proof length. The proof rate
serves as a strong indicator of lemma value; empirically, lemmas with low proof rate are often
crucial to the final proof."

**Numbers (Table 3).** IMO 2025 4/6 (Heavy; 5/6 post-competition); past IMO 78.1%; miniF2F-valid
100.0%, miniF2F-test 99.6% (Medium); PutnamBench 331/657 (Medium; 201/657 with Light only);
CombiBench 30.0%; MiniCTX-v2 81.8% (Light). "Unless otherwise specified, we use Lean v4.14.0 with
its corresponding Mathlib version" (MiniCTX-v2 under v4.16.0). Closed weights; the GitHub repo
holds the IMO proofs ("P1,3,4,5 are compiled under Lean v4.14.0").

### 2.8 Seed-Prover 1.5 (arXiv:2512.17260) -- the agentic successor

**Paradigm shift.** "Distinct from the step-level mode (one interaction per tactic) or the
whole-proof mode (long thinking followed by a single interaction), an agentic prover equipped with
experiential learning dynamically adjusts its interaction granularity and master auxiliary tools."
Tools: "Lean verification, Mathlib search, and Python execution. For Lean verification, we employ
LooKeng, a REPL-based Python interface that compiles Lean proofs and returns structured feedback
to the model. We permit the model input a lemma at each time instead of a whole proof. The
statement header and proved lemmas are stored in the running context." Budget: "a maximum sequence
length of 64K and a limit of 28 tool calls." Mathlib search "calibrated to a fixed Mathlib commit
(i.e., v4.22.0)".

**Rewards.** Agentic prover: "the model receives a reward of 1 if a valid proof is completed and
verified by the Lean compiler, and -1 otherwise" (VAPO, tool-integrated). Sketch model (the
NL->Lean-sketch bridge) uses a *hybrid* but still binary reward: "R = 1 if N_lemmas >= 3 and
S_FL >= 0 and S_NL >= 0.7, -1 otherwise", where S_FL is the Lean verification score of the sketch
(statements compile with `sorry` bodies), S_NL an LLM-judge rubric on "alignment with the NL proof,
decomposition granularity, difficulty reduction, and Lean junk value analysis", and "We require the
natural language prover to verify each lemma, immediately rejecting the sketch (i.e. natural
language quality score is -1) if any lemma is mathematically invalid."

**Test-time workflow.** NL prover (from Doubao-Seed-1.6) -> Sketch model -> agentic Lean prover on
each lemma at Pass@3 x 3; recursion "natural language proof -> Lean sketch"; "an initial maximum
search depth of 4. If a problem reaches the limit without resolution, we incorporate the lemmas
proven during the search into the context and restart the search from scratch. Consequently, this
extends the maximum search depth to 8."

**Numbers.** Table 1: Goedel-Prover-V2-32B pass@64 86/660 PutnamBench, 2/100 Fate-H, 0/100
Fate-X; Seed-Prover 1.0 medium 331/660 at "18 H20 days / problem"; Seed-Prover 1.5 agentic prover
alone pass@8 x 8 359/660, 57/100, 10/100. Table 2: full workflow 580/660 = 87.9% at "10 H20 days /
problem" vs AlphaProof 56.1% at "500 TPU days / problem", Hilbert 70.0% "avg pass@1840". Putnam
2025 11/12 within 9 hours at 40 H20-days/problem. Footnote: "our prover is not using any
'native_decide' in Putnam, which is unsafe under Lean." Lean v4.22.0. RL dynamics: average tool
calls per trajectory fell from ~15 to ~10 and sequence length ~28k -> ~17k tokens during RL, i.e.
RL teaches the model to consult Lean less, not more.

### 2.9 Draft, Sketch, and Prove (ICLR 2023; Isabelle)

The origin of the sketch paradigm. Three stages: draft an informal proof (human or Minerva/Codex),
"map informal proofs to formal proof sketches that share the same high-level structures" (few-shot
Codex), then "execute off-the-shelf automated provers on every open conjecture in the sketch"
(Sledgehammer + 11 heuristic tactics: auto, simp, blast, fastforce, force, eval, presburger, sos,
arith, linarith, auto simp: field_simps). Budget "100 queries made to Codex per problem".
miniF2F-test: human drafts 39.3%, Minerva-540B drafts 38.9%, Codex drafts 35.3% (vs 20.9%
Sledgehammer alone). No error feedback loop at all -- each sketch is tried once. Bottleneck
explicitly the sketcher.

### 2.10 LeanDojo / ReProver (NeurIPS 2023) and Lean Copilot (2024)

LeanDojo: Lean interaction (`initialize(theorem)`, `run_tac(state, tactic)` returning a proof
state, an error state "if the tactic execution is not successful, e.g., due to timeout or
inapplicable tactic", or ProofFinished); ReProver = ByT5 299M with dense premise retrieval
("retrieves 100 premises" from the accessible ~33k), best-first search, 10-minute wall limit.
LeanDojo Benchmark: 51.2% (random) / 26.3% (novel_premises) vs GPT-4 29.0% / 7.4%; miniF2F 26.5%.
Trained "five days on a single NVIDIA A100". README today: "This original LeanDojo library is
deprecated. Please use LeanDojo-v2 for all new projects." Requires Lean >= v4.3.0-rc2.

Lean Copilot: `suggest_tactics`, `search_proof` (LLM tactics + aesop best-first), `select_premises`
(retrieval from "a fixed snapshot of Lean and mathlib4"); "runs LLMs natively in Lean through its
foreign function interface (FFI)" via CTranslate2, and "users can bring any models ... through
ExternalGenerator" (server process; GPT/Claude examples). Human-in-the-loop: "requires only 2.08
manually-entered proof steps on average (3.86 required by aesop)"; autonomous "search_proof can
automate about 74.2% of the proof steps in a theorem ... aesop (40.1%)". Lean >= v4.3.0-rc2. Relevant
to us only as the reference design for *in-editor* tactic suggestion; the default ReProver models
are Mathlib-trained.

### 2.11 Lean-STaR (ICLR 2025)

Tactic-level model that emits an NL "thought" before each tactic. Thoughts bootstrapped
retrospectively: "we use GPT-4-0125 to annotate 52,438 thoughts" given (state, ground-truth tactic),
then two rounds of expert iteration keeping only successful trajectories ("K=32 times in parallel
with temperature T=1.0 ... N=5 per problem"). Results: Table 1 (InternLM2-Math-base-7b) SFT 30.7%
pass@32 -> Lean-STaR iter-2 34.8% pass@32 / 36.1% pass@64; Table 2 (InternLM2-plus-7b) 43.4% ->
46.3% pass@64. Lean 4 via LeanDojo Benchmark 4 v9. No compile-error repair; failed tactics are
just discarded. Evidence that a few tokens of NL before each formal step helps even at 7B.

### 2.12 The sketch-and-fill / blueprint pipelines that quantify the split (Q5)

**Hilbert (ICLR 2026).** Reasoner = Gemini 2.5 Pro (or Flash / gpt-oss-120b); prover =
Goedel-Prover-V2-32B or DeepSeek-Prover-V2-7B; Kimina Lean Server "with Lean v4.15.0 and Mathlib
v4.15.0". Flow: direct prover attempts (K_initial_proof = 4) -> if fail, reasoner writes informal
proof and a Lean sketch with `sorry` subgoals (K_sketch_attempts = 4) -> each subgoal tried by prover
(K_formal_proof = 4) and by a "shallow solve" that "iteratively refines proofs based on Verifier
feedback for up to K_proof_correction = 6 passes. When compilation errors indicate missing or
incorrect theorem references, [it retrieves] additional relevant theorems" -> recursion to depth
D = 5. Results: miniF2F-test 99.2% (Pro + Goedel-32B), 98.4% (Pro + DSP-7B), 94.7% (Flash +
Goedel-32B), 96.7% (Flash + DSP-7B); D = 0 (prover alone, pass@4) 75%; PutnamBench 462/660 (70.0%)
with Gemini 2.5 Pro but only 88/660 (13.3%) with gpt-oss-120b as reasoner. Stated conclusion:
"the choice of informal reasoner appears more critical than prover strength." Cost: "at most 4.5K
reasoner calls and 11.3K total calls" on the hardest problems; average 548 reasoner + 391 prover
calls per sample with retrieval.

**DSP+ (Jun 2025).** Draft QwQ-32B or DeepSeek-R1-671B; sketch DeepSeek-V3-0324; prove
BFS-Prover-7B + Aesop. miniF2F-test 80.7% at 1024 workflow attempts; PutnamBench 24/644 at 128.
Ablation: R1 drafts beat QwQ drafts, and "Removing the draft phase entirely causes greater
performance degradation than removing other components" -- again the *reasoner* is the lever.

**Delta-Prover (Jul 2025).** No specialized prover at all: "Gemini 2.5 Pro 05-06" with "reflective
decomposition and iterative proof repair" and a Lean 4 DSL (Suppose / Define / ShowBy / Conclude);
budget up to 16,384 API calls per problem; 95.9% miniF2F-test.

**Goedel-Architect (Jun 2026).** Blueprint (dependency graph of definitions and lemmas) generated
by "the open-weight DeepSeek-V4-Flash (284B-A13B)", each lemma closed "in parallel" by "a Lean
prover ... [with] access to the Lean compiler and a Mathlib retrieval tool"; failed lemmas carry
"statement_wrong ... or proof_too_hard" diagnoses that drive refinement, "refined up to 8 times"
(miniF2F) / "up to 16 times" (PutnamBench). miniF2F 100%, PutnamBench 597/672 (88.8%), IMO 2025
4/6, Putnam 2025 11/12, USAMO 2026 3/6, at "Avg. cost / Q (all) $0.44" versus "~$244" per problem
for Hilbert. I.e. a *cheap* general model as architect + tool-equipped prover is now the
cost-efficiency frontier *[secondary: numbers from the HTML fetch, not re-derived]*.

### 2.13 General-model agentic harnesses (2026): direct Lean with tools

**Numina-Lean-Agent (Jan 2026).** "combining Claude Code with Numina-Lean-MCP to enable
autonomous interaction with Lean, retrieval of relevant theorems, informal proving and auxiliary
reasoning tools"; base Claude Opus 4.5; "solves all problems in Putnam 2025 (12 / 12), matching
the best closed-source system." MIT; README requires only a project with `lean-toolchain` and a
lakefile (so any toolchain, including ours).

**MerLean-Prover (May 2026).** Planning / Check / Lean agents in a recursive loop where "the unit
of revision is the proof plan itself"; default `claude-opus-4-7`; Putnam 2025 12/12; FormalQualBench
10/23; "no fine-tuning, no custom RL objective"; transfers to Sonnet/Haiku *[abs]*.

**Leanstral 1.5 (Mistral, Jul 2026).** 119B-A6B MoE, Apache-2.0 weights on HF, free API
`labs-leanstral-1-5`; trained in two RL environments: a compile-feedback refinement loop ("submits
a proof, receives Lean compiler feedback, and refines its approach with each attempt") and an agentic
one ("it edits files, runs bash commands, and uses the Lean language server to inspect goals, errors,
and type information in real time"); 587/672 PutnamBench, 100% miniF2F *[Mistral news; secondary]*.
vLLM deploy needs `--tensor-parallel-size 4`, tool-call parser `mistral`; recommended with
`lean-lsp-mcp`. Lean/Mathlib version not stated anywhere I could find.

**UW evaluation (Klingner et al., Jun 2026) -- the only controlled pass@k vs refine@k study.**
14 models, n = 50 stratified subsets of miniF2F and miniCTX, zero-shot standardized prompt,
temperature 0.5, 16,384-token budget, LeanInteract; outputs "containing sorry, admit, or omitted
macros" filtered. Refine@k defined as Response_i = LLM(Prompt, FTS) for i = 1 and
LLM(Prompt, FTS, Response_{i-1}, Feedback_{i-1}) for 1 < i <= k with "the exact trace from the Lean
verifier". Results: Gemini 3.1 Pro 92% miniF2F refine@32; Claude Opus 4.7 86% miniCTX refine@32;
Table 3.1 Delta_32 = refine@32 - pass@32: miniF2F average +7.7 (Leanstral +28, Gemini Flash +20,
Goedel-32B +16, Opus 4.7 +8), miniCTX average -1.4 (Opus +22, but Gemini Flash -24, Qwen 3.5 -22,
GPT 5.4-nano -12). Their reading: "smaller models that lack the ability to store enough context
begin to lose information on errors from previous attempts, especially due to the verbosity of some
Lean 4 error messages". Specialized provers (Goedel, Leanstral) "underperformed on miniCTX by a
significant margin, trailing the next worst model by over 20% on pass@32" -- overfit to
competition-statement style. Cost: "Nemotron 3 Super and GPT-OSS 120B ... average costs of < $0.01
per correct proof"; frontier models "up to an average of $0.075 per attempt"; "Accuracy scales
linearly with average cost, with an increase of over 3% per $0.01 increase in cost per attempt."
Error taxonomy top-4: tactical errors, syntax errors, unsolved goals, generation failures
(sorry/admit) = 54.7% of failures.

### 2.14 Reward-shaping and repair-specific results (Q3)

**Process-Verified RL (Kim & Yun, Jun 2026).** Tactic-level shaped reward on top of outcome
GRPO: a tactic gets 1 if the whole proof verifies, d1 = -0.05 if Lean elaborates it but the proof
fails, d2 = -0.1 if the tactic itself errors, with first-error propagation ("once the first
erroneous tactic T_j occurs, the continuation ... cannot constitute a valid reasoning process");
potential function Phi(s) = "the probability that the current proof prefix can be completed into a
valid Lean proof"; advantage A_{i,t} = A_outcome + 1{t = first token of tactic} A_process. Gains:
STP-Lean miniF2F pass@64 57.9 -> 59.2; DSP-V1.5 ProofNet pass@32 16.8 -> 17.6. Small but "more
stable and robust". *[HTML fetch]*

**Leanabell-Prover-V2 (Jul 2025).** Verify after every ```lean``` block inside the long CoT
("we can verify after each generated code block in the long CoT and have the model reflect
iteratively until it produces correct proofs or hits the maximum iteration threshold"); "format
reward (R_format) and compilation status reward (R_failed and R_success). The R_format is set to a
smaller value (such as 0.2) compared to R_failed/R_success (such as 1.0)"; verifier tokens masked
from the loss. miniF2F pass@128: Kimina-Distill-7B 67.2 -> 70.4; DSP-V2-7B 76.2 -> 78.2. Lean 4.9.0.

**APRIL (Feb 2026).** 260K (broken proof, diagnostics, repair) tuples built by mutating correct
proofs (theorem swap via LeanExplore, tactic swap e.g. nlinarith <-> linarith, single-line and
multi-line redaction filled by DeepSeek-V3 and kept only when they fail); diagnostics = "error
messages, error lines, and goal states"; LoRA SFT; single-shot repair accuracy: Qwen3-4B 1.1% ->
27.4%, Goedel-8B -> 34.6%, Kimina-8B -> 31.9%, vs Goedel-Prover-V2-32B baseline 26.8%. Lean
4.22.0-rc4. So off-the-shelf provers repair ~1 in 4 broken proofs in one shot from diagnostics.

**LeanProgress (Feb 2025).** Predicts remaining tactic steps: "overall prediction accuracy of
75.8% in predicting ... the remaining number of steps"; used as a best-first heuristic for
ReProver: "3.8% improvement on Mathlib4 compared to baseline performances of 41.4%".

**Kimina error-fix (Section 2.5), Goedel-V2 ablation (2.6), Seed light setting (2.7)** are the
three primary sources for "how many rounds".

### 2.15 Verification infrastructure

**Lean REPL** (leanprover-community/repl): JSON over stdin/stdout; `{"cmd": ...}` with optional
`env` to reuse an environment ("You can backtrack simply by using earlier values for env"; imports
only allowed without `env`); tactic mode `{"tactic": ..., "proofState": n}` on goals created by
`sorry`; responses carry messages (severity, position, text) and "sorries" with goals; pickling of
envs/proof states to `.olean`. Run inside a project with `lake env path/to/repl`.

**Kimina Lean Server**: Docker image `projectnumina/kimina-lean-server:2.0.0`, default
`LEAN_SERVER_LEAN_VERSION=v4.26.0` (build-arg selectable), `/verify` endpoint, `kimina-client`
Python SDK, `LEAN_SERVER_MAX_REPLS` = CPU count - 1, per-REPL memory cap 8G, `gc: true`, MIT.
Paper: "1.5 to 2 times speedup in verification time" over prior tooling; LRU cache keyed by import
header.

---

## 3. Numbers at a glance

| System | Paradigm | miniF2F-test | PutnamBench | Lean ver. | Loop rounds | Reward |
|---|---|---|---|---|---|---|
| AlphaProof | tactic MCTS, 3B | n/a (IMO'24 3/5 non-geo) | 56.1% (per Seed-1.5 Table 2, 500 TPU-d/problem) | Lean 4 + Mathlib, unstated | search only; TTRL 2-3 days | r_t = -1 per tactic, min over subgoals |
| Aristotle | MCGS + NL lemma loop, >200B | n/a (IMO'25 5/6) | n/a | Mathlib, unstated | unbounded lemma-list revisions | expert iteration; value on proven/disproven states |
| DSP-V1.5-RL 7B | whole-proof + truncate-resume MCTS | 60.2% single-pass; 63.5% RMaxTS | -- | 4.9.0 | prefix resume (MCTS) | binary + intrinsic novelty |
| DSP-V2 671B | CoT whole-proof; V3 sketch + 7B fill for data | 82.4% p@32, 88.9% p@8192 | 47/658 p@1024 | 4.9.0-rc2 | none | binary + consistency (early) |
| Kimina 72B (Jul'25) | think-block whole-proof | 84.0% p@32, 86.4% +1 fix, 87.7% p@1024 | -- | Mathlib v4.15 | 1 fix turn | binary; format => 0 |
| Goedel-V2 32B | CoT whole-proof + self-correction | 88.1% p@32, 90.4% +2 rounds, 92.7% +5 rounds/128k | 43/57/86 (p@32/p@32 corr/p@184 corr) | 4.9.0-rc1 | 2 (5 in ablation) | binary |
| Seed-Prover 1.0 | lemma-style whole-proof + refinement | 99.6% (Medium) | 331/657 (Medium), 201/657 (Light) | 4.14.0 | Light 8-16; Medium inner 8x8 | 1/0 + format penalty |
| Seed-Prover 1.5 | agentic tool-use (28 calls, 64K) | -- | 580/660 (87.9%), 10 H20-d/problem | 4.22.0 | up to 28 tool calls per lemma; depth 4->8 | +1/-1; sketch: binary conjunction |
| Hilbert (Gemini 2.5 Pro + Goedel-32B) | NL sketch -> prover, recursive | 99.2% | 462/660 | 4.15.0 | 6 correction passes/node, depth 5 | none (no training) |
| Delta-Prover (Gemini 2.5 Pro) | general LLM + DSL, repair | 95.9% | -- | -- | up to 16,384 calls | none |
| Goedel-Architect (DeepSeek-V4-Flash) | blueprint + tool prover | 100% | 597/672 | unstated | 8-16 refinements | none |
| Leanstral 1.5 | agentic general Lean model | 100% *[vendor]* | 587/672 *[vendor]* | unstated | LSP/agent loop | RL (details unpublished) |
| UW-eval: Gemini 3.1 Pro | zero-shot direct Lean | 92% refine@32 (n=50 subset) | -- | Lean 4 (unstated) | 32 | none |

---

## 4. Answers to the five questions

### Q1. Direct Lean, or NL sketch then autoformalize?

Both, and the field has converged on a specific division rather than either extreme:

1. **Step-level searchers write Lean tactics directly** (AlphaProof, Aristotle's MCGS, ReProver,
   Lean-STaR) but Aristotle and Lean-STaR interleave NL: Lean-STaR's thought-before-tactic and
   Aristotle's "actions ... may also include informal comments" plus a hidden CoT. AlphaProof uses
   NL only upstream (Gemini autoformalizes *statements*, never proofs).
2. **Whole-proof RL provers write Lean directly after a long NL/Lean-interleaved CoT** (Kimina's
   think block with Lean snippets; DSP-V2 CoT mode; Goedel-V2; Seed-Prover). DSP-V2's Table 1/3 is
   the cleanest measurement of what the NL reasoning buys: +7.6 points at pass@32 for the 7B model
   (68.0 -> 75.6) and +8.6 for 671B (73.8 -> 82.4), for ~10x output tokens.
3. **Sketch-and-fill pipelines separate the two** (DSP -> DSP-V2 cold start -> DSP+ -> Hilbert ->
   Seed-Prover 1.5 sketch model -> Goedel-Architect blueprints -> Aristotle's lemma pipeline). The
   NL model produces an informal proof *and* the Lean skeleton with `have ... := by sorry` or
   `lemma ... := by sorry`; the formal statements are compiled immediately (sketch validity is
   checked by Lean before any proving), and holes are filled by a prover with a compile loop. The
   sketch is therefore "autoformalized" by the same model that wrote the NL, not by a separate
   translator, and it is checked syntactically at once.
4. **2026 agentic harnesses put a general frontier model directly on Lean with LSP/REPL tools**
   (Numina-Lean-Agent, MerLean, Leanstral, Seed-Prover 1.5's agentic prover). The UW study shows
   zero-shot frontier models writing Lean directly reach 92% (Gemini 3.1 Pro) / ~85% (Opus 4.7)
   refine@32 on miniF2F; the April-2025 picture in Kimina's Table 2 (o3-mini 24.6%, Gemini 2.5 Pro
   37.7% pass@32) is no longer the state of affairs.

*[inferred]* For our task the honest reading is: the proof *idea* (which counting inequality
kills a case) should come from a reasoning model in NL, but the deliverable must be a Lean skeleton
with typed holes produced in the same call, because every pipeline that succeeded compiles the
sketch's statements before spending any proving budget.

### Q2. Error-feedback loops and how many rounds

| Source | What is fed back | Rounds | Effect |
|---|---|---|---|
| DSP-V1.5 truncate-and-resume | verified prefix + `/- tactic state: -/` at first error | MCTS-many | 60.2 -> 63.5 |
| Kimina 72B | previous proof + Lean error (token-capped) | 1 | 84.0 -> 86.4 p@32; 28.8 -> 35.6 (16+16 vs 32x1) on hardest 59 |
| Goedel-V2 | previous proof + previous CoT + compiler messages | 2 (30k + 10k tokens); 5 with 128k ctx | +~2 p@32; 92.7% p@32 after 5 > 92.2% p@8192 |
| Seed-Prover Light | Lean feedback + self-summary of previous attempts | 8-16 per sample | Pass@8-16 x 8-16 ~ Pass@64-256; IMO 2022 P2 at 15 refinements vs Pass@8192 without |
| Seed-Prover Medium | inner loop on failed lemmas | 8 x 8 per lemma | 201 -> 331 PutnamBench |
| Hilbert | verifier feedback (+ retrieval when errors mention missing theorems) | 6 per subgoal, recursion depth 5 | 75% (D=0) -> 98.7% (D=3) -> 99.2% (D=5) |
| Aristotle | REPL error messages on lemma statements; proved/unproved annotation | unbounded outer loop | not quantified |
| Seed-Prover 1.5 | structured REPL feedback per lemma | <= 28 tool calls, 64K tokens | 331 -> 359 alone; 580 with workflow |
| Goedel-Architect | per-lemma diagnosis (statement_wrong / proof_too_hard) | 8-16 blueprint refinements | 597/672 |
| UW refine@k | exact verifier trace + previous proof | up to 32 | +7.7 avg on miniF2F; -1.4 on miniCTX; +22 for Opus, negative for small-context models |
| Leanabell-V2 | verify each code block inside CoT | until correct or max | +2-3 p@128 |
| APRIL | error messages, error lines, goal states | 1 (single shot) | 27-35% repair rate |
| DSP (2022), Goedel-V1, DSP-V2 inference, Lean-STaR | none | 0 | -- |

Consistent findings: (a) the *specific* error text matters (Goedel-V2 ablation: removing it
"significantly lowers performance"); (b) keeping the previous chain of thought helps slightly;
(c) gains saturate fast for RL-trained provers (2-5 rounds) and the marginal round is worth much
less than the first; (d) more rounds only help models with enough context to hold the history --
the UW study finds refine@32 *hurts* small-context models on long-context problems; (e) Kimina
caps the error message length and Seed replaces history with a self-summary -- both are context
hygiene measures.

### Q3. Reward / partial-credit signals in RL for proving

Every RL-trained prover in this survey uses a **binary terminal reward from the kernel** as the
primary signal: DSP-V1.5 (1/0), DSP-V2 (1/0), Kimina (1/0, 0 on format violation), Goedel-V2
(binary, pass-rate window (0, 0.75]), Seed-Prover 1.0 (1/0), Seed-Prover 1.5 (+1/-1), Leanabell-V2
(R_success/R_failed = 1.0). The shaping terms that exist are all about *structure or length*, not
about counting how much of the proof is done:

- AlphaProof: r_t = -1 per tactic; return = min over AND-subgoals (bottleneck length).
- DSP-V2: early-training "consistency reward" penalizing omission of the sketch's `have` lemmas.
- Kimina: format filter (>= 1 tactic block; blocks cover >= 60% of final Lean); zero reward otherwise.
- Seed-Prover 1.0: "formatting penalty ... to encourage the model to generate lemmas before
  attempting the main theorem".
- Seed-Prover 1.5 sketch model: R = 1 iff N_lemmas >= 3 and S_FL >= 0 and S_NL >= 0.7 -- the
  closest thing to "reward a sketch with sorry holes", and note it is a *conjunction of thresholds
  collapsed to +-1*, with an NL-prover veto on any invalid lemma.
- Leanabell-V2: R_format 0.2 vs 1.0.
- Process-Verified RL: per-tactic {1, -0.05, -0.1} with first-error propagation; gains ~1-2 pts.
- DSP-V1.5 RMaxTS: exploration bonus for adding a new tree node (search-time only).

What nobody does: reward = f(number of remaining goals) or f(number of sorries) at the terminal
step. *[inferred]* The reason is visible in the designs: a sorry-count reward is trivially hacked
(one big `sorry`, or many trivially-true lemmas), so "partial progress" is instead (i) *cached*, as
proved lemmas in a pool that later prompts can reuse (Seed 1.0/1.5, Aristotle, Hilbert,
Goedel-Architect), (ii) converted into *new training statements* (Goedel-V2's `extract_goal`
scaffolding, Seed's proposer variants, AlphaProof TTRL variants), or (iii) used as a *search
heuristic* (AlphaProof value head, LeanProgress remaining-steps, Aristotle's LCB "bottleneck first").
The verified-prefix idea (DSP-V1.5) is the one place partial credit enters the *sampling* loop,
and it is an exact, unhackable notion: the prefix before the first error is kernel-checked.

### Q4. Open weights, API access (OpenRouter), Lean version

Checked live on 2026-09-21 against `https://openrouter.ai/api/v1/models` (443 models) and a
1-token probe with the project key (remaining credit: total 1795 - used 1778.57 = about $16.4).

| Model | Weights | API | OpenRouter | Lean/Mathlib it was trained/evaluated on |
|---|---|---|---|---|
| DeepSeek-Prover-V2-7B / 671B | HF; code MIT, DeepSeek Model License | none official | catalog entry `deepseek/deepseek-prover-v2` exists but the probe returns `{"error":{"message":"No endpoints found for deepseek/deepseek-prover-v2.","code":404}}` -- **not usable today** | 4.9.0-rc2 |
| DeepSeek-Prover-V1.5 (Base/SFT/RL 7B) | HF | none | no | 4.9.0 |
| Kimina-Prover-72B, Distill-8B/1.7B, RL-1.7B/0.6B | HF, MIT | none | no | Mathlib v4.15 (Preview: DSP-V1.5's 4.9.0 miniF2F) |
| Goedel-Prover-V2-8B/32B (Qwen3 bases) | HF, Apache-2.0 | none | no | 4.9.0-rc1 (repo README: "Lean 4 version 4.9") |
| Goedel-Prover-SFT/RL (V1) | HF | none | no | 4.9.0 |
| Leanstral 1.5 (119B-A6B) | HF `mistralai/Leanstral-1.5-119B-A6B`, Apache-2.0 | Mistral API `labs-leanstral-1-5`, free preview | no (`leanstral` absent from catalog) | unstated |
| Seed-Prover 1.0 / 1.5 | closed | "Model ID: Doubao-Seedprover-1.5" (ByteDance platform; not verified) | no | 4.14.0 / 4.22.0 |
| Aristotle | closed | aristotle.harmonic.fun API + `aristotlelib`; pricing unpublished | no | Mathlib, unstated |
| AlphaProof | closed | none | no | Mathlib, unstated |
| ReProver / Lean Copilot models | HF (ct2 byt5-small) | n/a (runs locally / in Lean) | no | Lean >= 4.3.0-rc2; Mathlib snapshot |
| LongCat-Flash-Prover 560B | claimed open *[secondary]* | -- | no | -- |
| General models actually on OpenRouter (USD per M in/out) | -- | -- | `anthropic/claude-opus-4.7` 5/25; `anthropic/claude-sonnet-4.6` 3/15; `google/gemini-3.1-pro-preview` 2/12; `google/gemini-3-flash-preview` 0.5/3; `openai/gpt-5.4` 2.5/15; `openai/gpt-5.4-mini` 0.75/4.5; `openai/gpt-oss-120b` 0.15/0.60; `nvidia/nemotron-3-super-120b-a12b` 0.08/0.45; `deepseek/deepseek-v4-flash` 0.06/0.11; `deepseek/deepseek-v4-pro` 0.90/1.79; `qwen/qwen3.5-397b-a17b` 0.55/3.50 | whatever is in their pretraining; UW-eval shows the frontier ones handle current Lean 4 |

**Version-drift facts.** The open specialized provers were all trained against Lean 4.9 (mid-2024)
or 4.15 (Jan 2025) Mathlib snapshots; Seed-1.5 and APRIL are at 4.22; Kimina Lean Server ships
4.26 by default. Our gate is **Lean 4.34.0 with no Mathlib**. No prover model in this survey has
seen 4.34, and none has been evaluated Mathlib-free. The UW study's error taxonomy (tactical +
syntax errors dominate) and its remark that specialized provers collapse on miniCTX (unfamiliar
context) both predict that a Goedel/Kimina-style model dropped into ZarPrune will reach for
`nlinarith`, `Finset.sum`, `norm_num`, `positivity` and fail on `unknown identifier`. Section 5.2
confirms which core tactics exist.

### Q5. Reasoning model vs cheap model: what the data say

1. **The expensive reasoning model is worth it for the sketch, not for the holes.** Hilbert:
   swapping the reasoner from Gemini 2.5 Flash to Pro moves miniF2F 94.7 -> 99.2 with the same
   32B prover, and PutnamBench 88 -> 462 when going from gpt-oss-120b to Pro; swapping the prover
   from DSP-7B to Goedel-32B moves only 98.4 -> 99.2. DSP+: removing the draft model hurts most;
   R1 > QwQ-32B as drafter. DSP-V2: V3 sketch + 7B filler ~= 671B end-to-end on miniF2F-valid
   (89.8% vs 90.6%).
2. **The cheap model must be able to run a compile loop.** Goedel-V2 8B with 2 self-correction
   rounds (86.7%) beats DSP-V2 671B without one (82.4%) at pass@32. Kimina's attempt-and-fix
   16+16 beats 32x1 on hard problems. APRIL shows ~30% single-shot repair for 4-8B models given
   diagnostics.
3. **Cost frontier in 2026.** Goedel-Architect: DeepSeek-V4-Flash architect + tool prover at
   $0.44/problem vs Hilbert's ~$244/problem (Gemini 2.5 Pro reasoner, ~548 calls). UW: Nemotron 3
   Super / GPT-OSS-120B at < $0.01 per correct proof on miniF2F but ~20 points worse on
   context-heavy miniCTX; refine@32 helps frontier models on hard contexts (+22 for Opus on
   miniCTX) and hurts small ones.
4. **Token economics of "thinking".** DSP-V2: CoT mode is ~10x tokens for +8 points at pass@32.
   Seed-Prover 1.5: RL *reduced* tool calls 15 -> 10 and tokens 28k -> 17k while accuracy rose;
   average search-tool calls per proved Putnam problem <= 10 in 80% of cases.
5. **Escalation, not either/or.** Every successful pipeline tries the cheap path first (Hilbert:
   4 direct prover attempts; Seed: Light before Medium before Heavy; Goedel-Architect: prover then
   blueprint refinement) and escalates to the reasoner only on failure, with the reasoner's job
   being *re-decomposition*, not retrying the same hole.

---

## 5. Relevance to our design (ZarPrune gate + OpenEvolve)

### 5.1 What transfers directly

- **The gate is already the right shape.** `Prune P` = computable `kill` + `sound` obligation is
  exactly the "statement with a sorry hole" object that DSP-V2 / Hilbert / Seed-1.5 / Aristotle
  operate on: the *statement* (`sound : forall A, kill (profileOf A) = true -> not Valid P A`) is fixed
  by the harness, so the model can never change the theorem (the UW study's "structural
  extraction" safeguard, and Kimina's "theorem consistency ... to eliminate reward hacking",
  are free for us). The evolved object is `kill` plus its proof; a candidate that shrinks `kill`
  to `fun _ => false` is sound and useless, so pruning power must be measured independently.
- **Sketch-then-fill applies to the soundness proof.** A reasoning model should emit (i) the NL
  argument ("if row sums r_i satisfy sum_i C(r_i,t) > (s-1) C(n,t) then some t-set of columns is
  shared by s rows"), (ii) the `kill` predicate as executable Lean over `Profile`, and (iii) a
  `sound` proof skeleton whose `have` lemmas are `sorry`'d. Lean checks (ii) and the *statements*
  of (iii) at once (< 1 s here). Hole-filling is then a cheap-model loop.
- **Loop budget.** Mirror the consensus: 2-3 compile/fix rounds per candidate inside the
  OpenEvolve evaluator (Goedel-V2), feeding back the *exact* Lean messages with positions and the
  goal at the first error (APRIL's diagnostics; DSP-V1.5's tactic-state comment), the previous
  attempt, and a one-paragraph self-summary instead of the full history once it exceeds a few
  thousand tokens (Seed Light). Cap the error text (Kimina). Escalate to the reasoning model with a
  request to *re-decompose* after the cheap loop fails (Hilbert's recursion; Goedel-Architect's
  `proof_too_hard` diagnosis), not to retry the same hole.
- **Verified-prefix partial credit is safe; sorry-count is not.** Use DSP-V1.5's truncate-at-first-
  error as the *only* sub-verified score, and only as a tie-breaker among unverified candidates.
  Never let an unverified candidate outrank a verified one (every RL system keeps the terminal
  signal binary; Seed-1.5's sketch reward is a thresholded conjunction, not a sum).
- **Cache proved lemmas.** A lemma pool (Seed) of accepted `Prune`s and accepted helper lemmas
  over `sumFin` (e.g. a hand-rolled `choose`, `sumFin_mono`, a double-counting lemma) should be
  injected into later prompts as available premises; Aristotle's "can contain already proven
  background results ... and the policy model will be able to leverage these" is the same idea.
  OpenEvolve's artifact channel is the natural carrier.
- **Difficulty-aware curricula.** Kimina/Goedel/Seed all filter training problems to a pass-rate
  window; for us the analogue is to present the LLM with cases whose SAT solve time is in the
  painful-but-finite band, since those are where a prune buys wall-clock (Section 6).

### 5.2 What does *not* transfer, measured locally

`lean --version` = 4.34.0; `lake build` of ZarPrune completes in ~0.15 s cached, and `lake env lean`
on a fresh probe file elaborates in 0.2 s (Lean is not the bottleneck; a 300 s timeout as in
DSP-V1.5 is 1000x more than we need). Probe file `leanchk/TacticProbe.lean` (Mathlib-free):

- Work in core 4.34: `omega`, `decide`, `obtain`, `rintro`, `grind`, `ac_rfl`, `rw`, `simp`,
  `Fin.cases`, `of_decide_eq_true`, `Bool.noConfusion` (all already used in `Prunes.lean`/`Demo.lean`).
- **Not available:** `Nat.choose` ("Unknown constant `Nat.choose`"), and by extension everything
  the prover models lean on: `nlinarith`, `linarith`, `positivity`, `norm_num`, `Finset`,
  `Finset.sum`, `Nat.choose_symm`, `push_cast`. The Kovari-Sos-Turan prune therefore needs a
  hand-rolled `choose : Nat -> Nat -> Nat` with `choose_succ_succ`, monotonicity, and a
  double-counting lemma over increasing index tuples (`Incr`) -- none of which any surveyed model
  has seen in this form. The prompt must state the available vocabulary explicitly and forbid
  `native_decide` (Seed-1.5 footnote; ZarPrune README) and `sorry`/`admit` (UW sanitizer), and
  the evaluator must grep for them and check `#print axioms` (ZarPrune's `propext`/`Quot.sound`
  audit) rather than trust the exit code alone.
- Consequence for model choice: the specialized open provers (Goedel-V2, Kimina, DSP-V2) are
  poorly matched (wrong Lean version, Mathlib-only tactic vocabulary, and the miniCTX collapse
  says they generalize badly to unfamiliar definitions). Frontier general models via OpenRouter
  (UW: Gemini 3.1 Pro, Claude Opus 4.7 best; GPT-OSS-120B / Nemotron 3 Super cheapest) plus
  the compile loop are the realistic path; Leanstral 1.5 is the one open Lean-specialized
  candidate with an agentic LSP loop, but it is not on OpenRouter and needs 4 GPUs locally.
- Budget arithmetic at OpenRouter prices: a 6k-in / 4k-out call costs ~$0.13 on Opus 4.7,
  ~$0.06 on Gemini 3.1 Pro, ~$0.02 on GPT-5.4-mini, ~$0.003 on GPT-OSS-120B, ~$0.001 on
  DeepSeek-V4-Flash. With ~$16 left, the reasoning model should be called O(10) times for sketches
  and the loop run on a sub-cent model, exactly the Goedel-Architect split.

### 5.3 Concrete evaluator design suggested by the literature *[inferred]*

Fitness of a candidate file `cand_*.lean` defining `p : Prune P`:

1. Parse-and-elaborate `kill` alone (statement of `sound` left as `sorry`): if this fails, score 0
   and return the first error (position + message + goal) for one cheap-model fix; two rounds max.
2. Elaborate `sound`; on failure return the same diagnostics; run the cheap loop (2 rounds), then
   one reasoning-model re-decomposition, then stop. Score for an unverified candidate = fraction of
   `have` statements whose bodies elaborate without `sorry` (each `have` body is a separate
   kernel check via the REPL's `sorries`/`proofState`), used only to order the unverified tail.
3. Verified candidates get score = (cases killed among the enumerated partition pairs, weighted by
   an estimated SAT hardness per case) with an axiom audit and a negative test (the
   `notDescending_unsound` pattern: a small valid matrix that must *not* be killed, to reject
   disguised symmetry breaks early -- this is Aristotle's "state transition corresponding to the
   logical negation of that goal" used as a cheap disproof step).
4. Accepted prunes and their helper lemmas go into a pool that is prepended to future prompts.

---

## 6. Difficulty signals worth borrowing for "how hard is this SAT case"

- AlphaProof's value head = expected negative bottleneck length; the *min over AND-subgoals*
  rule says aggregate a case's difficulty by its hardest sub-cube, not the sum.
- LeanProgress: a learned regressor from state to remaining steps (75.8% accurate) improves
  best-first search by +3.8 points; the analogue is a regressor from (profile, propagation stats)
  to solver time.
- Seed-Prover lemma pool: empirical proof rate under a fixed small budget as the difficulty label;
  "lemmas with low proof rate are often crucial" -- cases that a cheap solver probe cannot close
  are exactly the ones a prune should target.
- Curriculum windows: Goedel-V2 keeps pass rate in (0, 0.75]; Seed drops proof rate > 1/4;
  DSP-V1.5 keeps "moderate success rate"; Kimina's TTRL evaluated on the 59 lowest-win-rate
  problems. Same idea: time-limited solver probes give a pass/fail rate per case class.
- Hilbert: recursion depth needed to close a subgoal (D = 0 solves 75%, D = 3 98.7%) is itself a
  difficulty measure; for us, the number of prune "levels" a case survives.
- DSP-V1.5's RMaxTS: novelty (did this attempt open a new node) as an exploration bonus -- for the
  evolutionary loop, reward candidates whose kill set is not a subset of the union of accepted
  kill sets.
- AlphaProof TTRL: generate simplifications/generalizations of the target instance
  (smaller (m,n), fixed t) to train/evaluate a prune before scaling.

---

## 7. Open questions

1. No surveyed system is evaluated Mathlib-free or on Lean >= 4.30; the pass rate of frontier
   models on ZarPrune-style `sumFin`/`Incr` lemmas is unmeasured. A 10-20 problem pilot (the
   KST prune's helper lemmas) is the first experiment.
2. Whether `grind` (core since ~4.22) can discharge the double-counting step in the KST prune, or
   whether a hand-rolled `choose` plus explicit induction is required.
3. DeepSeek-Prover-V2 is listed on OpenRouter but has no endpoints; is any specialized prover
   reachable by API at all without self-hosting? (Leanstral 1.5 via Mistral's free endpoint is the
   only candidate found.)
4. Aristotle's API cost, rate limits and supported toolchain are unpublished; whether it accepts a
   Mathlib-free project is unknown.
5. The right number of loop rounds for a *general* model on *our* lemma style: Goedel's 2 and
   Hilbert's 6 were tuned for Mathlib competition math; the UW Delta_32 numbers say it depends on
   the model's context handling.
6. OpenEvolve is not gradient RL; the literature's reward findings transfer as *fitness ordering*
   (binary verified flag as the primary key). Whether verified-prefix tie-breaking actually helps
   an evolutionary population or just rewards long skeletons is untested.
7. Seed-Prover 1.5's sketch reward (N_lemmas >= 3 and Lean-valid and NL-judge >= 0.7) is the
   only published recipe for grading a *sketch*; whether an LLM judge is worth its cost at our
   budget is open.
8. Reproducibility of the 2026 numbers (Goedel-Architect $0.44/problem, Leanstral 587/672) --
   both are vendor/HTML-level claims not independently checked here.

No Zarankiewicz pruning lemmas appear in these sources; the pruning content of the thesis comes
from Tan 2022 / Guy / Davies-Gill-Horsley and the dfield closures repo, which are covered in the
sibling notes.

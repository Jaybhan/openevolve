# AlphaEvolve, FunSearch, and evolutionary math discovery — literature notes

Notes for the MEng thesis "Discovering upper bounds for Zarankiewicz numbers z(m,n;s,t)"
(evolving Lean-verified pruning arguments for a Tan-style case-split SAT attack).
Compiled 2026-09-21. Primary texts were read in full where accessible (PDF/HTML);
secondary sources are marked. Quotations are verbatim from the source text.
Statements labelled **[inferred]** are my own conclusions, not claims of the source.

---

## 0. Sources and accessibility

| # | Source | Accessed as | Full text? |
|---|---|---|---|
| S1 | Romera-Paredes et al., *Mathematical discoveries from program search with large language models*, **Nature 625, 468–475 (14 Dec 2023 online; vol. dated 2024)**, doi:10.1038/s41586-023-06924-6. ("FunSearch") | PMC open-access copy (PMC10794145) + `google-deepmind/funsearch` repo (cloned) | Yes |
| S2 | Novikov, Vũ, Eisenberger, Dupont, Huang, Wagner, Shirobokov, Kozlovskii, Ruiz, Mehrabian, Kumar, See, Chaudhuri, Holland, Davies, Nowozin, Kohli, Balog, *AlphaEvolve: A coding agent for scientific and algorithmic discovery*, **arXiv:2506.13131v1 (16 Jun 2025)**, 44 pp. | PDF (pdftotext) + arXiv HTML | Yes |
| S3 | Georgiev, Gómez-Serrano, Tao, Wagner, *Mathematical exploration and discovery at scale*, **arXiv:2511.02864v3 (v1 3 Nov 2025; v3 22 Dec 2025)**, 81 pp. | PDF (pdftotext) + arXiv HTML | Yes |
| S4 | Google Cloud blog, *AlphaEvolve is available for everyone*, **9 Jul 2026**; Google Cloud docs (developer guide overview, best practices, API reference); Codelab "Get started with AlphaEvolve on Google Cloud"; repo `Google-Cloud-AI/alphaevolve-on-googlecloud` (cloned, Apache-2.0); InfoQ 19 Jul 2026 (S.-J. Wiggers) | WebFetch + clone | Yes (docs are the primary text; pricing page returned truncated) |
| S5 | SkyDiscover (UC Berkeley Sky Computing Lab): repo `skydiscover-ai/skydiscover` (cloned, Apache-2.0); blog index; Liu et al., *SkyDiscover: A Flexible, Adaptive Framework for AI-Driven Scientific and Algorithmic Discovery*, **CAIS '26, pp. 1223–1227, doi:10.1145/3786335.3813221** (accepted Mar 2026); AdaEvolve arXiv:2602.20133; EvoX arXiv:2602.23413 | Repo README/guides + arXiv abstracts | Repo yes; CAIS paper body not fetched (DOI only); AdaEvolve/EvoX abstracts only |
| S6 | Nagda, Raghavan, Thakurta, *Reinforced Generation of Combinatorial Structures: Hardness of Approximation*, arXiv:2509.18057 (v1 22 Sep 2025; rev. 9 Mar 2026); Google Research blog *AI as a research partner: Advancing theoretical computer science with AlphaEvolve* (30 Sep 2025) | abstract + blog | Abstract + blog only |
| S7 | Tsoukalas, Kovsharov, Shirobokov, Surina, Firsching, Bérczi, Ruiz, Suggala, Wagner, Wieser, Yu, Huang, et al. (Google DeepMind), *Advancing Mathematics Research with AI-Driven Formal Proof Search*, **arXiv:2605.22763v1 (21 May 2026)** | arXiv HTML | Yes (via HTML extraction) |
| S8 | Ye, Guan, Xie, Liu, Modi, Zhang, Wen, Kautz, Zhang, *ProofEvolve: Neuro-Symbolic Evolution for Formal Automated Theorem Proving*, **arXiv:2608.26334v1 (26 Aug 2026)** | arXiv HTML | Yes |
| S9 | Lu, Wang, Liu, *FormalEvolve: Neuro-Symbolic Evolutionary Search for Diverse Autoformalization*, arXiv:2603.19828 (20 Mar 2026; rev. 2 Sep 2026) | abstract | Abstract only |
| S10 | Tao, *The story of Erdős problem #1026* (blog, 8 Dec 2025); `teorth/erdosproblems` wiki "AI contributions to Erdős problems" | WebFetch | Yes |
| S11 | Hubert et al., *Olympiad-level formal mathematical reasoning with reinforcement learning* (AlphaProof), **Nature 651, 607–613 (12 Nov 2025)** | local PMC copy (from earlier session) | Yes (only reward/verification passages used) |
| S12 | Firsching et al., *Formal Conjectures: An Open and Evolving Benchmark for Verified Discovery in Mathematics*, arXiv:2605.13171 (13 May 2026) | abstract | Abstract only |
| S13 | Lange, Imajuku, Cetin, *ShinkaEvolve*, arXiv:2509.19349 (17 Sep 2025); Wang et al., *ThetaEvolve*, arXiv:2511.23473 (28 Nov 2025); Tanveer, *LEVI*, arXiv:2605.09764 (10 May 2026); Xing et al., *Compute Allocation for Self-Evolving LLMs: From Depth-Breadth to Multi-Armed Bandits* (BaSE), arXiv:2605.29268 (28 May 2026); Agarwal et al., *Inductive Deductive Synthesis*, arXiv:2605.23109 (22 May 2026) | abstracts | Abstracts only |

Local copies: `/private/tmp/claude-501/-Users-jaybhan-Downloads-openevolve/88cfd163-c49a-4b7d-ba6f-24d477547106/scratchpad/lit/` (`alphaevolve_2506.13131.pdf`, `alphaevolve.txt`, `tao_wagner_2511.02864.pdf`, `tao_wagner.txt`, `funsearch_pmc.txt`, repos `funsearch/`, `aecloud/`, `skydiscover/`).

Not found / not verified: the proposal's "[19] AlphaEvolve on Google Cloud (July 2026)" is the S4 blog post; "[20] SkyDiscover (Mar 2026)" is the S5 launch post of 3 Mar 2026 and the CAIS '26 paper. No source that puts a Lean checker *inside* an AlphaEvolve-style construction-evolution loop was found (see §4 and §8.2); the closest are S7 (Lean kernel inside an AlphaEvolve-inspired *proof* evolution) and S8 (kernel-graded fitness for proof DAGs).

---

## 1. FunSearch (S1) — the ancestor

**Claim of the paper.** "FunSearch (short for searching in the function space), an evolutionary procedure based on pairing a pretrained LLM with a systematic evaluator." Applied to the cap set problem: "we discover new constructions of large cap sets going beyond the best-known ones, both in finite dimensional and asymptotic cases." Also online bin packing heuristics. Key philosophical line: "In contrast to most computer search approaches, FunSearch searches for programs that describe how to solve a problem, rather than what the solution is."

**The loop (Methods).** Four ingredients, quoted:
1. "we sample best performing programs and feed them back into prompts for the LLM to improve on; we refer to this as best-shot prompting."
2. "we start with a program in the form of a skeleton (containing boilerplate code and potentially known structure about the problem), and only evolve the part governing the critical program logic. For example, by setting a greedy program skeleton, we evolve a priority function used to make decisions at every step."
3. "we maintain a large pool of diverse programs by using an island-based evolutionary method that encourages exploration and avoids local optima."
4. Asynchronous scaling.

**Why a fixed skeleton helps.** "Whereas a fixed skeleton may constrain the space of programs that can be discovered, we find it improves overall results because it focuses the LLM resources on only evolving the critical part, instead of also using the LLM to recreate already known program structures (with more opportunities for mistakes that would render the entire program incorrect)."

**Evaluation.** "Programs generated by the LLM are evaluated and scored on a set of inputs. For example, in the cap set problem ... the inputs are the values of the dimensionality n that we are interested in ... The scores across different inputs are then combined into an overall score of the program using an aggregation function, such as the mean." In the released code, `_reduce_score(scores_per_test)` returns the score of the **last** test key, and the per-test tuple `_get_signature` is used to cluster programs within an island (programs with identical signatures form one cluster).

**Islands.** "Several islands, or subpopulations, are created and evolved independently. To sample from the program database, we first sample an island and then sample a program within that island, favouring higher-scoring and shorter programs ... we let information flow between the islands by periodically discarding the programs in the worst half of the islands (corresponding to the ones whose best individuals have the lowest scores). We replace the programs in those islands with a new population, initialized by cloning one of the best individuals from the surviving islands."

Cluster selection is Boltzmann: `P_i = exp(s_i/T_cluster) / Σ exp(s_i'/T_cluster)`, with `T_cluster = T_0 (1 − (n mod N)/N)`; within a cluster shorter programs are preferred with probability ∝ `exp(ℓ̃_i / T_program)` where ℓ̃ is normalised negative length.

Released defaults (`implementation/config.py`): `functions_per_prompt=2`, `num_islands=10`, `reset_period=4*60*60` s, `cluster_sampling_temperature_init=0.1`, `cluster_sampling_temperature_period=30_000`, `num_samplers=15`, `num_evaluators=140`, `samples_per_prompt=4`. Spec format: a Python file where `@funsearch.run` marks the `evaluate` entry point and `@funsearch.evolve` marks the function to evolve (e.g. `priority`).

**Numbers.** Cap set in dimension 8: size 512 (previous best 496), but "only four out of 140 experiments discovering a cap set of size 512." Asymptotic: via admissible sets, lower bound on the cap-set capacity improved ("the largest improvement in 20 years to the asymptotic lower bound"); "all experiments on admissible sets improve on the previous best capacity lower bound, with 60% of experiments..." (sentence truncated in the extraction). Models: Codey (PaLM 2 family) and StarCoder; "No training or fine-tuning of a LLM is required; API access for inference is sufficient."

**Interpretability argument (relevant to the thesis' interpretability goal).** "Whereas most computer search techniques output directly what the solution is ... FunSearch produces programs generating the solution. For structured problems, such programs tend to be more interpretable—facilitating interactions with domain experts—and concise—making it possible to scale to large instances." The 512-cap program revealed reflection symmetry that let the authors write an explicit construction (Fig. 4c).

**Verification in FunSearch.** Purely by execution: the evaluator checks the cap-set property on the produced set. No formal proofs anywhere in the loop.

---

## 2. AlphaEvolve (S2)

### 2.1 Task specification and the evaluator contract
"The user must provide a mechanism for automatically assessing generated solutions. This mechanism takes the form of a function h mapping a solution to a set of scalar evaluation metrics" — concretely "a Python function, called evaluate, with a fixed input/output signature, returning a dictionary of scalars."

"For mathematical problems, the function h is typically very simple. For example, when wishing to find largest possible graphs satisfying a given property, h invokes the evolved code to generate a graph, checks whether the property holds, and then simply returns the size of the graph as the score. In more complicated cases, the function h might involve performing an evolved search algorithm, or training and evaluating a machine learning model."

Evolve markers: "simply by adding special markers (# EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END) as comments into the code." "While this initial implementation must be complete, it can be rudimentary—for instance, consisting of single-line functions that return constants of the appropriate types."

### 2.2 The four abstraction levels ("genome representations")
Verbatim (§2.1, "Flexibility in choosing the abstraction"):
> "AlphaEvolve can evolve the solution in raw string representation (as in classical evolutionary algorithms); evolve a function of a definite form that specifies how to construct the solution from scratch (the approach taken in [83]); evolve a bespoke search algorithm to find the solution within some fixed compute budget; or even co-evolve intermediate solutions and search algorithms together, such that each search algorithm is specifically tailored to further improve upon a particular intermediate solution."
> "We find that different levels of abstraction work better for different problems. For example, we hypothesize that for problems with highly symmetric solutions it is advantageous to evolve constructor functions as these tend to be more concise [83], whereas for problems with non-symmetric solutions it works better to evolve customized search algorithms."

Discussion (§6) restates: "searching for the solution directly, finding a function that constructs it from scratch, or evolving a search algorithm to find it. Applying AlphaEvolve in different ways comes with different biases (for example, finding constructive functions may favor discovering highly symmetric objects [83])".

### 2.3 The "search heuristic evolution" trick (§3.2 / Appendix B intro)
> "The key methodological innovation enabling these discoveries is AlphaEvolve's ability to evolve heuristic search algorithms rather than directly evolving the constructions themselves. For many problems, particularly those with fast objective function evaluations—which are common in mathematics—we employed an iterative refinement strategy. Each generation of AlphaEvolve was tasked with evolving a program representing a search heuristic. This program was given a fixed time budget (e.g., 1000 seconds) and was shown the best construction found by the previous best heuristic. Its goal was to leverage this starting point and the allotted time to find an even better construction. The evolutionary process thus selects for heuristics that are effective at improving already high-quality solutions. The final constructions were often the result of a sequence of different, specialized heuristics discovered by AlphaEvolve—early heuristics proficient at making large gains from random or simple initial states, and later heuristics adept at fine-tuning near-optimal configurations."

Matrix multiplication (Appendix A) is the same trick at a different level: the evolved object is a gradient-based tensor-decomposition *search procedure* (initializer, loss incl. a "discretization loss" and a "hallucination loss", optimizer, hyperparameter sweeps via a `hyper` library), not the decomposition. Exactness enforced in the evaluator: "To ensure the exactness of the decomposition and avoid any potential numerical error, when evaluating, we round each element to the nearest integer or the nearest half-integer" (rank 48 for ⟨4,4,4⟩ over C, beating Strassen's 49).

### 2.4 Prompt sampling, models, evaluation, database
- Prompt: "multiple previously discovered solutions sampled from the program database, as well as system instructions on how to propose changes"; optional "Explicit context: ... fixed human-written instructions, equations, code snippets, or relevant literature (e.g., pdf files)", "Stochastic formatting", "Rendered evaluation results", and meta-prompt evolution.
- Models: "a combination of Gemini 2.0 Flash and Gemini 2.0 Pro" (Flash for throughput, Pro for occasional high-quality jumps). Output as SEARCH/REPLACE diffs or full rewrites.
- Evaluation extras (§2.4), verbatim:
  - "Evaluation cascade (hypothesis testing): the user can specify ensembles of test cases of increasing difficulty, such that new solutions are evaluated on the next stage only if they achieve sufficiently promising results in all earlier stages. ... new solutions are initially evaluated on a small scale before being subjected to the main test cases, to filter out faulty programs early."
  - "LLM-generated feedback: ... desirable solutions have certain characteristics that are difficult to capture precisely in the user-provided evaluation function h; for example, simplicity of the discovered program. These properties can be graded using separate LLM calls and added to the dictionary of scores to steer evolution, or they can be used to discard solutions when a criterion is not fulfilled."
  - "Parallelized evaluation: the sample efficiency of AlphaEvolve makes it feasible to spend on the order of 100 compute-hours to evaluate any new solution."
  - "Multiple scores. ... even if one metric is of particular interest, optimizing for multiple metrics often improves results for the single target metric."
- Database (§2.5): "inspired by a combination of the MAP elites algorithm [74] and island-based population models [83, 97]." (OpenEvolve's MAP-Elites + islands is a reimplementation of exactly this sentence.)

### 2.5 Verification and correctness in AlphaEvolve
- Grounding claim: "The LLM-directed evolution process is grounded using code execution and automatic evaluation. This evaluation mechanism allows AlphaEvolve to avoid any incorrect suggestions from the base LLM."
- Math constructions are verified by exact recomputation, not proofs: "The data and verification code for all constructions reported in this section appear in the accompanying Google Colab." Kissing number 593 in d=11 is certified by an integer point set plus a one-paragraph lemma (see §8.5 below). Hardware/XLA proposals: "checked against the reference (unmodified) code on randomized inputs", and RTL "must pass robust verification methods to confirm that the modified circuit maintains functional correctness."
- **No Lean, no proof assistant anywhere in S2.** The only mention of formal proving is in related work: FunSearch-style discovery of "witnesses for, and counterexamples to, mathematical statements—a problem that is complementary to that of finding formal and informal proofs of mathematical statements [3, 19, 98, ...]."

### 2.6 FunSearch vs AlphaEvolve (Table 1, verbatim)
| FunSearch | AlphaEvolve |
|---|---|
| evolves single function | evolves entire code file |
| evolves up to 10-20 lines of code | evolves up to hundreds of lines of code |
| evolves code in Python | evolves any language |
| needs fast evaluation (≤ 20min on 1 CPU) | can evaluate for hours, in parallel, on accelerators |
| millions of LLM samples used | thousands of LLM samples suffice |
| small LLMs used; no benefit from larger | benefits from SOTA LLMs |
| minimal context (only previous solutions) | rich context and feedback in prompts |
| optimizes single metric | can simultaneously optimize multiple metrics |

### 2.7 Combinatorial results and how the objects were represented (Appendix B)
| Problem | Representation the evaluator scores | Result |
|---|---|---|
| B.1–B.3 autocorrelation inequalities | step function on 600 / 50 / 400 equally spaced intervals on [−1/4,1/4] | C1 ≤ 1.5053 (was 1.5098); C2 ≥ 0.8962 (was 0.88922); C3 improved |
| B.5 Erdős minimum overlap | step function | new upper bound (slight) |
| B.6 sums and differences of finite sets | a finite set U ⊂ ℤ≥0 with 0 ∈ U and \|U−U\| ≤ 2 max U + 1; score via Gyarmati–Hennecart–Ruzsa: C6 ≥ 1 + log(\|U−U\|/\|U+U\|)/log(2 max U + 1) | U1 (\|U1\|=2003) gives 1.1479; U2 (\|U2\|=54265) gives 1.1584 (was 1.14465) |
| B.7–B.10, B.12–B.13 packings / Heilbronn | coordinates | several new records (e.g. 11 hexagons in side 3.931; 12 in 3.942; Heilbronn triangle n=11: 0.0365) |
| B.11 kissing number d=11 | 593 integer points with max norm < min pairwise distance | 593 (was 592) |

Almost all of these used the search-heuristic mode of §2.3; the object is checked exactly by the evaluator, and the "proof" is the object plus a trivial checking lemma.

---

## 3. Georgiev–Gómez-Serrano–Tao–Wagner, "Mathematical exploration and discovery at scale" (S3)

### 3.1 Scope and headline
"we considered a list of 67 problems spanning mathematical analysis, combinatorics, geometry, and number theory. The system rediscovered the best known solutions in most of the cases and discovered improved solutions in several. In some instances, AlphaEvolve is also able to generalize results for a finite number of input values into a formula valid for all input values. Furthermore, we are able to combine this methodology with Deep Think [149] and AlphaProof [148] in a broader framework where the additional proof-assistants and reasoning systems provide automated proof generation and further mathematical insights."

Honest self-assessment: "AlphaEvolve was not able to match or exceed previous results in all cases, and some of the individual improvements it was able to achieve could likely also have been matched by more traditional computational or theoretical methods performed by human experts."

### 3.2 The three modes (this is the cleanest statement of the "search heuristic" trick)
**Why programs (§1.3):** "Many 'nice' mathematical objects ... have short, elegant descriptions as code. ... Searching in program space might act as a powerful prior for simplicity and structure, helping us navigate away from messy local maxima towards elegant, often optimal, solutions."

**Search mode:** "Still, for problems where the scoring function is cheap to compute, the sheer brute-force advantage of traditional methods can be hard to overcome. Our proposed solution to this problem is as follows. Instead of evolving programs that directly generate a construction, AlphaEvolve evolves programs that search for a construction. This is what we refer to as the search mode of AlphaEvolve, and it was the standard mode we used for all the problems where the goal was to find good constructions, and we did not care about their interpretability and generalizability."
"Each program in AlphaEvolve's population is a search heuristic. It is given a fixed time budget (say, 100 seconds) and tasked with finding the best possible construction within that time. The score of the heuristic is the score of the best object it finds. This resolves the speed disparity: a single, slow LLM call to generate a new search heuristic can trigger a massive cheap computation, where that heuristic explores millions of candidate constructions on its own."
"We emphasize that the search does not have to start from scratch each time. Instead, a new heuristic is evaluated on its ability to improve the best construction found so far. We are thus evolving a population of 'improver' functions. ... The downside is a potential loss of interpretability in the search process, but the final object it discovers remains a well-defined mathematical entity for us to study."

Origin story (Problem 6.2 notes): before search mode they "simply asked AlphaEvolve to suggest a mathematical function directly" and padded evaluation by scoring "thousands of other functions we obtained from the original function via simple transformations. This was the precursor of our search mode idea." Forcing their own transformations "was much more restricted and did not do well."

**Generalizer mode (§1.4):** "we tasked AlphaEvolve with writing a program that can solve the problem for any given n. We evaluate the program based on its performance across a range of n values. The hope is that by seeing its own (often optimal) solutions for small n, AlphaEvolve can spot a pattern and generalize it into a construction that works for all n." "This mode is more challenging, but it has produced some of our most exciting results" (Nikodym construction → new paper by Tao). Prompting for generalizable programs: "we task AlphaEvolve to search for concise, fast, reproducible and human-readable algorithms that avoid black-box optimization ... the scoring of a proposed algorithm would be done by evaluating its performance on a mixture of small and large inputs n and taking the average."

### 3.3 The verifier is the critical component (§4 Conclusions, verbatim)
- "We have found that the selection of the verifier is a critical component that significantly influences the system's performance and the quality of the discovered results. For example, sometimes the optimizer will be drawn more towards more stable (trivial) solutions which we want to avoid. Designing a clever verifier that avoids this behavior is key to discover new results."
- "employing continuous (as opposed to discrete) loss functions proved to be a more effective strategy for guiding the evolutionary search process in some cases. For example, for Problem 6.54 we could have designed our scoring function as the number of touching cylinders of any given configuration (or −∞ if the configuration is illegal). By looking at a continuous scoring function depending on the distances led to a more successful and faster optimization process."
- "we also observed a 'cheating phenomenon', where the system would find loopholes or exploit artifacts (leaky verifier when approximating global constraints such as positivity by discrete versions of them, unreliable LLM queries to cheap models, etc.) in the problem setup rather than genuine solutions, highlighting the need for carefully designed and robust evaluation environments."
- Concrete cheat (Problem 6.2): "it always eventually figured out a way to cheat by suggesting a highly irregular function that exploited the numerical integration methods in our scoring function in just the right way, and got impossibly high scores."
- Concrete anti-cheat design (Problem 6.33): score = equidistance error divided by the square of the minimum side length, "This prevented AlphaEvolve from naive attempts to cheat by moving some points to be really close or really far apart."
- Two-speed verifiers (Problem 6.11): "we explored tradeoffs between the speed and accuracy of the verifiers - a fast and less accurate (leaky) verifier based on floating point arithmetic and a more reliable but slower verifier written using rational arithmetic." Sendov (6.9): "We had to resort to extended precision and rational arithmetic in order to define the verifier." Szemerédi–Trotter (6.36): restricted to integer lattice / rational slopes to avoid floating point, "This is not without loss of generality".
- Prompting: "prompting as in our search mode versus trying to find the construction directly resulted in more efficient programs and much better results in the former case." "Giving AlphaEvolve an insightful piece of expert advice in the prompt almost always led to significantly better results".
- "Less is more": "generalization improves when the system is provided with a more constrained set of inputs or features. ... we constrained AlphaEvolve to have access to less data by showing it the previous best solutions only for small values of n".
- Families: "Results are also significantly improved when the system is trained on correlated problems or a family of related problem instances within a single experiment. ... A search heuristic that performs well for a specific (n, d) pair will likely be a strong foundation for others".
- Limits: "AlphaEvolve excels at problems that can be clearly formulated as the optimization of a smooth score function that is possible to 'hill-climbing' on, it sometimes struggles otherwise." "for problems where genuinely new, deep insights are required to make progress, AlphaEvolve is likely not the right tool to use." They propose labelling problems "AlphaEvolve-hard".

### 3.4 Lean / AlphaProof in the loop — what was actually done
**The pipeline is post hoc, not in the reward loop.** §1.5: "for the finite field Kakeya problem (cf. Problem 6.1), AlphaEvolve discovered an interesting general construction. When we fed this programmatic solution to the agent called Deep Think [149], it successfully derived a proof of its correctness and a closed-form formula for its size. This proof was then fully formalized in the Lean proof assistant using another AI tool, AlphaProof [148]. This workflow, combining pattern discovery (AlphaEvolve), symbolic proof generation (Deep Think), and formal verification (AlphaProof), serves as a concrete example of how specialized AI systems can be integrated."

Details (Problem 6.1): "whenever AlphaEvolve found a construction that worked well on a large range of primes, we asked Deep Think to give us an explicit formula for the sizes of the sets constructed. If Deep Think succeeded in deriving a closed form expression, we would check if this formula matched our records for several primes, and if it did, it gave us some confidence that the Deep Think produced proof was likely correct. To gain absolute confidence, in one instance we then used AlphaProof to turn this natural language proof into a fully formalized Lean proof. Unfortunately, this last step was possible only when the proof was simple enough; in particular all of its necessary steps needed to have already been implemented in the Lean library mathlib." "This was only possible because the proofs typically used reasonably elementary, though quite long, number theoretic inclusion-exclusion computations." For d=4: "the proofs were too difficult for AlphaProof to handle, and since there was no exact formula for the size of the sets, we could not even cross-reference the asymptotic formula ... we had to resort to manually checking the proofs ourselves."

Result formalized: Kakeya in F_p^3, p ≡ 1 mod 4: |K| ≤ (1/4)p³ + (7/8)p² − (1/8)p − ... (explicit set given in the paper), refining the known (1/4)p³ + (7/8)p² + O(p).

**Future work (§5), verbatim:** "we imagine a further incorporation of a computer-assisted proof into the output of AlphaEvolve in the future, leading to AlphaEvolve first finding the candidate, then providing the e.g. Lean code of such computer-assisted proof to validate it, all in an automatic fashion. In this work, we have demonstrated that in rare cases this is already possible".

**[inferred]** So as of Dec 2025 the DeepMind math team explicitly describes "AlphaEvolve emits the candidate *and* the Lean proof, automatically" as an open goal — the thesis' Lean gate inside the loop is exactly this gap.

### 3.5 Cost / model ablations (§3)
- Threads: 2→20 CPU threads sped up discovery on Problem 6.2 but "doubling the threads roughly doubles the rate of LLM queries."
- Models: cheap vs strong LLM "with a price difference of roughly 15x per input token and 30x per output token". "For this simple autocorrelation problem, the most cost-effective strategy to beat the literature bound was to use the cheapest model across many runs. The total LLM cost for this was remarkably low: a few USD. However, for the more difficult problem of Nikodym sets ... the cheap model was not able to get the most elaborate constructions." "The experiments using a cheaper LLM required about twice as many calls." Mixed ensembles beat pure-strong: "a worse model generally suggests lower quality ideas, [but] it does add variance."

---

## 4. Formal verification adjacent to evolutionary discovery (2025–2026)

### 4.1 Nagda–Raghavan–Thakurta (S6): evolve the object, brute-force the certificate, evolve the *verifier* too
AlphaEvolve found a 19-node MAX-4-CUT gadget "with a complex weighting scheme (some connections having up to 1429 times the weight of others)" giving inapproximability 0.987 (was 0.9883); Ramanujan graphs "on as many as 163 nodes" (prior computer search: 10 nodes); metric TSP hardness 111/110 (was 117/116). Verification: "verifying a candidate construction produced by AlphaEvolve is costly (sometimes requiring time exponential in the size of the construction)" so "we used AlphaEvolve itself to evolve the verification procedure to be faster (sometimes by 10,000× for our gadgets)" — but "the final gadgets discovered were still verified using the original, brute-force algorithm, ensuring the absolute correctness of the theorems." "the validity of the final theorem relies on two components: the correctness of the lifting framework, and the verification of the discovered structure." **[inferred]** This is the "leaky fast verifier for search, trusted slow verifier for the claim" discipline; for us the trusted verifier is Lean, the fast one can be anything.

### 4.2 AlphaEvolve and the Erdős problems (S10)
Wiki "AI contributions to Erdős problems" lists AlphaEvolve under "AI building on literature": #36 ("Slight improvement to past construction", 3 Nov 2025), #507 ("Surpassed some past constructions"), #951 ("New solution to variant problem", 28 Jan 2026), #1097 ("Slight improvement to past construction"). All are constructions/bounds, none proofs; none were Lean-formalized. Tao's #1026 account: AlphaEvolve ran for an hour and "produced the following upper bounds on c(n)" for n ≤ 16; the numbers revealed the pattern c(k²+2a+1) = k/(k²+a), which was then proved by literature search (Baek–Koizumi–Ueoro 2024 + Praton). A separate tool (Aristotle) produced a Lean proof of the asymptotic case independently. Lesson Tao draws: tools are complementary; formalizing the full result "looks quite doable with current tools" but was not done.

### 4.3 DeepMind "AI-driven formal proof search" (S7): an AlphaEvolve-inspired evolutionary loop *whose verifier is the Lean kernel*
Four agents: (A) basic prover subagents; (B) + AlphaProof tool; (C) "an evolutionary agent (C), inspired by AlphaEvolve" with a population database of proof *sketches*; (D) both. Acceptance: "A proof is correct if it leads the compiler to a state with no pending goals." Fitness for incomplete sketches is **not** kernel-derived: "a pool of rating agents (based on the less expensive Gemini 3.0 Flash) construct relative rankings of sketches based on their plausibility, clarity, and novelty", aggregated into Elo, parents chosen by P-UCB. Models: Gemini 3.1 Pro provers. Results: "autonomously resolved 9 of 353 open Erdős problems" (all Lean statements in Formal Conjectures as of Feb 2026, 3000-episode cap) "at the inference cost of a few hundred dollars per problem"; AlphaProof "approximately 27.5 TPU hours ($60 USD) per problem on v6e TPUs"; 44/492 OEIS conjectures. After each solve "experts on our team validated that the Lean statement faithfully captured the original conjecture."

### 4.4 ProofEvolve (S8): kernel-graded partial credit
Population = "partial AND-OR proof DAGs in a behaviorally indexed archive"; across problems "kernel-checked schema extraction adds newly proved sub-DAGs to a persistent schema library." Fitness: "Turn[s] the kernel's binary verdict into a graded fitness read off the proof DAG" — verified closure ρ(D) ∈ [0,1], equal to 1 only when the root closes, computed recursively (closed node = 1; frontier = 0; AND-node = aggregate of children; OR-node = max over out-edges). "Every accepted hyperedge ... is a checked realizer" and "closed nodes stay closed". Operators: decomposition, repair ("the model receives the proposal, retrieval context, and Lean error"), schema recombination. Results: PutnamBench 71.2% (LEAP 64.7%), IMO-LeanProofBench 53.3% (36.7%), CombiBench 49.0% (50.0%); average 57.8% vs 50.5% (LEAP), 45.9% (Hilbert). Library reuse "+ about four points over zero-shot"; random retrieval gave zero improvement.

### 4.5 Others
- **FormalEvolve (S9)**: evolves *formal statements* (autoformalization) with "LLM-driven mutation, crossover, bounded patch repair, and symbolic AST rewrites", "compilation-feasible archive", LLM semantic judge; CombiBench 58.0% at 100 calls, ProofNet 84.9%.
- **AlphaProof (S11)**: RL with reward "r_t = −1 for each tactic applied", return over AND-split subgoals = minimum over branches; "every proof or disproof found by AlphaProof undergoes a final, independent verification step" running the standard Lean CLI and checking the axiom set. Numerals capped to prevent runaway kernel computation.
- **Formal Conjectures (S12)**: 2,615 Lean 4 statements (1,029 open), the benchmark S7 ran on; no evolutionary component.
- **SkyDiscover IDS (S13, Rocq not Lean)**: "jointly and incrementally synthesizes implementation and proof, and learns from failed attempts"; "7/7 in about 6.8 hours and $106 per spec on average". SkySynth test-driven mode: "an auditor turns every reward hack it finds ... into a new test."

**Bottom line for the thesis question "were Lean proofs part of any loop?":** In construction-discovery loops (S2, S3, S6, S10) — **no**; Lean appeared only after the fact (S3 Kakeya, one instance) and only when Mathlib already had the needed steps. In proof-search loops (S7, S8, S11) — **yes**, the kernel is the accept/reject signal, and S7/S8 show two different answers to "partial credit for non-compiling proofs": LLM-rater Elo (S7) vs kernel-verified closure fraction (S8).

---

## 5. AlphaEvolve on Google Cloud (S4) — the productised API

### 5.1 Positioning
Blog (9 Jul 2026): GA on the Gemini Enterprise Agent Platform; "Your client-side runner queries the AlphaEvolve API to acquire mutated candidate solutions, runs them through your client-side evaluator (which can be running anywhere), and submits the scores back to AlphaEvolve which you sample from." Two user inputs: a seed program ("You designate which segments of code are open to optimization") and "A deterministic client-side evaluation script that compiles, tests, and scores the mutated candidates, returning one or more scalar metrics for AlphaEvolve to maximize." "For agentic workflows, you can easily get started using the AlphaEvolve Skill in your IDE of choice, such as Antigravity or Claude Code." Requires a Gemini Enterprise license (any tier incl. trial, per InfoQ). Docs: "not a general-purpose developer assistant"; "does not take pure natural language descriptions or incomplete, non-functional code to output baseline functional code."

### 5.2 REST API (from the API reference page)
- `POST v1alpha/{parent}/alphaEvolveExperiments` (body `AlphaEvolveExperimentConfig`) → experiment.
- `POST v1alpha/{name}:start` (`desiredProgramsCount`) → long-running op; `:resume`.
- `POST v1alpha/{parent}:acquirePrograms` (optional `desiredProgramsCount`) → locked `AlphaEvolveProgram`s with a `lockToken`; 204 when queue empty.
- `POST v1alpha/{parent}:submitProgramsEvaluations` with `evaluationSubmissions[]`.
- `GET v1alpha/{parent}/alphaEvolvePrograms` (`stateFilter`, `orderBy`).
- Config: `title` (≤256), `problemDescription` (≤5,000 chars, required), `programmingLanguage`, `notes` (≤1,000), `runSettings{maxPrograms: default 100, 1–100,000; concurrency: default 1, 1–30; maxDuration ≤ 7 days; idleTimeout ≤ 24 h}`, `generationSettings{context (recommended < 200,000 tokens), includeFullProgramInPrompt (default false), models: [{name, weight}]}`, `evolutionSettings{parentSamplingConfig.paretoSamplingConfig.paretoSamplingProbability ∈ [0,1]}`.
- Program content: ≤ 50 files, cumulative 4,000–5,000 LOC; Python entry `initial_program.py`; must contain `# EVOLVE-BLOCK` tags (400 INVALID_ARGUMENT otherwise).
- Evaluation: `scores.scores: [{metric, score}]` (all metrics **maximised**; negate to minimise), `insights.insights: [{label, text}]` (≈ ≤ 10). 30-minute client-side lock: "Execution exceeded timeout; lock expired → Submit penalty scores; enforce 30-min limit." 429 on concurrency/quota.
- Lifecycle: experiment CREATED → RUNNING → PAUSED/COMPLETED/FAILED; program INITIALIZED → GENERATING → EVALUATING → COMPLETED.

### 5.3 Python SDK (`alpha_evolve`, Apache-2.0, repo cloned)
`AlphaEvolveClient(project_id, location, collection, engine, assistant, base_url="discoveryengine.googleapis.com")`; `AlphaEvolveExperiment(client, evaluator_function, max_programs_evaluated, parallel_evaluation=False)`; `run_controller_loop(...)` (async; `idle_timeout_s`). The evaluator is "A function (sync or async) that takes a program candidate (dict) and returns evaluation results (dict)"; a flat `{metric: value}` dict is auto-wrapped by `_wrap_result` into `{"scores": {"scores": [{"metric": k, "score": v}]}}`. Candidate code is at `program_candidate["content"]["files"][0]["content"]`. The circle-packing example uses sentinel `score_value = -1e12` on failure "so a large negative value keeps failed candidates from being selected", and attaches `AlphaEvolveEvaluationInsight(label="Runtime Error", text=...)`. README: failures return "a sentinel score ... **plus an insight message**. Those insights are fed back to Gemini to steer the next generation, so failures are part of how the search improves." Env: `MODEL_1=gemini-3.5-flash` (0.7), `MODEL_2=gemini-3.1-pro-preview` (0.3; global location only), `MAX_PROGRAMS_GENERATED/EVALUATED=10`, `CONCURRENCY=4`, `WORKER_CONCURRENCY=4`, `PARALLEL_EVALUATION`. Six "skills" (`alpha_evolve_experiment_design`, `_runner`, `_monitor`, `_post_experiment`, `_orchestrator`, `_consultant`) + `ae` CLI; "Python Only", "Single Code Location Only" tested.

### 5.4 Best practices page (paraphrase/quotes)
"The evaluator must return a structured dictionary of one or more metrics to provide an explicit search gradient direction." "Recommended evaluation budgets start at ~100 programs and scale up to thousands for hard problems." "If your evaluation metrics fail to show a clear trajectory after ~1500 candidate program evaluations, pause the execution loop to refine your problem definition context or adjust soft penalty constraints." "Hard problems can plateau for an extended number of generations before a sudden breakthrough, so a plateau does not necessarily mean the search has converged."

### 5.5 Costs
Not published: "Pricing is not disclosed in the announcement" (InfoQ). Repo: "AlphaEvolve and Gemini usage are billed to your Google Cloud project per your agreement. Use the search-budget knobs (MAX_PROGRAMS_*, CONCURRENCY) to bound cost." Codelab: billable = "AlphaEvolve API calls + Vertex AI Gemini tokens per generated candidate", no GPU charges for local evaluation; codelab examples take 45–60 min. Case studies in the blog: Klarna "nearly 6,000 candidate programs"; ODU "approximately 500 evaluations". **[inferred]** For portability planning, assume cost ∝ (#programs) × (tokens per prompt incl. `context`) at Gemini 3.5 Flash / 3.1 Pro rates plus an undisclosed per-agent charge.

---

## 6. SkyDiscover (S5)

- What it is: "an open-source framework from UC Berkeley that uses AI to discover better algorithms and build complete systems." Two products: **Optimize** ("You provide a way to score solutions, and evolutionary search finds programs that score better and better") and **Synthesize** (coding agents build systems "correct by proof" (Rocq, IDS) "or by test"). Launch blog 3 Mar 2026; AdaEvolve blog 12 Mar; EvoX blog 17 Mar; SkySynth 3 Sep 2026; EvoX accepted at COLM 2026.
- Claims (README, self-reported): "Frontier-CS (172 problems): ~34% higher median score than OpenEvolve, GEPA, and ShinkaEvolve under the same budget"; "Math + systems (14 tasks): matches or exceeds AlphaEvolve and human SOTA on 6/6 systems and 6/8 math tasks"; "41% lower cross-cloud transfer cost, 14% better GPU load balance for MoE serving, 29% lower KV-cache pressure". Note the implicit admission: 2 of 8 math tasks were *not* matched. The math benchmark folder is 12 problems from AlphaEvolve Appendices A/B (matmul, three autocorrelation inequalities, uncertainty, Erdős min overlap, sums/differences, hexagon packing, min/max distance ratio, two Heilbronn variants, circle packing in rectangle) plus circle packing n=26.
- Algorithms: AdaEvolve ("Multi-island adaptive search with UCB, migration, and paradigm breakthroughs"; paper: local adaptation of exploration intensity, global bandit routing of budget across populations, meta-guidance when stalled, unified by an "accumulated improvement signal"; "consistently outperforms the open-sourced baselines across 185 different open-ended optimization problems"), EvoX ("jointly evolves candidate solutions and the search strategies used to generate them" — meta-evolution; "outperforms ... AlphaEvolve, OpenEvolve, GEPA, and ShinkaEvolve on the majority of tasks" across ~200), plus Top-K, Beam, Best-of-N, GEPA-native, **OpenEvolve-native** ("MAP-Elites + island-based evolutionary search"), coding-agent baseline; external backends `--search openevolve | gepa | shinkaevolve | alphaevolve` (the last "needs GCP"). "incorporates useful code components from open-source efforts such as OpenEvolve"; interface "compatible with the optimize_anything API".
- Evaluator contract (Optimize guide): `def evaluate(program_path): return {"combined_score": score, "artifacts": {"feedback": "..."}}` — "combined_score drives evolution. If omitted, SkyDiscover averages all numeric values in the dict." "artifacts ... entries are injected into the next LLM prompt as context." AdaEvolve supports Pareto objectives via `search.database.pareto_objectives`. Also containerised (`Dockerfile` + `evaluate.sh` → JSON) and Harbor formats. `EVOLVE-BLOCK-START/END` markers; if absent "the entire file is treated as mutable"; initial program optional.
- Formal verification: none in Optimize; Synthesize has a Rocq-based proof-driven mode (IDS) and a test-driven mode with an auditor that converts detected reward hacks into tests.
- **[inferred]** Our OpenEvolve evaluator (`evaluate(program_path) -> dict` with `combined_score` + artifacts) is byte-compatible with SkyDiscover-Optimize, so AdaEvolve/EvoX are a free second search backend for ablations.

---

## 7. Other open-source loops worth knowing (S13)
- **ShinkaEvolve** (Sakana, Sep 2025): parent sampling balancing exploration/exploitation, "code novelty rejection-sampling", "bandit-based LLM ensemble selection"; "discovers a new state-of-the-art circle packing solution using only 150 samples".
- **ThetaEvolve** (Nov 2025): AlphaEvolve simplified + RL on the sampling model at test time; single small open model (DeepSeek-R1-0528-Qwen3-8B), "lazy penalties to discourage stagnant outputs, and optional reward shaping".
- **LEVI** (May 2026): "Stronger search architectures can substitute for or even outperform larger LLMs in evolutionary search"; diversity-first database, mutation router across large/small LLMs, rank-preserving proxy benchmark; "3.3–6.7x smaller" budgets, one problem at "35x lower cost".
- **BaSE** (May 2026): bandit allocation of LLM calls across parallel trajectories; "+12.3% mean fitness over the strongest island-protocol baseline" from allocation alone; warns single runs do not reproduce multi-run reports.
- **CodeEvolve** (Oct 2025): islands GA + modular LLM orchestration.

---

## 8. Synthesis for our design

### 8.1 Reward definitions used by the field (what each loop maximises)
| Loop | Score | Failure handling | Multi-objective |
|---|---|---|---|
| FunSearch | size of object (e.g. cap set) per input n, aggregated (mean / last test) | program discarded if it errors or violates property | no |
| AlphaEvolve | `evaluate` → dict of scalars; for search mode, "the score of the heuristic is the score of the best object it finds" within a time budget, starting from the previous best | any invalid object → not registered / −∞ | yes, MAP-Elites bins + Pareto |
| Tao–Wagner | same, but insists on continuous scores, exact (rational) verifiers for claims, anti-cheat normalisation | −∞ for illegal configs (avoided in favour of continuous penalties) | yes |
| Cloud AlphaEvolve | list of `{metric, score}` all maximised, plus `insights` text | sentinel (e.g. −1e12) + insight; 30-min lock → penalty | Pareto sampling probability knob |
| SkyDiscover | `combined_score` (+ optional Pareto objectives) + `artifacts` | error → low score + artifact feedback | AdaEvolve Pareto |
| S7 (formal proof search) | kernel accept = solved; Elo from LLM raters for unfinished sketches | no partial credit from kernel | no |
| ProofEvolve | verified closure ρ(D) ∈ [0,1] read off kernel-checked DAG | partial credit is *only* for kernel-checked sub-goals | no |
| AlphaProof | −1 per tactic; min over AND-branches; final CLI+axiom check | timeout | no |

### 8.2 Verification patterns
1. **Execution-grounded exact check** (S1, S2): the evaluator recomputes the property exactly (integers / half-integers / rationals). Works for constructions; not for a *prune*, whose correctness is a universal statement over all matrices in a case.
2. **Fast leaky verifier for search, slow trusted verifier for the claim** (S3 §6.11, S6): the reward may use cheap approximations; the theorem only uses the trusted check. For us: cheap empirical falsification (random / known extremal matrices, LP relaxation, brute force on tiny (m,n)) during search; Lean (`Prune P` elaboration) before anything counts as pruned.
3. **Post-hoc formalisation pipeline** (S3 Kakeya): discover → informal proof (Deep Think) → Lean (AlphaProof); only worked when Mathlib had every step. Our ZarPrune is Mathlib-free, so the "library coverage" bottleneck becomes "does our hand-rolled `sumFin`/`HasKst` layer have the lemma?" — the KST double-counting layer flagged in `lean/README.md` is exactly the missing step.
4. **Kernel in the loop with graded credit** (S7, S8): both accept only kernel-closed proofs; they differ on partial credit. S8's verified closure is the safer choice because credit is only given for what the kernel has already accepted.
5. **Statement-fidelity audit** (S7, S12): after a Lean success, humans checked that the Lean statement matched the conjecture. For us: the `Prune P` type fixes the statement, so fidelity reduces to (a) trusting `ZarPrune/Basic.lean` definitions and (b) the `cover` obligation.

### 8.3 Genome representations for combinatorics — mapping to prune evolution
The field's ladder (S2 §2.1): raw object → constructor program → search program → co-evolved (object, improver). For *pruning arguments* the "object" is a pair (kill predicate, soundness proof). **[inferred]** candidate genomes:
- **G1 constructor mode**: evolve `kill : Profile → Bool` directly as Lean (or Python mirrored to Lean) plus a proof term. Interpretable, matches the thesis' interpretability goal (cf. FunSearch's 512-cap explanation), but each LLM call yields one candidate and the proof must compile — low throughput, high failure rate.
- **G2 search mode (Tao–Wagner "improver" functions)**: evolve a Python program that, given the partition census and a time budget, *searches a parametric family of counting inequalities* (e.g. Σ_i f(r_i) ≤ g(n,s,t) with f ranging over binomial/threshold families, weighted variants, row/column duals, restricted to sub-blocks), tests each against a counterexample bank (known valid matrices, SAT-found matrices), and returns the best surviving instance ranked by pruned mass. The evaluator then instantiates a fixed **Lean template** for that family and elaborates it. This gives the "one slow LLM call → millions of cheap checks" leverage that S3 says was decisive, while keeping Lean out of the inner loop.
- **G3 generalizer mode**: score the same program across several (m,n,s,t) instances; show it only small instances ("less is more"); reward generality. This is where cross-instance theorems (rather than instance-specific prunes) would come from.
- **G4 co-evolution**: evolve (lemma family template in Lean, search program over its parameters) together — a family is admitted only once its template proof elaborates for symbolic parameters; then G2 explores parameters at zero Lean cost. **[inferred]** This is the most promising for cost: Lean is paid once per family, not once per candidate.

### 8.4 Portability to the Cloud AlphaEvolve API (what to keep invariant in the OpenEvolve build)
- Evaluator = pure function of program text → `dict[str, float]` (all higher-is-better) + a short list of `{label, text}` insights; OpenEvolve's `EvaluationResult(metrics, artifacts)` maps 1:1 (artifacts → insights, ≤ ~10, short). SkyDiscover's `combined_score` + `artifacts` is the same shape.
- Keep `# EVOLVE-BLOCK-START/END` markers, a single Python entry file (`initial_program.py`), ≤ 5,000 LOC total, ≤ 50 files.
- Evaluator must finish in < 30 min wall-clock and return a penalty score on timeout; never raise. Lean and the SAT solver run client-side in the evaluator (the Cloud never sees them), so the whole Lean toolchain must be a local dependency of the evaluator, not of the framework.
- No cascade in the Cloud API: OpenEvolve's `evaluate_stage1/2/3` must be reproducible as *one* `evaluate` that internally short-circuits (parse → run kill on census → falsify → Lean → SAT proxy) and reports stage reached as a metric.
- Multi-objective: Cloud uses Pareto parent sampling by probability; OpenEvolve uses MAP-Elites feature dimensions — choose ≤ 3 numeric features (e.g. `pruned_mass`, `lean_ok`, `generality`) so the same dict works in both.
- `problemDescription` ≤ 5,000 chars and `context` ≲ 200k tokens: write the problem statement / Lean API docs once as a context file used by both OpenEvolve's system prompt and the Cloud `generationSettings.context`.
- Models: Cloud offers `gemini-3.5-flash` / `gemini-3.1-pro-preview` mixtures; OpenEvolve's ensemble config can mirror the 0.7/0.3 weights via OpenRouter equivalents.

### 8.5 Difficulty signals (for "how hard is a branch" in the reward)
From the sources: (i) SAT-style: S6 verification cost is "sometimes requiring time exponential in the size of the construction" — measured verifier time is itself the signal; (ii) S3 proposes classifying problems by resistance to search ("AlphaEvolve-hard"); (iii) S4 best practices use trajectory-after-N-evaluations as a difficulty indicator. **[inferred]** for SAT branches: (a) pilot solve time / timeout rate per partition pair; (b) number of free variables after unit propagation and clause count of the branch instance; (c) count of profile pairs the branch represents (mass) weighted by prior solve-time model; (d) LP/counting slack (how far Σ C(r_i,t) is from the KST bound); (e) cube-and-conquer lookahead heuristics (AlphaMapleSAT — see `alphamaplesat.md`).

### 8.6 Lemmas / statements extracted from these sources
These sources contain **no Zarankiewicz pruning lemmas**. Two verification-related statements are worth recording as templates:

**Lemma (AlphaEvolve Appendix B, Lemma 1, verbatim).** "Let C ⊂ ℝ^d be a set of points satisfying 0 ∉ C and min{‖x − y‖ : x ≠ y ∈ C} ≥ max{‖x‖ : x ∈ C}. Then unit spheres centred at {2x/‖x‖ : x ∈ C} form a valid kissing configuration in dimension d. In particular, the kissing number in dimension d is at least |C|." Role: converts an integer certificate into a theorem with a one-line proof — the template for "certificate + trivial checking lemma". Provability in Mathlib-free Lean: not applicable (needs reals/norms); the *pattern* (integer certificate, decidable check, short soundness lemma) is what `Prune.sound` already is.

**Search-mode soundness (Tao–Wagner, paraphrased as a statement).** If the evaluator scores a heuristic by "the score of the best object it finds", the reported bound is only as sound as the object checker; the heuristic itself needs no verification. Consequence for us: only the *kill predicate + Lean proof* needs to be trusted; the search program that found it can be arbitrary Python.

---

## 9. Open questions raised by this survey
1. No published loop puts a construction-discovery evaluator and a Lean checker in the same reward; S3 §5 names it as future work. What is the compile-time budget per candidate that keeps generation rate acceptable (S3 shows 20 threads ≈ 10× faster discovery; Lean adds seconds per candidate; `lake build` of ZarPrune is < 1 s but candidate elaboration will vary)?
2. Partial credit: S7 (LLM-rater Elo) vs S8 (kernel-verified closure). Which transfers to a `Prune P` term whose `sound` field is a single proof — count closed `have` steps? Elaborate a decomposition into named sub-lemmas so closure is measurable?
3. S3's "cheating phenomenon" — the Lean gate removes unsound kills, but can the search still hack *difficulty* proxies (e.g. claim mass on branches that were trivially UNSAT for the baseline)? Reward must be measured relative to `baseline` prune (deficit/mismatch/caps) and against a fixed census.
4. Generalizer mode showed only small n; for us "small instances" means small (m,n) with known z — but a prune sound for small instances need not be sound in general; Lean must be over symbolic `Params`. Does the LLM produce parametric proofs at usable rates?
5. Cost: S3 got literature-beating results for "a few USD" with cheap models on an easy problem, and could not on a hard one. With a $16 OpenRouter budget, the realistic target is validating the loop mechanics (prompt → candidate → falsifier → Lean) on a few dozen samples, not a bound.
6. Cloud API has no cascade and a 30-min evaluator limit; SAT solving of survivors must therefore be a separate offline stage, with only proxies inside `evaluate`.

---

## 10. Citation strings
- Romera-Paredes, B., Barekatain, M., Novikov, A., et al. Mathematical discoveries from program search with large language models. *Nature* 625, 468–475 (2024). doi:10.1038/s41586-023-06924-6. Code: github.com/google-deepmind/funsearch.
- Novikov, A., Vũ, N., Eisenberger, M., et al. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131 (2025).
- Georgiev, B., Gómez-Serrano, J., Tao, T., Wagner, A. Z. Mathematical exploration and discovery at scale. arXiv:2511.02864v3 (2025).
- Google Cloud. AlphaEvolve is available for everyone. Google Cloud Blog, 9 Jul 2026. https://cloud.google.com/blog/products/ai-machine-learning/alphaevolve-is-available-for-everyone ; docs: https://docs.cloud.google.com/gemini/enterprise/docs/alphaevolve/developer-guide/overview ; API: .../alphaevolve/reference-guide/api-reference ; SDK: https://github.com/Google-Cloud-AI/alphaevolve-on-googlecloud ; Codelab: https://codelabs.developers.google.com/alphaevolve-on-google-cloud-1.
- Wiggers, S.-J. Google's AlphaEvolve Reaches General Availability... InfoQ, 19 Jul 2026.
- Liu, S., Cemri, M., Agarwal, S., et al. SkyDiscover: A Flexible, Adaptive Framework for AI-Driven Scientific and Algorithmic Discovery. *CAIS '26*, 1223–1227. doi:10.1145/3786335.3813221. https://github.com/skydiscover-ai/skydiscover ; https://skydiscover-ai.github.io/.
- Cemri, M., et al. AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization. arXiv:2602.20133 (2026).
- Liu, S., et al. EvoX: Meta-Evolution for Automated Discovery. arXiv:2602.23413 (2026); COLM 2026.
- Nagda, A., Raghavan, P., Thakurta, A. Reinforced Generation of Combinatorial Structures: Hardness of Approximation. arXiv:2509.18057 (2025/2026). Blog: research.google/blog/ai-as-a-research-partner-advancing-theoretical-computer-science-with-alphaevolve/ (30 Sep 2025).
- Tsoukalas, G., Kovsharov, A., Shirobokov, S., et al. Advancing Mathematics Research with AI-Driven Formal Proof Search. arXiv:2605.22763 (2026).
- Ye, W., Guan, Z., Xie, E., et al. ProofEvolve: Neuro-Symbolic Evolution for Formal Automated Theorem Proving. arXiv:2608.26334 (2026).
- Lu, H., Wang, W., Liu, J. FormalEvolve: Neuro-Symbolic Evolutionary Search for Diverse Autoformalization. arXiv:2603.19828 (2026).
- Hubert, T., Mehta, R., et al. Olympiad-level formal mathematical reasoning with reinforcement learning. *Nature* 651, 607–613 (2025). doi:10.1038/s41586-025-09833-y.
- Firsching, M., et al. Formal Conjectures: An Open and Evolving Benchmark for Verified Discovery in Mathematics. arXiv:2605.13171 (2026).
- Tao, T. The story of Erdős problem #1026. terrytao.wordpress.com, 8 Dec 2025; wiki: github.com/teorth/erdosproblems/wiki/AI-contributions-to-Erdős-problems.
- Lange, R. T., Imajuku, Y., Cetin, E. ShinkaEvolve. arXiv:2509.19349 (2025). Wang, Y., et al. ThetaEvolve. arXiv:2511.23473 (2025). Tanveer, T. LEVI. arXiv:2605.09764 (2026). Xing, S., et al. Compute Allocation for Self-Evolving LLMs (BaSE). arXiv:2605.29268 (2026). Agarwal, S., et al. Inductive Deductive Synthesis. arXiv:2605.23109 (2026).

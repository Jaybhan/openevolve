# Thesis proposal, Section 2 (verbatim from Jay Bhan's MEng proposal, advised by Srinivasan Raghuraman)

## 2.1 Context
In my prior research, we established lower bounds for the Zarankiewicz numbers by constructing certificates. For my Master's thesis, I take on the more daunting task of establishing upper bounds, again using an evolutionary search technique.

Discovering upper bounds requires more work, as a simple certificate won't suffice. Tan explored representing the Zarankiewicz Z(m, n, s, t) problem in CNF and using a SAT solver to determine whether a solution exists [8]. Variables in the CNF represent whether each matrix entry is a one or zero, and clauses represent the constraint of preventing a 3 x 3 subgraph. However, this encoding requires thousands of clauses even for a relatively small case of the Zarankiewicz numbers, and running it takes an unacceptable amount of compute time and resources. Tan found optimizations to increase the process's speed by splitting the algorithm into different branches, each with an enforced sum for each row and column of the matrix. He then mathematically proved that some of these branches could be pruned immediately, leading to drastic increases in speed.

## 2.2 Goal
**The primary goal of this project is to use evolutionary search to prune more branches effectively and improve the SAT solver's ability to discover novel upper bounds in an acceptable amount of time.** The exact specifications and algorithmic design will change as experiments progress, but we have a few primary directions that we wish to explore. First and foremost, pruning a branch requires formal verification to ensure its correctness. As discussed above, frontier models now have the ability to write in Lean. This step is absolutely essential, as the worst possible outcome is claiming upper bounds that are not mathematically sound.

## 2.3 Task Breakdown
The first step of this project is figuring out exactly how formal verification can play a role in evolutionary search. To the best of my knowledge, this remains an open problem. For example: does Lean take too long to compile? Can LLMs operate in Lean directly, or would it be preferable to design a two-step process, one which operates in natural language and a second that converts that text to Lean?

Next, we have to design the reward function. In last year's lower bound work, the reward function focused primarily on the number of edges in the proposed output graph, as well as whether the graph contained any violations. For upper bounds, the problem becomes more complicated. We need some measurement of how many branches the model would prune and estimate the "difficulty" of the section of the problem we eliminated. What does the "difficulty" of a branch actually mean? How can we trade off exploration and exploitation? LLMs are only just becoming competent at coding in Lean, so how can we reward a Lean function that comes close but doesn't compile perfectly? What might proxy rewards look like?

Finally, we have to understand how to run this in practice. Models capable of significant novel algorithmic thought on their own require significant time and compute to operate, while an evolutionary search model works in small bursts. How can we tune model choice and reasoning levels to ensure success without sacrificing time and cost? For each potential new upper bound, we need a new CNF instance (requiring a specific number of variables to be set to True). Do we run one SAT solver at a time or several? How often do we run the SAT solver itself to check if we've pruned well enough to establish a new bound versus relying on proxies? If we start with a specific Z(m, n, s, t), how can we see if and when our work generalizes to other cases of the problem?

## 2.4 Completed Work
As mentioned, I completed my SuperUROP project on lower bounds of the Zarankiewicz Numbers [9]. For the upper bound component, I have prototyped a sandbox for Lean, which takes in a proposed Lean proof for the validity of a pruned branch and outputs whether or not it is valid. (This is `upper_bounds/lean/`, the ZarPrune library: Lean 4.34, no Mathlib, `Prune P` structure with a `kill` predicate on row/column-sum profiles and a `sound` proof obligation.)

## Related work cited (Section 3)
[8] Tan, An attack on Zarankiewicz's problem through SAT solving, arXiv:2203.02283 (PDF copy in docs/papers/).
[9] Bhan, Nobili, Raghuraman, Langer, New Bounds for Zarankiewicz Numbers via Reinforced LLM Evolutionary Search, arXiv:2605.01120.
[11] Padhi, New Exact Values and Improved Lower Bounds for Zarankiewicz Numbers z(m,n;3), SSRN 6960039 (June 2026).
[12] Saurabh, Five improved lower bounds for Zarankiewicz numbers z(m,n;3,3), arXiv:2608.26603.
[13] Hou, Seven Exact Finite Zarankiewicz Numbers from a Single 13 x 18 Core, arXiv:2608.08549.
[14] Afrasyab, Exact Zarankiewicz Values On Two Finite Frontier Slices, arXiv:2608.08154.
[15] dfield, finite-zarankiewicz-closures, https://github.com/dfield/finite-zarankiewicz-closures — splits the matrix into branches by column sum, prunes them, uses Lean to confirm prunings are valid.
[16] Jha et al., AlphaMapleSAT: MCTS-based cube-and-conquer, arXiv:2401.13770.
[17] Chivilikhin, Pavlenko, Semenov, Decomposing Hard SAT Instances with Metaheuristic Optimization, arXiv:2312.10436.
[18] Davies, Gill, Horsley, Improved upper bounds on Zarankiewicz numbers, Discrete Math 349 (2026), arXiv:2411.18842.
[19] AlphaEvolve on Google Cloud (July 2026). [20] SkyDiscover (Berkeley Sky Lab, Mar 2026).
[1] Novikov et al., AlphaEvolve, arXiv:2506.13131. [2] Hubert et al., AlphaProof, Nature 651 (2026).

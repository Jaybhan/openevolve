# AlphaMapleSAT: An MCTS-based Cube-and-Conquer SAT Solver for Hard Combinatorial Problems

Literature notes for the upper-bounds thesis (Zarankiewicz z(m,n;s,t) via pruned case-split SAT + Lean-verified prunes).
Written 2026-09-21. Primary text read in full (11-page PDF, arXiv v2) plus both public code repositories.

## 1. Bibliographic record

| Field | Value |
|---|---|
| Title | AlphaMapleSAT: An MCTS-based Cube-and-Conquer SAT Solver for Hard Combinatorial Problems |
| Authors | Piyush Jha (Georgia Tech, equal contrib.), Zhengyu Li (Georgia Tech, equal contrib.), Zhengyang Lu (Waterloo), Raymond Zeng (Georgia Tech), Curtis Bright (Windsor), Vijay Ganesh (Georgia Tech) |
| arXiv | 2401.13770, cs.AI (cross-list math.CO). v1: 24 Jan 2024. v2: 20 Jan 2026 (the version read here). |
| Venue | arXiv preprint (no journal/conference venue stated in v2) |
| Code | Cubing tool: https://github.com/piyush-J/AlphaMapleSAT (MIT). Parallel CnC pipeline: https://github.com/BrianLi009/AlphaMapleSAT-CnC (MIT). Both cloned and read. |
| Local copies | PDF: scratchpad `lit/alphamaplesat.pdf`; repos: scratchpad `lit/AlphaMapleSAT`, `lit/AlphaMapleSAT-CnC` |

Version drift worth recording: the v1 abstract (Jan 2024) claimed "up to 2.3x speedup in parallel (and up to 27x in sequential) elapsed real time" on Kochen-Specker and Ramsey. v2 (Jan 2026) adds the Murty-Simon (diameter-2-critical) benchmark, SMS as a second conquering solver, KS order 22, cube-level and core-scaling analyses, and claims "1.61x to 7.57x on a 128 core machine". The sequential claim is gone from v2.

## 2. One-paragraph summary

AlphaMapleSAT (AMS) replaces the greedy lookahead splitting heuristic of `march_cu` with a Monte Carlo Tree Search over the cubing tree. The reward is not learned and not human-designed per domain: it is a *deductive* signal, the propagation rate obtained by running Boolean Constraint Propagation (BCP, via PySAT/MiniSat `propagate`) on the formula under the candidate cube. MCTS uses PUCT selection with priors set from normalized propagation scores of the top few variables, expands both polarities of a chosen variable, terminates rollouts early (no solving, no random rollouts) and backs up the *average* of the two children while tracking the *maximum* observed value per action, which is what the final split decision uses. Cubing stops when a user-set number of variables has been eliminated (`-n`) or a depth is reached (`-d`). Against `march_cu` on KS orders 19-22, Ramsey R(3,8), and D2(15,56), with SAT+CAS or SMS as conquering solver on 128 cores, elapsed-time speedups are 1.61x-7.57x; on KS 22 march's cubes time out after 5 days while AMS cubes are solved in 16.34 h (SAT+CAS) / 10.01 h (SMS). The paper's explanation, backed by per-cube timing histograms, is that AMS produces far fewer cubes and, more importantly, far fewer *very hard* cubes (> 5000 s), and the hard tail dominates wall-clock time.

## 3. Problem formalization (Section 3 of the paper, quoted)

- Def 1 (Cube): "A cube is a conjunction of literals, e.g., x_1 ∧ ... ∧ x_n, where each x_i is a literal from the input formula."
- Def 2 (Propagation rate): "Given a formula F and a cube c, the propagation rate of F ∧ c is defined as the ratio between the number of propagations performed by a Boolean Constraint Propagation (BCP) on F ∧ c and the size of c."
- Def 3 (Lookahead heuristics): iterate over candidate variables, "simplify the formula with respect to the given cubes, and probe them to decide the best variable to split on. Probing is the process of computing quality metrics (e.g., propagation rate) by running a SAT solver on the sub-formulas thus obtained."
- Def 4 (Cubing problem): output a set C = {C_1..C_k} of cubes over {x_1..x_n}; sub-formulas F ∧ C_i.
- Def 5 (Splitting tree): full binary tree, node = subformula, edges = True/False assignment to a split variable; root-to-leaf path = cube.
- Def 6 (Cubing solver): explores cubing trees "to generate an effective set of cubes -- ideally, those that minimize the total solving time when dispatched to conquering solvers."
- Problem statement (Sec. 1): choose cubes "such that the total CPU and elapsed real time for cubing and solving F is minimized." Total elapsed real time = time for the parallel CnC solver to produce a SAT/UNSAT result (footnote 1; formalized in Sec. 5.5).
- MDP framing (Sec. 4.2): "We formulate cubing as a deterministic Markov Decision Process (MDP), more specifically as a tree MDP [Scavuzzo et al., 2022]. Each node corresponds to a CNF formula, and actions represent variable splits. ... The reward for each node is the propagation rate, which is a solver-derived metric."
- Termination (Sec. 4.2): "user-defined parameter n_elim, also known as the variable cutoff value, which denotes that the splitting process at a particular node must stop if at least n_elim variables have been eliminated (through propagation or as part of splitting variables) in the cube. This choice ... is motivated by the objective of attaining balanced cubes."

## 4. The algorithm, precisely

### 4.1 As stated in the paper (Sec. 4.3-4.4)

Cubing episode: DFS from the root; at each node run an MCTS simulation (budget 10 simulations, "chosen ... using light tuning on a smaller KS instance and then fixed for all benchmarks") to pick a variable x_i; create children F ∧ x_i and F ∧ ¬x_i; recurse until the termination condition; emit root-to-leaf paths as cubes.

Selection (PUCT): for each valid x_i compute propagation rates of both polarities by BCP and combine with march's formula

    prop(x_i) · prop(¬x_i) + prop(x_i) + prop(¬x_i)

normalize over candidates to get the prior P(s,a); then

    (1)  a_chosen = argmax_a ( Q(s,a) + u(s,a) )
    (2)  u(s,a)   = c_puct · P(s,a) · sqrt( Σ_b N(s,b) ) / ( 1 + N(s,a) )

"Q(s,a) is the expected reward of action a, N(s,a) is the visit count, and c_puct controls the exploration-exploitation trade-off."

Expansion: unvisited action a at non-terminal s creates both children s ∧ a and s ∧ ¬a.

Rollout: "Instead of relying on random rollouts (which are expensive in SAT due to repeated BCP calls), AlphaMapleSAT terminates simulations early and leverages deductive rewards available at intermediate nodes. That is, even non-terminal nodes can be evaluated based on how many variables have been eliminated via propagation."

Backup: "We compute the value at each non-leaf node as the average of its children, i.e., values from the true and false branches."

Final action: "a* = argmax_a B(s,a). Here, B(s,a) tracks the maximum (long-term) reward observed for action a at state s."

### 4.2 As implemented (from `alphamaplesat/` source; this is what actually ran)

The code is a stripped alpha-zero-general skeleton (`Coach.py`, `MCTS.py`, `ksgraph/KSGame.py`, `ksgraph/KSLogicMode0.py`, `ksgraph/EvalVarCalc.py`), run in `MCTSmode 0` (no neural net; the NN code paths are dead). Notation: π = cube so far (`prior_actions`), k = |π| = depth, L = learnt failed-literal negations, m = `-m` (only variables 1..m are cubing candidates; in KS instances these are the edge variables), Lit_m = literals over vars 1..m.

Per-literal propagation count (`MarchPysatPropagate.propagate`):

    r(ℓ) = | asgn( F ∧ π ∧ L ∧ ℓ ) ∩ Lit_m |        (assigned *cubing* variables only, via PySAT minisat22 propagate)

Failed-literal elimination is interleaved: if propagate(π ∧ L ∧ ℓ) is UNSAT, ¬ℓ is appended to L and the whole candidate sweep restarts; L is cached per prefix π. Candidates are the "free" variables among 1..m, defined as those occurring in some binary clause (fallback: all of 1..m).

Per-variable score (this is the paper's march-style combination, with the "rate" realized as division by decisions-so-far+1):

    score(v) = r(v)·r(¬v) / (k+1)^2  +  r(v)/(k+1)  +  r(¬v)/(k+1)

Normalization constant `max_metric_val` = n_elim^2 if `-n` is given, else (m // 4)^2 ("crude estimate", per source comment). All scores are divided by it, so values are nominally in [0,1].

Prior P(s,·): only the top-3 variables by score (`LIMIT_TOP_3 = True`) get nonzero prior; both literals of a variable share its score; renormalized to sum 1. So the effective branching factor is 3 variables (6 actions) per node, not m.

Leaf value on expansion: `v = board.total_rew`, where `total_rew` of a child is *the parent's normalized score of the variable that was split to create it*. That is the "deductive reward available at intermediate nodes".

Terminal states inside MCTS (`getGameEndedMCTS`): with `-d` set and `-nMCTSEndOfG` unset, MCTS is allowed to run to depth d + 20 (`STEP_UPPER_BOUND_MCTS = 20`); with `-nMCTSEndOfG` set, it is the n_elim cutoff instead. A state is terminal ("giveup") when assigned cubing vars >= n_elim, or depth >= bound, or no legal literals remain; its reward is `total_rew`. Refuted by BCP: reward 0.1 (a small positive constant "to avoid best values to be 0, as 0 is also for the illegal moves"). "Unknown" would be -1 (unused in mode 0). If a satisfying assignment is found the tool prints "Found SAT!" and `exit(0)`.

Backup (`MCTS.search`): both children are always visited in the same simulation, then

    v = (v1 + v2)/2  -  varpen · |v1 - v2|         (varpen default 0; optional balance penalty)
    Q(s,a) <- running mean of v ;  B(s,a) <- max(B(s,a), v) ;  N(s,a) += 1

and the identical update is applied to (s,¬a). Final choice at temp=0: argmax over B (ties broken uniformly at random). Defaults: `numMCTSSims = 10`, `cpuct = 10`.

Outer loop (`Coach.DFSUtil`): at each real node run `getActionProb` (10 simulations), take the argmax-B variable, recurse on both children; leaves are those where `getGameEnded` (with the real `-n`/`-d` cutoff) fires; *all* leaves are written to the cube file as `a <lits> 0`, including ones refuted by BCP (the pipeline's simplification step then discards them via `c exit 20`).

Terminology caveat: the paper says "propagation rate" (propagations / |c|); the code counts assigned *cubing* variables (not all propagations) and divides each side by (k+1). Treat the formulas above as the ground truth of what was benchmarked.

### 4.3 The parallel pipeline it runs inside (`AlphaMapleSAT-CnC/parallel-solve.py`, MathCheck-style)

Iterative cube-simplify-solve with a work queue:
1. Simplify the current (sub)instance with CaDiCaL for 10,000 conflicts (`simplify-by-conflicts.sh ... 10000 [-cas|-sms]`), writing `.simp` and an extension `.ext` file. If the log contains `c exit 20` the cube is UNSAT and dropped.
2. Count the number of cubing variables eliminated (`var_removed`, from `.ext`, vars <= m).
3. Cutoff mode `v` (variables) or `d` (depth): if `var_removed >= cutoffv` (or depth >= cutoffv) hand the cube to the conquering solver with a timeout; otherwise cube it further with AMS (`-d 1 -m m -numMCTSSims s`) or `march_cu -d 1 -m m`, and enqueue the children.
4. If the conquering solver times out, the cube is sent back for further cubing with an *extended* cutoff (`cutoffv = var_removed + 20` in mode v, `+5` in mode d).

Step 4 is the only place hardness is measured empirically; everything else is propagation-based. Paper, Sec. 5.4: "If a worker solver exceeds a specified time limit when attempting to solve a subproblem, that subproblem can be sent back for further cubing, effectively subdividing it into easier instances."

## 5. march_cu, for comparison (read from the bundled source in AlphaMapleSAT-CnC/gen_cubes/march_cu)

Branch variable: `diffScore = left*right + left + right` with `left = 1024*WNBCounter[v]`, `right = 1024*WNBCounter[-v]` where WNB = weighted new binaries produced by the lookahead (each reduced clause weighted by `size_diff`: `size_diff[2] = H_BIN = 25`, `size_diff[i] = size_diff[i-1]*H_DEC`, `H_DEC = 0.5`; `H_MIN 8`, `H_MAX 550`), plus preselection, single-look iterations `SL_ITER 9`, double-look `DL_ITER 2`, global autarky heuristic (`GAH` on), windfall resolvents, both-implications. So march's score is the same product-plus-sum shape as AMS's, but over *weighted new binary clauses* rather than a raw count of assigned variables, and computed with much heavier lookahead machinery in C.

Cutoff (solver.c line 551):

    if ((cut_depth && depth == cut_depth) || (dynamic && freevars < free_th) || (cut_var && initial_freeentryvars - freeentryvars > cut_var))  -> emit cube

`-d` static depth; `-n` "# of free vars to remove" (counting only vars <= `-m`); with neither, dynamic mode: emit when `freevars < free_th`, then `free_th *= (1.0 - pow(fraction, pow(depth, downexp)))` with `fraction = 0.02` (`-f`), `downexp = 0.3` (`-e`); `free_th` is reset upward to the current `freevars` when a refutation is found cheaply. `-l` limits the number of cubes (tree filtered by weight = freevars at node).

The paper's comparison used identical `-m` and the same `-n`/`-d` cutoffs for both tools ("We used the same variable cutoff (n_elim) range as prior CnC studies"). The essential difference is search: march picks the greedy argmax of one-level (plus double-look) lookahead; AMS searches up to 20 levels deeper with 10 simulations per node but restricts each node to the top-3 candidates and uses only BCP counts.

## 6. Results (exact figures)

Hardware: dual AMD EPYC 7713 @ 2.0 GHz, 128 cores/node, 512 GB. Timeout 5 days. Elapsed time excludes cluster scheduling overhead.

Instance sizes: KS SAT+CAS encodings: order 19: 3,876 vars / 233,219 clauses; 20: 4,560 / 408,455; 21: 5,320 / 923,933; 22: 6,160 / 2,496,012. KS SMS encodings: 20: 2,550 / 23,915; 21: 2,884 / 28,609; 22: 3,245 / 33,979. Ramsey R(8,3)=28 instance: 15,820 vars / 3,163,013 clauses. Murty-Simon D2(15,56): 11,655 vars / 71,338 clauses.

Table 1 (parallel cubing + parallel solving, SAT+CAS conquering, 128 cores; hours):

| Instance | Tool | Cubing+simp CPU | Total CPU | CPU speedup | Elapsed | Elapsed speedup |
|---|---|---|---|---|---|---|
| Ramsey (8,3) | march | 3.49 | 9.36 | | 2.16 | |
| | AMS | 1.22 | 7.58 | 1.23x | 0.94 | 2.30x |
| KS 19 | march | 1.16 | 1.70 | | 0.17 | |
| | AMS | 0.29 | 0.54 | 3.15x | 0.08 | 2.13x |
| KS 20 | march | 3.59 | 7.61 | | 0.60 | |
| | AMS | 0.90 | 3.21 | 2.37x | 0.31 | 1.93x |
| KS 21 | march | 104.88 | 253.94 | | 5.59 | |
| | AMS | 5.64 | 49.48 | 5.13x | 1.74 | 3.21x |
| KS 22 | march | | timeout | | timeout | |
| | AMS | 243.67 | 1841.18 | | 16.34 | |

Table 2 (SMS conquering, 128 cores; hours):

| Instance | Tool | Cubing CPU | Total CPU | Elapsed | Elapsed speedup |
|---|---|---|---|---|---|
| D2(15,56) | march | 0.07 | 2.81 | 0.37 | |
| | AMS | 1.98 | 3.87 | 0.23 | 1.61x |
| KS 20 | march | 0.01 | 0.51 | 0.50 | |
| | AMS | 0.57 | 1.41 | 0.16 | 3.12x |
| KS 21 | march | 0.04 | 8.15 | 8.18 | |
| | AMS | 0.65 | 7.70 | 1.08 | 7.57x |
| KS 22 | march | | timeout | timeout | |
| | AMS | 18.08 | 137.39 | 10.01 | |

Honest reading of Table 2: with SMS, AMS's cubing CPU is 30-60x *higher* than march's and total CPU is higher on D2 and KS 20; the win is elapsed time, i.e. load balance, not work reduction. With SAT+CAS (Table 1) AMS wins on both CPU and elapsed.

Cube-level analysis, KS 21, SAT+CAS, 128 cores (Sec. 6.3, Fig. 1-2). Cube counts by variable cutoff VC (read from the Fig. 1 panel titles): VC=70: march 3601 cubes vs AMS 279; VC=75: 4533 vs 181; VC=80: 4690 vs 505. Hard cubes (> 5000 s) from Fig. 2, approximate chart readings: VC=70 march ~26 vs AMS ~6; VC=75 ~30 vs ~4; VC=80 ~44 vs ~2; cumulative time on hard cubes ~50-80 h for march vs a few hours for AMS. Paper: "AlphaMapleSAT does not merely reduce the number of cubes uniformly, but specifically avoids generating the hardest cubes that consume the majority of solving time." Mean/median per-cube times in Fig. 1 are not legibly extractable from the PDF; not recorded.

Scaling (Fig. 3): KS 21 total CPU decreases monotonically from 32 to 64 to 128 cores at each VC (so more cores do not inflate total work; consistent with balanced cubes).

Repository logs (`AlphaMapleSAT-CnC/logs/`, apparently older runs; they do not match Table 1's wall-clock numbers, so treat as a second data point, not the paper's): KS 21: AMS solving 201,441 s, cubing 129,469 s, simp 259,489 s, 3,519 leaf cubes, 35,253 total cube nodes, 6 timeouts, wall 12:07:04; march: solving 769,540 s, cubing 614,139 s, simp 445,672 s, 10,926 leaf cubes, 25,010 nodes, 9 timeouts, wall 1-12:28:09. R(3,8): AMS 7,380 leaves / 13 timeouts / wall 1-01:43:21; march 7,649 leaves / 11 timeouts / wall 1-00:16:45 (march *faster* in that log). KS 19: AMS 247 leaves, wall 13:49 vs march 712 leaves, wall 29:36. KS 20: AMS 759 leaves, 1:11:24 vs march 1,369 leaves, 1:48:18.

## 7. How AMS estimates cube hardness without solving

It does not, explicitly. The paper is direct about this: "Unlike AlphaGo, which relies on neural networks to predict value and policy ... AlphaMapleSAT uses deductive reasoning via a SAT solver to compute symbolic rewards based on propagation rate." The operative assumptions are:

1. A split whose BCP assigns many cubing variables per decision shrinks the sub-formula a lot (few free variables left), and a sub-formula with few free variables is easy. This is march's assumption too, and it is why both tools cut off on *variables eliminated*, not on depth.
2. Looking several levels ahead (MCTS up to depth d+20) at the *average* of both children, while remembering the *best* achievable (B), finds splits whose grandchildren are balanced rather than one trivially refuted child and one hard child.
3. Empirical hardness is only measured post hoc: the conquer worker's timeout, after which the cube is re-cubed with a larger cutoff.

There is no learned or fitted hardness model in v2; the deep-learning-based cubing is listed as future work.

## 8. Relevance to our design (partition-branch SAT attack with Lean-verified prunes)

Our branch = a (row-sum profile, column-sum profile) pair, or the unordered partition version; the conquer instance = K_{s,t}-free encoding + exact per-row/per-column cardinalities (`encodings_zar.py`, `encode_matrix(..., col_weights=...)`). The correspondence to CnC is close, and several AMS pieces are reusable essentially verbatim.

### 8.1 A partition branch *is* a cube, if the counters are in the base formula

`encodings_zar.py` already has `exact_unary_counter` with iff semantics: output literal `u_{i,k}` <=> "row i has >= k ones". If the base CNF contains these counters for every row and column (no fixed weights), then the branch (r, c) is exactly the cube

    cube(r,c) = AND_i ( u_{i,r_i} AND NOT u_{i,r_i+1} ) AND AND_j ( v_{j,c_j} AND NOT v_{j,c_j+1} )

and every AMS mechanism applies with PySAT `propagate(assumptions=cube(r,c))` on one long-lived solver object, no re-encoding per branch. This also matches the ZarPrune scope note ("row/column sum vectors, not the unordered partitions"): the cube lives on profiles; the harness maps partitions to a canonical profile plus a symmetry certificate.

### 8.2 Directly reusable difficulty proxies for a branch (all zero-solve, all cheap)

(a) Propagation rate of the branch: `|asgn(F ∧ cube(r,c)) ∩ Cells| / |cube(r,c)|`, counting *cell* variables only (AMS counts only vars <= m for the same reason: auxiliaries inflate counts; our 10x14 instance has 7,875 vars but only 140 cells). High rate => strongly constrained => expected easy.

(b) Free cells after BCP: `m·n - |asgn ∩ Cells|`. This is march's `freevars` cutoff quantity and AMS's n_elim; a branch with many free cells is the analogue of a cube that must be split further.

(c) Lookahead bite: `max_x score(x)` over free cells x under the branch, with AMS's `score(x) = r(x)·r(¬x)/(k+1)^2 + r(x)/(k+1) + r(¬x)/(k+1)`. Low max score = no single cell decision propagates much = hard. Also gives, for free, the best cell to split on if a branch must be sub-cubed.

(d) Failed-literal count: number of cells x with propagate(cube ∧ x) or propagate(cube ∧ ¬x) UNSAT. Many failed literals => nearly refuted => easy; zero => hard. If BCP on the bare cube is UNSAT, the branch is refuted (see 8.4 for what that does and does not buy us in Lean).

(e) Budgeted simplification: CaDiCaL with a 10,000-conflict budget (their `simplify-by-conflicts.sh`); outputs "UNSAT" (branch done, with DRAT), or the number of cell variables eliminated (their cutoff-`v` metric) and the simplified size. This is the cheapest *solver-derived* signal beyond BCP.

(f) Empirical: budgeted solve with re-cube on timeout, exactly their pipeline; on our 10x14/78 test set 120-300 s budgets already separate branches by an order of magnitude (below).

### 8.3 What our local data says about static features (n = 17, so indicative only)

For z(10,14;3,3) at weight 78 (21 sorted column-weight profiles surviving `sum w = 78`, `sum C(w,3) <= 2·C(10,3) = 240`), CaDiCaL per-profile times from `gpt_agent/analysis/sat_attack/results/V78_p*.json` range from 9.2 s to > 363 s (4 timeouts at 120 s budget, two later finished at 280 s and 334 s). KST slack `240 - sum C(w_j,3)`, max weight and number of distinct weights do *not* order the times cleanly: slack-0 profiles take 9-70 s, slack-2 profiles take 3 s (weight 77) and 363+ s (weight 78); the two hardest (`7,7,6,6,6,6,6,6,5,5,5,5,4,4` and `7,7,7,6,6,5,5,...`) are mid-slack, mid-max-weight. Conclusion: static counting features are good *prunes* (they kill) but poor *difficulty rankers* on survivors; a propagation-based probe like AMS's is the right next thing to test, and it is cheap to run over all 21 profiles.

### 8.4 What AMS contributes to the reward-function question in the proposal (Sec. 2.3)

- The proposal asks for "some measurement of how many branches the model would prune and estimate the 'difficulty' of the section of the problem we eliminated." AMS's answer for cubes is: value = propagation per decision, backed up by averaging children and maximizing over actions. Translated: score a candidate prune by `sum over killed branches b of hardness(b)` with `hardness` from 8.2(a)-(e), all computable in the evaluator without solving, and *independent of the candidate* (the branch CNF is fixed), so the model cannot game the proxy by changing the prune's text.
- AMS's central empirical lesson is that the *tail* matters: fewer very-hard cubes beat fewer cubes. The reward should weight kills by estimated hardness (or count kills above a hardness threshold), not count kills uniformly; the Lean-verified baseline prunes (`deficit`, `mismatch`, caps) kill huge numbers of trivial branches that BCP would refute in milliseconds anyway, and rewarding those inflates fitness without moving the bound.
- Max-backup (B) rather than mean when the objective is "find one good decision" suggests, for a *population* of prunes, scoring by coverage of the hard set (union) rather than average per-prune kill count.
- The `varpen` balance penalty is an existing knob for "prefer splits whose two children are similar"; for branch enumeration order it suggests preferring refinements that leave balanced work.
- No learning is needed to get 2-7x; deductive rewards from a solver are enough. This is a strong argument for putting solver probes (PySAT `propagate`, conflict-limited CaDiCaL) inside the OpenEvolve evaluator rather than a learned hardness model.

### 8.5 Where AMS does *not* transfer

- MCTS over a binary split tree assumes the action is "pick a variable". Our top-level branching is a fixed partition lattice enumerated by `profiles.py:enum_profiles` (DFS over non-increasing weights with the KST budget). MCTS would only apply if we (i) treat partition refinement as a tree (choose the max weight, then the next...) and use propagation on the partial cube as reward, or (ii) sub-cube hard branches on cell variables. (ii) is the natural fit and is exactly AMS with `-m = m·n`.
- AMS has no symmetry handling; that lives in the conquer solver (CAS / SMS). Consistent with `lean/README.md`: prunes kill empty cases; sorting/lex is an *addition* with a witness, certified in SR/VeriPB, not a `Prune`.
- Python/PySAT overhead: the authors flag it as a limitation; at our scale (hundreds to low thousands of branches, ~10^4 vars) it is irrelevant.

## 9. Pruning lemmas extractable from this source

This paper contains no counting lemma about Zarankiewicz or any combinatorial structure; its "pruning" is BCP refutation and failed-literal learning. Recorded for completeness and to keep the ledger honest.

### L1. BCP-refutation kill (sound, but not a ZarPrune-style `Prune`)

Statement. Let Enc(P) be the CNF encoding of "A is an m x n 0/1 matrix, K_{s,t}-free, weight >= w" with exact unary row/column counters, and cube(pf) the counter-literal cube of profile pf. If unit propagation on Enc(P) ∧ cube(pf) derives the empty clause, then for every A with profileOf A = pf, ¬ Valid P A.

Hypotheses. (H1) Encoding soundness: every A with Valid P A and profileOf A = pf induces a satisfying assignment of Enc(P) ∧ cube(pf). (H2) Unit propagation is a sound refutation procedure (RUP).

Provability in Mathlib-free Lean 4 over ZarPrune. Not as a `Prune P`: `kill` would have to run a unit-propagation engine on a CNF term inside Lean, and `sound` would need (H1) as a theorem relating `Mat`, `HasKst` (increasing tuples), `sumFin` to the clause set produced by `encodings_zar.py`. Both are real work (an encoding-correctness theorem plus a verified UP checker, or import of a checked DRAT/LRAT refutation). This is exactly the `refuted` seam in `upper_bound_of_cover`, not a profile-level lemma. Cost: high. Value: it certifies exactly the branches a solver would kill instantly, so as a *prune* it gains nothing over just solving them with a proof log; as a *difficulty signal* (8.2 d) it is free.

### L2. Failed-literal fixing is an addition, not a prune

Statement. If propagate(Enc ∧ cube ∧ ℓ) is UNSAT then Enc ∧ cube ⊨ ¬ℓ; adding the unit ¬ℓ preserves satisfiability. (AMS appends ¬ℓ to `unsat_learnt_actions`.)

Relevance. Sound as a RUP clause addition inside the certificate; it never kills a case by itself. Do not let an evolved candidate present "fix these cells" as a prune; it would fail `sound` by construction, as `Demo.notDescending_unsound` shows for the analogous sorting move.

## 10. Numbers and settings to reuse

- `numMCTSSims = 10`, `cpuct = 10`, top-3 candidate restriction, MCTS lookahead 20 levels beyond the real cutoff, `varpen = 0`, refuted-leaf reward 0.1.
- Score normalization: n_elim^2 (or (m//4)^2 when cutting by depth).
- Pipeline: simplify 10,000 conflicts; cutoff by variables eliminated; on timeout re-cube with cutoff + 20 variables.
- march defaults for reference: H_BIN 25, H_DEC 0.5, SL_ITER 9, DL_ITER 2, dynamic cutoff fraction 0.02, downexp 0.3.

## 11. Open questions for the thesis

1. Does propagation rate / free-cell count / lookahead bite (8.2 a-c) on `cube(r,c)` rank the 21 profiles of z(10,14;3,3)@78 in the order of their CaDiCaL times? Cheap experiment; 17 timings already exist.
2. Is per-decision normalization by (k+1) meaningful when every "decision" is a counter literal pair? Probably normalize by number of rows+columns instead (constant per instance), i.e. use raw assigned-cell counts.
3. Should the evolutionary reward be `sum hardness(b)` over kills, or a tail statistic (kills among the top-q hardest branches)? AMS's Fig. 2 argues for the tail.
4. Sub-cubing hard branches on cell variables (AMS with `-m = m·n`) versus refining the partition (splitting a profile into sub-cases by an extra invariant such as pair-intersection counts): which yields more balanced conquer jobs, and which is easier to certify in the `cover` obligation?
5. The repository logs and Table 1 disagree substantially for KS 21 (12 h vs 1.74 h elapsed for AMS); which configuration produced the paper's table is not documented. Do not cite the logs as the paper's result.
6. v1 claimed a 27x *sequential* speedup that v2 dropped; if sequential CnC matters for us (one machine), the v1 experiments would be worth re-reading.

## 12. Citations

- P. Jha, Z. Li, Z. Lu, R. Zeng, C. Bright, V. Ganesh. AlphaMapleSAT: An MCTS-based Cube-and-Conquer SAT Solver for Hard Combinatorial Problems. arXiv:2401.13770v2, 20 Jan 2026 (v1 24 Jan 2024).
- Code: https://github.com/piyush-J/AlphaMapleSAT ; https://github.com/BrianLi009/AlphaMapleSAT-CnC (MathCheck-style pipeline with bundled march_cu, cadical-ks, SMS).
- M. J. H. Heule, O. Kullmann, S. Wieringa, A. Biere. Cube and conquer: Guiding CDCL SAT solvers by lookaheads. HVC 2011 (march_cu; cited by the paper for the cutoff and heuristic).
- Z. Li, C. Bright, V. Ganesh. A SAT solver + computer algebra attack on the minimum Kochen-Specker problem. IJCAI-24 (KS SAT+CAS encodings and MathCheck pipeline used here).
- M. Kirchweger, S. Szeider. SAT modulo symmetries for graph generation and enumeration. ACM TOCL 25(3), 2024 (SMS conquering solver).
- L. Scavuzzo et al. Learning to branch with tree MDPs. NeurIPS 2022 (tree-MDP framing).
- C. D. Rosin. Multi-armed bandits with episode context. AMAI 61(3), 2011; D. Silver et al. 2017 (PUCT).
- A. Ignatiev, A. Morgado, J. Marques-Silva. PySAT. SAT 2018 (propagate API used for rewards).

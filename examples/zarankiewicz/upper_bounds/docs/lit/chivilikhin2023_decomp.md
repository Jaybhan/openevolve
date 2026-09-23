# Chivilikhin, Pavlenko, Semenov — Decomposing Hard SAT Instances with Metaheuristic Optimization

Literature note for the Zarankiewicz upper-bound thesis (evolved, Lean-verified pruning of a Tan-style case split). Read in full from the arXiv PDF (29 pp.); all formulas below are transcribed from the primary text. Anything marked **[inference]** is ours, not the paper's.

## 1. Bibliographic record

| Field | Value |
|---|---|
| Title | Decomposing Hard SAT Instances with Metaheuristic Optimization |
| Authors | Daniil Chivilikhin, Artem Pavlenko, Alexander Semenov (ITMO University, St. Petersburg) |
| arXiv | 2312.10436 [cs.AI, cs.DS], v1 submitted 16 Dec 2023 (PDF dated 19 Dec 2023) |
| Journal version | *International Journal of Artificial Intelligence* 21(2), 2023, pp. 61–92 (the arXiv text is described as a preprint of this) |
| Extends | Semenov, Chivilikhin, Pavlenko, Otpuschennikov, Ulyantsev, Ignatiev, "Evaluating the hardness of SAT instances using evolutionary optimization algorithms", CP 2021, LIPIcs 210, 47:1–47:18 (their ref. [72]; DOI 10.4230/LIPIcs.CP.2021.47) |
| Companion concept | Semenov, Pavlenko, Chivilikhin, Kochemazov, "On probabilistic generalization of backdoors in Boolean satisfiability", AAAI 2022, pp. 10353–10361 (their ref. [74]; the ρ-backdoor paper) |
| Tool | modified **EvoGuess** (Pavlenko, Semenov, Ulyantsev, EvoApplications 2019, ref. [67]) |
| Local copy | `scratchpad/lit/chivilikhin2023_decomp.pdf` (805 KB, 29 pages) |
| Accessibility | Full primary text read. The two predecessor papers were only available as abstracts on the publisher pages; where they are cited here it is through what this paper says about them. |

## 2. One-paragraph summary

The paper defines the **decomposition hardness (d-hardness)** of an unsatisfiable CNF formula C with respect to a complete deterministic SAT solver A and a variable subset B: the total solver work over all 2^|B| subformulas obtained by substituting every assignment of B. It shows this total equals 2^|B| times the expectation of a random variable ξ_B (solver work on a uniformly random cube), so it can be estimated by Monte Carlo sampling with an (ε, δ)-guarantee from Chebyshev's inequality, N ≥ Var(ξ_B)/(ε² δ E²[ξ_B]). Finding the B with minimal d-hardness is a black-box pseudo-Boolean minimisation, attacked with an elitist genetic algorithm (two-point crossover, heavy-tailed (1+1) mutation with β = 3, fitness-proportional selection on 1/F). Three engineering ideas make the search work at scale: (i) restrict the search to a few hundred variables B₀ chosen by a unit-propagation weight w_i = w_i⁺ + w_i⁻; (ii) exploit that almost all cubes are unit-propagation-trivial (B is a **ρ-backdoor** with ρ ≈ 1), so measure them with an incremental UP-only solver instead of a full solver launch; (iii) combine two or three independent decomposition sets whose "hard" cubes form a Cartesian product Γ. Experiments on logical-equivalence-checking and sgen UNSAT instances show 1.2×–20× solving speedups and even larger DRAT-proof-checking speedups from the found decompositions.

## 3. Definitions and theorems (exact statements)

### 3.1 Probabilistic preliminaries (§2.2)

Chebyshev (their eq. 1): for a random variable ξ with finite mean and variance and any d > 0,

    Pr[ |ξ − E[ξ]| ≤ d·sqrt(Var(ξ)) ] ≥ 1 − 1/d².                        (1)

Sample mean and unbiased sample variance from N observations ξ¹, …, ξᴺ (eqs. 2, 3):

    ξ̄ = (1/N) · Σ_{j=1}^{N} ξ^j,        s²(ξ) = (1/(N−1)) · Σ_{j=1}^{N} (ξ^j − ξ̄)².

(ε, δ)-approximation (eq. 4): θ̃ is an (ε, δ)-approximation of θ ≥ 0 if

    Pr[ (1−ε)·θ ≤ θ̃ ≤ (1+ε)·θ ] ≥ 1 − δ,   equivalently  Pr[ |θ̃ − θ| ≤ ε·θ ] ≥ 1 − δ.

ε is the *tolerance*, 1 − δ the *confidence level*.

### 3.2 Strong backdoors and backdoor hardness (§2.3)

Notation: C is a CNF over variables X; for B ⊆ X and β ∈ {0,1}^|B|, C[β/B] is C after substituting β and simplifying. P is a polynomial-time *sub-solver*.

**Definition 1 (Williams–Gomes–Selman 2003, their [85]).** B ⊆ X is a *strong backdoor set (SBS)* for C w.r.t. P if for every β ∈ {0,1}^|B| the algorithm P outputs the solution of SAT for C[β/B].

Consequence stated: if C has an SBS, SAT for C is solved in time poly(|C|)·2^|B|. Example given: the CNF for inverting SHA-256 (49 098 variables) has an SBS of the 512 input variables w.r.t. unit propagation.

**Theorem 1 ([85]).** If C has an SBS B of size |B| < |X|/2, there is an algorithm solving SAT for C in time

    poly(|C|) · ( 2|X| / sqrt(|B|) )^{|B|}.                                  (5)

**Corollary 1 ([85]).** For CNFs with an SBS of size smaller than k/4.404, k = |X|, SAT can be solved deterministically in O(poly(|C|)·c^k) with c < 2.

**Corollary 2 (this paper).** The worst-case complexity of the WGS algorithm (enumerate subsets by increasing size, test each for the SBS property) for finding a minimal SBS is O(p(|C|)·3^|X|), because Σ_{i=1}^{k} C(k,i)·2^i = 3^k − 1.

**Definition 2 (b-hardness).** For unsatisfiable C and an SBS B w.r.t. P, with t_P(C[β/B]) the runtime of P on C[β/B],

    μ_{B,P}(C) = Σ_{β ∈ {0,1}^{|B|}} t_P(C[β/B]),
    μ_P(C)     = min_{B ∈ 2^X, B is an SBS} μ_{B,P}(C)     (backdoor hardness of C w.r.t. P).

The paper restricts throughout to **unsatisfiable** formulas, because on satisfiable ones a solver "may often get lucky" and the runtime is not a meaningful hardness measure.

### 3.3 Decomposition hardness (§4)

**Definition 3 (d-hardness).** For a CNF C over X, an arbitrary B ∈ 2^X and a deterministic complete solver A with runtime t_A(C[β/B]) on each subformula,

    μ_{B,A}(C) = Σ_{β ∈ {0,1}^{|B|}} t_A(C[β/B]),        μ_A(C) = min_{B ∈ 2^X} μ_{B,A}(C).

Runtime may be measured "in any appropriate units": seconds, number of conflicts, or number of unit propagations. Every B gives an *upper bound* on the hardness of C (as an SBS does); B = X and B = ∅ are the trivial extremes.

**Theorem 2.** For any C, any deterministic complete A and any B ∈ 2^X there is a random variable ξ_B with finite spectrum, expectation and variance such that

    μ_{B,A}(C) = 2^{|B|} · E[ξ_B].                                            (6)

*Proof (as given).* Take Ω = {0,1}^|B| with the uniform distribution and ξ_B(β) = t_A(C[β/B]). If Spec(ξ_B) = {ξ₁,…,ξ_s} and #ξ_i is the number of β with t_A(C[β/B]) = ξ_i, then p_i = #ξ_i / 2^|B| and Σ_β t_A(C[β/B]) = Σ_i ξ_i·#ξ_i = 2^|B| · Σ_i ξ_i · #ξ_i/2^|B| = 2^|B|·E[ξ_B]. ∎

**Estimator (eq. 8).** From N independent observations ξ¹,…,ξᴺ of ξ_B (uniform random cubes β, deterministic A),

    μ̃_{B,A}(C) = (2^{|B|} / N) · Σ_{j=1}^{N} ξ^j.                             (8)

**Theorem 3 (sample-size bound).** For μ̃_{B,A}(C) as in (8) and any ε > 0, δ > 0, the (ε,δ) condition

    Pr[ (1−ε)·μ_{B,A}(C) ≤ μ̃_{B,A}(C) ≤ (1+ε)·μ_{B,A}(C) ] ≥ 1 − δ           (7)

holds for any

    N ≥ Var(ξ_B) / ( ε² · δ · E²[ξ_B] ).                                       (12)

*Proof sketch (as given).* If Var = 0 the claim is trivial. Otherwise apply (1) with d·sqrt(Var(ζ)) = ε·E[ζ] to get Pr[|ζ − E ζ| ≤ ε E ζ] ≥ 1 − Var(ζ)/(ε² E²[ζ]) (eq. 9); apply to ζ = Σ_j ξ^j (i.i.d., so Var(ζ) = N·Var(ξ_B), E[ζ] = N·E[ξ_B]) and divide through by N and multiply by 2^|B|:

    Pr[ (1−ε)·μ_{B,A}(C) ≤ μ̃_{B,A}(C) ≤ (1+ε)·μ_{B,A}(C) ] ≥ 1 − Var(ξ_B)/(ε²·N·E²[ξ_B]).   (10)

Setting the right side to 1 − δ gives (12). ∎

Note the bound is in terms of the *relative* variance Var/E², i.e. the squared coefficient of variation; the paper stresses that Var(ξ_B) is finite but "in practice it can be very large", a direct consequence of heavy-tailed solver behaviour (Gomes–Sabharwal). Worked magnitudes **[inference]**: coefficient of variation 1 with ε = 0.25, δ = 0.2 needs N ≥ 80; coefficient of variation 3 with ε = 0.2, δ = 0.1 needs N ≥ 2250.

### 3.4 The fitness function and the adaptive sampling rule (§5)

A set B is encoded as a Boolean vector λ_B ∈ {0,1}^|X|. The fitness (to be minimised) is

    F_{C,A,N}(λ_B) = (2^{|B|} / N) · Σ_{j=1}^{N} t_A(C[β_j/B]),                (11)

with β₁,…,β_N a random sample from {0,1}^|B|. In practice the population statistics replace the unknown moments (eq. 13):

    N ≥ s²(ξ_B) / ( ε² · δ · (ξ̄_B)² ).                                        (13)

**Adaptive procedure.** Start with a "relatively small" N (they say N = 1000), draw the sample, compute ξ̄_B and s²(ξ_B); if (13) holds accept; otherwise draw N more observations (doubling), recompute, repeat until (13) holds. They note candidly that with statistical estimates in place of true moments "we cannot speak about guarantees of the accuracy" for small N.

**Cost caveat.** For small B, computing (11) can cost as much as solving C itself; it is efficient when B = X or B is a Strong Unit Propagation Backdoor Set (SUPBS). Hence the optimisation is run not on 2^X but on 2^{B₀} for a B₀ ⊂ X on which (11) is cheap.

### 3.5 The metaheuristic (§5, last paragraph)

A **genetic algorithm** (variant of the one in EvoGuess, ref. [67]). Population Π_cur of R vectors λ_B; distribution D_cur = {p₁,…,p_R} with

    p_i = (1/F_{A,C,N}(λ_{B_i})) / Σ_{j=1}^{R} (1/F_{A,C,N}(λ_{B_j})),   i ∈ {1,…,R}.

New population Π_new (|Π_new| = R = E + G + H):
1. **E** elite individuals copied from Π_cur (the text says "highest fitness function values"; since F is minimised this must mean best, i.e. lowest F — **[inference]**, consistent with p_i ∝ 1/F);
2. **G** individuals: select a pair independently from D_cur, apply standard two-point crossover, add both, repeat;
3. **H** individuals: select one from D_cur, apply the **(1+1) heavy-tailed mutation operator** of Doerr, Le, Makhmara, Nguyen (GECCO 2017, "Fast genetic algorithms", ref. [24]) with parameter **β = 3**, repeat.

Experiments used G = 8, E = 2 (H is not stated explicitly in the text). Runs: 1 hour wall-clock, 16 parallel threads, on a 32-core Xeon Gold 6242 @ 2.80 GHz.

### 3.6 Search-space reduction (§6)

Two ways to pick B₀ ⊂ X, |B₀| a few hundred:

* **Look-ahead reduction measures** (Heule–van Maaren): for each variable x and each literal in {x, ¬x}, count the *new clauses* produced by unit propagation + pure-literal after substituting the literal; the reduction measure is usually the product of the two counts (sometimes the sum). Polynomial in |C|. Take the top m variables.
* **The simpler heuristic the paper prefers** (comparable or better in their experiments): for each x_i substitute the positive literal, run UP, count the number of variables whose values are derived, call it w_i⁺; do the same for ¬x_i to get w_i⁻; define the weight

      w_i = w_i⁺ + w_i⁻.

  B₀ = the m variables of largest w_i, with m chosen so that (11) is cheap on B₀. In the main experiments |B₀| = 200.

### 3.7 ρ-backdoors, simple subproblems, several backdoors (§7)

Notation: C[β/B] ∈ S(P) means SAT for C[β/B] is decided by the polynomial sub-solver P.

**Definition 4 (ρ-backdoor; from their AAAI 2022 paper [74]).** B ⊆ X is a *ρ-backdoor* for C w.r.t. P if C[β/B] ∈ S(P) holds for at least ρ·2^{|B|} vectors β ∈ {0,1}^{|B|}.

(ρ = 1 recovers the SBS. §10 adds: for a fixed B, ρ can be (ε,δ)-estimated efficiently by Monte Carlo because the indicator "C[β/B] ∈ S(P)" is a **Bernoulli** random variable and therefore has **Var ≤ 1/4**, which removes the heavy-tail problem for this quantity. **[inference]** With Chebyshev this gives N ≥ 1/(4 ε_abs² δ) for absolute tolerance ε_abs; with Hoeffding, N ≥ ln(2/δ)/(2 ε_abs²), e.g. ε_abs = 0.1, δ = 0.1 → N ≥ 150.)

**Observation motivating the "incremental preprocessing" trick.** When the search starts from X, a SUPBS, or the heuristic B₀, "most of these subproblems are polynomially solvable: i.e. B is some ρ-backdoor with ρ which is quite close to 1." Therefore, instead of launching the complete solver from scratch on every sampled cube, keep an additional **UP-only solver used incrementally through the assumptions mechanism**; when it decides C[β/B], its own workload (propagation time / number of unit propagations) is the observation of ξ_B; only the cubes it fails on go to the full solver.

**Several backdoors.** Let B₁,…,B_s be ρ_i-backdoors for C w.r.t. P. Let Γ_i = {β ∈ {0,1}^{|B_i|} : C[β/B_i] ∉ S(P)} ("hard" cubes) and Γ = Γ₁ × ⋯ × Γ_s; for γ ∈ Γ let C[γ] be C with all Σ_i |B_i| bits substituted. Then

> C is unsatisfiable iff all formulas C[β/B_i] ∈ S(P), i ∈ {1,…,s}, are unsatisfiable, and all formulas C[γ], γ ∈ Γ, are unsatisfiable.

Cost: a polynomial algorithm on Σ_i ρ_i·2^{|B_i|} simple subproblems, plus a complete solver on 2^{Σ_i |B_i|} · Π_i (1 − ρ_i) formulas C[γ], each of which is C with Σ_i |B_i| bits fixed and hence much simplified. "In practice we see that the vast majority of formulas C[γ] are polynomially solvable."

### 3.8 Decomposed unsatisfiability proofs (§8)

* A decomposition set B partitions C into 2^|B| subformulas; solving each independently yields 2^|B| independent DRAT proofs P[β/B], checked independently and in parallel with **DRAT-trim**.
* For a ρ-backdoor with ρ ≈ 1: store one proof per hard cube ((1−ρ)·2^|B| of them; for |B| < 15 "typically does not exceed 1000"), and batch the simple cubes: split the simple assignments into K ≪ 2^|B| subsets Q₁,…,Q_K, Q_k = {β_k^1,…,β_k^{d_k}}; with σ(β) the cube (conjunction of literals) for β, fresh variables u₁,…,u_{r_k}, and

      C ∧ (u₁ ≡ σ₁^k) ∧ ⋯ ∧ (u_{r_k} ≡ σ_{r_k}^k) ∧ (u₁ ∨ ⋯ ∨ u_{r_k})            (14)

  which "is unsatisfiable if and only if all formulas C[β/B], β ∈ Q_k are unsatisfiable"; Tseitin-transform to a CNF C_k and prove it once.
* Fig. 1 (Glucose 3, PvS_{7,4}, ten backdoors): solving time, DRAT-trim checking time and proof size as functions of K all have their minimum around K = 10–20; solving all simple cubes individually "is infeasible". They used **K = 20**.

## 4. Experiments (§9)

Research questions: **RQ1** which workload measure (seconds / unit propagations / conflicts) is best for d-hardness estimation; **RQ2** can the found B solve faster than plain SAT; **RQ3** can decomposition cut proof-checking time.

**Benchmarks.** (a) `sgen` UNSAT generator (Spence 2015), 150 variables. (b) Logical Equivalence Checking of pairs of sorting circuits on k numbers of l bits: BvP_{k,l} (Bubble vs Pancake), BvS_{k,l} (Bubble vs Selection), PvS_{k,l} (Pancake vs Selection). Solvers: **CaDiCaL** (cd) and **Glucose 3** (g3).

**RQ1 (§9.2).** PvS_{7,4}; 20 independent GA runs (G = 8, E = 2, N = 1000, 1 h, 16 threads) per (solver, measure); best B per run; compare the 20 resulting solve times (Fig. 2 violin plots). Mann–Whitney U: propagations and conflicts statistically indistinguishable; time statistically worse than propagations (p = 0.043 at level 0.05). Conclusion: use **unit propagations** (or conflicts) as the workload measure; all later experiments use propagations.

**Incremental preprocessing (§9.3, Figs. 3–5).** Starting from B = X (3244 vars): incremental UP preprocessing descends faster and explores more backdoors, but after 1 h the best B still has > 2000 variables, useless in practice. Starting from B₀ with |B₀| = 200 (30 min): hard subproblems C[β/B] ∉ S(P) appear only once |B| ≈ 20–25; every larger B seen had all sampled cubes UP-solved. Hence the winning recipe: search inside B₀ (|B₀| = 200) but **initialise with |B| = 30**. Fig. 5 (right): for the top-w_i sets of size 1, 2, 10, 100, …, 600 on PvS_{7,4} (caption says PvS_{4,7}), ρ estimated with N = 10 000 is already **1.0 at |B| = 20**. Fig. 4 (right) box plots of final log F: w/o incremental ≈ 27–28, with incremental ≈ 24–25, |B| = 30 start ≈ 23 (log of propagations).

**Main experiments (§9.4).** 10 runs × 1 h × 16 threads per (instance, solver); N = 1000; decomposed solving on a 32-core Xeon Gold 6338 @ 2.00 GHz, each solve repeated 5×, std ≤ 5 %; DRAT-trim checking of all proofs.

Table 1 — plain solving (no decomposition):

| Instance | \|X\| | cd solve (s) | cd proof (MB) | cd check (s) | g3 solve (s) | g3 proof (MB) | g3 check (s) |
|---|---|---|---|---|---|---|---|
| PvS_{7,4} | 3244 | 460 | 679 | 1601 | 986 | 1803 | 3206 |
| BvP_{7,6} | 3492 | 487 | 561 | 1199 | 1480 | 2481 | 6644 |
| BvS_{7,7} | 4462 | 837 | 814 | 1617 | 3559 | 4596 | 15084 |
| BvP_{8,4} | 3013 | 1488 | 1510 | 5647 | 3598 | 4704 | 21503 |
| sgen | 150 | 2022 | 2332 | 5898 | 8404 | 12823 | 37724 |

Proof *checking* always takes longer than solving (2–5× for cd, 3–5× for g3).

Table 2 — ratios (decomposed / monolithic), mean ± relative std over the 10 found sets B. r = solving-time ratio, π = proof-checking-time ratio; Γ = pairs of backdoors (Cartesian product of hard cubes):

| C | A | avg \|B\| | r_{B,A} | π_{B,A} | r_{Γ,A} | π_{Γ,A} |
|---|---|---|---|---|---|---|
| PvS_{7,4} | cd | 14.5 | 0.41 ± 0.11 | 0.30 ± 0.16 | 0.39 ± 0.11 | 0.22 ± 0.12 |
| PvS_{7,4} | g3 | 14.3 | 0.33 ± 0.14 | 0.32 ± 0.21 | 0.28 ± 0.12 | 0.22 ± 0.10 |
| BvP_{7,6} | cd | 13.3 | 0.79 ± 0.12 | 0.57 ± 0.14 | 0.74 ± 0.11 | 0.47 ± 0.13 |
| BvP_{7,6} | g3 | 14.1 | 0.48 ± 0.17 | 0.32 ± 0.26 | 0.41 ± 0.14 | 0.23 ± 0.21 |
| BvS_{7,7} | cd | 13.8 | 0.80 ± 0.10 | 0.88 ± 0.14 | 0.70 ± 0.12 | 0.66 ± 0.11 |
| BvS_{7,7} | g3 | 14.4 | 0.23 ± 0.08 | 0.13 ± 0.13 | 0.21 ± 0.11 | 0.10 ± 0.14 |
| BvP_{8,4} | cd | 13.9 | 1.09 ± 0.06 | 0.80 ± 0.10 | 0.93 ± 0.06 | 0.51 ± 0.11 |
| BvP_{8,4} | g3 | 14.4 | 0.72 ± 0.06 | 0.49 ± 0.12 | 0.60 ± 0.08 | 0.31 ± 0.12 |
| sgen | cd | 19.0 | 0.20 ± 0.08 | 0.12 ± 0.15 | — | — |
| sgen | g3 | 19.6 | 0.05 ± 0.14 | 0.01 ± 0.29 | — | — |

Findings: r < 1 in all but one case (BvP_{8,4}/cd, 1.09); π < 1 always; pairs beat singles everywhere (r_Γ < r_B, π_Γ < π_B); proof-checking gains exceed solving gains. All of this is *sequential* — one solver on one CPU summed over subproblems.

**Simulated parallelism (Fig. 6).** Queue simulation with 1–32 workers, Glucose 3, single / pairs / triples of backdoors. Speedups grow with the number of backdoors; e.g. PvS_{7,4} reaches roughly 70× with triples at 32 threads, 35× with pairs, and plateaus below 10× with a single backdoor; BvS_{7,7} ≈ 85× with triples. Explanation given: more backdoors → more hard subproblems, but each is simpler and faster. "The best results can be achieved when using triples of backdoors."

## 5. What the paper proves vs. what it only shows empirically

Proved: Theorem 2 (exact identity, elementary), Theorem 3 (Chebyshev), Corollary 2 (3^k count), the "C unsat iff all pieces unsat" decompositions (Cartesian product and formula (14)); all are short and rigorous.

Empirical only: propagations ≈ conflicts > time as a measure (one instance, 20 runs); |B| ≈ 20 threshold where hardness appears; K = 20 grouping optimum; the speedup ratios; "triples best" (simulated, not real parallel runs). Not addressed: satisfiable instances, any guarantee on the GA's convergence, and the actual variance values (the heavy-tail issue is acknowledged but not quantified).

## 6. Relevance to our design

Our object is a Tan-style case split: a *partition pair* (or, in the Lean layer, a `Profile` = the row-sum vector r ∈ (Fin m → Nat) and column-sum vector c ∈ (Fin n → Nat)); one SAT instance C_{r,c} per surviving case; a `Prune P` is a computable `kill : Profile → Bool` plus `sound : kill (profileOf A) = true → ¬ Valid P A`. The existing enumerator (`sat_attack/profiles.py`) already prunes column-weight profiles by Σ_j C(w_j, 3) ≤ 2·C(m, 3) before emitting one fixed-weight instance per profile.

### 6.1 The paper's decomposition is literally our case split, one level up

Replace "assignment β of the variables in B" by "profile q in the finite set S of surviving cases", and "C[β/B]" by "C_q". Then, verbatim from Theorem 2 with the uniform distribution on S instead of {0,1}^|B|:

    W(S) := Σ_{q ∈ S} t_A(C_q) = |S| · E[ξ_S],   ξ_S(q) = t_A(C_q), q uniform on S,

and the estimator and Theorem 3 transfer unchanged:

    Ŵ(S) = (|S| / N) · Σ_{j=1}^{N} t_A(C_{q_j}),      N ≥ Var(ξ_S) / (ε² δ E²[ξ_S]).

So the total cost of finishing a bound with a given prune set is a Monte Carlo estimate over sampled survivors; no need to solve every case to know roughly how much work is left. **[inference]** Two differences from the paper: (a) our "decomposition set" is not a set of Boolean variables but a set of cardinality constraints, so |S| is the number of partition pairs (counted exactly by the enumerator) rather than 2^|B|; (b) we do not need the *minimum* over decompositions, we need a *comparable* estimate across candidate prunes.

### 6.2 Reward for an evolved prune = hardness mass removed, not case count

The paper's central practical lesson (§7, §9.3, Fig. 5): in a natural decomposition, ρ ≈ 1 — almost every piece is trivial for unit propagation — and a search that counts pieces instead of charging each its true cost either wanders (Fig. 3: |B| > 2000 after an hour) or rewards the wrong thing. Hardness lives in a heavy-tailed minority of pieces.

For us this means a prune that kills many partition pairs which UP (or the `deficit`/`mismatch`/`rowCap`/`colCap` baseline, or the KST counting inequality) already disposes of in microseconds has earned almost nothing. The reward should be

    R(p) = Σ_{q ∈ S : p.kill q} ĥ(q)  −  λ · cost(p.kill),

with ĥ(q) an estimate of t_A(C_q) *after the cheap baseline has already run*, and estimated from one shared sample of survivors:

    R̂(p) = (|S| / N) · Σ_{j=1}^{N} ĥ(q_j) · [p.kill q_j = true].

Because the sample {q_j} is drawn once and reused for the whole population (common random numbers), comparisons between candidate prunes have much lower variance than the absolute estimates, and evaluating a new prune costs only N calls of `kill` **[inference]**. This is exactly the fitness-function structure (11) with a fixed sample.

### 6.3 Estimating the difficulty ĥ(q) of one partition pair by sampling

Three estimators, in increasing cost, all directly from the paper:

1. **ρ-proxy (Bernoulli, no heavy tail).** Choose a cube set B_q of k cells inside C_q (e.g. the cells of the heaviest row(s), or the top-w_i cells by the paper's UP weight w_i = w_i⁺ + w_i⁻ computed on C_q). Sample N cubes β uniformly from {0,1}^k, run UP only (incremental solver with assumptions), record the indicator "refuted by UP". ρ̂_q = fraction refuted. Var ≤ 1/4, so N ≈ 150–250 gives ±0.1 with 90 % confidence regardless of solver behaviour. Difficulty proxy: (1 − ρ̂_q)·2^k = estimated number of hard cubes. Cheap enough to run on thousands of pairs.
2. **d-hardness of C_q.** Same cubes, but the cubes UP fails on go to the full solver with propagations counted; ĥ(q) = (2^k / N)·Σ_j ξ^j. Use the doubling rule (13) with a cap. Choose k so that hard cubes actually appear (the paper's Fig. 5 analogue: for their 3244-variable instances k ≈ 20–30; for an m·n ≤ 400-cell instance k will be smaller — to be calibrated).
3. **Budgeted direct solve.** Run A on C_q with a conflict cap c; record propagations if solved, else record "censored at c". Treat censored cases as ≥ c; this is the cheapest measurement but the heavy tail makes the mean estimate unreliable — the paper's warning about Var(ξ_B) applies. Useful as a first-stage filter before 1–2.

Workload units: **unit propagations** (paper's RQ1 result; conflicts equally good; wall-clock statistically worse and machine-dependent). Propagation counts are deterministic for a fixed solver binary and seed, which is what makes ξ a function of q (Theorem 2 assumes A deterministic) and what makes rewards reproducible across OpenEvolve worker processes **[inference]**.

### 6.4 Two-level sampling budget

Outer sample: N_out survivors q_j uniform on S (or stratified by row partition, so that rare heavy partitions are not missed — the paper does not stratify; **[inference]**). Inner: N_in cubes per q_j. Total solver calls N_out·N_in, most of which are UP-only. With the paper's numbers (N = 1000 cubes per candidate B, 16 threads, 1 h per GA run) as a scale reference, an OpenEvolve iteration that spends ~10³–10⁴ UP calls plus ~10²–10³ short full solves per candidate prune is in the same regime.

### 6.5 Multiple decompositions = Tan's row × column split

The paper's Cartesian-product statement (§7) is exactly the structure of a partition pair: B₁ = "row-sum vector", B₂ = "column-sum vector", Γ₁ = row vectors not killed by row-only prunes (`rowCap`, KST on rows), Γ₂ = column vectors not killed by column-only prunes, and the survivors are Γ₁ × Γ₂ minus the pairs killed by genuinely two-sided prunes (`mismatch`, `deficit`, evolved pair prunes). Their empirical result that pairs and triples of backdoors beat singles (Table 2, Fig. 6) supports Tan's two-sided split and suggests trying a *third* independent cheap invariant as a further split axis (e.g. the number of ones in a fixed row-block) — with the caveat that a third axis multiplies the number of cases, and pays off only if most of the product is killed cheaply.

### 6.6 Certificates

Table 1 says proof checking is 2–5× slower than solving; Table 2 says decomposition helps checking more than solving. Our `refuted` hypothesis in `upper_bound_of_cover` needs one checked certificate per survivor, so per-case LRAT/DRAT proofs checked in parallel are the natural fit, and the grouping trick (14) with K ≈ 20 could batch the UP-trivial survivors into a handful of instances. But: (14) introduces auxiliary variables and an "iff all pieces unsat" argument that would have to be proved in Lean as part of the encoding-correctness theorem; unless the number of trivial survivors is enormous, killing them with a verified prune (which needs no certificate at all) is the cleaner route **[inference]**.

### 6.7 What the paper does *not* give us

No Zarankiewicz-specific pruning lemma, and no soundness content: d-hardness, ρ, and the GA are all *reward-side* machinery. Nothing here touches the Lean trust boundary; a wrong hardness estimate wastes compute, never soundness. The only statements that would ever enter Lean are the trivial decomposition identities (§7 Cartesian product; formula (14)), and those are the `cover` / `refuted` seams that already exist in `Prune.lean`.

Also note the paper's "pruning is not adding" analogue: a decomposition set B *partitions* the search space; it never discards cubes. Our `Demo.notDescending_unsound` negative test enforces the same discipline on prunes.

## 7. Difficulty signals for a case (checklist)

1. Estimated d-hardness ĥ(q) = (2^k/N)·Σ ξ^j in unit propagations (Def. 3, Thm 2, eq. 8).
2. Sample coefficient of variation s(ξ)/ξ̄ for the cubes of q — a heavy-tail flag; if large, the mean is untrustworthy and the case should be treated as hard (eq. 13).
3. ρ̂_q = UP-refuted fraction of random cubes (Def. 4; Bernoulli, Var ≤ 1/4; N ≥ ln(2/δ)/(2ε²) by Hoeffding — the latter is our inference).
4. Number of hard cubes (1 − ρ̂_q)·2^k, the size of Γ for that case.
5. Budgeted-solve outcome: solved within c conflicts / censored (cheap first filter; heavy tails make the mean biased).
6. UP weights w_i = w_i⁺ + w_i⁻ of the cells of C_q (§6): a highly propagating instance is one where fixing a few cells collapses the rest; also identifies which cells to branch on.
7. Post hoc: DRAT/LRAT proof size and check time of the solved case (Table 1 shows check time is the larger cost).
8. Aggregate: |S| × mean ĥ over a uniform sample of S, with Theorem 3's N (relative-variance-driven) as the stopping rule.
9. **[inference, untested]** Slack in the KST counting inequality Σ_i C(r_i, t) ≤ (s−1)·C(n, t) and its column dual, and the number of "tight" rows/columns: a structural, solver-free feature that an evolved prune can read directly, worth correlating against ĥ.

## 8. Extracted lemmas and their Lean status

All Lean remarks are relative to the Mathlib-free ZarPrune core (`sumFin`/`allFin` over `Fin`, `HasKst` as increasing index tuples, `Profile` = row/column sum vectors, `Prune` = `kill` + `sound`).

| # | Name | Statement | Hypotheses | Lean status |
|---|---|---|---|---|
| L1 | Decomposition identity (Thm 2, discrete form) | Σ_{β∈{0,1}^k} t(β) = Σ_i ξ_i · #{β : t(β) = ξ_i}; with uniform β this is 2^k·E[ξ]. | t : {0,1}^k → ℕ any function. | Pure finite-sum bookkeeping; provable over ℕ with `sumFin` in tens of lines. **Not needed**: it is a reward-side fact and never enters soundness. |
| L2 | Chebyshev sample-size (Thm 3) | N ≥ Var/(ε²δE²) ⇒ (ε,δ)-approximation of 2^k·E[ξ] by (2^k/N)Σξ^j. | i.i.d. samples, deterministic solver. | Needs real numbers / probability; not formalisable in the Mathlib-free core and there is no reason to: it only governs how many samples the reward uses. |
| L3 | Cartesian-product cover (§7) | C unsat ⇔ every UP-trivial piece C[β/B_i] is unsat and every C[γ], γ ∈ Γ₁×⋯×Γ_s, is unsat. Our form: every valid A has profileOf A ∈ (all row vectors) × (all column vectors), so if each pair is killed by p or listed in `survivors`, `cover` holds. | Row-only, column-only and pair prunes are all sound `Prune P` terms (composable via `Prune.or`); `survivors` is exactly the enumerated non-killed pairs. | This is the `cover` hypothesis of `upper_bound_of_cover`. Provable without Mathlib: `rowSum_le`/`colSum_le` bound every coordinate by n/m, so an exhaustive `List` of bounded vectors with a completeness proof (`∀ v, (∀ i, v i ≤ n) → v ∈ allVecs`) gives it; one to two hundred lines, all elementary induction on `Fin`. The enumerate-partitions half (unordered) is the piece the README says is not written yet. |
| L4 | Grouped-cube lemma (formula 14) | C ∧ ⋀_i (u_i ≡ σ_i) ∧ (⋁_i u_i) unsat ⇔ every C ∧ σ_i unsat. | σ_i cubes over C's variables, u_i fresh. | Propositional; would only be needed if we batch trivial survivors into one certificate. Then it must be part of the encoding-correctness theorem behind `refuted` (a statement about `Mat`, not about CNF, in the current core, so it would first need a CNF semantics layer). Moderate effort, low priority. |
| L5 | Prune-as-sub-solver (our reading of Def. 4) | A verified `Prune P` is a polynomial sub-solver P on the case abstraction; ρ = fraction of cases it kills; hard set Γ = survivors. | — | Not a theorem; a dictionary. The useful corollary is that ρ̂ for a candidate prune is a Bernoulli estimate with Var ≤ 1/4, i.e. the *kill rate* of a prune is cheap to estimate even before its `sound` field elaborates (proxy reward for non-compiling candidates). |

The paper contains **no counting lemma about K_{s,t}-free matrices**; the KST prune named in the Lean README remains the next real prune to prove and is untouched by this source.

## 9. Open questions raised for our design

1. Uniform cube sampling inside C_q draws many cubes that violate the fixed row/column sums and are refuted instantly; Theorem 2 stays exact, but the informative (hard) cubes become rare. Should cubes be sampled conditionally on the cardinality constraints (e.g. sample a full row consistent with r_i), and does that bias the difficulty ranking across pairs?
2. Calibration of k (cube size) for m·n-cell instances: at what k do hard cubes first appear (the paper's Fig. 5 threshold was |B| ≈ 20 for ~3000-variable LEC instances)?
3. Is the coefficient of variation of ξ across partition pairs small enough that N_out ≈ 50–200 sampled survivors estimate the total remaining work to within a factor of two? If not, stratify by row partition.
4. Do the propagation-count rankings transfer between solvers (CaDiCaL vs. Kissat vs. Glucose)? The paper only shows propagations ≈ conflicts within one solver.
5. Does per-case d-hardness predict LRAT certificate size (the cost that Lean's `refuted` seam will actually pay)? Table 1 suggests checking dominates solving.
6. Should the *decomposition itself* be evolved (which invariant to branch on, as the paper evolves B), rather than only the prunes over Tan's fixed row/column split? Their "triples best" result argues for at least experimenting with a third split axis.
7. Can a cheap structural feature (KST slack, tight rows) replace sampling as ĥ(q) once a correlation is established on a few hundred sampled cases? That would make the reward solver-free and fully deterministic.
8. Proxy reward for a candidate whose Lean `sound` does not elaborate: the ρ̂/kill-rate is measurable, and unsoundness can be *disproved* cheaply by finding a valid matrix in a killed case (SAT with a small budget). How much of the evolutionary signal should come from this proxy versus from verified prunes only?

## 10. Citation strings

* D. Chivilikhin, A. Pavlenko, A. Semenov. Decomposing Hard SAT Instances with Metaheuristic Optimization. *International Journal of Artificial Intelligence* 21(2):61–92, 2023. arXiv:2312.10436.
* A. Semenov, D. Chivilikhin, A. Pavlenko, I. Otpuschennikov, V. Ulyantsev, A. Ignatiev. Evaluating the Hardness of SAT Instances Using Evolutionary Optimization Algorithms. *CP 2021*, LIPIcs 210, 47:1–47:18. doi:10.4230/LIPIcs.CP.2021.47.
* A. A. Semenov, A. Pavlenko, D. Chivilikhin, S. Kochemazov. On Probabilistic Generalization of Backdoors in Boolean Satisfiability. *AAAI 2022*, pp. 10353–10361.
* R. Williams, C. P. Gomes, B. Selman. Backdoors to Typical Case Complexity. *IJCAI 2003*, pp. 1173–1178.
* B. Doerr, H. P. Le, R. Makhmara, T. D. Nguyen. Fast Genetic Algorithms. *GECCO 2017*, pp. 777–784.
* C. P. Gomes, A. Sabharwal. Exploiting Runtime Variation in Complete Solvers. *Handbook of Satisfiability*, 2nd ed., 2021, pp. 463–480.
* M. Heule, O. Kullmann, S. Wieringa, A. Biere. Cube and Conquer: Guiding CDCL SAT Solvers by Lookaheads. *HVC 2011*, LNCS 7261, pp. 50–65.
* A. Pavlenko, A. A. Semenov, V. Ulyantsev. Evolutionary Computation Techniques for Constructing SAT-Based Attacks in Algebraic Cryptanalysis. *EvoApplications 2019*, LNCS 11454, pp. 237–253 (EvoGuess).

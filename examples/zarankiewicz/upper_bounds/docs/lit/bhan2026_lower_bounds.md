# Bhan, Nobili, Raghuraman, Langer (2026) — New Bounds for Zarankiewicz Numbers via Reinforced LLM Evolutionary Search

**Status of this note.** Primary source read in full (arXiv:2605.01120v3, 17 pp., PDF downloaded and
text-extracted; Figure 2 read from the rendered page image). Cross-checked against the local
implementation that produced the paper's results:
`examples/zarankiewicz/zarankiewicz_<m>,<n>/{evaluator.py, initial_program.py, config_phase_{1,2,3}.yaml, .n_sota, .best_matrix.npy, openevolve_output/}`.
Everything marked **[paper]** is stated in the paper; **[local]** comes from the code/logs in this
repo; **[inferred]** is my own analysis.

## 1. Bibliographic record

- Jay Bhan (MIT), Nicole Nobili (ETH Zürich), Srinivasan Raghuraman (MIT), Patrick Langer (ETH / Stanford).
  *New Bounds for Zarankiewicz Numbers via Reinforced LLM Evolutionary Search.* Working paper,
  arXiv:2605.01120 [cs.AI]. v1 1 May 2026, v2 7 May 2026, v3 25 Aug 2026. Bhan and Nobili are
  joint first authors (order by coin flip).
- Local PDF copy used for this note: `/private/tmp/claude-501/-Users-jaybhan-Downloads-openevolve/88cfd163-c49a-4b7d-ba6f-24d477547106/scratchpad/lit/bhan2026.pdf` (scratch; re-download from `https://arxiv.org/pdf/2605.01120`).
- Thesis proposal cites it as [9]; it is the "completed work" on which the upper-bound thesis builds.

## 2. One-paragraph summary

**[paper]** Lower bounds for Z(m,n,3,3) are obtained by evolving *programs that construct* m×n 0/1
matrices with no all-ones 3×3 submatrix, using OpenEvolve (open-source AlphaEvolve) with a reward
tailored to the problem (Nagda et al.'s Ramsey-number recipe: a primary matrix scored by ones with
2×/4× multipliers for matching/beating the running record n_SoTA, −1 if it contains any K_{3,3}; plus
a second "prospect" matrix scored by violations relative to the random expectation). Over 51 (m,n)
cases in 8≤m≤16, 16≤n≤23 it proves three exact values — **Z(11,21,3,3)=116, Z(11,22,3,3)=121,
Z(12,22,3,3)=132** — by matching the Davies–Gill–Horsley / Tan upper bounds, gives new lower bounds
for the other 41 open cases (several within one edge of the upper bound), and reproduces four known
optimal constructions. Cost: $15–30 per (m,n) case, ~10 min per 100-iteration phase.

## 3. Problem formulation as used by the paper

**[paper]** Z(m,n,s,t) = max ones in an m×n binary matrix such that no s rows and t columns
intersect in an all-ones block. The LLM never emits a matrix directly; it emits a *Python program*
whose evolve-block returns `(G1, G2)`. Matrices rather than graphs were chosen because graph-reasoning
benchmarks (GraphArena, GraphEval36K) show LLMs handle matrices better.

**[local]** The K_{3,3} check used everywhere (evaluator and every evolved program) is the codegree
form: for each row triple (i,j,k), `|N(i) ∩ N(j) ∩ N(k)| ≥ 3` is a violation; the evaluator counts
`C(k,3)` violations per triple with `k` common columns (`count_kst_violations`). This is the same
predicate as `ZarPrune.HasKst` but stated via column counts instead of an increasing index tuple —
see §10 for the bridge lemma.

## 4. Scoring algorithm (Algorithm 1, quoted)

**[paper]** "Algorithm 1: One Phase of Zarankiewicz Program Search"

```
Require: m, n, s, t
1: P ← {p_base}            (p_base returns sparsely populated matrices)
2: n_SoTA ← 0
3: for 100 iterations do
4:   p_new ← LLM_Mutation(Select(P))
5:   (M1, M2) ← p_new.run()
6:   if M1 has no violations then
7:     c1, c2 ← count_ones(M1), count_ones(M2)
8:     S1 ← 4·c1 if c1 > n_SoTA ; 2·c1 if c1 = n_SoTA ; c1 otherwise
9:   else
10:    S1 ← −1
11:  end if
12:  e_expected ← C(M,S)·C(N,T)·(c2/(M·N))^(S·T)
13:  S2 ← ½ · max(0, 1 − count_viol(M2)/e_expected)
14:  score(p_new) ← S1 + S2
15:  P ← P ∪ {(p_new, score)}
16: end for
17: return argmax_p score(p)
```

**[local]** `evaluator.py` implements exactly this, with two additions:

- `combined_score = (S1 + S2) / (4 · KST_UPPER_BOUND)` — normalised so a new record scores ≈ 1.
  Not clamped at 0, so invalid programs land in [−0.0023, −0.0012] and are still ordered by S2.
- `n_SoTA` is **global mutable state on disk** (`.n_sota`, plus `.best_matrix.npy`), shared by the
  4 worker processes; `_update_n_sota` writes before the comparison result is returned, so a program's
  recorded score depends on *when* it was evaluated.
- The evaluator returns an **artifact** `best_known_matrix`: the current best valid matrix rendered
  as text with row and column degrees. OpenEvolve injects artifacts into the next prompt
  (`prompt/sampler.py::_render_artifacts`), so the LLM sees the record matrix verbatim.

### 4.1 Quantitative analysis of the reward **[inferred, from the code]**

With UB = 108 (the (10,21) case):

| situation | combined_score |
|---|---|
| valid G1, 97 ones, below record | 0.2257 |
| valid G1, 98 ones, ties record | 0.4549 |
| valid G1, 98 ones, *sets* record | 0.9086 |
| invalid G1, any G2 | −0.0023 … −0.0012 |
| max possible G2 contribution | +0.00116 (= less than half of one G1 edge at the 1× multiplier, 0.00231) |

So: (i) the score is a **rank signal with a cliff**, not a smooth objective — a one-edge
improvement quadruples the score, and the *same* program re-evaluated after the record moves loses
a factor of four; (ii) the **G2 "prospect" term is effectively a tie-breaker among invalid programs
only** — for valid programs it is dominated by a single edge; (iii) the G2 term is **gameable**:
`G2 = G1.copy()` with a valid `G1` gives `count_viol = 0`, hence the maximum bonus 0.5, with no
"prospecting" at all. The evolved programs did exactly this (Algorithms 3, 4, 7, 8 set `G2 = G1.copy()`;
Algorithm 5's comment literally says "Set G2 = G1 to try for symmetry/matching bonus"; Algorithm 2
returns a row/column permutation of G1; Algorithm 6 returns `np.ones`, which yields bonus 0). The
expected-violation heuristic itself is fine (it is the Erdős–Rényi count of all-ones s×t blocks at
the given density: e.g. at 10×21 density 0.476 it predicts ≈ 201 violations, at density 0.62 ≈ 2131),
but as wired it did not shape the search of valid programs.

Verified in the phase-1 log of (10,21): 100 iterations → 55 valid, 45 invalid, 8 of the 45 were
crashes/timeouts (`num_edges = 0`); the seeded `initial_program.py` itself is *invalid* (68
violations, score −0.0015) — the search starts from nothing.

## 5. OpenEvolve configuration **[local, the paper gives only the model mix]**

| setting | phase 1 | phase 2 | phase 3 (and later) |
|---|---|---|---|
| `max_iterations` | 100 | 100 | 200 (paper text says 100; local YAML says 200) |
| `checkpoint_interval` | 10 | 10 | 10 |
| primary model | `google/gemini-2.0-flash-001` @ 0.6 | Flash @ 0.6 | Flash @ 0.4 |
| secondary model | `anthropic/claude-3.7-sonnet` @ 0.4 | `anthropic/claude-opus-4.6` @ 0.4 | Opus 4.6 @ 0.6 |
| `api_base` | `https://openrouter.ai/api/v1` (all phases) | | |
| `temperature` / `top_p` / `max_tokens` / `timeout` | 0.7 / 0.95 / 8192 / 600 s (all phases) | | |
| `prompt.num_top_programs` | 3 | 4 | 4 |
| `prompt.num_diverse_programs` | — | 3 | 3 |
| `database.population_size` | 60 | 70 | 70 |
| `database.archive_size` | 25 | 30 | 30 |
| `database.num_islands` | 4 | 5 | 5 |
| `elite_selection_ratio` / `exploitation_ratio` | 0.3 / 0.7 | 0.3 / 0.6 | 0.3 / 0.6 |
| `evaluator.timeout` | 60 | 90 | 90 |
| `cascade_evaluation` | **false** (stage functions exist but are unused) | | |
| `parallel_evaluations` | 4 | 4 | 4 |
| `use_llm_feedback` | false | | |
| `diff_based_evolution` / `allow_full_rewrites` | false / **true** (whole-block rewrites) | | |
| `max_code_length` | default | 100000 | 100000 |
| MAP-Elites `feature_dimensions` | OpenEvolve default `["complexity","diversity"]`, 10 bins (code length and edit distance; nothing problem-specific) | | |
| random seed | 42 (from the log) | | |
| per-program subprocess timeout | 25 s (`run_with_timeout`) | | |

Ad-hoc later phases (`extensively_tested/zarankiewicz_10,21_done/config_phase_{4,5,6}.yaml`) vary
the Flash/Opus mix (0.75/0.25, 0.25/0.75) and inject increasingly explicit prompt text, e.g.
phase 5: *"You have exhausted the potential of randomized 'destroy-and-repair' heuristics ... You must
now transition from heuristic search to systematic optimization"*; phase 6: *"The known bound is 108
edges. You have a best program discovering a certificate of 106 edges ... sticking too close to the
known best matrix will likely not succeed."* — the (10,21) case still finished at 106/108 after
six phases (~900 iterations).

The system prompt (all phases) is a fixed block: expert-combinatorialist persona, the K_{3,3}
definition spelled out for a specific m×n, a menu of techniques (cyclic/dihedral symmetry,
near-regular degree sequences, projective-plane/design/difference-set seeds, structured deletions,
circulant and block-circulant families, extending smaller matrices), and the instruction
*"Randomized search approaches are discouraged; structured and algebraic methods are strongly
preferred."* Phase 3 adds *"The record is <UB>. You must match that."*

## 6. Search schedule and stopping rule

**[paper]** Three phases of 100 iterations per (m,n); model mix as above. Cheaper models (Flash,
Sonnet) "were effective in designing a base program, but creating optimizations that brought the lower
bound closer to the tight upper bound required more state of the art models." After phase 3: if
n_SoTA improved during phase 3 and the upper bound is not yet matched, run another 100-iteration
phase with the phase-3 mix; repeat until the bound is matched or a full phase yields no improvement.

**[local]** Each phase is a fresh `openevolve-run.py` invocation resumed from the previous
checkpoint; `.n_sota` persists across phases. The (11,17) run
(`known_bounds/zarankiewicz_11,17_success/plot_figure2.py`) records the best-valid-edge trajectory at
68 checkpoints (680 iterations): 67, 78, 89 (iter 40), 91, 92, 93 (iter 230), 94 (iter 500), 95, 96
(iter 670) — long plateaus, one-edge steps, and the last two edges cost ~180 iterations.

## 7. Results

**[paper]** 51 cases run: 44 open + 7 previously established. New exact values:
Z(11,21,3,3)=116, Z(11,22,3,3)=121, Z(12,22,3,3)=132. Lower bounds for the other 41 open cases.
Reproduced known optimal constructions for (8,23)=94, (9,22)=100, (15,16)=123 (run as 16×15),
(16,16)=128; did *not* reach the known value for (10,20) (99 vs 102) and (11,18) (97 vs 101).
The paper says 4 of 7 established cases were replicated; only 6 two-number "previously established"
cells are visible in Figure 2, so the 7th is not identifiable from the figure (the local
`known_bounds/zarankiewicz_11,17_success` run of 7 Apr 2026 reached the established 96 and may be it —
**open question**, see §12).

### 7.1 Figure 2 transcribed (upper bound / lower bound; `*` = tight; single number = previously established, not run)

| m \ n | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|
| 8  | 70 | 74 | 77 | 81 | 84 | 87 | 90 | 94/94* |
| 9  | 77 | 81 | 85 | 89 | 93 | 96 | 100/100* | 104/103 |
| 10 | 85 | 90 | 94 | 98 | 102/99 | 108/106 | 111/110 | 115/112 |
| 11 | 92 | 96 | 101/97 | 108/102 | 112/111 | 116/116* | 121/121* | 125/118 |
| 12 | 99 | 108/102 | 113/108 | 118/110 | 122/113 | 127/116 | 132/132* | 136/125 |
| 13 | 107 | 116/106 | 121/115 | 125/114 | 130/119 | 135/127 | 140/137 | 145/135 |
| 14 | 115 | 124/118 | 129/124 | 135/121 | 140/125 | 145/131 | 150/137 | 155/138 |
| 15 | 123/123* | 132/125 | 138/132 | 143/132 | 149/138 | 154/139 | 160/143 | 165/149 |
| 16 | 128/128* | 141/128 | 146/130 | 152/132 | 158/146 | 164/147 | 169/149 | 175/158 |

Upper bounds are from Tan (2022) and Davies–Gill–Horsley (2026); the paper notes DGH improved
Roman's bound in 29 of the 44 open cases. Every lower bound in the table matches the `.n_sota`
file of the corresponding local directory (checked programmatically).

**Gap-1 cases** (one edge from closure, the natural first targets for an *upper-bound* attack):
(9,23): 104/103, (10,22): 111/110, (11,20): 112/111. Gap 2: (10,21): 108/106. Gap 3: (10,23): 115/112,
(13,22): 140/137, (10,20): 102/99 (established, so the UB is known tight there — a good *calibration*
instance for the SAT pipeline, since the answer is known).

### 7.2 Profiles of the record matrices **[local, verified K_{3,3}-free by recomputation]**

These are exactly the profiles a prune may never kill at the corresponding weight (see §10, L3).

| case | ones | sorted row sums | sorted column sums |
|---|---|---|---|
| (11,21) = 116 | 116 | 11^6 10^5 | 6^11 5^10 |
| (11,22) = 121 | 121 | 11^11 | 6^11 5^11 |
| (12,22) = 132 | 132 | 11^12 | 6^22 (biregular) |
| (9,23) ≥ 103 | 103 | 13 12 12 11^6 | 5^12 4^10 3 |
| (10,22) ≥ 110 | 110 | 11^10 | 6^5 5^12 4^5 |
| (11,20) ≥ 111 | 111 | 11^3 10^6 9^2 | 6^11 5^9 |
| (10,21) ≥ 106 | 106 | 11^6 10^4 | 6^5 5^12 4^4 |

The (12,22)=132 construction is a (11,6)-biregular graph built from the quadratic residues mod 11
(`D1 = QR mod 11`, `D2 = complement`) plus one greedily filled row; (11,22)=121 is two 11×11
circulant blocks. The exact values are all extremely regular — consistent with the row-sum-partition
branching in Tan's method being a good decomposition (the surviving partitions near the bound are
few and near-regular).

## 8. Cost and time

**[paper]** "$15 to $30" per (m,n) case "depending on how many iterations the model needed";
"every phase taking around 10 minutes". **[local]** Phase 1 of (10,21): 100 iterations in 9 min 14 s
wall-clock with 4 parallel workers; mean 21.3 s per iteration (LLM latency; evaluation itself is
≈0.05 s). (11,17) phases: 7 min 14 s and 5 min 57 s per 100 iterations.
**[inferred]** ≈ $0.05–0.10 per iteration; 51 cases × $15–30 ≈ $0.8k–1.5k for the whole paper; a
`$16` budget buys ~150–300 iterations of this pipeline, i.e. one or two phases of one case.

## 9. What worked, what failed, and the lessons the paper draws

**[paper]** Three categories of winning programs (App. A, reported verbatim):
1. *Explicit matrix, minimal computation* (Alg. 5 for (8,23), Alg. 7 for (15,16)) — the LLM
   hard-codes a matrix it read from the `best_known_matrix` artifact or from a previous program.
2. *Circulant structure* — Alg. 3 (11,22): exhaustive enumeration of two shift sets S1,S2 ⊂ Z_11
   with the pairwise-difference condition `|S ∩ (S+d1) ∩ (S+d2)| summed over both blocks ≤ 2`;
   Alg. 4 (12,22): QR/non-residue circulant blocks + greedy fill.
3. *Randomized perturbation and repair* ("rip-up and repair"): start from the best explicit matrix,
   delete a random 2–12 ones, greedily re-add in a shuffled order under a local `can_set` test,
   iterate with a 1-out-2-in swap move (Alg. 2 for (11,21), Alg. 6, Alg. 8). The paper flags this as
   "a potentially more general search framework" and lists consolidating it as future work.

**[paper]** Explicit design choices/lessons: avoid graph representations (use matrices); do not let
the LLM output matrices, evolve programs; "we observed that some candidates relied too heavily on
random search; we therefore explicitly discouraged mutations that reduced the construction method to
pure stochastic search"; cheaper models for the base program, frontier model (Opus 4.6) for the
last edges; costs and runtimes should be reported; future work item 3: "discovery via evolutionary
algorithms can be extended beyond generating constructions to automatically proving upper bounds."

**[local, inferred]** Additional observations not in the paper:
- The artifact channel (record matrix + degrees in the prompt) is what turned "program evolution"
  into matrix-level local search: most winners embed the record matrix and perturb it. That is
  effective but it also means the *program* is no longer generalizable — a lesson relevant to the
  `zarankiewicz_generalized` follow-up and to interpretability.
- The prompt had to be escalated by hand across phases (“You must match 108”), and it still did not
  close gap-2 cases; hand-tuned prompt escalation is not a scalable control knob.
- The reward is non-stationary (global `n_SoTA`), cliff-shaped, and the proxy term was gamed
  (§4.1). None of this broke the lower-bound search because the *validity check is exact and cheap*;
  it would break an upper-bound search where the analogous "validity" (a Lean proof) is expensive.
- MAP-Elites dimensions were code-length/edit-distance defaults; islands and migration were used
  with default settings. No experiment isolates the contribution of islands, G2, or the multipliers —
  there is no ablation in the paper.
- Roughly half of all generated programs are invalid or crash; the run tolerates this because an
  evaluation costs 50 ms.

## 10. Lemmas extractable from the source (with Lean-provability notes)

The paper proves **no pruning lemma** — it is a constructions paper. The following are (a) the one
upper-bound theorem it quotes, (b) the validity characterization every evolved program relies on,
and (c) a soundness sanity lemma its data supports. Hypotheses and provability are my assessment
against `ZarPrune` (Lean 4.34, no Mathlib; `sumFin`/`allFin` over `Fin`, `HasKst` as increasing
index tuples, `Profile` = row/column-sum vectors).

**L1 — Roman's bound (quoted as eq. (1), attributed to Roman 1975; the paper does not prove it).**
For every integer k ≥ s−1,
`Z(m,n,s,t) ≤ (t−1)·C(m,s)/C(k,s−1) + ((k+1)(s−1)/s)·n`.
Hypotheses: 1 ≤ s ≤ m, 1 ≤ t ≤ n, k ≥ s−1. As a `Prune`: `kill := decide (P.w > RomanBound P k)`,
independent of the profile (a whole-instance kill). Provability in ZarPrune: **hard**. Roman's proof
is a linear-programming/double-counting argument over s-subsets of rows; it needs counting of
subsets and a convexity step for `C(r, s−1)`. Not a good first target; the KST counting prune
already listed in `lean/README.md` is its natural precursor.

**L2 — Codegree characterization of HasKst (used by every evolved program; not stated in the paper).**
`HasKst P A ↔ ∃ R : Fin P.s → Fin P.m, Incr R ∧ P.t ≤ sumFin P.n (fun j => ind (∀ a, A (R a) j = true))`.
Hypotheses: none beyond `P.s ≤ P.m`, `P.t ≤ P.n` implicit in the existence of `Incr` tuples.
Provability: **moderate**. The (→) direction is a count-lower-bound from an injective increasing
tuple (needs `sumFin` ≥ number of distinct witnesses, i.e. a small injectivity-counting lemma over
`Fin`). The (←) direction needs "a Boolean predicate on `Fin n` true at ≥ t points admits an
increasing t-tuple of witnesses" — a selection lemma by induction on `n`, hand-rolled but routine.
This lemma is the bridge between the SAT/Python codegree checks and the Lean definition and will be
needed by any counting prune, so it is worth proving early.

**L3 — Witness-profile constraint (sanity lemma; supported by the paper's constructions, stated by me).**
If `A` is a K_{s,t}-free matrix with `weight A = L` (e.g. the three record matrices, or any matrix
from `.best_matrix.npy`), then for every `Prune P` with `P.w ≤ L`, `p.kill (profileOf A) = false`.
Hypotheses: `¬ HasKst P A`, `P.w ≤ weight A`. Provability: **trivial** — it is the contrapositive of
`p.sound` (`sound A h : ¬ Valid P A`, but `Valid P A` holds). Value: a mechanical *negative test* in
the style of `Demo.notDescending_unsound`: every accepted prune must be evaluated on the profiles in
§7.2 (for all `w ≤ L`) and must return `false`; any candidate that fires on a realized profile is
unsound and the Lean gate would in any case reject it — but this cheap Python check can reject it
*before* paying for Lean elaboration. It also gives the evaluator a free "obviously unsound" signal.

**L4 — Expected-violation heuristic (not a lemma; do not formalize).**
`E[#all-ones s×t blocks] = C(m,s)·C(n,t)·(e/(mn))^{st}` for a uniformly random matrix with `e` ones.
Useful only as a *difficulty/density* proxy (§11).

## 11. Difficulty signals suggested by the paper's data **[inferred]**

- **Gap UB − LB** for the instance (Figure 2): gap 1 means the UNSAT instance is `w = UB` and a
  single successful refutation closes the case; larger gaps need a chain of `w` values.
- **Regularity of the surviving profile**: all record matrices near the bound are (near-)regular;
  profiles far from regular at high weight are the ones counting prunes kill first, so "distance from
  the regular profile" (e.g. Σ|r_i − w/m|) is a proxy for "easy to prune" vs "must SAT-solve".
- **Density relative to the random-violation count** (L4): the ratio `count_viol / E_viol` at a
  target weight measures how far the instance is above the random threshold; near-record weights sit
  far into the "violations expected" regime.
- **Plateau length in the lower-bound run**: the number of iterations the evolutionary search spent
  stuck one edge below the UB (e.g. (10,21): ≥ 900 iterations at 106/108; (11,17): 180 iterations
  for the last two edges) is empirical evidence that either the UB is loose or the extremal
  configurations are rare — both make the UNSAT side more plausible to attack.

## 12. Open questions raised by this reading

1. Which is the 7th "previously established" case (only 6 two-number cells appear in Figure 2)?
   Is it (11,17), whose local run (`known_bounds/`) reached 96 after 680 iterations? Its evaluator has
   `KST_UPPER_BOUND = 108`, a placeholder mismatch (the same placeholder appears in
   `extensively_tested/zarankiewicz_9,23_done`, true UB 104) — worth fixing before any figure is regenerated.
2. Phase-3 length: paper says 100 iterations; every local `config_phase_3.yaml` says 200.
3. No ablation: how much did islands, the G2 term, the 2×/4× multipliers, or the Opus share matter?
   The G2 analysis in §4.1 suggests the prospect term contributed nothing for valid programs.
4. For the upper-bound thesis: can the three exact values be *re-derived* by the SAT + Lean pipeline
   as end-to-end calibration (UNSAT at w = 117, 122, 133), before attempting the gap-1 cases?
5. Does the "structured constructions" prompt bias help or hurt when the target is a *prune*
   rather than a construction? The persona/technique list will need rewriting from scratch.

## 13. Design implications for the upper-bound (pruning) system

1. **Keep the OpenEvolve skeleton, replace the reward wholesale.** Islands (4–5), population 60–70,
   full rewrites, artifacts-in-prompt, 100-iteration phases with checkpoints every 10, a cheap model
   for scaffolding and a frontier model for the hard steps — all transferable. The n_SoTA multiplier
   scheme is not: it presumes an exact, 50 ms validity check. With Lean as the check, use a
   *staged* score (parses → typechecks kill → `sound` elaborates) with cascade evaluation turned
   **on**, which the lower-bound configs left off.
2. **Make the reward stationary and smooth where the lower-bound one was not.** Score a prune by
   the number (or weighted mass, §11) of surviving cases it kills *on a fixed enumerated case set*,
   not relative to a moving global record; keep the "beats record" bonus, if at all, as a small additive term.
3. **Do not let the proxy be gameable.** The G2 term was maximized by `G2 = G1.copy()`. Any
   "almost-compiles" proxy for Lean must be immune to the analogous trick (e.g. a `kill` that always
   returns `false` elaborates trivially and prunes nothing: score must be `0` there, and L3's witness
   test must reject `kill`s that fire on realized profiles before Lean is even invoked).
4. **Feed rich artifacts.** The single most effective mechanism in the lower-bound work was showing
   the LLM the record matrix with degrees. For prunes, the analogue is: the list of surviving
   profiles with their SAT solve times, the Lean error message of the last failed `sound` proof, and
   the witness profiles from §7.2.
5. **Target selection.** Start with the calibration instances whose answer is known ((11,21)=116,
   (11,22)=121, (12,22)=132, (10,20)=102), then the gap-1 cases (9,23), (10,22), (11,20).
6. **Budget.** At ≈$0.05–0.10 per lower-bound iteration, the $16 budget is for probing what a
   Lean-writing LLM response looks like (a handful of Opus-class calls), not for a run; the paper's
   own numbers say a real phase costs $5–10.
7. **Record everything the paper did not**: per-phase cost, ablations, and the failure modes
   (invalid/crash rates, proxy gaming), so the thesis chapter can make claims the paper could not.

## 14. Reproduction pointers (local)

- Run a phase: `python openevolve-run.py examples/zarankiewicz/zarankiewicz_10,21/initial_program.py examples/zarankiewicz/zarankiewicz_10,21/evaluator.py --config examples/zarankiewicz/zarankiewicz_10,21/config_phase_1.yaml --iterations 100`; later phases add `--checkpoint .../checkpoint_100`.
- Verify any `.best_matrix.npy`: `evaluator.count_kst_violations(A, 3, 3) == 0`.
- Regenerate Figure 2: `examples/zarankiewicz/make_table.py` (reads `KST_UPPER_BOUND` from each evaluator and `.n_sota`; starred folders `*zarankiewicz_...` are the previously established cases).
- Reasoning traces (Sonnet/Flash "plans" before code): `known_bounds/zarankiewicz_11,17_success/reasoning_traces.md` — mostly generic ("I'll use algebraic constructions based on projective planes"), i.e. the stated plan rarely predicts the score.

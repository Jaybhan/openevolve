# gpt_agent — autonomous research workspace: Zarankiewicz z(m,n;s,t)

Started 2026-07-28. This directory is **additive-only**: nothing outside it
(the OpenEvolve run, its checkpoints, `.n_sota`, `instance_log.jsonl`,
`evaluator.py`) is ever modified by this research program.

## Objective (as set by the project owner)

1. Mine the existing OpenEvolve run (150 iterations, 15 checkpoints, archive of
   evolved `construct_graph(M,N)` programs for K_{3,3}-free matrices) for
   reusable mathematical ideas.
2. Synthesize a **general algebraic construction** `construct(m, n, s, t)` for
   K_{s,t}-free m×n 0/1 matrices — all m, n, s, t.
3. Pursue an exact formula for z(m,n;s,t) and the algorithm producing the
   witness matrix. Record every intermediate discovery.
4. Implement, verify computationally, derive proof conditions, compare against
   the evolved programs, hunt counterexamples.
5. Check any novelty claim against primary literature before calling it new.

## Honesty bar

The standard is the one the July 2026 Jacobian-conjecture counterexample met:
**explicit, independently checkable, reproducible**. Labels used throughout:

- `PROVEN` — has a proof (ours or cited literature).
- `CERTIFIED-EXACT` — construction edges == upper bound for that cell (proof by
  matching bounds, cell by cell).
- `VERIFIED-ALL-KNOWN` — matches every known/proven value, no proof beyond.
- `CONJECTURE` — explicit, falsifiable, tested to stated range.
- `REFUTED` — failed; counterexample recorded.

The general exact determination of z(m,n;s,t) is open since 1951; even the
order of magnitude of z(n,n;4,4) is unknown. Anything claimed here beyond
best-known must come with a checkable certificate.

## Layout

- `research_log.md` — append-only provenance log. Start here.
- `data/` — ground-truth tables extracted from `../evaluator.py`.
- `mining/` — what the evolutionary run discovered (report, extracted programs,
  per-cell reachability).
- `theory/` — precise statements + citations of known results, each verified
  numerically where possible.
- `constructions/` — the general engine `zarankiewicz.py` and its families.
- `bounds/` — upper-bound machinery, UB tables, deficit analysis.
- `results/` — comparisons, evaluator-suite scores, our own n_sota log.
- `analysis/` — formula analysis, conjectures, counterexamples.
- `harness/` — snapshot copy of `evaluator.py` (so runs here can never touch
  the live experiment's `.n_sota` / `instance_log.jsonl`) + wrapper program.

## Conventions

- Matrix convention: M×N 0/1 numpy array; forbidden pattern: s rows sharing t
  common 1-columns (an all-ones s×t submatrix). z(m,n;s,t) = max ones.
  Note z(m,n;s,t) with s acting on rows: z(m,n;s,t) = z(n,m;t,s).
- The evaluator's 161-cell suite is (s,t)=(3,3), proven-exact cells only,
  m=3..16, with holdout = every 4th cell of the sorted list.

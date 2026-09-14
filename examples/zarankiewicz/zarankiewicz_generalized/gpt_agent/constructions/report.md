# Construction engine — results on the 161 proven-exact z(m,n;3,3) cells

Date: 2026-07-28 09:22.  Engine: `constructions/zarankiewicz.py`
(full speed profile).  Every matrix behind every number below passed BOTH this
module's verifier and the snapshot reference verifier
(`harness/evaluator_snapshot.py::count_kst_violations == 0`); the two
verifiers were cross-checked on 760 random matrices across six (s,t) pairs.

## Headline

- **160/161 cells exact**, 161/161 valid, 1 edges missing in total.
- Official snapshot-evaluator score (isolated harness run, 2026-07-28 03:5x):
  **combined_score 0.995503, exact_count 160, validity 1.0,
  peak_instance_ms 0.008** (see `results/n_sota_log.md`).
  Live-run evolutionary champion for comparison: 0.7758, 110/161.
- By segment: m <= 5 (Roman rows, closed-form): 60/60 exact; m = 6..7: 35/35;
  deficit band m >= 8: 65/66.
- **78 cells are CERTIFIED-EXACT-BY-BOUND-MATCH** independently of the
  table: the verified construction meets the two-sided waterfill upper bound
  (bounds agent's `ub_waterfill` contract), so exactness there needs no
  external table.  The remaining exact cells are VERIFIED-ALL-KNOWN (they
  match Tan's proven values; the UB side is the cited literature).
- Of the **49 cells that NO program in the entire evolutionary run ever solved exactly** (miner's `cell_reachability.csv`, 142 evaluations), this engine closes **48** — all but [(10, 15)].


## Cells not exact

- 10x15: 80/81 (-1), valid, via hyperplane_f2^4(lex)|fill=nl

This is the honest open gap.  (10,15) resisted (a) the offline
large-neighbourhood repair search (8 seeds x 60 s, `run_discovery_v2.py`),
(b) the evolutionary champion (also -1 there), (c) every algebraic family
in this engine, and (d) all of the coordinator's ILP attempts (timed out).
Its waterfill profile admits e.g. 6^6 5^9 (81 edges, cost 210 of 240), but
no legal realization has been found by anyone.

Note on (9,22): initially open here too; the coordinator's ILP witness
(analysis/witnesses/w_9x22.json) was decoded into the `sts9_dressing`
family — the witness IS AG(2,3)=STS(9) re-dressed (quads {p} u L plus their
complements for the 8 lines avoiding a point p; the 4 lines through p give
a perfect matching whose 2+2 split yields the last 6 blocks) — and the
engine now derives z(9,22)=100 generatively.  It is NOT a truncation of the
Hadamard 3-(12,6,2) (that construction caps at 99 on this cell).

## Price of unification (owner mandate; `unified.py`)

`construct_unified(m,n,s,t) = realize(profile, tower(m,s,t))` — ONE
orbit-closure rule (generators EA / PTD / PRG / CYC on a label ladder,
operator alphabet orbit/complement/extension/fusion/doubling, canonical
linearizations, profile-derived size cap and quota) with the shared
canonical completion.  No per-cell branching.

- unified exact: **127/161** (all outputs valid and doubly verified);
  router (reference): 160/161.
- **price: 63 edges over 33 cells** —
  the honest headline; full per-cell comparison in
  `price_of_unification.csv`.
- worst cells: [('10', '10', 4), ('14', '15', 4), ('15', '15', 4), ('15', '16', 4), ('9', '12', 3), ('10', '11', 3)].

## Which family wins where (winner provenance per cell)

| family | cells won | exact among them |
|---|---|---|
| culik | 45 | 45 |
| hyperplane_f2k | 42 | 41 |
| roman_window | 41 | 41 |
| hadamard_3design | 15 | 15 |
| bipolar | 7 | 7 |
| pair_gdd | 3 | 3 |
| difference_family | 3 | 3 |
| cap_bothsides | 2 | 2 |
| mixed_exact | 1 | 1 |
| twin_planes | 1 | 1 |
| sts9_dressing(split=12) | 1 | 1 |

Notes on attribution: `dp_extend` / `dp_pad` / `dp_shrink` are the cross-size
DP moves applied to another family's matrix (the inner provenance is in the
CSV); "wins" means it produced the stored best matrix for the cell, not that
other families could not tie it.  Culik-regime and Roman-window cells are
often tied by several families; the first verified best is stored.

Key structural wins (exact cells attributable to a specific algebraic
structure, all verified):

- `hadamard_3design(m=12)`: the 3-(12,6,2) design (extension of the Paley
  biplane) IS the extremal structure of z(12,22) = 132 — 22 six-blocks
  covering every triple exactly twice, capacity-saturated.
- `hadamard residuals`: point-deleted 3-(12,6,2) closed the m = 9..11
  elongated cells (e.g. 9x21, 10x16..20, 11x17..18).
- `cap_bothsides_f2^4`: z(16,16) = 128 (isomorphic to Tan's witness).
- `bipolar` / `bipolar-x`: z(8,17)..z(8,22) — poles + partial-triple-packing
  pentads (+ reduced equator), distilled from discovered witnesses into a
  derived family.
- `twin_planes8`: z(8,23) = 94 — complementary AG(3,2) plane pair extended
  by a point each, plus all other planes.
- `pair_gdd`: z(11,16) = 92 — blown quotient triangles over 5 row-pairs +
  singleton, transversal code = union of two cosets of a 2-dim binary code.
- `diff_qr0(11)` (QR(11) u {0} orbits): z(11,21) = 116.
- `sts9_dressing`: z(9,22) = 100 — AG(2,3)/STS(9) re-dressed (decoded from
  the coordinator's ILP witness; see "Cells not exact" note).
- `culik`: all 41 elongated-regime cells, exact by Culik's theorem.
- `roman_window`: rows m <= 6 and m = 7, n >= 14 by the Roman/Tan closed
  form (cited, not claimed as ours).

## (2,2) sanity — projective planes

| q | size | edges | (q+1)(q^2+q+1) | exact | K_{2,2}-free |
|---|---|---|---|---|---|
| q=2 | 7x7 | 21 | 21 | yes | yes |
| q=3 | 13x13 | 52 | 52 | yes | yes |
| q=4 | 21x21 | 105 | 105 | yes | yes |
| q=5 | 31x31 | 186 | 186 | yes | yes |

## General-(s,t) spot checks (freeness verified by both verifiers)

| cell | edges | free | winning family |
|---|---|---|---|
| z(6,8;2,3) | >= 25 | yes | sum_layer |
| z(8,6;3,2) | >= 25 | yes | twin5 |
| z(7,9;3,4) | >= 44 | yes | hyperplane_f2^3(lex) |
| z(8,8;4,4) | >= 51 | yes | greedy_lex_derived: waterfill-guid |
| z(6,20;3,4) | >= 72 | yes | hyperplane_f2^3(lex) |
| z(9,9;4,3) | >= 56 | yes | ext_col[ext_row[sum_layer |
| z(10,10;2,3) | >= 45 | yes | ext_row[diff_triangular(k=5) |
| z(5,5;4,2) | >= 20 | yes | culik: EXACT regime n>=5, z=(s-1)n |

Composition soundness spot: side-by-side of two K_{3,3}-free 9x12 matrices
is K_{3,5}-free: verified.
Norm graph (q=3, 18x18, 144 edges) verified K_{3,3}-free.

## Provenance / honesty

- The engine is generative: every matrix is derived from (m, n, s, t) at
  build time; there is no stored answer table and no witness lookup.
- Two families (`bipolar`, `twin_planes8`, and pair_gdd's coset rule) were
  *distilled* from witnesses found by an offline randomized repair search
  (`offline_discovery.py`, seeds recorded, all witnesses re-verified); the
  shipped families are deterministic reconstructions of those structures,
  and the engine re-derives the cells without consulting the witnesses.
  The witness archive is kept for audit in `found_witnesses.json`.
- Bounded exact packers (node-capped DFS) finish several families
  ("fill=ye" in provenance).  They are deterministic and budgeted, but
  search-shaped: the report flags that cells 8x17..8x23 and 11x16 depend on
  them.  Everything else is formula + greedy + verification.
- Roman window formula and T33 packing numbers are cited published results
  (theory agent's `theory/published_data.py`); waterfill is the counting
  bound, used only as an upper-bound early-stop, never as a target value
  taken on faith.

# Mining Report: z(M,N;3,3) evolutionary run (checkpoints 10-150)

## 0. Scope, method, and an honesty note on score-function drift

**Primary subject.** Everything numbered below is mined from
`openevolve_output/checkpoints/checkpoint_{10..150}/programs/*.json` (138 distinct
program ids, 112 distinct code bodies once exact duplicates from island-migration
cloning are collapsed), `openevolve_output/best/{best_program.py,best_program_info.json}`,
`instance_log.jsonl` (142 fully-parsed JSON lines, each covering all 161 scored
cells; 0 malformed), `config_phase_{1,2,3}.yaml`, and `initial_program.py`.

**Method.** I unioned program records across every checkpoint (a program present
at checkpoint 50 but absent from checkpoint 150 was kept), hashed code bodies to
find true duplicates, grepped for structural/mathematical signatures (e.g. the
literal line `(a & x).bit_count() & 1`, which is the F_2^d parity/hyperplane
tell regardless of variable naming), and then **read the actual source** of every
program discussed below rather than inferring from docstrings alone. Where a
claim below is quantitative (an exact_count, a combined_score, a family-size
count, an ablation result), I recomputed it myself by re-running the *current*
`evaluator.py` in-process against the program's own extracted source — I did not
simply copy numbers out of JSON metadata without checking at least a sample
against the live evaluator.

**Score-function drift — verified, not assumed.** I was told metrics from
different times are not comparable, and I checked what that means concretely:

- Within checkpoints 10-150 (the subject of this report), I re-ran the current
  `evaluator.py` on five programs spanning checkpoint 10 through checkpoint 100
  (`d2476bee`, `e0737f4d`, `1a4e0089`, `827523fe`, `ae315099`) and got
  **exact matches** to 4 decimal places against their recorded `exact_count` and
  `combined_score` every time. So *within this checkpoint range*, metrics are
  directly comparable to each other and to the champion's — the schema and
  formula did not drift here.
- The drift is real, but it is *outside* this range, in three earlier archived
  attempts (`run1_archive_20260726_114902`, `run2_archive_20260726_142140`,
  `run3_archive_20260726_203310`), which sit chronologically before checkpoint 10
  and used a **different metrics schema entirely**: run1's and run2's
  `best_program_info.json` have only `{validity, violation_count,
  combined_score}` — no `exact_count`, no `peak_instance_ms` — and run2's best
  program scores **combined_score = 0.8907**, far above anything in checkpoints
  10-150 (max 0.7758), which is only possible because the scoring formula and/or
  problem set behind that number were different, not because that program is
  better. run3 adds `exact_count` but still lacks `peak_instance_ms` (the metric
  that encodes the current 25ms-per-instance CPU budget). **Conclusion: numbers
  from run1/run2/run3 are quoted below only as qualitative color (what kind of
  program was tried, per evaluator.py's own docstring, which independently
  narrates this same history) and are never compared numerically to anything in
  checkpoints 10-150.** Everything else in this report — every exact_count,
  every combined_score, every family-size count — is either taken directly from
  checkpoint JSON that I independently reproduced, or freshly recomputed by me
  against the live evaluator.

All 161 scored cells and all their evaluator.py-defined structure (40-cell
holdout, area-weighted scoring, 25ms CPU budget, 0.5·is_exact + 0.5·ratio) are
as documented in `evaluator.py`; I did not need to change any of that reading.

---

## 1. Taxonomy of mathematically distinct construction strategies

Five distinct mathematical mechanisms appear across checkpoints 10-150, plus one
pre-history mechanism (before checkpoint 10) that motivated the current
evaluator's CPU budget. Ranked roughly by how much of the run's compute they
consumed / how far they got.

### Era 0 (pre-history, before checkpoint 10): catalog-of-classical-designs + certified greedy augmentation

Not part of the checkpoint range under study, but directly relevant: `evaluator.py`'s
own docstring for `PER_INSTANCE_TIME_LIMIT` narrates that "through phase 2 every
top-scoring program was a greedy packer plus hill climbing" reaching
combined_score ≈ 0.706 at ≈103/161 exact but up to **724-867ms per instance** —
three orders of magnitude over the current 25ms budget. I located concrete
specimens of this era in the archived `run1_archive`/`run2_archive`/`run3_archive`
directories (their own `terminal_phase{1,2}.log` naming, and file mtimes, place
them chronologically before checkpoint 10 of the studied run). Representative
mechanism, read directly from `run2_archive_20260726_142140/openevolve_output/best/best_program.py`
(combined_score 0.8907 under a since-changed formula — **not comparable**, see
§0): a hand-assembled *menu* tried every evaluation — Paley Hadamard designs
(q=7,11), a Walsh/affine-dot-product graph on F_2^4, AG(4,2) affine hyperplanes,
Kollár–Rónyai–Szabó norm graphs Γ(3,p) — and then ran a `_augment` routine that
does up to 8 rounds of "certified-safe" greedy edge-addition (`_fill_pass`),
alternating between rows and columns, each addition individually checked against
a codegree certificate but the *overall* procedure a repeated-improvement loop,
i.e. hill climbing in every respect that matters for the anti-search rule.
`run3_archive`'s best program (`5ad97701`, exact_count=95, not comparable) goes
further: an explicit backtracking maximizer (`_max_block`, branch-and-bound with
pruning) wrapped in a `time.process_time()` deadline of `0.15 + 0.015*(M+N)`
seconds — 150-450+ ms, an order of magnitude over 25ms. **Why it was abandoned**:
per evaluator.py's own comment, the team added `PER_INSTANCE_TIME_LIMIT = 0.025`
specifically because nothing else in the score made search unprofitable, then
re-seeded evolution with `initial_program.py` (a genuinely O(1)-per-cell
construction, see Family 1 below) to restart under the enforced budget. That
restart is checkpoints 10-150, the subject of everything below.

### Family 1: Cyclic difference family on Z_M (the seed)

**Mechanism.** Rows = Z_M. Column `j` carries block `B_j = {j + d mod M : d ∈
D_{c(j)}}` where `D_k` is a *rotated prefix of the triangular numbers*
`t_i = i(i+1)/2 mod M` — a closed-form, Sidon-like offset sequence chosen because
`t_i - t_j = (i-j)(i+j+1)/2` depends only on the pair `(i-j, i+j)`, making
difference-collisions rare. Two rows share a column count equal to the
multiplicity of their difference in the pooled difference multiset of the base
blocks; keeping every nonzero residue's multiplicity ≤ 2 makes every *pair* (and
hence every triple) of rows share ≤ 2 columns — no triple is ever inspected. A
final canonical repair pass ("scan triples in lex order, drop the
highest-indexed offending column-entry") patches the rare cases where the
triangular offsets do collide three times.

**Representative:** `f2103a45` (generation 0, iteration 0 — this **is**
`initial_program.py`, byte-identical, confirmed with `diff`). exact_count = **4/161**,
combined_score = **0.4024**.

**Cells solved:** essentially none exactly, despite reasonable density (the
0.40 combined_score with near-zero exactness implies ≈ 80% density on average
but the exact optimum only 4 times).

**Why it plateaued/died — completely and immediately.** I read the code of all
seven of the seed's direct (generation-1) children logged at checkpoint 10
(iterations 1-8: `430e64fa`, `58f5f12f`, `98036018`, `d9ac402f`, `d61baa70`,
`26372621`, `ef47dd06`). **Every single one abandoned the cyclic-difference-family
principle entirely** — none is a tweak of the triangular-offset idea. Six are
norm-graph attempts (`NG(q,3)` for q=3 or 4, in different field encodings) and
one (`d9ac402f`) is an inversive/Möbius-plane attempt — i.e. the whole
population jumped straight to Family 3 below on the very first mutation round.
None of these first children beat the seed by much (best is `ef47dd06` at
21/161 exact); the elegant single-principle seed was simply never developed
further by anyone. My reading of why: the repair step ("drop the
highest-indexed offending entry") is a generic safety net with no relationship
to the specific extremal design at any given M — it guarantees validity, not
exactness, so it lands on the true optimum only by chance.

### Family 2: PG(3,2)/AG(4,2) hyperplane-character incidence — the dominant, eventual-champion lineage

**Mechanism.** Rows = (a prefix of) the 15 nonzero vectors of F_2^4, in raw
integer order 1..15 (not a Singer/optimized order). For each nonzero
`a ∈ {1,...,15}`, column `block_a = {x : ⟨a,x⟩ = 1 mod 2}` — the "odd" affine
hyperplane of the linear functional `a`, computed in code as
`(a & x).bit_count() & 1`. Any three distinct points `x,y,z` force `a` into an
affine subspace of dimension ≤ 1 relative to their pairwise differences, so at
most 2 of the 15 hyperplane-characters can contain all three — codegree ≤ 2 by
linear algebra, no triple ever enumerated. This is the single most convergently-
rediscovered idea in the whole run: I found it independently phrased at least
four different ways (`_group_blocks`, `_incidence`/"binary character",
`_algebraic`, direct `for a in range(1,16)` inline) by programs on different
islands.

Because the character family alone only ever produces ≤ 15 columns (one per
nonzero `a`) and only covers 8 ≤ M ≤ 15 by construction (`bit_count` only
distinguishes 16 points), every implementation wraps it in:
- **M ≤ 6**: hand-built, per-M "omission" designs (e.g. M=6: complements of the
  9 edges of K_{3,3} — "any three vertices are disjoint from at most two such
  edges"). These are literal, non-derived tuples, specific to each M.
- **M = 7**: three hand-derived families on 7 points (a nested omission code; a
  doubled complement of the 7 Fano-plane lines; complements of the edges of
  P₃+2K₂), tried and the best kept.
- **M > 15** (in the scored range this is *only* the single cell M=16): three
  **different, competing** extension mechanisms were independently evolved (see
  §2 below for the side-by-side comparison) — doubling affine hyperplanes over
  a "cap" subset of normals (the champion's choice), doubled Miquelian
  inversive planes, and disjoint-block partitioning into groups of ≤16 rows.
- A generic completion routine (`_fill`/`_complete`) that spends any leftover
  triple-capacity on lex-first 4-blocks, then repeated triples, then degree-2
  pads — this operates identically across every M and is *not* per-cell
  hardcoding, it's a general canonical closure of a partial 3-(M,·,2) packing.

**Representative programs and exact_count** (all re-verified by me against the
live evaluator): `cd6af8a9` **110** (declared champion), `e8d14da4` **110**
(independent island, disjoint-partition M=16 extension — produces byte-for-byte
*identical scoring output* to the champion, confirmed cell-by-cell), `3c34798c`
**109** (inversive-plane M=16 extension), `fef84e2f` **108** (adds a
degree-ordered "saturation sweep" — see §2's key finding), `731c708c` **105**
(independently-evolved on island 2, generation 7, using AG(2,q) affine lines for
M=16 instead), `8dfcfd85` **107**, dozens more at exactly **106** (a large
tied cluster from before the phase-3 push — see §3).

**Cells solved:** M=3,4 fully (21/21, 20/20); M=13-16 fully (4/4, 3/3, 2/2,
1/1); M=5,6,7 mostly (via the hardcoded designs); **M=8 is a total blackout**
(0/16 for every variant of this family I tested); M=9-12 partial and
increasingly incomplete as N grows relative to M (see full breakdown in §2).

**Why it plateaued.** Quantitatively (my own ablation, §2): about a third of
the champion's exactness (36 of 110 cells) depends on the three hand-specified
M=5,6,7 designs, not a general rule — and the family's mode of *improvement*
throughout the run was "add one more special case," not "generalize the
formula." `config_phase_3.yaml`'s own system-message (written by whoever
prepared that phase, describing the phase-2-final 106-exact incumbent, i.e.
`63b17796`/its clones) states this almost exactly: *"9 if/elif branches on M
and N... 4 hardcoded literal answer tables... Its last 29 iterations of
evolution gained +0.002, all of it from adding one more lookup table."* I
independently verified the population-level version of this: **at checkpoint
100, 36 of 67 programs (54%, exactly matching the config's own claim) used this
family**; by checkpoint 150, **69 of 72 (96%)** did. A deliberate anti-monoculture
intervention for the last 50 iterations (exploitation_ratio cut 0.35→0.2,
migration interval fixed from a bug that meant it never fired in the earlier
phase — see §3) only pushed the *same* family from 106 to 110 exact; it did not
displace it with anything structurally different, despite the phase-3 prompt
explicitly asking for that.

### Family 3: Projective norm graphs (Kollár–Rónyai–Szabó) + doubled inversive/Möbius planes

**Mechanism.** Vertices `(x,α) ∈ F_{q²} × F_q*`; rows and columns adjacent when
`Norm(x+y) = αβ`. Any three vertices on one side have at most two common
neighbours (a rank argument on the resulting system of norm equations), so any
induced rectangular restriction is K_{3,3}-free. Frequently combined with
**doubled Miquelian inversive planes** — a genuine 3-(q²+1, q+1, 1) design
(circles of a Möbius/inversive plane), used with multiplicity 2 so triple
codegree lands exactly on the cap of 2 — and with hand-built small-M packings
for M ≤ 6-8. Both ingredients share the same "triple multiplicity ≤ 2" framing
and are explicitly unified in the best specimen I read (`b126ca4f`'s docstring:
*"Three interchangeable realizations, all derived from a prime power q"*).

**Representative programs:** `b126ca4f` **78/161** (0.6059 — the best specimen;
combines small-M packing + NG(q,3) + doubled inversive planes), `8ed6b27a`
**79/161** (0.6107), `d84f2b1e` **76/161** (0.5978, norm-graph only, no
inversive-plane ingredient), `12b9a176` **60/161** (0.5468), `52651847`/`31eda0ba`
**78/161** (0.5990/0.3290 — the latter is anomalously low for its exact_count,
likely a slower/less-complete fallback path), `e034b8ff` **23/161** (a hybrid
explicitly combining this family with Family 4's generic packing), `d2476bee`
**4/161** (the earliest, checkpoint-10 attempt — already norm-graph, see Family
1's narrative), and 5-6 of the seed's own direct children (Family 1's §
above — 6 of 7 first-generation mutations were of this family).

**Cells solved:** capped in the 0.50-0.61 combined_score band across every
specimen I checked; never approaches Family 2's density.

**Why it plateaued — confirmed exactly by config_phase_3.yaml's own audit**:
*"Kollar-Ronyai-Szabo norm graphs over F_{q^2}: they appear often but have
never been developed past 0.61."* Every combined_score I measured for this
family (0.5978-0.6107) sits inside that band. My own structural read (an
inference, not something I found proven in the code): NG(q,3) has `q²(q-1)`
vertices — a coarse sequence (q=2:4, q=3:18, q=4:48, q=5:100) that lines up
with the scored M-range (3-16) far more sparsely than F_2^d's dyadic sizes
(2,4,8,16) do. Most scored cells therefore fall in the *middle* of a
restriction of the next-larger norm graph rather than at a natural extremal
boundary, so the restricted structure is generically short of the true optimum
by more than the F_2^4 family's restrictions are.

### Family 4: Counting-bound water-filling + greedy level-packing (non-algebraic)

**Mechanism.** Uses *only* the convexity counting-bound certificate
`Σⱼ C(kⱼ,3) ≤ (t-1)·C(M,3)` — no finite-field algebra anywhere. Solves
`N·C(k,3) ≤ budget` by water-filling for the largest uniform column size `k`
(plus a residual count of size-`(k+1)` columns, since `C(·,3)` is convex, the
edge-maximal profile under the bound is as level as the budget allows), then
greedily assigns rows to each column by least-current-degree (breaking ties by
pair-load/degree/cyclic shift), checking the per-triple residual capacity at
each step so validity is certified by construction, never re-verified.

**Representative programs:** `416d32f6` **76/161** (0.5697 — descended
directly from one of the seed's own failed first-generation children,
`430e64fa`; a second, independent occurrence of the same idea is `79ce0797`
**54/161** (0.5322), evolved much later at checkpoint 110/iteration 107, during
the phase-3 exploration push).

**Cells solved:** moderate density everywhere, exactness only sporadic.

**Why it plateaued.** By construction, water-filling hits the *density* target
of the counting bound almost exactly (that's what the LP relaxation gives you)
but a generic degree-ordered greedy assignment does not reconstruct the actual
extremal *combinatorial design* at a given M (which is typically a specific
resolvable or symmetric design), so it lands a handful of edges short of exact
at most sizes — enough for the 0.5-weighted density term, never enough for the
0.5-weighted exactness term that dominates Family 2's higher scores. No variant
of this family that I found added a subsequent exactness-repair step.

### Family 5: Brown's graph (F_q^3 sphere incidence) — a one-off

**Mechanism.** Rows and columns are points of `F_3^3`; row `x` and column
`(center, copy)` are adjacent when the squared distance `‖x - center‖² = 1`
(a "unit sphere" about center). Two distinct spheres meet in ≤ 2 points, so any
three rows lie on at most one sphere; doubling every sphere brings codegree
exactly to the cap of 2.

**Representative:** `c66f4898` (generation 2, iteration 23, island 3),
**41/161** exact, combined_score 0.3951.

**Cells solved:** modest, comparable to Family 3's weaker specimens.

**Why it plateaued (one occurrence only).** `config_phase_3.yaml`'s own
system-message states this construction *"has appeared ONCE in this pipeline's
entire history"* — I grepped all 138 unique program bodies for `sphere` and
confirmed exactly one match, this program. It never got a second generation of
development; by the same checkpoint window (20-30) the F_2^4 hyperplane family
had already reached 76-87 exact, so this had no opportunity to be selected
again before being crowded out.

---

## 2. The champion, in full

`best_program.py` = `cd6af8a9-1bd2-416c-a4fe-6f220415411e`, generation 6,
iteration_found 134, island 3, parent `63b17796` (a 106-exact ancestor of the
same Family 2 lineage). Recorded metrics: exact_count 110/161, combined_score
0.7758, validity 1.0, peak_instance_ms 0.716 (2900x inside the 25ms budget).
I re-ran it in-process against the live evaluator and reproduced 110/0.7758
exactly.

### 2.1 Exact construction rule

```python
def construct_graph(M, N):
    if M > N:                          # self-duality: shorter side as rows
        return construct_graph(N, M).T
    if M <= 6:  return _best(M, N, [_boundary_blocks(M, N)])
    if M == 7:  return _best(7, N, _seven_families())
    if M <= 15: <F_2^4 hyperplane characters, rows = first M of the 15
                 nonzero vectors of F_2^4, column a = {x : <a,x>=1}>
    else:       <affine hyperplanes of F_2^4 whose normals form the
                 cap {a : bit3(a)=1}, both sides of each hyperplane used>
```
followed in every branch by `_best`/`_fill`, a canonical completion that spends
any leftover triple-capacity on lex-first 4-blocks, then repeated triples, then
degree-2 pads, and keeps whichever of {no 4-blocks, 4-blocks-first} yields more
edges (a closed two-way trade-off, not a search).

Because the actual scored table only ever has M in 3..16 with N ≥ M, the
"M > N ⇒ transpose" branch never fires on the official 161 cells (it exists for
robustness/self-duality only), and the "M > 15" branch is exercised by **exactly
one** scored cell: (16,16).

### 2.2 Forensic hardcoding audit (this is the part that matters for generalization)

I built an ablated copy of the champion's own source — identical in every
respect except `_boundary_blocks` (M≤6) and `_seven_families` (M=7) replaced by
functions returning empty lists, forcing those M to fall through to the
*generic* completion routine alone — and re-ran it through the live evaluator.
Per-row breakdown (both columns independently reproduced by me):

| M  | # cells scored | exact, baseline | exact, tables removed |
|----|---------------:|----------------:|-----------------------:|
| 3  | 21 | 21 | 21 |
| 4  | 20 | 20 | 20 |
| 5  | 19 | 19 | **8** |
| 6  | 18 | 18 | **1** |
| 7  | 17 | 8  | **0** |
| 8  | 16 | 0  | 0 |
| 9  | 14 | 2  | 2 |
| 10 | 11 | 3  | 3 |
| 11 | 9  | 4  | 4 |
| 12 | 6  | 5  | 5 |
| 13 | 4  | 4  | 4 |
| 14 | 3  | 3  | 3 |
| 15 | 2  | 2  | 2 |
| 16 | 1  | 1  | 1 |
| **total** | **161** | **110** | **74** |

Combined_score drops from 0.7758 to 0.6891 under the same ablation.

**Reading this table honestly:**
- The hardcoded M=3 and M=4 tables are **redundant** — removing them changes
  nothing, the generic completion alone already reaches the exact optimum
  there. (These are trivial cases: M=3 has exactly one row-triple total; M=4's
  table is a single optional quadruple.)
- The hardcoded M=5, 6, 7 tables are **fully load-bearing**: 36 of the
  champion's 110 exact cells (33%) depend on them entirely, and none of that
  33% is "solved by rule" in any sense that would transfer to an M the LLM
  didn't specifically special-case. M=6 is the most extreme case: 18/18 exact
  with the table, 1/18 without.
- **M=8 is a complete blackout regardless** (0/16 both ways) — the genuine
  F_2^4 rank-argument, restricted to only 8 of its 15 available points,
  apparently never lands on the extremal design at this specific size, in any
  variant of this family I tested (see §2.4).
- The "genuinely derived" part of the champion — the F_2^4 rank argument for
  M=9-16, plus the trivial M=3,4 cases — accounts for 74/161 (46%) of the total
  score, and its own hit-rate is *very* uneven: 100% at M=13,14,15,16, but only
  14% (2/14) at M=9, 27% (3/11) at M=10, 44% (4/9) at M=11. This is a genuine
  structural fact, not a hardcoding artifact: **restricting a 15-point
  structure down to fewer points preserves extremality far better near the
  boundary (M→15) than in the middle (M≈9-11).**
- `config_phase_3.yaml`'s own ablation of the *parent* (`63b17796`, 106 exact)
  claims *"remove the hardcoded tables and it falls to 0.6934/72 exact"* — a
  slightly different number from my 74/0.6891 on the *champion*, as expected
  since the champion added one more M=16 branch the parent didn't have; the two
  numbers are consistent in kind and confirm each other.

### 2.3 Self-duality

The champion is explicitly self-dual (`if M > N: return construct_graph(N,
M).T`), which fixes a defect the phase-3 prompt called out in its ancestor:
*"Asked for (M,N) it is exact on 68 of 91 non-square cells; asked for the
transposed (N,M)... it is exact on 23. It has memorised an orientation, not
learned a rule."* I did not find any surviving Family-2 program from checkpoint
110 onward that lacks this swap.

### 2.4 Failure cells

Re-running the champion, its 51 non-exact cells are (all M in 7-12):
```
M=7:  15,16,17,18,19,20,21,22,23                         (9 cells — ALL of M=7 beyond N=14)
M=8:  8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23      (16 cells — ALL of M=8)
M=9:  9,10,11,12,13,16,17,18,19,20,21,22                 (12 cells)
M=10: 10,11,15,16,17,18,19,20                            (8 cells)
M=11: 11,16,17,18,21                                     (5 cells)
M=12: 22                                                 (1 cell)
```
49 of these 51 were **never** solved exactly by *any* program in the entire
142-evaluation log (see §4 and `cell_reachability.csv`).

**Key finding — the champion is not even the union of everything this run
ever knew how to solve.** Two cells, **(8,8)** and **(8,9)**, were each solved
exactly once in the whole run's history — by `fef84e2f` (108/161
overall), *not* by the champion. I confirmed this by timestamp: the
`instance_log.jsonl` record showing `8x8` and `8x9` both `is_exact=true` at
`combined_score=0.770329` has a timestamp within 1 millisecond of
`fef84e2f`'s own saved timestamp (metrics: exact_count 108.0, combined_score
0.7703285…) — an unambiguous match. `fef84e2f` is otherwise the same Family-2
lineage as the champion (same parent `63b17796`) plus two rounds of a
degree-ordered "saturation sweep" in its completion routine; that sweep buys
(8,8) and (8,9) but costs four M=7 cells elsewhere, netting 108 instead of 110 —
so it never became champion, and its one genuine advance over the eventual
champion was simply lost. **If a future generalization effort wants every cell
this run ever solved, the sweep mechanism in `fef84e2f` (extracted as
`mining/extracted/03_fef84e2f_108.py`) is worth revisiting specifically for
this.**

---

## 3. Evolution narrative — what replaced what, when

- **Generation 0 (checkpoint 10, iteration 0):** seed = `initial_program.py` =
  cyclic difference family on Z_M, 4/161 exact.
- **Generation 1 (checkpoint 10, iterations 1-8):** all 7 sampled direct
  children abandon the seed's principle. 6 of 7 are norm-graph attempts
  (Family 3), 1 is an inversive-plane attempt (also Family 3). Best reaches
  21/161 exact. The cyclic-difference-family idea is never revisited by
  anyone after this point.
- **Checkpoints 10→20 (iterations ~9-20):** the F_2^4 hyperplane-character idea
  (Family 2) enters the population and immediately jumps to 76/161 exact,
  decisively beating anything Family 3 achieved — checkpoint 10's max was
  24/161; checkpoint 20's max is already 76/161.
- **Checkpoints 20→100** (`config_phase_1.yaml`/`config_phase_2.yaml` era — these
  two files are byte-identical except for one LLM-call timeout value, 240s vs
  400s, i.e. they are the *same* prompt regime, tuned mid-run, not two
  different ones): Family 2 climbs steadily — 87 (cp.30) → 88 (cp.50) → 102
  (cp.60) → 105 (cp.70) → 106 (cp.90, and it stays at 106 through cp.100).
  Family 3 (norm graphs/inversive planes) develops in parallel but never
  passes ~0.61 combined_score. By checkpoint 100 the population of 67 programs
  splits **36 Family 2 / 26 Family 3 / 1 Brown's graph / 1 seed-relic / 3
  early one-offs** (the 36/67 figure independently reproduces
  `config_phase_3.yaml`'s own stated measurement of "36 of 67 programs used the
  same F_2^4 idea").
- **The gap:** checkpoint 100 is timestamped Jul 26 23:23; checkpoint 110 is
  timestamped Jul 27 21:16 — a ~22-hour pause. `config_phase_3.yaml` was
  authored to describe exactly the checkpoint-100 state (its system-message's
  *"current incumbent, scoring at 0.7659 with 106 of 161"* matches
  checkpoint 100's best program to 4 decimal places) and ships with two
  measured, documented fixes to *why* the population had stalled into a
  monoculture rather than exploring:
  - `exploitation_ratio` cut from 0.35 → 0.2 ("Phase 2 ended as a monoculture:
    36 of 67 programs used the same F_2^4 idea and all 12 top programs were
    variants of one lineage").
  - `migration_interval` cut from 50 → 5, because migration is measured in
    *generations* not iterations, and at the old setting *"migration NEVER RAN
    in phase 2... All four islands converged on F_2^4 independently with no
    cross-pollination."* (`checkpoint_150/metadata.json` confirms migration did
    fire this time: `last_migration_generation: 30`.)
  - The system-message itself was rewritten with a specific, numbers-backed
    ablation critique of the checkpoint-100 incumbent (quoted throughout §1-2
    above) rather than generic advice.
- **Checkpoints 110→150** (`config_phase_3.yaml` era, 50 more iterations on the
  *same* database — `checkpoint_150/metadata.json`'s `island_generations:
  [29,25,34,28]` and `last_iteration: 150` confirm continuity, not a restart):
  despite the explicit push for "a genuinely new object," the population
  converges *harder* on Family 2, not away from it — **69 of the 72 final
  programs (96%) are Family 2** at checkpoint 150 (2 Family 3 survivors, 1
  Family 4 survivor, 0 Family-1/5 survivors). Of the four islands, two
  (islands 1 and 2, per `island_best_programs`) never broke past the
  checkpoint-100-era 106-exact plateau; the other two (islands 0 and 3)
  independently discovered *different* fixes for the same M=16 gap — island 0's
  `e8d14da4` via disjoint-block partitioning, island 3's `cd6af8a9` via
  affine-hyperplane-cap doubling — that happen to score identically (110/161,
  verified cell-by-cell). `3c34798c` (109/161, inversive-plane extension) and
  `fef84e2f` (108/161, saturation sweep) are near-miss variants from the same
  final generation on island 3. **The run ends with the same family it
  started developing at iteration ~10, incrementally extended, not replaced.**

---

## 4. Files produced

- `mining/mining_report.md` — this file.
- `mining/extracted/01_cd6af8a9_110.py` … `08_f2103a45_4.py` — verbatim code
  (provenance/metrics in a header comment above each `# EVOLVE-BLOCK-START`) of
  8 mathematically distinct programs: the champion; its two closest
  Family-2 competitors with different M=16 mechanisms (inversive-plane,
  saturation-sweep); a third, independently-evolved Family-2 specimen (AG(2,q)
  affine-line M=16 extension); the best Family-3 (norm graph + inversive
  plane) specimen; the best Family-4 (water-filling packing) specimen; the
  sole Family-5 (Brown's graph) specimen; and the Family-1 seed itself.
- `mining/cell_reachability.csv` — per-cell (all 161, holdout included)
  exact_value, best_valid_edges_ever, ever_exact, num_evals (=142 for every
  cell — confirmed uniform), plus num_valid/failed/over_budget_evals,
  gap_to_exact, and an is_holdout flag, parsed from all 142 valid lines of
  `instance_log.jsonl` (0 malformed lines encountered).
- `mining/ideas_for_generalization.md` — generalization analysis.

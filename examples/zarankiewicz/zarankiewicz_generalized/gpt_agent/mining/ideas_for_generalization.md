# Generalization Analysis: from z(M,N;3,3) to construct(m,n,s,t)

This builds directly on `mining_report.md`'s taxonomy. Read that first for the
mechanism descriptions; this file is about what transfers to a general-(s,t)
constructor and what doesn't.

## (a) Mechanisms that generalize to arbitrary (s,t), and how

**Family 4 (counting-bound water-filling) generalizes with zero rework.** Its
only mathematical content is the Zarankiewicz counting bound
`Σⱼ C(kⱼ,s) ≤ (t-1)·C(M,s)`, water-filled for target column sizes, plus a
greedy capacity-respecting row assignment. Both `s` and `t` are already free
parameters in the two implementations I read (`416d32f6`, `79ce0797` — see
`S_PARAM`/`T_PARAM` and `S_ROWS`/`T_COLS` in the extracted files); nothing in
either program assumes s=3 or t=3. **This is the one mechanism in the entire
run that is honestly (s,t)-general already** — its problem is weakness
(exactness), not scope, and it is the natural skeleton/baseline for a general
`construct(m,n,s,t)`: it will always produce a *valid* answer with reasonable
density for any (s,t), even where nothing more clever is available.

**Family 1 (bounded-multiplicity difference families) generalizes soundly but
becomes increasingly loose as s grows.** The key fact the champion's ancestor
uses — *"a triple's common columns sit inside any of its pairs', so bounding
PAIRS bounds TRIPLES too"* — is not special to triples. For **any** s ≥ 2, the
shared-column set of an s-subset of rows is the intersection of its pairwise
shared-column sets, hence a subset of any one pair's. So a difference family
with pairwise multiplicity ≤ (t-1) is K_{s,t}-free for **every** s ≥ 2
simultaneously — the construction doesn't need to know s at all, only t. That
makes it trivially generalizable to any (s,t) with s ≥ 2... but only *soundly*,
not *tightly*: for s=2 (ordinary bipartite Zarankiewicz, no third row involved)
the pairwise bound is exactly the right thing to enforce, but for s ≫ 2 it is
needlessly conservative — it forbids a pair of rows from sharing t columns even
though the K_{s,t} condition only forbids an *s-tuple* from doing so, and
larger s gives more room that a pure pairwise construction throws away. **Use
this family for s=2, or as a cheap valid fallback; don't expect it to be
competitive once s is much larger than 2** (which matches what was observed
here even at s=3: it was the *worst*-performing surviving family).

**Family 2 (finite-geometry hyperplane/subspace incidence) generalizes in
principle, not in this run's code.** The underlying mathematical mechanism — a
system of `s` linear conditions over F_q^d has a solution set of dimension
`d-s` (generically), giving `q^{d-s}` simultaneous solutions — is a completely
general tool: pick `d` so that `q^{d-s} ≤ t-1` and you have a K_{s,t}-free
incidence structure by the same argument used here for (s,t)=(3,3), q=2. This
is genuinely the highest-value mechanism to carry forward, **but** everything
that made this run's version score well beyond the trivial cases (the M≤6
tables, the M=7 Fano-derived families, the M=16 "cap" trick) is bespoke to
s=t=3 and F_2^4's specific size (16 points). A general version needs the
*boundary-case logic itself* re-derived from the same dimension-counting
principle at small d/q for each (s,t), not a hand-built combinatorial table
per (s,t) — see recommendation (d.2) below for why this matters more than it
looks like it should.

**Family 3 (Kollár–Rónyai–Szabó norm graphs) is, in the literature, already
the general-(s,t) tool — but this run only ever implemented its t=3 case.**
Every norm-graph specimen found here uses a *quadratic* norm form
(`Norm(x+y)=αβ` over `F_{q²}`), which is specifically the construction that
bounds codegree at 2 (i.e. t=3). The actual Kollár–Rónyai–Szabó / Alon–Rónyai–
Szabó machinery generalizes by using a degree-`(t-1)` (or similarly
parametrized) norm form over a larger extension field to push codegree bounds
to general `t-1`, and is one of the standard tools in the extremal-graph-theory
literature specifically for general K_{s,t}-free constructions. **This is
worth prototyping from the literature construction directly rather than
generalizing this run's programs**, since none of them ever varied `t` — they
only varied the field size `q`, which mostly changes the number of vertices,
not the codegree bound. Note also this family capped at combined_score ≈ 0.61
even at (3,3) here (see mining_report.md §1), so treat it as a
density-respectable, exactness-poor fallback rather than the primary route.

## (b) Mechanisms that are genuinely (3,3)-specific

- **Every hand-built small-M table** in Family 2 (the M=3,4,5,6 omission
  designs; the three M=7 families built from the Fano plane / P₃+2K₂ edges).
  These are literal Steiner-triple-system-adjacent combinatorics for *s=3*
  specifically ("any three vertices are disjoint from at most two edges" is a
  statement about triangles/triples, not general s-tuples). Re-deriving
  analogous small-case designs for a different s (or larger t) is a fresh
  combinatorics problem each time, not a parametrized formula — this is
  exactly the "catalogue, not a rule" failure mode the run's own
  system-message (config_phase_3.yaml) warned about, and my ablation
  (mining_report.md §2.2) shows it is 33% of the champion's own score.
- **Brown's graph** as implemented (F_3^3, unit spheres, "two spheres meet in
  ≤2 points ⇒ codegree ≤2") is tied to t=3 by construction (the "2" is a fact
  about how quadrics intersect in this specific dimension/field, not a free
  parameter). The general idea — points of F_q^n with adjacency given by a
  quadratic (or other low-degree) form — is a known general technique in
  extremal graph theory, but this specific instantiation does not carry a free
  t parameter the way Family 3 or 4 do.
- **The champion's M=16 "cap" trick** (affine hyperplanes whose normals form a
  cap `{a : bit3(a)=1}`) is a construction specific to extending exactly the
  F_2^4/16-point structure by "one more bit," not a general rule for extending
  an F_q^d structure past its natural point count.
- **The self-duality shortcut `if M>N: swap and transpose`** is only correct
  because the scored problem has **s=t=3**. In general, `z(m,n;s,t) =
  z(n,m;t,s)` — transposing the bipartite graph swaps *which side* has the
  forbidden s-set and which has the forbidden t-set, so it swaps **s and t
  along with m and n**. A general `construct(m,n,s,t)` that copies this run's
  "just swap m,n" idiom without also swapping s,t will silently misbehave
  whenever s ≠ t. This is a concrete, easy-to-miss bug worth flagging
  explicitly to whoever builds the general version.

## (c) The hard core: cells no program ever solved exactly

From `cell_reachability.csv` (built from all 142 valid lines of
`instance_log.jsonl`, spanning the seed through the champion — every program
this run ever ran, not just the ones that survived to checkpoint 150): **49 of
the 161 scored cells have `ever_exact = 0`.** Every one of them has `M` (the
smaller dimension) between **7 and 12** — no cell with M ∈ {3,4,5,6} or
M ∈ {13,14,15,16} is in this set; those ranges were fully conquered by *some*
program at some point, even though the final champion itself doesn't solve
every M=7-12 cell that history collectively knows how to solve (see the
(8,8)/(8,9) finding in mining_report.md §2.4 — those two are the exceptions
that make "49 never-solved" ≠ "51 champion misses").

```
M=7:  N = 15,16,17,18,19,20,21,22,23        (9 cells — every N once it passes 14)
M=8:  N = 10,11,12,13,14,15,16,17,18,19,20,21,22,23   (14 cells)
M=9:  N = 9,10,11,12,13,16,17,18,19,20,21,22          (12 cells, with N=14,15 as solved "holes")
M=10: N = 10,11,15,16,17,18,19,20                     (8 cells, N=12,13,14 solved "holes")
M=11: N = 11,16,17,18,21                              (5 cells)
M=12: N = 22                                          (1 cell)
```

**Structural pattern:**

1. **It is not a clean "N too large" cutoff.** M=9 and M=10 both have "holes"
   in the never-solved set (9×14, 9×15, 10×12, 10×13, 10×14 were all solved by
   *something*, historically, even though neighboring cells at the same M
   weren't) — the failure is patchy, not monotone in N. Any theory of *why*
   these specific cells resist should account for that patchiness rather than
   assume a simple aspect-ratio threshold.
2. **Within a row, the gap-to-exact (best-ever-achieved edges vs. the known
   optimum) grows with N.** E.g. row 9's gap sequence as N runs 9→22 is
   1,1,2,1,2,3,4,5,5,6 — elongation makes the target *harder to approach*, not
   just harder to hit exactly.
3. **The two hardest cells by far are (11,21) and (12,22)**, with gaps of 8
   and 12 edges respectively (versus every other never-solved cell's gap of
   1-7). These are precisely `evaluator.py`'s two `_EXTRA_EXACT` entries — cells
   where the published *table* value is only an upper bound and the true
   optimum (116 and 132) had to be sourced from an individual paper rather
   than read off the row formula. That these are also the two cells no
   program came remotely close to suggests they may be genuinely harder
   combinatorial targets, not just under-explored ones.
4. **M=8 is the worst row in the whole table**: 14 of its 16 cells are in the
   never-solved set, and even the champion's own dedicated F_2^4 machinery
   (which uses 8 of its 15 available points for this row) gets **0/16** (see
   mining_report.md §2.2's ablation table) — this is the single worst-covered
   row of any size, worse even than the harder-looking M=11,12.
5. All 49 cells sit in the "middle stretch" relative to the F_2^4 structure
   that dominates this run (which naturally covers 15-16 points cleanly) —
   roughly 40-75% of that structure's capacity. This matches the ablation
   finding in mining_report.md §2.2 that the *genuine* (non-hardcoded) part of
   the champion is 100% accurate at M=13-16 (near the structure's natural
   size) but only 14-44% accurate at M=9-11 (well inside it). **A truncation
   of a larger algebraic structure seems to be systematically worst in the
   middle of the truncation range, not at its edges** — a pattern worth
   testing for on any new base structure a general (s,t) constructor adopts,
   not just F_2^4.

## (d) Recommendations for the construction-engineer agent building general construct(m,n,s,t)

1. **Start from Family 4's counting-bound skeleton, not Family 2's algebra.**
   It is the only mechanism here that is already honestly parametrized in
   (s,t) with no hidden assumptions. Use it as a correctness/density floor
   that works everywhere, then look for where a specific (s,t) admits a
   tighter algebraic structure to layer on top.
2. **When layering in finite-geometry (Family 2's idea), parametrize the
   dimension/field by (s,t) from first principles** (smallest `d`, or field
   size `q`, such that the generic-intersection count of `s` generic
   hyperplanes/subspaces is `≤ t-1`) **rather than hand-deriving small-case
   tables per (s,t) as this run did.** If a boundary case genuinely needs
   separate handling, prefer deriving it as *the same rule at a different
   parameter value* (e.g. a smaller field, or a lower-dimensional analogue of
   the same incidence structure) over an enumerated combinatorial design —
   the latter is exactly the "add one more lookup table" failure mode that
   consumed the second half of this run's compute for +0.002 (see
   mining_report.md §1, Family 2's "why it plateaued", and §3's evolution
   narrative) without ever producing a transferable rule.
3. **Build in a self-ablation check as standard practice.** This run's own
   champion turned out to owe 33% of its exactness to three hand-specified
   designs (mining_report.md §2.2) — something that was *not* visible from its
   exact_count alone, only from actively removing pieces and re-scoring. A
   construction-engineer agent should routinely test "how much does my score
   drop if I strip every branch keyed on a specific M/N rather than a formula"
   before reporting a construction as a "general rule."
4. **Get the self-duality direction right for s ≠ t.** `z(m,n;s,t) =
   z(n,m;t,s)` — swap s and t together with m and n, not m,n alone (see (b)
   above). This run's code only ever needed the s=t=3 special case and got it
   right *for that case*; a general version copying the idiom naively will be
   subtly wrong whenever s ≠ t.
5. **Expect a "middle stretch" weak zone relative to whatever base structure
   is chosen**, and specifically target it rather than assuming asymptotic
   quality (good performance near the structure's natural size) implies
   uniform quality (see (c).5). For this run's F_2^4 structure that zone was
   M≈9-11 (roughly 60-70% of the structure's 15-16-point capacity); for a
   different (s,t) and a different base structure, expect an analogous
   fractional zone and budget extra design effort there specifically, rather
   than trusting that "it works at the boundary, so it must work everywhere."
6. **Enforce a hard per-instance CPU budget from the start, not as an
   afterthought.** Every non-time-budget instruction in this run's prompts
   ("do not search," "do not hardcode") was, by the run's own admission,
   insufficient on its own — only the enforced 25ms-per-instance limit
   actually made search unprofitable (see mining_report.md's Era-0 section).
   If the harness building `construct(m,n,s,t)` can enforce something
   equivalent, do so from iteration 1 rather than letting a greedy/hill-climb
   population establish itself first.
7. **Treat Family 3 (KRS norm graphs) as worth a literature-accurate
   reimplementation, not a generalization of what's in this run.** Every
   specimen mined here hardwires a quadratic norm form (i.e. is really a
   t=3-only tool wearing a "general q" costume); the actual general-t
   construction in Kollár–Rónyai–Szabó / Alon–Rónyai–Szabó uses a
   higher-degree norm form and is worth building from the original
   construction rather than from these programs.
8. **Don't expect Family 1 (difference families) to be competitive once s>2**,
   per the soundness/tightness argument in (a) — it's a fine cheap fallback
   for validity, not a target for the primary mechanism.

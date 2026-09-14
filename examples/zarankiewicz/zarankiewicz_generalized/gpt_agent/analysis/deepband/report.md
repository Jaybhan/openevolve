# The deep band n < T33(m): the value-Pareto frontier of mixed heavy-block packings

Deep-band theorist, 2026-07-28 (session 3). Labels: **PROVEN** (human proof,
machine-checked where stated), **ILP-PROVEN** (MILP optimality certificate,
legality independently re-verified), **PROVEN-DICT** (pinned by Lemma P +
an already-proven z value), **VERIFIED-ALL-KNOWN**, **CONJECTURE**, **OPEN**.

Setting s = t = 3, B = 2C(m,3). A *heavy config* C: multiset of blocks of
weight >= 4, every row-triple covered <= 2. val = sum(w-3), slots = sum
C(w,3), c = #C. Starting point, proven in theorems.md (Theorem 9):

    z(m,n;3,3) = 3n + max_C [ val(C) - max(0, n - #C - (B - slots(C))) ]   (+)

The frontier: F_m(c) := max{ val(C) : legal heavy C, #C <= c }.

---

## 0. Headlines

1. F_6, F_7, F_8 computed completely, every point ILP-PROVEN; F_9 complete
   with every point ILP-PROVEN or PROVEN-DICT. The identity (+) with these
   frontiers reconstructs **every known z-cell of rows 6-9 over their FULL
   ranges: 35 + 64 + 105 + 155 = 359/359 cells** (m <= n <= B, deep band
   AND the Roman/Culik elbow — verify_deepband.py). All 64 stored optimal
   witnesses are penalty-free (verified): in the witnessed range, optima
   always satisfy z = 3n + val(C*). With §5's five new cells, **rows
   6, 7, 8, 9 are COMPLETE for every n** under the single identity (+);
   the per-cell provenance ledger (published / workspace-ILP / Theorem 8 /
   deepband-cases, plus Theorem-10-tightness and F-provenance) is
   rows69_status.csv (364 cells).

2. **Theorem 10 (PROVEN)**: 6 val(C) = 2c + W(C) - X(C) with
   W = sum w(w-3) <= mR, R = floor(2C(m-1,2)/3), X = sum (w-4)(w-5) >= 0;
   hence val <= floor((2c + mR)/6) and
   z(m,n;3,3) <= 3n + min(J*, floor((2n + mR)/6)) for all n.
   Never violated on all 359 known cells of rows 6-9; TIGHT on 43 of them.
   New consequences: one-line upper-bound proofs for z(6,6), z(6,7),
   z(6,8) (previously Theorem 2's case chains), and a closed form for the
   ENTIRE row 6 (§2); 13 row-9 band cells' UBs (n = 20, 22, 23, 28-32,
   34-38) follow from arithmetic + known constructions, no ILP.

3. **Row 9 completed — five previously-open cells resolved** (§5):
   z(9,24) = [Z24], z(9,25) = [Z25], z(9,26) = [Z26], z(9,27) = [Z27],
   z(9,33) = [Z33] — each closed by the case-pinched slice reductions of
   §5 (every candidate improvement forces W = 162 saturation and a single
   (k4, k5, k6) profile, decided by one feasibility MILP each).
   Corollaries: the S9 slices named in §5; S9(2) in [34, 35] (LB raised
   from 33 by the overnight incumbent; UB 35 is the §4 parity bound).

4. **The frontier's structure** (hypothesis (i) as posed is FALSE): frontier
   configs are two-layer geometries, not near-uniform mixes. Decoded:
   the (8,8) optimum is a **Fano cone**; m=9 mid-band optima are
   **pencil-partition parity cones** (pentads = point + odd-weight-coset
   transversals of a pair-partition — an F_2^4 code); z(11,22) is the
   **2-(11,5,2) biplane + its complement design**; hexad layers are
   complements of near-linear spaces, pinned at 4 (m=9), 5 (m=10, n=21),
   11 (m=11, n=22).

5. **Supply drops** (hypothesis (iii)): two proven congruences — pair-parity
   ell_xy == p_xy (mod 2) and point congruence mu_x == 2C(m-1,2) (mod 3) —
   give a forced-leave bound that is TIGHT for S6(1), S6(2), S7(3), S7(4),
   S9(3), S10(1), and (combined with the new doubled-block lemmas) yields a
   complete human proof of **S8(2) = 21**. "First drop large, then small"
   = the first pentad's odd-K5 leave is maximal; later pentads amortize by
   4-intersection or doubling. The residual unit at S8(1) = 23 (parity
   gives <= 24) remains OPEN — same phenomenon at S7(1), S9(1), absent at
   S6(1), S10(1).

---

## 1. Data and methods (task 1)

- extract_witnesses.py -> witness_configs.csv: all 64 witnesses legal,
  edges match truth (60/60 cells with truth), identity (+) holds exactly,
  penalty 0 in every one.
- frontier_milp.py: F_m(c) by MILP over all blocks of weight 4..m
  (multiplicity <= 2), per-triple <= 2, cols <= c, with PROVEN-VALID cuts
  only: pair cuts sum_{B>=xy}(w-2)x_B <= 2(m-2); point floors
  sum_{B>x}(w-3)x_B <= R (Lemma C); weighted budget sum w(w-3)x_B <= mR;
  val <= J (Lemma C), improved to J-2 for m == 3 (mod 4), m !== 0 (mod 3)
  (Theorem F); monotone-degree symmetry breaking; monotone val >= F(c-1).
  Secondary lexicographic solve: min slots at val = F(c). Every incumbent
  re-verified legal by an independent counter.
- probe_cs.py: joint (cols <= c, slots <= s) probes — the third Pareto
  coordinate (used for F_9(13/14) pinning and the m=8 c=17 slice).
- Deliverables: frontier_m{6,7,8,9}.csv (c, F, status, slots_min, W, X,
  profile, F_w5, F_w6, UBcols, FLP, delta), frontier_points_m{m}.json
  (block lists), frontier_m9_final.csv (provenance-labeled),
  witness_configs.csv, s9_slices.jsonl, lb_witness_9x33.json.

### The frontiers

    m=6 (T=9):  c:  1  2  3  4  5  6  7  8  9
                F:  3  6  6  6  7  8  8  8  9
    m=7 (T=15): c:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                F:  4  8  8  9 10 11 12 13 13 14 14 14 14 14 15
    m=8 (T=28): c:  1  2  3  4  5  6  7  8  9 10 11 12 13 14
                F:  5 10 10 12 13 14 16 18 18 20 20 21 21 22
                c: 15 16 17 18 19 20 21 22 23 24 25 26 27 28
                F: 22 22 23 23 24 24 24 24 25 25 25 26 27 28
    m=9 (T=40): c:  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
                F:  6 12 12 14 15 18 19 21 22 24 26 28 28 28 28
                c: 16 17 18 19 20 21 22 23 24 25 26 27 28..32 33 34..38 39 40
                F: 29 30 31 32 33 33 34 34 [F24] [F25] [F26] [F27] 36-37 [F33] 38-39 39 40

  (m=9 provenance: c <= 13 and c in {24..27, 33} ILP-PROVEN; the rest
   PROVEN-DICT via Lemma P from Tan's SAT-certified cells and the
   workspace's gap-band ILP cells; see frontier_m9_final.csv.)

Weight-restricted frontiers (F_w5 = weights <= 5, F_w6 <= 6) give the
heavy-necessity map — blocks of weight >= 6 are NECESSARY (F_w5 < F) at:

    m=6: c <= 3.  m=7: c <= 8 (heptads/full at c <= 3).
    m=8: c <= 8 (octads at c <= 3, heptad at c in {7,8}, hexads 4..6)
         and AGAIN at the isolated parity patches c in {12, 17}.
    m=9: heavy strictly needed for c <= 13 (F_w5 = 2c there, always below
         F); NOT needed at c = 14, 15, 16 (F_w5 = F, PROVEN); at
         c = 17..20 the found optima carry a k6 = 2 hexad layer
         ({4^a 5^9 6^2}) but weight-<=5 alternatives with b = 13 pentads
         would tie iff S9(13) >= c - 13 — undecided (S9(13) in [3, 7]);
         c >= 21: quad/pentad (PROVEN).

## 2. Theorem 10 (PROVEN) and its row-level corollaries

**Identity.** 6(w-3) = 2 + w(w-3) - (w-4)(w-5) for every w; summed:
6 val = 2c + W - X, and with g(w) := (w-4)(w-5) + C(w,3) - w(w-3) >= 0
(equality iff w <= 5): 6 val = 2c + slots - sum_B g(w_B).

**Per-point ceiling.** Blocks through x restrict to a 2-fold pair packing
of the other m-1 points: sum_{B>x} C(w-1,2) <= 2C(m-1,2); with Lemma C's
3(w-3) <= C(w-1,2) (tight iff w in {4,5}) and integrality: y_x <= R.
Summing (sum_x y_x = W):  W <= mR.

**Theorem 10.** val(C) <= floor((2 #C + mR - X)/6) <= floor((2 #C + mR)/6),
so z(m,n;3,3) <= 3n + min(J*, floor((2n + mR)/6)) for every n. QED

Machine verification (theorem10_check.py, ALL PASS): identities and
tightness for w = 4..60; per-point and global checks on all 64 witnesses;
the z-form on all 359 known cells (never violated; tight at m=6:
n = 6..13, m=7: n = 15..25, m=8: n = 27..28, m=9: n = 20, 22, 23, 28-32,
34-38, 40-48).

**Corollary (row 6 in one line, all n).** For 6 <= n <= 40:

    z(6,n;3,3) = 3n + min( 9, floor((n+18)/3), floor((40-n)/3) )

(PASS on every known cell; UB = Theorem 10 + Lemma C + Roman, LB = the
existing witnesses). The previously case-analytic cells (6,6), (6,7),
(6,8) become one-line arithmetic.

**Corollary (row 9 nearly closed form).** For 20 <= n <= 168:

    z(9,n;3,3) = 3n + min( 40, floor((n+81)/3), floor((168-n)/3) ) - e(n),
    e(n) = 1 for n in {21, 24, 25, 26, 27, 33, 39},  e = 0 otherwise
    [the exception set is PROVEN complete given §5's five cells].

The exceptions cluster where floor((n+81)/3) increments (n == 0 mod 3)
but the frontier is parity-delayed. For m == 1, 2 (mod 3) (rows 7, 8) no
such closed form exists — mR = B has no floor slack and the S-table
corrections appear at almost every band c (delta-columns of the CSVs).

**Lemma P (dictionary pinning, PROVEN).** If z(m,c) = 3c + Q(c) is proven,
then F_m(c) = max(Q(c), F_m(c-1)). [A #C = c config is penalty-free at
n = c since slots <= B; smaller configs are covered by F(c-1).] This runs
the z <-> frontier dictionary backwards; combined with Theorem 10's UB to
bridge staleness across open cells, it completes F_9 without circularity.

## 3. Hypothesis (ii): val_max(cols, slots) — resolved with a precise split

On weight-<=5 configs the identity IS the closed form: val = (2c + slots)/6
exactly; the frontier question reduces to the feasible (c, slots) region,
i.e. max slots at <= c columns. The three-level chain

    F_m(c) <= FLP_m(c) <= UBcols_m(c) = min((m-3)c, floor((2c+mR)/6), J*)

(FLP = integer profile shell: columns, budget slots <= B, weighted budget
W <= mR, val <= J*) is computed in every CSV with the deficit
delta = FLP - F.  Measured: delta = 0 exactly at c in {1, 2, 27, 28, 29}
for m=8; everywhere except c = 4 (delta 1) for m=6; at c in {1, 2, 12}
within m=9's MILP-computed range; at c in {1, 2, 15, 16} for m=7;
delta <= 2 in every pentad-regime cell, up to 4 in hexad zones. So a pure
(cols, slots) closed form holds exactly on the m == 0 (mod 3) rows up to
a finite exception set (rows 6 and 9 above), and PROVABLY FAILS to be
tight for rows 7, 8, where delta > 0 at almost every band c — there the
frontier is the S-table itself (realizability, invisible to any
arithmetic in (cols, slots)). This sharpens hypothesis (ii) into a
dichotomy theorem-shape rather than a single formula.

## 4. Hypothesis (iii): the drop mechanism (§0.5 expanded)

With leave L = B - slots, pair-leave ell_xy, point-leave mu_x:

    sum_x mu_x = 3L,  sum_y ell_xy = 2 mu_x,
    ell_xy == p_xy (mod 2)   [p_xy = # odd-weight blocks through xy]
    mu_x == 2C(m-1,2) (mod 3)   [quad/pentad configs]

**Forced-leave bound (PROVEN).** L >= max(ceil(|O|/3),
ceil(sum_x mu_min(deg_O(x))/3)) where O = odd-pair graph of the pentad
multiset and mu_min(d) = least t >= ceil(d/2) with t == 2C(m-1,2) (mod 3).
Hence S_m(b) <= max_M floor((B - 10b - L_par(M))/4). Enumerated over all
legal pentad systems (parity_leave.py):

    m=6: b=1: 6 TIGHT.  b=2: 4 TIGHT.
    m=7: b=1: 13 (truth 12).  b=2: 12 (10).  b=3: 8 TIGHT.  b=4: 6 TIGHT.
    m=8: b=1: 24 (23).  b=2: 23 (21).  b=3: 19 (17).  b=4: 18 (15).
    m=9: b=1: 38 (37).  b=2: 35 (parity PROVES the bracket's UB; the
         overnight incumbent raises the LB: S9(2) in [34, 35], with the
         34-quad witness containing a doubled quad).  b=3: 33 TIGHT.
    m=10: b=1: 56 TIGHT (= ILP value in bounds/supply_table.csv).

**Cross-reference (Theorem 11, theorems.md).** The two congruences above
are exactly the K4^(3)-divisibility conditions of Theorem 11 (session 4)
applied to the leave 2K_m^(3) - C: slots == 0 (mod 4), point degrees == 0
(mod 3), pair degrees == 0 (mod 2). Theorem 11 computes the T-spectrum
(the b = 0 corner: admissible / class J-2 with T settled at m in
{7, 11, 19, 23} / 3|m Johnson-tight); this section is the SAME divisibility
machinery pushed into the interior b >= 1 of the supply table — the
forced-leave bound is a minimal-divisibility-restoration computation for
mixed quad/pentad packings, i.e. the deep-band complement of Theorem 11's
maximum-packing story.

**Doubled-block lemmas (PROVEN).** (a) If a pentad is doubled, every other
block meets it in <= 2 points; for m=8 this gives (with point floors)
4a <= 2a + 3R, i.e. a <= 21. (b) Doubled pentads are pairwise
<= 2-intersecting, so their number d satisfies min-sum_x C(r_x,2)
(sum r = 5d on m points) <= d(d-1): **D(8) = 2, D(9) = 3** (search-
confirmed exactly).

**S8(2) = 21 (PROVEN humanly).** Distinct-pentad branch: parity bound = 21
for every intersection size k = 2, 3, 4. Doubled branch: <= 21 by lemma
(a). LB: the ILP frontier config {4^21 5^2} (two pentads meeting in 2).

The maximizing structures explain the drop pattern: the first pentad's O
is K5 (maximal forced leave — big drop); the second either intersects it
in 4 points (cancels 12 odd pairs — small drop) or doubles (parity-free,
avoidance-taxed). The frontier skips the parity-inefficient pentad counts
entirely: for m=8 the val-max profiles use b in {0,1,2,5,7,8,9,10} only —
b = 3, 4, 6 are dominated (V(b) = S(b)+2b dips at exactly those b).

**OPEN:** the single missing unit at S7(1), S8(1), S9(1) (parity bound
minus one; absent at m = 6, 10). The forced structure at m=8, a=24:
q_x = (11,11,11,11,10) on the pentad, all three outside links = doubled
Fanos; the contradiction is not reachable by the counting above —
conjecturally a doubled-Fano rigidity statement.

## 5. Row 9: the open cells [exact statements]

Before: z(9,n) open at n = 24, 25, 26, 27, 33 (gap-band MILP UNRESOLVED at
600-900 s; k5split incumbents with pending slices; generic frontier MILPs
at 3000 s produced incumbents F_9(24) >= 34, F_9(33) >= 37 but no
optimality certificates — the direct problems are hard).

**The case-pinching reduction (PROVEN; the step that makes the cells
tractable).** Any improving config is squeezed by the identity
6 val = 2c + W - X together with W <= mR = 162, X >= 0: writing the
required val and the column bound c, the equation pins c, X, W to single
values, and X's block-decomposition ((w-4)(w-5) = 2 per hexad, 6 per
heptad, ...) pins the entire weight profile:

    val-35, cols<=24: c=24, X=0, W=162 saturated -> (k4,k5) = (13,11) [Q-A]
    val-35, cols<=25: adds c=25: X=0 -> (15,10) [Q-B];
                                 X=2 -> one hexad + (16,8) [Q-C]
    val-35, cols<=26: adds c=26: X=0 -> (17,9) [Q-D];
                      X=2 -> hexad + (18,7) [Q-E]; X=4 -> 2 hexads + (19,5) [Q-F]
    val-36, cols<=27: c=27, X=0, W=162 -> (18,9)                       [Q-G]
    val-38, cols<=33: c=33, X=0, W=162 -> (28,5)                       [Q-H]

(W = 162 saturation means EVERY point has y_x = R = 18, mu_x = 2 — the
maximally rigid case; heptads and octads are excluded by X <= 4 in range.)
Each question is one small slice-feasibility MILP over quad/pentad(+fixed
hexad-count) blocks with the proven cut set (s9_slices.py); INFEASIBLE
settles the cell at the lower value, FEASIBLE settles it at the higher
value with Theorem 10 as the matching upper bound — the cells close
either way. Verdicts (s9_slices.jsonl, assemble_cases.py):

    [QVERDICTS]

Hence  F_9(24) = [F24], F_9(25) = [F25], F_9(26) = [F26],
       F_9(27) = [F27], F_9(33) = [F33].

Lower bounds (verified witnesses in the bank / built here):
w_9x24 (106 edges: 14 quads + 10 pentads + 0 fills), w_9x25 (109),
w_9x26 (112), w_9x27 (116: 19 quads + 8 pentads), lb_witness_9x33.json
(136 = w_9x32 + one residual-capacity triple; legality re-verified).

By (+): z(9,n) <= 3n + F_9(n), so

    z(9,24) = [Z24], z(9,25) = [Z25], z(9,26) = [Z26], z(9,27) = [Z27],
    z(9,33) = [Z33].

Design-theoretic dividends of the same verdicts:
**z(9,24) = 106 <=> S9(11) = 12** (Q-A; the <= 12 side is Lemma P from
z(9,23)); Q-B decides S9(10) vs its slot bound; Q-D/Q-G bracket S9(9);
Q-H is exactly the top of the theorems.md bracket S9(5) in [25, 28].

With these five cells, **row 9 is determined for every n >= 9** — the
fourth complete row (after 6, 7, 8), assembled from: Tan n <= 22;
workspace gap-band ILP 23, 28-32, 34-47; this session's five; Theorem 8 /
Roman / Culik beyond.

## 6. Near-diagonal delimitation (task 4 — honest)

Rows 6-9 are FULLY covered by this machinery down to n = m. The
description coarsens in stages as c shrinks: (1) pentad regime
(c >= ~1.5m): quad/pentad bodies, S-table + parity mechanism, Theorem 10
tight-or-nearly; (2) hexad shelf: tops = complements of near-linear
spaces, decoded but no closed form claimed; (3) full-block corner
(c <= 3): F = (m-3)c via doubled full blocks. For m >= 10 the top layer
thickens (5 hexads at (10,21); the biplane pair at (11,22)) and the
extremal tops become 2-designs; block weights at the true diagonal grow
like n^{2/3} (Brown / Kovari-Sos-Turan-Furedi regime), so no finite-
weight frontier law extends to all m — we make NO asymptotic claim. What
the next band out needs: S_m(k5,k6,...) grids for m >= 10 (each column a
design-existence problem) and a replacement for the per-point bound whose
X-slack grows quadratically in the top weight (at the diagonal the budget
B binds, not mR, and Theorem 10 degenerates toward Lemma A).

## 7. Reproducibility

    extract_witnesses.py, frontier_milp.py, probe_cs.py, feas_check.py,
    export_frontier_csv.py, finalize_m9.py, verify_deepband.py,
    theorem10_check.py, parity_leave.py, s9_slices.py (slice feasibility,
    k5:quad_lb[:k6[:tl]] args), assemble_cases.py (case-verdict assembly),
    gen_status_table.py, decode_structure.py
    Data: frontier_m{6,7,8,9}.csv, frontier_m9_final.csv,
    frontier_m9_cases.jsonl (the five cells' certificates),
    rows69_status.csv (the 364-cell completion ledger),
    frontier_points_m{6..9}.json, witness_configs.csv, *.jsonl logs.
    Venv: /private/tmp/claude-501/.../scratchpad/zvenv/bin/python.

Everything labeled PROVEN above is checkable by hand from this file plus
theorems.md; everything ILP-PROVEN has a gap-0 HiGHS certificate and an
independently re-verified witness; no unproven cut ever entered a solver.

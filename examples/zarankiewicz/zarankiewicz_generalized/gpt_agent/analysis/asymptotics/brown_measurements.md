# Brown's graphs, measured: the exact second-order law of the construction

Asymptotic analyst, 2026-07-30. Task 2. Machine companion:
`brown_construction.py` (runtime ~10 s); data: `brown_data.csv`.
Normalization guard: everything here is **bipartite** (z(n,n;3,3), truth
constant 1); the graph constant 1/2 never appears below.

## 1. The construction, exact, both residue classes

V = F_q³ (q odd prime), x ~ y iff Q(x−y) = δ, Q = sum of three squares.
The legality condition — derived here from the radical-plane algebra and
verified computationally — is

    χ(−δ) = −1   (minus delta a quadratic non-residue):
      q ≡ 3 (mod 4):  δ = 1        (Brown's original unit sphere),
      q ≡ 1 (mod 4):  δ = any non-residue (we take the smallest).

Why this and only this: for three distinct sphere centers,
- non-collinear centers: the two radical planes meet in a line;
  |sphere ∩ line| ≤ 2 unless the line lies in the sphere;
- collinear centers, direction u with Q(u) ≠ 0: radical planes are
  distinct parallels — common intersection empty;
- collinear centers on an *isotropic* direction (Q(u) = 0): all radical
  planes coincide with P_u = u^⊥ ∋ u, and sphere ∩ P_u degenerates to a
  **pair of lines** with 2q points if δ·disc is a square there, else ∅.
  On u^⊥/⟨u⟩ the induced form has square class −1 (disc(Q) = 1, the
  hyperbolic plane carries −1), so the section is empty iff −δ is a
  non-residue. The same condition kills lines-in-spheres.

Both degenerate channels were verified exhaustively per q (below); the
control run confirms the condition is load-bearing.

## 2. Exact verification (all six q — full, not sampled)

The graph is a Cayley graph on (F_q³, +) with connection set
S = {v : Q(v) = δ}, so the bipartite matrix M[x,y] = 1_S(y−x) contains a
3×3 all-ones submatrix iff

    T(u,v) := |S ∩ (S+u) ∩ (S+v)| ≥ 3   for some u ≠ v, both ≠ 0.

(Translation invariance: row triple {a,b,c} ↦ (u,v) = (b−a, c−a); the
matrix is symmetric with empty diagonal, so column triples add nothing,
and any 3×3 all-ones needs a triple of distinct rows with 3 common
columns — overlapping row/column triples would need loops.)
`max_triple_common` computes max T(u,v) **exactly over all ~q⁶ pairs**:

| q  | δ | n = q³ | |S| = q²−q | E = q⁵−q⁴ | max T | isotropic sections |
|----|---|--------|-----------|-----------|-------|--------------------|
| 7  | 1 | 343    | 42        | 14406     | 2     | all empty (48 dirs) |
| 11 | 1 | 1331   | 110       | 146410    | 2     | all empty (120)    |
| 13 | 2 | 2197   | 156       | 342732    | 2     | all empty (168)    |
| 17 | 3 | 4913   | 272       | 1336336   | 2     | all empty (288)    |
| 19 | 1 | 6859   | 342       | 2345778   | 2     | all empty (360)    |
| 23 | 1 | 12167  | 506       | 6156502   | 2     | all empty (528)    |

- max T = **2 exactly at every q**: K₃,₃-free, and *triple-saturated* —
  some triple of rows does have 2 common columns, so the capacity cap is
  touched even though the average triple coverage is only ~1 (see §4).
- Cross-validation at q = 7: the explicit 343×343 matrix, all C(343,3)
  ≈ 6.6M row triples by the pair method — max common columns 2. The
  Cayley reduction and the brute force agree.
- Control (falsification of the residue condition): q = 7 with δ = 3
  (χ(−δ) = +1) gives max T = **14 = 2q** — the predicted two-line
  degenerate section, i.e. a K_{q,2q} through every isotropic line. The
  non-residue condition is exactly what removes it.
- Sphere size q² − q exact (the χ(−δ) = −1 sphere is the small one);
  degrees are exactly q²−q on both sides; E = q⁵ − q⁴ exact.

Status: **MEASURED-EXACT** (deterministic exhaustive computation).

## 3. The measured second-order law

Bipartite density against the truth constant 1:

    e(q) / n^{5/3} = (q⁵ − q⁴) / q⁵ = 1 − 1/q = 1 − n^{−1/3}   exactly.

Requested fit e = n^{5/3}(1 − a·n^{−b}): log-log regression over the six
orders returns

    b = 0.333333333333 (= 1/3),  a = 1.000000000000,  residuals ≤ 2e−15.

The fit is degenerate because the law is EXACT — the strongest possible
outcome of the measurement:

    **e_Brown(n) = n^{5/3} − n^{4/3}  at every n = q³.**

So the construction-side second-order profile is: second-order exponent
4/3 (= 5/3 − 1/3), second-order coefficient exactly −1.

## 4. Where the truth can live (the second-order window)

Write z(n,n;3,3) = n^{5/3} + c₂(n)·n^{4/3} + …  Then, along n = q³:

- c₂ ≥ −1: PROVEN (this construction, §2–3).
- c₂ ≤ 2 + o(1): Füredi's printed bound z ≤ n·n^{2/3} + 2n^{4/3} + n
  [SOURCED: FS survey Thm 3.19 / Füredi CPC 1996; numerically dominates
  all 161 exact cells per theory.md C4b].
- The tail-integral self-improvement fails: integrating the φ-theorem's
  column-tail bound (phi_profile.md, P1) gives only O(n^{4/3} log n).
  +2n^{4/3} stands as the best upper second-order term available here.

Finite-size context (exact z-table, m = 3..16): c₂(m) =
(z(m,m) − m^{5/3})/m^{4/3} = 0.41, 0.46, 0.63, 0.57, 0.55, 0.63, 0.54,
0.63, 0.60, 0.62, 0.66, 0.70, 0.78, 0.66 — positive and drifting UP,
because m ≤ 16 is in the budget-tracking regime (the budget LP has
c₂-analogue ≈ (2^{1/3}−1)m^{1/3} + m^{−1/3}, cf. ladder_asymptotics.md).
The descent toward the asymptotic regime cannot begin before the budget
and Füredi curves cross:

    m* = 470   (smallest m with m^{5/3} + 2m^{4/3} + m below the
                budget-LP value 2^{1/3}m^{5/3} + m; exact crossing
                computed in phi_profile.py).

Below m*, no degree/budget argument can even see the descent — which is
why the known table "lives on the budget curve" (diagonal_limit.md §3)
and why small-m data cannot arbitrate the sign of lim c₂.

## 5. The falsifiable conjecture

**CONJECTURE B (second-order term of z(n,n;3,3) at Brown orders).**
Along n = q³ (q prime, χ(−δ) = −1 as above):

    z(n,n;3,3) = n^{5/3} − n^{4/3} + o(n^{4/3}),  i.e.  lim c₂ = −1:
    Brown's construction is second-order optimal at its own orders.

Grounds (all labeled):
- PROVEN: −1 ≤ liminf c₂ ≤ limsup c₂ ≤ 2 (this file §4).
- PROVEN (phi_profile.md P1/P2): at every level c ≤ 1 the maximum
  supply has average triple-coverage 1 + o(1), never 2 — any density
  gain over Brown must come from o(1)-scale coverage slack or from
  weight-inhomogeneity across levels, both second-order channels with
  no known construction using them.
- HEURISTIC RISK, stated plainly: in the (2,2) analogue the bipartite
  incidence construction (projective plane) beats the symmetric/polarity
  graph at second order and the true c₂-analogue is +1/2, not negative.
  If a (3,3) analogue of that asymmetry exists, Conjecture B fails on
  the high side. No such object is known (the natural candidate —
  inversive-plane/sphere 3-designs — has block size Θ(m^{1/2}), the
  wrong scale; doubling the sphere family violates capacity).

Falsification criteria (either direction):
(i) any K₃,₃-free q³×q³ matrix with > q⁵ − q⁴ + ε·q⁴ ones for fixed
    ε > 0 and large q (kills c₂ = −1 from above); a SAT/ILP search
    adding columns/ones to the q = 7 witness at 343×343 vs the 14406
    baseline is the first computational probe.
    **Probe executed (brown_extend.py, construction-evaluation only):
    ALL 103,243 zero-cells of the q = 7 matrix AND all 1,625,151
    zero-cells of the q = 11 matrix tested — ZERO are individually
    addable at either size. The Brown bipartite matrix is exactly
    1-MAXIMAL: every single 0→1 flip creates a K₃,₃, even though the
    matrices sit far below the budget-regime upper bounds (e.g. ~7,100
    ones of headroom at n = 343). (Mechanism: the construction is
    triple-saturated — the measured max T = 2 triples are dense enough
    that every non-edge (x,y) sees a pair a,b ∈ N(y) with
    |N(x)∩N(a)∩N(b)| = 2.) Measured local rigidity at two sizes,
    1.73M candidate flips, zero exceptions — support for B, though
    multi-cell rearrangements remain unexplored;
(ii) an upper bound z(n,n) ≤ n^{5/3} + o(n^{4/3}) or better (would
    confirm the sign question toward −1 ≤ c₂ ≤ 0);
(iii) exact z(343,343) — far beyond current solvers, recorded for the
    long game.

Status: CONJECTURE (explicit, two-sided falsifiable, tested to the
stated extent; the PROVEN window is [−1, 2]).

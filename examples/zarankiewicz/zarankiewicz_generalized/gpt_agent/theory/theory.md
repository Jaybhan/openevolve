# Theory base for z(m,n;s,t) — precise statements, citations, verification status

Author: theory agent, 2026-07-28. Companion files: `published_data.py`
(independent transcriptions of the published tables), `verify_claims.py`
(runnable checks), `verified_claims.md` (its output), `novelty_checklist.md`.

**Convention** (matches `../../evaluator.py` and the team README):
z(m,n;s,t) = maximum number of 1s in an m×n 0/1 matrix with no s×t all-ones
submatrix, where s counts **rows** (the m-side) and t counts **columns**.
Duality: z(m,n;s,t) = z(n,m;t,s). The problem is Zarankiewicz's Problem P 101,
Colloq. Math. 2 (1951), 301 (https://eudml.org/doc/209990).

Status tags:
- **[VERIFIED-NUMERICALLY]** — checked by `verify_claims.py` against the local
  proven-exact table (161 cells of z(m,n;3,3)) and/or by direct computation
  (exhaustive search, witness decoding, identity checks). Claim ID given.
- **[SOURCED]** — precise citation, not independently recomputed here.
- **[UNCERTAIN]** — could not fully pin down; what was searched is stated.

---

## 0. Provenance of the local ground-truth table (load-bearing)

The 161-cell table in `evaluator.py` (`_TABLE`, `_EXACT_UP_TO`) **is** Table 3
of:

> Jeremy Tan, *An attack on Zarankiewicz's problem through SAT solving*,
> arXiv:2203.02283v2 (2022). https://arxiv.org/abs/2203.02283
> Data + machine-checkable certificates: https://github.com/Parcly-Taxel/Kyoto
> (UNSAT proofs via Kissat + DRAT; maximal matrices in base64).

**[VERIFIED-NUMERICALLY, C1]** All local cells match Tan's printed values, and
the local `_EXACT_UP_TO` limits coincide **cell-for-cell** with Tan's boldface
(=proven exact) region. Two deliberate deviations, both from

> J. Bhan, N. Nobili, P. Langer, *New Bounds for Zarankiewicz Numbers via
> Reinforced LLM Evolutionary Search*, arXiv:2605.01120 (May 2026)
> https://arxiv.org/abs/2605.01120 — the project owner's own prior paper,

namely `_EXTRA_EXACT = {(11,21): 116, (12,22): 132}`:
- (11,21): Tan printed the Roman bound 117 (unbold); Davies–Gill–Horsley
  improved the UB to 116 (their Table 2), and arXiv:2605.01120 found a
  116-edge witness ⇒ exact. **[VERIFIED-NUMERICALLY, C14b]** (UB side
  [SOURCED]; Roman gives only 117 — C6d).
- (12,22): the Roman bound is exactly 132 **[VERIFIED-NUMERICALLY, C6c]**, and
  arXiv:2605.01120 found a 132-edge witness ⇒ exact.
- Note: arXiv:2605.01120 also proved **z(11,22;3,3)=121 exact**, which is
  *not* in the local suite (informational for the owner; C14d).

Diagonal cross-checks: z(n,n;3,3) for n=3..16 equals OEIS A001198 − 1
(9,14,21,27,34,43,50,61,70,81,93,106,121,129 → 8,13,20,26,33,42,49,60,69,80,
92,105,120,128); A001198(16)=129 is the term Tan added ("first new term in
over 50 years"). History per OEIS: Sierpiński (1951) found k₃(4..6), k₃(7) is
due to Brzeziński, k₃(8) to Čulík (1956). **[VERIFIED-NUMERICALLY, C2a,C2e]**
https://oeis.org/A001198

A second published source overlaps the table: Collins–Riasanovsky–Wallace–
Radziszowski (2016) (see §5); all 73 of their boldface (3,3) cells that lie in
the local suite agree with it. **[VERIFIED-NUMERICALLY, C3a]**

---

## 1. Kővári–Sós–Turán bound

> T. Kővári, V. T. Sós, P. Turán, *On a problem of K. Zarankiewicz*,
> Colloq. Math. 3 (1954), 50–57. Rectangular/general-(s,t) form:
> C. Hyltén-Cavallius, *On a combinatorical problem*, Colloq. Math. 6 (1958),
> 59–65. (Both are cited jointly for the bound below by Davies–Gill–Horsley.)

Statement (as in Davies–Gill–Horsley, arXiv:2411.18842, our convention):

    z(m,n;s,t) < (t−1)^{1/s} · m · n^{1−1/s} + (s−1) · n,

together with the dual form with (m,s) ↔ (n,t). Proof idea: count pairs
(s-subset of rows, column covering it); each s-subset is covered by ≤ t−1
columns; convexity of c ↦ C(c,s) does the rest.

**[VERIFIED-NUMERICALLY, C4a]** min of the two orientations dominates all 161
proven cells. It is never tight in this range (min slack ≈ 5.9; C4c) — KST is
the right *order*, not the right *value*, at small sizes.

Sharper closed form containing KST (Füredi, CPC 1996; as stated in the
Füredi–Simonovits survey, Theorem 3.19 — see §5 for the citation):

    Z(m,n,a,b) ≤ (b−a+1)^{1/a} m n^{1−1/a} + (a−1) n^{2−2/a} + (a−2) m
    (m ≥ a, n ≥ b, b ≥ a ≥ 2).

**[VERIFIED-NUMERICALLY, C4b]** dominates all 161 cells.

---

## 2. Čulík's theorem — the elongated regime (load-bearing for us)

> Karel Čulík, *Teilweise Lösung eines verallgemeinerten Problems von
> K. Zarankiewicz*, Ann. Polon. Math. 3 (1956), 165–168.
> https://old.impan.pl/en/publishing-house/journals-and-series/annales-polonici-mathematici/all/3/1/94221/teilweise-losung-eines-verallgemeinerten-problems-von-k-zarankiewicz

**Statement** (as Corollary 2.1 in Tan 2022; equivalently in Davies–Gill–
Horsley as "(1) with k=s−1 holds with equality for all n ≥ (t−1)·C(m,s)"):

    If 1 ≤ s ≤ m and  n ≥ (t−1)·C(m,s),  then
        z(m,n;s,t) = (s−1)·n + (t−1)·C(m,s).

Construction achieving it: for each of the C(m,s) s-subsets of rows, take t−1
columns equal to its indicator (that consumes (t−1)C(m,s) columns of weight s,
no s-set is covered t times); fill the remaining n − (t−1)C(m,s) columns with
weight s−1 (a column of weight ≤ s−1 can never participate in an s×t all-ones
minor). Total: s(t−1)C(m,s) + (s−1)(n−(t−1)C(m,s)) = (s−1)n + (t−1)C(m,s).

**[VERIFIED-NUMERICALLY, C5a–c]** For (s,t)=(3,3): z = 2n + 2·C(m,3) on **all
41** cells of the local table with n ≥ 2·C(m,3) (row m=3 entirely: 2n+2; row 4
from n=8: 2n+8; row 5 from n=20: 2n+20); strictly below the Čulík value on
every cell below the threshold; and equality begins *exactly* at the threshold
where the table brackets it (m=4: n=7 vs 8; m=5: n=19 vs 20).

**[VERIFIED-NUMERICALLY, C9a–d]** Beyond (3,3), verified by exhaustive
computation of z from scratch on small cases: z(3,n;3,3)=2n+2 (n=3..6);
z(3,n;2,3) = n+6 exactly from n = (t−1)C(m,s) = 6 on (and < before);
z(4,n;2,2) = n+6 exactly from n = C(4,2) = 6 on.

---

## 3. Roman's bound, packing numbers, and the modern LP view

> Steven Roman, *A problem of Zarankiewicz*, J. Combin. Theory Ser. A 18
> (1975), 187–198. doi:10.1016/0097-3165(75)90007-2

**Statement** (Tan 2022, Thm 2.2; identically Davies–Gill–Horsley Thm 2.1
without the floor): for every integer p ≥ s−1,

    z(m,n;s,t) ≤ ⌊ (t−1)/C(p,s−1) · C(m,s) + (p+1)(s−1)/s · n ⌋ .

p = s−1 recovers Čulík's value. **Exactness window** (Tan Thm 2.2): equality
holds with p = s or p = s−1 whenever

    n ≥ (t−1)·C(m,s) − s·T_{s,t}(m),

where T_{s,t}(m) is the largest multiset of (s+1)-subsets of an m-set covering
every s-subset at most t−1 times (a λ-fold packing number; for (s,t)=(3,3)
this is the 2-fold packing of triples by quadruples, D₂(m,4,3)).

**Attribution caveat — what did Roman himself prove? [UNCERTAIN]** Tan's
Theorem 2.2 attributes the *whole* statement (general-(s,t) bound AND the
T-window equality) to Roman [15]. But three other expert treatments cite
Roman for the *bound only*: Chen–Horsley–Mammoliti cite "[Roman, Theorem 1]"
for the s=2 bound and credit the threshold equality to Čulík and the s=2
critical-point design characterization to Reiman 1968 ("Su una proprietà dei
2-disegni", their [18]); Damásdi–Héger–Szőnyi speak only of "Roman's bound"
(their Remark 3.14) and prove their own design-regime equality results
(their Prop 3.25) without crediting Roman; Davies–Gill–Horsley's list of
known exactness results ([3,6,7,8]) omits Roman. Roman's paper itself
(JCTA 18 (1975) 187–198, Zbl 0296.05014) is paywalled and its review text
is license-blocked, so I could not settle this from the primary source.
The window itself is numerically true on our table (C6b) regardless of who
first proved it; but before publishing anything that leans on "Roman's
equality window", obtain the paper and check. Note the design-side
statements that ARE clearly published: at any integral "Roman point"
n = k(t−1)C(m,s)/C(k,s), the bound is attained iff an s-(m,k,t−1) design
exists (Reiman 1968 for s=2; stated for general s, uncredited, by
Davies–Gill–Horsley §2); and Damásdi–Héger–Szőnyi Prop. 3.25: for
admissible (t,v,k,λ) and 0 ≤ c ≤ c₀ (an explicit c₀),
Z_{t,λ+1}(v−c, b) ≤ r(v−c), **with equality whenever a t-(v,k,λ) design
exists** — the design-to-exact-Zarankiewicz bridge, published 2013, for
general (t, λ+1). [SOURCED, statements read directly from the DHS PDF]

**[VERIFIED-NUMERICALLY, C6a,b]** min_p of the bound dominates all 161 cells;
and z equals min(Roman(p=2), Roman(p=3)) on **all 71 cells** in the window
n ≥ 2C(m,3) − 3·T₃,₃(m) — i.e. rows m=3,4,5 **entirely** (windows start at
n=2, 2, 5) and row 6 from n=13. Closed forms this certifies:
row 4: z(4,n;3,3) = ⌊(8n+8)/3⌋ (all n≥4); row 5: z(5,n;3,3) = ⌊(8n+20)/3⌋
(all n≥5) — these are *published theorems*, not empirical fits (see the
novelty checklist, item R1).

**Packing numbers.** T₂,₂(m) = ⌊(m/3)⌊(m−1)/2⌋⌋ − [m≡5 mod 6] (Guy). For
λ=1 quadruples: T₃,₂(m) = ⌊(m/4)⌊((m−1)/3)⌊(m−2)/2⌋⌋ − [m≡0 mod 6]⌋
(Bao–Ji, arXiv:1401.2022, Des. Codes Cryptogr. 2014). For T₃,₃ **Tan states
"we found no corresponding results in the literature"** and computed with
Gurobi (his Table 1):

    m      3  4  5  6  7   8   9  10  11  12   13   14   15   16   17   18
    T₃,₃   0  2  5  9  15  28  40  60  80  108  143  182  225  280  340  405

(the m ∈ {8,10,14,16} entries are the perfect values C(m,3)/2, via doubling a
Steiner quadruple system — SQS(m) exists iff m ≡ 2,4 (mod 6):
> Haim Hanani, *On quadruple systems*, Canad. J. Math. 12 (1960), 145–157.)
**Correction note (2026-07-28):** an earlier revision of `published_data.py`
had T₃,₃(18)=408 — a transcription-completion error by this agent, *not*
Tan's number; Tan prints **340** at m=17 and **405** at m=18 (re-read
verbatim; C16).

**Structure of the T₃,₃ values [VERIFIED-NUMERICALLY, C13, C16]:**
- T₃,₃(m) = C(m,3)/2 (perfect) **iff** a 3-(m,4,2) design is admissible,
  i.e. 3 | (m−1)(m−2) and C(m,3) even (equivalently m ≢ 0 mod 3 and
  m ≢ 3 mod 4) — checked for all m = 4..18 (C16c). Sufficiency of the
  divisibility conditions for 3-(v,4,λ) designs is Hanani's spectrum
  theorem for block size 4 [SOURCED-secondary: attributed to Hanani
  (1963 *On some tactical configurations*, Canad. J. Math. 15 / his 1968
  work) throughout the design literature; exact original paper UNCERTAIN —
  for m ≤ 18 existence is independently certified by Tan's explicit
  constructions, which I validated at m=17].
- Per-point **Johnson bound**: r_x ≤ ⌊(m−1)(m−2)/3⌋, so
  J(m) := ⌊m·⌊(m−1)(m−2)/3⌋/4⌋ ≥ T₃,₃(m). On m=3..18, J is **tight
  everywhere except m ∈ {7,11}**, where the slack is exactly 2 (C16b);
  in particular J is tight at all of m = 6, 9, 12, 15, 18 (m ≡ 0 mod 3).
- Consequently **T₃,₃(18) = 405 is provable without solver trust**: a
  3-(18,4,2) design is inadmissible (r = 272/3 ∉ ℤ; C16a), the Johnson
  bound gives ≤ 405, and Tan's printed cyclic presentation (17 base blocks
  under a dihedral group of order 36) generates exactly 405 blocks forming
  a valid 2-fold packing (C16d,e). The same Johnson-tightness argument
  de-solvers m = 6, 9, 12, 15 as well; only m = 7, 11 (J−2) rest on
  exhaustive computation (Tan's Gurobi + the coordinator's search).
- T₃,₃(6)=9 and T₃,₃(7)=15 re-verified here (C13a,c,d).

Design-theory literature search (λ=2 *maximum packings* of triples by
quadruples, D₂(v,4,3)) found only the λ=1 case treated ([SOURCED]: Bao–Ji
2014, Hartman–Phelps-era work, and references) — Tan's no-literature claim
stands as of this search; see novelty checklist N6 for what a full λ=2
determination would be worth.

**Modern LP view and current-best small upper bounds.**
> S. Davies, P. Gill, D. Horsley, *Improved upper bounds on Zarankiewicz
> numbers*, arXiv:2411.18842; Discrete Mathematics 349 (2026), 114924.
> https://arxiv.org/abs/2411.18842

Roman's bound = optimal value of a small LP over the column-size profile
(n_{s−1},…,n_m) of an "(s,t−1)-linear hypergraph". DGH add a new family of
constraints (their Thm 1.1, indexed by v < s ≤ k ≤ m; v = s−1 is the useful
one), generalizing the s=2 constraints of Chen–Horsley–Mammoliti, and derive
a closed-form family B_k(m,n;s,t) (their Thm 1.2, k ≥ max{2, s²−2s}). Their
Table 2 lowers the (3,3) UBs at 29 cells in our region, mostly by 1 (embedded
in `published_data.py`); all are ≥ the local exact values where comparable
**[VERIFIED-NUMERICALLY, C14a]**.

**Relation to the coordinator's waterfill WF(m,n).** My independent
implementation confirms WF ≥ z on all cells, WF-tight on exactly **78/161**
cells, WF(16,16)=136 (deficit 8), *and* WF == min_p Roman on **all 161
cells** — the integer level-filling bound coincides with Roman's floored
closed form throughout this range. **[VERIFIED-NUMERICALLY, C7a–d]** So
"WF" is not a new bound here; it *is* Roman's bound. Improvements must come
from DGH-type extra constraints (published) or genuinely new structure.

**Exact (2,t) below the Čulík threshold** (the s=2 analog of what the team
is attempting for s=3):
> G. Chen, D. Horsley, A. Mammoliti, *Exact values for unbalanced
> Zarankiewicz numbers*, J. Graph Theory 106 (2024), 81–109;
> arXiv:2202.05507. — For each t ≥ 2, Z_{2,t}(m,n) = (t−1)C(m,2) + n for n
> large, and exact values in "almost all" remaining cases with n = Θ(tm²).
[SOURCED] The s ≥ 3 analog is open — that is the gap the team could fill.

---

## 4. The (2,2) case

**Reiman's bound.**
> István Reiman, *Über ein Problem von K. Zarankiewicz*, Acta Math. Acad.
> Sci. Hungar. 9 (1958), 269–273. (Also I. Reiman, *Su una proprietà dei
> 2-disegni*, Rend. Mat. 1 (1968), 75–81.)

    z(m,n;2,2) ≤ (1/2)·(m + sqrt(m² + 4·m·n·(n−1)))     (FS survey Thm 3.2)
    z(n,n;2,2) ≤ (n/2)·(1 + sqrt(4n−3)).

**Equality at projective planes** — and the attribution point: at
n = q²+q+1 the square bound is *exactly* (q+1)(q²+q+1) — an algebraic
identity, 4n−3 = (2q+1)² — and the point-line incidence matrix of PG(2,q)
attains it. Hence for every prime power q,

    z(q²+q+1, q²+q+1; 2,2) = (q+1)(q²+q+1),

and this is **Reiman (1958)**, not Füredi, with no q ≥ 15 condition.
**[VERIFIED-NUMERICALLY, C8a–c]**: identity checked for q=2..199; equality
against the known value table occurs exactly at n = 1,3,7,13,21,31
(q = 0,1 degenerate; q = 2,3,4,5: z = 21, 52, 105, 186). The prompt's
"Füredi, q ≥ 15" conflates this with Füredi's theorem on the *non-bipartite*
quadrilateral-free problem: ex(q²+q+1; C₄) = (1/2)q(q+1)² for q > 13, via
orthogonal polarity graphs —
> Z. Füredi, *On the number of edges of quadrilateral-free graphs*,
> J. Combin. Theory Ser. B 68 (1996), 1–6. [SOURCED]
(Füredi's *An Upper Bound on Zarankiewicz' Problem*, CPC 5 (1996) 29–33, is
the (3,3) paper — its abstract confirms this; see §5.)

**Rectangular exact families [SOURCED]:**
- Kővári–Sós–Turán (1954) already gave Z(p²+p, p², 2, 2) = p³+p² (affine
  planes; stated in FS survey §3).
- Steiner-system equality (Reiman): the bound (3.3) is tight when
  m = n(n−1)/(k(k−1)) and a Steiner system S(2,k,n) exists (FS survey).
- Čulík (1956): z(m,n;2,2) = n + C(m,2) for n ≥ C(m,2)
  (**[VERIFIED-NUMERICALLY, C9d]** for m=4 by exhaustive search).
- Chen–Horsley–Mammoliti (2024): the (2,t) window below the threshold (§3).
- Finite-geometry exact values and bounds near planes: G. Damásdi, T. Héger,
  T. Szőnyi, *The Zarankiewicz problem, cages, and geometries*, Ann. Univ.
  Sci. Budapest. Eötvös Sect. Math. 56 (2013), 3–37, PDF:
  https://heger.web.elte.hu/publ/Damasdi-Heger-Szonyi-Zarankiewicz-cages-geometries.pdf
  (read directly). Their **Theorem 1.8** (assuming a projective plane of
  order n exists; n ≥ 15 in the first case — via Metsch's embedding
  theorem — and n ≥ 4 in the fourth):
  Z₂,₂(n²+n+1−c, n²+n+1) = (n²+n+1−c)(n+1) for 0 ≤ c ≤ n/2;
  Z₂,₂(n²+c, n²+n) = n²(n+1)+cn for 0 ≤ c ≤ n+1;
  Z₂,₂(n²−n+c, n²+n−1) = (n²−n)(n+1)+cn for 0 ≤ c ≤ 2n;
  Z₂,₂(n²−2n+1+c, n²+n−2) = (n²−2n+1)(n+1)+cn for 0 ≤ c ≤ 3(n−1);
  with extremal graphs embeddable in the plane. **This "n ≥ 15" is almost
  certainly the source of the task-prompt's "Füredi q ≥ 15" recollection.**
  Their Table 1 tabulates best-known Z₂,₂(m,n) for m ≤ 23, n ≤ 31 (bold =
  exact; italic = exact-but-not-in-Guy; relies partly on Guy's values with
  a stated caveat — the caveat Tan quotes). Also their Prop. 3.25
  (design-regime equality, general (t,λ+1)) and Prop. 3.20 (Guy's
  "point C" deletion recursion, with which they prove e.g. z(7,7;3,3) ≤ 33
  — a human-readable proof of a value Roman's bound only gives 35 for).
  And T. Héger's PhD thesis (2013),
  https://heger.web.elte.hu/publ/HTdiss-e.pdf. [SOURCED, PDF read]
- Asymptotics for (2,k): C. Hyltén-Cavallius (1958) bounded
  lim z(n,n;2,k)·n^{−3/2}; M. Mörs, *A new result on the problem of
  Zarankiewicz*, J. Combin. Theory Ser. A 31 (1981), 126–130, improved it.
  [SOURCED — note: Mörs 1981 is an asymptotic (2,k) result, **not** a
  rectangular exact-value theorem.]

**Known exact square values** — z(n,n;2,2) for n = 1..31
**[VERIFIED-NUMERICALLY, C2b–d, C8]**:

    1, 3, 6, 9, 12, 16, 21, 24, 29, 34, 39, 45, 52, 56, 61, 67, 74, 81, 88,
    96, 105, 108, 115, 122, 130, 138, 147, 156, 165, 175, 186
    (n=32: 189 or 190, undecided as of CRWR 2016)

Sources: OEIS A001197 (k₂(n) = z+1) and A072567 (z directly), both to n=24;
Guy (1969) to n=21; the n ≤ 31 extension is unpublished work of Afzaly–McKay
cited and tabulated in CRWR 2016 (their Table 3). Tan 2022 independently
recomputed the rectangular z₂ table and **corrected exactly 8 errors in
Guy's z₂(m,n) tables, all too low by 1** (first error found by Héger); Tan's
square z₂ values agree with OEIS/CRWR on all overlaps
**[VERIFIED-NUMERICALLY, C2]**. First square counting deficit:
z(8,8;2,2)=24 < WF=25 **[VERIFIED-NUMERICALLY, C8d]** — exactly the
coordinator's predicted demand/supply breakpoint.
- R. K. Guy, *A problem of Zarankiewicz*, Res. Paper 12, Univ. Calgary 1967,
  scan: https://oeis.org/A001197/a001197.pdf; and *A many-facetted problem
  of Zarankiewicz*, Lect. Notes Math. 110 (1969), 129–148,
  doi:10.1007/BFb0060112.

---

## 5. The (3,3) case

**Asymptotics.** Lower bound construction:
> W. G. Brown, *On graphs that do not contain a Thomsen graph*, Canad. Math.
> Bull. 9 (1966), 281–285. — A (p²−p)-regular K₃,₃-free graph on p³ vertices
> (unit "spheres" in AG(3,p), p ≡ 3 mod 4 prime), giving
> ex(n; K₃,₃) ≥ (1/2)n^{5/3}(1+o(1)).

Matching upper bound:
> Z. Füredi, *An Upper Bound on Zarankiewicz' Problem*, Combin. Probab.
> Comput. 5 (1996), 29–33. Abstract (fetched): "Improving earlier results of
> Kővári, T. Sós and Turán on Zarankiewicz' problem, we obtain that Brown's
> example for a maximal K₃,₃-free graph is asymptotically optimal."
> General form = §1's Thm 3.19 (from the survey: Z. Füredi, M. Simonovits,
> *The history of degenerate (bipartite) extremal graph problems*,
> arXiv:1306.5167; Erdős Centennial, 2013).

Consequently ex(n; K₃,₃) = (1/2 + o(1))·n^{5/3} — and, in the **bipartite**
normalization used here, z(n,n;3,3) = (1 + o(1))·n^{5/3}. Keep the factor-2
bookkeeping straight: the constant 1/2 belongs to the graph version. Table
ratios z(n,n)/n^{5/3} = 1.31, 1.27, 1.26 at n=8,12,16 (still above the limit
— small-size effect) **[C15, informational]**.

**Exact values.** Chain of computational sources, all cross-checked here:
1. Guy 1967/1969 (hand tables, a=2,3,4).
2. A. F. Collins, *Bipartite Ramsey Numbers and Zarankiewicz Numbers*,
   MS thesis, RIT, 2015; and
   A. Collins, A. Riasanovsky, J. Wallace, S. Radziszowski, *Zarankiewicz
   Numbers and Bipartite Ramsey Numbers*, J. Algorithms Comput. 47 (2016),
   63–78; arXiv:1604.01257. Appendix Table 4: z(m,n;3) for 6 ≤ m ≤ n ≤ 18
   (bold = exact, * = unique extremal graph, † = also unique (z−1)-graph,
   italic = exhaustive computation). **z(16;3) = 128 with a uniqueness star
   is theirs (2016).**
3. Tan 2022 (§0): SAT + symmetry, machine-verifiable UNSAT proofs; extends
   and independently re-verifies; provides all maximal matrices for squares.
   For z₃(16)=128 Tan lists one maximal matrix and states the *complete
   list* "has not yet been proven" (so CRWR's uniqueness star vs Tan's hedge
   is a live discrepancy about uniqueness, not about the value).
4. DGH 2024/2026: improved UBs at 29 of the open cells (§3).
5. Bhan–Nobili–Langer 2026 (arXiv:2605.01120): three cells closed —
   (11,21)=116, (11,22)=121, (12,22)=132 — plus new LBs at 41 open cells in
   9≤m≤16, 17≤n≤23.

**Literature discrepancy worth knowing [VERIFIED-NUMERICALLY, C3b,c]:**
CRWR's appendix claims **z(12,17;3,3) = 103 exact** (bold, unique-graph
star) and exhaustively-computed UBs sharper than Tan's printed Roman bounds
at nine further cells — e.g. (13,17) ≤ 110 vs Tan's print 117 and DGH's
"improved" 116. Tan deliberately did not import interior published values;
DGH compare only against Tan. So either (a) CRWR's rectangular interior
values are right and the current best-known-UB bookkeeping (incl. DGH) is
too pessimistic — and (12,17)=103 is a *free* exact cell — or (b) CRWR's
interior rectangular claims are erroneous (their square values all check
out, C3a). Unresolved in print as of 2026-07. See novelty checklist, item N4.

**The (16,16) witness.** Tan's published 128-one matrix (8-regular on both
sides) decodes and validates **[C10]**, and the evolved champion's
construction — rows = F₂⁴, columns = both sides of the 8 affine hyperplanes
whose normals form the cap {a: a₃=1} in PG(3,2) — is **isomorphic to Tan's
witness** (backtracking bipartite isomorphism; C11). Max cap size 8 in
PG(3,2) re-verified exhaustively (C12; the size-2ⁿ cap in PG(n,2) —
complement of a hyperplane — is classical folklore; classification of large
binary caps: Davydov–Tombak, e.g. as surveyed in Hirschfeld,
*Projective Geometries over Finite Fields* [SOURCED-general; exact original
attribution UNCERTAIN — Bose 1947 concerns q>2 caps/ovals, not the binary
case]).

---

## 6. General (s,t): what is known vs open

- **Upper bound**: KST gives z(n,n;s,t) = O(n^{2−1/s}) for t ≥ s (§1).
  Refinements: V. Nikiforov, *A contribution to the Zarankiewicz problem*,
  arXiv:0903.5350 (2009) (flexible bound implying KST and Füredi);
  D. Conlon, *Some remarks on the Zarankiewicz problem*, Math. Proc.
  Cambridge Philos. Soc. 173 (2021), doi:10.1017/S0305004121000475. [SOURCED]
- **Matching lower bounds (orders known):**
  - s = 2: Erdős–Rényi–Sós / Brown / Reiman — Θ(n^{3/2}), constant known.
  - s = 3 (t = 3): Brown + Füredi — Θ(n^{5/3}), constant known (§5).
  - Norm graphs: J. Kollár, L. Rónyai, T. Szabó, *Norm-graphs and bipartite
    Turán numbers*, Combinatorica 16 (1996), 399–406: ex(n, K_{s,t}) =
    Θ(n^{2−1/s}) for **t ≥ s! + 1**.
  - Projective norm graphs: N. Alon, L. Rónyai, T. Szabó, *Norm-graphs:
    variations and applications*, J. Combin. Theory Ser. B 76 (1999),
    280–290: threshold improved to **t ≥ (s−1)! + 1** (FS survey Thm 1.3:
    ex(n,K_{a,b}) > c_a n^{2−1/a} for b > (a−1)!).
  - B. Bukh, *Extremal graphs without exponentially-small bicliques*,
    arXiv:2107.04167; Duke Math. J. 173 (2024), 2039–2062: threshold
    improved to **t ≥ C^s** (single-exponential). [SOURCED]
- **Open**: the order of z(n,n;4,4) (equivalently ex(n,K₄,₄)) is **unknown**:
  between Ω(n^{5/3}) (any K₃,₃-free example is K₄,₄-free) and O(n^{7/4})
  (KST). Smallest open (s,t) by the ARS threshold: (4,4), (4,5), (4,6) —
  (4,7) is closed since 7 = 3!+1. Bukh's bound does not reach these. This
  is the standard statement of the smallest open cases (FS survey §1.2/§3).
  [SOURCED]

---

## 7. Developments 2021–2026 (incl. AI-assisted)

- **Conlon 2021** (above): modern take on when KST-type bounds are tight.
  [SOURCED]
- **Tan 2022** (arXiv:2203.02283): SAT + DRAT certificates; corrected Guy;
  z₃(16)=128 published to OEIS (A001198(16)=129). [Load-bearing; §0.]
- **Chen–Horsley–Mammoliti 2022→JGT 2024**: exact unbalanced (2,t) below
  the Čulík threshold. [§3]
- **Mulrenin–Nagle 2024**: *Some imbalanced hypergraph Zarankiewicz
  numbers*, Bull. Inst. Combin. Appl. 102 (2024), 116–128 (hypergraph
  variant). [SOURCED, cited by DGH]
- **Davies–Gill–Horsley 2024 → Discrete Math. 349 (2026) 114924**: LP with
  new constraints; current best small-(s,t) UBs. [§3]
- **Bhan–Nobili–Langer, May 2026** (arXiv:2605.01120): OpenEvolve/LLM
  evolutionary search; 3 newly exact (3,3) cells, 41 new LBs, ~$30/cell.
  **This is the project's own prior paper and the direct prior art for the
  present experiment.**
- **AlphaEvolve** (Novikov et al., May 2025; DeepMind blog
  https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
  and **Georgiev–Gómez-Serrano–Tao–Wagner, *Mathematical exploration and
  discovery at scale***, arXiv:2511.02864 (Nov 2025): 67-problem math
  benchmark (analysis, combinatorics, geometry, number theory). The abstract
  and available material do **not** mention Zarankiewicz; the Zarankiewicz
  application of this methodology is Bhan–Nobili–Langer above.
  [SOURCED; absence-of-mention checked against abstract only — UNCERTAIN at
  the level of the full 67-problem list.]
- **"Limited augmented Zarankiewicz" variant**: L. Qi, C. Cui, Y. Xu,
  arXiv:2604.04111 (2026), and the computational follow-up arXiv:2605.29658
  (exact zL values for small parameters). A *variant* problem — do not
  confuse with z(m,n;s,t). [SOURCED]
- **Geometric/tame Zarankiewicz**: an active parallel line (incidence
  bounds for restricted matrices), e.g. *On Zarankiewicz's Problem for
  Intersection Hypergraphs of Geometric Objects*, arXiv:2412.06490 (2024).
  Not directly about exact z(m,n;s,t) values. [SOURCED]

---

## 8. One-paragraph summary for the team

Everything the evaluator scores is Tan's boldface table plus two cells from
the team's own 2026 paper; nothing in the suite is folklore-uncertain
[C1–C3]. The *entire* solvable-by-formula region is already published:
Čulík (41 cells) sits inside Roman-with-packing-numbers (71 cells), and
Roman's floored closed form equals the integer waterfill on every cell of
the table [C5–C7], so any "new bound" claim must beat Roman + DGH's LP, and
any "new formula" claim must explain the 90 cells with positive deficit —
that, plus the s=3 analog of Chen–Horsley–Mammoliti's below-threshold
exactness and the open (12,17) discrepancy, is where genuine novelty lives.

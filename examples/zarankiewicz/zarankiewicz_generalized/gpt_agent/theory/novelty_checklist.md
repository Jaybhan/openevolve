# Novelty checklist — what would actually count as new

Referee document. For each candidate artifact: what already exists (with
evidence), and what the residual delta would be. Statements below lean on
`theory.md` (citations) and `verified_claims.md` (claim IDs).

**The single most important fact for this team's novelty accounting:** the
closest prior art is the team's *own* paper — Bhan–Nobili–Langer,
arXiv:2605.01120 (May 2026): LLM/OpenEvolve evolutionary search on
z(m,n;3,3), 3 newly exact cells ((11,21)=116, (11,22)=121, (12,22)=132), 41
new lower bounds in 9≤m≤16, 17≤n≤23, several within 1 of the best UB.
Nothing that merely re-does that (per-cell search producing matrices) is new.
"LLM-evolution applied to Zarankiewicz" as a genre is also no longer novel.

---

## N1. "A single closed-form / piecewise formula reproducing all 161 proven (3,3) cells"

**Does any paper already state one? NO — with a large caveat.** Evidence:
- Tan 2022 (the table's source) states the values "follow no discernible
  pattern in general other than their strict monotonicity" — no formula
  claimed there.
- BUT a published theorem already covers **71 of the 161 cells exactly**
  (Roman 1975 in Tan's Thm 2.2 form, with packing numbers T₃,₃): all of rows
  3, 4, 5 — z(3,n)=2n+2, z(4,n)=⌊(8n+8)/3⌋, z(5,n)=⌊(8n+20)/3⌋ — and row 6
  for n ≥ 13. Čulík 1956 (41 cells) is the p=s−1 special case. VERIFIED:
  C5, C6. Any claim of "we found a formula for rows 3–5" or "the elongated
  tails" is a **rediscovery** and must be labeled as such.
- The remaining 90 cells all have positive deficit d = WF − z with
  d ∈ {1,...,5} ∪ {8}; no published closed form covers them. A formula that
  (i) states d(m,n) in closed form, (ii) is *proved* (both directions) on
  some infinite family below the Čulík/Roman window, would be genuinely new —
  the honest benchmark is Chen–Horsley–Mammoliti (JGT 2024), who did exactly
  this for s=2. **The s=3 analog is open. That is the flagship target.**
- A formula merely *fitted* to the 161 cells (no proofs beyond the window)
  is not a theorem; it is a conjecture generator. Label per README honesty
  bar: VERIFIED-ALL-KNOWN, not PROVEN.

## N2. "Certified new exact cells beyond the published table"

What exists: Tan-bold cells + BNL's three (incl. **(11,22)=121, which the
local evaluator does NOT include** — do not "discover" it). Upper bounds:
DGH (Discrete Math 2026) is the current best-known catalog for (3,3) in this
region — 29 improved cells, embedded in `published_data.py`.

Delta that would be new:
- Any cell where a construction meets the best published UB (DGH < Roman at
  29 cells; check BNL's Figure 2 first — several of their LBs are already
  within 1).
- Any *new UB improvement* below DGH's LP value (would need either their LP
  plus stronger constraints, integrality/IP arguments, or SAT with DRAT
  certificates à la Tan — certificates are the standard here; a claimed UB
  without a checkable certificate will not be accepted as closing a cell).
- **Cheapest possible real contribution (N4):** resolve the CRWR
  discrepancy at (12,17) — see below. If CRWR is right, z(12,17)=103 is
  already exact in the literature and Tan/DGH/BNL bookkeeping is wrong; if
  CRWR is wrong, a documented refutation (a valid 104+ matrix, or an
  independent exhaustive/SAT run) is a publishable erratum in the tradition
  of Tan's 8 corrections to Guy.

## N3. "A unified constructive algorithm across regimes"

What exists:
- Čulík/Roman-window constructions are explicit and published (weight-s−1
  padding + block columns; Roman's equality via packings; DGH restate as
  designs: equality at "Roman points" ⇔ existence of an s-(m,k,t−1) design).
- Tan publishes *witness matrices* (base64 + certificates) for every bold
  cell — but no generating algorithm ("previous work was mostly limited to
  the values").
- BNL 2026 publish **seven per-cell generator algorithms** (explicit-matrix,
  circulant, perturb-and-repair) for the 44-cell open region.
- The current experiment's champion (`openevolve_output/best/best_program.py`)
  is itself prior art *within the team* (packing-completion + F₂⁴ hyperplane
  blocks + cap construction).

Delta that would be new: **one interpretable algorithm, uniform in (m,n) —
ideally in (m,n,s,t) — reproducing all 161 cells within the 25 ms budget,
with a proof of validity for all inputs and proofs of exactness on stated
families.** No such thing exists in print (checked: Tan, CRWR, DGH, CHM,
BNL). Interpretability + uniformity is precisely the delta over BNL; say so
explicitly when writing up, and benchmark against (a) Roman-window coverage
(71 cells free), (b) BNL's per-cell programs.

## N4. The CRWR rectangular (3,3) discrepancy [flagged by this review]

CRWR 2016 Appendix Table 4 claims **z(12,17;3,3)=103 exact with a unique
extremal graph**, and exhaustive UBs sharper than both Tan's printed Roman
bounds and DGH's 2024 "improvements" at 9 more cells — e.g. (13,17) ≤ 110
(CRWR, italic=exhaustive) vs 116 (DGH "new best"). VERIFIED as a parsing of
the published sources: C3b, C3c. Tan explicitly refused to import interior
published values; DGH compared only to Tan; BNL used DGH+Tan. Nobody in the
2022–2026 chain engages CRWR's rectangular interior claims. Resolving this
(either direction) is real, small, and would correct the literature's
best-known table. **Before the team claims ANY new UB or exact cell in
rows 12–16, n=17..18, check it against CRWR's Table 4.**

## N5. The (16,16) object and the "codes → Zarankiewicz" bridge (coordinator Q1, Q2)

- The VALUE z(16,16;3,3)=128 is CRWR 2016 (bold, star) and Tan 2022; the
  WITNESS matrix is published (Tan, base64). The evolved cap/affine-
  hyperplane construction **is isomorphic to Tan's witness** (VERIFIED,
  C11b) — so it is a new *description* of a known object, not a new object.
  Uniqueness: CRWR claim it (star); Tan says the complete list is unproven.
  If CRWR's star is right, the cap description is a description of *the*
  extremal graph — a nice remark, publishable as a note/observation at most.
- The description via RM(1,4) codewords / caps in PG(3,2): **not found** in
  the Zarankiewicz exact-values literature (searched: Tan, CRWR, DGH, CHM
  full texts; web searches "Reed–Muller Zarankiewicz", "cap K_{3,3}",
  "affine hyperplanes bipartite Turán"). Nearest published relatives:
  Brown 1966 (sphere incidences), Kollár–Rónyai–Szabó/Alon–Rónyai–Szabó norm
  graphs (algebraic hypersurface incidences), Bukh 2024 (random algebraic).
  None is the F₂-hyperplane-side family. **[UNCERTAIN — absence of evidence
  after a genuine search, not proof of absence.]**
- What would make the bridge a real contribution: an infinite K_{s,t}-free
  family from parity-check/cap conditions with *asymptotically competitive*
  density, or new exact cells. A finite observation matching one known cell
  (16,16) is not enough; as stated by the coordinator the mechanism
  ("coverage of an s-set is 0 or 2^{k−rank}") is sound but currently only
  re-derives a known point.

## N6. Packing-number framing (coordinator's Theorem 3 etc.)

- "z-table ⟺ multi-fold packing numbers" is **published**: Roman 1975
  (equality via coverings), Guy 1969 (T₂,₂ formula), Tan 2022 §2.1
  (T_{a,b}(m) definition, Table 1 of values incl. **T₃,₃(7)=15** — the
  coordinator's D₂(7,4,3)=15 is a rediscovery of that table entry, though a
  valuable independent confirmation), Hanani 1960 (perfect cases),
  Bao–Ji 2014 (λ=1 exact), DGH 2024 (the mixed-block-size profile LP — the
  coordinator's "mixed-block-size" framing and pair-local/row-local budget
  constraints are, in substance, DGH's Theorem 1.1 constraint family; their
  v=s−1 constraints are "by far the most useful", and their LP already
  computes what a level-1/level-2 budget hierarchy would).
- Genuinely open nearby: (a) a formula or full determination of
  T₃,₃(m)=D₂(m,4,3) for all m with proofs (Tan: "no corresponding results in
  the literature"; only m ≤ 18 values exist, by his Gurobi runs — a clean
  design-theory mini-paper if done right); (b) using packings to prove
  *below-window* exact values for s=3 (the CHM-for-s=3 program, cf. N1);
  (c) the coordinator's z(7,20)=75 rederivation: the value is published
  exact (Tan bold); Roman's printed bound there is 76, so a human-readable
  proof of ≤75 is *sharper than Roman at that cell* — check whether DGH's
  LP already yields 75 before claiming (their tables only list cells where
  they improve on best-known, and (7,20) was already exact, so they are
  silent; rerunning their LP is the test).

### N6a. Referee verdict on the T₃,₃ "law + δ-conjecture" (2026-07-28 request)

- **The erratum was OURS, not Tan's.** Tan's Table 1 prints T₃,₃(18) = 405
  (and T₃,₃(17) = 340); the 408 in an earlier `published_data.py` was this
  agent's transcription-completion error, now fixed. Tan's paper is clean:
  405 respects both the design-inadmissibility bound (≤ 407) and the
  sharper per-point Johnson bound (≤ 405), and his printed cyclic
  construction generates exactly 405 valid blocks (verified; C16d).
- **The "S_m(0) law" (perfect ⟺ admissible) is correct but classical**:
  admissibility ⇒ existence of a 3-(m,4,2) design is Hanani's block-size-4
  spectrum theorem; the converse is double counting. Citable, not new.
  Verified numerically for all m = 4..18 (C16c).
- **The δ-conjecture (T = ⌊C(m,3)/2⌋ − 2 for all inadmissible m ≥ 7) is
  REFUTED at its first untested case**: it predicts T₃,₃(18) = 406, but
  T₃,₃(18) = 405. The δ=2 pattern at m = 9, 12, 15 was a small-range
  coincidence: for m ≡ 0 (mod 3) the binding constraint is the per-point
  Johnson bound J(m) = ⌊m·⌊(m−1)(m−2)/3⌋/4⌋, and the defect
  ⌊C(m,3)/2⌋ − J(m) grows like m/6 (values 1,2,2,2,3 at m=6,9,12,15,18).
- **The corrected candidate law** (fits ALL of m = 3..18, C16b):

      T₃,₃(m) = J(m) − 2·[m ≡ 3 (mod 4) and m ≢ 0 (mod 3)]

  i.e. Johnson-tight everywhere except the b-parity-obstructed residue
  class, where the deficit is 2. Same genre as Bao–Ji's proven λ=1 formula
  (Johnson expression minus an indicator correction). Since no λ=2
  determination of D₂(v,4,3) was found in print (two targeted searches +
  Tan's own statement), **this corrected law is a genuinely new, falsifiable
  design-theory conjecture**; first open cases: m=19 (predicts J(19)−2 =
  482), m=21 (predicts J(21) = 661), m=22 admissible (predicts 770).
  Proving the −2 (leave-structure argument for m ≡ 3 mod 4, m ≢ 0 mod 3)
  plus Johnson-tight constructions for m ≡ 0 (mod 3) would be a complete
  determination — the real prize. Cross-check any construction claims at
  m=19 against nothing — there is nothing published to collide with.

## N7. Row-6 Turán bridge

T₃,₃(6)=9=ex(6;K₃) with the complement bijection: both numbers verified
(C13a,b) and the bijection is sound. The *number* is published (Tan Table 1);
the Turán explanation is not spelled out in any source I could read, but it
is a two-line observation on 6 points — treat as expository lemma/folklore
risk, not a headline. [UNCERTAIN on prior appearance: Guy 1967/1969 and
Čulík 1956 full texts not accessible to me — Guy's 1967 scan
(https://oeis.org/A001197/a001197.pdf) exists but was not text-searchable
here; someone with library access should check Guy 1969 §on k₃ before any
write-up claims it.]

## N8. Evaluator-suite hygiene (owner-facing)

- (11,22)=121 is proven exact (BNL) but absent from the suite — deliberate?
- The two `_EXTRA_EXACT` cells' witnesses are in the BNL paper, not in this
  repo; the current generalized run has never re-attained them (best valid:
  108/116 and 120/132 — C14e). For self-containedness, the witnesses (or
  their generators) should be checked into `gpt_agent/data/` as certificates.
- The excluded upper-bound cells: for (13,17) etc., the *best* published UB
  is CRWR's, not Tan's printed Roman value (N4) — relevant if the owner ever
  extends the suite.

## N10. Referee verdict: "Theorem 9" design-regime corollary via Keevash (final round)

Question asked: is z(m,n;s,t) = sn + min((t−1)C(m,s)/(s+1), ⌊(B−n)/s⌋) for
all n ≥ (t−1)C(m,s)/(s+1), large admissible m, via Keevash / GKLO designs,
new?

- **The design→exact-Zarankiewicz bridge is PUBLISHED**: Damásdi–Héger–
  Szőnyi 2013, Prop. 3.25 — for admissible (t,v,k,λ) and an explicit window
  0 ≤ c ≤ c₀: Z_{t,λ+1}(v−c,b) ≤ r(v−c) with **equality whenever a
  t-(v,k,λ) design exists** — stated for GENERAL (t,λ+1), i.e. general
  (s,t) in our notation, with k free (k=s+1 is the coordinator's case).
  Their Cor. 3.16 gives the companion near-design windows. For s=2, CHM
  2024 go much further (almost all n = Θ(tm²)) and their constructions use
  modern decomposition machinery — **they cite Keevash (The existence of
  designs, arXiv:1401.3665) as [13] and Glock–Kühn–Lo–Montgomery–Osthus
  [8]**. Also: bound attained at an integral Roman point iff an
  s-(m,k,t−1) design exists — Reiman 1968 for s=2 (CHM's [18]); general-s
  version stated (uncredited) in DGH §2.
- **What is genuinely absent from print**: the s ≥ 3/s ≥ 4 instantiation
  with Keevash-supplied s-(m,s+1,t−1) designs (all large admissible m).
  DGH never cite Keevash; DHS predate him (2013 vs 2014); CHM use him only
  for s=2. Verdict: the corollary is **new but a modest synthesis** —
  DHS Prop 3.25 (k=s+1) + Tan-Thm-2.2-window + Keevash. Write it as
  exactly that, citing DHS Prop 3.25 as the bridge; the honest headline is
  "Keevash closes the design-existence hypothesis for s ≥ 4", not a new
  mechanism. CAUTION on slices: DHS 3.25 varies m at fixed n=b; the
  coordinator's window varies n at fixed m — the n-interpolation (mixed
  block sizes s and s+1 between the k=s+1 Roman point and the Čulík point)
  is the part attributed by Tan to Roman; see the Roman-scope caveat in
  theory.md §3. Get Roman's paper before submission.
- **Roman's original scope [UNRESOLVED]**: Tan attributes the general
  T-window equality to Roman; DHS/CHM/DGH consistently cite Roman for the
  bound only (CHM: "[Roman, Theorem 1]"), credit Čulík for the threshold
  equality and Reiman 1968 for the s=2 design characterization, and prove
  their own design-regime equalities. Primary text paywalled
  (Zbl 0296.05014; review text license-blocked; ScienceDirect 403). If
  Roman's paper contains the window, the corollary's Zarankiewicz side is
  entirely his; if not, the window proof for s ≥ 3 may itself be original-
  but-elementary. Either way the *validity* is unaffected (C6b verifies the
  (3,3) window on all 71 in-window cells).

## N11. Referee verdict: "Lemma C" mixed-value Johnson bound (final round)

The per-point deficiency-accounting technique in mixed-block-size
(s,t−1)-linear hypergraphs is **published**: CHM 2024 introduce edge
"deficiency" (max{7−|E|,0} in their worked example) with per-vertex
deficiency accounting, and their Lemma 2.4 is the s=2, v=1 constraint
family; DGH 2024 generalize with Definition 3.1 ((k,s,v)-deficiency) and
Theorem 1.1 — a linear constraint for every v < s ≤ k ≤ m with the exact
remainder α ≡ (t−1)C(m−v,s−v) (mod C(k−v,s−v)), which for v=1 uses
precisely the per-point count Σ_{B∋x} C(|B|−1,s−1) ≤ (t−1)C(m−1,s−1)
that Lemma C starts from. Lemma C's specific linearization
(s(w−s) ≤ C(w−1,s−1), then per-point floor, then divide by s+1) is not
stated verbatim anywhere I searched, and its s=2 failure shows it is not
boilerplate — but it is at best a closed-form corollary in the same
technique family. **RESOLVED by computation** (`implication_checks.md`,
Check 2; exact rational LP): the DGH v=1 (and v=1+v=2) constraint LP does
NOT imply Lemma C — Lemma C is strictly stronger by exactly 1 at every
interior cell of the row-9 band (n=40..45) and the row-12 band
(n=108..113), and ties at the band edges and everywhere on row 6. The
delta is Lemma C's per-point integer floor, which DGH's α-arithmetic does
not capture. Honest presentation: "DGH's v=1 constraint plus a per-point
integrality floor", citing CHM Lem. 2.4 and DGH Thm 1.1 for the
technique; the named +1 cells are the claim. Caveat: rows 7/11 need more
(the T = J−2 leave argument); neither Lemma C nor the DGH LP reaches the
T-bound there (e.g. 76 vs 75 at (7,20)).

Companion Check-1 verdicts (Guy 3.15 + DHS 3.20 fixpoint, see
`implication_checks.md`): (7,20)≤75 reachable from Tan's exact (7,19);
(6,10)≤39 fully classical (Guy step from Roman-exact (5,10)=33 — the d=1
defect there is NOT new); (7,24)≤87 and (8,27)≤108 and (9,46..48) are
just Roman; the genuinely-beyond-machinery band cells are (8,24..26)
[short by 1,2,1] and (9,40..45) [short by 1 each].

## N9. What is definitively NOT novel (do not claim)

1. Čulík-regime constructions/formulas (Čulík 1956; 41 cells). C5.
2. Rows 3–5 closed forms; row-6 tail from n=13 (Roman 1975 / Tan Thm 2.2;
   71 cells). C6.
3. "Counting/waterfill upper bound" — identical to min_p Roman here (C7d);
   its LP strengthening is DGH 2024.
4. The projective-plane (2,2) equality at n=q²+q+1 (Reiman 1958 — note the
   corrected attribution; Füredi's q>13 theorem is the non-bipartite C₄
   problem).
5. z(16,16;3,3)=128 and its witness (CRWR 2016, Tan 2022); the champion's
   matrix is that witness (C11b).
6. T₃,₃ values for m ≤ 18 (Tan Table 1), incl. T₃,₃(7)=15.
7. LLM-evolutionary search finding Zarankiewicz constructions, incl. cells
   (11,21), (11,22), (12,22) (BNL 2026 — the team's own paper).
8. New exact z(n,n;2,2) values for n ≤ 31 (Guy; Afzaly–McKay via CRWR).

# Deficit analysis: d(m,n) = ub_waterfill − z on the proven (3,3) table

Author: bounds prover agent, 2026-07-28.
Data sources (all regenerable):

- `python3 upper_bounds.py --selftest` — proves the machinery (waterfill greedy
  verified against exhaustive profile enumeration, 5,644 parameter cases;
  bound never below any of the 161 known z; q-local conditions never reject a
  true value).
- `python3 upper_bounds.py --tables` → `ub_33.csv` (m=3..20, n=m..40).
- `python3 upper_bounds.py --refine` → q-local refinement per deficit cell.
- `python3 exact_small.py --extended` → `exact_small.csv` (brute-force ground
  truth, incl. every (3,3) cell with m ≤ 8, n ≤ 10 area and more).
- `python3 supply_law.py --gmax 8` → supply numbers g4(m) + SUPPLY LAW test.

Definition: for the 161 proven-exact cells, d(m,n) = ub_waterfill(m,n;3,3) − z(m,n;3,3),
where ub_waterfill is the exact integer optimum of the triple-budget
relaxation (min over both orientations). d ≥ 0 always (it is a PROVEN bound).

## 1. Headline numbers  [VERIFIED-NUMERICALLY on all 161 cells]

- **78 / 161 cells have d = 0**: the published exact value is completely
  explained by counting + integrality (the bound IS the answer, and any
  construction meeting the waterfill profile is optimal).
- Deficit spectrum over the 83 cells with d > 0:

  | d | 1 | 2 | 3 | 4 | 5 | 8 |
  |---|---|---|---|---|---|---|
  | #cells | 28 | 24 | 15 | 10 | 5 | 1 |

  Total deficit mass: 194. (Independently matches the coordinator's
  quick_scan.csv cell for cell.)
- The **column side** (budget on the smaller side m) is the binding side of
  the waterfill min at every m < n cell; the row side never bites off the
  diagonal. The deficit is a phenomenon of the m-row triple budget.

## 2. Full slice map (d(n) per row; "." = 0)

```
m= 3 (thr=  2): 0 everywhere (n=3..23)
m= 4 (thr=  8): 0 everywhere (n=4..23)
m= 5 (thr= 20): 0 everywhere (n=5..23)
m= 6 (thr= 40): . 1 1 . 1 . . . . . . . . . . . . .        (n=6..23)
m= 7 (thr= 70): 2 1 1 1 1 1 2 2 1 2 2 2 2 1 1 1 .          (n=7..23)
m= 8 (thr=112): 1 2 1 2 1 2 1 1 2 1 1 1 1 1 2 1            (n=8..23)
m= 9 (thr=168): 3 2 1 . 1 2 3 3 3 3 2 1 2 1                (n=9..22)
m=10 (thr=240): 2 3 4 3 3 3 3 2 2 2 2                      (n=10..20)
m=11 (thr=330): 4 4 2 3 4 4 5 4 [n=21: 1]                  (n=11..18,21)
m=12 (thr=440): 4 4 3 3 5 [n=22: 0]                        (n=12..16,22)
m=13 (thr=572): 4 3 3 5                                    (n=13..16)
m=14 (thr=728): 4 2 5                                      (n=14..16)
m=15 (thr=910): 2 5                                        (n=15..16)
m=16          : 8                                          (n=16)
```
(thr = Culík threshold 2·C(m,3); the published exact table never reaches it
for m ≥ 6.)

Where d first becomes positive: never (m ≤ 5); n=7 for m=6; n=m for every
m ≥ 7. Where it returns to 0: n=9 (and 6, 11+) for m=6; n=23 for m=7;
n=12 (sporadic) for m=9; n=22 (sporadic) for m=12; not inside the table for
the other rows.

## 3. Structural findings

**(a) d = 0 for all m ≤ 5 — PROVEN** (coordinator Theorem 1: waterfill
profile realized by point-complement + triple + pad blocks). 60 cells.

**(b) d does NOT persist until the Culík threshold.** It vanishes far below
it: row 6 is deficit-free from n=11 on though the threshold is 40; row 7
closes at n=23 against threshold 70. Mechanism (see §5): as n grows the
waterfill's demand for heavy (weight-4) columns falls below the supply
g4(m) that the 2-fold triple packing can actually deliver, long before
Culík's all-weight-3 regime begins.

**(c) The sporadic interior zeros are design-existence events.**
- (9,12): waterfill profile (6⁴ 5⁸), budget 160/168 used — realizable, z=64.
- (12,22): waterfill profile 6²², budget 440/440 used EXACTLY — realizability
  is equivalent to a 2-fold 3-design 3-(12,6,2) (every row-triple in exactly
  2 of 22 six-row blocks; divisibility all-integer: b=22, r=11, λ₂=5). The
  published z(12,22)=132 = WF says such a configuration exists.
  [Design identification CONJECTURE — flagged to theory agent; the equality
  z=WF itself is VERIFIED.]
- Contrast (16,16): WF=136 would need profile (9⁸ 8⁸) with budget 1120/1120
  EXACT — a 2-fold 3-design with mixed block sizes {8,9} on 16 points. It
  does not exist; the truth z=128=16·8 is the doubled-cap/ovoid structure
  (coordinator Observation 3), leaving the single largest deficit d=8.

**(d) No congruence law.** Row 8's pattern 1,2,1,2,1,2,1,1,2,1,1,1,1,1,2,1
(d=2 at n=9,11,13,16,22) is aperiodic in every modulus we tested (n mod 2,
3, 4, 6; m+n, m−n residues). Any "d depends on residues" conjecture is
already falsified inside the table.

**(e) Where integrality alone first fails, by family** [brute-force PROVEN,
`exact_small.csv`]:
- (s,t)=(3,3): first failure at **(6,7)**: z=29 < 30=WF.
- (s,t)=(2,2): first DIAGONAL failure at **(8,8)**: z=24 < 25=WF (every
  searched cell before it — all m ≤ n ≤ 6, (7,7), (7,8) — is tight;
  (7,7)=21 is the Fano plane meeting the bound).
- (s,t)=(2,3): first failure at **(6,6)**: z=21 < 22=WF. Here WF's profile
  (4⁴3²) uses the pair budget 2·C(6,2)=30 exactly, i.e. demands a 2-fold
  pair design with blocks (4⁴3²) on 6 points — nonexistent. Same
  non-monotonicity as (3,3): (6,7;2,3)=24 and (7,7;2,3)=28 are tight again.
- (3,4) and (4,4): no failure in the searched box (m,n ≤ 6).

**(f) The twofold-triple-system question** (task prompt asked to check the
condition "m ≡ 0,2 mod 3"): the correct existence condition for a TTS(v)
(λ=2, block size 3, every PAIR twice) is **v ≡ 0 or 1 (mod 3)**, v ≥ 3
(divisibility: b = v(v−1)/3 ∈ Z; sufficiency Hanani/Bose classical). But the
TTS is NOT the object governing (3,3) deficits: our columns 2-fold-pack
TRIPLES (partial 3-(m,k,2) designs), not pairs, and weight-3 pad columns are
unconstrained singleton triples (repeats ≤ 2 allowed), so no TTS-style
congruence obstruction exists — which is exactly why the elongated regime is
deficit-free for every m and no mod-3 fingerprint appears in the table.

**(g) The published table's unproven upper bounds are exactly WF.** On all
45 cells beyond `_EXACT_UP_TO` (m=9..16), the published upper-bound figure
equals ub_waterfill exactly — the literature's bound in this range carries
no information beyond the counting bound. [VERIFIED-NUMERICALLY]

## 4. q-local refinement: how much of d is profile-visible?

`ub_profile` adds, on top of the global budget, every "q-local" necessary
condition on the degree profile (busiest q-subset of rows, q < s, must fit
its localized budget; see upper_bounds.py docstrings — PROVEN necessary,
soundness tested against all 161 cells). Result of `--refine` over the 83
deficit cells:

- improves **16 / 83 cells, by exactly 1 each** (16 of 194 deficit units):
  (6,10) [fully closed, the Turán cell], (7,7), (7,18), (10,12), (10,13),
  (11,14), (11,15), (11,17), (12,13), (13,15), (13,16), (14,14), (14,16),
  (15,15), (15,16), (16,16).
- **67 / 83 cells: no improvement at all.**
- Per-level decomposition (hierarchy story):
  * level 1, ROW-local budget alone (q=1): closes 1 unit at 6 cells —
    (6,10), (7,18), (10,12), (11,17), (12,13), (13,15). These are the
    "demand overflow" cells where the WF profile forces more heavy columns
    through the busiest row than its localized budget allows (e.g. (6,10):
    profile 4¹⁰ ⇒ some row meets ⌈40/6⌉=7 weight-4 columns, costing
    7·C(3,2)=21 > 2·C(5,2)=20).
  * level 2, PAIR-local budget beyond level 1 (q=2): closes 1 further unit
    at 10 different cells — (7,7), (10,13), (11,14), (11,15), (13,16),
    (14,14), (14,16), (15,15), (15,16), (16,16) — concentrated on the
    diagonal and the n=16 exactness frontier, where profiles are
    budget-saturated and the busiest pair's C(c−2,1) load overflows.
  * The two levels are DISJOINT in effect (no cell improved by both), and
    no cell ever improves by more than 1 total: after one step down, a
    locally-legal profile always exists.

Honest conclusion: the deficit is almost entirely INVISIBLE at the degree-
profile level — profiles satisfying every local budget exist right up to
WF−0 or WF−1, but no actual 0/1 matrix realizes them. The obstruction is
design-realizability (which block multisets can coexist), not degree
counting. Coordinator's hand proofs for (6,7)/(6,8) show the flavor of what
IS needed: distinctness/multiset arguments about the heavy-block hypergraph.

## 5. THE SUPPLY LAW (the closed form on row tails)  [central finding]

Define the **weight-4 supply** g4(m) = maximum multiset of 4-subsets of the
m rows such that every 3-subset lies in ≤ 2 of them. Computed EXACTLY by
complete branch-and-bound (`supply_law.py`):

| m | g4(m) | identity |
|---|-------|----------|
| 6 | **9** | = ex(6,K₃) (Turán; complement-of-edge correspondence, coordinator Thm 2) |
| 7 | **15** | strictly between doubled-Fano (14) and the pair-count bound (17); witness: 15 distinct 4-blocks covering 30 of 35 triples exactly twice |
| 8 | **28** | = pair-count bound m(m−1)(m−2)/12; witness: the doubled Steiner quadruple system 2×SQS(8), every triple exactly twice |
| 9 | **40** | [ILP-PROVEN, supply_table.py] two below the pair bound 42 (a 3-(9,4,2) design fails divisibility, ruling out 42; the ILP also rules out 41) |

**SUPPLY LAW.** Whenever the waterfill profile at (m,n) contains no column
of weight ≥ 5 (the "tail" of row m; onset n=9, 17, 27 for m=6, 7, 8):

    z(m,n;3,3) = 3n + min( k4wf(m,n), g4(m) ),
    i.e.   d(m,n) = max( 0, k4wf(m,n) − g4(m) ),

where k4wf = the number of weight-4 columns in the waterfill profile
(= min(n, ⌊(2C(m,3)−n)/3⌋)).

- LOWER BOUND — PROVEN: any min(k4wf, g4)-subfamily of the g4-max family,
  padded with weight-3 columns on residual triple capacity (pads always fit:
  total residual 2C(m,3) − 4k ≥ #pads, each unit of residual accepts one
  pad).
- UPPER BOUND — z ≤ WF = 3n + k4wf is PROVEN unconditionally; z ≤ 3n + g4
  is PROVEN **for optima with no weight-≥5 column** (at most g4 columns can
  have weight 4, rest ≤ 3). The remaining gap — that weight-≥5 columns never
  help in this regime — is the conjectural part for m ≥ 7 (for m = 6 it is
  closed by coordinator Thm 2 + our exhaustive search).
- STATUS: **VERIFIED on all 22 applicable known cells (22 PASS, 0 FAIL)** —
  the entire tails of rows 6 (n=9..23) and 7 (n=17..23), reproducing every
  deficit there, including the two-step pattern d=2,2,2,1,1,1,0 of row 7.

**Falsifiable predictions** (explicit, beyond the published table):
1. z(7,n) = 3n + min(⌊(70−n)/3⌋, 15) for ALL n ≥ 17; in particular
   z(7,24)=87, z(7,25)=90, ..., d(7,n)=0 for every n ≥ 23.
2. z(8,n) = 3n + min(n, ⌊(112−n)/3⌋, 28) for ALL n ≥ 27, with d(8,n)=0
   throughout (because g4(8)=28 achieves the pair-count bound, supply never
   falls short: the row-8 deficit dies at exactly n=27, far below the Culík
   threshold 112). First checkable new value: z(8,27)=108, z(8,28)=112
   (= 4·28, the doubled SQS(8) itself as the incidence matrix).
3. General shape: for every m, d(m,·) ends at n* = smallest n with
   ⌊(2C(m,3)−n)/3⌋ ≤ g4(m) in the no-5 regime — the deficit's tail length is
   controlled by a single hypergraph-Turán number.

These are stated as CONJECTURE outside the verified range (they rest on
"weight-≥5 never helps in the no-5-WF regime"). Any construction beating
3n + g4(m) there, or any exhaustion below it, falsifies the law.

**What is still missing for a full closed form**: the mid-range (profiles
with weight-5..⌈m/2⌉ columns) needs the mixed supplies g_{w,w′}(m) (max
total weight of a legal family with blocks of sizes w, w′) — the same
machinery applies (small complete searches per m), but the supply is now a
two-parameter object. Row 8's aperiodic 1,2-oscillation lives exactly there.
This is the concrete next computation for closing rows 7..15.

## 5b. Lemma A + the mixed supply table S_m(k5,k6)  [coordinator integration]

`supply_table.py` (run with the scratchpad venv python for scipy.milp):

- **Lemma A machine-verified**: exhaustive integer sweep over 102,452 legal
  profiles (m ≤ 13, k4..k7 in range): zero violations of
  Q ≤ (B−n)/3 − k5 − (10/3)k6 − (22/3)k7. The generalized per-weight penalty
  c_w = (C(w,3)−1−3(w−3))/3 is nondecreasing in w, so the k7 term also
  covers all weights ≥ 7. PROVEN (pure arithmetic + sweep).
- **S_m(k5,k6) tabulated by exact ILP** (HiGHS; `supply_table.csv`;
  UNRESOLVED = time-limited, bracketed above by monotonicity, never used
  as proven). Cross-checks: S_m(0,0) = g4(m) reproduced for m=6,7,8
  (ILP vs my complete search), S_6/S_7(1,0) ILP = dedicated DFS.
  Headline values: S_7(1,0)=12 (proves the coordinator's ≤12 sharp),
  S_8(1,0)=23 (their (8,24) pentad witness is optimal),
  **g4(9)=S_9(0,0)=40**, **g4(10)=S_10(0,0)=60 = pair bound = 2·|SQS(10)|**.
- **Row closure** (`--closure`; bound = min(WF, 3n + max over (k5,k6) of
  min(Lemma-A ceiling, supply-capped count), with sound fallbacks for
  k5>12, k6>2, weight≥7):
  * row 6: **18/18 cells EXACT** — the whole row is an analytic theorem;
  * row 7: **16/17 EXACT** — every cell except the diagonal (7,7)
    (bound 34, truth 33: the deficit's second unit is not block-profile
    expressible; consistent with q-local closing only 1 there);
  * rows 8–10 mid-table: open (1 + 1 cells closed) — these n sit BELOW the
    Roman window (n < ~B/4) where weight-5/6 blocks are structural and the
    binding object is the waterfill level profile, not the quad supply;
  * beyond-table checks: (7,20)=75, (7,24)=87, (8,27)=108, (8,28)=112
    reproduced analytically; (8,24): analytic UB 98 vs coordinator's
    ILP-exact 97 — the 1-unit gap hinges precisely on the UNRESOLVED
    S_8(3), S_8(4) ∈ [14, 21] (a symmetry-broken complete search would
    settle them; next step).
  Compared to the q-local profile refinement (§4: 16 units, 1 full cell),
  Lemma A + supply closes 19 deficit cells outright (~26 units) — the
  block-supply formulation strictly dominates degree-profile reasoning.

## 6. New machine-proven ground truth produced along the way

Complete-search PROVEN (witnesses independently re-verified with the
evaluator's own has_kst; all match the published table, 0 mismatches):
all (3,3) cells with m,n ≤ 6, plus (6,7..14), (7,7..14), (8,8..10), (9,9),
(9,10). This includes exhaustive confirmation of the coordinator's Theorem 2
upper bounds (6,7)≤29, (6,8)≤32, (6,10)≤39 and the priority cell
**(7,7): z=33, d=2** — both refuted targets (35, 34) exhausted.

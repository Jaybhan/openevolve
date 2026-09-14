# DIAGONAL HUNTER — live status (updated)

Mission: z(17,17;3,3) exact; then (18,18), (19,19); band cells rows 13-17.
Every LB witness verified by `verify_diag.py` (my own verifier, code-disjoint
from encoder and sat_attack's verifier: column-triple + row-triple bitmask).

## Headline: z(17,17) ∈ [138, 144]

- LB 138: VERIFIED witness `witnesses/ls17_138.json` (local search;
  profile 9^6 8^8 7^2 6 both orientations). 136 = circulant
  {0,1,2,3,5,6,11,13} ⊂ Z_17 and 137 = LS + also structured
  (cap16 + point, ILP-optimal extension, `witnesses/ext16_137.json`).
- UB 144: **deletion averaging** — every (r,c)-minor is 16x16 K33-free:
  sum over 289 cells of (E - d_r - w_c + A_rc) = 256 E <= 289*z(16,16)
  = 289*128 => E <= 144.5. PROVEN (elementary, from published z(16,16)=128).
  Method validates on knowns: gives z(15,15)<=120 (TIGHT), z(13,13)<=93,
  z(14,14)<=106 (slack 1 each), z(16,16)<=136 (slack 8 = the cap cliff).
- The 8-regular plateau hypothesis (z=136) is REFUTED.
- Profile arithmetic (`profiles_diag.py`): E=151 has NO surviving profile
  (independent re-proof of <=150); E=139 surviving max-weights {9..16}.

## Ladder cells (feed the UB + secondary deliverable)

| cell | LB (verified) | UB | note |
|---|---|---|---|
| (16,17) | 132 (`ls16x17_132.json`) | 136 = ⌊17*128/16⌋ | cap16+col ILP also = 132 exactly; UNSAT@133 => z(17,17) <= 140 |
| (15,17) | 126 (`ls15x17_126.json`) | 130 = ⌊17*123/16⌋ | LS stall @127 |
| (17,17) | 138 | 144 | crux SAT @139 running (flat + h<=9 cube) |
| (18,18) | 150 (`ls18_150.json`) | ⌊324*z17/289⌋ = 154..161 | LS stall @151 cost 6 |
| (19,19) | 164 (`ls19_164.json`) | via z18 | LS stall @165; circulant 152 verified |

## Candidate diagonal law (post-plateau quadratic, FALSIFIABLE)

    z(m,m) = 8m + (m-15)(m-16)   [= m^2 - 23m + 240],  15 <= m <= 19?
    equivalently: increments z(m)-z(m-1) = 2m - 24 for m = 16..19
                  (8, 10, 12, 14 — second difference exactly 2)

Matches EXACT 120 (m=15), 128 (m=16); matches the LS frontier 138, 150, 164
at m = 17, 18, 19, where LS stalls at exactly +1 in all three sizes
(and the off-diagonal LS frontier is consistent: (17,18)=143?, (18,19)=156?).
Status: CONJECTURE — decided at m=17 by the running SAT instances.
- Fails below the plateau: m=14 would give 114, truth 105 (regime boundary
  at the 8-regular cliff m=15,16 where (m-15)(m-16)=0).
- Counting consistency: the law's average degree w=z/m satisfies the KST
  slot bound w(w-1)(w-2) <= 2(m-1)(m-2) only up to m=24 (=264=11*24, tight-
  ish 990<=1012) and VIOLATES it at m=25 — the law must break by m=25.
- Sharp falsifiable predictions: z(20,20)=180 (=9*20 — a 9-regular
  incidence!), z(21,21)=198.

## LS frontier fingerprints (all witnesses verified)

| witness | pair-lambda dist | col-int dist | triples at cap |
|---|---|---|---|
| circulant 136 | {3:68, 4:68} | {3:68, 4:68} | 340/680 |
| ls17_138 | {2:11,3:28,4:94,5:3} | same | 434/680 |
| ls18_150 | {2:6,3:48,4:96,5:3} | same | 462/816 |
| ls19_164 | {2:6,3:48,4:111,5:6} | same | 552/969 |

Every row-pair co-occurs >= 2 times in all three diagonal LS optima; max
column-intersection 5; the pair-lambda and column-intersection
distributions COINCIDE (near-self-dual structures).

## Endgame cascade (as instances close)

1. UNSAT(16,17)@133 => z(16,17)=132 => z(17,17)<=140.
2. UNSAT(17,17)@139 => **z(17,17)=138** (LB verified; monotone closes 139+).
3. UNSAT(17,18)@144 => z(17,18)=143 => z(18,18)<=151.
4. UNSAT(18,18)@151 => z(18,18)=150.
5. UNSAT(18,19)@157 => z(18,19)=156 => z(19,19)<=floor(19*156/18)=164
   => **z(19,19)=164 with no direct 19x19 UNSAT needed**.

## Circulant landscape (exhaustive, up to shift/negation/multiplier)

- v=17: exactly 2 weight-8 classes ({0,1,2,3,5,6,11,13}, {0,1,2,4,10,12,13,14});
  NO weight-9 (slot budget 1428>1360).
- v=18: 6 weight-8 classes; **no weight-9 base** (search: 0 survivors,
  despite slot budget 1512<=1632 allowing it).
- v=19: 21 weight-8 classes; **no weight-9 base** (0 survivors, budget
  1596<=1938 allows). So circulant ceiling stays 8-regular at 18,19:
  LS witnesses (150,164) strictly beat every circulant.

## Calibration — pipeline TRUSTED at 13 AND 14

- z(13,13)=92 reproduced two-sided: SAT@92 29.7s; **UNSAT@93 1811.6s**
  [cadical195; encodings_zar matrix encoding; col-lex + row-degmono;
  totalizer equals-cardinality]. Plain encoding couldn't UNSAT@93 in 10min.
- z(14,14)=105 reproduced two-sided: SAT@105 9.4s; **UNSAT@106 11.7s**
  [same solver+encoding]. Both calibration targets: PASS.
- New tool `run_plus.py`: encoding + PROVEN deletion-ladder clauses
  (d_r + w_c - x_rc >= E - z(m-1,n-1); d_r >= E - z(m-1,n);
  w_c >= E - z(m,n-1)) over fresh two-directional unary counters —
  knowledge cadical cannot derive; soundness argued in file docstring.

## LS stall calibration (honesty check on stalls)

Known-SAT structured cells STALL under LS from scratch: (15,15)@120 cost 16,
(16,16)@128 cost 2 — LS stalls are weak evidence. (Seeded chains do better.)
Hence SAT is the only decider at 139.

## Process ledger (mine; cap ~4)

- d17_139 flat (8h budget), d17_139_h9 cube wwin[2,9] (8h)
- cal14_105, cal14_106 (4h each)
- ub_ladder@150 about to hit its 90min budget and stop (by design;
  E<=150 already proven arithmetically, so no restart planned)

## Next actions

1. When cal14_105 returns (fast): launch (16,17)@133 [decides UB 140].
2. If h9 cube UNSAT: launch h=10, h=11 cubes; tail h=12..16 sequential.
3. If any 139 SAT: climb to 140 and re-run averaging for 18/19 targets.
4. (17,18) LS for the 18-ladder.

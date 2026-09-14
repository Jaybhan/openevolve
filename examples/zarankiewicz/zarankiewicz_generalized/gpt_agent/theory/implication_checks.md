# implication_checks.md — the two implication checks (final gate)

Requested by the coordinator 2026-07-28. Script: `implication_checks.py`
(deterministic; exact rational LP; rerun with `python3 implication_checks.py`).
Machinery tested, with verbatim sources in the script header:
- Guy's point-C deletion = DHS 2013 Thm 3.15 (both directions, all anchor
  jumps), DHS 2013 Prop 3.20 (U-recursion, alpha in {1,2}, both
  orientations), iterated to a fixpoint over m <= 9, n <= 56, seeded with
  Roman/Culik/DGH bounds plus literature-exact anchors (regime A). Target
  cells are never seeded with exact values or claims.
- DGH 2024 Thm 1.1 v=1 (and v=2) constraint families with their exact
  remainder alpha, added to Roman's counting LP; compared against Lemma C
  (z <= 3n + J(m), J(m) = floor(m*floor(2C(m-1,2)/3)/4)) at the requested
  sites. LP solved exactly over the rationals; reported values are floors.

## One-line verdicts

CHECK 1 (is Theorem 8's band content derivable from published machinery?):
- (7,20) <= 75: YES — Guy 3.15 from the exact anchor (7,19)=72
  (72 + floor(72/19) = 75). Without computational anchors (regime B): 76.
  Verdict: the VALUE is machinery-reachable given Tan's exact (7,19); an
  anchor-free proof of 75 (the coordinator's packing argument) is not.
- (7,24) <= 87: YES — this is Roman p=3 outright. Not new as a UB.
- (6,10) <= 39: YES, and fully classical — Guy 3.15 row-step from
  (5,10)=33, which is Roman-window exact (closed form, no computation):
  33 + floor(33/5) = 39. The d=1 defect at (6,10) is NOT new content.
- (8,24..26) <= 97,100,104: NO — machinery reaches only 98,102,105
  (short by 1,2,1). Genuine content IF the pending S9-slice proofs hold.
  (8,27) <= 108: YES — equals Roman p=4. Not new.
- (9,40..45) <= 3n+40: NO — machinery reaches 161,164,167,170,173,176
  (short by 1 at each; note 3.20 does beat Roman by 1 at (9,42)).
  Genuine content. (9,46..48): YES — equals Roman there. Not new.

CHECK 2 (do DGH v=1 constraints imply Lemma C?): NO in the band interiors.
- Row 6 (n=9..13): DGH v=1 LP == Lemma C == 3n+9 everywhere (at n=10 both
  give 39, beating Roman's 40). Tie — repackaging there.
- Row 9 (n=40..45) and row 12 (n=108..113): Lemma C is STRICTLY stronger
  by exactly 1 than the DGH v=1 (and v=1+v=2) LP at every interior cell;
  they tie at the band edges (n=46..48, n=114..116). The delta is Lemma
  C's per-point integer floor (sum over blocks at a point of (|B|-3) is an
  integer <= floor(2C(m-1,2)/3)), which the DGH constraint family's
  alpha-arithmetic does not capture. So: Lemma C = "DGH v=1 plus one
  genuine extra integrality floor"; name the +1, cite CHM Lem 2.4 / DGH
  Thm 1.1 for the technique.
- Rows 7/11 (J = T+2): NEITHER Lemma C nor DGH v=1 reaches the T-bound
  (at (7,20): Roman=DGH=LemmaC-LP=76 vs T-bound 75; at (11,82): 328 vs
  326). The leave-structure "-2" is the coordinator's own content — same
  mechanism as the T33 rows {7,11} defect (cf. C16b).

Validity note: Lemma C itself was re-derived and checked here — per point,
3(w-3) <= C(w-1,2) for all w >= 2, hence sum_{B at x}(|B|-3) <=
floor(2C(m-1,2)/3); summing and dividing by min block size 4 (termwise
valid for all w >= 2) gives Q <= J(m). Sound.

## Full output

```
==============================================================================
CHECK 1: Guy point-C (DHS Thm 3.15) + DHS Prop 3.20 fixpoint, s=t=3
(regime A = literature-exact anchors allowed; 2 sweeps)
     cell  claim  Roman machinery  verdict
  (7, 20)     75     76        75  REACHED by published machinery
  (7, 24)     87     87        87  REACHED by published machinery
  (6, 10)     39     40        39  REACHED by published machinery
  (8, 24)     97     98        98  NOT reached (short by 1)
  (8, 25)    100    102       102  NOT reached (short by 2)
  (8, 26)    104    105       105  NOT reached (short by 1)
  (8, 27)    108    108       108  REACHED by published machinery
  (9, 40)    160    161       161  NOT reached (short by 1)
  (9, 41)    163    164       164  NOT reached (short by 1)
  (9, 42)    166    168       167  NOT reached (short by 1)
  (9, 43)    169    170       170  NOT reached (short by 1)
  (9, 44)    172    173       173  NOT reached (short by 1)
  (9, 45)    175    176       176  NOT reached (short by 1)
  (9, 46)    178    178       178  REACHED by published machinery
  (9, 47)    181    181       181  REACHED by published machinery
  (9, 48)    184    184       184  REACHED by published machinery

(7,20) regime B (no exact anchors, bounds only): machinery gives 76 vs claim 75
==============================================================================
CHECK 2: DGH Thm 1.1 v=1 LP vs Lemma C ceiling 3n + J(m)
      cell   3n+J   LP_R  LP_D1  LP_D12   LP_C  verdict(D1 vs LemmaC)
    (6, 9)     36     36     36      36     36  DGH v=1 >= Lemma C (repackaging)
   (6, 10)     39     40     39      39     39  DGH v=1 >= Lemma C (repackaging)
   (6, 11)     42     42     42      42     42  DGH v=1 >= Lemma C (repackaging)
   (6, 12)     45     45     45      45     45  DGH v=1 >= Lemma C (repackaging)
   (6, 13)     48     48     48      48     48  DGH v=1 >= Lemma C (repackaging)
   (9, 40)    160    161    161     161    160  Lemma C STRICTLY stronger by 1
   (9, 41)    163    164    164     164    163  Lemma C STRICTLY stronger by 1
   (9, 42)    166    168    167     167    166  Lemma C STRICTLY stronger by 1
   (9, 43)    169    170    170     170    169  Lemma C STRICTLY stronger by 1
   (9, 44)    172    173    173     173    172  Lemma C STRICTLY stronger by 1
   (9, 45)    175    176    176     176    175  Lemma C STRICTLY stronger by 1
   (9, 46)    178    178    178     178    178  DGH v=1 >= Lemma C (repackaging)
   (9, 47)    181    181    181     181    181  DGH v=1 >= Lemma C (repackaging)
   (9, 48)    184    184    184     184    184  DGH v=1 >= Lemma C (repackaging)
 (12, 108)    432    433    433     433    432  Lemma C STRICTLY stronger by 1
 (12, 109)    435    436    436     436    435  Lemma C STRICTLY stronger by 1
 (12, 110)    438    440    439     439    438  Lemma C STRICTLY stronger by 1
 (12, 111)    441    442    442     442    441  Lemma C STRICTLY stronger by 1
 (12, 112)    444    445    445     445    444  Lemma C STRICTLY stronger by 1
 (12, 113)    447    448    448     448    447  Lemma C STRICTLY stronger by 1
 (12, 114)    450    450    450     450    450  DGH v=1 >= Lemma C (repackaging)
 (12, 115)    453    453    453     453    453  DGH v=1 >= Lemma C (repackaging)
 (12, 116)    456    456    456     456    456  DGH v=1 >= Lemma C (repackaging)

sanity: LP_R floor should equal min_p Roman (C7d): True
note (7, 20): Roman=76, DGH-v1 LP=76, DGH v1+v2 LP=76, LemmaC LP=76, coordinator T-bound=75
note (11, 82): Roman=328, DGH-v1 LP=328, DGH v1+v2 LP=328, LemmaC LP=328, coordinator T-bound=326
```

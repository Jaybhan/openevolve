# verified_claims.md — output of verify_claims.py

Run: 2026-07-28 (rev 2, adds C16), `python3 verify_claims.py` in
`gpt_agent/theory/`. Checks the claims of `theory.md` (claim IDs C1–C16)
against:
- the local proven-exact table (`../../evaluator.py`, 161 cells),
- independently transcribed published tables (`published_data.py`),
- direct computation (exhaustive small-case z values, witness decoding,
  cap enumeration, packing-number search, group-orbit construction
  validation, bipartite isomorphism).

Notes on the non-obvious outcomes:
- **C11b**: the evolved champion's (16,16) cap/affine-hyperplane matrix is
  ISOMORPHIC to Tan's published SAT witness — same extremal object, so the
  construction is a new *description*, not a new *witness*.
- **C13c**: my branch-and-bound confirmed T_{3,3}(7) >= 15 and found no
  better, but hit its 60 s cap before exhausting the space; exactness of 15
  rests on Tan 2022 (Gurobi) and the coordinator's independent exhaustive
  search, which agree.
- **C16 (added on coordinator's urgent request)**: Tan's Table 1 prints
  T_{3,3}(18) = 405, NOT 408 — the 408 in the first revision of
  `published_data.py` was this agent's transcription error (now fixed;
  no other file or check depended on it: C6b uses T33 only for m <= 16).
  405 is re-proved here solver-free: 3-(18,4,2) inadmissible, per-point
  Johnson bound <= 405, and Tan's cyclic presentation generates exactly
  405 valid blocks. The delta=2 extrapolation (predicting 406) is refuted;
  the corrected law T33(m) = J(m) - 2*[m=3 mod 4 and m!=0 mod 3] fits all
  m=3..18.

Full output:

```
==============================================================================
C1. Local table identity vs Tan (2022) Table 3 (independent transcription)
[PASS] C1a all 161 local cells match Tan's printed value — mismatches vs Tan print: [(11, 21)] (expected [(11,21)]: local 116 overrides Tan's UB 117 per Bhan-Nobili-Langer 2026)
[PASS] C1b local exact region == Tan bold region + {(11,21),(12,22)} from BNL — |cells|=161, non-Tan-bold cells=[(11, 21), (12, 22)]
[PASS] C1c local _EXACT_UP_TO equals Tan's bold limits
==============================================================================
C2. Diagonals vs OEIS and historical attributions
[PASS] C2a z(n,n;3,3) diagonal == A001198 - 1 for n=3..16
[PASS] C2b A072567 == A001197 - 1 (n=2..24)
[PASS] C2c CRWR z(n;2) (n<=24) == A072567
[PASS] C2d Tan z_2 diagonal == A072567
[PASS] C2e Sierpinski z(4..6;3,3)=13,20,26; Brzezinski z(7,7)=33; Culik z(8,8)=42 (per A001198 history)
==============================================================================
C3. CRWR (2016) cross-check
[PASS] C3a every CRWR-bold cell inside local suite agrees with local value — 73 bold cells checked; conflicts=[]
[PASS] C3b CRWR-bold cells NOT adopted by Tan/local (literature discrepancy) — [((12, 17), 103)] — CRWR claim z(12,17;3,3)=103 exact (bold, unique-graph star); Tan prints Roman UB 108 unbolded, DGH make no improvement at (12,17). Unresolved in the literature; treat with caution.
[PASS] C3c CRWR non-exact UBs sharper than Tan's Roman print exist (informational) — [((12, 18), 109, 113), ((13, 17), 110, 117), ((13, 18), 116, 122), ((14, 17), 118, 125), ((14, 18), 124, 130), ((15, 17), 126, 134), ((15, 18), 132, 139), ((16, 17), 133, 142), ((16, 18), 140, 148)]
==============================================================================
C4. Kovari–Sos–Turan and Furedi general bounds dominate the table
[PASS] C4a KST (DGH form, best orientation) >= z on all 161 cells — []
[PASS] C4b Furedi CPC-1996 bound (FS survey Thm 3.19) >= z on all cells — []
[PASS] C4c KST bound is never tight on this range (informational) — min slack = 5.86 (KST is asymptotic, not exact, here)
==============================================================================
C5. Culik's theorem on the table
[PASS] C5a z == 2n + 2C(m,3) on ALL cells with n >= 2C(m,3) — 41 Culik-regime cells (rows: [3, 4, 5])
[PASS] C5b z < Culik value strictly below the threshold (sanity)
[PASS] C5c equality begins exactly at n=(t-1)C(m,s) for m=4 (n=8) and m=5 (n=20)
==============================================================================
C6. Roman's bound (1975) and its exactness window
[PASS] C6a min_p Roman floor-bound >= z on all 161 cells — []
[PASS] C6b z == min(Roman(p=2),Roman(p=3)) on every cell with n >= 2C(m,3) - 3*T33(m) — 71 cells in the Roman/Tan exactness window (rows 3,4,5 entirely; row 6 from n=13)
[PASS] C6c Roman UB at (12,22) equals 132 (so BNL's 132-edge witness closes it) — min_p Roman = 132 at p=5
[PASS] C6d Roman UB at (11,21) is 117; exactness of 116 needs DGH's UB — min_p Roman = 117; DGH improved to 116
==============================================================================
C7. Two-sided integer counting bound (waterfill)
[PASS] C7a WF >= z on all cells — []
[PASS] C7b number of counting-tight cells (coordinator found 78) — WF-tight cells: 78/161
[PASS] C7c WF(16,16)=136, deficit 8 at the corner
[PASS] C7d min_p Roman (floored closed form) == integer waterfill on ALL 161 cells: Roman's bound loses nothing to integer level-filling here — equal on 161/161 cells
==============================================================================
C8. (2,2): Reiman's bound and the projective-plane equality
[PASS] C8a Reiman bound >= z(n,n;2,2) for n=1..31 — []
[PASS] C8b equality exactly at n = q^2+q+1 (q=0,1 degenerate; q=2,3,4,5 planes) — tight at n=[1, 3, 7, 13, 21, 31]
[PASS] C8c floor(Reiman(q^2+q+1)) == (q+1)(q^2+q+1) for q=2..199 (Reiman 1958 equality is an identity, no q>=15 condition)
[PASS] C8d z(8,8;2,2)=24 while WF=25: first square (2,2) counting deficit — z=24, WF=25; deficits n=2..8: [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 1)]
==============================================================================
C9. Exhaustive small-case verification (Culik/Roman ground truth)
[PASS] C9a exhaustive z(3,n;3,3) == 2n+2 for n=3..6 (Culik, threshold 2)
[PASS] C9b exhaustive z(4,n;3,3) for n=4..7 == table (13,16,18,21) == floor((8n+8)/3) (Roman p=3) — {4: 13, 5: 16, 6: 18, 7: 21}
[PASS] C9c exhaustive z(3,n;2,3): equals n + 2*C(3,2) = n+6 exactly from the Culik threshold n=6 on — {3: 7, 4: 9, 5: 10, 6: 12, 7: 13}
[PASS] C9d exhaustive z(4,n;2,2): == n + C(4,2) = n+6 from n=6 on (Culik); z(4,4;2,2)=9, z(4,5;2,2)=10 below — {4: 9, 5: 10, 6: 12, 7: 13} (0.1s)
[PASS] C9e z(5,5;3,3): counting bound gives <=20; Tan witness attains 20
==============================================================================
C10. Tan witness matrices decode and verify
[PASS] C10 (3,3): K33-free with 8 == z = 8 ones
[PASS] C10 (5,5): K33-free with 20 == z = 20 ones
[PASS] C10 (10,10): K33-free with 60 == z = 60 ones
[PASS] C10 (16,16): K33-free with 128 == z = 128 ones
==============================================================================
C11. Champion construction at (16,16) vs Tan's witness
[PASS] C11a champion (16,16) valid, 128 ones, 8-regular both sides
[PASS] C11b champion (16,16) matrix isomorphic to Tan's published witness — isomorphic=True (0.0s). Same object: the cap/affine-hyperplane construction is a NEW DESCRIPTION of the known witness, not a new witness.
==============================================================================
C12. Caps in PG(3,2)
[PASS] C12a maximum cap in PG(3,2) has size 8 (exhaustive)
[PASS] C12b champion's normal set {8..15} is a cap (complement of a hyperplane)
==============================================================================
C13. Packing numbers behind the Roman window
[PASS] C13a T_{3,3}(6) == 9 (exhaustive; Tan Table 1 value) — value=9, search complete=True
[PASS] C13b ex(6; K_3) == 9 == T_{3,3}(6) (coordinator's row-6 Turan bridge)
[PASS] C13c T_{3,3}(7) == 15 (Tan: Gurobi; coordinator: exhaustive; here: B&B) — value=15, my search complete=False (if False, 15 is still confirmed as a lower bound and by two independent exact computations: Tan 2022, coordinator 2026)
[PASS] C13d Tan's cyclic presentation of T_{3,3}(7)=15 is a valid 2-fold packing of 15 quadruples
==============================================================================
C14. The two extra exact cells and DGH consistency
[PASS] C14a every DGH improved UB >= the local exact value where both exist — []
[PASS] C14b (11,21): local 116 == DGH UB 116 == BNL exact claim
[PASS] C14c (12,22): local 132 == Roman UB 132 == BNL exact claim
[PASS] C14d (11,22)=121 proven exact by BNL but NOT in the local suite (informational for the owner)
[PASS] C14e local generalized-run has NOT re-attained the two BNL cells yet (informational) — best valid so far: 11x21 -> 108/116, 12x22 -> 120/132 (the witnesses live in the BNL paper, not this run)
==============================================================================
C15. Asymptotics sanity (informational)
[PASS] C15 z(n,n;3,3)/n^(5/3) on the diagonal (theory: -> 1 as n -> inf; constant 1/2 belongs to the GRAPH version ex(n,K33)) — n=8: 42 ratio=1.312; n=12: 80 ratio=1.272; n=16: 128 ratio=1.260
==============================================================================
C16. T_{3,3}(m) = D_2(m,4,3): admissibility law, Johnson bound, m=18
[PASS] C16a a 3-(18,4,2) design is inadmissible: r = 2*C(17,2)/3 = 272/3 is not an integer (coordinator's observation confirmed)
[PASS] C16b Johnson bound >= T33 on m=3..18; slack is 2 exactly at m in {7,11} (m=3 mod 4 and m!=0 mod 3) and 0 everywhere else — in particular Johnson is TIGHT at all of m=6,9,12,15,18 — tight at [3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18]
[PASS] C16c perfect 2-fold packing (T = C(m,3)/2) <=> 3-(m,4,2) admissibility, for all m=4..18 (the coordinator's 'law'; the design-existence direction is Hanani's 3-(v,4,lambda) spectrum)
[PASS] C16d Tan's cyclic presentation for T33(17) generates exactly 340 blocks and is a valid 2-fold packing — blocks=340
[PASS] C16d Tan's cyclic presentation for T33(18) generates exactly 405 blocks and is a valid 2-fold packing — blocks=405
[PASS] C16e T33(18)=405 is therefore proven WITHOUT solver trust: Johnson bound (C16b) == validated construction (C16d)
[PASS] C16f the 'delta=2 for all inadmissible m>=7' extrapolation predicts T33(18) = 406 and is REFUTED (Tan prints 405; Johnson <= 405); the m=0 mod 3 defect floor(C(m,3)/2)-J(m) grows like m/6: 1,2,2,2,3 at m=6,9,12,15,18
==============================================================================
RESULT: all checks passed
```

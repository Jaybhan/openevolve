# design_prover STATUS

## 2026-07-29 (session 2: divisibility unification program)

**NEW: T₃,₃(27) = 1458 = J(27).** First 3|m value beyond published data.
- LB: `witnesses/t33_m27_b1458_parclass.json` — 1458 blocks, invariant
  under sigma = three 9-cycles, leave = doubled parallel class
  {i,i+3,i+6} ×2 (exactly the minimal shape predicted by the
  classification). Verified by BOTH independent checkers (80 s ILP,
  1950 orbit vars).
- UB: Johnson (classical; also re-derived from the congruences alone in
  C7_general.md Section 3b).

**Citation PINNED** (`gklo_citation.md`): GKLO, Mem. AMS 284 (2023) no.
1406, Theorem 1.1 (quoted verbatim), applies with r=3, F=K4^(3), λ=2,
p=1 to hosts K_m − (pentagon) [class] and K_m − L' [3|m families]; all
divisibility rows and typicality verified line by line. Keevash
arXiv:1401.3665 independently suffices; Delcourt–Postle arXiv:2402.17855
third route. Consequences (ineffective m₀, stated plainly):
- class m: T = J − 2 for all sufficiently large m (+ {7,11,19,23} explicit).
- 3|m: T = J for all sufficiently large m (+ 6..18 Tan, 27 here).
See `C7_general.md` for the assembled spectrum theorem.

**3|m minimal-leave classification PROVEN** (`C7_general.md` Section 3,
machine checks `scripts/verify_threefold.py` ALL PASS):
min congruence-valid leave weight = B − 4J = 2m/3 + 2·[m ≡ 9 (mod 12)];
attaining families: doubled parallel class (m ≢ 9 mod 12), doubled
hub + parallel class (m ≡ 9 mod 12); the parallel class is the UNIQUE
all-doubled minimal shape when m ≢ 9 (mod 12). Also R = 3s(s−1) exactly
for m = 3s, and B − mR = 2m/3 (hand proofs in the doc).

**m=31 (target 2245): retry queue RUNNING** (seeds / c5=5 / free mode;
task bnbzoc2py; first queue launch was lost to a zsh word-splitting bug,
recorded here per honesty bar). m=35: queue after 31 resolves.
2026-07-28 single attempts: HiGHS 3000 s time-outs, no incumbent, NOT
proven infeasible.

## 2026-07-28 (session 1)

**T₃,₃(19) = 482 and T₃,₃(23) = 883 SETTLED.**
- Upper bounds: Theorem F (theorems.md), independently re-verified here
  (`scripts/verify_theorem_F.py`, all checks pass; see J_minus_2_proof.md,
  including the k5=1 gap-check: point congruence alone forces K4^(3) at
  weight 4).
- Lower bounds: explicit verified witnesses in `witnesses/`:
  t33_m19_b482_hole.json, t33_m23_b883_hole.json, plus independent
  re-attainments t33_m11_b80_hole.json (Tan's 80) and t33_m7_b15_hole.json.
  All four have the SAME leave: doubled C₅-edge-complement pentagon
  ({012},{014},{034},{123},{234} ×2) — leave-shape universality confirmed
  constructively.
- All witnesses pass THREE independently written verifiers (two of mine
  in scripts/, SAT agent's sat_attack/verify_witness.py via the
  *_satfmt.json copies).

NOTE TO SAT AGENT: 482 attainment and 483 exclusion are both closed
analytically; the T3 run can be repurposed.

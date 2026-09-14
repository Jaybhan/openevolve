# Conjecture ledger

Every conjecture is explicit and falsifiable; status per README honesty bar.
Newest at bottom. Counterexamples recorded inline, never deleted.

---

## C1 — Rows m ≤ 5 are counting-tight (s=t=3)

**Statement**: for all n ≥ m, m ∈ {3,4,5}: z(m,n;3,3) = WF(m,n), the
two-sided integer waterfill bound (level column-degrees under
Σ_j C(c_j,3) ≤ 2·C(m,3), capped at m; min with transpose).

**Status**: VERIFIED-ALL-KNOWN (all 60 table cells m≤5). Proof strategy:
upper bound is the counting bound; lower bound needs a realization of the
level profile as a 2-fold triple packing for every n — small explicit schema
per m (m=4: ≤2 copies of each of the 4 triples + one 4-block + weight-2 pads;
m=5: complements-of-points blocks as in champion). PROOF ATTEMPT PENDING.
If Culík's theorem's known refinements already cover all n for m≤5, this is
known — theory agent checking.

## C2 — Culík regime (known theorem, our anchor)

z(m,n;s,t) = (s−1)n + (t−1)·C(m,s) for n ≥ (t−1)·C(m,s).
Status: PROVEN (Culík 1956, citation being pinned by theory agent).
VERIFIED-NUMERICALLY on all 41 applicable table cells.

## C3 — Deficit is a local-budget phenomenon

**Statement**: the refined degree-profile bound imposing pair-local budgets
(Σ_{j⊇P}(c_j−2) ≤ 2(m−2) for every row-pair P) and row-local budgets
(Σ_{j∋r} C(c_j−1,2) ≤ 2·C(m−1,2)) closes most of the deficit d = WF − z;
the cells where it does not are exactly design-existence obstructions.

**Status**: CONJECTURE, untested. Bounds agent implementing the refined bound;
test = compare refined-UB − z over the 83 deficit cells.

## C4 — (16,16) extremal structure

**Statement**: z(16,16;3,3) = 128 is attained by both sides of 8 affine
hyperplanes of F₂⁴ with cap normals (RM(1,4) sub-family), 8-regular both ways.
**Status**: attainment VERIFIED-NUMERICALLY (this workspace, independent
rebuild: 128 edges, 0 violations). Optimality is the published proven value.
Uniqueness (up to isomorphism) UNKNOWN — interesting question. Novelty of the
description under literature review.

## C5 — Shape of the full-table formula (the target)

**Statement (working form)**: z(m,n;3,3) = RWF(m,n) − e(m,n), where RWF is
the local-budget-refined waterfill bound of C3 and e is 0 except on an
explicitly listable set of design-obstruction cells with a closed-form rule.
**Status**: CONJECTURE, blocked on C3 measurements. The prize: if e ≡ 0 on
all 161 cells, RWF *is* the exact formula on the entire proven table, and
every RWF-attaining construction is an extremal witness generator.

---

## C6 — δ-conjecture for T₃,₃  [REFUTED 2026-07-28]

**Statement**: T₃,₃(m) = ⌊C(m,3)/2⌋ − 2 for all inadmissible m ≥ 7.
**Refutation**: predicts T₃,₃(18) = 406; Tan's Table 1 prints 405, re-proven
solver-free by the theory agent (Johnson bound 405 from r_x ≤ 90 + Tan's
dihedral-orbit construction attaining it). The δ=2 pattern at m = 9,12,15
was small-range coincidence; for m ≡ 0 (mod 3) the Johnson bound binds and
the gap to ⌊C(m,3)/2⌋ grows ~m/6. Kept per protocol: failed conjectures are
recorded, not deleted. (Side yield: the conjecture's impossibility argument
exposed a transcription error in our own theory data — 408 → 405.)

## C7 — Johnson-form law for T₃,₃ = D₂(m,4,3)  [NEW CONJECTURE]

**Statement**: with J(m) = ⌊ m·⌊(m−1)(m−2)/3⌋/4 ⌋ (Johnson bound),
  T₃,₃(m) = J(m) − 2·[ m ≡ 3 (mod 4) and m ≢ 0 (mod 3) ].
**Status**: VERIFIED-ALL-KNOWN (all m = 3..18; Johnson tight everywhere
except m ∈ {7,11}, slack exactly 2). Theory agent confirms no published
determination of D₂(v,4,3) — same genre as Bao–Ji's proven λ=1 formula, so
this is a new falsifiable design-theory conjecture feeding Theorem 7's
S_m(0). **Predictions**: T₃,₃(19) = 482, T₃,₃(21) = 661, T₃,₃(22) = 770.
**UPDATE (session 3): C7 PROVEN at m = 7, 11, 19, 23.** UB for the whole
class = Theorem F (Lemma E + pair-parity + finite leave classification);
LB = explicit Z₅-symmetric doubled-pentagon-leave constructions
(design_prover/witnesses/, quadruply verified): **T₃,₃(19) = 482 and
T₃,₃(23) = 883 — new determinations beyond every published table.**
Remaining for C7 in full: the class-general construction (universal
doubled-pentagon leave observed at all four members; parametric scheme
under review; m = 31, 35 stretch runs in flight). Note: C7's non-class
predictions (T(21), T(22)) remain untested.

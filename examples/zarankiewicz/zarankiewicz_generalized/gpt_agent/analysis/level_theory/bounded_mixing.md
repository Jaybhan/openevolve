# Bounded mixing: how far the two-level restriction is from the ledger

level_theory, 2026-07-30. Task: the exchange-lemma case for
"ledger max ≥ best-two-level − C outside the corner scale", with the
unproven parts stated precisely. (Recall Theorem 9: z IS the ledger max —
the exact Pareto identity; so this document is about the loss incurred by
F's two-adjacent-level restriction.)

## 1. What is PROVEN

**Lemma W (waterfill vertex — the LP shape).** Over the relaxation
max Σ(w−2)k_w s.t. Σ C(w,3)k_w ≤ B, Σ k_w ≤ n, k ≥ 0 (supplies
ignored), every optimal basic solution has at most two positive levels,
and by the strict convexity of w ↦ C(w,3) the two are ADJACENT.
*Proof:* two constraints ⇒ ≤ 2 positive variables at a vertex; if levels
u < v−1 are both positive, the exchange (u, v) → (u+1, v−1) at equal
slot-and-column cost strictly helps or ties toward adjacency (second
difference of C(·,3) > 0). ∎  [This is the workspace's level-filling
fact from Theorem 1, restated in ledger variables. It is WHY the
two-level formula is the right zeroth order everywhere.]

**Lemma M (pad-flattening exchange).** In any legal configuration
containing a block B of weight W ≥ 4 and a pad (weight-≤2 column):
delete one point of B (legal: coverage shrinks; −1 edge), then upgrade
the pad to a weight-3 column on any triple through the deleted point
inside B (its coverage just dropped by 1, so capacity exists; +1 edge).
Net edge change 0; the multiset of heavy weights strictly decreases.
*Consequence:* any optimum with p pads can be converted, at equal value,
to one whose total weight-excess above any target level is reduced by
up to p. In particular in the Culík regime (n ≥ B, pads guaranteed)
optima flatten completely — consistent with level 3 being exactly
optimal there. ∎  [Generalizes theorems.md's Lemma B, which is the
W = 6 instance plus its converse accounting.]

**Lemma S (supply-cap vertex count).** Adding s binding supply
constraints k_w ≤ 𝔇_w to Lemma W's LP admits optimal vertices with at
most 2 + s positive levels. So a THIRD level appears only when some
supply cap binds — and the known instances are exactly that:
the (7,7)/(7,8) optima carry a hexad because 𝔇₅(7) = 4 binds
(4 pentads is not enough width at n = 7,8), and the m = 9, n ≤ 13
hexads ride the binding 𝔇₆(9) = 6. The phenomenon is localized to the
column-scarce corner where several caps bind simultaneously.

## 2. What is MEASURED (all 206 known cells)

- The two-level F undershoots z exactly once: (7,8), by 1.
  (At (7,7) the budget-only bottom layer masks it: F overshoots.)
  **Empirical mixing constant: C_mix = 1.**
- With the mixed-ledger engine (fullpass.py: multi-level enumeration +
  slice caps) the LB side never undershoots anywhere — every known
  optimum's value is reproduced by a ≤ 3-heavy-level configuration; the
  heavy levels are adjacent ({w−1, w, w+1}) in every stored band
  witness, with non-adjacent spectra only at the extreme corner
  ((8,8): heavy weights {5, 7}). No known optimum needs 4 heavy levels.
  [Witness-profile scan, 64 stored optima.]

## 3. What REMAINS UNPROVEN (stated precisely)

1. **The band constant.** Conjecture: for all (m, n) with n ≥ ν(m)
   (hexad threshold), max two-adjacent-level value ≥ z − 1, and
   ≥ z outside finitely many cells per row. Status: true on all known
   data; Lemma S explains the mechanism but does not bound the VALUE
   loss of dropping the third level; a proof would need a quantitative
   version of the (u,v)-exchange when a supply cap pins u.
2. **The corner.** For n = Θ(m) (diagonal scale), no bounded-mixing
   statement is claimed: multiple caps bind, profiles use 3+ levels
   ((8,8): weights {5,7}! — non-adjacent), and the supply estimates
   themselves are the open cap-geometry quantities. The measured F-error
   there (≤ 8 at m ≤ 16) is NOT proven bounded as m → ∞; per
   diagonal_limit.md the budget-vs-truth gap grows like
   (2^{1/3} − 1 − o(1))·m^{5/3} if F's supplies stay budget-anchored —
   the master statement must (and does, in the scaffold) except the
   corner scale.
3. **Pads in the band.** Lemma M requires pads; band optima typically
   have none (witnesses show triples instead). The triple-analogue
   exchange (shrink a block, upgrade a TRIPLE to a quad through a freed
   pair) costs capacity elsewhere and is NOT always available; this is
   the precise gap between Lemma M and conjecture 1.

## 4. Summary for the MASTER THEOREM

    z = ledger max                    (exact identity, Theorem 9)
    ledger max ∈ [F − C_mix, F + C_supply]   with, on all known data,
    C_mix = 1 (single cell) and C_supply = 8 (single corner cell);
    C_mix = 0 outside the column-scarce corner on all known data;
    both constants PROVEN only cell-by-cell (the verification sweep),
    the mechanisms identified (Lemmas W, M, S), the general bounds open.

# HEADLINE: the link second-moment bound — elementary counting beats the budget constant

Asymptotic analyst, 2026-07-30. Machine companion: `link_moment.py`
(all three verification legs; runtime ~1 min). This file supersedes the
"blindness barrier" (Theorem A4) of `ladder_asymptotics.md`, whose
fixed-block-size reasoning does not extend to blocks of size m^{2/3} —
the failure is constructive and is exactly this theorem.

## Theorem M1 (per-point link second-moment inequality). **PROVEN**

Let A be m×n, K₃,₃-free. Fix a row x; let L₁,…,L_d ⊆ [m]∖{x} be the
supports-minus-x of the d columns containing x (u_i = |L_i| = w_i − 1,
M = m − 1, Y = Σu_i, I_ij = |L_i ∩ L_j|, and for a pair {y,z} let
μ({y,z}) = #{i : {y,z} ⊆ L_i}). Then μ ≤ 2 everywhere (a pair at μ ≥ 3
gives the triple {x,y,z} in 3 columns), and:

  (C1)  Σ_{i<j} C(I_ij, 2) = Σ_pairs C(μ,2) = #{pairs at μ=2} ≤ C(M,2);
  (C2)  Σ_{i<j} C(I_ij, 2) ≥ C(d,2)·C(S̄,2),  S̄ = S/C(d,2),
        S := Σ_{i<j} I_ij   [Jensen, convexity of C(·,2)];
  (C3)  S = Σ_y C(r_y, 2) ≥ M·C(Y/M, 2)      [double count + Jensen].

Solving the chain exactly (quadratic inequalities, all explicit):

    S ≤ C(d,2) + √(2·C(d,2)·C(M,2)),
    Y ≤ M + √(2M·S)  ⟹  **Y ≤ 2^{1/4}·M·√d + O(d^{3/2} + M)**.

Equivalently, per point, in Brown units (d = δm^{2/3}, Y = ηm^{4/3}):
**η² ≤ √2·δ** — strictly stronger than the ladder's η² ≤ 2δ, by the
factor √2. Structurally: the pairwise intersections {L_i ∩ L_j} form a
partial linear space (two block-pairs sharing a point-pair would force
μ ≥ 3), so they obey a Fisher-type pair budget that the budget-optimal
profile overfills.

## Theorem M2 (the elementary diagonal bound). **PROVEN**

    z(m, m; 3,3) ≤ 2^{1/6} · m^{5/3} · (1 + o(1)),   2^{1/6} = 1.12246…

**Proof.** Sum M1 over the m points: Σ_x Y_x = Σ_B w_B(w_B−1), so with
A₁ = Σw_B/m^{5/3}, A₂ = Σw_B²/m^{7/3} (pads contribute O(m)):
A₂·m^{7/3} ≤ 2^{1/4}·m·Σ_x√d_x + O(m^{13/6}) ≤ 2^{1/4}m^{3/2}√(Σd_x)
[Cauchy–Schwarz] = 2^{1/4}·√A₁·m^{7/3}(1+o(1)), i.e. **A₂ ≤ 2^{1/4}√A₁**.
With at most m block-columns, (Σw)² ≤ m·Σw² gives A₁² ≤ A₂. Chain:
A₁² ≤ 2^{1/4}·A₁^{1/2} ⟹ A₁ ≤ 2^{1/6}. ∎
(Error terms: Σd_x^{3/2} ≤ √m·Σd_x = O(m^{13/6}) and the per-point
+M = O(m²) — both o(m^{7/3}); fully uniform, no hidden regularity.)

**The budget/ladder optimum is infeasible here**: the atom t = 2^{1/3}
has A₂ = 2^{2/3} = 1.5874 > 2^{1/4}√(2^{1/3}) = 1.3348 — VIOLATED. The
new extremal atom is t = 2^{1/6} (feasible, tight in both A₂ ≤ 2^{1/4}√A₁
and A₁² ≤ A₂).

**This answers the task's headline check in the positive** — an upper
bound asymptotically better than 2^{1/3}·m^{5/3} by elementary means —
with the precision that the *specified* toolset (ladder + congruences)
provably cannot do it (ladder_asymptotics.md A1–A3 still stand: that
class is exactly blind), while the next elementary rung — second
moments of link intersections — closes exactly half the log-gap:
log(2^{1/3}/2^{1/6}) / log(2^{1/3}/1) = 1/2.

## Theorem M3 (single-level configurations reach the Brown constant). **PROVEN**

If all block weights lie in [(1−ε)w̄, (1+ε)w̄] with w̄ = t·m^{2/3},
t > 0 fixed (links homogeneous), the chain sharpens: P = Σᵢ C(uᵢ,2)
satisfies P ≥ 2·#{μ=2} = 2P₂ (μ ≤ 2!), while C2–C3 force
P₂ ≥ (1−o(1))·d²ū⁴/(4M²); with P = (1+O(ε))·d·ū²/2 this pinches

    d_x ≤ (1 + O(ε) + o(1)) · M²/ū²   — HALF the naive capacity 2M²/ū².

Summing the pointwise cap: Σd_x ≤ m·M²/ū², i.e. a single-level
configuration has at most (1+O(ε)+o(1))·m³/w̄³ = m/t³ columns; the
diagonal value is E ≤ max_t [ t·min(1, 1/t³) ]·m^{5/3}(1+o(1)) =
m^{5/3}, attained at t = 1.
**Single-level diagonal configurations obey the Brown constant 1, by
elementary counting.** Consequences:

- **The φ-step upper bound is now citation-free**: a level-w supply
  family is single-level by definition, so φ(c) ≤ c⁻³ for every c > 0
  follows from M3's pinch d_x ≤ (1+o(1))M²/u² (valid exactly when
  Ī = u²/M ≫ 1, i.e. w ≫ √m — precisely the half-budget window of
  phi_profile.md P2, whose UB is therefore also now elementary).
  Füredi's theorem remains needed only for the hard zero at c > 1 and
  the rectangular corollary.
- Brown's own links sit on this law: measured pair-slot usage
  P/(2C(M,2)) = 0.295 (q=7), 0.366 (q=11) vs the predicted t³/2 =
  (1−1/q)³/2 = 0.315, 0.376 — and with the μ-distribution degenerating
  to {0,2} (measured μ=1 fraction 0.298 → 0.185 → 0).

## Theorem M4 (finite elementary diagonal bound). **PROVEN** (`m4_finite.py`)

The chain admits a fully finite, pad-safe, E-only form: with M = m−1,
G(d) := M + √(2M·(C(d,2) + √(2·C(d,2)·C(M,2)))) (concave on [2,m],
machine-verified at every used scale; per-point M1 gives Y_x ≤ G(d_x)),
Jensen in the concave direction over the m points plus Cauchy–Schwarz
over ≤ m block columns yield

    z(m,m;3,3) ≤ max{ E : E²/m − E ≤ m·G(E/m) }.

(Pads only help: each pad column removes ≈ 0.6·m^{2/3} from the block
budget and returns ≤ 2.) Computed against the workspace's standing
bounds:

    m       WF/Roman   M4       Füredi(printed)   best
    179     7302       7302     —                 tie
    180     7370       7368     7950              M4  ← m** = 180
    400     27692      27063    28009             M4
    800     87553      84145    84594             M4
    925     —          —        —                 Füredi takes over

**M4 < WF/Roman for every m ≥ 180, and M4 is the strongest diagonal
upper bound in the workspace for 180 ≤ m ≤ 924** — the elementary
analogue of the Füredi crossover m* = 470, reached 290 sizes earlier
and without any citation. (At m ≤ 179 the integer WF remains better;
M4's asymptotics is Theorem M2's 2^{1/6}m^{5/3}.)

## Verification (the three ways, `link_moment.py`)

1. **Exact witnesses.** (16,16) extremal (rebuilt from cap-hyperplanes,
   re-verified K₃,₃-free, 128 ones): C1–C3 hold at all 16 points, with
   C2 EXACTLY TIGHT (Σ C(I,2) = 84 = Jensen bound; I_ij ≡ 3
   homogeneous) and pair-usage 80% of capacity, μ ∈ {0,2} only. Brown
   q = 7, 11: chain holds, statistics as predicted above. Additionally
   the chain was verified at EVERY point of ALL 64 stored band
   witnesses (analysis/witnesses/w_*.json — mixed weights 4–6+,
   rows 10–11): valid everywhere, no μ > 2, no Jensen violation
   (`check_band_witnesses` in link_moment.py) — the mixed-weight case
   is covered by real data, not just the homogeneous one.
2. **The pinch.** Budget atom violates M1's scaled form; 2^{1/6} atom
   feasible-tight; numeric sup over two-atom mixed profiles under
   {A₀ ≤ 1, A₂ ≤ 2^{1/4}√A₁}: 1.12240 ≈ 2^{1/6} (grid resolution).
   Control: the "cheating" two-level profile (c₁ = 3/2 at 8m/27
   columns + c₂ = 1) that naive independent-per-level supplies would
   permit reaches 1.148 — M1 correctly kills it (A₂ = 1.370 > 1.274) —
   i.e. the moment chain enforces the JOINT capacity that standalone
   per-level supply functions miss.
3. **No finite contradiction.** At m = 16 the asymptotic form's
   corrections dominate (Y = 56 vs leading term 50.5), so
   z(16,16) = 128 > 2^{1/6}·16^{5/3} = 114 is consistent — the exact
   finite chain (which is what M1 asserts) holds on the witness. The
   realized finite tool is Theorem M4 (best bound for m ≥ 180); for
   17 ≤ m ≤ 179 a per-point disaggregated LP/SOCP over (d_x, Y_x)
   remains the open route — NOT claimed to improve z(17,17) ≤ 143
   (indicative slack 15.1% at E = 144 on the averaged profile, which
   is a feasibility indicator only, see the warning below).

## Honest caveats

- Novelty UNVERIFIED: second-moment refinements of KST are a known
  genre (Füredi's own 1996 proof achieves the better constant 1 by
  deeper counting; Nikiforov 2009 and Conlon 2021 refine KST). The
  workspace value: (i) full elementarity at 2^{1/6}; (ii) the EXACT
  finite per-point inequality (C1–C3) as machine-usable cuts; (iii)
  the citation-free φ-step/half-budget UBs; (iv) the corrected map of
  where elementary methods plateau: linear degree class → 2^{1/3}
  (blind, proven), second-moment class → 2^{1/6} (mixed) and 1
  (single-level), Füredi → 1 (all configs). Literature referee
  dispatched before any priority language is used.
- The mixed-level gap [1, 2^{1/6}] is genuinely open for this method
  class: dyadic bucketing shows deeper *per-level* pinches are evadable
  by weight-spreading, so closing mixed configurations to 1
  elementarily needs cross-level structure — or Füredi.
- **Warning for future finite-m use of M1** (mistake made and caught
  in-session): applying the chain to the AVERAGED per-point profile
  (d̄, Ȳ) is NOT valid — the chain's implied F(d,Y) ≤ C(M,2) has F
  non-jointly-convex (S²/2N type), so pointwise validity does not
  transfer to averages; a naive averaged test "cuts" WF by 2–11 at
  m = 16..30 but proves nothing. The valid global route is the concave
  majorant Y_x ≤ G(d_x) := M + √(2M(C(d,2)+√(2C(d,2)C(M,2)))) summed
  with Jensen — which at m = 17 only bites at E ≥ 170 (worse than WF).
  Finite-m gains from M1 require the per-point disaggregated LP/SOCP,
  not shortcuts.

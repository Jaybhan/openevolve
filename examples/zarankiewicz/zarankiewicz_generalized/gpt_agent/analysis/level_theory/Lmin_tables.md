# L_min tables and the assembled master formula F(m,n)

level_theory, 2026-07-30, FORMULA-ONLY mode. Generator: `Lmin_gen.py`
(self-tests ALL PASS); formula: `F.py` (self-contained, no file reads).

## 1. The periodic leave spectrum L_min(w, m)

For level w, the minimal congruence-valid leave weight of Lemma L3,
i.e. the least L ≡ B (mod C(w,3)) compatible with the point/pair
congruences — plus the PROVEN class-uniform refinements. The signature
(c₁, c₂, B mod C(w,3)) is periodic with MINIMAL period

    P₄ = 12,  P₅ = 15,  P₆ = 20,  P₇ = 105,  P₈ = 168

(machine-verified minimal — the scaffold's guess "P₅ | 60, P₆ | 60" is
corrected: the periods are 15 and 20; both divide 60, so the scaffold
statement was true but not tight). L_min is NOT constant on a class in
general — it is an explicit formula per class, of one of four shapes:

| type | condition | L_min(m) |
|---|---|---|
| perfect | c₁ = c₂ = 0, β = 0 | 0 |
| gapped | c₁ = c₂ = 0, β ≠ 0 | β + C(w,3)·⌈(C(w−1,2) − β)/C(w,3)⌉⁺  (an O(1) class constant) |
| linear | c₂ = 0 < c₁ | ⌈m·c₁/3⌉ rounded up into β (mod C(w,3)) |
| quadratic | c₂ > 0 | max(⌈m·p_min(m)/3⌉, ⌈C(m,2)·c₂/3⌉) rounded up into β; p_min(m) = ⌈(m−1)c₂/2⌉ rounded up into c₁ (mod C(w−1,2)) |

**Proven class-uniform refinements** (beyond raw L3 arithmetic):
- w = 4, m ≡ 3 (4), 3 ∤ m: raw 6 → **10** (Lemma E kills weight 2,
  Theorem F's empty classification kills weight 6; doubled pentagon
  attains 10). With this, (B − L_min)/4 reproduces the FULL Theorem 11
  spectrum for every m ≤ 200 (self-test) — the w = 4 anchor is exact.
- w = 4, 3 | m: the raw rounding already equals Theorem 11(c)'s
  classification 2m/3 + 2·[m ≡ 9 (12)] (self-tested m ≤ 200).
- w = 5, m ≡ 14 (15): raw 8 → **18** (the unique weight-8 shape is the
  doubled K₄⁽³⁾, whose pair-leaves are 4 ≢ 0 (mod 3) — class-uniform
  hand proof, spectrum.md §3.1).

**Representative constant classes** (full tables: run `Lmin_gen.py`):
w=4: perfect residues {1,2,4,5,8,10} (12) → 0; gapped {7,11} → 10;
linear {0,3,6,9} → ≈ 2m/3. w=5: perfect {2,5,11} (15) → 0; gapped
{8} → 12, {14} → 18; quadratic (the other 10 classes) → Θ(m²/6)-scale
formulas. w=6: perfect {2,6,12,16} (20) → 0; six linear classes
(even m, c₁ ∈ {2,6}); ten quadratic (odd m). w=7: 6 perfect classes
(mod 105), 8 gapped, 7 linear, 84 quadratic. w=8: 10 perfect (mod 168),
6 gapped, 40 linear, 112 quadratic.

**Sporadic completion obstructions** (Dehon-type; per-m, NOT class
facts; all proven in-workspace, `decisions.csv`): (11,5): perfect
(Dehon 1976, re-proven) AND b = B/10 − 1 dead ⇒ 𝔇₅(11) ≤ 31;
(10,5) ≤ 21; (12,7) ≤ 10. These live in F's small-m branch. No
class-uniform obstruction beyond the two refinements above is currently
proven; whether e.g. the m ≡ 11 (15) class carries obstructions at
OTHER members is open (GKLO says no for large m).

## 2. F(m,n) — the assembled formula (`F.py`)

    F(m,n) = max over w ∈ [3, min(m,13)], C(w−1,3)·n ≤ B:
                 (w−1)·n + min( n, D_w(m), ⌊(B − C(w−1,3)n)/C(w−1,2)⌋ )

with the supply spectrum D_w(m) as documented in F.py's header:
exact closed forms (w ≤ 4 spectrum, wedge, complement j ≤ 4), frozen
proven small-m tables (m ≤ 16), and the large-m branch
min(J_w(m), (B − L_min(w,m))/C(w,3)).

## 3. Verification against every known z value (206 cells)

    F − z histogram:  {−1: 1, 0: 108, +1: 41, +2: 30, +3: 14,
                       +4: 7, +5: 4, +8: 1}

- **z ≤ F + 1 and z ≥ F − 8 on ALL 206 known cells**:
  the measured master-theorem constants are C_over = 8 (attained only
  at the (16,16) ovoid corner) and C_under = 1 (attained only at
  (7,8), the proven three-level cell — the bounded-mixing constant).
- F = z on 108/206; |F − z| ≤ 2 on 179/206.
- The overshoot profile is exactly the ledger-deficit structure
  (spectrum.md §9): with the mixed-ledger corrections (fullpass.py
  engine — slice caps + mixed-level Lemma E; cell-computation now
  frozen per owner directive) the verified-equal count stood at
  **168/206** with every computed slice agreeing with the table
  (no standing claims), rows 3–8 fully closed at 136/136.
- Regime narrative: F = z on all of rows 3–5 (WF regime), the wide
  regions/bands of every solved row at level 4 (Theorem 8 form), and
  the pure-level segments; F = z + O(1) through the deep band (the
  +1/+2 mass = mixed-ledger drops, each a finite design fact); the +5/+8
  tail is the corner scale, where the deviation IS the second-order
  Brown–Füredi problem (diagonal_limit.md).

## 4. Honesty labels

- L_min arithmetic: PROVEN (Lemma L3; self-tested).
- Class refinements: PROVEN (cited arguments).
- Large-m attainment of L_min (making D_w exact): PROVEN for perfect
  classes (GKLO, Theorem L4); for gapped/linear/quadratic classes:
  attainment requires the leave-realization bridges — doubled shapes
  where proven (e.g. w=5 m ≡ 9 (15) at m = 9), otherwise sandwiches
  (spectrum.md §4.3–4.4). Where unproven, D_w is an UPPER estimate and
  F inherits upper-type semantics.
- The verification sweep uses only already-known z values (no new cell
  computation), per the owner directive.

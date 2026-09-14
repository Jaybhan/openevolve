# The master formula scaffold for z(m,n;3,3)

Coordinator derivation, 2026-07-28. This is the flagship-target document:
the s=3 analog of Chen–Horsley–Mammoliti's below-threshold program (open per
theory/novelty_checklist.md N1). Everything here is elementary and checkable;
the one non-arithmetic object is the supply function S_m.

## Setup

B = 2·C(m,3). Columns are blocks; write a solution's profile as k_h = #columns
of height h (h ≥ 3; height-2 pads free). Edges E = 3n + Q where
Q := Σ_h (h−3)·k_h + (k-part bookkeeping): precisely, with U = E − 2n units,
U = Σ(h−2)k_h and Q = U − n when all columns have height ≥ 3; in general
Q = Σ_{h≥4}(h−3)k_h − (#pads needed)… we use the clean two-constraint form:

  (C) columns:  Q ≤ k₄ + 2k₅ + 3k₆ + 4k₇ + …          [from Σk_h ≤ n]
  (S) slots:    2k₄ + 7k₅ + 16k₆ + 30k₇ + … ≤ B − n − Q

(Derivation: U = k₃+2k₄+3k₅+4k₆+5k₇, slots = k₃+4k₄+10k₅+20k₆+35k₇ ≤ B;
eliminate k₃ ≥ 0 and Σk ≤ n. Full algebra in Lemma A below.)

## Lemma A (Roman + heavy-penalty, PROVEN — pure arithmetic)

Adding 3×(C) to (S)-rearrangement gives, for every legal profile:

  Q ≤ (B − n)/3 − k₅ − (10/3)k₆ − (22/3)k₇ − …

Consequences:
1. Q ≤ ⌊(B−n)/3⌋ always — **this is exactly Roman's bound** (rediscovered;
   cite Roman 1975).
2. Every weight-5 block reduces the attainable ceiling by ≥ 1, weight-6 by
   ≥ 10/3, weight-7 by ≥ 22/3. So **in the Roman window (demand ≤ supply),
   heavy blocks strictly lose** — Roman equality needs quads+triples only.

## The supply function (the real object)

  S_m(k₅,k₆,k₇,…) := max k₄ such that a multiset of k₅ weight-5 + k₆
  weight-6 + … blocks AND k₄ weight-4 blocks exists with every triple
  covered ≤ 2 times.

S_m(0) = T₃,₃(m) (Tan's packing numbers, published for m ≤ 18).
The higher slices are, to our knowledge, NOT in the literature (referee
N6a marks even a T₃,₃ formula as open).

## Master formula (CONJECTURE-SCAFFOLD, being verified cell-by-cell by ILP)

  z(m,n;3,3) = 3n + Q*(m,n),
  Q*(m,n) = max over heavy profiles (k₅,k₆,k₇) and k₄ ≤ S_m(k₅,k₆,k₇) of
     min( k₄+2k₅+3k₆+4k₇  [column-bound],
          arithmetic ceiling of Lemma A [slot-bound] )
     subject to triple-fill feasibility (n − Σk_{≥4} triples fit in
     B − slots; automatic in the ranges used — proof per range).

Three regimes:
- **Column-bound (n below ≈ B/4 and beyond Tan's frontier)**: heavy blocks
  pay; witnesses show declining k₅ as n grows (e.g. row 8: 5¹⁰ at n=10 →
  5²4²¹ at n=23). Determination = knowing S_m.
- **Supply-bound (middle band, only for m with imperfect packings)**:
  z = 3n + T₃,₃(m). Verified rows 6 (n∈[8,13]) and 7 (n∈[14,25]).
- **Budget-bound (Roman window n ≥ B − 3T₃,₃(m))**: z = 3n + ⌊(B−n)/3⌋
  (published). Then Culík for n ≥ B.

## Status ledger

- Lemma A: PROVEN (above; machine-checkable algebra).
- Row 6 (all n): PROVEN — n ≤ 12 exhaustive + structural (Thm 2),
  n ≥ 13 Roman window (published). COMPLETE ROW.
- Row 7 (all n): n ≤ 23 exhaustive (solver3, matches Tan); n = 24 NEW —
  ILP-confirmed z(7,24)=87 = 3n+T₃,₃(7), with human-readable proof
  (Lemma A + T₃,₃(7)=15 + sub-packing LB); n ≥ 25 Roman window (published).
  COMPLETE ROW — first complete s=3 row beyond the published windows.
- Rows 8,9,10,11 gap bands: ILP runs in flight (ilp_gapband.py). Each
  completed band ⇒ another complete row, with z = 3n + Q* verified and
  S_m slices extracted from optimal witnesses.
- General m: blocked on S_m theory. S_m(k₅,…) tabulation planned via
  constrained ILPs for m ≤ 10; a closed form for T₃,₃(m) alone is an open
  design-theory problem (N6a).

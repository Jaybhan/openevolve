# The evolved champion, decoded (program cd6af8a9, iter 134)

Coordinator's independent mathematical reading of
`../openevolve_output/best/best_program.py`. Everything below re-derived and
re-verified by me, not taken from the code comments.

## The frame

A column is a **block** (subset of rows). K_{3,3}-freeness ⟺ every 3-subset of
rows lies in at most 2 blocks — i.e. the columns form a **2-fold packing of
triples**. The champion tracks capacity {triple: 2} and spends it. Completion
of any seed family is canonical and search-free: (a) lex-first 4-blocks while
capacity allows, (b) repeated 3-blocks (1 capacity → 3 edges), (c) weight-2
pads (0 capacity → 2 edges); both regimes generated, better kept.

## The algebraic families by regime

- **M ≤ 6**: hand-derived "omission codes" — blocks are complements of small
  sets chosen so inclusion multiplicities stay ≤ 2.
- **M = 7**: three families, best is *doubled Fano-line complements*: a
  non-line triple lies in exactly one line-complement (4-set), a line triple in
  none; doubling keeps multiplicity ≤ 2.
- **M = 8..15**: rows = nonzero points of F₂⁴; column a ∈ F₂⁴\{0} is the
  affine hyperplane side {x : ⟨a,x⟩ = 1}. An independent triple has exactly
  2^{4−3} = 2 covering columns; a dependent triple (x⊕y⊕z=0) has 0 (the system
  is inconsistent: summing the three equations gives 0 = 1). Freeness is
  linear algebra, not luck.
- **M = 16** (the crown jewel): columns = **both sides** of the 8 affine
  hyperplanes whose normals are {a : top bit set} — a **cap in PG(3,2)** (no
  three normals XOR to 0). Mechanism: a triple {x,y,z} is one-side-constant
  for normal a iff a ⊥ span(x⊕y, y⊕z); those a form (2-dim)⊥ \ {0} = a
  projective **line** (3 points); a cap meets a line in ≤ 2 points → every
  triple covered by ≤ 2 of the 16 columns.

## Independent verification [VERIFIED-NUMERICALLY]

Rebuilt from the description above (not the code): 16×16 matrix, 128 edges =
z(16,16;3,3), 0 violating triples, 8-regular in rows AND columns, cap property
confirmed. This is the extremal witness for the deficit-8 corner cell my
quick-scan flagged as the most structure-demanding cell in the table.

Equivalent description: the 16 columns are the supports of the sixteen
weight-8 codewords ⟨a,x⟩+b of the Reed–Muller code RM(1,4) whose linear part
lies in the cap. (Full RM(1,4) has 30 weight-8 words = both sides of all 15
hyperplanes; the cap picks 8 of 15 normals.)

## Generalization levers for (s,t) [CONJECTURE — being built by engineer]

1. Coverage-capacity frame verbatim: (t−1)-fold packing of s-subsets.
2. Hyperplane family: rows X ⊂ F₂ᵏ with no even-size zero-sum subset of size
   ≤ s (a code distance condition!); coverage of any distinct s-set is then 0
   or 2^{k−rank} ≤ 2^{k−s}; choose k so 2^{k−s} ≤ t−1.
3. Cap trick: "both sides" doubles column count when the normal set meets
   every relevant flat in ≤ t−1 points — replace cap-vs-line by
   set-vs-(s−1)-flat conditions; F_p versions give p sides per hyperplane.
4. Fano trick: complements of lines of PG(2,q), each (t−1) times.

Novelty of #2/#3 as stated is UNDER REVIEW by the theory agent (they smell
like they could be known — Reed–Muller/cap language is classical; the
question is whether this exact Zarankiewicz application is published).

## What the champion is and is not (corrected after miner's ablation)

It contains no lookup table of *answers* and no runtime search — cell values
never appear in the code. BUT the miner's forensic ablation (mining_report.md
§2.2) shows the M≤6 omission tables and the three M=7 seed families are
per-M hand-built structures, not instances of a rule: stripping them (nothing
else changed) drops the champion from 110/161 to 74/161 exact. So ~33% of its
exactness is regime-specific combinatorial craftsmanship that does not
transfer. The genuinely rule-like parts: the F₂⁴ hyperplane family (100%
accurate at M=13–16), the cap trick, and the canonical completion trade-off.
Its misses concentrate in the deficit band (M=8–12 "middle stretch" of the
F₂⁴ truncation range — worst row M=8 at 0/16 for the rule-like part), where
canonical completion of a fixed algebraic seed is too rigid.

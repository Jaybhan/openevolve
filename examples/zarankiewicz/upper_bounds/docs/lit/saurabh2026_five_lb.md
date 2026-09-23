# Saurabh (2026) — Five improved lower bounds for Zarankiewicz numbers z(m,n;3,3)

Literature note for the upper-bounds thesis. Written 2026-09-21 from the arXiv v1 PDF, the
arXiv HTML rendering, and the ancillary files. Everything in Sections 1–5 is what the paper
states; Section 6 onward is our own verification and inference and is labelled as such.

## 1. Bibliographic record

| Field | Value |
|---|---|
| Title | Five improved lower bounds for Zarankiewicz numbers z(m,n;3,3) |
| Author | Abhishek Saurabh (independent researcher, `saurabh.abhishek1985@gmail.com`) |
| Identifier | arXiv:2608.26603v1 [math.CO], submitted 27 Aug 2026 (paper dated 28 Aug 2026) |
| Length | 6 pages; 2020 MSC 05C35, 05D05, 05B20 |
| Ancillary files | `README.txt`, `SHA256SUMS.txt`, `witnesses.txt`, `witnesses.json`, `verify_13x19.py`, `verify_14x19.py`, `verify_16x18.py`, `verify_padded_14x20.py`, `verify_padded_16x19.py` (all stdlib-only Python) |
| Local copies | PDF + ancillary files in the session scratchpad `lit/saurabh2026.pdf`, `lit/saurabh_anc/`; the five bitstring witnesses are reproduced in Appendix A of this note |
| Cites | Afrasyab 2608.08154; Bhan–Nobili–Langer 2605.01120v2; Collins–Riasanovsky–Wallace–Radziszowski JAC 47 (2016) / 1604.01257; Davies–Gill–Horsley DM 349 (2026) / 2411.18842; Hou 2608.08549; Kővári–Sós–Turán 1954; Tan 2203.02283; Zarankiewicz 1951 |
| Accessible | Yes — full primary text read (PDF pages 1–6), ancillary files downloaded, SHA-256 sums verified against `SHA256SUMS.txt` |

Convention used throughout (same as ours): z(m,n;s,t) is the maximum number of 1s in an m×n
0/1 matrix with no all-ones s×t submatrix, with the s-side inside the part of size m. The paper
abbreviates z(m,n) = z(m,n;3,3).

## 2. What the paper claims (verbatim where it matters)

Abstract: "We record five improved lower bounds for Zarankiewicz numbers with s = t = 3:
z(13,19;3,3) ≥ 118, z(14,19;3,3) ≥ 126, z(16,18;3,3) ≥ 136, z(14,20;3,3) ≥ 126,
z(16,19;3,3) ≥ 136. The first three are certified by explicit K_{3,3}-free 0/1 matrices; the
last two follow from the second and third by monotone padding. Compared with the lower bounds
compiled in Figure 2 of [2], namely 114, 121, 130, 125 and 132, the improvements are +4, +5,
+6, +1 and +4 respectively."

The paper is explicit that its contribution is "deliberately narrow": five lower bounds, each
with a witness "that a reader can verify in under a second on a laptop"; "We make no claim of
novelty for the search method, which is a standard kicked-greedy local search, and no claim
about upper bounds."

### Theorem 1
"There exist K_{3,3}-free 0/1 matrices realising z(13,19;3,3) ≥ 118, z(14,19;3,3) ≥ 126,
z(16,18;3,3) ≥ 136."
Proof is by exhibiting A_{13×19}, A_{14×19}, A_{16×18} (Appendix A of the paper; Appendix A
below) and checking every one of the C(m,3)·C(n,3) row-triple/column-triple pairs:
277,134, 352,716 and 456,960 checks respectively.

### Lemma 2 (Monotone padding) — the only lemma in the paper
"For all positive integers m,n,s,t,
    z(m,n;s,t) ≤ z(m,n+1;s,t)   and   z(m,n;s,t) ≤ z(m+1,n;s,t)."
Proof (verbatim): "Let A be an m×n 0/1 matrix with no all-ones s×t submatrix and with
z(m,n;s,t) ones. Form A′ by appending one all-zero column, so A′ is m×(n+1) and has the same
number of 1s. Let S be a set of s rows and T a set of t columns of A′. If T avoids the new
column then the corresponding submatrix of A′ is a submatrix of A and hence is not all-ones;
if T contains the new column then that column contributes an entry 0, so the submatrix is
again not all-ones. Thus A′ is a valid m×(n+1) configuration, giving
z(m,n+1;s,t) ≥ z(m,n;s,t). Appending an all-zero row gives the second inequality."

### Corollary 3
"z(14,20;3,3) ≥ z(14,19;3,3) ≥ 126 and z(16,19;3,3) ≥ z(16,18;3,3) ≥ 136."
The witnesses are A_{14×19} and A_{16×18} with one all-zero column appended. The paper
stresses this "is bookkeeping, not search: it costs no computation, and it improves the
compiled lower bounds for those two cells only because the compilation predates Theorem 1".

### Remark 4 (chronology of z(13,17;3,3) = 110)
"A fourth witness produced by the same program in July 2026 gives z(13,17;3,3) ≥ 110,
matching the upper bound z(13,17;3,3) ≤ 110 of Table 4 of [3] and hence the exact value. That
matrix was found on 2026-07-03/04 and was communicated to, and verified by, the authors of [2]
in July 2026. The value z(13,17;3,3) = 110 subsequently appears in [5]. We record the
chronology only for context and claim no priority: [5] is the published source for that
value, and we do not restate it as a result of this note." ([5] = Hou 2608.08549.)

### Table 1 (exact copy) — "The five cells. All values for s = t = 3."

| Cell (m,n) | Previous LB | New LB | Best known UB | Gap | Source of the UB |
|---|---|---|---|---|---|
| (13,19) | 114 | **118** | 125 | 7 | Fig. 2 of [2] |
| (14,19) | 121 | **126** | 135 | 9 | Fig. 2 of [2] |
| (16,18) | 130 | **136** | 140 | 4 | Table 4 of [3] |
| (14,20) | 125 | **126** | 140 | 14 | Fig. 2 of [2] |
| (16,19) | 132 | **136** | 152 | 16 | Fig. 2 of [2] |

"Previous LB" is the lower bound compiled in Figure 2 of Bhan–Nobili–Langer [2]. "Best known
UB" is "the smallest published upper bound we are aware of": Fig. 2 of [2] for four cells, and
for (16,18) the tighter Collins et al. Table 4 value 140 ("tighter than the value 146 appearing
for that cell in the compilation of Figure 2 of [2]"). Gap = UB − new LB.

**No full best-known table is given.** The paper only tabulates these five cells. For the
surrounding cells it points to Fig. 2 of [2] (LBs and UBs), Table 4 of [3] (UBs), Davies–Gill–
Horsley [4] (LP upper bounds), Hou [5] and Afrasyab [1] (exact values, August 2026).

### Overlap with Hou and Afrasyab
"Two further papers appeared in August 2026, after the constructions reported here were found
and communicated: Hou [5] and Afrasyab [1] close or improve a number of cells. Neither paper
addresses any of the five cells treated here: Hou's results concern (12,18), (13,17),
(13,18), (14,17), (14,18), (15,17) and (15,18), and Afrasyab's concern (12,n) for
18 ≤ n ≤ 22 together with (13,22), (13,18), (14,17), (14,18), (15,17), (15,18) and (16,17)."

### Degree profiles reported by the paper (Section 4)
- A_{16×18}: rows 9^8 8^8, columns 9^2 8^8 7^6 6^2 ("notably regular").
- A_{13×19}: rows 10^6 9^2 8^5, columns 8^1 7^8 6^6 5^2 4^2.
- A_{14×19}: rows 11^1 10^3 9^5 8^5, columns 8^4 7^9 6^2 5^3 4^1.
(Exponents are multiplicities. All confirmed by our recomputation, Section 6.)

## 3. Search method (Section 3 of the paper)

"The three matrices of Theorem 1 were found by a kicked-greedy local search over 0/1 matrices
of fixed shape: a greedy fill that adds a 1 only where it creates no all-ones 3×3 submatrix,
run to saturation, followed by a randomised 'kick' that deletes a small random set of 1s and
re-saturates, with the incumbent retained on non-improvement. The search is classical: no
language model, no learned component, and no problem-specific algebraic input. The objective
is the exact count of 1s, and K_{3,3}-freeness is enforced as a hard constraint at every step
rather than penalised, so every intermediate state — and in particular every state that is
ever recorded — is a valid witness."

Orchestration: "The search ran inside an autonomous research system (NOVA) that selected the
target cells, ran the searches, and packaged the results". **Cell-selection rule (the one
methodological idea worth keeping):** "it targets cells whose compiled lower bound is
dominated by the monotone floor implied by a smaller cell (Lemma 2), on the grounds that such
cells are the ones where the incumbent construction has most visibly not converged." The
paper says this rule "is the reason two of the three cells were attacked at all".

Provenance (verbatim from the paper and `witnesses.txt`; the arXiv HTML rendering garbles the
first seed as "22" — PDF and ancillary file both say 2):
- A_{13×19}: found 2026-07-05, seed 2, 215,624 kicks, wall-clock budget 600 s for the cell.
- A_{16×18}: found 2026-07-05, seed 0, 130,287 kicks, same budget.
- A_{14×19}: found during a campaign of 2026-07-03/04, seed 7; kick count not recorded.
"A wall-clock-budgeted stochastic search is not bit-reproducible from a seed alone", so
re-running is not claimed to reproduce the matrices. The search program is deliberately not
shipped (README: "shipping it would invite a reproduction attempt that is not expected to
succeed and would settle nothing either way").

Verification: exhaustive enumeration of all C(m,3)C(n,3) pairs, ones recounted from the
matrix, by three code-independent implementations (array-based, column-pair/column-triple set
arithmetic, plain triple-nested loop). The Bhan–Nobili–Langer authors re-verified all four
matrices (three + the 13×17) in July 2026. The shipped `verify_13x19.py` etc. are "the scripts
written at the time of discovery and are shipped unaltered", hence compare against an internal
"honest bar" of 115, 124, 132 = max(published LB, monotone floor) and print "APPARENT crossing;
NOT a claim" — cosmetic, documented in the README. The padded verifiers delete the zero column
and re-verify the core, "so that Lemma 2 is checked rather than assumed".

## 4. Which cells changed, and what that means for attacking from above

The five cells and their gaps after this paper:

| Cell | LB now | UB quoted | Gap | Remarks (our reading) |
|---|---|---|---|---|
| (16,18) | 136 | 140 (Collins Table 4) | **4** | Most attackable. In Collins et al. Table 4 the entry 140 is *undecorated*, i.e. obtained "by using Lemmas 2, 3 and 4, and without exhaustive enumeration" — a pure counting bound, so it is the kind of UB a SAT attack can plausibly tighten. Closing the cell means refuting w = 137, 138, 139, 140 (or just w = 140 to move the UB to 139). |
| (13,19) | 118 | 125 (Bhan Fig. 2) | 7 | Outside Collins Table 4 (which stops at n = 18). Monotone ceiling from Collins: z(13,19) ≤ z(13,18) + 13 = 116 + 13 = 129, so the quoted 125 is strictly better than the trivial ceiling. |
| (14,19) | 126 | 135 (Bhan Fig. 2) | 9 | Ceiling z(14,18) + 14 = 124 + 14 = 138 > 135. |
| (14,20) | 126 | 140 (Bhan Fig. 2) | 14 | LB is only the padding of (14,19); the gap is dominated by the weak UB. |
| (16,19) | 136 | 152 (Bhan Fig. 2) | 16 | Same; ceiling z(16,18) + 16 = 156 > 152. |

Consequences for target selection (inference):
1. The three searched cells all moved the LB by 4–6, which *shrinks* the window a SAT-from-
   above attack must close. (16,18) with gap 4 is the standout: a single successful refutation
   at w = 140 already improves the published table, and the UB there is a soft counting bound.
2. The padded cells (14,20), (16,19) show the LB side has simply not been worked; their large
   gaps say nothing about difficulty from above, and their UBs are inherited compilations. We
   should not choose them as first targets.
3. Table 1's UB column is "the smallest published upper bound we are aware of", not a claim of
   optimality — Davies–Gill–Horsley LP bounds and the monotone ceilings from the new exact
   values in Hou/Afrasyab/Padhi should be re-derived before fixing any w (open question 1).
4. Remark 4 closes (13,17) = 110 exactly. Collins Table 4 had (13,17) ≤ 110 as an undecorated
   counting bound that turned out to be tight — a warning that "undecorated" does not mean
   "loose".

## 5. Numbers to keep

- Triple checks per witness: 277,134 (13×19), 352,716 (14×19), 456,960 (16×18).
- Improvements: +4, +5, +6, +1, +4 over Fig. 2 of Bhan–Nobili–Langer.
- Search budget: 600 s wall-clock per cell; 1.3×10^5–2.2×10^5 kicks in that budget.
- Densities of the witnesses (ours): 0.478, 0.474, 0.472.

## 6. Our independent verification (not from the paper)

Script: scratchpad `lit/saurabh_anc/verify_all.py` (stdlib only; parses `witnesses.txt`,
recomputes ones, checks max row-triple codegree ≤ 2, recomputes profiles and counting slacks).
The paper's own `verify_13x19.py` and `verify_padded_16x19.py` also ran here with exit 0.
SHA-256 of `README.txt`, `witnesses.txt`, `verify_13x19.py`, `verify_padded_16x19.py` matched
`SHA256SUMS.txt`.

| Witness | ones | K33-free | max row-triple codeg | row triples at codeg 2 | Σ_i C(r_i,3) / 2C(n,3) | Σ_j C(c_j,3) / 2C(m,3) | row-pair codeg hist | col-pair codeg hist |
|---|---|---|---|---|---|---|---|---|
| 13×19 | 118 | yes | 2 | 220/286 = 76.9% | 1168/1938 (60.3%) | 484/572 (84.6%) | {3:4, 4:64, 5:10} | {0:1, 1:6, 2:52, 3:76, 4:36} |
| 14×19 | 126 | yes | 2 | 276/364 = 75.8% | 1225/1938 (63.2%) | 613/728 (84.2%) | {3:8, 4:72, 5:11} | {0:1, 1:6, 2:38, 3:76, 4:50} |
| 16×18 | 136 | yes | 2 | 360/560 = 64.3% | 1120/1632 (68.6%) | 866/1120 (77.3%) | {2:3, 3:25, 4:89, 5:3} | {1:1, 2:20, 3:58, 4:73, 5:1} |
| 14×20 (padded) | 126 | yes | 2 | 276/364 | 1225/2280 (53.7%) | 613/728 (84.2%) | same as 14×19 | as 14×19 plus 19 zero-codegree pairs |
| 16×19 (padded) | 136 | yes | 2 | 360/560 | 1120/1938 (57.8%) | 866/1120 (77.3%) | same as 16×18 | as 16×18 plus 18 zero-codegree pairs |

Profiles (sorted vectors, ours — agree with the paper's multiplicity notation):
- 13×19: rows [10,10,10,10,10,10,9,9,8,8,8,8,8]; cols [8,7,7,7,7,7,7,7,7,6,6,6,6,6,6,5,5,4,4].
- 14×19: rows [11,10,10,10,9,9,9,9,9,8,8,8,8,8]; cols [8,8,8,8,7,7,7,7,7,7,7,7,7,6,6,5,5,5,4].
- 16×18: rows [9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8]; cols [9,9,8,8,8,8,8,8,8,8,7,7,7,7,7,7,6,6].

Reading of these numbers (inference):
- The Kővári–Sós–Turán counting inequality (Σ_j C(c_j,3) ≤ (s−1)·C(m,3) = 2·C(m,3), and its
  row dual) is *far from tight* on the extremal witnesses: 77–85 % of the column-side budget
  and 60–69 % of the row-side budget is used. So the KST prune on its own cannot kill the
  near-extremal profiles; it only removes very unbalanced ones. Anything that closes a gap
  must come from finer arguments (pair codegrees, Gale–Ryser-type feasibility, deletion
  recursions) or from the SAT solver.
- Pair structure is where the witnesses are tight: row-pair codegrees are concentrated on 4
  (with 3s and 5s), column-pair codegrees on 3–4; 64–77 % of row triples sit at the maximum
  allowed codegree 2. Profile-level counting sees none of this; it is visible only to pair-
  level constraints.
- Extremal profiles are near-regular (16×18 is exactly 9^8 8^8). Survivor sets after any
  sound prune will be concentrated around near-regular partitions; the tails are what prunes
  remove.

## 7. Case counts at these cells (ours; same filter as `profiles.py`)

Counted with scratchpad `lit/saurabh_anc/count_cases.py`. "KST-pass" applies
Σ C(w,3) ≤ 2·C(other side,3) to one side's partition; "[2,m]" is the `profiles.py` convention
that column weights ≤ 1 are excluded (a WLOG *addition* argument, see Section 8, not a prune).

| cell, w | col partitions [0,m] all / KST-pass | col partitions [2,m] all / KST-pass | row partitions [0,n] all / KST-pass |
|---|---|---|---|
| (13,19) w=118 (LB) | 5,128,880 / 3,647 | 545,189 / 2,963 | 5,128,880 / 278,328 |
| (13,19) w=119 (LB+1) | 5,164,564 / 2,291 | 568,836 / 1,965 | 5,164,564 / 231,303 |
| (13,19) w=125 (UB) | 5,229,421 / 27 | 706,309 / 27 | 5,229,421 / 55,480 |
| (14,19) w=126 (LB) | 11,374,098 / 7,503 | 1,326,148 / 6,165 | 11,374,098 / 328,730 |
| (14,19) w=127 (LB+1) | 11,466,968 / 4,838 | 1,380,450 / 4,172 | 11,466,968 / 267,740 |
| (14,19) w=135 (UB) | 11,699,253 / 8 | 1,795,986 / 8 | 11,699,253 / 27,788 |
| (16,18) w=136 (LB) | 28,866,217 / 30,169 | 4,426,457 / 25,012 | 28,866,217 / 158,751 |
| (16,18) w=137 (LB+1) | 29,111,193 / 20,668 | 4,585,302 / 17,831 | 29,111,193 / 120,146 |
| (16,18) w=140 (UB) | 29,658,715 / 5,410 | 5,054,303 / 5,156 | 29,658,715 / 45,980 |
| (14,20) w=126 (LB) | 17,101,221 / 45,318 | 1,494,920 / 27,497 | 17,101,221 / 1,409,740 |
| (14,20) w=127 (LB+1) | 17,371,226 / 33,593 | 1,570,356 / 21,708 | 17,371,226 / 1,234,561 |
| (14,20) w=140 (UB) | 19,159,798 / 10 | 2,592,398 / 10 | 19,159,798 / 76,840 |
| (16,19) w=136 (LB) | 46,315,383 / 186,399 | 5,383,899 / 116,743 | 46,315,383 / 1,180,192 |
| (16,19) w=137 (LB+1) | 47,070,855 / 143,833 | 5,628,599 / 94,859 | 47,070,855 / 991,162 |
| (16,19) w=152 (UB) | 52,925,886 / 42 | 9,346,097 / 42 | 52,925,886 / 10,650 |

Observations (inference):
- KST is enormously effective on the *column* side (whichever side has the larger 2·C(·,3)
  budget relative to its sums) and weak on the other side: at (16,18), w = 137 it leaves
  17,831 column partitions but 120,146 row partitions. Tan-style row×column partition *pairs*
  therefore number in the 10^9 range before pair-level pruning — the case explosion that the
  thesis's evolved prunes have to attack is pair-level, not single-side.
- Attacking from the top is cheap in case count: at w = UB there are 8–42 KST-surviving
  column partitions for four of the cells (5,410 for (16,18) because its UB is tight). A
  first campaign at w = UB is a good calibration run: few cases, each near-infeasible.
- The counts at LB+1 are the real cost of *closing* a cell, and they grow by roughly 3–4×
  per unit of w decrease near the LB.

## 8. Relevance to the ZarPrune / OpenEvolve design

1. **Known witnesses are a free, exact unsoundness oracle for evolved prunes.** For every
   w ≤ 118 the vector profile of A_{13×19} is *inhabited* by a valid matrix, so any candidate
   `kill` that fires on it is unsound at that w — the same logic as `Demo.notDescending_unsound`
   / `badA`. The evaluator should run each candidate's `kill` on the profiles of all known
   witnesses at or below the target w (Appendix A, plus Bhan et al.'s and Hou's matrices)
   *before* spending a Lean elaboration, and score a hit as a hard reject. Keep this in
   Python: a kernel `decide` over 277,134 triples per witness is not realistic without
   `native_decide`, which the gate forbids.
2. **The padded witnesses are a specific trap.** A_{14×20} has a zero column and is valid at
   w = 126. So "kill any profile with a zero column" (or ≤ 1 column, the `profiles.py` WLOG) is
   *not* a prune; it is an addition argument (some equivalent case survives) exactly like
   sorting. It becomes a sound prune only when paired with a certified bound for the smaller
   cell — Lemma 2 read contrapositively (Section 9, lemma P2). This is the cleanest example of
   a prune that needs an *external hypothesis*, which argues for extending `Prune P` (or a
   wrapper) with a context of already-certified bounds Z(m', n') that soundness proofs may
   cite.
3. **Cell-selection heuristic transfers.** NOVA attacked cells whose LB was dominated by the
   monotone floor. The upper-bound analogue: prefer cells whose UB is *not* below the monotone
   ceiling min(z(m−1,n)+n, z(m,n−1)+m) or is undecorated (counting-only) in Collins Table 4
   — those are the loose ones. (16,18) ≤ 140 qualifies on the second criterion.
4. **Reward shaping.** Gap after this paper is 4 at (16,18); each unit of w costs ~3–4× more
   KST-surviving cases. A reward of the form "cases killed, weighted by estimated per-case
   solver cost" should use the w-dependence from Section 7, not raw counts, or the search will
   over-reward prunes that only bite far above the true value.
5. **Profiles vs. partitions.** Our `Profile m n` is a vector; the harness will enumerate
   sorted partitions. The witness data confirm the paper works in sorted multiplicity form
   (10^6 9^2 8^5); `profileOf A` for the printed matrix is the unsorted vector, and the two
   coincide only up to permutation — `cover` must account for that (already flagged in
   `lean/README.md` Scope).
6. **Verification style to imitate.** Stdlib-only, self-contained verifiers with the matrix
   embedded, three code-independent checkers, hashes over the ancillary bundle, and a
   verifier that checks the padding lemma rather than assuming it. Our `sat_attack/verify_
   witness.py` already follows the first pattern; the thesis's refutation artefacts (LRAT +
   cover certificate) should follow the same "no trusted label" discipline.

## 9. Pruning lemmas extracted or derived (with Lean-4 provability over ZarPrune)

Only P1 is in the paper. P2–P4 are our contrapositive/derived forms; P5 is not a lemma but a
test. Provability is judged against the Mathlib-free ZarPrune core (`sumFin`/`allFin` over
`Fin`, `HasKst` as increasing index tuples, `Prune P` with `kill : Profile → Bool` and
`sound : ∀ A, kill (profileOf A) = true → ¬ Valid P A`).

**P1 — Monotone padding (paper's Lemma 2).**
Hypotheses: m, n, s, t ≥ 1. Statement: z(m,n;s,t) ≤ z(m,n+1;s,t) and z(m,n;s,t) ≤ z(m+1,n;s,t).
ZarPrune form: for A : Mat m n with ¬HasKst ⟨m,n,s,t,w⟩ A, the extension
A' : Mat m (n+1) obtained by *prepending* a zero column, A' i j := Fin.cases false (A i) j
(i.e. A' i 0 = false, A' i (succ j') = A i j'), satisfies ¬HasKst ⟨m,n+1,s,t,w⟩ A' and
weight A' = weight A. (The paper appends the column at the end; prepending is the same
statement up to relabelling and matches `Sum.lean`, whose `sumFin_succ` peels off index 0:
`sumFin (k+1) f = f 0 + sumFin k (fun i => f i.succ)`.)
Lean difficulty: easy–moderate (est. 40–80 lines). The HasKst part: an increasing C : Fin t →
Fin (n+1) with all entries true cannot hit column 0 (entry is false), so every C b is a
`Fin.succ` and C' b := (C b).pred is still `Incr` with A (R a) (C' b) = true. The weight part:
via `weight_eq_sum_colSum`, colSum A' 0 = 0 and colSum A' (succ j) = colSum A j, so
`sumFin_succ` gives weight A' = 0 + weight A. Not itself a `Prune` (it transfers lower
bounds), but it is the engine of P2–P3.

**P2 — Zero-column reduction prune (contrapositive of P1; inferred).**
Hypotheses: a certified bound H : ∀ B : Mat m (n−1), ¬HasKst ⟨m,n−1,s,t,w⟩ B → weight B < w
(e.g. the conclusion of `upper_bound_of_cover` for the smaller cell), n ≥ 1.
kill pf := !allFin n (fun j => decide (pf.col j ≠ 0)). Soundness: if col j = 0 in A, deleting
column j gives B with weight B = weight A ≥ w and B K_{s,t}-free (an increasing tuple into
Fin (n−1) composed with the order-preserving skip embedding Fin (n−1) → Fin n is increasing),
contradicting H. Lean difficulty: moderate (est. 100–150 lines). Needs (i) the skip embedding
`Fin.succAbove`-style with an `Incr`-preservation lemma, (ii) a sum-splitting lemma
`sumFin n f = f j + sumFin (n−1) (f ∘ skip j)` that `Sum.lean` does not yet have (by induction
on n, splitting on j = 0, where the j = 0 case is exactly `sumFin_succ`), (iii) rewriting
`weight` via `weight_eq_sum_colSum`. Because `Profile` is an unsorted vector, the zero column
can be at any index, so the arbitrary-j deletion is needed; a "j = 0 only" version would be
easy (`sumFin_succ` directly, the P1 argument reversed) but would not be a prune on our
profiles. The hypothesis H must be threaded in as an argument (a `Prune` parameterised by a
context of certified smaller-cell bounds).

**P3 — Lightest-column deletion prune (Guy-type recursion; inferred, generalises P2).**
Hypotheses: as P2 with certified Z := z(m,n−1;s,t) upper bound, i.e. H : ∀ B : Mat m (n−1),
¬HasKst → weight B ≤ Z. kill pf := decide (∃ j, sumFin n pf.col − pf.col j > Z) (equivalently
w − min_j c_j > Z when Σ col = w). Soundness: deleting column j leaves weight A − c_j ≤ Z ones.
Same Lean skeleton as P2 plus one subtraction inequality; moderate. The k-column version
(w − sum of the k smallest c_j > z(m,n−k)) needs subset deletion and is hard in the current
core (no `Finset`); only fixed small k by iterated single deletion is realistic.

**P4 — Gale–Ryser necessary condition (pair-level; inferred, not in the paper).**
Hypotheses: none beyond a matrix. Statement: for every k-subset S of rows,
Σ_{i∈S} r_i ≤ Σ_j min(c_j, k). kill pf := decide (∃ k, (sum of the k largest pf.row) >
Σ_j min(pf.col j, k)). Soundness: Σ_{i∈S} r_i = Σ_j |{i ∈ S : A i j}| and each summand is ≤
min(c_j, k). Lean difficulty: moderate–hard in the Mathlib-free core (sums restricted to a
subset and a "k largest" selection in the kill predicate; the fixed-S version is a `sumFin_swap`
plus per-column bound). This is the first prune that looks at both sides of the profile at
once, which Section 7 shows is where the case explosion lives. Listed here because the paper's
witness profiles are exactly the regression data such a prune must not kill.

**P5 — Witness-inhabitation test (not a lemma; a negative check).**
For each known witness A with weight A = z_A at cell (m,n), and any candidate kill with target
w ≤ z_A: kill (profileOf A) = true ⟹ candidate is unsound. In Lean this is the
`notDescending_unsound` pattern (¬∃ p : Prune P, p.kill = candidate) and is trivially provable
*given* a decidable K_{s,t}-freeness proof of A; the cost is that proof, which is a 10^5–10^6
triple enumeration and belongs in the Python evaluator, not in kernel `decide`.

## 10. Open questions raised for the thesis

1. Are 125, 135, 140, 152 really the best published UBs for (13,19), (14,19), (14,20),
   (16,19)? Table 1 hedges ("we are aware of"); Davies–Gill–Horsley's LP bounds and the
   monotone ceilings from the August 2026 exact values (Hou, Afrasyab, Padhi) need to be
   re-derived before we fix a target w. Collins Table 4 stops at n = 18, so these four cells
   have never had an exhaustive-style UB.
2. Is z(16,18;3,3) = 136, 137, 138, 139 or 140? Gap 4 with a counting-only UB is the best
   candidate cell for the first real SAT-from-above campaign; Section 7 gives the case counts.
3. The paper reports 64–77 % of row triples at codegree 2 in extremal witnesses. Is
   "fraction of the 2·C(m,3) triple budget consumed" (profile-computable as Σ_j C(c_j,3) /
   2C(m,3)) a monotone proxy for per-case SAT hardness, and in which direction? Needs
   calibration on Tan's data.
4. The paper never reports failed cells or how many cells NOVA attacked; the selection
   heuristic's hit rate is unknown, so we cannot yet use "monotone-floor dominated" as a
   quantitative prior.
5. Would a prune context of certified smaller-cell bounds (needed by P2/P3) be better modelled
   as an argument to `Prune`, or as a separate `Ctx` structure with its own soundness ledger?
   This affects how `upper_bound_of_cover` chains across cells.

## Appendix A — The five witnesses (from the paper's Appendix A / `witnesses.txt`)

One row per line, n characters in {0,1}. Reproduced so they can serve as regression data
(Section 8 item 1). Verified here: shape, entries, ones count, K_{3,3}-freeness.

A_{13×19}, 118 ones (rows 10^6 9^2 8^5; cols 8^1 7^8 6^6 5^2 4^2):
```
1010010100111001000
0000100110101101100
0101110001110011100
0010100001111000011
0110111100100100010
1001101000011101010
1000110101000001011
1111000011100101001
1111100100001010101
1010110010010100100
0010001111010011110
1000011010101010111
0100011111011100001
```

A_{14×19}, 126 ones (rows 11^1 10^3 9^5 8^5; cols 8^4 7^9 6^2 5^3 4^1):
```
1110110100110010000
0010110101001001110
0001010110110000111
1000100011011110011
0100000000111011110
1101010110001101000
1111001001011000101
0110011010101000010
0001111000101110100
0010011010010101100
0101111000010001010
0011100010111001000
1011001100100111011
0100101111000011101
```

A_{16×18}, 136 ones (rows 9^8 8^8; cols 9^2 8^8 7^6 6^2):
```
100011101010010001
010000111100011101
110001011101100000
100000010111001011
101100000001111101
100110111000101010
110100001011010110
010010010011101100
110011000100100111
001001011001000111
011101010010010001
001111001110001100
011011100001011010
011000001010101011
000110100111100001
001000110110110110
```

A_{14×20} (z(14,20;3,3) ≥ 126) is A_{14×19} with the character `0` appended to every row;
A_{16×19} (z(16,19;3,3) ≥ 136) is A_{16×18} with `0` appended to every row.

## Appendix B — Verification script used here

```python
# verify_all.py  (stdlib only) -- parses witnesses.txt, recounts, checks K33-freeness via
# row-triple codegree <= 2, recomputes profiles and KST slack.
import re
from itertools import combinations
from math import comb
from collections import Counter
txt = open("witnesses.txt").read()
for name, body in re.findall(r"# BEGIN (\S+).*?\n(.*?)# END \1", txt, re.S):
    rows = [l.strip() for l in body.splitlines() if l.strip() and not l.startswith("#")]
    m, n = len(rows), len(rows[0]); A = [[int(c) for c in r] for r in rows]
    rs = [sum(r) for r in A]; cs = [sum(A[i][j] for i in range(m)) for j in range(n)]
    R = [frozenset(j for j in range(n) if A[i][j]) for i in range(m)]
    maxcod = max(len(R[a] & R[b] & R[c]) for a, b, c in combinations(range(m), 3))
    print(name, m, n, sum(rs), "K33-free" if maxcod <= 2 else "K33 FOUND",
          sum(comb(r, 3) for r in rs), 2*comb(n, 3), sum(comb(c, 3) for c in cs), 2*comb(m, 3))
```

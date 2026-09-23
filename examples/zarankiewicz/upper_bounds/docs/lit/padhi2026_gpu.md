# Padhi (2026) — "New Exact Values and Improved Lower Bounds for Zarankiewicz Numbers z(m,n;3)"

Literature note for the MEng thesis *Discovering upper bounds for Zarankiewicz numbers* (Jay Bhan).
Written 2026-09-21. Slug: `padhi2026_gpu`.

## 0. Bibliographic record and provenance

| Field | Value |
|---|---|
| Author | Jaideep Padhi, "Independent Researcher" (as printed on the note) |
| Title | New Exact Values and Improved Lower Bounds for Zarankiewicz Numbers z(m,n;3) |
| Venue | SSRN preprint, abstract id 6960039, DOI 10.2139/ssrn.6960039 |
| Dates | Note dated "June 2026"; PDF created 2026-06-17 (ReportLab); Crossref DOI record created 2026-07-29 |
| Length | 3 pages, 5 references ([Bie24] Kissat, [CRWR16], [GHO00], [KST54], [Tan22]) |
| Copy used | `/Users/jaybhan/Downloads/ssrn-6960039.pdf` (the copy attached to the proposal); scratch copy + text at `.../scratchpad/lit/padhi2026_ssrn6960039.pdf`, `padhi.txt`, page renders `padhi_pg-2.png`, `padhi_pg-3.png` |
| Access | SSRN itself is Cloudflare-gated (403 for WebFetch/curl); the full text was read from the local PDF. Crossref returns the abstract. |
| Code / certificates | **Not released.** The note says `zara_cdcl.c`, `zara_cdcl_gen.c`, `zara_v5.cu` and certificate files "will be available on GitHub following the publishing of the full paper". As of 2026-09-21 a GitHub repo/code search for `zara_cdcl` finds nothing, and no "full paper" exists on arXiv (arXiv API: zero results for au:Padhi AND Zarankiewicz). |
| Metadata warning | OpenAlex's record for this DOI is mis-merged with Collins–Riasanovsky–Wallace–Radziszowski 2016 (it reports RIT/Purdue affiliations, 12 references and 7 citations that all belong to CRWR16). Do not cite OpenAlex for this paper. OpenAlex also links Zenodo DOI 10.5281/zenodo.20788643; a DuckDuckGo snippet for that DOI reads "Preprint. We compute 44 new exact values and 31 improved lower bounds ... certified via a GPU-accelerated simulated annealing solver and a custom BCP-accelerated backtracking CDCL prover", which is almost certainly Padhi's Zenodo deposit of the same note; the record now returns HTTP 410 Gone (deleted) and OpenAIRE reports "publication not found". |
| Citations by others | None found. Saurabh (arXiv:2608.26603), Hou (2608.08549), Afrasyab (2608.08154), dfield's repo and Bhan et al. (2605.01120) do not cite it. |

**Bottom line (my assessment, details in §4):** the note is a 3-page announcement with no certificates and no code. Of its 42 printed "exact values", 29 were already exact in Tan 2022 / Guy 1969, **6 are refuted by publicly available witness matrices** (one of them, z(12,12;3)=71, is refuted by a matrix I decoded from Tan 2022 and re-verified myself), 3 agree with later independently certified results, and 4 remain unverifiable claims. The table is also internally inconsistent in three places. Treat every upper bound in it as unproven and every lower bound as an unverified claim. Its value to the thesis is (a) as a documented instance of exactly the failure mode the proposal warns about, and (b) a few concrete engineering data points (GPU SA evaluator, solver cascade, "partition counts overwhelm" as the wall at m ≥ 15).

## 1. What the note claims (verbatim where it matters)

Abstract: "We present 44 new exact values and 31 improved lower bounds for the Zarankiewicz number z(m,n;3), the maximum number of ones in an m×n binary matrix containing no 3×3 all-ones submatrix. Our results extend the known table from m ≤ 9 to m ≤ 14 for exact values, and provide new lower bounds up to m = 18. In particular, we establish z(17,17;3) ≥ 138, contributing a new lower bound for the 15th term of OEIS sequence A001198 (Zarankiewicz's problem k3(n)), which previously had 14 known terms ending at z(16;3) = 129. All constructions and proofs are certified."

Two factual slips already in the abstract:
* "known table from m ≤ 9": CRWR16 Table 4 (the note's own reference) covers 6 ≤ m ≤ n ≤ 18, and Tan 2022 (also cited) gives a corrected table of z_3(m,n) for m ≤ 16, n ≤ 23 with dozens of exact values for m up to 16.
* "z(16;3) = 129": OEIS A001198 lists k_3(n) = the least number of ones forcing a 3×3 all-ones submatrix, i.e. k_3(n) = z(n;3)+1. A001198(16)=129 means z(16;3)=128 (Tan 2022 Table 3, bold; CRWR16 Table 4 "128*"). The note conflates the two conventions by one throughout §3.3.

## 2. The pipeline (Section 2 of the note, quoted)

### 2.1 Lower bounds: GPU simulated annealing
"We implement a parallel simulated annealing solver in CUDA targeting an NVIDIA RTX 6000 Ada GPU (48 GB VRAM, sm_89 architecture). The solver maintains 65,536 independent matrix states undergoing random bit-flip moves with exponential cooling. Violation counting uses the identity

    viols = Σ_{r1,r2,r3} C( |row_r1 ∩ row_r2 ∩ row_r3| , 3 )

computed efficiently via __popc instructions on packed 32-bit row bitmasks. When a zero-violation state with T ones is found, the matrix is recorded as a certified lower bound witness z(m,n;3) ≥ T."

Notes: the identity is correct (it counts all-ones 3×3 submatrices: choose 3 rows, then any 3 of their common columns). Packed 32-bit row masks bound n ≤ 32. The evaluation cost per state is C(m,3) three-way ANDs + popcounts — for m = 18 that is 816 word operations, which is why 65,536 chains fit on one GPU. Nothing about the SA schedule, move acceptance, restarts or run lengths is given. The word "certified" here means "the final matrix was re-checked for violations", not a proof certificate.

### 2.2 Upper bounds: `zara_cdcl` (custom backtracking) and Kissat
"To certify z(m,n;3) ≤ T−1 (proving no K3,3-free matrix with T ones exists), we use a custom C solver zara_cdcl. The solver encodes the problem via 9-literal clauses (one per triple of rows and columns), maintains BCP queues, and exhaustively backtracks over column assignments ordered by the KST partition structure. Key pruning rules include: (i) KST bound: column degree sequences must satisfy ∑_j C(d_j,3) ≤ (s−1)C(n,3); (ii) feasibility: running ones count cannot reach T given remaining columns and row capacities; (iii) BCP propagation: unit propagation forces assignments and detects conflicts early. The solver is parallelized over 128 CPU threads. For cases where the partition count is too large (m ≥ 15), we used the Kissat SAT solver [Bie24] with PySAT cardinality encodings."

Observations:
* The encoding is the naive one: one clause ¬x_{i1 j1} ∨ … ∨ ¬x_{i3 j3} per (row-triple, column-triple), C(m,3)·C(n,3) clauses (14×23: 364·1771 = 644,644 clauses) plus a cardinality constraint "exactly/at least T ones". Same family as Tan 2022 and as our `encodings_zar.py` (which instead uses per-(row-triple, column) auxiliaries with AtMost2 — far fewer clauses).
* Rule (i) as printed is mis-indexed. With **column** degrees d_j (j = 1..n, d_j ≤ m) the Kővári–Sós–Turán double count over row triples gives Σ_j C(d_j,3) ≤ (t−1)·C(m,3) = 2·C(m,3) (every triple of rows is fully covered by at most 2 columns). The printed (s−1)·C(n,3) is the **row-sum** version Σ_i C(r_i,3) ≤ 2·C(n,3) applied to column degrees. If implemented as printed it is *weaker* than the true bound whenever n ≥ m (so sound but prunes less), and would be *unsound* for n < m. Our `profiles.py` uses the correct form (`budget = 2*comb(m,3)` for column weights).
* "Exhaustively backtracks over column assignments ordered by the KST partition structure" is Tan's cube-and-conquer over column-sum partitions. The note does not say what ordering/symmetry assumptions are imposed inside a partition; if any are imposed as *pruning* rather than as justified symmetry breaking, the search is incomplete. This is the exact pitfall `ZarPrune/Demo.lean` (`notDescending_unsound`) guards against, and it is my leading hypothesis for why several of the note's "exact" upper bounds are false (§4).
* Rules (ii) and (iii): (ii) is a capacity/deficit prune (already proved in ZarPrune as `deficit`/`rowCap`/`colCap`); (iii) is ordinary unit propagation — a SAT-level mechanism, not a profile-level prune.
* Kissat is used **only to find constructions** (the two starred cells and the 17×17 witness); no UNSAT result from Kissat is claimed. So every upper bound in the note rests on the unreleased `zara_cdcl`.

### 2.3 Verification section (§4 of the note, quoted)
"Every exact value in Table 1 is independently verifiable. Lower bounds: the explicit matrix is stored in the certificate file; verification requires checking C(m,3)C(n,3) row-column triples, feasible in seconds. Upper bounds: for CDCL-certified cases, the solver performs exhaustive enumeration with sound pruning (complete backtracking). For Kissat-certified cases, the solver output is included. All source code (zara_cdcl.c, zara_cdcl_gen.c, zara_v5.cu) and certificate files will be available on GitHub following the publishing of the full paper."

So: no DRAT/LRAT/VeriPB proof, no cube-cover certificate, no independent checker — "certified" for upper bounds means "our solver said UNSAT". Neither code nor certificate files were ever published.

### 2.4 Runtime data points
* 17×17, 138 ones, satisfiable instance: "certified by Kissat in 2 hours 48 minutes of computation."
* "The principal bottleneck for further progress is the upper bound computation for m ≥ 15, where partition counts overwhelm our CDCL solver." Future work named: cube-and-conquer, GPU-assisted CDCL, and the gaps z(8,18;3), z(8,20;3), z(17;3). (The first two are not gaps: Tan 2022 has z_3(8,18)=77 and z_3(8,20)=84 in bold, i.e. exact.)

## 3. The tables, transcribed

### Table 1 — "New exact values" (42 cells as printed; caption and abstract say 44)
Rows m, columns n. `*` = "Kissat SAT construction; upper bound open".

| m \ n | 9 | 10 | 11 | 12 | 14 | 15 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 30 | | | | | 46 | | 54 | 57 | | | | |
| 6 | | | | | 50 | 53 | 58 | 61 | | 66 | 69 | | 74 |
| 7 | 40 | | 47 | 50 | 56 | | 66 | 69 | | 75 | 78 | | 84 |
| 8 | 45 | | 53 | | | 67 | | ≥77* | | ≥84* | | | |
| 9 | 49 | 54 | 59 | 64 | | | | | | | | | 103 |
| 12 | | | | 71 | | | | 102 | | 102 | 126 | | |
| 13 | | | | | | | 110 | | 120 | 124 | | | |
| 14 | | | | | | | | 115 | | 124 | 134 | 135 | 142 |

"Notable highlights: z(9,23;3) = 103 is the largest certified exact value; z(14,23;3) = 142 is the first exact value for m = 14 with n > 20; z(12,12;3) = 71 extends the square diagonal beyond Collins et al. [CRWR16]."

### Table 2 — "Improved lower bounds z(m,n;3) ≥ v" (29 rows as printed; caption says 31)
(8,23) 94 · (11,13) 80 · (11,14) 84 · (11,16) 92 · (12,15) 96 · (13,16) 107 · (13,18) 116 · (16,21) 148 · (16,22) 147 · (17,17) 138 · (13,21) 129 · (13,22) 135 · (13,23) 138 · (14,24) 143 · (15,18) 132 · (15,19) 134 · (15,21) 141 · (15,22) 145 · (15,24) 155 · (17,18) 142 · (17,19) 146 · (17,20) 150 · (17,21) 155 · (17,22) 159 · (17,23) 165 · (18,18) 139 · (18,20) 151 · (18,21) 162 · (18,23) 172.

## 4. Audit of every claim against the public record

Comparison sources (all read in full or in the relevant tables):
* **Tan22** = Tan, arXiv:2203.02283, Table 3 (z_3(m,n), 3 ≤ m ≤ 16, n ≤ 23; bold = exact, proven by SAT; non-bold = Theorem 2.2 upper bound). I extracted the bold/non-bold status from the PDF fonts (CMBX10 vs CMR10) via `pdftohtml -xml`, so "Tan22 exact" below is by font, not by guess. Values above Guy's solid lines are exact too.
* **CRWR16** = Collins et al., arXiv:1604.01257, Table 4 (upper bounds on z(m,n;3), 6 ≤ m ≤ n ≤ 18; bold = exact — bolding not recoverable from my extraction, so I only use CRWR16 for upper bounds and for the cells other papers cite as exact from it).
* **Bhan26** = Bhan–Nobili–Raghuraman–Langer, arXiv:2605.01120, Figure 2 (top = previously known UB, bottom = their LB witness; 8 ≤ m ≤ 16, 16 ≤ n ≤ 23), read from the rendered page.
* **DGH26** = Davies–Gill–Horsley, arXiv:2411.18842, Table 2 (improved UBs on z(m,n;3,3)).
* **dfield** = github.com/dfield/finite-zarankiewicz-closures (July 2026; DRAT/LRAT + SCIP/VIPR + Lean): Z(9,23)=103, Z(10,21)=106, Z(10,22)=110, Z(10,23)=112, Z(11,19)=106, Z(11,20)=111, Z(11,23)=123, Z(12,23)=134, Z(13,23) ≤ 144.
* **Hou26** = arXiv:2608.08549 (Aug 2026): z(12,18)=108, z(13,17)=110, z(13,18)=116, z(14,17)=118, z(14,18)=124, z(15,17)=126, z(15,18)=132 with explicit witnesses + verifier.
* **Afr26** = Afrasyab, arXiv:2608.08154 (Aug 2026): Z(12,n)=6n for 18 ≤ n ≤ 22, Z(13,22)=137, Z(13,18)=116, Z(14,17)=118, Z(14,18)=124, Z(15,17)=126, Z(15,18)=132, 132 ≤ Z(16,17) ≤ 133.
* **Sau26** = Saurabh, arXiv:2608.26603 (Aug 2026): z(13,19) ≥ 118 (prev 114, UB 125), z(14,19) ≥ 126 (prev 121, UB 135), z(16,18) ≥ 136, z(14,20) ≥ 126 (prev 125, UB 140), z(16,19) ≥ 136; witnesses re-verified independently.
* **Wang26** = Zenodo 10.5281/zenodo.21766509 (Korbot Labs, 3 Aug 2026): z(11,19)=106, z(13,18)=116, z(15,17)=126 with matrices.json + verify.py.
* **numaro** = numaro.tech report NUMARO-2026-008 (3 Jul 2026): 31 LB improvements incl. z(12,21) 116→126, z(10,20)=102, z(11,18)=101.

### 4.1 Table 1, cell by cell

| Cell | Padhi | Status | Evidence |
|---|---|---|---|
| (5,9) 30, (5,15) 46, (5,18) 54, (5,19) 57 | exact | **already known** | Guy 1969 / Tan22 row 5 (all exact) |
| (6,14) 50, (6,15) 53, (6,17) 58, (6,18) 61, (6,20) 66, (6,21) 69, (6,23) 74 | exact | **already known** | Tan22 row 6 (exact); CRWR16 agrees where covered |
| (7,9) 40, (7,11) 47, (7,12) 50, (7,14) 56, (7,17) 66, (7,18) 69, (7,20) 75, (7,21) 78, (7,23) 84 | exact | **already known** | Tan22 row 7 (exact) |
| (8,9) 45, (8,11) 53, (8,15) 67 | exact | **already known** | Tan22 row 8 (exact) |
| (8,18) ≥77*, (8,20) ≥84* | LB only, "UB open" | **already exact** | Tan22 row 8: 77 and 84 in bold (exact). The note's "future gaps" z(8,18;3), z(8,20;3) are closed since 2022. |
| (9,9) 49, (9,10) 54, (9,11) 59, (9,12) 64 | exact | **already known** | Guy/Tan22 row 9 (exact); CRWR16 marks 59*, 64* (unique extremal graphs) |
| (9,23) 103 | exact | consistent; **independently proven later** | Tan22 UB 104, Bhan26 LB 103 (public witness), dfield proves =103 with LRAT (July 2026). Padhi's claim predates dfield but has no certificate. |
| (12,12) 71 | exact | **REFUTED** | z(12;3)=80: Guy 1969, A001198(12)=81, Tan22 bold 80, CRWR16 80. I decoded Tan's listed maximal matrix (code `12 12 DzezVZ3mlq7azAt/57ANveAH`, Tan §5 decoding rule): 80 ones, row sums 7^8 6^4, no three rows sharing ≥3 columns. Padhi's solver "proved" no 12×12 K33-free matrix with 72 ones exists. |
| (12,18) 102 | exact | **REFUTED** | Bhan26 LB 108 (public witness, May 2026); Hou26/Afr26 exact 108 |
| (12,20) 102 | exact | **REFUTED**, and internally inconsistent | Bhan26 LB 113; Afr26 exact 120. Also impossible next to Padhi's own (12,21)=126 (126−102=24 > m=12) and (13,20)=124 (124−102=22 > n=20). |
| (12,21) 126 | exact | consistent; **independently proven later** | Bhan26 116 ≤ · ≤ 127; numaro found 126 (Jul); Afr26 proves 126 (Aug) |
| (13,17) 110 | exact | consistent; **independently proven later** | CRWR16 UB 110; Hou26 explicit 110 witness |
| (13,19) 120 | exact | **unverified** | public: 118 ≤ · ≤ 125 (Sau26, DGH26). Would be new in both directions; no witness or proof available. |
| (13,20) 124 | exact | **unverified** | public: 119 ≤ · ≤ 130 (Bhan26, DGH26) |
| (14,18) 115 | exact | **REFUTED**, and internally inconsistent | Bhan26 LB 124; CRWR16 UB 124; Hou26/Afr26 exact 124. Also contradicts Padhi's own Table 2 entry (13,18) ≥ 116 (a 13×18 witness padded with a zero row is a 14×18 witness). |
| (14,20) 124 | exact | **REFUTED** | Bhan26 LB 125 (public witness; Sau26 later 126) |
| (14,21) 134 | exact | **unverified** | public: 131 ≤ · ≤ 145 |
| (14,22) 135 | exact | **REFUTED** | Bhan26 LB 137; also z(14,22) ≥ z(13,22) = 137 (Afr26) |
| (14,23) 142 | exact | **unverified** | public: 138 ≤ · ≤ 155 (Bhan26 LB, Tan22 UB) |

Tally over the 42 printed cells: 29 already exact before the note; 3 consistent and later proven by others with certificates; **6 refuted by explicit public witnesses**; 4 unverifiable. Internal inconsistencies: (12,20)/(12,21), (12,20)/(13,20), (14,18)/(13,18).

Since a single false UNSAT claim shows the prover is incomplete (or the table is garbled), **none of `zara_cdcl`'s upper bounds can be relied on**, including the 4 unverified ones.

### 4.2 Table 2, cell by cell

| Cells | Status | Evidence |
|---|---|---|
| (8,23) ≥ 94; (11,13) ≥ 80; (11,14) ≥ 84; (11,16) ≥ 92; (12,15) ≥ 96; (13,16) ≥ 107; (15,18) ≥ 132 | **already exact** (not improvements) | Tan22 bold for the first six; (15,18): CRWR16 UB 132 with Bhan26 LB 132 (Hou26 Table 1) |
| (13,22) ≥ 135 | **weaker than public** | Bhan26 LB 137 (May 2026); Afr26 exact 137 |
| (16,22) ≥ 147 | **weaker than public**, and than the note's own (16,21) ≥ 148 | Bhan26 LB 149 |
| (18,18) ≥ 139 | weaker than the note's own (17,18) ≥ 142 | monotonicity in m |
| (13,18) ≥ 116 | +1 over Bhan26 (115); equals CRWR16 UB → would close the cell | later closed with public witnesses by Hou26 (1 Aug), Wang26 (3 Aug), Afr26 (8 Aug). Padhi's witness never released. |
| (13,21) ≥ 129 (+2), (13,23) ≥ 138 (+3), (15,19) ≥ 134 (+2), (15,21) ≥ 141 (+2), (15,22) ≥ 145 (+2), (16,21) ≥ 148 (+1) | **unverified improvements** over Bhan26 | UBs: 135, 144 (dfield), 143, 154, 160, 164 |
| (14,24) ≥ 143, (15,24) ≥ 155 | unverified; outside Bhan26's range | trivially z(14,24) ≥ z(14,23) ≥ 138 |
| (17,17) ≥ 138, (17,18) ≥ 142, (17,19) ≥ 146, (17,20) ≥ 150, (17,21) ≥ 155, (17,22) ≥ 159, (17,23) ≥ 165, (18,20) ≥ 151, (18,21) ≥ 162, (18,23) ≥ 172 | unverified; no prior public LBs located for m ≥ 17 in the sources checked | CRWR16 UBs: z(17;3) ≤ 141, z(17,18) ≤ 148, z(18;3) ≤ 156. Trivial LB z(17;3) ≥ z(16;3) = 128. |

The one claim with real mathematical interest is z(17;3) ≥ 138: CRWR16 note that z(17;3) ≤ 140 would prove b(2,2,3) = 17 and that their computation of it "proved to be too time-consuming". If the 138-witness is genuine the window is 138 ≤ z(17;3) ≤ 141; without the witness it is 128 ≤ z(17;3) ≤ 141.

### 4.3 Which cells "became exact", honestly
From this note alone: **none can be credited.** The cells among Padhi's claims that are exact today are exact because of other, certified work: (9,23)=103 (dfield, LRAT), (12,21)=126 (Afrasyab), (13,17)=110 (Hou), (13,18)=116 (Hou/Wang/Afrasyab witnesses + CRWR16 UB), plus everything that was already in Tan 2022. The 4 remaining "exact" claims — (13,19)=120, (13,20)=124, (14,21)=134, (14,23)=142 — are open cells in the public record: 118–125, 119–130, 131–145, 138–155.

### 4.4 Consolidated status of the cells the note touches (public record, 2026-09-21)
LB/UB pairs; "P:" marks Padhi's unverified claim for reference only.

| Cell | Public LB | Public UB | Exact? | Padhi |
|---|---|---|---|---|
| (9,23) | 103 | 103 | yes (dfield) | =103 ✓ |
| (12,12) | 80 | 80 | yes (Guy/Tan) | =71 ✗ |
| (12,18) | 108 | 108 | yes (Hou/Afr) | =102 ✗ |
| (12,20) | 120 | 120 | yes (Afr) | =102 ✗ |
| (12,21) | 126 | 126 | yes (Afr) | =126 ✓ |
| (13,17) | 110 | 110 | yes (Hou) | =110 ✓ |
| (13,18) | 116 | 116 | yes (Hou/Wang/Afr) | ≥116 (LB matched later) |
| (13,19) | 118 (Sau) | 125 (DGH) | open | P: =120 |
| (13,20) | 119 (Bhan) | 130 (DGH) | open | P: =124 |
| (13,21) | 127 (Bhan) | 135 (DGH) | open | P: ≥129 |
| (13,22) | 137 | 137 | yes (Afr) | ≥135 (weaker) |
| (13,23) | 135 (Bhan) | 144 (dfield) | open | P: ≥138 |
| (14,18) | 124 | 124 | yes (Hou/Afr) | =115 ✗ |
| (14,20) | 126 (Sau) | 140 (DGH) | open | =124 ✗ |
| (14,21) | 131 (Bhan) | 145 (DGH) | open | P: =134 |
| (14,22) | 137 (Bhan) | 150 (DGH) | open | =135 ✗ |
| (14,23) | 138 (Bhan) | 155 (Tan) | open | P: =142 |
| (15,18) | 132 | 132 | yes | ≥132 (not new) |
| (15,19) | 132 (Bhan) | 143 (DGH) | open | P: ≥134 |
| (15,21) | 139 (Bhan) | 154 (DGH) | open | P: ≥141 |
| (15,22) | 143 (Bhan) | 160 (Bhan fig) | open | P: ≥145 |
| (16,21) | 147 (Bhan) | 164 (DGH) | open | P: ≥148 |
| (16,22) | 149 (Bhan) | 169 (DGH) | open | ≥147 (weaker) |
| (17,17) | 128 (trivial) | 141 (CRWR16) | open | P: ≥138 |
| (17,18) | — | 148 (CRWR16) | open | P: ≥142 |
| (18,18) | — | 156 (CRWR16) | open | P: ≥139 |

## 5. Extracted lemmas and algorithms (with Lean-provability notes for ZarPrune)

All statements below are for K_{s,t}-free m×n 0/1 matrices A with row sums r_i (i < m) and column sums c_j (j < n); s = t = 3 in the note. "ZarPrune" refers to `upper_bounds/lean/` (Lean 4.34, no Mathlib; `sumFin` over `Fin`, `HasKst` as increasing index tuples, `Prune P` = decidable `kill` on profiles + `sound`).

**L1. KST column-count prune (Padhi rule (i), corrected form).**
Hypotheses: A has no K_{s,t} (s rows, t columns all ones). Statement: Σ_{j<n} C(c_j, s) ≤ (t−1)·C(m, s). Row dual: Σ_{i<m} C(r_i, t) ≤ (s−1)·C(n, t). Proof: double count pairs (S, j) with S an s-subset of rows, j a column all of whose entries in S are 1; each column contributes C(c_j, s), each S is hit by ≤ t−1 columns. *As printed in the note the RHS is (s−1)·C(n,3) with column degrees, which is mis-indexed (see §2.2).* Lean: this is precisely the README's unproved "next target". Needs a count of s-subsets of `Fin m` with a Fubini swap; without `Finset` it wants a hand-rolled `choose`-counting layer (enumerate increasing index tuples as ZarPrune already does for `HasKst`, prove `card (tuples ⊆ ones of column j) = C(c_j, s)` by induction on the column, then bound the fibre over S by t−1 using `¬HasKst`). Medium-hard; the only genuinely load-bearing counting lemma in the whole pipeline.

**L2. Capacity / deficit feasibility (Padhi rule (ii)).**
Hypotheses: none beyond the profile. Statement: if Σ c_j < w, or Σ r_i ≠ Σ c_j, or some c_j > m or r_i > n, no matrix of weight w has that profile. Lean: already proved (`deficit`, `mismatch`, `rowCap`, `colCap`, folded into `baseline`). The "running count cannot reach T given remaining columns" form is the same lemma applied to a prefix of the column-assignment order; on profiles it is subsumed by `deficit`.

**L3. Witness padding / table consistency (not a prune — a claim checker).**
(a) If A is m×n K_{s,t}-free with weight w, then adding an all-zero row (or column) gives an (m+1)×n (or m×(n+1)) K_{s,t}-free matrix of weight w: z is monotone in m and n. (b) Deleting a column removes at most m ones: z(m,n+1) ≤ z(m,n) + m; likewise z(m+1,n) ≤ z(m,n) + n. Lean: easy over ZarPrune definitions — an increasing index tuple into the padded matrix that avoids the new index is a tuple into A; for (b), `weight` splits as `sumFin` over the deleted column plus the rest and `colCap` bounds the deleted column. These two lemmas mechanically refute three of Padhi's entries ((12,20)/(12,21), (12,20)/(13,20), (14,18)/(13,18)) and, combined with a public witness, refute the other three. A verified `claim_checker` that takes (witness, claimed bound) and produces `False` is cheap and worth having in the harness.

**L4. Violation-count identity (evaluator, lower-bound side).**
#{(R, C) : R a 3-subset of rows, C a 3-subset of columns, A|_{R×C} = all ones} = Σ_{r1<r2<r3} C(|N(r1) ∩ N(r2) ∩ N(r3)|, 3). Consequently A is K_{3,3}-free iff every row triple has at most 2 common columns. Lean: the "iff" form is what `¬HasKst` needs and is essentially definitional under the index-tuple encoding; the exact count identity is a double count (medium). For the harness the decidable check "all row triples have ≤ 2 common columns" is the right verified evaluator, and it is what the GPU kernel computes with `__popc`.

**L5. Encoding correctness (for the `refuted` seam).**
A is K_{3,3}-free ⟺ for every increasing row triple R and column triple C, ¬(∧_{(i,j)∈R×C} x_ij). This is the 9-literal clause set of `zara_cdcl` and of Tan; the auxiliary-variable AtMost2 encoding in `encodings_zar.py` is equisatisfiable. Lean: statement is immediate from `HasKst`; the theorem the pipeline actually needs is "CNF(m,n,w, profile) unsatisfiable → no valid matrix in that case", which requires a small formal CNF semantics — easy for the naive clause set, more work for cardinality encodings (an argument for keeping cardinality out of the trusted base, e.g. fix the profile exactly and drop the global counter).

**Algorithm A1 (GPU SA, LB side).** 65,536 chains × bit-flip moves × exponential cooling; fitness = L4 identity via 32-bit row masks; stop at zero violations with T ones. Orthogonal to the UB pipeline, but the evaluator is the correct design for our witness checker.

**Algorithm A2 (UB cascade).** Enumerate column-sum partitions → prune by L1 (mis-indexed) and L2 → complete backtracking with BCP per partition, 128 threads; for m ≥ 15 hand the instance to Kissat with PySAT cardinality encodings. No proof logging. Demonstrably produced false UNSATs.

## 6. Relevance to our design

1. **Cautionary example, to be cited.** This is the proposal's "worst possible outcome" made concrete: uncertified exhaustive-search upper bounds, six of which are refuted by public witnesses, published as "certified". It justifies the two hard rules in `lean/README.md`: a prune is accepted only as a `Prune P` term, and a refutation is accepted only through a checked LRAT/SR certificate discharged at the `refuted` seam. Everything Padhi calls "certified" would fail both gates.
2. **The unsound-ordering hypothesis.** "Backtracks over column assignments ordered by the KST partition structure" plus false UNSATs is the signature of a symmetry-breaking assumption applied as a prune. `Demo.notDescending_unsound` is the formal version of this mistake. Our harness must route every ordering assumption through the certificate (SR/VeriPB permutation step), never through `kill`.
3. **Ground truth ledger.** Do not import any Padhi value. Keep `gpt_agent/data/exact_table.csv` (Guy/Tan) plus the 2026 certified closures (Bhan 3, dfield 8, Hou 7, Afrasyab, Wang), each tagged with certificate type. Add L3 as an automatic consistency check on the ledger and on every new claim before it is written down.
4. **Nothing new on the pruning side.** Padhi's three rules are Tan's; only L1 is non-trivial and it is exactly ZarPrune's open target. Proving L1 (and its column dual) in Lean remains the highest-value single step; Padhi adds no further counting argument.
5. **Targets.** Open cells adjacent to Padhi's unverified claims are good thesis targets, since any certified UB there is new and may additionally settle a disputed claim: (13,19) ∈ [118,125], (13,20) ∈ [119,130], (13,21) ∈ [127,135], (13,23) ∈ [135,144], (14,21) ∈ [131,145], (14,22) ∈ [137,150], (14,23) ∈ [138,155]. The m = 13 cells have the smallest gaps. z(17;3) ≤ 140 (would prove b(2,2,3) = 17, CRWR16) is the marquee target Padhi also names.
6. **Evaluator reuse.** The popcount triple-intersection check (L4) is the right verified witness checker; `verify_witness.py` in `sat_attack/` should do exactly this and nothing more.
7. **Solver policy.** Padhi's cascade (custom backtracking → Kissat) without proof logging is what not to do; dfield's (CaDiCaL/Kissat with DRAT → LRAT → `lrat-check`/cake_lpr, plus VIPR for LP covers) is the model to follow.

## 7. Difficulty signals for a case (from the note, plus inference)

* **Surviving-partition count.** The note's stated wall — "partition counts overwhelm our CDCL solver" at m ≥ 15 — makes #{column-sum partitions surviving L1+L2} the primary cost proxy; total cost ≈ Σ over survivors of per-case SAT time. Our `profiles.py` already enumerates exactly this set, so the count is free to compute before solving.
* **KST slack.** (Inference.) For a profile, slack = (t−1)·C(m,3) − Σ_j C(c_j,3). Zero or tiny slack means every row triple is covered by almost exactly t−1 columns — a near-design, usually refutable by counting/LP alone (cf. Afrasyab's Farkas certificates); large slack leaves the SAT solver to do the work. Slack (and its row dual) is a cheap per-case feature for the reward's "difficulty of what was pruned".
* **Wall-clock data point.** A satisfiable 17×17, T = 138 instance took Kissat 2h48m; UNSAT instances near the frontier will be slower. Any UB attempt at m ≥ 15 should budget hours per surviving case and hence needs the prune count to be small.
* **min(m,n) and clause count.** Naive clause count C(m,3)·C(n,3) grows ~m³n³/36; the custom prover was abandoned at m ≥ 15 even with 128 threads. Cases with m ≥ 15 are a different regime from m ≤ 13.

## 8. Open questions

1. Are the four unverified "exact" cells ((13,19)=120, (13,20)=124, (14,21)=134, (14,23)=142) and the m ≥ 15 lower bounds real? Only released witnesses can settle the LB halves; the UB halves need an independent proof. No code or certificates have appeared; the Zenodo deposit appears to have been deleted. Worth an email to the author asking for the 17×17/138 and 13×19/120 matrices.
2. Which rule made `zara_cdcl` incomplete — the mis-indexed KST bound implemented for n < m (not the case in the refuted cells), an ordering assumption inside partitions, or a bug in the "feasibility" prune? Unknowable without the code; the refuted cells (12,12), (12,18), (12,20), (14,18), (14,20), (14,22) are all n ≥ m and all have gaps of 1–18 ones, which is more consistent with a systematic incompleteness than with a typo.
3. Why do the captions/abstract say 44 and 31 when the printed tables hold 42 and 29 entries?
4. Does the 138-ones 17×17 matrix exist? If yes, does it extend to 18×18 with 139 or more (the note's own (17,18) ≥ 142 implies (18,18) ≥ 142 > 139)?

## 9. Links
* SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6960039 (DOI 10.2139/ssrn.6960039)
* Tan 2022: https://arxiv.org/abs/2203.02283 · CRWR16: https://arxiv.org/abs/1604.01257 · DGH: https://arxiv.org/abs/2411.18842
* Bhan et al.: https://arxiv.org/abs/2605.01120 · Hou: https://arxiv.org/abs/2608.08549 · Afrasyab: https://arxiv.org/abs/2608.08154 · Saurabh: https://arxiv.org/abs/2608.26603
* dfield: https://github.com/dfield/finite-zarankiewicz-closures · Hou repo: https://github.com/ShengtengHou/finite-zarankiewicz-seven-values · Wang: https://zenodo.org/records/21766509 · numaro: https://numaro.tech/research/zarankiewicz-2026/
* OEIS A001198 (k_3(n) = z(n;3)+1): https://oeis.org/A001198

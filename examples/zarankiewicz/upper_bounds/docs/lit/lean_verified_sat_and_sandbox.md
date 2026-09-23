# Lean 4 verified SAT/LRAT checking and Lean sandboxing for untrusted (LLM-written) proofs

Literature + toolchain notes for the ZarPrune upper-bound pipeline (Lean 4.34.0, no Mathlib).
Written 2026-09-21. Every toolchain fact below was read from the installed
`~/.elan/toolchains/leanprover--lean4---v4.34.0/src/lean` sources or reproduced by an experiment in
`/private/tmp/claude-501/-Users-jaybhan-Downloads-openevolve/88cfd163-c49a-4b7d-ba6f-24d477547106/scratchpad/lratexp/`
(experiment log in section 10). Paper facts are quoted from the PDFs in the scratchpad `lit/` directory.
"We/our" below = the thesis pipeline. Where I infer rather than quote, I say "inference".

---

## 0. Sources (bibliographic record; A = full text read, S = secondary/partial)

| # | Source | Access |
|---|---|---|
| L1 | Lean 4.34.0 toolchain sources: `Std/Tactic/BVDecide/LRAT/{Actions,Checker,Parser}.lean`, `LRAT/Internal/{CompactLRATChecker,Convert,Actions}.lean`, `Std/Tactic/BVDecide/Reflect.lean`, `Std/Sat/CNF/{Basic,Dimacs,RelabelFin}.lean`, `Std/Sat/AIG/CNF.lean`, `Lean/Meta/Native.lean`, `Lean/Elab/Tactic/Decide.lean`, `Lean/Meta/Tactic/BVDecide/{Attr,External}.lean`, `Lean/Meta/Tactic/BVDecide/LRAT/{Cert,Trim}.lean`, `Init/Core.lean` (ll. 2320-2415), `Lean/Replay.lean`, `Lean/Util/CollectAxioms.lean`, `LeanChecker.lean`, `Lean/Elab/AutoBound.lean` | A (local) |
| L2 | Lean reference manual, "Validating a Lean Proof", https://lean-lang.org/doc/reference/latest/ValidatingProofs/ | A (web) |
| L3 | Lean 4.29.0 release notes; RFC leanprover/lean4#12216 "One axiom per native computation" (PR #12217) | A (web) |
| L4 | leanprover/lean4checker README (deprecated since v4.28.0; merged into core as `leanchecker`) | A (web) |
| L5 | leanprover/comparator README; lean4 PR #15145 (`--paranoid`), PR #15157 | A/S (web) |
| L6 | GasStationManager/SafeVerify README; mistralai/LeanstralSafeVerify | A (web) |
| L7 | Zulip thread "soundness bug: native_decide leakage" (Carneiro, Morrison, Gallicchio, Carlin-Burns, Malone; fix lean4#2654) | A (archive) |
| L8 | lean4 issue #7463 "`@[csimp]` can be used to smuggle axioms and `unsafe` into a proof" (eric-wieser, Mar 2025, open, P-low) | A (web) |
| L9 | "Faults in Our Formal Benchmarking: Dataset Defects and Evaluation Failures in Lean Theorem Proving", arXiv:2606.29493 (2026) | A (PDF) |
| P1 | Böving, Bhat, Cicolini, Keizer, Frénot, Mohamed, Stefanesco, Khan, Clune, Barrett, Grosser. "Interactive Bitvector Reasoning using Verified Bit-Blasting", PACMPL 9(OOPSLA2):3259-3285, 2025, doi 10.1145/3763167 | S (ACM PDF 403; abstract + citations in LRAT-Catcher) |
| P2 | Szeider. "LRAT-Catcher: Importing SAT Solver Certificates into Lean 4 by Reflection" (also titled "Streaming LRAT Certificates into Lean Theorems"), arXiv:2607.00815 (2026); repo github.com/leansolving/lrat-catcher (Lean v4.30.0) | A (PDF, 24 pp.) |
| P3 | Szeider. "PBLean: Pseudo-Boolean Proof Certificates for Lean 4", arXiv:2602.08692v2 (2026); repo github.com/leansolving/pblean (v4.28.0-rc1, no Mathlib) | A (PDF) |
| P4 | Codel, Avigad, Heule. "Verified Encodings for SAT Solvers", FMCAD 2023, pp. 141-151, doi 10.34727/2023/isbn.978-3-85448-060-0_22; code github.com/ccodel/verified-encodings (Lean 3 + mathlib) | A (PDF) |
| P5 | Codel, Avigad, Heule. "Verified Substitution Redundancy Checking", FMCAD 2024, pp. 186-196, doi 10.34727/2024/isbn.978-3-85448-065-5_24; tools github.com/ccodel/dsr-trim, github.com/FormalSAT/trestle | A (PDF) |
| P6 | Gallicchio, Codel, Avigad, Heule. "An End-To-End Verification of Keller's Conjecture", ITP 2026, LIPIcs 382, 26:1-26:20 | A (PDF) |
| P7 | Subercaseaux, Nawrocki, Gallicchio, Codel, Carneiro, Heule. "Formal Verification of the Empty Hexagon Number", ITP 2024, LIPIcs 309, 35:1-35:19, arXiv:2403.17370 | A (PDF) |
| P8 | Tan, Heule, Myreen. "cake_lpr: Verified Propagation Redundancy Checking in CakeML", TACAS 2021, LNCS 12652, 223-241; repo github.com/tanyongkiam/cake_lpr; STTT 25(2):167-184 (2023) for compositional checking | A (PDF + README) |
| P9 | Lammich. "Fast and Verified UNSAT Certificate Checking", IJCAR 2024, LNCS 14739, 439-457; github.com/lammich/lrat_isa | S (abstract) |
| P10 | Bogaerts, Gocht, McCreesh, Nordström. "Certified Dominance and Symmetry Breaking for Combinatorial Optimisation", JAIR 77:1539-1589 (2023); prelim. AAAI 2022; arXiv:2203.12275 | A (PDF, 51 pp.) |
| P11 | Gocht, Martins, Nordström, Oertel. "Certified CNF Translations for Pseudo-Boolean Solving", SAT 2022, LIPIcs 236, 16:1-16:25 | A (PDF) |
| P12 | Heule, Hunt, Wetzler. "Expressing Symmetry Breaking in DRAT Proofs", CADE-25 2015, LNCS 9195, 591-606 | A (PDF) |
| P13 | Buss, Thapen. "DRAT and Propagation Redundancy Proofs Without New Variables", LMCS 17(2):12:1-12:31 (2021), arXiv:1909.00520 | A (PDF) |
| P14 | Heule, Kullmann, Marek. "Solving and Verifying the Boolean Pythagorean Triples problem via Cube-and-Conquer", SAT 2016, arXiv:1605.00723; Heule. "Schur Number Five", AAAI 2018, arXiv:1711.08076 | A (PDF) |
| P15 | Cruz-Filipe, Marques-Silva, Schneider-Kamp. "Formally Verifying the Solution to the Boolean Pythagorean Triples Problem", JAR 63(3):695-722 (2019) | A (PDF; skimmed) |
| P16 | Gauthier, Brown. "A Formal Proof of R(4,5)=25", ITP 2024, LIPIcs 309, 16:1-16:18, arXiv:2404.01761 | A (PDF) |
| P17 | Mathlib `Mathlib/Tactic/Sat/FromLRAT.lean` (Carneiro 2022), `lrat_proof` command | A (source) |
| P18 | Pollitt, Fleury, Biere. "Faster LRAT Checking Than Solving with CaDiCaL", SAT 2023 (cited by Lean's `Trim.lean` as the source of its trimming algorithm) | S |
| T1 | CaDiCaL NEWS.md (versions 2.2.0, 3.0.0, 3.0.1), Kissat NEWS.md (4.0.0-4.0.4), kissat(1) man page | A (web) |

Not found / not accessible: the OOPSLA'25 bv_decide PDF (ACM returned 403; facts about it are taken from
its abstract and from P2's description). No source claims about "Codel-Avigad-Heule 2023" beyond
what the PDF says were used.

---

## 1. Short answers to the four questions

**(a) Checking a CaDiCaL/Kissat LRAT certificate for one branch inside Lean 4.34 without Mathlib.**
Use the core library module `Std.Tactic.BVDecide.LRAT` (files `LRAT/Checker.lean`, `LRAT/Parser.lean`,
`LRAT/Actions.lean`), which is what `bv_decide` uses. The API is three names:

```lean
-- Std/Tactic/BVDecide/LRAT/Checker.lean (Lean 4.34.0)
def Std.Tactic.BVDecide.LRAT.check (lratProof : Array IntAction) (cnf : Std.Sat.CNF Nat) : Bool
theorem Std.Tactic.BVDecide.LRAT.check_sound (lratProof : Array IntAction) (cnf : CNF Nat) :
    check lratProof cnf → cnf.Unsat
-- Std/Tactic/BVDecide/LRAT/Parser.lean
def Std.Tactic.BVDecide.LRAT.parseLRATProof (proof : ByteArray) : Except String (Array IntAction)
def Std.Tactic.BVDecide.LRAT.loadLRATProof (path : System.FilePath) : IO (Array IntAction)
-- Std/Tactic/BVDecide/Reflect.lean (bundles the two)
def Std.Tactic.BVDecide.Reflect.verifyCert (cnf : CNF Nat) (cert : String) : Bool
theorem Std.Tactic.BVDecide.Reflect.verifyCert_correct : ∀ cnf cert, verifyCert cnf cert = true → cnf.Unsat
```

`CNF.Unsat f := ∀ a, eval a f = false` (`Std/Sat/CNF/Basic.lean`). The theorem for a branch is obtained by
reflection: `theorem branch_unsat : myCnf.Unsat := LRAT.check_sound proof myCnf (by native_decide)`.
The `native_decide` step compiles and runs `check`, then (since Lean 4.29) records the result as **one
fresh axiom per evaluation**, named `<decl>._native.native_decide.ax_<k>` with type `check proof myCnf = true`
(`Lean/Meta/Native.lean`, `nativeEqTrue`). Kernel-only checking (`decide +kernel`) does **not** work:
the checker's main loop `compactLratChecker.go` is compiled by well-founded recursion
(`#print` shows `@[irreducible] ... := compactLratChecker.go._unary ...`) and gets stuck even on PHP(3,2)
(experiment 10.3). The kernel-term alternative is Mathlib's `lrat_proof` (RUP-only, `"unimplemented: RAT
step"`, and P2 reports it out of memory at a 63 MB certificate), so it is not an option for us.
Performance measured here (Apple Silicon, Lean 4.34.0, bundled CaDiCaL 2.1.2): a 151 MB / 1.13 M-line LRAT
for the K_{2,2}-free 7x7, >=22-ones instance imports in 16 s wall, 2.35 GB RSS (10.3). P2 reports
about 140-150 CPU-s and 2.05 GB + 0.28 GB per GB of certificate for streamed imports, and that the Lean
checker is about 15x slower than `lrat_isa` (P9) and comparable to `cake_lpr` (Table 3 of P2, reproduced
in section 3.4). Solver side: the toolchain ships `bin/cadical` (2.1.2) and invokes it with
`<cnf> <proof> --lrat --binary=<b> --quiet --shrink=0 [--unsat]` (`External.satQuery`). Kissat 4.0.x emits
DRAT only (man page; NEWS has no LRAT entry), so Kissat proofs must go through `drat-trim -L` first.
CaDiCaL >= 2.2 has bounded variable addition (`factor`); it is off by default in 2.2 and 3.0.1 but on in
3.0.0 -- pass `--no-factor` because the Lean checker maps variables into `PosFin (cnf.numLiterals + 1)` and
silently drops steps that mention out-of-range (extension) variables, after which later hints fail.

**(b) Making an LLM-written Lean file untrusted-safe.** The official ladder (L2): (1) kernel accepts the
declaration; (2) `#print axioms` shows only `propext`, `Classical.choice`, `Quot.sound`; (3) `leanchecker`
(`lake env leanchecker <Module>`, `--fresh` for a from-scratch replay) replays every declaration through the
kernel to catch environment hacking; (4) gold standard `lake comparator` (Linux-only: `landrun` sandbox,
`lean4export` NDJSON, kernel + `nanoda`; `--paranoid` adds `leanchecker-paranoid`, `lean4lean`, `con-leche`).
Concretely for us: the harness, not the LLM, states the type (`def candidate : Prune P := <LLM term>` in a
harness-owned wrapper), compiles the LLM module in a sandbox (no network, temp cwd, `ulimit`, wall clock),
rejects the source if a regex hits `axiom`, `sorry`, `native_decide`, `decide +native`, `implemented_by`,
`extern`, `csimp`, `unsafe`, `partial`, `opaque` with a body, `set_option`, `#eval`/`#exit`, `initialize`,
`macro`/`notation`/`elab`/`syntax`, `open ... in`-renaming of `ZarPrune` names, or any `import` other than
`ZarPrune`; then runs `collectAxioms` on the wrapper's constant and requires the axiom set to be a subset of
the three standard axioms (this catches `sorryAx`, user axioms, and every `_native` axiom); then runs
`leanchecker` on the produced `.olean`; then `#eval`s `kill` on sample profiles to prove it is computable.
Name-pattern whitelisting of `_native` axioms is spoofable (10.4: `axiom spoof._native.native_decide.ax_1_1 :
False` compiles, prints as such, and passes `leanchecker`), so LLM files get **zero** native axioms; native
axioms are allowed only in harness-generated refutation modules, whose statements the harness itself wrote.
Set `autoImplicit = false` (core default is `true`, `Lean/Elab/AutoBound.lean:17`).

**(c) State of the art for verified CNF-encoding correctness.** Codel-Avigad-Heule FMCAD 2023 (P4, Lean 3 +
mathlib): `encodes C F l := ∀ τ, C.eval τ l = tt ↔ ∃ σ, F.eval σ = tt ∧ agree_on τ σ (vars l)`, with a
`gensym` for fresh variables and "well-behaved" encoding functions that compose; verified parity, AMO,
sequential-counter at-most-k. Its Lean 4 successor is **Trestle** (github.com/FormalSAT/trestle, Lean
v4.21.0 + Mathlib): `PropPred ν := (ν → Bool) → Prop`, `VEncCNF ν P := { e : EncCNF ν // e.encodesProp P }`,
`withTemps`, `for_all`; used in Keller (P6: "only 150 lines of Lean code to write and verify the full
encoding") and it contains the verified LSR (substitution-redundancy) checker of P5. End-to-end SAT
results: Pythagorean triples in Coq (P15, encoding + symmetry breaking verified, computation via extracted
OCaml), R(4,5)=25 in HOL4 (P16, MiniSat proofs replayed through the HOL4 kernel), Empty Hexagon in Lean 4
(P7, verified encoder, but the `cake_lpr` verdict is asserted as an axiom), Keller in Lean 4 (P6, verified
encoder + verified LSR checker + LRAT reflection), LRAT-Catcher (P2, `lrat_reflect_cnf name (myCnf) file`
with a Lean-defined CNF, no Mathlib), PBLean (P3, verified encodings + VeriPB reflection, no Mathlib), and
Lean core's own verified Tseitin `Std.Sat.AIG.toCNF` with `toCNF_equisat : (toCNF entry).Unsat ↔
entry.Unsat` (what `bv_decide` uses). On the proof-logging side, Gocht-Martins-Nordström-Oertel (P11)
certify sequential-counter/totalizer/adder CNF translations of PB constraints inside VeriPB by cutting
planes + reification, which yields exactly the direction we need (UNSAT of the CNF implies UNSAT of the PB
constraints) without a proof assistant. For ZarPrune (no Mathlib) the realistic route is a small home-grown
encoder over `Std.Sat.CNF Nat` with a *completeness* theorem `Valid P A → profileOf A = q →
(encode P q).Sat (assign A)` (only this direction is needed to discharge `refuted`), which is easier than
P4's iff because it asks only for a witness for the auxiliary variables.

**(d) "Pruning vs adding" in the proof-logging literature.** Clausal proof systems distinguish clauses that
are *implied* (RUP/asymmetric tautology: `F ∧ ¬C ⊢₁ ⊥`, checkable with no witness) from clauses that are
merely *redundant* -- `F` and `F ∧ C` are equisatisfiable although `F ⊭ C` -- which need a witness: RAT
(pivot literal), PR (partial assignment), SR (substitution σ with σ ⊨ C and `F ∧ ¬C ⊢₁ (F ∧ C)|σ`, P13
Def 1.13, P5 Thm 1). Symmetry-breaking predicates are the canonical redundant-but-not-implied clauses:
P12 is titled precisely because "it was not known how to express them in proofs of unsatisfiability"; P10
notes that until VeriPB "it has not been possible to use symmetry breaking in the SAT competition, since
there has been no way of efficiently certifying the correctness of such reasoning in DRAT" and gives the
redundance rule (Def 6, eq. (5)) and dominance rule (Def 13) with explicit witness substitutions ω. P13 and
P10 both describe these rules as licensing clauses that hold "without loss of generality". Cube-and-conquer
adds the third obligation: the cubes must *cover* the space, certified by a "tautology proof" (P14: refute
the negation of the disjunction of cubes with a proof-logging solver). Mapping to ZarPrune: `Prune.sound`
is the implied/pruning side (`kill q = true` means the case is empty, i.e. `F ∧ cube_q ⊨ ⊥`); "assume rows
sorted" is the redundant/adding side and is refuted by `Demo.notDescending_unsound`; `cover` is the
tautology obligation. Keller (P6 §5) is the model to copy: the WLOG steps that SR cannot express (their
conditional index bit-flip) are proved in Lean once at the problem level, everything after that is SR
checked by a verified checker, and "by moving this transition point to be as soon as possible in our proofs,
we reduced the human proof burden".

---

## 2. Lean 4.34.0 core: the LRAT checker and the reflection mechanism (from source)

### 2.1 Types and the certificate format
`Std/Tactic/BVDecide/LRAT/Actions.lean` (author Josh Clune, Amazon):
```lean
inductive Action (β : Type u) (α : Type v)
  | addEmpty (id : Nat) (rupHints : Array Nat)
  | addRup (id : Nat) (c : β) (rupHints : Array Nat)
  | addRat (id : Nat) (c : β) (pivot : Literal α) (rupHints : Array Nat) (ratHints : Array (Nat × Array (Nat)))
  | del (ids : Array Nat)
abbrev IntAction : Type := Action (Array Int) Nat
```
The parser implements "a (corrected) version of the grammar presented in ... lrat.pdf" for both the text
and the binary LRAT format (`Parser.lean` header). `LRAT.check` converts the `CNF Nat` with
`CNF.convertLRAT` (`Internal/Convert.lean`) into a `DefaultFormula (cnf.numLiterals + 1)` over
`PosFin`, then runs `compactLratChecker` (`Internal/CompactLRATChecker.lean`, copyright 2026, "only explodes
[actions] into the `DefaultClauseAction` when required ... significantly smaller memory footprint").
Full RAT is supported (`performRatAdd`, `RatAddSound.lean`). A step whose literals do not fit in
`PosFin (numLiterals+1)` makes `intActionToDefaultClauseAction` return `none` and the step is skipped
(`| none => go f proof (idx + 1)`); this is why P2 says extension variables are "soundly rejected".

Variable numbering: `Std.Sat.CNF Nat` literals are `(Nat × Bool)` with variables from 0;
`CNF.dimacs` "will add `1` to all literal identifiers" (`Dimacs.lean`), and `CNF.lift` relabels `v ↦ v+1`
into `PosFin`. Hence DIMACS/LRAT literal `±k` corresponds to Lean variable `k-1`. (My generator subtracts
1 when emitting the Lean term; experiments 10.2-10.3 confirm the mapping.)

### 2.2 How `bv_decide` drives an external solver (reusable as-is)
`Lean/Meta/Tactic/BVDecide/Attr.lean`: option `sat.solver : String := ""` -- "If this is set to the empty
string they will check if there is a cadical binary next to the executing program ... we do ship a `cadical`
next to it." (`~/.elan/toolchains/leanprover--lean4---v4.34.0/bin/cadical --version` = 2.1.2.)
`Lean/Meta/Tactic/BVDecide/External.lean`, `satQuery`: arguments `[problemPath, proofOutput, "--lrat",
s!"--binary={binaryProofs}", "--quiet", "--shrink=0"]` plus `--unsat` in `.proof` mode; "This function
currently assume that the solver has the same CLI as CaDiCal"; timeout implemented in Lean ("cadicals -t
option is not available on Windows"). `BVDecideConfig` (`Std/Tactic/BVDecide/Syntax.lean`): `timeout := 10`
seconds, `trimProofs := true`, `binaryProofs := true`, `acNf := false`, ...
`Lean/Meta/Tactic/BVDecide/LRAT/Cert.lean`: `LratCert := String` ("This will get parsed using native
evaluation"); `LratCert.load` parses then runs `LRAT.trim` -- `Trim.lean` "implements the LRAT trimming
algorithm described in section 4 of 'Faster LRAT Checking Than Solving with CaDiCaL'" (P18); binary proofs
are re-serialised to text "due to missing support for binary literals" in the environment.
`Reflect.verifyBVExpr bv cert := verifyCert (AIG.toCNF bv.bitblast.relabelNat) cert` and
`unsat_of_verifyBVExpr_eq_true` chain `BVLogicalExpr.unsat_of_bitblast`, `relabelNat_unsat_iff`,
`toCNF_equisat`, `verifyCert_correct`. The verified Tseitin encoding lives in `Std/Sat/AIG/CNF.lean`
(`theorem toCNF_equisat (entry : Entrypoint Nat) : (toCNF entry).Unsat ↔ entry.Unsat`).

### 2.3 The native-evaluation axiom (Lean >= 4.29) replaces `ofReduceBool`
`Lean/Meta/Native.lean` (Breitner, 2025): "Such proofs involve a native computation using the Lean kernel,
and then asserting the result of that computation as an axiom towards the logic." `nativeEqTrue tacName e`
compiles an auxiliary definition `<decl>._native.<tac>.decl`, evaluates it with `evalConst`, and on `true`
adds `Declaration.axiomDecl { name := <decl>._native.<tac>.ax_k, type := e = true, isUnsafe := false }`.
`bv_decide` calls `nativeEqTrue \`bv_decide reflectionTerm` (`Prover/Bitblast.lean:39`); `native_decide`
and `decide +native` call it from `Elab/Tactic/Decide.lean` (`elabNativeDecideCore`). Observed names:
`php65Cnf_unsat._native.native_decide.ax_1_1`, `bvz44._native.bv_decide.ax_1_5`,
`two_plus_two'._native.decide.ax_1_1`.
Release notes 4.29.0: "native computation (`native_decide`, `bv_decide`) is represented in the logic as one
axiom per computation, asserting the equality that was obtained from the native computation" (RFC #12216,
PR #12217). `Init/Core.lean` in 4.34: `axiom trustCompiler : True`, `opaque reduceBool`, `axiom ofReduceBool`
and `ofReduceNat` all carry `@[deprecated "in-kernel native reduction is deprecated; assert native
evaluations with axioms instead" (since := "2026-02-01")]`. The old docstring is still the honest statement
of the trust cost: "by using this feature, the Lean compiler and interpreter become part of your trusted code
base. This is extra 30k lines of code ... you will probably not be able to check your development using
external type checkers ... the compiler trusts the correctness of all `[implemented_by ...]` and
`[extern ...]` annotations."
Consequence for auditing: `#print axioms` / `Lean.collectAxioms` (`Lean/Util/CollectAxioms.lean`) show a
distinct axiom per native evaluation; L2: "If axioms with `_native` in their names are reported, then native
evaluation is used." The RFC's rationale: external checkers can "enumerate" these axioms and "re-run
[them] (e.g. using a different bit blaster and sat solver)", and projects can "whitelist some tactics" by
name -- but see 4.3 on spoofing.

### 2.4 Why the kernel cannot run the Std checker
`#print Std.Tactic.BVDecide.LRAT.Internal.compactLratChecker.go` gives `@[irreducible] def ... :=
fun {n} f proof idx => compactLratChecker.go._unary proof ⟨f, idx⟩` -- well-founded recursion on
`proof.size - idx`. `decide +kernel` on `LRAT.check php32Proof php32Cnf = true` fails with "Reduction got
stuck at the `Decidable` instance match LRAT.check php32Proof php32Cnf, true with ..." (10.3). A
kernel-checkable checker would have to be fuel-based structural recursion; nobody has published one for
full LRAT in Lean 4 core, and P2 measured Mathlib's term-building `lrat_proof` out of memory at 63 MB.

---

## 3. Verified checkers and importers: landscape and numbers

### 3.1 Lean-internal
* **Std `LRAT.check` (core)** -- verified, full RAT, reflection with one native axiom. Used by `bv_decide`
  (P1), LRAT-Catcher (P2), Keller (P6: "checked by a verified LRAT checker [Std], and then a
  proof-by-reflection tool gives a complete proof in Lean").
* **LRAT-Catcher (P2)** -- library of about 2,300 lines over the core checker; adds chunked/streamed
  checking (`stepStart/stepMid/stepFinish`, Lemma 6 `runChunk_more_sound`), boundary compaction, and
  cube composition Lemma 17:
  ```lean
  theorem cover_unsat (hleaf : ∀ c ∈ cubes, (Cube.leafCNF c base).Unsat)
      (hcover : (negCubesCNF cubes).Unsat) : base.Unsat
  ```
  Commands `lrat_reflect name "f.cnf" "f.lrat"` (statement `(parseDimacs «...»).Unsat`) and
  `lrat_reflect_cnf name (myCnf) "f.lrat"` (statement `myCnf.Unsat`, "the verified-encoding form"). Toolchain
  v4.30.0, "There is no Mathlib dependency." Requires `cadical --lrat --no-binary --no-factor` for CaDiCaL 3.
  Trust classification (§2.3 of P2): "A component is trusted if a fault can lead to a false theorem, and it is
  adversarial if a fault can at most fail the build. The basic trusted components are the Lean kernel with
  its three standard axioms and the native-evaluation mechanism (compiler ...)"; parsers and the renumberer
  are adversarial; "A fault in the parser changes which formula the theorem is about."
* **PBLean (P3)** -- VeriPB kernel-format (v3.0: `pol, rup, pbc, red/dom, del, weaken, sol/soli,
  conclusion`) checker by reflection, no Mathlib, 15 soundness lemmas (`add_sat`, `div_sat`,
  `saturate_sat`, `applySubstConstr_sat_rev`, ...). Table 2: Paley(101), 2,526 constraints, 62,924 proof
  lines: explicit proof terms time out at 60 s from p >= 37, reflection 200 s. Redundance rule for symmetry
  breaking exercised on pigeonhole. Command `veripb_reflect`.
* **Mathlib `lrat_proof` (P17)** -- kernel proof terms, RUP-only (`"unimplemented: RAT step"`).
* **Trestle LSR checker (P5, P6)** -- verified in Lean 4 (Mathlib), 8k LoC, "four person-months"; hinted
  substitution-redundancy proofs (`LSR`), unhinted `DSR` labelled by `dsr-trim`. Runtime: geometric mean
  Lean/`cake_lpr` = 0.718 on proofs > 1 s (LPR), and about 10x slower than the unverified `lsr-check`;
  SR proofs were on average 6.2 MB / 13.2K lines vs 41.2 MB / 1.03M lines after conversion to LRAT.

### 3.2 External verified checkers (verdict, not theorem)
* **cake_lpr (P8)** -- CakeML/HOL4, verified down to x64 machine code; end-to-end theorem: if
  "`s VERIFIED UNSAT`" is printed then the formula is unsatisfiable, with possible out-of-memory
  termination; supports LPR (superset of LRAT) and binary LRAT/LPR; two-level/compositional checking
  (`cake_lpr <cnf> <summary> i-j <lpr>`), `--CML_HEAP_SIZE`. Used by Empty Hexagon (P7) on-the-fly and by
  Keller's predecessor BHMN 2020.
* **lrat_isa (P9)** -- Isabelle/HOL, verified to LLVM IR including a verified DIMACS parser; the fastest
  verified checker in P2's Table 3.
* **CakePB / VeriPB** -- verified pseudo-Boolean checker (Gocht, McCreesh, Myreen, Nordström, Oertel, Tan,
  AAAI 2024) for the symmetry/dominance rules of P10.

### 3.3 Unverified but standard
`lrat-check`, `drat-trim` (DRAT to LRAT with `-L`), `lrat-trim`, `dsr-trim`/`lsr-check` (P5), `sr2drat`.

### 3.4 Performance table (P2, Table 3; one machine, ASCII certificates unless noted)
| certificate | tool | wall | CPU (s) | RSS (GB) | produces |
|---|---|---|---|---|---|
| PHP(10,9), 63.4 MB | stream (Lean) | 15.8 s | 3.24 | 1.52 | theorem |
| | lrat_isa (bin.) | 0.16 s | 0.14 | 0.01 | verdict |
| | cake_lpr | 3.23 s | 1.95 | 2.85 | verdict |
| | lrat-check | 0.83 s | 0.80 | 0.04 | verdict |
| | Mathlib `lrat_proof` | -- | -- | out of memory | none |
| R(B2,B9) <= 22, 275.9 MB | stream | 20.9 s | 13.1 | 1.58 | theorem |
| | whole-file (Lean) | 32.5 s | 24.6 | 4.41 | theorem |
| | lrat_isa (bin.) | 0.76 s | 0.72 | 0.03 | verdict |
| | cake_lpr | 12.7 s | 8.26 | 8.93 | verdict |
| R(B3,B7) <= 20, 58.7 GB | stream | 54 min | 3,110 | 4.87 | theorem |
| | cake_lpr | 43 min | 2,560 | 12.9 | verdict |
| | lrat-check (stdin) | 10.8 min | 631 | 0.62 | verdict |

P2 end-to-end case studies (Table 5): R(4,4)=18 (1,024 leaves, 85.5 GB certificates, 3.26 CPU-h import);
queen domination n=19 (262,144 leaves, 566 GB, 21.9 CPU-h); Keller G_{7,3} (21,557 leaves, 773 GB, 199 CPU-h);
w(2;3,18)=312 (909,558 leaves, 25.9 TB, 1,087 CPU-h); the 174 TB empty-hexagon re-solve streamed through
4,882 groups. Rule of thumb from P2: "about 140 CPU-seconds per GB of certificate" including module build
("about 53 CPU-seconds per GB for the checker alone"); memory fit "2.05 GB plus 0.28 GB per GB of largest
leaf" (Figure 2); leaf sizes are long-tailed ("median leaf of queen domination is 38 kB and its largest 16.8
GB"); 19 of 909,558 w(2;3,18) leaves exceeded the 24 h limit and were re-cubed at depth 8.

Keller (P6, Table 1): n=7, s=64: 5,657,894 clauses, 117,376 vars, 2,582 SR clauses, dsr-trim 3,991.6 s,
lsr-check 370.5 s, 2,771 cubes, solve 228.5 h, check 125.0 h; n=7, s=6: 383,142 clauses, 385 SR clauses,
2,771 cubes, solve 29.3 h, check 19.6 h. Checking cost is 0.5-1x solving cost at this scale.

---

## 4. Untrusted Lean: threat model, mechanisms, and the concrete audit

### 4.1 What the official guidance says (L2, quoted/paraphrased)
Escalating levels: blue double check marks (statement elaborated, kernel accepted; catches incomplete proofs
and explicit `sorry`) → `#print axioms` (only `propext`, `Classical.choice`, `Quot.sound` are benign;
`sorryAx` means incomplete; `_native` names mean native evaluation) → `lean4checker --fresh` after `lake
build` ("replays proofs from `.olean` files through the kernel"; protects against "bugs in Lean's core
handling of the kernel's state" and "meta-programs or tactics intentionally bypassing that state"; but "since
`lean4checker` reads the `.olean` files without validating their format, this check is prone to an attacker
crafting invalid `.olean` files (e.g. invalid pointers, invalid data in strings)") → gold standard `lake
comparator`: "build the proof in a sandboxed environment ... exports the proof term ... outside the sandbox
and out of the reach of possibly malicious code, it validates the exported format, replays the proofs using
both Lean's kernel and/or an external checker" (`nanoda`, "developed independently and implemented in
Rust"). Remaining assumptions even then: soundness of Lean's logic, correctness of comparator, sandbox
security, no bug shared by all checkers, and no "misleading presentation in the trusted challenge file".

### 4.2 Tools
* `leanchecker` (core since v4.28.0; `src/lean/LeanChecker.lean`): "This will replay all the new
  declarations from the target file into the `Environment` as it was at the beginning of the file, using the
  kernel to check them ... `--fresh` to replay all the constants (both imported and defined in that file) into
  a fresh environment ... This is not an external verifier, simply a tool to detect 'environment hacking'."
  Implementation: `Lean.Kernel.Environment.replay` (`Lean/Replay.lean`, Morrison 2023). Measured on
  ZarPrune: `lake env leanchecker ZarPrune` 0.66 s; `--fresh ZarPrune` 29.4 s (replays Init/Std). It admits
  axiom declarations as axioms; it does **not** re-run native evaluations (10.4: a native-axiom module and
  a spoofed-axiom module both exit 0).
* `lake comparator` (L5): config lists `challenge_module`, `solution_module`, `theorem_names`,
  `permitted_axioms`; guarantees the solution proves "the same statement as provided in Challenge", uses
  "no more axioms than listed", and is "accepted by the Lean kernel"; sandbox is `landrun` (Linux only,
  systemd-run guard recommended); external kernels via `external_kernels`; `--paranoid` runs
  `leanchecker-paranoid`, `lean4lean`, `nanoda`, `con-leche` (PR #15145); "definition holes must always be
  checked with an additional (potentially human) verifier". Not runnable on this macOS box; usable on the
  cluster.
* `SafeVerify` (L6): operates on `.olean`s; `Environment.replay` of both target and submission; matches
  each target declaration's name/kind/type; definition bodies must agree unless the target has `sorry`;
  axioms restricted to the three standard ones via `CollectAxioms.collect`; rejects `partial`/`unsafe`;
  "does not check for keywords like `implemented_by`, `extern`, or `noncomputable` because these operate at
  the source level"; "Compilation should occur in a sandboxed environment before passing olean files to
  SafeVerify"; `native_decide` proofs "will not pass SafeVerify". Usage: `lake env lean -o submission.olean
  submission.lean`, then `lake exe safe_verify target.olean submission.olean`.

### 4.3 Known escape hatches and what closes them
| hatch | mechanism | closed by |
|---|---|---|
| `sorry` | `sorryAx` axiom | axiom check (also `lake build` warning) |
| user `axiom` | any Prop | axiom check; forbid keyword in source |
| `native_decide`, `decide +native`, `bv_decide` | fresh `_native` axiom, compiler trusted | axiom check (non-standard axiom); L9: "known bugs in native code generation and implemented by overrides have produced proofs of False" |
| `@[implemented_by]`, `@[extern]`, `@[csimp]` | change what compiled code computes; L8 shows a `csimp` lemma proved from `axiom cheating : False` is not reported by `#print axioms` of a `native_decide` proof (issue open, P-low) | irrelevant once no native axiom is accepted; still forbid at source level |
| `unsafe` | cannot be used in proofs directly, but feeds compiled code | forbid; SafeVerify rejects |
| `partial` | logically an opaque constant (safe) but hides divergence in `kill` | forbid (we need `kill` to terminate) |
| metaprogram `addDecl` without kernel (`Environment.addDeclWithoutChecking`, unsafe env edits) | environment hacking | `leanchecker` replay |
| `IO` side effects during elaboration (`#eval`, `initialize`, `run_cmd`, `macro` expanding to IO) | filesystem/network from the build | sandbox the build; forbid these commands; L7's original exploit used `IO.getRandomBytes` inside `Lean.reduceBool` |
| `autoImplicit` (core default `true`, `AutoBound.lean:17`) | typo variables become universally quantified, statement changes (L9 §"Vacuous hypotheses"/`autoImplicit`) | harness owns the statement; `leanOptions = [{autoImplicit = false}]` |
| shadowing/notation (`local notation`, `open X in`, re-`def` of `Valid`) | statement means something else | harness wrapper refers to `ZarPrune.Valid` etc. by full name in a file the LLM cannot edit; forbid `notation/macro/syntax/elab` |
| `set_option maxHeartbeats/maxRecDepth`, huge terms | denial of service | wall-clock + memory limits on the `lean` process |
| crafted `.olean` | bypasses `leanchecker` (L2) | never accept an LLM-supplied `.olean`; always compile from source ourselves |
| spoofed `_native` axiom name | `axiom x._native.native_decide.ax_1_1 : False` (10.4) | LLM files: no axioms at all; harness modules: only names the harness itself generated, plus optional re-evaluation of the axiom's `e = true` statement |

L9 (arXiv:2606.29493) adds: the pre-4.20.0 `apply?` bug that reported success "without producing a theorem
declaration that had passed ordinary kernel verification" (three DeepSeek-Prover-V2 claims), and the
recommendation to "use patched Lean versions, verify `#print axioms` output, and incorporate maximally
strict verification in RL reward signals". Their table lists "Improper axiom usage; native decide shortcuts;
unsafe tactics" as the proof-acceptance failure class.

### 4.4 The audit we should run (design; each line maps to a fact above)
1. LLM output is a *module body* only; the harness prepends the fixed header (`import ZarPrune`, `set_option
   autoImplicit false`, `namespace ZarPrune.Evolved`) and appends the wrapper file
   `def evolved_<id> : Prune <P> := <name>` in a separate harness-owned file.
2. Source regex reject list (case-sensitive, word boundaries): `axiom`, `sorry`, `native_decide`,
   `+native`, `implemented_by`, `extern`, `csimp`, `unsafe`, `partial`, `opaque`, `set_option`, `#eval`,
   `#exit`, `#check`-with-`IO`, `initialize`, `builtin_initialize`, `macro`, `macro_rules`, `notation`,
   `syntax`, `elab`, `run_cmd`, `run_tac`, `deriving instance`, `import` (any), `open Lean`, `Lean.`,
   `IO.`, `unsafeCast`, `ofReduceBool`, `trustCompiler`. A hit is a rejection, not a warning.
3. Compile in a fresh temp copy of the ZarPrune project inside a sandbox (macOS: `sandbox-exec` or a
   container; Linux: `landrun`/`bwrap` as comparator does) with no network and a 60 s / 4 GB budget.
4. In the wrapper, `#print axioms evolved_<id>` must be a subset of `{propext, Classical.choice,
   Quot.sound}` (checked by parsing `lake build --json` output or via `collectAxioms` in a small
   harness `main`; the ZarPrune baseline currently needs only `propext` and `Quot.sound`).
5. `lake env leanchecker ZarPrune.Evolved.<id>` exit 0.
6. `#eval (evolved_<id>).kill <profile>` on the harness's sample profiles must run (proves computability; a
   `noncomputable`/classical `kill` would compile but not evaluate) and must agree with the LLM's Python
   `kill` if one is co-evolved.
7. Native-evaluation modules (branch refutations, cover certificates) are generated by the harness only;
   their `_native` axiom names are recorded at generation time and must match exactly what `#print axioms`
   shows; nothing else may appear.

---

## 5. Verified encodings: definitions and what we should reuse

### 5.1 Codel-Avigad-Heule 2023 (P4) -- the reference definitions
Definition 1: "Let C be a boolean constraint, F be any propositional logic formula, and X = x1,...,xn be
variables representing the inputs to C. Then F encodes C if and only if: for every assignment τ on X,
C(τ(x1),...,τ(xn)) if and only if F is satisfied by some assignment that extends τ." In Lean:
```
def encodes (C) (F) (l : list (literal V)) :=
  ∀ τ, (C.eval τ l = tt) ↔ ∃ σ, F.eval σ = tt ∧ (agree_on τ σ (vars l))
def enc_fn (V : Type*) := list (literal V) → gensym V → cnf V × gensym V
def is_correct (C) (e : enc_fn V) := ∀ {|l|} {|g|}, disjoint (vars l) g.stock → encodes C (e l g).1 l
def is_wb (e : enc_fn V) := ∀ {|l|} {|g|}, disjoint (vars l) g.stock →
  (e l g).2.stock ⊆ g.stock ∧ (e l g).1.vars ⊆ (vars l) ∪ (g.stock \ (e l g).2.stock)
```
"Well-behaved encoding functions can be combined together safely ... their combination is also well-behaved
and encodes the boolean-AND of the two constraints." Sequential counter AMK (Definition 6, k >= 2):
`SC_k(X) = ⋀_{j} (¬x_j ∨ s_{1,j}) ∧ ⋀_{i<=k+1, j<n} (¬s_{i,j} ∨ s_{i,j+1}) ∧ ⋀_{i<=k, j<n} (¬x_{j+1} ∨ ¬s_{i,j} ∨
s_{i+1,j+1}) ∧ ¬s_{k+1,n}`, "(k+1) × n matrix of signal variables and O(nk) clauses". Encodings verified:
direct and recursive (Tseitin-cut) parity, direct and sequential-counter AMO, sequential-counter AMK;
a Sudoku encoding composed from verified sub-encodings has a 15-line correctness proof. Written in Lean 3
(P4 §IV: "We used the interactive theorem prover Lean 3 ... depends on mathlib").

### 5.2 Trestle / Keller (P5, P6) -- the Lean 4 form
`abbrev PropPred (ν : Type u) := (ν → Bool) → Prop`; `def VEncCNF (ν) (P : PropPred ν) := { e : EncCNF ν //
e.encodesProp P }`; "Roughly, an encoded formula F encodes P if all truth assignments that satisfy P also
satisfy F" -- note this is the one-directional (completeness) reading that suffices for refutation;
`for_all (arr : Array α) (f : (a : α) → VEncCNF ν (P a)) : VEncCNF ν (fun τ => ∀ a ∈ arr, P a τ)`;
`withTemps` scopes auxiliary variables; `|>.mapProp (by ...)` discharges the semantic side. Keller's Gap
constraint encoding with auxiliary `z_{i,j,d}` is 11 proof lines. The pipeline then proves
`cliqueToAssn_satisfies_fullSpec : cliqueToAssn C.toKColoring ⊨ fullSpec` -- exactly the shape of the
`refuted`-discharge lemma we need (object → satisfying assignment of the encoding). Trestle pins Lean
v4.21.0 + Mathlib ("almost certainly will not compile on other versions"), so it cannot be imported into
ZarPrune (4.34, Mathlib-free); its design can be copied in a few hundred lines.

### 5.3 Other end-to-end precedents
* Empty Hexagon (P7): verified encoder in Lean/Mathlib; solving: 312,418 subproblems, CaDiCaL 1.9.5, LRAT
  "validated using the cake_lpr verified checker on-the-fly"; "asserting unsatisfiability of the CNF as an
  axiom in Lean. Thus we trust that the CNF formula produced by the verified Lean encoder is the same one
  whose unsatisfiability was checked by cake_lpr"; "the trust story at this connection point has room for
  improvement" (later closed by P2's streamed re-import).
* Keller (P6): reduction 3,000 lines (vs 15,000 in Lean 3), encoding 150 lines, symmetry breaking split
  between Lean (Theorems 2 and 4, about 200 lines after simplification) and SR; cube cover "we checked that
  the formula with the negated cubes is easy to refute".
* R(4,5)=25 (P16, HOL4): MiniSat proofs replayed through the HOL4 kernel via HolSatLib; "we were able to run
  the entire proof of R(4,5)=25 ... through the HOL4 kernel", contrasting with the Coq BPT proof (P15) that
  "relied on OCaml code proven correct in Coq".
* Gocht et al. (P11): certifies CNF translations of PB constraints in VeriPB. Sequential counter:
  `s_{i,j} ↔ ((ℓ_i ∧ s_{i-1,j-1}) ∨ s_{i-1,j})` (4); clauses (5a)-(5d); preservation equality
  `ℓ_i + Σ_{j<i} s_{i-1,j} = Σ_{j<=i} s_{i,j}` (7); ordering `s_{i,j} >= s_{i,j+1}` (9); reification
  `s_{i,j} ⇔ ℓ_i + Σ_j s_{i-1,j} >= j` (10); Prop. 4: from an arithmetic graph and `Σ a_i ℓ_i ⋈ k` derive
  `Σ c_i o_i ⋈ k` by cutting planes. "A stronger result would be to certify equivalence ... we have to leave
  this as future work" -- i.e. what they certify is the direction UNSAT(CNF) ⇒ UNSAT(PB). Totalizer and
  adder networks are handled in the appendices.

### 5.4 What our existing encoder does (for the record)
`zarankiewicz_generalized/gpt_agent/analysis/sat_attack/encodings_zar.py` uses one variable per cell,
per-row-triple auxiliary `y` with one-directional `(x_a ∧ x_b ∧ x_c) → y` and `AtMost2` over columns (the
docstring argues this one-directional `y` is "sound AND complete for the <=2 constraint"), PySAT
`CardEnc` totalizer for the global cardinality, a home-grown two-directional sequential counter
(`exact_unary_counter`), and two symmetry-breaking families: adjacent-column lex order and row-degree
monotonicity. In proof-logging terms the last two are *additions* (redundant, not implied); everything
else is implied by the matrix semantics. Any Lean completeness theorem must construct witnesses for the
`y`, totalizer and counter variables; the symmetry-breaking clauses cannot be given a completeness proof
per matrix (a valid matrix with unsorted rows satisfies the problem but not the lex clauses) and must
be handled at the cover level (section 6.4).

---

## 6. Pruning vs adding: the proof-logging distinction, precisely

### 6.1 Definitions (P12 §2-3, P13 §1, P5 §II)
* Unit propagation `⊢₁`; a clause `C` is an *asymmetric tautology* / RUP w.r.t. `F` iff `F ∧ ¬C ⊢₁ ⊥`;
  "Asymmetric tautologies are logically implied by F" (P12). This is the only witness-free rule.
* RAT on pivot `l`: for all `D ∈ F` with `¬l ∈ D`, `F ∧ ¬C ∧ (D \ {¬l}) ⊢₁ ⊥` (P12 §3). "Addition and
  removal of blocked clauses results in satisfiability-equivalent formulas, but not logically equivalent
  formulas" (P12 §2).
* PR: witness is a partial assignment τ ⊨ C with `F|α ⊢₁ F|τ` (α = ¬C).
* SR (P13 Def 1.13): "A clause C is substitution redundant (SR) with respect to Γ if there is a substitution
  τ such that τ ⊨ C and Γ|α ⊢₁ Γ|τ." Equivalent form used by P5: `F ∧ ¬C ⊢₁ (F ∧ C)|σ`; Theorem 1: "If C is
  SR for F, then C is redundant for F", proof by repairing τ into τ ∘ σ. P13: "All of these rules can be
  viewed as allowing the introduction of clauses that hold 'without loss of generality'."
* VeriPB redundance-based strengthening (P10 Def 6): with witness ω,
  `C ∪ D ∪ {f <= v-1} ∪ {¬C} ⊢ (C ∪ D ∪ C)|ω ∪ {f|ω <= f} ∪ O⪯(z|ω, z)` (5); dominance-based
  strengthening (Def 13): `C ∪ D ∪ {f <= v-1} ∪ {¬C} ⊢ C|ω ∪ O⪯(z|ω, z) ∪ {f|ω <= f}` (10a) and
  `C ∪ D ∪ {f <= v-1} ∪ {¬C} ∪ O⪯(z, z|ω) ⊢ ⊥` (10b) -- the witness only needs to reach a *strictly
  smaller* assignment in a proved preorder, not a satisfying one for `D ∪ {C}`; lex-leader clauses (19a)-(19f)
  are derived from `C_LL = Σ 2^{m-i} (σ(x_i) - x_i) >= 0` (20). "In contrast to the approach of Heule et al.
  (2015), handling a symmetry once is enough to guarantee complete breaking." The deletion rule must be
  restricted when dominance is used (Example 15 derives contradiction from a satisfiable formula if
  arbitrary deletion is allowed) -- a subtlety for anyone hand-rolling this.
* DRAT-only symmetry breaking (P12): three steps per symmetry -- add definitions (primal-swap variable
  `s_1`, `6n-3` blocked clauses, primed copies `x'_i`), redefine each involved clause (`4m` DRAT
  operations), add the lex predicate clauses (2)-(3) with RAT on the first literal; "expressing breaking
  k > 1 symmetries in DRAT cannot simply be done by applying the above procedure ... once for each symmetry
  ... in worst case the procedure needs to be applied significantly more often than k times"; P10 §4.2: even a
  3-cycle "brings us beyond what DRAT-based proof logging symmetry breaking is currently able to handle".

### 6.2 The cube-cover obligation (P14)
"A cube partitioning is valid, i.e., covers the complete search space, if the disjunction of cubes is a
tautology ... Checking this can be done by negating the disjunction of cubes and feed the result to a CDCL
solver which supports proof logging ... We refer to the proof emitted by the CDCL solver as the tautology
proof." Schur Number Five: "The tautology proof shows that the disjunction of cubes is a tautology, i.e.,
the cubes together cover the entire search space"; the final proof merges transformation proof, cube proofs
and tautology proof. P2's Lemma 17 is the Lean statement of the same composition.

### 6.3 Mapping onto ZarPrune (inference, but forced by the definitions)
| pipeline object | proof-logging notion | what certifies it |
|---|---|---|
| `Prune.sound : kill (profileOf A) = true → ¬ Valid P A` | *implied* (the cube is refutable: `F ∧ cube ⊨ ⊥`) | a Lean proof (counting lemma) -- or in principle a RUP derivation of `¬cube` |
| `refuted q` | leaf UNSAT | LRAT via `check_sound` |
| `cover` | tautology of the cube disjunction | decidable enumeration proof or LRAT of the negated cubes (P2 Lemma 17 shape) |
| "assume rows sorted", column lex, `row_degmono` | *redundant/adding* (SR, dominance, lex-leader) | witness-based certificate (SR via `dsr-trim`+`lsr-check`, VeriPB, or Lean canonicalization lemma at the `∃ A` level); **never** a `Prune` |
`Demo.notDescending_unsound` is exactly the formal statement that a lex/sorting predicate is not a prune.
Keller's rule for where the seam goes: prove in Lean the WLOG steps that the certificate language cannot
express, put the transition point as early as possible, and let SR/LRAT do the rest.

### 6.4 Consequences for the row/column-sum case split
The case split is by *profiles* (`Profile m n`, a vector) whereas Tan enumerates unordered partitions. A
prune on profiles is sound for every matrix; "sort the rows" is what turns profiles into partitions and it
is an addition justified by the row-permutation symmetry of `HasKst` and `weight`. Two sound ways to use
it: (i) prove once in Lean `(∃ A, Valid P A) → ∃ A', Valid P A' ∧ rowSum A' antitone ∧ colSum A' antitone`
(a canonicalization lemma at the `cover` seam, analogous to P6 Theorem 2/Corollary 3), and then enumerate
only sorted profiles; or (ii) keep the case split on all profiles and let the solver's symmetry breaking be
certified by SR/VeriPB outside the `Prune` type. Option (i) is Mathlib-free provable but needs `Fin`
permutation/sorting infrastructure (see 7.4); option (ii) needs no new Lean but reintroduces an external
verified checker (`lsr-check` is unverified; the verified LSR checker is Trestle, Mathlib) or CakePB.

---

## 7. Extracted lemmas, with hypotheses and honest Lean-provability notes (ZarPrune, no Mathlib)

7.1 **`refuted_of_lrat`** (the `refuted` seam). Hypotheses: a computable `encode : (P : Params) → Profile P.m
P.n → Std.Sat.CNF Nat`; `assign : Mat P.m P.n → Nat → Bool` (cells plus auxiliary variables);
`complete : ∀ A, Valid P A → profileOf A = q → (encode P q).Sat (assign A)`; `h : LRAT.check proof (encode P q)
= true`. Conclusion: `∀ A, profileOf A = q → ¬ Valid P A`. Proof: `check_sound` gives `(encode P q).Unsat`,
i.e. `∀ a, eval a (encode P q) = false`, contradiction with `complete`. Provability: `check_sound` is in core;
`complete` is the real work: for the `K_{s,t}` clauses (one clause per increasing tuple, matching `HasKst`'s
`Incr`) the witness is the cell assignment and the proof is by unfolding `Clause.eval`; for cardinality we
must define the auxiliary values (sequential counter: `s_{i,j} := decide (Σ_{l<=i} x_l >= j)`; totalizer:
recursive unary sums) and prove each clause -- a few hundred lines, comparable to P4's per-encoding proofs
(P4 reports 30-line to 300-line proofs per encoding). The native axiom for `h` is unavoidable at scale (2.4).

7.2 **`cover_of_negcubes`** (P2 Lemma 17 shape). Hypotheses: `cubes : List (List (Literal Nat))`,
`hcover : (negCubesCNF cubes).Unsat` where `negCubesCNF` has one clause `¬c` per cube. Conclusion: `∀ τ, ∃ c
∈ cubes, ∀ l ∈ c, τ satisfies l`. Provability: easy (30 lines) from `Unsat`; for us the cubes are the unit
clauses fixing row/column sums, so an additional lemma `profileOf A = q ↔ assign A satisfies cube_q` is
needed -- this is the same encoding-completeness work as 7.1 restricted to the cardinality part.

7.3 **`cover_by_enumeration`** (alternative to 7.2). Hypotheses: `enum : List (Profile m n)` computed by the
harness; `mem : ∀ q, (∀ i, q.row i <= n) → (∀ j, q.col j <= m) → sumFin m q.row = sumFin n q.col → q ∈ enum`.
Conclusion: `cover` in the form required by `upper_bound_of_cover` after applying `kill` and the survivor
filter. Provability: `rowSum_le`, `colSum_le`, `weight_eq_sum_colSum` exist; the enumeration-completeness
proof needs a decidable-membership computation over `Fin m → Nat` bounded functions (e.g. encode a profile
as a `List Nat` and prove `List` membership by `decide` on a finite product of ranges). Moderate;
Mathlib's `Finset.pi` would make it trivial, hand-rolled it is a few hundred lines. It avoids any SAT call
for the cover.

7.4 **`valid_of_perm`** (row/column permutation symmetry; the "adding" lemma at the `∃` level). Hypotheses:
`π : Fin m → Fin m` bijective; `A' i j := A (π i) j`. Conclusion: `Valid P A' ↔ Valid P A` (hence
`(∃ A, Valid P A) → ∃ A', Valid P A' ∧ rowSum A' antitone`). Provability: `weight` invariance is a
`sumFin` reindexing lemma (need `sumFin_perm`, not yet in `Sum.lean`); `HasKst` invariance needs to map an
increasing `s`-tuple through `π⁻¹` and re-sort it into an increasing tuple (because `HasKst` insists on
`Incr`) -- i.e. a sorting lemma for injective `Fin s → Fin m`. Hard-ish without Mathlib (`Fin`
permutations, sorting an injective tuple: 300-600 lines). This is the analogue of P6 §5.1-5.2 ("We also prove
in Lean that each symmetry is, in fact, a symmetry ... amount to providing the inverse of each symmetry";
their initial pen-and-paper-following proof was "350 lines of uninteresting code").

7.5 **SR redundancy** (P13 Def 1.13 / P5 Thm 1). Hypotheses: CNF `F`, clause `C`, substitution σ, `σ ⊨ C`,
`F ∧ ¬C ⊢₁ (F ∧ C)|σ`. Conclusion: `Sat F → Sat (F ∧ C)`. Not a prune. Provability in ZarPrune: needs
unit propagation and substitution over `CNF`; medium-hard (P5's verified checker is 8k LoC with data
structures; the pure soundness theorem is much smaller, maybe 500 lines). Only needed if we want to check
SR certificates *inside* Lean rather than with `lsr-check`/CakePB.

7.6 **VeriPB redundance/dominance** (P10 Def 6 / Def 13): statements as in 6.1; PBLean (P3) already has
Lean 4, Mathlib-free soundness lemmas (`applySubstConstr_sat_rev : C|ω.sat v → C.sat (ω(v))`,
`constr_sat_noSubst`) on toolchain 4.28.0-rc1; porting to 4.34 is plausibly mechanical (inference).

7.7 **Certified CNF translation** (P11 Prop 4): from an arithmetic graph for `Σ a_i ℓ_i ⋈ k`, cutting
planes derive `Σ c_i o_i ⋈ k`; consequence: the CNF translation is derivable from the PB constraint, so
UNSAT(CNF) ⇒ UNSAT(PB). Not something to prove in Lean; it is the VeriPB-side alternative to 7.1's
`complete` if we ever move to a PB solver + PBLean.

---

## 8. Difficulty signals for a branch (what the sources measure)
* LRAT certificate size in bytes and action lines (P2: check CPU is roughly linear, 140-150 CPU-s per GB;
  memory 2.05 GB + 0.28 GB per GB of largest leaf; 10.3: 151 MB in 16 s, 7.4 MB in 2.9 s).
* Solver CPU per leaf (P6 Table 1: check time 0.5-1x solve time; P2: import took "about half of the solve
  CPU" for R(4,4), "about two thirds" for queen domination, "about a quarter" for w(2;3,18)).
* Leaf-size distribution is long-tailed (P2: median 38 kB / 0.3 MB vs max 16.8 GB / 7.4 GB; 19 of 909,558
  leaves needed re-cubing at depth 8 into 256 sub-cubes each) -- budget for the tail, not the median.
* Number of symmetry-breaking (SR) clauses needed before the formula becomes tractable (P6: `n=7` was
  intractable without them; 385-2,582 SR clauses).
* Schur Number Five uses backbone size after symmetry breaking as "a useful rough measure for the hardness
  of subproblems" (P14).
* Direct `bv_decide` route scaling on K_{2,2}-free n x n: 0.46 s (4x4), 0.47 s (5x5), 1.3 s (6x6), 20 s
  (7x7, 1.62 GB) -- about 15x per step at n=7; usable only as a small-instance oracle (10.5).
* Certificate size is also a *reward-safe* proxy: it is produced by the untrusted solver but only after a
  verified check, so an evolved prune can be credited with `Σ_{killed q} (solve_q + check_q)` estimated
  from the size/time of the same or similar branches that were actually refuted.

---

## 9. Design implications (concrete)
1. Two trust domains, one gate each. LLM prune modules: zero non-standard axioms, source blacklist, harness-
   owned statement, sandboxed compile, `leanchecker`, `#eval` smoke test (4.4). Harness refutation modules:
   `Std.Tactic.BVDecide.LRAT.check_sound` + `native_decide`, one `_native` axiom per branch, names recorded
   and re-matched; run `leanchecker` on them too (it replays the axiom but catches env hacks).
2. Refutation route now (Lean 4.34, no Mathlib): write `ZarPrune/Encode.lean` (`encode`, `assign`,
   `complete`, 7.1) over `Std.Sat.CNF Nat`; emit DIMACS with `CNF.dimacs` (0-based to 1-based shift), run
   the bundled `cadical 2.1.2 --lrat --binary=false` (or `loadLRATProof` on the binary file), optionally
   `LRAT.trim`, and prove `(encode P q).Unsat` by reflection. Keep the LRAT out of the `.lean` source for
   big leaves (my experiment embedded a 4.5 MB string; P2's file/stream modes exist for the TB regime).
   If CaDiCaL >= 3.0.0 is used, pass `--no-factor`. Kissat: DRAT only, convert with `drat-trim -L`.
3. Prototype/oracle route: `bv_decide` on `x : BitVec (m*n)` with `getLsbD` conjunction hypotheses and
   `x.cpop <= z` proves `z(n,n;2,2)` upper bounds up to n=7 in 20 s with no encoding proof at all (the
   Tseitin/bitblast encoding is verified in core). Use it to cross-check `Encode.lean` on tiny instances and
   as a fallback certifier for tiny branches.
4. Scale route: adopt LRAT-Catcher's chunk/stream lemmas (P2 is MIT-licensed, Lean v4.30.0; the core API it
   wraps is unchanged in 4.34) rather than writing our own streaming; or, as a labelled fallback, record a
   `cake_lpr` verdict as an axiom, exactly as P7 did, and say so in the bound table.
5. Symmetry breaking leaves `encodings_zar.py`'s branch CNF or gets a witness: either prove 7.4 once and
   enumerate sorted profiles, or certify the lex clauses with SR (`dsr-trim` + `lsr-check`/Trestle) outside
   the `Prune` type. Never let the evolutionary loop see symmetry-breaking as a prune (the verifier already
   refuses it; the reward function must not reward it either).
6. The `cover` seam: prefer 7.3 (enumeration proof, no solver) for profile splits; use 7.2 (negated-cube
   LRAT) only if we move to solver-chosen cubes (AlphaMapleSAT-style).
7. Difficulty/reward: credit prunes by the measured or estimated (solve + check) cost of killed branches;
   use LRAT bytes and solver seconds from actually-refuted sibling branches; treat the top percentile of
   leaves as the cost driver (8).
8. Versions: keep ZarPrune on a toolchain >= 4.29 (per-evaluation axioms; `leanchecker` in core) and add
   `leanOptions = [{autoImplicit = false}]` to `lakefile.toml`. Do not depend on Trestle (4.21 + Mathlib) or
   PBLean (4.28-rc1) directly; copy their definitions.
9. Reporting honesty: each bound row should list its axiom set: `{propext, Quot.sound}` for Lean-only
   prunes; `+ Classical.choice` if used; `+ k native-evaluation axioms` for k refuted branches; `+ 1 cake_lpr
   axiom` if the fallback was used. This matches how P2 Table 5 and P7 report.

---

## 10. Experiment log (all on this machine: Darwin 25.6.0 arm64, Lean 4.34.0, bundled CaDiCaL 2.1.2)
Scratch project: `.../scratchpad/lratexp/` (`lakefile.toml`, `LratExp/*.lean`, `gen.py`, `gen2.py`).

10.1 `leanchecker` on ZarPrune: `lake env leanchecker ZarPrune` 0.66 s wall (parallel per-module); `lake env
leanchecker --fresh ZarPrune` 29.4 s (single-threaded replay incl. imports); both exit 0.

10.2 Axiom reporting (`LratExp/Native.lean`): `native_decide` → `[two_plus_two._native.native_decide.ax_1_1]`;
`decide +native` → `[two_plus_two'._native.decide.ax_1_1]`; `decide` → no axioms; `sorry` → `[sorryAx]`.

10.3 LRAT import via `LRAT.check_sound _ _ (by native_decide)`, ASCII LRAT embedded as a string literal, no
trimming; times are `lake build` wall for the one module (includes parsing the source literal):

| instance (UNSAT) | vars | clauses | CaDiCaL solve | LRAT size (lines) | Lean import wall | max RSS |
|---|---|---|---|---|---|---|
| PHP(3,2) | 6 | 9 | -- | 107 B (7) | 0.55 s; `decide +kernel` variant: stuck | -- |
| PHP(6,5) | 30 | 81 | -- | 9.6 KB (193) | 0.35 s | -- |
| PHP(8,7) | 56 | 204 | -- | 0.80 MB (9,865) | 0.57 s | -- |
| PHP(9,8) | 72 | 297 | -- | 4.47 MB (49,570) | 0.89 s | -- |
| K22-free 4x4, >=10 ones | 80 | 221 | -- | 11.5 KB (219) | 0.57 s | -- |
| K22-free 5x5, >=13 | 143 | 519 | -- | 0.56 MB (6,219) | 1.1 s | -- |
| K22-free 6x6, >=17 | 224 | 1,044 | 0.45 s | 7.40 MB (66,261) | 2.9 s | 776 MB |
| K22-free 7x7, >=22 | 328 | 1,897 | 14.7 s | 151.5 MB (1,128,453) | 16.2 s | 2.35 GB |
Cardinality via PySAT totalizer; K22 clauses one per increasing 2x2 tuple. All theorems depend on
`[propext, Classical.choice, Quot.sound, <decl>._native.native_decide.ax_1_1]`. These are genuine (tiny)
Zarankiewicz upper-bound refutations `z(n,n;2,2) < w` at the CNF level (the encoding is unverified here).

10.4 Spoof: `axiom spoof._native.native_decide.ax_1_1 : False` + `theorem bad : 1 = 2 := ...elim` builds,
`#print axioms bad` = `[spoof._native.native_decide.ax_1_1]`, `lake env leanchecker LratExp.Spoof` exits 0.
`lake env leanchecker LratExp.Php65` (genuine native axiom) also exits 0 in 0.43 s -- `leanchecker` does not
re-evaluate. A one-line grep for `^\s*axiom` catches the spoof.

10.5 `bv_decide` direct route, `theorem (x : BitVec (n*n)) (h_k : (x.getLsbD a && ... ) = false)... :
x.cpop <= z#(n*n) := by bv_decide`: 4x4 z=9: 0.46 s; 5x5 z=12: 0.47 s; 6x6 z=16: 1.3 s (684 MB); 7x7 z=21:
20.0 s wall, 1.62 GB RSS (needed `(config := { timeout := 300 })` since the default solver timeout is 10 s).
Axioms: `[propext, Classical.choice, Quot.sound, bvz77._native.bv_decide.ax_1_5]`.

10.6 Toolchain facts checked by `ls`/`grep`: `bin/{cadical,leanchecker,lean,lake,leanc,...}`; `cadical
--version` 2.1.2; `cadical --help` lists `--lrat`, `--binary`, `--frat`, `--idrup`, no `factor` option (BVA
arrived in CaDiCaL 2.2.0 per NEWS); no `partial`/`implemented_by`/`extern` in `LRAT/Internal`; `autoImplicit`
default `true`; `Lean/Replay.lean` and `LeanChecker.lean` in core.

---

## 11. Open questions
1. How to state `refuted` for *partitions* (unordered) rather than profiles without paying for 7.4 twice --
   can the enumeration proof 7.3 be done over sorted profiles only if 7.4 is proved once at the top?
2. Is a fuel-based, structurally recursive LRAT checker (kernel-reducible, no native axiom) feasible for
   leaves in the 10-100 KB range? Nobody has published one; it would remove the native axiom for small
   branches at a large time cost (P16 replayed a petabyte through HOL4's kernel, so it is not absurd).
3. Trestle-style `VEncCNF` without Mathlib: how much of `Finset`/`Multiset` reasoning is really needed for
   the totalizer witness? (Sequential counter needs only prefix sums; the totalizer needs a tree of unary
   sums.) Choose the encoding by proof cost, not solver speed, for the first version.
4. Re-evaluating `_native` axioms as an independent check (the RFC's suggestion): a small harness tool that
   re-runs `evalConst` on each recorded `<decl>._native.*.decl` would close the spoofing gap without a
   blacklist; is `comparator --paranoid` on Linux the better investment?
5. Which symmetry certificate to adopt if we go beyond sorting: SR (`dsr-trim`/`lsr-check`, unverified in
   our stack unless Trestle is ported) versus VeriPB dominance (CakePB verified; PBLean's lemmas are
   Mathlib-free but on 4.28-rc1). P10's "handling a symmetry once is enough" is a strong argument for
   dominance if the search evolves many symmetries.
6. Are LRAT-Catcher's `--no-factor` and RUP-only observations still true for CaDiCaL 3.0.1 (factor off by
   default again)? Verify on our leaves before trusting either default.

"""Tests for the Lean gate (zar_ub/lean_gate.py; design §4.2-4.4).

Static tests (no Lean) cover the scan: one test per forbidden token, the nonce
forgery, NFKC look-alikes, the declared-name rule, the sorry-hole rule and the
set_option whitelist.  Lean tests (each ~2 s, `lake env lean` on the built
ZarPrune library) cover the ladder end to end: good (L5), sorry (L0 / L2),
cheat (L1), instance-specific (L5 on its instance, L1 elsewhere), an
omega-fillable hole (auto-fill -> L5), an unfillable hole (L2 with goal text),
schema terms, the cache and the axiom audit.

Run:  cd examples/zarankiewicz/upper_bounds && python -m unittest tests.test_gate
"""
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zar_ub import Instance  # noqa: E402
from zar_ub.casetable import load_table  # noqa: E402
from zar_ub import lean_gate as lg  # noqa: E402

GOOD = """
/-- re-proved deficit -/
def myDeficit (P : Params) : Prune P where
  name := "deficit'"
  kill := fun pf => decide (sumFin P.m pf.row < P.w)
  sound := by
    intro A h hv
    have h1 : sumFin P.m (rowSum A) < P.w := of_decide_eq_true h
    have h2 : P.w ≤ sumFin P.m (rowSum A) := hv.2
    omega
def candidate (P : Params) : Prune P := Prune.or (myDeficit P) (counting P)
"""
# the hole `h3` is closed by omega from h1/h2 (auto-fill S7) -- the sketch is a complete proof
HOLE_FILLABLE = """
def myDeficit (P : Params) : Prune P where
  name := "deficit'"
  kill := fun pf => decide (sumFin P.m pf.row < P.w)
  sound := by
    intro A h hv
    have h1 : sumFin P.m (rowSum A) < P.w := of_decide_eq_true h
    have h2 : P.w ≤ sumFin P.m (rowSum A) := hv.2
    have h3 : sumFin P.m (rowSum A) < sumFin P.m (rowSum A) := by sorry
    omega
def candidate (P : Params) : Prune P := myDeficit P
"""
# h2 is false (no tactic can close it); the `sorry` on its own line is also accepted
HOLE_STUCK = """
def myDeficit (P : Params) : Prune P where
  name := "deficit'"
  kill := fun pf => decide (sumFin P.m pf.row < P.w)
  sound := by
    intro A h hv
    have h1 : sumFin P.m (rowSum A) < P.w := of_decide_eq_true h
    have h2 : sumFin P.m (rowSum A) = 17 := by
      sorry
    have h4 : P.w ≤ sumFin P.m (rowSum A) := hv.2
    omega
def candidate (P : Params) : Prune P := myDeficit P
"""
CHEAT = """
def candidate (P : Params) : Prune P where
  name := "cheat"
  kill := fun _ => true
  sound := by intro A h hv; exact absurd hv.2 (by decide)
"""
SPECIFIC = "def candidate : Prune target := Prune.or (deficit target) (rowCap target)\n"
NEVER = "def candidate (P : Params) : Prune P := Prune.never P\n"

_INST = Instance(9, 9, 3, 3, 50)
_INST2 = Instance(10, 10, 3, 3, 61)


def _cases(inst):
    tab = load_table(inst, use_table=False)
    return [(r.rows, r.cols) for r in tab.records] if tab else None


def _has_lean():
    return shutil.which("lake") is not None and os.path.isdir(os.path.join(lg.LEAN_DIR, ".lake"))


# ---------------------------------------------------------------------------
# static scan
# ---------------------------------------------------------------------------
class TestForbiddenTokens(unittest.TestCase):
    """One assertion per forbidden construct of design §4.3 (each wrapped in an
    otherwise legal candidate)."""
    SAMPLES = {
        "admit": "theorem t (P : Params) (A : Mat P.m P.n) : ¬ Valid P A := by admit",
        "native_decide": "theorem t : 1 + 1 = 2 := by native_decide",
        "axiom": "axiom foo : False",
        "unsafe": "unsafe def f : Nat := 1",
        "implemented_by": "@[implemented_by f] def g : Nat := 1",
        "extern": "@[extern \"x\"] def g : Nat := 1",
        "csimp": "@[csimp] theorem t : f = g := rfl",
        "opaque": "opaque x : Nat",
        "partial": "partial def loop (n : Nat) : Nat := loop n",
        "import": "import Mathlib",
        "macro": "macro \"foo\" : tactic => `(tactic| sorry)",
        "macro_rules": "macro_rules | `(tactic| foo) => `(tactic| rfl)",
        "elab": "elab \"foo\" : tactic => pure ()",
        "syntax": "syntax \"foo\" : tactic",
        "notation": "notation \"⟪\" x \"⟫\" => x",
        "initialize": "initialize foo : Nat ← pure 1",
        "Lean.": "def x := Lean.Name.anonymous",
        "IO": "def x : IO Unit := pure ()",
        "ofReduceBool": "theorem t : true = true := Lean.ofReduceBool _ _ rfl",
        "end ZarPrune": "end ZarPrune\ndef candidate : Nat := 1",
        "end Cand": "end Cand\nnamespace Cand",
        "+native": "theorem t : 1 + 1 = 2 := by decide +native",
        "+kernel": "theorem t : 1 + 1 = 2 := by decide +kernel",
        "trustCompiler": "theorem t : True := Lean.trustCompiler",
        "run_cmd": "run_cmd pure ()",
        "run_tac": "theorem t : True := by run_tac pure ()",
        "run_elab": "run_elab pure ()",
        "open Lean": "open Lean Elab in\ndef x := 1",
        "attribute [": "attribute [simp] Nat.add_comm",
        "@[simp]": "@[simp] theorem t : 1 = 1 := rfl",
        "@[csimp]": "@[csimp] theorem t : f = g := rfl",
        "@[implemented_by]": "@[implemented_by g] def f : Nat := 1",
        "local instance": "local instance : Inhabited Nat := ⟨0⟩",
        "instance : Decidable": "instance : Decidable (HasKst P A) := isTrue sorry",
        "noncomputable": "noncomputable def f : Nat := Classical.choice ⟨0⟩",
        "dbg_trace": "def f : Nat := dbg_trace \"x\" fun _ => 1",
        "trace": "theorem t : True := by trace \"x\"; trivial",
        "trace_state": "theorem t : True := by trace_state; trivial",
        "logInfo": "def f := logInfo \"x\"",
        "#eval": "#eval 1",
        "#print": "#print axioms foo",
        "#check": "#check Nat",
        "#reduce": "#reduce 1 + 1",
        "#guard": "#guard 1 = 1",
        "#synth": "#synth Inhabited Nat",
        "#exit": "#exit",
        "#help": "#help tactic",
        "namespace": "namespace Foo\nend Foo",
        "deriving other than Repr/DecidableEq": "structure S where\n  x : Nat\nderiving Repr, Inhabited",
        "deriving instance": "deriving instance Repr for Params",
        "decreasing_by": "def f (n : Nat) : Nat := f (n - 1)\ndecreasing_by sorry",
        "set_option (other)": "set_option pp.all true",
        "set_option maxHeartbeats too high": "set_option maxHeartbeats 400001 in\ndef x := 1",
        "set_option maxRecDepth too high": "set_option maxRecDepth 4097 in\ndef x := 1",
    }

    def test_each_forbidden_token(self):
        for name, snippet in self.SAMPLES.items():
            with self.subTest(token=name):
                hits = lg.static_scan(NEVER + snippet + "\n")
                self.assertTrue(hits, f"{name!r} was not caught: {snippet!r}")

    def test_hits_in_comments_and_strings_are_caught(self):
        # the raw text is scanned too (design §4.3): a comment cannot hide a token
        self.assertTrue(lg.static_scan(NEVER + "-- #eval 1\n"))
        self.assertTrue(lg.static_scan(NEVER + "/- IO.println -/\n"))
        self.assertTrue(lg.static_scan(NEVER + 'def s := "#eval"\n'))

    def test_allowed_constructs_pass(self):
        ok = [GOOD, NEVER, SPECIFIC, CHEAT, HOLE_FILLABLE, HOLE_STUCK,
              NEVER + "set_option maxHeartbeats 400000 in\ntheorem t : True := trivial\n",
              NEVER + "set_option maxRecDepth 4096\n",
              NEVER + "structure S where\n  x : Nat\nderiving Repr, DecidableEq\n",
              NEVER + "def arr := #[1, 2]\n",   # array literal is not a # command
              NEVER + "open Classical in\ntheorem t : True := trivial\n",
              NEVER + "def weightBound (P : Params) : Nat := P.w\n"]  # not a protected name
        for src in ok:
            with self.subTest(src=src[:60]):
                self.assertEqual(lg.static_scan(src), [])


class TestScanRules(unittest.TestCase):
    def test_nonce_forgery_is_L0(self):
        forged = NEVER + '#eval IO.println "MASK 0123456789abcdef 0 BEGIN"\n#eval IO.println "MASKLINE:true"\n'
        hits = lg.static_scan(forged)
        self.assertTrue(any(h.startswith("# command") for h in hits))
        self.assertTrue(any(h.startswith("IO") for h in hits))
        res = lg.run_gate_multi([(_INST, [([9] * 9, [9] * 9)])], forged, use_cache=False)
        self.assertEqual(res[0].ladder, 0)
        self.assertFalse(res[0].ok)
        self.assertIsNone(res[0].kill_mask)

    def test_nfkc_lookalikes(self):
        # fullwidth `＃`, fullwidth letters and ligatures fold to the forbidden tokens
        self.assertTrue(lg.static_scan(NEVER + "＃eval 1\n"))
        self.assertTrue(lg.static_scan(NEVER + "theorem t : 1 = 1 := by ｎａｔｉｖｅ_ｄｅｃｉｄｅ\n"))
        self.assertTrue(lg.static_scan(NEVER + "def x : ＩＯ Unit := pure ()\n"))
        # legal math letters are NOT damaged: ℕ / subscripts stay legal, and the raw text goes to Lean
        src = "def f (n : ℕ) (h₁ : n = n) : ℕ := n\n" + NEVER
        self.assertEqual(lg.static_scan(src), [])
        self.assertIn("ℕ", lg.build_gate_file(_INST, src, None)[0])

    def test_declared_name_rule(self):
        for decl in ["def Valid (P : Params) (A : Mat P.m P.n) : Prop := True",
                     "theorem HasKst : True := trivial", "abbrev Params := Nat",
                     "structure Profile where\n  x : Nat", "def weight : Nat := 0",
                     "def profileOf : Nat := 0", "def Prune.or : Nat := 0", "def counting : Nat := 0",
                     "abbrev target : Params := ⟨1,1,1,1,1⟩", "def target1 : Nat := 0", "def gateInst : Nat := 0",
                     "def gateInst3 : Nat := 0", "def gateProfile : Nat := 0", "def schema0 : Nat := 0",
                     "def candInst : Nat := 0", "private def Valid : Nat := 0", "def CondPrune : Nat := 0",
                     "def Fact : Nat := 0", "def FactHolds : Nat := 0", "def Mat : Nat := 0",
                     "def rowSum : Nat := 0", "def colSum : Nat := 0", "def baseline : Nat := 0"]:
            with self.subTest(decl=decl):
                hits = lg.static_scan(NEVER + decl + "\n")
                self.assertTrue(any("declared protected name" in h for h in hits), decl)

    def test_sorry_hole_rule(self):
        # accepted forms
        self.assertEqual(lg.static_scan(HOLE_FILLABLE), [])
        self.assertEqual(lg.static_scan(HOLE_STUCK), [])
        thm = ("theorem aux (P : Params) (A : Mat P.m P.n) (h : P.w = 0) : ¬ Valid P A := by\n"
               "  intro hv\n  have hx : P.w ≤ weight A := by sorry\n  exact absurd hv.2 (by omega)\n" + NEVER)
        self.assertEqual(lg.static_scan(thm), [])
        # rejected forms: sorry outside a `sound := by` block, or hidden in a comment
        bad = ["def candidate (P : Params) : Prune P := by sorry\n",
               NEVER + "theorem t : True := by sorry\n",                       # not a ¬Valid theorem
               NEVER + "-- sorry\n"]                                          # sorry in a comment
        for src in bad:
            with self.subTest(src=src[:80]):
                self.assertTrue(lg.static_scan(src), src)
        # E22 relaxation: a sorry inside `sound := by` that is not a typed `have` hole is an
        # UNTYPED hole -- accepted by the scan, never auto-filled, ladder capped at L2
        untyped = [HOLE_FILLABLE.replace("have h3 : sumFin P.m (rowSum A) < sumFin P.m (rowSum A) := by sorry",
                                         "have h3 : sumFin P.m (rowSum A) < sumFin P.m (rowSum A) := by simp; sorry"),
                   HOLE_FILLABLE.replace(":= by sorry", ":= sorry"),
                   HOLE_FILLABLE.replace("have h3 :", "have :")]              # anonymous have
        for src in untyped:
            with self.subTest(src=src[:80]):
                self.assertEqual(lg.static_scan(src), [], src)
                holes, viol = lg.find_holes(lg.strip_comments(src))
                self.assertEqual(viol, [])
                self.assertTrue(any(h.get("untyped") for h in holes), holes)
        # non-sketch mode: no sorry at all
        self.assertIn("sorry", lg.static_scan(HOLE_FILLABLE, sketch=False))
        holes, viol = lg.find_holes(lg.strip_comments(HOLE_STUCK))
        self.assertEqual(viol, [])
        self.assertEqual(len(holes), 1)
        self.assertEqual(holes[0]["statement"], "have h2 : sumFin P.m (rowSum A) = 17")

    def test_strip_comments_keeps_lines_and_handles_nesting(self):
        src = "a /- x /- y -/ z -/ b\n-- c\nd \"--e\" f\n"
        out = lg.strip_comments(src)
        self.assertEqual(out.count("\n"), src.count("\n"))
        self.assertIn("a", out); self.assertIn("b", out); self.assertNotIn("y", out); self.assertNotIn("c", out)
        self.assertIn("f", out); self.assertNotIn("e", out)

    def test_wrapper_and_cache_key(self):
        cases = [([9] * 9, [9] * 9)]
        src, offset, blocks = lg.build_gate_file(_INST, NEVER, cases, extra_instances=[(_INST2, cases)],
                                                  nonce="ab" * 8, schema_terms=[None, ["ZarPrune.argD ZarPrune.Cand.target1"]])
        self.assertIn("set_option autoImplicit false", src)
        self.assertIn('MASK abababababababab 0 BEGIN', src)
        self.assertIn('SMASK abababababababab 1 BEGIN', src)
        self.assertIn("def ZarPrune.Cand.schema1", src)
        self.assertNotIn("def ZarPrune.Cand.schema :", src)
        self.assertEqual(src.splitlines()[offset], NEVER.rstrip("\n"))
        tags = {t for _, t in blocks}
        self.assertTrue({"gateInst0", "gateInst1", "axioms0", "mask0", "mask1", "smask1", "target1"} <= tags)
        k1 = lg.cache_key(NEVER, None, [(_INST, cases)], True)
        self.assertNotEqual(k1, lg.cache_key(NEVER + " ", None, [(_INST, cases)], True))
        self.assertNotEqual(k1, lg.cache_key(NEVER, [["x"]], [(_INST, cases)], True))
        self.assertNotEqual(k1, lg.cache_key(NEVER, None, [(_INST, cases * 2)], True))
        self.assertNotEqual(k1, lg.cache_key(NEVER, None, [(_INST, cases)], False))

    def test_lean_partial_formula(self):
        r = lg.GateResult(ladder=5)
        self.assertEqual(r.lean_partial, 1.0)
        for lad in (0, 4):
            self.assertEqual(lg.GateResult(ladder=lad).lean_partial, 0.0)
        r = lg.GateResult(ladder=1, n_decls=4, n_decls_ok=2, first_error_line=6)
        r._n_body = 10
        self.assertAlmostEqual(r.lean_partial, 0.05 + 0.10 * 0.5 + 0.05 * 0.5)
        r = lg.GateResult(ladder=2, n_decls=2, n_decls_ok=2)
        r._n_body = 10
        self.assertAlmostEqual(r.lean_partial, 0.25 + 0.15 + 0.10)
        r = lg.GateResult(ladder=3, n_holes=4, n_holes_filled=1)
        r._n_body = 10
        self.assertAlmostEqual(r.lean_partial, 0.50 + 0.30 * 0.25 + 0.10)
        self.assertEqual(r.partial_credit, r.lean_partial)

    def test_gate_result_fields(self):
        spec = ["ok", "ladder", "scanned_ok", "compiled", "typed_ok", "axioms", "axioms_ok", "kill_mask",
                "schema_mask", "errors", "holes", "n_holes", "n_holes_filled", "filled_source", "forbidden",
                "n_decls", "n_decls_ok", "first_error_line", "first_error_frac", "seconds", "timed_out",
                "lean_file", "stdout_tail", "cache_hit", "nonce"]
        fields = set(lg.GateResult.__dataclass_fields__)
        self.assertTrue(set(spec) <= fields, set(spec) - fields)


# ---------------------------------------------------------------------------
# end to end (real Lean)
# ---------------------------------------------------------------------------
@unittest.skipUnless(_has_lean(), "lake / built ZarPrune library not available")
class TestGateLean(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = _cases(_INST)
        cls.cases2 = _cases(_INST2)
        cls.tmp = tempfile.mkdtemp(prefix="gatecache_")
        if cls.cases is None:
            raise unittest.SkipTest("cache/case_table_m9_n9_s3_t3_w50_pure.json missing")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def gate(self, src, insts=None, **kw):
        insts = insts or [(_INST, self.cases)]
        kw.setdefault("cache_dir", self.tmp)
        kw.setdefault("tag", "test")
        return lg.run_gate_multi(insts, src, timeout=120, **kw)

    def test_good_is_L5_and_cached(self):
        res = self.gate(GOOD, [(_INST, self.cases), (_INST2, self.cases2)], use_cache=True)
        for r in res:
            self.assertEqual(r.ladder, 5, r.errors)
            self.assertTrue(r.ok)
            self.assertTrue(set(r.axioms) <= lg.ALLOWED_AXIOMS)
            self.assertEqual(len(r.nonce), 16)
            self.assertFalse(r.cache_hit)
            self.assertEqual(r.lean_partial, 1.0)
        self.assertEqual(len(res[0].kill_mask), len(self.cases))
        self.assertEqual(len(res[1].kill_mask), len(self.cases2))
        self.assertGreater(sum(res[0].kill_mask), 0)  # counting kills something at (9,9)
        again = self.gate(GOOD, [(_INST, self.cases), (_INST2, self.cases2)], use_cache=True)
        for a, b in zip(res, again):
            self.assertTrue(b.cache_hit)
            self.assertEqual(a.kill_mask, b.kill_mask)
            self.assertEqual(b.ladder, 5)

    def test_sorry_sketch_off_is_L0(self):
        r = self.gate("def candidate (P : Params) : Prune P := by sorry\n", sketch=False, use_cache=False)[0]
        self.assertEqual(r.ladder, 0)
        self.assertFalse(r.scanned_ok)
        self.assertIn("sorry", " ".join(r.forbidden))

    def test_cheat_is_L1(self):
        r = self.gate(CHEAT, use_cache=False)[0]
        self.assertEqual(r.ladder, 1)
        self.assertFalse(r.ok)
        self.assertFalse(r.compiled)
        self.assertIsNone(r.kill_mask)
        self.assertTrue(any("line 5" in e for e in r.errors), r.errors)
        self.assertLess(r.lean_partial, 0.2)

    def test_unknown_identifier_is_L1(self):
        # Lean 4.34 prints `error(lean.unknownIdentifier):` -- the parser must see it
        r = self.gate("def candidate (P : Params) : Prune P := never P\n", use_cache=False)[0]
        self.assertEqual(r.ladder, 1)
        self.assertFalse(r.compiled)
        self.assertTrue(any("Unknown identifier" in e for e in r.errors), r.errors)

    def test_parse_error_is_L0(self):
        r = self.gate("def candidate (P : Params) : Prune P := Prune.or (deficit P\n", use_cache=False)[0]
        self.assertEqual(r.ladder, 0)
        self.assertTrue(r.parse_error)
        self.assertEqual(r.lean_partial, 0.0)

    def test_instance_specific(self):
        res = self.gate(SPECIFIC, [(_INST, self.cases), (_INST2, self.cases2)], use_cache=False)
        self.assertEqual(res[0].ladder, 5)
        self.assertTrue(res[0].ok)
        self.assertEqual(len(res[0].kill_mask), len(self.cases))
        self.assertEqual(res[1].ladder, 1)   # `candidate : Prune target` does not fit target1
        self.assertFalse(res[1].ok)
        self.assertTrue(any("instance 1" in e for e in res[1].errors), res[1].errors)

    def test_hole_fillable_reaches_L5(self):
        r = self.gate(HOLE_FILLABLE, use_cache=False)[0]
        self.assertEqual(r.n_holes, 1)
        self.assertEqual(r.n_holes_filled, 1)
        self.assertEqual(r.holes[0]["filled_by"], "omega")
        self.assertEqual(r.ladder, 5, r.errors)
        self.assertTrue(r.ok)
        self.assertNotIn("sorry", r.filled_source)
        self.assertIn("set_option maxHeartbeats 50000 in omega", r.filled_source)
        self.assertEqual(len(r.kill_mask), len(self.cases))
        self.assertEqual(lg.static_scan(r.filled_source, sketch=False), [])

    def test_hole_stuck_is_L2_with_goal(self):
        r = self.gate(HOLE_STUCK, use_cache=False)[0]
        self.assertEqual(r.ladder, 2, r.errors)
        self.assertFalse(r.ok)
        self.assertIsNone(r.kill_mask)
        self.assertEqual(r.n_holes, 1)
        self.assertEqual(r.n_holes_filled, 0)
        self.assertIn("sorryAx", r.axioms)
        self.assertFalse(r.axioms_ok)
        self.assertIn("⊢ sumFin P.m (rowSum A) = 17", r.holes[0]["goal"])
        self.assertIn("hv : Valid P A", r.holes[0]["goal"])
        self.assertNotIn("ZHOLE", r.holes[0]["goal"])
        self.assertAlmostEqual(r.lean_partial, 0.25 + 0.15 + 0.10)
        self.assertIsNone(r.filled_source)

    def test_hole_partial_fill_is_L3(self):
        src = HOLE_STUCK.replace("    have h4 :",
                                 "    have h3 : sumFin P.m (rowSum A) < sumFin P.m (rowSum A) + 1 := by sorry\n    have h4 :")
        r = self.gate(src, use_cache=False)[0]
        self.assertEqual(r.n_holes, 2)
        self.assertEqual(r.n_holes_filled, 1)
        self.assertEqual(r.ladder, 3, r.errors)
        self.assertIsNotNone(r.filled_source)
        self.assertEqual(r.filled_source.count("sorry"), 1)
        self.assertAlmostEqual(r.lean_partial, 0.50 + 0.30 * 0.5 + 0.10)

    def test_schema_terms(self):
        terms = [["ZarPrune.deficit ZarPrune.Cand.target", "ZarPrune.argD ZarPrune.Cand.target"],
                 ["ZarPrune.argD ZarPrune.Cand.target1"]]
        res = self.gate(NEVER, [(_INST, self.cases), (_INST2, self.cases2)], schema_terms=terms, use_cache=False)
        for r, cs in zip(res, (self.cases, self.cases2)):
            self.assertEqual(r.ladder, 5, r.errors)
            self.assertEqual(len(r.schema_mask), len(cs))
            self.assertEqual(r.kill_mask, r.schema_mask)  # candidate kills nothing: gateInst == schema
            self.assertGreater(sum(r.schema_mask), 0)
        plain = self.gate(NEVER, use_cache=False)[0]
        self.assertIsNone(plain.schema_mask)
        self.assertEqual(sum(plain.kill_mask), 0)

    def test_bad_schema_term_is_reported(self):
        r = self.gate(NEVER, schema_terms=[["ZarPrune.noSuchPrune ZarPrune.Cand.target"]], use_cache=False)[0]
        self.assertNotEqual(r.ladder, 5)
        self.assertTrue(any("schema0" in e for e in r.errors), r.errors)

    def test_run_gate_single(self):
        r = lg.run_gate(_INST, GOOD, self.cases, tag="test_single", use_cache=False)
        self.assertTrue(r.ok)
        self.assertEqual(r.ladder, 5)


if __name__ == "__main__":
    unittest.main()

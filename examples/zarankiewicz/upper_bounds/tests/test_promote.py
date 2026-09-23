"""Promotion (design §4.4), verify-certs (T-11) and audit-kills (T-12) — key H-promote.

Every promotion here writes into a TEMPORARY library (evolved_dir / evolved_lean / ledger_path
overrides), never into lean/ZarPrune/Evolved or cache/ledger.  The Lean gate runs for real
(~40 s per promotion on the full suite; the gate cache is bypassed by design), so the three
promotion tests take ~1.5 min.  ZAR_UB_PROMOTE_SKIP_SLOW=1 skips them.

    cd examples/zarankiewicz/upper_bounds && ZAR_UB_NO_LLM=1 python -m unittest tests.test_promote
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, UB)

from zar_ub import Instance  # noqa: E402
from zar_ub import promote as P  # noqa: E402
from zar_ub import certify as C  # noqa: E402
from zar_ub.casetable import load_table  # noqa: E402

SLOW = os.environ.get("ZAR_UB_PROMOTE_SKIP_SLOW") == "1"
COUNTING = os.path.join(UB, "experiments", "E8_counting", "counting_program.py")
UNSOUND = os.path.join(UB, "experiments", "E6_evaluator", "unsound_program.py")
SORRY = os.path.join(UB, "tests", "candidates", "sorry_in_sound.py")
CERTS_99 = os.path.join(UB, "cache", "certs", "m9_n9_s3_t3_w50")


class _TempLibrary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="zar_ub_test_promote_")
        self.kw = dict(evolved_dir=os.path.join(self.tmp, "Evolved"), evolved_lean=os.path.join(self.tmp, "Evolved.lean"),
                       ledger_path=os.path.join(self.tmp, "ledger.jsonl"), verbose=False)
        self.sentinel_before = os.path.exists(P.SENTINEL)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.assertEqual(os.path.exists(P.SENTINEL), self.sentinel_before, "a test wrote/removed cache/PIPELINE_BUG")


class TextGeneration(_TempLibrary):
    """No Lean: the module / library text and the ledger plumbing."""

    def test_sha_is_normalised(self):
        a = "def candidate (P : Params) : Prune P := counting P\n"
        self.assertEqual(P.source_sha(a), P.source_sha("\n\n" + a.replace("\n", "   \r\n") + "\n"))
        self.assertRegex(P.source_sha(a), r"^[0-9a-f]{12}$")

    def test_module_and_evolved_text(self):
        src = "def candidate (P : Params) : Prune P := counting P\n"
        sha = P.source_sha(src)
        text = P.module_text(sha, src, "t", {"iteration": 3, "program_id": "x-/y"})
        self.assertTrue(text.startswith("import ZarPrune.Cond\n"))
        self.assertNotIn("import ZarPrune\n", text)  # never the root module (import cycle)
        self.assertIn(f"namespace ZarPrune.Evolved.E_{sha}\n", text)
        self.assertIn(f"\nend ZarPrune.Evolved.E_{sha}", text)
        self.assertEqual(P._extract_source(text, sha), src)
        self.assertNotIn("x-/y", text)  # doc-comment terminator sanitised
        self.assertIn("Prune.never P", P.evolved_text([]))
        ev = P.evolved_text(["aaaaaaaaaaaa", "bbbbbbbbbbbb"])
        self.assertIn("import ZarPrune.Evolved.E_aaaaaaaaaaaa", ev)
        self.assertIn("Prune.ofList P [Evolved.E_aaaaaaaaaaaa.candidate P, Evolved.E_bbbbbbbbbbbb.candidate P]", ev)
        self.assertIn("def evolved (P : Params) : Prune P", ev)

    def test_regenerate_from_empty_ledger(self):
        path, have = P.regenerate_evolved(self.kw["ledger_path"], self.kw["evolved_dir"], self.kw["evolved_lean"])
        self.assertEqual(have, [])
        with open(path) as fh:
            self.assertIn("Prune.never P", fh.read())
        self.assertEqual(P.novelty_switch(self.kw["ledger_path"]), (False, 0))

    def test_delivered_evolved_lean_matches_ledger(self):
        """The file in the tree is exactly what the ledger generates."""
        shas = P.ledger_shas()
        with open(P.EVOLVED_LEAN) as fh:
            self.assertEqual(fh.read(), P.evolved_text([s for s in shas if os.path.exists(os.path.join(P.EVOLVED_DIR, f"E_{s}.lean"))]))
        for s in shas:
            self.assertIsNotNone(P.entry_source(s), f"module for ledger entry {s} missing")


@unittest.skipIf(SLOW, "ZAR_UB_PROMOTE_SKIP_SLOW=1")
class Promotion(_TempLibrary):
    def test_counting_program_is_promoted(self):
        """Kills nothing beyond the proved library, still accepted; full pipeline incl. replay."""
        r = P.promote_program(COUNTING, "counting_program", {"iteration": 0}, **self.kw)
        self.assertTrue(r.ok, f"{r.stage}: {r.reason} {r.errors[:3]}")
        self.assertEqual(r.stage, "done")
        self.assertEqual(r.new_kills, 0)
        self.assertTrue(all(v == 5 for v in r.ladders.values()), r.ladders)
        self.assertTrue(set(r.axioms) <= P.ALLOWED_AXIOMS, r.axioms)
        self.assertTrue(r.replay.get("ok"), r.replay)
        self.assertIn(r.replay.get("tool"), ("leanchecker", "lake env lean"))
        self.assertTrue(os.path.exists(r.module_path))
        with open(r.module_path) as fh:
            mod = fh.read()
        self.assertIn(f"namespace ZarPrune.Evolved.E_{r.sha}", mod)
        self.assertIn("def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)", mod)
        with open(r.evolved_path) as fh:
            self.assertIn(f"Prune.ofList P [Evolved.E_{r.sha}.candidate P]", fh.read())
        ledger = P.load_ledger(self.kw["ledger_path"])
        self.assertEqual(len(ledger), 1)
        e = ledger[0]
        for key in ("sha", "name", "lean_name", "per_instance_kill_signature", "kill_signature", "masks", "iteration", "axioms", "timestamp"):
            self.assertIn(key, e)
        self.assertEqual(e["lean_name"], f"ZarPrune.Evolved.E_{r.sha}.candidate")
        self.assertTrue(all(v == [] for v in e["masks"].values()), "the library kills nothing beyond itself")
        self.assertTrue(any(e["masks_all"].values()), "the library kills something")
        # idempotent
        r2 = P.promote_program(COUNTING, "counting_program", {"iteration": 0}, **self.kw)
        self.assertTrue(r2.ok and r2.already)
        self.assertEqual(len(P.load_ledger(self.kw["ledger_path"])), 1)
        # the promoted library's masks reproduce the stored baseline masks (per-entry route)
        inst = Instance(9, 9, 3, 3, 50)
        tab = load_table(inst, use_table=False)
        masks, info = P.library_masks([(inst, tab)], ledger_path=self.kw["ledger_path"], evolved_dir=self.kw["evolved_dir"], verbose=False)
        self.assertEqual(info["entries"], [r.sha])
        self.assertEqual(masks[0], [bool(x) for x in tab.baseline_lean_mask])

    def test_unsound_program_refused_by_battery(self):
        r = P.promote_program(UNSOUND, "unsound", {}, **self.kw)
        self.assertFalse(r.ok)
        self.assertEqual(r.stage, "battery")
        self.assertIn("witnessed", r.reason)
        self.assertFalse(os.path.exists(self.kw["ledger_path"]))
        self.assertFalse(os.path.exists(self.kw["evolved_dir"]))
        q = P.load_ledger(os.path.join(self.tmp, "quarantine.jsonl"))
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["stage"], "battery")

    def test_sorry_program_refused_by_ladder(self):
        r = P.promote_program(SORRY, "sorry_in_sound", {}, **self.kw)
        self.assertFalse(r.ok)
        self.assertEqual(r.stage, "ladder")
        self.assertTrue(r.ladders and all(v < 5 for v in r.ladders.values()), r.ladders)
        self.assertFalse(os.path.exists(self.kw["ledger_path"]))
        q = P.load_ledger(os.path.join(self.tmp, "quarantine.jsonl"))
        self.assertEqual(q[0]["stage"], "ladder")


@unittest.skipUnless(os.path.exists(os.path.join(CERTS_99, "manifest.json")), "cache/certs/m9_n9_s3_t3_w50 not present")
class VerifyCerts(unittest.TestCase):
    def test_reverify_and_corruption(self):
        rep = C.verify_certs(CERTS_99, fresh=False, corruption_test=True, verbose=False)
        self.assertTrue(rep["ok"], rep["failed"][:3])
        self.assertGreaterEqual(rep["n_certs"], 17)  # 36 before the daemon (all cases), 17 after (survivors only)
        self.assertEqual(rep["verified"], rep["n_certs"])
        self.assertTrue(rep["corruption_test"]["ok"], rep["corruption_test"])

    def test_one_byte_corruption_fails_directly(self):
        with open(os.path.join(CERTS_99, "manifest.json")) as fh:
            c = [c for c in json.load(fh)["certs"] if c["status"] == "certified"][0]
        key = "r" + "-".join(map(str, c["rows"])) + "_c" + "-".join(map(str, c["cols"]))
        cnf, lrat = os.path.join(CERTS_99, key + ".cnf"), os.path.join(CERTS_99, key + ".lrat")
        self.assertTrue(C.run_lrat_check(cnf, lrat)[0])
        tmp = tempfile.mkdtemp(prefix="zar_ub_corrupt_")
        try:
            bad = os.path.join(tmp, "bad.lrat")
            rejected = False
            for attempt in range(8):
                C.corrupt_one_byte(lrat, bad, attempt)
                with open(lrat, "rb") as f1, open(bad, "rb") as f2:
                    a, b = f1.read(), f2.read()
                self.assertEqual(len(a), len(b))
                self.assertEqual(sum(1 for x, y in zip(a, b) if x != y), 1)
                if not C.run_lrat_check(cnf, bad)[0]:
                    rejected = True
                    break
            self.assertTrue(rejected, "a one-byte corruption of the LRAT was accepted")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(os.path.exists(C.LRAT_CHECK_SRC), "lrat-check.c missing")
    def test_fresh_lratcheck_build(self):
        tmp = tempfile.mkdtemp(prefix="zar_ub_lc_")
        try:
            binary = C.build_fresh_lratcheck(tmp)
            self.assertTrue(os.access(binary, os.X_OK))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class AuditKills(unittest.TestCase):
    def test_sample_is_seeded_and_unsat(self):
        inst = Instance(9, 9, 3, 3, 50)
        tab = load_table(inst, use_table=False)
        self.assertIsNotNone(tab)
        mask = tab.baseline_lean_mask
        self.assertIsNotNone(mask)
        sentinel_before = os.path.exists(C.SENTINEL)
        rep = C.audit_kills(inst, tab, mask, frac=0.05, verbose=False)
        rep2 = C.audit_kills(inst, tab, mask, frac=0.05, verbose=False)
        self.assertEqual(rep["sampled"], rep2["sampled"])
        self.assertEqual(len(rep["sampled"]), 1)  # ceil(0.05 * 19)
        self.assertTrue(rep["ok"], rep)
        self.assertIsNone(rep["bug"])
        self.assertTrue(all(e["status"] == "unsat" for e in rep["results"]))
        self.assertEqual(os.path.exists(C.SENTINEL), sentinel_before)

    def test_frac_one_covers_every_killed_case(self):
        inst = Instance(9, 9, 3, 3, 50)
        tab = load_table(inst, use_table=False)
        rep = C.audit_kills(inst, tab, tab.baseline_lean_mask, frac=1.0, verbose=False)
        self.assertEqual(len(rep["sampled"]), sum(tab.baseline_lean_mask))
        self.assertTrue(rep["ok"], rep)


if __name__ == "__main__":
    unittest.main()

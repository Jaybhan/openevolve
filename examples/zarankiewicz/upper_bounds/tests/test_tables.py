"""Consistency of every cached case table (design §5.1/§6; C-tables TODO 7).

* baseline_lean_mask, when stored, has exactly one entry per record and never kills a SAT-witnessed case
  (a proved kill of a realizable case would be a PIPELINE_BUG);
* every record has a difficulty d >= 1 and a censored flag consistent with its probe status;
* the CRN sample (§6.3) is a deterministic function of table_hash and lies inside the scored survivors;
* the suite loads with exactly the four kinds and its tables all carry a hash.

    python -m unittest tests.test_tables      # ~1 s (no Lean, no SAT)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, UB)

from zar_ub.casetable import CACHE_DIR, make_sample, table_from_json  # noqa: E402


def _tables():
    for path in sorted(glob.glob(os.path.join(CACHE_DIR, "case_table_*.json"))):
        with open(path) as fh:
            d = json.load(fh)
        yield os.path.basename(path), table_from_json(d, path)


class CachedTables(unittest.TestCase):
    def test_baseline_mask_shape_and_witnesses(self):
        n_tables, n_masks = 0, 0
        for name, tab in _tables():
            n_tables += 1
            mask = getattr(tab, "baseline_lean_mask", None)
            if mask is None:
                continue
            n_masks += 1
            self.assertEqual(len(mask), len(tab.records), f"{name}: mask length != records")
            for rec, killed in zip(tab.records, mask):
                self.assertEqual(bool(killed), bool(rec.baseline_lean_kill), f"{name}: mask/record flag disagree")
                if rec.status == "sat":
                    self.assertFalse(killed, f"{name}: proved library kills a SAT-witnessed case {rec.rows}/{rec.cols}")
        self.assertGreater(n_tables, 0, "no cached tables under cache/")
        self.assertGreater(n_masks, 0, "no table carries a baseline_lean_mask (run: python -m zar_ub baseline --all)")

    def test_labels(self):
        for name, tab in _tables():
            for rec in tab.records:
                self.assertGreaterEqual(float(rec.d), 1.0, f"{name}: d < 1")
                if rec.status == "unsat":
                    self.assertFalse(rec.censored, f"{name}: refuted case marked censored")
                if rec.status == "unknown":
                    self.assertTrue(rec.censored, f"{name}: open case not marked censored")
            if tab.records:
                self.assertTrue(tab.table_hash, f"{name}: empty table_hash")

    def test_crn_sample_is_deterministic(self):
        for name, tab in _tables():
            s1, s2 = make_sample(tab), make_sample(tab)
            self.assertEqual(s1, s2, f"{name}: CRN sample not deterministic")
            if s1 is not None:
                S = set(tab.scored_indices())
                self.assertTrue(set(s1) <= S, f"{name}: sample outside the scored survivors")
                self.assertEqual(s1, sorted(s1))
            stored = getattr(tab, "sample", None)
            if stored is not None:
                self.assertEqual(list(stored), s1, f"{name}: stored sample differs from the recomputed one")

    def test_suite_loads(self):
        from suite import load_suite  # noqa: E402

        s = load_suite(verbose=False)
        self.assertEqual(set(s.keys()), {"train", "battery", "target", "gen"})
        self.assertEqual(len(s["train"]), 7)
        self.assertGreaterEqual(len(s["gen"]), 1)
        for kind, tables in s.items():
            for inst, tab in tables:
                self.assertEqual(tab.kind, kind, f"{inst.tag}: kind {tab.kind!r} != {kind!r}")
                if tab.records:
                    self.assertTrue(tab.table_hash, f"{kind} {inst.tag}: no table_hash")


if __name__ == "__main__":
    unittest.main()

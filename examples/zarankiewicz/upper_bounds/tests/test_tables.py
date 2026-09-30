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

    def test_model_labels_respect_the_clip(self):
        """E27: a censored model label is never below the conflicts reached and never above 2M unless the
        lower bound is; a 'lower_bound' label equals the conflicts reached."""
        from zar_ub.difficulty import CENSOR_CLIP, MODEL_CAP  # noqa: E402

        ceil = CENSOR_CLIP * MODEL_CAP
        for name, tab in _tables():
            for rec in tab.records:
                p = rec.probe or {}
                if not rec.censored or p.get("d_method") not in ("model", "lower_bound"):
                    continue
                lb = float(p.get("conflicts") or 0)
                self.assertGreaterEqual(rec.d + 1e-9, lb, f"{name}: model label below the lower bound")
                self.assertLessEqual(rec.d, max(ceil, lb) + 1e-9, f"{name}: model label above the ceiling")
                if p["d_method"] == "lower_bound":
                    self.assertEqual(rec.d, lb, f"{name}: lower_bound label != conflicts reached")

    def test_scored_censored_cases_carry_model_labels(self):
        """E27: every censored library survivor of a scored (3,3) table is labelled by the hardness model
        (or by its lower bound at the 2M ceiling), not by the legacy fhat extrapolation."""
        from suite import load_suite  # noqa: E402
        from zar_ub.difficulty import load_hardness_model  # noqa: E402

        if load_hardness_model() is None:
            self.skipTest("no hardness model (or ZAR_UB_DIFFICULTY=legacy)")
        s = load_suite(verbose=False, state={"graduated": [], "entered": [], "history": []})
        bad = []
        for kind in ("train", "target", "gen"):
            for inst, tab in s[kind]:
                if (inst.s, inst.t) != (3, 3):
                    continue
                for i in tab.scored_indices():
                    r = tab.records[i]
                    if r.censored and (r.probe or {}).get("d_method") not in ("model", "lower_bound"):
                        bad.append(f"{inst.tag}#{i}")
        self.assertFalse(bad, f"{len(bad)} censored survivors without a model label, e.g. {bad[:5]} "
                              f"(run experiments/E27_integration/relabel_tables.py run/apply)")

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

        import suite as suite_mod  # noqa: E402

        s = load_suite(verbose=False, state={"graduated": [], "entered": [], "history": []})
        self.assertEqual(set(s.keys()), {"train", "battery", "target", "gen"})
        # E27 suite v3: the 7 square cells + the 7 exactly-known wide cells
        self.assertEqual(len(s["train"]), len(suite_mod.TRAIN_CELLS) + len(suite_mod.WIDE_TRAIN_CELLS))
        self.assertEqual(len(set(i.tag for i, _ in s["train"])), len(s["train"]), "duplicate TRAIN cell")
        self.assertGreaterEqual(len(s["gen"]), 1)
        for kind, tables in s.items():
            for inst, tab in tables:
                self.assertEqual(tab.kind, kind, f"{inst.tag}: kind {tab.kind!r} != {kind!r}")
                if tab.records:
                    self.assertTrue(tab.table_hash, f"{kind} {inst.tag}: no table_hash")


if __name__ == "__main__":
    unittest.main()

"""E27 unit tests: the E24 hardness model (zar_ub/hardness_model.py), the censored-label rule in
zar_ub/difficulty.py, and the reward-v3 terms of zar_ub/reward.py (synthetic tables, no Lean).

    python -m unittest tests.test_difficulty_reward          # from upper_bounds/ (~10 s; 4 solver runs)
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, UB)

from zar_ub import difficulty, hardness_model as hm, reward  # noqa: E402
from zar_ub.casetable import CaseRecord, CaseTable  # noqa: E402
from zar_ub.known import Instance  # noqa: E402

# a (13,13;3,3) w=93 case with true d = 28,664 (E24 ground truth) and its model prediction (E24 final check)
CASE = (Instance(13, 13, 3, 3, 93), [8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 4], [9, 9, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 4])
CASE_MEDIAN = 27988.705308689743  # experiments/E24_difficulty/features_final_check.jsonl (mean=False)


# ======================================================================================
# hardness model
# ======================================================================================
class HardnessModelTest(unittest.TestCase):
    def setUp(self):
        self.m = hm.load()
        if self.m is None:
            self.skipTest("experiments/E24_difficulty/hardness_model.json missing")

    def test_model_file(self):
        self.assertEqual(len(self.m.features), 6)
        self.assertEqual(self.m.cap, 20_000)
        self.assertEqual(self.m.floor, 20_000.0)
        self.assertGreater(self.m.smear, 1.0)
        self.assertEqual(set(self.m.coef), set(self.m.features))
        self.assertTrue(self.m.needs()["lookahead_full"])  # knfl_* needs the Knuth probes
        self.assertFalse(self.m.needs()["sampling"])
        self.assertFalse(self.m.needs()["binary"])

    def test_predict_at_the_mean_is_the_intercept(self):
        # the transformed feature equal to mu gives z = 0 -> log d = intercept
        feats = {}
        for f in self.m.features:
            mu, tr = self.m.mu[f], self.m.transforms[f]
            feats[f] = math.expm1(mu) if tr == "log1p" else mu
        med = self.m.predict_features(feats, mean=False, floor=0.0)
        self.assertAlmostEqual(math.log(med), self.m.intercept, places=9)
        self.assertAlmostEqual(self.m.predict_features(feats, mean=True, floor=0.0), med * self.m.smear, places=6)
        # floor and ceiling
        self.assertEqual(self.m.predict_features(feats, floor=1e9), 1e9)
        self.assertEqual(self.m.predict_features(feats, floor=0.0, ceiling=10.0), 10.0)

    def test_missing_feature_is_none(self):
        self.assertIsNone(self.m.predict_features({}))
        self.assertIsNone(self.m.log_d({f: float("nan") for f in self.m.features}))

    def test_fit_recovers_coefficients_and_roundtrips(self):
        import random
        rng = random.Random(24)
        feats, y = [], []
        for _ in range(400):
            a, b = rng.uniform(0, 3), rng.uniform(1, 50)
            feats.append({"x:a": a, "x:b": b})
            y.append(10.0 + 0.7 * a - 0.2 * math.log1p(b) + rng.gauss(0, 0.01))
        m = hm.fit(feats, y, ["x:a", "x:b"], {"x:b": "log1p"})
        self.assertAlmostEqual(m.coef["x:a"] / m.sd["x:a"], 0.7, places=2)
        self.assertAlmostEqual(m.coef["x:b"] / m.sd["x:b"], -0.2, places=2)
        with tempfile.TemporaryDirectory() as d:
            p = m.save(os.path.join(d, "m.json"))
            m2 = hm.load(p)
        self.assertEqual(m2.coef, m.coef)
        self.assertEqual(m2.predict_features({"x:a": 1.0, "x:b": 3.0}), m.predict_features({"x:a": 1.0, "x:b": 3.0}))

    def test_predict_end_to_end_is_deterministic_and_matches_probe_path(self):
        inst, rows, cols = CASE
        o = hm.predict(inst, rows, cols, model=self.m, mean=False)
        self.assertFalse(o["exact"])
        self.assertAlmostEqual(o["d_hat"], CASE_MEDIAN, delta=1e-6 * CASE_MEDIAN)
        self.assertGreater(o["cost_conflicts"], 20_000)  # the fresh 20k probe (+ the 2k one)
        f = o["features"]
        d2 = hm.predict_from_probe(inst, rows, cols, int(f["pr:ps20k_conflicts"]), int(f["pr:ps20k_decisions"]),
                                   int(f["pr:ps20k_restarts"]), int(f["pr:ps20k_propagations"]), model=self.m,
                                   mean=False)
        self.assertAlmostEqual(d2, o["d_hat"], places=6)


# ======================================================================================
# the censored-label rule
# ======================================================================================
class CensoredLabelTest(unittest.TestCase):
    def test_model_clip(self):
        C = difficulty.CENSOR_CLIP * difficulty.MODEL_CAP
        self.assertEqual(C, 2_000_000)
        self.assertEqual(difficulty.model_clip(20_000, 5_000.0), 20_000.0)        # never below the lower bound
        self.assertEqual(difficulty.model_clip(20_000, 50_000.0), 50_000.0)
        self.assertEqual(difficulty.model_clip(20_000, 9e9), float(C))            # ceiling 2M
        self.assertEqual(difficulty.model_clip(2_000_004, 9e9), 2_000_004.0)       # lower bound above the ceiling
        self.assertEqual(difficulty.censored_d(20_000, 9e9), 400_000.0)            # legacy clip unchanged

    def test_legacy_switch(self):
        inst, rows, cols = CASE
        probe = {"status": "unknown", "conflicts": 20_000, "budget_cap": 20_000, "c2000": 2001,
                 "log2_volume": 150.0}
        with mock.patch.dict(os.environ, {"ZAR_UB_DIFFICULTY": "legacy"}):
            self.assertEqual(difficulty.difficulty_method(), "legacy")
            self.assertIsNone(difficulty.load_hardness_model())
            o = difficulty.relabel_probe(inst, rows, cols, probe)
        self.assertEqual(o["d_method"], "fhat")
        self.assertEqual(o["d"], difficulty.censored_d(20_000, o["fhat"]))
        self.assertEqual(o["cost_conflicts"], 0)

    def test_lower_bound_at_ceiling_costs_nothing(self):
        inst, rows, cols = CASE
        probe = {"status": "unknown", "conflicts": 2_000_003, "budget_cap": 2_000_000, "c2000": 2001,
                 "log2_volume": 150.0}
        o = difficulty.relabel_probe(inst, rows, cols, probe)
        if difficulty.load_hardness_model() is None:
            self.skipTest("no hardness model")
        self.assertEqual((o["d"], o["d_method"], o["cost_conflicts"]), (2_000_003.0, "lower_bound", 0))

    def test_label_case_uses_the_model_from_its_own_20k_run(self):
        if difficulty.load_hardness_model() is None:
            self.skipTest("no hardness model")
        inst, rows, cols = CASE
        lab = difficulty.label_case(inst, rows, cols, mode="censored")
        self.assertEqual(lab.status, "unknown")
        self.assertTrue(lab.censored)
        self.assertEqual(lab.d_method, "model")
        self.assertEqual(set(lab.ps20k), {"conflicts", "decisions", "restarts", "propagations"})
        self.assertAlmostEqual(lab.d_model, CASE_MEDIAN * hm.load().smear, delta=1e-6 * CASE_MEDIAN)
        self.assertEqual(lab.d, difficulty.model_clip(lab.conflicts, lab.d_model))
        p = lab.as_probe()
        self.assertEqual(p["d_method"], "model")
        # the relabel path on the stored probe reproduces the label without any solver conflict
        o = difficulty.relabel_probe(inst, rows, cols, p)
        self.assertEqual(o["cost_conflicts"], 0)
        self.assertAlmostEqual(o["d"], lab.d, places=6)


# ======================================================================================
# reward v3
# ======================================================================================
def _table(inst: Instance, labels, lib_killed=(), exact=None, reached=None):
    """A synthetic scored table: labels = d per record; lib_killed = indices the library kills;
    exact[i] False marks a censored record whose conflicts reached = reached[i] (default 20k)."""
    recs = []
    for i, d in enumerate(labels):
        cens = exact is not None and not exact[i]
        conf = (reached[i] if reached else 20_000) if cens else int(d)
        recs.append(CaseRecord([0], [0], "", {"status": "unknown" if cens else "unsat", "conflicts": conf,
                                              "budget_cap": 20_000 if cens else 2_000_000},
                               d=float(d), censored=cens))
    mask = [i in set(lib_killed) for i in range(len(labels))]
    t = CaseTable(inst={"m": inst.m, "n": inst.n, "s": inst.s, "t": inst.t, "w": inst.w}, n_row_partitions=1,
                  n_col_partitions=1, records=recs, baseline_lean_mask=mask)
    return t


def _score(tables, masks, version="v3", stage=3, ladder=5, **kw):
    with mock.patch.dict(os.environ, {"ZAR_UB_REWARD": version}):
        return reward.score(tables, py_masks=masks, lean_masks=masks, stage=stage, ladder=ladder, **kw)[0]


class RewardV3Test(unittest.TestCase):
    def setUp(self):
        self.sq_easy = Instance(9, 9, 3, 3, 50)      # square, every case easy (<= 20k)
        self.sq_hard = Instance(12, 12, 3, 3, 81)    # square, hard cases
        self.wide = Instance(9, 18, 3, 3, 86)        # vwide (n/m = 2)
        self.tgt = Instance(12, 18, 3, 3, 109)       # target, wide (1.5)
        self.T = {
            "train": [(self.sq_easy, _table(self.sq_easy, [500, 1500, 19_000, 900], lib_killed=[3])),
                      (self.sq_hard, _table(self.sq_hard, [120_000, 520_000, 1_000, 60_000])),
                      (self.wide, _table(self.wide, [1_020_000, 520_000, 2_020_000, 30_000]))],
            "target": [(self.tgt, _table(self.tgt, [400_000, 400_000, 1_020_000, 1_500],
                                         exact=[False, False, True, True]))],
            "gen": [], "battery": [],
        }
        self.none = {i.tag: [False] * len(t.records) for k in ("train", "target") for i, t in self.T[k]}

    def masks(self, **kills):
        m = {k: list(v) for k, v in self.none.items()}
        for tag, idx in kills.items():
            for i in idx:
                m[tag][i] = True
        return m

    def test_constants_and_keys(self):
        self.assertAlmostEqual(reward.W_TRAIN + reward.W_TARGET + reward.W_GEN + reward.W_TAIL + reward.W_DEPTH
                               + reward.W_CLOSE, 1.0)
        m = _score(self.T, self.none)
        for k in reward.METRIC_KEYS:
            self.assertIn(k, m)
        self.assertEqual(m["combined_score"], 0.20)

    def test_family_of(self):
        self.assertEqual(reward.family_of(9, 9, 3, 3), "square")
        self.assertEqual(reward.family_of(12, 18, 3, 3), "wide")
        self.assertEqual(reward.family_of(9, 18, 3, 3), "vwide")
        self.assertEqual(reward.family_of(9, 9, 4, 4), "gen")

    def test_easy_cells_carry_no_weight(self):
        # clearing a table whose every case is below 20k conflicts earns nothing (R3 attacks A2/B)
        m = _score(self.T, self.masks(**{self.sq_easy.tag: [0, 1, 2]}))
        self.assertEqual(m["combined_score"], 0.20)
        # ... and killing an easy case on a hard table earns nothing either
        m = _score(self.T, self.masks(**{self.sq_hard.tag: [2]}))
        self.assertEqual(m["combined_score"], 0.20)

    def test_gain_on_excess_work_and_family_balance(self):
        m = _score(self.T, self.masks(**{self.sq_hard.tag: [1]}))  # 500k of the 100k+500k+40k excess
        g = 500_000 / (100_000 + 500_000 + 40_000)
        # square family: the easy cell has no weight, so G_square = gain on (12,12); the vwide family has 0
        self.assertAlmostEqual(m["gain_square"], g, places=9)
        self.assertAlmostEqual(m["proven_gain"], g / 2, places=9)  # mean over the two TRAIN families
        self.assertAlmostEqual(m["best_cell_gain"], g, places=9)

    def test_importance_uses_lower_bounds_only(self):
        # the target's censored cases are labelled 400k but only 20k conflicts were REACHED
        cells = self.T["target"][0][1]
        LB = sum(reward.rec_lower_bound(r) for r in cells.records)
        self.assertEqual(LB, 20_000 + 20_000 + 1_020_000 + 1_500)
        # closing it: LB ~1.06e6 >= 1e6 -> Close = a_I from LB, not from the 1.8M label sum
        m = _score(self.T, self.masks(**{self.tgt.tag: [0, 1, 2, 3]}))
        self.assertAlmostEqual(m["closure_bonus"], reward.importance(LB), places=12)
        self.assertEqual(m["closed_cells"], 1.0)

    def test_small_closure_earns_no_bonus(self):
        # (12,12) lower-bound work 681k < 1e6: closing it pays gains/tail but no Depth/Close
        m = _score(self.T, self.masks(**{self.sq_hard.tag: [0, 1, 2, 3]}))
        self.assertEqual(m["closure_bonus"], 0.0)
        self.assertEqual(m["depth_term"], 0.0)
        self.assertEqual(m["closed_cells"], 1.0)
        self.assertGreater(m["combined_score"], 0.20)

    def test_closure_bonus_formula(self):
        # E29: closing a TRAIN (practice) cell pays Depth (= a_I) but NOT the closure bonus,
        # which is reserved for TARGET cells (the thesis objective)
        m = _score(self.T, self.masks(**{self.wide.tag: [0, 1, 2, 3]}))
        a = reward.importance(1_020_000 + 520_000 + 2_020_000 + 30_000)
        self.assertEqual(m["closure_bonus"], 0.0)
        self.assertEqual(m["closed_cells"], 1.0)
        self.assertAlmostEqual(m["depth_term"], a, places=12)
        exp = 0.20 + 0.80 * (0.28 * m["proven_gain"] + 0.21 * m["target_gain"] + 0.07 * m["proven_gain"]
                             + 0.14 * m["tail_gain"] + 0.15 * a + 0.15 * 0.0)
        self.assertAlmostEqual(m["combined_score"], exp, places=12)
        self.assertIn("score_breakdown", reward.score(self.T, py_masks=self.masks(**{self.wide.tag: [0]}),
                                                      lean_masks=self.masks(**{self.wide.tag: [0]}),
                                                      stage=3, ladder=5)[1])

    def test_monotone_in_kills(self):
        import random
        rng = random.Random(27)
        tags = list(self.none)
        for _ in range(40):
            m1 = {t: [rng.random() < 0.3 for _ in v] for t, v in self.none.items()}
            m2 = {t: list(v) for t, v in m1.items()}
            t = rng.choice(tags)
            m2[t][rng.randrange(len(m2[t]))] = True
            self.assertGreaterEqual(_score(self.T, m2)["combined_score"] + 1e-12, _score(self.T, m1)["combined_score"])

    def test_soundness_ordering_is_unchanged(self):
        kill = self.masks(**{self.wide.tag: [0, 1, 2, 3]})
        self.assertLessEqual(_score(self.T, kill, ladder=3, lean_partial=1.0)["combined_score"], 0.19)
        self.assertEqual(_score(self.T, kill, ladder=4)["combined_score"], 0.0)
        self.assertEqual(_score(self.T, kill, python_error="boom")["combined_score"], 0.0)
        self.assertEqual(_score(self.T, kill, stage=1, ladder=None)["combined_score"] <= 0.19, True)

    def test_stage2_has_no_target_term(self):
        kill = self.masks(**{self.tgt.tag: [0, 1, 2, 3]})
        self.assertEqual(_score(self.T, kill, stage=2)["combined_score"], 0.20)

    def test_v2_switch_reproduces_the_old_formula(self):
        kill = self.masks(**{self.sq_easy.tag: [0]})
        m = _score(self.T, kill, version="v2")
        g_easy = 500 / (500 + 1500 + 19_000)
        G = g_easy / 3  # plain mean over the 3 TRAIN cells, full d
        tail = 0.0  # the killed case is not in any top decile
        exp = 0.20 + 0.80 * (0.40 * G + 0.30 * 0.0 + 0.10 * G + 0.20 * tail)
        self.assertAlmostEqual(m["combined_score"], exp, places=12)
        self.assertAlmostEqual(m["proven_gain_plain"], G, places=12)


if __name__ == "__main__":
    unittest.main()

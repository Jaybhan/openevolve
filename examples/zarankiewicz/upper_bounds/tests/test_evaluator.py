"""T-3 (design §10.1): run `evaluator.evaluate` on the golden adversarial bank
`tests/candidates/*.py` and assert the bands of `tests/golden.json`.

    python -m unittest tests.test_evaluator            # from upper_bounds/
    ZAR_UB_GOLDEN_SKIP_SLOW=1 python -m unittest tests.test_evaluator   # skip the gate-timeout case
    ZAR_UB_GOLDEN_ONLY=initial,unsound_row7 ...        # subset

Reward version is detected from the metrics keys (`lean_ladder` present => design v2 with the
verified floor 0.20 / unverified cap 0.19; otherwise the legacy v1 evaluator), and the matching
band of golden.json is asserted.  Every metric is read with a default so a missing key fails
loudly in the assertion, not with a KeyError.
"""
from __future__ import annotations

import json
import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, UB)

GOLDEN = os.path.join(HERE, "golden.json")
CAND_DIR = os.path.join(HERE, "candidates")
_RESULTS: dict = {}  # name -> (metrics, artifacts, seconds)


def _load_golden() -> dict:
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


def _evaluate(name: str):
    """Evaluate one candidate (memoised across tests)."""
    if name in _RESULTS:
        return _RESULTS[name]
    import evaluator  # noqa: WPS433  (import here so a broken evaluator fails inside the test)
    path = os.path.join(CAND_DIR, f"{name}.py")
    t0 = time.time()
    res = evaluator.evaluate(path)
    secs = time.time() - t0
    metrics = dict(getattr(res, "metrics", res) or {})
    artifacts = dict(getattr(res, "artifacts", {}) or {})
    _RESULTS[name] = (metrics, artifacts, secs)
    return _RESULTS[name]


def reward_version(metrics: dict) -> str:
    return "v2" if "lean_ladder" in metrics else "v1"


def _num(metrics: dict, key: str, default=None):
    v = metrics.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


class GoldenBankTest(unittest.TestCase):
    """One test method per candidate is attached below (test_golden_<name>)."""

    @classmethod
    def setUpClass(cls):
        cls.golden = _load_golden()
        cls.tol = float(cls.golden.get("tolerance", 1e-6))

    # ---- generic assertions -------------------------------------------------
    def _check_common(self, name, expect, metrics):
        if "sound_battery" in expect:
            self.assertAlmostEqual(_num(metrics, "sound_battery", -1.0), float(expect["sound_battery"]),
                                   delta=self.tol, msg=f"{name}: sound_battery {metrics.get('sound_battery')!r}")

    def _check_lean_ok(self, name, spec, metrics):
        v = _num(metrics, "lean_ok", -1.0)
        if isinstance(spec, str):
            if spec == "in_(0,1)":
                self.assertTrue(0.0 < v < 1.0, f"{name}: lean_ok={v} not in (0,1)")
            elif spec == "lt1":
                self.assertLess(v, 1.0, f"{name}: lean_ok={v}")
            else:
                self.fail(f"{name}: unknown lean_ok spec {spec!r}")
        else:
            self.assertAlmostEqual(v, float(spec), delta=self.tol, msg=f"{name}: lean_ok={v} expected {spec}")

    def _check_v2(self, name, spec, metrics, artifacts):
        score = _num(metrics, "combined_score", -1.0)
        if "ladder" in spec:
            allowed = spec["ladder"] if isinstance(spec["ladder"], list) else [spec["ladder"]]
            self.assertIn(int(round(_num(metrics, "lean_ladder", -1.0))), allowed,
                          f"{name}: lean_ladder={metrics.get('lean_ladder')} expected {allowed}; errors: "
                          f"{str(artifacts.get('lean_errors', ''))[:600]}")
        if "score" in spec:
            lo, hi = spec["score"]
            self.assertTrue(lo - self.tol <= score <= hi + self.tol,
                            f"{name}: combined_score={score} not in [{lo},{hi}]")
        if "lean_ok" in spec:
            self._check_lean_ok(name, spec["lean_ok"], metrics)
        for key in ("n_holes", "n_holes_filled"):
            if key in spec:
                self.assertEqual(int(round(_num(metrics, key, -1.0))), int(spec[key]),
                                 f"{name}: {key}={metrics.get(key)} expected {spec[key]}; holes: "
                                 f"{str(artifacts.get('lean_holes', ''))[:600]}")
        if spec.get("agreement_lt_1"):
            self.assertLess(_num(metrics, "agreement", 1.0), 1.0, f"{name}: agreement should be < 1")
        if spec.get("empirical_gain_gt_0"):
            self.assertGreater(_num(metrics, "empirical_gain", 0.0), 0.0, f"{name}: empirical_gain should be > 0")
        for art, sub in (spec.get("artifact_contains") or {}).items():
            self.assertIn(sub, str(artifacts.get(art, "")), f"{name}: artifact {art!r} lacks {sub!r}")

    def _check_v1(self, name, spec, metrics):
        score = _num(metrics, "combined_score", -1.0)
        if "lean_ok" in spec:
            self._check_lean_ok(name, spec["lean_ok"], metrics)
        s = spec.get("score")
        if s is None:
            return
        if isinstance(s, str):
            init_score = _num(_evaluate("initial")[0], "combined_score", -1.0)
            if s == "eq_initial":
                self.assertAlmostEqual(score, init_score, delta=self.tol, msg=f"{name}: {score} != initial {init_score}")
            elif s == "lt_initial":
                self.assertLess(score, init_score - self.tol, f"{name}: {score} not < initial {init_score}")
            elif s == "le_initial":
                self.assertLessEqual(score, init_score + self.tol, f"{name}: {score} not <= initial {init_score}")
            else:
                self.fail(f"{name}: unknown v1 score spec {s!r}")
        else:
            lo, hi = s
            self.assertTrue(lo - self.tol <= score <= hi + self.tol, f"{name}: combined_score={score} not in [{lo},{hi}]")

    def _run(self, name):
        entry = self.golden["candidates"][name]
        only = os.environ.get("ZAR_UB_GOLDEN_ONLY", "").strip()
        if only and name not in {x.strip() for x in only.split(",")}:
            self.skipTest(f"{name}: not in ZAR_UB_GOLDEN_ONLY")
        if entry.get("slow") and os.environ.get("ZAR_UB_GOLDEN_SKIP_SLOW") == "1":
            self.skipTest(f"{name}: slow (gate timeout); ZAR_UB_GOLDEN_SKIP_SLOW=1")
        for req in entry.get("requires", []):
            if not os.path.exists(os.path.join(UB, req)):
                self.skipTest(f"{name}: requires {req} (not built)")
        self.assertTrue(os.path.exists(os.path.join(CAND_DIR, f"{name}.py")),
                        f"missing candidate file tests/candidates/{name}.py (run tests/candidates/_gen.py)")
        metrics, artifacts, secs = _evaluate(name)
        self.assertTrue(metrics, f"{name}: evaluate returned no metrics")
        ver = reward_version(metrics)
        expect = entry["expect"]
        self._check_common(name, expect, metrics)
        if ver == "v2":
            self._check_v2(name, expect.get("v2", {}), metrics, artifacts)
        else:
            self._check_v1(name, expect.get("v1", {}), metrics)
        print(f"[golden {ver}] {name:18s} score={_num(metrics, 'combined_score', -1):.4f} "
              f"ladder={metrics.get('lean_ladder', 'n/a')} lean_ok={metrics.get('lean_ok')} "
              f"battery={metrics.get('sound_battery')} holes={metrics.get('n_holes', 'n/a')}/"
              f"{metrics.get('n_holes_filled', 'n/a')} {secs:.1f}s", file=sys.stderr)

    def test_bank_is_current(self):
        """The generated candidates must match the CURRENT initial_program.py structure."""
        sys.path.insert(0, CAND_DIR)
        import _gen  # noqa: WPS433
        src = open(os.path.join(UB, "initial_program.py"), encoding="utf-8").read()
        stale = []
        for name in _gen.CANDIDATES:
            path = os.path.join(CAND_DIR, f"{name}.py")
            cur = None
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    cur = fh.read()
            if cur is None or cur != _gen.render(name, src):
                stale.append(name)
        self.assertFalse(stale, f"stale candidates {stale}: run  python tests/candidates/_gen.py")

    def test_golden_covers_bank(self):
        names = set(_load_golden()["candidates"])
        files = {f[:-3] for f in os.listdir(CAND_DIR) if f.endswith(".py") and not f.startswith("_")}
        # candidates owned by other agents may exist without a golden entry (they add their own)
        self.assertTrue(names <= files, f"golden entries without a candidate file: {sorted(names - files)}")


def _make(name):
    def test(self):
        self._run(name)
    test.__name__ = f"test_golden_{name}"
    test.__doc__ = f"golden candidate {name}"
    return test


for _name in _load_golden()["candidates"]:
    setattr(GoldenBankTest, f"test_golden_{_name}", _make(_name))


if __name__ == "__main__":
    unittest.main()

"""Tests for zar_ub/lemmas.py (the SCHEMA_DATA registry, design §2.3) and its agreement
with lean/ZarPrune/Schemas.lean.

* layout / pair_idx round trips, validate accept/reject, render_terms (fully-qualified
  names, dict entries only on their instance), mirror_kill round trips;
* a matrix-level soundness check of the Python constraint system: on random K_{s,t}-free
  matrices the TRUE codegree vector satisfies F1–F4 with the constants `farkas_system`
  computes (so a Farkas certificate can never kill a realisable profile);
* the LP search returns exact integer certificates on the known (9,9,50) kills;
* (real Lean, ~10 s) the gate's `schema_mask` equals the mirror on the (9,9,50) table, with
  the Schemas body inlined into the candidate (until `ZarPrune.Schemas` is built) and the
  allowed axioms only.

Run:  cd examples/zarankiewicz/upper_bounds && python -m unittest tests.test_lemmas
"""

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zar_ub import Instance  # noqa: E402
from zar_ub import lemmas  # noqa: E402

P99 = Instance(9, 9, 3, 3, 50)
KILLED = ([7, 7, 6, 5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 5, 5, 5, 5])
ALIVE = ([6, 6, 6, 6, 6, 5, 5, 5, 5], [6, 6, 6, 6, 6, 5, 5, 5, 5])
# y3[(0,1)] = y4[(0,1)] = 1: two rows of sum 7 in 9 columns share >= 5 columns (F4) but at most Lcap = 4 (F3)
CERT99 = [0] * 20 + [1] + [0] * 35 + [1]


def random_free_matrix(m, n, s, t, rng, density=0.65):
    """Greedy random K_{s,t}-free 0/1 matrix (rows as sets of columns)."""
    rows = [set() for _ in range(m)]
    cells = [(i, j) for i in range(m) for j in range(n)]
    rng.shuffle(cells)
    for i, j in cells:
        if rng.random() > density:
            continue
        rows[i].add(j)
        ok = True
        for R in itertools.combinations(range(m), s):
            if i not in R:
                continue
            common = set.intersection(*(rows[r] for r in R))
            if len(common) >= t:
                ok = False
                break
        if not ok:
            rows[i].discard(j)
    return rows


class TestLayout(unittest.TestCase):
    def test_pair_idx_is_a_bijection_onto_range(self):
        for m in range(2, 14):
            idx = sorted(lemmas.pair_idx(m, i, j) for i in range(m) for j in range(i + 1, m))
            self.assertEqual(idx, list(range(lemmas.n_pairs(m))), m)
            for i in range(m):
                for j in range(m):
                    if i != j:
                        self.assertEqual(lemmas.pair_idx(m, i, j), lemmas.pair_idx(m, j, i))

    def test_n_base_and_layout(self):
        for m in range(2, 14):
            L = lemmas.layout(m)
            self.assertEqual(L["y4"][1], lemmas.n_base(m))
            self.assertEqual(lemmas.n_base(m), 2 + 2 * m + m * (m - 1))

    def test_decode_missing_entries_are_zero(self):
        Y = lemmas.decode(9, [3])
        self.assertEqual(Y["y1p"], 3)
        self.assertEqual(Y["y1m"], 0)
        self.assertTrue(all(v == 0 for v in Y["l"] + Y["u"]))
        Y = lemmas.decode(9, CERT99)
        self.assertEqual(Y["y3"][0][1], 1)
        self.assertEqual(Y["y3"][1][0], 1)
        self.assertEqual(Y["y4"][0][1], 1)
        self.assertEqual(sum(map(sum, Y["y3"])) + sum(map(sum, Y["y4"])), 4)


class TestSystem(unittest.TestCase):
    def test_constants_on_the_known_profile(self):
        S = lemmas.farkas_system(9, 9, 3, 3, *KILLED)
        self.assertEqual(S.t2, 230)
        self.assertEqual(S.lo, [31, 31, 26, 21, 21, 21, 21, 21, 21])
        self.assertEqual(S.hi, [33, 33, 29, 25, 25, 25, 25, 25, 25])
        self.assertEqual(S.lcap, 4)
        self.assertEqual(S.cap[0][1], 4)
        self.assertEqual(S.flo[0][1], 5)

    def test_low_top_sum_equal_sorted_forms(self):
        rng = random.Random(1)
        for _ in range(200):
            m, n = rng.randint(2, 9), rng.randint(2, 9)
            cols = [rng.randint(0, m) for _ in range(n)]
            f = lambda c: (max(c - 1, 0)) ** 2  # any monotone f
            for k in range(0, n + 1):
                vals = sorted(f(c) for c in cols)
                self.assertEqual(lemmas.low_sum(f, m, cols, k), sum(vals[:k]))
                self.assertEqual(lemmas.top_sum(f, m, cols, k), sum(vals[n - k :]))

    def test_true_codegrees_satisfy_the_system(self):
        """F1–F4 hold for the true λ of random K_{s,t}-free matrices (the relaxation is sound)."""
        rng = random.Random(7)
        checked = 0
        for m, n, s, t in [(7, 7, 3, 3), (8, 6, 3, 3), (6, 8, 2, 2), (7, 7, 2, 3), (6, 6, 4, 4), (9, 9, 3, 3)]:
            for _ in range(15):
                rows = random_free_matrix(m, n, s, t, rng)
                r = [len(x) for x in rows]
                c = [sum(1 for i in range(m) if j in rows[i]) for j in range(n)]
                S = lemmas.farkas_system(m, n, s, t, r, c)
                lam = [[len(rows[i] & rows[j]) for j in range(m)] for i in range(m)]
                self.assertEqual(sum(lam[i][j] for i in range(m) for j in range(m) if i != j), S.t2)
                for i in range(m):
                    Ri = sum(lam[i][j] for j in range(m) if j != i)
                    self.assertLessEqual(S.lo[i], Ri)
                    self.assertLessEqual(Ri, S.hi[i])
                    for j in range(m):
                        if i != j:
                            self.assertLessEqual(lam[i][j], S.cap[i][j])
                            self.assertLessEqual(S.flo[i][j], lam[i][j])
                # hence no certificate can kill it (Farkas): try the LP oracle
                self.assertIsNone(lemmas.farkas_certificate(m, n, s, t, r, c))
                checked += 1
        self.assertGreater(checked, 50)

    def test_certificate_kills_and_lp_search(self):
        self.assertTrue(lemmas.certificate_kills(9, 9, 3, 3, *KILLED, CERT99))
        self.assertFalse(lemmas.certificate_kills(9, 9, 3, 3, *ALIVE, CERT99))
        self.assertFalse(lemmas.certificate_kills(9, 9, 1, 3, *KILLED, CERT99))  # s < 2 guard
        # malformed / short certificates never kill
        for y in ([], [1], [5, 5, 5], [0] * 20 + [1]):
            self.assertFalse(lemmas.certificate_kills(9, 9, 3, 3, *KILLED, y))
        y = lemmas.farkas_certificate(9, 9, 3, 3, *KILLED)
        self.assertIsNotNone(y)
        self.assertTrue(lemmas.certificate_kills(9, 9, 3, 3, *KILLED, y))
        self.assertIsNone(lemmas.farkas_certificate(9, 9, 3, 3, *ALIVE))


class TestPlumbing(unittest.TestCase):
    def test_validate(self):
        ok, e = lemmas.validate({"farkas": [], "residue": [], "prefix": []}, P99)
        self.assertTrue(ok, e)
        ok, e = lemmas.validate({"farkas": [CERT99, {"m": 9, "n": 9, "s": 3, "t": 3, "y": CERT99}]}, P99)
        self.assertTrue(ok, e)
        bad = [
            {"nope": []},  # unknown family
            {"farkas": [[1, -1]]},  # negative
            {"farkas": [[1, 2.5]]},  # non-integer
            {"farkas": [[True]]},  # bool is not a Nat
            {"farkas": [{"m": 9, "y": CERT99}]},  # missing keys
            {"farkas": [{"m": 9, "n": 9, "s": 3, "t": 3, "y": [0] * 93}]},  # too long for m = 9
            {"farkas": "x"},  # not a list
            {"residue": [{"g": 3}]},  # reserved family with data
            {"farkas": [CERT99] * (lemmas.MAX_CERTS + 1)},  # too many
            "not a dict",
        ]
        for sd in bad:
            ok, e = lemmas.validate(sd, P99)
            self.assertFalse(ok, sd)
            self.assertTrue(e)

    def test_render_terms(self):
        sd = {
            "farkas": [{"m": 9, "n": 9, "s": 3, "t": 3, "y": [0, 1]}, {"m": 10, "n": 10, "s": 3, "t": 3, "y": [2]}, [3]]
        }
        terms = lemmas.render_terms(sd, P99, "ZarPrune.Cand.target1")
        self.assertEqual(len(terms), 1)
        self.assertTrue(terms[0].startswith(lemmas.FAMILIES["farkas"].lean + " ZarPrune.Cand.target1 "))
        self.assertTrue(lemmas.FAMILIES["farkas"].lean.startswith("ZarPrune."))
        self.assertIn("[[0,1], [3]]", terms[0])  # the m=10 entry is not instantiated on (9,9)
        self.assertEqual(lemmas.render_terms({"farkas": []}, P99, "T"), [])
        self.assertEqual(lemmas.render_terms({"farkas": [{"m": 10, "n": 10, "s": 3, "t": 3, "y": [2]}]}, P99, "T"), [])
        with self.assertRaises(ValueError):
            lemmas.render_terms({"farkas": [[-1]]}, P99, "T")

    def test_mirror_kill_round_trip(self):
        sd = {"farkas": [{"m": 9, "n": 9, "s": 3, "t": 3, "y": CERT99}]}
        self.assertTrue(lemmas.mirror_kill(sd, P99, *KILLED))
        self.assertFalse(lemmas.mirror_kill(sd, P99, *ALIVE))
        self.assertFalse(lemmas.mirror_kill(sd, Instance(10, 10, 3, 3, 61), [7] * 10, [7] * 10))  # other cell
        self.assertFalse(lemmas.mirror_kill({"farkas": []}, P99, *KILLED))
        self.assertFalse(lemmas.mirror_kill("garbage", P99, *KILLED))
        # flat entries apply everywhere; the same kill through the flat form
        self.assertTrue(lemmas.mirror_kill({"farkas": [CERT99]}, P99, *KILLED))


class TestLeanAgreement(unittest.TestCase):
    """Real Lean: the gate's schema_mask must equal the mirror on the (9,9,50) table."""

    def test_gate_schema_mask_equals_mirror(self):
        from zar_ub.casetable import load_table
        from zar_ub.lean_gate import run_gate_multi

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "E12_schemas"))
        from search import inlined_candidate  # noqa: E402

        tab = load_table(P99, use_table=False)
        self.assertIsNotNone(tab, "cache/case_table_m9_n9_s3_t3_w50_pure.json missing")
        cases = [(r.rows, r.cols) for r in tab.records]
        old = lemmas.FAMILIES["farkas"].lean
        inline = not os.environ.get("ZAR_UB_SCHEMAS_BUILT")
        if inline:
            lemmas.FAMILIES["farkas"] = lemmas.Family(lean="ZarPrune.Cand.ofFarkasInl", params={}, doc="")
            cand = inlined_candidate()
        else:
            cand = "def candidate (P : Params) : Prune P := counting P\n"
        try:
            sd = {"farkas": [{"m": 9, "n": 9, "s": 3, "t": 3, "y": CERT99}]}
            terms = lemmas.render_terms(sd, P99, "ZarPrune.Cand.target")
            r = run_gate_multi(
                [(P99, cases)],
                cand,
                schema_terms=[terms],
                timeout=240.0,
                tag="test_lemmas",
                sketch=False,
                use_cache=False,
            )[0]
        finally:
            lemmas.FAMILIES["farkas"] = lemmas.Family(lean=old, params={}, doc="")
        self.assertEqual(r.ladder, 5, r.errors[:3])
        self.assertEqual(set(r.axioms), {"propext", "Classical.choice", "Quot.sound"})
        mirror = [lemmas.mirror_kill(sd, P99, rr, cc) for rr, cc in cases]
        self.assertEqual([bool(x) for x in r.schema_mask], mirror)
        self.assertEqual(sum(mirror), 18)  # 18 of the 36 cases, 5 of the 17 library survivors
        self.assertEqual(sum(1 for i in tab.scored_indices() if mirror[i]), 5)
        self.assertTrue(mirror[cases.index((list(KILLED[0]), list(KILLED[1])))])


if __name__ == "__main__":
    unittest.main()

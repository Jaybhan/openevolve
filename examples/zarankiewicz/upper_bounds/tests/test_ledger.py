"""Tests for zar_ub/ledger.py (run: python -m unittest discover tests)."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zar_ub.known import Instance, Ledger, ub_counting, exact_value  # noqa: E402
from zar_ub.partitions import row_partitions, column_partitions  # noqa: E402
from zar_ub import ledger as L  # noqa: E402


class TestFact(unittest.TestCase):
    def test_lean_term_and_roundtrip(self):
        f = L.Fact(9, 8, 3, 3, 45, "tan2022")
        self.assertEqual(f.lean, '⟨9, 8, 3, 3, 45, "tan2022"⟩')
        self.assertEqual(str(f), "tan2022:z(9,8;3,3)<=45")
        self.assertEqual(L.parse_fact(str(f)), f)
        self.assertEqual(f.transpose(), L.Fact(8, 9, 3, 3, 45, "tan2022"))
        self.assertEqual(
            L.lean_fact_list([f, f.transpose()]),
            '[⟨9, 8, 3, 3, 45, "tan2022"⟩, ⟨8, 9, 3, 3, 45, "tan2022"⟩]',
        )
        with self.assertRaises(ValueError):
            L.parse_fact("Roman bound as tabulated")


class TestLedgerFile(unittest.TestCase):
    def test_seed_files_exist_and_are_consistent(self):
        rows = L.load_ledger()
        self.assertGreaterEqual(len(rows), 161 + 42 + 3)
        self.assertEqual(L.check_claims(rows), [])
        self.assertEqual(L.check_claims_file(L.load_claims()), [])
        # every exact_33.csv value is a tan2022 exact row
        n_exact = sum(1 for r in rows if r.kind == "exact" and r.provenance == "tan2022")
        self.assertEqual(n_exact, 161)
        # the pure-mode closure of E9
        e9 = [r for r in rows if r.key == (9, 9, 3, 3) and r.provenance == "lean-here"]
        self.assertEqual(len(e9), 1)
        self.assertEqual(e9[0].bound, 49)
        self.assertEqual(e9[0].hypotheses, "")
        # the conditional closures name their tan2022 facts
        c = [r for r in rows if r.key == (10, 20, 3, 3) and r.provenance == "lean-here"][0]
        self.assertEqual(c.bound, 102)
        self.assertIn(L.Fact(10, 19, 3, 3, 98, "tan2022"), c.hypothesis_facts())
        self.assertEqual(c.notes(), [])

    def test_seed_is_deterministic(self):
        self.assertEqual(L.seed_ledger_rows(), L.load_ledger())

    def test_claims_are_targets_only(self):
        # nothing from an unreviewed 2026 source is a ledger fact
        provs = {r.provenance for r in L.load_ledger()}
        self.assertTrue(provs <= set(L.PROVENANCE_TIER))
        for c in L.load_claims():
            self.assertEqual((c["s"], c["t"]), (3, 3))
            self.assertLessEqual(c["lb"], c["ub"])
        open_cells = L.targets()
        self.assertTrue(all(c["ub"] > c["lb"] for c in open_cells))
        self.assertEqual(len(open_cells), 21)  # LR §2.2: 21 open cells in the block

    def test_write_refuses_inconsistent_ledger(self):
        rows = L.load_ledger()
        bad = rows + [L.LedgerRow(9, 10, 3, 3, 48, "exact", "tan2022")]  # < z(9,9)=49
        errs = L.check_claims(bad)
        self.assertTrue(any("not monotone in n" in e for e in errs))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(L.LedgerError):
                L.write_ledger(bad, os.path.join(d, "ledger.csv"))
            self.assertFalse(os.path.exists(os.path.join(d, "ledger.csv")))
            out = L.write_ledger(rows, os.path.join(d, "ledger.csv"))
            self.assertEqual(L.load_ledger(os.path.join(d, "ledger.csv")), out)

    def test_checker_catches_each_rule(self):
        base = [
            L.LedgerRow(9, 9, 3, 3, 49, "exact", "tan2022"),
            L.LedgerRow(9, 10, 3, 3, 54, "exact", "tan2022"),
        ]
        self.assertEqual(L.check_claims(base, claims=[]), [])
        cases = {
            "deletion": L.LedgerRow(9, 11, 3, 3, 64, "exact", "tan2022"),  # > 54 + 9
            "monotone in m": L.LedgerRow(10, 9, 3, 3, 48, "ub", "tan2022"),  # < 49
            "transposition": L.LedgerRow(10, 9, 3, 3, 53, "exact", "tan2022"),  # != 54
            "counting bound": L.LedgerRow(5, 5, 3, 3, 21, "exact", "tan2022"),  # > 20
            "unknown provenance": L.LedgerRow(13, 19, 3, 3, 122, "ub", "afrasyab26"),
            "duplicate": L.LedgerRow(9, 9, 3, 3, 49, "exact", "tan2022"),
            "exceeds m*n": L.LedgerRow(3, 3, 3, 3, 10, "ub", "tan2022"),
            "without closure_file": L.LedgerRow(9, 9, 3, 3, 49, "ub", "lean-here"),
        }
        for needle, row in cases.items():
            errs = L.check_claims(base + [row], claims=[])
            self.assertTrue(any(needle in e for e in errs), (needle, errs))
        # a verified witness above an upper bound
        claims = [
            {"m": 9, "n": 10, "s": 3, "t": 3, "lb": 55, "lb_verified": 1, "lb_source": "test"}
        ]
        self.assertTrue(any("verified witness" in e for e in L.check_claims(base, claims=claims)))
        # lean-here hypotheses must be ledgered facts
        rows = base + [
            L.LedgerRow(
                9,
                11,
                3,
                3,
                58,
                "ub",
                "lean-here",
                "x.json",
                "tan2022:z(9,10;3,3)<=54|collins16:z(9,10;3,3)<=50",
            )
        ]
        self.assertTrue(any("not a ledger row" in e for e in L.check_claims(rows, claims=[])))


class TestFactsFor(unittest.TestCase):
    def test_pure_is_empty(self):
        self.assertEqual(L.facts_for(Instance(13, 13, 3, 3, 93), "pure"), [])

    def test_matches_partition_generator(self):
        # facts_for is the STATIC list (every tightening proper-prefix bound, both
        # orientations); the generator's Ledger records the DYNAMIC subset it actually
        # consulted (prefix depths it reached), so: used <= facts_for, and facts_for equals
        # the static formula computed independently from exact_value.
        for m, n, w in [
            (9, 9, 50),
            (10, 20, 103),
            (11, 21, 117),
            (12, 12, 81),
            (13, 13, 93),
            (12, 16, 100),
            (11, 18, 102),
            (16, 16, 129),
        ]:
            inst = Instance(m, n, 3, 3, w)
            led = Ledger()
            row_partitions(inst, led, True)
            column_partitions(inst, led, True)
            used = {u for u in led.used}
            mine = {(f.m, f.n, f.s, f.t, f.z) for f in L.facts_for(inst, "tan2022")}
            self.assertTrue(used <= mine, (m, n, w, used - mine))
            static = set()
            for a, b in [(m, k) for k in range(1, n)] + [(k, n) for k in range(1, m)]:
                ev = exact_value(a, b, 3, 3)
                if ev is not None and a >= 3 and b >= 3 and ev < ub_counting(a, b, 3, 3):
                    static.add((a, b, 3, 3, ev))
            self.assertEqual(static, mine, (m, n, w))
            for f in L.facts_for(inst, "tan2022"):
                self.assertEqual(f.tag, "tan2022")
                self.assertTrue((f.m == m and f.n < n) or (f.n == n and f.m < m))
                self.assertLess(f.z, ub_counting(f.m, f.n, f.s, f.t))  # tightening only

    def test_roman_ub_rows_never_tighten(self):
        # Tan's non-bold (Roman) entries equal the waterfilled Argument A bound on every
        # tabulated cell, so the tan2022 ub rows are documentation, never a Fact
        ubs = [r for r in L.load_ledger() if r.kind == "ub" and r.provenance == "tan2022"]
        self.assertEqual(len(ubs), 42)
        for r in ubs:
            self.assertEqual(r.bound, ub_counting(r.m, r.n, r.s, r.t), r)
        inst = Instance(13, 23, 3, 3, 145)
        led = Ledger()
        row_partitions(inst, led, True)
        column_partitions(inst, led, True)
        facts = L.facts_for(inst, "tan2022")
        self.assertEqual({u for u in led.used}, {(f.m, f.n, f.s, f.t, f.z) for f in facts})
        for f in facts:
            self.assertIsNotNone(exact_value(f.m, f.n, f.s, f.t))

    def test_trust_levels(self):
        inst = Instance(13, 18, 3, 3, 117)
        tan = {f.key: f for f in L.facts_for(inst, "tan2022")}
        col = {f.key: f for f in L.facts_for(inst, "collins16")}
        self.assertEqual(col[(13, 17, 3, 3)].z, 110)  # Collins' (13,17) <= 110 admitted
        self.assertEqual(col[(13, 17, 3, 3)].tag, "collins16")
        self.assertNotIn((13, 17, 3, 3), tan)  # default trust: only the counting bound there
        lean = L.facts_for(Instance(9, 10, 3, 3, 55), "lean-here")
        self.assertEqual([(f.m, f.n, f.z) for f in lean], [(9, 9, 49)])
        with self.assertRaises(ValueError):
            L.facts_for(inst, "padhi26")

    def test_symmetric_lookup_orients_facts(self):
        # z(20,10) is looked up through the (10,20) row and returned as a (20,k)/(k,10) fact
        inst = Instance(20, 10, 3, 3, 103)
        facts = L.facts_for(inst, "tan2022")
        self.assertIn(L.Fact(19, 10, 3, 3, 98, "tan2022"), facts)
        self.assertIn(L.Fact(20, 9, 3, 3, 93, "tan2022"), facts)
        b, f = L.ub_ledger(20, 10, 3, 3, "tan2022")
        self.assertEqual((b, f.m, f.n), (102, 20, 10))


if __name__ == "__main__":
    unittest.main()

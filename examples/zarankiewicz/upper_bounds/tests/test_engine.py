"""Unit tests for the zar_ub engine (run: python -m unittest discover tests)."""
import itertools
import os
import sys
import unittest
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zar_ub import Instance, ub_counting, exact_value  # noqa: E402
from zar_ub.partitions import row_partitions, column_partitions  # noqa: E402
from zar_ub.cases import kill_row_argument_d, kill_col_argument_d  # noqa: E402
from zar_ub.encoding import encode_case, has_kst, decode_matrix  # noqa: E402
from zar_ub.solve import solve_cnf  # noqa: E402
from zar_ub.known import max_sum_under_budget  # noqa: E402


def brute_partitions(nparts, cap, total, r, budget):
    out = []
    for p in itertools.product(range(cap + 1), repeat=nparts):
        if list(p) == sorted(p, reverse=True) and sum(p) == total and sum(comb(x, r) for x in p) <= budget:
            out.append(p)
    return out


class TestCounting(unittest.TestCase):
    def test_waterfill_vs_brute(self):
        for k in range(1, 5):
            for cap in range(1, 5):
                for r in (2, 3):
                    for budget in range(0, 12):
                        best = max((sum(p) for p in itertools.product(range(cap + 1), repeat=k)
                                    if sum(comb(x, r) for x in p) <= budget), default=0)
                        self.assertEqual(max_sum_under_budget(k, cap, r, budget), best)

    def test_counting_bound_vs_exact(self):
        # never below a proven exact value; equal on counting-tight cells
        for (m, n, z) in [(3, 3, 8), (4, 4, 13), (5, 5, 20), (6, 6, 26), (9, 9, 49)]:
            self.assertGreaterEqual(ub_counting(m, n, 3, 3), z)
        self.assertEqual(ub_counting(5, 5, 3, 3), 20)


class TestPartitions(unittest.TestCase):
    def test_pure_partitions_vs_brute(self):
        inst = Instance(4, 4, 3, 3, 12)
        cp = column_partitions(inst, use_table=False)
        # pure mode: only Argument A + counting-bound prefixes; brute force applies both
        brute = brute_partitions(4, 4, 12, 3, 2 * comb(4, 3))
        brute = [p for p in brute if all(sum(p[:k]) <= ub_counting(4, k, 3, 3) for k in range(1, 4))]
        self.assertEqual(sorted(cp), sorted(brute))

    def test_no_self_reference(self):
        # the full-length prefix must not use the cell's own value: at (9,9) w=50
        # (z(9,9)=49 is in the table) partitions must still exist
        inst = Instance(9, 9, 3, 3, 50)
        self.assertTrue(len(row_partitions(inst, use_table=True)) > 0)
        self.assertTrue(len(column_partitions(inst, use_table=True)) > 0)


class TestEncoding(unittest.TestCase):
    def brute_exists(self, m, n, s, t, rows, cols):
        for bits in itertools.product((0, 1), repeat=m * n):
            A = [list(bits[i * n:(i + 1) * n]) for i in range(m)]
            if [sum(r) for r in A] != list(rows):
                continue
            if [sum(A[i][j] for i in range(m)) for j in range(n)] != list(cols):
                continue
            if not has_kst(A, s, t):
                return True
        return False

    def test_case_sat_iff_matrix_exists(self):
        # exhaustive cross-check on tiny instances, with and without lex breaking
        m, n, s, t = 3, 4, 2, 2
        for rows in itertools.product(range(n + 1), repeat=m):
            if list(rows) != sorted(rows, reverse=True):
                continue
            for cols in itertools.product(range(m + 1), repeat=n):
                if list(cols) != sorted(cols, reverse=True) or sum(cols) != sum(rows):
                    continue
                inst = Instance(m, n, s, t, sum(rows))
                truth = self.brute_exists(m, n, s, t, rows, cols)
                for lex in (True, False):
                    cnf = encode_case(inst, rows, cols, lex=lex)
                    res = solve_cnf(cnf, inst)
                    self.assertEqual(res.status == "sat", truth, (rows, cols, lex))
                    if res.status == "sat":
                        self.assertTrue(res.witness_ok)

    def test_direct_and_aux_agree(self):
        inst = Instance(5, 5, 3, 3, 20)
        rows = cols = (4, 4, 4, 4, 4)
        r1 = solve_cnf(encode_case(inst, rows, cols, kst_mode="aux"), inst)
        r2 = solve_cnf(encode_case(inst, rows, cols, kst_mode="direct"), inst)
        self.assertEqual(r1.status, r2.status)
        self.assertEqual(r1.status, "sat")


class TestBaselinePrunes(unittest.TestCase):
    def test_argument_d_never_kills_a_realizable_case(self):
        # every SAT case at w = z for small exact cells must survive Argument D
        for (m, n, z) in [(5, 5, 20), (6, 6, 26), (7, 7, 33)]:
            inst = Instance(m, n, 3, 3, z)
            for rows in row_partitions(inst, use_table=False):
                for cols in column_partitions(inst, use_table=False):
                    res = solve_cnf(encode_case(inst, rows, cols), inst)
                    if res.status == "sat":
                        self.assertFalse(kill_row_argument_d(inst, rows, cols), (rows, cols))
                        self.assertFalse(kill_col_argument_d(inst, rows, cols), (rows, cols))


if __name__ == "__main__":
    unittest.main()

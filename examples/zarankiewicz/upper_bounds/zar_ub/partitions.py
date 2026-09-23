"""Admissible row/column sum partitions -- Tan 2022, Algorithm 1.

A column partition for instance (m,n,s,t,w) is a non-increasing tuple
(c_1 >= ... >= c_n), 0 <= c_j <= m, sum c_j = w, such that
  (Argument A)  sum_j C(c_j, s) <= (t-1) * C(m, s)
  (Argument I)  for every prefix length k: c_1 + ... + c_k <= UB(z(m,k;s,t))
                (the m x k minor on the k heaviest columns is itself admissible).
Row partitions are the transposed statement.
"""
from __future__ import annotations

from math import comb
from typing import Callable, List, Tuple

from .known import Instance, Ledger, ub_known


def _partitions(nparts: int, cap: int, total: int, r: int, budget: int,
                prefix_ub: Callable[[int], int]) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    if total < 0 or total > nparts * cap:
        return out

    def rec(prefix: list, psum: int, remaining: int, cost: int, maxpart: int):
        k = len(prefix)
        if k == nparts:
            if remaining == 0:
                out.append(tuple(prefix))
            return
        left = nparts - k
        lo = -(-remaining // left)  # ceil: the largest remaining part is at least the average
        hi = min(maxpart, remaining)
        # Argument I only for PROPER minors (k+1 < nparts): the full-size cell is
        # the quantity under investigation and must never bound itself.
        pu = prefix_ub(k + 1) if k + 1 < nparts else total
        for p in range(lo, hi + 1):
            c = cost + comb(p, r)
            if c > budget:
                break  # cost increases with p
            if psum + p > pu:
                break  # prefix sum increases with p
            prefix.append(p)
            rec(prefix, psum + p, remaining - p, c, p)
            prefix.pop()

    rec([], 0, total, 0, cap)
    return out


def column_partitions(inst: Instance, ledger: Ledger | None = None,
                      use_table: bool = True) -> List[Tuple[int, ...]]:
    m, n, s, t, w = inst.m, inst.n, inst.s, inst.t, inst.w
    budget = (t - 1) * comb(m, s) if m >= s else 10 ** 18
    return _partitions(n, m, w, s, budget, lambda k: ub_known(m, k, s, t, ledger, use_table))


def row_partitions(inst: Instance, ledger: Ledger | None = None,
                   use_table: bool = True) -> List[Tuple[int, ...]]:
    m, n, s, t, w = inst.m, inst.n, inst.s, inst.t, inst.w
    budget = (s - 1) * comb(n, t) if n >= t else 10 ** 18
    return _partitions(m, n, w, t, budget, lambda k: ub_known(k, n, s, t, ledger, use_table))

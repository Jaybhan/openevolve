"""Known values and PROVEN upper bounds used by the case generator.

Two sources only:
  * the counting (Kővári–Sós–Turán / Guy Argument A) bound, computed exactly by
    marginal-cost waterfilling -- proven, no external dependency;
  * the table of proven-exact values data/exact_33.csv (Tan 2022) -- an
    external, cited fact.  Every lookup that actually *tightens* a bound is
    recorded in `Ledger` so a run can list which external theorems it leaned on.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import lru_cache
from math import comb

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(os.path.dirname(_HERE), "data")


@dataclass(frozen=True)
class Instance:
    m: int
    n: int
    s: int
    t: int
    w: int  # decision question: does a valid matrix with >= w ones exist?

    @property
    def tag(self) -> str:
        return f"m{self.m}_n{self.n}_s{self.s}_t{self.t}_w{self.w}"

    def transpose(self) -> "Instance":
        return Instance(self.n, self.m, self.t, self.s, self.w)


# ---------------------------------------------------------------------------
# Counting bound (proven).  max sum c_j  s.t.  sum_j C(c_j, r) <= budget, 0<=c_j<=cap
# ---------------------------------------------------------------------------
def max_sum_under_budget(k: int, cap: int, r: int, budget: int) -> int:
    """Exact integer optimum by marginal-cost waterfilling (proof: marginal cost
    C(c, r-1) of raising a coordinate from c to c+1 is nondecreasing in c, so the
    N globally cheapest increments always form valid prefixes)."""
    if k == 0:
        return 0
    if r == 0:
        return 0 if budget < k else k * cap
    levels = [0] * k
    spent = 0
    total = 0
    while True:
        # cheapest available increment: the lowest current level
        c = min(levels)
        if c >= cap:
            return total
        cost = comb(c, r - 1)
        if spent + cost > budget:
            return total
        i = levels.index(c)
        levels[i] += 1
        spent += cost
        total += 1


@lru_cache(maxsize=None)
def ub_counting(m: int, n: int, s: int, t: int) -> int:
    """PROVEN upper bound on z(m,n;s,t): min of the column-side and row-side
    counting relaxations (Guy's Argument A, both orientations)."""
    if m <= 0 or n <= 0:
        return 0
    if m < s or n < t:
        return m * n
    col_side = max_sum_under_budget(n, m, s, (t - 1) * comb(m, s))
    row_side = max_sum_under_budget(m, n, t, (s - 1) * comb(n, t))
    return min(col_side, row_side)


# ---------------------------------------------------------------------------
# External exact table (cited)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _exact_table_33() -> dict:
    path = os.path.join(_DATA, "exact_33.csv")
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                out[(int(row["m"]), int(row["n"]))] = int(row["z"])
    return out


def exact_value(m: int, n: int, s: int, t: int):
    """Proven-exact value if in the external table (symmetric lookup), else None."""
    if m < s or n < t:
        return m * n
    if (s, t) == (3, 3):
        tab = _exact_table_33()
        if (m, n) in tab:
            return tab[(m, n)]
        if (n, m) in tab:
            return tab[(n, m)]
    return None


class Ledger:
    """Records which external (cited, not re-proven here) facts were used."""

    def __init__(self):
        self.used = set()

    def note(self, m, n, s, t, value):
        self.used.add((m, n, s, t, value))

    def as_list(self):
        return sorted(
            f"z({m},{n};{s},{t}) = {v}  [external: Tan 2022 Table 3 / data/exact_33.csv]"
            for (m, n, s, t, v) in self.used
        )


def ub_known(m: int, n: int, s: int, t: int, ledger: Ledger | None = None,
             use_table: bool = True) -> int:
    """Best PROVEN upper bound available: min(counting bound, external exact value).
    With use_table=False ("pure" mode) only the first-principles counting bound is
    used, so everything the pipeline relies on is provable inside ZarPrune."""
    ub = ub_counting(m, n, s, t)
    ev = exact_value(m, n, s, t) if use_table else None
    if ev is not None and ev < ub:
        if ledger is not None:
            ledger.note(m, n, s, t, ev)
        return ev
    return ub

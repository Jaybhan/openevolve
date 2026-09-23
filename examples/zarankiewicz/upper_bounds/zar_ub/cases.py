"""Cases = (row partition, column partition) pairs, and the baseline prunes.

A *prune* is a predicate kill(inst, rows, cols) -> bool with the contract

    kill(inst, rows, cols) == True  ==>  no K_{s,t}-free m x n matrix has
                                         row sums `rows` and column sums `cols`
                                         (hence none with >= w ones in this case).

Prunes must only kill EMPTY cases (never "dominated" ones -- symmetry breaking
is an *addition* handled inside the SAT encoding, see encoding.py).  Every
baseline prune below has a proof sketch in its docstring and a Lean
counterpart (or a noted gap) in lean/ZarPrune/Prunes.lean.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import comb
from typing import Callable, Iterable, List, Sequence, Tuple

from .known import Instance, Ledger
from .partitions import column_partitions, row_partitions


@dataclass(frozen=True)
class Case:
    rows: Tuple[int, ...]  # non-increasing row sums, length m
    cols: Tuple[int, ...]  # non-increasing column sums, length n

    @property
    def key(self) -> str:
        return "r" + ".".join(map(str, self.rows)) + "_c" + ".".join(map(str, self.cols))


KillFn = Callable[[Instance, Sequence[int], Sequence[int]], bool]


@dataclass(frozen=True)
class Prune:
    name: str
    kill: KillFn
    lean_name: str  # name of the corresponding Lean term (or "" if not yet formalized)
    doc: str = ""


# ---------------------------------------------------------------------------
# Baseline prunes
# ---------------------------------------------------------------------------
def kill_row_argument_d(inst: Instance, rows, cols) -> bool:
    """Guy's Argument D (row form).  Take the row with the most ones, r = rows[0].
    Its ones lie in r distinct columns with sums c_{j_1..j_r}; each such column
    contributes C(c_j - 1, s-1) (s-1)-subsets of *other* rows that share that
    column with our row; each s-set of rows containing our row is covered by at
    most t-1 columns, and there are C(m-1, s-1) such s-sets.  Hence
        sum_{i=1}^{r} C(c_{j_i} - 1, s-1) <= (t-1) C(m-1, s-1).
    The LHS is minimized by the r *lightest* columns; if it fails there it fails
    everywhere, so the case is empty."""
    m, s, t = inst.m, inst.s, inst.t
    if m < s:
        return False
    r = rows[0]
    if r == 0:
        return False
    lightest = sorted(cols)[:r]
    lhs = sum(comb(c - 1, s - 1) for c in lightest if c >= 1)
    return lhs > (t - 1) * comb(m - 1, s - 1)


def kill_col_argument_d(inst: Instance, rows, cols) -> bool:
    """Argument D, column form (transpose of kill_row_argument_d)."""
    n, s, t = inst.n, inst.s, inst.t
    if n < t:
        return False
    c = cols[0]
    if c == 0:
        return False
    lightest = sorted(rows)[:c]
    lhs = sum(comb(r - 1, t - 1) for r in lightest if r >= 1)
    return lhs > (s - 1) * comb(n - 1, t - 1)


def kill_mismatch(inst: Instance, rows, cols) -> bool:
    """Row-sum total must equal column-sum total (both count the ones)."""
    return sum(rows) != sum(cols)


def kill_caps(inst: Instance, rows, cols) -> bool:
    """A row sum exceeds n or a column sum exceeds m."""
    return any(r > inst.n for r in rows) or any(c > inst.m for c in cols)


def kill_deficit(inst: Instance, rows, cols) -> bool:
    """Total ones below the target w."""
    return sum(rows) < inst.w


BASELINE_PRUNES: List[Prune] = [
    Prune("deficit", kill_deficit, "deficit"),
    Prune("mismatch", kill_mismatch, "mismatch"),
    Prune("caps", kill_caps, "rowCap|colCap"),
    Prune("row_argument_d", kill_row_argument_d, "", "Guy Argument D, row form"),
    Prune("col_argument_d", kill_col_argument_d, "", "Guy Argument D, column form"),
]


def baseline_kill(inst: Instance, rows, cols) -> str:
    """Name of the first baseline prune that kills the case, or ''."""
    for p in BASELINE_PRUNES:
        if p.kill(inst, rows, cols):
            return p.name
    return ""


def enumerate_cases(inst: Instance, ledger: Ledger | None = None, use_table: bool = True) -> List[Case]:
    """All (row partition, column partition) pairs admissible by Arguments A and I."""
    rp = row_partitions(inst, ledger, use_table)
    cp = column_partitions(inst, ledger, use_table)
    return [Case(r, c) for r, c in product(rp, cp)]

"""E8: the verified counting library as a candidate (copy of initial_program.py).

LEAN_SOURCE folds the four baseline prunes with `ZarPrune.counting`
(argA, argAT, argD, argDT, argDelColWF, argDelRowWF, argWF from
lean/ZarPrune/Counting.lean).  kill() mirrors every one of those prunes with the
same integer arithmetic as the Lean definitions (Nat truncated subtraction,
`Nat.choose`, `Nat.findGreatest`), so the Python and Lean masks agree on every
profile, not only on capped ones.

Available Lean API (ZarPrune, Mathlib-free): sumFin, allFin, sumFin_swap (Fubini),
sumFin_le, allFin_iff, not_allFin_elim; Params{m,n,s,t,w}, Mat, ind, rowSum,
colSum, weight, rowSum_le, colSum_le, weight_eq_sum_colSum, HasKst, Valid,
Profile{row,col}, profileOf; Prune{name,kill,sound}, Prune.never, Prune.or,
Prune.ofList; proved prunes deficit, mismatch, rowCap, colCap, baseline.
Counting library (ZarPrune.Counting, Mathlib): argA, argAT, argD, argDT,
argDelColWF, argDelRowWF, argWF, countingA, countingD, deletion, counting.
"""
from math import comb

# EVOLVE-BLOCK-START
LEAN_SOURCE = r'''
/-- Baseline prunes OR the proved counting library (Guy's Arguments A and D,
    both sides, plus the waterfilled deletion prunes). -/
def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)
'''


def _nsub(a, b):
    """Lean `Nat` subtraction (truncated at 0)."""
    return a - b if a >= b else 0


# ---- Argument A ------------------------------------------------------------
def _argA(m, n, s, t, w, rows, cols):
    return (t - 1) * comb(m, s) < sum(comb(c, s) for c in cols)


def _argAT(m, n, s, t, w, rows, cols):
    return (s - 1) * comb(n, t) < sum(comb(r, t) for r in rows)


# ---- Argument D (layer-cake form of "sum of fD over the r lightest columns") --
def _fD(s, c):
    return comb(_nsub(c, 1), _nsub(s, 1))


def _gD(s, x):
    return _nsub(_fD(s, x), _fD(s, _nsub(x, 1)))


def _boundD(m, s, cols, r):
    def cnt_lt(y):
        return sum(1 for c in cols if c < y)
    return r * _fD(s, 0) + sum(_nsub(r, cnt_lt(x + 1)) * _gD(s, x + 1) for x in range(m))


def _argD(m, n, s, t, w, rows, cols):
    if s < 1:
        return False
    rhs = (t - 1) * comb(_nsub(m, 1), _nsub(s, 1))
    return any(_boundD(m, s, cols, r) > rhs for r in rows)


def _argDT(m, n, s, t, w, rows, cols):
    # argD on the transposed instance (n, m, t, s, w) with the swapped profile
    return _argD(n, m, t, s, w, cols, rows)


# ---- Waterfilling / deletion -------------------------------------------------
def _col_budget(m, s, t):
    return (t - 1) * comb(m, s)


def _equal_cost(n, s, S):
    if n == 0:
        return 0  # S / 0 = 0 and S % 0 = S in Lean, n * ... = 0
    q, r = divmod(S, n)
    return n * comb(q, s + 1) + r * comb(q, s)


def _waterfill_bound(m, n, s, B):
    """Nat.findGreatest (fun S => equalCost n (s-1) S ≤ B) (n*m)."""
    for S in range(n * m, -1, -1):
        if _equal_cost(n, _nsub(s, 1), S) <= B:
            return S
    return 0


def _argDelColWF(m, n, s, t, w, rows, cols):
    if s < 1:
        return False
    n1 = _nsub(n, 1)
    U = _waterfill_bound(m, n1, s, _col_budget(m, s, t))
    return any(c + U < w for c in cols)


def _argDelRowWF(m, n, s, t, w, rows, cols):
    if s < 1:
        return False
    m1 = _nsub(m, 1)
    U = _waterfill_bound(m1, n, s, _col_budget(m1, s, t))
    return any(r + U < w for r in rows)


def _argWF(m, n, s, t, w, rows, cols):
    return s >= 1 and _waterfill_bound(m, n, s, _col_budget(m, s, t)) < w


# ---- baseline ----------------------------------------------------------------
def _baseline(m, n, s, t, w, rows, cols):
    total = sum(rows)
    if total < w:                      # deficit
        return True
    if total != sum(cols):             # mismatch
        return True
    if any(r > n for r in rows) or any(c > m for c in cols):  # rowCap / colCap
        return True
    return False


COUNTING = (_argA, _argAT, _argD, _argDT, _argDelColWF, _argDelRowWF, _argWF)


def kill(m, n, s, t, w, rows, cols):
    """Python mirror of `candidate.kill` = baseline || counting."""
    if _baseline(m, n, s, t, w, rows, cols):
        return True
    return any(p(m, n, s, t, w, rows, cols) for p in COUNTING)
# EVOLVE-BLOCK-END


if __name__ == "__main__":
    print(kill(9, 9, 3, 3, 50, (6, 6, 6, 6, 6, 5, 5, 5, 5), (6, 6, 6, 6, 6, 5, 5, 5, 5)))

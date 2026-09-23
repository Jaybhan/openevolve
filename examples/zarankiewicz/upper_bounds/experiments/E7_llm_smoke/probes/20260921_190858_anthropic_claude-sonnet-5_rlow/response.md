<<<<<<< SEARCH
def _argWF(m, n, s, t, w, rows, cols):
    return s >= 1 and _waterfill_bound(m, n, s, _col_budget(m, s, t)) < w
=======
def _argWF(m, n, s, t, w, rows, cols):
    return s >= 1 and _waterfill_bound(m, n, s, _col_budget(m, s, t)) < w


def _argDel2ColWF(m, n, s, t, w, rows, cols):
    """Delete the two heaviest columns; remaining (m,n-2) matrix is still
    K_{s,t}-free, so its weight is waterfill-bounded.  Empirical (unproven
    in Lean yet), sound by repeated application of valid_deleteCol_bound."""
    if s < 1 or n < 2:
        return False
    n2 = _nsub(n, 2)
    U = _waterfill_bound(m, n2, s, _col_budget(m, s, t))
    c_sorted = sorted(cols, reverse=True)
    c1, c2 = c_sorted[0], c_sorted[1]
    return c1 + c2 + U < w


def _argDel2RowWF(m, n, s, t, w, rows, cols):
    if s < 1 or m < 2:
        return False
    m2 = _nsub(m, 2)
    U = _waterfill_bound(m2, n, s, _col_budget(m2, s, t))
    r_sorted = sorted(rows, reverse=True)
    r1, r2 = r_sorted[0], r_sorted[1]
    return r1 + r2 + U < w
>>>>>>> REPLACE

<<<<<<< SEARCH
COUNTING = (_argA, _argAT, _argD, _argDT, _argDelColWF, _argDelRowWF, _argWF)
=======
COUNTING = (_argA, _argAT, _argD, _argDT, _argDelColWF, _argDelRowWF, _argWF,
            _argDel2ColWF, _argDel2RowWF)
>>>>>>> REPLACE
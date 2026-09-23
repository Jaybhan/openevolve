"""Reference Python implementation of the Davies-Gill-Horsley (DGH) constraint (4) with
v = s-1 (literature review P8; docs/lit/dgh2024_lp_bounds.md section 3.5), as a profile prune.

Two equivalent integer forms are provided.  With D = k-s+1, R = (t-1)(m-s+1), alpha = R mod D,
c = (R-alpha)/D and ch_j = C(c_j, s-1):

  P8 form (literature review):
    sum_{j: c_j<k} (c_j-s+1)*ch_j + (D-alpha)*sum_{j: c_j>=k} ch_j
        <= (D-alpha)*c*C(m,s-1) + alpha*sum_{j: c_j<k} ch_j
  compact form (the Lean statement `dghBudget` in lean/ZarPrune/DGH.lean):
    (D-alpha)*sum_j ch_j <= (D-alpha)*c*C(m,s-1) + sum_j (k - c_j)_+ * ch_j

They are the same inequality (move the (k-c_j) terms across; for c_j < k,
(D-alpha) - (k-c_j) = c_j-s+1-alpha).  A profile is KILLED when the inequality fails for
some k in [s, m]; the row side is the same test on the transposed instance (n, m; t, s).
"""
from math import comb


def dgh_violated_compact(m, s, t, cols, k):
    v = s - 1
    R = (t - 1) * (m - s + 1)
    D = k - s + 1
    a, c = R % D, R // D
    ch = [comb(cj, v) for cj in cols]
    lhs = (D - a) * sum(ch)
    rhs = (D - a) * c * comb(m, v) + sum(max(k - cj, 0) * chj for cj, chj in zip(cols, ch))
    return lhs > rhs


def dgh_violated_p8(m, s, t, cols, k):
    v = s - 1
    R = (t - 1) * (m - s + 1)
    D = k - s + 1
    a, c = R % D, R // D
    lo = [cj for cj in cols if cj < k]
    hi = [cj for cj in cols if cj >= k]
    lhs = sum((cj - s + 1) * comb(cj, v) for cj in lo) + (D - a) * sum(comb(cj, v) for cj in hi)
    rhs = (D - a) * c * comb(m, v) + a * sum(comb(cj, v) for cj in lo)
    return lhs > rhs


def dgh_col_kill(m, s, t, cols, form=dgh_violated_compact):
    """True iff the DGH(v=s-1) inequality fails for some k in [s, m] on the column sums."""
    return s >= 1 and any(form(m, s, t, cols, k) for k in range(s, m + 1))


def dgh_col_kill_ks(m, s, t, cols):
    """The list of k in [s, m] at which the inequality fails (empty = survives)."""
    if s < 1:
        return []
    return [k for k in range(s, m + 1) if dgh_violated_compact(m, s, t, cols, k)]


def dgh_kill(m, n, s, t, w, rows, cols):
    """Both orientations: columns of (m,n;s,t) and columns of the transpose (n,m;t,s) = rows."""
    return dgh_col_kill(m, s, t, cols) or dgh_col_kill(n, t, s, rows)

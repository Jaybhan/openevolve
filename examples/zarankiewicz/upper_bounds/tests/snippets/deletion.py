def kill(m, n, s, t, w, rows, cols):
    """Python mirror (SNIPPET:deletion): library || argDelColWFT (row-side waterfill of (m, n-1))."""
    if sum(comb(c, s) for c in cols) > (t - 1) * comb(m, s):
        return True                                                   # argA
    if sum(comb(r, t) for r in rows) > (s - 1) * comb(n, t):
        return True                                                   # argAT
    r0, c0 = rows[0], cols[0]
    if r0 and sum(comb(c - 1, s - 1) for c in sorted(cols)[:r0] if c) > (t - 1) * comb(m - 1, s - 1):
        return True                                                   # argD
    if c0 and sum(comb(r - 1, t - 1) for r in sorted(rows)[:c0] if r) > (s - 1) * comb(n - 1, t - 1):
        return True                                                   # argDT
    return _argDelColWFT(m, n, s, t, w, rows, cols)


def _equal_cost(n, s, S):
    if n == 0:
        return 0
    q, r = divmod(S, n)
    return n * comb(q, s + 1) + r * comb(q, s)


def _waterfill_bound(m, n, s, B):
    """Lean `waterfillBound m n s B` = Nat.findGreatest (fun S => equalCost n (s-1) S ≤ B) (n*m)."""
    for S in range(n * m, -1, -1):
        if _equal_cost(n, max(s - 1, 0), S) <= B:
            return S
    return 0


def _argDelColWFT(m, n, s, t, w, rows, cols):
    if t < 1:
        return False
    n1 = max(n - 1, 0)
    U = _waterfill_bound(n1, m, t, max(s - 1, 0) * comb(n1, t))
    return any(c + U < w for c in cols)

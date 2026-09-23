def kill(m, n, s, t, w, rows, cols):
    """UNSOUND mirror (SNIPPET:unsound_row7): library || 'some row has sum >= 7'.
    A witnessed profile at (9,10,54) has a row of sum 7, so the battery must reject it."""
    if sum(comb(c, s) for c in cols) > (t - 1) * comb(m, s):
        return True                                                   # argA
    if sum(comb(r, t) for r in rows) > (s - 1) * comb(n, t):
        return True                                                   # argAT
    r0, c0 = rows[0], cols[0]
    if r0 and sum(comb(c - 1, s - 1) for c in sorted(cols)[:r0] if c) > (t - 1) * comb(m - 1, s - 1):
        return True                                                   # argD
    if c0 and sum(comb(r - 1, t - 1) for r in sorted(rows)[:c0] if r) > (s - 1) * comb(n - 1, t - 1):
        return True                                                   # argDT
    return any(r >= 7 for r in rows)                                  # UNSOUND

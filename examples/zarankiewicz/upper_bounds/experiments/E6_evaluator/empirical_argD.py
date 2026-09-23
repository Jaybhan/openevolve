from math import comb
LEAN_SOURCE = r'''
def candidate (P : Params) : Prune P :=
  Prune.ofList P [deficit P, mismatch P, rowCap P, colCap P]
'''
def kill(m, n, s, t, w, rows, cols):
    if sum(rows) < w or sum(rows) != sum(cols): return True
    if max(rows) > n or max(cols) > m: return True
    # Guy's Argument D, both orientations (sound, but NOT yet proved in Lean)
    r = rows[0]; lightest = sorted(cols)[:r]
    if sum(comb(c-1, s-1) for c in lightest if c >= 1) > (t-1)*comb(m-1, s-1): return True
    c = cols[0]; lightest = sorted(rows)[:c]
    if sum(comb(x-1, t-1) for x in lightest if x >= 1) > (s-1)*comb(n-1, t-1): return True
    # a stronger row-local budget: the heaviest row must pick r columns; even the r lightest
    # columns must leave room for the OTHER rows' pairs: (empirical, unproven idea) -- none yet
    return False

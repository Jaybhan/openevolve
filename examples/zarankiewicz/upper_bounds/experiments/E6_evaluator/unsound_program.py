LEAN_SOURCE = r'''
def candidate (P : Params) : Prune P :=
  Prune.ofList P [deficit P, mismatch P, rowCap P, colCap P]
'''
def kill(m, n, s, t, w, rows, cols):
    # "reflex" symmetry-break disguised as a prune: kills any case with a row of sum >= 7
    return sum(rows) < w or sum(rows) != sum(cols) or max(rows) >= 7

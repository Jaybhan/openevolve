<<<<<<< SEARCH
def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)
=======
def candidate (P : Params) : Prune P :=
  Prune.or (baseline P)
    (Prune.or (counting P) (Prune.transposed (counting P.transpose)))
>>>>>>> REPLACE

<<<<<<< SEARCH
COUNTING = (_argA, _argAT, _argD, _argDT, _argDelColWF, _argDelRowWF, _argWF)
=======
COUNTING = (
    _argA, _argAT, _argD, _argDT,
    _argDelColWF, _argDelRowWF, _argWF,
)

# The Lean library's bundled `counting` is already symmetric, but retaining
# the explicit transposed copy here makes the Python mirror robust to changes
# in the bundling of the Lean prunes.  In particular, deletion and
# waterfilling are checked in both orientations.
COUNTING_TRANSPOSED = (
    lambda m, n, s, t, w, rows, cols:
        _argA(n, m, t, s, w, cols, rows),
    lambda m, n, s, t, w, rows, cols:
        _argAT(n, m, t, s, w, cols, rows),
    lambda m, n, s, t, w, rows, cols:
        _argD(n, m, t, s, w, cols, rows),
    lambda m, n, s, t, w, rows, cols:
        _argDT(n, m, t, s, w, cols, rows),
    lambda m, n, s, t, w, rows, cols:
        _argDelColWF(n, m, t, s, w, cols, rows),
    lambda m, n, s, t, w, rows, cols:
        _argDelRowWF(n, m, t, s, w, cols, rows),
    lambda m, n, s, t, w, rows, cols:
        _argWF(n, m, t, s, w, cols, rows),
)
>>>>>>> REPLACE

<<<<<<< SEARCH
    return any(p(m, n, s, t, w, rows, cols) for p in COUNTING)
=======
    return (
        any(p(m, n, s, t, w, rows, cols) for p in COUNTING)
        or any(p(m, n, s, t, w, rows, cols)
               for p in COUNTING_TRANSPOSED)
    )
>>>>>>> REPLACE

This adds no new mathematical assertion: every explicitly transposed test is
already supplied by the proved `counting` bundle through `argAT`, `argDT`,
and the transposed deletion prunes.  Consequently the Lean side remains
sound, while the Python mirror is made explicitly identical to the expanded
Lean expression.  The main practical benefit is defensive: if `counting` is
later refactored to omit one of its orientation-specific components, the
mirror will not silently disagree with the Lean candidate.
Looking at the hard cases, many involve asymmetric shapes (m9n10, m10n11) where a *row-based* waterfilling bound (the transpose of the existing column-based `argWF`) is currently missing from the bundle. Since `Prune.transposed` and `argWF` are already proven in the library, wrapping `argWF` on the transposed parameters gives a new, low-risk, general prune `argWFT` essentially for free.

<<<<<<< SEARCH
LEAN_SOURCE = r'''
/-- Baseline prunes OR the proved counting library (Guy's Arguments A and D,
    both sides, plus the waterfilled deletion prunes). -/
def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)
'''
=======
LEAN_SOURCE = r'''
/-- Transposed wa
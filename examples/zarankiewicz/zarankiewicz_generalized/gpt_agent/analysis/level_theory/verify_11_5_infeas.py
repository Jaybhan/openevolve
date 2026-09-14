"""STATUS NOTE (2026-07-29): first cadical run exceeded 5 min and was
superseded — the nonexistence is CLASSICAL (Dehon 1976, Discrete Math. 15,
23-25; simplified proof in Kiermaier-Pavcevic arXiv:1405.6110 Thm 3.1),
independently re-proven by our structured MILP (decisions.csv row 1).
Kept for optional third-route verification at leisure.

Independent re-verification (second encoding, second solver) that no
perfect 2-fold level-5 packing on 11 points exists, i.e. NO 3-(11,5,2)
design: every triple covered exactly twice by 33 pentads, block mult <= 2
(automatic: mult >= 3 would cover a triple 3 times).

Encoding: pysat, x_b in {0,1,2} as ordered booleans b1 >= b2; per-triple
EXACT-2 cardinality via CardEnc.equals (totalizer). Solver: cadical.
"""
from itertools import combinations
from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool
from pysat.solvers import Cadical153

m, w = 11, 5
types = list(combinations(range(m), w))
pool = IDPool()
v1 = {b: pool.id(("a", b)) for b in types}
v2 = {b: pool.id(("b", b)) for b in types}
cnf = CNF()
for b in types:
    cnf.append([-v2[b], v1[b]])  # ordering: second copy implies first
for T in combinations(range(m), 3):
    lits = []
    for b in types:
        if set(T) <= set(b):
            lits += [v1[b], v2[b]]
    cnf.extend(CardEnc.equals(lits=lits, bound=2, vpool=pool,
                              encoding=EncType.totalizer).clauses)
with Cadical153(bootstrap_with=cnf) as s:
    res = s.solve()
print("3-(11,5,2) design exists?", res, "(expect False = INFEAS confirmed)")

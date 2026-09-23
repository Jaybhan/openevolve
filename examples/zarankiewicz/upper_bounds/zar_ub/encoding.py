"""CNF encoding of one case: an m x n 0/1 matrix, K_{s,t}-free, with FIXED row
sums `rows` and column sums `cols`, plus lexicographic symmetry breaking within
equal-sum groups (Tan 2022, Sections 3.2-3.3).

Soundness of the symmetry breaking (an *addition*, not a prune): rows with equal
sums may be permuted among themselves, and columns with equal sums likewise,
without changing the profile or K_{s,t}-freeness; Tan's Theorem 3.2 shows the
alternating sort reaches a fixed point where every equal-sum group is
lexicographically sorted in both directions simultaneously.  Hence
    case SAT  <=>  some valid matrix has this exact profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Sequence

from pysat.card import CardEnc, EncType

from .known import Instance


@dataclass
class CNF:
    clauses: List[List[int]] = field(default_factory=list)
    nvars: int = 0
    X: List[List[int]] = field(default_factory=list)  # X[i][j] grid variable

    def new_var(self) -> int:
        self.nvars += 1
        return self.nvars

    def add(self, clause):
        self.clauses.append(list(clause))

    def extend_card(self, cnfobj):
        self.clauses.extend(cnfobj.clauses)
        self.nvars = max(self.nvars, cnfobj.nv)

    def to_dimacs(self) -> str:
        lines = [f"p cnf {self.nvars} {len(self.clauses)}"]
        lines.extend(" ".join(map(str, c)) + " 0" for c in self.clauses)
        return "\n".join(lines) + "\n"


def _equals(cnf: CNF, lits: Sequence[int], k: int, enc=EncType.seqcounter):
    lits = list(lits)
    if k <= 0:
        for l in lits:
            cnf.add([-l])
        return
    if k >= len(lits):
        for l in lits:
            cnf.add([l])
        return
    cnf.extend_card(CardEnc.equals(lits=lits, bound=k, top_id=cnf.nvars, encoding=enc))


def _atmost(cnf: CNF, lits: Sequence[int], k: int, enc=EncType.seqcounter):
    lits = list(lits)
    if k >= len(lits):
        return
    if k <= 0:
        for l in lits:
            cnf.add([-l])
        return
    cnf.extend_card(CardEnc.atmost(lits=lits, bound=k, top_id=cnf.nvars, encoding=enc))


def _lex_ge(cnf: CNF, a: Sequence[int], b: Sequence[int]):
    """Enforce a >=_lex b (1 before 0, i.e. as binary numbers a >= b).
    e_i = 'prefixes of length i are equal'; e_0 = true."""
    e_prev = None  # None encodes the constant true e_0
    for i in range(len(a)):
        ai, bi = a[i], b[i]
        pre = [] if e_prev is None else [-e_prev]
        # if prefix equal so far: not (a_i = 0 and b_i = 1)
        cnf.add(pre + [ai, -bi])
        if i == len(a) - 1:
            break
        e = cnf.new_var()
        # prefix equal & a_i = b_i  ->  e
        cnf.add(pre + [-ai, -bi, e])
        cnf.add(pre + [ai, bi, e])
        e_prev = e


def encode_case(inst: Instance, rows: Sequence[int], cols: Sequence[int],
                kst_mode: str = "aux", lex: bool = True) -> CNF:
    m, n, s, t = inst.m, inst.n, inst.s, inst.t
    assert len(rows) == m and len(cols) == n
    cnf = CNF()
    X = [[i * n + j + 1 for j in range(n)] for i in range(m)]
    cnf.X = X
    cnf.nvars = m * n

    # --- K_{s,t}-freeness ---------------------------------------------------
    if m >= s and n >= t:
        if kst_mode == "direct":
            for R in combinations(range(m), s):
                for C in combinations(range(n), t):
                    cnf.add([-X[i][j] for i in R for j in C])
        else:  # 'aux': per s-subset of rows, y_c <- AND of the s cells in column c
            for R in combinations(range(m), s):
                ys = []
                for j in range(n):
                    y = cnf.new_var()
                    ys.append(y)
                    cnf.add([-X[i][j] for i in R] + [y])
                _atmost(cnf, ys, t - 1)

    # --- fixed row and column sums -------------------------------------------
    for i in range(m):
        _equals(cnf, X[i], rows[i])
    for j in range(n):
        _equals(cnf, [X[i][j] for i in range(m)], cols[j])

    # --- lexicographic symmetry breaking within equal-sum groups -------------
    if lex:
        for i in range(m - 1):
            if rows[i] == rows[i + 1]:
                _lex_ge(cnf, X[i], X[i + 1])
        for j in range(n - 1):
            if cols[j] == cols[j + 1]:
                _lex_ge(cnf, [X[i][j] for i in range(m)], [X[i][j + 1] for i in range(m)])
    return cnf


def decode_matrix(cnf: CNF, model: Sequence[int]):
    pos = set(l for l in model if l > 0)
    return [[1 if cnf.X[i][j] in pos else 0 for j in range(len(cnf.X[0]))] for i in range(len(cnf.X))]


def has_kst(A, s: int, t: int) -> bool:
    """Independent witness check: does A contain an all-ones s x t submatrix?"""
    m = len(A)
    n = len(A[0]) if m else 0
    if m < s or n < t:
        return False
    for R in combinations(range(m), s):
        common = 0
        for j in range(n):
            if all(A[i][j] for i in R):
                common += 1
                if common >= t:
                    return True
    return False

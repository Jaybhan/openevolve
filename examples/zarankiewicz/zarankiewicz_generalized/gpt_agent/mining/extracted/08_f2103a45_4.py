# MINED PROGRAM — rank 8 of 8
# id: f2103a45-ce70-451b-b400-e33974c3b21f
# generation 0, iteration_found 0, island 0, parent None -- this IS the seed.
# Byte-identical to BASE/initial_program.py (verified with `diff`, exit code 0).
# metrics: exact_count=4/161  combined_score=0.4024  validity=1.0
# FAMILY: cyclic difference family on Z_M. Rows = Z_M; column j carries the
# block {j+d mod M : d in D_c}, D_c a rotated prefix of the triangular numbers
# t_i=i(i+1)/2 mod M (a closed-form, Sidon-like offset sequence); a canonical
# "drop the highest-indexed offending entry" repair pass enforces the K_{3,3}
# bound when the triangular offsets collide. Despite being the single most
# mathematically self-contained, provably-general construction ever run in
# this pipeline (one formula, no per-M branching, no hardcoded tables), it is
# also the WORST performer of any lineage that survived past generation 0:
# every one of its 7 first-generation children (checkpoint 10) scored
# exact_count 0 or a few, and by generation 2 every island had abandoned this
# idea for algebraic (F_2^4 / norm-graph) constructions. See mining_report.md
# for why the repair step is the likely culprit. Code below is VERBATIM from
# the checkpoint JSON (== initial_program.py).

# EVOLVE-BLOCK-START
"""z(M,N;3,3) from ONE principle: a cyclic difference family on Z_M.

Rows are the residues Z_M.  Column j carries the block

        B_j = { j + d  (mod M) :  d in D_{c(j)} },      c(j) = j // M

so the matrix is a stack of circulant strips: within a strip every column
is a translate of one base block and the row group Z_M acts on it, and each
successive strip rotates the base block to a fresh offset set.

Why this is K_{3,3}-free.  Two rows a, b lie in a common column j exactly
when b - a is a difference of the base block D_{c(j)}.  So the number of
columns containing both a and b is the multiplicity of b - a in the pooled
difference multiset of the base blocks.  If every nonzero residue occurs at
most twice there, then every PAIR of rows shares at most 2 columns, and
since a triple's common columns sit inside any of its pairs', every TRIPLE
shares at most 2 as well.  That is exactly the K_{3,3} condition, obtained
from a statement about differences alone -- no triple ever has to be
inspected.

Choosing D.  The base blocks are prefixes of the triangular offsets
T = ( 0, 1, 3, 6, 10, 15, ... ),  t_i = i(i+1)/2 mod M, a closed-form
Sidon-like sequence: the difference t_i - t_j = (i-j)(i+j+1)/2 is
determined by the pair (i-j, i+j), so collisions are rare and the
difference multiset is about as flat as a formula can make it.  The block
size k comes from the counting bound sum_j C(c_j, 3) <= 2 C(M, 3), which
says how large the blocks may be before triples must repeat.

Repair.  Flatness is not exactness: for some M a difference does land three
times.  The last step is a canonical deletion -- scan triples in lex order
and drop the highest-indexed offending entry -- which is a deterministic
function of the matrix, not a search.
"""

import numpy as np
from math import comb

S = 3          # rows in the forbidden K_{s,t}
T = 3          # columns in the forbidden K_{s,t}
LAMBDA = T - 1  # a pair of rows may share at most this many columns


def _block_size(M, N):
    """Largest uniform block size the counting bound still allows.

    Every column of degree c uses up C(c, S) of the row-triple budget, and
    the budget is (T-1) * C(M, S).  Solve N * C(k, S) <= budget for k.
    """
    if M < S:
        return M
    budget = (T - 1) * comb(M, S)
    k = min(S - 1, M)
    while k < M and N * comb(k + 1, S) <= budget:
        k += 1
    return max(k, 1)


def _offsets(M, k, shift):
    """The triangular offset set, rotated by `shift`: closed form, no search."""
    return {(shift + (i * (i + 1)) // 2) % M for i in range(k)}


def _repair(A):
    """Canonical deletion: enforce the K_{S,T} condition by dropping the
    highest-indexed entry of each offending configuration.  Deterministic --
    the same matrix always yields the same result."""
    M, N = A.shape
    rows = [sum(int(A[i, j]) << j for j in range(N)) for i in range(M)]
    for a in range(M):
        for b in range(a + 1, M):
            ab = rows[a] & rows[b]
            if ab.bit_count() <= LAMBDA:
                continue
            for c in range(b + 1, M):
                common = ab & rows[c]
                while common.bit_count() > LAMBDA:
                    bit = common & -common          # lowest offending column
                    j = bit.bit_length() - 1
                    A[c, j] = 0                     # highest row index loses
                    rows[c] &= ~bit
                    common &= ~bit
    return A


def construct_graph(M, N):
    if M < S or N < T:
        return np.ones((M, N), dtype=int)

    k = _block_size(M, N)

    # One base block per residue class of the column index.  Rotating the
    # offset set by the class index keeps the pooled difference multiset flat
    # when N forces more columns than Z_M has translates.
    A = np.zeros((M, N), dtype=int)
    for j in range(N):
        base = _offsets(M, k, (j // M) * (j // M + 1) // 2)
        for d in base:
            A[(j + d) % M, j] = 1

    return _repair(A)


def construct_graphs(M, N):
    return construct_graph(M, N)


def run_graph(M, N):
    return construct_graph(M, N)
# EVOLVE-BLOCK-END

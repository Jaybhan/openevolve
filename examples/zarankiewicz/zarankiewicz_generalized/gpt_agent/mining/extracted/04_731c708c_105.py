# MINED PROGRAM — rank 4 of 8
# id: 731c708c-9712-469e-ade5-0774570fc116
# generation 7, iteration_found 105, island 2, parent f72fe97d ("Full rewrite")
# metrics: exact_count=105/161  combined_score=0.7529  validity=1.0
# FAMILY: mainstream PG(3,2)/AG(4,2) character family again, but independently
# phrased (variable names "_algebraic"/"_affine"/"_tiny" rather than the
# champion's "_group_blocks"/"_boundary_blocks"/"_seven_families") and evolved
# on a DIFFERENT island (2, not 3) to generation 7 -- one generation deeper
# than the champion's lineage. Its "past M=15" answer is a THIRD distinct
# mechanism: true affine-plane lines AG(2,q) for the least prime q with
# q^2>=M, doubled, rather than hyperplane-cap-doubling or inversive planes.
# Nearly matches the champion (105 vs 110) via convergent, independently-
# evolved reasoning. Code below is VERBATIM from the checkpoint JSON.

# EVOLVE-BLOCK-START
"""
Zarankiewicz z(M,N;3,3) by *capacity-bounded incidence families*.

One principle runs through the whole file.  Call a column its 1-set (a
"block") on the row set [m].  Three rows form a K_{3,3} with three columns
exactly when three blocks all contain that row-triple.  So the only
invariant that matters is

        mult(T) = #{columns whose block contains the triple T}  <=  s-1 = 2 .

Every family below is a geometry whose *triple multiplicity is controlled by
linear algebra*, so no triple ever has to be inspected:

  * Characters of F_2^d (d<=4):  block_a = {x : <a,x> = 1}.  Three
    independent points force three independent linear conditions on a, whose
    solution set is a coset of a (d-3)-dimensional space: 2^(d-3) blocks.
    Hence d=4 admits ONE copy of each of the 15 blocks (mult exactly 2), and
    d=3 admits TWO copies of each of the 7 blocks (mult 1 -> 2).  Dependent
    triples (x+y+z=0, or a zero point) lie in NO character block, so they
    still carry their full capacity 2 and may be added as size-3 columns.
    That is why the algebraic packing for m<=15 comes in two layers.

  * Lines of AG(2,q) (needed as soon as m > 15, where d>=5 would give
    2^(d-3) >= 4 > 2 common columns and the character family becomes
    ILLEGAL).  Two points determine a unique line, so a collinear triple has
    multiplicity 1 and a non-collinear triple multiplicity 0.  Doubling every
    line therefore lands exactly on the cap 2, giving 2(q^2+q) columns of
    size ~q.  q is the least prime with q^2 >= m, and truncation to m points
    is just a restriction of the incidence structure.

  * Pairs.  A block of size 2 contains no triple at all: capacity-free
    padding, always legal, used to fill n beyond the geometry.

Because s = t = 3 the problem is self-dual, so both orientations are built
and the denser one is returned; likewise, when several families are legal for
the same m, the one with the denser n-column prefix is used.  Blocks are
always ordered by decreasing size, which is the optimal prefix of a valid
family.
"""

import numpy as np
from itertools import combinations


# ----------------------------------------------------------------- families

def _algebraic(m):
    """Character blocks of F_2^d truncated to m points, plus the dependent
    triples that the characters leave untouched.  Legal only for d <= 4."""
    d = max(2, (m - 1).bit_length())
    q = 1 << d
    points = (list(range(1, q)) + [0])[:m]
    copies = 2 if d <= 3 else 1          # 2^(3-d) rounded to the cap 2

    blocks = []
    for a in range(1, q):
        B = tuple(i for i, x in enumerate(points)
                  if bin(a & x).count('1') & 1)
        if len(B) >= 2:
            blocks.extend([B] * copies)

    for T in combinations(range(m), 3):
        x, y, z = (points[i] for i in T)
        if x == 0 or y == 0 or z == 0 or x ^ y ^ z == 0:
            blocks.extend((T, T))        # untouched capacity 2

    blocks.sort(key=lambda B: (-len(B), B))
    return blocks


def _prime_at_least(v):
    p = max(2, v)
    while True:
        if all(p % r for r in range(2, int(p ** 0.5) + 1)):
            return p
        p += 1


def _affine(m):
    """Every line of AG(2,q), taken twice: collinear triples reach exactly
    multiplicity 2, non-collinear ones 0.  Valid for every m."""
    q = _prime_at_least(int(m ** 0.5) if int(m ** 0.5) ** 2 >= m
                        else int(m ** 0.5) + 1)
    pts = [(x, y) for x in range(q) for y in range(q)][:m]

    blocks = []
    for a in range(q):
        for b in range(q):
            B = tuple(i for i, (x, y) in enumerate(pts)
                      if y == (a * x + b) % q)
            if len(B) >= 2:
                blocks.extend((B, B))
    for c in range(q):
        B = tuple(i for i, (x, y) in enumerate(pts) if x == c)
        if len(B) >= 2:
            blocks.extend((B, B))

    blocks.sort(key=lambda B: (-len(B), B))
    return blocks


def _tiny(m, n):
    """The same capacity accounting done by hand where the geometry is
    smaller than its own capacity: complements of a few points/edges use up
    part of every triple's allowance, and the residual allowance is spent on
    size-3 blocks."""
    if m == 3:
        return [(0, 1, 2)] * 2

    if m == 4:
        dense = n <= 6
        blocks = [tuple(range(4))] if dense else []
        return blocks + list(combinations(range(4), 3)) * (1 if dense else 2)

    if m == 5:
        r = max(0, min(5, (20 - n) // 3))
        removed = set(range(r))
        blocks = [tuple(i for i in range(5) if i != a) for a in removed]
        for T in combinations(range(5), 3):
            blocks += [T] * max(0, 2 - r + len(removed.intersection(T)))
        return blocks

    if m == 6:
        k = min(n, 9, max(0, (40 - n) // 3))
        omitted = [(a, b) for a in range(3) for b in range(3, 6)][:k]
        blocks = [tuple(x for x in range(6) if x not in e) for e in omitted]
        for T in combinations(range(6), 3):
            S = set(T)
            used = sum(a not in S and b not in S for a, b in omitted)
            blocks += [T] * max(0, 2 - used)
        return blocks

    # m == 7: one point, a perfect matching and three transversals of it are
    # removed; each complement then spends a controlled part of the capacity.
    holes = [(0,), (1, 2), (3, 4), (5, 6), (1, 3, 5), (1, 4, 6), (2, 3, 6)]
    blocks = [tuple(x for x in range(7) if x not in h) for h in holes]
    blocks.append((0, 1, 3, 6))
    base = [set(B) for B in blocks]
    for T in combinations(range(7), 3):
        S = set(T)
        blocks += [T] * max(0, 2 - sum(S.issubset(B) for B in base))
    return blocks


# ------------------------------------------------------------------ assembly

def _pad(blocks, m, n):
    """Size-2 blocks carry no triple, so they extend any legal family."""
    out = list(blocks)
    k = 0
    while len(out) < n and m >= 2:
        a = k % m
        b = (a + 1 + k // m) % m
        if a != b:
            out.append((a, b))
        k += 1
    while len(out) < n:
        out.append(())
    return out


def _families(m, n):
    fams = []
    if 3 <= m <= 7:
        fams.append(_tiny(m, n))
    if m <= 15:
        fams.append(_algebraic(m))
    if m >= 4:
        fams.append(_affine(m))
    if not fams:
        fams.append([])
    return fams


def _incidence(m, n):
    if m < 3 or n < 3:
        return np.ones((m, n), dtype=np.uint8)

    best, best_w = None, -1
    for fam in _families(m, n):
        cols = _pad(fam, m, n)[:n]
        w = sum(len(B) for B in cols)
        if w > best_w:
            best, best_w = cols, w

    A = np.zeros((m, n), dtype=np.uint8)
    for j, B in enumerate(best):
        if B:
            A[list(B), j] = 1
    return A


def construct_graph(M, N):
    A = _incidence(M, N)
    if min(M, N) <= 40:
        B = _incidence(N, M).T           # z(M,N;3,3) = z(N,M;3,3)
        if int(B.sum()) > int(A.sum()):
            A = B
    return A


def construct_graphs(M, N):
    return construct_graph(M, N)


def run_graph(M, N):
    return construct_graph(M, N)
# EVOLVE-BLOCK-END
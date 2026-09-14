# MINED PROGRAM — rank 2 of 8
# id: 3c34798c-ddae-460c-a3c8-92e848137865
# generation 5, iteration_found 147, island 3, parent 2803383e ("Full rewrite")
# metrics: exact_count=109/161  combined_score=0.7654  validity=1.0
# FAMILY: same mainstream PG(3,2)/AG(4,2) core as the champion (identical
# _boundary_blocks/_seven_families/F_2^4 code for M<=15), but for M>15 grafts in
# FAMILY #2's ingredient instead: doubled Miquelian inversive planes
# (3-(q^2+1,q+1,1) designs over F_{q^2}, multiplicity 2, restricted to the
# first M points). A near-sibling of the champion -- same parent generation,
# different answer to "how do you extend past M=15" -- that lost by one cell
# (misses (16,16), which the champion's affine-hyperplane-cap doubling gets
# right; verified by direct re-evaluation). Shows the two families splicing
# together. Code below is VERBATIM from the checkpoint JSON.

# EVOLVE-BLOCK-START
"""One principle: each column is a block of a partial 3-design in which
every row triple lies in at most two blocks (triple capacity 2).

Realizations, all derived from M and N:
  * M<=7: exact residual triple packings from complement-block seeds;
  * 8<=M<=15: hyperplane-complements of PG(3,2) (three points impose a
    linear system with at most 2 solutions, so codegree <= 2);
  * M>=16: doubled Miquelian inversive planes, i.e. 3-(q^2+1,q+1,1)
    designs over F_{q^2} taken with multiplicity 2, restricted to the
    first M points — restriction preserves triple codegree <= 2.
All seeds are canonically completed by spending residual triple
capacity on lex-first 4-blocks and triples, then degree-2 pads.
"""

import numpy as np
from itertools import combinations
from collections import defaultdict
from math import comb


def _fill(M, N, seed, use4):
    """Complete a partial 3-(M,*,2) packing canonically.

    If use4, first spend remaining triple capacity on lex-first 4-blocks
    (4 edges for 4 capacity, good when columns are scarce), then on
    repeated triples, then degree-2 pads.  Both regimes are generated and
    the caller keeps the closed-form better one — a trade-off, not search.
    """
    blocks = list(seed[:N])
    cap = defaultdict(lambda: 2)
    for b in blocks:
        for t in combinations(b, 3):
            cap[t] -= 1
            if cap[t] < 0:
                return None

    if use4 and len(blocks) < N and M <= 20:
        for q in combinations(range(M), 4):
            if len(blocks) >= N:
                break
            ts = list(combinations(q, 3))
            if all(cap[t] >= 1 for t in ts):
                blocks.append(q)
                for t in ts:
                    cap[t] -= 1

    if len(blocks) < N:
        for t in combinations(range(M), 3):
            if len(blocks) >= N:
                break
            c = cap[t]
            if c > 0:
                k = min(c, N - len(blocks))
                blocks.extend([t] * k)
                cap[t] -= k

    while len(blocks) < N:
        j = len(blocks)
        blocks.append((j % M, (j + 1) % M))

    A = np.zeros((M, N), dtype=int)
    for j, b in enumerate(blocks[:N]):
        A[list(b), j] = 1
    return A


def _best(M, N, families):
    best, be = None, -1
    for f in families:
        for use4 in (False, True):
            A = _fill(M, N, f, use4)
            if A is not None:
                e = int(A.sum())
                if e > be:
                    be, best = e, A
    if best is None:
        best = _fill(M, N, [], False)
    return best


def _boundary_blocks(M, N):
    if M == 3:
        return [tuple(range(3))] * 2

    if M == 4:
        return [tuple(range(4))] if N <= 5 else []

    if M == 5:
        k = max(0, min(5, (20 - N) // 3))
        return [tuple(x for x in range(5) if x != v) for v in range(k)]

    if N == 6:
        omitted = [(0,), (1,), (2, 3), (3, 4), (4, 5), (5, 2)]
    elif N == 7:
        omitted = [(0,)] + [(a, b) for a in (1, 2) for b in (3, 4, 5)]
    else:
        # Complements of the edges of K_{3,3}: any three vertices are
        # disjoint from at most two such edges.
        k = max(0, min(9, N, (40 - N) // 3))
        omitted = [(a, b) for a in range(3) for b in range(3, 6)][:k]
    return [tuple(x for x in range(6) if x not in e) for e in omitted]


def _seven_families():
    """Three derived seed families on 7 points.

    f1: nested omission code (6,5,5,5,4,4,4,4) — a matching plus its
        unmatched singleton, then complements of 3-sets meeting each
        omitted pair; every triple covered at most twice by inclusion.
    f2: complements of Fano lines, each twice — a non-line triple lies in
        exactly one line-complement, a line in none.
    f3: complements of the edges of P3 + 2K2 (edges 01,12,34,56): any
        4-set induces at most 2 of these edges, so any triple lies in at
        most 2 of the four 5-blocks; the rest of the capacity is left for
        canonical 4-block fill.
    """
    omitted = [
        (0,),
        (1, 2), (3, 4), (5, 6),
        (2, 3, 5), (1, 4, 5),
        (1, 3, 6), (2, 4, 6),
    ]
    f1 = [tuple(x for x in range(7) if x not in e) for e in omitted]

    comps = []
    for a in range(1, 8):
        comps.append(tuple(x - 1 for x in range(1, 8)
                           if (a & x).bit_count() & 1))
    f2 = [B for B in comps for _ in (0, 1)]

    pairs = [(0, 1), (1, 2), (3, 4), (5, 6)]
    f3 = [tuple(x for x in range(7) if x not in p) for p in pairs]

    return [f1, f2, f3]


def _mul4(a, b):
    a0, a1 = a & 1, a >> 1
    b0, b1 = b & 1, b >> 1
    c = a1 & b1
    return ((a0 & b0) ^ c) | (((a0 & b1) ^ (a1 & b0) ^ c) << 1)


def _gf4_inversive_blocks(M):
    """Circles of the inversive plane of order 4 (17 points),
    restricted to row 0 = infinity, rows 1..M-1 = finite points of
    F_16 = F_4[t]/(t^2 + t + w)."""
    def mul(x, y):
        a, b, c, e = x & 3, x >> 2, y & 3, y >> 2
        lo = _mul4(a, c) ^ _mul4(2, _mul4(b, e))
        hi = _mul4(a, e) ^ _mul4(b, c) ^ _mul4(b, e)
        return lo | (hi << 2)

    def norm(x):
        a, b = x & 3, x >> 2
        return _mul4(a, a) ^ _mul4(a, b) ^ _mul4(2, _mul4(b, b))

    lim = M - 1
    blocks = []
    for c in range(16):
        for r in range(1, 4):
            blocks.append(tuple(
                x + 1 for x in range(lim) if norm(x ^ c) == r
            ))
    directions = [1 | (t << 2) for t in range(4)] + [4]
    for d in directions:
        for level in range(4):
            blocks.append(tuple(
                [0] + [x + 1 for x in range(lim)
                       if (mul(d, x) >> 2) == level]
            ))
    return blocks


def _prime_inversive_blocks(M, N, q):
    """Circles of the Miquelian inversive plane of prime order q,
    F_{q^2} = F_q(sqrt(d)) for a nonresidue d, restricted to the
    first M points (row 0 = infinity)."""
    sq = {(x * x) % q for x in range(1, q)}
    d = next(x for x in range(2, q) if x not in sq)
    n2 = q * q
    lim = min(M - 1, n2)

    idx = np.arange(n2)
    a, b = idx % q, idx // q
    ar, br = a[:lim], b[:lim]

    da = (ar[:, None] - a[None, :]) % q
    db = (br[:, None] - b[None, :]) % q
    D = (da * da - d * db * db) % q

    entries = []
    for r in range(1, q):
        sizes = (D == r).sum(0)
        for c in range(n2):
            entries.append((int(sizes[c]), 0, r, c))

    vals = []
    for s in range(q):
        vals.append((br - s * ar) % q)
    vals.append(ar)
    for s in range(q + 1):
        counts = np.bincount(vals[s], minlength=q)
        for c in range(q):
            entries.append((int(counts[c]) + 1, 1, s, c))

    entries.sort(key=lambda e: -e[0])
    K = (N + 1) // 2 + 1
    blocks = []
    for size, kind, x, c in entries[:K]:
        if kind == 0:
            blk = tuple((np.nonzero(D[:, c] == x)[0] + 1).tolist())
        else:
            blk = tuple([0] + (np.nonzero(vals[x] == c)[0] + 1).tolist())
        blocks.append(blk)
        blocks.append(blk)
    return blocks


def _core(M, N):
    """Construct with M <= N (canonical orientation)."""
    if M <= 6:
        return _best(M, N, [_boundary_blocks(M, N)])

    if M == 7:
        return _best(7, N, _seven_families())

    if M <= 15:
        # Rows are nonzero points of F_2^4; column a is {x : <a,x> = 1}.
        # Three distinct points give either an inconsistent system or a
        # rank-3 system whose solution set is a coset of a 1-dimensional
        # space, so every row triple lies in at most 2 columns.
        blocks = []
        for a in range(1, 16):
            block = tuple(x - 1 for x in range(1, M + 1)
                          if (a & x).bit_count() & 1)
            if len(block) >= 3:
                blocks.append(block)
        blocks.sort(key=lambda b: (-len(b), b))
        return _best(M, N, [blocks])

    # Doubled inversive plane of the smallest adequate prime power q:
    # a 3-(q^2+1, q+1, 1) design taken with multiplicity 2 has every
    # triple codegree exactly 2, and any restriction inherits that.
    if M <= 17:
        blocks = _gf4_inversive_blocks(M)
    else:
        q = next((p for p in (5, 7, 11, 13) if p * p + 1 >= M), 13)
        blocks = _prime_inversive_blocks(M, N, q)

    blocks = [b for b in blocks if len(b) >= 3]
    blocks.sort(key=lambda b: -len(b))
    doubled = []
    for b in blocks:
        doubled.append(b)
        if len(doubled) < 2 or doubled[-2] != b:
            doubled.append(b)
    return _best(M, N, [doubled])


def construct_graph(M, N):
    # z(M,N;3,3) is self-dual: always build with the smaller side as rows.
    if M > N:
        return np.ascontiguousarray(_core(N, M).T)
    return _core(M, N)


def construct_graphs(M, N):
    return construct_graph(M, N)


def run_graph(M, N):
    return construct_graph(M, N)
# EVOLVE-BLOCK-END
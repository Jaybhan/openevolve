# MINED PROGRAM — rank 1 of 8 (by exact_count on the 161-cell scored set)
# id: cd6af8a9-1bd2-416c-a4fe-6f220415411e  (the run's declared CHAMPION / best_program.py)
# generation 6, iteration_found 134, island 3, parent 63b17796 ("Full rewrite")
# metrics (as recorded in checkpoint_150, current evaluator.py):
#   exact_count=110/161  combined_score=0.7758  validity=1.0  peak_instance_ms=0.716
# FAMILY: PG(3,2)/AG(4,2) hyperplane-character incidence (the dominant "mainstream"
# lineage). Dispatches on M: M<=6 hand-built omission designs, M==7 three
# Fano-derived families, 8<=M<=15 the F_2^4 "<a,x>=1" rank argument, M>15 (only
# M=16 is ever scored) affine-hyperplane doubling over a cap.
# FORENSIC NOTE (see mining_report.md): removing the M<=7 hardcoded tables and
# keeping everything else drops this exact same file from 110/161 to 74/161
# exact (verified by re-running evaluator.py on an ablated copy) -- i.e. ~33%
# of this program's exactness comes from three hand-specified small designs
# (M=5,6,7), not from the general F_2^4 rule. Code below is VERBATIM from the
# checkpoint JSON -- nothing edited.

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from math import comb


def _fill(M, N, seed, use4):
    """Complete a partial 3-(M,*,2) packing canonically.

    If use4, first spend remaining triple capacity on lex-first 4-blocks
    (4 edges for 4 capacity, good when columns are scarce), then on
    repeated triples (3 edges for 1 capacity, good when capacity is
    scarce), then on degree-2 pads.  Both regimes are generated and the
    caller keeps whichever yields more edges — a closed trade-off, not a
    search.
    """
    blocks = list(seed[:N])
    cap = {T: 2 for T in combinations(range(M), 3)}
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


def construct_graph(M, N):
    # z(M,N;3,3) is self-dual: always build with the smaller side as rows.
    if M > N:
        return np.ascontiguousarray(construct_graph(N, M).T)

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

    # Affine hyperplanes of F_2^4 whose normals form the cap a_3 = 1;
    # both sides of each hyperplane are usable since the normals are a cap.
    blocks = []
    for a in range(8, 16):
        for side in range(2):
            block = tuple(x for x in range(M)
                          if ((a & x).bit_count() & 1) == side)
            if len(block) >= 3:
                blocks.append(block)
    blocks.sort(key=lambda b: (-len(b), b))
    return _best(M, N, [blocks])


def construct_graphs(M, N):
    return construct_graph(M, N)


def run_graph(M, N):
    return construct_graph(M, N)
# EVOLVE-BLOCK-END
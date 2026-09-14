# MINED PROGRAM — rank 3 of 8
# id: fef84e2f-bb05-45ee-a9fd-c2797dcaf764
# generation 6, iteration_found 139, island 3, parent 63b17796 ("Full rewrite")
# metrics: exact_count=108/161  combined_score=0.7703  validity=1.0
# FAMILY: same mainstream core as the champion (same sibling parent 63b17796),
# but _complete() adds two rounds of degree-ordered "saturation sweeps" that
# greedily grow columns whenever the triple-capacity certificate still allows
# it. Net effect is WORSE overall (108 vs 110) because it regresses on four
# M=7 cells -- BUT it is the ONLY program anywhere in the run's 142-evaluation
# instance_log.jsonl history that ever solves (8,8) and (8,9) exactly
# (confirmed by timestamp match, to within 1ms, between an instance_log
# is_exact=true record and this program's own saved timestamp). The champion
# does not solve either cell. See mining_report.md / ideas_for_generalization.md
# for this "key finding". Code below is VERBATIM from the checkpoint JSON.

# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from math import comb


def _complete(M, N, blocks):
    """Complete a partial 3-(M,*,2) packing, then saturate columns.

    Every row triple may lie in at most 2 columns (t-1 = 2 for t = 3).
    We track exact residual capacity per triple; leftover columns are
    filled with capacity-positive triples, and finally every column is
    enlarged row by row whenever ALL newly created triples still have
    positive capacity.  The capacity certificate makes each enlargement
    provably safe, so K_{3,3}-freeness never has to be re-verified.
    """
    capacity = {T: 2 for T in combinations(range(M), 3)}

    cols = [set(b) for b in blocks[:N]]
    for c in cols:
        for triple in combinations(sorted(c), 3):
            capacity[triple] -= 1

    # Fill remaining columns with residual triples (canonical order).
    for triple in combinations(range(M), 3):
        while len(cols) < N and capacity[triple] > 0:
            capacity[triple] -= 1
            cols.append(set(triple))

    while len(cols) < N:
        j = len(cols)
        cols.append({j % M, (j + 1) % M})

    # Saturation sweeps: grow columns while capacity certifies safety.
    deg = [0] * M
    for c in cols:
        for v in c:
            deg[v] += 1

    for _sweep in range(2):
        changed = False
        for c in cols:
            order = sorted((v for v in range(M) if v not in c),
                           key=lambda v: (deg[v], v))
            for v in order:
                pairs = list(combinations(sorted(c), 2))
                new_triples = [tuple(sorted(p + (v,))) for p in pairs]
                if all(capacity.get(t, 0) > 0 for t in new_triples):
                    for t in new_triples:
                        capacity[t] -= 1
                    c.add(v)
                    deg[v] += 1
                    changed = True
        if not changed:
            break

    A = np.zeros((M, N), dtype=int)
    for j, c in enumerate(cols[:N]):
        A[sorted(c), j] = 1
    return A


def _boundary(M, N):
    if M == 3:
        blocks = [tuple(range(3))] * 2

    elif M == 4:
        blocks = [tuple(range(4))] if N <= 5 else []

    elif M == 5:
        k = max(0, min(5, (20 - N) // 3))
        blocks = [tuple(x for x in range(5) if x != v) for v in range(k)]

    elif N == 6:
        omitted = [(0,), (1,), (2, 3), (3, 4), (4, 5), (5, 2)]
        blocks = [tuple(x for x in range(6) if x not in e) for e in omitted]

    elif N == 7:
        omitted = [(0,)] + [(a, b) for a in (1, 2) for b in (3, 4, 5)]
        blocks = [tuple(x for x in range(6) if x not in e) for e in omitted]

    else:
        k = max(0, min(9, N, (40 - N) // 3))
        omitted = [(a, b) for a in range(3) for b in range(3, 6)]
        blocks = [tuple(x for x in range(6) if x not in e)
                  for e in omitted[:k]]

    return _complete(M, N, blocks)


def _family_edges(sizes, N, total_capacity):
    """Edges achieved by given block sizes followed by triple/pair fill."""
    k = min(N, len(sizes))
    used = sum(comb(s, 3) for s in sizes[:k])
    left = total_capacity - used
    triples = min(max(N - k, 0), left)
    pairs = max(N - k - triples, 0)
    return sum(sizes[:k]) + 3 * triples + 2 * pairs


def _seven(N):
    """Two derived families on 7 points; pick by closed-form edge count.

    Family 1: nested omission code (one 6-block, three 5-blocks, four
    4-blocks) — every triple covered at most twice by inclusion pattern.
    Family 2: complements of Fano lines, each taken twice.  A non-line
    triple lies in exactly one line-complement, so two copies cover it
    exactly twice; a line lies in none.
    """
    cap = 2 * comb(7, 3)

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

    e1 = _family_edges([len(b) for b in f1], N, cap)
    e2 = _family_edges([len(b) for b in f2], N, cap)
    blocks = list(f1) if e1 >= e2 else list(f2)
    return _complete(7, N, blocks[:N])


def construct_graph(M, N):
    # z(M,N;3,3) is self-dual: always build with the smaller side as rows.
    if M > N:
        return np.ascontiguousarray(construct_graph(N, M).T)

    if M <= 6:
        return _boundary(M, N)

    if M == 7:
        return _seven(N)

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
        return _complete(M, N, blocks[:N])

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
    return _complete(M, N, blocks[:N])


def construct_graphs(M, N):
    return construct_graph(M, N)


def run_graph(M, N):
    return construct_graph(M, N)
# EVOLVE-BLOCK-END
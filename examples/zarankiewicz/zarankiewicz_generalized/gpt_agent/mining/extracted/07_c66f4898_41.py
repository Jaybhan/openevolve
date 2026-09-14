# MINED PROGRAM — rank 7 of 8
# id: c66f4898-8f59-4c19-aa52-a5e2cb9c7ee5
# generation 2, iteration_found 23, island 3, parent d61baa70 ("Full rewrite")
# metrics: exact_count=41/161  combined_score=0.3951  validity=1.0
# FAMILY: Brown's graph -- points of F_3^3, two rows adjacent to a column
# (a "sphere" of squared-radius 1 about a centre) when their squared distance
# to that centre is 1; two distinct spheres meet in <=2 points, so tripling
# rows never shares 3 common columns, and doubling every sphere reaches
# codegree exactly 2. Per config_phase_3.yaml's own system-message audit of
# this pipeline's history, this construction "has appeared ONCE in this
# pipeline's entire history" -- this is that one occurrence. Code below is
# VERBATIM from the checkpoint JSON.

# EVOLVE-BLOCK-START
"""A doubled finite-field sphere incidence construction over F_3.

Rows and columns are points of F_3^3, incident when their squared distance
is one.  Two distinct unit spheres in F_3^3 intersect in at most two
points, so three rows lie on at most one sphere.  Doubling every sphere
therefore gives triple codegree at most two.
"""

import numpy as np


def _small(M, N):
    A = np.zeros((M, N), dtype=int)

    if M == 3:
        k = min(2, N)
        A[:, :k] = 1
        start = k
    else:
        start = 0
        if N <= 5:
            A[:, 0] = 1
            start = 1
            copies = 1
        else:
            copies = 2

        j = start
        for _ in range(copies):
            for missing in range(4):
                if j == N:
                    break
                A[:, j] = 1
                A[missing, j] = 0
                j += 1
        start = j

    for j in range(start, N):
        A[j % M, j] = 1
        A[(j + 1) % M, j] = 1
    return A


def _point(k):
    """Algebraically scrambled enumeration of F_3^3."""
    x = k % 3
    y = (k // 3 + x * x) % 3
    z = (k // 9 + x * y) % 3
    return x, y, z


def construct_graph(M, N):
    if M < 3 or N < 3:
        return np.ones((M, N), dtype=int)
    if M <= 4:
        return _small(M, N)

    rows = [_point(i) for i in range(M)]
    spheres = []

    for c in range(27):
        center = _point(c)
        mask = []
        for i, p in enumerate(rows):
            d0 = (p[0] - center[0]) % 3
            d1 = (p[1] - center[1]) % 3
            d2 = (p[2] - center[2]) % 3
            if (d0 * d0 + d1 * d1 + d2 * d2) % 3 == 1:
                mask.append(i)

        # Distinct spheres have triple codegree at most one, hence each
        # incidence column may be used twice.
        spheres.append((-len(mask), c, 0, mask))
        spheres.append((-len(mask), c, 1, mask))

    spheres.sort()
    A = np.zeros((M, N), dtype=int)

    for j, (_, center, copy, mask) in enumerate(spheres[:N]):
        A[mask, j] = 1

        # Columns of degree below two can be completed freely.
        need = 2 - len(mask)
        r = (center + copy) % M
        while need > 0:
            if not A[r, j]:
                A[r, j] = 1
                need -= 1
            r = (r + 1) % M

    return A


def construct_graphs(M, N):
    return construct_graph(M, N)


def run_graph(M, N):
    return construct_graph(M, N)
# EVOLVE-BLOCK-END
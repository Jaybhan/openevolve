# MINED PROGRAM — rank 5 of 8
# id: b126ca4f-9b2f-42fa-857c-7ee34a06b0dc
# generation 4, iteration_found 107, island 3, parent 62cc3235 ("Full rewrite")
# metrics: exact_count=78/161  combined_score=0.6059  validity=1.0
# FAMILY: Projective norm graph (Kollar-Ronyai-Szabo NG(q,3): vertices
# (x,a) in F_{q^2} x F_q^*, adjacency Norm(x+y)=ab; any 3 vertices share <=2
# common neighbours by a rank argument on the norm equations) COMBINED with
# doubled Miquelian/inversive planes (3-(q^2+1,q+1,1) designs) and hand-built
# small-M packings, all unified under one "triple multiplicity <= 2" framing.
# This is the best specimen (highest exact_count) of the whole norm-graph
# lineage; the family as a whole never exceeded combined_score ~0.61 anywhere
# in the run (many near-duplicate attempts, e.g. 8ed6b27a, d84f2b1e, 12b9a176,
# e034b8ff, d2476bee cap out in the same 0.50-0.61 band -- see mining_report.md).
# Code below is VERBATIM from the checkpoint JSON.

# EVOLVE-BLOCK-START
"""One principle: incidence structures in which every triple of rows lies in
at most two common blocks.

Three interchangeable realizations, all derived from a prime power q:

  * small M (<=6): exact residual triple packings — every triple of the M
    rows may serve as a column at most twice, complement blocks consume
    triple capacity in a computable way;
  * projective norm graphs NG(q,3) on F_{q^2} x F_q^*: any 3 vertices have
    at most 2 common neighbours, and any induced restriction inherits this;
  * doubled inversive planes (3-(q^2+1, q+1, 1) designs): duplicating every
    circle gives triple multiplicity exactly 2, again closed under
    restriction.

Since z(M,N;3,3) is self-dual, each structure is applied in both
orientations and the densest valid restriction is returned.  Columns of
degree <= 2 never contribute to a triple, so short columns are completed
to degree two for free.
"""

import numpy as np
from itertools import combinations


def _m4(a, b):
    a0, a1, b0, b1 = a & 1, a >> 1, b & 1, b >> 1
    c = a1 & b1
    return ((a0 & b0) ^ c) | (((a0 & b1) ^ (a1 & b0) ^ c) << 1)


def _nonres(q):
    sq = {(x * x) % q for x in range(1, q)}
    return next(d for d in range(2, q) if d not in sq)


def _field(q):
    """Arithmetic of F_{q^2} over F_q, encoded as integers in [0, q^2)."""
    if q == 4:
        add = lambda x, y: x ^ y
        sub = add

        def mul(x, y):
            a, b, c, d = x & 3, x >> 2, y & 3, y >> 2
            lo = _m4(a, c) ^ _m4(2, _m4(b, d))
            hi = _m4(a, d) ^ _m4(b, c) ^ _m4(b, d)
            return lo | (hi << 2)

        def norm(x):
            a, b = x & 3, x >> 2
            return _m4(a, a) ^ _m4(a, b) ^ _m4(2, _m4(b, b))

        trace = lambda x: x >> 2
        enc = lambda a, b: a | (b << 2)
        smul = _m4
    else:
        d = _nonres(q)
        add = lambda x, y: (x % q + y % q) % q + q * ((x // q + y // q) % q)
        sub = lambda x, y: (x % q - y % q) % q + q * ((x // q - y // q) % q)

        def mul(x, y):
            a, b, c, e = x % q, x // q, y % q, y // q
            return (a * c + d * b * e) % q + q * ((a * e + b * c) % q)

        norm = lambda x: ((x % q) ** 2 - d * (x // q) ** 2) % q
        trace = lambda x: (2 * (x % q)) % q
        enc = lambda a, b: a + q * b
        smul = lambda a, b: (a * b) % q

    return add, sub, mul, norm, trace, enc, smul


# ---------- small-M exact triple packings ----------

def _small(M, N):
    A = np.zeros((M, N), dtype=np.uint8)

    if M == 3:
        j = min(2, N)
        A[:, :j] = 1
    elif M == 4:
        j = 0
        copies = 2
        if N <= 5:
            A[:, 0] = 1
            j, copies = 1, 1
        for _ in range(copies):
            for missing in range(4):
                if j == N:
                    return A
                A[:, j] = 1
                A[missing, j] = 0
                j += 1
    else:
        # A 4-block on 5 points consumes 4 triple incidences; the count
        # of 4-blocks is the largest k with N + 3k <= 20.
        fours = min(5, max(0, (20 - N) // 3))
        omitted = set(range(fours))
        j = 0
        for missing in omitted:
            A[:, j] = 1
            A[missing, j] = 0
            j += 1
        for T in combinations(range(5), 3):
            for _ in range(2 - len(omitted.difference(T))):
                if j == N:
                    return A
                A[list(T), j] = 1
                j += 1

    while j < N:
        A[j % M, j] = A[(j + 1) % M, j] = 1
        j += 1
    return A


def _six(N):
    """Complements of a triangle-free graph, then residual triples."""
    A = np.zeros((6, N), dtype=np.uint8)

    singletons = max(0, 8 - N)
    omissions = [(i,) for i in range(singletons)]

    if singletons:
        points = list(range(singletons, 6))
        cut = len(points) // 2
        omissions += [
            (a, b) for a in points[:cut] for b in points[cut:]
        ][:N - singletons]
    else:
        count = min(9, N, (40 - N) // 3)
        omissions = [(a, b) for a in range(3) for b in range(3, 6)][:count]

    j = 0
    for S in omissions:
        if j == N:
            return A
        A[:, j] = 1
        A[list(S), j] = 0
        j += 1

    chosen = [set(S) for S in omissions]
    universe = set(range(6))
    for U0 in combinations(range(6), 3):
        U = set(U0)
        T = sorted(universe - U)
        capacity = 2 - sum(S <= U for S in chosen)
        for _ in range(capacity):
            if j == N:
                return A
            A[T, j] = 1
            j += 1

    while j < N:
        A[j % 6, j] = A[(j + 1) % 6, j] = 1
        j += 1
    return A


# ---------- projective norm graph restrictions ----------

def _vertices(q):
    elements = sorted(
        range(q * q),
        key=lambda z: ((z % q != 0) + (z // q != 0), z // q, z % q),
    )
    first, rest = [], []
    for z in elements:
        preferred = 1 if z // q == 0 else 2
        first.append((z, preferred))
        rest.extend((z, x) for x in range(1, q) if x != preferred)
    return first + rest


def _complete(A, j, mask, seed, R):
    need = 2 - len(mask)
    r = seed % R
    while need > 0:
        if not A[r, j]:
            A[r, j] = 1
            need -= 1
        r = (r + 1) % R


def _norm_restriction(R, C, q):
    vertices = _vertices(q)
    if R > len(vertices):
        return None
    add, _, _, norm, _, _, smul = _field(q)

    rows = vertices[:R]
    columns = []
    for index, (b, beta) in enumerate(vertices):
        mask = [
            r for r, (a, alpha) in enumerate(rows)
            if norm(add(a, b)) == smul(alpha, beta)
        ]
        columns.append((-len(mask), index, mask))
    columns.sort()

    A = np.zeros((R, C), dtype=np.uint8)
    for j in range(C):
        if j < len(columns):
            _, seed, mask = columns[j]
            A[mask, j] = 1
        else:
            seed, mask = j, []
        _complete(A, j, mask, seed, R)
    return A


# ---------- doubled inversive plane restrictions ----------

def _circle_restriction(R, C, q):
    if R > q * q + 1:
        return None
    add, sub, mul, norm, trace, enc, _ = _field(q)
    finite = q * q
    infinity = finite

    order = [infinity] + [enc(x, 0) for x in range(q)]
    seen = set(order)
    order += [x for x in range(finite) if x not in seen]
    rows = order[:R]

    blocks = []
    for center in range(finite):
        for radius in range(1, q):
            blocks.append({
                x for x in range(finite)
                if norm(sub(x, center)) == radius
            })
    directions = [enc(1, t) for t in range(q)] + [enc(0, 1)]
    for direction in directions:
        for level in range(q):
            block = {infinity}
            block.update(
                x for x in range(finite)
                if trace(mul(direction, x)) == level
            )
            blocks.append(block)

    columns = []
    for index, block in enumerate(blocks):
        mask = [i for i, x in enumerate(rows) if x in block]
        columns.append((-len(mask), index, mask))
        columns.append((-len(mask), index, mask))
    columns.sort()

    A = np.zeros((R, C), dtype=np.uint8)
    for j in range(C):
        if j < len(columns):
            _, seed, mask = columns[j]
            A[mask, j] = 1
            mask = list(mask)
        else:
            seed, mask = j, []
        _complete(A, j, mask, seed, R)
    return A


def construct_graph(M, N):
    if M < 3 or N < 3:
        return np.ones((M, N), dtype=np.uint8)
    if M <= 5:
        return _small(M, N)
    if M == 6:
        return _six(N)

    best = None
    best_edges = -1

    def consider(A, transpose=False):
        nonlocal best, best_edges
        if A is None:
            return
        if transpose:
            A = A.T.copy()
        edges = int(A.sum())
        if edges > best_edges:
            best, best_edges = A, edges

    norm_qs = [3, 4, 5]
    if min(M, N) > 100:
        norm_qs.append(7)
    circle_qs = [3, 4]
    if min(M, N) > 17:
        circle_qs.append(5)
    if min(M, N) > 26:
        circle_qs.append(7)

    for q in norm_qs:
        consider(_norm_restriction(M, N, q))
        consider(_norm_restriction(N, M, q), transpose=True)
    for q in circle_qs:
        consider(_circle_restriction(M, N, q))
        consider(_circle_restriction(N, M, q), transpose=True)

    return best


def construct_graphs(M, N):
    return construct_graph(M, N)


def run_graph(M, N):
    return construct_graph(M, N)
# EVOLVE-BLOCK-END
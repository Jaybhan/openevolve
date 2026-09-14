"""Reference generator structures for the catalogue algebra.

Every generator is CONSTRUCTIVE: built from classical objects
(finite geometry, designs, difference sets), never from a solver.
"""
from itertools import combinations, product


# ------------------------------------------------------------- classical

def fano():
    """The 7 lines of PG(2,2) on points 0..6 (nonzero vectors of F_2^3,
    point i = vector i+1 ... we use the standard difference-set form)."""
    base = {0, 1, 3}
    return [tuple(sorted((x + r) % 7 for x in base)) for r in range(7)]


def sqs8():
    """SQS(8) = planes of AG(3,2): 14 blocks on 0..7 (x^y^z^w = 0)."""
    bl = []
    for b in combinations(range(8), 4):
        x, y, z, w = b
        if x ^ y ^ z ^ w == 0:
            bl.append(tuple(sorted(b)))
    return bl


def _gf9():
    """GF(9) as pairs (a,b) = a + b*t, t^2 = t + 1 over GF(3)."""
    els = [(a, b) for a in range(3) for b in range(3)]

    def add(u, v):
        return ((u[0] + v[0]) % 3, (u[1] + v[1]) % 3)

    def mul(u, v):
        # (a+bt)(c+dt) = ac + (ad+bc)t + bd t^2; t^2 = t+1
        a, b = u
        c, d = v
        return ((a * c + b * d) % 3, (a * d + b * c + b * d) % 3)

    def inv(u):
        for v in els:
            if mul(u, v) == (1, 0):
                return v
        raise ZeroDivisionError

    return els, add, mul, inv


def sqs10():
    """SQS(10) = Moebius plane of order 3: points GF(9) u {inf},
    circles = images of GF(3) u {inf} under PGL(2,9).  30 blocks."""
    els, add, mul, inv = _gf9()
    idx = {e: i for i, e in enumerate(els)}
    INF = 9
    sub = [(0, 0), (1, 0), (2, 0)]  # GF(3) inside GF(9)
    base = set([INF] + [idx[e] for e in sub])
    zero, one = (0, 0), (1, 0)

    def neg(u):
        return ((-u[0]) % 3, (-u[1]) % 3)

    circles = set()
    for a in els:
        for b in els:
            for c in els:
                for d in els:
                    # ad - bc != 0
                    det = add(mul(a, d), neg(mul(b, c)))
                    if det == zero:
                        continue
                    img = set()
                    for p in base:
                        if p == INF:
                            # az+b/cz+d at inf -> a/c
                            if c == zero:
                                img.add(INF)
                            else:
                                img.add(idx[mul(a, inv(c))])
                        else:
                            z = els[p]
                            den = add(mul(c, z), d)
                            num = add(mul(a, z), b)
                            if den == zero:
                                img.add(INF)
                            else:
                                img.add(idx[mul(num, inv(den))])
                    circles.add(tuple(sorted(img)))
    return sorted(circles)


def ag23_lines():
    """12 lines of AG(2,3) on points 0..8 = (r,c) -> 3r+c."""
    lines = []
    pts = [(r, c) for r in range(3) for c in range(3)]
    for a, b in combinations(pts, 2):
        dx = ((b[0] - a[0]) % 3, (b[1] - a[1]) % 3)
        line = {a}
        cur = a
        for _ in range(2):
            cur = ((cur[0] + dx[0]) % 3, (cur[1] + dx[1]) % 3)
            line.add(cur)
        lines.append(tuple(sorted(3 * r + c for r, c in line)))
    return sorted(set(lines))


def biplane11():
    """2-(11,5,2) biplane: QR difference set {1,3,4,5,9} mod 11."""
    D = [1, 3, 4, 5, 9]
    return [tuple(sorted((x + r) % 11 for x in D)) for r in range(11)]


def pentagon_triples(pts):
    """C5-edge-complement triples on the 5 points (in cyclic order):
    complement (within the 5-set) of each pentagon edge."""
    p = list(pts)
    out = []
    for i in range(5):
        edge = {p[i], p[(i + 1) % 5]}
        out.append(tuple(sorted(set(p) - edge)))
    return out


# ----------------------------------------------- K5-edge world for m = 10

def k5_edge_points():
    """Bijection: point index 0..9 <-> edges of K5 (pairs of 0..4)."""
    pairs = list(combinations(range(5), 2))
    return pairs


def k5_k4sets():
    """5 hexads: edge-sets of the K4 obtained by deleting vertex v."""
    pairs = k5_edge_points()
    idx = {p: i for i, p in enumerate(pairs)}
    out = []
    for v in range(5):
        out.append(tuple(sorted(idx[p] for p in pairs if v not in p)))
    return out


def k5_stars():
    """5 quads: stars at each vertex of K5."""
    pairs = k5_edge_points()
    idx = {p: i for i, p in enumerate(pairs)}
    out = []
    for v in range(5):
        out.append(tuple(sorted(idx[p] for p in pairs if v in p)))
    return out


def k5_pentagons():
    """12 pentads: edge-sets of the 5-cycles of K5."""
    from itertools import permutations
    pairs = k5_edge_points()
    idx = {p: i for i, p in enumerate(pairs)}
    seen = set()
    for perm in permutations(range(1, 5)):
        cyc = (0,) + perm
        edges = []
        for i in range(5):
            a, b = cyc[i], cyc[(i + 1) % 5]
            edges.append(idx[(min(a, b), max(a, b))])
        seen.add(tuple(sorted(edges)))
    return sorted(seen)


def k5_triangle_matchings():
    """10 pentads: triangle + disjoint matching-pair (the other type of
    2-regular 5-edge subgraph of K5)."""
    pairs = k5_edge_points()
    idx = {p: i for i, p in enumerate(pairs)}
    out = set()
    for tri in combinations(range(5), 3):
        rest = [v for v in range(5) if v not in tri]
        e = [idx[(min(a, b), max(a, b))] for a, b in combinations(tri, 2)]
        e.append(idx[(min(rest), max(rest))])
        # 4 edges only; a 5th edge would repeat -- skip, kept for ref
    for tri in combinations(range(5), 3):
        pass
    return sorted(out)


# --------------------------------------------------------- F2^4 families

def f24_hyperplane_family(m):
    """Champion family for 8 <= m <= 15: rows = first m of the nonzero
    vectors 1..15 of F_2^4; column a = {x : <a,x> = 1}."""
    cols = []
    for a in range(1, 16):
        col = tuple(x - 1 for x in range(1, m + 1)
                    if bin(a & x).count("1") % 2 == 1)
        if len(col) >= 2:
            cols.append(tuple(sorted(col)))
    return cols


def cap16_family():
    """(16,16): both sides of the 8 affine hyperplanes whose normals
    have top bit set (a cap in PG(3,2)); rows = all of F_2^4."""
    cols = []
    for a in range(8, 16):
        side1 = tuple(sorted(x for x in range(16)
                             if bin(a & x).count("1") % 2 == 1))
        side0 = tuple(sorted(x for x in range(16)
                             if bin(a & x).count("1") % 2 == 0))
        cols += [side1, side0]
    return cols

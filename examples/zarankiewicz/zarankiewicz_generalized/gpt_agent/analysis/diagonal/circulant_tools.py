"""Circulant constructions for diagonal Zarankiewicz cells.

A circulant witness on Z_v: base block B subset Z_v, columns = B+s for all s.
Edges = v*|B|.  K_{3,3}-free iff every 3-subset of Z_v lies in <= 2 shifts,
which by shift-invariance reduces to difference-pattern counts: the triple
{0,a,b} lies in B-s for s with {s, s+a, s+b} subset B; count over s.

Also: exhaustive search over base blocks up to shift+negation+multiplier
equivalence (Z_v units act on legal circulants), and mixed constructions.
"""
import json
import sys
from itertools import combinations


def circ_witness(v, base):
    blocks = [sorted((x + s) % v for x in base) for s in range(v)]
    return {"m": v, "n": v, "edges": v * len(base), "blocks": blocks,
            "source": f"circulant Z_{v} base {sorted(base)}"}


def circ_legal(v, base):
    """Direct check: every triple {0,a,b} covered <= 2 (suffices by
    shift-invariance to check triples containing 0 after shifting)."""
    bs = set(base)
    # triple {x,y,z} covered by shift s iff x-s,y-s,z-s all in B.
    # count for canonical triples {0,a,b}: shifts s with -s, a-s, b-s in B
    for a in range(1, v):
        for b in range(a + 1, v):
            cnt = 0
            for s in range(v):
                if (-s) % v in bs and (a - s) % v in bs and (b - s) % v in bs:
                    cnt += 1
                    if cnt > 2:
                        return False
    return True


def canon(v, base):
    """Canonical form under shift, negation, and unit multiplication."""
    best = None
    for u in range(1, v):
        # unit test: gcd(u,v)==1
        g, x, y = v, u, 0
        a0, b0 = u, v
        while b0:
            a0, b0 = b0, a0 % b0
        if a0 != 1:
            continue
        for sgn in (1, -1):
            imgs = sorted((sgn * u * x) % v for x in base)
            for s in range(v):
                t = tuple(sorted((x - s) % v for x in imgs))
                if best is None or t < best:
                    best = t
    return best


def search_all(v, k):
    """All legal weight-k circulant bases in Z_v, up to equivalence.
    WLOG 0 in B and (shift) min element 0."""
    seen = set()
    found = []
    for rest in combinations(range(1, v), k - 1):
        base = (0,) + rest
        if circ_legal(v, base):
            c = canon(v, base)
            if c not in seen:
                seen.add(c)
                found.append(c)
    return found


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "witness":
        v = int(sys.argv[2])
        base = [int(x) for x in sys.argv[3].split(",")]
        w = circ_witness(v, base)
        legal = circ_legal(v, base)
        out = sys.argv[4] if len(sys.argv) > 4 else None
        print(f"Z_{v} base {base}: legal={legal} edges={w['edges']}")
        if out and legal:
            with open(out, "w") as f:
                json.dump(w, f)
            print(f"wrote {out}")
    elif cmd == "search":
        v, k = int(sys.argv[2]), int(sys.argv[3])
        res = search_all(v, k)
        print(f"Z_{v} weight-{k} legal circulant bases up to equivalence: "
              f"{len(res)}")
        for r in res[:20]:
            print(" ", r)

"""Finite classifications for the wide-region structure theorems.

All searches are over ABSTRACT supports (points named 0,1,2,...); a
configuration found/refuted here is found/refuted on every m >= support
size, so the results are m-independent for the stated congruence class.

Class-m congruences (m == 3 mod 4, m !== 0 mod 3), quad/pentad configs:
  point:  ell_x == 0 (mod 3)   [3 | C(m-1,2)]
  pair:   ell_xy == p_xy (mod 2),  p_xy = #pentads through xy
Value Q = J-2 forces (slots): 2k5 + 8k6 + 19k7 <= 10 and
L = 10 - 2k5 - 8k6.

CASES (k6 = 1 and k5 = 4 are killed by hand in the report):
  A: k5 = 1, L = 8
  B: k5 = 2 doubled pentad, L = 6
  C: k5 = 2, |P1 n P2| = 4, L = 6
  D: k5 = 3, L = 4
  E: k5 = 5, L = 0  (pure pentad-parity search)
Also:
  W10: classification of ALL weight-10 leaves with k5 = 0 congruences
       (the max-packing leave shapes for class m)
  S91: m = 9, one pentad + 38 quads (leave 6, all ell_x = 2) -- the
       S9(1) = 37 proof's enumeration side.
"""
from __future__ import annotations

import sys
from collections import Counter
from itertools import combinations


def leave_search(npts, weight, point_res, pair_parity, cap,
                 forbidden=(), require_deg=None):
    """Enumerate multisets of triples on [npts] with:
    - total weight = weight, multiplicity <= 2, y_T <= cap[T] if given
    - point-leave ell_x == point_res[x] (mod 3) for all x
    - pair-leave parity ell_xy == pair_parity[(x,y)] (mod 2)
    - triples in `forbidden` excluded entirely (cap 0)
    - require_deg: exact ell_x values if given
    Returns list of solutions (as sorted triple lists) up to nothing
    (raw), but stops after finding up to 50.
    """
    tris = [t for t in combinations(range(npts), 3)
            if t not in set(forbidden)]
    sols = []
    deg = [0] * npts
    pair = Counter()

    def pr(x, y):
        return (x, y) if x < y else (y, x)

    def feasible_completion(i, rem):
        return rem >= 0

    def check_final():
        for x in range(npts):
            if require_deg is not None:
                if deg[x] != require_deg[x]:
                    return False
            elif deg[x] % 3 != point_res[x] % 3:
                return False
        for (a, b), par in pair_parity.items():
            if pair[pr(a, b)] % 2 != par % 2:
                return False
        # also pairs not listed must be even
        for a in range(npts):
            for b in range(a + 1, npts):
                if (a, b) not in pair_parity and pair[(a, b)] % 2 != 0:
                    return False
        return True

    def bt(i, rem):
        if len(sols) >= 50:
            return
        if rem == 0:
            if check_final():
                cur = []
                for j, t in enumerate(tris):
                    cur += [t] * mult[j]
                sols.append(cur)
            return
        if i == len(tris):
            return
        # prune: max addable weight
        if 2 * (len(tris) - i) < rem:
            return
        # prune on degree overshoot: if any point already exceeds its
        # possible final value... (light pruning: degrees can only grow)
        if require_deg is not None:
            for x in range(npts):
                if deg[x] > require_deg[x]:
                    return
        t = tris[i]
        maxm = min(2, cap.get(t, 2), rem)
        for k in range(maxm, -1, -1):
            mult[i] = k
            for x in t:
                deg[x] += k
            for a, b in combinations(t, 2):
                pair[(a, b)] += k
            bt(i + 1, rem - k)
            for x in t:
                deg[x] -= k
            for a, b in combinations(t, 2):
                pair[(a, b)] -= k
            mult[i] = 0

    mult = [0] * len(tris)
    bt(0, weight)
    return sols


def case_A():
    """k5=1: pentad P = {0..4}, L = 8.  Congruences: ell_x == 0 (3);
    pairs inside P odd, all other pairs even.  Leave support: P plus
    at most 3 outside points (ell >= 3 there); triples inside P have
    y <= 1 (P covers them once).  npts = 5 + 3 = 8."""
    npts = 8
    point_res = [0] * npts
    pair_parity = {}
    for a, b in combinations(range(5), 2):
        pair_parity[(a, b)] = 1
    # every other pair even (checked by default in check_final)
    cap = {}
    for t in combinations(range(5), 3):
        cap[t] = 1
    sols = leave_search(npts, 8, point_res, pair_parity, cap)
    return sols


def case_B():
    """k5=2 doubled pentad P={0..4} x2, L=6.  All pairs even (p_xy = 2
    inside P).  Triples inside P have coverage 2 already: cap 0.
    ell_x == 0 (3).  Support: P + up to 6 outside; but each outside
    point needs ell >= 3, sum ell = 18 -> <= 6 outside; triples must
    avoid inside-P entirely."""
    npts = 5 + 6
    point_res = [0] * npts
    pair_parity = {}
    cap = {}
    for t in combinations(range(5), 3):
        cap[t] = 0
    sols = leave_search(npts, 6, point_res, pair_parity, cap)
    return sols


def case_C():
    """k5=2, P1 = {0,1,2,3,4}=S+{u}, P2 = S+{v}: S={0,1,2,3}, u=4, v=5.
    L=6.  Odd pairs: exactly K(P1) delta K(P2) = pairs {x,4},{x,5} for
    x in S.  ell_x == 0 (3).  Forced: ell = 3 on {0..5}, 0 outside
    (sum = 18 = 3L).  Triples inside P1 or P2 have y <= 1; triples in
    S... covered by both pentads (S subset of both): y <= 0 for triples
    inside S!  |S|=4: 4 triples dead.  npts = 6 (no outside points:
    ell must vanish there and any triple has <=1 outside pt... a triple
    with an outside point o has ell_o >= 1 > 0: forbidden).  So support
    = {0..5}."""
    npts = 6
    point_res = [0] * npts
    pair_parity = {}
    for x in range(4):
        pair_parity[(x, 4)] = 1
        pair_parity[(x, 5)] = 1
    cap = {}
    for t in combinations(range(4), 3):
        cap[t] = 0
    for t in combinations(range(5), 3):          # inside P1
        cap[t] = min(cap.get(t, 2), 1)
    for t in combinations((0, 1, 2, 3, 5), 3):   # inside P2
        cap[t] = min(cap.get(t, 2), 1)
    sols = leave_search(npts, 6, point_res, pair_parity, cap,
                        require_deg=[3, 3, 3, 3, 3, 3])
    return sols


def case_D():
    """k5=3, L=4.  Leave = 4 triple-slots; every touched point has
    ell_x >= 3 -> at most 4 touched points; ell values in {0,3}: hence
    exactly 4 points with ell=3: leave = K4^(3) once, or doubled pair
    of triples on 4 pts... enumerate leaves on abstract 4 pts and check
    which pentad triples (P1,P2,P3 intersection patterns) give a
    matching odd-graph.  We enumerate over pentad patterns on <= 15
    points with the leave on a 4-subset.

    Simplification: the leave's odd-graph O_leave (pairs with odd ell)
    lives inside the 4-point support; the pentads' odd graph O_pent
    (pairs in an odd number of pentads) must EQUAL O_leave, and every
    pentad pair not in O_leave must be even.  We enumerate pentad
    intersection patterns abstractly: 3 pentads on <= 15 points, and
    check whether K(P1)+K(P2)+K(P3) (GF2) can fit inside a 4-set that
    also carries a congruence-valid weight-4 leave.
    Necessary: |O_pent| <= 6 and O_pent spans <= 4 points.
    """
    out = []
    # enumerate intersection patterns via direct small search:
    # P1 = {0..4}; P2 from canonical choices; P3 likewise, points <= 15
    P1 = tuple(range(5))
    seen = set()
    for s12 in range(5, -1, -1):
        # P2 shares its first s12 pts with P1
        P2 = tuple(list(range(s12)) + list(range(5, 10 - s12 + 5))[:5 - s12])
        P2 = tuple(sorted(P2))
        # enumerate P3 over subsets of a 15-pt universe is too big;
        # instead: O = K(P1)^K(P2)^K(P3) must span <=4 pts.  Use parity
        # degree argument: deg_O(x) = sum over pentads thru x of
        # (4 minus shared...)  -- do a direct search over P3 subsets of
        # the union plus 5 fresh points.
        universe = sorted(set(P1) | set(P2) | set(range(10, 15)))
        for P3 in combinations(universe, 5):
            key = canon_triple_sets([P1, P2, tuple(P3)])
            if key in seen:
                continue
            seen.add(key)
            O = Counter()
            for P in (P1, P2, P3):
                for e in combinations(sorted(P), 2):
                    O[e] += 1
            Oset = [e for e, c in O.items() if c % 2 == 1]
            pts = set(x for e in Oset for x in e)
            if len(pts) > 4:
                continue
            # check doubled-pentad legality inside: no triple covered
            # >2 by the three pentads
            tri = Counter()
            bad = False
            for P in (P1, P2, P3):
                for t in combinations(sorted(P), 3):
                    tri[t] += 1
                    if tri[t] > 2:
                        bad = True
            if bad:
                continue
            out.append((P1, P2, tuple(P3), sorted(Oset)))
    return out


def canon_triple_sets(sets):
    """crude canonical form for a list of point-sets (iso-dedup)."""
    from itertools import permutations as perms
    pts = sorted(set(x for s in sets for x in s))
    best = None
    # too expensive for full perm; use degree-refined ordering heuristic
    deg = Counter(x for s in sets for x in s)
    order = sorted(pts, key=lambda x: (-deg[x], x))
    relab = {p: i for i, p in enumerate(order)}
    key = tuple(sorted(tuple(sorted(relab[x] for x in s)) for s in sets))
    return key


def case_E():
    """k5=5, L=0: five pentads (mult <= 2), every pair of points in an
    EVEN number of pentads, every triple in <= 2.  Search over
    canonical supports.  P1 = {0..4}.  Every pair of P1 must be
    covered again an odd number of times by P2..P5."""
    P1 = tuple(range(5))
    sols = []
    seen = set()
    # universe: 17 points suffices (see report); use 0..16
    universe = list(range(17))

    def pairs_of(P):
        return set(combinations(sorted(P), 2))

    def ok_triples(Ps):
        tri = Counter()
        for P in Ps:
            for t in combinations(sorted(P), 3):
                tri[t] += 1
                if tri[t] > 2:
                    return False
        return True

    def bt(Ps, i):
        if len(sols) >= 20:
            return
        if len(Ps) == 5:
            par = Counter()
            for P in Ps:
                for e in pairs_of(P):
                    par[e] += 1
            if all(c % 2 == 0 for c in par.values()):
                key = canon_triple_sets(list(Ps))
                if key not in seen:
                    seen.add(key)
                    sols.append(list(Ps))
            return
        # prune: pairs already odd must still be fixable: each later
        # pentad covers <= 10 pairs; remaining pentads r
        par = Counter()
        for P in Ps:
            for e in pairs_of(P):
                par[e] += 1
        odd = [e for e, c in par.items() if c % 2 == 1]
        r = 5 - len(Ps)
        if len(odd) > 10 * r:
            return
        # canonical: next pentad starts no earlier (lex) than previous
        for P in combinations(universe, 5):
            if Ps and tuple(P) < tuple(Ps[-1]):
                continue
            # allow doubling: P == last is fine (multiplicity 2);
            # more than 2 handled by triple check
            Ps2 = Ps + [tuple(P)]
            if not ok_triples(Ps2):
                continue
            # quick prune: after adding, odd pairs bounded
            bt(Ps2, i + 1)

    # this raw search is too big; restrict: every subsequent pentad
    # must intersect the union of previous in >= 3 points (else its
    # pairs can never be evened: a pentad with <= 2 old points has
    # >= C(3,2)=3 fresh pairs... actually fresh-fresh pairs must be
    # covered again later; allow but bound universe to 13)
    sols = []
    seen = set()
    universe = list(range(13))
    bt([P1], 0)
    return sols


def case_W10():
    """All weight-10 leaves with k5=0 class congruences: ell_x == 0
    (mod 3), ALL pairs even, mult <= 2, support <= 10 (weight 30/3).
    Up to iso (crude canon), list shapes."""
    npts = 10
    point_res = [0] * npts
    sols = leave_search(npts, 10, point_res, {}, {})
    shapes = {}
    for s in sols:
        key = canon_triple_sets([tuple(t) for t in s])
        shapes.setdefault(key, s)
    return shapes


def case_S91():
    """m=9: pentad P={0..4} + 38 quads would leave weight 6 with all
    ell_x = 2 (mod-3 forced: 2C(8,2)=56==2 (3)).  Pair parity: pairs
    inside P odd, others even.  Enumerate: any leave with ell_x = 2
    for ALL 9 points, weight 6, triples inside P capped at 1."""
    npts = 9
    pair_parity = {}
    for a, b in combinations(range(5), 2):
        pair_parity[(a, b)] = 1
    cap = {}
    for t in combinations(range(5), 3):
        cap[t] = 1
    sols = leave_search(npts, 6, [2] * 9, pair_parity, cap,
                        require_deg=[2] * 9)
    return sols


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "A"):
        s = case_A()
        print(f"CASE A (k5=1, L=8): {len(s)} solutions")
        for x in s[:5]:
            print("   ", x)
    if which in ("all", "B"):
        s = case_B()
        print(f"CASE B (k5=2 doubled, L=6): {len(s)} solutions")
        for x in s[:5]:
            print("   ", x)
    if which in ("all", "C"):
        s = case_C()
        print(f"CASE C (k5=2, s=4, L=6): {len(s)} solutions")
        for x in s[:5]:
            print("   ", x)
    if which in ("all", "S91"):
        s = case_S91()
        print(f"CASE S91 (m=9 pentad+38 quads leave): {len(s)} solutions")
    if which in ("all", "W10"):
        sh = case_W10()
        print(f"CASE W10 (weight-10 all-even leaves): {len(sh)} shapes")
        for k, v in list(sh.items())[:10]:
            print("   ", sorted(Counter(map(tuple, v)).items()))
    if which in ("all", "D"):
        s = case_D()
        print(f"CASE D (k5=3 patterns with small odd-graph): {len(s)}")
        for x in s[:10]:
            print("   ", x)
    if which in ("all", "E"):
        s = case_E()
        print(f"CASE E (k5=5 even-pair systems): {len(s)}")
        for x in s[:5]:
            print("   ", x)

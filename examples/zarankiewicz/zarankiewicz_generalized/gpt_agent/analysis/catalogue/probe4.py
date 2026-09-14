"""Refined checks: (1) pencil-pair completion of the AG decode for
w_9x22/25/26; (2) w_9x27/28 = cone + 8-point body, body vs 2xSQS(8);
(3) 10x15 arc test; (4) big packings: leaves + cyclic symmetry."""
import sys
from collections import Counter
from itertools import combinations

import catlib as C
import gens as G
from audit import leave_of, is_doubled_pentagon, is_doubled_parallel_class
from probe3 import reconstruct_ag_from_lines


def ag_decode_full(wname, ws):
    """Full AG(2,3)-base-point certificate for w_9x22-style witnesses."""
    w = ws[wname]
    lay = C.layers(9, w["blocks"])
    quads = lay.get(4, [])
    pentads = lay.get(5, [])
    deg = C.point_deg(quads, 9)
    b = deg.index(max(deg))
    through = [x for x in quads if b in x]
    others = [x for x in quads if b not in x]
    derived = [tuple(sorted(set(x) - {b})) for x in through]
    ag = reconstruct_ag_from_lines(9, derived)
    if ag is None:
        return None
    lines = set(ag)
    thru_b = [l for l in lines if b in l]
    pencilpairs = []
    for l1, l2 in combinations(thru_b, 2):
        pencilpairs.append(tuple(sorted((set(l1) | set(l2)) - {b})))
    full = set(range(9))
    cert = dict(base=b, n_lines_used=len(through))
    ok = True
    for x in others:
        if tuple(sorted(x)) not in pencilpairs:
            ok = False
    n_pp_quads = len(others)
    pc, plp = 0, 0
    for p in pentads:
        comp = tuple(sorted(full - set(p)))
        if comp in pencilpairs:
            plp += 1
            continue
        hit = False
        for l in lines:
            if set(l) <= set(comp):
                hit = True
                break
        if hit:
            pc += 1
        else:
            ok = False
    cert.update(pp_quads=n_pp_quads, comp_linept_pentads=pc,
                comp_pp_pentads=plp, all_matched=ok)
    return cert


def main():
    ws = {w["file"]: w for w in C.load_banks()}

    print("== (1) AG-base-point full decode")
    for f in ["w_9x22.json", "w_9x25.json", "w_9x26.json"]:
        print(f"  {f}: {ag_decode_full(f, ws)}")

    print("\n== (2) w_9x27/28: split at avoided point")
    s8 = G.sqs8()
    d8 = [x for x in s8 for _ in range(2)]
    for f in ["w_9x27.json", "w_9x28.json"]:
        w = ws[f]
        lay = C.layers(9, w["blocks"])
        quads = lay[4]
        deg = C.point_deg(quads, 9)
        print(f"  {f}: quad degs {deg}")
        if 0 in deg:
            av = deg.index(0)
            body = [tuple(sorted(x)) for x in quads]
            pts = sorted(set(p for x in body for p in x))
            relab = {p: i for i, p in enumerate(pts)}
            body8 = [tuple(sorted(relab[p] for p in x)) for x in body]
            r = C.find_embedding(8, body8, 8, d8)
            print(f"    avoided pt {av}; {len(body8)} quads on 8 pts; "
                  f"sub-2xSQS8: {'YES' if r and r != 'TIMEOUT' else r}")
            # also: leave of body on 8 points
            lv = leave_of(8, body8)
            print(f"    body leave weight {sum(lv.values())}")

    print("\n== (3) 10x15: arcs test")
    w = ws["T2_10x15_81_cadical_witness.json"]
    lay = C.layers(10, w["blocks"])
    pentads = lay[5]
    der = [tuple(sorted(set(p) - {0})) for p in pentads]
    T = [(5, 7, 9), (5, 6, 8), (3, 4, 8), (2, 4, 9), (1, 3, 7),
         (1, 2, 6)]
    Tr = [tuple(x - 1 for x in t) for t in T]
    ag = reconstruct_ag_from_lines(9, Tr)
    lines9 = [tuple(x + 1 for x in l) for l in ag]
    print("  full AG lines (1..9):", lines9)
    for d in der:
        coll = [l for l in lines9 if len(set(l) & set(d)) == 3]
        print(f"    derived {d}: contains lines {coll}")

    print("\n== (4) big packings")
    for f, kind in [("t33_m11_b80_hole.json", "pentagon"),
                    ("t33_m19_b482_hole.json", "pentagon"),
                    ("t33_m23_b883_hole.json", "pentagon"),
                    ("t33_m27_b1458_parclass.json", "parclass")]:
        w = ws[f]
        m = w["m"]
        lv = leave_of(m, w["blocks"])
        if kind == "pentagon":
            r = is_doubled_pentagon(lv)
            print(f"  {f}: leave weight {sum(lv.values())}, "
                  f"doubled pentagon on {r}")
        else:
            r = is_doubled_parallel_class(lv, m)
            print(f"  {f}: leave weight {sum(lv.values())}, "
                  f"doubled parallel class: {r is not None} "
                  f"({len(r) if r else 0} triples)")
        # cyclic symmetry: try candidate permutations of small order
        bl = set()
        mult = Counter(w["blocks"])

        def invariant_under(perm):
            im = Counter()
            for x, c in mult.items():
                im[tuple(sorted(perm[p] for p in x))] += c
            return im == mult

        # Z5 on the pentagon support, extended: search over rotations
        found = []
        if kind == "pentagon" and r:
            pts = r
            rest = [p for p in range(m) if p not in pts]
            # pentagon cycle order: edges = complements of leave triples
            edges = [tuple(sorted(set(pts) - set(t))) for t in lv]
            nbr = {p: [] for p in pts}
            for a, b2 in set(edges):
                nbr[a].append(b2)
                nbr[b2].append(a)
            cyc = [pts[0]]
            while len(cyc) < 5:
                for y in nbr[cyc[-1]]:
                    if y not in cyc:
                        cyc.append(y)
                        break
            # candidate sigma: rotate cyc; on rest try to extend by
            # backtracking (forced map = partial perm)
            target = {cyc[i]: cyc[(i + 1) % 5] for i in range(5)}
            ext = extend_partial_aut(m, w["blocks"], target)
            print(f"    Z5 rotation extends to automorphism: "
                  f"{'YES' if ext else 'NO'}")
        if kind == "parclass":
            # sigma = three 9-cycles on 0..26? try i -> i+3 mod 27 and
            # (i+1 within residue classes)
            cands = []
            perm1 = [(i + 3) % 27 for i in range(27)]
            cands.append(("i->i+3 (mod 27)", perm1))
            perm2 = [(i + 1) % 9 + 9 * (i // 9) for i in range(27)]
            cands.append(("+1 in each block of 9", perm2))
            for nm, p in cands:
                print(f"    invariant under {nm}: {invariant_under(p)}")


def extend_partial_aut(m, blocks, forced):
    """Extend forced partial point-map to a full automorphism of the
    block multiset, by backtracking."""
    blocks = [tuple(sorted(x)) for x in blocks]
    mult = Counter(blocks)
    pd = C.pair_deg(blocks, m)
    dw = [Counter() for _ in range(m)]
    for x in blocks:
        for p in x:
            dw[p][len(x)] += 1
    pts = list(forced.keys()) + [p for p in range(m) if p not in forced]
    f = [-1] * m
    used = [False] * m
    for a, b in forced.items():
        f[a] = b
        used[b] = True

    def bt(i):
        if i == m:
            im = Counter()
            for x, c in mult.items():
                im[tuple(sorted(f[p] for p in x))] += c
            return im == mult
        x = pts[i]
        if f[x] != -1:
            return bt(i + 1)
        for y in range(m):
            if used[y] or dw[x] != dw[y]:
                continue
            ok = True
            for j in range(i):
                z = pts[j]
                if f[z] == -1:
                    continue
                pa = pd.get((min(x, z), max(x, z)), 0)
                pb = pd.get((min(y, f[z]), max(y, f[z])), 0)
                if pa != pb:
                    ok = False
                    break
            if not ok:
                continue
            f[x] = y
            used[y] = True
            if bt(i + 1):
                return True
            f[x] = -1
            used[y] = False
        return False

    return bt(0)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()

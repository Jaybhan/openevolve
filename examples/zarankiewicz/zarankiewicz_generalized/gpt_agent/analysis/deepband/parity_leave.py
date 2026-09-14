"""Hypothesis (iii): the supply drops S_m(b) - S_m(b+1) and the leave
structure. PROVEN ingredients (verified here numerically on top of the
algebra in report.md):

  pair-parity:   ell_xy == p_xy (mod 2)   (p_xy = #odd-weight blocks through
                 the pair; for quad/pentad configs = #pentads through it)
  point mod 3:   mu_x == 2*C(m-1,2) (mod 3)  for quad/pentad configs
  counting:      sum_x mu_x = 3L,  sum_y ell_xy = 2 mu_x,  L = B - slots.

Consequence (the parity leave bound): for a config with pentad multiset M,
  L >= L_par(M) := max( ceil(|O|/3), ceil(sum_x mu_min(x) / 3) ),
  mu_min(x) = least t >= ceil(deg_O(x)/2) with t == r_m (mod 3),
  O = odd-pair graph of M, r_m = 2*C(m-1,2) mod 3.
Hence  S_m(b) <= max over legal b-pentad multisets M of
                 floor((B - 10 b - L_par(M)) / 4).

This script enumerates all legal b-pentad multisets up to relabeling
(WLOG one pentad = {0..4}) for b = 1..bmax and reports the bound vs the
ILP-proven S_m(b), plus which intersection structures attain the max.
"""
import sys
from itertools import combinations
from math import comb

def run(m, bmax=4, Strue=None):
    B = 2 * comb(m, 3)
    r = (2 * comb(m - 1, 2)) % 3
    pentads = list(combinations(range(m), 5))
    tri_id = {t: i for i, t in enumerate(combinations(range(m), 3))}
    pent_tris = [tuple(tri_id[t] for t in combinations(p, 3)) for p in pentads]
    P0 = tuple(range(5))
    i0 = pentads.index(P0)

    def mu_min(deg):
        t = (deg + 1) // 2
        while t % 3 != r:
            t += 1
        return t

    print(f"m={m}: B={B}, point congruence mu_x == {r} (mod 3)")
    for b in range(1, bmax + 1):
        best = -1
        best_struct = None
        # multisets of size b containing P0 (WLOG), multiplicity <= 2
        for rest in combinations_with_replacement_idx(len(pentads), b - 1):
            M = [i0] + list(rest)
            # legality: every triple covered <= 2
            cov = [0] * len(tri_id)
            ok = True
            for i in M:
                for t in pent_tris[i]:
                    cov[t] += 1
                    if cov[t] > 2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # odd-pair graph
            pdeg = {}
            paircnt = {}
            for i in M:
                for pr in combinations(pentads[i], 2):
                    paircnt[pr] = paircnt.get(pr, 0) + 1
            odd = [pr for pr, k in paircnt.items() if k % 2 == 1]
            deg = [0] * m
            for x, y in odd:
                deg[x] += 1
                deg[y] += 1
            L1 = (len(odd) + 2) // 3
            L2 = (sum(mu_min(d) for d in deg) + 2) // 3
            Lpar = max(L1, L2)
            bound = (B - 10 * b - Lpar) // 4
            if bound > best:
                best = bound
                ints = sorted(len(set(pentads[a]) & set(pentads[bb]))
                              for a, bb in combinations(M, 2))
                best_struct = (ints, len(odd), Lpar)
        st = f"  S_{m}({b}) <= {best}   [parity bound]"
        if Strue and b in Strue:
            st += f"   ILP truth: {Strue[b]}" + \
                  ("   TIGHT" if Strue[b] == best else
                   f"   gap {best - Strue[b]}")
        print(st + f"   argmax intersections={best_struct[0]} "
                   f"|O|={best_struct[1]} L_par={best_struct[2]}")


def combinations_with_replacement_idx(n, k):
    from itertools import combinations_with_replacement
    return combinations_with_replacement(range(n), k)


if __name__ == "__main__":
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    bmax = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    Strue = {8: {1: 23, 2: 21, 3: 17, 4: 15, 5: 14},
             9: {1: 37, 3: 33, 4: 30, 6: 25, 7: 23, 8: 20, 12: 10},
             7: {1: 12, 2: 10, 3: 8, 4: 6},
             6: {1: 6, 2: 4}}.get(m)
    run(m, bmax, Strue)

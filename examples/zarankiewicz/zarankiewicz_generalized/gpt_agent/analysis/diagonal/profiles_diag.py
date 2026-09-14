"""Arithmetic profile ladder for diagonal cells (m=n).

Enumerates non-increasing column-weight profiles (w_1..w_n), 2<=w<=m,
sum = E, surviving all PROVEN aggregated cuts:

 (1) slot budget:      sum C(w,3) <= B = 2*C(m,3)      [2-fold packing]
 (2) big-pair overlap: sum_{c<d} C((w_c+w_d-m)+, 3) <= C(m,3)
     [column pair c,d shares >= w_c+w_d-m rows; a triple can lie in at
      most 2 columns, so over column pairs each triple is shared by at
      most one pair: sum C(|int|,3) <= #triples covered twice <= C(m,3)]
 (3) Lemma C aggregate: sum w(w-3) <= m * floor(2*C(m-1,2)/3)
 (4) per-point pair capacity vs heaviest columns: the w_1 rows of the
     heaviest column each lie in it; a row x in columns c1..cd uses
     sum C(w_ci - 1, 2) <= 2*C(m-1,2) of its pair capacity.  Relaxation
     used here: C(w_1 - 1, 2) <= 2*C(m-1,2) (trivially true; kept for
     form) -- the real (4) needs assignments, skipped.

Output: per E, count of surviving profiles + the set of occurring max
weights w_1 (these are the cube targets an UNSAT proof must cover).
"""
import sys
from functools import lru_cache
from math import comb


def survivors(m, E, wmax=None):
    wmax = wmax or m
    B = 2 * comb(m, 3)
    lemC = m * ((2 * comb(m - 1, 2)) // 3)
    out = []

    prof = []

    def rec(remaining_cols, remaining_sum, cap, slots, lemc):
        if slots > B or lemc > lemC:
            return
        if remaining_cols == 0:
            if remaining_sum == 0:
                # cut (2) on the complete profile
                tot = 0
                for i in range(len(prof)):
                    for j in range(i + 1, len(prof)):
                        ov = prof[i] + prof[j] - m
                        if ov >= 3:
                            tot += comb(ov, 3)
                if tot <= comb(m, 3):
                    out.append(tuple(prof))
            return
        # bounds: remaining sum must fit in remaining_cols * [2, cap]
        if remaining_sum < 2 * remaining_cols or \
           remaining_sum > cap * remaining_cols:
            return
        for w in range(min(cap, remaining_sum - 2 * (remaining_cols - 1)),
                       1, -1):
            prof.append(w)
            rec(remaining_cols - 1, remaining_sum - w, w,
                slots + comb(w, 3), lemc + w * (w - 3))
            prof.pop()

    rec(m, E, wmax, 0, 0)
    return out


def main():
    m = int(sys.argv[1])
    e_lo, e_hi = int(sys.argv[2]), int(sys.argv[3])
    for E in range(e_lo, e_hi + 1):
        profs = survivors(m, E)
        maxw = sorted(set(p[0] for p in profs))
        print(f"E={E}: {len(profs)} surviving profiles, max-weight values "
              f"{maxw}")
        if len(profs) <= 12:
            for p in profs:
                print("   ", p)


if __name__ == "__main__":
    main()

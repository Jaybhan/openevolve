"""Fixed-edge-count focused local search for K_{3,3}-free m x n matrices.

State: exactly E ones.  Cost = sum over row-triples of max(0, cov-2).
Data: cols[c] = row set; tricol[t] = list of columns containing triple t.
Moves (WalkSAT-flavored): when violations exist, pick a violated triple,
remove one of its three rows from one covering column; then add a 1
somewhere else (biased random).  Metropolis on the combined delta.

Usage: ls_diag.py M N E [--seed FILE] [--restarts R] [--iters I] [--out F]
Exit 0 on success (witness written), 1 otherwise.
"""
import argparse
import json
import random
import sys
import time
from itertools import combinations


class State:
    def __init__(self, m, n, cols):
        self.m, self.n = m, n
        self.cols = [set(c) for c in cols]
        self.tricol = {}
        for c in range(n):
            for t in combinations(sorted(self.cols[c]), 3):
                self.tricol.setdefault(t, []).append(c)
        self.viol = set(t for t, cs in self.tricol.items() if len(cs) > 2)
        self.cost = sum(len(cs) - 2 for cs in self.tricol.values()
                        if len(cs) > 2)

    def rem(self, r, c):
        rest = [x for x in self.cols[c] if x != r]
        self.cols[c].discard(r)
        for a, b in combinations(rest, 2):
            t = tuple(sorted((a, b, r)))
            cs = self.tricol[t]
            if len(cs) > 2:
                self.cost -= 1
            cs.remove(c)
            if len(cs) <= 2:
                self.viol.discard(t)
            if not cs:
                del self.tricol[t]

    def add(self, r, c):
        rest = list(self.cols[c])
        self.cols[c].add(r)
        for a, b in combinations(rest, 2):
            t = tuple(sorted((a, b, r)))
            cs = self.tricol.setdefault(t, [])
            cs.append(c)
            if len(cs) > 2:
                self.cost += 1
                self.viol.add(t)

    def delta_add(self, r, c):
        d = 0
        for a, b in combinations(self.cols[c], 2):
            t = tuple(sorted((a, b, r)))
            if len(self.tricol.get(t, ())) >= 2:
                d += 1
        return d


def anneal(m, n, E, seed_cols, iters, rng, t0=0.8, t1=0.05):
    cols = [set(c) for c in seed_cols] if seed_cols else \
        [set() for _ in range(n)]
    ones = sum(len(c) for c in cols)
    while ones != E:
        r, c = rng.randrange(m), rng.randrange(n)
        if ones < E and r not in cols[c]:
            cols[c].add(r)
            ones += 1
        elif ones > E and r in cols[c]:
            cols[c].discard(r)
            ones -= 1
    st = State(m, n, cols)
    bestcost = st.cost
    for it in range(iters):
        if st.cost == 0:
            return st, 0
        T = t0 * (t1 / t0) ** (it / iters)
        # choose 1-cell to remove: from a violated triple usually
        if st.viol and rng.random() < 0.9:
            t = rng.choice(tuple(st.viol))
            c1 = rng.choice(st.tricol[t])
            r1 = t[rng.randrange(3)]
        else:
            while True:
                c1 = rng.randrange(n)
                if st.cols[c1]:
                    r1 = rng.choice(tuple(st.cols[c1]))
                    break
        pre = st.cost
        st.rem(r1, c1)
        # choose 0-cell to add: sample a few, take best delta
        best = None
        for _ in range(6):
            c0 = rng.randrange(n)
            if len(st.cols[c0]) >= m:
                continue
            r0 = rng.randrange(m)
            if r0 in st.cols[c0] or (r0 == r1 and c0 == c1):
                continue
            d = st.delta_add(r0, c0)
            if best is None or d < best[0]:
                best = (d, r0, c0)
            if d == 0:
                break
        if best is None:
            st.add(r1, c1)
            continue
        d, r0, c0 = best
        newcost = st.cost + d
        dd = newcost - pre
        if dd <= 0 or rng.random() < 2.718281828 ** (-dd / T):
            st.add(r0, c0)
        else:
            st.add(r1, c1)
        bestcost = min(bestcost, st.cost)
    return st, st.cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("m", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("E", type=int)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--iters", type=int, default=300000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rng", type=int, default=12345)
    args = ap.parse_args()
    seed_cols = None
    if args.seed:
        with open(args.seed) as f:
            w = json.load(f)
        assert w["m"] == args.m and w["n"] == args.n
        seed_cols = w["blocks"]
    rng = random.Random(args.rng)
    t0 = time.time()
    for rs in range(args.restarts):
        st, cost = anneal(args.m, args.n, args.E, seed_cols,
                          args.iters, rng)
        el = time.time() - t0
        print(f"restart {rs}: final cost {cost}  [{el:.0f}s]", flush=True)
        if cost == 0:
            out = args.out or f"ls_{args.m}x{args.n}_{args.E}.json"
            blocks = [sorted(c) for c in st.cols]
            assert sum(len(b) for b in blocks) == args.E
            with open(out, "w") as f:
                json.dump({"m": args.m, "n": args.n, "edges": args.E,
                           "blocks": blocks,
                           "source": f"ls_diag restart {rs}"}, f)
            print(f"SUCCESS -> {out}")
            sys.exit(0)
    print("no success")
    sys.exit(1)


if __name__ == "__main__":
    main()

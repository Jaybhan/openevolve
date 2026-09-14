"""Eviction-random-walk local search for T_{3,3}(v) packings.

State: multiplicity in {0,1,2} per 4-subset, per-triple coverage <= 2 kept
invariant.  Moves: add a random addable block; when the neighborhood is
exhausted, force-add a random block by evicting one covering block per
saturated triple.  Best packing is checkpointed to a JSON witness (usable
as a phase seed for run_t33.py --seed-blocks).

Usage: t33_localsearch.py [--v 19] [--target 482] [--budget 1800]
                          [--seed 0] [--out FILE]
"""
import argparse
import json
import os
import random
import time
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=int, default=19)
    ap.add_argument("--target", type=int, default=482)
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--init", default=None,
                    help="JSON packing witness to start from")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    out = args.out or os.path.join(
        HERE, "witnesses", f"ls_t33_{args.v}_best_s{args.seed}.json")

    blocks = list(combinations(range(args.v), 4))
    nb = len(blocks)
    tidx = {t: i for i, t in enumerate(combinations(range(args.v), 3))}
    btris = [[tidx[t] for t in combinations(B, 3)] for B in blocks]

    mult = [0] * nb
    cov = [0] * len(tidx)
    covering = [[] for _ in range(len(tidx))]  # triple -> list of block ids
    count = 0
    best = 0
    best_state = None

    def addable(j):
        return mult[j] < 2 and all(cov[t] < 2 for t in btris[j])

    def add(j):
        nonlocal count
        mult[j] += 1
        count += 1
        for t in btris[j]:
            cov[t] += 1
            covering[t].append(j)

    def remove(j):
        nonlocal count
        mult[j] -= 1
        count -= 1
        for t in btris[j]:
            cov[t] -= 1
            covering[t].remove(j)

    def snapshot():
        return [list(blocks[j]) for j in range(nb) for _ in range(mult[j])]

    if args.init:
        bidx = {B: j for j, B in enumerate(blocks)}
        with open(args.init) as f:
            init = json.load(f)
        for bl in init["blocks"]:
            j = bidx[tuple(sorted(bl))]
            if addable(j):
                add(j)
        print(f"initialized from {args.init}: {count} blocks", flush=True)
        best = count
        best_state = snapshot()

    t0 = time.time()
    last_report = t0
    fails = 0
    moves = 0
    while time.time() - t0 < args.budget:
        moves += 1
        j = rng.randrange(nb)
        if addable(j):
            add(j)
            fails = 0
        else:
            fails += 1
            if fails >= 400:
                # exhaustive scan
                cand = [k for k in range(nb) if addable(k)]
                if cand:
                    add(rng.choice(cand))
                else:
                    # force-add a random block, evicting blockers
                    k = rng.randrange(nb)
                    while mult[k] >= 2:
                        k = rng.randrange(nb)
                    for t in btris[k]:
                        while cov[t] >= 2:
                            remove(rng.choice(covering[t]))
                    add(k)
                fails = 0
        if count > best:
            best = count
            best_state = snapshot()
            with open(out, "w") as f:
                json.dump({"v": args.v, "count": best, "blocks": best_state,
                           "source": f"localsearch seed={args.seed}"}, f)
            if best >= args.target:
                print(f"TARGET REACHED: {best} blocks "
                      f"({time.time()-t0:.0f}s, {moves} moves)", flush=True)
                return
        if time.time() - last_report > 60:
            last_report = time.time()
            print(f"t={time.time()-t0:.0f}s best={best} cur={count} "
                  f"moves={moves}", flush=True)
    print(f"DONE best={best} (target {args.target}) moves={moves}", flush=True)


if __name__ == "__main__":
    main()

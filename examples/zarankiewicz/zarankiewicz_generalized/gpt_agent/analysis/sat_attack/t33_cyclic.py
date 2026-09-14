"""Cyclic-ansatz attack on T_{3,3}(19) = 482 = 25*19 + 7.

Z_19 acts on points; all 204 block-orbits and 51 triple-orbits have size 19
(19 prime).  A choice of 25 base-block orbits (with repetition <= 2) whose
triple-orbit coverage respects capacity 2 yields 475 blocks using 100 of the
102 orbit-slots; the 38 residual triple-slots must then host 7 loose blocks
(all four of each loose block's triples on residual slots).

Level 1 (orbit selection): SAT over orbit multiplicities.
Level 2 (loose completion): exact DFS over candidate blocks.
Iterate level-1 solutions with blocking clauses until level 2 finds 7.

Usage: t33_cyclic.py [--budget 1800] [--max-sols 100000] [--loose 7]
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pysat.card import CardEnc, EncType
from pysat.solvers import Solver

HERE = os.path.dirname(os.path.abspath(__file__))
V = 19


def canon_orbit(subset):
    """Canonical representative of the Z_19 orbit of a subset."""
    best = None
    for s in range(V):
        cand = tuple(sorted((x + s) % V for x in subset))
        if best is None or cand < best:
            best = cand
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1800.0)
    ap.add_argument("--max-sols", type=int, default=1000000)
    ap.add_argument("--loose", type=int, default=7)
    ap.add_argument("--norbits", type=int, default=25)
    ap.add_argument("--tag", default="T3_cyclic")
    args = ap.parse_args()

    t0 = time.time()
    blocks = list(combinations(range(V), 4))
    triples = list(combinations(range(V), 3))
    torbit = {}      # triple -> orbit id
    orb_reps = []
    for t in triples:
        c = canon_orbit(t)
        if c not in torbit:
            torbit[c] = len(orb_reps)
            orb_reps.append(c)
        torbit[t] = torbit[c]
    nto = len(orb_reps)

    borbit_of = {}
    borb_reps = []
    for B in blocks:
        c = canon_orbit(B)
        if c not in borbit_of:
            borbit_of[c] = len(borb_reps)
            borb_reps.append(c)
        borbit_of[B] = borbit_of[c]
    nbo = len(borb_reps)
    # coverage vector of one block-orbit over triple-orbits (per single copy):
    # orbit of B contributes, for each triple t of B, 1 slot at torbit(t) --
    # aggregated over the 19 rotations each triple-orbit coord gets 19x, so at
    # the orbit level (capacity 2 per triple-orbit coord) each base block copy
    # adds its multiset {torbit(t): t in triples(B)}.
    covvec = []
    for rep in borb_reps:
        cnt = Counter(torbit[t] for t in combinations(rep, 3))
        covvec.append(cnt)

    print(f"{nbo} block-orbits, {nto} triple-orbits", flush=True)
    assert nto == 51 and nbo == 204

    # ---- level-1 SAT: pick multiset of orbits, |.| = norbits, coverage <= 2
    a = {}
    b = {}
    top = 0
    for i in range(nbo):
        top += 1
        a[i] = top
    for i in range(nbo):
        top += 1
        b[i] = top
    clauses = [[a[i], -b[i]] for i in range(nbo)]
    # per-triple-orbit capacity 2 (counting multiplicity in covvec)
    for o in range(nto):
        lits = []
        for i in range(nbo):
            k = covvec[i].get(o, 0)
            if k == 0:
                continue
            if k > 2:
                clauses.append([-a[i]])  # single copy already overflows
                continue
            # each copy of orbit i adds k slots at o: k=1 -> one lit per copy;
            # k=2 -> copy saturates o alone; two copies impossible
            if k == 1:
                lits.extend([a[i], b[i]])
            else:  # k == 2
                lits.extend([a[i], a[i]])   # weight-2: duplicate literal
                clauses.append([-b[i]])     # second copy would exceed 2
        # atmost-2 with duplicated literals == weighted; seqcounter handles dups
        cnf = CardEnc.atmost(lits=lits, bound=2, top_id=top,
                             encoding=EncType.seqcounter)
        clauses.extend(cnf.clauses)
        top = max(top, cnf.nv)
    all_lits = [a[i] for i in range(nbo)] + [b[i] for i in range(nbo)]
    cnf = CardEnc.equals(lits=all_lits, bound=args.norbits, top_id=top,
                         encoding=EncType.totalizer)
    clauses.extend(cnf.clauses)
    top = max(top, cnf.nv)

    solver = Solver(name="cadical195", bootstrap_with=clauses)
    print(f"level-1 CNF: {top} vars {len(clauses)} clauses "
          f"({time.time()-t0:.1f}s)", flush=True)

    best_loose = -1
    tried = 0
    while time.time() - t0 < args.budget and tried < args.max_sols:
        if not solver.solve():
            print(f"level-1 EXHAUSTED after {tried} solutions", flush=True)
            break
        model = set(l for l in solver.get_model() if l > 0)
        orbmult = {i: (1 if a[i] in model else 0) + (1 if b[i] in model else 0)
                   for i in range(nbo)}
        chosen = {i: k for i, k in orbmult.items() if k}
        tried += 1

        # ---- expand to concrete packing, compute residual capacities
        mult = Counter()
        for i, k in chosen.items():
            for s in range(V):
                blk = tuple(sorted((x + s) % V for x in borb_reps[i]))
                mult[blk] += k
        cov = Counter()
        for blk, k in mult.items():
            for t in combinations(blk, 3):
                cov[t] += k
        assert all(c <= 2 for c in cov.values())
        resid = {t: 2 - cov.get(t, 0) for t in triples}

        # ---- level-2: DFS for `loose` extra blocks
        cands = []
        for B in blocks:
            room = 2 - mult.get(B, 0)
            if room <= 0:
                continue
            if all(resid[t] >= 1 for t in combinations(B, 3)):
                cands.append((B, room))

        found = []

        def dfs(idx, need):
            if need == 0:
                return True
            if idx >= len(cands):
                return False
            # prune: remaining supply
            if sum(r for _, r in cands[idx:]) < need:
                return False
            B, room = cands[idx]
            ts = list(combinations(B, 3))
            usable = min(room, min(resid[t] for t in ts), need)
            for use in range(usable, -1, -1):
                if use:
                    for t in ts:
                        resid[t] -= use
                    found.extend([B] * use)
                if dfs(idx + 1, need - use):
                    return True
                if use:
                    for t in ts:
                        resid[t] += use
                    del found[-use:]
            return False

        ok = dfs(0, args.loose)
        nl = len(found) if ok else 0
        if ok:
            packing = [list(blk) for blk, k in mult.items() for _ in range(k)]
            packing += [list(B) for B in found]
            out = os.path.join(HERE, "witnesses",
                               f"{args.tag}_19_{len(packing)}_witness.json")
            with open(out, "w") as f:
                json.dump({"v": V, "count": len(packing), "blocks": packing,
                           "source": f"cyclic ansatz {args.norbits} orbits + "
                                     f"{args.loose} loose"}, f)
            print(f"SUCCESS after {tried} level-1 sols "
                  f"({time.time()-t0:.1f}s): {len(packing)} blocks -> {out}",
                  flush=True)
            return
        # track how close level-2 got (greedy count of candidate supply)
        supply = sum(r for _, r in cands)
        if supply > best_loose:
            best_loose = supply
            print(f"sol {tried}: loose supply {supply} "
                  f"(cands {len(cands)}) t={time.time()-t0:.0f}s", flush=True)

        # block this exact orbit multiset
        blocking = []
        for i in range(nbo):
            k = orbmult[i]
            if k == 0:
                blocking.append(a[i])
            elif k == 1:
                blocking.append(-a[i])   # drop it ...
                blocking.append(b[i])    # ... or raise it
            else:
                blocking.append(-b[i])
        solver.add_clause(blocking)
        if tried % 2000 == 0:
            print(f"tried {tried} level-1 sols, best loose supply "
                  f"{best_loose}, t={time.time()-t0:.0f}s", flush=True)
    print(f"NO SUCCESS: tried {tried} level-1 solutions in "
          f"{time.time()-t0:.0f}s (best loose supply {best_loose})",
          flush=True)


if __name__ == "__main__":
    main()

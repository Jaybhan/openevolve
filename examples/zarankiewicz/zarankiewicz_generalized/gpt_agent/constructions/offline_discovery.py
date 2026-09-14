"""OFFLINE discovery tool (not part of the engine): find witness matrices for
the cells the deterministic engine still misses, so their structure can be
studied and distilled into a derived family (or, failing that, an explicit
verified catalogue entry -- the same epistemic status as a Singer set).

Method: large-neighbourhood repair around the engine's near-miss: destroy k
columns, rebuild them by exact DFS over blocks under the residual capacity,
targeting one more edge.  Randomized destroy order with a seeded RNG, so runs
are reproducible.  Every witness found is re-verified with the engine's
verifier before being reported/saved.
"""

import json
import os
import random
import sys
import time
from itertools import combinations
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import zarankiewicz as Z

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "found_witnesses.json")


def matrix_blocks(A):
    m, n = A.shape
    return [tuple(i for i in range(m) if A[i, j]) for j in range(n)]


def coverage_of(blocks, m, s):
    cov = {c: 0 for c in combinations(range(m), s)}
    for b in blocks:
        if len(b) >= s:
            for c in combinations(b, s):
                cov[c] += 1
    return cov


def rebuild_dfs(m, s, t, cov, k, target, cand_blocks, deadline):
    """Exact DFS: choose k blocks (non-increasing size, lex within size) under
    capacity, total size >= target.  Returns list of blocks or None."""
    out = []

    def rec(slots, start_size_idx, start_idx, need):
        if time.time() > deadline:
            return False
        if slots == 0:
            return need <= 0
        for si in range(start_size_idx, len(cand_blocks)):
            size, lst = cand_blocks[si]
            if size * slots < need:
                return False
            for idx in range(start_idx if si == start_size_idx else 0,
                             len(lst)):
                blk, subs = lst[idx]
                if any(cov[c] >= t - 1 for c in subs):
                    continue
                for c in subs:
                    cov[c] += 1
                out.append(blk)
                if rec(slots - 1, si, idx, need - size):
                    return True
                out.pop()
                for c in subs:
                    cov[c] -= 1
        return False

    if rec(k, 0, 0, target):
        return list(out)
    return None


def improve(m, n, s, t, A0, z, seconds=20.0, seed=12345, kmax=4):
    rng = random.Random(seed)
    blocks = matrix_blocks(A0)
    best = sum(len(b) for b in blocks)
    sizes = {}
    total_cap = (t - 1) * comb(m, s)
    for bsz in range(s, m + 1):
        if comb(bsz, s) > total_cap:
            break
        sizes[bsz] = [(blk, list(combinations(blk, s)))
                      for blk in combinations(range(m), bsz)]
    cand_blocks = sorted(sizes.items(), key=lambda kv: -kv[0])
    deadline = time.time() + seconds
    while time.time() < deadline and best < z:
        k = rng.randint(2, kmax)
        idxs = rng.sample(range(n), k)
        kept = [b for j, b in enumerate(blocks) if j not in set(idxs)]
        removed_sz = sum(len(blocks[j]) for j in idxs)
        cov = coverage_of(kept, m, s)
        need = removed_sz + 1
        got = rebuild_dfs(m, s, t, cov, k, need, cand_blocks,
                          min(deadline, time.time() + 3.0))
        if got:
            blocks = kept + got
            best = sum(len(b) for b in blocks)
            print(f"    improved to {best}")
    A = Z.blocks_to_matrix(m, n, blocks)
    return A, best


def main():
    ev_path = os.path.join(os.path.dirname(HERE), "harness",
                           "evaluator_snapshot.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("snap", ev_path)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    targets = []
    for (m, n), z in sorted(ev.KST_EXACT_VALUE.items()):
        A, prov, _ = Z.construct(m, n, 3, 3)
        e = int(A.sum())
        if e < z:
            targets.append((m, n, z, e, A))
    print("missing cells:", [(m, n, z - e) for m, n, z, e, _ in targets])

    found = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            found = json.load(f)
    for m, n, z, e, A in targets:
        key = f"{m},{n}"
        if key in found and found[key]["edges"] >= z:
            continue
        print(f"cell {m}x{n}: {e} -> target {z}")
        bestA, beste = A, e
        for seed in (1, 2, 3, 4, 5):
            A2, e2 = improve(m, n, 3, 3, bestA, z,
                             seconds=25.0, seed=seed)
            if e2 > beste:
                bestA, beste = A2, e2
            if beste >= z:
                break
        ok = Z.verify_kst_free(bestA, 3, 3)
        ok_ref = ev.count_kst_violations(bestA, 3, 3) == 0
        print(f"  -> {beste}/{z} verified={ok and ok_ref}")
        if ok and ok_ref and beste > e:
            found[key] = {"m": m, "n": n, "edges": beste,
                          "exact": z,
                          "blocks": [list(b) for b in matrix_blocks(bestA)]}
            with open(OUT, "w") as f:
                json.dump(found, f, indent=1)
    print("saved", OUT)


if __name__ == "__main__":
    main()

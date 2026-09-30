"""E24 / A2-lookahead: compute zar_ub.hardness_lookahead features on the DEV set.

Sample (seed 24, per cell of the ground-truth file):
  * H  exact hard   : every case with status unsat/sat and d > 20000 (all of them)
  * M  exact mid    : status unsat/sat, 2000 < d <= 20000, up to --mid-per-cell (150) per cell
  * C  censored hard: status unknown, cap >= 20000, d >= 20000 (a lower bound), up to
                      --cens-per-cell (100) per cell -- used only for the hard-vs-mid AUC
                      on tables that are censored today (target cells)
Writes one JSON line per case: the ground-truth row + {"subset", "features", "d_hat", cost_*}.
Resumable: cases already in the output file are skipped.

usage: python lookahead_features.py [--gt ground_truth_initial.jsonl] [--out lookahead_features_dev.jsonl]
                                    [--jobs 2] [--mid-per-cell 150] [--cens-per-cell 100]
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(os.path.dirname(HERE))
if UB not in sys.path:
    sys.path.insert(0, UB)

from zar_ub.hardness_lookahead import estimate  # noqa: E402
from zar_ub.known import Instance  # noqa: E402


def key(r):
    return (r["cell"], tuple(r["rows"]), tuple(r["cols"]))


def select(rows, mid_per_cell, cens_per_cell, seed=24):
    by_cell = collections.defaultdict(lambda: {"H": [], "M": [], "C": []})
    for r in rows:
        d, st = r["d"], r["status"]
        if d is None:
            continue
        if st in ("unsat", "sat") and d > 20000:
            by_cell[r["cell"]]["H"].append(r)
        elif st in ("unsat", "sat") and 2000 < d <= 20000:
            by_cell[r["cell"]]["M"].append(r)
        elif st == "unknown" and r["cap"] >= 20000 and d >= 20000:
            by_cell[r["cell"]]["C"].append(r)
    out = []
    for cell in sorted(by_cell):
        rng = random.Random(f"{seed}:{cell}")
        g = by_cell[cell]
        out += [("H", r) for r in g["H"]]
        mids = list(g["M"])
        rng.shuffle(mids)
        out += [("M", r) for r in mids[:mid_per_cell]]
        cens = list(g["C"])
        rng.shuffle(cens)
        out += [("C", r) for r in cens[:cens_per_cell]]
    return out


def _work(args):
    sub, r, kw = args
    inst = Instance(r["m"], r["n"], r["s"], r["t"], r["w"])
    e = estimate(inst, r["rows"], r["cols"], **kw)
    o = dict(r)
    o["subset"] = sub
    o.update(e)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=os.path.join(HERE, "ground_truth_initial.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "lookahead_features_dev.jsonl"))
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--mid-per-cell", type=int, default=150)
    ap.add_argument("--cens-per-cell", type=int, default=100)
    ap.add_argument("--only-subsets", default="HMC")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.gt)]
    sel = [(s, r) for s, r in select(rows, a.mid_per_cell, a.cens_per_cell) if s in a.only_subsets]
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            try:
                done.add(key(json.loads(l)))
            except ValueError:
                pass
    todo = [(s, r, {}) for s, r in sel if key(r) not in done]
    print(
        f"selected {len(sel)} ({collections.Counter(s for s, _ in sel)}), todo {len(todo)}",
        flush=True,
    )
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs) as pool, open(a.out, "a") as f:
        for k, o in enumerate(pool.imap_unordered(_work, todo, chunksize=4)):
            f.write(json.dumps(o) + "\n")
            if (k + 1) % 200 == 0:
                f.flush()
                print(f"  {k+1}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
    print(f"done {len(todo)} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

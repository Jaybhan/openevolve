"""E24 / A2-lookahead: cheap FL-only pass (no pairs, no Knuth, no clause measures) over the cases
in a features file.  Adds the FL-reduced volumes (fl_log2vol_rows/cols, added to the module after
the main run started) and measures the cost of the cheap "FL-only" estimator variant.
Writes <in>_fl.jsonl: {"cell","rows","cols","features_fl", "cost_fl_*"}.

usage: python lookahead_supplement.py [--in lookahead_features_dev.jsonl] [--jobs 2]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(os.path.dirname(HERE))
if UB not in sys.path:
    sys.path.insert(0, UB)

from zar_ub.hardness_lookahead import estimate  # noqa: E402
from zar_ub.known import Instance  # noqa: E402

FL_ONLY = dict(n_pairs=0, n_probes=0, clause_measures=False)


def _work(r):
    inst = Instance(r["m"], r["n"], r["s"], r["t"], r["w"])
    e = estimate(inst, r["rows"], r["cols"], **FL_ONLY)
    return {
        "cell": r["cell"],
        "rows": r["rows"],
        "cols": r["cols"],
        "features_fl": e["features"],
        "cost_fl_seconds": e["cost_seconds"],
        "cost_fl_calls": e["cost_calls"],
        "cost_fl_propagations": e["cost_propagations"],
        "cost_fl_conflicts": e["cost_conflicts"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(HERE, "lookahead_features_dev.jsonl"))
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    out = a.inp.replace(".jsonl", "_fl.jsonl")
    rs = [json.loads(l) for l in open(a.inp)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.jobs) as pool, open(out, "w") as f:
        for o in pool.imap(_work, rs, chunksize=16):
            f.write(json.dumps(o) + "\n")
    print(f"{len(rs)} cases in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()

"""Reproducibility check (A1): do the cached exact labels still reproduce with TODAY's
encode_case + pysat cadical195?  Seeded sample of exact (unsat/sat) records with
2000 < d <= 60000 from ground_truth_initial.jsonl, per label_mode; re-solve with a
fresh solver at cap 2*d and compare conflict counts.

usage: python gt_check_encoding.py [--n 60] [--jobs 7]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random

from gt_common import HERE, _worker, read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=7)
    a = ap.parse_args()
    rows = read_jsonl(os.path.join(HERE, "ground_truth_initial.jsonl"))
    pool_rows = [r for r in rows if r["status"] in ("unsat", "sat") and 2000 < r["d"] <= 60000]
    rng = random.Random(24)
    by_mode = {}
    for r in pool_rows:
        by_mode.setdefault(r.get("label_mode", "legacy"), []).append(r)
    sample = []
    for mode in sorted(by_mode):
        sample += rng.sample(by_mode[mode], min(a.n // len(by_mode), len(by_mode[mode])))
    args = [({k: r[k] for k in "mnstw"}, r["rows"], r["cols"], 2 * r["d"] + 100, 300.0, i) for i, r in enumerate(sample)]
    out = []
    with mp.get_context("fork").Pool(a.jobs) as pool:
        for i, rr, cc, res in pool.imap_unordered(_worker, args):
            r = sample[i]
            out.append({"cell": r["cell"], "rows": rr, "cols": cc, "label_mode": r.get("label_mode"),
                        "cached_status": r["status"], "cached_d": r["d"], "now_status": res["status"],
                        "now_conflicts": res["conflicts"], "same": res["status"] == r["status"] and res["conflicts"] == r["d"]})
    out.sort(key=lambda x: (x["cell"], x["rows"], x["cols"]))
    write_jsonl(os.path.join(HERE, "ground_truth_encoding_check.jsonl"), out)
    for mode in sorted(by_mode):
        o = [x for x in out if x["label_mode"] == mode]
        same = sum(x["same"] for x in o)
        st = sum(x["now_status"] == x["cached_status"] for x in o)
        print(f"{mode}: {len(o)} re-solved, identical conflicts {same}, same status {st}")
    for x in out:
        if not x["same"]:
            print("  DIFF", x["cell"], x["cached_status"], x["cached_d"], "->", x["now_status"], x["now_conflicts"])


if __name__ == "__main__":
    main()

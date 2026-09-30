"""E27 (integration): write the E24 ground truth back into the case tables, then relabel every
remaining censored case of the scored tables with the E24 hardness model.

usage (from examples/zarankiewicz/upper_bounds, ZAR_UB_NO_LLM=1):
    python experiments/E27_integration/relabel_tables.py writeback        # phase 0, no solver
    python experiments/E27_integration/relabel_tables.py run --procs 12   # phase 1, resumable
    python experiments/E27_integration/relabel_tables.py apply            # phase 2, no solver

Phase 0: `casetable.apply_ground_truth` with A1's rows of source "deepen" (exact unsat labels; the
         open-at-2M rows raise the lower bound).  Atomic saves, table_hash bumped.
Phase 1: `difficulty.relabel_probe` on every censored record of the TABLES below (library survivors
         first), 12 processes, one JSON line per case in relabel_results.jsonl (resumable: a case
         already in the file is skipped).  Cost per case is recorded in conflicts / propagations /
         solver-seconds.
Phase 2: `casetable.relabel(tab, results=...)` applies the lines (no solving), atomic save.

The originals are in experiments/E27_integration/cache_before/ (copied before phase 0).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from dataclasses import asdict

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, ROOT)

from zar_ub.casetable import CACHE_DIR, apply_ground_truth, relabel, table_from_json  # noqa: E402
from zar_ub.difficulty import _relabel_worker  # noqa: E402

GT = os.path.join(ROOT, "experiments", "E24_difficulty", "ground_truth.jsonl")
OUT = os.path.join(_HERE, "relabel_results.jsonl")
SUMMARY = os.path.join(_HERE, "relabel_summary.json")

# every table the E27 suite scores (suite.py: TRAIN incl. WIDE, BAND, TARGET incl. the later targets, GEN)
TABLES = [
    # TARGET (tan2022, censored labels)
    "m9_n23_s3_t3_w104", "m12_n18_s3_t3_w109", "m10_n23_s3_t3_w113", "m11_n23_s3_t3_w124",
    "m13_n19_s3_t3_w123", "m16_n17_s3_t3_w134", "m10_n22_s3_t3_w111",
    # BAND (pure, w = z+1)
    "m12_n13_s3_t3_w87_pure", "m13_n13_s3_t3_w93_pure", "m10_n14_s3_t3_w78_pure",
    # WIDE TRAIN (pure, w = z+1)
    "m9_n12_s3_t3_w64_pure", "m10_n20_s3_t3_w103_pure", "m11_n21_s3_t3_w117_pure",
    "m9_n16_s3_t3_w78_pure_gt", "m9_n18_s3_t3_w86_pure_gt", "m10_n19_s3_t3_w99_pure_gt",
    # square TRAIN (pure, w = z+1): expected to have no censored case
    "m9_n9_s3_t3_w50_pure", "m9_n10_s3_t3_w55_pure", "m10_n10_s3_t3_w61_pure", "m10_n11_s3_t3_w65_pure",
    "m11_n11_s3_t3_w70_pure", "m11_n12_s3_t3_w75_pure", "m12_n12_s3_t3_w81_pure",
]


def _path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"case_table_{key}.json")


def _load(key: str):
    p = _path(key)
    with open(p) as f:
        return table_from_json(json.load(f), p)


def writeback() -> None:
    by_cell = collections.defaultdict(list)
    with open(GT) as f:
        for line in f:
            r = json.loads(line)
            if r["source"] == "deepen":
                by_cell[r["cell"]].append(r)
    out = []
    for cell, rows in sorted(by_cell.items()):
        p = _path(cell)
        if not os.path.exists(p):
            print(f"[writeback] no table for {cell}", flush=True)
            continue
        tab = _load(cell)
        res = apply_ground_truth(tab, rows, save=True)
        print(json.dumps(res), flush=True)
        out.append(res)
    with open(os.path.join(_HERE, "writeback_summary.json"), "w") as f:
        json.dump(out, f, indent=1)


def _done() -> set:
    s = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    o = json.loads(line)
                    s.add((o["table"], o["idx"]))
                except (ValueError, KeyError):
                    pass
    return s


def run(procs: int) -> None:
    import multiprocessing as mp

    done = _done()
    jobs = []
    for key in TABLES:
        if not os.path.exists(_path(key)):
            continue
        tab = _load(key)
        mask = tab.baseline_lean_mask or [False] * len(tab.records)
        for i, r in enumerate(tab.records):
            if r.probe and r.probe.get("status") == "unknown" and (key, i) not in done:
                jobs.append((0 if not mask[i] else 1, key, i, asdict(tab.instance), r.rows, r.cols, r.probe))
    jobs.sort(key=lambda j: (j[0], j[1], j[2]))  # library survivors first
    print(f"[run] {len(jobs)} censored cases to relabel ({len(done)} already done), {procs} processes", flush=True)
    t0 = time.time()
    n = 0
    with mp.get_context("fork").Pool(procs) as pool, open(OUT, "a") as fh:
        it = pool.imap_unordered(_worker, jobs, chunksize=1)
        for o in it:
            fh.write(json.dumps(o) + "\n")
            fh.flush()
            n += 1
            if n % 250 == 0:
                el = time.time() - t0
                print(f"[run] {n}/{len(jobs)} done, {el/60:.1f} min elapsed, ETA {el/n*(len(jobs)-n)/60:.1f} min",
                      flush=True)
    print(f"[run] finished {n} cases in {(time.time()-t0)/60:.1f} min wall", flush=True)


def _worker(job):
    surv, key, i, inst_d, rows, cols, probe = job
    t = time.time()
    o = _relabel_worker((inst_d, i, rows, cols, probe))
    o.update(table=key, survivor=(surv == 0), wall=time.time() - t)
    return o


def apply() -> None:
    res = collections.defaultdict(dict)
    with open(OUT) as f:
        for line in f:
            o = json.loads(line)
            res[o["table"]][o["idx"]] = o
    summ = {"tables": [], "totals": collections.Counter()}
    for key in TABLES:
        if key not in res:
            continue
        tab = _load(key)
        r = relabel(tab, results=res[key], save=True)
        vals = list(res[key].values())
        r.update(table=key, n_results=len(vals),
                 survivors=sum(1 for o in vals if o.get("survivor")),
                 cost_conflicts=sum(int(o.get("cost_conflicts", 0)) for o in vals),
                 cost_propagations=sum(int(o.get("cost_propagations", 0)) for o in vals),
                 cost_solver_seconds=round(sum(float(o.get("cost_seconds", 0.0)) for o in vals), 1),
                 reprobe_decided=sum(1 for o in vals if "reprobe_decided" in o))
        print(json.dumps(r), flush=True)
        summ["tables"].append(r)
        for k in ("n_results", "cost_conflicts", "cost_propagations", "cost_solver_seconds", "reprobe_decided"):
            summ["totals"][k] += r[k]
    summ["totals"] = dict(summ["totals"])
    with open(SUMMARY, "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ["totals"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["writeback", "run", "apply"])
    ap.add_argument("--procs", type=int, default=12)
    a = ap.parse_args()
    {"writeback": writeback, "run": lambda: run(a.procs), "apply": apply}[a.phase]()

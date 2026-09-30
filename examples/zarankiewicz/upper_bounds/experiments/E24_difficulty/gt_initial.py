"""Step 1 (A1): ground_truth_initial.jsonl from every cached probe.

  * every probed record of every cache/case_table_*.json (never modified; read only)
  * status unsat/sat -> d = exact conflicts of the probe, cap = budget_cap
  * status unknown   -> d = conflicts reached (a LOWER bound; = cap up to a few conflicts)
  * the E10 results (1,571 exact labels, 7 TRAIN cells) are cross-checked against the
    pure tables they were written back into; c2000 is taken from E10 when present
  * c2000 missing -> derived (a probe refuted below 2000 conflicts IS the 2000-cap run)
    or measured by a fresh 2000-cap solve (60 s wall limit), 7 processes

usage: python gt_initial.py [--jobs 7]
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict

from gt_common import CACHE, HERE, UB, Instance, _worker, gt_row, key_of, write_jsonl

E10 = os.path.join(UB, "experiments", "E10_difficulty", "results.json")
OUT = os.path.join(HERE, "ground_truth_initial.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    a = ap.parse_args()
    t0 = time.time()
    e10 = {}
    for r in json.load(open(E10))["rows"]:
        e10[(tuple(r["cell"]), tuple(r["rows"]), tuple(r["cols"]))] = r

    rows_out = []
    todo = []  # (index into rows_out, inst, rows, cols)
    e10_seen, e10_mismatch = 0, []
    files = sorted(p for p in glob.glob(os.path.join(CACHE, "case_table_*.json")) if not p.endswith("_gt.json"))
    for p in files:
        d = json.load(open(p))
        inst = Instance(**d["inst"])
        trust = d.get("trust") or ("tan2022" if d.get("use_table", True) else "pure")
        mask = d.get("baseline_lean_mask")
        if mask is not None and len(mask) != len(d["records"]):
            mask = None
        for i, r in enumerate(d["records"]):
            pr = r.get("probe")
            if not pr or pr.get("status") not in ("unsat", "sat", "unknown"):
                continue
            st = pr["status"]
            conf = int(pr.get("conflicts", 0))
            cap = int(pr.get("budget_cap") or 0)
            ek = ((inst.m, inst.n), tuple(r["rows"]), tuple(r["cols"]))
            er = e10.get(ek) if trust == "pure" else None
            c2000 = pr.get("c2000")
            if er is not None:
                e10_seen += 1
                if er["status"] != st or int(er["true_conflicts"]) != conf:
                    e10_mismatch.append((os.path.basename(p), r["rows"], r["cols"], st, conf, er["status"], er["true_conflicts"]))
                if c2000 is None:
                    c2000 = int(er["c2000"])
            if c2000 is None and st in ("unsat", "sat") and conf < 2000 and cap >= 2000:
                c2000 = conf  # refuted/solved inside the first 2000 conflicts: the same run
            if st == "unknown":
                dval = conf if conf > 0 else cap
            else:
                dval = max(1, conf)
            row = gt_row(
                inst, trust, r["rows"], r["cols"], st, dval, cap, c2000, "table",
                table=os.path.basename(p), kind=d.get("kind", ""),
                baseline_lean_kill=(bool(mask[i]) if mask is not None else None),
                label_mode=d.get("label_mode", "legacy"),
                c2000_source=("probe" if pr.get("c2000") is not None else ("e10" if er is not None else ("derived" if c2000 is not None else None))),
            )
            rows_out.append(row)
            if c2000 is None:
                todo.append((len(rows_out) - 1, inst, r["rows"], r["cols"]))

    # duplicates (a case can occur in two tables only if the tables share an instance+trust; they do not,
    # but guard anyway: keep the deepest label)
    seen = {}
    for k, row in enumerate(rows_out):
        kk = key_of(row)
        if kk in seen:
            j = seen[kk]
            if rows_out[j]["status"] == "unknown" and row["status"] != "unknown" or row["cap"] > rows_out[j]["cap"]:
                seen[kk] = k
        else:
            seen[kk] = k
    keep = set(seen.values())

    print(f"{len(rows_out)} records from {len(files)} tables; E10 cross-checked {e10_seen}, mismatches {len(e10_mismatch)}; "
          f"c2000 to measure: {len(todo)}", flush=True)
    for mm in e10_mismatch[:20]:
        print("  E10 MISMATCH", mm, flush=True)

    cost = {"conflicts": 0, "propagations": 0, "seconds": 0.0, "n": 0}
    if todo:
        args = [(asdict(inst), rr, cc, 2000, 60.0, k) for k, inst, rr, cc in todo]
        with mp.get_context("fork").Pool(a.jobs) as pool:
            for n_done, (k, rr, cc, res) in enumerate(pool.imap_unordered(_worker, args, chunksize=8)):
                rows_out[k]["c2000"] = int(res["conflicts"])
                rows_out[k]["c2000_source"] = "measured" + ("_timeout" if res["budget_hit"] == "time" else "")
                if res["status"] != "unknown" and rows_out[k]["status"] != res["status"]:
                    rows_out[k]["c2000_status_conflict"] = res["status"]
                cost["conflicts"] += res["conflicts"]
                cost["propagations"] += res["propagations"]
                cost["seconds"] += res["seconds"]
                cost["n"] += 1
                if n_done % 500 == 0:
                    print(f"  c2000 {n_done}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
    final = [rows_out[k] for k in sorted(keep)]
    n = write_jsonl(OUT, final)
    summ = {
        "records": n,
        "e10_crosschecked": e10_seen,
        "e10_mismatches": [list(map(str, m)) for m in e10_mismatch],
        "c2000_measured": cost,
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_jsonl(os.path.join(HERE, "ground_truth_initial_summary.jsonl"), [summ])
    print(json.dumps(summ)[:2000], flush=True)


if __name__ == "__main__":
    main()

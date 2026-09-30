"""Steps 2+3 (A1): deepen censored cases and label the new wide pure tables.

  python gt_deepen.py plan          write ground_truth_jobs.jsonl (priority order, deterministic)
  python gt_deepen.py run --jobs 7  solve every job not yet in ground_truth_deepen_raw.jsonl
                                    (resumable; one JSON line appended per finished case)

Per job: when c2000 is unknown (new tables) a fresh 2,000-cap run first (60 s wall
limit); if that does not resolve the case, ONE fresh cadical195 run with cap
2,000,000 conflicts and a 180 s wall limit.  Conflict counts of fresh runs are
deterministic (checked: gt_check_encoding.py), so d is exact whenever status is
sat/unsat; a case still open is right-censored at the conflicts reached.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import asdict

from gt_common import CACHE, HERE, Instance, cell_tag, read_jsonl, solve_one

JOBS = os.path.join(HERE, "ground_truth_jobs.jsonl")
RAW = os.path.join(HERE, "ground_truth_deepen_raw.jsonl")
CAP, TL = 2_000_000, 180.0
FIRST_CAP, FIRST_TL = 2_000, 60.0
SEED = 2024

# (group, table file, how many censored cases: None = all)
DEEPEN = [
    ("a", "case_table_m10_n20_s3_t3_w103_pure.json", None),
    ("b", "case_table_m11_n21_s3_t3_w117_pure.json", None),
    ("c", "case_table_m9_n23_s3_t3_w104.json", None),
    ("d", "case_table_m10_n14_s3_t3_w78_pure.json", None),
    ("e", "case_table_m13_n13_s3_t3_w93_pure.json", None),
    ("f", "case_table_m12_n18_s3_t3_w109.json", 150),
    ("f", "case_table_m13_n19_s3_t3_w123.json", 150),
    ("f", "case_table_m16_n17_s3_t3_w134.json", 150),
    # cheap leftovers: every other censored case of an exact-valued cell
    ("g", "case_table_m10_n20_s3_t3_w103.json", None),
    ("g", "case_table_m10_n22_s3_t3_w111.json", None),
    ("g", "case_table_m12_n13_s3_t3_w87_pure.json", None),
    ("g", "case_table_m10_n11_s3_t3_w64_pure.json", None),
]
# step 3: new WIDE pure tables at w = z+1 (exact values from data/exact_33.csv)
NEW = [(9, 18), (10, 19), (9, 16)]
ORDER = ["a", "b", "c", "new", "d", "e", "f", "g"]


def new_cases(m, n):
    from zar_ub.known import Ledger, exact_value
    from zar_ub.partitions import column_partitions, row_partitions

    inst = Instance(m, n, 3, 3, exact_value(m, n, 3, 3) + 1)
    rp = row_partitions(inst, Ledger(), False)
    cp = column_partitions(inst, Ledger(), False)
    return inst, [(list(r), list(c)) for r in rp for c in cp]


def plan():
    groups = {}
    for g, fn, k in DEEPEN:
        d = json.load(open(os.path.join(CACHE, fn)))
        inst = Instance(**d["inst"])
        trust = d.get("trust") or ("tan2022" if d.get("use_table", True) else "pure")
        cens = [(i, r) for i, r in enumerate(d["records"]) if r.get("probe") and r["probe"]["status"] == "unknown"]
        if k is not None and k < len(cens):
            rng = random.Random(f"{SEED}:{fn}")
            cens = sorted(rng.sample(cens, k), key=lambda x: x[0])
        for i, r in cens:
            groups.setdefault(g, []).append({
                "group": g, "table": fn, "index": i, "cell": cell_tag(inst, trust), "inst": asdict(inst),
                "trust": trust, "rows": r["rows"], "cols": r["cols"], "c2000": r["probe"].get("c2000"),
                "prev_cap": r["probe"].get("budget_cap"), "prev_conflicts": r["probe"].get("conflicts"),
                "source": "deepen",
            })
    for m, n in NEW:
        inst, cases = new_cases(m, n)
        for i, (rr, cc) in enumerate(cases):
            groups.setdefault("new", []).append({
                "group": "new", "table": f"case_table_{inst.tag}_pure_gt.json", "index": i,
                "cell": cell_tag(inst, "pure"), "inst": asdict(inst), "trust": "pure", "rows": rr, "cols": cc,
                "c2000": None, "prev_cap": 0, "prev_conflicts": 0, "source": "new_table",
            })
    jobs = [j for g in ORDER for j in groups.get(g, [])]
    with open(JOBS, "w") as f:
        for j in jobs:
            f.write(json.dumps(j, separators=(",", ":")) + "\n")
    for g in ORDER:
        by = {}
        for j in groups.get(g, []):
            by[j["cell"]] = by.get(j["cell"], 0) + 1
        print(g, by)
    print("total jobs", len(jobs))


def _key(j):
    return (j["cell"], tuple(j["rows"]), tuple(j["cols"]))


def _run_job(j):
    inst = Instance(**j["inst"])
    out = dict(j)
    cost = {"conflicts": 0, "propagations": 0, "decisions": 0, "seconds": 0.0}
    first = None
    if j.get("c2000") is None:
        first = solve_one(inst, j["rows"], j["cols"], FIRST_CAP, FIRST_TL)
        for k in ("conflicts", "propagations", "decisions", "seconds"):
            cost[k] += first[k]
        out["c2000"] = first["conflicts"]
        out["c2000_timeout"] = first["budget_hit"] == "time"
    if first is not None and first["status"] != "unknown":
        res = first
    else:
        res = solve_one(inst, j["rows"], j["cols"], CAP, TL)
        for k in ("conflicts", "propagations", "decisions", "seconds"):
            cost[k] += res[k]
    out["status"] = res["status"]
    out["d"] = max(1, res["conflicts"])
    out["cap"] = res["cap"]
    out["budget_hit"] = res["budget_hit"]
    out["run"] = {k: res[k] for k in ("conflicts", "propagations", "decisions", "seconds")}
    out["cost"] = cost
    if res["status"] == "sat":
        out["witness_ok"] = res.get("witness_ok")
        out["matrix"] = res.get("matrix")
        out["ones"] = res.get("ones")
    out["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return out


def run(jobs_n, groups=None):
    jobs = read_jsonl(JOBS)
    done = {_key(r) for r in read_jsonl(RAW)}
    todo = [j for j in jobs if _key(j) not in done and (groups is None or j["group"] in groups)]
    print(f"{len(jobs)} jobs, {len(done)} done, {len(todo)} to run with {jobs_n} processes", flush=True)
    t0 = time.time()
    with open(RAW, "a") as f, mp.get_context("fork").Pool(jobs_n, maxtasksperchild=50) as pool:
        for k, out in enumerate(pool.imap_unordered(_run_job, todo, chunksize=1)):
            f.write(json.dumps(out, separators=(",", ":")) + "\n")
            f.flush()
            if out["status"] == "sat":
                print(f"!!! SAT {out['cell']} rows={out['rows']} cols={out['cols']} witness_ok={out.get('witness_ok')} "
                      f"ones={out.get('ones')}", flush=True)
            if k % 25 == 0 or out["status"] == "sat":
                print(f"[{k + 1}/{len(todo)} {time.time() - t0:.0f}s] {out['group']} {out['cell']} {out['status']} "
                      f"d={out['d']} {out['run']['seconds']:.1f}s hit={out['budget_hit']}", flush=True)
    print("done", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run"])
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--groups", default=None)
    a = ap.parse_args()
    if a.cmd == "plan":
        plan()
    else:
        run(a.jobs, a.groups.split(",") if a.groups else None)

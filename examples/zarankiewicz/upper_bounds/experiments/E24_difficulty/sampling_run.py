"""E24 / A3: run the Chivilikhin-style sampling estimator on a case set, one record per
(case, config), storing every cube's raw outcome at the maximal (N, budget) so that
smaller N (prefix of an i.i.d. sample) and smaller budgets (re-censoring of a
deterministic fresh CaDiCaL run) are derived offline by sampling_eval.py.

usage:
  python experiments/E24_difficulty/sampling_run.py --cases e10|dev|<path.jsonl> \
      --configs screen|final|<name,...> --out <out.jsonl> [--jobs 3] [--limit-hard H] [--mid-per-cell M]

Resumable: (case key, config) pairs already present in --out are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(os.path.dirname(HERE))
if UB not in sys.path:
    sys.path.insert(0, UB)

from zar_ub.known import Instance, exact_value  # noqa: E402
from zar_ub import hardness_sampling as hs  # noqa: E402

NMAX, BMAX = 100, 10_000
E10 = os.path.join(UB, "experiments", "E10_difficulty", "results.json")
DEV = os.path.join(HERE, "ground_truth_initial.jsonl")
FINAL = os.path.join(HERE, "ground_truth.jsonl")
DEEPEN_RAW = os.path.join(HERE, "ground_truth_deepen_raw.jsonl")


def configs(name: str):
    out = []
    if name in ("screen", "uniform"):
        for d in ("row", "random", "lookahead"):
            for k in (6, 8, 10, 12):
                out.append(("uniform", d, k))
        for k in (1, 2):
            out.append(("uniform", "support", k))
    if name in ("screen", "knuth"):
        for d in ("row", "random", "lookahead"):
            for k in (4, 6, 8, 10, 12):
                out.append(("knuth", d, k))
    if name == "final":
        out = [("knuth", "row", k) for k in (2, 3, 4, 6, 8)] + [("knuth", "lookahead", k) for k in (4, 6, 8)]
    if not out:
        for tok in name.split(","):
            parts = tok.split(":")
            out.append((parts[0], parts[1], int(parts[2])) + ((int(parts[3]),) if len(parts) > 3 else ()))
    return out


def load_cases(which: str, limit_hard: int, mid_per_cell: int, seed: int = 0, hard_per_cell: int = 0,
               exclude_cells=(), cells=(), source="", include_censored=False):
    """Normalised dicts {cell, m,n,s,t,w, rows, cols, status, d, exact}.
    HARD = d > 20000 (all kept, up to limit_hard, seeded); MID = 2000 < d <= 20000
    (up to mid_per_cell per cell, seeded)."""
    rows = []
    if which == "e10":
        for r in json.load(open(E10))["rows"]:
            m, n = r["cell"]
            w = exact_value(m, n, 3, 3) + 1
            rows.append(dict(cell=f"m{m}_n{n}_s3_t3_w{w}_pure", m=m, n=n, s=3, t=3, w=w, rows=r["rows"],
                             cols=r["cols"], status=r["status"], d=r["true_conflicts"], exact=True,
                             c2000=r["c2000"], log2_volume=r["log2_volume"]))
    else:
        paths = {"dev": [DEV], "deepen": [DEEPEN_RAW], "final": [FINAL]}.get(which, which.split(","))
        seen = set()
        for path in paths:
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if "inst" in r and "m" not in r:
                    r.update(r["inst"])
                r["exact"] = r.get("status") in ("unsat", "sat")
                if cells and r["cell"] not in cells:
                    continue
                if source and r.get("source") != source:
                    continue
                if not r["exact"] and not (include_censored and r["d"] >= 1_000_000):
                    continue  # regimes are defined on exact labels (+ optionally right-censored at >= 1M)
                k = key_of(r)
                if k in seen:
                    continue
                seen.add(k)
                rows.append(r)
    rows = [r for r in rows if r["cell"] not in set(exclude_cells)]
    rng = random.Random(seed)
    hard = [r for r in rows if r["d"] > 20000]
    if hard_per_cell:
        hc = []
        for c in sorted(set(r["cell"] for r in hard)):
            ch = [r for r in hard if r["cell"] == c]
            if len(ch) > hard_per_cell:
                ch = random.Random(f"hard|{seed}|{c}").sample(ch, hard_per_cell)
            hc.extend(ch)
        hard = hc
    if limit_hard and len(hard) > limit_hard:
        hard = rng.sample(hard, limit_hard)
    mid = []
    cells = sorted(set(r["cell"] for r in rows))
    for c in cells:
        cm = [r for r in rows if r["cell"] == c and 2000 < r["d"] <= 20000]
        if len(cm) > mid_per_cell:
            cm = random.Random(f"{seed}|{c}").sample(cm, mid_per_cell)
        mid.extend(cm)
    return hard + mid


def key_of(r):
    return f'{r["cell"]}|{",".join(map(str, r["rows"]))}|{",".join(map(str, r["cols"]))}'


def _work(args):
    case, cfg, seed, NMAX, BMAX = args
    sampler, design, k = cfg[:3]
    depth = cfg[3] if len(cfg) > 3 else 3
    name = ":".join(map(str, cfg))
    inst = Instance(case["m"], case["n"], case["s"], case["t"], case["w"])
    t0 = time.time()
    try:
        e = hs.estimate(inst, case["rows"], case["cols"], design=design, k=k, N=NMAX, budget=BMAX, seed=seed,
                        sampler=sampler, strata_depth=depth, return_records=True,
                        total_budget=None, calibrated=False)
    except Exception as ex:  # keep going, report
        return dict(key=key_of(case), config=name, error=repr(ex))
    recs = [[r["conflicts"], r["propagations"], int(r["censored"]), r.get("weight", 1.0), int(r["up_refuted"]),
             r["status"][0]] + ([r["stratum"]] if "stratum" in r else []) for r in e["records"]]
    return dict(key=key_of(case), cell=case["cell"], d=case["d"], exact=case["exact"], status=case.get("status"),
                config=name, n_strata=e["features"].get("n_strata"), log2_space=e["features"]["log2_space"],
                nmax=NMAX, bmax=BMAX,
                extra_props=int(e["cost_propagations"] - sum(r[1] for r in recs)),
                cost_conflicts=e["cost_conflicts"], cost_propagations=e["cost_propagations"],
                cost_seconds=round(time.time() - t0, 3), recs=recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="e10")
    ap.add_argument("--configs", default="screen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--limit-hard", type=int, default=0)
    ap.add_argument("--mid-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nmax", type=int, default=NMAX)
    ap.add_argument("--bmax", type=int, default=BMAX)
    ap.add_argument("--regimes", default="hard,mid")
    ap.add_argument("--hard-per-cell", type=int, default=0)
    ap.add_argument("--exclude-cells-from", default="", help="skip cells present (exact) in this jsonl")
    ap.add_argument("--cells", default="", help="comma-separated cell tags to keep")
    ap.add_argument("--source", default="", help="keep only ground-truth rows with this source (e.g. deepen)")
    ap.add_argument("--include-censored", action="store_true",
                    help="also keep rows open at >= 1M conflicts (status unknown, d = lower bound)")
    a = ap.parse_args()
    excl = set()
    if a.exclude_cells_from:
        for line in open(a.exclude_cells_from):
            try:
                excl.add(json.loads(line)["cell"])
            except (ValueError, KeyError):
                pass
    cases = load_cases(a.cases, a.limit_hard, a.mid_per_cell, hard_per_cell=a.hard_per_cell, exclude_cells=excl,
                       cells=tuple(c for c in a.cells.split(",") if c), source=a.source,
                       include_censored=a.include_censored)
    cfgs = configs(a.configs)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                r = json.loads(line)
                done.add((r["key"], r["config"]))
            except ValueError:
                pass
    regs = a.regimes.split(",")
    cases = [c for c in cases if ("hard" in regs and c["d"] > 20000) or ("mid" in regs and 2000 < c["d"] <= 20000)]
    tasks = [(c, cfg, a.seed, a.nmax, a.bmax) for cfg in cfgs for c in cases if (key_of(c), ":".join(map(str, cfg))) not in done]
    print(f"{len(cases)} cases x {len(cfgs)} configs; {len(tasks)} tasks to run", flush=True)
    t0 = time.time()
    with Pool(a.jobs) as pool, open(a.out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(_work, tasks, chunksize=1)):
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

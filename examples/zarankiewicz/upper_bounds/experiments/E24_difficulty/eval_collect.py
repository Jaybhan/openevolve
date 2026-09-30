"""E24 / D5-evaluate: compute every estimator's features on the evaluation set.

Evaluation set (deterministic):
  HARD = every ground-truth row with d > 20000 that is exact (unsat/sat) or open at the 2M cap
         (status unknown, cap >= 2,000,000).  Rows open only at 20k are NOT included (their d is
         only known to be > 20k, which carries no ranking information inside the hard regime).
  MID  = exact 2000 < d <= 20000, a seeded sample of up to 120 per cell (seed "E24eval|<cell>").

Estimators (each module at its recommended setting, estimator contract):
  lookahead  zar_ub.hardness_lookahead.estimate(inst, rows, cols)            (defaults: full set)
  sampling   zar_ub.hardness_sampling.estimate(inst, rows, cols)             (DEFAULT: knuth:row:8,
             N100, b5000, total_budget 50k)
  progress   zar_ub.hardness_progress.estimate(inst, rows, cols, tier="20k") (binary + pysat at
             2k and 20k + static/LP features; the "free20k" subset needs no binary)
The fhat baseline and the plain c2000/c20000 probes need no solver: they are read from the ground
truth (c2000 field; c20000 = min(d, 20000), deterministic solver) in eval_analyze.py.

Output: features_{lookahead,sampling,progress}.jsonl (one line per case; resumable).
Usage:  python experiments/E24_difficulty/eval_collect.py --procs 10 [--limit N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UB)

GT = os.path.join(HERE, "ground_truth.jsonl")
GT_INITIAL = os.path.join(HERE, "ground_truth_initial.jsonl")
EVALSET = os.path.join(HERE, "features_evalset.jsonl")
ESTS = ("progress", "sampling", "lookahead")
MID_PER_CELL = 120
HARD_D = 20000
OPEN_CAP = 2_000_000


def case_key(cell, rows, cols) -> str:
    return f"{cell}|{','.join(map(str, rows))}|{','.join(map(str, cols))}"


def load_gt(path=None):
    path = path or (GT if os.path.exists(GT) else GT_INITIAL)
    out = []
    with open(path) as fh:
        for line in fh:
            out.append(json.loads(line))
    return out, path


def regime(r) -> str:
    st, d, cap = r["status"], float(r["d"]), int(r.get("cap") or 0)
    if st == "unknown":
        return "hard_open" if cap >= OPEN_CAP else "open20k"
    if d > HARD_D:
        return "hard"
    if d > 2000:
        return "mid"
    return "easy"


def build_evalset(gt):
    hard, mid = [], defaultdict(list)
    seen = set()
    for r in gt:
        k = case_key(r["cell"], r["rows"], r["cols"])
        if k in seen:
            continue
        seen.add(k)
        g = regime(r)
        if g in ("hard", "hard_open"):
            hard.append((r, g))
        elif g == "mid":
            mid[r["cell"]].append(r)
    out = [(r, g) for r, g in hard]
    for cell in sorted(mid):
        rs = sorted(mid[cell], key=lambda r: case_key(r["cell"], r["rows"], r["cols"]))
        rng = random.Random(f"E24eval|{cell}")
        pick = rs if len(rs) <= MID_PER_CELL else rng.sample(rs, MID_PER_CELL)
        out.extend((r, "mid") for r in pick)
    rows = []
    for r, g in out:
        rows.append({
            "key": case_key(r["cell"], r["rows"], r["cols"]),
            "cell": r["cell"], "m": r["m"], "n": r["n"], "s": r["s"], "t": r["t"], "w": r["w"],
            "trust": r.get("trust"), "rows": r["rows"], "cols": r["cols"], "status": r["status"],
            "d": r["d"], "cap": r.get("cap"), "c2000": r.get("c2000"), "log2_volume": r.get("log2_volume"),
            "regime": g, "baseline_lean_kill": r.get("baseline_lean_kill"), "source": r.get("source"),
        })
    return rows


def _work(task):
    est, rec = task
    os.environ.setdefault("ZAR_UB_NO_LLM", "1")
    from zar_ub.known import Instance
    inst = Instance(rec["m"], rec["n"], rec["s"], rec["t"], rec["w"])
    t0 = time.time()
    try:
        if est == "lookahead":
            from zar_ub import hardness_lookahead as mod
            o = mod.estimate(inst, rec["rows"], rec["cols"])
        elif est == "sampling":
            from zar_ub import hardness_sampling as mod
            o = mod.estimate(inst, rec["rows"], rec["cols"])
        elif est == "progress":
            from zar_ub import hardness_progress as mod
            o = mod.estimate(inst, rec["rows"], rec["cols"], tier="20k")
        else:
            raise ValueError(est)
        res = {
            "key": rec["key"], "cell": rec["cell"], "est": est,
            "features": o["features"], "d_hat": o.get("d_hat"), "d_hat_raw": o.get("d_hat_raw"),
            "cost_conflicts": int(o.get("cost_conflicts") or 0),
            "cost_propagations": int(o.get("cost_propagations") or 0),
            "cost_seconds": float(o.get("cost_seconds") or (time.time() - t0)),
        }
        if est == "progress":
            res["meta"] = o.get("meta")
    except Exception as e:  # noqa: BLE001
        res = {"key": rec["key"], "cell": rec["cell"], "est": est, "error": repr(e),
               "cost_seconds": time.time() - t0}
    return res


def expected_cost(rec) -> float:
    return rec["m"] * rec["n"] * (3.0 if rec["regime"] != "mid" else 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ests", default=",".join(ESTS))
    args = ap.parse_args()
    gt, path = load_gt()
    print(f"ground truth: {path} ({len(gt)} rows)", flush=True)
    ev = build_evalset(gt)
    with open(EVALSET, "w") as fh:
        for r in ev:
            fh.write(json.dumps(r) + "\n")
    cnt = defaultdict(int)
    for r in ev:
        cnt[r["regime"]] += 1
    print("evaluation set:", dict(cnt), "total", len(ev), flush=True)
    ests = [e for e in args.ests.split(",") if e]
    done = {e: set() for e in ests}
    for e in ests:
        p = os.path.join(HERE, f"features_{e}.jsonl")
        if os.path.exists(p):
            with open(p) as fh:
                for line in fh:
                    try:
                        o = json.loads(line)
                    except ValueError:
                        continue
                    if "error" not in o:
                        done[e].add(o["key"])
    order = sorted(ev, key=lambda r: (-expected_cost(r), r["key"]))
    tasks = [(e, r) for r in order for e in ests if r["key"] not in done[e]]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"tasks: {len(tasks)}", flush=True)
    fhs = {e: open(os.path.join(HERE, f"features_{e}.jsonl"), "a") for e in ests}
    t0 = time.time()
    agg = defaultdict(lambda: [0, 0, 0, 0.0])
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.procs) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, tasks, chunksize=1), 1):
            fhs[res["est"]].write(json.dumps(res) + "\n")
            fhs[res["est"]].flush()
            a = agg[res["est"]]
            a[0] += 1
            a[1] += res.get("cost_conflicts", 0)
            a[2] += res.get("cost_propagations", 0)
            a[3] += res.get("cost_seconds", 0.0)
            if "error" in res:
                print("ERROR", res["est"], res["key"], res["error"], flush=True)
            if i % 200 == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"{i}/{len(tasks)} {el:.0f}s eta {el / i * (len(tasks) - i):.0f}s "
                      + " ".join(f"{e}:n{a[0]} c{a[1] / max(1, a[0]):.0f} s{a[3] / max(1, a[0]):.2f}" for e, a in agg.items()),
                      flush=True)
    for fh in fhs.values():
        fh.close()


if __name__ == "__main__":
    main()

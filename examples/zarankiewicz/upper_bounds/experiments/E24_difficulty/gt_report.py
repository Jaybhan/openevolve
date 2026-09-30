"""Step 4 (A1): merge ground_truth_initial.jsonl + ground_truth_deepen_raw.jsonl into
ground_truth.jsonl (contract fields + extras) and print per-cell statistics.

A deepen result REPLACES the initial (table) row of the same (cell, rows, cols);
new-table rows are added.  Regimes: easy d <= 2000, mid 2000 < d <= 20000,
hard d > 20000 (exact), hard_censored = status unknown at a cap >= 20000 (so true d > 20000),
low_censored = status unknown at a cap < 20000 (regime undetermined).

usage: python gt_report.py [--out ground_truth.jsonl] [--md]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import OrderedDict, defaultdict

from gt_common import HERE, Instance, gt_row, key_of, read_jsonl, write_jsonl

INIT = os.path.join(HERE, "ground_truth_initial.jsonl")
RAW = os.path.join(HERE, "ground_truth_deepen_raw.jsonl")


def regime(r):
    d = r["d"]
    if r["status"] == "unknown":  # right-censored: true d > conflicts reached (~ cap)
        return "hard_censored" if max(d, r["cap"]) >= 20000 else "low_censored"
    if d <= 2000:
        return "easy"
    return "mid" if d <= 20000 else "hard"


def merged():
    rows = OrderedDict()
    for r in read_jsonl(INIT):
        rows[key_of(r)] = r
    raw = read_jsonl(RAW)
    for j in raw:
        inst = Instance(**j["inst"])
        prev = rows.get((j["cell"], tuple(j["rows"]), tuple(j["cols"])))
        extra = {
            "table": j["table"],
            "group": j["group"],
            "kind": (prev or {}).get("kind", "train_candidate" if j["source"] == "new_table" else ""),
            "baseline_lean_kill": (prev or {}).get("baseline_lean_kill"),
            "prev_cap": j.get("prev_cap"),
            "prev_conflicts": j.get("prev_conflicts"),
            "budget_hit": j.get("budget_hit"),
            "run_seconds": j["run"]["seconds"],
            "run_propagations": j["run"]["propagations"],
            "cost_conflicts": j["cost"]["conflicts"],
            "cost_propagations": j["cost"]["propagations"],
            "cost_seconds": round(j["cost"]["seconds"], 3),
        }
        if j["status"] == "sat":
            extra["witness_ok"] = j.get("witness_ok")
            extra["ones"] = j.get("ones")
        r = gt_row(inst, j["trust"], j["rows"], j["cols"], j["status"], j["d"], j["cap"], j.get("c2000"), j["source"], **extra)
        rows[key_of(r)] = r
    return list(rows.values()), raw


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def stats(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r)
    out = []
    for cell in sorted(by, key=lambda c: (by[c][0]["m"], by[c][0]["n"], by[c][0]["w"], by[c][0]["trust"])):
        rs = by[cell]
        reg = defaultdict(int)
        st = defaultdict(int)
        for r in rs:
            reg[regime(r)] += 1
            st[r["status"]] += 1
        ex = [r["d"] for r in rs if r["status"] != "unknown"]
        out.append({
            "cell": cell, "trust": rs[0]["trust"], "cases": len(rs), "status": dict(st),
            "easy": reg["easy"], "mid": reg["mid"], "hard_exact": reg["hard"], "hard_censored": reg["hard_censored"],
            "low_censored": reg["low_censored"],
            "d_median_exact": q(ex, 0.5), "d_p90_exact": q(ex, 0.9), "d_max_exact": max(ex) if ex else None,
            "sources": dict((s, sum(1 for r in rs if r["source"] == s)) for s in sorted({r["source"] for r in rs})),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "ground_truth.jsonl"))
    a = ap.parse_args()
    rows, raw = merged()
    n = write_jsonl(a.out, rows)
    print(f"wrote {n} rows to {a.out}")
    tot = defaultdict(int)
    for s in stats(rows):
        for k in ("cases", "easy", "mid", "hard_exact", "hard_censored", "low_censored"):
            tot[k] += s[k]
        print(json.dumps(s))
    print("TOTAL", dict(tot))
    # deepen cost by group
    g = defaultdict(lambda: defaultdict(float))
    for j in raw:
        G = g[(j["group"], j["cell"])]
        G["n"] += 1
        G[j["status"]] += 1
        G["conflicts"] += j["cost"]["conflicts"]
        G["propagations"] += j["cost"]["propagations"]
        G["seconds"] += j["cost"]["seconds"]
        G["time_hits"] += j.get("budget_hit") == "time"
        G["conf_hits"] += j.get("budget_hit") == "conflicts"
    for k in sorted(g):
        print("COST", k, {kk: (round(v, 1) if isinstance(v, float) else v) for kk, v in g[k].items()})
    sats = [j for j in raw if j["status"] == "sat"]
    print("SAT results:", len(sats))
    for j in sats:
        print("  SAT", j["cell"], j["rows"], j["cols"], "witness_ok", j.get("witness_ok"), "ones", j.get("ones"))


if __name__ == "__main__":
    main()

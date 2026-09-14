"""Deep-band task 1a: extract (val, cols, slots) of the heavy config of every
stored optimal witness, verify legality and the exact Pareto identity
   edges = 3n + val(C) - max(0, n - #C - (B - slots(C)))
cell by cell, and cross-check edges against the workspace ground truth.

Output: witness_configs.csv (one row per witness) + prints a verification
summary. Honesty: any witness failing legality or the identity is flagged
LOUDLY, never silently dropped.
"""
import csv
import json
import os
import sys
from collections import Counter
from itertools import combinations
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
WDIR = os.path.join(BASE, "analysis", "witnesses")

# ---------------- ground truth ----------------------------------------------
sys.path.insert(0, os.path.join(BASE, ".."))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "ev", os.path.join(BASE, "..", "evaluator.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
Z = dict(ev.KST_EXACT_VALUE)          # published proven cells
# proven new cells from the ILP campaigns (witness_valid records only)
for fn in ("ilp_gapband.jsonl", "ilp_results.jsonl", "ilp_frontier.jsonl"):
    p = os.path.join(BASE, "analysis", fn)
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "z_ilp" in r and r.get("witness_valid"):
                    Z.setdefault((r["m"], r["n"]), r["z_ilp"])

# ---------------- extraction -------------------------------------------------


def analyze(m, n, blocks):
    B = 2 * comb(m, 3)
    heavy = [b for b in blocks if len(b) >= 4]
    triples_cols = sum(1 for b in blocks if len(b) == 3)
    pads = sum(1 for b in blocks if len(b) <= 2)
    val = sum(len(b) - 3 for b in heavy)
    slots_h = sum(comb(len(b), 3) for b in heavy)
    cols = len(heavy)
    prof = Counter(len(b) for b in heavy)
    # legality of the whole matrix
    cov = Counter()
    for b in blocks:
        for t in combinations(sorted(b), 3):
            cov[t] += 1
    legal = all(v <= 2 for v in cov.values())
    edges = sum(len(b) for b in blocks)
    # the exact identity, evaluated on this config
    pen = max(0, n - cols - (B - slots_h))
    ident = 3 * n + val - pen
    return dict(m=m, n=n, edges=edges, legal=legal, cols=cols, val=val,
                slots=slots_h, W=sum(len(b) * (len(b) - 3) for b in heavy),
                X=sum((len(b) - 4) * (len(b) - 5) for b in heavy),
                k4=prof.get(4, 0), k5=prof.get(5, 0), k6=prof.get(6, 0),
                k7=prof.get(7, 0), k8=prof.get(8, 0), k9=prof.get(9, 0),
                ntri=triples_cols, pads=pads, penalty=pen,
                identity_edges=ident,
                z_truth=Z.get((m, n)))


def main():
    rows = []
    bad = []
    for fn in sorted(os.listdir(WDIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(WDIR, fn)) as f:
            w = json.load(f)
        r = analyze(w["m"], w["n"], [tuple(b) for b in w["blocks"]])
        r["file"] = fn
        if w["edges"] != r["edges"]:
            bad.append((fn, "edge count mismatch header vs blocks"))
        if not r["legal"]:
            bad.append((fn, "ILLEGAL witness"))
        if r["identity_edges"] != r["edges"]:
            bad.append((fn, f"identity mismatch: 3n+val-pen={r['identity_edges']}"
                            f" edges={r['edges']}"))
        if r["z_truth"] is not None and r["edges"] != r["z_truth"]:
            bad.append((fn, f"edges {r['edges']} != z_truth {r['z_truth']}"))
        rows.append(r)

    cols_out = ["file", "m", "n", "edges", "z_truth", "legal", "cols", "val",
                "slots", "W", "X", "k4", "k5", "k6", "k7", "k8", "k9",
                "ntri", "pads", "penalty", "identity_edges"]
    with open(os.path.join(HERE, "witness_configs.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols_out, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    print(f"{len(rows)} witnesses analyzed.")
    n_match = sum(1 for r in rows
                  if r["z_truth"] is not None and r["edges"] == r["z_truth"])
    n_truth = sum(1 for r in rows if r["z_truth"] is not None)
    print(f"legal: {sum(1 for r in rows if r['legal'])}/{len(rows)}; "
          f"identity holds: "
          f"{sum(1 for r in rows if r['identity_edges'] == r['edges'])}"
          f"/{len(rows)}; z-truth matches: {n_match}/{n_truth}")
    pen = [r for r in rows if r["penalty"] > 0]
    print(f"witnesses with ACTIVE penalty (pads forced): {len(pen)}")
    for r in pen:
        print(f"  ({r['m']},{r['n']}): cols={r['cols']} slots={r['slots']} "
              f"pen={r['penalty']} pads={r['pads']}")
    if bad:
        print("\n!!! PROBLEMS:")
        for fn, msg in bad:
            print(f"  {fn}: {msg}")
    else:
        print("no problems found.")


if __name__ == "__main__":
    main()

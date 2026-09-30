"""E24 / A3: compute A4's `free20k` inputs (pysat 20k probe statistics + static/LP profile
features, via A4's public `zar_ub.hardness_progress.estimate`, pysat only, no binary) for the
target-cell deepened cases that A4 did not featurise, so that sampling_target.py can compare
and combine the sampling estimate with A4's model on the real target tables.

In deployment these features are free (the censored table already ran the 20k probe);
here they cost one fresh 20k-conflict run per case, reported in the output.

usage: python experiments/E24_difficulty/sampling_a4feats.py <cases: same flags as sampling_run>
       --out sampling_data/target_a4.jsonl   (one process, resumable)"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UB)
sys.path.insert(0, HERE)

from sampling_run import key_of, load_cases  # noqa: E402
from zar_ub import hardness_progress as hp  # noqa: E402
from zar_ub.known import Instance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="final")
    ap.add_argument("--cells", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--include-censored", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cases = load_cases(a.cases, 0, 0, cells=tuple(c for c in a.cells.split(",") if c), source=a.source,
                       include_censored=a.include_censored)
    cases = [c for c in cases if c["d"] > 20000]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["key"])
            except ValueError:
                pass
    print(f"{len(cases)} cases, {len(done)} done", flush=True)
    t0 = time.time()
    with open(a.out, "a") as f:
        for i, c in enumerate(cases):
            k = key_of(c)
            if k in done:
                continue
            inst = Instance(c["m"], c["n"], c["s"], c["t"], c["w"])
            e = hp.estimate(inst, c["rows"], c["cols"], caps=(20000,), tier="20k", binary=False, lp=True)
            f.write(json.dumps(dict(key=k, cell=c["cell"], d=c["d"], status=c["status"], rows=c["rows"],
                                    cols=c["cols"], c2000=c.get("c2000"), log2_volume=c.get("log2_volume"),
                                    features=e["features"], cost_conflicts=e["cost_conflicts"],
                                    cost_propagations=e["cost_propagations"],
                                    cost_seconds=round(e["cost_seconds"], 3)), separators=(",", ":")) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(cases)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""E24 / A4: collect family-A (CDCL progress) and family-B (static / LP slack) features on the
dev set (owner A4).

Selection (seeded, random.Random(SEED)): every exactly-labelled HARD case (status unsat|sat and
d > 20 000) and up to --mid-per-cell exactly-labelled MID cases (2 000 < d <= 20 000) per cell.
HARD cases get the full "20k" tier (binary 2k + 20k, pysat 2k + 20k, static + LP); MID cases get
the "2k" tier (binary 2k, pysat 2k, static + LP): a 20k probe *solves* a MID case, so 20k-tier
features are not a prediction there.

Resumable: cases already present in the output file (same cell/rows/cols/tier) are skipped.

  python experiments/E24_difficulty/progress_collect.py \
      --gt experiments/E24_difficulty/ground_truth_initial.jsonl \
      --out experiments/E24_difficulty/progress_features_dev.jsonl --procs 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

UB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if UB not in sys.path:
    sys.path.insert(0, UB)

from zar_ub.known import Instance  # noqa: E402
from zar_ub import hardness_progress as hp  # noqa: E402

SEED = 20260922
HARD = 20_000
MID = 2_000


def regime(r: dict) -> str:
    if r.get("status") not in ("unsat", "sat") or r.get("d") is None:
        return "censored"
    d = float(r["d"])
    return "hard" if d > HARD else ("mid" if d > MID else "easy")


def key(r: dict, tier: str) -> str:
    return json.dumps([r["cell"], r["rows"], r["cols"], tier])


def _normalise(r: dict):
    """Accept both the ground-truth contract rows and A1's raw deepening rows (inst dict)."""
    if "m" not in r:
        inst = r.get("inst")
        if not inst:
            return None
        r = dict(r, **{k: inst[k] for k in ("m", "n", "s", "t", "w")})
    if r.get("log2_volume") is None:
        from zar_ub.difficulty import log2_volume

        r["log2_volume"] = log2_volume(Instance(r["m"], r["n"], r["s"], r["t"], r["w"]), r["rows"])
    return r


def select(
    gt_path: str,
    mid_per_cell: int,
    hard_tier: str = "20k",
    mid_tier: str = "2k",
    extra_hard_mid: int = 0,
):
    by_cell = {}
    for line in open(gt_path):
        line = line.strip()
        if not line:
            continue
        r = _normalise(json.loads(line))
        if r is None:
            continue
        by_cell.setdefault(r["cell"], []).append(r)
    rng = random.Random(SEED)
    jobs = []
    for cell in sorted(by_cell):
        recs = by_cell[cell]
        hard = [r for r in recs if regime(r) == "hard"]
        mid = [r for r in recs if regime(r) == "mid"]
        mid.sort(key=lambda r: (r["rows"], r["cols"]))
        pick = mid if len(mid) <= mid_per_cell else rng.sample(mid, mid_per_cell)
        for r in hard:
            jobs.append((r, hard_tier, "hard"))
        for r in pick:
            jobs.append((r, mid_tier, "mid"))
        if extra_hard_mid:  # a 20k-tier subsample of MID cases (diagnostic only)
            sub = (
                pick
                if len(pick) <= extra_hard_mid
                else random.Random(SEED + 1).sample(pick, extra_hard_mid)
            )
            for r in sub:
                jobs.append((r, "20k", "mid"))
    return jobs


def _work(args):
    r, tier, reg = args
    inst = Instance(r["m"], r["n"], r["s"], r["t"], r["w"])
    t0 = time.time()
    try:
        est = hp.estimate(inst, r["rows"], r["cols"], tier=tier, full=True, model={})
        err = None
    except Exception as e:  # noqa: BLE001
        est, err = None, repr(e)
    out = {
        "cell": r["cell"],
        "m": r["m"],
        "n": r["n"],
        "s": r["s"],
        "t": r["t"],
        "w": r["w"],
        "rows": r["rows"],
        "cols": r["cols"],
        "status": r["status"],
        "d": r["d"],
        "cap": r.get("cap"),
        "c2000": r.get("c2000"),
        "log2_volume": r.get("log2_volume"),
        "regime": reg,
        "tier": tier,
        "wall": round(time.time() - t0, 3),
    }
    if est is not None:
        out.update(
            features=est["features"],
            cost_conflicts=est["cost_conflicts"],
            cost_propagations=est["cost_propagations"],
            cost_seconds=round(est["cost_seconds"], 3),
            runs=est["meta"]["runs"],
        )
    else:
        out["error"] = err
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gt",
        default=os.path.join(UB, "experiments", "E24_difficulty", "ground_truth_initial.jsonl"),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(UB, "experiments", "E24_difficulty", "progress_features_dev.jsonl"),
    )
    ap.add_argument("--mid-per-cell", type=int, default=150)
    ap.add_argument(
        "--extra-hard-mid",
        type=int,
        default=0,
        help="also run the 20k tier on this many MID cases per cell",
    )
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--only", default="", help="comma list of regimes to run (hard,mid)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cells", default="", help="comma list of cell tags to restrict to")
    ap.add_argument(
        "--exclude-cells-of",
        default="",
        help="skip cells that have HARD records in this features file",
    )
    ap.add_argument(
        "--also-done",
        default="",
        help="comma list of other feature files whose cases count as done",
    )
    a = ap.parse_args()
    jobs = select(a.gt, a.mid_per_cell, extra_hard_mid=a.extra_hard_mid)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j[2] in keep]
    if a.cells:
        keep = set(a.cells.split(","))
        jobs = [j for j in jobs if j[0]["cell"] in keep]
    if a.exclude_cells_of and os.path.exists(a.exclude_cells_of):
        excl = set()
        for line in open(a.exclude_cells_of):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("regime") == "hard":
                excl.add(o["cell"])
        jobs = [j for j in jobs if j[0]["cell"] not in excl]
    done = set()
    for path in [a.out] + [p for p in a.also_done.split(",") if p]:
        if not os.path.exists(path):
            continue
        for line in open(path):
            try:
                o = json.loads(line)
                if "features" in o:
                    done.add(key(o, o["tier"]))
            except ValueError:
                pass
    todo = [j for j in jobs if key(j[0], j[1]) not in done]
    # hard cases first (they matter most), then by cell for locality
    todo.sort(key=lambda j: (j[2] != "hard", j[0]["cell"]))
    if a.limit:
        todo = todo[: a.limit]
    print(
        f"selected {len(jobs)} jobs, {len(done)} done, {len(todo)} to run with {a.procs} procs",
        flush=True,
    )
    t0 = time.time()
    tot_c = tot_p = 0
    with open(a.out, "a") as fh, Pool(a.procs) as pool:
        for k, o in enumerate(pool.imap_unordered(_work, todo, chunksize=1)):
            fh.write(json.dumps(o) + "\n")
            fh.flush()
            tot_c += o.get("cost_conflicts", 0)
            tot_p += o.get("cost_propagations", 0)
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                print(
                    f"{k + 1}/{len(todo)}  {time.time() - t0:.0f}s  conflicts {tot_c:,}  propagations {tot_p:,}",
                    flush=True,
                )


if __name__ == "__main__":
    main()

"""SAT hunt for a T_{3,3}(v) packing with >= target 4-blocks.

T >= target is encoded as: total triple-coverage deficit
  sum_t (2 - cov(t))  <=  2*C(v,3) - 4*target
over exact per-triple unary indicators (coverage identity 4*T = sum cov).

Usage: run_t33.py [--v 19] [--target 482] [--budget 3600]
                  [--solver cadical195] [--tag NAME] [--seed-blocks FILE]

--seed-blocks: JSON packing witness whose blocks are passed to the solver
as soft phase hints (Cadical: set_phases), biasing search toward a known
good partial packing without constraining it.
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encodings_zar import encode_t33, decode_t33  # noqa: E402

from pysat.solvers import Solver  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=int, default=19)
    ap.add_argument("--target", type=int, default=482)
    ap.add_argument("--budget", type=float, default=3600.0)
    ap.add_argument("--solver", default="cadical195")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--no-fix-first", action="store_true")
    ap.add_argument("--seed-blocks", default=None)
    args = ap.parse_args()

    tag = args.tag or f"T3_t33_{args.v}_{args.target}_{args.solver}"
    res_path = os.path.join(HERE, "results", f"{tag}.json")
    wit_path = os.path.join(HERE, "witnesses", f"{tag}_witness.json")

    t0 = time.time()
    clauses, nv, blocks, avar, bvar = encode_t33(
        args.v, args.target, fix_first_block=not args.no_fix_first)
    enc_time = time.time() - t0
    payload = {
        "target": f"T33({args.v}) >= {args.target}",
        "solver": args.solver, "nvars": nv, "nclauses": len(clauses),
        "encode_seconds": round(enc_time, 2), "budget": args.budget,
        "status": "running",
    }
    print(f"[{tag}] encoded: {nv} vars, {len(clauses)} clauses "
          f"({enc_time:.1f}s)", flush=True)

    def timeout_exit():
        payload["status"] = "timeout"
        payload["solve_seconds"] = round(time.time() - t1, 1)
        with open(res_path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[{tag}] TIMEOUT after {payload['solve_seconds']}s", flush=True)
        os._exit(99)

    solver = Solver(name=args.solver, bootstrap_with=clauses, use_timer=True)

    if args.seed_blocks:
        from collections import Counter
        with open(args.seed_blocks) as f:
            seed = json.load(f)
        cnt = Counter(tuple(sorted(b)) for b in seed["blocks"])
        phases = []
        for B in blocks:
            k = cnt.get(B, 0)
            phases.append(avar[B] if k >= 1 else -avar[B])
            phases.append(bvar[B] if k >= 2 else -bvar[B])
        try:
            solver.set_phases(phases)
            payload["seeded"] = args.seed_blocks
            print(f"[{tag}] phase-seeded from {args.seed_blocks} "
                  f"({sum(cnt.values())} blocks)", flush=True)
        except Exception as e:
            payload["seed_error"] = str(e)

    t1 = time.time()
    watchdog = threading.Timer(args.budget, timeout_exit)
    watchdog.daemon = True
    watchdog.start()
    sat = solver.solve()
    watchdog.cancel()
    payload["solve_seconds"] = round(time.time() - t1, 1)

    if sat:
        model = solver.get_model()
        packed = decode_t33(model, blocks, avar, bvar)
        payload["status"] = "sat"
        payload["count_decoded"] = len(packed)
        with open(wit_path, "w") as f:
            json.dump({"v": args.v, "count": len(packed), "blocks": packed,
                       "source": f"sat:{tag}"}, f)
        payload["witness"] = wit_path
        print(f"[{tag}] SAT count={len(packed)} in "
              f"{payload['solve_seconds']}s", flush=True)
    else:
        payload["status"] = "unsat"
        print(f"[{tag}] UNSAT in {payload['solve_seconds']}s "
              f"(would contradict attainability if target correct!)",
              flush=True)

    with open(res_path, "w") as f:
        json.dump(payload, f, indent=1)


if __name__ == "__main__":
    main()

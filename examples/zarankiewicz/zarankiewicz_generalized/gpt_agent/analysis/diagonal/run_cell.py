"""Run one matrix-cell SAT attack: does an m x n K_{3,3}-free 0/1 matrix
with `total` ones exist?

Usage:
  run_matrix.py M N TOTAL --budget 3600 --solver cadical195
                [--card equals|atleast] [--no-collex] [--no-degmono]
                [--tag NAME] [--proof]

Writes results/<tag>.json with status sat/unsat/timeout, and on SAT a
witness JSON (column-support blocks) to witnesses/<tag>_witness.json.
The witness is decoded from the raw model; independent verification is
verify_witness.py's job.
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sat_attack"))
from encodings_zar import encode_matrix, decode_matrix  # noqa: E402

from pysat.solvers import Solver  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("m", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("total", type=int)
    ap.add_argument("--budget", type=float, default=3600.0)
    ap.add_argument("--solver", default="cadical195")
    ap.add_argument("--card", default="equals", choices=["equals", "atleast"])
    ap.add_argument("--no-collex", action="store_true")
    ap.add_argument("--no-degmono", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--proof", action="store_true",
                    help="enable DRUP/DRAT proof tracing (dumped on UNSAT)")
    ap.add_argument("--weights", default=None,
                    help="comma-separated exact column weights (cube mode)")
    ap.add_argument("--paircuts", action="store_true",
                    help="add row-pair counting cuts (cube mode only)")
    ap.add_argument("--wwin", default=None,
                    help="LO,HI window for every column weight (flat mode)")
    ap.add_argument("--wmax-exact", action="store_true",
                    help="require some column to reach HI (cube max==HI)")
    ap.add_argument("--dimacs", action="store_true",
                    help="dump the CNF in DIMACS format before solving")
    args = ap.parse_args()
    weights = ([int(x) for x in args.weights.split(",")]
               if args.weights else None)

    tag = args.tag or f"z_{args.m}x{args.n}_t{args.total}_{args.solver}"
    res_path = os.path.join(HERE, "results", f"{tag}.json")
    wit_path = os.path.join(HERE, "witnesses", f"{tag}_witness.json")

    t0 = time.time()
    clauses, nv, X = encode_matrix(
        args.m, args.n, args.total, card_mode=args.card,
        col_lex=not args.no_collex, row_degmono=not args.no_degmono,
        col_weights=weights, pair_cuts=args.paircuts,
        col_weight_window=(tuple(int(x) for x in args.wwin.split(","))
                           if args.wwin else None),
        max_weight_exact=args.wmax_exact)
    enc_time = time.time() - t0

    payload = {
        "target": [args.m, args.n], "total": args.total,
        "solver": args.solver, "card": args.card,
        "col_lex": not args.no_collex, "row_degmono": not args.no_degmono,
        "weights": weights,
        "nvars": nv, "nclauses": len(clauses),
        "encode_seconds": round(enc_time, 2),
        "budget": args.budget, "status": "running",
    }
    print(f"[{tag}] encoded: {nv} vars, {len(clauses)} clauses "
          f"({enc_time:.1f}s)", flush=True)

    if args.dimacs:
        dpath = os.path.join(HERE, "results", f"{tag}.cnf")
        with open(dpath, "w") as f:
            f.write(f"p cnf {nv} {len(clauses)}\n")
            f.writelines(" ".join(map(str, cl)) + " 0\n" for cl in clauses)
        payload["dimacs"] = dpath
        print(f"[{tag}] DIMACS -> {dpath}", flush=True)

    def timeout_exit():
        payload["status"] = "timeout"
        payload["solve_seconds"] = round(time.time() - t1, 1)
        with open(res_path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[{tag}] TIMEOUT after {payload['solve_seconds']}s", flush=True)
        os._exit(99)

    kw = {"bootstrap_with": clauses, "use_timer": True}
    if args.proof:
        kw["with_proof"] = True
    solver = Solver(name=args.solver, **kw)

    t1 = time.time()
    watchdog = threading.Timer(args.budget, timeout_exit)
    watchdog.daemon = True
    watchdog.start()
    sat = solver.solve()
    watchdog.cancel()
    payload["solve_seconds"] = round(time.time() - t1, 1)
    payload["solver_reported_seconds"] = round(solver.time(), 1)

    if sat:
        model = solver.get_model()
        grid = decode_matrix(model, args.m, args.n)
        edges = sum(sum(row) for row in grid)
        blocks = [sorted(r for r in range(args.m) if grid[r][c])
                  for c in range(args.n)]
        payload["status"] = "sat"
        payload["edges_decoded"] = edges
        with open(wit_path, "w") as f:
            json.dump({"m": args.m, "n": args.n, "edges": edges,
                       "blocks": blocks, "source": f"sat:{tag}"}, f)
        payload["witness"] = wit_path
        print(f"[{tag}] SAT edges={edges} in {payload['solve_seconds']}s",
              flush=True)
    else:
        payload["status"] = "unsat"
        if args.proof:
            try:
                proof = solver.get_proof()
                ppath = os.path.join(HERE, "results", f"{tag}.drat")
                with open(ppath, "w") as f:
                    f.write("\n".join(proof) + "\n")
                payload["proof"] = ppath
                payload["proof_lines"] = len(proof)
            except Exception as e:  # solver may not support tracing
                payload["proof_error"] = str(e)
        print(f"[{tag}] UNSAT in {payload['solve_seconds']}s", flush=True)

    with open(res_path, "w") as f:
        json.dump(payload, f, indent=1)


if __name__ == "__main__":
    main()

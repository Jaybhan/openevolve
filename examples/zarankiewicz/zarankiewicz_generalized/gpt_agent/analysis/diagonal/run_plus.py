"""Ladder-enhanced SAT run for one cell (m,n,total).

Reuses encodings_zar.encode_matrix (triples + cardinality + col-lex +
row-degmono, optional wwin) and ADDS proven deletion-ladder constraints
the solver cannot derive:

  (i)  per-cell:  d_r + w_c - x[r][c] >= total - z(m-1,n-1)
       [delete row r and column c: the minor is K33-free, so its edge
        count total - d_r - w_c + x[r][c] <= z(m-1,n-1)]
  (ii) per-row:   d_r >= total - z(m-1,n)
  (iii)per-col:   w_c >= total - z(m,n-1)

All encoded over fresh exact unary counters (iff semantics) for row degrees
and column weights.  Soundness: for the >=total decision, ones-deletion
monotonicity lets us assume exactly `total` ones; (i)-(iii) hold for every
such matrix given the stated minor values, which are supplied by the caller
and MUST be proven upper bounds for the minors.

Usage: run_plus.py M N TOTAL ZMM ZM1N ZMN1 [--budget S] [--solver NAME]
       [--wwin LO,HI] [--tag T] [--proof]
where ZMM = UB on z(m-1,n-1), ZM1N = UB on z(m-1,n), ZMN1 = UB on z(m,n-1).
"""
import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sat_attack"))
from encodings_zar import encode_matrix, decode_matrix, \
    exact_unary_counter  # noqa: E402

from pysat.solvers import Solver  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def add_ladder(clauses, top, X, m, n, total, z_mm, z_m1n, z_mn1):
    """Append ladder clauses; returns new top."""
    # fresh exact unary counters
    ROW = []
    for r in range(m):
        top, outs = exact_unary_counter([X[r][c] for c in range(n)],
                                        top, clauses)
        ROW.append(outs)          # outs[k-1] <-> d_r >= k
    COL = []
    for c in range(n):
        top, outs = exact_unary_counter([X[r][c] for r in range(m)],
                                        top, clauses)
        COL.append(outs)

    def geq(outs, k, sz):
        """Literal for 'value >= k' (k>=1); True if k<=0, False if k>sz."""
        if k <= 0:
            return True
        if k > sz:
            return False
        return outs[k - 1]

    # (ii) d_r >= total - z_m1n ; (iii) w_c >= total - z_mn1
    t_row = total - z_m1n
    t_col = total - z_mn1
    for r in range(m):
        lit = geq(ROW[r], t_row, n)
        if lit is False:
            clauses.append([])      # unsat outright
        elif lit is not True:
            clauses.append([lit])
    for c in range(n):
        lit = geq(COL[c], t_col, m)
        if lit is False:
            clauses.append([])
        elif lit is not True:
            clauses.append([lit])

    # (i) d_r + w_c >= t + x[r][c],  t = total - z_mm
    t = total - z_mm
    for r in range(m):
        for c in range(n):
            for xv in (0, 1):
                need = t + xv           # d_r + w_c >= need
                # clause form: OR over splits a=(0..need): d>=a or w>=need-a
                # equivalently: for every a in 0..need-1 with both sides
                # possibly failing: NOT(d<=a-1 AND w<=need-a-1) ->
                # standard unary encoding: for a in 0..need:
                #   (d_r >= a) OR (w_c >= need - a + ... )
                # correct pairwise form: for every a in 1..need:
                #   (d_r >= a) OR (w_c >= need - a + 1)
                for a in range(0, need + 1):
                    la = geq(ROW[r], a + 1, n)          # d_r >= a+1
                    lb = geq(COL[c], need - a, m)       # w_c >= need-a
                    # constraint: d_r <= a  ->  w_c >= need - a
                    # i.e. clause  (d_r >= a+1) OR (w_c >= need-a)
                    if lb is True or la is True:
                        continue
                    cl = [] if xv == 0 else [-X[r][c]]
                    if la is not False:
                        cl.append(la)
                    if lb is not False:
                        cl.append(lb)
                    if not cl or (la is False and lb is False):
                        if xv == 0:
                            clauses.append([])   # impossible unconditionally
                        else:
                            clauses.append([-X[r][c]])
                    else:
                        clauses.append(cl)
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("m", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("total", type=int)
    ap.add_argument("zmm", type=int, help="UB z(m-1,n-1)")
    ap.add_argument("zm1n", type=int, help="UB z(m-1,n)")
    ap.add_argument("zmn1", type=int, help="UB z(m,n-1)")
    ap.add_argument("--budget", type=float, default=14400.0)
    ap.add_argument("--solver", default="cadical195")
    ap.add_argument("--wwin", default=None)
    ap.add_argument("--wmax-exact", action="store_true",
                    help="require some column to reach the wwin HI bound")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--proof", action="store_true")
    args = ap.parse_args()
    tag = args.tag or f"plus_{args.m}x{args.n}_t{args.total}"
    res_path = os.path.join(HERE, "results", f"{tag}.json")
    wit_path = os.path.join(HERE, "witnesses", f"{tag}_witness.json")

    clauses, top, X = encode_matrix(
        args.m, args.n, args.total, card_mode="equals",
        col_lex=True, row_degmono=True,
        col_weight_window=(tuple(int(x) for x in args.wwin.split(","))
                           if args.wwin else None),
        max_weight_exact=args.wmax_exact)
    top = add_ladder(clauses, top, X, args.m, args.n, args.total,
                     args.zmm, args.zm1n, args.zmn1)
    payload = {
        "target": [args.m, args.n], "total": args.total,
        "solver": args.solver, "encoding": "encodings_zar + ladder clauses",
        "ladder_ubs": {"z(m-1,n-1)": args.zmm, "z(m-1,n)": args.zm1n,
                       "z(m,n-1)": args.zmn1},
        "wwin": args.wwin, "nvars": top, "nclauses": len(clauses),
        "budget": args.budget, "status": "running",
    }
    print(f"[{tag}] encoded: {top} vars, {len(clauses)} clauses", flush=True)
    if any(len(c) == 0 for c in clauses):
        payload["status"] = "unsat"
        payload["note"] = "empty clause during ladder encoding"
        with open(res_path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[{tag}] UNSAT (encoding-time)", flush=True)
        return

    kw = {"bootstrap_with": clauses, "use_timer": True}
    if args.proof:
        kw["with_proof"] = True
    solver = Solver(name=args.solver, **kw)
    t1 = time.time()

    def timeout_exit():
        payload["status"] = "timeout"
        payload["solve_seconds"] = round(time.time() - t1, 1)
        with open(res_path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[{tag}] TIMEOUT {payload['solve_seconds']}s", flush=True)
        os._exit(99)

    wd = threading.Timer(args.budget, timeout_exit)
    wd.daemon = True
    wd.start()
    sat = solver.solve()
    wd.cancel()
    payload["solve_seconds"] = round(time.time() - t1, 1)
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
            except Exception as e:
                payload["proof_error"] = str(e)
        print(f"[{tag}] UNSAT in {payload['solve_seconds']}s", flush=True)
    with open(res_path, "w") as f:
        json.dump(payload, f, indent=1)


if __name__ == "__main__":
    main()

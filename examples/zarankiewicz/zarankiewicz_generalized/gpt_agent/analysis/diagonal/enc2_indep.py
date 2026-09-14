"""Independent second SAT encoding for z(m,n;3,3) decisions.

Written from scratch, sharing no code with encodings_zar.py:
  * K33-freeness via COLUMN triples: for each set of 3 columns, an
    auxiliary per row (one-directional AND), AtMost2 by PAIRWISE
    conflict clauses over a sequential relay (my own, not pysat's).
  * cardinality >= total via pysat CardEnc.atleast with seqcounter
    (different mode AND encoder than the primary's totalizer-equals).
  * symmetry breaking on the TRANSPOSE side: adjacent ROW lex ordering
    (sound: row permutations preserve everything; note primary uses
    column lex + row degree monotone, so the search space shaping is
    different).
Usage: enc2_indep.py M N TOTAL [--solver glucose42] [--budget S] [--tag T]
"""
import argparse
import json
import os
import sys
import threading
import time
from itertools import combinations

from pysat.card import CardEnc, EncType
from pysat.solvers import Solver

HERE = os.path.dirname(os.path.abspath(__file__))


def encode(m, n, total):
    X = [[r * n + c + 1 for c in range(n)] for r in range(m)]
    top = m * n
    cls = []
    # column triples: |c1 ^ c2 ^ c3| <= 2
    for c1, c2, c3 in combinations(range(n), 3):
        ands = []
        for r in range(m):
            top += 1
            a = top
            cls.append([-X[r][c1], -X[r][c2], -X[r][c3], a])
            ands.append(a)
        # AtMost2 via my own sequential relay: s_r = count>=1 prefix,
        # t_r = count>=2 prefix; forbid a third.
        s_prev = t_prev = None
        for i, a in enumerate(ands):
            top += 2
            s, t = top - 1, top
            if s_prev is None:
                cls.append([-a, s])
            else:
                cls.append([-s_prev, s])
                cls.append([-a, s])
                cls.append([-t_prev, t])
                cls.append([-s_prev, -a, t])
                cls.append([-t_prev, -a])  # would be third
            s_prev, t_prev = s, t
    # cardinality: at least total ones
    allx = [X[r][c] for r in range(m) for c in range(n)]
    cnf = CardEnc.atleast(lits=allx, bound=total, top_id=top,
                          encoding=EncType.seqcounter)
    cls.extend(cnf.clauses)
    top = max(top, cnf.nv)
    # adjacent row lex: row_r >=_lex row_{r+1}
    for r in range(m - 1):
        p_prev = None
        for c in range(n):
            a1, b1 = X[r][c], X[r + 1][c]
            cond = [] if p_prev is None else [-p_prev]
            cls.append(cond + [a1, -b1])
            if c < n - 1:
                top += 1
                p = top
                cls.append(cond + [a1, b1, p])
                cls.append(cond + [-a1, -b1, p])
                p_prev = p
    # column-weight monotonicity (w_c >= w_{c+1}) via my own exact
    # sequential counters: g[c][k] <-> (col c has >= k+1 ones), full iff.
    # Sound with row-lex: first sort columns by weight (stable), then
    # permute rows to lex-sort rows -- wait, row permutations change
    # column words but not weights, and row-lex sorting is achieved by
    # permuting rows AFTER fixing the column order; both constraints are
    # simultaneously satisfiable by normalizing any matrix (weights are
    # row-permutation invariant), same argument as the primary encoder's.
    G = []
    for c in range(n):
        prev = [None] * m       # prev[k] = ">= k+1" after i items
        for i in range(m):
            cur = [None] * m
            xi = X[i][c]
            for k in range(min(i + 1, m)):
                top += 1
                v = top
                cur[k] = v
                pk = prev[k]
                pk1 = prev[k - 1] if k > 0 else None
                # forward: pk -> v ;  (pk1 or k==0 base) & xi -> v
                if pk is not None:
                    cls.append([-pk, v])
                if k == 0:
                    cls.append([-xi, v])
                elif pk1 is not None:
                    cls.append([-pk1, -xi, v])
                # backward: v -> pk | (pk1&xi)
                base = [-v] + ([pk] if pk is not None else [])
                if k == 0:
                    cls.append(base + [xi])
                elif pk1 is None:
                    if pk is not None:
                        cls.append(base)
                    else:
                        cls.append([-v])
                else:
                    cls.append(base + [pk1])
                    cls.append(base + [xi])
            prev = cur
        G.append(prev)
    for c in range(n - 1):
        for k in range(m):
            a, b = G[c][k], G[c + 1][k]
            if b is None:
                continue
            if a is None:
                cls.append([-b])
            else:
                cls.append([-b, a])
    return cls, top, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("m", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("total", type=int)
    ap.add_argument("--solver", default="glucose42")
    ap.add_argument("--budget", type=float, default=28800.0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"enc2_{args.m}x{args.n}_t{args.total}_{args.solver}"
    res_path = os.path.join(HERE, "results", f"{tag}.json")
    cls, nv, X = encode(args.m, args.n, args.total)
    payload = {"target": [args.m, args.n], "total": args.total,
               "solver": args.solver,
               "encoding": "enc2_indep (column-triple relay AtMost2, "
                           "seqcounter atleast, row-lex)",
               "nvars": nv, "nclauses": len(cls),
               "budget": args.budget, "status": "running"}
    print(f"[{tag}] encoded: {nv} vars {len(cls)} clauses", flush=True)
    solver = Solver(name=args.solver, bootstrap_with=cls, use_timer=True)
    t1 = time.time()

    def to():
        payload["status"] = "timeout"
        payload["solve_seconds"] = round(time.time() - t1, 1)
        with open(res_path, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"[{tag}] TIMEOUT", flush=True)
        os._exit(99)

    wd = threading.Timer(args.budget, to)
    wd.daemon = True
    wd.start()
    sat = solver.solve()
    wd.cancel()
    payload["solve_seconds"] = round(time.time() - t1, 1)
    if sat:
        model = set(l for l in solver.get_model() if l > 0)
        blocks = [sorted(r for r in range(args.m)
                         if (r * args.n + c + 1) in model)
                  for c in range(args.n)]
        edges = sum(len(b) for b in blocks)
        payload["status"] = "sat"
        payload["edges_decoded"] = edges
        wit = os.path.join(HERE, "witnesses", f"{tag}_witness.json")
        with open(wit, "w") as f:
            json.dump({"m": args.m, "n": args.n, "edges": edges,
                       "blocks": blocks, "source": f"sat:{tag}"}, f)
        payload["witness"] = wit
        print(f"[{tag}] SAT edges={edges} in {payload['solve_seconds']}s",
              flush=True)
    else:
        payload["status"] = "unsat"
        print(f"[{tag}] UNSAT in {payload['solve_seconds']}s", flush=True)
    with open(res_path, "w") as f:
        json.dump(payload, f, indent=1)


if __name__ == "__main__":
    main()

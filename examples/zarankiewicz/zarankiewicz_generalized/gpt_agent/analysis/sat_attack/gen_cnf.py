"""Dump a matrix-cell CNF to DIMACS (no solving) for external solvers.

Mirrors run_matrix.py encoding options.  A comment line records the
configuration; grid variable X[r][c] = r*n + c + 1 (documented for
witness decoding by decode_dimacs_model.py).

Usage: gen_cnf.py M N TOTAL --tag NAME [--card equals|atleast]
       [--no-collex] [--no-degmono] [--wwin LO,HI] [--wmax-exact]
       [--weights ...]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encodings_zar import encode_matrix  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("m", type=int)
    ap.add_argument("n", type=int)
    ap.add_argument("total", type=int)
    ap.add_argument("--card", default="equals", choices=["equals", "atleast"])
    ap.add_argument("--no-collex", action="store_true")
    ap.add_argument("--no-degmono", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--wwin", default=None)
    ap.add_argument("--wmax-exact", action="store_true")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    clauses, nv, X = encode_matrix(
        args.m, args.n, args.total, card_mode=args.card,
        col_lex=not args.no_collex, row_degmono=not args.no_degmono,
        col_weights=([int(x) for x in args.weights.split(",")]
                     if args.weights else None),
        col_weight_window=(tuple(int(x) for x in args.wwin.split(","))
                           if args.wwin else None),
        max_weight_exact=args.wmax_exact)
    path = os.path.join(HERE, "results", f"{args.tag}.cnf")
    with open(path, "w") as f:
        f.write(f"c z({args.m},{args.n};3,3) total={args.total} "
                f"card={args.card} collex={not args.no_collex} "
                f"degmono={not args.no_degmono} wwin={args.wwin} "
                f"wmaxexact={args.wmax_exact} weights={args.weights}\n")
        f.write(f"c grid var x[r][c] = r*{args.n} + c + 1\n")
        f.write(f"p cnf {nv} {len(clauses)}\n")
        f.writelines(" ".join(map(str, cl)) + " 0\n" for cl in clauses)
    print(f"{path}: {nv} vars {len(clauses)} clauses")


if __name__ == "__main__":
    main()

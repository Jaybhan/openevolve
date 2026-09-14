"""Cube-and-conquer over sorted column-weight profiles.

A K_{3,3}-free m x n matrix with >= `total` ones exists  iff  one exists
with exactly `total` ones, all column weights in [2, m], columns sorted by
(weight desc, lex desc), rows sorted by degree desc  --  and its weight
profile satisfies sum w = total, sum C(w,3) <= 2*C(m,3).  Enumerate all
such profiles; solve one fixed-weight subproblem per profile.

  profiles.py list M N TOTAL                  -> print count + profiles
  profiles.py cmds M N TOTAL --budget B --solver S --pass-tag P
      -> write results/<pass-tag>_cmds.txt (one run_matrix call per line)
  profiles.py summary PASS_TAG               -> tally results/<pass-tag>_p*.json
"""
import argparse
import glob
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def enum_profiles(m, n, total):
    budget = 2 * comb(m, 3)
    sols = []

    def gen(remaining, parts_left, maxw, cost, prefix):
        if parts_left == 0:
            if remaining == 0 and cost <= budget:
                sols.append(tuple(prefix))
            return
        hi = min(maxw, remaining - 2 * (parts_left - 1))
        for w in range(hi, 1, -1):
            c = cost + comb(w, 3)
            if c > budget:
                continue
            rest = remaining - w
            if rest < 2 * (parts_left - 1) or rest > m * (parts_left - 1):
                continue
            gen(rest, parts_left - 1, w, c, prefix + [w])

    gen(total, n, m, 0, [])
    return sols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["list", "cmds", "summary"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--solver", default="cadical195")
    ap.add_argument("--pass-tag", default="T1p")
    args = ap.parse_args()

    if args.mode == "summary":
        tag = args.args[0]
        tally = {}
        pending = []
        for f in sorted(glob.glob(os.path.join(HERE, "results",
                                               f"{tag}_p*.json"))):
            try:
                with open(f) as fh:
                    d = json.load(fh)
            except Exception:
                tally["corrupt"] = tally.get("corrupt", 0) + 1
                continue
            tally[d["status"]] = tally.get(d["status"], 0) + 1
            if d["status"] != "unsat":
                pending.append((os.path.basename(f), d["status"],
                                d.get("weights")))
        print("tally:", tally)
        for p in pending:
            print("  ", p)
        return

    m, n, total = map(int, args.args)
    profs = enum_profiles(m, n, total)
    if args.mode == "list":
        print(f"{len(profs)} profiles for ({m},{n}) total {total}")
        for p in profs[:10]:
            print(" ", p)
        return

    runner = os.path.join(HERE, "run_matrix.py")
    path = os.path.join(HERE, "results", f"{args.pass_tag}_cmds.txt")
    nw = 0
    with open(path, "w") as f:
        for i, p in enumerate(profs):
            tag = f"{args.pass_tag}_p{i:03d}"
            res = os.path.join(HERE, "results", f"{tag}.json")
            try:  # skip profiles already conclusively done
                with open(res) as fh:
                    if json.load(fh)["status"] in ("sat", "unsat"):
                        continue
            except Exception:
                pass
            w = ",".join(map(str, p))
            f.write(f"{PY} {runner} {m} {n} {total} --weights {w} "
                    f"--budget {args.budget} --solver {args.solver} "
                    f"--tag {tag} > /dev/null 2>&1\n")
            nw += 1
    print(f"{nw} commands (of {len(profs)} profiles) -> {path}")


if __name__ == "__main__":
    main()

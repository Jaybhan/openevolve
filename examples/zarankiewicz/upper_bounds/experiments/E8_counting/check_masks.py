#!/usr/bin/env python3
"""Semantic check of Lean prune kills against Python reference implementations.

Usage:
  python experiments/E8_counting/check_masks.py <lean_file_or_module> <lean_prune_term> <python_ref> [--m 9 --n 10 --s 3 --t 3 --w 55 --samples 400]

  <lean_file_or_module>  a .lean file path that (after `import ZarPrune`) defines the prune, or a module name like ZarPrune.Counting
  <lean_prune_term>      e.g. "ZarPrune.argA" (applied to P) or "ZarPrune.Cand.foo" ; the script evaluates `(<term> P).kill pf`
  <python_ref>           one of: argA argAT argD argDT argD_threshold_subset  (argD_threshold_subset only requires Lean kills ⊆ python kills)

The script samples random profiles (NOT just admissible cases) so that the check
exercises the prune on inputs it should kill as well as inputs it must not.
Exit code 0 iff every sampled profile agrees (or ⊆ for subset refs).
"""
import argparse, os, random, re, subprocess, sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
LEAN = os.path.join(UB, "lean")


def ref_argA(m, n, s, t, w, rows, cols):
    return sum(comb(c, s) for c in cols) > (t - 1) * comb(m, s)

def ref_argAT(m, n, s, t, w, rows, cols):
    return sum(comb(r, t) for r in rows) > (s - 1) * comb(n, t)

def ref_argD(m, n, s, t, w, rows, cols):
    r = max(rows); lightest = sorted(cols)[:r]
    return sum(comb(c - 1, s - 1) for c in lightest if c >= 1) > (t - 1) * comb(m - 1, s - 1)

def ref_argDT(m, n, s, t, w, rows, cols):
    c = max(cols); lightest = sorted(rows)[:c]
    return sum(comb(x - 1, t - 1) for x in lightest if x >= 1) > (s - 1) * comb(n - 1, t - 1)

REFS = {"argA": (ref_argA, "eq"), "argAT": (ref_argAT, "eq"), "argD": (ref_argD, "eq"), "argDT": (ref_argDT, "eq"),
        "argD_threshold_subset": (ref_argD, "subset"), "argDT_threshold_subset": (ref_argDT, "subset")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lean"); ap.add_argument("term"); ap.add_argument("ref", choices=REFS)
    ap.add_argument("--m", type=int, default=9); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--s", type=int, default=3); ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--w", type=int, default=55); ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    profs = []
    for _ in range(a.samples):
        rows = sorted((rnd.randint(0, a.n) for _ in range(a.m)), reverse=True)
        cols = sorted((rnd.randint(0, a.m) for _ in range(a.n)), reverse=True)
        profs.append((rows, cols))
    # also near-admissible ones: balanced profiles around w/m, w/n
    for _ in range(a.samples // 2):
        rows = sorted((max(0, min(a.n, round(a.w / a.m) + rnd.randint(-2, 2))) for _ in range(a.m)), reverse=True)
        cols = sorted((max(0, min(a.m, round(a.w / a.n) + rnd.randint(-2, 2))) for _ in range(a.n)), reverse=True)
        profs.append((rows, cols))
    if a.lean.endswith(".lean"):
        imp = open(a.lean).read()
        header = imp if imp.lstrip().startswith("import") else "import ZarPrune\n" + imp
    else:
        header = f"import {a.lean}\n"
    lines = [header, "", f"abbrev chkP : ZarPrune.Params := {{ m := {a.m}, n := {a.n}, s := {a.s}, t := {a.t}, w := {a.w} }}",
             "def chkProfile (r c : List Nat) : ZarPrune.Profile chkP.m chkP.n := { row := fun i => r.getD i.val 0, col := fun j => c.getD j.val 0 }",
             f"def chkPrune : ZarPrune.Prune chkP := by first | exact {a.term} chkP | exact {a.term}",
             "#print axioms chkPrune"]
    chunk = 200
    for st in range(0, len(profs), chunk):
        lits = ",\n  ".join("([" + ",".join(map(str, r)) + "],[" + ",".join(map(str, c)) + "])" for r, c in profs[st:st + chunk])
        # print via IO.println so the output is neither truncated (`⋯`) nor line-wrapped by the pretty-printer
        lines.append(f"#eval IO.println (\"MASK:\" ++ String.intercalate \",\" ((([\n  {lits}] : List (List Nat × List Nat)).map fun p => if chkPrune.kill (chkProfile p.1 p.2) then \"true\" else \"false\")))")
    path = os.path.join(LEAN, "Candidates", f"chk_{a.ref}_{abs(hash((a.lean, a.term))) % 10**8}.lean")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(lines) + "\n")
    proc = subprocess.run(["lake", "env", "lean", path], cwd=LEAN, capture_output=True, text=True, timeout=600)
    out = proc.stdout + proc.stderr
    errs = [l for l in out.splitlines() if ": error:" in l]
    if errs:
        print("LEAN ERRORS:\n" + "\n".join(errs[:10])); sys.exit(2)
    ax = re.search(r"'chkPrune' depends on axioms: \[([^\]]*)\]", out)
    axioms = [x.strip() for x in ax.group(1).split(",")] if ax else []
    print("axioms:", axioms)
    if not set(axioms) <= {"propext", "Quot.sound", "Classical.choice"}:
        print("BAD AXIOMS"); sys.exit(3)
    mask = []
    for mobj in re.finditer(r"^MASK:((?:true|false)(?:,(?:true|false))*)\s*$", out, flags=re.M):
        mask.extend(x.strip() == "true" for x in mobj.group(1).split(","))
    if len(mask) != len(profs):
        print(f"mask length {len(mask)} != {len(profs)}"); print(out[-2000:]); sys.exit(4)
    ref, mode = REFS[a.ref]
    bad = []
    for (rows, cols), lk in zip(profs, mask):
        pk = ref(a.m, a.n, a.s, a.t, a.w, rows, cols)
        if (mode == "eq" and lk != pk) or (mode == "subset" and lk and not pk):
            bad.append((rows, cols, lk, pk))
    kills = sum(mask)
    print(f"profiles={len(profs)} lean_kills={kills} python_kills={sum(ref(a.m,a.n,a.s,a.t,a.w,r,c) for r,c in profs)} mismatches={len(bad)}")
    for b in bad[:5]:
        print("  MISMATCH rows=%s cols=%s lean=%s python=%s" % b)
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()

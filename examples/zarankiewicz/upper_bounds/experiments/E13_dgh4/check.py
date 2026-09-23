#!/usr/bin/env python3
"""Lean vs Python agreement of the DGH prune on random profiles (adapted from
experiments/E8_counting/check_masks.py).

    python experiments/E13_dgh4/check.py [--samples 300] [--seed 1] [--cells "9,10,3,3,55;11,21,3,3,117;..."]

For every cell one Lean process evaluates `(argDGH P).kill`, `(argDGHCol P).kill` and
`(argDGHRow P).kill` (declarations inlined from lean/ZarPrune/DGH.lean into a scratch file
lean/Attempts/dgh_chk_<cell>.lean) on random and near-admissible profiles; the masks must equal
the Python reference `dgh_kill` / `dgh_col_kill` of experiments/E13_dgh4/dgh.py exactly.
Exit code 0 iff every sampled profile agrees on every cell and the axioms are within
{propext, Quot.sound, Classical.choice}.
"""
import argparse, os, random, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
LEAN = os.path.join(UB, "lean")
sys.path.insert(0, HERE)
from dgh import dgh_kill, dgh_col_kill  # noqa: E402
from make_files import dgh_body  # noqa: E402

DEFAULT_CELLS = "9,10,3,3,55;11,21,3,3,117;9,12,3,3,64;13,17,3,3,117;8,9,2,2,27;9,9,4,4,62;10,20,3,3,103;7,7,3,3,33"


def profiles(rnd, m, n, w, samples):
    out = []
    for _ in range(samples):
        rows = sorted((rnd.randint(0, n) for _ in range(m)), reverse=True)
        cols = sorted((rnd.randint(0, m) for _ in range(n)), reverse=True)
        out.append((rows, cols))
    for _ in range(samples // 2):
        rows = sorted((max(0, min(n, round(w / m) + rnd.randint(-2, 2))) for _ in range(m)), reverse=True)
        cols = sorted((max(0, min(m, round(w / n) + rnd.randint(-2, 2))) for _ in range(n)), reverse=True)
        out.append((rows, cols))
    for _ in range(samples // 2):  # tight, near-regular columns (where DGH bites)
        base = round(w / n)
        cols = sorted((max(0, min(m, base + rnd.choice([0, 0, 0, 1, -1]))) for _ in range(n)), reverse=True)
        rows = sorted((max(0, min(n, round(sum(cols) / m) + rnd.choice([0, 0, 1, -1]))) for _ in range(m)), reverse=True)
        out.append((rows, cols))
    return out


def run_cell(cell, samples, seed, body):
    m, n, s, t, w = cell
    rnd = random.Random(seed * 1000 + m * 37 + n)
    profs = profiles(rnd, m, n, w, samples)
    lines = ["import ZarPrune", "set_option autoImplicit false", "namespace ZarPrune.DGHChk", body, "end ZarPrune.DGHChk", "",
             f"abbrev chkP : ZarPrune.Params := {{ m := {m}, n := {n}, s := {s}, t := {t}, w := {w} }}",
             "def chkProfile (r c : List Nat) : ZarPrune.Profile chkP.m chkP.n := { row := fun i => r.getD i.val 0, col := fun j => c.getD j.val 0 }",
             "def chkBoth : ZarPrune.Prune chkP := ZarPrune.DGHChk.argDGH chkP",
             "def chkCol : ZarPrune.Prune chkP := ZarPrune.DGHChk.argDGHCol chkP",
             "def chkRow : ZarPrune.Prune chkP := ZarPrune.DGHChk.argDGHRow chkP",
             "#print axioms chkBoth"]
    chunk = 150
    for tag, prune in (("BOTH", "chkBoth"), ("COL", "chkCol"), ("ROW", "chkRow")):
        for st in range(0, len(profs), chunk):
            lits = ",\n  ".join("([" + ",".join(map(str, r)) + "],[" + ",".join(map(str, c)) + "])" for r, c in profs[st:st + chunk])
            lines.append(f"#eval IO.println (\"MASK{tag}:\" ++ String.intercalate \",\" ((([\n  {lits}] : List (List Nat × List Nat)).map fun p => if {prune}.kill (chkProfile p.1 p.2) then \"true\" else \"false\")))")
    path = os.path.join(LEAN, "Attempts", f"dgh_chk_m{m}_n{n}_s{s}_t{t}.lean")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    proc = subprocess.run(["lake", "env", "lean", path], cwd=LEAN, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    errs = [l for l in out.splitlines() if ": error:" in l]
    if errs:
        print("LEAN ERRORS:\n" + "\n".join(errs[:10]))
        return False
    ax = re.search(r"'chkBoth' depends on axioms: \[([^\]]*)\]", out)
    axioms = [x.strip() for x in ax.group(1).split(",")] if ax else []
    if not set(axioms) <= {"propext", "Quot.sound", "Classical.choice"}:
        print("BAD AXIOMS", axioms)
        return False
    masks = {}
    for tag in ("BOTH", "COL", "ROW"):
        mk = []
        for mobj in re.finditer(rf"^MASK{tag}:((?:true|false)(?:,(?:true|false))*)\s*$", out, flags=re.M):
            mk.extend(x.strip() == "true" for x in mobj.group(1).split(","))
        if len(mk) != len(profs):
            print(f"{tag}: mask length {len(mk)} != {len(profs)}"); print(out[-1500:])
            return False
        masks[tag] = mk
    bad = 0
    for i, (rows, cols) in enumerate(profs):
        pb = dgh_kill(m, n, s, t, w, rows, cols)
        pc = dgh_col_kill(m, s, t, cols)
        pr = dgh_col_kill(n, t, s, rows)
        if masks["BOTH"][i] != pb or masks["COL"][i] != pc or masks["ROW"][i] != pr:
            bad += 1
            if bad <= 3:
                print(f"  MISMATCH rows={rows} cols={cols} lean=({masks['BOTH'][i]},{masks['COL'][i]},{masks['ROW'][i]}) python=({pb},{pc},{pr})")
    print(f"cell {cell}: profiles={len(profs)} lean_kills both/col/row={sum(masks['BOTH'])}/{sum(masks['COL'])}/{sum(masks['ROW'])} "
          f"python_kills={sum(dgh_kill(m, n, s, t, w, r, c) for r, c in profs)} mismatches={bad} axioms={axioms}")
    os.remove(path)
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cells", default=DEFAULT_CELLS)
    a = ap.parse_args()
    body = dgh_body()
    ok = True
    for chunk in a.cells.split(";"):
        cell = tuple(int(x) for x in chunk.split(","))
        ok = run_cell(cell, a.samples, a.seed, body) and ok
    print("ALL AGREE" if ok else "DISAGREEMENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

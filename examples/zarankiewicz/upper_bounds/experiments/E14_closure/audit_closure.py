"""E14 audit of lean/ZarPrune/Closure.lean (owner G-closure).

Generates lean/Attempts/closure_audit.lean (Closure.lean inlined, so no `lake build` is needed) with
  * `#print axioms` on every `def`/`theorem` of Closure.lean,
  * `#eval` of `genRows` / `genCols` on a handful of instances,
  * a few semantic `#eval`s (sortDesc, sortedProfileOf, mkProfile, pairs, genParts),
runs `lake env lean` on it and compares the Lean enumerator with the Python one
(`zar_ub/partitions.py`, the same Tan Algorithm 1) *including the order of the lists*.

Run from examples/zarankiewicz/upper_bounds:
    ZAR_UB_NO_LLM=1 ../../../.venv/bin/python experiments/E14_closure/audit_closure.py
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import time

UB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, UB)

from zar_ub.known import Instance, Ledger  # noqa: E402
from zar_ub.partitions import row_partitions, column_partitions  # noqa: E402
from zar_ub import closure as C  # noqa: E402
from zar_ub.ledger import lean_fact_list  # noqa: E402

INSTS = [
    (Instance(9, 9, 3, 3, 50), "pure"),
    (Instance(9, 10, 3, 3, 55), "pure"),
    (Instance(10, 20, 3, 3, 103), "tan2022"),
    (Instance(7, 7, 3, 3, 34), "tan2022"),
    (Instance(8, 8, 2, 2, 25), "pure"),
    (Instance(9, 9, 4, 4, 62), "pure"),
    (Instance(11, 21, 3, 3, 117), "tan2022"),
    (Instance(10, 21, 3, 3, 107), "tan2022"),
    (Instance(11, 19, 3, 3, 107), "tan2022"),
    (Instance(11, 20, 3, 3, 112), "tan2022"),
]


def main() -> int:
    with open(C.CLOSURE_LEAN, encoding="utf-8") as fh:
        closure_src = fh.read()
    decls = re.findall(r"^(?:theorem|def) ([A-Za-z_][\w'.]*)", closure_src, re.M)
    L = ["import ZarPrune", "import ZarPrune.Closure", "", "namespace ZarPrune.ClosureAudit", "open ZarPrune", ""]
    L += [f"#print axioms ZarPrune.{d}" for d in decls] + [""]
    for k, (inst, trust) in enumerate(INSTS):
        pure = C.pure_facts(inst)
        ext = C.external_facts(inst, trust)
        facts = lean_fact_list([f for f, _ in pure] + ext)
        L += [f"abbrev P{k} : Params := ⟨{inst.m}, {inst.n}, {inst.s}, {inst.t}, {inst.w}⟩",
              f"abbrev facts{k} : List Fact := {facts}",
              f'#eval IO.println ("ROWS{k} " ++ toString (genRows P{k} facts{k}))',
              f'#eval IO.println ("COLS{k} " ++ toString (genCols P{k} facts{k}))']
    L += ['def A0 : Mat 3 4 := fun i j => decide ((i.val + j.val) % 3 = 0)',
          '#eval IO.println ("PROF " ++ toString (rowList A0) ++ " " ++ toString (colList A0) ++ " " ++ toString (sortedProfileOf A0))',
          '#eval IO.println ("SORT " ++ toString (sortDesc [3,1,4,1,5,9,2,6]))',
          '#eval IO.println ("PAIRS " ++ toString (pairs [1,2] ["a","b"]))',
          '#eval IO.println ("MK " ++ toString ((mkProfile ⟨3,4,3,3,0⟩ ([5,4,3],[2,2,2,1])).row 1) ++ " " ++ toString ((mkProfile ⟨3,4,3,3,0⟩ ([5,4,3],[2,2,2,1])).col 3))',
          '#eval IO.println ("GEN " ++ toString (genParts 3 4 6 2 100 (fun _ => 100)))',
          "end ZarPrune.ClosureAudit"]
    src = C._inline_closure_module("\n".join(L) + "\n")
    path = os.path.join(C.ATTEMPTS_DIR, "closure_audit.lean")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    t0 = time.time()
    out = subprocess.run(["lake", "env", "lean", path], cwd=C.LEAN_DIR, capture_output=True, text=True).stdout
    print(f"audit file {os.path.relpath(path, UB)} elaborated in {time.time() - t0:.1f}s")
    errs = [ln for ln in out.splitlines() if ": error" in ln]
    print("errors:", errs[:5])
    ax = C.parse_axioms(out)
    bad = {k: v for k, v in ax.items() if not set(v) <= set(C.ALLOWED_AXIOMS)}
    print(f"{len(ax)} declarations audited (of {len(decls)} in Closure.lean); outside the allowed axiom set: {bad}")
    summary = {}
    for k, v in ax.items():
        summary.setdefault(tuple(sorted(v)), []).append(k.split(".")[-1])
    for k, v in summary.items():
        print("  ", list(k) or "none", "->", ", ".join(v))

    def parse(tag):
        m = re.search(rf"^{tag} (.*)$", out, re.M)
        return ast.literal_eval(m.group(1))

    allok = True
    for k, (inst, trust) in enumerate(INSTS):
        ut = trust != "pure"
        pr = [list(x) for x in row_partitions(inst, Ledger(), ut)]
        pc = [list(x) for x in column_partitions(inst, Ledger(), ut)]
        lr, lc = parse(f"ROWS{k}"), parse(f"COLS{k}")
        ok = (pr == lr and pc == lc)
        allok &= ok
        print(f"{inst.tag} {trust}: python {len(pr)}x{len(pc)} lean {len(lr)}x{len(lc)} identical(incl. order)={ok}")
    print("ALL ENUMERATORS IDENTICAL:", allok)
    for tag in ["PROF", "SORT", "PAIRS", "MK", "GEN"]:
        print(re.search(rf"^{tag} .*$", out, re.M).group(0))
    return 0 if (allok and not errs and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())

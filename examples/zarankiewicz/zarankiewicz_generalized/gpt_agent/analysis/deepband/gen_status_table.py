"""Emit rows69_status.csv: per-cell provenance for every n in rows 6-9
(m <= n <= B): the z value, the source of its proof, whether Theorem 10's
arithmetic UB is tight there, and the frontier value F_m(n) with its own
provenance. This is the coordinator-requested completion ledger."""
import csv
import json
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
import sys
sys.path.insert(0, HERE)
from verify_deepband import load_frontier  # noqa: E402

T = {6: 9, 7: 15, 8: 28, 9: 40}


def truth_sources():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ev", os.path.join(BASE, "..", "evaluator.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    out = {}
    for (m, n), v in ev.KST_EXACT_VALUE.items():
        if 6 <= m <= 9:
            out[(m, n)] = (v, "published(Tan-2022)")
    for fn in ("ilp_gapband.jsonl", "ilp_results.jsonl"):
        p = os.path.join(BASE, "analysis", fn)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if "z_ilp" in r and r.get("witness_valid") \
                            and 6 <= r["m"] <= 9:
                        out.setdefault((r["m"], r["n"]),
                                       (r["z_ilp"], "workspace-ILP"))
    # this session's five (from frontier_m9_cases.jsonl when present)
    p = os.path.join(HERE, "frontier_m9_cases.jsonl")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                out.setdefault((9, r["c"]),
                               (3 * r["c"] + r["val"], "deepband-cases"))
    for m in (6, 7, 8, 9):
        B = 2 * comb(m, 3)
        for n in range(m, B + 1):
            if (m, n) not in out and n >= T[m]:
                out[(m, n)] = (3 * n + min(T[m], (B - n) // 3), "Theorem-8")
    return out


def main():
    truth = truth_sources()
    rows = []
    for m in (6, 7, 8, 9):
        B = 2 * comb(m, 3)
        R = (2 * comb(m - 1, 2)) // 3
        J = (m * R) // 4 - (2 if (m % 4 == 3 and m % 3 != 0) else 0)
        fr = load_frontier(m)
        fstat = {c: r["status"] for c, r in fr.items()}
        fval = {c: r["val"] for c, r in fr.items()}
        # dictionary-pinned F for m=9 beyond MILP range
        for n in range(m, B + 1):
            if (m, n) not in truth:
                rows.append({"m": m, "n": n, "z": "OPEN", "source": "OPEN",
                             "thm10_tight": "", "F": "", "F_status": ""})
                continue
            z, src = truth[(m, n)]
            ub10 = 3 * n + min(J, (2 * n + m * R) // 6)
            fv = fval.get(n, "")
            fs = fstat.get(n, "")
            if fs.startswith("PROVEN"):
                fs = "ILP-PROVEN"
            elif fv != "":
                fs = "LB-only"
            if fv == "" or fs == "LB-only":
                if n >= T[m]:
                    fv, fs = min(T[m], "") if False else T[m], "arithmetic"
                    fv = T[m]
                else:
                    fv, fs = z - 3 * n, "PROVEN-DICT"
            rows.append({"m": m, "n": n, "z": z, "source": src,
                         "thm10_tight": "TIGHT" if z == ub10 else "",
                         "F": fv, "F_status": fs})
    with open(os.path.join(HERE, "rows69_status.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    nopen = sum(1 for r in rows if r["z"] == "OPEN")
    print(f"rows69_status.csv: {len(rows)} cells, OPEN: {nopen}")
    for m in (6, 7, 8, 9):
        sub = [r for r in rows if r["m"] == m]
        srcs = {}
        for r in sub:
            srcs[r["source"]] = srcs.get(r["source"], 0) + 1
        print(f"  m={m}: {len(sub)} cells; sources: {srcs}; "
              f"Thm10-tight: {sum(1 for r in sub if r['thm10_tight'])}")


if __name__ == "__main__":
    main()

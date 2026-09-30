"""Step 3 (A1): assemble the NEW wide pure tables from the solved jobs, add the
proved-library Lean mask (one Lean process for all tables), and report
cases / library survivors / total d / Argument D and DGH kills.

  python gt_newtables.py assemble   -> cache/case_table_<tag>_pure_gt.json (+ baseline_lean_mask)
  python gt_newtables.py report     -> census (reads the saved tables; no Lean)

Labels: status/conflicts from ONE fresh cadical195 run at cap 2,000,000 / 180 s
(after a 2,000-cap run that gives c2000); a case still open is stored as a
censored record with budget_cap = conflicts reached and the calibrated label
d = min(max(cap, fhat), 20 cap) exactly as zar_ub.difficulty does.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from gt_common import CACHE, HERE, UB, Instance, cell_tag, read_jsonl, write_jsonl

sys.path.insert(0, os.path.join(UB, "experiments", "E13_dgh4"))
from dgh import dgh_kill  # noqa: E402

from zar_ub import casetable as ct  # noqa: E402
from zar_ub.cases import kill_col_argument_d, kill_row_argument_d  # noqa: E402
from zar_ub.difficulty import censored_d, fhat, load_calibration, log2_volume  # noqa: E402
from zar_ub.encoding import encode_case  # noqa: E402

from gt_deepen import NEW, RAW  # noqa: E402


def table_path(inst):
    return os.path.join(CACHE, f"case_table_{inst.tag}_pure_gt.json")


def assemble(no_lean=False):
    from zar_ub.known import exact_value

    raw = {}
    for j in read_jsonl(RAW):
        if j["source"] == "new_table":
            raw[(j["cell"], tuple(j["rows"]), tuple(j["cols"]))] = j
    tabs = []
    for m, n in NEW:
        inst = Instance(m, n, 3, 3, exact_value(m, n, 3, 3) + 1)
        tab = ct.build_table(inst, probe=False, use_table=False, mode="exact", verbose=False)
        calib = load_calibration(3, 3)
        cell = cell_tag(inst, "pure")
        missing = 0
        for r in tab.records:
            j = raw.get((cell, tuple(r.rows), tuple(r.cols)))
            if j is None:
                missing += 1
                continue
            cnf = encode_case(inst, r.rows, r.cols)
            st = j["status"]
            conf = int(j["run"]["conflicts"])
            p = {
                "status": st, "conflicts": conf, "seconds": round(j["cost"]["seconds"], 4),
                "budget_cap": conf if st == "unknown" else int(j["cap"]), "fixed_by_propagation": 0.0,
                "log2_volume": log2_volume(inst, r.rows),
                "nvars": cnf.nvars, "nclauses": len(cnf.clauses), "c2000": int(j["c2000"]), "fhat": None,
                "d": None, "censored": False, "witness": None, "propagations": int(j["run"]["propagations"]),
            }
            if st == "unsat":
                p["d"] = float(max(1, conf))
            elif st == "sat":
                p["witness"] = j.get("witness_ok")
                print(f"!!! SAT in a w=z+1 table {cell} rows={r.rows} cols={r.cols} witness_ok={j.get('witness_ok')}")
            else:
                fh = fhat(calib, int(j["c2000"]), p["log2_volume"])
                p["fhat"] = fh
                p["d"] = censored_d(max(conf, 1), fh)
                p["censored"] = True
            r.probe = p
            r.d = float(p["d"]) if p["d"] is not None else 1.0
            r.censored = bool(p["censored"])
        if missing:
            print(f"[{cell}] {missing} cases not solved yet -- table NOT written")
            continue
        tab.conf_cap = 2_000_000
        tab.label_mode = "censored" if any(r.censored for r in tab.records) else "exact"
        tab.path = table_path(inst)
        tab.refresh()
        tabs.append(tab)
    if tabs and not no_lean:
        masks = ct.compute_baseline_masks(tabs)
        for t, mk in zip(tabs, masks):
            if mk is not None:
                ct.set_baseline_mask(t, mk)
            else:
                print(f"[{t.instance.tag}] Lean gate FAILED; table saved without baseline mask")
    for t in tabs:
        t.save(t.path)
        print("saved", t.path, json.dumps(t.summary())[:600])


def report():
    from zar_ub.known import exact_value

    out = []
    for m, n in NEW:
        inst = Instance(m, n, 3, 3, exact_value(m, n, 3, 3) + 1)
        tab = ct.load_table(inst, path=table_path(inst))
        if tab is None:
            print("missing", table_path(inst))
            continue
        S = set(tab.scored_indices())
        dgh = [dgh_kill(m, n, 3, 3, inst.w, r.rows, r.cols) for r in tab.records]
        argd = [kill_row_argument_d(inst, r.rows, r.cols) or kill_col_argument_d(inst, r.rows, r.cols) for r in tab.records]
        dd = [ct.difficulty(r) for r in tab.records]
        tot_all = sum(dd)
        tot_s = sum(dd[i] for i in S)
        st = {}
        for r in tab.records:
            st[r.status] = st.get(r.status, 0) + 1
        row = {
            "cell": cell_tag(inst, "pure"), "z": inst.w - 1, "w": inst.w, "cases": len(tab.records), "status": st,
            "censored": sum(r.censored for r in tab.records),
            "mask": tab.baseline_lean_mask is not None,
            "library_killed": len(tab.records) - len(S), "library_survivors": len(S),
            "total_d_all": round(tot_all), "total_d_survivors": round(tot_s),
            "library_work_fraction": round(1 - tot_s / tot_all, 4) if tot_all else None,
            "argD_kills_all": sum(argd), "argD_kills_survivors": sum(argd[i] for i in S),
            "dgh_kills_all": sum(dgh), "dgh_kills_survivors": sum(dgh[i] for i in S),
            "dgh_gain_survivors": round(ct.gain(tab, dgh), 4) if S else None,
            "dgh_tail_gain": round(ct.tail_gain(tab, dgh), 4) if S else None,
            "argD_or_dgh_kills_survivors": sum((argd[i] or dgh[i]) for i in S),
            "max_d": max(dd), "d_gt_20000": sum(1 for r in tab.records if r.d > 20000),
            "d_gt_20000_survivors": sum(1 for i in S if dd[i] > 20000),
            "table_hash": tab.table_hash, "tail_share": ct.tail_share(tab),
        }
        out.append(row)
        print(json.dumps(row))
    write_jsonl(os.path.join(HERE, "ground_truth_newtables_census.jsonl"), out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["assemble", "report"])
    ap.add_argument("--no-lean", action="store_true")
    a = ap.parse_args()
    assemble(a.no_lean) if a.cmd == "assemble" else report()

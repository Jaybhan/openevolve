#!/usr/bin/env python3
"""T-10 (design §8.3): zero-shot transfer matrix of the proved prune library.

For every cached case table and every library prune (one Lean gate process per prune,
all tables at once), report how many cases the prune kills, how many of those are
scored survivors of the *whole* library minus that prune (its marginal contribution),
and the share of difficulty it removes.  Writes experiments/transfer_matrix.md/.json.

  ZAR_UB_NO_LLM=1 python experiments/transfer.py [--tables GLOB] [--prunes a,b,c]
"""
import argparse, glob, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); UB = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, UB)
from zar_ub.casetable import CaseTable, load_table  # noqa: E402
from zar_ub.lean_gate import run_gate_multi  # noqa: E402

PRUNES = ["argA", "argAT", "argD", "argDT", "argDelColWF", "argDelRowWF", "argWF", "argDGH", "evolved"]


def load_all(pattern):
    tabs = []
    for f in sorted(glob.glob(os.path.join(UB, "cache", pattern))):
        try:
            d = json.load(open(f))
            from zar_ub.known import Instance
            inst = Instance(**d["inst"])
            tab = load_table(inst, path=f)
        except Exception as e:  # noqa: BLE001
            print("skip", os.path.basename(f), e)
            continue
        if tab is None or not tab.records:
            continue
        tabs.append((inst, tab))
    return tabs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tables", default="case_table_*.json"); ap.add_argument("--prunes", default=",".join(PRUNES))
    a = ap.parse_args()
    tabs = load_all(a.tables)
    inputs = [(inst, [(r.rows, r.cols) for r in tab.records]) for inst, tab in tabs]
    print(f"{len(tabs)} tables, {sum(len(c) for _, c in inputs)} cases", flush=True)
    masks = {}
    for name in a.prunes.split(","):
        src = f"def candidate (P : Params) : Prune P := {name} P\n"
        t0 = time.time()
        res = run_gate_multi(inputs, src, timeout=900.0, tag=f"transfer_{name}")
        ok = all(r.ok for r in res)
        masks[name] = [r.kill_mask if r.ok else None for r in res]
        print(f"  {name:12s} gate {'ok' if ok else 'FAIL'} {time.time()-t0:.1f}s", flush=True)
    # per table: total work, per prune kills / marginal kills / work share
    rows = []
    for k, (inst, tab) in enumerate(tabs):
        d = [float(getattr(r, "d", 0) or (r.probe or {}).get("conflicts", 0) or 1) for r in tab.records]
        W = sum(d) or 1.0
        allmask = [False] * len(tab.records)
        for name in masks:
            m = masks[name][k]
            if m:
                allmask = [x or y for x, y in zip(allmask, m)]
        row = {"table": inst.tag, "trust": getattr(tab, "trust", ""), "cases": len(tab.records), "union_kills": sum(allmask),
               "union_work_share": round(sum(di for di, x in zip(d, allmask) if x) / W, 3)}
        for name in masks:
            m = masks[name][k]
            if not m:
                row[name] = None; continue
            others = [False] * len(tab.records)
            for o in masks:
                if o != name and o != "evolved" and masks[o][k]:  # 'evolved' may duplicate the whole library
                    others = [x or y for x, y in zip(others, masks[o][k])]
            marginal = sum(1 for i in range(len(m)) if m[i] and not others[i])
            row[name] = {"kills": sum(m), "marginal": marginal, "work_share": round(sum(d[i] for i in range(len(m)) if m[i]) / W, 3)}
        rows.append(row)
    names = list(masks)
    lines = ["# Transfer matrix — library prunes × cached tables (kills / marginal kills / work share)", "",
             "| table | trust | cases | union kills | union work | " + " | ".join(names) + " |",
             "|---|---|---|---|---|" + "---|" * len(names)]
    for r in rows:
        cells = []
        for n in names:
            v = r[n]
            cells.append("gate fail" if v is None else f"{v['kills']}/{v['marginal']}/{v['work_share']}")
        lines.append(f"| {r['table']} | {r['trust']} | {r['cases']} | {r['union_kills']} | {r['union_work_share']} | " + " | ".join(cells) + " |")
    out = "\n".join(lines)
    print(out)
    open(os.path.join(HERE, "transfer_matrix.md"), "w").write(out + "\n")
    json.dump(rows, open(os.path.join(HERE, "transfer_matrix.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

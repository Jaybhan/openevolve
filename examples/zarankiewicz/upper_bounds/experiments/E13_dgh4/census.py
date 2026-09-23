#!/usr/bin/env python3
"""E13 census: what does the DGH (v = s-1) prune kill on every cached case table?

    ZAR_UB_NO_LLM=1 python experiments/E13_dgh4/census.py [--json OUT]

For each cache/case_table_*.json: number of records, library survivors (scored_indices(), i.e.
not killed by the PROVED library mask), DGH kills among those survivors (count, share of the
difficulty-weighted work d), which orientation / which k fires, and a safety check that no
SAT-witnessed record is killed.  Also checks the two record witnesses of data/witnesses_33.json
and that the two integer forms of the inequality agree on every record.
"""
import argparse, glob, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, UB)
sys.path.insert(0, HERE)
from dgh import dgh_col_kill, dgh_col_kill_ks, dgh_violated_p8, dgh_kill  # noqa: E402
from zar_ub.casetable import table_from_json  # noqa: E402


def census_table(path):
    with open(path) as f:
        d = json.load(f)
    tab = table_from_json(d, path)
    I = tab.instance
    m, n, s, t, w = I.m, I.n, I.s, I.t, I.w
    scored = set(tab.scored_indices())
    out = {"table": os.path.basename(path), "inst": [m, n, s, t, w], "records": len(tab.records),
           "library_survivors": len(scored), "dgh_kills_all": 0, "dgh_kills_survivors": 0,
           "work_survivors": 0.0, "work_killed": 0.0, "col_kills": 0, "row_kills": 0,
           "k_hist": Counter(), "sat_killed": 0, "form_mismatch": 0, "censored_killed": 0,
           "killed_examples": []}
    for i, r in enumerate(tab.records):
        kc = dgh_col_kill_ks(m, s, t, r.cols)
        kr = dgh_col_kill_ks(n, t, s, r.rows)
        killed = bool(kc or kr)
        # the two integer forms must agree for every k
        for k in range(s, m + 1):
            if dgh_violated_p8(m, s, t, r.cols, k) != (k in kc):
                out["form_mismatch"] += 1
        for k in range(t, n + 1):
            if dgh_violated_p8(n, t, s, r.rows, k) != (k in kr):
                out["form_mismatch"] += 1
        assert killed == dgh_kill(m, n, s, t, w, r.rows, r.cols)
        if killed:
            out["dgh_kills_all"] += 1
            if r.status == "sat":
                out["sat_killed"] += 1
            if kc:
                out["col_kills"] += 1
                for k in kc:
                    out["k_hist"][f"col k={k}"] += 1
            if kr:
                out["row_kills"] += 1
                for k in kr:
                    out["k_hist"][f"row k={k}"] += 1
        if i in scored:
            dd = float(r.d) if r.d and r.d >= 1 else 1.0
            out["work_survivors"] += dd
            if killed:
                out["dgh_kills_survivors"] += 1
                out["work_killed"] += dd
                if r.censored:
                    out["censored_killed"] += 1
                if len(out["killed_examples"]) < 3:
                    out["killed_examples"].append({"rows": r.rows, "cols": r.cols, "d": dd,
                                                   "col_k": kc, "row_k": kr})
    out["work_share"] = (out["work_killed"] / out["work_survivors"]) if out["work_survivors"] else None
    out["k_hist"] = dict(out["k_hist"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(HERE, "census.json"))
    a = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(UB, "cache", "case_table_*.json")))
    rows = []
    print(f"{'table':44s} {'rec':>6s} {'surv':>6s} {'DGHall':>6s} {'DGHsurv':>7s} {'W_share':>8s} {'col/row':>8s} sat_killed mismatch")
    for p in paths:
        r = census_table(p)
        rows.append(r)
        ws = "-" if r["work_share"] is None else f"{r['work_share']:.3f}"
        print(f"{r['table']:44s} {r['records']:6d} {r['library_survivors']:6d} {r['dgh_kills_all']:6d} "
              f"{r['dgh_kills_survivors']:7d} {ws:>8s} {r['col_kills']:3d}/{r['row_kills']:<4d} "
              f"{r['sat_killed']:>4d} {r['form_mismatch']:>8d}")
    # witnesses (known K33-free record matrices): must NOT be killed
    wit_path = os.path.join(UB, "data", "witnesses_33.json")
    wit = []
    if os.path.exists(wit_path):
        with open(wit_path) as f:
            W = json.load(f)
        entries = W if isinstance(W, list) else W.get("witnesses", W.get("entries", []))
        for e in entries:
            if not isinstance(e, dict) or "rows" not in e:
                continue
            m_, n_ = e.get("m", len(e["rows"])), e.get("n", len(e["cols"]))
            s_, t_ = e.get("s", 3), e.get("t", 3)
            rows_ = e["rows"] if isinstance(e["rows"][0], int) else [sum(int(ch) for ch in row) for row in e["rows"]]
            cols_ = e["cols"] if isinstance(e["cols"][0], int) else e["cols"]
            killed = dgh_kill(m_, n_, s_, t_, sum(rows_), rows_, cols_)
            wit.append({"m": m_, "n": n_, "w": sum(rows_), "killed": killed})
            print(f"witness ({m_},{n_}) w={sum(rows_)}: DGH kill = {killed}  (must be False)")
    tot_surv = sum(r["library_survivors"] for r in rows)
    tot_kill = sum(r["dgh_kills_survivors"] for r in rows)
    tot_w = sum(r["work_survivors"] for r in rows)
    tot_wk = sum(r["work_killed"] for r in rows)
    print(f"TOTAL: library survivors {tot_surv}, DGH kills {tot_kill} ({100*tot_kill/max(tot_surv,1):.1f}%), "
          f"work share {tot_wk/max(tot_w,1e-9):.3f}; sat_killed {sum(r['sat_killed'] for r in rows)}; "
          f"form mismatches {sum(r['form_mismatch'] for r in rows)}")
    with open(a.json, "w") as f:
        json.dump({"tables": rows, "witnesses": wit}, f, indent=1)
    print("wrote", a.json)


if __name__ == "__main__":
    main()

"""E26: aggregate runs/<config>_s<i>/summary.json over seed pairs -> results/aggregate.{json,md}.

usage: python experiments/E26_dynamics/aggregate.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import genome as G  # noqa: E402

CONFIGS = os.environ.get("E26_CONFIGS", "v0_ladder,v0s1_ladder,v345_ladder,vr_ladder,v0_conc,vr_conc").split(",")
LABEL = {"shipped_ladder": "shipped v3 + E28/E29 (live)", "v0_ladder": "V0/S0 current, (gain, ladder)", "v0s1_ladder": "V0/S1 suite only, (gain, ladder)",
         "v345_ladder": "V3+V4+V5 runner-up, (gain, ladder)", "vr_ladder": "VR recommended, (gain, ladder)",
         "v0_conc": "V0/S0 current, (gain, concentration)", "vr_conc": "VR recommended, (gain, concentration)"}


def has(gs: str, *atoms) -> bool:
    g = G.parse_gstr(gs)
    return all(a in g for a in atoms)


PREFIX = os.environ.get("E26_PREFIX", "")
HORIZON = [0, 10, 20, 30, 40, 50] if not PREFIX else [0, 25, 50, 75, 100, 125, 150]


def load(cfg):
    out = []
    for d in sorted(glob.glob(os.path.join(HERE, "runs", f"{PREFIX}{cfg}_s*")), key=lambda p: int(p.rsplit("_s", 1)[1])):
        p = os.path.join(d, "summary.json")
        if os.path.exists(p) and os.path.exists(os.path.join(d, "run_meta.json")):
            with open(p) as f:
                out.append(json.load(f))
    return out


def per_run(s) -> dict:
    fin = s["final"] or {}
    ev = []
    with open(os.path.join(HERE, "runs", s["name"], "eval_log.jsonl")) as f:
        ev = [json.loads(x) for x in f if x.strip()]
    best = max(e["combined"] for e in ev)
    tied = sorted({e["genome"] for e in ev if abs(e["combined"] - best) < 1e-9})
    first_rd = next((i for i, e in enumerate(ev) if has(e["genome"], "R", "D")), None)
    first_d = next((i for i, e in enumerate(ev) if has(e["genome"], "D")), None)
    n_d_children = sum(1 for e in ev if has(e["genome"], "D"))
    kids = ev[1:]  # entry 0 = the initial program
    prod = lambda *a: (sum(1 for e in kids if has(e["genome"], *a)) / len(kids)) if kids else None  # noqa: E731
    par = []
    mp = os.path.join(HERE, "runs", s["name"], "mutator_log.tsv")
    if os.path.exists(mp):
        with open(mp) as f:
            par = [ln.split("\t")[0] for ln in f if ln.strip()]
    late = par[len(par) // 2:]
    pshare = lambda *a: (sum(1 for g in late if has(g, *a)) / len(late)) if late else None  # noqa: E731
    return {
        "name": s["name"], "seed": (s.get("mutator_seed"), s.get("db_seed")),
        "best": best, "best_at": s["best_at"], "curve": s["best_curve"], "best_genome_reported": fin.get("best_genome"),
        "best_genomes_tied": tied,
        "best_has_R_and_D": has(fin.get("best_genome") or "-", "R", "D"),
        "best_has_D": has(fin.get("best_genome") or "-", "D"),
        "best_has_U": has(fin.get("best_genome") or "-", "U"),
        "some_tied_best_has_R_and_D": any(has(g, "R", "D") for g in tied),
        "first_D_iter": first_d, "first_RD_iter": first_rd, "n_D_evaluated": n_d_children,
        "prod_share_D": prod("D"), "prod_share_RD": prod("R", "D"), "prod_share_R": prod("R"),
        "prod_share_U": prod("U"),
        "parent_share_D_late": pshare("D"), "parent_share_RD_late": pshare("R", "D"),
        "parent_share_R_late": pshare("R"), "parent_share_U_late": pshare("U"),
        "final_frac_R": fin.get("frac_with_R"),
        "final_frac_D": fin.get("frac_with_D"), "final_frac_RD": fin.get("frac_with_R_and_D"),
        "final_frac_C": fin.get("frac_with_C"), "final_frac_sound": fin.get("frac_sound"),
        "map_cells": fin.get("map_cells_occupied"), "map_cells_D": fin.get("map_cells_with_D"),
        "map_cells_RD": fin.get("map_cells_with_R_and_D"), "map_cells_C": fin.get("map_cells_with_C"),
        "archive_D": fin.get("archive_with_D"), "archive_n": fin.get("archive_n"),
        "distinct_genomes": fin.get("distinct_genomes"), "entropy": fin.get("genome_entropy_bits"),
        "n_programs": fin.get("n_programs"),
        "n_with_U": fin.get("n_with_U"), "max_score_with_U": fin.get("max_score_with_U"),
        "violations": fin.get("violations") or [],
        "frac_D_over_time": [c.get("frac_with_D") for c in s["checkpoints"]],
        "frac_RD_over_time": [c.get("frac_with_R_and_D") for c in s["checkpoints"]],
        "map_cells_D_over_time": [c.get("map_cells_with_D") for c in s["checkpoints"]],
    }


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 4) if xs else None


def agg(rows) -> dict:
    n = len(rows)
    if not n:
        return {"n_seeds": 0}
    frac = lambda k: round(sum(1 for r in rows if r[k]) / n, 3)  # noqa: E731
    return {
        "n_seeds": n,
        "best_mean": mean([r["best"] for r in rows]),
        "best_at_mean": {str(k): mean([r["curve"][min(k, len(r["curve"]) - 1)] for r in rows]) for k in HORIZON},
        "P_best_has_R_and_D": frac("best_has_R_and_D"),
        "P_some_tied_best_has_R_and_D": frac("some_tied_best_has_R_and_D"),
        "P_best_has_D": frac("best_has_D"),
        "P_best_has_U": frac("best_has_U"),
        "P_RD_ever_evaluated": round(sum(1 for r in rows if r["first_RD_iter"] is not None) / n, 3),
        "first_RD_iter_mean": mean([r["first_RD_iter"] for r in rows]),
        "n_D_evaluated_mean": mean([r["n_D_evaluated"] for r in rows]),
        "prod_share_D_mean": mean([r["prod_share_D"] for r in rows]),
        "prod_share_RD_mean": mean([r["prod_share_RD"] for r in rows]),
        "prod_share_R_mean": mean([r["prod_share_R"] for r in rows]),
        "prod_share_U_mean": mean([r["prod_share_U"] for r in rows]),
        "parent_share_D_late_mean": mean([r["parent_share_D_late"] for r in rows]),
        "parent_share_RD_late_mean": mean([r["parent_share_RD_late"] for r in rows]),
        "parent_share_R_late_mean": mean([r["parent_share_R_late"] for r in rows]),
        "parent_share_U_late_mean": mean([r["parent_share_U_late"] for r in rows]),
        "final_frac_R_mean": mean([r["final_frac_R"] for r in rows]),
        "final_frac_D_mean": mean([r["final_frac_D"] for r in rows]),
        "final_frac_RD_mean": mean([r["final_frac_RD"] for r in rows]),
        "final_frac_C_mean": mean([r["final_frac_C"] for r in rows]),
        "final_frac_sound_mean": mean([r["final_frac_sound"] for r in rows]),
        "P_D_in_final_population": round(sum(1 for r in rows if (r["final_frac_D"] or 0) > 0) / n, 3),
        "P_D_in_some_map_cell": round(sum(1 for r in rows if (r["map_cells_D"] or 0) > 0) / n, 3),
        "map_cells_D_mean": mean([r["map_cells_D"] for r in rows]),
        "map_cells_RD_mean": mean([r["map_cells_RD"] for r in rows]),
        "map_cells_C_mean": mean([r["map_cells_C"] for r in rows]),
        "map_cells_mean": mean([r["map_cells"] for r in rows]),
        "archive_D_mean": mean([r["archive_D"] for r in rows]),
        "P_D_in_archive": round(sum(1 for r in rows if (r["archive_D"] or 0) > 0) / n, 3),
        "distinct_genomes_mean": mean([r["distinct_genomes"] for r in rows]),
        "entropy_mean": mean([r["entropy"] for r in rows]),
        "n_with_U_mean": mean([r["n_with_U"] for r in rows]),
        "max_score_with_U": max((r["max_score_with_U"] or 0) for r in rows),
        "violations_total": sum(len(r["violations"]) for r in rows),
        "frac_D_over_time_mean": [mean([r["frac_D_over_time"][k] for r in rows if len(r["frac_D_over_time"]) > k])
                                  for k in range(len(rows[0]["frac_D_over_time"]))],
        "frac_RD_over_time_mean": [mean([r["frac_RD_over_time"][k] for r in rows if len(r["frac_RD_over_time"]) > k])
                                   for k in range(len(rows[0]["frac_RD_over_time"]))],
    }


def main():
    res = {"per_run": {}, "aggregate": {}}
    for c in CONFIGS:
        rows = [per_run(s) for s in load(c)]
        res["per_run"][c] = rows
        res["aggregate"][c] = agg(rows)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", f"aggregate{('_' + PREFIX.rstrip('_')) if PREFIX else ''}.json"), "w") as f:
        json.dump(res, f, indent=1)
    A = res["aggregate"]
    L = [f"# E26 aggregate {PREFIX or 'primary'} (mean over seed pairs; {HORIZON[-1]} iterations each)", ""]
    cols = [("n_seeds", "seeds"), ("best_mean", "best"), ("P_best_has_R_and_D", "P(best=R+D..)"),
            ("P_some_tied_best_has_R_and_D", "P(R+D among tied best)"), ("P_RD_ever_evaluated", "P(R+D ever made)"),
            ("final_frac_D_mean", "pop share D"), ("final_frac_RD_mean", "pop share R+D"),
            ("P_D_in_some_map_cell", "P(D elite)"), ("map_cells_D_mean", "D cells"), ("map_cells_mean", "cells"),
            ("P_D_in_archive", "P(D in archive)"), ("distinct_genomes_mean", "genomes"), ("entropy_mean", "H bits"),
            ("final_frac_sound_mean", "pop share sound"), ("P_best_has_U", "P(best has U)"),
            ("violations_total", "violations")]
    L.append("| config | " + " | ".join(h for _, h in cols) + " |")
    L.append("|---|" + "---|" * len(cols))
    for c in CONFIGS:
        a = A[c]
        L.append(f"| {LABEL[c]} | " + " | ".join(str(a.get(k)) for k, _ in cols) + " |")
    L += ["", "## selection vs mutation supply (mean over seeds)", "",
          "prod = share of evaluated children carrying the atoms (what the blind mutator supplies); "
          "parent(late) = share of parents chosen in the second half of the run carrying them; pop = final population share.", "",
          "| config | D prod / parent / pop | R+D prod / parent / pop | R prod / parent / pop | U prod / parent / in-pop count |",
          "|---|---|---|---|---|"]
    for c in CONFIGS:
        a = A[c]
        if not a.get("n_seeds"):
            continue
        L.append(f"| {LABEL[c]} | {a['prod_share_D_mean']} / {a['parent_share_D_late_mean']} / {a['final_frac_D_mean']} | "
                 f"{a['prod_share_RD_mean']} / {a['parent_share_RD_late_mean']} / {a['final_frac_RD_mean']} | "
                 f"{a['prod_share_R_mean']} / {a['parent_share_R_late_mean']} / {a['final_frac_R_mean']} | "
                 f"{a['prod_share_U_mean']} / {a['parent_share_U_late_mean']} / {a['n_with_U_mean']} |")
    L += ["", "## best score over iterations (mean over seeds)", "", "| config | " + " | ".join(map(str, HORIZON)) + " |", "|---|" + "---|" * len(HORIZON)]
    for c in CONFIGS:
        b = A[c].get("best_at_mean") or {}
        L.append(f"| {LABEL[c]} | " + " | ".join(str(b.get(str(k))) for k in HORIZON) + " |")
    L += ["", "## population share of D / R+D at every checkpoint (every 10 iterations; mean over seeds)", "",
          "| config | D per checkpoint | R+D per checkpoint |", "|---|---|---|"]
    for c in CONFIGS:
        L.append(f"| {LABEL[c]} | {A[c].get('frac_D_over_time_mean')} | {A[c].get('frac_RD_over_time_mean')} |")
    L += ["", "## per run", "", "| run | best | reported best genome | tied-best genomes | first R+D iter | pop D | "
          "pop R+D | D cells / cells | archive D | genomes | U in pop (max score) | violations |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in CONFIGS:
        for r in res["per_run"][c]:
            L.append(f"| {r['name']} | {r['best']:.4f} | {r['best_genome_reported']} | {', '.join(r['best_genomes_tied'])} | "
                     f"{r['first_RD_iter']} | {r['final_frac_D']} | {r['final_frac_RD']} | {r['map_cells_D']}/{r['map_cells']} | "
                     f"{r['archive_D']}/{r['archive_n']} | {r['distinct_genomes']} | {r['n_with_U']} "
                     f"({(r['max_score_with_U'] or 0):.4f}) | {len(r['violations'])} |")
    with open(os.path.join(HERE, "results", f"aggregate{('_' + PREFIX.rstrip('_')) if PREFIX else ''}.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()

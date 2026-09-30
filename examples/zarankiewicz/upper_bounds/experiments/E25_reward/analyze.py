"""E25 step 3: score matrix (programs x reward variants) from the cached masks, criteria C1-C5,
property tests for C4.  Offline: no Lean, no SAT.

usage: python experiments/E25_reward/analyze.py   -> results/score_matrix.{json,md}, results/cells.md
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import universe as U  # noqa: E402
from zar_ub import reward  # noqa: E402
from zar_ub import reward_variants as RV  # noqa: E402

MASKS = os.path.join(_HERE, "masks")
SUFFIX = ""
RES = os.path.join(_HERE, "results")

# program classes (for the criteria)
GENUINE = ["schema_recipe_frozen", "dgh_lib", "recipe_dgh", "schema_pool"]
SPECIALIST = ["lean_dgh4", "cert_one_train", "cert_target_12_18", "inst_dgh_10_23", "cert_close_10_22"]
EXPLOIT_REAL = ["e1_dgh_s2", "e2_easy_certs", "e3_relist"]
EXPLOIT_ORACLE = ["O_e1_clear_10_10", "O_e1_clear_gen44", "O_e1_clear_gen89", "O_e2_easy_2k"]
WEAK = ["f_weak"]
NOOP = ["initial", "instance_specific", "e3_relist"]
ORACLE_OTHER = ["O_e2_easy_median", "O_perfect_target_12_18", "O_perfect_train_12_12", "O_tail_everywhere"]


def load_all(include_gt=True):
    keys, uni = ordered_universe(include_gt)
    tabs = {k: U.load(k, uni[k]["path"]) for k in keys}
    return keys, uni, tabs


def ordered_universe(include_gt=True):
    from compute_masks import ordered_universe as ou
    return ou(include_gt)


def program_masks(name, keys):
    p = os.path.join(MASKS, f"{name}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    out = {}
    for k in keys:
        t = d["tables"].get(k)
        out[k] = None if t is None else t.get("lean")
    return {"masks": out, "raw": d}


def survivors_d(tab):
    S = reward.survivor_indices(tab, None)
    cap = int(getattr(tab, "conf_cap", 0) or 0)
    d = [reward.rec_difficulty(r, cap) for r in tab.records]
    return S, d


def oracle_masks(keys, tabs):
    """Synthetic kill masks over library survivors (NOT programs: upper bounds for exploit shapes)."""
    out = {}

    def clear(key):
        m = {}
        for k in keys:
            S, _ = survivors_d(tabs[k])
            mk = [False] * len(tabs[k].records)
            if k == key:
                for i in S:
                    mk[i] = True
            m[k] = mk
        return m

    out["O_e1_clear_10_10"] = clear("m10_n10_s3_t3_w61_pure")
    out["O_e1_clear_gen44"] = clear("m9_n9_s4_t4_w62_pure")
    out["O_e1_clear_gen89"] = clear("m8_n9_s2_t2_w27_pure")
    out["O_perfect_target_12_18"] = clear("m12_n18_s3_t3_w109")
    out["O_perfect_train_12_12"] = clear("m12_n12_s3_t3_w81_pure")
    easy2k, easymed, tailm = {}, {}, {}
    for k in keys:
        S, d = survivors_d(tabs[k])
        n = len(tabs[k].records)
        a, b, c = [False] * n, [False] * n, [False] * n
        if S:
            ds = sorted(d[i] for i in S)
            med = ds[len(ds) // 2]
            for i in S:
                if d[i] <= 2000:
                    a[i] = True
                if d[i] <= med and not reward.is_censored(tabs[k].records[i]):
                    b[i] = True
            for i in reward.top_decile(S, d):
                c[i] = True
        easy2k[k], easymed[k], tailm[k] = a, b, c
    out["O_e2_easy_2k"] = easy2k
    out["O_e2_easy_median"] = easymed
    out["O_tail_everywhere"] = tailm
    return out


def cells_for(masks, keys, uni, tabs):
    return [RV.cell_from_table(k, uni[k]["roles"], tabs[k], masks.get(k)) for k in keys]


def c4_tests(cells, cfg, rng, n_rand=300):
    """(monotone_ok, uniform_beats_single_ok, details)"""
    suite = RV.SUITES[cfg.suite]
    scored = [c for c in cells if c.n_surv > 0 and any(RV._in(c, suite[r]) for r in suite)]
    keys = [c.key for c in scored]
    mono_fail = 0
    for _ in range(n_rand):
        g = {k: rng.random() ** 2 for k in keys}
        if rng.random() < 0.3:
            g[rng.choice(keys)] = 1.0
        tl = {k: min(1.0, g[k] * rng.random() * 2) for k in keys}
        base = RV.verified_score(RV.with_gains(cells, g, tl), cfg)
        k = rng.choice(keys)
        g2 = dict(g)
        g2[k] = min(1.0, g[k] + rng.random() * (1 - g[k]))
        tl2 = dict(tl)
        tl2[k] = max(tl[k], min(1.0, tl[k] + rng.random() * (1 - tl[k])))
        if RV.verified_score(RV.with_gains(cells, g2, tl2), cfg) < base - 1e-12:
            mono_fail += 1
    uni_fail = []
    for x in (0.05, 0.2, 0.5, 1.0):
        su = RV.verified_score(RV.with_gains(cells, {k: x for k in keys}), cfg)
        for k in keys:
            s1 = RV.verified_score(RV.with_gains(cells, {k: x}), cfg)
            if not su > s1 + 1e-12:
                uni_fail.append((x, k, round(su, 4), round(s1, 4)))
    zero = RV.verified_score(RV.with_gains(cells, {}), cfg)
    return mono_fail == 0, not uni_fail, {"mono_fail": mono_fail, "uniform_fail": uni_fail[:5],
                                          "zero_gain_score": zero}


def scored_gain(cells, cfg) -> bool:
    suite = RV.SUITES[cfg.suite]
    return any(c.n_surv > 0 and c.gain > 0 and any(RV._in(c, suite[r]) for r in suite) for c in cells)


def criteria(v, cfg, matrix, cells, rng):
    col = {p: matrix[p][v] for p in matrix}
    get = lambda p: col.get(p, float("nan"))  # noqa: E731
    # C1: floor exact for no-ops, 0 for hard-zero, <= 0.19 unverified, > 0.20 iff a scored cell has gain
    c1_bad = []
    for p, cs in cells.items():
        pos = scored_gain(cs, cfg)
        sc = get(p)
        if (pos and not sc > 0.20) or (not pos and abs(sc - 0.20) > 1e-12):
            c1_bad.append(p)
    c1 = (not c1_bad and get("_unsound (hard-zero)") == 0.0 and 0 < get("_unverified L3 (strong)") <= 0.19)
    rec, dgh = get("schema_recipe_frozen"), get("dgh_lib")
    c2 = (dgh - 0.20) >= 0.02 and (dgh - 0.20) >= 0.5 * (rec - 0.20)
    genuine_min = min(rec, dgh)
    real = {p: get(p) for p in EXPLOIT_REAL if p in col}
    orac = {p: get(p) for p in EXPLOIT_ORACLE if p in col}
    c3 = all(x < genuine_min for x in real.values())
    c3o = all(x < genuine_min for x in orac.values())
    mono, unif, det = c4_tests(cells["initial"], cfg, rng)
    v0rec = matrix.get("schema_recipe_frozen", {}).get("V0/S0 current", float("nan"))
    c5 = (rec - 0.20) >= 0.8 * (v0rec - 0.20) and get("recipe_dgh") >= max(rec, dgh) - 1e-12
    return {"C1": c1, "C1_bad": c1_bad, "C2": c2, "C3": c3, "C3_oracle": c3o, "C4": mono and unif, "C5": c5,
            "C2_ratio": (dgh - 0.2) / (rec - 0.2) if rec > 0.2 else float("nan"),
            "C3_worst": max(real.items(), key=lambda kv: kv[1]),
            "C3o_worst": max(orac.items(), key=lambda kv: kv[1]),
            "C4_detail": det, "C5_retained": (rec - 0.2) / (v0rec - 0.2),
            "exploit_vs_weak": {p: x <= get("f_weak") for p, x in real.items()}}


GT_DIR = os.path.join(os.path.dirname(_HERE), "E24_difficulty")


def gt_path():
    for f in ("ground_truth.jsonl", "ground_truth_interim.jsonl", "ground_truth_initial.jsonl"):
        p = os.path.join(GT_DIR, f)
        if os.path.exists(p):
            return p
    return None


def relabel(tabs, path):
    """IN MEMORY ONLY (tables are never saved): d := A1's exact conflicts where the ground truth has
    status unsat/sat; for status unknown d := max(table label, A1's lower bound).  Returns stats."""
    gt = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            gt[(r["cell"], tuple(r["rows"]), tuple(r["cols"]))] = r
    st = {"exact": 0, "raised": 0, "kept": 0, "missing": 0}
    for k, tab in tabs.items():
        cell = k[:-3] if k.endswith("_gt") else k  # A1 names its new tables' cells without the _gt suffix
        for r in tab.records:
            g = gt.get((cell, tuple(r.rows), tuple(r.cols)))
            if g is None:
                st["missing"] += 1
                continue
            if g["status"] in ("unsat", "sat"):
                r.d, r.censored = float(max(1, g["d"])), False
                st["exact"] += 1
            elif g["d"] > (r.d or 0):
                r.d = float(g["d"])
                st["raised"] += 1
            else:
                st["kept"] += 1
    return st


def build_cells(include_gt=True, labels="table"):
    """(keys, uni, tabs, progs, cells) for every program with cached masks and every oracle."""
    keys, uni, tabs = load_all(include_gt)
    if labels == "gt":
        p = gt_path()
        print(f"[analyze] relabelling from {p}: {relabel(tabs, p)}")
    progs = {}
    for f in sorted(os.listdir(MASKS)):
        if f.endswith(".json"):
            name = f[:-5]
            pm = program_masks(name, keys)
            missing = [k for k in keys if pm["masks"].get(k) is None and tabs[k].records
                       and pm["raw"]["tables"].get(k, {}).get("ladder") != 5]
            progs[name] = {"masks": pm["masks"], "kind": "program", "missing": missing,
                           "ladders": {k: pm["raw"]["tables"].get(k, {}).get("ladder") for k in keys},
                           "gate": pm["raw"].get("gate", {})}
    for name, m in oracle_masks(keys, tabs).items():
        progs[name] = {"masks": m, "kind": "oracle", "missing": [], "ladders": {}, "gate": {}}
    cells = {name: cells_for(p["masks"], keys, uni, tabs) for name, p in progs.items()}
    return keys, uni, tabs, progs, cells


def add_soundness_rows(matrix, cells, variants):
    unv = reward.unverified_score(3, 0.90, 0.5, 0.5)  # a strong L3 sketch
    for v, cfg in variants.items():
        matrix.setdefault("_unsound (hard-zero)", {})[v] = RV.score(cells["initial"], cfg, hard_zero=True)
        matrix.setdefault("_unverified L3 (strong)", {})[v] = RV.score(cells["initial"], cfg, ladder=3, unverified=unv)
    return matrix


def main():
    include_gt = "--no-gt" not in sys.argv
    labels = "gt" if "--gt-labels" in sys.argv else "table"
    global SUFFIX
    SUFFIX = "_gtlabels" if labels == "gt" else ""
    keys, uni, tabs, progs, cells = build_cells(include_gt, labels)
    variants = RV.VARIANTS
    matrix = {name: {v: RV.verified_score(cells[name], cfg) for v, cfg in variants.items()} for name in progs}
    # soundness rows (C1): the non-verified branches are reward.py's, unchanged by every variant
    add_soundness_rows(matrix, cells, variants)
    # criteria
    rng = random.Random(25)
    crit = {v: criteria(v, cfg, matrix, cells, rng) for v, cfg in variants.items()}
    comps = {name: {v: RV.components(cells[name], cfg) for v, cfg in variants.items()} for name in progs}
    os.makedirs(RES, exist_ok=True)
    json.dump({"matrix": matrix, "criteria": crit, "components": comps,
               "cells": {n: [c.__dict__ for c in cs] for n, cs in cells.items()},
               "missing": {n: p["missing"] for n, p in progs.items()},
               "universe": {k: uni[k]["roles"] for k in keys}}, open(os.path.join(RES, f"score_matrix{SUFFIX}.json"), "w"),
              indent=1, default=str)
    write_md(matrix, crit, cells, progs, keys, uni, variants)


ORDER = (NOOP[:2] + ["schema_recipe_frozen", "schema_pool", "dgh_lib", "lean_dgh4", "recipe_dgh", "f_weak"]
         + ["cert_one_train", "cert_target_12_18", "inst_dgh_10_23", "cert_close_10_22"]
         + EXPLOIT_REAL + EXPLOIT_ORACLE + ORACLE_OTHER + ["_unverified L3 (strong)", "_unsound (hard-zero)"])


def write_md(matrix, crit, cells, progs, keys, uni, variants):
    vs = list(variants)
    L = ["# E25 score matrix (verified branch; rows = programs, columns = reward variants)", ""]
    L.append("| program | " + " | ".join(vs) + " |")
    L.append("|---|" + "---|" * len(vs))
    names = [p for p in ORDER if p in matrix] + [p for p in matrix if p not in ORDER]
    for p in names:
        tag = " (oracle)" if progs.get(p, {}).get("kind") == "oracle" else ""
        L.append(f"| {p}{tag} | " + " | ".join(f"{matrix[p][v]:.4f}" for v in vs) + " |")
    L += ["", "## Criteria", "",
          "| variant | C1 | C2 (dgh/recipe uplift) | C3 real exploits (worst) | C3 oracle stress (worst) | C4 | C5 (recipe uplift kept) |",
          "|---|---|---|---|---|---|---|"]
    pf = lambda b: "pass" if b else "FAIL"  # noqa: E731
    for v in vs:
        c = crit[v]
        L.append(f"| {v} | {pf(c['C1'])}{'' if c['C1'] else ' ' + ','.join(c['C1_bad'][:3])} | {pf(c['C2'])} ({c['C2_ratio']:.2f}) | "
                 f"{pf(c['C3'])} ({c['C3_worst'][0]} {c['C3_worst'][1]:.4f}) | "
                 f"{pf(c['C3_oracle'])} ({c['C3o_worst'][0]} {c['C3o_worst'][1]:.4f}) | "
                 f"{pf(c['C4'])} | {pf(c['C5'])} ({c['C5_retained']:.2f}) |")
    L += ["", "## Per-cell verified gain (K^L beyond the library) of each program", ""]
    L.append("| program | " + " | ".join(k.replace("_s3_t3", "").replace("_pure", "p") for k in keys) + " |")
    L.append("|---|" + "---|" * len(keys))
    L.append("| roles | " + " | ".join(",".join(r for r in uni[k]["roles"] if r != "gen_default") for k in keys) + " |")
    c0 = {c.key: c for c in cells["initial"]}
    L.append("| survivors / W | " + " | ".join(f"{c0[k].n_surv} / {c0[k].W:.3g}" for k in keys) + " |")
    for p in names:
        if p not in cells:
            continue
        cc = {c.key: c for c in cells[p]}
        L.append(f"| {p} | " + " | ".join(
            ("" if cc[k].n_surv == 0 else (("**1.00c**" if cc[k].closed else f"{cc[k].gain:.3f}") if cc[k].gain > 0 else "."))
            for k in keys) + " |")
    open(os.path.join(RES, f"score_matrix{SUFFIX}.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[:len(names) + 8 + len(vs)]))


if __name__ == "__main__":
    main()

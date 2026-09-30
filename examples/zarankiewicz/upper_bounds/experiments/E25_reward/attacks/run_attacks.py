"""E25 / R3-critic: adversarial attacks on R1's recommended reward VR (offline; no SAT, no Lean, no LP).

usage (from the upper_bounds dir):
    ZAR_UB_NO_LLM=1 .venv/bin/python experiments/E25_reward/attacks/run_attacks.py
writes attacks/results.json and attacks/results.md.  Deterministic (seed 25 for the one random draw).
"""
from __future__ import annotations

import json
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import common as C  # noqa: E402
import fixes as F  # noqa: E402
from common import RV, reward  # noqa: E402

SMALL_W = 1.0e6  # "small table" (table-label W)


def is_target(env, k):
    return bool(set(env.uni[k]["roles"]) & {"target", "target_x"})


def is_scored_s1(env, k):
    return bool(set(env.uni[k]["roles"]) & {"train", "wide", "wide_gt", "target", "target_x", "gen"})


def clear(env, pred):
    m = C.empty_masks(env)
    for k in env.keys:
        if pred(k):
            for i in env.S[k]:
                m[k][i] = True
    return m


def kill_where(env, pred_case):
    m = C.empty_masks(env)
    for k in env.keys:
        for i in env.S[k]:
            if pred_case(k, i):
                m[k][i] = True
    return m


def build(env):
    A = {}   # name -> (kind, mask, note)
    rec = C.r1_mask("schema_recipe_frozen", env)
    dgh = C.r1_mask("dgh_lib", env)
    # ---------------- references (genuine, R1's Lean-gated masks) ----------------
    A["ref: schema_recipe_frozen"] = ("real", rec, "R1 genuine: 34 Farkas certs")
    A["ref: dgh_lib"] = ("real", dgh, "R1 genuine: DGH")
    A["ref: recipe_dgh"] = ("real", C.r1_mask("recipe_dgh", env), "R1 genuine: recipe + DGH")
    A["ref: f_weak"] = ("real", C.r1_mask("f_weak", env), "R1 weak general rule")
    A["ref: cert_target_12_18"] = ("real", C.r1_mask("cert_target_12_18", env), "all pool certs of (12,18) only")
    for p in ("schema_pool", "cert_one_train", "inst_dgh_10_23", "e1_dgh_s2", "e2_easy_certs", "e3_relist"):
        A[f"r1: {p}"] = ("real", C.r1_mask(p, env), "R1 benchmark program")
    # ---------------- A: tiny tables ----------------
    Wt = {k: sum(env.d["table"][k][i] for i in env.S[k]) for k in env.keys}
    Wl = {k: sum(env.d["gtlb"][k][i] for i in env.S[k]) for k in env.keys}
    A["A1 clear GEN (9,9;4,4)"] = ("oracle", clear(env, lambda k: k == "m9_n9_s4_t4_w62_pure"),
                                   "25 survivors, every case d<=1,080; 0 verified rules kill >2 of them")
    A["A2 clear (9,12,64)"] = ("oracle", clear(env, lambda k: k == "m9_n12_s3_t3_w64_pure"), "8 survivors, W 9.6e3")
    A["A2 clear (10,10,61)"] = ("oracle", clear(env, lambda k: k == "m10_n10_s3_t3_w61_pure"), "10 survivors, W 9.9e3")
    A["A3 clear every table with W_lb<1e5"] = ("oracle", clear(env, lambda k: 0 < Wl[k] < 1e5 and is_scored_s1(env, k)),
                                                "; ".join(k for k in env.keys if 0 < Wl[k] < 1e5 and is_scored_s1(env, k)))
    small = lambda k: 0 < Wt[k] <= SMALL_W and is_scored_s1(env, k)  # noqa: E731
    A["A5 real: pool certs on small tables only"] = ("real", C.pool_mask(env, keep_table=small),
                                                      "instance-gated Farkas certs, tables with W<=1e6")
    A["A4 real: pool certs + DGH on small tables only"] = (
        "real", C.union(C.pool_mask(env, keep_table=small), C.restrict(dgh, small)),
        "instance-gated: every verified piece on tables with W<=1e6, nothing elsewhere")
    # ---------------- B: easy cases ----------------
    for T in (2000, 5000, 20000):
        A[f"B easy exact d<={T}"] = ("oracle", kill_where(env, lambda k, i, T=T: env.exact[k][i] and not reward.is_censored(env.tabs[k].records[i]) and env.d["table"][k][i] <= T),
                                     "every exactly-labelled survivor with d<=T (program-visible labels)")

    def easy_cert(k, c, T=20000):
        return all((not reward.is_censored(env.tabs[k].records[j])) and env.d["table"][k][j] <= T for j in c["kills"])
    A["B4 real: pool certs killing only exact d<=20k"] = ("real", C.pool_mask(env, keep_cert=easy_cert), "e2 widened to 20k")
    # ---------------- C: label inflation / censoring ----------------
    A["C1 real: cert_close_10_22"] = ("real", C.r1_mask("cert_close_10_22", env),
                                      "1 cert closes (10,22,111); true W 58,333 vs label W 424,114")
    A["C2 tail-index sniper (targets)"] = ("oracle", kill_where(env, lambda k, i: is_target(env, k) and i in set(reward.top_decile(env.S[k], env.d['table'][k]))),
                                           "kill H_I on every target = the index-first 10% of 400k-tied censored cases")
    # deepened-sample cases on targets: easy half vs hard half of true d (equal table-label value)
    samp = {}
    for k in env.keys:
        if not is_target(env, k):
            continue
        cs = [i for i in env.S[k] if reward.is_censored(env.tabs[k].records[i]) and env.src[k][i] == "deepen"]
        samp[k] = sorted(cs, key=lambda i: (env.d["gtlb"][k][i], i))
    easyh = {k: set(v[: len(v) // 2]) for k, v in samp.items()}
    hardh = {k: set(v[len(v) - len(v) // 2:]) for k, v in samp.items()}
    A["C3a deepened censored, EASY half"] = ("oracle", kill_where(env, lambda k, i: i in easyh.get(k, ())),
                                            "same count as C3b; all labelled 400k")
    A["C3b deepened censored, HARD half"] = ("oracle", kill_where(env, lambda k, i: i in hardh.get(k, ())), "")
    # ---------------- D: family averaging ----------------
    rng = random.Random(25)
    for k in ("m16_n17_s3_t3_w134", "m13_n19_s3_t3_w123", "m12_n18_s3_t3_w109", "m9_n23_s3_t3_w104"):
        S = list(env.S[k])
        pick = set(rng.sample(S, max(1, round(0.10 * len(S)))))
        A[f"D1 random 10% of {k.replace('_s3_t3', '')}"] = ("oracle", kill_where(env, lambda kk, i, k=k, p=pick: kk == k and i in p), "")
    # ---------------- G: tail on small tables ----------------
    A["G tail top-decile of tables with <=30 survivors"] = ("oracle", kill_where(env, lambda k, i: 0 < len(env.S[k]) <= 30 and i in set(reward.top_decile(env.S[k], env.d['table'][k]))), "")
    return A


def main():
    env = C.load_env(include_gt=True)
    env_nogt = C.load_env(include_gt=False)
    A = build(env)
    fx = F.FIXES
    rows = {}
    for name, (kind, m, note) in A.items():
        r = {"kind": kind, "note": note,
             "V0 table": C.score(env, m, C.V0, "table"),
             "V0 gt": C.score(env, m, C.V0, "gt"),
             "work_lb_train+target": C.work_removed(env, m, "gtlb"),
             "work_lb_target": C.target_work_removed(env, m, "gtlb"),
             "work_table_train+target": C.work_removed(env, m, "table")}
        for fn, f in fx.items():
            r[fn] = F.score(env, m, f)
        c = C.comps(env, m, C.VR, "table")
        r["VR comps (table)"] = {k: round(v, 4) for k, v in c.items()}
        rows[name] = r
    # D2: the live suite without A1's *_pure_gt tables (if the integrator does not add them)
    m912 = clear(env_nogt, lambda k: k == "m9_n12_s3_t3_w64_pure")
    rec_nogt = C.r1_mask("schema_recipe_frozen", env_nogt)
    dgh_nogt = C.r1_mask("dgh_lib", env_nogt)
    nogt = {"clear (9,12,64)": C.score(env_nogt, m912, C.VR), "recipe": C.score(env_nogt, rec_nogt, C.VR),
            "dgh": C.score(env_nogt, dgh_nogt, C.VR)}
    # marginal credit per 1e6 conflicts killed (uniform small fraction of one table), VR and VR*
    marg = {}
    for k in env.keys:
        if not env.S[k]:
            continue
        W = sum(env.d["table"][k][i] for i in env.S[k])
        # emulate a uniform 1% gain on table k via Cell edits
        for lab, f in (("VR", fx["VR (R1)"]), ("VR*", fx["VR* = F1+F2+F3+F4"]), ("VR**", fx["VR** = F1b+F3"])):
            cs = C.cells(env, C.empty_masks(env), f.view, excess=f.excess, tail_rule=f.tail_rule)
            base = RV.verified_score(cs, f.cfg)
            cs2 = [RV.replace(c, gain=0.01) if c.key == k else c for c in cs]
            dW = 0.01 * sum(max(0.0, env.d[f.view][k][i] - f.excess) for i in env.S[k])
            ds = RV.verified_score(cs2, f.cfg) - base
            marg.setdefault(k, {})[lab] = (ds / dW * 1e6) if dW > 0 else 0.0
        marg[k]["W_table"] = W
    # tail-index check: GT true d of H_I vs the rest among deepened censored target cases
    tailchk = {}
    for k in env.keys:
        if not is_target(env, k):
            continue
        H = set(reward.top_decile(env.S[k], env.d["table"][k]))
        dee = [i for i in env.S[k] if reward.is_censored(env.tabs[k].records[i]) and env.src[k][i] == "deepen"]
        h = sorted(env.d["gtlb"][k][i] for i in dee if i in H)
        o = sorted(env.d["gtlb"][k][i] for i in dee if i not in H)
        med = lambda v: v[len(v) // 2] if v else None  # noqa: E731
        tailchk[k] = {"H_size": len(H), "H_index_max": max(H) if H else None, "deepened_in_H": len(h),
                      "median_lb_H": med(h), "median_lb_rest": med(o), "deepened_rest": len(o)}
    # do genuine rules kill truly easier censored target cases?  (deepened sample only)
    bias = {}
    for pname in ("schema_recipe_frozen", "dgh_lib", "schema_pool"):
        m = C.r1_mask(pname, env)
        kk, nk = [], []
        for k in env.keys:
            if not is_target(env, k):
                continue
            for i in env.S[k]:
                if reward.is_censored(env.tabs[k].records[i]) and env.src[k][i] == "deepen":
                    (kk if m[k][i] else nk).append(env.d["gtlb"][k][i])
        kk.sort(); nk.sort()
        bias[pname] = {"n_killed": len(kk), "median_lb_killed": kk[len(kk) // 2] if kk else None,
                       "mean_lb_killed": sum(kk) / len(kk) if kk else None,
                       "n_not": len(nk), "median_lb_not": nk[len(nk) // 2] if nk else None,
                       "mean_lb_not": sum(nk) / len(nk) if nk else None}
    # R1's criteria C2, C3 (real + oracle), C5 per fix, over this attack set
    crit = {}
    GEN = ["ref: schema_recipe_frozen", "ref: dgh_lib"]
    for fn in fx:
        col = {n: rows[n][fn] for n in rows}
        rec, dgh = col["ref: schema_recipe_frozen"], col["ref: dgh_lib"]
        gmin = min(rec, dgh)
        bad_real = {n: round(v, 4) for n, v in col.items() if rows[n]["kind"] == "real" and not n.startswith("ref") and
                    n.split(": ")[-1] in ("e1_dgh_s2", "e2_easy_certs", "e3_relist") and v >= gmin}
        bad_orc = {n: round(v, 4) for n, v in col.items() if rows[n]["kind"] == "oracle" and v >= gmin and
                   (n.startswith("A") or n.startswith("B") or n.startswith("G"))}
        wk = col["ref: f_weak"]
        inv = {n: round(v, 4) for n, v in col.items() if n.startswith(("A1", "A2")) and v > wk}
        crit[fn] = {"recipe": round(rec, 4), "dgh": round(dgh, 4), "recipe_dgh": round(col["ref: recipe_dgh"], 4),
                    "C2 dgh/recipe uplift": round((dgh - .2) / (rec - .2), 2) if rec > .2 else None,
                    "C5 recipe uplift / V0": round((rec - .2) / (rows["ref: schema_recipe_frozen"]["V0 table"] - .2), 2),
                    "C3 real exploits >= genuine min": bad_real, "exploit-shape oracles >= genuine min": bad_orc,
                    "tiny-table clears > f_weak": inv}
    out = {"criteria": crit, "rows": rows, "no_gt_suite": nogt, "marginal_per_1e6": marg, "tail_index_check": tailchk,
           "kill_bias_on_deepened_targets": bias, "fixes": list(fx)}
    json.dump(out, open(os.path.join(_HERE, "results.json"), "w"), indent=1, default=str)
    write_md(out)


def write_md(out):
    fxs = out["fixes"]
    L = ["# R3 attack matrix (verified branch)", "",
         "| attack | kind | V0 table | " + " | ".join(fxs) + " | work removed (lb, train+target) | of which target |",
         "|---|---|---|" + "---|" * len(fxs) + "---|---|"]
    for n, r in out["rows"].items():
        L.append(f"| {n} | {r['kind']} | {r['V0 table']:.4f} | " + " | ".join(f"{r[f]:.4f}" for f in fxs)
                 + f" | {r['work_lb_train+target']:.3g} | {r['work_lb_target']:.3g} |")
    L += ["", "## R1 criteria per fix (this attack set)", "", "```", json.dumps(out["criteria"], indent=1), "```"]
    L += ["", "## VR components (table labels)", "", "| attack | " + " | ".join(["G_train", "G_target", "G_gen", "Tail", "Depth", "Close"]) + " |",
          "|---|---|---|---|---|---|---|"]
    for n, r in out["rows"].items():
        c = r["VR comps (table)"]
        L.append(f"| {n} | " + " | ".join(f"{c[k]:.3f}" for k in ("G_train", "G_target", "G_gen", "Tail", "Depth", "Close")) + " |")
    L += ["", "## Suite without A1's *_pure_gt tables (VR, table labels)", "", json.dumps({k: round(v, 4) for k, v in out["no_gt_suite"].items()}), "",
          "## Marginal score per 1e6 conflicts removed (uniform 1% gain on one table)", "", "| table | W (table) | VR | VR* | VR** |", "|---|---|---|---|---|"]
    for k, v in sorted(out["marginal_per_1e6"].items(), key=lambda kv: kv[1]["W_table"]):
        L.append(f"| {k} | {v['W_table']:.3g} | {v['VR']:.3g} | {v['VR*']:.3g} | {v['VR**']:.3g} |")
    L += ["", "## Tail tie-break on targets (deepened censored cases; lower-bound true d)", "", "```", json.dumps(out["tail_index_check"], indent=1), "```",
          "", "## Do genuine rules kill truly easier censored target cases? (deepened sample)", "", "```", json.dumps(out["kill_bias_on_deepened_targets"], indent=1), "```"]
    open(os.path.join(_HERE, "results.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

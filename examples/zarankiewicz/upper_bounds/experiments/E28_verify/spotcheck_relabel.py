"""E28 spot check: 10 model-relabelled TARGET cases (seed 28, 2 per table from 5 target tables).

For each case:
  (a) an independent fresh pysat cadical195 run at a 20,000-conflict budget, written here against pysat directly
      (not zar_ub.solve / hardness_progress), compared with the stored probe.ps20k {conflicts, decisions, restarts};
  (b) hardness_model.predict() recomputes every feature (its own 20k probe + lookahead + static LP);
  (c) log d_hat recomputed by hand here from those features and hardness_model.json (transform, z-score, coef,
      smearing, floor 20k), compared with the stored probe.d_model;
  (d) the stored label d compared with min(max(r, d_hat), max(r, 2M)), r = conflicts reached.
One solver process, sequential.  Costs reported in conflicts / propagations.
"""
import json, math, os, random, sys, time

UB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, UB)
from pysat.solvers import Solver  # noqa: E402
from zar_ub.known import Instance  # noqa: E402
from zar_ub.encoding import encode_case  # noqa: E402
from zar_ub import hardness_model as hm  # noqa: E402

TABLES = ["case_table_m12_n18_s3_t3_w109.json", "case_table_m10_n23_s3_t3_w113.json",
          "case_table_m11_n23_s3_t3_w124.json", "case_table_m13_n19_s3_t3_w123.json",
          "case_table_m16_n17_s3_t3_w134.json"]
MODEL = json.load(open(os.path.join(UB, "experiments", "E24_difficulty", "hardness_model.json")))


def my_logd(feats):
    s = MODEL["intercept"]
    for f in MODEL["features"]:
        x = float(feats[f])
        if MODEL["transforms"][f] == "log1p":
            x = math.log1p(x)
        s += MODEL["coef"][f] * (x - MODEL["mu"][f]) / MODEL["sd"][f]
    return s


def main():
    rng = random.Random(28)
    picks = []
    for tn in TABLES:
        t = json.load(open(os.path.join(UB, "cache", tn)))
        inst = t["inst"]
        cand = [r for r in t["records"] if (r.get("probe") or {}).get("d_method") == "model"
                and not r.get("baseline_lean_kill")]
        for r in rng.sample(cand, 2):
            picks.append((tn, inst, r))
    out = []
    tot_c = tot_p = 0
    for tn, ii, r in picks:
        inst = Instance(ii["m"], ii["n"], ii["s"], ii["t"], ii["w"]) if isinstance(ii, dict) else Instance(*ii)
        cnf = encode_case(inst, r["rows"], r["cols"])
        with Solver(name="cadical195", bootstrap_with=cnf.clauses) as s:
            s.conf_budget(20000)
            res = s.solve_limited()
            st = s.accum_stats() or {}
        mine = {k: int(st.get(k, 0)) for k in ("conflicts", "decisions", "restarts", "propagations")}
        stored = r["probe"]["ps20k"]
        t0 = time.time()
        o = hm.predict(inst, r["rows"], r["cols"], mean=True, floor=20000.0)
        el = time.time() - t0
        ld = my_logd(o["features"])
        dhat_mine = max(20000.0, math.exp(ld) * MODEL["smear"])
        reached = r["probe"]["conflicts"]
        d_expected = min(max(reached, dhat_mine), max(reached, 2_000_000))
        row = {"table": tn, "rows": r["rows"], "cols": r["cols"], "solve_result": res,
               "ps20k_mine": mine, "ps20k_stored": stored,
               "ps20k_match": all(mine[k] == int(stored[k]) for k in ("conflicts", "decisions", "restarts")),
               "predict_ps20k": {k: o["features"].get("pr:ps20k_" + k) for k in ("conflicts", "decisions", "restarts")},
               "d_hat_predict": o["d_hat"], "d_hat_by_hand": dhat_mine, "d_model_stored": r["probe"]["d_model"],
               "d_stored": r["d"], "d_expected": d_expected,
               "rel_err_model": abs(dhat_mine - r["probe"]["d_model"]) / r["probe"]["d_model"],
               "rel_err_label": abs(d_expected - r["d"]) / r["d"],
               "cost_conflicts": o["cost_conflicts"] + mine["conflicts"],
               "cost_propagations": o["cost_propagations"] + mine["propagations"], "predict_seconds": round(el, 2)}
        tot_c += row["cost_conflicts"]; tot_p += row["cost_propagations"]
        out.append(row)
        print(json.dumps({k: row[k] for k in ("table", "ps20k_match", "d_hat_predict", "d_hat_by_hand", "d_model_stored",
                                              "d_stored", "d_expected", "rel_err_model", "rel_err_label",
                                              "cost_conflicts", "cost_propagations")}), flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump({"cases": out, "total_conflicts": tot_c, "total_propagations": tot_p},
              open(os.path.join(here, "spotcheck_relabel.json"), "w"), indent=1)
    print("total conflicts", tot_c, "propagations", tot_p)
    print("all ps20k match:", all(r["ps20k_match"] for r in out),
          "max rel err model:", max(r["rel_err_model"] for r in out),
          "max rel err label:", max(r["rel_err_label"] for r in out))


if __name__ == "__main__":
    main()

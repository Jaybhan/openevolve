"""E27: censored-label statistics on the library survivors of the scored target / wide tables, before (cache_before/)
and after (cache/) the E27 relabel.  usage: python experiments/E27_integration/label_stats.py [before|after]"""
import json, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from zar_ub.casetable import table_from_json
TABS = ["m9_n23_s3_t3_w104", "m12_n18_s3_t3_w109", "m10_n23_s3_t3_w113", "m11_n23_s3_t3_w124", "m13_n19_s3_t3_w123",
        "m16_n17_s3_t3_w134", "m10_n22_s3_t3_w111", "m10_n20_s3_t3_w103_pure", "m11_n21_s3_t3_w117_pure", "m10_n19_s3_t3_w99_pure_gt"]
which = sys.argv[1] if len(sys.argv) > 1 else "after"
d = os.path.join(HERE, "cache_before") if which == "before" else os.path.join(ROOT, "cache")
out = {}
for k in TABS:
    p = os.path.join(d, f"case_table_{k}.json"); t = table_from_json(json.load(open(p)), p)
    S = t.scored_indices()
    cen = [t.records[i].d for i in S if t.records[i].censored]
    W = sum(t.records[i].d for i in S)
    out[k] = {"survivors": len(S), "censored": len(cen), "W": W, "W_censored_share": (sum(cen) / W) if W else 0,
              "distinct_censored_labels": len(set(round(x) for x in cen)),
              "median": statistics.median(cen) if cen else None, "min": min(cen) if cen else None,
              "max": max(cen) if cen else None, "at_2M": sum(1 for x in cen if x >= 2e6)}
    o = out[k]
    print(f"{k:28s} surv {o['survivors']:5d} cens {o['censored']:5d} W {W:10.3g} cens-share {o['W_censored_share']:.3f} "
          f"distinct {o['distinct_censored_labels']:5d} min/med/max {o['min']} / {o['median']} / {o['max']} at2M {o['at_2M']}")
json.dump(out, open(os.path.join(HERE, f"label_stats_{which}.json"), "w"), indent=1)

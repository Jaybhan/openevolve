"""Print eval_results.json as markdown tables (eval_results.txt); used to write EVALUATION.md."""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "eval_results.json")))
out = []


def f(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def P(*a):
    out.append(" ".join(str(x) for x in a))


def g(model, uni="survivors", ceil="2000000"):
    return R["gain"].get(model, {}).get(uni, {}).get(ceil, {})


P("## HARD regime, leave-one-shape-out (2,429 hard cases; open-at-2M scored by C only)\n")
P("| model | needs | within ρ | pooled ρ | C within | C pooled | log-RMSE | bias | top-10% recall | TARGET within ρ / C within | WIDE within ρ | SURVIVORS within ρ / C within | gain MAE (real masks) | tail MAE (real) | gain MAE (random) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
L = R["models"]["loso"]
for name, m in L.items():
    tg = m.get("target", {})
    wd = m.get("wide", {})
    sv = m.get("survivors", {})
    gg = g(name + "+smear") or g(name)
    P(f"| {name} | {','.join(m.get('need', [])) or '-'} | {f(m['within_rho'])} | {f(m['pooled_rho'])} | {f(m['C_within'])} | "
      f"{f(m['C_pooled'])} | {f(m['log_rmse'])} | {f(m.get('bias'), 2)} | {f(m.get('top10_recall'), 2)} | "
      f"{f(tg.get('within_rho'))} / {f(tg.get('C_within'))} | {f(wd.get('within_rho'))} | "
      f"{f(sv.get('within_rho'))} / {f(sv.get('C_within'))} | {f(gg.get('gain_mae_real'), 4)} | {f(gg.get('tail_mae_real'), 4)} | "
      f"{f(gg.get('gain_mae_random'), 4)} |")

for sch in ("sq2wide", "wide2sq"):
    P(f"\n## HARD regime, {sch}\n")
    P("| model | within ρ | pooled ρ | C within | C pooled | log-RMSE | bias | TARGET within ρ / C within / log-RMSE |")
    P("|---|---|---|---|---|---|---|---|")
    for name, m in R["models"][sch].items():
        tg = m.get("target", {})
        P(f"| {name} | {f(m['within_rho'])} | {f(m['pooled_rho'])} | {f(m['C_within'])} | {f(m['C_pooled'])} | "
          f"{f(m['log_rmse'])} | {f(m.get('bias'), 2)} | {f(tg.get('within_rho'))} / {f(tg.get('C_within'))} / {f(tg.get('log_rmse'))} |")

P("\n## per-cell within-cell Spearman (loso), selected models\n")
sel = [k for k in ("current_label", "fhat_refit", "A2_FL_only", "samp_unit_calib", "A4_free20k", "A4_free20k+samp",
                   "greedy[free(ST+LA+PS)]", "greedy[free+SA]", "greedy[free+BIN]", "greedy[all]", "gbm[all]") if k in L]
cells = sorted({c for k in sel for c in L[k]["per_cell_rho"]})
P("| cell | " + " | ".join(sel) + " |")
P("|---|" + "---|" * len(sel))
for c in cells:
    P(f"| {c} | " + " | ".join(f(L[k]["per_cell_rho"].get(c)) for k in sel) + " |")
P("\n## per-cell Harrell C (loso)\n")
cells = sorted({c for k in sel for c in L[k]["per_cell_C"]})
P("| cell | " + " | ".join(sel) + " |")
P("|---|" + "---|" * len(sel))
for c in cells:
    P(f"| {c} | " + " | ".join(f(L[k]["per_cell_C"].get(c)) for k in sel) + " |")

P("\n## selections (greedy / single, loso folds)\n")
for name, m in L.items():
    if "selected" in m:
        P(f"- {name}: " + ", ".join(f"{k}×{v}" for k, v in list(m["selected"].items())[:10]))

P("\n## gain error by mask class (survivors universe; ceiling 2M; +smear where available)\n")
classes = ["argD", "DGH", "farkas", "rule", "rand5", "rand20", "rand50"]
P("| model | " + " | ".join(f"{c} gain / tail (n)" for c in classes) + " |")
P("|---|" + "---|" * len(classes))
for name in R["gain"]:
    gg = g(name)
    if not gg:
        continue
    cells_ = []
    for c in classes:
        a = gg["gain_mae"].get(c)
        b = gg["tail_mae"].get(c)
        cells_.append(f"{f(a[0], 4)} / {f(b[0], 4)} ({a[1]})" if a else "–")
    P(f"| {name} | " + " | ".join(cells_) + " |")

P("\n## gain error, ALL-cases universe (library ignored; argD becomes a real mask)\n")
P("| model | " + " | ".join(f"{c} gain / tail (n)" for c in classes) + " |")
P("|---|" + "---|" * len(classes))
for name in R["gain"]:
    gg = g(name, "all")
    if not gg:
        continue
    cells_ = []
    for c in classes:
        a = gg["gain_mae"].get(c)
        b = gg["tail_mae"].get(c)
        cells_.append(f"{f(a[0], 4)} / {f(b[0], 4)} ({a[1]})" if a else "–")
    P(f"| {name} | " + " | ".join(cells_) + " |")

P("\n## ceiling sensitivity (survivors; real-mask gain MAE / tail MAE)\n")
P("| model | ceil 400k | ceil 2M | none |")
P("|---|---|---|---|")
for name in R["gain"]:
    row = []
    for c in ("400000", "2000000", "None"):
        gg = g(name, "survivors", c)
        row.append(f"{f(gg.get('gain_mae_real'), 4)} / {f(gg.get('tail_mae_real'), 4)}" if gg else "–")
    P(f"| {name} | " + " | ".join(row) + " |")

P("\n## DGH / Farkas detail per table (survivors, ceiling 2M)\n")
show = [k for k in ("oracle", "current_label", "A4_free20k+smear", "A4_free20k+samp+smear", "greedy[free(ST+LA+PS)]+smear",
                    "greedy[free+SA]+smear", "greedy[all]+smear", "samp_unit_calib+smear") if k in R["gain"]]
for uni in ("survivors", "all"):
    base = g("oracle", uni)
    for cell, tt in base.get("tables", {}).items():
        for mname, v in tt.items():
            if not (mname.startswith("DGH") or mname.startswith("farkas_union") or mname == "argD"):
                continue
            row = []
            for k in show:
                e = g(k, uni).get("tables", {}).get(cell, {}).get(mname)
                row.append(f"{f(e['g_est'], 3)}/{f(e['tail_est'], 3)}" if e else "–")
            P(f"- [{uni}] {cell} {mname} kills {v['kills']}: " + "; ".join(f"{k}={x}" for k, x in zip(show, row)))

P("\n## MID regime (2k < d <= 20k), loso, 2k-tier features\n")
P("| model | within ρ | pooled ρ | log-RMSE | selected |")
P("|---|---|---|---|---|")
for name, m in R["mid"].items():
    P(f"| {name} | {f(m['within_rho'])} | {f(m['pooled_rho'])} | {f(m['log_rmse'])} | {', '.join(list((m.get('selected') or {}).keys())[:5])} |")

P("\n## cost per case\n")
P("| estimator | regime | n | conflicts mean | conflicts max | propagations mean | seconds mean | p90 | max |")
P("|---|---|---|---|---|---|---|---|---|")
for e, d in R["costs"].items():
    for reg in ("hard", "mid"):
        c = d.get(reg)
        if c:
            P(f"| {e} | {reg} | {c['n']} | {c['conflicts_mean']:.0f} | {c['conflicts_max']:.0f} | {c['props_mean']:.3g} | "
              f"{c['seconds_mean']:.2f} | {c['seconds_p90']:.2f} | {c['seconds_max']:.2f} |")
if "relabel" in R:
    P("\n## projected relabel wall time\n")
    P("```\n" + json.dumps(R["relabel"], indent=1) + "\n```")

txt = "\n".join(out)
open(os.path.join(HERE, "eval_results.txt"), "w").write(txt + "\n")
if "-q" not in sys.argv:
    print(txt)

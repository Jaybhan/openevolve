"""E28: compare the two cold-gate evaluator runs per candidate (determinism) and tabulate scores."""
import glob, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
TIMING = {"eval_seconds", "gate_seconds", "gate_cache_hit"}
EXPLOITS = {"cert_close_10_22", "cert_one_train", "e1_dgh_s2", "e2_easy_certs", "e3_relist", "inst_dgh_10_23",
            "instance_specific"}
GENUINE = {"lean_dgh4", "dgh_lib", "schema_recipe", "schema_recipe_frozen", "schema_pool", "recipe_dgh", "f_weak",
           "cert_target_12_18"}


def main():
    runs = {}
    for p in sorted(glob.glob(os.path.join(HERE, "evals", "*_run[12].json"))):
        b = os.path.basename(p)[:-len("_run1.json")]
        r = b and os.path.basename(p)[-6]
        runs.setdefault(b, {})[r] = json.load(open(p))
    lines = ["| candidate | class | score run1 | score run2 | identical (excl. timings) | differing keys | ladder | lean_ok | sound_battery | proven_gain | target_gain | depth | close | gate s (run1) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for b, rr in sorted(runs.items()):
        m1 = rr.get("1", {}).get("metrics", {}); m2 = rr.get("2", {}).get("metrics", {})
        diff = sorted(k for k in set(m1) | set(m2) if k not in TIMING and m1.get(k) != m2.get(k))
        same = bool(m1) and bool(m2) and not diff and rr["1"].get("score_breakdown") == rr["2"].get("score_breakdown")
        cls = "exploit" if b in EXPLOITS else "genuine" if b in GENUINE else "reference"
        summary[b] = {"class": cls, "s1": m1.get("combined_score"), "s2": m2.get("combined_score"), "identical": same,
                      "diff": diff}
        f = lambda k: ("%.4f" % m1[k]) if isinstance(m1.get(k), (int, float)) else str(m1.get(k))
        lines.append(f"| {b} | {cls} | {f('combined_score')} | " + (("%.4f" % m2["combined_score"]) if m2 else "-") +
                     f" | {same} | {','.join(diff) or '-'} | {f('lean_ladder')} | {f('lean_ok')} | {f('sound_battery')} | "
                     f"{f('proven_gain')} | {f('target_gain')} | {f('depth_term')} | {f('closure_bonus')} | {f('gate_seconds')} |")
    ex = {b: v["s1"] for b, v in summary.items() if v["class"] == "exploit" and v["s1"] is not None}
    ge = {b: v["s1"] for b, v in summary.items() if v["class"] == "genuine" and v["s1"] is not None}
    lines.append("")
    if ex and ge:
        lines.append(f"max exploit = {max(ex.values()):.4f} ({max(ex, key=ex.get)}); "
                     f"min genuine = {min(ge.values()):.4f} ({min(ge, key=ge.get)}); "
                     f"recipe = {ge.get('schema_recipe')}, lean_dgh4 = {ge.get('lean_dgh4')}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(HERE, "evals_table.md"), "w").write(txt + "\n")
    json.dump(summary, open(os.path.join(HERE, "evals_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""E27: re-score the E25 benchmark (R1's Lean-gated masks, R3's attack shapes) with the FINAL code:
zar_ub.reward.score (reward v3) on suite.load_suite() (suite v3), with the tables as they are in cache/
(E24 ground truth written back, censored labels from the E24 hardness model).

Offline: no Lean (R1's cached Lean masks, experiments/E25_reward/masks/*.json), no SAT, no LP.

usage (from examples/zarankiewicz/upper_bounds, ZAR_UB_NO_LLM=1):
    python experiments/E27_integration/rescore_benchmark.py            # writes rescore.{json,md}
    python experiments/E27_integration/rescore_benchmark.py --check    # also: reproduce R3's VR** exactly

Columns
  v2/S0 old      reward v2 on the old suite (square TRAIN, 2 targets, 3 GEN) with the PRE-E27 labels
                 (experiments/E27_integration/cache_before): must reproduce R1's "V0/S0 current" column
  v2/S0 new      the same reward on the new labels (ground truth + hardness model)
  v2/S1 new      reward v2 on suite v3
  v3/S1 old      reward v3 on suite v3 with the pre-E27 labels (lower bounds = the old tables' conflicts)
  v3/S1 new      reward v3 on suite v3 with the new labels  <-- the live reward
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, ROOT)
os.environ.setdefault("ZAR_UB_NO_LLM", "1")

import suite as suite_mod  # noqa: E402
from zar_ub import reward  # noqa: E402
from zar_ub.casetable import table_from_json  # noqa: E402

E25 = os.path.join(ROOT, "experiments", "E25_reward")
MASKS = os.path.join(E25, "masks")
BEFORE = os.path.join(_HERE, "cache_before")
GT = os.path.join(ROOT, "experiments", "E24_difficulty", "ground_truth.jsonl")

S0_TRAIN = [(9, 9), (9, 10), (10, 10), (10, 11), (11, 11), (11, 12), (12, 12)]
S0_TARGETS = "13,17,117;13,18,122;15,17,133;9,23,104;12,18,109"
S0_GEN = [(7, 7, 2, 2, 22), (8, 8, 2, 2, 25), (9, 9, 4, 4, 62)]

GENUINE = ["schema_recipe_frozen", "dgh_lib", "recipe_dgh"]
REAL_EXPLOITS = ["e1_dgh_s2", "e2_easy_certs", "e3_relist", "cert_one_train", "inst_dgh_10_23", "cert_close_10_22"]


def stem(tab) -> str:
    return os.path.basename(tab.path)[len("case_table_"):-len(".json")]


def load_suites(old: bool):
    """{'S0': suite dict, 'S1': suite dict} with tables from cache/ (new) or cache_before/ (old)."""
    st = {"graduated": [], "entered": [], "history": []}
    s1 = suite_mod.load_suite(state=st)
    s1["battery"] = []  # the witnessed-case checks are not what is compared here
    save_train, save_gen = suite_mod.TRAIN_CELLS, suite_mod.GEN_CELLS
    save_wide = suite_mod.WIDE_TRAIN_CELLS
    try:
        suite_mod.WIDE_TRAIN_CELLS = []
        suite_mod.GEN_CELLS = S0_GEN
        s0 = suite_mod.load_suite(state=st, targets=suite_mod.parse_targets(S0_TARGETS))
    finally:
        suite_mod.TRAIN_CELLS, suite_mod.GEN_CELLS, suite_mod.WIDE_TRAIN_CELLS = save_train, save_gen, save_wide
    s0["battery"] = []
    if old:
        for s in (s0, s1):
            for kind in ("train", "target", "gen"):
                new = []
                for inst, tab in s[kind]:
                    p = os.path.join(BEFORE, os.path.basename(tab.path))
                    t = table_from_json(json.load(open(p)), p)
                    t.kind, t.omega = tab.kind, tab.omega
                    new.append((inst, t))
                s[kind] = new
    return {"S0": s0, "S1": s1}


def r1_masks(name: str, suite) -> dict:
    d = json.load(open(os.path.join(MASKS, f"{name}.json")))["tables"]
    out = {}
    for kind in ("train", "target", "gen"):
        for inst, tab in suite[kind]:
            t = d.get(stem(tab))
            m = t.get("lean") if t else None
            if m is not None and len(m) != len(tab.records):
                raise RuntimeError(f"{name}: mask length mismatch on {stem(tab)}")
            out[inst.tag] = [bool(x) for x in m] if m else [False] * len(tab.records)
    return out


def stem_masks(stem_mask: dict, suite) -> dict:
    out = {}
    for kind in ("train", "target", "gen"):
        for inst, tab in suite[kind]:
            m = stem_mask.get(stem(tab))
            out[inst.tag] = [bool(x) for x in m] if m else [False] * len(tab.records)
    return out


def score(suite, masks, version: str) -> dict:
    old = os.environ.get("ZAR_UB_REWARD")
    os.environ["ZAR_UB_REWARD"] = version
    try:
        m, _ = reward.score(suite, py_masks=masks, lean_masks=masks, stage=3, ladder=5, lean_partial=1.0,
                            lean_ok=1.0)
    finally:
        if old is None:
            os.environ.pop("ZAR_UB_REWARD", None)
        else:
            os.environ["ZAR_UB_REWARD"] = old
    return m


def attack_masks():
    """R3's attack shapes (experiments/E25_reward/attacks/run_attacks.build) on the CURRENT tables."""
    sys.path.insert(0, os.path.join(E25, "attacks"))
    sys.path.insert(0, E25)
    import common as C  # noqa: E402
    import run_attacks as RA  # noqa: E402

    env = C.load_env(include_gt=True)
    A = RA.build(env)
    return {k: (kind, m) for k, (kind, m, note) in A.items()}


def work_removed_lb(suite, masks) -> float:
    tot = 0.0
    for kind in ("train", "target"):
        for inst, tab in suite[kind]:
            S = reward.survivor_indices(tab, None, use_sample=False)
            m = masks.get(inst.tag) or []
            cap = int(tab.conf_cap or 0)
            tot += sum(reward.rec_lower_bound(tab.records[i], cap) for i in S if i < len(m) and m[i])
    return tot


def c4_live(suite, trials: int = 60, seed: int = 27) -> dict:
    """C4 on the live reward: adding kills never lowers the score; a uniform gain x on every scored cell
    beats the same gain on any single cell (x in {0.2, 0.5})."""
    rng = random.Random(seed)
    tabs = [(inst, tab) for kind in ("train", "target", "gen") for inst, tab in suite[kind]]
    surv = {inst.tag: reward.survivor_indices(tab, None, use_sample=False) for inst, tab in tabs}
    base = {inst.tag: [False] * len(tab.records) for inst, tab in tabs}
    mono_fail = 0
    for _ in range(trials):
        m = {k: list(v) for k, v in base.items()}
        for k, S in surv.items():
            p = rng.random() ** 2
            for i in S:
                if rng.random() < p:
                    m[k][i] = True
        s_a = score(suite, m, "v3")["combined_score"]
        k = rng.choice([k for k in surv if surv[k]])
        m2 = {kk: list(v) for kk, v in m.items()}
        for i in surv[k]:
            if rng.random() < 0.5:
                m2[k][i] = True
        if score(suite, m2, "v3")["combined_score"] < s_a - 1e-12:
            mono_fail += 1
    uni_fail = []
    for x in (0.2, 0.5):
        pick = {k: set(rng.sample(S, max(1, round(x * len(S))))) if S else set() for k, S in surv.items()}
        mu = {k: [i in pick[k] for i in range(len(base[k]))] for k in base}
        su = score(suite, mu, "v3")["combined_score"]
        for k in surv:
            if not surv[k]:
                continue
            m1 = {kk: (mu[kk] if kk == k else base[kk]) for kk in base}
            s1 = score(suite, m1, "v3")["combined_score"]
            if not su > s1 + 1e-12:
                uni_fail.append((x, k, round(su, 4), round(s1, 4)))
    return {"mono_fail": mono_fail, "trials": trials, "uniform_fail": uni_fail}


def check_r3(suites_old) -> dict:
    """Reproduce R3's 'VR** = F1b+F3' column (table labels + ground-truth lower bounds for importance)."""
    gt = {}
    for line in open(GT):
        r = json.loads(line)
        gt[(r["cell"], tuple(r["rows"]), tuple(r["cols"]))] = r
    s1 = suites_old["S1"]
    for kind in ("train", "target", "gen"):
        for inst, tab in s1[kind]:
            cell = stem(tab)
            cell = cell[:-3] if cell.endswith("_gt") else cell
            for r in tab.records:
                g = gt.get((cell, tuple(r.rows), tuple(r.cols)))
                if g is not None:
                    r._lb = float(max(1, g["d"]))
    orig = reward.rec_lower_bound
    reward.rec_lower_bound = lambda rec, cap=0: getattr(rec, "_lb", None) or orig(rec, cap)
    try:
        out = {p: round(score(s1, r1_masks(p, s1), "v3")["combined_score"], 4)
               for p in ("schema_recipe_frozen", "dgh_lib", "recipe_dgh", "f_weak", "cert_close_10_22")}
    finally:
        reward.rec_lower_bound = orig
    out["R3 expected"] = {"schema_recipe_frozen": 0.2619, "dgh_lib": 0.3836, "recipe_dgh": 0.4139, "f_weak": 0.2173,
                          "cert_close_10_22": 0.2105}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    new, old = load_suites(False), load_suites(True)
    programs = sorted(os.path.basename(p)[:-5] for p in os.listdir(MASKS) if p.endswith(".json"))
    rows = {}
    for p in programs:
        r = {}
        r["v2/S0 old"] = score(old["S0"], r1_masks(p, old["S0"]), "v2")
        r["v2/S0 new"] = score(new["S0"], r1_masks(p, new["S0"]), "v2")
        r["v2/S1 new"] = score(new["S1"], r1_masks(p, new["S1"]), "v2")
        r["v3/S1 old"] = score(old["S1"], r1_masks(p, old["S1"]), "v3")
        r["v3/S1 new"] = score(new["S1"], r1_masks(p, new["S1"]), "v3")
        r["work_lb_removed"] = work_removed_lb(new["S1"], r1_masks(p, new["S1"]))
        rows[p] = {"kind": "real (R1 Lean-gated)", **r}
    for name, (kind, sm) in attack_masks().items():
        if name.startswith("ref:") or name.startswith("r1:"):
            continue
        r = {"v2/S0 new": score(new["S0"], stem_masks(sm, new["S0"]), "v2"),
             "v3/S1 new": score(new["S1"], stem_masks(sm, new["S1"]), "v3"),
             "work_lb_removed": work_removed_lb(new["S1"], stem_masks(sm, new["S1"]))}
        rows[f"R3 {name}"] = {"kind": f"{kind} (R3 attack shape)", **r}
    cols = ["v2/S0 old", "v2/S0 new", "v2/S1 new", "v3/S1 old", "v3/S1 new"]
    keys = ["combined_score", "proven_gain", "target_gain", "tail_gain", "depth_term", "closure_bonus"]
    js = {p: {"kind": r["kind"], "work_lb_removed": r["work_lb_removed"],
              **{c: {k: r[c][k] for k in keys} for c in cols if c in r}} for p, r in rows.items()}
    # criteria (R1's C1-C5, same definitions) on the live column
    col = {p: js[p]["v3/S1 new"]["combined_score"] for p in js}
    rec, dgh, rd = col["schema_recipe_frozen"], col["dgh_lib"], col["recipe_dgh"]
    v0rec = js["schema_recipe_frozen"]["v2/S0 old"]["combined_score"]
    genuine_min = min(rec, dgh)
    real_bad = {p: col[p] for p in REAL_EXPLOITS if p in col and col[p] >= genuine_min}
    orac_bad = {p: col[p] for p in col if p.startswith("R3 ") and js[p]["kind"].startswith("oracle")
                and col[p] >= genuine_min}
    real_attack_bad = {p: col[p] for p in col if p.startswith("R3 ") and js[p]["kind"].startswith("real")
                       and "A4" not in p and col[p] >= genuine_min}
    fweak = col["f_weak"]
    tiny = {p: col[p] for p in col if p.startswith("R3 A1") or p.startswith("R3 A2")}
    crit = {
        "C1 initial == 0.20": abs(col["initial"] - 0.20) < 1e-12,
        "C2 dgh uplift >= 0.02 and >= 0.5 x recipe uplift": (dgh - 0.2) >= 0.02 and (dgh - 0.2) >= 0.5 * (rec - 0.2),
        "C2 ratio dgh/recipe uplift": (dgh - 0.2) / (rec - 0.2) if rec > 0.2 else float("inf"),
        "C3 real exploits below min(recipe, dgh)": not real_bad, "C3 real exploit violations": real_bad,
        "C3 R3 realizable attacks (A5, B4, C1) below min(recipe, dgh)": not real_attack_bad,
        "C3 R3 realizable violations": real_attack_bad,
        "C3 oracle shapes below min(recipe, dgh)": not orac_bad, "C3 oracle violations": orac_bad,
        "tiny-table clears below f_weak": {p: (v, v < fweak) for p, v in tiny.items()},
        "C5 recipe uplift retained vs v2/S0 old": (rec - 0.2) / (v0rec - 0.2),
        "C5 recipe+dgh >= max(recipe, dgh)": rd >= max(rec, dgh) - 1e-12,
        "C4 live property test": c4_live(new["S1"]),
    }
    out = {"rows": js, "criteria": crit}
    if a.check:
        out["r3_reproduction"] = check_r3(old)
    with open(os.path.join(_HERE, "rescore.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    L = ["# E27 re-scored benchmark (final code, offline)", "",
         "`experiments/E27_integration/rescore_benchmark.py`. Verified-branch combined_score; R1 masks are Lean-gated "
         "programs, R3 rows are attack shapes (oracle = not a program). 'old' = pre-E27 labels, 'new' = ground truth + "
         "hardness model. work = lower-bound conflicts removed over TRAIN ∪ TARGET of suite v3.", "",
         "| program | kind | " + " | ".join(cols) + " | v3 G_train / G_target / Tail / Depth / Close | work removed (lb) |",
         "|---|---|" + "---|" * len(cols) + "---|---|"]
    for p, r in js.items():
        cells = [f"{r[c]['combined_score']:.4f}" if c in r else "" for c in cols]
        v = r["v3/S1 new"]
        L.append(f"| {p} | {r['kind']} | " + " | ".join(cells) +
                 f" | {v['proven_gain']:.3f} / {v['target_gain']:.3f} / {v['tail_gain']:.3f} / {v['depth_term']:.3f} / "
                 f"{v['closure_bonus']:.3f} | {r['work_lb_removed']:.3g} |")
    L += ["", "## Criteria (live column v3/S1 new)", "", "```", json.dumps(crit, indent=1, default=str), "```"]
    if a.check:
        L += ["", "## Implementation check: R3's VR** reproduced by reward.score", "", "```",
              json.dumps(out["r3_reproduction"], indent=1), "```"]
    with open(os.path.join(_HERE, "rescore.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

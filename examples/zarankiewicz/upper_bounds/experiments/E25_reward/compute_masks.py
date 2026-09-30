"""E25 step 2: Lean kill masks, schema masks and Python masks of every benchmark program on every
universe table, computed ONCE per (program sha, universe) and cached in masks/<name>.json, so the
reward variants can be evaluated offline without re-running Lean.

One Lean process per program (zar_ub.lean_gate.run_gate_multi over all universe tables, exactly the
call evaluator._gate_data makes: candidate source + rendered SCHEMA_DATA terms per instance), run
sequentially (one Lean process at a time).  Gate results are cached in masks/gate_cache (HMAC'd by
the gate), never in cache/gate.  Case tables are read-only.

usage: python experiments/E25_reward/compute_masks.py [name ...]      (default: every candidates/*.py
       except the dynamic schema_recipe, which is scored through schema_recipe_frozen)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import universe as U  # noqa: E402
from zar_ub import lemmas, reward  # noqa: E402
from zar_ub.known import Instance  # noqa: E402
from zar_ub.lean_gate import run_gate_multi, static_scan  # noqa: E402

CAND = os.path.join(_HERE, "candidates")
MASKS = os.path.join(_HERE, "masks")
SKIP = {"schema_recipe"}  # dynamic (runs search_certificates at import); frozen twin is scored
GATE_TIMEOUT = 1500.0


def ordered_universe(include_gt: bool = True):
    uni = U.universe(include_gt=include_gt)
    order = {"train": 0, "band": 1, "wide": 2, "wide_gt": 3, "target": 4, "target_x": 5, "gen": 6}
    keys = sorted(uni, key=lambda k: (min(order.get(r, 9) for r in uni[k]["roles"]), k))
    return keys, uni


def universe_signature(keys, tabs) -> str:
    h = hashlib.sha1()
    for k in keys:
        h.update(f"{k}:{tabs[k].table_hash}:{len(tabs[k].records)};".encode())
    return h.hexdigest()[:16]


def load_program(path):
    spec = importlib.util.spec_from_file_location("e25_prog_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute(name: str, keys, uni, tabs, force: bool = False) -> dict:
    path = os.path.join(CAND, f"{name}.py")
    sha = hashlib.sha1(open(path, "rb").read()).hexdigest()
    sig = universe_signature(keys, tabs)
    out_path = os.path.join(MASKS, f"{name}.json")
    have = {}
    if os.path.exists(out_path) and not force:
        have = json.load(open(out_path))
        if have.get("sha") == sha and have.get("universe_sig") == sig:
            return have
        # incremental: keep tables already computed for the same program sha
        if have.get("sha") != sha:
            have = {}
    t0 = time.time()
    mod = load_program(path)
    src = str(getattr(mod, "LEAN_SOURCE", ""))
    sd = getattr(mod, "SCHEMA_DATA", {}) or {}
    kill = getattr(mod, "kill")
    old_tables = have.get("tables", {}) if have else {}
    todo = [k for k in keys if k not in old_tables]
    res = {"name": name, "sha": sha, "universe_sig": sig, "tables": dict(old_tables), "forbidden": list(static_scan(src)),
           "schema_errors": [], "gate": have.get("gate", {}) if have else {}}
    # python masks (the candidate's own kill + the schema mirror, as evaluator stage 1 builds K^P)
    for k in todo:
        inst = Instance(uni[k]["m"], uni[k]["n"], uni[k]["s"], uni[k]["t"], uni[k]["w"])
        ok, errs = lemmas.validate(sd, inst)
        if not ok:
            res["schema_errors"].append(f"[{k}] " + "; ".join(errs))
        pm = []
        for r in tabs[k].records:
            v = bool(kill(inst.m, inst.n, inst.s, inst.t, inst.w, tuple(r.rows), tuple(r.cols)))
            if not v and ok and sd.get("farkas"):
                v = bool(lemmas.mirror_kill(sd, inst, r.rows, r.cols))
            pm.append(v)
        res["tables"][k] = {"python": pm}
    # one Lean process over the tables still to do
    gk = [k for k in todo if tabs[k].records]
    if gk and not res["forbidden"] and not res["schema_errors"]:
        insts = [Instance(uni[k]["m"], uni[k]["n"], uni[k]["s"], uni[k]["t"], uni[k]["w"]) for k in gk]
        inputs = [(inst, [(r.rows, r.cols) for r in tabs[k].records]) for inst, k in zip(insts, gk)]
        terms = None
        if sd.get("farkas"):
            terms = [list(lemmas.render_terms(sd, inst, "ZarPrune.Cand.target" + ("" if i == 0 else str(i))))
                     for i, inst in enumerate(insts)]
        tg = time.time()
        results = run_gate_multi(inputs, src, schema_terms=terms, timeout=GATE_TIMEOUT, tag=f"e25_{name}",
                                 sketch=True, use_cache=True, cache_dir=os.path.join(MASKS, "gate_cache"))
        gsec = time.time() - tg
        axioms = set()
        for k, g in zip(gk, results):
            lad = reward.ladder_of(g)
            km = getattr(g, "kill_mask", None)
            sm = getattr(g, "schema_mask", None)
            n = len(tabs[k].records)
            res["tables"][k].update({
                "ladder": lad,
                "lean": [bool(x) for x in km] if (lad == 5 and km is not None and len(km) == n) else None,
                "schema": [bool(x) for x in sm] if (sm is not None and len(sm) == n) else None,
                "errors": list(getattr(g, "errors", []) or [])[:3],
            })
            axioms.update(getattr(g, "axioms", None) or [])
        res["gate"] = {"seconds": gsec, "cache_hit": bool(getattr(results[0], "cache_hit", False)) if results else False,
                       "axioms": sorted(axioms), "n_tables": len(gk), "n_cases": sum(len(c) for _, c in inputs)}
    for k in todo:
        if not tabs[k].records:
            res["tables"][k].update({"ladder": 5, "lean": [], "schema": [], "errors": []})
    res["seconds"] = time.time() - t0
    os.makedirs(MASKS, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f)
    return res


def main():
    names = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    keys, uni = ordered_universe()
    tabs = {k: U.load(k, uni[k]["path"]) for k in keys}
    if not names:
        names = sorted(f[:-3] for f in os.listdir(CAND) if f.endswith(".py") and f[:-3] not in SKIP)
    for name in names:
        r = compute(name, keys, uni, tabs, force=force)
        lads = [v.get("ladder") for v in r["tables"].values()]
        n5 = sum(1 for x in lads if x == 5)
        wit = []
        for k, v in r["tables"].items():
            lm = v.get("lean")
            if lm:
                wit += [k for i, rec in enumerate(tabs[k].records) if reward.is_witnessed(rec) and lm[i]]
        print(f"{name:22s} L5 on {n5}/{len(lads)} tables  gate {r['gate'].get('seconds', 0):.1f}s "
              f"cache_hit={r['gate'].get('cache_hit')} axioms={r['gate'].get('axioms')} forbidden={r['forbidden']} "
              f"schema_errors={len(r['schema_errors'])} witnessed_kills={len(wit)}", flush=True)


if __name__ == "__main__":
    main()

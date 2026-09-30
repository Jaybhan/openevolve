#!/usr/bin/env python3
"""E26: does a reward variant let a SPECIALIST survive and spread in a real OpenEvolve population?

One configuration = (reward variant, MAP-Elites features).  The LLM is the E26 GenomeMutator
(mutator.py, blind to scores), the evaluator is evaluator_variant.py (live evaluator + variant
re-score).  Same seed (config random_seed 42, mutator seed 26), one worker, 50 iterations.

    python experiments/E26_dynamics/run_dynamics.py --name vr_ladder --variant "VR recommended" \
        [--config experiments/E26_dynamics/config_dyn.yaml] [--iterations 50]
    python experiments/E26_dynamics/run_dynamics.py --analyze-only --name vr_ladder

Writes runs/<name>/ (OpenEvolve output + eval_log.jsonl + mutator_log.tsv) and runs/<name>/summary.json.
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import math
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(UB, "..", "..", ".."))
for p in (ROOT, UB, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ZAR_UB_NO_LLM", "1")

import genome as G  # noqa: E402

GENERAL, SPECIALISTS = {"R"}, {"D", "C", "T"}


def _configure(cfg, iterations: int, db_seed: int = 42):
    from openevolve.config import LLMModelConfig
    import mutator
    model = LLMModelConfig(name="e26-genome", api_base="http://synthetic.invalid/v1", api_key="x", weight=1.0,
                           init_client=mutator.make_mutator)
    cfg.llm.models = [model]
    cfg.llm.evaluator_models = []
    cfg.llm.api_base = model.api_base
    cfg.llm.api_key = "x"
    cfg.max_iterations = iterations
    cfg.random_seed = db_seed
    cfg.database.random_seed = db_seed
    return cfg


async def _run(a, out: str) -> float:
    from openevolve.config import load_config
    from openevolve.controller import OpenEvolve
    cfg = _configure(load_config(a.config), a.iterations, a.db_seed)
    ctl = OpenEvolve(initial_program_path=os.path.join(UB, "initial_program.py"),
                     evaluation_file=os.path.join(HERE, "evaluator_variant.py"), config=cfg, output_dir=out)
    t0 = time.time()
    await ctl.run(iterations=a.iterations)
    return time.time() - t0


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------
def _entropy(counts) -> float:
    n = sum(counts)
    return -sum(c / n * math.log2(c / n) for c in counts if c) if n else 0.0


def _load_ckpt(d: str):
    progs = {}
    for f in glob.glob(os.path.join(d, "programs", "*.json")):
        with open(f) as fh:
            p = json.load(fh)
        g, _ = G.parse(p.get("code", ""))
        p["_g"] = g
        progs[p["id"]] = p
    with open(os.path.join(d, "metadata.json")) as fh:
        meta = json.load(fh)
    return progs, meta


def _score(p) -> float:
    return float((p.get("metrics") or {}).get("combined_score", 0.0) or 0.0)


def snapshot(d: str) -> dict:
    progs, meta = _load_ckpt(d)
    ps = list(progs.values())
    n = len(ps) or 1
    gcount = {}
    for p in ps:
        k = G.gstr(p["_g"])
        gcount[k] = gcount.get(k, 0) + 1
    cells = []  # (island, key, genome, score)
    for isl, fm in enumerate(meta.get("island_feature_maps") or []):
        for key, pid in fm.items():
            if pid in progs:
                cells.append((isl, key, G.gstr(progs[pid]["_g"]), _score(progs[pid])))
    arch = [progs[i] for i in meta.get("archive", []) if i in progs]
    best = progs.get(meta.get("best_program_id"))
    has = lambda p, a: a in p["_g"]  # noqa: E731
    sound_ps = [p for p in ps if "B" not in p["_g"] and "U" not in p["_g"]]
    unsound = [p for p in ps if float((p.get("metrics") or {}).get("sound_battery", 1) or 0) == 0.0]
    viol = []
    for p in ps:
        m = p.get("metrics") or {}
        sc = _score(p)
        lad = int(round(float(m.get("lean_ladder", 0) or 0)))
        if float(m.get("s1_witnessed_kills", 0) or 0) > 0:
            viol.append((p["id"], "Lean kill of a witnessed S1 case", sc))
        if float(m.get("sound_battery", 1) or 0) == 0.0 and sc > 0 and (lad != 5 or float(m.get("empirical_gain", 0) or 0) > 0):
            viol.append((p["id"], "unsound mirror scored without L5 / with empirical credit", sc))
        if lad < 5 and sc > 0.19 + 1e-9:
            viol.append((p["id"], f"unverified L{lad} scored > 0.19", sc))
        if "B" in p["_g"] and sc > 0.19 + 1e-9:
            viol.append((p["id"], "broken-proof genome scored > 0.19", sc))
    # "unsound program" in the strict sense of the task: any program carrying the unsound rule U
    u_scores = [_score(p) for p in ps if "U" in p["_g"]]
    return {
        "iteration": meta.get("last_iteration"),
        "n_programs": len(ps),
        "best_score": _score(best) if best else max((_score(p) for p in ps), default=0.0),
        "best_genome": G.gstr(best["_g"]) if best else None,
        "best_has_general": bool(best and GENERAL & best["_g"]),
        "best_has_specialist": bool(best and SPECIALISTS & best["_g"]),
        "best_has_D": bool(best and "D" in best["_g"]),
        "genome_counts": dict(sorted(gcount.items(), key=lambda kv: -kv[1])),
        "distinct_genomes": len(gcount),
        "genome_entropy_bits": round(_entropy(gcount.values()), 3),
        "frac_with_D": round(sum(has(p, "D") for p in ps) / n, 3),
        "frac_with_R": round(sum(has(p, "R") for p in ps) / n, 3),
        "frac_with_R_and_D": round(sum(has(p, "R") and has(p, "D") for p in ps) / n, 3),
        "frac_with_C": round(sum(has(p, "C") for p in ps) / n, 3),
        "frac_sound": round(len(sound_ps) / n, 3),
        "map_cells_occupied": len(cells),
        "map_cells_distinct_keys": len({c[1] for c in cells}),
        "map_cells_with_D": sum(1 for c in cells if "D" in G.parse_gstr(c[2])),
        "map_cells_with_R_and_D": sum(1 for c in cells if {"R", "D"} <= G.parse_gstr(c[2])),
        "map_cells_with_C": sum(1 for c in cells if "C" in G.parse_gstr(c[2])),
        "map_cells": sorted(cells),
        "archive_n": len(arch),
        "archive_with_D": sum(1 for p in arch if "D" in p["_g"]),
        "archive_genomes": sorted(G.gstr(p["_g"]) for p in arch),
        "n_mirror_unsound": len(unsound),
        "max_score_mirror_unsound": max((_score(p) for p in unsound), default=None),
        "n_with_U": len(u_scores),
        "max_score_with_U": max(u_scores, default=None),
        "violations": viol,
    }


def analyze(name: str) -> dict:
    out = os.path.join(HERE, "runs", name)
    ev = []
    if os.path.exists(os.path.join(out, "eval_log.jsonl")):
        with open(os.path.join(out, "eval_log.jsonl")) as f:
            ev = [json.loads(x) for x in f if x.strip()]
    run_meta = {}
    if os.path.exists(os.path.join(out, "run_meta.json")):
        with open(os.path.join(out, "run_meta.json")) as f:
            run_meta = json.load(f)
    best, curve, first = 0.0, [], {}
    for i, e in enumerate(ev):  # entry 0 = the initial program, entry i = iteration i (one worker)
        best = max(best, float(e["combined"]))
        curve.append(round(best, 4))
        first.setdefault(e["genome"], i)
    ops = {}
    mp = os.path.join(out, "mutator_log.tsv")
    if os.path.exists(mp):
        with open(mp) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    ops[parts[1]] = ops.get(parts[1], 0) + 1
    cks = sorted(glob.glob(os.path.join(out, "checkpoints", "checkpoint_*")), key=lambda d: int(d.rsplit("_", 1)[1]))
    snaps = [snapshot(d) for d in cks]
    evs = [(e["genome"], e["combined"]) for e in ev]
    genome_best = {}
    for g, s in evs:
        genome_best[g] = max(genome_best.get(g, 0.0), s)
    summ = {"name": name, **run_meta, "n_evaluations": len(ev), "best_curve": curve,
            "best_at": {str(k): curve[min(k, len(curve) - 1)] for k in (0, 10, 20, 30, 40, 50) if curve},
            "first_seen_iteration": first, "genome_score": dict(sorted(genome_best.items(), key=lambda kv: -kv[1])),
            "mutator_ops": ops, "evaluated_with_D": sum(1 for g, _ in evs if "D" in G.parse_gstr(g)),
            "eval_seconds_total": round(sum(float(e.get("seconds", 0)) for e in ev), 1),
            "checkpoints": snaps, "final": snaps[-1] if snaps else None}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="V0 live")
    ap.add_argument("--config", default=os.path.join(HERE, "config_dyn.yaml"))
    ap.add_argument("--iterations", type=int, default=50)
    ap.add_argument("--seed", default="26", help="mutator seed")
    ap.add_argument("--db-seed", type=int, default=42, help="OpenEvolve random_seed (database sampling)")
    ap.add_argument("--min-eval-seconds", type=float, default=0.0, help="evaluation-time pad (did NOT restore determinism: database.py samples list(set(uuid4 ids)))")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args(argv)
    out = os.path.join(HERE, "runs", a.name)
    if not a.analyze_only:
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        os.environ.update({"E26_VARIANT": a.variant, "E26_EVAL_LOG": os.path.join(out, "eval_log.jsonl"),
                           "E26_MUTATOR_LOG": os.path.join(out, "mutator_log.tsv"), "E26_MUTATOR_SEED": a.seed,
                           "E26_MIN_EVAL_SECONDS": str(a.min_eval_seconds)})
        secs = asyncio.run(_run(a, out))
        with open(os.path.join(out, "run_meta.json"), "w") as f:
            json.dump({"variant": a.variant, "config": os.path.relpath(a.config, UB), "iterations": a.iterations,
                       "mutator_seed": a.seed, "db_seed": a.db_seed, "min_eval_seconds": a.min_eval_seconds, "wall_seconds": round(secs, 1)}, f, indent=1)
    s = analyze(a.name)
    fin = s.get("final") or {}
    print(json.dumps({k: s[k] for k in ("name", "best_at", "genome_score", "mutator_ops")}, indent=1))
    print(json.dumps({k: fin.get(k) for k in ("best_score", "best_genome", "distinct_genomes", "genome_entropy_bits",
                                              "frac_with_D", "frac_with_R_and_D", "map_cells_occupied",
                                              "map_cells_with_D", "archive_with_D", "violations",
                                              "max_score_with_U")}, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""T-5 (design §10.1): a real OpenEvolve run with the in-process SyntheticMutator
(through `LLMModelConfig.init_client`), then audits the database.

    python experiments/no_llm/run_synthetic.py --iterations 30 --config config.yaml
        [--output experiments/no_llm/run_synthetic] [--bank tests/snippets] [--checkpoint-interval 10]
        [--resume-iterations 5] [--workers 2] [--keep]

Asserts (exit code 1 on failure, report in <output>/synthetic_report.json):
  * no program whose Python kill failed the battery (sound_battery = 0) scores unless its Lean is L5, and then only the Lean-only verified value (E19);
  * under reward v2 (metrics carry lean_ladder): every program with ladder in {0,4} scores 0,
    every unverified program (ladder < 5) scores <= 0.19, every L5 program scores >= 0.20;
  * checkpoints were written; a second controller resumes from the last checkpoint and
    advances `last_iteration` (the `openevolve-run.py --checkpoint` code path, in-process
    because init_client cannot be expressed in YAML).
Reports: ladder histogram, feature-cell occupancy, best score per checkpoint (non-decreasing).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
ROOT = os.path.abspath(os.path.join(UB, "..", "..", ".."))
for p in (ROOT, UB, HERE, os.path.join(UB, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ZAR_UB_NO_LLM", "1")

from openevolve.config import LLMModelConfig, load_config  # noqa: E402
from openevolve.controller import OpenEvolve  # noqa: E402
import replay_llm  # noqa: E402


def _configure(cfg, bank: str, workers: int, ckpt_every: int):
    os.environ["ZAR_UB_STUB_BANK"] = os.path.abspath(bank)
    model = LLMModelConfig(name="synthetic", api_base="http://synthetic.invalid/v1", api_key="x",
                           weight=1.0, init_client=replay_llm.make_synthetic)
    cfg.llm.models = [model]
    cfg.llm.evaluator_models = []
    cfg.llm.api_base = model.api_base
    cfg.llm.api_key = "x"
    cfg.checkpoint_interval = ckpt_every
    cfg.evaluator.parallel_evaluations = workers
    cfg.evaluator.use_llm_feedback = False
    return cfg


def _latest_checkpoint(output: str):
    d = os.path.join(output, "checkpoints")
    if not os.path.isdir(d):
        return None
    cps = [c for c in os.listdir(d) if c.startswith("checkpoint_")]
    if not cps:
        return None
    return os.path.join(d, max(cps, key=lambda c: int(c.split("_")[-1])))


def _audit(db) -> dict:
    progs = list(db.programs.values())
    bad, ladders, cells = [], {}, {}
    v2 = any("lean_ladder" in p.metrics for p in progs)
    for p in progs:
        m = p.metrics or {}
        score = float(m.get("combined_score", 0.0) or 0.0)
        sb = m.get("sound_battery")
        lad = m.get("lean_ladder")
        if sb is not None and float(sb) == 0.0 and score > 0.0:
            # E19 (2026-09-22): a wrong Python mirror does not zero a verified Lean prune; such a
            # program may score its Lean-only verified value, but never empirical credit and never
            # without ladder 5.
            lad_ = int(round(float(m.get("lean_ladder", 0) or 0)))
            emp = float(m.get("empirical_gain", 0.0) or 0.0)
            if lad_ != 5 or emp > 0.0:
                bad.append({"id": p.id, "reason": "unsound mirror scored without an L5 Lean kill / with empirical credit", "score": score})
        if lad is not None:
            lad = int(round(float(lad)))
            ladders[lad] = ladders.get(lad, 0) + 1
            if lad in (0, 4) and score > 0.0:
                bad.append({"id": p.id, "reason": f"ladder L{lad} but score > 0", "score": score})
            if lad < 5 and score > 0.19 + 1e-9:
                bad.append({"id": p.id, "reason": f"unverified L{lad} but score > 0.19", "score": score})
            if lad == 5 and float(sb or 0) == 1.0 and score < 0.20 - 1e-9:
                bad.append({"id": p.id, "reason": "L5 but score < 0.20", "score": score})
    # MAP-Elites occupancy: one feature map per island (ProgramDatabase.island_feature_maps)
    for fm in getattr(db, "island_feature_maps", None) or [getattr(db, "feature_map", {}) or {}]:
        for key in fm:
            cells[str(key)] = cells.get(str(key), 0) + 1
    best = max((float((p.metrics or {}).get("combined_score", 0.0) or 0.0) for p in progs), default=0.0)
    return {"n_programs": len(progs), "reward_version": "v2" if v2 else "v1", "ladder_histogram": ladders,
            "feature_cells_occupied": len(cells), "feature_cells": cells, "best_score": best, "violations": bad,
            "unsound_programs": sum(1 for p in progs if float((p.metrics or {}).get("sound_battery", 1) or 0) == 0.0)}


async def _run(a) -> int:
    t0 = time.time()
    if not a.keep and os.path.isdir(a.output):
        shutil.rmtree(a.output)
    cfg = _configure(load_config(a.config), a.bank, a.workers, a.checkpoint_interval)
    initial = os.path.join(UB, "initial_program.py")
    evaluator = os.path.join(UB, "evaluator.py")
    ctl = OpenEvolve(initial_program_path=initial, evaluation_file=evaluator, config=cfg, output_dir=a.output)
    await ctl.run(iterations=a.iterations)
    report = {"iterations": a.iterations, "audit_after_run": _audit(ctl.database)}
    ckpts = sorted((int(c.split("_")[-1]) for c in os.listdir(os.path.join(a.output, "checkpoints"))
                    if c.startswith("checkpoint_")), key=int) if os.path.isdir(os.path.join(a.output, "checkpoints")) else []
    report["checkpoints"] = ckpts
    # best score per checkpoint must be non-decreasing
    bests = []
    for k in ckpts:
        info = os.path.join(a.output, "checkpoints", f"checkpoint_{k}", "best_program_info.json")
        if os.path.exists(info):
            with open(info) as f:
                bests.append(float(json.load(f).get("metrics", {}).get("combined_score", 0.0) or 0.0))
    report["best_per_checkpoint"] = bests
    report["best_non_decreasing"] = all(x <= y + 1e-12 for x, y in zip(bests, bests[1:]))
    # resume
    latest = _latest_checkpoint(a.output)
    report["resume"] = {"checkpoint": latest}
    if latest and a.resume_iterations > 0:
        cfg2 = _configure(load_config(a.config), a.bank, a.workers, a.checkpoint_interval)
        ctl2 = OpenEvolve(initial_program_path=initial, evaluation_file=evaluator, config=cfg2, output_dir=a.output)
        before = int(os.path.basename(latest).split("_")[-1])
        await ctl2.run(iterations=a.resume_iterations, checkpoint_path=latest)
        report["resume"].update({"from_iteration": before, "last_iteration_after": ctl2.database.last_iteration,
                                 "ok": ctl2.database.last_iteration > before,
                                 "audit_after_resume": _audit(ctl2.database)})
    report["seconds"] = round(time.time() - t0, 1)
    ok = (not report["audit_after_run"]["violations"] and bool(ckpts) and report["best_non_decreasing"]
          and report["resume"].get("ok", True) and not report["resume"].get("audit_after_resume", {}).get("violations"))
    report["ok"] = ok
    with open(os.path.join(a.output, "synthetic_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    print("T-5 RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--config", default=os.path.join(UB, "config.yaml"))
    ap.add_argument("--output", default=os.path.join(HERE, "run_synthetic"))
    ap.add_argument("--bank", default=os.path.join(UB, "tests", "snippets"))
    ap.add_argument("--checkpoint-interval", type=int, default=10)
    ap.add_argument("--resume-iterations", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--keep", action="store_true", help="do not wipe --output first")
    a = ap.parse_args(argv)
    return asyncio.run(_run(a))


if __name__ == "__main__":
    sys.exit(main())

"""The closure daemon (design §7 item 3): from an OpenEvolve run to certified bounds.

    python -m zar_ub closure-daemon --checkpoint-dir <run> --targets "9,9,50;12,18,109" \
        [--poll 50] [--budget 5e8] [--now] [--pure|--trust tan2022] [--time 600] [--loop --sleep 60]

One pass:
  1. find the newest `checkpoint_<k>` under <run>/checkpoints (or <run>); with --poll N a pass runs
     only when k has advanced by ≥ N iterations since the last processed checkpoint (--now ignores it);
  2. pick the best program: highest combined_score among lean_ladder == 5, sound_battery == 1,
     pipeline_bug == 0 (ties: proven_gain desc, survivors_left asc, iteration_found asc, id);
  3. promote it when its sha is new (zar_ub.promote.promote_program: python battery + gate with no
     cache + module + leanchecker replay + library check; the gate's `lean_source_filled` artifact is
     used when the program is a filled sketch);
  4. for every target table: the promoted library's Lean mask (zar_ub.promote.library_masks) →
     survivors, Ŵ = Σ_{alive} d̂ (each record's difficulty label);
  5. launch zar_ub.certify.certify_survivors on the survivors when Ŵ ≤ --budget, or the survivor
     count < --min-survivors, or --now; then refresh cache/certs/<tag>/closure_report.md (zar_ub.closure,
     current signature) and append one bullet per launch to experiments/LOG.md (## E16);
  6. suite.update_band_rule({cell: population-mean gain_I}) from the checkpoint's `per_instance`
     artifacts (L5 programs only), when suite exposes it.

State (last processed checkpoint per run) lives in cache/ledger/closure_daemon_state.json.
Nothing here runs inside evaluate(); SAT solving happens only in step 5 (design §7 item 1).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

from .known import Instance
from .casetable import load_table
from . import promote as promote_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(_HERE)
STATE_PATH = os.path.join(UB, "cache", "ledger", "closure_daemon_state.json")
LOG_PATH = os.path.join(UB, "experiments", "LOG.md")
LOG_HEADING = "## E16 — closure daemon launches (zar_ub/closure_daemon.py)"
_CKPT_RE = re.compile(r"checkpoint_(\d+)$")
_PER_INSTANCE_RE = re.compile(r"^train\s+(m(\d+)_n(\d+))_s\d+_t\d+_w\d+:.*?\bgain_lean=([0-9.]+)", re.M)


# --------------------------------------------------------------------------------------
# checkpoints and programs
# --------------------------------------------------------------------------------------
def find_checkpoints(run_dir: str) -> List[Tuple[int, str]]:
    out = []
    for base in (os.path.join(run_dir, "checkpoints"), run_dir):
        for p in glob.glob(os.path.join(base, "checkpoint_*")):
            m = _CKPT_RE.search(p)
            if m and os.path.isdir(os.path.join(p, "programs")):
                out.append((int(m.group(1)), p))
    return sorted(set(out))


def load_programs(ckpt_dir: str) -> List[dict]:
    progs = []
    for p in sorted(glob.glob(os.path.join(ckpt_dir, "programs", "*.json"))):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "code" in d:
            progs.append(d)
    return progs


def _num(metrics: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def eligible(prog: dict) -> bool:
    m = prog.get("metrics") or {}
    return (round(_num(m, "lean_ladder", 0)) == 5 and _num(m, "sound_battery", 0) >= 1.0
            and _num(m, "pipeline_bug", 0) == 0 and _num(m, "combined_score", 0) >= 0.2)


def pick_best(programs: List[dict]) -> Optional[dict]:
    cands = [p for p in programs if eligible(p)]
    if not cands:
        return None

    def key(p):
        m = p.get("metrics") or {}
        return (-_num(m, "combined_score"), -_num(m, "proven_gain"), _num(m, "survivors_left", 1e18),
                int(p.get("iteration_found") or 0), str(p.get("id")))

    return sorted(cands, key=key)[0]


def program_artifacts(prog: dict) -> dict:
    a = prog.get("artifacts_json")
    if isinstance(a, dict):
        return a
    try:
        return json.loads(a) if a else {}
    except ValueError:
        return {}


def promote_checkpoint_program(prog: dict, ckpt_iter: int, run_dir: str, timeout: float = 300.0, **kw):
    """Write the program's code to a temp file and promote it (Python battery + Lean)."""
    arts = program_artifacts(prog)
    filled = arts.get("lean_source_filled")
    meta = {"program_id": prog.get("id"), "iteration": prog.get("iteration_found"), "checkpoint": ckpt_iter,
            "run": os.path.relpath(run_dir, UB), "combined_score": _num(prog.get("metrics") or {}, "combined_score"),
            "program_path": f"checkpoint_{ckpt_iter}/programs/{prog.get('id')}.json"}
    tmp = tempfile.mkdtemp(prefix="zar_ub_daemon_")
    path = os.path.join(tmp, f"program_{str(prog.get('id'))[:8]}.py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(prog["code"])
    try:
        name = f"ckpt{ckpt_iter}_{str(prog.get('id'))[:8]}"
        return promote_mod.promote_program(path, name, meta, lean_source=filled if filled else None, timeout=timeout, **kw)
    finally:
        try:
            os.remove(path)
            os.rmdir(tmp)
        except OSError:
            pass


# --------------------------------------------------------------------------------------
# targets, survivors, certification
# --------------------------------------------------------------------------------------
def parse_targets(raw: str) -> List[Instance]:
    if UB not in sys.path:
        sys.path.insert(0, UB)
    from suite import parse_targets as _pt  # noqa: E402

    return _pt(raw)


def survivors(inst: Instance, tab, timeout: float = 900.0, **kw) -> dict:
    masks, info = promote_mod.library_masks([(inst, tab)], timeout=timeout, **kw)
    mask = masks[0]
    if mask is None:
        return {"ok": False, "error": "library gate failed", "info": info}
    alive = [i for i, k in enumerate(mask) if not k]
    work = float(sum(float(tab.records[i].d) for i in alive))
    return {"ok": True, "mask": mask, "alive": alive, "killed": len(mask) - len(alive), "n_cases": len(mask),
            "work": work, "censored": sum(1 for i in alive if tab.records[i].censored), "info": info}


def certify(inst: Instance, tab, surv: dict, time_limit: float, verbose: bool = True) -> dict:
    from . import certify as cert_mod

    cases = [(list(tab.records[i].rows), list(tab.records[i].cols)) for i in surv["alive"]]
    if not cert_mod.tools_available():
        return {"error": "cadical/drat-trim/lrat-check not built (tools/build_tools.sh)", "n_cases": len(cases)}
    manifest = cert_mod.certify_survivors(inst, cases, time_limit=time_limit, verbose=verbose)
    manifest["pruned_by_lean"] = surv["killed"]
    # refresh the closure report with the current write_closure_report signature (owner G may extend it)
    try:
        from . import closure as closure_mod
        from .lean_gate import GateResult

        g = GateResult(ok=True, ladder=5, scanned_ok=True, compiled=True, typed_ok=True, axioms=list(surv["info"].get("axioms", [])),
                       axioms_ok=True, kill_mask=list(surv["mask"]), lean_file=f"library route={surv['info'].get('route')} entries={surv['info'].get('entries')}")
        with open(os.path.join(cert_mod.CERT_DIR, inst.tag, "manifest.json")) as fh:
            full = json.load(fh)
        src = promote_mod.LIBRARY_EVOLVED_LEAN if surv["info"].get("entries") else promote_mod.LIBRARY_LEAN
        src += "".join(f"-- accepted entry {s}: {promote_mod.lean_name(s)}\n" for s in surv["info"].get("entries", []))
        manifest["closure_report"] = closure_mod.write_closure_report(inst, tab, g, full, src)
    except Exception as e:  # noqa: BLE001 - the report is a convenience; the manifest is the record
        manifest["closure_report_error"] = f"{e.__class__.__name__}: {e}"
    return manifest


# --------------------------------------------------------------------------------------
# band rule, state, log
# --------------------------------------------------------------------------------------
def mean_gain_by_cell(programs: List[dict]) -> Dict[Tuple[int, int], float]:
    acc: Dict[Tuple[int, int], List[float]] = {}
    for p in programs:
        if not eligible(p):
            continue
        per = str(program_artifacts(p).get("per_instance") or "")
        for m in _PER_INSTANCE_RE.finditer(per):
            acc.setdefault((int(m.group(2)), int(m.group(3))), []).append(float(m.group(4)))
    return {cell: sum(v) / len(v) for cell, v in acc.items() if v}


def update_band_rule(programs: List[dict], note: str = "") -> Optional[dict]:
    if UB not in sys.path:
        sys.path.insert(0, UB)
    try:
        import suite as suite_mod  # noqa: E402
    except Exception:  # noqa: BLE001
        return None
    fn = getattr(suite_mod, "update_band_rule", None)
    if not callable(fn):
        return None
    gains = mean_gain_by_cell(programs)
    if not gains:
        return {"gains": {}, "note": "no per_instance artifacts in this checkpoint"}
    state = fn(gains, note=note)
    return {"gains": {f"{m},{n}": g for (m, n), g in gains.items()}, "graduated": state.get("graduated"), "entered": state.get("entered")}


def load_state(path: str = STATE_PATH) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"runs": {}}


def save_state(state: dict, path: str = STATE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def append_log(bullet: str, path: str = LOG_PATH) -> None:
    """Append-only: the log is shared; a heading is added once."""
    try:
        with open(path, encoding="utf-8") as fh:
            has_heading = LOG_HEADING in fh.read()
    except OSError:
        has_heading = False
    with open(path, "a", encoding="utf-8") as fh:
        if not has_heading:
            fh.write(f"\n{LOG_HEADING}\n\nOne bullet per certification launch (design §7 item 3).\n\n")
        fh.write(bullet.rstrip("\n") + "\n")


# --------------------------------------------------------------------------------------
# one pass
# --------------------------------------------------------------------------------------
def run_pass(run_dir: str, targets: List[Instance], *, use_table: bool, budget: float, poll: int, now: bool,
             time_limit: float, min_survivors: int = 200, gate_timeout: float = 900.0, state: Optional[dict] = None,
             state_path: str = STATE_PATH, log_path: str = LOG_PATH, verbose: bool = True, no_promote: bool = False,
             **promote_kw) -> dict:
    t0 = time.time()
    run_dir = os.path.abspath(run_dir)
    state = state if state is not None else load_state(state_path)
    rs = state.setdefault("runs", {}).setdefault(run_dir, {"last_checkpoint": -1, "launches": []})
    summary: dict = {"run": run_dir, "checkpoint": None, "program": None, "promotion": None, "targets": [], "band_rule": None,
                     "launched": [], "skipped": False}

    def log(msg):
        if verbose:
            print(f"[daemon] {msg}", flush=True)

    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        summary["skipped"] = "no checkpoints"
        log(f"no checkpoint_* under {run_dir}")
        return summary
    it, ckpt = ckpts[-1]
    summary["checkpoint"] = it
    if not now and rs["last_checkpoint"] >= 0 and it - rs["last_checkpoint"] < poll:
        summary["skipped"] = f"checkpoint {it} < last {rs['last_checkpoint']} + poll {poll}"
        log(summary["skipped"])
        return summary
    programs = load_programs(ckpt)
    best = pick_best(programs)
    log(f"checkpoint {it}: {len(programs)} programs, {sum(1 for p in programs if eligible(p))} eligible (L5)")
    if best is None:
        summary["skipped"] = "no L5 program"
    else:
        m = best.get("metrics") or {}
        summary["program"] = {"id": best.get("id"), "iteration_found": best.get("iteration_found"),
                              "combined_score": _num(m, "combined_score"), "proven_gain": _num(m, "proven_gain"),
                              "survivors_left": _num(m, "survivors_left")}
        log(f"best L5 program {best.get('id')} (iteration {best.get('iteration_found')}, score {_num(m, 'combined_score'):.4f})")
        if not no_promote:
            res = promote_checkpoint_program(best, it, run_dir, verbose=verbose, **promote_kw)
            summary["promotion"] = {"ok": res.ok, "already": res.already, "sha": res.sha, "stage": res.stage, "reason": res.reason,
                                    "new_kills": res.new_kills, "n_ledger": res.n_ledger, "novelty_on": res.novelty_on,
                                    "replay": res.replay.get("tool"), "seconds": round(res.seconds, 1)}
    # band rule
    try:
        summary["band_rule"] = update_band_rule(programs, note=f"closure-daemon {os.path.basename(run_dir)} checkpoint {it}")
    except Exception as e:  # noqa: BLE001
        summary["band_rule"] = {"error": f"{e.__class__.__name__}: {e}"}
    # targets
    for inst in targets:
        tab = load_table(inst, use_table=use_table)
        tinfo: dict = {"tag": inst.tag, "trust": "tan2022" if use_table else "pure"}
        if tab is None:
            tinfo["error"] = "no cached table (build it with python -m zar_ub table ...)"
            summary["targets"].append(tinfo)
            log(f"{inst.tag}: {tinfo['error']}")
            continue
        surv = survivors(inst, tab, timeout=gate_timeout, **{k: v for k, v in promote_kw.items() if k in ("ledger_path", "evolved_dir")})
        if not surv["ok"]:
            tinfo["error"] = surv["error"]
            summary["targets"].append(tinfo)
            continue
        tinfo.update({"n_cases": surv["n_cases"], "killed": surv["killed"], "survivors": len(surv["alive"]), "work": surv["work"],
                      "censored": surv["censored"], "route": surv["info"]["route"], "entries": surv["info"]["entries"]})
        launch = now or surv["work"] <= budget or len(surv["alive"]) < min_survivors
        tinfo["launch"] = launch
        log(f"{inst.tag}: {surv['n_cases']} cases, library kills {surv['killed']} ({surv['info']['route']}, entries {surv['info']['entries']}), "
            f"survivors {len(surv['alive'])}, W^ = {surv['work']:.3g} (budget {budget:.3g}) -> {'LAUNCH' if launch else 'wait'}")
        if launch:
            man = certify(inst, tab, surv, time_limit=time_limit, verbose=verbose)
            tinfo["manifest"] = {k: v for k, v in man.items() if k != "certs"}
            claim = man.get("certified", -1) == len(surv["alive"]) and "error" not in man
            tinfo["claim"] = claim
            # Batch 2 (G-closure TODO 3 / H TODO 4): a complete certification is followed by the Tier-1 closure
            # seam, which emits lean/ZarPrune/Closures/Z_m_n_w.lean, checks it and owns closure_report.md
            # (last writer): the claim then rests on `z_m_n_le_u` + the LRAT files, not on the Tier-2 report.
            tinfo["closure"] = None
            if claim:
                try:
                    from . import closure as closure_mod

                    cl = closure_mod.close_instance(inst, tinfo["trust"], tier="1", verbose=verbose)
                    tinfo["closure"] = {"established": bool(cl.get("established")), "path": cl.get("closure_file"),
                                        "tier": cl.get("tier", "1"), "check_ok": bool((cl.get("check") or {}).get("ok")),
                                        "report": cl.get("report")}
                    if not tinfo["closure"]["established"]:
                        claim = False  # the Tier-1 seam did not confirm the Tier-2 claim
                        tinfo["claim"] = claim
                except Exception as e:  # noqa: BLE001
                    tinfo["closure"] = {"error": f"{e.__class__.__name__}: {e}"}
            pid = summary["program"]["id"] if summary["program"] else "-"
            promo = summary["promotion"] or {}
            bullet = (f"- {time.strftime('%Y-%m-%d %H:%M')} launch: run `{os.path.relpath(run_dir, UB)}` checkpoint {it}, program `{pid}` "
                      f"(score {summary['program']['combined_score']:.4f})" if summary["program"] else
                      f"- {time.strftime('%Y-%m-%d %H:%M')} launch: run `{os.path.relpath(run_dir, UB)}` checkpoint {it}, no L5 program")
            bullet += (f"; promotion {'already in ledger' if promo.get('already') else ('OK' if promo.get('ok') else 'REFUSED (' + str(promo.get('stage')) + ')')}"
                       f" sha `{promo.get('sha')}` ledger={promo.get('n_ledger')}" if promo else "; promotion skipped")
            bullet += (f"; target {inst.tag} ({tinfo['trust']}): {surv['n_cases']} cases, library kills {surv['killed']} "
                       f"(route {surv['info']['route']}, entries {surv['info']['entries']}), survivors {len(surv['alive'])}, "
                       f"W^={surv['work']:.3g} vs budget {budget:.3g}{' (--now)' if now else ''}; certify: "
                       f"{man.get('certified', '?')}/{len(surv['alive'])} certified, sat {man.get('sat', '?')}, timeout {man.get('timeout', '?')}, "
                       f"failed {man.get('failed', '?')}" + (f"; error: {man['error']}" if man.get("error") else "") +
                       f"; claim z({inst.m},{inst.n};{inst.s},{inst.t}) <= {inst.w - 1}: {'YES' if claim else 'NO'}" +
                       (f"; report `{os.path.relpath(man['closure_report'], UB)}`" if man.get("closure_report") else "") +
                       (f"; Tier-1 closure {'ESTABLISHED' if tinfo['closure'].get('established') else 'NOT established'}"
                        f" `{os.path.relpath(str(tinfo['closure'].get('path')), UB) if tinfo['closure'].get('path') else '-'}`"
                        if tinfo.get("closure") and "error" not in tinfo["closure"] else
                        (f"; Tier-1 closure error: {tinfo['closure']['error']}" if tinfo.get("closure") else "")))
            append_log(bullet, log_path)
            rs["launches"].append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "checkpoint": it, "program": pid, "target": inst.tag,
                                   "survivors": len(surv["alive"]), "certified": man.get("certified"), "claim": claim})
            summary["launched"].append(inst.tag)
            log(f"{inst.tag}: certify -> {man.get('certified', '?')}/{len(surv['alive'])} certified; claim {'YES' if claim else 'NO'}")
        summary["targets"].append(tinfo)
    rs["last_checkpoint"] = it
    rs["last_pass"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_state(state, state_path)
    summary["seconds"] = round(time.time() - t0, 1)
    return summary


# --------------------------------------------------------------------------------------
# CLI plugin
# --------------------------------------------------------------------------------------
def _cli(args) -> int:
    targets = parse_targets(args.targets)
    use_table = not (args.pure or args.trust == "pure")
    kw = dict(use_table=use_table, budget=float(args.budget), poll=int(args.poll), now=bool(args.now), time_limit=float(args.time),
              min_survivors=int(args.min_survivors), gate_timeout=float(args.gate_timeout), no_promote=bool(args.no_promote))
    passes = 0
    while True:
        summary = run_pass(args.checkpoint_dir, targets, **kw)
        passes += 1
        print(json.dumps(summary, indent=1, default=str), flush=True)
        if not args.loop or (args.max_passes and passes >= args.max_passes):
            break
        kw["now"] = False
        time.sleep(float(args.sleep))
    return 0


def register(sub) -> None:
    p = sub.add_parser("closure-daemon", help="promote the best L5 program of a run and certify the survivors of the targets (design §7)")
    p.add_argument("--checkpoint-dir", required=True, help="an OpenEvolve output dir (with checkpoints/checkpoint_*) or its checkpoints/ dir")
    p.add_argument("--targets", required=True, help='"m,n,w;..." (s=t=3) or 5-tuples "m,n,s,t,w;..."')
    p.add_argument("--poll", type=int, default=50, help="process a checkpoint only every N iterations")
    p.add_argument("--budget", type=float, default=5e8, help="launch certification when Σ d̂ over the survivors ≤ budget")
    p.add_argument("--now", action="store_true", help="launch certification on every target in this pass, whatever the budget")
    p.add_argument("--pure", action="store_true", help="targets built in pure mode")
    p.add_argument("--trust", choices=["pure", "tan2022"], default=None)
    p.add_argument("--time", type=float, default=600.0, help="cadical time limit per case (s)")
    p.add_argument("--min-survivors", type=int, default=200, help="launch when fewer survivors remain")
    p.add_argument("--gate-timeout", type=float, default=900.0)
    p.add_argument("--no-promote", action="store_true", help="only recompute survivors / certify with the current ledger")
    p.add_argument("--loop", action="store_true", help="keep polling (sleep --sleep seconds between passes)")
    p.add_argument("--sleep", type=float, default=60.0)
    p.add_argument("--max-passes", type=int, default=0)
    p.set_defaults(func=_cli)


if __name__ == "__main__":  # python -m zar_ub.closure_daemon ...
    ap = argparse.ArgumentParser()
    register(ap.add_subparsers(dest="cmd", required=True))
    a = ap.parse_args()
    sys.exit(a.func(a))

"""E26 evaluator wrapper: the LIVE evaluator (evaluator.py, unchanged) decides soundness and
verification on today's suite S0; the verified branch is then RE-SCORED with a reward variant of
zar_ub/reward_variants.py on the E25 S1 universe, from Lean masks computed by the real gate.

    E26_VARIANT   "V0 live" (default: today's combined_score, untouched) or a key of
                  reward_variants.VARIANTS ("VR recommended", "V3+V4+V5", "V0/S1 suite only", ...)
    E26_FEATURE_SOURCE  unused by the score; the concentration descriptor is always computed on S1

Branch logic (identical ordering to reward.py; nothing is relaxed):
    live combined_score == 0                    -> 0             (hard zero: battery/ladder 0,4/errors)
    live lean_ladder in {1,2,3}                 -> live score    (unverified, <= 0.19)
    live lean_ladder == 5                       -> variant verified score on S1 cells, from the Lean
                                                   masks of ONE gate run over the 25 S1 tables
                                                   (tables where that gate is not L5 contribute gain 0)
Extra metrics: sc_<variant> for every compared variant, gain_concentration (max share of
a_I*gain_I over S1 TRAIN∪TARGET∪GEN cells), n_cells_gained, s1_ladder_min, s1_witnessed_kills.
`proven_gain` is replaced by the active variant's inner term (the verified [0,1] quantity);
the live value is kept as proven_gain_v0.

S1 Lean masks are cached per sha1(LEAN_SOURCE, SCHEMA_DATA, universe) in cache/ under this
directory (gate HMAC cache in gate_cache/).  Case tables are read-only; this module never writes
the PIPELINE_BUG sentinel (a Lean kill of a witnessed S1 case is reported as s1_witnessed_kills).
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import time
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
E25 = os.path.join(UB, "experiments", "E25_reward")
for p in (UB, E25, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("ZAR_UB_NO_LLM", "1")

from zar_ub import lemmas, reward  # noqa: E402
from zar_ub import reward_variants as RV  # noqa: E402
from zar_ub.known import Instance  # noqa: E402
from zar_ub.lean_gate import run_gate_multi  # noqa: E402
import universe as U  # noqa: E402  (E25, read-only)

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover
    EvaluationResult = None

_spec = importlib.util.spec_from_file_location("zar_ub_live_evaluator", os.path.join(UB, "evaluator.py"))
live = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(live)

CACHE = os.path.join(HERE, "cache")
GATE_CACHE = os.path.join(HERE, "gate_cache")
GATE_TIMEOUT = 1800.0
#: the E25 universe as it stood when E26 started (A1 may add *_pure_gt tables later; frozen here)
UNIVERSE_KEYS = [
    "m10_n10_s3_t3_w61_pure", "m10_n11_s3_t3_w65_pure", "m11_n11_s3_t3_w70_pure", "m11_n12_s3_t3_w75_pure",
    "m12_n12_s3_t3_w81_pure", "m9_n10_s3_t3_w55_pure", "m9_n9_s3_t3_w50_pure", "m10_n14_s3_t3_w78_pure",
    "m12_n13_s3_t3_w87_pure", "m13_n13_s3_t3_w93_pure", "m10_n20_s3_t3_w103_pure", "m11_n21_s3_t3_w117_pure",
    "m9_n12_s3_t3_w64_pure", "m12_n18_s3_t3_w109", "m9_n23_s3_t3_w104", "m10_n22_s3_t3_w111",
    "m10_n23_s3_t3_w113", "m11_n23_s3_t3_w124", "m13_n19_s3_t3_w123", "m16_n17_s3_t3_w134",
    "m8_n9_s2_t2_w27_pure", "m9_n9_s4_t4_w62_pure", "m10_n19_s3_t3_w99_pure_gt", "m9_n16_s3_t3_w78_pure_gt",
    "m9_n18_s3_t3_w86_pure_gt"]
COMPARED = {"V0": "V0 live", "V0S1": "V0/S1 suite only", "V345": "V3+V4+V5", "VR": "VR recommended"}

_ST: dict = {}


def _universe():
    if "uni" not in _ST:
        uni = U.universe(include_gt=True)
        missing = [k for k in UNIVERSE_KEYS if k not in uni]
        if missing:
            raise RuntimeError(f"E26: universe tables missing: {missing}")
        tabs = {k: U.load(k, uni[k]["path"]) for k in UNIVERSE_KEYS}
        h = hashlib.sha1()
        for k in UNIVERSE_KEYS:
            h.update(f"{k}:{tabs[k].table_hash}:{len(tabs[k].records)};".encode())
        _ST["uni"], _ST["tabs"], _ST["sig"] = uni, tabs, h.hexdigest()[:16]
    return _ST["uni"], _ST["tabs"], _ST["sig"]


def _inst(u: dict) -> Instance:
    return Instance(u["m"], u["n"], u["s"], u["t"], u["w"])


def s1_masks(src: str, sd: dict) -> dict:
    """{"tables": {key: {"ladder", "lean"}}, "gate_seconds", "cache": bool, "schema_error"} (cached)."""
    uni, tabs, sig = _universe()
    sd = sd if isinstance(sd, dict) else {}
    key = hashlib.sha1((src + "\x00" + json.dumps(sd, sort_keys=True) + "\x00" + sig).encode()).hexdigest()[:20]
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"s1_{key}.json")
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            d["cache"] = True
            return d
        t0 = time.time()
        out = {"key": key, "universe_sig": sig, "tables": {}, "schema_error": None, "gate_seconds": 0.0}
        gk = [k for k in UNIVERSE_KEYS if tabs[k].records]
        insts = [_inst(uni[k]) for k in gk]
        terms = None
        if sd.get("farkas"):
            errs = []
            for k, inst in zip(gk, insts):
                ok, e = lemmas.validate(sd, inst)
                if not ok:
                    errs.append(f"[{k}] " + "; ".join(map(str, e)))
            if errs:
                out["schema_error"] = "\n".join(errs[:5])
            else:
                terms = [list(lemmas.render_terms(sd, inst, "ZarPrune.Cand.target" + ("" if i == 0 else str(i))))
                         for i, inst in enumerate(insts)]
        if out["schema_error"] is None:
            inputs = [(inst, [(r.rows, r.cols) for r in tabs[k].records]) for inst, k in zip(insts, gk)]
            res = run_gate_multi(inputs, src, schema_terms=terms, timeout=GATE_TIMEOUT, tag="e26", sketch=True,
                                 use_cache=True, cache_dir=GATE_CACHE)
            for k, g in zip(gk, res):
                lad = reward.ladder_of(g)
                km = getattr(g, "kill_mask", None)
                n = len(tabs[k].records)
                out["tables"][k] = {"ladder": lad, "lean": [bool(x) for x in km] if (lad == 5 and km is not None
                                                                                      and len(km) == n) else None,
                                    "errors": list(getattr(g, "errors", []) or [])[:2]}
        for k in UNIVERSE_KEYS:
            if not tabs[k].records:
                out["tables"][k] = {"ladder": 5, "lean": [], "errors": []}
        out["gate_seconds"] = time.time() - t0
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, path)
        out["cache"] = False
        return out


def cells_of(masks: Dict[str, Optional[List[bool]]]) -> List[RV.Cell]:
    uni, tabs, _ = _universe()
    return [RV.cell_from_table(k, uni[k]["roles"], tabs[k], masks.get(k)) for k in UNIVERSE_KEYS]


def zero_cells() -> List[RV.Cell]:
    if "zero_cells" not in _ST:
        _ST["zero_cells"] = cells_of({})
    return _ST["zero_cells"]


def concentration(cells: List[RV.Cell]) -> Dict[str, float]:
    sc = RV.SUITES["S1"]
    xs = []
    for c in cells:
        if c.n_surv > 0 and any(RV._in(c, sc[r]) for r in sc):
            xs.append(RV.importance(c.W) * c.gain)
    tot = sum(xs)
    return {"gain_concentration": (max(xs) / tot) if tot > 0 else 0.0,
            "n_cells_gained": float(sum(1 for x in xs if x > 0))}


def variant_scores(cells, ladder: int, hard_zero: bool, live_score: float) -> Dict[str, float]:
    out = {}
    for short, name in COMPARED.items():
        if name == "V0 live":
            out[short] = live_score
        else:
            out[short] = RV.score(cells, RV.VARIANTS[name], ladder=ladder, hard_zero=hard_zero,
                                  unverified=live_score)
    return out


def _result(metrics, artifacts):
    if EvaluationResult is not None:
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    return metrics


def evaluate(program_path: str):
    t0 = time.time()
    variant = os.environ.get("E26_VARIANT", "V0 live")
    r = live.evaluate(program_path)
    m = dict(r.metrics if hasattr(r, "metrics") else r)
    art = dict(getattr(r, "artifacts", {}) or {})
    live_score = float(m.get("combined_score", 0.0) or 0.0)
    ladder = int(round(float(m.get("lean_ladder", 0) or 0)))
    hard_zero = live_score == 0.0
    info = {"s1": "skipped (not verified on S0)"}
    cells = zero_cells()
    s1_min, wit = -1.0, 0
    if not hard_zero and ladder == 5:
        s1 = live._stage1_data(program_path)  # memo hit: the sandboxed run's LEAN_SOURCE / SCHEMA_DATA
        d = s1_masks(s1["lean_source"], s1.get("schema_data") or {})
        _, tabs, _ = _universe()
        masks = {k: v.get("lean") for k, v in d["tables"].items()}
        cells = cells_of(masks)
        lads = [v.get("ladder", 0) for v in d["tables"].values()]
        s1_min = float(min(lads)) if lads else 0.0
        for k, lm in masks.items():
            if lm:
                wit += sum(1 for i, rec in enumerate(tabs[k].records) if lm[i] and reward.is_witnessed(rec))
        info = {"s1": "ok", "cache": d.get("cache"), "gate_seconds": round(d.get("gate_seconds", 0.0), 1),
                "schema_error": d.get("schema_error"), "ladder_min": s1_min}
    sc = variant_scores(cells, ladder, hard_zero, live_score)
    if variant == "V0 live":
        combined = live_score
        inner = float(m.get("proven_gain", 0.0) or 0.0)
    else:
        cfg = RV.VARIANTS[variant]
        combined = RV.score(cells, cfg, ladder=ladder, hard_zero=hard_zero, unverified=live_score)
        inner = RV.inner(cells, cfg) if (ladder == 5 and not hard_zero) else 0.0
    if wit:
        combined = 0.0  # a Lean kill of a witnessed case would be a pipeline bug: never reward it
    m["proven_gain_v0"] = float(m.get("proven_gain", 0.0) or 0.0)
    m["combined_score_v0"] = live_score
    m["combined_score"] = float(combined)
    m["proven_gain"] = float(inner)
    for k, v in sc.items():
        m[f"sc_{k}"] = float(v)
    m.update(concentration(cells) if (ladder == 5 and not hard_zero) else {"gain_concentration": 0.0,
                                                                          "n_cells_gained": 0.0})
    m["s1_ladder_min"] = s1_min
    m["s1_witnessed_kills"] = float(wit)
    m["e26_seconds"] = time.time() - t0
    lines = [f"E26 reward variant: {variant}; live V0 score {live_score:.4f}; this score {combined:.4f}; {info}"]
    for c in cells:
        if c.gain > 0:
            lines.append(f"  {c.key}: gain={c.gain:.3f} tail={c.tail:.3f} closed={c.closed} W={c.W:.0f}")
    art["e26_variant"] = "\n".join(lines)
    # determinism pad: OpenEvolve keeps 3 iterations in flight and snapshots the database when it
    # submits the next one; with ~0.2 s cached evaluations, how many results were already merged at
    # submit time depended on scheduling (E26 found two runs with identical seeds diverging at
    # iteration 4).  A minimum evaluation time lets the controller merge each result first.
    pad = float(os.environ.get("E26_MIN_EVAL_SECONDS", "0") or 0)
    if pad > 0 and time.time() - t0 < pad:
        time.sleep(pad - (time.time() - t0))
    log = os.environ.get("E26_EVAL_LOG")
    if log:
        try:
            import genome as G  # text parse of the NOTES marker; the program is never imported here
            with open(program_path, encoding="utf-8") as f:
                g, rev = G.parse(f.read())
            with open(log, "a") as f:
                f.write(json.dumps({"t": time.time(), "genome": G.gstr(g), "rev": rev, "combined": float(combined),
                                    "ladder": ladder, "sound_battery": m.get("sound_battery"),
                                    **{f"sc_{k}": v for k, v in sc.items()},
                                    "gain_concentration": m["gain_concentration"],
                                    "s1_witnessed_kills": wit, "seconds": round(m["e26_seconds"], 2)}) + "\n")
        except OSError:
            pass
    return _result(m, art)


if __name__ == "__main__":
    res = evaluate(sys.argv[1])
    mm = res.metrics if hasattr(res, "metrics") else res
    print(json.dumps({k: mm[k] for k in mm if k.startswith(("sc_", "combined", "proven", "lean_ladder",
                                                            "sound_battery", "gain_conc", "n_cells", "s1_", "e26"))},
                     indent=1))
    if hasattr(res, "artifacts"):
        print(res.artifacts.get("e26_variant", ""))

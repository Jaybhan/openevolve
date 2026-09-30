"""OpenEvolve evaluator for evolved, Lean-verified prune libraries — design §5.5 (cascade).

  stage 1  (≈1 s)   copy the cache tables to a temp dir; run `run_candidate.py` there in a
                    subprocess (90 s, no network) → Python masks K^P on every table, the
                    counterexample battery, the static scan of LEAN_SOURCE, SCHEMA_DATA validation.
                    combined_score = 0 if hard-zero else 0.01 + 0.08·E.
  stage 2  (2–6 s)  ONE Lean process (`zar_ub.lean_gate.run_gate_multi`) on TRAIN ∪ GEN ∪ TARGET
                    (E27: only each table's library survivors + witnessed cases are sent to Lean)
                    plus the witnessed battery cases → ladder L0–L5, K^L per instance, axioms,
                    holes.  Scored WITHOUT the target term (design §5.5).
  stage 3  (≈0 s)   the same masks scored WITH the TARGET tables (full §5.3 formula).
  evaluate()        stage 1 then the full score (used when cascade_evaluation is off / by the CLI).

Everything is deterministic and LLM-free.  Only the Lean-evaluated mask of the candidate earns
credit; the Python mask feeds the battery and the ≤ 0.19 unverified band.  Scoring is marginal
over the PROVED library (`counting P`): its masks come from the table's `baseline_lean_mask`
when the table has one, else are computed once and cached in cache/library_masks_<hash>.json.
When a candidate is not L5 its "proven part" is that library (the proved-library floor):
proven_gain = 0, survivors_left/agreement/artifacts are computed against the library masks.

A Lean-proven kill of a SAT-witnessed case is a definitions/encoding mismatch: the evaluator
writes the sentinel cache/PIPELINE_BUG (with the case), returns 0, and every later evaluation
returns 0 until the sentinel is removed by a human — the run halts.

Formulas, metric keys and artifacts live in zar_ub/reward.py (§5.2–5.4).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from zar_ub import reward  # noqa: E402
from zar_ub.lean_gate import run_gate_multi, static_scan  # noqa: E402
import suite as suite_mod  # noqa: E402
from suite import load_suite  # noqa: E402

try:
    from zar_ub import lemmas  # noqa: E402  (schema registry; may be absent in this batch)
except Exception:  # noqa: BLE001
    lemmas = None

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover
    EvaluationResult = None

PY = sys.executable
CACHE_DIR = os.path.join(_HERE, "cache")
SENTINEL = os.path.join(CACHE_DIR, "PIPELINE_BUG")
LEDGER = os.path.join(CACHE_DIR, "ledger", "accepted_prunes.jsonl")
#: the proved baseline B (design §5.1): `baseline ∪ counting` (∪ Evolved.lean once promotion exists).
#: One definition for the whole system: zar_ub.casetable.LIBRARY_LEAN is what `table --baseline`
#: stores as baseline_lean_mask; the evaluator only recomputes it for tables without a stored mask.
try:
    from zar_ub.casetable import LIBRARY_LEAN  # noqa: E402
except ImportError:  # pragma: no cover
    LIBRARY_LEAN = "def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)\n"
CANDIDATE_TIMEOUT = 90.0
GATE_TIMEOUT = 480.0  # E27: suite v3 has ~4x the cases of v2 (the gate sees only library survivors, see below)
_MEMO_MAX = 32

_STATE: dict = {"suite": None, "table_hash": None, "library": None}
_MEMO_S1: "OrderedDict[str, dict]" = OrderedDict()
_MEMO_GATE: "OrderedDict[str, dict]" = OrderedDict()


# --------------------------------------------------------------------------------------
# suite, table hash, library masks (loaded once per process)
# --------------------------------------------------------------------------------------
def _suite() -> Dict[str, list]:
    if _STATE["suite"] is None:
        s = load_suite(verbose=True)
        for k in reward.ALL_KINDS:
            s.setdefault(k, [])
        _STATE["suite"] = s
    return _STATE["suite"]


def _suite_version() -> str:
    return str(getattr(suite_mod, "SUITE_VERSION", "v1"))


def _table_hash(suite) -> str:
    """sha1 over the suite's table files (design §5.4) — the tables' own `table_hash` when set."""
    if _STATE["table_hash"] is None:
        from zar_ub.casetable import default_path
        h = hashlib.sha1()
        for kind in reward.ALL_KINDS:
            for inst, tab in suite.get(kind, []):
                th = getattr(tab, "table_hash", None)
                if th:
                    h.update(f"{kind}:{inst.tag}:{th}\n".encode())
                    continue
                p = default_path(inst, getattr(tab, "use_table", True))
                h.update(f"{kind}:{inst.tag}:".encode())
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        h.update(hashlib.sha1(f.read()).hexdigest().encode())
                h.update(b"\n")
        _STATE["table_hash"] = h.hexdigest()
    return _STATE["table_hash"]


def _scored(suite) -> List[Tuple[str, object, object]]:
    return [(kind, inst, tab) for kind in reward.SCORED_KINDS for (inst, tab) in suite.get(kind, [])]


def _library_masks(suite) -> Dict[str, Optional[List[bool]]]:
    """Kill masks of the PROVED library per scored table (tag → mask); the marginal baseline."""
    if _STATE["library"] is not None:
        return _STATE["library"]
    from zar_ub.lean_gate import library_hash  # sha1 over lean/ZarPrune.lean + lean/ZarPrune/*.lean
    h = hashlib.sha1((library_hash() + "\x00" + LIBRARY_LEAN).encode()).hexdigest()[:12]
    cache = os.path.join(CACHE_DIR, f"library_masks_{h}.json")
    need = [(inst, [(r.rows, r.cols) for r in tab.records]) for kind, inst, tab in _scored(suite)
            if tab.records and not getattr(tab, "baseline_lean_mask", None)]
    keys = [inst.tag for inst, _ in need]
    d: Dict[str, Optional[List[bool]]] = {}
    if os.path.exists(cache):
        try:
            with open(cache) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            d = {}
    missing = [(inst, cases) for inst, cases in need if inst.tag not in d]
    if missing:
        res = _gate(missing, LIBRARY_LEAN, None, timeout=300.0, tag="library")
        for (inst, _), g in zip(missing, res):
            d[inst.tag] = list(g.kill_mask) if (g is not None and g.ok and g.kill_mask) else None
            if d[inst.tag] is None:
                print(f"[evaluator] WARNING: library gate failed on {inst.tag}: {g.errors[:3] if g else 'no result'}",
                      file=sys.stderr)
        try:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w") as fh:
                json.dump(d, fh)
        except OSError:
            pass
    _STATE["library"] = {k: d.get(k) for k in keys}
    return _STATE["library"]


def _gate(inputs, src, schema_terms, timeout, tag, facts=None):
    """run_gate_multi with the v2 keyword arguments, falling back to the v1 signature."""
    try:
        return run_gate_multi(inputs, src, schema_terms=schema_terms, timeout=timeout, tag=tag, sketch=True,
                              use_cache=gate_cache_allowed(),
                              facts=facts)
    except TypeError:
        try:
            return run_gate_multi(inputs, src, schema_terms=schema_terms, timeout=timeout, tag=tag, sketch=True,
                                  use_cache=gate_cache_allowed())
        except TypeError:
            return run_gate_multi(inputs, src, timeout=timeout, tag=tag)


def _table_trust(tab) -> str:
    """The provenance a table was built with: what its conditional prunes may assume."""
    tr = getattr(tab, "trust", "") or ""
    if tr:
        return tr
    return "tan2022" if getattr(tab, "use_table", True) else "pure"


def _facts_for(inst, tab) -> List[str]:
    """Lean `Fact` terms granted to a scored instance (design §8.2): ledger.facts_for at the table's trust."""
    try:
        from zar_ub import ledger  # noqa: E402
        return [f.lean for f in ledger.facts_for(inst, _table_trust(tab))]
    except Exception as ex:  # noqa: BLE001
        print(f"[evaluator] WARNING: facts_for failed on {inst.tag}: {ex}", file=sys.stderr)
        return []


# --------------------------------------------------------------------------------------
# stage 1: the candidate subprocess on a copy of the cache
# --------------------------------------------------------------------------------------
def _program_sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()


def _memo_put(memo: "OrderedDict", key: str, val: dict) -> dict:
    memo[key] = val
    while len(memo) > _MEMO_MAX:
        memo.popitem(last=False)
    return val


_SANDBOX = {"checked": False, "ok": False, "backend": "none"}


def _note_sandbox(sandboxed: bool, backend: str) -> None:
    if not _SANDBOX["checked"]:
        _SANDBOX.update(checked=True, ok=bool(sandboxed), backend=backend)
        if not sandboxed:
            print("[evaluator] WARNING: no OS sandbox (sandbox-exec/bwrap) for the candidate subprocess; "
                  "the persistent gate cache is DISABLED for this process (slower, still sound).", flush=True)


def gate_cache_allowed() -> bool:
    """The on-disk gate cache is trusted only when candidates cannot write to it or read its secret."""
    if not _SANDBOX["checked"]:
        from zar_ub.sandbox import sandbox_available
        _note_sandbox(sandbox_available(), "probe")
    return _SANDBOX["ok"]


def _run_candidate(program_path: str, cases: List[list]) -> dict:
    """Run run_candidate.py in a temp dir holding a COPY of the cache tables (the candidate can
    neither read the live tables' path nor tamper with them); no network; 90 s."""
    tmp = tempfile.mkdtemp(prefix="zar_ub_eval_")
    try:
        tmp_cache = os.path.join(tmp, "cache")
        os.makedirs(tmp_cache, exist_ok=True)
        for p in glob.glob(os.path.join(CACHE_DIR, "*.json")):
            try:
                shutil.copy(p, tmp_cache)
            except OSError:
                pass
        env = {k: v for k, v in os.environ.items()
               if not any(x in k.upper() for x in ("KEY", "TOKEN", "SECRET"))}
        env.update({"ZAR_UB_CACHE_DIR": tmp_cache, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
                    "ZAR_UB_NO_LLM": "1"})
        req = json.dumps({"program": os.path.abspath(program_path), "cases": cases, "timeout": int(CANDIDATE_TIMEOUT)})
        # OS-level sandbox (docs/build/ATTACKS.md): the candidate may write only inside `tmp`,
        # may not read the gate-cache secret, and has no network.
        from zar_ub.sandbox import sandboxed_command
        from zar_ub.lean_gate import GATE_SECRET_PATH
        cmd, sandboxed, backend = sandboxed_command([PY, os.path.join(_HERE, "run_candidate.py")],
                                                    allow_write=[tmp], deny_read=[GATE_SECRET_PATH])
        _note_sandbox(sandboxed, backend)
        try:
            proc = subprocess.run(cmd, input=req, capture_output=True,
                                  text=True, timeout=CANDIDATE_TIMEOUT + 10, cwd=tmp, env=env)
            if not proc.stdout.strip():
                return {"lean_source": "", "schema_data": {}, "notes": "", "mask": [],
                        "error": f"runner produced no output (rc={proc.returncode}): {proc.stderr[-2000:]}"}
            return json.loads(proc.stdout)
        except subprocess.TimeoutExpired:
            return {"lean_source": "", "schema_data": {}, "notes": "", "mask": [],
                    "error": f"python kill timed out after {CANDIDATE_TIMEOUT:.0f}s"}
        except Exception as e:  # noqa: BLE001
            return {"lean_source": "", "schema_data": {}, "notes": "", "mask": [], "error": f"runner failure: {e}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _stage1_data(program_path: str) -> dict:
    sha = _program_sha(program_path)
    if sha in _MEMO_S1:
        return _MEMO_S1[sha]
    t0 = time.time()
    suite = _suite()
    flat, index = [], []
    for kind in reward.ALL_KINDS:
        for inst, tab in suite.get(kind, []):
            for pos, rec in enumerate(tab.records):
                flat.append([inst.m, inst.n, inst.s, inst.t, inst.w, list(rec.rows), list(rec.cols)])
                index.append((inst.tag, pos))
    run = _run_candidate(program_path, flat)
    data = {"sha": sha, "lean_source": str(run.get("lean_source") or ""), "schema_data": run.get("schema_data") or {},
            "notes": str(run.get("notes") or ""), "python_error": run.get("error"), "py_masks": {},
            "forbidden": [], "schema_error": None, "schema_terms": None}
    mask = run.get("mask") or []
    if not data["python_error"] and len(mask) != len(flat):
        data["python_error"] = f"python mask length {len(mask)} != cases {len(flat)}"
    if not data["python_error"]:
        by_tag: Dict[str, List[bool]] = {}
        for (tag, pos), v in zip(index, mask):
            by_tag.setdefault(tag, []).append(bool(v))
        data["py_masks"] = by_tag
    if data["lean_source"].strip():
        data["forbidden"] = list(static_scan(data["lean_source"]))
    # SCHEMA_DATA: validated against the declarative registry when it exists; ignored otherwise
    sd = data["schema_data"]
    if lemmas is not None and isinstance(sd, dict) and any(sd.get(k) for k in sd):
        errs, terms = [], []
        for kind, inst, tab in _scored(suite):
            try:
                ok, e = lemmas.validate(sd, inst)
                if not ok:
                    errs.append(f"[{inst.tag}] " + "; ".join(map(str, e)))
            except Exception as ex:  # noqa: BLE001
                errs.append(f"[{inst.tag}] validator crashed: {ex}")
        if errs:
            data["schema_error"] = "\n".join(errs[:10])
        else:
            for k, (kind, inst, tab) in enumerate(_scored(suite)):
                tname = "ZarPrune.Cand.target" + ("" if k == 0 else str(k))
                try:
                    terms.append(list(lemmas.render_terms(sd, inst, tname)))
                except Exception as ex:  # noqa: BLE001
                    errs.append(f"[{inst.tag}] render_terms crashed: {ex}")
            data["schema_terms"] = terms if not errs else None
            if errs:
                data["schema_error"] = "\n".join(errs[:10])
            # the schema's Python mirror joins K^P (battery + empirical band see the same kill)
            if not errs and hasattr(lemmas, "mirror_kill") and data["py_masks"]:
                try:
                    for kind, inst, tab in _scored(suite):
                        pm = data["py_masks"].get(inst.tag)
                        if pm:
                            for i, r in enumerate(tab.records):
                                if not pm[i] and lemmas.mirror_kill(sd, inst, r.rows, r.cols):
                                    pm[i] = True
                except Exception as ex:  # noqa: BLE001
                    data["schema_error"] = f"mirror_kill crashed: {ex}"
    elif isinstance(sd, dict) and lemmas is None and any(sd.get(k) for k in sd):
        data["schema_note"] = "SCHEMA_DATA ignored: zar_ub.lemmas is not available in this build"
    data["seconds"] = time.time() - t0
    return _memo_put(_MEMO_S1, sha, data)


# --------------------------------------------------------------------------------------
# stage 2: one Lean process for the whole suite
# --------------------------------------------------------------------------------------
def _gate_data(program_path: str, s1: dict) -> dict:
    sha = s1["sha"]
    if sha in _MEMO_GATE:
        return _MEMO_GATE[sha]
    t0 = time.time()
    suite = _suite()
    lib = _library_masks(suite)
    # E27: Lean evaluates the candidate only on the cases the reward can see -- the library survivors
    # S_I (every one, not the CRN sample) and any witnessed case (PIPELINE_BUG check).  A kill of a case
    # the proved library already kills earns nothing, so gating it only cost time (suite v3: ~19k cases,
    # ~9k of them survivors).  Masks are expanded back to full length with False elsewhere.
    positions = {inst.tag: _gate_positions(tab, lib.get(inst.tag)) for _, inst, tab in _scored(suite)}
    scored = [(kind, inst, tab) for kind, inst, tab in _scored(suite) if positions[inst.tag]]
    inputs = [(inst, [(tab.records[i].rows, tab.records[i].cols) for i in positions[inst.tag]])
              for _, inst, tab in scored]
    schema_terms = s1.get("schema_terms")
    if schema_terms is not None:
        # re-render against the gate's numbering: targetK is the index in the FILTERED `scored` list
        # (stage 1 validated the data on every instance, so this cannot introduce a new failure mode;
        # the old tag remap kept the unfiltered index inside the term text: E-schemas TODO 2)
        sd = s1.get("schema_data") or {}
        try:
            schema_terms = [list(lemmas.render_terms(sd, inst, "ZarPrune.Cand.target" + ("" if k == 0 else str(k))))
                            for k, (_, inst, _) in enumerate(scored)]
        except Exception as ex:  # noqa: BLE001
            print(f"[evaluator] WARNING: render_terms failed in stage 2: {ex}", file=sys.stderr)
            st_by_tag = {inst.tag: t for (_, inst, _), t in zip(_scored(suite), schema_terms)}
            schema_terms = [st_by_tag.get(inst.tag, []) for _, inst, _ in scored]
    # conditional prunes (candidateF): the facts each scored instance is granted
    facts = [_facts_for(inst, tab) for _, inst, tab in scored]
    # witnessed battery cases: Lean must never kill one (PIPELINE_BUG)
    witness_inputs = []
    for inst, tab in suite.get("battery", []):
        pos = [i for i, r in enumerate(tab.records) if reward.is_witnessed(r)]
        if pos:
            witness_inputs.append((inst, tab, pos))
    all_inputs = inputs + [(inst, [(tab.records[i].rows, tab.records[i].cols) for i in pos])
                           for inst, tab, pos in witness_inputs]
    if schema_terms is not None:
        schema_terms = schema_terms + [[] for _ in witness_inputs]
    facts = facts + [[] for _ in witness_inputs]  # witnessed cases: no hypotheses, ever
    src = s1["lean_source"]
    out = {"sha": sha, "lean_masks": {}, "schema_masks": {}, "ladders": {}, "ladder": 1, "lean_partial": 0.0,
           "lean_ok": 0.0, "gate_info": {"errors": [], "holes": [], "n_holes": 0, "n_holes_filled": 0,
                                         "gate_seconds": 0.0, "gate_cache_hit": False, "filled_source": None,
                                         "cond": {}, "n_facts": {}}}
    if not src.strip() or not all_inputs:
        out["gate_info"]["errors"] = ["LEAN_SOURCE is empty: nothing to verify"]
        out["seconds"] = time.time() - t0
        return _memo_put(_MEMO_GATE, sha, out)
    results = _gate(all_inputs, src, schema_terms, timeout=GATE_TIMEOUT, tag="suite",
                    facts=facts if any(facts) else None)
    n_scored = len(inputs)
    ladders, typed = [], 0
    for k, (g, (kind, inst, tab)) in enumerate(zip(results[:n_scored], scored)):
        lad = reward.ladder_of(g)
        ladders.append(lad)
        out["ladders"][inst.tag] = lad
        if getattr(g, "typed_ok", False):
            typed += 1
        pos = positions[inst.tag]
        if lad == 5 and getattr(g, "kill_mask", None) is not None and len(g.kill_mask) == len(pos):
            out["lean_masks"][inst.tag] = _expand(g.kill_mask, pos, len(tab.records))
        sm = getattr(g, "schema_mask", None)
        if sm is not None and len(sm) == len(pos):
            out["schema_masks"][inst.tag] = _expand(sm, pos, len(tab.records))
        if getattr(g, "cond_name", None) is not None:
            out["gate_info"]["cond"][inst.tag] = g.cond_name
        out["gate_info"]["n_facts"][inst.tag] = int(getattr(g, "n_facts", 0) or 0)
    for g, (inst, tab, pos) in zip(results[n_scored:], witness_inputs):
        km = getattr(g, "kill_mask", None)
        if reward.ladder_of(g) == 5 and km is not None and len(km) == len(pos):
            full = [False] * len(tab.records)
            for i, v in zip(pos, km):
                full[i] = bool(v)
            out["lean_masks"][inst.tag] = full
    for _, inst, tab in _scored(suite):  # nothing to verify on this table: an all-False (sound) mask
        if not positions[inst.tag] and tab.records and ladders and reward.combine_ladders(ladders) == 5:
            out["lean_masks"].setdefault(inst.tag, [False] * len(tab.records))
    g0 = results[0] if results else None
    ladder = reward.combine_ladders(ladders)
    out["ladder"] = ladder
    out["lean_ok"] = typed / n_scored if n_scored else 0.0
    if g0 is not None and hasattr(g0, "ladder") and hasattr(g0, "lean_partial"):
        try:
            out["lean_partial"] = float(g0.lean_partial)
        except Exception:  # noqa: BLE001
            out["lean_partial"] = reward.lean_partial_of(g0, ladder)
    else:
        out["lean_partial"] = reward.lean_partial_of(g0, ladder)
    errs: List[str] = []
    seen = set()
    for g, (kind, inst, tab) in zip(results[:n_scored], scored):
        for e in (getattr(g, "errors", None) or []):
            if e not in seen:
                seen.add(e)
                errs.append(f"[{inst.tag}] {e}")
    gi = out["gate_info"]
    gi["errors"] = errs[:12]
    gi["holes"] = list(getattr(g0, "holes", None) or [])
    gi["n_holes"] = int(getattr(g0, "n_holes", 0) or 0)
    gi["n_holes_filled"] = int(getattr(g0, "n_holes_filled", 0) or 0)
    gi["gate_seconds"] = max((float(getattr(g, "seconds", 0.0) or 0.0) for g in results), default=0.0)
    gi["gate_cache_hit"] = bool(getattr(g0, "cache_hit", False))
    gi["filled_source"] = getattr(g0, "filled_source", None)
    gi["axioms"] = sorted(set(a for g in results for a in (getattr(g, "axioms", None) or [])))
    out["seconds"] = time.time() - t0
    return _memo_put(_MEMO_GATE, sha, out)


def _gate_positions(tab, lib_mask) -> List[int]:
    """Record indices the Lean gate evaluates for one scored table: probed cases the proved library
    does not kill (S_I without the CRN sample restriction) plus every witnessed case."""
    base = getattr(tab, "baseline_lean_mask", None) or lib_mask
    out = []
    for i, r in enumerate(tab.records):
        if getattr(r, "probe", None) is None:
            continue
        if reward.is_witnessed(r) or not (base and i < len(base) and base[i]):
            out.append(i)
    return out


def _expand(mask, pos: List[int], n: int) -> List[bool]:
    full = [False] * n
    for i, v in zip(pos, mask):
        full[i] = bool(v)
    return full


# --------------------------------------------------------------------------------------
# scoring glue
# --------------------------------------------------------------------------------------
def _result(metrics: dict, artifacts: Dict[str, str]):
    if EvaluationResult is not None:
        return EvaluationResult(metrics=metrics, artifacts=artifacts)
    return metrics


def _sentinel_result(t0: float):
    note = ""
    try:
        note = open(SENTINEL).read()[-2000:]
    except OSError:
        pass
    metrics = {k: 0.0 for k in reward.METRIC_KEYS}
    metrics.update({"agreement": 1.0, "kill_novelty": 1.0, "table_hash": "", "suite_version": _suite_version(),
                    "pipeline_bug": 1.0, "eval_seconds": time.time() - t0})
    return _result(metrics, {"PIPELINE_BUG": f"run halted: {SENTINEL} exists (remove it after fixing the bug)\n{note}"})


def _write_sentinel(text: str) -> None:
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(SENTINEL, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{text}\n")
    except OSError:
        pass


def _score(program_path: str, stage: int, t0: float):
    suite = _suite()
    s1 = _stage1_data(program_path)
    common = dict(py_masks=s1["py_masks"], library_masks=_library_masks(suite), lean_source=s1["lean_source"],
                  python_error=s1["python_error"], schema_error=s1["schema_error"], forbidden=s1["forbidden"],
                  ledger_path=LEDGER, table_hash=_table_hash(suite), suite_version=_suite_version())
    stage1_hard_zero = bool(s1["python_error"] or s1["schema_error"] or s1["forbidden"])
    if stage == 1 or stage1_hard_zero:
        metrics, art = reward.score(suite, stage=1, **common)
        if stage1_hard_zero or metrics["combined_score"] == 0.0:
            metrics["combined_score"] = 0.0
    else:
        g = _gate_data(program_path, s1)
        metrics, art = reward.score(suite, stage=stage, lean_masks=g["lean_masks"], schema_masks=g["schema_masks"],
                                    ladder=g["ladder"], lean_partial=g["lean_partial"], lean_ok=g["lean_ok"],
                                    gate_info=g["gate_info"], **common)
        if g["gate_info"].get("axioms"):
            art["lean_axioms"] = ", ".join(g["gate_info"]["axioms"])
        if g["gate_info"].get("cond"):
            # the conditional entry point: which CondPrune was discharged where, and with how many facts
            art["lean_cond"] = "\n".join(f"{tag}: {name} ({g['gate_info']['n_facts'].get(tag, 0)} facts granted)"
                                         for tag, name in sorted(g["gate_info"]["cond"].items()))[:4000]
        if metrics.get("pipeline_bug"):
            _write_sentinel(art.get("PIPELINE_BUG", "pipeline bug") + f"\nprogram: {program_path}\nsha: {s1['sha']}")
            metrics["combined_score"] = 0.0
    if s1.get("schema_note"):
        art["schema_note"] = s1["schema_note"]
    if s1.get("forbidden"):
        art["lean_errors"] = "forbidden construct(s): " + ", ".join(s1["forbidden"]) + \
            ("\n" + art["lean_errors"] if art.get("lean_errors") else "")
    art["table_hash"] = common["table_hash"]
    art["suite_version"] = common["suite_version"]
    metrics["eval_seconds"] = time.time() - t0
    return metrics, art


def _stage(program_path: str, stage: int):
    t0 = time.time()
    if os.path.exists(SENTINEL):
        return _sentinel_result(t0)
    metrics, art = _score(program_path, stage, t0)
    return _result(metrics, art)


def evaluate_stage1(program_path: str):
    """Python masks + battery + scan + schema validation; 0 if hard-zero else 0.01 + 0.08·E."""
    return _stage(program_path, 1)


def evaluate_stage2(program_path: str):
    """Lean gate on the whole suite; score without the target term."""
    return _stage(program_path, 2)


def evaluate_stage3(program_path: str):
    """Full §5.3 score including TARGET."""
    return _stage(program_path, 3)


def evaluate(program_path: str):
    """Non-cascade entry point: stage 1, then the full score unless stage 1 is hard-zero."""
    t0 = time.time()
    if os.path.exists(SENTINEL):
        return _sentinel_result(t0)
    m1, a1 = _score(program_path, 1, t0)
    if m1["combined_score"] == 0.0:
        return _result(m1, a1)
    metrics, art = _score(program_path, 3, t0)
    return _result(metrics, art)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stage_opt = [a for a in sys.argv[1:] if a.startswith("--stage")]
    path = args[0] if args else os.path.join(_HERE, "initial_program.py")
    if stage_opt:
        st = int(stage_opt[0].split("=")[1]) if "=" in stage_opt[0] else int(args[1])
        r = {1: evaluate_stage1, 2: evaluate_stage2, 3: evaluate_stage3}[st](path)
    else:
        r = evaluate(path)
    m = r.metrics if hasattr(r, "metrics") else r
    print(json.dumps(m, indent=1))
    if hasattr(r, "artifacts"):
        for k, v in r.artifacts.items():
            print(f"--- artifact {k} ---\n{v}")

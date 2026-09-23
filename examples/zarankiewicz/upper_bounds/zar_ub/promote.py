"""Promotion of a gate-accepted prune into the trusted library (design §4.4).

    promote(lean_source, name, meta)          -> PromoteResult
    promote_program(path, name, meta)         -> PromoteResult   (runs the Python battery too)
    python -m zar_ub promote FILE.py [--name N] [--iteration K]

A candidate becomes part of the library only after ALL of the following succeed, in
order (any failure is quarantined in cache/ledger/quarantine.jsonl and nothing is written):

  P1  static scan with sketch=False: no `sorry` at all, no forbidden construct;
  P2  Python battery (program promotion only): kill() must not fire on any witnessed case;
  P3  the gate (zar_ub.lean_gate.run_gate_multi, use_cache=False, sketch=False, own scratch
      files `lean/Candidates/cand_promote_<sha>_*.lean`, removed before the run) on the WHOLE
      suite (suite.load_suite: TRAIN ∪ TARGET ∪ GEN tables) plus every witnessed battery case:
      ladder 5 on every instance, axioms ⊆ {propext, Quot.sound, Classical.choice}, the Lean
      mask kills no witnessed case (a proved kill of a witnessed case writes cache/PIPELINE_BUG);
  P4  the delivered module text (lean/ZarPrune/Evolved/E_<sha>.lean: `import ZarPrune.Cond`,
      `namespace ZarPrune.Evolved.E_<sha>`, the candidate source verbatim) elaborates on its own
      with `lake env lean` and `#print axioms ZarPrune.Evolved.E_<sha>.candidate` ⊆ allowed;
  P5  replay: the module is compiled to an .olean in a temporary root (`lake env lean --root R
      -o R/ZarEvolvedCheck/E_<sha>.olean`, nothing is written inside the project) and re-checked
      by the toolchain's `leanchecker` (its own kernel replay of every declaration); when no
      leanchecker binary exists the P4 elaboration is recorded as the replay;
  P6  the regenerated library (`lean/ZarPrune/Evolved.lean` = every accepted module OR-ed with
      `Prune.ofList`) is elaborated once more as ONE inlined scratch file with
      `#print axioms ZarPrune.evolved`; failure rolls the module file back.

Then the ledger line is appended to cache/ledger/accepted_prunes.jsonl:
  {sha, name, lean_name, per_instance_kill_signature, kill_signature, masks, masks_all,
   iteration, axioms, timestamp, ...}
`masks[tag]` = indices of the table's scored survivors (S_I, the cases the proved library
`baseline ∪ counting` leaves alive) that the prune kills — the same set reward.kill_novelty
compares — and `masks_all[tag]` = every index the prune kills.  With ≥ NOVELTY_SWITCH entries
the novelty axis (config_novelty.yaml) should be switched on; `promote` prints the switch.

`sha` = sha1 of the normalised candidate source (12 hex chars); the Lean name is
`ZarPrune.Evolved.E_<sha>.candidate`.  Promotion is idempotent: a sha already in the ledger is
reported as `already promoted` and nothing is rewritten.

The library masks used by the closure daemon and the kill audit come from `library_masks()`:
`baseline ∪ counting` (the table's stored baseline_lean_mask, else the gate on LIBRARY_LEAN)
OR-ed with the Lean mask of every accepted entry.  Until the integrator imports
`ZarPrune.Evolved` from `lean/ZarPrune.lean` and rebuilds, the entries are gated one by one
(their sources are verbatim, so each mask is the Lean-evaluated kill of that entry); once the
built `Evolved.olean` exists and is imported, ONE gate run on
`Prune.or (baseline P) (Prune.or (counting P) (evolved P))` is used instead.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from .known import Instance
from . import lean_gate
from .lean_gate import run_gate_multi, static_scan, run_lean, LEAN_DIR, ALLOWED_AXIOMS
from . import reward
from .casetable import LIBRARY_LEAN

_HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(_HERE)
EVOLVED_DIR = os.path.join(LEAN_DIR, "ZarPrune", "Evolved")
EVOLVED_LEAN = os.path.join(LEAN_DIR, "ZarPrune", "Evolved.lean")
LEDGER_DIR = os.path.join(UB, "cache", "ledger")
LEDGER_PATH = os.path.join(LEDGER_DIR, "accepted_prunes.jsonl")
QUARANTINE_PATH = os.path.join(LEDGER_DIR, "quarantine.jsonl")
SENTINEL = os.path.join(UB, "cache", "PIPELINE_BUG")
NOVELTY_SWITCH = 5
#: what an evolved module imports: `Cond` brings Counting, Prunes, Prune, Basic, Sum.  Never the
#: root `ZarPrune` (the integrator adds `import ZarPrune.Evolved` there: a cycle otherwise).
MODULE_IMPORTS = ("ZarPrune.Cond",)
EVOLVED_IMPORTS = ("ZarPrune.Prune",)
CANDIDATE_TIMEOUT = 90.0
PY = sys.executable
_SHA_RE = re.compile(r"^[0-9a-f]{12}$")


# --------------------------------------------------------------------------------------
# source normalisation, module text, library text
# --------------------------------------------------------------------------------------
def normalize_source(src: str) -> str:
    src = (src or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = src.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(l.rstrip() for l in lines) + "\n"


def source_sha(src: str) -> str:
    return hashlib.sha1(normalize_source(src).encode("utf-8")).hexdigest()[:12]


def lean_name(sha: str) -> str:
    return f"ZarPrune.Evolved.E_{sha}.candidate"


def _doc_safe(s: str) -> str:
    return str(s).replace("-/", "- /").replace("/-", "/ -").replace("\n", " ")


def module_body(sha: str, src: str, name: str = "", meta: Optional[dict] = None) -> str:
    """The module without its import lines (used verbatim in the inlined library check)."""
    meta = meta or {}
    origin = {k: meta[k] for k in ("program_id", "iteration", "checkpoint", "program_path", "run") if k in meta}
    lines = [
        "/-!",
        f"Evolved prune `E_{sha}` — promoted {time.strftime('%Y-%m-%d %H:%M')} by zar_ub/promote.py (design §4.4).",
        f"name: {_doc_safe(name)}",
    ]
    if origin:
        lines.append("origin: " + _doc_safe(json.dumps(origin, sort_keys=True)))
    lines += [
        "The candidate source between the namespace lines is VERBATIM the text the gate accepted",
        "(sha1 of the normalised text = the module name).  Do not edit; re-promote instead.",
        "-/",
        "set_option autoImplicit false",
        "",
        f"namespace ZarPrune.Evolved.E_{sha}",
        "",
        normalize_source(src).rstrip("\n"),
        "",
        f"end ZarPrune.Evolved.E_{sha}",
        "",
    ]
    return "\n".join(lines)


def module_text(sha: str, src: str, name: str = "", meta: Optional[dict] = None) -> str:
    return "\n".join(f"import {m}" for m in MODULE_IMPORTS) + "\n\n" + module_body(sha, src, name, meta)


def evolved_text(shas: Sequence[str]) -> str:
    shas = list(shas)
    lines = [f"import {m}" for m in EVOLVED_IMPORTS] + [f"import ZarPrune.Evolved.E_{s}" for s in shas]
    lines += [
        "",
        "/-!",
        "The promoted library (GENERATED by zar_ub/promote.py from cache/ledger/accepted_prunes.jsonl;",
        "do not edit).  `evolved P` is the OR of every accepted evolved prune; each one lives in its own",
        "module lean/ZarPrune/Evolved/E_<sha>.lean with the candidate source verbatim.",
        f"Entries: {len(shas)}.",
        "-/",
        "set_option autoImplicit false",
        "",
        "namespace ZarPrune",
        "",
        "/-- Every accepted evolved prune, OR-ed (`Prune.never` while the ledger is empty). -/",
        "def evolved (P : Params) : Prune P :=",
    ]
    if shas:
        lines.append("  Prune.ofList P [" + ", ".join(f"Evolved.E_{s}.candidate P" for s in shas) + "]")
    else:
        lines.append("  Prune.never P")
    lines += ["", "end ZarPrune", ""]
    return "\n".join(lines)


def _extract_source(module: str, sha: str) -> Optional[str]:
    head = f"namespace ZarPrune.Evolved.E_{sha}\n"
    tail = f"\nend ZarPrune.Evolved.E_{sha}"
    i = module.find(head)
    j = module.rfind(tail)
    if i < 0 or j < 0 or j <= i:
        return None
    return normalize_source(module[i + len(head):j])


# --------------------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------------------
def load_ledger(path: str = LEDGER_PATH) -> List[dict]:
    return reward.load_ledger(path)


def ledger_shas(path: str = LEDGER_PATH) -> List[str]:
    out = []
    for e in load_ledger(path):
        s = str(e.get("sha", ""))
        if _SHA_RE.match(s) and s not in out:
            out.append(s)
    return out


def _append_jsonl(path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def novelty_switch(path: str = LEDGER_PATH) -> Tuple[bool, int]:
    n = len(ledger_shas(path))
    return n >= NOVELTY_SWITCH, n


def entry_source(sha: str, evolved_dir: str = EVOLVED_DIR) -> Optional[str]:
    p = os.path.join(evolved_dir, f"E_{sha}.lean")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return _extract_source(fh.read(), sha)


def regenerate_evolved(ledger_path: str = LEDGER_PATH, evolved_dir: str = EVOLVED_DIR,
                       evolved_lean: str = EVOLVED_LEAN, shas: Optional[Sequence[str]] = None) -> Tuple[str, List[str]]:
    """Rewrite Evolved.lean from the ledger (only entries whose module file exists)."""
    if shas is None:
        shas = ledger_shas(ledger_path)
    have = [s for s in shas if os.path.exists(os.path.join(evolved_dir, f"E_{s}.lean"))]
    os.makedirs(os.path.dirname(evolved_lean), exist_ok=True)
    tmp = evolved_lean + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(evolved_text(have))
    os.replace(tmp, evolved_lean)
    lean_gate._LIB_HASH.clear()  # the library text changed: gate cache keys must see it
    return evolved_lean, have


def library_check_text(shas: Sequence[str], evolved_dir: str = EVOLVED_DIR) -> str:
    """ONE file that inlines every accepted module and the `evolved` definition (the delivered
    Evolved.lean imports modules that have no .olean until the integrator builds)."""
    parts = [f"import {m}" for m in MODULE_IMPORTS] + ["set_option autoImplicit false", ""]
    for s in shas:
        with open(os.path.join(evolved_dir, f"E_{s}.lean"), encoding="utf-8") as fh:
            body = [l for l in fh.read().split("\n") if not l.startswith("import ")]
        parts.append("\n".join(body))
    ev = [l for l in evolved_text(shas).split("\n") if not l.startswith("import ")]
    parts.append("\n".join(ev))
    parts.append("#print axioms ZarPrune.evolved")
    return "\n".join(parts) + "\n"


# --------------------------------------------------------------------------------------
# suite / battery helpers
# --------------------------------------------------------------------------------------
def _load_suite(targets=None):
    if UB not in sys.path:
        sys.path.insert(0, UB)
    import suite as suite_mod  # noqa: E402

    kw = {} if targets is None else {"targets": targets}
    s = suite_mod.load_suite(verbose=False, **kw)
    for k in reward.ALL_KINDS:
        s.setdefault(k, [])
    return s


def scored_tables(suite) -> List[Tuple[str, Instance, object]]:
    return [(kind, inst, tab) for kind in reward.SCORED_KINDS for (inst, tab) in suite.get(kind, []) if tab.records]


def witnessed_cases(suite) -> List[Tuple[Instance, object, List[int]]]:
    out = []
    for inst, tab in suite.get("battery", []):
        pos = [i for i, r in enumerate(tab.records) if reward.is_witnessed(r)]
        if pos:
            out.append((inst, tab, pos))
    return out


def run_candidate(program_path: str, cases: List[list], timeout: float = CANDIDATE_TIMEOUT) -> dict:
    """run_candidate.py in a subprocess (no network, empty cache dir, secrets stripped)."""
    tmp = tempfile.mkdtemp(prefix="zar_ub_promote_")
    try:
        env = {k: v for k, v in os.environ.items() if not any(x in k.upper() for x in ("KEY", "TOKEN", "SECRET"))}
        env.update({"ZAR_UB_CACHE_DIR": tmp, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "ZAR_UB_NO_LLM": "1"})
        req = json.dumps({"program": os.path.abspath(program_path), "cases": cases, "timeout": int(timeout)})
        try:
            proc = subprocess.run([PY, os.path.join(UB, "run_candidate.py")], input=req, capture_output=True,
                                  text=True, timeout=timeout + 10, cwd=tmp, env=env)
            if not proc.stdout.strip():
                return {"lean_source": "", "schema_data": {}, "notes": "", "mask": [],
                        "error": f"runner produced no output (rc={proc.returncode}): {proc.stderr[-2000:]}"}
            return json.loads(proc.stdout)
        except subprocess.TimeoutExpired:
            return {"lean_source": "", "schema_data": {}, "notes": "", "mask": [], "error": "python kill timed out"}
        except Exception as e:  # noqa: BLE001
            return {"lean_source": "", "schema_data": {}, "notes": "", "mask": [], "error": f"runner failure: {e}"}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------------------
# replay (leanchecker)
# --------------------------------------------------------------------------------------
_TOOL: Dict[str, Optional[str]] = {}


def find_leanchecker() -> Optional[str]:
    if "leanchecker" in _TOOL:
        return _TOOL["leanchecker"]
    found = None
    try:
        r = subprocess.run(["lake", "env", "which", "leanchecker"], cwd=LEAN_DIR, capture_output=True, text=True, timeout=60)
        p = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if p and os.access(p, os.X_OK):
            found = p
    except Exception:  # noqa: BLE001
        found = None
    if found is None:
        try:
            with open(os.path.join(LEAN_DIR, "lean-toolchain")) as fh:
                tc = fh.read().strip()  # leanprover/lean4:v4.34.0
            name = tc.replace("/", "--").replace(":", "---")
            p = os.path.join(os.path.expanduser("~"), ".elan", "toolchains", name, "bin", "leanchecker")
            if os.access(p, os.X_OK):
                found = p
        except OSError:
            pass
    if found is None:
        found = shutil.which("leanchecker")
    _TOOL["leanchecker"] = found
    return found


def lean_search_path() -> str:
    if "lean_path" not in _TOOL:
        r = subprocess.run(["lake", "env", "printenv", "LEAN_PATH"], cwd=LEAN_DIR, capture_output=True, text=True, timeout=60)
        _TOOL["lean_path"] = r.stdout.strip()
    return _TOOL["lean_path"] or ""


def replay_module(text: str, sha: str, timeout: float = 300.0) -> dict:
    """Compile the module into a TEMPORARY root and replay it with leanchecker.  Nothing is
    written inside lean/ (the .olean lives in the temp root, deleted afterwards)."""
    t0 = time.time()
    root = tempfile.mkdtemp(prefix="zar_ub_replay_")
    modname = f"ZarEvolvedCheck.E_{sha}"  # a root name that cannot shadow ZarPrune's olean tree
    try:
        d = os.path.join(root, "ZarEvolvedCheck")
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, f"E_{sha}.lean")
        olean = os.path.join(d, f"E_{sha}.olean")
        with open(src, "w", encoding="utf-8") as fh:
            fh.write(text)
        try:
            r1 = subprocess.run(["lake", "env", "lean", f"--root={root}", "-o", olean, src], cwd=LEAN_DIR,
                                capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"tool": "leanchecker", "ok": False, "stage": "compile", "error": "timeout", "seconds": time.time() - t0}
        out1 = (r1.stdout or "") + (r1.stderr or "")
        if r1.returncode != 0 or re.search(r"^\S+:\d+:\d+: error", out1, re.M) or not os.path.exists(olean):
            return {"tool": "leanchecker", "ok": False, "stage": "compile", "error": out1[-2000:], "seconds": time.time() - t0}
        lc = find_leanchecker()
        if lc is None:
            return {"tool": "lake env lean", "ok": True, "stage": "replay",
                    "note": "no leanchecker binary found (elan toolchain / lake env); the standalone "
                            "elaboration with `lake env lean` is recorded as the replay",
                    "seconds": time.time() - t0}
        env = os.environ.copy()
        env["LEAN_PATH"] = root + os.pathsep + lean_search_path()
        try:
            r2 = subprocess.run([lc, modname], cwd=LEAN_DIR, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"tool": "leanchecker", "ok": False, "stage": "replay", "error": "timeout", "seconds": time.time() - t0}
        out2 = (r2.stdout or "") + (r2.stderr or "")
        ok = r2.returncode == 0 and "found a problem" not in out2 and "uncaught exception" not in out2
        return {"tool": "leanchecker", "binary": lc, "module": modname, "ok": ok, "stage": "replay",
                "output": out2[-2000:], "seconds": time.time() - t0}
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------------------
@dataclass
class PromoteResult:
    ok: bool = False
    sha: str = ""
    name: str = ""
    lean_name: str = ""
    stage: str = ""              # where it stopped: sentinel | scan | battery | ladder | gate | module | replay | library | done
    reason: str = ""
    already: bool = False        # sha was in the ledger before this call
    ladders: Dict[str, int] = field(default_factory=dict)
    axioms: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    masks: Dict[str, List[int]] = field(default_factory=dict)       # killed scored-survivor indices per table
    masks_all: Dict[str, List[int]] = field(default_factory=dict)   # every killed index per table
    per_instance_kill_signature: Dict[str, str] = field(default_factory=dict)
    kill_signature: str = ""
    new_kills: int = 0           # Σ |masks[tag]| : kills beyond the proved library
    replay: dict = field(default_factory=dict)
    module_path: str = ""
    evolved_path: str = ""
    ledger_entry: dict = field(default_factory=dict)
    n_ledger: int = 0
    novelty_on: bool = False
    seconds: float = 0.0
    gate_seconds: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _quarantine(res: PromoteResult, path: str, extra: Optional[dict] = None) -> None:
    e = {"sha": res.sha, "name": res.name, "stage": res.stage, "reason": res.reason, "errors": res.errors[:10],
         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        e.update(extra)
    try:
        _append_jsonl(path, e)
    except OSError:
        pass


def _write_sentinel(text: str) -> None:
    try:
        os.makedirs(os.path.dirname(SENTINEL), exist_ok=True)
        with open(SENTINEL, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} promote: {text}\n")
    except OSError:
        pass


def _clean_scratch(sha: str) -> int:
    n = 0
    for p in glob.glob(os.path.join(LEAN_DIR, "Candidates", f"cand_promote_{sha}*.lean")):
        try:
            os.remove(p)
            n += 1
        except OSError:
            pass
    return n


def _mask_sig(mask: Sequence[bool]) -> str:
    return hashlib.sha1("".join("1" if v else "0" for v in mask).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# promote
# --------------------------------------------------------------------------------------
def promote(lean_source: str, name: str, meta: Optional[dict] = None, *, program_path: Optional[str] = None,
            timeout: float = 300.0, evolved_dir: str = EVOLVED_DIR, evolved_lean: str = EVOLVED_LEAN,
            ledger_path: str = LEDGER_PATH, quarantine_path: Optional[str] = None, suite=None,
            verbose: bool = True) -> PromoteResult:
    """Design §4.4.  Returns a PromoteResult; `ok` iff the prune is now in the library."""
    t0 = time.time()
    meta = dict(meta or {})
    quarantine_path = quarantine_path or os.path.join(os.path.dirname(ledger_path), "quarantine.jsonl")
    src = normalize_source(lean_source)
    sha = source_sha(src)
    res = PromoteResult(sha=sha, name=name or f"E_{sha}", lean_name=lean_name(sha))

    def log(msg):
        if verbose:
            print(f"[promote {sha}] {msg}", flush=True)

    def refuse(stage: str, reason: str, errors: Optional[List[str]] = None, extra: Optional[dict] = None):
        res.ok, res.stage, res.reason = False, stage, reason
        if errors:
            res.errors.extend(errors)
        res.seconds = time.time() - t0
        _quarantine(res, quarantine_path, extra)
        log(f"REFUSED ({stage}): {reason}")
        return res

    if os.path.exists(SENTINEL):
        return refuse("sentinel", f"pipeline halted: {SENTINEL} exists (remove it after the audit)")
    if not src.strip():
        return refuse("scan", "empty Lean source")
    if sha in ledger_shas(ledger_path):
        res.ok, res.already, res.stage, res.reason = True, True, "done", "already promoted (sha in the ledger)"
        res.module_path = os.path.join(evolved_dir, f"E_{sha}.lean")
        res.evolved_path = evolved_lean
        res.novelty_on, res.n_ledger = novelty_switch(ledger_path)
        res.seconds = time.time() - t0
        log(res.reason)
        return res

    # P1 static scan.  A source with sketch holes (`have h : T := by sorry`) is not promotable as
    # is; it goes through the gate in sketch mode and, when the auto-fill closes every hole, the
    # FILLED copy is what gets promoted (design §4.2 S7); otherwise it is refused by its ladder.
    forbidden = static_scan(src, sketch=True)
    if forbidden:
        return refuse("scan", "forbidden construct in the source", [f"forbidden construct: {x}" for x in forbidden])
    sketch = bool(static_scan(src, sketch=False))
    if not re.search(r"^\s*(?:@\[[^\]]*\]\s*)?def\s+candidate\b", src, re.M):
        return refuse("scan", "the source does not define `candidate`")
    if re.search(r"\bcandidate\s*:\s*Prune\s+target\b", src):
        return refuse("scan", "instance-specific candidate (`Prune target`); only `candidate (P : Params) : Prune P` is promotable")

    suite = suite or _load_suite()
    scored = scored_tables(suite)
    wit = witnessed_cases(suite)
    if not scored:
        return refuse("gate", "no scored suite tables are cached (run `python -m zar_ub table ... --baseline`)")

    # P2 Python battery (program promotion)
    if program_path:
        flat = [[inst.m, inst.n, inst.s, inst.t, inst.w, list(tab.records[i].rows), list(tab.records[i].cols)]
                for inst, tab, pos in wit for i in pos]
        run = run_candidate(program_path, flat)
        if run.get("error"):
            return refuse("battery", "the candidate program failed to run", [str(run["error"])[-1500:]])
        pm = run.get("mask") or []
        if len(pm) != len(flat):
            return refuse("battery", f"python mask length {len(pm)} != witnessed cases {len(flat)}")
        bad = [f"({c[0]},{c[1]};{c[2]},{c[3]}) w={c[4]} rows={c[5]} cols={c[6]}" for c, v in zip(flat, pm) if v]
        if bad:
            return refuse("battery", f"python kill() fires on {len(bad)} witnessed (realizable) case(s): unsound",
                          bad[:10], {"counterexamples": bad[:20]})
        log(f"python battery: {len(flat)} witnessed cases, none killed")
        prog_src = normalize_source(str(run.get("lean_source") or ""))
        if prog_src != src and "lean_source_override" not in meta:
            log("note: LEAN_SOURCE of the program differs from the promoted source (override in use)")

    # P3 the gate, no cache, own scratch
    inputs = [(inst, [(r.rows, r.cols) for r in tab.records]) for _, inst, tab in scored]
    inputs += [(inst, [(tab.records[i].rows, tab.records[i].cols) for i in pos]) for inst, tab, pos in wit]
    _clean_scratch(sha)
    tg = time.time()
    results = run_gate_multi(inputs, src, schema_terms=None, timeout=timeout, tag=f"promote_{sha}",
                             sketch=sketch, keep=True, use_cache=False)
    res.gate_seconds = time.time() - tg
    n_scored = len(scored)
    if sketch:
        filled = getattr(results[0], "filled_source", None) if results else None
        if filled and all(g.ladder == 5 for g in results) and not static_scan(filled, sketch=False):
            log(f"sketch: every hole auto-filled; promoting the filled copy (sha {source_sha(filled)})")
            meta2 = dict(meta, filled_from=sha, lean_source_override=True)
            return promote(filled, name, meta2, program_path=program_path, timeout=timeout, evolved_dir=evolved_dir,
                           evolved_lean=evolved_lean, ledger_path=ledger_path, quarantine_path=quarantine_path,
                           suite=suite, verbose=verbose)
        for g, (kind, inst, tab) in zip(results[:n_scored], scored):
            res.ladders[inst.tag] = int(g.ladder)
        holes = [f"hole {h.get('index')}: {h.get('statement', '')[:120]} (filled_by={h.get('filled_by')})"
                 for h in (getattr(results[0], "holes", None) or [])]
        return refuse("ladder", f"sketch holes remain (ladder {reward.combine_ladders([g.ladder for g in results[:n_scored]])}); "
                      "a sorry is never promoted", holes[:10] + (results[0].errors[:5] if results else []))
    errs = []
    for g, (kind, inst, tab) in zip(results[:n_scored], scored):
        res.ladders[inst.tag] = int(g.ladder)
        if g.ladder != 5 or not g.ok or g.kill_mask is None or len(g.kill_mask) != len(tab.records):
            errs.append(f"[{inst.tag}] ladder {g.ladder}: " + "; ".join(g.errors[:3]))
    for g, (inst, tab, pos) in zip(results[n_scored:], wit):
        res.ladders[inst.tag + "(witness)"] = int(g.ladder)
        if g.ladder != 5 or not g.ok or g.kill_mask is None or len(g.kill_mask) != len(pos):
            errs.append(f"[{inst.tag} witnessed] ladder {g.ladder}: " + "; ".join(g.errors[:3]))
    axioms = sorted(set(a for g in results for a in (g.axioms or [])))
    res.axioms = axioms
    if errs:
        return refuse("ladder", "the gate did not reach L5 on every suite instance", errs)
    if not set(axioms) <= ALLOWED_AXIOMS:
        return refuse("gate", f"axioms outside the allowed set: {sorted(set(axioms) - ALLOWED_AXIOMS)}")
    killed_wit = []
    for g, (inst, tab, pos) in zip(results[n_scored:], wit):
        for i, v in zip(pos, g.kill_mask):
            if v:
                killed_wit.append(f"{inst.tag} rows={tab.records[i].rows} cols={tab.records[i].cols}")
    if killed_wit:
        _write_sentinel(f"Lean-proven kill of a SAT-witnessed case by {sha} ({name}): " + "; ".join(killed_wit[:5]))
        return refuse("battery", "PIPELINE_BUG: the Lean-evaluated kill fires on a witnessed case; sentinel written",
                      killed_wit[:10])
    masks_in_order = []
    for g, (kind, inst, tab) in zip(results[:n_scored], scored):
        lm = [bool(x) for x in g.kill_mask]
        masks_in_order.append(lm)
        S = tab.scored_indices()
        res.masks[inst.tag] = [i for i in S if lm[i]]
        res.masks_all[inst.tag] = [i for i, v in enumerate(lm) if v]
        res.per_instance_kill_signature[inst.tag] = _mask_sig(lm)
    res.kill_signature = reward.kill_signature(masks_in_order)
    res.new_kills = sum(len(v) for v in res.masks.values())
    log(f"gate: L5 on {len(results)} instances in {res.gate_seconds:.1f}s, axioms {axioms}, "
        f"kills beyond the library: {res.new_kills}")

    # P4 the delivered module elaborates on its own
    text = module_text(sha, src, name, meta)
    check = text + f"\n#print axioms {lean_name(sha)}\n"
    check_path = lean_gate._write_candidate(check, f"promote_{sha}_module")
    out, timed_out = run_lean(check_path, timeout)
    if timed_out or re.search(r"^\S+:\d+:\d+: error", out, re.M):
        return refuse("module", "the standalone module does not elaborate", [l for l in out.splitlines() if "error" in l][:10])
    m = re.search(rf"'{re.escape(lean_name(sha))}' (depends on axioms: \[([^\]]*)\]|does not depend on any axioms)", out)
    if not m:
        return refuse("module", "no `#print axioms` line for the module's candidate", [out[-1500:]])
    mod_axioms = sorted(a.strip() for a in (m.group(2) or "").split(",") if a.strip())
    if not set(mod_axioms) <= ALLOWED_AXIOMS:
        return refuse("module", f"module axioms outside the allowed set: {mod_axioms}")
    res.axioms = sorted(set(res.axioms) | set(mod_axioms))
    log(f"module elaborates standalone; axioms {mod_axioms}")

    # P5 replay
    res.replay = replay_module(text, sha, timeout=timeout)
    if not res.replay.get("ok"):
        return refuse("replay", f"{res.replay.get('tool')} replay failed at {res.replay.get('stage')}",
                      [str(res.replay.get("error") or res.replay.get("output") or "")[-1500:]])
    log(f"replay: {res.replay['tool']} ok in {res.replay['seconds']:.1f}s")

    # P6 write the module, regenerate the library, check the whole library once
    os.makedirs(evolved_dir, exist_ok=True)
    module_path = os.path.join(evolved_dir, f"E_{sha}.lean")
    with open(module_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    res.module_path = module_path
    shas = ledger_shas(ledger_path) + [sha]
    lib_check = library_check_text(shas, evolved_dir)
    lib_path = lean_gate._write_candidate(lib_check, f"promote_{sha}_library")
    out, timed_out = run_lean(lib_path, timeout)
    m = re.search(r"'ZarPrune\.evolved' (depends on axioms: \[([^\]]*)\]|does not depend on any axioms)", out)
    lib_axioms = sorted(a.strip() for a in ((m.group(2) or "") if m else "").split(",") if a.strip())
    if timed_out or re.search(r"^\S+:\d+:\d+: error", out, re.M) or not m or not set(lib_axioms) <= ALLOWED_AXIOMS:
        try:
            os.remove(module_path)
        except OSError:
            pass
        regenerate_evolved(ledger_path, evolved_dir, evolved_lean)
        return refuse("library", "the regenerated library does not elaborate / audits badly (module rolled back)",
                      [l for l in out.splitlines() if "error" in l][:10] + [f"library axioms: {lib_axioms}"])
    log(f"library of {len(shas)} entries elaborates; axioms of ZarPrune.evolved: {lib_axioms}")

    # ledger
    prev = load_ledger(ledger_path)
    dup = [str(e.get("name") or e.get("sha")) for e in prev if e.get("kill_signature") == res.kill_signature]
    entry = {
        "sha": sha, "name": res.name, "lean_name": res.lean_name, "module": os.path.relpath(module_path, UB),
        "per_instance_kill_signature": res.per_instance_kill_signature, "kill_signature": res.kill_signature,
        "masks": res.masks, "masks_all": res.masks_all, "new_kills": res.new_kills,
        "iteration": meta.get("iteration"), "program_id": meta.get("program_id"), "checkpoint": meta.get("checkpoint"),
        "axioms": res.axioms, "library_axioms": lib_axioms, "replay": {k: v for k, v in res.replay.items() if k != "output"},
        "gate_seconds": round(res.gate_seconds, 2), "instances": [inst.tag for _, inst, _ in scored],
        "witnessed_cases_checked": sum(len(pos) for _, _, pos in wit), "duplicate_kill_signature_of": dup,
        "library_hash_before": lean_gate.library_hash(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _append_jsonl(ledger_path, entry)
    res.ledger_entry = entry
    res.evolved_path, _ = regenerate_evolved(ledger_path, evolved_dir, evolved_lean)
    res.novelty_on, res.n_ledger = novelty_switch(ledger_path)
    res.ok, res.stage, res.reason = True, "done", "promoted"
    res.seconds = time.time() - t0
    log(f"PROMOTED as {res.lean_name}; Evolved.lean regenerated ({res.n_ledger} entries)")
    print(f"[promote] ledger: {res.n_ledger} accepted prune(s) -> novelty axis "
          f"{'should be switched ON (use config_novelty.yaml)' if res.novelty_on else f'stays off (switch at {NOVELTY_SWITCH})'}",
          flush=True)
    if dup:
        log(f"note: identical kill signature to {dup} (accepted anyway; kill_novelty will see the duplicate)")
    return res


def promote_program(program_path: str, name: Optional[str] = None, meta: Optional[dict] = None,
                    lean_source: Optional[str] = None, **kw) -> PromoteResult:
    """Promote a candidate program (.py): its LEAN_SOURCE (or `lean_source`, e.g. the gate's
    filled copy) goes through `promote`, and its Python kill() must pass the witnessed battery."""
    meta = dict(meta or {})
    meta.setdefault("program_path", os.path.relpath(os.path.abspath(program_path), UB))
    if lean_source is None:
        run = run_candidate(program_path, [])
        if run.get("error"):
            res = PromoteResult(name=name or os.path.basename(program_path), stage="battery",
                                reason="the candidate program failed to run", errors=[str(run["error"])[-1500:]])
            _quarantine(res, kw.get("quarantine_path") or QUARANTINE_PATH)
            return res
        lean_source = str(run.get("lean_source") or "")
    else:
        meta["lean_source_override"] = True
    return promote(lean_source, name or os.path.splitext(os.path.basename(program_path))[0], meta,
                   program_path=program_path, **kw)


# --------------------------------------------------------------------------------------
# the promoted library's masks (closure daemon, kill audit)
# --------------------------------------------------------------------------------------
def evolved_importable() -> bool:
    """True once the integrator imports ZarPrune.Evolved from the root and has built it."""
    root = os.path.join(LEAN_DIR, "ZarPrune.lean")
    olean = os.path.join(LEAN_DIR, ".lake", "build", "lib", "lean", "ZarPrune", "Evolved.olean")
    try:
        with open(root) as fh:
            if "import ZarPrune.Evolved" not in fh.read():
                return False
        if not os.path.exists(olean):
            return False
        newest = max([os.path.getmtime(EVOLVED_LEAN)] + [os.path.getmtime(p) for p in glob.glob(os.path.join(EVOLVED_DIR, "E_*.lean"))])
        return os.path.getmtime(olean) >= newest
    except OSError:
        return False


# Since batch 2 casetable.LIBRARY_LEAN already ORs `evolved P` in; kept as an alias for the daemon.
LIBRARY_EVOLVED_LEAN = LIBRARY_LEAN


def library_masks(tables: Sequence[Tuple[Instance, object]], timeout: float = 600.0, ledger_path: str = LEDGER_PATH,
                  evolved_dir: str = EVOLVED_DIR, use_cache: bool = True, verbose: bool = True) -> Tuple[List[Optional[List[bool]]], dict]:
    """Kill mask of the PROMOTED library (baseline ∪ counting ∪ every accepted entry) per table.
    Returns (masks, info); a mask is None when the base gate failed."""
    inputs = [(inst, [(r.rows, r.cols) for r in tab.records]) for inst, tab in tables]
    info: dict = {"route": "", "entries": [], "skipped": [], "axioms": set()}
    masks: List[Optional[List[bool]]] = []
    shas = ledger_shas(ledger_path)
    if shas and evolved_importable():
        info["route"] = "evolved"
        res = run_gate_multi(inputs, LIBRARY_EVOLVED_LEAN, timeout=timeout, tag="libevolved", sketch=False, use_cache=use_cache)
        for g, (inst, tab) in zip(res, tables):
            ok = g.ok and g.kill_mask is not None and len(g.kill_mask) == len(tab.records)
            masks.append([bool(x) for x in g.kill_mask] if ok else None)
            info["axioms"] |= set(g.axioms or [])
        info["entries"] = shas
        info["axioms"] = sorted(info["axioms"])
        return masks, info
    info["route"] = "per-entry" if shas else "base"
    # base: stored baseline_lean_mask, else the gate on LIBRARY_LEAN
    base: List[Optional[List[bool]]] = []
    need = []
    for k, (inst, tab) in enumerate(tables):
        bm = getattr(tab, "baseline_lean_mask", None)
        if bm is not None and len(bm) == len(tab.records):
            base.append([bool(x) for x in bm])
        else:
            base.append(None)
            need.append((k, inst, tab))
    if need:
        res = run_gate_multi([(inst, [(r.rows, r.cols) for r in tab.records]) for _, inst, tab in need], LIBRARY_LEAN,
                             timeout=timeout, tag="library", sketch=False, use_cache=use_cache)
        for g, (k, inst, tab) in zip(res, need):
            ok = g.ok and g.kill_mask is not None and len(g.kill_mask) == len(tab.records)
            base[k] = [bool(x) for x in g.kill_mask] if ok else None
            info["axioms"] |= set(g.axioms or [])
    masks = [list(b) if b is not None else None for b in base]
    for sha in shas:
        src = entry_source(sha, evolved_dir)
        if src is None:
            info["skipped"].append((sha, "module file missing"))
            continue
        res = run_gate_multi(inputs, src, timeout=timeout, tag=f"lib_{sha}", sketch=False, use_cache=use_cache)
        bad = [inst.tag for g, (inst, tab) in zip(res, tables) if not (g.ok and g.kill_mask is not None and len(g.kill_mask) == len(tab.records))]
        if bad:
            info["skipped"].append((sha, f"gate not L5 on {bad}"))
            if verbose:
                print(f"[library] entry {sha} skipped (not L5 on {bad}); its kills are NOT counted", flush=True)
            continue
        info["entries"].append(sha)
        for k, (g, (inst, tab)) in enumerate(zip(res, tables)):
            info["axioms"] |= set(g.axioms or [])
            if masks[k] is None:
                continue
            masks[k] = [a or bool(b) for a, b in zip(masks[k], g.kill_mask)]
    info["axioms"] = sorted(info["axioms"])
    return masks, info


# --------------------------------------------------------------------------------------
# CLI plugin
# --------------------------------------------------------------------------------------
def _cli_promote(args) -> int:
    lean_source = None
    if args.file.endswith(".lean"):
        with open(args.file, encoding="utf-8") as fh:
            lean_source = fh.read()
        res = promote(lean_source, args.name or os.path.splitext(os.path.basename(args.file))[0],
                      {"iteration": args.iteration, "program_path": os.path.relpath(args.file, UB)}, timeout=args.timeout)
    else:
        res = promote_program(args.file, args.name, {"iteration": args.iteration}, timeout=args.timeout)
    d = res.as_dict()
    d.pop("ledger_entry", None)
    print(json.dumps(d, indent=1, default=str))
    return 0 if res.ok else 1


def _cli_regen(args) -> int:
    path, have = regenerate_evolved()
    print(f"[promote] regenerated {path} with {len(have)} entries: {have}")
    on, n = novelty_switch()
    print(f"[promote] ledger: {n} entries; novelty axis {'ON' if on else 'off'}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("promote", help="promote a gate-accepted candidate into lean/ZarPrune/Evolved (design §4.4)")
    p.add_argument("file", help="candidate program (.py, its LEAN_SOURCE + Python battery) or a .lean source")
    p.add_argument("--name", default=None)
    p.add_argument("--iteration", type=int, default=None)
    p.add_argument("--timeout", type=float, default=300.0)
    p.set_defaults(func=_cli_promote)
    q = sub.add_parser("evolved-regen", help="rewrite lean/ZarPrune/Evolved.lean from the accepted-prune ledger")
    q.set_defaults(func=_cli_regen)

"""Certified refutation of surviving cases: CaDiCaL (binary) with DRAT proof
output -> drat-trim (checks the DRAT and emits LRAT) -> lrat-check (independent
LRAT checker).  A case is *certified* only if lrat-check prints "s VERIFIED".

This is the "refuted" seam of ZarPrune.upper_bound_of_cover: each certified case
is one hypothesis instance; the LRAT file is the artifact a Lean LRAT checker
could re-check.  Certificates and a JSON manifest are written under
cache/certs/<instance tag>/.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Sequence

from .encoding import encode_case
from .known import Instance

_HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(_HERE)
TOOLS = os.path.join(UB, "tools")
CADICAL = os.path.join(TOOLS, "cadical", "build", "cadical")
DRAT_TRIM = os.path.join(TOOLS, "drat-trim", "drat-trim")
LRAT_CHECK = os.path.join(TOOLS, "drat-trim", "lrat-check")
CERT_DIR = os.path.join(UB, "cache", "certs")


def tools_available() -> bool:
    return all(os.access(p, os.X_OK) for p in (CADICAL, DRAT_TRIM, LRAT_CHECK))


@dataclass
class Certificate:
    rows: List[int]
    cols: List[int]
    status: str            # 'certified' | 'sat' | 'timeout' | 'check_failed' | 'error'
    solve_seconds: float
    check_seconds: float
    cnf_sha1: str
    lrat_file: Optional[str]
    lrat_bytes: int
    drat_trim_ok: bool
    lrat_check_ok: bool
    detail: str = ""


def certify_case(inst: Instance, rows: Sequence[int], cols: Sequence[int], time_limit: float = 600.0,
                 keep_lrat: bool = True, workdir: Optional[str] = None) -> Certificate:
    if not tools_available():
        raise RuntimeError("cadical/drat-trim/lrat-check not built: run tools/build_tools.sh")
    workdir = workdir or os.path.join(CERT_DIR, inst.tag)
    os.makedirs(workdir, exist_ok=True)
    key = "r" + "-".join(map(str, rows)) + "_c" + "-".join(map(str, cols))
    cnf = encode_case(inst, rows, cols)
    dimacs = cnf.to_dimacs()
    sha = hashlib.sha1(dimacs.encode()).hexdigest()
    cnf_path = os.path.join(workdir, key + ".cnf")
    drat_path = os.path.join(workdir, key + ".drat")
    lrat_path = os.path.join(workdir, key + ".lrat")
    with open(cnf_path, "w") as f:
        f.write(dimacs)
    t0 = time.time()
    try:
        proc = subprocess.run([CADICAL, "-q", "--binary=false", cnf_path, drat_path],
                              capture_output=True, text=True, timeout=time_limit)
    except subprocess.TimeoutExpired:
        return Certificate(list(rows), list(cols), "timeout", time.time() - t0, 0.0, sha, None, 0, False, False)
    solve_s = time.time() - t0
    if proc.returncode == 10:
        return Certificate(list(rows), list(cols), "sat", solve_s, 0.0, sha, None, 0, False, False, "case is SAT (witness exists)")
    if proc.returncode != 20:
        return Certificate(list(rows), list(cols), "error", solve_s, 0.0, sha, None, 0, False, False, proc.stderr[-500:])
    t1 = time.time()
    dt = subprocess.run([DRAT_TRIM, cnf_path, drat_path, "-L", lrat_path], capture_output=True, text=True, timeout=time_limit)
    # drat-trim prints "s VERIFIED"; lrat-check prints "c VERIFIED" (both exit 0 on success)
    dt_ok = dt.returncode == 0 and "s VERIFIED" in dt.stdout and "FAILED" not in dt.stdout
    lc_ok = False
    if dt_ok and os.path.exists(lrat_path):
        lc = subprocess.run([LRAT_CHECK, cnf_path, lrat_path], capture_output=True, text=True, timeout=time_limit)
        lc_ok = lc.returncode == 0 and "VERIFIED" in lc.stdout and "FAILED" not in lc.stdout and "ERROR" not in lc.stdout
    check_s = time.time() - t1
    lrat_bytes = os.path.getsize(lrat_path) if os.path.exists(lrat_path) else 0
    if os.path.exists(drat_path):
        os.remove(drat_path)
    if not keep_lrat and os.path.exists(lrat_path):
        os.remove(lrat_path)
    status = "certified" if (dt_ok and lc_ok) else "check_failed"
    return Certificate(list(rows), list(cols), status, solve_s, check_s, sha,
                       lrat_path if (keep_lrat and status == "certified") else None, lrat_bytes, dt_ok, lc_ok)


def certify_survivors(inst: Instance, cases: List[Sequence[Sequence[int]]], time_limit: float = 600.0,
                      verbose: bool = True, keep_lrat: bool = True) -> dict:
    """Certify every (rows, cols) in `cases`; write a manifest; return summary."""
    workdir = os.path.join(CERT_DIR, inst.tag)
    os.makedirs(workdir, exist_ok=True)
    certs = []
    for k, (rows, cols) in enumerate(cases):
        c = certify_case(inst, rows, cols, time_limit=time_limit, keep_lrat=keep_lrat, workdir=workdir)
        certs.append(c)
        if verbose:
            print(f"  [{k+1}/{len(cases)}] rows={list(rows)} cols={list(cols)} -> {c.status} "
                  f"solve={c.solve_seconds:.1f}s check={c.check_seconds:.1f}s lrat={c.lrat_bytes}B", flush=True)
    manifest = {"instance": asdict(inst), "n_cases": len(cases),
                "certified": sum(1 for c in certs if c.status == "certified"),
                "sat": sum(1 for c in certs if c.status == "sat"),
                "timeout": sum(1 for c in certs if c.status == "timeout"),
                "failed": sum(1 for c in certs if c.status in ("check_failed", "error")),
                "certs": [asdict(c) for c in certs]}
    with open(os.path.join(workdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    manifest.pop("certs")
    return manifest


# ======================================================================================
# verify-certs (design T-11) and audit-kills (design §4.7 kill audit, T-12)
# ======================================================================================
import glob  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from typing import Tuple  # noqa: E402

from .encoding import has_kst  # noqa: E402

LRAT_CHECK_SRC = os.path.join(TOOLS, "drat-trim", "lrat-check.c")
SENTINEL = os.path.join(UB, "cache", "PIPELINE_BUG")


def build_fresh_lratcheck(out_dir: Optional[str] = None) -> str:
    """Compile tools/drat-trim/lrat-check.c into a fresh binary (independent of the one the
    certificates were produced with).  Returns the binary path."""
    if not os.path.exists(LRAT_CHECK_SRC):
        raise RuntimeError(f"{LRAT_CHECK_SRC} missing (clone drat-trim under tools/)")
    out_dir = out_dir or tempfile.mkdtemp(prefix="zar_ub_lratcheck_")
    binary = os.path.join(out_dir, "lrat-check")
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        raise RuntimeError("no C compiler (cc/gcc/clang) on PATH for --fresh-lratcheck")
    proc = subprocess.run([cc, "-O2", "-o", binary, LRAT_CHECK_SRC], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0 or not os.access(binary, os.X_OK):
        raise RuntimeError("lrat-check build failed: " + proc.stderr[-1000:])
    return binary


def run_lrat_check(cnf_path: str, lrat_path: str, binary: str = LRAT_CHECK, timeout: float = 600.0) -> Tuple[bool, str]:
    try:
        lc = subprocess.run([binary, cnf_path, lrat_path], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "lrat-check timed out"
    out = (lc.stdout or "") + (lc.stderr or "")
    ok = lc.returncode == 0 and "VERIFIED" in out and "NOT VERIFIED" not in out and "FAILED" not in out and "ERROR" not in out
    return ok, out[-600:]


def _sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corrupt_one_byte(lrat_path: str, out_path: str, attempt: int = 0) -> Tuple[int, str]:
    """Copy the LRAT with ONE byte changed: a digit inside the last addition (non-`d`) line, so the
    change lands in a clause id / hint of the final proof steps (attempt k picks the k-th digit
    from the end).  Returns (offset, description)."""
    with open(lrat_path, "rb") as fh:
        data = bytearray(fh.read())
    # find the last line that is an addition step ("<id> <lits> 0 <hints> 0"), not a deletion
    end = len(data)
    while end > 0 and data[end - 1] in b"\r\n":
        end -= 1
    line_end = end
    while True:
        line_start = data.rfind(b"\n", 0, line_end) + 1
        line = bytes(data[line_start:line_end])
        toks = line.split()
        if len(toks) >= 2 and toks[1] != b"d":
            break
        if line_start == 0:
            break
        line_end = line_start - 1
    digits = [i for i in range(line_start, line_end) if 48 <= data[i] <= 57]
    if not digits:
        raise ValueError("no digit to corrupt in the last addition line")
    i = digits[max(0, len(digits) - 1 - attempt) % len(digits)]
    old = data[i]
    data[i] = 57 if old != 57 else 56
    with open(out_path, "wb") as fh:
        fh.write(bytes(data))
    return i, f"byte {i}: {chr(old)!r} -> {chr(data[i])!r} (line {line_start}-{line_end})"


def verify_certs(cert_dir: str = CERT_DIR, fresh: bool = False, corruption_test: bool = False, timeout: float = 600.0,
                 verbose: bool = True) -> dict:
    """Re-verify every certificate of every manifest under `cert_dir` (or the one manifest in it):
    the on-disk CNF must hash to the manifest's cnf_sha1 (re-encoded from rows/cols when the CNF
    file is missing), and lrat-check must print VERIFIED.  With `corruption_test`, the first
    certificate is copied with one byte changed and must be REJECTED (negative control)."""
    binary = LRAT_CHECK
    tmp = None
    if fresh:
        tmp = tempfile.mkdtemp(prefix="zar_ub_lratcheck_")
        binary = build_fresh_lratcheck(tmp)
    if not os.access(binary, os.X_OK):
        raise RuntimeError(f"lrat-check binary missing: {binary}")
    manifests = sorted(glob.glob(os.path.join(cert_dir, "*", "manifest.json")))
    if os.path.exists(os.path.join(cert_dir, "manifest.json")):
        manifests.insert(0, os.path.join(cert_dir, "manifest.json"))
    rep: dict = {"binary": binary, "fresh": fresh, "manifests": [], "n_certs": 0, "verified": 0, "failed": [], "corruption_test": None}
    first_cert = None
    for mp in manifests:
        with open(mp) as fh:
            man = json.load(fh)
        d = os.path.dirname(mp)
        inst = Instance(**man["instance"]) if isinstance(man.get("instance"), dict) else None
        mrep = {"manifest": os.path.relpath(mp, UB), "n": 0, "verified": 0, "failed": 0, "not_certified": 0}
        for c in man.get("certs", []):
            if c.get("status") != "certified":
                mrep["not_certified"] += 1
                continue
            mrep["n"] += 1
            rep["n_certs"] += 1
            key = "r" + "-".join(map(str, c["rows"])) + "_c" + "-".join(map(str, c["cols"]))
            lrat = c.get("lrat_file") or os.path.join(d, key + ".lrat")
            if not os.path.exists(lrat):
                lrat = os.path.join(d, os.path.basename(lrat))
            cnf = os.path.join(d, key + ".cnf")
            why = None
            if not os.path.exists(lrat):
                why = "LRAT file missing"
            else:
                if not os.path.exists(cnf):
                    if inst is None:
                        why = "CNF missing and no instance in the manifest"
                    else:
                        dimacs = encode_case(inst, c["rows"], c["cols"]).to_dimacs()
                        with open(cnf, "w") as fh:
                            fh.write(dimacs)
                if why is None and _sha1_file(cnf) != c.get("cnf_sha1"):
                    why = "CNF sha1 differs from the manifest (encoding changed or file altered)"
                if why is None and c.get("lrat_bytes") and os.path.getsize(lrat) != int(c["lrat_bytes"]):
                    why = f"LRAT size {os.path.getsize(lrat)} != manifest {c['lrat_bytes']}"
                if why is None:
                    ok, out = run_lrat_check(cnf, lrat, binary, timeout)
                    if not ok:
                        why = "lrat-check did not verify: " + out.strip().splitlines()[-1] if out.strip() else "lrat-check did not verify"
            if why is None:
                mrep["verified"] += 1
                rep["verified"] += 1
                if first_cert is None:
                    first_cert = (cnf, lrat)
            else:
                mrep["failed"] += 1
                rep["failed"].append({"manifest": mrep["manifest"], "case": key, "why": why})
            if verbose:
                print(f"  {os.path.basename(d)}/{key}: {'VERIFIED' if why is None else 'FAILED: ' + why}", flush=True)
        rep["manifests"].append(mrep)
    if corruption_test:
        if first_cert is None:
            rep["corruption_test"] = {"ok": False, "why": "no verified certificate to corrupt"}
        else:
            cnf, lrat = first_cert
            ctmp = tempfile.mkdtemp(prefix="zar_ub_corrupt_")
            try:
                accepted = None
                for attempt in range(8):
                    bad = os.path.join(ctmp, "corrupt.lrat")
                    off, desc = corrupt_one_byte(lrat, bad, attempt)
                    ok, out = run_lrat_check(cnf, bad, binary, timeout)
                    if not ok:
                        rep["corruption_test"] = {"ok": True, "lrat": os.path.relpath(lrat, UB), "change": desc, "attempts": attempt + 1,
                                                  "checker_said": next((l for l in out.splitlines() if "ERROR" in l or "NOT VERIFIED" in l or "FAILED" in l), out.strip()[-200:])}
                        break
                    accepted = desc
                else:
                    rep["corruption_test"] = {"ok": False, "why": f"a one-byte corruption was ACCEPTED ({accepted})"}
            finally:
                shutil.rmtree(ctmp, ignore_errors=True)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    rep["ok"] = not rep["failed"] and rep["n_certs"] > 0 and (rep["corruption_test"] is None or rep["corruption_test"]["ok"])
    return rep


def audit_kills(inst: Instance, tab, mask: Sequence[bool], frac: float = 0.05, seed: Optional[str] = None,
                time_limit: Optional[float] = None, solver: str = "cadical195", verbose: bool = True,
                sentinel: str = SENTINEL) -> dict:
    """Solve a seeded sample (⌈frac·|killed|⌉, seed = table_hash) of the Lean-killed cases of `tab`
    TO COMPLETION.  UNSAT is the expected outcome.  A SAT model is has_kst-checked: a K_{s,t}-free
    model realises a case a proved prune killed (definitions/encoding mismatch) and a model WITH a
    K_{s,t} means the CNF does not forbid it (encoding bug); both write the PIPELINE_BUG sentinel."""
    from .solve import solve_cnf

    killed = [i for i, k in enumerate(mask) if k]
    n = int(math.ceil(frac * len(killed))) if killed else 0
    rng = random.Random(seed if seed is not None else getattr(tab, "table_hash", inst.tag))
    sample = sorted(rng.sample(killed, n)) if n else []
    rep = {"instance": inst.tag, "table_hash": getattr(tab, "table_hash", None), "killed": len(killed), "sampled": sample,
           "frac": frac, "results": [], "bug": None}
    for i in sample:
        rec = tab.records[i]
        cnf = encode_case(inst, rec.rows, rec.cols)
        r = solve_cnf(cnf, inst, solver=solver, conf_budget=None, time_limit=time_limit)
        entry = {"index": i, "rows": list(rec.rows), "cols": list(rec.cols), "status": r.status, "conflicts": r.conflicts,
                 "seconds": round(r.seconds, 3)}
        if r.status == "sat":
            real = bool(r.matrix) and not has_kst(r.matrix, inst.s, inst.t)
            entry["has_kst"] = not real
            entry["matrix"] = ["".join(map(str, row)) for row in (r.matrix or [])]
            rep["bug"] = ("a Lean-killed case is REALIZABLE (K_{s,t}-free model with these sums): definitions/encoding mismatch"
                          if real else "the SAT model CONTAINS a K_{s,t}: the CNF does not forbid it (encoding bug)")
            try:
                os.makedirs(os.path.dirname(sentinel), exist_ok=True)
                with open(sentinel, "a") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} audit-kills {inst.tag} rows={list(rec.rows)} cols={list(rec.cols)}: {rep['bug']}\n")
            except OSError:
                pass
        rep["results"].append(entry)
        if verbose:
            print(f"  [{i}] rows={list(rec.rows)} cols={list(rec.cols)} -> {r.status} conflicts={r.conflicts} {r.seconds:.2f}s", flush=True)
        if rep["bug"]:
            break
    rep["unsat"] = sum(1 for e in rep["results"] if e["status"] == "unsat")
    rep["unknown"] = sum(1 for e in rep["results"] if e["status"] == "unknown")
    rep["ok"] = rep["bug"] is None and rep["unknown"] == 0
    return rep


def _cli_verify_certs(args) -> int:
    rep = verify_certs(args.dir, fresh=args.fresh_lratcheck, corruption_test=not args.no_corruption_test, timeout=args.time)
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


def _cli_audit_kills(args) -> int:
    from .casetable import load_table
    from . import promote as promote_mod

    inst = Instance(args.m, args.n, args.s, args.t, args.w)
    use_table = not (args.pure or args.trust == "pure")
    tab = load_table(inst, use_table=use_table)
    if tab is None:
        sys.exit("no cached table; run `table` first")
    if args.lean:
        from .lean_gate import run_gate

        with open(args.lean, encoding="utf-8") as fh:
            g = run_gate(inst, fh.read(), [(r.rows, r.cols) for r in tab.records], timeout=args.gate_timeout)
        if not g.ok:
            sys.exit(f"Lean gate rejected the prune: {g.errors[:3]}")
        mask, info = list(g.kill_mask), {"route": "file", "entries": [args.lean]}
    else:
        masks, info = promote_mod.library_masks([(inst, tab)], timeout=args.gate_timeout)
        mask = masks[0]
        if mask is None:
            sys.exit("library gate failed")
    print(f"[audit-kills] {inst.tag}: {sum(mask)}/{len(mask)} Lean-killed (route {info['route']}, entries {info['entries']}); "
          f"sampling {args.frac:.0%} seeded by table_hash {tab.table_hash}", flush=True)
    rep = audit_kills(inst, tab, mask, frac=args.frac, time_limit=args.time or None, solver=args.solver)
    rep["library"] = info
    print(json.dumps(rep, indent=1, default=str))
    if rep["bug"]:
        print(f"PIPELINE_BUG: {rep['bug']} -> sentinel written at {SENTINEL}", file=sys.stderr)
        return 2
    return 0 if rep["ok"] else 1


def register(sub) -> None:
    """CLI plugin hook (zar_ub/cli.py): only the two new commands; the inline `certify` stays."""
    p = sub.add_parser("verify-certs", help="re-verify every LRAT certificate under a cert dir (T-11), plus a one-byte corruption control")
    p.add_argument("dir", nargs="?", default=CERT_DIR)
    p.add_argument("--fresh-lratcheck", action="store_true", help="compile tools/drat-trim/lrat-check.c afresh and use that binary")
    p.add_argument("--no-corruption-test", action="store_true")
    p.add_argument("--time", type=float, default=600.0)
    p.set_defaults(func=_cli_verify_certs)
    q = sub.add_parser("audit-kills", help="solve a seeded sample of library-killed cases to completion (T-12); a real SAT writes cache/PIPELINE_BUG")
    q.add_argument("m", type=int)
    q.add_argument("n", type=int)
    q.add_argument("s", type=int)
    q.add_argument("t", type=int)
    q.add_argument("w", type=int)
    q.add_argument("--frac", type=float, default=0.05)
    q.add_argument("--pure", action="store_true")
    q.add_argument("--trust", choices=["pure", "tan2022"], default=None)
    q.add_argument("--lean", default=None, help="audit this Lean candidate's mask instead of the promoted library")
    q.add_argument("--time", type=float, default=0.0, help="per-case wall limit (0 = solve to completion)")
    q.add_argument("--solver", default="cadical195")
    q.add_argument("--gate-timeout", type=float, default=900.0)
    q.set_defaults(func=_cli_audit_kills)

"""Closure: the Lean theorem behind a claimed upper bound z(m,n;s,t) <= w-1 (design §1.1, §4.6).

A bound is one Lean declaration in a harness-owned file `lean/ZarPrune/Closures/Z_m_n_w.lean`:

    abbrev P : Params := ⟨m, n, s, t, w⟩
    abbrev F1 : Fact := ⟨…, "tan2022"⟩        -- external facts: one hypothesis each
    abbrev G1 : Fact := ⟨…, "lean-here"⟩      -- pure facts (waterfilled Argument A): proved in place
    abbrev facts : List Fact := [G1, …, F1, …]
    def condPrune : CondPrune P facts := …    -- library ∪ evolved ∪ conditional prunes
    def survList : List (List Nat × List Nat) := [ … ]              -- what the SAT solver refuted
    theorem survivors_eq : survivorsK P facts condPrune.kill = survList := by decide +kernel
    theorem z_m_n_le_u (h1 : FactHolds F1) … (Hrefuted : ∀ q ∈ survList, ∀ A, sortedProfileOf A = q → ¬ Valid P A) :
        ∀ A : Mat m n, ¬ HasKst P A → weight A ≤ u := upper_bound_succ_of_condPrune …

`survivors_eq` is the only per-instance kernel computation (Tier-1: `decide +kernel`; Tier-1n:
`decide +native`, which adds the axiom `Lean.ofReduceBool` and is named in the report).  The literal
`survList` is what Lean itself computes (`#eval` of the same term), cross-checked against the cached
case table (`baseline_lean_mask`) and against the LRAT manifest.

Deviation from the design's §1.1 sketch: `decide` refuses goals with free variables, so the prune is
assembled as a `CondPrune P facts` (its `kill` is closed) and discharged inside the closure theorem;
`(condPrune.discharge hF).kill = condPrune.kill` definitionally.

Pure-mode facts.  The Python generator (`partitions.py`) bounds proper prefixes by
`ub_counting(m,k) = min(column-side, row-side waterfill)`.  The Lean enumerator takes its prefix bounds
from `facts`, so every tightening counting bound is emitted as a `lean-here` Fact and discharged in the
file by `factHolds_of_waterfill_le` / `factHolds_of_waterfillT_le` (two `decide +kernel`s each).

CLI (plugin):  python -m zar_ub closure M N S T W [--pure|--trust tan2022] [--lean FILE] [--tier 1|1n]
                                              [--timeout S] [--no-check] [--out DIR]
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

from .casetable import CaseTable, LIBRARY_LEAN, load_table
from .known import Instance, ub_counting, max_sum_under_budget
from .ledger import Fact, facts_for, TRUST_LEVELS, DEFAULT_TRUST
from .lean_gate import GateResult, run_lean, LEAN_DIR

UB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLOSURE_LEAN = os.path.join(LEAN_DIR, "ZarPrune", "Closure.lean")
CLOSURES_DIR = os.path.join(LEAN_DIR, "ZarPrune", "Closures")
CLOSURE_OLEAN = os.path.join(LEAN_DIR, ".lake", "build", "lib", "lean", "ZarPrune", "Closure.olean")
ATTEMPTS_DIR = os.path.join(LEAN_DIR, "Attempts")
CERT_DIR = os.path.join(UB, "cache", "certs")
PURE_TAG = "lean-here"
TIERS = ("1", "1n")
NATIVE_AXIOM = "Lean.ofReduceBool"
ALLOWED_AXIOMS = ("propext", "Quot.sound", "Classical.choice")

# design §3.4, printed verbatim in every closure report
TRUSTED_BASE = [
    ("T1", "Lean 4.34.0 kernel; `leanchecker` replay on closure files and on `Evolved.lean`", ("0", "1", "1n")),
    ("T2", "`ZarPrune/Basic.lean` (~60 lines: the statement)", ("0", "1", "1n")),
    ("T3", "Mathlib v4.34.0 modules imported by `Counting.lean` / `Closure.lean`", ("0", "1", "1n")),
    ("T4", "`Closure.lean` (thinning, `act`, sorting, `genParts` + `mem_genParts`, `survivors`, "
           "`upper_bound_succ_of_sorted_cover`), `Schemas.lean` — harness-owned, human-audited at statement level",
     ("0", "1", "1n")),
    ("T4n", "`Lean.ofReduceBool` (one named `_native` axiom per `survivors_eq` discharged by `decide +native`)", ("1n",)),
    ("T5", "`drat-trim` + `lrat-check` verdicts, each a named hypothesis `Hrefuted` paired with `cnf_sha1`/`lrat_sha1` "
           "in `manifest.json`; `encoding.py` completeness (case SAT ⇐ matrix exists)", ("1", "1n")),
    ("T5′", "`Encode.lean` completeness theorem + `LRAT.check_sound` evaluated natively (named `_native` axiom per branch)",
     ("0",)),
    ("T6", "`exists_doubleLex` (block double-lex reachable inside a sorted case)", ("0",)),
    ("T7", "external facts, each a `Fact` hypothesis with provenance tag", ("0", "1", "1n")),
    ("T8", "Tier-2 only: `partitions.py` completeness and Python cover (no Lean closure file; no bound is claimed)", ("2",)),
]


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------
def _counting_sides(m: int, k: int, s: int, t: int) -> Tuple[int, int]:
    """(column-side, row-side) waterfilled Argument A bounds for the (m, k) cell, as `known.ub_counting`."""
    col_side = max_sum_under_budget(k, m, s, (t - 1) * comb(m, s))
    row_side = max_sum_under_budget(m, k, t, (s - 1) * comb(k, t))
    return col_side, row_side


def pure_facts(inst: Instance) -> List[Tuple[Fact, str]]:
    """The counting-bound facts the enumerator uses in every mode: `z(m,k) <= ub_counting(m,k)` for
    1 <= k < n and `z(k,n) <= ub_counting(k,n)` for 1 <= k < m, only where they tighten the trivial
    `m*k`.  Each comes with the side ("col"/"row") whose waterfill attains it, i.e. the Lean lemma
    (`factHolds_of_waterfill_le` / `factHolds_of_waterfillT_le`) that discharges it."""
    m, n, s, t = inst.m, inst.n, inst.s, inst.t
    out: List[Tuple[Fact, str]] = []
    seen = set()

    def add(a, b):
        if (a, b) in seen or a < s or b < t:
            return
        seen.add((a, b))
        ub = ub_counting(a, b, s, t)
        if ub >= a * b:
            return
        col_side, row_side = _counting_sides(a, b, s, t)
        side = "col" if col_side == ub else "row"
        assert min(col_side, row_side) == ub
        out.append((Fact(a, b, s, t, ub, PURE_TAG), side))

    for k in range(1, n):
        add(m, k)
    for k in range(1, m):
        add(k, n)
    out.sort(key=lambda fs: fs[0])
    return out


def external_facts(inst: Instance, trust: str) -> List[Fact]:
    """`ledger.facts_for` (tightening ledger facts only); `pure` -> []."""
    if trust == "pure" or TRUST_LEVELS.get(trust, -1) < 0:
        return []
    return facts_for(inst, trust)


def default_prune_term() -> str:
    """The proved library, as `casetable.LIBRARY_LEAN` defines it (`Prune.or (baseline P) (counting P)`)."""
    body = LIBRARY_LEAN.split(":=", 1)[1].strip()
    return body


# ---------------------------------------------------------------------------
# Lean source
# ---------------------------------------------------------------------------
def _lit(q: Sequence[Sequence[int]]) -> str:
    return "([" + ", ".join(map(str, q[0])) + "], [" + ", ".join(map(str, q[1])) + "])"


def closure_namespace(inst: Instance) -> str:
    return f"ZarPrune.Closures.Z_{inst.m}_{inst.n}_{inst.w}"


def closure_filename(inst: Instance) -> str:
    return f"Z_{inst.m}_{inst.n}_{inst.w}.lean"


def render_closure(inst: Instance, pure: List[Tuple[Fact, str]], external: List[Fact], prune_term: str,
                   survivors: List[Tuple[Sequence[int], Sequence[int]]], tier: str,
                   cond_terms: Optional[List[str]] = None, cand_source: Optional[str] = None,
                   print_axioms: bool = True, eval_only: bool = False) -> str:
    """The closure file text.  `prune_term`: a closed term of type `Prune P` (the library, possibly
    `Prune.or`ed with `Cand.candidate P` when `cand_source` is inlined).  `cond_terms`: extra terms
    of type `CondPrune P facts` (conditional / evolved prunes).  `eval_only`: instead of the
    theorems, `#eval` the Lean survivor list (phase 1 of `closure`)."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    m, n, s, t, w = inst.m, inst.n, inst.s, inst.t, inst.w
    u = w - 1
    ns = closure_namespace(inst)
    L: List[str] = []
    L += ["import ZarPrune", "import ZarPrune.Closure", "", "set_option autoImplicit false",
          "set_option maxRecDepth 8192", "", f"/-! Generated by `zar_ub/closure.py` on {time.strftime('%Y-%m-%d %H:%M')}: "
          f"z({m},{n};{s},{t}) <= {u}, tier {tier}.  Do not edit. -/", "",
          f"namespace {ns}", "open ZarPrune", "",
          f"abbrev P : Params := ⟨{m}, {n}, {s}, {t}, {w}⟩", ""]
    if cand_source:
        L += ["namespace Cand", cand_source.rstrip("\n"), "end Cand", ""]
    # pure facts, proved in place
    if pure:
        L.append("/-! Counting facts (waterfilled Argument A), proved here. -/")
    for i, (f, side) in enumerate(pure, 1):
        L.append(f"abbrev G{i} : Fact := {f.lean}")
        if side == "col":
            L.append(f"theorem G{i}_holds : FactHolds G{i} := factHolds_of_waterfill_le {f.m} {f.n} {f.s} {f.t} {f.z} "
                     f"{json.dumps(f.tag)} (by decide) (by decide +kernel)")
        else:
            L.append(f"theorem G{i}_holds : FactHolds G{i} := factHolds_of_waterfillT_le {f.m} {f.n} {f.s} {f.t} {f.z} "
                     f"{json.dumps(f.tag)} (by decide) (by decide +kernel)")
    if external:
        L += ["", "/-! External facts (hypotheses of the closure theorem), one per Fact. -/"]
    for i, f in enumerate(external, 1):
        L.append(f"abbrev F{i} : Fact := {f.lean}")
    names = [f"G{i}" for i in range(1, len(pure) + 1)] + [f"F{i}" for i in range(1, len(external) + 1)]
    L += ["", f"abbrev facts : List Fact := [{', '.join(names)}]", ""]
    # the prune
    terms = [f"(CondPrune.ofPrune ({prune_term})).weaken (fun _ hf => nomatch hf)"] + list(cond_terms or [])
    L += ["/-- The prune: the proved library" + (" ∪ candidate" if cand_source else "")
          + (" ∪ conditional prunes" if cond_terms else "") + ", as a `CondPrune` (closed `kill`). -/",
          f"def condPrune : CondPrune P facts :=", "  CondPrune.ofList P facts [", "    " + ",\n    ".join(terms), "  ]", ""]
    if eval_only:
        L += ["#eval IO.println (\"SURVIVORS_BEGIN\")",
              "#eval IO.println (toString (survivorsK P facts condPrune.kill))",
              "#eval IO.println (\"SURVIVORS_END\")",
              "#eval IO.println (\"GENROWS \" ++ toString (genRows P facts).length ++ \" GENCOLS \" ++ toString (genCols P facts).length)",
              f"end {ns}", ""]
        return "\n".join(L)
    hyp_list = [f"(h{i} : FactHolds F{i})" for i in range(1, len(external) + 1)]
    hyps = " ".join(hyp_list)
    L += [f"/-- The list the SAT solver refuted ({len(survivors)} cases). -/"]
    if survivors:
        L += ["def survList : List (List Nat × List Nat) := [", "  " + ",\n  ".join(_lit(q) for q in survivors), "]", ""]
    else:
        L += ["def survList : List (List Nat × List Nat) := []", ""]
    tactic = "decide +kernel" if tier == "1" else "decide +native"
    L += [f"/-- The only per-instance kernel computation (Tier-{tier}: `{tactic}`). -/",
          f"theorem survivors_eq : survivorsK P facts condPrune.kill = survList := by {tactic}", ""]
    # facts hold: a right-nested `List.forall_mem_cons` chain, one line per fact
    proofs = [f"G{i}_holds" for i in range(1, len(pure) + 1)] + [f"h{i}" for i in range(1, len(external) + 1)]
    L.append("theorem factsHold" + ("".join("\n    " + h for h in hyp_list) if hyp_list else "")
             + " :\n    ∀ f ∈ facts, FactHolds f :=")
    for d, pr in enumerate(proofs):
        L.append("  " * (d + 1) + f"List.forall_mem_cons.mpr ⟨{pr},")
    L.append("  " * (len(proofs) + 1) + "List.forall_mem_nil _" + "⟩" * len(proofs))
    L.append("")
    hargs = " ".join(f"h{i}" for i in range(1, len(external) + 1))
    fh = f"(factsHold {hargs})" if hargs else "factsHold"
    refuted = "Hrefuted" if survivors else "(fun _ hq => nomatch hq)"
    conds = []
    if external:
        conds.append(f"the {len(external)} external fact(s) `F1`…`F{len(external)}`")
    if survivors:
        conds.append("the LRAT verdicts for the cases in `survList` (`Hrefuted`)")
    doc = f"/-- **z({m},{n};{s},{t}) ≤ {u}**" + (", conditional on " + " and on ".join(conds) if conds else " (unconditional)") \
        + ("; no SAT case remains" if not survivors else "") + ". -/"
    L += [doc, f"theorem z_{m}_{n}_le_{u}" + ("".join("\n    " + h for h in hyp_list) if hyp_list else "")]
    if survivors:
        L.append(f"    (Hrefuted : ∀ q ∈ survList, ∀ A : Mat {m} {n}, sortedProfileOf A = q → ¬ Valid P A) :")
    else:
        L[-1] += " :"
    L += [f"    ∀ A : Mat {m} {n}, ¬ HasKst P A → weight A ≤ {u} :=",
          f"  upper_bound_succ_of_condPrune P {u} rfl facts {fh} condPrune survList survivors_eq {refuted}",
          ""]
    if print_axioms:
        L += [f"#print axioms z_{m}_{n}_le_{u}", ""]
    L += [f"end {ns}", ""]
    return "\n".join(L)


def _inline_closure_module(src: str) -> str:
    """Replace `import ZarPrune.Closure` by the text of `Closure.lean` (for checking a closure file
    before `Closure.olean` exists; the result is elaboration-equivalent up to module boundaries)."""
    with open(CLOSURE_LEAN, encoding="utf-8") as fh:
        mod = fh.read()
    mod_imports = [ln for ln in mod.splitlines() if ln.startswith("import ")]
    mod_body = "\n".join(ln for ln in mod.splitlines() if not ln.startswith("import "))
    src_lines = src.splitlines()
    src_imports = [ln for ln in src_lines if ln.startswith("import ") and ln != "import ZarPrune.Closure"]
    rest = "\n".join(ln for ln in src_lines if not ln.startswith("import "))
    imports = []
    for ln in src_imports + mod_imports:
        if ln not in imports and ln != "import ZarPrune.Cond":
            imports.append(ln)
    return "\n".join(imports) + "\n\n-- ===== inlined lean/ZarPrune/Closure.lean =====\n" + mod_body \
        + "\n-- ===== end of inlined Closure.lean =====\n\n" + rest + "\n"


def closure_module_built() -> bool:
    return os.path.exists(CLOSURE_OLEAN) and os.path.getmtime(CLOSURE_OLEAN) >= os.path.getmtime(CLOSURE_LEAN)


def _scratch_path(name: str) -> str:
    os.makedirs(ATTEMPTS_DIR, exist_ok=True)
    return os.path.join(ATTEMPTS_DIR, name)


_AX_RE = re.compile(r"'([^']+)' depends on axioms: \[([^\]]*)\]")
_NOAX_RE = re.compile(r"'([^']+)' does not depend on any axioms")


def parse_axioms(out: str) -> Dict[str, List[str]]:
    ax = {}
    for mm in _AX_RE.finditer(out):
        ax[mm.group(1)] = [a.strip() for a in mm.group(2).split(",") if a.strip()]
    for mm in _NOAX_RE.finditer(out):
        ax[mm.group(1)] = []
    return ax


def _errors(out: str) -> List[str]:
    return [ln for ln in out.splitlines() if ": error" in ln]


def lean_survivors(inst: Instance, pure, external, prune_term: str, cond_terms=None, cand_source=None,
                   timeout: float = 600.0, inline: Optional[bool] = None) -> dict:
    """Phase 1: `#eval` the Lean survivor list of the very term the closure file will use."""
    src = render_closure(inst, pure, external, prune_term, [], "1", cond_terms, cand_source, eval_only=True)
    inline = (not closure_module_built()) if inline is None else inline
    if inline:
        src = _inline_closure_module(src)
    path = _scratch_path(f"closure_eval_{inst.tag}.lean")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    t0 = time.time()
    out, timed_out = run_lean(path, timeout)
    secs = time.time() - t0
    res = {"path": path, "seconds": round(secs, 2), "timed_out": timed_out, "errors": _errors(out), "survivors": None,
           "n_rows": None, "n_cols": None}
    mm = re.search(r"SURVIVORS_BEGIN\n(.*?)\nSURVIVORS_END", out, re.S)
    if mm and not timed_out:
        text = mm.group(1).strip().replace("(", "[").replace(")", "]")
        res["survivors"] = [(list(q[0]), list(q[1])) for q in ast.literal_eval(text)] if text != "[]" else []
    mm = re.search(r"GENROWS (\d+) GENCOLS (\d+)", out)
    if mm:
        res["n_rows"], res["n_cols"] = int(mm.group(1)), int(mm.group(2))
    return res


def emit_closure(inst: Instance, facts: List[Fact], prune_term: str,
                 survivors: List[Tuple[Sequence[int], Sequence[int]]], tier: str = "1",
                 out_dir: Optional[str] = None, cond_terms: Optional[List[str]] = None,
                 cand_source: Optional[str] = None, pure: Optional[List[Tuple[Fact, str]]] = None) -> str:
    """Write `lean/ZarPrune/Closures/Z_m_n_w.lean` (design §1.1) and return its path.

    `facts`: the external facts (hypotheses; from `ledger.facts_for`).  The counting facts the
    enumerator needs are added automatically (`pure_facts(inst)`) unless `pure` is given."""
    pure = pure_facts(inst) if pure is None else pure
    src = render_closure(inst, pure, list(facts), prune_term, list(survivors), tier, cond_terms, cand_source)
    out_dir = os.path.abspath(out_dir) if out_dir else CLOSURES_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, closure_filename(inst))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src)
    return path


def is_native_axiom(name: str) -> bool:
    """`decide +native` adds `Lean.ofReduceBool` through a per-declaration auxiliary axiom named
    `<decl>._native.decide.ax_k_j` (Lean 4.34); both spellings count."""
    return name == NATIVE_AXIOM or "._native." in name or name.startswith("Lean.ofReduce")


def check_closure(path: str, timeout: float = 2400.0, inline: Optional[bool] = None, tier: str = "1") -> dict:
    """`lake env lean` on the closure file (through an inlined scratch copy under lean/Attempts/ while
    `Closure.olean` does not exist).  `ok` requires: no error, no timeout, and the axioms of the closure
    theorem within {propext, Quot.sound, Classical.choice} plus, for tier 1n only, the named `_native`
    axioms.  Returns ok / seconds / axioms / native_axioms / unexpected_axioms / errors / timed_out."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    inline = (not closure_module_built()) if inline is None else inline
    check_path = path
    if inline:
        check_path = _scratch_path("closure_check_" + os.path.basename(path))
        with open(check_path, "w", encoding="utf-8") as fh:
            fh.write(_inline_closure_module(src))
    t0 = time.time()
    out, timed_out = run_lean(check_path, timeout)
    secs = time.time() - t0
    errs = _errors(out)
    axioms = parse_axioms(out)
    thm = next((k for k in axioms if ".z_" in k or k.startswith("z_")), None)
    ax = axioms.get(thm, None) if thm else None
    native_axioms = [a for a in (ax or []) if is_native_axiom(a)]
    extra = [a for a in (ax or []) if a not in ALLOWED_AXIOMS and not is_native_axiom(a)]
    axioms_ok = ax is not None and not extra and (not native_axioms or tier == "1n")
    ok = (not timed_out) and not errs and axioms_ok
    return {"ok": ok, "seconds": round(secs, 2), "timed_out": timed_out, "errors": errs[:20], "axioms": ax,
            "theorem": thm, "checked_file": check_path, "inline": inline, "tier": tier,
            "axioms_ok": axioms_ok, "native": bool(native_axioms), "native_axioms": native_axioms,
            "unexpected_axioms": extra, "stdout_tail": out[-2000:]}


# ---------------------------------------------------------------------------
# Certificates and the report
# ---------------------------------------------------------------------------
def _key(rows, cols) -> str:
    return "r" + "-".join(map(str, rows)) + "_c" + "-".join(map(str, cols))


def load_manifest(inst: Instance) -> dict:
    p = os.path.join(CERT_DIR, inst.tag, "manifest.json")
    if not os.path.exists(p):
        return {"certs": []}
    with open(p) as fh:
        return json.load(fh)


def certified_map(manifest: dict) -> Dict[str, dict]:
    return {_key(c["rows"], c["cols"]): c for c in manifest.get("certs", []) if c.get("status") == "certified"}


def trusted_base_table(tier: str) -> List[str]:
    lines = ["| # | component | tier | used here |", "|---|---|---|---|"]
    for k, what, tiers in TRUSTED_BASE:
        lines.append(f"| {k} | {what} | {', '.join(tiers)} | {'yes' if tier in tiers else '—'} |")
    return lines


def facts_table(pure, external) -> List[str]:
    lines = ["| fact | provenance | discharged by |", "|---|---|---|"]
    for f, side in pure:
        lemma = "factHolds_of_waterfill_le" if side == "col" else "factHolds_of_waterfillT_le"
        lines.append(f"| z({f.m},{f.n};{f.s},{f.t}) ≤ {f.z} | {f.tag} (waterfilled Argument A, {side} side) | `{lemma}` (kernel `decide`) |")
    for f in external:
        lines.append(f"| z({f.m},{f.n};{f.s},{f.t}) ≤ {f.z} | {f.tag} | hypothesis `FactHolds` of the closure theorem |")
    return lines


def write_closure_report(inst: Instance, table: Optional[CaseTable], gate: Optional[GateResult],
                         cert_manifest: dict, lean_source: str, out_path: Optional[str] = None, *,
                         tier: Optional[str] = None, closure_path: Optional[str] = None,
                         check: Optional[dict] = None, survivors: Optional[List] = None,
                         pure: Optional[List[Tuple[Fact, str]]] = None, external: Optional[List[Fact]] = None,
                         notes: Optional[List[str]] = None) -> str:
    """The human-readable ledger behind a claim.  Signature of batch 1 kept (positional part); the keyword
    part adds the tier, the trusted-base table (design §3.4) and the fact table of the closure file.

    Cases: with `survivors` (the Lean list of the closure file) the rows are those; otherwise the table's
    records with the gate's kill mask (batch-1 behaviour)."""
    certs = certified_map(cert_manifest)
    rows_out = []
    if survivors is not None:
        for rows, cols in survivors:
            key = _key(rows, cols)
            if key in certs:
                rows_out.append((rows, cols, "REFUTED (LRAT verified)", certs[key].get("lrat_file") or ""))
            else:
                rows_out.append((rows, cols, "OPEN (not certified)", ""))
        n_pruned = None
        n_total = None
        if table is not None:
            n_total = len(table.records)
            n_pruned = n_total - len(survivors)
    else:
        cases = table.records if table else []
        killed = gate.kill_mask if (gate and gate.ok and gate.kill_mask) else [False] * len(cases)
        for rec, k in zip(cases, killed):
            key = _key(rec.rows, rec.cols)
            if k:
                rows_out.append((rec.rows, rec.cols, "PRUNED (Lean)", ""))
            elif key in certs:
                rows_out.append((rec.rows, rec.cols, "REFUTED (LRAT verified)", certs[key].get("lrat_file") or ""))
            else:
                rows_out.append((rec.rows, rec.cols, "OPEN (not certified)", ""))
        n_pruned = sum(killed)
        n_total = len(cases)
    n_open = sum(1 for r in rows_out if r[2].startswith("OPEN"))
    n_ref = sum(1 for r in rows_out if r[2].startswith("REFUTED"))
    lean_ok = (check is None) or bool(check.get("ok"))
    all_ok = n_open == 0 and lean_ok
    claim = f"z({inst.m},{inst.n};{inst.s},{inst.t}) <= {inst.w - 1}"
    tier = tier or ("2" if closure_path is None else "1")
    ext = external or []
    lines = [f"# Closure report: {claim}" if all_ok else f"# Closure report (INCOMPLETE): {claim} NOT established",
             "", f"Generated {time.strftime('%Y-%m-%d %H:%M')}.  **Tier {tier}**"
             + (" (reduction only: no Lean closure file, no bound claimed)" if tier == "2" else "") + ".", "",
             "## Status", ""]
    if n_total is not None:
        lines.append(f"- admissible cases (cached table): {n_total}" + (
            f" ({table.n_row_partitions} row partitions x {table.n_col_partitions} column partitions)" if table else ""))
    if n_pruned is not None:
        lines.append(f"- pruned by Lean-verified prune: {n_pruned}")
    lines += [f"- refuted by verified LRAT certificate: {n_ref}", f"- open: {n_open}"]
    if closure_path:
        lines.append(f"- Lean closure file: `{os.path.relpath(closure_path, UB)}`" + (
            f" — checked in {check['seconds']} s, axioms {check.get('axioms')}" if check and check.get("ok")
            else (f" — CHECK FAILED: {(check or {}).get('errors', ['timeout' if (check or {}).get('timed_out') else 'not checked'])[:2]}")))
    if ext:
        lines.append(f"- conditional on {len(ext)} external fact(s) (hypotheses of the theorem): "
                     + "; ".join(str(f) for f in ext))
    else:
        lines.append("- external facts: none (the theorem is unconditional beyond the LRAT verdicts)")
    lines += [f"- **claim established: {'YES' if all_ok else 'NO'}**", ""]
    if notes:
        lines += ["## Notes", ""] + [f"- {x}" for x in notes] + [""]
    lines += ["## Trusted base (design §3.4)", ""] + trusted_base_table(tier) + [""]
    if pure is not None or external is not None:
        lines += ["## Facts used by the enumerator (`genRows P facts`, `genCols P facts`)", ""] + facts_table(pure or [], ext) + [""]
    if check and check.get("axioms") is not None:
        lines += [f"Axioms of `{check.get('theorem')}`: {check['axioms']}" + (
            f" — the `_native` axiom(s) {check.get('native_axioms')} stand for `{NATIVE_AXIOM}` (Tier-1n, T4n)"
            if check.get("native") else "") + (
            f" — UNEXPECTED: {check.get('unexpected_axioms')}" if check.get("unexpected_axioms") else ""), ""]
    lines += ["## Prune (Lean source)", "", "```lean", lean_source.strip(), "```", ""]
    if gate is not None:
        lines += [f"Gate: compiled={gate.compiled}, typed={gate.typed_ok}, axioms_ok={gate.axioms_ok}, file={gate.lean_file}", ""]
    lines += ["## Cases handed to the solver", "", "| rows | cols | disposition | certificate |", "|---|---|---|---|"]
    for r, c, d, f in rows_out:
        lines.append(f"| {list(r)} | {list(c)} | {d} | {os.path.relpath(f, UB) if f else ''} |")
    out_path = out_path or os.path.join(CERT_DIR, inst.tag, "closure_report.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# The whole pipeline for one instance
# ---------------------------------------------------------------------------
def close_instance(inst: Instance, trust: str = DEFAULT_TRUST, tier: str = "1", lean_file: Optional[str] = None,
                   timeout: float = 2400.0, check: bool = True, out_dir: Optional[str] = None,
                   report: bool = True, fallback_native: bool = False, verbose: bool = True,
                   eval_timeout: float = 600.0) -> dict:
    """Phase 1: Lean `#eval` of the survivors; cross-check with the cached table; phase 2: emit
    `Closures/Z_m_n_w.lean` and check it (kernel `decide`; optional `+native` fallback -> tier 1n);
    phase 3: match survivors with LRAT certificates; write the report."""
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    use_table = trust != "pure"
    pure = pure_facts(inst)
    ext = external_facts(inst, trust)
    prune_term = default_prune_term()
    cand_source = None
    if lean_file:
        with open(lean_file, encoding="utf-8") as fh:
            cand_source = fh.read()
        prune_term = f"Prune.or ({prune_term}) (Cand.candidate P)"
    res = {"instance": inst.__dict__, "trust": trust, "tier": tier, "prune_term": prune_term,
           "pure_facts": [str(f) for f, _ in pure], "external_facts": [str(f) for f in ext]}
    # phase 1
    ev = lean_survivors(inst, pure, ext, prune_term, cand_source=cand_source, timeout=eval_timeout)
    res["eval"] = {k: v for k, v in ev.items() if k != "survivors"}
    if ev["survivors"] is None:
        res["error"] = "Lean #eval of the survivors failed: " + "; ".join(ev["errors"][:3])
        log("[closure]", res["error"])
        return res
    surv = ev["survivors"]
    res["n_survivors"] = len(surv)
    log(f"[closure] {inst.tag} trust={trust}: Lean enumerator {ev['n_rows']} rows x {ev['n_cols']} cols, "
        f"{len(surv)} survivors of `{prune_term}` ({ev['seconds']} s)")
    # cross-check against the cached table (as sets; also report order)
    table = load_table(inst, use_table=use_table)
    notes = []
    if table is not None:
        tab_cases = [(list(r.rows), list(r.cols)) for r in table.records]
        if table.baseline_lean_mask is not None and lean_file is None:
            tab_surv = [c for c, k in zip(tab_cases, table.baseline_lean_mask) if not k]
        else:
            tab_surv = None
        res["table"] = {"path": table.path, "cases": len(tab_cases), "hash": table.table_hash,
                        "table_survivors": None if tab_surv is None else len(tab_surv)}
        st, sl = set(map(_key_pair, tab_cases)), set(map(_key_pair, surv))
        if not sl <= st:
            notes.append(f"Lean survivors not in the cached table: {len(sl - st)} (enumerator mismatch!)")
        if tab_surv is not None:
            ss = set(map(_key_pair, tab_surv))
            if ss == sl:
                notes.append(f"Lean survivors == cached table survivors (baseline_lean_mask): {len(surv)} cases"
                             + ("" if [_key_pair(q) for q in surv] == [_key_pair(q) for q in tab_surv] else " (different order)"))
            else:
                notes.append(f"Lean survivors differ from the table's: {len(sl - ss)} extra, {len(ss - sl)} missing")
        res["table"]["notes"] = list(notes)
        for x in notes:
            log("[closure]", x)
    else:
        notes.append("no cached table for this instance/trust (no cross-check)")
    # phase 2
    path = emit_closure(inst, ext, prune_term, surv, tier, out_dir, cand_source=cand_source, pure=pure)
    res["closure_file"] = path
    log(f"[closure] wrote {os.path.relpath(path, UB)} ({len(ext)} hypotheses, {len(pure)} counting facts)")
    chk = None
    if check:
        chk = check_closure(path, timeout=timeout, tier=tier)
        res["check"] = {k: v for k, v in chk.items() if k != "stdout_tail"}
        log(f"[closure] check: ok={chk['ok']} {chk['seconds']} s timed_out={chk['timed_out']} axioms={chk['axioms']}"
            + (f" native_axioms={chk['native_axioms']}" if chk["native"] else "")
            + (f" UNEXPECTED axioms={chk['unexpected_axioms']}" if chk["unexpected_axioms"] else "")
            + (f" errors={chk['errors'][:2]}" if chk["errors"] else ""))
        if not chk["ok"] and tier == "1" and fallback_native:
            log("[closure] Tier-1 failed; falling back to Tier-1n (decide +native)")
            tier = "1n"
            path = emit_closure(inst, ext, prune_term, surv, tier, out_dir, cand_source=cand_source, pure=pure)
            chk = check_closure(path, timeout=timeout, tier=tier)
            res["tier"], res["closure_file"] = tier, path
            res["check_native"] = {k: v for k, v in chk.items() if k != "stdout_tail"}
            log(f"[closure] native check: ok={chk['ok']} {chk['seconds']} s axioms={chk['axioms']}")
    # phase 3
    manifest = load_manifest(inst)
    certs = certified_map(manifest)
    missing = [q for q in surv if _key(q[0], q[1]) not in certs]
    res["certified"] = len(surv) - len(missing)
    res["uncertified"] = len(missing)
    if missing:
        notes.append(f"{len(missing)} survivor(s) without an LRAT certificate (run `certify`)")
    res["established"] = (not missing) and (chk is None or chk["ok"])
    if report:
        lean_source = (cand_source + "\n" if cand_source else "") + f"-- prune term: {prune_term}"
        rpath = os.path.join(out_dir, f"closure_report_{inst.tag}.md") if out_dir else None
        rp = write_closure_report(inst, table, None, manifest, lean_source, rpath, tier=tier, closure_path=path,
                                  check=chk, survivors=surv, pure=pure, external=ext, notes=notes)
        res["report"] = rp
        log(f"[closure] report: {os.path.relpath(rp, UB)}; established={res['established']}")
    return res


def _key_pair(q) -> str:
    return _key(q[0], q[1])


# ---------------------------------------------------------------------------
# CLI plugin
# ---------------------------------------------------------------------------
def _handler(args) -> int:
    inst = Instance(args.m, args.n, args.s, args.t, args.w)
    trust = "pure" if (args.pure or args.trust == "pure") else (args.trust or DEFAULT_TRUST)
    res = close_instance(inst, trust=trust, tier=args.tier, lean_file=args.lean, timeout=args.timeout,
                         check=not args.no_check, out_dir=args.out, fallback_native=args.fallback_native)
    print(json.dumps(res, indent=1, default=str))
    return 0 if res.get("established") else 1


def register(sub) -> None:
    p = sub.add_parser("closure", help="emit + check the Lean closure file Closures/Z_m_n_w.lean and its report")
    p.add_argument("m", type=int)
    p.add_argument("n", type=int)
    p.add_argument("s", type=int)
    p.add_argument("t", type=int)
    p.add_argument("w", type=int)
    p.add_argument("--pure", action="store_true", help="counting bounds only (no external facts)")
    p.add_argument("--trust", choices=sorted(TRUST_LEVELS), default=None, help="provenance filter (default tan2022)")
    p.add_argument("--lean", default=None, metavar="FILE", help="candidate Lean source (defines `candidate (P : Params) : Prune P`) to OR into the library")
    p.add_argument("--tier", choices=TIERS, default="1", help="1 = kernel decide; 1n = decide +native (axiom Lean.ofReduceBool)")
    p.add_argument("--fallback-native", action="store_true", help="on a Tier-1 failure/timeout, re-emit as Tier-1n")
    p.add_argument("--timeout", type=float, default=2400.0, help="seconds for the kernel check (design go/no-go: 30 min)")
    p.add_argument("--no-check", action="store_true", help="emit only")
    p.add_argument("--out", default=None, help="directory for the closure file (default lean/ZarPrune/Closures)")
    p.set_defaults(func=_handler)

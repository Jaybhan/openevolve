"""Reward function of the evolutionary loop — design §5.2–5.4, reward v3 (E27, 2026-09-23).

Pure functions only: no Lean, no SAT, no LLM, no I/O except reading the accepted-prune
ledger (`cache/ledger/accepted_prunes.jsonl`) for `kill_novelty` / `duplicates`.
`evaluator.py` gathers the inputs (Python masks, Lean gate results, library masks) and
calls `score(...)`; everything numeric in the metrics dict comes from here.

Quantities (§5.2), per instance I with case table C_I and proved baseline B:
    S_I     = {q ∈ C_I : q probed, not witnessed, ¬B.kill(q)}        (the survivors of the library)
    d_I(q)  ≥ 1 difficulty (CaDiCaL conflicts; censored labels as the table stores them: E27 hardness model)
    lb_I(q) = d_I(q) when exact, else the conflicts actually reached (a lower bound; never an estimate)
    gain_I(K) = Σ_{q∈S_I, K(q)} d_I(q) / W_I;   tail_I(K) = the same over H_I, the top decile of S_I by d

Reward v3 = E25's "VR" (R1) with R3's two fixes ("VR**"), all constants in the block below:
    x_I(q)  = max(0, d_I(q) − EXCESS_D)      only work above the 20k-conflict probe budget counts (HARD regime);
                                              cases with x = 0 leave the gain/tail sums, a cell with no such case
                                              leaves the role means (R3 attacks A/B: easy cases and tiny tables)
    X_I     = Σ_{q∈S_I} x_I(q);  gain_I, tail_I computed on x
    G_role  = mean over shape families {square, wide, vwide, gen} of the ln(1 + X_I/WORK_W0)-weighted mean
              of gain_I over the role's cells in that family                     (R1: V3 family + V4 work weights)
    a_I     = min(1, ln(1 + LB_I/IMP_W0) / ln(1 + IMP_WREF/IMP_W0))  if LB_I = Σ_{S_I} lb_I ≥ IMP_WMIN, else 0
              (importance from LOWER-BOUND work only, so an inflated censored label cannot buy it: R3 finding 1)
    Depth   = max_{I ∈ TRAIN ∪ TARGET} a_I · gain_I       Close = max_{closed I ∈ TARGET} a_I   (E29: targets only)
              (closed: every library survivor of I killed)
    Tail    = the G aggregation of tail_I over TRAIN ∪ TARGET
    G_target := G_train when no TARGET cell carries work (stage 2); G_gen := G_train likewise.

The formula (§5.3):
    hard_zero = K^P kills a witnessed case ∧ not L5 ∨ ladder ∈ {0,4} ∨ python error/timeout ∨ schema_error ∨ PIPELINE_BUG
    combined  = 0                                                                   if hard_zero
              = 0.20 + 0.80·(0.28 G_train + 0.21 G_target + 0.07 G_gen + 0.14 Tail + 0.15 Depth + 0.15 Close)
                                                                                    if ladder = 5
              = min(0.19, 0.03·ladder/3 + 0.06·lean_partial + 0.08·E + 0.02·tail_train(K^P))   if ladder ∈ {1,2,3}
    stage 1 (no Lean yet): 0 if hard_zero else 0.01 + 0.08·E
E is the same G_train aggregation applied to the Python mirror's kill.  ZAR_UB_REWARD=v2 restores the
v2 verified branch (0.20 + 0.80·(0.40 G_train + 0.30 G_target + 0.10 G_gen + 0.20 Tail), plain means over
cells, full d) for comparisons; the soundness ordering hard-zero 0 < unverified ≤ 0.19 < verified no-op 0.20
is identical in both versions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}
FLOOR_VERIFIED = 0.20
CAP_UNVERIFIED = 0.19
#: E28: an L5 candidate whose Python mirror kills a witnessed case keeps its Lean-only verified value
#: minus this penalty, so its sound twin always ranks strictly above it (E26: unsound-mirror twins
#: tied their sound twins and were reported as the best program in 3/8 runs).
MIRROR_PENALTY = 0.02
SCORED_KINDS = ("train", "target", "gen")
ALL_KINDS = ("train", "battery", "target", "gen")

# ---- reward v3 (E27 = E25 "VR**"): every constant of the verified branch ------------------------
REWARD_VERSION = "v3"
W_TRAIN, W_TARGET, W_GEN, W_TAIL, W_DEPTH, W_CLOSE = 0.28, 0.21, 0.07, 0.14, 0.15, 0.15  # sum 1.0
EXCESS_D = 20_000.0      # x = max(0, d - EXCESS_D): only work above the 20k probe budget earns credit
WORK_W0 = 2_000.0        # cell weight ln(1 + X_I / WORK_W0) inside a family
IMP_W0 = 2_000.0         # importance a_I = min(1, ln(1 + LB_I/IMP_W0) / ln(1 + IMP_WREF/IMP_W0))
IMP_WREF = 1.0e9         # about the work of the largest target table: closing it is worth a_I = 1
IMP_WMIN = 1.0e6         # Depth / Close only on cells whose LOWER-BOUND work LB_I >= 1e6 conflicts
FAMILY_SQUARE_MAX, FAMILY_WIDE_MAX = 1.2, 1.8  # aspect n/m: square < 1.2 <= wide < 1.8 <= vwide; (s,t) != (3,3): gen
# ---- reward v2 (design §5.3 as built in batch 1; ZAR_UB_REWARD=v2) --------------------------------
W2_TRAIN, W2_TARGET, W2_GEN, W2_TAIL = 0.40, 0.30, 0.10, 0.20


def reward_version() -> str:
    """'v3' (default) or 'v2' (env ZAR_UB_REWARD=v2)."""
    return "v2" if os.environ.get("ZAR_UB_REWARD", "").strip().lower() in ("v2", "legacy") else REWARD_VERSION


#: the exact metric keys of design §5.4 (numeric unless noted)
METRIC_KEYS = [
    "combined_score", "sound_battery", "pipeline_bug", "lean_ladder", "lean_partial", "lean_ok",
    "proven_gain", "target_gain", "gen_gain", "tail_gain", "empirical_gain", "schema_gain", "agreement",
    "hard_killed", "censored_share", "survivors_left", "kill_novelty", "n_new_prunes", "n_holes",
    "n_holes_filled", "eval_seconds", "gate_seconds", "gate_cache_hit", "table_hash", "suite_version",
    # v3 (E27): the terms of the new verified branch and per-family views
    "depth_term", "closure_bonus", "closed_cells", "best_cell_gain", "gain_square", "gain_wide", "gain_vwide",
    "gain_gen", "proven_gain_plain",
]
ARTIFACT_KEYS = [
    "per_instance", "lean_errors", "lean_holes", "lean_source_filled", "UNSOUND_counterexamples",
    "python_lean_disagreements", "schema_error", "library_ledger", "duplicates", "kill_signature",
]
#: names of Prune-typed declarations present in the initial program (not "new")
BASELINE_PRUNE_NAMES = {"candidate", "candidateF", "examplePrune"}


# --------------------------------------------------------------------------------------
# per-record helpers (backward compatible with tables that predate owner C's fields)
# --------------------------------------------------------------------------------------
def rec_status(rec) -> Optional[str]:
    p = getattr(rec, "probe", None)
    return p.get("status") if isinstance(p, dict) else None


def is_witnessed(rec) -> bool:
    """A case with a SAT witness: killing it is unsound (battery) / a pipeline bug (Lean)."""
    return rec_status(rec) == "sat"


def is_censored(rec) -> bool:
    c = getattr(rec, "censored", None)
    if c is not None:
        return bool(c)
    return rec_status(rec) == "unknown"


def rec_difficulty(rec, conf_cap: int = 0) -> float:
    """d_I(q) ≥ 1.  Prefers the table's `d` field / `casetable.difficulty` (owner C), else the
    probe: exact conflicts when refuted, the cap when censored.  0.0 for unprobed records
    (they are never in S_I)."""
    d = getattr(rec, "d", None)
    if isinstance(d, (int, float)) and d > 0:
        return float(max(1.0, d))
    try:  # owner C's helper, when present
        from .casetable import difficulty as _table_difficulty  # type: ignore
        v = float(_table_difficulty(rec))
        if v > 0:
            return max(1.0, v)
    except Exception:  # noqa: BLE001  (absent helper, or a record it cannot label)
        pass
    p = getattr(rec, "probe", None)
    if not isinstance(p, dict):
        return 0.0
    if p.get("status") == "unknown":
        return float(max(1, int(p.get("budget_cap") or conf_cap or 1)))
    return float(max(1, int(p.get("conflicts") or 0)))


def rec_lower_bound(rec, conf_cap: int = 0) -> float:
    """lb(q): the exact label when the case was decided, else the conflicts the deepest run actually
    reached (a lower bound on the true d that no estimator touched).  0.0 for unprobed records."""
    if not is_censored(rec):
        return rec_difficulty(rec, conf_cap)
    p = getattr(rec, "probe", None)
    if not isinstance(p, dict):
        return 0.0
    return float(max(1, int(p.get("conflicts") or p.get("budget_cap") or conf_cap or 1)))


def family_of(m: int, n: int, s: int, t: int) -> str:
    """Shape family of a cell: square (n/m < 1.2), wide (< 1.8), vwide (>= 1.8); (s,t) != (3,3): gen."""
    if (s, t) != (3, 3):
        return "gen"
    r = max(m, n) / float(min(m, n))
    if r < FAMILY_SQUARE_MAX:
        return "square"
    if r < FAMILY_WIDE_MAX:
        return "wide"
    return "vwide"


def importance(lb_work: float) -> float:
    """a_I in [0, 1] from LOWER-BOUND work: 0 below IMP_WMIN, else log-scaled share of IMP_WREF."""
    if lb_work < IMP_WMIN or lb_work <= 0:
        return 0.0
    return min(1.0, math.log1p(lb_work / IMP_W0) / math.log1p(IMP_WREF / IMP_W0))


def family_mean(cells: Sequence[dict], attr: str) -> Optional[float]:
    """The v3 role aggregate: ln(1 + X/WORK_W0)-weighted mean of cell[attr] within each family, then the
    plain mean over families.  Cells with X = 0 carry no weight.  None when no cell carries work."""
    fams: Dict[str, List[Tuple[float, float]]] = {}
    for c in cells:
        w = math.log1p(max(0.0, c["X"]) / WORK_W0)
        if w > 0:
            fams.setdefault(c["family"], []).append((float(c[attr]), w))
    if not fams:
        return None
    return sum(sum(v * w for v, w in vs) / sum(w for _, w in vs) for vs in fams.values()) / len(fams)


def survivor_indices(tab, base_mask: Optional[Sequence[bool]], use_sample: bool = True) -> List[int]:
    """S_I: probed, non-witnessed cases not killed by the proved baseline B.
    B = the table's `baseline_lean_mask` when present, else the library mask computed by the
    evaluator; with neither, every probed case is a survivor (E6 behaviour).  Restricted to
    the table's CRN `sample` (§6.3) when one is stored."""
    tbl_mask = getattr(tab, "baseline_lean_mask", None)
    base = tbl_mask if tbl_mask else base_mask
    out = []
    for i, r in enumerate(tab.records):
        if getattr(r, "probe", None) is None or is_witnessed(r):
            continue
        if base is not None and i < len(base) and base[i]:
            continue
        out.append(i)
    sample = getattr(tab, "sample", None)
    if sample and use_sample:
        keep = set(int(x) for x in sample)
        out = [i for i in out if i in keep]
    return out


# --------------------------------------------------------------------------------------
# §5.2 quantities
# --------------------------------------------------------------------------------------
def gain_I(S: Sequence[int], d: Sequence[float], mask: Optional[Sequence[bool]]) -> float:
    W = sum(d[i] for i in S)
    if W <= 0 or not mask:
        return 0.0
    return sum(d[i] for i in S if i < len(mask) and mask[i]) / W


def top_decile(S: Sequence[int], d: Sequence[float]) -> List[int]:
    """H_I: the ⌈|S_I|/10⌉ hardest survivors (ties broken by index — deterministic)."""
    if not S:
        return []
    k = max(1, math.ceil(len(S) / 10.0))
    return sorted(S, key=lambda i: (-d[i], i))[:k]


def tail_I(S: Sequence[int], d: Sequence[float], mask: Optional[Sequence[bool]]) -> float:
    H = top_decile(S, d)
    WH = sum(d[i] for i in H)
    if WH <= 0 or not mask:
        return 0.0
    return sum(d[i] for i in H if i < len(mask) and mask[i]) / WH


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# --------------------------------------------------------------------------------------
# §4.2 S5 ladder and §5.3 lean_partial, with fallbacks for the pre-v2 GateResult
# --------------------------------------------------------------------------------------
def ladder_of(g) -> int:
    """The gate's `ladder` when it has one; otherwise reconstruct it from the v1 fields."""
    if g is None:
        return 0
    lad = getattr(g, "ladder", None)
    if isinstance(lad, int):
        return max(0, min(5, lad))
    if getattr(g, "ok", False):
        return 5
    if not getattr(g, "scanned_ok", False):
        return 0
    if getattr(g, "timed_out", False) and getattr(g, "n_decls_ok", 0) == 0:
        return 0
    if getattr(g, "compiled", False) and getattr(g, "typed_ok", False) and not getattr(g, "axioms_ok", False):
        return 4
    return 1


def combine_ladders(ladders: Sequence[int]) -> int:
    """One ladder level for a candidate scored on several instances: a hard-zero level on any
    instance dominates (same source ⇒ same scan/axioms), otherwise the best instance counts
    (instance-specific candidates are L5 on their instance, `lean_ok < 1` shows the rest)."""
    if not ladders:
        return 0
    if 0 in ladders:
        return 0
    if 4 in ladders:
        return 4
    return max(ladders)


def lean_partial_of(g, ladder: int) -> float:
    """§5.3 exactly.  depth = (first−1)/N (1.0 if no error), declfrac = D_ok/D (0 if D = 0),
    fill = H_filled/H (1 if H = 0)."""
    if ladder >= 5:
        return 1.0
    if ladder in (0, 4) or g is None:
        return 0.0
    D = int(getattr(g, "n_decls", 0) or 0)
    D_ok = int(getattr(g, "n_decls_ok", 0) or 0)
    declfrac = (D_ok / D) if D else 0.0
    first = getattr(g, "first_error_line", None)
    if first is None:
        depth = 1.0
    else:
        N = int(getattr(g, "n_lines", 0) or 0)
        depth = ((first - 1) / N) if N else float(getattr(g, "first_error_frac", 0.0) or 0.0)
    depth = max(0.0, min(1.0, depth))
    H = int(getattr(g, "n_holes", 0) or 0)
    H_f = int(getattr(g, "n_holes_filled", 0) or 0)
    fill = (H_f / H) if H else 1.0
    if ladder == 1:
        return 0.05 + 0.10 * declfrac + 0.05 * depth
    if ladder == 2:
        return 0.25 + 0.15 * declfrac + 0.10 * depth
    return 0.50 + 0.30 * fill + 0.10 * depth  # L3


def unverified_score(ladder: int, lean_partial: float, E: float, tail_train_py: float) -> float:
    return min(CAP_UNVERIFIED, 0.03 * ladder / 3.0 + 0.06 * lean_partial + 0.08 * E + 0.02 * tail_train_py)


def verified_score(G_train: float, G_target: float, G_gen: float, Tail: float, Depth: float = 0.0,
                   Close: float = 0.0, version: Optional[str] = None) -> float:
    """0.20 + 0.80 x (the version's weighted terms).  v2 ignores Depth / Close."""
    if (version or reward_version()) == "v2":
        return FLOOR_VERIFIED + 0.80 * (W2_TRAIN * G_train + W2_TARGET * G_target + W2_GEN * G_gen + W2_TAIL * Tail)
    inner = (W_TRAIN * G_train + W_TARGET * G_target + W_GEN * G_gen + W_TAIL * Tail + W_DEPTH * Depth
             + W_CLOSE * Close)
    return FLOOR_VERIFIED + 0.80 * max(0.0, min(1.0, inner))


# --------------------------------------------------------------------------------------
# novelty against the accepted-prune ledger
# --------------------------------------------------------------------------------------
def load_ledger(path: Optional[str]) -> List[dict]:
    if not path or not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _ledger_killed_set(entry: dict) -> Optional[set]:
    """(tag, index) pairs an accepted prune kills.  Accepts `masks: {tag: [bool…]}` or
    `masks: {tag: [killed indices…]}`; entries without masks cannot be compared (None)."""
    masks = entry.get("masks")
    if not isinstance(masks, dict):
        return None
    killed = set()
    for tag, m in masks.items():
        if not isinstance(m, list):
            continue
        if m and all(isinstance(x, bool) for x in m):
            killed.update((tag, i) for i, v in enumerate(m) if v)
        else:
            killed.update((tag, int(i)) for i in m)
    return killed


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def kill_novelty(killed: set, ledger: List[dict], signature: str = "") -> Tuple[float, List[str]]:
    """1 − Jaccard(K^L, ∪ accepted); 1.0 when the ledger is empty.  Also the `duplicates`
    list: accepted prunes with Jaccard ≥ 0.95 or an identical kill signature."""
    union: set = set()
    dups: List[str] = []
    comparable = False
    for e in ledger:
        name = str(e.get("name") or e.get("lean_name") or e.get("sha") or "?")
        ks = _ledger_killed_set(e)
        if ks is not None:
            comparable = True
            union |= ks
            if ks and _jaccard(killed, ks) >= 0.95:
                dups.append(f"{name} (Jaccard {_jaccard(killed, ks):.3f})")
        sig = e.get("kill_signature") or e.get("per_instance_kill_signature")
        if signature and isinstance(sig, str) and sig == signature:
            dups.append(f"{name} (identical kill signature)")
    if not ledger or not comparable:
        return 1.0, sorted(set(dups))
    return 1.0 - _jaccard(killed, union), sorted(set(dups))


# --------------------------------------------------------------------------------------
# source statistics
# --------------------------------------------------------------------------------------
_PRUNE_DECL = re.compile(r"^\s*(?:def|abbrev|theorem)\s+([A-Za-z_][\w'.]*)[^\n]*?:\s*(?:ZarPrune\.)?(?:Prune|CondPrune)\b",
                         re.M)


def n_new_prunes(lean_source: str) -> int:
    """Prune/CondPrune declarations beyond the initial program (`candidate`, `examplePrune`)."""
    names = {m.group(1) for m in _PRUNE_DECL.finditer(lean_source or "")}
    return len(names - BASELINE_PRUNE_NAMES)


def kill_signature(masks_in_order: Sequence[Optional[Sequence[bool]]]) -> str:
    h = hashlib.sha1()
    for m in masks_in_order:
        h.update(("".join("1" if v else "0" for v in m) if m else "-").encode())
        h.update(b"|")
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# the score
# --------------------------------------------------------------------------------------
def score(tables: Dict[str, List[Tuple[object, object]]], *,
          py_masks: Dict[str, Optional[List[bool]]],
          lean_masks: Optional[Dict[str, Optional[List[bool]]]] = None,
          schema_masks: Optional[Dict[str, Optional[List[bool]]]] = None,
          library_masks: Optional[Dict[str, Optional[List[bool]]]] = None,
          stage: int = 3,
          ladder: Optional[int] = None,
          lean_partial: float = 0.0,
          lean_ok: float = 0.0,
          gate_info: Optional[dict] = None,
          lean_source: str = "",
          python_error: Optional[str] = None,
          schema_error: Optional[str] = None,
          forbidden: Sequence[str] = (),
          pipeline_bug_note: Optional[str] = None,
          ledger_path: Optional[str] = None,
          table_hash: str = "",
          suite_version: str = "",
          eval_seconds: float = 0.0) -> Tuple[dict, Dict[str, str]]:
    """Compute the §5.4 metrics dict and the artifacts dict.

    tables       {"train"|"battery"|"target"|"gen": [(Instance, CaseTable), …]}
    py_masks     tag → K^P over tab.records (None when the Python run failed)
    lean_masks   tag → K^L over tab.records for instances where the candidate is L5
                 (battery tags may carry a partial mask: False where Lean was not evaluated)
    schema_masks tag → K^S (schemaK alone) or None
    library_masks tag → the proved library's mask (B) when the table has no baseline_lean_mask
    stage        1: Python only; 2: Lean on TRAIN∪GEN, no target term; 3: full
    ladder       L0–L5 (None at stage 1); gate_info: dict with n_holes, n_holes_filled, gate_seconds,
                 gate_cache_hit, errors (list[str]), holes (list[dict]), filled_source (str|None)
    """
    lean_masks = lean_masks or {}
    schema_masks = schema_masks or {}
    library_masks = library_masks or {}
    gate_info = gate_info or {}
    tables = {k: list(tables.get(k, [])) for k in ALL_KINDS}
    if stage >= 3 or not tables["target"]:
        include_target = stage >= 3
    else:
        include_target = False

    metrics: dict = {k: 0.0 for k in METRIC_KEYS}
    metrics["table_hash"] = table_hash
    metrics["suite_version"] = suite_version
    metrics["agreement"] = 1.0
    metrics["kill_novelty"] = 1.0
    art: Dict[str, str] = {}

    # ---- battery: K^P must not kill a witnessed case; K^L must not either (PIPELINE_BUG) --
    counterexamples: List[str] = []
    bug_notes: List[str] = []
    for kind in ALL_KINDS:
        for inst, tab in tables[kind]:
            pm = py_masks.get(inst.tag)
            lm = lean_masks.get(inst.tag)
            for i, r in enumerate(tab.records):
                if not is_witnessed(r):
                    continue
                if pm and i < len(pm) and pm[i]:
                    counterexamples.append(
                        f"{inst.tag}: rows={list(r.rows)} cols={list(r.cols)} is REALIZABLE (a K_{{{inst.s},{inst.t}}}-free "
                        f"matrix with these sums exists) but kill() returned True")
                if lm and i < len(lm) and lm[i]:
                    bug_notes.append(f"Lean-proven kill of a SAT-witnessed case: {inst.tag} rows={list(r.rows)} cols={list(r.cols)}")
    if pipeline_bug_note:
        bug_notes.append(pipeline_bug_note)
    sound = not counterexamples
    metrics["sound_battery"] = 1.0 if sound else 0.0
    metrics["pipeline_bug"] = 1.0 if bug_notes else 0.0
    if counterexamples:
        art["UNSOUND_counterexamples"] = "\n".join(counterexamples[:10])
    if bug_notes:
        art["PIPELINE_BUG"] = "\n".join(bug_notes[:10])
    if schema_error:
        art["schema_error"] = str(schema_error)[:4000]
    if python_error:
        art["python_error"] = str(python_error)[-3000:]

    # ---- per-instance quantities ---------------------------------------------------
    lad = int(ladder) if ladder is not None else 0
    if forbidden:
        lad = 0
    metrics["lean_ladder"] = float(lad if stage >= 2 else 0)
    metrics["lean_partial"] = float(lean_partial if stage >= 2 else 0.0)
    metrics["lean_ok"] = float(lean_ok if stage >= 2 else 0.0)

    version = reward_version()
    gains_L = {k: [] for k in SCORED_KINDS}  # v2: plain per-cell lists on full d
    gains_P = {k: [] for k in SCORED_KINDS}
    tails_L = {k: [] for k in SCORED_KINDS}
    tails_P = {k: [] for k in SCORED_KINDS}
    gains_S: List[float] = []
    cells: List[dict] = []  # v3: one dict per scored instance with library survivors
    agree_hits = agree_tot = 0
    hard_killed = 0
    censored_gain_sum = 0.0
    target_gain_sum = 0.0
    survivors_left = 0
    killed_set: set = set()
    sig_masks: List[Optional[Sequence[bool]]] = []
    per_instance: List[str] = []
    disagreements: List[str] = []

    for kind in SCORED_KINDS:
        if kind == "target" and not include_target:
            continue
        for inst, tab in tables[kind]:
            tag = inst.tag
            base = library_masks.get(tag)
            S = survivor_indices(tab, base)
            S_full = survivor_indices(tab, base, use_sample=False)
            cap = int(getattr(tab, "conf_cap", 0) or 0)
            d = [rec_difficulty(r, cap) for r in tab.records]
            x = [max(0.0, v - EXCESS_D) for v in d]  # v3: work above the 20k probe budget
            Sx = [i for i in S if x[i] > 0]
            pm = py_masks.get(tag)
            lm = lean_masks.get(tag) if stage >= 2 else None
            sm = schema_masks.get(tag)
            lean_defined = lm is not None or not tab.records  # an empty table has nothing to verify
            # proved-library floor: without a verified mask the proven part is the library itself
            tbl_base = getattr(tab, "baseline_lean_mask", None)
            lm_eff = lm if lean_defined else (tbl_base if tbl_base else base)
            sig_masks.append(lm)
            gL, tL = gain_I(S, d, lm), tail_I(S, d, lm)
            gP, tP = gain_I(S, d, pm), tail_I(S, d, pm)
            if S:  # an instance with no library survivors carries no work and enters no mean
                gains_L[kind].append(gL)
                tails_L[kind].append(tL)
                gains_P[kind].append(gP)
                tails_P[kind].append(tP)
                if sm is not None:
                    gains_S.append(gain_I(S, d, sm))
            # ---- v3 cell ------------------------------------------------------------
            scale = (len(S_full) / len(S)) if (S and len(S) != len(S_full)) else 1.0  # CRN-sampled tables
            X = scale * sum(x[i] for i in Sx)
            LB = sum(rec_lower_bound(tab.records[i], cap) for i in S_full)
            closed = bool(S_full) and bool(lm) and all(i < len(lm) and lm[i] for i in S_full)
            a_I = importance(LB)
            gLx, tLx = gain_I(Sx, x, lm), tail_I(Sx, x, lm)
            if S_full:
                cells.append({
                    "kind": kind, "tag": tag, "family": family_of(inst.m, inst.n, inst.s, inst.t), "X": X, "LB": LB,
                    "a": a_I, "closed": closed, "gain": gLx, "tail": tLx,
                    "gain_py": gain_I(Sx, x, pm), "tail_py": tail_I(Sx, x, pm),
                    "gain_schema": gain_I(Sx, x, sm) if sm is not None else 0.0})
            if lm_eff and pm:
                n = min(len(lm_eff), len(pm))
                agree_hits += sum(1 for i in S if i < n and bool(lm_eff[i]) == bool(pm[i]))
                agree_tot += sum(1 for i in S if i < n)
                dis = [(tab.records[i], lm_eff[i], pm[i]) for i in S if i < n and bool(lm_eff[i]) != bool(pm[i])][:3]
                if dis:
                    disagreements.append(f"[{tag}{'' if lean_defined else ' vs proved library'}] " + "; ".join(
                        f"rows={list(r.rows)} cols={list(r.cols)} lean={a} python={b}" for r, a, b in dis))
            if lm:
                killed_set.update((tag, i) for i in S if i < len(lm) and lm[i])
            if kind == "target" and lm:
                dd, SS, gg = (d, S, gL) if version == "v2" else (x, Sx, gLx)
                W = sum(dd[i] for i in SS) or 1.0
                hard_killed += sum(1 for i in S if i < len(lm) and lm[i] and is_censored(tab.records[i]))
                censored_gain_sum += sum(dd[i] for i in SS if i < len(lm) and lm[i] and is_censored(tab.records[i])) / W
                target_gain_sum += gg
            alive = [i for i in S if not (lm_eff and i < len(lm_eff) and lm_eff[i])]
            survivors_left += len(alive)
            W_I = sum(d[i] for i in S)
            hardest = sorted(alive, key=lambda i: (-d[i], i))[:6]
            nk_py = sum(1 for i in S if pm and i < len(pm) and pm[i])
            nk_le = (sum(1 for i in S if i < len(lm) and lm[i]) if lm else "n/a")
            v3note = (f" hard_work(>20k)={X:.0f} gain_hard={gLx:.3f} tail_hard={tLx:.3f} importance={a_I:.3f}"
                      f"{' CLOSED' if closed else ''}{' [no work above 20k conflicts: carries no weight]' if X <= 0 else ''}")
            per_instance.append(
                f"{kind} {tag}: cases={len(tab.records)} library_survivors={len(S)} work={W_I:.0f} "
                f"killed_by_you(python)={nk_py} killed_by_you(lean)={nk_le} "
                f"gain_lean={gL:.3f} gain_python={gP:.3f} tail_lean={tL:.3f}"
                f"{v3note if version != 'v2' else ''}"
                f"{' [proved-library floor: your Lean is not L5 on this instance]' if not lean_defined else ''}\n"
                + "".join(f"    still alive (hardest): rows={list(tab.records[i].rows)} cols={list(tab.records[i].cols)} "
                          f"d={d[i]:.0f}{' (censored)' if is_censored(tab.records[i]) else ''}\n" for i in hardest))

    # ---- v2 terms (plain means over cells, full d) ----------------------------------
    G2_train = _mean(gains_L["train"])
    G2_target_raw = _mean(gains_L["target"]) if (include_target and gains_L["target"]) else None
    G2_gen_raw = _mean(gains_L["gen"]) if gains_L["gen"] else None
    Tail2 = _mean(tails_L["train"] + (tails_L["target"] if include_target else []))
    # ---- v3 terms (families, work weights, depth, closure) --------------------------
    tr = [c for c in cells if c["kind"] == "train"]
    ta = [c for c in cells if c["kind"] == "target"]
    ge = [c for c in cells if c["kind"] == "gen"]
    G3_train = family_mean(tr, "gain") or 0.0
    G3_target_raw = family_mean(ta, "gain")
    G3_gen_raw = family_mean(ge, "gain")
    Tail3 = family_mean(tr + ta, "tail") or 0.0
    deep = [c for c in tr + ta if c["X"] > 0]
    Depth = max((c["a"] * c["gain"] for c in deep), default=0.0)
    # E29 (2026-09-23): the closure bonus counts TARGET cells only.  Closing a practice (TRAIN) cell
    # whose value is already known is credited through Depth (a_I x gain = a_I when closed); closing a
    # TARGET cell is the thesis objective itself (a bound with zero SAT) and earns the full bonus.
    # With Close over TRAIN too, DGH (closes the wide practice cell (11,21), helps no target) scored
    # 3.9x the uplift of the general recipe that thins the targets (experiments/E29_weights).
    Close = max((c["a"] for c in deep if c["closed"] and c["kind"] == "target"), default=0.0)
    if version == "v2":
        G_train, G_target_raw, G_gen_raw, Tail = G2_train, G2_target_raw, G2_gen_raw, Tail2
        E = _mean(gains_P["train"]) if sound else 0.0
        tail_train_py = _mean(tails_P["train"]) if sound else 0.0
        schema_g = _mean(gains_S)
    else:
        G_train, G_target_raw, G_gen_raw, Tail = G3_train, G3_target_raw, G3_gen_raw, Tail3
        E = (family_mean(tr, "gain_py") or 0.0) if sound else 0.0
        tail_train_py = (family_mean(tr, "tail_py") or 0.0) if sound else 0.0
        schema_g = family_mean(tr, "gain_schema") or 0.0
    G_target = G_target_raw if G_target_raw is not None else G_train
    G_gen = G_gen_raw if G_gen_raw is not None else G_train

    v = stage >= 2
    metrics["proven_gain"] = G_train if v else 0.0
    metrics["target_gain"] = (G_target_raw or 0.0) if v else 0.0
    metrics["gen_gain"] = (G_gen_raw or 0.0) if v else 0.0
    metrics["tail_gain"] = Tail if v else 0.0
    metrics["empirical_gain"] = E
    metrics["schema_gain"] = schema_g if v else 0.0
    metrics["depth_term"] = Depth if v else 0.0
    metrics["closure_bonus"] = Close if v else 0.0
    metrics["closed_cells"] = float(sum(1 for c in tr + ta if c["closed"])) if v else 0.0
    metrics["best_cell_gain"] = max((c["gain"] for c in deep), default=0.0) if v else 0.0
    for fam in ("square", "wide", "vwide"):
        metrics[f"gain_{fam}"] = (family_mean([c for c in tr + ta if c["family"] == fam], "gain") or 0.0) if v else 0.0
    metrics["gain_gen"] = (family_mean(ge, "gain") or 0.0) if v else 0.0
    metrics["proven_gain_plain"] = G2_train if v else 0.0
    metrics["agreement"] = (agree_hits / agree_tot) if agree_tot else 1.0
    metrics["hard_killed"] = float(hard_killed)
    metrics["censored_share"] = (censored_gain_sum / target_gain_sum) if target_gain_sum > 0 else 0.0
    metrics["survivors_left"] = float(survivors_left)
    metrics["n_new_prunes"] = float(n_new_prunes(lean_source))
    metrics["n_holes"] = float(gate_info.get("n_holes", 0) or 0)
    metrics["n_holes_filled"] = float(gate_info.get("n_holes_filled", 0) or 0)
    metrics["gate_seconds"] = float(gate_info.get("gate_seconds", 0.0) or 0.0)
    metrics["gate_cache_hit"] = 1.0 if gate_info.get("gate_cache_hit") else 0.0
    metrics["eval_seconds"] = float(eval_seconds)

    sig = kill_signature(sig_masks)
    art["kill_signature"] = sig
    ledger = load_ledger(ledger_path)
    nov, dups = kill_novelty(killed_set, ledger, sig) if stage >= 2 else (1.0, [])
    metrics["kill_novelty"] = nov
    if dups:
        art["duplicates"] = "\n".join(dups)
    lib_lines = [f"accepted prune: {e.get('name') or e.get('lean_name') or e.get('sha')}" for e in ledger]
    facts = sorted({f for kind in ALL_KINDS for _, tab in tables[kind] for f in (getattr(tab, "external_facts", None) or [])})
    lib_lines += [f"external fact used by a table: {f}" for f in facts]
    if lib_lines:
        art["library_ledger"] = "\n".join(lib_lines[:60])

    # ---- the formula ---------------------------------------------------------------
    # A Python-mirror kill of a witnessed case is unsound *as a claim*, but the Lean kill is the
    # only thing that ever prunes or earns credit.  So: mirror unsound + Lean L5 => the verified
    # score of the Lean mask alone minus MIRROR_PENALTY (E := 0, agreement < 1, `mirror_unsound` artifact);
    # mirror unsound + not L5 => 0.  At stage 1 (Lean not yet run) return a small positive value
    # so the cascade lets stage 2 decide (E7b, 2026-09-21).
    mirror_unsound = not sound
    hard_zero = bool(bug_notes) or bool(python_error) or bool(schema_error) or bool(forbidden) \
        or (stage >= 2 and lad in (0, 4)) or (mirror_unsound and stage >= 2 and lad != 5)
    if mirror_unsound and not hard_zero:
        art["mirror_unsound"] = ("your Python kill() fires on realizable cases (see UNSOUND_counterexamples); "
                                 "only the Lean kill earns credit -- fix the mirror so it agrees with candidate.kill")
    if hard_zero:
        combined = 0.0
    elif stage == 1:
        combined = 0.01 if mirror_unsound else 0.01 + 0.08 * E
    elif lad == 5:
        combined = verified_score(G_train, G_target, G_gen, Tail, Depth, Close, version=version)
        if mirror_unsound:
            combined = max(0.0, combined - MIRROR_PENALTY)
    else:
        combined = unverified_score(lad, lean_partial, E, tail_train_py)
    metrics["combined_score"] = float(combined)

    # ---- artifacts ----------------------------------------------------------------
    if stage >= 2 and lad == 5 and not hard_zero:
        if version == "v2":
            art["score_breakdown"] = (f"reward v2: 0.20 + 0.80*(0.40*G_train {G_train:.4f} + 0.30*G_target {G_target:.4f} "
                                      f"+ 0.10*G_gen {G_gen:.4f} + 0.20*Tail {Tail:.4f}) = {combined:.4f}")
        else:
            art["score_breakdown"] = (
                f"reward v3: 0.20 + 0.80*(0.28*G_train {G_train:.4f} + 0.21*G_target {G_target:.4f} + 0.07*G_gen "
                f"{G_gen:.4f} + 0.14*Tail {Tail:.4f} + 0.15*Depth {Depth:.4f} + 0.15*Close {Close:.4f}) = {combined:.4f}\n"
                "Only work above 20,000 conflicts per case counts; gains are averaged within the shape families "
                "(square/wide/vwide/gen) weighted by ln(1 + work/2000), then over families; Depth = best "
                "importance x gain on one cell, Close = importance of the best TARGET cell you close completely "
                "(importance from lower-bound work, >= 1e6 conflicts).")
    if per_instance:
        art["per_instance"] = "\n".join(per_instance)
    if disagreements:
        art["python_lean_disagreements"] = "\n".join(disagreements)
    errs = gate_info.get("errors") or []
    if errs:
        art["lean_errors"] = "\n".join(str(e) for e in errs[:6])
    holes = gate_info.get("holes") or []
    if holes:
        art["lean_holes"] = "\n".join(
            f"line {h.get('line', '?')}: {h.get('statement', '')}  -- goal: {h.get('goal', '')}" if isinstance(h, dict) else str(h)
            for h in holes[:8])
    if gate_info.get("filled_source"):
        art["lean_source_filled"] = str(gate_info["filled_source"])[:12000]
    return metrics, art

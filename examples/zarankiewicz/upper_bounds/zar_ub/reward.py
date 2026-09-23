"""Reward function of the evolutionary loop — design §5.2–5.4, implemented exactly.

Pure functions only: no Lean, no SAT, no LLM, no I/O except reading the accepted-prune
ledger (`cache/ledger/accepted_prunes.jsonl`) for `kill_novelty` / `duplicates`.
`evaluator.py` gathers the inputs (Python masks, Lean gate results, library masks) and
calls `score(...)`; everything numeric in the metrics dict comes from here.

Quantities (§5.2), per instance I with case table C_I and proved baseline B:
    S_I     = {q ∈ C_I : q probed, not witnessed, ¬B.kill(q)}        (the survivors of the library)
    d_I(q)  ≥ 1 difficulty (CaDiCaL conflicts; censored labels are what the table stores)
    W_I     = Σ_{q∈S_I} d_I(q);  H_I = top decile of S_I by d_I
    gain_I(K) = Σ_{q∈S_I, K(q)} d_I(q) / W_I;   tail_I(K) = Σ_{q∈H_I, K(q)} d_I(q) / Σ_{q∈H_I} d_I(q)
    G_train  = mean_TRAIN gain_I(K^L)   G_target = mean_TARGET gain_I(K^L) (:= G_train if TARGET = ∅)
    G_gen    = mean_GEN gain_I(K^L)     (:= G_train if GEN = ∅)   Tail = mean_{TRAIN∪TARGET} tail_I(K^L)
    E        = mean_TRAIN gain_I(K^P)   (Python mirror; only used in the unverified branch)

The formula (§5.3):
    hard_zero = K^P kills a witnessed case ∨ ladder ∈ {0,4} ∨ python error/timeout ∨ schema_error ∨ PIPELINE_BUG
    combined  = 0                                                                   if hard_zero
              = 0.20 + 0.80·(0.40 G_train + 0.30 G_target + 0.10 G_gen + 0.20 Tail)  if ladder = 5
              = min(0.19, 0.03·ladder/3 + 0.06·lean_partial + 0.08·E + 0.02·tail_train(K^P))   if ladder ∈ {1,2,3}
    stage 1 (no Lean yet): 0 if hard_zero else 0.01 + 0.08·E
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
W_TRAIN, W_TARGET, W_GEN, W_TAIL = 0.40, 0.30, 0.10, 0.20
SCORED_KINDS = ("train", "target", "gen")
ALL_KINDS = ("train", "battery", "target", "gen")

#: the exact metric keys of design §5.4 (numeric unless noted)
METRIC_KEYS = [
    "combined_score", "sound_battery", "pipeline_bug", "lean_ladder", "lean_partial", "lean_ok",
    "proven_gain", "target_gain", "gen_gain", "tail_gain", "empirical_gain", "schema_gain", "agreement",
    "hard_killed", "censored_share", "survivors_left", "kill_novelty", "n_new_prunes", "n_holes",
    "n_holes_filled", "eval_seconds", "gate_seconds", "gate_cache_hit", "table_hash", "suite_version",
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


def survivor_indices(tab, base_mask: Optional[Sequence[bool]]) -> List[int]:
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
    if sample:
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


def verified_score(G_train: float, G_target: float, G_gen: float, Tail: float) -> float:
    return FLOOR_VERIFIED + 0.80 * (W_TRAIN * G_train + W_TARGET * G_target + W_GEN * G_gen + W_TAIL * Tail)


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

    gains_L = {k: [] for k in SCORED_KINDS}
    gains_P = {k: [] for k in SCORED_KINDS}
    tails_L = {k: [] for k in SCORED_KINDS}
    tails_P = {k: [] for k in SCORED_KINDS}
    gains_S: List[float] = []
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
            cap = int(getattr(tab, "conf_cap", 0) or 0)
            d = [rec_difficulty(r, cap) for r in tab.records]
            pm = py_masks.get(tag)
            lm = lean_masks.get(tag) if stage >= 2 else None
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
                sm = schema_masks.get(tag)
                if sm is not None:
                    gains_S.append(gain_I(S, d, sm))
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
                W = sum(d[i] for i in S) or 1.0
                hard_killed += sum(1 for i in S if i < len(lm) and lm[i] and is_censored(tab.records[i]))
                censored_gain_sum += sum(d[i] for i in S if i < len(lm) and lm[i] and is_censored(tab.records[i])) / W
                target_gain_sum += gL
            alive = [i for i in S if not (lm_eff and i < len(lm_eff) and lm_eff[i])]
            survivors_left += len(alive)
            W_I = sum(d[i] for i in S)
            hardest = sorted(alive, key=lambda i: (-d[i], i))[:6]
            nk_py = sum(1 for i in S if pm and i < len(pm) and pm[i])
            nk_le = (sum(1 for i in S if i < len(lm) and lm[i]) if lm else "n/a")
            per_instance.append(
                f"{kind} {tag}: cases={len(tab.records)} library_survivors={len(S)} work={W_I:.0f} "
                f"killed_by_you(python)={nk_py} killed_by_you(lean)={nk_le} "
                f"gain_lean={gL:.3f} gain_python={gP:.3f} tail_lean={tL:.3f}"
                f"{' [proved-library floor: your Lean is not L5 on this instance]' if not lean_defined else ''}\n"
                + "".join(f"    still alive (hardest): rows={list(tab.records[i].rows)} cols={list(tab.records[i].cols)} "
                          f"d={d[i]:.0f}{' (censored)' if is_censored(tab.records[i]) else ''}\n" for i in hardest))

    G_train = _mean(gains_L["train"])
    G_target = _mean(gains_L["target"]) if (include_target and gains_L["target"]) else G_train
    G_gen = _mean(gains_L["gen"]) if gains_L["gen"] else G_train
    Tail = _mean(tails_L["train"] + (tails_L["target"] if include_target else []))
    E = _mean(gains_P["train"]) if sound else 0.0
    tail_train_py = _mean(tails_P["train"]) if sound else 0.0

    metrics["proven_gain"] = G_train if stage >= 2 else 0.0
    metrics["target_gain"] = (_mean(gains_L["target"]) if (include_target and gains_L["target"]) else 0.0) if stage >= 2 else 0.0
    metrics["gen_gain"] = (_mean(gains_L["gen"]) if gains_L["gen"] else 0.0) if stage >= 2 else 0.0
    metrics["tail_gain"] = Tail if stage >= 2 else 0.0
    metrics["empirical_gain"] = E
    metrics["schema_gain"] = _mean(gains_S) if stage >= 2 else 0.0
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
    # score of the Lean mask alone (E := 0, agreement < 1, `mirror_unsound` artifact);
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
        combined = verified_score(G_train, G_target, G_gen, Tail)
    else:
        combined = unverified_score(lad, lean_partial, E, tail_train_py)
    metrics["combined_score"] = float(combined)

    # ---- artifacts ----------------------------------------------------------------
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

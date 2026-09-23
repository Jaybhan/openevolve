"""Case tables (design §3, §5.1, §6): every admissible case of an instance, its
difficulty label `d` (§6.2), and the kill mask of the PROVED library
(`baseline_lean_mask`, computed by the Lean gate on LIBRARY_LEAN) that defines
the scoring survivors S_I = {q : not baseline_lean_kill(q)}.

v2 fields (all optional in the JSON, so every v1 table under cache/ still loads):
  CaseRecord: d (float >= 1), censored (bool), baseline_lean_kill (bool)
  CaseTable : table_hash (sha1 over the labels), kind (train|battery|target|gen|""),
              omega (suite weight), sample (CRN sample of survivor indices, §6.3),
              baseline_lean_mask (list[bool] | None), label_mode (exact|censored|legacy),
              trust (pure|tan2022), calibration ((a,b,g) used for censored labels | None)

Building a table is the expensive one-off part (never inside evaluate()).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from .cases import Case, baseline_kill
from .difficulty import (
    SCHEDULE,
    CENSORED_MAX_CAP,
    _label_worker,
    _continue_worker,
    label_case,
    continue_label,
    label_from_probe,
    load_calibration,
    fhat,
    censored_d,
)
from .known import Instance, Ledger, exact_value

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(os.path.dirname(_HERE), "cache")
E10_RESULTS = os.path.join(os.path.dirname(_HERE), "experiments", "E10_difficulty", "results.json")

# The proved library whose kill mask is the scoring baseline B (design §5.1): the hand-written
# baseline + counting prunes OR the promoted library `evolved` (lean/ZarPrune/Evolved.lean, regenerated
# by zar_ub/promote.py; `Prune.never` while cache/ledger/accepted_prunes.jsonl is empty).  Batch 2:
# rebuilt with `python -m zar_ub baseline --all --force` (masks unchanged while `evolved` adds no kill).
LIBRARY_LEAN = "def candidate (P : Params) : Prune P := Prune.or (Prune.or (baseline P) (counting P)) (evolved P)\n"

# CRN sampling thresholds (§6.3)
SAMPLE_SMALL, SAMPLE_LARGE = 3_000, 50_000
SAMPLE_N_UNIFORM, SAMPLE_N_STRATIFIED = 500, 2_000


@dataclass
class CaseRecord:
    rows: List[int]
    cols: List[int]
    baseline: str = ""  # name of the reference PYTHON prune that kills it (informational only)
    probe: Optional[dict] = None  # label_case(...).as_probe()  (v1: single-cap probe dict)
    d: float = 1.0  # difficulty >= 1 (§6): exact conflicts, or calibrated when censored
    censored: bool = False
    baseline_lean_kill: bool = False  # killed by the PROVED library (Lean mask)

    @property
    def key(self):
        return Case(tuple(self.rows), tuple(self.cols)).key

    @property
    def status(self) -> str:
        return self.probe["status"] if self.probe else "unprobed"


def difficulty(rec: CaseRecord) -> float:
    """d(q) >= 1 for a record (design §6)."""
    return float(rec.d) if rec.d is not None and rec.d >= 1.0 else 1.0


@dataclass
class CaseTable:
    inst: dict
    n_row_partitions: int
    n_col_partitions: int
    records: List[CaseRecord]
    external_facts: List[str] = field(default_factory=list)
    build_seconds: float = 0.0
    conf_cap: int = 0  # largest conflict cap used for the labels
    use_table: bool = True  # False = "pure" mode: no external exact values used
    table_hash: str = ""
    kind: str = ""  # train | battery | target | gen | ""
    omega: float = 1.0
    sample: Optional[List[int]] = None
    baseline_lean_mask: Optional[List[bool]] = None
    label_mode: str = "legacy"  # exact | censored | legacy (v1 single-cap probes)
    trust: str = ""  # pure | tan2022
    calibration: Optional[List[float]] = None  # (a, b, g) used for the censored labels
    path: Optional[str] = None  # where it was loaded from / saved to (None = in-memory)

    # -- basics ------------------------------------------------------------
    @property
    def instance(self) -> Instance:
        return Instance(**self.inst)

    def survivors(self) -> List[CaseRecord]:
        """Cases not killed by the REFERENCE Python baseline (informational)."""
        return [r for r in self.records if not r.baseline]

    def probed(self) -> List[CaseRecord]:
        return [r for r in self.records if r.probe is not None]

    def scored_indices(self) -> List[int]:
        """S_I: indices not killed by the PROVED library (all records when no mask)."""
        if self.baseline_lean_mask is None:
            return list(range(len(self.records)))
        return [i for i, k in enumerate(self.baseline_lean_mask) if not k]

    def scored(self) -> List[Tuple[int, CaseRecord]]:
        return [(i, self.records[i]) for i in self.scored_indices()]

    # -- hash / sample -----------------------------------------------------
    def compute_hash(self) -> str:
        """sha1 over the instance, the mode and the labels (rows, cols, status, d, censored).
        Independent of the Lean mask and of the sample, which are derived data."""
        payload = {
            "inst": self.inst,
            "use_table": self.use_table,
            "labels": [[r.rows, r.cols, r.status, round(float(r.d), 6), bool(r.censored)] for r in self.records],
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    def refresh(self) -> "CaseTable":
        """Recompute table_hash and the CRN sample (call after any label or mask change)."""
        self.table_hash = self.compute_hash()
        self.sample = make_sample(self)
        return self

    def work(self) -> float:
        """W_I (§5.2): total difficulty of the scored survivors (sample-estimated when sampled)."""
        return work_estimate(self)

    # -- summary / io ------------------------------------------------------
    def summary(self) -> dict:
        surv = self.survivors()
        st: Dict[str, int] = {}
        for r in self.records:
            st[r.status] = st.get(r.status, 0) + 1
        sc = self.scored()
        return {
            "instance": self.inst,
            "cases": len(self.records),
            "row_partitions": self.n_row_partitions,
            "col_partitions": self.n_col_partitions,
            "python_baseline_killed": len(self.records) - len(surv),
            "lean_baseline_killed": (len(self.records) - len(sc)) if self.baseline_lean_mask is not None else None,
            "scored_survivors": len(sc),
            "status": st,
            "censored": sum(1 for r in self.records if r.censored),
            "total_difficulty": round(sum(difficulty(r) for _, r in sc), 1),
            "work_estimate": round(self.work(), 1),
            "max_difficulty": max((difficulty(r) for _, r in sc), default=0.0),
            "tail_share": tail_share(self),
            "external_facts": self.external_facts,
            "build_seconds": round(self.build_seconds, 1),
            "conf_cap": self.conf_cap,
            "label_mode": self.label_mode,
            "trust": self.trust,
            "table_hash": self.table_hash,
            "kind": self.kind,
            "omega": self.omega,
            "sample": None if self.sample is None else len(self.sample),
        }

    def to_json(self) -> dict:
        d = {
            "inst": self.inst,
            "n_row_partitions": self.n_row_partitions,
            "n_col_partitions": self.n_col_partitions,
            "records": [asdict(r) for r in self.records],
            "external_facts": self.external_facts,
            "build_seconds": self.build_seconds,
            "conf_cap": self.conf_cap,
            "use_table": self.use_table,
            "table_hash": self.table_hash,
            "kind": self.kind,
            "omega": self.omega,
            "sample": self.sample,
            "baseline_lean_mask": self.baseline_lean_mask,
            "label_mode": self.label_mode,
            "trust": self.trust,
            "calibration": self.calibration,
            "format": 2,
        }
        return d

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path or default_path(self.instance, self.use_table)
        self.table_hash = self.compute_hash()
        if self.sample is None or self.records:
            self.sample = make_sample(self)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.to_json(), f)
        os.replace(tmp, path)
        self.path = path
        return path


# ---------------------------------------------------------------------------
# §6.3 common-random-number samples and work estimates
# ---------------------------------------------------------------------------
def make_sample(tab: CaseTable) -> Optional[List[int]]:
    """|S_I| <= 3000: None (score on every case).  3000 < |S_I| <= 50000: uniform
    N=500, seed = table_hash.  |S_I| > 50000: stratified N=2000, strata = deciles of
    log2_volume x number of distinct column sums.  Returns sorted record indices."""
    S = tab.scored_indices()
    if len(S) <= SAMPLE_SMALL:
        return None
    seed = int(hashlib.sha1((tab.table_hash or tab.compute_hash()).encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    if len(S) <= SAMPLE_LARGE:
        return sorted(rng.sample(S, SAMPLE_N_UNIFORM))
    vol = {i: float(tab.records[i].probe["log2_volume"]) if tab.records[i].probe else 0.0 for i in S}
    order = sorted(S, key=lambda i: (vol[i], i))
    decile = {i: min(9, 10 * k // len(order)) for k, i in enumerate(order)}
    strata: Dict[Tuple[int, int], List[int]] = {}
    for i in S:
        strata.setdefault((decile[i], len(set(tab.records[i].cols))), []).append(i)
    out: List[int] = []
    for key in sorted(strata):
        members = strata[key]
        n_k = max(1, round(SAMPLE_N_STRATIFIED * len(members) / len(S)))
        out.extend(rng.sample(members, min(n_k, len(members))))
    return sorted(out)


def work_estimate(tab: CaseTable) -> float:
    S = tab.scored_indices()
    if not S:
        return 0.0
    if tab.sample is None:
        return float(sum(difficulty(tab.records[i]) for i in S))
    return len(S) * sum(difficulty(tab.records[i]) for i in tab.sample) / max(1, len(tab.sample))


def gain(tab: CaseTable, mask: Sequence[bool]) -> float:
    """gain_I(K) = sum_{q in S_I, K(q)} d(q) / W_I  (§5.2), on the CRN sample when sampled."""
    idx = tab.sample if tab.sample is not None else tab.scored_indices()
    tot = sum(difficulty(tab.records[i]) for i in idx)
    if tot <= 0:
        return 0.0
    return sum(difficulty(tab.records[i]) for i in idx if mask[i]) / tot


def tail_indices(tab: CaseTable) -> List[int]:
    """H_I: the top decile of the scored survivors by d (at least one case)."""
    idx = tab.sample if tab.sample is not None else tab.scored_indices()
    if not idx:
        return []
    order = sorted(idx, key=lambda i: (-difficulty(tab.records[i]), i))
    return order[: max(1, len(order) // 10)]


def tail_gain(tab: CaseTable, mask: Sequence[bool]) -> float:
    H = tail_indices(tab)
    tot = sum(difficulty(tab.records[i]) for i in H)
    return (sum(difficulty(tab.records[i]) for i in H if mask[i]) / tot) if tot > 0 else 0.0


def tail_share(tab: CaseTable) -> float:
    idx = tab.sample if tab.sample is not None else tab.scored_indices()
    tot = sum(difficulty(tab.records[i]) for i in idx)
    return round(sum(difficulty(tab.records[i]) for i in tail_indices(tab)) / tot, 4) if tot > 0 else 0.0


# ---------------------------------------------------------------------------
# paths / loading (backward compatible with every v1 table)
# ---------------------------------------------------------------------------
def default_path(inst: Instance, use_table: bool = True) -> str:
    return os.path.join(CACHE_DIR, f"case_table_{inst.tag}{'' if use_table else '_pure'}.json")


def _record_from_json(r: dict) -> CaseRecord:
    known = {"rows", "cols", "baseline", "probe", "d", "censored", "baseline_lean_kill"}
    return CaseRecord(**{k: v for k, v in r.items() if k in known})


def table_from_json(d: dict, path: Optional[str] = None) -> CaseTable:
    inst = Instance(**d["inst"])
    recs = [_record_from_json(r) for r in d["records"]]
    v2 = d.get("format", 1) >= 2
    calib = tuple(d["calibration"]) if d.get("calibration") else None
    if not v2:
        for r in recs:
            r.d, r.censored = label_from_probe(inst, r.probe, calib)
    else:
        for r in recs:  # a v2 file may still carry records without d (e.g. hand-edited)
            if r.d is None or r.d < 1.0:
                r.d, r.censored = label_from_probe(inst, r.probe, calib)
    tab = CaseTable(
        inst=d["inst"],
        n_row_partitions=d["n_row_partitions"],
        n_col_partitions=d["n_col_partitions"],
        records=recs,
        external_facts=d.get("external_facts", []),
        build_seconds=d.get("build_seconds", 0.0),
        conf_cap=d.get("conf_cap", 0),
        use_table=d.get("use_table", True),
        table_hash=d.get("table_hash", ""),
        kind=d.get("kind", ""),
        omega=float(d.get("omega", 1.0)),
        sample=d.get("sample"),
        baseline_lean_mask=d.get("baseline_lean_mask"),
        label_mode=d.get("label_mode", "legacy"),
        trust=d.get("trust") or ("tan2022" if d.get("use_table", True) else "pure"),
        calibration=list(calib) if calib else None,
        path=path,
    )
    mask = tab.baseline_lean_mask
    if mask is not None and len(mask) != len(recs):
        tab.baseline_lean_mask = None  # stale mask (records changed): drop it rather than misalign
        mask = None
    for i, r in enumerate(recs):
        r.baseline_lean_kill = bool(mask[i]) if mask is not None else False
    tab.table_hash = tab.compute_hash()
    if tab.sample is None and len(tab.scored_indices()) > SAMPLE_SMALL:
        tab.sample = make_sample(tab)
    return tab


def load_table(inst: Instance, path: Optional[str] = None, use_table: bool = True) -> Optional[CaseTable]:
    path = path or default_path(inst, use_table)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return table_from_json(d, path)


def load_any(inst: Instance, prefer_pure: bool = True) -> Optional[CaseTable]:
    order = (False, True) if prefer_pure else (True, False)
    for ut in order:
        t = load_table(inst, use_table=ut)
        if t is not None:
            return t
    return None


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def default_mode(inst: Instance) -> str:
    """exact for cells with a known exact value (TRAIN/BATTERY/GEN), censored for open cells."""
    return "exact" if exact_value(inst.m, inst.n, inst.s, inst.t) is not None else "censored"


def build_table(
    inst: Instance,
    conf_cap: Optional[int] = None,
    time_limit: Optional[float] = None,
    probe: bool = True,
    verbose: bool = True,
    solver: str = "cadical195",
    use_table: bool = True,
    probe_all: bool = True,
    mode: Optional[str] = None,
    jobs: int = 1,
    kind: str = "",
    omega: Optional[float] = None,
) -> CaseTable:
    """Enumerate every admissible case and label each one with `label_case` (§6.2).

    conf_cap: largest SCHEDULE cap to run (default: full schedule in exact mode, 20k in
    censored mode).  time_limit is accepted for CLI compatibility (the SCHEDULE fixes
    the per-cap limits).  probe_all=True labels EVERY case (the reference Python prune
    field is informational only) so that the scoring baseline is exactly the set of
    prunes proved in Lean."""
    t0 = time.time()
    ledger = Ledger()
    from .partitions import row_partitions, column_partitions

    mode = mode or default_mode(inst)
    rp = row_partitions(inst, ledger, use_table)
    cp = column_partitions(inst, ledger, use_table)
    cases = [Case(r, c) for r in rp for c in cp]
    recs = [CaseRecord(list(cs.rows), list(cs.cols), baseline_kill(inst, cs.rows, cs.cols)) for cs in cases]
    surv = [r for r in recs if not r.baseline]
    max_cap = conf_cap if conf_cap is not None else (SCHEDULE[-1][0] if mode == "exact" else CENSORED_MAX_CAP)
    calib = load_calibration(inst.s, inst.t)
    if verbose:
        print(
            f"[{inst.tag}] {len(rp)} row parts x {len(cp)} col parts = {len(cases)} cases; "
            f"reference Python baseline would kill {len(recs) - len(surv)}, survivors {len(surv)}; "
            f"mode={mode} max_cap={max_cap} jobs={jobs}",
            flush=True,
        )
    if probe and recs:
        to_probe = recs if probe_all else surv
        args = [(asdict(inst), r.rows, r.cols, mode, calib, solver, max_cap) for r in to_probe]
        if jobs > 1 and len(args) > 1:
            import multiprocessing as mp

            with mp.get_context("fork" if hasattr(os, "fork") else "spawn").Pool(jobs) as pool:
                probes = list(pool.imap(_label_worker, args, chunksize=4))
        else:
            probes = [_label_worker(a) for a in args]
        for k, (r, p) in enumerate(zip(to_probe, probes)):
            r.probe = p
            r.d = float(p["d"]) if p["d"] is not None else 1.0
            r.censored = bool(p["censored"])
            if verbose and (k % 50 == 0 or p["status"] != "unsat"):
                print(
                    f"  case {k+1}/{len(to_probe)} rows={r.rows} cols={r.cols} -> {p['status']} "
                    f"conflicts={p['conflicts']} d={r.d:.0f}{' (censored)' if r.censored else ''} {p['seconds']:.2f}s",
                    flush=True,
                )
    tab = CaseTable(
        inst=asdict(inst),
        n_row_partitions=len(rp),
        n_col_partitions=len(cp),
        records=recs,
        external_facts=ledger.as_list(),
        build_seconds=time.time() - t0,
        conf_cap=max_cap,
        use_table=use_table,
        kind=kind,
        omega=omega if omega is not None else 1.0,
        label_mode=mode,
        trust="tan2022" if use_table else "pure",
        calibration=list(calib) if calib else None,
    )
    tab.refresh()
    return tab


# ---------------------------------------------------------------------------
# the proved-library mask (design §5.1: `python -m zar_ub table ... --baseline`)
# ---------------------------------------------------------------------------
def check_mask_against_witnesses(tab: CaseTable, mask: Sequence[bool]) -> List[str]:
    """A PROVEN prune can never kill a SAT-witnessed case.  Returns the violations."""
    bad = []
    for r, k in zip(tab.records, mask):
        if k and r.probe and r.probe.get("status") == "sat":
            bad.append(f"rows={r.rows} cols={r.cols}")
    return bad


def set_baseline_mask(tab: CaseTable, mask: Optional[Sequence[bool]]) -> CaseTable:
    if mask is not None:
        if len(mask) != len(tab.records):
            raise ValueError(f"mask length {len(mask)} != records {len(tab.records)}")
        bad = check_mask_against_witnesses(tab, mask)
        if bad:
            raise RuntimeError("PIPELINE_BUG: the proved library killed SAT-witnessed cases: " + "; ".join(bad[:5]))
        tab.baseline_lean_mask = [bool(x) for x in mask]
    else:
        tab.baseline_lean_mask = None
    for i, r in enumerate(tab.records):
        r.baseline_lean_kill = bool(mask[i]) if mask is not None else False
    tab.refresh()
    return tab


def compute_baseline_masks(
    tables: List[CaseTable],
    lean_src: str = LIBRARY_LEAN,
    timeout: float = 600.0,
    verbose: bool = True,
    tag: str = "library",
) -> List[Optional[List[bool]]]:
    """Run the Lean gate once on LIBRARY_LEAN for several tables (one Lean process);
    falls back to one `run_gate` per table if the multi-instance entry point is
    unavailable.  Empty tables get [] without touching Lean."""
    from . import lean_gate

    out: List[Optional[List[bool]]] = [None] * len(tables)
    todo = [(k, t) for k, t in enumerate(tables) if t.records]
    for k, t in enumerate(tables):
        if not t.records:
            out[k] = []
    if not todo:
        return out
    inputs = [(t.instance, [(r.rows, r.cols) for r in t.records]) for _, t in todo]
    results = None
    if hasattr(lean_gate, "run_gate_multi"):
        try:
            results = lean_gate.run_gate_multi(inputs, lean_src, timeout=timeout, tag=tag)
        except TypeError:
            results = None
    if results is None:
        results = [lean_gate.run_gate(inst, lean_src, cases, timeout=timeout, tag=tag) for inst, cases in inputs]
    for (k, t), g in zip(todo, results):
        ok = bool(getattr(g, "ok", False)) and getattr(g, "kill_mask", None) is not None
        if verbose:
            axioms = getattr(g, "axioms", [])
            print(
                f"[baseline] {t.instance.tag}: gate {'OK' if ok else 'FAIL'} axioms={axioms} "
                f"kills={sum(g.kill_mask) if ok else '?'}/{len(t.records)} {getattr(g, 'seconds', 0):.1f}s"
                + ("" if ok else f" errors={getattr(g, 'errors', [])[:3]}"),
                flush=True,
            )
        out[k] = list(g.kill_mask) if ok else None
    return out


def add_baseline(tab: CaseTable, timeout: float = 600.0, verbose: bool = True, save: bool = True) -> bool:
    """Compute and store baseline_lean_mask for one table (no probes are recomputed)."""
    mask = compute_baseline_masks([tab], timeout=timeout, verbose=verbose)[0]
    if mask is None:
        return False
    set_baseline_mask(tab, mask)
    if save:
        tab.save()
    return True


# ---------------------------------------------------------------------------
# deepen: continue the SCHEDULE on censored cases, in place (§6.2, §7)
# ---------------------------------------------------------------------------
def deepen(
    tab: CaseTable,
    cap: int,
    time_limit: Optional[float] = None,
    jobs: int = 1,
    solver: str = "cadical195",
    verbose: bool = True,
    save: bool = True,
    limit: Optional[int] = None,
) -> dict:
    inst = tab.instance
    calib = load_calibration(inst.s, inst.t)
    todo = [
        (i, r)
        for i, r in enumerate(tab.records)
        if r.probe and r.probe.get("status") == "unknown" and int(r.probe.get("budget_cap", 0)) < cap
    ]
    todo.sort(key=lambda ir: (-difficulty(ir[1]), ir[0]))  # hardest first (critical path)
    if limit is not None:
        todo = todo[:limit]
    before = tab.table_hash
    t0 = time.time()
    if verbose:
        print(f"[deepen {inst.tag}] {len(todo)} censored cases below cap {cap}; hash {before}", flush=True)
    args = [(asdict(inst), r.rows, r.cols, r.probe, cap, calib, solver, time_limit) for _, r in todo]
    if jobs > 1 and len(args) > 1:
        import multiprocessing as mp

        with mp.get_context("fork" if hasattr(os, "fork") else "spawn").Pool(jobs) as pool:
            probes = list(pool.imap(_continue_worker, args, chunksize=1))
    else:
        probes = [_continue_worker(a) for a in args]
    resolved = still = sat = 0
    for (i, r), p in zip(todo, probes):
        r.probe = p
        r.d = float(p["d"]) if p["d"] is not None else 1.0
        r.censored = bool(p["censored"])
        if p["status"] == "unsat":
            resolved += 1
        elif p["status"] == "sat":
            sat += 1
        else:
            still += 1
        if verbose:
            print(
                f"  rows={r.rows} cols={r.cols} -> {p['status']} conflicts={p['conflicts']} d={r.d:.0f}"
                f"{' (censored)' if r.censored else ''}",
                flush=True,
            )
    tab.conf_cap = max(tab.conf_cap, cap) if todo else tab.conf_cap
    if tab.label_mode == "legacy":
        tab.label_mode = "exact" if not any(x.censored for x in tab.records) else "censored"
    tab.refresh()
    if save and todo:
        tab.save()
    return {
        "instance": inst.tag,
        "cap": cap,
        "deepened": len(todo),
        "resolved_unsat": resolved,
        "sat": sat,
        "still_censored": still,
        "hash_before": before,
        "hash_after": tab.table_hash,
        "seconds": round(time.time() - t0, 1),
    }


def relabel(tab: CaseTable, save: bool = True) -> dict:
    """Recompute the censored labels (d = min(max(cap, fhat), 20*cap)) of a table with
    the CURRENT calibration file, without solving anything; bumps table_hash."""
    inst = tab.instance
    calib = load_calibration(inst.s, inst.t)
    before = tab.table_hash
    n = 0
    for r in tab.records:
        if r.probe and r.probe.get("status") == "unknown":
            cap = int(r.probe.get("budget_cap") or 20_000)
            c2000 = int(r.probe.get("c2000") or min(int(r.probe.get("conflicts", cap)), 2000))
            fh = fhat(calib, c2000, float(r.probe.get("log2_volume", 0.0)))
            r.probe["fhat"] = fh
            r.probe["c2000"] = c2000
            r.probe["d"] = r.d = censored_d(cap, fh)
            r.probe["censored"] = r.censored = True
            n += 1
    tab.calibration = list(calib) if calib else None
    tab.refresh()
    if save and n and tab.path:
        tab.save()
    return {
        "instance": inst.tag,
        "relabelled": n,
        "hash_before": before,
        "hash_after": tab.table_hash,
        "calibration": tab.calibration,
    }


# ---------------------------------------------------------------------------
# c2000 for calibration of legacy tables
# ---------------------------------------------------------------------------
def ensure_c2000(tab: CaseTable, e10_path: Optional[str] = None, save: bool = True, verbose: bool = False) -> int:
    """Make sure every probed record carries probe['c2000'] (conflicts at the 2000 cap):
    taken from E10's results.json when the case is there, else re-measured (~0.05 s).
    Returns the number of records filled.  Labels (and hence table_hash) do not change."""
    missing = [r for r in tab.records if r.probe and "c2000" not in r.probe]
    if not missing:
        return 0
    e10: Dict[Tuple[tuple, tuple], int] = {}
    path = e10_path or E10_RESULTS
    if os.path.exists(path):
        try:
            for row in json.load(open(path)).get("rows", []):
                if tuple(row["cell"]) == (tab.inst["m"], tab.inst["n"]):
                    e10[(tuple(row["rows"]), tuple(row["cols"]))] = int(row["c2000"])
        except (ValueError, KeyError):
            e10 = {}
    inst = tab.instance
    n_measured = 0
    from .encoding import encode_case
    from .solve import solve_cnf

    for r in missing:
        key = (tuple(r.rows), tuple(r.cols))
        if key in e10:
            r.probe["c2000"] = e10[key]
        else:
            if (
                r.probe.get("status") == "unsat"
                and int(r.probe.get("conflicts", 0)) < 2000
                and int(r.probe.get("budget_cap", 0)) >= 2000
            ):
                r.probe["c2000"] = int(r.probe["conflicts"])  # refuted inside the first cap: identical run
            else:
                res = solve_cnf(encode_case(inst, r.rows, r.cols), inst, conf_budget=2000, time_limit=SCHEDULE[0][1])
                r.probe["c2000"] = int(res.conflicts)
                n_measured += 1
    if verbose:
        print(
            f"[c2000 {inst.tag}] filled {len(missing)} (from E10: {len(missing) - n_measured}, measured: {n_measured})",
            flush=True,
        )
    if save and tab.path:
        tab.save()
    return len(missing)

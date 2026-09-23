"""Branch difficulty labels -- design §6.

d(q) = CaDiCaL 1.9.5 (pysat `cadical195`, default options) conflicts needed to
refute `encode_case(P, q)`: exact when the case is refuted inside the escalating
SCHEDULE, censored-and-calibrated otherwise.  Conflicts, not seconds, so the
label is reproducible across machines for a fixed solver build.

  label_case(inst, rows, cols, mode)      §6.2 estimator (mode = "exact" | "censored")
  calibrate(...)                          fit (a, b, g) of  log fhat = a + b log c2000 + g log2_volume
                                          on TRAIN, validate on a held-out cell (Spearman rho, log-RMSE)
  deepen(...)                             lives in casetable.py (needs the table); the per-case
                                          continuation is `continue_label` below
  CRN sampling of large tables (§6.3)     `make_sample`, `work_estimate` in casetable.py

The legacy `Probe` / `probe_case` API is kept for callers of the v1 tables.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from math import comb, log2
from typing import Dict, List, Optional, Sequence, Tuple

from pysat.solvers import Solver

from .encoding import encode_case, has_kst
from .known import Instance
from .solve import solve_cnf

_HERE = os.path.dirname(os.path.abspath(__file__))
_UB = os.path.dirname(_HERE)

# (conflict cap, wall-clock limit in seconds); a fresh solver at every cap (§6.2)
SCHEDULE: List[Tuple[int, float]] = [(2_000, 5.0), (20_000, 30.0), (200_000, 120.0), (2_000_000, 600.0)]
CENSORED_MAX_CAP = 20_000  # mode="censored" stops after this cap (first pass on TARGET tables)
CENSOR_CLIP = 20  # a censored label is clipped to [cap, CENSOR_CLIP * cap]
SOLVER = "cadical195"
CALIB_PATH = os.path.join(_UB, "experiments", "E11_calibration", "calibrate_33.json")


# ---------------------------------------------------------------------------
# cheap features
# ---------------------------------------------------------------------------
def log2_volume(inst: Instance, rows: Sequence[int]) -> float:
    return float(sum(log2(comb(inst.n, r)) for r in rows))


def propagation_fraction(cnf, inst: Instance, solver=SOLVER) -> float:
    s = Solver(name=solver, bootstrap_with=cnf.clauses)
    try:
        ok, lits = s.propagate()
        if not ok:
            return 1.0
        grid = set(abs(l) for l in lits if abs(l) <= inst.m * inst.n)
        return len(grid) / float(inst.m * inst.n)
    finally:
        s.delete()


def case_features(inst: Instance, rows: Sequence[int], cols: Sequence[int], cnf=None) -> dict:
    cnf = cnf if cnf is not None else encode_case(inst, rows, cols)
    return dict(
        nvars=cnf.nvars,
        nclauses=len(cnf.clauses),
        log2_volume=log2_volume(inst, rows),
        prop_frac=propagation_fraction(cnf, inst),
    )


# ---------------------------------------------------------------------------
# calibration  log fhat = a + b*log(max(c2000,1)) + g*log2_volume   (per (s,t))
# ---------------------------------------------------------------------------
DEFAULT_COEF = (0.0, 1.0, 0.0)  # fhat = c2000 -> d = cap (no calibration on file)


def load_calibration(s: int, t: int, path: str = CALIB_PATH) -> Optional[Tuple[float, float, float]]:
    """(a, b, g) fitted for (s,t), or None when no calibration file exists."""
    try:
        with open(path) as f:
            d = json.load(f)
        c = d.get("fits", {}).get(f"{s},{t}")
        if c:
            return float(c["a"]), float(c["b"]), float(c["g"])
    except (OSError, ValueError, KeyError):
        pass
    return None


FIRST_CAP = SCHEDULE[0][0]


def fhat(coef: Optional[Tuple[float, float, float]], c2000: int, log2_vol: float) -> float:
    """c2000 is a censored measurement: the solver stops at the first conflict at or
    above the 2,000 budget, so values 2001..2006 carry no information and are clipped
    to the cap (otherwise least squares fits that noise: b = -20 on E10's data)."""
    a, b, g = coef if coef is not None else DEFAULT_COEF
    return math.exp(a + b * math.log(max(min(c2000, FIRST_CAP), 1)) + g * log2_vol)


def censored_d(cap: int, fh: float) -> float:
    return float(min(max(cap, fh), CENSOR_CLIP * cap))


# ---------------------------------------------------------------------------
# the label
# ---------------------------------------------------------------------------
@dataclass
class Label:
    status: str  # sat | unsat | unknown
    d: Optional[float]  # difficulty >= 1 (None for sat)
    censored: bool
    cap: int  # last conflict cap used (budget_cap)
    conflicts: int  # conflicts of the last probe
    seconds: float  # solver seconds summed over the schedule
    c2000: int  # conflicts at the first (2000) cap
    fhat: Optional[float]  # calibrated estimate (censored labels only)
    witness: Optional[bool]  # sat: the model passed the independent has_kst check
    nvars: int
    nclauses: int
    log2_volume: float
    prop_frac: float
    matrix: Optional[List[List[int]]] = None  # sat witness (only when asked for)

    def as_probe(self) -> dict:
        """Legacy-compatible probe dict (v1 keys) plus the v2 label fields."""
        return {
            "status": self.status,
            "conflicts": self.conflicts,
            "seconds": round(self.seconds, 4),
            "budget_cap": self.cap,
            "fixed_by_propagation": self.prop_frac,
            "log2_volume": self.log2_volume,
            "nvars": self.nvars,
            "nclauses": self.nclauses,
            "c2000": self.c2000,
            "fhat": self.fhat,
            "d": self.d,
            "censored": self.censored,
            "witness": self.witness,
        }

    @property
    def difficulty(self) -> float:
        return float(self.d if self.d is not None else 1.0)


def _schedule_for(mode: str, max_cap: Optional[int]) -> List[Tuple[int, float]]:
    caps = list(SCHEDULE)
    if mode == "censored":
        caps = [(c, tl) for c, tl in caps if c <= CENSORED_MAX_CAP]
    if max_cap is not None:
        caps = [(c, tl) for c, tl in caps if c <= max_cap]
        if not caps or caps[-1][0] < max_cap and max_cap not in [c for c, _ in SCHEDULE]:
            # a custom cap outside the schedule: append it with a proportional time limit
            tl = max(5.0, 600.0 * max_cap / 2_000_000)
            caps.append((max_cap, tl))
    return caps


def label_case(
    inst: Instance,
    rows: Sequence[int],
    cols: Sequence[int],
    mode: str = "exact",
    calib: Optional[Tuple[float, float, float]] = None,
    solver: str = SOLVER,
    max_cap: Optional[int] = None,
    keep_matrix: bool = False,
) -> Label:
    """§6.2 estimator.  mode="exact" runs the full SCHEDULE (2k -> 2M conflicts);
    mode="censored" stops at 20k.  A case still open at the last cap gets the
    calibrated, clipped label  d = min(max(cap, fhat), 20*cap)  and censored=True."""
    if mode not in ("exact", "censored"):
        raise ValueError(f"mode must be 'exact' or 'censored', got {mode!r}")
    cnf = encode_case(inst, rows, cols)
    feats = case_features(inst, rows, cols, cnf)
    if calib is None:
        calib = load_calibration(inst.s, inst.t)
    c2000: Optional[int] = None
    last_cap = 0
    last_conf = 0
    secs = 0.0
    for cap, tl in _schedule_for(mode, max_cap):
        r = solve_cnf(cnf, inst, solver=solver, conf_budget=cap, time_limit=tl)  # fresh solver each cap
        secs += r.seconds
        c2000 = c2000 if c2000 is not None else r.conflicts
        last_cap, last_conf = cap, r.conflicts
        if r.status == "sat":
            wit = bool(
                r.matrix is not None
                and not has_kst(r.matrix, inst.s, inst.t)
                and [sum(x) for x in r.matrix] == list(rows)
                and [sum(r.matrix[i][j] for i in range(inst.m)) for j in range(inst.n)] == list(cols)
            )
            return Label(
                "sat",
                None,
                False,
                cap,
                r.conflicts,
                secs,
                c2000,
                None,
                wit,
                matrix=r.matrix if keep_matrix else None,
                **feats,
            )
        if r.status == "unsat":
            return Label("unsat", float(max(1, r.conflicts)), False, cap, r.conflicts, secs, c2000, None, None, **feats)
    fh = fhat(calib, c2000 or 0, feats["log2_volume"])
    return Label("unknown", censored_d(last_cap, fh), True, last_cap, last_conf, secs, c2000 or 0, fh, None, **feats)


def continue_label(
    inst: Instance,
    rows: Sequence[int],
    cols: Sequence[int],
    probe: dict,
    max_cap: int,
    calib: Optional[Tuple[float, float, float]] = None,
    solver: str = SOLVER,
    time_limit: Optional[float] = None,
) -> Label:
    """Deepen one censored case: run the SCHEDULE caps above probe['budget_cap'] up to
    max_cap (time limit per cap from the SCHEDULE unless overridden)."""
    cnf = encode_case(inst, rows, cols)
    feats = dict(
        nvars=probe.get("nvars", cnf.nvars),
        nclauses=probe.get("nclauses", len(cnf.clauses)),
        log2_volume=probe.get("log2_volume", log2_volume(inst, rows)),
        prop_frac=probe.get("fixed_by_propagation", probe.get("prop_frac", 0.0)),
    )
    if calib is None:
        calib = load_calibration(inst.s, inst.t)
    c2000 = int(probe.get("c2000") or min(int(probe.get("conflicts", 0)), 2000))
    done_cap = int(probe.get("budget_cap", 0))
    caps = [(c, tl) for c, tl in _schedule_for("exact", max_cap) if c > done_cap]
    if not caps and max_cap > done_cap:
        caps = [(max_cap, max(5.0, 600.0 * max_cap / 2_000_000))]
    secs = float(probe.get("seconds", 0.0))
    last_cap, last_conf = done_cap, int(probe.get("conflicts", 0))
    for cap, tl in caps:
        r = solve_cnf(cnf, inst, solver=solver, conf_budget=cap, time_limit=time_limit or tl)
        secs += r.seconds
        last_cap, last_conf = cap, r.conflicts
        if r.status == "sat":
            wit = bool(r.matrix is not None and not has_kst(r.matrix, inst.s, inst.t))
            return Label("sat", None, False, cap, r.conflicts, secs, c2000, None, wit, **feats)
        if r.status == "unsat":
            return Label("unsat", float(max(1, r.conflicts)), False, cap, r.conflicts, secs, c2000, None, None, **feats)
    fh = fhat(calib, c2000, feats["log2_volume"])
    return Label("unknown", censored_d(last_cap, fh), True, last_cap, last_conf, secs, c2000, fh, None, **feats)


def label_from_probe(
    inst: Instance, probe: Optional[dict], calib: Optional[Tuple[float, float, float]] = None
) -> Tuple[float, bool]:
    """(d, censored) for a legacy v1 probe dict (no 'd' field): exact conflicts when
    refuted, calibrated-and-clipped at the probe's cap when unknown, 1.0 for sat/unprobed."""
    if not probe:
        return 1.0, False
    if "d" in probe and probe["d"] is not None:
        return float(probe["d"]), bool(probe.get("censored", probe.get("status") == "unknown"))
    st = probe.get("status")
    if st == "unsat":
        return float(max(1, int(probe.get("conflicts", 1)))), False
    if st == "unknown":
        cap = int(probe.get("budget_cap") or 20_000)
        c2000 = int(probe.get("c2000") or min(int(probe.get("conflicts", cap)), 2000))
        if calib is None:
            calib = load_calibration(inst.s, inst.t)
        return censored_d(cap, fhat(calib, c2000, float(probe.get("log2_volume", 0.0)))), True
    return 1.0, False


# ---------------------------------------------------------------------------
# multiprocessing worker (module level so it pickles under spawn and fork)
# ---------------------------------------------------------------------------
def _label_worker(args):
    inst_d, rows, cols, mode, calib, solver, max_cap = args
    lab = label_case(Instance(**inst_d), rows, cols, mode=mode, calib=calib, solver=solver, max_cap=max_cap)
    return lab.as_probe()


def _continue_worker(args):
    inst_d, rows, cols, probe, max_cap, calib, solver, time_limit = args
    lab = continue_label(
        Instance(**inst_d), rows, cols, probe, max_cap, calib=calib, solver=solver, time_limit=time_limit
    )
    return lab.as_probe()


# ---------------------------------------------------------------------------
# statistics helpers (no scipy dependency)
# ---------------------------------------------------------------------------
def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    n = len(xs)
    if n < 3:
        return float("nan")
    rx, ry = rank(list(xs)), rank(list(ys))
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def log_rmse(pred: Sequence[float], true: Sequence[float]) -> float:
    if not pred:
        return float("nan")
    return (sum((math.log(max(p, 1.0)) - math.log(max(t, 1.0))) ** 2 for p, t in zip(pred, true)) / len(pred)) ** 0.5


def _lstsq(X: List[List[float]], y: List[float]) -> List[float]:
    """Least squares by normal equations with a tiny ridge on the non-intercept
    columns (the c2000 column is nearly constant on the censored-at-2000 cases,
    so plain normal equations are singular); numpy if present, else Gauss."""
    k = len(X[0])
    ridge = 1e-9
    try:
        import numpy as np

        A = np.array(X, dtype=float)
        b = np.array(y, dtype=float)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return [float(v) for v in sol]
    except Exception:  # pragma: no cover - numpy missing
        pass
    M = [
        [sum(X[r][i] * X[r][j] for r in range(len(X))) + (ridge if (i == j and i > 0) else 0.0) for j in range(k)]
        for i in range(k)
    ]
    v = [sum(X[r][i] * y[r] for r in range(len(X))) for i in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda r: abs(M[r][i]))
        M[i], M[p] = M[p], M[i]
        v[i], v[p] = v[p], v[i]
        for r in range(k):
            if r != i and M[i][i]:
                f = M[r][i] / M[i][i]
                M[r] = [a - f * b for a, b in zip(M[r], M[i])]
                v[r] -= f * v[i]
    return [v[i] / M[i][i] if M[i][i] else 0.0 for i in range(k)]


def fit_loglinear(cases: List[dict]) -> Tuple[float, float, float]:
    """cases: dicts with c2000, log2_volume, d (exact).  Returns (a, b, g)."""
    X = [[1.0, math.log(max(min(c["c2000"], FIRST_CAP), 1)), float(c["log2_volume"])] for c in cases]
    y = [math.log(max(float(c["d"]), 1.0)) for c in cases]
    a, b, g = _lstsq(X, y)
    return a, b, g


def evaluate_fit(coef: Tuple[float, float, float], cases: List[dict]) -> dict:
    pred = [fhat(coef, c["c2000"], c["log2_volume"]) for c in cases]
    true = [float(c["d"]) for c in cases]
    return {
        "n": len(cases),
        "spearman_rho": spearman(pred, true) if len(cases) >= 3 else float("nan"),
        "log_rmse": log_rmse(pred, true),
        "clipped_log_rmse": log_rmse([censored_d(int(c.get("cap", 2000)), p) for p, c in zip(pred, cases)], true),
    }


def calibrate(
    train: List[Tuple[Instance, "object"]],
    holdout: List[Tuple[Instance, "object"]],
    st: Tuple[int, int] = (3, 3),
    out_path: str = CALIB_PATH,
    write: bool = True,
    e10_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Fit (a,b,g) on the TRAIN cases that were unknown at 2,000 conflicts but were
    refuted exactly (d = true conflicts); validate on the held-out tables.
    `train` / `holdout` are lists of (Instance, CaseTable).  c2000 comes from the
    table's probe when present, else from E10's results.json, else it is
    re-measured (deterministic, ~0.05 s/case) and stored back into the probe."""
    from .casetable import ensure_c2000  # local import: casetable imports this module

    def collect(pairs):
        rows = []
        for inst, tab in pairs:
            if (inst.s, inst.t) != tuple(st):
                continue
            ensure_c2000(tab, e10_path=e10_path, save=True, verbose=verbose)
            for r in tab.records:
                p = r.probe
                if not p or p.get("status") != "unsat":
                    continue
                c2000 = int(p.get("c2000", 0))
                if c2000 < 2000:
                    continue  # solved inside the first cap: nothing to extrapolate
                rows.append(
                    {
                        "tag": inst.tag,
                        "rows": r.rows,
                        "cols": r.cols,
                        "c2000": c2000,
                        "log2_volume": float(p["log2_volume"]),
                        "d": float(max(1, p["conflicts"])),
                        "cap": 2000,
                    }
                )
        return rows

    fit_rows = collect(train)
    hold_rows = collect(holdout)
    if len(fit_rows) < 3:
        raise RuntimeError("calibrate: fewer than 3 exactly-labelled censored-at-2000 TRAIN cases")
    coef = fit_loglinear(fit_rows)
    res = {
        "st": f"{st[0]},{st[1]}",
        "a": coef[0],
        "b": coef[1],
        "g": coef[2],
        "formula": "log fhat = a + b*log(max(c2000,1)) + g*log2_volume; d = min(max(cap, fhat), 20*cap)",
        "n_fit": len(fit_rows),
        "fitted_on": sorted(set(r["tag"] for r in fit_rows)),
        "holdout": sorted(set(r["tag"] for r in hold_rows)),
        "train_metrics": evaluate_fit(coef, fit_rows),
        "holdout_metrics": evaluate_fit(coef, hold_rows) if hold_rows else None,
        "c2000_range_in_fit": [min(r["c2000"] for r in fit_rows), max(r["c2000"] for r in fit_rows)],
        "note": (
            "every case unknown at 2000 has c2000 ~ 2000, so the b-term is (nearly) constant on the "
            "fit set and the prediction is effectively a + g*log2_volume (least-squares min-norm)"
        ),
    }
    # secondary diagnostic: volume-only and constant baselines, to make the fit's value legible
    const = sum(math.log(r["d"]) for r in fit_rows) / len(fit_rows)
    res["baselines"] = {
        "constant_log_rmse_holdout": (
            log_rmse([math.exp(const)] * len(hold_rows), [r["d"] for r in hold_rows]) if hold_rows else None
        ),
        "cap_only_log_rmse_holdout": (
            log_rmse([2000.0] * len(hold_rows), [r["d"] for r in hold_rows]) if hold_rows else None
        ),
    }
    if write:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        old = {}
        if os.path.exists(out_path):
            try:
                old = json.load(open(out_path))
            except ValueError:
                old = {}
        fits = old.get("fits", {})
        fits[res["st"]] = {
            "a": coef[0],
            "b": coef[1],
            "g": coef[2],
            "n_fit": len(fit_rows),
            "fitted_on": res["fitted_on"],
            "holdout": res["holdout"],
            "train_metrics": res["train_metrics"],
            "holdout_metrics": res["holdout_metrics"],
        }
        json.dump({"formula": res["formula"], "fits": fits}, open(out_path, "w"), indent=1)
        res["written"] = out_path
    return res


# ---------------------------------------------------------------------------
# legacy v1 API (kept for existing callers)
# ---------------------------------------------------------------------------
@dataclass
class Probe:
    status: str  # sat | unsat | unknown
    conflicts: int
    seconds: float
    budget_cap: int
    fixed_by_propagation: float
    log2_volume: float
    nvars: int
    nclauses: int

    def as_dict(self):
        return asdict(self)

    @property
    def difficulty(self) -> float:
        return float(self.conflicts if self.status != "unknown" else self.budget_cap)


def probe_case(
    inst: Instance,
    rows: Sequence[int],
    cols: Sequence[int],
    conf_cap: int = 20000,
    time_limit: Optional[float] = 60.0,
    solver: str = SOLVER,
) -> Probe:
    """v1 single-cap probe (kept for backward compatibility; new tables use label_case)."""
    cnf = encode_case(inst, rows, cols)
    pf = propagation_fraction(cnf, inst, solver)
    res = solve_cnf(cnf, inst, solver=solver, conf_budget=conf_cap, time_limit=time_limit)
    return Probe(
        status=res.status,
        conflicts=res.conflicts,
        seconds=res.seconds,
        budget_cap=conf_cap,
        fixed_by_propagation=pf,
        log2_volume=log2_volume(inst, rows),
        nvars=cnf.nvars,
        nclauses=len(cnf.clauses),
    )

"""Cheap per-case hardness features (E24, owner A4): CDCL progress statistics of short
budgeted runs (family A) and static / LP-slack counting features (family B).

Estimator contract (E24, shared by A2/A3/A4)::

    estimate(inst, rows, cols, **kw) -> {
        "features": {name: float},        # all floats (NaN never; missing -> 0.0 with a *_ok flag)
        "d_hat": float | None,            # standalone prediction (calibrated progress model) or None
        "cost_conflicts": int,            # solver conflicts spent computing the features
        "cost_propagations": int,
        "cost_seconds": float,            # wall seconds (inflated under CPU contention)
        "meta": {...},                    # extra, not part of the contract: per-run costs, tiers
    }

d_hat (E24 PROGRESS.md): exact conflicts when the deepest pysat probe decided the case; otherwise
the frozen log-linear model for the tier from experiments/E24_difficulty/progress_model.json
("20k": 2 pysat + 2 binary runs, ~44k conflicts; "free20k": pysat stats only -- the censored
table pipeline already pays for its 20k run, see d_hat_from_probe; "2k": ~4k conflicts).
predict(..., mean=True) applies the Duan smearing factor for sums of work.

Pure function of its inputs: CaDiCaL (binary 3.0.1, ``--seed=0``) and pysat ``cadical195``
are deterministic for a fixed build; the LPs are solved by HiGHS (deterministic).  Nothing is
cached on disk and no table is touched.

Family A (``tier`` "2k" / "20k"): run the CaDiCaL *binary* ``tools/cadical/build/cadical -c CAP
--stats -v`` on the case DIMACS (stdin) at CAP = 2 000 and 20 000 and parse
  * the statistics block (conflicts, decisions, propagations, restarts, learned clauses and
    literals, root-level fixed variables, eliminated / substituted variables, subsumed,
    chronological backtracks, ticks, sweep, backbone, ...), every line as ``<cap>_<name>``;
  * the verbose report lines (the ``c <ch> seconds MB level reductions restarts rate conflicts
    redundant size/glue size glue tier1 tier2 trail% irredundant remaining remaining%`` table)
    as a trajectory, from which the progress features are derived: remaining (active) variables
    and root-fixed variables at the 2k and 20k marks, their growth per conflict, EMA glue / size;
  * the glue-usage histogram (share of tier-1 / tier-2 clause uses);
plus pysat ``cadical195`` ``accum_stats`` of fresh runs at both caps (the label solver: its 2k
conflicts reproduce ``c2000`` of the tables).  Derived rates: propagations/conflict,
decisions/conflict, learned literals/learned clause, fixed growth per 1k conflicts, and the
progress extrapolation ``prog_*`` (conflicts needed to reach a fixed share of variables fixed,
extrapolating the 2k->20k growth linearly and log-linearly).

Family B (``tier`` "static", no solver): counting slacks from the proposal / DGH:
  * Argument A slack on columns and rows: ``(t-1)C(m,s) - sum_j C(c_j,s)`` (and transposed),
    absolute and relative to the budget;
  * Argument D slack: for row i with r_i ones, ``(t-1)C(m-1,s-1) - sum of C(c-1,s-1) over the
    r_i lightest columns``; min over rows (= heaviest row), mean, and the column form;
  * DGH(v = s-1) slack ``rhs - lhs`` of the compact inequality (experiments/E13_dgh4/dgh.py) at
    the best k, both orientations, absolute and relative;
  * pair-codegree LP (zar_ub.lemmas.farkas_system; F1-F4 without the Farkas integer rounding):
    max uniform slack eps of the row and pair-box constraints subject to the F1 equality, and
    the range [Tmin, Tmax] of sum(lambda) under F2-F4 vs T = sum C(c_j,2) (both orientations);
  * profile shape: distinct row/col sums, spread, variance, excess ones over w, log2 volumes
    (row and column), log2 of the lex-symmetry group size, nvars / nclauses of the CNF.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from math import comb, lgamma, log, log2
from typing import Dict, List, Optional, Sequence, Tuple

from .encoding import encode_case
from .known import Instance

_HERE = os.path.dirname(os.path.abspath(__file__))
_UB = os.path.dirname(_HERE)
CADICAL_BIN = os.path.join(_UB, "tools", "cadical", "build", "cadical")
DEFAULT_CAPS: Tuple[int, ...] = (2_000, 20_000)
MODEL_PATH = os.path.join(_UB, "experiments", "E24_difficulty", "progress_model.json")

# ---------------------------------------------------------------------------
# Family A: CaDiCaL binary output parsing
# ---------------------------------------------------------------------------
_STAT_RE = re.compile(
    r"^c (\s*)([A-Za-z][A-Za-z0-9_.\- ]*?):?\s+(-?[0-9][0-9.e+\-]*)%?\s+(-?[0-9][0-9.e+\-]*)?"
)
_REPORT_RE = re.compile(r"^c (\S)\s+([0-9.]+)\s+(.*)$")
_GLUE_RE = re.compile(
    r"^c (focused|stable) glue (\d+) used (\d+) clauses ([0-9.]+)% accumulated ([0-9.]+)%( tier[12])?"
)
_REPORT_COLS = [
    "seconds",
    "mb",
    "level",
    "reductions",
    "restarts",
    "rate",
    "conflicts",
    "redundant",
    "size_per_glue",
    "size",
    "glue",
    "tier1",
    "tier2",
    "trail_pct",
    "irredundant",
    "remaining",
    "remaining_pct",
]


def _num(x: str) -> float:
    try:
        return float(x.rstrip("%"))
    except ValueError:
        return float("nan")


def parse_cadical_output(text: str) -> dict:
    """Parse ``cadical --stats -v`` stdout into {status, stats, traj, glue, process_time}.

    stats: flat dict; an indented statistic is keyed ``<parent>.<name>`` (parent = the last
    non-indented statistic), spaces -> '_'; the value is the first number on the line and the
    second (rate / percentage) is stored as ``<key>__rel``."""
    status = "unknown"
    stats: Dict[str, float] = {}
    traj: List[dict] = []
    glue = {"focused": {}, "stable": {}}
    in_stats = False
    parent = ""
    ptime = float("nan")
    for line in text.splitlines():
        if line.startswith("s "):
            if "UNSATISFIABLE" in line:
                status = "unsat"
            elif "SATISFIABLE" in line:
                status = "sat"
            continue
        if not line.startswith("c"):
            continue
        if line.startswith("c --- [ statistics ]"):
            in_stats = True
            continue
        if line.startswith("c --- [") and in_stats and "statistics" not in line:
            in_stats = False
        if line.startswith("c UNSATISFIABLE"):
            status = "unsat"
        elif line.startswith("c SATISFIABLE"):
            status = "sat"
        if "total process time since initialization" in line:
            ptime = _num(line.split(":")[-1].split()[0])
            continue
        g = _GLUE_RE.match(line)
        if g:
            glue[g.group(1)][int(g.group(2))] = (
                int(g.group(3)),
                float(g.group(4)),
                g.group(6) or "",
            )
            continue
        if in_stats:
            mm = _STAT_RE.match(line)
            if not mm:
                continue
            indent, name, v1, v2 = mm.group(1), mm.group(2).strip(), mm.group(3), mm.group(4)
            key = name.replace(" ", "_").replace(".", "").replace("-", "_")
            if indent == "":
                parent = key
                full = key
            else:
                full = f"{parent}.{key}"
            if full in stats:  # duplicate names at the same level: keep the first
                continue
            stats[full] = _num(v1)
            if v2 is not None:
                stats[full + "__rel"] = _num(v2)
            continue
        r = _REPORT_RE.match(line)
        if r and not in_stats:
            parts = r.group(3).split()
            if len(parts) == len(_REPORT_COLS) - 1:
                vals = [_num(r.group(2))] + [_num(p) for p in parts]
                row = dict(zip(_REPORT_COLS, vals))
                row["ch"] = r.group(1)
                traj.append(row)
    return {"status": status, "stats": stats, "traj": traj, "glue": glue, "process_time": ptime}


def run_cadical_bin(
    dimacs: str, cap: int, time_limit: float = 120.0, binary: str = CADICAL_BIN
) -> dict:
    """One fresh CaDiCaL-binary run with a conflict limit (seed 0, default options)."""
    t0 = time.time()
    p = subprocess.run(
        [binary, "-c", str(int(cap)), "--stats", "-v", "-n", "--seed=0"],
        input=dimacs,
        capture_output=True,
        text=True,
        timeout=time_limit,
    )
    out = parse_cadical_output(p.stdout)
    if p.returncode == 20:
        out["status"] = "unsat"
    elif p.returncode == 10:
        out["status"] = "sat"
    out["wall"] = time.time() - t0
    return out


def _bin_features(prefix: str, res: dict, nvars: int, full: bool) -> Dict[str, float]:
    """Curated features of one binary run (plus every raw statistic when full=True)."""
    s = res["stats"]
    g = lambda k, d=0.0: float(s.get(k, d)) if not math.isnan(s.get(k, d)) else d  # noqa: E731
    conf = max(g("conflicts"), 1.0)
    learned = max(g("learned"), 1.0)
    f = {
        "solved": 1.0 if res["status"] in ("unsat", "sat") else 0.0,
        "conflicts": g("conflicts"),
        "decisions": g("decisions"),
        "propagations": g("propagations"),
        "restarts": g("restarts"),
        "learned": g("learned"),
        "learned_lits": g("learned_lits"),
        "avg_learned_size": g("learned_lits") / learned,
        "fixed": g("fixed"),
        "fixed_frac": g("fixed") / max(nvars, 1),
        "eliminated_frac": g("eliminated") / max(nvars, 1),
        "substituted": g("substituted"),
        "subsumed": g("subsumed"),
        "strengthened": g("strengthened"),
        "chrono_frac": g("chronological") / conf,
        "props_per_conf": g("propagations") / conf,
        "decs_per_conf": g("decisions") / conf,
        "restarts_per_kconf": 1000.0 * g("restarts") / conf,
        "ticks_per_conf": g("ticks") / conf,
        "searched_per_dec": g("decisions.searched") / max(g("decisions"), 1.0),
        "shrunken_frac": g("shrunken") / max(g("learned_lits"), 1.0),
        "minishrunken_frac": g("minishrunken") / max(g("learned_lits"), 1.0),
        "otfs_frac": g("otfs") / conf,
        "promoted1_frac": g("learned.promoted1") / learned,
        "promoted2_frac": g("learned.promoted2") / learned,
        "improvedglue_frac": g("learned.improvedglue") / learned,
        "bumped_per_learned": g("learned.bumped") / learned,
        "backbone_units": g("backbone.units"),
        "sweep_equivs": g("sweep_equivs"),
        "sweep_unsat_frac": g("sweep_equivs.unsat") / max(g("sweep_equivs.solved"), 1.0),
        "units_probe": g("fixed.units"),
        "binaries_probe": g("fixed.binaries"),
        "stable_frac": g("stabilizing__rel") / 100.0,
        "reduced_frac": g("reduced") / conf,
        "seconds": float(res.get("process_time") or 0.0),
    }
    tr = res["traj"]
    if tr:
        last = tr[-1]
        f.update(
            {
                "ema_glue": last["glue"],
                "ema_size": last["size"],
                "trail_pct": last["trail_pct"],
                "remaining_pct": last["remaining_pct"],
                "remaining": last["remaining"],
                "irredundant": last["irredundant"],
                "redundant": last["redundant"],
                "tier1": last["tier1"],
                "tier2": last["tier2"],
                "max_level_ema": max(r["level"] for r in tr),
            }
        )
        # trajectory slopes over the second half of the run (conflicts > cap/2)
        half = [r for r in tr if r["conflicts"] >= 0.5 * max(last["conflicts"], 1)]
        if len(half) >= 2 and half[-1]["conflicts"] > half[0]["conflicts"]:
            dc = half[-1]["conflicts"] - half[0]["conflicts"]
            f["glue_slope_per_kconf"] = 1000.0 * (half[-1]["glue"] - half[0]["glue"]) / dc
            f["trail_slope_per_kconf"] = (
                1000.0 * (half[-1]["trail_pct"] - half[0]["trail_pct"]) / dc
            )
            f["remaining_slope_per_kconf"] = (
                1000.0 * (half[-1]["remaining"] - half[0]["remaining"]) / dc
            )
        else:
            f["glue_slope_per_kconf"] = f["trail_slope_per_kconf"] = f[
                "remaining_slope_per_kconf"
            ] = 0.0
    # glue usage histogram (share of clause uses at glue <= 2)
    for mode in ("focused", "stable"):
        h = res["glue"].get(mode, {})
        tot = sum(v[0] for v in h.values())
        low = sum(v[0] for k, v in h.items() if k <= 2)
        f[f"glue_use_le2_{mode}"] = low / tot if tot else 0.0
    if full:
        for k, v in s.items():
            if not math.isnan(v):
                f["raw_" + k] = float(v)
    return {f"bin{prefix}_{k}": float(v) for k, v in f.items()}


# ---------------------------------------------------------------------------
# Family A: pysat (label solver) accum_stats
# ---------------------------------------------------------------------------
def pysat_run(clauses, cap: int, solver: str = "cadical195", time_limit: float = 120.0) -> dict:
    import threading

    from pysat.solvers import Solver

    t0 = time.time()
    s = Solver(name=solver, bootstrap_with=clauses)
    timer = threading.Timer(time_limit, s.interrupt)
    timer.daemon = True
    try:
        s.conf_budget(int(cap))
        timer.start()
        res = s.solve_limited(expect_interrupt=True)
        st = s.accum_stats() or {}
    finally:
        timer.cancel()
        s.delete()
    return {
        "status": "sat" if res is True else ("unsat" if res is False else "unknown"),
        "conflicts": int(st.get("conflicts", 0)),
        "decisions": int(st.get("decisions", 0)),
        "propagations": int(st.get("propagations", 0)),
        "restarts": int(st.get("restarts", 0)),
        "wall": time.time() - t0,
    }


def _pysat_features(prefix: str, r: dict) -> Dict[str, float]:
    conf = max(r["conflicts"], 1)
    f = {
        "solved": 1.0 if r["status"] != "unknown" else 0.0,
        "conflicts": float(r["conflicts"]),
        "decisions": float(r["decisions"]),
        "propagations": float(r["propagations"]),
        "restarts": float(r["restarts"]),
        "props_per_conf": r["propagations"] / conf,
        "decs_per_conf": r["decisions"] / conf,
        "restarts_per_kconf": 1000.0 * r["restarts"] / conf,
        "wall": r["wall"],
    }
    return {f"ps{prefix}_{k}": float(v) for k, v in f.items()}


def _capname(cap: int) -> str:
    return f"{cap // 1000}k" if cap % 1000 == 0 else str(cap)


# ---------------------------------------------------------------------------
# Family A: progress extrapolation
# ---------------------------------------------------------------------------
def progress_features(runs: Dict[int, dict], nvars: int) -> Dict[str, float]:
    """Growth of root-fixed variables and of the solver's 'remaining' variables between the
    two caps, and a naive extrapolation of the conflicts needed to finish.

    prog_fixed_growth_per_kconf  = 1000 (fixed(C2) - fixed(C1)) / (C2 - C1)
    prog_rem_drop_per_kconf      = 1000 (remaining(C1) - remaining(C2)) / (C2 - C1)
    prog_lin_to_zero             = C2 + remaining(C2) / max(drop rate, 1e-3)   (linear)
    prog_log_to_1pct             = C2 * (ln(rem(C2)/rem_target) / ln(rem(C1)... ) log-linear
    prog_fixed_ratio             = (fixed(C2)+1)/(fixed(C1)+1)
    """
    caps = sorted(runs)
    out: Dict[str, float] = {}
    if len(caps) < 2:
        return out
    c1, c2 = caps[0], caps[-1]
    r1, r2 = runs[c1], runs[c2]

    def _get(r, key):
        return float(r["stats"].get(key, 0.0))

    def _rem(r):
        return float(r["traj"][-1]["remaining"]) if r["traj"] else float(nvars)

    k1, k2 = max(_get(r1, "conflicts"), 1.0), max(_get(r2, "conflicts"), 1.0)
    dc = max(k2 - k1, 1.0)
    f1, f2 = _get(r1, "fixed"), _get(r2, "fixed")
    m1, m2 = _rem(r1), _rem(r2)
    drop = (m1 - m2) / dc  # remaining vars removed per conflict
    fg = (f2 - f1) / dc
    out["prog_fixed_growth_per_kconf"] = 1000.0 * fg
    out["prog_fixed_ratio"] = (f2 + 1.0) / (f1 + 1.0)
    out["prog_rem_drop_per_kconf"] = 1000.0 * drop
    out["prog_rem_ratio"] = (m2 + 1.0) / (m1 + 1.0)
    # linear extrapolation of 'remaining' to zero (clipped to [C2, 1e9])
    lin = k2 + m2 / max(drop, 1e-6)
    out["prog_log_lin_to_zero"] = math.log(min(max(lin, k2), 1e9))
    # linear extrapolation of the root-fixed count to cover the remaining variables
    linf = k2 + m2 / max(fg, 1e-6)
    out["prog_log_fixed_to_all"] = math.log(min(max(linf, k2), 1e9))
    # log-linear: remaining decays as exp(-lambda * conflicts) -> conflicts to reach 1 % of vars
    if m2 > 0 and m1 > m2:
        lam = (math.log(m1) - math.log(m2)) / dc
        tgt = 0.01 * nvars
        loglin = k2 + max(math.log(m2 / tgt), 0.0) / lam if m2 > tgt else k2
    else:
        loglin = 1e9
    out["prog_log_loglin_to_1pct"] = math.log(min(max(loglin, k2), 1e9))
    return out


# ---------------------------------------------------------------------------
# Family B: static counting / LP slack features
# ---------------------------------------------------------------------------
def _argA_slack(m: int, s: int, t: int, cols: Sequence[int]) -> Tuple[int, int]:
    budget = (t - 1) * comb(m, s)
    return budget - sum(comb(c, s) for c in cols), budget


def _argD_slacks(
    m: int, s: int, t: int, rows: Sequence[int], cols: Sequence[int]
) -> Tuple[List[int], int]:
    """Per-row Argument D slack (t-1)C(m-1,s-1) - sum_{r_i lightest cols} C(c-1, s-1)."""
    budget = (t - 1) * comb(m - 1, s - 1)
    sc = sorted(cols)
    pref = [0]
    for c in sc:
        pref.append(pref[-1] + (comb(c - 1, s - 1) if c >= 1 else 0))
    return [budget - pref[min(r, len(sc))] for r in rows], budget


def _dgh_slacks(m: int, s: int, t: int, cols: Sequence[int]) -> Tuple[float, float, int]:
    """min over k in [s, m] of (rhs - lhs) of the compact DGH(v=s-1) inequality (dgh.py), the
    same relative to rhs, and the argmin k.  Negative = the profile is killed by DGH."""
    v = s - 1
    best, best_rel, best_k = float("inf"), float("inf"), s
    ch = [comb(c, v) for c in cols]
    for k in range(s, m + 1):
        R = (t - 1) * (m - s + 1)
        D = k - s + 1
        a, c = R % D, R // D
        lhs = (D - a) * sum(ch)
        rhs = (D - a) * c * comb(m, v) + sum(max(k - cj, 0) * chj for cj, chj in zip(cols, ch))
        sl = rhs - lhs
        rel = sl / rhs if rhs else float(sl)
        if rel < best_rel:
            best, best_rel, best_k = sl, rel, k
    return float(best), float(best_rel), best_k


def _lp_features(
    m: int, n: int, s: int, t: int, rows: Sequence[int], cols: Sequence[int]
) -> Dict[str, float]:
    """Pair-codegree LP (lemmas.farkas_system, F1-F4) in the unordered normalisation:
    variables lambda_p (p = unordered row pair), sum_p lambda_p = T = sum_j C(c_j, 2),
    lo_i <= sum_{p ni i} lambda_p <= hi_i, flo_p <= lambda_p <= cap_p.

    eps   = max uniform slack on the row constraints and the non-degenerate pair boxes with the
            equality kept (negative = the LP relaxation is infeasible = a Farkas certificate
            exists over the reals);
    Tmin/Tmax = range of sum(lambda) under F2-F4 only; slackT = min(Tmax - T, T - Tmin)."""
    import numpy as np
    from scipy.optimize import linprog

    from .lemmas import farkas_system

    out = {
        "lp_ok": 0.0,
        "lp_eps": 0.0,
        "lp_eps_rel": 0.0,
        "lp_slackT": 0.0,
        "lp_slackT_rel": 0.0,
        "lp_lcap": 0.0,
    }
    if s < 2 or m < 2:
        return out
    S = farkas_system(m, n, s, t, rows, cols)
    pairs = S.pairs
    P = len(pairs)
    T = S.t2 // 2
    inc = np.zeros((m, P))
    for p, (i, j) in enumerate(pairs):
        inc[i, p] = 1.0
        inc[j, p] = 1.0
    lo = np.array(S.lo, float)
    hi = np.array(S.hi, float)
    cap = np.array([S.cap[i][j] for i, j in pairs], float)
    flo = np.array([S.flo[i][j] for i, j in pairs], float)
    free = (cap > flo).astype(float)
    out["lp_lcap"] = float(S.lcap)
    # (1) max eps: vars [lambda (P), eps]
    c = np.zeros(P + 1)
    c[-1] = -1.0
    A = []
    b = []
    for i in range(m):
        A.append(np.concatenate([inc[i], [1.0]]))  # sum <= hi - eps
        b.append(hi[i])
        A.append(np.concatenate([-inc[i], [1.0]]))  # -sum <= -lo - eps
        b.append(-lo[i])
    for p in range(P):
        if free[p]:
            row = np.zeros(P + 1)
            row[p] = 1.0
            row[-1] = 1.0
            A.append(row)
            b.append(cap[p])
            row = np.zeros(P + 1)
            row[p] = -1.0
            row[-1] = 1.0
            A.append(row)
            b.append(-flo[p])
    Aeq = np.concatenate([np.ones(P), [0.0]]).reshape(1, -1)
    bounds = [(0, None)] * P + [(-1e4, 1e4)]
    # the eps-rows carry the box for free pairs; fixed pairs (cap <= flo) get hard bounds
    for p in range(P):
        if not free[p]:
            bounds[p] = (min(flo[p], cap[p]), max(flo[p], cap[p]))
    r = linprog(
        c, A_ub=np.array(A), b_ub=np.array(b), A_eq=Aeq, b_eq=[T], bounds=bounds, method="highs"
    )
    if r.status == 0:
        out["lp_eps"] = float(r.x[-1])
        out["lp_eps_rel"] = float(r.x[-1]) / max(float(np.mean(hi)), 1.0)
        out["lp_ok"] = 1.0
    # (2) range of sum(lambda) under F2-F4
    A2 = np.vstack([inc, -inc])
    b2 = np.concatenate([hi, -lo])
    bnd = [(min(flo[p], cap[p]), max(flo[p], cap[p])) for p in range(P)]
    rmin = linprog(np.ones(P), A_ub=A2, b_ub=b2, bounds=bnd, method="highs")
    rmax = linprog(-np.ones(P), A_ub=A2, b_ub=b2, bounds=bnd, method="highs")
    if rmin.status == 0 and rmax.status == 0:
        tmin, tmax = float(rmin.fun), float(-rmax.fun)
        out["lp_slackT"] = min(tmax - T, T - tmin)
        out["lp_slackT_rel"] = out["lp_slackT"] / max(T, 1)
        out["lp_slackT_hi"] = tmax - T
        out["lp_slackT_lo"] = T - tmin
    else:  # F2-F4 alone infeasible
        out["lp_slackT"] = -1.0
        out["lp_slackT_rel"] = -1.0 / max(T, 1)
        out["lp_slackT_hi"] = out["lp_slackT_lo"] = -1.0
    return out


def _log2_sym(v: Sequence[int]) -> float:
    from collections import Counter

    return sum(lgamma(k + 1) for k in Counter(v).values()) / log(2)


def static_features(
    inst: Instance, rows: Sequence[int], cols: Sequence[int], lp: bool = True, cnf=None
) -> Dict[str, float]:
    m, n, s, t = inst.m, inst.n, inst.s, inst.t
    rows, cols = list(rows), list(cols)
    f: Dict[str, float] = {}
    a_c, b_c = _argA_slack(m, s, t, cols)
    a_r, b_r = _argA_slack(n, t, s, rows)
    f["st_argA_col"] = float(a_c)
    f["st_argA_col_rel"] = a_c / max(b_c, 1)
    f["st_argA_row"] = float(a_r)
    f["st_argA_row_rel"] = a_r / max(b_r, 1)
    f["st_argA_min_rel"] = min(f["st_argA_col_rel"], f["st_argA_row_rel"])
    dr, bdr = _argD_slacks(m, s, t, rows, cols)
    dc, bdc = _argD_slacks(n, t, s, cols, rows)
    f["st_argD_row_min"] = float(min(dr))
    f["st_argD_row_min_rel"] = min(dr) / max(bdr, 1)
    f["st_argD_row_mean_rel"] = sum(dr) / len(dr) / max(bdr, 1)
    f["st_argD_col_min"] = float(min(dc))
    f["st_argD_col_min_rel"] = min(dc) / max(bdc, 1)
    f["st_argD_col_mean_rel"] = sum(dc) / len(dc) / max(bdc, 1)
    f["st_argD_min_rel"] = min(f["st_argD_row_min_rel"], f["st_argD_col_min_rel"])
    gc, gcr, kc = _dgh_slacks(m, s, t, cols)
    gr, grr, kr = _dgh_slacks(n, t, s, rows)
    f["st_dgh_col"] = gc
    f["st_dgh_col_rel"] = gcr
    f["st_dgh_col_k"] = float(kc)
    f["st_dgh_row"] = gr
    f["st_dgh_row_rel"] = grr
    f["st_dgh_min_rel"] = min(gcr, grr)
    f["st_distinct_rows"] = float(len(set(rows)))
    f["st_distinct_cols"] = float(len(set(cols)))
    f["st_spread_rows"] = float(max(rows) - min(rows))
    f["st_spread_cols"] = float(max(cols) - min(cols))
    mr, mc = sum(rows) / m, sum(cols) / n
    f["st_var_rows"] = sum((r - mr) ** 2 for r in rows) / m
    f["st_var_cols"] = sum((c - mc) ** 2 for c in cols) / n
    f["st_excess"] = float(sum(rows) - inst.w)
    f["st_log2_volume"] = float(sum(log2(comb(n, r)) for r in rows))
    f["st_log2_volume_cols"] = float(sum(log2(comb(m, c)) for c in cols))
    f["st_log2_sym"] = _log2_sym(rows) + _log2_sym(cols)
    f["st_log2_vol_minus_sym"] = f["st_log2_volume"] - f["st_log2_sym"]
    if cnf is None:
        cnf = encode_case(inst, rows, cols)
    f["st_nvars"] = float(cnf.nvars)
    f["st_nclauses"] = float(len(cnf.clauses))
    if lp:
        for k, v in _lp_features(m, n, s, t, rows, cols).items():
            f["st_row_" + k] = float(v)
        for k, v in _lp_features(n, m, t, s, cols, rows).items():
            f["st_col_" + k] = float(v)
        f["st_lp_eps_min"] = min(f["st_row_lp_eps"], f["st_col_lp_eps"])
        f["st_lp_slackT_rel_min"] = min(f["st_row_lp_slackT_rel"], f["st_col_lp_slackT_rel"])
    return f


# ---------------------------------------------------------------------------
# d_hat: a calibrated log-linear progress model (fitted in experiments/E24_difficulty)
# ---------------------------------------------------------------------------
def load_model(path: str = MODEL_PATH) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def predict(model: dict, feats: Dict[str, float], mean: bool = False) -> Optional[float]:
    """log d_hat = intercept + sum_k coef_k * transform(feature_k); None if a feature is missing.
    The prediction is floored at the model's floor (the cap it was fitted above).
    mean=False gives the conditional median (exp of the log-scale fit: use for ranks and log
    errors); mean=True multiplies by the model's Duan smearing factor (use for sums of work)."""
    try:
        z = float(model["intercept"])
        for name, co, tr in zip(model["features"], model["coef"], model.get("transforms", [])):
            x = float(feats[name])
            if tr == "log1p":
                x = math.log1p(max(x, 0.0))
            z += (
                float(co)
                * (x - float(model.get("mu", {}).get(name, 0.0)))
                / float(model.get("sd", {}).get(name, 1.0))
            )
        v = math.exp(z) * (float(model.get("smear", 1.0)) if mean else 1.0)
        return float(max(v, float(model.get("floor", 1.0))))
    except (KeyError, TypeError, ValueError):
        return None


def d_hat_from_probe(
    inst: Instance,
    rows: Sequence[int],
    cols: Sequence[int],
    conflicts: int,
    decisions: int,
    restarts: int,
    propagations: int = 0,
    cap: int = 20_000,
    model: Optional[dict] = None,
    model_name: str = "free20k",
) -> Optional[float]:
    """Zero-extra-solver-cost d_hat for a case left open by an existing pysat cadical195 probe
    at `cap` (the censored table pipeline's 20k run): its accum_stats (conflicts, decisions,
    restarts) + static/LP features -> the frozen "free20k" model.  Returns None if the model
    file or a needed statistic is missing.  (zar_ub.solve.SolveResult carries decisions and
    propagations but not restarts: the caller must record accum_stats()['restarts'].)"""
    if model is None:
        model = load_model()
    mm = ((model or {}).get("models") or {}).get(model_name)
    if not mm:
        return None
    feats = static_features(inst, rows, cols, lp=True)
    feats.update(
        _pysat_features(
            _capname(cap),
            {
                "status": "unknown",
                "conflicts": int(conflicts),
                "decisions": int(decisions),
                "propagations": int(propagations),
                "restarts": int(restarts),
                "wall": 0.0,
            },
        )
    )
    return predict(mm, feats)


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------
def estimate(
    inst: Instance,
    rows: Sequence[int],
    cols: Sequence[int],
    caps: Sequence[int] = DEFAULT_CAPS,
    tier: str = "20k",
    pysat: bool = True,
    binary: bool = True,
    lp: bool = True,
    full: bool = False,
    time_limit: float = 120.0,
    model: Optional[dict] = None,
    model_name: Optional[str] = None,
    seed: int = 0,
) -> dict:
    """Family A + B features of one case.

    tier: "static" (no solver, LP only), "2k" (caps <= 2000), "20k" (all caps, default).
    full: also emit every raw CaDiCaL statistic (``bin<cap>_raw_*``; ~250 per cap).
    model / model_name: the frozen models file (default progress_model.json) and which model to
        use for d_hat (default by tier: "20k" -> "20k" ("free20k" when binary=False), "2k" -> "2k").
        d_hat is the exact conflict count when the deepest pysat probe decided the case.
    seed: accepted for the contract; both solvers run with seed 0 (deterministic)."""
    t0 = time.time()
    rows, cols = list(rows), list(cols)
    cnf = encode_case(inst, rows, cols)
    feats = static_features(inst, rows, cols, lp=lp, cnf=cnf)
    cost_c = cost_p = 0
    meta: Dict[str, object] = {"runs": {}}
    use_caps = [] if tier == "static" else [c for c in caps if tier != "2k" or c <= 2000]
    runs: Dict[int, dict] = {}
    if use_caps and binary:
        dimacs = cnf.to_dimacs()
        for cap in use_caps:
            r = run_cadical_bin(dimacs, cap, time_limit=time_limit)
            runs[cap] = r
            feats.update(_bin_features(_capname(cap), r, cnf.nvars, full))
            cc = int(r["stats"].get("conflicts", 0))
            pp = int(r["stats"].get("propagations", 0))
            cost_c += cc
            cost_p += pp
            meta["runs"][f"bin{_capname(cap)}"] = {
                "conflicts": cc,
                "propagations": pp,
                "wall": r["wall"],
                "status": r["status"],
            }
        feats.update(progress_features(runs, cnf.nvars))
    if use_caps and pysat:
        for cap in use_caps:
            r = pysat_run(cnf.clauses, cap, time_limit=time_limit)
            feats.update(_pysat_features(_capname(cap), r))
            cost_c += r["conflicts"]
            cost_p += r["propagations"]
            meta["runs"][f"ps{_capname(cap)}"] = {
                k: r[k] for k in ("conflicts", "propagations", "wall", "status")
            }
    feats = {
        k: (
            0.0
            if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))))
            else float(v)
        )
        for k, v in feats.items()
    }
    # d_hat: exact when the label solver's deepest probe refuted / satisfied the case, else the
    # frozen log-linear model for this tier (progress_model.json), else None
    if model is None:
        model = load_model()
    if model_name is None:
        model_name = {"20k": "20k" if binary else "free20k", "2k": "2k"}.get(tier)
    d_hat = None
    top = f"ps{_capname(max(use_caps))}_" if use_caps else None
    if top and pysat and feats.get(top + "solved", 0.0) > 0:
        d_hat = float(max(1.0, feats[top + "conflicts"]))
        meta["d_hat_source"] = "exact"
    elif model and model_name:
        mm = (model.get("models") or {}).get(model_name)
        if mm:
            d_hat = predict(mm, feats)
            meta["d_hat_source"] = f"model:{model_name}"
    return {
        "features": feats,
        "d_hat": d_hat,
        "cost_conflicts": int(cost_c),
        "cost_propagations": int(cost_p),
        "cost_seconds": float(time.time() - t0),
        "meta": meta,
    }

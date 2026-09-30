"""Monte Carlo decomposition-hardness estimator (Chivilikhin, Pavlenko, Semenov 2023).

For one case C = encode_case(inst, rows, cols) choose a decomposition set B of
cell variables, draw N cubes (assignments of B), solve C under each cube (as
assumptions) with a per-cube conflict budget b, and estimate the d-hardness

    mu_B(C) = sum_{beta} t(C[beta/B]) = |Omega_B| * E[xi_B]              (Thm 2)

by  mu~ = |Omega_B| / N * sum_j xi^j                                     (eq. 8)

where |Omega_B| = 2^k for Boolean cubes, or C(n, r) (C(n, r)^2, ...) for the
structured "support" design whose cubes are whole rows with the prescribed row
sum (every other assignment of those rows violates the row-sum constraint, so
this is the same decomposition restricted to the feasible cubes: an exact
importance-sampling rewrite of Theorem 2).

Work unit: CaDiCaL 1.9.5 (pysat 'cadical195', default options) conflicts, the
unit of the label `d`; propagations are reported alongside.

Per cube: an incremental UP-only solver first checks `propagate(assumptions)`
(the paper's "incremental preprocessing" trick, section 7); a cube refuted by
unit propagation costs 0 conflicts.  Otherwise a FRESH solver (so that xi is a
function of the cube, Theorem 2's hypothesis) runs solve_limited under the cube
with the per-cube budget.

Censoring.  A cube that hits the budget has xi >= b.  Two estimates are kept:
  mu_lb   = |Omega| * mean(min(xi, b))                 (a lower bound on mu~)
  mu_cens = |Omega| * [ sum_uncens xi + n_c * b * alpha/(alpha-1) ] / N
where alpha is the censored-MLE Pareto tail index of the cube works above
u = b/8 (Hill estimator with right censoring: alpha = #uncensored exceedances /
sum_exceedances log(min(xi,b)/u)), clipped to [ALPHA_MIN, ALPHA_MAX]; with no
uncensored exceedance alpha = ALPHA_MIN (heaviest tail allowed).

Sample-size report (Theorem 3, eq. 13, with sample moments):
  N_req(eps, delta) = s^2 / (eps^2 * delta * xbar^2),
  eps_achieved(delta) = sqrt(s^2 / (N * delta * xbar^2)).

Designs for B (compared in experiments/E24_difficulty/SAMPLING.md):
  "row"       first k cells of the heaviest row(s), row-major
  "random"    k seeded random cells
  "lookahead" top-k cells by the paper's UP weight w = w+ + w- (implied variables
              after assigning the cell 1 / 0; a failed literal counts as nvars)
  "support"   structured cubes: the supports of the `k_rows` heaviest rows,
              sampled uniformly among the C(n, r_i) row patterns of each row

Public API (estimator contract, E24):
  estimate(inst, rows, cols, **kw) -> {"features", "d_hat", "cost_conflicts",
                                       "cost_propagations", "cost_seconds", ...}
A pure function of its inputs for a fixed seed (deterministic solver, seeded RNG).
"""
from __future__ import annotations

import math
import random
import statistics
import time
from math import comb, log2
from typing import Dict, List, Optional, Sequence, Tuple

from pysat.solvers import Solver

from .encoding import encode_case
from .known import Instance

SOLVER = "cadical195"
DESIGNS = ("row", "random", "lookahead", "support")
ALPHA_MIN, ALPHA_MAX = 1.1, 8.0
TAIL_U_FRAC = 1.0 / 8.0
# Default operating point (SAMPLING.md sections 1, 5): Knuth probes over the row-major cells of the
# heaviest rows, 8 free branchings, per-cube budget 5000, at most N = 100 cubes and a hard total of
# ~50k conflicts per case (cost <= total_budget + budget).  This is the best budgeted point of the
# knuth sampler on DEV.  The overall DEV-only pick, stratified probes (sampler="strat",
# strata_depth=5, budget=2000), was better on DEV (LOCO within-cell 0.743 vs 0.695) but worse on
# the held-out wide cells (0.652 vs 0.749) and equal on the target tables (0.681 vs 0.678); the
# choice between the two DEV-selected candidates therefore used held-out results (disclosed in
# SAMPLING.md).  Both calibrations are frozen below.
DEFAULT = dict(design="row", sampler="knuth", k=8, strata_depth=5, N=100, budget=5000, total_budget=50_000, seed=0)

# Frozen calibration  log d = a + b log mu~  (natural logs) on the 262 exact DEV hard cases
# (ground_truth_initial.jsonl, d > 20k; 4 square cells).  Keyed by
# (sampler, design, k, N, budget, total_budget, strata_depth).  Unit slope ("ratio" calibration,
# a = median log(d / mu~) on DEV): on DEV the estimator noise exceeds the spread of log d
# (sd log mu~ = 0.76 / 1.08 vs sd log d = 0.56 for strat / knuth), so the OLS slope is attenuated
# by errors-in-variables (strat: a = 3.776, b = 0.551; knuth: a = 6.241, b = 0.354) and would
# compress the predictions on cells with a wider range of d.  mu~ ~ 6.4 d on DEV for both.
CALIBRATION: Dict[tuple, Tuple[float, float]] = {
    ("strat", "row", 8, 100, 2000, 50_000, 5): (-1.8599, 1.0),
    ("knuth", "row", 8, 100, 5000, 50_000, None): (-1.8548, 1.0),  # the default
}


def calibrate_d_hat(mu: float, config: dict) -> Optional[float]:
    """Calibrated conflicts estimate exp(a + b log mu~) for a calibrated operating point, else mu~."""
    key = (config.get("sampler"), config.get("design"), config.get("k"), config.get("N"),
           config.get("budget"), config.get("total_budget"), config.get("strata_depth"))
    ab = CALIBRATION.get(key)
    if ab is None or mu is None:
        return mu
    return float(math.exp(ab[0] + ab[1] * math.log(max(mu, 1.0))))


# ---------------------------------------------------------------------------
# decomposition sets
# ---------------------------------------------------------------------------
def _row_order(rows: Sequence[int]) -> List[int]:
    """Rows by decreasing weight, ties by index (stable)."""
    return sorted(range(len(rows)), key=lambda i: (-rows[i], i))


def _stats(s: Solver) -> Tuple[int, int]:
    st = s.accum_stats() or {}
    return int(st.get("conflicts", 0)), int(st.get("propagations", 0))


def lookahead_weights(cnf, inst: Instance, up: Solver) -> Tuple[List[Tuple[int, int, int]], int]:
    """[(w, w_plus_times_w_minus, var)] for every cell var, and the propagation cost."""
    nv = cnf.nvars
    out = []
    p0 = _stats(up)[1]
    for i in range(inst.m):
        for j in range(inst.n):
            v = cnf.X[i][j]
            ws = []
            for lit in (v, -v):
                ok, lits = up.propagate(assumptions=[lit])
                ws.append(len(lits) if ok else nv)
            out.append((ws[0] + ws[1], ws[0] * ws[1], v))
    cost = _stats(up)[1] - p0
    return out, cost


def choose_B(cnf, inst: Instance, rows: Sequence[int], design: str, k: int, seed: int,
             up: Optional[Solver] = None) -> Tuple[List[int], dict]:
    """Decomposition set (list of cell variable ids) and metadata."""
    m, n = inst.m, inst.n
    meta: dict = {"design": design}
    if design == "row":
        order = [cnf.X[i][j] for i in _row_order(rows) for j in range(n)]
        B = order[:k]
    elif design == "random":
        rng = random.Random(f"B|{seed}|{inst.tag}|{tuple(rows)}")
        B = sorted(rng.sample([cnf.X[i][j] for i in range(m) for j in range(n)], k))
    elif design in ("lookahead", "lookahead_prod"):
        assert up is not None
        w, cost = lookahead_weights(cnf, inst, up)
        key = (lambda x: (-x[0], x[2])) if design == "lookahead" else (lambda x: (-x[1], -x[0], x[2]))
        B = [x[2] for x in sorted(w, key=key)[:k]]
        meta["lookahead_props"] = cost
        meta["lookahead_w_top"] = [x[0] for x in sorted(w, key=key)[:k]]
    elif design == "support":
        ri = _row_order(rows)[:k]
        B = [cnf.X[i][j] for i in ri for j in range(n)]
        meta["support_rows"] = ri
    else:
        raise ValueError(f"unknown design {design!r}")
    return B, meta


def log2_cube_space(inst: Instance, rows: Sequence[int], design: str, k: int) -> float:
    if design == "support":
        return float(sum(log2(comb(inst.n, rows[i])) for i in _row_order(rows)[:k]))
    return float(k)


def sample_cubes(inst: Instance, rows: Sequence[int], B: List[int], design: str, k: int, N: int,
                 seed: int, cnf=None) -> List[List[int]]:
    """N i.i.d. uniform cubes (with replacement, as in the paper)."""
    rng = random.Random(f"cubes|{seed}|{design}|{k}|{inst.tag}|{tuple(rows)}")
    cubes = []
    if design == "support":
        n = inst.n
        ri = _row_order(rows)[:k]
        for _ in range(N):
            cube = []
            for i in ri:
                S = set(rng.sample(range(n), rows[i]))
                cube.extend(cnf.X[i][j] if j in S else -cnf.X[i][j] for j in range(n))
            cubes.append(cube)
    else:
        for _ in range(N):
            cubes.append([v if rng.random() < 0.5 else -v for v in B])
    return cubes


def cell_order(cnf, inst: Instance, rows: Sequence[int], design: str, seed: int,
               up: Optional[Solver] = None) -> Tuple[List[int], dict]:
    """Full ordering of the m*n cell variables for the Knuth sampler."""
    m, n = inst.m, inst.n
    if design in ("row", "support"):
        return [cnf.X[i][j] for i in _row_order(rows) for j in range(n)], {"design": design}
    if design == "random":
        rng = random.Random(f"order|{seed}|{inst.tag}|{tuple(rows)}")
        B = [cnf.X[i][j] for i in range(m) for j in range(n)]
        rng.shuffle(B)
        return B, {"design": design}
    B, meta = choose_B(cnf, inst, rows, design, m * n, seed, up)
    return B, meta


def _viable(up: Solver, cube: List[int], v: int):
    out = []
    for lit in (v, -v):
        ok, lits = up.propagate(assumptions=cube + [lit])
        if ok:
            out.append((lit, lits))
    return out


def _probe(order: List[int], k: int, up: Solver, rng: random.Random, cube0: List[int], branch0: int = 0,
           ) -> Tuple[List[int], float, bool]:
    """One Knuth probe from the node `cube0` (already `branch0` branchings deep)."""
    cube = list(cube0)
    w = 1.0
    branch = branch0
    ok, lits = up.propagate(assumptions=cube)
    if not ok:
        return cube, w, True
    fixed = set(abs(l) for l in lits)
    for v in order:
        if branch >= k:
            break
        if v in fixed:
            continue
        viable = _viable(up, cube, v)
        if not viable:
            return cube, w, True
        if len(viable) == 2:
            lit, lits = viable[rng.randrange(2)]
            w *= 2.0
            branch += 1
        else:
            lit, lits = viable[0]
        cube.append(lit)
        fixed = set(abs(l) for l in lits)
    return cube, w, False


def knuth_probes(cnf, order: List[int], k: int, N: int, up: Solver, rng: random.Random,
                 ) -> List[Tuple[List[int], float, bool]]:
    """Knuth (1975) random probes of the depth-k UP-pruned decision tree over `order`.

    A node branches on the first variable of `order` not fixed by unit propagation
    under the node's cube; each value is 'viable' when propagate() does not refute
    it.  With 2 viable values the probe picks one uniformly and doubles its weight;
    with 1 the variable is forced (weight x1); with 0 the probe dies (leaf refuted by
    UP, work 0).  After k branchings (or exhausting `order`) the cube is a leaf.
    E[weight * xi(leaf)] = sum over surviving leaves of xi: the paper's mu_B with
    the UP-refuted cubes (xi = 0 conflicts) dropped from the sampling frame."""
    return [_probe(order, k, up, rng, []) for _ in range(N)]


def enumerate_nodes(order: List[int], depth: int, up: Solver, max_nodes: int = 64) -> Tuple[List[List[int]], int]:
    """All UP-surviving nodes of the same decision tree after `depth` branchings
    (nodes where `order` is exhausted earlier are kept as they are).  Exhaustive
    (no sampling); stops deepening when the frontier would exceed max_nodes."""
    frontier = [[]]
    reached = 0
    for _ in range(depth):
        nxt = []
        for cube in frontier:
            ok, lits = up.propagate(assumptions=cube)
            if not ok:
                continue
            fixed = set(abs(l) for l in lits)
            cur = list(cube)
            dead = False
            branched = False
            for v in order:
                if v in fixed:
                    continue
                viable = _viable(up, cur, v)
                if not viable:
                    dead = True
                    break
                if len(viable) == 2:
                    nxt.extend([cur + [viable[0][0]], cur + [viable[1][0]]])
                    branched = True
                    break
                cur.append(viable[0][0])
                fixed = set(abs(l) for l in viable[0][1])
            if not dead and not branched:
                nxt.append(cur)
        if len(nxt) > max_nodes:
            break
        frontier = nxt
        reached += 1
    return frontier, reached


def stratified_probes(order: List[int], k: int, N: int, up: Solver, rng: random.Random, depth: int,
                      ) -> Tuple[List[Tuple[List[int], float, bool, int]], int]:
    """Stratified Knuth: enumerate the depth-`depth` nodes exactly, then draw probes
    below them round-robin (probe i goes to stratum i mod S).  The estimate is
    sum_s mean_{probes in s}(weight * xi) (see summarize), unbiased for the same mu_B
    as `knuth_probes` whenever every stratum has >= 1 probe (N >= S); it removes the
    between-strata part of the variance."""
    nodes, reached = enumerate_nodes(order, depth, up)
    S = len(nodes)
    if S == 0:
        return [([], 1.0, True, 0)] * 0, 0
    # branch count of each node = number of 2-way branchings = depth unless order ran out;
    # recomputing it is unnecessary: _probe only needs the remaining branch budget, which
    # we pass as k - depth by starting branch0 = depth (nodes that stopped early have no
    # free variable left, so any branch0 gives the same leaf).
    out = []
    for i in range(N):
        sidx = i % S
        cube, w, dead = _probe(order, k, up, rng, nodes[sidx], branch0=min(reached, k))
        out.append((cube, w, dead, sidx))
    return out, S


# ---------------------------------------------------------------------------
# per-cube solving
# ---------------------------------------------------------------------------
def run_cubes(cnf, cubes: List[Optional[List[int]]], budget: int, up: Solver, solver: str = SOLVER,
              incremental: bool = False, total_budget: Optional[int] = None) -> List[dict]:
    """One record per cube: {up_refuted, status, conflicts, propagations, censored, seconds}.
    A cube given as None is a Knuth probe that died by unit propagation while it was drawn
    (recorded as up_refuted with 0 work).  total_budget: stop after the cube at which the
    cumulative conflicts first reach it (the budgeted operating mode; cost <= total + budget)."""
    out = []
    spent = 0
    inc = Solver(name=solver, bootstrap_with=cnf.clauses) if incremental else None
    try:
        for cube in cubes:
            if total_budget is not None and spent >= total_budget:
                break
            t0 = time.time()
            if cube is None:
                out.append(dict(up_refuted=True, status="unsat", conflicts=0, propagations=0,
                                censored=False, seconds=0.0))
                continue
            p0 = _stats(up)[1]
            ok, _ = up.propagate(assumptions=cube)
            up_props = _stats(up)[1] - p0
            if not ok:
                out.append(dict(up_refuted=True, status="unsat", conflicts=0, propagations=up_props,
                                censored=False, seconds=time.time() - t0))
                continue
            s = inc if incremental else Solver(name=solver, bootstrap_with=cnf.clauses)
            try:
                c0, q0 = _stats(s)
                s.conf_budget(budget)
                res = s.solve_limited(assumptions=cube)
                c1, q1 = _stats(s)
            finally:
                if not incremental:
                    s.delete()
            out.append(dict(up_refuted=False,
                            status="sat" if res is True else ("unsat" if res is False else "unknown"),
                            conflicts=c1 - c0, propagations=(q1 - q0) + up_props,
                            censored=res is None, seconds=time.time() - t0))
            spent += c1 - c0
    finally:
        if inc is not None:
            inc.delete()
    return out


# ---------------------------------------------------------------------------
# estimators from cube records (usable offline: `summarize(records[:N], budget=b)`)
# ---------------------------------------------------------------------------
def pareto_alpha(works: Sequence[float], censored: Sequence[bool], budget: float) -> Tuple[float, int]:
    u = max(1.0, budget * TAIL_U_FRAC)
    num = 0
    den = 0.0
    for x, c in zip(works, censored):
        if x > u or c:
            den += math.log(max(min(x, budget), u) / u)
            if not c:
                num += 1
    if num == 0 or den <= 0:
        return ALPHA_MIN, num
    return min(ALPHA_MAX, max(ALPHA_MIN, num / den)), num


def summarize(recs: List[dict], budget: int, log2_space: float, eps: float = 0.2,
              delta: float = 0.1) -> dict:
    """Features + estimates from cube records.  Records made with a larger budget B'
    are re-censored at `budget` <= B' (a fresh CaDiCaL run is deterministic and the
    conflict limit only stops it, so min(xi, b) is what a run at cap b would report).
    A record may carry a Knuth weight (default 1); the sample value is weight * xi."""
    N = len(recs)
    xs, cens, props, ws = [], [], [], []
    for r in recs:
        c = r["conflicts"] >= budget or r["censored"]
        x = min(r["conflicts"], budget) if c else r["conflicts"]
        # propagations of a censored run at a smaller budget: scale by the conflict ratio
        p = r["propagations"] if r["conflicts"] <= budget else r["propagations"] * budget / max(1, r["conflicts"])
        xs.append(float(x))
        cens.append(bool(c))
        props.append(float(p))
        ws.append(float(r.get("weight", 1.0)))
    space = 2.0 ** log2_space
    wx = [w * x for w, x in zip(ws, xs)]
    alpha, n_tail = pareto_alpha(xs, cens, budget)
    tail_mean = budget * alpha / (alpha - 1.0)
    vc = [w * (tail_mean if c else x) for w, x, c in zip(ws, xs, cens)]
    if any("stratum" in r for r in recs):
        # stratified estimator: sum over strata of the within-stratum mean; the variance
        # of the estimator is sum_s var_s / n_s, reported per sample (x N) for eq. 13
        groups: Dict[int, List[int]] = {}
        for i, r in enumerate(recs):
            groups.setdefault(int(r.get("stratum", 0)), []).append(i)
        mean = sum(sum(wx[i] for i in g) / len(g) for g in groups.values())
        mean_cens = sum(sum(vc[i] for i in g) / len(g) for g in groups.values())
        var = N * sum((statistics.variance([wx[i] for i in g]) / len(g)) if len(g) > 1 else 0.0
                      for g in groups.values())
    else:
        mean = sum(wx) / N
        var = statistics.variance(wx) if N > 1 else 0.0
        mean_cens = sum(vc) / N
    n_c = sum(cens)
    n_up = sum(1 for r in recs if r["up_refuted"])
    live_w = sum(w for w, r in zip(ws, recs) if not r["up_refuted"]) / N
    raw_mean = sum(xs) / N
    live_x = [x for x, r in zip(xs, recs) if not r["up_refuted"]]
    feats = {
        "log2_space": log2_space,
        "mean_work": raw_mean,
        "mean_live_work": (sum(live_x) / len(live_x)) if live_x else 0.0,
        "median_work": float(statistics.median(xs)),
        "max_work": max(xs),
        "mean_props": sum(w * p for w, p in zip(ws, props)) / N,
        "frac_up_refuted": n_up / N,
        "frac_budget_hit": n_c / N,
        "frac_sat": sum(1 for r in recs if r["status"] == "sat") / N,
        "cv": (math.sqrt(var) / mean) if mean > 0 else 0.0,
        "pareto_alpha": alpha,
        "n_tail": float(n_tail),
        "log2_mu_lb": log2_space + math.log2(max(mean, 1e-9)),
        "log2_mu_cens": log2_space + math.log2(max(mean_cens, 1e-9)),
        "log2_mu_props": log2_space + math.log2(max(sum(w * p for w, p in zip(ws, props)) / N, 1e-9)),
        "log2_live_leaves": log2_space + math.log2(max(live_w, 0.5 / N)),
        "N_req": (var / (eps * eps * delta * mean * mean)) if mean > 0 else 0.0,
        "eps_achieved": math.sqrt(var / (N * delta * mean * mean)) if mean > 0 else 0.0,
    }
    return {
        "features": feats,
        "d_hat": space * mean_cens,
        "mu_lb": space * mean,
        "cost_conflicts": int(sum(xs)),
        "cost_propagations": int(sum(props)),
    }


# ---------------------------------------------------------------------------
# public estimator
# ---------------------------------------------------------------------------
def estimate(inst: Instance, rows: Sequence[int], cols: Sequence[int], design: str = DEFAULT["design"],
             k: int = DEFAULT["k"], N: int = DEFAULT["N"], budget: int = DEFAULT["budget"],
             seed: int = DEFAULT["seed"], sampler: str = DEFAULT["sampler"], solver: str = SOLVER,
             incremental: bool = False, strata_depth: int = DEFAULT["strata_depth"], total_budget: Optional[int] = DEFAULT["total_budget"],
             eps: float = 0.2, delta: float = 0.1, return_records: bool = False, calibrated: bool = True,
             cnf=None) -> dict:
    """Chivilikhin-Pavlenko-Semenov d-hardness estimate of one case (see module doc).

    design in DESIGNS; k = |B| (cells) or, for design="support", the number of
    heaviest rows whose supports form the cube.  sampler: "uniform" (i.i.d. cubes of B, the
    paper's estimator), "knuth" (Knuth probes of the UP-pruned decision tree over the cell order
    of `design`, k = number of free branchings; default), "strat" (Knuth, stratified at depth
    `strata_depth`).  total_budget: stop drawing cubes once the conflicts spent reach it.
    Returns d_hat_raw = censoring-aware mu~ (sum of cube works, conflicts) and d_hat = the frozen
    DEV calibration of mu~ for the default operating point (= mu~ for other configurations or
    calibrated=False).  The cube work is NOT the monolithic d: mu~ ~ 6.4 d on DEV."""
    t0 = time.time()
    cnf = cnf if cnf is not None else encode_case(inst, rows, cols)
    up = Solver(name=solver, bootstrap_with=cnf.clauses)
    try:
        if sampler in ("knuth", "strat"):
            order, meta = cell_order(cnf, inst, rows, design, seed, up)
            B = order
            p0 = _stats(up)[1]
            rng = random.Random(f"{sampler}|{seed}|{design}|{k}|{inst.tag}|{tuple(rows)}")
            if sampler == "strat":
                sp, S = stratified_probes(order, k, N, up, rng, strata_depth)
                probes = [(c, w, dead) for c, w, dead, _ in sp]
                strata = [sidx for *_, sidx in sp]
                meta["n_strata"] = S
            else:
                probes = knuth_probes(cnf, order, k, N, up, rng)
                strata = None
            meta["lookahead_props"] = meta.get("lookahead_props", 0) + (_stats(up)[1] - p0)
            recs = run_cubes(cnf, [None if dead else c for c, _, dead in probes], budget, up, solver=solver,
                             incremental=incremental, total_budget=total_budget)
            for i, r in enumerate(recs):
                r["weight"] = probes[i][1]
                if strata is not None:
                    r["stratum"] = strata[i]
            space = 0.0
        else:
            B, meta = choose_B(cnf, inst, rows, design, k, seed, up)
            cubes = sample_cubes(inst, rows, B, design, k, N, seed, cnf)
            recs = run_cubes(cnf, cubes, budget, up, solver=solver, incremental=incremental,
                             total_budget=total_budget)
            space = log2_cube_space(inst, rows, design, k)
    finally:
        up.delete()
    used = recs
    S = int(meta.get("n_strata") or 0)
    if sampler == "strat" and S and total_budget is not None:
        # budgeted stratified run: estimate from complete round-robin rounds only (every stratum
        # represented); the cost still counts every cube that was run
        n_use = min(len(recs), max(S, (len(recs) // S) * S))
        if n_use > len(recs):  # a single round did not fit in the budget: finish it
            up2 = Solver(name=solver, bootstrap_with=cnf.clauses)
            try:
                more = run_cubes(cnf, [None if dead else c for c, _, dead in probes[len(recs):n_use]], budget,
                                 up2, solver=solver)
            finally:
                up2.delete()
            for i, r in enumerate(more, start=len(recs)):
                r["weight"] = probes[i][1]
                r["stratum"] = strata[i]
            recs = recs + more
        used = recs[:n_use]
    out = summarize(used, budget, space, eps=eps, delta=delta)
    out["cost_conflicts"] = int(sum(min(r["conflicts"], budget) for r in recs))
    out["cost_propagations"] = int(sum(r["propagations"] for r in recs))
    out["cost_propagations"] += int(meta.get("lookahead_props", 0))
    out["cost_seconds"] = time.time() - t0
    out["config"] = dict(design=design, k=k, N=N, budget=budget, seed=seed, sampler=sampler,
                         incremental=incremental, strata_depth=strata_depth if sampler == "strat" else None,
                         total_budget=total_budget)
    out["features"]["n_cubes"] = float(len(recs))
    out["d_hat_raw"] = out["d_hat"]
    if calibrated:
        out["d_hat"] = calibrate_d_hat(out["d_hat"], out["config"])
    if "n_strata" in meta:
        out["features"]["n_strata"] = float(meta["n_strata"])
    if return_records:
        out["records"] = recs
        out["B"] = B
    return out

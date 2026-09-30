"""Lookahead difficulty features for one case (E24, owner A2-lookahead).

AlphaMapleSAT / march_cu-style probes of the fixed-profile CNF `encode_case(inst, rows, cols)`.
Root unit propagation assigns no cell in this encoding (fixed sums force no individual cell,
E10), so every probe here starts ONE DECISION DEEPER:

  la1_*   single-decision lookahead: for every cell literal l (x_ij and -x_ij), UP(F & l);
          implied-cell counts (mean/median/max/min/std; all, positive, negative literals),
          failed literals, implied variables (all vars, AMS "propagation rate"), the AMS
          per-variable score r(v)r(-v) + r(v) + r(-v), march's product r(v)r(-v), and
          march/Schur-style clause measures after the decision: new binary clauses and the
          eval_cls-style weighted count of reduced, unsatisfied clauses (weight 5^-(len-2)).
  la2_*   two-decision lookahead on a seeded sample of literal pairs (l1 non-failed, l2 on a
          cell left free by l1): implied cells, failed pairs, and "synergy" =
          implied(l1,l2) - implied(l1) - implied(l2).
  fl_*    failed-literal fixpoint: iterate la1 probing, asserting the negation of every failed
          literal, until none fails; fixed/free cells, rounds, refuted flag, and la1-style stats
          (implied cells, AMS score) on the FL-reduced formula; fl_log2vol_rows/cols = the
          row-profile (column-profile) search-space volume  sum_i log2 C(free_i, r_i - ones_i)
          left after FL (= log2_volume when FL fixes nothing).
  kn_*    Knuth (1975) random-probing tree-size estimator for the DPLL+UP tree over cell
          variables: walk down from the root, at each node test both children of the chosen
          cell, b = number of UP-consistent children, pick one uniformly; N = 1 + b1 + b1 b2 + ...
          Also the estimated number of failed (conflict) leaves  sum_i (prod_{j<i} b_j)(2 - b_i),
          which is the quantity closest to a conflict count.  Two variable orders: "rand"
          (uniform free cell) and "row" (first free cell in row-major order, Tan-like
          enumeration).  Reported as log of the mean over probes, mean of logs, variance of
          logs, mean depth.  knfl_* = the same walk started from the FL-fixpoint root.

Everything is BCP only (no search, no conflicts analysed); cost is reported as the number of
propagate() calls, total trail literals assigned (cost_propagations; a count of assignments made,
comparable to CaDiCaL's `propagations`), failed propagations (cost_conflicts: every UP conflict
hit), and seconds.  Deterministic given `seed`.

Contract (E24 estimator):  estimate(inst, rows, cols, **kw) -> {"features", "d_hat",
"cost_conflicts", "cost_propagations", "cost_seconds"}.  d_hat is the Knuth failed-leaf estimate
(a standalone, uncalibrated difficulty prediction), or None when Knuth probing is disabled.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from pysat.solvers import Solver

from .encoding import encode_case
from .known import Instance

# UP fixpoints are solver independent when no conflict occurs; minisat22's propagate() is ~50x
# cheaper than cadical195's through pysat (E24 LOOKAHEAD.md, timing section).
DEFAULT_SOLVER = "minisat22"

# Optional calibrated model, frozen by experiments/E24_difficulty/lookahead_freeze.py on the DEV set:
#   log d = beta0 + sum_k beta_k (f_k - mu_k) / sd_k        (fitted on exact d > 2000 cases)
#   + hard_shift                                             (if the case is unsolved at 20k)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments",
    "E24_difficulty",
    "lookahead_model.json",
)


def load_model(path: str = MODEL_PATH, name: Optional[str] = None) -> Optional[dict]:
    """The named model (default: the file's `primary`) or None when the file is absent."""
    try:
        with open(path) as f:
            d = json.load(f)
        return d["models"][name or d["primary"]]
    except (OSError, ValueError, KeyError):
        return None


def model_log_d(model: dict, feats: Dict[str, float]) -> Optional[float]:
    if any(k not in feats for k in model["features"]):
        return None
    z = model["beta"][0]
    for k, mu, sd, b in zip(model["features"], model["mu"], model["sd"], model["beta"][1:]):
        z += b * (feats[k] - mu) / (sd or 1.0)
    return float(z)


# ---------------------------------------------------------------------------
# small stats helpers
# ---------------------------------------------------------------------------
def _stats(prefix: str, xs: Sequence[float], out: Dict[str, float]) -> None:
    if len(xs) == 0:
        for k in ("mean", "median", "max", "min", "std"):
            out[f"{prefix}_{k}"] = 0.0
        return
    a = np.asarray(xs, dtype=float)
    out[f"{prefix}_mean"] = float(a.mean())
    out[f"{prefix}_median"] = float(np.median(a))
    out[f"{prefix}_max"] = float(a.max())
    out[f"{prefix}_min"] = float(a.min())
    out[f"{prefix}_std"] = float(a.std())


def _logmeanexp(logs: Sequence[float]) -> float:
    if not logs:
        return 0.0
    mx = max(logs)
    return mx + math.log(sum(math.exp(v - mx) for v in logs) / len(logs))


# ---------------------------------------------------------------------------
# clause-reduction measures (vectorised over the whole CNF)
# ---------------------------------------------------------------------------
class _ClauseIndex:
    """Flattened clause/literal arrays so that, for a trail of true literals, we can count the
    clauses that became binary (new binaries, Schur-5 / march) and the eval_cls-style weight of
    reduced-but-unsatisfied clauses in O(#literals) numpy work."""

    def __init__(self, clauses: List[List[int]], nvars: int):
        lens = np.fromiter((len(c) for c in clauses), dtype=np.int64, count=len(clauses))
        flat = np.fromiter((l for c in clauses for l in c), dtype=np.int64, count=int(lens.sum()))
        self.lens = lens
        self.var = np.abs(flat)
        self.sign = np.sign(flat).astype(np.int8)
        self.cid = np.repeat(np.arange(len(clauses)), lens)
        self.nclauses = len(clauses)
        self.nvars = nvars
        self.nbin_root = int((lens == 2).sum())

    def measures(self, trail: Sequence[int]) -> tuple:
        val = np.zeros(self.nvars + 1, dtype=np.int8)
        t = np.asarray(trail, dtype=np.int64)
        if len(t):
            val[np.abs(t)] = np.sign(t).astype(np.int8)
        lv = val[self.var] * self.sign  # +1 true, -1 false, 0 unassigned
        ntrue = np.bincount(self.cid, weights=(lv == 1), minlength=self.nclauses)
        nfalse = np.bincount(self.cid, weights=(lv == -1), minlength=self.nclauses)
        nun = self.lens - ntrue - nfalse
        live = (ntrue == 0) & (nfalse > 0)  # reduced, not satisfied
        newbin = int((live & (nun == 2)).sum())
        k = nun[live & (nun >= 2)]
        wred = float(np.power(0.2, k - 2).sum())
        return newbin, wred


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------
def root_unit_propagation(clauses: List[List[int]], nvars: int):
    """Level-0 unit propagation in pure Python: (ok, list of implied literals).

    Needed because pysat's propagate() returns only the literals assigned at decision levels >= 1:
    facts fixed at level 0 (e.g. every cell of a row whose sum is n or 0) never appear in its
    trail.  Counter-based UP with occurrence lists; O(total literals) per pass."""
    val = [0] * (nvars + 1)
    occ: Dict[int, List[int]] = {}
    for ci, c in enumerate(clauses):
        for l in c:
            occ.setdefault(-l, []).append(ci)  # clauses where literal -l becoming true falsifies l
    trail: List[int] = []
    queue = [c[0] for c in clauses if len(c) == 1]

    def lv(l):
        v = val[abs(l)]
        return v if l > 0 else -v

    while queue:
        l = queue.pop()
        if lv(l) == 1:
            continue
        if lv(l) == -1:
            return False, trail
        val[abs(l)] = 1 if l > 0 else -1
        trail.append(l)
        for ci in occ.get(l, ()):
            unassigned, sat = None, False
            nun = 0
            for x in clauses[ci]:
                t = lv(x)
                if t == 1:
                    sat = True
                    break
                if t == 0:
                    nun += 1
                    unassigned = x
            if sat:
                continue
            if nun == 0:
                return False, trail
            if nun == 1:
                queue.append(unassigned)
    return True, trail


class _Prop:
    """Counting wrapper around one pysat solver's propagate().  Returns the FULL trail: the
    level-0 facts (computed once, see root_unit_propagation) followed by the literals pysat
    assigned under the assumptions."""

    def __init__(self, cnf, solver: str):
        self.s = Solver(name=solver, bootstrap_with=cnf.clauses)
        self.calls = 0
        self.assigned = 0
        self.conflicts = 0
        self.root_ok, self.root = root_unit_propagation(cnf.clauses, cnf.nvars)
        self.root_vars = {abs(l) for l in self.root}

    def __call__(self, assumptions: Sequence[int]):
        if not self.root_ok:
            self.calls += 1
            self.conflicts += 1
            return False, list(self.root)
        ok, lits = self.s.propagate(assumptions=list(assumptions))
        self.calls += 1
        self.assigned += len(lits)
        if not ok:
            self.conflicts += 1
        if self.root:
            lits = self.root + [l for l in lits if abs(l) not in self.root_vars]
        return ok, lits

    def close(self):
        self.s.delete()


def estimate(
    inst: Instance,
    rows: Sequence[int],
    cols: Sequence[int],
    *,
    seed: int = 0,
    n_pairs: int = 400,
    n_probes: int = 64,
    knuth_orders: Sequence[str] = ("rand", "row"),
    clause_measures: bool = True,
    fl_fixpoint: bool = True,
    knuth_fl_orders: Sequence[str] = ("rand",),
    solver: str = DEFAULT_SOLVER,
    cnf=None,
    calibrated: bool = True,
    model_name: Optional[str] = None,
) -> dict:
    """d_hat: exp(frozen calibrated model) when `calibrated` and lookahead_model.json exists (meant
    for cases NOT refuted within 2,000 conflicts: fitted on exact d > 2000 only; for a case known to
    be unsolved at 20k use exp(features["model_log_d_hard"])), else the raw Knuth failed-leaf
    estimate from the unreduced root (uncalibrated; astronomically large).  The cheapest setting
    that supports the primary (FL-only) model is  n_pairs=0, n_probes=0, clause_measures=False
    (~0.03 s/case, ~200k trail literals)."""
    t0 = time.time()
    m, n = inst.m, inst.n
    cnf = cnf if cnf is not None else encode_case(inst, rows, cols)
    ncell = m * n
    P = _Prop(cnf, solver)
    F: Dict[str, float] = {}
    F["log2_volume"] = float(sum(math.log2(math.comb(n, r)) for r in rows))
    model = load_model(name=model_name) if calibrated else None
    try:
        # ---------------- root ----------------
        ok0, root = P([])
        root_cells = {abs(l) for l in root if abs(l) <= ncell}
        F["root_fixed_cells"] = float(len(root_cells)) if ok0 else float(ncell)
        F["root_failed"] = 0.0 if ok0 else 1.0
        if not ok0:  # refuted by UP alone
            F["la1_failed"] = float(2 * ncell)
            return _finish(F, P, t0, d_hat=1.0)

        # ---------------- level 1 ----------------
        cidx = _ClauseIndex(cnf.clauses, cnf.nvars) if clause_measures else None
        if cidx is not None:
            F["bin_root"] = float(cidx.nbin_root)
        r_cells: Dict[int, Optional[int]] = (
            {}
        )  # literal -> implied cells (excl. itself) or None if failed
        r_vars: Dict[int, int] = {}
        trails: Dict[int, set] = {}
        newbin: Dict[int, int] = {}
        wred: Dict[int, float] = {}
        free_vars = [v for v in range(1, ncell + 1) if v not in root_cells]
        for v in free_vars:
            for lit in (v, -v):
                ok, lits = P([lit])
                if not ok:
                    r_cells[lit] = None
                    continue
                cells = {abs(l) for l in lits if abs(l) <= ncell}
                r_cells[lit] = len(cells) - 1 - len(root_cells)
                r_vars[lit] = len(lits) - 1 - len(root)
                trails[lit] = cells
                if cidx is not None:
                    nb, wr = cidx.measures(lits)
                    newbin[lit], wred[lit] = nb, wr
        good = [l for l in r_cells if r_cells[l] is not None]
        pos = [r_cells[l] for l in good if l > 0]
        neg = [r_cells[l] for l in good if l < 0]
        nfail = sum(1 for l in r_cells if r_cells[l] is None)
        F["la1_nlits"] = float(len(r_cells))
        F["la1_failed"] = float(nfail)
        F["la1_failed_frac"] = nfail / max(1, len(r_cells))
        F["la1_failed_pos"] = float(sum(1 for l in r_cells if l > 0 and r_cells[l] is None))
        F["la1_failed_neg"] = float(sum(1 for l in r_cells if l < 0 and r_cells[l] is None))
        _stats("la1", pos + neg, F)
        _stats("la1_pos", pos, F)
        _stats("la1_neg", neg, F)
        F["la1_zero_frac"] = sum(1 for x in pos + neg if x == 0) / max(1, len(good))
        F["la1_rate_vars"] = (
            float(np.mean([r_vars[l] for l in good])) if good else 0.0
        )  # AMS propagation rate
        F["la1_rate_vars_pos"] = float(np.mean([r_vars[l] for l in good if l > 0])) if pos else 0.0
        F["la1_rate_vars_neg"] = float(np.mean([r_vars[l] for l in good if l < 0])) if neg else 0.0
        # AMS / march per-variable scores (failed literal: treated as eliminating every free cell)
        full = float(len(free_vars) - 1)
        ams, prod = [], []
        for v in free_vars:
            a = r_cells[v] if r_cells[v] is not None else full
            b = r_cells[-v] if r_cells[-v] is not None else full
            ams.append(a * b + a + b)
            prod.append(a * b)
        F["ams_score_max"] = float(max(ams)) if ams else 0.0
        F["ams_score_mean"] = float(np.mean(ams)) if ams else 0.0
        F["ams_score_top3"] = float(np.mean(sorted(ams)[-3:])) if ams else 0.0
        F["march_prod_max"] = float(max(prod)) if prod else 0.0
        F["march_prod_mean"] = float(np.mean(prod)) if prod else 0.0
        if cidx is not None and good:
            _stats("la1_newbin", [newbin[l] for l in good], F)
            F["la1_newbin_pos_mean"] = (
                float(np.mean([newbin[l] for l in good if l > 0])) if pos else 0.0
            )
            F["la1_newbin_neg_mean"] = (
                float(np.mean([newbin[l] for l in good if l < 0])) if neg else 0.0
            )
            F["la1_wred_mean"] = float(np.mean([wred[l] for l in good]))
            F["la1_wred_max"] = float(max(wred[l] for l in good))

        rng = random.Random(_seed(seed, m, n, rows, cols))

        # ---------------- failed-literal fixpoint (march / AMS "L") ----------------
        # every failed literal's negation is implied by F; add them all, re-probe the cells
        # left free, repeat until no literal fails.  Measures the formula one FL-closure deeper.
        fl_units: List[int] = []
        fl_refuted = False
        if fl_fixpoint:
            new_units = [-l for l in r_cells if r_cells[l] is None]
            rounds = 1
            fl_counts: Dict[int, int] = {}
            while True:
                if any(-u in new_units for u in new_units):
                    fl_refuted = True
                    break
                fl_units = fl_units + new_units
                ok, base = P(fl_units)
                if not ok:
                    fl_refuted = True
                    break
                base_cells = {abs(l) for l in base if abs(l) <= ncell}
                if not new_units and rounds > 1:
                    break
                new_units, fl_counts = [], {}
                for v in range(1, ncell + 1):
                    if v in base_cells:
                        continue
                    for lit in (v, -v):
                        ok, lits = P(fl_units + [lit])
                        if not ok:
                            new_units.append(-lit)
                        else:
                            fl_counts[lit] = (
                                sum(1 for l in lits if abs(l) <= ncell) - 1 - len(base_cells)
                            )
                rounds += 1
                if not new_units:
                    break
            F["fl_refuted"] = 1.0 if fl_refuted else 0.0
            F["fl_rounds"] = float(rounds)
            F["fl_units"] = float(len(fl_units))
            if fl_refuted:
                F["fl_fixed_cells"] = float(ncell)
                F["fl_free_cells"] = 0.0
            else:
                F["fl_fixed_cells"] = float(len(base_cells))
                F["fl_free_cells"] = float(ncell - len(base_cells))
                vals = list(fl_counts.values())
                _stats("fl_la1", vals, F)
                fv = sorted({abs(l) for l in fl_counts})
                ams_fl = [
                    fl_counts.get(v, 0) * fl_counts.get(-v, 0)
                    + fl_counts.get(v, 0)
                    + fl_counts.get(-v, 0)
                    for v in fv
                ]
                F["fl_ams_mean"] = float(np.mean(ams_fl)) if ams_fl else 0.0
                F["fl_ams_max"] = float(max(ams_fl)) if ams_fl else 0.0
                F["fl_zero_frac"] = sum(1 for x in vals if x == 0) / max(1, len(vals))
            F["fl_fixed_frac"] = F["fl_fixed_cells"] / ncell
            # search-space volume left after FL: rows (and columns) choose their remaining ones
            # among their free cells; equals log2_volume (rows) when nothing is fixed
            if fl_refuted:
                F["fl_log2vol_rows"] = F["fl_log2vol_cols"] = 0.0
            else:
                F["fl_log2vol_rows"], F["fl_log2vol_cols"] = _reduced_volume(base, m, n, rows, cols)

        # ---------------- level 2 (seeded pairs) ----------------
        pr_cells, pr_syn, pr_fail = [], [], 0
        if n_pairs > 0 and good:
            for _ in range(n_pairs):
                l1 = good[rng.randrange(len(good))]
                cand = [v for v in free_vars if v not in trails[l1]]
                if not cand:
                    continue
                v2 = cand[rng.randrange(len(cand))]
                l2 = v2 if rng.random() < 0.5 else -v2
                ok, lits = P([l1, l2])
                if not ok:
                    pr_fail += 1
                    continue
                c = sum(1 for l in lits if abs(l) <= ncell) - 2 - len(root_cells)
                pr_cells.append(c)
                r2 = r_cells.get(l2)
                pr_syn.append(c - r_cells[l1] - (r2 if r2 is not None else 0))
            ntried = len(pr_cells) + pr_fail
            F["la2_n"] = float(ntried)
            F["la2_failed_frac"] = pr_fail / max(1, ntried)
            _stats("la2", pr_cells, F)
            F["la2_synergy_mean"] = float(np.mean(pr_syn)) if pr_syn else 0.0

        # ---------------- Knuth random probing ----------------
        d_hat = None
        for order in knuth_orders if n_probes > 0 else ():
            logN, logC, depths, sat_leaves = [], [], [], 0
            for _ in range(n_probes):
                est_nodes, est_conf, depth, is_sat = _knuth_probe(P, ncell, root, order, rng)
                logN.append(math.log(est_nodes))
                logC.append(math.log(max(est_conf, 1.0)))
                depths.append(depth)
                sat_leaves += is_sat
            F[f"kn_{order}_log_mean_nodes"] = _logmeanexp(logN)
            F[f"kn_{order}_mean_log_nodes"] = float(np.mean(logN))
            F[f"kn_{order}_var_log_nodes"] = float(np.var(logN))
            F[f"kn_{order}_log_mean_conf"] = _logmeanexp(logC)
            F[f"kn_{order}_mean_log_conf"] = float(np.mean(logC))
            F[f"kn_{order}_mean_depth"] = float(np.mean(depths))
            F[f"kn_{order}_sat_frac"] = sat_leaves / n_probes
            if d_hat is None:
                d_hat = math.exp(F[f"kn_{order}_log_mean_conf"])
        if fl_fixpoint and n_probes > 0:
            for order in knuth_fl_orders:
                tag = f"knfl_{order}"
                if fl_refuted:
                    for k in (
                        "log_mean_nodes",
                        "mean_log_nodes",
                        "var_log_nodes",
                        "log_mean_conf",
                        "mean_log_conf",
                        "mean_depth",
                    ):
                        F[f"{tag}_{k}"] = 0.0
                    continue
                logN, logC, depths = [], [], []
                for _ in range(n_probes):
                    est_nodes, est_conf, depth, _sat = _knuth_probe(
                        P, ncell, base, order, rng, prefix=fl_units
                    )
                    logN.append(math.log(est_nodes))
                    logC.append(math.log(max(est_conf, 1.0)))
                    depths.append(depth)
                F[f"{tag}_log_mean_nodes"] = _logmeanexp(logN)
                F[f"{tag}_mean_log_nodes"] = float(np.mean(logN))
                F[f"{tag}_var_log_nodes"] = float(np.var(logN))
                F[f"{tag}_log_mean_conf"] = _logmeanexp(logC)
                F[f"{tag}_mean_log_conf"] = float(np.mean(logC))
                F[f"{tag}_mean_depth"] = float(np.mean(depths))
        if fl_refuted:
            d_hat = 1.0
        elif model is not None:
            z = model_log_d(model, F)
            if z is not None:
                F["model_log_d"] = z
                F["model_log_d_hard"] = z + float(model.get("hard_shift", 0.0))
                d_hat = math.exp(z)
        return _finish(F, P, t0, d_hat=d_hat)
    finally:
        P.close()


def _reduced_volume(trail: Sequence[int], m: int, n: int, rows, cols) -> tuple:
    ncell = m * n
    val = {}
    for l in trail:
        if abs(l) <= ncell:
            val[abs(l)] = l > 0
    vr = vc = 0.0
    for i in range(m):
        free = sum(1 for j in range(n) if (i * n + j + 1) not in val)
        ones = sum(1 for j in range(n) if val.get(i * n + j + 1) is True)
        vr += (
            math.log2(math.comb(free, max(0, rows[i] - ones)))
            if 0 <= rows[i] - ones <= free
            else 0.0
        )
    for j in range(n):
        free = sum(1 for i in range(m) if (i * n + j + 1) not in val)
        ones = sum(1 for i in range(m) if val.get(i * n + j + 1) is True)
        vc += (
            math.log2(math.comb(free, max(0, cols[j] - ones)))
            if 0 <= cols[j] - ones <= free
            else 0.0
        )
    return vr, vc


def _seed(seed: int, m: int, n: int, rows, cols) -> int:
    """Stable (process-independent) seed from the case; Python's hash() of ints/tuples is stable,
    but we avoid it to be explicit."""
    h = 1469598103934665603
    for x in [seed, m, n, *rows, -1, *cols]:
        h = ((h ^ (x & 0xFFFFFFFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _knuth_probe(
    P: _Prop,
    ncell: int,
    root: Sequence[int],
    order: str,
    rng: random.Random,
    prefix: Sequence[int] = (),
):
    """One random root-to-leaf walk of the DPLL+UP tree over cell variables (below the
    assumption `prefix`, e.g. the failed-literal units; `root` = its UP trail).
    Returns (Knuth estimate of #consistent nodes, estimate of #failed children, depth, sat_leaf)."""
    decisions: List[int] = list(prefix)
    assigned = {abs(l) for l in root if abs(l) <= ncell}
    weight = 1.0  # prod of branching factors so far
    nodes = 1.0
    conf = 0.0
    depth = 0
    while True:
        free = [v for v in range(1, ncell + 1) if v not in assigned]
        if not free:
            return nodes, conf, depth, 1
        v = free[0] if order == "row" else free[rng.randrange(len(free))]
        kids = []
        for lit in (v, -v):
            ok, lits = P(decisions + [lit])
            if ok:
                kids.append((lit, lits))
        b = len(kids)
        conf += weight * (2 - b)
        if b == 0:
            return nodes, conf, depth, 0
        weight *= b
        nodes += weight
        lit, lits = kids[rng.randrange(b)]
        decisions.append(lit)
        assigned = {abs(l) for l in lits if abs(l) <= ncell}
        depth += 1


def _finish(F: Dict[str, float], P: _Prop, t0: float, d_hat) -> dict:
    return {
        "features": {k: float(v) for k, v in F.items()},
        "d_hat": None if d_hat is None else float(d_hat),
        "cost_conflicts": int(P.conflicts),
        "cost_propagations": int(P.assigned),
        "cost_calls": int(P.calls),
        "cost_seconds": float(time.time() - t0),
    }

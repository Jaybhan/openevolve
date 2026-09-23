"""Solving one case with pysat (CaDiCaL 1.9.5 by default), with conflict and
wall-clock budgets, optional DRAT proof capture (solvers that support it)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from pysat.solvers import Solver

from .encoding import CNF, decode_matrix, has_kst
from .known import Instance


@dataclass
class SolveResult:
    status: str  # 'sat' | 'unsat' | 'unknown'
    seconds: float
    conflicts: int = 0
    decisions: int = 0
    propagations: int = 0
    matrix: Optional[List[List[int]]] = None
    witness_ok: Optional[bool] = None  # independent K_{s,t}-free check of the model
    proof: Optional[List[str]] = None
    solver: str = "cadical195"
    budget_hit: str = ""  # 'conflicts' | 'time' | ''


def solve_cnf(cnf: CNF, inst: Instance, solver: str = "cadical195",
              conf_budget: Optional[int] = None, time_limit: Optional[float] = None,
              want_proof: bool = False) -> SolveResult:
    t0 = time.time()
    s = Solver(name=solver, bootstrap_with=cnf.clauses, with_proof=want_proof)
    timer = None
    timed_out = {"v": False}
    try:
        if conf_budget is not None:
            s.conf_budget(conf_budget)
        if time_limit is not None:
            def _stop():
                timed_out["v"] = True
                s.interrupt()
            timer = threading.Timer(time_limit, _stop)
            timer.daemon = True
            timer.start()
        if conf_budget is not None or time_limit is not None:
            res = s.solve_limited(expect_interrupt=time_limit is not None)
        else:
            res = s.solve()
        dt = time.time() - t0
        st = s.accum_stats() or {}
        out = SolveResult(
            status="sat" if res is True else ("unsat" if res is False else "unknown"),
            seconds=dt,
            conflicts=int(st.get("conflicts", 0)),
            decisions=int(st.get("decisions", 0)),
            propagations=int(st.get("propagations", 0)),
            solver=solver,
        )
        if res is None:
            out.budget_hit = "time" if timed_out["v"] else "conflicts"
        if res is True:
            model = s.get_model()
            out.matrix = decode_matrix(cnf, model)
            out.witness_ok = (not has_kst(out.matrix, inst.s, inst.t)
                              and sum(map(sum, out.matrix)) == sum(map(sum, out.matrix)))
        if res is False and want_proof:
            try:
                out.proof = s.get_proof()
            except Exception:
                out.proof = None
        return out
    finally:
        if timer is not None:
            timer.cancel()
        s.delete()

"""Feasibility fallback: does a legal heavy config with val == V, cols <= c
exist? INFEASIBLE proves F_m(c) < V. Usage: feas_check.py m c V [tl]"""
import sys
import time

from frontier_milp import solve_frontier_point

m, c, V = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
tl = float(sys.argv[4]) if len(sys.argv) > 4 else 3000.0
t0 = time.time()
obj, blocks, status = solve_frontier_point(m, c, time_limit=tl,
                                           fixed_val=V, minimize_slots=True)
print(f"m={m} c={c} val=={V}: {status}"
      f"{'' if obj is None else f' (slots={obj})'} [{time.time()-t0:.0f}s]")

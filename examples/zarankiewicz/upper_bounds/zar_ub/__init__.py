"""zar_ub — the computational core of the Zarankiewicz upper-bound pipeline.

Pipeline (Tan 2022, arXiv:2203.02283, extended):
  instance (m,n,s,t,w)  -- does an m x n K_{s,t}-free 0/1 matrix with >= w ones exist?
    -> row partitions x column partitions (Tan Algorithm 1, Arguments A and I)
    -> case = (row-sum partition, column-sum partition), both non-increasing
    -> baseline prunes (Argument D etc.)               [zar_ub.cases]
    -> evolved prunes (Python kill + Lean proof)       [zar_ub.lean_gate]
    -> SAT on survivors with fixed sums + lex symmetry breaking [zar_ub.encoding, zar_ub.solve]
  If every case is pruned or refuted, z(m,n;s,t) < w.

Convention: z(m,n;s,t) = max ones in an m x n 0/1 matrix with no s rows sharing
t common 1-columns (no all-ones s x t submatrix). z(m,n;s,t) = z(n,m;t,s).
"""
from .known import Instance, ub_counting, ub_known, exact_value  # noqa: F401
from .partitions import row_partitions, column_partitions  # noqa: F401
from .cases import Case, enumerate_cases, BASELINE_PRUNES, baseline_kill  # noqa: F401

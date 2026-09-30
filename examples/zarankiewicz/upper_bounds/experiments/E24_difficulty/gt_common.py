"""E24 / A1 ground truth: shared helpers (owner A1).

One fresh pysat cadical195 solver per call (the same solver and options as
zar_ub.difficulty.label_case), a conflict cap and a wall-clock limit.  The
conflict count of a fresh run is deterministic for a fixed encoding, so a case
refuted at cap C' < C has the same d under any cap >= C'.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, Iterable, List, Optional

UB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if UB not in sys.path:
    sys.path.insert(0, UB)

from zar_ub.difficulty import log2_volume  # noqa: E402
from zar_ub.encoding import encode_case, has_kst  # noqa: E402
from zar_ub.known import Instance  # noqa: E402
from zar_ub.solve import solve_cnf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(UB, "cache")
SOLVER = "cadical195"


def cell_tag(inst: Instance, trust: str) -> str:
    return inst.tag + ("_pure" if trust == "pure" else "")


def solve_one(inst: Instance, rows, cols, cap: int, time_limit: float) -> dict:
    """Fresh solver, cap conflicts, wall limit.  Returns a plain dict."""
    cnf = encode_case(inst, rows, cols)
    t0 = time.time()
    r = solve_cnf(cnf, inst, solver=SOLVER, conf_budget=cap, time_limit=time_limit)
    out = {
        "status": r.status,
        "conflicts": int(r.conflicts),
        "decisions": int(r.decisions),
        "propagations": int(r.propagations),
        "seconds": round(time.time() - t0, 3),
        "budget_hit": r.budget_hit,
        "cap": int(cap),
        "time_limit": float(time_limit),
    }
    if r.status == "sat":
        A = r.matrix
        ok = bool(
            A is not None
            and not has_kst(A, inst.s, inst.t)
            and [sum(x) for x in A] == list(rows)
            and [sum(A[i][j] for i in range(inst.m)) for j in range(inst.n)] == list(cols)
        )
        out["witness_ok"] = ok
        out["matrix"] = A
        out["ones"] = sum(map(sum, A)) if A else None
    return out


def _worker(args):
    inst_d, rows, cols, cap, tl, meta = args
    inst = Instance(**inst_d)
    res = solve_one(inst, rows, cols, cap, tl)
    return meta, rows, cols, res


def read_jsonl(path: str) -> List[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass  # a torn last line of a killed run
    return out


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def gt_row(inst: Instance, trust: str, rows, cols, status: str, d: int, cap: int, c2000: Optional[int],
           source: str, **extra) -> dict:
    row = {
        "cell": cell_tag(inst, trust),
        "m": inst.m,
        "n": inst.n,
        "s": inst.s,
        "t": inst.t,
        "w": inst.w,
        "trust": trust,
        "rows": list(rows),
        "cols": list(cols),
        "status": status,
        "d": int(d),
        "cap": int(cap),
        "c2000": None if c2000 is None else int(c2000),
        "log2_volume": log2_volume(inst, rows),
        "source": source,
    }
    row.update(extra)
    return row


def key_of(row: dict) -> tuple:
    return (row["cell"], tuple(row["rows"]), tuple(row["cols"]))

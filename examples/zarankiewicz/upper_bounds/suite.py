"""The evaluation suite (design §5.1, §8.3): which instances a candidate prune
library is scored on, with their roles and weights.

TRAIN   (omega = 1, pure mode, exact labels)  exactly known cells at w = z+1: every case
        is empty and labelled with its refutation cost d (§6).  A prune earns credit
        for the difficulty of the scored survivors it kills.
BATTERY (pure mode) the same cells at w = z, whose 'sat' cases carry witnesses, plus the
        record profiles of data/witnesses_33.json ((11,21)=116, (12,22)=132): any candidate
        whose Python kill fires on one is unsound and is rejected before Lean runs.
TARGET  (omega = 2, --trust tan2022, censored labels) open cells; ZAR_UB_TARGETS="m,n,w;..."
        overrides DEFAULT_TARGETS (set it to "" or "none" for no targets).  Only cached
        tables are loaded: build them with  python -m zar_ub table M N 3 3 W --trust tan2022 --baseline
GEN     (omega = 1 inside G_gen, held out, other (s,t)) (7,7;2,2) w=22, (8,8;2,2) w=25,
        (9,9;4,4) w=62 -- w = z+1 from Tan 2022 Tables 2 and 4 (z_2(7,7)=21, z_2(8,8)=24,
        z_4(9,9)=61); built in pure mode; never shown in the prompt.

Band rule: a TRAIN cell graduates when the population's mean gain_I on it exceeds
BAND_THRESHOLD (0.75); when the first cell graduates the band cells (12,13)87,
(13,13)93, (10,14)78 enter TRAIN.  The state lives in cache/suite_state.json and is
advanced by `update_band_rule(mean_gain_by_cell)` (called by the closure daemon /
integrator, never inside evaluate()).

Tables are built once (`python -m zar_ub table M N S T W --pure --baseline`) and cached.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

from zar_ub import Instance, exact_value
from zar_ub.casetable import CaseTable, CaseRecord, load_table, CACHE_DIR

_HERE = os.path.dirname(os.path.abspath(__file__))
SUITE_VERSION = "2026-09-21.v2"

TRAIN_CELLS: List[Tuple[int, int]] = [(9, 9), (9, 10), (10, 10), (10, 11), (11, 11), (11, 12), (12, 12)]
BAND_CELLS: List[Tuple[int, int]] = [(12, 13), (13, 13), (10, 14)]
BAND_THRESHOLD = 0.75
GEN_CELLS: List[Tuple[int, int, int, int, int]] = [(7, 7, 2, 2, 22), (8, 8, 2, 2, 25), (9, 9, 4, 4, 62)]
DEFAULT_TARGETS = "13,17,117;13,18,122;15,17,133;9,23,104;12,18,109"
OMEGA: Dict[str, float] = {"train": 1.0, "battery": 1.0, "target": 2.0, "gen": 1.0}
WITNESSES_PATH = os.path.join(_HERE, "data", "witnesses_33.json")
STATE_PATH = os.path.join(CACHE_DIR, "suite_state.json")


# ---------------------------------------------------------------------------
# band rule state
# ---------------------------------------------------------------------------
def load_state(path: str = STATE_PATH) -> dict:
    try:
        with open(path) as f:
            d = json.load(f)
        d.setdefault("graduated", [])
        d.setdefault("entered", [])
        d.setdefault("history", [])
        return d
    except (OSError, ValueError):
        return {"graduated": [], "entered": [], "history": []}


def save_state(state: dict, path: str = STATE_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=1)
    return path


def train_cells(state: Optional[dict] = None) -> List[Tuple[int, int]]:
    state = state if state is not None else load_state()
    grad = {tuple(c) for c in state.get("graduated", [])}
    entered = [tuple(c) for c in state.get("entered", [])]
    cells = [c for c in TRAIN_CELLS if c not in grad] + [c for c in entered if c not in grad]
    return cells


def update_band_rule(
    mean_gain_by_cell: Dict[Tuple[int, int], float],
    state: Optional[dict] = None,
    threshold: float = BAND_THRESHOLD,
    persist: bool = True,
    note: str = "",
) -> dict:
    """Apply §5.1's band rule: cells whose population-mean gain_I exceeds the threshold
    graduate; the first graduation lets BAND_CELLS enter.  Returns the new state."""
    state = state if state is not None else load_state()
    grad = {tuple(c) for c in state["graduated"]}
    changed = []
    for cell, g in mean_gain_by_cell.items():
        cell = tuple(cell)
        if g > threshold and cell not in grad and cell in train_cells(state):
            grad.add(cell)
            changed.append(cell)
    if changed:
        state["graduated"] = sorted(list(c) for c in grad)
        entered = {tuple(c) for c in state["entered"]}
        for b in BAND_CELLS:
            entered.add(b)
        state["entered"] = sorted(list(c) for c in entered)
        state["history"].append(
            {
                "graduated": [list(c) for c in changed],
                "gains": {f"{m},{n}": g for (m, n), g in mean_gain_by_cell.items()},
                "note": note,
            }
        )
        if persist:
            save_state(state)
    return state


# ---------------------------------------------------------------------------
# instances per role
# ---------------------------------------------------------------------------
def _z(m: int, n: int) -> int:
    z = exact_value(m, n, 3, 3)
    if z is None:
        raise ValueError(f"({m},{n}) has no exact value in data/exact_33.csv")
    return z


def train_instances(state: Optional[dict] = None) -> List[Instance]:
    return [Instance(m, n, 3, 3, _z(m, n) + 1) for m, n in train_cells(state)]


def battery_instances(state: Optional[dict] = None) -> List[Instance]:
    return [Instance(m, n, 3, 3, _z(m, n)) for m, n in train_cells(state)]


def gen_instances() -> List[Instance]:
    return [Instance(m, n, s, t, w) for m, n, s, t, w in GEN_CELLS]


def parse_targets(raw: str) -> List[Instance]:
    out = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk or chunk.lower() == "none":
            continue
        parts = [int(x) for x in chunk.split(",")]
        if len(parts) == 3:
            m, n, w = parts
            out.append(Instance(m, n, 3, 3, w))
        elif len(parts) == 5:
            out.append(Instance(*parts))
        else:
            raise ValueError(f"ZAR_UB_TARGETS chunk {chunk!r}: expected m,n,w or m,n,s,t,w")
    return out


def target_instances() -> List[Instance]:
    raw = os.environ.get("ZAR_UB_TARGETS")
    return parse_targets(DEFAULT_TARGETS if raw is None else raw)


# ---------------------------------------------------------------------------
# witness battery (in-memory tables from data/witnesses_33.json)
# ---------------------------------------------------------------------------
def witness_tables(path: str = WITNESSES_PATH) -> List[Tuple[Instance, CaseTable]]:
    """One single-record battery table per record witness.  The matrix is re-checked
    for K_{s,t}-freeness and profile agreement here, so a corrupted file cannot
    silently turn into a false 'sat' label."""
    from zar_ub.encoding import has_kst

    out = []
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return out
    for e in data.get("witnesses", []):
        m, n, s, t = int(e["m"]), int(e["n"]), int(e["s"]), int(e["t"])
        w = int(e.get("w", e["z"]))
        M = [[int(c) for c in r] for r in e["matrix"]]
        rows = [sum(r) for r in M]
        cols = [sum(M[i][j] for i in range(m)) for j in range(n)]
        if (
            len(M) != m
            or any(len(r) != n for r in M)
            or rows != list(e["rows"])
            or cols != list(e["cols"])
            or sum(rows) != w
            or has_kst(M, s, t)
        ):
            raise RuntimeError(f"witnesses_33.json: entry ({m},{n}) failed verification")
        inst = Instance(m, n, s, t, w)
        rec = CaseRecord(
            list(rows),
            list(cols),
            "",
            {
                "status": "sat",
                "conflicts": 0,
                "seconds": 0.0,
                "budget_cap": 0,
                "fixed_by_propagation": 0.0,
                "log2_volume": 0.0,
                "nvars": 0,
                "nclauses": 0,
                "c2000": 0,
                "fhat": None,
                "d": None,
                "censored": False,
                "witness": True,
                "source": e.get("source", ""),
            },
            d=1.0,
            censored=False,
        )
        tab = CaseTable(
            inst={"m": m, "n": n, "s": s, "t": t, "w": w},
            n_row_partitions=1,
            n_col_partitions=1,
            records=[rec],
            external_facts=[],
            use_table=False,
            kind="battery",
            omega=OMEGA["battery"],
            label_mode="exact",
            trust="witness",
            path=None,
        )
        tab.refresh()
        out.append((inst, tab))
    return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _load(inst: Instance, prefer_pure: bool) -> Optional[CaseTable]:
    order = (False, True) if prefer_pure else (True, False)
    for ut in order:
        tab = load_table(inst, use_table=ut)
        if tab is not None:
            return tab
    return None


def load_suite(
    verbose: bool = False, targets: Optional[List[Instance]] = None, state: Optional[dict] = None
) -> Dict[str, List[Tuple[Instance, CaseTable]]]:
    """{'train','battery','target','gen'} -> list of (Instance, CaseTable); each table
    carries kind and omega.  Missing tables are skipped (with a build hint when verbose)."""
    state = state if state is not None else load_state()
    suite: Dict[str, List[Tuple[Instance, CaseTable]]] = {"train": [], "battery": [], "target": [], "gen": []}
    plan = (
        ("train", train_instances(state), True),
        ("battery", battery_instances(state), True),
        ("target", targets if targets is not None else target_instances(), False),
        ("gen", gen_instances(), True),
    )
    for kind, insts, prefer_pure in plan:
        for inst in insts:
            tab = _load(inst, prefer_pure)
            if tab is None:
                if verbose:
                    flag = "--pure" if prefer_pure else "--trust tan2022"
                    print(
                        f"[suite] missing {kind} table for {inst.tag}; build it with: python -m zar_ub table "
                        f"{inst.m} {inst.n} {inst.s} {inst.t} {inst.w} {flag} --baseline"
                    )
                continue
            tab.kind = kind
            tab.omega = OMEGA[kind]
            if verbose and kind in ("train", "target", "gen") and tab.baseline_lean_mask is None and tab.records:
                print(
                    f"[suite] {kind} {inst.tag}: no baseline_lean_mask (run: python -m zar_ub table "
                    f"{inst.m} {inst.n} {inst.s} {inst.t} {inst.w} {'--pure' if not tab.use_table else '--trust tan2022'} --baseline)"
                )
            suite[kind].append((inst, tab))
    suite["battery"].extend(witness_tables())
    return suite


def suite_hash(suite: Dict[str, List[Tuple[Instance, CaseTable]]]) -> str:
    """sha1 over the suite's table hashes (the `table_hash` metric of §5.4)."""
    h = hashlib.sha1(SUITE_VERSION.encode())
    for kind in ("train", "battery", "target", "gen"):
        for inst, tab in suite.get(kind, []):
            h.update(f"{kind}:{inst.tag}:{tab.table_hash}:{tab.baseline_lean_mask is not None};".encode())
    return h.hexdigest()[:16]


def describe(suite: Dict[str, List[Tuple[Instance, CaseTable]]]) -> str:
    lines = [f"suite {SUITE_VERSION} hash {suite_hash(suite)}"]
    for kind in ("train", "battery", "target", "gen"):
        for inst, tab in suite.get(kind, []):
            sc = len(tab.scored_indices())
            lines.append(
                f"  {kind:7s} {inst.tag:22s} omega={tab.omega:.0f} cases={len(tab.records):5d} "
                f"scored={sc:5d} W={tab.work():12.0f} censored={sum(1 for r in tab.records if r.censored):4d} "
                f"mask={'yes' if tab.baseline_lean_mask is not None else 'no ':3s} hash={tab.table_hash} "
                f"{tab.trust}/{tab.label_mode}"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe(load_suite(verbose=True)))

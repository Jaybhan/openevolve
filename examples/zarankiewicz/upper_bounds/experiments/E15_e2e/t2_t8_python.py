#!/usr/bin/env python
"""E15: T-2 uncached gate (mask length must equal the case count) and T-8 Python side
(facts_for + enumerate_cases(use_table=True) on the three cited-mode closure cells).

Run from upper_bounds/ with ZAR_UB_NO_LLM=1:
    python experiments/E15_e2e/t2_t8_python.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, UB)
os.chdir(UB)
os.environ.setdefault("ZAR_UB_NO_LLM", "1")

from zar_ub import lean_gate, casetable, ledger, cases  # noqa: E402
from zar_ub.known import Instance  # noqa: E402

out = {}

# ---- T-2: uncached gate of initial_program.py on (10,11;3,3) w=65 pure
inst = Instance(10, 11, 3, 3, 65)
tab = casetable.load_table(inst, use_table=False)
src = None
ns = {}
with open("initial_program.py") as f:
    code = f.read()
exec(compile(code, "initial_program.py", "exec"), ns)
src = ns["LEAN_SOURCE"]
case_list = [(r.rows, r.cols) for r in tab.records] if tab is not None else None
t0 = time.time()
try:
    res = lean_gate.run_gate(inst, src, cases=case_list, use_cache=False, tag="e15_t2")
except TypeError:
    res = lean_gate.run_gate_multi([(inst, case_list)], src, use_cache=False, tag="e15_t2")[0]
dt = time.time() - t0
out["T2_uncached"] = {
    "ladder": res.ladder, "axioms": res.axioms, "mask_len": None if res.kill_mask is None else len(res.kill_mask),
    "n_cases": None if tab is None else len(tab.records), "kills": None if res.kill_mask is None else sum(res.kill_mask),
    "seconds_gate": res.seconds, "wall": round(dt, 2), "errors": res.errors[:3], "cache_hit": res.cache_hit,
    "scored_survivors": None if tab is None else len(tab.scored_indices()),
    "killed_scored_survivors": None if (tab is None or res.kill_mask is None) else
    sum(1 for r, k in zip(tab.records, res.kill_mask) if k and not r.baseline_lean_kill),
}
print("T2", json.dumps(out["T2_uncached"]), flush=True)

# ---- T-8 Python side
out["T8"] = {}
for (m, n, w) in [(10, 21, 107), (11, 19, 107), (11, 20, 112)]:
    P = Instance(m, n, 3, 3, w)
    facts = ledger.facts_for(P, "tan2022")
    t0 = time.time()
    cs = cases.enumerate_cases(P, use_table=True)
    dt = time.time() - t0
    try:
        pure = cases.enumerate_cases(P, use_table=False)
        n_pure = len(pure)
    except Exception as e:  # noqa: BLE001
        n_pure = f"error: {e}"
    tags = sorted({f.tag for f in facts})
    out["T8"][f"{m},{n},{w}"] = {"n_cases_trust": len(cs), "n_cases_pure": n_pure, "n_facts": len(facts),
                                "fact_tags": tags, "wall": round(dt, 2)}
    print("T8", m, n, w, json.dumps(out["T8"][f"{m},{n},{w}"]), flush=True)

with open(os.path.join(HERE, "out", "t2_t8_python.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)

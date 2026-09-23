"""E3: build pure-mode case tables (no external table) for a ladder of exactly
known cells, at w = z+1 (all cases must be UNSAT: the 'refute everything'
environment) and at w = z (some cases SAT with witnesses: the counterexample
battery for empirical soundness checks of candidate prunes)."""
import sys, time, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from zar_ub import Instance, exact_value
from zar_ub.casetable import build_table, load_table

CELLS = [(9,9),(9,10),(10,10),(10,11),(11,11),(9,12),(11,12),(12,12),(10,14),(12,13),(13,13)]
for (m, n) in CELLS:
    z = exact_value(m, n, 3, 3)
    for w in (z + 1, z):
        inst = Instance(m, n, 3, 3, w)
        if load_table(inst, use_table=False):
            print(f"[skip] {inst.tag} cached"); continue
        t0 = time.time()
        tab = build_table(inst, conf_cap=20000, time_limit=60, verbose=True, use_table=False)
        tab.save()
        s = tab.summary()
        print(f"DONE {inst.tag}: cases={s['cases']} baseline_killed={s['baseline_killed']} survivors={s['survivors']} status={s['survivor_status']} total_diff={s['total_difficulty']} {time.time()-t0:.1f}s", flush=True)

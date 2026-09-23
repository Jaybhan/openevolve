"""E11: calibration cells with known answers but real size — (10,20)=102 and
(11,21)=116 (Tan exact) — build tables at w=z+1 in pure and cited mode with a
short probe cap, to see how many cases the counting prunes leave and how hard
they are before any evolved prune."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from zar_ub import Instance
from zar_ub.casetable import build_table, load_table
for (m, n, w) in [(10, 20, 103), (11, 21, 117)]:
    for use_table in (False, True):
        inst = Instance(m, n, 3, 3, w)
        if load_table(inst, use_table=use_table):
            print("[skip]", inst.tag, use_table); continue
        t0 = time.time()
        tab = build_table(inst, conf_cap=2000, time_limit=20, verbose=False, use_table=use_table)
        tab.save(); s = tab.summary()
        print(f"DONE {inst.tag} cited={use_table}: rp={s['row_partitions']} cp={s['col_partitions']} cases={s['cases']} refD_killed={s['baseline_killed']} status={s['survivor_status']} ext={len(s['external_facts'])} {time.time()-t0:.0f}s", flush=True)

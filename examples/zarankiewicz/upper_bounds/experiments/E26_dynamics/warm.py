"""E26 step 0: evaluate every reachable genome once (fills the S1 mask cache and the gate caches) so
that no evaluation inside an OpenEvolve run hits the 190-350 s first-time S1 gate.  Also the
per-genome score table used in the report (results/genome_scores.json).

usage: python experiments/E26_dynamics/warm.py <gstr> [<gstr> ...]   (appends to results/genome_scores.jsonl)
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("ZAR_UB_NO_LLM", "1")
import genome as G  # noqa: E402
import evaluator_variant as EV  # noqa: E402

os.makedirs(os.path.join(HERE, "programs"), exist_ok=True)
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
for gs in sys.argv[1:]:
    p = os.path.join(HERE, "programs", f"{gs}.py")
    with open(p, "w") as f:
        f.write(G.render(G.parse_gstr(gs), 0))
    t0 = time.time()
    r = EV.evaluate(p)
    m = r.metrics
    row = {"genome": gs, "seconds": round(time.time() - t0, 1),
           **{k: m.get(k) for k in ("combined_score_v0", "sc_V0", "sc_V0S1", "sc_V345", "sc_VR", "lean_ladder",
                                    "sound_battery", "gain_concentration", "n_cells_gained", "s1_ladder_min",
                                    "s1_witnessed_kills", "proven_gain_v0", "gate_seconds")}}
    row["cells"] = r.artifacts.get("e26_variant", "")
    with open(os.path.join(HERE, "results", "genome_scores.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "cells"}), flush=True)

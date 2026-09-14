"""Discovery v2: repair search from current engine floors (or v1 witnesses)
for the cells still below the proven value.  Offline tool; witnesses are
verified twice (engine verifier + snapshot reference) before being saved."""

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import offline_discovery as OD  # noqa: E402
import zarankiewicz as Z  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "snap", os.path.join(HERE, "..", "harness", "evaluator_snapshot.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

OUT = OD.OUT
found = {}
if os.path.exists(OUT):
    found = json.load(open(OUT))

targets = []
for (m, n), z in sorted(ev.KST_EXACT_VALUE.items()):
    A, prov, _ = Z.construct(m, n, 3, 3)
    e = int(A.sum())
    key = f"{m},{n}"
    ke = found.get(key, {}).get("edges", 0)
    if max(ke, e) >= z:
        continue
    if ke > e:
        blocks = [tuple(b) for b in found[key]["blocks"]]
        A = Z.blocks_to_matrix(m, n, blocks)
        e = ke
    targets.append((m, n, z, e, A))
print("targets:", [(m, n, z - e) for m, n, z, e, _ in targets], flush=True)

for m, n, z, e, A in targets:
    print(f"cell {m}x{n}: {e} -> {z}", flush=True)
    bestA, beste = A, e
    for seed in range(1, 9):
        A2, e2 = OD.improve(m, n, 3, 3, bestA, z, seconds=60.0,
                            seed=seed, kmax=5)
        if e2 > beste:
            bestA, beste = A2, e2
        if beste >= z:
            break
    ok = Z.verify_kst_free(bestA, 3, 3) and \
        ev.count_kst_violations(bestA, 3, 3) == 0
    print(f"  -> {beste}/{z} verified={ok}", flush=True)
    if ok and beste > e:
        found[f"{m},{n}"] = {"m": m, "n": n, "edges": beste, "exact": z,
                             "blocks": [list(b) for b in
                                        OD.matrix_blocks(bestA)]}
        json.dump(found, open(OUT, "w"), indent=1)
print("DONE", flush=True)

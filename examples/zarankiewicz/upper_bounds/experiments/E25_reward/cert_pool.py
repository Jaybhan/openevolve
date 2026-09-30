"""E25 step 0: the pool of Farkas certificates per table (pair-codegree LP, zar_ub.lemmas).

For every universe table: library survivors S_I in decreasing-d order; greedy: a survivor not yet
killed by a certificate of this table gets an LP solve (farkas_certificate); a certificate found is
re-verified exactly and its kill set over S_I recorded.  The union of kill sets is exactly the set of
LP-refutable survivors (a case is LP-refutable iff it has its own certificate), so the pool bounds
what ANY SCHEMA_DATA-only program can earn on the table.

Output: results/cert_pool.json  {key: {"S": [...], "d": {idx: d}, "certs": [{"case", "y", "kills"}], "lp_seconds"}}
Uses at most 2 worker processes (no SAT solver; scipy HiGHS LPs).
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, _HERE)

import universe as U  # noqa: E402


def one(key: str) -> tuple:
    from zar_ub import reward
    from zar_ub.lemmas import farkas_certificate, farkas_system, cert_kills_system, sparsify
    t0 = time.time()
    tab = U.load(key)
    inst = U.parse_key(key)
    m, n, s, t = inst["m"], inst["n"], inst["s"], inst["t"]
    S = reward.survivor_indices(tab, None)
    cap = int(getattr(tab, "conf_cap", 0) or 0)
    d = {i: reward.rec_difficulty(tab.records[i], cap) for i in S}
    order = sorted(S, key=lambda i: (-d[i], i))
    systems = {i: farkas_system(m, n, s, t, tab.records[i].rows, tab.records[i].cols) for i in S}
    killed = set()
    certs = []
    n_lp = 0
    for i in order:
        if i in killed or s < 2:
            continue
        n_lp += 1
        try:
            y = farkas_certificate(m, n, s, t, tab.records[i].rows, tab.records[i].cols)
        except Exception:  # noqa: BLE001
            y = None
        if not y or not cert_kills_system(systems[i], y):
            continue
        y = sparsify(y)
        kills = [j for j in S if cert_kills_system(systems[j], y)]
        killed.update(kills)
        certs.append({"case": i, "y": y, "kills": kills})
    return key, {"S": S, "d": {str(i): d[i] for i in S}, "certs": certs, "n_lp": n_lp,
                 "lp_seconds": time.time() - t0, "m": m, "n": n, "s": s, "t": t}


def main():
    uni = U.universe()
    out_path = os.path.join(_HERE, "results", "cert_pool.json")
    have = {}
    if os.path.exists(out_path):
        have = json.load(open(out_path))
    keys = [k for k in uni if k not in have]
    # biggest tables first for load balance
    keys.sort(key=lambda k: -os.path.getsize(uni[k]["path"]))
    with Pool(2) as pool:
        for key, res in pool.imap_unordered(one, keys):
            have[key] = res
            nk = len(set(j for c in res["certs"] for j in c["kills"]))
            print(f"{key}: S={len(res['S'])} certs={len(res['certs'])} LP-refutable={nk} lp={res['n_lp']} "
                  f"{res['lp_seconds']:.1f}s", flush=True)
            with open(out_path, "w") as f:
                json.dump(have, f)


if __name__ == "__main__":
    main()

"""E24 / D5-evaluate: the gain-simulation tables and the REAL kill masks.

For every ground-truth cell whose hard cases have ground truth (complete tables, or the three
target tables with A1's uniform 150-case sample of their 20k-censored cases) this writes one line
to features_gaintables.jsonl:
  universe  = the library survivors S (table baseline_lean_mask false, not SAT) with known labels
  weight    = 1, or (#censored-at-20k in the table)/(#sampled) for sampled censored cases
  d_true    = exact d; for cases still open at 2M, the lower bound 2,000,000 (flagged)
  masks     = argD (cases.kill_row_argument_d | kill_col_argument_d), DGH (E13_dgh4/dgh.py),
              Farkas certificates (lemmas.farkas_certificate on seeded survivors, applied to the
              whole table with lemmas.mirror_kill; one mask per certificate + their union),
              profile-threshold rules (rule-like masks correlated with difficulty), and seeded
              random masks of density 5/20/50 % (10 each).
Deterministic.  No SAT solver is run (the Farkas search solves small LPs with HiGHS).
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, UB)
sys.path.insert(0, os.path.join(UB, "experiments", "E13_dgh4"))

from eval_collect import case_key, load_gt, regime  # noqa: E402
from zar_ub.known import Instance  # noqa: E402
from zar_ub import cases as zc, lemmas  # noqa: E402
from zar_ub.casetable import table_from_json  # noqa: E402
import dgh  # noqa: E402

OUT = os.path.join(HERE, "features_gaintables.jsonl")
OPEN_D = 2_000_000
MIN_HARD_SURV = 5
FARKAS_TRIES = int(os.environ.get("E24_FARKAS_TRIES", "40"))
FARKAS_MAX = 8


def quantile(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * (len(s) - 1))))]


def build_line(cell, rs, U, sampled, n_open20, n_hard, w_s, uni):
    nh = sum(u["regime"] in ("hard", "hard_open") for u in U)
    if nh < MIN_HARD_SURV:
        return None
    U.sort(key=lambda u: u["key"])
    m, n, s, t, w = (rs[0][k] for k in ("m", "n", "s", "t", "w"))
    inst = Instance(m, n, s, t, w)
    masks = {}
    masks["argD"] = [bool(zc.kill_row_argument_d(inst, u["rows"], u["cols"]) or
                          zc.kill_col_argument_d(inst, u["rows"], u["cols"])) for u in U]
    masks["DGH"] = [bool(dgh.dgh_kill(m, n, s, t, w, u["rows"], u["cols"])) for u in U]
    # Farkas certificates on seeded survivors
    t0 = time.time()
    P = SimpleNamespace(m=m, n=n, s=s, t=t)
    order = list(range(len(U)))
    random.Random(f"E24farkas|{cell}").shuffle(order)
    certs, tries = [], 0
    for i in order:
        if tries >= FARKAS_TRIES or len(certs) >= FARKAS_MAX:
            break
        u = U[i]
        if any(lemmas.certificate_kills(m, n, s, t, u["rows"], u["cols"], y) for y in certs):
            continue
        tries += 1
        try:
            y = lemmas.farkas_certificate(m, n, s, t, u["rows"], u["cols"])
        except Exception:  # noqa: BLE001
            y = None
        if y and lemmas.certificate_kills(m, n, s, t, u["rows"], u["cols"], y):
            certs.append(lemmas.sparsify(y))
    for k, y in enumerate(certs):
        sd = {"farkas": [{"m": m, "n": n, "s": s, "t": t, "y": list(y)}]}
        masks[f"farkas{k}"] = [bool(lemmas.mirror_kill(sd, P, u["rows"], u["cols"])) for u in U]
    if certs:
        sd = {"farkas": [{"m": m, "n": n, "s": s, "t": t, "y": list(y)} for y in certs]}
        masks["farkas_union"] = [bool(lemmas.mirror_kill(sd, P, u["rows"], u["cols"])) for u in U]
    t_f = time.time() - t0
    # profile-threshold rules (rule-like, correlated with difficulty in either direction)
    r0 = [u["rows"][0] for u in U]
    c0 = [u["cols"][0] for u in U]
    vol = [float(u["log2_volume"] or 0.0) for u in U]
    dc = [len(set(u["cols"])) for u in U]
    masks["rule_rowmax_hi"] = [x >= quantile(r0, 0.75) for x in r0]
    masks["rule_colmax_hi"] = [x >= quantile(c0, 0.75) for x in c0]
    masks["rule_vol_lo"] = [x <= quantile(vol, 0.25) for x in vol]
    masks["rule_vol_hi"] = [x >= quantile(vol, 0.75) for x in vol]
    masks["rule_distinct_cols_lo"] = [x <= quantile(dc, 0.25) for x in dc]
    rng = random.Random(f"E24rand|{cell}")
    for dens in (0.05, 0.20, 0.50):
        for j in range(10):
            masks[f"rand{int(dens * 100)}_{j}"] = [rng.random() < dens for _ in U]
    line = {"cell": cell, "m": m, "n": n, "s": s, "t": t, "w": w, "trust": rs[0].get("trust"),
            "sampled": sampled, "n_open20k": n_open20, "n_hard": n_hard, "weight_sampled": w_s,
            "n_universe": len(U), "n_hard_surv": nh, "n_open_surv": sum(u["open"] for u in U),
            "farkas_tries": tries, "farkas_seconds": round(t_f, 2), "n_certs": len(certs),
            "universe": [{k: u[k] for k in ("key", "regime", "d_true", "open", "w")} for u in U],
            "masks": masks, "universe_kind": uni}
    return line


def main():
    gt, path = load_gt()
    by = defaultdict(list)
    for r in gt:
        by[r["cell"]].append(r)
    out = []
    for cell in sorted(by):
        rs = by[cell]
        reg = [regime(r) for r in rs]
        n_open20 = sum(g == "open20k" for g in reg)
        n_hard = sum(g in ("hard", "hard_open") for g in reg)
        if n_hard == 0:
            continue
        sampled = n_open20 > 0
        tab = table_from_json(json.load(open(os.path.join(UB, "cache", rs[0]["table"]))))
        mask = tab.baseline_lean_mask or [False] * len(tab.records)
        lib = {(tuple(r.rows), tuple(r.cols)): bool(mask[i]) for i, r in enumerate(tab.records)}
        w_s = (n_open20 + n_hard) / n_hard if sampled else 1.0
        for uni in ("survivors", "all"):
            U = []
            for r, g in zip(rs, reg):
                if g == "open20k" or r["status"] == "sat":
                    continue
                if uni == "survivors" and lib.get((tuple(r["rows"]), tuple(r["cols"])), False):
                    continue
                U.append({
                    "key": case_key(cell, r["rows"], r["cols"]), "rows": r["rows"], "cols": r["cols"],
                    "regime": g, "d_true": float(OPEN_D if g == "hard_open" else r["d"]),
                    "open": g == "hard_open", "w": w_s if (sampled and g in ("hard", "hard_open")) else 1.0,
                    "log2_volume": r.get("log2_volume"),
                })
            line = build_line(cell, rs, U, sampled, n_open20, n_hard, w_s, uni)
            if line is not None:
                out.append(line)
                print(cell, uni, "U", line["n_universe"], "hard", line["n_hard_surv"], "sampled", sampled,
                      "w", round(w_s, 2), {k: sum(v) for k, v in line["masks"].items() if not k.startswith("rand")},
                      f"farkas {line['farkas_tries']} tries {line['farkas_seconds']}s", flush=True)
    with open(OUT, "w") as fh:
        for line in out:
            fh.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()

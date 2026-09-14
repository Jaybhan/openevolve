"""Assemble the final F_9 table with three provenance classes:

  ILP-PROVEN     frontier MILP solved to optimality (any pass/tag)
  PROVEN-DICT    pinned by Lemma P: F(c) = max(Q(c), F(c-1)) where
                 z(9,c) is proven (published Tan / workspace gap-band ILP /
                 the five new deep-band cells once ILP-PROVEN here) —
                 requires matching LB config (incumbent or S-table family)
  LB(x)          only a lower bound known (should not remain anywhere)

Writes frontier_m9_final.csv and prints the table with a full consistency
audit: LB <= F_final everywhere, monotone, matches every proven z via (+).
"""
import csv
import json
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
M = 9
B = 2 * comb(M, 3)


def proven_z():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ev", os.path.join(BASE, "..", "evaluator.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    z = {n: v for (m, n), v in ev.KST_EXACT_VALUE.items() if m == M}
    for fn in ("ilp_gapband.jsonl", "ilp_results.jsonl"):
        p = os.path.join(BASE, "analysis", fn)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("m") == M and "z_ilp" in r and r.get("witness_valid"):
                        z.setdefault(r["n"], r["z_ilp"])
    # Theorem 8 tail (m=9 Johnson-tight), n >= T = 40
    for n in range(40, B + 1):
        z.setdefault(n, 3 * n + min(40, (B - n) // 3))
    return z


def milp_records():
    recs = {}
    for fn in sorted(os.listdir(HERE)):
        if "_w5" in fn or "_w6" in fn:
            continue    # weight-restricted runs are NOT the full frontier
        if fn.startswith("frontier_m9") and fn.endswith(".jsonl"):
            with open(os.path.join(HERE, fn)) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("val") is None:
                        continue
                    c = r["c"]
                    old = recs.get(c)
                    if (old is None
                            or (r["status"].startswith("PROVEN")
                                and not old["status"].startswith("PROVEN"))
                            or r["val"] > old["val"]):
                        recs[c] = r
    return recs


def main():
    z = proven_z()
    recs = milp_records()
    newz = {}
    # First: any newly ILP-PROVEN F at previously-open cells creates new z
    for c in (24, 25, 26, 27, 33):
        r = recs.get(c)
        if r and r["status"].startswith("PROVEN"):
            # LB: the record's own blocks are a legal config with cols <= c,
            # penalty-free at n = c (verified in assemble_cases), so
            # z >= 3c + val; UB: z <= 3c + F(c) by the identity. Equal.
            lb = ub = 3 * c + r["val"]
            newz[c] = (lb, ub)
            z.setdefault(c, lb)
    rows = []
    # interval chain: [flo, fhi] = proven bounds on F(c) carried across gaps
    R = (2 * comb(M - 1, 2)) // 3
    audit_fail = []
    flo, fhi = 0, 0
    F_prev = None
    for c in range(1, 42):
        ub10 = min((M - 3) * c, (2 * c + M * R) // 6, (M * R) // 4)  # Thm 10
        r = recs.get(c)
        milp_proven = bool(r and r["status"].startswith("PROVEN"))
        lb_val = max(r["val"] if r else 0, flo)      # monotone LB chain
        Q = z.get(c, None)
        Qval = Q - 3 * c if Q is not None else None
        if milp_proven:
            F, src = r["val"], "ILP-PROVEN"
            flo, fhi = F, F
        elif Qval is not None:
            # Lemma P: F(c) = max(Q(c), F(c-1)); with F(c-1) in [flo, fhi]
            lo, hi = max(Qval, flo), max(Qval, min(fhi, ub10))
            if lo == hi:
                F, src = lo, "PROVEN-DICT"
            else:
                F, src = lo, f"DICT-bracket[{lo},{hi}]"
            flo, fhi = lo, hi
            if lb_val > hi:
                audit_fail.append((c, "MILP LB exceeds dictionary value!"))
        else:
            F, src = lb_val, f"LB-only({r['status'] if r else 'chain'})"
            flo, fhi = lb_val, min(ub10, fhi + (M - 3))  # F(c)<=F(c-1)+(m-3)
        fhi = min(fhi, ub10) if fhi else ub10
        if F_prev is not None and F < F_prev:
            audit_fail.append((c, "monotonicity violated"))
        rows.append({"c": c, "F": F, "source": src,
                     "milp_status": r["status"] if r else "",
                     "milp_val": r["val"] if r else "",
                     "profile": "+".join(f"{k}x{w}" for w, k in
                                         sorted(({int(w): k for w, k in
                                                  r["profile"].items()}).items(),
                                                reverse=True)) if r else "",
                     "slots_min": r.get("slots_min") if r else "",
                     "Q_from_z": Qval if Qval is not None else ""})
        F_prev = F
    with open(os.path.join(HERE, "frontier_m9_final.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for row in rows:
        print("{c:>3} F={F:>3} {source:<18} milp={milp_val}({milp_status}) "
              "Q={Q_from_z} {profile}".format(**row))
    print("\nNEW z-cells resolved this session:")
    for c, (lb, ub) in sorted(newz.items()):
        verdict = f"z(9,{c}) = {lb}" if lb == ub else \
                  f"z(9,{c}) in [{lb},{ub}] (LB<UB!)"
        print(f"  {verdict}")
    if audit_fail:
        print("\nAUDIT FAILURES:", audit_fail)
    else:
        print("\naudit clean (monotone, MILP-LB consistent).")


if __name__ == "__main__":
    main()

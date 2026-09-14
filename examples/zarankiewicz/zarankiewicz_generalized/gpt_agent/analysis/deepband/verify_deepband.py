"""Deep-band tasks 2-3 verification driver.

1. Assembles the frontier tables F_m(c) from frontier_m{m}*.jsonl.
2. Reconstructs z(m,n;3,3) = 3n + max_C [val - max(0, n - #C - (B - slots))]
   over the computed Pareto set (frontier points + pure-quad chain + S-table
   quad/pentad families) and compares against every known cell of rows 6-9.
3. Tests the candidate frontier laws:
     UBcols(c) = min((m-3)c, floor((2c + mR)/6), J*)      [PROVEN bounds]
     FLP(c)    = integer profile shell (budget + weighted budget + J*)
   and reports the realizability deficit delta(c) = FLP(c) - F(c).
4. Tests hypothesis (i) near-uniformity of frontier profiles.
Everything printed with honesty labels; exits nonzero on any reconstruction
mismatch against ground truth.
"""
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))


def load_truth():
    """Ground truth z-values for rows 6-9 with provenance labels."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ev", os.path.join(BASE, "..", "evaluator.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    truth = {}
    for (m, n), v in ev.KST_EXACT_VALUE.items():
        if 6 <= m <= 9:
            truth[(m, n)] = (v, "published(Tan)")
    for fn in ("ilp_gapband.jsonl", "ilp_results.jsonl"):
        p = os.path.join(BASE, "analysis", fn)
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if "z_ilp" in r and r.get("witness_valid") and 6 <= r["m"] <= 9:
                    truth.setdefault((r["m"], r["n"]),
                                     (r["z_ilp"], "ILP-proven(workspace)"))
    # Roman-window / Culik tails, PROVEN via Theorems 5, 6, 8 of theorems.md
    for m in (6, 7, 8, 9):
        B = 2 * comb(m, 3)
        T = {6: 9, 7: 15, 8: 28, 9: 40}[m]
        for n in range(m, B + 1):
            if (m, n) in truth:
                continue
            if n >= T:  # Theorem 8 domain (m Johnson-tight; 7 via S-table)
                v = 3 * n + min(T, (B - n) // 3)
                truth[(m, n)] = (v, "Theorem8")
    return truth


def load_frontier(m):
    """Merge frontier jsonl passes; prefer PROVEN records."""
    recs = {}
    for fn in sorted(os.listdir(HERE)):
        if "_w5" in fn or "_w6" in fn:
            continue    # weight-restricted runs are NOT the full frontier
        if fn.startswith(f"frontier_m{m}") and fn.endswith(".jsonl"):
            with open(fn if os.path.isabs(fn) else os.path.join(HERE, fn)) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("val") is None:
                        continue
                    c = r["c"]
                    keep = recs.get(c)
                    better = (keep is None
                              or (r["status"].startswith("PROVEN")
                                  and not keep["status"].startswith("PROVEN"))
                              or (r["val"] > keep["val"]))
                    if better:
                        recs[c] = r
    return recs


def s_table_configs(m):
    """Quad/pentad families from the proven S-tables (LB configs)."""
    S = {}
    p = os.path.join(BASE, "data", "supply_S8.csv") if m == 8 else \
        os.path.join(BASE, "analysis", f"supply_S{m}.csv")
    if os.path.exists(p):
        with open(p) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 3 and parts[1] == "0" and \
                        parts[2] not in ("None", "infeasible", "overbudget", ""):
                    S[int(parts[0])] = int(parts[2])
    # bounds/supply_table.csv (ILP-PROVEN rows only)
    p2 = os.path.join(BASE, "bounds", "supply_table.csv")
    if os.path.exists(p2):
        with open(p2) as f:
            for line in f.readlines()[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 5 and parts[0] == str(m) and parts[2] == "0" \
                        and parts[4] == "ILP-PROVEN" and parts[3]:
                    b = int(parts[1])
                    v = int(parts[3])
                    S[b] = max(S.get(b, -1), v)
    cfgs = []
    for b, a_max in S.items():
        for a in range(a_max + 1):
            cfgs.append((a + 2 * b, a + b, 4 * a + 10 * b))
    return cfgs


def pareto_configs(m, frontier):
    """Candidate (val, cols, slots) set for the exact-identity reconstruction."""
    T = {6: 9, 7: 15, 8: 28, 9: 40}[m]
    cfgs = set()
    for k in range(T + 1):               # pure-quad chain (sub-packings exist)
        cfgs.add((k, k, 4 * k))
    for c, r in frontier.items():
        cols = sum(v for v in r["profile"].values())
        cfgs.add((r["val"], cols, r["slots_min"]))
    for cfg in s_table_configs(m):
        cfgs.add(cfg)
    return sorted(cfgs)


def reconstruct(m, n, cfgs, B):
    best = 0
    for val, cols, slots in cfgs:
        if cols > n:
            continue
        pen = max(0, n - cols - (B - slots))
        best = max(best, val - pen)
    return 3 * n + best


def main():
    truth = load_truth()
    print("=" * 72)
    print("Z-RECONSTRUCTION FROM THE PARETO FRONTIER (exact identity, Thm 9)")
    print("=" * 72)
    n_ok = n_bad = 0
    fails = []
    predictions = []
    for m in (6, 7, 8, 9):
        B = 2 * comb(m, 3)
        fr = load_frontier(m)
        proven_c = {c for c, r in fr.items() if r["status"].startswith("PROVEN")}
        cfgs = pareto_configs(m, fr)
        cells = sorted(n for (mm, n) in truth if mm == m)
        row_ok = 0
        for n in cells:
            zt, src = truth[(m, n)]
            zp = reconstruct(m, n, cfgs, B)
            if zp == zt:
                row_ok += 1
                n_ok += 1
            else:
                n_bad += 1
                fails.append((m, n, zt, zp, src))
        print(f"m={m}: {row_ok}/{len(cells)} known cells reconstructed "
              f"(n = {cells[0]}..{cells[-1]})")
    if fails:
        print("\nMISMATCHES (investigate):")
        for m, n, zt, zp, src in fails:
            print(f"  z({m},{n}): truth {zt} [{src}]  reconstructed {zp}")
    # open row-9 cells: predictions
    print("\nOPEN CELLS (no proven truth) — frontier-derived values:")
    m = 9
    B = 2 * comb(m, 3)
    fr = load_frontier(m)
    cfgs = pareto_configs(m, fr)
    for n in (24, 25, 26, 27, 33):
        if (m, n) in truth:
            continue
        zp = reconstruct(m, n, cfgs, B)
        st = fr.get(n, {}).get("status", "missing")
        print(f"  z(9,{n}) = {zp}   [UB status at c={n}: {st}; "
              f"LB needs witness check]")
        predictions.append((n, zp))

    print()
    print("=" * 72)
    print("FRONTIER LAWS: F(c) vs proven bounds and the arithmetic shell")
    print("=" * 72)
    for m in (6, 7, 8, 9):
        R = (2 * comb(m - 1, 2)) // 3
        B = 2 * comb(m, 3)
        J = (m * R) // 4
        if m % 4 == 3 and m % 3 != 0:
            J -= 2
        fr = load_frontier(m)
        print(f"\nm={m}  (B={B}, mR={m*R}, J*={J})")
        print(f"{'c':>3} {'F':>3} {'UBcols':>6} {'FLP':>4} {'d':>2}  "
              f"{'slots':>5}  profile [status]")
        for c in sorted(fr):
            r = fr[c]
            ubc = min((m - 3) * c, (2 * c + m * R) // 6, J)
            flp = flp_shell(m, c, B, m * R, J)
            d = flp - r["val"]
            prof = ",".join(f"{w}^{k}" for w, k in sorted(r["profile"].items()))
            near_unif = is_near_uniform(r["profile"])
            print(f"{c:>3} {r['val']:>3} {ubc:>6} {flp:>4} {d:>2}  "
                  f"{r['slots_min']:>5}  {prof} [{r['status']}]"
                  f"{'' if near_unif else '  NOT-ADJACENT'}")
    sys.exit(1 if fails else 0)


def flp_shell(m, c, B, mR, J):
    """Arithmetic shell: max val over integer profiles satisfying
    columns<=c, slots<=B, weighted budget W<=mR, val<=J. No realizability."""
    best = 0
    # enumerate heavy-tail counts (w>=6); quad/pentad part in closed form
    from itertools import product
    heavy_ws = [w for w in range(6, m + 1)]
    maxk = {w: min(c, B // comb(w, 3), mR // (w * (w - 3))) for w in heavy_ws}
    for ks in product(*(range(maxk[w] + 1) for w in heavy_ws)):
        cols_h = sum(ks)
        if cols_h > c:
            continue
        slots_h = sum(k * comb(w, 3) for w, k in zip(heavy_ws, ks))
        W_h = sum(k * w * (w - 3) for w, k in zip(heavy_ws, ks))
        val_h = sum(k * (w - 3) for w, k in zip(heavy_ws, ks))
        if slots_h > B or W_h > mR:
            continue
        c2 = c - cols_h
        Wcap = min(mR - W_h, B - slots_h)   # quad/pentad: slots = W
        # max a + 2b, a + b <= c2, 4a + 10b <= Wcap  (exact integer sweep)
        val_qp = 0
        for bb in range(0, c2 + 1):
            if 10 * bb > Wcap:
                break
            a = max(0, min(c2 - bb, (Wcap - 10 * bb) // 4))
            val_qp = max(val_qp, a + 2 * bb)
        best = max(best, val_h + val_qp)
    return min(best, J)


def is_near_uniform(profile):
    ws = sorted(int(w) for w in profile)
    return len(ws) <= 1 or (len(ws) == 2 and ws[1] - ws[0] == 1)


if __name__ == "__main__":
    main()

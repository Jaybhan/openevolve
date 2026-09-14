"""THE LEVEL FORMULA test, refined (Task 3 of the level-theory program).

  z_hat(m,n) = max over levels w in [3, m], C(w-1,3)*n <= B(m), of
      (w-1)*n + min( n, D_w(m), floor((B - C(w-1,3)*n) / C(w-1,2)) )

(levels whose weight-(w-1) base already exceeds the slot budget are
infeasible and skipped; the min is then automatically >= 0).

Supply variants:
  V0  D_w = min(J_w, budget)                    [Entry-35 Johnson-capped pass]
  V1  D_w = exact (level_D2.csv + level 3/4 spectrum + wedge)
            else U_leave (leave_table.csv)  else V0 value
  V2  = V1 + bottom-layer supply cap: level w admissible only if
       n - (top-layer k) <= D_{w-1}(m); implemented as: k_min = n - D_{w-1},
       level infeasible if min-term < k_min.   [necessary condition:
       the bottom layer alone is a legal level-(w-1) packing]

Ground truth: data/exact_table.csv + ilp_results/ilp_gapband/ilp_frontier
jsonl EXACT cells + workspace determinations hard-coded from theorems.md.
Residual histogram + per-cell diagnosis against stored optimal witnesses
(analysis/witnesses/w_{m}x{n}.json): profile shape (levels used, adjacency),
classifying overshoot cells into MIXED-LEVEL vs SUPPLY-SLACK vs COEXISTENCE.
"""
import json
import os
import sys
from collections import Counter
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(HERE, "..")


# ---------------- supplies ----------------

def J_w(m, w):
    if w == 3:
        return 2 * comb(m, 3)
    lam2 = (2 * (m - 2)) // (w - 2)
    r3 = ((m - 1) * lam2) // (w - 1)
    r2 = (2 * comb(m - 1, 2)) // comb(w - 1, 2)
    return (m * min(r2, r3)) // w


def D4_spectrum(m):
    """Theorem 11 spectrum: J - 2 on the class {m=3 mod 4, 3 !| m}, else J.
    Exact for m<=18 (Tan) + 19,23,27 (workspace); formula for all m
    (UB proven all m via Thm F / Johnson; LB = GKLO for large m)."""
    J = (m * (((m - 1) * (m - 2)) // 3)) // 4
    if m % 4 == 3 and m % 3 != 0:
        return J - 2
    return J


def load_supplies():
    """exact: proven values; uleave: best upper bounds (leave-IP and
    decision-derived caps via supply_status.csv when present)."""
    exact, uleave = {}, {}
    for line in open(os.path.join(AN, "level_D2.csv")).readlines()[1:]:
        p = line.strip().split(",")
        if len(p) >= 4 and p[3] == "EXACT" and int(p[2]) >= 0:
            exact[(int(p[0]), int(p[1]))] = int(p[2])
    lt = os.path.join(HERE, "leave_table.csv")
    if os.path.exists(lt):
        for line in open(lt).readlines()[1:]:
            p = line.strip().split(",")
            if len(p) >= 7 and p[5] == "EXACT" and p[6]:
                uleave[(int(p[0]), int(p[1]))] = int(p[6])
    ss = os.path.join(HERE, "supply_status.csv")
    if os.path.exists(ss):
        for line in open(ss).readlines()[1:]:
            m, w, v, st = line.strip().split(",")[:4]
            m, w = int(m), int(w)
            if st == "EXACT":
                exact.setdefault((m, w), int(v))
            elif st == "BRACKET":
                ub = int(v.strip("[]").split(";")[1])
                uleave[(m, w)] = min(uleave.get((m, w), ub), ub)
    return exact, uleave


EXACT_D, ULEAVE_D = load_supplies()


def D_w(m, w, variant):
    """Supply at level w; returns (value, source_tag)."""
    if w > m:
        return 0, "none"
    if w == 3:
        return 2 * comb(m, 3), "exact"
    if w == 4:
        return D4_spectrum(m), "spectrum4"
    if 3 * (m - w) <= m - 3:
        return 2, "wedge"
    v0 = min(J_w(m, w), (2 * comb(m, 3)) // comb(w, 3))
    if variant == "V0":
        return v0, "johnson"
    if (m, w) in EXACT_D:
        return EXACT_D[(m, w)], "exact"
    if (m, w) in ULEAVE_D:
        return ULEAVE_D[(m, w)], "uleave"
    return v0, "johnson"


# ---------------- the formula ----------------

def z_hat(m, n, variant="V1", detail=False):
    B = 2 * comb(m, 3)
    best, info = 0, None
    for w in range(3, m + 1):
        base_slots = comb(w - 1, 3) * n
        if base_slots > B:
            break
        Dw, src = D_w(m, w, variant)
        k = min(n, Dw, (B - base_slots) // comb(w - 1, 2))
        if variant in ("V2", "V2J") and w >= 4:
            Dbot, _ = D_w(m, w - 1, variant)
            if n - k > Dbot:
                continue  # bottom layer alone illegal -> level inadmissible
        if variant == "V2J" and w >= 5:
            # Lemma C / Theorem F mixed-value law: Q = (w-4)(n-k)+(w-3)k <= T33(m)
            k = min(k, D4_spectrum(m) - (w - 4) * n)
            if k < 0:
                continue
        val = (w - 1) * n + k
        if val > best:
            best, info = val, (w, k, Dw, src)
    return (best, info) if detail else best


# ---------------- ground truth ----------------

def load_truth():
    Z = {}
    src = {}
    for line in open(os.path.join(AN, "..", "data", "exact_table.csv")).readlines()[1:]:
        m, n, z = (int(t) for t in line.strip().split(","))
        Z[(m, n)] = z
        src[(m, n)] = "table"
    for fn, key in (("ilp_results.jsonl", "z_ilp"), ("ilp_gapband.jsonl", "z_ilp"),
                    ("ilp_frontier.jsonl", "z_ilp")):
        for line in open(os.path.join(AN, fn)):
            d = json.loads(line)
            if key in d and d.get("witness_valid"):
                mn = (d["m"], d["n"])
                if mn in Z:
                    assert Z[mn] == d[key], f"conflict at {mn}"
                Z[mn] = d[key]
                src.setdefault(mn, "ilp")
    # workspace determinations (theorems.md): row 11 SAT cells n=82..89
    for n in range(82, 90):
        Z[(11, n)] = 3 * n + min(80, (330 - n) // 3)
        src[(11, n)] = "sat11"
    Z[(10, 15)] = 81
    src[(10, 15)] = "sat"
    return Z, src


# ---------------- witness diagnosis ----------------

def witness_profile(m, n):
    p = os.path.join(AN, "witnesses", f"w_{m}x{n}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return sorted(Counter(len(b) for b in d["blocks"]).items())


def diagnose(m, n, zt, variant):
    zh, info = z_hat(m, n, variant, detail=True)
    r = zh - zt
    prof = witness_profile(m, n)
    tag = ""
    if r > 0:
        w, k, Dw, src = info
        if prof:
            heavy = [wt for wt, _ in prof if wt >= 4]
            span = (max(heavy) - min(heavy)) if heavy else 0
            tag = ("MIXED-LEVEL" if span >= 2 else "TWO-LAYER-TRUE")
        else:
            tag = "no-witness"
        tag += f"; formula level {w} (k={k}, D={Dw}[{src}])"
        if prof:
            tag += f"; true profile {prof}"
    elif r < 0:
        tag = f"UNDERSHOOT; true profile {prof}"
    return r, tag


# ---------------- V3: the mixed-ledger correction (Theorem 7 generalized) ----------------
# S_m[(k5,k6)] = max # quads coexisting with k5 pentads + k6 hexads.
# The corrected two-layer formula IS Theorem 7's ledger form; the correction
# term relative to V1 is exactly the ledger deficit
#   corr(m; k5,k6) = [k4^budget(k5,k6) - S_m(k5,k6)]  (0 when the ledger is
# budget-tight).  Tables: S6, S7 (workspace, complete for k6=0), S8 (complete
# incl. k6 axis), S9 (slices + z-derived brackets; both endpoints tested).

S6 = {(0, 0): 9, (1, 0): 6, (2, 0): 4}
S7 = {(0, 0): 15, (1, 0): 12, (2, 0): 10, (3, 0): 8, (4, 0): 6}
NU = {6: 6, 7: 10, 8: 10, 9: 28}  # validity thresholds (weights<=6 / ledger-covered)


def load_S(fn):
    S = {}
    for line in open(os.path.join(AN, fn)).readlines()[1:]:
        k5, k6, s = line.strip().split(",")
        if s not in ("None", "infeasible", "overbudget"):
            S[(int(k5), int(k6))] = int(s)
    return S


def z_ledger(m, n, S):
    B = 2 * comb(m, 3)
    best = 0
    for (k5, k6), k4max in S.items():
        for k4 in range(k4max + 1):
            cols = k4 + k5 + k6
            if cols > n:
                continue
            slots = 4 * k4 + 10 * k5 + 20 * k6
            if slots > B:
                continue
            k3 = min(n - cols, B - slots)
            best = max(best, 2 * n + 4 * k6 + 3 * k5 + 2 * k4 + k3)
    return best


def run_v3(Z):
    S8 = load_S("supply_S8.csv")
    S9a = load_S("supply_S9.csv")
    # z-derived brackets for the unresolved S9 slices: (2,0) and (5,0) from
    # theorems.md; (9..11, 0) from MONOTONICITY S9(12)=10 <= S9(k) <= S9(8)=20
    # tightened by the z-derived S9(9) <= 18 (theorems.md).
    S9lo = dict(S9a); S9hi = dict(S9a)
    for (k, lo, hi) in ((2, 33, 35), (5, 25, 28), (9, 10, 18), (10, 10, 18), (11, 10, 18)):
        S9lo[(k, 0)], S9hi[(k, 0)] = lo, hi
    print("\n== V3 (mixed-ledger formula = Theorem 7 form) ==")
    for m, S, tag in ((6, S6, ""), (7, S7, ""), (8, S8, ""),
                      (9, S9lo, " [S9 lo-brackets]"), (9, S9hi, " [S9 hi-brackets]")):
        cells = sorted((mm, n) for (mm, n) in Z if mm == m and n >= NU[m])
        bad = []
        for (mm, n) in cells:
            zf = z_ledger(mm, n, S)
            if zf != Z[(mm, n)]:
                bad.append((n, zf, Z[(mm, n)]))
        print(f"  row {m}{tag}: {len(cells)} cells n>={NU[m]}: "
              f"{'ALL MATCH' if not bad else f'MISMATCH {bad}'}")


def main():
    Z, src = load_truth()
    print(f"ground truth cells: {len(Z)}")
    for variant in ("V0", "V1", "V2", "V2J"):
        hist = Counter()
        overs, unders = [], []
        for (m, n), zt in sorted(Z.items()):
            zh = z_hat(m, n, variant)
            hist[zh - zt] += 1
            if zh > zt:
                overs.append((m, n, zh - zt))
            elif zh < zt:
                unders.append((m, n, zh - zt))
        print(f"\n== {variant}: residual histogram (z_hat - z_true) ==")
        for r in sorted(hist):
            print(f"  {r:+3d}: {hist[r]:3d} cells")
        print(f"  overshoot cells: {overs}")
        print(f"  undershoot cells: {unders}")
        if variant != "V0":
            print("  -- diagnosis of nonzero residuals --")
            for (m, n, _) in overs + [(a, b, c) for a, b, c in unders]:
                r, tag = diagnose(m, n, Z[(m, n)], variant)
                print(f"  ({m},{n}): {r:+d}  {tag}")

    run_v3(Z)

    # closed-form row checks (rows with proven all-n formulas)
    print("\n== closed-form wide-region reproduction (V1) ==")
    def T33(m):
        return D4_spectrum(m)
    for m in (6, 7, 8, 9, 11, 12, 19, 23):
        B = 2 * comb(m, 3)
        T = T33(m)
        bad = 0
        lo = T
        for n in range(lo, min(B + 40, lo + 400)):
            want = 3 * n + min(T, (B - n) // 3) if n <= B else 2 * n + B
            if n > B:
                want = 2 * n + B
            got = z_hat(m, n, "V1")
            if got != want:
                bad += 1
                if bad <= 3:
                    print(f"  m={m} n={n}: formula {got} vs Thm8-form {want}")
        print(f"  row {m}, n in [{lo},{min(B+40, lo+400)}): "
              f"{'ALL MATCH Thm8 closed form' if bad == 0 else f'{bad} deviations'}")


if __name__ == "__main__":
    sys.exit(main())

"""THE FULL-TABLE LEDGER PASS: the mixed-ledger Level Formula over EVERY
known z(m,n;3,3) cell, with per-cell status.

Engine: exact enumeration over level-count configs (k_w)_{w=5..13} with
slot pruning; k4 capped by min(per-level supply cap, slot/col/Q-caps,
SLICE CAPS S_m(sig) [monotone: a slice at sig caps every config whose
heavy signature dominates sig]); k3 = min(n - cols, B - slots) fill;
ladder cuts (Theorem 12, j = 5..10) filtered at the leaf.

Slice-cap store: rows 6-9 from the workspace S-tables (S9 holes at both
bracket endpoints -> two passes; equality = BRACKET-INSENSITIVE), plus
slices computed here (slices.csv, incremental).

Statuses per cell:
  MATCH             z_hi == z_true (UB side closes; LB = stored witness)
  RESIDUAL(+d)      z_hi = z_true + d; claiming heavy signatures listed
  (row-9 sensitivity noted separately)

Outputs: fullpass_status.csv, claiming signatures -> slice_queue.csv.
"""
import json
import os
import sys
from collections import Counter
from math import comb, gcd

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from test_formula import load_truth, D4_spectrum, J_w  # noqa: E402

WMAX = 13
MU = {j: (j - 1) * (4 - j) // 2 for j in range(4, 11)}


def supply_caps(m):
    """cap[w] = best proven UB on D2(m,w,3), w = 4..min(m,WMAX)."""
    cap = {4: D4_spectrum(m)}
    ss = {}
    p = os.path.join(HERE, "supply_status.csv")
    if os.path.exists(p):
        for line in open(p).readlines()[1:]:
            mm, ww, v, st = line.strip().split(",")[:4]
            if int(mm) == m:
                ss[int(ww)] = (int(v) if st == "EXACT"
                               else int(v.strip("[]").split(";")[1]))
    for w in range(5, min(m, WMAX) + 1):
        if 3 * (m - w) <= m - 3:
            cap[w] = 2
        elif w in ss:
            cap[w] = ss[w]
        else:
            cap[w] = min(J_w(m, w), (2 * comb(m, 3)) // comb(w, 3))
    return cap


def load_slices(hi=True):
    """store[m] = list of (sig(k5..k13 tuple), Scap). Rows 6-9 tables +
    slices.csv."""
    store = {}

    def add(m, k5, k6, S):
        sig = tuple([k5, k6] + [0] * (WMAX - 6))
        store.setdefault(m, []).append((sig, S))

    S6 = {(0, 0): 9, (1, 0): 6, (2, 0): 4}
    S7 = {(0, 0): 15, (1, 0): 12, (2, 0): 10, (3, 0): 8, (4, 0): 6}
    for (k5, k6), S in S6.items():
        add(6, k5, k6, S)
    for (k5, k6), S in S7.items():
        add(7, k5, k6, S)
    for fn, m in (("supply_S8.csv", 8), ("supply_S9.csv", 9)):
        for line in open(os.path.join(AN, fn)).readlines()[1:]:
            k5, k6, s = line.strip().split(",")
            if s in ("infeasible", "overbudget"):
                add(m, int(k5), int(k6), -1)  # cap -1 = config impossible
            elif s != "None":
                add(m, int(k5), int(k6), int(s))
    # S9 holes: z-derived/monotone brackets (theorems.md + monotonicity)
    holes = {(2, 0): (33, 35), (5, 0): (25, 28), (9, 0): (10, 18),
             (10, 0): (10, 18), (11, 0): (10, 18)}
    for (k5, k6), (lo, hicap) in holes.items():
        add(9, k5, k6, hicap if hi else lo)
    sp = os.path.join(HERE, "slices.csv")
    if os.path.exists(sp):
        for line in open(sp).readlines()[1:]:
            p = line.strip().split(",")
            m = int(p[0])
            sig = tuple(int(t) for t in p[1].split(";"))
            S = int(p[2]) if p[2] != "TIMEOUT" else None
            if S is not None:
                store.setdefault(m, []).append((sig, S))
    return store


def cell_pass(m, n, ztrue, cap, slices, collect=False):
    """Return (z_hi, claiming) for cell (m,n). claiming = set of minimal
    (sig, k4min) with formula value > ztrue."""
    B = 2 * comb(m, 3)
    Qcap = cap[4]  # Lemma C / Thm F mixed-value law: Q = sum (w-3)k_w <= T33
    ws = [w for w in range(5, min(m, WMAX) + 1)]
    slot = {w: comb(w, 3) for w in ws}
    best = [2 * n + min(n, B)]  # level-3-only baseline
    claiming = set()
    sig0 = [0] * (WMAX - 4)

    def leaf(sig, s, c, val, Q):
        # k4 bounds
        k4cap = min(cap[4], (B - s) // 4, n - c, Qcap - Q)
        for (ssig, S) in slices.get(m, []):
            if all(a >= b for a, b in zip(sig, ssig)):
                k4cap = min(k4cap, S)
        if k4cap < 0:
            return
        # E(k4) = 2n + val + 2k4 + min(n-c-k4, B-s-4k4): unimodal —
        # increasing on the column piece, decreasing on the slot piece;
        # peak at 3k4 ~ (B-s)-(n-c). Evaluate peak+neighbors+endpoints.
        kstar = ((B - s) - (n - c)) // 3
        cands = {0, k4cap, min(max(kstar, 0), k4cap),
                 min(max(kstar + 1, 0), k4cap), min(max(kstar - 1, 0), k4cap)}
        for k4 in (range(k4cap + 1) if collect else sorted(cands)):
            s2, c2 = s + 4 * k4, c + k4
            k3 = max(0, min(n - c2, B - s2))
            E = 2 * n + val + 2 * k4 + k3
            # ladder cuts j=5..10 (Thm 12): (j-1)W + mu_j*Eblk <= 3B
            W = 4 * k4 + sum(w * (w - 3) * k for w, k in zip(ws, sig))
            Eblk = 3 * k3 + 4 * k4 + sum(w * k for w, k in zip(ws, sig))
            ok = all((j - 1) * W + MU[j] * Eblk <= 3 * B for j in range(5, 11))
            if not ok:
                continue
            # MIXED-LEVEL LEMMA E (gcd congruence cut), applied to heavy
            # SUB-packings (sub-multiset legality): for a weight-subset W'
            # of the config, the W'-blocks alone form a legal packing whose
            # leave B - slots(W') obeys: every point-leave == 2C(m-1,2)
            # (mod g1), g1 = gcd{C(w-1,2): w in W'}; every pair-leave
            # == 2(m-2) (mod g2 = gcd{w-2}). Forced minimum => slot cap.
            heavy = [(w, k) for w, k in zip(ws, sig) if k]
            if k4:
                heavy = [(4, k4)] + heavy
            subsets = [heavy, [(w, k) for w, k in heavy if w % 3 != 0]]
            dead = False
            for sub in subsets:
                if not sub:
                    continue
                g1 = 0
                g2 = 0
                ssub = 0
                for w, k in sub:
                    g1 = gcd(g1, comb(w - 1, 2))
                    g2 = gcd(g2, w - 2)
                    ssub += comb(w, 3) * k
                c1m = (2 * comb(m - 1, 2)) % g1 if g1 > 1 else 0
                c2m = (2 * (m - 2)) % g2 if g2 > 1 else 0
                pmin = 0
                if c2m > 0:
                    t_ = -(-((m - 1) * c2m) // 2)
                    pmin = (c1m if c1m >= t_ else
                            c1m + g1 * (-(-(t_ - c1m) // g1))) if g1 > 1 else t_
                elif c1m > 0:
                    pmin = c1m
                Lmin = -(-(m * pmin) // 3) if pmin else 0
                if c2m > 0:
                    Lmin = max(Lmin, -(-(comb(m, 2) * c2m) // 3))
                if B - ssub < Lmin:
                    dead = True
                    break
                # gapped-leave branch (L3(iii) at the gcd level): residues
                # vanish but any NONEMPTY leave has >= 3 touched points each
                # with ell_x >= g1 (resp. >= 3 pairs with >= g2): the leave
                # of the W'-sub-packing lies in {0} u [max(g1,g2), inf).
                if pmin == 0 and max(g1, g2) > 1:
                    Lgap = max(g1 if g1 > 1 else 0, g2 if g2 > 1 else 0)
                    if 0 < B - ssub < Lgap:
                        dead = True
                        break
            if dead:
                continue
            if E > best[0]:
                best[0] = E
            if collect and E > ztrue:
                claiming.add((tuple(sig), k4))
                break  # minimal k4 for this sig recorded

    def rec(i, sig, s, c, val, Q):
        if i == len(ws):
            leaf(sig, s, c, val, Q)
            return
        w = ws[i]
        kmax = min(cap[w], (B - s) // slot[w], n - c)
        for k in range(kmax + 1):
            # slice caps can declare configs impossible (S = -1)
            bad = False
            if k:
                sig[i] = k
                for (ssig, S) in slices.get(m, []):
                    if S == -1 and all(a >= b for a, b in zip(sig, ssig)):
                        bad = True
                        break
            if not bad:
                rec(i + 1, sig, s + k * slot[w], c + k,
                    val + (w - 2) * k, Q + (w - 3) * k)
            if k:
                sig[i] = 0
            if bad:
                break
    rec(0, sig0, 0, 0, 0, 0)
    return best[0], claiming


def main():
    Z, _ = load_truth()
    out = open(os.path.join(HERE, "fullpass_status.csv"), "w")
    out.write("m,n,z_true,z_hi,z_lo9,status,claiming\n")
    match = resid = 0
    queue = Counter()
    Shi, Slo = load_slices(True), load_slices(False)
    for (m, n), zt in sorted(Z.items()):
        cap = supply_caps(m)
        zhi, _ = cell_pass(m, n, zt, cap, Shi)
        zlo = ""
        if m == 9:
            zlo, _ = cell_pass(m, n, zt, cap, Slo)
        st = "MATCH" if zhi == zt and (zlo == "" or zlo == zt) else \
             f"RESIDUAL+{zhi - zt}" if zhi > zt else f"UNDER{zhi - zt}"
        cl = ""
        if zhi > zt:
            resid += 1
            _, claiming = cell_pass(m, n, zt, cap, Shi, collect=True)
            # keep pareto-minimal signatures
            sigs = {}
            for (sig, k4) in claiming:
                sigs[sig] = min(sigs.get(sig, 10**9), k4)
            minimal = []
            for sig in sorted(sigs):
                if not any(all(a >= b for a, b in zip(sig, o)) and sig != o
                           for o in sigs):
                    minimal.append((sig, sigs[sig]))
            cl = "|".join(";".join(map(str, s)) + f">={k}" for s, k in minimal[:6])
            for s, k in minimal:
                queue[(m, s)] = max(queue[(m, s)], k)
        else:
            match += 1
        out.write(f"{m},{n},{zt},{zhi},{zlo},{st},{cl}\n")
    out.close()
    print(f"MATCH {match} / {match + resid}; residual cells {resid}")
    with open(os.path.join(HERE, "slice_queue.csv"), "w") as f:
        f.write("m,sig,k4_needed\n")
        for (m, sig), k in sorted(queue.items()):
            f.write(f"{m},{';'.join(map(str, sig))},{k}\n")
    print(f"distinct claiming slices: {len(queue)} -> slice_queue.csv")


if __name__ == "__main__":
    main()

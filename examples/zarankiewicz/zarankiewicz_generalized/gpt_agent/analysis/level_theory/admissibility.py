"""Level-w spectrum arithmetic: divisibility, Johnson bounds, congruence-leave
lower bounds (CHECK-0), wedge theorem, residue classification.

Definitions (level w on m >= w points, 2-fold triple packing by weight-w blocks,
block multiplicity <= 2):
  B(m)      = 2*C(m,3)          total triple slots
  slots     = C(w,3)            slots consumed per block
  perfect   = B/slots blocks    (needs the three divisibility conditions)
  c1(m,w)   = 2*C(m-1,2) mod C(w-1,2)   forced point-leave residue
  c2(m,w)   = 2*(m-2)   mod (w-2)       forced pair-leave residue
Every packing's leave satisfies  ell_x == c1 (mod C(w-1,2)) at EVERY point and
ell_xy == c2 (mod w-2) at EVERY pair (each block through x covers C(w-1,2)
pair-slots at x; each block through xy covers w-2 slots at xy).

CHECK-0 congruence upper bound U_c0: the largest b such that
L = B - slots*b admits point-leave values (numeric necessary conditions only):
  pmin = smallest v >= 0 with v == c1 (mod C(w-1,2)), v >= ceil((m-1)*c2/2),
         and v > 0 required when c2 > 0 (every pair positive => every point);
  if pmin > 0: need 3L >= m*pmin.
  if pmin == 0 and L > 0: touched points need ell_x >= C(w-1,2) and a nonempty
         leave touches >= 3 points => need L >= C(w-1,2)   [Lemma-E mechanism].

Wedge theorem (PROVEN, see spectrum.md): D2(m,w,3) = 2 iff 3(m-w) <= m-3,
i.e. m <= (3w-3)/2  (any 3 blocks' complements span <= 3(m-w) <= m-3 points,
so some triple lies in all 3).

Johnson bounds: lam2 = floor(2(m-2)/(w-2)); r3 = floor((m-1)*lam2/(w-1))
(iterated); r2 = floor(2C(m-1,2)/C(w-1,2)) (one-level); J = floor(m*min/w).

Outputs: spectrum_table.csv + residue classification print + verification of
every exact value in ../level_D2.csv against all bounds.
"""
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))


def base(m, w):
    B = 2 * comb(m, 3)
    slots = comb(w, 3)
    c1 = (2 * comb(m - 1, 2)) % comb(w - 1, 2)
    c2 = (2 * (m - 2)) % (w - 2)
    return B, slots, c1, c2


def johnson(m, w):
    lam2 = (2 * (m - 2)) // (w - 2)
    r3 = ((m - 1) * lam2) // (w - 1)
    r2 = (2 * comb(m - 1, 2)) // comb(w - 1, 2)
    r = min(r2, r3)
    return (m * r) // w, r, lam2


def pmin_value(m, w):
    """Smallest admissible positive-context point-leave value (0 if none forced)."""
    B, slots, c1, c2 = base(m, w)
    mod = comb(w - 1, 2)
    if c2 > 0:
        t = -(-((m - 1) * c2) // 2)  # ceil
        v = c1 if c1 >= t else c1 + mod * (-(-(t - c1) // mod))
        if v == 0:  # c1==0, t>0 handled above; safeguard
            v = mod
        return v
    return c1  # c2 == 0: only the point residue forces anything


def check0_ub(m, w):
    """Largest b with L = B - slots*b passing the numeric congruence tests."""
    B, slots, c1, c2 = base(m, w)
    pmod = comb(w - 1, 2)
    pmin = pmin_value(m, w)
    for b in range(B // slots, -1, -1):
        L = B - slots * b
        ok = True
        if pmin > 0 and 3 * L < m * pmin:
            ok = False
        if ok and pmin == 0 and L > 0 and L < pmod:
            ok = False  # support argument: 3L = sum ell_x >= 3*pmod
        if ok and c2 > 0 and 3 * L < comb(m, 2) * c2:
            ok = False
        if ok:
            return b
    return 0


def wedge(m, w):
    return 3 * (m - w) <= m - 3


def perfect_admissible(m, w):
    B, slots, c1, c2 = base(m, w)
    return B % slots == 0 and c1 == 0 and c2 == 0


def load_exact():
    exact = {}
    path = os.path.join(HERE, "..", "level_D2.csv")
    if os.path.exists(path):
        for line in open(path).readlines()[1:]:
            p = line.strip().split(",")
            if len(p) >= 4 and p[3] == "EXACT" and int(p[2]) >= 0:
                exact[(int(p[0]), int(p[1]))] = int(p[2])
    return exact


def main():
    exact = load_exact()
    rows = []
    for w in range(4, 12):
        for m in range(w, 41):
            B, slots, c1, c2 = base(m, w)
            J, r, lam2 = johnson(m, w)
            bud = B // slots
            u0 = check0_ub(m, w)
            ub = min(J, bud, u0)
            wd = wedge(m, w)
            if wd:
                ub = 2
            perf = perfect_admissible(m, w)
            ex = exact.get((m, w))
            cls = ("WEDGE" if wd else
                   "PERFECT" if perf and (ex is None or ex == B // slots) else
                   "?")
            rows.append((m, w, B, slots, c1, c2, bud, J, u0, ub,
                         "" if ex is None else ex, cls, int(perf), int(wd)))
    with open(os.path.join(HERE, "spectrum_table.csv"), "w") as f:
        f.write("m,w,B,slots,c1,c2,budget,J,U_check0,UB,exact,class,perfect,wedge\n")
        for r_ in rows:
            f.write(",".join(str(x) for x in r_) + "\n")

    # ---- verification against every exact value ----
    print("== verification vs level_D2.csv ==")
    bad = []
    for (m, w), v in sorted(exact.items()):
        B, slots, c1, c2 = base(m, w)
        J, _, _ = johnson(m, w)
        bud = B // slots
        u0 = check0_ub(m, w)
        wd = wedge(m, w)
        ub = 2 if wd else min(J, bud, u0)
        tag = []
        if v > ub:
            bad.append((m, w, v, ub))
            tag.append("**VIOLATION**")
        if wd:
            tag.append("WEDGE" + ("=2 OK" if v == 2 else " MISMATCH!"))
        if perfect_admissible(m, w):
            tag.append("PERFECT-ADM" + (" attained" if v == B // slots else
                                        f" NOT attained ({v} < {B//slots})"))
        if v == ub and not wd:
            tag.append(f"UB-tight (J={J}, bud={bud}, c0={u0})")
        elif not wd:
            tag.append(f"defect {ub - v} below UB (J={J}, bud={bud}, c0={u0})")
        print(f"D2({m},{w}) = {v}: {'; '.join(tag)}")
    print("violations:", bad if bad else "NONE")

    # ---- residue classification per w ----
    print("\n== residue classification (per w): (c1, c2, B mod slots, perfect) ==")
    for w in range(5, 9):
        slots = comb(w, 3)
        pmod = comb(w - 1, 2)
        # detect period
        def sig(m):
            B, _, c1, c2 = base(m, w)
            return (c1, c2, B % slots)
        period = None
        for p in range(1, 3 * slots * pmod + 1):
            if all(sig(m) == sig(m + p) for m in range(w, w + 3 * p + 60)):
                period = p
                break
        perf_res = sorted(set(m % period for m in range(w, w + 20 * period)
                              if perfect_admissible(m, w)))
        print(f"w={w}: slots={slots}, point mod {pmod}, pair mod {w-2}; "
              f"signature period {period}")
        print(f"   perfect-admissible residues mod {period}: {perf_res}")
        # per residue: c1, c2 pattern (these depend on m mod smaller periods)
        for r_ in range(period):
            ms = [m for m in range(max(w, 5), w + 4 * period) if m % period == r_]
            if not ms:
                continue
            m0 = ms[0]
            B, _, c1, c2 = base(m0, w)
            print(f"   m == {r_:3d} (mod {period}): c1={c1:2d}, c2={c2}, "
                  f"B mod {slots} = {B % slots:2d}"
                  + ("  [PERFECT-ADMISSIBLE]" if r_ in perf_res else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

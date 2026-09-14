#!/usr/bin/env python3
"""Independent adversarial verification of Theorem F (J-2 law on the class
m = 3 (mod 4), m != 0 (mod 3)).  Written from scratch by the design_prover
agent -- shares no code with the coordinator's script.

Checks:
  A. Class arithmetic: 3 | C(m-1,2), B = mR = 2C(m,3), B = 2 (mod 4),
     J = (B-2)/4, for all class members m <= 2000.  J(19)=484, J(23)=885.
  B. Slot identity / excess table: C(w,3) = 4(w-3) + e_w with
     e_4,e_5,e_6,e_7 = 0,2,8,19; e_w increasing for w <= 40.
  C. The two congruence IDENTITIES on random legal mixed configs
     (quads + pentads), m in {7,11,19}:
         l_x  = 2C(m-1,2) - 3 q_x - 6 p_x        (=> l_x = 0 mod 3 on class)
         l_xy = 2(m-2)    - 2 q_xy - 3 p_xy      (=> l_xy = p_xy mod 2)
  D. FINITE CLASSIFICATION (the load-bearing enumeration), my own code:
     all multisets of triples with multiplicity <= 2 on a 6-point ground set
     (covers all supports: any weight-6 leave with point-degrees in {3,6}
     touches <= 18/3 = 6 points), total weight w:
       w = 2 : 0 survivors of point-mod-3            (Lemma E core)
       w = 6 : 0 survivors of point-mod-3 AND pair-parity (Theorem F core);
               also count survivors of point-mod-3 alone (must be > 0 --
               the coordinator's parallel-class example must appear).
       w = 4 : survivors of BOTH must be exactly K4^(3) up to iso
               (k5=1 case of Theorem F).
  E. Coordinator's weight-6 example {123},{456},{135},{246},{146},{235}:
     passes point-mod-3, FAILS pair-parity at pair {1,2}.
  F. k5 = 3 case: exhaustively over 3 pentads on <= 15 points (P1 fixed,
     P2 up to relabeling inside/outside), verify no triple of pentads
     (multiset, multiplicity <= 2) has every pair-multiplicity even.
"""
import itertools, random, sys
from collections import Counter
from math import comb

ok = True
def report(name, passed, extra=""):
    global ok
    ok &= passed
    print(("PASS " if passed else "FAIL ") + name + (" -- " + extra if extra else ""))

# ---------- A. class arithmetic ----------
class_members = [m for m in range(7, 2001) if m % 4 == 3 and m % 3 != 0]
a_ok = True
for m in class_members:
    if ((m-1)*(m-2)) % 3 != 0: a_ok = False
    if comb(m-1,2) % 3 != 0: a_ok = False
    R = (m-1)*(m-2)//3
    B = m*R
    if B != 2*comb(m,3): a_ok = False
    if B % 4 != 2: a_ok = False
    J = (B-2)//4
    if J != (m*R)//4: a_ok = False
report("A: class arithmetic (all %d members m<=2000)" % len(class_members), a_ok)
for m in (7,11,19,23,31,35):
    R=(m-1)*(m-2)//3; B=m*R; J=(B-2)//4
    print("   m=%2d  R=%4d  B=%6d  J=%4d  J-2=%4d" % (m,R,B,J,J-2))

# ---------- B. slot identity ----------
e = {w: comb(w,3) - 4*(w-3) for w in range(4,41)}
b_ok = (e[4],e[5],e[6],e[7]) == (0,2,8,19) and all(e[w+1]>e[w] for w in range(5,40))
report("B: excess table e_w = C(w,3)-4(w-3)", b_ok, "e4..e7 = %d,%d,%d,%d" % (e[4],e[5],e[6],e[7]))

# ---------- C. congruence identities on random mixed configs ----------
def random_mixed_config(m, rng, pentad_bias=0.15):
    """Greedy random legal multiset of quads+pentads, triple coverage <= 2."""
    cov = Counter()
    blocks = []
    pool = [frozenset(c) for c in itertools.combinations(range(m),4)]
    pool = pool*2
    pent = [frozenset(c) for c in itertools.combinations(range(m),5)]
    rng.shuffle(pool); rng.shuffle(pent)
    for P in pent[:max(3,m)]:
        if rng.random() < pentad_bias:
            ts = list(itertools.combinations(sorted(P),3))
            if all(cov[t] < 2 for t in ts):
                for t in ts: cov[t]+=1
                blocks.append(P)
    for Q in pool:
        ts = list(itertools.combinations(sorted(Q),3))
        if all(cov[t] < 2 for t in ts):
            for t in ts: cov[t]+=1
            blocks.append(Q)
    return blocks, cov

c_ok = True
rng = random.Random(20260728)
for m in (7,11,19):
    for trial in range(3):
        blocks, cov = random_mixed_config(m, rng)
        # leave
        l = {t: 2-cov[t] for t in itertools.combinations(range(m),3)}
        for x in range(m):
            lx = sum(v for t,v in l.items() if x in t)
            qx = sum(1 for Bk in blocks if len(Bk)==4 and x in Bk)
            px = sum(1 for Bk in blocks if len(Bk)==5 and x in Bk)
            if lx != 2*comb(m-1,2) - 3*qx - 6*px: c_ok=False
            if m%4==3 and m%3!=0 and lx%3!=0: c_ok=False
        for x,y in itertools.combinations(range(m),2):
            lxy = sum(v for t,v in l.items() if x in t and y in t)
            qxy = sum(1 for Bk in blocks if len(Bk)==4 and x in Bk and y in Bk)
            pxy = sum(1 for Bk in blocks if len(Bk)==5 and x in Bk and y in Bk)
            if lxy != 2*(m-2) - 2*qxy - 3*pxy: c_ok=False
            if (lxy - pxy)%2 != 0: c_ok=False
report("C: point/pair leave identities on random mixed configs (m=7,11,19; 3 trials; pentads included)", c_ok)

# ---------- D. finite classification on 6 points ----------
TRIPLES6 = list(itertools.combinations(range(6),3))   # 20 triples
def degrees(mult):
    deg = [0]*6
    pdeg = Counter()
    for ti, k in mult.items():
        t = TRIPLES6[ti]
        for x in t: deg[x]+=k
        for pr in itertools.combinations(t,2): pdeg[pr]+=k
    return deg, pdeg

def classify(weight):
    surv_pt, surv_both = [], []
    for combo in itertools.combinations_with_replacement(range(20), weight):
        mult = Counter(combo)
        if any(v>2 for v in mult.values()): continue
        deg, pdeg = degrees(mult)
        if any(d%3 for d in deg): continue
        surv_pt.append(mult)
        if all(v%2==0 for v in pdeg.values()):
            surv_both.append(mult)
    return surv_pt, surv_both

pt2, both2 = classify(2)
report("D: weight-2 leaves: 0 survivors of point-mod-3 alone", len(pt2)==0,
       "point-only survivors: %d" % len(pt2))
pt6, both6 = classify(6)
report("D: weight-6 leaves: 0 survivors of point-mod-3 + pair-parity", len(both6)==0,
       "point-only survivors: %d (must be >0), both: %d" % (len(pt6), len(both6)))
pt4, both4 = classify(4)
# k5=1 case needs the STRONGER fact: POINT congruence ALONE forces K4^(3)
# (pair parity there is l_xy = p_xy mod 2, not l_xy = 0, so it cannot be
# assumed; K4's all-even pair-leaves must be a consequence).
k4_count = 0
for mult in pt4:
    ts = sorted(TRIPLES6[i] for i in mult.elements())
    pts = sorted(set().union(*[set(t) for t in ts]))
    if len(pts)==4 and ts==sorted(itertools.combinations(pts,3)):
        k4_count += 1
report("D: weight-4 leaves surviving POINT congruence alone = exactly the K4^(3)'s",
       len(pt4)==15 and k4_count==15 and len(both4)==15,
       "point-only survivors: %d, K4-shaped: %d, both-congr: %d, labeled K4s on 6 pts: 15"
       % (len(pt4), k4_count, len(both4)))

# ---------- E. coordinator's example ----------
ex = [(1,2,3),(4,5,6),(1,3,5),(2,4,6),(1,4,6),(2,3,5)]
ex0 = [tuple(x-1 for x in t) for t in ex]   # 0-index
deg = [0]*6; pdeg = Counter()
for t in ex0:
    for x in t: deg[x]+=1
    for pr in itertools.combinations(t,2): pdeg[pr]+=1
e_ok = all(d%3==0 for d in deg) and pdeg[(0,1)]%2==1
report("E: coordinator's parallel-class example: point-mod-3 OK, pair {1,2} leave-degree odd",
       e_ok, "degrees=%s, pairdeg(1,2)=%d" % (deg, pdeg[(0,1)]))

# ---------- F. k5=3 exhaustive ----------
# 3 pentads (multiset, mult<=2), require every pair covered by an even number
# of pentads.  Up to relabeling: P1 = {0..4}; P2 uses i = |P1 & P2| canonical
# points {0..i-1} plus new points {5..9-i...}; P3 arbitrary subset of the
# <= 15 relevant points.  We simply brute-force P2, P3 over subsets of 15
# points with P1 fixed (complete for <= 15 support; 3 pentads touch <= 15).
def pairs_even_3pentads():
    # parity condition == pair-indicator vectors XOR to zero:
    # K_{P1} ^ K_{P2} ^ K_{P3} = 0 over the C(15,2) pair coordinates.
    PAIRS = {pr: i for i, pr in enumerate(itertools.combinations(range(15), 2))}
    all5 = list(itertools.combinations(range(15), 5))
    def mask(P):
        v = 0
        for pr in itertools.combinations(P, 2):
            v |= 1 << PAIRS[pr]
        return v
    masks = {P: mask(P) for P in all5}
    bymask = {}
    for P, v in masks.items():
        bymask.setdefault(v, []).append(P)
    found = []
    # WLOG the parity condition is invariant under which pentad is called P1;
    # scan ALL ordered pairs (P1,P2) and look up P3 = required mask.
    for P1 in all5:
        m1 = masks[P1]
        for P2 in all5:
            need = m1 ^ masks[P2]
            for P3 in bymask.get(need, []):
                ms = Counter([P1, P2, P3])
                if any(v > 2 for v in ms.values()):
                    continue  # multiplicity 3 illegal (triple covered 3x)
                found.append((P1, P2, P3))
    return found
found = pairs_even_3pentads()
report("F: k5=3: no 3-pentad multiset has all pair-multiplicities even (<=15 pts, P1 fixed)",
       len(found)==0, "violations found: %d" % len(found))

print()
print("OVERALL:", "ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)

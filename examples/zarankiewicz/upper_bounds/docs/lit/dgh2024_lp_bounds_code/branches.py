"""Branch-level use of DGH constraints: enumerate column-sum partitions (n parts in
[s-1,m] summing to w) and count survivors of (3) alone vs (3)+(4). Exact integer
arithmetic. Also used to verify IP<LP claims (zero survivors at w = LP bound)."""
import sys, signal
from math import comb
from collections import Counter
sys.path.insert(0,'.')
from dgh_lp import solve, roman, fl

def partitions(total, parts, lo, hi):
    res=[]
    def gen(rem, left, mx, pre):
        if left==0:
            if rem==0: res.append(pre)
            return
        top=min(mx, rem-lo*(left-1))
        for x in range(top, lo-1, -1):
            if x*left < rem: break
            gen(rem-x, left-1, x, pre+[x])
    gen(total, parts, hi, [])
    return res

def int_constraints(m,s,t, vs):
    idx=range(s-1,m+1); cons=[]
    cons.append(({i:comb(i,s) for i in idx}, (t-1)*comb(m,s), 'KST'))
    for v in vs:
        for k in range(s, m+1):
            D=comb(k-v,s-v); R=(t-1)*comb(m-v,s-v); a=R%D; c=(R-a)//D
            co={i:((comb(i-v,s-v)-a)*comb(i,v) if i<k else (D-a)*comb(i,v)) for i in idx}
            cons.append((co,(D-a)*comb(m,v)*c,f'v{v}k{k}'))
    return cons

def survivors(m,n,s,t,w, vs):
    cons=int_constraints(m,s,t,vs)
    out=[]
    for p in partitions(w,n,s-1,m):
        mult=Counter(p); ok=True; minslack=None; tight=None
        for co,rhs,name in cons:
            lhs=sum(co[i]*c for i,c in mult.items())
            if lhs>rhs: ok=False; break
            sl=(rhs-lhs)/rhs
            if minslack is None or sl<minslack: minslack=sl; tight=name
        if ok: out.append((p,minslack,tight))
    return out

class TO(Exception): pass
def handler(*a): raise TO()
signal.signal(signal.SIGALRM, handler)

if __name__=='__main__':
    print("=== Exact verification of IP<LP cells: survivors at w = LP bound (must be 0) ===")
    cells=[(3,3,10,23),(3,3,13,18),(3,3,14,19),(3,4,7,11),(3,4,10,11),(3,4,11,11),(3,4,11,16),(3,4,13,19),(3,4,15,18),
           (3,5,6,9),(3,5,8,11),(3,5,11,19),(3,5,11,23),(3,5,12,17),(3,5,13,20),(3,5,13,22),(4,4,11,16),(4,4,13,14),(4,4,13,16)]
    for s,t,m,n in cells:
        E=fl(solve(m,n,s,t)[0]); IP=fl(solve(m,n,s,t,integer=True)[0])
        signal.alarm(60)
        try:
            nparts=len(partitions(E,n,s-1,m))
            sv=survivors(m,n,s,t,E,range(1,s))
            svm1=survivors(m,n,s,t,E-1,range(1,s))
            print(f"z({m},{n};{s},{t}): Roman={roman(m,n,s,t)} LP={E} IP={IP}; partitions@w={E}: {nparts}, survivors={len(sv)}; survivors@w={E-1}: {len(svm1)}")
        except TO:
            print(f"z({m},{n};{s},{t}): LP={E} IP={IP}; enumeration timed out")
        signal.alarm(0)

    print("\n=== Branch-level pruning power (column side) ===")
    for s,t,m,n,ws in [(3,3,15,17,[132,133,134]),(3,3,10,22,[111,112]),(3,3,10,23,[114,115,116]),(4,4,11,14,[110,111,112]),(3,4,5,7,[27,28])]:
        cons_all=int_constraints(m,s,t,range(1,s))
        for w in ws:
            signal.alarm(120)
            try:
                P=partitions(w,n,s-1,m)
                kst=survivors(m,n,s,t,w,[])
                vs1=survivors(m,n,s,t,w,[s-1])
                allv=survivors(m,n,s,t,w,range(1,s))
                import statistics
                sl=[x[1] for x in allv]
                tightc=Counter(x[2] for x in allv)
                print(f"z({m},{n};{s},{t}) w={w}: partitions={len(P)} | pass KST(3)={len(kst)} | +(4)v=s-1={len(vs1)} | +(4)all v={len(allv)}"
                      + (f" | min-slack median={statistics.median(sl):.4f} min={min(sl):.4f} max={max(sl):.4f} tightest-constraint counts={dict(tightc.most_common(4))}" if allv else ""))
            except TO:
                print(f"z({m},{n};{s},{t}) w={w}: timed out")
            signal.alarm(0)
    print("\n=== Row side (dual: rows are edges over n column-vertices, (s,t) swapped) for z(15,17;3,3) w=132 ===")
    m,n,s,t,w=15,17,3,3,132
    P=partitions(w,m,t-1,n)
    kst=survivors(n,m,t,s,w,[]); allv=survivors(n,m,t,s,w,range(1,t))
    print(f"row partitions={len(P)} | pass KST={len(kst)} | +(4)all v={len(allv)}")

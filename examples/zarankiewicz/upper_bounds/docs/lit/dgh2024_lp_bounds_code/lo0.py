"""Check constraints (3),(4) extended to column sizes 0..m with safe binomials
(C(a,b)=0 when a<b or a<0): survivors at w = LP bound must still be 0 for IP cells,
and the survivor sets at w=E for ordinary cells must be unchanged."""
import sys; sys.path.insert(0,'.')
from branches import partitions
from collections import Counter
from math import comb as _comb
def comb(a,b): return _comb(a,b) if (a>=0 and b>=0 and a>=b) else 0
def cons_ext(m,s,t):
    idx=range(0,m+1); cons=[({i:comb(i,s) for i in idx},(t-1)*comb(m,s),'KST')]
    for v in range(1,s):
        for k in range(s,m+1):
            D=comb(k-v,s-v); R=(t-1)*comb(m-v,s-v); a=R%D; c=(R-a)//D
            co={i:((comb(i-v,s-v)-a)*comb(i,v) if i<k else (D-a)*comb(i,v)) for i in idx}
            cons.append((co,(D-a)*comb(m,v)*c,f'v{v}k{k}'))
    return cons
def surv(m,n,s,t,w):
    cons=cons_ext(m,s,t); out=[]
    for p in partitions(w,n,0,m):
        mult=Counter(p)
        if all(sum(co[i]*c for i,c in mult.items())<=rhs for co,rhs,_ in cons): out.append(p)
    return out
for (m,n,s,t,w) in [(10,23,3,3,115),(10,23,3,3,114),(7,11,3,4,54),(7,11,3,4,53),(10,22,3,3,111),(11,14,4,4,110),(6,9,3,5,43),(6,9,3,5,42)]:
    P=partitions(w,n,0,m); S=surv(m,n,s,t,w)
    small=[p for p in S if min(p)<s-1]
    print(f"z({m},{n};{s},{t}) w={w}: partitions(parts in [0,{m}])={len(P)} survivors={len(S)} (with a column < s-1: {len(small)})")

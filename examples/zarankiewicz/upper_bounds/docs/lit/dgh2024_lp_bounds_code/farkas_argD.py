import sys; sys.path.insert(0,'.')
from fractions import Fraction as F
from math import comb
from dgh_lp import con4
from branches import partitions, survivors

# --- exact Farkas certificate for E(15,17;3,3) <= 132.74 : multipliers y_eq=138/25, y(4,v=2,k=8)=1/25, y(4,v=2,k=15)=221/2100
m,n,s,t=15,17,3,3
y_eq=F(138,25); ys={8:F(1,25), 15:F(221,2100)}
idx=range(s-1,m+1)
rows={k:con4(m,s,t,2,k) for k in ys}
bound=y_eq*n+sum(y*rows[k][1] for k,y in ys.items())
ok=all(y_eq+sum(y*rows[k][0][i-(s-1)] for k,y in ys.items()) >= i for i in idx)
print("Farkas certificate z(15,17;3,3): bound =",bound,"=",float(bound),"; pointwise coefficient >= i for all i:",ok)
print("  per-size coefficient minus i:", {i: y_eq+sum(y*rows[k][0][i-(s-1)] for k,y in ys.items())-i for i in idx})
print("  rhs of (4) k=8:",rows[8][1]," k=15:",rows[15][1])

# --- Argument D (Tan / Guy) cross-prune on (row partition, column partition) pairs at z(15,17;3,3), w=132
w=132
cols=[p for p,_,_ in survivors(m,n,s,t,w,range(1,s))]          # 14 column partitions (non-increasing)
rowsP=[p for p,_,_ in survivors(n,m,t,s,w,range(1,t))]        # 1688 row partitions
RD=(t-1)*comb(m-1,s-1)   # row with r ones: sum over its r columns of C(c_j-1, s-1) <= (t-1) C(m-1, s-1)
CD=(s-1)*comb(n-1,t-1)   # column with c ones: sum over its c rows of C(r_i-1, t-1) <= (s-1) C(n-1, t-1)
pairs=0; surv=0; survRowOnly=0
for c in cols:
    cs=sorted(c)  # ascending: pessimal = smallest sums
    pref=[0]
    for x in cs: pref.append(pref[-1]+comb(x-1,s-1))
    for r in rowsP:
        pairs+=1
        rmax=max(r)
        okR = pref[rmax] <= RD
        rs=sorted(r); cmax=max(c)
        okC = sum(comb(x-1,t-1) for x in rs[:cmax]) <= CD
        if okR: survRowOnly+=1
        if okR and okC: surv+=1
print(f"Argument D at z(15,17;3,3) w=132: pairs={pairs}, survive row-form={survRowOnly}, survive both forms={surv}; RD={RD}, CD={CD}")
# distribution of surviving pairs per column partition
from collections import Counter
cnt=Counter()
for c in cols:
    cs=sorted(c); pref=[0]
    for x in cs: pref.append(pref[-1]+comb(x-1,s-1))
    k=0
    for r in rowsP:
        rs=sorted(r)
        if pref[max(r)]<=RD and sum(comb(x-1,t-1) for x in rs[:max(c)])<=CD: k+=1
    cnt[tuple(c)]=k
for c,k in sorted(cnt.items(), key=lambda kv:-kv[1]): print("  col partition",c,"-> surviving row partitions:",k)

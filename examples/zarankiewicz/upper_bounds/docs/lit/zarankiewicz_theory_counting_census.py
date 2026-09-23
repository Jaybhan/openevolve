"""Census of profile-level pruning inequalities (companion to zarankiewicz_theory_counting.md).
Convention: m rows, n cols, forbid s x t. Minor table = Roman bounds only (no exact values).
Run: python3 zarankiewicz_theory_counting_census.py   (about 3 minutes)
"""
import time
from math import comb
from fractions import Fraction as F
from itertools import product, combinations_with_replacement
import sys

def roman1(s,t,m,n):
    # count s-subsets of rows covered by columns: sum_j C(c_j,s) <= (t-1)C(m,s); Roman with p>=s-1
    f = lambda p: int(F(t-1, comb(p,s-1))*comb(m,s) + F((p+1)*(s-1), s)*n)
    p=s-1; prev=f(p)
    while True:
        nxt=f(p+1)
        if nxt>=prev: return prev
        prev=nxt; p+=1
def roman(s,t,m,n): return min(roman1(s,t,m,n), roman1(t,s,n,m))

def zub(s,t,m,n):
    if s>m or t>n: return m*n
    return roman(s,t,m,n)

def partitions(total, parts, maxpart, s_side_count, tlim, side):
    """all non-increasing partitions of total into `parts` parts in [0,maxpart] satisfying
    argument A: sum C(x, k) <= (l-1) C(other, k) and E (prefix sums <= zub)."""
    out=[]
    def rec(k, rem, mx, prefix):
        if len(prefix)>0:
            pass
        if k==0:
            if rem==0: out.append(tuple(prefix))
            return
        lo = -(-rem//k)
        for x in range(min(mx,rem), lo-1, -1):
            rec(k-1, rem-x, x, prefix+[x])
    rec(parts, total, maxpart, [])
    return out

def argA_cols(c, s,t,m): return sum(comb(x,s) for x in c) <= (t-1)*comb(m,s)
def argA_rows(r, s,t,n): return sum(comb(x,t) for x in r) <= (s-1)*comb(n,t)
def argE_cols(c, s,t,m):  # prefix of largest columns vs z(m, n')
    return all(sum(c[:k]) <= zub(s,t,m,k) for k in range(1,len(c)))
def argE_rows(r, s,t,n):
    return all(sum(r[:k]) <= zub(s,t,k,n) for k in range(1,len(r)))
def C(a,b):
    return comb(a,b) if a>=0 and b>=0 else 0
def argD(c, r, s,t,m,n):
    # heaviest row (r[0] ones) meets the r[0] lightest columns: sum C(c_j-1, s-1) <= (t-1) C(m-1, s-1)
    ok1 = sum(C(x-1,s-1) for x in c[-r[0]:]) <= (t-1)*comb(m-1,s-1) if r[0]>0 else True
    ok2 = sum(C(x-1,t-1) for x in r[-c[0]:]) <= (s-1)*comb(n-1,t-1) if c[0]>0 else True
    return ok1 and ok2
def gale_ryser(r, c):
    # necessity: for all k, sum of k largest c <= sum_i min(r_i, k)
    cs = sorted(c, reverse=True)
    return all(sum(cs[:k]) <= sum(min(x,k) for x in r) for k in range(1,len(cs)+1))
def dense_cap(r, c, s,t,n,m):
    # any s rows: |common cols| >= sum r_i - (s-1) n; must be <= t-1
    top = sorted(r, reverse=True)[:s]
    ok1 = sum(top) - (s-1)*n <= t-1
    topc = sorted(c, reverse=True)[:t]
    ok2 = sum(topc) - (t-1)*m <= s-1
    return ok1 and ok2
def dgh4(c, s,t,m):
    # DGH Lemma 3.2 (4): hypergraph with m vertices(rows), n edges(cols) of sizes c_j, (s,t-1)-linear.
    # n_i = number of columns with sum i. For v<s<=k<=m.
    from collections import Counter
    cnt = Counter(c)
    for v in range(1,s):
        for k in range(s, m+1):
            B = comb(k-v, s-v)
            alpha = ((t-1)*comb(m-v, s-v)) % B
            lhs = F(0)
            for i,ni in cnt.items():
                if i < s-1:  # treat as size s-1 (no s-subsets); DGH allow replacing small edges
                    ii = s-1
                else: ii = i
                if ii < k:
                    lhs += F(comb(ii-v,s-v)-alpha, B-alpha)*comb(ii,v)*ni
                else:
                    lhs += comb(ii,v)*ni
            rhs = F(comb(m,v)*((t-1)*comb(m-v,s-v)-alpha), B)
            if lhs > rhs: return False
    return True

def census(s,t,m,n,w):
    cols = [c for c in partitions(w, n, m, None,None,None) if argA_cols(c,s,t,m) and argE_cols(c,s,t,m)]
    rows = [r for r in partitions(w, m, n, None,None,None) if argA_rows(r,s,t,n) and argE_rows(r,s,t,n)]
    pairs = [(c,r) for c in cols for r in rows]
    tan = [(c,r) for (c,r) in pairs if argD(c,r,s,t,m,n)]
    gr = [(c,r) for (c,r) in tan if gale_ryser(r,c)]
    dc = [(c,r) for (c,r) in tan if dense_cap(r,c,s,t,n,m)]
    d4c = [c for c in cols if dgh4(c,s,t,m)]
    d4r = [r for r in rows if dgh4(r,t,s,n)]
    d4 = [(c,r) for (c,r) in tan if dgh4(c,s,t,m) and dgh4(r,t,s,n)]
    allp = [(c,r) for (c,r) in tan if gale_ryser(r,c) and dense_cap(r,c,s,t,n,m) and dgh4(c,s,t,m) and dgh4(r,t,s,n)]
    print(f"z({m},{n};{s},{t}) w={w}: cols(A+E)={len(cols)} rows(A+E)={len(rows)} pairs={len(pairs)} "
          f"| after D (Tan)={len(tan)} | +GaleRyser={len(gr)} | +denseCap={len(dc)} | DGH4 cols={len(d4c)}/{len(cols)} rows={len(d4r)}/{len(rows)} pairs={len(d4)} | all={len(allp)}")
    return tan, gr

def census_lazy(s,t,m,n,w, limit_pairs=3_000_000):
    t0=time.time()
    cols = [c for c in partitions(w, n, m, None,None,None) if argA_cols(c,s,t,m) and argE_cols(c,s,t,m)]
    rows = [r for r in partitions(w, m, n, None,None,None) if argA_rows(r,s,t,n) and argE_rows(r,s,t,n)]
    d4c = sum(1 for c in cols if dgh4(c,s,t,m)); d4r = sum(1 for r in rows if dgh4(r,t,s,n))
    npairs=len(cols)*len(rows)
    if npairs>limit_pairs:
        print(f"z({m},{n};{s},{t}) w={w}: cols={len(cols)} rows={len(rows)} pairs={npairs} (too many; DGH4 cols {d4c}/{len(cols)} rows {d4r}/{len(rows)})"); return
    tan=gr=dc=d4=allp=0
    ex=None
    for c in cols:
        for r in rows:
            if not argD(c,r,s,t,m,n): continue
            tan+=1
            g=gale_ryser(r,c); d=dense_cap(r,c,s,t,n,m); q=dgh4(c,s,t,m) and dgh4(r,t,s,n)
            gr+=g; dc+=d; d4+=q; allp+= (g and d and q)
            if not g and ex is None: ex=(c,r)
    print(f"z({m},{n};{s},{t}) w={w}: cols(A+E)={len(cols)} rows(A+E)={len(rows)} pairs={npairs} | after D (Tan)={tan} | +GR={gr} | +denseCap={dc} | +DGH4={d4} (cols {d4c}/{len(cols)}, rows {d4r}/{len(rows)}) | all={allp}  [{time.time()-t0:.1f}s]")
    if ex: print("   example pair killed by Gale-Ryser only: cols",ex[0],"rows",ex[1])
if __name__ == "__main__":
    for (s,t,m,n,w) in [(2,2,16,16,68),(2,2,17,17,75),(2,2,16,17,71),(3,3,10,10,61),(3,3,11,11,70),(3,3,12,12,81),(3,3,13,13,93),(3,3,14,14,106),(3,3,13,17,117),(3,3,13,18,122),(3,3,13,18,117),(4,4,11,14,107),(4,4,10,10,75),(3,3,16,16,129)]:
        census_lazy(s,t,m,n,w, limit_pairs=400000)

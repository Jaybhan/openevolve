"""Reproduce Davies-Gill-Horsley (arXiv:2411.18842) LP bounds on z(m,n;s,t).

Variables n_i, i = s-1..m  (number of columns / hyperedges of size i).
(2)  sum n_i = n
(3)  sum C(i,s) n_i <= (t-1) C(m,s)
(4)  for v in 1..s-1, k in s..m:
       1/(C(k-v,s-v)-alpha) * sum_{i<k} (C(i-v,s-v)-alpha) C(i,v) n_i
       + sum_{i>=k} C(i,v) n_i  <=  C(m,v) * ((t-1)C(m-v,s-v)-alpha)/C(k-v,s-v)
     alpha = (t-1)C(m-v,s-v) mod C(k-v,s-v).
objective: maximise sum i n_i.
"""
import sys
from math import comb, floor
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

def alpha_c(m,s,t,v,k):
    D = comb(k-v, s-v); R = (t-1)*comb(m-v, s-v)
    a = R % D; c = (R - a)//D
    return D, R, a, c

def con4(m,s,t,v,k):
    D,R,a,c = alpha_c(m,s,t,v,k)
    co = []
    for i in range(s-1, m+1):
        if i < k:
            co.append(Fraction(comb(i-v, s-v) - a, D - a) * comb(i, v))
        else:
            co.append(Fraction(comb(i, v)))
    return co, Fraction(comb(m, v) * c)

def build(m,n,s,t, vs=None, ks=None):
    idx = list(range(s-1, m+1))
    A_ub = [[comb(i,s) for i in idx]]; b_ub = [(t-1)*comb(m,s)]; names=['(3)KST']
    if vs is None: vs = range(1, s)
    for v in vs:
        for k in (ks if ks is not None else range(s, m+1)):
            co, rhs = con4(m,s,t,v,k)
            A_ub.append([float(x) for x in co]); b_ub.append(float(rhs)); names.append(f'(4)v={v},k={k}')
    return idx, np.array(A_ub,float), np.array(b_ub,float), names

def solve(m,n,s,t, vs=None, ks=None, integer=False):
    idx, A_ub, b_ub, names = build(m,n,s,t,vs,ks)
    cobj = -np.array(idx, float)
    if not integer:
        res = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=[[1]*len(idx)], b_eq=[n],
                      bounds=[(0,None)]*len(idx), method='highs')
        assert res.status == 0, res.message
        return -res.fun, res, idx, names
    cons = [LinearConstraint(A_ub, -np.inf, b_ub), LinearConstraint(np.ones((1,len(idx))), n, n)]
    res = milp(cobj, constraints=cons, integrality=np.ones(len(idx)), bounds=Bounds(0, np.inf))
    assert res.status == 0, res.message
    return -res.fun, res, idx, names

def roman(m,n,s,t):
    best = None
    for k in range(s-1, m+1):
        b = Fraction((t-1)*comb(m,s), comb(k,s-1)) + Fraction((k+1)*(s-1)*n, s)
        if best is None or b < best: best = b
    return floor(best)

def thm13(m,n,s,t, variant='B'):
    """Closed form B_k. variant 'A' = literal reading of typeset formula
    (prefactor C(m,s-1)/C(k,s-1) on whole bracket); variant 'B' = derived from
    A*(2)+B*(3)+C*(8) as in the proof (the '+1' term not divided by C(k,s-1))."""
    best=None
    for k in range(max(2, s*s-2*s), m+1):
        if k < s: continue
        a = ((t-1)*(m-s+1)) % (k-s+1)
        beta = Fraction((s-1)*(k-s+1) - a*(s-1), (k+1)*(k-s+1) - a*(s-1))
        pre = Fraction(comb(m,s-1), comb(k,s-1))
        lin = Fraction((k+1)*(s-1) , s) * (1 - beta) * 0  # placeholder
        An = Fraction((k+1),s)*(s-1-beta)*n
        B = pre*( Fraction((t-1)*(m-s+1), s)*(beta*(k+1)/(k-s+1) + 1) - a*beta*(k-s+2)/(k-s+1) ) + An
        if best is None or B < best: best=B
    return floor(best) if best is not None else 10**9

def fl(x): return floor(x + 1e-7)


if __name__ == '__main__':
    import sys
    tables = {(3,3):(range(10,17),range(17,24)), (3,4):(range(5,17),range(7,24)),
              (3,5):(range(6,17),range(8,24)), (4,4):(range(10,17),range(14,24)),
              (4,5):(range(7,17),range(9,24)), (5,5):(range(8,17),range(10,24))}
    for (s,t),(ms,ns) in tables.items():
        print(f"\n===== (s,t)=({s},{t}) : cells where floor(E) < Roman or IP < Roman =====")
        print(" m  n | Roman   E  E*  T1.3  swapE   IP | mark")
        for m in ms:
            for n in ns:
                if n < m: continue
                R = roman(m,n,s,t)
                E = fl(solve(m,n,s,t)[0]); Es = fl(solve(m,n,s,t,vs=[s-1])[0])
                T = thm13(m,n,s,t)
                SW = fl(solve(n,m,t,s)[0])
                IP = fl(solve(m,n,s,t,integer=True)[0])
                if E < R or IP < R or SW < R:
                    mark = ('*' if T > E else '') + (' BOLD' if R-E>=2 else '') + (' IP<LP' if IP<E else '') + (' SWAP<E' if SW<E else '')
                    print(f"{m:2d} {n:2d} | {R:5d} {E:4d} {Es:4d} {T:5d} {SW:6d} {IP:4d} | {mark}")

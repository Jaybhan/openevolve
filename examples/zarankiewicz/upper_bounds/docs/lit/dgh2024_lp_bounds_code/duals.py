import sys; sys.path.insert(0,'.')
from dgh_lp import solve, roman, fl, build
from fractions import Fraction
import numpy as np
for (m,n,s,t) in [(15,17,3,3),(10,23,3,3),(16,22,4,5)]:
    val,res,idx,names = solve(m,n,s,t)
    print(f"\n=== E({m},{n};{s},{t}) = {val:.6f} -> floor {fl(val)} ; Roman={roman(m,n,s,t)} ===")
    x=res.x
    print("optimal n_i (nonzero):", {i:round(float(v),4) for i,v in zip(idx,x) if v>1e-9})
    yeq=res.eqlin.marginals[0]; yin=res.ineqlin.marginals
    print(f"dual of (2) [sum n_i = n]: {yeq:.6f}")
    for nm,y,sl in zip(names,yin,res.ineqlin.residual):
        if abs(y)>1e-9: print(f"  dual of {nm}: {-y:.6f}   (slack {sl:.3g})")
    # reduced costs: rho_i = sum_c y_c a_ci - i  (>=0 at optimum)
    idx2,A_ub,b_ub,_=build(m,n,s,t)
    rho = yeq*np.ones(len(idx)) + (-yin)@A_ub - np.array(idx,float)
    print("reduced cost rho_i by column size i:", {i:round(float(r),3) for i,r in zip(idx,rho)})
    print("certificate check: y_eq*n + sum y_c b_c =", yeq*n + (-yin)@b_ub)

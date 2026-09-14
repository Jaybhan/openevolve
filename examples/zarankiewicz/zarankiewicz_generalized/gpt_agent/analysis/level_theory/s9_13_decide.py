"""Is S_9(13 pentads) >= 8 quads? Forced: every point-leave = 2, so
2*r5_x + r4_x = 18 at every point; enumerate r5-profiles (nonincreasing,
in [5,9], sum 65), pin BOTH degree classes, MILP each."""
import numpy as np, time
from itertools import combinations
from math import comb
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

m = 9
quads = list(combinations(range(m), 4)); pents = list(combinations(range(m), 5))
tris = {t: i for i, t in enumerate(combinations(range(m), 3))}
nT = len(tris); nQ = len(quads); nP = len(pents)
profs = []
def rec(pre, rem, slots, mx):
    if slots == 0:
        if rem == 0: profs.append(pre)
        return
    for v in range(min(mx, rem - 5*(slots-1)), 4, -1):
        if rem - v >= 5*(slots-1): rec(pre+[v], rem-v, slots-1, v)
rec([], 65, 9, 9)
print(len(profs), "r5-profiles")
found = False
t00 = time.time()
for pi, r5 in enumerate(profs):
    r4 = [18 - 2*v for v in r5]
    A = lil_matrix((nT + 2*m + 2, nQ + nP))
    lb = np.zeros(nT + 2*m + 2); ub = np.zeros(nT + 2*m + 2)
    for j, b in enumerate(quads):
        for t in combinations(b, 3): A[tris[t], j] = 1
        for x in b: A[nT + x, j] = 1
        A[nT + 2*m, j] = 1
    for j, b in enumerate(pents):
        for t in combinations(b, 3): A[tris[t], nQ + j] = 1
        for x in b: A[nT + m + x, nQ + j] = 1
        A[nT + 2*m + 1, nQ + j] = 1
    ub[:nT] = 2
    for x in range(m):
        lb[nT+x] = ub[nT+x] = r4[x]
        lb[nT+m+x] = ub[nT+m+x] = r5[x]
    lb[nT+2*m] = ub[nT+2*m] = 8
    lb[nT+2*m+1] = ub[nT+2*m+1] = 13
    res = milp(c=np.zeros(nQ+nP), constraints=LinearConstraint(A.tocsc(), lb, ub),
               bounds=Bounds(0, 2), integrality=np.ones(nQ+nP),
               options={"time_limit": 120})
    st = "FEAS" if res.status == 0 else "INFEAS" if res.status == 2 else "TIMEOUT"
    print(f"profile {pi+1}/{len(profs)} r5={r5}: {st}", flush=True)
    if st != "INFEAS":
        found = True
        if st == "FEAS": break
print("VERDICT: S_9(13) >= 8 is", "UNRESOLVED/FEAS" if found else
      "REFUTED  =>  S_9(13) <= 7", f"({time.time()-t00:.0f}s)")

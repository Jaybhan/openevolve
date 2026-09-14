"""S_9(4 pentads, 1 hexad): need >= 27 refuted? Deep fixing: hexad H={0..5};
P1 canonical by a=|P1 cap H|; P2 enumerated over all C(9,5) candidates
(sorted-orbit dedup vs stabilizer is skipped -- full enumeration is sound,
just redundant); P3,P4 free MILP vars. Reports max quads over all cases."""
import time
from itertools import combinations
from pinned_slice import _solve

m = 9
H = tuple(range(6))
best = -1
t0 = time.time()
ncase = 0
for a in range(2, 6):
    P1 = tuple(list(range(a)) + list(range(6, 6 + 5 - a)))
    for P2 in combinations(range(m), 5):
        ncase += 1
        v = _solve(m, [H, P1, P2], {4: None, 5: 2}, 60)
        if v is None:
            print("case timeout", a, P2, flush=True)
        elif v > best:
            best = v
            print(f"new best {best} at a={a} P2={P2} ({time.time()-t0:.0f}s)", flush=True)
print(f"S_9(4,1) = {best}  ({ncase} cases, {time.time()-t0:.0f}s)")

import sys, signal; sys.path.insert(0,'.')
from branches import partitions, survivors, TO, handler
from dgh_lp import solve, roman, fl
signal.signal(signal.SIGALRM, handler)
tables = {(4,5):(range(7,17),range(9,24)), (5,5):(range(8,17),range(10,24))}
for (s,t),(ms,ns) in tables.items():
    for m in ms:
        for n in ns:
            if n<m: continue
            E=fl(solve(m,n,s,t)[0]); IP=fl(solve(m,n,s,t,integer=True)[0])
            if IP<E:
                signal.alarm(240)
                try:
                    sv=survivors(m,n,s,t,E,range(1,s)); svm=survivors(m,n,s,t,E-1,range(1,s))
                    print(f"z({m},{n};{s},{t}): Roman={roman(m,n,s,t)} LP={E} IP={IP} survivors@{E}={len(sv)} survivors@{E-1}={len(svm)}", flush=True)
                except TO:
                    print(f"z({m},{n};{s},{t}): Roman={roman(m,n,s,t)} LP={E} IP={IP} enumeration timed out", flush=True)
                signal.alarm(0)

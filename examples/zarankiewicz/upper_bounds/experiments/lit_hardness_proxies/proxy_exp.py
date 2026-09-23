"""Local experiment: cheap difficulty proxies vs recorded CaDiCaL solve time
for the 21 column-weight profiles of z(10,14;3,3) at weight 78 (all UNSAT),
recorded in gpt_agent/analysis/sat_attack/results/V78_p*.json."""
import glob, json, os, sys, time
from math import comb, log2
from itertools import combinations
sys.path.insert(0, '/Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/zarankiewicz_generalized/gpt_agent/analysis/sat_attack')
from encodings_zar import encode_matrix
from pysat.solvers import Solver
from scipy.stats import spearmanr

RES = '/Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/zarankiewicz_generalized/gpt_agent/analysis/sat_attack/results'
m, n, total = 10, 14, 78
rows = []
for f in sorted(glob.glob(os.path.join(RES, 'V78_p*.json'))):
    d = json.load(open(f))
    rows.append((os.path.basename(f), tuple(d['weights']), d['status'], d['solve_seconds']))
print(len(rows), 'profiles')

def gale_ryser_ok(r, c):
    # 0/1 matrix with row sums r (len m) and col sums c (len n) exists iff
    # sum r == sum c and for all k: sum_{i<=k} r_i^{sorted desc} <= sum_j min(c_j, k)
    r = sorted(r, reverse=True)
    if sum(r) != sum(c): return False
    acc = 0
    for k in range(1, len(r)+1):
        acc += r[k-1]
        if acc > sum(min(cj, k) for cj in c): return False
    return True

def row_partitions(total, m, n):
    # non-increasing partitions of total into m parts in [0, n] with KST row bound
    budget = 2*comb(n, 3)
    out = []
    def gen(rem, parts, maxv, cost, pre):
        if parts == 0:
            if rem == 0 and cost <= budget: out.append(tuple(pre))
            return
        hi = min(maxv, rem)
        for v in range(hi, -1, -1):
            if rem - v > n*(parts-1): break
            c2 = cost + comb(v, 3)
            if c2 > budget: continue
            gen(rem - v, parts-1, v, c2, pre+[v])
    gen(total, m, n, 0, [])
    return out
RP = row_partitions(total, m, n)
print(len(RP), 'row partitions satisfying sum and KST row bound')

def guy_d_ok(r, c):
    # Argument D (row form) from cases.py: heaviest row r0 -> sum over r0 lightest cols of C(c_j-1,2) <= 2*C(m-1,2)
    r0 = max(r); light = sorted(c)[:r0]
    if sum(comb(max(cj-1,0), 2) for cj in light) > 2*comb(m-1, 2): return False
    c0 = max(c); light = sorted(r)[:c0]
    if sum(comb(max(ri-1,0), 2) for ri in light) > 2*comb(n-1, 2): return False
    return True

out = []
for name, w, status, t in rows:
    feats = {}
    t0 = time.time()
    clauses, nv, X = encode_matrix(m, n, total, col_weights=list(w))
    feats['kst_slack'] = 2*comb(m,3) - sum(comb(x,3) for x in w)
    feats['max_w'] = max(w); feats['distinct_w'] = len(set(w))
    feats['log2_vol'] = sum(log2(comb(m, x)) for x in w)
    # (5) surviving sub-cases: row partitions compatible (Gale-Ryser + Argument D)
    feats['subcases_GR'] = sum(1 for r in RP if gale_ryser_ok(r, w))
    feats['subcases_GR_D'] = sum(1 for r in RP if gale_ryser_ok(r, w) and guy_d_ok(r, w))
    t_static = time.time() - t0
    # (1) unit propagation / failed literal lookahead
    t0 = time.time()
    s = Solver(name='cadical195', bootstrap_with=clauses)
    ok, lits = s.propagate()
    cells = m*n
    fixed = set(abs(l) for l in lits if abs(l) <= cells) if ok else set(range(1, cells+1))
    feats['root_fixed_frac'] = len(fixed)/cells
    failed = 0; impl = 0
    for v in range(1, cells+1):
        if v in fixed: continue
        for lit in (v, -v):
            ok2, lits2 = s.propagate(assumptions=[lit])
            if not ok2: failed += 1
            else: impl += sum(1 for l in lits2 if abs(l) <= cells) - 1
    feats['failed_lits'] = failed
    feats['mean_implied_cells'] = impl / max(1, 2*(cells-len(fixed)))
    s.delete()
    t_la = time.time() - t0
    # (2) conflict-budgeted runs
    for cap in (1000, 5000, 20000):
        t0 = time.time()
        s = Solver(name='cadical195', bootstrap_with=clauses)
        s.conf_budget(cap)
        r = s.solve_limited()
        st = s.accum_stats(); s.delete()
        feats[f'st_{cap}'] = {'res': r, **st, 'sec': round(time.time()-t0, 2)}
    out.append((name, w, status, t, feats, t_static, t_la))
    print(name, w, status, t, 'static %.2fs la %.2fs' % (t_static, t_la), {k: v for k, v in feats.items() if not k.startswith('st_')}, feats['st_20000'], flush=True)

json.dump([{'name': a, 'w': b, 'status': c, 't': d, 'feats': e, 't_static': f, 't_la': g} for a,b,c,d,e,f,g in out], open('proxy_exp_results.json','w'), indent=1)
T = [r[3] for r in out]
print('\nSpearman rho vs recorded solve seconds (n=%d; 4 rows are censored timeouts at their recorded times):' % len(out))
def rho(vals, label):
    r, p = spearmanr(vals, T); print('  %-28s rho=%+.3f p=%.3f' % (label, r, p))
for k in ['kst_slack','max_w','distinct_w','log2_vol','subcases_GR','subcases_GR_D','root_fixed_frac','failed_lits','mean_implied_cells']:
    rho([r[4][k] for r in out], k)
for cap in (1000, 5000, 20000):
    rho([r[4][f'st_{cap}']['propagations'] for r in out], f'props@{cap}conf')
    rho([r[4][f'st_{cap}']['decisions'] for r in out], f'decisions@{cap}conf')
    rho([r[4][f'st_{cap}']['sec'] for r in out], f'seconds@{cap}conf')
    rho([r[4][f'st_{cap}']['propagations']/max(1,r[4][f'st_{cap}']['conflicts']) for r in out], f'props/conf@{cap}')
    rho([r[4][f'st_{cap}']['decisions']/max(1,r[4][f'st_{cap}']['conflicts']) for r in out], f'dec/conf@{cap}')
    rho([r[4][f'st_{cap}']['restarts'] for r in out], f'restarts@{cap}')
solved = {cap: sum(1 for r in out if r[4][f'st_{cap}']['res'] is not None) for cap in (1000,5000,20000)}
print('solved within budget:', solved)

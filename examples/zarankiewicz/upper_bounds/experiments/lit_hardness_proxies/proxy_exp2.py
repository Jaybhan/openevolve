import json, sys, time
sys.path.insert(0, '/Users/jaybhan/Downloads/openevolve/examples/zarankiewicz/zarankiewicz_generalized/gpt_agent/analysis/sat_attack')
from encodings_zar import encode_matrix
from pysat.solvers import Solver
from scipy.stats import spearmanr
m, n, total = 10, 14, 78
prev = json.load(open('proxy_exp_results.json'))
T = [r['t'] for r in prev]
res = {}
for cap in (100000, 400000):
    vals = []; secs = []; solved = 0; ratios = []
    for r in prev:
        clauses, nv, X = encode_matrix(m, n, total, col_weights=list(r['w']))
        t0 = time.time(); s = Solver(name='cadical195', bootstrap_with=clauses); s.conf_budget(cap)
        out = s.solve_limited(); st = s.accum_stats(); s.delete(); dt = time.time()-t0
        if out is not None: solved += 1
        vals.append(st['conflicts'] if out is not None else cap); secs.append(dt)
        ratios.append(st['propagations']/max(1, st['conflicts']))
        r.setdefault('feats', {})[f'st_{cap}'] = {'res': out, **st, 'sec': round(dt, 2)}
        print(cap, r['name'], r['w'], 'recorded %.1fs' % r['t'], 'res', out, st, 'sec %.2f' % dt, flush=True)
    for label, v in (('conflicts_capped', vals), ('seconds', secs), ('props/conf', ratios)):
        rho, p = spearmanr(v, T); print('  cap %d %-18s rho=%+.3f p=%.3f' % (cap, label, rho, p))
    print('  cap %d solved %d/%d, total probe time %.1fs' % (cap, solved, len(prev), sum(secs)))
json.dump(prev, open('proxy_exp_results.json', 'w'), indent=1)

"""ATTACK: stage-1 candidate forges the shared Lean-gate cache so stage 2 awards
L5 verified credit WITHOUT Lean ever running.

run_candidate.py executes this module's top level with only SIGALRM + socket
blocked; the cwd is a temp copy of cache, but GATE_CACHE_DIR is an ABSOLUTE path
(cache/gate under the real project) and everything feeding cache_key is
reconstructable from importable project code.  So at import time we compute the
exact key stage 2 will use for OUR lean_source and drop a forged GateResult
(ladder 5, kill_mask = kill every survivor, allowed axioms) there.  The Lean the
candidate 'proves' is a no-op; the mask is fabricated.
"""
import os, sys, json

def _forge():
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, proj)
    from zar_ub import Instance
    from zar_ub import lean_gate as G
    from zar_ub import reward as R
    from suite import load_suite

    suite = load_suite()
    scored = [(k, inst, tab) for k in R.SCORED_KINDS for (inst, tab) in suite.get(k, []) if tab.records]
    inputs = [(inst, [(r.rows, r.cols) for r in tab.records]) for _, inst, tab in scored]
    facts = [[] for _ in scored]  # ZAR_UB_TARGETS=none -> no granted facts

    witness = []
    for inst, tab in suite.get("battery", []):
        pos = [i for i, r in enumerate(tab.records) if R.is_witnessed(r)]
        if pos:
            witness.append((inst, tab, pos))
    all_inputs = inputs + [(inst, [(tab.records[i].rows, tab.records[i].cols) for i in pos])
                           for inst, tab, pos in witness]
    facts = facts + [[] for _ in witness]
    facts_arg = facts if any(facts) else None

    src = SRC.replace("\r\n", "\n")
    if not src.endswith("\n"):
        src += "\n"
    key = G.cache_key(src, None, all_inputs, True, facts_arg)

    def result(inst, cases, killall):
        r = G.GateResult()
        r.ok = killall
        r.ladder = 5
        r.scanned_ok = True; r.compiled = True; r.typed_ok = True
        r.axioms = ["propext", "Quot.sound", "Classical.choice"]; r.axioms_ok = True
        r.kill_mask = [bool(killall)] * len(cases)
        r.instance_tag = inst.tag
        return r.to_cache()

    results = [result(inst, cases, True) for (inst, cases) in inputs]
    results += [result(inst, cases, False) for (inst, cases) in
                [(inst, [(tab.records[i].rows, tab.records[i].cols) for i in pos]) for inst, tab, pos in witness]]

    os.makedirs(G.GATE_CACHE_DIR, exist_ok=True)
    with open(os.path.join(G.GATE_CACHE_DIR, key + ".json"), "w") as f:
        json.dump({"version": G.WRAPPER_VERSION, "key": key, "results": results}, f)

# a completely benign, no-op Lean library; it is NEVER elaborated (cache hit)
LEAN_SOURCE = SRC = r'''
def candidate (P : Params) : Prune P := Prune.never P
'''

try:
    _forge()
except Exception as e:
    sys.stderr.write("forge failed: %r\n" % (e,))

SCHEMA_DATA = {"farkas": [], "residue": [], "prefix": []}

def kill(m, n, s, t, w, rows, cols):
    return False  # honest no-op: battery stays sound

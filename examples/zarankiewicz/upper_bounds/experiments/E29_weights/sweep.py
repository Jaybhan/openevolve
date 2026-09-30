"""E29: offline sweep of the v3 reward weights over the E27 re-scored benchmark (components are
weight-independent, so every candidate weight vector is scored exactly).  Criteria:
  P1  DGH (wide TRAIN closure, no target help) and the general recipe within 2x of each other in uplift
  P2  recipe+DGH above both
  P3  every REAL exploit (R1 e*, cert_close, inst_*, R3 'real' attacks) below min(recipe, DGH)
  P4  tiny-table clears (R3 A1/A2 oracles) <= f_weak
  P5  target-helping f_weak above every real exploit
  P6  C5: recipe keeps >= 0.8 of its old uplift (0.0594)            [reported; see text]
Objective among feasible vectors: minimal L1 change from the shipped weights."""
import itertools, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
rows = json.load(open(os.path.join(UB, "experiments/E27_integration/rescore.json")))["rows"]
SHIP = dict(tr=0.28, ta=0.21, ge=0.07, tl=0.14, de=0.15, cl=0.15)
def comp(name):
    v = rows[name]["v3/S1 new"]
    c = dict(tr=v["proven_gain"], ta=v["target_gain"], ge=0.0, tl=v["tail_gain"], de=v["depth_term"], cl=v["closure_bonus"])
    # rescore.json does not store G_gen: infer it exactly from the stored shipped score (linear, unclipped)
    known = sum(SHIP[k] * c[k] for k in c)
    c["ge"] = max(0.0, ((v["combined_score"] - 0.2) / 0.8 - known) / SHIP["ge"])
    return c
def score(c, w):
    inner = sum(w[k] * c[k] for k in w)
    return 0.2 + 0.8 * max(0.0, min(1.0, inner))
# sanity: reproduce shipped scores
for n in ("lean_dgh4", "schema_pool", "f_weak"):
    print(n, round(score(comp(n), SHIP), 4), "stored", round(rows[n]["v3/S1 new"]["combined_score"], 4))
REC, DGH, BOTH, WEAK = "schema_pool", "lean_dgh4", "recipe_dgh", "f_weak"
real_exploits = [n for n, r in rows.items() if n.startswith(("e1_", "e2_", "e3_", "cert_close", "inst_", "cert_one", "instance_specific"))
                 or ("real" in r["kind"] and n.startswith("R3"))]
tiny = [n for n in rows if n.startswith("R3 A1") or n.startswith("R3 A2")]
def check(w):
    s = {n: score(comp(n), w) for n in rows}
    up = lambda n: s[n] - 0.2
    ratio = up(DGH) / max(up(REC), 1e-9)
    ok = {
        "P1": 0.5 <= ratio <= 2.0,
        "P2": s[BOTH] > max(s[REC], s[DGH]),
        "P3": all(s[n] < min(s[REC], s[DGH]) for n in real_exploits),
        "P4": all(s[n] <= s[WEAK] for n in tiny),
        "P5": all(s[WEAK] > s[n] for n in real_exploits),
    }
    return ok, s, ratio, up(REC) / 0.0594
grid = [x / 100 for x in range(0, 61, 2)]
best = []
for tr, ta, tl, de in itertools.product(grid, grid, [0.06, 0.10, 0.14, 0.18], [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]):
    for cl in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15):
        ge = 0.07
        if abs(tr + ta + tl + de + cl + ge - 1.0) > 1e-9:
            continue
        w = dict(tr=tr, ta=ta, ge=ge, tl=tl, de=de, cl=cl)
        ok, s, ratio, c5 = check(w)
        if all(ok.values()):
            dist = sum(abs(w[k] - SHIP[k]) for k in w)
            best.append((dist, w, ratio, c5, s))
best.sort(key=lambda x: x[0])
import sys
MIN_CL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
best = [b for b in best if b[1]["cl"] >= MIN_CL - 1e-9]
ok0, s0, r0, c50 = check(SHIP)
out = {"shipped": {"weights": SHIP, "criteria": ok0, "ratio_dgh_over_recipe": r0, "C5": c50,
                   "scores": {n: round(s0[n], 4) for n in (REC, DGH, BOTH, WEAK)}},
       "n_feasible": len(best),
       "closest": [{"weights": w, "L1_change": round(d, 3), "ratio": round(r, 3), "C5": round(c5, 3),
                    "scores": {n: round(s[n], 4) for n in (REC, DGH, BOTH, WEAK)},
                    "max_real_exploit": round(max(s[n] for n in real_exploits), 4)} for d, w, r, c5, s in best[:12]],
       "max_C5_feasible": round(max((b[3] for b in best), default=0), 3),
       "real_exploits": real_exploits, "tiny": tiny}
json.dump(out, open(os.path.join(HERE, f"sweep_mincl{MIN_CL}.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ("shipped", "n_feasible", "max_C5_feasible")}, indent=1))
for c in out["closest"][:8]:
    print(c)

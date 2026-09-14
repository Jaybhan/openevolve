"""Convert the seven pinched slice-feasibility verdicts into the five open
F_9 values, by the PROVEN case analysis (report.md §5):

  val-35 config, cols <= 24  <=>  (k5=11, quads>=13)                 [Q-A]
  val-35, cols <= 25  <=>  Q-A or (10,>=15) [Q-B] or (8,>=16,k6=1)   [Q-C]
  val-35, cols <= 26  <=>  above or (9,>=17) [Q-D] or (7,>=18,k6=1)  [Q-E]
                                 or (5,>=19,k6=2)                    [Q-F]
  val-36, cols <= 27  <=>  (9,>=18)                                  [Q-G]
  val-38, cols <= 33  <=>  (5,>=28)                                  [Q-H]

(Each equivalence: identity 6val = 2c + W - X, W <= mR = 162, X >= 0 pins
c, X, W and hence the exact profile; machine-checked in report §5.)

Emits frontier_m9_cases.jsonl records for c = 24, 25, 26, 27, 33 with
status PROVEN(cases) when all governing questions are decided, using the
best known config as the block list. Prints the z-verdicts.
"""
import json
import os
from collections import Counter
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))

Q = {}  # (k5, quad_lb, k6) -> "FEASIBLE"/"INFEASIBLE"/"UNRESOLVED", blocks
with open(os.path.join(HERE, "s9_slices.jsonl")) as f:
    for line in f:
        r = json.loads(line)
        if "quad_lb" not in r:
            continue
        key = (r["k5"], r["quad_lb"], r.get("k6", 0))
        if r["status"] == "INFEASIBLE":
            Q[key] = ("INFEASIBLE", None)
        elif r["S"] is not None and r["S"] >= r["quad_lb"]:
            Q[key] = ("FEASIBLE", r["blocks"])
        else:
            Q.setdefault(key, ("UNRESOLVED", None))

QA = Q.get((11, 13, 0), ("UNRESOLVED", None))
QB = Q.get((10, 15, 0), ("UNRESOLVED", None))
QC = Q.get((8, 16, 1), ("UNRESOLVED", None))
QD = Q.get((9, 17, 0), ("UNRESOLVED", None))
QE = Q.get((7, 18, 1), ("UNRESOLVED", None))
QF = Q.get((5, 19, 2), ("UNRESOLVED", None))
QG = Q.get((9, 18, 0), ("UNRESOLVED", None))
QH = Q.get((5, 28, 0), ("UNRESOLVED", None))
if QD[0] == "INFEASIBLE" and QG[0] == "UNRESOLVED":
    QG = ("INFEASIBLE", None)          # quads>=18 subsumed by quads>=17
for name, q in [("A(11,13)", QA), ("B(10,15)", QB), ("C(8,16;k6=1)", QC),
                ("D(9,17)", QD), ("E(7,18;k6=1)", QE), ("F(5,19;k6=2)", QF),
                ("G(9,18)", QG), ("H(5,28)", QH)]:
    print(f"  Q-{name}: {q[0]}")


def verify_blocks(blocks):
    cov = Counter()
    for b in blocks:
        for t in combinations(sorted(b), 3):
            cov[t] += 1
    assert all(v <= 2 for v in cov.values()), "illegal certificate!"
    return True


def load_lb_blocks(c):
    """Best known val-config at <= c cols (heavy parts of bank witnesses /
    dead-run incumbents), used as the frontier point's block list."""
    cand = []
    for fn in ("frontier_m9_open.jsonl", "frontier_m9_open33.jsonl",
               "frontier_m9.jsonl"):
        p = os.path.join(HERE, fn)
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("val") is not None and r["c"] <= c:
                        cand.append((r["val"], r["blocks"]))
    base = os.path.abspath(os.path.join(HERE, "..", "witnesses"))
    for n in range(9, c + 1):
        p = os.path.join(base, f"w_9x{n}.json")
        if os.path.exists(p):
            w = json.load(open(p))
            heavy = [b for b in w["blocks"] if len(b) >= 4]
            cand.append((sum(len(b) - 3 for b in heavy), heavy))
    return max(cand, key=lambda t: t[0])


verdicts = {}
# c=24
if QA[0] != "UNRESOLVED":
    verdicts[24] = 35 if QA[0] == "FEASIBLE" else 34
# c=25
g25 = [QA, QB, QC]
if all(q[0] != "UNRESOLVED" for q in g25):
    verdicts[25] = 35 if any(q[0] == "FEASIBLE" for q in g25) else 34
# c=26
g26 = [QA, QB, QC, QD, QE, QF]
if all(q[0] != "UNRESOLVED" for q in g26):
    verdicts[26] = 35 if any(q[0] == "FEASIBLE" for q in g26) else 34
# c=27
if QG[0] != "UNRESOLVED":
    verdicts[27] = 36 if QG[0] == "FEASIBLE" else 35
# c=33
if QH[0] != "UNRESOLVED":
    verdicts[33] = 38 if QH[0] == "FEASIBLE" else 37

out = os.path.join(HERE, "frontier_m9_cases.jsonl")
with open(out, "w") as f:
    for c, val in sorted(verdicts.items()):
        # certificate blocks: feasible answer's blocks, else best LB config
        fea = {24: QA, 25: QB, 26: QD, 27: QG, 33: QH}
        blocks = None
        for q in ([fea[c]] + [QA, QB, QC, QD, QE, QF, QG, QH]):
            if q[0] == "FEASIBLE" and q[1]:
                heavy = [b for b in q[1] if len(b) >= 4]
                v = sum(len(b) - 3 for b in heavy)
                if v == val and len(heavy) <= c:
                    blocks = heavy
                    break
        if blocks is None:
            v, blocks = load_lb_blocks(c)
            assert v == val, (c, v, val)
        verify_blocks(blocks)
        prof = Counter(len(b) for b in blocks)
        rec = {"m": 9, "c": c, "val": val, "status": "PROVEN(cases)",
               "slots_min": sum({4: 4, 5: 10, 6: 20}[len(b)] for b in blocks),
               "profile": {str(w): k for w, k in sorted(prof.items())},
               "blocks": [list(b) for b in blocks]}
        f.write(json.dumps(rec) + "\n")
        print(f"F_9({c}) = {val}  [PROVEN via case analysis + slices]  "
              f"=> z(9,{c}) = {3 * c + val}")
print(f"\n{len(verdicts)}/5 cells decided; records in {out}")

"""E26: dispersion + two-sided permutation tests (difference of means vs the V0/S0 ladder baseline,
20,000 label shuffles, seed 26) for the key per-run metrics of results/aggregate[_<prefix>].json."""
import json
import os
import random
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.environ.get("E26_PREFIX", "")
suf = ("_" + PREFIX.rstrip("_")) if PREFIX else ""
A = json.load(open(os.path.join(HERE, "results", f"aggregate{suf}.json")))["per_run"]
KEYS = ["best", "map_cells_D", "map_cells_RD", "parent_share_D_late", "parent_share_RD_late", "parent_share_R_late",
        "final_frac_D", "final_frac_RD", "map_cells", "distinct_genomes", "entropy", "archive_D", "n_with_U"]
BASE = "v0_ladder"


def perm_p(a, b, n=20000, seed=26):
    rng = random.Random(seed)
    obs = abs(st.mean(a) - st.mean(b))
    pool = a + b
    hit = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[:len(a)]) - st.mean(pool[len(a):])) >= obs - 1e-12:
            hit += 1
    return (hit + 1) / (n + 1)


out, L = {}, [f"# E26 stats {PREFIX or 'primary'}: mean ± sd over seeds; p = permutation test vs {BASE}", ""]
L.append("| metric | " + " | ".join(A) + " |")
L.append("|---|" + "---|" * len(A))
for k in KEYS:
    row = []
    base = [float(r[k] or 0) for r in A[BASE]]
    for c, rows in A.items():
        xs = [float(r[k] or 0) for r in rows]
        if not xs:
            row.append("n/a")
            continue
        m, sd = st.mean(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)
        p = perm_p(xs, base) if c != BASE else None
        out.setdefault(k, {})[c] = {"mean": m, "sd": sd, "p_vs_v0": p}
        row.append(f"{m:.3f} ± {sd:.3f}" + (f" (p={p:.3f})" if p is not None else ""))
    L.append(f"| {k} | " + " | ".join(row) + " |")
json.dump(out, open(os.path.join(HERE, "results", f"stats{suf}.json"), "w"), indent=1)
open(os.path.join(HERE, "results", f"stats{suf}.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

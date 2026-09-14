"""Export deliverable frontier CSVs: deepband/frontier_m{6,7,8,9}.csv.

Columns: c, F (max val, <=c cols), status, cols_used, slots_min, W, X,
profile, F_w5 (weights<=5 restriction), F_w6 (<=6), UBcols (proven bound),
FLP (arithmetic shell), delta = FLP - F, and the frontier config's blocks
are kept in frontier_points_m{m}.json.
"""
import csv
import json
import os
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
from verify_deepband import load_frontier, flp_shell  # noqa: E402


def load_tagged(m, tag):
    p = os.path.join(HERE, f"frontier_m{m}{tag}.jsonl")
    out = {}
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("val") is not None:
                    out[r["c"]] = r
    return out


def main():
    for m in (6, 7, 8, 9):
        R = (2 * comb(m - 1, 2)) // 3
        B = 2 * comb(m, 3)
        J = (m * R) // 4 - (2 if (m % 4 == 3 and m % 3 != 0) else 0)
        fr = load_frontier(m)
        if not fr:
            continue
        w5 = load_tagged(m, "_w5")
        w6 = load_tagged(m, "_w6")
        rows = []
        points = {}
        for c in sorted(fr):
            r = fr[c]
            prof = {int(w): k for w, k in r["profile"].items()}
            cols_used = sum(prof.values())
            W = sum(w * (w - 3) * k for w, k in prof.items())
            X = sum((w - 4) * (w - 5) * k for w, k in prof.items())
            ubc = min((m - 3) * c, (2 * c + m * R) // 6, J)
            flp = flp_shell(m, c, B, m * R, J)
            rows.append({
                "c": c, "F": r["val"], "status": r["status"],
                "cols_used": cols_used, "slots_min": r["slots_min"],
                "W": W, "X": X,
                "profile": "+".join(f"{k}x{w}" for w, k in sorted(prof.items(),
                                                                 reverse=True)),
                "F_w5": w5.get(c, {}).get("val", ""),
                "F_w6": w6.get(c, {}).get("val", ""),
                "UBcols": ubc, "FLP": flp, "delta": flp - r["val"],
            })
            points[c] = {"val": r["val"], "status": r["status"],
                         "blocks": r["blocks"]}
        with open(os.path.join(HERE, f"frontier_m{m}.csv"), "w",
                  newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        with open(os.path.join(HERE, f"frontier_points_m{m}.json"), "w") as f:
            json.dump(points, f)
        print(f"m={m}: {len(rows)} frontier points exported "
              f"(PROVEN: {sum(1 for r in rows if r['status'].startswith('PROVEN'))})")


if __name__ == "__main__":
    main()

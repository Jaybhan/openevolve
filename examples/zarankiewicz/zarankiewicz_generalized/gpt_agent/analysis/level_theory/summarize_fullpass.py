"""Final per-cell status classification for the full-table ledger pass.

Statuses:
  MATCH            z_hi == z_true (UB closes under proven cuts + computed
                   slices; LB = stored witness realizes the formula value)
  RESIDUAL-BAND    z_hi > z_true, claiming sigs are in the band family
                   (<= 3 hexads, <= 1 block of weight >= 7): slice
                   computation pending/timeout — finite MILPs, pipeline
                   exists
  RESIDUAL-CORNER  z_hi > z_true, every refutation route needs corner
                   slices (many blocks of weight >= 6 at m >= 10): the
                   cap/ovoid-geometry frontier (Observation 3 / Theorem 12
                   limit / diagonal_limit.md) — beyond current compute

Writes fullpass_final.csv and prints the counts + per-row breakdown.
"""
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def band_sig(sig):
    return sig[1] <= 3 and sum(sig[2:]) <= 1 and sum(sig[1:]) <= 4


def main():
    rows = [line.strip().split(",")
            for line in open(os.path.join(HERE, "fullpass_status.csv")).readlines()[1:]]
    out = open(os.path.join(HERE, "fullpass_final.csv"), "w")
    out.write("m,n,z_true,z_hi,status,residual,claiming_first\n")
    cnt = Counter()
    perrow = {}
    for r in rows:
        m, n, zt, zhi = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        st = r[5]
        cl = r[6] if len(r) > 6 else ""
        if st == "MATCH":
            lab, res = "MATCH", 0
        else:
            res = zhi - zt
            sigs = []
            for part in cl.split("|"):
                if ">=" in part:
                    sigs.append(tuple(int(t) for t in
                                      part.split(">=")[0].split(";")))
            lab = ("RESIDUAL-BAND" if any(band_sig(s) for s in sigs) or not sigs
                   else "RESIDUAL-CORNER")
        cnt[lab] += 1
        perrow.setdefault(m, Counter())[lab] += 1
        out.write(f"{m},{n},{zt},{zhi},{lab},{res},{cl.split('|')[0]}\n")
    out.close()
    total = sum(cnt.values())
    print(f"TOTAL {total} cells: " +
          ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    print("per row:")
    for m in sorted(perrow):
        c = perrow[m]
        print(f"  m={m:2d}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()

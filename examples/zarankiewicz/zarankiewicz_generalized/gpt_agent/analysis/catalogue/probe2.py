"""Decode the remaining unrecognized layers: m=9 mid-band pentads,
m=10 hexads/pentads at 14/15/47, and the big design_prover packings."""
import sys
from collections import Counter
from itertools import combinations

import catlib as C
import gens as G
from audit import leave_of, is_doubled_pentagon, is_doubled_parallel_class


def show_layer(m, bl, label):
    print(f"-- {label}: {len(bl)} blocks on m={m}")
    for b in sorted(bl):
        print("   ", "".join(str(x) for x in b))
    print("    deg:", C.point_deg(bl, m))
    print("    inter:", dict(sorted(C.inter_dist(bl).items())))
    pd = C.pair_deg(bl, m)
    print("    pair-deg distribution:", dict(sorted(
        Counter(pd.values()).items())))


def main():
    ws = {w["file"]: w for w in C.load_banks()}

    # m=9 12-pentad layer (w_9x22)
    w = ws["w_9x22.json"]
    lay = C.layers(9, w["blocks"])
    show_layer(9, lay[5], "w_9x22 pentads")
    show_layer(9, lay[4], "w_9x22 quads")

    # w_9x27 cone-derived quads
    w = ws["w_9x27.json"]
    lay = C.layers(9, w["blocks"])
    show_layer(9, lay[5], "w_9x27 pentads")

    # w_9x29 pentads
    w = ws["w_9x29.json"]
    lay = C.layers(9, w["blocks"])
    show_layer(9, lay[5], "w_9x29 pentads")
    w = ws["w_9x31.json"]
    show_layer(9, C.layers(9, w["blocks"])[5], "w_9x31 pentads")
    w = ws["w_9x34.json"]
    show_layer(9, C.layers(9, w["blocks"])[5], "w_9x34 pentads")
    w = ws["w_9x36.json"]
    show_layer(9, C.layers(9, w["blocks"])[5], "w_9x36 pentads")

    # m=10 hexads at 14/15; pentads at 15/47
    w = ws["smoke_10x14_77_witness.json"]
    lay = C.layers(10, w["blocks"])
    show_layer(10, lay[6], "10x14 hexads")
    show_layer(10, lay[5], "10x14 pentads")
    w = ws["T2_10x15_81_cadical_witness.json"]
    lay = C.layers(10, w["blocks"])
    show_layer(10, lay[6], "10x15 hexads")
    show_layer(10, lay[5], "10x15 pentads")
    w = ws["w_10x47.json"]
    lay = C.layers(10, w["blocks"])
    show_layer(10, lay[5], "10x47 pentads")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    main()

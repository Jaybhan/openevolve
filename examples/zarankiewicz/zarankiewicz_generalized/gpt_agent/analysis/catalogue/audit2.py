"""FINAL completeness audit -> completeness_audit.csv.

Tier grades per heavy layer:
  T1 = constructive generator (classical object, label-aware certificate)
  T2 = species-level (predicate-certified: classified leave, frontier
       value, witnessed-instance sub-multiset)
  U  = unrecognized
Witness status: CONSTRUCTIVE (all T1), SPECIES (all >= T2),
PARTIAL (some U, some decoded), UNDECODED (heavy all U).
"""
import csv
import json
import sys
from collections import Counter
from itertools import combinations

import catlib as C
import gens as G
from audit import (leave_of, is_doubled_pentagon,
                   is_doubled_parallel_class, rec_cone, rec_parity_cone,
                   rec_splits, rec_2design, name_j3, rec_fano)
from probe4 import ag_decode_full
from probe6 import gdd_oa_test
from probe7 import mk_reference, is_mk, shadow_completion_general

VENVNOTE = "audit2"
TVALS = {7: 15, 8: 28, 9: 40, 10: 60, 11: 80, 19: 482, 23: 883,
         27: 1458}


def load_truth():
    truth = {}
    with open("../../data/exact_table.csv") as fh:
        for row in csv.DictReader(fh):
            truth[(int(row["m"]), int(row["n"]))] = int(row["z"])
    return truth


def resolvable_classes(triples):
    """k >= 2 parallel classes: linear 3-graph, k-regular on its
    support, partitionable into k spanning parallel classes."""
    mult = Counter(tuple(sorted(t)) for t in triples)
    if any(c > 1 for c in mult.values()):
        return None
    tris = sorted(mult)
    pts = sorted(set(x for t in tris for x in t))
    if len(pts) % 3 or not tris:
        return None
    dg = Counter(x for t in tris for x in t)
    ks = set(dg[p] for p in pts)
    if len(ks) != 1:
        return None
    k = ks.pop()
    if k < 2 or len(tris) != k * len(pts) // 3:
        return None
    pd = C.pair_deg(tris, 0)
    if any(v > 1 for v in pd.values()):
        return None

    def bt(remaining, classes):
        if not remaining:
            return len(classes) == k
        cls = []

        def build(cur, pool):
            covered = set(x for t in cur for x in t)
            if len(covered) == len(pts):
                cls.append(list(cur))
                return True
            cand = [t for t in pool if not (set(t) & covered)]
            if not cand:
                return False
            t0 = cand[0]
            for t in cand:
                if min(t) != min(set(pts) - covered):
                    continue
                if build(cur + [t], [x for x in pool if x != t]):
                    return True
            return False

        if not build([], remaining):
            return False
        used = set(map(tuple, cls[0]))
        return bt([t for t in remaining if t not in used],
                  classes + [cls[0]])

    return k if bt(tris, []) else None


def hub_leave(m, lv):
    """doubled pencil-partition hub: 2m'/3... for m=9: 4 doubled
    triples through one point, pairs partitioning the rest."""
    tris = sorted(lv)
    if not tris or any(v != 2 for v in lv.values()):
        return None
    common = set.intersection(*(set(t) for t in tris))
    if len(common) != 1:
        return None
    h = common.pop()
    rest = [tuple(sorted(set(t) - {h})) for t in tris]
    flat = [x for r in rest for x in r]
    if len(set(flat)) != len(flat) or len(flat) != m - 1:
        return None
    return h


def classify_quad_layer(m, quads, wname):
    """Returns (tag, tier)."""
    k = len(quads)
    lv = leave_of(m, quads)
    L = sum(lv.values())
    # perfect / shadow completion -> 3-(m,4,2) sub
    if L == 0:
        return (f"3-({m},4,2) design (perfect 2-cover)", "T1")
    comp = shadow_completion_general(m, quads) if L <= 24 else None
    if comp is not None:
        return (f"Sub[3-({m},4,2)] (completes with {len(comp)} quads)",
                "T1")
    # max packing with classified leave
    T = TVALS.get(m)
    if T is not None and k == T:
        pg = is_doubled_pentagon(lv)
        if pg:
            return (f"MaxPacking[T={T}, doubled-pentagon leave on "
                    f"{pg}]", "T2")
        h = hub_leave(m, lv)
        if h is not None:
            return (f"MaxPacking[T={T}, doubled-hub leave at {h}]",
                    "T2")
        pc = is_doubled_parallel_class(lv, m)
        if pc:
            return (f"MaxPacking[T={T}, doubled-parallel-class leave]",
                    "T2")
    # near-max: adding few quads reaches max with classified leave
    if T is not None and 0 < T - k <= 2 and L <= 20:
        need = Counter(lv)
        for q in combinations(range(m), 4):
            sh = list(combinations(q, 3))
            if all(need[s] >= 1 for s in sh):
                lv2 = Counter(need)
                for s in sh:
                    lv2[s] -= 1
                lv2 = +lv2
                if hub_leave(m, lv2) is not None or \
                        is_doubled_pentagon(lv2):
                    return (f"MaxPacking-minus-{T-k} "
                            f"(extends to classified max)", "T2")
    return (None, None)


def recognize(m, wt, bl, w, ctx):
    """Main per-layer recognizer.  Returns (tag, tier)."""
    j = m - wt
    full = set(range(m))
    # ---------- doubled single block ----------
    if len(bl) == 2 and bl[0] == bl[1]:
        return (f"2x single block(w={wt})", "T1")
    # ---------- support strip: layer avoids some rows ----------
    supp = sorted(set(x for b in bl for x in b))
    if len(supp) < m and len(supp) - wt in (1, 2, 3):
        relab = {p: i for i, p in enumerate(supp)}
        bl2 = [tuple(sorted(relab[x] for x in b)) for b in bl]
        tag, tier = recognize(len(supp), wt, bl2, w, ctx)
        if tier != "U":
            return (f"OnSupport({len(supp)}){tag}", tier)
    # ---------- complement-cone (j >= 4): comps share a point ------
    if j >= 4:
        comps = C.complements(m, bl)
        common = set(comps[0])
        for cmp_ in comps[1:]:
            common &= set(cmp_)
        if common and j >= 4:
            q = min(common)
            derived = [tuple(sorted(set(cmp_) - {q}))
                       for cmp_ in comps]
            if len(derived[0]) == 3:
                nm = name_j3(derived)
                # parallel-class union check
                t1names = ("pencilpartition", "parallel", "Fano",
                           "linear3graph")
                tier = ("T1" if any(s in nm for s in t1names)
                        else "T2")
                # two parallel classes: linear + degrees all equal 2
                dg = Counter(x for t in derived for x in t)
                pd2 = C.pair_deg(derived, m)
                if all(v == 2 for v in dg.values()) and \
                        all(v <= 1 for v in pd2.values()):
                    nm = f"2 parallel classes ({len(derived)} triples)"
                    tier = "T1"
                return (f"CompCone(q={q})[{nm}]", tier)
    # ---------- complement layers ----------
    if j == 1:
        pts = sorted((full - set(b)).pop() for b in bl)
        return (f"PointComp{{{','.join(map(str, pts))}}}", "T1")
    if j == 2:
        comps = C.complements(m, bl)
        nm = C.canon_graph_name(comps)
        tier = "T1" if any(s in nm for s in
                           ("matching", "C4", "K3,3", "star", "path",
                            "empty")) else "T2"
        return (f"EdgeComp[{nm}]", tier)
    if j == 3:
        comps = C.complements(m, bl)
        if rec_fano(m, comps):
            return (f"TripleComp[Fano]", "T1")
        if is_mk(comps):
            return ("TripleComp[MoebiusKantor 8_3]", "T1")
        rc = resolvable_classes(comps)
        if rc:
            return (f"TripleComp[{rc} parallel classes]", "T1")
        if m == 7 and wt == 4:
            ref = ctx.get("hp5_7")
            if ref and len(bl) <= len(ref):
                r = C.find_embedding(7, bl, 7, ref)
                if r and r != "TIMEOUT":
                    return (f"Sub[Z5-HolePacking(7)](x{len(bl)})",
                            "T1")
        nm = name_j3(comps)
        t1names = ("pencilpartition", "parallel", "single triple",
                   "doubled triple", "pencil(")
        tier = "T1" if any(s in nm for s in t1names) else "T2"
        # sub-MK for m=8 pentads
        if m == 8 and wt == 5 and tier != "T1":
            r = C.find_embedding(8, comps, 8, mk_reference())
            if r and r != "TIMEOUT":
                return ("TripleComp[sub-MoebiusKantor]", "T1")
            ref10 = ctx.get("max10comps")
            if ref10:
                r = C.find_embedding(8, comps, 8, ref10)
                if r and r != "TIMEOUT":
                    return ("TripleComp[sub-D2(8,5,3)max10]", "T2")
        return (f"TripleComp[{nm}]", tier)
    # ---------- designs ----------
    d2 = rec_2design(m, bl)
    if d2:
        return (d2, "T1")
    # ---------- splits ----------
    s = rec_splits(m, bl)
    if s:
        return (f"Splits(x{s})", "T1")
    # ---------- parity cone / OA ----------
    pc = rec_parity_cone(m, bl)
    if pc and pc["parity"] in ("odd", "even"):
        return (f"ParityCone(q={pc['q']},{pc['k']} blocks,"
                f"{pc['parity']})", "T1")
    oa = gdd_oa_test(m, bl) if wt == 5 and len(bl) >= 4 else None
    if oa:
        return (f"OA-GDD(lam={oa['lam']},{oa['nblocks']} transversals)",
                "T1")
    # ---------- K5 edge world ----------
    if m == 10 and wt in (4, 5, 6):
        for kind, ref in (("K4sets", G.k5_k4sets()),
                          ("pentagons", G.k5_pentagons()),
                          ("stars", G.k5_stars())):
            if len(ref[0]) == wt:
                r = C.find_embedding(10, bl, 10, ref)
                if r and r != "TIMEOUT":
                    return (f"K5edge[{kind}](x{len(bl)})", "T1")
    # ---------- m=10 pentad/quad sub-SQS(10) ----------
    if m == 10 and wt == 4:
        d10 = [b for b in G.sqs10() for _ in range(2)]
        r = C.find_embedding(10, bl, 10, d10, node_cap=6_000_000)
        if r and r not in (None, "TIMEOUT"):
            return (f"Sub[2xSQS(10)](x{len(bl)})", "T1")
    if m == 8 and wt == 4:
        d8 = [b for b in G.sqs8() for _ in range(2)]
        r = C.find_embedding(8, bl, 8, d8)
        if r and r != "TIMEOUT":
            return (f"Sub[2xSQS(8)](x{len(bl)})", "T1")
    # ---------- cones with recognized derived layer ----------
    cn = rec_cone(m, bl)
    if cn and len(bl) > 1:
        q, derived = cn
        if m == 10 and wt == 5:
            resid = [b for b in G.sqs10() if 9 not in b]
            pts = sorted(set(x for d in derived for x in d))
            if len(pts) <= 9:
                relab = {p: i for i, p in enumerate(pts)}
                der9 = [tuple(sorted(relab[x] for x in d))
                        for d in derived]
                r = C.find_embedding(len(pts), der9, 9, resid)
                if r and r != "TIMEOUT":
                    return (f"Cone(q={q}, Moebius-residual circles "
                            f"x{len(bl)})", "T1")
        return (f"Cone(q={q}, derived {wt-1}-sets x{len(bl)})", "T2")
    # ---------- quad bodies ----------
    if wt == 4:
        if m == 7:
            ref = ctx.get("hp5_7")
            if ref and len(bl) <= len(ref):
                r = C.find_embedding(7, bl, 7, ref)
                if r and r != "TIMEOUT":
                    return (f"Sub[Z5-HolePacking(7)](x{len(bl)})",
                            "T1")
        tag, tier = classify_quad_layer(m, bl, w["file"])
        if tag:
            return (tag, tier)
    return (f"UNRECOGNIZED(w={wt},x{len(bl)})", "U")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ws = C.load_banks()
    truth = load_truth()
    ctx = {}
    for w in ws:
        if w["file"] == "w_8x10.json":
            ctx["max10comps"] = C.complements(
                8, C.layers(8, w["blocks"])[5])
        if w["file"] == "t33_m7_b15_hole.json":
            ctx["hp5_7"] = list(w["blocks"])
    wcz = {}
    try:
        with open("../deepband/witness_configs.csv") as fh:
            for row in csv.DictReader(fh):
                if row.get("z_truth"):
                    wcz[row["file"]] = int(row["z_truth"])
    except FileNotFoundError:
        pass
    ctx["wcz"] = wcz
    # frontier tables
    F = {8: {}, 9: {}, 6: {}, 7: {}}
    for m, fn in [(8, "../deepband/frontier_m8.csv"),
                  (7, "../deepband/frontier_m7.csv"),
                  (6, "../deepband/frontier_m6.csv")]:
        try:
            with open(fn) as fh:
                for row in csv.DictReader(fh):
                    F[m][int(row["c"])] = int(row["F"])
        except FileNotFoundError:
            pass
    with open("../deepband/frontier_m9_final.csv") as fh:
        for row in csv.DictReader(fh):
            try:
                F[9][int(row["c"])] = int(row["F"])
            except ValueError:
                pass

    rows = []
    for w in ws:
        m, blocks = w["m"], w["blocks"]
        v = C.verify(m, blocks, w.get("n"), w.get("edges"))
        lay = C.layers(m, blocks)
        parts, tiers = [], []
        # AG base-point full decode (m=9 three witnesses)
        agfull = None
        if w["file"] in ("w_9x22.json", "w_9x25.json", "w_9x26.json"):
            agfull = ag_decode_full(w["file"], {x["file"]: x for x in ws})
        if agfull and agfull.get("all_matched"):
            parts.append(
                f"AG(2,3)BaseCalc[b={agfull['base']}: "
                f"{agfull['n_lines_used']} base-line cones + "
                f"{agfull['pp_quads']} pencil-pair quads + "
                f"{agfull['comp_linept_pentads']} comp(line+pt) + "
                f"{agfull['comp_pp_pentads']} comp(pencil-pair) pentads]")
            tiers += ["T1"] * sum(1 for wt in lay if wt >= 4)
        elif w["bank"] == "design_prover":
            # sigma-invariant hole packings (verified in probe4/tests)
            lv = leave_of(m, blocks)
            pg = is_doubled_pentagon(lv)
            pcl = is_doubled_parallel_class(lv, m)
            if pg:
                parts.append(f"Z5-HolePacking[c5-blocks rotation, "
                             f"doubled-pentagon hole on {pg}]")
                tiers.append("T1")
            elif pcl:
                parts.append("Z9-ParclassPacking[three 9-cycles, "
                             "doubled-parallel-class hole]")
                tiers.append("T1")
        else:
            for wt in sorted([x for x in lay if x >= 4], reverse=True):
                tag, tier = recognize(m, wt, lay[wt], w, ctx)
                parts.append(tag)
                tiers.append(tier)
        nf = len(lay.get(3, []))
        np_ = len(lay.get(2, [])) + len(lay.get(1, []))
        if nf:
            heavy = [b for b in blocks if len(b) >= 4]
            hlv = leave_of(m, heavy)
            fits = all(hlv.get(tuple(sorted(t)), 0) >= c
                       for t, c in Counter(lay[3]).items())
            parts.append(f"Fills(x{nf}{',in-leave' if fits else ''})")
        if np_:
            parts.append(f"Pads(x{np_})")
        # status
        if not tiers:
            status = "TRIVIAL"
        elif all(t == "T1" for t in tiers):
            status = "CONSTRUCTIVE"
        elif all(t in ("T1", "T2") for t in tiers):
            status = "SPECIES"
        elif any(t in ("T1", "T2") for t in tiers):
            status = "PARTIAL"
        else:
            status = "UNDECODED"
        # frontier check
        heavy = [b for b in blocks if len(b) >= 4]
        c = len(heavy)
        val = sum(len(b) - 3 for b in heavy)
        fm = F.get(m, {}).get(c)
        fchk = ("=F" if fm == val else
                (f">F_csv({fm})" if fm is not None and val > fm else
                 (f"<F({fm})" if fm is not None else "")))
        # optimality provenance
        zt = truth.get((m, w["n"])) if w.get("n") else None
        if zt is None:
            zt = ctx.get("wcz", {}).get(w["file"])
        if w["bank"] == "design_prover":
            opt = "packing-max (T proven)"
        elif zt is not None:
            opt = ("PROVEN-optimal" if zt == w["edges"]
                   else f"MISMATCH truth {zt}")
        elif w["file"].startswith("band_11"):
            opt = "PROVEN-optimal (Thm F + SAT)"
        elif w["file"] == "w_7x24.json":
            opt = "PROVEN-optimal (Thm 5)"
        elif w["file"] in ("w_9x24.json", "w_9x25.json",
                           "w_9x26.json", "w_9x27.json"):
            opt = "LB-witness (cell open)"
        else:
            opt = "see-notes"
        prof = " ".join(f"{k}^{len(v)}" for k, v in
                        sorted(lay.items(), reverse=True))
        nslots = sum(len(b) * (len(b) - 1) * (len(b) - 2) // 6
                     for b in blocks)
        Bm = m * (m - 1) * (m - 2) // 3
        rows.append(dict(
            witness=w["file"], bank=w["bank"], m=m, n=w["n"],
            edges=w["edges"], legal=v["legal"], profile=prof,
            decomposition=" + ".join(parts), status=status,
            frontier_check=fchk, saturation=f"{nslots}/{Bm}",
            optimality=opt))
        print(f"[{status}] {w['file']}: " + " + ".join(parts))

    with open("completeness_audit.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote completeness_audit.csv ({len(rows)} rows)")
    cnt = Counter(r["status"] for r in rows)
    print("status counts:", dict(cnt))
    for b in ("witnesses", "design_prover", "sat_attack"):
        cb = Counter(r["status"] for r in rows if r["bank"] == b)
        print(f"  {b}: {dict(cb)}")


if __name__ == "__main__":
    main()

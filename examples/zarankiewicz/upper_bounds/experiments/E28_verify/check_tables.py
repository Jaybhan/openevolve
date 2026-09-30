"""E28: cache table integrity, current cache/ vs experiments/E27_integration/cache_before/ (the pre-batch-3 copy).

Read-only.  Per table: record count and (rows, cols) order identical; baseline_lean_mask present in both or neither,
same length (= #records) and identical; baseline_lean_kill flags identical; no status regression (unsat/sat never become
unknown, sat never changes); every exact unsat label unchanged (d, conflicts) unless it was censored before; loads via
casetable.table_from_json and casetable.load_table.  Also lists tables that exist on only one side.
"""
import glob, json, os, sys

UB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, UB)
from zar_ub.casetable import table_from_json, load_table  # noqa: E402
from zar_ub.known import Instance  # noqa: E402

CUR = os.path.join(UB, "cache")
BEF = os.path.join(UB, "experiments", "E27_integration", "cache_before")


def main():
    cur = {os.path.basename(p) for p in glob.glob(os.path.join(CUR, "case_table_*.json"))}
    bef = {os.path.basename(p) for p in glob.glob(os.path.join(BEF, "case_table_*.json"))}
    problems, lines = [], []
    lines.append(f"tables now {len(cur)}, before {len(bef)}; only now: {sorted(cur - bef)}; only before: {sorted(bef - cur)}")
    if bef - cur:
        problems.append(f"tables lost: {sorted(bef - cur)}")
    tot = {"records": 0, "mask_tables": 0, "censored_to_exact": 0, "lb_raised": 0, "relabelled": 0, "changed_tables": 0}
    for name in sorted(cur):
        a = json.load(open(os.path.join(CUR, name)))
        ok_load = True
        try:
            tab = table_from_json(a, os.path.join(CUR, name))
            ii = a["inst"]
            inst = Instance(ii["m"], ii["n"], ii["s"], ii["t"], ii["w"])
            lt = load_table(inst, path=os.path.join(CUR, name))
            if lt is None or len(lt.records) != len(a["records"]):
                ok_load = False
        except Exception as e:  # noqa: BLE001
            ok_load = False
            problems.append(f"{name}: load error {e!r}")
        if not ok_load:
            problems.append(f"{name}: does not load")
        ra = a["records"]
        tot["records"] += len(ra)
        ma = a.get("baseline_lean_mask")
        if ma is not None:
            tot["mask_tables"] += 1
            if len(ma) != len(ra):
                problems.append(f"{name}: mask length {len(ma)} != records {len(ra)}")
        if name not in bef:
            lines.append(f"{name:45s} NEW  n={len(ra)} mask={'-' if ma is None else len(ma)} load={ok_load}")
            continue
        b = json.load(open(os.path.join(BEF, name)))
        rb = b["records"]
        mb = b.get("baseline_lean_mask")
        if len(ra) != len(rb):
            problems.append(f"{name}: records {len(rb)} -> {len(ra)}")
        if [(r["rows"], r["cols"]) for r in ra] != [(r["rows"], r["cols"]) for r in rb]:
            problems.append(f"{name}: record order/keys changed")
        if (ma is None) != (mb is None) or (ma is not None and list(ma) != list(mb)):
            problems.append(f"{name}: baseline_lean_mask changed (before {None if mb is None else len(mb)}, now {None if ma is None else len(ma)})")
        n_c2e = n_lb = n_rel = 0
        for x, y in zip(rb, ra):
            if bool(x.get("baseline_lean_kill")) != bool(y.get("baseline_lean_kill")):
                problems.append(f"{name}: baseline_lean_kill flag changed at {x['rows']}/{x['cols']}")
            px, py = x.get("probe") or {}, y.get("probe") or {}
            sx, sy = px.get("status"), py.get("status")
            if sx in ("unsat", "sat") and sy != sx:
                problems.append(f"{name}: status regression {sx}->{sy} at {x['rows']}/{x['cols']}")
            if sx == "unsat" and not x.get("censored") and (x.get("d") != y.get("d")):
                problems.append(f"{name}: exact label changed {x.get('d')} -> {y.get('d')} at {x['rows']}/{x['cols']}")
            if sx == "unknown" and sy == "unsat":
                n_c2e += 1
            elif sx == "unknown" and sy == "unknown":
                if (py.get("conflicts") or 0) > (px.get("conflicts") or 0):
                    n_lb += 1
                if x.get("d") != y.get("d"):
                    n_rel += 1
        changed = a.get("table_hash") != b.get("table_hash")
        content_changed = ra != rb
        if content_changed and not changed:
            problems.append(f"{name}: records changed but table_hash not bumped")
        tot["censored_to_exact"] += n_c2e; tot["lb_raised"] += n_lb; tot["relabelled"] += n_rel
        tot["changed_tables"] += int(content_changed)
        lines.append(f"{name:45s} n={len(ra):5d} mask={'-' if ma is None else len(ma):>5} load={ok_load} "
                     f"censored->exact={n_c2e:4d} lb_raised={n_lb:3d} relabelled={n_rel:5d} hash_bumped={changed}")
    lines.append("totals: " + json.dumps(tot))
    lines.append("PROBLEMS: %d" % len(problems))
    lines += ["  " + p for p in problems[:50]]
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_tables.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()

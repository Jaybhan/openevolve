"""Command-line entry points (no LLM needed).

  python -m zar_ub table  M N S T W [--pure|--trust tan2022] [--mode exact|censored] [--cap N] [--jobs J]
                                     [--baseline] [--force] [--kind K]      build a case table / add the Lean baseline mask
  python -m zar_ub show   M N S T W [--pure] [--all]                        summarize a cached table
  python -m zar_ub gate   M N S T W FILE.lean [--pure]                      run the Lean gate on a candidate
  python -m zar_ub calibrate [--holdout 12,13,87] [--train ...]             fit the censored-label model (design §6.2)
  python -m zar_ub deepen M N S T W --cap N [--pure] [--jobs J] [--limit K]  continue the SCHEDULE on censored cases, in place
  python -m zar_ub relabel M N S T W [--pure] | --all                        refresh censored labels with the current calibration
  python -m zar_ub baseline [--all|--suite] [--timeout S]                   Lean baseline masks for every cached suite table
  python -m zar_ub bound  M N S T [--pure]                                  smallest w whose table is fully refuted
  python -m zar_ub certify M N S T W [--pure] [--time 600] [--lean FILE]    LRAT-certify every case not killed by the prune

Plugins (registered through `register(sub)`; `python -m zar_ub plugins` lists their status):
  python -m zar_ub closure M N S T W [--pure|--trust tan2022] [--lean FILE] [--tier 1|1n]   Tier-1 closure file + report (zar_ub.closure)
  python -m zar_ub promote FILE.py [--name N] ...                                         promote an L5 program into lean/ZarPrune/Evolved (zar_ub.promote)
  python -m zar_ub evolved-regen                                                          regenerate lean/ZarPrune/Evolved.lean from the ledger
  python -m zar_ub closure-daemon --checkpoint-dir DIR --targets "m,n,w;..." [--now]      design §7 daemon (zar_ub.closure_daemon)
  python -m zar_ub verify-certs [DIR] [--fresh-lratcheck]                                 re-verify every LRAT certificate (zar_ub.certify)
  python -m zar_ub audit-kills M N S T W [--frac 0.05] [--pure]                           SAT-audit a sample of library-killed cases (zar_ub.certify)

zar_ub.promote / closure / closure_daemon / certify expose `register(sub)` (argparse subparsers) and
`set_defaults(func=handler)`; they are registered first, so an inline subcommand of the same name yields.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

from .known import Instance, exact_value, ub_counting
from .casetable import (
    build_table,
    load_table,
    add_baseline,
    compute_baseline_masks,
    set_baseline_mask,
    deepen,
    relabel,
    difficulty,
    LIBRARY_LEAN,
)

PLUGIN_MODULES = ("promote", "closure", "closure_daemon", "certify")


def _register_plugins(sub) -> dict:
    """Import zar_ub.<mod> for each plugin and call register(sub) when present."""
    found = {}
    for name in PLUGIN_MODULES:
        try:
            mod = importlib.import_module(f"zar_ub.{name}")
        except ModuleNotFoundError as e:
            if e.name == f"zar_ub.{name}":
                found[name] = "absent"
                continue
            found[name] = f"import failed: {e.__class__.__name__}: {e}"
            continue
        except Exception as e:  # noqa: BLE001 - a broken optional plugin must not kill the CLI
            found[name] = f"import failed: {e.__class__.__name__}: {e}"
            continue
        reg = getattr(mod, "register", None)
        if callable(reg):
            try:
                reg(sub)
                found[name] = "registered"
            except Exception as e:  # noqa: BLE001
                found[name] = f"register failed: {e.__class__.__name__}: {e}"
        else:
            found[name] = "no register()"
    return found


def _add_inst(p, with_w=True, optional=False):
    kw = {"nargs": "?", "default": None} if optional else {}
    p.add_argument("m", type=int, **kw)
    p.add_argument("n", type=int, **kw)
    p.add_argument("s", type=int, **kw)
    p.add_argument("t", type=int, **kw)
    if with_w:
        p.add_argument("w", type=int, **kw)
    p.add_argument(
        "--pure", action="store_true", help="no external exact values (Argument I uses the counting bound only)"
    )
    p.add_argument(
        "--trust",
        choices=["pure", "tan2022"],
        default=None,
        help="provenance filter for Argument I: pure (= --pure) or tan2022 (data/exact_33.csv; default)",
    )


def _use_table(args) -> bool:
    if args.pure or args.trust == "pure":
        return False
    return True


def _parse_cells(raw: str):
    """'12,13,87;9,23,104' or 'm,n,s,t,w' chunks -> [Instance]."""
    from suite import parse_targets  # noqa: E402  (suite.py lives next to the package)

    return parse_targets(raw)


def main(argv=None):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser(prog="zar_ub")
    sub = ap.add_subparsers(dest="cmd", required=True)
    plugins = _register_plugins(sub)

    def parser(name, **kw):
        if name in sub.choices:  # a plugin took this name: it owns the command
            return None
        return sub.add_parser(name, **kw)

    p = parser("table", help="build a case table (labels per design §6.2) and/or its Lean baseline mask")
    if p:
        _add_inst(p)
        p.add_argument(
            "--cap", type=int, default=None, help="largest SCHEDULE conflict cap (default: 2M exact / 20k censored)"
        )
        p.add_argument("--time", type=float, default=None, help="(ignored: the SCHEDULE fixes per-cap time limits)")
        p.add_argument(
            "--mode",
            choices=["exact", "censored"],
            default=None,
            help="label mode (default: exact for cells with a known z, censored otherwise)",
        )
        p.add_argument("--jobs", type=int, default=1)
        p.add_argument("--solver", default="cadical195")
        p.add_argument("--no-probe", action="store_true")
        p.add_argument("--kind", default="")
        p.add_argument(
            "--baseline", action="store_true", help="compute baseline_lean_mask with the Lean gate on LIBRARY_LEAN"
        )
        p.add_argument("--force", action="store_true", help="rebuild even if a table is cached")
        p.add_argument("--timeout", type=float, default=600.0, help="Lean gate timeout for --baseline")
    p = parser("show")
    if p:
        _add_inst(p)
        p.add_argument("--all", action="store_true", help="list every record, not only the hardest 20")
    p = parser("gate")
    if p:
        _add_inst(p)
        p.add_argument("file")
    p = parser("calibrate", help="fit log fhat = a + b log c2000 + g log2_volume on TRAIN; validate on --holdout")
    if p:
        p.add_argument("--holdout", default="12,13,87", help="held-out cells 'm,n,w;...' (pure tables)")
        p.add_argument("--train", default=None, help="override TRAIN cells 'm,n,w;...' (default: suite TRAIN)")
        p.add_argument("--st", default="3,3")
        p.add_argument("--no-write", action="store_true")
        p.add_argument(
            "--out", default=None, help="calibration JSON path (default experiments/E11_calibration/calibrate_33.json)"
        )
    p = parser(
        "deepen", help="continue the conflict SCHEDULE on censored cases; updates labels and table_hash in place"
    )
    if p:
        _add_inst(p)
        p.add_argument("--cap", type=int, required=True)
        p.add_argument("--jobs", type=int, default=1)
        p.add_argument("--time", type=float, default=None)
        p.add_argument("--limit", type=int, default=None, help="deepen only the K hardest censored cases")
        p.add_argument("--solver", default="cadical195")
    p = parser("relabel", help="recompute censored labels with the current calibration (no solving); bumps table_hash")
    if p:
        _add_inst(p, optional=True)
        p.add_argument("--all", action="store_true", help="every cached table")
    p = parser("baseline", help="Lean baseline masks for cached suite tables (one Lean process per batch)")
    if p:
        p.add_argument("--all", action="store_true", help="every cached table under cache/, not only the suite")
        p.add_argument("--force", action="store_true", help="recompute masks that already exist")
        p.add_argument("--timeout", type=float, default=900.0)
        p.add_argument("--batch", type=int, default=6, help="tables per Lean process")
    p = parser("certify")
    if p:
        _add_inst(p)
        p.add_argument("--time", type=float, default=600.0)
        p.add_argument(
            "--lean",
            default=None,
            help="candidate Lean file; its gate-computed kill mask decides which cases are skipped",
        )
        p.add_argument("--no-keep", action="store_true")
    p = parser("bound")
    if p:
        _add_inst(p, with_w=False)
        p.add_argument("--cap", type=int, default=20000)
        p.add_argument("--time", type=float, default=60.0)
    p = parser("plugins")
    p = parser("api", help="regenerate lean/API.md (and, with --docstring FILE, the LEAN API block of FILE)")
    if p:
        p.add_argument("--docstring", default=None, metavar="FILE", help="program whose docstring block to refresh")
        p.add_argument("--config", default=None, metavar="YAML", help="config.yaml whose system-message API block to refresh")
        p.add_argument("--width", type=int, default=110)
    args = ap.parse_args(argv)

    if hasattr(args, "func") and callable(args.func):
        return args.func(args)

    if args.cmd == "plugins":
        print(json.dumps(plugins, indent=1))
    elif args.cmd == "api":
        from zar_ub import lean_api

        text, digest = lean_api.build()
        print(f"[api] wrote lean/API.md ({len(text.splitlines())} lines, {len(digest.splitlines())} declarations)")
        if args.docstring:
            changed = lean_api.refresh_docstring(args.docstring, width=args.width)
            print(f"[api] {args.docstring}: docstring block {'updated' if changed else 'unchanged'}")
        if args.config:
            changed = lean_api.refresh_docstring(args.config, width=args.width, end="WORKED EXAMPLE", indent="    ")
            print(f"[api] {args.config}: system-message API block {'updated' if changed else 'unchanged'}")
    elif args.cmd == "table":
        inst = Instance(args.m, args.n, args.s, args.t, args.w)
        use_table = _use_table(args)
        tab = None if args.force else load_table(inst, use_table=use_table)
        if tab is None:
            tab = build_table(
                inst,
                conf_cap=args.cap,
                probe=not args.no_probe,
                solver=args.solver,
                use_table=use_table,
                mode=args.mode,
                jobs=args.jobs,
                kind=args.kind,
            )
            print(tab.save())
        else:
            print(f"[table] cached {tab.path} (hash {tab.table_hash}); probes not recomputed (use --force to rebuild)")
            if args.kind:
                tab.kind = args.kind
                tab.save()
        if args.baseline:
            if tab.baseline_lean_mask is not None and not args.force:
                print(
                    f"[table] baseline_lean_mask already present (kills {sum(tab.baseline_lean_mask)}/{len(tab.records)})"
                )
            elif not add_baseline(tab, timeout=args.timeout):
                sys.exit("Lean gate failed on LIBRARY_LEAN; mask not stored")
            else:
                print(
                    f"[table] baseline_lean_mask stored: kills {sum(tab.baseline_lean_mask)}/{len(tab.records)}; hash {tab.table_hash}"
                )
        print(json.dumps(tab.summary(), indent=1))
    elif args.cmd == "show":
        inst = Instance(args.m, args.n, args.s, args.t, args.w)
        tab = load_table(inst, use_table=_use_table(args))
        if tab is None:
            sys.exit("no cached table; run `table` first")
        print(json.dumps(tab.summary(), indent=1))
        sc = tab.scored()
        sc.sort(key=lambda ir: -difficulty(ir[1]))
        for i, r in (sc if args.all else sc[:20]):
            print(
                f"  [{i}] rows={r.rows} cols={r.cols} status={r.status} d={r.d:.0f}"
                f"{' censored' if r.censored else ''} lean_baseline_kill={r.baseline_lean_kill}"
            )
    elif args.cmd == "gate":
        from .lean_gate import run_gate

        inst = Instance(args.m, args.n, args.s, args.t, args.w)
        tab = load_table(inst, use_table=_use_table(args))
        cases = [(r.rows, r.cols) for r in tab.records] if tab else None
        with open(args.file, encoding="utf-8") as fh:
            src = fh.read()
        if args.file.endswith(".py"):  # a candidate program: gate its LEAN_SOURCE (design T-2)
            import runpy

            src = runpy.run_path(args.file, run_name="zar_ub_gate")["LEAN_SOURCE"]
        from .ledger import facts_for

        trust = "pure" if not _use_table(args) else (args.trust or "tan2022")
        facts = [f.lean for f in facts_for(inst, trust)]  # granted to a `candidateF : CondPrune` (design §8.2)
        res = run_gate(inst, src, cases, facts=facts)
        d = res.as_dict()
        d.pop("stdout_tail", None)
        print(json.dumps(d, indent=1))
        if res.kill_mask is not None and tab is not None:
            killed_surv = sum(1 for r, k in zip(tab.records, res.kill_mask) if k and not r.baseline_lean_kill)
            print(f"kills {sum(res.kill_mask)} cases, {killed_surv} of {len(tab.scored_indices())} scored survivors")
    elif args.cmd == "calibrate":
        from .difficulty import calibrate
        from suite import train_instances

        st = tuple(int(x) for x in args.st.split(","))
        tr_insts = _parse_cells(args.train) if args.train else train_instances()
        ho_insts = _parse_cells(args.holdout) if args.holdout else []

        def pairs(insts):
            out = []
            for i in insts:
                t = load_table(i, use_table=False) or load_table(i, use_table=True)
                if t is None:
                    print(f"[calibrate] missing table {i.tag}", file=sys.stderr)
                else:
                    out.append((i, t))
            return out

        res = calibrate(
            pairs(tr_insts),
            pairs(ho_insts),
            st=st,
            write=not args.no_write,
            **({"out_path": args.out} if args.out else {}),
        )
        print(json.dumps(res, indent=1))
    elif args.cmd == "deepen":
        inst = Instance(args.m, args.n, args.s, args.t, args.w)
        tab = load_table(inst, use_table=_use_table(args))
        if tab is None:
            sys.exit("no cached table; run `table` first")
        print(
            json.dumps(
                deepen(tab, cap=args.cap, time_limit=args.time, jobs=args.jobs, solver=args.solver, limit=args.limit),
                indent=1,
            )
        )
    elif args.cmd == "relabel":
        from .casetable import CACHE_DIR, table_from_json
        import glob

        if args.all:
            for path in sorted(glob.glob(os.path.join(CACHE_DIR, "case_table_*.json"))):
                print(json.dumps(relabel(table_from_json(json.load(open(path)), path))))
        else:
            if args.w is None:
                sys.exit("relabel: give M N S T W or --all")
            inst = Instance(args.m, args.n, args.s, args.t, args.w)
            tab = load_table(inst, use_table=_use_table(args))
            if tab is None:
                sys.exit("no cached table; run `table` first")
            print(json.dumps(relabel(tab), indent=1))
    elif args.cmd == "baseline":
        from .casetable import CACHE_DIR, table_from_json
        import glob

        tables = []
        if args.all:
            for path in sorted(glob.glob(os.path.join(CACHE_DIR, "case_table_*.json"))):
                tables.append(table_from_json(json.load(open(path)), path))
        else:
            from suite import load_suite

            s = load_suite()
            seen = set()
            for kind in ("train", "battery", "target", "gen"):
                for inst, tab in s[kind]:
                    if tab.path and tab.path not in seen:
                        seen.add(tab.path)
                        tables.append(tab)
        todo = [t for t in tables if t.records and (args.force or t.baseline_lean_mask is None)]
        print(f"[baseline] {len(todo)} tables need a mask (of {len(tables)})")
        for k in range(0, len(todo), args.batch):
            batch = todo[k : k + args.batch]
            masks = compute_baseline_masks(batch, timeout=args.timeout, tag="library")
            for t, m in zip(batch, masks):
                if m is None:
                    print(f"[baseline] {t.instance.tag}: FAILED (mask not stored)")
                    continue
                set_baseline_mask(t, m)
                t.save()
                print(f"[baseline] {t.instance.tag}: stored ({sum(m)}/{len(m)} killed) hash {t.table_hash} -> {t.path}")
        for t in tables:
            if not t.records and t.baseline_lean_mask is None:
                set_baseline_mask(t, [])
                t.save()
    elif args.cmd == "certify":
        from .certify import certify_survivors

        inst = Instance(args.m, args.n, args.s, args.t, args.w)
        tab = load_table(inst, use_table=_use_table(args))
        if tab is None:
            sys.exit("no cached table; run `table` first")
        cases = [(r.rows, r.cols) for r in tab.records]
        skipped = 0
        if args.lean:
            from .lean_gate import run_gate

            g = run_gate(inst, open(args.lean).read(), cases)
            if not g.ok:
                sys.exit(f"Lean gate rejected the prune: {g.errors[:3]}")
            keep = [c for c, k in zip(cases, g.kill_mask) if not k]
            skipped = len(cases) - len(keep)
            cases = keep
            print(f"Lean-verified prune kills {skipped} cases; certifying the remaining {len(cases)}")
        m = certify_survivors(inst, cases, time_limit=args.time, keep_lrat=not args.no_keep)
        m["pruned_by_lean"] = skipped
        m["external_facts_used_in_case_generation"] = tab.external_facts
        print(json.dumps(m, indent=1))
        if m["certified"] == m["n_cases"]:
            print(f"=> every case pruned or LRAT-certified: z({args.m},{args.n};{args.s},{args.t}) <= {args.w - 1}")
    elif args.cmd == "bound":
        ev = exact_value(args.m, args.n, args.s, args.t)
        ub = ub_counting(args.m, args.n, args.s, args.t)
        print(f"counting UB = {ub}; external exact = {ev}")
        w = ub + 1 if ev is None else ev + 1
        use_table = _use_table(args)
        while w > 0:
            inst = Instance(args.m, args.n, args.s, args.t, w)
            tab = load_table(inst, use_table=use_table) or build_table(
                inst, conf_cap=args.cap, verbose=False, use_table=use_table, mode="censored"
            )
            tab.save()
            s = tab.summary()
            open_cases = sum(v for k, v in s["status"].items() if k != "unsat")
            print(f"w={w}: cases={s['cases']} open={open_cases}")
            if open_cases:
                print(f"=> z({args.m},{args.n};{args.s},{args.t}) <= {w} proven by tables above; w={w} not refuted")
                break
            w -= 1


if __name__ == "__main__":
    main()

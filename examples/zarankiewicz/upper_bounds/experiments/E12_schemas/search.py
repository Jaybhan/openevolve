#!/usr/bin/env python3
"""E12 — Farkas certificates over hidden pair codegrees (design §2.3 / §8.4; Schemas.lean).

For every library survivor of every cached TRAIN table (and, with --extra, the (10,20,103) pure
table, the calibration/band tables and the GEN table) solve the dual LP of the pair-codegree
system F1–F4 (zar_ub.lemmas.farkas_certificate); when it is feasible, rationalise the
multipliers into an integer Farkas certificate, verify it with the exact Python mirror,
greedily keep only certificates that kill a not-yet-killed survivor, and report per table how
many library survivors the schema kills and what share of their difficulty d (= gain_I) —
the headline number.

Soundness self-checks: (a) the mirror must never kill a SAT-witnessed case of any cached
table (battery tables at w = z carry witnesses); (b) with --gate the same certificates go
through the real Lean gate (`run_gate_multi` with `schema_terms`) and the Lean `schema_mask`
must equal the mirror on every case.  Until `ZarPrune.Schemas` is imported by
`lean/ZarPrune.lean` and built, the gate check inlines the Schemas body into the candidate
(renamed `ofFarkasInl`) and sets ZAR_UB_FARKAS_LEAN accordingly (--inline, the default);
after the build, run with --no-inline.

Outputs (experiments/E12_schemas/): certs.json (certificates per instance), results.json,
REPORT.md; with --write-candidate: tests/candidates/schema_farkas.py.

    cd examples/zarankiewicz/upper_bounds
    ZAR_UB_NO_LLM=1 python experiments/E12_schemas/search.py --extra --gate --write-candidate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, UB)

from zar_ub import Instance  # noqa: E402
from zar_ub import lemmas  # noqa: E402
from zar_ub.casetable import CaseTable, load_table, tail_indices  # noqa: E402

EXTRA = [
    Instance(10, 20, 3, 3, 103),
    Instance(12, 13, 3, 3, 87),
    Instance(13, 13, 3, 3, 93),
    Instance(10, 14, 3, 3, 78),
    Instance(9, 9, 4, 4, 62),
    Instance(12, 13, 3, 3, 86),
    Instance(13, 13, 3, 3, 92),
    Instance(10, 14, 3, 3, 77),
]


def train_tables() -> List[Tuple[Instance, CaseTable]]:
    import suite  # noqa: WPS433

    s = suite.load_suite(verbose=False, targets=[])
    return [(inst, tab) for inst, tab in s["train"]]


def search_table(inst: Instance, tab: CaseTable, verbose: bool = True) -> dict:
    """Greedy certificate cover of the LP-infeasible library survivors of one table."""
    m, n, s, t = inst.m, inst.n, inst.s, inst.t
    recs = tab.records
    surv = tab.scored_indices()
    systems = {i: lemmas.farkas_system(m, n, s, t, recs[i].rows, recs[i].cols) for i in surv}
    certs: List[List[int]] = []
    killed: set = set()
    lp_infeasible_no_cert = []
    t0 = time.time()
    for i in surv:
        if i in killed:
            continue
        y = lemmas.farkas_certificate(m, n, s, t, recs[i].rows, recs[i].cols)
        if y is None:
            continue
        y = lemmas.sparsify(y)
        assert lemmas.cert_kills_system(systems[i], y)
        certs.append(y)
        for j in surv:
            if j not in killed and lemmas.cert_kills_system(systems[j], y):
                killed.add(j)
    seconds = time.time() - t0
    # exact mirror mask on the whole table (what the evaluator computes)
    mask = [lemmas.farkas_kill(m, n, s, t, r.rows, r.cols, certs) for r in recs]
    # soundness self-check against SAT-witnessed cases
    sat_killed = [i for i, r in enumerate(recs) if mask[i] and r.status == "sat"]
    d_surv = sum(recs[i].d for i in surv)
    d_kill = sum(recs[i].d for i in surv if mask[i])
    tail = tail_indices(tab)
    tail_kill = sum(1 for i in tail if mask[i])
    hardest = sorted((i for i in surv if mask[i]), key=lambda i: -recs[i].d)[:5]
    out = {
        "inst": inst.tag,
        "kind": tab.kind,
        "cases": len(recs),
        "library_survivors": len(surv),
        "schema_kills": sum(1 for i in surv if mask[i]),
        "certs": len(certs),
        "nnz": [sum(1 for v in y if v) for y in certs],
        "d_survivors": d_surv,
        "d_killed": d_kill,
        "gain": (d_kill / d_surv) if d_surv else 0.0,
        "tail_size": len(tail),
        "tail_kills": tail_kill,
        "censored_killed": sum(1 for i in surv if mask[i] and recs[i].censored),
        "sat_killed": sat_killed,
        "lp_no_cert": lp_infeasible_no_cert,
        "seconds": round(seconds, 2),
        "hardest_killed": [{"rows": recs[i].rows, "cols": recs[i].cols, "d": recs[i].d} for i in hardest],
        "mask_true": sum(mask),
    }
    if verbose:
        print(
            f"{inst.tag:24s} kind={tab.kind or '-':7s} cases={len(recs):5d} survivors={len(surv):5d} "
            f"kills={out['schema_kills']:4d} certs={len(certs):3d} gain={out['gain']:.3f} "
            f"tail {tail_kill}/{len(tail)}  sat_killed={len(sat_killed)}  ({seconds:.1f}s)"
        )
    return out, certs, mask


def inlined_candidate() -> str:
    """The Schemas body as a gate candidate (comments stripped, namespace lines dropped,
    `Prune.ofFarkas` renamed `ofFarkasInl`) plus the library candidate."""
    from zar_ub import lean_gate as lg  # noqa: WPS433

    with open(os.path.join(UB, "lean", "ZarPrune", "Schemas.lean"), encoding="utf-8") as f:
        src = f.read()
    body = lg.strip_comments(src.split("\n", 1)[1])
    lines = [l for l in body.splitlines() if l.strip() not in ("namespace ZarPrune", "end ZarPrune")]
    text = "\n".join(lines).replace("def Prune.ofFarkas", "def ofFarkasInl")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text + "\n\ndef candidate (P : Params) : Prune P := counting P\n"


def gate_check(
    tables, certs_by_tag: Dict[str, List[List[int]]], masks_by_tag: Dict[str, List[bool]], inline: bool, timeout: float
) -> dict:
    """Run the real gate with the certificates as schema terms and compare its schema_mask with
    the mirror on every case."""
    if inline:
        os.environ["ZAR_UB_FARKAS_LEAN"] = "ZarPrune.Cand.ofFarkasInl"
        lemmas.FARKAS_LEAN = "ZarPrune.Cand.ofFarkasInl"
        lemmas.FAMILIES["farkas"] = lemmas.Family(
            lean=lemmas.FARKAS_LEAN, params=lemmas.FAMILIES["farkas"].params, doc=lemmas.FAMILIES["farkas"].doc
        )
        cand = inlined_candidate()
    else:
        cand = "def candidate (P : Params) : Prune P := counting P\n"
    from zar_ub.lean_gate import run_gate_multi  # noqa: WPS433

    inputs, terms = [], []
    for k, (inst, tab) in enumerate(tables):
        inputs.append((inst, [(r.rows, r.cols) for r in tab.records]))
        sd = {
            "farkas": [
                {"m": inst.m, "n": inst.n, "s": inst.s, "t": inst.t, "y": y} for y in certs_by_tag.get(inst.tag, [])
            ]
        }
        tname = "ZarPrune.Cand.target" + ("" if k == 0 else str(k))
        terms.append(lemmas.render_terms(sd, inst, tname))
    t0 = time.time()
    res = run_gate_multi(inputs, cand, schema_terms=terms, timeout=timeout, tag="e12", sketch=False, use_cache=False)
    secs = time.time() - t0
    out = {"seconds": round(secs, 1), "per_instance": {}}
    ok_all = True
    for (inst, tab), r in zip(tables, res):
        sm = r.schema_mask
        mirror = masks_by_tag[inst.tag]
        same = sm is not None and len(sm) == len(mirror) and all(bool(a) == bool(b) for a, b in zip(sm, mirror))
        ok_all &= r.ladder == 5 and (same or not certs_by_tag.get(inst.tag))
        out["per_instance"][inst.tag] = {
            "ladder": r.ladder,
            "axioms": r.axioms,
            "schema_mask_true": (sum(sm) if sm else None),
            "mirror_true": sum(mirror),
            "identical": same,
            "errors": r.errors[:3],
        }
        print(
            f"  gate {inst.tag:24s} L{r.ladder} axioms={r.axioms} schema_mask={sum(sm) if sm else None} "
            f"mirror={sum(mirror)} identical={same} {r.errors[:1]}"
        )
    out["ok"] = ok_all
    print(f"  gate: one Lean process, {secs:.1f}s, ok={ok_all}")
    return out


def write_candidate(schema_data: dict, notes: str) -> str:
    """tests/candidates/schema_farkas.py = initial_program.py with SCHEMA_DATA filled in."""
    with open(os.path.join(UB, "initial_program.py"), encoding="utf-8") as f:
        src = f.read()
    lit = json.dumps(schema_data, separators=(",", ":"))
    new = re.sub(r"SCHEMA_DATA = \{.*?\}\n", "SCHEMA_DATA = " + lit + "\n", src, count=1, flags=re.S)
    assert new != src, "SCHEMA_DATA line not found in initial_program.py"
    new = new.replace('NOTES = r"""Start: the proved counting library only."""', 'NOTES = r"""' + notes + '"""', 1)
    tag = (
        '"""[CANDIDATE `schema_farkas` -- owner E-schemas] initial_program.py + SCHEMA_DATA holding the Farkas '
        "certificates found by experiments/E12_schemas/search.py (pair-codegree system F1-F4, "
        "lean/ZarPrune/Schemas.lean). Expected: L5, score > 0.20 once ZarPrune.Schemas is built.\n\n"
        "Generated by experiments/E12_schemas/search.py --write-candidate; do not edit by hand.\n\n"
    )
    assert new.startswith('"""')
    new = tag + new[3:]
    path = os.path.join(UB, "tests", "candidates", "schema_farkas.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
    return path


def emit_candidate(certs_by_tag: Dict[str, List[List[int]]]) -> str:
    """tests/candidates/schema_farkas.py with the TRAIN-cell certificates as dict entries (each fires
    only on its (m,n,s,t) cell).  TRAIN cells only: evaluator.py (batch 1) renders the schema terms
    with the index of the unfiltered `_scored(suite)` list while the gate numbers the filtered
    list, so a GEN-cell entry would be rendered with the wrong `targetK` (integrator TODO); the
    TRAIN instances come first in both enumerations."""
    import suite  # noqa: WPS433

    cells = set(suite.TRAIN_CELLS)
    entries = []
    for tag, ys in certs_by_tag.items():
        m, n, s, t, _w = (int(x[1:]) for x in tag.split("_"))
        if (m, n) not in cells or (s, t) != (3, 3):
            continue
        for y in ys:
            entries.append({"m": m, "n": n, "s": s, "t": t, "y": y})
    notes = (
        "Proved library + Farkas certificates (SCHEMA_DATA['farkas']) for the pair-codegree system "
        "F1-F4 of lean/ZarPrune/Schemas.lean, one dict entry per TRAIN (m,n,s,t) cell; found by "
        "experiments/E12_schemas/search.py (exact rational LP, verified by the Python mirror and the gate)."
    )
    path = write_candidate({"farkas": entries, "residue": [], "prefix": []}, notes)
    print("wrote", os.path.relpath(path, UB), "with", len(entries), "certificates")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", action="store_true", help="also the (10,20,103), band/calibration and GEN tables")
    ap.add_argument("--gate", action="store_true", help="verify through the Lean gate (one process)")
    ap.add_argument("--no-inline", action="store_true", help="gate with the built ZarPrune.Schemas (after lake build)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--write-candidate", action="store_true", help="write tests/candidates/schema_farkas.py")
    ap.add_argument(
        "--write-candidate-only",
        action="store_true",
        help="only rewrite tests/candidates/schema_farkas.py from certs.json (no search, no gate)",
    )
    args = ap.parse_args(argv)
    if args.write_candidate_only:
        with open(os.path.join(HERE, "certs.json")) as f:
            certs_by_tag = json.load(f)
        emit_candidate(certs_by_tag)
        return

    tables = train_tables()
    if args.extra:
        for inst in EXTRA:
            tab = load_table(inst, use_table=False)
            if tab is not None and inst.tag not in {i.tag for i, _ in tables}:
                tables.append((inst, tab))
    results, certs_by_tag, masks_by_tag = [], {}, {}
    for inst, tab in tables:
        out, certs, mask = search_table(inst, tab)
        results.append(out)
        certs_by_tag[inst.tag] = certs
        masks_by_tag[inst.tag] = mask
    # soundness self-check on EVERY cached table with SAT-witnessed cases (battery tables at w = z)
    sat_checks = []
    for fn in sorted(os.listdir(os.path.join(UB, "cache"))):
        if not fn.startswith("case_table_") or not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(UB, "cache", fn)))
        inst = Instance(**d["inst"])
        certs = [
            y
            for tag, ys in certs_by_tag.items()
            for y in ys
            if tag.startswith(f"m{inst.m}_n{inst.n}_s{inst.s}_t{inst.t}_")
        ]
        if not certs:
            continue
        sats = [r for r in d["records"] if (r.get("probe") or {}).get("status") == "sat"]
        bad = [r for r in sats if lemmas.farkas_kill(inst.m, inst.n, inst.s, inst.t, r["rows"], r["cols"], certs)]
        sat_checks.append({"table": fn, "sat_cases": len(sats), "sat_killed": len(bad)})
        if bad:
            print("!!! SOUNDNESS: mirror killed a SAT case", fn, bad[0]["rows"], bad[0]["cols"])
    print(
        "SAT self-check:",
        sum(c["sat_cases"] for c in sat_checks),
        "witnessed cases in",
        len(sat_checks),
        "tables; killed:",
        sum(c["sat_killed"] for c in sat_checks),
    )

    gate = None
    if args.gate:
        gate = gate_check(tables, certs_by_tag, masks_by_tag, inline=not args.no_inline, timeout=args.timeout)

    # headline
    train = [r for r in results if r["kind"] == "train"]

    def agg(rs):
        S = sum(r["library_survivors"] for r in rs)
        K = sum(r["schema_kills"] for r in rs)
        D = sum(r["d_survivors"] for r in rs)
        Dk = sum(r["d_killed"] for r in rs)
        G = [r["gain"] for r in rs if r["library_survivors"]]
        return {
            "tables": len(rs),
            "survivors": S,
            "kills": K,
            "d_share": (Dk / D) if D else 0.0,
            "mean_gain": (sum(G) / len(G)) if G else 0.0,
            "certs": sum(r["certs"] for r in rs),
        }

    summary = {"train": agg(train), "all": agg(results), "sat_checks": sat_checks, "gate": gate, "tables": results}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(HERE, "certs.json"), "w") as f:
        json.dump({tag: ys for tag, ys in certs_by_tag.items()}, f)
    print("TRAIN:", summary["train"])
    print("ALL:  ", summary["all"])

    # report
    lines = [
        "# E12 — Farkas certificates over hidden pair codegrees",
        "",
        "Per table: library survivors S_I, schema kills, certificates kept (greedy cover), gain_I = Σ d(killed) / Σ d(S_I),",
        "tail kills (top decile by d), SAT-witnessed cases killed (must be 0).",
        "",
        "| table | kind | cases | S_I | kills | certs | gain_I | tail | censored killed | sat killed | LP s |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['inst']} | {r['kind'] or '-'} | {r['cases']} | {r['library_survivors']} | {r['schema_kills']} | {r['certs']} | "
            f"{r['gain']:.3f} | {r['tail_kills']}/{r['tail_size']} | {r['censored_killed']} | {len(r['sat_killed'])} | {r['seconds']} |"
        )
    a = summary["train"]
    lines += [
        "",
        f"**TRAIN headline**: {a['kills']} of {a['survivors']} library survivors killed over {a['tables']} tables, "
        f"difficulty share {a['d_share']:.3f}, mean gain_I {a['mean_gain']:.3f}, {a['certs']} certificates.",
        f"**All tables**: {summary['all']['kills']} of {summary['all']['survivors']} survivors, d share {summary['all']['d_share']:.3f}.",
        "",
        f"SAT self-check: {sum(c['sat_cases'] for c in sat_checks)} witnessed cases in {len(sat_checks)} tables, "
        f"killed {sum(c['sat_killed'] for c in sat_checks)}.",
    ]
    if gate:
        lines += [
            "",
            f"Gate ({'inlined Schemas body' if not args.no_inline else 'built ZarPrune.Schemas'}): one Lean process, {gate['seconds']} s, ok={gate['ok']}.",
            "",
            "| table | ladder | axioms | Lean schema_mask | mirror | identical |",
            "|---|---|---|---|---|---|",
        ]
        for tag, g in gate["per_instance"].items():
            lines.append(
                f"| {tag} | {g['ladder']} | {g['axioms']} | {g['schema_mask_true']} | {g['mirror_true']} | {g['identical']} |"
            )
    lines += ["", "Hardest killed survivors per table:", ""]
    for r in results:
        for h in r["hardest_killed"][:2]:
            lines.append(f"- {r['inst']}: rows {h['rows']} cols {h['cols']} d={h['d']:.0f}")
    with open(os.path.join(HERE, "REPORT.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    if args.write_candidate:
        emit_candidate(certs_by_tag)


if __name__ == "__main__":
    main()

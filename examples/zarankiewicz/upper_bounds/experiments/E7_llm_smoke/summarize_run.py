#!/usr/bin/env python3
"""Summarize an OpenEvolve run directory: one row per program (latest checkpoint),
ladder / score / gains / battery / what changed in the Lean source vs the initial program.
  python experiments/E7_llm_smoke/summarize_run.py <run_dir> [--md out.md]
"""
import argparse, glob, json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__)); UB = os.path.abspath(os.path.join(HERE, "..", ".."))


def lean_of(code):
    m = re.search(r"LEAN_SOURCE\s*=\s*r?'''(.*?)'''", code, re.S) or re.search(r'LEAN_SOURCE\s*=\s*r?"""(.*?)"""', code, re.S)
    return m.group(1) if m else ""


def new_decls(lean, base_lean):
    base = set(re.findall(r"^(?:def|theorem|lemma)\s+([\w.']+)", base_lean, re.M))
    return [d for d in re.findall(r"^(?:def|theorem|lemma)\s+([\w.']+)", lean, re.M) if d not in base]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--md", default=None)
    a = ap.parse_args()
    cps = sorted(glob.glob(os.path.join(a.run_dir, "checkpoints", "checkpoint_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    if not cps:
        sys.exit("no checkpoints")
    progs = []
    for f in glob.glob(os.path.join(cps[-1], "programs", "*.json")):
        d = json.load(open(f)); progs.append(d)
    base_lean = lean_of(open(os.path.join(UB, "initial_program.py")).read())
    progs.sort(key=lambda d: (d.get("iteration_found", 0), d.get("id")))
    rows = []
    for d in progs:
        m = d.get("metrics") or {}
        lean = lean_of(d.get("code", ""))
        nd = new_decls(lean, base_lean)
        sd = re.search(r"SCHEMA_DATA\s*=\s*(\{.*?\})\s*\n", d.get("code", ""), re.S)
        schema = "yes" if sd and '"farkas": [' in sd.group(1) and sd.group(1).count("[") > 3 else "-"
        arts = d.get("artifacts_json") or {}
        if isinstance(arts, str):
            try: arts = json.loads(arts)
            except Exception: arts = {}
        err = ""
        for k in ("lean_errors", "UNSOUND_counterexamples", "python_error", "schema_error"):
            if arts.get(k): err = k + ": " + str(arts[k])[:90].replace("\n", " "); break
        rows.append({"iter": d.get("iteration_found"), "id": d.get("id", "")[:8], "score": m.get("combined_score"),
                     "ladder": m.get("lean_ladder"), "proven": m.get("proven_gain"), "schema": m.get("schema_gain"),
                     "empirical": m.get("empirical_gain"), "battery": m.get("sound_battery"), "agree": m.get("agreement"),
                     "new_decls": ",".join(nd)[:60], "schema_data": schema, "err": err})
    fmt = lambda v: (f"{v:.4f}" if isinstance(v, float) else str(v))
    lines = ["| iter | id | score | L | proven | schema | empirical | battery | agree | new Lean decls | SCHEMA_DATA | first issue |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[k]) for k in ("iter", "id", "score", "ladder", "proven", "schema", "empirical", "battery", "agree", "new_decls", "schema_data", "err")) + " |")
    best = max(rows, key=lambda r: (r["score"] or 0))
    lines.append(f"\nprograms: {len(rows)}; best score {best['score']} (iter {best['iter']}, {best['id']}); ladders: " +
                 str({l: sum(1 for r in rows if r['ladder'] == l) for l in sorted(set(r['ladder'] for r in rows), key=lambda x: (x is None, x))}))
    out = "\n".join(lines); print(out)
    if a.md:
        open(a.md, "w").write(out + "\n")


if __name__ == "__main__":
    main()

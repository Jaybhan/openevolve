"""E26 genome: a program is a SET of rule atoms, rendered deterministically into a legal evolved
program (initial_program.py's structure: NOTES, LEAN_SOURCE, SCHEMA_DATA, kill).

Atoms (every source is an existing, audited artifact of this repo; nothing new is proved here):
  R  general   the schema recipe, frozen: the 34 Farkas certificates search_certificates returns
               (experiments/E25_reward/results/recipe_certs_frozen.json; E20/E21 "one-line recipe")
  D  specialist DGH(4) proved inline: tests/candidates/lean_dgh4.py LEAN_SOURCE + its Python mirror
  C  specialist one-cell certificates: the 3 pool certificates of the WIDE cell (10,19;3,3) w=99
               (E25 cert_pool; A1's exactly-labelled *_pure_gt table).  Not in R; S1-visible only.
  T  specialist one-cell certificates: the 2 pool certificates of the TARGET (12,18;3,3) w=109.
               Identical to 2 of R's certificates (T is SUBSUMED by R); S0-visible (V0 0.2099).
  B  broken     tests/snippets/broken_proof.lean's `farkasBroken` appended to the candidate (L1)
  U  unsound    tests/snippets/unsound_row7.py's rule `any(r >= 7 for r in rows)` in the Python
               mirror only (fires on a witnessed battery case; Lean unchanged)

The genome is recorded in NOTES as `[E26 genome: <atoms> rev=<k>]`, so the mutator can recover the
parent from the prompt and emit one SEARCH/REPLACE block over the whole EVOLVE block.
"""
from __future__ import annotations

import json
import os
import re
from typing import FrozenSet, Iterable, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))

ATOMS = ("R", "D", "C", "T", "B", "U")
SOUND_ATOMS = ("R", "D", "C", "T")
MARK_RE = re.compile(r"\[E26 genome: ([A-Z+\-]*) rev=(\d+)\]")

_START = "# EVOLVE-BLOCK-START"
_END = "# EVOLVE-BLOCK-END"


def _read(p: str) -> str:
    with open(os.path.join(UB, p), encoding="utf-8") as f:
        return f.read()


def _initial() -> str:
    return _read("initial_program.py")


def _dgh_parts() -> Tuple[str, str]:
    """(Lean declarations of lean_dgh4 without its `candidate`, Python DGH helpers)."""
    src = _read("tests/candidates/lean_dgh4.py")
    lean = re.search(r"LEAN_SOURCE = r'''\n(.*?)'''", src, re.S).group(1)
    cut = lean.index("/-- The evolved library: the proved counting library plus the DGH prune")
    lean_decls = lean[:cut].rstrip() + "\n"
    py = src[src.index("def _dgh_col_kill"):src.index("def kill(")].rstrip() + "\n"
    return lean_decls, py


def _broken_decl() -> str:
    s = _read("tests/snippets/broken_proof.lean")
    a = s.index("def farkasBroken")
    b = s.index("def candidate")
    return s[a:b].rstrip() + "\n"


def _certs():
    with open(os.path.join(HERE, "atoms_certs.json")) as f:
        return json.load(f)


EXAMPLE_PRUNE = '''/-- Pattern for a NEW prune: `kill` on the profile, `sound` via library lemmas. This one kills nothing. -/
def examplePrune (P : Params) : Prune P where
  name := "example (kills nothing)"
  kill := fun _ => false
  sound := by intro A h; simp at h
'''

LIB_KILL = '''def kill(m, n, s, t, w, rows, cols):
    """Python mirror of candidate.kill (library part mirrors ZarPrune.counting)."""
    if sum(comb(c, s) for c in cols) > (t - 1) * comb(m, s):
        return True                                                   # argA
    if sum(comb(r, t) for r in rows) > (s - 1) * comb(n, t):
        return True                                                   # argAT
    r0, c0 = rows[0], cols[0]
    if r0 and sum(comb(c - 1, s - 1) for c in sorted(cols)[:r0] if c) > (t - 1) * comb(m - 1, s - 1):
        return True                                                   # argD
    if c0 and sum(comb(r - 1, t - 1) for r in sorted(rows)[:c0] if r) > (s - 1) * comb(n - 1, t - 1):
        return True                                                   # argDT
'''

DESCR = {"R": "schema recipe (34 frozen Farkas certificates)", "D": "DGH(4) proved inline",
         "C": "certificates for the one wide cell (10,19)", "T": "certificates for the one target (12,18)",
         "B": "an extra prune with a broken proof", "U": "an unsound extra rule in the Python mirror"}


def norm(genome: Iterable[str]) -> FrozenSet[str]:
    return frozenset(a for a in genome if a in ATOMS)


def gstr(genome: Iterable[str]) -> str:
    g = norm(genome)
    return "+".join(a for a in ATOMS if a in g) or "-"


def parse_gstr(s: str) -> FrozenSet[str]:
    return frozenset() if s in ("", "-") else norm(s.split("+"))


def schema_certs(genome: Iterable[str]) -> list:
    g = norm(genome)
    c = _certs()
    out, seen = [], set()
    for a in ("R", "C", "T"):
        if a in g:
            for e in c[a]:
                k = (e["m"], e["n"], e["s"], e["t"], tuple(e["y"]))
                if k not in seen:
                    seen.add(k)
                    out.append({"m": e["m"], "n": e["n"], "s": e["s"], "t": e["t"], "y": list(e["y"])})
    return out


def lean_source(genome: Iterable[str]) -> str:
    g = norm(genome)
    parts, items = [], ["counting P"]
    if "D" in g:
        parts.append(_dgh_parts()[0])
        items.append("argDGH P")
    else:
        parts.append(EXAMPLE_PRUNE)
        items.append("examplePrune P")
    if "B" in g:
        parts.append(_broken_decl())
        items.append("farkasBroken P")
    parts.append("/-- The evolved library. Keep `counting P` first; append new prunes. -/\n"
                 "def candidate (P : Params) : Prune P :=\n  Prune.ofList P [" + ", ".join(items) + "]\n")
    return "\n" + "\n".join(parts)


def evolve_block(genome: Iterable[str], rev: int = 0) -> str:
    g = norm(genome)
    desc = "; ".join(DESCR[a] for a in ATOMS if a in g) or "the proved counting library only"
    notes = f'NOTES = r"""[E26 genome: {gstr(g)} rev={int(rev)}] {desc}."""'
    sd = {"farkas": schema_certs(g), "residue": [], "prefix": []}
    body = [_START, notes, "", "LEAN_SOURCE = r'''" + lean_source(g) + "'''", "",
            "SCHEMA_DATA = " + json.dumps(sd, separators=(",", ":")), "", ""]
    if "D" in g:
        body.append(_dgh_parts()[1])
        body.append("")
    k = LIB_KILL
    tail = []
    if "D" in g:
        tail.append("_dgh_kill(m, n, s, t, w, rows, cols)")
    if "U" in g:
        tail.append("any(r >= 7 for r in rows)")
    if tail:
        k += "    return " + " or ".join(tail) + "\n"
    else:
        k += "    return False                                                      # your prunes go above\n"
    body.append(k.rstrip("\n"))
    body.append(_END)
    return "\n".join(body)


def render(genome: Iterable[str], rev: int = 0) -> str:
    """The full program text for a genome (initial_program.py with its EVOLVE block replaced)."""
    src = _initial()
    a = src.index(_START)
    b = src.index(_END) + len(_END)
    return src[:a] + evolve_block(genome, rev) + src[b:]


def block_of(code: str) -> Optional[str]:
    if _START not in code or _END not in code:
        return None
    a = code.index(_START)
    b = code.index(_END) + len(_END)
    return code[a:b]


def parse(code: str) -> Tuple[FrozenSet[str], int]:
    """(genome, rev) of a program; the initial program (no marker) is (∅, 0)."""
    m = MARK_RE.search(code or "")
    if not m:
        return frozenset(), 0
    return parse_gstr(m.group(1)), int(m.group(2))


if __name__ == "__main__":
    import sys
    g = parse_gstr(sys.argv[1] if len(sys.argv) > 1 else "R+D")
    print(render(g, 1))

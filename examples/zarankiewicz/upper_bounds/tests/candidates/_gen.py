#!/usr/bin/env python3
"""(Re)generate the golden adversarial bank (design §10.1, T-3) from the CURRENT
`initial_program.py`, so every candidate keeps the exact program structure the
evaluator expects (docstring, EVOLVE block, LEAN_SOURCE, Python helpers, kill).

    python tests/candidates/_gen.py            # writes tests/candidates/<name>.py

Each candidate = initial_program.py with (a) a tag line prepended to the module
docstring, (b) LEAN_SOURCE replaced, (c) optionally the `def kill` region replaced.
Candidates owned by other agents (schema_farkas, lean_dgh4, python_only_dgh4) are
never touched.  `tests/golden.json` holds the expected outcome of each candidate.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.abspath(os.path.join(HERE, "..", ".."))
INITIAL = os.path.join(UB, "initial_program.py")
SNIPPETS = os.path.join(UB, "tests", "snippets")
NOT_MINE = {"schema_farkas", "lean_dgh4", "python_only_dgh4"}

LIBRARY = "def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)\n"

KILL_LIBRARY = '''def kill(m, n, s, t, w, rows, cols):
    """Python mirror of `candidate.kill` = baseline || counting."""
    if _baseline(m, n, s, t, w, rows, cols):
        return True
    return any(p(m, n, s, t, w, rows, cols) for p in COUNTING)
'''

# --------------------------------------------------------------------------------------
# name -> (description, LEAN_SOURCE body or None (= keep initial's),
#          extra Python clause replacing the final `return False` of kill(), or None)
# --------------------------------------------------------------------------------------
CANDIDATES = {}


def cand(name, desc, lean, python=None):
    CANDIDATES[name] = (desc, lean, python)


cand("initial", "initial_program.py verbatim (the proved library): verified, kills nothing beyond "
     "the baseline B -> L5, score exactly the verified floor (0.20 under reward v2).", None)

cand("unsound_row7", "Lean = library; Python kill adds 'any row >= 7' which fires on a "
     "witnessed (SAT) case of the battery -> sound_battery 0, score 0.", LIBRARY,
     "any(r >= 7 for r in rows)  # UNSOUND: a realizable profile at (9,10,54) has a row of sum 7")

cand("sorry_in_kill", "a `sorry` as the kill body (not a `have … := by sorry` hole inside "
     "`sound`) -> forbidden -> L0, score 0.", '''def candidate (P : Params) : Prune P where
  name := "sorry in kill"
  kill := fun _ => sorry
  sound := by
    intro A h hv
    exact absurd hv (by simp at h)
''')

cand("sorry_in_sound", "one typed hole `have h3 : … := by sorry` inside `sound` (the only "
     "place sketch mode allows a sorry). The hole is LOAD-BEARING (it is `hv.2`, which none of "
     "the auto-fill tactics can see through `Valid`), so the context is not already contradictory "
     "and omega cannot fill it -> L2, n_holes 1, n_holes_filled 0, score <= 0.19 (v1: scan reject).",
     '''/-- Deficit restated, with the weight lower bound left as a hole. -/
def deficitSketch (P : Params) : Prune P where
  name := "deficit (sketch, one hole)"
  kill := fun pf => decide (sumFin P.m pf.row < P.w)
  sound := by
    intro A h hv
    have h1 : sumFin P.m (rowSum A) < P.w := of_decide_eq_true h
    have h3 : P.w ≤ sumFin P.m (rowSum A) := by sorry
    omega

def candidate (P : Params) : Prune P :=
  Prune.or (baseline P) (Prune.or (counting P) (deficitSketch P))
''')

cand("sketch_two_holes", "two holes inside `sound`: h4 follows from h1 by `omega` (filled), "
     "h3 is the load-bearing `hv.2` no auto-fill tactic can reach -> L3 with n_holes = 2, "
     "n_holes_filled = 1, score <= 0.19. (A hole in an already-contradictory context is filled by "
     "omega trivially, so the unfillable hole must be the one that supplies the contradiction.)",
     '''/-- Deficit restated with two holes; exactly one is omega-fillable from the context. -/
def deficitSketch2 (P : Params) : Prune P where
  name := "deficit (sketch, two holes)"
  kill := fun pf => decide (sumFin P.m pf.row < P.w)
  sound := by
    intro A h hv
    have h1 : sumFin P.m (rowSum A) < P.w := of_decide_eq_true h
    have h4 : sumFin P.m (rowSum A) + 1 ≤ P.w := by sorry
    have h3 : P.w ≤ sumFin P.m (rowSum A) := by sorry
    omega

def candidate (P : Params) : Prune P :=
  Prune.or (baseline P) (Prune.or (counting P) (deficitSketch2 P))
''')

cand("native_decide", "`native_decide` inside the proof (would add the `Lean.ofReduceBool` "
     "axiom / trust the compiler) -> L0, score 0.", '''def candidate (P : Params) : Prune P where
  name := "native"
  kill := fun pf => decide (sumFin P.m pf.row < P.w)
  sound := by
    intro A h hv
    have h1 : sumFin P.m (rowSum A) < P.w := by native_decide
    have h2 : P.w ≤ sumFin P.m (rowSum A) := hv.2
    omega
''')

cand("axiom_smuggle", "a REAL `axiom` hidden from a comment-stripping scanner: the block-"
     "comment delimiters `/-` and `-/` sit inside string literals, so a regex that strips "
     "`/- … -/` deletes the axiom line while Lean elaborates it; the raw-text scan must "
     "still fire -> L0, score 0.", '''/-- helper strings (the comment-stripper sees one big block comment from here …) -/
def tagOpen : String := "/-"
axiom cheat (P : Params) (A : Mat P.m P.n) : ¬ Valid P A
def tagClose : String := "-/"

def candidate (P : Params) : Prune P where
  name := "smuggled axiom"
  kill := fun _ => true
  sound := fun A _ => cheat P A
''')

cand("implemented_by", "`@[implemented_by]` swaps the runtime kill (kills everything) for a "
     "logically trivial one (kills nothing): the #eval mask would lie -> L0, score 0.",
     '''def fastKill (P : Params) : Profile P.m P.n → Bool := fun _ => true

@[implemented_by fastKill]
def logicalKill (P : Params) : Profile P.m P.n → Bool := fun _ => false

def candidate (P : Params) : Prune P where
  name := "implemented_by swap"
  kill := logicalKill P
  sound := by
    intro A h _
    simp [logicalKill] at h
''')

cand("opaque", "an `opaque` kill component (no definitional unfolding, could hide an "
     "`implemented_by`-style body) -> L0, score 0.", '''opaque hiddenKill (P : Params) : Profile P.m P.n → Bool

def candidate (P : Params) : Prune P where
  name := "opaque component"
  kill := fun pf => hiddenKill P pf && decide (sumFin P.m pf.row < P.w)
  sound := by
    intro A h hv
    rw [Bool.and_eq_true] at h
    have h1 : sumFin P.m (rowSum A) < P.w := of_decide_eq_true h.2
    have h2 : P.w ≤ sumFin P.m (rowSum A) := hv.2
    omega
''')

cand("unicode_lookalike", "the keyword `axiom` written with FULLWIDTH letters (U+FF41…): "
     "NFKC normalisation maps it back to `axiom` -> L0; a scanner without NFKC lets it "
     "through to Lean, where it is a parse error (v1: lean_ok 0).",
     '''ａｘｉｏｍ cheat (P : Params) (A : Mat P.m P.n) : ¬ Valid P A

def candidate (P : Params) : Prune P where
  name := "lookalike axiom"
  kill := fun _ => true
  sound := fun A _ => cheat P A
''')

cand("mask_spoof", "prints forged `MASK … BEGIN/MASKLINE/END` lines from inside the "
     "candidate (`#eval IO.println`) to spoof the kill mask -> forbidden `#`/`IO` -> L0, score 0.",
     '''def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)

#eval IO.println "MASK 0 BEGIN"
#eval IO.println "MASKLINE:true,true,true,true,true,true,true,true,true,true,true,true"
#eval IO.println "MASK 0 END"
''')

cand("redefine_valid", "shadows `Valid` inside the candidate namespace with `False` and "
     "'proves' every case empty -> declared-name rule -> L0, score 0 (v1: the wrapper's "
     "`ZarPrune.Valid` is fixed, so the proof does not type-check; lean_ok 0).",
     '''def Valid (P : Params) (A : Mat P.m P.n) : Prop := False

def candidate (P : Params) : Prune P where
  name := "redefined Valid"
  kill := fun _ => true
  sound := fun A _ hv => hv
''')

cand("slow_kill", "a kill that spins through 2^(m+n+4) steps per case before answering "
     "(enumeration-style blow-up) -> #eval times out -> L1 'kill too slow', score <= 0.19. "
     "SLOW: waits for the gate timeout.", '''/-- Tail-recursive counter: `spin k 0 = k`, in `k` interpreter steps. -/
def spin : Nat → Nat → Nat
  | 0, acc => acc
  | k + 1, acc => spin k (acc + 1)

/-- Deficit, but the kill first "enumerates" `2^(m+n+4)` subsets. -/
def slowDeficit (P : Params) : Prune P where
  name := "deficit (slow kill)"
  kill := fun pf =>
    decide (spin (2 ^ (P.m + P.n + 4)) 0 = 2 ^ (P.m + P.n + 4)) && decide (sumFin P.m pf.row < P.w)
  sound := by
    intro A h hv
    rw [Bool.and_eq_true] at h
    have h1 : sumFin P.m (rowSum A) < P.w := of_decide_eq_true h.2
    have h2 : P.w ≤ sumFin P.m (rowSum A) := hv.2
    omega

def candidate (P : Params) : Prune P :=
  Prune.or (baseline P) (Prune.or (counting P) (slowDeficit P))
''')

cand("instance_specific", "`candidate : Prune target` (the gate's first instance only): "
     "type-checks on that instance, fails on every other -> lean_ok = 1/|instances|.",
     '''/-- Instance-specific: `target` is the first suite instance injected by the gate. -/
def candidate : Prune target := Prune.or (baseline target) (counting target)
''')

cand("cond_deletion", "a `CondPrune target [fact]` (design §8.2, lean/ZarPrune/Cond.lean) "
     "that deletes a column of the (9,9,50) instance against the cited fact z(9,8;3,3) = 45 "
     "[Tan 2022]; the unconditional `candidate` is the library. SKIPPED while Cond.lean is "
     "not built. Expected once the gate discharges facts from the ledger: L5, kills the "
     "cases with a column of sum <= 4.", '''/-- z(9,8;3,3) = 45 (Tan 2022, cited): every K_{3,3}-free 9 × 8 matrix has ≤ 45 ones. -/
def fact98 : Fact := { m := 9, n := 8, s := 3, t := 3, z := 45, tag := "tan2022" }

/-- Conditional deletion prune for the first suite instance (9,9;3,3) at w = 50:
kill when some column has `c_j + 45 < 50`. -/
def condDelCol : CondPrune target [fact98] where
  name := "delete a column vs z(9,8;3,3)=45 [tan2022]"
  kill := fun pf => (List.finRange target.n).any (fun j => decide (pf.col j + fact98.z < target.w))
  sound := by
    intro hF A h
    have hf : FactHolds fact98 := hF fact98 (List.mem_singleton.mpr rfl)
    exact (argDelCol target fact98.z (fun B hB => hf B hB)).sound A h

/-- The unconditional part: the proved library. -/
def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)
''')

cand("cond_candidateF", "the CondPrune ENTRY POINT (gate v2.1, design §2.1/§8.2): `candidateF : CondPrune "
     "target [z(9,8;3,3)=45 tan2022]` beside the library `candidate`. The gate discharges it against the "
     "facts the ledger grants per instance: on the pure TRAIN table (9,9,50) no fact is granted, so the "
     "conditional part is `CondPrune.never` (artifact lean_cond) -> L5, exactly the floor 0.20; through "
     "`python -m zar_ub gate 9 9 3 3 50 FILE --trust tan2022` the fact is granted and cond_name is the prune.",
     '''/-- z(9,8;3,3) = 45 (Tan 2022, cited): every K_{3,3}-free 9 × 8 matrix has ≤ 45 ones. -/
def fact98 : Fact := { m := 9, n := 8, s := 3, t := 3, z := 45, tag := "tan2022" }

/-- Conditional deletion prune for the first suite instance (9,9;3,3) at w = 50. -/
def condDelCol : CondPrune target [fact98] where
  name := "delete a column vs z(9,8;3,3)=45 [tan2022]"
  kill := fun pf => (List.finRange target.n).any (fun j => decide (pf.col j + fact98.z < target.w))
  sound := by
    intro hF A h
    have hf : FactHolds fact98 := hF fact98 (List.mem_singleton.mpr rfl)
    exact (argDelCol target fact98.z (fun B hB => hf B hB)).sound A h

/-- The conditional entry point: discharged by the gate where `fact98` is granted. -/
def candidateF : CondPrune target [fact98] := condDelCol

/-- The unconditional part: the proved library. -/
def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)
''')

cand("notDescending", "symmetry breaking disguised as a prune (kill cases whose row sums "
     "are not non-increasing): no soundness proof exists, the attempted `omega` fails -> "
     "L1, score <= 0.19. The Python mirror never fires on the (sorted) tables, so the "
     "battery passes; only the gate can reject it.", '''/-- NOT a prune: rows not sorted non-increasingly. Killing such a case is the *adding*
move (symmetry breaking), which needs a permutation witness, not an emptiness proof. -/
def notDescending (P : Params) : Prune P where
  name := "rows not non-increasing (symmetry break)"
  kill := fun pf => !allFin P.m (fun i => allFin P.m (fun j =>
      decide (i.val < j.val → pf.row j ≤ pf.row i)))
  sound := by
    intro A h hv
    omega

def candidate (P : Params) : Prune P :=
  Prune.or (baseline P) (Prune.or (counting P) (notDescending P))
''', "any(rows[i] < rows[i + 1] for i in range(len(rows) - 1))  # never true on the (sorted) tables")


_LEAN_RE = re.compile(r"LEAN_SOURCE = r'''\n?(.*?)'''", re.S)
_KILL_RE = re.compile(r"def kill\(.*?(?=# EVOLVE-BLOCK-END)", re.S)
_RET_RE = re.compile(r"^(\s*)return False\b[^\n]*$", re.M)


def _with_extra_clause(kill_region: str, extra: str) -> str:
    """Replace the LAST `return False` of the Python kill with `return <extra>`: the mirror
    of the proved library stays exactly what initial_program.py ships, plus one rule."""
    hits = list(_RET_RE.finditer(kill_region))
    assert hits, "initial_program.py: the kill() body must end with a `return False` line"
    last = hits[-1]
    return kill_region[:last.start()] + f"{last.group(1)}return {extra}" + kill_region[last.end():]


def render(name: str, src: str) -> str:
    desc, lean, python = CANDIDATES[name]
    assert _LEAN_RE.search(src), "initial_program.py: LEAN_SOURCE block not found"
    out = src
    if lean is not None:
        out = _LEAN_RE.sub(lambda _m: "LEAN_SOURCE = r'''\n" + lean.rstrip("\n") + "\n'''", out, count=1)
    if python is not None:
        assert _KILL_RE.search(out), "initial_program.py: def kill … # EVOLVE-BLOCK-END not found"
        out = _KILL_RE.sub(lambda m: _with_extra_clause(m.group(0), python), out, count=1)
    tag = f'"""[GOLDEN CANDIDATE `{name}` -- tests/golden.json] {desc}\n\nGenerated by tests/candidates/_gen.py from initial_program.py; do not edit by hand.\n\n'
    assert out.startswith('"""'), "initial_program.py must start with a module docstring"
    return tag + out[3:]


def main(argv=None):
    src = open(INITIAL, encoding="utf-8").read()
    names = argv[1:] if argv and len(argv) > 1 else list(CANDIDATES)
    for name in names:
        if name in NOT_MINE:
            continue
        path = os.path.join(HERE, f"{name}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(name, src))
        print("wrote", os.path.relpath(path, UB))


if __name__ == "__main__":
    main(sys.argv)

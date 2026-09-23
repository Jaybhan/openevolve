"""The Lean gate: the only way a candidate prune becomes trusted (design §4.2-4.4).

A candidate is a Lean 4 source fragment (no imports; it is spliced into a fixed
wrapper inside `namespace ZarPrune.Cand`) that must define

    def candidate (P : Params) : Prune P        -- general prune, or
    def candidate : Prune target                -- instance-specific prune

where `target : Params` (and `target1`, `target2`, ... for the other suite
instances) is injected by the gate.  Optionally (design §2.1/§8.2) it may also
define a conditional prune

    def candidateF (P : Params) (facts : List Fact) : CondPrune P facts   -- or
    def candidateF : CondPrune target [f1, f2, ...]                       -- literal facts

which the wrapper discharges against `gateFactsK`, the facts the run's ledger
grants to instance K (`ledger.facts_for(inst, table trust)`: [] on pure tables);
a literal fact list must be a sub-list (checked by `decide`), otherwise the
conditional part is `CondPrune.never` on that instance (reported as cond_name).

Procedure (S0-S7 of design §4.2):
  S0 normalise   NFKC copy for the scan; line/block comments and string literals
                 stripped (nesting-aware) for the comment-stripped scan.
  S1 scan        FORBIDDEN tokens (design §4.3) on the NFKC text, the raw text and
                 both comment-stripped versions; the sorry-hole rule; the
                 declared-name rule; the set_option whitelist.   Any hit -> L0.
  S2 cache       key = sha1(wrapper version, source, schema terms, sketch flag,
                 instances + case lists); cache/gate/<key>.json.
  S3 wrapper     import ZarPrune / set_option autoImplicit false / namespace
                 ZarPrune.Cand / targetK / candidate verbatim / end / footer with
                 candInstK, schemaK, gateInstK, #print axioms, gateProfileK and the
                 mask blocks guarded by a per-evaluation nonce.
  S4 elaborate   `lake env lean <file>` in lean/, hard timeout, process-group kill.
  S5 ladder      L0..L5 exactly as §4.2 (see `_ladder`).
  S6 masks       only the text between `MASK <nonce> K BEGIN/END`; length must equal
                 the case count.
  S7 auto-fill   deterministic tactic list, each under maxHeartbeats 50000, all tried
                 in ONE extra Lean run: every hole becomes
                   first | (have ZHOLE_k : True := trivial; trace_state; fail)
                         | (set_option maxHeartbeats 50000 in (tac_i; done); trace "ZFILL k i") | ...
                         | sorry
                 (the goal probe goes through the trace state, which survives `first`'s
                 backtracking, so the goal of every hole is recorded even when a later
                 tactic hits a resource limit); a hole is filled by the first tactic whose
                 `ZFILL` line appears with no error on the hole's lines.  If every hole
                 fills, the filled copy re-enters S3-S6 as a normal candidate (its source
                 is `filled_source`); a partial fill is L3, none is L2.

NFKC note.  NFKC folds legal, distinct Lean identifier characters (ℕ→N, h₁→h1,
𝒜→A), so the text handed to Lean is the raw candidate; the NFKC copy is scanned
in addition to the raw text, so nothing Lean sees escapes the scan.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

from .known import Instance

_HERE = os.path.dirname(os.path.abspath(__file__))
LEAN_DIR = os.path.join(os.path.dirname(_HERE), "lean")
GATE_CACHE_DIR = os.path.join(os.path.dirname(_HERE), "cache", "gate")
GATE_SECRET_PATH = os.path.join(GATE_CACHE_DIR, ".secret")  # 0600; the candidate sandbox denies reading it


def _gate_secret() -> Optional[bytes]:
    """Per-installation secret used to MAC every cache entry (docs/build/ATTACKS.md: a
    candidate that could write the cache dir could otherwise forge an L5 result)."""
    try:
        with open(GATE_SECRET_PATH, "rb") as fh:
            sec = fh.read()
        if len(sec) >= 32:
            return sec
    except OSError:
        pass
    try:
        os.makedirs(GATE_CACHE_DIR, exist_ok=True)
        sec = os.urandom(32)
        fd = os.open(GATE_SECRET_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(sec)
        return sec
    except OSError:
        return None


def _mac(secret: bytes, key: str, payload: str) -> str:
    import hmac
    return hmac.new(secret, (key + "\x00" + payload).encode(), "sha256").hexdigest()
WRAPPER_VERSION = "gate-v2.1"  # v2.1: CondPrune entry point (candidateF) + per-instance facts
ALLOWED_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}
SORRY_AXIOM = "sorryAx"
MAX_HEARTBEATS = 400000
MAX_RECDEPTH = 4096
FILL_HEARTBEATS = 50000
MASK_CHUNK = 200

# ---------------------------------------------------------------------------
# S1: forbidden constructs (design §4.3, final list).  (regex, reported name)
# Every pattern is applied to the NFKC text, the raw text and both comment-
# stripped versions; a hit anywhere (including comments and strings) is L0.
# `sorry` is handled by the hole rule, `set_option` by the whitelist.
# ---------------------------------------------------------------------------
FORBIDDEN: List[Tuple[str, str]] = [
    (r"\badmit\b", "admit"),
    (r"\bnative_decide\b", "native_decide"),
    (r"\baxiom\b", "axiom"),
    (r"\bunsafe\b", "unsafe"),
    (r"implemented_by", "implemented_by"),
    (r"\bextern\b", "extern"),
    (r"\bcsimp\b", "csimp"),
    (r"\bopaque\b", "opaque"),
    (r"\bpartial\b", "partial"),
    (r"^\s*import\b", "import"),
    (r"\bmacro\b", "macro"),
    (r"\bmacro_rules\b", "macro_rules"),
    (r"\belab\b", "elab"),
    (r"\belab_rules\b", "elab_rules"),
    (r"\bsyntax\b", "syntax"),
    (r"\bnotation\b", "notation"),
    (r"\binitialize\b", "initialize"),
    (r"\bLean\.", "Lean. (meta namespace)"),
    (r"\bIO\b", "IO"),
    (r"\bofReduceBool\b", "ofReduceBool"),
    (r"\bofReduceNat\b", "ofReduceNat"),
    (r"\bend\s+(ZarPrune|Cand)\b", "end ZarPrune/Cand (escaping the sandbox namespace)"),
    # added by design §4.3
    (r"\+\s*native\b", "+native"),
    (r"\+\s*kernel\b", "+kernel"),
    (r"\btrustCompiler\b", "trustCompiler"),
    (r"\brun_cmd\b", "run_cmd"),
    (r"\brun_tac\b", "run_tac"),
    (r"\brun_elab\b", "run_elab"),
    (r"^\s*open\b[^\n]*\bLean\b", "open Lean"),
    (r"\battribute\s*\[", "attribute ["),
    (r"@\[\s*simp\b", "@[simp]"),
    (r"@\[\s*csimp\b", "@[csimp]"),
    (r"@\[\s*implemented_by\b", "@[implemented_by]"),
    (r"@\[\s*extern\b", "@[extern]"),
    (r"\b(local|scoped)\s+instance\b", "local/scoped instance"),
    (r"\binstance\b[^\n]*?:\s*(Decidable|DecidableEq|DecidablePred|DecidableRel)\b", "instance : Decidable"),
    (r"\bnoncomputable\b", "noncomputable"),
    (r"\bdbg_trace\b", "dbg_trace"),
    (r"\btrace(_state)?\b", "trace"),
    (r"\btrace\s*\[", "trace["),
    (r"\blogInfo\b", "logInfo"),
    (r"\blogWarning\b", "logWarning"),
    (r"#[A-Za-z_]", "# command (#eval/#print/#check/... ; also the `#s` card notation: use `.card`)"),
    (r"\bnamespace\b", "namespace"),
    (r"\bderiving\s+instance\b", "deriving instance"),
    (r"\bdecreasing_by\b", "decreasing_by"),
]
_DERIVING_OK = {"Repr", "DecidableEq"}
_DERIVING = re.compile(r"\bderiving\s+([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*)")
_SET_OPTION = re.compile(r"\bset_option\b")
_SET_OPTION_OK = re.compile(r"\bset_option\s+(maxHeartbeats|maxRecDepth)\s+(\d+)(\s+in\b|\s*$)", re.M)
_SET_OPTION_LIMITS = {"maxHeartbeats": MAX_HEARTBEATS, "maxRecDepth": MAX_RECDEPTH}
# declared-name rule (S1): wrapper/library names the candidate may not (re)declare
PROTECTED_NAMES = (r"Valid|HasKst|Params|Profile|Mat|weight|rowSum|colSum|profileOf|Prune|CondPrune|Fact|FactHolds|"
                   r"counting|baseline|evolved|target\d*|gateInst\d*|gateProfile\d*|schema\d*|candInst\d*|"
                   r"gateFacts\d*|condF\d*|gateKill\d*|gateKill_eq\d*")
# the conditional entry point (design §2.1 / §8.2): `candidateF` is a CondPrune, discharged by the wrapper
_HAS_CANDF = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable)\s+)*(?:def|abbrev)\s+candidateF\b", re.M)
_DECL_KW = r"(?:def|theorem|lemma|abbrev|structure|inductive|instance|class|example|opaque|axiom)"
_DECLARED_NAME = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable|unsafe|partial)\s+)*"
                            rf"{_DECL_KW}\s+({PROTECTED_NAMES})\b", re.M)
_DECL_LINE = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected)\s+)?"
                        r"(theorem|def|lemma|instance|example|abbrev|structure|inductive|class)\b")
_SORRY = re.compile(r"\bsorry\b")
_HOLE_TAIL = re.compile(r":=\s*by\s+sorry\s*$")
_HOLE_TAIL_BY = re.compile(r":=\s*by\s*$")
_HOLE_HAVE = re.compile(r"^(\s*)have\s+([A-Za-z_][\w'.]*)\s*(?::|$)")
_SOUND_BY = re.compile(r"^(\s*)sound\s*:=\s*by\s*$")
_THEOREM_HEAD = re.compile(r"^(?:@\[[^\]]*\]\s*)?(?:(?:private|protected)\s+)?(?:theorem|lemma)\b")

# S7 auto-fill tactic list (design §4.2), in this order; deterministic, $0.
FILL_LEMMAS = ["colBudget", "rowBudget", "rowLocalBudget", "weight_deleteCol", "weight_deleteRow",
               "choose_tangent", "sum_le_waterfillBound", "budget_general", "hasKst_of_subsets"]
FILL_TACTICS: List[str] = ["omega", "simp_all", "decide", "linarith", "nlinarith", "positivity", "grind"] + \
    [f"exact {L}" for L in FILL_LEMMAS] + \
    [f"(apply {L} <;> first | assumption | exact hv.1 | exact hv.2 | omega)" for L in FILL_LEMMAS]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    ok: bool = False                     # L5: the prune is TRUSTED on this instance
    ladder: int = 0                      # 0..5, design §4.2 S5 (per instance; the suite ladder is
                                         # reward.combine_ladders: L0/L4 anywhere dominate, else max — §8.1)
    scanned_ok: bool = False
    compiled: bool = False               # no elaboration error inside the candidate (sorry warnings allowed)
    typed_ok: bool = False               # gateInstK : Prune targetK elaborated
    axioms: List[str] = field(default_factory=list)
    axioms_ok: bool = False              # axioms ⊆ ALLOWED_AXIOMS (sorryAx never ok)
    kill_mask: Optional[List[bool]] = None
    schema_mask: Optional[List[bool]] = None
    cond_name: Optional[str] = None      # name of the discharged CondPrune (candidateF) on this instance; None = no candidateF
    n_facts: int = 0                     # facts injected for this instance (ledger.facts_for)
    errors: List[str] = field(default_factory=list)
    holes: List[dict] = field(default_factory=list)   # {index, line, statement, goal, filled_by}
    n_holes: int = 0
    n_holes_filled: int = 0
    filled_source: Optional[str] = None
    forbidden: List[str] = field(default_factory=list)
    n_decls: int = 0
    n_decls_ok: int = 0
    first_error_line: Optional[int] = None
    first_error_frac: float = 0.0        # 0..1: how far into the candidate the first error is
    seconds: float = 0.0
    timed_out: bool = False
    lean_file: str = ""
    stdout_tail: str = ""
    cache_hit: bool = False
    nonce: str = ""
    parse_error: bool = False
    instance_tag: str = ""

    @property
    def lean_partial(self) -> float:
        """Design §5.3, exact."""
        if self.ladder >= 5:
            return 1.0
        if self.ladder in (0, 4):
            return 0.0
        n_body = max(self._n_body, 1)
        depth = 1.0 if self.first_error_line is None else (self.first_error_line - 1) / n_body
        declfrac = (self.n_decls_ok / self.n_decls) if self.n_decls else 0.0
        fill = (self.n_holes_filled / self.n_holes) if self.n_holes else 1.0
        if self.ladder == 1:
            v = 0.05 + 0.10 * declfrac + 0.05 * depth
        elif self.ladder == 2:
            v = 0.25 + 0.15 * declfrac + 0.10 * depth
        else:
            v = 0.50 + 0.30 * fill + 0.10 * depth
        return round(v, 4)

    @property
    def partial_credit(self) -> float:  # backward-compatible alias (evaluator.py v1)
        return self.lean_partial

    _n_body: int = 0

    def as_dict(self):
        d = asdict(self)
        d.pop("_n_body", None)
        d["lean_partial"] = self.lean_partial
        d["partial_credit"] = self.lean_partial
        for key in ("kill_mask", "schema_mask"):
            if d[key] is not None:
                d[key + "_true"] = int(sum(d[key]))
                d[key] = None
        return d

    def to_cache(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_cache(d: dict) -> "GateResult":
        return GateResult(**{k: v for k, v in d.items() if k in GateResult.__dataclass_fields__})


# ---------------------------------------------------------------------------
# S0: normalisation and comment stripping
# ---------------------------------------------------------------------------
def nfkc(src: str) -> str:
    return unicodedata.normalize("NFKC", src)


def strip_comments(src: str) -> str:
    """Replace line comments, (nested) block comments and string literals by
    spaces, keeping every newline so line numbers are preserved."""
    out = []
    i, n = 0, len(src)
    depth = 0
    while i < n:
        c = src[i]
        if depth > 0:
            if src.startswith("/-", i):
                depth += 1; out.append("  "); i += 2; continue
            if src.startswith("-/", i):
                depth -= 1; out.append("  "); i += 2; continue
            out.append("\n" if c == "\n" else " "); i += 1; continue
        if src.startswith("/-", i):
            depth = 1; out.append("  "); i += 2; continue
        if src.startswith("--", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i)); i = j; continue
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"' and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            j = min(j + 1, n)
            out.append('"' + " " * max(j - i - 2, 0) + ('"' if j - i >= 2 else ""))
            i = j; continue
        out.append(c); i += 1
    return "".join(out)


def _scan_text(text: str, where: str) -> List[str]:
    hits = []
    for pat, name in FORBIDDEN:
        if re.search(pat, text, flags=re.M):
            hits.append(f"{name} ({where})")
    for m in _DERIVING.finditer(text):
        bad = [x.strip() for x in m.group(1).split(",") if x.strip() not in _DERIVING_OK]
        if bad:
            hits.append(f"deriving {','.join(bad)} (only Repr, DecidableEq allowed) ({where})")
    for m in _SET_OPTION.finditer(text):
        ok = _SET_OPTION_OK.match(text, m.start())
        if not ok:
            hits.append(f"set_option (only maxHeartbeats/maxRecDepth allowed) ({where})")
        elif int(ok.group(2)) > _SET_OPTION_LIMITS[ok.group(1)]:
            hits.append(f"set_option {ok.group(1)} {ok.group(2)} > {_SET_OPTION_LIMITS[ok.group(1)]} ({where})")
    m = _DECLARED_NAME.search(text)
    if m:
        hits.append(f"declared protected name {m.group(1)} ({where})")
    return hits


def _inside_sound(lines: List[str], i: int) -> bool:
    """Is line i (0-based) inside a `sound := by` tactic block (deeper-indented than it,
    with no intervening line at or below the block's indentation)?"""
    ind_i = len(lines[i]) - len(lines[i].lstrip())
    s = i - 1
    while s >= 0:
        if _DECL_LINE.match(lines[s]):
            return False
        sm = _SOUND_BY.match(lines[s])
        if sm:
            si = len(sm.group(1))
            return si < ind_i and all(not lines[k].strip() or (len(lines[k]) - len(lines[k].lstrip())) > si
                                      for k in range(s + 1, i))
        s -= 1
    return False


def find_holes(src_nc: str) -> Tuple[List[dict], List[str]]:
    """Sorry-hole rule (S1): every `sorry` must be the whole body of
    `have <id> : <T> := by sorry` (the `sorry` may also stand alone on the next
    line after `:= by`) inside a `sound := by` block or inside a
    `theorem … : … → ¬ Valid …` body.  Returns (holes, violations)."""
    lines = src_nc.splitlines()
    holes, bad = [], []

    def reject(i: int, msg: str) -> None:
        if _inside_sound(lines, i):
            holes.append({"index": len(holes), "line": i + 1, "end_line": i + 1, "statement": "",
                          "goal": "untyped sorry (not auto-filled): write it as `have h : <T> := by sorry`",
                          "filled_by": None, "untyped": True})
        else:
            bad.append(msg)

    for i, line in enumerate(lines):
        if not _SORRY.search(line):
            continue
        if len(_SORRY.findall(line)) > 1:
            bad.append(f"line {i + 1}: more than one sorry"); continue
        end = i
        if _HOLE_TAIL.search(line):
            start = i
        elif line.strip() == "sorry" and i > 0 and _HOLE_TAIL_BY.search(lines[i - 1]):
            start = i - 1
        else:
            # E22 relaxation: a `sorry` anywhere inside a `sound := by` block is an UNTYPED hole
            # (never trusted: sorryAx; not auto-filled; ladder capped at L2); elsewhere it is L0.
            reject(i, f"line {i + 1}: sorry outside a `sound := by` block (and not a `have <id> : <T> := by sorry` hole)"); continue
        # walk back to the `have` line (multi-line statements: deeper-indented continuation lines)
        j = start
        while j >= 0 and not _HOLE_HAVE.match(lines[j]):
            if not lines[j].strip() or j < start and ":=" in lines[j]:
                j = -1; break
            j -= 1
        if j < 0:
            reject(i, f"line {i + 1}: sorry not attached to a `have` statement"); continue
        hm = _HOLE_HAVE.match(lines[j])
        have_indent = len(hm.group(1))
        if any(lines[k].strip() and (len(lines[k]) - len(lines[k].lstrip())) <= have_indent for k in range(j + 1, end + 1)):
            reject(i, f"line {i + 1}: malformed multi-line have statement"); continue
        stmt = " ".join(l.strip() for l in lines[j:end + 1])
        stmt = re.sub(r":=\s*by\s+sorry\s*$", "", stmt).strip()
        stmt = re.sub(r":=\s*by\s*$", "", stmt).strip()
        if ":" not in stmt[len("have"):]:
            reject(i, f"line {i + 1}: hole `have` has no type ascription"); continue
        # enclosure: nearest preceding top-level declaration
        d = j
        while d >= 0 and not _DECL_LINE.match(lines[d]):
            d -= 1
        if d < 0:
            reject(i, f"line {i + 1}: hole outside any declaration"); continue
        enclosed = False
        if _THEOREM_HEAD.match(lines[d]):
            head = []
            for k in range(d, j):
                head.append(lines[k])
                if ":=" in lines[k]:
                    break
            htxt = " ".join(head).split(":=")[0]
            enclosed = ("¬" in htxt and "Valid" in htxt)
            if not enclosed:
                reject(i, f"line {i + 1}: hole in a theorem whose statement is not `… → ¬ Valid …`"); continue
        else:
            s = j - 1
            while s > d:
                sm = _SOUND_BY.match(lines[s])
                if sm:
                    si = len(sm.group(1))
                    if si < have_indent and all(not lines[k].strip() or (len(lines[k]) - len(lines[k].lstrip())) > si
                                                for k in range(s + 1, j)):
                        enclosed = True
                    break
                s -= 1
            if not enclosed:
                reject(i, f"line {i + 1}: hole not inside a `sound := by` block"); continue
        holes.append({"index": len(holes), "line": j + 1, "end_line": end + 1, "statement": stmt,
                      "goal": "", "filled_by": None})
    return holes, bad


def static_scan(src: str, sketch: bool = True) -> List[str]:
    """S0+S1.  Returns the sorted list of violations (empty = passes)."""
    raw = src.replace("\r\n", "\n")
    norm = nfkc(raw)
    raw_nc, norm_nc = strip_comments(raw), strip_comments(norm)
    hits: List[str] = []
    for text, where in ((norm_nc, "code"), (raw_nc, "code/raw"), (norm, "comment or string"), (raw, "comment or string/raw")):
        hits.extend(_scan_text(text, where))
    # sorry: the hole rule on the comment-stripped NFKC text; the raw/comment copies
    # may not contain any sorry that the stripped text does not (count equality)
    n_nc, n_raw, n_norm = len(_SORRY.findall(norm_nc)), len(_SORRY.findall(raw)), len(_SORRY.findall(norm))
    if n_nc or n_raw or n_norm:
        if not sketch:
            hits.append("sorry")
        else:
            holes, bad = find_holes(norm_nc)
            hits.extend(f"sorry hole rule: {b}" for b in bad)
            if n_raw != n_nc or n_norm != n_nc:
                hits.append("sorry (in a comment or string, or hidden from the comment stripper)")
    # collapse "(code)" and "(code/raw)" duplicates of the same construct
    seen, out = set(), []
    for h in hits:
        key = re.sub(r" \((code|code/raw|comment or string|comment or string/raw)\)$", "", h)
        if key not in seen:
            seen.add(key)
            out.append(h)
    return sorted(out)


# ---------------------------------------------------------------------------
# Lean output parsing
# ---------------------------------------------------------------------------
# Lean 4.34 prints named diagnostics as `error(lean.unknownIdentifier): ...`
_ERR_HEAD = re.compile(r"^(.*?):(\d+):(\d+): error(?:\([^)]*\))?: (.*)$")
_DIAG_HEAD = re.compile(r"^.*?:\d+:\d+: (error|warning|info)(?:\([^)]*\))?:")
_MSG_STOP = re.compile(r"^('|S?MASK |MASKLINE:|ZFILL |ZHOLE_)")  # harness output lines never belong to a message
_PARSE_ERR = re.compile(r"unexpected token|unexpected end of input|expected (term|command|token)|unterminated|"
                        r"unknown tactic|invalid 'end'|expected '")


def _parse_errors(out: str, offset: int, max_msg_lines: int = 12) -> List[Tuple[int, str]]:
    """(line-in-candidate, message) per Lean error; the message continues until
    the next diagnostic header or a blank line."""
    errs: List[Tuple[int, str]] = []
    lines = out.splitlines()
    i = 0
    while i < len(lines):
        m = _ERR_HEAD.match(lines[i])
        if m:
            msg = [m.group(4).strip()]
            j = i + 1
            while j < len(lines) and len(msg) < max_msg_lines and lines[j].strip() and not _DIAG_HEAD.match(lines[j]) \
                    and not _MSG_STOP.match(lines[j]):
                msg.append(lines[j].rstrip())
                j += 1
            errs.append((int(m.group(2)) - offset, " ⏎ ".join(msg)))
            i = j
        else:
            i += 1
    return errs


def _decl_lines(src: str) -> List[int]:
    return [i + 1 for i, l in enumerate(src.splitlines()) if _DECL_LINE.match(l)]


def _profile_literal(rows: Sequence[int], cols: Sequence[int]) -> str:
    return "([" + ",".join(map(str, rows)) + "],[" + ",".join(map(str, cols)) + "])"


# ---------------------------------------------------------------------------
# S3: wrapper
# ---------------------------------------------------------------------------
def _header(inst: Instance) -> List[str]:
    return [
        "import ZarPrune",
        "set_option autoImplicit false",
        "namespace ZarPrune",
        "namespace Cand",
        f"abbrev target : Params := {{ m := {inst.m}, n := {inst.n}, s := {inst.s}, t := {inst.t}, w := {inst.w} }}",
        "",
    ]


def build_gate_file(inst: Instance, candidate_src: str, cases: Optional[List[Tuple[Sequence[int], Sequence[int]]]],
                    chunk: int = MASK_CHUNK,
                    extra_instances: Optional[List[Tuple[Instance, Optional[List[Tuple[Sequence[int], Sequence[int]]]]]]] = None,
                    nonce: str = "0" * 16, schema_terms: Optional[List[Optional[List[str]]]] = None,
                    masks: bool = True, facts: Optional[List[Optional[List[str]]]] = None) -> Tuple[str, int, List[Tuple[int, str]]]:
    """Return (lean source, line offset of the candidate, footer block map).

    The primary instance is `target`; instance k >= 1 is `targetK`.  Per
    instance the footer defines candInstK (the candidate at targetK), schemaK
    (Prune.ofList of the schema terms, if any), gateInstK (= candInstK, or
    Prune.or candInstK schemaK), prints its axioms, and evaluates the kill
    mask of gateInstK (and of schemaK alone) between nonce-guarded markers.
    The block map lists (absolute line, tag) for every footer line so that
    errors in the footer can be attributed to an instance and a block.

    Conditional entry point (design §2.1/§8.2): when the candidate declares
    `candidateF` (a `CondPrune`), the footer also defines per instance
    `gateFactsK : List Fact` (facts[k]: the Lean terms of ledger.facts_for),
    `condFK : CondPrune targetK gateFactsK` (candidateF applied to / weakened to
    the injected facts by `decide`d membership; `CondPrune.never` when the
    candidate's facts are not granted on this instance), `gateInstK (hF) :=
    Prune.or candInstK (condFK.discharge hF)`, the closed `gateKillK` with
    `theorem gateKill_eqK : (gateInstK hF).kill pf = gateKillK pf := rfl`, and a
    `COND <nonce> K <name>` line; the mask is evaluated through gateKillK."""
    header = _header(inst)
    has_condF = bool(_HAS_CANDF.search(candidate_src))
    offset = len(header)
    body = candidate_src.rstrip("\n").splitlines()
    footer: List[Tuple[str, str]] = [("", "trailer"), ("end Cand", "trailer"), ("end ZarPrune", "trailer"),
                                     ("", "trailer"), ("-- ===== gate checks (generated) =====", "trailer")]
    all_insts = [(inst, cases)] + list(extra_instances or [])
    for k, (ik, ck) in enumerate(all_insts):
        sfx = "" if k == 0 else str(k)
        T = f"ZarPrune.Cand.target{sfx}"
        C, S, G, PF = (f"ZarPrune.Cand.{x}{sfx}" for x in ("candInst", "schema", "gateInst", "gateProfile"))
        if k > 0:
            footer.append((f"abbrev {T} : ZarPrune.Params := {{ m := {ik.m}, n := {ik.n}, s := {ik.s}, t := {ik.t}, w := {ik.w} }}", f"target{k}"))
        footer += [(f"def {C} : ZarPrune.Prune {T} := by", f"gateInst{k}"),
                   ("  first", f"gateInst{k}"),
                   ("  | exact ZarPrune.Cand.candidate", f"gateInst{k}"),
                   (f"  | exact ZarPrune.Cand.candidate {T}", f"gateInst{k}")]
        terms = (schema_terms[k] if schema_terms and k < len(schema_terms) else None) or []
        if terms:
            footer.append((f"def {S} : ZarPrune.Prune {T} := ZarPrune.Prune.ofList _ [{', '.join(terms)}]", f"schema{k}"))
        base_term = f"ZarPrune.Prune.or {C} {S}" if terms else C
        gate_kill = f"{G}.kill"
        if has_condF:
            FK, CF, GK = (f"ZarPrune.Cand.{x}{sfx}" for x in ("gateFacts", "condF", "gateKill"))
            fk = (facts[k] if facts and k < len(facts) else None) or []
            footer += [(f"def {FK} : List ZarPrune.Fact := [{', '.join(fk)}]", f"cond{k}"),
                       (f"def {CF} : ZarPrune.CondPrune {T} {FK} := by", f"cond{k}"),
                       ("  first", f"cond{k}"),
                       (f"  | exact ZarPrune.Cand.candidateF {T} {FK}", f"cond{k}"),
                       ("  | exact ZarPrune.CondPrune.weaken ZarPrune.Cand.candidateF (by decide)", f"cond{k}"),
                       (f"  | exact ZarPrune.CondPrune.weaken (ZarPrune.Cand.candidateF {T}) (by decide)", f"cond{k}"),
                       ("  | exact ZarPrune.CondPrune.never _ _", f"cond{k}"),
                       (f"def {G} (hF : ∀ f ∈ {FK}, ZarPrune.FactHolds f) : ZarPrune.Prune {T} := "
                        f"ZarPrune.Prune.or ({base_term}) ({CF}.discharge hF)", f"gateInst{k}"),
                       (f"def {GK} : ZarPrune.Profile {T}.m {T}.n → Bool := fun pf => ({base_term}).kill pf || {CF}.kill pf", f"gateInst{k}"),
                       (f"theorem {GK}_eq : ∀ hF pf, ({G} hF).kill pf = {GK} pf := fun _ _ => rfl", f"gateInst{k}"),
                       (f'#eval IO.println ("COND {nonce} {k} " ++ {CF}.name)', f"cond{k}")]
            gate_kill = GK
        else:
            footer.append((f"def {G} : ZarPrune.Prune {T} := {base_term}", f"gateInst{k}"))
        footer.append((f"#print axioms {G}", f"axioms{k}"))
        if ck and masks:
            footer.append((f"def {PF} (r c : List Nat) : ZarPrune.Profile {T}.m {T}.n :=", f"profile{k}"))
            footer.append(("  { row := fun i => r.getD i.val 0, col := fun j => c.getD j.val 0 }", f"profile{k}"))
            for prune, marker, tag in ((gate_kill, "MASK", f"mask{k}"),) + (((f"{S}.kill", "SMASK", f"smask{k}"),) if terms else ()):
                footer.append((f'#eval IO.println "{marker} {nonce} {k} BEGIN"', tag))
                for start in range(0, len(ck), chunk):
                    lits = ",\n  ".join(_profile_literal(r, c) for r, c in ck[start:start + chunk])
                    # one IO.println string per chunk: a raw `#eval` of a `List Bool` goes through the
                    # pretty-printer, which wraps and truncates long lists with `⋯` (E8 verifier bug).
                    footer.append((f'#eval IO.println ("MASKLINE:" ++ String.intercalate "," ((([\n  {lits}] : List (List Nat × List Nat)).map fun p => if {prune} ({PF} p.1 p.2) then "true" else "false")))', tag))
                footer.append((f'#eval IO.println "{marker} {nonce} {k} END"', tag))
    lines = header + body + [t for t, _ in footer]
    first_footer = offset + len(body) + 1
    blocks = []
    ln = first_footer
    for text, tag in footer:
        n_lines = text.count("\n") + 1
        for _ in range(n_lines):
            blocks.append((ln, tag)); ln += 1
    return "\n".join(lines) + "\n", offset, blocks


# ---------------------------------------------------------------------------
# S4: run Lean
# ---------------------------------------------------------------------------
def _limit_memory():  # pragma: no cover - Linux only (macOS does not enforce RLIMIT_AS)
    if sys.platform.startswith("linux"):
        try:
            import resource
            lim = 4 * 1024 ** 3
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except Exception:
            pass


def run_lean(path: str, timeout: float) -> Tuple[str, bool]:
    """`lake env lean <path>` in LEAN_DIR (writes nothing; never `lake build`).
    Returns (combined output, timed_out).  The whole process group is killed on
    timeout so no orphan `lean` keeps running."""
    proc = subprocess.Popen(["lake", "env", "lean", path], cwd=LEAN_DIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True,
                            preexec_fn=_limit_memory if sys.platform.startswith("linux") else None)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return out or "", False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            out, _ = proc.communicate(timeout=5)
        except Exception:
            out = ""
        return out or "", True


def _write_candidate(src: str, tag: str, suffix: str = "") -> str:
    h = hashlib.sha1(src.encode()).hexdigest()[:10]
    cand_dir = os.path.join(LEAN_DIR, "Candidates")
    os.makedirs(cand_dir, exist_ok=True)
    path = os.path.join(cand_dir, f"cand_{tag + '_' if tag else ''}{h}{suffix}.lean")
    with open(path, "w") as f:
        f.write(src)
    return path


# ---------------------------------------------------------------------------
# S2: cache
# ---------------------------------------------------------------------------
def _cases_hash(instances) -> str:
    h = hashlib.sha1()
    for inst, cases in instances:
        h.update(f"{inst.m},{inst.n},{inst.s},{inst.t},{inst.w}|".encode())
        if cases is None:
            h.update(b"none;")
        else:
            for r, c in cases:
                h.update((",".join(map(str, r)) + "/" + ",".join(map(str, c)) + ";").encode())
    return h.hexdigest()


_LIB_HASH: Dict[str, str] = {}


def library_hash() -> str:
    """sha1 over the Lean library sources (lean/ZarPrune.lean + lean/ZarPrune/*.lean), computed once per
    process: a cached gate result must not outlive a change of the library it elaborated against."""
    if "v" not in _LIB_HASH:
        h = hashlib.sha1()
        root = os.path.join(LEAN_DIR, "ZarPrune")
        files = [os.path.join(LEAN_DIR, "ZarPrune.lean")] + sorted(
            os.path.join(root, f) for f in (os.listdir(root) if os.path.isdir(root) else []) if f.endswith(".lean"))
        for f in files:
            try:
                with open(f, "rb") as fh:
                    h.update(os.path.basename(f).encode() + b"\x00" + fh.read() + b"\x00")
            except OSError:
                pass
        _LIB_HASH["v"] = h.hexdigest()[:16]
    return _LIB_HASH["v"]


def cache_key(candidate_src: str, schema_terms, instances, sketch: bool, facts=None) -> str:
    h = hashlib.sha1()
    for part in (WRAPPER_VERSION, library_hash(), nfkc(candidate_src), candidate_src,
                 json.dumps(schema_terms, sort_keys=True), _cases_hash(instances), str(bool(sketch)),
                 json.dumps(facts, sort_keys=True) if facts else ""):
        h.update(part.encode()); h.update(b"\x00")
    return h.hexdigest()


def _cache_load(key: str, cache_dir: str) -> Optional[List[GateResult]]:
    path = os.path.join(cache_dir, key + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            d = json.load(fh)
        if d.get("version") != WRAPPER_VERSION:
            return None
        sec = _gate_secret()
        if sec is None:
            return None
        payload = json.dumps(d["results"], sort_keys=True)
        import hmac as _hmac
        if not _hmac.compare_digest(str(d.get("mac", "")), _mac(sec, key, payload)):
            try:
                os.remove(path)  # forged or stale: never trust it
            except OSError:
                pass
            return None
        return [GateResult.from_cache(r) for r in d["results"]]
    except Exception:
        return None


def _cache_store(key: str, cache_dir: str, results: List[GateResult]) -> None:
    try:
        os.makedirs(cache_dir, exist_ok=True)
        sec = _gate_secret()
        if sec is None:
            return
        res = [r.to_cache() for r in results]
        mac = _mac(sec, key, json.dumps(res, sort_keys=True))
        tmp = os.path.join(cache_dir, key + ".tmp")
        with open(tmp, "w") as f:
            json.dump({"version": WRAPPER_VERSION, "key": key, "results": res, "mac": mac}, f)
        os.replace(tmp, os.path.join(cache_dir, key + ".json"))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# S5/S6: per-instance analysis and the ladder
# ---------------------------------------------------------------------------
def _axioms_of(out: str, gname: str) -> Optional[List[str]]:
    m = re.search(rf"'{re.escape(gname)}' (depends on axioms: \[([^\]]*)\]|does not depend on any axioms)", out)
    if not m:
        return None
    return [a.strip() for a in (m.group(2) or "").split(",") if a.strip()]


def _parse_mask(out: str, marker: str, nonce: str, k: int, n_cases: int) -> Tuple[Optional[List[bool]], str]:
    seg = re.search(rf"^{marker} {nonce} {k} BEGIN\n(.*?)^{marker} {nonce} {k} END\s*$", out, flags=re.S | re.M)
    if not seg:
        return None, f"{marker.lower()} block for instance {k} missing (kill did not evaluate)"
    mask: List[bool] = []
    for line in seg.group(1).splitlines():
        m = re.match(r"^MASKLINE:((?:true|false)(?:,(?:true|false))*)\s*$", line)
        if not m:
            return None, f"unexpected text inside the {marker.lower()} block of instance {k}: {line[:80]!r}"
        mask.extend(x == "true" for x in m.group(1).split(","))
    if len(mask) != n_cases:
        return None, f"kill mask length {len(mask)} != cases {n_cases} (instance {k})"
    return mask, ""


def _ladder(r: GateResult, sketch: bool, holes_left: bool, holes_filled_some: bool) -> int:
    if not r.scanned_ok or r.parse_error:
        return 0
    if r.timed_out and not r._progress:
        return 0  # timeout with nothing observable elaborated
    if not r.compiled or not r.typed_ok:
        return 1
    extra = set(r.axioms) - ALLOWED_AXIOMS
    if r.n_holes > 0 and sketch:
        if extra and extra != {SORRY_AXIOM}:
            return 4
        if not holes_left:
            return 5 if r.axioms_ok and r.kill_mask is not None else 1  # filled copy re-gated
        return 3 if holes_filled_some else 2
    if extra:
        return 4  # includes sorryAx with no declared hole = a construct slipped S1
    if r.kill_mask is None and r._cases_expected:
        return 1
    return 5


# ---------------------------------------------------------------------------
# S7: auto-fill
# ---------------------------------------------------------------------------
def _fill_chain(k: int, tactics: List[str]) -> str:
    """`first | <goal probe> | <tac_0> | ... | sorry`.  The goal probe records the
    hole's goal through `trace_state` (the trace state survives `first`'s
    backtracking, the message log does not) with a marker hypothesis
    `ZHOLE_k : True` in the context, then fails; each tactic runs under
    maxHeartbeats 50000 and, on success, logs `ZFILL k i`."""
    alts = [f"(have ZHOLE_{k} : True := trivial; trace_state; fail)"]
    alts += [f'(set_option maxHeartbeats {FILL_HEARTBEATS} in ({t}; done); trace "ZFILL {k} {i}")'
             for i, t in enumerate(tactics)]
    alts.append("sorry")
    return "first | " + " | ".join(alts)


_GOAL_MARK = re.compile(r"^ZHOLE_(\d+) : True\s*$", re.M)


def _goal_blocks(out: str) -> Dict[int, str]:
    """Split the trace_state dumps of the fill run into per-hole goal texts: a
    block runs from the line after the previous block's `⊢` goal (plus its
    indented continuation lines) to the end of this block's goal; the marker
    hypothesis `ZHOLE_k : True` names the hole and is removed."""
    goals: Dict[int, str] = {}
    block: List[str] = []
    in_goal = False
    for line in out.splitlines():
        if _DIAG_HEAD.match(line) or line.startswith(("ZFILL ", "'")):
            continue
        if in_goal and not line.startswith(" "):
            m = _GOAL_MARK.search("\n".join(block))
            if m:
                k = int(m.group(1))
                goals[k] = "\n".join(l for l in block if not _GOAL_MARK.match(l)).strip()[:4000]
            block, in_goal = [], False
        block.append(line)
        if line.startswith("⊢"):
            in_goal = True
    if block:
        m = _GOAL_MARK.search("\n".join(block))
        if m:
            goals[int(m.group(1))] = "\n".join(l for l in block if not _GOAL_MARK.match(l)).strip()[:4000]
    return goals


def _replace_hole(lines: List[str], hole: dict, replacement: str) -> None:
    """Replace the `sorry` of a hole (in place, same line count)."""
    e = hole["end_line"] - 1
    if _HOLE_TAIL.search(lines[e]):
        lines[e] = _HOLE_TAIL.sub(f":= by {replacement}", lines[e])
    else:  # `sorry` alone on its line
        lines[e] = re.sub(r"\bsorry\b", replacement, lines[e], count=1)


def autofill(inst: Instance, candidate_src: str, holes: List[dict], timeout: float, tag: str,
             tactics: Optional[List[str]] = None) -> Tuple[List[dict], Optional[str], str, bool]:
    """One Lean run that tries every tactic on every hole (design §4.2 S7).
    Returns (holes with goal/filled_by, filled source or None if nothing
    filled, probe path, timed_out).  A hole counts as filled only when its
    `ZFILL` trace appeared AND no error was reported on its lines."""
    tactics = tactics or FILL_TACTICS
    lines = candidate_src.rstrip("\n").splitlines()
    typed = [h for h in holes if not h.get("untyped")]
    for h in holes:
        if h.get("untyped"):
            h["filled_by"] = None
    if not typed:
        return holes, None, "", False
    for h in typed:
        _replace_hole(lines, h, _fill_chain(h["index"], tactics))
    header = _header(inst)
    header[1:2] = ["set_option autoImplicit false", "set_option linter.unusedTactic false",
                   "set_option linter.unreachableTactic false"]
    src = "\n".join(header + lines + ["", "end Cand", "end ZarPrune"]) + "\n"
    path = _write_candidate(src, tag, "_fill")
    out, timed_out = run_lean(path, timeout)
    errs = _parse_errors(out, len(header))
    goals = _goal_blocks(out)
    filled_any = False
    for h in typed:
        k = h["index"]
        m = re.search(rf"^ZFILL {k} (\d+)\s*$", out, flags=re.M)
        here = [msg for ln, msg in errs if h["line"] <= ln <= h["end_line"]]
        if m and not here and not timed_out:
            h["filled_by"] = tactics[int(m.group(1))]
            h["goal"] = ""
            filled_any = True
        else:
            h["filled_by"] = None
            h["goal"] = goals.get(k, "")
            if here:
                h["goal"] = (h["goal"] + "\n" if h["goal"] else "") + \
                    "[a fill tactic hit a resource limit: " + "; ".join(x[:120] for x in here)[:400] + "]"
    filled_src = None
    if filled_any:
        flines = candidate_src.rstrip("\n").splitlines()
        for h in typed:
            if h["filled_by"]:
                _replace_hole(flines, h, f"set_option maxHeartbeats {FILL_HEARTBEATS} in {h['filled_by']}")
        filled_src = "\n".join(flines) + "\n"
    return holes, filled_src, path, timed_out


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------
def run_gate_multi(instances: List[Tuple[Instance, Optional[List[Tuple[Sequence[int], Sequence[int]]]]]],
                   candidate_src: str, schema_terms: Optional[List[Optional[List[str]]]] = None,
                   timeout: float = 240.0, tag: str = "", sketch: bool = True, keep: bool = True,
                   use_cache: bool = True, cache_dir: Optional[str] = None,
                   _fill_depth: int = 0, facts: Optional[List[Optional[List[str]]]] = None) -> List[GateResult]:
    """One Lean process for a whole suite; one GateResult per instance.
    schema_terms[k] = Lean term strings of type `Prune targetK` OR-ed into
    gateInstK (and evaluated alone as schema_mask).
    facts[k] = Lean terms of type `Fact` (ledger.Fact.lean) granted to instance k:
    the hypotheses a `candidateF : CondPrune` may use there (see build_gate_file)."""
    t0 = time.time()
    instances = list(instances)
    if not instances:
        return []
    cache_dir = cache_dir or GATE_CACHE_DIR
    candidate_src = candidate_src.replace("\r\n", "\n")
    if not candidate_src.endswith("\n"):
        candidate_src += "\n"
    key = cache_key(candidate_src, schema_terms, instances, sketch, facts)
    if use_cache:
        cached = _cache_load(key, cache_dir)
        if cached is not None and len(cached) == len(instances):
            for r in cached:
                r.cache_hit = True
                r.seconds = time.time() - t0
            return cached

    body_lines = candidate_src.rstrip("\n").splitlines()
    n_body = len(body_lines)
    base = GateResult()
    base._n_body = n_body
    base.nonce = os.urandom(8).hex()
    base.forbidden = static_scan(candidate_src, sketch=sketch)
    base.scanned_ok = not base.forbidden
    base.n_decls = len(_decl_lines(candidate_src))
    if not base.scanned_ok:
        base.errors = [f"forbidden construct: {x}" for x in base.forbidden]
        base.ladder = 0
        out: List[GateResult] = []
        for inst, _ in instances:
            r = GateResult.from_cache(base.to_cache())
            r.instance_tag, r.seconds = inst.tag, time.time() - t0
            out.append(r)
        return out
    holes = find_holes(strip_comments(nfkc(candidate_src)))[0] if sketch else []
    base.holes, base.n_holes = holes, len(holes)

    inst0, cases0 = instances[0]
    has_condF = bool(_HAS_CANDF.search(candidate_src))
    src, offset, blocks = build_gate_file(inst0, candidate_src, cases0, extra_instances=instances[1:],
                                          nonce=base.nonce, schema_terms=schema_terms, masks=not holes, facts=facts)
    path = _write_candidate(src, tag)
    base.lean_file = path
    out_text, base.timed_out = run_lean(path, timeout)
    slow_kill = False
    if base.timed_out:
        base.errors.append(f"lean timed out after {timeout}s")
        # Lean buffers stdout, so a SIGKILLed process leaves no output at all (measured: the
        # `#print axioms` line of a slow-kill candidate never arrives).  To tell design §4.2's
        # L1 "kill too slow" from L0 "nothing elaborated", re-run the elaboration alone once
        # (same header/body, footer without the mask blocks).  Never cached (timed_out stays).
        if not holes and any(ck for _, ck in instances):
            src_e, _, blocks_e = build_gate_file(inst0, candidate_src, cases0, extra_instances=instances[1:],
                                                 nonce=base.nonce, schema_terms=schema_terms, masks=False, facts=facts)
            path_e = _write_candidate(src_e, (tag + "_elab") if tag else "elab")
            t_e = time.time()
            out_e, to_e = run_lean(path_e, timeout)
            if not to_e:
                out_text, blocks, slow_kill = out_e, blocks_e, True
                base.errors.append(f"kill too slow: the mask evaluation did not finish within {timeout}s "
                                   f"(elaboration alone succeeded in {time.time() - t_e:.1f}s)")
    base.stdout_tail = out_text[-4000:]

    # --- candidate errors --------------------------------------------------
    err_lines = _parse_errors(out_text, offset)
    cand_errs = [(ln, msg) for ln, msg in err_lines if 1 <= ln <= n_body]
    foot_errs = [(ln + offset, msg) for ln, msg in err_lines if not (1 <= ln <= n_body)]
    block_of = dict(blocks)
    foot_tagged = [(block_of.get(ln, "trailer"), msg) for ln, msg in foot_errs]
    base.errors.extend(f"line {ln}: {msg}" for ln, msg in cand_errs[:20])
    base.parse_error = any(_PARSE_ERR.search(msg) for _, msg in cand_errs) or \
        any(t == "trailer" and _PARSE_ERR.search(msg) for t, msg in foot_tagged)
    base.compiled = not cand_errs and not base.timed_out
    decls = _decl_lines(candidate_src)
    if decls:
        bad = set()
        for ln, _ in cand_errs:
            owner = max([d for d in decls if d <= ln], default=None)
            if owner is not None:
                bad.add(owner)
        base.n_decls_ok = len(decls) - len(bad)
    if cand_errs:
        base.first_error_line = min(ln for ln, _ in cand_errs)
        base.first_error_frac = (base.first_error_line - 1) / max(n_body, 1)
    else:
        base.first_error_frac = 1.0

    # --- S7 auto-fill (only for a candidate that compiles with holes) ---------
    holes_filled_some, all_filled, filled_src = False, False, None
    if holes and base.compiled and _fill_depth == 0:
        holes, filled_src, _, fill_to = autofill(inst0, candidate_src, holes, timeout, tag)
        base.holes = holes
        base.n_holes_filled = sum(1 for h in holes if h["filled_by"])
        holes_filled_some = base.n_holes_filled > 0
        all_filled = base.n_holes_filled == len(holes)
        base.filled_source = filled_src
        if fill_to:
            base.errors.append("auto-fill probe timed out")
        if all_filled and filled_src:
            # the filled copy re-enters S3-S6 as a normal candidate (design S7)
            res = run_gate_multi(instances, filled_src, schema_terms=schema_terms, timeout=timeout,
                                 tag=(tag + "_filled") if tag else "filled", sketch=False, keep=keep,
                                 use_cache=use_cache, cache_dir=cache_dir, _fill_depth=1, facts=facts)
            for r in res:
                r.holes, r.n_holes, r.n_holes_filled = holes, len(holes), len(holes)
                r.filled_source = filled_src
                r.nonce, r.cache_hit = base.nonce, False
                r.seconds = time.time() - t0
                r.errors = [f"[filled copy] {e}" for e in r.errors]
            if use_cache and not any(r.timed_out for r in res):
                _cache_store(key, cache_dir, res)
            return res

    # --- per instance ----------------------------------------------------------
    results: List[GateResult] = []
    for k, (ik, ck) in enumerate(instances):
        r = GateResult.from_cache(base.to_cache())
        r.errors = list(base.errors)
        r.instance_tag = ik.tag
        r._cases_expected = bool(ck)
        r._progress = slow_kill or bool(cand_errs) or ("depends on axioms" in out_text) or \
            ("does not depend on any axioms" in out_text)
        gname = "ZarPrune.Cand.gateInst" + ("" if k == 0 else str(k))
        inst_errs = [(t, m) for t, m in foot_tagged if t in (f"gateInst{k}", f"schema{k}", f"target{k}", f"axioms{k}", f"cond{k}")]
        r.axioms = _axioms_of(out_text, gname) or []
        r.n_facts = len((facts[k] if facts and k < len(facts) else None) or [])
        if has_condF:
            mc = re.search(rf"^COND {base.nonce} {k} (.*)$", out_text, flags=re.M)
            r.cond_name = mc.group(1).strip() if mc else None
        r.typed_ok = base.compiled and not inst_errs and _axioms_of(out_text, gname) is not None
        if inst_errs:
            r.errors.extend(f"gate check (instance {k}, {t}): {m}" for t, m in inst_errs[:5])
        r.axioms_ok = r.typed_ok and set(r.axioms) <= ALLOWED_AXIOMS
        if ck and not holes:
            if r.typed_ok:
                mask_errs = [m for t, m in foot_tagged if t in (f"mask{k}", f"profile{k}")]
                r.kill_mask, err = _parse_mask(out_text, "MASK", base.nonce, k, len(ck))
                if r.kill_mask is None:
                    r.errors.append((f"kill mask (instance {k}): " + mask_errs[0][:300]) if mask_errs else err)
                terms = (schema_terms[k] if schema_terms and k < len(schema_terms) else None) or []
                if terms:
                    r.schema_mask, serr = _parse_mask(out_text, "SMASK", base.nonce, k, len(ck))
                    if r.schema_mask is None:
                        r.errors.append("schema " + serr)
        r.ladder = _ladder(r, sketch, holes_left=bool(holes) and not all_filled, holes_filled_some=holes_filled_some)
        if r.ladder == 4 and not r.errors:
            r.errors.append(f"axioms outside the allowed set: {sorted(set(r.axioms) - ALLOWED_AXIOMS)}")
        r.ok = r.ladder == 5 and r.axioms_ok and (not ck or r.kill_mask is not None)
        r.seconds = time.time() - t0
        results.append(r)
    if not keep and os.path.exists(path):
        os.remove(path)
    if use_cache and not base.timed_out:
        _cache_store(key, cache_dir, results)
    return results


# per-result flags used by the ladder (not dataclass fields; not cached)
GateResult._cases_expected = False
GateResult._progress = False


def run_gate(inst: Instance, candidate_src: str,
             cases: Optional[List[Tuple[Sequence[int], Sequence[int]]]] = None,
             timeout: float = 120.0, keep: bool = True, tag: str = "",
             schema_terms: Optional[List[str]] = None, sketch: bool = True, use_cache: bool = True,
             facts: Optional[List[str]] = None) -> GateResult:
    """Single-instance gate (CLI)."""
    return run_gate_multi([(inst, cases)], candidate_src, schema_terms=[schema_terms] if schema_terms else None,
                          timeout=timeout, tag=tag, sketch=sketch, keep=keep, use_cache=use_cache,
                          facts=[facts] if facts else None)[0]

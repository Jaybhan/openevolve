"""Fact ledger: provenance-tagged upper bounds z(m,n;s,t) <= bound (design §8.2).

Two files under data/:

  ledger.csv       rows (m, n, s, t, bound, kind, provenance, closure_file, hypotheses)
                   kind        in {exact, ub}
                   provenance  in PROVENANCE_TIER (lean-here / tan2022 / bhan26 / collins16)
                   closure_file  for lean-here rows: the certificate manifest / closure report
                   hypotheses  '|'-separated facts a *conditional* closure rests on
                               ("tan2022:z(10,10;3,3)<=60|tan2022:z(9,20;3,3)<=93"); empty for
                               unconditional rows; free-text items are notes
  claims_2026.csv  the frontier block of docs/literature_review.md §2.2 (dfield / Hou /
                   Afrasyab / Saurabh / Wang values).  Read ONLY to choose targets and, for
                   the verified lower bounds, as a witness sanity check; never a Fact.

`facts_for(P, trust)` returns the exact list of `Fact`s the partition generator
(`zar_ub/partitions.py`, Argument I on proper prefixes) and the conditional prunes
(`argDelColF` / `argDelRowF` / `Prune.ofPrefixF` in lean/ZarPrune/Cond.lean) can use
for the instance `P` under a trust level, so that a closure theorem's hypotheses are
exactly the facts that were load-bearing.  A fact is listed only when it *tightens*
the first-principles counting bound (`ub_counting`), mirroring `known.ub_known`,
which records a table lookup in its `Ledger` only in that case.

`check_claims()` is the mechanical consistency checker (monotonicity in m and n,
line deletion, transposition symmetry, counting bound, verified-witness comparison);
`write_ledger()` runs it before every write and refuses to write an inconsistent
ledger.

CLI:  python -m zar_ub.ledger check
      python -m zar_ub.ledger facts M N S T W [--trust tan2022]
      python -m zar_ub.ledger seed          (regenerate data/ledger.csv and data/claims_2026.csv)
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .known import Instance, ub_counting

_HERE = os.path.dirname(os.path.abspath(__file__))
UB_DIR = os.path.dirname(_HERE)
DATA_DIR = os.path.join(UB_DIR, "data")
LEDGER_PATH = os.path.join(DATA_DIR, "ledger.csv")
CLAIMS_PATH = os.path.join(DATA_DIR, "claims_2026.csv")

LEDGER_FIELDS = ["m", "n", "s", "t", "bound", "kind", "provenance", "closure_file", "hypotheses"]
CLAIMS_FIELDS = [
    "m",
    "n",
    "s",
    "t",
    "lb",
    "ub",
    "status",
    "lb_source",
    "ub_source",
    "lb_verified",
    "ub_reviewed",
    "reviewed_ub",
    "reviewed_ub_source",
    "notes",
]
KINDS = ("exact", "ub")

# Trust tiers.  A trust level admits every provenance whose tier is <= the level's tier.
#   lean-here : closed by this pipeline (Lean prunes + LRAT certificates; see the
#               closure_file).  Until lean/ZarPrune/Closure.lean (design §4.6) lands the
#               closure itself is a checked report, not a Lean theorem.
#   tan2022   : Tan 2022 Table 3 (bold = exact, non-bold = Roman upper bound); the
#               (11,21) and (12,22) rows are Tan's upper bounds made exact by bhan26 witnesses.
#   bhan26    : Bhan-Nobili-Raghuraman-Langer 2026 (reproduced witnesses); tier as tan2022.
#   collins16 : Collins et al. 2016 Table 4, transcribed from the literature review, not
#               re-derived here: a separate, opt-in tier.
PROVENANCE_TIER: Dict[str, int] = {"lean-here": 0, "tan2022": 1, "bhan26": 1, "collins16": 2}
TRUST_LEVELS: Dict[str, int] = {"pure": -1, "lean-here": 0, "tan2022": 1, "collins16": 2}
DEFAULT_TRUST = "tan2022"


class LedgerError(ValueError):
    """Raised by write_ledger when the claim checker finds an inconsistency."""


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------
@dataclass(frozen=True, order=True)
class Fact:
    """`z(m,n;s,t) <= z`, with its provenance tag.  Mirrors `ZarPrune.Fact`."""

    m: int
    n: int
    s: int
    t: int
    z: int
    tag: str

    @property
    def lean(self) -> str:
        """The Lean term of type `ZarPrune.Fact`."""
        return f"⟨{self.m}, {self.n}, {self.s}, {self.t}, {self.z}, {json.dumps(self.tag)}⟩"

    @property
    def key(self) -> Tuple[int, int, int, int]:
        return (self.m, self.n, self.s, self.t)

    def transpose(self) -> "Fact":
        return Fact(self.n, self.m, self.t, self.s, self.z, self.tag)

    def __str__(self) -> str:
        return f"{self.tag}:z({self.m},{self.n};{self.s},{self.t})<={self.z}"


def lean_fact_list(facts: Sequence[Fact]) -> str:
    """`[⟨…⟩, ⟨…⟩]` — the `facts` argument of a `CondPrune`."""
    return "[" + ", ".join(f.lean for f in facts) + "]"


# ---------------------------------------------------------------------------
# Ledger rows
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LedgerRow:
    m: int
    n: int
    s: int
    t: int
    bound: int
    kind: str
    provenance: str
    closure_file: str = ""
    hypotheses: str = ""

    @property
    def key(self) -> Tuple[int, int, int, int]:
        return (self.m, self.n, self.s, self.t)

    @property
    def tier(self) -> int:
        return PROVENANCE_TIER.get(self.provenance, 99)

    def fact(self) -> Fact:
        return Fact(self.m, self.n, self.s, self.t, self.bound, self.provenance)

    def hypothesis_facts(self) -> List[Fact]:
        """The machine-readable items of `hypotheses` (`tag:z(m,n;s,t)<=z`); items that
        do not match the fact grammar are free-text notes and are ignored."""
        return [parse_fact(h) for h in self.hypotheses.split("|") if _FACT_RE.match(h.strip())]

    def notes(self) -> List[str]:
        return [
            h.strip()
            for h in self.hypotheses.split("|")
            if h.strip() and not _FACT_RE.match(h.strip())
        ]


_FACT_RE = re.compile(r"^([\w.-]+):z\((\d+),(\d+);(\d+),(\d+)\)<=(\d+)$")


def parse_fact(text: str) -> Fact:
    """Inverse of `str(Fact)`: `tag:z(m,n;s,t)<=z`."""
    mo = _FACT_RE.match(text.strip())
    if not mo:
        raise ValueError(f"not a fact: {text!r}")
    tag, m, n, s, t, z = mo.groups()
    return Fact(int(m), int(n), int(s), int(t), int(z), tag)


def _row_from_dict(d: dict) -> LedgerRow:
    return LedgerRow(
        int(d["m"]),
        int(d["n"]),
        int(d["s"]),
        int(d["t"]),
        int(d["bound"]),
        d["kind"].strip(),
        d["provenance"].strip(),
        (d.get("closure_file") or "").strip(),
        (d.get("hypotheses") or "").strip(),
    )


def load_ledger(path: str = LEDGER_PATH) -> List[LedgerRow]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [_row_from_dict(d) for d in csv.DictReader(f)]


def load_claims(path: str = CLAIMS_PATH) -> List[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            r = dict(d)
            for k in ("m", "n", "s", "t", "lb", "ub", "lb_verified", "ub_reviewed"):
                r[k] = int(r[k]) if r.get(k, "") != "" else None
            r["reviewed_ub"] = int(r["reviewed_ub"]) if r.get("reviewed_ub") else None
            out.append(r)
    return out


def _sorted(rows: Iterable[LedgerRow]) -> List[LedgerRow]:
    return sorted(rows, key=lambda r: (r.s, r.t, r.m, r.n, r.tier, r.kind, r.bound, r.provenance))


def write_ledger(
    rows: Iterable[LedgerRow], path: str = LEDGER_PATH, claims: Optional[List[dict]] = None
) -> List[LedgerRow]:
    """Run the claim checker, then write.  Refuses (LedgerError) on any error."""
    rows = _sorted(rows)
    errors = check_claims(rows, claims)
    if errors:
        raise LedgerError(
            "ledger not written; %d inconsistencies:\n  " % len(errors) + "\n  ".join(errors)
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    os.replace(tmp, path)
    return rows


def add_rows(new_rows: Iterable[LedgerRow], path: str = LEDGER_PATH) -> List[LedgerRow]:
    """Append (or replace an identical (m,n,s,t,kind,provenance) row) and write."""
    rows = {(r.key, r.kind, r.provenance): r for r in load_ledger(path)}
    for r in new_rows:
        rows[(r.key, r.kind, r.provenance)] = r
    return write_ledger(rows.values(), path)


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
def _admitted(rows: Iterable[LedgerRow], trust: str) -> List[LedgerRow]:
    if trust not in TRUST_LEVELS:
        raise ValueError(f"unknown trust level {trust!r}; one of {sorted(TRUST_LEVELS)}")
    level = TRUST_LEVELS[trust]
    return [r for r in rows if r.tier <= level]


def best_row(
    m: int,
    n: int,
    s: int,
    t: int,
    trust: str = DEFAULT_TRUST,
    rows: Optional[List[LedgerRow]] = None,
) -> Optional[LedgerRow]:
    """The tightest admitted ledger row for z(m,n;s,t) (symmetric lookup: z(n,m;t,s) too).
    Ties are broken towards the more trusted provenance."""
    rows = load_ledger() if rows is None else rows
    best = None
    for r in _admitted(rows, trust):
        if r.key == (m, n, s, t) or r.key == (n, m, t, s):
            if best is None or (r.bound, r.tier) < (best.bound, best.tier):
                best = r
    return best


def ub_ledger(
    m: int,
    n: int,
    s: int,
    t: int,
    trust: str = DEFAULT_TRUST,
    rows: Optional[List[LedgerRow]] = None,
) -> Tuple[int, Optional[Fact]]:
    """min(counting bound, ledger bound).  Returns (bound, fact-or-None); the fact is
    non-None exactly when the ledger tightened the counting bound -- the same rule as
    `known.ub_known`, which notes a lookup only when it tightens."""
    if m < s or n < t:
        return m * n, None
    ub = ub_counting(m, n, s, t)
    r = best_row(m, n, s, t, trust, rows)
    if r is not None and r.bound < ub:
        f = r.fact()
        if r.key != (m, n, s, t):
            f = f.transpose()  # orient the fact the way the caller asked
        return r.bound, f
    return ub, None


def facts_for(
    P: Instance, trust: str = DEFAULT_TRUST, rows: Optional[List[LedgerRow]] = None
) -> List[Fact]:
    """The Facts the generator and the conditional prunes use for `P`:

      * Argument I, column prefixes: z(m, k) for 1 <= k < n   (`Prune.ofPrefixF P k f`)
      * Argument I, row prefixes:    z(k, n) for 1 <= k < m   (`Prune.ofPrefixRowF P k f`)
      * deletion neighbours (m, n-1) and (m-1, n) (`argDelColF` / `argDelRowF`) -- these are
        the k = n-1 and k = m-1 prefixes, so they need no separate entry.

    Only facts that tighten the counting bound are listed (see `ub_ledger`); `pure`
    returns [].  Sorted, duplicate-free, oriented as (m,k) / (k,n)."""
    rows = load_ledger() if rows is None else rows
    if trust not in TRUST_LEVELS:
        raise ValueError(f"unknown trust level {trust!r}; one of {sorted(TRUST_LEVELS)}")
    if TRUST_LEVELS[trust] < 0:
        return []
    out = set()
    for k in range(1, P.n):
        _, f = ub_ledger(P.m, k, P.s, P.t, trust, rows)
        if f is not None:
            out.add(f)
    for k in range(1, P.m):
        _, f = ub_ledger(k, P.n, P.s, P.t, trust, rows)
        if f is not None:
            out.add(f)
    return sorted(out)


def all_facts_hold_hypotheses(facts: Sequence[Fact]) -> str:
    """The hypotheses column of a conditional closure (`|`-separated facts)."""
    return "|".join(str(f) for f in facts)


# ---------------------------------------------------------------------------
# The mechanical claim checker
# ---------------------------------------------------------------------------
def check_claims(
    rows: Optional[List[LedgerRow]] = None, claims: Optional[List[dict]] = None
) -> List[str]:
    """Consistency checks over the ledger; returns a list of error strings (empty = ok).

    With E(m,n) an exact row and B(m,n) any row (exact or ub) of the same (s,t):
      1. field validity (kind, provenance tier, positivity, bound <= m*n, no duplicate
         (m,n,s,t,kind,provenance) rows, hypotheses parse and are themselves ledgered)
      2. E(m,n) <= ub_counting(m,n,s,t)
      3. monotonicity      E(m,n) <= B(m,n+1),   E(m,n) <= B(m+1,n)
      4. line deletion     E(m,n+1) <= B(m,n) + m,   E(m+1,n) <= B(m,n) + n
      5. transposition     rows at (m,n,s,t) and (n,m,t,s): exact = exact, exact <= ub
      6. verified witnesses (claims_2026.csv, lb_verified = 1): B(m,n) >= lb
      7. two exact rows for one cell agree
    """
    rows = load_ledger() if rows is None else list(rows)
    claims = load_claims() if claims is None else claims
    errs: List[str] = []
    seen = set()
    by_cell: Dict[Tuple[int, int, int, int], List[LedgerRow]] = {}
    for r in rows:
        tag = f"z({r.m},{r.n};{r.s},{r.t}) {r.kind} {r.provenance}"
        if r.kind not in KINDS:
            errs.append(f"{tag}: unknown kind")
        if r.provenance not in PROVENANCE_TIER:
            errs.append(f"{tag}: unknown provenance (2026 claims belong in claims_2026.csv)")
        if min(r.m, r.n, r.s, r.t) <= 0 or r.bound < 0:
            errs.append(f"{tag}: non-positive field")
        if r.bound > r.m * r.n:
            errs.append(f"{tag}: bound {r.bound} exceeds m*n")
        k = (r.key, r.kind, r.provenance)
        if k in seen:
            errs.append(f"{tag}: duplicate row")
        seen.add(k)
        by_cell.setdefault(r.key, []).append(r)
        hyps = r.hypothesis_facts()
        if r.provenance == "lean-here" and r.notes():
            errs.append(f"{tag}: lean-here hypotheses must be facts only, got notes {r.notes()}")
        for h in hyps:
            if h.tag not in PROVENANCE_TIER:
                errs.append(f"{tag}: hypothesis {h} has unknown provenance")
            if r.provenance == "lean-here" and not any(
                (q.key == h.key or q.key == h.transpose().key)
                and q.bound <= h.z
                and q.provenance == h.tag
                for q in rows
            ):
                errs.append(f"{tag}: hypothesis {h} is not a ledger row")
        if r.provenance == "lean-here" and not r.closure_file:
            errs.append(f"{tag}: lean-here row without closure_file")

    def exact(m, n, s, t):
        return [q for q in by_cell.get((m, n, s, t), []) if q.kind == "exact"]

    def anyrow(m, n, s, t):
        return by_cell.get((m, n, s, t), [])

    for r in rows:
        m, n, s, t = r.key
        tag = f"z({m},{n};{s},{t})"
        if r.kind == "exact":
            if m >= s and n >= t and r.bound > ub_counting(m, n, s, t):
                errs.append(f"{tag} exact {r.bound} > counting bound {ub_counting(m, n, s, t)}")
            for q in anyrow(m, n + 1, s, t):
                if r.bound > q.bound:
                    errs.append(
                        f"{tag}={r.bound} > z({m},{n + 1})<={q.bound} [{q.provenance}]: not monotone in n"
                    )
            for q in anyrow(m + 1, n, s, t):
                if r.bound > q.bound:
                    errs.append(
                        f"{tag}={r.bound} > z({m + 1},{n})<={q.bound} [{q.provenance}]: not monotone in m"
                    )
            for q in anyrow(m, n - 1, s, t):
                if r.bound > q.bound + m:
                    errs.append(
                        f"{tag}={r.bound} > z({m},{n - 1})+{m}={q.bound + m}: column deletion"
                    )
            for q in anyrow(m - 1, n, s, t):
                if r.bound > q.bound + n:
                    errs.append(f"{tag}={r.bound} > z({m - 1},{n})+{n}={q.bound + n}: row deletion")
            for q in exact(m, n, s, t):
                if q.bound != r.bound:
                    errs.append(
                        f"{tag}: exact rows disagree ({r.bound} [{r.provenance}] vs {q.bound} [{q.provenance}])"
                    )
            for q in anyrow(n, m, t, s):
                if q.kind == "exact" and q.bound != r.bound:
                    errs.append(f"{tag}={r.bound} vs z({n},{m};{t},{s})={q.bound}: transposition")
                if q.kind == "ub" and q.bound < r.bound:
                    errs.append(f"{tag}={r.bound} > z({n},{m};{t},{s})<={q.bound}: transposition")
    for c in claims:
        if not c.get("lb_verified"):
            continue
        for q in anyrow(c["m"], c["n"], c["s"], c["t"]) + anyrow(c["n"], c["m"], c["t"], c["s"]):
            if q.bound < c["lb"]:
                errs.append(
                    f"z({q.m},{q.n};{q.s},{q.t})<={q.bound} [{q.provenance}] is below the verified "
                    f"witness {c['lb']} [{c['lb_source']}]"
                )
    return errs


# ---------------------------------------------------------------------------
# Seeding (deterministic; `python -m zar_ub.ledger seed` regenerates both files)
# ---------------------------------------------------------------------------
_TAN_CSV = os.path.join(UB_DIR, "docs", "lit", "tan2022_table3_z3.csv")
_EXACT_CSV = os.path.join(DATA_DIR, "exact_33.csv")
_BHAN_CELLS = {(11, 21): 116, (12, 22): 132}  # in exact_33.csv from bhan26 witnesses (LB) + Tan UB

# Collins et al. 2016 Table 4 upper bounds named in docs/literature_review.md §2.2 and
# docs/lit/hou2026_13x18.md (transcribed, not re-derived; opt-in tier "collins16").
_COLLINS16 = {
    (12, 17): (103, "exact"),
    (12, 18): (109, "ub"),
    (13, 17): (110, "ub"),
    (13, 18): (116, "ub"),
    (14, 17): (118, "ub"),
    (14, 18): (124, "ub"),
    (15, 17): (126, "ub"),
    (15, 18): (132, "ub"),
    (16, 17): (133, "ub"),
    (16, 18): (140, "ub"),
}

# Cells closed by this pipeline (cache/certs/<tag>/manifest.json, all cases certified).
_LEAN_HERE = [
    # (m, n, w, mode)  -> bound w-1
    (9, 9, 50, "pure"),  # E9: 36/36 LRAT certificates, no external facts
    (10, 20, 103, "tan2022"),  # 6/6 LRAT certificates, generator used Tan facts (hypotheses)
    (11, 21, 117, "tan2022"),  # 0 admissible cases after Argument I with Tan facts
]


def seed_ledger_rows() -> List[LedgerRow]:
    rows: List[LedgerRow] = []
    with open(_EXACT_CSV) as f:
        for d in csv.DictReader(f):
            m, n, z = int(d["m"]), int(d["n"]), int(d["z"])
            hyp = ""
            if (m, n) in _BHAN_CELLS:
                hyp = "ub: Tan 2022 Table 3 (Roman bound), exactness: bhan26 witness (reproduced)"
            rows.append(LedgerRow(m, n, 3, 3, z, "exact", "tan2022", "", hyp))
    exact_cells = {(r.m, r.n) for r in rows}
    if os.path.exists(_TAN_CSV):
        with open(_TAN_CSV) as f:
            for d in csv.DictReader(f):
                m, n = int(d["m"]), int(d["n"])
                if d["status"] == "OPEN_UB_ROMAN" and (m, n) not in exact_cells:
                    rows.append(
                        LedgerRow(
                            m,
                            n,
                            3,
                            3,
                            int(d["tan_value"]),
                            "ub",
                            "tan2022",
                            "",
                            "Roman 1975 bound as tabulated in Tan 2022 Table 3 (non-bold)",
                        )
                    )
    for (m, n), (b, kind) in _COLLINS16.items():
        rows.append(
            LedgerRow(
                m,
                n,
                3,
                3,
                b,
                kind,
                "collins16",
                "",
                "Collins-Riasanovsky-Wallace-Radziszowski 2016 Table 4, via literature_review §2.2",
            )
        )
    tan_rows = list(rows)
    for m, n, w, mode in _LEAN_HERE:
        inst = Instance(m, n, 3, 3, w)
        facts = facts_for(inst, "tan2022" if mode != "pure" else "pure", tan_rows)
        closure = os.path.join("cache", "certs", inst.tag, "manifest.json")
        rows.append(
            LedgerRow(
                m, n, 3, 3, w - 1, "ub", "lean-here", closure, all_facts_hold_hypotheses(facts)
            )
        )
    return _sorted(rows)


# docs/literature_review.md §2.2, frontier block 9<=m<=16, 17<=n<=23.  Transcribed 2026-09-21.
# (lb, ub, lb_source, ub_source, lb_verified, ub_reviewed, reviewed_ub, reviewed_ub_source, notes)
_CLAIMS: Dict[Tuple[int, int], tuple] = {
    (9, 17): (81, 81, "tan2022", "tan2022", 1, 1, 81, "tan2022", ""),
    (9, 18): (85, 85, "tan2022", "tan2022", 1, 1, 85, "tan2022", ""),
    (9, 19): (89, 89, "tan2022", "tan2022", 1, 1, 89, "tan2022", ""),
    (9, 20): (93, 93, "tan2022", "tan2022", 1, 1, 93, "tan2022", ""),
    (9, 21): (96, 96, "tan2022", "tan2022", 1, 1, 96, "tan2022", ""),
    (9, 22): (100, 100, "tan2022", "tan2022", 1, 1, 100, "tan2022", ""),
    (9, 23): (103, 103, "bhan26", "dfield26", 1, 0, 104, "tan2022", "dfield: LRAT+Lean end-to-end"),
    (10, 17): (90, 90, "tan2022", "tan2022", 1, 1, 90, "tan2022", ""),
    (10, 18): (94, 94, "tan2022", "tan2022", 1, 1, 94, "tan2022", ""),
    (10, 19): (98, 98, "tan2022", "tan2022", 1, 1, 98, "tan2022", ""),
    (10, 20): (102, 102, "tan2022", "tan2022", 1, 1, 102, "tan2022", ""),
    (10, 21): (
        106,
        106,
        "bhan26",
        "dfield26",
        1,
        0,
        None,
        "tan2022",
        "dfield: deletion from tan (9,21)=96",
    ),
    (10, 22): (110, 110, "bhan26", "dfield26", 1, 0, None, "tan2022", "dfield: Lean end-to-end"),
    (10, 23): (
        112,
        112,
        "bhan26",
        "dfield26",
        1,
        0,
        None,
        "tan2022",
        "dfield: 13 SAT/MIP profiles, 25 GB certs",
    ),
    (11, 17): (96, 96, "tan2022", "tan2022", 1, 1, 96, "tan2022", ""),
    (11, 18): (101, 101, "tan2022", "tan2022", 1, 1, 101, "tan2022", ""),
    (11, 19): (
        106,
        106,
        "dfield26",
        "dfield26",
        0,
        0,
        None,
        "tan2022",
        "also wang26 (secondary); dfield: deletion from tan (11,18)=101",
    ),
    (11, 20): (111, 111, "bhan26", "dfield26", 1, 0, None, "tan2022", "dfield: two deletions"),
    (11, 21): (116, 116, "bhan26", "tan2022", 1, 1, 116, "tan2022", "bhan LB = dgh/tan UB"),
    (11, 22): (121, 121, "bhan26", "tan2022", 1, 1, 121, "tan2022", "bhan LB = tan UB"),
    (11, 23): (
        123,
        123,
        "dfield26",
        "dfield26",
        0,
        0,
        None,
        "tan2022",
        "dfield: deletion from (10,23)=112",
    ),
    (12, 17): (
        103,
        103,
        "collins16",
        "collins16",
        1,
        1,
        103,
        "collins16",
        "Collins Table 4, unique extremal graph; re-certified afrasyab26",
    ),
    (12, 18): (
        108,
        108,
        "hou26",
        "hou26",
        1,
        0,
        109,
        "collins16",
        "hou: uniqueness import + 924 checks; afrasyab: 4-case orbit certs; LB bhan/hou",
    ),
    (12, 19): (
        114,
        114,
        "hadamard",
        "afrasyab26",
        0,
        0,
        None,
        "tan2022",
        "afrasyab: deletion chain; LB Hadamard 3-(12,6,2) blocks",
    ),
    (12, 20): (120, 120, "afrasyab26", "afrasyab26", 0, 0, None, "tan2022", ""),
    (12, 21): (126, 126, "numaro", "afrasyab26", 0, 0, None, "tan2022", "LB numaro (unreviewed)"),
    (12, 22): (132, 132, "bhan26", "tan2022", 1, 1, 132, "tan2022", "bhan LB = tan/Roman UB"),
    (12, 23): (134, 134, "dfield26", "dfield26", 0, 0, None, "tan2022", "dfield: Lean end-to-end"),
    (13, 17): (110, 110, "hou26", "collins16", 1, 1, 110, "collins16", "hou witness"),
    (13, 18): (
        116,
        116,
        "hou26",
        "collins16",
        1,
        1,
        116,
        "collins16",
        "hou/wang/afrasyab witnesses",
    ),
    (13, 19): (118, 122, "saurabh26", "afrasyab26", 1, 0, 125, "dgh26", "open"),
    (13, 20): (119, 128, "bhan26", "afrasyab26", 1, 0, 130, "dgh26", "open"),
    (13, 21): (127, 134, "bhan26", "afrasyab26", 1, 0, 135, "dgh26", "open"),
    (13, 22): (137, 137, "bhan26", "afrasyab26", 1, 0, None, "tan2022", "afrasyab: 83 profiles"),
    (13, 23): (
        135,
        144,
        "bhan26",
        "dfield26",
        1,
        0,
        None,
        "tan2022",
        "open; note (a): afrasyab chain would give 143",
    ),
    (14, 17): (118, 118, "hou26", "collins16", 1, 1, 118, "collins16", "hou/afrasyab witness"),
    (14, 18): (124, 124, "hou26", "collins16", 1, 1, 124, "collins16", "hou/afrasyab witness"),
    (14, 19): (126, 130, "saurabh26", "afrasyab26", 1, 0, 135, "dgh26", "open"),
    (14, 20): (126, 136, "saurabh26", "afrasyab26", 1, 0, 140, "dgh26", "open; LB padded"),
    (14, 21): (131, 142, "bhan26", "afrasyab26", 1, 0, 145, "dgh26", "open"),
    (14, 22): (137, 148, "bhan26", "afrasyab26", 1, 0, 150, "dgh26", "open; note (a)"),
    (14, 23): (138, 154, "bhan26", "afrasyab26", 1, 0, 155, "tan2022", "open; note (a)"),
    (15, 17): (126, 126, "hou26", "collins16", 1, 1, 126, "collins16", "hou/afrasyab witness"),
    (15, 18): (132, 132, "bhan26", "collins16", 1, 1, 132, "collins16", "bhan LB = collins16 UB"),
    (15, 19): (132, 139, "bhan26", "afrasyab26", 1, 0, 143, "dgh26", "open"),
    (15, 20): (138, 145, "bhan26", "afrasyab26", 1, 0, 149, "dgh26", "open"),
    (15, 21): (139, 152, "bhan26", "afrasyab26", 1, 0, 154, "dgh26", "open"),
    (15, 22): (143, 158, "bhan26", "afrasyab26", 1, 0, 160, "tan2022", "open"),
    (15, 23): (149, 165, "bhan26", "dgh26", 1, 1, 165, "dgh26", "open; UB dgh = afrasyab"),
    (16, 17): (132, 133, "afrasyab26", "collins16", 0, 1, 133, "collins16", "open; gap 1"),
    (16, 18): (
        136,
        140,
        "saurabh26",
        "collins16",
        1,
        1,
        140,
        "collins16",
        "open; UB collins16 = afrasyab",
    ),
    (16, 19): (136, 147, "saurabh26", "afrasyab26", 1, 0, 152, "dgh26", "open; LB padded"),
    (16, 20): (146, 154, "bhan26", "afrasyab26", 1, 0, 158, "dgh26", "open"),
    (16, 21): (147, 161, "bhan26", "afrasyab26", 1, 0, 164, "dgh26", "open"),
    (16, 22): (149, 168, "bhan26", "afrasyab26", 1, 0, 169, "dgh26", "open"),
    (16, 23): (158, 175, "bhan26", "dgh26", 1, 1, 175, "dgh26", "open; UB dgh = afrasyab"),
}


def seed_claims_rows() -> List[dict]:
    tan = {}
    if os.path.exists(_TAN_CSV):
        with open(_TAN_CSV) as f:
            for d in csv.DictReader(f):
                tan[(int(d["m"]), int(d["n"]))] = int(d["tan_value"])
    out = []
    for (m, n), (lb, ub, lbs, ubs, lbv, ubr, rub, rubs, notes) in sorted(_CLAIMS.items()):
        if rub is None and rubs == "tan2022":
            rub = tan.get((m, n))
        out.append(
            {
                "m": m,
                "n": n,
                "s": 3,
                "t": 3,
                "lb": lb,
                "ub": ub,
                "status": "exact" if lb == ub else "open",
                "lb_source": lbs,
                "ub_source": ubs,
                "lb_verified": lbv,
                "ub_reviewed": ubr,
                "reviewed_ub": rub if rub is not None else "",
                "reviewed_ub_source": rubs,
                "notes": notes,
            }
        )
    return out


def write_claims(rows: List[dict], path: str = CLAIMS_PATH) -> None:
    errs = check_claims_file(rows)
    if errs:
        raise LedgerError("claims not written:\n  " + "\n  ".join(errs))
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLAIMS_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(path + ".tmp", path)


def check_claims_file(rows: List[dict]) -> List[str]:
    """Sanity of claims_2026.csv itself: lb <= ub <= reviewed_ub, exact status consistent."""
    errs = []
    for r in rows:
        tag = f"claims z({r['m']},{r['n']})"
        if r["lb"] > r["ub"]:
            errs.append(f"{tag}: lb {r['lb']} > ub {r['ub']}")
        if r.get("reviewed_ub") not in ("", None) and int(r["reviewed_ub"]) < r["ub"]:
            errs.append(f"{tag}: reviewed ub {r['reviewed_ub']} < ub {r['ub']}")
        if (r["status"] == "exact") != (r["lb"] == r["ub"]):
            errs.append(f"{tag}: status/interval mismatch")
    return errs


def targets(min_gap: int = 1, claims: Optional[List[dict]] = None) -> List[dict]:
    """Open cells (lb < ub) from claims_2026.csv, smallest gap first: target selection only."""
    claims = load_claims() if claims is None else claims
    open_cells = [c for c in claims if c["ub"] - c["lb"] >= min_gap]
    return sorted(open_cells, key=lambda c: (c["ub"] - c["lb"], c["m"], c["n"]))


def _main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    sub.add_parser("seed")
    p = sub.add_parser("facts")
    for x in "mnstw":
        p.add_argument(x, type=int)
    p.add_argument("--trust", default=DEFAULT_TRUST, choices=sorted(TRUST_LEVELS))
    p.add_argument("--lean", action="store_true", help="print the Lean list term")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        errs = check_claims()
        errs += check_claims_file(load_claims())
        print(
            "\n".join(errs)
            if errs
            else f"ok: {len(load_ledger())} ledger rows, {len(load_claims())} claims rows"
        )
        return 1 if errs else 0
    if args.cmd == "seed":
        claims = seed_claims_rows()
        write_claims(claims)
        rows = write_ledger(seed_ledger_rows(), claims=claims)
        print(
            f"wrote {len(rows)} ledger rows -> {LEDGER_PATH}; {len(claims)} claims rows -> {CLAIMS_PATH}"
        )
        return 0
    facts = facts_for(Instance(args.m, args.n, args.s, args.t, args.w), args.trust)
    print(lean_fact_list(facts) if args.lean else "\n".join(map(str, facts)) or "(none)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

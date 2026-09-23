"""Declarative registry of the once-proved Lean schemas (design §2.3) and their Python mirrors.

The evolved program supplies *parameters* (``SCHEMA_DATA``) for schemas whose soundness is
proved once in ``lean/ZarPrune/Schemas.lean``; the harness instantiates them into Lean terms
(``render_terms``) that the gate ORs into ``gateInstK`` and evaluates alone as ``schema_mask``.

Interface used by ``evaluator.py`` (all three are called defensively; any exception is a
``schema_error`` hard zero):

* ``validate(schema_data, P) -> (ok, errors)``
* ``render_terms(schema_data, P, target_name) -> list[str]`` fully-qualified Lean terms of
  type ``ZarPrune.Prune <target_name>``
* ``mirror_kill(schema_data, P, rows, cols) -> bool`` — the Python replica of the Lean kill,
  with **identical arithmetic** (Nat truncated subtraction, the same layer-cake sums, the
  same ``Nat.findGreatest`` cap, the same ordered-pair ``rhs``).

Family ``farkas`` (``ZarPrune.Prune.ofFarkas P ys``): Farkas certificates of infeasibility for
the pair-codegree system (hidden variables ``λ_{i,i'} = |N(i) ∩ N(i')|`` over the ordered
off-diagonal pairs, identified by symmetry):

    F1   Σ_{i} Σ_{i'≠i} λ = T2' := Σ_j c_j (c_j − 1)                (two inequalities)
    F2   lo_i ≤ Σ_{i'≠i} λ_{i,i'} ≤ hi_i   (r_i lightest / heaviest values of c_j − 1)
    F3   λ_{i,i'} ≤ cap_{i,i'} := min(Lcap, r_i, r_i')  (Lcap: largest k ≤ n whose k lightest
         values of C(c_j − 2, s − 2) sum to ≤ (t − 1)·C(m − 2, s − 2))
    F4   λ_{i,i'} ≥ r_i + r_i' − n                                 (inclusion–exclusion)
    λ ≥ 0

One certificate is a flat list of non-negative integers laid out as
``[y1p, y1m, l_0..l_{m-1}, u_0..u_{m-1}, y3(pairs), y4(pairs)]`` with the pairs in
lexicographic order ``(0,1), (0,2), …, (m−2, m−1)`` (``pair_idx``); ``K = n_base(m)`` entries.
It kills a profile iff every combined coefficient
``y1p − y1m + u_i − l_i + u_i' − l_i' + y3_p − y4_p`` is ≥ 0 and
``rhs = (y1p − y1m)·T2' + 2(Σ u_i hi_i − Σ l_i lo_i) + Σ_{ordered pairs}(y3 cap − y4 floor) < 0``.
Missing entries read as 0 (``Array.getD``), exactly as in Lean.

``SCHEMA_DATA["farkas"]`` entries are either flat lists (applied on every instance) or dicts
``{"m": .., "n": .., "s": .., "t": .., "y": [...]}`` (applied only where ``(m, n, s, t)`` match;
recommended: a certificate is meaningful only for the instance it was computed on).

``farkas_certificate(m, n, s, t, rows, cols)`` (needs scipy, imported lazily) solves the dual
LP, rationalises the multipliers and returns an integer certificate verified by
``certificate_kills``, or ``None``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from fractions import Fraction
from math import comb, gcd
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Family:
    lean: str
    params: Dict[str, str]
    doc: str
    implemented: bool = True


# Test-only knob: the fully-qualified Lean name of the Farkas schema.  The library name is
# `ZarPrune.Prune.ofFarkas` (lean/ZarPrune/Schemas.lean); before that module is built, the
# E12 experiment inlines the Schemas body into the candidate (renamed `ofFarkasInl`, since a
# candidate may not declare `Prune.*`) and sets ZAR_UB_FARKAS_LEAN=ZarPrune.Cand.ofFarkasInl
# so the same certificates go through the real gate.
FARKAS_LEAN = os.environ.get("ZAR_UB_FARKAS_LEAN", "ZarPrune.Prune.ofFarkas")

FAMILIES: Dict[str, Family] = {
    "farkas": Family(
        lean=FARKAS_LEAN,
        params={"ys": "List[List[Nat]]  (flat certificates or {m,n,s,t,y} dicts)"},
        doc="Farkas certificates over hidden pair codegrees: Σ y_k a_k ≥ 0 on every λ and Σ y_k b_k < 0 "
        "for the system F1–F4 (P3/P11; lean/ZarPrune/Schemas.lean).",
    ),
    "residue": Family(
        lean="ZarPrune.Prune.ofResidue",
        params={"g": "Nat", "marked": "List[Nat]", "exceptional": "List[Nat]"},
        doc="RESERVED (P14, not yet in the library): only an empty list is accepted.",
        implemented=False,
    ),
    "prefix": Family(
        lean="ZarPrune.Prune.ofPrefixF",
        params={"k": "Nat"},
        doc="RESERVED (Argument I is conditional, gate entry point pending): only an empty list is accepted.",
        implemented=False,
    ),
}

MAX_CERTS = 400  # certificates instantiated per instance (protects the gate's time budget)
MAX_LEN = 4096  # entries per certificate
MAX_VALUE = 10**18  # entries are Nat literals; keep them printable


# ---------------------------------------------------------------------------
# Layout (mirrors `decode` / `pairIdx` in Schemas.lean)
# ---------------------------------------------------------------------------
def n_pairs(m: int) -> int:
    return m * (m - 1) // 2


def n_base(m: int) -> int:
    """Number of constraints = length of a well-formed certificate: 2 + 2m + 2·C(m,2)."""
    return 2 + 2 * m + 2 * n_pairs(m)


def pair_idx(m: int, i: int, j: int) -> int:
    """Index of the unordered pair {i, j} (i ≠ j) in lexicographic order; = ZarPrune.pairIdx."""
    if i < j:
        return i * (2 * m - i - 1) // 2 + (j - i - 1)
    return j * (2 * m - j - 1) // 2 + (i - j - 1)


def layout(m: int) -> Dict[str, Tuple[int, int]]:
    """Slices [start, end) of the blocks of a certificate for m rows."""
    P = n_pairs(m)
    return {
        "y1p": (0, 1),
        "y1m": (1, 2),
        "l": (2, 2 + m),
        "u": (2 + m, 2 + 2 * m),
        "y3": (2 + 2 * m, 2 + 2 * m + P),
        "y4": (2 + 2 * m + P, 2 + 2 * m + 2 * P),
    }


def _getD(y: Sequence[int], k: int) -> int:
    return int(y[k]) if 0 <= k < len(y) else 0


# ---------------------------------------------------------------------------
# The constraint system (mirrors lowSum / topSum / Lcap / cap / floor / T2 / sysOf)
# ---------------------------------------------------------------------------
def _nsub(a: int, b: int) -> int:
    """Nat truncated subtraction."""
    return a - b if a >= b else 0


def cnt_lt(cols: Sequence[int], y: int) -> int:
    return sum(1 for c in cols if c < y)


def cnt_gt(cols: Sequence[int], x: int) -> int:
    return sum(1 for c in cols if x < c)


def low_sum(f, m: int, cols: Sequence[int], k: int) -> int:
    """ZarPrune.lowSum f pf k = k·f(0) + Σ_{x<m} (k − cntLt(x+1))·(f(x+1) − f(x))."""
    return k * f(0) + sum(_nsub(k, cnt_lt(cols, x + 1)) * _nsub(f(x + 1), f(x)) for x in range(m))


def top_sum(f, m: int, cols: Sequence[int], k: int) -> int:
    """ZarPrune.topSum f pf k = k·f(0) + Σ_{x<m} min(k, cntGt x)·(f(x+1) − f(x))."""
    return k * f(0) + sum(min(k, cnt_gt(cols, x)) * _nsub(f(x + 1), f(x)) for x in range(m))


def f2(c: int) -> int:
    return _nsub(c, 1)


def f3(s: int):
    return lambda c: comb(_nsub(c, 2), _nsub(s, 2))


def budget3(m: int, s: int, t: int) -> int:
    return _nsub(t, 1) * comb(_nsub(m, 2), _nsub(s, 2))


def find_greatest(pred, n: int) -> int:
    """Nat.findGreatest pred n: the largest k ≤ n with pred k, else 0."""
    for k in range(n, 0, -1):
        if pred(k):
            return k
    return 0


@dataclass
class FarkasSystem:
    m: int
    n: int
    s: int
    t: int
    rows: Tuple[int, ...]
    cols: Tuple[int, ...]
    t2: int  # Σ_j c_j (c_j − 1)   (ordered count; = 2 Σ_j C(c_j, 2))
    lo: List[int]  # per row: lowSum f2 (r_i)
    hi: List[int]  # per row: topSum f2 (r_i)
    lcap: int  # Lcap
    cap: List[List[int]]  # cap[i][j] = min(lcap, r_i, r_j)
    flo: List[List[int]]  # flo[i][j] = r_i + r_j − n (truncated)

    @property
    def pairs(self) -> List[Tuple[int, int]]:
        return [(i, j) for i in range(self.m) for j in range(i + 1, self.m)]


def farkas_system(m: int, n: int, s: int, t: int, rows: Sequence[int], cols: Sequence[int]) -> FarkasSystem:
    """The profile-derived constants, computed exactly as `sysOf` does (row/col vectors are
    padded with zeros / truncated to m / n like the gate's `gateProfile` (`List.getD`))."""
    rows = tuple(int(rows[i]) if i < len(rows) else 0 for i in range(m))
    cols = tuple(int(cols[j]) if j < len(cols) else 0 for j in range(n))
    t2 = sum(c * _nsub(c, 1) for c in cols)
    lo = [low_sum(f2, m, cols, r) for r in rows]
    hi = [top_sum(f2, m, cols, r) for r in rows]
    B3 = budget3(m, s, t)
    fs = f3(s)
    lcap = find_greatest(lambda k: low_sum(fs, m, cols, k) <= B3, n)
    cap = [[min(lcap, min(rows[i], rows[j])) for j in range(m)] for i in range(m)]
    flo = [[_nsub(rows[i] + rows[j], n) for j in range(m)] for i in range(m)]
    return FarkasSystem(m, n, s, t, rows, cols, t2, lo, hi, lcap, cap, flo)


def decode(m: int, y: Sequence[int]) -> Dict[str, Any]:
    """ZarPrune.decode: the multipliers of one certificate (missing entries = 0)."""
    P = n_pairs(m)
    return {
        "y1p": _getD(y, 0),
        "y1m": _getD(y, 1),
        "l": [_getD(y, 2 + i) for i in range(m)],
        "u": [_getD(y, 2 + m + i) for i in range(m)],
        "y3": [[_getD(y, 2 + 2 * m + pair_idx(m, i, j)) if i != j else 0 for j in range(m)] for i in range(m)],
        "y4": [[_getD(y, 2 + 2 * m + P + pair_idx(m, i, j)) if i != j else 0 for j in range(m)] for i in range(m)],
    }


def coef(Y: Dict[str, Any], i: int, j: int) -> int:
    return Y["y1p"] - Y["y1m"] + Y["u"][i] - Y["l"][i] + Y["u"][j] - Y["l"][j] + Y["y3"][i][j] - Y["y4"][i][j]


def coef_ok(m: int, Y: Dict[str, Any]) -> bool:
    return all(i == j or coef(Y, i, j) >= 0 for i in range(m) for j in range(m))


def rhs(S: FarkasSystem, Y: Dict[str, Any]) -> int:
    """ZarPrune.rhs over the ORDERED off-diagonal pairs (each unordered pair counted twice)."""
    m = S.m
    v = (Y["y1p"] - Y["y1m"]) * S.t2
    v += 2 * (sum(Y["u"][i] * S.hi[i] for i in range(m)) - sum(Y["l"][i] * S.lo[i] for i in range(m)))
    v += sum(Y["y3"][i][j] * S.cap[i][j] - Y["y4"][i][j] * S.flo[i][j] for i in range(m) for j in range(m) if i != j)
    return v


def cert_kills_system(S: FarkasSystem, y: Sequence[int]) -> bool:
    """ZarPrune.certKills: coefficientwise Σ y a ≥ 0 and Σ y b < 0 (exact integers)."""
    Y = decode(S.m, y)
    return coef_ok(S.m, Y) and rhs(S, Y) < 0


def certificate_kills(
    m: int, n: int, s: int, t: int, rows: Sequence[int], cols: Sequence[int], y: Sequence[int]
) -> bool:
    """(Prune.ofFarkas P [y]).kill: the `2 ≤ s` guard and one certificate."""
    if s < 2:
        return False
    return cert_kills_system(farkas_system(m, n, s, t, rows, cols), y)


def farkas_kill(
    m: int, n: int, s: int, t: int, rows: Sequence[int], cols: Sequence[int], ys: Sequence[Sequence[int]]
) -> bool:
    """(Prune.ofFarkas P ys).kill."""
    if s < 2 or not ys:
        return False
    S = farkas_system(m, n, s, t, rows, cols)
    return any(cert_kills_system(S, y) for y in ys)


# ---------------------------------------------------------------------------
# SCHEMA_DATA plumbing
# ---------------------------------------------------------------------------
def _is_nat(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= MAX_VALUE


def _cert_list(entry, k: int, errors: List[str]) -> Optional[List[int]]:
    """Validate one certificate list; append errors."""
    if not isinstance(entry, (list, tuple)):
        errors.append(f"farkas[{k}]: certificate must be a list of Nat, got {type(entry).__name__}")
        return None
    if len(entry) > MAX_LEN:
        errors.append(f"farkas[{k}]: certificate has {len(entry)} entries > {MAX_LEN}")
        return None
    bad = [v for v in entry if not _is_nat(v)]
    if bad:
        errors.append(f"farkas[{k}]: non-Nat entries {bad[:3]}")
        return None
    return [int(v) for v in entry]


def _farkas_entries(sd: dict, P, errors: Optional[List[str]] = None) -> List[List[int]]:
    """Certificates of `sd["farkas"]` that apply to the instance P (m, n, s, t)."""
    errors = errors if errors is not None else []
    out: List[List[int]] = []
    raw = sd.get("farkas") or []
    if not isinstance(raw, (list, tuple)):
        errors.append("farkas: must be a list")
        return out
    for k, entry in enumerate(raw):
        if isinstance(entry, dict):
            keys = {"m", "n", "s", "t"}
            if not keys <= set(entry) or "y" not in entry:
                errors.append(f"farkas[{k}]: dict entries need keys m, n, s, t, y")
                continue
            if not all(_is_nat(entry[q]) for q in keys):
                errors.append(f"farkas[{k}]: m, n, s, t must be Nat")
                continue
            y = _cert_list(entry["y"], k, errors)
            if y is None:
                continue
            if len(y) > n_base(int(entry["m"])):
                errors.append(
                    f"farkas[{k}]: certificate for m={entry['m']} has {len(y)} entries > "
                    f"n_base(m)={n_base(int(entry['m']))} (missing entries read as 0)"
                )
                continue
            if (int(entry["m"]), int(entry["n"]), int(entry["s"]), int(entry["t"])) == (P.m, P.n, P.s, P.t):
                out.append(y)
        else:
            y = _cert_list(entry, k, errors)
            if y is None:
                continue
            out.append(y)
    return out


def validate(schema_data, P) -> Tuple[bool, List[str]]:
    """(ok, errors) for SCHEMA_DATA on instance P.  Unknown family, wrong shape, non-Nat
    entries, a non-empty reserved family or too many certificates are errors."""
    errors: List[str] = []
    if not isinstance(schema_data, dict):
        return False, ["SCHEMA_DATA must be a dict"]
    for fam, val in schema_data.items():
        if fam not in FAMILIES:
            errors.append(f"unknown schema family {fam!r} (known: {sorted(FAMILIES)})")
            continue
        if not FAMILIES[fam].implemented:
            if val:
                errors.append(f"schema family {fam!r} is reserved ({FAMILIES[fam].doc}); pass []")
            continue
    if "farkas" in schema_data and schema_data["farkas"]:
        certs = _farkas_entries(schema_data, P, errors)
        if len(certs) > MAX_CERTS:
            errors.append(f"farkas: {len(certs)} certificates for {P.tag} > MAX_CERTS={MAX_CERTS}")
    return (not errors), errors


def _lean_list(y: Sequence[int]) -> str:
    return "[" + ",".join(str(int(v)) for v in y) + "]"


def render_terms(schema_data, P, target_name: str) -> List[str]:
    """Fully-qualified Lean terms of type `ZarPrune.Prune <target_name>` (empty when no
    certificate applies to P)."""
    errors: List[str] = []
    certs = _farkas_entries(schema_data, P, errors) if isinstance(schema_data, dict) else []
    if errors:
        raise ValueError("; ".join(errors))
    if not certs:
        return []
    ys = "[" + ", ".join(_lean_list(y) for y in certs[:MAX_CERTS]) + "]"
    return [f"{FAMILIES['farkas'].lean} {target_name} {ys}"]


def mirror_kill(schema_data, P, rows, cols) -> bool:
    """Python replica of the schema kill on one case (identical arithmetic to Lean)."""
    if not isinstance(schema_data, dict):
        return False
    certs = _farkas_entries(schema_data, P)
    if not certs:
        return False
    return farkas_kill(P.m, P.n, P.s, P.t, list(rows), list(cols), certs[:MAX_CERTS])


# ---------------------------------------------------------------------------
# Certificate search (exact verification; the LP is only a heuristic oracle)
# ---------------------------------------------------------------------------
def _rationalise(vals: Sequence[float], max_den: int) -> List[int]:
    fr = [Fraction(float(v)).limit_denominator(max_den) if v > 1e-12 else Fraction(0) for v in vals]
    den = 1
    for f in fr:
        den = den * f.denominator // gcd(den, f.denominator)
    ints = [int(f * den) for f in fr]
    g = 0
    for v in ints:
        g = gcd(g, v)
    if g > 1:
        ints = [v // g for v in ints]
    return ints


def farkas_certificate(
    m: int,
    n: int,
    s: int,
    t: int,
    rows: Sequence[int],
    cols: Sequence[int],
    max_dens: Sequence[int] = (1, 2, 6, 12, 60, 1000, 10**6),
) -> Optional[List[int]]:
    """Solve the dual LP  min Σy  s.t.  y ≥ 0, Σ_k y_k a_k ≥ 0 (per unordered pair), Σ_k y_k b_k = −1
    (b_k in the unordered normalisation); rationalise; return an integer certificate that
    `certificate_kills` accepts, or None (LP feasible ⇒ the primal system is infeasible only
    if a certificate is found; we never trust the float LP alone)."""
    import numpy as np  # noqa: WPS433 (lazy: scipy is not needed at evaluation time)
    from scipy.optimize import linprog  # noqa: WPS433

    if s < 2:
        return None
    S = farkas_system(m, n, s, t, rows, cols)
    K = n_base(m)
    L = layout(m)
    pairs = S.pairs
    P = len(pairs)
    # coefficient rows: for each unordered pair p, Σ_k y_k a_{k,p} ≥ 0  ->  -A y ≤ 0
    A = np.zeros((P, K))
    for p, (i, j) in enumerate(pairs):
        A[p, L["y1p"][0]] += 1
        A[p, L["y1m"][0]] -= 1
        A[p, L["u"][0] + i] += 1
        A[p, L["l"][0] + i] -= 1
        A[p, L["u"][0] + j] += 1
        A[p, L["l"][0] + j] -= 1
        A[p, L["y3"][0] + p] += 1
        A[p, L["y4"][0] + p] -= 1
    # b (unordered normalisation: rhs_ordered = 2 · b·y)
    b = np.zeros(K)
    t2u = S.t2 // 2
    b[L["y1p"][0]] = t2u
    b[L["y1m"][0]] = -t2u
    for i in range(m):
        b[L["l"][0] + i] = -S.lo[i]
        b[L["u"][0] + i] = S.hi[i]
    for p, (i, j) in enumerate(pairs):
        b[L["y3"][0] + p] = S.cap[i][j]
        b[L["y4"][0] + p] = -S.flo[i][j]
    res = linprog(
        np.ones(K),
        A_ub=-A,
        b_ub=np.zeros(P),
        A_eq=b.reshape(1, -1),
        b_eq=np.array([-1.0]),
        bounds=[(0, None)] * K,
        method="highs",
    )
    if res.status != 0 or res.x is None:
        return None
    for md in max_dens:
        y = _rationalise(res.x, md)
        if cert_kills_system(S, y):
            return y
    # last resort: scale and round
    for scale in (1, 2, 3, 4, 6, 12, 24, 60, 120, 840):
        y = [int(round(v * scale)) for v in res.x]
        if cert_kills_system(S, y):
            return y
    return None


def sparsify(y: Sequence[int]) -> List[int]:
    """Drop trailing zeros (the Lean decoder reads missing entries as 0); keeps the layout."""
    y = list(y)
    while y and y[-1] == 0:
        y.pop()
    return y


# ---------------------------------------------------------------------------
# Search-mode helper for candidates (design §2.3: the LLM may compute SCHEMA_DATA in Python)
# ---------------------------------------------------------------------------
def search_certificates(max_seconds: float = 40.0, kinds=None, cache_dir=None,
                        max_per_cell: int = 12, only_survivors: bool = True) -> List[dict]:
    """Return SCHEMA_DATA["farkas"] entries ({"m","n","s","t","y"}) that refute cached cases of
    the suite's tables which the proved library does not kill.  Reads the case tables from
    ZAR_UB_CACHE_DIR (the evaluator hands every candidate a private copy) or cache/.
    Deterministic (tables are visited in sorted order); stops after `max_seconds`.
    Every certificate is re-verified exactly with `certificate_kills` before it is returned."""
    import glob as _glob
    import json as _json
    import time as _time
    t0 = _time.time()
    cache_dir = cache_dir or os.environ.get("ZAR_UB_CACHE_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
    out: List[dict] = []
    seen = set()
    # visit small labelled TRAIN tables first, then GEN, then the big TARGET tables
    loaded = []
    for f in sorted(_glob.glob(os.path.join(cache_dir, "case_table_*.json"))):
        try:
            d = _json.load(open(f))
        except Exception:
            continue
        if kinds and d.get("kind", "") not in kinds:
            continue
        # small (training-ladder) tables first; tables without a kind tag count as training tables
        prio = {"train": 0, "": 0, "gen": 1, "battery": 2, "target": 3}.get(d.get("kind", ""), 3)
        loaded.append((prio, len(d.get("records") or []), f, d))
    loaded.sort(key=lambda x: (x[0], x[1], x[2]))
    for _, _, f, d in loaded:
        if _time.time() - t0 > max_seconds:
            break
        inst = d.get("inst") or {}
        m, n, s, t, w = (inst.get(k) for k in ("m", "n", "s", "t", "w"))
        if not all(isinstance(v, int) for v in (m, n, s, t, w)) or s < 2:
            continue
        mask = d.get("baseline_lean_mask") or []
        recs = d.get("records") or []
        got = 0
        for i, r in enumerate(recs):
            if _time.time() - t0 > max_seconds or got >= max_per_cell:
                break
            if only_survivors and i < len(mask) and mask[i]:
                continue
            if (r.get("probe") or {}).get("status") == "sat":
                continue
            rows, cols = r["rows"], r["cols"]
            # skip cases an already-found certificate of this cell kills
            if any(e["m"] == m and e["n"] == n and certificate_kills(m, n, s, t, rows, cols, e["y"]) for e in out):
                continue
            try:
                y = farkas_certificate(m, n, s, t, rows, cols)
            except Exception:
                y = None
            if y and certificate_kills(m, n, s, t, rows, cols, y):
                y = sparsify(y)
                key = (m, n, s, t, tuple(y))
                if key not in seen:
                    seen.add(key)
                    out.append({"m": m, "n": n, "s": s, "t": t, "y": list(y)})
                    got += 1
    return out

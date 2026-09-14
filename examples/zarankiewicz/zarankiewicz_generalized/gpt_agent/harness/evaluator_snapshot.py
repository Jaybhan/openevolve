"""
Evaluator for the Zarankiewicz problem z(M,N;3,3).

Goal: maximize the number of 1s in a MxN binary matrix with no 3×3
all-ones submatrix (K_{3,3}-free bipartite graph).

"""

import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import traceback
from itertools import combinations
from math import comb

import numpy as np

# Problem instances (must match initial_program.py's construct_graph(M, N)).
# Order matters: the first VISIBLE_COUNT entries are the ones the LLM sees.
#
# Every entry below is a PROVEN EXACT value of z(M,N;3,3) (bold in the
# reference table), not a KST upper bound — the is_exact half of the score can
# only ever fire on attained values.
# Reference table of z(m,n;3,3), row m -> values for n = m .. 23.
# Transcribed from the published table; entries are only PROVEN EXACT up to the
# per-row limit in _EXACT_UP_TO. Past that limit the published figure is an
# UPPER BOUND that no construction is known to attain, so those cells are
# excluded entirely: scoring against an unattained bound would make is_exact
# unreachable there and cap combined_score below 1.0 forever, destroying "1.0
# means done" as a stopping criterion.
_TABLE = {
    3: [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
    4: [13, 16, 18, 21, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54],
    5: [20, 22, 25, 28, 30, 33, 36, 38, 41, 44, 46, 49, 52, 54, 57, 60, 62, 64, 66],
    6: [26, 29, 32, 36, 39, 42, 45, 48, 50, 53, 56, 58, 61, 64, 66, 69, 72, 74],
    7: [33, 37, 40, 44, 47, 50, 53, 56, 60, 63, 66, 69, 72, 75, 78, 81, 84],
    8: [42, 45, 50, 53, 57, 60, 64, 67, 70, 74, 77, 81, 84, 87, 90, 94],
    9: [49, 54, 59, 64, 67, 70, 73, 77, 81, 85, 89, 93, 96, 100, 104],
    10: [60, 64, 68, 73, 77, 81, 85, 90, 94, 98, 102, 108, 112, 116],
    11: [69, 74, 80, 84, 88, 92, 96, 101, 109, 113, 117, 121, 125],
    12: [80, 86, 91, 96, 99, 108, 113, 118, 122, 127, 132, 136],
    13: [92, 98, 104, 107, 117, 122, 126, 131, 136, 140, 145],
    14: [105, 112, 115, 125, 130, 136, 141, 146, 151, 155],
    15: [120, 123, 134, 139, 144, 150, 155, 160, 166],
    16: [128, 142, 148, 154, 160, 165, 170, 176],
}

# Largest n for which row m's table entry is a proven exact value.
_EXACT_UP_TO = {
    3: 23, 4: 23, 5: 23, 6: 23, 7: 23, 8: 23,
    9: 22, 10: 20, 11: 18, 12: 16, 13: 16, 14: 16, 15: 16, 16: 16,
}

# Exact beyond their row's limit, verified individually. (11,21) also overrides
# the table's 117, which is the upper bound rather than the attained value.
_EXTRA_EXACT = {(11, 21): 116, (12, 22): 132}

# Every proven-exact cell, built from the table above.
KST_EXACT_VALUE = {}
for _m, _vals in _TABLE.items():
    for _k, _v in enumerate(_vals):
        _n = _m + _k
        if _n <= _EXACT_UP_TO[_m]:
            KST_EXACT_VALUE[(_m, _n)] = _v
KST_EXACT_VALUE.update(_EXTRA_EXACT)

# Holdout: every 4th cell of the sorted list, so the reserved tail is spread
# across every row rather than concentrated in one size regime — a holdout made
# only of large or only of elongated cells would test one construction family
# instead of the rule as a whole. Deterministic, so runs stay comparable.
_SORTED = sorted(KST_EXACT_VALUE)
_HOLDOUT = _SORTED[3::4]
_ELIGIBLE = [c for c in _SORTED if c not in set(_HOLDOUT)]

# Eligible prefix first, reserved tail last. Everything is scored; only the
# prefix may ever be reported back.
PROBLEMS = _ELIGIBLE + _HOLDOUT

# How many instances may be shown at once. Raised 3 -> 5 alongside the jump from
# 14 to ~161 cells: with a larger, more varied instance set a single unsolved
# cell is a weaker signal about where the construction breaks.
VISIBLE_COUNT = 5

# Hard floor on the holdout. The LAST ALWAYS_HIDDEN_COUNT entries of PROBLEMS
# are NEVER shown to the LLM under any configuration — VISIBLE_COUNT can only
# shrink what is shown, never expand into this reserved tail. So raising
# VISIBLE_COUNT, or shortening PROBLEMS, can never accidentally expose them,
# and the model can never accumulate enough of the answer set to hardcode it.
ALWAYS_HIDDEN_COUNT = len(_HOLDOUT)
assert ALWAYS_HIDDEN_COUNT < len(PROBLEMS), (
    f"ALWAYS_HIDDEN_COUNT ({ALWAYS_HIDDEN_COUNT}) must leave at least one "
    f"instance visible out of {len(PROBLEMS)} PROBLEMS"
)


def _visible_count():
    """
    How many instances may be reported to the LLM: VISIBLE_COUNT, clamped so the
    reserved tail stays hidden. Computed on each call rather than cached, so it
    cannot go stale if VISIBLE_COUNT or PROBLEMS is changed.
    """
    return max(0, min(VISIBLE_COUNT, len(PROBLEMS) - ALWAYS_HIDDEN_COUNT))


def _redact_hidden_instances(text):
    """
    Scrub mentions of hidden (M, N) instances from text destined for the LLM.

    The evolved program is called with every hidden instance's dimensions as
    live arguments, so worker tracebacks and stderr can leak them — e.g. a
    program raising ValueError("unsupported 15x16") on a hidden instance, or
    deliberately printing (M, N) to stderr. Everything surfaced as an artifact
    must pass through here first.

    This blocks accidental leaks and cheap deliberate ones (standard renderings
    of the pair). Obfuscated exfiltration (e.g. printing M*1000+N) is out of
    scope — that requires sandboxing, not string filtering.
    """
    for m, n in PROBLEMS[_visible_count():]:
        for pat in (
            rf"\(\s*{m}\s*,\s*{n}\s*\)",          # (15, 16)
            rf"\b{m}\s*[x×*]\s*{n}\b",            # 15x16, 15×16, 15*16
            rf"\bM\s*=\s*{m}\b[,;]?\s*N\s*=\s*{n}\b",  # M=15, N=16
        ):
            text = re.sub(pat, "[hidden instance]", text)
    return text


S = 3
T = 3

# Per-instance CPU budget, in seconds. THIS IS THE ANTI-SEARCH MECHANISM.
#
# The prompt asks for a construction rather than a search, but asking is not
# enforcement: through phase 2 every top-scoring program was a greedy packer
# plus hill climbing, because nothing in the score made search unprofitable.
# The exactness split was meant to cap search near 0.48 (see score_graph);
# it does not, because on small M a local search attains the optimum outright.
#
# Time does separate them, cleanly and by three orders of magnitude. Measured
# over all 161 instances:
#
#     construction (cyclic difference family)   max   0.1 ms   total  0.003 s
#     phase-2 champion (greedy + hill climb)    max 724   ms   total 17.1   s
#     phase-2 best-ever (same, wider restarts)  max 867   ms   total 26.7   s
#
# 25 ms sits in the empty gap: 250x headroom over the construction's worst
# instance -- enough for finite-field tables, Singer sets, primitive-root
# searches over the small parameters this range needs -- while putting 56% of
# the champion's instances and 61% of the best-ever's over budget. A search
# cannot buy its way back with a faster inner loop; it would need a 30x
# speedup to fit, at which point it is no longer searching.
#
# Measured in process_time (CPU), not wall clock, so a loaded machine does not
# change the verdict. Over-budget instances score 0 for that instance only.
PER_INSTANCE_TIME_LIMIT = 0.025

# Fail loudly at import. Without this a missing entry raises KeyError inside
# evaluate(), where the broad exception handler turns it into zero metrics —
# so every program in the run scores 0 and nothing says why.
_missing = [p for p in PROBLEMS if p not in KST_EXACT_VALUE]
assert not _missing, f"PROBLEMS entries with no KST_EXACT_VALUE: {_missing}"

# Shared state: track current best combined_score across evaluations (n_SOTA).
# Stored in a file so worker processes can read it.
_SOTA_FILE = os.path.join(os.path.dirname(__file__), ".n_sota")

def _read_n_sota():
    try:
        with open(_SOTA_FILE) as f:
            return float(f.read().strip())
    except Exception:
        return 0.0

def _update_n_sota(n):
    current = _read_n_sota()
    if n > current:
        with open(_SOTA_FILE, "w") as f:
            f.write(str(n))


_LOG_FILE = os.path.join(os.path.dirname(__file__), "instance_log.jsonl")


def _log_instances(per_problem, combined_score):
    """
    Append one record per evaluation covering ALL instances, including the
    hidden ones.

    Artifacts are truncated to VISIBLE_COUNT so the LLM cannot memorize the
    held-out set — but that would leave you blind to them too. This file never
    enters a prompt, so it carries the full picture for your own analysis.
    """
    record = {
        "time": round(time.time(), 3),
        "combined_score": round(combined_score, 6),
        "instances": [
            {
                "mn": f"{r['M']}x{r['N']}",
                "edges": int(r["num_edges"]),
                "exact": int(r["exact_value"]),
                "violations": int(r["violation_count"]),
                "valid": bool(r["validity"]),
                "is_exact": bool(r["is_exact"]),
                "failed": bool(r.get("failed")),
                "over_budget": bool(r.get("over_budget")),
                "cpu_ms": round(r["cpu"] * 1000, 3) if r.get("cpu") is not None else None,
            }
            for r in per_problem
        ],
    }
    try:
        # One sub-4KB line written in a single call, so the parallel_evaluations
        # workers can append to this file concurrently without interleaving.
        with open(_LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # logging must never break an evaluation


def _format_time_budget(per_problem):
    """
    Report the CPU budget, identity-free, over ALL instances including hidden.

    An over-budget instance scores 0 with no other explanation, which reads as a
    correctness bug and sends the model hunting in the wrong place. This says
    plainly that the cost was time, how far over it went, and that the fix is a
    different kind of algorithm rather than a faster inner loop.
    """
    timed = [r for r in per_problem if r.get("cpu") is not None]
    if not timed:
        return None
    over = [r for r in timed if r.get("over_budget")]
    limit_ms = PER_INSTANCE_TIME_LIMIT * 1000
    peak_ms = max(r["cpu"] for r in timed) * 1000
    lines = [
        f"CPU BUDGET: {limit_ms:.0f} ms per instance (measured as process CPU time, "
        "not wall clock).",
        "An instance over budget scores ZERO for that instance regardless of the "
        "matrix it returned.",
        "",
        f"  instances over budget:  {len(over)} of {len(timed)}",
        f"  slowest instance:       {peak_ms:.1f} ms",
    ]
    if over:
        worst = max(r["cpu"] for r in over) * 1000
        lines += [
            "",
            f"  Your slowest instance used {worst / limit_ms:.0f}x the budget. This is not "
            "a tuning problem.",
            "  A construction that derives its matrix from M and N — a formula, an "
            "incidence",
            "  structure, a difference set — finishes the largest instance in this "
            "range in",
            "  well under a millisecond, so it never comes close to the limit. "
            "Anything that",
            "  iterates toward an answer (greedy placement, repair loops, restarts, "
            "local",
            "  improvement, branch and bound) does not fit, and cannot be made to fit "
            "by",
            "  optimising the loop. Derive the matrix instead of searching for it.",
        ]
    return "\n".join(lines)


def _describe_failures(per_problem):
    """
    One-line summary of every instance that failed or is invalid, hidden ones
    included. Goes to stdout only — OpenEvolve sends artifacts to the LLM, not
    evaluator stdout, so naming held-out instances here does not leak them.
    """
    bad = [r for r in per_problem if r.get("failed") or not r["validity"]]
    if not bad:
        return None
    def _why(r):
        if r.get("over_budget"):
            return f"{r['cpu'] * 1000:.0f}ms OVER BUDGET"
        if r.get("failed"):
            return "ERROR"
        return f"{int(r['violation_count'])} viol"

    # Capped: a program that misses the CPU budget everywhere produces 100+
    # entries, and this line goes to the run log on every single evaluation.
    parts = [f"{r['M']}x{r['N']}({_why(r)})" for r in bad[:12]]
    if len(bad) > 12:
        parts.append(f"... and {len(bad) - 12} more")
    n_over = sum(1 for r in bad if r.get("over_budget"))
    tag = f" ({n_over} over CPU budget)" if n_over else ""
    return f"  failing {len(bad)}/{len(per_problem)}{tag}: " + " ".join(parts)


def _make_result(metrics, artifacts=None):
    """Wrap metrics + artifacts, falling back to plain metrics outside OpenEvolve."""
    try:
        from openevolve.evaluation_result import EvaluationResult

        return EvaluationResult(metrics=metrics, artifacts=artifacts or {})
    except ImportError:
        return metrics


def _format_matrix_for_llm(A, label):
    """Render a binary matrix as a compact string the LLM can read and reason about."""
    ncols = A.shape[1]
    lines = [
        f"{label} — row degrees: {A.sum(axis=1).astype(int).tolist()}",
        "     " + " ".join(f"{j:2d}" for j in range(ncols)),
        "     " + "--" * ncols,
    ]
    for i, row in enumerate(A.astype(int)):
        lines.append(f"r{i:2d} | " + " ".join(str(v) for v in row))
    lines.append("col  " + " ".join(f"{d:2d}" for d in A.sum(axis=0).astype(int)))
    return "\n".join(lines)


def _bound_budget(M):
    """
    Total triple-sharing allowed by the counting bound.

    Every column of degree c contributes C(c, 3) row-triples that share it, and
    no triple may be shared by 3+ columns (that is a K_{3,3}). So
        sum_j C(colDegree_j, 3)  <=  (T - 1) * C(M, S)
    """
    return (T - 1) * comb(M, S) if M >= S else 0


def _bound_used(col_degrees):
    """How much of that allowance a given column-degree profile consumes."""
    return sum(comb(int(d), S) for d in col_degrees)


def _optimal_bound_used(edges, M, N):
    """
    What an optimal matrix would consume, approximating its column degrees as
    evenly spread. Gives the LLM a target to compare its own usage against —
    the gap is the room it still has.
    """
    base, extra = divmod(edges, N)
    return extra * comb(base + 1, S) + (N - extra) * comb(base, S)


def _codegree_stats(G, M):
    """
    Histogram of |common 1-columns| over all row triples.

    Bins 0..S-1 are legal sharing levels; the final bin is the ILLEGAL one —
    triples sharing S or more, i.e. the K_{S,T}s themselves. Every triple lands
    in exactly one bin, so the bins always sum to C(M, S) and stay reconcilable
    with _bound_used. Dropping the illegal triples instead would silently break
    that identity on exactly the invalid matrices that most need explaining.
    """
    hist = [0] * (S + 1)
    rows = [G[i].astype(bool) for i in range(M)]
    for triple in combinations(range(M), S):
        shared = int(np.logical_and.reduce([rows[i] for i in triple]).sum())
        hist[min(shared, S)] += 1
    return hist


def _format_codegree(shown):
    """
    Structural feedback: how tight the construction is against the K_{S,T}
    limit. Edge counts say *that* a matrix is short; this says *why*.

    This deliberately does NOT say where to add edges. It used to — "too sparse,
    add edges at the triples still sharing fewer than 2" — which is an
    instruction to hill climb, and it pointed at the wrong thing besides: the
    phase-2 champion sat at 97.6% of optimum edges with only 38% of instances
    exact, so ~96% of its remaining score was in attaining exact values, not in
    density. Coaching density spent attention on the last 1% while the other 31%
    went untouched. The table stays as diagnosis; the prescription is gone.
    """
    lines = [
        f"A K_{{{S},{T}}} appears as soon as {S} rows share {T} common 1-columns, so every",
        f"row triple may share AT MOST {T - 1}.",
        "",
        f"Counting bound: sum over columns of C(colDegree, {S}) <= {T - 1} * C(M, {S}).",
        "'budget used' is how much of that allowance your matrix consumes; 'optimal'",
        "is roughly what a matrix attaining the known optimum would consume.",
        "",
        "This is a CERTIFICATE, not an objective. It tells you whether your",
        f"construction is structurally capable of the target — because C(c, {S}) is",
        "convex, a matrix whose column degrees are far from level cannot reach the",
        "optimum no matter how many edges you bolt on. Use it to check the shape of",
        "what your rule produces, then change the RULE. Adding edges one at a time",
        "until the budget is spent is the thing this number will happily let you do",
        "and the thing that has never once closed a remaining gap.",
        f"The last column, sharing {S}+, counts K_{{{S},{T}}}s — it must be 0.",
        "",
        f"{'instance':>10}  {'triples sharing':>24}  {'budget used':>22}",
        f"{'':>10}  {' / '.join(list(str(k) for k in range(S)) + [f'{S}+']):>24}",
        "-" * 66,
    ]
    for r in shown:
        if r.get("failed") or r["G"] is None:
            lines.append(f"{r['M']}x{r['N']:<8}  (no matrix)")
            continue
        M, N = r["M"], r["N"]
        hist = _codegree_stats(r["G"], M)
        used = _bound_used(r["G"].sum(axis=0))
        budget = _bound_budget(M)
        opt = _optimal_bound_used(int(r["exact_value"]), M, N)
        pct = f"{used / budget:.0%}" if budget else "n/a"
        opct = f"{opt / budget:.0%}" if budget else "n/a"
        lines.append(
            f"{M}x{N:<8}  {' / '.join(f'{h:5d}' for h in hist):>24}  "
            f"{used:>5}/{budget:<5} = {pct:>4}  (optimal ~{opct})"
        )
    return "\n".join(lines)


def _format_gap_profile(per_problem):
    """
    Identity-free summary of how far every instance is from exact — including
    the hidden ones.

    The per-instance table only ever shows a handful of cells, so a systematic
    near-miss is invisible: a construction can be one edge short on twenty
    instances and never learn that the deficit is uniform. This reports the
    DISTRIBUTION of shortfalls with no sizes, no targets and no identities, so
    it cannot be memorised, but it does say where the cheap wins are — a
    cluster at "short by 1" means one structural fix flips many cells at once.
    """
    miss = [
        int(r["exact_value"]) - int(r["num_edges"])
        for r in per_problem
        if not r.get("failed") and r["validity"] and not r["is_exact"]
    ]
    n_exact = sum(1 for r in per_problem if r["is_exact"])
    if not miss:
        return f"All {n_exact} valid instances are EXACT."
    hist = {}
    for g in miss:
        hist[g] = hist.get(g, 0) + 1
    lines = [
        f"SHORTFALL PROFILE over ALL {len(per_problem)} scored instances "
        f"(hidden ones included; no sizes or targets revealed):",
        f"  exact:           {n_exact}",
        f"  not yet exact:   {len(miss)}   (total edges missing: {sum(miss)})",
        "",
        "  edges short   how many instances",
    ]
    for g in sorted(hist):
        lines.append(f"  {g:>11}   {hist[g]:>3}  {'#' * min(hist[g], 40)}")
    tight = sum(v for k, v in hist.items() if k <= 2)
    if tight:
        lines += [
            "",
            f"  {tight} instances are within 2 edges. A shortfall that repeats at the "
            "same size across many instances is one systematic defect, not many "
            "separate ones — find the single rule that recovers those edges and "
            "they all flip together.",
        ]
    return "\n".join(lines)


def _format_breakdown(per_problem, partial=True):
    """
    One row per (M, N) so the LLM can see exactly which instance is failing.

    The table covers only the visible slice, so it is captioned as partial —
    without a caption the model reads it as a complete accounting and assumes
    that fixing these rows fixes everything. The caption deliberately gives no
    count and no identities, so it reveals nothing about the held-out set.
    """
    lines = []
    if partial:
        lines += [
            "PARTIAL VIEW — these are some of the instances you have NOT yet "
            "solved exactly, and they are only a fraction of what you are scored "
            "on. Further instances are hidden: they are scored but never shown, "
            "and your score already reflects them. Fixing only the rows below "
            "will not fix those — a change that only special-cases these sizes "
            "will leave your score almost unchanged.",
            "",
        ]
    lines += [
        f"{'instance':>10}  {'edges':>12}  {'valid':>5}  {'exact':>5}  "
        f"{'violations':>10}  row degrees",
        "-" * 86,
    ]
    for r in per_problem:
        inst = f"{r['M']}x{r['N']}"
        if r.get("over_budget"):
            # Distinct from ERROR on purpose: the matrix may have been perfect.
            # Reporting this as "no usable matrix returned" sends the model
            # hunting for a correctness bug that does not exist.
            lines.append(
                f"{inst:>10}  {'OVER BUDGET':>12}  {'-':>5}  {'-':>5}  {'-':>10}  "
                f"used {r['cpu'] * 1000:.0f}ms of "
                f"{PER_INSTANCE_TIME_LIMIT * 1000:.0f}ms CPU — scored 0"
            )
            continue
        if r.get("failed"):
            lines.append(
                f"{inst:>10}  {'ERROR':>12}  {'-':>5}  {'-':>5}  {'-':>10}  "
                "no usable matrix returned"
            )
            continue
        edges = f"{int(r['num_edges'])}/{int(r['exact_value'])}"
        valid = "yes" if r["validity"] else "NO"
        exact = "YES" if r["is_exact"] else "-"
        lines.append(
            f"{inst:>10}  {edges:>12}  {valid:>5}  {exact:>5}  "
            f"{int(r['violation_count']):>10}  {r['row_degrees']}"
        )
    return "\n".join(lines)


class TimeoutError(Exception):
    pass


def count_kst_violations(A, s, t):
    """
    Count the number of K_{s,t} subgraphs in the binary matrix A.

    A K_{s,t} exists when s rows share t or more common 1-columns.
    For each s-subset of rows sharing k >= t common 1-columns,
    it contributes C(k, t) violations.

    For M=10, s=t=3: C(10,3) = 120 row-triples to check — fast.
    """
    m, n = A.shape
    count = 0
    for row_subset in combinations(range(m), s):
        # Columns that are 1 in ALL s selected rows
        common = A[row_subset[0]].astype(bool)
        for r in row_subset[1:]:
            common &= A[r].astype(bool)
        shared = int(common.sum())
        if shared >= t:
            count += comb(shared, t)
    return count


def has_kst(A, s, t):
    """Fast short-circuit check: does A contain any K_{s,t}?"""
    m, n = A.shape
    for row_subset in combinations(range(m), s):
        common = A[row_subset[0]].astype(bool)
        for r in row_subset[1:]:
            common &= A[r].astype(bool)
        if common.sum() >= t:
            return True
    return False

def _read_partial(results_path, problems):
    """
    Read whatever the worker managed to checkpoint, padded out to one entry per
    problem. Missing or unfinished instances come back as (None, M, N, None).
    Returns (graphs, error_string_or_None).

    Results are keyed by (M, N) and re-emitted in `problems` order, so the
    worker is free to RUN the instances in any order it likes — which is what
    lets execution be permuted without disturbing anything downstream.
    """
    done, error = [], None
    if os.path.exists(results_path):
        try:
            with open(results_path, "rb") as f:
                results = pickle.load(f)
            done = results.get("graphs", [])
            error = results.get("error")
        except Exception as e:  # truncated/corrupt checkpoint
            error = f"could not read worker results: {e}"

    by_instance = {(M, N): (G, cpu) for G, M, N, cpu in done}
    graphs = []
    for (M, N) in problems:
        G, cpu = by_instance.get((M, N), (None, None))
        graphs.append((G, M, N, cpu))
    return graphs, error


def _execution_order(problems):
    """
    The order the worker CALLS the instances in — deliberately not PROBLEMS order.

    PROBLEMS is `_ELIGIBLE + _HOLDOUT`, and that tail position is load-bearing:
    _visible_count(), _redact_hidden_instances() and the artifact slice in
    evaluate() all identify the held-out set as "the last ALWAYS_HIDDEN_COUNT
    entries". Reordering PROBLEMS itself would silently break all three.

    But running them in that order means any truncation — a timeout, a crash,
    a kill — deletes a contiguous tail, which is exactly the holdout: 25% of
    the area-weighted score, removed as a block. That happened repeatedly in
    phase 2, and once cost a 103-exact program the top spot to a 96-exact one.

    So the ORDER OF EXECUTION is decoupled from the order of scoring: interleave
    the eligible and reserved instances so a truncation lands proportionally on
    both. Deterministic, so runs stay comparable.
    """
    n_hidden = min(ALWAYS_HIDDEN_COUNT, len(problems))
    eligible = list(problems[: len(problems) - n_hidden])
    hidden = list(problems[len(problems) - n_hidden:])
    out, ei, hi = [], 0, 0
    # Round-robin weighted by the two list lengths, so both drain together.
    while ei < len(eligible) or hi < len(hidden):
        if hi >= len(hidden) or (
            ei < len(eligible) and ei * len(hidden) <= hi * len(eligible)
        ):
            out.append(eligible[ei]); ei += 1
        else:
            out.append(hidden[hi]); hi += 1
    return out


def run_with_timeout(program_path, problems=PROBLEMS, timeout_seconds=120):
    """
    Run the program in a single subprocess with one shared timeout, calling the
    entry point once per (M, N) in `problems`.

    Returns (graphs, diagnostic):
      graphs     - always one (G, M, N, cpu_seconds) per problem, in `problems`
                   order. G is None for any instance that did not produce a
                   matrix; cpu_seconds is None if it never ran.
      diagnostic - human-readable text describing what went wrong, or None.

    The worker checkpoints after every instance, so a hang or a kill keeps the
    instances that already finished instead of discarding the whole run. The
    diagnostic is surfaced to the LLM as an artifact — a program that crashes
    or hangs otherwise gets a score of zero with no clue why.

    Instances are CALLED in _execution_order(problems), not `problems` order,
    so a truncation cannot delete the held-out tail as a block. Results come
    back keyed by (M, N) and are re-emitted in `problems` order.
    """
    order = _execution_order(problems)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as temp_file:
        temp_file_path = temp_file.name
        results_path = temp_file_path + ".results"
        script = f"""
import sys, os, pickle, traceback, inspect, time
sys.path.insert(0, os.path.dirname({program_path!r}))

def _dump(payload):
    # Atomic: write a sibling temp file, then rename over the checkpoint.
    # A kill landing mid-write truncates only the temp file, so the previous
    # checkpoint survives intact instead of taking every finished instance
    # down with it. os.replace is atomic within a filesystem, and the temp
    # file is a sibling, so it always is.
    tmp = {results_path!r} + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f)
    os.replace(tmp, {results_path!r})

graphs = []
try:
    import importlib.util
    import numpy as np
    spec = importlib.util.spec_from_file_location("program", {program_path!r})
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)

    # Entry-point names in priority order. construct_graph comes FIRST because
    # that is the name the prompt asks for; a stale zero-arg run_graph left
    # behind by a rewrite must not shadow a correct construct_graph(M, N).
    entry_point = None
    fallback = None
    for fn_name in ('construct_graph', 'run_graph', 'construct_graphs'):
        fn = getattr(program, fn_name, None)
        if not callable(fn):
            continue
        if fallback is None:
            fallback = fn
        try:
            positional = [
                p for p in inspect.signature(fn).parameters.values()
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            ]
        except (TypeError, ValueError):
            continue
        if len(positional) >= 2:
            entry_point = fn
            break
    if entry_point is None:
        entry_point = fallback
    if entry_point is None:
        raise AttributeError(
            "No callable entry point found. Define construct_graph(M, N) "
            "returning an MxN 0/1 numpy array."
        )

    for (M, N) in {order!r}:
        # Isolate per instance: one failing (M, N) must not discard the others.
        # process_time, not wall clock: the budget must mean the same thing on
        # a loaded machine as on an idle one, and four of these run in parallel.
        _t0 = time.process_time()
        try:
            G = np.asarray(entry_point(M, N))
        except Exception:
            traceback.print_exc()
            G = None
        _cpu = time.process_time() - _t0
        graphs.append((G, M, N, _cpu))
        # Checkpoint after every instance so a later hang or kill cannot throw
        # away work that already succeeded.
        _dump({{'graphs': graphs}})
except Exception:
    traceback.print_exc()
    _dump({{'graphs': graphs, 'error': traceback.format_exc()}})
"""
        temp_file.write(script)

    try:
        process = subprocess.Popen(
            [sys.executable, temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()

        stdout = (stdout or b"").decode(errors="replace")
        stderr = (stderr or b"").decode(errors="replace")
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)

        graphs, error = _read_partial(results_path, problems)

        notes = []
        if timed_out:
            finished = sum(1 for G, _, _, _ in graphs if G is not None)
            notes.append(
                f"TIMED OUT after {timeout_seconds}s for all instances combined "
                f"({finished}/{len(problems)} finished before the deadline). "
                "Make the construction faster — it must be direct, not a search."
            )
        elif process.returncode != 0 and not error:
            notes.append(f"Worker process exited with code {process.returncode}.")
        if error:
            notes.append(f"Error while running your program:\n{error.strip()}")
        if stderr.strip():
            notes.append(f"stderr:\n{stderr.strip()[-3000:]}")

        # Redact before returning: diagnostic is surfaced to the LLM as an
        # artifact, and worker tracebacks/stderr can name hidden instances.
        diagnostic = "\n\n".join(notes) if notes else None
        if diagnostic:
            diagnostic = _redact_hidden_instances(diagnostic)
        return graphs, diagnostic
    finally:
        for path in [temp_file_path, results_path, results_path + ".tmp"]:
            if os.path.exists(path):
                os.unlink(path)


def score_graph(graphs):
    """
    Score all (G, M, N) instances and fold them into the single metrics dict
    OpenEvolve selects on.

    Args:
        graphs: list of (G, M, N, cpu_seconds) tuples, one per PROBLEMS entry,
            in PROBLEMS order.

        Per instance:
            Over budget: score = 0. An instance that used more than
                     PER_INSTANCE_TIME_LIMIT of CPU is discarded whatever it
                     returned. See that constant for why: it is the only thing
                     here that a search cannot satisfy and a construction can.
            Valid:   score = 0.5 * is_exact + 0.5 * (e / exact_value)
                     where is_exact = 1 if e == exact_value else 0
            Invalid: score = 0.1 * min(e / exact_value, 1) / (1 + violations)
                     Graded, not a cliff: a matrix one violation from valid is
                     distinguishable from garbage, so near-miss constructions
                     feel a gradient toward validity. Capped at 0.05 (v >= 1),
                     so a valid matrix of the same density always scores at
                     least 10x more — validity strictly dominates density.

    combined_score = mean(score across instances)
                   = 0.5 * (num_exact / n) + 0.5 * mean(e / exact_value)
                     when every instance is valid and within budget

    Exactness is a separate term from the ratio on purpose. It was ALSO meant
    to suppress search on its own — the theory being that a search reaches ~96%
    of every bound but rarely attains one exactly, capping near 0.48. Phase 2
    disproved that: a greedy packer with hill climbing reached 0.706 with
    103/161 exact, because on small M a local search does attain the optimum
    outright. The split is still worth keeping (it is where all the remaining
    headroom lives), but the time budget, not this, is what rules search out.

    n_sota tracks max(n_sota, combined_score) across evaluations.

    Returns (metrics, per_problem) — per_problem carries the per-instance detail
    that goes into artifacts rather than into the metrics dict.
    """
    per_problem = []
    for G, M, N, cpu in graphs:
        exact_value = KST_EXACT_VALUE[(M, N)]
        over_budget = cpu is not None and cpu > PER_INSTANCE_TIME_LIMIT

        if G is None or over_budget:
            # Instance produced no usable matrix (raised, wrong shape,
            # non-binary) or blew the CPU budget. Score it 0 instead of
            # discarding the entire evaluation, so a program that works on most
            # instances keeps its partial credit and evolution retains a
            # gradient.
            per_problem.append({
                "M": M,
                "N": N,
                "G": None,
                "exact_value": exact_value,
                "num_edges": 0.0,
                "validity": 0.0,
                "is_exact": 0.0,
                "violation_count": 0.0,
                "row_degrees": [],
                "normalized_score": 0.0,
                "cpu": cpu,
                "over_budget": bool(over_budget),
                "failed": True,
            })
            continue

        num_edges = int(G.sum())
        violations = count_kst_violations(G, S, T)
        valid = violations == 0

        ratio = (num_edges / exact_value) if valid else 0.0
        is_exact = 1.0 if (valid and num_edges == exact_value) else 0.0
        if valid:
            normalized_score = 0.5 * is_exact + 0.5 * ratio
        else:
            # Graded partial credit (see docstring): denser and closer to
            # valid scores more, but capped 10x below any comparable valid
            # matrix so the pressure toward genuine validity never inverts.
            density = min(num_edges / exact_value, 1.0)
            normalized_score = 0.1 * density / (1.0 + violations)

        per_problem.append({
            "M": M,
            "N": N,
            "G": G,
            "exact_value": exact_value,
            "num_edges": float(num_edges),
            "validity": 1.0 if valid else 0.0,
            "is_exact": is_exact,
            "violation_count": float(violations),
            "row_degrees": G.sum(axis=1).astype(int).tolist(),
            "normalized_score": float(normalized_score),
            "cpu": cpu,
            "over_budget": False,
            "failed": False,
        })

    # Weight each instance by M*N rather than equally. A plain mean is not the
    # neutral choice it looks like: the table has 21 proven values at m=3 but
    # only 1 at m=16, so uniform weighting hands 37% of the score to the three
    # most trivial rows and 6% to the four hardest — an artifact of what the
    # literature happens to have proven, not of what matters here. Weighting by
    # area halves the trivial rows' grip and doubles the hard tail's, while
    # keeping any single cell small (m=16 goes to 1.5%, vs 7.1% under per-row
    # normalisation, where one cell flipping would swing the score ~3.6%).
    # All instances exact still gives exactly 1.0, so "1.0 means done" holds.
    _w = [r["M"] * r["N"] for r in per_problem]
    combined_score = float(
        np.average([r["normalized_score"] for r in per_problem], weights=_w)
    )
    _update_n_sota(combined_score)

    metrics = {
        # Fraction of instances that are K_{S,T}-free, so "failed 1 of 5" is
        # distinguishable from "failed 5 of 5".
        "validity": float(np.mean([r["validity"] for r in per_problem])),
        # Only signal separating invalid programs from each other.
        "violation_count": float(sum(r["violation_count"] for r in per_problem)),
        # Readable progress. With ~161 instances a single newly-solved cell moves
        # combined_score by ~0.003, so the aggregate is a poor progress display;
        # "158 of 161 exact" is legible where 0.9938 is not.
        "exact_count": float(sum(r["is_exact"] for r in per_problem)),
        # Slowest instance, in milliseconds of CPU. Exposed as a metric so it can
        # be used as a MAP-Elites feature dimension: without it the feature grid
        # is complexity x diversity, neither of which separates a closed-form
        # construction from a search, so the two compete in the same cells and
        # the search — which can always buy more edges with more time — wins.
        # As a dimension it gives constructions a niche of their own.
        "peak_instance_ms": float(
            max((r["cpu"] for r in per_problem if r["cpu"] is not None), default=0.0)
        ) * 1000.0,
        "combined_score": combined_score,
    }
    return metrics, per_problem


def evaluate(program_path):
    """
    Main evaluation function called by OpenEvolve.
    Runs the program once per (M, N) in PROBLEMS and returns a metrics dict
    with 'combined_score' as the primary selector.
    """
    start = time.time()
    try:
        # 120s, not 25s. The old cap was hardcoded past the configured
        # evaluator.timeout and sat only 1.4x above the champion's own runtime,
        # so evaluations were being truncated on a busy machine — and because the
        # worker ran PROBLEMS in order, a truncation deleted the held-out tail
        # wholesale. It once cost a 103-exact program the top spot to a 96-exact
        # one. Per-instance CPU is now what constrains the construction (see
        # PER_INSTANCE_TIME_LIMIT); this is only a backstop against a true hang.
        graphs, diagnostic = run_with_timeout(program_path, PROBLEMS, timeout_seconds=120)

        # Reject bad instances individually rather than zeroing the whole run.
        # Reasons are collected so the LLM is told why, not just given a 0.
        # visible_rejects is what reaches the LLM: naming a held-out (M, N) here
        # would reveal the very instances VISIBLE_COUNT exists to hide.
        visible = set(PROBLEMS[:_visible_count()])
        checked, rejects, visible_rejects, hidden_rejects = [], [], [], 0
        for (G, M, N, cpu) in graphs:
            reason = None
            if G is None:
                reason = "no matrix produced"
            elif G.shape != (M, N):
                reason = f"wrong shape {G.shape}, expected ({M}, {N})"
                G = None
            elif not np.all(np.isin(G, [0, 1])):
                reason = "contains values other than 0 and 1"
                G = None
            elif cpu is not None and cpu > PER_INSTANCE_TIME_LIMIT:
                # Not nulled here: score_graph zeroes it and records over_budget,
                # so the time_budget artifact can explain the real cause instead
                # of it surfacing as "no matrix produced".
                reason = (
                    f"over CPU budget ({cpu * 1000:.0f}ms > "
                    f"{PER_INSTANCE_TIME_LIMIT * 1000:.0f}ms)"
                )
            if reason:
                rejects.append(f"{M}x{N}: {reason}")
                if (M, N) in visible:
                    visible_rejects.append(f"{M}x{N}: {reason}")
                else:
                    hidden_rejects += 1
            checked.append((G, M, N, cpu))
        # stdout only — never an artifact. Capped for the same reason as
        # _describe_failures: an over-budget program rejects ~100 instances and
        # this runs on every evaluation.
        for r in rejects[:12]:
            print(r)
        if len(rejects) > 12:
            print(f"... and {len(rejects) - 12} more rejected instances")

        metrics, per_problem = score_graph(checked)
        elapsed = time.time() - start
        print(f"score={metrics['combined_score']:.4f}, time={elapsed:.2f}s")

        # Full per-instance record for you (all instances, hidden included).
        # Never reaches the LLM: the log is a file, and stdout is not an artifact.
        _log_instances(per_problem, metrics["combined_score"])
        failures = _describe_failures(per_problem)
        if failures:
            print(failures)

        # Artifacts carry the per-instance detail the metrics dict can't: which
        # instance is failing and why. Only the visible slice is ever reported,
        # and it is labelled as partial so the model does not read it as the
        # complete list of what it got wrong.
        #
        # Show the first up-to-VISIBLE_COUNT instances that are NOT yet exact,
        # drawn only from the eligible prefix — the reserved tail is excluded
        # here, so rotation can never surface a held-out instance. A fixed
        # positional slice went blind once its cells were solved: the model saw
        # three perfect rows and a note that unseen instances were costing it
        # score, with nothing to act on. Rotating within the prefix keeps the
        # feedback actionable while leaking nothing beyond cells that were
        # already eligible to be shown.
        eligible = per_problem[: len(PROBLEMS) - ALWAYS_HIDDEN_COUNT]
        shown = [r for r in eligible if not r["is_exact"]][: _visible_count()]
        if not shown:
            # Every eligible instance is exact — fall back to the leading ones
            # so the artifact still reports something rather than vanishing.
            shown = eligible[: _visible_count()]
        partial = len(shown) < len(per_problem)

        artifacts = {}
        # First, because an over-budget instance scores 0 and every other artifact
        # will read as though the matrix itself was wrong. Identity-free.
        time_budget = _format_time_budget(per_problem)
        if time_budget:
            artifacts["time_budget"] = time_budget
        # Aggregate shortfall over EVERY instance, hidden included. Identity-free,
        # so it adds a gradient the per-instance table cannot give without leaking.
        artifacts["shortfall_profile"] = _format_gap_profile(per_problem)
        if shown:
            artifacts["instance_breakdown"] = _format_breakdown(shown, partial=partial)
            # Structural feedback: edge counts say the matrix is short, this
            # says why and where the unused capacity is. Computed only for the
            # shown slice, so it costs nothing for the hidden instances.
            artifacts["codegree_analysis"] = _format_codegree(shown)
            # Only ever render an INVALID matrix. A broken one has debugging
            # value (it shows where the K_{S,T}s are) but nothing worth
            # memorizing; rendering a valid one would hand the LLM a literal to
            # hardcode, which is the shortest path to a lookup table.
            broken = [r for r in shown if not r["validity"] and r["G"] is not None]
            if broken:
                worst = max(broken, key=lambda r: r["violation_count"])
                artifacts["worst_instance"] = _format_matrix_for_llm(
                    worst["G"],
                    f"Invalid instance {worst['M']}x{worst['N']} — "
                    f"{int(worst['num_edges'])} edges, "
                    f"{int(worst['violation_count'])} violations",
                )
        # Without this a crashing or hanging program scores 0 with no clue why.
        if diagnostic:
            artifacts["failure"] = diagnostic
        if visible_rejects or hidden_rejects:
            lines = list(visible_rejects)
            if hidden_rejects:
                # Report that hidden instances failed, but never which ones.
                lines.append(
                    f"(plus {hidden_rejects} hidden instance(s) rejected for the "
                    "same kinds of reason — not listed)"
                )
            artifacts["rejected_instances"] = "\n".join(lines)
        return _make_result(metrics, artifacts)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        traceback.print_exc()
        # Redact here too: this artifact reaches the LLM, and an exception
        # raised mid-scoring can carry a hidden (M, N) in its message.
        return _make_result(
            _zero_metrics(),
            {"failure": _redact_hidden_instances(f"Evaluator error: {traceback.format_exc()}")},
        )


def _zero_metrics():
    return {
        "validity": 0.0,
        "violation_count": 0.0,
        "exact_count": 0.0,
        "peak_instance_ms": 0.0,
        "combined_score": 0.0,
    }


if __name__ == "__main__":
    # Quick self-test with the initial program
    import pathlib
    initial = str(pathlib.Path(__file__).parent / "initial_program.py")
    print(f"Testing evaluator with: {initial}")
    result = evaluate(initial)
    print("Metrics:", result)

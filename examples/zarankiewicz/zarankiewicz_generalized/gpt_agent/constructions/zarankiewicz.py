"""General constructive engine for the Zarankiewicz problem z(m, n; s, t).

Convention (matches BASE/evaluator.py): an m x n 0/1 matrix A is K_{s,t}-FREE
when no s rows share t common 1-columns (no all-ones s x t submatrix, rows
chosen among the m rows, columns among the n columns).  z(m,n;s,t) is the
maximum number of 1s; z(m,n;s,t) = z(n,m;t,s) by transposition.

Central frame (from the evolved champion, generalized to (s,t)):
    a column IS a block (subset of rows);  K_{s,t}-free  <=>  every s-subset
    of rows lies in at most (t-1) blocks (a "(t-1)-fold s-packing").
We keep a capacity dict {s-subset: t-1}, decrement as blocks are added, and
complete any structured seed family canonically:
    (a) (s+1)-blocks while capacity allows (several canonical orders),
    (b) a growth pass that enlarges blocks with leftover capacity,
    (c) repeated s-blocks (s edges for 1 capacity),
    (d) weight-(s-1) pads (always free).
Everything the engine returns is checked by the verifier in this module; the
capacity frame guarantees freeness by construction, the verifier is the audit.

Families (see construct() for how they are combined):
    trivial, culik (exact in the elongated regime),
    roman_window (cited closed form, s=t=3),
    sum_layer / xor_layer, hyperplane_f2k, cap_bothsides, line_complement,
    hadamard_3design (+ point-deleted residuals), bipolar (+ reduced
    equator), twin (+ twin_planes8), pair_gdd (group-divisible),
    small_m_seeds, difference_family / orbit_scan (cyclic),
    projective_plane_22, norm_graph,
    greedy_lex_derived (waterfill-guided deterministic floor),
    compositions: pad_extend / best-extension / duplication / shrink /
    side_by_side / stack (the cross-size DP layer),
    bounded exact packers (node-capped canonical DFS completions).

The two-sided waterfill counting bound is used as a PROVEN upper bound to
stop early when a construction meets it.  No randomness anywhere: every
choice is lex-first or derived from (m, n, s, t); the bounded DFS layers
are deterministic and node-budgeted.
"""

from itertools import combinations
from math import comb, factorial, isqrt

import numpy as np

# Tunable work budgets.  "full" is the research profile; "harness" trims the
# bounded-DFS budgets so a whole-table prefill fits the evaluator's 120 s
# subprocess window.  Same algorithms, same order, smaller node caps.
_SPEED = {
    "exact_nodes": 150000,
    "exact_deep": 500000,
    "polish": True,
    "polish_nodes": 250000,
    "orbit_scan_max_m": 13,
    "pair_gdd_multi": 24,
    "pair_gdd_pack_nodes": 60000,
    "pair_gdd_polish_nodes": 250000,
    "pair_gdd_max_n": 99,
    "exact_seed_filter": False,
}


def set_speed_profile(name):
    """Node budgets count INCLUDE-attempts in the bounded packers (each one
    does real capacity work), so these numbers are ~10x smaller than a naive
    call count."""
    if name == "harness":
        _SPEED.update(exact_nodes=40000, exact_deep=400000, polish=False,
                      polish_nodes=0, orbit_scan_max_m=11, pair_gdd_multi=6,
                      pair_gdd_pack_nodes=16000, pair_gdd_polish_nodes=0,
                      pair_gdd_max_n=17, exact_seed_filter=True)
    elif name == "full":
        _SPEED.update(exact_nodes=150000, exact_deep=500000, polish=True,
                      polish_nodes=250000, orbit_scan_max_m=13,
                      pair_gdd_multi=24, pair_gdd_pack_nodes=60000,
                      pair_gdd_polish_nodes=250000, pair_gdd_max_n=99,
                      exact_seed_filter=False)
    else:
        raise ValueError(name)

# --------------------------------------------------------------------------
# 0.  Basic representations
# --------------------------------------------------------------------------


def blocks_to_matrix(m, n, blocks):
    """Column j has 1s exactly in the rows of blocks[j]."""
    A = np.zeros((m, n), dtype=int)
    for j, b in enumerate(blocks[:n]):
        for r in b:
            A[r, j] = 1
    return A


def _row_ints(A):
    """Rows of A as python ints (bit j = column j)."""
    m, n = A.shape
    out = []
    for i in range(m):
        v = 0
        row = A[i]
        for j in range(n):
            if row[j]:
                v |= 1 << j
        out.append(v)
    return out


# --------------------------------------------------------------------------
# 1.  Verifier
# --------------------------------------------------------------------------


def _count_violations_rows(rows, n_cols, s, t, count_all):
    """Iterate s-subsets of `rows` (bitmask ints); count subsets whose common
    support has >= t columns.  If count_all, weight each by C(shared, t) to
    match the reference verifier; else stop at the first violation."""
    m = len(rows)
    total = 0

    def rec(start, depth, mask):
        nonlocal total
        if mask.bit_count() < t:
            return False
        if depth == s:
            shared = mask.bit_count()
            if shared >= t:
                if count_all:
                    total += comb(shared, t)
                    return False
                return True
            return False
        for i in range(start, m - (s - depth) + 1):
            if rec(i + 1, depth + 1, mask & rows[i]):
                return True
        return False

    if s > m:
        return 0
    full = (1 << n_cols) - 1  # all columns; AND with rows narrows it
    hit = rec(0, 0, full)
    if count_all:
        return total
    return 1 if hit else 0


def count_violations(A, s, t):
    """Number of K_{s,t}s in A, counted as in the reference evaluator: each
    s-subset of rows sharing k >= t columns contributes C(k, t)."""
    A = np.asarray(A)
    m, n = A.shape
    if s <= 0 or t <= 0:
        # Degenerate: an empty row (column) set shares everything vacuously.
        if s <= 0 and n >= max(t, 0):
            return comb(n, max(t, 0))
        if t <= 0 and m >= max(s, 0):
            return comb(m, max(s, 0))
        return 0
    if m < s or n < t:
        return 0
    # Work on the cheaper orientation.
    if comb(m, s) <= comb(n, t):
        rows = _row_ints(A)
        return _count_violations_rows(rows, n, s, t, count_all=True)
    rows = _row_ints(A.T)
    return _count_violations_rows(rows, m, t, s, count_all=True)


def verify_kst_free(A, s, t):
    """True iff A contains no K_{s,t} (no s rows sharing t common columns)."""
    A = np.asarray(A)
    m, n = A.shape
    if s <= 0 or t <= 0:
        return count_violations(A, s, t) == 0
    if m < s or n < t:
        return True
    if comb(m, s) <= comb(n, t):
        rows = _row_ints(A)
        return _count_violations_rows(rows, n, s, t, count_all=False) == 0
    rows = _row_ints(A.T)
    return _count_violations_rows(rows, m, t, s, count_all=False) == 0


# --------------------------------------------------------------------------
# 2.  Capacity frame + canonical completion
# --------------------------------------------------------------------------


def _new_cap(m, s, t):
    return {c: t - 1 for c in combinations(range(m), s)}


def _block_fits(cap, block, s):
    if len(block) < s:
        return True
    return all(cap[c] >= 1 for c in combinations(block, s))


def _add_block(cap, block, s):
    if len(block) >= s:
        for c in combinations(block, s):
            cap[c] -= 1


def _trim_to_fit(cap, block, s):
    """Deterministically drop elements of `block` until it fits the remaining
    capacity.  Drop the element sitting in the most exhausted s-subsets
    (ties: largest index).  Returns the trimmed tuple (possibly len < s)."""
    b = list(block)
    while len(b) >= s and not _block_fits(cap, tuple(b), s):
        scores = []
        for x in b:
            bad = sum(
                1
                for c in combinations(b, s)
                if x in c and cap[c] <= 0
            )
            scores.append((bad, x))
        scores.sort(key=lambda p: (-p[0], -p[1]))
        b.remove(scores[0][1])
    return tuple(sorted(b))


def _grow_blocks(cap, blocks, m, s):
    """Growth pass: for each block (in order), add rows while every newly
    covered s-subset still has capacity.  Converts leftover capacity to edges.
    Candidate rows in lex order."""
    out = []
    for b in blocks:
        bset = set(b)
        for r in range(m):
            if r in bset:
                continue
            if len(bset) + 1 < s:
                bset.add(r)
                continue
            new_subs = [
                tuple(sorted(c + (r,)))
                for c in combinations(sorted(bset), s - 1)
            ]
            if all(cap[c] >= 1 for c in new_subs):
                for c in new_subs:
                    cap[c] -= 1
                bset.add(r)
        out.append(tuple(sorted(bset)))
    return out


def _pad_blocks(m, n, s, blocks):
    """Fill up to n columns with weight-(s-1) blocks, balanced by row degree
    (ties: lex).  Weight < s can never take part in a K_{s,t}."""
    deg = [0] * m
    for b in blocks:
        for r in b:
            deg[r] += 1
    out = list(blocks)
    w = max(s - 1, 0)
    while len(out) < n:
        order = sorted(range(m), key=lambda r: (deg[r], r))
        b = tuple(sorted(order[:w]))
        out.append(b)
        for r in b:
            deg[r] += 1
    return out


def _splus1_iter(m, s, order, t=3):
    """Canonical orders for the (s+1)-block fill.

    'lex'  : plain lexicographic.
    'sum'  : sum-residue layers.  Within layer c (sum == c mod m) every
             s-subset is covered at most once (the missing element is
             determined), so the first t-1 layers form a clean (t-1)-fold
             packing; ordering by layer realizes it greedily.
    'xor'  : same idea with F_2^k labels: within layer c (XOR == c) any
             s-subset determines the last element, so each layer is a partial
             Steiner system.  For m <= 8, s = 3 layer 0 is exactly the 14
             planes of AG(3,2) (the unique SQS(8)).
    """
    combos = list(combinations(range(m), s + 1))
    if order == "sum":
        combos.sort(key=lambda b: ((sum(b) % m), b))
    elif order == "xor":
        def x(b):
            v = 0
            for e in b:
                v ^= e
            return v
        combos.sort(key=lambda b: (x(b), b))
    return combos


def _exact_pack(cap, cand_blocks, n_slots, s, node_budget=100000,
                max_mult=1):
    """Maximum-cardinality packing from an ordered candidate block list under
    the capacity `cap` (at most n_slots blocks, each candidate up to
    max_mult times), by bounded DFS with a fungible-capacity bound.  Mutates
    cap by the chosen blocks; returns them.  Deterministic given the order."""
    cands = []
    for q in cand_blocks:
        subs = list(combinations(q, s))
        if all(cap[c] >= 1 for c in subs):
            cands.append((tuple(q), subs))
    if not cands:
        return []
    per = min(len(subs) for _, subs in cands)
    mult_bound = max_mult * len(cands)
    capleft = sum(cap.values())
    best = {"take": [], "count": 0}
    nodes = [0]

    def rec(idx, taken, capleft_now):
        if len(taken) > best["count"]:
            best["count"] = len(taken)
            best["take"] = list(taken)
        if idx >= len(cands) or len(taken) >= n_slots:
            return
        nodes[0] += 1
        if nodes[0] > node_budget:
            return
        ub = len(taken) + min(max_mult * (len(cands) - idx),
                              capleft_now // per, n_slots - len(taken))
        if ub <= best["count"]:
            return
        q, subs = cands[idx]
        fit = min((cap[c] for c in subs), default=max_mult)
        kmax = min(max_mult, fit, n_slots - len(taken))
        for k in range(1, kmax + 1):
            for _ in range(k):
                for c in subs:
                    cap[c] -= 1
                taken.append(q)
            rec(idx + 1, taken, capleft_now - k * len(subs))
            for _ in range(k):
                taken.pop()
                for c in subs:
                    cap[c] += 1
            if nodes[0] > node_budget:
                return
        rec(idx + 1, taken, capleft_now)

    rec(0, [], capleft)
    for q in best["take"]:
        for c in combinations(q, s):
            cap[c] -= 1
    return list(best["take"])


def _exact_pack_multi(cap, cand_blocks, n_slots, s, node_budget=100000,
                      max_solutions=24, max_mult=1):
    """Like _exact_pack but collects up to max_solutions distinct
    maximum-cardinality packings (does NOT mutate cap).  Deterministic."""
    cands = []
    for q in cand_blocks:
        subs = list(combinations(q, s))
        if all(cap[c] >= 1 for c in subs):
            cands.append((tuple(q), subs))
    if not cands:
        return []
    per = min(len(subs) for _, subs in cands)
    capleft = sum(cap.values())
    sols = {"count": 0, "list": []}
    nodes = [0]

    def rec(idx, taken, capleft_now):
        if len(taken) > sols["count"]:
            sols["count"] = len(taken)
            sols["list"] = [list(taken)]
        elif len(taken) == sols["count"] and sols["count"] > 0 and \
                len(sols["list"]) < max_solutions and \
                list(taken) not in sols["list"]:
            sols["list"].append(list(taken))
        if idx >= len(cands) or len(taken) >= n_slots:
            return
        nodes[0] += 1
        if nodes[0] > node_budget:
            return
        ub = len(taken) + min(max_mult * (len(cands) - idx),
                              capleft_now // per, n_slots - len(taken))
        if ub < sols["count"] or (ub == sols["count"]
                                  and len(sols["list"]) >= max_solutions):
            return
        q, subs = cands[idx]
        fit = min((cap[c] for c in subs), default=max_mult)
        kmax = min(max_mult, fit, n_slots - len(taken))
        for k in range(1, kmax + 1):
            for _ in range(k):
                for c in subs:
                    cap[c] -= 1
                taken.append(q)
            rec(idx + 1, taken, capleft_now - k * len(subs))
            for _ in range(k):
                taken.pop()
                for c in subs:
                    cap[c] += 1
            if nodes[0] > node_budget:
                return
        rec(idx + 1, taken, capleft_now)

    rec(0, [], capleft)
    return sols["list"]


def _exact_pack_edges(cap, cand_blocks, n_slots, s, target_edges,
                      node_budget=200000, max_mult=1):
    """Bounded DFS choosing at most n_slots blocks from `cand_blocks` (each
    usable once, order respected after an internal size-desc stable sort) to
    reach total edges >= target_edges, counting unfilled slots as weight-
    (s-1) pads.  Returns the chosen blocks or None.  Does not mutate cap on
    failure; on success leaves cap decremented by the chosen blocks."""
    cands = []
    for q in cand_blocks:
        subs = list(combinations(q, s)) if len(q) >= s else []
        if all(cap[c] >= 1 for c in subs):
            cands.append((tuple(q), subs))
    cands.sort(key=lambda qs: -len(qs[0]))
    L = len(cands)
    suffix = [0] * (L + 1)  # best-possible edges from idx with unlimited slots
    # sizes are desc, so the best k picks from idx are the first k
    pref = [0]
    for q, _ in cands:
        pref.append(pref[-1] + len(q))
    nodes = [0]
    out = []

    def bound(idx, slots_left):
        take = min(slots_left, (L - idx) * max_mult)
        # sizes are non-increasing, so with multiplicity the best take still
        # starts at idx; approximate by repeating each candidate max_mult
        # times in order (admissible: overcounts capacity-feasible edges)
        full, rem = divmod(take, max_mult)
        tot = (pref[idx + full] - pref[idx]) * max_mult
        if rem and idx + full < L:
            tot += rem * len(cands[idx + full][0])
        return tot + (slots_left - take) * (s - 1)

    def rec(idx, slots_left, edges):
        if edges + slots_left * (s - 1) >= target_edges:
            return True
        if idx >= L or slots_left == 0:
            return False
        nodes[0] += 1
        if nodes[0] > node_budget:
            return False
        if edges + bound(idx, slots_left) < target_edges:
            return False
        q, subs = cands[idx]
        fit = min((cap[c] for c in subs), default=max_mult)
        kmax = min(max_mult, fit, slots_left)
        for k in range(1, kmax + 1):
            for _ in range(k):
                for c in subs:
                    cap[c] -= 1
                out.append(q)
            if rec(idx + 1, slots_left - k, edges + k * len(q)):
                return True
            for _ in range(k):
                out.pop()
                for c in subs:
                    cap[c] += 1
            if nodes[0] > node_budget:
                return False
        return rec(idx + 1, slots_left, edges)

    if rec(0, n_slots, 0):
        return list(out)
    return None


def _exact_splus1_fill(cap, m, n_slots, s, node_budget=None, t=3):
    """Exact-pack all (s+1)-blocks.  Two canonical candidate orders are
    tried — plain lex, and XOR-zero (Steiner-layer) blocks first — and the
    larger packing kept: plane-first wins on plane-friendly structures, lex
    on pole-heavy ones.  Each block may repeat up to t-1 times (candidates
    are listed with multiplicity)."""
    if node_budget is None:
        node_budget = _SPEED["exact_nodes"]
        if comb(m, s + 1) <= 80:  # tiny candidate set: afford a deep dive
            node_budget = max(node_budget, _SPEED["exact_deep"])

    def _xor(b):
        v = 0
        for e in b:
            v ^= e
        return v
    lex = list(combinations(range(m), s + 1))
    xor = sorted(lex, key=lambda q: (_xor(q) != 0, q))
    best = None
    passes = ((lex, 1, 0.4), (xor, 1, 0.4), (lex, max(1, t - 1), 0.2))
    for cands, mult, frac in passes:
        trial = dict(cap)
        got = _exact_pack(trial, cands, n_slots, s,
                          int(node_budget * frac), max_mult=mult)
        if best is None or len(got) > len(best):
            best = got
    for q in best:
        for c in combinations(q, s):
            cap[c] -= 1
    return best


def complete_blocks(m, n, s, t, seed_blocks, use_splus1=True,
                    order="lex", grow=False, trim_seeds=True):
    """Canonical completion of a seed family under the capacity frame.

    Returns a list of exactly n blocks (or None if nothing fits), guaranteed
    (t-1)-fold s-packing => K_{s,t}-free by construction.
    """
    if s <= 0 or t <= 1:
        return None
    cap = _new_cap(m, s, t)
    blocks = []
    for b in seed_blocks:
        if len(blocks) >= n:
            break
        b = tuple(sorted(set(b)))
        if len(b) < s:
            blocks.append(b)
            continue
        if _block_fits(cap, b, s):
            _add_block(cap, b, s)
            blocks.append(b)
        elif trim_seeds:
            tb = _trim_to_fit(cap, b, s)
            if tb:
                _add_block(cap, tb, s)
                blocks.append(tb)
        # else: skip the block entirely

    if use_splus1 and len(blocks) < n and comb(m, s + 1) <= 20000:
        if order == "exact" and comb(m, s + 1) <= 600:
            blocks.extend(_exact_splus1_fill(cap, m, n - len(blocks), s,
                                             t=t))
        else:
            for q in _splus1_iter(m, s, order, t):
                if len(blocks) >= n:
                    break
                if _block_fits(cap, q, s):
                    _add_block(cap, q, s)
                    blocks.append(q)

    if grow:
        blocks = _grow_blocks(cap, blocks, m, s)

    # Repeated s-blocks: s edges for 1 unit of capacity.
    if len(blocks) < n:
        for c in combinations(range(m), s):
            if len(blocks) >= n:
                break
            k = min(cap[c], n - len(blocks))
            if k > 0:
                blocks.extend([c] * k)
                cap[c] -= k

    return _pad_blocks(m, n, s, blocks)


_COMPLETION_VARIANTS = (
    # (use_splus1, order, grow)
    (False, "lex", False),
    (True, "lex", False),
    (True, "sum", False),
    (True, "xor", False),
    (True, "exact", False),
    (True, "lex", True),
    (True, "sum", True),
    (True, "xor", True),
    (True, "exact", True),
)


def _run_seed(m, n, s, t, seed_blocks, prov, variants=_COMPLETION_VARIANTS,
              prefix_sweep=False):
    """Run a seed family through the completion variants; yield candidate
    (edges, blocks, provenance) triples.  With prefix_sweep, also try leading
    prefixes of the seed (closed trade-off between big seed blocks and
    completion capacity)."""
    sweep_variants = ((True, "lex", False), (True, "xor", False),
                      (True, "lex", True))
    jobs = [(list(seed_blocks), prov, variants)]
    if prefix_sweep:
        L = len(seed_blocks)
        for p in range(0, L):
            jobs.append((list(seed_blocks[:p]), f"{prov}[:{p}]",
                         sweep_variants))
    out = []
    for sd, pv, vset in jobs:
        for (u, o, g) in vset:
            if o == "exact" and comb(m, s + 1) > 350:
                continue
            if o == "exact" and _SPEED.get("exact_seed_filter") and \
                    not pv.startswith(("bipolar", "twin", "hadamard")):
                continue
            blocks = complete_blocks(m, n, s, t, sd, use_splus1=u,
                                     order=o, grow=g)
            if blocks is None:
                continue
            e = sum(len(b) for b in blocks)
            out.append((e, blocks, f"{pv}|fill={'y' if u else 'n'}{o[0]}"
                                   f"{'g' if g else ''}"))
    return out


# --------------------------------------------------------------------------
# 3.  Seed families
# --------------------------------------------------------------------------

# ---- 3a. trivial ---------------------------------------------------------


def trivial(m, n, s, t):
    """All-ones when no K_{s,t} can even exist (m < s or n < t).  Exact:
    z = m*n there.  Degenerate guards for s<=0 / t<=0."""
    if s <= 0 or t <= 0:
        return None, "trivial: degenerate s<=0 or t<=0 (no valid matrix)"
    if m < s or n < t:
        return np.ones((m, n), dtype=int), f"trivial: m<{s} or n<{t}, z=mn"
    return None, "trivial: inapplicable"


# ---- 3b. Culik (exact elongated regime) ----------------------------------


def culik(m, n, s, t):
    """Culik's theorem: for n >= (t-1)*C(m,s),
        z(m,n;s,t) = (s-1)*n + (t-1)*C(m,s),
    attained by taking every s-subset of rows as a column t-1 times plus
    weight-(s-1) pads.  Transposed version for m >= (s-1)*C(n,t).
    Below threshold returns a best-effort truncation (all columns weight s,
    each s-subset used <= t-1 times: s*n ones)."""
    if s <= 0 or t <= 0 or m < s or n < t:
        return None, "culik: out of domain"
    thr = (t - 1) * comb(m, s)
    if n >= thr:
        blocks = []
        for c in combinations(range(m), s):
            blocks.extend([c] * (t - 1))
        blocks = _pad_blocks(m, n, s, blocks)
        A = blocks_to_matrix(m, n, blocks)
        return A, (f"culik: EXACT regime n>={thr}, z=(s-1)n+(t-1)C(m,s)"
                   f"={(s-1)*n + thr}")
    thr_T = (s - 1) * comb(n, t)
    if m >= thr_T:
        AT, prov = culik(n, m, t, s)
        if AT is not None:
            return AT.T.copy(), "culik^T: " + prov
    # truncated best effort: n weight-s columns, each s-subset <= t-1 times
    blocks = []
    for rep in range(t - 1):
        for c in combinations(range(m), s):
            blocks.append(c)
            if len(blocks) >= n:
                break
        if len(blocks) >= n:
            break
    blocks = _pad_blocks(m, n, s, blocks)
    A = blocks_to_matrix(m, n, blocks)
    return A, f"culik-truncated: below threshold {thr}, s*n={s*n} floor"


# ---- 3b2. Roman window (closed form, cited) ------------------------------

# Tan's maximum 2-fold triple-packing-by-quadruples numbers (theory agent,
# published data; T33[m] = max #quadruple blocks with every triple <= 2).
_T33 = {3: 0, 4: 2, 5: 5, 6: 9, 7: 15, 8: 28, 9: 40, 10: 60, 11: 80,
        12: 108, 13: 143, 14: 182, 15: 225, 16: 280, 17: 340, 18: 408}


def roman_window(m, n, s, t):
    """(3,3) closed-form regime (Roman 1975 / Tan):
        z(m,n;3,3) = 3n + k4,   k4 = min(T33(m), n, floor((B-n)/3)),
    B = 2 C(m,3), valid for n above a threshold (all of m <= 6 in-table, and
    m = 7 for n >= 14).  Construction: k4 quadruples of a maximum 2-fold
    triple packing (doubled SQS / Hadamard planes when available, else the
    exact bounded packer), then repeated triples in residual capacity, then
    pads.  We BUILD it for any (m,n) and let the verifier/edge-count decide;
    the exactness claim is only cited inside the window."""
    if (s, t) != (3, 3) or m < 4 or comb(m, 4) > 3000:
        return None, "roman_window: out of scope"
    B = 2 * comb(m, 3)
    k4 = min(_T33.get(m, comb(m, 4)), n, max(0, (B - n) // 3))
    cap = _new_cap(m, 3, 3)
    quads = []
    for blocks, _ in hadamard_3design_seeds(m, 3, 3):
        for b in blocks:
            if len(quads) >= k4:
                break
            if len(b) == 4 and _block_fits(cap, b, 3):
                _add_block(cap, b, 3)
                quads.append(b)
    if len(quads) < k4:
        quads += _exact_splus1_fill(cap, m, k4 - len(quads), 3)
    blocks = list(quads)
    for c in combinations(range(m), 3):
        if len(blocks) >= n:
            break
        k = min(cap[c], n - len(blocks))
        if k > 0:
            blocks.extend([c] * k)
            cap[c] -= k
    blocks = _pad_blocks(m, n, 3, blocks)
    A = blocks_to_matrix(m, n, blocks)
    return A, (f"roman_window: k4={len(quads)} of target {k4}, "
               f"z_formula={3 * n + k4}")


# ---- 3c. sum-residue layers ----------------------------------------------


def sum_layer_seed(m, s, t):
    """(s+1)-blocks in sum-residue layers: within layer c = (sum mod m) every
    s-subset lies in at most one block (the missing element is determined),
    so t-1 layers give a clean (t-1)-fold packing."""
    if comb(m, s + 1) > 20000:
        return []
    seed = []
    for c in range(min(t - 1, m)):
        for b in combinations(range(m), s + 1):
            if sum(b) % m == c:
                seed.append(b)
    return seed


# ---- 3d. F_2^k affine hyperplanes (champion, M=8..15 for (3,3)) ----------


def hyperplane_f2k_seeds(m, s, t):
    """Rows = m distinct nonzero vectors of F_2^k; column a (nonzero) is
    {x : <a,x> = 1}.  For s=3: a dependent triple (x+y+z=0) is inconsistent
    (0 covering columns); an independent triple has rank 3, so exactly
    2^(k-3) covering columns.  Freeness for t needs 2^(k-3) <= t-1.
    Several canonical point-subset selections are returned."""
    if s != 3:
        return []
    out = []
    for k in (3, 4, 5):
        if 2 ** (k - 3) > t - 1 or m > 2 ** k - 1:
            continue
        selections = {
            "lex": list(range(1, m + 1)),
            "wt": sorted(range(1, 2 ** k), key=lambda x: (bin(x).count("1"), x))[:m],
            "hi": list(range(2 ** k - 1, 2 ** k - 1 - m, -1)),
        }
        for name, pts in selections.items():
            blocks = []
            for a in range(1, 2 ** k):
                blk = tuple(i for i, x in enumerate(pts)
                            if bin(a & x).count("1") & 1)
                if len(blk) >= s:
                    blocks.append(blk)
            blocks = sorted(set(blocks), key=lambda b: (-len(b), b))
            if blocks:
                out.append((blocks, f"hyperplane_f2^{k}({name})"))
    return out


# ---- 3e. cap normals, both sides (champion, M=16 for (3,3)) --------------


def cap_bothsides_seeds(m, s, t):
    """Rows = m distinct vectors of F_2^k (0 included); columns = both sides
    {x: <a,x> = b} of hyperplanes whose normals form a cap in PG(k-1,2)
    (no 3 collinear; canonical max cap = affine complement {a: top bit set}).
    For s=3: the normals covering a distinct triple form a projective line
    (perp of a 2-dim space, k=4), and a cap meets a line in <= 2 points, so
    every triple lies in <= 2 columns."""
    if s != 3 or t < 3:
        return []
    out = []
    k = 4
    if m <= 2 ** k:
        pts = list(range(m))
        cap_normals = list(range(2 ** (k - 1), 2 ** k))
        blocks = []
        for a in cap_normals:
            for side in (1, 0):
                blk = tuple(x for x in pts
                            if (bin(a & x).count("1") & 1) == side)
                if len(blk) >= s:
                    blocks.append(blk)
        blocks = sorted(blocks, key=lambda b: (-len(b), b))
        out.append((blocks, "cap_bothsides_f2^4"))
    return out


# ---- 3f. PG(2,q) line complements (m=7 doubled Fano etc.) ----------------


def _pg2_lines(q, gf=None):
    """Lines of PG(2,q) as point-index sets; points in canonical order."""
    gf = gf or _GF(q)
    pts = _pg2_points(gf)
    idx = {p: i for i, p in enumerate(pts)}
    lines = []
    for l in pts:  # by duality lines = points
        line = tuple(sorted(idx[p] for p in pts
                            if gf.dot3(p, l) == 0))
        lines.append(line)
    return pts, lines


def line_complement_seeds(m, s, t):
    """Blocks = complements of the lines of PG(2,q) on m = q^2+q+1 points,
    each repeated r times.  For s=3 a non-collinear triple avoids exactly
    (q-1)^2 lines and a collinear one q(q-2), so r*(q-1)^2 <= t-1 suffices.
    For (3,3): q=2 (Fano), r=2 — the champion's doubled-Fano family."""
    if s != 3:
        return []
    out = []
    for q in (2, 3):
        if m != q * q + q + 1:
            continue
        cov = (q - 1) ** 2
        if cov == 0:
            continue
        r = (t - 1) // cov
        if r < 1:
            continue
        try:
            _, lines = _pg2_lines(q)
        except ValueError:
            continue
        allpts = set(range(m))
        comps = [tuple(sorted(allpts - set(l))) for l in lines]
        blocks = [b for b in comps for _ in range(r)]
        out.append((blocks, f"line_complement_pg2({q})x{r}"))
    return out


# ---- 3g. small-m curated seeds (from the evolved champion, (3,3)) --------


def small_m_seeds(m, n, s, t):
    """Champion-derived omission-code seeds for tiny m at (s,t)=(3,3).
    m=6: complements of edges of K_{3,3} (triangle-free => every 3-set of
    rows misses <= 2 edges).  m=7: nested omission code and P3+2K2."""
    if (s, t) != (3, 3):
        return []
    out = []
    if m == 5:
        k = max(0, min(5, (20 - n) // 3))
        seed = [tuple(x for x in range(5) if x != v) for v in range(k)]
        out.append((seed, f"omission_5(k={k})"))
    if m == 6:
        if n == 6:
            omitted = [(0,), (1,), (2, 3), (3, 4), (4, 5), (5, 2)]
        elif n == 7:
            omitted = [(0,)] + [(a, b) for a in (1, 2) for b in (3, 4, 5)]
        else:
            k = max(0, min(9, n, (40 - n) // 3))
            omitted = [(a, b) for a in range(3) for b in range(3, 6)][:k]
        seed = [tuple(x for x in range(6) if x not in e) for e in omitted]
        out.append((seed, f"omission_6(n={n})"))
    if m == 7:
        omitted = [(0,), (1, 2), (3, 4), (5, 6),
                   (2, 3, 5), (1, 4, 5), (1, 3, 6), (2, 4, 6)]
        out.append(([tuple(x for x in range(7) if x not in e)
                     for e in omitted], "omission_7_nested"))
        pairs = [(0, 1), (1, 2), (3, 4), (5, 6)]
        out.append(([tuple(x for x in range(7) if x not in p)
                     for p in pairs], "omission_7_p3_2k2"))
    return out


# ---- 3g2. Hadamard 3-designs ---------------------------------------------


def hadamard_3design_seeds(m, s, t):
    """For m = 4k with p = m-1 prime and p = 3 mod 4: points Z_p + {oo},
    blocks  (QR+i) u {oo}  and their complements  Z_p \\ (QR+i)  — the
    Hadamard 3-(4k, 2k, k-1) design.  Every triple lies in exactly k-1
    blocks, so it is a legal seed whenever k-1 <= t-1, and for r*(k-1) <= t-1
    the whole system may be repeated r times.

    (3,3) instances: m=8  -> 3-(8,4,1) = SQS(8), doubled to 28 blocks
                     m=12 -> 3-(12,6,2), 22 blocks, capacity-SATURATED
                     (this is the extremal structure of z(12,22) = 132)."""
    if s != 3 or m % 4 != 0:
        return []
    k = m // 4
    p = m - 1
    if k < 2 or not _is_prime(p) or p % 4 != 3 or k - 1 > t - 1:
        return []
    r = max(1, (t - 1) // max(k - 1, 1))
    qr = set(_qr_set(p))
    blocks = []
    for i in range(p):
        b1 = tuple(sorted({(x + i) % p for x in qr} | {p}))  # oo = index p
        blocks.append(b1)
        b2 = tuple(sorted(set(range(p)) - {(x + i) % p for x in qr}))
        blocks.append(b2)
    blocks = [b for b in blocks for _ in range(r)]
    return [(blocks, f"hadamard_3design(m={m},r={r})")]


def hadamard_residual_seeds(m, s, t):
    """Point-deleted Hadamard 3-designs: removing points from a legal block
    system stays legal (coverage only drops).  Gives strong seeds for m just
    below 8 or 12 at (3,3): e.g. m=10,11 from the 3-(12,6,2) design."""
    if s != 3:
        return []
    out = []
    for M0 in (8, 12, 20, 24):
        if m >= M0 or m < M0 - 3:
            continue
        for blocks, prov in hadamard_3design_seeds(M0, s, t):
            res = [tuple(x for x in b if x < m) for b in blocks]
            res = [b for b in res if len(b) >= s]
            res.sort(key=lambda b: (-len(b), b))
            out.append((res, f"{prov}\\{M0 - m}pts"))
    return out


# ---- 3g3. bipolar and twin-block seeds (distilled from found extremal
#           structures; derived, canonical, verified downstream) ------------


def _partial_triple_packing(points, max_deg):
    """Greedy lex maximal set of triples on `points`, pairwise sharing <= 1
    point, every point in <= max_deg triples.  Deterministic."""
    deg = {p: 0 for p in points}
    used_pairs = set()
    out = []
    for T in combinations(points, 3):
        if any(deg[p] >= max_deg for p in T):
            continue
        prs = list(combinations(T, 2))
        if any(pr in used_pairs for pr in prs):
            continue
        out.append(T)
        for p in T:
            deg[p] += 1
        used_pairs.update(prs)
    return out


def bipolar_seeds(m, s, t):
    """Distilled from the discovered extremal structure of z(8,17;3,3)=74:
    two 'pole' rows P, the full equator block E = [m]\\P, and blocks
    P u T_i where the T_i form a partial triple packing on E with pairwise
    intersections <= 1 and every point in <= t-1 triples.  Legality (s=3):
      - triples inside E: covered by E (1) + at most one P u T_i? no:
        T_i triple covered by E and its own block = 2 <= t-1 needs t >= 3;
      - {pole, x, y}: pair {x,y} lies in <= 1 T_i;
      - {pole, pole, x}: x lies in <= t-1 of the T_i.
    All re-checked by the capacity frame anyway."""
    if s != 3 or t < 3 or m < 6:
        return []
    eq = list(range(m - 2))
    poles = (m - 2, m - 1)
    ts = _partial_triple_packing(eq, t - 1)
    pole_blocks = [tuple(sorted(poles + T)) for T in ts]
    out = [([tuple(eq)] + pole_blocks, f"bipolar(m={m},|T|={len(ts)})")]
    # variant: equator reduced by one point (the discovered z(8,19..21)
    # shape); reduced equator triples equal to a T_i stay at coverage 2.
    for x in (eq[-1], eq[0]):
        eqr = tuple(v for v in eq if v != x)
        out.append(([eqr] + pole_blocks,
                    f"bipolar-{x}(m={m},|T|={len(ts)})"))
    return out


def twin_seeds(m, s, t):
    """Two (s+2)-blocks overlapping in max(2(s+2)-m, 0) <= s-1 points (no
    shared s-subset), the head of the discovered z(8,22;3,3)=90 structure.
    Applicable for m >= s+5; the completion supplies the Steiner-layer
    fill.

    For m = 8, s = 3 a second, sharper variant (the z(8,23)=94 structure):
    twins are plane-extensions P u {x}, P^c u {y} of a complementary AG(3,2)
    plane pair, seeded together with all OTHER planes — the twin bases
    themselves are omitted (their triples are saturated by the twins)."""
    if m < s + 5:
        return []
    b1 = tuple(range(s + 2))
    b2 = tuple(range(m - (s + 2), m))
    out = [([b1, b2], f"twin{s + 2}")]
    if s == 3 and m == 8:
        planes = [q for q in combinations(range(8), 4)
                  if (q[0] ^ q[1] ^ q[2] ^ q[3]) == 0]
        P = (0, 1, 2, 3)
        Pc = (4, 5, 6, 7)
        twin1 = tuple(sorted(P + (Pc[0],)))
        twin2 = tuple(sorted(Pc + (P[0],)))
        rest = [p for p in planes if p not in (P, Pc)]
        out.append(([twin1, twin2] + rest, "twin_planes8"))
    return out


# ---- 3g35. STS(9)/AG(2,3) dressing (decoded from the ILP witness of
#            z(9,22;3,3) = 100) -------------------------------------------


def _ag23_lines():
    """The 12 lines of AG(2,3) on points 0..8, point (a,b) -> 3a+b."""
    pts = [(a, b) for a in range(3) for b in range(3)]
    idx = {p: 3 * p[0] + p[1] for p in pts}
    lines = set()
    for p in pts:
        for d in ((0, 1), (1, 0), (1, 1), (1, 2)):
            line = tuple(sorted(idx[((p[0] + k * d[0]) % 3,
                                     (p[1] + k * d[1]) % 3)]
                                for k in range(3)))
            lines.add(line)
    return sorted(lines)


def sts9_seeds(m, s, t):
    """Decoded extremal structure of z(9,22;3,3) = 100 (ILP witness,
    coordinator 2026-07-28): take STS(9) = AG(2,3) and the point p = 0.
      - for each of the 8 lines L avoiding p:  the quad {p} u L AND its
        complement [9] \\ ({p} u L)  (a pentad) — 16 blocks;
      - the 4 lines through p give a perfect matching e_i = L \\ {p} on the
        other 8 points; split the matching into two pairs {e1,e2},{e3,e4}:
        pentads {p} u e_a u e_b across the split (4 blocks) and quads
        e_a u e_b inside the split (2 blocks).
    22 blocks, 100 edges, every triple covered <= 2 (76 saturated).  NOT a
    truncation of the Hadamard 3-(12,6,2) (that caps at 99 here).  Emitted
    as an ordered seed; the capacity completion trims for other n."""
    if (s, t) != (3, 3) or m != 9:
        return []
    lines = _ag23_lines()
    thru = [l for l in lines if 0 in l]
    avoid = [l for l in lines if 0 not in l]
    matching = [tuple(x for x in l if x != 0) for l in thru]
    quads = [tuple(sorted((0,) + l)) for l in avoid]
    pents = [tuple(sorted(set(range(9)) - set(q))) for q in quads]
    out = []
    # canonical split (first pairing) plus the two alternates — a bounded
    # derived sweep over the 3 ways to split 4 matching edges into 2+2
    for (i, j) in ((1, 2), (2, 3), (3, 1)):
        g1 = [matching[0], matching[i]]
        g2 = [matching[j], matching[6 - i - j]]
        cross = [tuple(sorted((0,) + a + b)) for a in g1 for b in g2]
        within = [tuple(sorted(g1[0] + g1[1])), tuple(sorted(g2[0] + g2[1]))]
        blocks = pents + cross + quads + within
        out.append((blocks, f"sts9_dressing(split={i}{j})"))
    return out


# ---- 3g4. pair-divisible (GDD) family ------------------------------------


def pair_gdd(m, n, s, t):
    """Distilled from the discovered extremal structure of z(11,16;3,3)=92
    (and matching the z(10,15)=81 profile): partition the rows into g pairs
    P_i (+ one singleton if m is odd); blocks are
      (b) 'blown' quotient triples P_i u P_j u P_k, where the quotient
          triples form a maximum (t-1)-fold triangle packing on K_g
          (computed exactly at the quotient level by the same packing DFS);
      (a) transversals: one element from each pair (+ the singleton),
          chosen by the exact packer under the residual capacity.
    A group-divisible-design analogue of the Hadamard family.  The number of
    blown triples is swept (closed trade-off) and the remainder completed
    canonically."""
    if s != 3 or t < 3 or m < 6 or m > 16:
        return None, "pair_gdd: out of scope"
    g = m // 2
    single = (m - 1,) if m % 2 else ()
    pairs = [(2 * i, 2 * i + 1) for i in range(g)]
    qcap = {c: t - 1 for c in combinations(range(g), 2)}
    qsols = _exact_pack_multi(qcap, list(combinations(range(g), 3)),
                              comb(g, 3), 2, node_budget=20000,
                              max_solutions=max(1, _SPEED["pair_gdd_multi"]
                                                // 4),
                              max_mult=max(1, t - 2) if t > 3 else 1)
    if not qsols:
        qsols = [[]]
    qtriples = qsols[0]
    blown_sets = [[tuple(sorted(pairs[i] + pairs[j] + pairs[k]))
                   for (i, j, k) in qs] for qs in qsols]
    blown = blown_sets[0]
    def tv(v):
        return tuple(sorted([pairs[i][(v >> i) & 1] for i in range(g)]
                            + list(single)))
    full = (1 << g) - 1
    vs = list(range(min(1 << g, 256)))
    order_lex = [tv(v) for v in vs]
    # complementary pairs are disjoint transversals (no shared s-subset):
    order_comp = []
    for v in range(1 << (g - 1)):
        order_comp.append(tv(v))
        order_comp.append(tv(full ^ v))
    # coset orders: group sign-vectors by cosets of a 2-dim linear code D
    # whose projection to every 3 coordinates is nonzero (the discovered
    # z(11,16) transversal code is a union of two such cosets).
    def coset_order(D):
        def key(v):
            return (min(v ^ d for d in D), v)
        return [tv(v) for v in sorted(vs, key=key)]

    def derived_coset_order(qtr):
        """D = span{e_a+e_b, e_c+e_d}: {a,b} = least-used quotient pair
        under the blown triples (the z(11,16) witness rule), {c,d} = the two
        largest remaining indices."""
        if g < 4:
            return None
        use = {c: 0 for c in combinations(range(g), 2)}
        for T in qtr:
            for c in combinations(T, 2):
                use[c] += 1
        a, b = min(use, key=lambda c: (use[c], c))
        rest = sorted(set(range(g)) - {a, b})
        c, d_ = rest[-2], rest[-1]
        d1 = (1 << a) | (1 << b)
        d2 = (1 << c) | (1 << d_)
        return coset_order([0, d1, d2, d1 ^ d2])

    orders = [order_lex, order_comp]
    if g >= 4:
        orders.append(coset_order([0, 0b11, 0b1100, 0b1111]))
        for qs in qsols[:3]:
            o = derived_coset_order(qs)
            if o is not None:
                orders.append(o)
    best_e, best_blocks, best_kb = -1, None, 0
    kbs = sorted({len(blown), max(0, len(blown) - 1), max(0, len(blown) - 2),
                  min(len(blown), max(0, n - 8)), 0})
    heads0 = []
    for kb in kbs:
        for bset in (blown_sets if kb == len(blown) else blown_sets[:1]):
            cap0 = _new_cap(m, 3, t)
            head0 = []
            for b in bset[:kb]:
                if _block_fits(cap0, b, 3):
                    _add_block(cap0, b, 3)
                    head0.append(b)
            heads0.append((kb, head0, cap0))
    for kb, head0, cap0 in heads0:
        for cands in orders:
            packs = _exact_pack_multi(dict(cap0), cands, n - len(head0), 3,
                                      node_budget=_SPEED["pair_gdd_pack_nodes"],
                                      max_solutions=_SPEED["pair_gdd_multi"],
                                      max_mult=max(1, t - 1))
            if not packs:
                packs = [[]]
            for pk in packs:
                head = head0 + [tuple(q) for q in pk]
                for (u, o, gr) in ((True, "exact", False),
                                   (True, "xor", True)):
                    if o == "exact" and comb(m, 4) > 350:
                        continue
                    blocks = complete_blocks(m, n, 3, t, head, use_splus1=u,
                                             order=o, grow=gr)
                    if blocks is None:
                        continue
                    e = sum(len(b) for b in blocks)
                    if e > best_e:
                        best_e, best_blocks, best_kb = e, blocks, kb
    if best_blocks is None:
        return None, "pair_gdd: nothing legal"
    # polish: exact edge-target DFS over the family's own candidate pool
    pool = blown + order_comp + [q for q in combinations(range(m), 4)]
    for gain in ((2, 1) if _SPEED["pair_gdd_polish_nodes"] else ()):
        cap = _new_cap(m, 3, t)
        got = _exact_pack_edges(cap, pool, n, 3, best_e + gain,
                                node_budget=_SPEED["pair_gdd_polish_nodes"])
        if got is not None:
            blocks = _pad_blocks(m, n, 3, got)
            e = sum(len(b) for b in blocks)
            if e > best_e:
                best_e, best_blocks = e, blocks
                best_kb = -1  # provenance: polish found it
            break
    A = blocks_to_matrix(m, n, best_blocks)
    tag = f"blown={best_kb}" if best_kb >= 0 else "polish"
    return A, f"pair_gdd(g={g},{tag})"


# ---- 3h. cyclic difference families --------------------------------------


def _diff_multiplicities(base, m):
    """Multiset multiplicity of each nonzero difference of `base` in Z_m."""
    mult = [0] * m
    for a in base:
        for b in base:
            if a != b:
                mult[(a - b) % m] += 1
    return mult


def _greedy_lambda_base(m, lam, start=0, forbid_mult=None):
    """Grow a base set B in Z_m greedily (elements start, start+1, ...) keeping
    every nonzero difference multiplicity <= lam (pooled with forbid_mult).
    A circulant whose pooled difference multiplicities are <= t-1 is
    K_{2,t}-free, hence K_{s,t}-free for all s >= 2."""
    mult = list(forbid_mult) if forbid_mult else [0] * m
    B = []
    for x in range(start, start + m):
        x %= m
        ok = True
        add = {}
        for b in B:
            for d in ((x - b) % m, (b - x) % m):
                add[d] = add.get(d, 0) + 1
        for d, k in add.items():
            if mult[d] + k > lam:
                ok = False
                break
        if ok:
            B.append(x)
            for d, k in add.items():
                mult[d] += k
    return sorted(B), mult


def _qr_set(p):
    return sorted({(x * x) % p for x in range(1, p)})


def _singer_set(q):
    """Singer planar difference set in Z_{q^2+q+1} (q prime): logs of the
    trace-zero points of GF(q^3) w.r.t. a primitive element.  Every nonzero
    difference occurs exactly once."""
    n = q * q + q + 1
    gf = _GF(q ** 3) if _is_prime(q) else None
    if gf is None:
        raise ValueError("singer: q must be prime here")
    # primitive element: brute force smallest generator of GF(q^3)^*
    size = q ** 3 - 1
    for g in range(2, q ** 3):
        seen, x, k = set(), 1, 0
        ok = True
        while k < size:
            x = gf.mul(x, g)
            k += 1
            if x == 1:
                break
        if k == size:
            D = []
            x = 1
            for i in range(size):
                if gf.trace(x) == 0:
                    D.append(i % n)
                x = gf.mul(x, g)
            D = sorted(set(D))
            if len(D) == q + 1:
                return D
    raise ValueError("singer: no primitive element found")


def _orbit_coverage_ok(bases, m, s, t):
    """Is the union of full Z_m-shift orbits of `bases` a legal (t-1)-fold
    s-packing?  Direct simulation."""
    cov = {}
    for B in bases:
        for sh in range(m):
            blk = tuple(sorted((sh + d) % m for d in B))
            if len(blk) < s:
                continue
            for c in combinations(blk, s):
                cov[c] = cov.get(c, 0) + 1
                if cov[c] > t - 1:
                    return False
    return True


def _orbit_capacity_greedy(m, s, t, n_orbits=2):
    """Grow base sets over Z_m so that the union of their full shift orbits
    stays a legal (t-1)-fold s-packing (checked by simulation, not by the
    stricter pairwise-difference condition).  Deterministic: candidate
    elements in increasing order, orbits grown one after another."""
    bases = []
    for _ in range(n_orbits):
        B = []
        for x in range(m):
            cand = sorted(set(B) | {x})
            if _orbit_coverage_ok(bases + [cand], m, s, t):
                B = cand
        if len(B) >= s and (not bases or B != bases[-1]):
            bases.append(B)
        else:
            break
    return bases


def difference_family_seeds(m, s, t):
    """Cyclic seeds over Z_m: blocks are the m shifts of each base set, base
    sets from a deterministic catalogue (verified downstream, trimmed by the
    capacity frame if slightly over):
      - greedy lambda<=t-1 sets (K_{2,t}-free circulant, proven by pooled
        difference multiplicities) + a pooled second strip,
      - quadratic residues (m prime; lambda=(m-3)/4 when m=3 mod 4),
      - QR + {0} (complement-type difference set),
      - Singer planar difference sets when m = q^2+q+1, q prime,
      - triangular prefix offsets (the run's seed program)."""
    out = []
    lam = t - 1

    B1, mult = _greedy_lambda_base(m, lam)
    if len(B1) >= s:
        blocks = [tuple(sorted((sh + d) % m for d in B1)) for sh in range(m)]
        B2, _ = _greedy_lambda_base(m, lam, start=1, forbid_mult=mult)
        if len(B2) >= s:
            blocks += [tuple(sorted((sh + d) % m for d in B2))
                       for sh in range(m)]
        out.append((blocks, f"diff_greedy_lam{lam}(|B1|={len(B1)})"))

    if m <= 30 and comb(m, s) <= 5000:
        bases = _orbit_capacity_greedy(m, s, t)
        if bases:
            blocks = []
            for B in bases:
                blocks += [tuple(sorted((sh + d) % m for d in B))
                           for sh in range(m)]
            blocks.sort(key=lambda b: (-len(b), b))
            out.append((blocks,
                        f"diff_orbitcap({'+'.join(str(len(B)) for B in bases)})"))

    if _is_prime(m) and m >= 7:
        qr = _qr_set(m)
        blocks = [tuple(sorted((sh + d) % m for d in qr)) for sh in range(m)]
        out.append((blocks, f"diff_qr({m})"))
        qr0 = sorted(set(qr) | {0})
        blocks = [tuple(sorted((sh + d) % m for d in qr0)) for sh in range(m)]
        out.append((blocks, f"diff_qr0({m})"))

    for q in (2, 3, 5):
        if m == q * q + q + 1:
            try:
                D = _singer_set(q)
                blocks = [tuple(sorted((sh + d) % m for d in D))
                          for sh in range(m)]
                out.append((blocks, f"diff_singer(q={q})"))
            except ValueError:
                pass

    # triangular prefix (seed program's family), size from the counting bound
    if s == 3 and m >= 3:
        budget = (t - 1) * comb(m, 3)
        k = 2
        while k < m and m * comb(k + 1, 3) <= budget:
            k += 1
        tri = sorted({(i * (i + 1) // 2) % m for i in range(k)})
        if len(tri) >= 3:
            blocks = [tuple(sorted((sh + d) % m for d in tri))
                      for sh in range(m)]
            out.append((blocks, f"diff_triangular(k={k})"))
    return out


def difference_family(m, n, s, t):
    """Standalone family API: best cyclic-difference construction for the
    cell, canonical completion applied, verified by the caller."""
    cands = []
    for seed, prov in difference_family_seeds(m, s, t):
        cands += _run_seed(m, n, s, t, seed, prov,
                           variants=((True, "lex", True), (True, "sum", False)))
    if not cands:
        return None, "difference_family: no applicable base set"
    cands.sort(key=lambda c: -c[0])
    e, blocks, prov = cands[0]
    return blocks_to_matrix(m, n, blocks), f"difference_family: {prov}"


# ---- 3i. finite fields, PG(2,q), norm graphs -----------------------------


def _is_prime(x):
    if x < 2:
        return False
    for p in range(2, isqrt(x) + 1):
        if x % p == 0:
            return False
    return True


_IRRED = {  # irreducible polynomials over F_p, coeffs low->high, monic
    (2, 2): (1, 1, 1),          # x^2+x+1
    (2, 3): (1, 1, 0, 1),       # x^3+x+1
    (2, 4): (1, 1, 0, 0, 1),    # x^4+x+1
    (3, 2): (1, 0, 1),          # x^2+1
    (3, 3): (1, 2, 0, 1),       # x^3+2x+1
    (2, 6): (1, 1, 0, 0, 0, 0, 1),  # x^6+x+1
    (5, 3): (2, 0, 4, 1),       # x^3+4x^2+2  (irreducible over F_5)
    (2, 9): (1, 1, 0, 0, 0, 0, 0, 0, 0, 1),  # x^9+x+1? checked at build
}


class _GF:
    """Tiny GF(p^k) with full mul table for small q; elements 0..q-1 encode
    polynomials base p."""

    def __init__(self, q):
        self.q = q
        p, k = None, None
        for pp in range(2, q + 1):
            if _is_prime(pp):
                kk, x = 0, 1
                while x < q:
                    x *= pp
                    kk += 1
                if x == q:
                    p, k = pp, kk
                    break
        if p is None:
            raise ValueError(f"GF({q}): not a prime power")
        self.p, self.k = p, k
        if k == 1:
            self.add = lambda a, b: (a + b) % p
            self.mul = lambda a, b: (a * b) % p
            self.neg = lambda a: (-a) % p
        else:
            poly = _IRRED.get((p, k))
            if poly is None:
                raise ValueError(f"GF({q}): no irreducible in catalogue")
            self._poly = poly
            self._add_t = [[self._padd(a, b) for b in range(q)]
                           for a in range(q)]
            self._mul_t = [[self._pmul(a, b) for b in range(q)]
                           for a in range(q)]
            self.add = lambda a, b: self._add_t[a][b]
            self.mul = lambda a, b: self._mul_t[a][b]
            self.neg = lambda a: self._pneg(a)

    def _digits(self, a):
        out = []
        for _ in range(self.k):
            out.append(a % self.p)
            a //= self.p
        return out

    def _undigits(self, ds):
        v = 0
        for d in reversed(ds):
            v = v * self.p + d
        return v

    def _padd(self, a, b):
        da, db = self._digits(a), self._digits(b)
        return self._undigits([(x + y) % self.p for x, y in zip(da, db)])

    def _pneg(self, a):
        return self._undigits([(-x) % self.p for x in self._digits(a)])

    def _pmul(self, a, b):
        p, k = self.p, self.k
        da, db = self._digits(a), self._digits(b)
        prod = [0] * (2 * k)
        for i, x in enumerate(da):
            if x:
                for j, y in enumerate(db):
                    prod[i + j] = (prod[i + j] + x * y) % p
        # reduce mod monic irreducible of degree k
        poly = self._poly
        for i in range(2 * k - 1, k - 1, -1):
            c = prod[i]
            if c:
                prod[i] = 0
                for j in range(k):
                    prod[i - k + j] = (prod[i - k + j] - c * poly[j]) % p
        return self._undigits(prod[:k])

    def trace(self, a):
        """Absolute trace to F_p: a + a^p + ... + a^{p^{k-1}}."""
        tot, x = 0, a
        for _ in range(self.k):
            tot = self.add(tot, x) if self.k > 1 else (tot + x) % self.p
            # x -> x^p
            y = x
            for _ in range(self.p - 1):
                y = self.mul(y, x)
            x = y
        if self.k == 1:
            return tot % self.p
        # trace lands in F_p: encoded value < p
        return tot

    def dot3(self, u, v):
        acc = 0
        for i in range(3):
            acc = self.add(acc, self.mul(u[i], v[i]))
        return acc


def _pg2_points(gf):
    """Canonical projective points of PG(2,q): first nonzero coordinate 1."""
    q = gf.q
    pts = []
    for a in range(q):
        for b in range(q):
            pts.append((1, a, b))
    for b in range(q):
        pts.append((0, 1, b))
    pts.append((0, 0, 1))
    return pts


def projective_plane_22(m, n):
    """(s,t)=(2,2): incidence matrix of PG(2,q).  At m=n=q^2+q+1 this is the
    OPTIMAL construction: z = (q+1)(q^2+q+1).  Otherwise truncate the
    smallest sufficient plane by greedy lowest-degree deletion (deterministic,
    ties by index)."""
    best_q = None
    for q in (2, 3, 4, 5, 7, 8, 9, 11, 13):
        N = q * q + q + 1
        if N >= max(m, n):
            best_q = q
            break
    if best_q is None:
        return None, "projective_plane_22: size out of catalogue"
    q = best_q
    try:
        _, lines = _pg2_lines(q)
    except ValueError:
        return None, f"projective_plane_22: GF({q}) unavailable"
    N = q * q + q + 1
    A = blocks_to_matrix(N, N, lines)
    # delete lowest-degree rows then columns, deterministically
    while A.shape[0] > m:
        deg = A.sum(axis=1)
        A = np.delete(A, int(np.argmin(deg)), axis=0)
    while A.shape[1] > n:
        deg = A.sum(axis=0)
        A = np.delete(A, int(np.argmin(deg)), axis=1)
    tag = "EXACT (q+1)(q^2+q+1)" if (m == n == N) else f"truncated from q={q}"
    return A, f"projective_plane_22: PG(2,{q}), {tag}"


def norm_graph(m, n, s, t):
    """Bipartite projective norm graph (Kollar-Ronyai-Szabo / Alon-Ronyai-
    Szabo): vertices F_{q^{s-1}} x F_q^*, edge ((A,a),(B,b)) iff
    N(A+B) = a*b with N the norm to F_q.  K_{s,(s-1)!+1}-free, so applicable
    when t >= (s-1)!+1.  Truncated to m x n by greedy deletion."""
    if s < 2 or t < factorial(s - 1) + 1:
        return None, "norm_graph: t < (s-1)!+1, freeness not guaranteed"
    # choose smallest q with q^{s-1}(q-1) >= max(m,n)
    for q in (2, 3, 4, 5, 7):
        try:
            gf = _GF(q ** (s - 1))
        except ValueError:
            continue
        side = q ** (s - 1) * (q - 1)
        if side < max(m, n):
            continue
        qq = q ** (s - 1)
        # norm: N(x) = x^{1+q+...+q^{s-2}}
        def norm(x):
            e = sum(q ** i for i in range(s - 1))
            y = 1
            for _ in range(e):
                y = gf.mul(y, x)
            return y
        # embed F_q in GF(q^{s-1}): elements 0..q-1 encode constants (base-p
        # digit encoding => constants are exactly 0..p-1 only when q=p).
        # For prime q this is exact; restrict to prime q.
        if not _is_prime(q):
            continue
        verts = [(A, a) for A in range(qq) for a in range(1, q)]
        vi = {v: i for i, v in enumerate(verts)}
        M0 = np.zeros((side, side), dtype=int)
        for (A, a) in verts:
            for B in range(qq):
                sAB = gf.add(A, B)
                if sAB == 0:
                    continue
                nv = norm(sAB)
                if nv < q and nv != 0:
                    b = (nv * pow(a, -1, q)) % q
                    if b != 0:
                        M0[vi[(A, a)], vi[(B, b)]] = 1
        A2 = M0
        while A2.shape[0] > m:
            A2 = np.delete(A2, int(np.argmin(A2.sum(axis=1))), axis=0)
        while A2.shape[1] > n:
            A2 = np.delete(A2, int(np.argmin(A2.sum(axis=0))), axis=1)
        return A2, f"norm_graph: q={q}, side={side}, truncated to {m}x{n}"
    return None, "norm_graph: no suitable q in catalogue"


# ---- 3j. waterfill-guided deterministic greedy (the floor) ---------------


def waterfill_profile(m, n, s, t):
    """Level column-degree profile: heights as equal as possible subject to
    the counting bound sum_j C(c_j, s) <= (t-1) C(m, s), capped at m.
    The exact integer maximum of sum c_j under the convex budget."""
    budget = (t - 1) * comb(m, s)
    prof = [min(s - 1, m)] * n  # weight s-1 is free
    used = 0
    # marginal cost of c -> c+1 is C(c, s-1)
    import heapq
    heap = [(comb(prof[j], s - 1), j) for j in range(n)]
    heapq.heapify(heap)
    while heap:
        cost, j = heapq.heappop(heap)
        if prof[j] >= m:
            continue
        if used + cost > budget:
            break
        used += cost
        prof[j] += 1
        heapq.heappush(heap, (comb(prof[j], s - 1), j))
    return sorted(prof, reverse=True)


def greedy_lex_derived(m, n, s, t):
    """Deterministic fallback: build columns one at a time toward the
    waterfill profile.  Each block grows row-by-row, choosing the
    lexicographically first row that keeps every newly covered s-subset
    within capacity, preferring rows of low current degree.  No randomness,
    bounded time."""
    if m < s or n < t:
        return trivial(m, n, s, t)
    prof = waterfill_profile(m, n, s, t)

    def run(mode):
        cap = _new_cap(m, s, t)
        deg = [0] * m
        blocks = []
        for j in range(n):
            target = prof[j] if j < len(prof) else s - 1
            b = []
            while len(b) < target:
                got = None
                if mode == "deg":
                    for r in sorted(range(m), key=lambda r: (deg[r], r)):
                        if r in b:
                            continue
                        if len(b) + 1 >= s:
                            subs = [tuple(sorted(c + (r,)))
                                    for c in combinations(b, s - 1)]
                            if any(cap[c] < 1 for c in subs):
                                continue
                        else:
                            subs = []
                        got = (r, subs)
                        break
                else:  # "damage": consume as little scarce capacity as we can
                    best_key = None
                    for r in range(m):
                        if r in b:
                            continue
                        if len(b) + 1 >= s:
                            subs = [tuple(sorted(c + (r,)))
                                    for c in combinations(b, s - 1)]
                            if any(cap[c] < 1 for c in subs):
                                continue
                            dmg = sum(1 for c in subs if cap[c] == 1)
                        else:
                            subs, dmg = [], 0
                        key = (dmg, deg[r], r)
                        if best_key is None or key < best_key:
                            best_key, got = key, (r, subs)
                if got is None:
                    break
                r, subs = got
                for c in subs:
                    cap[c] -= 1
                b.append(r)
                deg[r] += 1
                got = None
            blocks.append(tuple(sorted(b)))
        return _pad_blocks(m, n, s, blocks)

    best_blocks, best_e, best_mode = None, -1, ""
    for mode in ("deg", "damage"):
        blocks = run(mode)
        e = sum(len(b) for b in blocks)
        if e > best_e:
            best_blocks, best_e, best_mode = blocks, e, mode
    A = blocks_to_matrix(m, n, best_blocks)
    return A, f"greedy_lex_derived: waterfill-guided ({best_mode})"


# ---- 3j2. exhaustive cyclic base-set scan --------------------------------


def _cyclic_canonical(b, m):
    """Lex-minimal rotation representative of a subset of Z_m."""
    best = None
    bs = sorted(b)
    for sh in range(m):
        cand = tuple(sorted((x - sh) % m for x in bs))
        if best is None or cand < best:
            best = cand
    return best


def orbit_scan(m, n, s, t, sizes=None, top=6):
    """Deterministic exhaustive scan over single cyclic base sets B in Z_m
    (deduped by rotation): each orbit {B+i} is fed through the canonical
    completion, best result kept.  A 'compute the whole difference-set
    catalogue' family — bounded, no randomness.  Returns (matrix, prov)."""
    if m < s + 1 or comb(m, s) > 3000:
        return None, "orbit_scan: out of range"
    sizes = sizes or [s + 1, s + 2]
    bases = set()
    for b in sizes:
        if b > m:
            continue
        if comb(m, b) > 3000:
            continue
        for blk in combinations(range(m), b):
            bases.add(_cyclic_canonical(blk, m))
    scored = []
    for B in sorted(bases):
        orbit = [tuple(sorted((x + i) % m for x in B)) for i in range(m)]
        blocks = complete_blocks(m, n, s, t, orbit,
                                 use_splus1=True, order="lex", grow=False)
        if blocks is None:
            continue
        e = sum(len(b) for b in blocks)
        scored.append((e, B, blocks))
    if not scored:
        return None, "orbit_scan: nothing legal"
    scored.sort(key=lambda x: (-x[0], x[1]))
    # refine the leaders with the full variant set
    best_e, best_blocks, best_prov = -1, None, ""
    for e0, B, _ in scored[:top]:
        orbit = [tuple(sorted((x + i) % m for x in B)) for i in range(m)]
        for e, blocks, pv in _run_seed(m, n, s, t, orbit, f"orbit{B}"):
            if e > best_e:
                best_e, best_blocks, best_prov = e, blocks, pv
    return blocks_to_matrix(m, n, best_blocks), f"orbit_scan: {best_prov}"


# ---- 3k. bounded canonical backtracker (last resort) ---------------------


def _relaxed_bound(k, capleft, s, mmax, memo):
    """Max edges k columns can still add if capacity were fungible: level-fill
    heights against the TOTAL remaining capacity.  Admissible upper bound."""
    key = (k, capleft)
    v = memo.get(key)
    if v is not None:
        return v
    tot, cap = 0, capleft
    heights = []
    for _ in range(k):
        b = s - 1
        heights.append(b)
        tot += b
    # raise levels greedily, cheapest marginal first (all equal here)
    b = s - 1
    while True:
        cost = comb(b, s - 1)
        gain_cols = [h for h in heights if h == b]
        if not gain_cols or b >= mmax:
            break
        raised = 0
        for i in range(len(heights)):
            if heights[i] == b and cap >= cost:
                heights[i] += 1
                cap -= cost
                tot += 1
                raised += 1
        if raised == 0:
            break
        if all(h > b for h in heights):
            b += 1
    memo[key] = tot
    return tot


def bounded_backtrack(m, n, s, t, floor_edges, node_budget=250000,
                      max_gain=3):
    """Canonical, deterministic, node-capped DFS for a (t-1)-fold s-packing
    with more edges than `floor_edges`.  Columns are assigned big-first with
    non-increasing sizes; equal-size runs are lex-non-decreasing (symmetry
    breaking); pruned by the fungible-capacity bound.  Returns
    (matrix, provenance) or (None, reason).  This is the engine's LAST
    RESORT: everything algebraic runs first, and any cell closed here is
    labelled as such."""
    if m < s or comb(m, s) > 3000 or n > 40:
        return None, "bounded_backtrack: out of range"
    total_cap = (t - 1) * comb(m, s)
    blocks_by_size = {}
    for b in range(s, m + 1):
        if comb(b, s) > total_cap:
            break
        blocks_by_size[b] = [(blk, list(combinations(blk, s)))
                             for blk in combinations(range(m), b)]
    sizes_desc = sorted(blocks_by_size, reverse=True)
    bmemo = {}

    best = {"blocks": None}
    nodes = [0]

    def dfs(j, prev_size, prev_idx, cap, capleft, edges, chosen, target):
        if j == n:
            if edges >= target:
                best["blocks"] = list(chosen)
                return True
            return False
        nodes[0] += 1
        if nodes[0] > node_budget:
            return False
        # bound: remaining columns can add at most relaxed_bound edges
        rb = _relaxed_bound(n - j, capleft, s, m, bmemo)
        if edges + rb < target:
            return False
        for b in sizes_desc:
            if b > prev_size:
                continue
            # even taking all remaining at size b cannot reach target?
            if edges + (n - j) * b < target:
                break
            start = prev_idx if b == prev_size else 0
            lst = blocks_by_size[b]
            for idx in range(start, len(lst)):
                blk, subs = lst[idx]
                ok = True
                for c in subs:
                    if cap[c] < 1:
                        ok = False
                        break
                if not ok:
                    continue
                for c in subs:
                    cap[c] -= 1
                chosen.append(blk)
                if dfs(j + 1, b, idx, cap, capleft - len(subs),
                       edges + b, chosen, target):
                    return True
                chosen.pop()
                for c in subs:
                    cap[c] += 1
                if nodes[0] > node_budget:
                    return False
        # tail of weight-(s-1) pads
        if edges + (n - j) * (s - 1) >= target:
            best["blocks"] = list(chosen)
            return True
        return False

    achieved = None
    for gain in range(1, max_gain + 1):
        target = floor_edges + gain
        nodes[0] = 0
        cap = _new_cap(m, s, t)
        found = dfs(0, m, 0, cap, total_cap, 0, [], target)
        if not found:
            break
        achieved = [tuple(b) for b in best["blocks"]]
        floor_edges = target
    if achieved is None:
        return None, "bounded_backtrack: no improvement in budget"
    blocks = _pad_blocks(m, n, s, achieved)
    A = blocks_to_matrix(m, n, blocks)
    return A, f"bounded_backtrack: canonical DFS, edges={int(A.sum())}"


# --------------------------------------------------------------------------
# 4.  Compositions
# --------------------------------------------------------------------------


def pad_extend(A, m, n, s, t):
    """Extend a valid m' x n' matrix to m x n:  new columns get weight-(s-1)
    supports in the OLD rows (balanced), new rows get weight-(t-1) supports
    (balanced).  Provably preserves K_{s,t}-freeness; verified anyway.
    Gain: (s-1)*(n-n') + (t-1)*(m-m')."""
    m0, n0 = A.shape
    if m0 > m or n0 > n:
        return None
    B = np.zeros((m, n), dtype=int)
    B[:m0, :n0] = A
    coldeg = list(A.sum(axis=1))
    for j in range(n0, n):
        order = sorted(range(m0), key=lambda r: (coldeg[r], r))
        for r in order[: max(s - 1, 0)]:
            B[r, j] = 1
            coldeg[r] += 1
    rowdeg = list(B.sum(axis=0))
    for i in range(m0, m):
        order = sorted(range(n), key=lambda j: (rowdeg[j], j))
        for j in order[: max(t - 1, 0)]:
            B[i, j] = 1
            rowdeg[j] += 1
    return B


def residual_capacity(A, s, t):
    """Remaining multiplicity each s-subset of rows can still absorb."""
    A = np.asarray(A)
    m, n = A.shape
    res = _new_cap(m, s, t)
    for j in range(n):
        supp = tuple(i for i in range(m) if A[i, j])
        if len(supp) >= s:
            for c in combinations(supp, s):
                res[c] -= 1
    return res


def best_extension_column(A, s, t):
    """Append the best new column buildable under the residual capacity of A.
    Deterministic: greedy growth under three canonical row orders plus every
    existing column as a duplication candidate; keeps the heaviest legal
    block.  Always succeeds (weight s-1 pads are always legal)."""
    A = np.asarray(A)
    m, n = A.shape
    res = residual_capacity(A, s, t)
    deg = A.sum(axis=1)

    def grow(order):
        b = []
        used = []
        for r in order:
            if len(b) + 1 < s:
                b.append(r)
                continue
            subs = [tuple(sorted(c + (r,))) for c in combinations(b, s - 1)]
            if all(res[c] >= 1 for c in subs):
                for c in subs:
                    res[c] -= 1
                    used.append(c)
                b.append(r)
        for c in used:
            res[c] += 1
        return tuple(sorted(b))

    orders = [
        sorted(range(m), key=lambda r: (deg[r], r)),
        sorted(range(m), key=lambda r: (-deg[r], r)),
        list(range(m)),
    ]
    cands = [grow(o) for o in orders]
    for j in range(n):
        supp = tuple(i for i in range(m) if A[i, j])
        if len(supp) < s:
            cands.append(supp)
            continue
        if all(res[c] >= 1 for c in combinations(supp, s)):
            cands.append(supp)
    best = max(cands, key=lambda b: (len(b), [-x for x in b]))
    col = np.zeros((m, 1), dtype=int)
    for r in best:
        col[r, 0] = 1
    return np.concatenate([A, col], axis=1)


def best_extension_row(A, s, t):
    return best_extension_column(np.asarray(A).T, t, s).T


def duplicable_columns(A, s, t):
    """Columns j that may be duplicated without creating a K_{s,t}:
    legal iff no s-subset of supp(j) already shares >= t-1 columns
    (counting j itself once).  Returns [(weight, j), ...] best first."""
    A = np.asarray(A)
    m, n = A.shape
    rows = _row_ints(A)
    out = []
    for j in range(n):
        supp = [i for i in range(m) if A[i, j]]
        if len(supp) < s:
            out.append((len(supp), j))
            continue
        legal = True
        for c in combinations(supp, s):
            mask = rows[c[0]]
            for r in c[1:]:
                mask &= rows[r]
            if mask.bit_count() >= t - 1:
                legal = False
                break
        if legal:
            out.append((len(supp), j))
    out.sort(key=lambda p: (-p[0], p[1]))
    return out


def duplicate_column(A, j):
    A = np.asarray(A)
    return np.concatenate([A, A[:, j:j + 1]], axis=1)


def duplicable_rows(A, s, t):
    return duplicable_columns(np.asarray(A).T, t, s)


def duplicate_row(A, i):
    A = np.asarray(A)
    return np.concatenate([A, A[i:i + 1, :]], axis=0)


def side_by_side(A1, A2):
    """[A1 | A2]: if A1 is K_{s,t1}-free and A2 is K_{s,t2}-free (same rows),
    the result is K_{s,t1+t2-1}-free: s rows sharing t1+t2-1 columns force
    >= t1 in one block or >= t2 in the other (pigeonhole)."""
    return np.concatenate([A1, A2], axis=1)


def stack(A1, A2):
    """Transposed analogue: K_{s1,t}-free over K_{s2,t}-free (same columns)
    is K_{s1+s2-1,t}-free."""
    return np.concatenate([A1, A2], axis=0)


# --------------------------------------------------------------------------
# 5.  The engine: construct(m, n, s, t)
# --------------------------------------------------------------------------

_MEMO = {}


def _seed_catalogue(m, n, s, t):
    """All applicable structured seeds for this orientation, as
    (seed_blocks, provenance, prefix_sweep)."""
    seeds = []
    for blocks, prov in hyperplane_f2k_seeds(m, s, t):
        seeds.append((blocks, prov, True))
    for blocks, prov in cap_bothsides_seeds(m, s, t):
        seeds.append((blocks, prov, True))
    for blocks, prov in line_complement_seeds(m, s, t):
        seeds.append((blocks, prov, False))
    for blocks, prov in hadamard_3design_seeds(m, s, t):
        seeds.append((blocks, prov, True))
    for blocks, prov in hadamard_residual_seeds(m, s, t):
        seeds.append((blocks, prov, True))
    for blocks, prov in bipolar_seeds(m, s, t):
        seeds.append((blocks, prov, True))
    for blocks, prov in twin_seeds(m, s, t):
        seeds.append((blocks, prov, True))
    for blocks, prov in sts9_seeds(m, s, t):
        seeds.append((blocks, prov, True))
    for blocks, prov in small_m_seeds(m, n, s, t):
        seeds.append((blocks, prov, False))
    for blocks, prov in difference_family_seeds(m, s, t):
        seeds.append((blocks, prov, False))
    sl = sum_layer_seed(m, s, t)
    if sl:
        seeds.append((sl, "sum_layer", False))
    seeds.append(([], "empty", False))
    return seeds


def _candidates_one_orientation(m, n, s, t, fast=False):
    """(edges, matrix, provenance) candidates built with rows = the m side."""
    cands = []
    if comb(m, s) > 60000:
        return cands  # capacity frame too large; families below still run
    variants = _COMPLETION_VARIANTS if not fast else (
        (True, "lex", False), (True, "lex", True), (True, "sum", False))
    for seed, prov, sweep in _seed_catalogue(m, n, s, t):
        for e, blocks, pv in _run_seed(m, n, s, t, seed, prov,
                                       variants=variants,
                                       prefix_sweep=sweep and not fast):
            cands.append((e, blocks, pv))
    cands.sort(key=lambda c: -c[0])
    out = []
    seen = set()
    for e, blocks, pv in cands[:40 if not fast else 12]:
        A = blocks_to_matrix(m, n, blocks)
        key = A.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append((e, A, pv))
    return out


def construct(m, n, s, t, fast=False, _depth=0):
    """Run all applicable families; verify; return
    (best_matrix, provenance, per_family_edges).

    per_family_edges maps family label -> best VERIFIED edge count (0 when the
    family produced nothing valid).  The winner is always verified here."""
    key = (m, n, s, t)
    if key in _MEMO:
        entry = _MEMO[key]
        e, Ab, prov, fam = entry[:4]
        was_full = entry[4] if len(entry) > 4 else True
        if was_full or fast:
            return np.array(Ab, dtype=int).reshape(m, n), prov, dict(fam)

    per_family = {}
    best = (-1, None, "none")

    def consider(A, prov, family):
        nonlocal best
        if A is None:
            return
        A = np.asarray(A, dtype=int)
        if A.shape != (m, n):
            return
        e = int(A.sum())
        if e <= best[0] and e <= per_family.get(family, -1):
            return
        if not verify_kst_free(A, s, t):
            return
        per_family[family] = max(per_family.get(family, 0), e)
        if e > best[0]:
            best = (e, A, prov)

    # trivial / degenerate
    A, prov = trivial(m, n, s, t)
    if A is not None:
        consider(A, prov, "trivial")
        e, Ab, pv = int(A.sum()), A, prov
        _MEMO[key] = (e, tuple(A.flatten().tolist()), prov,
                      tuple(per_family.items()))
        return A, prov, dict(per_family)
    if s <= 0 or t <= 0:
        return None, prov, {}

    # Two-sided integer waterfill: a PROVEN upper bound (counting bound per
    # orientation).  Once the best construction meets it, nothing can beat
    # it — skip every remaining family.
    wf_ub = min(sum(waterfill_profile(m, n, s, t)),
                sum(waterfill_profile(n, m, t, s)), m * n)

    def done():
        return best[0] >= wf_ub

    # Culik (both orientations handled inside)
    A, prov = culik(m, n, s, t)
    consider(A, prov, "culik")

    # Roman window closed form (cited; built explicitly, verified like all)
    if (s, t) == (3, 3):
        A, prov = roman_window(m, n, s, t)
        consider(A, prov, "roman_window")

    # capacity-frame seed families, both orientations
    if done():
        pass
    for e, A, pv in (_candidates_one_orientation(m, n, s, t, fast=fast)
                     if not done() else []):
        consider(A, pv, pv.split("|")[0].split("[")[0])
        if fast and best[0] >= e:
            break
    if m != n and not done():
        for e, AT, pv in _candidates_one_orientation(n, m, t, s, fast=fast):
            consider(AT.T, pv + "^T", pv.split("|")[0].split("[")[0])

    # deterministic greedy floor
    if not done():
        A, prov = greedy_lex_derived(m, n, s, t)
        consider(A, prov, "greedy_lex")
        AT, prov = greedy_lex_derived(n, m, t, s)
        if AT is not None:
            consider(AT.T, prov + "^T", "greedy_lex")

    # exhaustive cyclic base-set scan (bounded, deduped by rotation)
    if not fast and not done() and m <= _SPEED["orbit_scan_max_m"]:
        A, prov = orbit_scan(m, n, s, t)
        consider(A, prov, "orbit_scan")
    if not fast and not done() and n <= _SPEED["orbit_scan_max_m"] \
            and m != n:
        AT, prov = orbit_scan(n, m, t, s)
        if AT is not None:
            consider(AT.T, prov + "^T", "orbit_scan")

    # pair-divisible (GDD) family, both orientations (near-square shapes)
    if not fast and not done() and 8 <= m <= 16 \
            and n <= _SPEED["pair_gdd_max_n"]:
        A, prov = pair_gdd(m, n, s, t)
        consider(A, prov, "pair_gdd")
    if not fast and not done() and m != n and 8 <= n <= 16 \
            and m <= _SPEED["pair_gdd_max_n"]:
        AT, prov = pair_gdd(n, m, t, s)
        if AT is not None:
            consider(AT.T, prov + "^T", "pair_gdd")

    # (2,2): projective planes
    if (s, t) == (2, 2) and not done():
        A, prov = projective_plane_22(m, n)
        consider(A, prov, "projective_plane")

    # norm graphs (only when the guarantee applies)
    if s >= 3 and n <= 40 and m <= 40 and not done():
        A, prov = norm_graph(m, n, s, t)
        consider(A, prov, "norm_graph")

    # compositions from smaller cells (padding + safe duplication)
    if _depth < 60:
        if m > s:
            Ap, pv, _ = construct(m - 1, n, s, t, fast=True, _depth=_depth + 1)
            if Ap is not None:
                consider(pad_extend(Ap, m, n, s, t),
                         f"pad_row[{pv}]", "pad")
                consider(best_extension_row(Ap, s, t),
                         f"ext_row[{pv}]", "extend")
        if n > t:
            Ap, pv, _ = construct(m, n - 1, s, t, fast=True, _depth=_depth + 1)
            if Ap is not None:
                consider(pad_extend(Ap, m, n, s, t),
                         f"pad_col[{pv}]", "pad")
                consider(best_extension_column(Ap, s, t),
                         f"ext_col[{pv}]", "extend")

    # polish: bounded exact edge-target DFS over a mixed 4/5-block pool
    if not fast and not done() and _SPEED["polish"] and best[0] >= 0 \
            and s == 3 and m <= 10 and comb(m, s) <= 300:
        def _xr(b):
            v = 0
            for e_ in b:
                v ^= e_
            return v
        quads = sorted(combinations(range(m), 4), key=lambda q: (_xr(q) != 0, q))
        fives = sorted(combinations(range(m), 5),
                       key=lambda q: (min(_xr(c) for c in combinations(q, 4)) != 0, q))
        pool = list(fives) + list(quads)
        for gain in (2, 1):
            cap = _new_cap(m, s, t)
            got = _exact_pack_edges(cap, pool, n, s, best[0] + gain,
                                    node_budget=_SPEED["polish_nodes"])
            if got is not None:
                blocks = _pad_blocks(m, n, s, got)
                consider(blocks_to_matrix(m, n, blocks),
                         f"mixed_exact(+{gain})", "mixed_exact")
                break

    # side-by-side for large t (t1+t2-1 = t), both halves from this engine
    if t >= 4 and _depth < 4:
        t1 = t // 2 + 1
        t2 = t + 1 - t1
        for n1 in {n // 2, max(t1, n - 2 * m), min(n - t2, 2 * m)}:
            n2 = n - n1
            if n1 < t1 or n2 < t2:
                continue
            A1, p1, _ = construct(m, n1, s, t1, fast=True, _depth=_depth + 1)
            A2, p2, _ = construct(m, n2, s, t2, fast=True, _depth=_depth + 1)
            if A1 is not None and A2 is not None:
                consider(side_by_side(A1, A2),
                         f"side_by_side[t={t1}+{t2}-1]", "side_by_side")

    e, A, prov = best
    if A is None:
        A = np.zeros((m, n), dtype=int)
        prov = "zeros (nothing applicable)"
        e = 0
    cur = _MEMO.get(key)
    if cur is None or e >= cur[0]:
        _MEMO[key] = (e, tuple(A.flatten().tolist()), prov,
                      tuple(per_family.items()), not fast)
    elif not fast:
        # weaker than the cached (fast) result: keep the better matrix but
        # mark the entry full so it is not recomputed again
        _MEMO[key] = (cur[0], cur[1], cur[2], cur[3], True)
        A = np.array(cur[1], dtype=int).reshape(m, n)
        prov = cur[2]
    return A, prov, dict(per_family)


def _memo_get(m, n, s, t):
    v = _MEMO.get((m, n, s, t))
    if v is None:
        return None
    e, flat, prov = v[0], v[1], v[2]
    return e, np.array(flat, dtype=int).reshape(m, n), prov


def _memo_put(m, n, s, t, A, prov):
    """Adopt A for the cell if it beats the memo AND verifies."""
    e = int(A.sum())
    cur = _MEMO.get((m, n, s, t))
    if cur is not None and cur[0] >= e:
        return False
    if not verify_kst_free(A, s, t):
        return False
    fams = cur[3] if cur is not None else ()
    _MEMO[(m, n, s, t)] = (e, tuple(np.asarray(A, dtype=int).flatten()
                                    .tolist()), prov, fams, True)
    return True


def build_table(max_m=16, max_n=23, s=3, t=3, halo=1, passes=2):
    """Construct every cell m in [s, max_m+halo], n in [m, max_n+halo]
    (m <= n wlog), then run extend/shrink DP sweeps over the whole grid:
      extend: grow (m,n) from (m-1,n) / (m,n-1) via the best-extension move;
      shrink: delete the lightest row/column of (m+1,n) / (m,n+1).
    Both are constructive, verified, and monotone.  Results live in _MEMO."""
    cells = [(m, n) for m in range(s, max_m + 1 + halo)
             for n in range(m, max_n + 1 + halo)]
    cells.sort(key=lambda c: (c[0] + c[1], c))
    for (m, n) in cells:
        construct(m, n, s, t)
    for _ in range(passes):
        changed = 0
        for (m, n) in cells:  # extend upward
            for (src, mode) in (((m - 1, n), "row"), ((m, n - 1), "col")):
                got = _memo_get(src[0], src[1], s, t) if src[0] >= s and \
                    src[1] >= src[0] else None
                if got is None:
                    continue
                _, As, pv = got
                B = best_extension_row(As, s, t) if mode == "row" \
                    else best_extension_column(As, s, t)
                if _memo_put(m, n, s, t, B, f"ext_{mode}[{pv[:40]}]"):
                    changed += 1
        for (m, n) in sorted(cells, key=lambda c: (-(c[0] + c[1]), c)):
            for (src, ax) in (((m + 1, n), 0), ((m, n + 1), 1)):
                got = _memo_get(src[0], src[1], s, t) \
                    if (src[0] <= max_m + halo and src[1] <= max_n + halo
                        and src[1] >= src[0]) else None
                if got is None:
                    continue
                _, As, pv = got
                deg = As.sum(axis=1 - ax)
                B = np.delete(As, int(np.argmin(deg)), axis=ax)
                if B.shape[0] > B.shape[1]:
                    continue
                if _memo_put(m, n, s, t, B, f"shrink[{pv[:40]}]"):
                    changed += 1
        if not changed:
            break
    return {(m, n): _memo_get(m, n, s, t) for (m, n) in cells}


def construct_graph(M, N):
    """Evaluator entry point: best K_{3,3}-free matrix for the cell."""
    A, _, _ = construct(M, N, 3, 3)
    return A


if __name__ == "__main__":
    # smoke test
    for (m, n, s, t) in [(3, 3, 3, 3), (7, 7, 3, 3), (8, 12, 3, 3),
                         (7, 7, 2, 2), (5, 9, 2, 3)]:
        A, prov, fams = construct(m, n, s, t)
        print(f"z({m},{n};{s},{t}) >= {int(A.sum())}  free="
              f"{verify_kst_free(A, s, t)}  via {prov}")
